"""
Ensemble Engine — runs all strategy agents in parallel, produces weighted consensus.
"""

import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    action: str           # BUY / SELL / HOLD
    confidence: float
    net_score: float
    buy_score: float
    sell_score: float
    agents_agreeing: int
    agents_total: int
    signals: list = field(default_factory=list)
    source: str = "ensemble"
    agents_ok: bool = True   # False when ≥1 configured agent errored during this run


class EnsembleEngine:
    BASE_AGENT_WEIGHTS = {"smc": 0.35, "technical": 0.40,
                          "macro_flow": 0.25, "mean_reversion": 0.0}

    def __init__(self, smc_agent, tech_agent, macro_agent=None,
                 trend_filter=None, mr_agent=None):
        self.smc     = smc_agent
        self.tech    = tech_agent
        self.macro   = macro_agent
        # mr_agent: optional mean-reversion agent (Phase 3). When present the
        # ensemble routes weight to it by regime; when None the engine behaves
        # exactly as the legacy 3-agent ensemble.
        self.mr      = mr_agent
        # trend_filter: dict from config.trend_filter — vetoes counter-trend signals
        # using a per-symbol higher-TF EMA(fast)/EMA(slow) cross check.
        # When `use_two_tier=True`, route through bot/engine/trend_filter.TrendFilter.
        self.trend_filter = trend_filter or {}
        self._tf2 = None
        if self.trend_filter.get("use_two_tier"):
            from .trend_filter import TrendFilter
            self._tf2 = TrendFilter(self.trend_filter)

    def _regime_weights(self, regime: str) -> dict:
        w = dict(self.BASE_AGENT_WEIGHTS)
        r = (regime or "").upper()
        if self.mr is None:
            # Legacy 3-agent weighting — unchanged so the live bot is untouched.
            if "TREND" in r:
                w.update({"technical": 0.50, "smc": 0.28, "macro_flow": 0.22})
            elif "RANG" in r:
                w.update({"smc": 0.50, "technical": 0.30, "macro_flow": 0.20})
            elif "VOL" in r or "CRASH" in r:
                w.update({"macro_flow": 0.40, "technical": 0.35, "smc": 0.25})
            return w
        # Phase 3 — 4-agent regime routing: momentum in trends, mean-reversion
        # in ranges, defensive (macro-heavy) in volatile/crash regimes.
        if "TREND" in r:
            w.update({"technical": 0.46, "smc": 0.26,
                      "macro_flow": 0.18, "mean_reversion": 0.10})
        elif "RANG" in r:
            w.update({"mean_reversion": 0.55, "smc": 0.22,
                      "technical": 0.13, "macro_flow": 0.10})
        elif "VOL" in r or "CRASH" in r:
            w.update({"macro_flow": 0.35, "technical": 0.25,
                      "smc": 0.20, "mean_reversion": 0.20})
        else:
            w.update({"mean_reversion": 0.15})
        return w

    def run(self, symbol: str, dfs: dict, profile,
            market_ctx: dict = None) -> EnsembleResult:
        if not isinstance(dfs, dict):
            dfs = {"1h": dfs}  # back-compat: caller passed a bare DataFrame
        df = dfs.get("1h")
        agents = {"smc": self.smc, "technical": self.tech}
        if self.macro is not None:
            agents["macro_flow"] = self.macro
        if self.mr is not None:
            agents["mean_reversion"] = self.mr

        signals = []
        with ThreadPoolExecutor(max_workers=min(4, len(agents))) as pool:
            futures = {}
            for name, agent in agents.items():
                if name == "smc":
                    futures[pool.submit(agent.analyze, df, profile, market_ctx)] = name
                else:
                    futures[pool.submit(agent.analyze, df, profile)] = name
            n_submitted = len(futures)
            for future in as_completed(futures):
                name = futures[future]
                try:
                    sig = future.result()
                    signals.append(sig)
                    log.debug(f"Ensemble {symbol} | {name}: net={sig.net_score:+.3f} | {sig.reasoning[:60]}")
                except Exception as e:
                    log.warning(f"Ensemble {symbol} | {name}: error {e}")

        if not signals:
            return EnsembleResult("HOLD", 0.0, 0.0, 0.0, 0.0, 0, 0, signals, "no_agents")

        n_failed = n_submitted - len(signals)
        if n_failed > 1:
            log.warning(f"Ensemble {symbol}: {n_failed}/{n_submitted} agents failed — returning HOLD")
            return EnsembleResult("HOLD", 0.0, 0.0, 0.0, 0.0, 0, n_submitted, [], "multiple_agent_failures",
                                  agents_ok=False)
        threshold_mult = 1.5 if n_failed == 1 else 1.0

        # Mean-reversion abstains when it has no opinion. A neutral MR signal
        # (|net| ~ 0) would otherwise drag weighted confidence and agent
        # agreement down and suppress trading; an abstaining agent simply
        # leaves the decision to the others.
        signals = [s for s in signals
                   if not (s.agent == "mean_reversion" and abs(s.net_score) < 0.05)]
        if not signals:
            return EnsembleResult("HOLD", 0.0, 0.0, 0.0, 0.0, 0,
                                  n_submitted, [], "all_abstained")

        ctx     = market_ctx or {}
        # Prefer MarketRegimeGate regime (uses ADX + breadth) over HMM (log-return only, lags 3 bars)
        regime  = ctx.get("regime") or ctx.get("hmm_regime") or "RANGING"
        result = self._aggregate(signals, profile, market_ctx=ctx, regime=regime,
                                 threshold_mult=threshold_mult)
        if n_failed >= 1:
            result.agents_ok = False

        # ── Higher-TF trend veto ───────────────────────────────────────────
        # Vetoes counter-trend signals via _apply_trend_filter, which routes
        # through TrendFilter (two-tier) or _check_trend_veto (legacy) based
        # on the trend_filter.use_two_tier flag. Convert to HOLD only —
        # never flip direction.
        if self.trend_filter.get("enabled") and result.action in ("BUY", "SELL"):
            veto_reason = self._apply_trend_filter(result.action, dfs)
            if veto_reason:
                log.info(f"TREND VETO {symbol} → HOLD (was {result.action}): {veto_reason}")
                result.action = "HOLD"
                result.source = f"trend_veto:{veto_reason}"
        return result

    def _apply_trend_filter(self, action: str, dfs: dict) -> Optional[str]:
        """Dispatch to two-tier or legacy filter. Returns veto reason string or None."""
        if self._tf2 is not None:
            v = self._tf2.check(dfs)
            if action == "BUY"  and not v["long_allowed"]:  return f"2tier:{v['reasoning']}"
            if action == "SELL" and not v["short_allowed"]: return f"2tier:{v['reasoning']}"
            return None
        # Legacy path: existing _check_trend_veto on the 1h df.
        return self._check_trend_veto(dfs.get("1h"), action)

    def _check_trend_veto(self, df: pd.DataFrame, action: str) -> Optional[str]:
        """Return a reason string if `action` should be vetoed, else None."""
        cfg       = self.trend_filter
        ema_fast  = int(cfg.get("ema_fast", 50))
        ema_slow  = int(cfg.get("ema_slow", 200))
        veto_sh   = bool(cfg.get("veto_shorts", True))
        veto_lo   = bool(cfg.get("veto_longs", True))
        try:
            if df is None or len(df) < ema_slow + 5:
                return None  # not enough history → no veto
            close = df["close"]
            ef = float(close.ewm(span=ema_fast, adjust=False).mean().iloc[-1])
            es = float(close.ewm(span=ema_slow, adjust=False).mean().iloc[-1])
        except Exception as e:
            log.debug(f"trend_filter compute failed: {e}")
            return None
        tf = cfg.get("tf", "1h")
        if action == "SELL" and veto_sh and ef > es:
            return f"uptrend EMA{ema_fast}>{ema_slow} on {tf} (SHORT blocked)"
        if action == "BUY" and veto_lo and ef < es:
            return f"downtrend EMA{ema_fast}<{ema_slow} on {tf} (LONG blocked)"
        return None

    def _aggregate(self, signals: list, profile,
                   market_ctx: dict = None, regime: str = "RANGING",
                   threshold_mult: float = 1.0) -> EnsembleResult:
        weights = self._regime_weights(regime)
        net = 0.0
        buy_score  = 0.0
        sell_score = 0.0
        total_w = 0.0

        for s in signals:
            w = weights.get(s.agent, 0.30)
            net += s.net_score * w
            buy_score  += s.buy_score * w
            sell_score += s.sell_score * w
            total_w += w

        if total_w > 0:
            net        = net / total_w
            buy_score  = buy_score / total_w
            sell_score = sell_score / total_w

        log.debug(f"ENSEMBLE DEBUG: regime={regime} weights={weights} net={net:.4f} buy={buy_score:.4f} sell={sell_score:.4f}")
        log.debug(f"ENSEMBLE DEBUG: market_ctx={market_ctx}")

        # ─── TREND BIAS (single application, 30% reduction only) ───
        # Previous code applied *0.05 then *0.7 = *0.035 — near-permanent HOLD.
        # Now single 30% reduction on counter-trend signals only.
        trend_dir = (market_ctx or {}).get("trend_direction", "NEUTRAL") if market_ctx else "NEUTRAL"
        log.debug(f"ENSEMBLE DEBUG: trend_dir={trend_dir} net_before_trend={net:.4f}")
        if trend_dir == "BULLISH" and net < 0:
            net = net * 0.7
            buy_score = buy_score * 0.7
            sell_score = sell_score * 0.7
            log.debug(f"Ensemble: bullish trend → SELL reduced to {net:.3f}")
        elif trend_dir == "BEARISH" and net > 0:
            net = net * 0.7
            buy_score = buy_score * 0.7
            sell_score = sell_score * 0.7
            log.debug(f"Ensemble: bearish trend → BUY reduced to {net:.3f}")

        # ── Hard block: volume so thin price discovery is unreliable ────────────
        ctx = market_ctx or {}
        vol_ratio = ctx.get("vol_ratio", 1.0)
        log.debug(f"ENSEMBLE DEBUG: vol_ratio={vol_ratio} adx={ctx.get('adx', 25.0)}")
        if vol_ratio < 0.25:
            log.debug(f"ENSEMBLE DEBUG: LOW VOLUME BLOCK triggered (vol_ratio={vol_ratio})")
            return EnsembleResult("HOLD", 0.0, 0.0, 0.0, 0.0, 0, len(signals), signals, "low_volume_block")

        # ── Confidence decay for low-quality market conditions ───────────────
        decay = 1.0
        if vol_ratio < 0.7:           # low-but-not-empty volume band
            decay *= 0.75
        if abs(buy_score - sell_score) < 0.05:         # conflicting agents
            decay *= 0.80
        if ctx.get("adx", 25.0) < 15:                  # weak trend
            decay *= 0.85
        net        *= decay
        buy_score  *= decay
        sell_score *= decay
        log.debug(f"ENSEMBLE DEBUG: decay={decay} net_after_decay={net:.4f}")

        # ── Dynamic threshold ────────────────────────────────────────────────────
        base_threshold = getattr(profile, 'net_score_threshold', 0.25) * threshold_mult
        direction_conviction = abs(buy_score - sell_score)
        threshold = max(0.03, base_threshold * (1.0 - direction_conviction * 0.25))
        if net > threshold:
            action = "BUY"
        elif net < -threshold:
            action = "SELL"
        else:
            action = "HOLD"

        # Agent agreement
        agreement = sum(1 for s in signals if s.net_score * net > 0)
        log.debug(f"ENSEMBLE DEBUG: agreement={agreement}/{len(signals)} threshold={threshold:.4f} net={net:.4f}")

        # ── Variance-weighted confidence: penalise agent disagreement ──────────
        agent_weights_list = [weights.get(s.agent, 0.30) for s in signals]
        total_w_conf = sum(agent_weights_list) + 1e-9
        weighted_conf = sum(s.confidence * w for s, w in zip(signals, agent_weights_list)) / total_w_conf
        log.debug(f"ENSEMBLE DEBUG: agent_confidences={[s.confidence for s in signals]} weighted_conf={weighted_conf:.4f}")

        net_scores = [s.net_score for s in signals]
        net_var = float(np.var(net_scores)) if len(net_scores) > 1 else 0.0
        consensus_factor = 1.0 / (1.0 + net_var * 3.0)

        alignment = max(0.3, agreement / max(len(signals), 1))

        confidence = min(0.95, max(0.35, weighted_conf * consensus_factor * alignment))
        log.debug(f"ENSEMBLE DEBUG: net_var={net_var:.4f} consensus={consensus_factor:.4f} alignment={alignment:.4f} confidence={confidence:.4f}")

        return EnsembleResult(
            action=action,
            confidence=round(confidence, 4),
            net_score=round(net, 4),
            buy_score=round(buy_score, 4),
            sell_score=round(sell_score, 4),
            agents_agreeing=agreement,
            agents_total=len(signals),
            signals=signals,
        )
