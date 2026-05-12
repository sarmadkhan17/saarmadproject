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


class EnsembleEngine:
    BASE_AGENT_WEIGHTS = {"smc": 0.35, "technical": 0.40, "macro_flow": 0.25}

    def __init__(self, smc_agent, tech_agent, macro_agent=None):
        self.smc     = smc_agent
        self.tech    = tech_agent
        self.macro   = macro_agent

    def _regime_weights(self, regime: str) -> dict:
        w = dict(self.BASE_AGENT_WEIGHTS)
        r = (regime or "").upper()
        if "TREND" in r:
            w.update({"technical": 0.50, "smc": 0.28, "macro_flow": 0.22})
        elif "RANG" in r:
            w.update({"smc": 0.50, "technical": 0.30, "macro_flow": 0.20})
        elif "VOL" in r or "CRASH" in r:
            w.update({"macro_flow": 0.40, "technical": 0.35, "smc": 0.25})
        return w

    def run(self, symbol: str, df: pd.DataFrame, profile,
            market_ctx: dict = None) -> EnsembleResult:
        agents = {"smc": self.smc, "technical": self.tech}
        if self.macro is not None:
            agents["macro_flow"] = self.macro

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
            return EnsembleResult("HOLD", 0.0, 0.0, 0.0, 0.0, 0, n_submitted, [], "multiple_agent_failures")
        threshold_mult = 1.5 if n_failed == 1 else 1.0

        ctx     = market_ctx or {}
        # Prefer MarketRegimeGate regime (uses ADX + breadth) over HMM (log-return only, lags 3 bars)
        regime  = ctx.get("regime") or ctx.get("hmm_regime") or "RANGING"
        return self._aggregate(signals, profile, market_ctx=ctx, regime=regime,
                               threshold_mult=threshold_mult)

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

        # ── Hard block: volume so thin price discovery is unreliable ────────────
        ctx = market_ctx or {}
        if ctx.get("vol_ratio", 1.0) < 0.5:
            return EnsembleResult("HOLD", 0.0, 0.0, 0.0, 0.0, 0, len(signals), signals, "low_volume_block")

        # ── Confidence decay for low-quality market conditions ───────────────
        decay = 1.0
        if ctx.get("vol_ratio", 1.0) < 0.7:           # low-but-not-empty volume band
            decay *= 0.75
        if abs(buy_score - sell_score) < 0.05:         # conflicting agents
            decay *= 0.80
        if ctx.get("adx", 25.0) < 15:                  # weak trend
            decay *= 0.85
        net        *= decay
        buy_score  *= decay
        sell_score *= decay

        # ── Dynamic threshold ────────────────────────────────────────────────────
        base_threshold = getattr(profile, 'net_score_threshold', 0.25) * threshold_mult
        direction_conviction = abs(buy_score - sell_score)
        threshold = max(0.20, base_threshold * (1.0 - direction_conviction * 0.25))
        if net > threshold:
            action = "BUY"
        elif net < -threshold:
            action = "SELL"
        else:
            action = "HOLD"

        # Agent agreement
        agreement = sum(1 for s in signals if s.net_score * net > 0)

        # ── Variance-weighted confidence: penalise agent disagreement ──────────
        agent_weights_list = [weights.get(s.agent, 0.30) for s in signals]
        total_w_conf = sum(agent_weights_list) + 1e-9
        weighted_conf = sum(s.confidence * w for s, w in zip(signals, agent_weights_list)) / total_w_conf

        net_scores = [s.net_score for s in signals]
        net_var = float(np.var(net_scores)) if len(net_scores) > 1 else 0.0
        consensus_factor = 1.0 / (1.0 + net_var * 3.0)

        alignment = max(0.3, agreement / max(len(signals), 1))

        confidence = min(0.95, max(0.35, weighted_conf * consensus_factor * alignment))

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
