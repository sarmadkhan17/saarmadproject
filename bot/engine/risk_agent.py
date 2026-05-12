"""
Risk Decision Agent — profile-gated trade approval.
Sequential gates from profile: confidence, agreement, SMC quality, regime,
HTF bias, sizing, hard risk blocks.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RiskDecision:
    approved: bool
    reasons: list = field(default_factory=list)
    adjusted_conf: float = 0.0
    position_size: float = 0.0
    est_usdt: float = 0.0
    htf_bias: str = "NEUTRAL"
    hmm_regime: str = "UNKNOWN"
    profile: str = "BALANCED"
    quality_score: float = 0.0


class RiskDecisionAgent:
    def __init__(self, risk, gnn, hmm_regime_model=None):
        self.risk       = risk
        self.gnn        = gnn
        self.hmm_model  = hmm_regime_model

    # ── Quality Score ─────────────────────────────────────────────────────────
    def _compute_quality_score(self, ensemble, df_1h, regime_ctx: dict) -> float:
        """
        Composite quality score — weighted additive formula, all inputs normalized 0–1.
          quality = confidence*0.4 + trend_strength*0.3 + volume_factor*0.2 + regime_factor*0.1
        Expected ranges: weak 0.20–0.35 | medium 0.40–0.60 | strong 0.65+
        """
        ctx = regime_ctx or {}
        adx = ctx.get("adx", 25.0)
        vol_ratio = ctx.get("vol_ratio", 1.0)

        # Confidence: already 0–1
        confidence = float(ensemble.confidence)

        # Trend strength: ADX 15→40 maps linearly to 0.0→1.0
        trend_strength = min(1.0, max(0.0, (adx - 15.0) / 25.0))

        # Volume factor: vol_ratio 0.5→2.0 maps linearly to 0.0→1.0
        volume_factor = min(1.0, max(0.0, (vol_ratio - 0.5) / 1.5))

        # Regime factor — direction-aware: CRASH/STRONG_TREND scored by alignment with trade
        regime_scores = {
            "STRONG_TREND":    1.00,
            "TRENDING":        0.90,
            "WEAK_TREND":      0.72,
            "RANGING":         0.52,
            "HIGH_VOLATILITY": 0.32,
            "CHOPPY":          0.10,
            "CRASH":           0.08,
        }
        action = getattr(ensemble, "action", "BUY")
        regime = ctx.get("hmm_regime", ctx.get("regime", "RANGING"))
        if regime == "CRASH":
            # Shorts align with crash direction; longs are contra-crash
            regime_factor = 0.70 if action == "SELL" else 0.05
        elif regime == "STRONG_TREND":
            trend_dir = ctx.get("trend_direction", "NEUTRAL")
            if (action == "BUY" and trend_dir == "BEARISH") or (action == "SELL" and trend_dir == "BULLISH"):
                regime_factor = 0.30  # contra-trend quality penalty
            else:
                regime_factor = 1.00
        else:
            regime_factor = regime_scores.get(regime, 0.52)

        quality = (confidence    * 0.4
                   + trend_strength * 0.3
                   + volume_factor  * 0.2
                   + regime_factor  * 0.1)
        return round(quality, 4)

    # ── Confluence Score (CONFLUENCE profile) ─────────────────────────────────
    def _confluence_score(self, ensemble, df_1h, regime_ctx: dict, profile) -> tuple:
        """Compute weighted confluence score and regime-dynamic threshold.
        Returns (score, threshold).
        """
        net_score = getattr(ensemble, 'net_score', 0.0) or 0.0
        d_ensemble = min(abs(net_score) / 0.5, 1.0)

        total = max(ensemble.agents_total, 1)
        d_agreement = ensemble.agents_agreeing / total

        d_ml = max(0.0, min(1.0, (ensemble.confidence - 0.35) / 0.60))

        try:
            vol_series = df_1h["volume"].dropna()
            if len(vol_series) >= 21:
                vol_avg = float(vol_series.iloc[-21:-1].mean())
                vol_cur = float(vol_series.iloc[-1])
                vol_ratio = vol_cur / max(vol_avg, 1e-9)
            else:
                vol_ratio = 1.0
        except Exception:
            vol_ratio = 1.0
        d_volume = max(0.0, min((vol_ratio - 1.0) / 2.0, 1.0))

        regime_scores = {"TRENDING": 1.0, "STRONG_TREND": 1.0,
                         "WEAK_TREND": 0.65, "RANGING": 0.6,
                         "HIGH_VOLATILITY": 0.35, "CRASH": 0.1, "CHOPPY": 0.05}
        hmm = regime_ctx.get("hmm_regime", regime_ctx.get("regime", "RANGING"))
        d_regime = regime_scores.get(hmm, 0.5)

        score = (profile.w_ensemble_strength * d_ensemble
                 + profile.w_agent_agreement   * d_agreement
                 + profile.w_ml_confidence     * d_ml
                 + profile.w_volume            * d_volume
                 + profile.w_regime            * d_regime)

        thresholds = {
            "TRENDING":     profile.confluence_threshold_trending,
            "STRONG_TREND": profile.confluence_threshold_trending,
            "WEAK_TREND":   profile.confluence_threshold_ranging,
            "RANGING":      profile.confluence_threshold_ranging,
            "HIGH_VOL":     profile.confluence_threshold_high_vol,
            "HIGH_VOLATILITY": profile.confluence_threshold_high_vol,
            "CRASH":        profile.confluence_threshold_crash,
            "CHOPPY":       profile.confluence_threshold_crash,
        }
        threshold = thresholds.get(hmm, profile.confluence_threshold_ranging)

        return round(score, 4), round(threshold, 4)

    def evaluate(self, ensemble, symbol: str, df_1h, profile,
                 regime_ctx: dict, btc_return: float, open_trades: list,
                 balance: float, get_price_fn, get_atr_fn,
                 htf_bias: str = "NEUTRAL", all_trades: list = None,
                 ) -> RiskDecision:
        reasons = []
        action = ensemble.action
        conf   = ensemble.confidence

        # ── Gate 0: Stale data check ─────────────────────────────────
        try:
            import pandas as pd
            last_idx = df_1h.index[-1]
            if hasattr(last_idx, 'tzinfo') and last_idx.tzinfo is None:
                last_idx = last_idx.tz_localize("UTC")
            candle_age_h = (pd.Timestamp.now(tz="UTC") - last_idx).total_seconds() / 3600
            if candle_age_h > 4:
                reasons.append(f"stale data: last candle {candle_age_h:.1f}h ago")
                return RiskDecision(False, reasons, conf, profile=profile.name)
        except Exception:
            pass

        # ── Confluence Gate (CONFLUENCE profile only) ────────────────
        if getattr(profile, 'use_confluence_scoring', False):
            c_score, c_threshold = self._confluence_score(ensemble, df_1h, regime_ctx or {}, profile)
            hmm_for_log = (regime_ctx or {}).get("hmm_regime", (regime_ctx or {}).get("regime", "?"))
            log.debug(f"Confluence score={c_score:.3f} threshold={c_threshold:.3f} regime={hmm_for_log}")
            if conf < profile.min_confidence:
                reasons.append(f"conf floor: {conf:.2f} < {profile.min_confidence}")
                return RiskDecision(False, reasons, conf, profile=profile.name)
            if c_score < c_threshold:
                reasons.append(f"confluence={c_score:.3f} < {c_threshold:.3f} ({hmm_for_log})")
                return RiskDecision(False, reasons, conf, profile=profile.name)
            reasons.append(f"confluence={c_score:.3f} >= {c_threshold:.3f}")
            # Skip boolean gates 1-3; fall through to Gate 3.5 onwards

        else:
            # ── Gate 1: Confidence ───────────────────────────────────────
            if conf < profile.min_confidence:
                reasons.append(f"conf={conf:.2f} < {profile.min_confidence}")
                return RiskDecision(False, reasons, conf, profile=profile.name)

            # ── Gate 2: Agent agreement ──────────────────────────────────
            if ensemble.agents_agreeing < profile.min_agent_agreement:
                reasons.append(
                    f"agents={ensemble.agents_agreeing}/{ensemble.agents_total} < {profile.min_agent_agreement}"
                )
                return RiskDecision(False, reasons, conf, profile=profile.name)

            # ── Gate 3: SMC sub-checks minimum ───────────────────────────
            smc_sig = next((s for s in ensemble.signals if s.agent == "smc"), None)
            if smc_sig and smc_sig.confidence == 0 and "sub-checks" in smc_sig.reasoning:
                reasons.append(f"smc={smc_sig.reasoning}")

        # ── Gate 3.5: ADX minimum (per-profile trend quality gate) ───
        hmm_regime = (regime_ctx or {}).get("hmm_regime", "UNKNOWN")
        adx = (regime_ctx or {}).get("adx", 25.0)
        adx_min = getattr(profile, 'adx_min', 20.0)
        if adx < adx_min:
            reasons.append(f"ADX={adx:.1f} < profile_min={adx_min:.0f} ({profile.name})")
            return RiskDecision(False, reasons, conf, profile=profile.name, hmm_regime=hmm_regime)

        # ── Gate 4: Regime gate ──────────────────────────────────────
        if regime_ctx:
            hmm_regime = regime_ctx.get("hmm_regime", "UNKNOWN")
            if not regime_ctx.get("gate", True):
                reasons.append(f"regime={regime_ctx.get('regime','?')} gate closed")
                return RiskDecision(False, reasons, conf, profile=profile.name, hmm_regime=hmm_regime)
            if action == "BUY" and not regime_ctx.get("allow_longs", True):
                reasons.append(f"longs blocked in {regime_ctx.get('regime','?')}")
                return RiskDecision(False, reasons, conf, profile=profile.name, hmm_regime=hmm_regime)
            if action == "SELL" and not regime_ctx.get("allow_shorts", True):
                reasons.append(f"shorts blocked in {regime_ctx.get('regime','?')}")
                return RiskDecision(False, reasons, conf, profile=profile.name, hmm_regime=hmm_regime)

        # ── Gate 4b: Contra-trend breadth confidence penalty ─────────
        if regime_ctx:
            breadth      = regime_ctx.get("breadth",      0.5)
            bear_breadth = regime_ctx.get("bear_breadth", 0.5)
            _gate_regime = regime_ctx.get("regime", "")
            # Hard-block shorts in confirmed bull STRONG_TREND (breadth > 70%)
            if action == "SELL" and _gate_regime == "STRONG_TREND" and breadth > 0.70:
                reasons.append(
                    f"shorts blocked: STRONG_TREND breadth={breadth:.0%}"
                )
                return RiskDecision(False, reasons, conf, profile=profile.name, hmm_regime=hmm_regime)
            if action == "SELL" and breadth > 0.50:
                penalty = min(0.15, (breadth - 0.50) * 1.5)
                penalised_floor = round(min(
                    getattr(profile, 'min_confidence', 0.45) + penalty,
                    0.80,
                ), 4)
                if conf < penalised_floor:
                    reasons.append(
                        f"short penalised: bullish breadth={breadth:.0%} → need conf>={penalised_floor:.2f}"
                    )
                    return RiskDecision(False, reasons, conf, profile=profile.name, hmm_regime=hmm_regime)
            elif action == "BUY" and bear_breadth > 0.50:
                penalty = min(0.15, (bear_breadth - 0.50) * 1.5)
                penalised_floor = round(min(
                    getattr(profile, 'min_confidence', 0.45) + penalty,
                    0.80,
                ), 4)
                if conf < penalised_floor:
                    reasons.append(
                        f"long penalised: bearish breadth={bear_breadth:.0%} → need conf>={penalised_floor:.2f}"
                    )
                    return RiskDecision(False, reasons, conf, profile=profile.name, hmm_regime=hmm_regime)

        # ── Gate 4c: Momentum filter (price vs 20EMA) ───────────────────
        if symbol != "BTC/USDT":
            try:
                ema20 = df_1h["close"].ewm(span=20).mean().iloc[-1]
                price = get_price_fn(symbol)
                if price is not None and price > 0:
                    if action == "BUY" and price < ema20 * 0.99:
                        reasons.append(f"price {price:.4f} below 20EMA {ema20:.4f} (bearish)")
                        return RiskDecision(False, reasons, conf, profile=profile.name, hmm_regime=hmm_regime)
                    elif action == "SELL" and price > ema20 * 1.01:
                        reasons.append(f"price {price:.4f} above 20EMA {ema20:.4f} (bullish)")
                        return RiskDecision(False, reasons, conf, profile=profile.name, hmm_regime=hmm_regime)
            except Exception:
                pass

        # ── Gate 5: Confidence (post-regime effective) ───────────────
        eff_conf = max(
            getattr(profile, 'min_confidence', 0.50),
            regime_ctx.get("min_conf", profile.min_confidence) if regime_ctx else profile.min_confidence,
        )
        if conf < eff_conf - 0.005:
            reasons.append(f"conf={conf:.2f} < eff_min={eff_conf:.2f} ({hmm_regime})")
            return RiskDecision(False, reasons, conf, profile=profile.name, hmm_regime=hmm_regime)

        # ── Gate 6: HTF bias filter ──────────────────────────────────
        htf_mode = getattr(profile, 'htf_filter_mode', 'soft')
        htf_conflict = (action == "BUY" and htf_bias == "SELL") or (action == "SELL" and htf_bias == "BUY")
        if htf_conflict:
            if htf_mode == "soft":
                conf = round(conf * 0.80, 4)
                reasons.append(f"HTF {htf_bias} softened → conf={conf:.2f}")
            elif htf_mode == "hard":
                if conf < 0.65:
                    reasons.append(f"HTF {htf_bias} hard-block (conf={conf:.2f} < 0.65)")
                    return RiskDecision(False, reasons, conf, profile=profile.name, htf_bias=htf_bias, hmm_regime=hmm_regime)
            else:  # strict
                reasons.append(f"HTF {htf_bias} strict-block")
                return RiskDecision(False, reasons, conf, profile=profile.name, htf_bias=htf_bias, hmm_regime=hmm_regime)
            if conf < profile.min_confidence:
                reasons.append(f"post-HTF conf={conf:.2f} < {profile.min_confidence}")
                return RiskDecision(False, reasons, conf, profile=profile.name, htf_bias=htf_bias, hmm_regime=hmm_regime)

        # ── Gate 7: BTC momentum ────────────────────────────────────
        if symbol != "BTC/USDT" and getattr(profile, 'btc_momentum_filter', True):
            if action == "BUY" and btc_return < -0.015:
                conf = round(conf * 0.85, 4)
            elif action == "SELL" and btc_return > 0.015:
                conf = round(conf * 0.85, 4)
            if conf < profile.min_confidence:
                reasons.append(f"BTC momentum: conf={conf:.2f} < {profile.min_confidence}")
                return RiskDecision(False, reasons, conf, profile=profile.name, htf_bias=htf_bias, hmm_regime=hmm_regime)

        # ── Gate 8: Position sizing ──────────────────────────────────
        price = get_price_fn(symbol)
        if price is None or price <= 0:
            reasons.append("no price")
            return RiskDecision(False, reasons, conf, profile=profile.name, htf_bias=htf_bias, hmm_regime=hmm_regime)

        amount, est_usdt = self.risk.get_position_size(
            confidence=conf, balance=balance, price=price,
            df=df_1h, recent_trades=all_trades or [], regime_ctx=regime_ctx,
            all_agree=(ensemble.agents_agreeing >= 3), open_trades=open_trades,
            sl_atr_mult=getattr(profile, "stop_loss_atr_mult", 2.5),
        )
        if est_usdt < 10:
            reasons.append(f"size too small: ${est_usdt:.2f}")
            return RiskDecision(False, reasons, conf, profile=profile.name, htf_bias=htf_bias, hmm_regime=hmm_regime)

        # ── Gate 9: Portfolio risk ─────────────────────────────────
        ok, reason = self.risk.can_open_trade(
            symbol=symbol, open_trades=open_trades,
            balance=balance, new_usdt=est_usdt, get_price_fn=get_price_fn,
        )
        if not ok:
            reasons.append(reason)
            return RiskDecision(False, reasons, conf, profile=profile.name, htf_bias=htf_bias, hmm_regime=hmm_regime)

        # ── Gate 10: GNN correlation ────────────────────────────────
        open_syms = [t["symbol"] for t in open_trades]
        gnn_ok, gnn_msg, gnn_score = self.gnn.check(symbol, open_syms)
        if not gnn_ok:
            reasons.append(f"gnn: {gnn_msg}")
            return RiskDecision(False, reasons, conf, profile=profile.name, htf_bias=htf_bias, hmm_regime=hmm_regime)

        # ── Gate 11: Composite quality score (all profiles) ──────────
        # quality = conf*0.4 + trend*0.3 + volume*0.2 + regime*0.1  (additive, each 0–1)
        quality = self._compute_quality_score(ensemble, df_1h, regime_ctx)
        min_quality = getattr(profile, 'min_quality_score', 0.40)
        if quality < min_quality:
            reasons.append(f"quality={quality:.3f} < min={min_quality:.2f} ({profile.name})")
            return RiskDecision(False, reasons, conf, profile=profile.name,
                                htf_bias=htf_bias, hmm_regime=hmm_regime, quality_score=quality)

        # ── ALL PASSED ─────────────────────────────────────────────
        return RiskDecision(
            approved=True, reasons=reasons if reasons else ["all checks passed"],
            adjusted_conf=conf, position_size=amount, est_usdt=est_usdt,
            htf_bias=htf_bias, hmm_regime=hmm_regime, quality_score=quality,
        )
