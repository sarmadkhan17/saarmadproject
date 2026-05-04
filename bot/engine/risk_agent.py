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


class RiskDecisionAgent:
    def __init__(self, risk, gnn, hmm_regime_model=None):
        self.risk       = risk
        self.gnn        = gnn
        self.hmm_model  = hmm_regime_model

    def evaluate(self, ensemble, symbol: str, df_1h, profile,
                 regime_ctx: dict, btc_return: float, open_trades: list,
                 balance: float, get_price_fn, get_atr_fn,
                 htf_bias: str = "NEUTRAL",
                 ) -> RiskDecision:
        reasons = []
        action = ensemble.action
        conf   = ensemble.confidence

        # ── Gate 1: Confidence ───────────────────────────────────────
        if conf < profile.min_confidence:
            reasons.append(f"conf={conf:.2f} < {profile.min_confidence}")
            return RiskDecision(False, reasons, conf, profile=profile.name)

        # ── Gate 2: Agent agreement ──────────────────────────────────
        if ensemble.agents_agreeing < profile.min_agent_agreement:
            reasons.append(
                f"agents={ensemble.agents_agreeing}/{ensemble.agents_total} < {profile.min_agent_agreement}"
            )
            if profile.name != "AGGRESSIVE":
                return RiskDecision(False, reasons, conf, profile=profile.name)

        # ── Gate 3: SMC sub-checks minimum ───────────────────────────
        smc_sig = next((s for s in ensemble.signals if s.agent == "smc"), None)
        if smc_sig and smc_sig.confidence == 0 and "sub-checks" in smc_sig.reasoning:
            reasons.append(f"smc={smc_sig.reasoning}")

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
                conf = round(conf * 0.70, 4)
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

        # ── Gate 8: Position sizing ──────────────────────────────────
        price = get_price_fn(symbol)
        if price is None or price <= 0:
            reasons.append("no price")
            return RiskDecision(False, reasons, conf, profile=profile.name, htf_bias=htf_bias, hmm_regime=hmm_regime)

        amount, est_usdt = self.risk.get_position_size(
            confidence=conf, balance=balance, price=price,
            df=df_1h, recent_trades=[], regime_ctx=regime_ctx, all_agree=(ensemble.agents_agreeing >= 3),
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

        # ── ALL PASSED ─────────────────────────────────────────────
        return RiskDecision(
            approved=True, reasons=reasons if reasons else ["all checks passed"],
            adjusted_conf=conf, position_size=amount, est_usdt=est_usdt,
            htf_bias=htf_bias, hmm_regime=hmm_regime,
        )
