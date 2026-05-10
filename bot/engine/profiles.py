"""Trading Profile — controls signal sensitivity, entry timing, and risk across ALL agents."""
import os
import yaml
import logging
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class TradingProfile:
    name: str

    # ── Signal Requirements ──
    min_confidence: float = 0.50
    min_agent_agreement: int = 2
    net_score_threshold: float = 0.25
    ml_prob_threshold: float = 0.65
    smc_sub_checks_min: int = 2

    # ── Trend Quality Gate ──
    adx_min: float = 20.0            # minimum ADX at entry
    min_quality_score: float = 0.47  # minimum composite quality score (additive formula)

    # ── SMC Rules ──
    smc_liquidity_sweep_pct: float = 0.005
    smc_bos_body_pct: float = 0.60
    smc_fvg_required: bool = True
    smc_volume_spike_ratio: float = 1.5
    smc_pattern_completion: float = 0.80

    # ── Entry Rules ──
    allow_market_entry: bool = False
    entry_at_fvg: bool = True
    entry_at_retracement: bool = True

    # ── Macro / Flow ──
    funding_filter_enabled: bool = True
    funding_extreme_required: bool = False
    flow_imbalance_ratio: float = 1.2
    macro_alignment_required: bool = False
    sentiment_standalone: bool = False

    # ── HTF Context ──
    htf_filter_mode: str = "soft"
    btc_momentum_filter: bool = True

    # ── Risk Parameters ──
    position_size_pct: float = 0.025
    stop_loss_atr_mult: float = 2.0
    take_profit_atr_mult: float = 2.5
    max_correlation: float = 0.55
    max_portfolio_heat: float = 0.50
    trailing_activation_atr: float = 1.0
    size_mult: float = 1.0

    # ── Confluence Scoring (CONFLUENCE profile only) ──
    use_confluence_scoring: bool = False
    confluence_threshold_trending: float = 0.55
    confluence_threshold_ranging: float = 0.70
    confluence_threshold_high_vol: float = 0.75
    confluence_threshold_crash: float = 0.90
    w_ensemble_strength: float = 0.30
    w_agent_agreement: float = 0.20
    w_ml_confidence: float = 0.25
    w_volume: float = 0.15
    w_regime: float = 0.10

    @classmethod
    def load(cls, name: str) -> "TradingProfile":
        """Return the canonical preset (read-only reference). Use from_config() for a mutable copy."""
        name_upper = name.upper()
        if name_upper in _PRESETS:
            return _PRESETS[name_upper]
        log.warning(f"Unknown profile '{name}' — falling back to BALANCED")
        return _PRESETS["BALANCED"]

    @classmethod
    def from_config(cls, config: dict) -> "TradingProfile":
        name = config.get("strategy", {}).get("trading_profile", "BALANCED")
        profile = dc_replace(cls.load(name))   # shallow copy — never mutates _PRESETS
        overrides = config.get("training", {}).get("profile_overrides", {})
        for key, value in overrides.items():
            if hasattr(profile, key):
                setattr(profile, key, value)
        return profile


_PRESETS = {
    # ── STRICT: 1–2 day swing trades, elite quality only ──────────────────────
    # High ADX requirement + 3-agent consensus + large RR target.
    # Designed for 3–5 trades/week at high win-rate, not frequency.
    "STRICT": TradingProfile(
        name="STRICT",
        # Requires all 3 agents in agreement at high confidence
        min_confidence=0.70, min_agent_agreement=3, net_score_threshold=0.48,
        ml_prob_threshold=0.78, smc_sub_checks_min=4,
        # Hard trend requirement — no chop trades
        adx_min=28.0, min_quality_score=0.60,
        # Near-complete SMC structure
        smc_liquidity_sweep_pct=0.008, smc_bos_body_pct=0.65,
        smc_fvg_required=True, smc_volume_spike_ratio=2.0, smc_pattern_completion=0.85,
        # FVG entries only — no chasing
        allow_market_entry=False, entry_at_fvg=True, entry_at_retracement=False,
        # All filters on for swing context
        funding_filter_enabled=True, funding_extreme_required=True,
        flow_imbalance_ratio=2.0, macro_alignment_required=True, sentiment_standalone=False,
        # Strict HTF alignment
        htf_filter_mode="strict", btc_momentum_filter=True,
        # Wide TP for multi-day swing (1:3+ RR), tight max heat
        position_size_pct=0.022, stop_loss_atr_mult=1.8, take_profit_atr_mult=5.0,
        max_correlation=0.35, max_portfolio_heat=0.20, trailing_activation_atr=2.0, size_mult=0.85,
    ),

    # ── BALANCED: 10–12h intraday, steady compounding ─────────────────────────
    # Medium conviction with 2-agent agreement, moderate ADX floor.
    # Good for consistent daily compounding without over-trading.
    "BALANCED": TradingProfile(
        name="BALANCED",
        # 2-agent agreement, solid confidence
        min_confidence=0.58, min_agent_agreement=2, net_score_threshold=0.32,
        ml_prob_threshold=0.68, smc_sub_checks_min=2,
        # Moderate trend quality gate
        adx_min=22.0, min_quality_score=0.47,
        # SMC with volume confirmation
        smc_liquidity_sweep_pct=0.003, smc_bos_body_pct=0.45,
        smc_fvg_required=False, smc_volume_spike_ratio=1.6, smc_pattern_completion=0.68,
        # Retracement entries
        allow_market_entry=False, entry_at_fvg=False, entry_at_retracement=True,
        # Standard filters
        funding_filter_enabled=True, funding_extreme_required=False,
        flow_imbalance_ratio=1.4, macro_alignment_required=False, sentiment_standalone=False,
        # Soft HTF bias
        htf_filter_mode="soft", btc_momentum_filter=True,
        # 1:2 RR for intraday holds
        position_size_pct=0.022, stop_loss_atr_mult=1.6, take_profit_atr_mult=3.2,
        max_correlation=0.50, max_portfolio_heat=0.40, trailing_activation_atr=1.5, size_mult=1.0,
    ),

    # ── AGGRESSIVE: momentum scalp, breakout/volume-driven ────────────────────
    # Volume spike + ADX confirmation required. Avoids sideways completely.
    # Still quality-filtered — NOT a "take everything" mode.
    "AGGRESSIVE": TradingProfile(
        name="AGGRESSIVE",
        # Requires 2-agent consensus even for scalps (1 was too permissive)
        min_confidence=0.54, min_agent_agreement=2, net_score_threshold=0.28,
        ml_prob_threshold=0.62, smc_sub_checks_min=1,
        # Breakout focus: ADX floor + volume spike primary signal
        adx_min=20.0, min_quality_score=0.30,
        # Volume spike is the primary confirmation
        smc_liquidity_sweep_pct=0.003, smc_bos_body_pct=0.20,
        smc_fvg_required=False, smc_volume_spike_ratio=2.0, smc_pattern_completion=0.45,
        # Allow market entry for fast breakout fills
        allow_market_entry=True, entry_at_fvg=False, entry_at_retracement=True,
        # Minimal macro filtering — too slow for scalping
        funding_filter_enabled=False, funding_extreme_required=False,
        flow_imbalance_ratio=0.0, macro_alignment_required=False, sentiment_standalone=True,
        # Soft HTF (not "none" — blind to daily trend causes outsized losses)
        htf_filter_mode="soft", btc_momentum_filter=True,
        # Tight SL/TP for scalp (1:1.6 RR), conservative sizing
        position_size_pct=0.015, stop_loss_atr_mult=0.9, take_profit_atr_mult=1.6,
        max_correlation=0.55, max_portfolio_heat=0.50, trailing_activation_atr=0.6, size_mult=1.0,
    ),

    # ── CONFLUENCE: quality overlay — highest bar, all dimensions required ─────
    # Not an "aggression" mode. Uses weighted confluence scoring across all
    # dimensions: ensemble strength, agent agreement, ML confidence, volume, regime.
    # Only fires when everything aligns. Best for high-conviction swing setups.
    "CONFLUENCE": TradingProfile(
        name="CONFLUENCE",
        # Confluence scoring replaces boolean gates
        use_confluence_scoring=True,
        confluence_threshold_trending=0.55,
        confluence_threshold_ranging=0.72,
        confluence_threshold_high_vol=0.78,
        confluence_threshold_crash=0.92,
        w_ensemble_strength=0.28,
        w_agent_agreement=0.22,
        w_ml_confidence=0.25,
        w_volume=0.15,
        w_regime=0.10,
        # Hard floor even with confluence scoring
        min_confidence=0.52, min_agent_agreement=2, net_score_threshold=0.28,
        ml_prob_threshold=0.65, smc_sub_checks_min=2,
        # Trend quality overlay
        adx_min=22.0, min_quality_score=0.55,
        # SMC structure required
        smc_liquidity_sweep_pct=0.004, smc_bos_body_pct=0.35,
        smc_fvg_required=False, smc_volume_spike_ratio=1.5, smc_pattern_completion=0.60,
        # Retracement entries
        allow_market_entry=False, entry_at_fvg=False, entry_at_retracement=True,
        # Full macro stack
        funding_filter_enabled=True, funding_extreme_required=False,
        flow_imbalance_ratio=1.4, macro_alignment_required=False, sentiment_standalone=False,
        # HTF required for confluence validity
        htf_filter_mode="hard", btc_momentum_filter=True,
        # 1:2.5 RR, conservative heat
        position_size_pct=0.020, stop_loss_atr_mult=1.8, take_profit_atr_mult=3.8,
        max_correlation=0.45, max_portfolio_heat=0.38, trailing_activation_atr=1.5, size_mult=1.0,
    ),
}
