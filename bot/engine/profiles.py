"""Trading Profile — controls signal sensitivity, entry timing, and risk across ALL agents."""
import os
import yaml
import logging
from dataclasses import dataclass, field
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
        name_upper = name.upper()
        if name_upper in _PRESETS:
            return _PRESETS[name_upper]
        log.warning(f"Unknown profile '{name}' — falling back to BALANCED")
        return _PRESETS["BALANCED"]

    @classmethod
    def from_config(cls, config: dict) -> "TradingProfile":
        name = config.get("strategy", {}).get("trading_profile", "BALANCED")
        profile = cls.load(name)
        overrides = config.get("training", {}).get("profile_overrides", {})
        if overrides:
            for key, value in overrides.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
        return profile


_PRESETS = {
    "STRICT": TradingProfile(
        name="STRICT",
        # Signal requirements — all agents, high conviction
        min_confidence=0.72, min_agent_agreement=3, net_score_threshold=0.45,
        ml_prob_threshold=0.78, smc_sub_checks_min=4,
        # SMC — near-complete setup required
        smc_liquidity_sweep_pct=0.01, smc_bos_body_pct=0.70,
        smc_fvg_required=True, smc_volume_spike_ratio=2.0, smc_pattern_completion=0.85,
        # Entry — FVG only, no market orders
        allow_market_entry=False, entry_at_fvg=True, entry_at_retracement=False,
        # Macro / funding — all filters on
        funding_filter_enabled=True, funding_extreme_required=True,
        flow_imbalance_ratio=2.0, macro_alignment_required=True, sentiment_standalone=False,
        # HTF / BTC
        htf_filter_mode="strict", btc_momentum_filter=True,
        # Risk — tight SL, wide TP for full-day hold (R/R 1:3)
        position_size_pct=0.025, stop_loss_atr_mult=1.5, take_profit_atr_mult=4.5,
        max_correlation=0.35, max_portfolio_heat=0.25, trailing_activation_atr=2.0, size_mult=0.8,
    ),
    "BALANCED": TradingProfile(
        name="BALANCED",
        # Signal requirements — 2 agents, moderate conviction
        min_confidence=0.55, min_agent_agreement=2, net_score_threshold=0.28,
        ml_prob_threshold=0.68, smc_sub_checks_min=2,
        # SMC — 2 checks, volume confirmation
        smc_liquidity_sweep_pct=0.002, smc_bos_body_pct=0.40,
        smc_fvg_required=False, smc_volume_spike_ratio=1.6, smc_pattern_completion=0.65,
        # Entry — retracement, no market orders
        allow_market_entry=False, entry_at_fvg=False, entry_at_retracement=True,
        # Macro / funding
        funding_filter_enabled=True, funding_extreme_required=False,
        flow_imbalance_ratio=1.2, macro_alignment_required=False, sentiment_standalone=False,
        # HTF / BTC
        htf_filter_mode="soft", btc_momentum_filter=True,
        # Risk — 1:2 R/R for 1-3 hour holds
        position_size_pct=0.022, stop_loss_atr_mult=1.5, take_profit_atr_mult=3.0,
        max_correlation=0.55, max_portfolio_heat=0.45, trailing_activation_atr=1.5, size_mult=1.0,
    ),
    "AGGRESSIVE": TradingProfile(
        name="AGGRESSIVE",
        # Signal requirements — speed priority, volume is main gate
        min_confidence=0.50, min_agent_agreement=1, net_score_threshold=0.22,
        ml_prob_threshold=0.60, smc_sub_checks_min=1,
        # SMC — volume spike is primary confirmation
        smc_liquidity_sweep_pct=0.0025, smc_bos_body_pct=0.15,
        smc_fvg_required=False, smc_volume_spike_ratio=1.8, smc_pattern_completion=0.40,
        # Entry — market orders allowed for fast fills
        allow_market_entry=True, entry_at_fvg=False, entry_at_retracement=True,
        # Macro / funding — all off (too slow for scalping)
        funding_filter_enabled=False, funding_extreme_required=False,
        flow_imbalance_ratio=0.0, macro_alignment_required=False, sentiment_standalone=True,
        # HTF / BTC — off for scalping speed
        htf_filter_mode="none", btc_momentum_filter=False,
        # Risk — very tight SL/TP for scalping (R/R 1:1.5)
        position_size_pct=0.015, stop_loss_atr_mult=0.8, take_profit_atr_mult=1.2,
        max_correlation=0.65, max_portfolio_heat=0.60, trailing_activation_atr=0.5, size_mult=1.2,
    ),
    "CONFLUENCE": TradingProfile(
        name="CONFLUENCE",
        # Confluence scoring — replaces boolean gates
        use_confluence_scoring=True,
        confluence_threshold_trending=0.55,
        confluence_threshold_ranging=0.70,
        confluence_threshold_high_vol=0.75,
        confluence_threshold_crash=0.90,
        w_ensemble_strength=0.30,
        w_agent_agreement=0.20,
        w_ml_confidence=0.25,
        w_volume=0.15,
        w_regime=0.10,
        # Floor confidence regardless of score
        min_confidence=0.45, min_agent_agreement=1, net_score_threshold=0.15,
        ml_prob_threshold=0.60, smc_sub_checks_min=1,
        # SMC defaults (not primary gate — score handles it)
        smc_liquidity_sweep_pct=0.003, smc_bos_body_pct=0.30,
        smc_fvg_required=False, smc_volume_spike_ratio=1.4, smc_pattern_completion=0.55,
        # Entry
        allow_market_entry=False, entry_at_fvg=False, entry_at_retracement=True,
        # Macro / funding
        funding_filter_enabled=True, funding_extreme_required=False,
        flow_imbalance_ratio=1.2, macro_alignment_required=False, sentiment_standalone=False,
        # HTF / BTC
        htf_filter_mode="soft", btc_momentum_filter=True,
        # Risk — 1:2 R/R, adaptive position sizing
        position_size_pct=0.020, stop_loss_atr_mult=1.8, take_profit_atr_mult=3.6,
        max_correlation=0.50, max_portfolio_heat=0.40, trailing_activation_atr=1.5, size_mult=1.0,
    ),
}
