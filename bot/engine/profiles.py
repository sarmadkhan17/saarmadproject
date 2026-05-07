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
        min_confidence=0.70, min_agent_agreement=3, net_score_threshold=0.40,
        ml_prob_threshold=0.80, smc_sub_checks_min=4,
        smc_liquidity_sweep_pct=0.01, smc_bos_body_pct=0.70,
        smc_fvg_required=True, smc_volume_spike_ratio=2.0, smc_pattern_completion=1.0,
        allow_market_entry=False, entry_at_fvg=True, entry_at_retracement=False,
        funding_filter_enabled=True, funding_extreme_required=True,
        flow_imbalance_ratio=2.0, macro_alignment_required=True, sentiment_standalone=False,
        htf_filter_mode="strict", btc_momentum_filter=True,
        position_size_pct=0.015, stop_loss_atr_mult=2.5, take_profit_atr_mult=3.0,
        max_correlation=0.40, max_portfolio_heat=0.30, trailing_activation_atr=1.0, size_mult=0.8,
    ),
    "BALANCED": TradingProfile(
        name="BALANCED",
        min_confidence=0.42, min_agent_agreement=1, net_score_threshold=0.15,
        ml_prob_threshold=0.65, smc_sub_checks_min=1,
        smc_liquidity_sweep_pct=0.002, smc_bos_body_pct=0.003,
        smc_fvg_required=False, smc_volume_spike_ratio=1.2, smc_pattern_completion=0.50,
        allow_market_entry=False, entry_at_fvg=False, entry_at_retracement=True,
        funding_filter_enabled=True, funding_extreme_required=False,
        flow_imbalance_ratio=1.2, macro_alignment_required=False, sentiment_standalone=False,
        htf_filter_mode="soft", btc_momentum_filter=True,
        position_size_pct=0.025, stop_loss_atr_mult=2.0, take_profit_atr_mult=2.5,
        max_correlation=0.55, max_portfolio_heat=0.50, trailing_activation_atr=1.0, size_mult=1.0,
    ),
    "AGGRESSIVE": TradingProfile(
        name="AGGRESSIVE",
        min_confidence=0.40, min_agent_agreement=1, net_score_threshold=0.15,
        ml_prob_threshold=0.50, smc_sub_checks_min=1,
        smc_liquidity_sweep_pct=0.0025, smc_bos_body_pct=0.15,
        smc_fvg_required=False, smc_volume_spike_ratio=1.3, smc_pattern_completion=0.50,
        allow_market_entry=True, entry_at_fvg=False, entry_at_retracement=True,
        funding_filter_enabled=False, funding_extreme_required=False,
        flow_imbalance_ratio=0.0, macro_alignment_required=False, sentiment_standalone=True,
        htf_filter_mode="soft", btc_momentum_filter=False,
        position_size_pct=0.04, stop_loss_atr_mult=1.5, take_profit_atr_mult=2.0,
        max_correlation=0.70, max_portfolio_heat=0.70, trailing_activation_atr=1.5, size_mult=1.2,
    ),
}
