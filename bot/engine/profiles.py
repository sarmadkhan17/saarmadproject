"""Trading Profile — controls signal sensitivity, entry timing, and risk across ALL agents.

v5 note: this dataclass holds ONLY the fields the v5 engine actually reads. The
old v3 profiles carried ~16 extra knobs (adx_min, min_quality_score, entry-style
flags, macro/funding filters, position_size_pct, max_correlation,
max_portfolio_heat, size_mult, ml_prob_threshold, …) that no v5 code path
consumed — they were removed so a profile honestly reflects what it enforces.

Where each field is consumed:
  · min_confidence, min_agent_agreement, net_score_threshold → ensemble + risk_agent gates
  · smc_*                                                     → SMCAgent.analyze
  · htf_filter_mode, btc_momentum_filter                      → risk_agent HTF / BTC-momentum gates
  · stop_loss_atr_mult                                        → KellyCriterionSizer (volatility scalar) + ExecutionEngine SL
  · take_profit_atr_mult, tp1_*, trail_atr_mult,
    early_exit_enabled, dynamic_tp_enabled                    → ExitEngine
  · use_confluence_scoring, confluence_*, w_*                 → risk_agent CONFLUENCE gate

Position size %, portfolio-heat cap, and correlation limits are NOT profile-driven
in v5: sizing uses KellyCriterionSizer.BASE_PCT/MAX_PCT, heat uses
config.risk.max_portfolio_heat, and correlation uses CorrelationFilter's fixed groups.
"""
import logging
from dataclasses import dataclass, replace as dc_replace

log = logging.getLogger(__name__)


@dataclass
class TradingProfile:
    name: str

    # ── Signal Requirements (ensemble + risk_agent) ──
    min_confidence: float = 0.50
    min_agent_agreement: int = 2
    net_score_threshold: float = 0.25
    smc_sub_checks_min: int = 2

    # ── SMC Detection Thresholds (SMCAgent) ──
    smc_liquidity_sweep_pct: float = 0.005
    smc_bos_body_pct: float = 0.60
    smc_volume_spike_ratio: float = 1.5
    smc_pattern_completion: float = 0.80

    # ── HTF / BTC Context (risk_agent) ──
    htf_filter_mode: str = "soft"      # soft | hard | strict
    btc_momentum_filter: bool = True

    # ── Risk / Exit (sizer + ExecutionEngine + ExitEngine) ──
    stop_loss_atr_mult: float = 2.5
    take_profit_atr_mult: float = 4.5
    tp1_fraction: float = 0.40         # fraction closed at TP1
    tp1_r_mult: float = 2.5            # TP1 fires at N × entry_atr gain
    trail_atr_mult: float = 2.8        # trailing stop multiplier (post-TP1)
    early_exit_enabled: bool = True    # exit failing trades on invalidation signals
    dynamic_tp_enabled: bool = True    # skip fixed TP backstop when trend accelerates

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
    # 3-agent consensus at high confidence, strict net-score, near-complete SMC
    # structure, hard HTF block, wide swing TP. ~3–5 trades/week at high win-rate.
    "STRICT": TradingProfile(
        name="STRICT",
        min_confidence=0.70, min_agent_agreement=3, net_score_threshold=0.48,
        smc_sub_checks_min=4,
        smc_liquidity_sweep_pct=0.008, smc_bos_body_pct=0.65,
        smc_volume_spike_ratio=2.0, smc_pattern_completion=0.85,
        # Hard HTF: counter-HTF signals need conf >= 0.65 to pass
        htf_filter_mode="hard", btc_momentum_filter=True,
        # Wide TP for multi-day swing (SL 3×, TP 7×, trail 3.5×) — survives crypto noise.
        # early_exit off: swing holds need room to breathe.
        stop_loss_atr_mult=3.0, take_profit_atr_mult=7.0,
        tp1_fraction=0.40, tp1_r_mult=3.0, trail_atr_mult=3.5,
        early_exit_enabled=False, dynamic_tp_enabled=True,
    ),

    # ── BALANCED: 10–12h intraday, steady compounding ─────────────────────────
    # 2-agent agreement, solid confidence, SMC with volume confirmation, soft HTF bias.
    "BALANCED": TradingProfile(
        name="BALANCED",
        min_confidence=0.58, min_agent_agreement=2, net_score_threshold=0.32,
        smc_sub_checks_min=2,
        smc_liquidity_sweep_pct=0.003, smc_bos_body_pct=0.45,
        smc_volume_spike_ratio=1.6, smc_pattern_completion=0.68,
        htf_filter_mode="soft", btc_momentum_filter=True,
        # Intraday: SL 2.5×, TP 4.5×, trail 2.8×. Early invalidation enabled.
        stop_loss_atr_mult=2.5, take_profit_atr_mult=4.5,
        tp1_fraction=0.40, tp1_r_mult=2.5, trail_atr_mult=2.8,
        early_exit_enabled=True, dynamic_tp_enabled=True,
    ),

    # ── AGGRESSIVE: momentum scalp, breakout/volume-driven ────────────────────
    # 1-agent minimum + low net-score floor; volume spike is the primary
    # confirmation. Still quality-filtered (SMC + ensemble), NOT "take everything".
    "AGGRESSIVE": TradingProfile(
        name="AGGRESSIVE",
        min_confidence=0.42, min_agent_agreement=1, net_score_threshold=0.05,
        smc_sub_checks_min=1,
        smc_liquidity_sweep_pct=0.003, smc_bos_body_pct=0.20,
        smc_volume_spike_ratio=2.0, smc_pattern_completion=0.45,
        # Soft HTF (not "none" — blind to daily trend causes outsized losses)
        htf_filter_mode="soft", btc_momentum_filter=True,
        # Momentum: SL 2×, TP 3.5×, trail 3× — respects crypto noise, harvests fast moves.
        stop_loss_atr_mult=2.0, take_profit_atr_mult=3.5,
        tp1_fraction=0.50, tp1_r_mult=2.0, trail_atr_mult=3.0,
        early_exit_enabled=True, dynamic_tp_enabled=True,
    ),

    # ── CONFLUENCE: quality overlay — weighted scoring across all dimensions ───
    # Not an aggression mode. Replaces the boolean agreement/SMC gates with a
    # regime-dynamic weighted confluence score (ensemble, agreement, confidence,
    # volume, regime). Only fires when everything aligns.
    "CONFLUENCE": TradingProfile(
        name="CONFLUENCE",
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
        smc_sub_checks_min=2,
        smc_liquidity_sweep_pct=0.004, smc_bos_body_pct=0.35,
        smc_volume_spike_ratio=1.5, smc_pattern_completion=0.60,
        # HTF soft: counter-HTF signals penalised 20% (high-conf still passes)
        htf_filter_mode="soft", btc_momentum_filter=True,
        # Quality swing: SL 2.8×, TP 6×, trail 3× — let high-conviction ideas run far.
        stop_loss_atr_mult=2.8, take_profit_atr_mult=6.0,
        tp1_fraction=0.40, tp1_r_mult=2.8, trail_atr_mult=3.0,
        early_exit_enabled=True, dynamic_tp_enabled=True,
    ),
}
