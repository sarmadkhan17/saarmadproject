"""
Tests for the trend-change signal and the symmetric counter-trend walls.

Background (data/trade_memory.db, first 154 closed trades): counter-trend
longs ran 29% WR / -0.44 avg R because Gate 4b hard-blocked shorts in bull
STRONG_TREND but only soft-penalised longs in bearish STRONG_TREND. The fix:

  1. MarketRegimeGate emits trend_change ("up"/"down"/None) when the BTC 1h
     EMA state has crossed against the standing 4h trend with breadth confirm.
  2. Gate 4b gains the missing symmetric long wall, and a detected turn lifts
     the wall / penalties for the new direction.
  3. The ensemble two-tier trend veto is overridden when the fast tier already
     points the way of a detected turn.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bot"))

from bot.risk.manager import MarketRegimeGate
from bot.engine.risk_agent import RiskDecisionAgent


# ─── helpers ────────────────────────────────────────────────────────────────

def _df_1h(trend="flat", n=120):
    if trend == "up":
        close = np.linspace(100, 150, n)
    elif trend == "down":
        close = np.linspace(150, 100, n)
    else:
        close = np.full(n, 100.0)
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="1h")
    return pd.DataFrame({"close": close, "high": close * 1.001,
                         "low": close * 0.999, "volume": np.full(n, 1000.0)},
                        index=idx)


def _ensemble_stub(action, confidence=0.60):
    return SimpleNamespace(action=action, confidence=confidence, agents_ok=True,
                           agents_agreeing=3, agents_total=3, signals=[],
                           net_score=0.40 if action == "BUY" else -0.40)


def _profile_stub():
    return SimpleNamespace(name="TEST", min_confidence=0.45,
                           min_agent_agreement=2, use_confluence_scoring=False)


class _FakeRisk:
    def get_position_size(self, *a, **k):
        return 1.0, 100.0

    def can_open_trade(self, *a, **k):
        return True, "OK"


class _FakeGnn:
    def check(self, *a, **k):
        return True, "ok", 0.0


def _regime_ctx(breadth, bear_breadth, trend_change=None, regime="STRONG_TREND"):
    return dict(regime=regime, gate=True, allow_longs=True, allow_shorts=True,
                breadth=breadth, bear_breadth=bear_breadth,
                trend_change=trend_change, hmm_regime="UNKNOWN",
                vol_ratio=1.0, adx=30.0)


def _evaluate(action, regime_ctx, confidence=0.60):
    agent = RiskDecisionAgent(risk=_FakeRisk(), gnn=_FakeGnn())
    df = _df_1h("flat")
    try:
        return agent.evaluate(
            _ensemble_stub(action, confidence), "ETH/USDT", df, _profile_stub(),
            regime_ctx, 0.0, [], 1000.0,
            get_price_fn=lambda s: 100.0, get_atr_fn=lambda s: 1.0,
        )
    except Exception:
        # A later gate needed live infra the stubs don't provide — for these
        # tests that still proves the decision made it PAST Gate 4b.
        return None


def _gate4b_reason(decision, needle):
    return (decision is not None and not decision.approved
            and any(needle in r for r in decision.reasons))


# ─── MarketRegimeGate._trend_change_signal ──────────────────────────────────

class TestTrendChangeSignal:
    def setup_method(self):
        self.gate = MarketRegimeGate()

    def test_bear_trend_with_1h_up_and_breadth_is_up(self):
        assert self.gate._trend_change_signal(
            False, True, "up", breadth=0.40, bear_breadth=0.55) == "up"

    def test_bear_trend_with_1h_up_but_no_breadth_is_none(self):
        assert self.gate._trend_change_signal(
            False, True, "up", breadth=0.10, bear_breadth=0.85) is None

    def test_bull_trend_with_1h_down_and_bear_breadth_is_down(self):
        assert self.gate._trend_change_signal(
            True, False, "down", breadth=0.50, bear_breadth=0.40) == "down"

    def test_bull_trend_with_1h_down_but_no_bear_breadth_is_none(self):
        assert self.gate._trend_change_signal(
            True, False, "down", breadth=0.80, bear_breadth=0.10) is None

    def test_aligned_trends_are_none(self):
        assert self.gate._trend_change_signal(
            False, True, "down", breadth=0.50, bear_breadth=0.50) is None
        assert self.gate._trend_change_signal(
            True, False, "up", breadth=0.50, bear_breadth=0.50) is None

    def test_neutral_4h_is_none(self):
        assert self.gate._trend_change_signal(
            False, False, "up", breadth=0.50, bear_breadth=0.50) is None

    def test_flat_1h_is_none(self):
        assert self.gate._trend_change_signal(
            False, True, "flat", breadth=0.50, bear_breadth=0.50) is None


class TestH1State:
    def setup_method(self):
        self.gate = MarketRegimeGate()

    def test_rising_series_is_up(self):
        assert self.gate._h1_state(_df_1h("up")) == "up"

    def test_falling_series_is_down(self):
        assert self.gate._h1_state(_df_1h("down")) == "down"

    def test_flat_series_is_flat(self):
        assert self.gate._h1_state(_df_1h("flat")) == "flat"

    def test_missing_or_short_history_is_flat(self):
        assert self.gate._h1_state(None) == "flat"
        assert self.gate._h1_state(_df_1h("up", n=30)) == "flat"


# ─── Gate 4b symmetry ────────────────────────────────────────────────────────

class TestGate4bSymmetry:
    def test_long_blocked_in_bearish_strong_trend(self):
        d = _evaluate("BUY", _regime_ctx(breadth=0.10, bear_breadth=0.80))
        assert _gate4b_reason(d, "longs blocked: STRONG_TREND")

    def test_long_allowed_when_trend_turning_up(self):
        d = _evaluate("BUY", _regime_ctx(breadth=0.10, bear_breadth=0.80,
                                         trend_change="up"))
        assert not _gate4b_reason(d, "longs blocked: STRONG_TREND")

    def test_short_blocked_in_bullish_strong_trend(self):
        d = _evaluate("SELL", _regime_ctx(breadth=0.80, bear_breadth=0.10))
        assert _gate4b_reason(d, "shorts blocked: STRONG_TREND")

    def test_short_allowed_when_trend_turning_down(self):
        d = _evaluate("SELL", _regime_ctx(breadth=0.80, bear_breadth=0.10,
                                          trend_change="down"))
        assert not _gate4b_reason(d, "shorts blocked: STRONG_TREND")

    def test_turn_does_not_override_extreme_guards(self):
        # Capitulation / blow-off walls at 0.85 stay regardless of the turn.
        d = _evaluate("BUY", _regime_ctx(breadth=0.90, bear_breadth=0.05,
                                         trend_change="up", regime="WEAK_TREND"))
        assert _gate4b_reason(d, "longs blocked: blow-off")

    def test_soft_penalty_lifted_by_turn(self):
        # conf 0.50 < 0.45 + penalty(0.30*1.5 capped 0.15) → rejected without
        # a turn, but passes the penalty gate when the trend is turning up.
        ctx = _regime_ctx(breadth=0.20, bear_breadth=0.80, regime="WEAK_TREND")
        d = _evaluate("BUY", ctx, confidence=0.50)
        assert _gate4b_reason(d, "long penalised")
        ctx_turn = _regime_ctx(breadth=0.20, bear_breadth=0.80,
                               trend_change="up", regime="WEAK_TREND")
        d2 = _evaluate("BUY", ctx_turn, confidence=0.50)
        assert not _gate4b_reason(d2, "long penalised")


# ─── Ensemble trend-veto override ───────────────────────────────────────────

class TestTrendVetoOverride:
    def _engine(self, fast_dir, long_allowed, short_allowed):
        from bot.engine.ensemble import EnsembleEngine
        eng = EnsembleEngine(None, None, None,
                             trend_filter={"enabled": True, "use_two_tier": True})
        eng._tf2 = SimpleNamespace(check=lambda dfs: {
            "long_allowed": long_allowed, "short_allowed": short_allowed,
            "reasoning": "test", "fast": {"direction": fast_dir},
            "slow": {"direction": "down", "strong": True},
        })
        return eng

    def test_veto_stands_without_turn(self):
        eng = self._engine("up", long_allowed=False, short_allowed=False)
        assert eng._apply_trend_filter("BUY", {}, None) is not None

    def test_turn_with_fast_alignment_overrides_buy_veto(self):
        eng = self._engine("up", long_allowed=False, short_allowed=False)
        assert eng._apply_trend_filter("BUY", {}, "up") is None

    def test_turn_without_fast_alignment_keeps_veto(self):
        eng = self._engine("flat", long_allowed=False, short_allowed=False)
        assert eng._apply_trend_filter("BUY", {}, "up") is not None

    def test_sell_veto_overridden_on_down_turn(self):
        eng = self._engine("down", long_allowed=False, short_allowed=False)
        assert eng._apply_trend_filter("SELL", {}, "down") is None
        assert eng._apply_trend_filter("SELL", {}, "up") is not None

    def test_veto_preserves_original_action_for_shadow(self):
        """A trend veto converts the result to HOLD but must keep the original
        side in vetoed_action so the rejection can be shadow-tracked."""
        from bot.engine.ensemble import EnsembleEngine
        from bot.engine.smc_agent import AgentSignal

        bull = AgentSignal("smc", 0.8, 0.0, 0.8, 0.8, reasoning="stub")
        bull_t = AgentSignal("technical", 0.8, 0.0, 0.8, 0.8, reasoning="stub")
        smc = SimpleNamespace(analyze=lambda df, p, ctx=None: bull)
        tech = SimpleNamespace(analyze=lambda df, p: bull_t)
        eng = EnsembleEngine(smc, tech, None,
                             trend_filter={"enabled": True, "use_two_tier": True})
        eng._tf2 = SimpleNamespace(check=lambda dfs: {
            "long_allowed": False, "short_allowed": False,
            "reasoning": "fast=flat → no direction admitted",
            "fast": {"direction": "flat"},
            "slow": {"direction": "down", "strong": False},
        })
        profile = SimpleNamespace(net_score_threshold=0.05)
        df = _df_1h("up")
        res = eng.run("BTC/USDT", {"1h": df}, profile, market_ctx={})
        assert res.action == "HOLD"
        assert res.source.startswith("trend_veto")
        assert res.vetoed_action == "BUY"

    def test_no_veto_leaves_vetoed_action_none(self):
        from bot.engine.ensemble import EnsembleEngine
        from bot.engine.smc_agent import AgentSignal

        bull = AgentSignal("smc", 0.8, 0.0, 0.8, 0.8, reasoning="stub")
        bull_t = AgentSignal("technical", 0.8, 0.0, 0.8, 0.8, reasoning="stub")
        smc = SimpleNamespace(analyze=lambda df, p, ctx=None: bull)
        tech = SimpleNamespace(analyze=lambda df, p: bull_t)
        eng = EnsembleEngine(smc, tech, None,
                             trend_filter={"enabled": True, "use_two_tier": True})
        eng._tf2 = SimpleNamespace(check=lambda dfs: {
            "long_allowed": True, "short_allowed": False,
            "reasoning": "fast=up slow=up → LONG allowed",
            "fast": {"direction": "up"},
            "slow": {"direction": "up", "strong": False},
        })
        profile = SimpleNamespace(net_score_threshold=0.05)
        res = eng.run("BTC/USDT", {"1h": _df_1h("up")}, profile, market_ctx={})
        assert res.action == "BUY"
        assert res.vetoed_action is None


# ─── Risk rejection → shadow gate label ─────────────────────────────────────

class TestRiskGateLabel:
    def test_categories(self):
        from bot.agents.shadow_tracker import risk_gate_label
        assert risk_gate_label(["shorts blocked: STRONG_TREND breadth=75%"]) == "risk_breadth"
        assert risk_gate_label(["longs blocked: blow-off breadth=85%"]) == "risk_breadth"
        assert risk_gate_label(["price 1.00 below 20EMA 1.10 (bearish)"]) == "risk_ema20"
        assert risk_gate_label(["HTF SELL hard-block (conf=0.50 < 0.65)"]) == "risk_htf"
        assert risk_gate_label(["BTC momentum: conf=0.40 < eff=0.50"]) == "risk_btc"
        assert risk_gate_label(["regime=CHOPPY gate closed"]) == "risk_regime"
        assert risk_gate_label(["conf=0.44 < eff=0.50"]) == "risk_conf"
        assert risk_gate_label(["agents=1/3 < 2"]) == "risk_agreement"
        assert risk_gate_label(["size too small: $5.00"]) == "risk_other"
        assert risk_gate_label([]) == "risk_other"

    def test_uses_last_reason(self):
        """Earlier reasons are passed gates' annotations; the failure is last.
        A post-HTF confidence failure attributes to the HTF gate — the
        softening is what pushed conf under the floor."""
        from bot.agents.shadow_tracker import risk_gate_label
        assert risk_gate_label(
            ["agents=2/3 ok",
             "post-HTF conf=0.48 < eff=0.50"]) == "risk_htf"
        assert risk_gate_label(
            ["HTF SELL softened → conf=0.48",
             "size too small: $5.00"]) == "risk_other"
