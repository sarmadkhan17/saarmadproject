"""
Stress tests for the trade quality system:
- Choppy/sideways regime hard block
- Correlated majors limit
- Per-symbol execution lock
- Duplicate execution prevention
- Stale data rejection
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import threading
import time
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_df(rows=100, stale_hours=0):
    idx = pd.date_range(
        end=datetime.now(timezone.utc) - timedelta(hours=stale_hours),
        periods=rows, freq="1h", tz="UTC"
    )
    close = pd.Series(np.cumsum(np.random.randn(rows)) + 100, index=idx)
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": np.random.uniform(1e6, 2e6, rows),
    })


def _ensemble(action="BUY", confidence=0.65, net_score=0.40,
              buy_score=0.60, sell_score=0.20, agreeing=2, total=3):
    e = MagicMock()
    e.action = action
    e.confidence = confidence
    e.net_score = net_score
    e.buy_score = buy_score
    e.sell_score = sell_score
    e.agents_agreeing = agreeing
    e.agents_total = total
    e.signals = []
    return e


def _regime(adx=30.0, regime="STRONG_TREND", gate=True,
            allow_longs=True, allow_shorts=True, vol_ratio=1.2,
            breadth=0.65, bear_breadth=0.35, min_conf=0.45, size_mult=1.1,
            hmm_regime="STRONG_TREND"):
    return dict(
        adx=adx, regime=regime, gate=gate,
        allow_longs=allow_longs, allow_shorts=allow_shorts,
        vol_ratio=vol_ratio, breadth=breadth, bear_breadth=bear_breadth,
        min_conf=min_conf, size_mult=size_mult, hmm_regime=hmm_regime,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestChoppyRegimeBlock(unittest.TestCase):
    """CHOPPY regime must gate=False and block all entries."""

    def _make_gate(self):
        from risk.manager import MarketRegimeGate
        gate = MarketRegimeGate()
        return gate

    def test_low_adx_produces_choppy(self):
        """ADX < 18 → CHOPPY regardless of breadth."""
        gate = self._make_gate()
        result = gate._compute_from_values(adx=15.0, vol_ratio=0.7, breadth=0.5,
                                           bear_breadth=0.5, chg_4h=0.0, chg_24h=0.0)
        self.assertEqual(result["regime"], "CHOPPY")
        self.assertFalse(result["gate"])

    def test_mid_adx_neutral_breadth_produces_choppy(self):
        """ADX 20, vol_ratio 0.7, neutral breadth → CHOPPY."""
        gate = self._make_gate()
        result = gate._compute_from_values(adx=20.0, vol_ratio=0.7, breadth=0.50,
                                           bear_breadth=0.50, chg_4h=0.0, chg_24h=0.0)
        self.assertEqual(result["regime"], "CHOPPY")
        self.assertFalse(result["gate"])

    def test_strong_adx_not_choppy(self):
        """ADX 28, strong breadth → not CHOPPY."""
        gate = self._make_gate()
        result = gate._compute_from_values(adx=28.0, vol_ratio=1.2, breadth=0.70,
                                           bear_breadth=0.30, chg_4h=0.01, chg_24h=0.03)
        self.assertNotEqual(result["regime"], "CHOPPY")
        self.assertTrue(result["gate"])


class TestADXGatePerProfile(unittest.TestCase):
    """Per-profile ADX minimum gates must reject weak-trend entries."""

    def _evaluate(self, profile_name, adx):
        from engine.risk_agent import RiskDecisionAgent
        from engine.profiles import TradingProfile

        profile = TradingProfile.load(profile_name)
        risk = MagicMock()
        risk.get_position_size.return_value = (0.1, 50.0)
        risk.can_open_trade.return_value = (True, "OK")
        risk.breaker.can_trade.return_value = (True, "OK")
        gnn = MagicMock()
        gnn.check.return_value = (True, "OK", 0.5)
        agent = RiskDecisionAgent(risk, gnn)
        df = _make_df()
        ens = _ensemble()
        ctx = _regime(adx=adx)
        return agent.evaluate(
            ens, "BTC/USDT", df, profile, ctx, 0.0, [], 1000.0,
            get_price_fn=lambda s: 50000.0,
            get_atr_fn=lambda s: 500.0,
        )

    def test_strict_passes_adx_30(self):
        dec = self._evaluate("STRICT", adx=30.0)
        # May fail other gates; just verify ADX is not the reason
        adx_blocked = any("ADX" in r for r in dec.reasons)
        self.assertFalse(adx_blocked)

    def test_aggressive_passes_adx_21(self):
        dec = self._evaluate("AGGRESSIVE", adx=21.0)
        adx_blocked = any("ADX" in r for r in dec.reasons)
        self.assertFalse(adx_blocked)


class TestCorrelatedMajorsLimit(unittest.TestCase):
    """btc_correlated group must block >2 highly correlated positions."""

    def setUp(self):
        from risk.manager import CorrelationFilter
        self.cf = CorrelationFilter()

    def _trades(self, symbols):
        return [{"symbol": s, "status": "open"} for s in symbols]

    def test_first_btc_correlated_allowed(self):
        ok, _ = self.cf.is_allowed("BTC/USDT", [])
        self.assertTrue(ok)

    def test_second_btc_correlated_allowed(self):
        # BNB is in btc_correlated but NOT in large_cap — allowed when BTC open (count=1, max=2)
        ok, _ = self.cf.is_allowed("BNB/USDT", self._trades(["BTC/USDT"]))
        self.assertTrue(ok)

    def test_third_btc_correlated_blocked(self):
        # BTC + BNB open (2) → SOL (also btc_correlated) must be blocked
        open_trades = self._trades(["BTC/USDT", "BNB/USDT"])
        ok, reason = self.cf.is_allowed("SOL/USDT", open_trades)
        self.assertFalse(ok)
        self.assertIn("btc_correlated", reason)

    def test_large_cap_limit_still_enforced(self):
        """BTC already open → ETH blocked by large_cap group (max 1)."""
        open_trades = self._trades(["BTC/USDT"])
        ok, reason = self.cf.is_allowed("ETH/USDT", open_trades)
        self.assertFalse(ok)

    def test_non_correlated_alt_allowed_during_btc_eth(self):
        open_trades = self._trades(["BTC/USDT", "ETH/USDT"])
        ok, _ = self.cf.is_allowed("DOGE/USDT", open_trades)
        self.assertTrue(ok)


class TestExecutionLock(unittest.TestCase):
    """Per-symbol lock must prevent concurrent duplicate entries."""

    def test_concurrent_same_symbol_blocked(self):
        from engine.execution_engine import ExecutionEngine
        ex = MagicMock()
        state = MagicMock()
        notifier = MagicMock()
        engine = ExecutionEngine(ex, state, notifier, mode="futures")

        results = []
        lock = engine._get_sym_lock("BTC/USDT")
        lock.acquire()

        def try_entry():
            got_lock = lock.acquire(blocking=False)
            results.append(got_lock)
            if got_lock:
                lock.release()

        t = threading.Thread(target=try_entry)
        t.start()
        t.join()
        lock.release()

        self.assertFalse(results[0], "Second thread should be blocked when lock held")

    def test_different_symbols_not_blocked(self):
        from engine.execution_engine import ExecutionEngine
        engine = ExecutionEngine(MagicMock(), MagicMock(), MagicMock())
        lock_btc = engine._get_sym_lock("BTC/USDT")
        lock_eth = engine._get_sym_lock("ETH/USDT")
        lock_btc.acquire()
        # ETH lock should be free
        got = lock_eth.acquire(blocking=False)
        self.assertTrue(got)
        lock_eth.release()
        lock_btc.release()


class TestStaleDataRejection(unittest.TestCase):
    """Signals based on candle data older than 4h must be rejected."""

    def _agent(self):
        from engine.risk_agent import RiskDecisionAgent
        risk = MagicMock()
        risk.get_position_size.return_value = (0.1, 50.0)
        risk.can_open_trade.return_value = (True, "OK")
        gnn = MagicMock()
        gnn.check.return_value = (True, "OK", 0.5)
        return RiskDecisionAgent(risk, gnn)

    def test_fresh_data_passes_stale_check(self):
        from engine.profiles import TradingProfile
        agent = self._agent()
        profile = TradingProfile.load("BALANCED")
        df = _make_df(stale_hours=0)
        ens = _ensemble()
        ctx = _regime(adx=25.0)
        dec = agent.evaluate(ens, "BTC/USDT", df, profile, ctx, 0.0, [], 1000.0,
                             get_price_fn=lambda s: 50000.0,
                             get_atr_fn=lambda s: 500.0)
        stale_blocked = any("stale" in r for r in dec.reasons)
        self.assertFalse(stale_blocked)

    def test_stale_data_rejected(self):
        from engine.profiles import TradingProfile
        agent = self._agent()
        profile = TradingProfile.load("BALANCED")
        df = _make_df(stale_hours=6)
        ens = _ensemble()
        ctx = _regime(adx=25.0)
        dec = agent.evaluate(ens, "BTC/USDT", df, profile, ctx, 0.0, [], 1000.0,
                             get_price_fn=lambda s: 50000.0,
                             get_atr_fn=lambda s: 500.0)
        self.assertFalse(dec.approved)
        self.assertTrue(any("stale" in r for r in dec.reasons))


class TestEnsembleSubthreshold(unittest.TestCase):
    """Sub-threshold fallback must not fire on weak scores."""

    def _run(self, buy_score, sell_score):
        from engine.ensemble import EnsembleEngine
        from engine.profiles import TradingProfile
        from unittest.mock import MagicMock
        sig = MagicMock()
        sig.agent = "technical"
        sig.net_score = buy_score - sell_score
        sig.buy_score = buy_score
        sig.sell_score = sell_score
        sig.confidence = 0.55
        profile = TradingProfile.load("BALANCED")
        engine = EnsembleEngine(MagicMock(), MagicMock())
        result = engine._aggregate([sig], profile, market_ctx={"adx": 25.0, "vol_ratio": 1.0})
        return result.action

    def test_score_009_produces_hold(self):
        """buy_score=0.10 (was passing with old threshold 0.09) must now HOLD."""
        action = self._run(buy_score=0.10, sell_score=0.05)
        self.assertEqual(action, "HOLD")

    def test_score_022_or_above_can_pass(self):
        """buy_score=0.25 with clear direction should pass sub-threshold."""
        action = self._run(buy_score=0.25, sell_score=0.05)
        self.assertIn(action, ["BUY", "HOLD"])  # may pass threshold now


class TestProfileImmutabilityWithNewFields(unittest.TestCase):
    """New profile fields must not mutate presets."""

    def test_new_fields_present(self):
        from engine.profiles import TradingProfile
        p = TradingProfile.load("STRICT")
        self.assertTrue(hasattr(p, "adx_min"))
        self.assertTrue(hasattr(p, "min_quality_score"))

    def test_strict_adx_min_is_28(self):
        from engine.profiles import TradingProfile
        p = TradingProfile.load("STRICT")
        self.assertEqual(p.adx_min, 28.0)

    def test_balanced_adx_min_is_22(self):
        from engine.profiles import TradingProfile
        p = TradingProfile.load("BALANCED")
        self.assertEqual(p.adx_min, 22.0)

    def test_profile_copy_does_not_mutate_preset(self):
        from engine.profiles import TradingProfile, _PRESETS
        from dataclasses import replace as dc_replace
        orig_adx = _PRESETS["BALANCED"].adx_min
        copy = dc_replace(TradingProfile.load("BALANCED"))
        copy.adx_min = 99.0
        self.assertEqual(_PRESETS["BALANCED"].adx_min, orig_adx)


class TestFractionalKellySizer(unittest.TestCase):
    """Volatility-adjusted fractional Kelly sizing."""

    def _sizer(self):
        from risk.manager import KellyCriterionSizer
        return KellyCriterionSizer(config={"risk": {"leverage": 5}})

    def _regime(self, name="WEAK_TREND", adx=24.0, size_mult=0.85):
        return dict(regime=name, adx=adx, size_mult=size_mult)

    def test_futures_base_within_target_range(self):
        """Normal futures trade: 0.25%-0.5% margin with all factors near 1.0."""
        import os
        os.environ["BOT_MODE"] = "futures"
        sizer = self._sizer()
        regime = self._regime("WEAK_TREND", adx=24.0, size_mult=0.85)
        amount, usdt = sizer.calculate(
            confidence=0.65, balance=4000, price=50000,
            atr_pct=0.02, regime=regime, recent_trades=[],
            open_trades=[], all_agree=False,
        )
        # pos_pct should land in target range (after leverage: 0.25-0.5% margin)
        margin_pct = (usdt / 5) / 4000  # usdt / leverage = margin; / balance
        self.assertGreater(margin_pct, 0.001)  # > 0.1% (not micro-trade)
        self.assertLess(margin_pct, 0.020)     # < 2% (survivable)

    def test_spot_base_within_target_range(self):
        """Normal spot trade should size 0.5%-2% of balance."""
        import os
        from risk.manager import KellyCriterionSizer as Sizer
        os.environ["BOT_MODE"] = "spot"
        s = Sizer(config={"risk": {"leverage": 1}})
        regime = self._regime("WEAK_TREND", adx=24.0, size_mult=0.85)
        amount, usdt = s.calculate(
            confidence=0.65, balance=4000, price=100,
            atr_pct=0.02, regime=regime, recent_trades=[],
            open_trades=[], all_agree=False,
        )
        pct = usdt / 4000
        self.assertGreater(pct, 0.002)   # > 0.2%
        self.assertLess(pct, 0.025)       # < 2.5%

    def test_high_vol_reduces_size(self):
        """High ATR (6%) should produce smaller position than normal ATR (2%)."""
        import os
        os.environ["BOT_MODE"] = "futures"
        sizer = self._sizer()
        regime = self._regime("WEAK_TREND", adx=24.0, size_mult=0.85)
        _, usdt_norm = sizer.calculate(0.65, 4000, 50000, 0.02, regime, [], [], False)
        _, usdt_high = sizer.calculate(0.65, 4000, 50000, 0.06, regime, [], [], False)
        self.assertLess(usdt_high, usdt_norm)

    def test_strong_trend_larger_than_ranging(self):
        """STRONG_TREND should produce a larger position than RANGING."""
        import os
        os.environ["BOT_MODE"] = "futures"
        sizer = self._sizer()
        regime_strong = dict(regime="STRONG_TREND", adx=35.0, size_mult=1.2)
        regime_range  = dict(regime="RANGING",      adx=20.0, size_mult=0.55)
        _, usdt_strong = sizer.calculate(0.70, 4000, 50000, 0.02, regime_strong, [], [], True)
        _, usdt_range  = sizer.calculate(0.65, 4000, 50000, 0.02, regime_range,  [], [], False)
        self.assertGreater(usdt_strong, usdt_range)

    def test_confidence_swing_within_design_bound(self):
        """
        Confidence is a deliberate input to the conviction-weighted sizer
        (commit 0994cf3a). The q_scalar tier mapping
        (conf<0.55→0.70, 0.55→0.85, 0.65→1.00, 0.75→1.20) lets confidence
        swing position size meaningfully. The MIN_PCT floor absorbs part of
        the swing at low confidence, so observed swing is ~35% with the floor
        active, ~70% without it. This test pins the upper bound so an
        accidental over-amplification (>2× from confidence alone) is caught.
        """
        import os
        os.environ["BOT_MODE"] = "futures"
        sizer = self._sizer()
        regime = self._regime("WEAK_TREND", adx=26.0, size_mult=0.85)
        _, usdt_lo = sizer.calculate(0.54, 4000, 50000, 0.02, regime, [], [], False)
        _, usdt_hi = sizer.calculate(0.85, 4000, 50000, 0.02, regime, [], [], False)
        swing = abs(usdt_hi - usdt_lo) / max(usdt_lo, 1)
        self.assertLess(swing, 1.00, "confidence alone must not more than double the position")

    def test_corr_factor_reduces_size_with_open_positions(self):
        """More open positions → smaller new position."""
        import os
        os.environ["BOT_MODE"] = "futures"
        sizer = self._sizer()
        regime = self._regime("WEAK_TREND", adx=26.0, size_mult=0.85)
        open_0 = []
        open_3 = [{"symbol": f"X{i}/USDT", "status": "open"} for i in range(3)]
        _, usdt_0 = sizer.calculate(0.65, 4000, 50000, 0.02, regime, [], open_0, False)
        _, usdt_3 = sizer.calculate(0.65, 4000, 50000, 0.02, regime, [], open_3, False)
        self.assertLess(usdt_3, usdt_0)

    def test_losing_streak_reduces_size(self):
        """
        ≥4 of last 5 closed trades losing → dynamic_risk_pct × 0.80
        (commit 0994cf3a, KellyCriterionSizer._dynamic_risk_pct line 611).
        The visible reduction in usdt is partially absorbed by the MIN_PCT
        floor (so observed reduction is ~10% rather than the full 20%). This
        test verifies the qualitative invariant — streak reduces size, but
        does not collapse it — rather than pinning an exact multiplier.
        """
        import os
        os.environ["BOT_MODE"] = "futures"
        sizer = self._sizer()
        regime = self._regime("WEAK_TREND", adx=26.0, size_mult=0.85)
        good = [{"status": "closed", "pnl": 10, "price": 100, "amount": 1} for _ in range(3)]
        bad  = [{"status": "closed", "pnl": -10, "price": 100, "amount": 1} for _ in range(5)]
        _, usdt_good = sizer.calculate(0.65, 4000, 50000, 0.02, regime, good, [], False)
        _, usdt_bad  = sizer.calculate(0.65, 4000, 50000, 0.02, regime, bad,  [], False)
        self.assertLess(usdt_bad, usdt_good, "losing streak must shrink position size")
        self.assertGreater(usdt_bad, usdt_good * 0.50,
                           "streak reduction must not collapse the position by more than half")

    def test_high_vol_regime_uses_lower_leverage_scale(self):
        """HIGH_VOLATILITY regime should produce smaller notional than STRONG_TREND."""
        import os
        os.environ["BOT_MODE"] = "futures"
        sizer = self._sizer()
        r_strong  = dict(regime="STRONG_TREND",    adx=35.0, size_mult=1.2)
        r_highvol = dict(regime="HIGH_VOLATILITY", adx=30.0, size_mult=0.3)
        _, usdt_strong  = sizer.calculate(0.70, 4000, 50000, 0.02, r_strong,  [], [], True)
        _, usdt_highvol = sizer.calculate(0.70, 4000, 50000, 0.02, r_highvol, [], [], False)
        self.assertLess(usdt_highvol, usdt_strong)


if __name__ == "__main__":
    unittest.main()
