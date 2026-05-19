import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from unittest.mock import MagicMock, patch
from engine.ensemble import EnsembleEngine


def _dummy_dfs():
    idx = pd.date_range("2026-01-01", periods=260, freq="1h", tz="UTC")
    df1h = pd.DataFrame({"open":  [100]*260, "high": [101]*260, "low": [99]*260,
                          "close": [100 + i*0.01 for i in range(260)],
                          "volume":[1]*260}, index=idx)
    idx15 = pd.date_range("2026-01-01", periods=120, freq="15min", tz="UTC")
    df15 = pd.DataFrame({"open":  [100]*120, "high": [101]*120, "low": [99]*120,
                          "close": [100 + i*0.02 for i in range(120)],
                          "volume":[1]*120}, index=idx15)
    return {"15m": df15, "1h": df1h}


def _cfg_with_flag(use_two_tier: bool) -> dict:
    return {
        "enabled":      True,
        "use_two_tier": use_two_tier,
        "tf":           "1h", "ema_fast": 50, "ema_slow": 200,
        "veto_longs":   True, "veto_shorts": True,
        "fast": {"tf": "15m", "ema_fast": 20, "ema_slow": 50, "slope_lookback": 10},
        "slow": {"tf": "1h",  "ema_fast": 50, "ema_slow": 200, "slope_lookback": 20},
        "strong_slope_pct": 0.002,
    }


def test_flag_false_uses_legacy_path():
    eng = EnsembleEngine(MagicMock(), MagicMock(), MagicMock(),
                         trend_filter=_cfg_with_flag(False))
    assert eng._tf2 is None, "two-tier filter must not be constructed when flag is off"
    assert hasattr(eng, "_check_trend_veto"), "legacy method must remain"


def test_flag_true_constructs_two_tier_filter():
    from engine.trend_filter import TrendFilter
    eng = EnsembleEngine(MagicMock(), MagicMock(), MagicMock(),
                         trend_filter=_cfg_with_flag(True))
    assert isinstance(eng._tf2, TrendFilter)


def test_flag_true_routes_buy_through_two_tier(monkeypatch):
    eng = EnsembleEngine(MagicMock(), MagicMock(), MagicMock(),
                         trend_filter=_cfg_with_flag(True))
    called = {"n": 0}
    def fake_check(dfs):
        called["n"] += 1
        return {"long_allowed": False, "short_allowed": True,
                "reasoning": "slow strongly down → LONG blocked",
                "fast": {"direction": "up", "slope_pct": 0.01},
                "slow": {"direction": "down", "slope_pct": -0.005, "strong": True}}
    monkeypatch.setattr(eng._tf2, "check", fake_check)
    veto_reason = eng._apply_trend_filter("BUY", _dummy_dfs())
    assert called["n"] == 1
    assert veto_reason is not None
    assert "LONG blocked" in veto_reason


def test_flag_true_admits_when_two_tier_allows(monkeypatch):
    eng = EnsembleEngine(MagicMock(), MagicMock(), MagicMock(),
                         trend_filter=_cfg_with_flag(True))
    monkeypatch.setattr(eng._tf2, "check", lambda dfs: {
        "long_allowed": True, "short_allowed": False, "reasoning": "ok",
        "fast": {"direction": "up", "slope_pct": 0.01},
        "slow": {"direction": "up", "slope_pct": 0.001, "strong": False}})
    veto_reason = eng._apply_trend_filter("BUY", _dummy_dfs())
    assert veto_reason is None
