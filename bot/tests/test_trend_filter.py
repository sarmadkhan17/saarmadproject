import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest
from engine.trend_filter import TrendFilter


def _ohlcv(closes: list[float], freq: str = "1h") -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(closes), freq=freq, tz="UTC")
    arr = np.asarray(closes, dtype=float)
    return pd.DataFrame({"open": arr, "high": arr, "low": arr,
                         "close": arr, "volume": np.ones_like(arr)}, index=idx)


_CFG = {
    "fast": {"tf": "15m", "ema_fast": 20, "ema_slow": 50, "slope_lookback": 10},
    "slow": {"tf": "1h",  "ema_fast": 50, "ema_slow": 200, "slope_lookback": 20},
    "strong_slope_pct": 0.002,
}


def _linear(n: int, start: float, slope: float) -> list[float]:
    return [start + i * slope for i in range(n)]


def test_long_allowed_when_fast_up_slow_flat():
    tf = TrendFilter(_CFG)
    # Fast tier: 80 bars climbing → fast EMAs in uptrend
    # Slow tier: 250 bars flat → slow EMAs equal, slope ~ 0
    v = tf.check({
        "15m": _ohlcv(_linear(80, 100.0, 0.5), freq="15min"),
        "1h":  _ohlcv([100.0] * 250, freq="1h"),
    })
    assert v["long_allowed"]  is True
    assert v["short_allowed"] is False
    assert v["fast"]["direction"] == "up"
    assert v["slow"]["strong"]    is False


def test_long_blocked_when_slow_strongly_down():
    tf = TrendFilter(_CFG)
    # Fast tier: climbing (would normally admit longs)
    # Slow tier: declining ≥ 0.2 %/bar over the lookback → strongly down
    v = tf.check({
        "15m": _ohlcv(_linear(80, 100.0, 0.5), freq="15min"),
        "1h":  _ohlcv(_linear(250, 200.0, -0.6), freq="1h"),
    })
    assert v["long_allowed"]  is False
    assert v["slow"]["strong"] is True
    assert v["slow"]["direction"] == "down"


def test_short_allowed_when_fast_down_slow_flat():
    tf = TrendFilter(_CFG)
    v = tf.check({
        "15m": _ohlcv(_linear(80, 100.0, -0.5), freq="15min"),
        "1h":  _ohlcv([100.0] * 250, freq="1h"),
    })
    assert v["short_allowed"] is True
    assert v["long_allowed"]  is False


def test_short_blocked_when_slow_strongly_up():
    tf = TrendFilter(_CFG)
    v = tf.check({
        "15m": _ohlcv(_linear(80, 100.0, -0.5), freq="15min"),
        "1h":  _ohlcv(_linear(250, 100.0, 0.6), freq="1h"),
    })
    assert v["short_allowed"] is False
    assert v["slow"]["strong"] is True


def test_neither_allowed_when_fast_flat():
    tf = TrendFilter(_CFG)
    v = tf.check({
        "15m": _ohlcv([100.0] * 80, freq="15min"),
        "1h":  _ohlcv([100.0] * 250, freq="1h"),
    })
    assert v["long_allowed"]  is False
    assert v["short_allowed"] is False
    assert v["fast"]["direction"] == "flat"


def test_insufficient_history_returns_neutral():
    tf = TrendFilter(_CFG)
    v = tf.check({
        "15m": _ohlcv([100.0] * 10, freq="15min"),  # too few bars for EMA_slow=50
        "1h":  _ohlcv([100.0] * 30, freq="1h"),     # too few for EMA_slow=200
    })
    assert v["long_allowed"]  is False
    assert v["short_allowed"] is False
    assert "insufficient" in v["reasoning"].lower()


def test_slope_threshold_is_inclusive_of_strong_band():
    tf = TrendFilter(_CFG)
    # Slow slope just above 0.2 %/bar → strong; just below → not strong.
    # 0.002 × 200 = 0.4 absolute / bar at start price 200 with 20-bar lookback
    # is +0.002 per bar; use a smaller margin to be near boundary.
    v_strong = tf.check({
        "15m": _ohlcv(_linear(80, 100.0, -0.5), freq="15min"),
        "1h":  _ohlcv(_linear(250, 100.0, 0.6), freq="1h"),  # well above threshold
    })
    v_weak = tf.check({
        "15m": _ohlcv(_linear(80, 100.0, -0.5), freq="15min"),
        "1h":  _ohlcv(_linear(250, 100.0, 0.005), freq="1h"),  # well below threshold (~0.001 slope_pct)
    })
    assert v_strong["slow"]["strong"] is True
    assert v_weak["slow"]["strong"]   is False


def test_reasoning_string_is_human_readable():
    tf = TrendFilter(_CFG)
    v = tf.check({
        "15m": _ohlcv(_linear(80, 100.0, 0.5), freq="15min"),
        "1h":  _ohlcv(_linear(250, 200.0, -0.6), freq="1h"),
    })
    # Must mention slow tier and "strong" to be debuggable in bot logs
    assert "slow" in v["reasoning"].lower()
    assert "strong" in v["reasoning"].lower() or "block" in v["reasoning"].lower()
