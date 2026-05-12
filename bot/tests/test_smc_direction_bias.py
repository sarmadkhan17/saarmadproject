import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from engine.smc_agent import SMCAgent


def _make_df(n=100):
    np.random.seed(0)
    close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.002,
        "low":    close * 0.998,
        "close":  close,
        "volume": np.random.uniform(1000, 3000, n),
    })


def _profile():
    p = MagicMock()
    p.smc_liquidity_sweep_pct = 0.001
    p.smc_bos_body_pct = 0.001
    p.smc_volume_spike_ratio = 2.0
    p.smc_pattern_completion = 0.6
    p.smc_sub_checks_min = 1
    return p


def _patched_agent(sweep_dir=None, bos_dir=None):
    """SMCAgent with all detectors mocked to return known directions."""
    agent = SMCAgent()
    agent._detect_liquidity_sweep = lambda *a, **k: (
        {"direction": sweep_dir, "pct": 0.005, "level": 100.0} if sweep_dir else {"direction": None}
    )
    agent._detect_bos = lambda *a, **k: (
        {"direction": bos_dir, "body_pct": 0.01} if bos_dir else {"direction": None}
    )
    agent._detect_fvg             = lambda *a, **k: {"direction": None}
    agent._detect_volume_spike    = lambda *a, **k: {}
    agent._detect_pattern_completion = lambda *a, **k: {"direction": None}
    return agent


def test_bearish_sweep_in_bearish_trend_gets_full_credit():
    """Bearish sweep must score 0.35 (full) in STRONG_TREND + BEARISH."""
    agent = _patched_agent(sweep_dir="bearish")
    result = agent.analyze(_make_df(), _profile(),
                           {"regime": "STRONG_TREND", "trend_direction": "BEARISH"})
    assert result.sell_score == pytest.approx(0.35), f"sell_score={result.sell_score}"
    assert result.buy_score == 0.0


def test_bearish_sweep_in_bullish_trend_is_suppressed():
    """Bearish sweep must score 0.0 (suppressed) in STRONG_TREND + BULLISH."""
    agent = _patched_agent(sweep_dir="bearish")
    result = agent.analyze(_make_df(), _profile(),
                           {"regime": "STRONG_TREND", "trend_direction": "BULLISH"})
    assert result.sell_score == 0.0, f"sell_score should be 0, got {result.sell_score}"


def test_bearish_sweep_no_trend_gets_full_credit():
    """Bearish sweep must score 0.35 when not in a trend regime."""
    agent = _patched_agent(sweep_dir="bearish")
    result = agent.analyze(_make_df(), _profile(),
                           {"regime": "RANGING", "trend_direction": "NEUTRAL"})
    assert result.sell_score == pytest.approx(0.35)


def test_bullish_sweep_in_bullish_trend_gets_full_credit():
    agent = _patched_agent(sweep_dir="bullish")
    result = agent.analyze(_make_df(), _profile(),
                           {"regime": "STRONG_TREND", "trend_direction": "BULLISH"})
    assert result.buy_score == pytest.approx(0.35)


def test_bullish_sweep_in_bearish_trend_is_suppressed():
    agent = _patched_agent(sweep_dir="bullish")
    result = agent.analyze(_make_df(), _profile(),
                           {"regime": "STRONG_TREND", "trend_direction": "BEARISH"})
    assert result.buy_score == 0.0


def test_bearish_bos_in_bearish_trend_gets_full_credit():
    agent = _patched_agent(bos_dir="bearish")
    result = agent.analyze(_make_df(), _profile(),
                           {"regime": "STRONG_TREND", "trend_direction": "BEARISH"})
    assert result.sell_score == pytest.approx(0.30)


def test_bearish_bos_in_bullish_trend_is_suppressed():
    agent = _patched_agent(bos_dir="bearish")
    result = agent.analyze(_make_df(), _profile(),
                           {"regime": "STRONG_TREND", "trend_direction": "BULLISH"})
    assert result.sell_score == 0.0
