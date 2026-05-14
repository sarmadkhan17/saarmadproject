import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock
from engine.ensemble import EnsembleEngine
from engine.smc_agent import AgentSignal


def _signal(agent, net):
    s = AgentSignal(agent=agent, buy_score=max(net, 0), sell_score=max(-net, 0),
                    net_score=net, confidence=0.65)
    return s


def _engine():
    smc  = MagicMock()
    tech = MagicMock()
    macro = MagicMock()
    return EnsembleEngine(smc, tech, macro)


def test_buy_net_reduced_in_bearish_trend():
    """Positive net_score is multiplied by 0.7 when trend_direction=BEARISH."""
    engine = _engine()
    signals = [
        _signal("smc",       0.50),
        _signal("technical", 0.60),
        _signal("macro_flow",0.40),
    ]
    result = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                               market_ctx={"trend_direction": "BEARISH", "vol_ratio": 1.0, "adx": 30.0},
                               regime="STRONG_TREND")
    neutral = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                                market_ctx={"trend_direction": "NEUTRAL", "vol_ratio": 1.0, "adx": 30.0},
                                regime="STRONG_TREND")
    assert result.net_score < neutral.net_score, (
        f"bearish-trend net={result.net_score} should be < neutral net={neutral.net_score}"
    )


def test_sell_net_reduced_in_bullish_trend():
    """Negative net_score is multiplied by 0.7 when trend_direction=BULLISH."""
    engine = _engine()
    signals = [
        _signal("smc",       -0.50),
        _signal("technical", -0.60),
        _signal("macro_flow",-0.40),
    ]
    result = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                               market_ctx={"trend_direction": "BULLISH", "vol_ratio": 1.0, "adx": 30.0},
                               regime="STRONG_TREND")
    neutral = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                                market_ctx={"trend_direction": "NEUTRAL", "vol_ratio": 1.0, "adx": 30.0},
                                regime="STRONG_TREND")
    assert result.net_score > neutral.net_score, (
        f"bullish-trend net={result.net_score} should be > neutral net={neutral.net_score}"
    )


def test_neutral_trend_no_bias():
    """NEUTRAL trend direction applies no bias — same as absent key."""
    engine = _engine()
    signals = [_signal("smc", 0.50), _signal("technical", 0.60)]
    with_neutral = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                                     market_ctx={"trend_direction": "NEUTRAL", "vol_ratio": 1.0, "adx": 30.0},
                                     regime="RANGING")
    without_key = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                                    market_ctx={"vol_ratio": 1.0, "adx": 30.0},
                                    regime="RANGING")
    assert with_neutral.net_score == pytest.approx(without_key.net_score, abs=0.001)


def test_buy_in_bullish_trend_not_damped():
    """BUY net_score in a BULLISH trend should NOT be reduced."""
    engine = _engine()
    signals = [_signal("smc", 0.50), _signal("technical", 0.60)]
    bullish = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                                market_ctx={"trend_direction": "BULLISH", "vol_ratio": 1.0, "adx": 30.0},
                                regime="STRONG_TREND")
    neutral = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                                market_ctx={"trend_direction": "NEUTRAL", "vol_ratio": 1.0, "adx": 30.0},
                                regime="STRONG_TREND")
    assert bullish.net_score == pytest.approx(neutral.net_score, abs=0.001)


def test_30pct_reduction_magnitude():
    """Verify the reduction factor is exactly 0.70 (30% reduction)."""
    engine = _engine()
    signals = [_signal("smc", 0.50), _signal("technical", 0.50)]
    bearish = engine._aggregate(signals, MagicMock(net_score_threshold=0.01),
                                market_ctx={"trend_direction": "BEARISH", "vol_ratio": 1.0, "adx": 30.0},
                                regime="RANGING")
    neutral = engine._aggregate(signals, MagicMock(net_score_threshold=0.01),
                                market_ctx={"trend_direction": "NEUTRAL", "vol_ratio": 1.0, "adx": 30.0},
                                regime="RANGING")
    if neutral.net_score > 0:
        ratio = bearish.net_score / neutral.net_score
        assert abs(ratio - 0.70) < 0.05, f"Expected ~0.70 reduction, got ratio={ratio:.3f}"
