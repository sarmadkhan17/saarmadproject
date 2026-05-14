import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta


def _make_macro_agent(trend):
    """Return a mock MacroAgent whose analyze() returns market_trend=trend."""
    m = MagicMock()
    m.analyze.return_value = {"market_trend": trend, "dom_signal": "NEUTRAL"}
    return m


def test_strong_bull_maps_to_positive_net():
    from engine.macro_agent import MacroFlowAgent
    agent = MacroFlowAgent(_make_macro_agent("STRONG_BULL"))
    sig = agent.analyze(df=None, profile=None)
    assert sig.net_score == pytest.approx(0.7)
    assert sig.buy_score == pytest.approx(0.7)
    assert sig.sell_score == pytest.approx(0.0)
    assert sig.agent == "macro_flow"


def test_strong_bear_maps_to_negative_net():
    from engine.macro_agent import MacroFlowAgent
    agent = MacroFlowAgent(_make_macro_agent("STRONG_BEAR"))
    sig = agent.analyze(df=None, profile=None)
    assert sig.net_score == pytest.approx(-0.7)
    assert sig.sell_score == pytest.approx(0.7)
    assert sig.buy_score == pytest.approx(0.0)


def test_mild_bull_maps_to_0_4():
    from engine.macro_agent import MacroFlowAgent
    agent = MacroFlowAgent(_make_macro_agent("MILD_BULL"))
    sig = agent.analyze(df=None, profile=None)
    assert sig.net_score == pytest.approx(0.4)


def test_mild_bear_maps_to_minus_0_4():
    from engine.macro_agent import MacroFlowAgent
    agent = MacroFlowAgent(_make_macro_agent("MILD_BEAR"))
    sig = agent.analyze(df=None, profile=None)
    assert sig.net_score == pytest.approx(-0.4)


def test_neutral_maps_to_zero():
    from engine.macro_agent import MacroFlowAgent
    agent = MacroFlowAgent(_make_macro_agent("NEUTRAL"))
    sig = agent.analyze(df=None, profile=None)
    assert sig.net_score == pytest.approx(0.0)
    assert sig.confidence == pytest.approx(0.2)


def test_confidence_formula():
    from engine.macro_agent import MacroFlowAgent
    # STRONG_BULL: abs(0.7) * 0.8 + 0.2 = 0.76
    agent = MacroFlowAgent(_make_macro_agent("STRONG_BULL"))
    sig = agent.analyze(df=None, profile=None)
    assert sig.confidence == pytest.approx(0.76)


def test_http_called_once_within_ttl():
    """MacroAgent.analyze() should only be called once per TTL window."""
    from engine.macro_agent import MacroFlowAgent
    mock_macro = _make_macro_agent("MILD_BULL")
    agent = MacroFlowAgent(mock_macro)
    agent.analyze(df=None, profile=None)
    agent.analyze(df=None, profile=None)
    assert mock_macro.analyze.call_count == 1


def test_http_called_again_after_ttl():
    """MacroAgent.analyze() should be called again once TTL expires."""
    from engine.macro_agent import MacroFlowAgent
    mock_macro = _make_macro_agent("MILD_BULL")
    agent = MacroFlowAgent(mock_macro)
    agent.analyze(df=None, profile=None)

    # Force cache expiry by back-dating _cache_time
    agent._cache_time = datetime.now(timezone.utc) - timedelta(seconds=MacroFlowAgent.TTL + 1)
    agent.analyze(df=None, profile=None)
    assert mock_macro.analyze.call_count == 2


def test_unknown_trend_maps_to_zero():
    """Unrecognised trend string should map to net_score=0.0 (NEUTRAL fallback)."""
    from engine.macro_agent import MacroFlowAgent
    agent = MacroFlowAgent(_make_macro_agent("SOMETHING_WEIRD"))
    sig = agent.analyze(df=None, profile=None)
    assert sig.net_score == pytest.approx(0.0)
