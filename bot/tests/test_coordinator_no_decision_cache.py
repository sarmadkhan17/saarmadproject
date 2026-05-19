import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from unittest.mock import MagicMock
from agents.coordinator import AgentCoordinator


def _make_coord():
    coord = AgentCoordinator.__new__(AgentCoordinator)
    # Minimal fields the decision path touches.
    coord.fear_greed = MagicMock(); coord.fear_greed.analyze.return_value = {"value": 50}
    coord.macro      = MagicMock(); coord.macro.analyze.return_value = {"market_trend": "NEUTRAL"}
    coord.technical  = MagicMock(); coord.technical.analyze.return_value = {"signal": "NEUTRAL"}
    coord.news       = MagicMock(); coord.news.analyze.return_value = {"signal": "NEUTRAL", "score": 0}
    coord.onchain    = MagicMock(); coord.onchain.analyze.return_value = {"ath_signal": "N/A"}
    coord.master     = MagicMock()
    coord.master.decide.return_value = {
        "action": "BUY", "confidence": 0.7, "source": "ensemble",
        "reasoning": "test", "risk_level": "MEDIUM",
    }
    coord.tracker = MagicMock()
    coord._fresh_slow = None
    coord._fg_cache = None
    coord._macro_cache = None
    coord._slow_time = None
    coord._decision_actions = {}
    return coord


def test_master_decide_called_every_call_no_30min_cache():
    coord = _make_coord()
    ml_sig = {"action": "BUY", "confidence": 0.7, "indicators": {"buy_votes": 3, "sell_votes": 0}, "strategy": "ml"}
    coord.analyze("BTC/USDT", df=None, ml_signal=ml_sig)
    coord.analyze("BTC/USDT", df=None, ml_signal=ml_sig)
    coord.analyze("BTC/USDT", df=None, ml_signal=ml_sig)
    assert coord.master.decide.call_count == 3


def test_decision_cache_attribute_removed():
    coord = _make_coord()
    assert not hasattr(coord, "_decision_cache")
    assert not hasattr(coord, "_decision_time")


def test_macro_weight_full_when_fresh():
    from agents.coordinator import macro_decay_weight
    assert macro_decay_weight(staleness_seconds=0) == 1.0
    assert macro_decay_weight(staleness_seconds=1800) == 1.0


def test_macro_weight_zero_when_very_stale():
    from agents.coordinator import macro_decay_weight
    assert macro_decay_weight(staleness_seconds=3600) == 0.0
    assert macro_decay_weight(staleness_seconds=10_000) == 0.0


def test_macro_weight_linear_decay():
    from agents.coordinator import macro_decay_weight
    # Halfway through the decay window [1800, 3600] → weight ~0.5
    w = macro_decay_weight(staleness_seconds=2700)
    assert 0.49 < w < 0.51
