import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from models.ai_strategy import AIStrategyEngine


def _engine_with_mocked_models(rf_probs, lgbm_probs):
    """AIStrategyEngine with RF and LGBM mocked to return specific probs.
    probs format: [sell_p, hold_p, buy_p].
    Action is determined by argmax of probs (matching production logic).
    """
    engine = AIStrategyEngine()
    rf_action   = {0: "SELL", 1: "HOLD", 2: "BUY"}[int(np.argmax(rf_probs))]
    lgbm_action = {0: "SELL", 1: "HOLD", 2: "BUY"}[int(np.argmax(lgbm_probs))]
    engine.rf.predict   = lambda df: {"action": rf_action,   "confidence": float(max(rf_probs)),   "probs": list(rf_probs)}
    engine.lgbm.predict = lambda df: {"action": lgbm_action, "confidence": float(max(lgbm_probs)), "probs": list(lgbm_probs)}
    engine._get_dynamic_weights = lambda: (0.5, 0.5)
    return engine


def test_zero_hold_prob_no_penalty():
    """With hold_prob=0.0, confidence is unchanged."""
    # probs=[sell=0.10, hold=0.00, buy=0.90] → BUY wins argmax
    # buy_prob=0.90, hold_prob=0.00
    # conf = min(0.90, 0.95) = 0.90; penalised = 0.90 * 1.0 = 0.90
    engine = _engine_with_mocked_models([0.10, 0.00, 0.90], [0.10, 0.00, 0.90])
    result = engine.predict(pd.DataFrame(), "TEST")
    assert result["action"] == "BUY"
    assert abs(result["confidence"] - 0.90) < 0.01, f"conf={result['confidence']}"


def test_moderate_hold_prob_reduces_confidence():
    """With hold_prob=0.30, confidence is reduced by 15%."""
    # probs=[sell=0.10, hold=0.30, buy=0.60] → BUY wins argmax
    # buy_prob=0.60, hold_prob=0.30
    # conf = min(0.60, 0.95) = 0.60; penalised = 0.60 * (1 - 0.15) = 0.60 * 0.85 = 0.51
    engine = _engine_with_mocked_models([0.10, 0.30, 0.60], [0.10, 0.30, 0.60])
    result = engine.predict(pd.DataFrame(), "TEST")
    assert result["action"] == "BUY"
    assert abs(result["confidence"] - 0.51) < 0.01, f"conf={result['confidence']}"


def test_high_hold_prob_significant_penalty():
    """With hold_prob=0.45, confidence is reduced by 22.5%."""
    # probs=[sell=0.05, hold=0.45, buy=0.50] → BUY wins argmax (0.50 > 0.45)
    # buy_prob=0.50, hold_prob=0.45
    # conf = min(0.50, 0.95) = 0.50; penalised = 0.50 * (1 - 0.225) = 0.50 * 0.775 = 0.3875
    engine = _engine_with_mocked_models([0.05, 0.45, 0.50], [0.05, 0.45, 0.50])
    result = engine.predict(pd.DataFrame(), "TEST")
    assert result["action"] == "BUY"
    assert abs(result["confidence"] - 0.3875) < 0.01, f"conf={result['confidence']}"


def test_confidence_floored_at_0_35():
    """Penalised confidence is floored at 0.35 even when arithmetic goes below."""
    # probs=[sell=0.05, hold=0.60, buy=0.35] → argmax=1 → HOLD action
    # → HOLD branch fires: buy_prob=0.35 < 0.60 and sell_prob=0.05 < 0.60
    # → action="HOLD", conf=max(0.35, 0.05)=0.35
    # hold_prob=0.60; penalised = 0.35 * (1 - 0.60*0.5) = 0.35 * 0.70 = 0.245 → floor 0.35
    engine = _engine_with_mocked_models([0.05, 0.60, 0.35], [0.05, 0.60, 0.35])
    result = engine.predict(pd.DataFrame(), "TEST")
    assert result["confidence"] == pytest.approx(0.35, abs=0.005)


def test_sell_action_also_penalised():
    """SELL confidence is penalised proportionally to hold_prob."""
    # probs=[sell=0.60, hold=0.20, buy=0.20] → SELL wins argmax
    # sell_prob=0.60, hold_prob=0.20
    # conf = min(0.60, 0.95) = 0.60; penalised = 0.60 * (1 - 0.10) = 0.60 * 0.90 = 0.54
    engine = _engine_with_mocked_models([0.60, 0.20, 0.20], [0.60, 0.20, 0.20])
    result = engine.predict(pd.DataFrame(), "TEST")
    assert result["action"] == "SELL"
    assert abs(result["confidence"] - 0.54) < 0.01, f"conf={result['confidence']}"


def test_penalty_increases_with_hold_prob():
    """Higher hold_prob always produces lower confidence (monotonic)."""
    low_hold  = _engine_with_mocked_models([0.10, 0.10, 0.80], [0.10, 0.10, 0.80])
    high_hold = _engine_with_mocked_models([0.10, 0.30, 0.60], [0.10, 0.30, 0.60])
    r_low  = low_hold.predict(pd.DataFrame(), "TEST")
    r_high = high_hold.predict(pd.DataFrame(), "TEST")
    assert r_low["confidence"] > r_high["confidence"], (
        f"low_hold conf={r_low['confidence']} should exceed high_hold conf={r_high['confidence']}"
    )
