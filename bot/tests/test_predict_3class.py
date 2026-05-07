# bot/tests/test_predict_3class.py
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.ai_strategy import RandomForestStrategy, LightGBMStrategy, make_labels, make_features


def _make_volatile_df(n=500):
    np.random.seed(42)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    closes = np.maximum(closes, 10)
    return pd.DataFrame({
        "open":   closes * (1 + np.random.uniform(-0.002, 0.002, n)),
        "high":   closes * (1 + np.random.uniform(0.001, 0.01, n)),
        "low":    closes * (1 - np.random.uniform(0.001, 0.01, n)),
        "close":  closes,
        "volume": np.random.uniform(1000, 5000, n),
    })


def test_rf_predict_returns_valid_action():
    df = _make_volatile_df(500)
    rf = RandomForestStrategy(mode="spot")
    rf.train(df, forward_bars=1, atr_k=0.3, n_jobs=1)
    result = rf.predict(df)
    assert result["action"] in ("BUY", "SELL", "HOLD"), f"Got: {result['action']}"
    assert 0.0 <= result["confidence"] <= 1.0


def test_rf_predict_probs_are_3_elements():
    df = _make_volatile_df(500)
    rf = RandomForestStrategy(mode="spot")
    rf.train(df, forward_bars=1, atr_k=0.3, n_jobs=1)
    result = rf.predict(df)
    assert len(result["probs"]) == 3, f"Expected 3 probs, got {len(result['probs'])}"
    assert abs(sum(result["probs"]) - 1.0) < 1e-5


def test_lgbm_predict_probs_are_3_elements():
    df = _make_volatile_df(500)
    lgbm = LightGBMStrategy(mode="spot")
    lgbm.train(df, forward_bars=1, atr_k=0.3, n_jobs=1)
    result = lgbm.predict(df)
    assert len(result["probs"]) == 3, f"Expected 3 probs, got {len(result['probs'])}"
    assert abs(sum(result["probs"]) - 1.0) < 1e-5


def test_untrained_returns_hold_with_3_probs():
    rf = RandomForestStrategy(mode="spot")
    rf.is_trained = False
    result = rf.predict(pd.DataFrame())
    assert result["action"] == "HOLD"
    assert len(result["probs"]) == 3
