# bot/tests/test_make_labels.py
import pandas as pd
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.ai_strategy import make_labels


def _make_df(closes, high_mult=1.01, low_mult=0.99):
    n = len(closes)
    closes = np.array(closes, dtype=float)
    return pd.DataFrame({
        "open":   closes * 0.999,
        "high":   closes * high_mult,
        "low":    closes * low_mult,
        "close":  closes,
        "volume": np.ones(n) * 1000,
    })


def test_returns_series_of_length_n_minus_forward_bars():
    df = _make_df([100.0] * 30)
    labels = make_labels(df, forward_bars=2, atr_k=0.5)
    assert len(labels) == 28  # 30 - 2


def test_three_classes_present_on_volatile_data():
    closes = [100, 102, 100, 98, 100, 102, 100, 98, 100, 102,
              100, 98,  100, 102, 100, 98,  100, 102, 100, 98] * 3
    df = _make_df(closes, high_mult=1.05, low_mult=0.95)
    labels = make_labels(df, forward_bars=1, atr_k=0.3)
    classes = set(labels.unique())
    assert classes == {0, 1, 2}, f"Expected {{0,1,2}}, got {classes}"


def test_values_are_only_0_1_2():
    df = _make_df([100 + i * 0.1 for i in range(50)], high_mult=1.02, low_mult=0.98)
    labels = make_labels(df, forward_bars=1, atr_k=0.5)
    assert set(labels.unique()).issubset({0, 1, 2})


def test_flat_market_produces_mostly_hold():
    np.random.seed(1)
    closes = [100.0 + np.random.uniform(-0.01, 0.01) for _ in range(100)]
    df = _make_df(closes)
    labels = make_labels(df, forward_bars=1, atr_k=0.5)
    hold_frac = (labels == 1).mean()
    assert hold_frac > 0.5, f"Expected >50% HOLD in flat market, got {hold_frac:.2%}"


def test_backward_compat_atr_k_none_raises():
    df = _make_df([100.0] * 30)
    with pytest.raises(TypeError):
        make_labels(df, forward_bars=1, atr_k=None)


def test_short_dataframe_under_14_bars_returns_all_hold():
    # ATR requires 14 bars (min_periods=14) — shorter dfs produce NaN ATR → all HOLD
    df = _make_df([100.0 + i for i in range(10)])
    labels = make_labels(df, forward_bars=1, atr_k=0.5)
    assert set(labels.unique()).issubset({0, 1, 2})
    # All values should be HOLD since ATR is NaN for all bars
    assert (labels == 1).all(), f"Expected all HOLD for <14 bars, got: {labels.value_counts().to_dict()}"
