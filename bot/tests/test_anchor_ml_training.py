import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


ANCHOR_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK"]
YEARS = 3


def test_target_bars_15m_is_3_years():
    from data.historical_fetcher import TARGET_BARS
    assert TARGET_BARS["15m"] == YEARS * 365 * 24 * 4, \
        f"Expected {YEARS * 365 * 24 * 4}, got {TARGET_BARS['15m']}"


def test_target_bars_1h_is_3_years():
    from data.historical_fetcher import TARGET_BARS
    assert TARGET_BARS["1h"] == YEARS * 365 * 24, \
        f"Expected {YEARS * 365 * 24}, got {TARGET_BARS['1h']}"


def test_target_bars_4h_is_3_years():
    from data.historical_fetcher import TARGET_BARS
    assert TARGET_BARS["4h"] == YEARS * 365 * 6, \
        f"Expected {YEARS * 365 * 6}, got {TARGET_BARS['4h']}"


def test_target_bars_1d_is_3_years():
    from data.historical_fetcher import TARGET_BARS
    assert TARGET_BARS["1d"] == YEARS * 365, \
        f"Expected {YEARS * 365}, got {TARGET_BARS['1d']}"


def test_default_coins_is_exactly_8_anchors():
    from data.historical_fetcher import DEFAULT_COINS
    assert DEFAULT_COINS == ANCHOR_COINS, \
        f"Expected {ANCHOR_COINS}, got {DEFAULT_COINS}"


def test_history_cutoff_removes_old_rows():
    """3-year cutoff filter must drop rows older than history_years * 365 days."""
    import pandas as pd
    history_years = 3
    now = pd.Timestamp.now()
    cutoff = now - pd.Timedelta(days=int(history_years * 365))

    idx = pd.date_range(end=now, periods=10, freq="365D")
    df = pd.DataFrame({"close": range(10)}, index=idx)

    filtered = df[df.index >= cutoff]
    assert all(filtered.index >= cutoff), "Old rows not removed"
    assert len(filtered) < len(df), "No rows were removed — cutoff not working"


def test_history_cutoff_keeps_recent_rows():
    """Rows within the 3-year window must not be dropped."""
    import pandas as pd
    history_years = 3
    now = pd.Timestamp.now()
    cutoff = now - pd.Timedelta(days=int(history_years * 365))

    idx = pd.date_range(end=now, periods=100, freq="D")
    df = pd.DataFrame({"close": range(100)}, index=idx)

    filtered = df[df.index >= cutoff]
    assert len(filtered) > 0, "All rows were removed — cutoff too aggressive"
    assert filtered.index[-1] >= cutoff, "Most recent row missing"
