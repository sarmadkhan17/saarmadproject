"""
Tests for ExitEngine._check_invalidation:

1. The 0.3R underwater gate is too tight — entry slippage trips it. We widen to 1.0R.
2. The 3-rising-closes (or 3-falling-closes) momentum check used `iloc[-4:-1]`,
   so it looked at the 3 candles that *preceded* the entry. A short opened on
   the top of a bounce would invalidate on the very next 30-second scan even
   though no fresh candle had closed since entry. The check must look at
   candles whose index is strictly after the entry time.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import risk.manager as rm  # noqa: E402
from risk.manager import ExitEngine  # noqa: E402


class _Profile:
    early_exit_enabled    = True
    dynamic_tp_enabled    = True
    tp1_fraction          = 0.40
    tp1_r_mult            = 1.0
    trail_atr_mult        = 2.5
    take_profit_atr_mult  = 2.5


def _make_candles(closes, volumes=None, end_time=None, freq_min=15):
    """DatetimeIndex (tz-naive UTC) — same shape as feed.fetch_ohlcv output."""
    n = len(closes)
    end = end_time or datetime(2026, 5, 16, 12, 0)
    idx = pd.date_range(end=end, periods=n, freq=f"{freq_min}min")
    return pd.DataFrame({
        "open":   closes,
        "high":   [c * 1.001 for c in closes],
        "low":    [c * 0.999 for c in closes],
        "close":  closes,
        "volume": volumes if volumes is not None else [1000.0] * n,
    }, index=idx)


@pytest.fixture
def engine(tmp_path, monkeypatch):
    monkeypatch.setattr(rm, "DATA", tmp_path)
    eng = ExitEngine()
    eng._file = tmp_path / "exit_engine_test.json"
    return eng


# ── 0.3R underwater gate widened to 1.0R ──────────────────────────────────────

def test_underwater_below_1R_does_not_invalidate_even_with_rising_closes(engine):
    """At 0.5R underwater + 3 rising closes (pre-fix bug case), must NOT cut.

    Use enough candles to bypass the `len < 5` early-return, and have the trade
    aged so candles count as post-entry — this isolates the underwater-gate
    behaviour from the post-entry filter.
    """
    closes = [99.5, 99.7, 100.0, 100.5, 101.0, 101.5]  # last 3 closed = rising
    df = _make_candles(closes, end_time=datetime(2026, 5, 16, 12, 0))
    entry, atr = 101.5, 0.5
    engine.should_exit("t_under_05R", entry, 101.5, atr, side="short",
                       profile=_Profile(), candle_df=df)
    engine._entry_times["t_under_05R"] = datetime(2026, 5, 16, 9, 0, tzinfo=timezone.utc)
    # gain = -(101.75 - 101.5) = -0.25 = -0.5R underwater (past old 0.3R gate)
    fraction, reason = engine.should_exit(
        "t_under_05R", entry, 101.75, atr, side="short",
        profile=_Profile(), candle_df=df,
    )
    assert fraction == 0.0, f"At 0.5R underwater must not invalidate (got: {reason!r})"


def test_underwater_past_1R_with_post_entry_rising_closes_invalidates(engine):
    """Past 1.0R underwater AND 3 fresh post-entry rising closes → invalidate."""
    base = [100.0, 100.0, 100.0, 100.0]
    df0 = _make_candles(base, end_time=datetime(2026, 5, 16, 12, 0))
    entry, atr = 100.0, 0.5
    engine.should_exit("t_past_1R", entry, 100.0, atr, side="short",
                       profile=_Profile(), candle_df=df0)
    # Age the trade so subsequent candles are post-entry
    engine._entry_times["t_past_1R"] = datetime(2026, 5, 16, 12, 0, tzinfo=timezone.utc)

    later = base + [100.3, 100.6, 100.9, 101.0]   # 3 rising closes (post-entry)
    df1 = _make_candles(later, end_time=datetime(2026, 5, 16, 14, 0))
    # gain = -1.0 = -2.0R underwater → past 1.0R gate
    fraction, reason = engine.should_exit(
        "t_past_1R", entry, 101.0, atr, side="short",
        profile=_Profile(), candle_df=df1,
    )
    assert fraction == 1.0
    assert "momentum_reversal" in reason


# ── 3-rising-closes must use POST-entry candles only ──────────────────────────

def test_just_opened_short_not_invalidated_by_preentry_rising_closes(engine):
    """Short opened on the top of a bounce: the 3 closed candles preceding
    entry are rising, but no post-entry candles exist yet. Must NOT invalidate."""
    closes = [100.0, 100.5, 101.0, 101.5, 102.0, 102.5]  # all pre-entry, last 3 rising
    df = _make_candles(closes, end_time=datetime(2026, 5, 16, 12, 0))
    entry, atr = 102.5, 0.5
    # Entry time = now → ALL candles are pre-entry. gain = -2R (past the 1.0R gate).
    fraction, reason = engine.should_exit(
        "t_just_opened", entry, 103.5, atr, side="short",
        profile=_Profile(), candle_df=df,
    )
    assert fraction == 0.0, (
        f"Just-opened short must not invalidate on pre-entry rising candles "
        f"(got: {reason!r})"
    )


def test_long_not_invalidated_by_preentry_falling_closes(engine):
    """Symmetric: long opened on the bottom of a dip — pre-entry candles falling
    must not trigger invalidation."""
    closes = [105.0, 104.0, 103.0, 102.0, 101.0, 100.0]  # all pre-entry, last 3 falling
    df = _make_candles(closes, end_time=datetime(2026, 5, 16, 12, 0))
    entry, atr = 100.0, 0.5
    # gain = -(100 - 99) = -1.0 = -2R underwater → past gate.
    fraction, reason = engine.should_exit(
        "t_just_opened_long", entry, 99.0, atr, side="long",
        profile=_Profile(), candle_df=df,
    )
    assert fraction == 0.0, (
        f"Just-opened long must not invalidate on pre-entry falling candles "
        f"(got: {reason!r})"
    )


def test_volume_collapse_still_fires_when_underwater(engine):
    """Volume-collapse path is unchanged and still fires when sufficiently
    underwater. (Sanity that the rewrite didn't kill the other branch.)"""
    n = 30
    closes = [100.0] * n
    # iloc[-2] is the last *completed* bar — the volume-collapse check reads that.
    volumes = [1000.0] * (n - 2) + [50.0, 1000.0]
    df = _make_candles(closes, volumes, end_time=datetime(2026, 5, 16, 12, 0))
    entry, atr = 100.0, 0.5
    engine.should_exit("t_vol_collapse", entry, 100.0, atr, side="long",
                       profile=_Profile(), candle_df=df)
    # gain = -(100-99) = -1 = -2R underwater
    fraction, reason = engine.should_exit(
        "t_vol_collapse", entry, 99.0, atr, side="long",
        profile=_Profile(), candle_df=df,
    )
    assert fraction == 1.0
    assert "volume_collapse" in reason


# ── Entry-time persistence ────────────────────────────────────────────────────

def test_entry_time_recorded_at_init(engine):
    df = _make_candles([100.0] * 6)
    engine.should_exit("t_record", 100.0, 100.0, 0.5, side="long",
                       profile=_Profile(), candle_df=df)
    assert "t_record" in engine._entry_times
    et = engine._entry_times["t_record"]
    # Must be tz-aware UTC datetime (or ISO string round-trip)
    if isinstance(et, str):
        et = datetime.fromisoformat(et)
    assert et.tzinfo is not None
    assert (datetime.now(timezone.utc) - et).total_seconds() < 5


def test_entry_time_round_trips_through_save_load(engine, tmp_path, monkeypatch):
    df = _make_candles([100.0] * 6)
    engine.should_exit("t_persist", 100.0, 100.0, 0.5, side="long",
                       profile=_Profile(), candle_df=df)
    engine.flush()

    monkeypatch.setattr(rm, "DATA", tmp_path)
    eng2 = ExitEngine()
    eng2._file = engine._file
    eng2._load()
    assert "t_persist" in eng2._entry_times
