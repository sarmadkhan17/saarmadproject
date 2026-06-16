"""
Tests for ShadowTracker's post-exit excursion tracker — the read-only instrument
that watches price AFTER a trade exits to measure (Q3) how far the move continued
and (Q2) whether a stopped trade later reached its original TP (noise vs genuine).
"""

import os
import sys
from datetime import timedelta

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot.agents.shadow_tracker import ShadowTracker
from bot.core.tz import LOCAL_TZ
import datetime as _dt


@pytest.fixture
def tracker(tmp_path):
    return ShadowTracker(tmp_path / "t.db", "futures", {})


def _candles(start, highs, lows, closes, step_min=15):
    """Build a 15m OHLCV frame with a naive-UTC DatetimeIndex starting at `start`."""
    idx = pd.date_range(start=start, periods=len(highs), freq=f"{step_min}min")
    return pd.DataFrame({"high": highs, "low": lows,
                         "close": closes, "open": closes}, index=idx)


def _feed(df):
    def fetch(symbol, tf, limit):
        return df
    return fetch


def test_start_track_validates_input(tracker):
    assert tracker.start_post_exit_track(
        "taken", "BTCUSDT", "bad_side", 100, 2.0, 101, "x") is None
    assert tracker.start_post_exit_track(
        "taken", "BTCUSDT", "long", 100, 0.0, 101, "x") is None   # r_unit<=0
    assert tracker.start_post_exit_track(
        "taken", "BTCUSDT", "long", 100, 2.0, 101, "x") is not None


def test_long_winner_continued_running(tracker):
    # Exited a long at 110 (entry 100, r_unit=10 → exited +1R). Price then ran to
    # 130 → 2R MORE was available past our exit.
    now = _dt.datetime.now(LOCAL_TZ)
    exit_at = now - timedelta(hours=9)   # already older than the 8h window
    tid = tracker.start_post_exit_track(
        "taken", "BTCUSDT", "long", entry_price=100, r_unit=10,
        exit_price=110, exit_reason="TP_BACKSTOP", exit_r=1.0,
        orig_tp=110, orig_sl=90, watch_hours=8.0, now=exit_at)
    assert tid
    # candles after exit: high reaches 130, low never below 108
    df = _candles(exit_at.astimezone(_dt.timezone.utc).replace(tzinfo=None),
                  highs=[115, 130, 125], lows=[108, 120, 118], closes=[112, 128, 124])
    n = tracker.update_post_exit_tracks(_feed(df), now=now)
    assert n == 1
    row = tracker.post_exit_rows(days=1)[0]
    assert row["status"] == "done"
    assert row["post_mfe_r"] == pytest.approx((130 - 110) / 10)   # +2R further
    assert row["continued_r"] == pytest.approx((124 - 100) / 10)  # last close vs entry


def test_short_stop_was_noise_recovered_to_tp(tracker):
    # Short entry 100, r_unit=10, stopped out at 110 (exit_r=-1). Price then FELL
    # to the original TP at 85 → the stop was noise (would have won).
    now = _dt.datetime.now(LOCAL_TZ)
    exit_at = now - timedelta(hours=9)
    tracker.start_post_exit_track(
        "taken", "ETHUSDT", "short", entry_price=100, r_unit=10,
        exit_price=110, exit_reason="ATR_TRAIL SL $110", exit_r=-1.0,
        orig_tp=85, orig_sl=110, watch_hours=8.0, now=exit_at)
    df = _candles(exit_at.astimezone(_dt.timezone.utc).replace(tzinfo=None),
                  highs=[110, 105, 95], lows=[104, 84, 86], closes=[106, 88, 90])
    tracker.update_post_exit_tracks(_feed(df), now=now)
    row = tracker.post_exit_rows(days=1)[0]
    assert row["recovered_to_tp"] == 1            # reached TP after the stop → noise
    # favorable (down for a short) past the exit at 110: best low 84 → (110-84)/10
    assert row["post_mfe_r"] == pytest.approx((110 - 84) / 10)


def test_stop_was_genuine_not_recovered(tracker):
    # Long stopped at 90 (entry 100, exit_r=-1); price keeps falling, never reaches
    # the 120 TP → genuine stop.
    now = _dt.datetime.now(LOCAL_TZ)
    exit_at = now - timedelta(hours=9)
    tracker.start_post_exit_track(
        "taken", "SOLUSDT", "long", entry_price=100, r_unit=10,
        exit_price=90, exit_reason="ATR_TRAIL SL $90", exit_r=-1.0,
        orig_tp=120, orig_sl=90, watch_hours=8.0, now=exit_at)
    df = _candles(exit_at.astimezone(_dt.timezone.utc).replace(tzinfo=None),
                  highs=[91, 88, 85], lows=[86, 83, 80], closes=[88, 85, 82])
    tracker.update_post_exit_tracks(_feed(df), now=now)
    row = tracker.post_exit_rows(days=1)[0]
    assert row["recovered_to_tp"] == 0


def test_not_finalized_before_watch_window(tracker):
    # exit just now → still inside the 8h window → stays 'watching', no metrics.
    now = _dt.datetime.now(LOCAL_TZ)
    tracker.start_post_exit_track(
        "taken", "BTCUSDT", "long", 100, 10, 105, "x", exit_r=0.5,
        watch_hours=8.0, now=now)
    df = _candles(now.astimezone(_dt.timezone.utc).replace(tzinfo=None),
                  highs=[106], lows=[104], closes=[105])
    n = tracker.update_post_exit_tracks(_feed(df), now=now)
    assert n == 0
    assert tracker.post_exit_rows(days=1) == []          # nothing finalized
    assert len(tracker.post_exit_rows(days=1, status="watching")) == 1


def test_recovered_only_set_for_losing_exits(tracker):
    # A winning exit (exit_r>0) must not get a recovered_to_tp verdict.
    now = _dt.datetime.now(LOCAL_TZ)
    exit_at = now - timedelta(hours=9)
    tracker.start_post_exit_track(
        "taken", "BTCUSDT", "long", entry_price=100, r_unit=10,
        exit_price=112, exit_reason="TP_BACKSTOP", exit_r=1.2,
        orig_tp=112, orig_sl=90, watch_hours=8.0, now=exit_at)
    df = _candles(exit_at.astimezone(_dt.timezone.utc).replace(tzinfo=None),
                  highs=[120], lows=[111], closes=[118])
    tracker.update_post_exit_tracks(_feed(df), now=now)
    row = tracker.post_exit_rows(days=1)[0]
    assert row["recovered_to_tp"] is None
