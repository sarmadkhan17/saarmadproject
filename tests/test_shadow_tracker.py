"""
Tests for ShadowTracker — forward-tracking of rejected signals.

TDD approach — tests written before implementation.

A "shadow trade" is a hypothetical trade recorded when a directional signal
is rejected at a gate (microstructure / actor_prefilter / actor / risk).
It is resolved forward against live candles to learn whether the gate
blocked a winner or saved a loss. Observational only — never auto-tunes.
"""

import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot.agents.shadow_tracker import ShadowTracker, load_stats
from bot.core.tz import LOCAL_TZ
from bot.engine.profiles import TradingProfile


# ─── helpers ────────────────────────────────────────────────────────────────

DEFAULT_CFG = {
    "enabled": True,
    "max_open": 60,
    "max_age_hours": 48,
    "per_symbol_cooldown_min": 60,
}

CTX = {
    "regime": "TRENDING_BULLISH", "ensemble_score": 0.42, "confidence": 0.55,
    "ob_imbalance": 1.8, "cvd_direction": "up", "cvd_divergence": False,
    "btc_d": 54.2, "usdt_d": 4.9,
}


def make_tracker(tmp_path, mode="futures", **overrides):
    cfg = {**DEFAULT_CFG, **overrides}
    return ShadowTracker(tmp_path / "trade_memory.db", mode, cfg)


def make_ohlcv(start, bars, tf_minutes=15):
    """Build a fetch_ohlcv-shaped DataFrame (naive-UTC DatetimeIndex,
    matching DataFeed.fetch_ohlcv output). `bars` = list of (high, low, close)."""
    start_utc = start.astimezone(timezone.utc).replace(tzinfo=None)
    idx = pd.to_datetime([start_utc + timedelta(minutes=tf_minutes * i)
                          for i in range(len(bars))])
    return pd.DataFrame(
        {
            "open":   [b[2] for b in bars],
            "high":   [b[0] for b in bars],
            "low":    [b[1] for b in bars],
            "close":  [b[2] for b in bars],
            "volume": [100.0] * len(bars),
        },
        index=pd.DatetimeIndex(idx, name="timestamp"),
    )


def fetch_fn_for(df):
    return lambda symbol, timeframe="15m", limit=200: df


def get_row(tracker, shadow_id):
    with sqlite3.connect(tracker.db_path) as c:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT * FROM shadow_trades WHERE id=?", (shadow_id,)).fetchone()
    return dict(row) if row else None


# ─── creation: SL/TP math ───────────────────────────────────────────────────

class TestCreation:
    def test_long_sl_tp_from_profile_atr(self, tmp_path):
        tracker = make_tracker(tmp_path)
        profile = TradingProfile.load("BALANCED")  # sl 2.5x, tp 4.5x
        sid = tracker.record_rejection(
            "ETH/USDT", "long", "microstructure", "CVD divergence",
            entry_price=100.0, atr=2.0, profile=profile, ctx=CTX,
        )
        assert sid is not None
        row = get_row(tracker, sid)
        assert row["sl_price"] == pytest.approx(100.0 - 2.5 * 2.0)   # 95
        assert row["tp_price"] == pytest.approx(100.0 + 4.5 * 2.0)   # 109
        assert row["r_target"] == pytest.approx(9.0 / 5.0)
        assert row["status"] == "open"
        assert row["mode"] == "futures"

    def test_short_sl_tp_mirrored(self, tmp_path):
        tracker = make_tracker(tmp_path)
        profile = TradingProfile.load("BALANCED")
        sid = tracker.record_rejection(
            "ETH/USDT", "short", "actor", "low confidence",
            entry_price=100.0, atr=2.0, profile=profile, ctx=CTX,
        )
        row = get_row(tracker, sid)
        assert row["sl_price"] == pytest.approx(105.0)
        assert row["tp_price"] == pytest.approx(91.0)

    def test_sl_min_distance_clamp(self, tmp_path):
        """Tiny ATR → SL clamped to 1.5% min distance (ExecutionEngine parity)."""
        tracker = make_tracker(tmp_path)
        profile = TradingProfile.load("BALANCED")
        sid = tracker.record_rejection(
            "BTC/USDT", "long", "risk", "portfolio heat",
            entry_price=100.0, atr=0.1, profile=profile, ctx=CTX,  # 2.5*0.1=0.25 < 1.5
            )
        row = get_row(tracker, sid)
        assert row["sl_price"] == pytest.approx(98.5)

    def test_sl_max_distance_cap(self, tmp_path):
        """Huge ATR → SL capped at 25% from entry."""
        tracker = make_tracker(tmp_path)
        profile = TradingProfile.load("BALANCED")
        sid = tracker.record_rejection(
            "DOGE/USDT", "long", "risk", "heat",
            entry_price=100.0, atr=20.0, profile=profile, ctx=CTX,  # 2.5*20=50 > 25
        )
        row = get_row(tracker, sid)
        assert row["sl_price"] == pytest.approx(75.0)

    def test_context_persisted_and_created_at_aware(self, tmp_path):
        tracker = make_tracker(tmp_path)
        sid = tracker.record_rejection(
            "SOL/USDT", "long", "actor_prefilter", "conf below floor",
            entry_price=150.0, atr=3.0, profile=TradingProfile.load("BALANCED"), ctx=CTX,
        )
        row = get_row(tracker, sid)
        assert row["regime"] == "TRENDING_BULLISH"
        assert row["ensemble_score"] == pytest.approx(0.42)
        assert row["ob_imbalance"] == pytest.approx(1.8)
        assert row["btc_d"] == pytest.approx(54.2)
        assert row["gate"] == "actor_prefilter"
        created = datetime.fromisoformat(row["created_at"])
        assert created.tzinfo is not None


# ─── creation: guards ───────────────────────────────────────────────────────

class TestGuards:
    def test_bad_inputs_skipped(self, tmp_path):
        tracker = make_tracker(tmp_path)
        p = TradingProfile.load("BALANCED")
        assert tracker.record_rejection("X/USDT", "long", "risk", "r", 0.0, 1.0, p, CTX) is None
        assert tracker.record_rejection("X/USDT", "long", "risk", "r", 100.0, 0.0, p, CTX) is None
        assert tracker.record_rejection("X/USDT", "long", "risk", "r", -5.0, 1.0, p, CTX) is None

    def test_disabled_noop(self, tmp_path):
        tracker = make_tracker(tmp_path, enabled=False)
        sid = tracker.record_rejection(
            "ETH/USDT", "long", "risk", "r", 100.0, 2.0, TradingProfile.load("BALANCED"), CTX)
        assert sid is None

    def test_per_symbol_cooldown(self, tmp_path):
        """Same (symbol, side) within cooldown → deduped (kills 30s-rescan spam)."""
        tracker = make_tracker(tmp_path)
        p = TradingProfile.load("BALANCED")
        first = tracker.record_rejection("ETH/USDT", "long", "micro", "r", 100.0, 2.0, p, CTX)
        dup = tracker.record_rejection("ETH/USDT", "long", "actor", "r2", 101.0, 2.0, p, CTX)
        assert first is not None and dup is None
        # Opposite side is NOT deduped
        other = tracker.record_rejection("ETH/USDT", "short", "actor", "r", 100.0, 2.0, p, CTX)
        assert other is not None

    def test_max_open_cap(self, tmp_path):
        tracker = make_tracker(tmp_path, max_open=2, per_symbol_cooldown_min=0)
        p = TradingProfile.load("BALANCED")
        assert tracker.record_rejection("A/USDT", "long", "risk", "r", 100.0, 2.0, p, CTX)
        assert tracker.record_rejection("B/USDT", "long", "risk", "r", 100.0, 2.0, p, CTX)
        assert tracker.record_rejection("C/USDT", "long", "risk", "r", 100.0, 2.0, p, CTX) is None


# ─── resolution ─────────────────────────────────────────────────────────────

class TestResolution:
    def _shadow(self, tracker, side="long", entry=100.0, atr=2.0, symbol="ETH/USDT"):
        # BALANCED: long → SL 95 / TP 109; short → SL 105 / TP 91
        return tracker.record_rejection(
            symbol, side, "microstructure", "test",
            entry_price=entry, atr=atr, profile=TradingProfile.load("BALANCED"), ctx=CTX)

    def test_tp_first_long(self, tmp_path):
        tracker = make_tracker(tmp_path)
        sid = self._shadow(tracker)
        created = datetime.fromisoformat(get_row(tracker, sid)["created_at"])
        df = make_ohlcv(created + timedelta(minutes=15), [
            (102, 99, 101),
            (110, 100, 109),   # TP 109 hit, SL 95 never touched
        ])
        resolved = tracker.resolve_open(fetch_fn_for(df))
        assert len(resolved) == 1
        row = get_row(tracker, sid)
        assert row["status"] == "tp"
        assert row["outcome_r"] == pytest.approx(9.0 / 5.0)

    def test_sl_first_long(self, tmp_path):
        tracker = make_tracker(tmp_path)
        sid = self._shadow(tracker)
        created = datetime.fromisoformat(get_row(tracker, sid)["created_at"])
        df = make_ohlcv(created + timedelta(minutes=15), [
            (101, 94, 95),     # SL 95 hit
            (115, 100, 112),   # TP later is irrelevant
        ])
        tracker.resolve_open(fetch_fn_for(df))
        row = get_row(tracker, sid)
        assert row["status"] == "sl"
        assert row["outcome_r"] == pytest.approx(-1.0)

    def test_both_in_same_candle_counts_sl(self, tmp_path):
        """Conservative tie-break: candle spans both SL and TP → SL."""
        tracker = make_tracker(tmp_path)
        sid = self._shadow(tracker)
        created = datetime.fromisoformat(get_row(tracker, sid)["created_at"])
        df = make_ohlcv(created + timedelta(minutes=15), [
            (112, 93, 100),    # touches both 109 and 95
        ])
        tracker.resolve_open(fetch_fn_for(df))
        assert get_row(tracker, sid)["status"] == "sl"

    def test_candles_before_creation_ignored(self, tmp_path):
        """No lookahead: candles opening before created_at must not resolve."""
        tracker = make_tracker(tmp_path)
        sid = self._shadow(tracker)
        created = datetime.fromisoformat(get_row(tracker, sid)["created_at"])
        df = make_ohlcv(created - timedelta(hours=10), [
            (115, 90, 100),    # old candle spanning SL+TP — must be ignored
        ])
        tracker.resolve_open(fetch_fn_for(df))
        assert get_row(tracker, sid)["status"] == "open"

    def test_short_side_mirrored(self, tmp_path):
        tracker = make_tracker(tmp_path)
        sid = self._shadow(tracker, side="short")  # SL 105 / TP 91
        created = datetime.fromisoformat(get_row(tracker, sid)["created_at"])
        df = make_ohlcv(created + timedelta(minutes=15), [
            (102, 96, 98),
            (99, 90, 91),      # TP 91 hit
        ])
        tracker.resolve_open(fetch_fn_for(df))
        row = get_row(tracker, sid)
        assert row["status"] == "tp"
        assert row["outcome_r"] > 0

    def test_no_hit_stays_open(self, tmp_path):
        tracker = make_tracker(tmp_path)
        sid = self._shadow(tracker)
        created = datetime.fromisoformat(get_row(tracker, sid)["created_at"])
        df = make_ohlcv(created + timedelta(minutes=15), [
            (102, 98, 100), (103, 99, 101),
        ])
        assert tracker.resolve_open(fetch_fn_for(df)) == []
        assert get_row(tracker, sid)["status"] == "open"

    def test_expiry_mark_to_market(self, tmp_path):
        """>48h unresolved → expired with signed mark-to-market R."""
        tracker = make_tracker(tmp_path, max_age_hours=48)
        sid = self._shadow(tracker)  # long, entry 100, SL 95 (risk 5)
        # Backdate creation 50h
        old = (datetime.now(LOCAL_TZ) - timedelta(hours=50)).isoformat()
        with sqlite3.connect(tracker.db_path) as c:
            c.execute("UPDATE shadow_trades SET created_at=? WHERE id=?", (old, sid))
        created = datetime.fromisoformat(old)
        df = make_ohlcv(created + timedelta(minutes=15), [
            (103, 99, 102.5),  # never hits SL/TP; last close 102.5
        ])
        tracker.resolve_open(fetch_fn_for(df))
        row = get_row(tracker, sid)
        assert row["status"] == "expired"
        assert row["outcome_r"] == pytest.approx((102.5 - 100.0) / 5.0)
        assert row["resolve_price"] == pytest.approx(102.5)

    def test_raising_fetch_fn_is_isolated(self, tmp_path):
        tracker = make_tracker(tmp_path)
        self._shadow(tracker)

        def boom(symbol, timeframe="15m", limit=200):
            raise RuntimeError("exchange down")

        assert tracker.resolve_open(boom) == []  # no raise


# ─── stats ──────────────────────────────────────────────────────────────────

class TestGateStats:
    def test_aggregation_per_gate(self, tmp_path):
        tracker = make_tracker(tmp_path, per_symbol_cooldown_min=0)
        p = TradingProfile.load("BALANCED")
        now = datetime.now(LOCAL_TZ).isoformat()
        ids = [
            tracker.record_rejection(f"S{i}/USDT", "long", gate, "r", 100.0, 2.0, p, CTX)
            for i, gate in enumerate(["microstructure", "microstructure", "actor"])
        ]
        with sqlite3.connect(tracker.db_path) as c:
            c.execute("UPDATE shadow_trades SET status='tp', outcome_r=1.8, resolved_at=? WHERE id=?", (now, ids[0]))
            c.execute("UPDATE shadow_trades SET status='sl', outcome_r=-1.0, resolved_at=? WHERE id=?", (now, ids[1]))

        stats = tracker.gate_stats(days=30)
        micro = stats["microstructure"]
        assert micro["n"] == 2
        assert micro["tp"] == 1 and micro["sl"] == 1
        assert micro["win_rate"] == pytest.approx(0.5)
        assert micro["net_r"] == pytest.approx(0.8)
        assert stats["actor"]["n"] == 1
        assert stats["actor"]["resolved"] == 0

    def test_mode_isolation(self, tmp_path):
        db = tmp_path / "trade_memory.db"
        fut = ShadowTracker(db, "futures", DEFAULT_CFG)
        spot = ShadowTracker(db, "spot", DEFAULT_CFG)
        p = TradingProfile.load("BALANCED")
        fut.record_rejection("ETH/USDT", "long", "risk", "r", 100.0, 2.0, p, CTX)
        assert spot.gate_stats() == {}
        assert "risk" in fut.gate_stats()

    def test_load_stats_pure_read(self, tmp_path):
        tracker = make_tracker(tmp_path)
        tracker.record_rejection(
            "ETH/USDT", "long", "actor", "r", 100.0, 2.0, TradingProfile.load("BALANCED"), CTX)
        stats = load_stats(tmp_path / "trade_memory.db", "futures")
        assert stats["actor"]["n"] == 1

    def test_load_stats_missing_db(self, tmp_path):
        assert load_stats(tmp_path / "nope.db", "futures") == {}
