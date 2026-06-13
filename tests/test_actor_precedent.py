"""
Tests for the counterfactual-aware Actor precedent (Phase 1) and the regime
feature-vector taxonomy fix (Phase 2).

Phase 1: TradeMemory.find_similar_precedent blends realized (executed) trades
with counterfactual (resolved shadow) trades so the Actor stops learning only
from setups it already approved — the selection-bias loop.
Phase 2: vector_store.feature_vector one-hots the LIVE regime names so regime
actually discriminates similarity.
"""

import sqlite3
import sys, os
from datetime import datetime, timedelta

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot.agents.trade_memory import TradeMemory, smoothed_stats
from bot.agents import vector_store as vs
from bot.core.tz import LOCAL_TZ


# ─── Phase 2: regime taxonomy ───────────────────────────────────────────────

class TestRegimeFeatureVector:
    def test_live_regimes_are_distinct(self):
        """CHOPPY / WEAK_TREND / STRONG_TREND must produce different vectors —
        before the fix they all collapsed to the all-zeros legacy bucket."""
        base = dict(side="long", ensemble_score=0.2, confidence=0.5, btc_d=55.0,
                    usdt_d=6.0, ob_imbalance=1.0, cvd_direction="bullish",
                    cvd_divergence=False)
        v_ch = vs.feature_vector(regime="CHOPPY", **base)
        v_wt = vs.feature_vector(regime="WEAK_TREND", **base)
        v_st = vs.feature_vector(regime="STRONG_TREND", **base)
        assert not np.array_equal(v_ch, v_wt)
        assert not np.array_equal(v_wt, v_st)
        # Same regime → identical
        assert np.array_equal(v_ch, vs.feature_vector(regime="CHOPPY", **base))

    def test_unknown_regime_is_all_zero_block(self):
        """An unlisted regime maps to the zero one-hot (no regime signal)."""
        base = dict(side="long", ensemble_score=0.2, confidence=0.5, btc_d=55.0,
                    usdt_d=6.0, ob_imbalance=1.0, cvd_direction="bullish",
                    cvd_divergence=False)
        v_unknown = vs.feature_vector(regime="NONSENSE_REGIME", **base)
        # The regime one-hot tail must be all zeros
        regime_block = v_unknown[-len(vs._REGIMES):]
        assert np.count_nonzero(regime_block) == 0


# ─── smoothed_stats ─────────────────────────────────────────────────────────

class TestSmoothedStats:
    def _rows(self, wins, r, n, age_h=1.0, sw=1.0):
        return [{"win": wins, "r": r, "age_h": age_h, "sw": sw} for _ in range(n)]

    def test_small_all_loss_sample_does_not_read_zero(self):
        """5 fresh losers must not read as 0% win / hard-negative avg_r —
        that was the freeze-loop anchor."""
        s = smoothed_stats(self._rows(False, -1.0, 5), half_life_h=24, prior=3.0)
        assert s["win_rate"] > 0.20      # shrunk toward 0.5, not 0%
        assert s["avg_r"] > -1.0         # shrunk toward 0, not the raw -1.0
        assert s["raw_wins"] == 0

    def test_avg_r_is_shrunk_toward_zero(self):
        """avg_r (not just win-rate) is shrunk — the bug the old prompt had."""
        small = smoothed_stats(self._rows(False, -2.0, 2), half_life_h=24, prior=3.0)
        large = smoothed_stats(self._rows(False, -2.0, 50), half_life_h=24, prior=3.0)
        assert small["avg_r"] > large["avg_r"]   # tiny sample shrinks more
        assert large["avg_r"] < -1.0             # large sample approaches raw

    def test_large_consistent_sample_approaches_raw(self):
        s = smoothed_stats(self._rows(True, 1.5, 100), half_life_h=24, prior=3.0)
        assert s["win_rate"] > 0.9
        assert s["avg_r"] > 1.2

    def test_source_weight_scales_influence(self):
        """A row's `sw` scales its weight in the blend."""
        mixed_full = smoothed_stats(
            self._rows(True, 1.0, 10, sw=1.0) + self._rows(False, -1.0, 10, sw=1.0),
            half_life_h=24, prior=1.0)
        mixed_discounted = smoothed_stats(
            self._rows(True, 1.0, 10, sw=1.0) + self._rows(False, -1.0, 10, sw=0.1),
            half_life_h=24, prior=1.0)
        # Down-weighting the losers lifts the blended win-rate
        assert mixed_discounted["win_rate"] > mixed_full["win_rate"]

    def test_empty(self):
        s = smoothed_stats([], half_life_h=24, prior=1.0)
        assert s == {"n": 0, "raw_wins": 0, "win_rate": 0.5, "avg_r": 0.0, "weight": 0.0}


# ─── find_similar_precedent (Phase 1 integration) ───────────────────────────

def _seed_db(path):
    """Build a DB whose REALIZED longs are all losers but whose COUNTERFACTUAL
    (shadow) longs are mostly winners — the exact selection-bias scenario."""
    tm = TradeMemory(path)  # creates the trades table
    now = datetime.now(LOCAL_TZ)
    with tm._conn() as c:
        # shadow_trades is created by ShadowTracker in production; create it here.
        c.execute("""CREATE TABLE IF NOT EXISTS shadow_trades (
            id TEXT PRIMARY KEY, mode TEXT, symbol TEXT, side TEXT, gate TEXT,
            reason TEXT, entry_price REAL, atr REAL, sl_price REAL, tp_price REAL,
            r_target REAL, regime TEXT, ensemble_score REAL, confidence REAL,
            ob_imbalance REAL, cvd_direction TEXT, cvd_divergence INTEGER,
            btc_d REAL, usdt_d REAL, created_at TEXT, status TEXT,
            resolved_at TEXT, resolve_price REAL, outcome_r REAL, redundant_gates TEXT)""")
        # 6 realized long LOSSES in CHOPPY
        for i in range(6):
            c.execute(
                "INSERT INTO trades (symbol,side,mode,pnl,r_multiple,regime,"
                "ensemble_score,confidence,ob_imbalance,cvd_direction,cvd_divergence,"
                "btc_d,usdt_d,closed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"L{i}/USDT","long","futures",-5.0,-0.8,"CHOPPY",0.15,0.45,1.0,
                 "bullish",0,55.0,6.0,(now - timedelta(hours=i)).isoformat()))
        # 30 counterfactual long shadows, 24 TP / 6 SL → 80% hypothetical win
        for i in range(30):
            status = "tp" if i < 24 else "sl"
            oc = 1.7 if status == "tp" else -1.0
            c.execute(
                "INSERT INTO shadow_trades (id,mode,symbol,side,gate,regime,"
                "ensemble_score,confidence,ob_imbalance,cvd_direction,cvd_divergence,"
                "btc_d,usdt_d,created_at,status,resolved_at,outcome_r) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (f"s{i}","futures",f"S{i}/USDT","long","actor","CHOPPY",0.15,0.45,1.0,
                 "bullish",0,55.0,6.0,(now-timedelta(hours=i)).isoformat(),status,
                 (now-timedelta(hours=i)).isoformat(),oc))
    return tm


def test_precedent_blends_in_counterfactual(tmp_path):
    tm = _seed_db(tmp_path / "tm.db")
    p = tm.find_similar_precedent(
        "BTC/USDT", "BUY", "CHOPPY", 0.15, 55.0, confidence=0.45, usdt_d=6.0,
        ob_imbalance=1.0, cvd_direction="bullish", cvd_divergence=False,
        half_life_h=24.0, prior=3.0, shadow_weight=0.7)
    # Realized is the biased all-loss sample; counterfactual is mostly winners.
    assert p["realized"]["n"] == 6 and p["realized"]["win_rate"] < 0.35
    assert p["counterfactual"]["n"] >= 20 and p["counterfactual"]["win_rate"] > 0.6
    # Blended must sit ABOVE the biased realized view (the whole point).
    assert p["blended"]["win_rate"] > p["realized"]["win_rate"]
    assert p["blended"]["avg_r"] > 0  # counterfactual winners pull avg_r positive


def test_precedent_maps_shadow_status_to_win(tmp_path):
    """status='tp' must count as a win, status='sl' as a loss (shadows have no
    pnl column — the bug the prototype caught)."""
    tm = _seed_db(tmp_path / "tm.db")
    p = tm.find_similar_precedent(
        "BTC/USDT", "BUY", "CHOPPY", 0.15, 55.0, confidence=0.45, usdt_d=6.0,
        ob_imbalance=1.0, cvd_direction="bullish", cvd_divergence=False)
    # 24/30 tp → raw_wins on the counterfactual side must be > 0
    assert p["counterfactual"]["raw_wins"] > 0
    srcs = {e["source"] for e in p["examples"]}
    assert "counterfactual" in srcs and "realized" in srcs   # examples are a mix


def test_precedent_survives_missing_shadow_table(tmp_path):
    """No shadow_trades table → realized precedent still returned, counterfactual zero."""
    tm = TradeMemory(tmp_path / "tm.db")  # trades only, no shadow_trades
    now = datetime.now(LOCAL_TZ)
    with tm._conn() as c:
        c.execute("INSERT INTO trades (symbol,side,mode,pnl,r_multiple,regime,"
                  "ensemble_score,confidence,ob_imbalance,cvd_direction,cvd_divergence,"
                  "btc_d,usdt_d,closed_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  ("X/USDT","long","futures",10.0,1.2,"CHOPPY",0.15,0.5,1.0,
                   "bullish",0,55.0,6.0,now.isoformat()))
    p = tm.find_similar_precedent("BTC/USDT","BUY","CHOPPY",0.15,55.0,
                                  confidence=0.5,cvd_direction="bullish")
    assert p["realized"]["n"] == 1
    assert p["counterfactual"]["n"] == 0   # gracefully empty, no crash
