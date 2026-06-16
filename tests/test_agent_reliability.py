"""
Tests for the agent-reliability aggregator (agent_reliability Phase A).

Verifies the debiased per-(agent, regime) reliability math: an agent that
votes the correct direction earns a >1 weight multiplier, a wrong-voting agent
earns <1, both taken and rejected-shadow populations contribute, and thin
buckets stay near 1.0 (advisory-only).
"""

import os
import sqlite3
import sys
from datetime import datetime, timedelta

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bot"))

from bot.agents import agent_reliability as ar
from bot.core.tz import LOCAL_TZ


def _db(tmp_path):
    p = tmp_path / "tm.db"
    with sqlite3.connect(p) as c:
        c.execute("""CREATE TABLE trades (
            id TEXT, mode TEXT, side TEXT, regime TEXT, r_multiple REAL,
            closed_at TEXT, smc_score REAL, tech_score REAL, macro_score REAL)""")
        c.execute("""CREATE TABLE shadow_trades (
            id TEXT, mode TEXT, side TEXT, regime TEXT, outcome_r REAL,
            status TEXT, resolved_at TEXT,
            smc_score REAL, tech_score REAL, macro_score REAL)""")
    return str(p)


def _now_iso(hours_ago=1.0):
    return (datetime.now(LOCAL_TZ) - timedelta(hours=hours_ago)).isoformat()


def _add_trade(db, side, regime, r, smc, tech, macro, n=1):
    with sqlite3.connect(db) as c:
        for i in range(n):
            c.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?)",
                      (f"t{side}{regime}{r}{i}{smc}", "futures", side, regime, r,
                       _now_iso(), smc, tech, macro))


def _add_shadow(db, side, regime, r, smc, tech, macro, n=1):
    with sqlite3.connect(db) as c:
        for i in range(n):
            c.execute("INSERT INTO shadow_trades VALUES (?,?,?,?,?,?,?,?,?,?)",
                      (f"s{side}{regime}{r}{i}{smc}", "futures", side, regime, r,
                       "tp" if r > 0 else "sl", _now_iso(), smc, tech, macro))


def test_correct_agent_gets_higher_multiplier_than_wrong(tmp_path):
    db = _db(tmp_path)
    # 40 long winners. smc voted long (correct), macro voted short (wrong).
    _add_trade(db, "long", "RANGING", 1.5, smc=0.6, tech=0.0, macro=-0.6, n=40)
    rep = ar.compute(db, "futures", days=30)
    r = rep["regimes"]["RANGING"]
    assert r["smc"]["multiplier"] > 1.0
    assert r["macro_flow"]["multiplier"] < 1.0
    assert r["smc"]["accuracy"] == 1.0
    assert r["macro_flow"]["accuracy"] == 0.0


def test_shadow_population_contributes(tmp_path):
    db = _db(tmp_path)
    # No taken trades; only rejected shadows that WON as shorts. tech voted short.
    _add_shadow(db, "short", "WEAK_TREND", 2.0, smc=0.0, tech=-0.5, macro=0.0, n=40)
    rep = ar.compute(db, "futures", days=30)
    tech = rep["regimes"]["WEAK_TREND"]["technical"]
    assert tech["accuracy"] == 1.0
    assert tech["multiplier"] > 1.0
    assert rep["samples"] == 40


def test_thin_bucket_stays_near_one(tmp_path):
    db = _db(tmp_path)
    # Only 3 correct votes — shrink should keep the multiplier close to 1.0.
    _add_trade(db, "long", "CHOPPY", 1.0, smc=0.6, tech=0.0, macro=0.0, n=3)
    rep = ar.compute(db, "futures", days=30)
    smc = rep["regimes"]["CHOPPY"]["smc"]
    assert smc["actionable"] is False
    assert abs(smc["multiplier"] - 1.0) < 0.10


def test_abstaining_agent_excluded(tmp_path):
    db = _db(tmp_path)
    # tech votes ~0 (below VOTE_EPS) → no opinion → not scored at all.
    _add_trade(db, "long", "RANGING", 1.0, smc=0.6, tech=0.01, macro=0.0, n=20)
    rep = ar.compute(db, "futures", days=30)
    assert "technical" not in rep["regimes"]["RANGING"]
    assert "smc" in rep["regimes"]["RANGING"]


def test_multiplier_bounds_respected(tmp_path):
    db = _db(tmp_path)
    _add_trade(db, "long", "STRONG_TREND", 3.0, smc=0.9, tech=-0.9, macro=0.0, n=500)
    rep = ar.compute(db, "futures", days=30)
    r = rep["regimes"]["STRONG_TREND"]
    assert ar.MULT_LO <= r["smc"]["multiplier"] <= ar.MULT_HI
    assert ar.MULT_LO <= r["technical"]["multiplier"] <= ar.MULT_HI


def test_empty_db_is_safe(tmp_path):
    db = _db(tmp_path)
    rep = ar.compute(db, "futures", days=30)
    assert rep["samples"] == 0
    assert rep["regimes"] == {}
    assert "no agent-tagged outcomes yet" in ar.format_report(rep)
