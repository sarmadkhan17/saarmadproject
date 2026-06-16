"""
Tests for agent_reliability Phase B — adaptive ensemble weights.

The reliability multiplier must: (a) no-op when the flag is off, (b) re-balance
relative agent influence when on, (c) ignore thin/non-actionable buckets, and
(d) survive an absent/garbage JSON without disturbing the legacy behaviour.
"""

import json
import os
import sys

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bot"))

from engine.ensemble import EnsembleEngine, _ReliabilityWeights
from engine.smc_agent import AgentSignal


class _StubAgent:
    """Returns a fixed AgentSignal so the ensemble math is deterministic."""
    def __init__(self, name, net):
        self._name, self._net = name, net

    def analyze(self, df, profile, *a, **k):
        return AgentSignal(self._name, max(self._net, 0.0), max(-self._net, 0.0),
                           self._net, 0.6, reasoning="stub")


def _engine(adaptive=False, path=None):
    # smc votes BUY, technical votes SELL — opposite so re-weighting changes net.
    return EnsembleEngine(
        _StubAgent("smc", 0.6), _StubAgent("technical", -0.6),
        macro_agent=None, adaptive_weights=adaptive, reliability_path=path,
    )


class _Profile:
    net_score_threshold = 0.05


def _write_json(path, regime, agent, mult, actionable=True):
    path.write_text(json.dumps({
        "regimes": {regime: {agent: {
            "multiplier": mult, "actionable": actionable,
            "n": 50, "accuracy": 0.7, "edge": 0.2,
        }}}
    }))


def test_flag_off_ignores_json(tmp_path):
    p = tmp_path / "rel.json"
    _write_json(p, "RANGING", "smc", 1.4)
    eng = _engine(adaptive=False, path=str(p))
    assert eng._rel is None  # loader not even constructed when off


def test_actionable_multiplier_shifts_net(tmp_path):
    dfs = {"1h": "df"}
    base = _engine(adaptive=False).run("X", dfs, _Profile(),
                                       market_ctx={"regime": "RANGING"}).net_score
    p = tmp_path / "rel.json"
    _write_json(p, "RANGING", "smc", 1.4)          # boost the BUY-voting agent
    boosted = _engine(adaptive=True, path=str(p)).run(
        "X", dfs, _Profile(), market_ctx={"regime": "RANGING"}).net_score
    # smc (BUY) gains weight vs technical (SELL) → net moves up (less negative).
    assert boosted > base


def test_thin_bucket_is_ignored(tmp_path):
    dfs = {"1h": "df"}
    base = _engine(adaptive=False).run("X", dfs, _Profile(),
                                       market_ctx={"regime": "RANGING"}).net_score
    p = tmp_path / "rel.json"
    _write_json(p, "RANGING", "smc", 1.4, actionable=False)  # not enough samples
    same = _engine(adaptive=True, path=str(p)).run(
        "X", dfs, _Profile(), market_ctx={"regime": "RANGING"}).net_score
    assert same == base


def test_missing_file_is_safe(tmp_path):
    dfs = {"1h": "df"}
    base = _engine(adaptive=False).run("X", dfs, _Profile(),
                                       market_ctx={"regime": "RANGING"}).net_score
    missing = _engine(adaptive=True, path=str(tmp_path / "nope.json")).run(
        "X", dfs, _Profile(), market_ctx={"regime": "RANGING"}).net_score
    assert missing == base


def test_loader_reloads_on_mtime(tmp_path):
    p = tmp_path / "rel.json"
    _write_json(p, "RANGING", "smc", 1.3)
    rel = _ReliabilityWeights(str(p))
    assert rel.mult("smc", "RANGING") == pytest.approx(1.3)
    _write_json(p, "RANGING", "smc", 0.7)
    os.utime(p, None)  # bump mtime
    # mtime change forces a reload on next access.
    assert rel.mult("smc", "RANGING") == pytest.approx(0.7)
    # Unknown agent/regime is always neutral.
    assert rel.mult("macro_flow", "RANGING") == 1.0
    assert rel.mult("smc", "CRASH") == 1.0
