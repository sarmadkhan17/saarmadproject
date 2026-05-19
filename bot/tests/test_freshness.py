import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import pytest
from data.freshness import Freshness, CacheMiss


def test_set_then_fresh_get_returns_value():
    f = Freshness()
    f.set("k", "v")
    assert f.get("k", max_age_seconds=10) == "v"


def test_get_returns_none_when_unset():
    f = Freshness()
    assert f.get("k", max_age_seconds=10) is None


def test_get_returns_none_when_stale():
    f = Freshness()
    f.set("k", "v")
    # Backdate the timestamp deterministically.
    key_meta = f._times["k"]
    f._times["k"] = key_meta - 100
    assert f.get("k", max_age_seconds=10) is None


def test_age_seconds_reports_correctly():
    f = Freshness()
    f.set("k", "v")
    f._times["k"] = f._times["k"] - 7
    assert 6 <= f.age_seconds("k") <= 8


def test_age_seconds_returns_none_for_missing_key():
    f = Freshness()
    assert f.age_seconds("nope") is None


def test_fetch_calls_loader_on_miss():
    f = Freshness()
    calls = {"n": 0}
    def loader():
        calls["n"] += 1
        return "loaded"
    v = f.fetch("k", max_age_seconds=10, loader=loader)
    assert v == "loaded"
    assert calls["n"] == 1
    # second call within ttl uses cache
    v2 = f.fetch("k", max_age_seconds=10, loader=loader)
    assert v2 == "loaded"
    assert calls["n"] == 1


def test_fetch_recalls_loader_when_stale():
    f = Freshness()
    calls = {"n": 0}
    def loader():
        calls["n"] += 1
        return calls["n"]
    f.fetch("k", max_age_seconds=10, loader=loader)
    f._times["k"] -= 100
    v = f.fetch("k", max_age_seconds=10, loader=loader)
    assert v == 2
    assert calls["n"] == 2


def test_fetch_propagates_loader_exception():
    f = Freshness()
    def loader():
        raise CacheMiss("upstream failure")
    with pytest.raises(CacheMiss):
        f.fetch("k", max_age_seconds=10, loader=loader)


def test_invalidate_removes_key():
    f = Freshness()
    f.set("k", "v")
    f.invalidate("k")
    assert f.get("k", max_age_seconds=10) is None
