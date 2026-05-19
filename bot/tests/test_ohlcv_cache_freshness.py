import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
from data.feed import OHLCVCache


def test_needs_refresh_when_unset():
    c = OHLCVCache()
    assert c.needs_refresh("BTC/USDT", "1h") is True


def test_needs_refresh_false_when_fresh_under_max_age():
    c = OHLCVCache()
    c.set("BTC/USDT", "1h", "df-marker")
    assert c.needs_refresh("BTC/USDT", "1h", max_age_seconds=300) is False


def test_needs_refresh_true_when_older_than_max_age():
    c = OHLCVCache()
    c.set("BTC/USDT", "1h", "df-marker")
    key = c._key("BTC/USDT", "1h")
    c._fresh._times[key] -= 500  # backdate via underlying freshness store
    assert c.needs_refresh("BTC/USDT", "1h", max_age_seconds=300) is True


def test_get_returns_value_within_default_tier_ttl():
    c = OHLCVCache()
    c.set("BTC/USDT", "1h", "df-marker")
    # Default per-tf ceiling still applies when max_age not given.
    assert c.get("BTC/USDT", "1h") == "df-marker"


def test_get_returns_none_when_caller_demands_fresher_than_we_have():
    c = OHLCVCache()
    c.set("BTC/USDT", "1h", "df-marker")
    key = c._key("BTC/USDT", "1h")
    c._fresh._times[key] -= 120
    # We have 120s-old data; caller demands < 60s.
    assert c.get("BTC/USDT", "1h", max_age_seconds=60) is None
