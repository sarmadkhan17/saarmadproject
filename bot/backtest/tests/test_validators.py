import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backtest.validators import (
    rally_window_admits_longs, monotonic_decline_vetoes_longs,
)


def _rec(symbol, ts, long_allowed, short_allowed=False,
         fast="flat", slow="flat", strong=False):
    return {"symbol": symbol, "ts": ts,
            "long_allowed": long_allowed, "short_allowed": short_allowed,
            "fast_direction": fast, "slow_direction": slow, "slow_strong": strong}


def test_rally_window_admits_longs_passes_with_evidence():
    # Synthesise: window 2026-04-01T00..2026-04-02T00, 4 long admits, 8 symbols
    # rallying (simulated via aux "rally_evidence" hook is not used here —
    # the validator works on the per-eval long_allowed records alone).
    records = []
    for h in range(24):
        ts = f"2026-04-01T{h:02d}:00:00+00:00"
        records.append(_rec("BTCUSDT", ts, True))
    result = rally_window_admits_longs(records, min_long_admits=4)
    assert result["passed"] is True
    assert result["details"]["max_admits_in_24h_window"] >= 4


def test_rally_window_admits_longs_fails_when_no_admits():
    records = [_rec("BTCUSDT", f"2026-04-01T{h:02d}:00:00+00:00", False) for h in range(24)]
    result = rally_window_admits_longs(records, min_long_admits=4)
    assert result["passed"] is False


def test_monotonic_decline_vetoes_longs_passes():
    records = [_rec("BTCUSDT", f"2026-04-01T{h:02d}:00:00+00:00", False,
                    fast="down", slow="down", strong=True) for h in range(24)]
    result = monotonic_decline_vetoes_longs(records)
    assert result["passed"] is True

def test_monotonic_decline_vetoes_longs_fails_when_long_slips_in():
    records = [_rec("BTCUSDT", f"2026-04-01T{h:02d}:00:00+00:00", False) for h in range(24)]
    # Inject one offending long_allowed=True
    records[10]["long_allowed"] = True
    result = monotonic_decline_vetoes_longs(records)
    assert result["passed"] is False
