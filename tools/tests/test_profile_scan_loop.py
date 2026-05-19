import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tools.profile_scan_loop import summarise_durations


def test_summarise_durations_basic():
    summary = summarise_durations([0.1, 0.2, 0.3, 0.4, 0.5])
    assert abs(summary["p50"] - 0.3) < 1e-6
    # p95 of 5 samples is the largest value
    assert abs(summary["p95"] - 0.5) < 1e-6
    assert abs(summary["min"] - 0.1) < 1e-6
    assert abs(summary["max"] - 0.5) < 1e-6
    assert summary["count"] == 5


def test_summarise_durations_empty():
    summary = summarise_durations([])
    assert summary["count"] == 0
    assert summary["p50"] is None
    assert summary["p95"] is None
