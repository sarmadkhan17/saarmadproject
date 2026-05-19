"""Acceptance-criteria validators for pipeline_eval replay outputs.

Each validator takes a list of per-evaluation records (from a single
component) and returns:

    {"name": str, "passed": bool, "details": dict}

Validators are pluggable per component (see PipelineComponent.validators()).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


def rally_window_admits_longs(records: list, min_long_admits: int = 4) -> dict:
    """Spec §4.3.1: on any 24h window, the new filter must admit ≥4 long entries.

    Implementation: slide a 24h window over all records sorted by timestamp;
    count long_allowed=True within each window; report the max.
    """
    rows = sorted(records, key=lambda r: r["ts"])
    if not rows:
        return {"name": "rally_window_admits_longs", "passed": False,
                "details": {"reason": "no records"}}
    # Pre-extract timestamps and long_allowed flags as parallel arrays
    ts_arr   = [_parse(r["ts"]) for r in rows]
    flag_arr = [bool(r["long_allowed"]) for r in rows]
    window = timedelta(hours=24)
    best = 0
    best_start = None
    left = 0
    running = 0
    for right in range(len(ts_arr)):
        if flag_arr[right]:
            running += 1
        while ts_arr[right] - ts_arr[left] > window:
            if flag_arr[left]:
                running -= 1
            left += 1
        if running > best:
            best = running
            best_start = ts_arr[left]
    passed = best >= min_long_admits
    return {
        "name": "rally_window_admits_longs",
        "passed": passed,
        "details": {
            "max_admits_in_24h_window": best,
            "window_start": best_start.isoformat() if best_start else None,
            "min_required": min_long_admits,
        },
    }


def monotonic_decline_vetoes_longs(records: list) -> dict:
    """Spec §4.3.2: on a monotonic-decline window, longs must remain vetoed.

    Implementation: assert that no record in the replay set has
    long_allowed=True. Any long admission is considered a veto failure.
    Offending records are capped at 5 in the details output.
    """
    offending: list = []
    for r in records:
        if r["long_allowed"]:
            offending.append({"symbol": r["symbol"], "ts": r["ts"],
                               "long_allowed": True})
            if len(offending) >= 5:
                break
    passed = len(offending) == 0
    return {
        "name": "monotonic_decline_vetoes_longs",
        "passed": passed,
        "details": {"offending": offending},
    }
