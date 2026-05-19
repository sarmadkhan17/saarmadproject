"""Measure scan-loop p50 / p95 / max latency by tailing logs/<mode>_bot.log.

Reads the existing 'HMM regime: ...' heartbeat lines (one per scan cycle)
and computes the inter-arrival time. That is the scan-loop period and a
good proxy for total per-cycle cost. Writes a JSON summary to stdout.

Used to enforce the latency budget gate (spec §6 P6d).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")


def summarise_durations(durations: list[float]) -> dict:
    if not durations:
        return {"count": 0, "p50": None, "p95": None, "min": None, "max": None,
                "mean": None}
    sorted_d = sorted(durations)
    def _pct(p: float) -> float:
        if len(sorted_d) == 1:
            return sorted_d[0]
        rank = int(round((p / 100.0) * (len(sorted_d) - 1)))
        return sorted_d[rank]
    return {
        "count": len(sorted_d),
        "p50": _pct(50),
        "p95": _pct(95),
        "min": sorted_d[0],
        "max": sorted_d[-1],
        "mean": statistics.fmean(sorted_d),
    }


def _parse_ts(line: str) -> Optional[datetime]:
    m = TS_RE.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group(1)}.{m.group(2)}", "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None


def measure_log(log_path: Path, marker: str = "HMM regime", limit: int = 500) -> dict:
    timestamps: list[datetime] = []
    with open(log_path) as fh:
        for line in fh:
            if marker not in line:
                continue
            ts = _parse_ts(line)
            if ts is not None:
                timestamps.append(ts)
                if len(timestamps) > limit:
                    timestamps.pop(0)
    durations = [
        (timestamps[i] - timestamps[i - 1]).total_seconds()
        for i in range(1, len(timestamps))
    ]
    return summarise_durations(durations)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="profile_scan_loop",
                                description="Measure scan-loop latency from bot log.")
    p.add_argument("--log", required=True, help="Path to <mode>_bot.log")
    p.add_argument("--marker", default="HMM regime",
                   help="Per-scan marker substring (default: 'HMM regime')")
    p.add_argument("--limit", type=int, default=500,
                   help="How many recent markers to use")
    args = p.parse_args(argv)
    summary = measure_log(Path(args.log), args.marker, args.limit)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
