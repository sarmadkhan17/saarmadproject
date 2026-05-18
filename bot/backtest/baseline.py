"""Capture pre-overhaul live baseline metrics from a state file.

Used once at the start of the signal-quality overhaul so every later phase
can compare against an honest 'this is what the bot was doing before'
snapshot. Reads a state.json / futures_state.json shape and writes a small
JSON summary.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Union


def capture_baseline(state_path: Union[str, Path], out_path: Union[str, Path]) -> dict:
    state_path = Path(state_path)
    out_path = Path(out_path)
    with open(state_path) as fh:
        state = json.load(fh)

    trades = state.get("trades", [])
    closed = [t for t in trades if t.get("status") == "closed"]
    wins = [t for t in closed if (t.get("pnl") or 0) > 0]
    losses = [t for t in closed if (t.get("pnl") or 0) <= 0]
    side_mix = Counter(t.get("side", "?") for t in closed)
    durations = [float(t.get("duration_hours", 0)) for t in closed]

    summary = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "source_state": str(state_path),
        "closed_count": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(closed)) if closed else 0.0,
        "net_pnl": sum((t.get("pnl") or 0) for t in closed),
        "avg_duration_hours": (sum(durations) / len(durations)) if durations else 0.0,
        "side_mix": dict(side_mix),
        "stats_balance": state.get("stats", {}).get("balance"),
        "stats_total_pnl": state.get("stats", {}).get("total_pnl"),
    }

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as fh:
        json.dump(summary, fh, indent=2)
    os.replace(tmp, out_path)
    return summary


def _main(argv: list[str]) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Capture pre-overhaul baseline metrics.")
    parser.add_argument("--state", required=True, help="Path to state.json or futures_state.json")
    parser.add_argument("--out",   required=True, help="Path to write baseline summary JSON")
    args = parser.parse_args(argv)
    summary = capture_baseline(args.state, args.out)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    import sys as _sys
    raise SystemExit(_main(_sys.argv[1:]))
