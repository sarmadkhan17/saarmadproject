#!/usr/bin/env python3
"""
Agent-reliability report — agent_reliability Phase A (advisory).

Reads data/trade_memory.db (taken + resolved-shadow outcomes, both carrying the
per-agent vote captured at entry/rejection), computes each agent's debiased
per-regime reliability, prints it, and writes data/agent_reliability_<mode>.json.

ADVISORY ONLY — nothing here changes a live trading decision. The live ensemble
does not read the JSON until Phase B (config flag ensemble.adaptive_weights).

Usage:
    python scripts/agent_reliability.py                 # both modes, 30d
    python scripts/agent_reliability.py --mode futures --days 14
    python scripts/agent_reliability.py --no-write      # print only
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.agents import agent_reliability as ar

DATA = ROOT / "data"
DB   = DATA / "trade_memory.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["spot", "futures"], default=None,
                    help="default: both")
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--no-write", action="store_true",
                    help="print only, do not write the advisory JSON")
    args = ap.parse_args()

    modes = [args.mode] if args.mode else ["futures", "spot"]
    for mode in modes:
        report = ar.compute(str(DB), mode, days=args.days)
        print(ar.format_report(report))
        if not args.no_write:
            path = ar.write_report(report, DATA)
            print(f"  → wrote {path}")
        print()


if __name__ == "__main__":
    main()
