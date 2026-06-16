#!/usr/bin/env python3
"""
Post-exit excursion report — reads the post_exit_tracks table (taken + shadow
exits watched for watch_hours after they closed) and answers two questions:

  Q3  "let winners run until where?" — how far the move CONTINUED past our exit,
      especially for TP_BACKSTOP exits capped at +3.5R (right-censored before).
  Q2  "genuine stop vs noise?" — of stopped/losing exits, how many later reached
      the original TP (recovered → the stop was premature/noise).

READ-ONLY. Nothing here changes a live decision; it just summarises observations.

Usage:
    python3 scripts/post_exit_report.py                 # both modes, 30d
    python3 scripts/post_exit_report.py --mode futures --days 14
"""

import argparse
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bot.agents.shadow_tracker import ShadowTracker  # noqa: E402

DB = ROOT / "data" / "trade_memory.db"


def _bucket_mfe(rows):
    b = {"<0": 0, "0-0.5R": 0, "0.5-1R": 0, "1-2R": 0, "2-3.5R": 0, ">=3.5R": 0}
    for r in rows:
        x = r.get("post_mfe_r")
        if x is None:
            continue
        if x < 0:           b["<0"] += 1
        elif x < 0.5:       b["0-0.5R"] += 1
        elif x < 1:         b["0.5-1R"] += 1
        elif x < 2:         b["1-2R"] += 1
        elif x < 3.5:       b["2-3.5R"] += 1
        else:               b[">=3.5R"] += 1
    return b


def report(mode: str, days: int):
    rows = ShadowTracker(DB, mode).post_exit_rows(days=days)
    print(f"Post-exit tracks [{mode}] — {len(rows)} finalized, last {days}d "
          f"(READ-ONLY instrument):")
    if not rows:
        print("  (no finalized tracks yet — watch window is 8h by default)\n")
        return

    for src in ("taken", "shadow"):
        srows = [r for r in rows if r.get("source") == src]
        if not srows:
            continue
        print(f"  ── {src} ({len(srows)}) ──")

        # Q3 — continuation past exit
        mfe = [r["post_mfe_r"] for r in srows if r.get("post_mfe_r") is not None]
        if mfe:
            print(f"    continuation past exit (post_mfe_r): "
                  f"median {st.median(mfe):+.2f}R  mean {st.mean(mfe):+.2f}R")
            b = _bucket_mfe(srows)
            print("      " + "  ".join(f"{k}:{v}" for k, v in b.items()))

        # Q3 — backstop censoring: how far did capped winners actually run on?
        cap = [r for r in srows if "BACKSTOP" in (r.get("exit_reason") or "")]
        capmfe = [r["post_mfe_r"] for r in cap if r.get("post_mfe_r") is not None]
        if capmfe:
            ran_on = sum(1 for x in capmfe if x > 0.2)
            print(f"    TP_BACKSTOP exits: {len(capmfe)} | continued >0.2R further: "
                  f"{ran_on}/{len(capmfe)} | median further {st.median(capmfe):+.2f}R "
                  f"(money left on the table if large)")

        # Q2 — stop quality: noise vs genuine
        stops = [r for r in srows if r.get("recovered_to_tp") is not None]
        if stops:
            noise = sum(1 for r in stops if r["recovered_to_tp"] == 1)
            cont = [r["continued_r"] for r in stops if r.get("continued_r") is not None]
            print(f"    losing/stopped exits: {len(stops)} | later reached TP (noise): "
                  f"{noise}/{len(stops)} ({noise/len(stops):.0%}) | "
                  f"median continued {st.median(cont):+.2f}R" if cont else "")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["spot", "futures"], default=None)
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()
    for mode in ([args.mode] if args.mode else ["futures", "spot"]):
        report(mode, args.days)


if __name__ == "__main__":
    main()
