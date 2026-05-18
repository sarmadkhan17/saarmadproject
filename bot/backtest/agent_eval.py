"""Per-agent walk-forward backtest harness (skeleton).

Filled in during Phase 4 (per the signal-quality overhaul spec). Phase 1
ships the CLI shape only so dependent tooling can wire against it.
"""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agent_eval", description="Per-agent backtest harness.")
    p.add_argument("--run", action="store_true", help="Execute the backtest.")
    p.add_argument("--parquet", help="Path to training_dataset.parquet")
    p.add_argument("--agent",   help="Agent name (smc | technical | macro | all)")
    p.add_argument("--out",     help="Output JSON path")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.run:
        print("agent_eval: not implemented — scheduled for Phase 4 of the signal-quality overhaul",
              file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
