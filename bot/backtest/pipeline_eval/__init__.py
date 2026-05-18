"""End-to-end pipeline backtest harness (skeleton).

Filled in during Phase 6 (per the signal-quality overhaul spec). Phase 1
ships the CLI shape only.
"""
from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline_eval", description="End-to-end pipeline backtest.")
    p.add_argument("--run", action="store_true", help="Execute the backtest.")
    p.add_argument("--parquet", help="Path to training_dataset.parquet")
    p.add_argument("--config",  help="Path to config_<mode>.yaml")
    p.add_argument("--out",     help="Output JSON path")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if args.run:
        print("pipeline_eval: not implemented — scheduled for Phase 6 of the signal-quality overhaul",
              file=sys.stderr)
        return 2
    return 0
