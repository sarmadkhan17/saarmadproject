"""End-to-end pipeline backtest harness.

Plugin architecture: components implement evaluate(symbol, ts, dfs) and
optional validators(). The harness walks the per-symbol per-tf parquets
forward in time, slices each tf up to the current timestamp (no look-ahead),
and invokes every registered component at each evaluation point.

Used by Phase 2+ to validate components against 3 years × 8 training coins.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Protocol, Sequence

import pandas as pd


class PipelineComponent(Protocol):
    name: str
    def evaluate(self, symbol: str, ts: pd.Timestamp,
                 dfs: dict) -> dict: ...
    def validators(self) -> list: ...


def load_symbol_dfs(parquet_dir: Path, symbol: str,
                    tfs: Sequence[str]) -> dict:
    """Load OHLCV parquet for each requested tf for a single symbol.

    File layout: <parquet_dir>/<SYMBOL>_<tf>.parquet (e.g. BTCUSDT_15m.parquet).
    """
    dfs = {}
    for tf in tfs:
        p = Path(parquet_dir) / f"{symbol}_{tf}.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        dfs[tf] = df
    return dfs


def walk_forward_iter(dfs: dict, cadence: str, base_tf: str):
    """Yield (ts, sliced_dfs) for each cadence point in the base_tf history.

    sliced_dfs[tf] is df[:ts] (closed-right), so no look-ahead is possible.
    """
    base = dfs.get(base_tf)
    if base is None or len(base) == 0:
        return
    # Resample base to cadence to get evaluation timestamps.
    cadence_pd = cadence.replace("m", "min") if cadence.endswith("m") and cadence != "1m" else cadence
    stamps = base.resample(cadence_pd).last().dropna(how="all").index
    for ts in stamps:
        sliced = {tf: df.loc[:ts] for tf, df in dfs.items()}
        yield ts, sliced


def run_components(parquet_dir: Path, symbols: Sequence[str], tfs: Sequence[str],
                   cadence: str, base_tf: str, components: Sequence[PipelineComponent],
                   out_path: Path) -> dict:
    """Run all components over all (symbol, ts) points and write output JSON."""
    results: dict = {c.name: [] for c in components}
    coverage_symbols: list = []
    n_evaluations = 0
    start_ts: Optional[pd.Timestamp] = None
    end_ts:   Optional[pd.Timestamp] = None

    for symbol in symbols:
        dfs = load_symbol_dfs(Path(parquet_dir), symbol, tfs)
        if not dfs:
            continue
        coverage_symbols.append(symbol)
        for ts, sdfs in walk_forward_iter(dfs, cadence=cadence, base_tf=base_tf):
            if start_ts is None or ts < start_ts: start_ts = ts
            if end_ts is None   or ts > end_ts:   end_ts = ts
            for c in components:
                rec = c.evaluate(symbol, ts, sdfs)
                results[c.name].append({"symbol": symbol, **rec})
            n_evaluations += 1

    summary = {c.name: _summarise(results[c.name]) for c in components}
    validators_out: dict = {}
    for c in components:
        for v in (c.validators() or []):
            vres = v(results[c.name])
            validators_out[vres["name"]] = vres

    out = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "components":  [c.name for c in components],
        "coverage": {
            "symbols": coverage_symbols,
            "start":   start_ts.isoformat() if start_ts is not None else None,
            "end":     end_ts.isoformat()   if end_ts   is not None else None,
            "cadence": cadence,
            "n_evaluations": n_evaluations,
        },
        "summary":    summary,
        "validators": validators_out,
    }
    tmp = Path(out_path).with_suffix(Path(out_path).suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    os.replace(tmp, out_path)
    return out


def _summarise(records: list) -> dict:
    if not records:
        return {"count": 0}
    out: dict = {"count": len(records)}
    # Per-component summarisation is component-specific; the harness only
    # contributes the count. Components may post-process out["summary"]
    # in their own way by inspecting results before the file is written
    # (Phase 3+ may want richer roll-ups).
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pipeline_eval",
                                description="End-to-end pipeline backtest harness.")
    p.add_argument("--run", action="store_true", help="Execute the backtest.")
    p.add_argument("--parquet", default="data/training",
                   help="Directory containing <SYMBOL>USDT_<tf>.parquet files")
    p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,SOLUSDT,DOGEUSDT,ADAUSDT,LINKUSDT")
    p.add_argument("--tfs", default="15m,1h")
    p.add_argument("--cadence", default="1h")
    p.add_argument("--base-tf", default="1h")
    p.add_argument("--component", default="trend_filter",
                   choices=["trend_filter"])
    p.add_argument("--out", default="data/baselines/p3_replay.json")
    p.add_argument("--config", default="config_futures.yaml")
    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    if not args.run:
        return 0

    # Component registration is wave-by-wave; Phase 2 registers trend_filter.
    if args.component == "trend_filter":
        from .components.trend_filter import TrendFilterReplayComponent
        import yaml
        cfg = yaml.safe_load(open(args.config))["trend_filter"]
        # Replay uses the two-tier branch regardless of live flag (it's a
        # what-if check); pass the two-tier sub-config to TrendFilter.
        replay_cfg = {
            "fast": cfg["fast"], "slow": cfg["slow"],
            "strong_slope_pct": cfg["strong_slope_pct"],
        }
        components = [TrendFilterReplayComponent(replay_cfg)]
    else:
        print(f"unknown component: {args.component}", file=sys.stderr)
        return 2

    run_components(
        parquet_dir=Path(args.parquet),
        symbols=args.symbols.split(","),
        tfs=args.tfs.split(","),
        cadence=args.cadence,
        base_tf=args.base_tf,
        components=components,
        out_path=Path(args.out),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
