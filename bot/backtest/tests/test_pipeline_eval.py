import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from backtest.pipeline_eval import (
    load_symbol_dfs, walk_forward_iter, run_components, PipelineComponent
)


def _make_parquet(tmp: Path, symbol: str, tf: str, n: int, slope: float = 0.0):
    idx = pd.date_range("2026-01-01", periods=n, freq=tf.replace("m", "min"), tz="UTC")
    closes = 100 + np.arange(n) * slope
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes,
                        "close": closes, "volume": np.ones(n)}, index=idx)
    df.index.name = "timestamp"
    out = tmp / f"{symbol}USDT_{tf}.parquet"
    df.to_parquet(out)
    return out


def test_load_symbol_dfs_reads_both_tfs(tmp_path):
    _make_parquet(tmp_path, "BTC", "15m", 300, 0.05)
    _make_parquet(tmp_path, "BTC", "1h",  100, 0.1)
    dfs = load_symbol_dfs(tmp_path, "BTCUSDT", ["15m", "1h"])
    assert "15m" in dfs and "1h" in dfs
    assert len(dfs["15m"]) == 300
    assert len(dfs["1h"]) == 100


def test_walk_forward_iter_emits_truncated_slices(tmp_path):
    _make_parquet(tmp_path, "BTC", "15m", 100, 0.05)
    _make_parquet(tmp_path, "BTC", "1h",   25, 0.1)
    dfs = load_symbol_dfs(tmp_path, "BTCUSDT", ["15m", "1h"])
    points = list(walk_forward_iter(dfs, cadence="1h", base_tf="1h"))
    # Each yielded item is (ts, sliced_dfs); 1h has 25 bars → up to 25 iterations
    assert len(points) == 25
    ts0, dfs0 = points[0]
    assert dfs0["1h"].index[-1] == ts0
    # No look-ahead: 15m slice does not contain bars beyond ts0
    assert dfs0["15m"].index.max() <= ts0


class _CountingComponent:
    name = "counter"
    def __init__(self):
        self.calls = 0
    def evaluate(self, symbol, ts, dfs):
        self.calls += 1
        return {"symbol": symbol, "ts": ts.isoformat()}
    def validators(self):
        return []


def test_run_components_calls_each_point(tmp_path):
    _make_parquet(tmp_path, "BTC", "15m", 100, 0.05)
    _make_parquet(tmp_path, "BTC", "1h",   25, 0.1)
    c = _CountingComponent()
    out_path = tmp_path / "out.json"
    summary = run_components(
        parquet_dir=tmp_path,
        symbols=["BTCUSDT"],
        tfs=["15m", "1h"],
        cadence="1h",
        base_tf="1h",
        components=[c],
        out_path=out_path,
    )
    assert c.calls == 25
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "summary" in data
    assert "validators" in data
    assert data["coverage"]["n_evaluations"] == 25
