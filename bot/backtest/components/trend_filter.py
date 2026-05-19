"""TrendFilter replay component for bot.backtest.pipeline_eval.

Wraps engine.trend_filter.TrendFilter so it can be invoked from the harness's
walk-forward iterator. Each evaluation records the verdict (long_allowed,
short_allowed, fast/slow direction, slow.strong) for later aggregation by
the validators in bot/backtest/validators.py.
"""
from __future__ import annotations

import pandas as pd
from typing import Callable, List

from engine.trend_filter import TrendFilter


class TrendFilterReplayComponent:
    name = "trend_filter"

    def __init__(self, cfg: dict) -> None:
        self._tf = TrendFilter(cfg)

    def evaluate(self, symbol: str, ts: pd.Timestamp, dfs: dict) -> dict:
        v = self._tf.check(dfs)
        return {
            "ts":             ts.isoformat(),
            "long_allowed":   bool(v["long_allowed"]),
            "short_allowed":  bool(v["short_allowed"]),
            "fast_direction": v["fast"]["direction"],
            "slow_direction": v["slow"]["direction"],
            "slow_strong":    bool(v["slow"].get("strong", False)),
        }

    def validators(self) -> List[Callable]:
        # Wired up in Task 7.
        return []
