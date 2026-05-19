"""Two-tier slope-magnitude trend filter (Phase 2 / P3).

Pure-function check over OHLCV dataframes for two timeframes (fast + slow).
Decides whether long / short entries are admissible based on:

  fast_up    = EMA_fast > EMA_slow  on fast tier AND fast slope > 0
  fast_down  = EMA_fast < EMA_slow  on fast tier AND fast slope < 0
  slow_strongly_up   = slow slope >  strong_slope_pct AND EMA_fast > EMA_slow
  slow_strongly_down = slow slope < -strong_slope_pct AND EMA_fast < EMA_slow

  long_allowed  = fast_up   AND NOT slow_strongly_down
  short_allowed = fast_down AND NOT slow_strongly_up

Slope is the relative change of EMA_fast over `slope_lookback` bars (state,
not edge-trigger — the filter cares whether the trend *is* up/down, not
whether it just crossed).
"""
from __future__ import annotations

from typing import Dict
import pandas as pd


class TrendFilter:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.strong_slope_pct: float = float(cfg.get("strong_slope_pct", 0.002))
        self._fast_cfg = cfg.get("fast", {})
        self._slow_cfg = cfg.get("slow", {})

    def _tier_state(self, df: pd.DataFrame, tier_cfg: dict) -> dict:
        ema_fast = int(tier_cfg.get("ema_fast", 20))
        ema_slow = int(tier_cfg.get("ema_slow", 50))
        lookback = int(tier_cfg.get("slope_lookback", 10))
        required = ema_slow + lookback + 1
        if df is None or len(df) < required:
            return {"ok": False, "direction": "flat", "slope_pct": 0.0,
                    "ema_fast": None, "ema_slow": None}
        close = df["close"]
        ef_series = close.ewm(span=ema_fast, adjust=False).mean()
        es_series = close.ewm(span=ema_slow, adjust=False).mean()
        ef_now  = float(ef_series.iloc[-1])
        es_now  = float(es_series.iloc[-1])
        ef_prev = float(ef_series.iloc[-1 - lookback])
        slope_pct = (ef_now - ef_prev) / (ef_prev + 1e-12)
        if ef_now > es_now and slope_pct > 0:
            direction = "up"
        elif ef_now < es_now and slope_pct < 0:
            direction = "down"
        else:
            direction = "flat"
        return {"ok": True, "direction": direction, "slope_pct": float(slope_pct),
                "ema_fast": ef_now, "ema_slow": es_now}

    def check(self, dfs: Dict[str, pd.DataFrame]) -> dict:
        fast_tf = self._fast_cfg.get("tf", "15m")
        slow_tf = self._slow_cfg.get("tf", "1h")
        fast = self._tier_state(dfs.get(fast_tf), self._fast_cfg)
        slow = self._tier_state(dfs.get(slow_tf), self._slow_cfg)

        if not fast["ok"] or not slow["ok"]:
            return {
                "long_allowed": False, "short_allowed": False,
                "reasoning": "insufficient history on fast or slow tier",
                "fast": {"direction": fast["direction"], "slope_pct": fast["slope_pct"]},
                "slow": {"direction": slow["direction"], "slope_pct": slow["slope_pct"],
                         "strong": False},
            }

        slow_strong_up   = slow["slope_pct"] >  self.strong_slope_pct and slow["direction"] == "up"
        slow_strong_down = slow["slope_pct"] < -self.strong_slope_pct and slow["direction"] == "down"
        slow_strong = slow_strong_up or slow_strong_down

        long_allowed  = (fast["direction"] == "up")   and not slow_strong_down
        short_allowed = (fast["direction"] == "down") and not slow_strong_up

        if long_allowed:
            reason = f"fast={fast['direction']} slow={slow['direction']} → LONG allowed"
        elif short_allowed:
            reason = f"fast={fast['direction']} slow={slow['direction']} → SHORT allowed"
        elif fast["direction"] == "up" and slow_strong_down:
            reason = (f"fast=up but slow strongly down "
                      f"(slope={slow['slope_pct']:.4f}) → LONG blocked")
        elif fast["direction"] == "down" and slow_strong_up:
            reason = (f"fast=down but slow strongly up "
                      f"(slope={slow['slope_pct']:.4f}) → SHORT blocked")
        else:
            reason = f"fast={fast['direction']} → no direction admitted"

        return {
            "long_allowed":  long_allowed,
            "short_allowed": short_allowed,
            "reasoning":     reason,
            "fast": {"direction": fast["direction"], "slope_pct": fast["slope_pct"]},
            "slow": {"direction": slow["direction"], "slope_pct": slow["slope_pct"],
                     "strong": slow_strong},
        }
