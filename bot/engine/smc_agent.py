"""
SMC (Smart Money Concepts) Agent.
Detects liquidity zones, BOS/CHOCH, FVGs, and institutional price structures.
Produces buy/sell scores independent of ML models.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class AgentSignal:
    agent: str
    buy_score: float
    sell_score: float
    net_score: float
    confidence: float
    reasoning: str = ""
    zones: list = field(default_factory=list)


class SMCAgent:
    def __init__(self, lookback: int = 5):
        self.lookback = lookback

    def analyze(self, df: pd.DataFrame, profile, market_ctx=None) -> AgentSignal:
        if df is None or len(df) < 50:
            return AgentSignal("smc", 0, 0, 0, 0, reasoning="insufficient data")

        close = df["close"].values
        high  = df["high"].values
        low   = df["low"].values
        vol   = df["volume"].values

        if len(close) > 1:
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
            )
            _atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
        else:
            _atr = 0.0

        pivots_high, pivots_low = self._detect_pivots(high, low)
        structure = self._track_structure(pivots_high, pivots_low)

        _regime_str    = ((market_ctx or {}).get("regime") or "").upper()
        _trend_dir     = ((market_ctx or {}).get("trend_direction") or "").upper()
        _is_bull_trend = "TREND" in _regime_str and _trend_dir == "BULLISH"
        _is_bear_trend = "TREND" in _regime_str and _trend_dir == "BEARISH"

        # Sub-checks
        checks = {}
        checks["sweep"] = self._detect_liquidity_sweep(high, low, close, pivots_high, pivots_low, profile.smc_liquidity_sweep_pct)
        checks["bos"]   = self._detect_bos(close, pivots_high, pivots_low, structure, profile.smc_bos_body_pct)
        checks["fvg"]   = self._detect_fvg(high, low, close, _atr)
        checks["volume"]= self._detect_volume_spike(vol, close, profile.smc_volume_spike_ratio)
        checks["pattern"] = self._detect_pattern_completion(high, low, close, profile.smc_pattern_completion)

        # Map to scores
        buy_score  = 0.0
        sell_score = 0.0
        reasons_parts = []
        active_checks = 0

        # Sweep: ±0.35 — suppressed only when counter-trend
        sweep = checks["sweep"]
        if sweep.get("direction") == "bullish" and not _is_bear_trend:
            buy_score += 0.35
            reasons_parts.append(f"sweep+{sweep.get('pct',0):.2%}")
            active_checks += 1
        elif sweep.get("direction") == "bearish" and not _is_bull_trend:
            sell_score += 0.35
            reasons_parts.append(f"sweep-{sweep.get('pct',0):.2%}")
            active_checks += 1

        # BOS: ±0.30 — suppressed only when counter-trend
        bos = checks["bos"]
        if bos.get("direction") == "bullish" and not _is_bear_trend:
            buy_score += 0.30
            reasons_parts.append(f"BOS+{bos.get('body_pct',0):.0%}")
            active_checks += 1
        elif bos.get("direction") == "bearish" and not _is_bull_trend:
            sell_score += 0.30
            reasons_parts.append(f"BOS-{bos.get('body_pct',0):.0%}")
            active_checks += 1

        # FVG: ±0.25
        fvg = checks["fvg"]
        if fvg.get("direction") == "bullish":
            buy_score += 0.25
            reasons_parts.append("FVG+")
            active_checks += 1
        elif fvg.get("direction") == "bearish":
            sell_score += 0.25
            reasons_parts.append("FVG-")
            active_checks += 1

        # Volume: ±0.15 — direction biased by price location within recent range
        vol_chk = checks["volume"]
        if vol_chk:
            price_loc = vol_chk.get("price_location", 0.5)
            if price_loc > 0.8:
                sell_score += 0.15
            elif price_loc < 0.2:
                buy_score += 0.15
            elif structure in ("uptrend", "ranging_bull"):
                buy_score += 0.15
            else:
                sell_score += 0.15
            active_checks += 1
            reasons_parts.append(f"vol={vol_chk.get('ratio',0):.1f}x loc={price_loc:.2f}")

        # Pattern: ±0.10
        pat = checks["pattern"]
        if pat.get("direction") == "bullish":
            buy_score += 0.10
            active_checks += 1
            reasons_parts.append(f"pattern+{pat.get('pct',0):.0%}")
        elif pat.get("direction") == "bearish":
            sell_score += 0.10
            active_checks += 1
            reasons_parts.append(f"pattern-{pat.get('pct',0):.0%}")

        # Sub-check minimum
        smc_min = getattr(profile, 'smc_sub_checks_min', 2)
        if active_checks < smc_min:
            buy_score = 0.0
            sell_score = 0.0
            reasons_parts = [f"sub-checks={active_checks}/{smc_min}"]

        net_score = max(-1.0, min(1.0, buy_score - sell_score))
        confidence = min(1.0, max(0.40, abs(net_score) + (active_checks * 0.10)))

        zones = self._map_zones(close, pivots_high, pivots_low, structure)

        return AgentSignal(
            agent="smc",
            buy_score=round(buy_score, 4),
            sell_score=round(sell_score, 4),
            net_score=round(net_score, 4),
            confidence=round(confidence, 4),
            reasoning=" | ".join(reasons_parts) if reasons_parts else "no setup",
            zones=zones,
        )

    def _detect_pivots(self, high: np.ndarray, low: np.ndarray):
        """5-bar fractal pivot detection."""
        n = len(high)
        pivots_high = []
        pivots_low  = []
        lb = self.lookback
        for i in range(lb, n - lb):
            if all(high[i] >= high[j] for j in range(i-lb, i+lb+1) if j != i):
                pivots_high.append((i, float(high[i])))
            if all(low[i] <= low[j] for j in range(i-lb, i+lb+1) if j != i):
                pivots_low.append((i, float(low[i])))
        return pivots_high, pivots_low

    def _track_structure(self, pivots_high: list, pivots_low: list) -> str:
        """HH/HL = uptrend, LH/LL = downtrend, else ranging."""
        if len(pivots_high) < 2 or len(pivots_low) < 2:
            return "ranging"
        hh = all(pivots_high[j][1] >= pivots_high[j-1][1] for j in range(1, min(3, len(pivots_high))))
        hl = all(pivots_low[j][1] >= pivots_low[j-1][1] for j in range(1, min(3, len(pivots_low))))
        lh = all(pivots_high[j][1] <= pivots_high[j-1][1] for j in range(1, min(3, len(pivots_high))))
        ll = all(pivots_low[j][1] <= pivots_low[j-1][1] for j in range(1, min(3, len(pivots_low))))
        if hh and hl: return "uptrend"
        if lh and ll: return "downtrend"
        if hl and not ll: return "ranging_bull"
        if ll and not hh: return "ranging_bear"
        return "ranging"

    def _detect_liquidity_sweep(self, high, low, close, pivots_high, pivots_low, min_pct: float) -> dict:
        """Wick pierces a pivot level then price closes back inside."""
        if not pivots_high or not pivots_low:
            return {"direction": None}
        n = len(close)
        if n < 10:
            return {"direction": None}
        # Check last 3 candles against recent pivots (up to 20 bars back)
        for i in range(max(0, n-3), n):
            hi = float(high[i])
            lo = float(low[i])
            cl = float(close[i])
            # Sweep above pivot high → bearish
            for idx, lvl in pivots_high[-5:]:
                if idx >= i - 20 and hi > lvl * 1.001 and cl < lvl * 0.999:
                    sweep_pct = (hi - lvl) / (lvl + 1e-9)
                    if sweep_pct >= min_pct:
                        return {"direction": "bearish", "pct": round(sweep_pct, 5), "level": round(lvl, 2)}
            # Sweep below pivot low → bullish
            for idx, lvl in pivots_low[-5:]:
                if idx >= i - 20 and lo < lvl * 0.999 and cl > lvl * 1.001:
                    sweep_pct = (lvl - lo) / (lvl + 1e-9)
                    if sweep_pct >= min_pct:
                        return {"direction": "bullish", "pct": round(sweep_pct, 5), "level": round(lvl, 2)}
        return {"direction": None}

    def _detect_bos(self, close, pivots_high, pivots_low, structure: str, min_body_pct: float) -> dict:
        """Break of Structure: price breaks latest pivot with displacement."""
        if not pivots_high or not pivots_low:
            return {"direction": None}
        n = len(close)
        if n < 5:
            return {"direction": None}
        # Check body over last 3 candles (accumulated displacement)
        body_pct = abs(float(close[-1]) - float(close[-4])) / (float(close[-4]) + 1e-9)
        if body_pct < min_body_pct:
            return {"direction": None, "body_pct": round(body_pct, 4)}

        last_p = float(close[-1])
        recent_high = pivots_high[-1][1] if pivots_high else float('inf')
        recent_low  = pivots_low[-1][1]  if pivots_low  else 0

        if structure in ("downtrend", "ranging_bear", "ranging"):
            if last_p > recent_high:
                return {"direction": "bullish", "body_pct": round(body_pct, 4)}
        if structure in ("uptrend", "ranging_bull", "ranging"):
            if last_p < recent_low:
                return {"direction": "bearish", "body_pct": round(body_pct, 4)}
        return {"direction": None}

    def _detect_fvg(self, high, low, close, atr: float = 0.0) -> dict:
        """3-candle Fair Value Gap: gap between candle 1's price and candle 3's price."""
        n = len(close)
        if n < 5:
            return {"direction": None}
        gap_threshold = (atr / (float(close[-1]) + 1e-9)) * 0.3 if atr > 0 else 0.0005
        # Check last 2 possible FVG windows
        for offset in [0, 1]:
            i = n - 1 - offset
            if i < 3:
                continue
            h0, l0 = float(high[i]), float(low[i])
            h2, l2 = float(high[i-2]), float(low[i-2])
            # Bullish FVG: candle-3 low > candle-1 high
            if l2 > h0:
                gap = abs(l2 - h0) / (h0 + 1e-9)
                if gap > gap_threshold:
                    return {"direction": "bullish", "gap_pct": round(gap, 4)}
            # Bearish FVG: candle-3 high < candle-1 low
            if h2 < l0:
                gap = abs(l0 - h2) / (l0 + 1e-9)
                if gap > gap_threshold:
                    return {"direction": "bearish", "gap_pct": round(gap, 4)}
        return {"direction": None}

    def _detect_volume_spike(self, vol, close, min_ratio: float) -> dict:
        """Current volume vs 20-bar average, with price location within recent range."""
        if len(vol) < 21 or len(close) < 20:
            return {}
        vol_ma = np.mean(vol[-21:-1])
        ratio = float(vol[-1]) / (vol_ma + 1e-9)
        if ratio >= min_ratio:
            high_20 = np.max(close[-20:])
            low_20  = np.min(close[-20:])
            price_location = (float(close[-1]) - float(low_20)) / (float(high_20) - float(low_20) + 1e-9)
            return {"ratio": round(ratio, 2), "price_location": round(price_location, 3)}
        return {}

    def _detect_pattern_completion(self, high, low, close, min_pct: float) -> dict:
        """Simple pattern detection: recent higher lows = bullish, lower highs = bearish."""
        n = len(close)
        if n < 10:
            return {"direction": None}
        recent_highs = [float(high[i]) for i in range(n-5, n)]
        recent_lows  = [float(low[i]) for i in range(n-5, n)]
        hl_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] > recent_lows[i-1])
        lh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] < recent_highs[i-1])
        total = len(recent_lows) - 1
        if total > 0:
            bull_pct = hl_count / total
            bear_pct = lh_count / total
            if bull_pct >= min_pct:
                return {"direction": "bullish", "pct": round(bull_pct, 2)}
            elif bear_pct >= min_pct:
                return {"direction": "bearish", "pct": round(bear_pct, 2)}
        return {"direction": None}

    def _map_zones(self, close, pivots_high, pivots_low, structure: str) -> list:
        """Map support/resistance from recent pivot clusters."""
        zones = []
        curr = float(close[-1])

        # Support from recent pivot lows
        recent_lows = sorted([(idx, val) for idx, val in pivots_low if idx >= len(close) - 50], key=lambda x: x[1], reverse=True)
        for idx, val in recent_lows[:2]:
            if val < curr:
                zones.append({"type": "support", "level": round(val, 4), "distance_pct": round((curr - val) / curr * 100, 2)})

        # Resistance from recent pivot highs
        recent_highs = sorted([(idx, val) for idx, val in pivots_high if idx >= len(close) - 50], key=lambda x: x[1])
        for idx, val in recent_highs[:2]:
            if val > curr:
                zones.append({"type": "resistance", "level": round(val, 4), "distance_pct": round((val - curr) / curr * 100, 2)})

        return zones
