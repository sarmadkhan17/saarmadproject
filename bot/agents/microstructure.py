"""
MicrostructureAgent — CVD + order book imbalance.

Reads:
  - Binance order book (bid/ask depth) → imbalance ratio
  - Binance trade stream (aggregated) → Cumulative Volume Delta
  - Funding rate (futures only)

Outputs a signal that CONFIRMS or KILLS the structural bias from SMCAgent.
It does not generate direction — it validates it.

CVD divergence rule:
  Price making new highs + CVD making lower highs → KILL long signal
  Price making new lows  + CVD making higher lows  → KILL short signal
  CVD and price aligned  → CONFIRM signal

Order book imbalance rule:
  bid_volume / ask_volume > 2.0 → bullish pressure
  ask_volume / bid_volume > 2.0 → bearish pressure
  < 1.5 either way         → neutral / inconclusive
"""

import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("Microstructure")


@dataclass
class MicrostructureSignal:
    confirmed: bool         # True = confirms direction, False = kills it
    kill: bool              # Hard kill — strong counter-signal
    ob_imbalance: float     # bid/ask ratio (>1 = buy pressure)
    cvd_direction: str      # "bullish" | "bearish" | "neutral"
    cvd_divergence: bool    # True = CVD diverges from price
    funding_extreme: bool   # True = funding rate > 0.1% (extreme longs)
    reasoning: str


class MicrostructureAgent:
    OB_DEPTH = 20           # order book levels to read
    CVD_WINDOW = 50         # candles for CVD calculation
    IMBALANCE_STRONG = 2.0  # ratio for strong imbalance
    IMBALANCE_MILD   = 1.4  # ratio for mild imbalance

    def analyze(
        self,
        exchange,
        symbol: str,
        df,                  # OHLCV dataframe (trading timeframe)
        action: str,         # "BUY" or "SELL" — the structural bias to validate
        funding_rate: float = 0.0,
    ) -> MicrostructureSignal:
        ob_ratio   = self._order_book_imbalance(exchange, symbol)
        cvd_dir, cvd_div = self._cvd_analysis(df, action)
        funding_extreme = abs(funding_rate) > 0.001  # 0.1% per 8h threshold

        reasons = []
        kill = False
        confirmed = True

        # ── Order book check ──────────────────────────────────────────
        if ob_ratio is not None:
            if action == "BUY":
                if ob_ratio < (1.0 / self.IMBALANCE_STRONG):
                    # Strong ask-side pressure against longs
                    kill = True
                    reasons.append(f"OB: heavy ask pressure {ob_ratio:.2f}x")
                elif ob_ratio > self.IMBALANCE_MILD:
                    reasons.append(f"OB: bid support {ob_ratio:.2f}x ✓")
                else:
                    reasons.append(f"OB: neutral {ob_ratio:.2f}x")
            elif action == "SELL":
                if ob_ratio > self.IMBALANCE_STRONG:
                    kill = True
                    reasons.append(f"OB: heavy bid pressure {ob_ratio:.2f}x")
                elif ob_ratio < (1.0 / self.IMBALANCE_MILD):
                    reasons.append(f"OB: ask support {ob_ratio:.2f}x ✓")
                else:
                    reasons.append(f"OB: neutral {ob_ratio:.2f}x")
        else:
            ob_ratio = 1.0
            reasons.append("OB: unavailable")

        # ── CVD divergence check ──────────────────────────────────────
        if cvd_div:
            kill = True
            reasons.append(f"CVD: divergence vs price ({cvd_dir}) — signal weakening")
        else:
            if (action == "BUY"  and cvd_dir == "bullish") or \
               (action == "SELL" and cvd_dir == "bearish"):
                reasons.append(f"CVD: aligned {cvd_dir} ✓")
            else:
                reasons.append(f"CVD: neutral ({cvd_dir})")

        # ── Funding rate check (futures) ──────────────────────────────
        if funding_extreme:
            if action == "BUY" and funding_rate > 0.001:
                # Longs already crowded — reduce conviction but don't kill
                confirmed = False
                reasons.append(f"Funding: extreme long {funding_rate*100:.3f}% — crowded")
            elif action == "SELL" and funding_rate < -0.001:
                confirmed = False
                reasons.append(f"Funding: extreme short {funding_rate*100:.3f}% — crowded")

        if kill:
            confirmed = False

        return MicrostructureSignal(
            confirmed=confirmed,
            kill=kill,
            ob_imbalance=round(ob_ratio, 3),
            cvd_direction=cvd_dir,
            cvd_divergence=cvd_div,
            funding_extreme=funding_extreme,
            reasoning=" | ".join(reasons),
        )

    def _order_book_imbalance(self, exchange, symbol: str) -> Optional[float]:
        """Fetch order book and return bid_volume / ask_volume ratio."""
        try:
            ob = exchange.fetch_order_book(symbol, limit=self.OB_DEPTH)
            bid_vol = sum(qty for _, qty in ob.get("bids", []))
            ask_vol = sum(qty for _, qty in ob.get("asks", []))
            if ask_vol <= 0:
                return None
            return round(bid_vol / ask_vol, 3)
        except Exception as e:
            log.warning(f"OB fetch failed {symbol}: {e}")
            return None

    def _cvd_analysis(self, df, action: str) -> tuple:
        """
        Compute CVD from OHLCV using the taker estimate:
          CVD_bar = (close > open) ? +volume : -volume
        Then check for divergence between price direction and CVD direction.

        Returns (cvd_direction, divergence_detected)
        """
        if df is None or len(df) < 10:
            return "neutral", False

        try:
            close  = df["close"].values[-self.CVD_WINDOW:]
            open_  = df["open"].values[-self.CVD_WINDOW:]
            volume = df["volume"].values[-self.CVD_WINDOW:]

            # Taker-side CVD approximation
            delta = np.where(close > open_, volume, -volume)
            cvd   = np.cumsum(delta)

            # Direction from last 5 bars
            recent_cvd   = cvd[-5:]
            recent_price = close[-5:]

            cvd_rising   = recent_cvd[-1]   > recent_cvd[0]
            price_rising = recent_price[-1] > recent_price[0]

            cvd_dir = "bullish" if cvd_rising else "bearish"

            # Divergence: price and CVD moving opposite directions
            divergence = (
                (action == "BUY"  and price_rising  and not cvd_rising) or
                (action == "SELL" and not price_rising and cvd_rising)
            )

            return cvd_dir, divergence

        except Exception as e:
            log.debug(f"CVD calc error: {e}")
            return "neutral", False
