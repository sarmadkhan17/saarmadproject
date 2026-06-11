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
    confirmed: bool              # True = confirms direction, False = kills it
    kill: bool                   # Hard kill — strong counter-signal
    ob_imbalance: float          # bid/ask ratio (>1 = buy pressure)
    cvd_direction: str           # "bullish" | "bearish" | "neutral"
    cvd_divergence: bool         # True = CVD diverges from price
    funding_extreme: bool        # True = funding rate > 0.1% (extreme longs)
    absorption_confirmed: bool   # True = limit orders absorbing aggressive flow
    reasoning: str


class MicrostructureAgent:
    OB_DEPTH = 20           # order book levels to read
    CVD_WINDOW = 50         # candles for CVD calculation
    IMBALANCE_STRONG = 2.0  # ratio for strong imbalance
    IMBALANCE_MILD   = 1.4  # ratio for mild imbalance
    IMBALANCE_VETO   = 2.5  # overwhelming contra wall — hard veto regardless of CVD

    def analyze(
        self,
        exchange,
        symbol: str,
        df,                  # OHLCV dataframe (trading timeframe)
        action: str,         # "BUY" or "SELL" — the structural bias to validate
        funding_rate: float = 0.0,
        df_5m=None,          # 5-minute OHLCV dataframe for absorption check
    ) -> MicrostructureSignal:
        ob_ratio   = self._order_book_imbalance(exchange, symbol)
        cvd_dir, cvd_div = self._cvd_analysis(df, action)
        funding_extreme = abs(funding_rate) > 0.001  # 0.1% per 8h threshold

        side = "LONG" if action == "BUY" else "SHORT"
        absorption = self.check_cvd_absorption(df_5m if df_5m is not None else df, side)

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
                if ob_ratio > self.IMBALANCE_VETO:
                    # Overwhelming bid wall — squeeze risk overrides any CVD
                    # confirmation. (Shorts were approved at ob 2.0-3.4 during
                    # the Jun-6/7 squeeze; a stacked book must veto the side.)
                    kill = True
                    reasons.append(f"OB: overwhelming bid pressure {ob_ratio:.2f}x (veto)")
                elif ob_ratio > self.IMBALANCE_STRONG:
                    if cvd_dir != "bearish":
                        # OB heavy bids + CVD not confirming SELL → kill
                        kill = True
                        reasons.append(f"OB: heavy bid pressure {ob_ratio:.2f}x")
                    else:
                        # CVD confirms bearish direction — OB wall alone not enough
                        reasons.append(f"OB: heavy bid pressure {ob_ratio:.2f}x | CVD confirms ✓")
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

        # ── Order flow absorption check (5m) ──────────────────────────
        if not absorption:
            kill = True
            reasons.append(
                f"Absorption: no absorption on 5m — aggressive {'selling' if action == 'BUY' else 'buying'} ongoing (veto)"
            )
        else:
            reasons.append("Absorption: limit orders absorbing flow ✓")

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
            absorption_confirmed=absorption,
            reasoning=" | ".join(reasons),
        )

    # ── Order Flow Absorption ─────────────────────────────────────────────────

    ABSORPTION_WINDOW  = 5    # candles to inspect for absorption signal
    ABSORPTION_MIN_LEN = 5    # minimum rows required in df_5m

    def check_cvd_absorption(self, df_5m, side: str) -> bool:
        """
        Detect whether passive limit orders are absorbing aggressive market flow.

        For LONG at support:
          - Absorption confirmed when price is falling but CVD is NOT falling
            aggressively (CVD higher low vs price lower low, or CVD flattening).
          - No absorption when price AND CVD both fall — genuine dump.

        For SHORT at resistance:
          - Absorption confirmed when price is rising but CVD is NOT rising
            aggressively (CVD lower high vs price higher high, or CVD flattening).
          - No absorption when price AND CVD both rise — genuine pump.

        Returns True if absorption is detected, False otherwise (safe default).
        """
        if df_5m is None or len(df_5m) < self.ABSORPTION_MIN_LEN:
            return False

        try:
            w = self.ABSORPTION_WINDOW
            close  = df_5m["close"].values[-w:]
            open_  = df_5m["open"].values[-w:]
            volume = df_5m["volume"].values[-w:]

            # Taker-side CVD: green=+vol, red=-vol, doji=0
            delta = np.where(close > open_, volume,
                             np.where(close < open_, -volume, 0.0))
            cvd   = np.cumsum(delta)

            price_falling = close[-1] < close[0]
            price_rising  = close[-1] > close[0]

            cvd_start, cvd_end = cvd[0], cvd[-1]
            cvd_falling = cvd_end < cvd_start
            cvd_rising  = cvd_end > cvd_start

            # Full-window slope: net delta relative to total volume
            total_volume   = volume.sum() or 1.0
            cvd_slope_norm = abs(cvd_end - cvd_start) / total_volume

            # Recent slope: last 3 bars only (captures late flattening)
            recent_net_delta  = abs(delta[-3:].sum())
            recent_volume     = volume[-3:].sum() or 1.0
            recent_slope_norm = recent_net_delta / recent_volume

            CVD_FLAT_THRESHOLD = 0.15  # ≤15 % net delta/volume = flat

            if side == "LONG":
                if not price_falling:
                    return True
                # Genuine dump requires: CVD falling AND recent slope still aggressive
                if cvd_falling and cvd_slope_norm > CVD_FLAT_THRESHOLD \
                        and recent_slope_norm > CVD_FLAT_THRESHOLD:
                    return False
                # CVD flat/rising OR recently flattened → absorption detected
                return True

            elif side == "SHORT":
                if not price_rising:
                    return True
                if cvd_rising and cvd_slope_norm > CVD_FLAT_THRESHOLD \
                        and recent_slope_norm > CVD_FLAT_THRESHOLD:
                    return False
                return True

            return True  # unknown side — don't veto

        except Exception as e:
            log.debug(f"Absorption check error: {e}")
            return False

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
