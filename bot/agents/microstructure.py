"""
MicrostructureAgent — CVD + order book imbalance.

Reads:
  - Binance order book (bid/ask depth) → imbalance ratio
  - Binance trade stream (aggregated) → Cumulative Volume Delta
  - Funding rate (futures only)

Outputs a signal that CONFIRMS or KILLS the structural bias from SMCAgent.
It does not generate direction — it validates it.

Hard kill vs soft size reduction (no-double-jeopardy with itself):
  KILL only when independent flow reads agree the trade is wrong —
  overwhelming contra wall (≥2.5x), contra wall + CVD not confirming,
  CVD divergence + order book against, or no absorption on 5m.
  A LONE contra signal (wall with CVD confirming, or divergence with a
  neutral book) emits size_mult 0.7/0.75 instead of blocking.

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
    size_mult: float = 1.0       # soft contra-flow → position size multiplier (≤1.0)


class MicrostructureAgent:
    OB_DEPTH = 20           # order book levels to read
    CVD_WINDOW = 50         # candles for CVD calculation
    IMBALANCE_STRONG = 2.0  # ratio for strong imbalance
    IMBALANCE_MILD   = 1.4  # ratio for mild imbalance
    IMBALANCE_VETO   = 2.5  # overwhelming contra wall — hard veto regardless of CVD

    # Soft-kill multipliers: a LONE contra-flow signal shrinks the position
    # instead of blocking it (shadow data showed lone-signal kills blocking
    # winners); hard kills are reserved for signal COMBOS where independent
    # flow reads agree the trade is wrong, plus the overwhelming-wall veto.
    SOFT_WALL_MULT = 0.7    # contra wall present but CVD confirms the trade
    SOFT_DIV_MULT  = 0.75   # CVD divergence alone, order book not against

    # Absorption-veto calibration (overridable from config — see __init__).
    # The veto fires only on a GENUINE, unabsorbed adverse move: a meaningful
    # price displacement (≥ MOVE_MULT × the window's own typical bar size) that
    # CVD strongly confirms (slope ≥ CVD_STRONG_THRESHOLD). A low threshold here
    # mistakes ordinary 5m chop for an aggressive dump/pump and vetoes both
    # sides indiscriminately — the failure mode that soft-halted the bot.
    CVD_STRONG_THRESHOLD = 0.45  # net delta/volume that counts as decisive flow
    ABSORPTION_MOVE_MULT = 1.0   # adverse move must exceed this × typical bar move

    def __init__(self, config: dict = None):
        cfg = ((config or {}).get("absorption") or {})
        self.cvd_strong_threshold = float(
            cfg.get("cvd_strong_threshold", self.CVD_STRONG_THRESHOLD))
        self.absorption_move_mult = float(
            cfg.get("move_mult", self.ABSORPTION_MOVE_MULT))

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
        size_mult = 1.0

        # ── Order book check ──────────────────────────────────────────
        # Hard kill only for an overwhelming wall (squeeze evidence) or a
        # COMBO of contra wall + CVD not confirming the trade. A lone wall
        # with CVD on the trade's side shrinks the position instead.
        if ob_ratio is not None:
            if action == "BUY":
                if ob_ratio < (1.0 / self.IMBALANCE_VETO):
                    kill = True
                    reasons.append(f"OB: overwhelming ask pressure {ob_ratio:.2f}x (veto)")
                elif ob_ratio < (1.0 / self.IMBALANCE_STRONG):
                    if cvd_dir != "bullish":
                        kill = True
                        reasons.append(f"OB: heavy ask pressure {ob_ratio:.2f}x + CVD not confirming")
                    else:
                        size_mult *= self.SOFT_WALL_MULT
                        reasons.append(f"OB: heavy ask pressure {ob_ratio:.2f}x | CVD confirms — size ×{self.SOFT_WALL_MULT}")
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
                        # CVD confirms bearish — wall alone shrinks, not blocks
                        size_mult *= self.SOFT_WALL_MULT
                        reasons.append(f"OB: heavy bid pressure {ob_ratio:.2f}x | CVD confirms — size ×{self.SOFT_WALL_MULT}")
                elif ob_ratio < (1.0 / self.IMBALANCE_MILD):
                    reasons.append(f"OB: ask support {ob_ratio:.2f}x ✓")
                else:
                    reasons.append(f"OB: neutral {ob_ratio:.2f}x")
        else:
            ob_ratio = 1.0
            reasons.append("OB: unavailable")

        # ── CVD divergence check ──────────────────────────────────────
        # Divergence + order book against the trade = two independent flow
        # reads agreeing → hard kill. Lone divergence shrinks the position.
        if cvd_div:
            ob_against = (action == "BUY"  and ob_ratio < (1.0 / self.IMBALANCE_MILD)) or \
                         (action == "SELL" and ob_ratio > self.IMBALANCE_MILD)
            if ob_against:
                kill = True
                reasons.append(f"CVD: divergence vs price ({cvd_dir}) + OB against (veto)")
            else:
                size_mult *= self.SOFT_DIV_MULT
                reasons.append(f"CVD: divergence vs price ({cvd_dir}) — size ×{self.SOFT_DIV_MULT}")
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
            size_mult = 0.0
        else:
            size_mult = max(0.5, round(size_mult, 3))  # stacked softs floor at 0.5

        return MicrostructureSignal(
            confirmed=confirmed,
            kill=kill,
            ob_imbalance=round(ob_ratio, 3),
            cvd_direction=cvd_dir,
            cvd_divergence=cvd_div,
            funding_extreme=funding_extreme,
            absorption_confirmed=absorption,
            reasoning=" | ".join(reasons),
            size_mult=size_mult,
        )

    # ── Order Flow Absorption ─────────────────────────────────────────────────

    ABSORPTION_WINDOW  = 5    # candles to inspect for absorption signal
    ABSORPTION_MIN_LEN = 5    # minimum rows required in df_5m

    def check_cvd_absorption(self, df_5m, side: str) -> bool:
        """
        Detect whether passive limit orders are absorbing aggressive market flow.

        Returns True (absorption present / no decisive opposing flow → don't
        veto) UNLESS a genuine, unabsorbed adverse move is in progress. "No
        absorption" requires ALL of:
          1. Significant adverse price displacement — the net move over the
             window exceeds MOVE_MULT × the window's own typical bar move
             (volatility-scaled, so ordinary 5m chop nets ~0 and never vetoes).
          2. CVD strongly confirms it — directional slope ≥ CVD_STRONG_THRESHOLD
             on BOTH the full window and the last 3 bars.
          3. No rescuing divergence — for a LONG, falling price on net-buying
             candles (CVD rising) IS absorption; symmetric for a SHORT.

        For LONG at support: veto only on a real dump (price down + CVD down).
        For SHORT at resistance: veto only on a real pump (price up + CVD up).
        A small/noisy move, a flat/diverging CVD, or a late flattening → pass.

        Returns False (no absorption) only on a decisive adverse move; True
        otherwise. Insufficient data defaults to False (safe).
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

            cvd_start, cvd_end = cvd[0], cvd[-1]
            cvd_falling = cvd_end < cvd_start
            cvd_rising  = cvd_end > cvd_start

            # Full-window slope: net delta relative to total volume
            total_volume   = volume.sum() or 1.0
            cvd_slope_norm = abs(cvd_end - cvd_start) / total_volume

            # Recent slope: last 3 bars only (so a late flattening still rescues)
            recent_net_delta  = abs(delta[-3:].sum())
            recent_volume     = volume[-3:].sum() or 1.0
            recent_slope_norm = recent_net_delta / recent_volume

            cvd_decisive = (cvd_slope_norm > self.cvd_strong_threshold
                            and recent_slope_norm > self.cvd_strong_threshold)

            # Adverse price displacement, scaled by the window's own volatility:
            # the net move must exceed MOVE_MULT × the typical per-bar move
            # (mean abs bar-to-bar return). Chop → net≈0 ≪ scale → no veto.
            base = close[0] if close[0] else 1.0
            net_move = (close[-1] - close[0]) / base
            steps = np.abs(np.diff(close)) / base
            typical_bar = float(steps.mean()) if steps.size else 0.0
            move_floor = self.absorption_move_mult * typical_bar
            # Degenerate (flat synthetic data, no volatility) → fall back to the
            # sign of the net move so directional test fixtures still resolve.
            significant_against = (
                abs(net_move) > move_floor if move_floor > 0 else net_move != 0
            )

            if side == "LONG":
                price_against = net_move < 0  # price falling
                if price_against and significant_against \
                        and cvd_falling and cvd_decisive:
                    return False                       # genuine dump
                return True                            # absorption / no decisive flow

            elif side == "SHORT":
                price_against = net_move > 0  # price rising
                if price_against and significant_against \
                        and cvd_rising and cvd_decisive:
                    return False                       # genuine pump
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
