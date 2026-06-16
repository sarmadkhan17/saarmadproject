"""
TechnicalAgent — pure rule-based multi-indicator agent.

Replaces the old MLTechnicalAgent (which used RF/LightGBM predictions).
No ML model, no feature scaler, no training. Just standard indicators
computed live: RSI, MACD, EMA alignment, Bollinger position, momentum.

Produces buy/sell scores in the same AgentSignal format the ensemble
expects, so it drops into the existing EnsembleEngine unchanged.

Signal-quality upgrades (2026-06-15):
  1. Multi-TF confluence — the 1h signal is gated by the 4h trend bias.
     Aligned signals keep full weight; counter-HTF signals are damped (not
     vetoed — high conviction can still pass the downstream gates).
  2. Closed-candle reads — indicators are computed on completed candles only
     (the forming last bar is dropped), so a half-built candle can't flicker
     the signal between scans.
  3. Adaptive thresholds — momentum / MACD significance scale with the
     symbol's own ATR%, and the RSI interpretation flips between mean-reversion
     (ranges) and momentum-confirmation (trends). No per-coin fitted numbers.
"""

import logging
import numpy as np
import pandas as pd

from engine.smc_agent import AgentSignal

log = logging.getLogger("TechnicalAgent")


class TechnicalAgent:
    """Multi-timeframe technical confluence. Coin-agnostic, no ML."""

    # 4h-confluence multipliers applied to the 1h score.
    CONFLUENCE_ALIGNED  = 1.00   # 4h agrees with the signal direction
    CONFLUENCE_FLAT     = 0.85   # 4h is directionless
    CONFLUENCE_AGAINST  = 0.50   # 4h opposes — counter-trend, damp hard

    def analyze(self, df: pd.DataFrame, profile,
                market_ctx: dict = None, htf_df: pd.DataFrame = None) -> AgentSignal:
        # Need ≥1 forming bar + 50 closed bars for a meaningful EMA50.
        if df is None or len(df) < 52:
            return AgentSignal("technical", 0, 0, 0, 0, reasoning="insufficient data")

        try:
            ctx      = market_ctx or {}
            regime   = str(ctx.get("regime", "")).upper()
            trending = "TREND" in regime or "CRASH" in regime

            # ── Closed-candle reads: drop the forming (incomplete) last bar so
            # the signal comes only from completed candles. iloc[-1] on a live
            # feed is the in-progress candle (same convention as ExitEngine /
            # microstructure, which read iloc[-2] for the "last completed bar").
            sig   = df.iloc[:-1]
            close = sig["close"]
            high  = sig["high"]
            low   = sig["low"]
            price = float(close.iloc[-1])

            # Volatility anchor — drives the adaptive thresholds below.
            atr_abs = self._atr(high, low, close, 14)
            atr_pct = atr_abs / (price + 1e-9)

            buy_score  = 0.0
            sell_score = 0.0
            reasons    = []

            # ── 1. EMA alignment (trend) ±0.30 ────────────────────────
            ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
            ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
            if price > ema20 > ema50:
                buy_score += 0.30
                reasons.append("EMA bull stack")
            elif price < ema20 < ema50:
                sell_score += 0.30
                reasons.append("EMA bear stack")

            # ── 2. RSI ±0.25 — regime-aware ───────────────────────────
            # Ranges → mean-reversion (fade extremes). Trends → momentum
            # confirmation (RSI strength supports the trend; don't fade it).
            rsi = self._rsi(close, 14)
            if trending:
                if rsi > 55:
                    buy_score += 0.15
                    reasons.append(f"RSI momo {rsi:.0f}")
                elif rsi < 45:
                    sell_score += 0.15
                    reasons.append(f"RSI momo {rsi:.0f}")
            else:
                if rsi < 35:
                    buy_score += 0.25
                    reasons.append(f"RSI oversold {rsi:.0f}")
                elif rsi > 65:
                    sell_score += 0.25
                    reasons.append(f"RSI overbought {rsi:.0f}")
                elif rsi < 45:
                    buy_score += 0.10
                elif rsi > 55:
                    sell_score += 0.10

            # ── 3. MACD histogram ±0.25 — ATR-scaled significance ─────
            # "Meaningful" histogram ≈ 10% of one ATR, instead of a fixed
            # 0.05% of price (which meant nothing on a high-vol coin and
            # everything on a calm one).
            macd_hist = self._macd_hist(close)
            sig_cut   = 0.10 * atr_abs
            if macd_hist > 0:
                buy_score += 0.25 if macd_hist > sig_cut else 0.12
                reasons.append("MACD+")
            elif macd_hist < 0:
                sell_score += 0.25 if abs(macd_hist) > sig_cut else 0.12
                reasons.append("MACD-")

            # ── 4. Momentum (5-bar ROC) ±0.20 — vol-scaled threshold ──
            mom_thr = max(0.004, 0.5 * atr_pct)
            if len(close) >= 6:
                roc = (price - float(close.iloc[-6])) / (float(close.iloc[-6]) + 1e-9)
                if roc > mom_thr:
                    buy_score += 0.20
                    reasons.append(f"mom+{roc:.1%}")
                elif roc < -mom_thr:
                    sell_score += 0.20
                    reasons.append(f"mom{roc:.1%}")

            net_score = max(-1.0, min(1.0, buy_score - sell_score))

            # ── 5. Multi-TF confluence — gate the 1h signal by the 4h trend ──
            conf_mult, htf_dir = self._confluence_mult(net_score, htf_df)
            if conf_mult != 1.0:
                buy_score  *= conf_mult
                sell_score *= conf_mult
                net_score  *= conf_mult
                reasons.append(f"4h:{htf_dir}×{conf_mult:.2f}")

            net_score  = max(-1.0, min(1.0, net_score))
            confidence = min(0.95, max(0.40, abs(net_score) + 0.30))

            return AgentSignal(
                agent="technical",
                buy_score=round(buy_score, 4),
                sell_score=round(sell_score, 4),
                net_score=round(net_score, 4),
                confidence=round(confidence, 4),
                reasoning=" | ".join(reasons) if reasons else "neutral",
            )
        except Exception as e:
            log.warning(f"TechnicalAgent error: {e}")
            return AgentSignal("technical", 0, 0, 0, 0, reasoning=f"error: {e}")

    def _confluence_mult(self, net_score: float, htf_df: pd.DataFrame) -> tuple:
        """Return (multiplier, htf_direction) for the higher-TF confluence gate."""
        htf_dir = self._htf_bias(htf_df)
        if htf_dir == "flat":
            return self.CONFLUENCE_FLAT, htf_dir
        if net_score == 0:
            return 1.0, htf_dir
        agrees = (net_score > 0 and htf_dir == "up") or (net_score < 0 and htf_dir == "down")
        return (self.CONFLUENCE_ALIGNED if agrees else self.CONFLUENCE_AGAINST), htf_dir

    def _htf_bias(self, htf_df: pd.DataFrame) -> str:
        """4h trend bias from EMA20/50 stack on closed candles. up | down | flat."""
        if htf_df is None or len(htf_df) < 52:
            return "flat"
        try:
            c   = htf_df.iloc[:-1]["close"]
            e20 = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
            e50 = float(c.ewm(span=50, adjust=False).mean().iloc[-1])
            p   = float(c.iloc[-1])
        except Exception:
            return "flat"
        if p > e20 > e50:
            return "up"
        if p < e20 < e50:
            return "down"
        return "flat"

    def _atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> float:
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        val = tr.rolling(period).mean().iloc[-1]
        if pd.isna(val):
            return float(abs(high.iloc[-1] - low.iloc[-1]))
        return float(val)

    def _rsi(self, close: pd.Series, period: int = 14) -> float:
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        val = rsi.iloc[-1]
        return float(val) if not pd.isna(val) else 50.0

    def _macd_hist(self, close: pd.Series) -> float:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = (macd - signal).iloc[-1]
        return float(hist) if not pd.isna(hist) else 0.0
