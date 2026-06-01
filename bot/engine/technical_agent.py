"""
TechnicalAgent — pure rule-based multi-indicator agent.

Replaces the old MLTechnicalAgent (which used RF/LightGBM predictions).
No ML model, no feature scaler, no training. Just standard indicators
computed live: RSI, MACD, EMA alignment, Bollinger position, momentum.

Produces buy/sell scores in the same AgentSignal format the ensemble
expects, so it drops into the existing EnsembleEngine unchanged.
"""

import logging
import numpy as np
import pandas as pd

from engine.smc_agent import AgentSignal

log = logging.getLogger("TechnicalAgent")


class TechnicalAgent:
    """Multi-timeframe technical confluence. Coin-agnostic, no ML."""

    def analyze(self, df: pd.DataFrame, profile) -> AgentSignal:
        if df is None or len(df) < 50:
            return AgentSignal("technical", 0, 0, 0, 0, reasoning="insufficient data")

        try:
            close = df["close"]
            high  = df["high"]
            low   = df["low"]

            buy_score  = 0.0
            sell_score = 0.0
            reasons    = []

            # ── 1. EMA alignment (trend) ±0.30 ────────────────────────
            ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
            ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
            price = float(close.iloc[-1])
            if price > ema20 > ema50:
                buy_score += 0.30
                reasons.append("EMA bull stack")
            elif price < ema20 < ema50:
                sell_score += 0.30
                reasons.append("EMA bear stack")

            # ── 2. RSI ±0.25 ──────────────────────────────────────────
            rsi = self._rsi(close, 14)
            if rsi < 35:
                buy_score += 0.25
                reasons.append(f"RSI oversold {rsi:.0f}")
            elif rsi > 65:
                sell_score += 0.25
                reasons.append(f"RSI overbought {rsi:.0f}")
            elif 45 <= rsi <= 55:
                pass  # neutral
            elif rsi < 50:
                buy_score += 0.10
            else:
                sell_score += 0.10

            # ── 3. MACD histogram ±0.25 ───────────────────────────────
            macd_hist = self._macd_hist(close)
            if macd_hist > 0:
                buy_score += 0.25 if macd_hist > abs(close.iloc[-1]) * 0.0005 else 0.12
                reasons.append("MACD+")
            elif macd_hist < 0:
                sell_score += 0.25 if abs(macd_hist) > abs(close.iloc[-1]) * 0.0005 else 0.12
                reasons.append("MACD-")

            # ── 4. Momentum (5-bar rate of change) ±0.20 ──────────────
            if len(close) >= 6:
                roc = (price - float(close.iloc[-6])) / (float(close.iloc[-6]) + 1e-9)
                if roc > 0.01:
                    buy_score += 0.20
                    reasons.append(f"mom+{roc:.1%}")
                elif roc < -0.01:
                    sell_score += 0.20
                    reasons.append(f"mom{roc:.1%}")

            net_score  = max(-1.0, min(1.0, buy_score - sell_score))
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
