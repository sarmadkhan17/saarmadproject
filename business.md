# CryptoBot v4 — Business Overview

## What It Is

CryptoBot v4 is an autonomous algorithmic trading bot for cryptocurrency markets. It runs 24/7, selects its own coin watchlist, generates ML-driven trade signals, manages positions with dynamic risk controls, and continuously improves from its own trade results — all without manual intervention.

Currently operates on **Binance Demo** (paper trading). Designed to migrate to live exchange credentials by swapping one environment variable.

---

## Trading Modes

| Mode | Direction | Leverage | Min Volume Filter |
|---|---|---|---|
| Spot | BUY / SELL | 1× | $30M 24h volume |
| Futures | LONG / SHORT | 5× (configurable) | $50M 24h volume |

Both modes run the identical strategy engine. Only order placement differs.

---

## How It Generates Signals

Signals come from a **3-agent ensemble** that runs in parallel on every symbol every 30 seconds:

1. **SMC Agent (35% weight)** — Smart Money Concepts: detects institutional footprints — liquidity sweeps, Break of Structure (BOS), Change of Character (CHOCH), Fair Value Gaps (FVG), and abnormal volume spikes.

2. **Technical Agent (40% weight)** — Classic technical analysis across multiple timeframes: RSI (7/14/21), MACD, Bollinger Bands, EMA alignment (9/21/50), ATR.

3. **Macro/Flow Agent (25% weight)** — Funding rates and order flow imbalance to detect leverage crowding.

The ensemble produces a weighted net score → BUY/SELL/HOLD with a confidence level. A **ML fallback layer** (Random Forest + LightGBM) provides an independent probability estimate that must align before a trade is entered.

---

## Market Regime Awareness

A **4-state Hidden Markov Model** classifies current market conditions every 15 minutes using BTC 4-hour data (breadth, volatility, ADX):

| Regime | Effect on Trading |
|---|---|
| TRENDING | Lower confidence threshold, larger position size (1.2×) |
| RANGING | Slightly tighter threshold, smaller size (0.9×) |
| HIGH_VOL | Much tighter threshold, reduced size (0.6×) |
| CRASH | Strictest threshold, minimal size (0.4×) |

The HMM never blocks trades — it only adjusts thresholds and sizing.

---

## Risk Management

Every potential trade passes through multiple independent risk gates:

- **Confidence gate**: signal must clear `min_confidence` (default 0.42, tuned automatically)
- **Agent agreement**: configurable minimum number of agents must agree
- **Correlation filter**: caps exposure by asset class (max 1 BTC/ETH, 2 Layer-1, 1 DeFi, etc.)
- **Portfolio heat**: total open exposure capped at 60% of capital
- **Position sizing**: Kelly Criterion-based, capped at 3% per trade
- **ATR stop loss**: dynamic stop at 2.0× ATR below/above entry
- **ATR take profit**: target at 2.5× ATR
- **Daily loss circuit breaker**: halts all new entries after 5% daily drawdown or 10 consecutive losses

Three risk profiles are available — `STRICT`, `BALANCED`, `AGGRESSIVE` — controlling all thresholds jointly.

---

## Autonomous Coin Selection

The **CoinScanner** autonomously selects the trading watchlist every 2–4 hours:

- Scans top Binance USDT pairs by 24h volume
- Filters out stablecoins and low-liquidity pairs
- Always includes market leaders (BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX, LINK, DOT)
- Spot: top 40 coins | Futures: top 15 coins

---

## Self-Improvement

The **SelfLearner** reviews closed trade results every 2 hours using a Groq LLM (Llama 3.1 8B):

- Requires ≥10 closed trades for statistical validity
- Adjusts `min_confidence` threshold up or down by ±0.05 based on win rate
- Logs all insights and adjustments to `data/learning_insights.json`
- The **DQN reinforcement learning agent** also improves over time by managing open positions (scale in/out/close decisions)

---

## ML Models

| Model | Role | Retrain Cadence |
|---|---|---|
| Random Forest | BUY/HOLD/SELL classifier | Every 24h |
| LightGBM | BUY/HOLD/SELL classifier | Every 24h |
| HMM | Market regime (4-state) | Per-cycle |
| DQN | Open position management | Online (continuous) |

Training uses multi-timeframe data (15m, 1h, 4h, 1d) across 15 coins pulled from Binance. Models are validated with walk-forward cross-validation. A champion/challenger system ensures only better models go live.

---

## Monitoring & Alerts

**Web Dashboard** (`http://localhost:5002`):
- Real-time P&L, open trades, win rate
- Agent performance breakdown
- Market regime indicator
- Trade history with entry/exit details
- Responsive — works on mobile and tablet

**Telegram Bot**:
- Trade entry/exit alerts with symbol, side, price, P&L
- Crash alerts with auto-restart notifications
- Interactive commands: `/status` `/pnl` `/trades` `/agents` `/health`
- Works in private chats and group channels

---

## Deployment

| Component | Detail |
|---|---|
| Exchange | Binance Demo (spot + futures) |
| Process manager | GNU Screen with auto-restart + exponential backoff |
| Circuit breaker | Stops after 10 consecutive crashes |
| Containerization | Docker + docker-compose available |
| VPS target | Contabo (deploy scripts included) |
| Logs | Rotating, 10MB × 5 files per mode |
| State | JSON with atomic writes + backup copy |

---

## Configuration Summary

All tunable parameters live in `config_spot.yaml` and `config_futures.yaml`. No code changes needed to adjust:

- Trading profile (risk appetite)
- Watchlist size and volume filters
- Confidence thresholds
- Position sizing and leverage
- Stop loss / take profit multipliers
- Retrain schedule
