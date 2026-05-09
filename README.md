# CryptoBot v3

An autonomous cryptocurrency trading bot for Binance Demo, supporting both spot and futures modes. Features an ML ensemble, multi-agent signal aggregation, dynamic risk management, and a live web dashboard.

## Quickstart

```bash
bash ~/cryptobot_v3/start.sh          # interactive menu
bash ~/cryptobot_v3/start.sh 1        # spot mode
bash ~/cryptobot_v3/start.sh 2        # futures mode
pkill -f 'cryptobot_v3'               # stop all instances
screen -r cryptobot_v3_spot           # attach to screen session
```

Dashboard: `http://localhost:5002`

## How It Works

Each scan cycle runs:

1. **Sync** open positions with exchange
2. **Check exits** — ATR stops, take-profit, trailing stops
3. **DQN** manages open positions (HOLD / SCALE\_IN / SCALE\_OUT / CLOSE)
4. **Regime gate** — filters signals based on BTC 4h market breadth
5. **EnsembleEngine** per symbol:
   - SMCAgent 35% — liquidity sweeps, BOS/CHOCH, FVG
   - TechnicalAgent 40% — RSI/MACD/BB/EMA multi-timeframe
   - MacroAgent 25% — funding rates, flow imbalance
6. **RiskDecisionAgent** — correlation filter + portfolio heat check
7. **ExecutionEngine** — order placement with fill confirmation

## ML Stack

- **RF + LightGBM** ensemble with walk-forward validation; champion/challenger model promotion
- **4-state HMM** (TRENDING / RANGING / HIGH\_VOL / CRASH) adjusts confidence thresholds and position size multipliers
- **DQN** reinforcement learning agent manages open positions only; falls back to HOLD on failure

## Trading Profiles

| Profile | Min Confidence | Position Size | Entry Style |
|---|---|---|---|
| STRICT | 0.70 | 1.5% | FVG entry, macro alignment required |
| BALANCED | 0.42 | 2.5% | Retracement entry, soft HTF filter |
| AGGRESSIVE | 0.40 | 4.0% | Market entry, no funding filter |

Set via `strategy.trading_profile` in `config_spot.yaml` / `config_futures.yaml`.

## Key Config

| Key | Default | Notes |
|---|---|---|
| `risk.take_profit_pct` | 0.02 | |
| `risk.stop_loss_atr_multiplier` | 2.0 | |
| `risk.leverage` | 5 | Futures only |
| `risk.max_daily_loss_pct` | 0.05 | Circuit breaker |
| `risk.max_portfolio_heat` | 0.6 | |
| `scanner.top_n` | 40 (spot) / 15 (futures) | Watchlist size |
| `ml.retrain_hours` | 24 | |
| `bot.scan_interval_seconds` | 30 | Main loop cadence |

## Telegram Commands

`/status` `/pnl` `/trades` `/agents` `/health` `/help`

## Exchange

Runs against **Binance Demo** (`demo-api.binance.com` for spot, `demo-fapi.binance.com` for futures). No real funds are used.
