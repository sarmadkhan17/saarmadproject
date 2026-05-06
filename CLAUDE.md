# CLAUDE.md

> Developer reference for Claude Code. Do not assume — always use the superpowers plugin before writing any code. Use `writing-plans` + `executing-plans` for non-trivial tasks.

## Run
```bash
bash ~/cryptobot_v3/start.sh          # interactive menu (spot=1, futures=2)
bash ~/cryptobot_v3/start.sh 1        # spot mode
bash ~/cryptobot_v3/start.sh 2        # futures mode
pkill -f 'cryptobot_v3'               # stop all instances
screen -r cryptobot_v3_spot           # attach (Ctrl+A D to detach)
screen -r cryptobot_v3_futures
```
Launcher: exponential backoff restart, circuit breaks at 10 crashes, Telegram crash alerts.

Dashboard: `http://localhost:5002`

## File Layout
```
~/cryptobot_v3/
├── bot/
│   ├── launcher.py                  ← entry point, auto-restart with backoff
│   ├── env_config.py                ← config loader + exchange factory
│   ├── agents/coordinator.py        ← agent performance tracker + confidence gate
│   ├── core/config.py, types.py     ← shared config loader + Trade dataclass
│   ├── data/feed.py, ws_feed.py     ← OHLCV data feed + WebSocket price feed
│   ├── engine/
│   │   ├── bot.py                   ← BaseBot: StateManager, Trade, main loop
│   │   ├── ensemble.py              ← parallel agent aggregator (SMC+Tech+Macro)
│   │   ├── execution_engine.py      ← order placement + fill confirmation
│   │   ├── futures.py               ← FuturesBot (long/short overrides)
│   │   ├── profiles.py              ← TradingProfile presets (STRICT/BALANCED/AGGRESSIVE)
│   │   ├── risk_agent.py            ← risk decision layer
│   │   ├── smc_agent.py             ← Smart Money Concepts agent
│   │   └── spot.py                  ← SpotBot (buy/sell overrides)
│   ├── exchange/demo_api.py         ← Binance Demo HTTP client (no ccxt)
│   ├── features/
│   │   ├── pipeline.py              ← multi-asset Parquet training dataset builder
│   │   └── feature_builder.py       ← feature engineering helpers
│   ├── models/
│   │   ├── ai_strategy.py           ← RF+LightGBM ensemble + walk-forward validation
│   │   ├── gnn.py                   ← GNN correlation filter
│   │   ├── hmm.py                   ← 4-state Gaussian HMM regime model
│   │   ├── lgbmmodel.py             ← LightGBM wrapper
│   │   ├── rfmodel.py               ← Random Forest wrapper
│   │   └── rl_agent.py              ← DQN trade manager (HOLD/SCALE_IN/SCALE_OUT/CLOSE)
│   ├── notify/telegram.py           ← Telegram notifier + command polling
│   ├── risk/manager.py              ← ATR stops, Kelly sizing, circuit breaker, correlation filter
│   └── tuning/
│       ├── learner.py               ← SelfLearner (Groq LLM, min 10 trades to activate)
│       └── scanner.py               ← autonomous coin scanner (top N by 24h volume)
├── dashboard/
│   ├── server.py                    ← Flask/FastAPI dashboard backend
│   └── static/index.html            ← dark-themed CryptoBot v4 UI
├── config_spot.yaml                 ← spot bot configuration
├── config_futures.yaml              ← futures bot configuration
├── data/                            ← model files (.pkl/.pt) + runtime JSON state
│   ├── spot/ futures/               ← per-mode model artifacts
│   └── training/                    ← per-coin Parquet training data
└── logs/                            ← spot_bot.log, futures_bot.log (rotating, 10MB×5)
```

## Key Config (`config_*.yaml`)
| Key | Spot | Futures | Notes |
|---|---|---|---|
| `strategy.min_confidence` | 0.42 | 0.42 | SelfLearner auto-tunes ±0.05 |
| `strategy.trading_profile` | BALANCED | BALANCED | STRICT / BALANCED / AGGRESSIVE |
| `risk.take_profit_pct` | 0.02 | 0.02 | |
| `risk.stop_loss_atr_multiplier` | 2.0 | 2.0 | |
| `risk.leverage` | — | 5 | Futures only |
| `risk.max_open_trades` | 20 | 20 | |
| `risk.max_portfolio_heat` | 0.6 | 0.6 | |
| `risk.max_daily_loss_pct` | 0.05 | 0.05 | circuit breaker |
| `scanner.top_n` | 40 | 15 | watchlist size |
| `scanner.min_volume_usdt` | 30M | 50M | |
| `scanner.rescan_hours` | 2 | 4 | |
| `ml.retrain_hours` | 24 | 24 | |
| `bot.scan_interval_seconds` | 30 | 30 | main loop cadence |

## Architecture (v4/v5)

**Inheritance**: `BaseBot` (engine/bot.py) holds 100% of strategy logic. `SpotBot`/`FuturesBot` override only 5 methods: `_setup_exchange`, `_place_buy`, `_place_sell`, `_place_close`, `_calc_pnl`.

**Signal flow per cycle**:
```
sync_with_exchange
  → check_exits (ATR stops, take-profit, trailing)
  → rl_manage_trades (DQN on open positions)
  → regime_gate (MarketRegimeGate: BTC 4h breadth/ADX)
  → for each symbol:
      fetch_ohlcv
      → EnsembleEngine.run(SMCAgent + TechnicalAgent + MacroAgent)
      → RiskDecisionAgent (correlation filter + portfolio heat)
      → ExecutionEngine.place_with_confirmation
```

**EnsembleEngine** (engine/ensemble.py) — parallel agent aggregation:
- `SMCAgent` 35% — liquidity sweeps, BOS/CHOCH, FVG, volume spikes
- `TechnicalAgent` 40% — RSI/MACD/BB/EMA multi-timeframe
- `MacroAgent` 25% — funding rates, flow imbalance (optional)
- Weighted net score → consensus BUY/SELL/HOLD + confidence

**ML ensemble** (models/ai_strategy.py) — walk-forward validated:
- RF + LightGBM classifiers (3-class: BUY/HOLD/SELL)
- ATR-dynamic labels: `(ATR/close×atr_k).clip(0.001, 0.02)`
- Features: 60+ — returns, EMA slopes, RSI, MACD, BB, ATR, vol ratios
- Champion/challenger: new model promoted only if it beats the current one

**Trading Profiles** (engine/profiles.py):
| Profile | `min_confidence` | `min_agent_agreement` | `position_size_pct` | Notes |
|---|---|---|---|---|
| STRICT | 0.70 | 3 | 1.5% | FVG entry required, macro alignment required |
| BALANCED | 0.42 | 1 | 2.5% | Retracement entry, soft HTF filter |
| AGGRESSIVE | 0.40 | 1 | 4.0% | Market entry allowed, no funding filter |

**HMM** (models/hmm.py): 4-state Gaussian HMM → TRENDING/RANGING/HIGH_VOL/CRASH.
Adjusts `min_conf_delta` and `size_mult` only — never blocks signals. Smoothed: 3 consecutive same state to change.
| State | `min_conf_delta` | `size_mult` |
|---|---|---|
| TRENDING | −0.03 | 1.20× |
| RANGING | +0.01 | 0.90× |
| HIGH_VOL | +0.05 | 0.60× |
| CRASH | +0.08 | 0.40× |

**DQN** (models/rl_agent.py): manages OPEN positions only. Actions: HOLD/SCALE_IN/SCALE_OUT/CLOSE. 8-dim state (pnl_pct, volatility, time_in_trade, regime×4, momentum_1h). SCALE_IN blocked until 500 replay experiences. Always falls back to HOLD on failure.

**RiskManager** (risk/manager.py):
- ATR-based dynamic stop loss
- Kelly Criterion position sizing
- Portfolio heat tracking (`max_portfolio_heat`)
- Correlation filter: max 1 large-cap, 2 layer-1, 1 DeFi, etc.
- Circuit breaker: halts trading after `max_consecutive_losses` or `max_daily_loss_pct`

**SelfLearner** (tuning/learner.py): reviews trade results every 2h using Groq LLM (llama-3.1-8b-instant). Requires ≥10 closed trades. Adjusts `min_confidence` ±0.05 and logs insights to `data/learning_insights.json`.

**Binance Demo**: `demo-api.binance.com` (spot) / `demo-fapi.binance.com` (futures). Only `/api/v3/*` and `/fapi/v1/*` supported. Error −4046 ("no need to change margin type") is benign — log at DEBUG not ERROR.

## Key Design Rules
1. Decision hierarchy: SIGNAL → CONTEXT → EXECUTION → RISK. Context sets `action="HOLD"` — no early return.
2. DQN never decides entries. `decide()` returns `(action, confidence)` — always unpack both.
3. HMM adjusts thresholds only, never overrides signal direction.
4. Stale model files → delete them; bot auto-retrains next cycle.
5. State is persisted atomically (write to `.tmp.json` then `replace()`). Spot and Futures use separate state files.
6. `BOT_MODE` env var (`spot` | `futures`) controls which config and data directory is active.
7. SpotBot and FuturesBot share identical strategy — only the 5 exchange-interaction methods differ.

## Telegram Commands
`/status` `/pnl` `/trades` `/agents` `/health` `/help`
Supports private chats and group chats (negative chat IDs).
