# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Download from VM

From your **local machine**, run one of these to copy the full project off the VM:

```bash
# scp (replace <VM_IP> with your server's IP)
scp -r root@<VM_IP>:/root/cryptobot_v3 ./cryptobot_v3

# rsync (faster for large dirs, skips unchanged files)
rsync -avz --exclude 'venv/' --exclude '__pycache__/' \
  root@<VM_IP>:/root/cryptobot_v3 ./cryptobot_v3
```

The `rsync` version skips the `venv/` and `__pycache__/` directories to keep the download small. Recreate the venv on the new machine with:

```bash
cd cryptobot_v3
python3 -m venv venv
venv/bin/pip install -r requirements.txt   # if requirements.txt exists
# or: venv/bin/pip install ccxt lightgbm tensorflow groq python-telegram-bot pyyaml
```

## Running the Bot

```bash
# Interactive mode (prompts for spot or futures)
cd ~/cryptobot_v3 && bash start.sh

# Non-interactive
bash start.sh 1   # spot
bash start.sh 2   # futures

# Direct (no screen session)
cd ~/cryptobot_v3/bot && BOT_MODE=spot python3 launcher.py
cd ~/cryptobot_v3/bot && BOT_MODE=futures python3 launcher.py
```

The launcher wraps the bot with auto-restart (exponential backoff, circuit breaks at 10 consecutive crashes) and Telegram crash alerts.

**Screen sessions**: `screen -r cryptobot_v3_spot` or `screen -r cryptobot_v3_futures` (Ctrl+A D to detach)

**Stop**: `pkill -f 'cryptobot_v3'`

## File Layout

```
~/cryptobot_v3/
├── bot/                    ← this directory (working dir)
│   ├── launcher.py         ← entry point with crash-restart loop
│   ├── base_bot.py         ← ALL shared trading logic
│   ├── spot_bot.py         ← 5-method subclass for spot
│   ├── futures_bot.py      ← 5-method subclass for futures
│   ├── ai_strategy.py      ← RF + LightGBM + LSTM ensemble
│   ├── agents.py           ← multi-agent system (Groq confidence gate)
│   ├── risk_manager.py     ← ATR stops, Kelly sizing, circuit breaker
│   ├── data_feed.py        ← OHLCV caching + live price polling
│   ├── coin_scanner.py     ← autonomous watchlist (top 50 → top N)
│   ├── self_learner.py     ← Groq-driven param tuning every 2h
│   ├── notifier.py         ← Telegram reports + command handler
│   ├── binance_demo.py     ← custom Binance Demo API client (no ccxt)
│   └── env_config.py       ← config loader, exchange factory
├── config_spot.yaml        ← spot bot config (read by BaseBot.__init__)
├── config_futures.yaml     ← futures bot config
├── .env                    ← API keys (not in repo)
├── data/                   ← runtime state (JSON + model files)
└── logs/                   ← rotating logs (spot_bot.log, futures_bot.log)
```

## Configuration

Config files live at `~/cryptobot_v3/config_spot.yaml` and `config_futures.yaml`. Key tunable params:

| Key | Default | Effect |
|---|---|---|
| `strategy.min_confidence` | 0.42 | Minimum ML confidence to open a trade |
| `risk.take_profit_pct` | 0.02 | Take profit threshold |
| `risk.stop_loss_atr_multiplier` | 2.0 | ATR multiplier for trailing stop |
| `risk.leverage` | 5 | Futures leverage (futures only) |
| `risk.max_open_trades` | 20 | Max simultaneous positions |
| `risk.max_portfolio_heat` | 0.6 | Max fraction of balance in open trades |
| `scanner.top_n` | 15 | Watchlist size |
| `scanner.rescan_hours` | 4 | How often CoinScanner rescans |
| `bot.scan_interval_seconds` | 30 | Main loop sleep |

The `SelfLearner` auto-adjusts `min_confidence` within ±0.05 based on trade results — the config file will change at runtime.

## Architecture

### Inheritance: BaseBot → SpotBot / FuturesBot

`BaseBot` contains 100% of the trading strategy. `SpotBot` and `FuturesBot` only override five methods:

```python
_setup_exchange()  # spot vs futures demo endpoint
_place_buy()       # market buy vs open long
_place_sell()      # market sell vs open short
_place_close()     # sell vs reduceOnly close
_calc_pnl()        # (close-entry)*amount vs direction-aware
```

Everything else — ML models, agent system, risk management, state, scanning — is identical.

### Signal Flow (per scan cycle)

```
BaseBot.run_once()
  ├── sync_with_exchange()          # reconcile state with Binance positions (futures only)
  ├── check_exits()                 # ATR trailing stop + take profit for all open trades
  ├── risk.detect_market_regime()   # BTC 4h + breadth → regime gate (15-min cache)
  └── for each symbol:
        DataFeed.fetch_multi_timeframe()   # 1h/4h/1d OHLCV (cached per-candle-duration)
        AIStrategyEngine.predict()         # RF(25%) + LGBM(35%) + LSTM(40%) weighted vote
        AgentCoordinator.analyze()         # confidence gate → Groq or ML-only
        HTF bias filter (4h/1d EMA)
        RiskManager.can_open_trade()       # circuit breaker, correlation, portfolio heat
        RiskManager.get_position_size()    # Kelly Criterion + regime multiplier
        _place_buy() / _place_sell()
```

### ML Models (`ai_strategy.py`)

- **Random Forest** (25% weight): walk-forward 5-fold, 500 trees, champion/challenger — keeps old model if new is >2% worse
- **LightGBM** (35% weight): 1000 estimators, early stopping, same champion/challenger logic
- **LSTM** (40% weight): bidirectional, seq_len=20, hard floor at 35% accuracy — discards below-floor models; backs up to `lstm_model_backup.keras` before overwriting
- Labels: 3-class (BUY=2, HOLD=1, SELL=0), `forward_bars=2`, `min_move=0.002`
- Ensemble: requires ≥2 models to agree for BUY or SELL; otherwise HOLD
- Models saved to `~/cryptobot_v3/data/` as `.pkl` / `.keras`
- Retrained daily + whenever watchlist changes

### Agent System (`agents.py`)

`AgentCoordinator` has a **confidence gate**: Groq (Llama 3.1 8B) is only called when ≥2 ML models agree AND ML confidence ≥ 0.50. This cuts Groq calls by ~90%.

- **Groq cache**: 30-min per symbol
- **Slow cache** (Fear/Greed + Macro): 2h
- **Daily token budget**: 90,000 tokens; falls back to ML-only when exhausted
- Budget and agent accuracy tracked in `~/cryptobot_v3/data/token_budget.json` and `agent_performance.json`

### Risk System (`risk_manager.py`)

| Component | Purpose |
|---|---|
| `MarketRegimeGate` | BTC 4h regime (CRASH/HIGH_VOL/STRONG_TREND/RANGING/WEAK_TREND) — gates new entries and adjusts `min_conf` and `size_mult` |
| `ATRTrailingStop` | Trailing stop tracks peak price; persisted in `trailing_stops.json` |
| `KellyCriterionSizer` | 2–10% of balance per trade, scaled by Kelly fraction + regime + losing streak |
| `PortfolioHeatTracker` | Blocks new entries when open exposure > `max_portfolio_heat` |
| `CircuitBreaker` | Stops trading if daily loss > 5% or 10 consecutive losses; resets at midnight |
| `CorrelationFilter` | Limits e.g. 1 position per large-cap, 1 per meme group |

### State Files

All runtime state is in `~/cryptobot_v3/data/`:

| File | Contents |
|---|---|
| `state.json` / `state.backup.json` | Spot trades, signals, cumulative stats |
| `futures_state.json` / `futures_state.backup.json` | Futures trades |
| `rf_model.pkl`, `lgbm_model.pkl`, `lstm_model.keras` | Trained ML models |
| `lstm_model_backup.keras` | LSTM backup before each overwrite |
| `scanner_cache.json` | Cached watchlist from last scan |
| `circuit_breaker.json` | Daily PnL + consecutive loss counter |
| `trailing_stops.json` | ATR peak prices per open trade |
| `token_budget.json` | Groq daily token usage |
| `agent_performance.json` | Agent prediction accuracy |
| `learning_insights.json` | Self-learner review history |
| `trading_paused.json` | Dashboard pause flag |

### Binance Demo Client (`binance_demo.py`)

ccxt is NOT used. The demo endpoints (`demo-api.binance.com` for spot, `demo-fapi.binance.com` for futures) only support basic `/api/v3/*` and `/fapi/v1/*` paths. `BinanceDemoClient` is a minimal direct implementation. `DemoExchangeAdapter` wraps it to expose the ccxt-style interface that the rest of the bot expects.

### Telegram Commands

The notifier thread polls for commands in both private and group chats (group IDs start with `-`):

`/status` `/pnl` `/trades` `/agents` `/health` `/mode` `/switch_spot` `/switch_futures` `/restart` `/stop` `/help`

`/switch_*` and `/restart` use `screen` + `pkill` to restart the bot in-process.

### Self-Learner (`self_learner.py`)

Runs every 2 hours if ≥10 closed trades exist. Sends performance stats to Groq and applies returned adjustments to `min_confidence` in both config files (clamped to 0.42–0.70). Requires statistical significance (|z-score| > 1.65) before applying changes.
