# CLAUDE.md

## Run
```bash
bash ~/cryptobot_v3/start.sh          # interactive (spot=1, futures=2)
bash ~/cryptobot_v3/start.sh 1        # spot
bash ~/cryptobot_v3/start.sh 2        # futures
pkill -f 'cryptobot_v3'               # stop
screen -r cryptobot_v3_spot           # attach (Ctrl+A D to detach)
```
Launcher: auto-restart with exponential backoff, circuit breaks at 10 crashes, Telegram crash alerts.

Dont assume things. always use plugin superpowers before writing any code also make use of writing-plans and executing-plans

## File Layout
```
~/cryptobot_v3/
├── bot/
│   ├── launcher.py, base_bot.py, spot_bot.py, futures_bot.py
│   ├── ai_strategy.py   ← RF+LightGBM+LSTM+TFT ensemble + OOF meta-model
│   ├── agents.py        ← Groq confidence gate (called only if ≥2 models agree + conf≥0.50)
│   ├── risk_manager.py  ← ATR stops, Kelly sizing, circuit breaker, correlation filter
│   ├── regime_model.py  ← 4-state Gaussian HMM (TRENDING/RANGING/HIGH_VOL/CRASH)
│   ├── rl_agent.py      ← DQN trade manager (open positions only)
│   ├── binance_demo.py  ← direct Binance Demo HTTP client (no ccxt)
│   └── env_config.py    ← config loader + exchange factory
├── config_spot.yaml / config_futures.yaml
├── data/                ← model files (.pkl/.keras/.pt) + runtime JSON state
└── logs/                ← spot_bot.log, futures_bot.log
```

## Key Config (`config_*.yaml`)
| Key | Default | Notes |
|---|---|---|
| `strategy.min_confidence` | 0.42 | SelfLearner auto-tunes ±0.05 at runtime |
| `risk.take_profit_pct` | 0.02 | |
| `risk.stop_loss_atr_multiplier` | 2.0 | |
| `risk.leverage` | 5 | Futures only |
| `risk.max_open_trades` | 20 | |
| `risk.max_portfolio_heat` | 0.6 | |
| `scanner.top_n` | 15 | Watchlist size |

## Architecture

**Inheritance**: `BaseBot` holds 100% of strategy. `SpotBot`/`FuturesBot` override only 5 methods: `_setup_exchange`, `_place_buy`, `_place_sell`, `_place_close`, `_calc_pnl`.

**Signal flow per cycle**: `sync_with_exchange → check_exits → _rl_manage_trades → regime_gate → for each symbol: fetch_ohlcv → SIGNAL(ensemble+agents) → CONTEXT(HTF/BTC/HMM) → EXECUTION(filters+sizing+place) → RISK(can_open_trade)`

**ML ensemble** (weighted vote → meta-model stacker):
- RF 25%, LightGBM 35%, LSTM 40% (bidirectional, seq=20), TFT 15%/35%-TRENDING (seq=30)
- Meta: LogisticRegression on 5-fold OOF predictions (15 features: 3×RF+3×LGBM+3×LSTM+3×TFT+3×context)
- Features: 66 total — RSI/MACD/BB/ATR/EMA + vol_expansion/vol_delta/liq_sweep_up/down/htf_bull/bear/htf_align
- Labels: 3-class BUY/HOLD/SELL, ATR-dynamic threshold `(ATR/close×0.5).clip(0.001,0.02)`
- MC Dropout: 15 samples, HOLD if uncertainty > 0.03
- Online learning gated: `total_trades≥50 AND win_rate≥52%`

**HMM**: adjusts `min_conf_delta` and `size_mult` only — never vetoes signals. Smoothed (3 consecutive same state to change).

**DQN**: HOLD/SCALE_IN/SCALE_OUT/CLOSE on open positions. Returns `(action, confidence)` — always unpack both. SCALE_IN blocked until 500 experiences.

**Binance Demo**: `demo-api.binance.com` (spot) / `demo-fapi.binance.com` (futures). Only `/api/v3/*` and `/fapi/v1/*` supported. Error -4046 ("no need to change margin type") is benign — logged DEBUG not ERROR.

## Key Design Rules
1. Decision hierarchy: SIGNAL → CONTEXT → EXECUTION → RISK. Context sets `action="HOLD"` (no early return).
2. RL never decides entries. `decide()` returns `(action, confidence)` — always unpack both.
3. HMM adjusts thresholds only, never overrides direction.
4. Uncertainty: `max(ensemble_var, mc_uncertainty)` — worst wins. HOLD above 0.03.
5. OOF meta: fresh base models per fold, min 400 bars, min 50 OOF rows.
6. Stale model files → delete them; bot auto-retrains next cycle.
