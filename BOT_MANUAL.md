# CryptoBot v5 — Complete Technical Manual

> **Purpose of this document:** A reference for an AI agent (or human) to fully understand how this bot works — its architecture, decision logic, configuration, and operational behavior. No prior knowledge of the codebase is assumed.

---

## 1. Overview

CryptoBot v5 is an autonomous cryptocurrency trading bot that runs on **Binance Demo Trading** (paper money, no real capital at risk). It supports two independent modes:

- **Spot** — BUY/SELL only, no leverage, configured via `config_spot.yaml`
- **Futures** — LONG/SHORT with leverage (default 5×), configured via `config_futures.yaml`

Both modes share 100% of the strategy logic. The only differences are order direction (buy vs open-long), PnL calculation (simple vs leveraged), and exchange connection endpoint.

### What makes v5 different from earlier versions

v5 removed an entire ML stack that failed in backtesting and live trading:
- HMM regime model (always 3+ bars late)
- LightGBM + Random Forest direction predictors (overfit on noisy crypto)
- DQN reinforcement learning position manager (never converged)
- All model files, retraining pipelines, and feature engineering

**Replaced by:**
- Pure rule-based technical + structural agents
- DeepSeek LLM reasoning layer (Actor / Judge / Meta-Judge)
- CoinGecko macro dominance data
- Binance order book + CVD microstructure signals

---

## 2. Directory Structure

```
cryptobot_v5/
├── bot/
│   ├── launcher.py              # Entry point — crash-recovery loop + watchdog
│   ├── env_config.py            # Environment variable helpers
│   ├── core/
│   │   ├── config.py            # Exchange factory, secrets loader, paths
│   │   ├── types.py             # Trade dataclass
│   │   └── tz.py                # Timezone helper (UTC+3 local)
│   ├── data/
│   │   ├── feed.py              # DataFeed — OHLCV fetching + caching
│   │   ├── ws_feed.py           # Binance WebSocket price feed
│   │   └── freshness.py         # Data staleness checks
│   ├── engine/
│   │   ├── bot.py               # BaseBot — shared strategy for both modes
│   │   ├── spot.py              # SpotBot — overrides for buy/sell
│   │   ├── futures.py           # FuturesBot — overrides for long/short + leverage
│   │   ├── profiles.py          # TradingProfile presets (STRICT/BALANCED/AGGRESSIVE/CONFLUENCE)
│   │   ├── ensemble.py          # EnsembleEngine — aggregates 3 agent votes
│   │   ├── smc_agent.py         # SMCAgent — Smart Money Concepts analysis
│   │   ├── technical_agent.py   # TechnicalAgent — RSI/MACD/EMA/momentum
│   │   ├── macro_agent.py       # MacroFlowAgent — BTC.D direction signal
│   │   ├── trend_filter.py      # TrendFilter — two-tier HTF EMA veto
│   │   ├── risk_agent.py        # RiskDecisionAgent — position sizing + approval gates
│   │   ├── execution_engine.py  # ExecutionEngine — order placement + SL management
│   │   └── correlation_check.py # CorrelationCheck — pass-through for group caps
│   ├── agents/
│   │   ├── coordinator.py       # AgentCoordinator — legacy wrapper (win-rate tracking)
│   │   ├── macro_context.py     # MacroContextAgent — CoinGecko BTC.D/USDT.D kill switch
│   │   ├── microstructure.py    # MicrostructureAgent — order book + CVD + funding
│   │   ├── trade_memory.py      # TradeMemory — SQLite store + RAG retrieval
│   │   ├── vector_store.py      # VectorStore — semantic embedding for RAG
│   │   ├── llm_reasoning.py     # DeepSeek Actor / Judge / Meta-Judge
│   │   └── usage_tracker.py     # DeepSeek token cost tracking
│   ├── risk/
│   │   └── manager.py           # RiskManager — circuit breaker, Kelly sizing, ATR stops, correlation filter, regime gate
│   ├── notify/
│   │   ├── telegram.py          # TelegramNotifier — alerts + command handler
│   │   └── nl_ops.py            # Natural-language ops via DeepSeek
│   └── tuning/
│       └── scanner.py           # CoinScanner — autonomous coin selection
├── dashboard/
│   ├── server.py                # Flask API (port 5002)
│   └── static/index.html        # Web UI
├── data/                        # Runtime state files (JSON, SQLite)
├── logs/                        # Rotating log files
├── scripts/
│   ├── signal_monitor.py        # Offline signal analysis tool
│   └── system_auditor.py        # Health audit tool
├── config_spot.yaml
├── config_futures.yaml
└── .env                         # Secrets (API keys, Telegram token)
```

---

## 3. Entry Point and Process Management

### Launch

```bash
cd bot
BOT_MODE=futures python3 launcher.py   # Futures mode
BOT_MODE=spot    python3 launcher.py   # Spot mode
# If BOT_MODE is unset, the launcher prompts interactively.
```

### `launcher.py` responsibilities

1. **Mode selection** — reads `BOT_MODE` env var or prompts the user
2. **Crash-recovery loop** — restarts the bot on any uncaught exception with exponential back-off (30s → 300s cap); stops after 10 consecutive crashes
3. **Telegram crash/recovery alerts** — sends messages on each crash, recovery, and final give-up
4. **Watchdog thread** — daemon thread that reads `data/bot_heartbeat_<mode>.json`; if the heartbeat goes stale beyond `WATCHDOG_STALE_SECONDS` (default 240s) it `os.execv`-restarts the entire process. This catches silent hangs where no exception is raised.

---

## 4. The 7-Gate Decision Pipeline

Each scan cycle, every watched symbol passes through these gates **in order**. Failing any gate returns immediately and the symbol is skipped for this cycle.

```
Gate 0/1  Macro kill + universe filter   (MacroContextAgent)
   ↓
Layer 1   Ensemble vote                  (SMC + Technical + MacroFlow)
   ↓
Gate 4    Microstructure confirm/kill    (MicrostructureAgent)
   ↓
Gate 5    DeepSeek Actor reasoning       (LLM with RAG)
   ↓
Gate 6    Risk decision + sizing         (RiskDecisionAgent + Kelly)
   ↓
          Execute order on Binance Demo
   ↓
          On close → Judge (DeepSeek R1)
          Every 20 closes → Meta-Judge (R1) → update rules
```

**Note:** "Gate 2" (regime gate) and "Gate 3" (ensemble minimum) are handled inside Layer 1's ensemble aggregation and trend-filter veto, not as separate named steps.

---

## 5. Component Reference

### 5.1 MacroContextAgent (`agents/macro_context.py`)

**Source:** CoinGecko `/api/v3/global` — polled every 5 minutes, cached.

**Computes:**
- `btc_d` — Bitcoin dominance %
- `usdt_d` — Tether dominance %
- `btc_d_roc` / `usdt_d_roc` — rate of change per hour over rolling 1-hour history

**Kill conditions (block ALL new entries):**
- `usdt_d_roc > 0.4%/hr` → capital fleeing to stablecoins
- `btc_d_roc > 0.8%/hr` → altcoin sell-off in progress

**Universe rules (determines which coins are eligible):**
| BTC.D | BTC price | Universe |
|-------|-----------|----------|
| Rising | Rising | `btc_only` — BTC pairs only |
| Falling | Rising | `full` — all alts open |
| Rising | Falling | `defensive` — shorts only, reduce exposure |
| USDT.D falling | — | `full` — risk-on |

---

### 5.2 MarketRegimeGate (`risk/manager.py` → `MarketRegimeGate`)

**Source:** BTC/USDT 4h OHLCV + up to 8 watchlist coins. Cached 15 minutes.

**Computes:**
- `adx` — 14-period ADX on BTC 4h
- `breadth` — % of watchlist coins above their 20-period EMA (bull breadth)
- `bear_breadth` — % of watchlist coins below their 20-period EMA
- `vol_ratio` — recent 8-bar volatility / 40-bar baseline
- `regime` — one of: `TRENDING_BULLISH`, `TRENDING_BEARISH`, `RANGING`, `HIGH_VOLATILITY`, `CRASH`
- `trend_direction` — `BULLISH` / `BEARISH` / `NEUTRAL`
- `gate` — `OPEN` (trade) / `CLOSED` (choppy — no new entries)

**Hard block:** `vol_ratio < 0.25` in ensemble → HOLD (price discovery unreliable)

---

### 5.3 EnsembleEngine (`engine/ensemble.py`)

Runs 3 agents in parallel via `ThreadPoolExecutor`, aggregates their votes into a single directional decision.

**Agents and base weights:**
| Agent | Base weight | Description |
|-------|-------------|-------------|
| `smc` | 0.35 | Smart Money Concepts structural analysis |
| `technical` | 0.40 | RSI / MACD / EMA / momentum rules |
| `macro_flow` | 0.25 | BTC.D directional signal |

**Regime-adaptive weight routing:**
- `TRENDING` regime → technical promoted to 0.50, SMC to 0.28
- `RANGING` regime → SMC promoted to 0.50, technical to 0.30
- `HIGH_VOL / CRASH` → macro promoted to 0.40, technical 0.35

**Aggregation:**
1. Weighted `net_score` = Σ(agent.net_score × weight) / Σ(weights)
2. **Counter-trend reduction** — if BTC trend is BULLISH and net < 0 (short signal): multiply net by 0.70. Likewise for BEARISH + long. (Single 30% reduction — not the old 96.5% that caused permanent HOLD.)
3. **Confidence decay** — low vol (×0.75), conflicting agents (×0.80), weak ADX < 15 (×0.85)
4. **Dynamic threshold** — `threshold = max(0.03, base_threshold × (1 - direction_conviction × 0.25))`
5. **Action** — `BUY` if net > threshold, `SELL` if net < -threshold, else `HOLD`

**Outputs (`EnsembleResult`):**
- `action` — BUY / SELL / HOLD
- `confidence` — 0.35–0.95 (variance-weighted, penalises agent disagreement)
- `net_score` — signed [-1, 1]
- `agents_agreeing` — count of agents aligned with final direction

---

### 5.4 TrendFilter (`engine/trend_filter.py`)

Two-tier EMA veto applied inside the ensemble after aggregation.

**Fast tier (15m):** EMA(20) / EMA(50), `slope_lookback=10`
**Slow tier (1h):** EMA(50) / EMA(200), `slope_lookback=20`

**Veto logic:**
- A LONG signal is vetoed if both tiers show downtrend
- A SHORT signal is vetoed if both tiers show uptrend
- "Strongly trending" = EMA_fast relative change > `strong_slope_pct` (2% over 20 bars on 1h) — mild grinds do NOT veto

---

### 5.5 SMCAgent (`engine/smc_agent.py`)

Rule-based Smart Money Concepts analysis. Produces `buy_score` and `sell_score` from 5 sub-checks:

| Check | Weight | Description |
|-------|--------|-------------|
| `sweep` | ±0.35 | Liquidity sweep detection (pivot-based) |
| `bos` | ±0.35 | Break of Structure / Change of Character |
| `fvg` | ±0.25 | Fair Value Gap (imbalance between candles) |
| `volume` | ±0.20 | Volume spike at key level |
| `pattern` | ±0.15 | Pattern completion (double top/bottom, etc.) |

Profile-controlled thresholds (e.g. `smc_liquidity_sweep_pct`, `smc_bos_body_pct`, `smc_volume_spike_ratio`, `smc_pattern_completion`, `smc_sub_checks_min`) determine how many sub-checks must fire and how strictly.

---

### 5.6 TechnicalAgent (`engine/technical_agent.py`)

Rule-based indicator confluence replacing the old ML predictor. Uses RSI overbought/oversold, MACD crossovers, EMA alignment, and momentum scoring. Produces the same `AgentSignal` dataclass as SMCAgent.

---

### 5.7 MicrostructureAgent (`agents/microstructure.py`)

**Does not generate direction.** Takes the ensemble's `action` as input and **confirms or kills it**.

**Order book imbalance** (top 20 levels of Binance L2):
- `bid_vol / ask_vol > 2.0` → bullish pressure (confirms BUY)
- `ask_vol / bid_vol > 2.0` → bearish pressure (kills BUY or confirms SELL)
- `> 2.5` overwhelming contra wall → hard veto regardless of CVD

**CVD (Cumulative Volume Delta)** over last 50 candles:
- Price new highs + CVD lower highs → CVD divergence → KILL long
- Price new lows + CVD higher lows → CVD divergence → KILL short
- Aligned → confirms direction

**Funding rate** (futures only): `|funding_rate| > 0.1%` per 8h → `funding_extreme=True` (soft warning to Actor, not a hard kill)

---

### 5.8 DeepSeek Reasoning Layer (`agents/llm_reasoning.py`)

Three distinct LLM calls, each with a different model and purpose:

#### Actor (DeepSeek V3 / `deepseek-chat`)
- Runs **every scan cycle** per symbol that passes Gates 0–4
- Receives: regime, trend direction, BTC.D/USDT.D, ensemble score, microstructure signal, 5 similar past trades from RAG, lessons from Judge critiques
- Computes recency-decayed + Bayesian-smoothed win-rate over similar trades (prevents freeze-loop from a bad session)
- Outputs: `approved` (bool), `confidence` (0–1), `reasoning`, `tp_note`, `sl_note`, `risk_flag`
- **Verdict cache:** identical quantized setup signature within `cache_ttl_seconds` (default 180s) reuses prior decision instead of re-calling the LLM

#### Judge (DeepSeek R1 / `deepseek-reasoner`)
- Runs **after every closed trade** in a background thread
- Receives: full entry context at trade time + outcome
- **No hindsight allowed:** evaluates decision quality as of entry, not exit
- Outputs: `decision_quality` (good/acceptable/poor), `entry_valid`, `risk_managed`, `missed_warnings`, `lesson`, `pattern_tag`
- Stored in `TradeMemory` (SQLite) and embedded for RAG

#### Meta-Judge (DeepSeek R1 / `deepseek-reasoner`)
- Runs **every 20 new critiques** since last run
- Receives: last 20 Judge critiques
- Synthesizes patterns across many trades into updated rules
- Outputs: `summary`, `updated_rules[]`, `avoid_patterns[]`, `favour_patterns[]`
- Rules are injected into the Actor's system prompt going forward
- Sends summary to Telegram

**Cost estimate:** Under $1/day total (V3 ≈ $0.50/day, R1 ≈ $0.07/day)

---

### 5.9 TradeMemory + RAG (`agents/trade_memory.py`, `agents/vector_store.py`)

**Storage:** SQLite at `data/trade_memory.db`

**Schema per trade:** `id, symbol, side, mode, entry_price, close_price, pnl, r_multiple, duration_hours, close_reason, regime, btc_d, usdt_d, ensemble_score, confidence, ob_imbalance, cvd_direction, cvd_divergence, actor_reasoning, judge_critique, pattern_tag, closed_at`

**RAG retrieval for Actor:**
- `find_similar()` — hybrid semantic (if embeddings available) + feature-similarity (cosine on numeric features) — returns top 5 similar past setups
- `find_similar_critiques()` — semantic search over Judge lessons — returns top 3 relevant warnings
- Feature vector includes: action, regime, ensemble score, confidence, BTC.D, USDT.D, OB imbalance, CVD direction/divergence

**Embeddings** (`vector_store.py`): uses `sentence-transformers` if available; silently falls back to feature-only matching if not installed.

---

### 5.10 RiskDecisionAgent (`engine/risk_agent.py`)

Sequential gates that must all pass for approval:

| Gate | Check |
|------|-------|
| -1 | Ensemble completeness (rejects if ≥2 agents errored) |
| A | Confidence ≥ `profile.min_confidence` |
| B | Agent agreement ≥ `profile.min_agent_agreement` |
| C | SMC sub-checks ≥ `profile.smc_sub_checks_min` |
| D | Net score threshold ≥ `profile.net_score_threshold` |
| E | Regime gate (`CHOPPY` → no new entries) |
| F | HTF bias (soft/hard/strict — based on `htf_filter_mode`) |
| G | BTC momentum filter (if enabled) |
| H | Correlation filter (group caps — no more than N of the same sector) |
| I | Maximum open trades cap |
| J | Portfolio heat < `max_portfolio_heat` (default 50%) |
| K | Circuit breaker (daily loss / drawdown / consecutive losses) |

**Position sizing** uses Kelly Criterion scaled by confidence and ATR:
- Base: 2% of balance per trade
- Max: 8% of balance per trade
- Scaled by `profile.stop_loss_atr_mult` (wider stop → smaller size)

---

### 5.11 ExecutionEngine (`engine/execution_engine.py`)

- `execute_entry()` — acquires per-symbol lock, places market order with 3-retry confirmation loop, places exchange-side stop-loss (futures only), records Trade in StateManager
- `_place_sl()` — places stop-limit SL on the exchange (futures); falls back if exchange rejects
- Returns `"SL_FAILED"` string if SL placement fails (triggers 2h cooldown for that symbol)

---

### 5.12 RiskManager / ExitEngine (`risk/manager.py`)

**check_exits()** is called at the start of every scan cycle, before any new entry analysis.

**Exit conditions (checked per open trade):**
| Condition | Action |
|-----------|--------|
| Price ≤ SL level | Full close (ATR-based stop loss) |
| Price ≥ TP1 (`tp1_r_mult × entry_atr` gain) | Partial close (`tp1_fraction` of position, default 40%), move SL to breakeven |
| Price ≥ full TP (`take_profit_atr_mult × ATR`) | Full close (fixed take profit) |
| Trailing stop triggered (post-TP1) | Full close (ATR trailing stop, `trail_atr_mult`) |
| Reversal signal (regime flip + EMA cross) | Early exit if `early_exit_enabled` |

After a TP1 partial close, the remaining position gets a new exchange-side SL placed at breakeven.

**CircuitBreaker** (stops all trading when tripped):
- Daily loss > `max_daily_loss_pct` (futures default 3%, spot 5%)
- Total drawdown > `max_drawdown_pct` (10%)
- Consecutive losses > `max_consecutive_losses` (futures 6, spot 10)

**KellyCriterionSizer** — sizes positions proportional to edge × confidence, capped by portfolio heat.

**CorrelationFilter** — prevents over-concentration in correlated asset groups:
```
btc_correlated (max 2): BTC, ETH, BNB, SOL, AVAX, XRP
large_cap (max 1):       BTC, ETH
layer1 (max 2):          SOL, ADA, AVAX, DOT
defi (max 1):            LINK, UNI, AAVE
meme (max 1):            DOGE, SHIB
layer2 (max 1):          MATIC, OP, ARB
```

---

### 5.13 CoinScanner (`tuning/scanner.py`)

Autonomously selects the trading watchlist. Re-scans every `rescan_hours` (futures: 4h, spot: 2h).

**Scoring criteria:**
- Minimum volume: 100M USDT/day (futures) / 30M (spot)
- Minimum price: $0.50
- Maximum daily volatility: 15%
- Blacklists stable coins (USDC, BUSD, TUSD, FDUSD)

**Always includes:** BTC, ETH, BNB, SOL, XRP, DOGE, ADA, AVAX, LINK, DOT (regardless of score)

**Top N selected:** 20 (futures) / 40 (spot)

Results cached in `data/scanner_cache.json`.

---

### 5.14 StateManager (`engine/bot.py` → `StateManager`)

**State files:**
- `data/state.json` — spot trades
- `data/futures_state.json` — futures trades

**Safety:**
- Writes atomically via `.tmp.json` rename
- Batches writes every 10s (dirty flag)
- Keeps `.backup.json` copy after every flush
- Keeps `_archive.json` for historical trades

**`sync_with_exchange()`** runs every cycle to reconcile:
- Updates live PnL and duration for open positions
- Imports exchange positions not tracked locally
- Closes "ghost trades" (open in state, missing from exchange for >60s)

---

### 5.15 TelegramNotifier (`notify/telegram.py`)

Sends alerts and responds to slash commands polled via long-polling.

**Automated alerts:**
- Bot start / stop / crash / recovery / stall
- Every trade entry (symbol, side, size, price, SL, TP)
- Every trade exit (symbol, PnL, exit reason)
- Watchlist changes (≥5 coins added/removed)
- Meta-Judge rule updates
- Periodic PnL report (every 30 minutes)

**Slash commands (from Telegram):**
```
/status    — current regime, open trades count, balance, last signal
/agents    — agent win-rates and performance
/pnl       — today's P&L summary
/trades    — list of open trades with live PnL
/health    — system health: data feed, exchange, heartbeat age, DeepSeek cost
/shadows   — hypothetical outcomes of rejected signals (per gate)
/help      — command list
```

**Natural language ops (`nl_ops.py`):** Free-form messages (e.g. "what's my ETH position?", "pause trading") are routed through DeepSeek V3 to map to one of the above commands. The LLM can only select from the fixed allow-list — no arbitrary code execution.

---

### 5.16 ShadowTracker (`agents/shadow_tracker.py`)

Forward-tracks **rejected** signals as hypothetical "shadow trades" to close the survivorship-bias hole in the learning loop (Judge/Meta-Judge only see trades that were taken).

**Capture:** when a directional signal (ensemble said BUY/SELL) is rejected at one of four gates — `microstructure`, `actor_prefilter`, `actor`, `risk` — a shadow is recorded with entry = current price and SL/TP computed with the *exact same* ATR math a real trade would have used (`profile.stop_loss_atr_mult` with the 1.5% min / 25% max clamps from `ExecutionEngine._place_sl`, `profile.take_profit_atr_mult`). HOLDs and macro/universe kills are not shadowed (no direction). A per-`(symbol, side)` cooldown (default 60 min) dedups rejections re-fired every scan cycle, and `max_open` (default 60) caps growth.

**Resolution:** every `resolve_every_n_cycles` cycles, open shadows are checked against 15m candles (cached `DataFeed.fetch_ohlcv`). Only candles at/after creation count (no lookahead). SL touch → `sl` (−1R); TP touch → `tp` (+r_target); both in one candle → SL (conservative); older than `max_age_hours` (48) → `expired` with mark-to-market R.

**Storage:** `shadow_trades` table in `data/trade_memory.db` (mode column separates spot/futures).

**Reporting (strictly observational — never auto-tunes thresholds):**
- Per-gate stats (n, hypothetical win rate, net R "avoided by the gate") appended to the Meta-Judge prompt on its regular every-20-critiques run — zero extra LLM calls.
- `/shadows` Telegram command — per-gate table with a "may be too strict / protective / gathering data" verdict.
- `GET /api/shadow_stats` on the dashboard.

Config block `shadow_tracker:` in both YAMLs. All hooks in `engine/bot.py` are try/except-isolated — a tracker fault can never break the scan loop.

---

## 6. Trading Profiles

Profiles control signal sensitivity, entry thresholds, and exit parameters. Set in `config_*.yaml` under `strategy.trading_profile`. Hot-swappable without restart.

| Profile | Min Confidence | Agent Agreement | Net Score Threshold | Use Case |
|---------|---------------|-----------------|--------------------| ---------|
| `STRICT` | 0.70 | 3 of 3 | 0.48 | 3–5 trades/week, max quality, swing holds |
| `BALANCED` | 0.58 | 2 of 3 | 0.32 | 10–12h intraday, steady compounding |
| `AGGRESSIVE` | 0.42 | 1 of 3 | 0.05 | Momentum scalp, breakout/volume-driven |
| `CONFLUENCE` | 0.52 | 2 of 3 | 0.28 | Weighted scoring across all dimensions |

**Exit parameters by profile:**
| Profile | SL mult | TP mult | Trail mult | TP1 fraction |
|---------|---------|---------|-----------|--------------|
| STRICT | 3.0× ATR | 7.0× ATR | 3.5× ATR | 40% |
| BALANCED | 2.5× ATR | 4.5× ATR | 2.8× ATR | 40% |
| AGGRESSIVE | 2.0× ATR | 3.5× ATR | 3.0× ATR | 50% |
| CONFLUENCE | 2.8× ATR | 6.0× ATR | 3.0× ATR | 40% |

---

## 7. Configuration Reference

### `config_futures.yaml` (futures-specific values)

```yaml
bot:
  scan_interval_seconds: 60       # How often to scan all symbols

actor:
  cache_ttl_seconds: 180          # Reuse Actor verdict for unchanged setup
  winrate_half_life_hours: 24     # Age at which past trade weight halves
  winrate_prior_strength: 3.0     # Beta smoothing strength toward 50%

exchange:
  margin_type: ISOLATED           # Per-position margin (not cross)

risk:
  leverage: 5                     # Futures leverage multiplier
  max_open_trades: 10
  max_consecutive_losses: 6
  max_daily_loss_pct: 0.03        # 3% daily loss trips circuit breaker
  max_drawdown_pct: 0.10          # 10% total drawdown
  max_portfolio_heat: 0.50        # Max 50% of balance in positions
  stop_loss_atr_multiplier: 3.0
  take_profit_atr_multiplier: 6.5

scanner:
  top_n: 20
  min_volume_usdt: 100000000      # 100M USDT minimum daily volume
  rescan_hours: 4

strategy:
  trading_profile: AGGRESSIVE

trend_filter:
  enabled: true
  use_two_tier: true
  fast: { tf: 15m, ema_fast: 20, ema_slow: 50, slope_lookback: 10 }
  slow: { tf: 1h,  ema_fast: 50, ema_slow: 200, slope_lookback: 20 }
  strong_slope_pct: 0.02          # 2% EMA change over 20 bars = "strongly trending"
```

### `config_spot.yaml` (spot-specific values)

```yaml
bot:
  scan_interval_seconds: 30       # Faster scan (no leverage = lower stakes)

risk:
  max_open_trades: 5
  max_consecutive_losses: 10
  max_daily_loss_pct: 0.05
  stop_loss_atr_multiplier: 4.0
  take_profit_atr_multiplier: 3.0

scanner:
  top_n: 40
  min_volume_usdt: 30000000       # 30M (lower threshold for spot)
  rescan_hours: 2

trend_filter:
  strong_slope_pct: 0.002         # Tighter veto for spot (less leverage)
```

### `.env` secrets

```
BINANCE_API_KEY=...
BINANCE_SECRET=...
DEEPSEEK_API_KEY=...
TELEGRAM_TOKEN=...
TELEGRAM_CHAT_ID=...
COINGECKO_API_KEY=...   # Optional — use "" for free tier
BOT_MODE=futures        # Written by launcher
```

---

## 8. Data Flow — One Scan Cycle

```
run_once() called
    │
    ├─ maybe_swap_profile()           # Hot-reload profile if YAML changed
    ├─ sync_with_exchange()           # Reconcile state vs exchange
    ├─ check_exits()                  # Exit open trades (SL/TP/trail/reversal)
    ├─ get_usdt_balance()
    ├─ close_all_positions()          # If dashboard flagged
    ├─ is_trading_paused()            # Dashboard pause check
    ├─ circuit_breaker.can_trade()
    ├─ scanner.get_coins()            # Get watchlist (cached)
    ├─ feed.subscribe_many()          # Ensure WS streaming
    ├─ detect_market_regime()         # MarketRegimeGate (cached 15m)
    ├─ macro_context.get()            # CoinGecko (cached 5m)
    ├─ pre-fetch 1h OHLCV parallel    # All symbols at once
    ├─ pre-fetch multi-TF OHLCV parallel  # [15m, 1h, 4h, 1d] for all symbols
    │
    └─ For each symbol (sequential):
           analyze_symbol()
               ├─ Gate 0/1: macro kill + universe check
               ├─ ensemble.run()          # SMC + Technical + MacroFlow in parallel
               ├─ trend_filter veto       # Inside ensemble
               ├─ Gate 4: microstructure.analyze()
               ├─ Actor pre-filter        # Skip LLM if conf < min_confidence
               ├─ trade_memory.find_similar()   # RAG: 5 similar past trades
               ├─ trade_memory.find_similar_critiques()  # RAG: 3 lessons
               ├─ llm_reasoning.actor_evaluate()   # DeepSeek V3
               ├─ blend ensemble + actor confidence
               ├─ risk_agent.evaluate()   # All risk gates + sizing
               └─ execution.execute_entry()   # Place order + SL
```

---

## 9. Trade Lifecycle

### Entry
1. All 7 gates pass
2. `ExecutionEngine.execute_entry()` places market order (3 retries)
3. Stop-loss placed on exchange (futures) or managed internally (spot)
4. Trade recorded in StateManager with full context
5. Entry context stored in `_entry_contexts` (persisted to disk)
6. Telegram entry alert sent

### During trade
- Every cycle: `check_exits()` reads live price vs SL/TP levels
- Every cycle: `sync_with_exchange()` updates live PnL
- Dashboard shows live PnL, mark price, duration

### Exit
1. `check_exits()` detects exit condition
2. Close order placed (`_place_close()`)
3. Fill price looked up from exchange trades (falls back to detection price)
4. PnL calculated (leveraged for futures)
5. StateManager updated (partial or full close)
6. If partial (TP1): remaining SL moved to breakeven on exchange
7. `_on_trade_closed()` called:
   - Entry context removed from disk
   - Trade recorded to `TradeMemory` SQLite
   - Judge thread spawned (async, non-blocking)
8. Telegram exit alert sent

### Judge cycle (background)
1. Judge (DeepSeek R1) reviews closed trade
2. Critique stored in TradeMemory
3. If 20 critiques since last Meta-Judge run:
   - Meta-Judge (DeepSeek R1) synthesizes rules
   - New rules stored, injected into next Actor prompt
   - Summary sent to Telegram

---

## 10. Dashboard

**URL:** `http://localhost:5002` (or `http://<server-ip>:5002`)

**Run separately from the bot:**
```bash
cd dashboard && python3 server.py
```

**Panels:**
- Live strategy state (regime, profile, BTC.D, USDT.D, macro universe)
- Open trades with live PnL
- Trade history and performance statistics
- Signal log (last 500 signals with decisions and reasons)
- DeepSeek usage cost today
- Agent performance metrics
- Data quality (bars per coin, freshness)

**Control actions (write command files read by next bot cycle):**
- Pause / Resume trading (`data/trading_paused.json`)
- Close all positions (`data/close_all_positions.json`)
- Stop bot (`data/bot_control.json`)

---

## 11. Key Design Decisions and Known Behaviors

### Why DeepSeek and not GPT/Claude?
Cost: DeepSeek V3 is ~20× cheaper than GPT-4 at the same context size. R1 (reasoning model) is used for Judge/Meta-Judge because it shows its chain-of-thought and produces more consistent structured output for critique tasks.

### Actor verdict cache
Without the cache, every scan of every symbol (up to 20 symbols × 60s cycle) calls the LLM serially. With `cache_ttl_seconds=180`, an unchanged setup (same regime, net score bucket, OB imbalance bucket, CVD) reuses the prior verdict. Cache hit rate is ~70–80% in practice.

### Actor freeze-loop prevention
A raw win-rate computed over 5 similar trades collapses to 0% after a single bad session and pins Actor confidence below approval threshold → no new trades → win-rate never recovers. The fix: recency-weighted (half-life 24h) + Beta smoothing (prior=3.0) so no cluster of losses reads as "0% and never trade again."

### Single trend bias reduction
Old code applied `net * 0.05` (counter-trend penalty) then `net * 0.7` (directional bias) = `net * 0.035`. This caused near-permanent HOLD for counter-trend signals. Fixed to a single 30% reduction only.

### Two-tier trend filter
The old single-tier 1h EMA(50/200) with `strong_slope_pct=0.002` (0.2%/bar) vetoed almost any short in mild uptrends. The two-tier filter (15m fast + 1h slow) with `strong_slope_pct=0.02` (2% over 20 bars) only vetoes against genuinely strong trends; mild grinds let actor/risk gates decide.

### Ghost trade cleanup
If a position disappears from the exchange (e.g. SL hit, manual close) but the bot doesn't know, it would sit on a "ghost" open trade forever. The sync runs every cycle and closes any trade missing from the exchange for >60 seconds.

### Futures vs Spot PnL
Futures PnL = `(exit - entry) / entry × amount × entry × leverage` (leveraged)
Spot PnL = `(exit - entry) × amount` (simple)

Both calculate from actual fill prices looked up from exchange trade history, not from the detection price.

---

## 12. Deployment

### Minimum viable setup
1. Clone repo, install requirements: `pip install -r requirements.txt`
2. Fill in `.env` with Binance Demo API keys, DeepSeek API key, Telegram bot token + chat ID
3. Start bot: `cd bot && BOT_MODE=futures python3 launcher.py`
4. Start dashboard: `cd dashboard && python3 server.py`

### Recommended: systemd service
```ini
[Unit]
Description=CryptoBot v5 Futures
After=network.target

[Service]
WorkingDirectory=/home/sarmad/cryptobot_v5/bot
ExecStart=/home/sarmad/cryptobot_v5/venv/bin/python3 launcher.py
Environment=BOT_MODE=futures
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Paper trading requirements before live capital
- Minimum 2 weeks on Binance Demo
- Minimum 50 closed trades
- Review every Judge critique
- Check Meta-Judge output after first 20 trades
- Only go live after Judge rates >70% of decisions "good" or "acceptable"

---

## 13. File Glossary — Runtime Data Files

| File | Contents |
|------|----------|
| `data/state.json` | Spot trade state, stats, signals (last 500) |
| `data/futures_state.json` | Futures trade state, stats, signals |
| `data/futures_state.backup.json` | Auto-backup of last flush |
| `data/futures_state_archive.json` | Historical closed trades (older ones) |
| `data/trade_memory.db` | SQLite: all closed trades + Judge critiques + Meta-Judge rules + shadow trades |
| `data/scanner_cache.json` | Cached coin watchlist |
| `data/bot_heartbeat_futures.json` | Liveness timestamp (written before/after each cycle) |
| `data/entry_contexts_futures.json` | Entry context for open trades (for Judge on close) |
| `data/deepseek_usage.json` | Per-day token usage + cost tracking |
| `data/agent_performance.json` | Per-agent win-rate and performance stats |
| `data/circuit_breaker.json` | Circuit breaker state (daily loss, consecutive losses) |
| `data/trading_paused.json` | Dashboard pause flag |
| `data/close_all_positions.json` | Dashboard close-all flag |
| `data/bot_control.json` | Dashboard stop command |
| `logs/futures_bot.log` | Rotating log (10MB × 5 backups) |
