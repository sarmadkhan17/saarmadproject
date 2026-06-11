# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

CryptoBot v5 is a rule-based + LLM-reasoning crypto trading bot that paper-trades on Binance Demo (spot or futures). v3's ML stack (HMM, LightGBM, RF, DQN) was removed after it overfit twice — see `README_v5.md`. `BOT_MANUAL.md` is the authoritative, exhaustive component reference; consult it before deep changes. This file is the orientation layer on top of it.

## Commands

```bash
# Run the bot (must cd into bot/ — modules rely on sys.path.insert of that dir)
cd bot && BOT_MODE=futures python3 launcher.py   # futures: LONG/SHORT + leverage
cd bot && BOT_MODE=spot    python3 launcher.py   # spot: BUY/SELL only
# Unset BOT_MODE → launcher prompts interactively.

# Dashboard (separate process, Flask on :5002)
cd dashboard && python3 server.py

# Meta-oversight auditor (see Memory: proposal workflow)
python scripts/system_auditor.py            # one audit pass
python scripts/system_auditor.py --poll     # process pending Telegram replies
python scripts/system_auditor.py --loop     # audit every 6h + poll every 10min
python scripts/system_auditor.py --no-telegram   # print only, no sends

# Tests — run from REPO ROOT (tests import via the `bot.` package)
pytest tests/                               # all
pytest tests/test_microstructure_absorption.py -q
pytest tests/test_microstructure_absorption.py::<test_name>   # single test
```

Gotcha: the project `venv/` does **not** have `pytest` installed — use the system `pytest` (`/usr/local/bin/pytest`). The bot itself runs under `venv`.

## Secrets / config

- `.env` (gitignored) holds `BINANCE_API_KEY/SECRET_KEY`, `BINANCE_DEMO`, `TELEGRAM_TOKEN/CHAT_ID`, `DEEPSEEK_API_KEY`, `COINGECKO_API_KEY`, `BOT_MODE`, `BOT_EXECUTION_MODE`. Loaded via `bot/core/config.py:load_env()` (uses `setdefault`, so real env vars win).
- `config_futures.yaml` / `config_spot.yaml` — strategy params per mode (profile thresholds, scan interval, SMC sub-check strictness, etc.).
- **The LLM provider is DeepSeek, not OpenAI** — the `openai` package is just the SDK pointed at DeepSeek's endpoint (`deepseek-chat` = V3 Actor, `deepseek-reasoner` = R1 Judge/Meta-Judge). Don't assume OpenAI semantics.

## Architecture

**Inheritance:** `BaseBot` (`bot/engine/bot.py`, ~1700 lines — the scan loop, state, lifecycle) is subclassed by `SpotBot` (`engine/spot.py`) and `FuturesBot` (`engine/futures.py`). The strategy is identical across both; only order direction/leverage differ. `launcher.py` wraps the chosen bot in a crash-recovery loop + a heartbeat **watchdog** thread that re-execs the process if the scan loop silently stalls.

**The decision pipeline** runs per symbol, per scan cycle (`scan_interval_seconds`, default 30). Each gate short-circuits — failing one skips the symbol for that cycle:

```
Gate 0/1  Macro kill + universe filter   agents/macro_context.py (CoinGecko BTC.D/USDT.D RoC)
Layer 1   Ensemble vote                  engine/ensemble.py = SMC + Technical + MacroFlow,
                                           regime-adaptive weights + TrendFilter veto
Gate 4    Microstructure confirm/kill    agents/microstructure.py (orderbook imbalance, CVD, funding)
Gate 5    DeepSeek Actor reasoning       agents/llm_reasoning.py (LLM endorses, RAG over past trades)
Gate 6    Risk decision + Kelly sizing   engine/risk_agent.py + risk/manager.py
          → execute on Binance Demo      engine/execution_engine.py
On close          → Judge (R1) writes a verbal critique  (agents/llm_reasoning.py)
Every 20 closes   → Meta-Judge (R1) synthesizes rules injected into the Actor prompt
```

The "regime gate" and "ensemble minimum" (the `BOT_MANUAL` calls them Gate 2/3) live *inside* Layer 1's aggregation, not as separate steps.

**Learning loop (no ML training):** `agents/trade_memory.py` is a SQLite store (`data/trade_memory.db`) of every closed trade + its entry context + the Judge's critique. The Actor pulls similar past trades as RAG context (`agents/vector_store.py`, sentence-transformers embeddings). Meta-Judge distills accumulated critiques into rules. This is the entire "self-improvement" mechanism — there are no model files or retrain pipelines.

**`data/` is the IPC bus.** The bot, dashboard, and `system_auditor.py` are separate processes that coordinate through JSON/SQLite files in `data/`, not shared memory: `*_state.json` (per-mode trades), `bot_heartbeat_<mode>.json` (watchdog liveness), `circuit_breaker.json`, `scanner_cache.json`, `deepseek_usage.json`, `proposed_changes.md` / `audit_proposals.json` (auditor output). Dashboard control endpoints and the auditor mutate the same files the bot reads.

**Telegram is a full control surface** (`bot/notify/telegram.py`) — slash commands + a natural-language ops layer (`bot/notify/nl_ops.py`) that maps free-form messages onto the *same* allow-listed command set (one DeepSeek call per typed message, reactive only, state-changing actions require explicit YES). It never executes free-form model output and never touches exchange keys.

**Timezone:** all timestamps use `bot/core/tz.py:LOCAL_TZ` (UTC+3). Don't introduce naive `datetime.now()`.

## Conventions

- Modules import siblings via `sys.path.insert(0, str(Path(__file__).parent))` — running the bot from anywhere other than `bot/` will break imports.
- Spot and futures keep **separate** state files so the two modes never interfere.
- New ensemble agents should return the shared `AgentSignal` dataclass and slot into the weight table in `engine/ensemble.py`.
