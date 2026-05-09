# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> Always invoke the superpowers plugin before writing code. Use `writing-plans` + `executing-plans` for non-trivial tasks.

## Run
```bash
bash ~/cryptobot_v3/start.sh          # interactive menu (1=spot, 2=futures)
bash ~/cryptobot_v3/start.sh 1        # spot mode
bash ~/cryptobot_v3/start.sh 2        # futures mode
pkill -f 'cryptobot_v3'               # stop all
screen -r cryptobot_v3_spot           # attach (Ctrl+A D to detach)
```
Dashboard: `http://localhost:5002` — `BOT_MODE` env var (`spot`|`futures`) controls active config/data dir.

## Architecture

**Signal flow per cycle:** `sync_with_exchange → check_exits → rl_manage_trades → regime_gate → EnsembleEngine → RiskDecisionAgent → ExecutionEngine`

**EnsembleEngine** aggregates: SMCAgent (35%) + TechnicalAgent (40%) + MacroAgent (25%) → BUY/SELL/HOLD + confidence.

**BaseBot** holds all strategy logic. `SpotBot`/`FuturesBot` override only 5 exchange-interaction methods.

**ML:** RF + LightGBM walk-forward ensemble; champion/challenger promotion. HMM adjusts thresholds only — never overrides signals. DQN manages open positions only (HOLD/SCALE_IN/SCALE_OUT/CLOSE).

## Key Rules
1. Decision hierarchy: SIGNAL → CONTEXT → EXECUTION → RISK. Context sets `action="HOLD"` — no early return.
2. `decide()` always returns `(action, confidence)` — always unpack both.
3. Stale `.pkl`/`.pt` model files → delete; bot retrains next cycle.
4. State written atomically via `.tmp.json` → `replace()`.
