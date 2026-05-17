# CLAUDE.md

Guidance for Claude Code in this repo.

> Always invoke the superpowers plugin before coding. Use `writing-plans` + `executing-plans` for non-trivial tasks. **Never edit code without explicit user approval** — present a plan and wait for "go ahead" / "do it" / "yes".

## Run
```bash
bash ~/cryptobot_v3/start.sh          # menu (1=spot, 2=futures)
bash ~/cryptobot_v3/start.sh 1        # spot
bash ~/cryptobot_v3/start.sh 2        # futures
pkill -f 'cryptobot_v3'               # stop all
screen -r cryptobot_v3_{spot,futures} # attach (Ctrl+A D detach)
cd bot && python -m pytest tests/ -q  # tests
```
Dashboard `http://localhost:5002`. `BOT_MODE` (`spot`|`futures`) selects config + per-mode data dir. Only one bot at a time (shared port).

## Architecture

**Cycle:** `sync_with_exchange → check_exits → rl_manage_trades → regime_gate → EnsembleEngine → RiskDecisionAgent → ExecutionEngine`

**Ensemble (regime-adaptive weights, base):** SMC 35% + Technical 40% + Macro 25% → BUY/SELL/HOLD + confidence. Any agent error sets `agents_ok=False` → RiskDecisionAgent rejects.

**BaseBot** (`bot/engine/bot.py`) owns all strategy. `SpotBot`/`FuturesBot` override only 5 exchange methods (`_setup_exchange`, `_place_buy/sell/close`, `_calc_pnl`).

**Profiles** (`bot/engine/profiles.py`, picked via `strategy.trading_profile`): STRICT (swing, 3-agent, ADX≥28) · BALANCED (intraday, 2-agent) · **AGGRESSIVE** (current default, momentum, 1-agent OK) · CONFLUENCE (weighted quality overlay). Immutable presets; `from_config` returns a copy with `training.profile_overrides` applied.

**ML:** RF + LightGBM walk-forward ensemble (champion/challenger) → 3-class (SELL=0/HOLD=1/BUY=2) with precision-adaptive class weights. HMM **adjusts thresholds only**, never overrides signals. DQN manages *open positions only* (HOLD/SCALE_IN/SCALE_OUT/CLOSE). GNN caps correlated exposure.

**Key modules:** `core/config.py` (env+paths), `data/feed.py` + `ws_feed.py`, `exchange/demo_api.py` (direct Binance demo — bypasses ccxt `/sapi`), `features/`, `agents/coordinator.py`, `risk/manager.py` (Kelly, trailing, heat, circuit breaker), `tuning/scanner.py`, `tuning/learner.py` (propose-only), `notify/telegram.py`. UI: `dashboard/server.py`.

## Rules
1. Decision hierarchy: SIGNAL → CONTEXT → EXECUTION → RISK. Context sets `action="HOLD"` — **no early return**.
2. `decide()` always returns `(action, confidence)` — always unpack both.
3. Stale `.pkl`/`.pt` → delete; bot retrains next cycle (`ml.retrain_hours: 24`).
4. State writes are atomic: `*.tmp.json` → `os.replace()`. State lives in `data/state.json` (spot) / `data/futures_state.json`.
5. "Reset circuit breaker" = zero `consec_losses` + `disabled_until` only; never touch pnl_history/balance.
6. `demo_api.py` must match live API — never stub/simulate responses.
7. Self-learner is propose-only (`data/pending_recommendations.json`); never auto-edit configs.
