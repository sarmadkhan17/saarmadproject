# Signal-Quality Overhaul — Design Spec

**Date:** 2026-05-18
**Mode:** futures (spot inherits)
**Status:** draft, awaiting user review

---

## 1. Background — verified findings (no assumptions)

Audit of the last 20 hours of `data/futures_state.json` + `logs/futures_bot.log` plus live Binance Futures klines:

| Metric | Bot (20 h) | Real market (20 h) |
|---|---|---|
| Entries | 50 — **all SHORT** | 12 of 14 traded symbols closed UP |
| Closes | 52 (25 W / 27 L, 48.1 %) | — |
| Net PnL | –$4.48 | — |
| Avg duration | 2.47 h | — |
| Avg confidence | 0.32 (abs) | — |
| HMM regime | `RANGING` 143/143 samples | — |
| Open positions | 5 — all SHORT, all underwater | — |

**Root causes confirmed by code+config inspection:**

- **R1. Trend filter mis-tuned.** `config_futures.yaml`: `trend_filter.tf=1h, ema_fast=50, ema_slow=200, veto_longs=true, veto_shorts=true`. A 200-bar EMA on 1h is ≈ 8 days of context for a profile with 2.5 h average trade. BTC 1h is BEAR (EMA50 77 369 < EMA200 79 125) → `veto_longs` fires on every long, structural 100 % short lock.
- **R2. HMM stuck.** Regime label has not changed once across the entire current log. Cause is two-fold: training distribution + the 5 min cache TTL (`_CACHE_TTL = 300`) preventing rapid re-evaluation.
- **R3. Stale decision cache.** `AgentCoordinator._decision_cache` returns a cached ensemble verdict for 30 min (`GROQ_CACHE_SECS=1800`) per symbol. On a 30 s scan loop that is **60 consecutive cycles returning the same action**. Markets can move > $300 on BTC in that window. This is the single highest-impact bug.
- **R4. Macro / fear-greed stale.** `SLOW_CACHE_SECS=7200` (2 h) feeds the master ensemble. Cache age can exceed the trade horizon itself.
- **R5. Confidence threshold ratcheted down without evidence.** Recent commits show `strategy.min_confidence` walked 0.43 → 0.40 → 0.38 by an auto-tuner. Most taken signals cluster at the floor (0.35–0.49). DeepSeek was the auto-tuner; it is now stopped.
- **R6. Ensemble agent weights are static** (`SMC 0.35 / Technical 0.40 / Macro 0.25`). No empirical justification recorded; agents may be correlated on the same lagging features.
- **R7. ML probabilities are uncalibrated.** RF + LGBM `predict_proba` outputs are used directly as `confidence` without isotonic / Platt calibration; "0.38" is not a probability.
- **R8. Self-learner did not surface the 100 % short asymmetry.** Should have appeared in `pending_recommendations.json`; did not.
- **R9. Monitor died silently** ~10 h before the audit. No watchdog.

Each finding is referenced by R-code in the pillars below.

---

## 2. Goals & non-goals

**Goals.**

1. Replace mechanisms that **structurally bias direction** with mechanisms that admit either direction whenever evidence supports it.
2. Make every cache freshness contract **explicit per consumer**, not silently TTL-blanket.
3. Make `min_confidence` an **evidence-derived** value, not a knob.
4. Ensure the ensemble has **statistically-positive-edge agents only**, and no single agent can dominate without justification.
5. Enforce a **backtest + shadow** validation gate per pillar before promotion to live.
6. Restore monitor observability with a watchdog so silent failures are impossible.

**Non-goals (deferred).**

- DQN / GNN logic changes. They are out of the decision path for entries; only manage open positions / correlation caps.
- Spot-mode-specific rework. Spot inherits `BaseBot` changes automatically; spot-specific tuning waits.
- New agents (e.g. orderbook flow). YAGNI for this overhaul.
- LLM re-introduction in any form. Removed; not coming back.

---

## 3. Architecture overview

**Preserved.** Pipeline shape (`sync → exits → rl_manage → regime_gate → Ensemble → RiskDecision → Execution`); `BaseBot`/`SpotBot`/`FuturesBot` separation; `EnsembleEngine` and `MasterAgent` interfaces; `RiskManager` (Kelly, trailing, heat, circuit breaker); state-file atomic-write contract; `demo_api.py` ↔ live API parity rule.

**Replaced or added.**

```
                          ┌──────────────────────────┐
                          │  Freshness contract (P1) │
                          │  per-call max_age check  │
                          └────────────┬─────────────┘
                                       │
   ┌───────────────────┐               │             ┌───────────────────────┐
   │ Regime layer (P2) │◄──────────────┘             │ Validation infra (P5) │
   │ HMM + det. overlay│                             │ backtest + shadow     │
   └─────────┬─────────┘                             └────────────┬──────────┘
             │                                                    │
             ▼                                                    ▼
   ┌───────────────────────┐    ┌─────────────────────┐    ┌────────────────┐
   │ Two-tier trend filter │───▶│ Ensemble agents (P3)│───▶│ Calibration +  │
   │ 15m fast + 1h slow    │    │ backtest-gated,     │    │ threshold (P4) │
   │ slope-magnitude veto  │    │ equal-weight Phase1 │    │ derived, not   │
   └───────────────────────┘    └─────────────────────┘    │ tuned          │
                                                           └────────────────┘
                                       │
                                       ▼
                          ┌──────────────────────────┐
                          │ Observability (P6)       │
                          │ monitor + watchdog       │
                          └──────────────────────────┘
```

Pillars 1–6 below.

---

## 4. Pillars

### P1 — Freshness contract (`bot/data/freshness.py`, new)

**Goal.** Kill silent staleness; every cached read carries an explicit `max_age_seconds` from the consumer.

**Design.**

- New `Freshness` mixin / decorator: `Freshness.fetch(key, max_age_s, loader)` returns the cached value if `now - cached_at < max_age_s`, otherwise calls `loader()` and refreshes.
- Three category contracts, one place to read them:

| Consumer | Field | Max age |
|---|---|---|
| Decision path (`AgentCoordinator.analyze`) | action / confidence | scan_interval (30 s) |
| Decision path | indicators snapshot | 60 s |
| Regime gate | HMM regime label | 60 s (with deterministic overlay vote — see P2) |
| OHLCV 1m / 5m / 15m | forming candle | 30 / 60 / 60 s + always overlay WS last tick |
| OHLCV 1h | forming candle | 120 s |
| OHLCV 4h / 1d | forming candle | 300 / 600 s |
| Macro / Fear-Greed | snapshot | 1800 s (was 7200) and emit `staleness_s` |
| Scanner watchlist | symbols | 4 h (unchanged) |

- **`AgentCoordinator._decision_cache` is deleted.** The agent decision is recomputed each scan — but the **features** powering it are cached at 30–60 s (the expensive computation is feature pipeline, not the agent vote). Net per-scan cost measured before/after in P5.
- **Macro stale-decay rule:** ensemble weight on `MacroAgent` linearly decays to 0 over `[1800 s, 3600 s]` of staleness; below 1800 s full weight, above 3600 s zero weight.
- **No method may silently fall back to a stale value.** A cache miss returns `None` and the consumer must explicitly choose whether to skip the decision or use a defined default.

**Acceptance.**

1. Unit test: every entry in `Freshness` registry has a documented `max_age_s`.
2. Unit test: stale read returns `None`, never a stale value silently.
3. Live test (shadow): with the contract on, an entry's `confidence` value is observed to change at least once per 60 s on average per symbol over a 1 h window.

---

### P2 — Regime layer (HMM + deterministic overlay)

**Goal.** Regime must move when reality moves. A broken HMM cannot freeze the system.

**Design.**

- **P2a. Retrain HMM** with class-balanced training: stratify the parquet by labelled regimes (TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE) using a deterministic labeler (return + ADX + realised vol), oversample minority classes during fit. Save to `data/<mode>/hmm.pkl` with metadata.
- **P2b. Deterministic overlay** (new `bot/models/regime_overlay.py`): a feature-rule classifier from `ADX_14`, `EMA_slope_15m_50`, `EMA_slope_1h_50`, `realised_vol_5m_60`. Outputs the same 4-class label and a confidence.
- **P2c. Regime fusion.** Final regime label = consensus of HMM and overlay. If they disagree, use the **higher-confidence** vote AND log a `regime_disagreement` event (sampled into shadow log). After 1 week, evaluate disagreement rate; if > 30 %, HMM is broken and is dropped from fusion until retrained.

**Acceptance.**

1. Backtest on 3-year parquet: HMM label changes at least once per 24 h on average per symbol.
2. Overlay label changes at least once per 4 h on average per symbol.
3. Fusion label distribution across the parquet is within ±10 % of empirical regime distribution (computed from labeler).

---

### P3 — Two-tier trend filter

**Goal.** Match filter speed to the trading profile, and allow direction whenever evidence supports it.

**Design.**

- Replace single-tier `trend_filter` config with two tiers in `config_<mode>.yaml`:

  ```yaml
  trend_filter:
    fast:
      tf: 15m
      ema_fast: 20
      ema_slow: 50
      slope_lookback: 10   # bars
    slow:
      tf: 1h
      ema_fast: 50
      ema_slow: 200
      slope_lookback: 20
    long_rule:  "fast_up AND NOT slow_strongly_down"
    short_rule: "fast_down AND NOT slow_strongly_up"
    strong_slope_pct: 0.002   # 0.2 % per bar = "strongly"
  ```

- **`veto_longs` / `veto_shorts` flags are removed.** Direction is admitted whenever its rule holds; the filter only vetoes when the slow tier is *strongly opposed* (slope magnitude > `strong_slope_pct`).
- Implementation: new `bot/engine/trend_filter.py` with `TrendFilter.check(symbol) -> {"long_allowed": bool, "short_allowed": bool, "reasoning": str}`.

**Acceptance.**

1. Replay-based test: on any historical 24 h window where ≥ 60 % of traded symbols rallied > 1 %, the new filter must admit at least 4 long entries (versus the 0 produced by the current filter on the audited 2026-05-18 window).
2. On any historical 24 h window where a symbol declined > 2 % monotonically, longs on that symbol must remain vetoed.
3. Unit tests for each branch of the rule.

---

### P4 — Ensemble agents (backtest-gated, equal-weight Phase-1)

**Goal.** Only agents with statistically-positive historical edge get a vote; no static weights that we can't justify.

**Design (Phase 1 — building, where we are now).**

- **P4a. Per-agent backtest harness** (`bot/backtest/agent_eval.py`, new): replays each agent's `analyze()` over the 3-year parquet, walk-forward, regime-stratified, no look-ahead. Emits per-agent / per-regime metrics: trade count, OOS Sharpe (with 95 % CI), expectancy, hit rate, avg duration.
- **P4b. Agent gating.** An agent is included in the live ensemble **only if** its OOS Sharpe 95 % CI lower bound is > 0 on at least one regime. If all agents fail, the system emits a hard error and refuses to start — silent shipping is not allowed.
- **P4c. Phase-1 weighting.** Surviving agents → **equal weight**, `min_votes` = max(2, ceil(n_surviving/2)). Ensemble outputs HOLD when consensus is not reached. The two existing ensemble code paths (`bot/engine/ensemble.py` and `bot/agents/coordinator.py:MasterAgent`) must be reconciled — exactly one becomes the live path; the other is deleted. (Implementation phase will choose; current evidence favors `bot/engine/ensemble.py` since it matches the CLAUDE.md SMC/Technical/Macro shape and is the one driving the observed signals.)

**Design (Phase 2 — graduation).**

- After 200 trades / agent / regime have accumulated **with the corrected pipeline** (no broken-data trades count): allow live to nudge each agent's weight within ±25 % of the equal-weight baseline.
- Live performance can only nudge if its sign agrees with backtest sign for the same agent+regime. If they disagree, freeze weights and emit a `live_backtest_divergence` alert.

**Acceptance.**

1. Backtest harness runs on every `python -m bot.backtest.agent_eval` invocation, deterministic for a fixed seed.
2. Agent gating fail produces a non-zero exit code.
3. Phase-1 weights are exactly equal; no auto-adjustment until graduation gate trips.

---

### P5 — Calibration + threshold derivation

**Goal.** `confidence` becomes a probability; `min_confidence` becomes an EV-derived cutoff.

**Design.**

- **P5a. Calibration.** Wrap RF and LGBM in `sklearn.calibration.CalibratedClassifierCV(method="isotonic", cv="prefit")` using a held-out 20 % fold from the walk-forward final window. Persist the calibrators alongside the models. `predict_proba` now returns actual probabilities.
- **P5b. Threshold derivation.** During retrain, for each side (BUY / SELL), compute on the OOS window:
  - For each candidate cutoff `p ∈ [0.50, 0.95]` step 0.01: simulate trades where `predict_proba >= p`, compute expectancy per trade.
  - `min_confidence_buy` = lowest `p` where expectancy ≥ `target_ev`. Same for SELL.
  - `target_ev` = `1.5 × estimated_round_trip_cost`. Round-trip cost = `taker_fee × 2 + 2 × estimated_slippage_bps / 10000` (configured in `risk.cost_model`).
- Persisted in `data/<mode>/threshold.json`:
  ```json
  {"min_confidence_buy": 0.71, "min_confidence_sell": 0.68, "target_ev": 0.0024, "computed_at": "2026-05-18T..."}
  ```
- `strategy.min_confidence` is removed from config. Reader code loads `threshold.json` at startup and on retrain.

**Acceptance.**

1. Calibration unit test: ECE (expected calibration error) on the held-out fold ≤ 0.05.
2. Threshold file regenerated on every retrain; bot reload picks it up without restart.
3. If `threshold.json` is missing, bot refuses to take entries (HOLD-only) — never falls back to a hard-coded number.

---

### P6 — Validation infrastructure (backtest + shadow)

**Goal.** Nothing reaches live until backtest and shadow both confirm improvement.

**Design.**

- **P6a. Backtester (`bot/backtest/`).** Two scripts:
  - `bot/backtest/agent_eval.py` (P4a) — per-agent evaluation.
  - `bot/backtest/pipeline_eval.py` — replays the **entire** decision pipeline (freshness, regime, filter, ensemble, calibration, risk) across the parquet. Emits Sharpe, expectancy, max DD, hit rate per regime per side.
- **P6b. Shadow mode.** New config flag `strategy.shadow: true`. When set, on every scan loop the bot computes both:
  - The **live decision** using the currently-promoted code path.
  - The **candidate decision** using the candidate code path under test (selectable via `strategy.shadow_pipeline: "candidate_v2"`).
  Both are logged to `logs/shadow_decisions.jsonl`. Only the **live decision** triggers orders. Divergence summarised by a daily `tools/shadow_report.py`.
- **P6c. Promotion gate** (per-pillar). Promotion live ⇔ all of:
  1. Backtest expectancy on OOS window > current live baseline.
  2. Shadow run ≥ 48 h.
  3. Shadow Sharpe is ≥ live Sharpe on the overlap window OR shadow makes signal where live abstained, with ≥ 55 % directional accuracy on those.
- **P6d. Performance budget.** `tools/profile_scan_loop.py` measures p50 / p95 scan-loop latency before each pillar lands. Regressions > 25 % require justification before merge.

**Acceptance.**

1. `pytest bot/backtest/tests/` green.
2. Shadow mode writes structured JSONL records with both decisions per scan.
3. p95 scan loop after P1 (cache contract) is ≤ 1.5 × pre-P1.

---

### P7 — Observability (monitor revival, watchdog, self-learner)

**Goal.** Silent failure becomes impossible; self-learner actually flags asymmetries.

**Design.**

- **P7a. Watchdog.** Existing `monitor_trades.py` gets a sibling: `tools/watchdog.py`, run by systemd or `start.sh`. It checks every 60 s that the monitor's last write to `logs/monitor_<mode>.jsonl` is < 5 min old; if not, it restarts the monitor and logs to `logs/watchdog.log`. The bot itself also writes `data/bot_heartbeat_<mode>.json` every scan; watchdog checks bot freshness too.
- **P7b. Bias detector** (`bot/tuning/learner.py` enhancement). New rule: if last-N-trades side-mix exceeds 80 %/20 % for N ≥ 30, emit a recommendation to review trend filter / regime layer. Propose-only; written to `pending_recommendations.json`.
- **P7c. Divergence alerts.** New `alerts/` channel (Telegram), gated by `notify.alerts_enabled`. Events: `regime_disagreement` > 30 % over 1 h, `live_backtest_divergence`, `shadow_divergence > target`, watchdog restarts.

**Acceptance.**

1. Killing the monitor process causes a watchdog restart within 90 s.
2. Killing the bot process causes a Telegram alert within 5 min.
3. Manually injecting a 100 %-shorts state into the recent-trades window causes a `bias_detected` recommendation to be written within one learner cycle.

---

## 5. Validation gates (cross-pillar)

Every pillar PR must satisfy:

1. **Unit tests green.**
2. **Backtest gate** — `bot/backtest/pipeline_eval.py` on OOS window shows non-regression on the **live-baseline** (= current production code's metrics on the same OOS window, captured once at the start of the overhaul and stored at `data/baselines/pre_overhaul.json`), and improvement on the metric the pillar targets (e.g. P3 must improve long-side hit rate when long-opportunities exist).
3. **Shadow gate** — 48 h shadow mode on live data, divergence summary attached to the PR.
4. **Latency gate** — scan-loop p95 within budget (see P6d).
5. **Rollback plan** documented in the PR (every pillar is reversible by a feature flag + git revert).

---

## 6. Rollout sequence

Pillars execute in this order (dependency-driven):

| # | Pillar | Why first | Risk to live bot |
|---|---|---|---|
| 0 | DeepSeek cleanup | already stopped, lowest blast radius | none |
| 1 | P6a/d skeleton: backtester + profiler | needed to validate everything else | none (offline) |
| 2 | P7a: watchdog + monitor restart | restore observability before changing anything live | none |
| 3 | P1: freshness contract (no behavioural change yet — just kill 30-min cache) | unblocks every other behavioural change | moderate (scan cost rises; mitigated by feature micro-cache) |
| 4 | P3: two-tier trend filter | highest-impact behavioural fix | moderate (longs become possible — but only when filter agrees) |
| 5 | P2: regime overlay + HMM retrain | improves regime quality, fixes P1's regime input | moderate |
| 6 | P4: per-agent backtest + Phase-1 weighting | gates which agents are even allowed to vote | high (some agents may be dropped) |
| 7 | P5: calibration + threshold derivation | requires P4 agents are stable | high (changes `min_confidence` semantics) |
| 8 | P6b/c: shadow + promotion gate enforced | final guardrail | none (additive) |
| 9 | P7b/c: bias detector + alerts | nice-to-have once core is solid | none |

**Each pillar lands as its own PR**, gated by §5.

**During rollout the live bot may be paused** via `data/trading_paused.json` (already implemented) at pillar boundaries 4, 6, 7 only — explicit user authorisation required each time per CLAUDE.md.

---

## 7. In-flight position handling

At the moment of each pillar promotion, the bot may hold open positions sized under the old logic. Policy:

- **Open positions are never force-closed by a code change.** They run to their existing exit logic (stops, trailing, RL-managed).
- **New entries** use the new logic immediately after promotion.
- If circuit breaker trips during rollout, the existing reset rule applies (zero `consec_losses` + `disabled_until` only; never touch `pnl_history` / balance — per CLAUDE.md and memory).

---

## 8. Cleanup tasks (folded in)

- Delete `deepseek_admin.py`, `deepseek_actions.log`, `deepseek_actions_futures.log`, `deepseek_actions_spot.log`.
- Remove DeepSeek references from `CLAUDE.md`.
- Remove DeepSeek references from `README.md` if any.
- Reconcile the two ensemble code paths (`bot/engine/ensemble.py` vs `bot/agents/coordinator.py:MasterAgent`) — pick one, delete the other. (Decided in P4 implementation.)
- Remove `strategy.min_confidence` and `strategy.min_votes` from config files (replaced by `threshold.json` and derived from surviving-agent count respectively).

---

## 9. Open questions / explicit assumptions

- **A1.** Backtest accuracy depends on the 3-year parquet being representative. We will inspect class balance and date coverage before relying on it. If gaps are found, P4 backtest harness includes a "gap report" step.
- **A2.** Shadow mode assumes the candidate code can run side-by-side without state mutation. Implementation must enforce read-only access for the candidate.
- **A3.** "Round-trip cost" used in P5 threshold derivation assumes Binance taker fee (live) + a configured slippage estimate. The slippage estimate starts at 5 bps and is revised post-shadow if live fills diverge.
- **A4.** No agent may be added to the ensemble outside this overhaul. Adding agents post-overhaul re-triggers P4 gating.

---

## 10. Out of scope (named explicitly to prevent drift)

- DQN scale-in/scale-out logic.
- GNN correlation cap parameters.
- Trailing-stop / take-profit ATR multipliers.
- Scanner symbol-selection logic.
- Dashboard UI changes.
- Anything in spot-only paths beyond what `BaseBot` inheritance gives for free.

---

*End of design spec. Awaiting user review per brainstorming skill before invoking writing-plans.*
