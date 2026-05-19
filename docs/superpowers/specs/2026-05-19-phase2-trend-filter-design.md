# Phase 2 — Two-tier Trend Filter (P3) Design Spec

**Date:** 2026-05-19
**Mode:** futures (spot inherits)
**Status:** approved, ready for plan

**Parent spec:** `docs/superpowers/specs/2026-05-18-signal-quality-overhaul-design.md` §4 P3 (trend filter pillar).
**Predecessor:** Phase 1 (`phase1-foundation-complete`, commit `7960acda`) — freshness contract, watchdog, decision cache removed, backtest skeletons.

---

## 1. Why now

Phase 1 made cache freshness honest and observability reliable but **did not change which entries the bot takes**. The pre-overhaul baseline captured in `data/baselines/pre_overhaul_futures.json` is unchanged from the audit:

| Metric | Value |
|---|---|
| Closed trades (window) | 196 |
| Side mix | 175 short / 21 long (89 % short) |
| Win rate | 29.1 % |
| Net PnL | −$41.25 |

The audit identified R1 — **trend filter mis-tuned, structurally vetoes longs** — as the proximate cause of the 100 %-short bias. The current single-tier filter (`tf=1h, ema_fast=50, ema_slow=200, veto_longs=true, veto_shorts=true`) blocks every long entry on the bot's intraday-hold timescale because the 200-bar EMA on 1h is ~8 days of context, much longer than the 2.5 h average trade.

Phase 2 (P3 pillar) replaces this with a two-tier slope-magnitude filter that admits direction whenever the fast tier supports it and the slow tier is not *strongly opposed*. This is the single highest-impact behavioral fix in the overhaul roadmap.

---

## 2. Goals & non-goals

**Goals.**

1. Replace the single-tier `veto_longs` / `veto_shorts` filter with a two-tier filter that admits long entries whenever the fast tier (15m) supports them and the slow tier (1h) is not strongly opposed.
2. Land the change **behind a feature flag** (`trend_filter.use_two_tier`, default `false`) so the code-only PR is behaviorally inert and the flip is a separate, reversible commit.
3. Build out `bot/backtest/pipeline_eval.py` (currently a Phase-1 skeleton) into a plugin-style replay harness so the trend filter can be empirically validated against 3 years of OHLCV on the 8 training coins.
4. Apply the spec's acceptance criteria (§4.3 of the parent spec) as automated validators in the replay output.

**Non-goals (deferred to later phases or explicitly out of scope).**

- HMM retraining or regime overlay (Phase 3 / P2).
- Ensemble agent weighting changes (Phase 4 / P4).
- Calibration or threshold derivation (Phase 5 / P5).
- Shadow infrastructure for live A/B comparison (Phase 6 / P6b). The feature flag + replay harness provides equivalent rollback safety for P3 without building shadow infra prematurely.
- The full `pipeline_eval` covering ensemble + risk + calibration — Phase 2 builds the infrastructure and registers only the trend filter; subsequent phases register their components against the same harness.

---

## 3. Architecture

### 3.1 New module: `bot/engine/trend_filter.py`

A single class with a pure-function entry point:

```python
class TrendFilter:
    def __init__(self, cfg: dict) -> None:
        # cfg = config["trend_filter"]; reads .fast, .slow, .strong_slope_pct
        ...

    def check(self, dfs: dict[str, pd.DataFrame]) -> dict:
        # dfs = {"15m": df_15m, "1h": df_1h}
        # returns:
        #   {
        #     "long_allowed":  bool,
        #     "short_allowed": bool,
        #     "reasoning":     str,
        #     "fast":  {"direction": "up"|"down"|"flat", "slope_pct": float},
        #     "slow":  {"direction": "up"|"down"|"flat", "slope_pct": float, "strong": bool},
        #   }
```

- **No I/O.** Pure over the dfs + config. Constructed once at bot startup; called per scan.
- **Slope = `(EMA_t − EMA_{t−lookback}) / EMA_{t−lookback}`** (relative change of EMA_fast on each tier over `slope_lookback` bars; **state**, not edge-trigger — the filter cares whether the trend *is* up, not whether it *just crossed*).
- **`fast_up`** = on the fast tier, `EMA_fast > EMA_slow` AND fast slope > 0.
- **`fast_down`** = on the fast tier, `EMA_fast < EMA_slow` AND fast slope < 0.
- **`slow_strongly_up`** = on the slow tier, slope > `strong_slope_pct` AND `EMA_fast > EMA_slow`.
- **`slow_strongly_down`** = on the slow tier, slope < −`strong_slope_pct` AND `EMA_fast < EMA_slow`.
- **`long_allowed`** = `fast_up AND NOT slow_strongly_down`.
- **`short_allowed`** = `fast_down AND NOT slow_strongly_up`.
- **`strong_slope_pct`** = 0.002 (0.2 % per bar) — the threshold above which the slow tier is considered "strongly" trending.

### 3.2 Wiring (behind feature flag)

`bot/engine/ensemble.py` keeps the existing `_check_trend_veto` method unchanged. A new branch reads the feature flag:

```python
# In EnsembleEngine.decide(), after _aggregate produces a result with action ∈ {BUY, SELL}:
if self.trend_filter.get("enabled"):
    if self.trend_filter.get("use_two_tier"):
        verdict = self._tf2.check({"15m": df_15m, "1h": df_1h})
        if (action == "BUY" and not verdict["long_allowed"]) \
                or (action == "SELL" and not verdict["short_allowed"]):
            log.info(f"TREND VETO {symbol} → HOLD (was {action}): {verdict['reasoning']}")
            result.action = "HOLD"
            result.source = f"trend_veto_2tier:{verdict['reasoning'][:40]}"
    else:
        veto_reason = self._check_trend_veto(df, result.action)
        if veto_reason:
            ...  # existing behavior unchanged
```

The bot caller (`bot/engine/bot.py`) already passes `df_1h` to `decide()`; we will additionally pass `df_15m` (already fetched by the per-symbol scan loop). No new fetches.

### 3.3 Config shape (additive)

`config_futures.yaml` and `config_spot.yaml` gain the two-tier subsections; the existing fields stay untouched while the flag is `false`:

```yaml
trend_filter:
  enabled: true
  use_two_tier: false       # NEW — flag; set to true in the flip commit
  # legacy single-tier (used when use_two_tier=false; unchanged)
  tf: 1h
  ema_fast: 50
  ema_slow: 200
  veto_longs: true
  veto_shorts: true
  # NEW two-tier (used when use_two_tier=true)
  fast:
    tf: 15m
    ema_fast: 20
    ema_slow: 50
    slope_lookback: 10
  slow:
    tf: 1h
    ema_fast: 50
    ema_slow: 200
    slope_lookback: 20
  strong_slope_pct: 0.002
```

When the flag is flipped to `true`, the legacy fields become dead config but stay in place for one more PR (Phase 2 cleanup commit can remove them later, but only after a week of post-flip observation).

---

## 4. Replay harness — `bot/backtest/pipeline_eval.py` build-out

### 4.1 Plugin architecture

Pipeline_eval is built once as the harness for **all** future phase validations. Components plug in through a small interface:

```python
class PipelineComponent(Protocol):
    name: str
    def evaluate(self, symbol: str, ts: pd.Timestamp,
                 dfs: dict[str, pd.DataFrame]) -> dict: ...

    # Optional: component-specific acceptance-criterion validators
    def validators(self) -> list[Callable[[dict], dict]]: ...
```

Phase 2 registers exactly one component: `TrendFilterReplayComponent` (a thin wrapper over `engine.trend_filter.TrendFilter`). Phase 3+ register `RegimeReplayComponent`, `EnsembleReplayComponent`, etc.

### 4.2 Data source

The 8 training coins (BTC, ETH, BNB, XRP, SOL, DOGE, ADA, LINK) each have ~105K bars of 15m and ~26K bars of 1h covering 2023-05 → 2026-05 (full 3 years) in `data/training/<SYMBOL>USDT_<tf>.parquet`. No new fetches; the harness reads what's already on disk.

Optional secondary sweep: the 32 scanner-watchlist symbols with ~11 days of 15m can be enabled with `--include-recent`. They are **not** the validation gate — they're a sanity check that the filter behaves on broader live symbols too.

### 4.3 Walk-forward iteration

For each `(symbol, ts)` evaluation point:

1. Slice the 15m parquet up to and including `ts`: `df_15m = full_15m.loc[:ts]`.
2. Slice the 1h parquet up to and including `ts`: `df_1h = full_1h.loc[:ts]`.
3. Skip if either df has fewer bars than the required EMA span + slope_lookback.
4. Call `component.evaluate(symbol, ts, {"15m": df_15m, "1h": df_1h})`.
5. Append the result to a per-symbol log.

Look-ahead bias is structurally impossible — only data up to `ts` is visible.

### 4.4 Sampling cadence

Full-resolution walk-forward over 3 years × 8 coins is ~840K evaluations. Replays at every-1h cadence (one evaluation per hour-of-history) reduce this to ~210K, which runs in minutes on the existing hardware. Granularity is configurable via `--cadence` (`1h` default, `15m` for fine-grained).

### 4.5 Output schema

```json
{
  "captured_at": "2026-05-19T...",
  "components": ["trend_filter"],
  "coverage": {
    "symbols": ["BTCUSDT", "ETHUSDT", ...],
    "start": "2023-05-18T23:00:00",
    "end":   "2026-05-19T09:00:00",
    "cadence": "1h",
    "n_evaluations": 210000
  },
  "summary": {
    "trend_filter": {
      "long_allowed":  142000,
      "long_vetoed":    68000,
      "short_allowed": 134000,
      "short_vetoed":   76000,
      "per_symbol": {"BTCUSDT": {...}, ...}
    }
  },
  "validators": {
    "rally_window_admits_longs":     {"passed": true,  "details": {...}},
    "monotonic_decline_vetoes_longs":{"passed": true,  "details": {...}}
  }
}
```

### 4.6 Validators (P3 acceptance criteria)

Two automated validators implement the parent spec's §4.3 criteria:

1. **`rally_window_admits_longs`** — Find any 24h window in the parquet where ≥60 % of the 8 coins closed >+1 %. Assert the new filter admits **≥4 long entries** somewhere in that window's evaluations.
2. **`monotonic_decline_vetoes_longs`** — Find a symbol where its 24h window declined >2 % monotonically (every hour close ≤ previous hour close). Assert **all long evaluations in that window are vetoed** for that symbol.

Failure of either validator surfaces as `validators.<name>.passed = false` and the runner exits with non-zero code. The plan task that runs the replay treats non-zero as a stop-condition.

---

## 5. Validation gates

Per the parent spec §5, every pillar PR satisfies:

| Gate | How Phase 2 satisfies it |
|---|---|
| Unit tests green | `bot/engine/tests/test_trend_filter.py` covers each rule branch; full bot suite stays green |
| Backtest gate | `pipeline_eval` runs on 3-year × 8-coin data; output JSON committed under `data/baselines/p3_replay.json` |
| Shadow gate | **Substituted** by feature-flag + same-PR flip commit. P6b shadow infra is not yet built; the feature flag provides equivalent rollback safety for a single-component change |
| Latency gate | Post-flip `tools.profile_scan_loop --limit 10` p95 within 1.5× of `scan_latency_post_phase1.json` |
| Rollback plan | Documented in §7 below |

---

## 6. Rollout within Phase 2

The PR lands in **two waves**, both inside the same PR:

**Wave 1 — Code + replay (flag off, behaviorally inert):**

1. New `bot/engine/trend_filter.py` + unit tests.
2. Wiring in `bot/engine/ensemble.py` behind the feature flag.
3. Config shape additive in `config_futures.yaml` + `config_spot.yaml`.
4. `bot/backtest/pipeline_eval.py` build-out with `TrendFilterReplayComponent`.
5. Replay run, output to `data/baselines/p3_replay.json` (committed).

**Approval gate.** I show the replay JSON to the user, summarise admit/veto counts and validator results, and ask **"flip the flag?"**.

**Wave 2 — Flip (only on explicit user approval):**

6. Single commit setting `use_two_tier: true` in both config files.
7. Bot restart (`pkill -f cryptobot_v3.*launcher.py; bash start.sh 2`).
8. Live observation **30–60 min**. **GO/NO-GO criteria** for the observation: (a) no crash / no `WARNING TREND VETO` storm (>50 % of evaluations being vetoed by the new filter would indicate a tuning bug); (b) at least one `long_allowed=true` evaluation appears in the bot log (proves the filter is no longer structurally short-locked); (c) post-flip scan-loop p95 ≤ 1.5× of `scan_latency_post_phase1.json` (latency budget). NO-GO on any of these reverts the flip commit before merge.
9. Commit the latency baseline + brief observation report; PR ready to merge.

Both waves stay on the `checkpoint/signal-quality-fixes` branch (or a new `phase2/trend-filter` branch, decided at plan-writing time).

---

## 7. Rollback plan

- **Wave 1 code is behaviorally inert.** No live behavior changes until the flag flip in Wave 2.
- **Single-commit flip.** Reverting the flip commit (or hotfix setting `use_two_tier: false`) returns the bot to single-tier behavior on the next scan. Bot restart not strictly required — config is re-read every cycle.
- **Full revert.** `git revert` the entire PR cleanly removes the new module, the harness changes, and the config additions. Only the new file in `data/baselines/` would remain (harmless artifact).
- **In-flight positions** are untouched per parent spec §7 — they exit on their existing stop / trailing / RL logic regardless of which filter is active.

---

## 8. Open questions / explicit assumptions

- **A1.** The 8 training-coin OHLCV parquets are assumed lossless from 2023-05 onward. If gap analysis (part of Task 4 in the plan) reveals significant holes, those gaps are skipped in the iteration and reported in the replay output but do not invalidate the validator gates.
- **A2.** The flag flip applies to both futures and spot in the same commit. If the live spot bot is not running, only the futures smoke test executes; the spot config still flips but lies dormant until the next spot start.
- **A3.** `pipeline_eval`'s plugin architecture is forward-compatible with regime / ensemble / calibration / risk components added in later phases. Each later phase adds its component without modifying the harness.

---

## 9. Out of scope (named to prevent drift)

- HMM regime layer changes — Phase 3 (P2).
- Reconciliation of the two ensemble code paths (`bot/engine/ensemble.py` vs `bot/agents/coordinator.py:MasterAgent`) — Phase 4 (P4).
- Trailing-stop / take-profit changes.
- Removal of the legacy single-tier config fields — deferred until ≥1 week of post-flip observation.
- Dashboard UI surfacing of the new filter verdict — out of overhaul scope.

---

*End of design spec. Awaiting user review per brainstorming skill before invoking writing-plans.*
