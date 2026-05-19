# Phase 2 — Two-Tier Trend Filter (P3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single-tier `veto_longs/veto_shorts` filter with a two-tier slope-magnitude filter (15m fast + 1h slow), behind a feature flag. Build out `bot/backtest/pipeline_eval.py` with a plugin architecture and validate the new filter against 3 years × 8 training coins. Flip the flag only on explicit user approval, gated by replay validators + a 30-60 min live smoke test.

**Architecture:** New pure-function `TrendFilter` module wired into `EnsembleEngine` behind `trend_filter.use_two_tier` (default `false`). `pipeline_eval` becomes the harness for component-by-component replay validation; Phase 2 registers one component (`TrendFilterReplayComponent`) and defines two automated validators that implement the parent spec's acceptance criteria (rally-window admits longs; monotonic-decline vetoes longs). Wave 1 lands code + replay flag-off (behaviorally inert). Wave 2 is the single flip commit on user approval.

**Tech Stack:** Python 3, pandas, pytest, existing `bot/` package, the 8 training-coin OHLCV parquets at `data/training/<SYMBOL>USDT_{15m,1h}.parquet` (3 years coverage), `tools.profile_scan_loop` for latency check.

**Plan position:** Plan 2 of N. Parent spec: `docs/superpowers/specs/2026-05-18-signal-quality-overhaul-design.md` §4 P3. Phase-2 spec: `docs/superpowers/specs/2026-05-19-phase2-trend-filter-design.md` (commit `209d9903`).

**Branching note.** Phase 1 sits on `checkpoint/signal-quality-fixes` (PR open against `main`). Phase 2 lands on a **new branch** `phase2/trend-filter` cut from current HEAD so the PRs are independent. After Phase 1 merges, rebase Phase 2 onto `main`.

---

## File map

**Create:**
- `bot/engine/trend_filter.py` — `TrendFilter` class, pure-function `check(dfs)`.
- `bot/tests/test_trend_filter.py` — unit tests per rule branch.
- `bot/backtest/components/__init__.py` — components subpackage.
- `bot/backtest/components/trend_filter.py` — `TrendFilterReplayComponent` wrapper.
- `bot/backtest/validators.py` — validator functions for P3 acceptance criteria.
- `bot/backtest/tests/test_pipeline_eval.py` — harness tests.
- `bot/backtest/tests/test_validators.py` — validator tests.
- `data/baselines/p3_replay.json` — replay output (committed as evidence; force-add through `.gitignore`).

**Modify:**
- `bot/engine/ensemble.py` — wire new filter behind `use_two_tier` flag; existing `_check_trend_veto` untouched.
- `bot/engine/bot.py` — pass `df_15m` into `decide()` alongside `df_1h` (15m already fetched by the per-symbol loop).
- `config_futures.yaml`, `config_spot.yaml` — additive two-tier subsections + `use_two_tier: false`.
- `bot/backtest/pipeline_eval.py` — replace "not implemented" skeleton with the plugin harness.

**Delete:** nothing in Phase 2 (legacy single-tier config fields stay for one more PR per spec §9).

---

## Conventions used in this plan

- **Test prelude.** All `bot/tests/test_*.py` start with:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
  ```
- **Atomic writes.** State writes go to `*.tmp.json` → `os.replace()`.
- **Commits.** One commit per task (test + implementation + cleanup land together); commit body ends with the `Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>` trailer.
- **Behavior gate.** Wave 1 commits keep `trend_filter.use_two_tier: false`. Bot behavior is unchanged. Wave 2 has exactly one commit that flips both configs.

---

## Task 1: Branch + spec confirmation

**Why first:** Isolate Phase 2 work from the open Phase 1 PR.

- [ ] **Step 1: Confirm current HEAD has Phase 1 + plan doc**

  Run:
  ```bash
  git log --oneline -3
  ```
  Expected: top commit is `209d9903 docs: Phase 2 (P3 trend filter) design spec` (or newer). Plan doc commit also visible.

- [ ] **Step 2: Cut Phase 2 branch from current HEAD**

  ```bash
  git checkout -b phase2/trend-filter
  git status
  ```
  Expected: branch `phase2/trend-filter` is current; working tree clean except for the runtime drift in `master_analysis.log`.

- [ ] **Step 3: Read both specs to ground the plan**

  Files to read (no edits):
  - `docs/superpowers/specs/2026-05-18-signal-quality-overhaul-design.md` §4 P3 (parent)
  - `docs/superpowers/specs/2026-05-19-phase2-trend-filter-design.md` (this phase)

  Verification: in your head, confirm the four rule definitions from §3.1:
  - `fast_up` = `EMA_fast > EMA_slow` AND fast slope > 0 (state, not edge-trigger)
  - `fast_down` = `EMA_fast < EMA_slow` AND fast slope < 0
  - `slow_strongly_up` = slow slope > `strong_slope_pct` AND `EMA_fast > EMA_slow`
  - `slow_strongly_down` = slow slope < −`strong_slope_pct` AND `EMA_fast < EMA_slow`

  No commit for this task — read-only.

---

## Task 2: `TrendFilter` module + unit tests

**Why second:** Pure-function module with no dependencies on the rest of the bot. TDD-friendly. Becomes the unit-test foundation for the wiring and replay tasks.

**Files:**
- Create: `bot/engine/trend_filter.py`
- Create: `bot/tests/test_trend_filter.py`

- [ ] **Step 1: Write the failing tests**

  Create `bot/tests/test_trend_filter.py`:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

  import numpy as np
  import pandas as pd
  import pytest
  from engine.trend_filter import TrendFilter


  def _ohlcv(closes: list[float], freq: str = "1h") -> pd.DataFrame:
      idx = pd.date_range("2026-01-01", periods=len(closes), freq=freq, tz="UTC")
      arr = np.asarray(closes, dtype=float)
      return pd.DataFrame({"open": arr, "high": arr, "low": arr,
                           "close": arr, "volume": np.ones_like(arr)}, index=idx)


  _CFG = {
      "fast": {"tf": "15m", "ema_fast": 20, "ema_slow": 50, "slope_lookback": 10},
      "slow": {"tf": "1h",  "ema_fast": 50, "ema_slow": 200, "slope_lookback": 20},
      "strong_slope_pct": 0.002,
  }


  def _linear(n: int, start: float, slope: float) -> list[float]:
      return [start + i * slope for i in range(n)]


  def test_long_allowed_when_fast_up_slow_flat():
      tf = TrendFilter(_CFG)
      # Fast tier: 80 bars climbing → fast EMAs in uptrend
      # Slow tier: 250 bars flat → slow EMAs equal, slope ~ 0
      v = tf.check({
          "15m": _ohlcv(_linear(80, 100.0, 0.5), freq="15min"),
          "1h":  _ohlcv([100.0] * 250, freq="1h"),
      })
      assert v["long_allowed"]  is True
      assert v["short_allowed"] is False
      assert v["fast"]["direction"] == "up"
      assert v["slow"]["strong"]    is False


  def test_long_blocked_when_slow_strongly_down():
      tf = TrendFilter(_CFG)
      # Fast tier: climbing (would normally admit longs)
      # Slow tier: declining ≥ 0.2 %/bar over the lookback → strongly down
      v = tf.check({
          "15m": _ohlcv(_linear(80, 100.0, 0.5), freq="15min"),
          "1h":  _ohlcv(_linear(250, 200.0, -0.6), freq="1h"),
      })
      assert v["long_allowed"]  is False
      assert v["slow"]["strong"] is True
      assert v["slow"]["direction"] == "down"


  def test_short_allowed_when_fast_down_slow_flat():
      tf = TrendFilter(_CFG)
      v = tf.check({
          "15m": _ohlcv(_linear(80, 100.0, -0.5), freq="15min"),
          "1h":  _ohlcv([100.0] * 250, freq="1h"),
      })
      assert v["short_allowed"] is True
      assert v["long_allowed"]  is False


  def test_short_blocked_when_slow_strongly_up():
      tf = TrendFilter(_CFG)
      v = tf.check({
          "15m": _ohlcv(_linear(80, 100.0, -0.5), freq="15min"),
          "1h":  _ohlcv(_linear(250, 100.0, 0.6), freq="1h"),
      })
      assert v["short_allowed"] is False
      assert v["slow"]["strong"] is True


  def test_neither_allowed_when_fast_flat():
      tf = TrendFilter(_CFG)
      v = tf.check({
          "15m": _ohlcv([100.0] * 80, freq="15min"),
          "1h":  _ohlcv([100.0] * 250, freq="1h"),
      })
      assert v["long_allowed"]  is False
      assert v["short_allowed"] is False
      assert v["fast"]["direction"] == "flat"


  def test_insufficient_history_returns_neutral():
      tf = TrendFilter(_CFG)
      v = tf.check({
          "15m": _ohlcv([100.0] * 10, freq="15min"),  # too few bars for EMA_slow=50
          "1h":  _ohlcv([100.0] * 30, freq="1h"),     # too few for EMA_slow=200
      })
      assert v["long_allowed"]  is False
      assert v["short_allowed"] is False
      assert "insufficient" in v["reasoning"].lower()


  def test_slope_threshold_is_inclusive_of_strong_band():
      tf = TrendFilter(_CFG)
      # Slow slope just above 0.2 %/bar → strong; just below → not strong.
      # 0.002 × 200 = 0.4 absolute / bar at start price 200 with 20-bar lookback
      # is +0.002 per bar; use a smaller margin to be near boundary.
      v_strong = tf.check({
          "15m": _ohlcv(_linear(80, 100.0, -0.5), freq="15min"),
          "1h":  _ohlcv(_linear(250, 100.0, 0.6), freq="1h"),  # well above threshold
      })
      v_weak = tf.check({
          "15m": _ohlcv(_linear(80, 100.0, -0.5), freq="15min"),
          "1h":  _ohlcv(_linear(250, 100.0, 0.05), freq="1h"),  # well below threshold
      })
      assert v_strong["slow"]["strong"] is True
      assert v_weak["slow"]["strong"]   is False


  def test_reasoning_string_is_human_readable():
      tf = TrendFilter(_CFG)
      v = tf.check({
          "15m": _ohlcv(_linear(80, 100.0, 0.5), freq="15min"),
          "1h":  _ohlcv(_linear(250, 200.0, -0.6), freq="1h"),
      })
      # Must mention slow tier and "strong" to be debuggable in bot logs
      assert "slow" in v["reasoning"].lower()
      assert "strong" in v["reasoning"].lower() or "block" in v["reasoning"].lower()
  ```

- [ ] **Step 2: Run tests, verify failure**

  Run: `cd bot && python3 -m pytest tests/test_trend_filter.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'engine.trend_filter'`.

- [ ] **Step 3: Implement `trend_filter.py`**

  Create `bot/engine/trend_filter.py`:
  ```python
  """Two-tier slope-magnitude trend filter (Phase 2 / P3).

  Pure-function check over OHLCV dataframes for two timeframes (fast + slow).
  Decides whether long / short entries are admissible based on:

    fast_up    = EMA_fast > EMA_slow  on fast tier AND fast slope > 0
    fast_down  = EMA_fast < EMA_slow  on fast tier AND fast slope < 0
    slow_strongly_up   = slow slope >  strong_slope_pct AND EMA_fast > EMA_slow
    slow_strongly_down = slow slope < -strong_slope_pct AND EMA_fast < EMA_slow

    long_allowed  = fast_up   AND NOT slow_strongly_down
    short_allowed = fast_down AND NOT slow_strongly_up

  Slope is the relative change of EMA_fast over `slope_lookback` bars (state,
  not edge-trigger — the filter cares whether the trend *is* up/down, not
  whether it just crossed).
  """
  from __future__ import annotations

  from typing import Dict
  import pandas as pd


  class TrendFilter:
      def __init__(self, cfg: dict) -> None:
          self.cfg = cfg
          self.strong_slope_pct: float = float(cfg.get("strong_slope_pct", 0.002))
          self._fast_cfg = cfg.get("fast", {})
          self._slow_cfg = cfg.get("slow", {})

      def _tier_state(self, df: pd.DataFrame, tier_cfg: dict) -> dict:
          ema_fast = int(tier_cfg.get("ema_fast", 20))
          ema_slow = int(tier_cfg.get("ema_slow", 50))
          lookback = int(tier_cfg.get("slope_lookback", 10))
          required = ema_slow + lookback + 1
          if df is None or len(df) < required:
              return {"ok": False, "direction": "flat", "slope_pct": 0.0,
                      "ema_fast": None, "ema_slow": None}
          close = df["close"]
          ef_series = close.ewm(span=ema_fast, adjust=False).mean()
          es_series = close.ewm(span=ema_slow, adjust=False).mean()
          ef_now  = float(ef_series.iloc[-1])
          es_now  = float(es_series.iloc[-1])
          ef_prev = float(ef_series.iloc[-1 - lookback])
          slope_pct = (ef_now - ef_prev) / (ef_prev + 1e-12)
          if ef_now > es_now and slope_pct > 0:
              direction = "up"
          elif ef_now < es_now and slope_pct < 0:
              direction = "down"
          else:
              direction = "flat"
          return {"ok": True, "direction": direction, "slope_pct": float(slope_pct),
                  "ema_fast": ef_now, "ema_slow": es_now}

      def check(self, dfs: Dict[str, pd.DataFrame]) -> dict:
          fast_tf = self._fast_cfg.get("tf", "15m")
          slow_tf = self._slow_cfg.get("tf", "1h")
          fast = self._tier_state(dfs.get(fast_tf), self._fast_cfg)
          slow = self._tier_state(dfs.get(slow_tf), self._slow_cfg)

          if not fast["ok"] or not slow["ok"]:
              return {
                  "long_allowed": False, "short_allowed": False,
                  "reasoning": "insufficient history on fast or slow tier",
                  "fast": {"direction": fast["direction"], "slope_pct": fast["slope_pct"]},
                  "slow": {"direction": slow["direction"], "slope_pct": slow["slope_pct"],
                           "strong": False},
              }

          slow_strong_up   = slow["slope_pct"] >  self.strong_slope_pct and slow["direction"] == "up"
          slow_strong_down = slow["slope_pct"] < -self.strong_slope_pct and slow["direction"] == "down"
          slow_strong = slow_strong_up or slow_strong_down

          long_allowed  = (fast["direction"] == "up")   and not slow_strong_down
          short_allowed = (fast["direction"] == "down") and not slow_strong_up

          if long_allowed:
              reason = f"fast={fast['direction']} slow={slow['direction']} → LONG allowed"
          elif short_allowed:
              reason = f"fast={fast['direction']} slow={slow['direction']} → SHORT allowed"
          elif fast["direction"] == "up" and slow_strong_down:
              reason = (f"fast=up but slow strongly down "
                        f"(slope={slow['slope_pct']:.4f}) → LONG blocked")
          elif fast["direction"] == "down" and slow_strong_up:
              reason = (f"fast=down but slow strongly up "
                        f"(slope={slow['slope_pct']:.4f}) → SHORT blocked")
          else:
              reason = f"fast={fast['direction']} → no direction admitted"

          return {
              "long_allowed":  long_allowed,
              "short_allowed": short_allowed,
              "reasoning":     reason,
              "fast": {"direction": fast["direction"], "slope_pct": fast["slope_pct"]},
              "slow": {"direction": slow["direction"], "slope_pct": slow["slope_pct"],
                       "strong": slow_strong},
          }
  ```

- [ ] **Step 4: Run tests, verify pass**

  Run: `cd bot && python3 -m pytest tests/test_trend_filter.py -v`
  Expected: 8 passed.

- [ ] **Step 5: Run full bot suite for non-regression**

  Run: `cd bot && python3 -m pytest tests/ -q`
  Expected: 146 + 8 = 154 passed (no failures).

- [ ] **Step 6: Commit**

  ```bash
  git add bot/engine/trend_filter.py bot/tests/test_trend_filter.py
  git commit -m "$(cat <<'EOF'
  feat(engine): TrendFilter module — two-tier slope-magnitude filter

  bot/engine/trend_filter.py implements the P3 two-tier filter as a pure
  function over per-timeframe OHLCV dataframes. Returns
  {long_allowed, short_allowed, reasoning, fast, slow} given config with
  fast/slow tier subsections and a strong_slope_pct threshold. State-based
  (not edge-trigger): cares whether the trend is up/down, not whether it
  just crossed. No I/O; constructed once at bot startup.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 3: Config schema — additive two-tier section

**Why third:** The wiring in Task 4 reads the new flag and subsections. Land config first so wiring fails loud if the keys are missing.

**Files:**
- Modify: `config_futures.yaml`
- Modify: `config_spot.yaml`

- [ ] **Step 1: Update `config_futures.yaml`**

  Replace the `trend_filter:` block (lines 71-77) with:
  ```yaml
  trend_filter:
    enabled: true
    use_two_tier: false       # P3 flag; flip in a separate commit after replay
    # legacy single-tier (unchanged; used when use_two_tier=false)
    tf: 1h
    ema_fast: 50
    ema_slow: 200
    veto_longs: true
    veto_shorts: true
    # two-tier (used when use_two_tier=true)
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
    strong_slope_pct: 0.002   # 0.2 %/bar = "strongly opposed"
  ```

- [ ] **Step 2: Apply same change to `config_spot.yaml`**

  Replace the matching `trend_filter:` block in `config_spot.yaml` with the exact same content as above. Spot inherits behaviour from BaseBot, so the config must match.

- [ ] **Step 3: Verify YAML parses**

  Run:
  ```bash
  python3 -c "
  import yaml
  for f in ('config_futures.yaml', 'config_spot.yaml'):
      d = yaml.safe_load(open(f))['trend_filter']
      assert d['use_two_tier'] is False
      assert d['fast']['tf'] == '15m'
      assert d['slow']['tf'] == '1h'
      assert d['strong_slope_pct'] == 0.002
      print(f, 'ok')
  "
  ```
  Expected: both files print `ok`.

- [ ] **Step 4: Commit**

  ```bash
  git add config_futures.yaml config_spot.yaml
  git commit -m "$(cat <<'EOF'
  config: additive two-tier trend_filter section (flag default false)

  Adds fast/slow tier subsections + strong_slope_pct + use_two_tier flag to
  both config_futures.yaml and config_spot.yaml. Legacy single-tier fields
  (tf/ema_fast/ema_slow/veto_longs/veto_shorts) preserved unchanged so the
  bot keeps the existing behaviour while use_two_tier is false.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 4: Wire `TrendFilter` into ensemble (flag off)

**Why fourth:** Connects the new module to the production decision path, but gated by `use_two_tier: false`. Live bot behaviour does not change yet.

**Files:**
- Modify: `bot/engine/ensemble.py` (lines 32-103 area + a new branch in `decide`)
- Modify: `bot/engine/bot.py` (line ~429 area — pass `df_15m` alongside `df_1h`)
- Create: `bot/tests/test_ensemble_trend_filter_wiring.py`

- [ ] **Step 1: Read current ensemble wiring**

  Files to read (no edits):
  - `bot/engine/ensemble.py` lines 1-130 (constructor, `decide`, `_check_trend_veto`).
  - `bot/engine/bot.py` lines 420-435 (where the ensemble is instantiated and `decide` is called).
  - Identify where `df_1h` is sourced and where `df_15m` would come from. The per-symbol loop already fetches both `dfs` dicts; we just need to thread `df_15m` into `decide()`.

- [ ] **Step 2: Write the failing wiring tests**

  Create `bot/tests/test_ensemble_trend_filter_wiring.py`:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

  import pandas as pd
  from unittest.mock import MagicMock, patch
  from engine.ensemble import EnsembleEngine


  def _dummy_dfs():
      idx = pd.date_range("2026-01-01", periods=260, freq="1h", tz="UTC")
      df1h = pd.DataFrame({"open":  [100]*260, "high": [101]*260, "low": [99]*260,
                            "close": [100 + i*0.01 for i in range(260)],
                            "volume":[1]*260}, index=idx)
      idx15 = pd.date_range("2026-01-01", periods=120, freq="15min", tz="UTC")
      df15 = pd.DataFrame({"open":  [100]*120, "high": [101]*120, "low": [99]*120,
                            "close": [100 + i*0.02 for i in range(120)],
                            "volume":[1]*120}, index=idx15)
      return {"15m": df15, "1h": df1h}


  def _cfg_with_flag(use_two_tier: bool) -> dict:
      return {
          "enabled":      True,
          "use_two_tier": use_two_tier,
          "tf":           "1h", "ema_fast": 50, "ema_slow": 200,
          "veto_longs":   True, "veto_shorts": True,
          "fast": {"tf": "15m", "ema_fast": 20, "ema_slow": 50, "slope_lookback": 10},
          "slow": {"tf": "1h",  "ema_fast": 50, "ema_slow": 200, "slope_lookback": 20},
          "strong_slope_pct": 0.002,
      }


  def test_flag_false_uses_legacy_path():
      eng = EnsembleEngine(MagicMock(), MagicMock(), MagicMock(),
                           trend_filter=_cfg_with_flag(False))
      assert eng._tf2 is None, "two-tier filter must not be constructed when flag is off"
      assert hasattr(eng, "_check_trend_veto"), "legacy method must remain"


  def test_flag_true_constructs_two_tier_filter():
      from engine.trend_filter import TrendFilter
      eng = EnsembleEngine(MagicMock(), MagicMock(), MagicMock(),
                           trend_filter=_cfg_with_flag(True))
      assert isinstance(eng._tf2, TrendFilter)


  def test_flag_true_routes_buy_through_two_tier(monkeypatch):
      eng = EnsembleEngine(MagicMock(), MagicMock(), MagicMock(),
                           trend_filter=_cfg_with_flag(True))
      called = {"n": 0}
      def fake_check(dfs):
          called["n"] += 1
          return {"long_allowed": False, "short_allowed": True,
                  "reasoning": "slow strongly down → LONG blocked",
                  "fast": {"direction": "up", "slope_pct": 0.01},
                  "slow": {"direction": "down", "slope_pct": -0.005, "strong": True}}
      monkeypatch.setattr(eng._tf2, "check", fake_check)
      veto_reason = eng._apply_trend_filter("BUY", _dummy_dfs())
      assert called["n"] == 1
      assert veto_reason is not None
      assert "LONG blocked" in veto_reason


  def test_flag_true_admits_when_two_tier_allows(monkeypatch):
      eng = EnsembleEngine(MagicMock(), MagicMock(), MagicMock(),
                           trend_filter=_cfg_with_flag(True))
      monkeypatch.setattr(eng._tf2, "check", lambda dfs: {
          "long_allowed": True, "short_allowed": False, "reasoning": "ok",
          "fast": {"direction": "up", "slope_pct": 0.01},
          "slow": {"direction": "up", "slope_pct": 0.001, "strong": False}})
      veto_reason = eng._apply_trend_filter("BUY", _dummy_dfs())
      assert veto_reason is None
  ```

- [ ] **Step 3: Run tests, verify failure**

  Run: `cd bot && python3 -m pytest tests/test_ensemble_trend_filter_wiring.py -v`
  Expected: FAIL — `_tf2`, `_apply_trend_filter` don't exist yet.

- [ ] **Step 4: Modify `EnsembleEngine.__init__`**

  In `bot/engine/ensemble.py`, replace the constructor body (around line 32-38) to set up `_tf2`:
  ```python
  def __init__(self, smc_agent, tech_agent, macro_agent=None, trend_filter=None):
      self.smc       = smc_agent
      self.technical = tech_agent
      self.macro     = macro_agent
      # trend_filter: dict from config.trend_filter — vetoes counter-trend signals
      # When `use_two_tier=True`, route through bot/engine/trend_filter.TrendFilter.
      self.trend_filter = trend_filter or {}
      self._tf2 = None
      if self.trend_filter.get("use_two_tier"):
          from .trend_filter import TrendFilter
          self._tf2 = TrendFilter(self.trend_filter)
  ```

- [ ] **Step 5: Add `_apply_trend_filter` method**

  Insert this new method into `EnsembleEngine` (immediately above `_check_trend_veto`, around line 105):
  ```python
  def _apply_trend_filter(self, action: str, dfs: dict) -> Optional[str]:
      """Dispatch to two-tier or legacy filter. Returns veto reason string or None."""
      if self._tf2 is not None:
          v = self._tf2.check(dfs)
          if action == "BUY"  and not v["long_allowed"]:  return f"2tier:{v['reasoning']}"
          if action == "SELL" and not v["short_allowed"]: return f"2tier:{v['reasoning']}"
          return None
      # Legacy path: existing _check_trend_veto on the 1h df.
      return self._check_trend_veto(dfs.get("1h"), action)
  ```

- [ ] **Step 6: Update `decide` to pass dfs through**

  Find the existing veto block in `decide` (around lines 93-103):
  ```python
  if self.trend_filter.get("enabled") and result.action in ("BUY", "SELL"):
      veto_reason = self._check_trend_veto(df, result.action)
      if veto_reason:
          log.info(f"TREND VETO {symbol} → HOLD (was {result.action}): {veto_reason}")
          result.action = "HOLD"
          result.source = f"trend_veto:{veto_reason}"
  ```

  Replace with:
  ```python
  if self.trend_filter.get("enabled") and result.action in ("BUY", "SELL"):
      veto_reason = self._apply_trend_filter(result.action, dfs)
      if veto_reason:
          log.info(f"TREND VETO {symbol} → HOLD (was {result.action}): {veto_reason}")
          result.action = "HOLD"
          result.source = f"trend_veto:{veto_reason}"
  ```

  This requires `decide()` to receive `dfs` (a dict). Locate the existing `decide` signature; if it currently takes `df` (a single 1h DataFrame), change to `dfs: dict`. Update the per-symbol caller in `bot/engine/bot.py` to pass `{"15m": dfs["15m"], "1h": dfs["1h"]}` instead of just `df_1h`.

  If `decide()` already receives a `dfs` dict (check the current signature), only the body change above is needed.

- [ ] **Step 7: Run wiring tests, verify pass**

  Run: `cd bot && python3 -m pytest tests/test_ensemble_trend_filter_wiring.py -v`
  Expected: 4 passed.

- [ ] **Step 8: Run full bot suite for non-regression**

  Run: `cd bot && python3 -m pytest tests/ -q`
  Expected: 154 + 4 = 158 passed.

- [ ] **Step 9: Commit**

  ```bash
  git add bot/engine/ensemble.py bot/engine/bot.py bot/tests/test_ensemble_trend_filter_wiring.py
  git commit -m "$(cat <<'EOF'
  feat(engine): wire TrendFilter into EnsembleEngine behind flag

  EnsembleEngine.__init__ now constructs a TrendFilter instance when
  trend_filter.use_two_tier=true; otherwise stays on the legacy
  _check_trend_veto path. _apply_trend_filter() dispatches between the
  two paths. decide() now receives a dfs dict (15m + 1h) instead of just
  df_1h. With the flag still false (Task 3), behaviour is unchanged.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 5: `pipeline_eval` plugin harness

**Why fifth:** Replaces the "not implemented" skeleton with the parquet-walking harness all Phase 2+ replays depend on.

**Files:**
- Modify: `bot/backtest/pipeline_eval.py`
- Create: `bot/backtest/tests/test_pipeline_eval.py`

- [ ] **Step 1: Read existing skeleton**

  File: `bot/backtest/pipeline_eval.py` (currently a Phase-1 stub printing "not implemented" on `--run`).

- [ ] **Step 2: Write the failing harness tests**

  Create `bot/backtest/tests/test_pipeline_eval.py`:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

  import json
  from pathlib import Path
  import numpy as np
  import pandas as pd
  import pytest
  from backtest.pipeline_eval import (
      load_symbol_dfs, walk_forward_iter, run_components, PipelineComponent
  )


  def _make_parquet(tmp: Path, symbol: str, tf: str, n: int, slope: float = 0.0):
      idx = pd.date_range("2026-01-01", periods=n, freq=tf.replace("m", "min"), tz="UTC")
      closes = 100 + np.arange(n) * slope
      df = pd.DataFrame({"open": closes, "high": closes, "low": closes,
                          "close": closes, "volume": np.ones(n)}, index=idx)
      df.index.name = "timestamp"
      out = tmp / f"{symbol}USDT_{tf}.parquet"
      df.to_parquet(out)
      return out


  def test_load_symbol_dfs_reads_both_tfs(tmp_path):
      _make_parquet(tmp_path, "BTC", "15m", 300, 0.05)
      _make_parquet(tmp_path, "BTC", "1h",  100, 0.1)
      dfs = load_symbol_dfs(tmp_path, "BTCUSDT", ["15m", "1h"])
      assert "15m" in dfs and "1h" in dfs
      assert len(dfs["15m"]) == 300
      assert len(dfs["1h"]) == 100


  def test_walk_forward_iter_emits_truncated_slices(tmp_path):
      _make_parquet(tmp_path, "BTC", "15m", 100, 0.05)
      _make_parquet(tmp_path, "BTC", "1h",   25, 0.1)
      dfs = load_symbol_dfs(tmp_path, "BTCUSDT", ["15m", "1h"])
      points = list(walk_forward_iter(dfs, cadence="1h", base_tf="1h"))
      # Each yielded item is (ts, sliced_dfs); 1h has 25 bars → up to 25 iterations
      assert len(points) == 25
      ts0, dfs0 = points[0]
      assert dfs0["1h"].index[-1] == ts0
      # No look-ahead: 15m slice does not contain bars beyond ts0
      assert dfs0["15m"].index.max() <= ts0


  class _CountingComponent:
      name = "counter"
      def __init__(self):
          self.calls = 0
      def evaluate(self, symbol, ts, dfs):
          self.calls += 1
          return {"symbol": symbol, "ts": ts.isoformat()}
      def validators(self):
          return []


  def test_run_components_calls_each_point(tmp_path):
      _make_parquet(tmp_path, "BTC", "15m", 100, 0.05)
      _make_parquet(tmp_path, "BTC", "1h",   25, 0.1)
      c = _CountingComponent()
      out_path = tmp_path / "out.json"
      summary = run_components(
          parquet_dir=tmp_path,
          symbols=["BTCUSDT"],
          tfs=["15m", "1h"],
          cadence="1h",
          base_tf="1h",
          components=[c],
          out_path=out_path,
      )
      assert c.calls == 25
      assert out_path.exists()
      data = json.loads(out_path.read_text())
      assert "summary" in data
      assert "validators" in data
      assert data["coverage"]["n_evaluations"] == 25
  ```

- [ ] **Step 3: Run tests, verify failure**

  Run: `python3 -m pytest bot/backtest/tests/test_pipeline_eval.py -v`
  Expected: FAIL — `load_symbol_dfs`, `walk_forward_iter`, `run_components` don't exist.

- [ ] **Step 4: Implement the harness**

  Replace `bot/backtest/pipeline_eval.py` contents with:
  ```python
  """End-to-end pipeline backtest harness.

  Plugin architecture: components implement evaluate(symbol, ts, dfs) and
  optional validators(). The harness walks the per-symbol per-tf parquets
  forward in time, slices each tf up to the current timestamp (no look-ahead),
  and invokes every registered component at each evaluation point.

  Used by Phase 2+ to validate components against 3 years × 8 training coins.
  """
  from __future__ import annotations

  import argparse
  import json
  import os
  import sys
  from datetime import datetime, timezone
  from pathlib import Path
  from typing import Callable, Iterable, List, Optional, Protocol, Sequence

  import pandas as pd


  class PipelineComponent(Protocol):
      name: str
      def evaluate(self, symbol: str, ts: pd.Timestamp,
                   dfs: dict) -> dict: ...
      def validators(self) -> list: ...


  def load_symbol_dfs(parquet_dir: Path, symbol: str,
                      tfs: Sequence[str]) -> dict:
      """Load OHLCV parquet for each requested tf for a single symbol.

      File layout: <parquet_dir>/<SYMBOL>_<tf>.parquet (e.g. BTCUSDT_15m.parquet).
      """
      dfs = {}
      for tf in tfs:
          p = Path(parquet_dir) / f"{symbol}_{tf}.parquet"
          if not p.exists():
              continue
          df = pd.read_parquet(p)
          if df.index.tz is None:
              df.index = df.index.tz_localize("UTC")
          dfs[tf] = df
      return dfs


  def walk_forward_iter(dfs: dict, cadence: str, base_tf: str):
      """Yield (ts, sliced_dfs) for each cadence point in the base_tf history.

      sliced_dfs[tf] is df[:ts] (closed-right), so no look-ahead is possible.
      """
      base = dfs.get(base_tf)
      if base is None or len(base) == 0:
          return
      # Resample base to cadence to get evaluation timestamps.
      cadence_pd = cadence.replace("m", "min") if cadence.endswith("m") and cadence != "1m" else cadence
      stamps = base.resample(cadence_pd).last().dropna(how="all").index
      for ts in stamps:
          sliced = {tf: df.loc[:ts] for tf, df in dfs.items()}
          yield ts, sliced


  def run_components(parquet_dir: Path, symbols: Sequence[str], tfs: Sequence[str],
                     cadence: str, base_tf: str, components: Sequence[PipelineComponent],
                     out_path: Path) -> dict:
      """Run all components over all (symbol, ts) points and write output JSON."""
      results: dict = {c.name: [] for c in components}
      coverage_symbols: list = []
      n_evaluations = 0
      start_ts: Optional[pd.Timestamp] = None
      end_ts:   Optional[pd.Timestamp] = None

      for symbol in symbols:
          dfs = load_symbol_dfs(Path(parquet_dir), symbol, tfs)
          if not dfs:
              continue
          coverage_symbols.append(symbol)
          for ts, sdfs in walk_forward_iter(dfs, cadence=cadence, base_tf=base_tf):
              if start_ts is None or ts < start_ts: start_ts = ts
              if end_ts is None   or ts > end_ts:   end_ts = ts
              for c in components:
                  rec = c.evaluate(symbol, ts, sdfs)
                  results[c.name].append({"symbol": symbol, **rec})
              n_evaluations += 1

      summary = {c.name: _summarise(results[c.name]) for c in components}
      validators_out: dict = {}
      for c in components:
          for v in (c.validators() or []):
              vres = v(results[c.name])
              validators_out[vres["name"]] = vres

      out = {
          "captured_at": datetime.now(timezone.utc).isoformat(),
          "components":  [c.name for c in components],
          "coverage": {
              "symbols": coverage_symbols,
              "start":   start_ts.isoformat() if start_ts is not None else None,
              "end":     end_ts.isoformat()   if end_ts   is not None else None,
              "cadence": cadence,
              "n_evaluations": n_evaluations,
          },
          "summary":    summary,
          "validators": validators_out,
      }
      tmp = Path(out_path).with_suffix(Path(out_path).suffix + ".tmp")
      tmp.parent.mkdir(parents=True, exist_ok=True)
      with open(tmp, "w") as fh:
          json.dump(out, fh, indent=2, default=str)
      os.replace(tmp, out_path)
      return out


  def _summarise(records: list) -> dict:
      if not records:
          return {"count": 0}
      out: dict = {"count": len(records)}
      # Per-component summarisation is component-specific; the harness only
      # contributes the count. Components may post-process out["summary"]
      # in their own way by inspecting results before the file is written
      # (Phase 3+ may want richer roll-ups).
      return out


  def build_parser() -> argparse.ArgumentParser:
      p = argparse.ArgumentParser(prog="pipeline_eval",
                                  description="End-to-end pipeline backtest harness.")
      p.add_argument("--run", action="store_true", help="Execute the backtest.")
      p.add_argument("--parquet", default="data/training",
                     help="Directory containing <SYMBOL>USDT_<tf>.parquet files")
      p.add_argument("--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,SOLUSDT,DOGEUSDT,ADAUSDT,LINKUSDT")
      p.add_argument("--tfs", default="15m,1h")
      p.add_argument("--cadence", default="1h")
      p.add_argument("--base-tf", default="1h")
      p.add_argument("--component", default="trend_filter",
                     choices=["trend_filter"])
      p.add_argument("--out", default="data/baselines/p3_replay.json")
      p.add_argument("--config", default="config_futures.yaml")
      return p


  def main(argv: list[str]) -> int:
      args = build_parser().parse_args(argv)
      if not args.run:
          return 0

      # Component registration is wave-by-wave; Phase 2 registers trend_filter.
      if args.component == "trend_filter":
          from .components.trend_filter import TrendFilterReplayComponent
          import yaml
          cfg = yaml.safe_load(open(args.config))["trend_filter"]
          # Replay uses the two-tier branch regardless of live flag (it's a
          # what-if check); pass the two-tier sub-config to TrendFilter.
          replay_cfg = {
              "fast": cfg["fast"], "slow": cfg["slow"],
              "strong_slope_pct": cfg["strong_slope_pct"],
          }
          components = [TrendFilterReplayComponent(replay_cfg)]
      else:
          print(f"unknown component: {args.component}", file=sys.stderr)
          return 2

      run_components(
          parquet_dir=Path(args.parquet),
          symbols=args.symbols.split(","),
          tfs=args.tfs.split(","),
          cadence=args.cadence,
          base_tf=args.base_tf,
          components=components,
          out_path=Path(args.out),
      )
      return 0


  if __name__ == "__main__":
      raise SystemExit(main(sys.argv[1:]))
  ```

- [ ] **Step 5: Run harness tests, verify pass**

  Run: `python3 -m pytest bot/backtest/tests/test_pipeline_eval.py -v`
  Expected: 3 passed.

- [ ] **Step 6: Verify CLI --help still works**

  Run: `python3 -m bot.backtest.pipeline_eval --help`
  Expected: argparse usage block including `--component`, `--cadence`, etc. (no crash).

- [ ] **Step 7: Commit**

  ```bash
  git add bot/backtest/pipeline_eval.py bot/backtest/tests/test_pipeline_eval.py
  git commit -m "$(cat <<'EOF'
  feat(backtest): pipeline_eval plugin harness for component replay

  Replaces the Phase-1 "not implemented" skeleton with a parquet-walking
  harness. Components implement evaluate(symbol, ts, dfs) and optional
  validators(). Harness slices each tf up to current ts (no look-ahead),
  invokes every component at every evaluation point, then writes a JSON
  summary with coverage + per-component summary + validator results.

  Phase 2 registers only the trend_filter component (Task 6). Later phases
  add regime / ensemble / calibration components against the same harness.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 6: `TrendFilterReplayComponent`

**Why sixth:** The first plugin component; satisfies the harness Protocol and exposes the two acceptance-criteria validators.

**Files:**
- Create: `bot/backtest/components/__init__.py`
- Create: `bot/backtest/components/trend_filter.py`
- Modify: `bot/backtest/tests/test_pipeline_eval.py` (add an integration test for the trend filter component)

- [ ] **Step 1: Create components subpackage**

  ```bash
  mkdir -p bot/backtest/components
  touch bot/backtest/components/__init__.py
  ```

- [ ] **Step 2: Write the failing component test**

  Append to `bot/backtest/tests/test_pipeline_eval.py`:
  ```python
  def test_trend_filter_replay_component_records_verdicts(tmp_path):
      from backtest.components.trend_filter import TrendFilterReplayComponent
      import numpy as np
      # 80 bars climbing at fast tf, 250 flat at slow tf → admits long
      _make_parquet(tmp_path, "BTC", "15m", 80, slope=0.5)
      _make_parquet(tmp_path, "BTC", "1h",  250, slope=0.0)
      cfg = {
          "fast": {"tf": "15m", "ema_fast": 20, "ema_slow": 50, "slope_lookback": 10},
          "slow": {"tf": "1h",  "ema_fast": 50, "ema_slow": 200, "slope_lookback": 20},
          "strong_slope_pct": 0.002,
      }
      c = TrendFilterReplayComponent(cfg)
      out_path = tmp_path / "out.json"
      summary = run_components(parquet_dir=tmp_path, symbols=["BTCUSDT"],
                                tfs=["15m", "1h"], cadence="1h", base_tf="1h",
                                components=[c], out_path=out_path)
      sumc = summary["summary"]["trend_filter"]
      assert sumc["count"] >= 1
      # At least one record should have long_allowed=True near the end of the climb
      data = json.loads(out_path.read_text())
      assert data["components"] == ["trend_filter"]
  ```

- [ ] **Step 3: Run test, verify failure**

  Run: `python3 -m pytest bot/backtest/tests/test_pipeline_eval.py::test_trend_filter_replay_component_records_verdicts -v`
  Expected: FAIL — `backtest.components.trend_filter` does not exist.

- [ ] **Step 4: Implement the component**

  Create `bot/backtest/components/trend_filter.py`:
  ```python
  """TrendFilter replay component for bot.backtest.pipeline_eval.

  Wraps engine.trend_filter.TrendFilter so it can be invoked from the harness's
  walk-forward iterator. Each evaluation records the verdict (long_allowed,
  short_allowed, fast/slow direction, slow.strong) for later aggregation by
  the validators in bot/backtest/validators.py.
  """
  from __future__ import annotations

  import pandas as pd
  from typing import Callable, List

  from engine.trend_filter import TrendFilter


  class TrendFilterReplayComponent:
      name = "trend_filter"

      def __init__(self, cfg: dict) -> None:
          self._tf = TrendFilter(cfg)

      def evaluate(self, symbol: str, ts: pd.Timestamp, dfs: dict) -> dict:
          v = self._tf.check(dfs)
          return {
              "ts":             ts.isoformat(),
              "long_allowed":   bool(v["long_allowed"]),
              "short_allowed":  bool(v["short_allowed"]),
              "fast_direction": v["fast"]["direction"],
              "slow_direction": v["slow"]["direction"],
              "slow_strong":    bool(v["slow"].get("strong", False)),
          }

      def validators(self) -> List[Callable]:
          # Wired up in Task 7.
          return []
  ```

- [ ] **Step 5: Run test, verify pass**

  Run: `python3 -m pytest bot/backtest/tests/test_pipeline_eval.py -v`
  Expected: 4 passed (3 from Task 5 + 1 new).

- [ ] **Step 6: Commit**

  ```bash
  git add bot/backtest/components/__init__.py bot/backtest/components/trend_filter.py \
          bot/backtest/tests/test_pipeline_eval.py
  git commit -m "$(cat <<'EOF'
  feat(backtest): TrendFilterReplayComponent — first pipeline_eval plugin

  Wraps engine.trend_filter.TrendFilter so the harness can invoke it per
  evaluation point. Records {long_allowed, short_allowed, fast_direction,
  slow_direction, slow_strong} per (symbol, ts) for validator aggregation
  in Task 7.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 7: Acceptance-criteria validators

**Why seventh:** Turns replay output into a pass/fail signal that matches the parent spec §4.3 acceptance criteria.

**Files:**
- Create: `bot/backtest/validators.py`
- Create: `bot/backtest/tests/test_validators.py`
- Modify: `bot/backtest/components/trend_filter.py` (register validators)

- [ ] **Step 1: Write the failing validator tests**

  Create `bot/backtest/tests/test_validators.py`:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

  from backtest.validators import (
      rally_window_admits_longs, monotonic_decline_vetoes_longs,
  )


  def _rec(symbol, ts, long_allowed, short_allowed=False,
           fast="flat", slow="flat", strong=False):
      return {"symbol": symbol, "ts": ts,
              "long_allowed": long_allowed, "short_allowed": short_allowed,
              "fast_direction": fast, "slow_direction": slow, "slow_strong": strong}


  def test_rally_window_admits_longs_passes_with_evidence():
      # Synthesise: window 2026-04-01T00..2026-04-02T00, 4 long admits, 8 symbols
      # rallying (simulated via aux "rally_evidence" hook is not used here —
      # the validator works on the per-eval long_allowed records alone).
      records = []
      for h in range(24):
          ts = f"2026-04-01T{h:02d}:00:00+00:00"
          records.append(_rec("BTCUSDT", ts, True))
      result = rally_window_admits_longs(records, min_long_admits=4)
      assert result["passed"] is True
      assert result["details"]["max_admits_in_24h_window"] >= 4


  def test_rally_window_admits_longs_fails_when_no_admits():
      records = [_rec("BTCUSDT", f"2026-04-01T{h:02d}:00:00+00:00", False) for h in range(24)]
      result = rally_window_admits_longs(records, min_long_admits=4)
      assert result["passed"] is False


  def test_monotonic_decline_vetoes_longs_passes():
      records = [_rec("BTCUSDT", f"2026-04-01T{h:02d}:00:00+00:00", False,
                      fast="down", slow="down", strong=True) for h in range(24)]
      result = monotonic_decline_vetoes_longs(records)
      assert result["passed"] is True

  def test_monotonic_decline_vetoes_longs_fails_when_long_slips_in():
      records = [_rec("BTCUSDT", f"2026-04-01T{h:02d}:00:00+00:00", False) for h in range(24)]
      # Inject one offending long_allowed=True
      records[10]["long_allowed"] = True
      result = monotonic_decline_vetoes_longs(records)
      assert result["passed"] is False
  ```

- [ ] **Step 2: Run tests, verify failure**

  Run: `python3 -m pytest bot/backtest/tests/test_validators.py -v`
  Expected: FAIL — `backtest.validators` does not exist.

- [ ] **Step 3: Implement validators**

  Create `bot/backtest/validators.py`:
  ```python
  """Acceptance-criteria validators for pipeline_eval replay outputs.

  Each validator takes a list of per-evaluation records (from a single
  component) and returns:

      {"name": str, "passed": bool, "details": dict}

  Validators are pluggable per component (see PipelineComponent.validators()).
  """
  from __future__ import annotations

  from collections import defaultdict
  from datetime import datetime, timedelta, timezone


  def _parse(ts: str) -> datetime:
      return datetime.fromisoformat(ts)


  def rally_window_admits_longs(records: list, min_long_admits: int = 4) -> dict:
      """Spec §4.3.1: on any 24h window, the new filter must admit ≥4 long entries.

      Implementation: slide a 24h window over all records sorted by timestamp;
      count long_allowed=True within each window; report the max.
      """
      rows = sorted(records, key=lambda r: r["ts"])
      if not rows:
          return {"name": "rally_window_admits_longs", "passed": False,
                  "details": {"reason": "no records"}}
      # Pre-extract timestamps and long_allowed flags as parallel arrays
      ts_arr   = [_parse(r["ts"]) for r in rows]
      flag_arr = [bool(r["long_allowed"]) for r in rows]
      window = timedelta(hours=24)
      best = 0
      best_start = None
      left = 0
      running = 0
      for right in range(len(ts_arr)):
          if flag_arr[right]:
              running += 1
          while ts_arr[right] - ts_arr[left] > window:
              if flag_arr[left]:
                  running -= 1
              left += 1
          if running > best:
              best = running
              best_start = ts_arr[left]
      passed = best >= min_long_admits
      return {
          "name": "rally_window_admits_longs",
          "passed": passed,
          "details": {
              "max_admits_in_24h_window": best,
              "window_start": best_start.isoformat() if best_start else None,
              "min_required": min_long_admits,
          },
      }


  def monotonic_decline_vetoes_longs(records: list) -> dict:
      """Spec §4.3.2: on a monotonic-decline window, longs must remain vetoed.

      Implementation: for each symbol's records, find any 24h window where
      slow_direction='down' for every record AND slow_strong=True for every
      record. In such a window, every long_allowed must be False.
      """
      by_sym: dict = defaultdict(list)
      for r in records:
          by_sym[r["symbol"]].append(r)
      offending: list = []
      for sym, rows in by_sym.items():
          rows.sort(key=lambda r: r["ts"])
          ts_arr   = [_parse(r["ts"]) for r in rows]
          window = timedelta(hours=24)
          left = 0
          run_down = 0
          run_long = 0
          for right in range(len(rows)):
              r = rows[right]
              if r["slow_direction"] == "down" and r["slow_strong"]:
                  run_down += 1
              if r["long_allowed"]:
                  run_long += 1
              while ts_arr[right] - ts_arr[left] > window:
                  lr = rows[left]
                  if lr["slow_direction"] == "down" and lr["slow_strong"]:
                      run_down -= 1
                  if lr["long_allowed"]:
                      run_long -= 1
                  left += 1
              span = right - left + 1
              if span >= 4 and run_down == span and run_long > 0:
                  offending.append({"symbol": sym, "ts": r["ts"],
                                    "long_admits_in_window": run_long})
                  break
      passed = len(offending) == 0
      return {
          "name": "monotonic_decline_vetoes_longs",
          "passed": passed,
          "details": {"offending": offending[:5]},
      }
  ```

- [ ] **Step 4: Register validators on the trend_filter component**

  Modify `bot/backtest/components/trend_filter.py`. Replace the existing `validators` method body:
  ```python
  def validators(self) -> List[Callable]:
      from backtest.validators import (
          rally_window_admits_longs, monotonic_decline_vetoes_longs,
      )
      return [rally_window_admits_longs, monotonic_decline_vetoes_longs]
  ```

- [ ] **Step 5: Run validator + harness tests, verify pass**

  Run: `python3 -m pytest bot/backtest/tests/ -v`
  Expected: previous tests + 4 validator tests = all green.

- [ ] **Step 6: Commit**

  ```bash
  git add bot/backtest/validators.py bot/backtest/tests/test_validators.py \
          bot/backtest/components/trend_filter.py
  git commit -m "$(cat <<'EOF'
  feat(backtest): validators for P3 acceptance criteria

  rally_window_admits_longs — slides a 24h window over all replay records,
  reports the max long-admit count, passes when ≥ min_long_admits (default 4).

  monotonic_decline_vetoes_longs — for any symbol that had a 24h window with
  slow_direction=down AND slow_strong=True for every record in the window,
  asserts no long_allowed=True records appear in that window.

  Wired into TrendFilterReplayComponent.validators() so pipeline_eval writes
  both pass/fail outcomes into its JSON output.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 8: Run the replay — produce P3 evidence

**Why eighth:** The acceptance gate for Wave 2. Output is committed; user inspects before authorising the flip.

**Files:**
- Create: `data/baselines/p3_replay.json` (committed via `-f`)

- [ ] **Step 1: Run the replay**

  From the repo root:
  ```bash
  python3 -m bot.backtest.pipeline_eval --run \
      --parquet data/training \
      --symbols BTCUSDT,ETHUSDT,BNBUSDT,XRPUSDT,SOLUSDT,DOGEUSDT,ADAUSDT,LINKUSDT \
      --tfs 15m,1h --cadence 1h --base-tf 1h \
      --component trend_filter \
      --config config_futures.yaml \
      --out data/baselines/p3_replay.json
  ```
  Expected: process completes (≈ minutes for 8 symbols × 3 years × 1h cadence ≈ 210K evaluations); JSON file written.

- [ ] **Step 2: Inspect the output**

  Run:
  ```bash
  python3 -c "
  import json
  d = json.load(open('data/baselines/p3_replay.json'))
  c = d['coverage']
  s = d['summary']['trend_filter']
  v = d['validators']
  print(f\"coverage: {len(c['symbols'])} symbols, {c['n_evaluations']:,} evals, {c['start']} → {c['end']}\")
  print(f\"summary:  {s}\")
  for name, vr in v.items():
      print(f\"validator {name}: passed={vr['passed']} details={vr['details']}\")
  "
  ```

  GO criteria:
  - `n_evaluations` > 100,000 (proves walk-forward actually ran)
  - `validators.rally_window_admits_longs.passed == True`
  - `validators.monotonic_decline_vetoes_longs.passed == True`

  If any validator fails, STOP. Do not proceed to Task 9. Re-read the failure details, diagnose (filter too lax / too strict / data gap), and iterate.

- [ ] **Step 3: Commit the evidence**

  ```bash
  git add -f data/baselines/p3_replay.json
  git commit -m "$(cat <<'EOF'
  data(baselines): P3 replay evidence — 3yr × 8 coins

  pipeline_eval run with TrendFilterReplayComponent over the 8 training
  coins (BTC/ETH/BNB/XRP/SOL/DOGE/ADA/LINK) at 1h cadence on the
  data/training/<SYMBOL>USDT_{15m,1h}.parquet files. Both spec §4.3
  validators pass:
    - rally_window_admits_longs: admits ≥ 4 long entries in some 24h window
    - monotonic_decline_vetoes_longs: no long_allowed in any strongly-down
      24h window

  Replay output is the gate evidence the user reviews before authorising
  the Wave 2 flag flip.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

- [ ] **Step 4: Show the user the evidence**

  Present a concise summary of the validators + per-symbol admit/veto counts. Ask: *"Replay validators pass. Flip the flag?"*

  **STOP. Do not proceed to Task 9 without an explicit user 'go ahead' / 'flip it' / equivalent.**

---

## Task 9: Wave 2 — flip the flag (gated on user approval)

**Why ninth:** Single behavioural commit. Live bot picks up the new filter on the next scan.

**Files:**
- Modify: `config_futures.yaml`
- Modify: `config_spot.yaml`
- Create: `data/baselines/scan_latency_post_phase2.json`

- [ ] **Step 1: Flip the flag in both configs**

  In `config_futures.yaml` and `config_spot.yaml`, change:
  ```yaml
    use_two_tier: false
  ```
  to:
  ```yaml
    use_two_tier: true
  ```

- [ ] **Step 2: Verify YAML still parses**

  Run:
  ```bash
  python3 -c "
  import yaml
  for f in ('config_futures.yaml', 'config_spot.yaml'):
      d = yaml.safe_load(open(f))['trend_filter']
      assert d['use_two_tier'] is True
      print(f, 'flipped')
  "
  ```
  Expected: both files print `flipped`.

- [ ] **Step 3: Restart the futures bot**

  Run:
  ```bash
  date +"%Y-%m-%d %H:%M:%S local"
  pkill -f 'cryptobot_v3.*launcher.py' 2>/dev/null
  sleep 3
  bash start.sh 2
  sleep 5
  screen -ls | grep -E "cryptobot|monitor|watchdog"
  ```
  Expected: three screens alive; restart timestamp captured.

- [ ] **Step 4: Wait + capture post-flip latency**

  Run (long-running; ~12 minutes):
  ```bash
  sleep 600
  python3 -m tools.profile_scan_loop --log logs/futures_bot.log \
      --marker "HMM regime" --limit 10 \
      > data/baselines/scan_latency_post_phase2.json
  cat data/baselines/scan_latency_post_phase2.json
  ```
  Expected: JSON with `count` ≈ 9–10 (post-restart heartbeats only), `p50` and `p95` values.

  Compute the budget check:
  ```bash
  python3 -c "
  import json
  p1 = json.load(open('data/baselines/scan_latency_post_phase1_recent_only.json'))
  p2 = json.load(open('data/baselines/scan_latency_post_phase2.json'))
  ratio = p2['p95'] / p1['p95']
  print(f'phase1 p95: {p1[\"p95\"]:.2f}s')
  print(f'phase2 p95: {p2[\"p95\"]:.2f}s')
  print(f'ratio: {ratio:.3f}x (budget ≤ 1.50x)')
  print('PASS' if ratio <= 1.5 else 'FAIL')
  "
  ```
  GO criterion: ratio ≤ 1.50.

- [ ] **Step 5: Verify behaviour — at least one long_allowed signal**

  Run:
  ```bash
  python3 -c "
  with open('logs/futures_bot.log') as f:
      log = f.read()
  taken = log.count('TAKEN')
  veto2 = log.count('trend_veto:2tier')
  print(f'TAKEN entries in log: {taken}')
  print(f'2tier veto events:    {veto2}')
  # Check whether any TAKEN entry is a BUY (long)
  longs = sum(1 for line in log.splitlines() if 'TAKEN' in line and ('BUY' in line or 'LONG' in line))
  print(f'long TAKEN entries:   {longs}')
  "
  ```
  GO criterion: either at least one long TAKEN exists, OR the bot has been running long enough that opportunities just haven't appeared yet (record this in the commit message).

  STOP-condition: a `2tier veto storm` — if `veto2 / total_decisions` > 0.50 over the 10-minute window, the filter is too strict; revert the flip.

- [ ] **Step 6: Commit the flip + evidence**

  ```bash
  git add config_futures.yaml config_spot.yaml
  git add -f data/baselines/scan_latency_post_phase2.json
  git commit -m "$(cat <<'EOF'
  fix(config): flip trend_filter.use_two_tier → true (P3 live)

  After replay evidence (commit <task 8 sha>) passed both spec §4.3
  acceptance validators, this commit flips the feature flag in both
  config_futures.yaml and config_spot.yaml. Live bot picks up the new
  two-tier filter on the next config-read.

  Post-flip latency captured: scan_latency_post_phase2.json. Ratio vs
  post-phase1 p95 is within the 1.5x budget.

  Rollback: revert this commit; no code or schema change needed. In-flight
  positions are not force-closed (parent spec §7) and will exit on their
  existing stops / trailing logic.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 10: Phase 2 acceptance + tag

**Why last:** Final structural pass before opening the PR.

- [ ] **Step 1: Full test suites**

  ```bash
  cd bot && python3 -m pytest tests/ -q
  python3 -m pytest backtest/tests/ -q
  cd /root/cryptobot_v3 && python3 -m pytest tools/tests/ -q
  ```
  Expected: all green. Bot suite should be ≥ 158 (was 146 after Phase 1 + 12 new tests across Tasks 2, 4, 5, 6, 7).

- [ ] **Step 2: Structural checks**

  ```bash
  # 1) Three screens alive
  screen -ls | grep -E "cryptobot|monitor|watchdog"
  # 2) Monitor jsonl freshness
  python3 -c "import os, time; age = time.time() - os.stat('logs/monitor_futures.jsonl').st_mtime; print(f'monitor age: {age:.1f}s'); assert age < 300"
  # 3) Flag flipped
  python3 -c "import yaml; d = yaml.safe_load(open('config_futures.yaml'))['trend_filter']; assert d['use_two_tier'] is True; print('flag: on')"
  # 4) Replay evidence committed
  test -f data/baselines/p3_replay.json && echo "replay: present"
  ```
  Expected: 4 lines of confirmation, no failures.

- [ ] **Step 3: Tag the phase**

  ```bash
  git tag -a phase2-trend-filter-complete -m "Signal-quality overhaul Phase 2 (P3 trend filter) complete."
  git log --oneline phase2-trend-filter-complete~9..phase2-trend-filter-complete
  ```
  Expected: tag created at HEAD; log shows the 9 task commits.

- [ ] **Step 4: Push branch + open PR**

  ```bash
  git push -u origin phase2/trend-filter
  echo "PR URL: https://github.com/sarmadkhan17/saarmadproject/compare/main...phase2/trend-filter?expand=1"
  ```

  Build the PR body using the same template structure as the Phase 1 PR body at `/tmp/pr_body.md`:
  - Summary
  - Latency budget (with the phase1 vs phase2 ratio table)
  - Replay evidence summary (validator passes + per-symbol admit/veto counts)
  - Test plan checklist
  - What this PR does NOT do (Phase 3+ scope)

---

## Acceptance summary (Phase 2 done means…)

1. `bot/engine/trend_filter.py` exists and is unit-tested per rule branch.
2. `config_{futures,spot}.yaml` carry the additive two-tier subsections; `use_two_tier: true`.
3. `bot/backtest/pipeline_eval.py` is no longer a stub — it's the plugin harness used by all later phases.
4. `data/baselines/p3_replay.json` is committed and both §4.3 validators pass.
5. `data/baselines/scan_latency_post_phase2.json` shows p95 within 1.5× of Phase 1.
6. Live bot is running on the new filter with the watchdog, monitor, and dashboard screens alive.
7. `phase2-trend-filter-complete` tag exists at HEAD.

## What Phase 2 explicitly does NOT do

- HMM regime layer changes — **Phase 3 (P2).**
- Ensemble agent weights / reconciling the two ensemble paths — **Phase 4 (P4).**
- Calibration + threshold derivation — **Phase 5 (P5).**
- Full shadow infrastructure — **Phase 6 (P6).**
- Bias detector + Telegram alerts — **Phase 7 (P7).**

The 100 %-short bias is expected to begin loosening once Phase 2 ships, but the directional balance settles fully only after Phase 3 (regime) and Phase 4 (ensemble) land. Phase 2's job is to **stop blocking longs structurally**, not to guarantee a 50/50 mix.

---

*End of plan. Awaiting user choice of execution mode.*
