# Signal-Quality Overhaul — Phase 1 (Foundation) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the dependency-free, behaviorally-inert foundation for the signal-quality overhaul: remove the dead DeepSeek tuner, stand up the backtest skeleton + scan-loop profiler, restore monitor observability via a watchdog, and replace ad-hoc cache TTLs with a per-consumer freshness contract that kills the 30-minute decision cache. After Phase 1 the bot's entry-decision logic is unchanged but its data freshness is correct and its observability is restored.

**Architecture:** Foundation work only. We do **not** change the trend filter, regime layer, ensemble weights, or threshold semantics yet — those are Phase 2–5. We do introduce a `bot/data/freshness.py` module that every cache now routes through, a `bot/backtest/` package with two CLI entry-points and a baseline-capture utility, a `tools/watchdog.py` service, and a `tools/profile_scan_loop.py` measurement tool. We persist a one-shot "live baseline" snapshot of pre-overhaul metrics to `data/baselines/pre_overhaul.json` so every future phase has an honest comparator.

**Tech Stack:** Python 3, pandas / polars, pytest, existing `bot/` package, shell `start.sh`, GNU screen.

**Plan position:** This is **Plan 1 of N**. Subsequent plans cover the trend filter (P3), regime layer (P2), agent backtest + ensemble gating (P4), calibration + threshold (P5), shadow + promotion gate (P6 full), bias detector + alerts (P7 full). Each subsequent plan is written **after** the previous phase ships and its gates pass.

**Reference spec:** `docs/superpowers/specs/2026-05-18-signal-quality-overhaul-design.md` (commit `84e13472`).

---

## File map (what this plan creates or modifies)

**Create:**
- `bot/backtest/__init__.py`
- `bot/backtest/baseline.py`
- `bot/backtest/agent_eval.py` (CLI skeleton; pillar P4 fills it later)
- `bot/backtest/pipeline_eval.py` (CLI skeleton; pillar P6 fills it later)
- `bot/backtest/tests/__init__.py`
- `bot/backtest/tests/test_skeleton.py`
- `bot/backtest/tests/test_baseline.py`
- `bot/data/freshness.py`
- `bot/tests/test_freshness.py`
- `bot/tests/test_coordinator_no_decision_cache.py`
- `tools/watchdog.py`
- `tools/profile_scan_loop.py`
- `tools/tests/__init__.py`
- `tools/tests/test_watchdog.py`
- `data/baselines/.gitkeep`

**Modify:**
- `bot/data/feed.py` — route `OHLCVCache` through `Freshness`; remove blanket per-tf TTL fallback
- `bot/models/hmm.py` — remove `_CACHE_TTL = 300` constant, route regime cache through `Freshness` (60 s)
- `bot/agents/coordinator.py` — **delete** the 30-min `_decision_cache` branch; reduce `SLOW_CACHE_SECS` 7200 → 1800; add `staleness_seconds` field to `_fg_cache` / `_macro_cache`; macro stale-decay weight in `MasterAgent`
- `start.sh` — launch watchdog in its own screen session
- `bot/tests/test_macro_flow_agent.py` if it asserts macro weight (verify; only modify if it would break)

**Delete:**
- `deepseek_admin.py`
- `deepseek_actions.log`
- `deepseek_actions_futures.log`
- `deepseek_actions_spot.log`

---

## Conventions used in this plan

- **Test pattern.** All tests sit in `bot/tests/` or `<package>/tests/` and start with the standard prelude already in this repo:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
  ```
- **Atomic writes.** All state writes go to `*.tmp.json` then `os.replace()` (project rule, CLAUDE.md).
- **Commits.** One commit per task (test + implementation + cleanup land together); commit message body always ends with the `Co-Authored-By` trailer.
- **Behavior gate.** Phase 1 must not change which entries the bot takes. Smoke test (Task 10) verifies the bot still starts and the scan loop runs.

---

## Task 1: Remove the stopped DeepSeek tuner

**Why first:** Smallest blast radius, no behavioral risk (already stopped), gets dead code out of the tree before we depend on its absence.

**Files:**
- Delete: `deepseek_admin.py`
- Delete: `deepseek_actions.log`
- Delete: `deepseek_actions_futures.log`
- Delete: `deepseek_actions_spot.log`

- [ ] **Step 1: Verify no live references**

  Run:
  ```bash
  grep -rn "deepseek\|DeepSeek\|DEEPSEEK" bot/ dashboard/ tools/ 2>/dev/null
  ps aux | grep -i deepseek | grep -v grep
  crontab -l 2>/dev/null | grep -i deepseek
  ```
  Expected output: empty (no references in code, no live process, no cron entry). If any reference appears, stop and update this plan with the additional cleanup.

- [ ] **Step 2: Delete the script and logs**

  ```bash
  git rm deepseek_admin.py deepseek_actions.log deepseek_actions_futures.log deepseek_actions_spot.log
  ```

- [ ] **Step 3: Verify the bot still imports**

  Run:
  ```bash
  cd bot && python -c "from engine.bot import BaseBot; print('ok')"
  ```
  Expected output: `ok`.

- [ ] **Step 4: Commit**

  ```bash
  git commit -m "$(cat <<'EOF'
  chore: remove stopped DeepSeek auto-tuner

  DeepSeek script and action logs deleted. The tuner has been stopped since
  2026-05-17 and no bot/ code references it. Removing it as the first step of
  the signal-quality overhaul so subsequent threshold-governance changes have
  no dead actor to coexist with.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 2: Baseline-capture utility

**Why second:** Every future phase compares against "live baseline metrics on the same OOS window". We capture that baseline once, now, from real state and the parquet, before any change can pollute it.

**Files:**
- Create: `data/baselines/.gitkeep`
- Create: `bot/backtest/__init__.py`
- Create: `bot/backtest/baseline.py`
- Create: `bot/backtest/tests/__init__.py`
- Create: `bot/backtest/tests/test_baseline.py`

- [ ] **Step 1: Create empty package + gitkeep**

  ```bash
  mkdir -p bot/backtest/tests data/baselines
  touch bot/backtest/__init__.py bot/backtest/tests/__init__.py data/baselines/.gitkeep
  ```

- [ ] **Step 2: Write the failing baseline test**

  Create `bot/backtest/tests/test_baseline.py`:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

  import json
  from pathlib import Path
  from backtest.baseline import capture_baseline


  def test_capture_baseline_emits_required_metrics(tmp_path):
      state = {
          "trades": [
              {"status": "closed", "side": "short", "pnl":  1.5,
               "close_timestamp": "2026-05-18T10:00:00+00:00", "duration_hours": 1.0},
              {"status": "closed", "side": "short", "pnl": -1.0,
               "close_timestamp": "2026-05-18T11:00:00+00:00", "duration_hours": 2.0},
              {"status": "closed", "side": "long",  "pnl":  2.0,
               "close_timestamp": "2026-05-18T12:00:00+00:00", "duration_hours": 0.5},
              {"status": "open",   "side": "short", "live_pnl": -0.3},
          ],
          "stats": {"balance": 3422.59, "total_pnl": -197.47, "wins": 1, "losses": 1},
      }
      state_path = tmp_path / "state.json"
      state_path.write_text(json.dumps(state))
      out = tmp_path / "baseline.json"

      capture_baseline(state_path, out)

      data = json.loads(out.read_text())
      assert data["closed_count"] == 3
      assert data["wins"] == 2
      assert data["losses"] == 1
      assert abs(data["win_rate"] - (2/3)) < 1e-6
      assert abs(data["net_pnl"] - 2.5) < 1e-6
      assert data["side_mix"] == {"short": 2, "long": 1}
      assert "captured_at" in data
      assert data["source_state"] == str(state_path)
  ```

- [ ] **Step 3: Run test, verify failure**

  Run: `cd bot && python -m pytest backtest/tests/test_baseline.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'backtest.baseline'`.

- [ ] **Step 4: Implement `baseline.py`**

  Create `bot/backtest/baseline.py`:
  ```python
  """Capture pre-overhaul live baseline metrics from a state file.

  Used once at the start of the signal-quality overhaul so every later phase
  can compare against an honest 'this is what the bot was doing before'
  snapshot. Reads a state.json / futures_state.json shape and writes a small
  JSON summary.
  """
  from __future__ import annotations

  import json
  import os
  from collections import Counter
  from datetime import datetime, timezone
  from pathlib import Path
  from typing import Union


  def capture_baseline(state_path: Union[str, Path], out_path: Union[str, Path]) -> dict:
      state_path = Path(state_path)
      out_path = Path(out_path)
      with open(state_path) as fh:
          state = json.load(fh)

      trades = state.get("trades", [])
      closed = [t for t in trades if t.get("status") == "closed"]
      wins = [t for t in closed if (t.get("pnl") or 0) > 0]
      losses = [t for t in closed if (t.get("pnl") or 0) <= 0]
      side_mix = Counter(t.get("side", "?") for t in closed)
      durations = [float(t.get("duration_hours", 0)) for t in closed]

      summary = {
          "captured_at": datetime.now(timezone.utc).isoformat(),
          "source_state": str(state_path),
          "closed_count": len(closed),
          "wins": len(wins),
          "losses": len(losses),
          "win_rate": (len(wins) / len(closed)) if closed else 0.0,
          "net_pnl": sum((t.get("pnl") or 0) for t in closed),
          "avg_duration_hours": (sum(durations) / len(durations)) if durations else 0.0,
          "side_mix": dict(side_mix),
          "stats_balance": state.get("stats", {}).get("balance"),
          "stats_total_pnl": state.get("stats", {}).get("total_pnl"),
      }

      tmp = out_path.with_suffix(out_path.suffix + ".tmp")
      tmp.parent.mkdir(parents=True, exist_ok=True)
      with open(tmp, "w") as fh:
          json.dump(summary, fh, indent=2)
      os.replace(tmp, out_path)
      return summary


  def _main(argv: list[str]) -> int:
      import argparse
      parser = argparse.ArgumentParser(description="Capture pre-overhaul baseline metrics.")
      parser.add_argument("--state", required=True, help="Path to state.json or futures_state.json")
      parser.add_argument("--out",   required=True, help="Path to write baseline summary JSON")
      args = parser.parse_args(argv)
      summary = capture_baseline(args.state, args.out)
      print(json.dumps(summary, indent=2))
      return 0


  if __name__ == "__main__":
      import sys as _sys
      raise SystemExit(_main(_sys.argv[1:]))
  ```

- [ ] **Step 5: Run test, verify pass**

  Run: `cd bot && python -m pytest backtest/tests/test_baseline.py -v`
  Expected: 1 passed.

- [ ] **Step 6: Capture the actual live baseline**

  Run (from repo root):
  ```bash
  python -m bot.backtest.baseline --state data/futures_state.json --out data/baselines/pre_overhaul_futures.json
  ```
  Expected output: a JSON summary printed; file present at `data/baselines/pre_overhaul_futures.json`.

  Verify it picked up the 100%-short bias we saw in the audit:
  ```bash
  python3 -c "import json; d=json.load(open('data/baselines/pre_overhaul_futures.json')); print(d['side_mix'], d['win_rate'], d['net_pnl'])"
  ```
  Expected: a dict dominated by `short`, `win_rate` around 0.42–0.49, `net_pnl` negative.

- [ ] **Step 7: Commit**

  ```bash
  git add bot/backtest/__init__.py bot/backtest/tests/__init__.py bot/backtest/baseline.py \
          bot/backtest/tests/test_baseline.py data/baselines/.gitkeep data/baselines/pre_overhaul_futures.json
  git commit -m "$(cat <<'EOF'
  feat(backtest): live-baseline capture utility

  bot/backtest/baseline.py captures pre-overhaul metrics from a state file
  (closed_count, win_rate, net_pnl, side_mix, avg_duration). Used as the
  honest comparator for every subsequent phase of the signal-quality overhaul.
  Live snapshot of futures state captured to data/baselines/pre_overhaul_futures.json.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 3: Backtester skeleton — `agent_eval` + `pipeline_eval` CLIs

**Why third:** Phase 2+ will fill these in with the real logic. We land the *skeleton* now (CLI shape, exit codes, output schema, "not yet implemented" stub) so the file structure and entry points are stable and discoverable.

**Files:**
- Create: `bot/backtest/agent_eval.py`
- Create: `bot/backtest/pipeline_eval.py`
- Create: `bot/backtest/tests/test_skeleton.py`

- [ ] **Step 1: Write the failing skeleton test**

  Create `bot/backtest/tests/test_skeleton.py`:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

  import subprocess
  from pathlib import Path

  ROOT = Path(__file__).resolve().parents[3]


  def _run(module: str, *args: str) -> subprocess.CompletedProcess:
      return subprocess.run(
          ["python", "-m", module, *args],
          cwd=ROOT,
          capture_output=True,
          text=True,
      )


  def test_agent_eval_cli_help():
      result = _run("bot.backtest.agent_eval", "--help")
      assert result.returncode == 0
      assert "agent_eval" in result.stdout.lower() or "agent" in result.stdout.lower()


  def test_agent_eval_unimplemented_returns_nonzero_with_clear_message():
      result = _run("bot.backtest.agent_eval", "--run")
      assert result.returncode != 0
      assert "not implemented" in result.stderr.lower() or "not implemented" in result.stdout.lower()


  def test_pipeline_eval_cli_help():
      result = _run("bot.backtest.pipeline_eval", "--help")
      assert result.returncode == 0


  def test_pipeline_eval_unimplemented_returns_nonzero_with_clear_message():
      result = _run("bot.backtest.pipeline_eval", "--run")
      assert result.returncode != 0
      assert "not implemented" in result.stderr.lower() or "not implemented" in result.stdout.lower()
  ```

- [ ] **Step 2: Run test, verify failure**

  Run: `python -m pytest bot/backtest/tests/test_skeleton.py -v`
  Expected: FAIL with `No module named 'bot.backtest.agent_eval'`.

- [ ] **Step 3: Implement `agent_eval.py` skeleton**

  Create `bot/backtest/agent_eval.py`:
  ```python
  """Per-agent walk-forward backtest harness (skeleton).

  Filled in during Phase 4 (per the signal-quality overhaul spec). Phase 1
  ships the CLI shape only so dependent tooling can wire against it.
  """
  from __future__ import annotations

  import argparse
  import sys


  def build_parser() -> argparse.ArgumentParser:
      p = argparse.ArgumentParser(prog="agent_eval", description="Per-agent backtest harness.")
      p.add_argument("--run", action="store_true", help="Execute the backtest.")
      p.add_argument("--parquet", help="Path to training_dataset.parquet")
      p.add_argument("--agent",   help="Agent name (smc | technical | macro | all)")
      p.add_argument("--out",     help="Output JSON path")
      return p


  def main(argv: list[str]) -> int:
      args = build_parser().parse_args(argv)
      if args.run:
          print("agent_eval: not implemented — scheduled for Phase 4 of the signal-quality overhaul",
                file=sys.stderr)
          return 2
      return 0


  if __name__ == "__main__":
      raise SystemExit(main(sys.argv[1:]))
  ```

- [ ] **Step 4: Implement `pipeline_eval.py` skeleton**

  Create `bot/backtest/pipeline_eval.py`:
  ```python
  """End-to-end pipeline backtest harness (skeleton).

  Filled in during Phase 6 (per the signal-quality overhaul spec). Phase 1
  ships the CLI shape only.
  """
  from __future__ import annotations

  import argparse
  import sys


  def build_parser() -> argparse.ArgumentParser:
      p = argparse.ArgumentParser(prog="pipeline_eval", description="End-to-end pipeline backtest.")
      p.add_argument("--run", action="store_true", help="Execute the backtest.")
      p.add_argument("--parquet", help="Path to training_dataset.parquet")
      p.add_argument("--config",  help="Path to config_<mode>.yaml")
      p.add_argument("--out",     help="Output JSON path")
      return p


  def main(argv: list[str]) -> int:
      args = build_parser().parse_args(argv)
      if args.run:
          print("pipeline_eval: not implemented — scheduled for Phase 6 of the signal-quality overhaul",
                file=sys.stderr)
          return 2
      return 0


  if __name__ == "__main__":
      raise SystemExit(main(sys.argv[1:]))
  ```

- [ ] **Step 5: Run tests, verify pass**

  Run: `python -m pytest bot/backtest/tests/test_skeleton.py -v`
  Expected: 4 passed.

- [ ] **Step 6: Commit**

  ```bash
  git add bot/backtest/agent_eval.py bot/backtest/pipeline_eval.py bot/backtest/tests/test_skeleton.py
  git commit -m "$(cat <<'EOF'
  feat(backtest): agent_eval + pipeline_eval CLI skeletons

  Two backtest entry points stubbed with deterministic CLI shape and
  'not implemented' exit code 2. Filled in during Phases 4 and 6 of the
  signal-quality overhaul. Tests assert the CLIs parse --help, refuse to
  silently no-op on --run, and surface a clear message about where the
  real implementation lives.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 4: Scan-loop profiler

**Why fourth:** Phase 1's freshness changes will increase per-scan work (no more 30-min cache hits). We need a profiler in tree *before* that change so we can measure pre/post latency and enforce the §6 P6d budget.

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/profile_scan_loop.py`
- Create: `tools/tests/__init__.py`
- Create: `tools/tests/test_profile_scan_loop.py`

- [ ] **Step 1: Create tools package**

  ```bash
  mkdir -p tools/tests
  touch tools/__init__.py tools/tests/__init__.py
  ```

- [ ] **Step 2: Write the failing profiler test**

  Create `tools/tests/test_profile_scan_loop.py`:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

  from tools.profile_scan_loop import summarise_durations


  def test_summarise_durations_basic():
      summary = summarise_durations([0.1, 0.2, 0.3, 0.4, 0.5])
      assert abs(summary["p50"] - 0.3) < 1e-6
      # p95 of 5 samples is the largest value
      assert abs(summary["p95"] - 0.5) < 1e-6
      assert abs(summary["min"] - 0.1) < 1e-6
      assert abs(summary["max"] - 0.5) < 1e-6
      assert summary["count"] == 5


  def test_summarise_durations_empty():
      summary = summarise_durations([])
      assert summary["count"] == 0
      assert summary["p50"] is None
      assert summary["p95"] is None
  ```

- [ ] **Step 3: Run test, verify failure**

  Run: `python -m pytest tools/tests/test_profile_scan_loop.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'tools.profile_scan_loop'`.

- [ ] **Step 4: Implement the profiler**

  Create `tools/profile_scan_loop.py`:
  ```python
  """Measure scan-loop p50 / p95 / max latency by tailing logs/<mode>_bot.log.

  Reads the existing 'HMM regime: ...' heartbeat lines (one per scan cycle)
  and computes the inter-arrival time. That is the scan-loop period and a
  good proxy for total per-cycle cost. Writes a JSON summary to stdout.

  Used to enforce the latency budget gate (spec §6 P6d).
  """
  from __future__ import annotations

  import argparse
  import json
  import re
  import statistics
  import sys
  from datetime import datetime
  from pathlib import Path
  from typing import Optional


  TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d{3})")


  def summarise_durations(durations: list[float]) -> dict:
      if not durations:
          return {"count": 0, "p50": None, "p95": None, "min": None, "max": None,
                  "mean": None}
      sorted_d = sorted(durations)
      def _pct(p: float) -> float:
          if len(sorted_d) == 1:
              return sorted_d[0]
          rank = int(round((p / 100.0) * (len(sorted_d) - 1)))
          return sorted_d[rank]
      return {
          "count": len(sorted_d),
          "p50": _pct(50),
          "p95": _pct(95),
          "min": sorted_d[0],
          "max": sorted_d[-1],
          "mean": statistics.fmean(sorted_d),
      }


  def _parse_ts(line: str) -> Optional[datetime]:
      m = TS_RE.match(line)
      if not m:
          return None
      try:
          return datetime.strptime(f"{m.group(1)}.{m.group(2)}", "%Y-%m-%d %H:%M:%S.%f")
      except ValueError:
          return None


  def measure_log(log_path: Path, marker: str = "HMM regime", limit: int = 500) -> dict:
      timestamps: list[datetime] = []
      with open(log_path) as fh:
          for line in fh:
              if marker not in line:
                  continue
              ts = _parse_ts(line)
              if ts is not None:
                  timestamps.append(ts)
                  if len(timestamps) > limit:
                      timestamps.pop(0)
      durations = [
          (timestamps[i] - timestamps[i - 1]).total_seconds()
          for i in range(1, len(timestamps))
      ]
      return summarise_durations(durations)


  def main(argv: list[str]) -> int:
      p = argparse.ArgumentParser(prog="profile_scan_loop",
                                  description="Measure scan-loop latency from bot log.")
      p.add_argument("--log", required=True, help="Path to <mode>_bot.log")
      p.add_argument("--marker", default="HMM regime",
                     help="Per-scan marker substring (default: 'HMM regime')")
      p.add_argument("--limit", type=int, default=500,
                     help="How many recent markers to use")
      args = p.parse_args(argv)
      summary = measure_log(Path(args.log), args.marker, args.limit)
      print(json.dumps(summary, indent=2))
      return 0


  if __name__ == "__main__":
      raise SystemExit(main(sys.argv[1:]))
  ```

- [ ] **Step 5: Run test, verify pass**

  Run: `python -m pytest tools/tests/test_profile_scan_loop.py -v`
  Expected: 2 passed.

- [ ] **Step 6: Capture pre-Phase-1 latency baseline**

  Run (from repo root):
  ```bash
  python -m tools.profile_scan_loop --log logs/futures_bot.log --marker "HMM regime" --limit 500 \
      > data/baselines/scan_latency_pre_phase1.json
  cat data/baselines/scan_latency_pre_phase1.json
  ```
  Expected: a JSON with `count`, `p50`, `p95`. Record these numbers; Phase 1's freshness changes must not push p95 above `1.5 × p95_pre`.

- [ ] **Step 7: Commit**

  ```bash
  git add tools/__init__.py tools/tests/__init__.py tools/profile_scan_loop.py tools/tests/test_profile_scan_loop.py \
          data/baselines/scan_latency_pre_phase1.json
  git commit -m "$(cat <<'EOF'
  feat(tools): scan-loop latency profiler

  Reads HMM heartbeat timestamps from <mode>_bot.log and emits p50/p95/max
  inter-arrival times. Pre-Phase-1 latency captured to
  data/baselines/scan_latency_pre_phase1.json so the freshness contract
  rollout can be gated on a measured budget (spec §6 P6d).

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 5: Monitor + bot watchdog

**Why fifth:** We will not touch live behavior again until we can detect when our observability dies. The audit found `monitor_trades.py` had been dead for ~10 hours with no alarm.

**Files:**
- Create: `tools/watchdog.py`
- Create: `tools/tests/test_watchdog.py`
- Modify: `start.sh`

- [ ] **Step 1: Write the failing watchdog tests**

  Create `tools/tests/test_watchdog.py`:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

  import json
  import time
  from pathlib import Path
  from tools.watchdog import is_stale, check_targets


  def test_is_stale_recent(tmp_path: Path):
      p = tmp_path / "fresh.jsonl"
      p.write_text("line\n")
      assert is_stale(p, max_age_seconds=300) is False


  def test_is_stale_old(tmp_path: Path):
      p = tmp_path / "old.jsonl"
      p.write_text("line\n")
      old = time.time() - 3600
      os.utime(p, (old, old))
      assert is_stale(p, max_age_seconds=300) is True


  def test_is_stale_missing(tmp_path: Path):
      assert is_stale(tmp_path / "nope.jsonl", max_age_seconds=300) is True


  def test_check_targets_reports_each(tmp_path: Path):
      fresh = tmp_path / "fresh.jsonl"; fresh.write_text("x")
      stale = tmp_path / "stale.jsonl"; stale.write_text("x")
      os.utime(stale, (time.time() - 3600, time.time() - 3600))
      report = check_targets([
          {"name": "fresh", "path": fresh, "max_age_seconds": 300},
          {"name": "stale", "path": stale, "max_age_seconds": 300},
      ])
      assert report["fresh"]["stale"] is False
      assert report["stale"]["stale"] is True
  ```

- [ ] **Step 2: Run tests, verify failure**

  Run: `python -m pytest tools/tests/test_watchdog.py -v`
  Expected: FAIL with `No module named 'tools.watchdog'`.

- [ ] **Step 3: Implement the watchdog**

  Create `tools/watchdog.py`:
  ```python
  """Restart the monitor / alert on bot heartbeat staleness.

  Run as a long-lived process (started by start.sh in its own screen). Every
  60 s it checks two staleness gates:

  1. logs/monitor_<mode>.jsonl — if older than 5 minutes, restart monitor.
  2. data/bot_heartbeat_<mode>.json — if older than 5 minutes, log a fatal
     and write a restart-request to data/watchdog_alerts.jsonl. (We do NOT
     auto-restart the bot itself in Phase 1 — that's a Phase 7 concern. We
     surface it so the user knows.)
  """
  from __future__ import annotations

  import argparse
  import json
  import os
  import subprocess
  import sys
  import time
  from datetime import datetime, timezone
  from pathlib import Path


  ROOT = Path(__file__).resolve().parents[1]
  ALERTS_PATH = ROOT / "data" / "watchdog_alerts.jsonl"
  LOG_PATH    = ROOT / "logs" / "watchdog.log"


  def is_stale(path: Path, max_age_seconds: float) -> bool:
      if not path.exists():
          return True
      age = time.time() - path.stat().st_mtime
      return age > max_age_seconds


  def check_targets(targets: list[dict]) -> dict:
      out = {}
      now = time.time()
      for t in targets:
          path = Path(t["path"])
          stale = is_stale(path, t["max_age_seconds"])
          age = (now - path.stat().st_mtime) if path.exists() else None
          out[t["name"]] = {"stale": stale, "age_seconds": age}
      return out


  def _log(msg: str) -> None:
      LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
      ts = datetime.now(timezone.utc).isoformat()
      with open(LOG_PATH, "a") as fh:
          fh.write(f"[{ts}] {msg}\n")


  def _alert(event: str, **kwargs) -> None:
      ALERTS_PATH.parent.mkdir(parents=True, exist_ok=True)
      record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event, **kwargs}
      with open(ALERTS_PATH, "a") as fh:
          fh.write(json.dumps(record) + "\n")


  def _restart_monitor(mode: str) -> None:
      _log(f"restarting monitor for mode={mode}")
      subprocess.Popen(
          ["screen", "-dmS", f"monitor_{mode}",
           "bash", "-c", f"cd {ROOT} && BOT_MODE={mode} python3 monitor_trades.py"],
          cwd=str(ROOT),
      )


  def run(mode: str, interval_seconds: int = 60) -> None:
      monitor_jsonl = ROOT / "logs" / f"monitor_{mode}.jsonl"
      bot_heartbeat = ROOT / "data" / f"bot_heartbeat_{mode}.json"
      _log(f"watchdog started mode={mode} interval={interval_seconds}s")
      while True:
          report = check_targets([
              {"name": "monitor",      "path": monitor_jsonl, "max_age_seconds": 300},
              {"name": "bot_heartbeat","path": bot_heartbeat, "max_age_seconds": 300},
          ])
          if report["monitor"]["stale"]:
              _log(f"monitor stale (age={report['monitor']['age_seconds']}s); restarting")
              _alert("monitor_restart", age_seconds=report["monitor"]["age_seconds"])
              _restart_monitor(mode)
          if report["bot_heartbeat"]["stale"]:
              _log(f"bot heartbeat stale (age={report['bot_heartbeat']['age_seconds']}s)")
              _alert("bot_heartbeat_stale", age_seconds=report["bot_heartbeat"]["age_seconds"])
          time.sleep(interval_seconds)


  def main(argv: list[str]) -> int:
      p = argparse.ArgumentParser(prog="watchdog", description="Monitor + bot heartbeat watchdog.")
      p.add_argument("--mode", default=os.environ.get("BOT_MODE", "futures"),
                     choices=["spot", "futures"])
      p.add_argument("--interval", type=int, default=60)
      args = p.parse_args(argv)
      run(args.mode, args.interval)
      return 0


  if __name__ == "__main__":
      raise SystemExit(main(sys.argv[1:]))
  ```

- [ ] **Step 4: Run tests, verify pass**

  Run: `python -m pytest tools/tests/test_watchdog.py -v`
  Expected: 4 passed.

- [ ] **Step 5: Wire watchdog into `start.sh`**

  Modify `start.sh`, after the bot screen launch (right after `sleep 2` near the bottom), add:
  ```bash
  # Start watchdog in its own screen
  WATCHDOG_NAME="watchdog_$MODE"
  pkill -f "tools.watchdog.*--mode $MODE" 2>/dev/null || true
  screen -dmS "$WATCHDOG_NAME" bash -c "cd $BOT_DIR && python3 -m tools.watchdog --mode $MODE --interval 60"
  ```

  Verify the resulting `start.sh` still parses:
  ```bash
  bash -n start.sh && echo "ok"
  ```
  Expected output: `ok`.

- [ ] **Step 6: Smoke test the watchdog**

  Run (in a separate terminal):
  ```bash
  python -m tools.watchdog --mode futures --interval 5 &
  WDPID=$!
  sleep 12
  kill $WDPID
  tail -5 logs/watchdog.log
  ```
  Expected: at least one `watchdog started ...` line and check entries; if the monitor was dead at audit time, expect a `monitor stale` / `monitor_restart` entry too.

- [ ] **Step 7: Commit**

  ```bash
  git add tools/watchdog.py tools/tests/test_watchdog.py start.sh
  git commit -m "$(cat <<'EOF'
  feat(tools): monitor + bot heartbeat watchdog

  tools/watchdog.py polls every 60 s for monitor jsonl freshness and bot
  heartbeat freshness. Monitor is auto-restarted; bot heartbeat staleness
  is alerted (auto-restart for the bot is Phase 7). start.sh launches the
  watchdog in its own screen alongside the bot.

  Addresses the audit finding that monitor_trades.py had been dead for
  ~10h with no alarm (spec §1 R9).

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 6: Freshness contract module

**Why sixth:** All cache modifications in Tasks 7–9 route through this module. We build the contract once, with tests, then refactor consumers onto it.

**Files:**
- Create: `bot/data/freshness.py`
- Create: `bot/tests/test_freshness.py`

- [ ] **Step 1: Write the failing freshness tests**

  Create `bot/tests/test_freshness.py`:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

  import time
  import pytest
  from data.freshness import Freshness, CacheMiss


  def test_set_then_fresh_get_returns_value():
      f = Freshness()
      f.set("k", "v")
      assert f.get("k", max_age_seconds=10) == "v"


  def test_get_returns_none_when_unset():
      f = Freshness()
      assert f.get("k", max_age_seconds=10) is None


  def test_get_returns_none_when_stale():
      f = Freshness()
      f.set("k", "v")
      # Backdate the timestamp deterministically.
      key_meta = f._times["k"]
      f._times["k"] = key_meta - 100
      assert f.get("k", max_age_seconds=10) is None


  def test_age_seconds_reports_correctly():
      f = Freshness()
      f.set("k", "v")
      f._times["k"] = f._times["k"] - 7
      assert 6 <= f.age_seconds("k") <= 8


  def test_age_seconds_returns_none_for_missing_key():
      f = Freshness()
      assert f.age_seconds("nope") is None


  def test_fetch_calls_loader_on_miss():
      f = Freshness()
      calls = {"n": 0}
      def loader():
          calls["n"] += 1
          return "loaded"
      v = f.fetch("k", max_age_seconds=10, loader=loader)
      assert v == "loaded"
      assert calls["n"] == 1
      # second call within ttl uses cache
      v2 = f.fetch("k", max_age_seconds=10, loader=loader)
      assert v2 == "loaded"
      assert calls["n"] == 1


  def test_fetch_recalls_loader_when_stale():
      f = Freshness()
      calls = {"n": 0}
      def loader():
          calls["n"] += 1
          return calls["n"]
      f.fetch("k", max_age_seconds=10, loader=loader)
      f._times["k"] -= 100
      v = f.fetch("k", max_age_seconds=10, loader=loader)
      assert v == 2
      assert calls["n"] == 2


  def test_fetch_propagates_loader_exception():
      f = Freshness()
      def loader():
          raise CacheMiss("upstream failure")
      with pytest.raises(CacheMiss):
          f.fetch("k", max_age_seconds=10, loader=loader)


  def test_invalidate_removes_key():
      f = Freshness()
      f.set("k", "v")
      f.invalidate("k")
      assert f.get("k", max_age_seconds=10) is None
  ```

- [ ] **Step 2: Run tests, verify failure**

  Run: `cd bot && python -m pytest tests/test_freshness.py -v`
  Expected: FAIL with `ModuleNotFoundError: No module named 'data.freshness'`.

- [ ] **Step 3: Implement `freshness.py`**

  Create `bot/data/freshness.py`:
  ```python
  """Per-consumer freshness contract for caches.

  Replaces the scattered TTL constants (HMM._CACHE_TTL, OHLCV CANDLE_SECONDS,
  AgentCoordinator.GROQ_CACHE_SECS/SLOW_CACHE_SECS) with one mechanism whose
  rule is simple: every caller declares max_age_seconds for the value it's
  about to read. A miss returns None; the caller decides what to do — no
  silent staleness.

  Used by:
    bot/data/feed.py (OHLCV)
    bot/models/hmm.py (regime label)
    bot/agents/coordinator.py (Fear&Greed + Macro snapshots)
  """
  from __future__ import annotations

  import threading
  import time
  from typing import Any, Callable, Dict, Optional


  class CacheMiss(Exception):
      """Raised by a loader when an upstream source cannot provide data."""


  class Freshness:
      def __init__(self) -> None:
          self._lock: threading.Lock = threading.Lock()
          self._values: Dict[str, Any] = {}
          self._times:  Dict[str, float] = {}

      def set(self, key: str, value: Any) -> None:
          with self._lock:
              self._values[key] = value
              self._times[key]  = time.time()

      def get(self, key: str, max_age_seconds: float) -> Optional[Any]:
          with self._lock:
              t = self._times.get(key)
              if t is None:
                  return None
              if (time.time() - t) > max_age_seconds:
                  return None
              return self._values.get(key)

      def age_seconds(self, key: str) -> Optional[float]:
          with self._lock:
              t = self._times.get(key)
              if t is None:
                  return None
              return time.time() - t

      def invalidate(self, key: str) -> None:
          with self._lock:
              self._values.pop(key, None)
              self._times.pop(key, None)

      def fetch(self, key: str, max_age_seconds: float,
                loader: Callable[[], Any]) -> Any:
          """Get a fresh value or call loader() and cache its result.

          The loader's exception (including CacheMiss) propagates to the caller.
          """
          cached = self.get(key, max_age_seconds)
          if cached is not None:
              return cached
          value = loader()
          self.set(key, value)
          return value
  ```

- [ ] **Step 4: Run tests, verify pass**

  Run: `cd bot && python -m pytest tests/test_freshness.py -v`
  Expected: 9 passed.

- [ ] **Step 5: Commit**

  ```bash
  git add bot/data/freshness.py bot/tests/test_freshness.py
  git commit -m "$(cat <<'EOF'
  feat(data): Freshness module — per-consumer cache contract

  bot/data/freshness.py centralises cache freshness. Callers declare
  max_age_seconds per read; misses return None (no silent staleness).
  Replaces the scattered TTL constants in subsequent tasks.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 7: Route OHLCVCache through Freshness

**Why seventh:** OHLCV is the first consumer. Its existing `CANDLE_SECONDS` per-tf TTL becomes a per-call `max_age_seconds` argument; behaviour is identical at the default tier but callers can demand fresher data.

**Files:**
- Modify: `bot/data/feed.py:113-149` (the `OHLCVCache` class)
- Modify: `bot/data/feed.py:285` (the `needs_refresh` call site)
- Create: `bot/tests/test_ohlcv_cache_freshness.py`

- [ ] **Step 1: Read the existing OHLCVCache implementation**

  Open `bot/data/feed.py` and re-read lines 113-149 plus the call site near 285 to confirm the contract before changing it.

- [ ] **Step 2: Write the failing freshness test for OHLCVCache**

  Create `bot/tests/test_ohlcv_cache_freshness.py`:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

  import time
  from data.feed import OHLCVCache


  def test_needs_refresh_when_unset():
      c = OHLCVCache()
      assert c.needs_refresh("BTC/USDT", "1h") is True


  def test_needs_refresh_false_when_fresh_under_max_age():
      c = OHLCVCache()
      c.set("BTC/USDT", "1h", "df-marker")
      assert c.needs_refresh("BTC/USDT", "1h", max_age_seconds=300) is False


  def test_needs_refresh_true_when_older_than_max_age():
      c = OHLCVCache()
      c.set("BTC/USDT", "1h", "df-marker")
      key = c._key("BTC/USDT", "1h")
      c._fresh._times[key] -= 500  # backdate via underlying freshness store
      assert c.needs_refresh("BTC/USDT", "1h", max_age_seconds=300) is True


  def test_get_returns_value_within_default_tier_ttl():
      c = OHLCVCache()
      c.set("BTC/USDT", "1h", "df-marker")
      # Default per-tf ceiling still applies when max_age not given.
      assert c.get("BTC/USDT", "1h") == "df-marker"


  def test_get_returns_none_when_caller_demands_fresher_than_we_have():
      c = OHLCVCache()
      c.set("BTC/USDT", "1h", "df-marker")
      key = c._key("BTC/USDT", "1h")
      c._fresh._times[key] -= 120
      # We have 120s-old data; caller demands < 60s.
      assert c.get("BTC/USDT", "1h", max_age_seconds=60) is None
  ```

- [ ] **Step 3: Run test, verify failure**

  Run: `cd bot && python -m pytest tests/test_ohlcv_cache_freshness.py -v`
  Expected: FAIL — current `OHLCVCache.needs_refresh` does not accept `max_age_seconds`, and `get` does not accept it either.

- [ ] **Step 4: Refactor `OHLCVCache` onto `Freshness`**

  Replace lines 113-149 of `bot/data/feed.py` with:
  ```python
  # ── OHLCV Cache ───────────────────────────────────────────────────────────────

  from .freshness import Freshness


  class OHLCVCache:
      """Per-timeframe OHLCV cache backed by the shared Freshness contract.

      Each tf has a default ceiling TTL (forming-candle refresh cadence). Callers
      may pass a tighter ``max_age_seconds`` to demand fresher data.
      """
      CANDLE_SECONDS = {
          "1m": 30, "5m": 60, "15m": 180,
          "1h": 300, "4h": 900, "1d": 1800,
      }

      def __init__(self) -> None:
          self._fresh = Freshness()

      def _key(self, symbol: str, tf: str) -> str:
          return f"{symbol}_{tf}"

      def _ceiling(self, tf: str) -> int:
          return self.CANDLE_SECONDS.get(tf, 3600)

      def needs_refresh(self, symbol: str, tf: str,
                        max_age_seconds: float | None = None) -> bool:
          age = self._fresh.age_seconds(self._key(symbol, tf))
          if age is None:
              return True
          ceiling = max_age_seconds if max_age_seconds is not None else self._ceiling(tf)
          return age >= ceiling

      def get(self, symbol: str, tf: str,
              max_age_seconds: float | None = None):
          ceiling = max_age_seconds if max_age_seconds is not None else self._ceiling(tf)
          return self._fresh.get(self._key(symbol, tf), ceiling)

      def set(self, symbol: str, tf: str, df) -> None:
          self._fresh.set(self._key(symbol, tf), df)
  ```

  No other call sites need to change — `needs_refresh(symbol, tf)` is still valid; `get(symbol, tf)` is still valid; the new `max_age_seconds` argument is optional.

- [ ] **Step 5: Run new + existing OHLCV-related tests, verify pass**

  Run:
  ```bash
  cd bot && python -m pytest tests/test_ohlcv_cache_freshness.py tests/test_freshness.py -v
  ```
  Expected: 14 passed (5 new + 9 from Task 6).

- [ ] **Step 6: Run the full bot test suite for non-regression**

  Run: `cd bot && python -m pytest tests/ -q`
  Expected: same pass count as before this task (no new failures). If any test fails because it imported `OHLCVCache` internals, fix the test rather than the new module.

- [ ] **Step 7: Commit**

  ```bash
  git add bot/data/feed.py bot/tests/test_ohlcv_cache_freshness.py
  git commit -m "$(cat <<'EOF'
  refactor(data): OHLCVCache uses Freshness contract

  OHLCVCache is now a thin wrapper over bot/data/freshness.Freshness with
  per-call max_age_seconds. The previous per-tf CANDLE_SECONDS values become
  the default ceiling. Existing callers (DataFeed) unaffected — the API is
  backwards-compatible. Future callers (regime gate, ensemble) can demand
  fresher data by passing max_age_seconds explicitly.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 8: Route HMM regime cache through Freshness

**Why eighth:** HMM has its own private 5-minute TTL. Decision path wants 60 s. Without this change, the regime label still ages 5 minutes regardless of caller demand.

**Files:**
- Modify: `bot/models/hmm.py:79` (the `_CACHE_TTL = 300` constant and surrounding cache logic, lines 79-235)
- Create: `bot/tests/test_hmm_freshness.py`

- [ ] **Step 1: Re-read HMM cache logic**

  Open `bot/models/hmm.py` lines 70-240 and confirm: (a) `_CACHE_TTL = 300`, (b) `_last_infer_time` initialised in `__init__`, (c) the cache check at ~line 212.

- [ ] **Step 2: Write the failing HMM freshness test**

  Create `bot/tests/test_hmm_freshness.py`:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

  import pandas as pd
  from unittest.mock import patch
  from models.hmm import HMMRegimeDetector


  def _df(n=300):
      idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
      return pd.DataFrame({
          "open":   [100.0] * n,
          "high":   [101.0] * n,
          "low":    [ 99.0] * n,
          "close":  [100.0 + i * 0.01 for i in range(n)],
          "volume": [1000.0] * n,
      }, index=idx)


  def test_regime_is_inferred_when_no_cache():
      det = HMMRegimeDetector()
      with patch.object(det, "_run_inference", return_value="RANGING") as m:
          out = det.predict_regime(_df(), max_age_seconds=60)
          assert out == "RANGING"
          assert m.call_count == 1


  def test_regime_uses_cache_within_max_age():
      det = HMMRegimeDetector()
      with patch.object(det, "_run_inference", return_value="TRENDING_UP") as m:
          det.predict_regime(_df(), max_age_seconds=60)
          det.predict_regime(_df(), max_age_seconds=60)
          assert m.call_count == 1  # second call served from cache


  def test_regime_reinfers_when_caller_demands_fresher():
      det = HMMRegimeDetector()
      with patch.object(det, "_run_inference", return_value="RANGING") as m:
          det.predict_regime(_df(), max_age_seconds=300)
          # Simulate 90 s elapsed by backdating the cache stamp.
          det._fresh._times["regime"] -= 90
          det.predict_regime(_df(), max_age_seconds=60)
          assert m.call_count == 2
  ```

- [ ] **Step 3: Run test, verify failure**

  Run: `cd bot && python -m pytest tests/test_hmm_freshness.py -v`
  Expected: FAIL — the current `predict_regime` either doesn't accept `max_age_seconds` or doesn't expose `_run_inference` separately.

- [ ] **Step 4: Refactor HMM to use Freshness**

  In `bot/models/hmm.py`:

  Remove the `_CACHE_TTL = 300` constant (line 79).

  Remove the `_last_infer_time` initialisation (line 89).

  Add at top of file (after existing imports):
  ```python
  from data.freshness import Freshness
  ```

  In `HMMRegimeDetector.__init__`, replace the cache-stamp init with:
  ```python
  self._fresh = Freshness()
  ```

  Rename the existing inference body (currently inside `predict_regime` after the cache check) to a private method `_run_inference(self, df) -> str` that returns the regime label.

  Replace the `predict_regime` method body with:
  ```python
  def predict_regime(self, df, max_age_seconds: float = 60.0) -> str:
      cached = self._fresh.get("regime", max_age_seconds)
      if cached is not None:
          return cached
      regime = self._run_inference(df)
      self._fresh.set("regime", regime)
      return regime
  ```

  All existing callers of `predict_regime(df)` continue to work — `max_age_seconds` defaults to 60. The decision gate calls it once per scan and gets a fresh inference whenever the prior result is ≥ 60 s old.

- [ ] **Step 5: Run new + non-regression tests**

  Run:
  ```bash
  cd bot && python -m pytest tests/test_hmm_freshness.py tests/test_freshness.py -q
  ```
  Expected: 12 passed.

  Run full bot suite:
  ```bash
  cd bot && python -m pytest tests/ -q
  ```
  Expected: same pass count as after Task 7.

- [ ] **Step 6: Commit**

  ```bash
  git add bot/models/hmm.py bot/tests/test_hmm_freshness.py
  git commit -m "$(cat <<'EOF'
  refactor(models): HMM regime cache uses Freshness contract

  Removed HMMRegimeDetector._CACHE_TTL=300 hardcoded in favour of a
  per-call max_age_seconds on predict_regime (default 60s, was 300s).
  Decision gate now gets a regime re-inference every minute instead of
  every 5 minutes — partial cause of the 'stuck on RANGING' audit
  finding (spec §1 R2). HMM retraining + deterministic overlay come
  in Phase 3.

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 9: Delete the 30-minute decision cache + macro stale-decay

**Why ninth and most impactful:** This is the single highest-impact fix in Phase 1. `AgentCoordinator._decision_cache` returned a cached action for 1800 s — 60 full scan cycles — per symbol. Deleting it forces every cycle to recompute.

**Files:**
- Modify: `bot/agents/coordinator.py:341-475` (`AgentCoordinator` class)
- Create: `bot/tests/test_coordinator_no_decision_cache.py`

- [ ] **Step 1: Write the failing tests**

  Create `bot/tests/test_coordinator_no_decision_cache.py`:
  ```python
  import sys, os
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

  from unittest.mock import MagicMock
  from agents.coordinator import AgentCoordinator


  def _make_coord():
      coord = AgentCoordinator.__new__(AgentCoordinator)
      # Minimal fields the decision path touches.
      coord.fear_greed = MagicMock(); coord.fear_greed.analyze.return_value = {"value": 50}
      coord.macro      = MagicMock(); coord.macro.analyze.return_value = {"market_trend": "NEUTRAL"}
      coord.technical  = MagicMock(); coord.technical.analyze.return_value = {"signal": "NEUTRAL"}
      coord.news       = MagicMock(); coord.news.analyze.return_value = {"signal": "NEUTRAL", "score": 0}
      coord.onchain    = MagicMock(); coord.onchain.analyze.return_value = {"ath_signal": "N/A"}
      coord.master     = MagicMock()
      coord.master.decide.return_value = {
          "action": "BUY", "confidence": 0.7, "source": "ensemble",
          "reasoning": "test", "risk_level": "MEDIUM",
      }
      coord.tracker = MagicMock()
      coord._fresh_slow = None
      coord._fg_cache = None
      coord._macro_cache = None
      coord._slow_time = None
      coord._decision_actions = {}
      return coord


  def test_master_decide_called_every_call_no_30min_cache():
      coord = _make_coord()
      ml_sig = {"action": "BUY", "confidence": 0.7, "indicators": {"buy_votes": 3, "sell_votes": 0}, "strategy": "ml"}
      coord.analyze("BTC/USDT", df=None, ml_signal=ml_sig)
      coord.analyze("BTC/USDT", df=None, ml_signal=ml_sig)
      coord.analyze("BTC/USDT", df=None, ml_signal=ml_sig)
      assert coord.master.decide.call_count == 3


  def test_decision_cache_attribute_removed():
      coord = _make_coord()
      assert not hasattr(coord, "_decision_cache")
      assert not hasattr(coord, "_decision_time")


  def test_macro_weight_full_when_fresh():
      from agents.coordinator import macro_decay_weight
      assert macro_decay_weight(staleness_seconds=0) == 1.0
      assert macro_decay_weight(staleness_seconds=1800) == 1.0


  def test_macro_weight_zero_when_very_stale():
      from agents.coordinator import macro_decay_weight
      assert macro_decay_weight(staleness_seconds=3600) == 0.0
      assert macro_decay_weight(staleness_seconds=10_000) == 0.0


  def test_macro_weight_linear_decay():
      from agents.coordinator import macro_decay_weight
      # Halfway through the decay window [1800, 3600] → weight ~0.5
      w = macro_decay_weight(staleness_seconds=2700)
      assert 0.49 < w < 0.51
  ```

- [ ] **Step 2: Run tests, verify failure**

  Run: `cd bot && python -m pytest tests/test_coordinator_no_decision_cache.py -v`
  Expected: FAIL — `_decision_cache` still present, `macro_decay_weight` does not exist.

- [ ] **Step 3: Edit `AgentCoordinator`**

  In `bot/agents/coordinator.py`:

  Add this module-level function near the top of the file (after imports):
  ```python
  def macro_decay_weight(staleness_seconds: float) -> float:
      """Linear decay of macro signal weight: full < 1800s, zero > 3600s."""
      if staleness_seconds <= 1800.0:
          return 1.0
      if staleness_seconds >= 3600.0:
          return 0.0
      return 1.0 - (staleness_seconds - 1800.0) / 1800.0
  ```

  In `AgentCoordinator.__init__` (around line 354):
  - Change `SLOW_CACHE_SECS = 7200` to `SLOW_CACHE_SECS = 1800` (3600 → 1800 is the spec target for the macro snapshot; the linear decay extends another 1800 s before weight reaches zero).
  - Keep `GROQ_CACHE_SECS` removed entirely or set to 0 (the variable is referenced only by the deleted cache path).
  - **Delete** the initialisation lines `self._decision_cache = {}` and `self._decision_time = {}`.

  In `AgentCoordinator.analyze` (lines 378-465):
  - **Delete** the cache check block at lines 381-390 (`last_dt = ...; cached = ...; if cached and last_dt and ...: return cached`).
  - **Delete** the two cache-write lines `self._decision_cache[symbol] = signal` and `self._decision_time[symbol] = now` at both the ML-only return (~lines 422-423) and the ensemble return (~lines 462-463).
  - Keep `self._decision_actions[symbol] = ...` — that's used by `record_trade_result`.

  In `AgentCoordinator.invalidate_cache` (lines 471-475):
  - Replace body with `pass  # decision cache removed; nothing to invalidate` and a one-line docstring noting decision cache was removed.

- [ ] **Step 4: Run the new tests**

  Run: `cd bot && python -m pytest tests/test_coordinator_no_decision_cache.py -v`
  Expected: 5 passed.

- [ ] **Step 5: Run full bot test suite**

  Run: `cd bot && python -m pytest tests/ -q`
  Expected: same pass count as after Task 8. If `test_macro_flow_agent.py` or `test_ensemble_directional_bias.py` fails because it asserted on a cached path, update the test to assert on the live path (no cached return is the new contract).

- [ ] **Step 6: Smoke run — measure post-Phase-1 scan latency**

  Run a short live cycle:
  ```bash
  # Stop the running bot
  pkill -f 'cryptobot_v3' 2>/dev/null
  sleep 3
  # Start it fresh
  bash start.sh 2
  # Let it run 10 minutes
  sleep 600
  # Capture new latency
  python -m tools.profile_scan_loop --log logs/futures_bot.log --marker "HMM regime" --limit 500 \
      > data/baselines/scan_latency_post_phase1.json
  cat data/baselines/scan_latency_post_phase1.json
  # Compare
  python3 -c "import json; pre=json.load(open('data/baselines/scan_latency_pre_phase1.json')); post=json.load(open('data/baselines/scan_latency_post_phase1.json')); print(f'pre  p95={pre[\"p95\"]:.2f}s'); print(f'post p95={post[\"p95\"]:.2f}s'); print(f'ratio={post[\"p95\"]/pre[\"p95\"]:.2f}x (budget=1.50x)')"
  ```
  Expected: ratio ≤ 1.50. If exceeded, do not commit Task 9 — open a debugging session before proceeding (likely culprit: features pipeline being re-run per call; add a 30-60 s features micro-cache as a Phase 1b task).

- [ ] **Step 7: Commit**

  ```bash
  git add bot/agents/coordinator.py bot/tests/test_coordinator_no_decision_cache.py data/baselines/scan_latency_post_phase1.json
  git commit -m "$(cat <<'EOF'
  fix(agents): delete 30-min decision cache; macro stale-decay

  AgentCoordinator no longer caches the master ensemble verdict for 30 min
  per symbol. Every scan now recomputes the decision; this was the single
  highest-impact stale-data bug (spec §1 R3 — 60 consecutive cycles per
  symbol could return the same cached action while the market moved).

  Macro/Fear&Greed snapshot TTL halved (7200 -> 1800), and a linear-decay
  weight (macro_decay_weight) lets the master ensemble taper macro influence
  to zero between 1800-3600 s of staleness instead of relying on stale data.

  Latency budget verified: post-Phase-1 p95 within 1.5x pre-baseline (see
  data/baselines/scan_latency_post_phase1.json).

  Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
  EOF
  )"
  ```

---

## Task 10: Phase 1 acceptance smoke test

**Why last:** Validate the phase end-to-end before stamping it done. No new code; only a structured verification pass.

- [ ] **Step 1: Run the full bot test suite**

  Run: `cd bot && python -m pytest tests/ -q && python -m pytest backtest/tests/ -q`
  Expected: all green.

- [ ] **Step 2: Run the tool tests**

  Run: `python -m pytest tools/tests/ -q`
  Expected: all green.

- [ ] **Step 3: Verify the watchdog screen is running alongside the bot**

  Run: `screen -ls | grep -E "cryptobot|watchdog|monitor"`
  Expected: three screens — `cryptobot_v3_futures`, `monitor_futures`, `watchdog_futures`.

- [ ] **Step 4: Verify the monitor is being kept fresh**

  Run:
  ```bash
  python3 -c "import os, time; age = time.time() - os.stat('logs/monitor_futures.jsonl').st_mtime; print(f'monitor age: {age:.1f}s')"
  ```
  Expected: age well under 120 s.

- [ ] **Step 5: Verify no stale decision cache survives**

  Run:
  ```bash
  python3 -c "import sys; sys.path.insert(0, 'bot'); from agents.coordinator import AgentCoordinator; c = AgentCoordinator.__new__(AgentCoordinator); assert not hasattr(c, '_decision_cache'); print('ok')"
  ```
  Expected: `ok`.

- [ ] **Step 6: Verify the baseline file is present and reasonable**

  Run:
  ```bash
  python3 -c "import json; d = json.load(open('data/baselines/pre_overhaul_futures.json')); assert d['closed_count'] > 0; assert 'side_mix' in d; print(d['side_mix'], d['win_rate'])"
  ```
  Expected: side mix dominated by `short`, win_rate near 0.42–0.49.

- [ ] **Step 7: Capture a 1-hour observation window after deploy**

  Wait one hour after Task 9's deploy, then:
  ```bash
  python3 -c "
  import json
  with open('logs/monitor_futures.jsonl') as f:
      events = [json.loads(line) for line in f if line.strip()]
  recent = events[-200:]
  closes = [e for e in recent if e.get('type') == 'close']
  print(f'closes in window: {len(closes)}')
  if closes:
      sides = {}
      for c in closes:
          sides[c.get('side','?')] = sides.get(c.get('side','?'),0) + 1
      print(f'side mix: {sides}')
  "
  ```
  Note: Phase 1 does **not** fix the directional bias (that's Phase 2's P3). What this verifies is that the bot is still operating normally and observability is intact. If the bot has crashed or the monitor has gone stale, escalate before declaring Phase 1 done.

- [ ] **Step 8: Tag the Phase 1 completion**

  ```bash
  git tag -a phase1-foundation-complete -m "Signal-quality overhaul Phase 1 (foundation) complete."
  git log --oneline phase1-foundation-complete~10..phase1-foundation-complete
  ```
  Expected: tagged at the last Task 9 commit, log shows the 9 Phase-1 commits.

---

## Acceptance summary (Phase 1 done means…)

1. DeepSeek tuner and its logs are removed from the tree.
2. `bot/backtest/` exists with two CLI skeletons and a working baseline-capture utility; `data/baselines/pre_overhaul_futures.json` is committed.
3. `tools/watchdog.py` runs in its own screen alongside the bot; killing the monitor causes a restart within 90 s.
4. `tools/profile_scan_loop.py` captures p50 / p95 / max; pre and post Phase-1 measurements are committed.
5. `bot/data/freshness.py` exists and is used by `OHLCVCache` and `HMMRegimeDetector`.
6. `AgentCoordinator._decision_cache` is gone; every scan recomputes the ensemble verdict.
7. Macro snapshot TTL halved (1800 s), and `macro_decay_weight` provides a graceful zero-by-3600 s decay.
8. Full bot + tools + backtest test suites all green.
9. Phase-1 latency p95 ≤ 1.5 × pre-Phase-1 p95.
10. Bot still trades (we haven't changed entry logic), `cryptobot_v3_futures` / `monitor_futures` / `watchdog_futures` screens all alive.

## What Phase 1 explicitly does NOT do

- Does not change `trend_filter` (still vetoes longs on weekly downtrend). **Phase 2 (P3).**
- Does not retrain HMM (still likely stuck on RANGING). **Phase 3 (P2).**
- Does not change ensemble agent weights. **Phase 4 (P4).**
- Does not derive `min_confidence` from EV. **Phase 5 (P5).**
- Does not add shadow mode. **Phase 6 (P6 full).**
- Does not add the bias detector. **Phase 7 (P7 full).**

The directional bias the audit found will likely still be visible after Phase 1 — that is expected. Phase 1's job is to make the data freshness honest and the observability reliable so the *next* phase can be evaluated cleanly.

---

*End of plan. Awaiting user choice of execution mode.*
