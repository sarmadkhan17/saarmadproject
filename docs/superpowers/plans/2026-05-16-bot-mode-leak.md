# BOT_MODE Leak — Fix Cross-Mode Model Load

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the futures bot from loading the spot RF model (67 features) when its feature pipeline emits 268 features.

**Architecture:** Three compounding bugs cause `AIStrategyEngine` to default to "spot" inside a futures process. Fix each independently — defense in depth — so any one of them being wrong doesn't recreate the mismatch:

1. `deepseek_admin.py` restart path drops env vars → set them explicitly.
2. `launcher.py` writes BOT_MODE to the .env file but not to `os.environ` of the current process → propagate to current process too.
3. `ai_strategy.py` reads `BOT_MODE` at module import → read at call time inside `RandomForestStrategy`/`LightGBMStrategy` instead.

After the code fix, restart the bot so the corrected `RandomForestStrategy` actually runs. The stale `/data/spot/rf_*.pkl` (67-feature legacy artifact) is left in place — it gets rebuilt the next time a spot bot trains, and removing it now has no effect on futures.

**Tech Stack:** Python 3, `subprocess`, `os.environ`, `joblib`/sklearn model artifacts.

---

### Task 1: Fix deepseek_admin.py — preserve env on subprocess restart

**Files:**
- Modify: `/root/cryptobot_v3/deepseek_admin.py:102-105`

- [ ] **Step 1: Read current `restart_bot()` to confirm line numbers**

Run: `grep -n "def restart_bot\|subprocess.Popen" /root/cryptobot_v3/deepseek_admin.py`
Expected output includes `def restart_bot():` and `subprocess.Popen(["python3", str(BOT_DIR / "bot/launcher.py")], cwd=BOT_DIR / "bot")`.

- [ ] **Step 2: Apply the edit**

Replace the body of `restart_bot()` so it (a) reads the current mode from `.env`, (b) passes a full env dict to the child including `BOT_MODE`. Read the existing file with the Read tool, then use Edit:

```python
def _read_bot_mode_from_env_file():
    env_path = BOT_DIR / ".env"
    if not env_path.exists():
        return ""
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("BOT_MODE="):
            return line.split("=", 1)[1].strip().strip('"').strip("'").lower()
    return ""


def restart_bot():
    subprocess.run(["pkill", "-f", "launcher.py"])
    time.sleep(2)
    mode = _read_bot_mode_from_env_file() or "futures"
    env = {**os.environ, "BOT_MODE": mode}
    subprocess.Popen(
        ["python3", str(BOT_DIR / "bot/launcher.py")],
        cwd=BOT_DIR / "bot",
        env=env,
    )
```

Also ensure `import os` is present at the top of the file (it already is — verify with grep, do not add a duplicate).

- [ ] **Step 3: Static-check the change**

Run: `python3 -m py_compile /root/cryptobot_v3/deepseek_admin.py`
Expected: exits 0, no output.

- [ ] **Step 4: Commit**

```bash
cd /root/cryptobot_v3
git add deepseek_admin.py
git commit -m "fix: deepseek_admin propagates BOT_MODE to restarted bot subprocess"
```

---

### Task 2: Fix launcher.py — also set os.environ when writing BOT_MODE

**Files:**
- Modify: `/root/cryptobot_v3/bot/launcher.py:50-66` (`_write_bot_mode_to_env`)
- Modify: `/root/cryptobot_v3/bot/launcher.py:114` (call site — ensure env is set even if file write fails)

- [ ] **Step 1: Re-read the function to confirm shape**

Run: `sed -n '50,67p' /root/cryptobot_v3/bot/launcher.py`
Expected: shows the existing `_write_bot_mode_to_env(mode: str)`.

- [ ] **Step 2: Edit `_write_bot_mode_to_env` to also update `os.environ`**

The function currently writes to `.env` only. Make it set the current process's env var too — *before* touching disk so a write failure doesn't strand us with stale env.

Replace the function body with:

```python
def _write_bot_mode_to_env(mode: str):
    os.environ["BOT_MODE"] = mode
    env_path = Path.home() / "cryptobot_v3" / ".env"
    if not env_path.exists():
        return
    lines = env_path.read_text().splitlines(keepends=True)
    found = False
    for i, line in enumerate(lines):
        if line.startswith("BOT_MODE="):
            lines[i] = f'BOT_MODE="{mode}"\n'
            found = True
            break
    if not found:
        lines.append(f'BOT_MODE="{mode}"\n')
    try:
        env_path.write_text("".join(lines))
    except OSError:
        pass  # Docker mounts .env read-only — BOT_MODE already set via env var
```

The single added line is `os.environ["BOT_MODE"] = mode` at the top of the function.

- [ ] **Step 3: Static-check the change**

Run: `python3 -m py_compile /root/cryptobot_v3/bot/launcher.py`
Expected: exits 0.

- [ ] **Step 4: Commit**

```bash
cd /root/cryptobot_v3
git add bot/launcher.py
git commit -m "fix: launcher writes BOT_MODE to os.environ as well as .env file"
```

---

### Task 3: Fix ai_strategy.py — read BOT_MODE at call time, not import time

**Files:**
- Modify: `/root/cryptobot_v3/bot/models/ai_strategy.py:31-32` (module-level constants)
- Modify: `/root/cryptobot_v3/bot/models/ai_strategy.py:264` (`RandomForestStrategy.__init__`)
- Modify: `/root/cryptobot_v3/bot/models/ai_strategy.py:416` (`LightGBMStrategy.__init__`)

The current module-level `BOT_MODE = os.environ.get("BOT_MODE", "spot")` freezes the value at import time. Replace with a helper that reads at call time, and use it in both strategy `__init__`s. Keep the module-level `DATA` for callers that still rely on it.

- [ ] **Step 1: Confirm current code shape**

Run:
```bash
sed -n '28,35p' /root/cryptobot_v3/bot/models/ai_strategy.py
sed -n '262,275p' /root/cryptobot_v3/bot/models/ai_strategy.py
sed -n '414,425p' /root/cryptobot_v3/bot/models/ai_strategy.py
```
Expected: shows the three regions.

- [ ] **Step 2: Replace module-level `BOT_MODE` and `DATA`**

Edit `/root/cryptobot_v3/bot/models/ai_strategy.py` at the top of the file (around lines 31-32):

Old:
```python
BOT_MODE = os.environ.get("BOT_MODE", "spot")
DATA = DATA_DIR / BOT_MODE
```

New:
```python
def _current_bot_mode() -> str:
    return os.environ.get("BOT_MODE", "spot")

DATA = DATA_DIR / _current_bot_mode()
```

Note: `DATA` keeps the existing behavior (resolved at import) — only the per-strategy data dir resolution changes. Module-level `DATA` is used at line 583 (`p = DATA / "trade_results.json"`); leaving its capture-at-import behavior is intentional to keep this change minimal.

- [ ] **Step 3: Update `RandomForestStrategy.__init__` to call the helper**

Old (line 264):
```python
data_dir = DATA_DIR / (mode or BOT_MODE)
```

New:
```python
data_dir = DATA_DIR / (mode or _current_bot_mode())
```

- [ ] **Step 4: Update `LightGBMStrategy.__init__` to call the helper**

Old (line 416):
```python
data_dir = DATA_DIR / (mode or BOT_MODE)
```

New:
```python
data_dir = DATA_DIR / (mode or _current_bot_mode())
```

- [ ] **Step 5: Static-check the change**

Run: `python3 -m py_compile /root/cryptobot_v3/bot/models/ai_strategy.py`
Expected: exits 0.

- [ ] **Step 6: Behavior check — module-level reference no longer dangling**

Run:
```bash
grep -n "BOT_MODE" /root/cryptobot_v3/bot/models/ai_strategy.py
```
Expected: only references are `_current_bot_mode`, plus any pre-existing references inside the module that we didn't touch. There should be **no** bare `BOT_MODE` identifier referenced anywhere outside `os.environ.get("BOT_MODE", ...)`.

- [ ] **Step 7: Commit**

```bash
cd /root/cryptobot_v3
git add bot/models/ai_strategy.py
git commit -m "fix: ai_strategy reads BOT_MODE at call time, not module import"
```

---

### Task 4: End-to-end verification — confirm the futures bot loads the futures RF

**Files:** none modified. This is a runtime check before declaring the fix done.

- [ ] **Step 1: Stop the running bot**

Run: `pkill -f 'cryptobot_v3.*launcher.py'`
Then: `sleep 3 && ps aux | grep -E 'launcher\.py' | grep -v grep`
Expected: no `launcher.py` process listed.

- [ ] **Step 2: Restart via deepseek_admin (the same path that was broken)**

`deepseek_admin.py` has its own restart path we just fixed. Trigger it the same way the user does:

Run: `pgrep -af deepseek_admin.py`
Expected: shows existing `python3 deepseek_admin.py` running.

If it's running, it will call `restart_bot()` on its next cycle automatically. Otherwise, restart manually via:
```bash
cd /root/cryptobot_v3
nohup python3 deepseek_admin.py >> deepseek_actions.log 2>&1 &
```

Wait ~45s for the bot to come up:
```bash
sleep 45
```

- [ ] **Step 3: Verify the bot is in futures mode AND loaded the 268-feature RF**

```bash
BOT_PID=$(pgrep -f 'cryptobot_v3.*launcher.py' | head -1)
echo "bot pid=$BOT_PID"
tr '\0' '\n' < /proc/$BOT_PID/environ | grep BOT_MODE
tail -50 /root/cryptobot_v3/logs/futures_bot.log | grep -E "RF loaded|LightGBM loaded|MODE: FUTURES|MODE: SPOT"
```

Expected:
- `BOT_MODE=futures` in the process environ (was missing before).
- `RF loaded (accuracy=0.6027)` in the log (futures accuracy, not the spot 0.5333).
- `CRYPTOBOT v4 STARTED — MODE: FUTURES`.

- [ ] **Step 4: Confirm no more feature-mismatch warnings in the next cycle**

Wait one scan cycle, then:
```bash
sleep 60
tail -100 /root/cryptobot_v3/logs/futures_bot.log | grep -E "expecting 67 features|SIGNAL .*HOLD \| conf=0\.00"
```

Expected: no matches. (Previously every symbol every cycle produced both lines.)

- [ ] **Step 5: Confirm signals now carry non-zero confidence**

```bash
tail -100 /root/cryptobot_v3/logs/futures_bot.log | grep -E "SIGNAL " | tail -20
```

Expected: at least some lines with `conf=` values > 0.00 (BUY/SELL/HOLD with non-trivial confidence).

- [ ] **Step 6: Commit (no-op if nothing changed)**

This task makes no code changes; only run the verification. If any of the expected outputs are missing, **STOP** — do not declare the fix done. Re-investigate before committing anything else.

---

## Out of scope

- Deleting `/data/spot/rf_*.pkl`: harmless to leave; spot bot will retrain on next run.
- Refactoring `risk/manager.py` which has the *same* `BOT_MODE = os.environ.get(...)` module-level pattern at line 22. With Tasks 1+2 in place, `os.environ["BOT_MODE"]` will be set correctly before `risk/manager.py` is ever imported, so the latent bug there can't manifest. Leaving for a future cleanup pass.
- Refactoring `exchange/factory.py:67,72` which reads env at call time — already correct.
