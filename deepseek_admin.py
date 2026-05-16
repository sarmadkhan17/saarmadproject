#!/usr/bin/env python3
"""
DeepSeek Autonomous Admin – full control with model health monitoring.
"""

import os
import sys
import time
import json
import yaml
import requests
import subprocess
import shutil
import re
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

# ========== CONFIGURATION (your credentials) ==========
DEEPSEEK_API_KEY = "sk-2a5f5a8a34f34ffcbb3463c1b8a3f645"
TELEGRAM_BOT_TOKEN = "8735492279:AAFhM25BjKK7hyNpvqVatdDTIXqMVsLe_Tg"
TELEGRAM_CHAT_ID = "-5155369332"
# =====================================================

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
BOT_DIR = Path("/root/cryptobot_v3")
INTERVAL_SECONDS = 1800          # 30 minutes
AUTO_COMMIT = True
AUTO_PUSH = False

def get_current_mode():
    env_path = BOT_DIR / ".env"
    if not env_path.exists():
        return "futures"
    for line in env_path.read_text().splitlines():
        if line.startswith("BOT_MODE="):
            mode = line.split("=", 1)[1].strip().strip('"').strip("'").lower()
            if mode in ("spot", "futures"):
                return mode
    return "futures"

CURRENT_MODE = get_current_mode()

# Paths
LOG_FILE = BOT_DIR / "logs" / f"{CURRENT_MODE}_bot.log"
STATE_FILE = BOT_DIR / "data" / (f"{CURRENT_MODE}_state.json" if CURRENT_MODE == "futures" else "state.json")
CONFIG_FILE = BOT_DIR / f"config_{CURRENT_MODE}.yaml"
ACTION_LOG = BOT_DIR / f"deepseek_actions_{CURRENT_MODE}.log"
CONFIG_BACKUP = BOT_DIR / f"config_{CURRENT_MODE}.backup.yaml"
AI_BRANCH = f"ai-agent-{CURRENT_MODE}"
MODEL_DIR = BOT_DIR / "data" / CURRENT_MODE

def log_action(msg: str):
    timestamp = datetime.now().isoformat()
    with open(ACTION_LOG, 'a') as f:
        f.write(f"{timestamp} [{CURRENT_MODE}] - {msg}\n")
    print(f"[{timestamp}] {msg}")

def send_telegram(msg: str):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                          json={"chat_id": TELEGRAM_CHAT_ID, "text": msg[:4000]}, timeout=10)
        except Exception:
            pass

def git_ensure_branch():
    subprocess.run(["git", "checkout", "-b", AI_BRANCH], cwd=BOT_DIR, capture_output=True)
    subprocess.run(["git", "checkout", AI_BRANCH], cwd=BOT_DIR)
    log_action(f"Working on branch {AI_BRANCH}")

def git_commit_and_push(message: str):
    if not AUTO_COMMIT:
        return
    subprocess.run(["git", "add", "."], cwd=BOT_DIR)
    subprocess.run(["git", "commit", "-m", message], cwd=BOT_DIR)
    if AUTO_PUSH:
        subprocess.run(["git", "push", "origin", AI_BRANCH], cwd=BOT_DIR)

def read_file(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text()

def collect_project_snapshot() -> str:
    snapshot = f"=== MODE: {CURRENT_MODE} ===\n"
    snapshot += f"=== CONFIG ===\n{read_file(CONFIG_FILE)[:3000]}\n"
    snapshot += f"=== LAST 500 LOG LINES ===\n{read_file(LOG_FILE)[-5000:]}\n"
    snapshot += f"=== STATE (last 30 trades) ===\n{read_file(STATE_FILE)[-2000:]}\n"
    key_files = [
        "bot/engine/ensemble.py",
        "bot/engine/risk_agent.py",
        "bot/engine/smc_agent.py",
        "bot/models/hmm.py",
    ]
    for f in key_files:
        fp = BOT_DIR / f
        if fp.exists():
            snapshot += f"\n=== {f} ===\n{read_file(fp)[:2000]}\n"
    return snapshot

def apply_config_changes(changes: Dict) -> bool:
    backup = CONFIG_FILE.with_suffix('.yaml.bak')
    shutil.copy(CONFIG_FILE, backup)
    config = yaml.safe_load(read_file(CONFIG_FILE))
    modified = False
    for section, kv in changes.items():
        if section not in config:
            config[section] = {}
        for k, v in kv.items():
            config[section][k] = v
            modified = True
    if modified:
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f)
        log_action(f"Config changed: {changes}")
        return True
    return False

def apply_code_patch(file_path: str, diff_text: str) -> Tuple[bool, str]:
    full_path = BOT_DIR / file_path
    if not full_path.exists():
        return False, f"File not found: {file_path}"
    backup = full_path.with_suffix('.py.bak')
    shutil.copy(full_path, backup)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.diff', delete=False) as tf:
        tf.write(diff_text)
        diff_path = tf.name
    try:
        result = subprocess.run(
            ["patch", "--forward", "--quiet", str(full_path), diff_path],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            log_action(f"Patch applied to {file_path}")
            return True, ""
        else:
            shutil.copy(backup, full_path)
            return False, f"Patch failed: {result.stderr}"
    except Exception as e:
        shutil.copy(backup, full_path)
        return False, str(e)
    finally:
        os.unlink(diff_path)

def run_command(cmd: str) -> Tuple[bool, str]:
    try:
        result = subprocess.run(cmd, shell=True, cwd=BOT_DIR, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            log_action(f"Command succeeded: {cmd[:80]}")
            return True, result.stdout[:500]
        else:
            return False, result.stderr[:500]
    except Exception as e:
        return False, str(e)

def restart_bot():
    subprocess.run(["pkill", "-f", "launcher.py"], stderr=subprocess.DEVNULL)
    time.sleep(2)
    env = {**os.environ, "BOT_MODE": CURRENT_MODE}
    subprocess.Popen(
        ["bash", str(BOT_DIR / "start.sh")],
        cwd=BOT_DIR,
        env=env,
    )
    time.sleep(5)
    result = subprocess.run(["pgrep", "-f", "launcher.py"], capture_output=True)
    if result.returncode == 0:
        log_action("Bot restarted successfully via start.sh")
        return True
    else:
        log_action("Bot failed to start via start.sh")
        return False

def models_healthy() -> Tuple[bool, str]:
    """Check if RF and LGBM models exist and are non‑empty."""
    rf_path = MODEL_DIR / "rf_model.pkl"
    lgbm_path = MODEL_DIR / "lgbm_model.pkl"
    if not rf_path.exists():
        return False, f"Missing {rf_path}"
    if rf_path.stat().st_size < 1000:
        return False, f"{rf_path} is too small (corrupted)"
    if not lgbm_path.exists():
        return False, f"Missing {lgbm_path}"
    if lgbm_path.stat().st_size < 1000:
        return False, f"{lgbm_path} is too small (corrupted)"
    return True, "OK"

def fix_models():
    """Delete corrupted model files and restart bot to force retrain."""
    log_action("Model corruption detected – deleting models and restarting bot")
    for p in MODEL_DIR.glob("*.pkl"):
        p.unlink()
    send_telegram("⚠️ ML models were corrupted. Deleting and restarting bot to force retrain.")
    restart_bot()
    # Wait a bit for training to start
    time.sleep(30)

def ask_deepseek(prompt: str) -> str:
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 4000
    }
    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        else:
            return f"API error: {resp.text}"
    except Exception as e:
        return f"Request error: {e}"

def compute_performance(state_text: str) -> Tuple[float, int]:
    try:
        state = json.loads(state_text)
        trades = state.get('trades', [])
        closed = [t for t in trades if t.get('status') == 'closed']
        if not closed:
            return 0.0, 0
        pnls = [t.get('pnl', 0) for t in closed]
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(closed) * 100
        mom = sum(1 for t in closed if 'momentum_reversal' in t.get('close_reason', ''))
        return wr, mom
    except:
        return 0.0, 0

def main():
    log_action(f"DeepSeek Full Agent started in {CURRENT_MODE} mode")
    git_ensure_branch()
    send_telegram(f"🤖 DeepSeek Agent online – monitoring {CURRENT_MODE} mode")
    while True:
        try:
            # ===== MODEL HEALTH CHECK =====
            healthy, msg = models_healthy()
            if not healthy:
                log_action(f"Model health check failed: {msg}")
                fix_models()
                # After fixing, wait one cycle before further actions
                time.sleep(INTERVAL_SECONDS)
                continue

            snapshot = collect_project_snapshot()
            wr, mom = compute_performance(read_file(STATE_FILE))
            
            system_prompt = """You are an expert trading bot engineer. Analyse the project snapshot and output a JSON array of actions to improve profitability.

**IMPORTANT RULES:**
- Never change `min_confidence` by more than ±0.05 per cycle.
- If win rate is undefined (no trades) and models are healthy, consider lowering `min_confidence` slightly.
- Prefer small, incremental config changes over large jumps.
- Output ONLY a JSON array. Example: [{"type": "config_change", "section": "strategy", "key": "min_confidence", "value": 0.44}]
- If no action needed, output [].

Available action types:
1. config_change (section, key, value)
2. code_patch (file, diff)
3. run_command (command)
4. restart (reason)
5. telegram (message)
"""
            user_prompt = f"""Bot mode: {CURRENT_MODE}
Win rate (last closed trades): {wr:.1f}%
Momentum reversal count: {mom}
Full project snapshot:
{snapshot[:15000]}

Now output a JSON array of actions.
"""
            full_prompt = system_prompt + "\n\n" + user_prompt
            response = ask_deepseek(full_prompt)
            log_action(f"DeepSeek raw response: {response[:500]}")

            start = response.find('[')
            end = response.rfind(']') + 1
            if start != -1 and end > start:
                actions = json.loads(response[start:end])
            else:
                actions = []

            changes_made = []
            for act in actions:
                typ = act.get("type")
                if typ == "config_change":
                    section = act.get("section")
                    key = act.get("key")
                    value = act.get("value")
                    if section and key and value is not None:
                        if apply_config_changes({section: {key: value}}):
                            changes_made.append(f"config {section}.{key}={value}")
                elif typ == "code_patch":
                    file_path = act.get("file")
                    diff = act.get("diff")
                    if file_path and diff:
                        ok, err = apply_code_patch(file_path, diff)
                        if ok:
                            changes_made.append(f"patched {file_path}")
                        else:
                            send_telegram(f"⚠️ Patch failed on {file_path}: {err}")
                elif typ == "run_command":
                    cmd = act.get("command")
                    if cmd:
                        ok, output = run_command(cmd)
                        if ok:
                            changes_made.append(f"ran: {cmd[:50]}")
                        else:
                            send_telegram(f"⚠️ Command failed: {cmd}\n{output[:200]}")
                elif typ == "restart":
                    if restart_bot():
                        changes_made.append("restarted bot")
                elif typ == "telegram":
                    msg = act.get("message", "")
                    if msg:
                        send_telegram(f"🤖 Agent: {msg[:500]}")
                        changes_made.append("telegram sent")

            if changes_made:
                summary = f"DeepSeek actions: {', '.join(changes_made)}"
                log_action(summary)
                send_telegram(f"✅ {summary}")
                git_commit_and_push(f"DeepSeek auto: {summary[:80]}")
            else:
                log_action("No actions taken")
        except Exception as e:
            err_msg = f"Agent loop error: {e}"
            log_action(err_msg)
            send_telegram(f"⚠️ {err_msg}")
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
