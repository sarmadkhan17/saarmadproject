#!/usr/bin/env python3
"""
DeepSeek Autonomous Admin – watches the bot, fixes problems, saves changes on 'ai-agent' branch.
"""

import os
import sys
import time
import json
import yaml
import requests
import subprocess
import shutil
from pathlib import Path
from datetime import datetime

# ========== SETTINGS – YOU MUST CHANGE THESE ==========
DEEPSEEK_API_KEY = "sk-2a5f5a8a34f34ffcbb3463c1b8a3f645"   # <-- PUT YOUR REAL API KEY HERE
TELEGRAM_BOT_TOKEN = "8735492279:AAFhM25BjKK7hyNpvqVatdDTIXqMVsLe_Tg"          # Optional: leave empty for now
TELEGRAM_CHAT_ID = "-5155369332"            # Optional: leave empty for now
# ======================================================

DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
BOT_DIR = Path("/root/cryptobot_v3")
LOG_FILE = BOT_DIR / "logs/futures_bot.log"
STATE_FILE = BOT_DIR / "data/futures_state.json"
CONFIG_FILE = BOT_DIR / "config_futures.yaml"
CONFIG_BACKUP = BOT_DIR / "config_futures.backup.yaml"
ACTION_LOG = BOT_DIR / "deepseek_actions.log"
AI_BRANCH = "ai-agent"
INTERVAL_SECONDS = 1800   # 30 minutes
AUTO_APPLY_CONFIG = True
AUTO_COMMIT = True
AUTO_PUSH = False   # Set to True if you have a git remote

def log_action(msg):
    with open(ACTION_LOG, 'a') as f:
        f.write(f"{datetime.now().isoformat()} - {msg}\n")
    print(msg)

def send_telegram(msg):
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        try:
            requests.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                          json={"chat_id": TELEGRAM_CHAT_ID, "text": msg[:4000]}, timeout=10)
        except Exception:
            pass

def git_ensure_branch():
    subprocess.run(["git", "checkout", "-b", AI_BRANCH], cwd=BOT_DIR, capture_output=True)
    subprocess.run(["git", "checkout", AI_BRANCH], cwd=BOT_DIR)

def git_commit_and_push(message):
    if not AUTO_COMMIT:
        return
    subprocess.run(["git", "add", "."], cwd=BOT_DIR)
    subprocess.run(["git", "commit", "-m", message], cwd=BOT_DIR)
    if AUTO_PUSH:
        subprocess.run(["git", "push", "origin", AI_BRANCH], cwd=BOT_DIR)

def read_logs():
    if not LOG_FILE.exists():
        return ""
    with open(LOG_FILE, 'r') as f:
        return ''.join(f.readlines()[-400:])

def read_state():
    if not STATE_FILE.exists():
        return {}
    with open(STATE_FILE, 'r') as f:
        return json.load(f)

def get_performance(state):
    trades = state.get('trades', [])
    closed = [t for t in trades if t.get('status') == 'closed']
    if not closed:
        return "No closed trades", 0
    pnls = [t.get('pnl', 0) for t in closed]
    wins = sum(1 for p in pnls if p > 0)
    wr = wins / len(closed) * 100
    total = sum(pnls)
    mom = sum(1 for t in closed if 'momentum_reversal' in t.get('close_reason', ''))
    return f"WR: {wr:.1f}% | PnL: {total:+.2f} | momentum: {mom}", mom

def apply_config(changes):
    if not AUTO_APPLY_CONFIG:
        return False
    shutil.copy(CONFIG_FILE, CONFIG_BACKUP)
    config = yaml.safe_load(open(CONFIG_FILE))
    modified = False
    for sec in ['strategy', 'risk']:
        if sec in changes:
            for k, v in changes[sec].items():
                if k in config.get(sec, {}):
                    config[sec][k] = v
                    modified = True
    if modified:
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(config, f)
    return modified

def restart_bot():
    subprocess.run(["pkill", "-f", "launcher.py"])
    time.sleep(2)
    subprocess.Popen(["python3", str(BOT_DIR / "bot/launcher.py")], cwd=BOT_DIR / "bot")

def ask_deepseek(prompt):
    headers = {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 800
    }
    resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=30)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    return f"API error: {resp.text}"

def main():
    log_action("DeepSeek Admin started")
    git_ensure_branch()
    while True:
        try:
            logs = read_logs()
            state = read_state()
            perf, mom = get_performance(state)
            prompt = f"""Bot performance: {perf}
Momentum reversal count: {mom}
Log tail:
{logs[-2000:]}

If win rate <45% or momentum_reversal >3, suggest one config change (min_confidence or stop_loss_atr_mult). Output ONLY a JSON like:
{{"min_confidence": 0.45}} or {{"stop_loss_atr_mult": 2.5}} or {{}} if no change.
"""
            answer = ask_deepseek(prompt)
            log_action(f"DeepSeek says: {answer[:200]}")
            changes = {}
            try:
                if '"min_confidence"' in answer:
                    import re
                    m = re.search(r'"min_confidence":\s*([0-9.]+)', answer)
                    if m:
                        changes["strategy"] = {"min_confidence": float(m.group(1))}
                if '"stop_loss_atr_mult"' in answer:
                    m = re.search(r'"stop_loss_atr_mult":\s*([0-9.]+)', answer)
                    if m:
                        changes["risk"] = {"stop_loss_atr_mult": float(m.group(1))}
            except:
                pass

            if changes:
                if apply_config(changes):
                    msg = f"Applied config: {changes}"
                    log_action(msg)
                    send_telegram(msg)
                    git_commit_and_push(f"DeepSeek auto: {changes}")
                    restart_bot()
            else:
                log_action("No config change needed")
        except Exception as e:
            log_action(f"Error: {e}")
            send_telegram(f"Admin error: {e}")
        time.sleep(INTERVAL_SECONDS)

if __name__ == "__main__":
    main()
