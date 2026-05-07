"""
CryptoBot v4 Launcher
Select trading mode: spot or futures (both run on Binance Demo)
"""

import os
import sys
import time
import logging
import traceback
import requests
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

log = logging.getLogger("Launcher")

MAX_CONSECUTIVE_CRASHES = 10


def _load_telegram_config():
    env_path = Path.home() / "cryptobot_v3" / ".env"
    token, chat_id = "", ""
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("TELEGRAM_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                elif line.startswith("TELEGRAM_CHAT_ID="):
                    chat_id = line.split("=", 1)[1].strip()
    return token, chat_id


def _send_telegram_alert(text: str):
    try:
        token, chat_id = _load_telegram_config()
        if not token or not chat_id:
            return
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


def _write_bot_mode_to_env(mode: str):
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


def _alert_crash(mode: str, error: str, crash_count: int):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _send_telegram_alert(
        f"🚨 <b>BOT CRASH #{crash_count}</b>\n"
        f"<b>Mode:</b> {mode.upper()}\n"
        f"<b>Time:</b> {now}\n"
        f"<b>Error:</b> {str(error)[:300]}\n"
        f"<i>Auto-restarting…</i>"
    )


def _alert_recovery(mode: str, crash_count: int):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _send_telegram_alert(
        f"✅ <b>BOT RECOVERED</b>\n"
        f"<b>Mode:</b> {mode.upper()}\n"
        f"<b>Time:</b> {now}\n"
        f"<b>Crashes before recovery:</b> {crash_count}"
    )


def _alert_giving_up(mode: str, crash_count: int):
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    _send_telegram_alert(
        f"💀 <b>BOT STOPPED — TOO MANY CRASHES</b>\n"
        f"<b>Mode:</b> {mode.upper()}\n"
        f"<b>Time:</b> {now}\n"
        f"<b>Consecutive crashes:</b> {crash_count}\n"
        f"<i>Manual intervention required.</i>"
    )


def main():
    mode = os.environ.get("BOT_MODE", "").lower()

    if not mode:
        print("\n" + "=" * 50)
        print("  CryptoBot v4 — Binance Demo Trading")
        print("=" * 50)
        print("  1. Spot Trading    (BUY/SELL only)")
        print("  2. Futures Trading (LONG/SHORT + leverage)")
        print("=" * 50)
        choice = input("  Enter choice (1 or 2): ").strip()
        mode = "futures" if choice == "2" else "spot"

    _write_bot_mode_to_env(mode)

    restart_delay     = 30
    max_delay         = 300
    crash_count       = 0
    recovered_once    = False

    while True:
        try:
            if mode == "futures":
                print("\nStarting FUTURES bot (LONG + SHORT, demo trading)...")
                from engine.futures import FuturesBot
                bot = FuturesBot()
            else:
                print("\nStarting SPOT bot (BUY/SELL, demo trading)...")
                from engine.spot import SpotBot
                bot = SpotBot()

            if crash_count > 0 and not recovered_once:
                _alert_recovery(mode, crash_count)
                recovered_once = True

            restart_delay = 30   # reset on successful start
            crash_count   = 0
            bot.run()

        except KeyboardInterrupt:
            print("\nBot stopped by user.")
            break

        except Exception as e:
            crash_count += 1
            tb  = traceback.format_exc()
            msg = f"CRASH [{mode.upper()}] #{crash_count}: {type(e).__name__}: {e}\n{tb}"
            print(msg)

            # Write to log file
            log_path = Path.home() / "cryptobot_v3" / "logs" / f"{mode}_bot.log"
            try:
                log_path.parent.mkdir(exist_ok=True)
                with open(log_path, "a") as f:
                    f.write(f"\n[LAUNCHER CRASH] {msg}\n")
            except Exception:
                pass
            logging.error(msg)

            # Telegram crash alert
            _alert_crash(mode, f"{type(e).__name__}: {e}", crash_count)

            # Circuit breaker: give up after too many consecutive crashes
            if crash_count >= MAX_CONSECUTIVE_CRASHES:
                _alert_giving_up(mode, crash_count)
                print(f"Too many consecutive crashes ({crash_count}). Stopping.")
                break

            recovered_once = False
            print(f"Restarting in {restart_delay}s... (crash #{crash_count})")
            time.sleep(restart_delay)
            restart_delay = min(restart_delay * 2, max_delay)


if __name__ == "__main__":
    main()
