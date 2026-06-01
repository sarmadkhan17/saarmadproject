"""
CryptoBot v4 Launcher
Select trading mode: spot or futures (both run on Binance Demo)
"""

import os
import sys
import json
import time
import logging
import threading
import traceback
import requests
from datetime import datetime, timezone
from core.tz import LOCAL_TZ
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

BOT_ROOT = Path(__file__).parent.parent  # /root/cryptobot_v5

log = logging.getLogger("Launcher")

MAX_CONSECUTIVE_CRASHES = 10


def _load_telegram_config():
    env_path = BOT_ROOT / ".env"
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
    os.environ["BOT_MODE"] = mode
    env_path = BOT_ROOT / ".env"
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
    now = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M UTC+3")
    _send_telegram_alert(
        f"🚨 <b>BOT CRASH #{crash_count}</b>\n"
        f"<b>Mode:</b> {mode.upper()}\n"
        f"<b>Time:</b> {now}\n"
        f"<b>Error:</b> {str(error)[:300]}\n"
        f"<i>Auto-restarting…</i>"
    )


def _alert_recovery(mode: str, crash_count: int):
    now = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M UTC+3")
    _send_telegram_alert(
        f"✅ <b>BOT RECOVERED</b>\n"
        f"<b>Mode:</b> {mode.upper()}\n"
        f"<b>Time:</b> {now}\n"
        f"<b>Crashes before recovery:</b> {crash_count}"
    )


def _alert_giving_up(mode: str, crash_count: int):
    now = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M UTC+3")
    _send_telegram_alert(
        f"💀 <b>BOT STOPPED — TOO MANY CRASHES</b>\n"
        f"<b>Mode:</b> {mode.upper()}\n"
        f"<b>Time:</b> {now}\n"
        f"<b>Consecutive crashes:</b> {crash_count}\n"
        f"<i>Manual intervention required.</i>"
    )


def _heartbeat_age(mode: str):
    """Seconds since the bot last updated its heartbeat, or None if unreadable.

    The bot writes data/bot_heartbeat_<mode>.json before and after every scan
    cycle (engine/bot.py:_write_heartbeat).
    """
    hb = BOT_ROOT / "data" / f"bot_heartbeat_{mode}.json"
    try:
        ts = json.loads(hb.read_text())["timestamp"]
        last = datetime.fromisoformat(ts)
        if last.tzinfo is None:
            last = last.replace(tzinfo=LOCAL_TZ)
        return (datetime.now(LOCAL_TZ) - last).total_seconds()
    except Exception:
        return None


def _start_watchdog(mode: str):
    """Detect a *silent* scan-loop stall and self-recover.

    The launcher's crash handler only catches exceptions; a hang inside
    bot.run() (e.g. a wedged exchange request) raises nothing, so the bot can
    sit frozen indefinitely with no alert. This daemon thread watches the
    heartbeat and, if it goes stale, alerts on Telegram and re-execs the
    process (the only reliable recovery when the main thread is wedged and
    there is no external supervisor).
    """
    stale_after = int(os.environ.get("WATCHDOG_STALE_SECONDS", "240"))
    poll_every  = 30
    grace_until = time.time() + 120  # let the bot boot before judging liveness

    def _loop():
        while True:
            time.sleep(poll_every)
            if time.time() < grace_until:
                continue
            age = _heartbeat_age(mode)
            if age is None or age <= stale_after:
                continue
            now = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M UTC+3")
            msg = (
                f"🛑 <b>BOT STALLED — auto-restarting</b>\n"
                f"<b>Mode:</b> {mode.upper()}\n"
                f"<b>Time:</b> {now}\n"
                f"<b>Heartbeat age:</b> {int(age)}s (limit {stale_after}s)\n"
                f"<i>Scan loop frozen with no exception; re-execing process.</i>"
            )
            _send_telegram_alert(msg)
            logging.error(f"WATCHDOG: heartbeat stale {int(age)}s — re-execing process")
            # Replace the wedged process image with a fresh launcher. BOT_MODE
            # is already in the environment, so the restart is non-interactive.
            os.execv(sys.executable, [sys.executable, "-u", str(Path(__file__).resolve())])

    threading.Thread(target=_loop, name="watchdog", daemon=True).start()
    logging.info(f"Watchdog started (mode={mode}, stale_after={stale_after}s)")


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
    _start_watchdog(mode)

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
            log_path = BOT_ROOT / "logs" / f"{mode}_bot.log"
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
