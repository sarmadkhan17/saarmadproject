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
