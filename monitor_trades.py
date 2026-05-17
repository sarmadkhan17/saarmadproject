#!/usr/bin/env python3
"""Background trade monitor — writes to log file, survives SSH disconnect."""
import json, time, os
from datetime import datetime, timezone

LOG = "/root/cryptobot_v3/logs/monitor_report.log"
STATE = "/root/cryptobot_v3/data/futures_state.json"

def run():
    with open(LOG, "a") as f:
        f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] === MONITOR STARTED ===\n")
    last_pnl = None
    while True:
        try:
            with open(STATE) as fh:
                d = json.load(fh)
            open_trades = [t for t in d["trades"] if t["status"] == "open"]
            stats = d.get("stats", {})
            balance = stats.get("balance", 0)
            total_pnl = stats.get("total_pnl", 0)
            live_pnl = sum(t.get("live_pnl", 0) for t in open_trades)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{now}] Open={len(open_trades)} | Live PnL=${live_pnl:+.4f} | Balance=${balance:.2f} | Total PnL=${total_pnl:+.4f}"
            if open_trades:
                for t in open_trades:
                    line += f" | {t['symbol']} {t['side']} entry={t['price']:.4f} mark={t.get('mark_price',0):.4f} PnL={t.get('live_pnl',0):+.4f} dur={t.get('duration_hours',0):.1f}h"
            with open(LOG, "a") as f:
                f.write(line + "\n")
            # Check for new closed trades
            closed = [t for t in d["trades"] if t["status"] == "closed"]
            if closed:
                latest = max(closed, key=lambda t: t.get("close_timestamp", ""))
                ts = latest.get("close_timestamp", "")
                if ts != last_pnl:
                    last_pnl = ts
                    with open(LOG, "a") as f:
                        f.write(f"  CLOSED: {latest['symbol']} {latest['side']} PnL={latest['pnl']:+.4f} strat={latest['strategy']} dur={latest.get('duration_hours',0):.1f}h\n")
        except Exception as e:
            with open(LOG, "a") as f:
                f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Error: {e}\n")
        time.sleep(30)

if __name__ == "__main__":
    run()
