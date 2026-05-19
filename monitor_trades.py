#!/usr/bin/env python3
"""Background trade monitor — deduped snapshots, all-close emission,
bot-log signal correlation, rolling 1h summary every 5 min.
Supports spot + futures via BOT_MODE env (default futures).

Outputs:
  logs/monitor_{mode}.log    — human-readable
  logs/monitor_{mode}.jsonl  — structured for downstream analysis
"""
import json, time, os, re
from datetime import datetime, timezone, timedelta
from pathlib import Path

MODE = os.environ.get("BOT_MODE", "futures")
ROOT = Path("/root/cryptobot_v3")
STATE = ROOT / "data" / ("futures_state.json" if MODE == "futures" else "state.json")
BOT_LOG = ROOT / "logs" / f"{MODE}_bot.log"
OUT_LOG = ROOT / "logs" / f"monitor_{MODE}.log"
OUT_JSONL = ROOT / "logs" / f"monitor_{MODE}.jsonl"
POLL_S = 30
SUMMARY_EVERY_S = 300       # 5 min
SUMMARY_WINDOW_H = 1.0

SIGNAL_PATTERNS = re.compile(
    r"TAKEN|HOLD reason|ENSEMBLE DEBUG|HMM regime|HARD OVERRIDE|"
    r"Sync.*(imported|skipping|saving)|trained|circuit|invalidation|"
    r"WARNING|ERROR",
    re.IGNORECASE,
)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def write_human(line: str) -> None:
    with open(OUT_LOG, "a") as f:
        f.write(line + "\n")


def write_jsonl(event_type: str, **kwargs) -> None:
    record = {"ts": datetime.now(timezone.utc).isoformat(), "type": event_type, "mode": MODE}
    record.update(kwargs)
    with open(OUT_JSONL, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def snapshot_fingerprint(open_trades, balance) -> str:
    parts = sorted(
        f"{t['symbol']}:{t.get('mark_price', 0):.4f}:{t.get('live_pnl', 0):.4f}"
        for t in open_trades
    )
    return f"{balance:.2f}|{'|'.join(parts)}"


def fmt_trade(t: dict) -> str:
    return (
        f"{t['symbol']} {t['side']} entry={t['price']:.4f} "
        f"mark={t.get('mark_price', 0):.4f} PnL={t.get('live_pnl', 0):+.4f} "
        f"dur={t.get('duration_hours', 0):.1f}h lev={t.get('leverage', '?')}"
    )


def rolling_summary(closed_trades, hours: float = SUMMARY_WINDOW_H):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    recent = []
    for t in closed_trades:
        ts = t.get("close_timestamp", "")
        if not ts:
            continue
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt >= cutoff:
                recent.append(t)
        except Exception:
            continue
    if not recent:
        return None
    wins = [t for t in recent if t.get("pnl", 0) > 0]
    losses = [t for t in recent if t.get("pnl", 0) <= 0]
    total = sum(t.get("pnl", 0) for t in recent)
    by_side, by_strategy, by_symbol = {}, {}, {}
    for t in recent:
        by_side.setdefault(t.get("side", "?"), []).append(t.get("pnl", 0))
        strat = (t.get("strategy") or "?").split(":")[0]
        by_strategy.setdefault(strat, []).append(t.get("pnl", 0))
        by_symbol.setdefault(t.get("symbol", "?"), []).append(t.get("pnl", 0))

    def grp_fmt(g):
        return " ".join(
            f"{k}={len(v)}({sum(1 for p in v if p > 0)}W,${sum(v):+.2f})"
            for k, v in sorted(g.items())
        )

    best = max(recent, key=lambda t: t.get("pnl", 0))
    worst = min(recent, key=lambda t: t.get("pnl", 0))
    return {
        "window_h": hours,
        "count": len(recent),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": len(wins) / len(recent) * 100,
        "total_pnl": total,
        "avg_pnl": total / len(recent),
        "avg_win": (sum(t.get("pnl", 0) for t in wins) / len(wins)) if wins else 0.0,
        "avg_loss": (sum(t.get("pnl", 0) for t in losses) / len(losses)) if losses else 0.0,
        "best": best,
        "worst": worst,
        "by_side": grp_fmt(by_side),
        "by_strategy": grp_fmt(by_strategy),
        "by_symbol": grp_fmt(by_symbol),
    }


def open_log_at_end(path: Path):
    try:
        fh = open(path)
        fh.seek(0, 2)
        return fh, path.stat().st_ino
    except Exception:
        return None, None


def read_new_signal_lines(fh, ino, path: Path):
    """Yield matching new lines from the bot log; reopen on rotation."""
    events = []
    try:
        try:
            cur_ino = path.stat().st_ino
        except Exception:
            cur_ino = ino
        if fh is None or cur_ino != ino:
            if fh is not None:
                try: fh.close()
                except Exception: pass
            fh, ino = open_log_at_end(path)
            return events, fh, ino
        for line in fh:
            line = line.rstrip()
            if SIGNAL_PATTERNS.search(line):
                events.append(line)
    except Exception as e:
        events.append(f"[monitor] bot log read error: {e}")
    return events, fh, ino


def run():
    OUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_human(f"\n[{now_str()}] === MONITOR STARTED (mode={MODE}) ===")
    write_jsonl("started", state=str(STATE), bot_log=str(BOT_LOG))

    last_fp = None
    seen_close_ids = set()
    initial_load = True
    last_summary = 0.0
    bot_log_fh, bot_log_ino = open_log_at_end(BOT_LOG)

    while True:
        try:
            with open(STATE) as fh:
                d = json.load(fh)
            trades = d.get("trades", [])
            open_trades = [t for t in trades if t.get("status") == "open"]
            closed_trades = [t for t in trades if t.get("status") == "closed"]
            stats = d.get("stats", {})
            balance = float(stats.get("balance", 0))
            total_pnl = float(stats.get("total_pnl", 0))
            wins_s = int(stats.get("wins", 0))
            losses_s = int(stats.get("losses", 0))
            live_pnl = sum(t.get("live_pnl", 0) for t in open_trades)

            # Seed historical closes on first pass — don't blast 244 old trades
            if initial_load:
                for t in closed_trades:
                    tid = t.get("id") or f"{t.get('symbol')}_{t.get('close_timestamp')}"
                    seen_close_ids.add(tid)
                initial_load = False
                write_human(f"[{now_str()}] Seeded {len(seen_close_ids)} historical closes")

            # 1) Snapshot — only when changed
            fp = snapshot_fingerprint(open_trades, balance)
            if fp != last_fp:
                wr = (wins_s / (wins_s + losses_s) * 100) if (wins_s + losses_s) else 0.0
                line = (
                    f"[{now_str()}] Open={len(open_trades)} Live=${live_pnl:+.4f} "
                    f"Bal=${balance:.2f} Total=${total_pnl:+.4f} W/L={wins_s}/{losses_s} ({wr:.1f}%w)"
                )
                for t in open_trades:
                    line += f" | {fmt_trade(t)}"
                write_human(line)
                write_jsonl(
                    "snapshot", open=len(open_trades), live_pnl=live_pnl,
                    balance=balance, total_pnl=total_pnl, wins=wins_s, losses=losses_s,
                    positions=[
                        {k: t.get(k) for k in (
                            "symbol", "side", "price", "mark_price", "live_pnl",
                            "duration_hours", "leverage", "amount", "strategy",
                            "timeframe", "sl_order_id"
                        )} for t in open_trades
                    ],
                )
                last_fp = fp

            # 2) All new closes (not just the latest)
            for t in closed_trades:
                tid = t.get("id") or f"{t.get('symbol')}_{t.get('close_timestamp')}"
                if tid in seen_close_ids:
                    continue
                seen_close_ids.add(tid)
                tag = "WIN " if t.get("pnl", 0) > 0 else "LOSS"
                write_human(
                    f"  [{tag}] CLOSED {t['symbol']} {t['side']} "
                    f"entry={t.get('price', 0):.4f} exit={t.get('close_price', 0):.4f} "
                    f"PnL={t.get('pnl', 0):+.4f} dur={t.get('duration_hours', 0):.1f}h "
                    f"lev={t.get('leverage', '?')} strat={t.get('strategy', '?')}"
                )
                write_jsonl("close", **{k: t.get(k) for k in (
                    "id", "symbol", "side", "price", "close_price", "pnl", "live_pnl",
                    "duration_hours", "strategy", "leverage", "amount", "timeframe",
                    "timestamp", "close_timestamp"
                )})

            # 3) Signal/regime/sync events from bot log
            events, bot_log_fh, bot_log_ino = read_new_signal_lines(bot_log_fh, bot_log_ino, BOT_LOG)
            for ev in events:
                write_human(f"  [BOT] {ev}")
                write_jsonl("bot_log", line=ev)

            # 4) Rolling 1h summary every 5 min
            now_s = time.time()
            if now_s - last_summary >= SUMMARY_EVERY_S:
                last_summary = now_s
                s = rolling_summary(closed_trades)
                if s:
                    write_human(
                        f"[{now_str()}] === SUMMARY last {s['window_h']:.1f}h: "
                        f"n={s['count']} W/L={s['wins']}/{s['losses']} ({s['win_rate']:.1f}%w) "
                        f"PnL=${s['total_pnl']:+.4f} avg=${s['avg_pnl']:+.4f} "
                        f"avgW=${s['avg_win']:+.4f} avgL=${s['avg_loss']:+.4f} "
                        f"best={s['best']['symbol']}{s['best'].get('pnl', 0):+.4f} "
                        f"worst={s['worst']['symbol']}{s['worst'].get('pnl', 0):+.4f}"
                    )
                    write_human(f"           by_side:     {s['by_side']}")
                    write_human(f"           by_strategy: {s['by_strategy']}")
                    write_human(f"           by_symbol:   {s['by_symbol']}")
                    write_jsonl(
                        "summary",
                        window_h=s["window_h"], count=s["count"], wins=s["wins"], losses=s["losses"],
                        win_rate=s["win_rate"], total_pnl=s["total_pnl"], avg_pnl=s["avg_pnl"],
                        avg_win=s["avg_win"], avg_loss=s["avg_loss"],
                        by_side=s["by_side"], by_strategy=s["by_strategy"], by_symbol=s["by_symbol"],
                        best_symbol=s["best"]["symbol"], best_pnl=s["best"].get("pnl", 0),
                        worst_symbol=s["worst"]["symbol"], worst_pnl=s["worst"].get("pnl", 0),
                    )
        except Exception as e:
            write_human(f"[{now_str()}] Error: {e}")
            write_jsonl("error", error=str(e))
        time.sleep(POLL_S)


if __name__ == "__main__":
    run()
