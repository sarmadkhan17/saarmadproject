"""
Dashboard Server v4
Same as v2 but with separate Spot & Futures sections
"""

import fcntl
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone, date
from core.tz import LOCAL_TZ
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "bot"))
from env_config import DATA_DIR, LOGS_DIR

BOT_ROOT   = Path.home() / "cryptobot_v3"
CFG_SPOT   = BOT_ROOT / "config_spot.yaml"
CFG_FUT    = BOT_ROOT / "config_futures.yaml"
HTF_MODES  = ("strict", "soft", "hard")
VALID_TFS  = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d")

# Optional API key — set DASHBOARD_KEY env var to enable auth on mutation endpoints
API_KEY = os.environ.get("DASHBOARD_KEY", "")

app  = Flask(__name__, static_folder="static")
# Restrict CORS to localhost — blocks cross-origin requests from other domains
CORS(app, origins=[
    "http://localhost:5002", "http://127.0.0.1:5002",
    "http://localhost:3000", "http://127.0.0.1:3000",
])
DATA = DATA_DIR

_EMPTY_STATE = {
    "trades": [], "signals": [],
    "stats": {"total_trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0},
}


# ── Auth ──────────────────────────────────────────────────────────────────────

def _check_auth():
    """Returns (None, None) if auth passes, or (response, status) if denied."""
    if not API_KEY:
        return None, None
    key = request.headers.get("X-Dashboard-Key", "")
    if key != API_KEY:
        return jsonify({"error": "unauthorized"}), 401
    return None, None


# ── State loading ──────────────────────────────────────────────────────────────

def load_state(filename="state.json"):
    """Load state JSON. Falls back to .backup.json on parse error."""
    p = DATA / filename
    backup = p.with_suffix(".backup.json")
    for path in (p, backup):
        if path.exists():
            try:
                with open(path) as f:
                    data = json.load(f)
                # Ensure required keys exist
                data.setdefault("trades", [])
                data.setdefault("signals", [])
                data.setdefault("stats", {})
                return data
            except (json.JSONDecodeError, OSError, ValueError):
                continue
    return dict(_EMPTY_STATE)


def load_archive_trades(stem: str) -> list:
    """Load archived trades from <stem>_archive.json (list or {trades:[...]} format)."""
    p = DATA / f"{stem}_archive.json"
    if not p.exists():
        return []
    try:
        with open(p) as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("trades", [])
    except (json.JSONDecodeError, OSError, ValueError):
        return []


def load_combined():
    spot    = load_state("state.json")
    futures = load_state("futures_state.json")
    # Prepend archived trades so stats include full history; display endpoints
    # limit to most-recent N so old archives don't clutter the UI.
    spot_archived    = load_archive_trades("state")
    futures_archived = load_archive_trades("futures_state")
    return {
        "spot":    spot,
        "futures": futures,
        # Full lists (archive + live) used for stats calculations.
        "spot_all_trades":    spot_archived    + spot.get("trades", []),
        "futures_all_trades": futures_archived + futures.get("trades", []),
        "combined_trades":    (spot_archived    + spot.get("trades", []) +
                               futures_archived + futures.get("trades", [])),
        "combined_signals": spot.get("signals", []) + futures.get("signals", []),
    }


# ── Config write with file locking ────────────────────────────────────────────

def _write_config_safe(cfg_path: Path, cfg: dict):
    """Atomic read-modify-write on a YAML config with exclusive file lock."""
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = cfg_path.with_suffix(".tmp.yaml")
    with open(cfg_path, "a") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        with open(tmp, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)
        tmp.replace(cfg_path)
        fcntl.flock(lf, fcntl.LOCK_UN)


# ── Stats ─────────────────────────────────────────────────────────────────────

def _risk_metrics(trades: list) -> dict:
    """Compute Sharpe, Sortino, and Profit Factor from closed trades.

    Uses daily-bucketed PnL so the result is comparable to standard
    annualized metrics (×√252). Requires ≥5 trading days to be meaningful;
    returns None for each metric when there is insufficient data.
    """
    import numpy as np
    closed = [t for t in trades if t.get("status") == "closed"
              and t.get("close_timestamp")]
    daily: dict = {}
    for t in closed:
        day = str(t.get("close_timestamp", ""))[:10]
        if day and day != "None":
            daily[day] = daily.get(day, 0.0) + float(t.get("pnl", 0))

    n_days = len(daily)
    if n_days < 5:
        return {"sharpe": None, "sortino": None, "profit_factor": None,
                "trading_days": n_days}

    arr     = np.array(list(daily.values()), dtype=float)
    mean_r  = float(np.mean(arr))
    std_r   = float(np.std(arr, ddof=1)) if n_days > 1 else 0.0
    neg     = arr[arr < 0]
    down_std = float(np.std(neg, ddof=1)) if len(neg) > 1 else (float(abs(neg[0])) if len(neg) == 1 else 1e-9)

    sharpe  = round(mean_r / (std_r + 1e-9) * (252 ** 0.5), 2) if std_r > 0 else None
    sortino = round(mean_r / (down_std + 1e-9) * (252 ** 0.5), 2) if down_std > 0 else None

    gross_win  = sum(t.get("pnl", 0) for t in closed if t.get("pnl", 0) > 0)
    gross_loss = sum(abs(t.get("pnl", 0)) for t in closed if t.get("pnl", 0) < 0)
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else None

    return {
        "sharpe":        sharpe,
        "sortino":       sortino,
        "profit_factor": pf,
        "trading_days":  n_days,
        "sharpe_note":   f"Based on {n_days}d; need ≥30d for reliable annualization",
    }


def calculate_stats(trades):
    closed = [t for t in trades if t.get("status") == "closed"]
    open_trades = [t for t in trades if t.get("status") == "open"]
    wins   = sum(1 for t in closed if t.get("pnl", 0) > 0)
    losses = sum(1 for t in closed if t.get("pnl", 0) < 0)  # break-even not a loss
    total  = wins + losses

    # Realized PnL = closed trade PnL + partial TP1 PnL already locked in on open trades
    closed_pnl       = sum(t.get("pnl", 0) for t in closed)
    open_realized    = sum(t.get("pnl", 0) for t in open_trades if t.get("pnl", 0) != 0)
    total_realized   = closed_pnl + open_realized

    result = {
        "total_trades":   len(trades),
        "open_trades":    len(open_trades),
        "wins":           wins,
        "losses":         losses,
        "win_rate":       round(wins / total * 100, 1) if total else 0.0,
        "total_pnl":      round(total_realized, 4),
        "closed_pnl":     round(closed_pnl, 4),
        "open_realized":  round(open_realized, 4),
    }
    result.update(_risk_metrics(trades))
    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/api/stats")
def stats():
    d      = load_combined()
    result = calculate_stats(d["combined_trades"])

    fut_stats       = d["futures"].get("stats", {})
    spot_stats_data = d["spot"].get("stats", {})

    balance = fut_stats.get("balance", 0) or spot_stats_data.get("balance", 0)
    result["balance"]        = round(float(balance or 0), 2)
    live_pnl = fut_stats.get("total_live_pnl", 0) or 0
    result["total_live_pnl"] = round(float(live_pnl), 4)
    result["last_sync"]      = fut_stats.get("last_sync", "never")

    today        = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
    today_trades = [
        t for t in d["combined_trades"]
        if t.get("status") == "closed" and t.get("close_timestamp", "")[:10] == today
    ]
    result["today_pnl"]    = round(sum(t.get("pnl", 0) for t in today_trades), 4)
    result["today_trades"] = len(today_trades)
    result["today_wins"]   = sum(1 for t in today_trades if t.get("pnl", 0) > 0)
    result["start_balance"] = round(float(d["futures"].get("stats", {}).get("balance", 0)
                                          - result["total_pnl"]
                                          - float(live_pnl or 0)), 2)

    p = DATA / "trading_paused.json"
    if p.exists():
        try:
            with open(p) as f:
                pause_data = json.load(f)
            result["trading_paused"] = pause_data.get("paused", False)
        except (json.JSONDecodeError, OSError):
            result["trading_paused"] = False
    else:
        result["trading_paused"] = False

    return jsonify(result)


@app.route("/api/spot_stats")
def spot_stats():
    d = load_combined()
    return jsonify(calculate_stats(d["spot_all_trades"]))


@app.route("/api/futures_stats")
def futures_stats():
    d = load_combined()
    return jsonify(calculate_stats(d["futures_all_trades"]))


@app.route("/api/spot_trades")
def spot_trades():
    d     = load_combined()
    all_t = d["spot"].get("trades", [])
    limit = min(request.args.get("limit", 500, type=int), 1000)
    return jsonify(all_t[-limit:][::-1])


@app.route("/api/futures_trades")
def futures_trades():
    d     = load_combined()
    all_t = d["futures"].get("trades", [])
    limit = min(request.args.get("limit", 500, type=int), 1000)
    return jsonify(all_t[-limit:][::-1])


@app.route("/api/futures")
def futures_combined():
    d             = load_combined()
    futures_data  = d["futures"]
    trades        = futures_data.get("trades", [])
    signals       = futures_data.get("signals", [])
    stats         = calculate_stats(d["futures_all_trades"])
    live_strategy = futures_data.get("live_strategy", {})
    limit         = min(request.args.get("limit", 500, type=int), 1000)
    return jsonify({
        "trades":        trades[-limit:][::-1],
        "signals":       signals[-50:][::-1],
        "stats":         stats,
        "live_strategy": live_strategy,
    })


@app.route("/api/trades")
def trades():
    d     = load_combined()
    all_t = d["combined_trades"]
    limit = min(request.args.get("limit", 500, type=int), 2000)
    return jsonify(all_t[-limit:][::-1])


@app.route("/api/signals")
def signals():
    d     = load_combined()
    sigs  = d["combined_signals"]
    limit = request.args.get("limit", 30, type=int)
    sigs.sort(key=lambda x: x.get("timestamp", ""))
    seen = {}
    for s in sigs:
        seen[s.get("symbol", "?")] = s
    unique = sorted(seen.values(), key=lambda x: x.get("timestamp", ""), reverse=True)
    return jsonify(unique[:limit])


@app.route("/api/pnl_chart")
def pnl_chart():
    d      = load_combined()
    closed = [t for t in d["combined_trades"] if t.get("status") == "closed"]
    closed.sort(key=lambda t: t.get("close_timestamp", ""))
    cum  = 0.0
    data = []
    for t in closed:
        cum += t.get("pnl", 0.0)
        data.append({
            "timestamp":  t.get("close_timestamp", ""),
            "pnl":        round(t.get("pnl", 0.0), 4),
            "cumulative": round(cum, 4),
            "symbol":     t.get("symbol", ""),
            "mode":       t.get("mode", "spot"),
        })
    return jsonify(data)


@app.route("/api/circuit_breaker")
def circuit_breaker_status():
    p = DATA / "circuit_breaker.json"
    try:
        with open(p) as f:
            cb = json.load(f)
    except Exception:
        return jsonify({"tripped": False, "reason": ""})

    consec        = cb.get("consec_losses", 0)
    pnl_history   = cb.get("pnl_history", {})
    initial_bal   = cb.get("initial_balance") or 0
    peak_bal      = cb.get("peak_balance") or initial_bal

    # Honour disabled_until grace period
    disabled_until = cb.get("disabled_until")
    if disabled_until:
        from datetime import datetime, timezone
        try:
            until_dt = datetime.fromisoformat(disabled_until)
            if datetime.now(LOCAL_TZ) < until_dt:
                remaining = (until_dt - datetime.now(LOCAL_TZ)).total_seconds() / 3600
                return jsonify({"tripped": False, "reason": f"breaker disabled ({remaining:.1f}h remaining)"})
        except Exception:
            pass

    # Read config thresholds
    try:
        import yaml
        mode = _read_env_var("BOT_MODE") or "futures"
        cfg_path = BOT_ROOT / f"config_{mode}.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        risk_cfg = cfg.get("risk", {})
    except Exception:
        risk_cfg = {}

    max_consec   = risk_cfg.get("max_consecutive_losses", 10)
    max_daily    = risk_cfg.get("max_daily_loss_pct", 0.05)
    max_drawdown = risk_cfg.get("max_drawdown_pct", 0.10)
    window_days  = 2

    # Check consecutive losses
    if consec >= max_consec:
        return jsonify({
            "tripped": True,
            "reason": f"Consecutive losses: {consec}/{max_consec}"
        })

    # Check rolling loss
    from datetime import timedelta
    total_loss = sum(
        pnl_history.get(str(date.today() - timedelta(days=i)), 0.0)
        for i in range(window_days)
    )
    threshold = (initial_bal or 1) * max_daily * window_days
    if total_loss < -threshold:
        return jsonify({
            "tripped": True,
            "reason": f"Rolling {window_days}-day loss ${total_loss:.2f} (limit -${threshold:.2f})"
        })

    # Check drawdown
    try:
        state = load_state("futures_state.json") if _read_env_var("BOT_MODE") == "futures" else load_state("state.json")
        balance = state.get("stats", {}).get("balance", 0) or 0
    except Exception:
        balance = 0
    if peak_bal > 0 and balance > 0:
        drawdown = (peak_bal - balance) / peak_bal
        if drawdown > max_drawdown:
            return jsonify({
                "tripped": True,
                "reason": f"Drawdown {drawdown*100:.1f}% from peak ${peak_bal:.0f} (limit {max_drawdown*100:.0f}%)"
            })

    return jsonify({"tripped": False, "reason": ""})


@app.route("/api/circuit_breaker/reset", methods=["POST"])
def circuit_breaker_reset():
    p = DATA / "circuit_breaker.json"
    try:
        with open(p) as f:
            cb = json.load(f)
    except Exception:
        cb = {}
    cb["consec_losses"] = 0
    cb["pnl_history"] = {}
    cb.pop("disabled_until", None)
    try:
        with open(p, "w") as f:
            json.dump(cb, f)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/api/health")
def health():
    d    = load_combined()
    sigs = d["combined_signals"]
    last = None
    if sigs:
        sigs.sort(key=lambda x: x.get("timestamp", ""))
        last = sigs[-1].get("timestamp")
    is_active = False
    if last:
        try:
            last_dt = datetime.fromisoformat(last)
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=LOCAL_TZ)
            is_active = (datetime.now(LOCAL_TZ) - last_dt) < timedelta(minutes=5)
        except (ValueError, TypeError):
            pass
    return jsonify({"status": "active" if is_active else "idle", "last_signal": last})


@app.route("/api/logs")
def logs():
    lines = []
    for log_file in ["spot_bot.log", "futures_bot.log", "bot.log"]:
        p = LOGS_DIR / log_file
        if p.exists():
            try:
                with open(p) as f:
                    lines.extend(f.readlines()[-50:])
            except OSError:
                pass
    lines = sorted(lines)[-100:]
    return jsonify({"lines": lines[::-1]})


@app.route("/api/model_health")
def model_health():
    """v5: rule-based agents + DeepSeek reasoning. No ML model files to check."""
    active_mode = _detect_active_mode()
    result = {}

    # SMC structure agent (rule-based, always active)
    result["smc"] = {
        "loaded": True, "accuracy": 0, "wf_accuracy": 0,
        "trained_at": "rule-based", "status": "active",
        "usage": "Structure: BOS / CHoCH / FVG / liquidity sweeps",
    }

    # Technical agent (rule-based, always active)
    result["technical"] = {
        "loaded": True, "accuracy": 0, "wf_accuracy": 0,
        "trained_at": "rule-based", "status": "active",
        "usage": "Indicators: RSI / MACD / EMA stack / momentum",
    }

    # Regime gate (rules: ADX + breadth)
    result["regime_gate"] = {
        "loaded": True, "accuracy": 0, "wf_accuracy": 0,
        "trained_at": "rule-based", "status": "active",
        "usage": "Regime: CHOPPY / RANGING / WEAK_TREND / STRONG_TREND / CRASH",
    }

    # DeepSeek Actor (V3)
    usage_p = DATA / "deepseek_usage.json"
    deepseek_ok = False
    calls_today = 0
    cost_today = 0.0
    if usage_p.exists():
        try:
            with open(usage_p) as f:
                u = json.load(f)
            from datetime import date
            today = u.get("daily", {}).get(str(date.today()), {})
            calls_today = today.get("calls", 0)
            cost_today  = round(today.get("cost_usd", 0.0), 4)
            deepseek_ok = calls_today > 0
        except (json.JSONDecodeError, OSError):
            pass

    result["deepseek_actor"] = {
        "loaded": deepseek_ok, "accuracy": 0, "wf_accuracy": 0,
        "trained_at": f"{calls_today} calls today",
        "status": "active" if deepseek_ok else "idle",
        "usage": f"DeepSeek V3 reasoning · ${cost_today:.3f} today",
    }

    # Judge / Meta-Judge feedback loop
    db = DATA / "trade_memory.db"
    result["judge_loop"] = {
        "loaded": db.exists(), "accuracy": 0, "wf_accuracy": 0,
        "trained_at": "continuous",
        "status": "active" if db.exists() else "pending",
        "usage": "DeepSeek R1 verbal feedback loop (every 20 trades)",
    }

    return jsonify(result)


def _detect_active_mode():
    """Detect which bot mode is actively running from heartbeat or state file freshness."""
    now = datetime.now(LOCAL_TZ).timestamp()
    for mode in ("futures", "spot"):
        hb = DATA / f"bot_heartbeat_{mode}.json"
        if hb.exists() and (now - hb.stat().st_mtime) < 120:
            return mode
    # Fallback to state file
    futures_state = DATA / "futures_state.json"
    spot_state = DATA / "state.json"
    futures_active = futures_state.exists() and (now - futures_state.stat().st_mtime) < 300
    spot_active = spot_state.exists() and (now - spot_state.stat().st_mtime) < 300
    if futures_active:
        return "futures"
    if spot_active:
        return "spot"
    if (DATA / "futures").is_dir():
        return "futures"
    if (DATA / "spot").is_dir():
        return "spot"
    return ""


@app.route("/api/agent_performance")
def agent_performance():
    p = DATA / "agent_performance.json"
    if p.exists():
        try:
            with open(p) as f:
                data = json.load(f)
            return jsonify({
                a: {
                    "accuracy": round(s["correct"] / s["total"] * 100, 1),
                    "total":    s["total"],
                    "correct":  s["correct"],
                }
                for a, s in data.items() if s.get("total", 0) > 0
            })
        except (json.JSONDecodeError, OSError, ZeroDivisionError):
            pass
    return jsonify({})


@app.route("/api/token_budget")
def token_budget():
    # v5: reads deepseek_usage.json (was Groq's token_budget.json)
    p = DATA / "deepseek_usage.json"
    if p.exists():
        try:
            with open(p) as f:
                data = json.load(f)
            from datetime import date
            today = data.get("daily", {}).get(str(date.today()), {})
            return jsonify({
                "used_today":     today.get("input_tokens", 0) + today.get("output_tokens", 0),
                "limit":          500000,
                "calls_today":    today.get("calls", 0),
                "cost_today_usd": round(today.get("cost_usd", 0.0), 4),
                "cost_total_usd": round(data.get("total_cost_usd", 0.0), 4),
                "provider":       "DeepSeek",
            })
        except (json.JSONDecodeError, OSError):
            pass
    return jsonify({"used_today": 0, "limit": 500000, "provider": "DeepSeek"})


@app.route("/api/exchange_mode")
def exchange_mode():
    env_path = BOT_ROOT / ".env"
    exec_mode = "demo"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("BOT_EXECUTION_MODE="):
                    exec_mode = line.split("=", 1)[1].strip().strip("\"'").lower()
                    break
    return jsonify({
        "execution_mode": exec_mode,
        "training_source": "api.binance.com (real, public)" if exec_mode == "demo" else "api.binance.com (live)",
        "is_demo": exec_mode == "demo",
    })


@app.route("/api/training_data")
def training_data():
    try:
        from data.feed import TrainingDataStore
        manifest = TrainingDataStore.get_manifest()
    except Exception:
        manifest = {"coins": [], "status": "error"}
    import yaml
    cfg_path = BOT_ROOT / "config_futures.yaml"
    min_bars = 3000
    min_coins = 8
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
            training_cfg = cfg.get("training", {})
            min_bars = training_cfg.get("min_bars_per_coin", 3000)
            min_coins = training_cfg.get("min_coins", 8)
        except Exception:
            pass
    coins = manifest.get("coins", [])
    good_coins = sum(1 for c in coins if c.get("bars", 0) >= min_bars)
    total_rows = sum(c.get("bars", 0) for c in coins)
    status = "good" if good_coins >= min_coins else ("degraded" if good_coins >= 2 else "critical")
    return jsonify({
        "coins": coins,
        "total_clean_rows": total_rows,
        "coins_passed": good_coins,
        "min_required_bars": min_bars,
        "min_required_coins": min_coins,
        "status": status,
        "source": "api.binance.com (real, public)",
    })


@app.route("/api/scanner")
def scanner():
    p = DATA / "scanner_cache.json"
    if p.exists():
        try:
            with open(p) as f:
                return jsonify(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return jsonify({"top_coins": [], "last_scan": None})


@app.route("/api/detailed_health")
def detailed_health():
    d       = load_combined()
    trades  = d["combined_trades"]
    signals = d["combined_signals"]

    sig_age = 9999
    if signals:
        signals.sort(key=lambda x: x.get("timestamp", ""))
        try:
            last = datetime.fromisoformat(signals[-1].get("timestamp", ""))
            if last.tzinfo is None:
                last = last.replace(tzinfo=LOCAL_TZ)
            sig_age = int((datetime.now(LOCAL_TZ) - last).total_seconds())
        except (ValueError, TypeError):
            pass

    # Check bot/dash liveness via state file freshness (works in Docker too)
    now = datetime.now(LOCAL_TZ).timestamp()
    bot_count = 0
    dash_count = 0
    for sf, mode in [("state.json", "spot"), ("futures_state.json", "futures")]:
        sp = DATA / sf
        if sp.exists() and (now - sp.stat().st_mtime) < 120:
            bot_count += 1
    # Dashboard is running if we're serving this request
    dash_count = 1
    bots = list(range(bot_count))
    dashes = list(range(dash_count))

    open_t = sum(1 for t in trades if t.get("status") == "open")
    closed = sum(1 for t in trades if t.get("status") == "closed")
    pnl    = sum(t.get("pnl", 0) for t in trades if t.get("status") == "closed")

    # DeepSeek Actor call count from usage tracker
    deepseek_calls_today = 0
    p_usage = DATA / "deepseek_usage.json"
    if p_usage.exists():
        try:
            with open(p_usage) as f:
                usage = json.load(f)
            today = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
            deepseek_calls_today = usage.get("daily", {}).get(today, {}).get("calls", 0)
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    reviews   = 0
    p2        = DATA / "learning_insights.json"
    if p2.exists():
        try:
            with open(p2) as f:
                reviews = json.load(f).get("total_reviews", 0)
        except (json.JSONDecodeError, OSError):
            pass

    token_pct = 0
    p3        = DATA / "token_budget.json"
    if p3.exists():
        try:
            with open(p3) as f:
                tb        = json.load(f)
                token_pct = round(tb.get("used_today", 0) / max(tb.get("limit", 90000), 1) * 100, 1)
        except (json.JSONDecodeError, OSError):
            pass

    spot_open    = sum(1 for t in d["spot"].get("trades", []) if t.get("status") == "open")
    futures_open = sum(1 for t in d["futures"].get("trades", []) if t.get("status") == "open")

    exec_mode = "demo"
    env_path = BOT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.startswith("BOT_EXECUTION_MODE="):
                    exec_mode = line.split("=", 1)[1].strip().strip("\"'").lower()
                    break

    # Profile info from bot state
    profile_name = "BALANCED"
    live_regime = "UNKNOWN"
    for state_name in ("futures_state.json", "spot_state.json"):
        sp = DATA / state_name
        if sp.exists():
            try:
                with open(sp) as f:
                    st = json.load(f)
                live = st.get("live_strategy", {})
                if live:
                    profile_name = live.get("profile", "BALANCED")
                    live_regime = live.get("market_regime", "UNKNOWN")
                    break
            except (json.JSONDecodeError, OSError):
                pass

    return jsonify({
        "bot_running":      len(bots) >= 1,
        "bot_instances":    len(bots),
        "dash_running":     len(dashes) >= 1,
        "signal_age":             sig_age,
        "open_trades":            open_t,
        "closed_trades":          closed,
        "total_pnl":              round(pnl, 4),
        "reviews":                reviews,
        "token_usage_pct":        token_pct,
        "spot_open":              spot_open,
        "futures_open":           futures_open,
        "exchange_mode":          exec_mode,
        "deepseek_actor_active":  True,
        "deepseek_calls_today":   deepseek_calls_today,
        "profile":                profile_name,
        "live_regime":            live_regime,
        "smc_agent":              True,
    })


@app.route("/api/agent_votes")
def api_agent_votes():
    """Per-symbol breakdown of agent buy/sell/agree scores from recent signals."""
    combined = {}
    for fname in ("state.json", "futures_state.json"):
        state = load_state(fname)
        for sig in state.get("signals", []):
            sym = sig.get("symbol", "")
            ind = sig.get("indicators", {})
            if not sym:
                continue
            if sym not in combined:
                combined[sym] = {"buy_score": [], "sell_score": [], "agree": [], "action": []}
            combined[sym]["buy_score"].append(float(ind.get("buy_score", 0)))
            combined[sym]["sell_score"].append(float(ind.get("sell_score", 0)))
            combined[sym]["agree"].append(int(ind.get("agents_agree", 0)))
            combined[sym]["action"].append(sig.get("action", "HOLD"))
    result = {}
    for sym, d in combined.items():
        n = len(d["action"]) or 1
        result[sym] = {
            "avg_buy_score":  round(sum(d["buy_score"]) / n, 4),
            "avg_sell_score": round(sum(d["sell_score"]) / n, 4),
            "avg_agree":      round(sum(d["agree"]) / n, 2),
            "buy_pct":        round(d["action"].count("BUY") / n * 100, 1),
            "sell_pct":       round(d["action"].count("SELL") / n * 100, 1),
            "hold_pct":       round(d["action"].count("HOLD") / n * 100, 1),
            "count":          n,
        }
    return jsonify(result)


@app.route("/api/confidence_heatmap")
def api_confidence_heatmap():
    """Per-symbol mean confidence and action distribution across last 200 signals."""
    combined = {}
    for fname in ("state.json", "futures_state.json"):
        state = load_state(fname)
        for sig in state.get("signals", [])[-200:]:
            sym = sig.get("symbol", "")
            if not sym:
                continue
            if sym not in combined:
                combined[sym] = {"conf": [], "action": []}
            combined[sym]["conf"].append(float(sig.get("confidence", 0)))
            combined[sym]["action"].append(sig.get("action", "HOLD"))
    result = {}
    for sym, d in combined.items():
        n = len(d["action"]) or 1
        result[sym] = {
            "mean_confidence": round(sum(d["conf"]) / n, 4),
            "buy_pct":         round(d["action"].count("BUY") / n * 100, 1),
            "sell_pct":        round(d["action"].count("SELL") / n * 100, 1),
            "hold_pct":        round(d["action"].count("HOLD") / n * 100, 1),
            "count":           n,
        }
    return jsonify(result)


@app.route("/api/liquidation_monitor")
def api_liquidation_monitor():
    """Estimated liquidation price and distance % for each open futures trade."""
    state = load_state("futures_state.json")
    result = []
    for t in state.get("trades", []):
        if t.get("status") != "open":
            continue
        try:
            entry   = float(t.get("price", 0))
            lev     = float(t.get("leverage", 5))
            side    = t.get("side", "long")
            symbol  = t.get("symbol", "")
            if entry <= 0 or lev <= 0:
                continue
            if side == "long":
                liq_price = entry * (1.0 - 1.0 / lev)
                dist_pct  = (entry - liq_price) / entry * 100
            else:
                liq_price = entry * (1.0 + 1.0 / lev)
                dist_pct  = (liq_price - entry) / entry * 100
            result.append({
                "symbol":       symbol,
                "side":         side,
                "entry_price":  entry,
                "leverage":     lev,
                "liq_price":    round(liq_price, 6),
                "dist_pct":     round(dist_pct, 2),
                "trade_id":     t.get("id", ""),
            })
        except Exception:
            continue
    return jsonify(result)


@app.route("/api/execution_queue")
def api_execution_queue():
    """Return pending bot control commands from bot_control.json."""
    p = DATA / "bot_control.json"
    if not p.exists():
        return jsonify({})
    try:
        with open(p) as f:
            return jsonify(json.load(f))
    except Exception:
        return jsonify({})


@app.route("/api/hold_status")
def api_hold_status():
    """Explain why all signals are HOLD / new entries are paused."""
    mode = request.args.get("mode", "futures")
    fname = "futures_state.json" if mode == "futures" else "state.json"
    cfg_path = CFG_FUT if mode == "futures" else CFG_SPOT

    state = load_state(fname)
    ls = state.get("live_strategy", {})
    trades = state.get("trades", [])
    open_trades = [t for t in trades if t.get("status") == "open"]
    n_open = len(open_trades)

    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        max_open = cfg.get("risk", {}).get("max_open_trades", 10)
    except Exception:
        max_open = 10

    reasons = []
    severity = "info"

    # Max positions gate
    if n_open >= max_open:
        reasons.append(f"Max positions reached: {n_open}/{max_open} — waiting for exits")
        severity = "warn"

    # HMM regime
    hmm = ls.get("hmm_regime", "UNKNOWN")
    eff_conf = ls.get("eff_min_conf")
    base_conf = ls.get("base_min_conf")
    if hmm == "CRASH":
        delta = round(eff_conf - base_conf, 2) if eff_conf and base_conf else 0.07
        reasons.append(f"HMM: CRASH regime — min_conf raised +{delta:.2f} (effective: {eff_conf or '?'})")
        severity = "warn"
    elif hmm in ("HIGH_VOLATILITY", "CHOPPY"):
        reasons.append(f"HMM: {hmm} — reduced sizing, elevated confidence threshold")

    # Market regime
    regime      = ls.get("market_regime", "UNKNOWN")
    breadth     = ls.get("breadth")       # fraction of coins bullish (0–1)
    bear_breadth = ls.get("bear_breadth") # fraction of coins bearish (0–1)

    if regime == "STRONG_TREND":
        parts = []
        if bear_breadth is not None and bear_breadth > 0.50:
            needed = round(min(0.70, 0.42 + min(0.15, (bear_breadth - 0.50) * 1.5)), 2)
            parts.append(f"longs need conf≥{needed} (bearish breadth {bear_breadth:.0%})")
        if breadth is not None and breadth > 0.70:
            parts.append(f"shorts hard-blocked (bullish breadth {breadth:.0%})")
        elif breadth is not None and breadth > 0.50:
            parts.append(f"shorts penalised (bullish breadth {breadth:.0%})")
        if parts:
            reasons.append("STRONG_TREND: " + " | ".join(parts))
    elif regime == "WEAK_TREND":
        parts = []
        if bear_breadth is not None and bear_breadth > 0.50:
            needed = round(min(0.70, 0.42 + min(0.15, (bear_breadth - 0.50) * 1.5)), 2)
            parts.append(f"longs need conf≥{needed} (bearish breadth {bear_breadth:.0%})")
        elif breadth is not None and breadth > 0.50:
            needed = round(min(0.70, 0.42 + min(0.15, (breadth - 0.50) * 1.5)), 2)
            parts.append(f"shorts need conf≥{needed} (bullish breadth {breadth:.0%})")
        else:
            parts.append("shorts blocked, longs need higher confluence")
        reasons.append("WEAK_TREND: " + " | ".join(parts))
    elif regime in ("CHOPPY", "CRASH"):
        reasons.append(f"Market: {regime} — all new entries blocked by regime gate")
        severity = "warn"

    # Signal quality — check if ML models are flat
    signals = state.get("signals", [])
    recent = signals[-30:] if len(signals) >= 30 else signals
    if recent:
        ml_only = sum(1 for s in recent if s.get("source") == "ml_only")
        hold_ct = sum(1 for s in recent if s.get("action") == "HOLD")
        avg_conf = sum(s.get("confidence", 0) for s in recent) / len(recent)
        if ml_only >= len(recent) * 0.75:
            reasons.append(f"ML models outputting flat signal (avg conf {avg_conf:.2f}) — full ensemble skipped")
        elif hold_ct >= len(recent) * 0.85:
            reasons.append(f"{hold_ct}/{len(recent)} recent signals are HOLD (avg conf {avg_conf:.2f})")

    # Coordinator cache stale note
    if recent and all(s.get("source") == "ml_only" for s in recent[-5:]):
        reasons.append("Signal cache active — coordinator refreshes every 30 min")

    holding = len(reasons) > 0
    return jsonify({
        "holding":      holding,
        "severity":     severity if holding else "ok",
        "reasons":      reasons,
        "n_open":       n_open,
        "max_open":     max_open,
        "hmm_regime":   hmm,
        "market_regime": regime,
        "eff_min_conf": eff_conf,
    })


@app.route("/api/regime_detail")
def api_regime_detail():
    """Full regime context dict from live_strategy in state files."""
    result = {}
    for fname, mode in (("state.json", "spot"), ("futures_state.json", "futures")):
        state = load_state(fname)
        ls = state.get("live_strategy", {})
        result[mode] = {
            "regime":        ls.get("market_regime", "UNKNOWN"),
            "hmm_regime":    ls.get("hmm_regime", "UNKNOWN"),
            "eff_min_conf":  ls.get("eff_min_conf"),
            "eff_size_mult": ls.get("eff_size_mult"),
            "profile":       ls.get("profile"),
            "updated_at":    ls.get("updated_at"),
        }
    return jsonify(result)


ENV_PATH = Path.home() / "cryptobot_v3" / ".env"
BOT_ROOT_PATH = Path.home() / "cryptobot_v3"


def _read_env_var(key):
    if ENV_PATH.exists():
        with open(ENV_PATH) as f:
            for line in f:
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    return "spot"


def _write_env_var(key, value):
    try:
        ENV_PATH.parent.mkdir(parents=True, exist_ok=True)
        if ENV_PATH.exists():
            with open(ENV_PATH) as f:
                lines = f.readlines()
        else:
            lines = []
        new_lines = []
        found = False
        for line in lines:
            if line.startswith(key + "="):
                new_lines.append(f'{key}="{value}"\n')
                found = True
            else:
                new_lines.append(line)
        if not found:
            new_lines.append(f'{key}="{value}"\n')
        with open(ENV_PATH, "w") as f:
            f.writelines(new_lines)
        return True
    except (OSError, IOError):
        return False


def _is_bot_running(mode: str) -> bool:
    now = datetime.now(LOCAL_TZ).timestamp()
    hb = DATA / f"bot_heartbeat_{mode}.json"
    if hb.exists() and (now - hb.stat().st_mtime) < 120:
        return True
    return False


def _write_bot_command(mode: str, command: str):
    """Write a command file to the shared data volume. The bot reads this each cycle."""
    p = DATA / "bot_control.json"
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump({
                "command": command,
                "mode": mode,
                "timestamp": datetime.now(LOCAL_TZ).isoformat(),
            }, f)
        return True
    except OSError:
        return False


def _get_running_mode() -> str:
    if _is_bot_running("futures"):
        return "futures"
    if _is_bot_running("spot"):
        return "spot"
    return "none"


def _trading_paused() -> bool:
    p = DATA / "trading_paused.json"
    if p.exists():
        try:
            with open(p) as f:
                return json.load(f).get("paused", False)
        except (json.JSONDecodeError, OSError):
            pass
    return False


def _has_open_trades(mode: str = None) -> bool:
    if mode is None:
        mode = _read_env_var("BOT_MODE")
    state_file = DATA / ("state.json" if mode == "spot" else f"{mode}_state.json")
    if not state_file.exists():
        return False
    try:
        with open(state_file) as f:
            data = json.load(f)
        opens = [t for t in data.get("trades", []) if t.get("status") == "open"]
        return len(opens) > 0
    except Exception:
        return False


def _systemctl_cmd(mode: str, action: str):
    """Run systemctl action on cryptobot-{mode} service. Returns (success: bool, msg: str)."""
    svc = f"cryptobot-{mode}"
    try:
        r = subprocess.run(["systemctl", action, svc], capture_output=True, text=True, timeout=15)
        if r.returncode == 0:
            return True, f"systemctl {action} {svc} OK"
        return False, r.stderr.strip() or r.stdout.strip() or f"exit {r.returncode}"
    except subprocess.TimeoutExpired:
        return False, f"systemctl {action} {svc} timed out"
    except Exception as e:
        return False, str(e)


# ── Bot Control Endpoints ────────────────────────────────────────────────────

@app.route("/api/bot_status", methods=["GET"])
def bot_status():
    mode = _get_running_mode()
    running = mode != "none"
    paused = _trading_paused()
    has_open = _has_open_trades(mode) if running else False
    has_open_fut = _has_open_trades("futures")
    return jsonify({
        "mode": mode if running else (_read_env_var("BOT_MODE")),
        "running": running,
        "trading_paused": paused,
        "has_open_trades": has_open,
        "has_open_futures": has_open_fut,
    })


@app.route("/api/start_bot", methods=["POST"])
def start_bot():
    err, code = _check_auth()
    if err:
        return err, code

    data = request.get_json(force=True) or {}
    mode = data.get("mode", "futures")
    if mode not in ("spot", "futures"):
        return jsonify({"error": "Invalid mode. Choose 'spot' or 'futures'"}), 400

    # Stop the OTHER mode's bot first to prevent dual-running
    other_mode = "spot" if mode == "futures" else "futures"
    if _is_bot_running(other_mode):
        print(f"[dashboard] START_BOT: stopping {other_mode} before starting {mode}")
        _systemctl_cmd(other_mode, "stop")
        _write_bot_command(other_mode, "stop")
        import time as _time
        _time.sleep(2)

    _write_env_var("BOT_MODE", mode)
    print(f"[dashboard] START_BOT: {mode}")

    success, msg = _systemctl_cmd(mode, "start")
    if not success:
        return jsonify({"error": f"Failed to start bot: {msg}"}), 502
    return jsonify({"mode": mode, "started": True})


@app.route("/api/stop_bot", methods=["POST"])
def stop_bot():
    err, code = _check_auth()
    if err:
        return err, code

    mode = _get_running_mode()
    if mode == "none":
        return jsonify({"error": "No bot is running"}), 409

    print(f"[dashboard] STOP_BOT: {mode}")
    success, msg = _systemctl_cmd(mode, "stop")
    if not success:
        _write_bot_command(mode, "stop")
        print(f"[dashboard] STOP_BOT: systemctl failed ({msg}), wrote command file for {mode}")
    return jsonify({"stopped": True, "mode": mode, "method": "systemctl" if success else "command_file"})


@app.route("/api/switch_mode", methods=["POST"])
def switch_mode():
    err, code = _check_auth()
    if err:
        return err, code

    old_mode = _get_running_mode()

    # Only block switching FROM futures (leveraged), spot trades are safe with exchange SL/TP
    if old_mode == "futures" and _has_open_trades("futures"):
        return jsonify({
            "error": "Cannot switch from futures while futures trades are open"
        }), 409

    data = request.get_json(force=True) or {}
    new_mode = data.get("mode", "spot")
    if new_mode not in ("spot", "futures"):
        return jsonify({"error": "Invalid mode. Choose 'spot' or 'futures'"}), 400

    print(f"[dashboard] SWITCH_MODE: {old_mode} → {new_mode}")

    _write_env_var("BOT_MODE", new_mode)

    if old_mode != "none":
        success, _ = _systemctl_cmd(old_mode, "stop")
        if not success:
            _write_bot_command(old_mode, "stop")

    success, msg = _systemctl_cmd(new_mode, "start")
    if not success:
        # Rollback: restart old service so bot doesn't stay dead
        if old_mode != "none":
            print(f"[dashboard] SWITCH_MODE: failed to start {new_mode}, rolling back to {old_mode}")
            _systemctl_cmd(old_mode, "start")
        return jsonify({"error": f"Failed to start {new_mode} bot: {msg}"}), 502

    return jsonify({"mode": new_mode, "switched": True})


@app.route("/api/stop_trading", methods=["POST"])
def stop_trading():
    err, code = _check_auth()
    if err:
        return err, code

    p = DATA / "trading_paused.json"
    p.parent.mkdir(exist_ok=True)
    with open(p, "w") as f:
        json.dump({"paused": True, "timestamp": datetime.now(LOCAL_TZ).isoformat()}, f)
    return jsonify({"status": "paused", "trading_paused": True})


@app.route("/api/start_trading", methods=["POST"])
def start_trading():
    err, code = _check_auth()
    if err:
        return err, code

    p = DATA / "trading_paused.json"
    p.parent.mkdir(exist_ok=True)
    with open(p, "w") as f:
        json.dump({"paused": False, "timestamp": datetime.now(LOCAL_TZ).isoformat()}, f)
    return jsonify({"status": "running", "trading_paused": False})


@app.route("/api/close_all_positions", methods=["POST"])
def close_all_positions():
    err, code = _check_auth()
    if err:
        return err, code

    p = DATA / "close_all_positions.json"
    p.parent.mkdir(exist_ok=True)

    count = 0
    state_path = DATA / "futures_state.json"
    try:
        if state_path.exists():
            with open(state_path) as f:
                state = json.load(f)
            count = sum(1 for t in state.get("trades", []) if t.get("status") == "open")
    except Exception:
        pass

    with open(p, "w") as f:
        json.dump({"close_all": True, "timestamp": datetime.now(LOCAL_TZ).isoformat()}, f)
    return jsonify({"status": "queued", "count": count})


@app.route("/api/strategy_config", methods=["GET"])
def get_strategy_config():
    base = {
        "forward_bars": 1, "timeframe": "1h",
        "min_confidence": 0.40, "min_votes": 1,
        "htf_filter_mode": "strict",
        "training_min_bars": 3000, "training_min_coins": 8,
        "dynamic_min_conf": True,
    }
    for cfg_path in (CFG_FUT, CFG_SPOT):
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
                strat = cfg.get("strategy", {})
                ml    = cfg.get("ml", {})
                train = cfg.get("training", {})
                base = {
                    "forward_bars":       ml.get("forward_bars", 1),
                    "timeframe":          ml.get("timeframe", "1h"),
                    "min_confidence":     strat.get("min_confidence", 0.40),
                    "min_votes":          strat.get("min_votes", 1),
                    "htf_filter_mode":    strat.get("htf_filter_mode", "strict"),
                    "training_min_bars":  train.get("min_bars_per_coin", 3000),
                    "training_min_coins": train.get("min_coins", 8),
                    "dynamic_min_conf":   True,
                }
            except (yaml.YAMLError, OSError):
                pass

    # Inject live effective values from bot state
    for state_name in ("futures_state.json", "spot_state.json"):
        sp = DATA / state_name
        if sp.exists():
            try:
                with open(sp) as f:
                    st = json.load(f)
                live = st.get("live_strategy", {})
                if live:
                    base["live"] = {
                        "eff_min_conf":    live.get("eff_min_conf", base["min_confidence"]),
                        "eff_size_mult":   live.get("eff_size_mult", 1.0),
                        "market_regime":   live.get("market_regime", "UNKNOWN"),
                        "hmm_regime":      live.get("hmm_regime", "UNKNOWN"),
                        "profile":         live.get("profile", "BALANCED"),
                        "updated_at":      live.get("updated_at", ""),
                    }
                    break
            except (json.JSONDecodeError, OSError):
                pass

    return jsonify(base)


@app.route("/api/strategy_config", methods=["POST"])
def set_strategy_config():
    err, code = _check_auth()
    if err:
        return err, code

    data = request.get_json(force=True) or {}

    # Input validation
    try:
        if "forward_bars" in data:
            v = int(data["forward_bars"])
            if not (1 <= v <= 10):
                return jsonify({"error": "forward_bars must be 1–10"}), 400
            data["forward_bars"] = v

        if "timeframe" in data:
            if str(data["timeframe"]) not in VALID_TFS:
                return jsonify({"error": f"timeframe must be one of {VALID_TFS}"}), 400
            data["timeframe"] = str(data["timeframe"])

        if "min_confidence" in data:
            v = float(data["min_confidence"])
            if not (0.30 <= v <= 0.95):
                return jsonify({"error": "min_confidence must be 0.30–0.95"}), 400
            data["min_confidence"] = round(v, 4)

        if "min_votes" in data:
            v = int(data["min_votes"])
            if not (1 <= v <= 2):
                return jsonify({"error": "min_votes must be 1–2"}), 400
            data["min_votes"] = v
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid value: {e}"}), 400

    saved = {}
    for cfg_path in (CFG_SPOT, CFG_FUT):
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
                if "forward_bars" in data:
                    cfg.setdefault("ml", {})["forward_bars"] = data["forward_bars"]
                if "timeframe" in data:
                    cfg.setdefault("ml", {})["timeframe"] = data["timeframe"]
                if "min_confidence" in data:
                    cfg.setdefault("strategy", {})["min_confidence"] = data["min_confidence"]
                if "min_votes" in data:
                    cfg.setdefault("strategy", {})["min_votes"] = data["min_votes"]
                _write_config_safe(cfg_path, cfg)
                # Return what was actually written
                saved = {
                    "forward_bars":    cfg.get("ml", {}).get("forward_bars", 1),
                    "timeframe":       cfg.get("ml", {}).get("timeframe", "1h"),
                    "min_confidence":  cfg.get("strategy", {}).get("min_confidence", 0.40),
                    "min_votes":       cfg.get("strategy", {}).get("min_votes", 1),
                    "htf_filter_mode": cfg.get("strategy", {}).get("htf_filter_mode", "strict"),
                }
            except (yaml.YAMLError, OSError) as e:
                return jsonify({"error": f"Config write failed: {e}"}), 500

    return jsonify({"status": "ok", "config": saved})


# ── Trading Profile ──────────────────────────────────────────────────────────

VALID_PROFILES = ("STRICT", "BALANCED", "AGGRESSIVE", "CONFLUENCE")


@app.route("/api/trading_profile", methods=["GET"])
def get_trading_profile():
    profile_name = "BALANCED"
    for cfg_path in (CFG_FUT, CFG_SPOT):
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
                profile_name = cfg.get("strategy", {}).get("trading_profile", "BALANCED")
            except (yaml.YAMLError, OSError):
                pass

    # Resolve full profile from engine module
    profile = {"name": profile_name}
    try:
        from engine.profiles import TradingProfile
        p = TradingProfile.load(profile_name)
        # v5: only the fields the engine actually enforces (see engine/profiles.py)
        profile = {
            "name": p.name, "min_confidence": p.min_confidence,
            "min_agent_agreement": p.min_agent_agreement,
            "net_score_threshold": p.net_score_threshold,
            "smc_sub_checks_min": p.smc_sub_checks_min,
            "smc_liquidity_sweep_pct": p.smc_liquidity_sweep_pct,
            "smc_bos_body_pct": p.smc_bos_body_pct,
            "smc_volume_spike_ratio": p.smc_volume_spike_ratio,
            "smc_pattern_completion": p.smc_pattern_completion,
            "htf_filter_mode": p.htf_filter_mode,
            "btc_momentum_filter": p.btc_momentum_filter,
            "stop_loss_atr_mult": p.stop_loss_atr_mult,
            "take_profit_atr_mult": p.take_profit_atr_mult,
            "tp1_fraction": p.tp1_fraction,
            "tp1_r_mult": p.tp1_r_mult,
            "trail_atr_mult": p.trail_atr_mult,
            "early_exit_enabled": p.early_exit_enabled,
            "dynamic_tp_enabled": p.dynamic_tp_enabled,
            "use_confluence_scoring": p.use_confluence_scoring,
        }
    except Exception:
        pass

    # Inject live effective values
    for state_name in ("futures_state.json", "spot_state.json"):
        sp = DATA / state_name
        if sp.exists():
            try:
                with open(sp) as f:
                    st = json.load(f)
                live = st.get("live_strategy", {})
                if live:
                    profile["live"] = {
                        "eff_min_conf": live.get("eff_min_conf", profile["min_confidence"]),
                        "eff_size_mult": live.get("eff_size_mult", 1.0),
                        "market_regime": live.get("market_regime", "UNKNOWN"),
                        "hmm_regime": live.get("hmm_regime", "UNKNOWN"),
                        "updated_at": live.get("updated_at", ""),
                    }
                    break
            except (json.JSONDecodeError, OSError):
                pass

    return jsonify(profile)


@app.route("/api/trading_profile", methods=["POST"])
def set_trading_profile():
    err, code = _check_auth()
    if err:
        return err, code

    data = request.get_json(force=True) or {}
    profile_name = str(data.get("profile", "")).upper()
    if profile_name not in VALID_PROFILES:
        return jsonify({"error": f"Invalid profile. Choose: {', '.join(VALID_PROFILES)}"}), 400

    saved = False
    for cfg_path in (CFG_SPOT, CFG_FUT):
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
                cfg.setdefault("strategy", {})["trading_profile"] = profile_name
                _write_config_safe(cfg_path, cfg)
                saved = True
            except (yaml.YAMLError, OSError) as e:
                return jsonify({"error": f"Config write failed: {e}"}), 500

    if not saved:
        return jsonify({"error": "No config files found"}), 500

    return jsonify({"status": "ok", "profile": profile_name, "message": f"Switched to {profile_name} — takes effect within 30 s"})


@app.route("/api/retrain", methods=["POST"])
def request_retrain():
    err, code = _check_auth()
    if err:
        return err, code

    # Rate limit: block if any retrain is already running
    for mode in ("futures", "spot"):
        sp = DATA / f"retrain_status_{mode}.json"
        if sp.exists():
            try:
                with open(sp) as f:
                    st = json.load(f)
                if st.get("status") == "running":
                    return jsonify({"error": "retrain already running", "status": "running", "mode": mode}), 429
            except (json.JSONDecodeError, OSError):
                pass

    data = request.get_json(force=True, silent=True) or {}
    pkg = {
        "requested": True,
        "timestamp": datetime.now(LOCAL_TZ).isoformat(),
        "source": data.get("source", "dashboard"),
        "reason": data.get("reason", "manual"),
    }

    # Write to both modes (spot + futures run in parallel via Docker)
    written = []
    for mode in ("futures", "spot"):
        p = DATA / f"retrain_requested_{mode}.json"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp.json")
            with open(tmp, "w") as f:
                json.dump(pkg, f)
            tmp.replace(p)
            written.append(mode)
        except OSError as e:
            return jsonify({"error": f"Failed to write retrain request for {mode}: {e}"}), 500

    return jsonify({"status": "requested", "modes": written, "message": f"Retrain signal sent to {', '.join(written)}"})


@app.route("/api/retrain_status", methods=["GET"])
def get_retrain_status():
    # Check currently running mode first
    running_mode = _get_running_mode()
    check_modes = []
    if running_mode != "none":
        check_modes.append(running_mode)
    for mode in ("futures", "spot"):
        if mode not in check_modes:
            check_modes.append(mode)
    for mode in check_modes:
        p = DATA / f"retrain_status_{mode}.json"
        if p.exists():
            try:
                with open(p) as f:
                    st = json.load(f)
                if st.get("status") in ("running", "completed", "error"):
                    st["mode"] = mode
                    return jsonify(st)
            except (json.JSONDecodeError, OSError):
                pass
    return jsonify({"status": "idle"})


@app.route("/")
def index():
    from flask import make_response
    response = make_response(send_from_directory("static", "index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"]        = "no-cache"
    response.headers["Expires"]       = "0"
    return response


@app.route("/favicon.ico")
def favicon():
    return "", 204


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    if API_KEY:
        print(f"Auth enabled — set X-Dashboard-Key header or store key in localStorage('dashboard_key')")
    print(f"Dashboard v4: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
