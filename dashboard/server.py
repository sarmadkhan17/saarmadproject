"""
Dashboard Server v3
Same as v2 but with separate Spot & Futures sections
"""

import json
import os
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
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

app  = Flask(__name__, static_folder="static")
CORS(app)
DATA = DATA_DIR


def load_state(filename="state.json"):
    p = DATA / filename
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"trades":[], "signals":[], "stats":{
        "total_trades":0,"wins":0,"losses":0,"total_pnl":0.0}}


def load_combined():
    spot    = load_state("state.json")
    futures = load_state("futures_state.json")
    return {
        "spot":    spot,
        "futures": futures,
        "combined_trades":  spot.get("trades",[]) + futures.get("trades",[]),
        "combined_signals": spot.get("signals",[]) + futures.get("signals",[]),
    }


def calculate_stats(trades):
    wins   = sum(1 for t in trades if t.get("status")=="closed" and t.get("pnl",0)>0)
    losses = sum(1 for t in trades if t.get("status")=="closed" and t.get("pnl",0)<=0)
    total  = wins + losses
    pnl    = sum(t.get("pnl",0) for t in trades if t.get("status")=="closed")
    open_t = len([t for t in trades if t.get("status")=="open"])
    return {
        "total_trades": len(trades),
        "open_trades":  open_t,
        "wins":         wins,
        "losses":       losses,
        "win_rate":     round(wins/total*100, 1) if total else 0.0,
        "total_pnl":    round(pnl, 4),
    }


@app.route("/api/stats")
def stats():
    d      = load_combined()
    result = calculate_stats(d["combined_trades"])

    # Add balance and live PnL from futures state
    fut_stats  = d["futures"].get("stats", {})
    spot_stats_data = d["spot"].get("stats", {})

    balance = fut_stats.get("balance", 0) or spot_stats_data.get("balance", 0)
    result["balance"]        = round(float(balance), 2) if balance else 0
    live_pnl = fut_stats.get("total_live_pnl", 0) or 0
    result["total_live_pnl"] = round(float(live_pnl), 4)
    result["last_sync"]      = fut_stats.get("last_sync", "never")

    # Today PnL
    today  = datetime.utcnow().strftime("%Y-%m-%d")
    trades_list = d["combined_trades"]
    today_trades = [t for t in trades_list
                    if t.get("status") == "closed"
                    and t.get("close_timestamp", "")[:10] == today]
    result["today_pnl"]    = round(sum(t.get("pnl", 0) for t in today_trades), 4)
    result["today_trades"] = len(today_trades)
    result["today_wins"]   = sum(1 for t in today_trades if t.get("pnl", 0) > 0)

    # Trading paused state
    p = DATA / "trading_paused.json"
    if p.exists():
        with open(p) as f:
            pause_data = json.load(f)
        result["trading_paused"] = pause_data.get("paused", False)
    else:
        result["trading_paused"] = False

    return jsonify(result)


@app.route("/api/spot_stats")
def spot_stats():
    d = load_combined()
    return jsonify(calculate_stats(d["spot"].get("trades", [])))


@app.route("/api/futures_stats")
def futures_stats():
    d = load_combined()
    return jsonify(calculate_stats(d["futures"].get("trades", [])))


@app.route("/api/spot_trades")
def spot_trades():
    d     = load_combined()
    all_t = d["spot"].get("trades", [])
    limit = request.args.get("limit", 30, type=int)
    return jsonify(all_t[-limit:][::-1])


@app.route("/api/futures_trades")
def futures_trades():
    d     = load_combined()
    all_t = d["futures"].get("trades", [])
    limit = request.args.get("limit", 30, type=int)
    return jsonify(all_t[-limit:][::-1])


@app.route("/api/trades")
def trades():
    d     = load_combined()
    all_t = d["combined_trades"]
    limit = request.args.get("limit", 50, type=int)
    return jsonify(all_t[-limit:][::-1])


@app.route("/api/signals")
def signals():
    d     = load_combined()
    sigs  = d["combined_signals"]
    limit = request.args.get("limit", 30, type=int)
    sigs.sort(key=lambda x: x.get("timestamp",""))
    seen = {}
    for s in sigs:
        seen[s.get("symbol","?")] = s
    unique = sorted(seen.values(), key=lambda x: x.get("timestamp",""), reverse=True)
    return jsonify(unique[:limit])


@app.route("/api/pnl_chart")
def pnl_chart():
    d      = load_combined()
    closed = [t for t in d["combined_trades"] if t.get("status")=="closed"]
    closed.sort(key=lambda t: t.get("close_timestamp",""))
    cum  = 0.0
    data = []
    for t in closed:
        cum += t.get("pnl", 0.0)
        data.append({
            "timestamp":  t.get("close_timestamp",""),
            "pnl":        round(t.get("pnl",0.0), 4),
            "cumulative": round(cum, 4),
            "symbol":     t.get("symbol",""),
            "mode":       t.get("mode","spot"),
        })
    return jsonify(data)


@app.route("/api/health")
def health():
    d    = load_combined()
    sigs = d["combined_signals"]
    last = None
    if sigs:
        sigs.sort(key=lambda x: x.get("timestamp",""))
        last = sigs[-1]["timestamp"]
    is_active = False
    if last:
        last_dt   = datetime.fromisoformat(last)
        is_active = (datetime.utcnow() - last_dt) < timedelta(minutes=5)
    return jsonify({"status": "active" if is_active else "idle", "last_signal": last})


@app.route("/api/logs")
def logs():
    lines = []
    for log_file in ["spot_bot.log", "futures_bot.log", "bot.log"]:
        p = LOGS_DIR / log_file
        if p.exists():
            with open(p) as f:
                lines.extend(f.readlines()[-50:])
    lines = sorted(lines)[-100:]
    return jsonify({"lines": lines[::-1]})


@app.route("/api/model_health")
def model_health():
    models = {
        "rf":   ("rf_model.pkl",    "rf_meta.json"),
        "lgbm": ("lgbm_model.pkl",  "lgbm_meta.json"),
        "lstm": ("lstm_model.keras","lstm_meta.json"),
    }
    result = {}
    for name, (mf, meta_f) in models.items():
        meta = {}
        if (DATA / meta_f).exists():
            with open(DATA / meta_f) as f:
                meta = json.load(f)
        result[name] = {
            "loaded":      (DATA / mf).exists(),
            "accuracy":    meta.get("accuracy", 0),
            "wf_accuracy": meta.get("wf_accuracy", 0),
            "trained_at":  meta.get("trained_at", "never"),
            "epochs":      meta.get("epochs", 0),
        }
    return jsonify(result)


@app.route("/api/agent_performance")
def agent_performance():
    p = DATA / "agent_performance.json"
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        return jsonify({
            a: {"accuracy": round(s["correct"]/s["total"]*100,1),
                "total": s["total"], "correct": s["correct"]}
            for a, s in data.items() if s.get("total",0) > 0
        })
    return jsonify({})


@app.route("/api/token_budget")
def token_budget():
    p = DATA / "token_budget.json"
    if p.exists():
        with open(p) as f:
            return jsonify(json.load(f))
    return jsonify({"used_today": 0, "limit": 90000})


@app.route("/api/scanner")
def scanner():
    p = DATA / "scanner_cache.json"
    if p.exists():
        with open(p) as f:
            return jsonify(json.load(f))
    return jsonify({"top_coins": [], "last_scan": None})


@app.route("/api/detailed_health")
def detailed_health():
    d       = load_combined()
    trades  = d["combined_trades"]
    signals = d["combined_signals"]

    sig_age = 9999
    if signals:
        signals.sort(key=lambda x: x.get("timestamp",""))
        last    = datetime.fromisoformat(signals[-1]["timestamp"])
        sig_age = int((datetime.utcnow() - last).total_seconds())

    result = subprocess.run(["ps","aux"], capture_output=True, text=True)
    lines  = result.stdout.split("\n")
    bots   = [l for l in lines if ("launcher.py" in l or "spot_bot.py" in l or "futures_bot.py" in l)
              and "grep" not in l and "SCREEN" not in l]
    dashes = [l for l in lines if "server.py" in l and "grep" not in l and "SCREEN" not in l]

    open_t = len([t for t in trades if t.get("status")=="open"])
    closed = len([t for t in trades if t.get("status")=="closed"])
    pnl    = sum(t.get("pnl",0) for t in trades if t.get("status")=="closed")

    models_ok = all([
        (DATA / "rf_model.pkl").exists(),
        (DATA / "lgbm_model.pkl").exists(),
        (DATA / "lstm_model.keras").exists(),
    ])

    reviews = 0
    p2      = DATA / "learning_insights.json"
    if p2.exists():
        with open(p2) as f:
            reviews = json.load(f).get("total_reviews", 0)

    token_pct = 0
    p3        = DATA / "token_budget.json"
    if p3.exists():
        with open(p3) as f:
            tb        = json.load(f)
            token_pct = round(tb.get("used_today",0) / tb.get("limit",90000) * 100, 1)

    spot_open    = len([t for t in d["spot"].get("trades", []) if t.get("status")=="open"])
    futures_open = len([t for t in d["futures"].get("trades", []) if t.get("status")=="open"])

    return jsonify({
        "bot_running":    len(bots) >= 1,
        "bot_instances":  len(bots),
        "dash_running":   len(dashes) >= 1,
        "rf_model":       (DATA / "rf_model.pkl").exists(),
        "xgb_model":      (DATA / "lgbm_model.pkl").exists(),
        "lstm_model":     (DATA / "lstm_model.keras").exists(),
        "models_ok":      models_ok,
        "signal_age":     sig_age,
        "open_trades":    open_t,
        "closed_trades":  closed,
        "total_pnl":      round(pnl, 4),
        "reviews":        reviews,
        "token_usage_pct":token_pct,
        "spot_open":      spot_open,
        "futures_open":   futures_open,
    })


@app.route("/api/htf_mode", methods=["GET"])
def get_htf_mode():
    mode = "strict"
    if CFG_SPOT.exists():
        with open(CFG_SPOT) as f:
            cfg = yaml.safe_load(f)
        mode = cfg.get("strategy", {}).get("htf_filter_mode", "strict")
    return jsonify({"htf_filter_mode": mode})


@app.route("/api/htf_mode", methods=["POST"])
def set_htf_mode():
    data = request.get_json(force=True)
    mode = data.get("mode", "strict")
    if mode not in HTF_MODES:
        return jsonify({"error": f"Invalid mode. Choose from: {HTF_MODES}"}), 400
    for cfg_path in (CFG_SPOT, CFG_FUT):
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            cfg.setdefault("strategy", {})["htf_filter_mode"] = mode
            with open(cfg_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)
    return jsonify({"htf_filter_mode": mode})


@app.route("/api/stop_trading", methods=["POST"])
def stop_trading():
    p = DATA / "trading_paused.json"
    with open(p, "w") as f:
        json.dump({"paused": True, "timestamp": datetime.utcnow().isoformat()}, f)
    return jsonify({"status": "paused"})


@app.route("/api/start_trading", methods=["POST"])
def start_trading():
    p = DATA / "trading_paused.json"
    with open(p, "w") as f:
        json.dump({"paused": False, "timestamp": datetime.utcnow().isoformat()}, f)
    return jsonify({"status": "running"})


@app.route("/")
def index():
    from flask import make_response
    response = make_response(send_from_directory("static", "index.html"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    print(f"Dashboard v3: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
