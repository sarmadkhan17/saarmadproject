"""
Dashboard Server v3
Same as v2 but with separate Spot & Futures sections
"""

import fcntl
import json
import os
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone
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


def load_combined():
    spot    = load_state("state.json")
    futures = load_state("futures_state.json")
    return {
        "spot":    spot,
        "futures": futures,
        "combined_trades":  spot.get("trades", []) + futures.get("trades", []),
        "combined_signals": spot.get("signals", []) + futures.get("signals", []),
    }


# ── Config write with file locking ────────────────────────────────────────────

def _write_config_safe(cfg_path: Path, cfg: dict):
    """Atomic read-modify-write on a YAML config with exclusive file lock."""
    cfg_path.parent.mkdir(exist_ok=True)
    # Write to temp file then rename for atomicity
    tmp = cfg_path.with_suffix(".tmp.yaml")
    with open(tmp, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)
        fcntl.flock(f, fcntl.LOCK_UN)
    tmp.replace(cfg_path)


# ── Stats ─────────────────────────────────────────────────────────────────────

def calculate_stats(trades):
    closed = [t for t in trades if t.get("status") == "closed"]
    wins   = sum(1 for t in closed if t.get("pnl", 0) > 0)
    losses = sum(1 for t in closed if t.get("pnl", 0) < 0)  # break-even not a loss
    total  = wins + losses
    pnl    = sum(t.get("pnl", 0) for t in closed)
    open_t = sum(1 for t in trades if t.get("status") == "open")
    return {
        "total_trades": len(trades),
        "open_trades":  open_t,
        "wins":         wins,
        "losses":       losses,
        "win_rate":     round(wins / total * 100, 1) if total else 0.0,
        "total_pnl":    round(pnl, 4),
    }


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

    today        = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_trades = [
        t for t in d["combined_trades"]
        if t.get("status") == "closed" and t.get("close_timestamp", "")[:10] == today
    ]
    result["today_pnl"]    = round(sum(t.get("pnl", 0) for t in today_trades), 4)
    result["today_trades"] = len(today_trades)
    result["today_wins"]   = sum(1 for t in today_trades if t.get("pnl", 0) > 0)

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
    return jsonify(calculate_stats(d["spot"].get("trades", [])))


@app.route("/api/futures_stats")
def futures_stats():
    d = load_combined()
    return jsonify(calculate_stats(d["futures"].get("trades", [])))


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
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            is_active = (datetime.now(timezone.utc) - last_dt) < timedelta(minutes=5)
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
    models = {
        "rf":   ("rf_model.pkl",    "rf_meta.json",   0.35),
        "lgbm": ("lgbm_model.pkl",  "lgbm_meta.json", 0.50),
        "lstm": ("lstm_model.keras","lstm_meta.json", 0.00),
        "tft":  ("tft_model.pt",    "tft_meta.json",  0.15),
    }
    result = {}
    for name, (mf, meta_f, weight) in models.items():
        best_meta = {}
        loaded    = False
        for subdir in ["spot", "futures", ""]:
            base  = DATA / subdir if subdir else DATA
            mpath = base / mf
            if mpath.exists():
                loaded = True
                meta_p = base / meta_f
                if meta_p.exists():
                    try:
                        with open(meta_p) as f:
                            m = json.load(f)
                        if m.get("accuracy", 0) > best_meta.get("accuracy", 0):
                            best_meta = m
                    except (json.JSONDecodeError, OSError):
                        pass
        in_ensemble = weight > 0
        result[name] = {
            "loaded":       loaded,
            "accuracy":     best_meta.get("accuracy", 0),
            "wf_accuracy":  best_meta.get("wf_accuracy", 0),
            "trained_at":   best_meta.get("trained_at", "never"),
            "epochs":       best_meta.get("epochs", 0),
            "in_ensemble":  in_ensemble,
            "weight":       weight,
            "status":       "active" if in_ensemble else "archived",
        }
    return jsonify(result)


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
    p = DATA / "token_budget.json"
    if p.exists():
        try:
            with open(p) as f:
                return jsonify(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    return jsonify({"used_today": 0, "limit": 90000})


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
        from data_feed import TrainingDataStore
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
                last = last.replace(tzinfo=timezone.utc)
            sig_age = int((datetime.now(timezone.utc) - last).total_seconds())
        except (ValueError, TypeError):
            pass

    try:
        result = subprocess.run(["ps", "aux"], capture_output=True, text=True, timeout=5)
        lines  = result.stdout.split("\n")
        bots   = [l for l in lines
                  if ("launcher.py" in l or "spot_bot.py" in l or "futures_bot.py" in l)
                  and "grep" not in l and "SCREEN" not in l]
        dashes = [l for l in lines
                  if "server.py" in l and "grep" not in l and "SCREEN" not in l]
    except Exception:
        bots, dashes = [], []

    open_t = sum(1 for t in trades if t.get("status") == "open")
    closed = sum(1 for t in trades if t.get("status") == "closed")
    pnl    = sum(t.get("pnl", 0) for t in trades if t.get("status") == "closed")

    def _model_exists(name):
        for subdir in ["spot", "futures", ""]:
            base = DATA / subdir if subdir else DATA
            if (base / name).exists():
                return True
        return False

    models_ok = all([
        _model_exists("rf_model.pkl"),
        _model_exists("lgbm_model.pkl"),
        _model_exists("lstm_model.keras"),
    ])

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

    return jsonify({
        "bot_running":      len(bots) >= 1,
        "bot_instances":    len(bots),
        "dash_running":     len(dashes) >= 1,
        "rf_model":         _model_exists("rf_model.pkl"),
        "xgb_model":        _model_exists("lgbm_model.pkl"),
        "lstm_model":       _model_exists("lstm_model.keras"),
        "models_ok":        models_ok,
        "signal_age":       sig_age,
        "open_trades":      open_t,
        "closed_trades":    closed,
        "total_pnl":        round(pnl, 4),
        "reviews":          reviews,
        "token_usage_pct":  token_pct,
        "spot_open":        spot_open,
        "futures_open":     futures_open,
        "exchange_mode":    exec_mode,
        "training_source":  "api.binance.com",
        "ensemble_v4":      True,
        "lstm_archived":    True,
    })


ENV_PATH = Path.home() / "cryptobot_v3" / ".env"
SERVICE_PATH = Path("/etc/systemd/system/cryptobot-futures.service")


def _read_env_var(key):
    if ENV_PATH.exists():
        with open(ENV_PATH) as f:
            for line in f:
                if line.startswith(key + "="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    # Fall back to systemd service file
    if SERVICE_PATH.exists():
        with open(SERVICE_PATH) as f:
            for line in f:
                if f'BOT_MODE=' in line:
                    return line.split('BOT_MODE=', 1)[1].strip().strip("\"'")
    return "spot"


def _write_env_var(key, value):
    if not ENV_PATH.exists():
        return False
    with open(ENV_PATH) as f:
        lines = f.readlines()
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


def _has_open_trades():
    """Check if there are any open trades in the current mode's state file."""
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


@app.route("/api/get_mode", methods=["GET"])
def get_mode():
    mode = _read_env_var("BOT_MODE")
    has_open = _has_open_trades()
    return jsonify({"mode": mode, "has_open_trades": has_open})


@app.route("/api/set_mode", methods=["POST"])
def set_mode():
    err, code = _check_auth()
    if err:
        return err, code

    # Block switch if there are open trades
    if _has_open_trades():
        return jsonify({
            "error": "Cannot switch mode while trades are open. Close all trades first."
        }), 409

    data = request.get_json(force=True) or {}
    mode = data.get("mode", "spot")
    if mode not in ("spot", "futures"):
        return jsonify({"error": "Invalid mode. Choose 'spot' or 'futures'"}), 400

    _write_env_var("BOT_MODE", mode)

    # Update systemd service file
    if SERVICE_PATH.exists():
        with open(SERVICE_PATH) as f:
            svc_lines = f.readlines()
        new_svc = []
        for line in svc_lines:
            if "BOT_MODE=" in line:
                new_svc.append(f'Environment="BOT_MODE={mode}"\n')
            else:
                new_svc.append(line)
        with open(SERVICE_PATH, "w") as f:
            f.writelines(new_svc)

    # Restart bot service
    try:
        subprocess.run(["systemctl", "restart", f"cryptobot-{mode}"], timeout=10)
    except Exception:
        pass

    return jsonify({"mode": mode, "restarting": True})


@app.route("/api/stop_trading", methods=["POST"])
def stop_trading():
    err, code = _check_auth()
    if err:
        return err, code

    p = DATA / "trading_paused.json"
    p.parent.mkdir(exist_ok=True)
    with open(p, "w") as f:
        json.dump({"paused": True, "timestamp": datetime.now(timezone.utc).isoformat()}, f)
    return jsonify({"status": "paused", "trading_paused": True})


@app.route("/api/start_trading", methods=["POST"])
def start_trading():
    err, code = _check_auth()
    if err:
        return err, code

    p = DATA / "trading_paused.json"
    p.parent.mkdir(exist_ok=True)
    with open(p, "w") as f:
        json.dump({"paused": False, "timestamp": datetime.now(timezone.utc).isoformat()}, f)
    return jsonify({"status": "running", "trading_paused": False})


@app.route("/api/strategy_config", methods=["GET"])
def get_strategy_config():
    for cfg_path in (CFG_FUT, CFG_SPOT):
        if cfg_path.exists():
            try:
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
                strat = cfg.get("strategy", {})
                ml    = cfg.get("ml", {})
                train = cfg.get("training", {})
                return jsonify({
                    "forward_bars":       ml.get("forward_bars", 1),
                    "timeframe":          ml.get("timeframe", "1h"),
                    "min_confidence":     strat.get("min_confidence", 0.42),
                    "min_votes":          strat.get("min_votes", 2),
                    "htf_filter_mode":    strat.get("htf_filter_mode", "strict"),
                    "training_min_bars":  train.get("min_bars_per_coin", 3000),
                    "training_min_coins": train.get("min_coins", 8),
                    "dynamic_min_conf":   True,
                })
            except (yaml.YAMLError, OSError):
                pass
    return jsonify({
        "forward_bars": 1, "timeframe": "1h",
        "min_confidence": 0.42, "min_votes": 2,
        "htf_filter_mode": "strict",
        "training_min_bars": 3000, "training_min_coins": 8,
        "dynamic_min_conf": True,
    })


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
            if not (1 <= v <= 3):
                return jsonify({"error": "min_votes must be 1–3"}), 400
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
                    "min_confidence":  cfg.get("strategy", {}).get("min_confidence", 0.42),
                    "min_votes":       cfg.get("strategy", {}).get("min_votes", 2),
                    "htf_filter_mode": cfg.get("strategy", {}).get("htf_filter_mode", "strict"),
                }
            except (yaml.YAMLError, OSError) as e:
                return jsonify({"error": f"Config write failed: {e}"}), 500

    return jsonify({"status": "ok", "config": saved})


@app.route("/api/retrain", methods=["POST"])
def request_retrain():
    err, code = _check_auth()
    if err:
        return err, code

    # Rate limit: block if a retrain is already running
    status_p = DATA / "retrain_status.json"
    if status_p.exists():
        try:
            with open(status_p) as f:
                st = json.load(f)
            if st.get("status") == "running":
                return jsonify({"error": "retrain already running", "status": "running"}), 429
        except (json.JSONDecodeError, OSError):
            pass

    data = request.get_json(force=True, silent=True) or {}
    p    = DATA / "retrain_requested.json"
    p.parent.mkdir(exist_ok=True)
    with open(p, "w") as f:
        json.dump({
            "requested": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source":    data.get("source", "dashboard"),
            "reason":    data.get("reason", "manual"),
        }, f)
    return jsonify({"status": "requested", "message": "Retrain signal sent to bot"})


@app.route("/api/retrain_status", methods=["GET"])
def get_retrain_status():
    p = DATA / "retrain_status.json"
    if p.exists():
        try:
            with open(p) as f:
                return jsonify(json.load(f))
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    if API_KEY:
        print(f"Auth enabled — set X-Dashboard-Key header or store key in localStorage('dashboard_key')")
    print(f"Dashboard v3: http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
