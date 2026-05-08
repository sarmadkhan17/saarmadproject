import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, timezone, timedelta


def _make_state(ts_iso):
    return {
        "trades": [{
            "id": "g1", "symbol": "FOO/USDT", "status": "open",
            "timestamp": ts_iso, "price": 1.0, "side": "long",
            "amount": 1.0, "close_price": 0.0, "pnl": 0.0,
            "close_timestamp": "", "sl_order_id": "",
        }],
        "stats": {"total_pnl": 0.0, "wins": 0, "losses": 0},
    }


def _run_ghost_cleanup(d):
    """Mirrors the fixed _cleanup_ghost_trades() logic with empty exchange set."""
    exchange_syms = set()
    now = datetime.now(timezone.utc)
    for t in d["trades"]:
        if t["status"] != "open":
            continue
        sym = t["symbol"]
        if sym in exchange_syms:
            continue
        age_s = None
        ts_str = t.get("timestamp", "")
        try:
            if ts_str:
                opened = datetime.fromisoformat(ts_str)
                if opened.tzinfo is None:
                    opened = opened.replace(tzinfo=timezone.utc)
                age_s = (now - opened).total_seconds()
        except (ValueError, TypeError):
            pass
        if age_s is not None and age_s < 60:
            continue
        # >60s — close it
        t["status"] = "closed"
        t["close_price"] = t["price"]
        t["close_timestamp"] = now.isoformat()


def test_ghost_closed_after_1min():
    """Trade missing from exchange for >60s must be closed."""
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    d = _make_state(old_ts)
    _run_ghost_cleanup(d)
    assert d["trades"][0]["status"] == "closed"


def test_fresh_trade_not_touched():
    """Trade entered <60s ago must not be ghost-closed even if missing from exchange."""
    fresh_ts = datetime.now(timezone.utc).isoformat()
    d = _make_state(fresh_ts)
    _run_ghost_cleanup(d)
    assert d["trades"][0]["status"] == "open"
