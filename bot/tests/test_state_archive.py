import sys, os, json, tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_trade(tid, status, close_ts=""):
    return {
        "id": tid, "symbol": "BTC/USDT", "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "price": 100.0, "side": "long", "amount": 1.0,
        "close_price": 0.0, "pnl": 0.0,
        "close_timestamp": close_ts, "sl_order_id": "",
    }


def _run_archive_logic(trades, main_path, archive_path):
    """Mirrors the archive logic in _do_save_locked."""
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=3)
    to_archive = []
    to_keep = []
    for t in trades:
        if t["status"] == "closed" and t.get("close_timestamp"):
            try:
                ct = datetime.fromisoformat(t["close_timestamp"])
                if ct.tzinfo is None:
                    ct = ct.replace(tzinfo=timezone.utc)
                if ct < cutoff:
                    to_archive.append(t)
                    continue
            except (ValueError, TypeError):
                pass
        to_keep.append(t)

    if to_archive:
        existing = []
        if archive_path.exists():
            with open(archive_path) as f:
                existing = json.load(f)
        tmp = archive_path.with_suffix(".tmp.json")
        with open(tmp, "w") as f:
            json.dump(existing + to_archive, f)
        tmp.replace(archive_path)

    tmp2 = main_path.with_suffix(".tmp.json")
    with open(tmp2, "w") as f:
        json.dump({"trades": to_keep}, f)
    tmp2.replace(main_path)

    return to_keep, to_archive


def test_old_closed_trades_archived():
    """Closed trades older than 3 days move to archive; open and recent stay in main."""
    now = datetime.now(timezone.utc)
    trades = [
        _make_trade("t1", "open", ""),
        _make_trade("t2", "closed", (now - timedelta(days=2)).isoformat()),
        _make_trade("t3", "closed", (now - timedelta(days=4)).isoformat()),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        main_path = Path(tmpdir) / "state.json"
        archive_path = Path(tmpdir) / "state_archive.json"
        kept, archived = _run_archive_logic(trades, main_path, archive_path)
        assert {t["id"] for t in kept} == {"t1", "t2"}
        assert {t["id"] for t in archived} == {"t3"}
        with open(archive_path) as f:
            saved_archive = json.load(f)
        assert any(t["id"] == "t3" for t in saved_archive)


def test_recent_closed_not_archived():
    """Closed trades within 3 days stay in main state."""
    now = datetime.now(timezone.utc)
    trades = [
        _make_trade("t1", "closed", (now - timedelta(days=1)).isoformat()),
        _make_trade("t2", "closed", (now - timedelta(hours=6)).isoformat()),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        main_path = Path(tmpdir) / "state.json"
        archive_path = Path(tmpdir) / "state_archive.json"
        kept, archived = _run_archive_logic(trades, main_path, archive_path)
        assert len(kept) == 2
        assert archived == []
        assert not archive_path.exists()
