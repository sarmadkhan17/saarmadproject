"""Migrate already-stored timestamps from their current tz to fixed UTC+3.

Same instant, relabeled offset (e.g. ...T19:48+00:00 -> ...T22:48+03:00).
Touches: data/*.json (recursively, full ISO datetimes only) and
trade_memory.db timestamp columns. Date-only bucket keys (YYYY-MM-DD) are
left untouched. A full backup of data/ is taken first.

Idempotent: a value already at +03:00 converts to itself.
"""
import json
import re
import shutil
import sqlite3
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

LOCAL_TZ = timezone(timedelta(hours=3))
DATA = Path(__file__).resolve().parent.parent / "data"

# Full ISO datetime (with time); optional fractional seconds; optional offset.
ISO_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?([+-]\d{2}:\d{2}|Z)?$"
)


def conv(s: str) -> str:
    """Convert one ISO datetime string to the same instant in LOCAL_TZ."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:           # legacy naive values were UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(LOCAL_TZ).isoformat()


def walk(obj):
    """Recursively convert ISO datetime strings in a JSON-like structure."""
    if isinstance(obj, dict):
        return {k: walk(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [walk(v) for v in obj]
    if isinstance(obj, str) and ISO_RE.match(obj):
        try:
            return conv(obj)
        except Exception:
            return obj
    return obj


def migrate_json():
    for p in sorted(DATA.glob("*.json")):
        data = json.loads(p.read_text())
        new = walk(data)
        if new != data:
            p.write_text(json.dumps(new, indent=2))
            print(f"  json  {p.name}: converted")
        else:
            print(f"  json  {p.name}: no change")


# table -> timestamp columns
DB_COLS = {
    "trades": ["closed_at"],
    "judge_critiques": ["reviewed_at"],
    "meta_rules": ["synthesized_at"],
}


def migrate_db():
    db = DATA / "trade_memory.db"
    if not db.exists():
        print("  db    trade_memory.db: missing, skipped")
        return
    c = sqlite3.connect(db)
    existing = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    for table, cols in DB_COLS.items():
        if table not in existing:
            continue
        pk = "rowid"
        for col in cols:
            n = 0
            rows = c.execute(
                f"SELECT {pk}, {col} FROM {table} WHERE {col} IS NOT NULL"
            ).fetchall()
            for rid, val in rows:
                if isinstance(val, str) and ISO_RE.match(val):
                    new = conv(val)
                    if new != val:
                        c.execute(f"UPDATE {table} SET {col}=? WHERE {pk}=?", (new, rid))
                        n += 1
            print(f"  db    {table}.{col}: {n} converted")
    c.commit()
    c.close()


def main():
    if not DATA.exists():
        sys.exit(f"data dir not found: {DATA}")
    stamp = datetime.now(LOCAL_TZ).strftime("%Y%m%d_%H%M%S")
    backup = DATA.parent / f"data_backup_tz_{stamp}"
    shutil.copytree(DATA, backup)
    print(f"backup -> {backup}")
    migrate_json()
    migrate_db()
    print("done.")


if __name__ == "__main__":
    main()
