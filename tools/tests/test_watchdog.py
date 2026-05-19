import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
import time
from pathlib import Path
from tools.watchdog import is_stale, check_targets


def test_is_stale_recent(tmp_path: Path):
    p = tmp_path / "fresh.jsonl"
    p.write_text("line\n")
    assert is_stale(p, max_age_seconds=300) is False


def test_is_stale_old(tmp_path: Path):
    p = tmp_path / "old.jsonl"
    p.write_text("line\n")
    old = time.time() - 3600
    os.utime(p, (old, old))
    assert is_stale(p, max_age_seconds=300) is True


def test_is_stale_missing(tmp_path: Path):
    assert is_stale(tmp_path / "nope.jsonl", max_age_seconds=300) is True


def test_check_targets_reports_each(tmp_path: Path):
    fresh = tmp_path / "fresh.jsonl"; fresh.write_text("x")
    stale = tmp_path / "stale.jsonl"; stale.write_text("x")
    os.utime(stale, (time.time() - 3600, time.time() - 3600))
    report = check_targets([
        {"name": "fresh", "path": fresh, "max_age_seconds": 300},
        {"name": "stale", "path": stale, "max_age_seconds": 300},
    ])
    assert report["fresh"]["stale"] is False
    assert report["stale"]["stale"] is True
