import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import json
from pathlib import Path
from backtest.baseline import capture_baseline


def test_capture_baseline_emits_required_metrics(tmp_path):
    state = {
        "trades": [
            {"status": "closed", "side": "short", "pnl":  1.5,
             "close_timestamp": "2026-05-18T10:00:00+00:00", "duration_hours": 1.0},
            {"status": "closed", "side": "short", "pnl": -1.0,
             "close_timestamp": "2026-05-18T11:00:00+00:00", "duration_hours": 2.0},
            {"status": "closed", "side": "long",  "pnl":  2.0,
             "close_timestamp": "2026-05-18T12:00:00+00:00", "duration_hours": 0.5},
            {"status": "open",   "side": "short", "live_pnl": -0.3},
        ],
        "stats": {"balance": 3422.59, "total_pnl": -197.47, "wins": 1, "losses": 1},
    }
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(state))
    out = tmp_path / "baseline.json"

    capture_baseline(state_path, out)

    data = json.loads(out.read_text())
    assert data["closed_count"] == 3
    assert data["wins"] == 2
    assert data["losses"] == 1
    assert abs(data["win_rate"] - (2/3)) < 1e-6
    assert abs(data["net_pnl"] - 2.5) < 1e-6
    assert data["side_mix"] == {"short": 2, "long": 1}
    assert "captured_at" in data
    assert data["source_state"] == str(state_path)
