import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from unittest.mock import patch
from models.hmm import HMMRegimeModel


def _df(n=300):
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open":   [100.0] * n,
        "high":   [101.0] * n,
        "low":    [ 99.0] * n,
        "close":  [100.0 + i * 0.01 for i in range(n)],
        "volume": [1000.0] * n,
    }, index=idx)


def test_regime_is_inferred_when_no_cache():
    det = HMMRegimeModel()
    with patch.object(det, "_run_inference", return_value="RANGING") as m:
        out = det.predict(_df(), max_age_seconds=60)
        assert out == "RANGING"
        assert m.call_count == 1


def test_regime_uses_cache_within_max_age():
    det = HMMRegimeModel()
    with patch.object(det, "_run_inference", return_value="TRENDING") as m:
        det.predict(_df(), max_age_seconds=60)
        det.predict(_df(), max_age_seconds=60)
        assert m.call_count == 1  # second call served from cache


def test_regime_reinfers_when_caller_demands_fresher():
    det = HMMRegimeModel()
    with patch.object(det, "_run_inference", return_value="RANGING") as m:
        det.predict(_df(), max_age_seconds=300)
        # Simulate 90 s elapsed by backdating the cache stamp.
        det._fresh._times["regime"] -= 90
        det.predict(_df(), max_age_seconds=60)
        assert m.call_count == 2
