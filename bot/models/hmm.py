"""
HMM Regime Model — market state filter.
Inputs: log returns + rolling volatility.
Outputs: TRENDING | RANGING | HIGH_VOL | CRASH
Role: adjusts thresholds only, never overrides trade signals.
"""

import json
import logging
import numpy as np
import joblib
from collections import deque
from datetime import datetime, timezone
from typing import Tuple, Dict, Optional
from pathlib import Path

log = logging.getLogger("HMMRegime")

try:
    from hmmlearn.hmm import GaussianHMM
    _HMM_AVAILABLE = True
except ImportError:
    _HMM_AVAILABLE = False
    log.warning("hmmlearn not installed — HMM regime disabled")

from core.config import DATA_DIR
from data.freshness import Freshness

DATA = DATA_DIR

# ── State → threshold adjustments ─────────────────────────────────────────────
# Only adjusts min_conf + size_mult. Never blocks or overrides signal direction.
# Profile-aware: AGGRESSIVE ignores RANGING; STRICT tightens all regimes.
REGIME_ADJUSTMENTS: Dict[str, Dict] = {
    "RANGING":   {"min_conf_delta": -0.10, "size_mult": 0.70},
    "TRENDING":  {"min_conf_delta": -0.05, "size_mult": 1.15},
    "HIGH_VOL":  {"min_conf_delta":  0.03, "size_mult": 0.60},
    # TEMP: lowered CRASH min_conf_delta from 0.08 to 0.02 for testing
    "CRASH":     {"min_conf_delta":  0.02, "size_mult": 0.40},
}

_PROFILE_MODIFIERS = {
    "STRICT":    {"min_conf_bias":  0.02, "size_factor": 0.90},
    "BALANCED":  {"min_conf_bias":  0.00, "size_factor": 1.00},
    "AGGRESSIVE":{"min_conf_bias": -0.01, "size_factor": 1.05},
}

def _get_regime_adjustments(regime: str, profile: str = "BALANCED") -> Dict:
    base = REGIME_ADJUSTMENTS.get(regime, _NULL_ADJ).copy()
    mod = _PROFILE_MODIFIERS.get(profile, _PROFILE_MODIFIERS["BALANCED"])
    base["min_conf_delta"] = round(base["min_conf_delta"] + mod["min_conf_bias"], 4)
    base["size_mult"] = round(base["size_mult"] * mod["size_factor"], 4)
    return base

_NULL_ADJ = {"min_conf_delta": 0.0, "size_mult": 1.0}


def _extract_features(df) -> Optional[np.ndarray]:
    """
    Build (n, 2) feature matrix: [log_return, rolling_vol].
    Returns None if too few rows.
    """
    close = df["close"]
    if len(close) < 30:
        return None
    log_ret = np.log(close / close.shift(1)).fillna(0).values
    vol     = (np.log(close / close.shift(1)).rolling(20).std()
               .bfill().fillna(0).values)
    return np.column_stack([log_ret, vol])


class HMMRegimeModel:
    """
    Gaussian HMM with 4 latent states mapped to named regimes.
    Trained on BTC 1h data. Used as a soft filter — never veto signals.
    """

    N_STATES  = 4
    MIN_BARS  = 200

    def __init__(self):
        self.model_path = DATA / "hmm_regime.pkl"
        self.meta_path  = DATA / "hmm_regime_meta.json"
        self.model: Optional[GaussianHMM] = None
        self.state_map: Dict[int, str]    = {}   # raw index → name
        self.is_trained                   = False
        self.metadata: Dict               = {}
        self._last_regime: str            = "RANGING"
        self._fresh: Freshness            = Freshness()
        self._regime_history: deque       = deque(maxlen=3)  # Step 5: smoothing
        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if not self.model_path.exists():
            return
        try:
            ckpt           = joblib.load(self.model_path)
            self.model     = ckpt["model"]
            self.state_map = ckpt["state_map"]
            self.is_trained = True
            if self.meta_path.exists():
                with open(self.meta_path) as f:
                    self.metadata = json.load(f)
            log.info(f"HMM loaded (states={self.state_map})")
        except Exception as e:
            log.warning(f"HMM load failed: {e}")
            self.is_trained = False

    def _save(self):
        joblib.dump({"model": self.model, "state_map": self.state_map},
                    self.model_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

    # ── State labelling ───────────────────────────────────────────────────────

    def _label_states(self) -> Dict[int, str]:
        """
        Assign TRENDING / RANGING / HIGH_VOL / CRASH to raw HMM state indices
        by examining learned emission means and covariances.

        Heuristics (applied in priority order):
          CRASH     → lowest mean log-return
          HIGH_VOL  → highest return variance (among remaining)
          TRENDING  → highest |mean log-return| (among remaining)
          RANGING   → the rest
        """
        means  = self.model.means_    # (N, 2): col0=log_ret, col1=vol
        covars = self.model.covars_   # (N, 2, 2) regardless of covariance_type in hmmlearn 0.3+

        # Variance of log-return (first feature) per state — safe for any covariance shape
        if covars.ndim == 3:
            ret_var = np.array([float(covars[i, 0, 0]) for i in range(self.N_STATES)])
        elif covars.ndim == 2:
            ret_var = np.array([float(covars[i, 0]) for i in range(self.N_STATES)])
        else:
            ret_var = np.ones(self.N_STATES)

        remaining = set(range(self.N_STATES))
        mapping   = {}

        # CRASH: lowest mean log-return (most negative)
        crash_idx = min(remaining, key=lambda i: means[i, 0])
        mapping[crash_idx] = "CRASH"
        remaining.discard(crash_idx)

        # HIGH_VOL: highest return variance among remaining
        hv_idx = max(remaining, key=lambda i: ret_var[i])
        mapping[hv_idx] = "HIGH_VOL"
        remaining.discard(hv_idx)

        # TRENDING: highest |mean log-return| among remaining
        tr_idx = max(remaining, key=lambda i: abs(means[i, 0]))
        mapping[tr_idx] = "TRENDING"
        remaining.discard(tr_idx)

        # RANGING: whatever is left
        for idx in remaining:
            mapping[idx] = "RANGING"

        return mapping

    # ── Training ──────────────────────────────────────────────────────────────

    def train(self, df) -> Dict:
        if not _HMM_AVAILABLE:
            return {"error": "hmmlearn not installed"}
        if len(df) < self.MIN_BARS:
            return {"error": f"need {self.MIN_BARS}+ bars, got {len(df)}"}
        try:
            X = _extract_features(df)
            if X is None:
                return {"error": "feature extraction failed"}

            model = GaussianHMM(
                n_components   = self.N_STATES,
                covariance_type = "diag",
                n_iter          = 200,
                tol             = 1e-4,
                random_state    = 42,
            )
            model.fit(X)

            self.model     = model
            self.state_map = self._label_states()
            self.is_trained = True
            self.metadata  = {
                "n_samples":  len(X),
                "state_map":  self.state_map,
                "trained_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save()
            log.info(f"HMM trained! States={self.state_map} n={len(X)}")
            return self.metadata
        except Exception as e:
            log.error(f"HMM training failed: {e}", exc_info=True)
            return {"error": str(e)}

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_inference(self, df) -> str:
        """
        Run HMM decoding on df and return regime label string.
        Does NOT touch caching state — Freshness handles that in predict().
        Falls back to self._last_regime on any error.
        """
        if not self.is_trained or not _HMM_AVAILABLE:
            return "RANGING"
        try:
            X = _extract_features(df)
            if X is None:
                return self._last_regime
            # Use last 100 bars for faster inference; more context → better decode
            X_window      = X[-100:]
            states        = self.model.predict(X_window)
            current_state = int(states[-1])
            raw_regime    = self.state_map.get(current_state, "RANGING")

            # Step 5: smoothing — only switch regime after 3 consecutive identical predictions
            self._regime_history.append(raw_regime)
            if len(self._regime_history) == 3 and len(set(self._regime_history)) == 1:
                self._last_regime = raw_regime
                log.debug(f"HMM regime confirmed: {raw_regime} (3× consecutive)")
            else:
                log.debug(
                    f"HMM raw={raw_regime} (smoothing: {list(self._regime_history)}) "
                    f"→ holding {self._last_regime}"
                )

            return self._last_regime
        except Exception as e:
            log.warning(f"HMM predict error: {e}")
            return self._last_regime

    def predict(self, df, max_age_seconds: float = 60.0) -> str:
        """
        Return current regime name. Uses Freshness contract — callers declare
        their max_age_seconds; default 60 s (was 300 s hardcoded).
        Falls back to 'RANGING' when untrained (_run_inference handles that).
        """
        cached = self._fresh.get("regime", max_age_seconds)
        if cached is not None:
            return cached
        regime = self._run_inference(df)
        self._fresh.set("regime", regime)
        return regime

    # ── Adjustment API ────────────────────────────────────────────────────────

    def get_adjustments(self, regime: str, profile: str = "BALANCED") -> Dict:
        """Return threshold adjustments for regime. Never veto signals."""
        return _get_regime_adjustments(regime, profile)

    def get_regime_and_adjustments(self, df, profile: str = "BALANCED") -> Tuple[str, Dict]:
        regime = self.predict(df)
        return regime, self.get_adjustments(regime, profile)

    def predict_fallback(self, df=None, profile: str = "BALANCED") -> Tuple[str, Dict]:
        """
        Rule-based fallback when HMM is untrained or BTC data is unavailable.
        Uses ADX + volatility + EMA alignment on provided dataframe.
        Falls back to 'RANGING' if no data.
        """
        if df is None or len(df) < 50:
            return "RANGING", _get_regime_adjustments("RANGING", profile)

        try:
            close = df["close"]
            high = df["high"]
            low = df["low"]

            import ta
            adx = ta.trend.ADXIndicator(high, low, close, 14).adx().iloc[-1]
            ema9 = ta.trend.EMAIndicator(close, 9).ema_indicator().iloc[-1]
            ema50 = ta.trend.EMAIndicator(close, 50).ema_indicator().iloc[-1]
            ret = close.pct_change()
            vol_ratio = ret.rolling(20).std().iloc[-1] / (ret.rolling(50).std().iloc[-1] + 1e-9)
            price = close.iloc[-1]
            ch_24 = (price - close.iloc[-96] if len(close) >= 96 else close.iloc[0]) / close.iloc[-96 if len(close) >= 96 else 0]

            if ch_24 < -0.10:
                return "CRASH", _get_regime_adjustments("CRASH", profile)
            elif vol_ratio > 2.0:
                return "HIGH_VOL", _get_regime_adjustments("HIGH_VOL", profile)
            elif adx > 25:
                return "TRENDING", _get_regime_adjustments("TRENDING", profile)
            return "RANGING", _get_regime_adjustments("RANGING", profile)
        except Exception:
            return "RANGING", _get_regime_adjustments("RANGING", profile)
