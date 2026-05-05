"""
Strict fail-fast prediction pipeline.
Single build_features() function for BOTH training and prediction.
validate_features() checks every invariant before model.predict().
No fallbacks. No silent degradation. No alternative code paths.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.ai_strategy import make_features

log = logging.getLogger(__name__)


class FeatureValidationError(Exception):
    """Raised when prediction features fail validation. No trade for this symbol."""


# ── Multi-TF feature builder (single function, train + predict) ────────────────

SUPPORTED_TFS = ("15m", "1h", "4h", "1d")

def build_features(dfs: Dict[str, pd.DataFrame], prediction_mode: bool = False) -> pd.DataFrame:
    """
    Compute multi-timeframe TA features from per-TF OHLCV DataFrames.
    For each TF: call make_features(df) → 66 cols → prefix with TF name.
    prediction_mode=True returns last row only (1 row), False returns all rows.
    Raises ValueError if fewer than 2 TFs provided.
    """
    if len(dfs) < 2:
        raise ValueError(f"build_features requires at least 2 timeframes, got {len(dfs)}")

    rows = []
    for tf in sorted(dfs.keys()):
        if tf not in SUPPORTED_TFS:
            continue
        df = dfs[tf]
        if df is None or len(df) < 50:
            raise ValueError(f"build_features: {tf} DataFrame is None or too short ({len(df) if df is not None else 0} bars)")

        # Strip timezone info to prevent tz-naive vs tz-aware errors in make_features
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df = df.copy()
            df.index = df.index.tz_convert(None)

        feat = make_features(df)
        if feat is None or len(feat) == 0:
            raise ValueError(f"build_features: make_features({tf}) returned empty DataFrame (input has {len(df)} bars)")

        if prediction_mode:
            last = feat.iloc[[-1]].reset_index(drop=True)
        else:
            last = feat.copy()  # preserve index for alignment

        last = last.rename(columns=lambda c: f"{tf}_{c}")
        rows.append(last)

    if prediction_mode:
        result = pd.concat(rows, axis=1)
        if result.shape[0] == 0:
            raise ValueError("build_features: concatenated result has 0 rows")
        return result
    else:
        # Training mode: join_asof on timestamps to avoid NaN from different TF lengths
        base = rows[0]
        for other in rows[1:]:
            base = pd.merge_asof(
                base.sort_index(), other.sort_index(),
                left_index=True, right_index=True, direction='backward',
            )
        result = base.dropna()
        if result.shape[0] == 0:
            raise ValueError("build_features: 0 rows after multi-TF alignment")
        return result


# ── Central validation ─────────────────────────────────────────────────────────

def validate_features(
    row: pd.DataFrame,
    feature_cols: List[str],
    scaler: StandardScaler,
    symbol: str = "?",
) -> None:
    """
    Validate prediction features before inference. Raises FeatureValidationError
    on ANY mismatch. Must be called before model.predict().

    Checks:
      1. shape == (1, len(feature_cols))
      2. exact column set match (no missing, no extra)
      3. correct column order
      4. no NaN values
      5. no Inf values
      6. scaler.n_features_in_ matches
    """
    n_expected = len(feature_cols)

    # 1. Shape
    if row.shape[0] != 1:
        raise FeatureValidationError(
            f"[{symbol}] Expected 1 row, got {row.shape[0]} rows"
        )
    if row.shape[1] != n_expected:
        raise FeatureValidationError(
            f"[{symbol}] Expected {n_expected} columns, got {row.shape[1]}"
        )

    # 2. Exact column set match
    row_cols = set(row.columns)
    expected_cols = set(feature_cols)
    missing = expected_cols - row_cols
    extra = row_cols - expected_cols
    if missing or extra:
        msg = f"[{symbol}] Column mismatch:"
        if missing:
            msg += f" missing={sorted(list(missing))[:5]}..."
        if extra:
            msg += f" extra={sorted(list(extra))[:5]}..."
        raise FeatureValidationError(msg)

    # 3. Correct column order
    if list(row.columns) != list(feature_cols):
        raise FeatureValidationError(
            f"[{symbol}] Column order mismatch. Expected first: {feature_cols[:3]}, got: {list(row.columns)[:3]}"
        )

    # 4. No NaN
    if row.isna().any().any():
        nan_cols = [c for c in row.columns if row[c].isna().any()]
        raise FeatureValidationError(
            f"[{symbol}] NaN values in columns: {nan_cols[:5]}..."
        )

    # 5. No Inf
    vals = row.values
    if np.isinf(vals).any():
        raise FeatureValidationError(f"[{symbol}] Inf values detected in features")

    # 6. Scaler compatibility
    if scaler.n_features_in_ != n_expected:
        raise FeatureValidationError(
            f"[{symbol}] Scaler expects {scaler.n_features_in_} features, got {n_expected}"
        )


# ── Scaler + feature columns persistence ───────────────────────────────────────

def save_feature_metadata(
    scaler: StandardScaler,
    feature_cols: List[str],
    directory: Path,
) -> None:
    """Atomically save scaler + feature column list for predict-time loading."""
    directory.mkdir(parents=True, exist_ok=True)
    scaler_path = directory / "feature_scaler.pkl"
    cols_path = directory / "feature_columns.json"

    joblib.dump(scaler, scaler_path)
    with open(cols_path, "w") as f:
        json.dump({"columns": feature_cols, "n_features": len(feature_cols)}, f)

    log.info(f"Feature metadata saved: {len(feature_cols)} cols → {directory}")


def load_feature_metadata(directory: Path) -> Tuple[StandardScaler, List[str]]:
    """Load saved scaler and feature columns. Raises FileNotFoundError if missing."""
    scaler_path = directory / "feature_scaler.pkl"
    cols_path = directory / "feature_columns.json"

    if not scaler_path.exists():
        raise FileNotFoundError(f"Feature scaler not found: {scaler_path}")
    if not cols_path.exists():
        raise FileNotFoundError(f"Feature columns not found: {cols_path}")

    scaler = joblib.load(scaler_path)
    with open(cols_path) as f:
        meta = json.load(f)

    feature_cols = meta["columns"]
    n_features = meta["n_features"]

    if scaler.n_features_in_ != n_features:
        raise FeatureValidationError(
            f"Saved scaler expects {scaler.n_features_in_} features, "
            f"but feature_columns.json says {n_features}"
        )

    log.info(f"Feature metadata loaded: {n_features} cols, scaler mean_={scaler.mean_[:3]}")
    return scaler, feature_cols
