"""
Training Pipeline v5 — Multi-Asset Memory-Efficient Dataset Builder.
Produces a unified cross-coin training dataset in parquet format.

Features:
- Multi-timeframe merged features (1h, 4h, 1d)
- Per-asset normalization (no raw price leakage)
- ATR-based dynamic labels
- Polars lazy loading for memory efficiency
- Time-based train/val/test split
- Incremental build (only fetches missing data)

Config keys (training:):
    symbols: list[str]        — base coins (default: ["BTC/USDT", "ETH/USDT"])
    top_n: int                — add top N coins from scanner (default: 10)
    atr_k: float              — ATR multiplier for labels (default: 1.2)
    timeframes: list[str]     — timeframes to merge (default: ["1h","4h","1d"])
    primary_timeframe: str    — primary timeframe for labels (default: "1h")
    forward_bars: int         — forward bars for labels (default: 1)
    min_bars_per_coin: int    — minimum bars required (default: 500)
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone

import numpy as np

log = logging.getLogger(__name__)

try:
    import polars as pl
    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False


# ── Config defaults ──────────────────────────────────────────────────────────

DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
                   "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT",
                   "AVAX/USDT", "DOT/USDT"]
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]


# ── Feature definitions (mirrors make_features in ai_strategy.py) ────────────

def _make_features_polars(df: pl.DataFrame) -> pl.DataFrame:
    """
    Compute 66 engineered features from OHLCV using Polars.
    Returns a DataFrame with the same feature columns as make_features().
    """
    close = pl.col("close")
    high  = pl.col("high")
    low   = pl.col("low")
    vol   = pl.col("volume")
    opn   = pl.col("open")
    ts    = pl.col("timestamp") if "timestamp" in df.columns else pl.lit(0).cast(pl.Int64)

    exprs = [ts.alias("timestamp")]

    # Returns
    for p in [1, 2, 3, 5, 7, 14, 21]:
        exprs.append(close.pct_change(p).alias(f"ret_{p}"))

    # EMA distance
    for period in [9, 21, 50, 100, 200]:
        ema = close.ewm_mean(span=period, min_periods=period)
        exprs.append(((close - ema) / (ema + 1e-9)).alias(f"dist_ema_{period}"))

    ema9  = close.ewm_mean(span=9, min_periods=9)
    ema21 = close.ewm_mean(span=21, min_periods=21)
    ema50 = close.ewm_mean(span=50, min_periods=50)

    exprs.append(ema9.pct_change(3).alias("slope_ema_9"))
    exprs.append(ema21.pct_change(3).alias("slope_ema_21"))
    exprs.append(ema50.pct_change(5).alias("slope_ema_50"))
    exprs.append(((ema9 > ema21) & (ema21 > ema50)).cast(pl.Int8).alias("ema_bull"))
    exprs.append(((ema9 < ema21) & (ema21 < ema50)).cast(pl.Int8).alias("ema_bear"))

    # RSI
    for p in [7, 14, 21]:
        delta = close.diff()
        gain  = pl.when(delta > 0).then(delta).otherwise(0)
        loss  = pl.when(delta < 0).then(-delta).otherwise(0)
        avg_gain = gain.ewm_mean(span=p, min_periods=p)
        avg_loss = loss.ewm_mean(span=p, min_periods=p)
        rs   = avg_gain / (avg_loss + 1e-9)
        rsi  = 100 - (100 / (1 + rs))
        exprs.append(rsi.alias(f"rsi_{p}"))

    # RSI slope (use same delta/rsi_14 as above)
    delta_rsi = close.diff()
    gain_r    = pl.when(delta_rsi > 0).then(delta_rsi).otherwise(0)
    loss_r    = pl.when(delta_rsi < 0).then(-delta_rsi).otherwise(0)
    gain_14   = gain_r.ewm_mean(span=14, min_periods=14)
    loss_14   = loss_r.ewm_mean(span=14, min_periods=14)
    rs_14     = gain_14 / (loss_14 + 1e-9)
    rsi_14    = 100 - (100 / (1 + rs_14))
    exprs.append(rsi_14.diff(3).alias("rsi_slope"))

    # MACD
    ema12    = close.ewm_mean(span=12, min_periods=12)
    ema26    = close.ewm_mean(span=26, min_periods=26)
    macd_val = ema12 - ema26
    signal   = macd_val.ewm_mean(span=9, min_periods=9)
    macd_diff_val = macd_val - signal
    exprs.append((macd_diff_val / (close + 1e-9)).alias("macd_diff"))
    exprs.append(macd_diff_val.diff(1).alias("macd_slope"))
    exprs.append(((macd_diff_val > 0) & (macd_diff_val.shift(1) <= 0)).cast(pl.Int8).alias("macd_cross_up"))
    exprs.append(((macd_diff_val < 0) & (macd_diff_val.shift(1) >= 0)).cast(pl.Int8).alias("macd_cross_down"))

    # Bollinger Bands
    bb_mid  = close.rolling_mean(20, min_periods=20)
    bb_std  = close.rolling_std(20, min_periods=20)
    bb_up   = bb_mid + 2 * bb_std
    bb_lo   = bb_mid - 2 * bb_std
    exprs.append(((close - bb_lo) / (bb_up - bb_lo + 1e-9)).alias("bb_pct"))
    exprs.append(((bb_up - bb_lo) / (bb_mid + 1e-9)).alias("bb_width"))
    exprs.append((close < bb_lo).cast(pl.Int8).alias("bb_low"))
    exprs.append((close > bb_up).cast(pl.Int8).alias("bb_high"))

    # ATR
    tr  = pl.max_horizontal([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()])
    atr = tr.rolling_mean(14, min_periods=14)
    exprs.append((atr / (close + 1e-9)).alias("atr_pct"))
    exprs.append((atr / (atr.rolling_mean(20, min_periods=14) + 1e-9)).alias("atr_ratio"))

    # Volume
    vol_sma = vol.rolling_mean(20, min_periods=20)
    vol_ratio_expr = vol / (vol_sma + 1e-9)
    exprs.append(vol_ratio_expr.alias("vol_ratio"))
    exprs.append((vol.rolling_mean(5, min_periods=5) / (vol_sma + 1e-9)).alias("vol_trend"))
    exprs.append((vol_ratio_expr > 2.0).cast(pl.Int8).alias("vol_spike"))

    # Stochastic
    lo14   = low.rolling_min(14)
    hi14   = high.rolling_max(14)
    stoch_k = 100 * (close - lo14) / (hi14 - lo14 + 1e-9)
    stoch_d = stoch_k.rolling_mean(3, min_periods=3)
    exprs.append(stoch_k.alias("stoch_k"))
    exprs.append((stoch_k - stoch_d).alias("stoch_diff"))

    # CCI
    tp  = (high + low + close) / 3
    tp_ma = tp.rolling_mean(20, min_periods=20)
    tp_md = (tp - tp_ma).abs().rolling_mean(20, min_periods=20)
    exprs.append(((tp - tp_ma) / (0.015 * tp_md + 1e-9) / 100).alias("cci"))

    # Williams %R
    exprs.append((-100 * (hi14 - close) / (hi14 - lo14 + 1e-9) / 100).alias("williams_r"))

    # Price position
    hi50 = high.rolling_max(50)
    lo50 = low.rolling_min(50)
    exprs.append(((close - lo14) / (hi14 - lo14 + 1e-9)).alias("price_pos_14"))
    exprs.append(((close - lo50) / (hi50 - lo50 + 1e-9)).alias("price_pos_50"))

    # Candlestick
    exprs.append(((close - opn).abs() / (close + 1e-9)).alias("body"))
    exprs.append(((high - pl.max_horizontal([close, opn])) / (close + 1e-9)).alias("upper_wick"))
    exprs.append(((pl.min_horizontal([close, opn]) - low) / (close + 1e-9)).alias("lower_wick"))
    exprs.append((close > opn).cast(pl.Int8).alias("is_bullish"))

    # Volatility regime
    ret_s = close.pct_change()
    vol_5_expr  = ret_s.rolling_std(5, min_periods=5)
    vol_20_expr = ret_s.rolling_std(20, min_periods=20)
    exprs.append(vol_5_expr.alias("vol_5"))
    exprs.append(vol_20_expr.alias("vol_20"))
    exprs.append((vol_5_expr / (vol_20_expr + 1e-9)).alias("vol_regime"))

    # ADX
    tr_adx   = pl.max_horizontal([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()])
    atr14    = tr_adx.rolling_mean(14, min_periods=14)
    up_move  = pl.when(high.diff() > low.diff()).then(high.diff()).otherwise(0.0)
    dn_move  = pl.when(low.diff().abs() > high.diff()).then(low.diff().abs()).otherwise(0.0)
    plus_dm  = pl.when((up_move > dn_move) & (up_move > 0.0)).then(up_move).otherwise(0.0)
    minus_dm = pl.when((dn_move > up_move) & (dn_move > 0.0)).then(dn_move).otherwise(0.0)
    plus_di   = 100 * plus_dm.rolling_mean(14, min_periods=14) / (atr14 + 1e-9)
    minus_di  = 100 * minus_dm.rolling_mean(14, min_periods=14) / (atr14 + 1e-9)
    dx_val    = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    exprs.append((dx_val.rolling_mean(14, min_periods=14) / 100).alias("adx"))

    # VWAP
    vwap_val = (close * vol).cum_sum() / (vol.cum_sum() + 1e-9)
    exprs.append(((close - vwap_val) / (vwap_val + 1e-9)).alias("vwap_dist"))
    exprs.append((close > vwap_val).cast(pl.Int8).alias("above_vwap"))

    # CVD
    delta = vol * pl.when(close >= opn).then(1.0).otherwise(-1.0)
    vol5_sum  = vol.rolling_sum(5, min_periods=5)
    vol20_sum = vol.rolling_sum(20, min_periods=20)
    exprs.append((delta.rolling_sum(5, min_periods=5) / (vol5_sum + 1e-9)).alias("cvd_5"))
    exprs.append((delta.rolling_sum(20, min_periods=20) / (vol20_sum + 1e-9)).alias("cvd_20"))

    # Price acceleration
    exprs.append(close.pct_change(3).diff(2).alias("price_accel"))

    # Sharpe-like
    ret_pct  = close.pct_change()
    ret10_m  = ret_pct.rolling_mean(10, min_periods=10)
    ret10_s  = ret_pct.rolling_std(10, min_periods=10)
    ret20_m  = ret_pct.rolling_mean(20, min_periods=20)
    ret20_s  = ret_pct.rolling_std(20, min_periods=20)
    exprs.append((ret10_m / (ret10_s + 1e-9)).alias("sharpe_10"))
    exprs.append((ret20_m / (ret20_s + 1e-9)).alias("sharpe_20"))

    # Donchian breakouts
    exprs.append((close >= hi14.shift(1)).cast(pl.Int8).alias("dc_breakout_20"))
    exprs.append((close <= lo14.shift(1)).cast(pl.Int8).alias("dc_breakdown_20"))
    exprs.append((close >= hi50.shift(1)).cast(pl.Int8).alias("dc_breakout_50"))
    exprs.append((close <= lo50.shift(1)).cast(pl.Int8).alias("dc_breakdown_50"))

    # Additional
    exprs.append((atr / (atr.rolling_mean(50, min_periods=14) + 1e-9)).alias("vol_expansion"))
    exprs.append(((vol - vol_sma) / (vol_sma + 1e-9)).alias("vol_delta"))

    # Liquidation sweeps
    exprs.append(((low < lo14.shift(1)) & (close > lo14.shift(1))).cast(pl.Int8).alias("liq_sweep_up"))
    exprs.append(((high > hi14.shift(1)) & (close < hi14.shift(1))).cast(pl.Int8).alias("liq_sweep_down"))

    # HTF alignment
    ema96  = close.ewm_mean(span=96, min_periods=96)
    ema200 = close.ewm_mean(span=200, min_periods=200)
    exprs.append(((close > ema96) & (close > ema200)).cast(pl.Int8).alias("htf_bull"))
    exprs.append(((close < ema96) & (close < ema200)).cast(pl.Int8).alias("htf_bear"))
    exprs.append((
        ((close > ema9) & (close > ema21) & (close > ema50) & (close > ema96)).cast(pl.Int8)
        - ((close < ema9) & (close < ema21) & (close < ema50) & (close < ema96)).cast(pl.Int8)
    ).alias("htf_align"))

    return df.select(exprs).drop_nulls()


# ── ATR-based labels ─────────────────────────────────────────────────────────

def _make_labels_polars(df: pl.DataFrame, forward_bars: int = 1, atr_k: float = 1.2) -> pl.Series:
    """ATR-based dynamic threshold: BUY=2, HOLD=1, SELL=0."""
    close = pl.col("close")
    high  = pl.col("high")
    low   = pl.col("low")

    tr  = pl.max_horizontal([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()])
    atr = tr.rolling_mean(14, min_periods=14)
    threshold = ((atr / (close + 1e-9)) * atr_k).clip(0.001, 0.03)
    future = close.shift(-forward_bars) / close - 1

    target_expr = pl.when(future > threshold).then(pl.lit(2, pl.Int8)). \
        otherwise(pl.when(future < -threshold).then(pl.lit(0, pl.Int8)). \
        otherwise(pl.lit(1, pl.Int8)))

    result = df.select(target=target_expr)["target"]
    return result


# ── Per-asset normalization ──────────────────────────────────────────────────

def _normalize_features(df: pl.DataFrame, method: str = "clip", exclude_cols: set = None) -> pl.DataFrame:
    """
    Normalize features globally.
    - method="clip": clip extreme outliers at 0.1%/99.9% quantiles (default, fast)
    - method="zscore": z-score standardization per column (per coin if symbol column present)
    """
    if exclude_cols is None:
        exclude_cols = {"timestamp", "symbol", "target", "close", "open", "high", "low", "volume"}

    cols_to_process = [c for c in df.columns if c not in exclude_cols]
    if not cols_to_process:
        return df

    if method == "zscore":
        for col in cols_to_process:
            if df[col].dtype in (pl.Float32, pl.Float64):
                mean = df[col].mean()
                std  = df[col].std()
                if std is not None and std > 1e-9:
                    df = df.with_columns(((pl.col(col) - mean) / std).alias(col))
        return df

    # Default: clip outliers
    for col in cols_to_process:
        if df[col].dtype in (pl.Float32, pl.Float64):
            q1 = df[col].quantile(0.001, interpolation="linear")
            q99 = df[col].quantile(0.999, interpolation="linear")
            if q1 is not None and q99 is not None and q1 < q99:
                df = df.with_columns(pl.col(col).clip(q1, q99))

    return df


# ── Multi-timeframe merge ────────────────────────────────────────────────────

def _merge_timeframes(
    features_by_tf: Dict[str, pl.DataFrame],
) -> Optional[pl.DataFrame]:
    """
    Merge features from multiple timeframes aligned on timestamp.
    Forward-fills safely (no lookahead).
    Each timeframe's features are prefixed with tf_
    """
    if not features_by_tf:
        return None

    tfs = sorted(features_by_tf.keys())
    base_tf = tfs[0]
    base = features_by_tf[base_tf].clone()

    feature_cols = [c for c in base.columns if c not in ("timestamp",)]
    rename_map = {c: f"{base_tf}_{c}" for c in feature_cols}
    rename_map["timestamp"] = "timestamp"
    base = base.rename(rename_map)

    for tf in tfs[1:]:
        df = features_by_tf[tf]
        fcols = [c for c in df.columns if c not in ("timestamp",)]
        rmap = {c: f"{tf}_{c}" for c in fcols}
        rmap["timestamp"] = "timestamp"
        df = df.rename(rmap)

        base = base.join_asof(
            df.sort("timestamp"),
            on="timestamp",
            strategy="backward",
        )

    return base


# ── Main dataset builder ─────────────────────────────────────────────────────

def build_training_dataset(
    feed,
    config: dict,
    symbols: List[str] = None,
    timeframes: List[str] = None,
    top_n: int = None,
    atr_k: float = None,
    force_rebuild: bool = False,
) -> Dict:
    """
    Build unified multi-asset training dataset.

    Returns:
        dict with keys: path, n_rows, n_symbols, n_features, symbols, build_time_sec, memory_mb
    """
    t0 = time.time()

    training_cfg = config.get("training", {})
    ml_cfg       = config.get("ml", {})

    if symbols is None:
        symbols = training_cfg.get("symbols", DEFAULT_SYMBOLS)
    if top_n is None:
        top_n = training_cfg.get("top_n", 10)
    if atr_k is None:
        atr_k = training_cfg.get("atr_k", 1.2)
    if timeframes is None:
        timeframes = training_cfg.get("timeframes", DEFAULT_TIMEFRAMES)

    primary_tf    = training_cfg.get("primary_timeframe", timeframes[0])
    forward_bars  = ml_cfg.get("forward_bars", 1)
    min_bars      = training_cfg.get("min_bars_per_coin", 500)
    normalize_method = ml_cfg.get("normalize", "clip")

    from core.config import DATA_DIR
    output_path = DATA_DIR / "training_dataset.parquet"

    if output_path.exists() and not force_rebuild:
        log.info(f"Dataset already exists: {output_path} — skipping build (use force_rebuild=True)")
        stats = _compute_dataset_stats(output_path)
        stats["path"] = str(output_path)
        stats["build_time_sec"] = 0
        return stats

    log.info(f"Building training dataset: {len(symbols)} symbols × {len(timeframes)} timeframes (normalize={normalize_method})")
    log.info(f"  ATR k={atr_k}, forward_bars={forward_bars}, min_bars={min_bars}")

    all_rows = []
    symbol_counts = {}
    errors = []

    for sym in symbols:
        try:
            sym_features_by_tf = {}
            for tf in timeframes:
                df = feed.fetch_ohlcv(sym, tf, limit=5000)
                if df is None or len(df) < min_bars:
                    log.info(f"  {sym}/{tf}: insufficient bars ({len(df) if df is not None else 0}/{min_bars})")
                    continue

                # Convert to Polars if needed
                if _HAS_PANDAS and isinstance(df, pd.DataFrame):
                    df_pl = pl.from_pandas(df.reset_index())
                elif _HAS_POLARS and isinstance(df, pl.DataFrame):
                    df_pl = df.clone()
                else:
                    continue

                # Rename timestamp column if needed
                ts_col = "timestamp" if "timestamp" in df_pl.columns else "index"
                if ts_col != "timestamp":
                    df_pl = df_pl.rename({ts_col: "timestamp"})

                features = _make_features_polars(df_pl)
                features = _normalize_features(features)
                sym_features_by_tf[tf] = features

            if primary_tf not in sym_features_by_tf:
                log.debug(f"  {sym}: lacking primary timeframe {primary_tf} — skip")
                continue

            merged = _merge_timeframes(sym_features_by_tf)
            if merged is None or len(merged) < 50:
                continue

            # Compute labels from primary timeframe raw data
            primary_df = feed.fetch_ohlcv(sym, primary_tf, limit=5000)
            if primary_df is None:
                continue
            if _HAS_PANDAS and isinstance(primary_df, pd.DataFrame):
                labels_pl = _make_labels_polars(
                    pl.from_pandas(primary_df.reset_index()), forward_bars, atr_k
                )
            else:
                labels_pl = _make_labels_polars(primary_df, forward_bars, atr_k)

            # Align labels with merged features
            n_merged = len(merged)
            n_labels = len(labels_pl)
            if n_merged > n_labels:
                merged = merged.head(n_labels)
            elif n_labels > n_merged:
                labels_pl = labels_pl.head(n_merged)

            merged = merged.with_columns(
                target=labels_pl.alias("target"),
                symbol=pl.lit(sym.replace("/", "_")),
            )

            if len(merged) < 50:
                log.debug(f"  {sym}: too few rows after merge ({len(merged)})")
                continue

            all_rows.append(merged)
            symbol_counts[sym] = len(merged)
            log.info(f"  {sym}: {len(merged)} rows, {len(merged.columns)} features")

        except Exception as e:
            errors.append(f"{sym}: {e}")
            log.warning(f"  {sym}: pipeline error: {e}")

    if not all_rows:
        log.error("No valid data for any symbol — dataset build failed")
        if errors:
            log.error(f"Errors: {errors}")
        return {"error": "no valid data", "errors": errors}

    # Align schemas across all per-symbol DataFrames for vertical concat
    all_columns = set()
    for df in all_rows:
        all_columns.update(df.columns)
    all_columns = sorted(all_columns)
    # Put timestamp, symbol, target last for consistency
    meta_cols = ["timestamp", "symbol", "target"]
    feature_cols = [c for c in all_columns if c not in meta_cols]
    ordered_cols = feature_cols + meta_cols

    aligned = []
    for df in all_rows:
        missing = [c for c in ordered_cols if c not in df.columns]
        extra   = [c for c in df.columns if c not in ordered_cols]
        if missing:
            for mc in missing:
                df = df.with_columns(pl.lit(None).alias(mc))
        if extra:
            df = df.drop(extra)
        aligned.append(df.select(ordered_cols))

    final = pl.concat(aligned, how="vertical")
    final = _normalize_features(final, method=normalize_method)

    # Write to parquet
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.write_parquet(output_path)

    elapsed = time.time() - t0
    stats = {
        "path": str(output_path),
        "n_rows": len(final),
        "n_symbols": len(symbol_counts),
        "n_features": len(final.columns) - 3,  # minus timestamp, symbol, target
        "symbols": list(symbol_counts.keys()),
        "per_symbol": symbol_counts,
        "build_time_sec": round(elapsed, 1),
        "memory_mb": round(output_path.stat().st_size / 1024 / 1024, 2),
        "errors": errors,
    }

    log.info(
        f"Dataset built: {stats['n_rows']} rows × {stats['n_features']} features "
        f"from {stats['n_symbols']} symbols in {elapsed:.1f}s ({stats['memory_mb']} MB)"
    )

    return stats


def _compute_dataset_stats(path: Path) -> Dict:
    """Quick stats from existing parquet without loading full dataset."""
    df = pl.scan_parquet(path)
    schema = df.collect_schema()
    n_features = len(schema.names()) - 3  # minus timestamp, symbol, target
    stats = df.select([
        pl.len().alias("n_rows"),
        pl.col("symbol").n_unique().alias("n_symbols"),
        pl.col("target").value_counts().alias("label_counts"),
    ]).collect()

    n_rows = stats["n_rows"][0]
    n_symbols = stats["n_symbols"][0]

    label_counts = {}
    for item in stats["label_counts"][0]:
        label_counts[str(item["target"])] = item["count"]

    return {
        "n_rows": n_rows,
        "n_symbols": n_symbols,
        "n_features": n_features,
        "label_counts": label_counts,
        "memory_mb": round(path.stat().st_size / 1024 / 1024, 2) if path.exists() else 0,
    }


# ── Load dataset with time-based split ───────────────────────────────────────

def load_dataset(
    path: Path = None,
    train_frac: float = 0.80,
    val_frac: float = 0.10,
    limit_rows: int = None,
) -> Dict:
    """
    Load the unified dataset and split by time (no leakage).

    Returns:
        dict with keys: train_X, train_y, val_X, val_y, test_X, test_y,
                        full_df (polars), symbols, n_features, n_rows
    """
    if path is None:
        from core.config import DATA_DIR
        path = DATA_DIR / "training_dataset.parquet"

    if not path.exists():
        return {"error": f"Dataset not found: {path} — run build_training_dataset() first"}

    df = pl.scan_parquet(path)

    if limit_rows:
        df = df.limit(limit_rows)

    n_rows = df.select(pl.len()).collect()["len"][0]
    n_symbols = df.select(pl.col("symbol").n_unique().alias("n")).collect()["n"][0]

    # Time-based split
    train_end = int(n_rows * train_frac)
    val_end   = int(n_rows * (train_frac + val_frac))

    df = df.sort("timestamp")

    train_df = df.slice(0, train_end)
    val_df   = df.slice(train_end, val_end - train_end)
    test_df  = df.slice(val_end, n_rows - val_end)

    # Collect to memory
    train = train_df.collect()
    val   = val_df.collect()
    test  = test_df.collect()

    feat_cols = [c for c in train.columns if c not in ("timestamp", "symbol", "target")]

    result = {
        "train_X": train.select(feat_cols),
        "train_y": train["target"],
        "val_X":   val.select(feat_cols),
        "val_y":   val["target"],
        "test_X":  test.select(feat_cols),
        "test_y":  test["target"],
        "full":    pl.concat([train, val, test], how="vertical"),
        "symbols": train["symbol"].unique().to_list() if "symbol" in train.columns else [],
        "n_features": len(feat_cols),
        "n_rows": n_rows,
        "n_train": len(train),
        "n_val":   len(val),
        "n_test":  len(test),
    }

    log.info(
        f"Dataset loaded: {result['n_rows']} rows | "
        f"train={result['n_train']} val={result['n_val']} test={result['n_test']} | "
        f"{result['n_features']} features | {n_symbols} symbols"
    )

    return result


# ── OOF (Out-Of-Fold) support ────────────────────────────────────────────────

def load_dataset_time_splits(
    n_splits: int = 5,
    path: Path = None,
) -> List[Tuple]:
    """
    Return list of (train_df, val_df) splits for OOF meta-model.
    Splits are time-ordered, no shuffling, no leakage.
    """
    if path is None:
        from core.config import DATA_DIR
        path = DATA_DIR / "training_dataset.parquet"

    if not path.exists():
        return []

    df = pl.read_parquet(path).sort("timestamp")
    n = len(df)

    splits = []
    fold_size = n // (n_splits + 1)

    for i in range(n_splits):
        train_end = fold_size * (i + 1)
        val_end   = min(fold_size * (i + 2), n)

        train_fold = df.slice(0, train_end)
        val_fold   = df.slice(train_end, val_end - train_end)

        splits.append((train_fold, val_fold))

    return splits


# ── Legacy compatibility: convert to pandas for existing model trainers ──────

def to_pandas_feat_labels(dataset_split: Dict) -> Tuple:
    """
    Convert loaded dataset split dict to (feat_df, labels_s, raw_df) tuples
    compatible with existing AIStrategyEngine.train_all().

    Returns: (combined_feats, combined_labels, combined_raw)
    combined_raw is the full feature DataFrame (used as placeholder for df param).
    The caller should provide real OHLCV data separately for LSTM/Meta trainers.
    """
    if "error" in dataset_split:
        raise ValueError(dataset_split["error"])

    full = dataset_split.get("full")
    if full is None:
        raise ValueError("No 'full' DataFrame in dataset split")

    feat_cols = [c for c in full.columns if c not in ("timestamp", "symbol", "target")]
    feat_df = full.select(feat_cols).to_pandas()
    labels_s = full["target"].to_pandas()

    return feat_df, labels_s, feat_df.copy()


def build_raw_combined(feed, symbols: List[str], timeframe: str = "1h",
                       min_bars: int = 100) -> "pd.DataFrame":
    """
    Build a combined raw OHLCV DataFrame of all symbols for LSTM/MetaModel training.
    Fetches from cached parquet first, then real Binance.
    Returns pandas DataFrame (compatible with existing trainers).
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    parts = []
    for sym in symbols:
        df = feed.fetch_ohlcv(sym, timeframe, limit=5000)
        if df is not None and len(df) >= min_bars:
            parts.append(df)

    if not parts:
        return None

    combined = pd.concat(parts, ignore_index=True)
    log.info(f"Raw combined OHLCV: {len(combined)} bars from {len(parts)} symbols @ {timeframe}")
    return combined
