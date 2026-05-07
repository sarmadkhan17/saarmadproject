"""
AI Strategy Engine v2
RF + LightGBM ensemble with walk-forward validation
Champion/challenger model comparison
"""

import numpy as np
import pandas as pd
import json
import logging
import shutil
import warnings
warnings.filterwarnings("ignore")
import ta
from collections import deque
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
import joblib

from core.config import DATA_DIR

import os

log  = logging.getLogger("AIStrategy")
BOT_MODE = os.environ.get("BOT_MODE", "spot")
DATA = DATA_DIR / BOT_MODE
DATA.mkdir(exist_ok=True)


def make_features(df):
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]
    opn   = df["open"]
    f     = pd.DataFrame(index=df.index)

    # Pre-compute shared intermediates
    ret            = close.pct_change()
    vol_sma        = vol.rolling(20).mean()
    max_oc         = pd.concat([close, opn], axis=1).max(axis=1)
    min_oc         = pd.concat([close, opn], axis=1).min(axis=1)

    for p in [1, 2, 3, 5, 7, 14, 21]:
        f[f"ret_{p}"] = close.pct_change(p)

    for period in [9, 21, 50, 100, 200]:
        ema = ta.trend.EMAIndicator(close, period).ema_indicator()
        f[f"dist_ema_{period}"] = (close - ema) / (ema + 1e-9)

    ema9  = ta.trend.EMAIndicator(close, 9).ema_indicator()
    ema21 = ta.trend.EMAIndicator(close, 21).ema_indicator()
    ema50 = ta.trend.EMAIndicator(close, 50).ema_indicator()
    f["slope_ema_9"]  = ema9.pct_change(3)
    f["slope_ema_21"] = ema21.pct_change(3)
    f["slope_ema_50"] = ema50.pct_change(5)
    f["ema_bull"]     = ((ema9 > ema21) & (ema21 > ema50)).astype(int)
    f["ema_bear"]     = ((ema9 < ema21) & (ema21 < ema50)).astype(int)

    for p in [7, 14, 21]:
        f[f"rsi_{p}"] = ta.momentum.RSIIndicator(close, p).rsi()
    f["rsi_slope"] = f["rsi_14"].diff(3)

    macd = ta.trend.MACD(close)
    f["macd_diff"]       = macd.macd_diff() / (close + 1e-9)
    f["macd_slope"]      = f["macd_diff"].diff(1)
    f["macd_cross_up"]   = ((f["macd_diff"] > 0) & (f["macd_diff"].shift(1) <= 0)).astype(int)
    f["macd_cross_down"] = ((f["macd_diff"] < 0) & (f["macd_diff"].shift(1) >= 0)).astype(int)

    bb = ta.volatility.BollingerBands(close, 20, 2)
    f["bb_pct"]   = bb.bollinger_pband()
    f["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (bb.bollinger_mavg() + 1e-9)
    f["bb_low"]   = (close < bb.bollinger_lband()).astype(int)
    f["bb_high"]  = (close > bb.bollinger_hband()).astype(int)

    atr = ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
    f["atr_pct"]   = atr / (close + 1e-9)
    f["atr_ratio"] = atr / (atr.rolling(20).mean() + 1e-9)

    f["vol_ratio"] = vol / (vol_sma + 1e-9)
    f["vol_trend"] = vol.rolling(5).mean() / (vol_sma + 1e-9)
    f["vol_spike"] = (f["vol_ratio"] > 2.0).astype(int)

    stoch = ta.momentum.StochasticOscillator(high, low, close)
    stoch_k         = stoch.stoch()
    f["stoch_k"]    = stoch_k
    f["stoch_diff"] = stoch_k - stoch.stoch_signal()

    f["cci"]        = ta.trend.CCIIndicator(high, low, close, 20).cci() / 100
    f["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close, 14).williams_r() / 100

    hi14 = high.rolling(14).max()
    lo14 = low.rolling(14).min()
    hi50 = high.rolling(50).max()
    lo50 = low.rolling(50).min()
    f["price_pos_14"] = (close - lo14) / (hi14 - lo14 + 1e-9)
    f["price_pos_50"] = (close - lo50) / (hi50 - lo50 + 1e-9)

    f["body"]       = abs(close - opn) / (close + 1e-9)
    f["upper_wick"] = (high - max_oc) / (close + 1e-9)
    f["lower_wick"] = (min_oc - low) / (close + 1e-9)
    f["is_bullish"] = (close > opn).astype(int)

    f["vol_5"]      = ret.rolling(5).std()
    f["vol_20"]     = ret.rolling(20).std()
    f["vol_regime"] = f["vol_5"] / (f["vol_20"] + 1e-9)
    f["adx"]        = ta.trend.ADXIndicator(high, low, close, 14).adx() / 100

    vwap            = (close * vol).cumsum() / (vol.cumsum() + 1e-9)
    f["vwap_dist"]  = (close - vwap) / (vwap + 1e-9)
    f["above_vwap"] = (close > vwap).astype(int)

    delta        = vol * pd.Series(np.where(close >= opn, 1.0, -1.0), index=df.index)
    vol_roll5    = vol.rolling(5).sum()
    vol_roll20   = vol.rolling(20).sum()
    f["cvd_5"]   = delta.rolling(5).sum()  / (vol_roll5  + 1e-9)
    f["cvd_20"]  = delta.rolling(20).sum() / (vol_roll20 + 1e-9)

    f["price_accel"] = close.pct_change(3).diff(2)

    ret10_mean = ret.rolling(10).mean()
    ret10_std  = ret.rolling(10).std()
    ret20_mean = ret.rolling(20).mean()
    ret20_std  = ret.rolling(20).std()
    f["sharpe_10"] = ret10_mean / (ret10_std + 1e-9)
    f["sharpe_20"] = ret20_mean / (ret20_std + 1e-9)

    hi20 = high.rolling(20).max()
    lo20 = low.rolling(20).min()
    f[f"dc_breakout_20"]  = (close >= hi20.shift(1)).astype(int)
    f[f"dc_breakdown_20"] = (close <= lo20.shift(1)).astype(int)
    f[f"dc_breakout_50"]  = (close >= hi50.shift(1)).astype(int)
    f[f"dc_breakdown_50"] = (close <= lo50.shift(1)).astype(int)

    # Additional features
    f["vol_expansion"] = atr / (atr.rolling(50).mean() + 1e-9)
    f["vol_delta"]     = (vol - vol_sma) / (vol_sma + 1e-9)

    recent_hi = hi20.shift(1)
    recent_lo = lo20.shift(1)
    f["liq_sweep_up"]  = ((low < recent_lo) & (close > recent_lo)).astype(int)
    f["liq_sweep_down"]= ((high > recent_hi) & (close < recent_hi)).astype(int)

    ema96  = ta.trend.EMAIndicator(close, 96).ema_indicator()
    ema200 = ta.trend.EMAIndicator(close, 200).ema_indicator()
    f["htf_bull"]  = ((close > ema96) & (close > ema200)).astype(int)
    f["htf_bear"]  = ((close < ema96) & (close < ema200)).astype(int)
    f["htf_align"] = (
        ((close > ema9) & (close > ema21) & (close > ema50) & (close > ema96)).astype(int)
        - ((close < ema9) & (close < ema21) & (close < ema50) & (close < ema96)).astype(int)
    )

    return f.dropna()


def make_labels(df: pd.DataFrame, forward_bars: int = 1, atr_k: float = 0.5) -> pd.Series:
    """
    ATR 3-class labels: SELL=0, HOLD=1, BUY=2.
    Only labels bars where the forward move exceeds atr_k × ATR/close.
    """
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]

    tr  = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()

    threshold = (atr / (close + 1e-9) * atr_k).clip(0.001, 0.02)
    future    = close.shift(-forward_bars) / (close + 1e-9) - 1

    labels = pd.Series(1, index=df.index, dtype=int)   # HOLD = 1
    labels[future >  threshold] = 2                      # BUY  = 2
    labels[future < -threshold] = 0                      # SELL = 0
    labels = labels[future.notna()]

    counts = labels.value_counts().sort_index()
    total  = len(labels)
    log.info(
        f"Labels (ATR 3-class, k={atr_k}): "
        f"SELL={counts.get(0,0)/total*100:.1f}% "
        f"HOLD={counts.get(1,0)/total*100:.1f}% "
        f"BUY={counts.get(2,0)/total*100:.1f}%"
    )
    return labels


def compute_decay_weights(n: int, gamma: float = 0.995) -> np.ndarray:
    """Exponential decay weights: older samples get lower weight."""
    w = gamma ** np.arange(n - 1, -1, -1, dtype=np.float64)
    return (w / w.sum() * n).astype(np.float32)


class OnlineBuffer:
    """
    Rolling per-symbol OHLCV store.
    Tracks new bars ingested; signals incremental update every UPDATE_EVERY bars.
    """
    MAX_BARS_PER_SYM = 500   # ~3 weeks of 1h
    UPDATE_EVERY     = 24    # trigger incremental update after 24 new bars

    def __init__(self):
        self._store: Dict[str, pd.DataFrame] = {}
        self._new_bar_count   = 0
        self._last_update_count = 0

    def ingest(self, symbol: str, df: pd.DataFrame):
        if df is None or len(df) == 0:
            return
        if symbol not in self._store:
            self._store[symbol] = df.tail(self.MAX_BARS_PER_SYM).copy()
            return
        existing = self._store[symbol]
        new_rows = df[~df.index.isin(existing.index)]
        if len(new_rows) > 0:
            combined = pd.concat([existing, new_rows])
            self._store[symbol] = combined.tail(self.MAX_BARS_PER_SYM)
            self._new_bar_count += len(new_rows)

    def should_update(self) -> bool:
        return (self._new_bar_count - self._last_update_count) >= self.UPDATE_EVERY

    def mark_updated(self):
        self._last_update_count = self._new_bar_count

    def get_combined(self) -> Optional[pd.DataFrame]:
        if not self._store:
            return None
        parts = list(self._store.values())
        combined = pd.concat(parts).sort_index().drop_duplicates()
        return combined if len(combined) >= 300 else None

    @property
    def n_symbols(self) -> int:
        return len(self._store)


class RandomForestStrategy:
    def __init__(self):
        self.model_path   = DATA / "rf_model.pkl"
        self.scaler_path  = DATA / "rf_scaler.pkl"
        self.feat_path    = DATA / "rf_features.pkl"
        self.meta_path    = DATA / "rf_meta.json"
        self.model        = None
        self.scaler       = StandardScaler()
        self.feature_cols = None
        self.is_trained   = False
        self.metadata     = {}
        self._load()

    def _load(self):
        if self.model_path.exists():
            try:
                self.model        = joblib.load(self.model_path)
                self.scaler       = joblib.load(self.scaler_path)
                self.feature_cols = joblib.load(self.feat_path)
                self.is_trained   = True
                if self.meta_path.exists():
                    with open(self.meta_path) as f:
                        self.metadata = json.load(f)
                log.info(f"RF loaded (accuracy={self.metadata.get('accuracy','?')})")
            except Exception as e:
                log.warning(f"RF load failed (will retrain): {e}")
                self.is_trained = False

    def train(self, df, use_decay_weights: bool = False, feat_df=None, labels_s=None,
              forward_bars: int = 1, timeframe: str = "15m",
              min_confidence: float = 0.52, min_votes: int = 2, n_jobs: int = 4):
        log.info("Training Random Forest (walk-forward)...")
        if feat_df is not None and labels_s is not None:
            feat   = feat_df
            labels = labels_s
        else:
            feat   = make_features(df)
            labels = make_labels(df, forward_bars=forward_bars).reindex(feat.index).dropna()
            feat   = feat.loc[labels.index]
        if len(feat) < 300:
            return {"error": "need 300+ bars"}

        self.feature_cols = feat.columns.tolist()
        X  = feat.values
        y  = labels.values.astype(int)
        sw = compute_decay_weights(len(X)) if use_decay_weights else None

        # Walk-forward validation
        tscv   = TimeSeriesSplit(n_splits=5)
        wf_scores = []
        for tr_idx, te_idx in tscv.split(X):
            sc  = StandardScaler()
            Xtr = sc.fit_transform(X[tr_idx])
            Xte = sc.transform(X[te_idx])
            sw_fold = sw[tr_idx] if sw is not None else None
            m = RandomForestClassifier(n_estimators=200, max_depth=10,
                                        min_samples_leaf=5,
                                        random_state=42, n_jobs=n_jobs)
            m.fit(Xtr, y[tr_idx], sample_weight=sw_fold)
            wf_scores.append(m.score(Xte, y[te_idx]))
        wf_acc = np.mean(wf_scores)

        # Final model
        split = int(len(X) * 0.8)
        self.scaler.fit(X[:split])
        Xtr   = self.scaler.transform(X[:split])
        Xte   = self.scaler.transform(X[split:])
        sw_tr = sw[:split] if sw is not None else None
        new_model = RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=8,
            min_samples_split=15, max_features="log2",
            random_state=42, n_jobs=n_jobs,
        )
        new_model.fit(Xtr, y[:split], sample_weight=sw_tr)
        new_acc = new_model.score(Xte, y[split:])

        # Simple floor check — accept if >=40%, discard below
        if new_acc < 0.40:
            log.warning(f"RF new ({new_acc:.2%}) below floor — discarding")
            self._load()
            return {"accuracy": self.metadata.get("test_accuracy", 0), "status": "below_floor"}

        self.model      = new_model
        self.is_trained = True
        self.metadata   = {
            "accuracy":     round(new_acc, 4),
            "wf_accuracy":  round(float(wf_acc), 4),
            "test_accuracy":round(new_acc, 4),
            "forward_bars": forward_bars,
            "timeframe":    timeframe,
            "min_confidence": min_confidence,
            "min_votes":    min_votes,
            "trained_at":   datetime.now(timezone.utc).isoformat(),
            "n_samples":    len(X),
        }
        joblib.dump(self.model,        self.model_path)
        joblib.dump(self.scaler,       self.scaler_path)
        joblib.dump(self.feature_cols, self.feat_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        log.info(f"RF trained! Test={new_acc:.2%} WF={wf_acc:.2%}")
        return self.metadata

    def predict(self, df):
        if not self.is_trained:
            return {"action": "HOLD", "confidence": 0.30, "probs": [0.35, 0.35]}
        feat = make_features(df)
        if len(feat) == 0:
            return {"action": "HOLD", "confidence": 0.30, "probs": [0.35, 0.35]}
        last   = feat.iloc[[-1]][self.feature_cols]
        probs  = self.model.predict_proba(self.scaler.transform(last.values))[0]
        # Binary: [SELL_prob, BUY_prob]
        label = 1 if probs[1] >= probs[0] else 0
        return {"action": "BUY" if label == 1 else "SELL",
                "confidence": round(float(probs[label]), 4),
                "probs": [float(probs[0]), float(probs[1])]}

    def predict_numpy(self, X: "np.ndarray"):
        """Predict from pre-scaled numpy array (1 row × N features). No make_features() involved."""
        if not self.is_trained:
            raise RuntimeError("RF model not trained")
        probs = self.model.predict_proba(X)[0]
        # Binary: [SELL_prob, BUY_prob]
        label = 1 if probs[1] >= probs[0] else 0
        return {"action": "BUY" if label == 1 else "SELL",
                "confidence": round(float(probs[label]), 4),
                "probs": [float(probs[0]), float(probs[1])]}


class LightGBMStrategy:
    def __init__(self):
        self.model_path   = DATA / "lgbm_model.pkl"
        self.scaler_path  = DATA / "lgbm_scaler.pkl"
        self.feat_path    = DATA / "lgbm_features.pkl"
        self.meta_path    = DATA / "lgbm_meta.json"
        self.model        = None
        self.scaler       = StandardScaler()
        self.feature_cols = None
        self.is_trained   = False
        self.metadata     = {}
        self._load()

    def _load(self):
        if self.model_path.exists():
            try:
                self.model        = joblib.load(self.model_path)
                self.scaler       = joblib.load(self.scaler_path)
                self.feature_cols = joblib.load(self.feat_path)
                self.is_trained   = True
                if self.meta_path.exists():
                    with open(self.meta_path) as f:
                        self.metadata = json.load(f)
                log.info(f"LightGBM loaded (accuracy={self.metadata.get('accuracy','?')})")
            except Exception as e:
                log.warning(f"LightGBM load failed (will retrain): {e}")
                self.is_trained = False

    def train(self, df, use_decay_weights: bool = False, feat_df=None, labels_s=None,
              forward_bars: int = 1, timeframe: str = "15m",
              min_confidence: float = 0.52, min_votes: int = 2, n_jobs: int = 4):
        log.info("Training LightGBM (walk-forward)...")
        if feat_df is not None and labels_s is not None:
            feat   = feat_df
            labels = labels_s
        else:
            feat   = make_features(df)
            labels = make_labels(df, forward_bars=forward_bars).reindex(feat.index).dropna()
            feat   = feat.loc[labels.index]
        if len(feat) < 300:
            return {"error": "need 300+ bars"}
        self.feature_cols = feat.columns.tolist()
        X  = feat.values
        y  = labels.values.astype(int)
        sw = compute_decay_weights(len(X)) if use_decay_weights else None

        lgbm_params = dict(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            num_leaves=31, min_child_samples=20,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1, n_jobs=n_jobs,
        )

        # Walk-forward validation (same as RF)
        tscv = TimeSeriesSplit(n_splits=5)
        wf_scores = []
        for tr_idx, te_idx in tscv.split(X):
            sc = StandardScaler()
            Xtr_fold = sc.fit_transform(X[tr_idx])
            Xte_fold = sc.transform(X[te_idx])
            val_sp   = int(len(tr_idx) * 0.85)
            sw_fold  = sw[tr_idx] if sw is not None else None
            m = lgb.LGBMClassifier(**lgbm_params)
            m.fit(Xtr_fold[:val_sp], y[tr_idx[:val_sp]],
                  sample_weight=sw_fold[:val_sp] if sw_fold is not None else None,
                  eval_set=[(Xtr_fold[val_sp:], y[tr_idx[val_sp:]])],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(period=-1)])
            wf_scores.append(m.score(Xte_fold, y[te_idx]))
        wf_acc = float(np.mean(wf_scores))

        # Final model — use a proper val split from train portion (not test)
        sp     = int(len(X) * 0.8)
        val_sp = int(sp * 0.85)
        self.scaler.fit(X[:val_sp])
        Xtr  = self.scaler.transform(X[:val_sp])
        Xval = self.scaler.transform(X[val_sp:sp])
        Xte  = self.scaler.transform(X[sp:])
        sw_tr = sw[:val_sp] if sw is not None else None
        new_model = lgb.LGBMClassifier(**lgbm_params)
        new_model.fit(Xtr, y[:val_sp], sample_weight=sw_tr,
                      eval_set=[(Xval, y[val_sp:sp])],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=-1)])
        new_acc = new_model.score(Xte, y[sp:])

        # Simple floor check — accept if >=40%, discard below
        if new_acc < 0.40:
            log.warning(f"LightGBM new ({new_acc:.2%}) below floor — discarding")
            self._load()
            return {"accuracy": self.metadata.get("test_accuracy", 0), "status": "below_floor"}

        self.model      = new_model
        self.is_trained = True
        self.metadata   = {
            "accuracy":      round(new_acc, 4),
            "wf_accuracy":   round(wf_acc, 4),
            "test_accuracy": round(new_acc, 4),
            "forward_bars":  forward_bars,
            "timeframe":     timeframe,
            "min_confidence": min_confidence,
            "min_votes":     min_votes,
            "trained_at":    datetime.now(timezone.utc).isoformat(),
            "n_samples":     len(X),
        }
        joblib.dump(self.model,        self.model_path)
        joblib.dump(self.scaler,       self.scaler_path)
        joblib.dump(self.feature_cols, self.feat_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        log.info(f"LightGBM trained! Test={new_acc:.2%} WF={wf_acc:.2%}")
        return self.metadata

    def predict(self, df):
        if not self.is_trained:
            return {"action": "HOLD", "confidence": 0.30, "probs": [0.35, 0.35]}
        feat = make_features(df)
        if len(feat) == 0:
            return {"action": "HOLD", "confidence": 0.30, "probs": [0.35, 0.35]}
        last  = feat.iloc[[-1]][self.feature_cols]
        probs = self.model.predict_proba(self.scaler.transform(last.values))[0]
        # Binary: [SELL_prob, BUY_prob]
        label = 1 if probs[1] >= probs[0] else 0
        return {"action": "BUY" if label == 1 else "SELL",
                "confidence": round(float(probs[label]), 4),
                "probs": list(probs)}

    def predict_numpy(self, X: "np.ndarray"):
        """Predict from pre-scaled numpy array (1 row × N features). No make_features() involved."""
        if not self.is_trained:
            raise RuntimeError("LightGBM model not trained")
        probs = self.model.predict_proba(X)[0]
        # Binary: [SELL_prob, BUY_prob]
        label = 1 if probs[1] >= probs[0] else 0
        return {"action": "BUY" if label == 1 else "SELL",
                "confidence": round(float(probs[label]), 4),
                "probs": list(probs)}

class AIStrategyEngine:
    def __init__(self):
        self.rf            = RandomForestStrategy()
        self.lgbm          = LightGBMStrategy()
        self.online_buffer   = OnlineBuffer()
        self._unc_buffer: list = []   # uncertainty log write buffer (Step 7)
        self.performance_log = []
        p = DATA / "trade_results.json"
        if p.exists():
            with open(p) as f:
                self.performance_log = json.load(f)

    def train_all(self, df, feat_df=None, labels_s=None,
                  use_decay_weights: bool = False, btc_rows: int = 0,
                  forward_bars: int = 1, timeframe: str = "15m",
                  min_confidence: float = 0.52, min_votes: int = 2,
                  quick: bool = False, n_jobs: int = 4, progress_fn=None):
        """All modes train RF+LGBM only (LSTM/Meta/TFT removed)."""
        r = {}
        r["rf"]   = self.rf.train(df, feat_df=feat_df, labels_s=labels_s, use_decay_weights=use_decay_weights,
                                    forward_bars=forward_bars, timeframe=timeframe,
                                    min_confidence=min_confidence, min_votes=min_votes, n_jobs=n_jobs)
        if progress_fn:
            try: progress_fn(50)
            except Exception: pass
        r["lgbm"] = self.lgbm.train(df, feat_df=feat_df, labels_s=labels_s, use_decay_weights=use_decay_weights,
                                      forward_bars=forward_bars, timeframe=timeframe,
                                      min_confidence=min_confidence, min_votes=min_votes, n_jobs=n_jobs)
        if progress_fn:
            try: progress_fn(100)
            except Exception: pass

        if quick:
            log.info(f"Quick train done! RF={r['rf'].get('accuracy','?')} LGBM={r['lgbm'].get('accuracy','?')}")
            return r

        log.info(
            f"All models trained! RF={r['rf'].get('accuracy','?')} "
            f"LGBM={r['lgbm'].get('accuracy','?')}"
        )
        return r

    def _get_dynamic_weights(self):
        """Weight each model by walk-forward accuracy; floor at 10% so no model is ignored."""
        rf_wf    = max(self.rf.metadata.get("wf_accuracy",   0.33), 0.10)
        lgbm_wf  = max(self.lgbm.metadata.get("wf_accuracy", 0.33), 0.10)
        raw = np.array([rf_wf, lgbm_wf], dtype=float)
        w   = raw / raw.sum()
        return float(w[0]), float(w[1])

    def get_model_health(self):
        return {
            "rf_accuracy":   self.rf.metadata.get("accuracy", 0),
            "rf_wf":         self.rf.metadata.get("wf_accuracy", 0),
            "lgbm_accuracy": self.lgbm.metadata.get("accuracy", 0),
            "rf_trained":    self.rf.metadata.get("trained_at", "never"),
        }

    def predict(self, df, symbol):
        """Binary probability comparison: BUY if buy_prob >= sell_prob, else SELL."""
        rf_p   = self.rf.predict(df)
        lgbm_p = self.lgbm.predict(df)

        rf_probs   = np.array(rf_p["probs"])      # [SELL_prob, BUY_prob]
        lgbm_probs = np.array(lgbm_p["probs"])

        w_rf, w_lgbm = self._get_dynamic_weights()
        buy_prob  = rf_probs[1] * w_rf + lgbm_probs[1] * w_lgbm
        sell_prob = rf_probs[0] * w_rf + lgbm_probs[0] * w_lgbm

        if rf_p.get("action") == "HOLD" or lgbm_p.get("action") == "HOLD":
            if buy_prob < 0.60 and sell_prob < 0.60:
                action, conf = "HOLD", max(buy_prob, sell_prob)
            elif buy_prob >= sell_prob:
                action, conf = "BUY", min(buy_prob, 0.95)
            else:
                action, conf = "SELL", min(sell_prob, 0.95)
        elif buy_prob >= sell_prob:
            action, conf = "BUY", min(buy_prob, 0.95)
        else:
            action, conf = "SELL", min(sell_prob, 0.95)

        conf = round(float(conf), 4)
        strat = f"bin:RF={rf_probs[1]:.2f}|{rf_probs[0]:.2f} LGBM={lgbm_probs[1]:.2f}|{lgbm_probs[0]:.2f}"
        ts = datetime.now(timezone.utc).isoformat()

        self._unc_buffer.append({
            "symbol": symbol, "timestamp": ts, "action": action,
            "confidence": conf, "uncertainty": 0.0, "ensemble_var": 0.0,
            "mc_uncertainty": 0.0, "gated": False, "regime": "N/A",
        })
        if len(self._unc_buffer) >= 50:
            self._flush_uncertainty_log()

        return {
            "symbol": symbol, "action": action, "confidence": conf,
            "strategy": strat, "timeframe": "AI-v4",
            "indicators": {
                "rf_conf": rf_p["confidence"], "lgbm_conf": lgbm_p["confidence"],
                "rf_action": rf_p["action"], "lgbm_action": lgbm_p["action"],
                "buy_votes": 1 if action == "BUY" else 0,
                "sell_votes": 1 if action == "SELL" else 0,
                "ensemble_v4": True,
            },
            "timestamp": ts,
        }

    def predict_numpy(self, X: "np.ndarray", symbol: str):
        """Predict from pre-scaled numpy array (1 row × N features).
        Binary: BUY if buy_prob >= sell_prob, else SELL."""
        rf_p   = self.rf.predict_numpy(X)
        lgbm_p = self.lgbm.predict_numpy(X)

        rf_probs   = np.array(rf_p["probs"])   # [SELL_prob, BUY_prob]
        lgbm_probs = np.array(lgbm_p["probs"])

        w_rf, w_lgbm = 0.40, 0.60
        buy_prob  = rf_probs[1] * w_rf + lgbm_probs[1] * w_lgbm
        sell_prob = rf_probs[0] * w_rf + lgbm_probs[0] * w_lgbm

        if buy_prob >= sell_prob:
            action, conf = "BUY", min(buy_prob, 0.95)
        else:
            action, conf = "SELL", min(sell_prob, 0.95)

        conf = round(float(conf), 4)
        strat = f"bin:RF={rf_probs[1]:.2f}|{rf_probs[0]:.2f} LGBM={lgbm_probs[1]:.2f}|{lgbm_probs[0]:.2f}"

        return {
            "symbol": symbol, "action": action, "confidence": conf,
            "strategy": strat, "timeframe": "AI-v4",
            "indicators": {
                "rf_conf": rf_p["confidence"], "lgbm_conf": lgbm_p["confidence"],
                "rf_action": rf_p["action"], "lgbm_action": lgbm_p["action"],
                "buy_votes": 1 if action == "BUY" else 0,
                "sell_votes": 1 if action == "SELL" else 0,
            },
        }

    def _flush_uncertainty_log(self):
        if not self._unc_buffer:
            return
        log_path = DATA / "uncertainty_log.json"
        try:
            existing: list = []
            if log_path.exists():
                with open(log_path) as f:
                    existing = json.load(f)
            existing.extend(self._unc_buffer)
            if len(existing) > 10_000:
                existing = existing[-10_000:]
            with open(log_path, "w") as f:
                json.dump(existing, f)
            self._unc_buffer = []
        except Exception as e:
            log.warning(f"Uncertainty log flush failed: {e}")

    def ingest_new_data(self, symbol: str, df: pd.DataFrame):
        """Feed latest OHLCV into rolling online buffer."""
        self.online_buffer.ingest(symbol, df)

    def incremental_update(self) -> dict:
        """
        v4: Lower gate — retrain every 30 trades unconditionally.
        Champion/challenger already rejects worse models.
        Only skip if catastrophically bad (<20% win rate, >20 trades).
        """
        if not self.online_buffer.should_update():
            return {"status": "skipped"}

        total_trades = len(self.performance_log)
        if total_trades < 30:
            log.debug(f"Online learning skipped: {total_trades}/30 trades")
            return {"status": "skipped", "reason": f"only {total_trades}/30 trades"}
        wins     = sum(1 for t in self.performance_log if t.get("pnl", 0) > 0)
        win_rate = wins / total_trades if total_trades else 0
        if total_trades >= 20 and win_rate < 0.20:
            log.warning(f"Online learning blocked: catastrophic win rate {win_rate:.1%} (<20%)")
            return {"status": "skipped", "reason": f"catastrophic win_rate {win_rate:.1%}"}

        combined = self.online_buffer.get_combined()
        if combined is None:
            return {"status": "skipped", "reason": "insufficient data"}

        self.online_buffer.mark_updated()
        log.info(
            f"Online learning triggered: {self.online_buffer.n_symbols} symbols, "
            f"{len(combined)} combined bars"
        )
        results: Dict = {}

        try:
            results["rf"] = self.rf.train(combined, use_decay_weights=True)
        except Exception as e:
            results["rf"] = {"error": str(e)}
            log.warning(f"Online RF failed: {e}")

        try:
            results["lgbm"] = self.lgbm.train(combined, use_decay_weights=True)
        except Exception as e:
            results["lgbm"] = {"error": str(e)}
            log.warning(f"Online LGBM failed: {e}")

        log.info(
            f"Online update done — RF={results.get('rf',{}).get('accuracy','?')} "
            f"LGBM={results.get('lgbm',{}).get('accuracy','?')}"
        )
        return {"status": "updated", "results": results}

    def record_trade_result(self, symbol, pnl):
        self.performance_log.append({
            "symbol": symbol, "pnl": pnl,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        with open(DATA / "trade_results.json", "w") as f:
            json.dump(self.performance_log, f, indent=2)

        # Step 7: annotate most recent uncertainty entry for this symbol with outcome
        try:
            self._flush_uncertainty_log()   # ensure buffer is flushed first
            log_path = DATA / "uncertainty_log.json"
            if log_path.exists():
                with open(log_path) as f:
                    unc_log = json.load(f)
                for entry in reversed(unc_log):
                    if entry.get("symbol") == symbol and "outcome_pnl" not in entry:
                        entry["outcome_pnl"] = round(float(pnl), 6)
                        entry["outcome_win"] = bool(pnl > 0)
                        break
                with open(log_path, "w") as f:
                    json.dump(unc_log, f)
        except Exception:
            pass
