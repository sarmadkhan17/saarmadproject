"""
AI Strategy Engine v2
RF + LightGBM + LSTM with walk-forward validation
Champion/challenger model comparison
Correct labels: forward_bars=1, min_move=0.003
"""

import numpy as np
import pandas as pd
import json
import logging
import shutil
import warnings
warnings.filterwarnings("ignore")
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
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
tf.get_logger().setLevel("ERROR")

from env_config import DATA_DIR

import os

try:
    from models.tft_model import TFTStrategy
    _TFT_AVAILABLE = True
except Exception:
    _TFT_AVAILABLE = False

log  = logging.getLogger("AIStrategy")
BOT_MODE = os.environ.get("BOT_MODE", "spot")
DATA = DATA_DIR / BOT_MODE
DATA.mkdir(exist_ok=True)

MC_SAMPLES            = 15    # MC Dropout forward passes
UNCERTAINTY_THRESHOLD = 0.06  # combined uncertainty → force HOLD


def make_features(df):
    import ta
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
    f["atr_ratio"] = atr / (vol_sma.rolling(0).mean() + 1e-9) if False else atr / (atr.rolling(20).mean() + 1e-9)

    f["vol_ratio"] = vol / (vol_sma + 1e-9)
    f["vol_trend"] = vol.rolling(5).mean() / (vol_sma + 1e-9)
    f["vol_spike"] = (f["vol_ratio"] > 2.0).astype(int)

    stoch = ta.momentum.StochasticOscillator(high, low, close)
    f["stoch_k"]    = stoch.stoch()
    f["stoch_diff"] = stoch.stoch() - stoch.stoch_signal()

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


def make_labels(df, forward_bars=3, atr_multiplier=1.0):
    """
    ATR-based dynamic threshold labels.
    threshold = (ATR/close) * atr_multiplier, clipped [0.001, 0.02].
    Adapts to market volatility — wider threshold in volatile regimes.
    """
    import ta as _ta
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]

    atr       = _ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
    threshold = (atr / (close + 1e-9) * atr_multiplier).clip(lower=0.001, upper=0.02)

    future = close.shift(-forward_bars) / close - 1
    labels = pd.Series(1, index=df.index)
    labels[future >  threshold] = 2   # BUY
    labels[future < -threshold] = 0   # SELL
    labels = labels.dropna()
    counts = labels.value_counts().sort_index()
    total  = len(labels)
    log.info(
        f"Labels (ATR-dynamic): SELL={counts.get(0,0)/total*100:.1f}% "
        f"HOLD={counts.get(1,0)/total*100:.1f}% "
        f"BUY={counts.get(2,0)/total*100:.1f}% "
        f"avg_threshold={threshold.mean():.4f}"
    )
    return labels


def make_sequences(X, y, seq_len=48):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


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

    def train(self, df, use_decay_weights: bool = False, feat_df=None, labels_s=None):
        log.info("Training Random Forest (walk-forward)...")
        if feat_df is not None and labels_s is not None:
            feat   = feat_df
            labels = labels_s
        else:
            feat   = make_features(df)
            labels = make_labels(df).reindex(feat.index).dropna()
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
            cw  = dict(zip(*[np.unique(y[tr_idx]),
                              compute_class_weight("balanced",
                              classes=np.unique(y[tr_idx]), y=y[tr_idx])]))
            sw_fold = sw[tr_idx] if sw is not None else None
            m = RandomForestClassifier(n_estimators=200, max_depth=10,
                                       min_samples_leaf=5, class_weight=cw,
                                       random_state=42, n_jobs=-1)
            m.fit(Xtr, y[tr_idx], sample_weight=sw_fold)
            wf_scores.append(m.score(Xte, y[te_idx]))
        wf_acc = np.mean(wf_scores)

        # Final model
        split = int(len(X) * 0.8)
        self.scaler.fit(X[:split])
        Xtr   = self.scaler.transform(X[:split])
        Xte   = self.scaler.transform(X[split:])
        cw    = dict(zip(*[np.unique(y[:split]),
                            compute_class_weight("balanced",
                            classes=np.unique(y[:split]), y=y[:split])]))
        sw_tr = sw[:split] if sw is not None else None
        new_model = RandomForestClassifier(
            n_estimators=500, max_depth=8, min_samples_leaf=8,
            min_samples_split=15, max_features="log2",
            class_weight=cw, random_state=42, n_jobs=-1,
        )
        new_model.fit(Xtr, y[:split], sample_weight=sw_tr)
        new_acc = new_model.score(Xte, y[split:])

        # Champion/challenger — only deploy if better
        if self.is_trained and self.metadata.get("test_accuracy", 0) > new_acc + 0.02:
            log.warning(f"RF new model ({new_acc:.2%}) worse — keeping old")
            self._load()  # restore in-memory state to match saved files
            return {"accuracy": self.metadata.get("test_accuracy", 0), "status": "kept_old"}

        self.model      = new_model
        self.is_trained = True
        self.metadata   = {
            "accuracy":     round(new_acc, 4),
            "wf_accuracy":  round(float(wf_acc), 4),
            "test_accuracy":round(new_acc, 4),
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
            return {"action": "HOLD", "confidence": 0.34, "probs": [0.33, 0.34, 0.33]}
        feat = make_features(df)
        if len(feat) == 0:
            return {"action": "HOLD", "confidence": 0.34, "probs": [0.33, 0.34, 0.33]}
        last   = feat.iloc[[-1]][self.feature_cols]
        probs  = self.model.predict_proba(self.scaler.transform(last.values))[0]
        full   = np.zeros(3)
        for i, c in enumerate(self.model.classes_):
            full[c] = probs[i]
        label = int(np.argmax(full))
        return {"action": {0:"SELL",1:"HOLD",2:"BUY"}[label],
                "confidence": round(float(full[label]), 4),
                "probs": full.tolist()}


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

    def train(self, df, use_decay_weights: bool = False, feat_df=None, labels_s=None):
        log.info("Training LightGBM (walk-forward)...")
        if feat_df is not None and labels_s is not None:
            feat   = feat_df
            labels = labels_s
        else:
            feat   = make_features(df)
            labels = make_labels(df).reindex(feat.index).dropna()
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
            class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1,
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

        if self.is_trained and self.metadata.get("test_accuracy", 0) > new_acc + 0.02:
            log.warning(f"LightGBM new ({new_acc:.2%}) worse — keeping old")
            self._load()  # restore in-memory state to match saved files
            return {"accuracy": self.metadata.get("test_accuracy", 0), "status": "kept_old"}

        self.model      = new_model
        self.is_trained = True
        self.metadata   = {
            "accuracy":      round(new_acc, 4),
            "wf_accuracy":   round(wf_acc, 4),
            "test_accuracy": round(new_acc, 4),
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
            return {"action": "HOLD", "confidence": 0.34, "probs": [0.33, 0.34, 0.33]}
        feat = make_features(df)
        if len(feat) == 0:
            return {"action": "HOLD", "confidence": 0.34, "probs": [0.33, 0.34, 0.33]}
        last  = feat.iloc[[-1]][self.feature_cols]
        probs = self.model.predict_proba(self.scaler.transform(last.values))[0]
        label = int(np.argmax(probs))
        return {"action": {0:"SELL",1:"HOLD",2:"BUY"}[label],
                "confidence": round(float(probs[label]), 4),
                "probs": list(probs)}


class LSTMStrategy:
    SEQ_LEN      = 20
    MIN_ACCURACY = 0.42  # Raised floor — anything below is overfitting or noise

    def __init__(self):
        self.model_path         = DATA / "lstm_model.keras"
        self.scaler_path        = DATA / "lstm_scaler.pkl"
        self.meta_path          = DATA / "lstm_meta.json"
        self.backup_path        = DATA / "lstm_model_backup.keras"
        self.backup_scaler_path = DATA / "lstm_scaler_backup.pkl"
        self.backup_meta_path   = DATA / "lstm_meta_backup.json"
        self.model              = None
        self.scaler             = MinMaxScaler()
        self.n_features         = None
        self.is_trained         = False
        self.metadata           = {}
        self._load()

    def _load(self):
        # Auto-restore from backup when main model file is missing or corrupted
        if not self.model_path.exists() and self.backup_path.exists():
            log.warning("LSTM model missing — restoring from backup")
            try:
                shutil.copy2(str(self.backup_path), str(self.model_path))
                if self.backup_scaler_path.exists():
                    shutil.copy2(str(self.backup_scaler_path), str(self.scaler_path))
                if self.backup_meta_path.exists():
                    shutil.copy2(str(self.backup_meta_path), str(self.meta_path))
                log.info("LSTM restored from backup")
            except Exception as e:
                log.error(f"LSTM backup restore failed: {e}")

        if self.model_path.exists():
            try:
                self.model      = load_model(str(self.model_path))
                self.scaler     = joblib.load(self.scaler_path)
                self.n_features = self.model.input_shape[-1]
                self.is_trained = True
                if self.meta_path.exists():
                    with open(self.meta_path) as f:
                        self.metadata = json.load(f)
                log.info(f"LSTM loaded (accuracy={self.metadata.get('accuracy','?')})")
            except Exception as e:
                log.warning(f"LSTM load failed: {e}")

    def _backup_current(self):
        """Copy current model to backup slot before overwriting."""
        if not self.model_path.exists():
            return
        try:
            shutil.copy2(str(self.model_path), str(self.backup_path))
            if self.scaler_path.exists():
                shutil.copy2(str(self.scaler_path), str(self.backup_scaler_path))
            if self.meta_path.exists():
                shutil.copy2(str(self.meta_path), str(self.backup_meta_path))
        except Exception as e:
            log.warning(f"LSTM backup failed: {e}")

    def train(self, df):
        log.info("Training LSTM...")
        feat   = make_features(df)
        labels = make_labels(df).reindex(feat.index).dropna()
        feat   = feat.loc[labels.index]
        if len(feat) < self.SEQ_LEN + 200:
            return {"error": "need more data for LSTM"}
        self.n_features = feat.shape[1]
        self.scaler.fit(feat.values)
        X_sc         = self.scaler.transform(feat.values)
        y            = labels.values.astype(int)
        X_seq, y_seq = make_sequences(X_sc, y, self.SEQ_LEN)
        split        = int(len(X_seq) * 0.8)
        val_split    = int(split * 0.9)

        cw_arr = compute_class_weight("balanced", classes=np.unique(y_seq[:val_split]), y=y_seq[:val_split])
        cw     = dict(enumerate(cw_arr))

        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True,
                               input_shape=(self.SEQ_LEN, self.n_features))),
            Dropout(0.4),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(3, activation="softmax"),
        ])
        model.compile(optimizer=Adam(0.001),
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        history = model.fit(
            X_seq[:val_split], y_seq[:val_split],
            validation_data=(X_seq[val_split:split], y_seq[val_split:split]),
            class_weight=cw,
            epochs=100, batch_size=32, verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6),
            ],
        )
        _, test_acc = model.evaluate(X_seq[split:], y_seq[split:], verbose=0)

        # Hard floor: worse than random baseline → discard immediately
        if test_acc < self.MIN_ACCURACY:
            log.warning(
                f"LSTM new ({test_acc:.2%}) below floor {self.MIN_ACCURACY:.2%} — discarding"
            )
            return {
                "accuracy":     self.metadata.get("test_accuracy", 0),
                "new_accuracy": round(float(test_acc), 4),
                "status":       "below_floor",
            }

        # Champion/challenger: metadata is authoritative regardless of is_trained flag
        stored_acc = self.metadata.get("test_accuracy", 0)
        if stored_acc > test_acc + 0.02:
            log.warning(f"LSTM new ({test_acc:.2%}) worse — keeping old")
            return {"accuracy": stored_acc, "status": "kept_old"}

        # Backup before overwriting, then atomic save via temp rename
        self._backup_current()
        tmp_path = self.model_path.with_name("lstm_model.tmp.keras")
        try:
            model.save(str(tmp_path))
            shutil.move(str(tmp_path), str(self.model_path))
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"LSTM save failed: {e}") from e

        joblib.dump(self.scaler, self.scaler_path)
        self.model      = model
        self.is_trained = True
        epochs_ran      = len(history.history["loss"])
        self.metadata   = {
            "accuracy":     round(float(test_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "epochs":       epochs_ran,
            "trained_at":   datetime.now(timezone.utc).isoformat(),
        }
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        log.info(f"LSTM trained! Accuracy={test_acc:.2%} Epochs={epochs_ran}")
        return self.metadata

    def predict(self, df):
        if not self.is_trained:
            return {"action": "HOLD", "confidence": 0.34, "probs": [0.33, 0.34, 0.33]}
        try:
            feat = make_features(df)
            if len(feat) < self.SEQ_LEN:
                return {"action": "HOLD", "confidence": 0.34, "probs": [0.33, 0.34, 0.33]}
            if feat.shape[1] != self.n_features:
                log.warning(
                    f"LSTM stale: make_features={feat.shape[1]} model={self.n_features}. "
                    f"Retrain LSTM."
                )
                return {"action": "HOLD", "confidence": 0.34, "probs": [0.33, 0.34, 0.33]}
            X_sc  = self.scaler.transform(feat.values)
            seq   = X_sc[-self.SEQ_LEN:].reshape(1, self.SEQ_LEN, self.n_features)
            probs = self.model.predict(seq, verbose=0)[0]
            label = int(np.argmax(probs))
            return {"action": {0:"SELL",1:"HOLD",2:"BUY"}[label],
                    "confidence": round(float(probs[label]), 4),
                    "probs": probs.tolist()}
        except Exception as e:
            log.error(f"LSTM predict error: {e}")
            return {"action": "HOLD", "confidence": 0.34, "probs": [0.33, 0.34, 0.33]}

    def predict_mc(self, df, n_samples: int = MC_SAMPLES):
        """
        MC Dropout inference — call model with training=True N times.
        Returns mean probs + per-class variance (epistemic uncertainty).
        Falls back to None on any failure so caller can use regular predict().
        """
        if not self.is_trained:
            return None
        try:
            feat = make_features(df)
            if len(feat) < self.SEQ_LEN:
                return None
            if feat.shape[1] != self.n_features:
                log.warning(
                    f"LSTM stale (MC): make_features={feat.shape[1]} model={self.n_features}. "
                    f"Retrain LSTM."
                )
                return None
            X_sc = self.scaler.transform(feat.values)
            seq  = X_sc[-self.SEQ_LEN:].reshape(1, self.SEQ_LEN, self.n_features)
            seq_t = tf.constant(seq, dtype=tf.float32)
            preds = np.array([
                self.model(seq_t, training=True).numpy()[0]
                for _ in range(n_samples)
            ])                              # (n_samples, 3)
            mean_probs  = preds.mean(axis=0)
            uncertainty = float(preds.var(axis=0).mean())
            label       = int(np.argmax(mean_probs))
            return {
                "action":      {0: "SELL", 1: "HOLD", 2: "BUY"}[label],
                "confidence":  round(float(mean_probs[label]), 4),
                "probs":       mean_probs.tolist(),
                "uncertainty": round(uncertainty, 6),
            }
        except Exception as e:
            log.warning(f"LSTM predict_mc error: {e}")
            return None

    def fine_tune(self, df: pd.DataFrame, n_epochs: int = 3) -> dict:
        """
        Low-LR gradient steps on recent data. Nudges weights without catastrophic forgetting.
        Recompiles with Adam(1e-5); weights preserved, optimizer state reset.
        """
        if not self.is_trained:
            return {"status": "not_trained"}
        try:
            feat   = make_features(df)
            labels = make_labels(df).reindex(feat.index).dropna()
            feat   = feat.loc[labels.index]
            if len(feat) < self.SEQ_LEN + 20:
                return {"status": "too_few_bars", "n": len(feat)}
            X_sc          = self.scaler.transform(feat.values)
            y             = labels.values.astype(int)
            X_seq, y_seq  = make_sequences(X_sc, y, self.SEQ_LEN)
            if len(X_seq) < 10:
                return {"status": "too_few_sequences", "n": len(X_seq)}
            self.model.compile(
                optimizer=Adam(1e-5),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            self.model.fit(X_seq, y_seq, epochs=n_epochs, batch_size=16, verbose=0)
            log.info(f"LSTM fine-tuned: {len(X_seq)} sequences, {n_epochs} epochs")
            return {"status": "ok", "sequences": len(X_seq), "epochs": n_epochs}
        except Exception as e:
            log.warning(f"LSTM fine_tune failed: {e}")
            return {"status": "failed", "error": str(e)}


class MetaModel:
    """
    Stacked meta-learner. Inputs: 9 base-model probs + volatility + trend_strength + vol_ratio.
    Replaces static weighted voting when trained. Falls back to ensemble if not trained.
    """

    def __init__(self):
        self.model_path  = DATA / "meta_model.pkl"
        self.scaler_path = DATA / "meta_scaler.pkl"
        self.meta_path   = DATA / "meta_meta.json"
        self.model       = None
        self.scaler      = StandardScaler()
        self.is_trained  = False
        self.metadata    = {}
        self._load()

    def _load(self):
        if self.model_path.exists():
            try:
                self.model      = joblib.load(self.model_path)
                self.scaler     = joblib.load(self.scaler_path)
                self.is_trained = True
                if self.meta_path.exists():
                    with open(self.meta_path) as f:
                        self.metadata = json.load(f)
                log.info(f"MetaModel loaded (accuracy={self.metadata.get('accuracy','?')})")
            except Exception as e:
                log.warning(f"MetaModel load failed: {e}")
                self.is_trained = False

    def train(self, meta_X: np.ndarray, y: np.ndarray):
        """Train stacker on OOF meta-features + labels."""
        mask   = ~(np.isnan(meta_X).any(axis=1) | np.isinf(meta_X).any(axis=1))
        meta_X = meta_X[mask]
        y      = y[mask]

        if len(y) < 30:
            return {"error": f"insufficient clean samples: {len(y)}"}

        split = max(1, int(len(meta_X) * 0.8))
        self.scaler.fit(meta_X[:split])
        X_sc = self.scaler.transform(meta_X)

        try:
            new_model = lgb.LGBMClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, num_leaves=15,
                class_weight="balanced", random_state=42, verbose=-1,
            )
            new_model.fit(X_sc[:split], y[:split])
        except Exception:
            from sklearn.linear_model import LogisticRegression
            new_model = LogisticRegression(
                C=1.0, class_weight="balanced", max_iter=500, random_state=42
            )
            new_model.fit(X_sc[:split], y[:split])

        new_acc = (new_model.score(X_sc[split:], y[split:])
                   if split < len(y) else new_model.score(X_sc, y))

        if self.is_trained and self.metadata.get("accuracy", 0) > new_acc + 0.02:
            log.warning(f"MetaModel new ({new_acc:.2%}) worse — keeping old")
            return {"accuracy": self.metadata.get("accuracy", 0), "status": "kept_old"}

        self.model      = new_model
        self.is_trained = True
        self.metadata   = {
            "accuracy":        round(float(new_acc), 4),
            "n_samples":       int(len(y)),
            "n_meta_features": int(meta_X.shape[1]),
            "trained_at":      datetime.now(timezone.utc).isoformat(),
        }
        joblib.dump(self.model,  self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        log.info(f"MetaModel trained! Accuracy={new_acc:.2%} n={len(y)} features={meta_X.shape[1]}")
        return self.metadata

    def predict_single(self, rf_probs, lgbm_probs, lstm_probs,
                       volatility, trend_strength, vol_ratio, tft_probs=None):
        """Predict for one bar. Returns dict or None on failure/mismatch."""
        if not self.is_trained:
            return None
        try:
            features = [*rf_probs, *lgbm_probs, *lstm_probs]
            if tft_probs is not None:
                features.extend(tft_probs)
            features.extend([volatility, trend_strength, vol_ratio])
            row = np.array([features])

            # Feature-count guard: use model's ground-truth n_features_in_ so stale
            # metadata (missing n_meta_features key) doesn't bypass the check.
            expected = (self.metadata.get("n_meta_features")
                        or getattr(self.model, "n_features_in_", None))
            if expected is not None and row.shape[1] != expected:
                log.warning(
                    f"MetaModel feature mismatch ({row.shape[1]} vs {expected}) "
                    f"— meta needs retrain (TFT added?). Ensemble fallback."
                )
                return None

            if np.isnan(row).any() or np.isinf(row).any():
                return None
            X_sc  = self.scaler.transform(row)
            probs = self.model.predict_proba(X_sc)[0]
            full  = np.zeros(3)
            for i, c in enumerate(self.model.classes_):
                full[int(c)] = probs[i]
            label = int(np.argmax(full))
            return {
                "action":     {0: "SELL", 1: "HOLD", 2: "BUY"}[label],
                "confidence": round(float(full[label]), 4),
                "probs":      full.tolist(),
            }
        except Exception as e:
            log.error(f"MetaModel predict error: {e}")
            return None


class AIStrategyEngine:
    def __init__(self):
        self.rf            = RandomForestStrategy()
        self.lgbm          = LightGBMStrategy()
        self.lstm          = LSTMStrategy()
        self.tft           = TFTStrategy() if _TFT_AVAILABLE else None
        self.meta          = MetaModel()
        self.online_buffer   = OnlineBuffer()
        self._unc_buffer: list = []   # uncertainty log write buffer (Step 7)
        self.performance_log = []
        p = DATA / "trade_results.json"
        if p.exists():
            with open(p) as f:
                self.performance_log = json.load(f)

    def train_all(self, df, feat_df=None, labels_s=None, ctx_arrays=None, use_decay_weights: bool = False):
        r = {}
        r["rf"]   = self.rf.train(df, feat_df=feat_df, labels_s=labels_s, use_decay_weights=use_decay_weights)
        r["lgbm"] = self.lgbm.train(df, feat_df=feat_df, labels_s=labels_s, use_decay_weights=use_decay_weights)
        r["lstm"] = self.lstm.train(df)
        if self.tft is not None:
            r["tft"] = self.tft.train(df)
        r["meta"] = self._train_meta(df)
        log.info(
            f"All models trained! RF={r['rf'].get('accuracy','?')} "
            f"LGBM={r['lgbm'].get('accuracy','?')} "
            f"LSTM={r['lstm'].get('accuracy','?')} "
            f"TFT={r.get('tft',{}).get('accuracy','N/A')} "
            f"Meta={r['meta'].get('accuracy','?')}"
        )
        return r

    @staticmethod
    def _compute_context_features(df):
        """Return (volatility, trend_strength, vol_ratio) arrays aligned to df rows."""
        close          = df["close"]
        vol            = df["volume"]
        ret            = close.pct_change()
        volatility     = ret.rolling(20).std().values
        ema9           = close.ewm(span=9).mean().values
        ema50          = close.ewm(span=50).mean().values
        trend_strength = np.abs(ema9 - ema50) / (ema50 + 1e-9)
        vol_ma         = vol.rolling(20).mean().values
        vol_ratio      = vol.values / (vol_ma + 1e-9)
        return volatility, trend_strength, vol_ratio

    def _train_meta(self, df):
        """5-fold TimeSeriesSplit OOF — fresh base models per fold, no data leakage."""
        log.info("Training MetaModel (5-fold OOF, no leakage)...")
        try:
            feat   = make_features(df)
            labels = make_labels(df).reindex(feat.index).dropna()
            feat   = feat.loc[labels.index]
            if len(feat) < 400:
                log.warning("MetaModel OOF: need 400+ rows")
                return {"error": "need 400+ rows for OOF"}

            df_aligned = df.reindex(feat.index)
            X  = feat.values
            y  = labels.values.astype(int)
            n  = len(X)

            tscv     = TimeSeriesSplit(n_splits=5)
            oof_rf   = np.full((n, 3), 1/3)
            oof_lgbm = np.full((n, 3), 1/3)
            oof_lstm = np.full((n, 3), 1/3)
            oof_mask = np.zeros(n, dtype=bool)

            for fold_i, (tr_idx, val_idx) in enumerate(tscv.split(X)):
                log.info(
                    f"MetaModel OOF fold {fold_i+1}/5  "
                    f"train={len(tr_idx)}  val={len(val_idx)}"
                )

                # ── RF fold ──────────────────────────────────────────────────
                try:
                    sc_rf   = StandardScaler()
                    Xtr_rf  = sc_rf.fit_transform(X[tr_idx])
                    Xval_rf = sc_rf.transform(X[val_idx])
                    cw_rf   = dict(zip(*[np.unique(y[tr_idx]),
                                         compute_class_weight("balanced",
                                         classes=np.unique(y[tr_idx]),
                                         y=y[tr_idx])]))
                    m_rf = RandomForestClassifier(
                        n_estimators=200, max_depth=8, min_samples_leaf=8,
                        class_weight=cw_rf,
                        random_state=42 + fold_i, n_jobs=-1,
                    )
                    m_rf.fit(Xtr_rf, y[tr_idx])
                    raw = m_rf.predict_proba(Xval_rf)
                    p   = np.zeros((len(val_idx), 3))
                    for ci, c in enumerate(m_rf.classes_):
                        p[:, int(c)] = raw[:, ci]
                    oof_rf[val_idx] = p
                except Exception as e:
                    log.warning(f"RF OOF fold {fold_i+1}: {e}")

                # ── LGBM fold ─────────────────────────────────────────────────
                try:
                    sc_lgbm   = StandardScaler()
                    Xtr_lgbm  = sc_lgbm.fit_transform(X[tr_idx])
                    Xval_lgbm = sc_lgbm.transform(X[val_idx])
                    vsp       = int(len(tr_idx) * 0.85)
                    m_lgbm    = lgb.LGBMClassifier(
                        n_estimators=200, max_depth=5, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8, num_leaves=15,
                        class_weight="balanced",
                        random_state=42 + fold_i, verbose=-1, n_jobs=-1,
                    )
                    m_lgbm.fit(
                        Xtr_lgbm[:vsp], y[tr_idx[:vsp]],
                        eval_set=[(Xtr_lgbm[vsp:], y[tr_idx[vsp:]])],
                        callbacks=[lgb.early_stopping(20, verbose=False),
                                   lgb.log_evaluation(period=-1)],
                    )
                    raw = m_lgbm.predict_proba(Xval_lgbm)
                    p   = np.zeros((len(val_idx), 3))
                    for ci, c in enumerate(m_lgbm.classes_):
                        p[:, int(c)] = raw[:, ci]
                    oof_lgbm[val_idx] = p
                except Exception as e:
                    log.warning(f"LGBM OOF fold {fold_i+1}: {e}")

                # ── LSTM fold (lightweight) ───────────────────────────────────
                seq_len = self.lstm.SEQ_LEN
                if len(tr_idx) > seq_len + 50:
                    try:
                        sc_l = MinMaxScaler()
                        sc_l.fit(X[tr_idx])
                        X_sc  = sc_l.transform(X)  # fit on train, apply to all
                        X_seq_tr, y_seq_tr = make_sequences(
                            X_sc[tr_idx], y[tr_idx], seq_len
                        )
                        if len(X_seq_tr) >= 20:
                            m_lstm = Sequential([
                                Bidirectional(LSTM(32, return_sequences=False,
                                                  input_shape=(seq_len, X.shape[1]))),
                                Dropout(0.3),
                                Dense(3, activation="softmax"),
                            ])
                            m_lstm.compile(
                                optimizer=Adam(0.001),
                                loss="sparse_categorical_crossentropy",
                            )
                            m_lstm.fit(
                                X_seq_tr, y_seq_tr,
                                epochs=20, batch_size=32, verbose=0,
                                callbacks=[EarlyStopping(patience=5,
                                           restore_best_weights=True)],
                            )
                            for vi in val_idx:
                                if vi >= seq_len:
                                    seq = X_sc[vi - seq_len:vi].reshape(
                                        1, seq_len, X.shape[1])
                                    oof_lstm[vi] = m_lstm.predict(seq, verbose=0)[0]
                            del m_lstm
                    except Exception as e:
                        log.warning(f"LSTM OOF fold {fold_i+1}: {e}")

                oof_mask[val_idx] = True

            # ── TFT OOF — single-pass (too expensive to fold) ────────────────
            tft_probs = None
            if self.tft is not None and self.tft.is_trained:
                try:
                    import torch
                    seq_tft  = self.tft.SEQ_LEN
                    X_sc_tft = self.tft.scaler.transform(X)
                    tft_probs = np.full((n, 3), 1/3)
                    self.tft.model.eval()
                    with torch.no_grad():
                        for i in range(seq_tft, n):
                            s = torch.FloatTensor(
                                X_sc_tft[i - seq_tft:i]).unsqueeze(0)
                            tft_probs[i] = torch.softmax(
                                self.tft.model(s), dim=-1)[0].numpy()
                except Exception as e:
                    log.warning(f"TFT OOF: {e}")
                    tft_probs = None

            # ── Context features ──────────────────────────────────────────────
            ctx_vol, ctx_trend, ctx_vratio = self._compute_context_features(df_aligned)
            ctx_vol    = np.nan_to_num(ctx_vol)
            ctx_trend  = np.nan_to_num(ctx_trend)
            ctx_vratio = np.nan_to_num(ctx_vratio, nan=1.0)

            parts = [oof_rf, oof_lgbm, oof_lstm]
            if tft_probs is not None:
                parts.append(tft_probs)
            parts += [
                ctx_vol.reshape(-1, 1),
                ctx_trend.reshape(-1, 1),
                ctx_vratio.reshape(-1, 1),
            ]
            meta_X = np.column_stack(parts)

            n_valid = int(oof_mask.sum())
            if n_valid < 50:
                return {"error": f"too few OOF rows: {n_valid}"}

            result = self.meta.train(meta_X[oof_mask], y[oof_mask])
            log.info(f"MetaModel 5-fold OOF: {result}")
            return result

        except Exception as e:
            log.error(f"MetaModel training failed: {e}", exc_info=True)
            return {"error": str(e)}

    def _get_dynamic_weights(self):
        """Weight each model by walk-forward accuracy; floor at 10% so no model is ignored."""
        rf_wf    = max(self.rf.metadata.get("wf_accuracy",   0.33), 0.10)
        lgbm_wf  = max(self.lgbm.metadata.get("wf_accuracy", 0.33), 0.10)
        lstm_acc = max(self.lstm.metadata.get("accuracy",    0.33), 0.10)
        raw = np.array([rf_wf, lgbm_wf, lstm_acc], dtype=float)
        w   = raw / raw.sum()
        return {"rf": float(w[0]), "lgbm": float(w[1]), "lstm": float(w[2])}

    def get_model_health(self):
        h = {
            "rf_accuracy":   self.rf.metadata.get("accuracy", 0),
            "rf_wf":         self.rf.metadata.get("wf_accuracy", 0),
            "lgbm_accuracy": self.lgbm.metadata.get("accuracy", 0),
            "lstm_accuracy": self.lstm.metadata.get("accuracy", 0),
            "meta_accuracy": self.meta.metadata.get("accuracy", 0),
            "meta_trained":  self.meta.is_trained,
            "rf_trained":    self.rf.metadata.get("trained_at", "never"),
            "lstm_epochs":   self.lstm.metadata.get("epochs", 0),
        }
        if self.tft is not None:
            h["tft_accuracy"] = self.tft.metadata.get("accuracy", 0)
            h["tft_trained"]  = self.tft.is_trained
        return h

    def predict(self, df, symbol, regime: str = "RANGING"):
        rf_p   = self.rf.predict(df)
        lgbm_p = self.lgbm.predict(df)

        # MC Dropout for LSTM: use mc result when available, fall back to regular
        lstm_mc = self.lstm.predict_mc(df)
        lstm_p  = lstm_mc if lstm_mc is not None else self.lstm.predict(df)
        lstm_uncertainty = lstm_mc["uncertainty"] if lstm_mc is not None else 0.0

        # TFT: MC Dropout when trained
        if self.tft is not None and self.tft.is_trained:
            tft_mc = self.tft.predict_mc(df)
            tft_p  = tft_mc if tft_mc is not None else self.tft.predict(df)
            tft_uncertainty = tft_mc["uncertainty"] if tft_mc is not None else 0.0
        elif self.tft is not None:
            tft_p           = self.tft.predict(df)
            tft_uncertainty = 0.0
        else:
            tft_p           = None
            tft_uncertainty = 0.0

        # Ensemble variance — disagreement between model families (epistemic uncertainty)
        all_probs = [rf_p["probs"], lgbm_p["probs"], lstm_p["probs"]]
        if tft_p is not None:
            all_probs.append(tft_p["probs"])
        ensemble_var = float(np.var(np.array(all_probs, dtype=float), axis=0).mean())

        # MC uncertainty: mean over available MC components only
        mc_vals = [v for v in [lstm_uncertainty, tft_uncertainty] if v > 0]
        mc_uncertainty = float(np.mean(mc_vals)) if mc_vals else 0.0

        # Combined = max(ensemble_var, mc_uncertainty) — worst signal wins, no dilution
        combined_uncertainty = max(ensemble_var, mc_uncertainty)

        # Context features for last bar (tail ensures rolling window is stable)
        tail = df.tail(30)
        ctx_vol, ctx_trend, ctx_vratio = self._compute_context_features(tail)

        def _safe(arr, default=0.0):
            v = float(arr[-1])
            return default if (np.isnan(v) or np.isinf(v)) else v

        # Only pass TFT probs when TFT is actually trained (avoids scaler dimension mismatch)
        tft_probs_for_meta = (
            tft_p["probs"]
            if (self.tft is not None and self.tft.is_trained and tft_p is not None)
            else None
        )
        meta_result = self.meta.predict_single(
            rf_probs       = rf_p["probs"],
            lgbm_probs     = lgbm_p["probs"],
            lstm_probs     = lstm_p["probs"],
            volatility     = _safe(ctx_vol),
            trend_strength = _safe(ctx_trend),
            vol_ratio      = _safe(ctx_vratio, default=1.0),
            tft_probs      = tft_probs_for_meta,
        )

        # Bypass meta if it's performing worse than base models
        meta_acc = self.meta.metadata.get("accuracy", 0)
        if meta_result is not None and meta_acc < 0.50:
            log.debug(
                f"Meta bypass: accuracy={meta_acc:.2%} < 50% — "
                f"using weighted ensemble instead"
            )
            meta_result = None

        tft_tag = f"+TFT:{tft_p['action']}" if tft_p is not None else ""

        if meta_result is not None:
            action = meta_result["action"]
            conf   = meta_result["confidence"]
            avg    = np.array(meta_result["probs"])
            strat  = (f"META+RF:{rf_p['action']}+LGBM:{lgbm_p['action']}"
                      f"+LSTM:{lstm_p['action']}{tft_tag}")
        else:
            # Fallback: dynamic weighted ensemble (original logic)
            W   = self._get_dynamic_weights()
            avg = (np.array(rf_p["probs"])   * W["rf"] +
                   np.array(lgbm_p["probs"]) * W["lgbm"] +
                   np.array(lstm_p["probs"]) * W["lstm"])
            if tft_p is not None:
                # Step 4: regime-aware TFT weight — higher in trending markets
                tft_w = 0.35 if regime == "TRENDING" else 0.15
                avg   = avg * (1.0 - tft_w) + np.array(tft_p["probs"]) * tft_w
            avg   /= avg.sum()
            label  = int(np.argmax(avg))
            conf   = float(avg[label])
            action = {0:"SELL",1:"HOLD",2:"BUY"}[label] if conf >= 0.40 else "HOLD"
            strat  = (f"ENS+RF:{rf_p['action']}+LGBM:{lgbm_p['action']}"
                      f"+LSTM:{lstm_p['action']}{tft_tag}")

        # Safety gate: require ≥2 base models to agree
        base_actions = [rf_p["action"], lgbm_p["action"], lstm_p["action"]]
        if tft_p is not None:
            base_actions.append(tft_p["action"])
        buy_votes  = base_actions.count("BUY")
        sell_votes = base_actions.count("SELL")
        if action == "BUY"  and buy_votes  < 2: action = "HOLD"
        if action == "SELL" and sell_votes < 2: action = "HOLD"

        # Uncertainty gate: high disagreement / MC variance → HOLD
        if combined_uncertainty > UNCERTAINTY_THRESHOLD and action != "HOLD":
            log.debug(
                f"{symbol} HOLD — uncertainty {combined_uncertainty:.4f} > "
                f"{UNCERTAINTY_THRESHOLD} (ens_var={ensemble_var:.4f} mc={mc_uncertainty:.4f})"
            )
            action = "HOLD"
            strat  = f"UNCERTAIN({combined_uncertainty:.3f})+" + strat

        ts = datetime.now(timezone.utc).isoformat()

        # Step 7: buffer uncertainty entry for periodic disk flush
        self._unc_buffer.append({
            "symbol":           symbol,
            "timestamp":        ts,
            "action":           action,
            "confidence":       round(float(conf), 4),
            "uncertainty":      round(combined_uncertainty, 4),
            "ensemble_var":     round(ensemble_var, 4),
            "mc_uncertainty":   round(mc_uncertainty, 4),
            "gated":            combined_uncertainty > UNCERTAINTY_THRESHOLD,
            "regime":           regime,
        })
        if len(self._unc_buffer) >= 50:
            self._flush_uncertainty_log()

        return {
            "symbol":     symbol,
            "action":     action,
            "confidence": round(float(conf), 4),
            "strategy":   strat,
            "timeframe":  "AI-v3",
            "indicators": {
                "rf_conf":            rf_p["confidence"],
                "lgbm_conf":          lgbm_p["confidence"],
                "lstm_conf":          lstm_p["confidence"],
                "tft_conf":           tft_p["confidence"] if tft_p else None,
                "rf_action":          rf_p["action"],
                "lgbm_action":        lgbm_p["action"],
                "lstm_action":        lstm_p["action"],
                "tft_action":         tft_p["action"] if tft_p else None,
                "buy_votes":          buy_votes,
                "sell_votes":         sell_votes,
                "avg_probs":          avg.tolist(),
                "meta_used":          meta_result is not None,
                "uncertainty":        round(combined_uncertainty, 4),
                "ensemble_var":       round(ensemble_var, 4),
                "mc_uncertainty":     round(mc_uncertainty, 4),
                "uncertainty_gated":  combined_uncertainty > UNCERTAINTY_THRESHOLD,
            },
            "timestamp": ts,
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
        Incrementally retrain base models on rolling buffered data with decay weights.
        Step 6 gates: requires 50+ closed trades AND 52%+ win rate before updating.
        Triggered every OnlineBuffer.UPDATE_EVERY new bars.
        """
        if not self.online_buffer.should_update():
            return {"status": "skipped"}

        # Step 6: safety gates — only retrain with sufficient proven performance
        total_trades = len(self.performance_log)
        if total_trades < 50:
            log.debug(f"Online learning skipped: {total_trades}/50 trades")
            return {"status": "skipped", "reason": f"only {total_trades}/50 trades"}
        wins     = sum(1 for t in self.performance_log if t.get("pnl", 0) > 0)
        win_rate = wins / total_trades
        if win_rate < 0.52:
            log.debug(f"Online learning skipped: win_rate {win_rate:.2%} < 52%")
            return {"status": "skipped", "reason": f"win_rate {win_rate:.2%} < 52%"}

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

        try:
            results["lstm"] = self.lstm.fine_tune(combined)
        except Exception as e:
            results["lstm"] = {"error": str(e)}
            log.warning(f"Online LSTM fine-tune failed: {e}")

        try:
            results["meta"] = self._train_meta(combined)
        except Exception as e:
            results["meta"] = {"error": str(e)}
            log.warning(f"Online meta retrain failed: {e}")

        log.info(
            f"Online update done — RF={results.get('rf',{}).get('accuracy','?')} "
            f"LGBM={results.get('lgbm',{}).get('accuracy','?')} "
            f"LSTM={results.get('lstm',{}).get('status','?')}"
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
