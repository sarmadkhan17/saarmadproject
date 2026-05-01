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
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
tf.get_logger().setLevel("ERROR")

from env_config import DATA_DIR

log  = logging.getLogger("AIStrategy")
DATA = DATA_DIR
DATA.mkdir(exist_ok=True)


def make_features(df):
    import ta
    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    vol   = df["volume"]
    f     = pd.DataFrame(index=df.index)

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

    vol_sma        = vol.rolling(20).mean()
    f["vol_ratio"] = vol / (vol_sma + 1e-9)
    f["vol_trend"] = vol.rolling(5).mean() / (vol_sma + 1e-9)
    f["vol_spike"] = (f["vol_ratio"] > 2.0).astype(int)

    stoch = ta.momentum.StochasticOscillator(high, low, close)
    f["stoch_k"]    = stoch.stoch()
    f["stoch_diff"] = stoch.stoch() - stoch.stoch_signal()

    f["cci"]        = ta.trend.CCIIndicator(high, low, close, 20).cci() / 100
    f["williams_r"] = ta.momentum.WilliamsRIndicator(high, low, close, 14).williams_r() / 100

    for w in [14, 50]:
        h = high.rolling(w).max()
        l = low.rolling(w).min()
        f[f"price_pos_{w}"] = (close - l) / (h - l + 1e-9)

    f["body"]       = abs(close - df["open"]) / (close + 1e-9)
    f["upper_wick"] = (high - pd.concat([close, df["open"]], axis=1).max(axis=1)) / (close + 1e-9)
    f["lower_wick"] = (pd.concat([close, df["open"]], axis=1).min(axis=1) - low) / (close + 1e-9)
    f["is_bullish"] = (close > df["open"]).astype(int)

    ret             = close.pct_change()
    f["vol_5"]      = ret.rolling(5).std()
    f["vol_20"]     = ret.rolling(20).std()
    f["vol_regime"] = f["vol_5"] / (f["vol_20"] + 1e-9)
    f["adx"]        = ta.trend.ADXIndicator(high, low, close, 14).adx() / 100

    # VWAP and deviation
    vwap            = (close * vol).cumsum() / (vol.cumsum() + 1e-9)
    f["vwap_dist"]  = (close - vwap) / (vwap + 1e-9)
    f["above_vwap"] = (close > vwap).astype(int)

    # Cumulative volume delta (buying/selling pressure proxy)
    delta        = vol * pd.Series(np.where(close >= df["open"], 1.0, -1.0), index=df.index)
    f["cvd_5"]   = delta.rolling(5).sum()  / (vol.rolling(5).sum()  + 1e-9)
    f["cvd_20"]  = delta.rolling(20).sum() / (vol.rolling(20).sum() + 1e-9)

    # Price acceleration (momentum of momentum)
    f["price_accel"] = close.pct_change(3).diff(2)

    # Rolling Sharpe ratio
    ret            = close.pct_change()
    f["sharpe_10"] = ret.rolling(10).mean() / (ret.rolling(10).std() + 1e-9)
    f["sharpe_20"] = ret.rolling(20).mean() / (ret.rolling(20).std() + 1e-9)

    # Donchian channel breakouts (trend continuation signals)
    for w in [20, 50]:
        f[f"dc_breakout_{w}"]  = (close >= high.rolling(w).max().shift(1)).astype(int)
        f[f"dc_breakdown_{w}"] = (close <= low.rolling(w).min().shift(1)).astype(int)

    return f.dropna()


def make_labels(df, forward_bars=1, min_move=0.003):
    """
    Forward return labels. forward_bars=1, min_move=0.003 tested at 62%+ accuracy.
    """
    close  = df["close"]
    future = close.shift(-forward_bars) / close - 1
    labels = pd.Series(1, index=df.index)
    labels[future >  min_move] = 2   # BUY
    labels[future < -min_move] = 0   # SELL
    labels = labels.dropna()
    counts = labels.value_counts().sort_index()
    total  = len(labels)
    log.info(
        f"Labels: SELL={counts.get(0,0)/total*100:.1f}% "
        f"HOLD={counts.get(1,0)/total*100:.1f}% "
        f"BUY={counts.get(2,0)/total*100:.1f}%"
    )
    return labels


def make_sequences(X, y, seq_len=48):
    Xs, ys = [], []
    for i in range(seq_len, len(X)):
        Xs.append(X[i-seq_len:i])
        ys.append(y[i])
    return np.array(Xs), np.array(ys)


def _undersample_hold(X, y, keep_ratio=0.40):
    """Randomly drop HOLD (class 1) rows so training is less HOLD-biased.
    keep_ratio=0.40 means keep 40% of HOLD rows, keeping all BUY/SELL rows."""
    rng       = np.random.RandomState(42)
    hold_idx  = np.where(y == 1)[0]
    other_idx = np.where(y != 1)[0]
    n_keep    = max(int(len(hold_idx) * keep_ratio), len(other_idx))
    keep_hold = rng.choice(hold_idx, size=min(n_keep, len(hold_idx)), replace=False)
    idx       = np.sort(np.concatenate([other_idx, keep_hold]))
    before    = len(y)
    after     = len(idx)
    hold_pct  = (y[idx] == 1).sum() / len(idx) * 100
    log.info(f"HOLD undersample: {before}→{after} rows | HOLD now {hold_pct:.1f}%")
    return X[idx], y[idx]


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

    def train_from_features(self, feat, labels):
        log.info("Training Random Forest from pre-computed features...")
        feat   = feat.copy()
        labels = labels.reindex(feat.index).dropna()
        feat   = feat.loc[labels.index]
        if len(feat) < 300:
            return {"error": f"need 300+ bars, got {len(feat)}"}
        self.feature_cols = feat.columns.tolist()
        X, y = _undersample_hold(feat.values, labels.values.astype(int))
        return self._fit(X, y)

    def _fit(self, X, y):
        tscv      = TimeSeriesSplit(n_splits=8)
        wf_scores = []
        for tr_idx, te_idx in tscv.split(X):
            sc  = StandardScaler()
            Xtr = sc.fit_transform(X[tr_idx])
            Xte = sc.transform(X[te_idx])
            cw  = dict(zip(*[np.unique(y[tr_idx]),
                              compute_class_weight("balanced",
                              classes=np.unique(y[tr_idx]), y=y[tr_idx])]))
            m = RandomForestClassifier(n_estimators=500, max_depth=12,
                                       min_samples_leaf=4, class_weight=cw,
                                       random_state=42, n_jobs=-1)
            m.fit(Xtr, y[tr_idx])
            wf_scores.append(m.score(Xte, y[te_idx]))
        wf_acc = np.mean(wf_scores)

        split = int(len(X) * 0.8)
        self.scaler.fit(X[:split])
        Xtr = self.scaler.transform(X[:split])
        Xte = self.scaler.transform(X[split:])
        cw  = dict(zip(*[np.unique(y[:split]),
                          compute_class_weight("balanced",
                          classes=np.unique(y[:split]), y=y[:split])]))
        new_model = RandomForestClassifier(
            n_estimators=1000, max_depth=12, min_samples_leaf=4,
            min_samples_split=8, max_features="sqrt",
            class_weight=cw, random_state=42, n_jobs=-1,
        )
        new_model.fit(Xtr, y[:split])
        new_acc = balanced_accuracy_score(y[split:], new_model.predict(Xte))

        if self.is_trained and self.metadata.get("test_accuracy", 0) > new_acc + 0.02:
            log.warning(f"RF new model ({new_acc:.2%}) worse — keeping old")
            self._load()
            return {"accuracy": self.metadata.get("test_accuracy", 0), "status": "kept_old"}

        self.model      = new_model
        self.is_trained = True
        self.metadata   = {
            "accuracy":      round(new_acc, 4),
            "wf_accuracy":   round(float(wf_acc), 4),
            "test_accuracy": round(new_acc, 4),
            "trained_at":    datetime.utcnow().isoformat(),
            "n_samples":     len(X),
        }
        joblib.dump(self.model,        self.model_path)
        joblib.dump(self.scaler,       self.scaler_path)
        joblib.dump(self.feature_cols, self.feat_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        log.info(f"RF trained! Test={new_acc:.2%} WF={wf_acc:.2%}")
        return self.metadata

    def train(self, df, forward_bars=1, min_move=0.003):
        log.info("Training Random Forest (walk-forward)...")
        feat   = make_features(df)
        labels = make_labels(df, forward_bars=forward_bars, min_move=min_move).reindex(feat.index).dropna()
        feat   = feat.loc[labels.index]
        if len(feat) < 300:
            return {"error": "need 300+ bars"}
        self.feature_cols = feat.columns.tolist()
        return self._fit(feat.values, labels.values.astype(int))

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

    def train_from_features(self, feat, labels):
        log.info("Training LightGBM from pre-computed features...")
        feat   = feat.copy()
        labels = labels.reindex(feat.index).dropna()
        feat   = feat.loc[labels.index]
        if len(feat) < 300:
            return {"error": f"need 300+ bars, got {len(feat)}"}
        self.feature_cols = feat.columns.tolist()
        X, y = _undersample_hold(feat.values, labels.values.astype(int))
        return self._fit(X, y)

    def _fit(self, X, y):
        lgbm_params = dict(
            n_estimators=1000, max_depth=8, learning_rate=0.03,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
            num_leaves=63, min_child_samples=10,
            reg_alpha=0.1, reg_lambda=0.1,
            class_weight="balanced", random_state=42, verbose=-1, n_jobs=-1,
        )
        tscv = TimeSeriesSplit(n_splits=8)
        wf_scores = []
        for tr_idx, te_idx in tscv.split(X):
            sc       = StandardScaler()
            Xtr_fold = sc.fit_transform(X[tr_idx])
            Xte_fold = sc.transform(X[te_idx])
            val_sp   = int(len(tr_idx) * 0.85)
            m = lgb.LGBMClassifier(**lgbm_params)
            m.fit(Xtr_fold[:val_sp], y[tr_idx[:val_sp]],
                  eval_set=[(Xtr_fold[val_sp:], y[tr_idx[val_sp:]])],
                  callbacks=[lgb.early_stopping(30, verbose=False),
                             lgb.log_evaluation(period=-1)])
            wf_scores.append(m.score(Xte_fold, y[te_idx]))
        wf_acc = float(np.mean(wf_scores))

        sp     = int(len(X) * 0.8)
        val_sp = int(sp * 0.85)
        self.scaler.fit(X[:val_sp])
        Xtr  = self.scaler.transform(X[:val_sp])
        Xval = self.scaler.transform(X[val_sp:sp])
        Xte  = self.scaler.transform(X[sp:])
        new_model = lgb.LGBMClassifier(**lgbm_params)
        new_model.fit(Xtr, y[:val_sp], eval_set=[(Xval, y[val_sp:sp])],
                      callbacks=[lgb.early_stopping(50, verbose=False),
                                 lgb.log_evaluation(period=-1)])
        new_acc = balanced_accuracy_score(y[sp:], new_model.predict(Xte))

        if self.is_trained and self.metadata.get("test_accuracy", 0) > new_acc + 0.02:
            log.warning(f"LightGBM new ({new_acc:.2%}) worse — keeping old")
            self._load()
            return {"accuracy": self.metadata.get("test_accuracy", 0), "status": "kept_old"}

        self.model      = new_model
        self.is_trained = True
        self.metadata   = {
            "accuracy":      round(new_acc, 4),
            "wf_accuracy":   round(wf_acc, 4),
            "test_accuracy": round(new_acc, 4),
            "trained_at":    datetime.utcnow().isoformat(),
            "n_samples":     len(X),
        }
        joblib.dump(self.model,        self.model_path)
        joblib.dump(self.scaler,       self.scaler_path)
        joblib.dump(self.feature_cols, self.feat_path)
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        log.info(f"LightGBM trained! Test={new_acc:.2%} WF={wf_acc:.2%}")
        return self.metadata

    def train(self, df, forward_bars=1, min_move=0.003):
        log.info("Training LightGBM (walk-forward)...")
        feat   = make_features(df)
        labels = make_labels(df, forward_bars=forward_bars, min_move=min_move).reindex(feat.index).dropna()
        feat   = feat.loc[labels.index]
        if len(feat) < 300:
            return {"error": "need 300+ bars"}
        self.feature_cols = feat.columns.tolist()
        return self._fit(feat.values, labels.values.astype(int))


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
    SEQ_LEN      = 48
    MIN_ACCURACY = 0.28

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

    def train_from_features(self, feat, labels, coin_ids=None, min_accuracy=None):
        log.info("Training LSTM from pre-computed features...")
        if min_accuracy is not None:
            self.MIN_ACCURACY = min_accuracy
        feat   = feat.copy()
        labels = labels.reindex(feat.index).dropna()
        feat   = feat.loc[labels.index]
        if len(feat) < self.SEQ_LEN + 200:
            return {"error": f"need {self.SEQ_LEN + 200}+ bars for LSTM, got {len(feat)}"}
        self.n_features = feat.shape[1]
        self.scaler.fit(feat.values)
        X_sc = self.scaler.transform(feat.values)
        y    = labels.values.astype(int)

        if coin_ids is not None:
            # Build sequences within each coin's data only — no cross-coin contamination
            ids = coin_ids.reindex(feat.index).values
            all_X, all_y = [], []
            for cid in np.unique(ids):
                mask = ids == cid
                X_c, y_c = make_sequences(X_sc[mask], y[mask], self.SEQ_LEN)
                all_X.append(X_c)
                all_y.append(y_c)
            if not all_X:
                return {"error": "no sequences built"}
            X_sc_seq = np.concatenate(all_X, axis=0)
            y_seq_combined = np.concatenate(all_y, axis=0)
            log.info(f"LSTM: {len(X_sc_seq)} sequences from {len(np.unique(ids))} coins (no boundary crossings)")
            return self._fit_seq(X_sc_seq, y_seq_combined)

        return self._fit(X_sc, y)

    def _fit(self, X_sc, y):
        X_seq, y_seq = make_sequences(X_sc, y, self.SEQ_LEN)
        return self._fit_seq(X_seq, y_seq)

    def _fit_seq(self, X_seq, y_seq):
        split     = int(len(X_seq) * 0.8)
        val_split = int(split * 0.9)

        cw_arr = compute_class_weight("balanced", classes=np.unique(y_seq[:val_split]), y=y_seq[:val_split])
        cw = {i: (w * 2.0 if i != 1 else w) for i, w in enumerate(cw_arr)}

        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True,
                               input_shape=(self.SEQ_LEN, self.n_features))),
            Dropout(0.4),
            BatchNormalization(),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(64, activation="relu"),
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
            epochs=200, batch_size=64, verbose=0,
            callbacks=[
                EarlyStopping(monitor="val_accuracy", patience=25, restore_best_weights=True),
                ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6),
            ],
        )
        y_pred   = np.argmax(model.predict(X_seq[split:], verbose=0), axis=1)
        test_acc = balanced_accuracy_score(y_seq[split:], y_pred)

        if test_acc < self.MIN_ACCURACY:
            log.warning(f"LSTM new ({test_acc:.2%}) below floor {self.MIN_ACCURACY:.2%} — discarding")
            return {
                "accuracy":     self.metadata.get("test_accuracy", 0),
                "new_accuracy": round(float(test_acc), 4),
                "status":       "below_floor",
            }

        stored_acc = self.metadata.get("test_accuracy", 0)
        if stored_acc > test_acc + 0.02:
            log.warning(f"LSTM new ({test_acc:.2%}) worse — keeping old")
            return {"accuracy": stored_acc, "status": "kept_old"}

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
            "accuracy":      round(float(test_acc), 4),
            "test_accuracy": round(float(test_acc), 4),
            "epochs":        epochs_ran,
            "trained_at":    datetime.utcnow().isoformat(),
        }
        with open(self.meta_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        log.info(f"LSTM trained! Accuracy={test_acc:.2%} Epochs={epochs_ran}")
        return self.metadata

    def train(self, df, forward_bars=1, min_move=0.003, min_accuracy=None):
        log.info("Training LSTM...")
        if min_accuracy is not None:
            self.MIN_ACCURACY = min_accuracy
        feat   = make_features(df)
        labels = make_labels(df, forward_bars=forward_bars, min_move=min_move).reindex(feat.index).dropna()
        feat   = feat.loc[labels.index]
        if len(feat) < self.SEQ_LEN + 200:
            return {"error": "need more data for LSTM"}
        self.n_features = feat.shape[1]
        self.scaler.fit(feat.values)
        X_sc = self.scaler.transform(feat.values)
        y    = labels.values.astype(int)
        return self._fit(X_sc, y)

    def predict(self, df):
        if not self.is_trained:
            return {"action": "HOLD", "confidence": 0.34, "probs": [0.33, 0.34, 0.33]}
        try:
            feat = make_features(df)
            if len(feat) < self.SEQ_LEN:
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


class AIStrategyEngine:
    def __init__(self):
        self.rf   = RandomForestStrategy()
        self.lgbm = LightGBMStrategy()
        self.lstm = LSTMStrategy()
        self.performance_log = []
        p = DATA / "trade_results.json"
        if p.exists():
            with open(p) as f:
                self.performance_log = json.load(f)

    def train_all(self, df, forward_bars=1, min_move=0.003, lstm_min_accuracy=0.28):
        r = {}
        r["rf"]   = self.rf.train(df, forward_bars=forward_bars, min_move=min_move)
        r["lgbm"] = self.lgbm.train(df, forward_bars=forward_bars, min_move=min_move)
        r["lstm"] = self.lstm.train(df, forward_bars=forward_bars, min_move=min_move,
                                    min_accuracy=lstm_min_accuracy)
        log.info(f"All models trained! RF={r['rf'].get('accuracy','?')} "
                 f"LGBM={r['lgbm'].get('accuracy','?')} "
                 f"LSTM={r['lstm'].get('accuracy','?')}")
        return r

    def train_all_from_features(self, combined_feat, forward_bars=1, min_move=0.003, lstm_min_accuracy=0.28):
        """Train all models from pre-computed, per-symbol clean feature DataFrames.
        combined_feat must contain '_label' and '_coin_id' columns."""
        labels   = combined_feat["_label"].astype(int)
        coin_ids = combined_feat["_coin_id"] if "_coin_id" in combined_feat.columns else None
        feat     = combined_feat.drop(columns=[c for c in ["_label", "_coin_id"] if c in combined_feat.columns])
        r = {}
        r["rf"]   = self.rf.train_from_features(feat, labels)
        r["lgbm"] = self.lgbm.train_from_features(feat, labels)
        r["lstm"] = self.lstm.train_from_features(feat, labels, coin_ids=coin_ids, min_accuracy=lstm_min_accuracy)
        log.info(f"All models trained! RF={r['rf'].get('accuracy','?')} "
                 f"LGBM={r['lgbm'].get('accuracy','?')} "
                 f"LSTM={r['lstm'].get('accuracy','?')}")
        return r

    def _get_dynamic_weights(self):
        """Weight each model by walk-forward accuracy; floor at 10% so no model is ignored."""
        rf_wf    = max(self.rf.metadata.get("wf_accuracy",   0.33), 0.10)
        lgbm_wf  = max(self.lgbm.metadata.get("wf_accuracy", 0.33), 0.10)
        lstm_acc = max(self.lstm.metadata.get("accuracy",    0.33), 0.10)
        raw = np.array([rf_wf, lgbm_wf, lstm_acc], dtype=float)
        w   = raw / raw.sum()
        return {"rf": float(w[0]), "lgbm": float(w[1]), "lstm": float(w[2])}

    def get_model_health(self):
        return {
            "rf_accuracy":   self.rf.metadata.get("accuracy", 0),
            "rf_wf":         self.rf.metadata.get("wf_accuracy", 0),
            "lgbm_accuracy": self.lgbm.metadata.get("accuracy", 0),
            "lstm_accuracy": self.lstm.metadata.get("accuracy", 0),
            "rf_trained":    self.rf.metadata.get("trained_at", "never"),
            "lstm_epochs":   self.lstm.metadata.get("epochs", 0),
        }

    def predict(self, df, symbol):
        rf_p   = self.rf.predict(df)
        lgbm_p = self.lgbm.predict(df)
        lstm_p = self.lstm.predict(df)

        W   = self._get_dynamic_weights()
        avg = (np.array(rf_p["probs"])   * W["rf"] +
               np.array(lgbm_p["probs"]) * W["lgbm"] +
               np.array(lstm_p["probs"]) * W["lstm"])
        avg    /= avg.sum()
        label   = int(np.argmax(avg))
        conf    = float(avg[label])
        action  = {0:"SELL",1:"HOLD",2:"BUY"}[label] if conf >= 0.40 else "HOLD"

        actions    = [rf_p["action"], lgbm_p["action"], lstm_p["action"]]
        buy_votes  = actions.count("BUY")
        sell_votes = actions.count("SELL")
        if action == "BUY"  and buy_votes  < 2: action = "HOLD"
        if action == "SELL" and sell_votes < 2: action = "HOLD"

        return {
            "symbol":     symbol,
            "action":     action,
            "confidence": round(conf, 4),
            "strategy":   f"RF:{rf_p['action']}+LGBM:{lgbm_p['action']}+LSTM:{lstm_p['action']}",
            "timeframe":  "AI-v2",
            "indicators": {
                "rf_conf":    rf_p["confidence"],
                "lgbm_conf":  lgbm_p["confidence"],
                "lstm_conf":  lstm_p["confidence"],
                "rf_action":  rf_p["action"],
                "lgbm_action":lgbm_p["action"],
                "lstm_action":lstm_p["action"],
                "buy_votes":  buy_votes,
                "sell_votes": sell_votes,
                "avg_probs":  avg.tolist(),
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    def record_trade_result(self, symbol, pnl):
        self.performance_log.append({
            "symbol": symbol, "pnl": pnl,
            "timestamp": datetime.utcnow().isoformat(),
        })
        with open(DATA / "trade_results.json", "w") as f:
            json.dump(self.performance_log, f, indent=2)
