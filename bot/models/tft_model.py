"""
Lightweight Temporal Fusion Transformer (TFT).
PyTorch implementation — designed for crypto OHLCV sequences.
Supports optional static context (e.g. HTF bias features).
"""

import json
import logging
import numpy as np
import torch
import torch.nn as nn
import joblib
from datetime import datetime, timezone
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

log = logging.getLogger("TFTModel")


# ── Core building blocks ──────────────────────────────────────────────────────

class GatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc      = nn.Linear(input_size, output_size * 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x       = self.dropout(x)
        x, gate = self.fc(x).chunk(2, dim=-1)
        return x * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout: float = 0.1):
        super().__init__()
        self.fc1       = nn.Linear(input_size, hidden_size)
        self.fc2       = nn.Linear(hidden_size, hidden_size)
        self.glu       = GatedLinearUnit(hidden_size, output_size, dropout)
        self.norm      = nn.LayerNorm(output_size)
        self.skip      = (nn.Linear(input_size, output_size)
                          if input_size != output_size else nn.Identity())
        self.elu       = nn.ELU()

    def forward(self, x, context=None):
        residual = self.skip(x)
        h = self.elu(self.fc1(x))
        if context is not None:
            h = h + context
        h = self.elu(self.fc2(h))
        h = self.glu(h)
        return self.norm(h + residual)


# ── Lightweight TFT architecture ─────────────────────────────────────────────

class LightweightTFT(nn.Module):
    """
    Simplified TFT:
      temporal input → GRN variable selection → LSTM encoder →
      multi-head attention → GRN output → 3-class classifier
    Optional static context (HTF bias) injected into GRN.
    """

    def __init__(self, n_features: int, n_static: int = 0,
                 hidden_size: int = 64, n_heads: int = 4,
                 seq_len: int = 30, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.has_static  = n_static > 0

        self.var_grn = GatedResidualNetwork(n_features, hidden_size, hidden_size, dropout)

        if self.has_static:
            self.static_grn = GatedResidualNetwork(n_static, hidden_size, hidden_size, dropout)

        self.lstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=2, batch_first=True,
            dropout=dropout,
        )

        self.attention  = nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
        self.attn_norm  = nn.LayerNorm(hidden_size)
        self.output_grn = GatedResidualNetwork(hidden_size, hidden_size, hidden_size, dropout)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x_temporal, x_static=None):
        B, T, _ = x_temporal.shape
        flat    = x_temporal.reshape(B * T, -1)

        if self.has_static and x_static is not None:
            ctx  = self.static_grn(x_static)            # (B, H)
            ctx  = ctx.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            proc = self.var_grn(flat, ctx)
        else:
            proc = self.var_grn(flat)

        proc = proc.reshape(B, T, -1)                   # (B, T, H)

        lstm_out, _ = self.lstm(proc)                   # (B, T, H)

        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out    = self.attn_norm(attn_out + lstm_out)

        last = attn_out[:, -1, :]                       # (B, H)
        out  = self.output_grn(last)
        return self.classifier(out)                     # (B, n_classes)


# ── TFTStrategy: training + inference wrapper ─────────────────────────────────

class TFTStrategy:
    SEQ_LEN      = 30
    HIDDEN_SIZE  = 64
    N_HEADS      = 4
    N_STATIC     = 0
    MIN_ACCURACY = 0.40

    def __init__(self, data_dir: Path = None):
        import os
        from env_config import DATA_DIR
        if data_dir is not None:
            DATA = data_dir
        else:
            mode = os.environ.get("BOT_MODE", "spot")
            DATA = DATA_DIR / mode
            DATA.mkdir(exist_ok=True)

        self.model_path  = DATA / "tft_model.pt"
        self.scaler_path = DATA / "tft_scaler.pkl"
        self.meta_path   = DATA / "tft_meta.json"
        self._data_dir   = DATA
        self.model       = None
        self.scaler      = MinMaxScaler()
        self.n_features  = None
        self.is_trained  = False
        self.metadata    = {}
        self._load()

    def _load(self):
        if not self.model_path.exists():
            return
        try:
            ckpt            = torch.load(str(self.model_path), map_location="cpu",
                                         weights_only=False)
            self.n_features = ckpt["n_features"]
            self.model      = LightweightTFT(
                n_features  = self.n_features,
                hidden_size = self.HIDDEN_SIZE,
                n_heads     = self.N_HEADS,
                seq_len     = self.SEQ_LEN,
            )
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            self.scaler     = joblib.load(self.scaler_path)
            self.is_trained = True
            if self.meta_path.exists():
                with open(self.meta_path) as f:
                    self.metadata = json.load(f)
            log.info(f"TFT loaded (accuracy={self.metadata.get('accuracy','?')})")
        except Exception as e:
            log.warning(f"TFT load failed: {e}")
            self.is_trained = False

    def train(self, df):
        log.info("Training TFT...")
        try:
            from ai_strategy import make_features, make_labels, make_sequences
            feat   = make_features(df)
            labels = make_labels(df).reindex(feat.index).dropna()
            feat   = feat.loc[labels.index]

            if len(feat) < self.SEQ_LEN + 200:
                return {"error": "need more data for TFT"}

            self.n_features = feat.shape[1]
            self.scaler.fit(feat.values)
            X_sc         = self.scaler.transform(feat.values)
            y            = labels.values.astype(int)
            X_seq, y_seq = make_sequences(X_sc, y, self.SEQ_LEN)

            split     = int(len(X_seq) * 0.8)
            val_split = int(split * 0.9)

            X_tr  = torch.FloatTensor(X_seq[:val_split])
            y_tr  = torch.LongTensor(y_seq[:val_split])
            X_val = torch.FloatTensor(X_seq[val_split:split])
            y_val = torch.LongTensor(y_seq[val_split:split])
            X_te  = torch.FloatTensor(X_seq[split:])
            y_te  = torch.LongTensor(y_seq[split:])

            cw_arr = compute_class_weight("balanced",
                                          classes=np.unique(y_seq[:val_split]),
                                          y=y_seq[:val_split])
            cw = torch.FloatTensor(cw_arr)

            model     = LightweightTFT(n_features=self.n_features,
                                       hidden_size=self.HIDDEN_SIZE,
                                       n_heads=self.N_HEADS,
                                       seq_len=self.SEQ_LEN)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=5, factor=0.5, min_lr=1e-6
            )
            criterion = nn.CrossEntropyLoss(weight=cw)

            best_val_acc = 0.0
            best_state   = None
            patience_ct  = 0
            batch_size   = 32

            for epoch in range(100):
                model.train()
                idx = torch.randperm(len(X_tr))
                for i in range(0, len(X_tr), batch_size):
                    bi  = idx[i:i + batch_size]
                    xb, yb = X_tr[bi], y_tr[bi]
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_logits = model(X_val)
                    val_acc    = (val_logits.argmax(1) == y_val).float().mean().item()
                    val_loss   = nn.CrossEntropyLoss()(val_logits, y_val).item()

                scheduler.step(val_loss)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                    patience_ct  = 0
                else:
                    patience_ct += 1
                    if patience_ct >= 15:
                        break

            if best_state:
                model.load_state_dict(best_state)
            model.eval()

            with torch.no_grad():
                test_acc = (model(X_te).argmax(1) == y_te).float().mean().item()

            if test_acc < self.MIN_ACCURACY:
                log.warning(f"TFT new ({test_acc:.2%}) below floor — discarding")
                return {
                    "accuracy":     self.metadata.get("accuracy", 0),
                    "new_accuracy": round(float(test_acc), 4),
                    "status":       "below_floor",
                }

            stored_acc = self.metadata.get("test_accuracy", 0)
            if self.is_trained and stored_acc > test_acc + 0.02:
                log.warning(f"TFT new ({test_acc:.2%}) worse — keeping old")
                return {"accuracy": stored_acc, "status": "kept_old"}

            torch.save({"state_dict": model.state_dict(), "n_features": self.n_features},
                       str(self.model_path))
            joblib.dump(self.scaler, self.scaler_path)

            self.model      = model
            self.is_trained = True
            self.metadata   = {
                "accuracy":      round(float(test_acc), 4),
                "test_accuracy": round(float(test_acc), 4),
                "trained_at":    datetime.now(timezone.utc).isoformat(),
                "n_features":    self.n_features,
            }
            with open(self.meta_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            log.info(f"TFT trained! Accuracy={test_acc:.2%}")
            return self.metadata

        except Exception as e:
            log.error(f"TFT training failed: {e}", exc_info=True)
            return {"error": str(e)}

    def predict(self, df):
        _hold = {"action": "HOLD", "confidence": 0.34, "probs": [0.33, 0.34, 0.33]}
        if not self.is_trained:
            return _hold
        try:
            from ai_strategy import make_features
            feat = make_features(df)
            if len(feat) < self.SEQ_LEN:
                return _hold
            X_sc = self.scaler.transform(feat.values)
            seq  = torch.FloatTensor(X_sc[-self.SEQ_LEN:]).unsqueeze(0)
            self.model.eval()
            with torch.no_grad():
                probs = torch.softmax(self.model(seq), dim=-1)[0].numpy()
            label = int(np.argmax(probs))
            return {
                "action":     {0: "SELL", 1: "HOLD", 2: "BUY"}[label],
                "confidence": round(float(probs[label]), 4),
                "probs":      probs.tolist(),
            }
        except Exception as e:
            log.error(f"TFT predict error: {e}")
            return _hold

    def predict_mc(self, df, n_samples: int = 15):
        """
        MC Dropout inference — model kept in train() mode so dropout stays active.
        Returns mean probs + uncertainty (mean variance across classes).
        """
        if not self.is_trained:
            return None
        try:
            from ai_strategy import make_features
            feat = make_features(df)
            if len(feat) < self.SEQ_LEN:
                return None
            X_sc = self.scaler.transform(feat.values)
            seq  = torch.FloatTensor(X_sc[-self.SEQ_LEN:]).unsqueeze(0)
            self.model.train()   # dropout active
            preds = []
            with torch.no_grad():
                for _ in range(n_samples):
                    p = torch.softmax(self.model(seq), dim=-1)[0].numpy()
                    preds.append(p)
            self.model.eval()
            preds       = np.array(preds)           # (n_samples, 3)
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
            log.warning(f"TFT predict_mc error: {e}")
            return None
