# Training Data Quality Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace noisy binary training labels with ATR-threshold 3-class labels, add HOLD undersampling, and add precision-adaptive class weights — making RF + LightGBM models learn from meaningful price moves instead of noise.

**Architecture:** `make_labels()` in `ai_strategy.py` is updated to produce BUY(2)/HOLD(1)/SELL(0) using an ATR-based threshold. `bot.py` gains `_compute_class_weights()` and HOLD undersampling logic in `_train_with_pipeline()`. The predict path in `ai_strategy.py` is updated to correctly interpret 3-class probability outputs. Class weights flow from bot.py → `train_all()` → individual RF/LightGBM `train()` calls.

**Tech Stack:** Python, pandas, numpy, scikit-learn (RandomForestClassifier), lightgbm (LGBMClassifier), PyYAML configs

---

## Pre-flight: understand the codebase

Before writing any code, read these line ranges so you have accurate context:

- `bot/models/ai_strategy.py:162–179` — current `make_labels()` (binary)
- `bot/models/ai_strategy.py:260–332` — RF `train()` and `predict()`
- `bot/models/ai_strategy.py:389–495` — LightGBM `train()` and `predict()`
- `bot/models/ai_strategy.py:512–540` — `train_all()`
- `bot/models/ai_strategy.py:556–630` — `predict_ensemble()` and `predict_numpy()`
- `bot/engine/bot.py:395–533` — `_train_with_pipeline()` (where labels are created)
- `config_spot.yaml` lines 1–25 — `ml:` and `training:` sections
- `config_futures.yaml` lines 1–25 — same

Key facts discovered during planning:
- `training.atr_k` already exists in both configs at `0.45` — **do not add it again**
- `ml.forward_bars` is `1` in both configs with `primary_timeframe: 15m` — 15 min lookahead (too short)
- Trade dicts have `"side"` (`"buy"`/`"long"`) and `"pnl"` (float, negative = loss) fields
- `predict()` currently assumes binary probs `[SELL_prob, BUY_prob]` — must be updated for 3-class

---

## Task 1: Update `make_labels()` to ATR 3-class

**Files:**
- Modify: `bot/models/ai_strategy.py:162–179`
- Create: `bot/tests/test_make_labels.py`

- [ ] **Step 1: Create test file**

```python
# bot/tests/test_make_labels.py
import pandas as pd
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.ai_strategy import make_labels


def _make_df(closes, high_mult=1.01, low_mult=0.99):
    n = len(closes)
    closes = np.array(closes, dtype=float)
    return pd.DataFrame({
        "open":   closes * 0.999,
        "high":   closes * high_mult,
        "low":    closes * low_mult,
        "close":  closes,
        "volume": np.ones(n) * 1000,
    })


def test_returns_series_of_length_n_minus_forward_bars():
    df = _make_df([100.0] * 30)
    labels = make_labels(df, forward_bars=2, atr_k=0.5)
    assert len(labels) == 28  # 30 - 2


def test_three_classes_present_on_volatile_data():
    # alternating up/flat/down to guarantee all three classes
    closes = [100, 102, 100, 98, 100, 102, 100, 98, 100, 102,
              100, 98,  100, 102, 100, 98,  100, 102, 100, 98] * 3
    df = _make_df(closes, high_mult=1.05, low_mult=0.95)
    labels = make_labels(df, forward_bars=1, atr_k=0.3)
    classes = set(labels.unique())
    assert classes == {0, 1, 2}, f"Expected {{0,1,2}}, got {classes}"


def test_values_are_only_0_1_2():
    df = _make_df([100 + i * 0.1 for i in range(50)], high_mult=1.02, low_mult=0.98)
    labels = make_labels(df, forward_bars=1, atr_k=0.5)
    assert set(labels.unique()).issubset({0, 1, 2})


def test_flat_market_produces_mostly_hold():
    # prices that barely move → most labels should be HOLD
    closes = [100.0 + np.random.uniform(-0.01, 0.01) for _ in range(100)]
    df = _make_df(closes)
    labels = make_labels(df, forward_bars=1, atr_k=0.5)
    hold_frac = (labels == 1).mean()
    assert hold_frac > 0.5, f"Expected >50% HOLD in flat market, got {hold_frac:.2%}"


def test_backward_compat_atr_k_none_raises():
    """atr_k=None is not supported — caller must pass a value."""
    df = _make_df([100.0] * 30)
    with pytest.raises(TypeError):
        make_labels(df, forward_bars=1, atr_k=None)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/sarmad/cryptobot_v3/bot
python -m pytest tests/test_make_labels.py -v 2>&1 | head -40
```

Expected: multiple failures including `TypeError` and wrong class counts.

- [ ] **Step 3: Replace `make_labels()` in `ai_strategy.py`**

Find and replace the entire `make_labels` function (lines 162–179). The new version:

```python
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

    threshold = (atr / (close + 1e-9) * atr_k).clip(0.001, 0.03)
    future    = close.shift(-forward_bars) / close - 1

    labels = pd.Series(1, index=df.index, dtype=int)   # HOLD = 1
    labels[future >  threshold] = 2                      # BUY  = 2
    labels[future < -threshold] = 0                      # SELL = 0
    labels = labels[future.notna()].dropna()

    counts = labels.value_counts().sort_index()
    total  = len(labels)
    log.info(
        f"Labels (ATR 3-class, k={atr_k}): "
        f"SELL={counts.get(0,0)/total*100:.1f}% "
        f"HOLD={counts.get(1,0)/total*100:.1f}% "
        f"BUY={counts.get(2,0)/total*100:.1f}%"
    )
    return labels
```

- [ ] **Step 4: Run tests — all must pass**

```bash
cd /home/sarmad/cryptobot_v3/bot
python -m pytest tests/test_make_labels.py -v
```

Expected: 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/sarmad/cryptobot_v3
git add bot/models/ai_strategy.py bot/tests/test_make_labels.py
git commit -m "feat: replace make_labels() with ATR 3-class labeling (SELL=0, HOLD=1, BUY=2)"
```

---

## Task 2: Update predict() methods for 3-class output

The `predict()` methods in both RF and LightGBM currently assume a 2-element `probs` array (`[SELL_prob, BUY_prob]`). After training on 3-class labels, the model outputs `[SELL_prob, HOLD_prob, BUY_prob]`. This task fixes both `predict()` methods, `predict_numpy()`, and `predict_ensemble()`.

**Files:**
- Modify: `bot/models/ai_strategy.py` — RF predict (~line 336), LightGBM predict (~line 476), predict_numpy (~line 490), predict_ensemble (~line 556)

- [ ] **Step 1: Write a test that exercises predict() with a trained 3-class model**

```python
# bot/tests/test_predict_3class.py
import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.ai_strategy import RandomForestStrategy, make_labels, make_features


def _make_volatile_df(n=500):
    np.random.seed(42)
    closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
    closes = np.maximum(closes, 10)
    return pd.DataFrame({
        "open":   closes * (1 + np.random.uniform(-0.002, 0.002, n)),
        "high":   closes * (1 + np.random.uniform(0.001, 0.01, n)),
        "low":    closes * (1 - np.random.uniform(0.001, 0.01, n)),
        "close":  closes,
        "volume": np.random.uniform(1000, 5000, n),
    })


def test_predict_returns_valid_action():
    df = _make_volatile_df(500)
    rf = RandomForestStrategy(mode="spot")
    rf.train(df, forward_bars=1, atr_k=0.3, n_jobs=1)
    result = rf.predict(df)
    assert result["action"] in ("BUY", "SELL", "HOLD"), f"Got: {result['action']}"
    assert 0.0 <= result["confidence"] <= 1.0


def test_predict_probs_sum_to_one():
    df = _make_volatile_df(500)
    rf = RandomForestStrategy(mode="spot")
    rf.train(df, forward_bars=1, atr_k=0.3, n_jobs=1)
    result = rf.predict(df)
    assert abs(sum(result["probs"]) - 1.0) < 1e-6, f"probs={result['probs']}"
    assert len(result["probs"]) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /home/sarmad/cryptobot_v3/bot
python -m pytest tests/test_predict_3class.py -v 2>&1 | head -30
```

Expected: FAIL — `train()` doesn't accept `atr_k` yet and probs are length 2.

- [ ] **Step 3: Update `RandomForestStrategy.train()` to accept `atr_k`**

In `RandomForestStrategy.train()` (~line 260), add `atr_k: float = 0.5` to signature and pass it to `make_labels()`:

```python
def train(self, df, use_decay_weights: bool = False, feat_df=None, labels_s=None,
          forward_bars: int = 1, timeframe: str = "15m",
          min_confidence: float = 0.52, min_votes: int = 2, n_jobs: int = 4,
          atr_k: float = 0.5, class_weights: dict = None):
    log.info("Training Random Forest (walk-forward)...")
    if feat_df is not None and labels_s is not None:
        feat   = feat_df
        labels = labels_s
    else:
        feat   = make_features(df)
        labels = make_labels(df, forward_bars=forward_bars, atr_k=atr_k).reindex(feat.index).dropna()
        feat   = feat.loc[labels.index]
```

Then in both `m.fit(...)` calls inside this method, use `_make_sample_weights` (defined as module-level in Step 5 — skip adding it here):

Walk-forward fold fit (replace existing `m.fit` call):
```python
sw_fold = sw[tr_idx] if sw is not None else None
cw_fold = _make_sample_weights(y[tr_idx], class_weights)
if cw_fold is not None:
    sw_fold = cw_fold if sw_fold is None else sw_fold * cw_fold
m = RandomForestClassifier(n_estimators=200, max_depth=10,
                            min_samples_leaf=5,
                            random_state=42, n_jobs=n_jobs)
m.fit(Xtr, y[tr_idx], sample_weight=sw_fold)
```

Final model fit (replace existing `new_model.fit` call):
```python
sw_tr = sw[:split] if sw is not None else None
cw_tr = _make_sample_weights(y[:split], class_weights)
if cw_tr is not None:
    sw_tr = cw_tr if sw_tr is None else sw_tr * cw_tr
new_model.fit(Xtr, y[:split], sample_weight=sw_tr)
```

Also update metadata to store `n_classes`:
```python
self.metadata = {
    ...
    "n_classes": int(len(np.unique(y))),
    ...
}
```

- [ ] **Step 4: Update `RandomForestStrategy.predict()` for 3-class**

Replace the predict logic (around line 336) — the key change is reading `probs[2]` for BUY:

```python
def predict(self, df):
    if not self.is_trained:
        return {"action": "HOLD", "confidence": 0.30, "probs": [0.33, 0.34, 0.33]}
    feat = make_features(df)
    if len(feat) == 0:
        return {"action": "HOLD", "confidence": 0.30, "probs": [0.33, 0.34, 0.33]}
    try:
        X     = self.scaler.transform(feat.values[-1:])
        probs = self.model.predict_proba(X)[0]   # shape: (n_classes,)
        classes = self.model.classes_             # e.g. [0, 1, 2]
        prob_map = {int(c): float(p) for c, p in zip(classes, probs)}
        sell_p = prob_map.get(0, 0.0)
        hold_p = prob_map.get(1, 0.0)
        buy_p  = prob_map.get(2, 0.0)

        best_class = max(prob_map, key=prob_map.get)
        action = {0: "SELL", 1: "HOLD", 2: "BUY"}.get(best_class, "HOLD")
        confidence = round(prob_map[best_class], 4)
        return {
            "action":     action,
            "confidence": confidence,
            "probs":      [sell_p, hold_p, buy_p],
        }
    except Exception as e:
        log.warning(f"RF predict error: {e}")
        return {"action": "HOLD", "confidence": 0.30, "probs": [0.33, 0.34, 0.33]}
```

- [ ] **Step 5: Update `LightGBMStrategy.train()` similarly**

Add `atr_k: float = 0.5, class_weights: dict = None` to signature. Update `make_labels()` call. Apply class_weights as `sample_weight` in both walk-forward and final `fit()` calls using the same `_make_sample_weights` helper pattern from Step 3.

Note: define `_make_sample_weights` as a module-level function (not nested) so both RF and LightGBM can use it. Place it just after `compute_decay_weights()`.

```python
def _make_sample_weights(y_arr: np.ndarray, class_weights: dict) -> Optional[np.ndarray]:
    """Convert {class_int: weight} dict to a per-sample float32 weight array."""
    if class_weights is None:
        return None
    return np.vectorize(lambda c: class_weights.get(int(c), 1.0))(y_arr).astype(np.float32)
```

- [ ] **Step 6: Update `LightGBMStrategy.predict()` for 3-class**

Same pattern as RF — replace predict logic to use `prob_map`:

```python
def predict(self, df):
    if not self.is_trained:
        return {"action": "HOLD", "confidence": 0.30, "probs": [0.33, 0.34, 0.33]}
    feat = make_features(df)
    if len(feat) == 0:
        return {"action": "HOLD", "confidence": 0.30, "probs": [0.33, 0.34, 0.33]}
    try:
        X     = self.scaler.transform(feat.values[-1:])
        probs = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        prob_map = {int(c): float(p) for c, p in zip(classes, probs)}
        sell_p = prob_map.get(0, 0.0)
        hold_p = prob_map.get(1, 0.0)
        buy_p  = prob_map.get(2, 0.0)

        best_class = max(prob_map, key=prob_map.get)
        action = {0: "SELL", 1: "HOLD", 2: "BUY"}.get(best_class, "HOLD")
        confidence = round(prob_map[best_class], 4)
        return {
            "action":     action,
            "confidence": confidence,
            "probs":      [sell_p, hold_p, buy_p],
        }
    except Exception as e:
        log.warning(f"LightGBM predict error: {e}")
        return {"action": "HOLD", "confidence": 0.30, "probs": [0.33, 0.34, 0.33]}
```

- [ ] **Step 7: Update `predict_numpy()` for 3-class (LightGBM)**

Find `predict_numpy` in `LightGBMStrategy` (~line 490) and apply same `prob_map` pattern.

- [ ] **Step 8: Update `predict_ensemble()` to use index 2 for BUY**

In `AIStrategyEngine.predict_ensemble()` (~line 556), currently:
```python
buy_prob  = rf_probs[1] * w_rf + lgbm_probs[1] * w_lgbm
sell_prob = rf_probs[0] * w_rf + lgbm_probs[0] * w_lgbm
```

Update to:
```python
# probs layout: [SELL_prob, HOLD_prob, BUY_prob]
buy_prob  = rf_probs[2] * w_rf + lgbm_probs[2] * w_lgbm
sell_prob = rf_probs[0] * w_rf + lgbm_probs[0] * w_lgbm
hold_prob = rf_probs[1] * w_rf + lgbm_probs[1] * w_lgbm
```

Also update the HOLD detection logic in `predict_ensemble()`. Currently:
```python
if rf_p.get("action") == "HOLD" or lgbm_p.get("action") == "HOLD":
```
This is already correct — no change needed here.

- [ ] **Step 9: Update `train_all()` to accept and pass `atr_k` and `class_weights`**

In `AIStrategyEngine.train_all()` (~line 512), add parameters and pass through:

```python
def train_all(self, df, feat_df=None, labels_s=None,
              use_decay_weights: bool = False, btc_rows: int = 0,
              forward_bars: int = 1, timeframe: str = "15m",
              min_confidence: float = 0.52, min_votes: int = 2,
              quick: bool = False, n_jobs: int = 4, progress_fn=None,
              atr_k: float = 0.5, class_weights: dict = None):
    """All modes train RF+LGBM only."""
    r = {}
    r["rf"] = self.rf.train(df, feat_df=feat_df, labels_s=labels_s,
                             use_decay_weights=use_decay_weights,
                             forward_bars=forward_bars, timeframe=timeframe,
                             min_confidence=min_confidence, min_votes=min_votes,
                             n_jobs=n_jobs, atr_k=atr_k, class_weights=class_weights)
    if progress_fn:
        try: progress_fn(50)
        except Exception: pass
    r["lgbm"] = self.lgbm.train(df, feat_df=feat_df, labels_s=labels_s,
                                  use_decay_weights=use_decay_weights,
                                  forward_bars=forward_bars, timeframe=timeframe,
                                  min_confidence=min_confidence, min_votes=min_votes,
                                  n_jobs=n_jobs, atr_k=atr_k, class_weights=class_weights)
    ...
```

- [ ] **Step 10: Run predict tests**

```bash
cd /home/sarmad/cryptobot_v3/bot
python -m pytest tests/test_predict_3class.py tests/test_make_labels.py -v
```

Expected: all tests PASS.

- [ ] **Step 11: Commit**

```bash
cd /home/sarmad/cryptobot_v3
git add bot/models/ai_strategy.py bot/tests/test_predict_3class.py
git commit -m "feat: update predict() methods and train_all() for 3-class label output"
```

---

## Task 3: Add `_compute_class_weights()` and HOLD undersampling to `bot.py`

**Files:**
- Modify: `bot/engine/bot.py`
- Create: `bot/tests/test_class_weights.py`

- [ ] **Step 1: Write tests**

```python
# bot/tests/test_class_weights.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# We test _compute_class_weights as a standalone function
# Import it after defining — we'll add it as a module-level helper

def _compute_class_weights(closed_trades: list) -> dict:
    """Mirror of the function we'll add to bot.py — tested independently."""
    import numpy as np
    BUY_SIDES  = {"buy", "long"}
    SELL_SIDES = {"sell", "short"}
    FALLBACK   = {0: 2.0, 1: 1.0, 2: 2.0}

    buy_trades  = [t for t in closed_trades if t.get("side", "") in BUY_SIDES]
    sell_trades = [t for t in closed_trades if t.get("side", "") in SELL_SIDES]

    if len(buy_trades) < 20 or len(sell_trades) < 20:
        return FALLBACK

    buy_prec  = sum(1 for t in buy_trades  if t.get("pnl", 0) > 0) / len(buy_trades)
    sell_prec = sum(1 for t in sell_trades if t.get("pnl", 0) > 0) / len(sell_trades)

    buy_w  = float(np.clip(1.0 / (buy_prec  + 1e-9), 1.0, 4.0))
    sell_w = float(np.clip(1.0 / (sell_prec + 1e-9), 1.0, 4.0))

    return {0: sell_w, 1: 1.0, 2: buy_w}


def test_fallback_when_too_few_trades():
    result = _compute_class_weights([])
    assert result == {0: 2.0, 1: 1.0, 2: 2.0}


def test_fallback_when_fewer_than_20_per_side():
    trades = [{"side": "buy", "pnl": 1.0}] * 15 + [{"side": "sell", "pnl": -1.0}] * 15
    result = _compute_class_weights(trades)
    assert result == {0: 2.0, 1: 1.0, 2: 2.0}


def test_high_buy_precision_gives_low_buy_weight():
    trades = (
        [{"side": "buy",  "pnl":  1.0}] * 19 +
        [{"side": "buy",  "pnl": -1.0}] * 1  +   # 95% precision → weight ~1.05
        [{"side": "sell", "pnl":  1.0}] * 10 +
        [{"side": "sell", "pnl": -1.0}] * 10      # 50% precision → weight 2.0
    )
    result = _compute_class_weights(trades)
    assert result[2] < result[0], "Low-precision SELL should have higher weight than high-precision BUY"
    assert result[1] == 1.0


def test_hold_weight_always_1():
    trades = (
        [{"side": "buy",  "pnl": 1.0}] * 25 +
        [{"side": "sell", "pnl": 1.0}] * 25
    )
    result = _compute_class_weights(trades)
    assert result[1] == 1.0


def test_weights_clipped_to_4():
    # 0% precision → would be 1/0 → clipped to 4.0
    trades = (
        [{"side": "buy",  "pnl": -1.0}] * 25 +
        [{"side": "sell", "pnl": -1.0}] * 25
    )
    result = _compute_class_weights(trades)
    assert result[0] == 4.0
    assert result[2] == 4.0


def test_long_short_sides_counted_correctly():
    trades = (
        [{"side": "long",  "pnl":  1.0}] * 20 +
        [{"side": "short", "pnl": -1.0}] * 20
    )
    result = _compute_class_weights(trades)
    # long=100% precision → weight=1.0; short=0% → weight=4.0
    assert result[2] == 1.0
    assert result[0] == 4.0
```

- [ ] **Step 2: Run tests to verify they fail (function not yet in bot.py)**

```bash
cd /home/sarmad/cryptobot_v3/bot
python -m pytest tests/test_class_weights.py -v 2>&1 | head -20
```

Expected: all PASS — tests are self-contained and import nothing from bot.py yet. This step verifies the helper logic is correct before wiring it in.

- [ ] **Step 3: Add `_compute_class_weights()` to `bot.py`**

Add this as a standalone module-level function near the top of `bot/engine/bot.py` (after imports, before class definition):

```python
def _compute_class_weights(closed_trades: list) -> dict:
    """
    Precision-adaptive class weights from recent closed trades.
    Returns {SELL_class(0): w, HOLD_class(1): 1.0, BUY_class(2): w}.
    Falls back to {0: 2.0, 1: 1.0, 2: 2.0} if fewer than 20 trades per side.
    """
    import numpy as np
    BUY_SIDES  = {"buy", "long"}
    SELL_SIDES = {"sell", "short"}
    FALLBACK   = {0: 2.0, 1: 1.0, 2: 2.0}

    buy_trades  = [t for t in closed_trades if t.get("side", "") in BUY_SIDES]
    sell_trades = [t for t in closed_trades if t.get("side", "") in SELL_SIDES]

    if len(buy_trades) < 20 or len(sell_trades) < 20:
        return FALLBACK

    buy_prec  = sum(1 for t in buy_trades  if t.get("pnl", 0) > 0) / len(buy_trades)
    sell_prec = sum(1 for t in sell_trades if t.get("pnl", 0) > 0) / len(sell_trades)

    buy_w  = float(np.clip(1.0 / (buy_prec  + 1e-9), 1.0, 4.0))
    sell_w = float(np.clip(1.0 / (sell_prec + 1e-9), 1.0, 4.0))

    return {0: sell_w, 1: 1.0, 2: buy_w}
```

- [ ] **Step 4: Add HOLD undersampling and wire class weights in `_train_with_pipeline()`**

**Important:** `atr_k` must be read at the top of `_train_with_pipeline()` alongside the other config reads (near `fb`, `mc`, `mv`) so the nested `_build_symbol_features` closure can see it. Add this line after `fb = ml_cfg.get("forward_bars", 2)`:

```python
atr_k = training_cfg.get("atr_k", 0.5)
```

Then update the `make_labels` call inside `_build_symbol_features` to pass it:

```python
labels = make_labels(primary_df, forward_bars=fb, atr_k=atr_k)
```

In `bot/engine/bot.py`, inside `_train_with_pipeline()`, find the block that starts with:

```python
        # Cap dataset size
        max_rows = training_cfg.get("max_rows", 200000)
```

Insert the HOLD undersampling block **before** the cap:

```python
        # ── HOLD undersampling: cap HOLD at 40% of original row count ──
        hold_mask = combined_labels == 1
        hold_frac = hold_mask.sum() / len(combined_labels)
        if hold_frac > 0.40:
            target_hold = int(len(combined_labels) * 0.40)
            hold_idx    = combined_labels[hold_mask].index
            rng         = np.random.default_rng(42)
            drop_idx    = rng.choice(hold_idx, size=len(hold_idx) - target_hold, replace=False)
            combined_feats  = combined_feats.drop(index=drop_idx).reset_index(drop=True)
            combined_labels = combined_labels.drop(index=drop_idx).reset_index(drop=True)
            self.log.info(
                f"HOLD undersampled: {hold_frac:.1%} → "
                f"{(combined_labels==1).mean():.1%} of {len(combined_labels)} rows"
            )
```

Then read `atr_k` from config and compute class weights just before the `train_all()` call:

```python
        # ── Precision-adaptive class weights ──────────────────────────
        closed_trades = [t for t in self.state.state.get("trades", []) if t.get("status") == "closed"]
        closed_trades = closed_trades[-50:]  # last 50 only
        class_weights = _compute_class_weights(closed_trades)
        self.log.info(f"Class weights: {class_weights} (from {len(closed_trades)} closed trades)")
```

Then pass both to the `train_all()` call. Find the existing call:

```python
        results = self.ai.train_all(
            combined_feats,
            feat_df=pd.DataFrame(X_scaled, columns=feature_cols),
            labels_s=combined_labels,
            btc_rows=0,
            forward_bars=fb,
            timeframe=tf,
            min_confidence=mc,
            min_votes=mv,
            quick=quick,
            n_jobs=n_jobs,
            progress_fn=lambda p: self._write_training_status("running", source="pipeline", progress=p),
        )
```

Add `atr_k` and `class_weights`:

```python
        results = self.ai.train_all(
            combined_feats,
            feat_df=pd.DataFrame(X_scaled, columns=feature_cols),
            labels_s=combined_labels,
            btc_rows=0,
            forward_bars=fb,
            timeframe=tf,
            min_confidence=mc,
            min_votes=mv,
            quick=quick,
            n_jobs=n_jobs,
            atr_k=atr_k,
            class_weights=class_weights,
            progress_fn=lambda p: self._write_training_status("running", source="pipeline", progress=p),
        )
```

Also update the `make_labels` call inside `_build_symbol_features` (nested function in `_train_with_pipeline`) to pass `atr_k`:

```python
                labels = make_labels(primary_df, forward_bars=fb, atr_k=atr_k)
```

Since `atr_k` is read before the parallel worker loop, pass it via closure (Python closures capture by reference — this works as-is since `atr_k` is defined in the outer scope before `_build_symbol_features` is defined).

- [ ] **Step 5: Run the tests**

```bash
cd /home/sarmad/cryptobot_v3/bot
python -m pytest tests/test_class_weights.py tests/test_make_labels.py tests/test_predict_3class.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
cd /home/sarmad/cryptobot_v3
git add bot/engine/bot.py bot/tests/test_class_weights.py
git commit -m "feat: add HOLD undersampling and precision-adaptive class weights in _train_with_pipeline()"
```

---

## Task 4: Update `forward_bars` in configs

`ml.forward_bars` is currently `1` in both configs. With `primary_timeframe: 15m`, that's a 15-minute lookahead — far too short. Update to `8` (= 2 hours on 15m bars).

**Files:**
- Modify: `config_spot.yaml`
- Modify: `config_futures.yaml`

- [ ] **Step 1: Update spot config**

In `config_spot.yaml`, find:
```yaml
ml:
  forward_bars: 1
```
Change to:
```yaml
ml:
  forward_bars: 8
```

- [ ] **Step 2: Update futures config**

In `config_futures.yaml`, find:
```yaml
ml:
  forward_bars: 1
```
Change to:
```yaml
ml:
  forward_bars: 8
```

- [ ] **Step 3: Verify configs parse correctly**

```bash
cd /home/sarmad/cryptobot_v3
python -c "
import yaml
for f in ['config_spot.yaml', 'config_futures.yaml']:
    cfg = yaml.safe_load(open(f))
    fb = cfg['ml']['forward_bars']
    ak = cfg['training']['atr_k']
    print(f'{f}: forward_bars={fb}, atr_k={ak}')
"
```

Expected output:
```
config_spot.yaml: forward_bars=8, atr_k=0.45
config_futures.yaml: forward_bars=8, atr_k=0.45
```

- [ ] **Step 4: Commit**

```bash
cd /home/sarmad/cryptobot_v3
git add config_spot.yaml config_futures.yaml
git commit -m "config: increase forward_bars from 1 to 8 (15m × 8 = 2h lookahead)"
```

---

## Task 5: Smoke test end-to-end training path

Verify the full training pipeline runs without errors using the real bot configuration.

**Files:** No changes — read-only verification.

- [ ] **Step 1: Delete stale model files to force retrain**

```bash
rm -f /home/sarmad/cryptobot_v3/bot/data/spot/rf_model.pkl \
       /home/sarmad/cryptobot_v3/bot/data/spot/lgbm_model.pkl \
       /home/sarmad/cryptobot_v3/bot/data/futures/rf_model.pkl \
       /home/sarmad/cryptobot_v3/bot/data/futures/lgbm_model.pkl
```

- [ ] **Step 2: Run a minimal training smoke test**

```bash
cd /home/sarmad/cryptobot_v3/bot
python -c "
import os, sys, logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
os.environ['BOT_MODE'] = 'spot'
sys.path.insert(0, '.')

import pandas as pd
import numpy as np
from models.ai_strategy import make_labels, make_features, AIStrategyEngine

# Build synthetic OHLCV data
np.random.seed(0)
n = 600
closes = 100 + np.cumsum(np.random.randn(n) * 0.5)
df = pd.DataFrame({
    'open':   closes * 0.999,
    'high':   closes * 1.01,
    'low':    closes * 0.99,
    'close':  closes,
    'volume': np.random.uniform(1000, 5000, n),
})

# Test make_labels
labels = make_labels(df, forward_bars=8, atr_k=0.45)
counts = labels.value_counts()
print(f'Labels: SELL={counts.get(0,0)} HOLD={counts.get(1,0)} BUY={counts.get(2,0)}')
assert set(labels.unique()).issubset({0,1,2}), 'Bad label values'

# Test _compute_class_weights import
from engine.bot import _compute_class_weights
w = _compute_class_weights([])
print(f'Weights (fallback): {w}')
assert w == {0: 2.0, 1: 1.0, 2: 2.0}

# Test RF trains with 3-class labels
from models.ai_strategy import RandomForestStrategy
rf = RandomForestStrategy(mode='spot')
feat = make_features(df)
lbl  = make_labels(df, forward_bars=8, atr_k=0.45).reindex(feat.index).dropna()
feat = feat.loc[lbl.index]
result = rf.train(df, feat_df=feat, labels_s=lbl, atr_k=0.45,
                  class_weights={0:2.0, 1:1.0, 2:2.0}, n_jobs=1)
print(f'RF trained: accuracy={result.get(\"accuracy\",\"?\")}, n_classes={result.get(\"n_classes\",\"?\")}')

# Test predict returns 3-element probs
pred = rf.predict(df)
print(f'RF predict: action={pred[\"action\"]} confidence={pred[\"confidence\"]} probs={pred[\"probs\"]}')
assert len(pred['probs']) == 3, f'Expected 3 probs, got {len(pred[\"probs\"])}'
assert pred['action'] in ('BUY', 'SELL', 'HOLD')
print('ALL CHECKS PASSED')
"
```

Expected output ends with `ALL CHECKS PASSED`.

- [ ] **Step 3: Run all tests**

```bash
cd /home/sarmad/cryptobot_v3/bot
python -m pytest tests/ -v --tb=short
```

Expected: all tests PASS.

- [ ] **Step 4: Final commit**

```bash
cd /home/sarmad/cryptobot_v3
git add -A
git commit -m "test: training data quality smoke test passes end-to-end"
```

---

## Validation Checklist (after first live retrain)

After the bot runs its first retrain cycle (within 24h or manually triggered):

- [ ] Training log shows `Labels (ATR 3-class, k=0.45): SELL=X% HOLD=Y% BUY=Z%` with Y < 60%
- [ ] Training log shows `HOLD undersampled: X% → Y%` where Y ≤ 44%
- [ ] Training log shows `Class weights: {0: ..., 1: 1.0, 2: ...}`
- [ ] RF and LightGBM test accuracy both exceed 0.55 (vs ~0.52 before)
- [ ] `predict()` output includes 3-element `probs` list
- [ ] Trade count over 48h is 30–45% lower than baseline
