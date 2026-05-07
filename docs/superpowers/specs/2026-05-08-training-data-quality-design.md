# Training Data Quality Fix

**Date:** 2026-05-08
**Status:** Approved
**Scope:** `bot/models/ai_strategy.py`, `bot/engine/bot.py`, `config_spot.yaml`, `config_futures.yaml`

## Problem

The ML models (RF + LightGBM) produce noisy, low-confidence signals because the training labels are low quality:

1. **Binary labels with no threshold** — `make_labels()` in `ai_strategy.py` labels any positive tick as BUY and any negative tick as SELL, including 0.001% noise moves. Models learn direction on noise, not meaningful price action.
2. **No class imbalance handling** — RF and LightGBM use no class weights, making them free to exploit the majority class.
3. **ATR labeler bypassed** — `features/pipeline.py` already has a correct ATR 3-class labeler (`_make_labels_polars`) but it is never called in the actual training path (`bot.py:_train_with_pipeline()`), which calls the broken binary `make_labels()` instead.

**Symptoms:** Too many low-conviction entries, ~52–54% directional accuracy (near random), high false signal rate.

## Goals

- Raise directional accuracy to 60–68% on BUY/SELL predictions
- Reduce trade count by 30–45% while improving signal quality
- Keep the fix isolated to the training path — no changes to prediction, ensemble, or risk layers

## Design

### Change 1 — ATR 3-class labels (`ai_strategy.py`)

Replace `make_labels()` with ATR-threshold labeling:

```
BUY  (2) if forward_return > +atr_k × (ATR / close)
SELL (0) if forward_return < −atr_k × (ATR / close)
HOLD (1) otherwise
```

- `forward_bars = 2` (= 2h on 1h primary timeframe, already in config)
- `atr_k = 0.5` default — half an ATR unit is a meaningful move; tunable via `ml.atr_k` in YAML
- ATR computed as 14-period rolling mean of True Range on the primary timeframe DataFrame
- Labels clipped: threshold bounded between 0.1% and 3% to handle extreme volatility regimes
- Old binary `make_labels()` signature preserved as a passthrough for any callers that pass `atr_k=None`

Expected label distribution with `atr_k=0.5` on 1h crypto data: ~25% BUY, ~25% SELL, ~50% HOLD before undersampling.

### Change 2 — HOLD undersampling (`bot.py:_train_with_pipeline()`)

After labels are generated and before `train_all()` is called:

1. Count HOLD rows in the label Series
2. If HOLD fraction > 40%: randomly sample HOLD indices down to `0.4 × total_rows`
3. Drop the excess HOLD rows from both feature DataFrame and label Series
4. BUY and SELL rows are never dropped

This is applied once per training cycle, using a fixed random seed (42) for reproducibility. The undersampling ratio (40%) is not configurable — it is the correct operating point based on prior experience with this dataset.

Expected post-undersampling distribution: ~44% HOLD, ~28% BUY, ~28% SELL (HOLD reduced from 50% to 44% of final dataset — not perfectly balanced, but prevents collapse).

### Change 3 — Precision-adaptive class weights (`bot.py`)

New function `_compute_class_weights(closed_trades: list) -> dict`:

**Logic:**
1. Read the last 50 closed trades from `self.closed_trades` (in-memory list maintained by `BaseBot`)
2. For each of BUY and SELL: compute precision = `profitable_trades / total_trades_of_that_direction` where profitable = `pnl_pct > 0`
3. Weight = `1.0 / precision`, clipped to the range `[1.0, 4.0]`
4. HOLD weight is always 1.0 (baseline — we want to penalise wrong directional calls, not wrong abstentions)
5. **Fallback:** if fewer than 20 closed trades exist, return `{SELL: 2.0, HOLD: 1.0, BUY: 2.0}`

**Example:**
```
BUY precision = 0.45  → weight = 1/0.45 = 2.22 (clipped to 2.22)
SELL precision = 0.65 → weight = 1/0.65 = 1.54
HOLD weight = 1.0
→ {0: 1.54, 1: 1.0, 2: 2.22}
```

Weights are passed as `sample_weight` arrays to RF and as `class_weight` dict to LightGBM.

**RF:** Convert class weight dict to a per-sample weight array via `np.vectorize(weights.__getitem__)(y)`. Pass as `sample_weight` to `fit()` in both walk-forward folds and final model.

**LightGBM:** Pass as `params['class_weight']` — LightGBM natively supports this dict format for multiclass.

Weights are recomputed fresh every training cycle. They are logged but not persisted.

### Change 4 — Config (`config_spot.yaml` + `config_futures.yaml`)

Add one key under `ml:`:

```yaml
ml:
  atr_k: 0.5        # ATR multiplier for label threshold (new)
  forward_bars: 2   # already exists — no change
  retrain_hours: 24 # already exists — no change
```

## What Does Not Change

- `features/feature_builder.py` — feature set unchanged
- `features/pipeline.py` — pipeline dataset builder unchanged
- `engine/ensemble.py`, `engine/risk_agent.py` — signal aggregation unchanged
- `risk/manager.py` — Kelly sizing, circuit breaker unchanged
- `models/hmm.py`, `models/rl_agent.py` — HMM and DQN unchanged
- Prediction path in `ai_strategy.py` — `predict()` methods unchanged
- Telegram commands, dashboard, launcher — unchanged

## File Change Summary

| File | Change |
|---|---|
| `bot/models/ai_strategy.py` | Replace `make_labels()` with ATR 3-class version |
| `bot/engine/bot.py` | Add `_compute_class_weights()`, add HOLD undersampling in `_train_with_pipeline()` |
| `config_spot.yaml` | Add `ml.atr_k: 0.5` |
| `config_futures.yaml` | Add `ml.atr_k: 0.5` |

## Expected Outcomes

| Metric | Before | After |
|---|---|---|
| Label noise | High (0.001% moves labelled) | Low (only moves > 0.5×ATR) |
| Directional accuracy | ~52–54% | ~60–68% |
| Daily trade count | Baseline | −30 to −45% |
| Avg signal confidence | Low | Higher (fewer marginal calls) |
| HOLD collapse risk | High (no weighting) | Mitigated (undersampling + adaptive weights) |

## Validation

After deployment to demo trading:
1. Check label distribution in training logs — HOLD should be ≤ 40% post-undersampling
2. Monitor RF and LightGBM test accuracy logs — should exceed 58% after first retrain
3. Watch trade count vs baseline for 48h — expect meaningful reduction
4. After 50+ closed trades, verify `_compute_class_weights()` is returning non-fallback values in logs
