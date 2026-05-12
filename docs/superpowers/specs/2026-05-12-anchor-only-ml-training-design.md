# Design Spec: Anchor-Only ML Training with 3-Year History

**Date:** 2026-05-12
**Branch:** checkpoint/signal-quality-fixes
**Status:** Approved

---

## Problem

The ML models (RF + LightGBM) were being trained on an expanding pool of coins that mixed
two unrelated concerns:

1. **Training data quality** — the `training.symbols` config list grew to 40 coins (many with
   <12 months of history, high volatility, meme coins) when the watchlist was expanded.
2. **Scanner injection** — `_train_with_pipeline()` appended the scanner's top-N dynamic coins
   on top of the config list, making the training set non-deterministic and noisy.

This produced models trained on thin, volatile data instead of deep, liquid history.

---

## Goal

- Train RF + LightGBM exclusively on 8 **anchor coins** with 3 years of high-quality history.
- The scanner still dynamically selects 25 coins to trade each cycle.
- At predict time the same feature vector (coin-agnostic technical indicators) is computed
  for any scanner coin and scored by the anchor-trained model — no change to the predict path.

---

## Anchor Coins

```
BTC/USDT  ETH/USDT  SOL/USDT  BNB/USDT
XRP/USDT  DOGE/USDT  ADA/USDT  LINK/USDT
```

All 8 have 3+ years of continuous Binance history. Features are purely technical (RSI, ATR,
MACD, EMA ratios, volume profile) — they generalize across any USDT pair.

---

## Architecture

```
Backfill (one-time + weekly delta)
  historical_fetcher.backfill()
    └─ 1000-bar paginated fetch from Binance public API
    └─ TrainingDataStore.ingest() → data/training/{SYMBOL}_{TF}.parquet

Training cycle (startup / weekly / forced)
  _train_with_pipeline()
    └─ reads training.symbols = 8 anchors (no scanner injection)
    └─ TrainingFeed.fetch_ohlcv(limit=0) → reads from local parquet cache
    └─ filter to last 3 years (history_years cutoff)
    └─ build_features() × 4 TFs → RF + LightGBM fit

Predict cycle (every 30s per scanner coin)
  unchanged — same feature vector, same model
```

---

## Data Volume (3 Years per Anchor)

| Timeframe | Target bars | API calls to fill |
|-----------|-------------|-------------------|
| 15m       | 105,120     | ~106              |
| 1h        | 26,280      | ~27               |
| 4h        | 6,570       | ~7                |
| 1d        | 1,095       | ~2                |

8 anchors × 4 TFs = 32 parquet files. First backfill: ~1,144 API calls total at 0.4 s/call ≈ ~8 min.
Subsequent weekly delta: only new bars since last cache → ~1–2 min.

---

## Changes Required

### 1. `historical_fetcher.py` — bump TARGET_BARS to 3 years

```python
# Before (2 years)
TARGET_BARS = {"15m": 70080, "1h": 17520, "4h": 4380, "1d": 730}

# After (3 years)
TARGET_BARS = {"15m": 105120, "1h": 26280, "4h": 6570, "1d": 1095}
```

`DEFAULT_COINS` already contains the 8 anchors — no change needed there.

### 2. `config_futures.yaml` and `config_spot.yaml` — trim training.symbols

```yaml
training:
  atr_k: 0.45
  history_years: 3        # new — controls cutoff in _build_symbol_features
  max_rows: 200000
  min_bars_per_coin: 500
  min_coins: 4
  n_jobs: 8
  primary_timeframe: 15m
  symbols:                 # 8 anchors only
  - BTC/USDT
  - ETH/USDT
  - SOL/USDT
  - BNB/USDT
  - XRP/USDT
  - DOGE/USDT
  - ADA/USDT
  - LINK/USDT
  timeframes: [15m, 1h, 4h, 1d]
  use_dataset_pipeline: true
  use_real_api: true
  # top_n removed — scanner coins no longer mixed into training
```

### 3. `bot/engine/bot.py` — remove scanner injection (both paths)

**Pipeline path** `_train_with_pipeline()` lines ~527–529:
```python
# DELETE these 3 lines:
if self.scanner.top_coins:
    extra = [c for c in self.scanner.top_coins[:training_cfg.get("top_n", 10)] if c not in symbols]
    symbols = symbols + extra
```

**Add 3-year window filter** inside `_build_symbol_features()` after `fetch_ohlcv(limit=0)`:
```python
history_years = training_cfg.get("history_years", 3)
cutoff = pd.Timestamp.now() - pd.Timedelta(days=int(history_years * 365))
if df is not None and len(df) > 0:
    df = df[df.index >= cutoff]
```

**Legacy fallback path** `_train()` lines ~697–700:
```python
# DELETE these 4 lines:
if self.scanner.top_coins:
    for c in self.scanner.top_coins[:4]:
        if c not in train_symbols:
            train_symbols.append(c)
```

### 4. Run backfill once after deployment

```bash
cd /root/cryptobot_v3/bot
python3 -c "from data.historical_fetcher import backfill; backfill()"
```

This is idempotent — skips coins/TFs that already have enough bars.

---

## Error Handling

- `_build_symbol_features()` already wraps each symbol in `try/except` and skips on failure.
- `min_coins: 4` guard aborts training if fewer than 4 anchors produce clean rows.
- Backfill has its own empty-streak guard (3 consecutive empty responses → stops).

---

## Retrain Triggers (unchanged)

| Trigger | Frequency | Type |
|---------|-----------|------|
| Startup (models missing) | Once | Quick |
| Weekly | ISO week boundary | Full |
| Watchlist change ≥5 coins | As needed | Quick |
| Dashboard manual | On demand | Full |

Weekly retrain now reads from local parquet cache — no large API calls needed mid-session.

---

## Testing

- After backfill: verify bar counts via `TrainingDataStore.get_manifest()`
- After retrain: check log for `v5 dataset: N rows, M features, 8 symbols`
- Confirm scanner still selects 25 coins and predict path is unaffected
- Run `pytest bot/tests/` — no test changes expected (predict path unchanged)
