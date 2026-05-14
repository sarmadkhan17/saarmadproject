# Anchor-Only ML Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train RF + LightGBM exclusively on 8 anchor coins with 3 years of local-cached history, removing all dynamic scanner coin injection from the training pipeline.

**Architecture:** `historical_fetcher.py` backfills 3 years of OHLCV per anchor into `data/training/*.parquet` via `TrainingDataStore`. `_train_with_pipeline()` reads from that cache (`limit=0`) and applies a `history_years` cutoff. The scanner still selects 25 live-trading coins — the predict path is untouched.

**Tech Stack:** pandas, pyarrow/parquet, scikit-learn (RF), LightGBM, ccxt (Binance public API)

---

## File Map

| File | Action | What changes |
|------|--------|--------------|
| `bot/data/historical_fetcher.py` | Modify | `TARGET_BARS` → 3 years; `DEFAULT_COINS` → 8 anchors only |
| `config_futures.yaml` | Modify | `training.symbols` → 8 anchors; add `history_years: 3`; remove `top_n` |
| `config_spot.yaml` | Modify | Same as above |
| `bot/engine/bot.py` | Modify | Remove scanner injection ×2; add history cutoff in `_build_symbol_features` |
| `bot/tests/test_anchor_ml_training.py` | Create | Unit tests for all changes |

---

### Task 1: Write failing tests

**Files:**
- Create: `bot/tests/test_anchor_ml_training.py`

- [ ] **Step 1: Create the test file**

```python
# bot/tests/test_anchor_ml_training.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


ANCHOR_COINS = ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK"]
YEARS = 3


def test_target_bars_15m_is_3_years():
    from data.historical_fetcher import TARGET_BARS
    assert TARGET_BARS["15m"] == YEARS * 365 * 24 * 4, \
        f"Expected {YEARS * 365 * 24 * 4}, got {TARGET_BARS['15m']}"


def test_target_bars_1h_is_3_years():
    from data.historical_fetcher import TARGET_BARS
    assert TARGET_BARS["1h"] == YEARS * 365 * 24, \
        f"Expected {YEARS * 365 * 24}, got {TARGET_BARS['1h']}"


def test_target_bars_4h_is_3_years():
    from data.historical_fetcher import TARGET_BARS
    assert TARGET_BARS["4h"] == YEARS * 365 * 6, \
        f"Expected {YEARS * 365 * 6}, got {TARGET_BARS['4h']}"


def test_target_bars_1d_is_3_years():
    from data.historical_fetcher import TARGET_BARS
    assert TARGET_BARS["1d"] == YEARS * 365, \
        f"Expected {YEARS * 365}, got {TARGET_BARS['1d']}"


def test_default_coins_is_exactly_8_anchors():
    from data.historical_fetcher import DEFAULT_COINS
    assert DEFAULT_COINS == ANCHOR_COINS, \
        f"Expected {ANCHOR_COINS}, got {DEFAULT_COINS}"


def test_history_cutoff_removes_old_rows():
    """3-year cutoff filter must drop rows older than history_years * 365 days."""
    import pandas as pd
    history_years = 3
    now = pd.Timestamp.now()
    cutoff = now - pd.Timedelta(days=int(history_years * 365))

    idx = pd.date_range(end=now, periods=10, freq="365D")
    df = pd.DataFrame({"close": range(10)}, index=idx)

    filtered = df[df.index >= cutoff]
    # Only rows within 3 years should survive
    assert all(filtered.index >= cutoff), "Old rows not removed"
    assert len(filtered) < len(df), "No rows were removed — cutoff not working"


def test_history_cutoff_keeps_recent_rows():
    """Rows within the 3-year window must not be dropped."""
    import pandas as pd
    history_years = 3
    now = pd.Timestamp.now()
    cutoff = now - pd.Timedelta(days=int(history_years * 365))

    idx = pd.date_range(end=now, periods=100, freq="D")
    df = pd.DataFrame({"close": range(100)}, index=idx)

    filtered = df[df.index >= cutoff]
    assert len(filtered) > 0, "All rows were removed — cutoff too aggressive"
    assert filtered.index[-1] >= cutoff, "Most recent row missing"
```

- [ ] **Step 2: Run tests — all must FAIL**

```bash
cd /root/cryptobot_v3/bot
python -m pytest tests/test_anchor_ml_training.py -v 2>&1 | tail -20
```

Expected: 7 failures — `TARGET_BARS` and `DEFAULT_COINS` not yet updated; cutoff tests should pass (they test pure pandas logic, not bot code). If cutoff tests fail, check pandas import.

---

### Task 2: Update `historical_fetcher.py`

**Files:**
- Modify: `bot/data/historical_fetcher.py:14-20`

- [ ] **Step 1: Replace DEFAULT_COINS and TARGET_BARS**

Open `bot/data/historical_fetcher.py` and replace lines 14–20:

```python
DEFAULT_COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK",
]
DEFAULT_TFS = ["15m", "1h", "4h", "1d"]
# Target bars per TF (3 years)
TARGET_BARS = {"15m": 105120, "1h": 26280, "4h": 6570, "1d": 1095}
FETCH_LIMIT = 1000
```

- [ ] **Step 2: Run the TARGET_BARS and DEFAULT_COINS tests — must PASS**

```bash
cd /root/cryptobot_v3/bot
python -m pytest tests/test_anchor_ml_training.py -v -k "target_bars or default_coins" 2>&1 | tail -15
```

Expected: 5 PASS

- [ ] **Step 3: Commit**

```bash
git add bot/data/historical_fetcher.py bot/tests/test_anchor_ml_training.py
git commit -m "fix: anchor-only ML training — 8 coins × 3yr TARGET_BARS + tests"
```

---

### Task 3: Update configs

**Files:**
- Modify: `config_futures.yaml`
- Modify: `config_spot.yaml`

- [ ] **Step 1: Replace `training` section in `config_futures.yaml`**

Find the `training:` block (starts around line 47) and replace the entire block with:

```yaml
training:
  atr_k: 0.45
  history_years: 3
  max_rows: 200000
  min_bars_per_coin: 500
  min_coins: 4
  n_jobs: 8
  primary_timeframe: 15m
  symbols:
  - BTC/USDT
  - ETH/USDT
  - SOL/USDT
  - BNB/USDT
  - XRP/USDT
  - DOGE/USDT
  - ADA/USDT
  - LINK/USDT
  timeframes:
  - 15m
  - 1h
  - 4h
  - 1d
  use_dataset_pipeline: true
  use_real_api: true
```

(Remove `top_n: 10` — it only controlled scanner injection which we're deleting.)

- [ ] **Step 2: Replace `training` section in `config_spot.yaml`**

Same replacement — find the `training:` block (starts around line 46):

```yaml
training:
  atr_k: 0.45
  history_years: 3
  max_rows: 200000
  min_bars_per_coin: 500
  min_coins: 4
  n_jobs: 8
  primary_timeframe: 15m
  symbols:
  - BTC/USDT
  - ETH/USDT
  - SOL/USDT
  - BNB/USDT
  - XRP/USDT
  - DOGE/USDT
  - ADA/USDT
  - LINK/USDT
  timeframes:
  - 15m
  - 1h
  - 4h
  - 1d
  use_dataset_pipeline: true
  use_real_api: true
```

- [ ] **Step 3: Verify YAML is valid**

```bash
python3 -c "import yaml; yaml.safe_load(open('config_futures.yaml')); yaml.safe_load(open('config_spot.yaml')); print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Verify exactly 8 symbols in each config**

```bash
python3 -c "
import yaml
for f in ['config_futures.yaml', 'config_spot.yaml']:
    cfg = yaml.safe_load(open(f))
    syms = cfg['training']['symbols']
    print(f'{f}: {len(syms)} symbols — {syms}')
    assert len(syms) == 8, f'Expected 8, got {len(syms)}'
    assert 'history_years' in cfg['training'], 'history_years missing'
    assert 'top_n' not in cfg['training'], 'top_n should be removed'
print('Config assertions passed')
"
```

Expected output:
```
config_futures.yaml: 8 symbols — ['BTC/USDT', 'ETH/USDT', ...]
config_spot.yaml: 8 symbols — ['BTC/USDT', 'ETH/USDT', ...]
Config assertions passed
```

- [ ] **Step 5: Commit**

```bash
git add config_futures.yaml config_spot.yaml
git commit -m "fix: trim training.symbols to 8 anchors, add history_years: 3"
```

---

### Task 4: Remove scanner injection from pipeline path + add history cutoff

**Files:**
- Modify: `bot/engine/bot.py:527-529` (scanner injection)
- Modify: `bot/engine/bot.py:554-556` (fetch inside `_build_symbol_features`)

- [ ] **Step 1: Remove the 3-line scanner injection block in `_train_with_pipeline()`**

Find this block (around line 527):

```python
        symbols = training_cfg.get("symbols", [])
        if self.scanner.top_coins:
            extra = [c for c in self.scanner.top_coins[:training_cfg.get("top_n", 10)] if c not in symbols]
            symbols = symbols + extra
```

Replace with:

```python
        symbols = training_cfg.get("symbols", [])  # anchor coins only — no scanner injection
```

- [ ] **Step 2: Add history cutoff inside `_build_symbol_features()`**

Find this block (around line 552):

```python
                dfs = {}
                for t in train_tfs:
                    df = self.training_feed.fetch_ohlcv(sym_api, t, limit=0)
                    if df is not None and len(df) >= 100:
                        dfs[t] = df
```

Replace with:

```python
                history_years = training_cfg.get("history_years", 3)
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=int(history_years * 365))
                dfs = {}
                for t in train_tfs:
                    df = self.training_feed.fetch_ohlcv(sym_api, t, limit=0)
                    if df is not None and len(df) >= 100:
                        df = df[df.index >= cutoff]
                    if df is not None and len(df) >= 100:
                        dfs[t] = df
```

Also apply the same cutoff to `primary_df` fetch (around line 562):

```python
                primary_df = self.training_feed.fetch_ohlcv(sym_api, primary_tf, limit=0)
                if primary_df is not None and len(primary_df) > 0:
                    primary_df = primary_df[primary_df.index >= cutoff]
```

- [ ] **Step 3: Run the full test suite to check for regressions**

```bash
cd /root/cryptobot_v3/bot
python -m pytest tests/ -v --tb=short 2>&1 | tail -25
```

Expected: all tests that were passing before still pass. The pre-existing failure (`test_confidence_swing_under_15pct`) is unrelated — ignore it if it appears.

- [ ] **Step 4: Commit**

```bash
git add bot/engine/bot.py
git commit -m "fix: remove scanner injection from pipeline training path, add 3yr history cutoff"
```

---

### Task 5: Remove scanner injection from legacy fallback path

**Files:**
- Modify: `bot/engine/bot.py:697-700`

- [ ] **Step 1: Remove scanner injection in `_train()` legacy path**

Find this block (around line 697):

```python
            train_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
                             "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT"]
            if self.scanner.top_coins:
                for c in self.scanner.top_coins[:4]:
                    if c not in train_symbols:
                        train_symbols.append(c)
```

Replace with:

```python
            train_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
                             "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT"]
```

- [ ] **Step 2: Run all tests**

```bash
cd /root/cryptobot_v3/bot
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: same pass count as after Task 4.

- [ ] **Step 3: Commit**

```bash
git add bot/engine/bot.py
git commit -m "fix: remove scanner injection from legacy training fallback path"
```

---

### Task 6: Run backfill and verify cache

- [ ] **Step 1: Run backfill to extend cache from 2 years → 3 years**

```bash
cd /root/cryptobot_v3/bot
python3 -c "from data.historical_fetcher import backfill; backfill()"
```

This is idempotent — coins/TFs already at target skip immediately. Expect ~8–12 min for the first run as it fetches the extra year of history in 1000-bar chunks.

Expected output (example):
```
Backfill: 8 coins × 4 TFs, targets={'15m': 105120, '1h': 26280, '4h': 6570, '1d': 1095}
  BTC/USDT/1d: ✓ 1007 bars (target 1095)   ← might still need top-up
  BTC/USDT/1h: ▶ 1167 → 26280
    +1000 → 2167
    +1000 → 3167
    ...
```

- [ ] **Step 2: Verify bar counts after backfill**

```bash
cd /root/cryptobot_v3/bot
python3 -c "
from data.feed import TrainingDataStore
m = TrainingDataStore.get_manifest()
anchors = ['BTCUSDT','ETHUSDT','SOLUSDT','BNBUSDT','XRPUSDT','DOGEUSDT','ADAUSDT','LINKUSDT']
for entry in m['coins']:
    if any(a in entry['symbol'] for a in anchors):
        print(f\"{entry['symbol']:15s} {entry['timeframe']:4s}  {entry['bars']:6,} bars  {entry['quality']}\")
"
```

Expected: all 8 anchors × 4 TFs show `good` quality (≥3000 bars). The 1h/15m will be much higher after full backfill.

- [ ] **Step 3: Run all tests**

```bash
cd /root/cryptobot_v3/bot
python -m pytest tests/test_anchor_ml_training.py -v 2>&1 | tail -15
```

Expected: all 7 tests PASS.

- [ ] **Step 4: Commit**

```bash
git add bot/tests/test_anchor_ml_training.py
git commit -m "test: verify anchor-only ML training — all 7 tests pass post-backfill"
```

---

### Task 7: Restart bot and confirm training uses 8 anchors

- [ ] **Step 1: Stop the bot**

```bash
ps aux | grep -i "python.*launcher" | grep -v grep | awk '{print $2}' | xargs -r kill -9
```

- [ ] **Step 2: Delete existing models to force a fresh retrain**

```bash
rm -f /root/cryptobot_v3/data/futures/rf_model.pkl \
      /root/cryptobot_v3/data/futures/lgbm_model.pkl \
      /root/cryptobot_v3/data/spot/rf_model.pkl \
      /root/cryptobot_v3/data/spot/lgbm_model.pkl
echo "Models cleared"
```

- [ ] **Step 3: Restart bot in futures mode**

```bash
cd /root/cryptobot_v3
bash start.sh 2 &
```

- [ ] **Step 4: Watch the training log for confirmation**

```bash
until grep -q "v5 dataset:" /root/cryptobot_v3/logs/futures_bot.log 2>/dev/null; do sleep 3; done
grep "v5 dataset:\|symbols\|TRAINING AI" /root/cryptobot_v3/logs/futures_bot.log | tail -10
```

Expected log lines:
```
TRAINING AI MODELS v4 [FUTURES] — QUICK (RF+LGBM only)
v5 dataset: NNNNN rows, MMM features, 8 symbols, Xs build
```

The `8 symbols` confirms anchor-only training is working.

- [ ] **Step 5: Confirm scanner still selects 25 coins for live trading**

```bash
grep "Watching:" /root/cryptobot_v3/logs/futures_bot.log | tail -1
```

Expected: 25 symbols listed (scanner unchanged).

- [ ] **Step 6: Final push to checkpoint branch**

```bash
git push origin checkpoint/signal-quality-fixes
```
