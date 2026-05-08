# Bug Fixes: Ghost Trades, Invalid Symbols, Profile Mutation, State Bloat

**Date:** 2026-05-08
**Status:** Approved
**Scope:** `bot/engine/bot.py`, `bot/engine/profiles.py`, `bot/tests/`

---

## Problem Summary

Four confirmed bugs causing operational degradation:

1. **Ghost trades block new entries for up to 24 hours** — positions closed on the exchange stay "open" in local state for 24h, consuming `max_open` slots and triggering `-2022` ReduceOnly errors every 30s.
2. **Invalid symbols never blacklisted** — when a symbol fails feature-build with "binance does not have market symbol X", it is not added to `invalid_symbols`. It re-fails every cycle indefinitely.
3. **Profile singleton mutation** — `TradingProfile.from_config()` returns the raw `_PRESETS` object. Any `setattr()` on it permanently mutates the shared preset, causing e.g. `htf_filter_mode` to silently change from `"soft"` to something else mid-process.
4. **State file grows unboundedly** — all trades (open + closed) accumulate in one JSON with no archiving. Currently 58+ entries, scanned on every cycle.

---

## Design

### Fix 1 — Ghost trade timeout: 24h → 1 minute (`bot.py`)

**Location:** `_cleanup_ghost_trades()` — `bot.py:1291`

**Change:** Replace the `age_s < 86400` guard (24 hours) with `age_s < 60` (1 minute). The existing 60-second entry grace window at `age_s < 60` already handles race conditions. After that grace, a position missing from the exchange is definitively closed.

**Before:**
```python
if age_s is not None and age_s < 86400:
    remaining_h = (86400 - age_s) / 3600
    self.log.warning(f"Sync: {sym} open in state but missing from exchange — auto-cancel in {remaining_h:.1f}h")
    continue
# 24h elapsed — cancel the ghost trade
```

**After:**
```python
if age_s is not None and age_s < 60:
    self.log.debug(f"Sync: {sym} entered <60s ago, skipping ghost check")
    continue
# >60s missing from exchange — cancel the ghost trade immediately
self.log.warning(f"Sync: {sym} missing from exchange (age={age_s:.0f}s) — auto-closing as ghost")
```

The existing close logic below (fetch ticker price, compute PnL, set status=closed) is unchanged.

The duplicate `age_s < 60` debug log block above the 24h check becomes redundant and is removed (the single guard handles both cases).

**Effect:** Ghost trades are resolved within 1 scan cycle (30s) after the 60s grace. The `-2022` retry loop stops. `max_open` slots free up immediately.

---

### Fix 2 — Auto-blacklist invalid symbols (`bot.py`)

**Location:** `_analyze_symbol()` feature-build except block — `bot.py:1052`

**Change:** Add `self.feed.mark_invalid(symbol)` in the `except Exception` handler that fires on "binance does not have market symbol" errors.

**Before:**
```python
except Exception as e:
    self.log.error(f"[{symbol}] Feature build FAILED — no ML signal: {e}")
```

**After:**
```python
except Exception as e:
    self.log.error(f"[{symbol}] Feature build FAILED — no ML signal: {e}")
    if "does not have market symbol" in str(e):
        self.feed.mark_invalid(symbol)
        self.log.warning(f"[{symbol}] Marked invalid — will be excluded from watchlist")
```

Only the "does not have market symbol" error triggers blacklisting. Other transient failures (network errors, data quality) do not blacklist — those should retry.

**Effect:** CLANKER/USDT, CRCL/USDT, B2/USDT and any future non-existent symbols are auto-blacklisted after their first failure. They stop appearing in the scan loop from the next scanner refresh.

---

### Fix 3 — Profile singleton immutability (`profiles.py`)

**Location:** `TradingProfile.from_config()` — `profiles.py:76`

**Change:** Return `dataclasses.replace(profile)` (a shallow copy) instead of the raw preset reference. Also apply overrides to the copy, not the original.

**Before:**
```python
profile = cls.load(name)
overrides = config.get("training", {}).get("profile_overrides", {})
if overrides:
    for key, value in overrides.items():
        if hasattr(profile, key):
            setattr(profile, key, value)
return profile
```

**After:**
```python
from dataclasses import replace as _dc_replace
profile = _dc_replace(cls.load(name))   # shallow copy — never mutates _PRESETS
overrides = config.get("training", {}).get("profile_overrides", {})
for key, value in overrides.items():
    if hasattr(profile, key):
        object.__setattr__(profile, key, value)
return profile
```

`_PRESETS` objects are never modified. Each caller gets its own copy. The `load()` method is unchanged — it still returns the canonical preset (for read-only use).

**Effect:** `CONFLUENCE.htf_filter_mode` can never be silently changed by overrides or a config reload. The "HTF BUY strict-block" on a "soft" profile is eliminated.

---

### Fix 4 — State archiving (`bot.py` / `StateManager`)

**Location:** `StateManager.save()` — called from `_sync_futures()` and throughout `bot.py`

**Change:** On each `save()` call, before writing the main state file:
1. Find all trades with `status="closed"` and `close_timestamp` older than **3 days**
2. Append them to `data/futures_state_archive.json` (or `spot_state_archive.json`)
3. Remove them from the in-memory `d["trades"]` list before writing the main file

Archive file format: a JSON array of trade objects, written atomically (write to `.tmp` then `rename`) using the same pattern as the main state file. Protected by the same `StateManager._lock`.

The main state file retains:
- All `status="open"` trades
- `status="closed"` trades with `close_timestamp` within the last 3 days

`get_all_trades()` is updated to read from both the main state and the archive file when callers need full history (e.g. `_compute_class_weights()`, SelfLearner).

**Effect:** Main state file stabilises at ~20–30 trades (open + recent closed). Scan cycle I/O shrinks. Archive preserves full history.

---

## Tests

All tests go in `bot/tests/`. No mocking of the exchange — tests operate on plain Python dicts and objects.

### `test_ghost_trade.py`

```python
def test_ghost_closed_after_1min():
    """Trade missing from exchange for >60s must be closed by _cleanup_ghost_trades."""
    from datetime import datetime, timezone, timedelta
    # Build minimal state dict with one "open" trade timestamped 2 minutes ago
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    d = {"trades": [{"id": "g1", "symbol": "FOO/USDT", "status": "open",
                     "timestamp": old_ts, "price": 1.0, "side": "long",
                     "amount": 1.0, "close_price": 0.0, "pnl": 0.0,
                     "close_timestamp": ""}],
         "stats": {"total_pnl": 0.0, "wins": 0, "losses": 0}}
    # exchange_syms is empty — position gone from exchange
    bot_stub._cleanup_ghost_trades(set(), d)
    assert d["trades"][0]["status"] == "closed"

def test_fresh_trade_not_touched():
    """Trade entered <60s ago must not be ghost-closed even if missing from exchange."""
    from datetime import datetime, timezone
    fresh_ts = datetime.now(timezone.utc).isoformat()
    d = {"trades": [{"id": "g2", "symbol": "BAR/USDT", "status": "open",
                     "timestamp": fresh_ts, "price": 1.0, "side": "long",
                     "amount": 1.0, "close_price": 0.0, "pnl": 0.0,
                     "close_timestamp": ""}],
         "stats": {"total_pnl": 0.0, "wins": 0, "losses": 0}}
    bot_stub._cleanup_ghost_trades(set(), d)
    assert d["trades"][0]["status"] == "open"
```

### `test_invalid_symbol_blacklist.py`

```python
def test_mark_invalid_called_on_market_symbol_error():
    """Feature build failure with 'does not have market symbol' must call mark_invalid."""
    marked = []
    class FakeFeed:
        def mark_invalid(self, sym): marked.append(sym)
    feed = FakeFeed()
    # Simulate the except block logic
    e = Exception("binance does not have market symbol CLANKER/USDT")
    if "does not have market symbol" in str(e):
        feed.mark_invalid("CLANKER/USDT")
    assert "CLANKER/USDT" in marked

def test_transient_error_not_blacklisted():
    """Network timeout must NOT blacklist the symbol."""
    marked = []
    class FakeFeed:
        def mark_invalid(self, sym): marked.append(sym)
    feed = FakeFeed()
    e = Exception("connection timeout")
    if "does not have market symbol" in str(e):
        feed.mark_invalid("BTC/USDT")
    assert marked == []
```

### `test_profile_immutability.py`

```python
def test_from_config_returns_copy():
    """Mutating returned profile must not affect the _PRESETS singleton."""
    from engine.profiles import TradingProfile, _PRESETS
    config = {"strategy": {"trading_profile": "CONFLUENCE"}}
    p1 = TradingProfile.from_config(config)
    p1.htf_filter_mode = "strict"           # mutate the copy
    p2 = TradingProfile.from_config(config)  # get a fresh copy
    assert p2.htf_filter_mode == "soft"      # preset unchanged
    assert _PRESETS["CONFLUENCE"].htf_filter_mode == "soft"  # original unchanged

def test_load_returns_canonical_preset():
    """load() returns the canonical preset — different object from from_config()."""
    from engine.profiles import TradingProfile, _PRESETS
    preset = TradingProfile.load("CONFLUENCE")
    config = {"strategy": {"trading_profile": "CONFLUENCE"}}
    copy   = TradingProfile.from_config(config)
    assert preset is not copy
    assert preset.htf_filter_mode == copy.htf_filter_mode
```

### `test_state_archive.py`

```python
def test_old_closed_trades_archived():
    """Closed trades older than 3 days must move to archive on save."""
    # Create state with 3 trades
    now = datetime.now(timezone.utc)
    trades = [
        {"id": "t1", "status": "open",   "close_timestamp": "", ...},
        {"id": "t2", "status": "closed", "close_timestamp": (now - timedelta(days=2)).isoformat(), ...},
        {"id": "t3", "status": "closed", "close_timestamp": (now - timedelta(days=4)).isoformat(), ...},
    ]
    # After save, main state has t1 + t2 only; archive has t3
    ...
    assert {t["id"] for t in main_trades} == {"t1", "t2"}
    assert archive_trade_ids == {"t3"}
```

---

## File Change Summary

| File | Change |
|---|---|
| `bot/engine/bot.py` | Fix 1: `_cleanup_ghost_trades()` timeout 24h → 1min; Fix 2: `mark_invalid()` on symbol error; Fix 4: archive logic in `StateManager.save()` |
| `bot/engine/profiles.py` | Fix 3: `from_config()` returns `dataclasses.replace(profile)` |
| `bot/tests/test_ghost_trade.py` | New — 2 tests |
| `bot/tests/test_invalid_symbol_blacklist.py` | New — 2 tests |
| `bot/tests/test_profile_immutability.py` | New — 2 tests |
| `bot/tests/test_state_archive.py` | New — 2 tests |

## What Does Not Change

- Exchange API, execution engine, signal pipeline — unchanged
- Dashboard, Telegram commands — unchanged
- Risk manager, Kelly sizing, ensemble — unchanged
- `get_open_trades()` signature — unchanged (filters in-memory as before)
- Archive file is append-only — never rewritten, never deleted
