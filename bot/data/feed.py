"""
Data Feed v3
- OHLCV caching with polars internals (pandas-compatible output)
- Data validation
- Live price: WebSocket-first, REST fallback
- ATR calculation
"""

import logging
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

try:
    import polars as pl
    _POLARS = True
except ImportError:
    _POLARS = False

from core.config import DATA_DIR

log  = logging.getLogger("DataFeed")
DATA = DATA_DIR


# ── Polars helpers ────────────────────────────────────────────────────────────

def _raw_to_polars(raw: list) -> "pl.DataFrame":
    return pl.DataFrame({
        "timestamp": [r[0] for r in raw],
        "open":      [float(r[1]) for r in raw],
        "high":      [float(r[2]) for r in raw],
        "low":       [float(r[3]) for r in raw],
        "close":     [float(r[4]) for r in raw],
        "volume":    [float(r[5]) for r in raw],
    }).with_columns(
        pl.col("timestamp").cast(pl.Datetime("ms"))
    )


def _polars_to_pandas(df_pl: "pl.DataFrame") -> pd.DataFrame:
    df = df_pl.to_pandas()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df


# ── Validation ────────────────────────────────────────────────────────────────

class DataValidator:
    @staticmethod
    def validate(df, symbol: str):
        issues = []
        if df is None or len(df) == 0:
            return False, ["Empty dataframe"]
        if len(df) < 100:
            issues.append(f"Too few bars: {len(df)}")

        if _POLARS and isinstance(df, pl.DataFrame):
            null_ct = sum(df.null_count().row(0))
            if null_ct > 0:
                issues.append(f"NaN values: {null_ct}")
            if (df["close"] <= 0).any():
                issues.append("Zero/negative prices")
            dups = len(df) - df.n_unique(subset=["timestamp"])
            if dups > 0:
                issues.append(f"Duplicate timestamps: {dups}")
        else:
            nan_ct = df.isna().sum().sum()
            if nan_ct > 0:
                issues.append(f"NaN values: {nan_ct}")
            if (df["close"] <= 0).any():
                issues.append("Zero/negative prices")
            if hasattr(df, "index"):
                dups = df.index.duplicated().sum()
                if dups > 0:
                    issues.append(f"Duplicate timestamps: {dups}")

        # Only structural issues block validity (zero prices, duplicates)
        structural = [i for i in issues if "Too few" not in i and "NaN" not in i]
        is_valid = len(structural) == 0
        if issues:
            log.warning(f"Data quality for {symbol}: {issues}")
        return is_valid, issues

    @staticmethod
    def clean(df):
        if df is None or len(df) == 0:
            return df
        if _POLARS and isinstance(df, pl.DataFrame):
            df = df.unique(subset=["timestamp"], keep="first")
            df = df.drop_nulls()
            df = df.filter(
                ~pl.col("close").is_infinite() &
                ~pl.col("open").is_infinite()
            )
            df = df.sort("timestamp")
        else:
            df = df[~df.index.duplicated(keep="first")]
            df = df.dropna()
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            df = df.sort_index()
        return df


# ── OHLCV Cache ───────────────────────────────────────────────────────────────

from .freshness import Freshness


class OHLCVCache:
    """Per-timeframe OHLCV cache backed by the shared Freshness contract.

    Each tf has a default ceiling TTL (forming-candle refresh cadence). Callers
    may pass a tighter ``max_age_seconds`` to demand fresher data.
    """
    CANDLE_SECONDS = {
        "1m": 30, "5m": 60, "15m": 180,
        "1h": 300, "4h": 900, "1d": 1800,
    }

    def __init__(self) -> None:
        self._fresh = Freshness()

    def _key(self, symbol: str, tf: str) -> str:
        return f"{symbol}_{tf}"

    def _ceiling(self, tf: str) -> int:
        return self.CANDLE_SECONDS.get(tf, 3600)

    def needs_refresh(self, symbol: str, tf: str,
                      max_age_seconds: float | None = None) -> bool:
        age = self._fresh.age_seconds(self._key(symbol, tf))
        if age is None:
            return True
        ceiling = max_age_seconds if max_age_seconds is not None else self._ceiling(tf)
        return age >= ceiling

    def get(self, symbol: str, tf: str,
            max_age_seconds: float | None = None):
        ceiling = max_age_seconds if max_age_seconds is not None else self._ceiling(tf)
        return self._fresh.get(self._key(symbol, tf), ceiling)

    def set(self, symbol: str, tf: str, df) -> None:
        self._fresh.set(self._key(symbol, tf), df)


# ── REST price polling (fallback when WS unavailable) ─────────────────────────

class PriceMonitor:
    def __init__(self, exchange):
        self.exchange = exchange
        self._prices: Dict[str, float] = {}
        self._lock    = threading.Lock()
        self._symbols: set = set()
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def subscribe(self, symbol: str):
        self._symbols.add(symbol)

    def get_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._prices.get(symbol)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(
            target=self._poll_loop, daemon=True, name="PriceMonitor"
        )
        self._thread.start()
        log.info("REST price monitor started (WS fallback)")

    def _poll_loop(self):
        while self._running:
            symbols = list(self._symbols)
            if symbols:
                try:
                    tickers = self.exchange.fetch_tickers(symbols)
                    with self._lock:
                        for sym, ticker in tickers.items():
                            if ticker.get("last") is not None:
                                self._prices[sym] = float(ticker["last"])
                except Exception as e:
                    log.error(f"REST price poll error: {e}")
            time.sleep(5)


# ── DataFeed ──────────────────────────────────────────────────────────────────

class DataFeed:
    _INVALID_PATH = DATA_DIR / "invalid_symbols.json"

    def __init__(self, exchange, ws_feed=None):
        self.exchange        = exchange
        self.ws_feed         = ws_feed      # BinanceWSPriceFeed or None
        self.cache           = OHLCVCache()
        self.validator       = DataValidator()
        self.monitor         = PriceMonitor(exchange)
        self.invalid_symbols = self._load_invalid()
        self.monitor.start()

    # ── Invalid-symbol tracking ───────────────────────────────────────────────

    def _load_invalid(self) -> set:
        now = datetime.now(timezone.utc)
        expire_secs = 3 * 86400  # 3 days
        if self._INVALID_PATH.exists():
            try:
                with open(self._INVALID_PATH) as f:
                    raw = json.load(f)
                if isinstance(raw, list):
                    # Old format: list of strings → convert to dict with now timestamp
                    result = set()
                    for s in raw:
                        if isinstance(s, str):
                            result.add(s)
                    return result
                # New format: dict {symbol: iso_timestamp}
                result = set()
                for sym, ts in raw.items():
                    try:
                        age = (now - datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds()
                        if age < expire_secs:
                            result.add(sym)
                    except Exception:
                        result.add(sym)  # keep if timestamp unparseable
                return result
            except Exception:
                pass
        return set()

    def _save_invalid(self):
        now = datetime.now(timezone.utc).isoformat()
        data = {}
        # Preserve existing timestamps, add new ones
        existing = {}
        if self._INVALID_PATH.exists():
            try:
                with open(self._INVALID_PATH) as f:
                    raw = json.load(f)
                if isinstance(raw, dict):
                    existing = raw
            except Exception:
                pass
        for sym in self.invalid_symbols:
            existing[sym] = existing.get(sym, now)
        with open(self._INVALID_PATH, "w") as f:
            json.dump(existing, f)

    def mark_invalid(self, symbol: str):
        if symbol not in self.invalid_symbols:
            self.invalid_symbols.add(symbol)
            self._save_invalid()
            log.warning(f"Marked {symbol} invalid — excluded from watchlist (expires in 3 days)")

    # ── Subscription ─────────────────────────────────────────────────────────

    def subscribe(self, symbol: str):
        """Subscribe to REST monitor and WS feed (if available)."""
        self.monitor.subscribe(symbol)
        if self.ws_feed is not None:
            self.ws_feed.subscribe(symbol)

    def subscribe_many(self, symbols):
        for s in symbols:
            self.monitor.subscribe(s)
        if self.ws_feed is not None:
            self.ws_feed.subscribe_many(symbols)

    # ── OHLCV fetching ────────────────────────────────────────────────────────

    def fetch_ohlcv(
        self, symbol: str, timeframe: str = "1h",
        limit: int = 300, force_refresh: bool = False
    ) -> Optional[pd.DataFrame]:
        if symbol in self.invalid_symbols:
            return None
        if not force_refresh and not self.cache.needs_refresh(symbol, timeframe):
            cached = self.cache.get(symbol, timeframe)
            if cached is not None:
                n = len(cached)
                if n >= 100:
                    return _polars_to_pandas(cached) if (_POLARS and isinstance(cached, pl.DataFrame)) else cached
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not raw:
                return self._get_cached_pandas(symbol, timeframe)

            if _POLARS:
                df = _raw_to_polars(raw)
                df = self.validator.clean(df)
                is_valid, issues = self.validator.validate(df, symbol)
                if not is_valid:
                    log.warning(f"Validation failed {symbol}/{timeframe}: {issues}")
                    return self._get_cached_pandas(symbol, timeframe)
                self.cache.set(symbol, timeframe, df)
                return _polars_to_pandas(df)
            else:
                # Pandas fallback
                df = pd.DataFrame(
                    raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                df = self.validator.clean(df)
                is_valid, issues = self.validator.validate(df, symbol)
                if not is_valid:
                    log.warning(f"Validation failed {symbol}/{timeframe}: {issues}")
                    cached = self.cache.get(symbol, timeframe)
                    return cached if cached is not None else None
                self.cache.set(symbol, timeframe, df)
                return df

        except Exception as e:
            err = str(e)
            if "400" in err or "Bad Request" in err:
                self.mark_invalid(symbol)
            else:
                log.error(f"OHLCV error {symbol}/{timeframe}: {e}")
            return self._get_cached_pandas(symbol, timeframe)

    def _get_cached_pandas(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        cached = self.cache.get(symbol, timeframe)
        if cached is None:
            return None
        if _POLARS and isinstance(cached, pl.DataFrame):
            return _polars_to_pandas(cached)
        return cached

    def fetch_multi_timeframe(self, symbol: str, timeframes=None) -> Dict[str, pd.DataFrame]:
        if timeframes is None:
            timeframes = [("1h", 300), ("4h", 200), ("1d", 100)]
        dfs = {}
        with ThreadPoolExecutor(max_workers=len(timeframes)) as executor:
            future_to_tf = {
                executor.submit(self.fetch_ohlcv, symbol, tf, limit): tf
                for tf, limit in timeframes
            }
            for future in as_completed(future_to_tf):
                tf = future_to_tf[future]
                try:
                    df = future.result()
                    if df is not None and len(df) >= 50:
                        dfs[tf] = df
                except Exception:
                    pass
        return dfs

    # ── Live price ────────────────────────────────────────────────────────────

    def get_live_price(self, symbol: str) -> Optional[float]:
        self.subscribe(symbol)

        # 1. WebSocket (lowest latency)
        if self.ws_feed is not None:
            price = self.ws_feed.get_price(symbol)
            if price is not None:
                return price

        # 2. REST polling fallback
        price = self.monitor.get_price(symbol)
        if price is not None:
            return price

        # 3. Direct API call (final fallback)
        try:
            return float(self.exchange.fetch_ticker(symbol)["last"])
        except Exception as e:
            log.error(f"Price error {symbol}: {e}")
            return None

    # ── ATR ───────────────────────────────────────────────────────────────────

    def get_atr(self, symbol: str, timeframe: str = "1h", period: int = 14) -> float:
        df = self.fetch_ohlcv(symbol, timeframe, limit=period + 10)
        if df is None or len(df) < period:
            return 0.0
        try:
            high  = df["high"]
            low   = df["low"]
            close = df["close"]
            tr    = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs(),
            ], axis=1).max(axis=1)
            return float(tr.rolling(period).mean().iloc[-1])
        except Exception:
            return 0.0


# ── Training Data Store (v4) ──────────────────────────────────────────────────

class TrainingDataStore:
    """
    Persistent parquet cache for training OHLCV data.
    Data fetched from real Binance public API — full history, no auth.
    Each symbol+timeframe stored as a separate parquet file.
    Appends new bars daily, never deletes old data.
    """

    STORE_DIR = DATA_DIR / "training"

    @classmethod
    def _path(cls, symbol: str, timeframe: str) -> Path:
        clean = symbol.replace("/", "")
        cls.STORE_DIR.mkdir(exist_ok=True)
        return cls.STORE_DIR / f"{clean}_{timeframe}.parquet"

    @classmethod
    def ingest(cls, symbol: str, df: "pd.DataFrame", timeframe: str):
        path = cls._path(symbol, timeframe)
        if path.exists():
            try:
                existing = pd.read_parquet(path)
                combined = pd.concat([existing, df[~df.index.isin(existing.index)]])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                combined.to_parquet(path)
            except Exception:
                df.to_parquet(path)
        else:
            df.to_parquet(path)

    @classmethod
    def get(cls, symbol: str, timeframe: str) -> Optional["pd.DataFrame"]:
        path = cls._path(symbol, timeframe)
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                return None
        return None

    @classmethod
    def get_manifest(cls) -> dict:
        """Return stats about all cached training coins."""
        coins = []
        if not cls.STORE_DIR.exists():
            return {"coins": coins, "status": "empty"}

        for f in sorted(cls.STORE_DIR.glob("*.parquet")):
            try:
                df = pd.read_parquet(f)
                name = f.stem
                tf = "15m"
                if "_1h" in name:
                    tf = "1h"
                elif "_4h" in name:
                    tf = "4h"
                symbol = name.rsplit("_", 1)[0]
                if tf not in name:
                    symbol = name
                bars = len(df)
                quality = "good" if bars >= 3000 else ("ok" if bars >= 1000 else "low")
                coins.append({
                    "symbol": symbol,
                    "timeframe": tf,
                    "bars": bars,
                    "quality": quality,
                    "last_bar": str(df.index[-1])[:19],
                    "size_kb": f.stat().st_size // 1024,
                })
            except Exception:
                pass
        return {"coins": coins, "status": "ok"}

    @classmethod
    def needs_update(cls, symbol: str, timeframe: str, max_age_hours: int = 6) -> bool:
        df = cls.get(symbol, timeframe)
        if df is None or len(df) == 0:
            return True
        if len(df) < 100:
            return True
        age = (pd.Timestamp.now(tz="UTC") - df.index[-1]).total_seconds() / 3600
        return age > max_age_hours


# ── Training Feed (v4) ────────────────────────────────────────────────────────

class TrainingFeed:
    """
    Fetches OHLCV data from real Binance public API for model training.
    Uses ExchangeRouter.training — always real Binance regardless of execution mode.
    Caches to parquet via TrainingDataStore for persistence.
    """

    def __init__(self):
        from exchange.factory import get_exchange_router
        self.router = get_exchange_router()
        self.exchange = self.router.training
        self.store = TrainingDataStore
        self._last_fetch: Dict[str, str] = {}  # {symbol: iso_timestamp}

    def fetch_training_data(
        self, symbols: list, timeframe: str = "15m",
        limit: int = 5000, min_bars: int = 100,
    ) -> Dict[str, "pd.DataFrame"]:
        """
        Fetch OHLCV for training symbols from real Binance + parquet cache.
        Returns {symbol: DataFrame} for coins with >= min_bars.
        """
        import time
        result = {}
        for sym in symbols:
            try:
                cached = self.store.get(sym, timeframe)
                if cached is not None and len(cached) >= 100:
                    result[sym] = cached
                    self._last_fetch[sym] = pd.Timestamp.now(tz="UTC").isoformat()
                    log.debug(f"Training data cached: {sym} ({len(cached)} bars)")
                    continue
            except Exception:
                pass

            try:
                raw = self.exchange.fetch_ohlcv(sym, timeframe, limit=limit)
                if not raw or len(raw) < 50:
                    log.warning(f"Training data short: {sym} ({len(raw) if raw else 0} bars)")
                    continue
                df = pd.DataFrame(
                    raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                df = df[~df.index.duplicated(keep="last")].sort_index()
                self.store.ingest(sym, df, timeframe)
                if len(df) >= min_bars:
                    result[sym] = df
                self._last_fetch[sym] = pd.Timestamp.now(tz="UTC").isoformat()
                log.info(f"Training data fetched: {sym} ({len(df)} bars @ {timeframe})")
                time.sleep(0.1)
            except Exception as e:
                log.warning(f"Training fetch failed {sym}: {e}")
        return result

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 1000):
        """Fetch single-symbol OHLCV from persistent cache, falling back to Binance.
        limit=0 returns all cached bars (no API call)."""
        import time

        # ── 1. Persistent cache first (never re-fetch same candles) ──────
        cached = self.store.get(symbol, timeframe)
        if cached is not None and len(cached) > 0:
            if limit <= 0:
                return cached  # return ALL cached data
            if len(cached) >= limit:
                # Check freshness: last bar should be recent
                now_naive = pd.Timestamp.now()
                last_ts = cached.index[-1]
                if hasattr(last_ts, 'tz') and last_ts.tz is not None:
                    last_ts = last_ts.tz_convert(None)
                age_hours = (now_naive - last_ts).total_seconds() / 3600
                candle_secs = {"15m": 0.5, "30m": 0.5, "1h": 1, "4h": 4, "1d": 24}
                max_age = candle_secs.get(timeframe, 1)
                if age_hours <= max_age:
                    return cached.iloc[-limit:]

        # ── 2. Fetch latest from Binance ──────────────────────────────────
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=min(limit if limit > 0 else 1000, 1000))
        if not raw or len(raw) < 2:
            return cached if cached is not None and len(cached) > 0 else None
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[~df.index.duplicated(keep="last")].sort_index()
        time.sleep(0.05)

        # ── 3. Merge with cache ────────────────────────────────────────────
        current = self.store.get(symbol, timeframe)
        if current is not None and len(current) > 0:
            self.store.ingest(symbol, df, timeframe)
            merged = self.store.get(symbol, timeframe)
        else:
            self.store.ingest(symbol, df, timeframe)
            merged = df

        if limit <= 0:
            return merged
        return merged.iloc[-limit:] if merged is not None and len(merged) > 0 else None

    def get_last_fetch_time(self) -> Optional[str]:
        if not self._last_fetch:
            return None
        return max(self._last_fetch.values())

    @property
    def training_source(self) -> str:
        return "api.binance.com (real, public)"

    @property
    def is_demo_execution(self) -> bool:
        return self.router.is_demo
