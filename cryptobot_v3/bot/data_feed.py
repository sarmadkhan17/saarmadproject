"""
Data Feed v2
- OHLCV caching
- Data validation
- Live price monitoring
- ATR calculation
"""

import pandas as pd
import numpy as np
import logging
import json
import time
import threading
from datetime import datetime
from typing import Dict, Optional
from env_config import DATA_DIR

log  = logging.getLogger("DataFeed")
DATA = DATA_DIR


class DataValidator:
    @staticmethod
    def validate(df, symbol):
        issues = []
        if df is None or len(df) == 0:
            return False, ["Empty dataframe"]
        if len(df) < 100:
            issues.append(f"Too few bars: {len(df)}")
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            issues.append(f"NaN values: {nan_count}")
        if (df["close"] <= 0).any():
            issues.append("Zero/negative prices")
        dups = df.index.duplicated().sum()
        if dups > 0:
            issues.append(f"Duplicate timestamps: {dups}")
        is_valid = len([i for i in issues if "Too few" not in i and "NaN" not in i]) == 0
        if issues:
            log.warning(f"Data quality issues for {symbol}: {issues}")
        return is_valid, issues

    @staticmethod
    def clean(df):
        if df is None or len(df) == 0:
            return df
        df = df[~df.index.duplicated(keep="first")]
        df = df.dropna()
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        return df.sort_index()


class OHLCVCache:
    CANDLE_SECONDS = {
        "1m": 60, "5m": 300, "15m": 900,
        "1h": 3600, "4h": 14400, "1d": 86400,
    }

    def __init__(self):
        self._cache = {}
        self._times = {}

    def _key(self, symbol, tf):
        return f"{symbol}_{tf}"

    def needs_refresh(self, symbol, tf):
        key      = self._key(symbol, tf)
        last     = self._times.get(key)
        interval = self.CANDLE_SECONDS.get(tf, 3600)
        if last is None:
            return True
        return (datetime.utcnow() - last).total_seconds() >= interval

    def get(self, symbol, tf):
        return self._cache.get(self._key(symbol, tf))

    def set(self, symbol, tf, df):
        key             = self._key(symbol, tf)
        self._cache[key] = df
        self._times[key] = datetime.utcnow()


class PriceMonitor:
    def __init__(self, exchange):
        self.exchange  = exchange
        self._prices   = {}
        self._lock     = threading.Lock()
        self._symbols  = set()
        self._running  = False
        self._thread   = None

    def subscribe(self, symbol):
        self._symbols.add(symbol)

    def get_price(self, symbol):
        with self._lock:
            return self._prices.get(symbol)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info("Price monitor started")

    def _poll_loop(self):
        while self._running:
            symbols = list(self._symbols)
            if symbols:
                try:
                    tickers = self.exchange.fetch_tickers(symbols)
                    with self._lock:
                        for sym, ticker in tickers.items():
                            if ticker.get("last"):
                                self._prices[sym] = float(ticker["last"])
                except Exception as e:
                    log.error(f"Price monitor error: {e}")
            time.sleep(5)


class DataFeed:
    _INVALID_PATH = DATA_DIR / "invalid_symbols.json"

    def __init__(self, exchange):
        self.exchange        = exchange
        self.cache           = OHLCVCache()
        self.validator       = DataValidator()
        self.monitor         = PriceMonitor(exchange)
        self.invalid_symbols = self._load_invalid()
        self.monitor.start()

    def _load_invalid(self):
        if self._INVALID_PATH.exists():
            try:
                with open(self._INVALID_PATH) as f:
                    return set(json.load(f))
            except Exception:
                pass
        return set()

    def _save_invalid(self):
        with open(self._INVALID_PATH, "w") as f:
            json.dump(list(self.invalid_symbols), f)

    # Symbols that must never be excluded — needed for regime detection
    _PROTECTED = {"BTC/USDT", "ETH/USDT"}

    def mark_invalid(self, symbol):
        if symbol in self._PROTECTED:
            return
        if symbol not in self.invalid_symbols:
            self.invalid_symbols.add(symbol)
            self._save_invalid()
            log.warning(f"Marked {symbol} as invalid (demo API unsupported) — excluded from watchlist")

    def fetch_ohlcv(self, symbol, timeframe="1h", limit=300, force_refresh=False):
        if symbol in self.invalid_symbols:
            return None
        if not force_refresh and not self.cache.needs_refresh(symbol, timeframe):
            cached = self.cache.get(symbol, timeframe)
            if cached is not None and len(cached) >= 100:
                return cached
        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not raw:
                return self.cache.get(symbol, timeframe)
            df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            df = self.validator.clean(df)
            is_valid, issues = self.validator.validate(df, symbol)
            if not is_valid:
                log.warning(f"Data validation failed for {symbol}/{timeframe}: {issues}")
                cached = self.cache.get(symbol, timeframe)
                if cached is not None:
                    log.info(f"Using cached data for {symbol}/{timeframe}")
                    return cached
            self.cache.set(symbol, timeframe, df)
            return df
        except Exception as e:
            err = str(e)
            if "400" in err or "Bad Request" in err:
                self.mark_invalid(symbol)
            else:
                log.error(f"OHLCV error {symbol}/{timeframe}: {e}")
            return self.cache.get(symbol, timeframe)

    def fetch_multi_timeframe(self, symbol, timeframes=None):
        if timeframes is None:
            timeframes = [("1h", 500), ("4h", 300), ("1d", 200)]
        dfs = {}
        for tf, limit in timeframes:
            df = self.fetch_ohlcv(symbol, tf, limit)
            if df is not None and len(df) >= 50:
                dfs[tf] = df
        return dfs

    def get_live_price(self, symbol):
        self.monitor.subscribe(symbol)
        price = self.monitor.get_price(symbol)
        if price:
            return price
        try:
            return float(self.exchange.fetch_ticker(symbol)["last"])
        except Exception as e:
            log.error(f"Price error {symbol}: {e}")
            return None

    def get_atr(self, symbol, timeframe="1h", period=14):
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
