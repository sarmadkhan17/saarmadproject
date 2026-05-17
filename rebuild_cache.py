#!/usr/bin/env python3
"""
Rebuild OHLCV training cache from scratch.
Fetches 3 years of data for 8 anchor coins across 4 timeframes from real Binance API.
Saves to parquet files in /root/cryptobot_v3/data/training/
"""

import sys
import time
import os
from pathlib import Path

import ccxt
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "BNB/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "LINK/USDT",
]

TIMEFRAMES = ["15m", "1h", "4h", "1d"]
HISTORY_YEARS = 3
STORE_DIR = Path(__file__).parent / "data" / "training"
STORE_DIR.mkdir(parents=True, exist_ok=True)

# Binance API returns max 1000 bars per request
MAX_BARS_PER_REQUEST = 1000


def timeframe_to_ms(tf: str) -> int:
    """Convert timeframe to milliseconds."""
    mapping = {
        "1m": 60_000,
        "5m": 300_000,
        "15m": 900_000,
        "30m": 1_800_000,
        "1h": 3_600_000,
        "4h": 14_400_000,
        "1d": 86_400_000,
    }
    return mapping[tf]


def fetch_full_history(exchange, symbol: str, timeframe: str, years: int) -> pd.DataFrame:
    """
    Fetch complete OHLCV history for a symbol/timeframe pair.
    Paginates backwards from now to `years` ago.
    """
    tf_ms = timeframe_to_ms(timeframe)
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - int(years * 365.25 * 24 * 3600 * 1000)

    all_candles = []
    current_end = end_ms
    request_count = 0

    print(f"  Fetching {symbol} @ {timeframe}...", flush=True)

    while current_end > start_ms:
        try:
            candles = exchange.fetch_ohlcv(
                symbol, timeframe, limit=MAX_BARS_PER_REQUEST, params={"endTime": current_end}
            )
        except Exception as e:
            print(f"    Error at {current_end}: {e}", flush=True)
            time.sleep(5)
            continue

        if not candles:
            break

        all_candles.extend(candles)
        oldest_ts = candles[0][0]

        if oldest_ts >= current_end:
            # No progress — we've hit the start of available data
            break

        current_end = oldest_ts
        request_count += 1

        # Progress indicator
        bars_fetched = len(all_candles)
        elapsed_years = (end_ms - oldest_ts) / (365.25 * 24 * 3600 * 1000)
        print(f"    Request #{request_count}: {bars_fetched} bars, {elapsed_years:.1f}y covered", flush=True)

        # Rate limit: Binance allows 1200 req/min, but be conservative
        time.sleep(0.3)

    if not all_candles:
        print(f"  WARNING: No data fetched for {symbol} @ {timeframe}", flush=True)
        return None

    # Build DataFrame
    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[~df.index.duplicated(keep="last")]
    df.sort_index(inplace=True)

    # Trim to exact history window
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=int(years * 365))
    df = df[df.index >= cutoff]

    # Validate
    gaps = df.index[1:] - df.index[:-1]
    expected_gap = pd.Timedelta(milliseconds=tf_ms)
    large_gaps = gaps[gaps > expected_gap * 1.5]

    print(f"  Result: {len(df)} bars | {df.index[0]} → {df.index[-1]}")
    if len(large_gaps) > 0:
        print(f"  WARNING: {len(large_gaps)} gaps > 1.5x expected interval")
        for i, (idx, gap) in enumerate(large_gaps.items()):
            if i >= 3:
                break
            print(f"    Gap at {idx}: {gap}")

    return df


def save_parquet(df: pd.DataFrame, symbol: str, timeframe: str):
    """Save DataFrame to parquet file."""
    clean_symbol = symbol.replace("/", "")
    path = STORE_DIR / f"{clean_symbol}_{timeframe}.parquet"
    df.to_parquet(path)
    print(f"  Saved: {path} ({len(df)} bars, {df.memory_usage(deep=True).sum() / 1024:.1f} KB)", flush=True)


def main():
    print("=" * 60)
    print("Rebuilding OHLCV training cache")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"History: {HISTORY_YEARS} years")
    print(f"Store: {STORE_DIR}")
    print("=" * 60)

    # Initialize exchange (real Binance, public API)
    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    total_files = 0
    total_bars = 0
    start_time = time.time()

    for symbol in SYMBOLS:
        print(f"\n{'='*40}")
        print(f"Symbol: {symbol}")
        print(f"{'='*40}")

        for tf in TIMEFRAMES:
            df = fetch_full_history(exchange, symbol, tf, HISTORY_YEARS)
            if df is not None and len(df) >= 100:
                save_parquet(df, symbol, tf)
                total_files += 1
                total_bars += len(df)
            else:
                print(f"  SKIPPED: insufficient data ({len(df) if df is not None else 0} bars)")

            # Extra delay between symbols/timeframes
            time.sleep(0.5)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE: {total_files} files, {total_bars:,} bars in {elapsed:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
