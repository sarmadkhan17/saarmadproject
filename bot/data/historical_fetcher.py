"""
Historical Data Backfiller — Native (No Docker)
Usage: python3 -c 'from bot.data.historical_fetcher import backfill; backfill()'
       or run directly: python3 /opt/cryptobot/bot/data/historical_fetcher.py

Fetches ALL available historical candles from live Binance public API.
Stores per-symbol×timeframe parquet files via TrainingDataStore.
NEVER re-fetches candles that already exist in cache.
"""

import time
import pandas as pd

DEFAULT_COINS = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK",
]
DEFAULT_TFS = ["15m", "1h", "4h", "1d"]
# Target bars per TF (3 years)
TARGET_BARS = {"15m": 105120, "1h": 26280, "4h": 6570, "1d": 1095}
FETCH_LIMIT = 1000
RATE_LIMIT = 0.4  # seconds between API calls


def backfill(coins=None, timeframes=None, targets=None, rate_limit=None):
    """Backfill all missing historical OHLCV data. Skips coins/TFs already cached."""
    from exchange.factory import get_exchange_router
    from data.feed import TrainingDataStore
    from pathlib import Path
    import sys

    if coins is None:
        coins = DEFAULT_COINS
    if timeframes is None:
        timeframes = DEFAULT_TFS
    if targets is None:
        targets = TARGET_BARS
    if rate_limit is None:
        rate_limit = RATE_LIMIT

    exchange = get_exchange_router().training  # ccxt binance — public API
    store = TrainingDataStore

    print(f"Backfill: {len(coins)} coins × {len(timeframes)} TFs, targets={targets}")
    print(f"Store: {store.STORE_DIR}")
    print("=" * 60)

    for sym_base in coins:
        sym = f"{sym_base}/USDT"
        for tf in timeframes:
            target = targets.get(tf, 0)
            cached = store.get(sym, tf)
            cached_bars = len(cached) if cached is not None else 0

            if cached_bars >= target:
                print(f"  {sym}/{tf}: ✓ {cached_bars:,} bars (target {target:,})")
                continue

            print(f"  {sym}/{tf}: ▶ {cached_bars:,} → {target:,}")

            end_time = (
                int(cached.index[0].timestamp() * 1000) - 1
                if cached is not None and cached_bars > 0
                else None
            )
            total = cached_bars
            empty_streak = 0

            while total < target and empty_streak < 3:
                extra_params = {}
                if end_time is not None:
                    extra_params["endTime"] = end_time

                try:
                    raw = exchange.fetch_ohlcv(sym, tf, None, FETCH_LIMIT, extra_params)
                except Exception as e:
                    print(f"    API error: {e}")
                    empty_streak += 1
                    time.sleep(rate_limit * 3)
                    continue

                if not raw or len(raw) < 2:
                    empty_streak += 1
                    if empty_streak >= 3:
                        print(f"    No more history available")
                    break

                empty_streak = 0
                df = pd.DataFrame(
                    raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                df = df[~df.index.duplicated(keep="last")].sort_index()

                end_time = int(df.index[0].timestamp() * 1000) - 1
                store.ingest(sym, df, tf)
                fresh = store.get(sym, tf)
                total = len(fresh) if fresh is not None else total + len(df)

                print(f"    +{len(df):,} → {total:,}")

                if total >= target:
                    break
                time.sleep(rate_limit)

    # Summary
    print("=" * 60)
    print("COMPLETE")
    for sym_base in coins:
        sym = f"{sym_base}/USDT"
        parts = []
        for tf in timeframes:
            df = store.get(sym, tf)
            parts.append(f"{tf}={len(df):,}" if df is not None else f"{tf}=NONE")
        print(f"  {sym}: {', '.join(parts)}")


if __name__ == "__main__":
    backfill()
