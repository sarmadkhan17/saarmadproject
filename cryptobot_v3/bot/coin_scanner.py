"""
Autonomous Coin Scanner v2
Scans top 50 Binance USDT pairs by volume
Picks best coins to trade autonomously
"""

import pandas as pd
import logging
import json
from pathlib import Path
from datetime import datetime
from env_config import DATA_DIR

log  = logging.getLogger("Scanner")
DATA = DATA_DIR

BLACKLIST = [
    "USDC/USDT","BUSD/USDT","TUSD/USDT","USDP/USDT",
    "DAI/USDT","FDUSD/USDT","UST/USDT","USDD/USDT",
]

# Always include these market leaders regardless of score
MUST_INCLUDE = [
    "BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT",
    "XRP/USDT","DOGE/USDT","ADA/USDT","AVAX/USDT",
    "LINK/USDT","DOT/USDT",
]


class CoinScanner:
    def __init__(self, config=None):
        cfg            = config or {}
        sc             = cfg.get("scanner", {})
        self.top_n     = sc.get("top_n", 10)
        self.min_vol   = sc.get("min_volume_usdt", 10_000_000)
        self.rescan_h  = sc.get("rescan_hours", 4)
        self.blacklist = sc.get("blacklist", BLACKLIST)
        self.top_coins = []
        self.last_scan = None
        self._load()

    def _load(self):
        p = DATA / "scanner_cache.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            self.top_coins = d.get("top_coins", [])
            last = d.get("last_scan")
            if last:
                self.last_scan = datetime.fromisoformat(last)
            if self.top_coins:
                log.info(f"Cached coins: {self.top_coins}")

    def _save(self):
        DATA.mkdir(exist_ok=True)
        with open(DATA / "scanner_cache.json", "w") as f:
            json.dump({"top_coins": self.top_coins,
                       "last_scan": self.last_scan.isoformat()}, f, indent=2)

    def needs_scan(self):
        if not self.top_coins or not self.last_scan:
            return True
        return (datetime.utcnow() - self.last_scan).total_seconds() >= self.rescan_h * 3600

    def is_fake_volume(self, ticker, df):
        """
        Detect fake/wash trading volume.
        Signs of fake volume:
        1. High volume but tiny price movement (vol/price ratio anomaly)
        2. Volume spike with no volatility
        3. Round number trades (too perfect)
        4. Price barely moves despite huge volume
        """
        try:
            vol    = float(ticker.get("quoteVolume", 0) or 0)
            price  = float(ticker.get("last", 0) or 0)
            chg    = abs(float(ticker.get("percentage", 0) or 0))

            if price <= 0 or vol <= 0:
                return True

            # Check 1: Volume/Price ratio anomaly
            # Legitimate: high volume coins have proportional price moves
            # Fake: massive volume but zero price change
            if vol > 50_000_000 and chg < 0.1:
                return True  # Suspicious: huge volume, no price movement

            # Check 2: Volatility vs Volume mismatch
            if df is not None and len(df) > 20:
                ret     = df["close"].pct_change().dropna()
                vol_std = float(ret.std() * 100)
                avg_vol = float(df["volume"].mean())
                last_vol= float(df["volume"].iloc[-1])

                # Volume spike with zero volatility = wash trading
                if last_vol > avg_vol * 5 and vol_std < 0.1:
                    return True

                # Check 3: Price range vs volume
                # Price barely moves = fake volume
                price_range = float(df["high"].max() - df["low"].min())
                if price_range / price < 0.001 and vol > 20_000_000:
                    return True

            return False
        except Exception:
            return False

    def score(self, ticker, df):
        s = 0.0
        try:
            # Reject fake volume immediately
            if self.is_fake_volume(ticker, df):
                return -1  # Negative score = excluded

            vol = float(ticker.get("quoteVolume", 0) or 0)
            if vol >= 100_000_000: s += 30
            elif vol >= 50_000_000: s += 22
            elif vol >= 20_000_000: s += 15
            elif vol >= 10_000_000: s += 8

            chg = abs(float(ticker.get("percentage", 0) or 0))
            if chg >= 5: s += 25
            elif chg >= 3: s += 18
            elif chg >= 1: s += 10

            if df is not None and len(df) > 20:
                ret = df["close"].pct_change().dropna()
                v   = float(ret.std() * 100)
                if 1.5 <= v <= 6: s += 25
                elif 1.0 <= v <= 10: s += 15
                else: s += 5

                c   = df["close"]
                e20 = c.ewm(span=20).mean().iloc[-1]
                e50 = c.ewm(span=50).mean().iloc[-1]
                p   = float(c.iloc[-1])
                if (p > e20 > e50) or (p < e20 < e50):
                    s += 20
                else:
                    s += 8
        except Exception:
            pass
        return round(s, 1)

    def scan(self, exchange, invalid_symbols=None):
        log.info("Scanning market for best opportunities...")
        try:
            tickers = exchange.fetch_tickers()
        except Exception as e:
            log.error(f"Ticker fetch failed: {e}")
            return self.top_coins

        bad = set(invalid_symbols or [])

        # Get valid trading symbols from exchange
        valid_set = set()
        if hasattr(exchange, "get_valid_symbols"):
            valid_set = exchange.get_valid_symbols()

        def is_valid(sym):
            if not valid_set:
                return True  # Fallback if can't get valid list
            native = sym.replace("/", "")
            return native in valid_set

        usdt_pairs = [
            (sym, t) for sym, t in tickers.items()
            if sym.endswith("/USDT")
            and sym not in self.blacklist
            and sym not in bad
            and float(t.get("quoteVolume", 0) or 0) >= self.min_vol
            and float(t.get("last", 0) or 0) >= 0.10
            and is_valid(sym)
            and len(sym.replace("/USDT","")) >= 2
            and sym.replace("/USDT","").isascii()
        ]
        usdt_pairs.sort(key=lambda x: float(x[1].get("quoteVolume", 0) or 0), reverse=True)
        top100 = usdt_pairs[:100]
        log.info(f"Scoring top 100 coins...")

        scores = {}
        fake_vol_count = 0
        for sym, ticker in top100:
            try:
                ohlcv = exchange.fetch_ohlcv(sym, "1h", limit=100)
                df    = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
            except Exception:
                df = None
            sc = self.score(ticker, df)
            if sc < 0:
                fake_vol_count += 1
                log.info(f"Fake volume detected: {sym} — excluded")
                continue
            scores[sym] = sc

        if fake_vol_count > 0:
            log.info(f"Excluded {fake_vol_count} coins with fake/wash trading volume")

        sorted_coins = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        log.info(f"Scored {len(scores)} valid coins from top 100 candidates")

        # Always include market leaders first
        final_coins = []
        for coin in MUST_INCLUDE:
            if coin in scores and coin not in self.blacklist:
                # Verify it exists on this exchange
                native = coin.replace("/", "")
                if valid_set and native not in valid_set:
                    continue
                final_coins.append(coin)

        # Fill remaining slots with top scorers
        for sym, sc in sorted_coins:
            if sym not in final_coins and len(final_coins) < self.top_n:
                final_coins.append(sym)

        self.top_coins = final_coins[:self.top_n]
        self.last_scan = datetime.utcnow()
        self._save()

        log.info("Top coins selected:")
        for i, (sym, sc) in enumerate(sorted_coins[:self.top_n], 1):
            log.info(f"  {i:2}. {sym:<12} score={sc:.0f}")
        return self.top_coins

    def get_coins(self, exchange, invalid_symbols=None):
        if self.needs_scan():
            return self.scan(exchange, invalid_symbols=invalid_symbols)
        # Filter cached list against current invalid set
        bad = set(invalid_symbols or [])
        return [c for c in self.top_coins if c not in bad]
