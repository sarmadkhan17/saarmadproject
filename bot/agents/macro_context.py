"""
MacroContextAgent — BTC.D / USDT.D kill switch.

Uses CoinGecko global market data to determine:
  - BTC dominance trend direction + rate of change
  - USDT dominance (risk-off signal)
  - Which coin universe is open for trading

Kill conditions (veto ALL new entries):
  - USDT.D rising > 0.4% per hour → capital fleeing to stablecoins
  - BTC.D spiking > 0.8% per hour → altcoin sell-off in progress

Universe rules:
  BTC.D rising  + BTC price rising  → BTC pairs only
  BTC.D falling + BTC price rising  → Full altcoin universe open
  BTC.D rising  + BTC price falling → Shorts only, reduce exposure
  USDT.D falling                    → Risk-on, full universe
"""

import logging
import time
import requests
from datetime import datetime, timezone
from core.tz import LOCAL_TZ
from typing import Optional

log = logging.getLogger("MacroContext")


class MacroContextAgent:
    COINGECKO_GLOBAL = "https://api.coingecko.com/api/v3/global"
    CACHE_SECS = 300  # 5-minute cache — dominance moves slowly

    def __init__(self, coingecko_api_key: str = ""):
        self._api_key = coingecko_api_key
        self._cache: Optional[dict] = None
        self._cache_time: float = 0
        # Rolling history for rate-of-change calculation
        self._history: list = []  # list of (timestamp, btc_d, usdt_d)
        self._MAX_HISTORY = 12    # 12 × 5min = 1 hour of data

    def _fetch(self) -> Optional[dict]:
        """Fetch global market data from CoinGecko."""
        try:
            headers = {}
            if self._api_key:
                headers["x-cg-demo-api-key"] = self._api_key
            resp = requests.get(
                self.COINGECKO_GLOBAL,
                headers=headers,
                timeout=10,
            )
            if resp.status_code != 200:
                log.warning(f"CoinGecko HTTP {resp.status_code}")
                return None
            data = resp.json().get("data", {})
            pcts = data.get("market_cap_percentage", {})
            return {
                "btc_d":   float(pcts.get("btc", 50.0)),
                "usdt_d":  float(pcts.get("usdt", 5.0)),
                "total_mcap": float(data.get("total_market_cap", {}).get("usd", 0)),
                "fetched_at": datetime.now(LOCAL_TZ).isoformat(),
            }
        except Exception as e:
            log.warning(f"CoinGecko fetch error: {e}")
            return None

    def get(self) -> dict:
        """Return macro context dict, cached for CACHE_SECS."""
        now = time.time()
        if self._cache and (now - self._cache_time) < self.CACHE_SECS:
            return self._cache

        fresh = self._fetch()
        if not fresh:
            # Return last cache or safe defaults
            return self._cache or self._neutral()

        # Update rolling history
        self._history.append((now, fresh["btc_d"], fresh["usdt_d"]))
        if len(self._history) > self._MAX_HISTORY:
            self._history.pop(0)

        result = self._classify(fresh)
        self._cache = result
        self._cache_time = now

        log.info(
            f"[MacroContext] BTC.D={fresh['btc_d']:.1f}% "
            f"USDT.D={fresh['usdt_d']:.1f}% | "
            f"kill={result['kill']} universe={result['universe']} "
            f"btc_d_roc={result['btc_d_roc']:+.2f}%/hr "
            f"usdt_d_roc={result['usdt_d_roc']:+.2f}%/hr"
        )
        return result

    def _rate_of_change_per_hour(self) -> tuple:
        """Calculate BTC.D and USDT.D rate of change per hour."""
        if len(self._history) < 2:
            return 0.0, 0.0
        oldest_ts, oldest_btc, oldest_usdt = self._history[0]
        newest_ts, newest_btc, newest_usdt = self._history[-1]
        elapsed_hours = max((newest_ts - oldest_ts) / 3600, 1 / 60)
        btc_roc  = (newest_btc  - oldest_btc)  / elapsed_hours
        usdt_roc = (newest_usdt - oldest_usdt) / elapsed_hours
        return round(btc_roc, 3), round(usdt_roc, 3)

    def _classify(self, data: dict) -> dict:
        btc_d  = data["btc_d"]
        usdt_d = data["usdt_d"]
        btc_d_roc, usdt_d_roc = self._rate_of_change_per_hour()

        # ── Kill switch conditions ────────────────────────────────────
        kill = False
        kill_reason = ""

        if usdt_d_roc > 0.4:
            kill = True
            kill_reason = f"USDT.D rising {usdt_d_roc:+.2f}%/hr — capital fleeing to stables"
        elif btc_d_roc > 0.8:
            kill = True
            kill_reason = f"BTC.D spiking {btc_d_roc:+.2f}%/hr — altcoin sell-off"

        # ── Universe selection ────────────────────────────────────────
        if kill:
            universe = "none"
        elif btc_d > 58:
            # High BTC dominance — BTC and large caps only
            universe = "btc_only"
        elif btc_d < 48 and btc_d_roc < 0:
            # Falling BTC dominance — altseason conditions
            universe = "full"
        elif btc_d_roc > 0.3:
            # BTC dominance rising moderately — large caps only
            universe = "large_cap"
        else:
            universe = "full"

        # ── Risk sentiment ────────────────────────────────────────────
        if usdt_d_roc > 0.2 or btc_d_roc > 0.4:
            sentiment = "risk_off"
        elif usdt_d_roc < -0.2 or btc_d_roc < -0.3:
            sentiment = "risk_on"
        else:
            sentiment = "neutral"

        return {
            "btc_d":       btc_d,
            "usdt_d":      usdt_d,
            "btc_d_roc":   btc_d_roc,
            "usdt_d_roc":  usdt_d_roc,
            "kill":        kill,
            "kill_reason": kill_reason,
            "universe":    universe,   # "none" | "btc_only" | "large_cap" | "full"
            "sentiment":   sentiment,  # "risk_on" | "neutral" | "risk_off"
            "fetched_at":  data["fetched_at"],
        }

    def _neutral(self) -> dict:
        """Safe defaults when CoinGecko is unreachable."""
        return {
            "btc_d": 50.0, "usdt_d": 5.0,
            "btc_d_roc": 0.0, "usdt_d_roc": 0.0,
            "kill": False, "kill_reason": "",
            "universe": "full", "sentiment": "neutral",
            "fetched_at": datetime.now(LOCAL_TZ).isoformat(),
        }

    def is_symbol_allowed(self, symbol: str, universe: str) -> bool:
        """Check if a given symbol is allowed under current universe rule."""
        if universe == "none":
            return False
        if universe == "full":
            return True
        sym = symbol.upper()
        BTC_ONLY = {"BTC/USDT"}
        LARGE_CAP = {"BTC/USDT", "ETH/USDT", "BNB/USDT"}
        if universe == "btc_only":
            return sym in BTC_ONLY
        if universe == "large_cap":
            return sym in LARGE_CAP
        return True
