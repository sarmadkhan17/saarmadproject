"""
Binance Demo Trading Client
Direct API integration — bypasses ccxt limitations for demo mode.

Demo Mode endpoints (per Binance docs):
- Spot:    https://demo-api.binance.com/api
- Futures: https://demo-fapi.binance.com

Why this exists:
ccxt internally calls /sapi/v1/* endpoints (capital config, etc.)
Demo mode only supports /api/v3/* (basic spot) and /fapi/v1/* (basic futures)
We bypass ccxt and call the API directly.
"""

import time
import hmac
import hashlib
import requests
import logging
from urllib.parse import urlencode
from typing import Optional, List, Dict

log = logging.getLogger("BinanceDemo")


class BinanceDemoClient:
    """
    Direct Binance API client for demo trading.
    Implements the methods our bot needs:
    - fetch_balance
    - fetch_ticker / fetch_tickers
    - fetch_ohlcv
    - create_market_order
    - fetch_order
    Plus futures-specific:
    - set_leverage
    - set_margin_type
    - get_position
    """

    def __init__(self, api_key: str, api_secret: str, mode: str = "spot"):
        self.api_key    = api_key
        self.api_secret = api_secret.encode() if isinstance(api_secret, str) else api_secret
        self.mode       = mode

        if mode == "spot":
            self.base_url   = "https://demo-api.binance.com"
            self.api_prefix = "/api/v3"
        elif mode == "futures":
            self.base_url   = "https://demo-fapi.binance.com"
            self.api_prefix = "/fapi/v1"
        else:
            raise ValueError(f"Mode must be 'spot' or 'futures', got: {mode}")

        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})
        log.info(f"Binance Demo Client initialized — mode={mode} base={self.base_url}")

    def _sign(self, params: dict) -> str:
        """Generate HMAC-SHA256 signature for signed requests."""
        query = urlencode(params)
        return hmac.new(self.api_secret, query.encode(), hashlib.sha256).hexdigest()

    def _public_get(self, path: str, params: dict = None) -> dict:
        """Public endpoint - no signature needed."""
        url    = f"{self.base_url}{self.api_prefix}{path}"
        params = params or {}
        try:
            r = self.session.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError as e:
            # Suppress invalid symbol errors (just means coin not on this market)
            if r.status_code == 400 and "Invalid symbol" in r.text:
                log.debug(f"Symbol not available: {params.get('symbol','?')}")
            else:
                log.error(f"Public GET {path} failed: {r.text}")
            raise

    def _signed_get(self, path: str, params: dict = None) -> dict:
        """Signed GET request."""
        params = params or {}
        params["timestamp"]  = int(time.time() * 1000)
        params["recvWindow"] = 10000
        params["signature"]  = self._sign(params)
        url = f"{self.base_url}{self.api_prefix}{path}"
        try:
            r = self.session.get(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError:
            log.error(f"Signed GET {path}: {r.text}")
            raise

    def _signed_post(self, path: str, params: dict = None) -> dict:
        """Signed POST request (orders, leverage)."""
        params = params or {}
        params["timestamp"]  = int(time.time() * 1000)
        params["recvWindow"] = 10000
        params["signature"]  = self._sign(params)
        url = f"{self.base_url}{self.api_prefix}{path}"
        try:
            r = self.session.post(url, params=params, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.exceptions.HTTPError:
            try:
                err_code = r.json().get("code", 0)
            except Exception:
                err_code = 0
            if err_code == -4046:
                log.debug(f"Signed POST {path}: {r.text}")
            else:
                log.error(f"Signed POST {path}: {r.text}")
            raise

    # ── Symbol normalization ────────────────────────────────────
    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Convert 'BTC/USDT' to 'BTCUSDT' for Binance API."""
        return symbol.replace("/", "")

    @staticmethod
    def _denormalize_symbol(symbol: str) -> str:
        """Convert 'BTCUSDT' to 'BTC/USDT' for our bot."""
        if "/" in symbol:
            return symbol
        for quote in ["USDT", "USDC", "BUSD", "BTC", "ETH"]:
            if symbol.endswith(quote):
                base = symbol[:-len(quote)]
                return f"{base}/{quote}"
        return symbol

    # ── Public Endpoints ────────────────────────────────────────
    def fetch_ticker(self, symbol: str) -> dict:
        """Get current price for a symbol."""
        sym = self._normalize_symbol(symbol)
        data = self._public_get("/ticker/24hr", {"symbol": sym})
        return {
            "symbol":      symbol,
            "last":        float(data["lastPrice"]),
            "bid":         float(data.get("bidPrice", 0) or 0),
            "ask":         float(data.get("askPrice", 0) or 0),
            "high":        float(data["highPrice"]),
            "low":         float(data["lowPrice"]),
            "volume":      float(data["volume"]),
            "quoteVolume": float(data["quoteVolume"]),
            "percentage":  float(data["priceChangePercent"]),
            "info":        data,
        }

    def fetch_tickers(self, symbols: list = None) -> dict:
        """Get all tickers or filtered list."""
        data    = self._public_get("/ticker/24hr")
        result  = {}
        wanted  = set(self._normalize_symbol(s) for s in symbols) if symbols else None
        for d in data:
            sym_native = d["symbol"]
            if wanted and sym_native not in wanted:
                continue
            sym = self._denormalize_symbol(sym_native)
            result[sym] = {
                "symbol":      sym,
                "last":        float(d["lastPrice"]),
                "bid":         float(d.get("bidPrice", 0) or 0),
                "ask":         float(d.get("askPrice", 0) or 0),
                "high":        float(d["highPrice"]),
                "low":         float(d["lowPrice"]),
                "volume":      float(d["volume"]),
                "quoteVolume": float(d["quoteVolume"]),
                "percentage":  float(d["priceChangePercent"]),
                "info":        d,
            }
        return result

    def fetch_ohlcv(self, symbol: str, timeframe: str = "1h", limit: int = 500) -> list:
        """Get OHLCV candle data. Returns list of [timestamp, open, high, low, close, volume]."""
        sym  = self._normalize_symbol(symbol)
        data = self._public_get("/klines", {
            "symbol":   sym,
            "interval": timeframe,
            "limit":    min(limit, 1000),
        })
        return [
            [int(c[0]), float(c[1]), float(c[2]),
             float(c[3]), float(c[4]), float(c[5])]
            for c in data
        ]

    # ── Account / Balance ──────────────────────────────────────
    def fetch_balance(self) -> dict:
        """Get account balances."""
        if self.mode == "spot":
            data    = self._signed_get("/account")
            result  = {"info": data, "free": {}, "used": {}, "total": {}}
            for b in data.get("balances", []):
                asset = b["asset"]
                free  = float(b["free"])
                lock  = float(b["locked"])
                if free + lock > 0:
                    result["free"][asset]   = free
                    result["used"][asset]   = lock
                    result["total"][asset]  = free + lock
                    result[asset]           = {"free": free, "used": lock, "total": free + lock}
            return result
        else:
            # Futures uses /v2/account
            try:
                url    = f"{self.base_url}/fapi/v2/account"
                params = {"timestamp": int(time.time() * 1000), "recvWindow": 10000}
                params["signature"] = self._sign(params)
                r      = self.session.get(url, params=params, timeout=15)
                r.raise_for_status()
                data   = r.json()
                result = {"info": data, "free": {}, "used": {}, "total": {}}
                for asset_data in data.get("assets", []):
                    asset = asset_data["asset"]
                    wb    = float(asset_data.get("walletBalance", 0))
                    cb    = float(asset_data.get("crossWalletBalance", 0))
                    if wb > 0 or cb > 0:
                        result["free"][asset]  = float(asset_data.get("availableBalance", 0))
                        result["used"][asset]  = wb - float(asset_data.get("availableBalance", 0))
                        result["total"][asset] = wb
                        result[asset] = {
                            "free":  float(asset_data.get("availableBalance", 0)),
                            "used":  wb - float(asset_data.get("availableBalance", 0)),
                            "total": wb,
                        }
                return result
            except Exception as e:
                log.warning(f"Futures balance fetch failed: {e}")
                return {"info": {}, "free": {}, "used": {}, "total": {}}

    # ── Orders ──────────────────────────────────────────────────
    def create_market_order(self, symbol: str, side: str, amount: float, params: dict = None) -> dict:
        """Place market order. Side = 'buy' or 'sell'."""
        sym = self._normalize_symbol(symbol)

        # Get symbol precision
        amount = self._round_amount(symbol, amount)

        order_params = {
            "symbol":   sym,
            "side":     side.upper(),
            "type":     "MARKET",
            "quantity": amount,
        }

        if params:
            if params.get("reduceOnly"):
                order_params["reduceOnly"] = "true"

        path = "/order"
        data = self._signed_post(path, order_params)

        # Format response like ccxt
        avg_price = 0.0
        filled    = float(data.get("executedQty", 0))
        if filled > 0:
            cum_quote = float(data.get("cummulativeQuoteQty", 0))
            if cum_quote > 0:
                avg_price = cum_quote / filled

        return {
            "id":      str(data.get("orderId", "")),
            "symbol":  symbol,
            "side":    side,
            "amount":  filled,
            "price":   avg_price,
            "average": avg_price,
            "status":  data.get("status", "unknown").lower(),
            "info":    data,
        }

    def create_stop_market_order(self, symbol: str, side: str, amount: float, stop_price: float, params: dict = None) -> dict:
        """Place STOP_MARKET order (exchange-side stop loss)."""
        sym = self._normalize_symbol(symbol)
        amount = self._round_amount(symbol, amount)

        order_params = {
            "symbol":      sym,
            "side":        side.upper(),
            "type":        "STOP_MARKET",
            "stopPrice":   self._round_price(symbol, stop_price),
            "workingType": "CONTRACT_PRICE",
        }

        close_pos = False
        if params:
            if params.get("closePosition"):
                close_pos = True
                order_params["closePosition"] = "true"
            elif params.get("reduceOnly"):
                order_params["reduceOnly"] = "true"

        # closePosition doesn't need quantity — Binance closes entire position
        if not close_pos:
            order_params["quantity"] = amount

        path = "/order"
        data = self._signed_post(path, order_params)

        return {
            "id":        str(data.get("orderId", "")),
            "symbol":    symbol,
            "side":      side,
            "amount":    float(data.get("origQty", amount)),
            "stopPrice": float(data.get("stopPrice", stop_price)),
            "status":    data.get("status", "unknown").lower(),
            "type":      "stop_market",
            "info":      data,
        }

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """Cancel an open order."""
        sym = self._normalize_symbol(symbol)
        params = {
            "symbol":  sym,
            "orderId": order_id,
            "timestamp": int(time.time() * 1000),
            "recvWindow": 10000,
        }
        params["signature"] = self._sign(params)
        url = f"{self.base_url}{self.api_prefix}/order"
        try:
            r = self.session.delete(url, params=params, timeout=15)
            r.raise_for_status()
            data = r.json()
        except requests.exceptions.HTTPError:
            log.error(f"Cancel order {symbol}: {r.text}")
            raise

        return {
            "id":     str(data.get("orderId", "")),
            "symbol": symbol,
            "status": data.get("status", "cancelled").lower(),
            "info":   data,
        }

    def fetch_open_orders(self, symbol: str = None) -> list:
        """Get all open orders for a symbol or all symbols."""
        params = {}
        if symbol:
            params["symbol"] = self._normalize_symbol(symbol)
        data = self._signed_get("/openOrders", params)

        return [
            {
                "id":      str(d.get("orderId", "")),
                "symbol":  self._denormalize_symbol(d["symbol"]),
                "side":    d.get("side", "").lower(),
                "type":    d.get("type", "").lower(),
                "amount":  float(d.get("origQty", 0)),
                "price":   float(d.get("price", 0)),
                "stopPrice": float(d.get("stopPrice", 0)),
                "status":  d.get("status", "unknown").lower(),
                "info":    d,
            }
            for d in data
        ]

    def fetch_order(self, order_id: str, symbol: str) -> dict:
        """Get order status."""
        sym = self._normalize_symbol(symbol)
        try:
            data = self._signed_get("/order", {
                "symbol":  sym,
                "orderId": order_id,
            })
            return {
                "id":     str(data.get("orderId", "")),
                "symbol": symbol,
                "status": data.get("status", "unknown").lower(),
                "filled": float(data.get("executedQty", 0)),
                "info":   data,
            }
        except Exception:
            return {"id": order_id, "symbol": symbol, "status": "unknown", "filled": 0}

    # ── Symbol Info / Precision ────────────────────────────────
    _exchange_info_cache = None
    _exchange_info_time  = 0

    def _get_exchange_info(self) -> dict:
        """Cache exchange info for 1 hour."""
        now = time.time()
        if self._exchange_info_cache and (now - self._exchange_info_time) < 3600:
            return self._exchange_info_cache
        if self.mode == "spot":
            data = self._public_get("/exchangeInfo")
        else:
            data = self._public_get("/exchangeInfo")
        self._exchange_info_cache = data
        self._exchange_info_time  = now
        return data

    def get_valid_symbols(self) -> set:
        """Returns set of valid trading symbols (e.g., 'BTCUSDT')."""
        try:
            info = self._get_exchange_info()
            valid = set()
            for s in info.get("symbols", []):
                if s.get("status") == "TRADING":
                    valid.add(s["symbol"])
            return valid
        except Exception as e:
            log.error(f"get_valid_symbols failed: {e}")
            return set()

    def _round_price(self, symbol: str, price: float) -> float:
        """Round price per symbol's PRICE_FILTER."""
        sym = self._normalize_symbol(symbol)
        try:
            info = self._get_exchange_info()
            for s in info.get("symbols", []):
                if s["symbol"] == sym:
                    for f in s["filters"]:
                        if f["filterType"] == "PRICE_FILTER":
                            tick = float(f["tickSize"])
                            if tick > 0:
                                precision = max(0, len(f"{tick:.10f}".rstrip("0").split(".")[-1]))
                                rounded = round(price / tick) * tick
                                return round(rounded, precision)
        except Exception:
            pass
        return round(price, 2)

    def _round_amount(self, symbol: str, amount: float) -> float:
        """Round amount per symbol's LOT_SIZE filter."""
        sym = self._normalize_symbol(symbol)
        try:
            info = self._get_exchange_info()
            for s in info.get("symbols", []):
                if s["symbol"] == sym:
                    for f in s["filters"]:
                        if f["filterType"] == "LOT_SIZE":
                            step = float(f["stepSize"])
                            if step > 0:
                                precision = max(0, len(f"{step:.10f}".rstrip("0").split(".")[-1]))
                                rounded   = (amount // step) * step
                                result    = round(rounded, precision)
                                if result <= 0:
                                    return round(step, precision)  # minimum valid quantity
                                return result
        except Exception:
            pass
        return round(amount, 6)

    # ── Futures-specific ───────────────────────────────────────
    def set_leverage(self, leverage: int, symbol: str) -> dict:
        """Set leverage for a futures symbol."""
        if self.mode != "futures":
            return {}
        sym = self._normalize_symbol(symbol)
        return self._signed_post("/leverage", {
            "symbol":   sym,
            "leverage": leverage,
        })

    def set_margin_mode(self, margin_type: str, symbol: str) -> dict:
        """Set margin type: 'isolated' or 'cross'."""
        if self.mode != "futures":
            return {}
        sym = self._normalize_symbol(symbol)
        try:
            return self._signed_post("/marginType", {
                "symbol":     sym,
                "marginType": margin_type.upper(),
            })
        except Exception as e:
            # -4046 = already set, suppress this noise
            if "-4046" in str(e) or "No need to change" in str(e):
                return {"status": "already_set"}
            raise

    def get_position(self, symbol: str = None) -> list:
        """Get current futures positions."""
        if self.mode != "futures":
            return []
        url    = f"{self.base_url}/fapi/v2/positionRisk"
        params = {"timestamp": int(time.time() * 1000), "recvWindow": 10000}
        if symbol:
            params["symbol"] = self._normalize_symbol(symbol)
        params["signature"] = self._sign(params)
        r      = self.session.get(url, params=params, timeout=15)
        r.raise_for_status()
        data   = r.json()
        return [
            {
                "symbol":       self._denormalize_symbol(p["symbol"]),
                "side":         "long" if float(p["positionAmt"]) > 0 else "short",
                "amount":       abs(float(p["positionAmt"])),
                "entry_price":  float(p["entryPrice"]),
                "mark_price":   float(p["markPrice"]),
                "pnl":          float(p["unRealizedProfit"]),
                "leverage":     int(p.get("leverage", 1)),
            }
            for p in data
            if float(p.get("positionAmt", 0)) != 0
        ]

    # ── Sandbox mode (no-op for compatibility) ─────────────────
    def set_sandbox_mode(self, enabled: bool):
        """ccxt compatibility - we're already in demo mode."""
        pass

    def fetch_my_trades(self, symbol: str, limit: int = 50) -> list:
        """Fetch recent filled trades for a symbol (for accurate close PnL)."""
        sym = self._normalize_symbol(symbol)
        params = {
            "symbol":     sym,
            "limit":      limit,
            "timestamp":  int(time.time() * 1000),
            "recvWindow": 10000,
        }
        params["signature"] = self._sign(params)
        url = f"{self.base_url}{self.api_prefix}/userTrades"
        try:
            r = self.session.get(url, params=params, timeout=15)
            r.raise_for_status()
            return [
                {
                    "symbol":  symbol,
                    "price":   float(t.get("price", 0)),
                    "qty":     float(t.get("qty", 0)),
                    "quoteQty": float(t.get("quoteQty", 0)),
                    "side":    t.get("side", "").lower(),
                    "time":    t.get("time", 0),
                }
                for t in r.json()
            ]
        except Exception:
            return []


# ── Adapter Wrapper ──────────────────────────────────────────────
# Wraps the demo client to look like ccxt for our existing bot code

class DemoExchangeAdapter:
    """
    Wraps BinanceDemoClient with ccxt-like interface.
    This lets us drop it into our existing bot without changing other files.
    """

    def __init__(self, api_key: str, api_secret: str, mode: str = "spot"):
        self.client = BinanceDemoClient(api_key, api_secret, mode)
        self.mode   = mode

    def fetch_balance(self):
        return self.client.fetch_balance()

    def fetch_ticker(self, symbol):
        return self.client.fetch_ticker(symbol)

    def fetch_tickers(self, symbols=None):
        return self.client.fetch_tickers(symbols)

    def fetch_ohlcv(self, symbol, timeframe="1h", limit=500):
        return self.client.fetch_ohlcv(symbol, timeframe, limit)

    def create_market_order(self, symbol, side, amount, params=None):
        return self.client.create_market_order(symbol, side, amount, params)

    def create_stop_market_order(self, symbol, side, amount, stop_price, params=None):
        return self.client.create_stop_market_order(symbol, side, amount, stop_price, params)

    def cancel_order(self, order_id, symbol):
        return self.client.cancel_order(order_id, symbol)

    def fetch_open_orders(self, symbol=None):
        return self.client.fetch_open_orders(symbol)

    def fetch_order(self, order_id, symbol):
        return self.client.fetch_order(order_id, symbol)

    def set_leverage(self, leverage, symbol):
        return self.client.set_leverage(leverage, symbol)

    def set_margin_mode(self, margin_type, symbol):
        return self.client.set_margin_mode(margin_type, symbol)

    def get_position(self, symbol=None):
        return self.client.get_position(symbol)

    def fetch_my_trades(self, symbol, limit=50):
        return self.client.fetch_my_trades(symbol, limit)

    def set_sandbox_mode(self, enabled):
        self.client.set_sandbox_mode(enabled)

    def get_valid_symbols(self):
        return self.client.get_valid_symbols()
