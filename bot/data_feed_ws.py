"""
WebSocket price feed - Binance miniTicker streams.
Runs asyncio in a daemon thread. Falls back gracefully if unavailable.
"""

import asyncio
import json
import logging
import threading
import time
from typing import Dict, Optional, Set

log = logging.getLogger("WSPriceFeed")

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:
    WS_AVAILABLE = False


def _ccxt_to_ws(symbol: str) -> str:
    """BTC/USDT → btcusdt"""
    return symbol.replace("/", "").lower()


def _ws_to_ccxt(symbol: str) -> str:
    """BTCUSDT → BTC/USDT  (handles USDT, BTC, ETH, BNB, BUSD quotes)"""
    for quote in ("USDT", "BUSD", "BTC", "ETH", "BNB"):
        if symbol.endswith(quote):
            base = symbol[: -len(quote)]
            return f"{base}/{quote}"
    return symbol


class BinanceWSPriceFeed:
    """
    Live price feed via Binance WebSocket miniTicker streams.
    Thread-safe. Auto-reconnects on disconnect.
    """

    WS_BASE = "wss://stream.binance.com:9443/stream"
    _MIN_RECONNECT = 5
    _MAX_RECONNECT = 60

    def __init__(self):
        self._prices: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._subscribed: Set[str] = set()   # ccxt-format
        self._dirty = threading.Event()      # signals subscription change
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._connected = False

    # ── Public API ──────────────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """True if WS connected and receiving data."""
        return WS_AVAILABLE and self._connected

    def subscribe(self, symbol: str):
        """Register a symbol (BTC/USDT format). Thread-safe."""
        if symbol not in self._subscribed:
            self._subscribed.add(symbol)
            self._dirty.set()

    def subscribe_many(self, symbols):
        added = False
        for s in symbols:
            if s not in self._subscribed:
                self._subscribed.add(s)
                added = True
        if added:
            self._dirty.set()

    def get_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._prices.get(symbol)

    def start(self, symbols=None):
        if not WS_AVAILABLE:
            log.warning("websockets not installed — WS price feed disabled")
            return
        if self._running:
            return
        if symbols:
            self._subscribed.update(symbols)
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="WSPriceFeed"
        )
        self._thread.start()
        log.info("WS price feed started")

    def stop(self):
        self._running = False
        # Daemon thread will be cleaned up on process exit;
        # setting _running=False causes _reconnect_loop to exit on next iteration.

    # ── Internal asyncio loop ────────────────────────────────────────────────

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._reconnect_loop())
        except Exception as e:
            log.error(f"WS loop crashed: {e}")
        finally:
            self._connected = False
            self._loop.close()

    async def _reconnect_loop(self):
        delay = self._MIN_RECONNECT
        while self._running:
            if not self._subscribed:
                await asyncio.sleep(1)
                continue
            try:
                await self._stream()
                delay = self._MIN_RECONNECT
            except Exception as e:
                self._connected = False
                log.warning(f"WS error: {e} — reconnect in {delay}s")
                await asyncio.sleep(delay)
                delay = min(delay * 2, self._MAX_RECONNECT)

    async def _stream(self):
        streams = "/".join(
            f"{_ccxt_to_ws(s)}@miniTicker" for s in self._subscribed
        )
        url = f"{self.WS_BASE}?streams={streams}"
        log.info(f"WS connecting — {len(self._subscribed)} symbols")

        async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
            self._connected = True
            self._dirty.clear()
            log.info("WS connected")

            while self._running:
                # If subscriptions changed, break → reconnect with new list
                if self._dirty.is_set():
                    log.info("WS subscriptions changed — reconnecting")
                    break

                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                    msg = json.loads(raw)
                    data = msg.get("data", msg)
                    sym_raw = data.get("s", "")
                    price = data.get("c")
                    if sym_raw and price:
                        ccxt_sym = _ws_to_ccxt(sym_raw)
                        with self._lock:
                            self._prices[ccxt_sym] = float(price)
                except asyncio.TimeoutError:
                    pass  # normal between ticks; _running + _dirty checked above
                except Exception as e:
                    raise e
