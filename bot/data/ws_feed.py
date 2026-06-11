"""
BinanceWSPriceFeed — lightweight WebSocket price cache.

Subscribes to Binance @miniTicker stream for all USDT pairs and keeps the
latest mark price in memory. Falls back gracefully to REST if the WS is
unavailable. The bot's DataFeed checks this cache first to avoid hitting
REST on every price lookup.
"""

import json
import logging
import threading
import time
from typing import Optional

log = logging.getLogger("BinanceWS")


class BinanceWSPriceFeed:
    WS_URL = "wss://stream.binance.com:9443/ws/!miniTicker@arr"

    def __init__(self):
        self._prices: dict = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._last_msg_ts: float = 0
        self._connected = False

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log.info("BinanceWSPriceFeed started")

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)
        self._connected = False

    def subscribe(self, symbol: str):
        pass  # miniTicker@arr receives all symbols; no per-symbol subscription needed

    def subscribe_many(self, symbols):
        pass  # same — all symbols arrive via the single stream

    def get_price(self, symbol: str) -> Optional[float]:
        with self._lock:
            return self._prices.get(symbol.upper())

    def is_fresh(self, max_age_secs: float = 30) -> bool:
        return self._connected and (time.time() - self._last_msg_ts) < max_age_secs

    def _run(self):
        try:
            from websocket import WebSocketApp
        except ImportError:
            log.warning("websocket-client not installed — price feed will use REST fallback only")
            return

        backoff = 1
        while not self._stop.is_set():
            try:
                ws = WebSocketApp(
                    self.WS_URL,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                ws.run_forever(ping_interval=20, ping_timeout=10)
            except Exception as e:
                log.warning(f"WS connection error: {e}")
            self._connected = False
            if self._stop.is_set():
                break
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    def _on_open(self, ws):
        self._connected = True
        log.info("Binance miniTicker WS connected")

    def _on_message(self, ws, message):
        try:
            arr = json.loads(message)
            if not isinstance(arr, list):
                return
            now = time.time()
            with self._lock:
                for t in arr:
                    sym_raw = t.get("s", "")
                    if not sym_raw.endswith("USDT"):
                        continue
                    base = sym_raw[:-4]
                    sym = f"{base}/USDT"
                    try:
                        self._prices[sym] = float(t.get("c", 0))
                    except (ValueError, TypeError):
                        continue
            self._last_msg_ts = now
        except Exception as e:
            log.debug(f"WS message parse error: {e}")

    def _on_error(self, ws, error):
        log.debug(f"WS error: {error}")

    def _on_close(self, ws, code, reason):
        self._connected = False
        log.info(f"WS closed: code={code} reason={reason}")

    def snapshot(self) -> dict:
        with self._lock:
            return dict(self._prices)
