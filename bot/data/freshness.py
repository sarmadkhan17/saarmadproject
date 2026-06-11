"""Per-consumer freshness contract for caches.

Replaces the scattered TTL constants (HMM._CACHE_TTL, OHLCV CANDLE_SECONDS,
AgentCoordinator.GROQ_CACHE_SECS/SLOW_CACHE_SECS) with one mechanism whose
rule is simple: every caller declares max_age_seconds for the value it's
about to read. A miss returns None; the caller decides what to do — no
silent staleness.

Used by:
  bot/data/feed.py (OHLCV)
  bot/agents/coordinator.py (Fear&Greed + Macro snapshots)
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, Optional


class CacheMiss(Exception):
    """Raised by a loader when an upstream source cannot provide data."""


class Freshness:
    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._values: Dict[str, Any] = {}
        self._times:  Dict[str, float] = {}

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._values[key] = value
            self._times[key]  = time.time()

    def get(self, key: str, max_age_seconds: float) -> Optional[Any]:
        with self._lock:
            t = self._times.get(key)
            if t is None:
                return None
            if (time.time() - t) > max_age_seconds:
                return None
            return self._values.get(key)

    def age_seconds(self, key: str) -> Optional[float]:
        with self._lock:
            t = self._times.get(key)
            if t is None:
                return None
            return time.time() - t

    def invalidate(self, key: str) -> None:
        with self._lock:
            self._values.pop(key, None)
            self._times.pop(key, None)

    def fetch(self, key: str, max_age_seconds: float,
              loader: Callable[[], Any]) -> Any:
        """Get a fresh value or call loader() and cache its result.

        The loader's exception (including CacheMiss) propagates to the caller.
        """
        cached = self.get(key, max_age_seconds)
        if cached is not None:
            return cached
        value = loader()
        self.set(key, value)
        return value
