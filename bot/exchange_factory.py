"""
Exchange Factory v4
Single abstraction for demo/live API routing.
- Training data ALWAYS from real Binance public API (no auth needed, full history)
- Execution follows BOT_EXECUTION_MODE env var (demo|l1ive)
- Migration path: set BOT_EXECUTION_MODE=live, restart — no code changes
"""

import os
import logging
import ccxt
from typing import Optional

log = logging.getLogger("ExchangeFactory")

_router_instance: Optional["ExchangeRouter"] = None


class ExchangeRouter:
    """
    Provides two exchanges:
    - .training:   real Binance public API (api.binance.com, no auth, full history)
    - .execution:  demo or live exchange for order placement (auth required)
    - .mode:       "demo" or "live"

    Usage:
        router = get_exchange_router()
        # Training data (always real Binance):
        ohlcv = router.training.fetch_ohlcv("BTC/USDT", "15m", limit=5000)
        # Live trading:
        price = router.execution.fetch_ticker("ETH/USDT")
    """

    def __init__(self, execution_mode: str = "demo"):
        self.mode = execution_mode.lower()

        # Training: always real Binance public API — no auth, no rate limits, full history
        self.training = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        log.info("Training exchange: real Binance public API (api.binance.com)")

        # Execution: follows BOT_EXECUTION_MODE
        self.execution = None
        self._init_execution()

    def _init_execution(self):
        from env_config import get_exchange_config, create_demo_exchange
        from binance_demo import DemoExchangeAdapter

        cfg = get_exchange_config()

        if self.mode == "live":
            if not cfg["api_key"] or not cfg["api_secret"]:
                raise ValueError("BOT_EXECUTION_MODE=live requires BINANCE_API_KEY and BINANCE_API_SECRET")
            self.execution = ccxt.binance({
                "apiKey": cfg["api_key"],
                "secret": cfg["api_secret"],
                "enableRateLimit": True,
                "options": {"defaultType": "future"},
            })
            log.info("Execution exchange: LIVE Binance (api.binance.com)")
        else:
            if cfg["api_key"] and cfg["api_secret"]:
                self.execution = create_demo_exchange(
                    "futures" if os.environ.get("BOT_MODE") == "futures" else "spot"
                )
            else:
                self.execution = DemoExchangeAdapter(
                    cfg.get("api_key", ""), cfg.get("api_secret", ""),
                    "futures" if os.environ.get("BOT_MODE") == "futures" else "spot"
                )
            log.info("Execution exchange: DEMO (demo-api.binance.com)")

    def get_mode_display(self) -> str:
        return self.mode.upper()

    def can_switch_to_live(self) -> bool:
        from env_config import get_exchange_config
        cfg = get_exchange_config()
        return bool(cfg.get("api_key") and cfg.get("api_secret"))

    @property
    def is_demo(self) -> bool:
        return self.mode == "demo"

    @property
    def is_live(self) -> bool:
        return self.mode == "live"


def get_exchange_router() -> ExchangeRouter:
    global _router_instance
    if _router_instance is None:
        mode = os.environ.get("BOT_EXECUTION_MODE", "demo")
        _router_instance = ExchangeRouter(execution_mode=mode)
    return _router_instance


def create_execution_exchange(mode: str = "futures"):
    router = get_exchange_router()
    if router.is_live:
        exec_ex = router.execution
        exec_ex.options["defaultType"] = "future" if mode == "futures" else "spot"
        return exec_ex
    from env_config import create_demo_exchange
    return create_demo_exchange(mode)
