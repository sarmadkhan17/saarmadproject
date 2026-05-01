"""
v3 Config Loader
Supports: spot demo, futures demo
Group chat ID format: -1001234567890
"""

import os
import logging
from pathlib import Path

log = logging.getLogger("Config")

BOT_ROOT = Path.home() / "cryptobot_v3"
DATA_DIR = BOT_ROOT / "data"
LOGS_DIR = BOT_ROOT / "logs"


def load_env():
    env_path = BOT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip()
                    if val:
                        os.environ[key] = val


def get_secret(key, default=""):
    return os.environ.get(key, default)


def get_exchange_config():
    load_env()
    return {
        "api_key":    get_secret("BINANCE_API_KEY"),
        "api_secret": get_secret("BINANCE_API_SECRET"),
    }


def get_groq_key():
    load_env()
    return get_secret("GROQ_API_KEY")


def get_telegram_config():
    """
    Returns Telegram config.
    Group chat IDs start with - (e.g., -1001234567890)
    Personal chat IDs are positive (e.g., 1807747201)
    """
    load_env()
    return {
        "token":   get_secret("TELEGRAM_TOKEN"),
        "chat_id": get_secret("TELEGRAM_CHAT_ID"),
    }


def create_demo_exchange(mode: str = "spot"):
    """
    Creates Binance Demo Exchange (spot or futures).
    Returns DemoExchangeAdapter that mimics ccxt interface.
    """
    from binance_demo import DemoExchangeAdapter
    cfg = get_exchange_config()

    if not cfg["api_key"] or not cfg["api_secret"]:
        raise ValueError("Missing BINANCE_API_KEY or BINANCE_API_SECRET in .env")

    log.info(f"Creating Binance DEMO exchange — mode={mode}")
    return DemoExchangeAdapter(cfg["api_key"], cfg["api_secret"], mode)
