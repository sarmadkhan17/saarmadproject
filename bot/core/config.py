"""
Config Loader — CryptoBot v5
"""
import os
import logging
from pathlib import Path

log = logging.getLogger("Config")

BOT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR  = BOT_ROOT / "data"
LOGS_DIR  = BOT_ROOT / "logs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def load_env():
    env_path = BOT_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("\"'")
                    if val:
                        os.environ.setdefault(key, val)


def get_secret(key, default=""):
    return os.environ.get(key, default)


def get_exchange_config():
    load_env()
    return {
        "api_key":    get_secret("BINANCE_API_KEY"),
        "api_secret": get_secret("BINANCE_SECRET_KEY"),
        "demo":       get_secret("BINANCE_DEMO", "true").lower() == "true",
    }


def get_deepseek_key():
    load_env()
    return get_secret("DEEPSEEK_API_KEY")


def get_coingecko_key():
    load_env()
    return get_secret("COINGECKO_API_KEY")


def get_telegram_config():
    load_env()
    return {
        "token":   get_secret("TELEGRAM_TOKEN"),
        "chat_id": get_secret("TELEGRAM_CHAT_ID"),
    }


def create_demo_exchange(mode: str = "spot"):
    """Creates Binance Demo Exchange (spot or futures) via DemoExchangeAdapter."""
    from exchange.demo_api import DemoExchangeAdapter
    cfg = get_exchange_config()
    if not cfg["api_key"] or not cfg["api_secret"]:
        raise ValueError("Missing BINANCE_API_KEY or BINANCE_SECRET_KEY in .env")
    log.info(f"Creating Binance DEMO exchange — mode={mode}")
    return DemoExchangeAdapter(cfg["api_key"], cfg["api_secret"], mode)
