"""Core types shared across the bot."""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone


@dataclass
class Trade:
    id: str
    symbol: str
    side: str
    amount: float
    price: float
    timestamp: str
    strategy: str = ""
    timeframe: str = ""
    status: str = "open"
    mode: str = "spot"
    leverage: int = 1
    sl_order_id: str = ""
    pnl: float = 0.0
    close_price: float = 0.0
    close_timestamp: str = ""
    mark_price: float = 0.0
    live_pnl: float = 0.0
    liquidation_pct: float = 0.0


@dataclass
class Signal:
    symbol: str
    action: str
    confidence: float
    strategy: str = ""
    timeframe: str = ""
    timestamp: str = ""
    indicators: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    risk_level: str = "MEDIUM"


@dataclass
class Position:
    symbol: str
    side: str
    amount: float
    entry_price: float
    mark_price: float = 0.0
    pnl: float = 0.0
    leverage: int = 1
