"""Backward-compatibility re-export shims for the flat bot/ structure.
All actual logic now lives in the sub-packages.
"""

# core
from core.config import *  # noqa: F403

# engine
from engine.bot import BaseBot, Trade, StateManager  # noqa: F401
from engine.spot import SpotBot  # noqa: F401
from engine.futures import FuturesBot  # noqa: F401

# models
from models.ai_strategy import AIStrategyEngine, make_features, make_labels  # noqa: F401

# risk
from risk.manager import RiskManager, MarketRegimeGate, ATRTrailingStop  # noqa: F401

# agents
from agents.coordinator import AgentCoordinator  # noqa: F401

# data
from data.feed import DataFeed, TrainingFeed  # noqa: F401
from data.ws_feed import BinanceWSPriceFeed  # noqa: F401

# exchange
from exchange.factory import ExchangeRouter, get_exchange_router  # noqa: F401
from exchange.demo_api import DemoExchangeAdapter  # noqa: F401

# tuning
from tuning.scanner import CoinScanner  # noqa: F401
from tuning.learner import SelfLearner  # noqa: F401

# notify
from notify.telegram import TelegramNotifier  # noqa: F401

# models (sub-models)
from models.gnn import GNNCorrelationFilter  # noqa: F401
from models.hmm import HMMRegimeModel  # noqa: F401
from models.rl_agent import RLTradeManager  # noqa: F401
