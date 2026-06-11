"""
Base Bot - Shared Core Strategy
Both SpotBot and FuturesBot inherit from this.
100% identical strategy — only order direction differs.

Shared:
- ML models (RF + LightGBM + LSTM)
- Agent system (Groq confidence gate)
- Risk management (ATR stops, Kelly, portfolio heat)
- Data feed (validation + caching)
- Coin scanner (autonomous selection)
- Self learner (improves over time)
- Telegram notifier (alerts + commands)
"""

import json
import os
import time
import threading
import logging
import logging.handlers
import shutil
from datetime import datetime, timezone, timedelta
from core.tz import LOCAL_TZ
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

import numpy as np
import yaml
import dataclasses
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent))

from core.config   import (get_exchange_config, get_deepseek_key,
                          get_coingecko_key, DATA_DIR, LOGS_DIR, BOT_ROOT)
from data.feed    import DataFeed
from data.ws_feed import BinanceWSPriceFeed
from agents.coordinator       import AgentCoordinator
from risk.manager import RiskManager
from notify.telegram     import TelegramNotifier
from tuning.scanner import CoinScanner

# v5: rule-based + LLM reasoning (ML stack removed)
from engine.technical_agent  import TechnicalAgent
from engine.correlation_check import CorrelationCheck
from agents.macro_context    import MacroContextAgent
from agents.microstructure   import MicrostructureAgent
from agents.trade_memory     import TradeMemory
from agents.shadow_tracker   import ShadowTracker
from agents import trade_memory as trade_memory_mod
from agents import vector_store
from agents import llm_reasoning
from agents import skeptic as skeptic_agent
from agents import usage_tracker

# v5: New 4-layer architecture
from engine.profiles       import TradingProfile
from engine.smc_agent      import SMCAgent
from engine.ensemble       import EnsembleEngine
from engine.risk_agent     import RiskDecisionAgent
from engine.execution_engine import ExecutionEngine

DATA = DATA_DIR


def setup_logging(config: dict, log_file: str = "bot.log"):
    """Setup rotating log handler. Same for all bot modes."""
    log_cfg  = config.get("logging", {})
    log_path = LOGS_DIR / log_file
    log_path.parent.mkdir(exist_ok=True)
    handler  = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes    = log_cfg.get("max_bytes", 10_485_760),
        backupCount = log_cfg.get("backup_count", 5),
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s [%(levelname)s] %(message)s",
        handlers = [handler],
        force    = True,
    )


@dataclass
class Trade:
    id: str
    symbol: str
    side: str           # "buy"/"sell" for spot | "long"/"short" for futures
    amount: float
    price: float
    timestamp: str
    strategy: str
    timeframe: str
    status: str
    mode: str = "spot"  # "spot" or "futures"
    leverage: int = 1
    pnl: float = 0.0
    close_price: float = 0.0
    close_timestamp: str = ""
    sl_order_id: str = ""
    entry_price: float = 0.0
    live_pnl: float = 0.0
    mark_price: float = 0.0
    duration_hours: float = 0.0


class StateManager:
    """
    Manages trade state.
    Spot and Futures use separate state files so they don't interfere.
    Batches writes to reduce I/O — flushes every 10s or on demand.
    """

    _FLUSH_INTERVAL = 10  # seconds

    def __init__(self, filename: str = "state.json"):
        self.path = DATA / filename
        DATA.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._dirty = False
        self._last_flush = time.time()
        self._flush_thread = None
        self._stop_flushing = threading.Event()
        self.state = self._load()
        self._start_flush_thread()

    def _start_flush_thread(self):
        self._flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self._flush_thread.start()

    def _periodic_flush(self):
        while not self._stop_flushing.wait(self._FLUSH_INTERVAL):
            with self._lock:
                if self._dirty:
                    self._dirty = False
                    self._do_save_locked()

    def shutdown(self):
        self._stop_flushing.set()
        with self._lock:
            if self._dirty:
                self._dirty = False
                self._do_save_locked()
        if self._flush_thread:
            self._flush_thread.join(timeout=5)

    def _load(self):
        with self._lock:
            if self.path.exists():
                with open(self.path) as f:
                    return json.load(f)
            return {
                "trades": [], "signals": [],
                "stats": {
                    "total_trades": 0, "wins": 0, "losses": 0,
                    "total_pnl": 0.0,
                    "start_time": datetime.now(LOCAL_TZ).isoformat(),
                },
                "last_update": datetime.now(LOCAL_TZ).isoformat(),
            }

    def save(self, immediate: bool = False):
        """Mark state dirty for async flush. Set immediate=True to force write now."""
        with self._lock:
            self._dirty = True
            if immediate:
                self._dirty = False
                self._do_save_locked()

    def _do_save_locked(self):
        """Write state to disk. Caller must hold self._lock."""
        self.state["last_update"] = datetime.now(LOCAL_TZ).isoformat()
        tmp_path = self.path.with_suffix(".tmp.json")
        with open(tmp_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
        tmp_path.replace(self.path)
        try:
            shutil.copy2(str(self.path),
                         str(self.path.with_suffix(".backup.json")))
        except Exception:
            pass
        self._last_flush = time.time()

    def add_trade(self, trade: Trade):
        self.state["trades"].append(asdict(trade))
        self.state["stats"]["total_trades"] += 1
        self.save()

    def close_trade(self, trade_id, price, pnl):
        for t in self.state["trades"]:
            if t["id"] == trade_id:
                t["status"]          = "closed"
                t["close_price"]     = price
                t["pnl"]             = round(t.get("pnl", 0.0) + pnl, 8)
                t["close_timestamp"] = datetime.now(LOCAL_TZ).isoformat()
                self.state["stats"]["total_pnl"] += pnl
                if t["pnl"] > 0: self.state["stats"]["wins"]   += 1
                else:             self.state["stats"]["losses"] += 1
                break
        self.save()

    def partial_close_trade(self, trade_id, closed_amount, pnl):
        for t in self.state["trades"]:
            if t["id"] == trade_id:
                t["amount"] = max(0.0, round(t["amount"] - closed_amount, 8))
                t["pnl"]    = round(t.get("pnl", 0.0) + pnl, 8)
                self.state["stats"]["total_pnl"] += pnl
                if pnl > 0:
                    self.state["stats"]["wins"] += 1
                else:
                    self.state["stats"]["losses"] += 1
                break
        self.save()

    def update_trade_amount(self, trade_id, new_amount):
        for t in self.state["trades"]:
            if t["id"] == trade_id:
                t["amount"] = round(new_amount, 8)
                break
        self.save()

    def update_trade_sl(self, trade_id, sl_order_id):
        for t in self.state["trades"]:
            if t["id"] == trade_id:
                t["sl_order_id"] = sl_order_id
                break
        self.save()

    def update_trade_live_pnl(self, trade_id: str, live_pnl: float,
                               mark_price: float, duration_hours: float) -> None:
        with self._lock:
            for t in self.state["trades"]:
                if t.get("id") == trade_id and t.get("status") == "open":
                    t["live_pnl"]       = round(live_pnl, 6)
                    t["mark_price"]     = mark_price
                    t["duration_hours"] = round(duration_hours, 2)
                    break
        self.save()

    def add_signal(self, signal):
        self.state["signals"].append(signal)
        self.state["signals"] = self.state["signals"][-500:]
        self.save()

    def get_open_trades(self):
        return [t for t in self.state["trades"] if t["status"] == "open"]

    def get_all_trades(self):
        trades = list(self.state["trades"])
        archive_path = self.path.with_name(self.path.stem + "_archive.json")
        if archive_path.exists():
            try:
                with open(archive_path) as f:
                    trades.extend(json.load(f))
            except Exception:
                pass
        return trades


class BaseBot:
    """
    Core trading logic shared by SpotBot and FuturesBot.
    Subclasses only override:
    - _setup_exchange()   → spot vs futures connection
    - _place_buy()        → market buy vs open long
    - _place_sell()       → market sell vs open short
    - _place_close()      → spot sell vs reduce-only close
    - _calc_pnl()         → simple pnl vs leveraged pnl
    """

    MODE = "spot"   # Override in subclass

    def __init__(self, config_file: str = "config.yaml", log_file: str = "bot.log"):
        self.config_file = config_file
        cfg_path = BOT_ROOT / config_file
        with open(cfg_path) as f:
            self.config = yaml.safe_load(f)

        setup_logging(self.config, log_file)
        self.log = logging.getLogger(self.__class__.__name__)

        # Exchange setup (overridden by subclass)
        self.exchange = self._setup_exchange()

        # Settings
        risk               = self.config.get("risk", {})
        self.scan_interval = self.config.get("bot", {}).get("scan_interval_seconds", 30)
        # Short-TTL cache of DeepSeek Actor verdicts, keyed by symbol → quantized
        # setup signature. Keeps the LLM (and RAG) off the per-symbol hot path
        # when a setup is unchanged across consecutive 30s scans. See Gate 5.
        self._actor_cache: dict = {}
        self.max_open      = risk.get("max_open_trades", 8)
        self.min_conf      = self.config.get("strategy", {}).get("min_confidence", 0.52)
        self.htf_filter_mode = self.config.get("strategy", {}).get("htf_filter_mode", "strict")

        # State — separate files for spot and futures
        state_file    = "state.json" if self.MODE == "spot" else "futures_state.json"
        self.state    = StateManager(state_file)

        # WebSocket price feed (lower latency than REST polling)
        self.ws_feed  = BinanceWSPriceFeed()
        self.ws_feed.start()

        # Shared core systems
        self.feed       = DataFeed(self.exchange, ws_feed=self.ws_feed)
        self.risk       = RiskManager(self.config)
        self.agents     = AgentCoordinator()
        self.notifier   = TelegramNotifier()
        self.scanner    = CoinScanner(self.config)

        self._balance_cache: Optional[float] = None
        self._symbol_loss_cooldown: dict = {}
        self._sl_failure_cooldown: dict = {}
        self._symbol_locks: dict = {}
        self._symbol_locks_guard = threading.Lock()
        self._entry_contexts_file = DATA / f"entry_contexts_{self.MODE}.json"

        # ── v5: rule-based agents + DeepSeek reasoning ──────────────────────
        self.profile = TradingProfile.from_config(self.config)
        self.log.info(f"Trading profile: {self.profile.name} | "
                      f"min_conf={self.profile.min_confidence} | "
                      f"agents={self.profile.min_agent_agreement}/3 | "
                      f"net_threshold={self.profile.net_score_threshold}")

        # Initialize DeepSeek usage tracker + trade memory
        usage_tracker.init_tracker(DATA_DIR)
        self.trade_memory = TradeMemory(DATA_DIR / "trade_memory.db")
        # Embed any pre-existing trades/critiques so vector RAG works from
        # the first scan (no-op if embeddings are unavailable or already done).
        try:
            self.trade_memory.backfill_embeddings()
        except Exception as e:
            self.log.warning(f"Embedding backfill skipped: {e}")

        # Shadow tracker: rejected signals tracked forward as hypothetical
        # trades so Meta-Judge/operator learn whether gates block winners
        # or save losses. Observational only — never auto-tunes thresholds.
        try:
            self.shadow_tracker = ShadowTracker(
                DATA_DIR / "trade_memory.db", self.MODE,
                self.config.get("shadow_tracker") or {},
            )
        except Exception as e:
            self.log.warning(f"ShadowTracker disabled: {e}")
            self.shadow_tracker = None
        self._shadow_cycle_count = 0

        # New agents
        self.macro_context = MacroContextAgent(get_coingecko_key())
        self.microstructure = MicrostructureAgent()

        # Ensemble: SMC (structure) + Technical (indicators) + MacroFlow (dominance)
        self.smc_agent   = SMCAgent()
        self.tech_agent  = TechnicalAgent()
        from engine.macro_agent import MacroFlowAgent
        self.ensemble = EnsembleEngine(
            self.smc_agent, self.tech_agent, MacroFlowAgent(self.agents.macro),
            trend_filter=self.config.get("trend_filter") or {},
        )

        # Risk decision agent (GNN replaced by CorrelationCheck pass-through)
        self.risk_agent  = RiskDecisionAgent(self.risk, CorrelationCheck())
        self.execution   = ExecutionEngine(
            self.exchange, self.state, self.notifier, self.MODE,
            get_leverage_fn=self._get_leverage,
        )
        self.execution.sl_atr_mult = self.profile.stop_loss_atr_mult

        # Entry-context store for the Judge (trade_id -> context dict)
        # Persisted to disk so restarts don't wipe contexts for open trades.
        self._entry_contexts: dict = self._load_entry_contexts()

        self.log.info(f"CryptoBot v5 initialized [{self.MODE.upper()}] — "
                      f"DeepSeek reasoning active, ML stack removed")

        # Compatibility shims for removed ML objects. PnL-recording call sites
        # throughout the codebase call self.ai.record_trade_result and
        # self.rl_agent.* — these are now no-ops (the data lives in TradeMemory).
        class _NoOpAI:
            def record_trade_result(self, *a, **k): pass
            def predict(self, *a, **k): return {"action": "HOLD", "confidence": 0.5}
        class _NoOpRL:
            def record_external_close(self, *a, **k): pass
            def record_step(self, *a, **k): pass
            def prune_pending(self, *a, **k): pass
            def decide(self, *a, **k): return ("HOLD", 0.0)
        self.ai = _NoOpAI()
        self.rl_agent = _NoOpRL()

    def _setup_exchange(self):
        """Override in subclass to set spot or futures."""
        raise NotImplementedError

    def _place_buy(self, symbol, amount) -> Optional[dict]:
        """Override in subclass. Spot=market buy, Futures=open long."""
        raise NotImplementedError

    def _place_sell(self, symbol, amount) -> Optional[dict]:
        """Override in subclass. Spot=market sell, Futures=open short."""
        raise NotImplementedError

    def _place_close(self, symbol, amount, side) -> Optional[dict]:
        """Override in subclass. Closes an open position."""
        raise NotImplementedError

    def _calc_pnl(self, trade, close_price) -> float:
        """Override in subclass. Spot=simple, Futures=leveraged."""
        raise NotImplementedError

    def _post_scan(self, symbols):
        """Override in subclass. Called after scanner returns new watchlist."""

    def _maybe_swap_profile(self):
        """Hot-swap trading profile if the config YAML changed without a full retrain."""
        try:
            cfg_path = BOT_ROOT / self.config_file
            if not cfg_path.exists():
                return
            with open(cfg_path) as f:
                raw = yaml.safe_load(f) or {}
            new_name = raw.get("strategy", {}).get("trading_profile", self.profile.name)
            if new_name != self.profile.name:
                from engine.profiles import TradingProfile
                self.profile = dataclasses.replace(TradingProfile.load(new_name))
                self.execution.sl_atr_mult = self.profile.stop_loss_atr_mult
                self.log.info(f"Profile hot-swapped → {self.profile.name}")
        except Exception as e:
            self.log.warning(f"Profile hot-swap check failed: {e}")

    # ── ML training methods removed in v5 ──────────────────────────────
    # No RF/LightGBM/HMM training. The bot reasons live via rule-based agents
    # + DeepSeek. Self-improvement happens through the Judge/Meta-Judge verbal
    # feedback loop, not model retraining.

    def place_order_with_confirmation(
        self, symbol, side, amount, params=None, max_retries=3
    ) -> Optional[dict]:
        """
        Place order with confirmation loop.
        Shared by both spot and futures.
        Retries on failure, verifies fill.
        """
        for attempt in range(max_retries):
            try:
                if params:
                    order = self.exchange.create_market_order(
                        symbol, side, amount, params=params
                    )
                else:
                    order = self.exchange.create_market_order(symbol, side, amount)

                order_id = order.get("id")
                if not order_id:
                    continue

                time.sleep(1)
                try:
                    filled = self.exchange.fetch_order(order_id, symbol)
                    status = filled.get("status", "unknown")
                    if status in ["closed", "filled"]:
                        self.log.info(
                            f"Order confirmed: {side.upper()} {amount:.6f} {symbol}"
                        )
                        return filled

                    elif status == "open":
                        return filled

                except Exception:
                    return order  # Testnet may not support fetch_order

            except Exception as e:
                err_str = str(e).lower()
                if "insufficient" in err_str or "balance" in err_str:
                    self.log.error(f"Insufficient funds: {symbol}")
                    return None
                if "invalid" in err_str and "order" in err_str:
                    self.log.error(f"Invalid order {symbol}: {e}")
                    return None
                self.log.error(f"Order error {symbol} attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        return None

    def _get_symbol_lock(self, symbol: str) -> threading.Lock:
        with self._symbol_locks_guard:
            if symbol not in self._symbol_locks:
                self._symbol_locks[symbol] = threading.Lock()
            return self._symbol_locks[symbol]

    def get_price(self, symbol) -> float:
        price = self.feed.get_live_price(symbol)
        return price or 0.0

    def get_atr(self, symbol) -> float:
        tf = self.config.get("scanner", {}).get("timeframe", "15m")
        return self.feed.get_atr(symbol, tf, 14)

    def _record_shadow(self, symbol, gate, reason, ensemble, micro, regime, macro):
        """Record a rejected directional signal as a shadow trade.
        Must never break the scan loop — swallow everything."""
        if not self.shadow_tracker:
            return
        try:
            self.shadow_tracker.record_rejection(
                symbol=symbol,
                side="long" if ensemble.action == "BUY" else "short",
                gate=gate, reason=reason,
                entry_price=self.get_price(symbol),
                atr=self.get_atr(symbol),
                profile=self.profile,
                ctx={
                    "regime": regime,
                    "ensemble_score": ensemble.net_score,
                    "confidence": ensemble.confidence,
                    "ob_imbalance": getattr(micro, "ob_imbalance", 0.0),
                    "cvd_direction": getattr(micro, "cvd_direction", None),
                    "cvd_divergence": getattr(micro, "cvd_divergence", False),
                    "btc_d": (macro or {}).get("btc_d", 0.0),
                    "usdt_d": (macro or {}).get("usdt_d", 0.0),
                },
            )
        except Exception as e:
            self.log.debug(f"shadow record failed {symbol}: {e}")

    def _resolve_shadows(self):
        """Resolve open shadow trades against candles every N cycles."""
        if not self.shadow_tracker:
            return
        try:
            every = int((self.config.get("shadow_tracker") or {})
                        .get("resolve_every_n_cycles", 10))
            self._shadow_cycle_count += 1
            if self._shadow_cycle_count % max(1, every) != 0:
                return
            resolved = self.shadow_tracker.resolve_open(self.feed.fetch_ohlcv)
            if resolved:
                self.log.info(f"SHADOW resolved {len(resolved)} hypothetical trades")
        except Exception as e:
            self.log.debug(f"shadow resolve failed: {e}")

    def get_usdt_balance(self) -> float:
        try:
            bal = self.exchange.fetch_balance()
            value = float(bal["total"].get("USDT", 0.0))
            self._balance_cache = value
            return value
        except Exception as e:
            self.log.error(f"Balance error: {e}")
            return self._balance_cache or 0.0

    def get_htf_bias(self, dfs) -> str:
        """Higher timeframe bias — prevents trading against major trend."""
        votes = []
        for tf in ["4h", "1d"]:
            df = dfs.get(tf)
            if df is None or len(df) < 50:
                continue
            close = df["close"]
            ema20 = close.ewm(span=20).mean().iloc[-1]
            ema50 = close.ewm(span=50).mean().iloc[-1]
            price = float(close.iloc[-1])
            if price > ema20 > ema50:   votes.append("BUY")
            elif price < ema20 < ema50: votes.append("SELL")
            else:                       votes.append("NEUTRAL")

        if votes.count("BUY") >= 1 and "SELL" not in votes:   return "BUY"
        elif votes.count("SELL") >= 1 and "BUY" not in votes: return "SELL"
        return "NEUTRAL"

    def _resolve_close_pnl(self, trade: dict, symbol: str, detection_price: float, fraction: float) -> tuple:
        """
        After a close order is placed, look up the actual fill to get the real exit
        price and PnL. Always uses _calc_pnl (not exchange realizedPnl) because:
        - exchange realizedPnl on fills omits the leverage multiplier on the demo exchange
        - partial-close fraction would be applied twice (fill is already for the partial qty)
        Falls back to detection_price if no matching fill is found.
        Returns (pnl, actual_price).
        """
        try:
            closing_side = "sell" if trade["side"] == "long" else "buy"
            fills = self.exchange.fetch_my_trades(symbol, limit=10)
            closing_fills = [f for f in fills if f.get("side", "").lower() == closing_side]
            if closing_fills:
                best = max(closing_fills, key=lambda x: x["time"])
                actual_price = float(best.get("price", detection_price))
                return self._calc_pnl(trade, actual_price) * fraction, actual_price
        except Exception as e:
            self.log.debug(f"_resolve_close_pnl fallback to detection_price for {symbol}: {e}")
        return self._calc_pnl(trade, detection_price) * fraction, detection_price

    def check_exits(self):
        """
        Check open trades for exit conditions.
        Pre-fetches ATR + OHLCV in parallel; passes profile for exit engine params.
        """
        trades = self.state.get_open_trades()
        if not trades:
            return
        symbols = list(set(t["symbol"] for t in trades))
        atr_cache   = {}
        ohlcv_cache = {}
        tf = self.config.get("scanner", {}).get("timeframe", "15m")

        def _fetch_data(s):
            atr = self.get_atr(s)
            df  = self.feed.fetch_ohlcv(s, tf, limit=30)
            return s, atr, df

        with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as ex:
            for s, atr, df in ex.map(_fetch_data, symbols):
                atr_cache[s]   = atr
                ohlcv_cache[s] = df

        def _get_atr_cached(symbol):
            return atr_cache.get(symbol, 0.0)

        def _get_ohlcv_cached(symbol):
            return ohlcv_cache.get(symbol)

        # Reversal exit uses the latest regime context. check_exits runs before
        # this cycle's regime detection, so we read the previous cycle's context
        # (stored on self._regime_ctx) — fine since reversal is 4h/15-min based.
        exits = self.risk.check_exits(
            trades, self.get_price, _get_atr_cached, _get_ohlcv_cached, self.profile,
            reversal=getattr(self, "_regime_ctx", None),
        )
        for trade, price, reason, fraction in exits:
            close_amount = trade["amount"] * fraction
            _sym_lk = self._get_symbol_lock(trade["symbol"])
            with _sym_lk:
                if trade["id"] not in {t["id"] for t in self.state.get_open_trades()}:
                    continue  # already closed by RL or another path
                order = self._place_close(
                    trade["symbol"], close_amount, trade["side"]
                )
            if order:
                pnl, price = self._resolve_close_pnl(trade, trade["symbol"], price, fraction)
                if fraction >= 1.0:
                    self.state.close_trade(trade["id"], price, pnl)
                    self.risk.cleanup_trade(trade["id"])
                    self._cancel_exchange_stop_loss(
                        trade["symbol"], trade.get("sl_order_id", "")
                    )
                else:
                    self.state.partial_close_trade(trade["id"], close_amount, pnl)
                    if self.MODE == "futures" and trade.get("sl_order_id"):
                        remaining = trade["amount"] - close_amount
                        if remaining > 0:
                            self._cancel_exchange_stop_loss(
                                trade["symbol"], trade.get("sl_order_id", "")
                            )
                            atr = _get_atr_cached(trade["symbol"])
                            # TP1: move exchange SL to breakeven (atr=0 → entry ± min_pct buffer)
                            is_tp1 = "PARTIAL_TP1" in reason
                            sl_atr = 0.0 if is_tp1 else (atr if atr and atr > 0 else 0.0)
                            if sl_atr > 0 or is_tp1:
                                new_sl_id = self.execution._place_sl(
                                    trade["symbol"], trade["side"], remaining,
                                    float(trade["price"]), sl_atr,
                                )
                                if new_sl_id:
                                    self.state.update_trade_sl(trade["id"], new_sl_id)
                                    label = "breakeven" if is_tp1 else f"ATR×{atr:.4f}"
                                    self.log.info(f"SL re-placed [{label}] {trade['symbol']}: {remaining:.4f} remaining")
                self.ai.record_trade_result(trade["symbol"], pnl)
                self.agents.record_trade_result(trade["symbol"], pnl)
                self.risk.record_trade_result(pnl, self.get_usdt_balance())
                self.rl_agent.record_external_close(trade["id"], pnl)
                if fraction >= 1.0:
                    self._on_trade_closed(trade, price, pnl, reason)
                if pnl < 0:
                    self._symbol_loss_cooldown[trade["symbol"]] = datetime.now(LOCAL_TZ)
                icon    = "TP" if pnl > 0 else "SL"
                pct_str = f" ({fraction*100:.0f}%)" if fraction < 1.0 else ""
                self.log.info(f"{icon} EXIT{pct_str} {trade['symbol']} | PnL={pnl:+.4f} | {reason}")
                res_icon = "✅" if pnl > 0 else "❌"
                pnl_icon = "🟢" if pnl >= 0 else "🔴"
                self.notifier.send_alert(
                    f"{res_icon} <b>{icon} EXIT{pct_str}</b> · <b>{trade['symbol']}</b>\n"
                    f"  {pnl_icon} PnL: ${pnl:+.2f} USDT\n"
                    f"  📋 Reason: {reason}"
                )

    def _rl_manage_trades(self, regime_ctx=None):
        """RL position management removed in v5. ExitEngine (check_exits) owns
        all exits — partial TP, breakeven clamp, ATR trail, swing structure."""
        return

    def analyze_symbol(self, symbol, balance, open_trades, regime_ctx=None, btc_1h_return=None,
                        pre_multi=None, pre_train=None):
        """Decision pipeline v5:
        Gate 0/1 macro kill+universe → ensemble (SMC+Tech+MacroFlow) →
        Gate 4 microstructure confirm/kill → Gate 5 DeepSeek Actor →
        Gate 6 risk/sizing → execution. Records entry context for the Judge."""
        try:
            # Per-symbol cooldowns
            cooldown_until = self._symbol_loss_cooldown.get(symbol)
            if cooldown_until:
                elapsed = (datetime.now(LOCAL_TZ) - cooldown_until).total_seconds()
                if elapsed < 1800:
                    return
                else:
                    del self._symbol_loss_cooldown[symbol]

            sl_fail_until = self._sl_failure_cooldown.get(symbol)
            if sl_fail_until:
                elapsed = (datetime.now(LOCAL_TZ) - sl_fail_until).total_seconds()
                if elapsed < 7200:
                    return
                else:
                    del self._sl_failure_cooldown[symbol]

            # ── Gate 0/1: Macro context — kill switch + universe filter ──
            macro = self.macro_context.get()
            if macro["kill"]:
                self.log.info(f"[{symbol}] MACRO KILL: {macro['kill_reason']}")
                return
            if not self.macro_context.is_symbol_allowed(symbol, macro["universe"]):
                self.log.debug(f"[{symbol}] not in universe '{macro['universe']}'")
                return

            # Fetch data
            tf = self.config.get("ml", {}).get("timeframe") or self.config.get("scanner", {}).get("timeframe", "15m")
            if pre_multi:
                dfs = pre_multi
            else:
                train_tfs = self.config.get("training", {}).get("timeframes", ["1h", "4h", "1d"])
                multi_tfs = [(t, 500 if t == tf else (300 if t in ("1h","30m") else 200)) for t in train_tfs]
                dfs = self.feed.fetch_multi_timeframe(symbol, timeframes=multi_tfs)
            if not dfs or tf not in dfs:
                return

            df_tf = dfs[tf]
            df_1h = dfs.get("1h", df_tf)
            regime = (regime_ctx or {}).get("regime", "RANGING")
            trend_direction = (regime_ctx or {}).get("trend_direction", "NEUTRAL")
            trend_change = (regime_ctx or {}).get("trend_change")

            # ── Layer 1: Ensemble (SMC + Technical + MacroFlow) ─────────
            ensemble = self.ensemble.run(symbol, dfs, self.profile, market_ctx=regime_ctx)

            for s in ensemble.signals:
                self.log.debug(f"AGENT {symbol} | {s.agent}: net={s.net_score:+.3f} | {s.reasoning[:80]}")

            if ensemble.action == "HOLD":
                self.state.add_signal({
                    "symbol": symbol, "action": "HOLD", "confidence": ensemble.confidence,
                    "status": "hold", "reason": f"net={ensemble.net_score:+.3f}",
                    "strategy": f"ensemble:{ensemble.net_score:+.3f}", "timeframe": "AUTO",
                    "indicators": {"buy_score": ensemble.buy_score, "sell_score": ensemble.sell_score,
                                   "agents_agree": ensemble.agents_agreeing},
                    "timestamp": datetime.now(LOCAL_TZ).isoformat(),
                })
                self.log.info(f"SIGNAL {symbol} → HOLD | net={ensemble.net_score:+.3f} regime={regime}")
                return

            # ── Gate 4: Microstructure confirm/kill ─────────────────────
            funding_rate = 0.0
            try:
                if self.MODE == "futures" and hasattr(self.exchange, "fetch_funding_rate"):
                    fr = self.exchange.fetch_funding_rate(symbol)
                    funding_rate = float(fr.get("fundingRate", 0) or 0)
            except Exception:
                pass

            micro = self.microstructure.analyze(
                self.exchange, symbol, df_tf, ensemble.action, funding_rate
            )
            self.log.debug(f"MICRO {symbol} | {micro.reasoning}")
            if micro.kill:
                self.log.info(f"SIGNAL {symbol} → {ensemble.action} | MICRO KILL: {micro.reasoning}")
                self.state.add_signal({
                    "symbol": symbol, "action": ensemble.action, "confidence": ensemble.confidence,
                    "status": "rejected", "reason": f"microstructure kill: {micro.reasoning}",
                    "strategy": f"ensemble:{ensemble.net_score:+.3f}", "timeframe": "AUTO",
                    "timestamp": datetime.now(LOCAL_TZ).isoformat(),
                })
                self._record_shadow(symbol, "microstructure", micro.reasoning,
                                    ensemble, micro, regime, macro)
                return

            # ── Gate 5: DeepSeek Actor reasoning ────────────────────────
            # Pre-filter: skip the LLM entirely when ensemble confidence is
            # already below the approval floor — Actor would reject it anyway.
            if ensemble.confidence < self.profile.min_confidence:
                self.log.info(
                    f"ACTOR {symbol} | pre-filter reject "
                    f"conf={ensemble.confidence:.2f} < {self.profile.min_confidence:.2f}"
                )
                self.state.add_signal({
                    "symbol": symbol, "action": ensemble.action,
                    "confidence": ensemble.confidence,
                    "status": "rejected",
                    "reason": f"actor pre-filter: conf={ensemble.confidence:.2f} < {self.profile.min_confidence:.2f}",
                    "strategy": f"ensemble:{ensemble.net_score:+.3f}", "timeframe": "AUTO",
                    "timestamp": datetime.now(LOCAL_TZ).isoformat(),
                })
                self._record_shadow(
                    symbol, "actor_prefilter",
                    f"conf={ensemble.confidence:.2f} < {self.profile.min_confidence:.2f}",
                    ensemble, micro, regime, macro)
                return

            # Short-TTL verdict cache (Fix #5): an identical, quantized setup
            # signature within cache_ttl_seconds reuses the prior Actor decision
            # instead of re-running RAG + a serial DeepSeek call every 30s scan.
            actor_cfg = self.config.get("actor", {}) or {}
            adv_cfg   = self.config.get("adversary", {}) or {}
            cache_ttl = float(actor_cfg.get("cache_ttl_seconds", 180))
            # Coarse quantization so an unchanged setup keeps a stable signature
            # across 30-60s scans. Fine rounding (net_score/conf to 0.01,
            # ob_imbalance to 0.1) flipped almost every scan, dropping the cache
            # hit-rate to ~10%; these buckets target ~70-80% while still busting
            # on a genuine regime/score/flow shift.
            actor_sig = (
                ensemble.action, regime,
                round(ensemble.net_score * 20) / 20,     # 0.05 buckets
                round(ensemble.confidence * 20) / 20,    # 0.05 buckets
                round(micro.ob_imbalance * 2) / 2,       # 0.5 buckets
                micro.cvd_direction,
                bool(micro.cvd_divergence),
                trend_change,                            # turn flip invalidates cache
            )
            now_mono = time.monotonic()
            cached = self._actor_cache.get(symbol)
            if (cache_ttl > 0 and cached and cached["sig"] == actor_sig
                    and (now_mono - cached["ts"]) < cache_ttl):
                actor   = cached["decision"]
                skeptic = cached.get("skeptic")
                self.log.info(f"ACTOR {symbol} | cached approved={actor.approved} conf={actor.confidence:.2f}")
            else:
                # Vector RAG: retrieve similar past setups (hybrid semantic +
                # feature) and semantically-relevant Judge lessons.
                similar = self.trade_memory.find_similar(
                    symbol, ensemble.action, regime, ensemble.net_score, macro["btc_d"], n=5,
                    confidence=ensemble.confidence, usdt_d=macro["usdt_d"],
                    ob_imbalance=micro.ob_imbalance, cvd_direction=micro.cvd_direction,
                    cvd_divergence=micro.cvd_divergence,
                )
                side_word = "long" if ensemble.action == "BUY" else "short"
                query_text = trade_memory_mod.entry_text(symbol, side_word, {
                    "regime": regime, "ensemble_score": ensemble.net_score,
                    "confidence": ensemble.confidence,
                    "btc_d": macro["btc_d"], "usdt_d": macro["usdt_d"],
                    "ob_imbalance": micro.ob_imbalance,
                    "cvd_direction": micro.cvd_direction,
                    "cvd_divergence": micro.cvd_divergence,
                })
                lessons = self.trade_memory.find_similar_critiques(query_text, n=3)
                sem = "semantic" if vector_store.get_embedder().available else "feature-only"
                top_sim = f" top={similar[0]['symbol']}:{similar[0].get('_score',0):.2f}" if similar else ""
                self.log.info(
                    f"RAG {symbol} | {sem} | {len(similar)} similar trades, "
                    f"{len(lessons)} lessons{top_sim}"
                )
                extra_context = ""
                if lessons:
                    extra_context = "Relevant past lessons:\n" + "\n".join(
                        f"- [{l.get('outcome','?')}] {l.get('lesson','')}".strip()
                        for l in lessons if l.get("lesson")
                    )
                if trend_change:
                    extra_context += (
                        ("\n" if extra_context else "")
                        + f"TREND CHANGE: 1h momentum has turned {trend_change} against "
                          f"the standing 4h trend — early reversal window; do not "
                          f"penalise with-turn signals for opposing the old trend."
                    )
                # Fix #3: approval floor = profile.min_confidence (not a hidden
                # hardcoded 0.50). Fix #1: recency-decayed, smoothed win-rate.
                actor = llm_reasoning.actor_evaluate(
                    symbol=symbol, action=ensemble.action,
                    ensemble_score=ensemble.net_score,
                    ensemble_confidence=ensemble.confidence,
                    regime=regime, macro=macro, micro_signal=micro,
                    similar_trades=similar, extra_context=extra_context,
                    approve_threshold=self.profile.min_confidence,
                    winrate_half_life_h=float(actor_cfg.get("winrate_half_life_hours", 48.0)),
                    winrate_prior=float(actor_cfg.get("winrate_prior_strength", 1.0)),
                    trend_direction=trend_direction,
                )
                # Gate 5.5: skeptic argues against approved setups only — a
                # rejection is already the conservative outcome. Different
                # model family (Llama on Groq) so it doesn't share the Actor's
                # blind spots; it sees the thesis but never the confidence.
                skeptic = None
                if actor.approved and adv_cfg.get("enabled", True) and skeptic_agent.available():
                    skeptic = skeptic_agent.skeptic_evaluate(
                        symbol=symbol, action=ensemble.action,
                        thesis=actor.reasoning, regime=regime,
                        trend_direction=trend_direction, macro=macro,
                        micro_signal=micro, ensemble_score=ensemble.net_score,
                        trend_change=trend_change,
                        model=adv_cfg.get("model", skeptic_agent.DEFAULT_MODEL),
                    )
                self._actor_cache[symbol] = {"sig": actor_sig, "ts": now_mono,
                                             "decision": actor, "skeptic": skeptic}
                self.log.info(f"ACTOR {symbol} | approved={actor.approved} conf={actor.confidence:.2f} | {actor.reasoning}")
            if not actor.approved:
                self.state.add_signal({
                    "symbol": symbol, "action": ensemble.action, "confidence": actor.confidence,
                    "status": "rejected", "reason": f"actor: {actor.reasoning}",
                    "strategy": f"ensemble:{ensemble.net_score:+.3f}", "timeframe": "AUTO",
                    "timestamp": datetime.now(LOCAL_TZ).isoformat(),
                })
                self._record_shadow(symbol, "actor", actor.reasoning,
                                    ensemble, micro, regime, macro)
                return

            # ── Gate 5.5: Adversarial resolution (deterministic, no LLM) ──
            # effective = actor_conf − k × rebuttal_strength → veto/haircut/pass.
            # One-way authority: the skeptic can block or shrink, never enlarge.
            # mode "shadow" logs the verdict without enforcing it.
            skeptic_size_mult = 1.0
            if skeptic is not None:
                verdict, eff_conf, size_mult = skeptic_agent.combine(
                    actor.confidence, skeptic.rebuttal_strength,
                    self.profile.min_confidence,
                    k=float(adv_cfg.get("k", 0.4)),
                    haircut_band=float(adv_cfg.get("haircut_band", 0.10)),
                )
                self.log.info(
                    f"SKEPTIC {symbol} | strength={skeptic.rebuttal_strength:.2f} "
                    f"[{skeptic.objection}] → {verdict} eff={eff_conf:.2f} | {skeptic.statement}")
                if adv_cfg.get("mode", "enforce") == "enforce":
                    if verdict == "veto":
                        self.state.add_signal({
                            "symbol": symbol, "action": ensemble.action, "confidence": eff_conf,
                            "status": "rejected", "reason": f"skeptic: {skeptic.statement}",
                            "strategy": f"ensemble:{ensemble.net_score:+.3f}", "timeframe": "AUTO",
                            "timestamp": datetime.now(LOCAL_TZ).isoformat(),
                        })
                        self._record_shadow(symbol, "adversary",
                                            f"[{skeptic.objection}] {skeptic.statement}",
                                            ensemble, micro, regime, macro)
                        return
                    skeptic_size_mult = size_mult

            # Blend Actor confidence into the ensemble result for sizing.
            # Fix #3: ensemble-weighted (0.6/0.4) rather than a flat mean — the
            # ensemble is the quantitative score and drives sizing; the Actor
            # nudges it instead of systematically halving every position toward
            # the more skeptical of the two estimates.
            blended_conf = round(0.6 * ensemble.confidence + 0.4 * actor.confidence, 4)
            ensemble.confidence = blended_conf

            # HTF bias + price
            htf_bias = self.get_htf_bias(dfs)
            price = self.get_price(symbol)

            # ── Gate 6: Risk decision + sizing ──────────────────────────
            decision = self.risk_agent.evaluate(
                ensemble=ensemble, symbol=symbol, df_1h=df_1h,
                profile=self.profile, regime_ctx=regime_ctx,
                btc_return=btc_1h_return or 0.0,
                open_trades=open_trades, balance=balance,
                get_price_fn=self.get_price, get_atr_fn=self.get_atr,
                htf_bias=htf_bias, all_trades=self.state.get_all_trades(),
            )

            if not decision.approved:
                self.log.info(f"SIGNAL {symbol} → {ensemble.action} | REJECTED: {' | '.join(decision.reasons)}")
                self.state.add_signal({
                    "symbol": symbol, "action": ensemble.action, "confidence": ensemble.confidence,
                    "status": "rejected", "reason": " | ".join(decision.reasons),
                    "strategy": f"ensemble:{ensemble.net_score:+.3f}", "timeframe": "AUTO",
                    "indicators": {"buy_score": ensemble.buy_score, "sell_score": ensemble.sell_score,
                                   "agents_agree": ensemble.agents_agreeing, "profile": self.profile.name,
                                   "regime": regime},
                    "timestamp": datetime.now(LOCAL_TZ).isoformat(),
                })
                self._record_shadow(symbol, "risk", " | ".join(decision.reasons),
                                    ensemble, micro, regime, macro)
                return

            # Skeptic haircut: applied to the final size AFTER all risk gates so
            # it can only shrink what Kelly approved, never alter gate outcomes.
            if skeptic_size_mult < 1.0:
                decision.position_size *= skeptic_size_mult
                decision.est_usdt      *= skeptic_size_mult
                self.log.info(f"SKEPTIC {symbol} | position haircut ×{skeptic_size_mult:.1f} "
                              f"→ {decision.position_size:.6f} (${decision.est_usdt:.2f})")

            self.state.add_signal({
                "symbol": symbol, "action": ensemble.action, "confidence": decision.adjusted_conf,
                "status": "taken", "strategy": f"ensemble:{ensemble.net_score:+.3f}", "timeframe": "AUTO",
                "indicators": {"buy_score": ensemble.buy_score, "sell_score": ensemble.sell_score,
                               "agents_agree": ensemble.agents_agreeing, "profile": self.profile.name,
                               "regime": regime, "actor_conf": actor.confidence,
                               "ob_imbalance": micro.ob_imbalance, "cvd": micro.cvd_direction},
                "timestamp": datetime.now(LOCAL_TZ).isoformat(),
            })

            # ── Execution ───────────────────────────────────────────────
            if not price or price <= 0:
                return

            holding = self._find_position(symbol, open_trades)
            opp_side = {"BUY": "long", "SELL": "short"}.get(ensemble.action)
            if holding and holding["side"] != opp_side:
                order = self._place_close(symbol, holding["amount"], holding["side"])
                if order:
                    close_price = self.get_price(symbol) or price
                    pnl = self._calc_pnl(holding, close_price)
                    self.state.close_trade(holding["id"], close_price, pnl)
                    self.agents.record_trade_result(symbol, pnl)
                    self.risk.record_trade_result(pnl, balance)
                    self.risk.cleanup_trade(holding["id"])
                    self._cancel_exchange_stop_loss(symbol, holding.get("sl_order_id", ""))
                    self._on_trade_closed(holding, close_price, pnl, "close_opposite")
                    self.log.info(f"CLOSE OPPOSITE {symbol} | PnL={pnl:+.4f}")
                    pnl_icon = "🟢" if pnl >= 0 else "🔴"
                    self.notifier.send_alert(
                        f"🔄 <b>CLOSE OPPOSITE</b> · <b>{symbol}</b>\n"
                        f"  {pnl_icon} PnL: ${pnl:+.2f} USDT"
                    )
                open_trades = self.state.get_open_trades()

            holding = self._find_position(symbol, open_trades)
            if holding and holding["side"] == opp_side:
                self.log.info(f"SKIP {symbol}: already open {holding['side']}")
                return
            if ensemble.action == "SELL" and self.MODE == "spot" and not holding:
                return

            trade = self.execution.execute_entry(
                decision=decision, symbol=symbol, action=ensemble.action,
                price=price, get_atr_fn=self.get_atr,
                place_buy_fn=self._place_buy, place_sell_fn=self._place_sell,
                strat=f"ensemble:{ensemble.net_score:+.3f}",
            )
            if trade == "SL_FAILED":
                self._sl_failure_cooldown[symbol] = datetime.now(LOCAL_TZ)
                self.log.warning(f"[{symbol}] SL failure cooldown set — skipping for 2h")
            elif trade and hasattr(trade, "id"):
                # Store entry context for the Judge when this trade later closes.
                # Persisted to disk immediately so restarts don't lose the context.
                self._entry_contexts[trade.id] = {
                    "regime": regime,
                    "trend_direction": trend_direction,
                    "btc_d": macro["btc_d"], "usdt_d": macro["usdt_d"],
                    "btc_d_roc": macro["btc_d_roc"], "usdt_d_roc": macro["usdt_d_roc"],
                    "ensemble_score": ensemble.net_score,
                    "confidence": decision.adjusted_conf,
                    "ob_imbalance": micro.ob_imbalance,
                    "cvd_direction": micro.cvd_direction,
                    "cvd_divergence": micro.cvd_divergence,
                    "actor_reasoning": actor.reasoning,
                    "actor_approved": actor.approved,
                    "trend_change": trend_change,
                    "skeptic_strength": (skeptic.rebuttal_strength if skeptic else None),
                    "skeptic_objection": (skeptic.objection if skeptic else None),
                    "skeptic_size_mult": skeptic_size_mult,
                    "entry_price": price,
                }
                self._save_entry_contexts()

        except Exception as e:
            self.log.error(f"Error analyzing {symbol}: {e}", exc_info=True)

    def _passes_trade_filters(self, df_1h, symbol: str) -> bool:
        """
        Step 10: Pre-entry quality filters.
        Scoring: need >= 1/2 conditions (vol spike OR ATR expansion).
        Both pass = high conviction entry.
        """
        try:
            close = df_1h["close"]
            high  = df_1h["high"]
            low   = df_1h["low"]
            vol   = df_1h["volume"]

            vol_ma   = vol.rolling(20).mean()
            vol_ratio = float(vol.iloc[-1]) / (float(vol_ma.iloc[-1]) + 1e-9)
            vol_ok   = vol_ratio > 1.5

            import ta as _ta
            atr      = _ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
            atr_ma   = atr.rolling(20).mean()
            atr_ratio = float(atr.iloc[-1]) / (float(atr_ma.iloc[-1]) + 1e-9)
            atr_ok   = atr_ratio > 1.0

            score = int(vol_ok) + int(atr_ok)
            if score == 0:
                self.log.info(
                    f"FILTER SKIP {symbol}: vol={vol_ratio:.2f}x "
                    f"atr={atr_ratio:.2f}x (need >= 1)"
                )
                return False
            elif score == 2:
                self.log.info(f"FILTER PASS {symbol}: vol={vol_ratio:.2f}x atr={atr_ratio:.2f}x (both)")
            else:
                self.log.info(
                    f"FILTER PASS {symbol}: score=1/2 "
                    f"vol={'OK' if vol_ok else 'skip'}({vol_ratio:.2f}x) "
                    f"atr={'OK' if atr_ok else 'skip'}({atr_ratio:.2f}x)"
                )
            return True
        except Exception as e:
            self.log.warning(f"Trade filter error for {symbol}: {e}")
            return True

    def _on_trade_closed(self, trade: dict, close_price: float, pnl: float, reason: str = ""):
        """Record closed trade to memory and trigger the Judge (R1).
        Runs Meta-Judge every 20 trades."""
        try:
            trade_id = trade.get("id", "")
            entry_ctx = self._entry_contexts.pop(trade_id, {})
            self._save_entry_contexts()  # persist the removal

            # Compute R-multiple (pnl relative to risked amount)
            entry = float(trade.get("price", 0))
            amount = float(trade.get("amount", 0))
            risk_amt = entry * amount * 0.02  # approx 2% stop assumption
            r_mult = pnl / (risk_amt + 1e-9) if risk_amt > 0 else 0.0

            trade_record = dict(trade)
            trade_record["close_price"] = close_price
            trade_record["pnl"] = pnl
            trade_record["close_reason"] = reason

            # Store in memory
            self.trade_memory.record_trade(trade_record, entry_ctx, r_multiple=r_mult)

            # Judge review (R1) — runs async to not block the cycle
            def _judge():
                try:
                    entry_ctx["r_multiple"] = r_mult
                    critique = llm_reasoning.judge_review(trade_record, entry_ctx)
                    if critique:
                        self.trade_memory.add_judge_critique(trade_id, critique)
                        self.log.info(f"JUDGE {trade.get('symbol')} | {critique.get('decision_quality')} | {critique.get('lesson','')[:80]}")

                    # Meta-Judge every 20 new critiques since the last run.
                    # Uses the DB count so bot restarts don't break the cadence.
                    new_since_last = self.trade_memory.critiques_since_last_meta()
                    if new_since_last >= 20:
                        critiques = self.trade_memory.get_recent_critiques(20)
                        shadow_stats = None
                        if self.shadow_tracker:
                            try:
                                shadow_stats = self.shadow_tracker.gate_stats()
                            except Exception:
                                pass
                        meta = llm_reasoning.meta_judge_synthesize(
                            critiques, shadow_stats=shadow_stats)
                        if meta:
                            self.trade_memory.save_meta_rules(meta)
                            self.log.info(f"META-JUDGE: {meta.get('summary','')[:120]}")
                            self.notifier.send(
                                f"🧠 <b>Meta-Judge update</b>\n{meta.get('summary','')[:200]}"
                            )
                except Exception as e:
                    self.log.warning(f"Judge thread error: {e}")

            threading.Thread(target=_judge, daemon=True).start()
        except Exception as e:
            self.log.warning(f"_on_trade_closed error: {e}")

    def _find_position(self, symbol, open_trades):
        """Find existing position for a symbol."""
        return next(
            (t for t in open_trades if t["symbol"] == symbol),
            None
        )

    def _get_leverage(self) -> int:
        """Override in futures bot."""
        return 1

    def _load_entry_contexts(self) -> dict:
        """Load persisted entry contexts so they survive restarts."""
        try:
            if self._entry_contexts_file.exists():
                with open(self._entry_contexts_file) as f:
                    data = json.load(f)
                self.log.info(f"Loaded {len(data)} entry contexts from disk")
                return data
        except Exception as e:
            self.log.warning(f"Entry contexts load failed: {e}")
        return {}

    def _save_entry_contexts(self):
        """Atomically persist entry contexts to disk."""
        try:
            def _coerce(obj):
                # numpy bools/ints/floats are not JSON-native
                if hasattr(obj, "item"):
                    return obj.item()
                raise TypeError(f"Not serializable: {type(obj)}")
            tmp = self._entry_contexts_file.with_suffix(".tmp.json")
            with open(tmp, "w") as f:
                json.dump(self._entry_contexts, f, default=_coerce)
            tmp.replace(self._entry_contexts_file)
        except Exception as e:
            self.log.warning(f"Entry contexts save failed: {e}")

    def _place_exchange_stop_loss(self, symbol, side, amount, entry_price, atr):
        """Override in futures bot to place exchange-side SL."""
        return ""

    def _cancel_exchange_stop_loss(self, symbol, sl_order_id):
        """Override in futures bot to cancel exchange-side SL."""
        pass

    def sync_with_exchange(self):
        """Sync state with exchange. Closes local trades not found on exchange per mode."""
        if self.MODE == "futures":
            self._sync_futures()
        elif self.MODE == "spot":
            self._sync_spot()

    def _import_exchange_positions(self, positions: list, d: dict) -> None:
        """Import any exchange position not currently tracked as open in state.
        No symbol blocklists — if the exchange has it, the bot should know about it.
        """
        import time as _time
        our_syms = {t["symbol"] for t in d.get("trades", []) if t.get("status") == "open"}
        self.log.info(f"Sync import check: {len(positions)} positions, {len(our_syms)} tracked: {our_syms}")
        for pos in positions:
            try:
                sym = pos["symbol"]
                amt = float(pos.get("amount", 0))
                self.log.info(f"Sync import: checking {sym} amt={amt}")
                if sym in our_syms or amt == 0:
                    self.log.info(f"Sync: skipping {sym} (tracked={sym in our_syms} amt={amt})")
                    continue
                notional = amt * float(pos.get("entry_price", 0))
                if notional < 5.0:
                    self.log.info(f"Sync: skipping dust {sym} notional=${notional:.4f}")
                    continue
                trade = {
                    "id":              f"sync_pos_{sym.replace('/','_')}_{int(_time.time())}",
                    "symbol":          sym,
                    "side":            pos["side"],
                    "amount":          pos["amount"],
                    "price":           pos["entry_price"],
                    "mark_price":      pos.get("mark_price", 0),
                    "live_pnl":        round(pos["pnl"], 6),
                    "timestamp":       datetime.now(LOCAL_TZ).isoformat(),
                    "strategy":        "synced_from_position",
                    "timeframe":       f"AUTO-{self.MODE}",
                    "status":          "open",
                    "mode":            self.MODE,
                    "leverage":        pos.get("leverage", 5),
                    "sl_order_id":     "",
                    "pnl":             0.0,
                    "close_price":     0.0,
                    "close_timestamp": ""
                }
                d["trades"].append(trade)
                d["stats"]["total_trades"] += 1
                self.log.info(f"Sync: imported {sym} {pos['side']} from exchange (notional=${notional:.2f})")
            except Exception as e:
                self.log.warning(f"Sync import error for position: {e}")

    def _cleanup_ghost_trades(self, exchange_syms: set, d: dict) -> None:
        """Cancel trades that are open in state but missing from the exchange for >60s.
        Skips trades entered <60s ago (race condition grace).
        Trades missing >60s are immediately closed as ghost trades.
        """
        now = datetime.now(LOCAL_TZ)
        for t in d.get("trades", []):
            if t["status"] != "open":
                continue
            sym = t["symbol"]
            if sym in exchange_syms:
                continue
            ts_str = t.get("timestamp", "")
            age_s = None
            try:
                if ts_str:
                    opened = datetime.fromisoformat(ts_str)
                    if opened.tzinfo is None:
                        opened = opened.replace(tzinfo=LOCAL_TZ)
                    age_s = (now - opened).total_seconds()
            except (ValueError, TypeError):
                pass
            if age_s is not None and age_s < 60:
                self.log.debug(f"Sync: {sym} entered <60s ago, skipping ghost check")
                continue
            # >60s missing from exchange — cancel the ghost trade immediately
            close_price = float(t.get("price", 0))
            raw_rpnl = None
            try:
                # Look for the closing fill: the side opposite to the position
                # (long was closed by a sell fill; short was closed by a buy fill)
                closing_side = "sell" if t["side"] == "long" else "buy"
                fills = self.exchange.fetch_my_trades(sym, limit=20)
                closing_fills = [f for f in fills if f.get("side", "").lower() == closing_side]
                if closing_fills:
                    best = max(closing_fills, key=lambda x: x["time"])
                    close_price = float(best.get("price", close_price))
                    raw_rpnl = best.get("realizedPnl")
                elif fills:
                    # No closing fill found — fall back to ticker
                    close_price = self.exchange.fetch_ticker(sym)["last"]
            except Exception:
                try:
                    close_price = self.exchange.fetch_ticker(sym)["last"]
                except Exception:
                    pass
            pnl = self._calc_pnl(t, close_price)
            t["status"]          = "closed"
            t["close_price"]     = close_price
            t["pnl"]             = round(pnl, 8)
            t["close_timestamp"] = now.isoformat()
            d["stats"]["total_pnl"] += round(pnl, 6)
            if pnl > 0:   d["stats"]["wins"]   += 1
            elif pnl < 0: d["stats"]["losses"] += 1
            self.log.info(f"Sync: cancelled ghost trade {sym} (missing >60s) pnl={pnl:+.4f}")
            # NOTE: risk.record_trade_result intentionally excluded — ghost PnL is
            # estimated from mark price, not a real fill, and must not trip the circuit breaker.
            for fn in [
                lambda: self._cancel_exchange_stop_loss(sym, t.get("sl_order_id", "")),
                lambda: self.rl_agent.record_external_close(t["id"], pnl),
                lambda: self.ai.record_trade_result(sym, pnl),
                lambda: self.agents.record_trade_result(sym, pnl),
            ]:
                try: fn()
                except Exception: pass

    def _sync_futures(self):
        """Futures sync: update live PnL, import untracked positions, cancel 60s+ ghosts."""
        try:
            self.log.info("Sync running — mode=futures")
            position_fetch_ok = True
            try:
                positions = self.exchange.get_position()
            except Exception as pe:
                self.log.warning(f"Sync: get_position failed ({pe}) — skipping ghost cleanup this cycle")
                positions = []
                position_fetch_ok = False
            if not positions:
                positions = []

            exchange_syms = {p["symbol"] for p in positions if float(p.get("amount", 0)) != 0}
            self.log.info(f"Sync: got {len(exchange_syms)} active positions from exchange")

            bal  = self.exchange.fetch_balance()
            usdt = float(bal["total"].get("USDT", 0))

            d = self.state.state
            now_utc = datetime.now(LOCAL_TZ)
            with self.state._lock:
                # Update live PnL and duration for tracked positions
                for t in d.get("trades", []):
                    if t["status"] != "open":
                        continue
                    try:
                        ts_str = t.get("timestamp", "")
                        if ts_str:
                            opened = datetime.fromisoformat(ts_str)
                            if opened.tzinfo is None:
                                opened = opened.replace(tzinfo=LOCAL_TZ)
                            t["duration_hours"] = round((now_utc - opened).total_seconds() / 3600, 2)
                    except (ValueError, TypeError):
                        pass
                    sym = t["symbol"]
                    if sym in exchange_syms:
                        for pos in positions:
                            if pos["symbol"] == sym:
                                t["amount"]     = pos["amount"]
                                t["price"]      = pos["entry_price"]
                                t["leverage"]   = pos.get("leverage", 5)
                                t["live_pnl"]   = round(pos["pnl"], 6)
                                t["mark_price"] = pos.get("mark_price", 0)
                                break

                # Import any exchange position the bot doesn't know about
                self._import_exchange_positions(positions, d)

                # Cancel trades open in state but gone from exchange for >60s
                if position_fetch_ok:
                    self._cleanup_ghost_trades(exchange_syms, d)

                # Update stats inside the lock to avoid race condition with periodic flush
                d["stats"]["balance"]      = round(usdt, 2)
                d["stats"]["last_sync"]     = datetime.now(LOCAL_TZ).isoformat()
                d["stats"]["total_live_pnl"] = round(
                    sum(t.get("live_pnl", 0.0) for t in d.get("trades", []) if t.get("status") == "open"), 4
                )
            self.log.info(f"Sync saving balance=${usdt:.2f}")
        except Exception as e:
            self.log.warning(f"Sync error: {e}")
        finally:
            self.state.save(immediate=True)

    def _sync_spot(self):
        """Balance-based sync for spot mode."""
        try:
            self.log.info("Sync running — mode=spot")
            bal = self.exchange.fetch_balance()
            totals = bal.get("total", {})

            d = self.state.state
            our_open = {t["symbol"]: t for t in d["trades"] if t["status"] == "open"}
            our_syms = set(our_open.keys())
            all_syms = {t["symbol"] for t in d["trades"]}

            with self.state._lock:
                # RULE 1: Trade in bot, NOT on exchange → close it
                for t in d["trades"]:
                    if t["status"] != "open":
                        continue
                    ts_str = t.get("timestamp", "")
                    recent_entry = False
                    try:
                        if ts_str:
                            opened = datetime.fromisoformat(ts_str)
                            if opened.tzinfo is None:
                                opened = opened.replace(tzinfo=LOCAL_TZ)
                            if (datetime.now(LOCAL_TZ) - opened).total_seconds() < 120:
                                recent_entry = True
                    except (ValueError, TypeError):
                        pass
                    if recent_entry:
                        self.log.debug(f"Sync: skipping orphan check for {t['symbol']} (entered <120s ago)")
                        continue
                    asset = t["symbol"].split("/")[0]
                    held = float(totals.get(asset, 0))
                    min_held = float(t["amount"]) * 0.1
                    trade_value = float(t["amount"]) * float(t["price"])
                    if held < min_held or trade_value < 1.0:
                        close_price = float(t["price"])
                        try:
                            closing_side = "sell"   # spot is always long; closed by a sell
                            fills = self.exchange.fetch_my_trades(t["symbol"], limit=10)
                            closing_fills = [f for f in fills if f.get("side", "").lower() == closing_side]
                            if closing_fills:
                                best = max(closing_fills, key=lambda x: x["time"])
                                close_price = float(best.get("price", close_price))
                            elif fills:
                                close_price = self.exchange.fetch_ticker(t["symbol"])["last"]
                        except Exception:
                            try:
                                close_price = self.exchange.fetch_ticker(t["symbol"])["last"]
                            except Exception:
                                pass
                        pnl = self._calc_pnl(t, close_price)
                        t["status"]          = "closed"
                        t["close_price"]     = close_price
                        t["pnl"]             = round(pnl, 8)
                        t["close_timestamp"] = datetime.now(LOCAL_TZ).isoformat()
                        d["stats"]["total_pnl"] += round(pnl, 6)
                        if pnl > 0:
                            d["stats"]["wins"]   += 1
                        elif pnl < 0:
                            d["stats"]["losses"] += 1
                        self.log.info(f"Sync: closed {t['symbol']} (asset {asset} balance={held:.6f}) pnl={pnl:+.4f}")
                        try:
                            self.risk.record_trade_result(pnl, 5000)
                        except Exception:
                            pass
                        try:
                            self.ai.record_trade_result(t.get("symbol", "unknown"), pnl)
                        except Exception:
                            pass

                # RULE 2: Import watchlist exchange balances with >$50 value
                try:
                    watchlist_coins = set(self.scanner.get_coins(self.exchange, invalid_symbols=self.feed.invalid_symbols))
                except Exception:
                    watchlist_coins = set()
                for asset, free_val in bal.get("free", {}).items():
                    free = float(free_val)
                    if free <= 0:
                        continue
                    sym = f"{asset}/USDT"
                    if asset in ("USDT", "USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDD"):
                        continue
                    if sym in all_syms:
                        continue
                    if sym not in watchlist_coins:
                        continue
                    try:
                        ticker = self.exchange.fetch_ticker(sym)
                        price = float(ticker.get("last", 0))
                    except Exception:
                        continue
                    if price <= 0:
                        continue
                    if free * price < 50.0:
                        self.log.debug(f"Sync: skipping {sym} — dust (${free*price:.2f})")
                        continue
                    import time as _time
                    trade = {
                        "id":              f"sync_{sym.replace('/','_')}_{int(_time.time())}",
                        "symbol":          sym,
                        "side":            "buy",
                        "amount":          round(free, 8),
                        "price":           price,
                        "mark_price":      price,
                        "live_pnl":        0.0,
                        "timestamp":       datetime.now(LOCAL_TZ).isoformat(),
                        "strategy":        "synced_from_exchange",
                        "timeframe":       f"AUTO-{self.MODE}",
                        "status":          "open",
                        "mode":            self.MODE,
                        "leverage":        1,
                        "sl_order_id":     "",
                        "pnl":             0.0,
                        "close_price":     0.0,
                        "close_timestamp": ""
                    }
                    d["trades"].append(trade)
                    d["stats"]["total_trades"] += 1
                    self.log.info(f"Sync: imported {sym} buy {free:.6f} (${free*price:.2f})")

            now_utc = datetime.now(LOCAL_TZ)
            for t in d["trades"]:
                if t.get("status") != "open":
                    continue
                try:
                    ts_str = t.get("timestamp", "")
                    if ts_str:
                        opened = datetime.fromisoformat(ts_str)
                        if opened.tzinfo is None:
                            opened = opened.replace(tzinfo=LOCAL_TZ)
                        t["duration_hours"] = round((now_utc - opened).total_seconds() / 3600, 2)
                except (ValueError, TypeError):
                    pass

            usdt = float(totals.get("USDT", 0))
            self.log.info(f"Sync saving balance=${usdt:.2f}")
            self.state.state["stats"]["balance"]        = round(usdt, 2)
            self.state.state["stats"]["last_sync"]      = datetime.now(LOCAL_TZ).isoformat()
            self.state.state["stats"]["total_live_pnl"] = 0.0
        except Exception as e:
            import traceback
            self.log.warning(f"Sync error: {e}")
            self.log.warning(traceback.format_exc())
        finally:
            self.state.save(immediate=True)

    def _get_watchlist_hash(self, coins):
        return hash(tuple(sorted(coins)))

    def _retrain_if_watchlist_changed(self, new_coins):
        """Log watchlist changes but do NOT retrain — models learn general patterns
        from a fixed anchor-coin dataset (config training.symbols), not the live watchlist."""
        new_hash = self._get_watchlist_hash(new_coins)
        if hasattr(self, '_last_watchlist_hash') and self._last_watchlist_hash == new_hash:
            return
        if hasattr(self, '_last_watchlist_hash'):
            added   = set(new_coins) - set(getattr(self, '_last_watchlist', []))
            removed = set(getattr(self, '_last_watchlist', [])) - set(new_coins)
            change_count = len(added) + len(removed)
            if change_count >= 5:
                self.log.info(f"Watchlist changed by {change_count} coins Added:{added} Removed:{removed}")
                try:
                    self.notifier.send(
                        f"ℹ️ <b>Watchlist changed [{self.MODE.upper()}]</b> — {change_count} coins\n"
                        f"Added: {added or 'none'}\n"
                        f"Removed: {removed or 'none'}"
                    )
                except Exception:
                    pass
        self._last_watchlist_hash = new_hash
        self._last_watchlist      = list(new_coins)

    def is_trading_paused(self) -> bool:
        """Check if trading has been paused via dashboard."""
        p = DATA / "trading_paused.json"
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f).get("paused", False)
            except Exception:
                pass
        return False

    def _close_all_positions(self, open_trades, balance):
        """Close all open positions — triggered by dashboard Close All button."""
        if not open_trades:
            self.log.info("Close All: no open trades to close")
            return
        self.log.info(f"Close All: closing {len(open_trades)} positions")
        self.notifier.send_alert(f"🛑 <b>CLOSE ALL</b> — closing {len(open_trades)} positions…")
        for trade in list(open_trades):
            sym = trade["symbol"]
            try:
                price = self.get_price(sym)
                if not price or price <= 0:
                    price = float(trade.get("price", 0))
                order = self._place_close(sym, trade["amount"], trade["side"])
                if order:
                    pnl = self._calc_pnl(trade, price)
                    self.state.close_trade(trade["id"], price, pnl)
                    self.risk.cleanup_trade(trade["id"])
                    self._cancel_exchange_stop_loss(sym, trade.get("sl_order_id", ""))
                    try:
                        self.ai.record_trade_result(sym, pnl)
                    except Exception:
                        pass
                    try:
                        self.agents.record_trade_result(sym, pnl)
                    except Exception:
                        pass
                    try:
                        self.risk.record_trade_result(pnl, balance)
                    except Exception:
                        pass
                    try:
                        self.rl_agent.record_external_close(trade["id"], pnl)
                    except Exception:
                        pass
                    self.log.info(f"Close All: {sym} closed pnl={pnl:+.4f}")
                else:
                    self.log.warning(f"Close All: {sym} order failed — marking closed locally")
                    pnl = self._calc_pnl(trade, price)
                    self.state.close_trade(trade["id"], price, pnl)
                    self.risk.cleanup_trade(trade["id"])
                    self._cancel_exchange_stop_loss(sym, trade.get("sl_order_id", ""))
                    try:
                        self.ai.record_trade_result(sym, pnl)
                    except Exception:
                        pass
                    try:
                        self.agents.record_trade_result(sym, pnl)
                    except Exception:
                        pass
                    try:
                        self.risk.record_trade_result(pnl, balance)
                    except Exception:
                        pass
                    try:
                        self.rl_agent.record_external_close(trade["id"], pnl)
                    except Exception:
                        pass
            except Exception as e:
                self.log.error(f"Close All: error closing {sym}: {e}")
        self.state.save(immediate=True)
        self.notifier.send_alert(f"✅ <b>CLOSE ALL done</b> — {len(open_trades)} positions closed")

    def run_once(self):
        """One scan cycle — identical for both modes."""
        regime_ctx  = None   # initialise early; assigned after symbol scan
        self._maybe_swap_profile()
        self.sync_with_exchange()
        self.check_exits()
        self._resolve_shadows()

        balance     = self.get_usdt_balance()
        open_trades = self.state.get_open_trades()

        # Check for close-all request from dashboard
        close_flag = DATA / "close_all_positions.json"
        if close_flag.exists():
            self._close_all_positions(open_trades, balance)
            try:
                close_flag.unlink()
            except OSError:
                pass
            open_trades = self.state.get_open_trades()
            balance     = self.get_usdt_balance()
            max_reached = len(open_trades) >= self.max_open

        # Check if trading paused via dashboard
        trading_paused = self.is_trading_paused()
        if trading_paused:
            self.log.info("Trading PAUSED via dashboard — monitoring only")

        can_trade, reason = self.risk.breaker.can_trade(balance)
        if not can_trade:
            self.log.warning(f"Circuit breaker: {reason}")

        max_reached = len(open_trades) >= self.max_open
        _vol_ratio  = (regime_ctx or {}).get("vol_ratio", 1.0)
        symbols     = self.scanner.get_coins(self.exchange, invalid_symbols=self.feed.invalid_symbols,
                                             current_atr_ratio=_vol_ratio)
        self._post_scan(symbols)
        self.feed.subscribe_many(symbols)   # ensure WS + REST monitor tracking
        self.log.info(f"[{self.MODE.upper()}] Watching: {symbols}")

        regime_ctx = self.risk.detect_market_regime(self.feed, symbols)

        # BTC 1h return (for risk agent) + publish live strategy for dashboard
        btc_1h_return = 0.0
        macro = self.macro_context.get()
        try:
            btc_df = self.feed.fetch_ohlcv("BTC/USDT", "1h", limit=50)
            if btc_df is not None and len(btc_df) >= 2:
                btc_1h_return = float(btc_df["close"].pct_change().iloc[-1])
        except Exception:
            pass

        # Regime now comes purely from MarketRegimeGate (rules) + macro context.
        # No HMM. hmm_regime field kept for dashboard compatibility = gate regime.
        gate_regime = (regime_ctx or {}).get("regime", "UNKNOWN")
        if regime_ctx:
            regime_ctx = dict(regime_ctx)
            regime_ctx["hmm_regime"] = gate_regime   # alias for downstream code
            regime_ctx["macro_universe"]  = macro["universe"]
            regime_ctx["macro_sentiment"] = macro["sentiment"]
        # Persist for next cycle's check_exits (two-stage reversal exit).
        self._regime_ctx = regime_ctx
        try:
            self.state.state["live_strategy"] = {
                "eff_min_conf":    regime_ctx.get("min_conf", self.min_conf) if regime_ctx else self.min_conf,
                "eff_size_mult":   regime_ctx.get("size_mult", 1.0) if regime_ctx else 1.0,
                "market_regime":   gate_regime,
                "hmm_regime":      gate_regime,
                "profile":         self.profile.name,
                "base_min_conf":   self.min_conf,
                "htf_filter_mode": self.htf_filter_mode,
                "btc_d":           macro["btc_d"],
                "usdt_d":          macro["usdt_d"],
                "macro_universe":  macro["universe"],
                "macro_sentiment": macro["sentiment"],
                "breadth":         regime_ctx.get("breadth") if regime_ctx else None,
                "bear_breadth":    regime_ctx.get("bear_breadth") if regime_ctx else None,
                "adx":             regime_ctx.get("adx") if regime_ctx else None,
                "deepseek_cost_today": usage_tracker.today_summary().get("cost_usd", 0.0),
                "timeframe":       self.config.get("ml", {}).get("timeframe", "15m"),
                "updated_at":      datetime.now(LOCAL_TZ).isoformat(),
            }
            self.state.save()
            self.log.info(f"Regime: {gate_regime} | BTC.D={macro['btc_d']:.1f}% universe={macro['universe']}")
        except Exception as e:
            self.log.warning(f"live_strategy publish failed: {e}")

        # RL execution layer: manages open trades, runs after ATR stops + HMM context ready
        self._rl_manage_trades(regime_ctx)
        open_trades = self.state.get_open_trades()
        balance     = self.get_usdt_balance()
        max_reached = len(open_trades) >= self.max_open

        # Pre-fetch 1h OHLCV for all symbols once — shared between GNN and analysis.
        # NOTE: limit=300 not 168 so the cache holds enough bars for the
        # two-tier trend filter's slow-tier EMA(200) + lookback(20) = 221 required.
        # GNN itself only needs the trailing 168 bars and slices internally.
        try:
            ohlcv_1h = {}
            def _fetch_1h(s):
                df = self.feed.fetch_ohlcv(s, "1h", limit=300)
                return s, df
            with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
                for s, df in executor.map(_fetch_1h, symbols):
                    if df is not None and len(df) >= 24:
                        ohlcv_1h[s] = df

            # Correlation handled by RiskManager.CorrelationFilter (group caps).
            # GNN graph removed in v5.
        except Exception as e:
            self.log.warning(f"1h prefetch failed: {e}")

        # Process symbols — pre-fetch OHLCV data in parallel, then analyze sequentially
        tf = (self.config.get("ml", {}).get("timeframe") or self.config.get("scanner", {}).get("timeframe", "15m"))
        train_tfs = self.config.get("training", {}).get("timeframes", ["15m", "1h", "4h", "1d"])

        # Pre-fetch multi-TF OHLCV for ALL symbols in parallel
        multi_cache = {}
        def _prefetch_multi(symbol):
            try:
                multi_tfs = [(t, 500 if t == tf else (300 if t in ("1h","30m") else 200)) for t in train_tfs]
                dfs = self.feed.fetch_multi_timeframe(symbol, timeframes=multi_tfs)
                return symbol, dfs
            except Exception as e:
                self.log.warning(f"Pre-fetch multi-TF failed for {symbol}: {e}")
                return symbol, None
        with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as ex:
            for s, dfs in ex.map(_prefetch_multi, symbols):
                if dfs:
                    multi_cache[s] = dfs

        for symbol in symbols:
            if not can_trade or max_reached or trading_paused:
                # Trading blocked (paused / max positions / circuit breaker):
                # skip analysis this cycle. Monitoring only.
                continue

            self.analyze_symbol(symbol, balance, open_trades, regime_ctx, btc_1h_return,
                               pre_multi=multi_cache.get(symbol))
            open_trades = self.state.get_open_trades()
            max_reached = len(open_trades) >= self.max_open


    def run(self):
        self.log.info("=" * 50)
        self.log.info(f"CRYPTOBOT v4 STARTED — MODE: {self.MODE.upper()}")
        self.log.info("=" * 50)

        alert_sent = False

        while True:
            try:
                # Watchlist change notification (no retrain — there are no models)
                current_coins = self.scanner.get_coins(self.exchange, invalid_symbols=self.feed.invalid_symbols)
                self._retrain_if_watchlist_changed(current_coins)

                # Heartbeat: write liveness flag for dashboard detection
                self._write_heartbeat()

                # Check for command-file from dashboard (fallback when Docker socket unavailable)
                if self._check_bot_control():
                    self.log.info("Bot control stop signal received, shutting down...")
                    self.notifier.send_alert(
                        f"CryptoBot v5 Stopped (dashboard command)\n"
                        f"Mode: {self.MODE.upper()}"
                    )
                    break

                self.run_once()
                self._write_heartbeat()  # also write after cycle — halves gap to dashboard
                # Send startup alert only after first successful cycle — prevents spam during deploys
                if not alert_sent:
                    alert_sent = True
                    self.notifier.send_alert(
                        f"CryptoBot v5 Started\n"
                        f"Mode: {self.MODE.upper()}"
                    )
                self.notifier.exchange = self.exchange
                self.notifier.send_report(self.exchange)

            except Exception as e:
                self.log.error(f"Cycle error: {e}", exc_info=True)

            self.log.info(f"Sleeping {self.scan_interval}s...")
            time.sleep(self.scan_interval)

    def _write_heartbeat(self):
        """Write heartbeat file for dashboard liveness detection."""
        hb = DATA_DIR / f"bot_heartbeat_{self.MODE}.json"
        try:
            with open(hb, "w") as f:
                json.dump({"timestamp": datetime.now(LOCAL_TZ).isoformat()}, f)
        except OSError:
            pass

    def _check_bot_control(self) -> bool:
        """Check for control commands from dashboard. Returns True if should stop."""
        p = DATA_DIR / "bot_control.json"
        if not p.exists():
            return False
        try:
            with open(p) as f:
                cmd = json.load(f)
            target = cmd.get("mode", "all")
            if target not in (self.MODE, "all"):
                return False
            action = cmd.get("command", "")
            if action == "stop":
                p.unlink(missing_ok=True)
                return True
            p.unlink(missing_ok=True)
            return False
        except (json.JSONDecodeError, OSError):
            return False
