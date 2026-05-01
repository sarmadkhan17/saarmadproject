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
import logging
import logging.handlers
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
import yaml
from dataclasses import dataclass, asdict
import sys

sys.path.insert(0, str(Path(__file__).parent))

from env_config   import get_exchange_config, DATA_DIR, LOGS_DIR, BOT_ROOT
from data_feed    import DataFeed
from data_feed_ws import BinanceWSPriceFeed
from ai_strategy  import AIStrategyEngine
from regime_model import HMMRegimeModel
from rl_agent    import RLTradeManager
from gnn_model   import GNNCorrelationFilter
from agents       import AgentCoordinator
from risk_manager import RiskManager
from self_learner import SelfLearner
from notifier     import TelegramNotifier
from coin_scanner import CoinScanner

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
        handlers = [handler, logging.StreamHandler()],
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


class StateManager:
    """
    Manages trade state.
    Spot and Futures use separate state files so they don't interfere.
    """

    def __init__(self, filename: str = "state.json"):
        self.path = DATA / filename
        DATA.mkdir(exist_ok=True)
        self.state = self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {
            "trades": [], "signals": [],
            "stats": {
                "total_trades": 0, "wins": 0, "losses": 0,
                "total_pnl": 0.0,
                "start_time": datetime.utcnow().isoformat(),
            },
        }

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
        try:
            shutil.copy2(str(self.path),
                         str(self.path.with_suffix(".backup.json")))
        except Exception:
            pass

    def add_trade(self, trade: Trade):
        self.state["trades"].append(asdict(trade))
        self.state["stats"]["total_trades"] += 1
        self.save()

    def close_trade(self, trade_id, price, pnl):
        for t in self.state["trades"]:
            if t["id"] == trade_id:
                t["status"]          = "closed"
                t["close_price"]     = price
                t["pnl"]             = pnl
                t["close_timestamp"] = datetime.utcnow().isoformat()
                self.state["stats"]["total_pnl"] += pnl
                if pnl > 0: self.state["stats"]["wins"]   += 1
                else:       self.state["stats"]["losses"] += 1
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
                break
        self.save()

    def add_signal(self, signal):
        self.state["signals"].append(signal)
        self.state["signals"] = self.state["signals"][-500:]
        self.save()

    def get_open_trades(self):
        return [t for t in self.state["trades"] if t["status"] == "open"]

    def get_all_trades(self):
        return self.state["trades"]


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
        self.max_open      = risk.get("max_open_trades", 8)
        self.min_conf      = self.config.get("strategy", {}).get("min_confidence", 0.52)
        self.htf_filter_mode = self.config.get("strategy", {}).get("htf_filter_mode", "strict")
        self.take_profit   = risk.get("take_profit_pct", 0.05)

        # State — separate files for spot and futures
        state_file    = "state.json" if self.MODE == "spot" else "futures_state.json"
        self.state    = StateManager(state_file)

        # WebSocket price feed (lower latency than REST polling)
        self.ws_feed  = BinanceWSPriceFeed()
        self.ws_feed.start()

        # Shared core systems — IDENTICAL for both modes
        self.feed       = DataFeed(self.exchange, ws_feed=self.ws_feed)
        self.ai         = AIStrategyEngine()
        self.risk       = RiskManager(self.config)
        self.agents     = AgentCoordinator()
        self.learner    = SelfLearner()
        self.notifier   = TelegramNotifier()
        self.scanner    = CoinScanner(self.config)
        self.hmm_regime = HMMRegimeModel()
        self.rl_agent   = RLTradeManager()
        self.gnn_filter = GNNCorrelationFilter()

        self._train()

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

    def _train(self):
        self.log.info("=" * 50)
        self.log.info(f"TRAINING AI MODELS [{self.MODE.upper()} MODE]...")
        self.log.info("=" * 50)
        try:
            import pandas as pd
            watched      = self.scanner.get_coins(self.exchange, invalid_symbols=self.feed.invalid_symbols)
            # Use cached watchlist — don't trigger rescan during training
            train_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"]
            if self.scanner.top_coins:
                # Add up to 2 coins from cached watchlist
                for c in self.scanner.top_coins[:2]:
                    if c not in train_symbols:
                        train_symbols.append(c)
            dfs = []
            for sym in train_symbols:
                try:
                    df = self.feed.fetch_ohlcv(sym, "1h", limit=200)
                    if df is not None and len(df) > 100:
                        dfs.append(df)
                        self.log.info(f"Training data: {sym} ({len(df)} bars)")
                except Exception as e:
                    self.log.warning(f"Could not fetch {sym}: {e}")
            if not dfs:
                self.log.warning("No training data — using BTC fallback")
                df = self.feed.fetch_ohlcv("BTC/USDT", "1h", limit=500)
                if df is not None:
                    dfs = [df]
            if not dfs:
                self.log.error("Training skipped — no data available")
                return
            combined = pd.concat(dfs, ignore_index=True)
            self.log.info(f"Training on {len(combined)} bars from {len(dfs)} coins")
            results = self.ai.train_all(combined)
            self.log.info(f"Training complete: {results}")
            self._check_model_health(results)

            # HMM regime model: train on BTC 1h (captures broad market regimes)
            try:
                btc_df = self.feed.fetch_ohlcv("BTC/USDT", "1h", limit=500)
                if btc_df is not None and len(btc_df) >= 200:
                    hmm_result = self.hmm_regime.train(btc_df)
                    self.log.info(f"HMM regime trained: {hmm_result}")
            except Exception as e:
                self.log.warning(f"HMM training skipped: {e}")
        except Exception as e:
            self.log.error(f"Training failed: {e}", exc_info=True)
            try:
                self.notifier.send_error_alert(
                    f"{type(e).__name__}: {e}", context=f"Training failed [{self.MODE.upper()}]"
                )
            except Exception:
                pass

    def _check_model_health(self, results: dict):
        """Alert if any model trained with suspiciously low accuracy."""
        issues = []
        lstm_r = results.get("lstm", {})
        lstm_acc = lstm_r.get("accuracy", None)
        lstm_status = lstm_r.get("status", "")

        if lstm_status == "below_floor":
            new_acc = lstm_r.get("new_accuracy", 0)
            issues.append(
                f"LSTM new training ({new_acc:.1%}) was below floor — discarded, keeping old"
            )
        elif lstm_acc is not None and lstm_acc < 0.40 and lstm_status not in ("kept_old", "below_floor"):
            issues.append(f"LSTM accuracy {lstm_acc:.1%} is low — check training data")

        if issues:
            msg = f"⚠️ <b>Model Health Warning [{self.MODE.upper()}]</b>\n" + "\n".join(f"• {i}" for i in issues)
            self.log.warning(f"Model health: {'; '.join(issues)}")
            try:
                self.notifier.send(msg)
            except Exception:
                pass

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

    def get_price(self, symbol) -> float:
        price = self.feed.get_live_price(symbol)
        return price or 0.0

    def get_atr(self, symbol) -> float:
        return self.feed.get_atr(symbol, "1h", 14)

    def _get_btc_1h_return(self) -> float:
        try:
            df = self.feed.fetch_ohlcv("BTC/USDT", "1h", limit=3)
            if df is not None and len(df) >= 2:
                return float(df["close"].pct_change().iloc[-1])
        except Exception:
            pass
        return 0.0

    def get_usdt_balance(self) -> float:
        try:
            bal = self.exchange.fetch_balance()
            return float(bal["total"].get("USDT", 0.0))
        except Exception as e:
            self.log.error(f"Balance error: {e}")
            return 0.0

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

    def check_exits(self):
        """
        Check open trades for exit conditions.
        Uses shared RiskManager ATR trailing stops.
        """
        exits = self.risk.check_exits(
            self.state.get_open_trades(),
            self.get_price,
            self.get_atr,
        )
        for trade, price, reason, fraction in exits:
            close_amount = trade["amount"] * fraction
            order = self._place_close(
                trade["symbol"], close_amount, trade["side"]
            )
            if order:
                full_pnl = self._calc_pnl(trade, price)
                pnl      = full_pnl * fraction
                if fraction >= 1.0:
                    self.state.close_trade(trade["id"], price, pnl)
                    self.risk.cleanup_trade(trade["id"])
                else:
                    self.state.partial_close_trade(trade["id"], close_amount, pnl)
                self.ai.record_trade_result(trade["symbol"], pnl)
                self.agents.record_trade_result(trade["symbol"], pnl)
                self.risk.record_trade_result(pnl, self.get_usdt_balance())
                self.rl_agent.record_external_close(trade["id"], pnl)
                icon    = "TP" if pnl > 0 else "SL"
                pct_str = f" ({fraction*100:.0f}%)" if fraction < 1.0 else ""
                self.log.info(f"{icon} EXIT{pct_str} {trade['symbol']} | PnL={pnl:+.4f} | {reason}")
                self.notifier.send_alert(
                    f"{icon} EXIT{pct_str} {trade['symbol']}\n"
                    f"PnL: ${pnl:+.4f} USDT\n"
                    f"Reason: {reason}"
                )

    def _rl_manage_trades(self, regime_ctx=None):
        """
        Apply RL agent to open trades for execution decisions.
        Runs AFTER check_exits() so ATR safety stops always fire first.
        Fallback: HOLD (existing ATR logic handles everything).
        """
        open_trades = self.state.get_open_trades()
        if not open_trades:
            return

        regime  = (regime_ctx or {}).get("hmm_regime", "RANGING")
        balance = self.get_usdt_balance()

        # 1h price for momentum context (use BTC as proxy for market direction)
        price_1h_ago = 0.0
        try:
            btc_df = self.feed.fetch_ohlcv("BTC/USDT", "1h", limit=3)
            if btc_df is not None and len(btc_df) >= 2:
                price_1h_ago = float(btc_df["close"].iloc[-2])
        except Exception:
            pass

        for trade in list(open_trades):
            try:
                symbol        = trade["symbol"]
                current_price = self.get_price(symbol)
                if not current_price:
                    continue
                atr = self.get_atr(symbol)

                # Record step reward from previous cycle's decision
                self.rl_agent.record_step(
                    trade_id    = trade["id"],
                    next_price  = current_price,
                    next_atr    = atr,
                    done        = False,
                )

                action = self.rl_agent.decide(
                    trade         = trade,
                    current_price = current_price,
                    atr           = atr,
                    regime        = regime,
                    price_1h_ago  = price_1h_ago,
                )

                if action == "HOLD":
                    continue

                elif action == "CLOSE":
                    order = self._place_close(symbol, trade["amount"], trade["side"])
                    if order:
                        pnl = self._calc_pnl(trade, current_price)
                        self.state.close_trade(trade["id"], current_price, pnl)
                        self.risk.cleanup_trade(trade["id"])
                        self.ai.record_trade_result(symbol, pnl)
                        self.agents.record_trade_result(symbol, pnl)
                        self.risk.record_trade_result(pnl, balance)
                        self.rl_agent.record_step(trade["id"], current_price, atr, done=True, final_pnl=pnl)
                        self.log.info(f"RL CLOSE {symbol} | PnL={pnl:+.4f}")
                        self.notifier.send_alert(f"RL CLOSE {symbol}\nPnL: ${pnl:+.4f}")

                elif action == "SCALE_OUT":
                    close_amt = trade["amount"] * 0.25
                    if close_amt > 0:
                        order = self._place_close(symbol, close_amt, trade["side"])
                        if order:
                            full_pnl = self._calc_pnl(trade, current_price)
                            pnl      = full_pnl * 0.25
                            self.state.partial_close_trade(trade["id"], close_amt, pnl)
                            self.log.info(f"RL SCALE_OUT {symbol} 25% | PnL={pnl:+.4f}")

                elif action == "SCALE_IN":
                    entry_usdt   = trade["amount"] * float(trade.get("price", current_price) or current_price)
                    extra_usdt   = entry_usdt * 0.25
                    extra_amount = extra_usdt / current_price
                    ok, reason   = self.risk.can_open_trade(
                        symbol=symbol, open_trades=open_trades,
                        balance=balance, new_usdt=extra_usdt,
                        get_price_fn=self.get_price,
                    )
                    if ok and extra_usdt >= 5:
                        if trade["side"] in ("buy", "long"):
                            order = self._place_buy(symbol, extra_amount)
                        else:
                            order = self._place_sell(symbol, extra_amount)
                        if order:
                            self.log.info(f"RL SCALE_IN {symbol} +25% @ {current_price:.4f}")

            except Exception as e:
                self.log.error(f"RL manage error {trade.get('symbol','?')}: {e}")

    def analyze_symbol(self, symbol, balance, open_trades, regime_ctx=None):
        """
        Full autonomous analysis — IDENTICAL logic for spot and futures.
        The only difference is what happens at order placement.
        """
        try:
            dfs = self.feed.fetch_multi_timeframe(symbol)
            if not dfs or "1h" not in dfs:
                return

            df_1h    = dfs["1h"]
            self.ai.ingest_new_data(symbol, df_1h)   # feed online buffer
            htf_bias = self.get_htf_bias(dfs)
            ml_signal = self.ai.predict(df_1h, symbol)
            all_agree = (ml_signal.get("indicators", {}).get("buy_votes", 0) == 3 or
                         ml_signal.get("indicators", {}).get("sell_votes", 0) == 3)
            signal    = self.agents.analyze(symbol, df_1h, ml_signal)

            action = signal["action"]
            conf   = signal["confidence"]
            strat  = signal["strategy"]

            # HTF filter — same for both modes
            htf_conflict = (action == "BUY" and htf_bias == "SELL") or \
                           (action == "SELL" and htf_bias == "BUY")
            if htf_conflict:
                if self.htf_filter_mode == "soft":
                    conf = round(conf * 0.70, 4)
                    self.log.info(f"{symbol} {action} softened by HTF {htf_bias} → conf={conf:.2f}")
                elif self.htf_filter_mode == "hard":
                    if conf < 0.65:
                        action = "HOLD"
                        self.log.info(f"{symbol} {action} hard-blocked by HTF {htf_bias} (conf={conf:.2f} < 0.65)")
                    else:
                        self.log.info(f"{symbol} {action} passed hard HTF gate (conf={conf:.2f} >= 0.65)")
                else:  # strict (default)
                    action = "HOLD"
                    self.log.info(f"{symbol} {action} blocked by HTF {htf_bias} (strict)")

            # BTC momentum modifier — adjust confidence for non-BTC symbols
            if symbol != "BTC/USDT" and action in ("BUY", "SELL"):
                btc_ret = self._get_btc_1h_return()
                if action == "BUY" and btc_ret < -0.015:
                    conf = round(conf * 0.85, 4)
                elif action == "SELL" and btc_ret > 0.015:
                    conf = round(conf * 0.85, 4)
                elif action == "BUY" and btc_ret > 0.015:
                    conf = min(round(conf + 0.04, 4), 0.95)

            self.state.add_signal({
                "symbol": symbol, "action": action, "confidence": conf,
                "strategy": f"HTF:{htf_bias}+{strat}", "timeframe": "AUTO",
                "indicators": signal.get("indicators", {}),
                "timestamp":  datetime.utcnow().isoformat(),
            })

            self.log.info(f"{symbol} | {action} | conf={conf:.2f} | HTF={htf_bias}")

            # Find existing position
            holding = self._find_position(symbol, open_trades)

            # ── CLOSE LOGIC ──────────────────────────────────────
            if action == "SELL" and holding and conf >= self.min_conf:
                order = self._place_close(symbol, holding["amount"], holding["side"])
                if order:
                    price = self.get_price(symbol)
                    pnl   = self._calc_pnl(holding, price)
                    self.state.close_trade(holding["id"], price, pnl)
                    self.ai.record_trade_result(symbol, pnl)
                    self.agents.record_trade_result(symbol, pnl)
                    self.risk.record_trade_result(pnl, balance)
                    self.risk.cleanup_trade(holding["id"])
                    self.log.info(f"AI CLOSE {symbol} | PnL={pnl:+.4f}")
                    self.notifier.send_alert(f"AI CLOSE {symbol}\nPnL: ${pnl:+.4f}")
                return

            # ── REGIME GATE (new entries only) ───────────────────
            if regime_ctx:
                if not regime_ctx.get("gate", True):
                    if action in ["BUY", "SELL"]:
                        self.log.info(
                            f"{symbol} {action} blocked: regime={regime_ctx['regime']}"
                        )
                    return
                if action == "BUY" and not regime_ctx.get("allow_longs", True):
                    self.log.info(
                        f"{symbol} BUY blocked — longs off in {regime_ctx['regime']}"
                    )
                    return
                if action == "SELL" and not regime_ctx.get("allow_shorts", True):
                    self.log.info(
                        f"{symbol} SELL blocked — shorts off in {regime_ctx['regime']}"
                    )
                    return

            # ── OPEN LOGIC ───────────────────────────────────────
            if action not in ["BUY", "SELL"]: return
            eff_conf = max(
                self.min_conf,
                regime_ctx.get("min_conf", self.min_conf) if regime_ctx else self.min_conf,
            )
            if conf < eff_conf:
                self.log.info(
                    f"{symbol} conf={conf:.2f} < eff_min={eff_conf:.2f}"
                    f" ({regime_ctx.get('regime', '?') if regime_ctx else 'default'})"
                )
                return
            if holding:                       return

            # Double-check exchange directly to prevent duplicate positions
            if self.MODE == "futures":
                try:
                    positions = self.exchange.get_position()
                    if any(p["symbol"] == symbol for p in positions):
                        self.log.info(f"SKIP {symbol}: position already exists on exchange")
                        return
                except Exception:
                    pass


            price = self.get_price(symbol)
            if not price:
                return

            # Kelly Criterion position sizing — same for both modes
            portfolio_mult = regime_ctx.get("size_mult", 1.0) if regime_ctx else 1.0
            _, est_usdt = self.risk.get_position_size(
                confidence=conf, balance=balance, price=price,
                df=df_1h, recent_trades=self.state.get_all_trades(),
                portfolio_mult=portfolio_mult, all_agree=all_agree,
            )

            # Full risk checks — same for both modes
            ok, reason = self.risk.can_open_trade(
                symbol=symbol, open_trades=open_trades,
                balance=balance, new_usdt=est_usdt,
                get_price_fn=self.get_price,
            )
            if not ok:
                self.log.info(f"SKIP {symbol}: {reason}")
                return

            # GNN correlation filter (additional to rule-based CorrelationFilter)
            open_syms       = [t["symbol"] for t in open_trades]
            gnn_ok, gnn_msg, gnn_score = self.gnn_filter.check(symbol, open_syms)
            if not gnn_ok:
                self.log.info(f"GNN SKIP {symbol}: {gnn_msg}")
                return

            amount, usdt = self.risk.get_position_size(
                confidence=conf, balance=balance, price=price,
                df=df_1h, recent_trades=self.state.get_all_trades(),
                portfolio_mult=portfolio_mult, all_agree=all_agree,
            )
            if usdt < 10:
                return

            # Place order — subclass handles direction
            if action == "BUY":
                order = self._place_buy(symbol, amount)
                side  = "buy" if self.MODE == "spot" else "long"
            else:
                order = self._place_sell(symbol, amount)
                side  = "sell" if self.MODE == "spot" else "short"

            if order:
                fill_price = float(
                    order.get("average") or order.get("price") or price
                )
                trade = Trade(
                    id        = order.get("id", f"t_{int(time.time())}"),
                    symbol    = symbol,
                    side      = side,
                    amount    = amount,
                    price     = fill_price,
                    timestamp = datetime.utcnow().isoformat(),
                    strategy  = strat,
                    timeframe = f"AUTO-{self.MODE}",
                    status    = "open",
                    mode      = self.MODE,
                    leverage  = self._get_leverage(),
                )
                self.state.add_trade(trade)
                self.log.info(
                    f"{side.upper()} {symbol} | ${usdt:.2f} | conf={conf:.2f}"
                )
                self.notifier.send_alert(
                    f"{side.upper()} {symbol}\n"
                    f"Amount: ${usdt:.2f} USDT\n"
                    f"Price: ${fill_price:.4f}\n"
                    f"Confidence: {conf:.0%}\n"
                    f"HTF: {htf_bias}\n"
                    f"Mode: {self.MODE.upper()}"
                )

        except Exception as e:
            self.log.error(f"Error analyzing {symbol}: {e}", exc_info=True)

    def _find_position(self, symbol, open_trades):
        """Find existing position for a symbol."""
        return next(
            (t for t in open_trades if t["symbol"] == symbol),
            None
        )

    def _get_leverage(self) -> int:
        """Override in futures bot."""
        return 1

    def sync_with_exchange(self):
        """Sync state with real exchange positions."""
        if self.MODE != "futures":
            return
        try:
            self.log.info("Sync running — mode=futures")
            try:
                positions = self.exchange.get_position()
                self.log.info(f"Sync step 2: got {len(positions) if positions else 0} positions")
            except Exception as pe:
                self.log.warning(f"get_position failed: {pe}")
                positions = []
            if not positions:
                positions = []

            bal  = self.exchange.fetch_balance()
            usdt = float(bal["total"].get("USDT", 0))

            d        = self.state.state
            our_open = {t["symbol"]: t for t in d["trades"] if t["status"] == "open"}
            binance_syms = {p["symbol"] for p in positions}
            our_syms = set(our_open.keys())

            for pos in positions:
                sym = pos["symbol"]
                if sym in our_syms:
                    for t in d["trades"]:
                        if t["symbol"] == sym and t["status"] == "open":
                            t["amount"]    = pos["amount"]
                            t["price"]     = pos["entry_price"]
                            t["leverage"]  = pos.get("leverage", 5)
                            t["live_pnl"]  = round(pos["pnl"], 6)
                            t["mark_price"]= pos.get("mark_price", 0)

            import time as _time
            for pos in positions:
                sym = pos["symbol"]
                if sym not in our_syms:
                    trade = {
                        "id":              f"sync_{sym.replace('/','_')}_{int(_time.time())}",
                        "symbol":          sym,
                        "side":            pos["side"],
                        "amount":          pos["amount"],
                        "price":           pos["entry_price"],
                        "mark_price":      pos.get("mark_price", 0),
                        "live_pnl":        round(pos["pnl"], 6),
                        "timestamp":       datetime.utcnow().isoformat(),
                        "strategy":        "synced_from_exchange",
                        "timeframe":       f"AUTO-{self.MODE}",
                        "status":          "open",
                        "mode":            self.MODE,
                        "leverage":        pos.get("leverage", 5),
                        "pnl":             0.0,
                        "close_price":     0.0,
                        "close_timestamp": ""
                    }
                    d["trades"].append(trade)
                    d["stats"]["total_trades"] += 1
                    self.log.info(f"Sync: added {sym} {pos['side']}")


            for t in d["trades"]:
                if t["status"] == "open" and t["symbol"] not in binance_syms:
                    try:
                        last_price = self.exchange.fetch_ticker(t["symbol"])["last"]
                        entry  = float(t["price"])
                        amount = float(t["amount"])
                        side   = t.get("side", "long")
                        pnl    = (last_price - entry) * amount if side == "long" else (entry - last_price) * amount
                    except Exception:
                        last_price = float(t["price"])
                        pnl        = 0.0
                    t["status"]          = "closed"
                    t["close_price"]     = last_price
                    t["pnl"]             = round(pnl, 6)
                    t["close_timestamp"] = datetime.utcnow().isoformat()
                    self.log.info(f"Sync: closed orphan {t['symbol']} pnl={pnl:+.4f}")

            live_pnl = round(sum(p["pnl"] for p in positions), 4)
            self.log.info(f"Sync saving balance=${usdt:.2f} live_pnl={live_pnl:.4f}")
            self.state.state["stats"]["balance"]        = round(usdt, 2)
            self.state.state["stats"]["last_sync"]      = datetime.utcnow().isoformat()
            self.state.state["stats"]["total_live_pnl"] = live_pnl
            self.state.save()

        except Exception as e:
            import traceback
            self.log.warning(f"Sync error: {e}")
            self.log.warning(traceback.format_exc())

    def _get_watchlist_hash(self, coins):
        return hash(tuple(sorted(coins)))

    def _retrain_if_watchlist_changed(self, new_coins):
        new_hash = self._get_watchlist_hash(new_coins)
        if hasattr(self, '_last_watchlist_hash') and self._last_watchlist_hash == new_hash:
            return
        if hasattr(self, '_last_watchlist_hash'):
            added   = set(new_coins) - set(getattr(self, '_last_watchlist', []))
            removed = set(getattr(self, '_last_watchlist', [])) - set(new_coins)
            if added or removed:
                self.log.info(f"Watchlist changed! Added:{added} Removed:{removed}")
                try:
                    self.notifier.send(
                        f"ℹ️ <b>Watchlist changed [{self.MODE.upper()}]</b>\n"
                        f"Added: {added or 'none'}\n"
                        f"Removed: {removed or 'none'}\n"
                        f"Retraining models on new coin set…"
                    )
                except Exception:
                    pass
                # Retrain without deleting existing models:
                # champion/challenger in each model's train() will decide
                # whether to keep the old or save the new.
                self._train()
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

    def run_once(self):
        """One scan cycle — identical for both modes."""
        self.sync_with_exchange()
        self.check_exits()

        balance     = self.get_usdt_balance()
        open_trades = self.state.get_open_trades()

        # Check if trading paused via dashboard
        trading_paused = self.is_trading_paused()
        if trading_paused:
            self.log.info("Trading PAUSED via dashboard — monitoring only")

        can_trade, reason = self.risk.breaker.can_trade(balance)
        if not can_trade:
            self.log.warning(f"Circuit breaker: {reason}")

        max_reached = len(open_trades) >= self.max_open
        symbols     = self.scanner.get_coins(self.exchange, invalid_symbols=self.feed.invalid_symbols)
        self.feed.subscribe_many(symbols)   # ensure WS + REST monitor tracking
        self.log.info(f"[{self.MODE.upper()}] Watching: {symbols}")

        regime_ctx = self.risk.detect_market_regime(self.feed, symbols)

        # HMM overlay: adjusts thresholds only, never overrides signal direction
        try:
            btc_df = self.feed.fetch_ohlcv("BTC/USDT", "1h", limit=150)
            if btc_df is not None:
                hmm_regime, hmm_adj = self.hmm_regime.get_regime_and_adjustments(btc_df)
                if regime_ctx:
                    old_min_conf  = regime_ctx.get("min_conf", self.min_conf)
                    old_size_mult = regime_ctx.get("size_mult", 1.0)
                    regime_ctx["min_conf"]   = round(old_min_conf  + hmm_adj["min_conf_delta"], 4)
                    regime_ctx["size_mult"]  = round(old_size_mult * hmm_adj["size_mult"], 4)
                    regime_ctx["hmm_regime"] = hmm_regime
                self.log.info(
                    f"HMM regime: {hmm_regime} | "
                    f"min_conf_delta={hmm_adj['min_conf_delta']:+.2f} "
                    f"size_mult={hmm_adj['size_mult']:.2f}"
                )
        except Exception as e:
            self.log.warning(f"HMM overlay failed: {e}")

        # RL execution layer: manages open trades, runs after ATR stops + HMM context ready
        self._rl_manage_trades(regime_ctx)

        # GNN correlation graph: build/refresh from cached 1h OHLCV (fast — hits cache)
        try:
            gnn_dfs = {}
            for s in symbols:
                df = self.feed.fetch_ohlcv(s, "1h", limit=168)
                if df is not None and len(df) >= 24:
                    gnn_dfs[s] = df
            if len(gnn_dfs) >= 2:
                self.gnn_filter.update_graph(gnn_dfs)
                stats = self.gnn_filter.graph_stats()
                self.log.info(
                    f"GNN graph: {stats['n_nodes']} nodes "
                    f"{stats['n_edges']} edges "
                    f"avg_corr={stats['avg_corr']}"
                )
        except Exception as e:
            self.log.warning(f"GNN update failed: {e}")

        # Online learning: retrain with decay weights when enough new bars buffered
        try:
            ol_result = self.ai.incremental_update()
            if ol_result.get("status") == "updated":
                self.log.info(f"Online learning update: {ol_result.get('results', {})}")
        except Exception as e:
            self.log.warning(f"Online learning update failed: {e}")

        for symbol in symbols:
            if not can_trade or max_reached or trading_paused:
                # Still save signals for dashboard
                try:
                    dfs = self.feed.fetch_multi_timeframe(symbol)
                    if dfs and "1h" in dfs:
                        ml = self.ai.predict(dfs["1h"], symbol)
                        s  = self.agents.analyze(symbol, dfs["1h"], ml)
                        self.state.add_signal(s)
                except Exception:
                    pass
                continue

            self.analyze_symbol(symbol, balance, open_trades, regime_ctx)
            time.sleep(0.5)


    def run(self):
        self.log.info("=" * 50)
        self.log.info(f"CRYPTOBOT v3 STARTED — MODE: {self.MODE.upper()}")
        self.log.info("=" * 50)

        self.notifier.send_alert(
            f"CryptoBot v3 Started\n"
            f"Mode: {self.MODE.upper()}"
        )

        last_train = datetime.utcnow().date()

        while True:
            try:
                today = datetime.utcnow().date()
                if today != last_train:
                    self._train()
                    last_train = today

                current_coins = self.scanner.get_coins(self.exchange, invalid_symbols=self.feed.invalid_symbols)
                self._retrain_if_watchlist_changed(current_coins)

                if self.learner.should_run():
                    self.learner.run_learning_cycle()

                self.run_once()
                self.notifier.exchange = self.exchange
                self.notifier.send_report(self.exchange)

            except Exception as e:
                self.log.error(f"Cycle error: {e}", exc_info=True)

            self.log.info(f"Sleeping {self.scan_interval}s...")
            time.sleep(self.scan_interval)
