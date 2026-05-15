"""
Execution Engine — places orders, confirms fills, manages stop losses.
Extracted from BaseBot.place_order_with_confirmation and related methods.
"""

import time
import logging
import threading
from typing import Optional
from datetime import datetime, timezone

from core.types import Trade

log = logging.getLogger(__name__)


class ExecutionEngine:
    def __init__(self, exchange, state, notifier, mode: str = "spot", get_leverage_fn=None):
        self.exchange    = exchange
        self.state       = state
        self.notifier    = notifier
        self.mode        = mode
        self._get_leverage = get_leverage_fn or (lambda: 1)
        self._sym_locks: dict = {}
        self._lock_registry = threading.Lock()

    def _get_sym_lock(self, symbol: str) -> threading.Lock:
        with self._lock_registry:
            if symbol not in self._sym_locks:
                self._sym_locks[symbol] = threading.Lock()
            return self._sym_locks[symbol]

    def place_with_confirmation(self, symbol: str, side: str, amount: float,
                                 params: dict = None, max_retries: int = 3) -> Optional[dict]:
        for attempt in range(max_retries):
            try:
                if params:
                    order = self.exchange.create_market_order(symbol, side, amount, params=params)
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
                        log.info(f"Order confirmed: {side.upper()} {amount:.6f} {symbol}")
                        return filled
                    elif status == "open":
                        return filled
                except Exception:
                    return order

            except Exception as e:
                err_str = str(e).lower()
                if "insufficient" in err_str or "balance" in err_str:
                    log.error(f"Insufficient funds: {symbol}")
                    return None
                if "invalid" in err_str and "order" in err_str:
                    log.error(f"Invalid order {symbol}: {e}")
                    return None
                log.error(f"Order error {symbol} attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        return None

    def execute_entry(self, decision, symbol: str, action: str, price: float,
                      get_atr_fn, place_buy_fn, place_sell_fn,
                      strat: str = "") -> Optional[Trade]:
        lock = self._get_sym_lock(symbol)
        if not lock.acquire(blocking=False):
            log.warning(f"[{symbol}] Concurrent execution blocked — already in flight")
            return None
        try:
            return self._execute_entry_locked(
                decision, symbol, action, price, get_atr_fn, place_buy_fn, place_sell_fn, strat
            )
        finally:
            lock.release()

    def _execute_entry_locked(self, decision, symbol: str, action: str, price: float,
                               get_atr_fn, place_buy_fn, place_sell_fn,
                               strat: str = "") -> Optional[Trade]:
        amount = decision.position_size
        est_usdt = decision.est_usdt

        if action == "BUY":
            order = place_buy_fn(symbol, amount)
            side  = "buy" if self.mode == "spot" else "long"
        else:
            order = place_sell_fn(symbol, amount)
            side  = "sell" if self.mode == "spot" else "short"

        if not order:
            return None

        fill_price = float(order.get("average") or order.get("price") or price)

        sl_id = ""
        if self.mode == "futures":
            atr = get_atr_fn(symbol)
            sl_id = self._place_sl(symbol, "long" if action == "BUY" else "short",
                                    amount, fill_price, atr)
            if not sl_id:
                _is_demo = 'demo' in getattr(self.exchange, 'base_url', '')
                if _is_demo:
                    log.warning(f"[{symbol}] SL not supported on demo exchange — keeping trade open without hard stop")
                else:
                    log.error(f"[{symbol}] SL placement failed after fill — aborting position")
                    close_side = "sell" if action == "BUY" else "buy"
                    close_price = fill_price
                    try:
                        close_order = self.exchange.create_market_order(symbol, close_side, amount)
                        log.warning(f"[{symbol}] Emergency close placed after SL failure")
                        if close_order:
                            close_price = float(close_order.get("average") or close_order.get("price") or fill_price)
                    except Exception as ce:
                        log.error(f"[{symbol}] Emergency close also failed: {ce}")
                    leverage = self._get_leverage()
                    pnl = ((close_price - fill_price) if side in ("buy", "long") else (fill_price - close_price)) * amount * leverage
                    aborted = Trade(
                        id=order.get("id", f"t_{int(time.time())}"),
                        symbol=symbol, side=side, amount=amount, price=fill_price,
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        strategy=f"{strat}_sl_abort", timeframe=f"AUTO-{self.mode}",
                        status="open", mode=self.mode, leverage=leverage, sl_order_id="",
                    )
                    self.state.add_trade(aborted)
                    self.state.close_trade(aborted.id, close_price, round(pnl, 8))
                    return "SL_FAILED"

        trade = Trade(
            id=order.get("id", f"t_{int(time.time())}"),
            symbol=symbol, side=side, amount=amount, price=fill_price,
            timestamp=datetime.now(timezone.utc).isoformat(),
            strategy=strat, timeframe=f"AUTO-{self.mode}",
            status="open", mode=self.mode,
            leverage=self._get_leverage(), sl_order_id=sl_id,
        )
        self.state.add_trade(trade)

        direction = "BUY" if side in ("buy", "long") else "SELL"
        log.info(f"SIGNAL {symbol} → {direction} | conf={decision.adjusted_conf:.2f} | TAKEN ${est_usdt:.0f} @ ${fill_price:.4f} ({side})")
        profile_name = getattr(decision, 'profile', '?')
        self.notifier.send_alert(
            f"{side.upper()} {symbol} [{profile_name}]\n"
            f"Amount: ${est_usdt:.2f} USDT\n"
            f"Price: ${fill_price:.4f}\n"
            f"Confidence: {decision.adjusted_conf:.0%}\n"
            f"HTF: {decision.htf_bias} | Regime: {decision.hmm_regime}\n"
            f"Mode: {self.mode.upper()}"
        )
        return trade

    def execute_close(self, trade: dict, amount: float, side: str, place_close_fn) -> bool:
        order = place_close_fn(trade["symbol"], amount, side)
        return order is not None

    def _place_sl(self, symbol: str, side: str, amount: float,
                   entry_price: float, atr: float) -> str:
        try:
            sl_mult = getattr(self, 'sl_atr_mult', 3.0)
            sl_min_pct = getattr(self, 'sl_min_pct', 0.015)
            sl_price = (entry_price - sl_mult * atr) if side == "long" else (entry_price + sl_mult * atr)
            min_distance = entry_price * sl_min_pct
            if side == "long":
                sl_price = min(sl_price, entry_price - min_distance)
                sl_price = max(sl_price, entry_price * 0.75)
            else:
                sl_price = max(sl_price, entry_price + min_distance)
                sl_price = min(sl_price, entry_price * 1.25)
            order = self.exchange.create_stop_market_order(
                symbol,
                "sell" if side == "long" else "buy",
                amount,
                round(sl_price, 4),
                params={"reduceOnly": True, "workingType": "MARK_PRICE"},
            )
            return order.get("id", "") if order else ""
        except Exception as e:
            log.warning(f"SL placement failed {symbol}: {e}")
            return ""

    def cancel_sl(self, symbol: str, sl_order_id: str):
        if not sl_order_id:
            return
        try:
            self.exchange.cancel_order(sl_order_id, symbol)
        except Exception:
            pass
