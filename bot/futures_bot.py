"""
Futures Bot v3 — Inherits BaseBot, uses Binance Demo Futures Trading
SAME strategy as spot, but supports LONG and SHORT with leverage.
Uses https://demo-fapi.binance.com endpoint.
"""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from env_config import create_demo_exchange
from base_bot   import BaseBot

log = logging.getLogger("FuturesBot")


class FuturesBot(BaseBot):
    """
    Futures trading bot on Binance Demo Trading.

    LONG  = open long position (profit when price goes UP)
    SHORT = open short position (profit when price goes DOWN)
    Leverage amplifies both profits and losses.

    Uses SAME strategy as SpotBot:
    - Same ML models
    - Same confidence gate
    - Same ATR trailing stops
    - Same Kelly Criterion sizing
    - Same portfolio heat tracking
    - Same agent system

    Only difference: can trade both directions + leverage
    """

    MODE = "futures"

    def __init__(self):
        super().__init__(
            config_file="config_futures.yaml",
            log_file="futures_bot.log",
        )
        self.leverage = self.config.get("risk", {}).get("leverage", 5)
        log.info(f"FuturesBot v3 initialized — DEMO TRADING — {self.leverage}x leverage")
        self._setup_leverage_for_symbols()

    def _setup_exchange(self):
        return create_demo_exchange(mode="futures")

    def _setup_leverage_for_symbols(self):
        """Set leverage and margin mode for all watched symbols."""
        try:
            symbols     = self.scanner.get_coins(self.exchange)
            margin_type = self.config.get("exchange", {}).get("margin_type", "ISOLATED")
            for symbol in symbols:
                try:
                    self.exchange.set_leverage(self.leverage, symbol)
                except Exception as e:
                    log.debug(f"Leverage {symbol}: {e}")
                try:
                    self.exchange.set_margin_mode(margin_type.lower(), symbol)
                except Exception as e:
                    if "-4046" not in str(e):
                        log.debug(f"Margin {symbol}: {e}")
        except Exception as e:
            log.warning(f"Leverage setup failed: {e}")

    def _place_buy(self, symbol, amount):
        """Open LONG position."""
        return self.place_order_with_confirmation(symbol, "buy", amount)

    def _place_sell(self, symbol, amount):
        """Open SHORT position."""
        return self.place_order_with_confirmation(symbol, "sell", amount)

    def _place_close(self, symbol, amount, side):
        """
        Close position with reduce-only flag.
        Long position closed with sell.
        Short position closed with buy.
        """
        close_side = "sell" if side == "long" else "buy"
        return self.place_order_with_confirmation(
            symbol, close_side, amount,
            params={"reduceOnly": True},
        )

    def _calc_pnl(self, trade, close_price) -> float:
        """
        Futures PnL - matches Binance calculation exactly.
        PnL = (close - entry) * amount  (long)
        PnL = (entry - close) * amount  (short)
        Note: Leverage affects margin requirement, NOT raw PnL.
        Raw PnL already reflects leveraged exposure via position size.
        """
        entry  = float(trade["price"])
        amount = float(trade["amount"])
        if trade["side"] == "long":
            return (close_price - entry) * amount
        else:
            return (entry - close_price) * amount

    def _get_leverage(self) -> int:
        return self.leverage

    def _place_exchange_stop_loss(self, symbol: str, side: str, amount: float, entry_price: float, atr: float) -> str:
        """
        Place exchange-side STOP_MARKET order for safety.
        Returns the stop-loss order ID.
        """
        try:
            sl_atr_mult = self.config.get("risk", {}).get("stop_loss_atr_multiplier", 2.0)
            if side == "long":
                stop_price = entry_price - (atr * sl_atr_mult)
                close_side = "sell"
            else:
                stop_price = entry_price + (atr * sl_atr_mult)
                close_side = "buy"

            if stop_price <= 0:
                self.log.warning(f"Invalid SL price for {symbol}: {stop_price}")
                return ""

            order = self.exchange.create_stop_market_order(
                symbol, close_side, amount, stop_price,
                params={"reduceOnly": True},
            )
            self.log.info(f"Exchange SL placed for {symbol}: {side} @ {stop_price:.4f} (order_id={order['id']})")
            return order["id"]
        except Exception as e:
            self.log.error(f"Failed to place exchange SL for {symbol}: {e}")
            return ""

    def _cancel_exchange_stop_loss(self, symbol: str, sl_order_id: str):
        """Cancel an existing exchange stop-loss order."""
        if not sl_order_id:
            return
        try:
            self.exchange.cancel_order(sl_order_id, symbol)
            self.log.info(f"Exchange SL cancelled for {symbol}: {sl_order_id}")
        except Exception as e:
            self.log.debug(f"Failed to cancel exchange SL for {symbol}: {e}")


if __name__ == "__main__":
    bot = FuturesBot()
    bot.run()
