"""
Futures Bot v3 — Inherits BaseBot, uses Binance Demo Futures Trading
SAME strategy as spot, but supports LONG and SHORT with leverage.
Uses https://demo-fapi.binance.com endpoint.
"""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from core.config import create_demo_exchange
from engine.bot   import BaseBot

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
        self._leveraged_symbols = set()
        log.info(f"FuturesBot v3 initialized — DEMO TRADING — {self.leverage}x leverage")
        self._setup_leverage_for_symbols()

    def _setup_exchange(self):
        return create_demo_exchange(mode="futures")

    def _setup_leverage_for_symbols(self, symbols=None):
        """Set leverage and margin mode for symbols not yet configured."""
        try:
            if symbols is None:
                symbols = self.scanner.get_coins(self.exchange)
            new_symbols = [s for s in symbols if s not in self._leveraged_symbols]
            if not new_symbols:
                return
            margin_type = self.config.get("exchange", {}).get("margin_type", "ISOLATED")
            for symbol in new_symbols:
                try:
                    self.exchange.set_leverage(self.leverage, symbol)
                except Exception as e:
                    log.debug(f"Leverage {symbol}: {e}")
                try:
                    self.exchange.set_margin_mode(margin_type.lower(), symbol)
                except Exception as e:
                    if "-4046" not in str(e):
                        log.debug(f"Margin {symbol}: {e}")
                self._leveraged_symbols.add(symbol)
            if new_symbols:
                log.info(f"Leverage set for {len(new_symbols)} new coins ({self.leverage}x)")
        except Exception as e:
            log.warning(f"Leverage setup failed: {e}")

    def _liquidation_safe(self, symbol: str, side: str) -> bool:
        """Return False if current price is within 2×ATR of estimated liquidation price."""
        try:
            price = self.get_price(symbol)
            atr   = self.get_atr(symbol)
            lev   = self._get_leverage()
            if not price or not atr or lev <= 0:
                return True
            dist = price / lev  # distance to liquidation ≈ price / leverage
            if dist < 2.0 * atr:
                log.warning(f"[{symbol}] Liq too close: dist={dist:.4f} < 2×ATR={2*atr:.4f} — blocked")
                return False
            return True
        except Exception:
            return True  # fail-open

    def _effective_leverage(self, regime: str) -> int:
        """Halve leverage during HIGH_VOL or CRASH regimes."""
        if regime in ("HIGH_VOL", "CRASH"):
            return max(1, self.leverage // 2)
        return self.leverage

    def _place_buy(self, symbol, amount):
        """Open LONG position."""
        if not self._liquidation_safe(symbol, "buy"):
            return None
        regime = getattr(self, '_current_regime', 'RANGING')
        lev = self._effective_leverage(regime)
        if lev != self.leverage:
            try:
                self.exchange.set_leverage(lev, symbol)
            except Exception:
                pass
        result = self.place_order_with_confirmation(symbol, "buy", amount)
        if lev != self.leverage:
            try:
                self.exchange.set_leverage(self.leverage, symbol)
            except Exception:
                pass
        return result

    def _post_scan(self, symbols):
        self._setup_leverage_for_symbols(symbols)

    def _place_sell(self, symbol, amount):
        """Open SHORT position."""
        if not self._liquidation_safe(symbol, "sell"):
            return None
        regime = getattr(self, '_current_regime', 'RANGING')
        lev = self._effective_leverage(regime)
        if lev != self.leverage:
            try:
                self.exchange.set_leverage(lev, symbol)
            except Exception:
                pass
        result = self.place_order_with_confirmation(symbol, "sell", amount)
        if lev != self.leverage:
            try:
                self.exchange.set_leverage(self.leverage, symbol)
            except Exception:
                pass
        return result

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
        Futures PnL - leveraged, includes fee deduction.
        PnL = (close - entry) * amount * leverage  (long)
        PnL = (entry - close) * amount * leverage  (short)
        Fee = 0.04% taker × 2 sides (on notional, not margin)
        """
        entry  = float(trade["price"])
        amount = float(trade["amount"])
        lev    = float(trade.get("leverage", self._get_leverage()))
        fee    = entry * amount * 0.0004 * 2
        if trade["side"] == "long":
            return (close_price - entry) * amount * lev - fee
        else:
            return (entry - close_price) * amount * lev - fee

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
                params={"closePosition": True},
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
