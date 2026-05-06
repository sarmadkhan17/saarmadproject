"""
Spot Bot v3 — Inherits BaseBot, uses Binance Demo Trading
SAME strategy as v2 spot bot, just runs on demo instead of testnet.
"""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from core.config import create_demo_exchange
from engine.bot   import BaseBot

log = logging.getLogger("SpotBot")


class SpotBot(BaseBot):
    """
    Spot trading bot on Binance Demo Trading.
    BUY = open position
    SELL = close position
    No leverage.
    """

    MODE = "spot"

    def __init__(self):
        super().__init__(config_file="config_spot.yaml", log_file="spot_bot.log")
        log.info("SpotBot v3 initialized — DEMO TRADING — BUY/SELL only")

    def _setup_exchange(self):
        return create_demo_exchange(mode="spot")

    def _place_buy(self, symbol, amount):
        return self.place_order_with_confirmation(symbol, "buy", amount)

    def _place_sell(self, symbol, amount):
        return self.place_order_with_confirmation(symbol, "sell", amount)

    def _place_close(self, symbol, amount, side):
        # In spot, close = sell
        return self.place_order_with_confirmation(symbol, "sell", amount)

    def _calc_pnl(self, trade, close_price) -> float:
        entry  = float(trade["price"])
        amount = float(trade["amount"])
        pnl    = (close_price - entry) * amount
        fee    = entry * amount * 0.001 * 2   # 0.1% taker × 2 sides
        return pnl - fee

    def _get_leverage(self) -> int:
        return 1


if __name__ == "__main__":
    bot = SpotBot()
    bot.run()
