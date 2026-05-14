# bot/tests/test_exit_fill_price.py

def _calc_pnl_stub(trade, price):
    direction = -1 if trade.get("side") == "short" else 1
    return direction * (price - trade["price"]) * trade["amount"] * trade.get("leverage", 1)


def _get_close_pnl(trade, fills, fraction):
    """Simulate _resolve_close_pnl logic — always uses _calc_pnl with fill price."""
    closing_side = "sell" if trade["side"] == "long" else "buy"
    closing_fills = [f for f in fills if f.get("side", "").lower() == closing_side]
    if closing_fills:
        best = max(closing_fills, key=lambda x: x["time"])
        actual_price = float(best.get("price", trade["price"]))
        return _calc_pnl_stub(trade, actual_price) * fraction, actual_price
    detection_price = trade.get("_detection_price", float(trade["price"]))
    return _calc_pnl_stub(trade, detection_price) * fraction, detection_price


def test_uses_calc_pnl_with_fill_price_not_exchange_rpnl():
    """Exchange realizedPnl is ignored; _calc_pnl with fill price is used instead."""
    trade = {"side": "long", "price": 100.0, "amount": 1.0, "leverage": 5}
    fills = [{"side": "sell", "price": "105.0", "realizedPnl": "24.5", "time": 999}]
    pnl, price = _get_close_pnl(trade, fills, fraction=1.0)
    assert pnl == 25.0   # _calc_pnl: (105-100)*1*5
    assert price == 105.0


def test_uses_fill_price_when_no_realized_pnl():
    trade = {"side": "long", "price": 100.0, "amount": 1.0, "leverage": 5}
    fills = [{"side": "sell", "price": "106.0", "realizedPnl": "0", "time": 999}]
    pnl, price = _get_close_pnl(trade, fills, fraction=1.0)
    assert price == 106.0
    assert pnl == 30.0  # (106-100)*1*5


def test_falls_back_to_detection_price_when_no_fill():
    trade = {"side": "long", "price": 100.0, "amount": 1.0, "leverage": 5,
             "_detection_price": 103.0}
    pnl, price = _get_close_pnl(trade, [], fraction=1.0)
    assert price == 103.0
    assert pnl == 15.0  # (103-100)*1*5


def test_short_trade_uses_buy_fill():
    trade = {"side": "short", "price": 100.0, "amount": 1.0, "leverage": 5}
    fills = [
        {"side": "buy",  "price": "95.0", "realizedPnl": "0", "time": 999},
        {"side": "sell", "price": "97.0", "realizedPnl": "0", "time": 1000},
    ]
    pnl, price = _get_close_pnl(trade, fills, fraction=1.0)
    assert price == 95.0  # buy fill, not sell
    assert pnl == 25.0   # (100-95)*1*5


def test_partial_close_scales_pnl_by_fraction():
    """fraction applied to _calc_pnl(full_position), not to exchange realizedPnl."""
    trade = {"side": "long", "price": 100.0, "amount": 2.0, "leverage": 5}
    fills = [{"side": "sell", "price": "110.0", "realizedPnl": "50.0", "time": 999}]
    pnl, price = _get_close_pnl(trade, fills, fraction=0.5)
    # _calc_pnl(full 2.0 amount) = (110-100)*2*5 = 100 ; * fraction 0.5 = 50
    assert pnl == 50.0


def test_resolve_close_pnl_ignores_rpnl_uses_calc_pnl_with_fill_price():
    """BaseBot._resolve_close_pnl must use _calc_pnl with fill price, not exchange realizedPnl."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from engine.bot import BaseBot
    from engine.futures import FuturesBot

    class FakeExchange:
        def fetch_my_trades(self, symbol, limit=10):
            return [{"side": "sell", "price": "105.0", "realizedPnl": "24.5", "time": 999}]

    bot = object.__new__(FuturesBot)
    bot.exchange = FakeExchange()
    bot.leverage = 5
    bot.config = {}
    bot.log = type("L", (), {"debug": lambda self, *a, **k: None})()

    trade = {"side": "long", "price": 100.0, "amount": 1.0, "leverage": 5, "pnl": 0.0}
    pnl, price = bot._resolve_close_pnl(trade, "BTC/USDT", detection_price=103.0, fraction=1.0)
    # _calc_pnl: (105-100)*1*5 - fee ≈ 24.96
    expected = bot._calc_pnl(trade, 105.0)
    assert abs(pnl - expected) < 0.001
    assert price == 105.0
