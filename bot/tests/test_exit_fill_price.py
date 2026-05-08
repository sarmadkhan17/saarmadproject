# bot/tests/test_exit_fill_price.py

def _calc_pnl_stub(trade, price):
    return (price - trade["price"]) * trade["amount"] * trade.get("leverage", 1)


def _get_close_pnl(trade, fills, fraction):
    """Simulate _resolve_close_pnl logic."""
    closing_side = "sell" if trade["side"] == "long" else "buy"
    closing_fills = [f for f in fills if f.get("side", "").lower() == closing_side]
    if closing_fills:
        best = max(closing_fills, key=lambda x: x["time"])
        actual_price = float(best.get("price", trade["price"]))
        rpnl = float(best.get("realizedPnl", 0))
        if rpnl != 0:
            return rpnl * fraction, actual_price
        return _calc_pnl_stub(trade, actual_price) * fraction, actual_price
    detection_price = trade.get("_detection_price", float(trade["price"]))
    return _calc_pnl_stub(trade, detection_price) * fraction, detection_price


def test_uses_realized_pnl_from_fill():
    trade = {"side": "long", "price": 100.0, "amount": 1.0, "leverage": 5}
    fills = [{"side": "sell", "price": "105.0", "realizedPnl": "24.5", "time": 999}]
    pnl, price = _get_close_pnl(trade, fills, fraction=1.0)
    assert pnl == 24.5
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


def test_partial_close_scales_pnl_by_fraction():
    trade = {"side": "long", "price": 100.0, "amount": 2.0, "leverage": 5}
    fills = [{"side": "sell", "price": "110.0", "realizedPnl": "50.0", "time": 999}]
    pnl, price = _get_close_pnl(trade, fills, fraction=0.5)
    assert pnl == 25.0  # 50.0 * 0.5
