"""
Tests that _resolve_close_pnl always uses _calc_pnl (not exchange realizedPnl).
Regression for demo-exchange bug: exchange realizedPnl omits leverage multiplier.
"""
import sys, os, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import MagicMock, patch


TRADE_LONG = {
    "id": "101880004",
    "symbol": "AKT/USDT",
    "side": "long",
    "amount": 121.1,
    "price": 0.8493834779017,
    "strategy": "ensemble:+0.450",
    "status": "open",
    "mode": "futures",
    "leverage": 5,
}

TRADE_LONG_FULL = {
    "id": "fill_full",
    "symbol": "BNB/USDT",
    "side": "long",
    "amount": 242.1,
    "price": 0.8430,
    "strategy": "ensemble:+0.450",
    "status": "open",
    "mode": "futures",
    "leverage": 5,
}


def _make_bot_stub():
    """Minimal BaseBot stub with a real _resolve_close_pnl and a stub _calc_pnl."""
    from engine.bot import BaseBot
    bot = BaseBot.__new__(BaseBot)
    bot.log = MagicMock()
    bot.exchange = MagicMock()
    return bot


def test_resolve_close_pnl_ignores_exchange_rpnl_uses_calc_pnl():
    """
    When the exchange fill has a (wrong) realizedPnl, _resolve_close_pnl must NOT
    use it. It must use _calc_pnl with the actual fill price instead.
    """
    bot = _make_bot_stub()

    # Exchange fill: tiny realizedPnl (like demo exchange returns without leverage)
    bot.exchange.fetch_my_trades = MagicMock(return_value=[{
        "side": "sell",
        "price": 0.8669,
        "realizedPnl": "0.133",   # wrong (no leverage)
        "time": 1000,
    }])

    expected_pnl = (0.8669 - 0.8493834779017) * 121.1 * 5   # leveraged, no fee for stub
    with patch.object(bot, '_calc_pnl', return_value=expected_pnl) as mock_calc:
        pnl, price = bot._resolve_close_pnl(TRADE_LONG, "AKT/USDT", 0.8650, fraction=1.0)

    mock_calc.assert_called_once_with(TRADE_LONG, 0.8669)   # actual fill price used
    assert abs(pnl - expected_pnl) < 1e-9
    assert price == 0.8669


def test_resolve_close_pnl_partial_no_double_fraction():
    """
    For a 50% partial close: pnl = _calc_pnl(full_trade, fill_price) * 0.5
    Exchange realizedPnl (already for the 50%) must NOT be used to avoid ×2 error.
    """
    bot = _make_bot_stub()

    tp1_price = 0.867
    bot.exchange.fetch_my_trades = MagicMock(return_value=[{
        "side": "sell",
        "price": tp1_price,
        "realizedPnl": "0.030",   # exchange already reports for the 50%
        "time": 1000,
    }])

    full_calc_pnl = (tp1_price - 0.8430) * 242.1 * 5
    with patch.object(bot, '_calc_pnl', return_value=full_calc_pnl) as mock_calc:
        pnl, price = bot._resolve_close_pnl(TRADE_LONG_FULL, "BNB/USDT", tp1_price, fraction=0.5)

    mock_calc.assert_called_once_with(TRADE_LONG_FULL, tp1_price)
    assert abs(pnl - full_calc_pnl * 0.5) < 1e-9
    assert price == tp1_price


def test_resolve_close_pnl_falls_back_to_detection_price_on_error():
    """When fetch_my_trades throws, uses detection_price with _calc_pnl."""
    bot = _make_bot_stub()
    bot.exchange.fetch_my_trades = MagicMock(side_effect=Exception("network error"))

    detection = 0.8650
    stub_pnl = 5.0
    with patch.object(bot, '_calc_pnl', return_value=stub_pnl) as mock_calc:
        pnl, price = bot._resolve_close_pnl(TRADE_LONG, "AKT/USDT", detection, fraction=1.0)

    mock_calc.assert_called_once_with(TRADE_LONG, detection)
    assert abs(pnl - stub_pnl) < 1e-9
    assert price == detection


def test_futures_calc_pnl_includes_leverage():
    """_calc_pnl (FuturesBot) correctly multiplies by leverage."""
    from engine.futures import FuturesBot
    bot = FuturesBot.__new__(FuturesBot)
    bot.leverage = 5
    bot.config = {}

    trade = {"side": "long", "price": 0.8430, "amount": 121.1, "leverage": 5}
    close_price = 0.8669
    pnl = bot._calc_pnl(trade, close_price)

    expected_unleveraged = (close_price - 0.8430) * 121.1
    # pnl must be significantly larger than unleveraged (leverage multiplier applied)
    assert pnl > expected_unleveraged * 1.5, f"pnl={pnl} should be leveraged vs raw={expected_unleveraged}"
    # fee deducted: pnl = raw * lev - fee
    fee = 0.8430 * 121.1 * 0.0004 * 2
    assert abs(pnl - (expected_unleveraged * 5 - fee)) < 0.001
