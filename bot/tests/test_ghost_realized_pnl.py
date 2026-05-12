# bot/tests/test_ghost_realized_pnl.py


def _ghost_close_pnl(trade, closing_fill, fallback_price, calc_pnl_fn):
    """Simulate ghost-close PnL selection logic — always uses calc_pnl."""
    close_price = float(closing_fill.get("price", fallback_price)) if closing_fill else fallback_price
    return calc_pnl_fn(trade, close_price), close_price


def test_ghost_uses_calc_pnl_with_fill_price():
    """Ghost closes always use _calc_pnl with the fill price (exchange realizedPnl ignored)."""
    trade = {"side": "long", "price": 100.0, "amount": 1.0, "leverage": 5}
    fill  = {"price": "108.0", "realizedPnl": "37.9", "side": "sell", "time": 1}
    pnl, _ = _ghost_close_pnl(trade, fill, 100.0,
                               lambda t, p: (p - t["price"]) * t["amount"] * t["leverage"])
    assert pnl == 40.0   # (108-100)*1*5, NOT the exchange value 37.9


def test_ghost_uses_calc_pnl_when_realized_pnl_absent():
    trade = {"side": "long", "price": 100.0, "amount": 1.0, "leverage": 5}
    fill  = {"price": "108.0", "side": "sell", "time": 1}  # no realizedPnl key
    pnl, _ = _ghost_close_pnl(trade, fill, 100.0,
                               lambda t, p: (p - t["price"]) * t["amount"] * t["leverage"])
    assert pnl == 40.0  # (108-100)*1*5


def test_ghost_break_even_uses_calc_pnl():
    """Even when exchange says realizedPnl=0, _calc_pnl determines the value."""
    trade = {"side": "long", "price": 100.0, "amount": 1.0, "leverage": 5}
    fill  = {"price": "100.0", "realizedPnl": "0", "side": "sell", "time": 1}
    pnl, _ = _ghost_close_pnl(trade, fill, 100.0,
                               lambda t, p: (p - t["price"]) * t["amount"] * t["leverage"])
    assert pnl == 0.0  # _calc_pnl: (100-100)*1*5 = 0.0 (same result here)


def test_ghost_falls_back_to_calc_when_no_fill():
    trade = {"side": "long", "price": 100.0, "amount": 1.0, "leverage": 5}
    pnl, _ = _ghost_close_pnl(trade, None, 102.0,
                               lambda t, p: (p - t["price"]) * t["amount"] * t["leverage"])
    assert pnl == 10.0  # (102-100)*1*5


def test_cleanup_ghost_uses_calc_pnl_not_exchange_rpnl():
    """Real _cleanup_ghost_trades must use _calc_pnl, not exchange realizedPnl."""
    import sys
    from pathlib import Path
    from datetime import datetime, timezone, timedelta

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from engine.futures import FuturesBot

    class FakeExchange:
        def fetch_my_trades(self, symbol, limit=20):
            return [{"side": "sell", "price": "108.0", "realizedPnl": "37.9", "time": 999}]
        def fetch_ticker(self, symbol):
            return {"last": 999.0}

    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=120)).isoformat()
    trade = {
        "id": "ghost1", "symbol": "FOO/USDT", "status": "open",
        "timestamp": old_ts, "price": 100.0, "side": "long",
        "amount": 1.0, "leverage": 5, "close_price": 0.0,
        "pnl": 0.0, "close_timestamp": "", "sl_order_id": "",
    }
    d = {
        "trades": [trade],
        "stats": {"total_pnl": 0.0, "wins": 0, "losses": 0},
    }

    bot = object.__new__(FuturesBot)
    bot.exchange = FakeExchange()
    bot.leverage = 5
    bot.config = {}
    bot.log = type("L", (), {
        "debug": lambda self, *a, **k: None,
        "info":  lambda self, *a, **k: None,
        "warning": lambda self, *a, **k: None,
    })()
    bot.rl_agent = type("R", (), {"record_external_close": lambda self, *a, **k: None})()
    bot.ai = type("A", (), {"record_trade_result": lambda self, *a, **k: None})()
    bot.agents = type("AG", (), {"record_trade_result": lambda self, *a, **k: None})()
    bot._cancel_exchange_stop_loss = lambda sym, sl_id: None

    bot._cleanup_ghost_trades(set(), d)  # FOO/USDT not in exchange_syms

    assert trade["status"] == "closed"
    assert trade["close_price"] == 108.0
    # _calc_pnl: (108-100)*1*5 - fee = 40 - fee ≈ 39.92
    expected = bot._calc_pnl(trade, 108.0)
    assert abs(trade["pnl"] - expected) < 0.01
    assert trade["pnl"] != 37.9   # must NOT use the exchange's wrong realizedPnl
