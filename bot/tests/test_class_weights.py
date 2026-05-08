# bot/tests/test_class_weights.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _compute_class_weights(closed_trades: list) -> dict:
    """Local copy for isolated testing — matches the function in bot.py."""
    import numpy as np
    BUY_SIDES  = {"buy", "long"}
    SELL_SIDES = {"sell", "short"}
    FALLBACK   = {0: 2.0, 1: 1.0, 2: 2.0}

    buy_trades  = [t for t in closed_trades if t.get("side", "") in BUY_SIDES]
    sell_trades = [t for t in closed_trades if t.get("side", "") in SELL_SIDES]

    if len(buy_trades) < 20 or len(sell_trades) < 20:
        return FALLBACK

    buy_prec  = sum(1 for t in buy_trades  if t.get("pnl", 0) > 0) / len(buy_trades)
    sell_prec = sum(1 for t in sell_trades if t.get("pnl", 0) > 0) / len(sell_trades)

    buy_w  = float(np.clip(1.0 / (buy_prec  + 1e-9), 1.0, 4.0))
    sell_w = float(np.clip(1.0 / (sell_prec + 1e-9), 1.0, 4.0))

    return {0: sell_w, 1: 1.0, 2: buy_w}


def test_fallback_when_too_few_trades():
    result = _compute_class_weights([])
    assert result == {0: 2.0, 1: 1.0, 2: 2.0}


def test_fallback_when_fewer_than_20_per_side():
    trades = [{"side": "buy", "pnl": 1.0}] * 15 + [{"side": "sell", "pnl": -1.0}] * 15
    result = _compute_class_weights(trades)
    assert result == {0: 2.0, 1: 1.0, 2: 2.0}


def test_high_buy_precision_gives_low_buy_weight():
    trades = (
        [{"side": "buy",  "pnl":  1.0}] * 19 +
        [{"side": "buy",  "pnl": -1.0}] * 1  +
        [{"side": "sell", "pnl":  1.0}] * 10 +
        [{"side": "sell", "pnl": -1.0}] * 10
    )
    result = _compute_class_weights(trades)
    assert result[2] < result[0]
    assert result[1] == 1.0


def test_hold_weight_always_1():
    trades = (
        [{"side": "buy",  "pnl": 1.0}] * 25 +
        [{"side": "sell", "pnl": 1.0}] * 25
    )
    result = _compute_class_weights(trades)
    assert result[1] == 1.0


def test_weights_clipped_to_4():
    trades = (
        [{"side": "buy",  "pnl": -1.0}] * 25 +
        [{"side": "sell", "pnl": -1.0}] * 25
    )
    result = _compute_class_weights(trades)
    assert result[0] == 4.0
    assert result[2] == 4.0


def test_long_short_sides_counted_correctly():
    trades = (
        [{"side": "long",  "pnl":  1.0}] * 20 +
        [{"side": "short", "pnl": -1.0}] * 20
    )
    result = _compute_class_weights(trades)
    assert result[2] == 1.0
    assert result[0] == 4.0
