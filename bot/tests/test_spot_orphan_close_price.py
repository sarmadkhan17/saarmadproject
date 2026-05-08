# bot/tests/test_spot_orphan_close_price.py

def _run_spot_orphan_logic(fills, entry_price):
    """Simulate the orphan-close price logic."""
    closing_side = "sell"
    close_price = entry_price
    closing_fills = [f for f in fills if f.get("side", "").lower() == closing_side]
    if closing_fills:
        best = max(closing_fills, key=lambda x: x["time"])
        close_price = float(best.get("price", close_price))
    return close_price


def test_spot_orphan_uses_sell_fill_not_latest():
    """When fills contain both buy and sell, use the sell fill price."""
    fills = [
        {"side": "buy",  "price": 100.0, "time": 1000},  # newest — opening fill
        {"side": "sell", "price": 95.0,  "time": 900},   # closing fill
    ]
    assert _run_spot_orphan_logic(fills, entry_price=100.0) == 95.0


def test_spot_orphan_falls_back_to_entry_when_no_sell():
    """No closing-side fill → keep entry price as fallback."""
    fills = [{"side": "buy", "price": 110.0, "time": 1000}]
    assert _run_spot_orphan_logic(fills, entry_price=100.0) == 100.0
