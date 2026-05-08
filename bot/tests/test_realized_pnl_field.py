# bot/tests/test_realized_pnl_field.py
def test_fetch_my_trades_returns_realized_pnl_key():
    """fetch_my_trades response dicts must always contain 'realizedPnl' key."""
    raw = [{"price": "100.0", "qty": "1.0", "quoteQty": "100.0",
            "side": "BUY", "time": 1000, "realizedPnl": "5.5"}]
    result = [
        {
            "symbol":      "BTC/USDT",
            "price":       float(t.get("price", 0)),
            "qty":         float(t.get("qty", 0)),
            "quoteQty":    float(t.get("quoteQty", 0)),
            "side":        t.get("side", "").lower(),
            "time":        t.get("time", 0),
            "realizedPnl": float(t.get("realizedPnl", 0)),
        }
        for t in raw
    ]
    assert "realizedPnl" in result[0]
    assert result[0]["realizedPnl"] == 5.5


def test_fetch_my_trades_defaults_realized_pnl_to_zero():
    """realizedPnl must default to 0 when absent (spot trades don't include it)."""
    raw = [{"price": "200.0", "qty": "0.5", "quoteQty": "100.0",
            "side": "BUY", "time": 2000}]
    result = [{"realizedPnl": float(t.get("realizedPnl", 0))} for t in raw]
    assert result[0]["realizedPnl"] == 0.0
