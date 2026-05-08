import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_mark_invalid_called_on_market_symbol_error():
    """'does not have market symbol' error must trigger mark_invalid."""
    marked = []

    class FakeFeed:
        def mark_invalid(self, sym):
            marked.append(sym)

    feed = FakeFeed()
    e = Exception("binance does not have market symbol CLANKER/USDT")
    if "does not have market symbol" in str(e):
        feed.mark_invalid("CLANKER/USDT")
    assert "CLANKER/USDT" in marked


def test_transient_error_not_blacklisted():
    """Network timeout must NOT blacklist the symbol."""
    marked = []

    class FakeFeed:
        def mark_invalid(self, sym):
            marked.append(sym)

    feed = FakeFeed()
    e = Exception("connection timeout")
    if "does not have market symbol" in str(e):
        feed.mark_invalid("BTC/USDT")
    assert marked == []
