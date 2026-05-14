"""
Test that ExecutionEngine aborts a futures entry (closes position, returns None)
when _place_sl() fails to set a stop-loss order.
"""
import sys, os, threading
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import MagicMock, patch


def _make_engine(mode='futures'):
    from engine.execution_engine import ExecutionEngine
    engine = ExecutionEngine.__new__(ExecutionEngine)
    engine.exchange = MagicMock()
    engine.state = MagicMock()
    engine.notifier = MagicMock()
    engine.mode = mode
    engine._get_leverage = lambda: 5
    engine._sym_locks = {}
    engine._lock_registry = threading.Lock()
    return engine


def _make_decision(amount=242.0, est_usdt=204.0, conf=0.65):
    d = MagicMock()
    d.position_size = amount
    d.est_usdt = est_usdt
    d.adjusted_conf = conf
    d.profile = 'AGGRESSIVE'
    d.htf_bias = 'NEUTRAL'
    d.hmm_regime = 'RANGING'
    return d


def test_futures_entry_aborted_when_sl_fails():
    """
    When _place_sl returns '' (SL rejected), _execute_entry_locked must:
    - NOT persist the trade (state.add_trade never called)
    - Place an emergency market-close order
    - Return None
    """
    engine = _make_engine('futures')
    decision = _make_decision()

    filled_order = {'id': 'fill123', 'average': 0.84, 'price': 0.84}
    place_buy_fn  = MagicMock(return_value=filled_order)
    place_sell_fn = MagicMock()
    get_atr_fn    = MagicMock(return_value=0.012)

    with patch.object(engine, '_place_sl', return_value=''):
        result = engine._execute_entry_locked(
            decision, 'AKT/USDT', 'BUY', 0.84,
            get_atr_fn=get_atr_fn,
            place_buy_fn=place_buy_fn,
            place_sell_fn=place_sell_fn,
            strat='ensemble:+0.450',
        )

    assert result is None, "_execute_entry_locked must return None when SL fails"
    engine.state.add_trade.assert_not_called()
    # An emergency closing sell must have been issued
    engine.exchange.create_market_order.assert_called_once()
    call_args = engine.exchange.create_market_order.call_args[0]
    assert call_args[0] == 'AKT/USDT'
    assert call_args[1] == 'sell'


def test_futures_entry_succeeds_when_sl_ok():
    """Normal path: fill + SL both succeed → trade persisted, Trade object returned."""
    from engine.execution_engine import ExecutionEngine
    from core.types import Trade

    engine = _make_engine('futures')
    decision = _make_decision()

    filled_order = {'id': 'fill123', 'average': 0.84, 'price': 0.84}
    place_buy_fn  = MagicMock(return_value=filled_order)
    place_sell_fn = MagicMock()
    get_atr_fn    = MagicMock(return_value=0.012)

    with patch.object(engine, '_place_sl', return_value='sl_order_999'):
        result = engine._execute_entry_locked(
            decision, 'AKT/USDT', 'BUY', 0.84,
            get_atr_fn=get_atr_fn,
            place_buy_fn=place_buy_fn,
            place_sell_fn=place_sell_fn,
            strat='ensemble:+0.450',
        )

    assert result is not None, "Must return a Trade when entry + SL both succeed"
    assert isinstance(result, Trade)
    engine.state.add_trade.assert_called_once()


def test_spot_entry_not_aborted_when_sl_skipped():
    """Spot mode never places SL — no abort should occur."""
    from engine.execution_engine import ExecutionEngine
    from core.types import Trade

    engine = _make_engine('spot')
    decision = _make_decision()

    filled_order = {'id': 'fill123', 'average': 0.84, 'price': 0.84}
    place_buy_fn  = MagicMock(return_value=filled_order)
    place_sell_fn = MagicMock()
    get_atr_fn    = MagicMock(return_value=0.012)

    result = engine._execute_entry_locked(
        decision, 'AKT/USDT', 'BUY', 0.84,
        get_atr_fn=get_atr_fn,
        place_buy_fn=place_buy_fn,
        place_sell_fn=place_sell_fn,
        strat='ensemble:+0.450',
    )

    assert result is not None, "Spot entry must succeed even without SL"
    assert isinstance(result, Trade)
    engine.state.add_trade.assert_called_once()
    # No emergency close should have been issued in spot mode
    engine.exchange.create_market_order.assert_not_called()
