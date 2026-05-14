import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
from engine.risk_agent import RiskDecisionAgent


def _make_df(rows=100, close_val=100.0):
    """DataFrame with constant close so 20EMA == close_val."""
    idx = pd.date_range(
        end=datetime.now(timezone.utc),
        periods=rows, freq="1h", tz="UTC"
    )
    close = pd.Series([close_val] * rows, index=idx)
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": [1_000_000.0] * rows,
    })


def _make_agent():
    risk = MagicMock()
    risk.get_position_size.return_value = (0.1, 50.0)
    risk.can_open_trade.return_value = (True, "ok")
    gnn = MagicMock()
    gnn.check.return_value = (True, "ok", 0.3)
    return RiskDecisionAgent(risk, gnn)


def _make_ensemble(action="BUY", confidence=0.70):
    e = MagicMock()
    e.action = action
    e.confidence = confidence
    e.net_score = 0.40 if action == "BUY" else -0.40
    e.buy_score = 0.60 if action == "BUY" else 0.10
    e.sell_score = 0.10 if action == "BUY" else 0.60
    e.agents_agreeing = 3
    e.agents_total = 3
    e.signals = []
    return e


def _make_profile(name="BALANCED"):
    p = MagicMock()
    p.name = name
    p.min_confidence = 0.43
    p.min_agent_agreement = 2
    p.adx_min = 20.0
    p.net_score_threshold = 0.25
    p.htf_filter_mode = "soft"
    p.btc_momentum_filter = False
    p.use_confluence_scoring = False
    p.min_quality_score = 0.30
    return p


def _make_regime(adx=30.0, trend_dir="NEUTRAL"):
    return dict(
        adx=adx, regime="STRONG_TREND", gate=True,
        allow_longs=True, allow_shorts=True,
        vol_ratio=1.2, breadth=0.55, bear_breadth=0.45,
        min_conf=0.43, size_mult=1.0, hmm_regime="STRONG_TREND",
        trend_direction=trend_dir,
    )


def test_buy_blocked_when_price_below_ema():
    """BUY is rejected when live price is 2% below 20EMA."""
    agent = _make_agent()
    ema_val = 100.0
    live_price = 97.5  # 2.5% below → triggers gate (threshold is 1%)
    df = _make_df(close_val=ema_val)

    result = agent.evaluate(
        ensemble=_make_ensemble("BUY"),
        symbol="ETH/USDT",
        df_1h=df,
        profile=_make_profile(),
        regime_ctx=_make_regime(),
        btc_return=0.0,
        open_trades=[],
        balance=1000.0,
        get_price_fn=lambda sym: live_price,
        get_atr_fn=lambda sym: 1.0,
    )
    assert not result.approved
    assert "20EMA" in " ".join(result.reasons)


def test_buy_allowed_when_price_above_ema():
    """BUY is not blocked when live price is above 20EMA."""
    agent = _make_agent()
    ema_val = 100.0
    live_price = 101.5  # above EMA

    result = agent.evaluate(
        ensemble=_make_ensemble("BUY"),
        symbol="ETH/USDT",
        df_1h=_make_df(close_val=ema_val),
        profile=_make_profile(),
        regime_ctx=_make_regime(),
        btc_return=0.0,
        open_trades=[],
        balance=1000.0,
        get_price_fn=lambda sym: live_price,
        get_atr_fn=lambda sym: 1.0,
    )
    # Gate 4c should not fire — check reasons don't contain EMA rejection
    assert "20EMA" not in " ".join(result.reasons)


def test_sell_blocked_when_price_above_ema():
    """SELL is rejected when live price is 2% above 20EMA."""
    agent = _make_agent()
    ema_val = 100.0
    live_price = 102.5  # 2.5% above → triggers gate

    result = agent.evaluate(
        ensemble=_make_ensemble("SELL", confidence=0.70),
        symbol="ETH/USDT",
        df_1h=_make_df(close_val=ema_val),
        profile=_make_profile(),
        regime_ctx=_make_regime(),
        btc_return=0.0,
        open_trades=[],
        balance=1000.0,
        get_price_fn=lambda sym: live_price,
        get_atr_fn=lambda sym: 1.0,
    )
    assert not result.approved
    assert "20EMA" in " ".join(result.reasons)


def test_sell_allowed_when_price_below_ema():
    """SELL is not blocked when price is below 20EMA."""
    agent = _make_agent()
    live_price = 98.0  # below EMA

    result = agent.evaluate(
        ensemble=_make_ensemble("SELL", confidence=0.70),
        symbol="ETH/USDT",
        df_1h=_make_df(close_val=100.0),
        profile=_make_profile(),
        regime_ctx=_make_regime(),
        btc_return=0.0,
        open_trades=[],
        balance=1000.0,
        get_price_fn=lambda sym: live_price,
        get_atr_fn=lambda sym: 1.0,
    )
    assert "20EMA" not in " ".join(result.reasons)


def test_btc_exempt_from_ema_gate():
    """BTC/USDT is exempt from Gate 4c regardless of price vs EMA."""
    agent = _make_agent()
    live_price = 50_000 * 0.97  # 3% below EMA

    result = agent.evaluate(
        ensemble=_make_ensemble("BUY"),
        symbol="BTC/USDT",
        df_1h=_make_df(close_val=50_000.0),
        profile=_make_profile(),
        regime_ctx=_make_regime(),
        btc_return=0.0,
        open_trades=[],
        balance=1000.0,
        get_price_fn=lambda sym: live_price,
        get_atr_fn=lambda sym: 100.0,
    )
    assert "20EMA" not in " ".join(result.reasons)


def test_within_1pct_of_ema_not_blocked():
    """Price within 0.5% of EMA must not trigger the gate."""
    agent = _make_agent()
    live_price = 99.6  # 0.4% below — within 1% band

    result = agent.evaluate(
        ensemble=_make_ensemble("BUY"),
        symbol="ETH/USDT",
        df_1h=_make_df(close_val=100.0),
        profile=_make_profile(),
        regime_ctx=_make_regime(),
        btc_return=0.0,
        open_trades=[],
        balance=1000.0,
        get_price_fn=lambda sym: live_price,
        get_atr_fn=lambda sym: 1.0,
    )
    assert "20EMA" not in " ".join(result.reasons)
