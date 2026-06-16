"""
Tests for the TechnicalAgent signal-quality upgrades (2026-06-15):

  1. Multi-TF confluence — 1h signal gated by the 4h trend bias.
  2. Closed-candle reads  — the forming (last) bar never moves the signal.
  3. Adaptive thresholds  — momentum/MACD significance scale with ATR%.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bot"))

from engine.technical_agent import TechnicalAgent


def _df(closes, vol=1000.0, wick=0.002):
    """Build an OHLCV frame from a close series. `wick` sets the high/low
    spread (drives ATR) independently of the close path."""
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="1h")
    return pd.DataFrame({
        "open":   closes,
        "high":   closes * (1 + wick),
        "low":    closes * (1 - wick),
        "close":  closes,
        "volume": np.full(len(closes), vol),
    }, index=idx)


def _uptrend(n=120, start=100.0, step=0.4):
    return _df([start + i * step for i in range(n)])


def _downtrend(n=120, start=100.0, step=0.4):
    return _df([start - i * step for i in range(n)])


def _flat(n=120, level=100.0, noise=0.05):
    rng = np.random.default_rng(0)
    return _df(level + rng.normal(0, noise, n))


AGENT = TechnicalAgent()


def test_uptrend_is_buy():
    sig = AGENT.analyze(_uptrend(), profile=None)
    assert sig.net_score > 0
    assert sig.buy_score > sig.sell_score


def test_confluence_aligned_keeps_full_weight():
    df = _uptrend()
    htf_up = _uptrend(n=120, step=0.6)
    aligned = AGENT.analyze(df, None, market_ctx={"regime": "RANGING"}, htf_df=htf_up)
    flat_htf = AGENT.analyze(df, None, market_ctx={"regime": "RANGING"}, htf_df=_flat())
    # 4h agreement preserves more score than a directionless 4h.
    assert aligned.net_score >= flat_htf.net_score


def test_confluence_against_damps_signal():
    """A bullish 1h fighting a bearish 4h is damped vs the same 1h with no HTF."""
    df = _uptrend()
    no_htf = AGENT.analyze(df, None, market_ctx={"regime": "RANGING"}, htf_df=None)
    against = AGENT.analyze(df, None, market_ctx={"regime": "RANGING"},
                            htf_df=_downtrend(n=120, step=0.6))
    assert against.net_score < no_htf.net_score
    # Counter-HTF damp is ~0.5×, not a full veto — direction survives.
    assert against.net_score > 0
    assert "4h:down" in against.reasoning


def test_closed_candle_ignores_forming_bar():
    """A wild forming (last) candle must not change the signal."""
    df = _uptrend()
    base = AGENT.analyze(df, None)
    spiked = df.copy()
    spiked.iloc[-1, spiked.columns.get_loc("close")] = df["close"].iloc[-1] * 0.5
    spiked.iloc[-1, spiked.columns.get_loc("low")]   = df["close"].iloc[-1] * 0.5
    after = AGENT.analyze(spiked, None)
    assert base.net_score == after.net_score
    assert base.reasoning == after.reasoning


def test_adaptive_momentum_threshold_scales_with_volatility():
    """The SAME 0.8% move triggers momentum on a calm coin but not a volatile
    one — identical close path, ATR differs only via the wick width."""
    n = 120
    # Flat, then a clean +0.8% drift over the last 5 bars (ROC ≈ 0.008).
    closes = [100.0] * (n - 5) + [100.2, 100.4, 100.6, 100.8, 101.0]
    calm_sig  = AGENT.analyze(_df(closes, wick=0.002), None).reasoning   # ATR% ≈ 0.4%
    volat_sig = AGENT.analyze(_df(closes, wick=0.05),  None).reasoning   # ATR% ≈ 10%
    # Calm coin: 0.8% drift clears its low adaptive threshold (floor 0.4%).
    assert "mom+" in calm_sig
    # Volatile coin: the same 0.8% drift is below its high threshold (~5%).
    assert "mom+" not in volat_sig


def test_insufficient_data_is_neutral():
    sig = AGENT.analyze(_uptrend(n=10), None)
    assert sig.net_score == 0
    assert "insufficient" in sig.reasoning
