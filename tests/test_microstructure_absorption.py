"""
Tests for MicrostructureAgent.check_cvd_absorption and its veto integration.

TDD approach — tests written before implementation.
"""

import numpy as np
import pandas as pd
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot.agents.microstructure import MicrostructureAgent, MicrostructureSignal


# ─── helpers ────────────────────────────────────────────────────────────────

def make_df(closes, opens=None, volumes=None, n=20):
    """
    Build a minimal OHLCV DataFrame with the requested closing prices.
    Pads the front with flat candles so the window is always at least `n` rows.
    """
    if opens is None:
        opens = closes  # neutral (no delta)
    if volumes is None:
        volumes = [100.0] * len(closes)

    # Pad with flat candles at the front
    pad = max(0, n - len(closes))
    pad_price = closes[0]
    all_closes  = [pad_price] * pad + list(closes)
    all_opens   = [pad_price] * pad + list(opens)
    all_volumes = [100.0]    * pad + list(volumes)

    return pd.DataFrame({
        "open":   all_opens,
        "high":   all_closes,
        "low":    all_closes,
        "close":  all_closes,
        "volume": all_volumes,
    })


# ─── check_cvd_absorption — LONG side ───────────────────────────────────────

class TestCvdAbsorptionLong:
    """
    LONG absorption = price falling + CVD holding / rising  →  buyers absorbing sellers.
    """

    def test_lower_price_higher_cvd_low_confirms_absorption(self):
        """Classic bullish divergence: price lower low, CVD higher low."""
        agent = MicrostructureAgent()

        # Five 5m candles: price drops, but each candle has net buying (close > open)
        # so CVD keeps ticking up even as price moves lower overall.
        closes  = [100, 99, 98, 97, 96]      # price: lower lows
        opens   = [101, 100, 99.5, 98.5, 97.5]  # open > close  → bearish body (price falls)
        # Wait — close > open means bullish candle (green). We want price falling
        # but CVD rising. Price falling means close_i < close_{i-1}, but individual
        # candles can still close above their open (buyers step in).
        #
        # Let's use: open < close (green) → +volume in CVD, but overall close trail down.
        closes  = [100, 99.5, 99, 98.5, 98]   # price: lower lows
        opens   = [99,  99,   98, 98,   97.5]  # all green candles → CVD rising
        volumes = [200, 200, 200, 200, 200]

        df_5m = make_df(closes, opens, volumes)
        result = agent.check_cvd_absorption(df_5m, side="LONG")

        assert result is True, (
            "Expected absorption confirmed: price falling but CVD rising (green candles)"
        )

    def test_price_drops_cvd_drops_aggressively_no_absorption(self):
        """Genuine dump: price down AND CVD down — no absorption."""
        agent = MicrostructureAgent()

        closes  = [100, 99, 98, 97, 96]
        opens   = [100.5, 99.5, 98.5, 97.5, 96.5]  # open > close → red candles → CVD falls
        volumes = [300, 300, 300, 300, 300]

        df_5m = make_df(closes, opens, volumes)
        result = agent.check_cvd_absorption(df_5m, side="LONG")

        assert result is False, (
            "Expected no absorption: price falling AND CVD falling aggressively"
        )

    def test_price_drops_cvd_slope_flattens_confirms_absorption(self):
        """Slope-flattening: CVD was dropping but recent candles go neutral → absorption starting."""
        agent = MicrostructureAgent()

        # Earlier part of the window: selling (red candles, CVD dropping)
        # Last 3 candles: doji-like (open == close) → CVD flat
        early_closes = [100, 99, 98]
        early_opens  = [100.5, 99.5, 98.5]   # red
        late_closes  = [97.8, 97.5, 97.2]
        late_opens   = [97.8, 97.5, 97.2]    # doji — CVD flat

        closes  = early_closes + late_closes
        opens   = early_opens  + late_opens
        volumes = [200] * 6

        df_5m = make_df(closes, opens, volumes)
        result = agent.check_cvd_absorption(df_5m, side="LONG")

        assert result is True, (
            "Expected absorption: CVD slope flattened while price kept falling"
        )

    def test_insufficient_data_returns_false(self):
        """Too few rows → cannot determine absorption; default to False (safe)."""
        agent = MicrostructureAgent()

        df_5m = pd.DataFrame({
            "open": [100, 99], "high": [101, 100],
            "low":  [99, 98],  "close": [100, 99], "volume": [100, 100],
        })
        result = agent.check_cvd_absorption(df_5m, side="LONG")

        assert result is False


# ─── check_cvd_absorption — SHORT side ──────────────────────────────────────

class TestCvdAbsorptionShort:
    """
    SHORT absorption = price rising + CVD holding / falling → sellers absorbing buyers.
    """

    def test_higher_price_lower_cvd_high_confirms_absorption(self):
        """Classic bearish divergence: price higher high, CVD lower high."""
        agent = MicrostructureAgent()

        closes  = [96, 96.5, 97, 97.5, 98]    # price: higher highs
        opens   = [96.5, 97, 97.5, 98, 98.5]  # open > close → red candles → CVD falling
        volumes = [200] * 5

        df_5m = make_df(closes, opens, volumes)
        result = agent.check_cvd_absorption(df_5m, side="SHORT")

        assert result is True, (
            "Expected absorption: price rising but CVD falling (sellers absorbing buyers)"
        )

    def test_price_rises_cvd_rises_aggressively_no_absorption(self):
        """Genuine pump: price up AND CVD up → no absorption."""
        agent = MicrostructureAgent()

        closes  = [96, 96.5, 97, 97.5, 98]
        opens   = [95.5, 96, 96.5, 97, 97.5]  # all green → CVD rising
        volumes = [300] * 5

        df_5m = make_df(closes, opens, volumes)
        result = agent.check_cvd_absorption(df_5m, side="SHORT")

        assert result is False, (
            "Expected no absorption: price rising AND CVD rising aggressively"
        )


# ─── analyze() integration — absorption veto ────────────────────────────────

class FakeExchange:
    """Minimal stub — returns balanced book so OB doesn't interfere."""
    def fetch_order_book(self, symbol, limit=20):
        return {
            "bids": [["100", "500"]] * limit,
            "asks": [["100", "500"]] * limit,
        }


class TestAnalyzeAbsorptionVeto:

    def _make_long_setup(self, absorption: bool) -> pd.DataFrame:
        """
        Return a 5m DataFrame designed to produce the desired absorption outcome.

        absorption=True  → green candles (close > open) with falling price
        absorption=False → red candles (open > close) with falling price
        """
        if absorption:
            closes = [100, 99.5, 99, 98.5, 98]
            opens  = [99,  99,   98, 98,   97.5]
        else:
            closes = [100, 99, 98, 97, 96]
            opens  = [100.5, 99.5, 98.5, 97.5, 96.5]
        return make_df(closes, opens, [200] * 5, n=20)

    def test_analyze_includes_absorption_key(self):
        """analyze() dict must contain 'absorption_confirmed' key."""
        agent  = MicrostructureAgent()
        df_5m  = self._make_long_setup(absorption=True)
        result = agent.analyze(
            exchange=FakeExchange(),
            symbol="BTCUSDT",
            df=df_5m,
            df_5m=df_5m,
            action="BUY",
        )
        assert hasattr(result, "absorption_confirmed"), (
            "MicrostructureSignal must have an 'absorption_confirmed' field"
        )

    def test_long_no_absorption_with_book_against_triggers_kill(self):
        """
        LONG, absorption_confirmed=False AND the order book leaning against the
        trade = two concurring contra reads → hard-kill.
        """
        agent = MicrostructureAgent()
        df_5m = self._make_long_setup(absorption=False)
        result = agent.analyze(
            exchange=FakeBookExchange(0.6),   # ask-heavy book, against the long
            symbol="BTCUSDT",
            df=df_5m,
            df_5m=df_5m,
            action="BUY",
        )
        assert result.kill is True, (
            "Expected kill=True when LONG has no absorption AND book is against"
        )
        assert result.confirmed is False

    def test_long_no_absorption_neutral_book_soft_shrinks(self):
        """
        LONG, absorption_confirmed=False but the book is NEUTRAL — a LONE
        absorption-fail must shrink, not kill. Shadow data showed lone
        absorption-fail kills threw away ~70%-win weak-trend longs.
        """
        agent = MicrostructureAgent()
        df_5m = self._make_long_setup(absorption=False)
        result = agent.analyze(
            exchange=FakeExchange(),          # balanced book → not against
            symbol="BTCUSDT",
            df=df_5m,
            df_5m=df_5m,
            action="BUY",
        )
        assert result.kill is False
        assert result.size_mult < 1.0
        assert "no absorption" in result.reasoning.lower()

    def test_long_with_absorption_does_not_kill_on_absorption_grounds(self):
        """
        If absorption IS confirmed, the absorption check must NOT contribute a kill.
        (Other checks may still kill; we verify the absorption check is neutral.)
        """
        agent = MicrostructureAgent()
        df_5m = self._make_long_setup(absorption=True)
        result = agent.analyze(
            exchange=FakeExchange(),
            symbol="BTCUSDT",
            df=df_5m,
            df_5m=df_5m,
            action="BUY",
        )
        # Absorption confirmed → reasoning must NOT contain the absorption-kill phrase
        assert "no absorption" not in result.reasoning.lower(), (
            "Absorption kill message should not appear when absorption IS confirmed"
        )

    def test_short_no_absorption_with_book_against_triggers_kill(self):
        """
        SHORT, absorption_confirmed=False (genuine pump) AND a bid-heavy book
        leaning against the short = two concurring contra reads → hard-kill.
        """
        agent = MicrostructureAgent()
        # Genuine pump for a short setup: price rising + CVD rising
        closes = [96, 96.5, 97, 97.5, 98]
        opens  = [95.5, 96, 96.5, 97, 97.5]
        df_5m  = make_df(closes, opens, [300] * 5, n=20)

        result = agent.analyze(
            exchange=FakeBookExchange(1.7),   # bid-heavy book, against the short
            symbol="BTCUSDT",
            df=df_5m,
            df_5m=df_5m,
            action="SELL",
        )
        assert result.kill is True, (
            "Expected kill=True when SHORT has no absorption AND book is against"
        )


# ─── analyze() — soft-kill conversion (lone signal shrinks, combo kills) ─────

class FakeBookExchange:
    """Order book stub with a configurable bid/ask volume ratio."""
    def __init__(self, ratio: float):
        self.ratio = ratio

    def fetch_order_book(self, symbol, limit=20):
        return {
            "bids": [[100.0, 100.0 * self.ratio]] * limit,
            "asks": [[100.0, 100.0]] * limit,
        }


def rising_green_df():
    """Price rising on green bodies → CVD bullish, no divergence (BUY-friendly)."""
    closes = [100, 101, 102, 103, 104]
    opens  = [99.5, 100.5, 101.5, 102.5, 103.5]
    return make_df(closes, opens, [200] * 5, n=20)


def falling_red_df():
    """Price falling on red bodies → CVD bearish, no BUY divergence."""
    closes = [104, 103, 102, 101, 100]
    opens  = [104.5, 103.5, 102.5, 101.5, 100.5]
    return make_df(closes, opens, [200] * 5, n=20)


def rising_red_df():
    """Price rising while bodies are red → CVD falling = BUY divergence."""
    closes = [100, 101, 102, 103, 104]
    opens  = [100.6, 101.6, 102.6, 103.6, 104.6]
    return make_df(closes, opens, [200] * 5, n=20)


def absorbing_long_df():
    """Green bodies on falling price → absorption confirmed for LONG."""
    closes = [100, 99.5, 99, 98.5, 98]
    opens  = [99, 99, 98, 98, 97.5]
    return make_df(closes, opens, [200] * 5, n=20)


class TestSoftKillConversion:

    def test_lone_ask_wall_with_cvd_confirming_soft_shrinks_long(self):
        # ob 0.45: heavy ask pressure, but CVD bullish → ×0.7, no kill
        r = MicrostructureAgent().analyze(
            exchange=FakeBookExchange(0.45), symbol="X", df=rising_green_df(),
            df_5m=rising_green_df(), action="BUY")
        assert r.kill is False
        assert r.size_mult == pytest.approx(0.7)

    def test_overwhelming_ask_wall_with_bullish_cvd_defers_to_absorption(self):
        # ob 0.30 = squeeze-grade wall, BUT CVD bullish (buyers lifting it) →
        # absorption setup, not a squeeze. Shrinks instead of hard-killing; the
        # 5m absorption check (rising price → confirmed) lets the long through.
        # Shadow data: these longs won ~68%.
        r = MicrostructureAgent().analyze(
            exchange=FakeBookExchange(0.30), symbol="X", df=rising_green_df(),
            df_5m=rising_green_df(), action="BUY")
        assert r.kill is False
        assert r.size_mult == pytest.approx(0.7)

    def test_overwhelming_ask_wall_with_cvd_not_confirming_hard_kills_long(self):
        # ob 0.30 wall + CVD NOT bullish → genuine squeeze evidence, still kills.
        r = MicrostructureAgent().analyze(
            exchange=FakeBookExchange(0.30), symbol="X", df=falling_red_df(),
            df_5m=absorbing_long_df(), action="BUY")
        assert r.kill is True and r.size_mult == 0.0

    def test_ask_wall_plus_cvd_not_confirming_hard_kills_long(self):
        # combo: heavy ask AND CVD bearish (df bearish, absorption df keeps
        # the 5m gate out of the way) → kill
        r = MicrostructureAgent().analyze(
            exchange=FakeBookExchange(0.45), symbol="X", df=falling_red_df(),
            df_5m=absorbing_long_df(), action="BUY")
        assert r.kill is True

    def test_lone_bid_wall_with_cvd_confirming_soft_shrinks_short(self):
        # Symmetry fix: SELL into a 2.2x bid wall with CVD bearish used to
        # pass at full size — a standing contra wall is worth one reduction.
        r = MicrostructureAgent().analyze(
            exchange=FakeBookExchange(2.2), symbol="X", df=falling_red_df(),
            df_5m=falling_red_df(), action="SELL")
        assert r.kill is False
        assert r.size_mult == pytest.approx(0.7)

    def test_lone_divergence_soft_shrinks(self):
        # BUY divergence with a neutral book → ×0.75, no kill
        r = MicrostructureAgent().analyze(
            exchange=FakeBookExchange(1.0), symbol="X", df=rising_red_df(),
            df_5m=rising_red_df(), action="BUY")
        assert r.cvd_divergence is True
        assert r.kill is False
        assert r.size_mult == pytest.approx(0.75)

    def test_divergence_plus_ob_against_hard_kills(self):
        # Two independent contra reads (divergence + book leaning against,
        # 0.6 < 1/1.4) → kill
        r = MicrostructureAgent().analyze(
            exchange=FakeBookExchange(0.6), symbol="X", df=rising_red_df(),
            df_5m=rising_red_df(), action="BUY")
        assert r.cvd_divergence is True
        assert r.kill is True

    def test_clean_confirmation_full_size(self):
        r = MicrostructureAgent().analyze(
            exchange=FakeBookExchange(1.6), symbol="X", df=rising_green_df(),
            df_5m=rising_green_df(), action="BUY")
        assert r.kill is False
        assert r.size_mult == 1.0


# ─── displacement gate — noise must NOT veto, genuine moves must ─────────────
#
# Root-cause regression guards: the old detector vetoed on any directional 5m
# drift (slope > 0.15), so ordinary chop killed both sides and halted the bot.
# The redesign vetoes only on a significant adverse move that CVD confirms.

def noisy_down_long_df():
    """Tiny net drift down on red bodies — chop, not a dump. Must NOT veto a LONG.

    Net move is a fraction of the typical bar-to-bar move, so the displacement
    gate keeps it below the veto floor even though CVD leans bearish.
    """
    closes = [100.0, 100.3, 99.8, 100.1, 99.95]   # net ≈ -0.05%, bars ≈ ±0.3%
    opens  = [100.2, 100.5, 100.0, 100.3, 100.1]  # red bodies → CVD bearish
    return make_df(closes, opens, [200] * 5, n=20)


def noisy_up_short_df():
    """Tiny net drift up on green bodies — chop, not a pump. Must NOT veto a SHORT."""
    closes = [100.0, 99.7, 100.2, 99.9, 100.05]
    opens  = [99.8, 99.5, 100.0, 99.7, 99.9]      # green bodies → CVD bullish
    return make_df(closes, opens, [200] * 5, n=20)


class TestAbsorptionDisplacementGate:

    def test_noisy_down_does_not_veto_long(self):
        agent = MicrostructureAgent()
        assert agent.check_cvd_absorption(noisy_down_long_df(), "LONG") is True

    def test_noisy_up_does_not_veto_short(self):
        agent = MicrostructureAgent()
        assert agent.check_cvd_absorption(noisy_up_short_df(), "SHORT") is True

    def test_genuine_dump_still_vetoes_long(self):
        """A real sustained dump (price down ~4%, CVD strongly down) still kills."""
        agent = MicrostructureAgent()
        closes = [100, 99, 98, 97, 96]
        opens  = [100.5, 99.5, 98.5, 97.5, 96.5]   # red
        assert agent.check_cvd_absorption(make_df(closes, opens, [300]*5), "LONG") is False

    def test_genuine_pump_still_vetoes_short(self):
        agent = MicrostructureAgent()
        closes = [96, 97, 98, 99, 100]
        opens  = [95.5, 96.5, 97.5, 98.5, 99.5]    # green
        assert agent.check_cvd_absorption(make_df(closes, opens, [300]*5), "SHORT") is False

    def test_threshold_is_config_overridable(self):
        """move_mult/cvd_strong_threshold come from config; a strict config can
        still veto the noisy case, proving the knobs are wired through."""
        strict = MicrostructureAgent(
            {"absorption": {"cvd_strong_threshold": 0.05, "move_mult": 0.0}})
        assert strict.absorption_move_mult == 0.0
        assert strict.cvd_strong_threshold == 0.05
        # With the floor at 0 and a near-zero CVD bar, the noisy down move now
        # trips the veto — confirming both knobs feed the decision.
        assert strict.check_cvd_absorption(noisy_down_long_df(), "LONG") is False
