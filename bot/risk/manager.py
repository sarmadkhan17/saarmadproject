"""
Risk Manager v2 - Production Grade
- ATR-based dynamic stop loss
- Kelly Criterion position sizing
- Portfolio heat tracking
- Correlation filter
- Circuit breaker
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from pathlib import Path
from datetime import datetime, date, timezone
from typing import List, Tuple, Optional
from core.config import DATA_DIR

log  = logging.getLogger("RiskManager")
DATA = DATA_DIR
BOT_MODE = os.environ.get("BOT_MODE", "spot")


class CorrelationFilter:
    # btc_correlated is a cross-group limit: max 2 of these highly-correlated majors simultaneously
    GROUPS = {
        "btc_correlated": ["BTC/USDT","ETH/USDT","BNB/USDT","SOL/USDT","AVAX/USDT","XRP/USDT"],
        "large_cap":      ["BTC/USDT","ETH/USDT"],
        "layer1":         ["SOL/USDT","ADA/USDT","AVAX/USDT","DOT/USDT"],
        "defi":           ["LINK/USDT","UNI/USDT","AAVE/USDT"],
        "meme":           ["DOGE/USDT","SHIB/USDT"],
        "exchange":       ["BNB/USDT"],
        "payments":       ["XRP/USDT","XLM/USDT"],
        "layer2":         ["MATIC/USDT","OP/USDT","ARB/USDT"],
    }
    MAX_PER_GROUP = {
        "btc_correlated": 2,
        "large_cap": 1, "layer1": 2, "defi": 1,
        "meme": 1, "exchange": 1, "payments": 1, "layer2": 1,
    }

    def is_allowed(self, symbol, open_trades):
        open_symbols = [t["symbol"] for t in open_trades]
        for group, symbols in self.GROUPS.items():
            if symbol not in symbols:
                continue
            held = sum(1 for s in open_symbols if s in symbols)
            if held >= self.MAX_PER_GROUP.get(group, 2):
                return False, f"Already holding {held} from {group} (max {self.MAX_PER_GROUP.get(group,2)})"
        unknown_held = sum(1 for s in open_symbols if not any(s in g for g in self.GROUPS.values()))
        if not any(symbol in g for g in self.GROUPS.values()):
            if unknown_held >= 3:
                return False, f"Already holding {unknown_held} ungrouped tokens (max 3)"
        return True, "OK"


class MarketRegimeGate:
    CACHE_SECS = 900  # 15-minute cache — 4h bars move slowly

    def __init__(self):
        self._cache      = None
        self._cache_time = None

    def detect(self, feed, watchlist: list) -> dict:
        now = datetime.now(timezone.utc)
        if (self._cache and self._cache_time and
                (now - self._cache_time).total_seconds() < self.CACHE_SECS):
            return self._cache
        result = self._compute(feed, watchlist)
        self._cache      = result
        self._cache_time = now
        log.info(
            f"[MarketRegimeGate] {result['regime']} | gate={result['gate']} | "
            f"breadth={result['breadth']:.0%} | vol_ratio={result['vol_ratio']:.2f}x | "
            f"ADX={result['adx']:.1f}"
        )
        return result

    def _compute(self, feed, watchlist: list) -> dict:
        try:
            btc_4h = feed.fetch_ohlcv("BTC/USDT", "4h", limit=100)
        except Exception:
            btc_4h = None
        if btc_4h is None or len(btc_4h) < 50:
            return self._neutral()

        close = btc_4h["close"]
        high  = btc_4h["high"]
        low   = btc_4h["low"]

        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        price = float(close.iloc[-1])

        btc_bullish = price > float(ema20.iloc[-1]) > float(ema50.iloc[-1])
        btc_bearish = price < float(ema20.iloc[-1]) < float(ema50.iloc[-1])

        chg_4h  = (price - float(close.iloc[-2])) / float(close.iloc[-2])
        chg_24h = (price - float(close.iloc[-7])) / float(close.iloc[-7]) if len(close) > 7 else 0

        ret       = close.pct_change().dropna()
        vol_ratio = float(ret.iloc[-8:].std()) / (float(ret.iloc[-40:].std()) + 1e-9)

        adx_val = float(self._calc_adx(high, low, close).iloc[-1])
        if pd.isna(adx_val):
            adx_val = 25.0

        bull, bear, checked = 0, 0, 0
        for sym in [s for s in watchlist if s != "BTC/USDT"][:8]:
            try:
                df = feed.fetch_ohlcv(sym, "4h", limit=60)
                if df is None or len(df) < 50:
                    continue
                c   = df["close"]
                e20 = float(c.ewm(span=20).mean().iloc[-1])
                e50 = float(c.ewm(span=50).mean().iloc[-1])
                p   = float(c.iloc[-1])
                if p > e20 > e50:   bull += 1
                elif p < e20 < e50: bear += 1
                checked += 1
            except Exception:
                pass
        breadth      = bull / checked if checked else 0.5
        bear_breadth = bear / checked if checked else 0.5

        trend_direction = "BULLISH" if btc_bullish else ("BEARISH" if btc_bearish else "NEUTRAL")
        trend_strength  = "STRONG" if adx_val > 30 else ("MODERATE" if adx_val > 22 else "WEAK")

        if chg_4h < -0.05 or chg_24h < -0.10:
            # gate=True: allow short entries; allow_longs=False blocks the other side
            return dict(regime="CRASH", gate=True,
                        allow_longs=False, allow_shorts=True,
                        trend_direction="BEARISH", trend_strength="STRONG",
                        min_conf=0.65, size_mult=0.4,
                        breadth=breadth, bear_breadth=bear_breadth,
                        vol_ratio=vol_ratio, adx=adx_val)

        if vol_ratio > 2.0:
            return dict(regime="HIGH_VOLATILITY", gate=False,
                        allow_longs=False, allow_shorts=False,
                        trend_direction=trend_direction, trend_strength=trend_strength,
                        min_conf=0.70, size_mult=0.3,
                        breadth=breadth, bear_breadth=bear_breadth,
                        vol_ratio=vol_ratio, adx=adx_val)

        if adx_val > 25 and (btc_bullish or btc_bearish) and (breadth > 0.60 or bear_breadth > 0.60):
            # Both directions open — Gate 4b applies breadth-proportional confidence biasing
            # (restricts counter-trend shorts/longs without hard-blocking high-conviction signals)
            st_dir = "BULLISH" if breadth > 0.60 else "BEARISH"
            return dict(regime="STRONG_TREND", gate=True,
                        allow_longs=True,
                        allow_shorts=True,
                        trend_direction=st_dir, trend_strength="STRONG",
                        min_conf=0.45, size_mult=1.2,
                        breadth=breadth, bear_breadth=bear_breadth,
                        vol_ratio=vol_ratio, adx=adx_val)

        # CHOPPY: truly directionless low-momentum market — no new entries allowed
        if adx_val < 18 or (adx_val < 22 and vol_ratio < 0.85 and abs(breadth - 0.5) < 0.12):
            return dict(regime="CHOPPY", gate=False,
                        allow_longs=False, allow_shorts=False,
                        trend_direction=trend_direction, trend_strength="WEAK",
                        min_conf=0.75, size_mult=0.0,
                        breadth=breadth, bear_breadth=bear_breadth,
                        vol_ratio=vol_ratio, adx=adx_val)

        if adx_val < 22 and vol_ratio < 0.9 and 0.30 < breadth < 0.65:
            return dict(regime="RANGING", gate=True,
                        allow_longs=True, allow_shorts=True,
                        trend_direction=trend_direction, trend_strength=trend_strength,
                        min_conf=0.62, size_mult=0.55,
                        breadth=breadth, bear_breadth=bear_breadth,
                        vol_ratio=vol_ratio, adx=adx_val)

        # WEAK_TREND: both directions open — Gate 4b applies breadth-proportional biasing
        wt_dir = "BULLISH" if breadth > 0.55 else ("BEARISH" if bear_breadth > 0.55 else "NEUTRAL")
        return dict(regime="WEAK_TREND", gate=True,
                    allow_longs=True,
                    allow_shorts=True,
                    trend_direction=wt_dir, trend_strength=trend_strength,
                    min_conf=0.50, size_mult=0.85,
                    breadth=breadth, bear_breadth=bear_breadth,
                    vol_ratio=vol_ratio, adx=adx_val)

    def _compute_from_values(self, adx: float, vol_ratio: float, breadth: float,
                             bear_breadth: float, chg_4h: float, chg_24h: float,
                             btc_bullish: bool = False, btc_bearish: bool = False) -> dict:
        """Test helper: run classification logic without fetching live data."""
        trend_direction = "BULLISH" if btc_bullish else ("BEARISH" if btc_bearish else "NEUTRAL")
        trend_strength  = "STRONG" if adx > 30 else ("MODERATE" if adx > 22 else "WEAK")
        if chg_4h < -0.05 or chg_24h < -0.10:
            return dict(regime="CRASH", gate=True, allow_longs=False, allow_shorts=True,
                        trend_direction="BEARISH", trend_strength="STRONG",
                        min_conf=0.65, size_mult=0.4,
                        breadth=breadth, bear_breadth=bear_breadth, vol_ratio=vol_ratio, adx=adx)
        if vol_ratio > 2.0:
            return dict(regime="HIGH_VOLATILITY", gate=False, allow_longs=False, allow_shorts=False,
                        trend_direction=trend_direction, trend_strength=trend_strength,
                        min_conf=0.70, size_mult=0.3,
                        breadth=breadth, bear_breadth=bear_breadth, vol_ratio=vol_ratio, adx=adx)
        if adx > 25 and (btc_bullish or btc_bearish) and (breadth > 0.60 or bear_breadth > 0.60):
            st_dir = "BULLISH" if breadth > 0.60 else "BEARISH"
            return dict(regime="STRONG_TREND", gate=True,
                        allow_longs=True, allow_shorts=True,
                        trend_direction=st_dir, trend_strength="STRONG",
                        min_conf=0.45, size_mult=1.2,
                        breadth=breadth, bear_breadth=bear_breadth, vol_ratio=vol_ratio, adx=adx)
        if adx < 18 or (adx < 22 and vol_ratio < 0.85 and abs(breadth - 0.5) < 0.12):
            return dict(regime="CHOPPY", gate=False, allow_longs=False, allow_shorts=False,
                        trend_direction=trend_direction, trend_strength="WEAK",
                        min_conf=0.75, size_mult=0.0,
                        breadth=breadth, bear_breadth=bear_breadth, vol_ratio=vol_ratio, adx=adx)
        if adx < 22 and vol_ratio < 0.9 and 0.30 < breadth < 0.65:
            return dict(regime="RANGING", gate=True, allow_longs=True, allow_shorts=True,
                        trend_direction=trend_direction, trend_strength=trend_strength,
                        min_conf=0.62, size_mult=0.55,
                        breadth=breadth, bear_breadth=bear_breadth, vol_ratio=vol_ratio, adx=adx)
        wt_dir = "BULLISH" if breadth > 0.55 else ("BEARISH" if bear_breadth > 0.55 else "NEUTRAL")
        return dict(regime="WEAK_TREND", gate=True,
                    allow_longs=True, allow_shorts=True,
                    trend_direction=wt_dir, trend_strength=trend_strength,
                    min_conf=0.50, size_mult=0.85,
                    breadth=breadth, bear_breadth=bear_breadth, vol_ratio=vol_ratio, adx=adx)

    def _neutral(self) -> dict:
        return dict(regime="UNKNOWN", gate=True,
                    allow_longs=True, allow_shorts=True,
                    trend_direction="NEUTRAL", trend_strength="MODERATE",
                    min_conf=0.52, size_mult=0.8,
                    breadth=0.5, bear_breadth=0.5, vol_ratio=1.0, adx=25.0)

    def _calc_adx(self, high, low, close, period=14):
        try:
            up       = high.diff()
            down     = -low.diff()
            plus_dm  = up.where((up > down) & (up > 0), 0)
            minus_dm = down.where((down > up) & (down > 0), 0)
            tr = pd.concat([
                high - low,
                (high - close.shift()).abs(),
                (low  - close.shift()).abs(),
            ], axis=1).max(axis=1)
            atr      = tr.ewm(span=period).mean()
            plus_di  = 100 * plus_dm.ewm(span=period).mean() / (atr + 1e-9)
            minus_di = 100 * minus_dm.ewm(span=period).mean() / (atr + 1e-9)
            dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
            return dx.ewm(span=period).mean()
        except Exception:
            return pd.Series([25.0] * len(close), index=close.index)


class ExitEngine:
    """
    Expectancy-optimised exit engine.

    Per-trade state machine:
      · Early invalidation  — exits fast when breakout fails (volume/momentum)
      · TP1 partial         — closes tp1_fraction at tp1_r_mult×ATR; arms breakeven
      · Breakeven clamp     — trailing stop never goes below entry after TP1
      · ATR trail           — profile trail_atr_mult, widens when dynamic_tp active
      · Swing structure     — exits on swing-low/high break for more natural stops
      · Fixed TP backstop   — wide hard cap; skipped when trend is accelerating
    """

    def __init__(self, default_trail_mult: float = 2.0):
        self._default_trail = default_trail_mult
        self._peaks:      dict = {}   # best price seen (low for shorts, high for longs)
        self._entry_atrs: dict = {}   # ATR at initialisation
        self._tp1_done:   dict = {}
        self._be_active:  dict = {}   # breakeven clamp armed
        self._dirty = False
        self._file  = DATA / f"exit_engine_{BOT_MODE}.json"
        self._load()

    def _load(self):
        if not self._file.exists():
            return
        try:
            with open(self._file) as f:
                d = json.load(f)
            self._peaks      = d.get("peaks", {})
            self._entry_atrs = d.get("entry_atrs", {})
            self._tp1_done   = d.get("tp1_done", {})
            self._be_active  = d.get("be_active", {})
        except Exception:
            pass

    def _save(self):
        tmp = self._file.with_suffix(".tmp.json")
        with open(tmp, "w") as f:
            json.dump({
                "peaks":      self._peaks,
                "entry_atrs": self._entry_atrs,
                "tp1_done":   self._tp1_done,
                "be_active":  self._be_active,
            }, f)
        tmp.replace(self._file)
        self._dirty = False

    def flush(self):
        if self._dirty:
            self._save()

    def cleanup(self, trade_id: str):
        for store in (self._peaks, self._entry_atrs, self._tp1_done, self._be_active):
            store.pop(trade_id, None)
        self._save()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _init(self, trade_id: str, price: float, atr: float):
        self._peaks[trade_id]      = price
        self._entry_atrs[trade_id] = max(atr, 1e-9)
        self._tp1_done[trade_id]   = False
        self._be_active[trade_id]  = False
        self._dirty = True

    def _update_peak(self, trade_id: str, price: float, atr: float, is_short: bool):
        if is_short:
            if price < self._peaks.get(trade_id, price):
                self._peaks[trade_id] = price
        else:
            if price > self._peaks.get(trade_id, price):
                self._peaks[trade_id] = price
        self._dirty = True

    # ── Main exit decision ────────────────────────────────────────────────────

    def should_exit(
        self,
        trade_id: str,
        entry: float,
        price: float,
        atr: float,
        side: str = "long",
        profile=None,
        candle_df=None,
    ) -> tuple:
        """Returns (fraction_to_close, reason).  fraction=0.0 → hold."""
        is_short = side in ("short", "sell")

        # Initialise new trade state
        if trade_id not in self._peaks:
            self._init(trade_id, price, max(atr, 1e-9))

        entry_atr = self._entry_atrs.get(trade_id, max(atr, 1e-9))
        self._update_peak(trade_id, price, atr, is_short)

        # No-ATR fallback — hard 2.5% stop to keep trades safe without data
        if atr <= 0:
            gain_pct = ((entry - price) if is_short else (price - entry)) / (entry + 1e-9)
            if gain_pct < -0.025:
                return 1.0, "Fixed SL 2.5% (no ATR)"
            return 0.0, ""

        gain = (entry - price) if is_short else (price - entry)

        # Pull profile params
        tp1_fraction       = getattr(profile, "tp1_fraction",       0.40)
        tp1_r_mult         = getattr(profile, "tp1_r_mult",         1.0)
        trail_mult         = getattr(profile, "trail_atr_mult",      self._default_trail)
        early_exit_enabled = getattr(profile, "early_exit_enabled",  True)
        dynamic_tp_enabled = getattr(profile, "dynamic_tp_enabled",  True)
        tp_backstop_mult   = getattr(profile, "take_profit_atr_mult", 2.5)

        tp1_done  = self._tp1_done.get(trade_id, False)
        be_active = self._be_active.get(trade_id, False)

        # ── 1. Early invalidation (pre-TP1, only when trade is losing) ────────
        if early_exit_enabled and not tp1_done:
            reason = self._check_invalidation(entry, price, entry_atr, gain, is_short, candle_df)
            if reason:
                return 1.0, f"INVALIDATION: {reason}"

        # ── 2. TP1 partial — lock in fraction, arm breakeven ─────────────────
        if not tp1_done and gain >= tp1_r_mult * entry_atr:
            self._tp1_done[trade_id]  = True
            self._be_active[trade_id] = True
            self._dirty = True
            return tp1_fraction, f"PARTIAL_TP1 +{tp1_r_mult:.1f}R ({tp1_fraction:.0%})"

        # ── 3. Dynamic TP extension — skip backstop when trend accelerates ────
        skip_backstop = (
            dynamic_tp_enabled and tp1_done
            and candle_df is not None
            and self._trend_accelerating(candle_df, is_short)
        )

        # ── 4. Fixed TP backstop (wide — only fires on truly large moves) ─────
        if not skip_backstop and gain >= tp_backstop_mult * entry_atr:
            return 1.0, f"TP_BACKSTOP +{tp_backstop_mult:.1f}R"

        # ── 5. ATR trailing stop (clamped to breakeven after TP1) ────────────
        peak = self._peaks.get(trade_id, entry)
        if is_short:
            trail_stop = peak + trail_mult * atr
            if be_active:
                trail_stop = min(trail_stop, entry)  # never above entry for shorts
            if price >= trail_stop:
                return 1.0, f"ATR_TRAIL SL ${trail_stop:.4f} (trough=${peak:.4f})"
        else:
            trail_stop = peak - trail_mult * atr
            if be_active:
                trail_stop = max(trail_stop, entry)  # never below entry for longs
            if price <= trail_stop:
                return 1.0, f"ATR_TRAIL SL ${trail_stop:.4f} (peak=${peak:.4f})"

        # ── 6. Swing structure trailing (post-TP1, uses confirmed candle lows/highs) ─
        if tp1_done and candle_df is not None:
            reason = self._swing_trail(entry, price, atr, is_short, be_active, candle_df)
            if reason:
                return 1.0, reason

        return 0.0, ""

    # ── Signal helpers ────────────────────────────────────────────────────────

    def _check_invalidation(
        self, entry: float, price: float, entry_atr: float,
        gain: float, is_short: bool, candle_df,
    ) -> str:
        """Return non-empty reason when pre-TP1 trade should be cut fast."""
        if candle_df is None or len(candle_df) < 5:
            return ""
        # Only invalidate when clearly losing (more than 0.3R underwater)
        if gain > -0.3 * entry_atr:
            return ""

        # Volume collapse: last completed bar volume < 35% of 20-bar avg
        # Use iloc[-2] — iloc[-1] is the forming (incomplete) candle with near-zero volume
        try:
            vol = candle_df["volume"]
            avg = float(vol.iloc[-22:-2].mean()) if len(vol) >= 22 else float(vol.iloc[:-1].mean())
            cur = float(vol.iloc[-2])
            if avg > 0 and cur < 0.35 * avg:
                return f"volume_collapse ({cur:.0f} < 35% avg {avg:.0f})"
        except Exception:
            pass

        # Momentum reversal: 3 consecutive closed candles moving against trade
        try:
            c = candle_df["close"]
            if len(c) >= 4:
                c1, c2, c3 = float(c.iloc[-4]), float(c.iloc[-3]), float(c.iloc[-2])
                if is_short and c1 < c2 < c3:
                    return "momentum_reversal (3 rising closes vs short)"
                if not is_short and c1 > c2 > c3:
                    return "momentum_reversal (3 falling closes vs long)"
        except Exception:
            pass

        return ""

    def _trend_accelerating(self, candle_df, is_short: bool) -> bool:
        """Return True when volume is expanding AND price momentum is on-side."""
        try:
            if len(candle_df) < 8:
                return False
            vol = candle_df["volume"]
            c   = candle_df["close"]
            vol_avg = float(vol.iloc[-10:-2].mean())
            cur_vol = float(vol.iloc[-2])   # last completed bar
            # 3-bar price momentum using completed candles
            m3 = (float(c.iloc[-2]) - float(c.iloc[-5])) / (float(c.iloc[-5]) + 1e-9)
            on_side = (m3 < -0.005) if is_short else (m3 > 0.005)
            return cur_vol > 1.2 * vol_avg and on_side
        except Exception:
            return False

    def _swing_trail(
        self, entry: float, price: float, atr: float,
        is_short: bool, be_active: bool, candle_df,
    ) -> str:
        """Swing-low/high trail using last 10 confirmed candle bars."""
        try:
            recent = candle_df.iloc[-10:]
            if is_short:
                # Trail at confirmed swing high (3-bar rolling max, shift 1 bar back)
                swing_h = float(recent["high"].rolling(3).max().iloc[-2])
                stop = swing_h + 0.2 * atr
                if be_active:
                    stop = min(stop, entry)
                if price >= stop:
                    return f"SWING_TRAIL SL ${stop:.4f} (swing_high=${swing_h:.4f})"
            else:
                swing_l = float(recent["low"].rolling(3).min().iloc[-2])
                stop = swing_l - 0.2 * atr
                if be_active:
                    stop = max(stop, entry)
                if price <= stop:
                    return f"SWING_TRAIL SL ${stop:.4f} (swing_low=${swing_l:.4f})"
        except Exception:
            pass
        return ""


class KellyCriterionSizer:
    """
    Conviction-weighted, equity-adaptive position sizer.

    pos_pct = dynamic_risk_pct × quality_scalar × regime_scalar × volatility_scalar × correlation_scalar

    dynamic_risk_pct  — BASE_PCT scaled by drawdown/streak state; adapts gradually
    quality_scalar    — confidence tier + Kelly history + all-agree boost  [0.70, 1.40]
    regime_scalar     — regime size_mult + ADX modifier, floored to prevent collapse  [0.75, 1.20]
    volatility_scalar — ATR-normalised stop distance; slight upside in calm markets  [0.60, 1.10]
    correlation_scalar — portfolio concentration penalty  [0.50, 1.00]

    Scalars are bounded independently — they cannot cascade to near-zero simultaneously.
    Floor product: 0.70 × 0.75 × 0.60 × 0.50 ≈ 0.16 × base
    Ceiling product: 1.40 × 1.20 × 1.10 × 1.00 ≈ 1.85 × base
    """

    # Reference: sl_mult=2.5, atr_pct=2% → volatility_scalar=1.0
    _RISK_NORM = 2.5 * 0.02  # = 0.05

    KELLY_FRAC_MIN     = 0.10
    KELLY_FRAC_MAX     = 0.25
    KELLY_FRAC_DEFAULT = 0.15

    MIN_PCT  = 0.003   # 0.3% floor — prevents sub-$3 positions on $1k account
    BASE_PCT = {"spot": 0.008, "futures": 0.004}
    MAX_PCT  = {"spot": 0.020, "futures": 0.015}

    def __init__(self, config=None):
        cfg = config or {}
        self.leverage = cfg.get("risk", {}).get("leverage", 1)
        self._streak_active = False

    def calculate(self, confidence, balance, price, atr_pct, regime, recent_trades,
                  open_trades=None, all_agree=False, sl_atr_mult: float = 2.5):
        mode = BOT_MODE

        # ── 1. Equity-adaptive risk budget ───────────────────────────────────
        dynamic_rp = self._dynamic_risk_pct(balance, recent_trades)

        # ── 2. Bounded conviction scalars ────────────────────────────────────
        q_scalar = self._quality_scalar(confidence, all_agree, recent_trades)
        r_scalar = self._regime_scalar(regime)
        v_scalar = self._volatility_scalar(atr_pct, sl_atr_mult)
        c_scalar = self._correlation_scalar(open_trades)

        # ── 3. Combine ────────────────────────────────────────────────────────
        pos_pct = dynamic_rp * q_scalar * r_scalar * v_scalar * c_scalar

        # ── 4. Hard cap ───────────────────────────────────────────────────────
        cap = self.MAX_PCT.get(mode, 0.015)
        pos_pct = max(self.MIN_PCT, min(cap, pos_pct))

        usdt   = balance * pos_pct * self.leverage
        amount = usdt / price

        log.debug(
            f"Sizer [{mode}]: base={dynamic_rp*100:.3f}% "
            f"Q={q_scalar:.2f} R={r_scalar:.2f} V={v_scalar:.2f} C={c_scalar:.2f} "
            f"→ {pos_pct*100:.3f}% × {self.leverage}x = ${usdt:.2f} "
            f"(cap={cap*100:.1f}% conf={confidence:.2f} {regime.get('regime','')})"
        )
        return amount, usdt

    def _dynamic_risk_pct(self, balance: float, recent_trades: list) -> float:
        """Base risk pct scaled by drawdown state. Streak guard folds here."""
        base = self.BASE_PCT.get(BOT_MODE, 0.008)
        closed = [t for t in recent_trades if t.get("status") == "closed"]

        # Losing streak: ≥3 of last 5 closed trades losing → 65% of base
        recent_5 = closed[-5:]
        streak_on = len(recent_5) >= 3 and sum(1 for t in recent_5 if t.get("pnl", 0) <= 0) >= 3
        if streak_on:
            if not self._streak_active:
                log.warning("Losing streak — dynamic risk reduced to 65% of base")
                self._streak_active = True
            return round(base * 0.65, 6)
        if self._streak_active:
            log.info("Losing streak cleared — risk normalizing")
            self._streak_active = False

        # Rolling PnL ratio over last 10 closed trades
        if len(closed) >= 5 and balance > 0:
            rolling_pnl = sum(t.get("pnl", 0) for t in closed[-10:])
            ratio = rolling_pnl / (balance + 1e-9)
            if ratio < -0.04:   return round(base * 0.60, 6)   # heavy drawdown
            if ratio < -0.02:   return round(base * 0.75, 6)   # moderate drawdown
            if ratio >  0.03:   return round(base * 1.05, 6)   # stable profitability

        return base

    def _quality_scalar(self, confidence: float, all_agree: bool, recent_trades: list) -> float:
        """Conviction from confidence tier + Kelly history + all-agree. Range: [0.70, 1.40]."""
        if   confidence >= 0.75: base = 1.20
        elif confidence >= 0.65: base = 1.00
        elif confidence >= 0.55: base = 0.85
        else:                    base = 0.70

        if all_agree:
            base += 0.15

        kelly_frac  = self._kelly_fraction(recent_trades)
        kelly_bonus = (kelly_frac - self.KELLY_FRAC_DEFAULT) / self.KELLY_FRAC_DEFAULT * 0.30

        return round(max(0.70, min(1.40, base + kelly_bonus)), 4)

    def _regime_scalar(self, regime: dict) -> float:
        """Regime size_mult + ADX modifier. Floor at 0.75 prevents regime from collapsing size. Range: [0.75, 1.20]."""
        size_mult = regime.get("size_mult", 0.85)
        adx = regime.get("adx", 25.0)
        if   adx >= 35: adx_mod =  0.10
        elif adx >= 28: adx_mod =  0.05
        elif adx <  20: adx_mod = -0.10
        elif adx <  25: adx_mod = -0.05
        else:           adx_mod =  0.00
        return round(max(0.75, min(1.20, size_mult + adx_mod)), 4)

    def _volatility_scalar(self, atr_pct: float, sl_atr_mult: float) -> float:
        """ATR-normalised stop distance. Wider stop/higher vol → smaller size. Range: [0.60, 1.10]."""
        raw = self._RISK_NORM / max(sl_atr_mult * atr_pct, 0.001)
        return round(max(0.60, min(1.10, raw)), 4)

    def _correlation_scalar(self, open_trades) -> float:
        """Portfolio concentration: 8% reduction per open trade. Range: [0.50, 1.00]."""
        n_open = len(open_trades) if open_trades else 0
        return round(max(0.50, 1.0 - n_open * 0.08), 4)

    def _kelly_fraction(self, recent_trades: list) -> float:
        """Fractional Kelly from last 20 closed trades, capped at [0.10, 0.25]."""
        closed = [t for t in recent_trades if t.get("status") == "closed"]
        if len(closed) < 10:
            return self.KELLY_FRAC_DEFAULT
        last_20 = closed[-20:]
        pct_returns = []
        for t in last_20:
            ep  = float(t.get("price", 0))
            amt = float(t.get("amount", 0))
            pnl = float(t.get("pnl", 0))
            if ep > 0 and amt > 0:
                pct_returns.append(pnl / (ep * amt + 1e-9))
        if len(pct_returns) < 5:
            return self.KELLY_FRAC_DEFAULT
        wins   = [r for r in pct_returns if r > 0]
        losses = [abs(r) for r in pct_returns if r <= 0]
        if not wins or not losses:
            return self.KELLY_FRAC_DEFAULT
        wr     = len(wins) / len(pct_returns)
        payoff = np.mean(wins) / (np.mean(losses) + 1e-9)
        full_k = max(0.0, (wr * payoff - (1 - wr)) / (payoff + 1e-9))
        return round(max(self.KELLY_FRAC_MIN, min(self.KELLY_FRAC_MAX, full_k)), 4)


class PortfolioHeatTracker:
    def __init__(self, max_heat=0.40):
        self.max_heat = max_heat

    def get_heat(self, open_trades, balance, get_price_fn):
        if not open_trades or balance <= 0:
            return 0.0
        exposure = sum(
            (get_price_fn(t["symbol"]) or 0) * t["amount"]
            for t in open_trades
        )
        return round(exposure / (balance + exposure + 1e-9), 4)

    def can_add_position(self, open_trades, balance, new_usdt, get_price_fn):
        heat     = self.get_heat(open_trades, balance, get_price_fn)
        exposure = heat * (balance + 1e-9) / (1 - heat + 1e-9)
        new_exposure = exposure + new_usdt
        new_heat = new_exposure / (balance + new_exposure + 1e-9)
        if new_heat > self.max_heat:
            return False, f"Portfolio heat {heat*100:.1f}% (max {self.max_heat*100:.0f}%)"
        return True, f"Heat OK ({heat*100:.1f}%)"


class CircuitBreaker:
    WINDOW_DAYS = 2

    def __init__(self, config):
        self.max_daily_loss  = config.get("max_daily_loss_pct", 0.05)
        self.max_consec_loss = config.get("max_consecutive_losses", 4)
        self.max_drawdown    = config.get("max_drawdown_pct", 0.10)
        self._initial_balance = None
        self._peak_balance   = None
        self._load()

    def _load(self):
        p = DATA / "circuit_breaker.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
                self.pnl_history   = d.get("pnl_history", {})
                self.consec_losses = d.get("consec_losses", 0)
                self._initial_balance = d.get("initial_balance")
                self._peak_balance   = d.get("peak_balance")
        else:
            self.pnl_history   = {}
            self.consec_losses = 0
            self._initial_balance = None
            self._peak_balance   = None

    def _save(self):
        from datetime import timedelta
        cutoff = (date.today() - timedelta(days=self.WINDOW_DAYS)).isoformat()
        self.pnl_history = {k: v for k, v in self.pnl_history.items() if k >= cutoff}
        with open(DATA / "circuit_breaker.json", "w") as f:
            json.dump({"pnl_history": self.pnl_history,
                       "consec_losses": self.consec_losses,
                       "initial_balance": self._initial_balance,
                       "peak_balance": self._peak_balance}, f)

    def _get_rolling_loss(self):
        from datetime import timedelta
        today_str = str(date.today())
        if today_str not in self.pnl_history:
            self.pnl_history[today_str] = 0.0
        total = 0.0
        for i in range(self.WINDOW_DAYS):
            d = date.today() - timedelta(days=i)
            total += self.pnl_history.get(str(d), 0.0)
        return total

    def record_trade(self, pnl, balance):
        if self._initial_balance is None and balance > 0:
            self._initial_balance = balance
        if self._peak_balance is None or balance > self._peak_balance:
            self._peak_balance = balance
        today_str = str(date.today())
        self.pnl_history[today_str] = self.pnl_history.get(today_str, 0.0) + pnl
        self.consec_losses = self.consec_losses + 1 if pnl < 0 else 0
        self._save()

    def can_trade(self, balance):
        rolling_loss = self._get_rolling_loss()
        ref_balance = self._initial_balance or balance
        threshold = ref_balance * self.max_daily_loss * self.WINDOW_DAYS
        if rolling_loss < -threshold:
            return False, f"Rolling {self.WINDOW_DAYS}-day loss ${rolling_loss:.2f} (limit -${threshold:.2f})"
        if self.consec_losses >= self.max_consec_loss:
            return False, f"Consecutive losses: {self.consec_losses}"
        peak = self._peak_balance or self._initial_balance or balance
        if peak and peak > 0:
            drawdown = (peak - balance) / peak
            if drawdown > self.max_drawdown:
                return False, f"Drawdown {drawdown*100:.1f}% from peak ${peak:.2f} (limit {self.max_drawdown*100:.0f}%)"
        return True, "OK"


class RiskManager:
    def __init__(self, config):
        risk              = config.get("risk", {})
        self.market_gate  = MarketRegimeGate()
        self.correlation  = CorrelationFilter()
        self.exit_engine  = ExitEngine(risk.get("stop_loss_atr_multiplier", 2.0))
        self.sizer        = KellyCriterionSizer(config)
        self.breaker      = CircuitBreaker(risk)
        self.heat         = PortfolioHeatTracker(risk.get("max_portfolio_heat", 0.40))
        self.sl_min_pct   = risk.get("stop_loss_min_pct", 0.015)

    def check_exits(self, open_trades, get_price_fn, get_atr_fn,
                    get_ohlcv_fn=None, profile=None):
        """
        Evaluate all open trades for exit conditions via ExitEngine.
        get_ohlcv_fn(symbol) → pd.DataFrame | None  (trading-timeframe candles, limit≈30)
        profile → TradingProfile (controls TP1, trail multiplier, early-exit flags)
        """
        exits = []
        now   = datetime.now(timezone.utc)
        for trade in open_trades:
            price = get_price_fn(trade["symbol"])
            if price is None or price <= 0:
                continue
            entry    = float(trade["price"])
            trade_id = trade["id"]
            side     = trade.get("side", "long")
            is_short = side in ("short", "sell")

            # Stale exit: trade > 72h old and still losing
            try:
                opened = datetime.fromisoformat(trade.get("timestamp", ""))
                if opened.tzinfo is None:
                    opened = opened.replace(tzinfo=timezone.utc)
                if (now - opened).total_seconds() > 72 * 3600:
                    pnl_est = ((entry - price) if is_short else (price - entry)) / entry
                    if pnl_est < 0:
                        exits.append((trade, price, "Stale exit (72h, loss)", 1.0))
                        continue
            except (ValueError, TypeError):
                pass

            atr       = get_atr_fn(trade["symbol"])
            candle_df = get_ohlcv_fn(trade["symbol"]) if get_ohlcv_fn else None

            fraction, reason = self.exit_engine.should_exit(
                trade_id, entry, price, atr, side, profile, candle_df
            )
            if fraction > 0:
                exits.append((trade, price, reason, fraction))

        self.exit_engine.flush()
        return exits

    def can_open_trade(self, symbol, open_trades, balance, new_usdt, get_price_fn):
        ok, r = self.breaker.can_trade(balance)
        if not ok: return False, r
        if any(t["symbol"] == symbol for t in open_trades):
            return False, f"Already holding {symbol}"
        ok, r = self.correlation.is_allowed(symbol, open_trades)
        if not ok: return False, f"Correlation: {r}"
        ok, r = self.heat.can_add_position(open_trades, balance, new_usdt, get_price_fn)
        if not ok: return False, r
        return True, "OK"

    def detect_market_regime(self, feed, watchlist: list) -> dict:
        return self.market_gate.detect(feed, watchlist)

    def get_position_size(self, confidence, balance, price, df, recent_trades,
                           regime_ctx=None, all_agree=False, open_trades=None,
                           sl_atr_mult: float = 2.5):
        regime  = dict(regime_ctx) if regime_ctx else self.market_gate._neutral()
        high    = df["high"]
        low     = df["low"]
        close   = df["close"]
        tr      = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr     = tr.rolling(14).mean().iloc[-1]
        atr_pct = float(atr / (df["close"].iloc[-1] + 1e-9))
        log.debug(f"Regime: {regime['regime']} ADX={regime.get('adx','?')} size_mult={regime.get('size_mult',1.0):.2f} sl_atr={sl_atr_mult}")
        return self.sizer.calculate(
            confidence, balance, price, atr_pct, regime, recent_trades,
            open_trades=open_trades, all_agree=all_agree, sl_atr_mult=sl_atr_mult,
        )

    def record_trade_result(self, pnl, balance):
        self.breaker.record_trade(pnl, balance)

    def get_dynamic_min_conf(self, base_min_conf: float, recent_trades: list) -> float:
        """
        v4: Dynamically adjust min_confidence based on recent P&L trajectory.
        Aggressive when winning, defensive when losing.
        Returns adjusted min_conf clamped to [0.35, 0.65].
        """
        if len(recent_trades) < 10:
            return base_min_conf
        closed = [t for t in recent_trades if t.get("status") == "closed"]
        if len(closed) < 5:
            return base_min_conf
        wins = sum(1 for t in closed if t.get("pnl", 0) > 0)
        win_rate = wins / len(closed)
        adj = 0.0
        if win_rate > 0.60:
            adj = -0.03
        elif win_rate < 0.35:
            adj = 0.05
        elif win_rate < 0.45:
            adj = 0.02
        return round(max(0.35, min(0.65, base_min_conf + adj)), 4)

    def cleanup_trade(self, trade_id):
        self.exit_engine.cleanup(trade_id)

    def flush(self):
        self.exit_engine.flush()
