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

        if chg_4h < -0.05 or chg_24h < -0.10:
            return dict(regime="CRASH", gate=False,
                        allow_longs=False, allow_shorts=True,
                        min_conf=0.65, size_mult=0.4,
                        breadth=breadth, bear_breadth=bear_breadth,
                        vol_ratio=vol_ratio, adx=adx_val)

        if vol_ratio > 2.0:
            return dict(regime="HIGH_VOLATILITY", gate=False,
                        allow_longs=False, allow_shorts=False,
                        min_conf=0.70, size_mult=0.3,
                        breadth=breadth, bear_breadth=bear_breadth,
                        vol_ratio=vol_ratio, adx=adx_val)

        if adx_val > 25 and (btc_bullish or btc_bearish) and (breadth > 0.60 or bear_breadth > 0.60):
            # Only allow trades in the direction the trend is running
            return dict(regime="STRONG_TREND", gate=True,
                        allow_longs=btc_bullish,
                        allow_shorts=btc_bearish,
                        min_conf=0.45, size_mult=1.2,
                        breadth=breadth, bear_breadth=bear_breadth,
                        vol_ratio=vol_ratio, adx=adx_val)

        # CHOPPY: truly directionless low-momentum market — no new entries allowed
        if adx_val < 18 or (adx_val < 22 and vol_ratio < 0.85 and abs(breadth - 0.5) < 0.12):
            return dict(regime="CHOPPY", gate=False,
                        allow_longs=False, allow_shorts=False,
                        min_conf=0.75, size_mult=0.0,
                        breadth=breadth, bear_breadth=bear_breadth,
                        vol_ratio=vol_ratio, adx=adx_val)

        if adx_val < 22 and vol_ratio < 0.9 and 0.30 < breadth < 0.60:
            return dict(regime="RANGING", gate=True,
                        allow_longs=True, allow_shorts=True,
                        min_conf=0.62, size_mult=0.55,
                        breadth=breadth, bear_breadth=bear_breadth,
                        vol_ratio=vol_ratio, adx=adx_val)

        # WEAK_TREND: block the contra-trend direction when breadth is clearly one-sided
        return dict(regime="WEAK_TREND", gate=True,
                    allow_longs=bear_breadth < 0.60,
                    allow_shorts=breadth < 0.60,
                    min_conf=0.50, size_mult=0.85,
                    breadth=breadth, bear_breadth=bear_breadth,
                    vol_ratio=vol_ratio, adx=adx_val)

    def _compute_from_values(self, adx: float, vol_ratio: float, breadth: float,
                             bear_breadth: float, chg_4h: float, chg_24h: float,
                             btc_bullish: bool = False, btc_bearish: bool = False) -> dict:
        """Test helper: run classification logic without fetching live data."""
        if chg_4h < -0.05 or chg_24h < -0.10:
            return dict(regime="CRASH", gate=False, allow_longs=False, allow_shorts=True,
                        min_conf=0.65, size_mult=0.4,
                        breadth=breadth, bear_breadth=bear_breadth, vol_ratio=vol_ratio, adx=adx)
        if vol_ratio > 2.0:
            return dict(regime="HIGH_VOLATILITY", gate=False, allow_longs=False, allow_shorts=False,
                        min_conf=0.70, size_mult=0.3,
                        breadth=breadth, bear_breadth=bear_breadth, vol_ratio=vol_ratio, adx=adx)
        if adx > 25 and (btc_bullish or btc_bearish) and (breadth > 0.60 or bear_breadth > 0.60):
            return dict(regime="STRONG_TREND", gate=True,
                        allow_longs=btc_bullish, allow_shorts=btc_bearish,
                        min_conf=0.45, size_mult=1.2,
                        breadth=breadth, bear_breadth=bear_breadth, vol_ratio=vol_ratio, adx=adx)
        if adx < 18 or (adx < 22 and vol_ratio < 0.85 and abs(breadth - 0.5) < 0.12):
            return dict(regime="CHOPPY", gate=False, allow_longs=False, allow_shorts=False,
                        min_conf=0.75, size_mult=0.0,
                        breadth=breadth, bear_breadth=bear_breadth, vol_ratio=vol_ratio, adx=adx)
        if adx < 22 and vol_ratio < 0.9 and 0.30 < breadth < 0.60:
            return dict(regime="RANGING", gate=True, allow_longs=True, allow_shorts=True,
                        min_conf=0.62, size_mult=0.55,
                        breadth=breadth, bear_breadth=bear_breadth, vol_ratio=vol_ratio, adx=adx)
        return dict(regime="WEAK_TREND", gate=True,
                    allow_longs=bear_breadth < 0.60, allow_shorts=breadth < 0.60,
                    min_conf=0.50, size_mult=0.85,
                    breadth=breadth, bear_breadth=bear_breadth, vol_ratio=vol_ratio, adx=adx)

    def _neutral(self) -> dict:
        return dict(regime="UNKNOWN", gate=True,
                    allow_longs=True, allow_shorts=True,
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


class ATRTrailingStop:
    def __init__(self, atr_multiplier=2.0):
        self.multiplier  = atr_multiplier
        self.peak_prices = {}
        self.atr_values  = {}
        self.entry_atrs  = {}
        self.partial     = {}
        self._file = DATA / f"trailing_stops_{BOT_MODE}.json"
        self._load()

    def _load(self):
        if self._file.exists():
            with open(self._file) as f:
                d = json.load(f)
                self.peak_prices = d.get("peaks", {})
                self.atr_values  = d.get("atrs", {})
                self.entry_atrs  = d.get("entry_atrs", {})
                self.partial     = d.get("partial", {})

    def _save(self):
        with open(self._file, "w") as f:
            json.dump({
                "peaks":      self.peak_prices,
                "atrs":       self.atr_values,
                "entry_atrs": self.entry_atrs,
                "partial":    self.partial,
            }, f)

    def update(self, trade_id, price, atr, side="long"):
        if trade_id not in self.peak_prices:
            self.peak_prices[trade_id] = price
            self.atr_values[trade_id]  = atr
            self.entry_atrs[trade_id]  = atr
            self.partial[trade_id]     = {"tier1": False, "tier2": False}
        elif side in ("short", "sell"):
            if price < self.peak_prices[trade_id]:
                self.peak_prices[trade_id] = price
                self.atr_values[trade_id]  = atr
        else:
            if price > self.peak_prices[trade_id]:
                self.peak_prices[trade_id] = price
                self.atr_values[trade_id]  = atr
        self._dirty = True

    def flush(self):
        if getattr(self, "_dirty", False):
            self._save()
            self._dirty = False

    def cleanup(self, trade_id):
        self.peak_prices.pop(trade_id, None)
        self.atr_values.pop(trade_id, None)
        self.entry_atrs.pop(trade_id, None)
        self.partial.pop(trade_id, None)
        self._save()

    def should_exit(self, trade_id, entry, price, atr, side="long"):
        """Returns (fraction_to_close, reason). fraction=0.0 means hold."""
        self.update(trade_id, price, atr, side)
        is_short  = side in ("short", "sell")
        entry_atr = self.entry_atrs.get(trade_id, atr)
        partial   = self.partial.get(trade_id, {"tier1": False, "tier2": False})
        gain      = (entry - price) if is_short else (price - entry)

        # PARTIAL_TP1 removed: closing 40% at +1×ATR produced near-zero wins (~$0.01 avg)
        # because the remaining 60% frequently got stopped at break-even, netting a loss.
        # Mark tier1 silently so the tier2 check can still fire.
        if not partial["tier1"] and gain >= entry_atr:
            self.partial[trade_id]["tier1"] = True
            self._dirty = True

        # Partial exit at +2×ATR: take 50% off, let remainder trail to full TP
        if partial["tier1"] and not partial["tier2"] and gain >= 2 * entry_atr:
            self.partial[trade_id]["tier2"] = True
            self._dirty = True
            return 0.5, "PARTIAL_TP2 +2×ATR"

        # ATR trailing stop on remaining position
        extreme  = self.peak_prices.get(trade_id, entry)
        cur_atr  = self.atr_values.get(trade_id, atr)
        if is_short:
            trail_stop = extreme + self.multiplier * cur_atr
            init_stop  = entry  + self.multiplier * atr
            stop       = min(trail_stop, init_stop)
            if price >= stop:
                return 1.0, f"ATR SL ${stop:.4f} (trough ${extreme:.4f})"
        else:
            trail_stop = extreme - self.multiplier * cur_atr
            init_stop  = entry  - self.multiplier * atr
            stop       = max(trail_stop, init_stop)
            if price <= stop:
                return 1.0, f"ATR SL ${stop:.4f} (peak ${extreme:.4f})"
        return 0.0, ""


class KellyCriterionSizer:
    """
    Volatility-adjusted fractional Kelly position sizer.

    pos_pct = base_risk × kelly_scale × vol_factor × regime_factor × conf_factor × corr_factor × streak_factor

    base_risk  — mode-calibrated starting point (spot=0.8%, futures=0.4% of balance as margin)
    kelly_scale — derived from last 20 closed trades, capped at [0.10, 0.25] fractional Kelly
    vol_factor  — smooth ATR-aware reduction (2% ATR→1.0; 4% ATR→0.5; 6%+→0.4)
    regime_factor — regime size_mult + ADX modifier; capped at [0.30, 1.20]
    conf_factor — narrow [0.85, 1.0]; confidence contributes at most ±15%
    corr_factor — 10% per open position (portfolio concentration penalty)
    streak_factor — 0.5 if ≥3 of last 5 closed trades are losses
    """

    KELLY_FRAC_MIN     = 0.10
    KELLY_FRAC_MAX     = 0.25
    KELLY_FRAC_DEFAULT = 0.15   # cold-start (< 10 closed trades)

    MIN_PCT = 0.003   # 0.3% margin floor — prevents sub-$10 trades

    BASE_PCT = {"spot": 0.008, "futures": 0.004}  # baseline margin % of balance
    MAX_PCT  = {"spot": 0.020, "futures": 0.015}  # hard cap per mode

    # Reduce effective notional in risky regimes without touching exchange leverage
    _REGIME_LEV_SCALE = {
        "STRONG_TREND":    1.00,
        "TRENDING":        1.00,
        "WEAK_TREND":      0.85,
        "RANGING":         0.80,
        "HIGH_VOLATILITY": 0.65,
        "CRASH":           0.55,
        "CHOPPY":          0.50,
    }

    def __init__(self, config=None):
        cfg = config or {}
        self.leverage = cfg.get("risk", {}).get("leverage", 1)

    def calculate(self, confidence, balance, price, atr_pct, regime, recent_trades,
                  open_trades=None, all_agree=False):
        mode = BOT_MODE
        base = self.BASE_PCT.get(mode, 0.008)

        # ── 1. Fractional Kelly (0.10–0.25 from trade history) ────────
        kelly_frac = self._kelly_fraction(recent_trades)
        kelly_scale = kelly_frac / self.KELLY_FRAC_DEFAULT   # 1.0 at cold-start

        # ── 2. Volatility factor (smooth ATR-aware) ───────────────────
        # 2% ATR → 1.0 | 4% → 0.5 | 6%+ → 0.4 floor | <2% → capped at 1.0
        vol_factor = round(max(0.40, min(1.0, 0.02 / max(atr_pct, 0.001))), 4)

        # ── 3. Regime factor (size_mult + ADX modifier) ───────────────
        size_mult = regime.get("size_mult", 1.0)
        adx = regime.get("adx", 25.0)
        if   adx >= 35: adx_mod =  0.15
        elif adx >= 28: adx_mod =  0.05
        elif adx <  20: adx_mod = -0.20
        elif adx <  25: adx_mod = -0.10
        else:           adx_mod =  0.00
        regime_factor = round(max(0.30, min(1.20, size_mult + adx_mod)), 4)

        # ── 4. Confidence factor (narrow band — never dominant) ────────
        # conf=0.50 → 0.85 | conf=0.80 → 1.0 | max swing ±15%
        conf_factor = round(0.85 + min(0.15, max(0.0, (confidence - 0.50) * 0.5)), 4)

        # ── 5. Correlation / concentration factor ─────────────────────
        n_open = len(open_trades) if open_trades else 0
        corr_factor = round(max(0.60, 1.0 - n_open * 0.10), 4)

        # ── 6. Losing streak guard ─────────────────────────────────────
        recent_5 = [t for t in recent_trades if t.get("status") == "closed"][-5:]
        if len(recent_5) >= 3 and sum(1 for t in recent_5 if t.get("pnl", 0) <= 0) >= 3:
            streak_factor = 0.5
            log.warning("Losing streak — sizing halved")
        else:
            streak_factor = 1.0

        # ── Combine ────────────────────────────────────────────────────
        pos_pct = (base * kelly_scale * vol_factor * regime_factor
                   * conf_factor * corr_factor * streak_factor)

        # ── Hard cap (mode-aware, slight bonus on all-agree STRONG_TREND) ──
        cap = self.MAX_PCT.get(mode, 0.015)
        if all_agree and confidence >= 0.68 and regime.get("regime") == "STRONG_TREND":
            cap = min(cap * 1.15, cap + 0.004)
        pos_pct = max(self.MIN_PCT, min(cap, pos_pct))

        # ── Leverage scaling by regime ─────────────────────────────────
        regime_name = regime.get("regime", "RANGING")
        lev_scale   = self._REGIME_LEV_SCALE.get(regime_name, 0.85)
        effective_lev = self.leverage * lev_scale

        usdt   = balance * pos_pct * effective_lev
        amount = usdt / price

        log.debug(
            f"Sizer [{mode}]: base={base*100:.2f}% kelly={kelly_frac:.2f}(×{kelly_scale:.2f}) "
            f"vol={vol_factor:.2f} regime={regime_factor:.2f} conf={conf_factor:.2f} "
            f"corr={corr_factor:.2f} streak={streak_factor:.1f} "
            f"→ {pos_pct*100:.3f}% × {effective_lev:.1f}x = ${usdt:.2f} "
            f"(cap={cap*100:.1f}% adx={adx:.0f} {regime_name})"
        )
        return amount, usdt

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
        risk             = config.get("risk", {})
        self.market_gate = MarketRegimeGate()
        self.correlation = CorrelationFilter()
        self.trailing    = ATRTrailingStop(risk.get("stop_loss_atr_multiplier", 3.0))
        self.sizer       = KellyCriterionSizer(config)
        self.breaker     = CircuitBreaker(risk)
        self.heat        = PortfolioHeatTracker(risk.get("max_portfolio_heat", 0.40))
        self.sl_min_pct  = risk.get("stop_loss_min_pct", 0.015)
        self.tp_atr_mult = risk.get("take_profit_atr_multiplier", 2.5)
        self.fallback_tp = risk.get("take_profit_pct", 0.05)

    def check_exits(self, open_trades, get_price_fn, get_atr_fn):
        exits = []
        now = datetime.now(timezone.utc) if hasattr(self, 'tp_atr_mult') else None
        for trade in open_trades:
            price = get_price_fn(trade["symbol"])
            if price is None or price <= 0:
                continue
            entry    = float(trade["price"])
            trade_id = trade["id"]
            side     = trade.get("side", "long")
            is_short = side in ("short", "sell")

            stale_exit = False
            if now:
                try:
                    opened = datetime.fromisoformat(trade.get("timestamp", ""))
                    if opened.tzinfo is None:
                        opened = opened.replace(tzinfo=timezone.utc)
                    age_hours = (now - opened).total_seconds() / 3600
                    if age_hours > 72:
                        pnl_est = (price - entry) / entry * 100
                        if is_short:
                            pnl_est = -pnl_est
                        if pnl_est < 0:
                            stale_exit = True
                except (ValueError, TypeError):
                    pass

            if stale_exit:
                exits.append((trade, price, f"Stale exit (72h, loss)", 1.0))
                continue

            atr = get_atr_fn(trade["symbol"])
            if atr > 0:
                tp_distance = atr * self.tp_atr_mult
                if is_short:
                    tp_price = entry - tp_distance
                    if price <= tp_price:
                        exits.append((trade, price, f"ATR TP ${tp_price:.4f} ({self.tp_atr_mult}xATR)", 1.0))
                        continue
                else:
                    tp_price = entry + tp_distance
                    if price >= tp_price:
                        exits.append((trade, price, f"ATR TP ${tp_price:.4f} ({self.tp_atr_mult}xATR)", 1.0))
                        continue
            else:
                if is_short:
                    if price <= entry * (1 - self.fallback_tp):
                        exits.append((trade, price, f"Fixed TP {self.fallback_tp:.0%}", 1.0))
                        continue
                else:
                    if price >= entry * (1 + self.fallback_tp):
                        exits.append((trade, price, f"Fixed TP {self.fallback_tp:.0%}", 1.0))
                        continue

            if atr > 0:
                fraction, reason = self.trailing.should_exit(trade_id, entry, price, atr, side)
                if fraction > 0:
                    exits.append((trade, price, reason, fraction))
                    continue
                self.trailing.update(trade_id, price, atr, side)
            else:
                if is_short and price >= entry * 1.025:
                    exits.append((trade, price, "Fixed SL 2.5%", 1.0))
                elif not is_short and price <= entry * 0.975:
                    exits.append((trade, price, "Fixed SL 2.5%", 1.0))
        self.trailing.flush()
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
                           regime_ctx=None, all_agree=False, open_trades=None):
        regime  = dict(regime_ctx) if regime_ctx else self.market_gate._neutral()
        high    = df["high"]
        low     = df["low"]
        close   = df["close"]
        tr      = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr     = tr.rolling(14).mean().iloc[-1]
        atr_pct = float(atr / (df["close"].iloc[-1] + 1e-9))
        log.debug(f"Regime: {regime['regime']} ADX={regime.get('adx','?')} size_mult={regime.get('size_mult',1.0):.2f}")
        return self.sizer.calculate(
            confidence, balance, price, atr_pct, regime, recent_trades,
            open_trades=open_trades, all_agree=all_agree,
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
        self.trailing.cleanup(trade_id)

    def flush(self):
        self.trailing.flush()
