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
from pathlib import Path
from datetime import datetime, date, timezone
from typing import List, Tuple, Optional
from env_config import DATA_DIR
from technical_indicators import calc_adx

log  = logging.getLogger("RiskManager")
DATA = DATA_DIR


class RegimeDetector:
    def detect(self, df):
        try:
            close     = df["close"]
            high      = df["high"]
            low       = df["low"]
            adx       = calc_adx(high, low, close)
            ret       = close.pct_change()
            vol_ratio = float(ret.rolling(5).std().iloc[-1] / (ret.rolling(20).std().iloc[-1] + 1e-9))
            adx_val   = float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else 25
            bb_std    = close.rolling(20).std()
            bb_mid    = close.rolling(20).mean()
            bb_width  = float((bb_std.iloc[-1] * 4) / (bb_mid.iloc[-1] + 1e-9))

            if vol_ratio > 1.8:
                return {"regime": "VOLATILE", "adx": round(adx_val,1), "vol_ratio": round(vol_ratio,2), "size_mult": 0.4}
            elif adx_val > 25 and bb_width > 0.04:
                return {"regime": "TRENDING", "adx": round(adx_val,1), "vol_ratio": round(vol_ratio,2), "size_mult": 1.2}
            else:
                return {"regime": "RANGING",  "adx": round(adx_val,1), "vol_ratio": round(vol_ratio,2), "size_mult": 0.8}
        except Exception:
            return {"regime": "UNKNOWN", "size_mult": 0.8}


class CorrelationFilter:
    GROUPS = {
        "large_cap": ["BTC/USDT","ETH/USDT"],
        "layer1":    ["SOL/USDT","ADA/USDT","AVAX/USDT","DOT/USDT"],
        "defi":      ["LINK/USDT","UNI/USDT","AAVE/USDT"],
        "meme":      ["DOGE/USDT","SHIB/USDT"],
        "exchange":  ["BNB/USDT"],
        "payments":  ["XRP/USDT","XLM/USDT"],
        "layer2":    ["MATIC/USDT","OP/USDT","ARB/USDT"],
    }
    MAX_PER_GROUP = {"large_cap":1,"layer1":2,"defi":1,"meme":1,"exchange":1,"payments":1,"layer2":1}

    def is_allowed(self, symbol, open_trades):
        open_symbols = [t["symbol"] for t in open_trades]
        for group, symbols in self.GROUPS.items():
            if symbol not in symbols:
                continue
            held = sum(1 for s in open_symbols if s in symbols)
            if held >= self.MAX_PER_GROUP.get(group, 2):
                return False, f"Already holding {held} from {group} (max {self.MAX_PER_GROUP.get(group,2)})"
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
                        breadth=breadth, vol_ratio=vol_ratio, adx=adx_val)

        if vol_ratio > 2.0:
            return dict(regime="HIGH_VOLATILITY", gate=False,
                        allow_longs=False, allow_shorts=False,
                        min_conf=0.70, size_mult=0.3,
                        breadth=breadth, vol_ratio=vol_ratio, adx=adx_val)

        if adx_val > 25 and (btc_bullish or btc_bearish) and (breadth > 0.60 or bear_breadth > 0.60):
            return dict(regime="STRONG_TREND", gate=True,
                        allow_longs=True, allow_shorts=True,
                        min_conf=0.45, size_mult=1.2,
                        breadth=breadth, vol_ratio=vol_ratio, adx=adx_val)

        if adx_val < 20 and vol_ratio < 0.8 and 0.30 < breadth < 0.60:
            return dict(regime="RANGING", gate=True,
                        allow_longs=True, allow_shorts=True,
                        min_conf=0.58, size_mult=0.6,
                        breadth=breadth, vol_ratio=vol_ratio, adx=adx_val)

        return dict(regime="WEAK_TREND", gate=True,
                    allow_longs=True, allow_shorts=True,
                    min_conf=0.45, size_mult=0.9,
                    breadth=breadth, vol_ratio=vol_ratio, adx=adx_val)

    def _neutral(self) -> dict:
        return dict(regime="UNKNOWN", gate=True,
                    allow_longs=True, allow_shorts=True,
                    min_conf=0.52, size_mult=0.8,
                    breadth=0.5, vol_ratio=1.0, adx=25.0)

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
        self._load()

    def _load(self):
        p = DATA / "trailing_stops.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
                self.peak_prices = d.get("peaks", {})
                self.atr_values  = d.get("atrs", {})
                self.entry_atrs  = d.get("entry_atrs", {})
                self.partial     = d.get("partial", {})

    def _save(self):
        with open(DATA / "trailing_stops.json", "w") as f:
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
        self._save()

    def should_exit(self, trade_id, entry, price, atr, side="long"):
        """Returns (fraction_to_close, reason). fraction=0.0 means hold."""
        self.update(trade_id, price, atr, side)
        is_short  = side in ("short", "sell")
        entry_atr = self.entry_atrs.get(trade_id, atr)
        partial   = self.partial.get(trade_id, {"tier1": False, "tier2": False})
        gain      = (entry - price) if is_short else (price - entry)

        # Tier 1: +1× entry ATR → take 40%
        if not partial["tier1"] and gain >= entry_atr:
            self.partial[trade_id]["tier1"] = True
            self._save()
            return 0.4, "PARTIAL_TP1 +1×ATR"

        # Tier 2: +2× entry ATR → take 50% of remaining (30% of original)
        if partial["tier1"] and not partial["tier2"] and gain >= 2 * entry_atr:
            self.partial[trade_id]["tier2"] = True
            self._save()
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

    def cleanup(self, trade_id):
        self.peak_prices.pop(trade_id, None)
        self.atr_values.pop(trade_id, None)
        self.entry_atrs.pop(trade_id, None)
        self.partial.pop(trade_id, None)
        self._save()


class KellyCriterionSizer:
    KELLY_FRACTION = 0.25
    MIN_PCT        = 0.02
    MAX_PCT        = 0.15
    BASE_PCT       = 0.05

    def calculate(self, confidence, balance, price, atr_pct, regime, recent_trades, all_agree=False):
        closed = [t for t in recent_trades if t.get("status") == "closed"]
        if len(closed) < 10:
            pos_pct = self.BASE_PCT * (0.6 + confidence * 0.8)
        else:
            last_20 = closed[-20:]
            wins    = [t["pnl"] for t in last_20 if t.get("pnl", 0) > 0]
            losses  = [abs(t["pnl"]) for t in last_20 if t.get("pnl", 0) <= 0]
            if wins and losses:
                wr      = len(wins) / len(last_20)
                avg_win = np.mean(wins)
                avg_los = np.mean(losses)
                kelly   = (wr * avg_win - (1-wr) * avg_los) / (avg_win + 1e-9)
                pos_pct = max(0, kelly) * self.KELLY_FRACTION * (0.5 + confidence * 0.5)
            else:
                pos_pct = self.BASE_PCT

        if atr_pct > 0.04:    pos_pct *= 0.6
        elif atr_pct > 0.02:  pos_pct *= 0.8
        pos_pct *= regime.get("size_mult", 1.0)

        recent_5 = [t for t in recent_trades[-5:] if t.get("status") == "closed"]
        if len(recent_5) >= 3:
            if sum(1 for t in recent_5 if t.get("pnl", 0) <= 0) >= 3:
                pos_pct *= 0.5
                log.warning(f"Losing streak — reducing size to {pos_pct*100:.1f}%")

        # Tiered cap: scale up only when high-conviction setup
        if all_agree and confidence >= 0.70 and regime.get("regime") == "STRONG_TREND":
            cap = 0.15
        elif all_agree and confidence >= 0.60:
            cap = 0.12
        else:
            cap = 0.10
        pos_pct = max(self.MIN_PCT, min(cap, pos_pct))
        usdt    = balance * pos_pct
        amount  = usdt / price
        log.info(f"Kelly size: {pos_pct*100:.1f}% cap={cap*100:.0f}% (conf={confidence:.2f} agree={all_agree} regime={regime.get('regime','?')})")
        return amount, usdt


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
        new_heat = heat + new_usdt / (balance + 1e-9)
        if new_heat > self.max_heat:
            return False, f"Portfolio heat {heat*100:.1f}% (max {self.max_heat*100:.0f}%)"
        return True, f"Heat OK ({heat*100:.1f}%)"


class CircuitBreaker:
    def __init__(self, config):
        self.max_daily_loss  = config.get("max_daily_loss_pct", 0.05)
        self.max_consec_loss = config.get("max_consecutive_losses", 4)
        self._load()

    def _load(self):
        p = DATA / "circuit_breaker.json"
        if p.exists():
            with open(p) as f:
                d = json.load(f)
                self.daily_pnl     = d.get("daily_pnl", 0.0)
                self.consec_losses = d.get("consec_losses", 0)
                self.reset_date    = d.get("reset_date", str(date.today()))
        else:
            self.daily_pnl = 0.0; self.consec_losses = 0
            self.reset_date = str(date.today())

    def _save(self):
        with open(DATA / "circuit_breaker.json", "w") as f:
            json.dump({"daily_pnl": self.daily_pnl,
                       "consec_losses": self.consec_losses,
                       "reset_date": self.reset_date}, f)

    def _reset_if_new_day(self):
        if str(date.today()) != self.reset_date:
            self.daily_pnl = 0.0; self.consec_losses = 0
            self.reset_date = str(date.today())
            self._save()

    def record_trade(self, pnl, balance):
        self._reset_if_new_day()
        self.daily_pnl += pnl
        self.consec_losses = self.consec_losses + 1 if pnl < 0 else 0
        self._save()

    def can_trade(self, balance):
        self._reset_if_new_day()
        if self.daily_pnl < -(balance * self.max_daily_loss):
            return False, f"Daily loss limit: ${self.daily_pnl:.2f}"
        if self.consec_losses >= self.max_consec_loss:
            return False, f"Consecutive losses: {self.consec_losses}"
        return True, "OK"


class RiskManager:
    def __init__(self, config):
        risk             = config.get("risk", {})
        self.regime      = RegimeDetector()
        self.market_gate = MarketRegimeGate()
        self.correlation = CorrelationFilter()
        self.trailing    = ATRTrailingStop(risk.get("stop_loss_atr_multiplier", 2.0))
        self.sizer       = KellyCriterionSizer()
        self.breaker     = CircuitBreaker(risk)
        self.heat        = PortfolioHeatTracker(risk.get("max_portfolio_heat", 0.40))
        self.take_profit = risk.get("take_profit_pct", 0.05)

    def check_exits(self, open_trades, get_price_fn, get_atr_fn):
        exits = []
        for trade in open_trades:
            price = get_price_fn(trade["symbol"])
            if not price:
                continue
            entry    = float(trade["price"])
            trade_id = trade["id"]
            side     = trade.get("side", "long")
            is_short = side in ("short", "sell")

            if is_short:
                if price <= entry * (1 - self.take_profit):
                    exits.append((trade, price, "TAKE_PROFIT", 1.0))
                    continue
            else:
                if price >= entry * (1 + self.take_profit):
                    exits.append((trade, price, "TAKE_PROFIT", 1.0))
                    continue

            atr = get_atr_fn(trade["symbol"])
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

    def get_position_size(self, confidence, balance, price, df, recent_trades, portfolio_mult=1.0, all_agree=False):
        regime  = self.regime.detect(df)
        atr     = df["high"].sub(df["low"]).rolling(14).mean().iloc[-1]
        atr_pct = float(atr / (df["close"].iloc[-1] + 1e-9))
        regime["size_mult"] = regime.get("size_mult", 1.0) * portfolio_mult
        log.info(f"Regime: {regime['regime']} ADX={regime.get('adx','?')} portfolio_mult={portfolio_mult:.2f}")
        return self.sizer.calculate(confidence, balance, price, atr_pct, regime, recent_trades, all_agree=all_agree)

    def record_trade_result(self, pnl, balance):
        self.breaker.record_trade(pnl, balance)

    def cleanup_trade(self, trade_id):
        self.trailing.cleanup(trade_id)
