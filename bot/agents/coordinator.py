"""
Multi-Agent System v2 - Production Grade
- Agent performance tracking
- Confidence gate (deterministic ensemble, no LLM)
- Fallback chain: Ensemble → ML
"""

import json
import logging
import re
import requests
from pathlib import Path
from datetime import datetime, timezone
from core.tz import LOCAL_TZ
import numpy as np
from core.config import DATA_DIR

log  = logging.getLogger("Agents")
DATA = DATA_DIR


def macro_decay_weight(staleness_seconds: float) -> float:
    """Linear decay of macro signal weight: full < 1800s, zero > 3600s."""
    if staleness_seconds <= 1800.0:
        return 1.0
    if staleness_seconds >= 3600.0:
        return 0.0
    return 1.0 - (staleness_seconds - 1800.0) / 1800.0


class AgentPerformanceTracker:
    def __init__(self):
        self.path = DATA / "agent_performance.json"
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as f:
                return json.load(f)
        return {k: {"correct": 0, "total": 0}
                for k in ["technical","fear_greed","news","onchain","macro","master"]}

    def _save(self):
        DATA.mkdir(exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.data, f, indent=2)

    def record_prediction(self, agent, action, pnl):
        if agent not in self.data:
            self.data[agent] = {"correct": 0, "total": 0}
        self.data[agent]["total"] += 1
        if (action == "BUY" and pnl > 0) or (action == "SELL" and pnl > 0):
            self.data[agent]["correct"] += 1
        self._save()

    def get_accuracy(self, agent):
        d = self.data.get(agent, {"correct": 0, "total": 0})
        return d["correct"] / d["total"] if d["total"] >= 5 else 0.5

    def get_report(self):
        return {
            a: {"accuracy": round(s["correct"]/s["total"]*100, 1),
                "total": s["total"], "correct": s["correct"]}
            for a, s in self.data.items() if s["total"] > 0
        }


class TokenBudgetManager:
    DAILY_LIMIT   = 90000
    COST_PER_CALL = 200

    def __init__(self):
        self.path       = DATA / "token_budget.json"
        self.used_today = 0
        self.reset_date = str(datetime.now(LOCAL_TZ).date())
        self._load()

    def _load(self):
        if self.path.exists():
            with open(self.path) as f:
                d = json.load(f)
                self.used_today = d.get("used_today", 0)
                self.reset_date = d.get("reset_date", str(datetime.now(LOCAL_TZ).date()))
        today = str(datetime.now(LOCAL_TZ).date())
        if today != self.reset_date:
            self.used_today = 0
            self.reset_date = today
            self._save()

    def _save(self):
        DATA.mkdir(exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({"used_today": self.used_today,
                       "reset_date": self.reset_date,
                       "limit": self.DAILY_LIMIT}, f)

    def can_call(self):
        self._load()
        return self.used_today + self.COST_PER_CALL <= self.DAILY_LIMIT

    def record_call(self):
        self.used_today += self.COST_PER_CALL
        self._save()

    def get_usage_pct(self):
        return self.used_today / self.DAILY_LIMIT * 100

    def tokens_remaining(self):
        return max(0, self.DAILY_LIMIT - self.used_today)


class TechnicalAgent:
    def analyze(self, symbol, df, ai_signal):
        try:
            close     = df["close"]
            last      = df.iloc[-1]
            prev      = df.iloc[-2]
            price     = float(last["close"])
            ch1h      = (price - float(prev["close"])) / float(prev["close"]) * 100
            ch24h     = (price - float(df.iloc[-24]["close"])) / float(df.iloc[-24]["close"]) * 100 if len(df) > 24 else 0
            vol_ratio = float(last["volume"] / df["volume"].rolling(20).mean().iloc[-1])
            ema9      = float(close.ewm(span=9).mean().iloc[-1])
            ema21     = float(close.ewm(span=21).mean().iloc[-1])
            ema50     = float(close.ewm(span=50).mean().iloc[-1])
            trend     = "BULLISH" if ema9 > ema21 > ema50 else "BEARISH" if ema9 < ema21 < ema50 else "NEUTRAL"
            delta     = close.diff()
            rsi       = float(100 - (100 / (1 + delta.clip(lower=0).rolling(14).mean().iloc[-1] /
                        ((-delta.clip(upper=0)).rolling(14).mean().iloc[-1] + 1e-9))))
            rsi_st    = "OVERSOLD" if rsi < 30 else "OVERBOUGHT" if rsi > 70 else "NEUTRAL"
            return {
                "agent": "technical", "symbol": symbol,
                "price": round(price, 4), "change_1h": round(ch1h, 2),
                "change_24h": round(ch24h, 2), "trend": trend,
                "rsi": round(rsi, 1), "rsi_state": rsi_st,
                "vol_ratio": round(vol_ratio, 2),
                "ml_action": ai_signal.get("action", "HOLD"),
                "ml_conf":   ai_signal.get("confidence", 0.5),
            }
        except Exception as e:
            log.error(f"Technical agent error: {e}")
            return {"agent": "technical", "error": str(e)}


class FearGreedAgent:
    URL = "https://api.alternative.me/fng/?limit=2"

    def analyze(self):
        try:
            r     = requests.get(self.URL, timeout=10)
            data  = r.json()["data"]
            value = int(data[0]["value"])
            label = data[0]["value_classification"]
            chg   = value - int(data[1]["value"])
            if value <= 25:   signal = "STRONG_BUY"
            elif value <= 40: signal = "BUY"
            elif value <= 60: signal = "NEUTRAL"
            elif value <= 75: signal = "SELL"
            else:             signal = "STRONG_SELL"
            return {"agent": "fear_greed", "value": value, "label": label,
                    "change": chg, "signal": signal}
        except Exception as e:
            log.error(f"FearGreed error: {e}")
            return {"agent": "fear_greed", "value": 50, "signal": "NEUTRAL"}


class NewsAgent:
    FEEDS = [
        "https://feeds.feedburner.com/CoinDesk",
        "https://cointelegraph.com/rss",
    ]
    BULLISH = {"surge": 2, "rally": 2, "bull": 2, "adoption": 2,
               "breakout": 2, "record": 2, "etf": 3, "institutional": 2}
    BEARISH = {"crash": 3, "hack": 3, "ban": 2, "lawsuit": 2,
               "fraud": 3, "selloff": 2, "dump": 2, "bearish": 2}
    NEGATION = ["not", "no", "never", "without", "despite", "refutes", "denies"]

    def analyze(self, symbol):
        try:
            coin      = symbol.replace("/USDT", "").lower()
            headlines = self._fetch()
            relevant  = [h for h in headlines if coin in h.lower()
                         or "crypto" in h.lower() or "market" in h.lower()][:10]
            if not relevant:
                return {"agent": "news", "signal": "NEUTRAL", "score": 0}
            score = 0
            for h in relevant:
                hl = h.lower()
                for phrase, w in self.BULLISH.items():
                    if phrase in hl:
                        idx = hl.find(phrase)
                        ctx = hl[max(0, idx-30):idx]
                        score += -w if any(n in ctx for n in self.NEGATION) else w
                for phrase, w in self.BEARISH.items():
                    if phrase in hl:
                        idx = hl.find(phrase)
                        ctx = hl[max(0, idx-30):idx]
                        score += w if any(n in ctx for n in self.NEGATION) else -w
            signal = "BULLISH" if score >= 3 else "BEARISH" if score <= -3 else "NEUTRAL"
            return {"agent": "news", "signal": signal, "score": score,
                    "summary": f"{len(relevant)} headlines. Score: {score:+d}"}
        except Exception as e:
            return {"agent": "news", "signal": "NEUTRAL", "score": 0}

    def _fetch(self):
        headlines = []
        for url in self.FEEDS:
            try:
                r      = requests.get(url, timeout=8)
                titles = re.findall(r"<title><!\[CDATA\[(.*?)\]\]></title>", r.text)
                if not titles:
                    titles = re.findall(r"<title>(.*?)</title>", r.text)
                headlines.extend(titles[:10])
            except Exception:
                pass
        return list(set(headlines))


class OnChainAgent:
    COIN_MAP = {
        "BTC": "bitcoin", "ETH": "ethereum", "BNB": "binancecoin",
        "SOL": "solana",  "XRP": "ripple",   "DOGE": "dogecoin",
        "ADA": "cardano", "AVAX": "avalanche-2", "LINK": "chainlink",
        "DOT": "polkadot", "UNI": "uniswap",  "LTC": "litecoin",
    }

    def analyze(self, symbol):
        coin_key = symbol.replace("/USDT", "")
        coin_id  = self.COIN_MAP.get(coin_key, coin_key.lower())
        try:
            url  = (f"https://api.coingecko.com/api/v3/simple/price"
                    f"?ids={coin_id}&vs_currencies=usd"
                    f"&include_24hr_change=true&include_7d_change=true")
            r    = requests.get(url, timeout=10)
            data = r.json()
            if coin_id not in data:
                return {"agent": "onchain", "ath_signal": "UNKNOWN", "summary": "No data"}
            d      = data[coin_id]
            ch24   = d.get("usd_24h_change", 0) or 0
            ch7d   = d.get("usd_7d_change", 0) or 0
            sig    = "STRONG_MOMENTUM" if ch7d > 10 else "POSITIVE" if ch7d > 0 else "BEARISH"
            return {"agent": "onchain", "symbol": symbol,
                    "change_24h": round(float(ch24), 2),
                    "change_7d":  round(float(ch7d), 2),
                    "ath_signal": sig,
                    "summary":    f"{coin_key} 24h:{ch24:+.1f}% 7d:{ch7d:+.1f}% ({sig})"}
        except Exception as e:
            return {"agent": "onchain", "ath_signal": "UNKNOWN", "summary": "Unavailable"}


class MacroAgent:
    def analyze(self):
        try:
            r   = requests.get("https://api.coingecko.com/api/v3/global", timeout=10)
            d   = r.json()["data"]
            btc = d["market_cap_percentage"]["btc"]
            ch  = d.get("market_cap_change_percentage_24h_usd", 0) or 0
            dom = "RISK_OFF" if btc > 60 else "ALT_SEASON" if btc < 50 else "NEUTRAL"
            mkt = "STRONG_BULL" if ch > 3 else "MILD_BULL" if ch > 0 else "MILD_BEAR" if ch > -3 else "STRONG_BEAR"
            return {"agent": "macro", "btc_dominance": round(btc, 1),
                    "dom_signal": dom, "market_trend": mkt,
                    "summary": f"BTC dom {btc:.1f}% ({dom}). Market {ch:+.2f}% ({mkt})"}
        except Exception:
            return {"agent": "macro", "dom_signal": "NEUTRAL", "market_trend": "NEUTRAL"}


class MasterAgent:
    """
    Deterministic ensemble — replaces Groq LLM (37% accuracy).
    Combines technical, fear/greed, news, onchain, and macro signals
    with weighted voting. No API calls, no tokens wasted.
    """

    WEIGHTS = {
        "technical":  0.35,
        "fear_greed": 0.15,
        "news":       0.15,
        "onchain":    0.20,
        "macro":      0.15,
    }

    SIGNAL_MAP = {
        "STRONG_BUY": 1.0, "BUY": 0.6, "BULLISH": 0.5,
        "POSITIVE": 0.3, "STRONG_MOMENTUM": 0.4,
        "NEUTRAL": 0.0,
        "STRONG_SELL": -1.0, "SELL": -0.6, "BEARISH": -0.5,
        "NEGATIVE": -0.3,
    }

    def decide(self, symbol, technical, fear_greed, news, onchain, macro):
        scores = []
        fg = fear_greed or {}
        mc = macro or {}

        tech_score = self._map_signal(technical.get("trend", "NEUTRAL"))
        ml_boost = 0.1 if technical.get("ml_action") == "BUY" else (-0.1 if technical.get("ml_action") == "SELL" else 0)
        scores.append(("technical", tech_score + ml_boost))

        fg_signal = fg.get("signal", "NEUTRAL")
        fg_val = fg.get("value", 50)
        fg_score = self._map_signal(fg_signal)
        if fg_val < 20:
            fg_score = min(fg_score + 0.2, 1.0)
        elif fg_val > 80:
            fg_score = max(fg_score - 0.2, -1.0)
        scores.append(("fear_greed", fg_score))

        news_score = self._map_signal(news.get("signal", "NEUTRAL"))
        news_mag = min(abs(news.get("score", 0)) / 10.0, 0.3)
        news_score = news_score * (0.7 + news_mag)
        scores.append(("news", news_score))

        onchain_score = self._map_signal(onchain.get("ath_signal", "UNKNOWN"))
        ch7d = onchain.get("change_7d", 0)
        if abs(ch7d) > 5:
            onchain_score = onchain_score * 1.2
        scores.append(("onchain", onchain_score))

        macro_score = self._map_signal(mc.get("market_trend", "NEUTRAL"))
        scores.append(("macro", macro_score))

        weighted = sum(
            self.WEIGHTS.get(name, 0.1) * score
            for name, score in scores
        )

        weighted = max(-1.0, min(1.0, weighted))
        if weighted > 0.25:
            action = "BUY"
            conf = 0.55 + abs(weighted) * 0.35
        elif weighted < -0.25:
            action = "SELL"
            conf = 0.55 + abs(weighted) * 0.35
        else:
            action = technical.get("ml_action", "HOLD")
            conf = float(technical.get("ml_conf", 0.5)) * 0.9

        conf = max(0.3, min(0.95, conf))

        return {
            "action": action,
            "confidence": round(conf, 4),
            "reasoning": f"Ensemble={weighted:+.2f}",
            "risk_level": "HIGH" if abs(weighted) > 0.6 else "MEDIUM",
            "source": "ensemble",
        }

    def _map_signal(self, signal: str) -> float:
        return self.SIGNAL_MAP.get(signal.upper(), 0.0)


class AgentCoordinator:
    """
    Coordinates all agents.
    Uses deterministic ensemble (no LLM) to combine signals.
    """

    SLOW_CACHE_SECS = 1800

    def __init__(self):
        self.technical  = TechnicalAgent()
        self.fear_greed = FearGreedAgent()
        self.news       = NewsAgent()
        self.onchain    = OnChainAgent()
        self.macro      = MacroAgent()
        self.master     = MasterAgent()
        self.tracker    = AgentPerformanceTracker()

        self._fg_cache         = None
        self._macro_cache      = None
        self._slow_time        = None
        self._decision_actions = {}

    def _refresh_slow(self):
        now = datetime.now(LOCAL_TZ)
        if (self._slow_time is None or
                (now - self._slow_time).total_seconds() > self.SLOW_CACHE_SECS):
            self._fg_cache    = self.fear_greed.analyze()
            self._macro_cache = self.macro.analyze()
            self._slow_time   = now
            log.info(
                f"Fear&Greed: {self._fg_cache.get('value',50)}/100 | "
                f"Market: {self._macro_cache.get('market_trend','?')}"
            )

    def analyze(self, symbol, df, ml_signal):
        self._refresh_slow()
        now = datetime.now(LOCAL_TZ)

        ml_action  = ml_signal.get("action", "HOLD")
        ml_conf    = ml_signal.get("confidence", 0.5)
        indicators = ml_signal.get("indicators", {})
        buy_votes  = indicators.get("buy_votes", 0)
        sell_votes = indicators.get("sell_votes", 0)
        worth_ensemble = ml_action != "HOLD"

        if not worth_ensemble:
            log.info(
                f"ML-ONLY {symbol}: {ml_action} | "
                f"conf={ml_conf:.2f} | "
                f"votes={buy_votes}B/{sell_votes}S (ensemble skipped)"
            )
            signal = {
                "symbol":     symbol,
                "action":     ml_action,
                "confidence": ml_conf,
                "strategy":   f"ML-ONLY+{ml_signal.get('strategy','')}",
                "timeframe":  "ML",
                "reasoning":  "ML only — ensemble skipped",
                "risk_level": "MEDIUM",
                "source":     "ml_only",
                "indicators": {
                    **ml_signal.get("indicators", {}),
                    "fear_greed":   self._fg_cache.get("value", 50) if self._fg_cache else 50,
                    "market_trend": self._macro_cache.get("market_trend", "NEUTRAL") if self._macro_cache else "NEUTRAL",
                    "news_signal":  "SKIPPED",
                },
                "timestamp": now.isoformat(),
            }
            self._decision_actions[symbol] = ml_action
            return signal

        tech    = self.technical.analyze(symbol, df, ml_signal)
        news    = self.news.analyze(symbol)
        onchain = self.onchain.analyze(symbol)

        decision = self.master.decide(
            symbol=symbol, technical=tech,
            fear_greed=self._fg_cache, news=news,
            onchain=onchain, macro=self._macro_cache,
        )
        log.info(
            f"ENSEMBLE {symbol}: {decision['action']} | "
            f"conf={decision['confidence']:.2f} | "
            f"src={decision['source']} | "
            f"{decision['reasoning'][:50]}"
        )

        signal = {
            "symbol":     symbol,
            "action":     decision["action"],
            "confidence": decision["confidence"],
            "strategy":   f"ENSEMBLE+{ml_signal.get('strategy','')}",
            "timeframe":  "MULTI-AGENT",
            "reasoning":  decision["reasoning"],
            "risk_level": decision["risk_level"],
            "source":     decision["source"],
            "indicators": {
                **ml_signal.get("indicators", {}),
                "fear_greed":   self._fg_cache.get("value", 50) if self._fg_cache else 50,
                "market_trend": self._macro_cache.get("market_trend", "NEUTRAL") if self._macro_cache else "NEUTRAL",
                "news_signal":  news.get("signal", "NEUTRAL"),
                "news_score":   news.get("score", 0),
                "onchain":      onchain.get("ath_signal", "N/A"),
            },
            "timestamp": now.isoformat(),
        }
        self._decision_actions[symbol] = decision["action"]
        return signal

    def record_trade_result(self, symbol, pnl):
        action = self._decision_actions.get(symbol, "HOLD")
        self.tracker.record_prediction("master", action, pnl)

    def invalidate_cache(self):
        """No-op: per-symbol decision cache removed (every scan recomputes)."""
        pass

    def get_performance_report(self):
        return {
            "agent_accuracy":   self.tracker.get_report(),
            "token_usage_pct":  0,
            "tokens_remaining": 999999,
        }
