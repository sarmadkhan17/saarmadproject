from datetime import datetime, timezone
from core.tz import LOCAL_TZ
from engine.smc_agent import AgentSignal

_TREND_TO_NET = {
    "STRONG_BULL": 0.7,
    "MILD_BULL":   0.4,
    "NEUTRAL":     0.0,
    "MILD_BEAR":  -0.4,
    "STRONG_BEAR": -0.7,
}


class MacroFlowAgent:
    TTL = 7200  # 2 hours — matches AgentCoordinator.SLOW_CACHE_SECS

    def __init__(self, macro_agent):
        self.macro_agent = macro_agent
        self._cache = None
        self._cache_time = datetime.min.replace(tzinfo=LOCAL_TZ)

    def analyze(self, df, profile, market_ctx=None) -> AgentSignal:
        now = datetime.now(LOCAL_TZ)
        if (self._cache is None or
                (now - self._cache_time).total_seconds() > self.TTL):
            self._cache = self.macro_agent.analyze()
            self._cache_time = now

        trend = self._cache.get("market_trend", "NEUTRAL")
        net_score = _TREND_TO_NET.get(trend, 0.0)
        confidence = abs(net_score) * 0.8 + 0.2

        return AgentSignal(
            agent="macro_flow",
            buy_score=max(0.0, net_score),
            sell_score=max(0.0, -net_score),
            net_score=net_score,
            confidence=confidence,
            reasoning=f"Macro: {trend}",
        )
