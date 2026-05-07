"""
Ensemble Engine — runs all strategy agents in parallel, produces weighted consensus.
"""

import logging
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
import pandas as pd

log = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    action: str           # BUY / SELL / HOLD
    confidence: float
    net_score: float
    buy_score: float
    sell_score: float
    agents_agreeing: int
    agents_total: int
    signals: list = field(default_factory=list)
    source: str = "ensemble"


class EnsembleEngine:
    AGENT_WEIGHTS = {"smc": 0.35, "technical": 0.40, "macro_flow": 0.25}

    def __init__(self, smc_agent, tech_agent, macro_agent=None):
        self.smc     = smc_agent
        self.tech    = tech_agent
        self.macro   = macro_agent

    def run(self, symbol: str, df: pd.DataFrame, profile) -> EnsembleResult:
        agents = {"smc": self.smc, "technical": self.tech}
        if self.macro is not None:
            agents["macro_flow"] = self.macro

        signals = []
        with ThreadPoolExecutor(max_workers=len(agents)) as pool:
            futures = {
                pool.submit(agent.analyze, df, profile): name
                for name, agent in agents.items()
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    sig = future.result()
                    signals.append(sig)
                    log.debug(f"Ensemble {symbol} | {name}: net={sig.net_score:+.3f} | {sig.reasoning[:60]}")
                except Exception as e:
                    log.warning(f"Ensemble {symbol} | {name}: error {e}")

        if not signals:
            return EnsembleResult("HOLD", 0.0, 0.0, 0.0, 0.0, 0, 0, signals, "no_agents")

        return self._aggregate(signals, profile)

    def _aggregate(self, signals: list, profile) -> EnsembleResult:
        net = 0.0
        buy_score  = 0.0
        sell_score = 0.0
        total_w = 0.0

        for s in signals:
            w = self.AGENT_WEIGHTS.get(s.agent, 0.30)
            net += s.net_score * w
            buy_score  += s.buy_score * w
            sell_score += s.sell_score * w
            total_w += w

        if total_w > 0:
            net       = net / total_w
            buy_score  = buy_score / total_w
            sell_score = sell_score / total_w

        threshold = getattr(profile, 'net_score_threshold', 0.25)
        if net > threshold:
            action = "BUY"
        elif net < -threshold:
            action = "SELL"
        else:
            # Sub-threshold: lean toward strongest signal
            if buy_score > sell_score and buy_score > 0.15:
                action = "BUY"
            elif sell_score > buy_score and sell_score > 0.15:
                action = "SELL"
            else:
                action = "HOLD"

        # Agent agreement
        agreement = sum(1 for s in signals if s.net_score * net > 0)

        # Confidence: use the strongest agent's net_score + agreement bonus
        max_agent_net = max((abs(s.net_score) for s in signals), default=0)
        max_agent_conf = max((s.confidence for s in signals if abs(s.net_score) > 0.01), default=0.35)
        agreement_bonus = 0.15 if agreement >= 2 else 0.08 if agreement >= 1 else 0.0
        confidence = min(0.95, max(0.35, max_agent_net * 1.0 + max_agent_conf * 0.3 + agreement_bonus))

        return EnsembleResult(
            action=action,
            confidence=round(confidence, 4),
            net_score=round(net, 4),
            buy_score=round(buy_score, 4),
            sell_score=round(sell_score, 4),
            agents_agreeing=agreement,
            agents_total=len(signals),
            signals=signals,
        )
