"""
Test that RiskDecisionAgent rejects entry when the ensemble ran with one or more
agent failures (agents_ok=False on EnsembleResult).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import MagicMock


def test_risk_agent_rejects_when_agents_ok_false():
    """RiskDecisionAgent must reject entry when ensemble reports a partial failure."""
    from engine.risk_agent import RiskDecisionAgent
    from engine.ensemble import EnsembleResult

    agent = RiskDecisionAgent.__new__(RiskDecisionAgent)
    agent.risk = MagicMock()
    agent.gnn  = MagicMock()
    agent.hmm_model = None

    result = EnsembleResult(
        action='BUY',
        confidence=0.65,
        net_score=0.45,
        buy_score=0.65,
        sell_score=0.20,
        agents_agreeing=1,
        agents_total=2,
        agents_ok=False,
    )

    decision = agent._check_agents_ok(result)
    assert decision is not None, "_check_agents_ok must return a RiskDecision when agents_ok=False"
    assert decision.approved == False
    assert any('partial ensemble' in r.lower() for r in decision.reasons)


def test_risk_agent_passes_when_agents_ok_true():
    """Guard returns None (no block) when all agents succeeded."""
    from engine.risk_agent import RiskDecisionAgent
    from engine.ensemble import EnsembleResult

    agent = RiskDecisionAgent.__new__(RiskDecisionAgent)
    agent.risk = MagicMock()
    agent.gnn  = MagicMock()
    agent.hmm_model = None

    result = EnsembleResult(
        action='BUY',
        confidence=0.65,
        net_score=0.45,
        buy_score=0.65,
        sell_score=0.20,
        agents_agreeing=2,
        agents_total=2,
        agents_ok=True,
    )

    decision = agent._check_agents_ok(result)
    assert decision is None, "_check_agents_ok must return None when agents_ok=True"


def test_ensemble_sets_agents_ok_false_on_partial_failure():
    """EnsembleResult.agents_ok must be False when at least one agent errored."""
    from engine.ensemble import EnsembleEngine, EnsembleResult
    import pandas as pd, numpy as np

    # Build a minimal df
    n = 50
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    df = pd.DataFrame({
        "open": np.random.uniform(100, 110, n),
        "high": np.random.uniform(110, 120, n),
        "low":  np.random.uniform(90, 100, n),
        "close": np.random.uniform(100, 110, n),
        "volume": np.random.uniform(1000, 5000, n),
    }, index=idx)

    # SMC always raises
    bad_smc = MagicMock()
    bad_smc.analyze = MagicMock(side_effect=RuntimeError("simulated smc failure"))

    # Technical succeeds
    ok_tech = MagicMock()
    sig = MagicMock()
    sig.net_score = 0.3
    sig.buy_score = 0.6
    sig.sell_score = 0.2
    sig.confidence = 0.60
    sig.agent = "technical"
    sig.reasoning = "ok"
    ok_tech.analyze = MagicMock(return_value=sig)

    engine = EnsembleEngine(smc_agent=bad_smc, tech_agent=ok_tech)

    profile = MagicMock()
    profile.net_score_threshold = 0.25

    result = engine.run("TEST/USDT", df, profile, market_ctx={"vol_ratio": 1.2})

    assert result.agents_ok == False, "agents_ok must be False when smc agent failed"
