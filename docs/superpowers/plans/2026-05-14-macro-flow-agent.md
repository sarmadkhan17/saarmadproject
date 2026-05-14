# MacroFlowAgent Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the existing `MacroAgent` (CoinGecko global market data) into `EnsembleEngine` as the third agent (`macro_flow`, 25% weight) via a thin `MacroFlowAgent` adapter.

**Architecture:** Create `engine/macro_agent.py` with a `MacroFlowAgent` class that wraps `MacroAgent`, caches the HTTP result for 2 hours, maps `market_trend` → `net_score`, and returns an `AgentSignal`. Then pass it to `EnsembleEngine` in `engine/bot.py`.

**Tech Stack:** Python stdlib (`datetime`), `engine.smc_agent.AgentSignal`, `agents.coordinator.MacroAgent`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `bot/engine/macro_agent.py` | Create | `MacroFlowAgent` wrapper with TTL cache |
| `bot/engine/bot.py` | Modify (lines 425–427) | Instantiate and pass `MacroFlowAgent` to `EnsembleEngine` |
| `bot/tests/test_macro_flow_agent.py` | Create | Unit tests for signal mapping and caching |

---

## Task 1: Create `MacroFlowAgent` with failing tests

**Files:**
- Create: `bot/tests/test_macro_flow_agent.py`

- [ ] **Step 1: Write the failing tests**

```python
# bot/tests/test_macro_flow_agent.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta


def _make_macro_agent(trend):
    """Return a mock MacroAgent whose analyze() returns market_trend=trend."""
    m = MagicMock()
    m.analyze.return_value = {"market_trend": trend, "dom_signal": "NEUTRAL"}
    return m


def test_strong_bull_maps_to_positive_net():
    from engine.macro_agent import MacroFlowAgent
    agent = MacroFlowAgent(_make_macro_agent("STRONG_BULL"))
    sig = agent.analyze(df=None, profile=None)
    assert sig.net_score == pytest.approx(0.7)
    assert sig.buy_score == pytest.approx(0.7)
    assert sig.sell_score == pytest.approx(0.0)
    assert sig.agent == "macro_flow"


def test_strong_bear_maps_to_negative_net():
    from engine.macro_agent import MacroFlowAgent
    agent = MacroFlowAgent(_make_macro_agent("STRONG_BEAR"))
    sig = agent.analyze(df=None, profile=None)
    assert sig.net_score == pytest.approx(-0.7)
    assert sig.sell_score == pytest.approx(0.7)
    assert sig.buy_score == pytest.approx(0.0)


def test_mild_bull_maps_to_0_4():
    from engine.macro_agent import MacroFlowAgent
    agent = MacroFlowAgent(_make_macro_agent("MILD_BULL"))
    sig = agent.analyze(df=None, profile=None)
    assert sig.net_score == pytest.approx(0.4)


def test_mild_bear_maps_to_minus_0_4():
    from engine.macro_agent import MacroFlowAgent
    agent = MacroFlowAgent(_make_macro_agent("MILD_BEAR"))
    sig = agent.analyze(df=None, profile=None)
    assert sig.net_score == pytest.approx(-0.4)


def test_neutral_maps_to_zero():
    from engine.macro_agent import MacroFlowAgent
    agent = MacroFlowAgent(_make_macro_agent("NEUTRAL"))
    sig = agent.analyze(df=None, profile=None)
    assert sig.net_score == pytest.approx(0.0)
    assert sig.confidence == pytest.approx(0.2)


def test_confidence_formula():
    from engine.macro_agent import MacroFlowAgent
    # STRONG_BULL: abs(0.7) * 0.8 + 0.2 = 0.76
    agent = MacroFlowAgent(_make_macro_agent("STRONG_BULL"))
    sig = agent.analyze(df=None, profile=None)
    assert sig.confidence == pytest.approx(0.76)


def test_http_called_once_within_ttl():
    """MacroAgent.analyze() should only be called once per TTL window."""
    from engine.macro_agent import MacroFlowAgent
    mock_macro = _make_macro_agent("MILD_BULL")
    agent = MacroFlowAgent(mock_macro)
    agent.analyze(df=None, profile=None)
    agent.analyze(df=None, profile=None)
    assert mock_macro.analyze.call_count == 1


def test_http_called_again_after_ttl():
    """MacroAgent.analyze() should be called again once TTL expires."""
    from engine.macro_agent import MacroFlowAgent
    mock_macro = _make_macro_agent("MILD_BULL")
    agent = MacroFlowAgent(mock_macro)
    agent.analyze(df=None, profile=None)

    # Force cache expiry by back-dating _cache_time
    agent._cache_time = datetime.now(timezone.utc) - timedelta(seconds=MacroFlowAgent.TTL + 1)
    agent.analyze(df=None, profile=None)
    assert mock_macro.analyze.call_count == 2


def test_unknown_trend_maps_to_zero():
    """Unrecognised trend string should map to net_score=0.0 (NEUTRAL fallback)."""
    from engine.macro_agent import MacroFlowAgent
    agent = MacroFlowAgent(_make_macro_agent("SOMETHING_WEIRD"))
    sig = agent.analyze(df=None, profile=None)
    assert sig.net_score == pytest.approx(0.0)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /root/cryptobot_v3/bot && python -m pytest tests/test_macro_flow_agent.py -v 2>&1 | head -30
```

Expected: `ModuleNotFoundError: No module named 'engine.macro_agent'`

---

## Task 2: Implement `MacroFlowAgent`

**Files:**
- Create: `bot/engine/macro_agent.py`

- [ ] **Step 1: Write the implementation**

```python
# bot/engine/macro_agent.py
from datetime import datetime, timezone
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
        self._cache_time = None

    def analyze(self, df, profile) -> AgentSignal:
        now = datetime.now(timezone.utc)
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
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd /root/cryptobot_v3/bot && python -m pytest tests/test_macro_flow_agent.py -v
```

Expected: all 9 tests PASS

- [ ] **Step 3: Commit**

```bash
git add bot/engine/macro_agent.py bot/tests/test_macro_flow_agent.py
git commit -m "feat: add MacroFlowAgent adapter with 2h TTL cache"
```

---

## Task 3: Wire `MacroFlowAgent` into `EnsembleEngine`

**Files:**
- Modify: `bot/engine/bot.py` lines 425–427

- [ ] **Step 1: Add import and replace the ensemble constructor call**

In `bot/engine/bot.py`, find:
```python
        self.ensemble = EnsembleEngine(
            self.smc_agent, self.ml_agent, None  # Macro/Flow deferred
        )
```

Replace with:
```python
        from engine.macro_agent import MacroFlowAgent
        self.ensemble = EnsembleEngine(
            self.smc_agent, self.ml_agent, MacroFlowAgent(self.agents.macro)
        )
```

(`self.agents` is constructed at line 356, before this point.)

- [ ] **Step 2: Verify no import errors**

```bash
cd /root/cryptobot_v3/bot && python -c "
import sys; sys.path.insert(0, '.')
from engine.macro_agent import MacroFlowAgent
from agents.coordinator import MacroAgent
w = MacroFlowAgent(MacroAgent())
print('Import OK, TTL =', w.TTL)
"
```

Expected output: `Import OK, TTL = 7200`

- [ ] **Step 3: Run full test suite to check for regressions**

```bash
cd /root/cryptobot_v3/bot && python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all tests PASS (no new failures)

- [ ] **Step 4: Commit**

```bash
git add bot/engine/bot.py
git commit -m "feat: wire MacroFlowAgent into EnsembleEngine — restores 3-agent 35/40/25 weights"
```
