# MacroFlowAgent Integration Design

**Date:** 2026-05-14  
**Status:** Approved

## Problem

`EnsembleEngine.BASE_AGENT_WEIGHTS` allocates 25% to `macro_flow`, but `macro_agent=None` is passed at construction, leaving that weight unused. The two active agents (smc, technical) are implicitly re-normalized to 46.7% / 53.3% instead of their stated 35% / 40%, which is misleading and wastes a valid signal source.

`MacroAgent` already exists in `agents/coordinator.py` and fetches global market data (BTC dominance, 24h market cap change) from CoinGecko. It just needs a thin adapter to return an `AgentSignal`.

## Design

### New file: `engine/macro_agent.py`

A self-contained wrapper that:
- Holds a reference to the existing `MacroAgent` instance (`self.agents.macro`)
- Caches the HTTP result for 2 hours (matching `AgentCoordinator.SLOW_CACHE_SECS`)
- Maps `market_trend` → `net_score` via a fixed lookup table
- Derives `confidence` from `abs(net_score)` (range 0.2–0.76)
- Returns an `AgentSignal` with `agent="macro_flow"`

**Trend mapping:**

| market_trend  | net_score |
|---------------|-----------|
| STRONG_BULL   | +0.7      |
| MILD_BULL     | +0.4      |
| NEUTRAL       |  0.0      |
| MILD_BEAR     | -0.4      |
| STRONG_BEAR   | -0.7      |

**Confidence formula:** `abs(net_score) * 0.8 + 0.2`  
Produces 0.20 (NEUTRAL) → 0.76 (STRONG_BULL/BEAR).

**analyze signature:** `analyze(self, df, profile) -> AgentSignal`  
Matches what `EnsembleEngine` passes to non-SMC agents. `df` and `profile` are unused — macro is symbol-agnostic.

### Change: `engine/bot.py`

In `BaseBot.__init__`, replace:
```python
self.ensemble = EnsembleEngine(self.smc_agent, self.ml_agent, None)
```
With:
```python
from engine.macro_agent import MacroFlowAgent
macro_flow_agent = MacroFlowAgent(self.agents.macro)
self.ensemble = EnsembleEngine(self.smc_agent, self.ml_agent, macro_flow_agent)
```

`self.agents` is already constructed two lines above, so `self.agents.macro` is available.

### No changes: `engine/ensemble.py`

`BASE_AGENT_WEIGHTS` already includes `macro_flow: 0.25`. The `run()` method already conditionally includes macro when non-None. `_regime_weights()` already adjusts macro weight per regime (e.g., 40% in HIGH_VOL/CRASH).

## Caching

`MacroFlowAgent` manages its own TTL cache (2 hours). This avoids coupling to `AgentCoordinator` internals and prevents repeated CoinGecko HTTP calls when the ensemble processes multiple symbols per cycle.

## Files Changed

| File | Change |
|------|--------|
| `engine/macro_agent.py` | New — `MacroFlowAgent` wrapper |
| `engine/bot.py` | Wire `MacroFlowAgent` into `EnsembleEngine` |
