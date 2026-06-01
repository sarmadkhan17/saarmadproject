"""
CorrelationCheck — lightweight replacement for the removed GNNCorrelationFilter.

The GNN graph approach was part of the ML stack being removed. Correlation
risk is already handled by risk/manager.py CorrelationFilter (group caps).
This stub satisfies risk_agent's self.gnn.check() interface with a simple
pass-through, since the real correlation gate runs inside RiskManager.can_open_trade().
"""

import logging

log = logging.getLogger("CorrelationCheck")


class CorrelationCheck:
    """Drop-in for the old GNN filter. check() always passes — the real
    correlation limits are enforced by RiskManager.correlation (group caps)."""

    def check(self, symbol: str, open_symbols: list) -> tuple:
        # Real correlation enforcement happens in RiskManager.can_open_trade
        # via CorrelationFilter group caps. This is a no-op pass.
        return True, "ok", 0.0

    def update_graph(self, *args, **kwargs):
        pass

    def graph_stats(self) -> dict:
        return {"n_nodes": 0, "n_edges": 0, "avg_corr": 0.0}
