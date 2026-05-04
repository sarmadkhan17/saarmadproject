"""
GNN Coin Correlation Filter.
Builds return-correlation graph from OHLCV data.
GCN-style message passing aggregates neighbor features into node embeddings.
Risk score = average correlation of target coin with open positions.
Integrates into RiskManager as additional correlation filter.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

log = logging.getLogger("GNNCorr")

EDGE_THRESHOLD  = 0.65   # |Pearson r| above this → graph edge
RISK_THRESHOLD  = 0.60   # avg correlation with open positions → block
N_GCN_LAYERS   = 2


def _normalized_adj(adj: np.ndarray) -> np.ndarray:
    """D^{-1/2} A D^{-1/2} symmetric normalisation."""
    deg          = adj.sum(axis=1)
    deg_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg + 1e-9), 0.0)
    D            = np.diag(deg_inv_sqrt)
    return D @ adj @ D


def _gcn_pass(H: np.ndarray, A_norm: np.ndarray, n_layers: int = 2) -> np.ndarray:
    """n_layers of ReLU(A_norm @ H) — identity weights, pure aggregation."""
    emb = H.copy()
    for _ in range(n_layers):
        emb = np.maximum(0.0, A_norm @ emb)
    return emb


class GNNCorrelationFilter:
    """
    Correlation graph over watched coins.
    Nodes: coins. Edges: |return correlation| > EDGE_THRESHOLD.
    Node features: [mean_return_norm, volatility_norm, degree_norm].
    GCN aggregates neighbourhood info into embeddings.
    Risk score: mean |correlation| between target and open-position nodes.
    """

    def __init__(
        self,
        edge_threshold: float = EDGE_THRESHOLD,
        risk_threshold: float = RISK_THRESHOLD,
    ):
        self.edge_threshold = edge_threshold
        self.risk_threshold = risk_threshold

        self._symbols:    List[str]           = []
        self._corr:       Optional[np.ndarray] = None  # (N, N) raw Pearson
        self._adj:        Optional[np.ndarray] = None  # (N, N) thresholded |corr|
        self._embeddings: Optional[np.ndarray] = None  # (N, D)

    # ── Graph construction ─────────────────────────────────────────────────────

    def update_graph(self, ohlcv_data: Dict[str, pd.DataFrame], lookback: int = 168):
        """
        Rebuild graph from latest OHLCV.
        ohlcv_data: {symbol → DataFrame with 'close' column}
        lookback:   bars of history to use (default 168 = 1 week of 1h)
        """
        if len(ohlcv_data) < 2:
            return
        try:
            symbols  = sorted(ohlcv_data.keys())
            min_bars = min(len(df) for df in ohlcv_data.values())
            use      = min(min_bars - 1, lookback)

            if use < 20:
                log.debug("GNN: too few bars for correlation")
                return

            # Log-return matrix (T, N)
            ret_cols = []
            for sym in symbols:
                close = ohlcv_data[sym]["close"].tail(use + 1)
                ret   = np.log(close / close.shift(1)).dropna().values[-use:]
                ret_cols.append(ret)

            # Align lengths (may differ by 1 due to dropna)
            min_len = min(len(r) for r in ret_cols)
            R       = np.column_stack([r[-min_len:] for r in ret_cols])  # (T, N)

            # Pearson correlation matrix (N, N)
            corr = np.corrcoef(R.T)
            np.fill_diagonal(corr, 0.0)
            corr = np.nan_to_num(corr, nan=0.0)

            # Adjacency: weighted by |corr|, zeroed below threshold
            adj = np.where(np.abs(corr) >= self.edge_threshold, np.abs(corr), 0.0)
            np.fill_diagonal(adj, 0.0)

            # Node features (N, 3)
            mean_ret = R.mean(axis=0)
            vol      = R.std(axis=0)
            degree   = adj.sum(axis=1)
            max_deg  = max(degree.max(), 1e-9)

            H = np.column_stack([
                np.clip(mean_ret * 100, -1.0, 1.0),   # normalised mean return
                np.clip(vol      * 100,  0.0, 1.0),   # normalised volatility
                degree / max_deg,                      # normalised degree
            ])

            A_norm           = _normalized_adj(adj)
            self._symbols    = symbols
            self._corr       = corr
            self._adj        = adj
            self._embeddings = _gcn_pass(H, A_norm, N_GCN_LAYERS)

            n_edges = int((adj > 0).sum() // 2)
            log.debug(f"GNN graph: {len(symbols)} nodes, {n_edges} edges, "
                      f"avg|corr|={np.abs(corr[corr!=0]).mean():.2f}")

        except Exception as e:
            log.warning(f"GNN graph update failed: {e}")

    # ── Risk scoring ──────────────────────────────────────────────────────────

    def _idx(self, symbol: str) -> Optional[int]:
        try:
            return self._symbols.index(symbol)
        except ValueError:
            return None

    def compute_risk_score(
        self, target_symbol: str, open_symbols: List[str]
    ) -> float:
        """
        Average |correlation| of target with open-position symbols.
        0 = uncorrelated, 1 = perfectly correlated.
        """
        if self._corr is None or len(self._symbols) < 2:
            return 0.0
        t_idx = self._idx(target_symbol)
        if t_idx is None:
            return 0.0
        open_idxs = [
            self._idx(s) for s in open_symbols
            if s != target_symbol and self._idx(s) is not None
        ]
        if not open_idxs:
            return 0.0
        return float(np.mean([abs(float(self._corr[t_idx, i])) for i in open_idxs]))

    def check(
        self, symbol: str, open_symbols: List[str]
    ) -> Tuple[bool, str, float]:
        """
        Returns (allowed, reason, risk_score).
        allowed=False blocks the trade.
        """
        if self._corr is None:
            return True, "GNN not built", 0.0
        score = self.compute_risk_score(symbol, open_symbols)
        if score >= self.risk_threshold:
            return (
                False,
                f"GNN corr risk {score:.2f} >= {self.risk_threshold:.2f}",
                score,
            )
        return True, f"GNN OK (score={score:.2f})", score

    def get_most_correlated(self, symbol: str, top_n: int = 3) -> List[Tuple[str, float]]:
        """Return top-N most correlated coins to symbol (for logging)."""
        if self._corr is None:
            return []
        idx = self._idx(symbol)
        if idx is None:
            return []
        row   = [(self._symbols[i], abs(float(self._corr[idx, i])))
                 for i in range(len(self._symbols)) if i != idx]
        row.sort(key=lambda x: x[1], reverse=True)
        return row[:top_n]

    def graph_stats(self) -> dict:
        if self._adj is None:
            return {"status": "not_built"}
        n_edges  = int((self._adj > 0).sum() // 2)
        abs_corr = np.abs(self._corr[self._corr != 0])
        return {
            "n_nodes":  len(self._symbols),
            "n_edges":  n_edges,
            "avg_corr": round(float(abs_corr.mean()), 3) if len(abs_corr) else 0,
            "max_corr": round(float(abs_corr.max()),  3) if len(abs_corr) else 0,
        }
