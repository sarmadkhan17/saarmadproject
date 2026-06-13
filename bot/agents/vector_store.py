"""
Vector store primitives for trade-memory RAG.

This module supplies the embedding backend and the math used by
TradeMemory to do *real* vector retrieval (cosine similarity) over two
separate stores — closed-trade entry contexts and Judge critiques — and
the helpers to rerank candidates by recency and outcome.

Design notes
------------
* Embeddings come from a local sentence-transformers model
  (all-MiniLM-L6-v2, 384-dim). The model is loaded lazily on first use and
  cached as a process-wide singleton.
* If sentence-transformers (or its model download) is unavailable, the
  Embedder degrades to a no-op that returns None. Callers MUST treat a
  None embedding as "semantic match unavailable" and fall back to the
  numeric/categorical feature similarity. A missing optional dependency
  must never crash the live trading loop.
* Vectors are stored in SQLite as raw float32 bytes (see to_blob/from_blob).
  At this scale (hundreds-to-thousands of rows) a brute-force numpy cosine
  is far simpler and fast enough; no external vector DB is warranted.
"""

import logging
import numpy as np
from typing import Optional

log = logging.getLogger("VectorStore")

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # all-MiniLM-L6-v2 output dimensionality


# ─────────────────────────────────────────────────────────────────────────────
# Embedding backend (semantic, optional)
# ─────────────────────────────────────────────────────────────────────────────

class Embedder:
    """Lazy, fault-tolerant wrapper around a sentence-transformers model.

    The interface is intentionally tiny (`encode`) so the backend can be
    swapped (TF-IDF, a hosted API, etc.) without touching callers.
    """

    _instance: Optional["Embedder"] = None

    def __init__(self):
        self._model = None
        self._tried = False      # have we attempted to load the model yet?
        self._available = False  # did the load succeed?

    @classmethod
    def instance(cls) -> "Embedder":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _ensure_model(self):
        if self._tried:
            return
        self._tried = True
        try:
            from sentence_transformers import SentenceTransformer
            log.info(f"Loading embedding model '{EMBED_MODEL_NAME}' …")
            self._model = SentenceTransformer(EMBED_MODEL_NAME)
            self._available = True
            log.info("Embedding model ready.")
        except Exception as e:
            self._available = False
            log.warning(
                f"Semantic embeddings unavailable ({e}); "
                f"retrieval will fall back to feature similarity only."
            )

    @property
    def available(self) -> bool:
        self._ensure_model()
        return self._available

    def encode(self, text: str) -> Optional[np.ndarray]:
        """Return a unit-normalized float32 embedding, or None if unavailable."""
        if not text or not text.strip():
            return None
        self._ensure_model()
        if not self._available:
            return None
        try:
            vec = self._model.encode(
                text, normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(vec, dtype=np.float32)
        except Exception as e:
            log.warning(f"encode() failed: {e}")
            return None

    def encode_batch(self, texts: list) -> list:
        """Encode many texts; returns a list aligned with `texts` (None where empty)."""
        self._ensure_model()
        if not self._available:
            return [None] * len(texts)
        idx = [i for i, t in enumerate(texts) if t and t.strip()]
        out: list = [None] * len(texts)
        if not idx:
            return out
        try:
            vecs = self._model.encode(
                [texts[i] for i in idx],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for j, i in enumerate(idx):
                out[i] = np.asarray(vecs[j], dtype=np.float32)
        except Exception as e:
            log.warning(f"encode_batch() failed: {e}")
        return out


def get_embedder() -> Embedder:
    return Embedder.instance()


# ─────────────────────────────────────────────────────────────────────────────
# BLOB (de)serialization for SQLite storage
# ─────────────────────────────────────────────────────────────────────────────

def to_blob(vec: Optional[np.ndarray]) -> Optional[bytes]:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def from_blob(blob: Optional[bytes]) -> Optional[np.ndarray]:
    if not blob:
        return None
    try:
        return np.frombuffer(blob, dtype=np.float32)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity (vectors are already unit-normalized → dot product)
# ─────────────────────────────────────────────────────────────────────────────

def cosine_matrix(query: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """Cosine similarity of `query` (d,) against each row of `mat` (n, d).

    Inputs from the Embedder are unit-normalized, but we renormalize
    defensively so this also works on arbitrary feature vectors.
    """
    if mat.size == 0:
        return np.zeros((0,), dtype=np.float32)
    q = query.astype(np.float32)
    qn = np.linalg.norm(q) + 1e-9
    mn = np.linalg.norm(mat, axis=1) + 1e-9
    return (mat @ q) / (mn * qn)


# ─────────────────────────────────────────────────────────────────────────────
# Numeric / categorical feature vector for trade *entry* context
# ─────────────────────────────────────────────────────────────────────────────
#
# Numbers do not embed well as text in any backend, so the structural part
# of a setup (side, regime, scores, dominance, microstructure) is matched
# with an explicit feature vector. Each field is mapped onto a roughly
# unit-scaled axis so plain cosine over the vector is meaningful.

# Live regime taxonomy (bot/risk/manager.py MarketRegimeGate). Each name gets its
# own one-hot axis so "similar" trades must share regime. Previously this listed
# only legacy names (TRENDING/VOLATILE/…), so the live regimes — CHOPPY,
# WEAK_TREND, STRONG_TREND, EXHAUSTION_* — all collapsed to the all-zeros vector
# and regime contributed NOTHING to similarity. An unlisted/UNKNOWN regime still
# maps to all-zeros (no regime signal) rather than silently colliding.
_REGIMES = [
    "STRONG_TREND", "WEAK_TREND", "TRENDING", "RANGING", "CHOPPY",
    "VOLATILE", "HIGH_VOLATILITY", "CRASH",
    "EXHAUSTION_TOP", "EXHAUSTION_BOTTOM",
    "BULLISH", "BEARISH", "NEUTRAL",
]
_CVD = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0, "up": 1.0, "down": -1.0}


def feature_vector(
    side: str,
    regime: str,
    ensemble_score: float,
    confidence: float,
    btc_d: float,
    usdt_d: float,
    ob_imbalance: float,
    cvd_direction: str,
    cvd_divergence: bool,
) -> np.ndarray:
    """Build a fixed-layout feature vector for a trade entry context."""
    regime_oh = [1.0 if (regime or "").upper() == r else 0.0 for r in _REGIMES]
    side_axis = 1.0 if (side or "").lower() in ("long", "buy") else -1.0
    cvd_axis = _CVD.get((cvd_direction or "neutral").lower(), 0.0)
    # Order-book imbalance is a positive ratio centered on 1.0; log makes it
    # symmetric around 0, and clipping caps extreme books.
    ob_axis = float(np.clip(np.log(max(ob_imbalance, 1e-3)), -2.0, 2.0)) / 2.0

    feats = [
        side_axis,
        float(np.clip(ensemble_score, -1.0, 1.0)),
        float(np.clip(confidence, 0.0, 1.0)),
        float(np.clip((btc_d - 50.0) / 25.0, -2.0, 2.0)),   # dominance, recentered
        float(np.clip((usdt_d - 5.0) / 5.0, -2.0, 2.0)),
        ob_axis,
        cvd_axis,
        1.0 if cvd_divergence else 0.0,
        *regime_oh,
    ]
    return np.asarray(feats, dtype=np.float32)
