"""
TradeMemory — persistent store of all trades with full entry context.

Every closed trade is stored with:
  - The full signal state at entry
  - The outcome (PnL, R-multiple, exit reason)
  - The Judge critique (if available)

When the Actor evaluates a new setup, TradeMemory finds the 5 most
similar past trades using simple feature matching (no vector DB needed
at this scale — cosine similarity on a small feature vector).

Schema (SQLite, one row per closed trade):
  id, symbol, side, mode, entry_price, close_price, pnl, r_multiple,
  duration_hours, close_reason, regime, btc_d, usdt_d, ensemble_score,
  confidence, ob_imbalance, cvd_direction, cvd_divergence,
  actor_reasoning, judge_critique, pattern_tag, closed_at
"""

import json
import logging
import sqlite3
import numpy as np
from datetime import datetime, timezone
from core.tz import LOCAL_TZ
from pathlib import Path
from typing import Optional

from agents import vector_store as vs

log = logging.getLogger("TradeMemory")


# ── Text builders for the semantic stores ───────────────────────────────────
# These render an entry context (or critique) into the short natural-language
# string that gets embedded. The trade/query text deliberately EXCLUDES the
# outcome so a live setup (no outcome yet) matches stored setups on entry
# conditions; outcome is used only for reranking.

def entry_text(symbol: str, side: str, ctx: dict) -> str:
    div = " with CVD divergence" if ctx.get("cvd_divergence") else ""
    return (
        f"{side} {symbol} in a {ctx.get('regime','?')} regime. "
        f"Ensemble score {float(ctx.get('ensemble_score',0)):+.2f}, "
        f"confidence {float(ctx.get('confidence',0)):.2f}. "
        f"BTC dominance {float(ctx.get('btc_d',50)):.1f}%, "
        f"USDT dominance {float(ctx.get('usdt_d',5)):.1f}%. "
        f"Order-book imbalance {float(ctx.get('ob_imbalance',1)):.2f}, "
        f"CVD {ctx.get('cvd_direction','neutral')}{div}. "
        f"{ctx.get('actor_reasoning','')}"
    ).strip()


def critique_text(critique: dict) -> str:
    return (
        f"{critique.get('symbol','?')} "
        f"{critique.get('decision_quality','')} decision, "
        f"pattern {critique.get('pattern_tag','?')}. "
        f"Missed warnings: {critique.get('missed_warnings','') or 'none'}. "
        f"Lesson: {critique.get('lesson','')}"
    ).strip()


class TradeMemory:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def _init_db(self):
        with self._conn() as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT, side TEXT, mode TEXT,
                entry_price REAL, close_price REAL,
                pnl REAL, r_multiple REAL, duration_hours REAL,
                close_reason TEXT,
                regime TEXT, btc_d REAL, usdt_d REAL,
                btc_d_roc REAL, usdt_d_roc REAL,
                ensemble_score REAL, confidence REAL,
                ob_imbalance REAL, cvd_direction TEXT, cvd_divergence INTEGER,
                actor_reasoning TEXT, actor_approved INTEGER,
                judge_critique TEXT, pattern_tag TEXT,
                closed_at TEXT
            )""")
            c.execute("""
            CREATE TABLE IF NOT EXISTS judge_critiques (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT, symbol TEXT, outcome TEXT, pnl REAL,
                decision_quality TEXT, entry_valid INTEGER,
                risk_managed INTEGER, missed_warnings TEXT,
                lesson TEXT, pattern_tag TEXT, reviewed_at TEXT
            )""")
            c.execute("""
            CREATE TABLE IF NOT EXISTS meta_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rules_json TEXT, synthesized_at TEXT
            )""")
            # Migration: add embedding BLOB columns to existing DBs.
            self._ensure_column(c, "trades", "embedding", "BLOB")
            self._ensure_column(c, "judge_critiques", "embedding", "BLOB")

    @staticmethod
    def _ensure_column(c, table: str, col: str, decl: str):
        cols = [r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()]
        if col not in cols:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    def record_trade(self, trade: dict, entry_context: dict, r_multiple: float = 0.0):
        """Store a closed trade with its full entry context + entry embedding."""
        try:
            side = trade.get("side", "")
            # Embed the entry conditions (outcome excluded — see entry_text).
            emb = vs.get_embedder().encode(
                entry_text(trade.get("symbol", ""), side, entry_context)
            )
            with self._conn() as c:
                c.execute("""
                INSERT OR REPLACE INTO trades (
                    id, symbol, side, mode, entry_price, close_price,
                    pnl, r_multiple, duration_hours, close_reason,
                    regime, btc_d, usdt_d, btc_d_roc, usdt_d_roc,
                    ensemble_score, confidence, ob_imbalance,
                    cvd_direction, cvd_divergence, actor_reasoning,
                    actor_approved, judge_critique, pattern_tag,
                    closed_at, embedding
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    trade.get("id", ""),
                    trade.get("symbol", ""),
                    side,
                    trade.get("mode", "spot"),
                    float(trade.get("price", 0)),
                    float(trade.get("close_price", 0)),
                    float(trade.get("pnl", 0)),
                    round(r_multiple, 3),
                    float(trade.get("duration_hours", 0)),
                    trade.get("close_reason", ""),
                    entry_context.get("regime", ""),
                    float(entry_context.get("btc_d", 50)),
                    float(entry_context.get("usdt_d", 5)),
                    float(entry_context.get("btc_d_roc", 0)),
                    float(entry_context.get("usdt_d_roc", 0)),
                    float(entry_context.get("ensemble_score", 0)),
                    float(entry_context.get("confidence", 0)),
                    float(entry_context.get("ob_imbalance", 1)),
                    entry_context.get("cvd_direction", "neutral"),
                    int(entry_context.get("cvd_divergence", False)),
                    entry_context.get("actor_reasoning", ""),
                    int(entry_context.get("actor_approved", False)),
                    "",  # judge_critique filled in later
                    "",  # pattern_tag filled in later
                    datetime.now(LOCAL_TZ).isoformat(),
                    vs.to_blob(emb),
                ))
        except Exception as e:
            log.error(f"TradeMemory record error: {e}")

    def add_judge_critique(self, trade_id: str, critique: dict):
        """Store Judge critique and update the trade record."""
        try:
            emb = vs.get_embedder().encode(critique_text(critique))
            with self._conn() as c:
                c.execute("""
                INSERT INTO judge_critiques
                (trade_id, symbol, outcome, pnl, decision_quality,
                 entry_valid, risk_managed, missed_warnings, lesson,
                 pattern_tag, reviewed_at, embedding)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    trade_id,
                    critique.get("symbol", ""),
                    critique.get("outcome", ""),
                    float(critique.get("pnl", 0)),
                    critique.get("decision_quality", ""),
                    int(critique.get("entry_valid", True)),
                    int(critique.get("risk_managed", True)),
                    critique.get("missed_warnings", ""),
                    critique.get("lesson", ""),
                    critique.get("pattern_tag", ""),
                    critique.get("reviewed_at", ""),
                    vs.to_blob(emb),
                ))
                # Update the trade record with critique summary
                c.execute(
                    "UPDATE trades SET judge_critique=?, pattern_tag=? WHERE id=?",
                    (critique.get("lesson", ""),
                     critique.get("pattern_tag", ""),
                     trade_id)
                )
        except Exception as e:
            log.error(f"TradeMemory critique error: {e}")

    def save_meta_rules(self, meta_output: dict):
        """Save Meta-Judge synthesized rules."""
        try:
            with self._conn() as c:
                c.execute(
                    "INSERT INTO meta_rules (rules_json, synthesized_at) VALUES (?,?)",
                    (json.dumps(meta_output), meta_output.get("synthesized_at", ""))
                )
        except Exception as e:
            log.error(f"TradeMemory meta_rules error: {e}")

    def get_latest_meta_rules(self) -> Optional[dict]:
        """Return the most recent Meta-Judge output."""
        try:
            with self._conn() as c:
                row = c.execute(
                    "SELECT rules_json FROM meta_rules ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            log.error(f"TradeMemory meta_rules fetch error: {e}")
        return None

    def get_recent_critiques(self, limit: int = 20) -> list:
        """Return recent Judge critiques for Meta-Judge input."""
        try:
            with self._conn() as c:
                rows = c.execute(
                    "SELECT * FROM judge_critiques ORDER BY id DESC LIMIT ?",
                    (limit,)
                ).fetchall()
                cols = [d[0] for d in c.description]
                return [dict(zip(cols, r)) for r in rows]
        except Exception as e:
            log.error(f"TradeMemory critiques fetch error: {e}")
            return []

    # ── Reranking helpers ───────────────────────────────────────────────────

    @staticmethod
    def _recency_factor(closed_at: str) -> float:
        """1.0 for a trade closed now, decaying with a ~30-day half-life-ish curve."""
        try:
            ts = datetime.fromisoformat(closed_at)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=LOCAL_TZ)
            age_days = max(0.0, (datetime.now(LOCAL_TZ) - ts).total_seconds() / 86400.0)
            return float(np.exp(-age_days / 30.0))
        except Exception:
            return 0.5

    def find_similar(
        self,
        symbol: str,
        action: str,
        regime: str,
        ensemble_score: float,
        btc_d: float,
        n: int = 5,
        *,
        confidence: float = 0.0,
        usdt_d: float = 5.0,
        ob_imbalance: float = 1.0,
        cvd_direction: str = "neutral",
        cvd_divergence: bool = False,
        actor_reasoning: str = "",
    ) -> list:
        """Hybrid vector retrieval of the N most similar past trades.

        Combines a semantic embedding of the entry context (when the model is
        available) with an explicit numeric/categorical feature similarity,
        then reranks the blended score by recency and outcome magnitude.
        Gracefully degrades to feature-only similarity if embeddings are off.
        """
        side = "long" if action == "BUY" else "short"
        try:
            with self._conn() as c:
                rows = c.execute(
                    """SELECT symbol, side, mode, pnl, r_multiple, duration_hours,
                              regime, ensemble_score, confidence, ob_imbalance,
                              cvd_direction, cvd_divergence, btc_d, usdt_d,
                              actor_reasoning, judge_critique, pattern_tag,
                              closed_at, embedding
                       FROM trades
                       WHERE side=?
                       ORDER BY closed_at DESC
                       LIMIT 300""",
                    (side,)
                ).fetchall()
                cols = ["symbol","side","mode","pnl","r_multiple","duration_hours",
                        "regime","ensemble_score","confidence","ob_imbalance",
                        "cvd_direction","cvd_divergence","btc_d","usdt_d",
                        "actor_reasoning","judge_critique","pattern_tag",
                        "closed_at","embedding"]
                trades = [dict(zip(cols, r)) for r in rows]

            if not trades:
                return []

            # Query representations
            q_feat = vs.feature_vector(
                side, regime, ensemble_score, confidence, btc_d, usdt_d,
                ob_imbalance, cvd_direction, bool(cvd_divergence),
            )
            q_emb = vs.get_embedder().encode(entry_text(symbol, side, {
                "regime": regime, "ensemble_score": ensemble_score,
                "confidence": confidence, "btc_d": btc_d, "usdt_d": usdt_d,
                "ob_imbalance": ob_imbalance, "cvd_direction": cvd_direction,
                "cvd_divergence": cvd_divergence, "actor_reasoning": actor_reasoning,
            }))

            scored = []
            for t in trades:
                cand_feat = vs.feature_vector(
                    t["side"], t["regime"], t["ensemble_score"], t["confidence"],
                    t["btc_d"], t["usdt_d"], t["ob_imbalance"],
                    t["cvd_direction"], bool(t["cvd_divergence"]),
                )
                feat_sim = float(vs.cosine_matrix(q_feat, cand_feat[None, :])[0])

                cand_emb = vs.from_blob(t.get("embedding"))
                if q_emb is not None and cand_emb is not None and cand_emb.size == q_emb.size:
                    text_sim = float(vs.cosine_matrix(q_emb, cand_emb[None, :])[0])
                    base = 0.6 * text_sim + 0.4 * feat_sim
                else:
                    base = feat_sim  # semantic unavailable → feature-only

                # Rerank: weight by recency, nudge up decisive (high |R|) outcomes.
                recency = self._recency_factor(t.get("closed_at", ""))
                outcome_bonus = 0.05 * min(abs(t.get("r_multiple") or 0.0), 3.0)
                t["_score"] = base * (0.7 + 0.3 * recency) + outcome_bonus
                scored.append(t)

            scored.sort(key=lambda x: x["_score"], reverse=True)
            # Drop internal/bulky fields before returning to the Actor.
            for t in scored[:n]:
                t.pop("embedding", None)
            return scored[:n]

        except Exception as e:
            log.error(f"TradeMemory find_similar error: {e}")
            return []

    def find_similar_critiques(self, query_text: str, n: int = 3) -> list:
        """Semantic retrieval over the Judge-critique store, reranked by recency.

        Returns the most relevant past lessons for the current setup. Returns
        an empty list if embeddings are unavailable (no lexical fallback here —
        critiques are free-form prose where a lexical match adds little).
        """
        q_emb = vs.get_embedder().encode(query_text)
        if q_emb is None:
            return []
        try:
            with self._conn() as c:
                rows = c.execute(
                    """SELECT symbol, outcome, decision_quality, missed_warnings,
                              lesson, pattern_tag, reviewed_at, embedding
                       FROM judge_critiques
                       WHERE embedding IS NOT NULL
                       ORDER BY id DESC LIMIT 300"""
                ).fetchall()
                cols = ["symbol","outcome","decision_quality","missed_warnings",
                        "lesson","pattern_tag","reviewed_at","embedding"]
                crits = [dict(zip(cols, r)) for r in rows]

            scored = []
            for cr in crits:
                emb = vs.from_blob(cr.get("embedding"))
                if emb is None or emb.size != q_emb.size:
                    continue
                sim = float(vs.cosine_matrix(q_emb, emb[None, :])[0])
                recency = self._recency_factor(cr.get("reviewed_at", ""))
                cr["_score"] = sim * (0.7 + 0.3 * recency)
                scored.append(cr)

            scored.sort(key=lambda x: x["_score"], reverse=True)
            for cr in scored[:n]:
                cr.pop("embedding", None)
            return scored[:n]
        except Exception as e:
            log.error(f"TradeMemory find_similar_critiques error: {e}")
            return []

    def backfill_embeddings(self) -> int:
        """Embed any trades/critiques stored before vectors existed.

        Safe to call on startup; a no-op when embeddings are unavailable or
        everything is already embedded. Returns the number of rows updated.
        """
        if not vs.get_embedder().available:
            return 0
        updated = 0
        try:
            with self._conn() as c:
                # Trades
                rows = c.execute(
                    """SELECT id, symbol, side, regime, ensemble_score, confidence,
                              btc_d, usdt_d, ob_imbalance, cvd_direction,
                              cvd_divergence, actor_reasoning
                       FROM trades WHERE embedding IS NULL"""
                ).fetchall()
                for r in rows:
                    (tid, sym, side, regime, esc, conf, btc_d, usdt_d,
                     ob, cvd, cvd_div, reasoning) = r
                    emb = vs.get_embedder().encode(entry_text(sym, side, {
                        "regime": regime, "ensemble_score": esc, "confidence": conf,
                        "btc_d": btc_d, "usdt_d": usdt_d, "ob_imbalance": ob,
                        "cvd_direction": cvd, "cvd_divergence": cvd_div,
                        "actor_reasoning": reasoning,
                    }))
                    if emb is not None:
                        c.execute("UPDATE trades SET embedding=? WHERE id=?",
                                  (vs.to_blob(emb), tid))
                        updated += 1
                # Critiques
                rows = c.execute(
                    """SELECT id, symbol, decision_quality, missed_warnings,
                              lesson, pattern_tag
                       FROM judge_critiques WHERE embedding IS NULL"""
                ).fetchall()
                for r in rows:
                    cid, sym, dq, mw, lesson, ptag = r
                    emb = vs.get_embedder().encode(critique_text({
                        "symbol": sym, "decision_quality": dq,
                        "missed_warnings": mw, "lesson": lesson, "pattern_tag": ptag,
                    }))
                    if emb is not None:
                        c.execute("UPDATE judge_critiques SET embedding=? WHERE id=?",
                                  (vs.to_blob(emb), cid))
                        updated += 1
            if updated:
                log.info(f"Backfilled embeddings for {updated} rows.")
        except Exception as e:
            log.error(f"TradeMemory backfill error: {e}")
        return updated

    def count_trades(self) -> int:
        try:
            with self._conn() as c:
                return c.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        except Exception:
            return 0

    def critiques_since_last_meta(self) -> int:
        """Count judge_critiques added since the last meta_rules synthesis.
        Returns the total critique count when no meta_rules exist yet.
        Restart-safe: uses DB timestamps, not an in-memory counter."""
        try:
            with self._conn() as c:
                row = c.execute(
                    "SELECT synthesized_at FROM meta_rules ORDER BY id DESC LIMIT 1"
                ).fetchone()
                if not row:
                    return c.execute("SELECT COUNT(*) FROM judge_critiques").fetchone()[0]
                return c.execute(
                    "SELECT COUNT(*) FROM judge_critiques WHERE reviewed_at > ?",
                    (row[0],)
                ).fetchone()[0]
        except Exception as e:
            log.warning(f"critiques_since_last_meta error: {e}")
            return 0
