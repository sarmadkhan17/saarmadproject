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
from pathlib import Path
from typing import Optional

try:                                       # bot runtime (sys.path = bot/)
    from core.tz import LOCAL_TZ
    from agents import vector_store as vs
except ImportError:                        # tests import via the bot. package
    from bot.core.tz import LOCAL_TZ
    from bot.agents import vector_store as vs

log = logging.getLogger("TradeMemory")


def _opt_float(v):
    """Coerce to float, preserving None (so a missing agent vote stays NULL)."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


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


# Minimum blended sample below which the live edge is too thin to override a
# past lesson (avoids demoting a caution on the strength of one or two outcomes).
LESSON_OVERRIDE_MIN_N = 8


def annotate_lessons(lessons: list, blended: dict, regime: str, side: str,
                     min_n: int = LESSON_OVERRIDE_MIN_N) -> str:
    """Render retrieved Judge lessons into the Actor's 'past lessons' block,
    DEMOTING any cautionary (LOSS) lesson the live blended stat now contradicts.

    A frozen LOSS critique ("this setup loses") must never act as a veto once the
    continuously-updated blended edge for the SAME regime+side has turned
    positive. Rather than wait to "retire" the lesson from outcomes (lagging —
    the winning trades are missed first), the live blended stat is the arbiter:
    when it shows positive edge over a meaningful sample, the contradicted
    caution is relabelled as stale advisory context — the Actor still reads it
    but is told the live edge overrides it. Favourable (WIN) lessons, and
    cautions the live edge still corroborates, keep their normal label.
    """
    if not lessons:
        return ""
    n     = int(blended.get("n", 0) or 0)
    wr    = float(blended.get("win_rate", 0.0) or 0.0)
    avg_r = float(blended.get("avg_r", 0.0) or 0.0)
    live_edge_positive = (n >= min_n and wr >= 0.5 and avg_r > 0.0)

    lines = []
    for l in lessons:
        txt = (l.get("lesson") or "").strip()
        if not txt:
            continue
        cautionary = str(l.get("outcome", "")).upper() == "LOSS"
        if cautionary and live_edge_positive:
            lines.append(
                f"- [STALE ADVISORY — live blended {regime} {side} edge is now "
                f"{wr:.0%} win / {avg_r:+.2f}R over n={n}; this past caution is "
                f"contradicted by current data — context only, do not gate on it] {txt}"
            )
        else:
            lines.append(f"- [{l.get('outcome','?')}] {txt}")
    if not lines:
        return ""
    return "Relevant past lessons:\n" + "\n".join(lines)


def smoothed_stats(rows: list, half_life_h: float, prior: float,
                   now: Optional[datetime] = None) -> dict:
    """Recency-weighted, Bayesian-smoothed win-rate AND avg-R over `rows`.

    Each row is ``{"win": bool, "r": float, "age_h": float, "sw": float}`` where
    ``sw`` is an optional per-row source weight (realized=1.0, counterfactual<1).
    A trade's weight is ``sw · 0.5**(age_h/half_life)`` so same-session clusters
    fade over `half_life_h`. Both statistics are shrunk toward their neutral
    value with a Beta/pseudo-count `prior`:
      - win-rate toward 0.5 — a tiny sample can't read as 0% or 100%
      - avg_r toward 0.0    — the bug the old prompt had: avg_r was NEVER
        smoothed, so 5 losers read as a hard −0.6R anchor regardless of sample.
    Returns {n, raw_wins, win_rate, avg_r, weight}.
    """
    now = now or datetime.now(LOCAL_TZ)
    w_sum = w_win = wr_sum = 0.0
    raw_wins = 0
    for t in rows:
        w = float(t.get("sw", 1.0)) * 0.5 ** (t["age_h"] / max(half_life_h, 1e-6))
        w_sum += w
        if t["win"]:
            w_win += w
            raw_wins += 1
        wr_sum += w * (t["r"] or 0.0)
    n = len(rows)
    win_rate = (w_win + prior) / (w_sum + 2 * prior) if (w_sum + 2 * prior) > 0 else 0.5
    avg_r = wr_sum / (w_sum + prior) if (w_sum + prior) > 0 else 0.0
    return {"n": n, "raw_wins": raw_wins, "win_rate": win_rate,
            "avg_r": avg_r, "weight": w_sum}


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
            # Per-agent vote at entry (agent_reliability Phase A). Nullable —
            # old rows stay NULL and the aggregator excludes them.
            self._ensure_column(c, "trades", "smc_score", "REAL")
            self._ensure_column(c, "trades", "tech_score", "REAL")
            self._ensure_column(c, "trades", "macro_score", "REAL")

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
                    closed_at, embedding,
                    smc_score, tech_score, macro_score
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (
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
                    _opt_float(entry_context.get("smc_score")),
                    _opt_float(entry_context.get("tech_score")),
                    _opt_float(entry_context.get("macro_score")),
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

    def add_shadow_lesson(self, shadow: dict) -> bool:
        """Mint a templated lesson from a RESOLVED shadow (skipped) trade.

        The Judge only critiques trades that were TAKEN, so the lesson pool is
        one-sided (it can say "this long lost" but never "the long we skipped
        won"). This writes the missing half — a plain-language note for each
        rejected signal tracked to TP/SL — so the Actor's lesson channel becomes
        as representative as its stats channel. No LLM call: the text is a
        template over data the shadow already carries (the embedding is local).
        Idempotent per shadow id; only decided (tp/sl) shadows qualify.
        """
        status = shadow.get("status")
        if status not in ("tp", "sl"):
            return False
        tid = f"shadow:{shadow.get('id')}"
        side = shadow.get("side", "?")
        regime = shadow.get("regime") or "?"
        r = float(shadow.get("outcome_r") or 0.0)
        if status == "tp":
            outcome = "WIN"
            lesson = (f"A {side} in {regime} like this was SKIPPED but would have "
                      f"hit TP (+{r:.1f}R) — this setup type was profitable; do not "
                      f"reject it by default.")
        else:
            outcome = "LOSS"
            lesson = (f"A {side} in {regime} like this was SKIPPED and would have "
                      f"hit SL (-1R) — correctly avoided; this setup type was "
                      f"unprofitable.")
        try:
            ctx = {
                "regime": regime,
                "ensemble_score": shadow.get("ensemble_score", 0),
                "confidence": shadow.get("confidence", 0),
                "btc_d": shadow.get("btc_d", 50),
                "usdt_d": shadow.get("usdt_d", 5),
                "ob_imbalance": shadow.get("ob_imbalance", 1),
                "cvd_direction": shadow.get("cvd_direction", "neutral"),
                "cvd_divergence": shadow.get("cvd_divergence"),
            }
            emb = vs.get_embedder().encode(entry_text(shadow.get("symbol", ""), side, ctx))
            with self._conn() as c:
                exists = c.execute(
                    "SELECT 1 FROM judge_critiques WHERE trade_id=? LIMIT 1", (tid,)
                ).fetchone()
                if exists:
                    return False
                c.execute("""
                INSERT INTO judge_critiques
                (trade_id, symbol, outcome, pnl, decision_quality,
                 entry_valid, risk_managed, missed_warnings, lesson,
                 pattern_tag, reviewed_at, embedding)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", (
                    tid, shadow.get("symbol", ""), outcome, 0.0, "skipped",
                    1, 1, "", lesson,
                    f"skipped_{side}_{regime}",
                    shadow.get("resolved_at") or datetime.now(LOCAL_TZ).isoformat(),
                    vs.to_blob(emb),
                ))
            return True
        except Exception as e:
            log.debug(f"add_shadow_lesson failed {shadow.get('id')}: {e}")
            return False

    def backfill_shadow_lessons(self, limit: int = 400) -> int:
        """One-time mint of shadow lessons for already-resolved shadows so the
        balanced lesson pool exists immediately (bounded to the most recent
        `limit`). Safe/idempotent on startup; a no-op once they exist."""
        try:
            with self._conn() as c:
                c.row_factory = sqlite3.Row
                rows = [dict(r) for r in c.execute(
                    "SELECT * FROM shadow_trades WHERE status IN ('tp','sl') "
                    "ORDER BY resolved_at DESC LIMIT ?", (limit,)
                ).fetchall()]
        except Exception as e:
            log.debug(f"backfill_shadow_lessons query failed: {e}")
            return 0
        minted = sum(1 for s in rows if self.add_shadow_lesson(s))
        if minted:
            log.info(f"Backfilled {minted} shadow lessons (skipped-trade outcomes).")
        return minted

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
        """Return recent Judge critiques for Meta-Judge input.

        Excludes minted shadow lessons (trade_id 'shadow:%') — those balance the
        Actor's per-setup lesson channel, but Meta-Judge rules must still be
        synthesized only from REAL closed trades the Judge actually reviewed.
        """
        try:
            with self._conn() as c:
                cur = c.execute(
                    "SELECT * FROM judge_critiques "
                    "WHERE trade_id NOT LIKE 'shadow:%' "
                    "ORDER BY id DESC LIMIT ?",
                    (limit,)
                )
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
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

    # ── Counterfactual-aware precedent (the Actor's primary evidence) ────────

    @staticmethod
    def _empty_precedent(regime: str, side: str) -> dict:
        z = {"n": 0, "raw_wins": 0, "win_rate": 0.5, "avg_r": 0.0, "weight": 0.0}
        return {"realized": dict(z), "counterfactual": dict(z), "blended": dict(z),
                "examples": [], "regime": regime, "side": side}

    def find_similar_precedent(
        self, symbol: str, action: str, regime: str, ensemble_score: float,
        btc_d: float, *, confidence: float = 0.0, usdt_d: float = 5.0,
        ob_imbalance: float = 1.0, cvd_direction: str = "neutral",
        cvd_divergence: bool = False, half_life_h: float = 48.0,
        prior: float = 1.0, shadow_weight: float = 0.7, sim_floor: float = 0.7,
        max_realized: int = 40, max_shadow: int = 120, n_examples: int = 4,
    ) -> dict:
        """Precedent over the UNBIASED population for "setups like this".

        Pools two sources, each scored by *feature-only* similarity (same scale
        — shadows carry no text embedding) with NO outcome reranking so the
        retrieved sample is representative, not cherry-picked:
          - realized: executed trades (the `trades` table). Biased — it only
            contains setups the Actor already approved.
          - counterfactual: resolved shadow trades (rejected signals tracked to
            a hypothetical TP/SL). The unbiased complement; breaks the
            self-fulfilling loop where the Actor learns only from its own picks.

        Returns smoothed realized / counterfactual / blended stats (shadows
        down-weighted by `shadow_weight`) plus a few source-tagged examples.
        """
        side = "long" if action == "BUY" else "short"
        now = datetime.now(LOCAL_TZ)
        q = vs.feature_vector(side, regime, ensemble_score, confidence, btc_d,
                              usdt_d, ob_imbalance, cvd_direction, bool(cvd_divergence))

        def _age_h(ts) -> float:
            try:
                dt = datetime.fromisoformat(str(ts))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=LOCAL_TZ)
                return max(0.0, (now - dt).total_seconds() / 3600.0)
            except Exception:
                return 24.0

        def _sim(r) -> float:
            cf = vs.feature_vector(side, r["regime"], r["ensemble_score"],
                                   r["confidence"], r["btc_d"], r["usdt_d"],
                                   r["ob_imbalance"], r["cvd_direction"] or "neutral",
                                   bool(r["cvd_divergence"]))
            return float(vs.cosine_matrix(q, cf[None, :])[0])

        def _select(rows, cap):
            scored = sorted(((_sim(r), r) for r in rows), key=lambda x: x[0], reverse=True)
            kept = [(s, r) for s, r in scored if s >= sim_floor][:cap]
            # Sparse source above the floor → fall back to its top-8 so it still
            # contributes (a thin realized sample must not vanish entirely).
            if len(kept) < min(8, len(scored)):
                kept = scored[:min(8, len(scored))]
            return kept

        cols = ("regime,ensemble_score,confidence,ob_imbalance,"
                "cvd_direction,cvd_divergence,btc_d,usdt_d,symbol")
        # Realized and counterfactual are queried independently so a missing or
        # empty shadow_trades table only zeros the counterfactual, never the
        # realized precedent (and vice-versa).
        ex_rows, sh_rows = [], []
        try:
            with self._conn() as c:
                ex_rows = [dict(zip((cols + ",pnl,r_multiple,closed_at").split(","), r))
                           for r in c.execute(
                    f"SELECT {cols},pnl,r_multiple,closed_at FROM trades "
                    f"WHERE side=? ORDER BY closed_at DESC LIMIT 300", (side,))]
        except Exception as e:
            log.error(f"find_similar_precedent realized query error: {e}")
        try:
            with self._conn() as c:
                sh_rows = [dict(zip((cols + ",outcome_r,status,resolved_at").split(","), r))
                           for r in c.execute(
                    f"SELECT {cols},outcome_r,status,resolved_at FROM shadow_trades "
                    f"WHERE side=? AND status IN ('tp','sl') "
                    f"ORDER BY resolved_at DESC LIMIT 500", (side,))]
        except Exception as e:
            log.debug(f"find_similar_precedent counterfactual unavailable: {e}")

        realized = [{"win": (r["pnl"] or 0) > 0, "r": r["r_multiple"] or 0.0,
                     "age_h": _age_h(r["closed_at"]), "sw": 1.0,
                     "symbol": r["symbol"], "sim": s, "source": "realized"}
                    for s, r in _select(ex_rows, max_realized)]
        counter = [{"win": r["status"] == "tp", "r": r["outcome_r"] or 0.0,
                    "age_h": _age_h(r["resolved_at"]), "sw": shadow_weight,
                    "symbol": r["symbol"], "sim": s, "source": "counterfactual"}
                   for s, r in _select(sh_rows, max_shadow)]

        rz = smoothed_stats(realized, half_life_h, prior, now)
        cf = smoothed_stats(counter, half_life_h, prior, now)
        bl = smoothed_stats(realized + counter, half_life_h, prior, now)

        # Examples: the most-similar few from each source (mix realized + tracked)
        examples = []
        for pool in (realized, counter):
            for e in sorted(pool, key=lambda x: x["sim"], reverse=True)[:max(1, n_examples // 2)]:
                examples.append({"symbol": e["symbol"], "source": e["source"],
                                 "win": e["win"], "r": round(e["r"], 2)})
        return {"realized": rz, "counterfactual": cf, "blended": bl,
                "examples": examples[:n_examples], "regime": regime, "side": side}

    def find_similar_critiques(self, query_text: str, n: int = 3) -> list:
        """Semantic retrieval over the Judge-critique store, reranked by recency.

        Returns the most relevant past lessons for the current setup. Returns
        an empty list if embeddings are unavailable (no lexical fallback here —
        critiques are free-form prose where a lexical match adds little).

        SOURCE-BALANCED: the pool now holds two kinds of lesson — realized (from
        TAKEN trades, the Judge's biased sample) and shadow (from SKIPPED trades,
        the unbiased complement minted by add_shadow_lesson). We interleave the
        two so neither drowns the other; a realized-only top-n would re-impose the
        very selection bias the shadow lessons exist to correct (and shadows far
        outnumber realized, so a naive top-n would flip the bias the other way).
        """
        q_emb = vs.get_embedder().encode(query_text)
        if q_emb is None:
            return []
        try:
            with self._conn() as c:
                rows = c.execute(
                    """SELECT trade_id, symbol, outcome, decision_quality,
                              missed_warnings, lesson, pattern_tag, reviewed_at,
                              embedding
                       FROM judge_critiques
                       WHERE embedding IS NOT NULL
                       ORDER BY id DESC LIMIT 600"""
                ).fetchall()
                cols = ["trade_id","symbol","outcome","decision_quality",
                        "missed_warnings","lesson","pattern_tag","reviewed_at",
                        "embedding"]
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
            real = [c for c in scored if not str(c.get("trade_id") or "").startswith("shadow:")]
            shad = [c for c in scored if str(c.get("trade_id") or "").startswith("shadow:")]
            out, i, j = [], 0, 0
            while len(out) < n and (i < len(real) or j < len(shad)):
                if i < len(real):
                    out.append(real[i]); i += 1
                if len(out) < n and j < len(shad):
                    out.append(shad[j]); j += 1
            for cr in out:
                cr.pop("embedding", None)
            return out
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
