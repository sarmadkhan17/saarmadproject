"""
ShadowTracker — forward outcome tracking for REJECTED signals.

When a directional signal (ensemble said BUY/SELL) is rejected at a gate
(microstructure kill, Actor pre-filter, Actor rejection, risk gates), the
bot records a hypothetical "shadow trade": entry = price at rejection,
SL/TP = the exact ATR math a real trade would have used. Open shadows are
resolved forward against live candles — did the hypothetical trade hit TP
or SL first? — closing the survivorship-bias hole where the learning loop
(Judge/Meta-Judge) only ever sees trades that were taken.

Strictly observational: per-gate stats are surfaced to the Meta-Judge
prompt and to the operator (/shadows). Nothing here auto-tunes thresholds.
Forward paper data only — no historical replay, no LLM calls, no embeddings.

Shares data/trade_memory.db with TradeMemory (same connection-per-op +
mode-column conventions); shadows are learning data, not hot bot state.
"""

import logging
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

try:
    from core.tz import LOCAL_TZ          # bot runtime (sys.path = bot/)
except ImportError:
    from bot.core.tz import LOCAL_TZ      # tests import via the bot. package

log = logging.getLogger("ShadowTracker")

# SL geometry must mirror ExecutionEngine._place_sl so shadow outcomes are
# comparable to real trades.
SL_MIN_PCT = 0.015   # SL never closer than 1.5% of entry
SL_MAX_PCT = 0.25    # SL never further than 25% of entry


def _opt_float(v):
    """Coerce to float, preserving None so a missing agent vote stays NULL."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def risk_gate_label(reasons: list) -> str:
    """Sub-categorize a risk_agent rejection so each internal gate (4b breadth,
    4c EMA20, HTF, BTC momentum, confidence floor, …) is individually
    measurable in shadow stats. The failing gate's message is the last reason
    appended before evaluate() returned."""
    last = (reasons[-1] if reasons else "").lower()
    if "breadth" in last or "blow-off" in last or "capitulation" in last:
        return "risk_breadth"
    if "20ema" in last or "ema" in last:
        return "risk_ema20"
    if "htf" in last:
        return "risk_htf"
    if "btc momentum" in last:
        return "risk_btc"
    if "regime" in last or "blocked in" in last:
        return "risk_regime"
    if "conf" in last:
        return "risk_conf"
    if "agents=" in last:
        return "risk_agreement"
    return "risk_other"


def _to_utc_naive(dt: datetime) -> datetime:
    """Normalize any datetime to naive UTC (DataFeed candle index convention)."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(timezone.utc).replace(tzinfo=None)


class ShadowTracker:
    def __init__(self, db_path: Path, mode: str, cfg: Optional[dict] = None):
        cfg = cfg or {}
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = str(db_path)
        self.mode = mode
        self.enabled = bool(cfg.get("enabled", True))
        self.max_open = int(cfg.get("max_open", 60))
        self.max_age_hours = float(cfg.get("max_age_hours", 48))
        self.cooldown_min = float(cfg.get("per_symbol_cooldown_min", 60))
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def _init_db(self):
        with self._conn() as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS shadow_trades (
                id TEXT PRIMARY KEY,
                mode TEXT, symbol TEXT, side TEXT,
                gate TEXT, reason TEXT,
                entry_price REAL, atr REAL,
                sl_price REAL, tp_price REAL, r_target REAL,
                regime TEXT, ensemble_score REAL, confidence REAL,
                ob_imbalance REAL, cvd_direction TEXT, cvd_divergence INTEGER,
                btc_d REAL, usdt_d REAL,
                created_at TEXT,
                status TEXT DEFAULT 'open',
                resolved_at TEXT, resolve_price REAL,
                outcome_r REAL
            )""")
            c.execute("""
            CREATE INDEX IF NOT EXISTS idx_shadow_open
            ON shadow_trades(mode, status)""")
            # Redundancy instrumentation: which OTHER cheap gates would also
            # have blocked this signal (comma-joined). Added post-launch, so
            # migrate in place.
            cols = [r[1] for r in c.execute("PRAGMA table_info(shadow_trades)")]
            if "redundant_gates" not in cols:
                c.execute("ALTER TABLE shadow_trades ADD COLUMN redundant_gates TEXT")
            # Per-agent vote at rejection (agent_reliability Phase A). Nullable;
            # the aggregator excludes NULLs. This is the unbiased complement to
            # the taken-trade votes — rejected signals tracked to a real ±R.
            for col in ("smc_score", "tech_score", "macro_score"):
                if col not in cols:
                    c.execute(f"ALTER TABLE shadow_trades ADD COLUMN {col} REAL")

            # Post-exit excursion tracker — READ-ONLY instrument. After a trade
            # exits (taken) or a shadow resolves (hypothetical), watch price for
            # watch_hours and record how far the move CONTINUED past our exit and
            # whether a stopped-out trade later reached its original TP (noise vs
            # genuine). Never feeds a live decision.
            c.execute("""
            CREATE TABLE IF NOT EXISTS post_exit_tracks (
                id TEXT PRIMARY KEY,
                mode TEXT, symbol TEXT, side TEXT,
                source TEXT, ref_id TEXT,
                entry_price REAL, r_unit REAL,
                exit_price REAL, exit_at TEXT, exit_reason TEXT, exit_r REAL,
                orig_tp REAL, orig_sl REAL,
                watch_until TEXT,
                status TEXT DEFAULT 'watching',
                mfe_price REAL, mae_price REAL,
                post_mfe_r REAL, post_mae_r REAL,
                continued_r REAL, recovered_to_tp INTEGER,
                finalized_at TEXT
            )""")
            c.execute("""
            CREATE INDEX IF NOT EXISTS idx_pet_watching
            ON post_exit_tracks(mode, status)""")

    # ── capture ──────────────────────────────────────────────────────────

    def record_rejection(self, symbol: str, side: str, gate: str, reason: str,
                         entry_price: float, atr: float, profile,
                         ctx: dict) -> Optional[str]:
        """Record a rejected signal as an open shadow. Returns the shadow id,
        or None when skipped (disabled / bad inputs / cooldown / cap)."""
        if not self.enabled:
            return None
        if entry_price is None or atr is None or entry_price <= 0 or atr <= 0:
            return None

        now = datetime.now(LOCAL_TZ)
        try:
            with self._conn() as c:
                # Cap is PER GATE: the trend veto fires ~10× more often than
                # the downstream gates, and a shared cap would let it starve
                # their shadows entirely.
                n_open = c.execute(
                    "SELECT COUNT(*) FROM shadow_trades WHERE mode=? AND status='open' AND gate=?",
                    (self.mode, gate),
                ).fetchone()[0]
                if n_open >= self.max_open:
                    return None

                if self.cooldown_min > 0:
                    # The 30-60s scan + Actor verdict cache re-fires the same
                    # rejection every cycle — dedup per (symbol, side, gate) so
                    # a microstructure shadow doesn't block an adversary shadow
                    # for the same symbol within the cooldown window.
                    cutoff = (now - timedelta(minutes=self.cooldown_min)).isoformat()
                    recent = c.execute(
                        "SELECT 1 FROM shadow_trades "
                        "WHERE mode=? AND symbol=? AND side=? AND gate=? AND created_at>=? LIMIT 1",
                        (self.mode, symbol, side, gate, cutoff),
                    ).fetchone()
                    if recent:
                        return None

                sl_price, tp_price = self._sl_tp(side, entry_price, atr, profile)
                sl_dist = abs(entry_price - sl_price)
                tp_dist = abs(tp_price - entry_price)
                r_target = tp_dist / sl_dist if sl_dist > 0 else 0.0

                sid = uuid.uuid4().hex
                c.execute(
                    """INSERT INTO shadow_trades
                       (id, mode, symbol, side, gate, reason,
                        entry_price, atr, sl_price, tp_price, r_target,
                        regime, ensemble_score, confidence,
                        ob_imbalance, cvd_direction, cvd_divergence,
                        btc_d, usdt_d, created_at, status, redundant_gates,
                        smc_score, tech_score, macro_score)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,'open',?,?,?,?)""",
                    (
                        sid, self.mode, symbol, side, gate, str(reason)[:300],
                        float(entry_price), float(atr),
                        float(sl_price), float(tp_price), float(r_target),
                        ctx.get("regime"),
                        float(ctx.get("ensemble_score", 0) or 0),
                        float(ctx.get("confidence", 0) or 0),
                        float(ctx.get("ob_imbalance", 0) or 0),
                        ctx.get("cvd_direction"),
                        int(bool(ctx.get("cvd_divergence"))),
                        float(ctx.get("btc_d", 0) or 0),
                        float(ctx.get("usdt_d", 0) or 0),
                        now.isoformat(),
                        ctx.get("redundant_gates") or None,
                        _opt_float(ctx.get("smc_score")),
                        _opt_float(ctx.get("tech_score")),
                        _opt_float(ctx.get("macro_score")),
                    ),
                )
                log.debug(f"shadow {symbol} {side} gate={gate} sl={sl_price:.4f} tp={tp_price:.4f}")
                return sid
        except Exception as e:
            log.debug(f"record_rejection failed {symbol}: {e}")
            return None

    @staticmethod
    def _sl_tp(side: str, entry: float, atr: float, profile) -> tuple:
        """Same geometry as ExecutionEngine._place_sl + the profile TP."""
        sl_mult = getattr(profile, "stop_loss_atr_mult", 3.0)
        tp_mult = getattr(profile, "take_profit_atr_mult", 6.0)
        min_dist = entry * SL_MIN_PCT
        if side == "long":
            sl = entry - sl_mult * atr
            sl = min(sl, entry - min_dist)
            sl = max(sl, entry * (1 - SL_MAX_PCT))
            tp = entry + tp_mult * atr
        else:
            sl = entry + sl_mult * atr
            sl = max(sl, entry + min_dist)
            sl = min(sl, entry * (1 + SL_MAX_PCT))
            tp = entry - tp_mult * atr
        return sl, tp

    # ── resolution ───────────────────────────────────────────────────────

    def resolve_open(self, fetch_ohlcv_fn: Callable, now: Optional[datetime] = None) -> list:
        """Resolve open shadows against candles. `fetch_ohlcv_fn(symbol, tf, limit)`
        must return a DataFrame with a DatetimeIndex (naive UTC, per DataFeed)
        and high/low/close columns. Returns the list of resolved shadow dicts.
        Never raises — a feed failure just leaves shadows open."""
        now = now or datetime.now(LOCAL_TZ)
        resolved = []
        try:
            with self._conn() as c:
                c.row_factory = sqlite3.Row
                rows = [dict(r) for r in c.execute(
                    "SELECT * FROM shadow_trades WHERE mode=? AND status='open'",
                    (self.mode,),
                ).fetchall()]
        except Exception as e:
            log.debug(f"resolve_open query failed: {e}")
            return []

        by_symbol = {}
        for r in rows:
            by_symbol.setdefault(r["symbol"], []).append(r)

        for symbol, shadows in by_symbol.items():
            try:
                df = fetch_ohlcv_fn(symbol, "15m", 200)
            except Exception as e:
                log.debug(f"resolve_open fetch failed {symbol}: {e}")
                continue
            if df is None or len(df) == 0:
                continue
            for shadow in shadows:
                try:
                    outcome = self._resolve_one(shadow, df, now)
                    if outcome:
                        resolved.append(outcome)
                except Exception as e:
                    log.debug(f"resolve failed {shadow.get('id')}: {e}")
        return resolved

    def _resolve_one(self, shadow: dict, df, now: datetime) -> Optional[dict]:
        created = datetime.fromisoformat(shadow["created_at"])
        created_utc = _to_utc_naive(created)

        # Only candles opening at/after creation count — no lookahead into
        # history that predates the rejection.
        idx = df.index
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
        mask = idx >= created_utc
        window = df.loc[mask]

        sl, tp = shadow["sl_price"], shadow["tp_price"]
        is_long = shadow["side"] == "long"

        status = None
        resolve_price = None
        outcome_r = None
        for _, candle in window.iterrows():
            hi, lo = float(candle["high"]), float(candle["low"])
            hit_sl = (lo <= sl) if is_long else (hi >= sl)
            hit_tp = (hi >= tp) if is_long else (lo <= tp)
            if hit_sl:  # both in one candle → SL (conservative)
                status, resolve_price, outcome_r = "sl", sl, -1.0
                break
            if hit_tp:
                status, resolve_price, outcome_r = "tp", tp, float(shadow["r_target"])
                break

        if status is None:
            age_h = (now - created).total_seconds() / 3600.0
            if age_h <= self.max_age_hours or len(window) == 0:
                return None
            # Expire with signed mark-to-market R from the last close.
            last_close = float(window["close"].iloc[-1])
            risk = abs(shadow["entry_price"] - sl)
            if risk <= 0:
                return None
            move = (last_close - shadow["entry_price"]) if is_long \
                else (shadow["entry_price"] - last_close)
            status, resolve_price, outcome_r = "expired", last_close, move / risk

        try:
            with self._conn() as c:
                c.execute(
                    "UPDATE shadow_trades SET status=?, resolved_at=?, "
                    "resolve_price=?, outcome_r=? WHERE id=? AND status='open'",
                    (status, now.isoformat(), resolve_price, outcome_r, shadow["id"]),
                )
        except Exception as e:
            log.debug(f"resolve update failed {shadow['id']}: {e}")
            return None

        shadow.update(status=status, resolve_price=resolve_price, outcome_r=outcome_r)
        log.info(
            f"SHADOW {shadow['symbol']} {shadow['side']} gate={shadow['gate']} "
            f"→ {status.upper()} ({outcome_r:+.2f}R)"
        )
        return shadow

    # ── stats ────────────────────────────────────────────────────────────

    def gate_stats(self, days: int = 30) -> dict:
        return load_stats(self.db_path, self.mode, days)

    # ── post-exit excursion tracker (read-only instrument) ────────────────────

    def start_post_exit_track(
        self, source: str, symbol: str, side: str,
        entry_price: float, r_unit: float, exit_price: float,
        exit_reason: str, exit_r: Optional[float] = None,
        orig_tp: Optional[float] = None, orig_sl: Optional[float] = None,
        ref_id: str = "", watch_hours: float = 8.0,
        now: Optional[datetime] = None,
    ) -> Optional[str]:
        """Begin watching price AFTER a trade exits. `r_unit` is the trade's R in
        price terms (initial stop distance) so every excursion is reported in R.
        `side` is 'long'/'short'. Returns the track id, or None on bad input.
        READ-ONLY — never affects a live decision."""
        if (entry_price is None or exit_price is None
                or not r_unit or r_unit <= 0 or side not in ("long", "short")):
            return None
        now = now or datetime.now(LOCAL_TZ)
        tid = uuid.uuid4().hex
        try:
            with self._conn() as c:
                c.execute(
                    """INSERT INTO post_exit_tracks
                       (id, mode, symbol, side, source, ref_id,
                        entry_price, r_unit, exit_price, exit_at, exit_reason, exit_r,
                        orig_tp, orig_sl, watch_until, status, mfe_price, mae_price)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'watching', ?, ?)""",
                    (tid, self.mode, symbol, side, source, str(ref_id),
                     float(entry_price), float(r_unit), float(exit_price),
                     now.isoformat(), str(exit_reason)[:120],
                     (float(exit_r) if exit_r is not None else None),
                     (float(orig_tp) if orig_tp else None),
                     (float(orig_sl) if orig_sl else None),
                     (now + timedelta(hours=watch_hours)).isoformat(),
                     float(exit_price), float(exit_price)),
                )
        except sqlite3.Error as e:
            log.debug(f"start_post_exit_track failed {symbol}: {e}")
            return None
        return tid

    def update_post_exit_tracks(self, fetch_ohlcv_fn: Callable,
                                now: Optional[datetime] = None) -> int:
        """Poll watching tracks: extend the running max-favorable / max-adverse
        from candles since exit, and finalize those past watch_until. Returns the
        number finalized. Never raises — a feed failure just leaves them open."""
        now = now or datetime.now(LOCAL_TZ)
        try:
            with self._conn() as c:
                c.row_factory = sqlite3.Row
                rows = [dict(r) for r in c.execute(
                    "SELECT * FROM post_exit_tracks WHERE mode=? AND status='watching'",
                    (self.mode,),
                ).fetchall()]
        except sqlite3.Error as e:
            log.debug(f"post_exit query failed: {e}")
            return 0

        by_symbol: dict = {}
        for r in rows:
            by_symbol.setdefault(r["symbol"], []).append(r)

        finalized = 0
        for symbol, tracks in by_symbol.items():
            try:
                df = fetch_ohlcv_fn(symbol, "15m", 200)
            except Exception as e:
                log.debug(f"post_exit fetch failed {symbol}: {e}")
                continue
            if df is None or len(df) == 0:
                continue
            idx = df.index
            if getattr(idx, "tz", None) is not None:
                idx = idx.tz_convert("UTC").tz_localize(None)
            for t in tracks:
                try:
                    if self._update_one_track(t, df, idx, now):
                        finalized += 1
                except Exception as e:
                    log.debug(f"post_exit update failed {t.get('id')}: {e}")
        return finalized

    def _update_one_track(self, t: dict, df, idx, now: datetime) -> bool:
        """Update one track from its post-exit candle window. Returns True if it
        was finalized this call."""
        exit_at = _to_utc_naive(datetime.fromisoformat(t["exit_at"]))
        window = df.loc[idx >= exit_at]
        is_long = t["side"] == "long"
        ex, ru, entry = t["exit_price"], t["r_unit"], t["entry_price"]

        mfe_price, mae_price = t["mfe_price"], t["mae_price"]
        last = ex
        if len(window) > 0:
            hi = float(window["high"].max())
            lo = float(window["low"].min())
            last = float(window["close"].iloc[-1])
            if is_long:                       # favorable = up, adverse = down
                mfe_price = max(mfe_price, hi)
                mae_price = min(mae_price, lo)
            else:                             # short: favorable = down
                mfe_price = min(mfe_price, lo)
                mae_price = max(mae_price, hi)

        fields = {"mfe_price": mfe_price, "mae_price": mae_price}
        done = now >= datetime.fromisoformat(t["watch_until"])
        if done:
            # Excursion past the EXIT, signed in trade direction:
            #   post_mfe_r > 0 → the move kept going our way after we left
            #   post_mae_r < 0 → it went against the trade after exit
            post_mfe_r = ((mfe_price - ex) if is_long else (ex - mfe_price)) / ru
            post_mae_r = ((mae_price - ex) if is_long else (ex - mae_price)) / ru
            continued_r = ((last - entry) if is_long else (entry - last)) / ru
            # Noise test: only meaningful for a stopped/losing exit with a TP set.
            recovered = None
            tp = t["orig_tp"]
            if tp and t["exit_r"] is not None and t["exit_r"] <= 0:
                reached = (mfe_price >= tp) if is_long else (mfe_price <= tp)
                recovered = 1 if reached else 0
            fields.update(status="done", finalized_at=now.isoformat(),
                          post_mfe_r=post_mfe_r, post_mae_r=post_mae_r,
                          continued_r=continued_r, recovered_to_tp=recovered)

        set_clause = ", ".join(f"{k}=?" for k in fields)
        try:
            with self._conn() as c:
                c.execute(
                    f"UPDATE post_exit_tracks SET {set_clause} "
                    f"WHERE id=? AND status='watching'",
                    (*fields.values(), t["id"]),
                )
        except sqlite3.Error as e:
            log.debug(f"post_exit update write failed {t['id']}: {e}")
            return False
        return done

    def post_exit_rows(self, days: int = 30, status: str = "done") -> list:
        """Finalized post-exit tracks over the window — pure read for the report."""
        cutoff = (datetime.now(LOCAL_TZ) - timedelta(days=days)).isoformat()
        col = "finalized_at" if status == "done" else "exit_at"
        try:
            with self._conn() as c:
                c.row_factory = sqlite3.Row
                return [dict(r) for r in c.execute(
                    f"SELECT * FROM post_exit_tracks "
                    f"WHERE mode=? AND status=? AND {col}>=?",
                    (self.mode, status, cutoff),
                ).fetchall()]
        except sqlite3.Error:
            return []


def load_stats(db_path, mode: str, days: int = 30) -> dict:
    """Per-gate shadow stats — pure read, usable without a bot instance
    (Telegram handler, dashboard). Returns {} when the DB/table is absent."""
    db_path = Path(db_path)
    if not db_path.exists():
        return {}
    cutoff = (datetime.now(LOCAL_TZ) - timedelta(days=days)).isoformat()
    try:
        with sqlite3.connect(str(db_path), timeout=10) as c:
            rows = c.execute(
                """SELECT gate,
                          COUNT(*),
                          SUM(CASE WHEN status='tp' THEN 1 ELSE 0 END),
                          SUM(CASE WHEN status='sl' THEN 1 ELSE 0 END),
                          SUM(CASE WHEN status='expired' THEN 1 ELSE 0 END),
                          COALESCE(SUM(CASE WHEN status!='open' THEN outcome_r END), 0)
                   FROM shadow_trades
                   WHERE mode=? AND created_at>=?
                   GROUP BY gate""",
                (mode, cutoff),
            ).fetchall()
    except sqlite3.Error:
        return {}

    stats = {}
    for gate, n, tp, sl, expired, net_r in rows:
        tp, sl, expired = int(tp or 0), int(sl or 0), int(expired or 0)
        resolved = tp + sl + expired
        decided = tp + sl
        stats[gate] = {
            "n": int(n),
            "open": int(n) - resolved,
            "resolved": resolved,
            "tp": tp,
            "sl": sl,
            "expired": expired,
            "win_rate": (tp / decided) if decided else 0.0,
            "net_r": float(net_r or 0.0),
            "mean_r": (float(net_r) / resolved) if resolved else 0.0,
            "redundant_with": {},
        }

    # Redundancy matrix: of the signals gate X blocked, how many would another
    # cheap gate also have blocked? High overlap = the gates duplicate each
    # other rather than complement.
    try:
        with sqlite3.connect(str(db_path), timeout=10) as c:
            rrows = c.execute(
                """SELECT gate, redundant_gates, COUNT(*)
                   FROM shadow_trades
                   WHERE mode=? AND created_at>=? AND redundant_gates IS NOT NULL
                   GROUP BY gate, redundant_gates""",
                (mode, cutoff),
            ).fetchall()
        for gate, red, cnt in rrows:
            if gate not in stats:
                continue
            for other in str(red).split(","):
                other = other.strip()
                if other:
                    rw = stats[gate]["redundant_with"]
                    rw[other] = rw.get(other, 0) + int(cnt)
    except sqlite3.Error:
        pass
    return stats


def load_taken_stats(db_path, mode: str, days: int = 30) -> dict:
    """Closed REAL trades over the same window — the baseline the per-gate
    shadow precision is judged against. Pure read; {} when absent."""
    db_path = Path(db_path)
    if not db_path.exists():
        return {}
    cutoff = (datetime.now(LOCAL_TZ) - timedelta(days=days)).isoformat()
    try:
        with sqlite3.connect(str(db_path), timeout=10) as c:
            n, wins, net_r = c.execute(
                """SELECT COUNT(*),
                          SUM(CASE WHEN r_multiple > 0 THEN 1 ELSE 0 END),
                          COALESCE(SUM(r_multiple), 0)
                   FROM trades WHERE mode=? AND closed_at>=?""",
                (mode, cutoff),
            ).fetchone()
    except sqlite3.Error:
        return {}
    n = int(n or 0)
    return {
        "n": n,
        "win_rate": (int(wins or 0) / n) if n else 0.0,
        "net_r": float(net_r or 0.0),
        "mean_r": (float(net_r or 0.0) / n) if n else 0.0,
    }


def format_stats_block(stats: dict, days: int = 30) -> str:
    """Compact plain-text rendering shared by the Meta-Judge prompt."""
    if not stats:
        return "No shadow data yet."
    lines = [f"Shadow outcomes of REJECTED signals (last {days}d, hypothetical):"]
    for gate, s in sorted(stats.items()):
        lines.append(
            f"- {gate}: {s['n']} shadows, {s['resolved']} resolved "
            f"(hyp. win rate {s['win_rate']:.0%}, net {s['net_r']:+.1f}R "
            f"→ gate avoided {-s['net_r']:+.1f}R)"
        )
        red = s.get("redundant_with") or {}
        if red:
            overlap = ", ".join(f"{o}: {c}/{s['n']}" for o, c in sorted(red.items()))
            lines.append(f"    also blockable by → {overlap}")
    return "\n".join(lines)
