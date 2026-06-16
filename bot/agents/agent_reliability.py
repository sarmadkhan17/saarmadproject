"""
Agent reliability aggregator — agent_reliability Phase A (advisory).

Measures how often each ensemble agent's directional vote (smc / technical /
macro_flow) matched the ACTUAL ±R outcome, per regime, over BOTH populations:

  · taken trades      — `trades`        (biased: only setups gating approved)
  · rejected shadows  — `shadow_trades` (the unbiased complement: signals the
                        gates blocked, tracked forward to a real TP/SL)

Blending the two debiases the estimate the same way the Actor's precedent is
debiased — an agent that keeps voting for setups the gates reject, which then
win, gets credit it would never get from taken trades alone.

Output is a bounded weight multiplier per (agent, regime). Phase A is
ADVISORY: this module only computes and writes data/agent_reliability_<mode>.json
plus a log line. NOTHING reads it back into the live ensemble yet — that is
Phase B, behind the `ensemble.adaptive_weights` flag. Pure read over the DB;
never runs in the hot scan loop.
"""

import json
import logging
import math
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:                                       # bot runtime (sys.path = bot/)
    from core.tz import LOCAL_TZ
except ImportError:                        # tests import via the bot. package
    from bot.core.tz import LOCAL_TZ

log = logging.getLogger("AgentReliability")

AGENTS = ("smc", "tech", "macro")          # column prefixes: <agent>_score
_AGENT_LABEL = {"smc": "smc", "tech": "technical", "macro": "macro_flow"}

# Tunables — deliberately few and principled (no per-coin fitting).
VOTE_EPS      = 0.05    # |net_score| below this = abstain (excluded)
HALF_LIFE_H   = 168.0   # 1 week: a vote's weight halves every 7 days
SHRINK_K      = 20.0    # Bayesian prior: thin buckets pull the edge toward 0
GAIN          = 0.8     # edge → multiplier slope
MULT_LO       = 0.60
MULT_HI       = 1.40
MIN_N_LIVE    = 30      # below this, the multiplier stays advisory-only (flag)


def _sign(x: float) -> int:
    if x > VOTE_EPS:
        return 1
    if x < -VOTE_EPS:
        return -1
    return 0


def _side_sign(side: str) -> int:
    return 1 if (side or "").lower() in ("long", "buy") else -1


def _decay(age_h: float, half_life_h: float) -> float:
    return 0.5 ** (max(age_h, 0.0) / half_life_h) if half_life_h > 0 else 1.0


def _fetch_samples(db_path: str, mode: str, days: int) -> list:
    """Return [(regime, side, R, ts, {agent: score})] over both populations."""
    p = Path(db_path)
    if not p.exists():
        return []
    cutoff = (datetime.now(LOCAL_TZ) - timedelta(days=days)).isoformat()
    out = []
    try:
        with sqlite3.connect(str(p), timeout=10) as c:
            # Taken trades (resolved by definition — they are closed rows).
            for regime, side, r, ts, smc, tech, macro in c.execute(
                """SELECT regime, side, r_multiple, closed_at,
                          smc_score, tech_score, macro_score
                   FROM trades
                   WHERE mode=? AND closed_at>=? AND r_multiple IS NOT NULL""",
                (mode, cutoff),
            ):
                out.append((regime, side, r, ts,
                            {"smc": smc, "tech": tech, "macro": macro}))
            # Rejected shadows that resolved to a real ±R.
            for regime, side, r, ts, smc, tech, macro in c.execute(
                """SELECT regime, side, outcome_r, resolved_at,
                          smc_score, tech_score, macro_score
                   FROM shadow_trades
                   WHERE mode=? AND status IN ('tp','sl','expired')
                         AND resolved_at>=? AND outcome_r IS NOT NULL""",
                (mode, cutoff),
            ):
                out.append((regime, side, r, ts,
                            {"smc": smc, "tech": tech, "macro": macro}))
    except sqlite3.Error as e:
        log.debug(f"reliability fetch failed: {e}")
        return []
    return out


def compute(db_path: str, mode: str, days: int = 30,
            half_life_h: float = HALF_LIFE_H, now: Optional[datetime] = None) -> dict:
    """Compute per-(agent, regime) debiased reliability multipliers. Read-only."""
    now = now or datetime.now(LOCAL_TZ)
    samples = _fetch_samples(db_path, mode, days)

    # acc[regime][agent] = [sum_correct_w, sum_w]
    acc: dict = {}
    for regime, side, r, ts, scores in samples:
        if r is None:
            continue
        regime = regime or "UNKNOWN"
        correct_dir = _side_sign(side) if r > 0 else -_side_sign(side)
        try:
            age_h = (now - datetime.fromisoformat(ts)).total_seconds() / 3600.0
        except (TypeError, ValueError):
            age_h = 0.0
        w = _decay(age_h, half_life_h)
        for agent in AGENTS:
            sc = scores.get(agent)
            if sc is None:
                continue
            av = _sign(float(sc))
            if av == 0:
                continue                    # abstained — no opinion to score
            bucket = acc.setdefault(regime, {}).setdefault(agent, [0.0, 0.0])
            bucket[0] += w if av == correct_dir else 0.0
            bucket[1] += w

    regimes: dict = {}
    for regime, agents in acc.items():
        regimes[regime] = {}
        for agent, (cw, tw) in agents.items():
            if tw <= 0:
                continue
            accuracy = cw / tw
            edge = accuracy - 0.5
            # Bayesian shrink toward 0 edge by effective sample weight.
            shrunk = edge * (tw / (tw + SHRINK_K))
            mult = max(MULT_LO, min(MULT_HI, 1.0 + GAIN * shrunk))
            regimes[regime][_AGENT_LABEL[agent]] = {
                "n": round(tw, 2),
                "accuracy": round(accuracy, 4),
                "edge": round(edge, 4),
                "multiplier": round(mult, 4),
                "actionable": tw >= MIN_N_LIVE,
            }

    return {
        "generated_at": now.isoformat(),
        "mode": mode,
        "window_days": days,
        "params": {
            "half_life_h": half_life_h, "shrink_k": SHRINK_K, "gain": GAIN,
            "bounds": [MULT_LO, MULT_HI], "min_n_live": MIN_N_LIVE,
            "vote_eps": VOTE_EPS,
        },
        "samples": len(samples),
        "regimes": regimes,
    }


def write_report(report: dict, data_dir) -> Path:
    """Write the advisory JSON atomically. Returns the path."""
    data_dir = Path(data_dir)
    out = data_dir / f"agent_reliability_{report['mode']}.json"
    tmp = out.with_suffix(".tmp.json")
    with open(tmp, "w") as f:
        json.dump(report, f, indent=2)
    tmp.replace(out)
    return out


def format_report(report: dict) -> str:
    """Compact human-readable rendering for logs / Telegram / the auditor."""
    lines = [
        f"Agent reliability [{report['mode']}] — {report['samples']} samples, "
        f"last {report['window_days']}d (ADVISORY, not applied):"
    ]
    if not report["regimes"]:
        lines.append("  (no agent-tagged outcomes yet)")
        return "\n".join(lines)
    for regime in sorted(report["regimes"]):
        lines.append(f"  {regime}:")
        for agent in sorted(report["regimes"][regime]):
            s = report["regimes"][regime][agent]
            flag = "" if s["actionable"] else "  (thin — advisory only)"
            lines.append(
                f"    {agent:<11} acc={s['accuracy']:.0%} edge={s['edge']:+.2f} "
                f"n={s['n']:.0f} → weight ×{s['multiplier']:.2f}{flag}"
            )
    return "\n".join(lines)
