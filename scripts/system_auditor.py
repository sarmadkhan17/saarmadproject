#!/usr/bin/env python3
"""
SystemAuditor — meta-oversight agent for cryptobot_v5.

Detects systematic flaws across all components, reasons about root causes
via DeepSeek R1, and proposes ranked changes. User accepts/ignores via
Telegram reply; config changes apply immediately, code changes save to
data/proposed_changes.md for manual application in a Claude session.

Modes:
  python scripts/system_auditor.py            # run audit once
  python scripts/system_auditor.py --poll     # check Telegram for pending replies
  python scripts/system_auditor.py --loop     # audit every 6h + poll every 10min
  python scripts/system_auditor.py --no-telegram  # print only
"""

import argparse
import json
import logging
import math
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
import yaml

# Pure-read gate/shadow stats from the bot package (no bot instance needed)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from bot.agents.shadow_tracker import load_stats as _shadow_stats, \
        load_taken_stats as _taken_stats, format_stats_block as _shadow_block
except Exception:                                    # pragma: no cover
    _shadow_stats = _taken_stats = _shadow_block = None

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT           = Path(__file__).resolve().parent.parent
CONFIG_PATH    = ROOT / "config_futures.yaml"
STATE_PATH     = ROOT / "data" / "futures_state.json"
DB_PATH        = ROOT / "data" / "trade_memory.db"
USAGE_PATH     = ROOT / "data" / "deepseek_usage.json"
CB_PATH        = ROOT / "data" / "circuit_breaker.json"
SCANNER_PATH   = ROOT / "data" / "scanner_cache.json"
PROPOSALS_PATH = ROOT / "data" / "audit_proposals.json"
PROPOSED_CODE  = ROOT / "data" / "proposed_changes.md"
LOG_DIR        = ROOT / "logs"
ENV_PATH       = ROOT / ".env"

LOG_RECENT_LINES = 8000  # last N log lines ≈ 2h at 60s interval, 20 symbols
AUDIT_INTERVAL   = 360   # minutes between audit runs in --loop mode
POLL_INTERVAL  = 10     # minutes between polls in --loop mode
PROPOSAL_TTL   = 14     # days before pending proposals expire

# Mirror bot constants — keep in sync with source
WIN_RATE_THRESHOLD = 0.40
WIN_RATE_PENALTY   = 0.15
OB_KILL_STRONG     = 2.0
OB_KILL_MILD       = 1.4
PROFILE_THRESHOLDS = {"STRICT": 0.70, "BALANCED": 0.58, "AGGRESSIVE": 0.42, "CONFLUENCE": 0.55}

# Source snippets to pull for R1 context per anomaly code
SOURCE_MAP: Dict[str, Tuple[str, int, int]] = {
    "OB_ASYM":     ("bot/agents/microstructure.py", 55, 100),
    "WINRATE":     ("config_futures.yaml",          1,  15),
    "RAG_BIAS":    ("bot/agents/trade_memory.py",   235, 265),
}

log = logging.getLogger("SystemAuditor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class Anomaly:
    code: str           # e.g. "OB_ASYM"
    severity: str       # WARN / ERROR
    bullet: str         # single compact line for Telegram
    data: dict          # supporting metrics passed to R1


@dataclass
class Proposal:
    id: str
    run_ts: str
    num: int            # display number shown in Telegram ([1], [2] ...)
    priority: str       # HIGH / MEDIUM / LOW
    type: str           # "config" / "code" / "operational"
    title: str
    evidence: str
    status: str         # pending / accepted / ignored / expired

    # Config / operational fields
    config_path: str = ""   # dot-separated: "actor.winrate_prior_strength"
    from_value: str  = ""
    to_value: str    = ""

    # Code fields
    file_path: str   = ""
    current_code: str = ""
    proposed_code: str = ""
    explanation: str  = ""

    proposed_at: str = ""
    decided_at: str  = ""
    expires_at: str  = ""

    def icon(self) -> str:
        return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(self.priority, "•")


# ── Config & data loaders ──────────────────────────────────────────────────────

def _load_cfg() -> dict:
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    actor  = raw.get("actor", {})
    risk   = raw.get("risk", {})
    strat  = raw.get("strategy", {})
    return {
        "half_life_h":    float(actor.get("winrate_half_life_hours", 48)),
        "prior_k":        float(actor.get("winrate_prior_strength", 1.0)),
        "cache_ttl":      int(actor.get("cache_ttl_seconds", 180)),
        "max_consec":     int(risk.get("max_consecutive_losses", 6)),
        "max_daily_loss": float(risk.get("max_daily_loss_pct", 0.03)),
        "profile":        strat.get("trading_profile", "AGGRESSIVE"),
    }


def _load_state() -> Tuple[List[dict], dict]:
    if not STATE_PATH.exists():
        return [], {}
    with open(STATE_PATH) as f:
        d = json.load(f)
    return d.get("signals", []), d.get("live_strategy", {})


def _load_trades() -> List[dict]:
    if not DB_PATH.exists():
        return []
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=5)
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM trades ORDER BY closed_at ASC").fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        log.warning(f"DB load error: {e}")
        return []


def _load_log_lines() -> List[str]:
    """Return the most recent LOG_RECENT_LINES lines from the bot log.
    Uses line count (not timestamps) so timezone differences don't inflate the window.
    Sorts by mtime so futures_bot.log (current) is always included.
    """
    # Sort newest-to-oldest by mtime, take 2 most recent, then reverse to get
    # oldest-first so extending in order and slicing [-N:] gives true recent lines.
    candidates = sorted(LOG_DIR.glob("futures_bot.log*"),
                        key=lambda p: p.stat().st_mtime, reverse=True)[:2][::-1]
    lines: List[str] = []
    for lf in candidates:
        try:
            text = lf.read_bytes()[-2_000_000:].decode("utf-8", errors="replace")
            lines.extend(text.splitlines())
        except Exception:
            pass
    return lines[-LOG_RECENT_LINES:]


def _load_source_snippet(rel_path: str, line_start: int, line_end: int) -> str:
    path = ROOT / rel_path
    if not path.exists():
        return ""
    all_lines = path.read_text().splitlines()
    snippet   = all_lines[max(0, line_start - 1): line_end]
    return "\n".join(snippet)


def _load_telegram_cfg() -> dict:
    cfg: dict = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if line.startswith("TELEGRAM_TOKEN="):
                cfg["token"] = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("TELEGRAM_CHAT_ID="):
                cfg["chat_id"] = line.split("=", 1)[1].strip().strip('"')
    return cfg


# ── Metrics snapshot ───────────────────────────────────────────────────────────

def collect_metrics() -> dict:
    cfg        = _load_cfg()
    signals, live = _load_state()
    trades     = _load_trades()
    log_lines  = _load_log_lines()
    usage      = json.load(open(USAGE_PATH)) if USAGE_PATH.exists() else {}
    cb         = json.load(open(CB_PATH))    if CB_PATH.exists()    else {}
    watchlist  = (json.load(open(SCANNER_PATH)).get("top_coins", [])
                  if SCANNER_PATH.exists() else [])
    now        = datetime.now(timezone.utc)

    # ── Signal rejection breakdown ─────────────────────────────────────────────
    rejected   = [s for s in signals if s.get("status") == "rejected"]
    buy_rej    = sum(1 for s in rejected if s.get("action") == "BUY")
    sell_rej   = sum(1 for s in rejected if s.get("action") == "SELL")

    gate_counts: Counter = Counter()
    for s in rejected:
        reason = s.get("reason", "")
        if "actor" in reason:           gate_counts["actor_llm"] += 1
        elif "pre-filter" in reason or "conf floor" in reason:
                                        gate_counts["conf_threshold"] += 1
        elif "micro" in reason.lower(): gate_counts["micro_kill"] += 1
        elif "trend" in reason.lower(): gate_counts["trend_veto"] += 1
        else:                           gate_counts["other"] += 1

    # ── Micro kill breakdown from logs ────────────────────────────────────────
    kill_re  = re.compile(r"SIGNAL (\S+) → (BUY|SELL) \| MICRO KILL: (.+)")
    ratio_re = re.compile(r"(?:bid|ask) pressure ([\d.]+)x")
    sell_ob_kills = sell_ob_cvd_aligned = buy_ob_kills = 0
    sell_ob_examples: List[Tuple[str, float]] = []

    for line in log_lines:
        if "MICRO KILL" not in line:
            continue
        m = kill_re.search(line)
        if not m:
            continue
        sym, direction, reason = m.group(1), m.group(2), m.group(3)
        if direction == "SELL" and "OB: heavy bid pressure" in reason:
            sell_ob_kills += 1
            rm = ratio_re.search(reason)
            ratio = float(rm.group(1)) if rm else OB_KILL_STRONG
            sell_ob_examples.append((sym, ratio))
            if "CVD: aligned bearish" in reason:
                sell_ob_cvd_aligned += 1
        elif direction == "BUY" and "OB: heavy ask pressure" in reason:
            buy_ob_kills += 1

    # ── RAG top-match diversity ───────────────────────────────────────────────
    rag_re  = re.compile(r"RAG \S+ \| \S+ \| \d+ similar trades.*top=(\S+):([\d.]+)")
    rag_top: List[str] = []
    for line in log_lines:
        m = rag_re.search(line)
        if m:
            rag_top.append(m.group(1))
    rag_dominant      = ""
    rag_dominant_pct  = 0.0
    rag_dominant_loss = False
    if rag_top:
        top_sym, top_cnt = Counter(rag_top).most_common(1)[0]
        rag_dominant     = top_sym
        rag_dominant_pct = top_cnt / len(rag_top)
        # Check if dominant match is a loss in the DB
        for t in trades:
            if t.get("symbol") == top_sym:
                rag_dominant_loss = t.get("pnl", 0) < 0
                break

    # ── Pre-filter chronic symbols ────────────────────────────────────────────
    pre_re = re.compile(r"ACTOR (\S+) \| pre-filter reject conf=([\d.]+)")
    prefilter_counts: Counter = Counter()
    prefilter_confs: Dict[str, List[float]] = defaultdict(list)
    for line in log_lines:
        m = pre_re.search(line)
        if m:
            prefilter_counts[m.group(1)] += 1
            prefilter_confs[m.group(1)].append(float(m.group(2)))
    chronic_prefilter = {
        sym: round(sum(v) / len(v), 2)
        for sym, cnt in prefilter_counts.items()
        if cnt >= 8
        for v in [prefilter_confs[sym]]
    }

    # ── Data quality warnings ─────────────────────────────────────────────────
    dq_re     = re.compile(r"Data quality for (\S+): \['Too few bars: \d+'\]")
    dq_counts: Counter = Counter()
    for line in log_lines:
        m = dq_re.search(line)
        if m:
            dq_counts[m.group(1)] += 1
    persistent_dq = {s: c for s, c in dq_counts.items() if c >= 8}

    # ── Win-rate from DB ──────────────────────────────────────────────────────
    half_life = cfg["half_life_h"]
    prior     = cfg["prior_k"]
    w_win = w_sum = 0.0
    longs = shorts = wins = losses = 0
    long_r = short_r = long_pnl = short_pnl = long_wins = short_wins = 0.0
    exit_reasons: Counter = Counter()

    for t in trades:
        pnl   = t.get("pnl", 0) or 0
        age_h = 24.0
        ca    = t.get("closed_at", "")
        if ca:
            try:
                dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                age_h = max(0.0, (now - dt).total_seconds() / 3600)
            except Exception:
                pass
        w      = 0.5 ** (age_h / max(half_life, 1e-6))
        w_sum += w
        if pnl > 0:
            w_win += w
            wins  += 1
        else:
            losses += 1
        side = t.get("side", "")
        r    = t.get("r_multiple", 0) or 0
        if side == "long":
            longs    += 1; long_r += r; long_pnl += pnl
            if pnl > 0: long_wins += 1
        elif side == "short":
            shorts   += 1; short_r += r; short_pnl += pnl
            if pnl > 0: short_wins += 1
        exit_reasons[t.get("close_reason", "unknown")] += 1

    smoothed_wr = (w_win + prior) / (w_sum + 2 * prior) if (w_sum + 2 * prior) > 0 else 0.5

    # ── Drought ────────────────────────────────────────────────────────────────
    order_re = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*Order confirmed")
    order_ts: List[datetime] = []
    for lf in sorted(LOG_DIR.glob("futures_bot.log*"), reverse=True):
        try:
            for line in lf.read_bytes()[-5_000_000:].decode("utf-8", errors="replace").splitlines():
                m = order_re.match(line)
                if m:
                    try:
                        order_ts.append(datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                                        .replace(tzinfo=timezone.utc))
                    except ValueError:
                        pass
        except Exception:
            pass
    drought_h = ((now - max(order_ts)).total_seconds() / 3600) if order_ts else 0.0

    # ── LLM cost ──────────────────────────────────────────────────────────────
    daily = usage.get("daily", {})
    recent_costs = [daily[d].get("cost_usd", 0) for d in sorted(daily)[-7:]]
    avg_cost     = sum(recent_costs) / len(recent_costs) if recent_costs else 0

    # ── Assemble snapshot ──────────────────────────────────────────────────────
    total_trades = longs + shorts
    return {
        "cfg":               cfg,
        "profile":           cfg["profile"],
        "min_conf":          PROFILE_THRESHOLDS.get(cfg["profile"], 0.50),
        # Signals
        "signals_total":     len(signals),
        "rejected_total":    len(rejected),
        "buy_rej":           buy_rej,
        "sell_rej":          sell_rej,
        "gate_counts":       dict(gate_counts),
        # Micro kills
        "sell_ob_kills":     sell_ob_kills,
        "buy_ob_kills":      buy_ob_kills,
        "sell_ob_cvd_aligned": sell_ob_cvd_aligned,
        "sell_ob_examples":  sell_ob_examples[:5],
        # RAG
        "rag_dominant":      rag_dominant,
        "rag_dominant_pct":  round(rag_dominant_pct, 2),
        "rag_dominant_loss": rag_dominant_loss,
        "rag_total":         len(rag_top),
        # Trades
        "trade_count":       total_trades,
        "wins":              wins,
        "losses":            losses,
        "smoothed_wr":       round(smoothed_wr, 3),
        "longs":             longs,
        "shorts":            shorts,
        "long_wr":           round(long_wins / longs, 2) if longs else 0,
        "short_wr":          round(short_wins / shorts, 2) if shorts else 0,
        "long_avg_r":        round(long_r / longs, 2) if longs else 0,
        "short_avg_r":       round(short_r / shorts, 2) if shorts else 0,
        "exit_reasons":      dict(exit_reasons.most_common(6)),
        # Data quality
        "persistent_dq":     persistent_dq,
        "chronic_prefilter": chronic_prefilter,
        # Drought
        "drought_h":         round(drought_h, 1),
        # Live
        "regime":            live.get("market_regime", "?"),
        "adx":               live.get("adx", 0),
        "breadth":           live.get("breadth", 0),
        "btc_d":             live.get("btc_d", 0),
        # Cost
        "avg_daily_cost":    round(avg_cost, 3),
        "monthly_proj":      round(avg_cost * 30, 2),
        "total_cost":        round(usage.get("total_cost_usd", 0), 2),
        # Circuit breaker
        "consec_losses":     cb.get("consec_losses", 0),
        "max_consec":        cfg["max_consec"],
        "cb_disabled":       bool(cb.get("disabled_until")),
        # Watchlist
        "watchlist":         watchlist,
        # Gate complementarity (shadow tracker, pure read)
        "gate_shadow_stats": (_shadow_stats(DB_PATH, "futures", days=30)
                              if _shadow_stats else {}),
        "gate_taken_stats":  (_taken_stats(DB_PATH, "futures", days=30)
                              if _taken_stats else {}),
    }


# ── Anomaly detector ───────────────────────────────────────────────────────────

def detect_anomalies(m: dict) -> List[Anomaly]:
    found: List[Anomaly] = []

    # Drought
    if m["drought_h"] >= 24:
        x = m["drought_h"]
        found.append(Anomaly("DROUGHT", "ERROR" if x >= 48 else "WARN",
            f"{x:.0f}h no trade", {"drought_h": x}))

    # OB kill asymmetry — SELL blocked despite CVD agreement
    aligned = m["sell_ob_cvd_aligned"]
    if aligned >= 5:
        examples = Counter(s for s, _ in m["sell_ob_examples"]).most_common(4)
        avg_r    = (sum(r for _, r in m["sell_ob_examples"]) / len(m["sell_ob_examples"])
                    if m["sell_ob_examples"] else 0)
        found.append(Anomaly("OB_ASYM", "ERROR" if aligned >= 15 else "WARN",
            f"{aligned} shorts/2h blocked by OB bid wall with CVD aligned bearish "
            f"(avg {avg_r:.1f}x, threshold {OB_KILL_STRONG}x)",
            {"aligned": aligned, "sell_ob": m["sell_ob_kills"],
             "buy_ob": m["buy_ob_kills"], "examples": examples}))

    # Win-rate below threshold
    wr = m["smoothed_wr"]
    if wr < WIN_RATE_THRESHOLD:
        found.append(Anomaly("WINRATE", "ERROR" if wr < 0.30 else "WARN",
            f"Win-rate {wr:.0%} < {WIN_RATE_THRESHOLD:.0%} → actor penalising all signals",
            {"smoothed_wr": wr, "wins": m["wins"], "losses": m["losses"],
             "half_life": m["cfg"]["half_life_h"], "prior": m["cfg"]["prior_k"]}))

    # Long/short bias
    total = m["longs"] + m["shorts"]
    if total >= 8 and m["longs"] / total >= 0.68:
        found.append(Anomaly("DIR_BIAS", "WARN",
            f"Trade bias: {m['longs']}L / {m['shorts']}S "
            f"({m['longs']/total:.0%} long) — short side suppressed",
            {"longs": m["longs"], "shorts": m["shorts"],
             "long_wr": m["long_wr"], "short_wr": m["short_wr"]}))

    # Gate precision — a gate whose blocked signals would have WON well above
    # the taken-trade baseline is filtering winners, not protecting capital.
    gt = m.get("gate_taken_stats") or {}
    base_wr = gt.get("win_rate", 0.0)
    for gate, s in (m.get("gate_shadow_stats") or {}).items():
        decided = s["tp"] + s["sl"]
        if decided >= 20 and gt.get("n", 0) >= 20 \
                and s["win_rate"] > base_wr + 0.15:
            found.append(Anomaly("GATE_PRECISION",
                "ERROR" if s["win_rate"] > base_wr + 0.30 else "WARN",
                f"Gate '{gate}' blocks winners: blocked-signal WR "
                f"{s['win_rate']:.0%} vs taken baseline {base_wr:.0%} "
                f"({decided} decided shadows, {s['net_r']:+.1f}R foregone)",
                {"gate": gate, "blocked_wr": s["win_rate"], "baseline_wr": base_wr,
                 "decided": decided, "net_r": s["net_r"],
                 "redundant_with": s.get("redundant_with", {})}))

    # RAG monopoly
    if m["rag_dominant_pct"] >= 0.60 and m["rag_total"] >= 20:
        loss_str = " (it's a LOSS trade)" if m["rag_dominant_loss"] else ""
        found.append(Anomaly("RAG_BIAS", "WARN",
            f"{m['rag_dominant']} top RAG match in "
            f"{m['rag_dominant_pct']:.0%} of queries{loss_str}",
            {"symbol": m["rag_dominant"], "pct": m["rag_dominant_pct"],
             "is_loss": m["rag_dominant_loss"], "db_size": m["trade_count"]}))

    # Persistent data quality symbols in watchlist
    dq_in_watch = {s: c for s, c in m["persistent_dq"].items() if s in m["watchlist"]}
    if dq_in_watch:
        syms = ", ".join(f"{s}({c}×)" for s, c in
                         sorted(dq_in_watch.items(), key=lambda x: -x[1])[:4])
        found.append(Anomaly("DATA_QUALITY", "WARN",
            f"Data quality warnings/2h: {syms}",
            {"symbols": dq_in_watch}))

    # Exit pattern — volume_collapse dominance
    losses_total = m["losses"]
    if losses_total >= 4:
        vc = sum(c for r, c in m["exit_reasons"].items() if "volume_collapse" in r)
        if vc / losses_total >= 0.40:
            found.append(Anomaly("VOL_COLLAPSE", "WARN",
                f"volume_collapse = {vc}/{losses_total} losses ({vc/losses_total:.0%}) — "
                f"entries into thin liquidity",
                {"vol_collapse": vc, "total_losses": losses_total,
                 "exit_reasons": m["exit_reasons"]}))

    return found


# ── DeepSeek R1 brain ──────────────────────────────────────────────────────────

BRAIN_SYSTEM = """You are a trading bot performance auditor. Analyze the metrics and anomalies provided.
Propose up to 5 specific improvements, ranked by impact. Do NOT re-propose anything in ignored_proposals.

Key parameter semantics (read carefully before proposing changes):
- winrate_prior_strength: HIGHER = more lenient (smooths toward 50%, reduces penalty from losses).
  If win-rate is BELOW threshold causing drought → INCREASE this value.
- winrate_half_life_hours: LOWER = faster recovery from losses (losses decay weight faster).
  If bot is stuck penalising old losses → DECREASE this value.
- min_confidence: LOWER = easier to pass the gate. Only raise if too many bad trades.
- OB kill threshold (IMBALANCE_STRONG=2.0): for SELL signals, bid/ask ratio > 2.0 kills the short.
  Crypto books naturally skew bullish, so this over-kills SELL. To fix: raise SELL threshold OR
  require CVD to also disagree (code change in microstructure.py).
- scanner.blacklist: add symbol strings to stop scanning useless symbols.

Output a JSON array only — no text outside the array. Each item MUST follow this schema exactly:
{
  "priority": "HIGH|MEDIUM|LOW",
  "type": "config|code|operational",
  "title": "<10 words max, action-focused>",
  "evidence": "<1 sentence, cite specific numbers from the metrics>",

  For type=config or type=operational (config edit):
    "config_path": "<dot.separated.yaml.key e.g. actor.winrate_prior_strength>",
    "from_value": "<current value as string>",
    "to_value": "<proposed value as string>",

  For type=operational blacklist:
    "config_path": "scanner.blacklist",
    "from_value": "",
    "to_value": "<symbol string e.g. QNTX/USDT>",

  For type=code:
    "file_path": "<relative path from project root>",
    "current_code": "<exact lines to replace, preserve indentation>",
    "proposed_code": "<replacement lines, preserve indentation>",
    "explanation": "<2 sentences: what changes and why>"
}"""


def _build_brain_prompt(metrics: dict, anomalies: List[Anomaly],
                        ignored_titles: List[str]) -> str:
    # Compact metrics
    m = metrics
    wr      = m["smoothed_wr"]
    n_loss  = 5
    prior   = m["cfg"]["prior_k"]
    hl      = m["cfg"]["half_life_h"]
    lhs     = prior * 0.5 / WIN_RATE_THRESHOLD - prior
    t_rec   = hl * math.log2(n_loss / lhs) if lhs > 0 and n_loss > lhs else 0

    lines = [
        "=== METRICS ===",
        f"Regime: {m['regime']} | ADX={m['adx']:.0f} | breadth={m['breadth']:.0%} | "
        f"BTC.D={m['btc_d']:.1f}%",
        f"Drought: {m['drought_h']}h | Trades: {m['trade_count']} "
        f"({m['wins']}W/{m['losses']}L) | WR smoothed: {wr:.1%}",
        f"WR recovery at {hl}h half-life: ~{t_rec:.0f}h | prior_strength: {prior}",
        f"Trade direction: {m['longs']}L/{m['shorts']}S | "
        f"Long WR={m['long_wr']:.0%} avg R={m['long_avg_r']:+.2f} | "
        f"Short WR={m['short_wr']:.0%} avg R={m['short_avg_r']:+.2f}",
        f"Micro kills/2h: SELL OB={m['sell_ob_kills']} (CVD-aligned={m['sell_ob_cvd_aligned']}) "
        f"vs BUY OB={m['buy_ob_kills']}",
        f"RAG: {m['rag_dominant']} top match {m['rag_dominant_pct']:.0%} | "
        f"loss_trade={m['rag_dominant_loss']}",
        f"Rejections: {m['gate_counts']}",
        f"Data quality warnings: {m['persistent_dq']}",
        f"Exit reasons: {m['exit_reasons']}",
        f"Config: winrate_half_life_hours={hl} | winrate_prior_strength={prior} | "
        f"profile={m['profile']} | min_conf={m['min_conf']}",
    ]

    # Gate complementarity: per-gate blocked-trade precision vs the taken
    # baseline + redundancy overlap. A gate whose blocks beat the baseline is
    # filtering winners; high overlap means two gates duplicate each other.
    gs, gt = m.get("gate_shadow_stats") or {}, m.get("gate_taken_stats") or {}
    if gs and _shadow_block:
        lines += ["", "=== GATE COMPLEMENTARITY (forward shadow data) ==="]
        if gt:
            lines.append(f"Baseline taken trades: n={gt['n']} "
                         f"WR={gt['win_rate']:.0%} avg_R={gt['mean_r']:+.2f}")
        lines.append(_shadow_block(gs, days=30))

    lines += [
        "",
        "=== ANOMALIES ===",
    ]
    for a in anomalies:
        lines.append(f"[{a.code}] {a.severity}: {a.bullet}")

    if ignored_titles:
        lines += ["", "=== IGNORED (do not re-propose) ==="]
        lines += [f"- {t}" for t in ignored_titles]

    # Relevant source snippets
    snippets_added = set()
    for a in anomalies:
        if a.code in SOURCE_MAP and a.code not in snippets_added:
            rel, s, e = SOURCE_MAP[a.code]
            snippet = _load_source_snippet(rel, s, e)
            if snippet:
                lines += ["", f"=== SOURCE: {rel} (lines {s}-{e}) ===", snippet]
                snippets_added.add(a.code)

    return "\n".join(lines)


def run_brain(metrics: dict, anomalies: List[Anomaly],
              ignored_titles: List[str]) -> List[dict]:
    """Call DeepSeek V3 (chat), return list of raw proposal dicts."""
    # Use the same client setup as the rest of the bot
    sys.path.insert(0, str(ROOT / "bot"))
    try:
        from openai import OpenAI
        from core.config import get_deepseek_key
        api_key = get_deepseek_key()
    except Exception as e:
        log.warning(f"Could not load DeepSeek client: {e}")
        return []

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = _build_brain_prompt(metrics, anomalies, ignored_titles)

    try:
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": BRAIN_SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.1,
            max_tokens=1500,
        )
        raw = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        # Find outermost JSON array
        start = raw.find("[")
        end   = raw.rfind("]")
        if 0 <= start < end:
            try:
                return json.loads(raw[start: end + 1])
            except json.JSONDecodeError as je:
                log.warning(f"Brain JSON parse failed: {je} | raw: {raw[:300]!r}")
        else:
            log.warning(f"Brain: no JSON array in response: {raw[:200]!r}")
    except Exception as e:
        log.warning(f"Brain call failed: {e}")
    return []


# ── Proposal store ─────────────────────────────────────────────────────────────

def _load_store() -> dict:
    if PROPOSALS_PATH.exists():
        try:
            with open(PROPOSALS_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"last_message_id": None, "last_update_id": 0,
            "last_poll_ts": 0, "proposals": []}


def _save_store(store: dict):
    PROPOSALS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROPOSALS_PATH, "w") as f:
        json.dump(store, f, indent=2)


def _get_pending(store: dict) -> List[dict]:
    now = datetime.now(timezone.utc).isoformat()
    result = []
    for p in store.get("proposals", []):
        if p["status"] == "pending":
            exp = p.get("expires_at", "")
            if exp and exp < now:
                p["status"] = "expired"
            else:
                result.append(p)
    return result


def _get_ignored_titles(store: dict) -> List[str]:
    return [p["title"] for p in store.get("proposals", [])
            if p["status"] == "ignored"]


def _build_proposals(raw: List[dict], run_ts: str, start_num: int) -> List[dict]:
    now     = datetime.now(timezone.utc)
    expires = (now + timedelta(days=PROPOSAL_TTL)).isoformat()
    out: List[dict] = []
    for i, r in enumerate(raw[:5]):
        pid = f"p_{now.strftime('%Y%m%d')}_{start_num + i:03d}"
        p: dict = {
            "id":          pid,
            "run_ts":      run_ts,
            "num":         start_num + i,
            "priority":    r.get("priority", "MEDIUM"),
            "type":        r.get("type", "config"),
            "title":       r.get("title", ""),
            "evidence":    r.get("evidence", ""),
            "status":      "pending",
            "proposed_at": now.isoformat(),
            "decided_at":  "",
            "expires_at":  expires,
        }
        if p["type"] in ("config", "operational"):
            p["config_path"] = r.get("config_path", "")
            p["from_value"]  = str(r.get("from_value", ""))
            p["to_value"]    = str(r.get("to_value", ""))
        else:
            p["file_path"]     = r.get("file_path", "")
            p["current_code"]  = r.get("current_code", "")
            p["proposed_code"] = r.get("proposed_code", "")
            p["explanation"]   = r.get("explanation", "")
        out.append(p)
    return out


# ── Proposal applier ───────────────────────────────────────────────────────────

def _icon(priority: str) -> str:
    return {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(priority, "•")


def _apply_config(p: dict) -> str:
    """Edit config_futures.yaml in-place using regex (preserves comments).
    Only matches real key lines — not comment lines that mention the key.
    """
    key   = p["config_path"].split(".")[-1]
    to_v  = p["to_value"]
    text  = CONFIG_PATH.read_text()
    # Match only non-comment lines: optional whitespace, NOT '#', then key: value
    new_text, n = re.subn(
        rf"(?m)^([ \t]*)(?!#)({re.escape(key)}:[ \t]*)(\S+)",
        lambda m_: m_.group(1) + m_.group(2) + to_v,
        text, count=1
    )
    if n == 0:
        return f"❌ Could not find <code>{key}</code> in config_futures.yaml"
    CONFIG_PATH.write_text(new_text)
    return f"✅ config: {key} {p['from_value']} → {to_v}"


def _apply_blacklist(p: dict) -> str:
    """Add a symbol to scanner.blacklist in config_futures.yaml."""
    symbol = p["to_value"].strip()
    text   = CONFIG_PATH.read_text()
    # Find blacklist block and check if already there
    if symbol in text:
        return f"ℹ️ {symbol} already in config"
    new_text = re.sub(
        r"(blacklist:\n(?:[ \t]*- \S+\n)*)",
        lambda m_: m_.group(0) + f"  - {symbol}\n",
        text
    )
    if new_text == text:
        return f"❌ Could not locate blacklist in config_futures.yaml"
    CONFIG_PATH.write_text(new_text)
    return f"✅ {symbol} added to scanner.blacklist"


def _save_code_change(p: dict) -> str:
    """Append the code change to data/proposed_changes.md."""
    PROPOSED_CODE.parent.mkdir(parents=True, exist_ok=True)
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    block   = (
        f"\n## [{now_str}] {p['title']}\n"
        f"**Priority:** {p['priority']}  \n"
        f"**File:** `{p['file_path']}`  \n"
        f"**Evidence:** {p['evidence']}\n\n"
        f"> ⚠️ Review the logic below before applying — tell Claude:"
        f" \"review and apply the change in proposed_changes.md\"\n\n"
        f"**Current:**\n```python\n{p['current_code']}\n```\n\n"
        f"**Proposed:**\n```python\n{p['proposed_code']}\n```\n\n"
        f"**Why:** {p['explanation']}\n\n---\n"
    )
    with open(PROPOSED_CODE, "a") as f:
        f.write(block)
    return f"📄 Saved to proposed_changes.md → open Claude session and say 'review and apply proposed_changes.md'"


def apply_proposal(p: dict) -> str:
    ptype = p.get("type", "")
    if ptype == "code":
        return _save_code_change(p)
    if ptype == "operational" and "blacklist" in p.get("config_path", "").lower():
        return _apply_blacklist(p)
    if ptype in ("config", "operational"):
        return _apply_config(p)
    return f"❓ Unknown proposal type: {ptype}"


# ── Telegram ───────────────────────────────────────────────────────────────────

def tg_send(text: str) -> Optional[int]:
    cfg = _load_telegram_cfg()
    if not cfg.get("token") or not cfg.get("chat_id"):
        return None
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{cfg['token']}/sendMessage",
            json={"chat_id": cfg["chat_id"], "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        if resp.ok:
            return resp.json().get("result", {}).get("message_id")
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")
    return None


def tg_get_updates(last_update_id: int = 0) -> Tuple[List[dict], int]:
    cfg = _load_telegram_cfg()
    if not cfg.get("token"):
        return [], last_update_id
    try:
        resp = requests.get(
            f"https://api.telegram.org/bot{cfg['token']}/getUpdates",
            params={"offset": last_update_id + 1, "timeout": 0, "limit": 20},
            timeout=10,
        )
        if not resp.ok:
            return [], last_update_id
        updates  = resp.json().get("result", [])
        new_id   = max((u["update_id"] for u in updates), default=last_update_id)
        messages = [u["message"] for u in updates if "message" in u]
        return messages, new_id
    except Exception as e:
        log.warning(f"getUpdates failed: {e}")
    return [], last_update_id


def _parse_reply(text: str, pending: List[dict]) -> Tuple[List[str], List[str]]:
    """Return (accept_ids, ignore_ids) from a reply like '1 3 ignore 2'."""
    text      = text.strip().lower()
    accept_ns: List[int] = []
    ignore_ns: List[int] = []

    # "ignore N" or "ignore N M"
    for m in re.finditer(r"ignore\s+([\d\s,]+)", text):
        for n in re.findall(r"\d+", m.group(1)):
            ignore_ns.append(int(n))

    # Remaining numbers not after "ignore" → accept
    cleaned = re.sub(r"ignore\s+[\d\s,]+", "", text)
    for n in re.findall(r"\d+", cleaned):
        accept_ns.append(int(n))

    num_map = {p["num"]: p["id"] for p in pending}
    accept_ids = [num_map[n] for n in accept_ns if n in num_map]
    ignore_ids = [num_map[n] for n in ignore_ns if n in num_map]
    return accept_ids, ignore_ids


# ── Report formatter ───────────────────────────────────────────────────────────

def format_report(metrics: dict, anomalies: List[Anomaly],
                  proposals: List[dict], store: dict) -> str:
    now     = datetime.now(timezone.utc).strftime("%b %-d %H:%M")
    drought = f" | {metrics['drought_h']:.0f}h no trade" if metrics["drought_h"] >= 12 else ""
    parts   = [f"🔬 <b>Audit — {now}{drought}</b>"]

    if anomalies:
        parts.append("\n<b>Issues:</b>")
        for a in anomalies:
            parts.append(f"• {a.bullet}")

    pending = [p for p in proposals if p["status"] == "pending"]
    if pending:
        parts.append("\n<b>Proposed:</b>")
        for p in pending:
            line = f"[{p['num']}] {_icon(p['priority'])} "
            if p["type"] == "config":
                line += f"config: {p['config_path'].split('.')[-1]} "
                line += f"{p['from_value']} → {p['to_value']}"
            elif p["type"] == "code":
                line += f"{p['file_path'].split('/')[-1]} — {p['title']}"
            else:
                line += p["title"]
            parts.append(line)
        parts.append("\nReply numbers to apply · <i>ignore N</i> to dismiss")

    # Previous proposal status
    prev_pending  = [p for p in store.get("proposals", [])
                     if p["status"] == "pending" and p not in pending]
    prev_ignored  = [p for p in store.get("proposals", []) if p["status"] == "ignored"]
    prev_accepted = [p for p in store.get("proposals", []) if p["status"] == "accepted"]
    status_parts  = []
    if prev_pending:
        status_parts.append(f"{len(prev_pending)} pending")
    if prev_accepted:
        status_parts.append(f"{len(prev_accepted)} applied")
    if prev_ignored:
        status_parts.append(f"{len(prev_ignored)} ignored")
    if status_parts:
        parts.append(f"<i>Previous: {' · '.join(status_parts)}</i>")

    return "\n".join(parts)


# ── Main flows ─────────────────────────────────────────────────────────────────

def run_audit(no_telegram: bool = False):
    log.info("System Auditor — running audit")
    metrics   = collect_metrics()
    anomalies = detect_anomalies(metrics)

    log.info(f"Anomalies: {[a.code for a in anomalies]}")

    store          = _load_store()
    ignored_titles = _get_ignored_titles(store)

    # R1 brain — only when actionable anomalies exist
    raw_proposals: List[dict] = []
    if any(a.severity in ("WARN", "ERROR") for a in anomalies):
        raw_proposals = run_brain(metrics, anomalies, ignored_titles)
        log.info(f"R1 proposed {len(raw_proposals)} changes")
    else:
        log.info("No significant anomalies — skipping R1 call")

    # Build and save proposals
    existing_count = len(store.get("proposals", []))
    new_proposals  = _build_proposals(raw_proposals,
                                      datetime.now(timezone.utc).isoformat(),
                                      existing_count + 1)
    store.setdefault("proposals", []).extend(new_proposals)
    _save_store(store)

    report = format_report(metrics, anomalies, new_proposals, store)
    log.info("\n" + report)

    if not no_telegram:
        msg_id = tg_send(report)
        if msg_id:
            store["last_message_id"] = msg_id
            _save_store(store)
            log.info(f"Telegram: sent (message_id={msg_id})")
        else:
            log.info("Telegram: failed or not configured")


def run_poll(no_telegram: bool = False):
    store   = _load_store()
    pending = _get_pending(store)

    if not pending:
        log.info("Poll: no pending proposals")
        return

    messages, new_update_id = tg_get_updates(store.get("last_update_id", 0))
    if new_update_id > store.get("last_update_id", 0):
        store["last_update_id"] = new_update_id

    target_msg_id = store.get("last_message_id")
    replies = []
    for msg in messages:
        reply_to = msg.get("reply_to_message", {}).get("message_id")
        if reply_to == target_msg_id and msg.get("text"):
            replies.append(msg["text"])

    if not replies:
        _save_store(store)
        return

    results: List[str] = []
    all_proposals = store["proposals"]

    for reply_text in replies:
        accept_ids, ignore_ids = _parse_reply(reply_text, pending)

        for pid in accept_ids:
            p = next((x for x in all_proposals if x["id"] == pid), None)
            if not p or p["status"] != "pending":
                continue
            result = apply_proposal(p)
            p["status"]     = "accepted"
            p["decided_at"] = datetime.now(timezone.utc).isoformat()
            results.append(result)

        for pid in ignore_ids:
            p = next((x for x in all_proposals if x["id"] == pid), None)
            if not p or p["status"] != "pending":
                continue
            p["status"]     = "ignored"
            p["decided_at"] = datetime.now(timezone.utc).isoformat()
            results.append(f"🚫 [{p['num']}] {p['title']} — ignored (won't re-raise)")

    _save_store(store)

    if results:
        confirmation = "\n".join(results)
        log.info(f"Applied:\n{confirmation}")
        if not no_telegram:
            tg_send(confirmation)


def main():
    parser = argparse.ArgumentParser(description="SystemAuditor — meta-oversight agent")
    parser.add_argument("--poll",        action="store_true",
                        help="Check Telegram for pending replies and apply")
    parser.add_argument("--loop",        action="store_true",
                        help=f"Audit every {AUDIT_INTERVAL}min + poll every {POLL_INTERVAL}min")
    parser.add_argument("--no-telegram", action="store_true",
                        help="Print only, no Telegram")
    args = parser.parse_args()

    if args.poll:
        run_poll(no_telegram=args.no_telegram)
        return

    if args.loop:
        log.info(f"Loop: audit every {AUDIT_INTERVAL}min, poll every {POLL_INTERVAL}min")
        last_audit = 0.0
        while True:
            now_ts = time.time()
            if now_ts - last_audit >= AUDIT_INTERVAL * 60:
                run_audit(no_telegram=args.no_telegram)
                last_audit = now_ts
            run_poll(no_telegram=args.no_telegram)
            time.sleep(POLL_INTERVAL * 60)
        return

    # Default: run once
    run_audit(no_telegram=args.no_telegram)


if __name__ == "__main__":
    main()
