#!/usr/bin/env python3
"""
Signal Monitor Agent — evaluates rejected signals against the bot's own thresholds
and fires a Telegram alert when a systemic issue is detected.

Run once:  python3 scripts/signal_monitor.py
Loop:      python3 scripts/signal_monitor.py --loop [--interval 30]

All check thresholds are derived from config_futures.yaml or the fixed constants
in risk_agent.py / llm_reasoning.py — no arbitrary values invented here.
"""

import argparse
import json
import logging
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import requests
import yaml

# ── Paths ─────────────────────────────────────────────────────────
BOT_ROOT   = Path(__file__).parent.parent
DATA_DIR   = BOT_ROOT / "data"
LOGS_DIR   = BOT_ROOT / "logs"
CFG_FILE   = BOT_ROOT / "config_futures.yaml"
STATE_FILE = DATA_DIR / "futures_state.json"
MON_LOG    = LOGS_DIR / "signal_monitor.log"

# Fixed code constants (match risk_agent.py / llm_reasoning.py exactly)
BREADTH_SOFT_THRESHOLD   = 0.50   # risk_agent.py:175/186
BREADTH_HARD_BLOCK       = 0.70   # risk_agent.py:170  (STRONG_TREND shorts)
BREADTH_MAX_PENALTY      = 0.15   # risk_agent.py:176/187
WIN_RATE_PENALTY_TRIGGER = 0.40   # llm_reasoning.py:108  (below this → −0.15)
WIN_RATE_PENALTY_AMOUNT  = 0.15   # llm_reasoning.py:108
CONF_PENALTY_FLOOR_CAP   = 0.80   # risk_agent.py:179/191

# Detect log offset once at startup: logs use local system time
_local_now = datetime.now()
_utc_now   = datetime.now(timezone.utc).replace(tzinfo=None)
LOG_UTC_OFFSET = _local_now - _utc_now   # positive = local is ahead of UTC

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(MON_LOG, mode="a"),
    ],
)
log = logging.getLogger("signal_monitor")


# ── Data classes ──────────────────────────────────────────────────
@dataclass
class Finding:
    level: str          # "ERROR", "WARN", "INFO"
    check: str          # e.g. "Check 1"
    title: str
    detail: str
    suggestion: str = ""

    @property
    def emoji(self):
        return {"ERROR": "❌", "WARN": "⚠️", "INFO": "ℹ️"}.get(self.level, "•")


# ── Config loading ─────────────────────────────────────────────────
def load_config() -> dict:
    with open(CFG_FILE) as f:
        raw = yaml.safe_load(f)
    actor = raw.get("actor", {})
    ens   = raw.get("ensemble", {})
    tf    = raw.get("trend_filter", {})
    return {
        "half_life_h":   float(actor.get("winrate_half_life_hours", 48)),
        "prior_k":       float(actor.get("winrate_prior_strength", 1.0)),
        "min_conf":      float(ens.get("min_confidence", 0.45)),
        "slope_thresh":  float(tf.get("strong_slope_pct", 0.02)),
    }


# ── State loading ──────────────────────────────────────────────────
def load_state():
    with open(STATE_FILE) as f:
        state = json.load(f)
    signals = state.get("signals", [])
    live    = state.get("live_strategy", {})
    return signals, live


# ── Log parsing ───────────────────────────────────────────────────
def _log_ts_to_utc(ts_str: str) -> Optional[datetime]:
    """Parse log timestamp (local system time) and return as UTC-aware datetime."""
    try:
        dt = datetime.strptime(ts_str, "%Y-%m-%d %H:%M:%S")
        # dt is in local time; convert to UTC by subtracting the detected offset
        return (dt - LOG_UTC_OFFSET).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def parse_log_tail(minutes: int = 60) -> List[str]:
    """Read last `minutes` of futures_bot.log, return matching lines."""
    p = LOGS_DIR / "futures_bot.log"
    if not p.exists():
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=minutes)
    lines = []
    # Read tail: up to 500 KB to avoid loading multi-MB log fully
    with open(p, errors="replace") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 500_000))
        raw = f.readlines()
    for line in raw:
        m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        if m:
            ts = _log_ts_to_utc(m.group(1))
            if ts and ts >= cutoff:
                lines.append(line.rstrip())
    return lines


def parse_trend_veto_lines(lines: List[str]) -> List[dict]:
    """Extract TREND VETO entries for was-SELL from log lines."""
    result = []
    for line in lines:
        if "TREND VETO" not in line or "(was SELL)" not in line:
            continue
        ts_m   = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
        sym_m  = re.search(r"TREND VETO (\S+)", line)
        slope_m = re.search(r"slope=([0-9.]+)", line)
        reason_m = re.search(r"2tier:(.*)", line)
        ts = _log_ts_to_utc(ts_m.group(1)) if ts_m else None
        result.append({
            "ts":      ts,
            "symbol":  sym_m.group(1) if sym_m else "?",
            "slope":   float(slope_m.group(1)) if slope_m else None,
            "reason":  reason_m.group(1).strip() if reason_m else line,
        })
    return result


def parse_regime_gate_lines(lines: List[str]) -> List[dict]:
    """Extract [MarketRegimeGate] breadth readings."""
    result = []
    for line in lines:
        if "[MarketRegimeGate]" not in line:
            continue
        b_m   = re.search(r"breadth=(\d+)%", line)
        adx_m = re.search(r"ADX=([0-9.]+)", line)
        if b_m:
            breadth = int(b_m.group(1)) / 100.0
            result.append({
                "breadth":      breadth,
                "bear_breadth": round(1.0 - breadth, 4),
                "adx":          float(adx_m.group(1)) if adx_m else None,
            })
    return result


def parse_order_timestamps() -> List[datetime]:
    """Collect all 'Order confirmed' timestamps from all log files."""
    ts_list = []
    for fname in ["futures_bot.log.2", "futures_bot.log.1", "futures_bot.log"]:
        p = LOGS_DIR / fname
        if not p.exists():
            continue
        with open(p, errors="replace") as f:
            for line in f:
                if "Order confirmed" not in line:
                    continue
                m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                if m:
                    ts = _log_ts_to_utc(m.group(1))
                    if ts:
                        ts_list.append(ts)
    ts_list.sort()
    return ts_list


def get_breadth_ctx(live: dict, log_lines: List[str]) -> dict:
    """Return breadth context: prefer live_strategy, fall back to log."""
    if live.get("breadth") is not None:
        return {
            "breadth":      live["breadth"],
            "bear_breadth": live.get("bear_breadth", 1 - live["breadth"]),
            "adx":          live.get("adx"),
            "regime":       live.get("market_regime", "UNKNOWN"),
            "source":       "state",
        }
    # Fallback: last MarketRegimeGate line in current log
    gate_lines = parse_regime_gate_lines(log_lines)
    if gate_lines:
        latest = gate_lines[-1]
        return {**latest, "regime": live.get("market_regime", "UNKNOWN"), "source": "log"}
    # No data at all
    return {"breadth": None, "bear_breadth": None, "adx": None,
            "regime": live.get("market_regime", "UNKNOWN"), "source": "none"}


# ── Categorisation helpers ─────────────────────────────────────────
def categorise_rejection(s: dict) -> str:
    r = s.get("reason", "")
    if r.startswith("actor:"):              return "actor_llm"
    if r.startswith("microstructure kill"): return "micro_kill"
    if "breadth" in r:                      return "breadth_penalty"
    if "conf=" in r or "eff=" in r:         return "conf_threshold"
    if "regime" in r.lower():               return "regime_gate"
    if "HTF" in r:                          return "htf_filter"
    if "EMA" in r or "20EMA" in r:          return "ema_momentum"
    return "other"


def _net_score(s: dict) -> Optional[float]:
    m = re.search(r"ensemble:([+-]?\d+\.\d+)", s.get("strategy", ""))
    return float(m.group(1)) if m else None


# ── Checks ────────────────────────────────────────────────────────

def check_winrate_penalty(signals: List[dict], cfg: dict) -> Optional[Finding]:
    """Check 1: Signals blocked solely by the RAG win-rate penalty."""
    rejected = [s for s in signals if s.get("status") == "rejected"]
    actor_rej = [s for s in rejected if s.get("reason", "").startswith("actor:")]

    winrate_blocked = [
        s for s in actor_rej
        if s.get("confidence", 1.0) <= 0.21
        and any(kw in s.get("reason", "").lower()
                for kw in ("recency", "win-rate", "win rate", "winrate"))
    ]
    if len(winrate_blocked) < 5:
        return None

    symbols = set(s["symbol"] for s in winrate_blocked)
    nets = [n for s in winrate_blocked if (n := _net_score(s)) is not None]
    if not nets:
        return None

    neg_pct = sum(1 for n in nets if n < 0) / len(nets)
    pos_pct = 1 - neg_pct
    if max(neg_pct, pos_pct) < 0.70:
        return None  # ensemble is mixed — not a clear directional block

    direction = "bearish (SELL)" if neg_pct >= 0.70 else "bullish (BUY)"
    avg_net   = sum(nets) / len(nets)

    # Recovery time: wr(t) = prior*0.5 / (prior + n_losses * 2^(-t/h)) >= WIN_RATE_PENALTY_TRIGGER
    half_life_h = cfg["half_life_h"]
    prior_k     = cfg["prior_k"]
    n_losses    = 5  # RAG uses top-5 similar trades
    lhs = prior_k * 0.5 / WIN_RATE_PENALTY_TRIGGER - prior_k
    if lhs > 0 and n_losses > lhs:
        t_recover    = half_life_h * math.log2(n_losses / lhs)
        t_recover_24 = 24.0       * math.log2(n_losses / lhs)
    else:
        t_recover = t_recover_24 = 0.0

    current_wr = prior_k * 0.5 / (prior_k + n_losses)

    detail = (
        f"{len(winrate_blocked)} rejections across {len(symbols)} symbols — "
        f"win-rate penalty is sole blocker.\n"
        f"  Ensemble: {direction}, avg net={avg_net:+.3f}\n"
        f"  Current win-rate (n={n_losses}, age≈0): {current_wr:.1%}\n"
        f"  Recovery at half_life={half_life_h:.0f}h: ~{t_recover:.0f}h ({t_recover/24:.1f} days)"
    )
    suggestion = (
        f"Reduce <code>winrate_half_life_hours</code> {half_life_h:.0f}→24 "
        f"(recovers in ~{t_recover_24:.0f}h / {t_recover_24/24:.1f} days)\n"
        f"  OR reset trade memory for affected symbols"
    )
    return Finding("WARN", "Check 1", "RAG WIN-RATE PENALTY", detail, suggestion)


def check_trend_slopes(veto_lines: List[dict], cfg: dict, breadth_ctx: dict) -> Optional[Finding]:
    """Check 2: SELL vetoes where slope barely exceeds strong_slope_pct."""
    slope_thresh = cfg["slope_thresh"]
    bear_breadth = breadth_ctx.get("bear_breadth")

    slow_up = [v for v in veto_lines if "slow strongly up" in v.get("reason", "")]
    if not slow_up:
        return None

    with_slope = [v for v in slow_up if v["slope"] is not None]
    if not with_slope:
        return None

    borderline = [v for v in with_slope if 1.0 < v["slope"] / slope_thresh <= 2.0]
    macro_conflict = bear_breadth is not None and bear_breadth > BREADTH_SOFT_THRESHOLD

    if not borderline and not macro_conflict:
        return None

    symbols_bl  = sorted(set(v["symbol"] for v in borderline))
    symbols_all = sorted(set(v["symbol"] for v in slow_up))
    slopes_bl   = [v["slope"] for v in borderline]

    detail_parts = []
    if borderline:
        detail_parts.append(
            f"{len(borderline)} borderline vetoes (slope 1–2× threshold={slope_thresh}) "
            f"on: {', '.join(symbols_bl)}\n"
            f"  Slope range: {min(slopes_bl):.4f}–{max(slopes_bl):.4f} "
            f"({min(slopes_bl)/slope_thresh:.1f}×–{max(slopes_bl)/slope_thresh:.1f}× threshold)"
        )
    if macro_conflict:
        detail_parts.append(
            f"bear_breadth={bear_breadth:.0%} > 50% — macro confirms bearish direction "
            f"while {len(slow_up)} SELL signals vetoed by 1h uptrend slope"
        )
    detail = "\n  ".join(detail_parts)

    # Suggest new threshold that frees borderline symbols
    if slopes_bl:
        suggested = round(max(slopes_bl) * 1.1, 3)
        still_blocked = sorted(set(v["symbol"] for v in with_slope if v["slope"] > suggested))
        suggestion = (
            f"Raise <code>strong_slope_pct</code> {slope_thresh}→{suggested} "
            f"to free borderline symbols.\n"
            f"  Still blocked at {suggested}: "
            + (", ".join(still_blocked) if still_blocked else "none")
        )
    else:
        suggestion = (
            f"bear_breadth={bear_breadth:.0%} confirms bearish macro. "
            f"Consider adding a bear_breadth override to skip VETO_SLOW_UP when "
            f"bear_breadth > 65%."
        )

    level = "WARN" if borderline else "INFO"
    return Finding(level, "Check 2", "TREND SLOPE vs MACRO", detail, suggestion)


def check_breadth_gate(breadth_ctx: dict, cfg: dict) -> Finding:
    """Check 3: Always report current breadth gate requirements (INFO)."""
    breadth      = breadth_ctx.get("breadth")
    bear_breadth = breadth_ctx.get("bear_breadth")
    regime       = breadth_ctx.get("regime", "?")
    adx          = breadth_ctx.get("adx")
    min_conf     = cfg["min_conf"]

    lines = []
    adx_str = f" | ADX={adx}" if adx else ""
    if breadth is not None:
        lines.append(f"breadth={breadth:.0%} bear={bear_breadth:.0%} | {regime}{adx_str}")
    else:
        lines.append(f"regime={regime} (breadth not yet in state — run one bot cycle)")

    if bear_breadth is not None and bear_breadth > BREADTH_SOFT_THRESHOLD:
        pen = min(BREADTH_MAX_PENALTY, (bear_breadth - BREADTH_SOFT_THRESHOLD) * 1.5)
        floor = min(min_conf + pen, CONF_PENALTY_FLOOR_CAP)
        lines.append(f"BUY:  need conf ≥ {floor:.2f} (penalty +{pen:.3f} for bear_breadth={bear_breadth:.0%})")
    else:
        lines.append("BUY:  no breadth penalty")

    if breadth is not None:
        if breadth > BREADTH_HARD_BLOCK and regime == "STRONG_TREND":
            lines.append(f"SELL: HARD BLOCKED (breadth={breadth:.0%} > 70% STRONG_TREND)")
        elif breadth > BREADTH_SOFT_THRESHOLD:
            pen = min(BREADTH_MAX_PENALTY, (breadth - BREADTH_SOFT_THRESHOLD) * 1.5)
            floor = min(min_conf + pen, CONF_PENALTY_FLOOR_CAP)
            lines.append(f"SELL: need conf ≥ {floor:.2f} (penalty +{pen:.3f} for breadth={breadth:.0%})")
        else:
            lines.append(f"SELL: no breadth penalty — macro confirms bearish direction")

    return Finding("INFO", "Check 3", "BREADTH GATE (live)", "\n  ".join(lines))


def check_drought(order_ts: List[datetime]) -> Optional[Finding]:
    """Check 4: Current drought vs historical median inter-order gap."""
    if not order_ts:
        return Finding("WARN", "Check 4", "DROUGHT", "No order history found in logs.")

    gaps_h = [
        (order_ts[i + 1] - order_ts[i]).total_seconds() / 3600
        for i in range(len(order_ts) - 1)
    ]
    normal_gaps = [g for g in gaps_h if g < 24.0]
    if not normal_gaps:
        return None

    median_gap = sorted(normal_gaps)[len(normal_gaps) // 2]
    last_order = order_ts[-1]
    drought_h  = (datetime.now(timezone.utc) - last_order).total_seconds() / 3600
    ratio      = drought_h / median_gap if median_gap > 0 else 0

    if ratio < 2.0:
        return None

    last_str = last_order.strftime("%Y-%m-%d %H:%M UTC")
    detail = (
        f"Drought: {drought_h:.1f}h ({ratio:.0f}× median gap of {median_gap:.1f}h)\n"
        f"  Last confirmed order: {last_str}"
    )
    level = "ERROR" if ratio > 10 else "WARN"
    return Finding(level, "Check 4", "TRADING DROUGHT", detail)


def check_ob_feed(signals: List[dict]) -> Optional[Finding]:
    """Check 5: Order book data feed health."""
    rejected    = [s for s in signals if s.get("status") == "rejected"]
    micro_kills = [s for s in rejected if s.get("reason", "").startswith("microstructure kill")]
    if not micro_kills:
        return None

    ob_unavail = [s for s in micro_kills if "OB: unavailable" in s.get("reason", "")]
    pct = len(ob_unavail) / len(micro_kills)
    if pct < 0.50:
        return None

    detail = (
        f"{len(ob_unavail)}/{len(micro_kills)} micro-kills show \"OB: unavailable\" "
        f"({pct:.0%})\n"
        f"  Order book data is missing for all symbols — OB imbalance signal absent."
    )
    suggestion = "Check microstructure WebSocket / REST fallback connection for order book feed."
    return Finding("ERROR", "Check 5", "OB FEED DOWN", detail, suggestion)


def check_gate_monopoly(signals: List[dict], veto_lines: List[dict]) -> Optional[Finding]:
    """Check 6: One gate dominating rejections while ensemble has directional signal."""
    rejected = [s for s in signals if s.get("status") == "rejected"]
    if not rejected:
        return None

    cats = Counter(categorise_rejection(s) for s in rejected)
    total = len(rejected)
    dominant_cat, dominant_cnt = cats.most_common(1)[0]
    pct = dominant_cnt / total

    if pct < 0.90:
        return None

    # Check ensemble direction consistency for the dominant gate's rejections
    dom_sigs = [s for s in rejected if categorise_rejection(s) == dominant_cat]
    nets     = [n for s in dom_sigs if (n := _net_score(s)) is not None]
    if not nets:
        return None

    neg_pct = sum(1 for n in nets if n < 0) / len(nets)
    dir_pct = max(neg_pct, 1 - neg_pct)
    if dir_pct < 0.70:
        return None  # ensemble is mixed — gate may be correctly neutral

    direction = "bearish" if neg_pct >= 0.70 else "bullish"
    avg_net   = sum(nets) / len(nets)

    detail = (
        f"{dominant_cat} accounts for {dominant_cnt}/{total} rejections ({pct:.0%})\n"
        f"  Ensemble is {dir_pct:.0%} {direction} (avg net={avg_net:+.3f})\n"
        f"  Gate is effectively suppressing a consistent directional view"
    )
    suggestion = "Investigate the dominant gate — see Check 1/2 for specific cause."
    return Finding("WARN", "Check 6", f"GATE MONOPOLY: {dominant_cat}", detail, suggestion)


def check_winrate_recovery(signals: List[dict], cfg: dict) -> Finding:
    """Check 7: Always report win-rate recovery timeline (INFO)."""
    half_life_h = cfg["half_life_h"]
    prior_k     = cfg["prior_k"]
    n_losses    = 5

    current_wr = prior_k * 0.5 / (prior_k + n_losses)
    lhs = prior_k * 0.5 / WIN_RATE_PENALTY_TRIGGER - prior_k

    if lhs > 0 and n_losses > lhs:
        t_48 = half_life_h * math.log2(n_losses / lhs)
        t_24 = 24.0        * math.log2(n_losses / lhs)
        detail = (
            f"Current win-rate (n={n_losses} losses, age≈0): {current_wr:.1%} "
            f"(need ≥ {WIN_RATE_PENALTY_TRIGGER:.0%})\n"
            f"  At half_life={half_life_h:.0f}h: recovers in ~{t_48:.0f}h ({t_48/24:.1f} days)\n"
            f"  At half_life=24h: recovers in ~{t_24:.0f}h ({t_24/24:.1f} days)"
        )
    else:
        detail = f"Win-rate already above {WIN_RATE_PENALTY_TRIGGER:.0%} threshold."

    return Finding("INFO", "Check 7", "WIN-RATE RECOVERY ESTIMATE", detail)


# ── Reporting ─────────────────────────────────────────────────────

def _signal_stats(signals: List[dict], veto_lines: List[dict]) -> str:
    rejected = [s for s in signals if s.get("status") == "rejected"]
    holds    = [s for s in signals if s.get("status") == "hold"]
    takens   = [s for s in signals if s.get("status") == "taken"]

    cats = Counter(categorise_rejection(s) for s in rejected)
    veto_flat   = sum(1 for v in veto_lines if "flat" in v.get("reason", ""))
    veto_slowup = sum(1 for v in veto_lines if "slow strongly up" in v.get("reason", ""))
    veto_hist   = sum(1 for v in veto_lines if "insufficient history" in v.get("reason", ""))

    lines = [f"Last {len(signals)} signals: {len(rejected)} rejected | "
             f"{len(holds)} hold | {len(takens)} taken"]
    for cat, cnt in cats.most_common():
        lines.append(f"  {cat:22s}: {cnt:4d} ({100*cnt/max(1,len(rejected)):.0f}%)")
    if veto_lines:
        lines.append(f"  trend_veto (log, 60m)  : flat={veto_flat} slow_up={veto_slowup} "
                     f"history={veto_hist}")
    return "\n".join(lines)


def format_report(findings: List[Finding], signals: List[dict],
                  veto_lines: List[dict]) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines   = [f"<b>🔍 Signal Monitor — {now_str}</b>", ""]

    lines.append(_signal_stats(signals, veto_lines))
    lines.append("")

    for f in findings:
        lines.append(f"<b>{f.emoji} {f.check} — {f.title}</b>")
        for dl in f.detail.split("\n"):
            lines.append(f"  {dl}")
        if f.suggestion:
            lines.append(f"  → {f.suggestion}")
        lines.append("")

    return "\n".join(lines).strip()


# ── Telegram ──────────────────────────────────────────────────────

def _get_telegram_cfg() -> dict:
    sys.path.insert(0, str(BOT_ROOT / "bot"))
    try:
        from core.config import get_telegram_config
        return get_telegram_config()
    except Exception as e:
        log.warning(f"Could not load telegram config: {e}")
        return {}


def send_telegram(token: str, chat_id: str, message: str) -> bool:
    if not token or not chat_id:
        return False
    try:
        resp = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        return resp.status_code == 200
    except Exception as e:
        log.warning(f"Telegram send failed: {e}")
        return False


# ── Main run ──────────────────────────────────────────────────────

def run_once():
    log.info("Signal monitor running...")
    try:
        cfg = load_config()
    except Exception as e:
        log.error(f"Config load failed: {e}")
        return

    try:
        signals, live = load_state()
    except Exception as e:
        log.error(f"State load failed: {e}")
        return

    log_lines  = parse_log_tail(minutes=60)
    breadth_ctx = get_breadth_ctx(live, log_lines)
    veto_lines  = parse_trend_veto_lines(log_lines)
    order_ts    = parse_order_timestamps()

    findings: List[Finding] = []

    # Run all checks
    for fn in [
        lambda: check_winrate_penalty(signals, cfg),
        lambda: check_trend_slopes(veto_lines, cfg, breadth_ctx),
        lambda: check_breadth_gate(breadth_ctx, cfg),   # always returns Finding
        lambda: check_drought(order_ts),
        lambda: check_ob_feed(signals),
        lambda: check_gate_monopoly(signals, veto_lines),
        lambda: check_winrate_recovery(signals, cfg),    # always returns Finding
    ]:
        try:
            result = fn()
            if result:
                findings.append(result)
        except Exception as e:
            log.warning(f"Check error: {e}")

    report = format_report(findings, signals, veto_lines)
    log.info("\n" + report)

    # Send Telegram only if WARN or ERROR findings exist
    should_send = any(f.level in ("WARN", "ERROR") for f in findings)
    if should_send:
        tg = _get_telegram_cfg()
        sent = send_telegram(tg.get("token", ""), tg.get("chat_id", ""), report)
        log.info(f"Telegram: {'sent' if sent else 'failed (check credentials)'}")
    else:
        log.info("No WARN/ERROR findings — Telegram not sent")


def main():
    parser = argparse.ArgumentParser(description="Signal Monitor Agent")
    parser.add_argument("--loop",     action="store_true", help="Run continuously")
    parser.add_argument("--interval", type=int, default=30,
                        help="Loop interval in minutes (default: 30)")
    args = parser.parse_args()

    if args.loop:
        log.info(f"Loop mode: running every {args.interval} minutes")
        while True:
            run_once()
            time.sleep(args.interval * 60)
    else:
        run_once()


if __name__ == "__main__":
    main()
