"""
Self Learner v2
Improves bot parameters based on trade results.
Requires minimum 10 closed trades for statistical significance.

v3 (master-monitor): also tails the live futures log, detects toxic
patterns like repeated `momentum_reversal` invalidations (shorts in a
rising market), and asks Groq for code-level remediation suggestions.
"""

import json
import logging
import re
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta, timezone
try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False
from core.config import get_groq_key, DATA_DIR, BOT_ROOT

log  = logging.getLogger("SelfLearner")
DATA = DATA_DIR

# === v3 master-monitor constants ===
FUTURES_LOG_PATH    = BOT_ROOT / "logs" / "futures_bot.log"
MASTER_ANALYSIS_LOG = BOT_ROOT / "master_analysis.log"
_LOG_PATTERNS = ("SIGNAL", "ENSEMBLE", "SL EXIT", "INVALIDATION", "REJECTED")


class SelfLearner:
    MIN_TRADES   = 10
    REVIEW_HOURS = 2

    def __init__(self, notifier=None):
        self.client   = Groq(api_key=get_groq_key()) if _GROQ_AVAILABLE else None
        self.model    = "llama-3.1-8b-instant"
        self.insights = self._load_insights()
        # v3: optional Telegram notifier — may also be supplied at run_learning_cycle()
        self.notifier = notifier

    def _load_insights(self):
        p = DATA / "learning_insights.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return {"total_reviews": 0, "last_review": None,
                "performance_history": [], "adjustments": []}

    def _save_insights(self):
        with open(DATA / "learning_insights.json", "w") as f:
            json.dump(self.insights, f, indent=2)

    def _load_trades(self):
        # Load from both spot and futures state files, including archives
        trades = []
        for fname in ["state.json", "futures_state.json"]:
            p = DATA / fname
            if p.exists():
                with open(p) as f:
                    trades.extend(json.load(f).get("trades", []))
            # Also load from the corresponding archive file
            archive_fname = fname.replace(".json", "_archive.json")
            archive_p = DATA / archive_fname
            if archive_p.exists():
                try:
                    with open(archive_p) as f:
                        trades.extend(json.load(f))
                except Exception:
                    pass
        return trades

    def should_run(self):
        last = self.insights.get("last_review")
        if not last:
            return True
        dt = datetime.fromisoformat(last)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - dt) > timedelta(hours=self.REVIEW_HOURS)

    def analyze_performance(self):
        trades = self._load_trades()
        closed = [t for t in trades if t["status"] == "closed"]
        if len(closed) < self.MIN_TRADES:
            return {"error": f"Need {self.MIN_TRADES} closed trades (have {len(closed)})"}

        pnls   = [t.get("pnl", 0) for t in closed]
        wins   = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        wr     = len(wins) / len(closed) * 100
        pf     = abs(sum(wins) / sum(losses)) if losses and sum(losses) != 0 else 999

        # Statistical significance
        n       = len(closed)
        z_score = (wr/100 - 0.5) / (0.5 / n**0.5)

        return {
            "total_trades":  n,
            "win_rate":      round(wr, 1),
            "total_pnl":     round(sum(pnls), 4),
            "avg_win":       round(np.mean(wins), 4) if wins else 0,
            "avg_loss":      round(np.mean(losses), 4) if losses else 0,
            "profit_factor": round(pf, 2),
            "z_score":       round(z_score, 2),
            "significant":   abs(z_score) > 1.65,
            "recent_trades": closed[-10:],
        }

    def ask_ai_for_improvements(self, perf):
        prompt = f"""Bot performance analysis:
- Trades: {perf['total_trades']}, WR: {perf['win_rate']}%, PnL: ${perf['total_pnl']}
- Avg Win: ${perf['avg_win']}, Avg Loss: ${perf['avg_loss']}, PF: {perf['profit_factor']}
- Statistically significant: {perf['significant']}

JSON only:
{{"confidence_adjustment":-0.05_to_0.05,"stop_loss_adjustment":-0.005_to_0.005,"take_profit_adjustment":-0.01_to_0.01,"key_insight":"brief","should_retrain":true_or_false}}"""
        if not self.client:
            return {}
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role":"system","content":"Trading analyst. JSON only."},
                          {"role":"user","content":prompt}],
                max_tokens=150, temperature=0.1,
            )
            raw   = resp.choices[0].message.content.strip()
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            return json.loads(match.group()) if match else {}
        except Exception as e:
            log.error(f"AI improvement error: {e}")
            return {}

    def apply_improvements(self, improvements, perf=None):
        """Propose-only: writes recommendations to pending_recommendations.json
        and notifies via Telegram. Never mutates config files — user must
        confirm and apply manually."""
        if not improvements:
            return []
        proposals = []
        for cfg_file in ["config_spot.yaml", "config_futures.yaml"]:
            cfg_path = BOT_ROOT / cfg_file
            if not cfg_path.exists():
                continue
            with open(cfg_path) as f:
                config = yaml.safe_load(f)

            conf_adj = improvements.get("confidence_adjustment", 0)
            if conf_adj != 0:
                old = config["strategy"]["min_confidence"]
                new = round(max(0.42, min(0.70, old + conf_adj)), 3)
                if new != old:
                    proposals.append({
                        "file": cfg_file,
                        "section": "strategy.min_confidence",
                        "current": old,
                        "proposed": new,
                        "reason": (
                            f"AI insight: {improvements.get('key_insight','(none)')}. "
                            f"Suggested confidence_adjustment={conf_adj:+.3f}."
                        ),
                        "source": "ai_groq",
                    })

            sl_adj = improvements.get("stop_loss_adjustment", 0)
            if sl_adj != 0:
                old_sl = config["risk"]["stop_loss_atr_multiplier"]
                new_sl = round(max(0.5, min(4.0, old_sl + sl_adj * 100)), 2)
                if new_sl != old_sl:
                    proposals.append({
                        "file": cfg_file,
                        "section": "risk.stop_loss_atr_multiplier",
                        "current": old_sl,
                        "proposed": new_sl,
                        "reason": f"AI insight: {improvements.get('key_insight','(none)')}.",
                        "source": "ai_groq",
                    })

        return self._record_proposals(proposals, perf=perf)

    # ============================================================
    # === PROPOSAL LEDGER (propose-only, never auto-apply) =======
    # ============================================================

    PENDING_FILE = "pending_recommendations.json"

    def _load_pending(self) -> dict:
        p = DATA / self.PENDING_FILE
        if not p.exists():
            return {"pending": [], "applied": [], "rejected": []}
        try:
            with open(p) as f:
                d = json.load(f)
            d.setdefault("pending", [])
            d.setdefault("applied", [])
            d.setdefault("rejected", [])
            return d
        except Exception:
            return {"pending": [], "applied": [], "rejected": []}

    def _save_pending(self, data: dict) -> None:
        p = DATA / self.PENDING_FILE
        tmp = p.with_suffix(".tmp.json")
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        tmp.replace(p)

    @staticmethod
    def _proposal_key(p: dict) -> str:
        return f"{p['file']}::{p['section']}::{p['proposed']}"

    def _record_proposals(self, proposals: list, perf: dict | None = None) -> list:
        """Append unique proposals to the pending ledger and notify. Returns
        human-readable summary strings (for insights.adjustments log)."""
        if not proposals:
            return []
        ledger   = self._load_pending()
        existing = {self._proposal_key(p) for p in ledger["pending"]}
        ts       = datetime.now(timezone.utc).isoformat()
        new_recs = []
        for p in proposals:
            key = self._proposal_key(p)
            if key in existing:
                continue
            rec_id = f"rec_{int(datetime.now(timezone.utc).timestamp())}_{len(ledger['pending']) + len(new_recs)}"
            new_recs.append({
                "id":         rec_id,
                "timestamp":  ts,
                "status":     "pending",
                **p,
            })
        if not new_recs:
            log.info(f"Self-learner: {len(proposals)} proposals all duplicates of pending — no new alerts")
            return []
        ledger["pending"].extend(new_recs)
        self._save_pending(ledger)
        self._notify_proposals(new_recs, perf)
        return [
            f"PROPOSED {r['file']}: {r['section']} {r['current']}→{r['proposed']} ({r['source']})"
            for r in new_recs
        ]

    def _notify_proposals(self, proposals: list, perf: dict | None) -> None:
        if not self.notifier:
            log.warning("Self-learner has new proposals but no notifier set — see "
                        f"{DATA / self.PENDING_FILE}")
            return
        try:
            header = "SELF-LEARNER PROPOSALS (awaiting your approval)\n"
            if perf:
                header += (f"WR {perf.get('win_rate','?')}% PnL ${perf.get('total_pnl','?')} "
                           f"over {perf.get('total_trades','?')} trades\n")
            header += f"Ledger: {DATA / self.PENDING_FILE}\n\n"
            body_lines = []
            for r in proposals:
                body_lines.append(
                    f"[{r['id']}] {r['file']}\n"
                    f"  {r['section']}: {r['current']} → {r['proposed']}\n"
                    f"  Why: {r['reason']}\n"
                )
            footer = ("\nTo APPLY: edit the YAML manually OR move the entry from "
                      "'pending' to 'applied' in the ledger file, then restart the bot.")
            self.notifier.send_alert(header + "\n".join(body_lines) + footer)
        except Exception as e:
            log.warning(f"Notifier send_alert (proposals) failed: {e}")

    # ============================================================
    # === v3 MASTER-MONITOR ADDITIONS ============================
    # ============================================================

    def _analyse_recent_logs(self, lines: int = 300) -> dict:
        """
        Read the tail of futures_bot.log and extract signal/exit/invalidation
        lines. Returns a structured summary used by `_ask_groq_for_log_analysis`
        and surfaced in Telegram + master_analysis.log.

        Defensive: missing or unreadable log → returns empty summary, never raises.
        """
        summary = {
            "log_path":            str(FUTURES_LOG_PATH),
            "lines_scanned":       0,
            "signal_lines":        [],
            "ensemble_lines":      [],
            "sl_exit_lines":       [],
            "invalidation_lines":  [],
            "rejected_lines":      [],
            "momentum_reversal":   0,
            "short_signals":       0,
            "long_signals":        0,
            "error":               None,
        }
        try:
            if not FUTURES_LOG_PATH.exists():
                summary["error"] = f"log file not found: {FUTURES_LOG_PATH}"
                return summary

            # Tail efficiently — read last N lines without loading whole file.
            with open(FUTURES_LOG_PATH, "rb") as f:
                try:
                    f.seek(0, 2)
                    size  = f.tell()
                    block = 8192
                    data  = b""
                    while size > 0 and data.count(b"\n") <= lines:
                        step = min(block, size)
                        size -= step
                        f.seek(size)
                        data = f.read(step) + data
                except Exception:
                    f.seek(0)
                    data = f.read()
            tail = data.decode("utf-8", errors="replace").splitlines()[-lines:]
            summary["lines_scanned"] = len(tail)

            for ln in tail:
                if "SIGNAL" in ln:
                    summary["signal_lines"].append(ln)
                    if re.search(r"\bSELL\b|\bSHORT\b", ln):
                        summary["short_signals"] += 1
                    elif re.search(r"\bBUY\b|\bLONG\b", ln):
                        summary["long_signals"] += 1
                if "ENSEMBLE" in ln:
                    summary["ensemble_lines"].append(ln)
                if "SL EXIT" in ln:
                    summary["sl_exit_lines"].append(ln)
                if "INVALIDATION" in ln:
                    summary["invalidation_lines"].append(ln)
                if "momentum_reversal" in ln:
                    summary["momentum_reversal"] += 1
                if "REJECTED" in ln:
                    summary["rejected_lines"].append(ln)

            # Cap each bucket at 25 lines for token budget
            for k in ("signal_lines", "ensemble_lines", "sl_exit_lines",
                     "invalidation_lines", "rejected_lines"):
                summary[k] = summary[k][-25:]
        except Exception as e:
            log.warning(f"_analyse_recent_logs failed: {e}")
            summary["error"] = str(e)
        return summary

    def _ask_groq_for_log_analysis(self, log_summary: dict, perf: dict) -> str:
        """
        Send the log summary + perf snapshot to Groq and ask for a code-level
        diagnosis (specifically: why so many `momentum_reversal` invalidations,
        is the bot trading against trend, and what to change in which file).

        Returns the analysis text, or a fallback message on failure.
        """
        if not self.client:
            return "[Groq client unavailable — install `groq` and set API key to enable log analysis.]"

        # Compact slice for the prompt
        joined_sig  = "\n".join(log_summary.get("signal_lines",       [])[-10:])
        joined_ens  = "\n".join(log_summary.get("ensemble_lines",     [])[-10:])
        joined_sl   = "\n".join(log_summary.get("sl_exit_lines",      [])[-10:])
        joined_inv  = "\n".join(log_summary.get("invalidation_lines", [])[-10:])
        joined_rej  = "\n".join(log_summary.get("rejected_lines",     [])[-10:])

        prompt = f"""You are the master monitor for a crypto futures bot. Analyse these live logs.

PERFORMANCE
- Trades: {perf.get('total_trades','?')}, WR: {perf.get('win_rate','?')}%, PnL: ${perf.get('total_pnl','?')}
- PF: {perf.get('profit_factor','?')}, AvgWin ${perf.get('avg_win','?')}, AvgLoss ${perf.get('avg_loss','?')}

LOG COUNTS (last {log_summary.get('lines_scanned',0)} lines)
- momentum_reversal hits: {log_summary.get('momentum_reversal',0)}
- short signals: {log_summary.get('short_signals',0)}
- long signals:  {log_summary.get('long_signals',0)}
- SL exits:      {len(log_summary.get('sl_exit_lines',[]))}
- invalidations: {len(log_summary.get('invalidation_lines',[]))}
- rejected:      {len(log_summary.get('rejected_lines',[]))}

SIGNAL SAMPLES
{joined_sig or '(none)'}

ENSEMBLE SAMPLES
{joined_ens or '(none)'}

SL EXIT SAMPLES
{joined_sl or '(none)'}

INVALIDATION SAMPLES
{joined_inv or '(none)'}

REJECTED SAMPLES
{joined_rej or '(none)'}

KNOWN PROBLEM
The bot is taking SHORT (SELL) trades in a rising market and immediately
hitting stop-loss with "INVALIDATION: momentum_reversal (3 rising closes vs short)".

ANSWER ALL THREE, concretely:
1) Why are so many trades hitting `momentum_reversal`?
2) Is the bot trading against the trend? Cite log evidence and explain how
   we can detect this from the logs going forward.
3) Suggest SPECIFIC code or config changes — name the file (e.g. bot/engine/ensemble.py,
   bot/models/ai_strategy.py, config_futures.yaml) and the line/section, with the
   exact edit (e.g. "raise strategy.min_confidence from X to Y", "add a higher-TF
   trend filter that vetoes SHORTs when EMA50>EMA200 on 1h").
Keep it under 500 words, plain text, prioritised list."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "Senior quant + Python engineer. Diagnose live trading logs. Be specific, cite files and lines, no fluff."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=700, temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            log.error(f"_ask_groq_for_log_analysis error: {e}")
            return f"[Groq log-analysis failed: {e}]"

    def _persist_master_analysis(self, analysis: str, log_summary: dict, perf: dict) -> None:
        """Append the analysis (with header) to master_analysis.log. Never raises."""
        try:
            MASTER_ANALYSIS_LOG.parent.mkdir(parents=True, exist_ok=True)
            ts = datetime.now(timezone.utc).isoformat()
            header = (
                f"\n===== MASTER ANALYSIS @ {ts} =====\n"
                f"WR={perf.get('win_rate','?')}% PnL=${perf.get('total_pnl','?')} "
                f"Trades={perf.get('total_trades','?')} "
                f"momentum_reversal={log_summary.get('momentum_reversal',0)} "
                f"shorts={log_summary.get('short_signals',0)} longs={log_summary.get('long_signals',0)}\n"
            )
            with open(MASTER_ANALYSIS_LOG, "a", encoding="utf-8") as f:
                f.write(header)
                f.write(analysis.rstrip() + "\n")
        except Exception as e:
            log.warning(f"Failed to write master_analysis.log: {e}")

    # ============================================================

    def run_learning_cycle(self, notifier=None):
        log.info("SELF-LEARNING CYCLE")

        # v3: prefer call-site notifier, fall back to attribute set in __init__
        notifier = notifier or self.notifier
        if notifier is not None:
            self.notifier = notifier

        perf = self.analyze_performance()
        if "error" in perf:
            log.warning(f"Learning skipped: {perf['error']}")
            self.insights["last_review"] = datetime.now(timezone.utc).isoformat()
            self._save_insights()
            return perf

        log.info(f"Performance: WR={perf['win_rate']}% PnL=${perf['total_pnl']} Sig={perf['significant']}")

        changes = []
        if perf["significant"]:
            improvements = self.ask_ai_for_improvements(perf)
            if improvements:
                log.info(f"AI Insight: {improvements.get('key_insight','')}")
                changes = self.apply_improvements(improvements, perf=perf)

        # Rule-based proposals — emitted regardless of AI availability
        rule_changes = self._auto_tune_rules(perf, changes)
        changes.extend(rule_changes)

        # === v3 master-monitor hook: deep log analysis on bad performance ===
        master_analysis = None
        try:
            log_summary = self._analyse_recent_logs(lines=300)
            trigger_wr  = perf.get("win_rate", 100) < 45
            trigger_mr  = log_summary.get("momentum_reversal", 0) > 3
            if trigger_wr or trigger_mr:
                log.warning(
                    f"Master-monitor triggered (wr<45={trigger_wr}, "
                    f"momentum_reversal={log_summary.get('momentum_reversal',0)})"
                )
                master_analysis = self._ask_groq_for_log_analysis(log_summary, perf)
                self._persist_master_analysis(master_analysis, log_summary, perf)
                if self.notifier is not None:
                    try:
                        head = (
                            "MASTER MONITOR\n"
                            f"WR {perf.get('win_rate','?')}%  "
                            f"mom_rev {log_summary.get('momentum_reversal',0)}  "
                            f"shorts {log_summary.get('short_signals',0)}  "
                            f"longs {log_summary.get('long_signals',0)}\n\n"
                        )
                        # Telegram message limit ~4096; trim safely.
                        body = (master_analysis or "")[:3500]
                        self.notifier.send_alert(head + body)
                    except Exception as e:
                        log.warning(f"Notifier send_alert failed: {e}")
        except Exception as e:
            log.warning(f"Master-monitor block failed: {e}")

        self.insights["total_reviews"] += 1
        self.insights["last_review"]    = datetime.now(timezone.utc).isoformat()
        self.insights["performance_history"].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "win_rate":  perf["win_rate"],
            "total_pnl": perf["total_pnl"],
            "trades":    perf["total_trades"],
        })
        self.insights["performance_history"] = self.insights["performance_history"][-30:]
        if changes:
            self.insights.setdefault("adjustments", []).append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "changes": changes,
            })
            self.insights["adjustments"] = self.insights["adjustments"][-20:]
        self._save_insights()

        result = {"performance": perf, "changes": changes}
        if master_analysis is not None:
            result["master_analysis"] = master_analysis
        return result

    def _auto_tune_rules(self, perf, existing_changes):
        """
        Propose-only deterministic rules. Never mutates configs — emits
        proposals to pending_recommendations.json for user confirmation.
        """
        proposals = []
        history = self.insights.get("performance_history", [])
        if len(history) < 3:
            return []

        recent_pnls = [h.get("total_pnl", 0) for h in history[-3:]]
        consecutive_losses = all(p < 0 for p in recent_pnls)
        if not consecutive_losses:
            return []

        already_proposed_conf = any("min_confidence" in c for c in existing_changes)
        if not already_proposed_conf:
            for cfg_file in ["config_spot.yaml", "config_futures.yaml"]:
                cfg_path = BOT_ROOT / cfg_file
                if not cfg_path.exists():
                    continue
                with open(cfg_path) as f:
                    config = yaml.safe_load(f)
                old = config["strategy"]["min_confidence"]
                new = round(min(old + 0.03, 0.65), 3)
                if new != old:
                    proposals.append({
                        "file": cfg_file,
                        "section": "strategy.min_confidence",
                        "current": old,
                        "proposed": new,
                        "reason": (
                            f"Rule: 3 consecutive negative reviews "
                            f"(recent PnLs: {[round(p,2) for p in recent_pnls]}). "
                            f"Raise min_confidence to filter weaker signals."
                        ),
                        "source": "rule_auto_tune",
                    })

        if perf["win_rate"] < 35 and perf["total_trades"] >= 10:
            already_proposed_stop = any("stop_loss" in c for c in existing_changes)
            if not already_proposed_stop:
                for cfg_file in ["config_spot.yaml", "config_futures.yaml"]:
                    cfg_path = BOT_ROOT / cfg_file
                    if not cfg_path.exists():
                        continue
                    with open(cfg_path) as f:
                        config = yaml.safe_load(f)
                    old_sl = config["risk"]["stop_loss_atr_multiplier"]
                    new_sl = round(min(old_sl + 0.5, 4.0), 1)
                    if new_sl != old_sl:
                        proposals.append({
                            "file": cfg_file,
                            "section": "risk.stop_loss_atr_multiplier",
                            "current": old_sl,
                            "proposed": new_sl,
                            "reason": (
                                f"Rule: win_rate {perf['win_rate']}% < 35% across "
                                f"{perf['total_trades']} trades. Widen stop to reduce "
                                f"stop-outs from intra-bar noise."
                            ),
                            "source": "rule_auto_tune",
                        })

        return self._record_proposals(proposals, perf=perf)
