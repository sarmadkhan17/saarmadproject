"""
Natural-language ops layer for the Telegram interface.

Maps a free-form message ("how are we doing?", "why didn't you trade ETH?",
"flatten everything") onto the SAME command surface a human already has via slash
commands / the dashboard. One DeepSeek call per typed sentence, only when the user
messages the bot — nothing runs in the trading loop.

Guardrails (see plan):
  · The model may only pick a tool from a fixed allow-list; free-form output is never
    executed (no subprocess/eval of model text here).
  · Read intents run immediately; every state-changing action requires an explicit YES.
  · No exchange keys, no order placement — actions reuse existing notifier handlers or
    write the same control files the dashboard endpoints write.
  · Reactive only; the authorized-chat guard lives in telegram.py and is inherited.
"""

import json
import logging
import time
from datetime import datetime, timedelta

from core.tz import LOCAL_TZ
from core.config import DATA_DIR
from agents import llm_reasoning

log = logging.getLogger("NLOps")
DATA = DATA_DIR

# Pending confirmations expire after this many seconds.
_CONFIRM_TTL = 120

_AFFIRM = {"yes", "y", "yeah", "yep", "yup", "confirm", "ok", "okay",
           "do it", "go", "proceed", "sure", "affirmative"}
_NEGATE = {"no", "n", "nope", "cancel", "abort", "stop", "nevermind", "never mind"}


# Read intents — run immediately, no confirmation. Each maps to an existing
# notifier handler unless implemented locally (watchlist, explain_signal).
READ_TOOLS = [
    {"name": "status",       "description": "bot running state, last signal, profile, regime, watched coins"},
    {"name": "pnl",          "description": "account balance, today + all-time PnL, win rate, live position PnL"},
    {"name": "positions",    "description": "list current open trades / positions / exposure"},
    {"name": "health",       "description": "full system health, signal age, model + token status"},
    {"name": "agents",       "description": "per-agent accuracy and DeepSeek token usage"},
    {"name": "profile",      "description": "show current trading profile and its settings"},
    {"name": "mode",         "description": "show whether running in SPOT or FUTURES mode"},
    {"name": "watchlist",    "description": "list the coins the scanner is currently watching"},
    {"name": "explain_signal", "description": "explain the most recent decision for a coin (why it traded or skipped it)",
                               "args": ["symbol"]},
]

# Action intents — require an explicit YES confirmation before executing.
ACTION_TOOLS = [
    {"name": "pause_trading",  "description": "pause opening new trades (existing trades keep running)"},
    {"name": "resume_trading", "description": "resume opening new trades"},
    {"name": "flatten_all",    "description": "close ALL open positions now"},
    {"name": "set_profile",    "description": "switch the trading profile",
                               "args": ["name (STRICT|BALANCED|AGGRESSIVE)"]},
    {"name": "switch_mode",    "description": "switch between SPOT and FUTURES",
                               "args": ["mode (spot|futures)"]},
    {"name": "restart",        "description": "restart the bot process"},
    {"name": "stop",           "description": "stop the bot process"},
    {"name": "disable_circuit_breaker_24h", "description": "disable the circuit breaker for 24 hours"},
]

_READ_NAMES = {t["name"] for t in READ_TOOLS}
_ACTION_NAMES = {t["name"] for t in ACTION_TOOLS}
# "unknown" lets the model bail out gracefully instead of forcing a wrong tool.
_TOOL_SCHEMA = READ_TOOLS + ACTION_TOOLS + [
    {"name": "unknown", "description": "the message does not match any tool"}
]


class NLOpsHandler:
    def __init__(self, notifier):
        self.n = notifier
        # {chat_id: {"tool": str, "args": dict, "expires": float, "label": str}}
        self._pending: dict = {}

    # ── Confirmation flow ─────────────────────────────────────────────────────

    def maybe_confirm(self, text: str, chat_id: str) -> bool:
        """If a confirmation is pending for this chat, consume the reply.

        Returns True when the message was handled here (so the caller should not
        treat it as a fresh request). An affirmative reply executes the action; an
        explicit negative cancels; anything else clears the stale pending and lets
        the caller process the message as a new request.
        """
        pend = self._pending.get(chat_id)
        if not pend:
            return False
        if time.time() > pend["expires"]:
            self._pending.pop(chat_id, None)
            return False
        t = (text or "").strip().lower()
        if t in _AFFIRM:
            self._pending.pop(chat_id, None)
            self.n.send(f"⏳ Confirmed — {pend['label']}…")
            self._run_action(pend["tool"], pend["args"])
            return True
        if t in _NEGATE:
            self._pending.pop(chat_id, None)
            self.n.send("✅ Cancelled.")
            return True
        # Ambiguous → drop the stale pending and let it be reprocessed fresh.
        self._pending.pop(chat_id, None)
        return False

    # ── Entry point ───────────────────────────────────────────────────────────

    def handle(self, text: str, chat_id: str):
        intent = llm_reasoning.nl_intent(text, _TOOL_SCHEMA)
        if intent is None:
            self.n.send("🤔 I couldn't reach the interpreter. Try /help for commands.")
            return
        tool = intent["tool"]
        args = intent.get("args", {})

        if tool in _READ_NAMES:
            self._run_read(tool, args)
        elif tool in _ACTION_NAMES:
            label = self._action_label(tool, args)
            self._pending[chat_id] = {
                "tool": tool, "args": args,
                "expires": time.time() + _CONFIRM_TTL, "label": label,
            }
            self.n.send(
                f"⚠️ You asked to <b>{label}</b>.\n"
                f"Reply <b>YES</b> to confirm or <b>NO</b> to cancel (expires in {_CONFIRM_TTL}s)."
            )
        else:
            self._fallback()

    def _fallback(self):
        self.n.send(
            "🤖 I can answer questions like:\n"
            "  • \"how are we doing?\" / \"what's my PnL?\"\n"
            "  • \"what positions are open?\" / \"my exposure?\"\n"
            "  • \"why didn't you trade ETH?\"\n"
            "  • \"what's the current profile / mode?\"\n\n"
            "…or do (with confirmation): pause/resume trading, flatten all, "
            "switch profile/mode, restart/stop.\n"
            "Type /help for the classic command list."
        )

    # ── Read dispatch ─────────────────────────────────────────────────────────

    def _run_read(self, tool: str, args: dict):
        try:
            if tool == "status":
                self.n._cmd_status()
            elif tool in ("pnl", "performance"):
                self.n._cmd_pnl()
            elif tool in ("positions", "exposure"):
                self.n._cmd_trades()
            elif tool == "health":
                self.n._cmd_health()
            elif tool == "agents":
                self.n._cmd_agents()
            elif tool == "profile":
                self.n._cmd_profile()
            elif tool == "mode":
                self.n._cmd_current_mode()
            elif tool == "watchlist":
                self._watchlist()
            elif tool == "explain_signal":
                self._explain_signal(args.get("symbol", ""))
            else:
                self._fallback()
        except Exception as e:
            log.warning(f"NL read '{tool}' error: {e}")
            self.n.send(f"⚠️ Couldn't run that: {e}")

    def _watchlist(self):
        coins = []
        p = DATA / "scanner_cache.json"
        if p.exists():
            try:
                with open(p) as f:
                    coins = json.load(f).get("top_coins", [])
            except Exception:
                pass
        if not coins:
            self.n.send("📭 No watchlist cached yet.")
            return
        names = ", ".join(c.replace("/USDT", "") for c in coins)
        self.n.send(f"👀 <b>Watching {len(coins)} coins</b>\n{names}")

    def _explain_signal(self, symbol: str):
        """Read the most recent stored signal for `symbol` and explain it."""
        base = (symbol or "").upper().replace("/USDT", "").strip()
        if not base:
            self.n.send("Which coin? e.g. \"why didn't you trade ETH?\"")
            return
        _, signals, _ = self.n._load_combined()
        matches = [s for s in signals
                   if str(s.get("symbol", "")).upper().replace("/USDT", "") == base]
        if not matches:
            self.n.send(f"🔍 No recent signal recorded for {base}.")
            return
        matches.sort(key=lambda x: x.get("timestamp", ""))
        s = matches[-1]
        action = str(s.get("action", "?")).upper()
        conf = s.get("confidence", 0) or 0
        reason = s.get("reason") or s.get("strategy") or "(no reason recorded)"
        status = s.get("status", "?")
        ts = str(s.get("timestamp", ""))[:19]
        ind = s.get("indicators", {}) or {}
        outcome = {
            "hold": "it held (no trade)",
            "rejected": "the trade was rejected by a risk gate",
            "executed": "it entered a trade",
            "open": "it entered a trade",
        }.get(str(status).lower(), status)

        msg = (
            f"🔍 <b>{base} — last decision</b>\n"
            f"  🕐 {ts}\n"
            f"  Signal: <b>{action}</b> @ {float(conf):.0%} confidence\n"
            f"  Outcome: {outcome}\n"
            f"  Reason: <code>{str(reason)[:300]}</code>"
        )
        extras = []
        for k in ("buy_score", "sell_score", "agents_agree", "regime", "profile"):
            if k in ind:
                extras.append(f"{k}={ind[k]}")
        if extras:
            msg += "\n  " + " · ".join(extras)
        self.n.send(msg)

    # ── Action dispatch (only reached after explicit YES) ─────────────────────

    def _action_label(self, tool: str, args: dict) -> str:
        if tool == "set_profile":
            return f"switch profile to {str(args.get('name', '?')).upper()}"
        if tool == "switch_mode":
            return f"switch to {str(args.get('mode', '?')).upper()} mode"
        return {
            "pause_trading":  "pause new trades",
            "resume_trading": "resume new trades",
            "flatten_all":    "close ALL open positions",
            "restart":        "restart the bot",
            "stop":           "stop the bot",
            "disable_circuit_breaker_24h": "disable the circuit breaker for 24h",
        }.get(tool, tool)

    def _run_action(self, tool: str, args: dict):
        try:
            if tool == "pause_trading":
                self._write_paused(True)
                self.n.send("⏸ Trading paused — no new entries (open trades unaffected).")
            elif tool == "resume_trading":
                self._write_paused(False)
                self.n.send("▶️ Trading resumed.")
            elif tool == "flatten_all":
                self._flatten_all()
            elif tool == "set_profile":
                name = str(args.get("name", "")).upper()
                if name not in ("STRICT", "BALANCED", "AGGRESSIVE"):
                    self.n.send("❌ Unknown profile. Use STRICT, BALANCED, or AGGRESSIVE.")
                    return
                self.n._cmd_profile_set(name)
            elif tool == "switch_mode":
                mode = str(args.get("mode", "")).lower()
                if mode == "spot":
                    self.n._cmd_switch_spot()
                elif mode == "futures":
                    self.n._cmd_switch_futures()
                else:
                    self.n.send("❌ Mode must be spot or futures.")
            elif tool == "restart":
                self.n._cmd_restart()
            elif tool == "stop":
                self.n._cmd_stop()
            elif tool == "disable_circuit_breaker_24h":
                self._disable_breaker_24h()
            else:
                self._fallback()
        except Exception as e:
            log.warning(f"NL action '{tool}' error: {e}")
            self.n.send(f"⚠️ Action failed: {e}")

    # ── Control-file writers (mirror the dashboard endpoints) ─────────────────

    def _write_paused(self, paused: bool):
        p = DATA / "trading_paused.json"
        p.parent.mkdir(exist_ok=True)
        with open(p, "w") as f:
            json.dump({"paused": paused,
                       "timestamp": datetime.now(LOCAL_TZ).isoformat()}, f)

    def _flatten_all(self):
        count = 0
        sp = DATA / "futures_state.json"
        try:
            if sp.exists():
                with open(sp) as f:
                    st = json.load(f)
                count = sum(1 for t in st.get("trades", []) if t.get("status") == "open")
        except Exception:
            pass
        p = DATA / "close_all_positions.json"
        p.parent.mkdir(exist_ok=True)
        with open(p, "w") as f:
            json.dump({"close_all": True,
                       "timestamp": datetime.now(LOCAL_TZ).isoformat()}, f)
        self.n.send(f"🛑 Queued close of {count} open position(s). The bot will flatten on its next cycle.")

    def _disable_breaker_24h(self):
        p = DATA / "circuit_breaker.json"
        try:
            with open(p) as f:
                cb = json.load(f)
        except Exception:
            cb = {}
        cb["consec_losses"] = 0
        cb["disabled_until"] = (datetime.now(LOCAL_TZ) + timedelta(hours=24)).isoformat()
        with open(p, "w") as f:
            json.dump(cb, f)
        self.n.send("🟡 Circuit breaker disabled for 24h.")
