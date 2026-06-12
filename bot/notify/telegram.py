"""
Telegram Notifier v4
- Supports both personal and group chats (negative chat IDs for groups)
- Combined SPOT + FUTURES reporting
- Commands: /status /agents /pnl /trades /health /help
"""

import os
import requests
import logging
import json
import threading
import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone, date
from core.tz import LOCAL_TZ
from concurrent.futures import ThreadPoolExecutor, as_completed
import dataclasses
import yaml
from core.config import get_telegram_config, DATA_DIR, BOT_ROOT
from notify.nl_ops import NLOpsHandler

log      = logging.getLogger("Notifier")
DATA     = DATA_DIR

def _deepseek_today() -> dict:
    """Read deepseek_usage.json and return today's usage.

    The persisted file stores per-day stats under ``daily[<date>]`` and has no
    top-level ``used_today`` key, so callers must derive it here.
    Returns {"used_today": int, "cost_usd": float, "calls": int}.
    """
    p = DATA / "deepseek_usage.json"
    used = calls = 0
    cost = 0.0
    if p.exists():
        try:
            with open(p) as f:
                tb = json.load(f)
            day = tb.get("daily", {}).get(str(datetime.now(LOCAL_TZ).date()), {})
            used  = day.get("input_tokens", 0) + day.get("output_tokens", 0)
            cost  = day.get("cost_usd", 0.0)
            calls = day.get("calls", 0)
        except Exception:
            pass
    return {"used_today": used, "cost_usd": cost, "calls": calls}
CFG_SPOT = BOT_ROOT / "config_spot.yaml"
CFG_FUT  = BOT_ROOT / "config_futures.yaml"
ENV_PATH = BOT_ROOT / ".env"
SERVICE_PATH = Path("/etc/systemd/system/cryptobot-futures.service")


class TelegramNotifier:
    def __init__(self):
        cfg            = get_telegram_config()
        self.token     = cfg.get("token", "")
        self.chat_id   = cfg.get("chat_id", "")  # Can be group (-100...) or user (positive)
        self.last_sent = None
        self.interval  = 30  # minutes
        self.exchange  = None
        self.offset    = 0

        chat_type = "GROUP" if str(self.chat_id).startswith("-") else "PRIVATE"
        log.info(f"Telegram chat type: {chat_type} | ID: {self.chat_id}")

        # Natural-language ops layer — interprets free-form messages into the
        # existing command surface (read-only Q&A + confirmed actions).
        self.nl = NLOpsHandler(self)

        if self.token:
            t = threading.Thread(target=self._poll_commands, daemon=True)
            t.start()
            log.info("Telegram command listener started")

    def send(self, message):
        if not self.token or not self.chat_id:
            return False
        try:
            url  = f"https://api.telegram.org/bot{self.token}/sendMessage"
            resp = requests.post(url, json={
                "chat_id":    self.chat_id,
                "text":       message,
                "parse_mode": "HTML",
            }, timeout=10)
            return resp.status_code == 200
        except Exception as e:
            log.error(f"Telegram error: {e}")
            return False

    def send_alert(self, message):
        now = datetime.now(LOCAL_TZ).strftime("%H:%M UTC+3")
        self.send(f"<b>ALERT {now}</b>\n{message}")

    def should_send(self):
        if self.last_sent is None:
            return True
        return (datetime.now(LOCAL_TZ) - self.last_sent).total_seconds() >= self.interval * 60

    def send_report(self, exchange):
        if not self.should_send():
            return
        self.exchange = exchange
        try:
            msg = self._build_report(exchange)
            if self.send(msg):
                self.last_sent = datetime.now(LOCAL_TZ)
        except Exception as e:
            log.error(f"Report error: {e}")

    def _load_combined(self):
        """Combine spot + futures state for unified reporting."""
        trades  = []
        signals = []
        stats   = {"wins": 0, "losses": 0, "total_pnl": 0.0, "total_trades": 0}
        for fname in ["state.json", "futures_state.json"]:
            p = DATA / fname
            if p.exists():
                with open(p) as f:
                    d = json.load(f)
                trades.extend(d.get("trades", []))
                signals.extend(d.get("signals", []))
                s = d.get("stats", {})
                stats["wins"]         += s.get("wins", 0)
                stats["losses"]       += s.get("losses", 0)
                stats["total_pnl"]    += s.get("total_pnl", 0.0)
                stats["total_trades"] += s.get("total_trades", 0)
        return trades, signals, stats

    @staticmethod
    def _closed_stats(trades):
        """Win/loss/PnL derived from the actual closed-trade list (ground truth).

        The cumulative `state.stats` counters over-count (wins are incremented
        in several bot.py paths), so deriving from the stored trades keeps the
        report internally consistent: wins + losses always equals closed.
        """
        closed = [t for t in trades if t.get("status") == "closed"]
        wins   = sum(1 for t in closed if float(t.get("pnl", 0) or 0) > 0)
        losses = len(closed) - wins          # losses + breakeven
        pnl    = sum(float(t.get("pnl", 0) or 0) for t in closed)
        return closed, wins, losses, pnl

    def _build_report(self, exchange):
        trades, signals, stats = self._load_combined()
        open_t = [t for t in trades if t["status"] == "open"]
        closed, c_wins, c_losses, c_pnl = self._closed_stats(trades)

        sig_age = "N/A"
        if signals:
            signals.sort(key=lambda x: x.get("timestamp", ""))
            last    = datetime.fromisoformat(signals[-1]["timestamp"])
            if last.tzinfo is None:
                last = last.replace(tzinfo=LOCAL_TZ)
            sig_age = f"{int((datetime.now(LOCAL_TZ)-last).total_seconds())}s ago"

        # Batch fetch all tickers in one call
        total_live = 0.0
        lines      = []
        if open_t:
            try:
                symbols = list({t["symbol"] for t in open_t})
                tickers = exchange.fetch_tickers(symbols)
            except Exception:
                tickers = {}

            for t in open_t:
                try:
                    ticker = tickers.get(t["symbol"], {})
                    price  = ticker.get("last", t["price"])
                    entry  = t["price"]
                    side   = t.get("side", "long")
                    # Short positions profit when price falls — sign must flip.
                    pnl    = (entry - price) * t["amount"] if side == "short" \
                             else (price - entry) * t["amount"]
                    pct    = (price - entry) / entry * 100 * (-1 if side == "short" else 1)
                    total_live += pnl
                    picon    = "🟢" if pnl >= 0 else "🔴"
                    side_tag = "🔼 LONG" if side == "long" else "🔽 SHORT"
                    lines.append(
                        f"  {picon} <b>{t['symbol']}</b> · {side_tag}  "
                        f"{pct:+.2f}% (${pnl:+.2f})"
                    )
                except Exception:
                    pass

        total = c_wins + c_losses
        wr    = f"{c_wins/total*100:.1f}%" if total else "N/A"
        now   = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d %H:%M UTC+3")

        # Token usage
        token_info = ""
        if (DATA / "deepseek_usage.json").exists():
            ds = _deepseek_today()
            token_info = (f"  DeepSeek: {ds['used_today']:,} tok · "
                          f"${ds['cost_usd']:.2f} ({ds['calls']:,} calls) today\n")

        mode = self._get_mode_from_env()

        # Profile / regime / account balance from the current mode's state.
        profile_name = "?"
        live_regime = "?"
        balance = 0.0
        p_state = DATA / ("futures_state.json" if mode == "futures" else "state.json")
        if p_state.exists():
            with open(p_state) as f:
                st = json.load(f)
            live = st.get("live_strategy", {})
            profile_name = live.get("profile", "?")
            live_regime = live.get("market_regime", "?")
            balance = float(st.get("stats", {}).get("balance", 0.0) or 0.0)

        mode_icon = "🟣" if mode == "futures" else "🔵"
        pnl_icon  = "🟢" if c_pnl >= 0 else "🔴"
        live_icon = "🟢" if total_live >= 0 else "🔴"

        msg = (
            f"<b>{mode_icon} CryptoBot v4 Report</b>\n"
            f"🕐 <b>{now}</b>\n\n"
            f"📊 <b>STATUS</b>\n"
            f"  ⚙️ Mode: {mode.upper()}   🎯 Profile: {profile_name}\n"
            f"  🌐 Regime: {live_regime}   📡 Signal: {sig_age}\n"
            f"  📂 Open: {len(open_t)}   📁 Closed: {len(closed)}\n\n"
            f"💼 <b>ACCOUNT</b>\n"
            f"  💰 Balance: <b>${balance:,.2f}</b>\n\n"
            f"📈 <b>PERFORMANCE</b>\n"
            f"  🎯 Win Rate: {wr} (✅ {c_wins}W / ❌ {c_losses}L)\n"
            f"  {pnl_icon} Closed PnL: <b>${c_pnl:+.2f}</b>\n"
            f"  {live_icon} Live PnL: <b>${total_live:+.2f}</b>\n"
        )
        if lines:
            msg += f"\n📌 <b>POSITIONS</b>\n" + "\n".join(lines)
        if token_info:
            msg += f"\n\n🤖 {token_info.strip()}"
        return msg

    def _poll_commands(self):
        """Long-poll for commands. Works in both private and group chats."""
        while True:
            try:
                url  = f"https://api.telegram.org/bot{self.token}/getUpdates"
                resp = requests.get(
                    url,
                    params={"offset": self.offset, "timeout": 15},
                    timeout=20,
                )
                if resp.status_code != 200:
                    time.sleep(2)
                    continue

                updates = resp.json().get("result", [])
                max_offset = self.offset
                for update in updates:
                    update_id = update["update_id"]
                    if update_id >= max_offset:
                        max_offset = update_id + 1
                    msg  = update.get("message", {})
                    # Keep the raw, original-case text for the NL interpreter;
                    # slash-command matching uses the lowercased form.
                    raw_text = msg.get("text", "").strip()
                    text = raw_text.lower()
                    chat = str(msg.get("chat", {}).get("id", ""))

                    # Only respond to authorized chat (group or private)
                    if chat != self.chat_id:
                        continue

                    # Strip bot mention from group commands (e.g., /status@MyBot)
                    if "@" in text:
                        text = text.split("@")[0]

                    if text.startswith("/htf"):
                        self.send("ℹ️ <b>HTF panel removed.</b> Use dashboard for settings.")
                        continue

                    handler = {
                        "/help":            self._cmd_help,
                        "/start":           self._cmd_status,
                        "/status":          self._cmd_status,
                        "/agents":          self._cmd_agents,
                        "/pnl":             self._cmd_pnl,
                        "/trades":          self._cmd_trades,
                        "/health":          self._cmd_health,
                        "/shadows":         self._cmd_shadows,
                        "/gates":           self._cmd_gates,
                        "/switch_spot":     self._cmd_switch_spot,
                        "/switch_futures":  self._cmd_switch_futures,
                        "/restart":         self._cmd_restart,
                        "/stop":            self._cmd_stop,
                        "/mode":            self._cmd_current_mode,
                        "/profile":         self._cmd_profile,
                        "/profile_strict":  lambda: self._cmd_profile_set("STRICT"),
                        "/profile_balanced": lambda: self._cmd_profile_set("BALANCED"),
                        "/profile_aggressive": lambda: self._cmd_profile_set("AGGRESSIVE"),
                    }.get(text)

                    if handler is not None:
                        handler()
                    elif text.startswith("/"):
                        pass  # unknown slash command — stay silent (unchanged)
                    elif raw_text:
                        # Free-form message → confirmation reply or NL interpreter.
                        if not self.nl.maybe_confirm(raw_text, chat):
                            self.nl.handle(raw_text, chat)

                self.offset = max_offset
            except Exception as e:
                log.error(f"Command poll error: {e}")
                time.sleep(2)

    def _cmd_help(self):
        self.send(
            "<b>🤖 CryptoBot v4 Commands</b>\n\n"
            "<b>📊 Monitoring:</b>\n"
            "/status  — Bot status check\n"
            "/mode    — Show current mode\n"
            "/agents  — Agent performance + tokens\n"
            "/pnl     — Current PnL\n"
            "/trades  — Open positions\n"
            "/health  — Full system health\n"
            "/shadows — Hypothetical outcomes of rejected signals\n"
            "/gates   — Gate precision vs taken trades + redundancy\n\n"
            "<b>⚙️ Control:</b>\n"
            "/switch_spot     — Switch to SPOT mode\n"
            "/switch_futures  — Switch to FUTURES mode\n"
            "/restart         — Restart bot\n"
            "/stop            — Stop bot\n"
        )

    def _cmd_shadows(self):
        """Per-gate hypothetical outcomes of rejected signals (ShadowTracker)."""
        try:
            from agents.shadow_tracker import load_stats
            mode = self._get_mode_from_env()
            stats = load_stats(DATA / "trade_memory.db", mode, days=30)
            if not stats:
                self.send("👻 <b>Shadow trades</b>\nNo shadow data yet — "
                          "stats appear once rejected signals resolve.")
                return
            lines = [f"👻 <b>Shadow trades</b> ({mode.upper()}, last 30d)\n",
                     "<i>What rejected signals would have done:</i>\n"]
            for gate, s in sorted(stats.items()):
                verdict = "⚠️ may be too strict" if (s["tp"] + s["sl"]) >= 5 and s["win_rate"] >= 0.55 \
                    else ("✅ protective" if (s["tp"] + s["sl"]) >= 5 and s["win_rate"] <= 0.40 else "⏳ gathering data")
                lines.append(
                    f"<b>{gate}</b> — {s['n']} tracked, {s['open']} open\n"
                    f"  TP {s['tp']} / SL {s['sl']} / expired {s['expired']} · "
                    f"hyp. WR {s['win_rate']:.0%}\n"
                    f"  net {s['net_r']:+.1f}R (gate avoided {-s['net_r']:+.1f}R) {verdict}\n"
                )
            self.send("\n".join(lines))
        except Exception as e:
            self.send(f"⚠️ Shadow stats unavailable: {e}")

    def _cmd_gates(self):
        """Gate complementarity report: per-gate blocked-trade precision vs
        the taken-trade baseline, plus the cheap-gate redundancy matrix."""
        try:
            from agents.shadow_tracker import load_stats, load_taken_stats
            mode = self._get_mode_from_env()
            db = DATA / "trade_memory.db"
            stats = load_stats(db, mode, days=30)
            taken = load_taken_stats(db, mode, days=30)
            if not stats and not taken:
                self.send("🚪 <b>Gate report</b>\nNo data yet.")
                return
            lines = [f"🚪 <b>Gate report</b> ({mode.upper()}, last 30d)\n"]
            if taken:
                lines.append(
                    f"<b>Baseline — taken trades</b>: {taken['n']} closed, "
                    f"WR {taken['win_rate']:.0%}, avg {taken['mean_r']:+.2f}R\n")
            lines.append("<i>A gate complements when its blocks would have "
                         "done WORSE than the baseline:</i>\n")
            base_wr = taken.get("win_rate", 0.0) if taken else 0.0
            for gate, s in sorted(stats.items()):
                decided = s["tp"] + s["sl"]
                if decided >= 5:
                    flag = "🔴 blocking winners" if s["win_rate"] > base_wr + 0.10 \
                        else ("🟢 protective" if s["win_rate"] < base_wr - 0.10 else "🟡 ~neutral")
                else:
                    flag = "⏳ low sample"
                lines.append(
                    f"<b>{gate}</b> — blocked WR {s['win_rate']:.0%} "
                    f"({decided} decided) · net {s['net_r']:+.1f}R {flag}")
                red = s.get("redundant_with") or {}
                if red:
                    overlap = ", ".join(
                        f"{o} {c}/{s['n']}" for o, c in sorted(red.items()))
                    lines.append(f"  ↳ overlap: {overlap}")
            lines.append("\n<i>High overlap = gates duplicate each other; "
                         "decisions stay human-gated.</i>")
            self.send("\n".join(lines))
        except Exception as e:
            self.send(f"⚠️ Gate report unavailable: {e}")

    def _get_mode_from_env(self):
        """Read BOT_MODE from .env file."""
        if ENV_PATH.exists():
            with open(ENV_PATH) as f:
                for line in f:
                    if line.startswith("BOT_MODE="):
                        return line.split("=", 1)[1].strip().strip("\"'")
        return os.environ.get("BOT_MODE", "unknown")

    def _set_mode_and_restart(self, mode):
        """Update .env, systemd service, and restart."""
        if not ENV_PATH.exists():
            return False
        with open(ENV_PATH) as f:
            lines = f.readlines()
        new_lines = []
        for line in lines:
            if line.startswith("BOT_MODE="):
                new_lines.append(f'BOT_MODE="{mode}"\n')
            else:
                new_lines.append(line)
        with open(ENV_PATH, "w") as f:
            f.writelines(new_lines)

        if SERVICE_PATH.exists():
            with open(SERVICE_PATH) as f:
                svc_lines = f.readlines()
            new_svc = []
            for line in svc_lines:
                if "BOT_MODE=" in line:
                    new_svc.append(f'Environment="BOT_MODE={mode}"\n')
                else:
                    new_svc.append(line)
            with open(SERVICE_PATH, "w") as f:
                f.writelines(new_svc)

        subprocess.run(["systemctl", "daemon-reload"], timeout=5)
        subprocess.run(["systemctl", "restart", "cryptobot-futures"], timeout=10)
        return True

    def _check_open_trades(self):
        """Check if current mode has open trades."""
        mode = self._get_mode_from_env()
        state_file = DATA / ("state.json" if mode == "spot" else f"{mode}_state.json")
        if not state_file.exists():
            return False
        try:
            with open(state_file) as f:
                data = json.load(f)
            opens = [t for t in data.get("trades", []) if t.get("status") == "open"]
            return len(opens) > 0
        except Exception:
            return False

    def _cmd_status(self):
        try:
            _, signals, _ = self._load_combined()
            if signals:
                signals.sort(key=lambda x: x.get("timestamp", ""))
                last      = datetime.fromisoformat(signals[-1]["timestamp"])
                if last.tzinfo is None:
                    last = last.replace(tzinfo=LOCAL_TZ)
                age_sec   = int((datetime.now(LOCAL_TZ) - last).total_seconds())
                is_active = age_sec < 120
                age_str   = f"{age_sec}s ago"
                last_action = signals[-1].get("action", "?")
                last_conf   = signals[-1].get("confidence", 0)
                last_sym     = signals[-1].get("symbol", "?")
                signal_desc  = f" ({last_action} {last_sym} @ {last_conf:.0%})"
            else:
                is_active = False
                age_str   = "never"
                signal_desc = ""

            icon   = "✅" if is_active else "⚠️"
            status = "ACTIVE" if is_active else "IDLE"

            # Profile + regime from state
            profile_name = "?"
            regime  = "?"
            eff_min = "?"
            p = DATA / "futures_state.json"
            if p.exists():
                with open(p) as f:
                    st = json.load(f)
                live = st.get("live_strategy", {})
                profile_name = live.get("profile", "?")
                regime  = live.get("market_regime", "?")
                eff_min = live.get("eff_min_conf", "?")

            token_info = ""
            if (DATA / "deepseek_usage.json").exists():
                ds = _deepseek_today()
                token_info = f"\n  DeepSeek: {ds['used_today']:,} tok · ${ds['cost_usd']:.2f} today"

            coins = []
            p2 = DATA / "scanner_cache.json"
            if p2.exists():
                with open(p2) as f:
                    coins = json.load(f).get("top_coins", [])

            self.send(
                f"{icon} <b>Bot: {status} | Profile: {profile_name}</b>\n\n"
                f"  Last signal: {age_str}{signal_desc}\n"
                f"  Regime: {regime} | eff_min: {eff_min}\n"
                f"  Watching: {len(coins)} coins{token_info}\n"
                f"  Top: {', '.join([c.replace('/USDT','') for c in coins[:5]])}"
            )
        except Exception as e:
            self.send(f"Status error: {e}")

    def _cmd_agents(self):
        try:
            p   = DATA / "agent_performance.json"
            msg = "<b>🎯 Agent Performance</b>\n\n"
            if p.exists():
                with open(p) as f:
                    data = json.load(f)
                for agent, stats in data.items():
                    total = stats.get("total", 0)
                    if total == 0:
                        continue
                    acc  = round(stats["correct"] / total * 100, 1)
                    icon = "✅" if acc >= 55 else "⚠️" if acc >= 45 else "❌"
                    msg += f"{icon} {agent:12} {acc}% ({stats['correct']}/{total})\n"
            else:
                msg += "No data yet.\n"

            if (DATA / "deepseek_usage.json").exists():
                ds = _deepseek_today()
                msg  += f"\n<b>DeepSeek API</b>\n"
                msg  += f"  {ds['used_today']:,} tok · ${ds['cost_usd']:.2f} · {ds['calls']:,} calls today"
            self.send(msg)
        except Exception as e:
            self.send(f"Agents error: {e}")

    def _cmd_pnl(self):
        try:
            trades, _, stats = self._load_combined()
            open_t = [t for t in trades if t["status"] == "open"]
            closed, c_wins, c_losses, c_pnl = self._closed_stats(trades)
            total  = c_wins + c_losses
            wr     = f"{c_wins/total*100:.1f}%" if total else "N/A"

            # Account balance from current mode's state
            mode = self._get_mode_from_env()
            balance = 0.0
            p_state = DATA / ("futures_state.json" if mode == "futures" else "state.json")
            if p_state.exists():
                with open(p_state) as f:
                    balance = float(json.load(f).get("stats", {}).get("balance", 0.0) or 0.0)

            # Today stats
            today_pnl, today_wins, today_loss, today_total = self._calc_today_pnl(trades)
            today_icon = "🟢" if today_pnl >= 0 else "🔴"
            all_icon   = "🟢" if c_pnl >= 0 else "🔴"

            msg = (
                f"💰 <b>PnL Report</b>\n\n"
                f"💼 <b>ACCOUNT</b>\n"
                f"  💵 Balance: <b>${balance:,.2f}</b>\n\n"
                f"📅 <b>TODAY</b>\n"
                f"  {today_icon} PnL: ${today_pnl:+.2f} USDT\n"
                f"  🔢 Trades: {today_total} (✅ {today_wins}W / ❌ {today_loss}L)\n\n"
                f"📊 <b>ALL TIME</b>\n"
                f"  {all_icon} Total PnL: ${c_pnl:+.2f} USDT\n"
                f"  🎯 Win Rate: {wr} (✅ {c_wins}W / ❌ {c_losses}L)\n"
                f"  📁 Closed: {len(closed)} trades\n"
                f"  📂 Open: {len(open_t)} trades\n"
            )
            if self.exchange and open_t:
                live  = 0.0
                plines = []
                for t in open_t:
                    try:
                        price = self.exchange.fetch_ticker(t["symbol"])["last"]
                        entry = float(t["price"])
                        amt   = float(t["amount"])
                        side  = t.get("side", "long")
                        # Short positions profit when price falls — flip the sign.
                        pnl   = (entry - price) * amt if side == "short" \
                                else (price - entry) * amt
                        pct   = (price - entry) / entry * 100 * (-1 if side == "short" else 1)
                        live += pnl
                        picon    = "🟢" if pnl >= 0 else "🔴"
                        side_tag = "🔼 LONG" if side == "long" else "🔽 SHORT"
                        plines.append(
                            f"  {picon} <b>{t['symbol']}</b> · {side_tag}  "
                            f"{pct:+.2f}% (${pnl:+.2f})"
                        )
                    except Exception:
                        pass
                if plines:
                    live_icon = "🟢" if live >= 0 else "🔴"
                    msg += "\n📌 <b>POSITIONS</b>\n" + "\n".join(plines)
                    msg += f"\n\n{live_icon} <b>Live PnL: ${live:+.2f} USDT</b>"
            self.send(msg)
        except Exception as e:
            self.send(f"PnL error: {e}")

    def _cmd_trades(self):
        try:
            trades, _, _ = self._load_combined()
            open_t = [t for t in trades if t["status"] == "open"]
            if not open_t:
                self.send("📭 No open trades.")
                return
            msg = f"<b>📊 Open Positions ({len(open_t)})</b>\n\n"
            for t in open_t:
                mode = t.get("mode", "spot").upper()
                lev  = f" {t.get('leverage',1)}x" if t.get("leverage", 1) > 1 else ""
                msg += f"  [{mode}{lev}] {t['symbol']:10} {t.get('side','buy').upper()}\n"
                msg += f"  Entry: ${t['price']:.4f} | Amount: {t['amount']:.5f}\n\n"
            self.send(msg)
        except Exception as e:
            self.send(f"Trades error: {e}")

    def _cmd_health(self):
        try:
            models = {
                "RF":       "rf_model.pkl",
                "LightGBM": "lgbm_model.pkl",
            }
            msg = "<b>🩺 System Health</b>\n\n<b>Models:</b>\n"
            for name, fname in models.items():
                ok   = (DATA / fname).exists()
                msg += f"  {'✅' if ok else '❌'} {name}\n"

            # Profile + agent status
            profile_name = "?"
            p_state = DATA / "futures_state.json"
            if p_state.exists():
                with open(p_state) as f:
                    st = json.load(f)
                live = st.get("live_strategy", {})
                profile_name = live.get("profile", "?")
            msg += f"\n<b>Profile:</b> {profile_name}\n"
            msg += f"<b>Agents:</b> SMC ✅ | Technical ✅ | Macro/Flow ⏳\n"

            _, signals, _ = self._load_combined()
            if signals:
                signals.sort(key=lambda x: x.get("timestamp", ""))
                sig_ts = datetime.fromisoformat(signals[-1]["timestamp"])
                if sig_ts.tzinfo is None:
                    sig_ts = sig_ts.replace(tzinfo=LOCAL_TZ)
                age = int((datetime.now(LOCAL_TZ) - sig_ts).total_seconds())
                msg += f"\n{'✅' if age<120 else '⚠️'} Signal age: {age}s\n"

            if (DATA / "deepseek_usage.json").exists():
                ds = _deepseek_today()
                msg += f"✅ DeepSeek: {ds['used_today']:,} tok · ${ds['cost_usd']:.2f} today\n"

            p2 = DATA / "learning_insights.json"
            if p2.exists():
                with open(p2) as f:
                    ins = json.load(f)
                msg += f"✅ Self-learning: {ins.get('total_reviews',0)} reviews\n"

            self.send(msg)
        except Exception as e:
            self.send(f"Health error: {e}")

    def _is_docker(self):
        return Path("/.dockerenv").exists() or os.environ.get("DOCKER_CONTAINER") == "1"

    def _cmd_profile(self):
        """Show current trading profile with key settings."""
        try:
            mode = self._get_mode_from_env()
            cfg_path = CFG_FUT if mode != "spot" else CFG_SPOT
            profile_name = "AGGRESSIVE"
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f) or {}
                profile_name = cfg.get("strategy", {}).get("trading_profile", "AGGRESSIVE")

            state_file = "futures_state.json" if mode != "spot" else "state.json"
            p = DATA / state_file
            eff_min = "?"
            regime  = "UNKNOWN"
            if p.exists():
                with open(p) as f:
                    st = json.load(f)
                live = st.get("live_strategy", {})
                eff_min = live.get("eff_min_conf", "?")
                regime  = live.get("market_regime", "UNKNOWN")

            try:
                from engine.profiles import TradingProfile
                pr = dataclasses.replace(TradingProfile.load(profile_name))
                self.send(
                    f"🎯 <b>Trading Profile: {profile_name}</b>\n\n"
                    f"  Min Confidence: {pr.min_confidence}\n"
                    f"  Agent Agreement: {pr.min_agent_agreement}/3\n"
                    f"  Net Score Threshold: ±{pr.net_score_threshold}\n"
                    f"  SMC Sub-checks Min: {pr.smc_sub_checks_min}/5\n"
                    f"  Stop Loss: {pr.stop_loss_atr_mult}× ATR\n"
                    f"  Take Profit: {pr.take_profit_atr_mult}× ATR\n"
                    f"  Trail (post-TP1): {pr.trail_atr_mult}× ATR\n"
                    f"  HTF Filter: {pr.htf_filter_mode}\n\n"
                    f"<b>Live:</b> eff_min={eff_min} | regime={regime}\n\n"
                    f"/profile_strict  /profile_balanced  /profile_aggressive"
                )
            except Exception:
                self.send(f"🎯 <b>Profile: {profile_name}</b>\n\neff_min={eff_min} | regime={regime}")
        except Exception as e:
            self.send(f"Profile error: {e}")

    def _cmd_profile_set(self, profile_name: str):
        """Switch to a different trading profile."""
        try:
            import fcntl
            saved = False
            for cfg_path in (CFG_SPOT, CFG_FUT):
                if cfg_path.exists():
                    with open(cfg_path) as f:
                        cfg = yaml.safe_load(f) or {}
                    cfg.setdefault("strategy", {})["trading_profile"] = profile_name
                    tmp_path = cfg_path.with_suffix(".tmp.yaml")
                    with open(tmp_path, "w") as f:
                        fcntl.flock(f, fcntl.LOCK_EX)
                        yaml.dump(cfg, f, default_flow_style=False, sort_keys=True)
                        fcntl.flock(f, fcntl.LOCK_UN)
                    tmp_path.replace(cfg_path)
                    saved = True

            if saved:
                self.send(
                    f"✅ <b>Profile switched to {profile_name}</b>\n"
                    f"Bot will adopt on next sync cycle.\n"
                    f"Use /profile to verify settings."
                )
            else:
                self.send("❌ No config files found")
        except Exception as e:
            self.send(f"Profile switch error: {e}")

    def _cmd_current_mode(self):
        mode = self._get_mode_from_env()
        icons = {"spot": "🔵", "futures": "🟣", "unknown": "⚪"}
        self.send(f"{icons.get(mode, '⚪')} <b>Current Mode: {mode.upper()}</b>")

    def _cmd_switch_spot(self):
        if self._is_docker():
            self.send("ℹ️ <b>Docker mode</b>: both SPOT and FUTURES run as separate containers simultaneously.\nUse <code>docker compose stop futures-bot</code> on the server if you want only spot.")
            return
        current = self._get_mode_from_env()
        if current == "spot":
            self.send("ℹ️ Already running in SPOT mode")
            return

        self.send("🔄 <b>Switching to SPOT mode...</b>\nThis will take ~30 seconds")
        try:
            home = str(Path.home())
            subprocess.run(["screen", "-S", "cryptobot_v5_futures", "-X", "quit"], timeout=10)
            subprocess.run(["pkill", "-9", "-u", os.environ.get("USER", "root"), "-f", "futures_bot.py"], timeout=10)
            time.sleep(3)
            subprocess.run(["screen", "-wipe"], timeout=5)
            time.sleep(2)
            subprocess.Popen([
                "screen", "-dmS", "cryptobot_v5_spot",
                "bash", "-c",
                f"cd {home}/cryptobot_v5/bot && BOT_MODE=spot python3 launcher.py"
            ])
            time.sleep(3)
            self.send("✅ <b>Switched to SPOT mode!</b>\nUse /status to verify")
        except Exception as e:
            self.send(f"❌ Switch failed: {e}")

    def _cmd_switch_futures(self):
        if self._is_docker():
            self.send("ℹ️ <b>Docker mode</b>: both SPOT and FUTURES run as separate containers simultaneously.\nUse <code>docker compose stop spot-bot</code> on the server if you want only futures.")
            return
        current = self._get_mode_from_env()
        if current == "futures":
            self.send("ℹ️ Already running in FUTURES mode")
            return

        self.send("🔄 <b>Switching to FUTURES mode...</b>\nThis will take ~30 seconds")
        try:
            home = str(Path.home())
            subprocess.run(["screen", "-S", "cryptobot_v5_spot", "-X", "quit"], timeout=10)
            subprocess.run(["pkill", "-9", "-u", os.environ.get("USER", "root"), "-f", "spot_bot.py"], timeout=10)
            time.sleep(3)
            subprocess.run(["screen", "-wipe"], timeout=5)
            time.sleep(2)
            subprocess.Popen([
                "screen", "-dmS", "cryptobot_v5_futures",
                "bash", "-c",
                f"cd {home}/cryptobot_v5/bot && BOT_MODE=futures python3 launcher.py"
            ])
            time.sleep(3)
            self.send("✅ <b>Switched to FUTURES mode!</b>\nUse /status to verify")
        except Exception as e:
            self.send(f"❌ Switch failed: {e}")

    def _cmd_restart(self):
        current = self._get_mode_from_env()
        if current == "unknown":
            self.send("⚠️ Unknown mode. Try /switch_spot or /switch_futures")
            return

        self.send(f"🔄 <b>Restarting {current.upper()} bot...</b>")
        try:
            if self._is_docker():
                home = str(Path.home())
                screen = f"cryptobot_v5_{current}"
                script = f"cd {home}/cryptobot_v5/bot && BOT_MODE={current} python3 launcher.py"
                subprocess.run(["screen", "-S", screen, "-X", "quit"], timeout=10)
                time.sleep(2)
                subprocess.Popen(["screen", "-dmS", screen, "bash", "-c", script])
            else:
                svc = f"cryptobot-{current}"
                subprocess.run(["systemctl", "restart", svc], timeout=10)
            time.sleep(3)
            self.send(f"✅ <b>{current.upper()} bot restarted!</b>")
        except Exception as e:
            self.send(f"❌ Restart failed: {e}")

    def _cmd_stop(self):
        current = self._get_mode_from_env()
        if current == "unknown":
            self.send("ℹ️ Bot is not running")
            return

        self.send(f"⏹ <b>Stopping {current.upper()} bot...</b>")
        try:
            if self._is_docker():
                subprocess.run(["screen", "-S", f"cryptobot_v5_{current}", "-X", "quit"], timeout=10)
                subprocess.run(["pkill", "-9", "-u", os.environ.get("USER", "root"), "-f", f"{current}_bot.py"], timeout=10)
            else:
                subprocess.run(["systemctl", "stop", f"cryptobot-{current}"], timeout=10)
            self.send(f"✅ <b>{current.upper()} bot stopped.</b>\nStart again with /switch_spot or /switch_futures")
        except Exception as e:
            self.send(f"❌ Stop failed: {e}")

    def send_error_alert(self, error_msg, context=""):
        """Called by other modules to alert on errors."""
        msg = f"🚨 <b>BOT ERROR</b> 🚨\n\n"
        if context:
            msg += f"<b>Context:</b> {context}\n"
        msg += f"<b>Error:</b> {str(error_msg)[:200]}"
        self.send(msg)

