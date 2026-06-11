"""
DeepSeek usage tracker — logs API calls and estimated cost.
Writes to data/deepseek_usage.json for dashboard display.

Pricing (as of 2026):
  deepseek-chat (V3):     $0.27/M input, $1.10/M output
  deepseek-reasoner (R1): $0.55/M input, $2.19/M output
"""

import json
import logging
import threading
from datetime import date, datetime, timezone
from core.tz import LOCAL_TZ
from pathlib import Path
from typing import Optional

log = logging.getLogger("DeepSeekUsage")

_lock = threading.Lock()


class DeepSeekUsageTracker:
    PRICES = {
        "deepseek-chat":     {"input": 0.27 / 1e6, "output": 1.10 / 1e6},
        "deepseek-reasoner": {"input": 0.55 / 1e6, "output": 2.19 / 1e6},
        # Groq-hosted skeptic (Gate 5.5)
        "llama-3.3-70b-versatile": {"input": 0.59 / 1e6, "output": 0.79 / 1e6},
    }

    def __init__(self, data_dir: Path):
        data_dir.mkdir(parents=True, exist_ok=True)
        self._path = data_dir / "deepseek_usage.json"
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"daily": {}, "total_cost_usd": 0.0}

    def _save(self):
        try:
            tmp = self._path.with_suffix(".tmp.json")
            with open(tmp, "w") as f:
                json.dump(self._data, f, indent=2)
            tmp.replace(self._path)
        except Exception as e:
            log.error(f"Usage save error: {e}")

    def record(self, model: str, input_tokens: int, output_tokens: int):
        today = str(datetime.now(LOCAL_TZ).date())
        price = self.PRICES.get(model, self.PRICES["deepseek-chat"])
        cost  = input_tokens * price["input"] + output_tokens * price["output"]

        with _lock:
            day = self._data["daily"].setdefault(today, {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0
            })
            day["calls"]         += 1
            day["input_tokens"]  += input_tokens
            day["output_tokens"] += output_tokens
            day["cost_usd"]      = round(day["cost_usd"] + cost, 6)
            # Per-model breakdown so the dashboard can show DeepSeek (Actor/
            # Judge) and Groq (Skeptic) as separate components.
            m = day.setdefault("models", {}).setdefault(model, {
                "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0
            })
            m["calls"]         += 1
            m["input_tokens"]  += input_tokens
            m["output_tokens"] += output_tokens
            m["cost_usd"]      = round(m["cost_usd"] + cost, 6)
            self._data["total_cost_usd"] = round(
                self._data.get("total_cost_usd", 0) + cost, 6
            )
            self._data["last_call"] = datetime.now(LOCAL_TZ).isoformat()
            self._save()

    def today_summary(self) -> dict:
        today = str(datetime.now(LOCAL_TZ).date())
        with _lock:
            day = self._data["daily"].get(today, {})
            return {
                "calls":        day.get("calls", 0),
                "input_tokens": day.get("input_tokens", 0),
                "output_tokens":day.get("output_tokens", 0),
                "cost_usd":     day.get("cost_usd", 0.0),
                "total_cost_usd": self._data.get("total_cost_usd", 0.0),
                # For dashboard compatibility (was: used_today / limit)
                "used_today":   day.get("input_tokens", 0) + day.get("output_tokens", 0),
                "limit":        500000,
            }


# Module-level singleton, initialized by bot on startup
_tracker: Optional[DeepSeekUsageTracker] = None


def init_tracker(data_dir: Path):
    global _tracker
    _tracker = DeepSeekUsageTracker(data_dir)


def record(model: str, input_tokens: int, output_tokens: int):
    if _tracker:
        _tracker.record(model, input_tokens, output_tokens)


def today_summary() -> dict:
    if _tracker:
        return _tracker.today_summary()
    return {"calls": 0, "used_today": 0, "limit": 500000, "cost_usd": 0.0}
