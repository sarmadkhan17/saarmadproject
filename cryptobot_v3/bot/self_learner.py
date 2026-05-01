"""
Self Learner v2
Improves bot parameters based on trade results.
Requires minimum 10 closed trades for statistical significance.
"""

import json
import logging
import re
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from groq import Groq
from env_config import get_groq_key, DATA_DIR, BOT_ROOT

log  = logging.getLogger("SelfLearner")
DATA = DATA_DIR


class SelfLearner:
    MIN_TRADES   = 10
    REVIEW_HOURS = 2

    def __init__(self):
        self.client   = Groq(api_key=get_groq_key())
        self.model    = "llama-3.1-8b-instant"
        self.insights = self._load_insights()

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
        # Load from both spot and futures state files
        trades = []
        for fname in ["state.json", "futures_state.json"]:
            p = DATA / fname
            if p.exists():
                with open(p) as f:
                    trades.extend(json.load(f).get("trades", []))
        return trades

    def should_run(self):
        last = self.insights.get("last_review")
        if not last:
            return True
        return (datetime.utcnow() - datetime.fromisoformat(last)) > timedelta(hours=self.REVIEW_HOURS)

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

    def apply_improvements(self, improvements):
        if not improvements:
            return []
        import yaml
        changed = []
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
                config["strategy"]["min_confidence"] = new
                changed.append(f"{cfg_file}: confidence {old}→{new}")

            with open(cfg_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)

        return changed

    def run_learning_cycle(self):
        log.info("SELF-LEARNING CYCLE")
        perf = self.analyze_performance()
        if "error" in perf:
            log.warning(f"Learning skipped: {perf['error']}")
            self.insights["last_review"] = datetime.utcnow().isoformat()
            self._save_insights()
            return perf

        log.info(f"Performance: WR={perf['win_rate']}% PnL=${perf['total_pnl']} Sig={perf['significant']}")

        changes = []
        if perf["significant"]:
            improvements = self.ask_ai_for_improvements(perf)
            if improvements:
                log.info(f"Insight: {improvements.get('key_insight','')}")
                changes = self.apply_improvements(improvements)

        self.insights["total_reviews"] += 1
        self.insights["last_review"]    = datetime.utcnow().isoformat()
        self.insights["performance_history"].append({
            "timestamp": datetime.utcnow().isoformat(),
            "win_rate":  perf["win_rate"],
            "total_pnl": perf["total_pnl"],
            "trades":    perf["total_trades"],
        })
        self.insights["performance_history"] = self.insights["performance_history"][-30:]
        self._save_insights()
        return {"performance": perf, "changes": changes}
