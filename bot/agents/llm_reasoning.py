"""
DeepSeek Reasoning Layer — Actor / Judge / Meta-Judge

Actor  (deepseek-chat / V3):
  Runs every scan cycle per symbol.
  Receives: full signal context + 5 similar past trades from memory.
  Outputs: confidence score, entry rationale, TP/SL guidance.

Judge  (deepseek-reasoner / R1):
  Runs after every closed trade.
  Receives: trade context at entry + outcome.
  Outputs: natural-language critique stored in memory.

Meta-Judge (deepseek-reasoner / R1):
  Runs weekly (or every 20 trades).
  Receives: last 20 Judge critiques.
  Outputs: updated reasoning rules injected into Actor prompt.

All outputs are structured JSON parsed safely.
No gradient descent. No model files. No retraining pipeline.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from core.tz import LOCAL_TZ
from pathlib import Path
from typing import Optional

from openai import OpenAI

from agents import usage_tracker
from core.config import get_deepseek_key

log = logging.getLogger("DeepSeekLLM")


def _parse_json_response(raw: str, who: str) -> Optional[dict]:
    """Robustly parse a JSON object out of an LLM response.

    Handles the failure modes seen in production: an empty body (reasoner
    spent its whole budget on chain-of-thought), markdown fences, and prose
    wrapped around the JSON. Returns None on failure and logs the raw body so
    the cause is visible instead of a bare 'Expecting value' traceback.
    """
    raw = (raw or "").strip()
    if not raw:
        log.warning(f"{who}: empty response body")
        return None
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to the outermost {...} span if the model added prose.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        log.warning(f"{who}: unparseable response: {raw[:200]!r}")
        return None

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = get_deepseek_key()
        _client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )
    return _client


@dataclass
class ActorDecision:
    confidence: float          # 0.0 – 1.0
    approved: bool             # True = LLM endorses the setup
    reasoning: str             # short explanation
    tp_note: str               # any TP guidance from LLM
    sl_note: str               # any SL guidance from LLM
    risk_flag: str             # any concern raised


# ─────────────────────────────────────────────────────────────────────────────
# ACTOR — fast signal reasoning (V3)
# ─────────────────────────────────────────────────────────────────────────────

ACTOR_SYSTEM = """You are a professional cryptocurrency trading analyst.
Your job is to evaluate a trading setup and decide whether it is worth taking.

You will receive:
1. Market context: regime, BTC.D, USDT.D, trend direction
2. Structural signal: SMC analysis (BOS/FVG/sweep), ensemble score
3. Microstructure: order book, CVD, funding rate
4. Similar past trades: how setups like this performed before

Rules you NEVER break:
- You do not predict price. You assess setup quality.
- If microstructure.kill is true, output approved=false, confidence below 0.4.
- If macro.kill is true, output approved=false, confidence=0.0.
- If the recency-weighted win-rate is below 40%, reduce confidence by 0.15
  (use the provided smoothed win-rate, not a raw count off a tiny sample).
- Confidence 0.0–1.0. approved=true only when confidence >= the approval
  threshold stated in the final rule line below.

Output ONLY valid JSON. No preamble. No explanation outside JSON. Format:
{
  "approved": true or false,
  "confidence": 0.00,
  "reasoning": "one sentence max 100 chars",
  "tp_note": "one sentence or empty string",
  "sl_note": "one sentence or empty string",
  "risk_flag": "one concern or empty string"
}"""


def _decayed_winrate(similar_trades: list, half_life_h: float, prior: float) -> tuple:
    """Recency-weighted, Bayesian-smoothed win-rate over similar past trades.

    Fixes the freeze loop: a raw wins/total off the 5 most-recent similar trades
    collapses to ~0% after a single losing session, which pins Actor confidence
    below threshold and stops trading — so no new outcomes are ever recorded and
    the prior never recovers. Here each trade is weighted by exp-decay on its age
    (so a cluster of same-session losses fades over `half_life_h`) and the rate is
    smoothed toward 0.5 with a Beta(prior, prior) pseudo-count (so a tiny sample
    cannot read as 0% or 100%). Returns (smoothed_rate, raw_wins, total, avg_r).
    """
    now = datetime.now(LOCAL_TZ)
    w_sum = w_win = wr_sum = 0.0
    raw_wins = 0
    for t in similar_trades:
        pnl = t.get("pnl", 0) or 0
        age_h = 24.0  # default if timestamp missing/unparseable
        ca = t.get("closed_at")
        if ca:
            try:
                dt = datetime.fromisoformat(str(ca).replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=LOCAL_TZ)
                age_h = max(0.0, (now - dt).total_seconds() / 3600.0)
            except Exception:
                pass
        w = 0.5 ** (age_h / max(half_life_h, 1e-6))
        w_sum += w
        if pnl > 0:
            w_win += w
            raw_wins += 1
        wr_sum += w * (t.get("r_multiple", 0) or 0)
    total = len(similar_trades)
    smoothed_rate = (w_win + prior) / (w_sum + 2 * prior) if (w_sum + 2 * prior) > 0 else 0.5
    avg_r = (wr_sum / w_sum) if w_sum > 0 else 0.0
    return smoothed_rate, raw_wins, total, avg_r


def actor_evaluate(
    symbol: str,
    action: str,
    ensemble_score: float,
    ensemble_confidence: float,
    regime: str,
    macro: dict,
    micro_signal,
    similar_trades: list,
    extra_context: str = "",
    approve_threshold: float = 0.50,
    winrate_half_life_h: float = 48.0,
    winrate_prior: float = 1.0,
    trend_direction: str = "NEUTRAL",
) -> ActorDecision:
    """Call DeepSeek V3 to evaluate a trading setup.

    `approve_threshold` is the confidence floor for approval — wired to
    profile.min_confidence by the caller so the Actor gate matches the
    documented profile floor instead of a hidden hardcoded 0.50.
    """

    # Build similar trades summary (recency-weighted, smoothed — see _decayed_winrate)
    trade_summary = ""
    if similar_trades:
        rate, raw_wins, total, avg_r = _decayed_winrate(
            similar_trades, winrate_half_life_h, winrate_prior)
        trade_summary = (
            f"Similar past trades ({total}, recency-weighted): "
            f"est win-rate {rate:.0%} (smoothed; raw {raw_wins}/{total}), "
            f"avg R={avg_r:.2f}. "
        )
        for t in similar_trades[:3]:
            outcome = "WIN" if t.get("pnl", 0) > 0 else "LOSS"
            trade_summary += f"[{t.get('symbol','?')} {t.get('side','?')} {outcome} R={t.get('r_multiple',0):.1f}] "

    user_msg = f"""Symbol: {symbol} | Action: {action}
Regime: {regime} | Trend direction: {trend_direction}
Macro: BTC.D={macro.get('btc_d',50):.1f}% ({macro.get('btc_d_roc',0):+.2f}%/hr) | USDT.D={macro.get('usdt_d',5):.1f}% ({macro.get('usdt_d_roc',0):+.2f}%/hr) | kill={macro.get('kill',False)}
Ensemble: net_score={ensemble_score:+.3f} confidence={ensemble_confidence:.2f}
Microstructure: ob_ratio={micro_signal.ob_imbalance:.2f} cvd={micro_signal.cvd_direction} divergence={micro_signal.cvd_divergence} kill={micro_signal.kill}
{trade_summary}
{extra_context}"""

    system_content = (
        ACTOR_SYSTEM
        + f"\n\nFINAL RULE — approval threshold = {approve_threshold:.2f}: "
          f"set approved=true only when confidence >= {approve_threshold:.2f}."
    )
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=150,
        )
        try:
            usage_tracker.record("deepseek-chat",
                                 resp.usage.prompt_tokens,
                                 resp.usage.completion_tokens)
        except Exception: pass
        parsed = _parse_json_response(resp.choices[0].message.content, f"Actor {symbol}")
        if parsed is None:
            raise ValueError("unparseable Actor response")
        return ActorDecision(
            confidence=float(parsed.get("confidence", 0.5)),
            approved=bool(parsed.get("approved", False)),
            reasoning=str(parsed.get("reasoning", ""))[:150],
            tp_note=str(parsed.get("tp_note", "")),
            sl_note=str(parsed.get("sl_note", "")),
            risk_flag=str(parsed.get("risk_flag", "")),
        )
    except Exception as e:
        log.warning(f"Actor LLM error {symbol}: {e}")
        # Fallback: trust the ensemble score
        fallback_conf = min(0.65, ensemble_confidence)
        return ActorDecision(
            confidence=fallback_conf,
            approved=fallback_conf >= approve_threshold and not micro_signal.kill and not macro.get("kill"),
            reasoning=f"LLM unavailable — fallback conf={fallback_conf:.2f}",
            tp_note="", sl_note="", risk_flag="LLM unavailable",
        )


# ─────────────────────────────────────────────────────────────────────────────
# JUDGE — post-trade critique (R1)
# ─────────────────────────────────────────────────────────────────────────────

JUDGE_SYSTEM = """You are a senior trading risk manager reviewing a closed trade.
Analyze whether the entry decision was correct given the available information AT ENTRY TIME.
Do not use hindsight. Focus on: was the setup structurally valid? was risk managed correctly?
were there warning signs that were ignored?

Output ONLY valid JSON:
{
  "decision_quality": "good" | "acceptable" | "poor",
  "entry_valid": true or false,
  "risk_managed": true or false,
  "missed_warnings": "specific signals ignored or empty string",
  "lesson": "one actionable lesson under 120 chars",
  "pattern_tag": "short tag for this setup type e.g. BOS_long_RANGING"
}"""


def judge_review(trade: dict, entry_context: dict) -> Optional[dict]:
    """Call DeepSeek R1 to review a closed trade."""
    pnl   = trade.get("pnl", 0)
    r_mult = entry_context.get("r_multiple", 0)
    outcome = "WIN" if pnl > 0 else "LOSS"

    user_msg = f"""CLOSED TRADE — {outcome}
Symbol: {trade.get('symbol')} | Side: {trade.get('side')} | Mode: {trade.get('mode')}
Entry: ${trade.get('price',0):.4f} | Exit: ${trade.get('close_price',0):.4f}
PnL: ${pnl:+.4f} | R-multiple: {r_mult:+.2f}
Duration: {trade.get('duration_hours',0):.1f}h

AT ENTRY — context:
Regime: {entry_context.get('regime','?')} | Trend direction: {entry_context.get('trend_direction','?')}
BTC.D: {entry_context.get('btc_d',50):.1f}% roc={entry_context.get('btc_d_roc',0):+.2f}%/hr
USDT.D: {entry_context.get('usdt_d',5):.1f}%
Ensemble score: {entry_context.get('ensemble_score',0):+.3f}
Confidence: {entry_context.get('confidence',0):.2f}
OB imbalance: {entry_context.get('ob_imbalance',1):.2f}
CVD: {entry_context.get('cvd_direction','?')} divergence={entry_context.get('cvd_divergence',False)}
Actor reasoning: {entry_context.get('actor_reasoning','')}
Exit reason: {trade.get('close_reason','')}"""

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=3000,
        )
        try:
            usage_tracker.record("deepseek-reasoner",
                                 resp.usage.prompt_tokens,
                                 resp.usage.completion_tokens)
        except Exception: pass
        parsed = _parse_json_response(resp.choices[0].message.content, "Judge")
        if parsed is None:
            return None
        parsed["trade_id"] = trade.get("id", "")
        parsed["symbol"]   = trade.get("symbol", "")
        parsed["outcome"]  = outcome
        parsed["pnl"]      = pnl
        parsed["reviewed_at"] = datetime.now(LOCAL_TZ).isoformat()
        return parsed
    except Exception as e:
        log.warning(f"Judge LLM error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# META-JUDGE — weekly rule refinement (R1)
# ─────────────────────────────────────────────────────────────────────────────

META_JUDGE_SYSTEM = """You are a quantitative trading strategist reviewing a series of trade critiques.
Identify consistent patterns in what went wrong and what went right.
Produce updated rules that should be injected into the trading agent's reasoning.

Output ONLY valid JSON:
{
  "summary": "2-3 sentence overview of patterns found",
  "updated_rules": [
    "Rule 1 — specific, actionable, under 100 chars each",
    "Rule 2 — ...",
    "Rule 3 — ..."
  ],
  "avoid_patterns": ["pattern tag to avoid", ...],
  "favour_patterns": ["pattern tag to favour", ...]
}"""


def meta_judge_synthesize(critiques: list) -> Optional[dict]:
    """Call DeepSeek R1 to synthesize lessons from multiple Judge critiques."""
    if len(critiques) < 5:
        log.info("Meta-Judge: fewer than 5 critiques — skipping")
        return None

    critique_text = ""
    for i, c in enumerate(critiques[-20:], 1):  # last 20 max
        critique_text += (
            f"\n{i}. [{c.get('outcome','?')}] {c.get('symbol','?')} "
            f"pattern={c.get('pattern_tag','?')} "
            f"quality={c.get('decision_quality','?')} "
            f"lesson={c.get('lesson','')}\n"
        )

    user_msg = f"Review these {len(critiques)} trade critiques and synthesize updated rules:\n{critique_text}"

    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model="deepseek-reasoner",
            messages=[
                {"role": "system", "content": META_JUDGE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=3000,
        )
        try:
            usage_tracker.record("deepseek-reasoner",
                                 resp.usage.prompt_tokens,
                                 resp.usage.completion_tokens)
        except Exception: pass
        parsed = _parse_json_response(resp.choices[0].message.content, "Meta-Judge")
        if parsed is None:
            return None
        parsed["synthesized_at"] = datetime.now(LOCAL_TZ).isoformat()
        parsed["critiques_count"] = len(critiques)
        log.info(f"Meta-Judge synthesized {len(critiques)} critiques → {len(parsed.get('updated_rules',[]))} rules")
        return parsed
    except Exception as e:
        log.warning(f"Meta-Judge LLM error: {e}")
        return None
