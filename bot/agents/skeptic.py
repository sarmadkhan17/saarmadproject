"""
Skeptic — adversarial deliberation gate (Gate 5.5).

A second, independent LLM (Llama on Groq — deliberately a different model
family than the DeepSeek Actor, so its blind spots don't overlap) argues
AGAINST every trade the Actor approves. It sees the Actor's thesis but NOT
the Actor's confidence (no anchoring), and outputs the single strongest
honest objection with a 0–1 strength.

Resolution is deterministic code, never a third LLM:

    effective_conf = actor_conf − k × category_k[objection] × rebuttal_strength
      effective <  min_conf            → veto   (trade blocked, shadow-logged)
      effective <  min_conf + band     → haircut (position size × 0.5)
      otherwise                        → pass   (unchanged)

No-double-jeopardy: objection categories owned by upstream gates
(regime_mismatch, micro_contradiction) carry a discounted k and can at most
haircut — never veto. Full veto power is reserved for the skeptic's exclusive
domain: crowded_narrative, stale_precedent, event_risk.

Authority is one-way: the skeptic can block or shrink a trade, never enlarge
one — two LLMs agreeing must not compound into overconfidence. Any API
failure fails OPEN (the skeptic is an alpha filter, not a safety gate; the
macro/micro kill switches stay authoritative).
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from core.config import get_groq_key
from agents import usage_tracker

log = logging.getLogger("Skeptic")

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

OBJECTION_CATEGORIES = (
    "regime_mismatch", "micro_contradiction", "crowded_narrative",
    "stale_precedent", "event_risk", "other",
)

# No-double-jeopardy: regime/trend is owned by the ensemble and order flow by
# the microstructure gate — both already priced the setup before the skeptic
# sees it. Objections in those categories carry a discounted k (they can still
# shrink a position) and may never fully veto; the skeptic's exclusive veto
# domain is what no upstream gate can see.
DEFAULT_CATEGORY_K = {
    "regime_mismatch":    0.15,
    "micro_contradiction": 0.15,
    "crowded_narrative":  1.0,
    "stale_precedent":    1.0,
    "event_risk":         1.0,
    "other":              0.5,
}
DEFAULT_VETO_CATEGORIES = ("crowded_narrative", "stale_precedent", "event_risk")

SKEPTIC_SYSTEM = """You are the risk officer at a crypto trading desk. A trader
proposes a trade; your ONLY job is to find the strongest honest case AGAINST it.
You never approve trades and you never soften your objection to be agreeable.

Division of labour: the desk's quantitative gates have ALREADY priced the
regime, the trend, and the live order flow into this setup before it reached
you — restating "the trend is weak/neutral" adds nothing. Your unique value is
what those gates cannot see: a crowded narrative, precedent that no longer
applies, scheduled event risk, or an internal contradiction in the trader's
own thesis. Prefer those; only fall back to regime objections when the
mismatch is truly egregious.

Calibration rules:
- rebuttal_strength reflects how damaging your SINGLE BEST objection is,
  NOT how many objections you found.
- 0.0-0.2: nitpicks only — the setup is fundamentally sound.
- 0.3-0.5: a real concern that justifies a smaller position.
- 0.6-0.7: a serious structural flaw in the thesis.
- 0.8-1.0: the trade is very likely a mistake (fighting the regime, entering
  into a known squeeze setup, microstructure flatly contradicts the thesis).
- If the data genuinely supports the trade, say so with a LOW strength.
  A skeptic who always screams is ignored; calibration is your credibility.

Output ONLY valid JSON. No preamble. Format:
{
  "rebuttal_strength": 0.00,
  "objection": "one of: regime_mismatch | micro_contradiction | crowded_narrative | stale_precedent | event_risk | other",
  "statement": "your single strongest objection, one sentence, max 140 chars"
}"""


@dataclass
class SkepticDecision:
    rebuttal_strength: float   # 0.0 – 1.0, clamped
    objection: str             # category from OBJECTION_CATEGORIES
    statement: str             # one-sentence strongest objection


_client = None


def available() -> bool:
    return bool(get_groq_key())


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI  # lazy: keeps the module importable without the SDK
        _client = OpenAI(api_key=get_groq_key(), base_url=GROQ_BASE_URL)
    return _client


def _parse_json(raw: str) -> Optional[dict]:
    """Parse a JSON object out of an LLM response (markdown fences, wrapped
    prose). Mirrors llm_reasoning._parse_json_response, kept local so this
    module imports without the openai SDK."""
    raw = (raw or "").strip()
    if not raw:
        log.warning("Skeptic: empty response body")
        return None
    cleaned = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                pass
        log.warning(f"Skeptic: unparseable response: {raw[:200]!r}")
        return None


def skeptic_evaluate(
    symbol: str,
    action: str,
    thesis: str,
    regime: str,
    trend_direction: str,
    macro: dict,
    micro_signal,
    ensemble_score: float,
    trend_change: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> Optional[SkepticDecision]:
    """Ask the skeptic for its best case against the trade.

    Gets the same raw context as the Actor plus the Actor's one-line thesis —
    but never the Actor's confidence, so it argues the setup rather than
    negotiating the score. Returns None on any failure (fail-open).
    """
    turn_note = (f"\nTrend change: 1h momentum turned {trend_change} against "
                 f"the 4h trend (early reversal window)." if trend_change else "")
    user_msg = f"""Proposed trade: {action} {symbol}
Trader's thesis: {thesis or 'n/a'}
Regime: {regime} | Trend direction: {trend_direction}{turn_note}
Macro: BTC.D={macro.get('btc_d', 50):.1f}% ({macro.get('btc_d_roc', 0):+.2f}%/hr) | USDT.D={macro.get('usdt_d', 5):.1f}% ({macro.get('usdt_d_roc', 0):+.2f}%/hr)
Ensemble net score: {ensemble_score:+.3f}
Microstructure: ob_ratio={micro_signal.ob_imbalance:.2f} cvd={micro_signal.cvd_direction} divergence={micro_signal.cvd_divergence}

What is the strongest case against taking this trade right now?"""

    try:
        resp = _get_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SKEPTIC_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=150,
        )
        try:
            usage_tracker.record(model,
                                 resp.usage.prompt_tokens,
                                 resp.usage.completion_tokens)
        except Exception:
            pass
        data = _parse_json(resp.choices[0].message.content)
        if data is None:
            return None
        strength = max(0.0, min(1.0, float(data.get("rebuttal_strength", 0.0))))
        objection = str(data.get("objection", "other")).strip()
        if objection not in OBJECTION_CATEGORIES:
            objection = "other"
        return SkepticDecision(
            rebuttal_strength=strength,
            objection=objection,
            statement=str(data.get("statement", ""))[:200],
        )
    except Exception as e:
        log.warning(f"Skeptic call failed for {symbol} (fail-open): {e}")
        return None


def combine(actor_conf: float, rebuttal_strength: float, min_conf: float,
            k: float = 0.4, haircut_band: float = 0.10,
            objection: str = "other",
            category_k: Optional[dict] = None,
            veto_categories: Optional[tuple] = None) -> tuple:
    """Deterministic resolution of Actor vs Skeptic.

    Returns (verdict, effective_conf, size_mult) where verdict is
    "veto" | "haircut" | "pass". One-way: size_mult is never above 1.0.

    k is scaled per objection category (no-double-jeopardy: upstream-owned
    objections like regime_mismatch carry a discounted penalty), and only
    categories in veto_categories may fully block — everything else bottoms
    out at a haircut.
    """
    cat_k = {**DEFAULT_CATEGORY_K, **(category_k or {})}
    vetoable = tuple(veto_categories) if veto_categories is not None \
        else DEFAULT_VETO_CATEGORIES
    k_eff = k * float(cat_k.get(objection, cat_k.get("other", 0.5)))
    effective = round(actor_conf - k_eff * max(0.0, min(1.0, rebuttal_strength)), 4)
    if effective < min_conf:
        if objection in vetoable:
            return "veto", effective, 0.0
        return "haircut", effective, 0.5
    if effective < min_conf + haircut_band:
        return "haircut", effective, 0.5
    return "pass", effective, 1.0
