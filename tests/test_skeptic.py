"""
Tests for the Skeptic (Gate 5.5) — adversarial deliberation.

The deterministic combiner is the safety-critical part: it alone decides
veto / haircut / pass, and its authority must be strictly one-way (the
skeptic can block or shrink a trade, never enlarge one).
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bot"))

from bot.agents import skeptic


MIN_CONF = 0.45  # typical profile.min_confidence


FULL_K = "event_risk"  # category with category_k = 1.0 → undiscounted math


class TestCombine:
    def test_weak_rebuttal_passes_unchanged(self):
        verdict, eff, mult = skeptic.combine(0.70, 0.10, MIN_CONF, objection=FULL_K)
        assert verdict == "pass" and mult == 1.0
        assert eff == pytest.approx(0.66)

    def test_strong_rebuttal_vetoes(self):
        # 0.55 − 0.4×0.80 = 0.23 < 0.45
        verdict, eff, mult = skeptic.combine(0.55, 0.80, MIN_CONF, objection=FULL_K)
        assert verdict == "veto" and mult == 0.0

    def test_medium_rebuttal_haircuts(self):
        # 0.65 − 0.4×0.40 = 0.49 → inside [0.45, 0.55) band
        verdict, eff, mult = skeptic.combine(0.65, 0.40, MIN_CONF, objection=FULL_K)
        assert verdict == "haircut" and mult == 0.5

    def test_zero_rebuttal_never_upsizes(self):
        # One-way authority: even total agreement gives at most ×1.0.
        verdict, eff, mult = skeptic.combine(0.90, 0.0, MIN_CONF, objection=FULL_K)
        assert verdict == "pass" and mult == 1.0 and eff == 0.90

    def test_strength_clamped_to_unit_interval(self):
        # Out-of-range model output must not amplify the penalty…
        _, eff_hi, _ = skeptic.combine(0.80, 5.0, MIN_CONF, objection=FULL_K)
        assert eff_hi == pytest.approx(0.80 - 0.4 * 1.0)
        # …or turn the skeptic into a booster via negative strength.
        _, eff_lo, _ = skeptic.combine(0.80, -3.0, MIN_CONF, objection=FULL_K)
        assert eff_lo == pytest.approx(0.80)

    def test_boundaries(self):
        # effective exactly at min_conf → not a veto (strict <)
        verdict, _, _ = skeptic.combine(MIN_CONF, 0.0, MIN_CONF, objection=FULL_K)
        assert verdict == "haircut"
        # effective exactly at min_conf + band → pass (strict <)
        verdict, _, _ = skeptic.combine(MIN_CONF + 0.10, 0.0, MIN_CONF, objection=FULL_K)
        assert verdict == "pass"


class TestNoDoubleJeopardy:
    """Upstream-owned objection categories get a discounted k and can
    never fully veto — the regime is already priced by the ensemble."""

    def test_regime_objection_k_is_discounted(self):
        # Yesterday's live failure case: conf 0.50, strength 0.70.
        # Full k would give 0.50 − 0.28 = 0.22 → veto.
        # Discounted (0.15×k): 0.50 − 0.042 = 0.458 → pass/haircut, not veto.
        verdict, eff, mult = skeptic.combine(
            0.50, 0.70, 0.42, objection="regime_mismatch")
        assert verdict != "veto" and mult > 0.0
        assert eff == pytest.approx(0.50 - 0.4 * 0.15 * 0.70)

    def test_regime_objection_caps_at_haircut_even_below_floor(self):
        # Even when effective drops below min_conf, a regime objection
        # can only shrink the position, never block it.
        verdict, eff, mult = skeptic.combine(
            0.43, 1.0, 0.42, objection="regime_mismatch")
        assert eff < 0.42
        assert verdict == "haircut" and mult == 0.5

    def test_micro_contradiction_also_confined(self):
        verdict, _, mult = skeptic.combine(
            0.43, 1.0, 0.42, objection="micro_contradiction")
        assert verdict == "haircut" and mult == 0.5

    def test_owned_domain_keeps_full_veto(self):
        for cat in ("crowded_narrative", "stale_precedent", "event_risk"):
            verdict, _, mult = skeptic.combine(0.55, 0.80, MIN_CONF, objection=cat)
            assert verdict == "veto" and mult == 0.0, cat

    def test_other_category_half_k_no_veto(self):
        # "other" is ambiguous → half penalty, haircut at worst.
        verdict, eff, _ = skeptic.combine(0.55, 0.80, MIN_CONF, objection="other")
        assert eff == pytest.approx(0.55 - 0.4 * 0.5 * 0.80)
        assert verdict != "veto"

    def test_config_overrides_respected(self):
        # Operator can re-arm regime vetoes via config if desired.
        verdict, _, _ = skeptic.combine(
            0.55, 0.80, MIN_CONF, objection="regime_mismatch",
            category_k={"regime_mismatch": 1.0},
            veto_categories=("regime_mismatch",))
        assert verdict == "veto"


class TestSkepticEvaluate:
    MICRO = SimpleNamespace(ob_imbalance=1.2, cvd_direction="up",
                            cvd_divergence=False)

    def _call(self, **kw):
        return skeptic.skeptic_evaluate(
            symbol="ETH/USDT", action="BUY", thesis="test thesis",
            regime="WEAK_TREND", trend_direction="BEARISH",
            macro={"btc_d": 55.0, "btc_d_roc": 0.0, "usdt_d": 5.0, "usdt_d_roc": 0.0},
            micro_signal=self.MICRO, ensemble_score=0.4, **kw)

    def test_fails_open_on_api_error(self):
        with patch.object(skeptic, "_get_client",
                          side_effect=RuntimeError("groq down")):
            assert self._call() is None

    def test_fails_open_on_unparseable_response(self):
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="not json"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **k: resp)))
        with patch.object(skeptic, "_get_client", return_value=client):
            assert self._call() is None

    def test_parses_and_clamps_model_output(self):
        body = '{"rebuttal_strength": 1.7, "objection": "nonsense_category", "statement": "x" }'
        resp = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=body))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1))
        client = SimpleNamespace(chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **k: resp)))
        with patch.object(skeptic, "_get_client", return_value=client):
            d = self._call()
        assert d.rebuttal_strength == 1.0          # clamped
        assert d.objection == "other"              # unknown category coerced
        assert d.statement == "x"

    def test_available_reflects_key_presence(self):
        with patch.object(skeptic, "get_groq_key", return_value=""):
            assert not skeptic.available()
        with patch.object(skeptic, "get_groq_key", return_value="gsk_x"):
            assert skeptic.available()
