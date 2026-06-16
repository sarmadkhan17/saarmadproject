"""
Tests for trade_memory.annotate_lessons — the "no frozen veto" demotion of
cautionary Judge lessons that the LIVE blended edge now contradicts.

A LOSS lesson ("this setup loses") must stop carrying suppressive weight the
moment the continuously-updated blended stat for the same regime+side shows
positive edge — without waiting to "retire" the lesson from fresh outcomes.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot.agents.trade_memory import annotate_lessons, LESSON_OVERRIDE_MIN_N


def _loss(text="weak-trend longs lose, avoid them"):
    return {"outcome": "LOSS", "lesson": text}


def _win(text="with-trend longs after a sweep paid well"):
    return {"outcome": "WIN", "lesson": text}


def _blended(n, win_rate, avg_r):
    return {"n": n, "win_rate": win_rate, "avg_r": avg_r}


POS = _blended(n=40, win_rate=0.59, avg_r=0.34)   # live edge clearly positive
NEG = _blended(n=40, win_rate=0.40, avg_r=-0.12)  # live edge negative
THIN = _blended(n=3, win_rate=0.90, avg_r=0.80)   # positive but too few samples


class TestDemotion:

    def test_loss_lesson_demoted_when_live_edge_positive(self):
        block = annotate_lessons([_loss()], POS, "WEAK_TREND", "long")
        assert "STALE ADVISORY" in block
        assert "do not gate on it" in block
        assert "59% win" in block
        assert "weak-trend longs lose" in block  # text still shown as context

    def test_loss_lesson_kept_when_live_edge_negative(self):
        block = annotate_lessons([_loss()], NEG, "WEAK_TREND", "long")
        assert "STALE ADVISORY" not in block
        assert "- [LOSS]" in block

    def test_loss_lesson_kept_when_sample_too_thin(self):
        # Positive win-rate but n below the override floor → not trusted to demote.
        assert THIN["n"] < LESSON_OVERRIDE_MIN_N
        block = annotate_lessons([_loss()], THIN, "WEAK_TREND", "long")
        assert "STALE ADVISORY" not in block
        assert "- [LOSS]" in block

    def test_win_lesson_never_demoted(self):
        block = annotate_lessons([_win()], POS, "WEAK_TREND", "long")
        assert "STALE ADVISORY" not in block
        assert "- [WIN]" in block

    def test_positive_winrate_but_negative_avg_r_keeps_caution(self):
        # 55% win but negative avgR (small wins, big losses) is NOT positive edge.
        block = annotate_lessons([_loss()], _blended(40, 0.55, -0.05),
                                 "WEAK_TREND", "long")
        assert "STALE ADVISORY" not in block

    def test_mixed_lessons_demote_only_the_contradicted_loss(self):
        block = annotate_lessons([_loss(), _win()], POS, "WEAK_TREND", "long")
        assert "STALE ADVISORY" in block          # the LOSS demoted
        assert "- [WIN]" in block                 # the WIN untouched
        assert block.count("STALE ADVISORY") == 1

    def test_empty_lessons_returns_empty_string(self):
        assert annotate_lessons([], POS, "WEAK_TREND", "long") == ""

    def test_blank_lesson_text_skipped(self):
        block = annotate_lessons([{"outcome": "LOSS", "lesson": "  "}], POS,
                                 "WEAK_TREND", "long")
        assert block == ""

    def test_header_present_when_any_lesson_rendered(self):
        block = annotate_lessons([_win()], NEG, "WEAK_TREND", "long")
        assert block.startswith("Relevant past lessons:\n")
