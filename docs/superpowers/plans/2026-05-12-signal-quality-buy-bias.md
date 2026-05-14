# Signal Quality — BUY Bias Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the bot's persistent BUY bias in bearish strong-trend markets across four layers: SMC agent direction awareness, ML HOLD-probability confidence penalty, risk-gate 20EMA momentum filter, and ensemble directional damping.

**Architecture:** Each session is an isolated edit to one file. Sessions 1 and 4 damp counter-trend signals at different layers (agent score vs weighted net). Session 2 penalises ML confidence proportionally to model uncertainty. Session 3 is the only hard-block gate. All four can be applied in any order; Session 5 (config) is deferred.

**Tech Stack:** Python 3.12, pandas, numpy, pytest (`venv/bin/pytest`), existing test helpers in `bot/tests/`

---

## File Map

| File | Session | Change |
|---|---|---|
| `bot/engine/smc_agent.py` | 1 | Lines 51–90: replace `_trending` with directional flags; update sweep/BOS scoring |
| `bot/models/ai_strategy.py` | 2 | `AIStrategyEngine.predict()` ~line 645: add `hold_prob` penalty |
| `bot/engine/risk_agent.py` | 3 | `evaluate()` ~line 242: insert Gate 4c (20EMA filter) |
| `bot/engine/ensemble.py` | 4 | `_aggregate()` ~line 105: insert directional bias block |
| `bot/tests/test_smc_direction_bias.py` | 1 | New test file |
| `bot/tests/test_hold_prob_penalty.py` | 2 | New test file |
| `bot/tests/test_ema_momentum_gate.py` | 3 | New test file |
| `bot/tests/test_ensemble_directional_bias.py` | 4 | New test file |

---

## Task 1: SMC Agent — Direction-Aware Trend Filter

**Files:**
- Modify: `bot/engine/smc_agent.py:51–90`
- Create: `bot/tests/test_smc_direction_bias.py`

- [ ] **Step 1.1: Write the failing tests**

Create `bot/tests/test_smc_direction_bias.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from engine.smc_agent import SMCAgent


def _make_df(n=100):
    np.random.seed(0)
    close = 100 + np.cumsum(np.random.randn(n) * 0.3)
    return pd.DataFrame({
        "open":   close * 0.999,
        "high":   close * 1.002,
        "low":    close * 0.998,
        "close":  close,
        "volume": np.random.uniform(1000, 3000, n),
    })


def _profile():
    p = MagicMock()
    p.smc_liquidity_sweep_pct = 0.001
    p.smc_bos_body_pct = 0.001
    p.smc_volume_spike_ratio = 2.0
    p.smc_pattern_completion = 0.6
    p.smc_sub_checks_min = 1
    return p


def _patched_agent(sweep_dir=None, bos_dir=None):
    """SMCAgent with all detectors mocked to return known directions."""
    agent = SMCAgent()
    agent._detect_liquidity_sweep = lambda *a, **k: (
        {"direction": sweep_dir, "pct": 0.005, "level": 100.0} if sweep_dir else {"direction": None}
    )
    agent._detect_bos = lambda *a, **k: (
        {"direction": bos_dir, "body_pct": 0.01} if bos_dir else {"direction": None}
    )
    agent._detect_fvg             = lambda *a, **k: {"direction": None}
    agent._detect_volume_spike    = lambda *a, **k: {}
    agent._detect_pattern_completion = lambda *a, **k: {"direction": None}
    return agent


def test_bearish_sweep_in_bearish_trend_gets_full_credit():
    """Bearish sweep must score 0.35 (full) in STRONG_TREND + BEARISH."""
    agent = _patched_agent(sweep_dir="bearish")
    result = agent.analyze(_make_df(), _profile(),
                           {"regime": "STRONG_TREND", "trend_direction": "BEARISH"})
    assert result.sell_score == pytest.approx(0.35), f"sell_score={result.sell_score}"
    assert result.buy_score == 0.0


def test_bearish_sweep_in_bullish_trend_is_suppressed():
    """Bearish sweep must score 0.0 (suppressed) in STRONG_TREND + BULLISH."""
    agent = _patched_agent(sweep_dir="bearish")
    result = agent.analyze(_make_df(), _profile(),
                           {"regime": "STRONG_TREND", "trend_direction": "BULLISH"})
    assert result.sell_score == 0.0, f"sell_score should be 0, got {result.sell_score}"


def test_bearish_sweep_no_trend_gets_full_credit():
    """Bearish sweep must score 0.35 when not in a trend regime."""
    agent = _patched_agent(sweep_dir="bearish")
    result = agent.analyze(_make_df(), _profile(),
                           {"regime": "RANGING", "trend_direction": "NEUTRAL"})
    assert result.sell_score == pytest.approx(0.35)


def test_bullish_sweep_in_bullish_trend_gets_full_credit():
    agent = _patched_agent(sweep_dir="bullish")
    result = agent.analyze(_make_df(), _profile(),
                           {"regime": "STRONG_TREND", "trend_direction": "BULLISH"})
    assert result.buy_score == pytest.approx(0.35)


def test_bullish_sweep_in_bearish_trend_is_suppressed():
    agent = _patched_agent(sweep_dir="bullish")
    result = agent.analyze(_make_df(), _profile(),
                           {"regime": "STRONG_TREND", "trend_direction": "BEARISH"})
    assert result.buy_score == 0.0


def test_bearish_bos_in_bearish_trend_gets_full_credit():
    agent = _patched_agent(bos_dir="bearish")
    result = agent.analyze(_make_df(), _profile(),
                           {"regime": "STRONG_TREND", "trend_direction": "BEARISH"})
    assert result.sell_score == pytest.approx(0.30)


def test_bearish_bos_in_bullish_trend_is_suppressed():
    agent = _patched_agent(bos_dir="bearish")
    result = agent.analyze(_make_df(), _profile(),
                           {"regime": "STRONG_TREND", "trend_direction": "BULLISH"})
    assert result.sell_score == 0.0
```

- [ ] **Step 1.2: Run tests — confirm they fail**

```bash
cd /root/cryptobot_v3 && venv/bin/pytest bot/tests/test_smc_direction_bias.py -v --tb=short 2>&1 | tail -20
```

Expected: `FAILED` for all 7 tests (current code returns 0.15 for bearish in trend, not 0.35).

- [ ] **Step 1.3: Apply the fix to `bot/engine/smc_agent.py`**

In `smc_agent.py`, replace lines 51–90 (the comment through the BOS block):

**Remove** these lines (51–90):
```python
        # In STRONG_TREND/TRENDING regimes, bearish reversal patterns get partial credit
        # (0.15) instead of full credit (0.30-0.35) — they fire on pullbacks within trends.
        _regime_str = ((market_ctx or {}).get("regime") or "").upper()
        _trending = "TREND" in _regime_str
```
and:
```python
        # Sweep: ±0.35 (±0.15 during trend — partial credit, not full suppression)
        sweep = checks["sweep"]
        if sweep.get("direction") == "bullish":
            buy_score += 0.35
            reasons_parts.append(f"sweep+{sweep.get('pct',0):.2%}")
            active_checks += 1
        elif sweep.get("direction") == "bearish":
            sell_score += 0.15 if _trending else 0.35
            reasons_parts.append(f"sweep-{sweep.get('pct',0):.2%}")
            active_checks += 1

        # BOS: ±0.30 (±0.15 during trend — partial credit, not full suppression)
        bos = checks["bos"]
        if bos.get("direction") == "bullish":
            buy_score += 0.30
            reasons_parts.append(f"BOS+{bos.get('body_pct',0):.0%}")
            active_checks += 1
        elif bos.get("direction") == "bearish":
            sell_score += 0.15 if _trending else 0.30
            reasons_parts.append(f"BOS-{bos.get('body_pct',0):.0%}")
            active_checks += 1
```

**Replace** with:
```python
        _regime_str    = ((market_ctx or {}).get("regime") or "").upper()
        _trend_dir     = ((market_ctx or {}).get("trend_direction") or "").upper()
        _is_bull_trend = "TREND" in _regime_str and _trend_dir == "BULLISH"
        _is_bear_trend = "TREND" in _regime_str and _trend_dir == "BEARISH"
```
and:
```python
        # Sweep: ±0.35 — suppressed only when counter-trend
        sweep = checks["sweep"]
        if sweep.get("direction") == "bullish" and not _is_bear_trend:
            buy_score += 0.35
            reasons_parts.append(f"sweep+{sweep.get('pct',0):.2%}")
            active_checks += 1
        elif sweep.get("direction") == "bearish" and not _is_bull_trend:
            sell_score += 0.35
            reasons_parts.append(f"sweep-{sweep.get('pct',0):.2%}")
            active_checks += 1

        # BOS: ±0.30 — suppressed only when counter-trend
        bos = checks["bos"]
        if bos.get("direction") == "bullish" and not _is_bear_trend:
            buy_score += 0.30
            reasons_parts.append(f"BOS+{bos.get('body_pct',0):.0%}")
            active_checks += 1
        elif bos.get("direction") == "bearish" and not _is_bull_trend:
            sell_score += 0.30
            reasons_parts.append(f"BOS-{bos.get('body_pct',0):.0%}")
            active_checks += 1
```

- [ ] **Step 1.4: Run tests — confirm all pass**

```bash
cd /root/cryptobot_v3 && venv/bin/pytest bot/tests/test_smc_direction_bias.py -v --tb=short 2>&1 | tail -15
```

Expected: `7 passed`.

- [ ] **Step 1.5: Commit**

```bash
cd /root/cryptobot_v3 && git add bot/engine/smc_agent.py bot/tests/test_smc_direction_bias.py
git commit -m "fix: SMC agent — direction-aware trend filter for sweep/BOS scoring"
```

---

## Task 2: ML Confidence Penalty via HOLD Probability

**Files:**
- Modify: `bot/models/ai_strategy.py` — `AIStrategyEngine.predict()` (~line 645)
- Create: `bot/tests/test_hold_prob_penalty.py`

- [ ] **Step 2.1: Write the failing tests**

Create `bot/tests/test_hold_prob_penalty.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from models.ai_strategy import AIStrategyEngine


def _engine_with_mocked_models(rf_probs, lgbm_probs):
    """AIStrategyEngine with RF and LGBM mocked to return specific probs.
    probs format: [sell_p, hold_p, buy_p].
    Action is determined by argmax of probs (matching production logic).
    """
    engine = AIStrategyEngine()
    rf_action   = {0: "SELL", 1: "HOLD", 2: "BUY"}[int(np.argmax(rf_probs))]
    lgbm_action = {0: "SELL", 1: "HOLD", 2: "BUY"}[int(np.argmax(lgbm_probs))]
    engine.rf.predict   = lambda df: {"action": rf_action,   "confidence": float(max(rf_probs)),   "probs": list(rf_probs)}
    engine.lgbm.predict = lambda df: {"action": lgbm_action, "confidence": float(max(lgbm_probs)), "probs": list(lgbm_probs)}
    engine._get_dynamic_weights = lambda: (0.5, 0.5)
    return engine


def test_zero_hold_prob_no_penalty():
    """With hold_prob=0.0, confidence is unchanged."""
    # probs=[sell=0.10, hold=0.00, buy=0.90] → BUY wins argmax
    # buy_prob=0.90, hold_prob=0.00
    # conf = min(0.90, 0.95) = 0.90; penalised = 0.90 * 1.0 = 0.90
    engine = _engine_with_mocked_models([0.10, 0.00, 0.90], [0.10, 0.00, 0.90])
    result = engine.predict(pd.DataFrame(), "TEST")
    assert result["action"] == "BUY"
    assert abs(result["confidence"] - 0.90) < 0.01, f"conf={result['confidence']}"


def test_moderate_hold_prob_reduces_confidence():
    """With hold_prob=0.30, confidence is reduced by 15%."""
    # probs=[sell=0.10, hold=0.30, buy=0.60] → BUY wins argmax
    # buy_prob=0.60, hold_prob=0.30
    # conf = min(0.60, 0.95) = 0.60; penalised = 0.60 * (1 - 0.15) = 0.60 * 0.85 = 0.51
    engine = _engine_with_mocked_models([0.10, 0.30, 0.60], [0.10, 0.30, 0.60])
    result = engine.predict(pd.DataFrame(), "TEST")
    assert result["action"] == "BUY"
    assert abs(result["confidence"] - 0.51) < 0.01, f"conf={result['confidence']}"


def test_high_hold_prob_significant_penalty():
    """With hold_prob=0.45, confidence is reduced by 22.5%."""
    # probs=[sell=0.05, hold=0.45, buy=0.50] → BUY wins argmax (0.50 > 0.45)
    # buy_prob=0.50, hold_prob=0.45
    # conf = min(0.50, 0.95) = 0.50; penalised = 0.50 * (1 - 0.225) = 0.50 * 0.775 = 0.3875
    engine = _engine_with_mocked_models([0.05, 0.45, 0.50], [0.05, 0.45, 0.50])
    result = engine.predict(pd.DataFrame(), "TEST")
    assert result["action"] == "BUY"
    assert abs(result["confidence"] - 0.3875) < 0.01, f"conf={result['confidence']}"


def test_confidence_floored_at_0_35():
    """Penalised confidence is floored at 0.35 even when arithmetic goes below."""
    # probs=[sell=0.05, hold=0.60, buy=0.35] → argmax=1 → HOLD action
    # → HOLD branch fires: buy_prob=0.35 < 0.60 and sell_prob=0.05 < 0.60
    # → action="HOLD", conf=max(0.35, 0.05)=0.35
    # hold_prob=0.60; penalised = 0.35 * (1 - 0.60*0.5) = 0.35 * 0.70 = 0.245 → floor 0.35
    engine = _engine_with_mocked_models([0.05, 0.60, 0.35], [0.05, 0.60, 0.35])
    result = engine.predict(pd.DataFrame(), "TEST")
    assert result["confidence"] == pytest.approx(0.35, abs=0.005)


def test_sell_action_also_penalised():
    """SELL confidence is penalised proportionally to hold_prob."""
    # probs=[sell=0.60, hold=0.20, buy=0.20] → SELL wins argmax
    # sell_prob=0.60, hold_prob=0.20
    # conf = min(0.60, 0.95) = 0.60; penalised = 0.60 * (1 - 0.10) = 0.60 * 0.90 = 0.54
    engine = _engine_with_mocked_models([0.60, 0.20, 0.20], [0.60, 0.20, 0.20])
    result = engine.predict(pd.DataFrame(), "TEST")
    assert result["action"] == "SELL"
    assert abs(result["confidence"] - 0.54) < 0.01, f"conf={result['confidence']}"


def test_penalty_increases_with_hold_prob():
    """Higher hold_prob always produces lower confidence (monotonic)."""
    low_hold  = _engine_with_mocked_models([0.10, 0.10, 0.80], [0.10, 0.10, 0.80])
    high_hold = _engine_with_mocked_models([0.10, 0.30, 0.60], [0.10, 0.30, 0.60])
    r_low  = low_hold.predict(pd.DataFrame(), "TEST")
    r_high = high_hold.predict(pd.DataFrame(), "TEST")
    assert r_low["confidence"] > r_high["confidence"], (
        f"low_hold conf={r_low['confidence']} should exceed high_hold conf={r_high['confidence']}"
    )
```

- [ ] **Step 2.2: Run tests — confirm they fail**

```bash
cd /root/cryptobot_v3 && venv/bin/pytest bot/tests/test_hold_prob_penalty.py -v --tb=short 2>&1 | tail -15
```

Expected: most tests `FAILED` (no hold_prob penalty in current code).

- [ ] **Step 2.3: Apply the fix to `AIStrategyEngine.predict()`**

In `bot/models/ai_strategy.py`, find the `predict()` method of `AIStrategyEngine` (~line 637).

Locate this block (lines ~645–661):
```python
        w_rf, w_lgbm = self._get_dynamic_weights()
        buy_prob  = rf_probs[2] * w_rf + lgbm_probs[2] * w_lgbm
        sell_prob = rf_probs[0] * w_rf + lgbm_probs[0] * w_lgbm

        if rf_p.get("action") == "HOLD" or lgbm_p.get("action") == "HOLD":
            if buy_prob < 0.60 and sell_prob < 0.60:
                action, conf = "HOLD", max(buy_prob, sell_prob)
            elif buy_prob >= sell_prob:
                action, conf = "BUY", min(buy_prob, 0.95)
            else:
                action, conf = "SELL", min(sell_prob, 0.95)
        elif buy_prob >= sell_prob:
            action, conf = "BUY", min(buy_prob, 0.95)
        else:
            action, conf = "SELL", min(sell_prob, 0.95)

        conf = round(float(conf), 4)
```

Replace with:
```python
        w_rf, w_lgbm = self._get_dynamic_weights()
        buy_prob  = rf_probs[2] * w_rf + lgbm_probs[2] * w_lgbm
        sell_prob = rf_probs[0] * w_rf + lgbm_probs[0] * w_lgbm
        hold_prob = rf_probs[1] * w_rf + lgbm_probs[1] * w_lgbm

        if rf_p.get("action") == "HOLD" or lgbm_p.get("action") == "HOLD":
            if buy_prob < 0.60 and sell_prob < 0.60:
                action, conf = "HOLD", max(buy_prob, sell_prob)
            elif buy_prob >= sell_prob:
                action, conf = "BUY", min(buy_prob, 0.95)
            else:
                action, conf = "SELL", min(sell_prob, 0.95)
        elif buy_prob >= sell_prob:
            action, conf = "BUY", min(buy_prob, 0.95)
        else:
            action, conf = "SELL", min(sell_prob, 0.95)

        conf = conf * (1.0 - hold_prob * 0.5)
        conf = max(0.35, min(0.95, conf))
        conf = round(float(conf), 4)
```

- [ ] **Step 2.4: Run tests — confirm all pass**

```bash
cd /root/cryptobot_v3 && venv/bin/pytest bot/tests/test_hold_prob_penalty.py -v --tb=short 2>&1 | tail -15
```

Expected: `5 passed`.

- [ ] **Step 2.5: Commit**

```bash
cd /root/cryptobot_v3 && git add bot/models/ai_strategy.py bot/tests/test_hold_prob_penalty.py
git commit -m "fix: penalise ML confidence by HOLD probability (up to 50% reduction)"
```

---

## Task 3: Risk Agent — 20EMA Momentum Gate (Gate 4c)

**Files:**
- Modify: `bot/engine/risk_agent.py` — `evaluate()` (~line 242)
- Create: `bot/tests/test_ema_momentum_gate.py`

- [ ] **Step 3.1: Write the failing tests**

Create `bot/tests/test_ema_momentum_gate.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
from engine.risk_agent import RiskDecisionAgent


def _make_df(rows=100, close_val=100.0):
    """DataFrame with constant close so 20EMA == close_val."""
    idx = pd.date_range(
        end=datetime.now(timezone.utc),
        periods=rows, freq="1h", tz="UTC"
    )
    close = pd.Series([close_val] * rows, index=idx)
    return pd.DataFrame({
        "open": close, "high": close * 1.001, "low": close * 0.999,
        "close": close, "volume": [1_000_000.0] * rows,
    })


def _make_agent():
    risk = MagicMock()
    risk.get_position_size.return_value = (0.1, 50.0)
    risk.can_open_trade.return_value = (True, "ok")
    gnn = MagicMock()
    gnn.check.return_value = (True, "ok", 0.3)
    return RiskDecisionAgent(risk, gnn)


def _make_ensemble(action="BUY", confidence=0.70):
    e = MagicMock()
    e.action = action
    e.confidence = confidence
    e.net_score = 0.40 if action == "BUY" else -0.40
    e.buy_score = 0.60 if action == "BUY" else 0.10
    e.sell_score = 0.10 if action == "BUY" else 0.60
    e.agents_agreeing = 3
    e.agents_total = 3
    e.signals = []
    return e


def _make_profile(name="BALANCED"):
    p = MagicMock()
    p.name = name
    p.min_confidence = 0.43
    p.min_agent_agreement = 2
    p.adx_min = 20.0
    p.net_score_threshold = 0.25
    p.htf_filter_mode = "soft"
    p.btc_momentum_filter = False
    p.use_confluence_scoring = False
    p.min_quality_score = 0.30
    return p


def _make_regime(adx=30.0, trend_dir="NEUTRAL"):
    return dict(
        adx=adx, regime="STRONG_TREND", gate=True,
        allow_longs=True, allow_shorts=True,
        vol_ratio=1.2, breadth=0.55, bear_breadth=0.45,
        min_conf=0.43, size_mult=1.0, hmm_regime="STRONG_TREND",
        trend_direction=trend_dir,
    )


def test_buy_blocked_when_price_below_ema():
    """BUY is rejected when live price is 2% below 20EMA."""
    agent = _make_agent()
    ema_val = 100.0
    live_price = 97.5  # 2.5% below → triggers gate (threshold is 1%)
    df = _make_df(close_val=ema_val)

    result = agent.evaluate(
        ensemble=_make_ensemble("BUY"),
        symbol="ETH/USDT",
        df_1h=df,
        profile=_make_profile(),
        regime_ctx=_make_regime(),
        btc_return=0.0,
        open_trades=[],
        balance=1000.0,
        get_price_fn=lambda sym: live_price,
        get_atr_fn=lambda sym: 1.0,
    )
    assert not result.approved
    assert "20EMA" in " ".join(result.reasons)


def test_buy_allowed_when_price_above_ema():
    """BUY is not blocked when live price is above 20EMA."""
    agent = _make_agent()
    ema_val = 100.0
    live_price = 101.5  # above EMA

    result = agent.evaluate(
        ensemble=_make_ensemble("BUY"),
        symbol="ETH/USDT",
        df_1h=_make_df(close_val=ema_val),
        profile=_make_profile(),
        regime_ctx=_make_regime(),
        btc_return=0.0,
        open_trades=[],
        balance=1000.0,
        get_price_fn=lambda sym: live_price,
        get_atr_fn=lambda sym: 1.0,
    )
    # Gate 4c should not fire — check reasons don't contain EMA rejection
    assert "20EMA" not in " ".join(result.reasons)


def test_sell_blocked_when_price_above_ema():
    """SELL is rejected when live price is 2% above 20EMA."""
    agent = _make_agent()
    ema_val = 100.0
    live_price = 102.5  # 2.5% above → triggers gate

    result = agent.evaluate(
        ensemble=_make_ensemble("SELL", confidence=0.70),
        symbol="ETH/USDT",
        df_1h=_make_df(close_val=ema_val),
        profile=_make_profile(),
        regime_ctx=_make_regime(),
        btc_return=0.0,
        open_trades=[],
        balance=1000.0,
        get_price_fn=lambda sym: live_price,
        get_atr_fn=lambda sym: 1.0,
    )
    assert not result.approved
    assert "20EMA" in " ".join(result.reasons)


def test_sell_allowed_when_price_below_ema():
    """SELL is not blocked when price is below 20EMA."""
    agent = _make_agent()
    live_price = 98.0  # below EMA

    result = agent.evaluate(
        ensemble=_make_ensemble("SELL", confidence=0.70),
        symbol="ETH/USDT",
        df_1h=_make_df(close_val=100.0),
        profile=_make_profile(),
        regime_ctx=_make_regime(),
        btc_return=0.0,
        open_trades=[],
        balance=1000.0,
        get_price_fn=lambda sym: live_price,
        get_atr_fn=lambda sym: 1.0,
    )
    assert "20EMA" not in " ".join(result.reasons)


def test_btc_exempt_from_ema_gate():
    """BTC/USDT is exempt from Gate 4c regardless of price vs EMA."""
    agent = _make_agent()
    live_price = 50_000 * 0.97  # 3% below EMA

    result = agent.evaluate(
        ensemble=_make_ensemble("BUY"),
        symbol="BTC/USDT",
        df_1h=_make_df(close_val=50_000.0),
        profile=_make_profile(),
        regime_ctx=_make_regime(),
        btc_return=0.0,
        open_trades=[],
        balance=1000.0,
        get_price_fn=lambda sym: live_price,
        get_atr_fn=lambda sym: 100.0,
    )
    assert "20EMA" not in " ".join(result.reasons)


def test_within_1pct_of_ema_not_blocked():
    """Price within 0.5% of EMA must not trigger the gate."""
    agent = _make_agent()
    live_price = 99.6  # 0.4% below — within 1% band

    result = agent.evaluate(
        ensemble=_make_ensemble("BUY"),
        symbol="ETH/USDT",
        df_1h=_make_df(close_val=100.0),
        profile=_make_profile(),
        regime_ctx=_make_regime(),
        btc_return=0.0,
        open_trades=[],
        balance=1000.0,
        get_price_fn=lambda sym: live_price,
        get_atr_fn=lambda sym: 1.0,
    )
    assert "20EMA" not in " ".join(result.reasons)
```

- [ ] **Step 3.2: Run tests — confirm they fail**

```bash
cd /root/cryptobot_v3 && venv/bin/pytest bot/tests/test_ema_momentum_gate.py -v --tb=short 2>&1 | tail -20
```

Expected: `test_buy_blocked_when_price_below_ema` and `test_sell_blocked_when_price_above_ema` FAIL (gate doesn't exist yet).

- [ ] **Step 3.3: Apply the fix to `bot/engine/risk_agent.py`**

Find the end of Gate 4b in `evaluate()`. It ends with this `elif` block (~line 232–242):

```python
            elif action == "BUY" and bear_breadth > 0.50:
                penalty = min(0.15, (bear_breadth - 0.50) * 1.5)
                penalised_floor = round(min(
                    getattr(profile, 'min_confidence', 0.45) + penalty,
                    0.80,
                ), 4)
                if conf < penalised_floor:
                    reasons.append(
                        f"long penalised: bearish breadth={bear_breadth:.0%} → need conf>={penalised_floor:.2f}"
                    )
                    return RiskDecision(False, reasons, conf, profile=profile.name, hmm_regime=hmm_regime)

        # ── Gate 5: Confidence (post-regime effective) ───────────────
```

Insert Gate 4c **between** the Gate 4b closing and the Gate 5 comment:

```python
            elif action == "BUY" and bear_breadth > 0.50:
                penalty = min(0.15, (bear_breadth - 0.50) * 1.5)
                penalised_floor = round(min(
                    getattr(profile, 'min_confidence', 0.45) + penalty,
                    0.80,
                ), 4)
                if conf < penalised_floor:
                    reasons.append(
                        f"long penalised: bearish breadth={bear_breadth:.0%} → need conf>={penalised_floor:.2f}"
                    )
                    return RiskDecision(False, reasons, conf, profile=profile.name, hmm_regime=hmm_regime)

        # ── Gate 4c: Momentum filter (price vs 20EMA) ───────────────────
        if symbol != "BTC/USDT":
            try:
                ema20 = df_1h["close"].ewm(span=20).mean().iloc[-1]
                price = get_price_fn(symbol)
                if price is not None and price > 0:
                    if action == "BUY" and price < ema20 * 0.99:
                        reasons.append(f"price {price:.4f} below 20EMA {ema20:.4f} (bearish)")
                        return RiskDecision(False, reasons, conf, profile=profile.name, hmm_regime=hmm_regime)
                    elif action == "SELL" and price > ema20 * 1.01:
                        reasons.append(f"price {price:.4f} above 20EMA {ema20:.4f} (bullish)")
                        return RiskDecision(False, reasons, conf, profile=profile.name, hmm_regime=hmm_regime)
            except Exception:
                pass

        # ── Gate 5: Confidence (post-regime effective) ───────────────
```

- [ ] **Step 3.4: Run tests — confirm all pass**

```bash
cd /root/cryptobot_v3 && venv/bin/pytest bot/tests/test_ema_momentum_gate.py -v --tb=short 2>&1 | tail -15
```

Expected: `6 passed`.

- [ ] **Step 3.5: Commit**

```bash
cd /root/cryptobot_v3 && git add bot/engine/risk_agent.py bot/tests/test_ema_momentum_gate.py
git commit -m "fix: add Gate 4c — 20EMA momentum filter blocks counter-trend entries"
```

---

## Task 4: Ensemble Directional Bias

**Files:**
- Modify: `bot/engine/ensemble.py` — `_aggregate()` (~line 103)
- Create: `bot/tests/test_ensemble_directional_bias.py`

- [ ] **Step 4.1: Write the failing tests**

Create `bot/tests/test_ensemble_directional_bias.py`:

```python
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock
from engine.ensemble import EnsembleEngine
from engine.smc_agent import AgentSignal


def _signal(agent, net):
    s = AgentSignal(agent=agent, buy_score=max(net, 0), sell_score=max(-net, 0),
                    net_score=net, confidence=0.65)
    return s


def _engine():
    smc  = MagicMock()
    tech = MagicMock()
    macro = MagicMock()
    return EnsembleEngine(smc, tech, macro)


def test_buy_net_reduced_in_bearish_trend():
    """Positive net_score is multiplied by 0.7 when trend_direction=BEARISH."""
    engine = _engine()
    signals = [
        _signal("smc",       0.50),
        _signal("technical", 0.60),
        _signal("macro_flow",0.40),
    ]
    result = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                               market_ctx={"trend_direction": "BEARISH", "vol_ratio": 1.0, "adx": 30.0},
                               regime="STRONG_TREND")
    # Without bias: net ≈ (0.50*0.28 + 0.60*0.50 + 0.40*0.22) / 1.0 = 0.14+0.30+0.088 / 1.0 = 0.528/1.0
    # With 0.7 bias applied to positive net in BEARISH trend: net reduced
    # Just verify net_score < what we'd get with NEUTRAL trend
    neutral = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                                market_ctx={"trend_direction": "NEUTRAL", "vol_ratio": 1.0, "adx": 30.0},
                                regime="STRONG_TREND")
    assert result.net_score < neutral.net_score, (
        f"bearish-trend net={result.net_score} should be < neutral net={neutral.net_score}"
    )


def test_sell_net_reduced_in_bullish_trend():
    """Negative net_score is multiplied by 0.7 when trend_direction=BULLISH."""
    engine = _engine()
    signals = [
        _signal("smc",       -0.50),
        _signal("technical", -0.60),
        _signal("macro_flow",-0.40),
    ]
    result = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                               market_ctx={"trend_direction": "BULLISH", "vol_ratio": 1.0, "adx": 30.0},
                               regime="STRONG_TREND")
    neutral = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                                market_ctx={"trend_direction": "NEUTRAL", "vol_ratio": 1.0, "adx": 30.0},
                                regime="STRONG_TREND")
    assert result.net_score > neutral.net_score, (
        f"bullish-trend net={result.net_score} should be > neutral net={neutral.net_score}"
    )


def test_neutral_trend_no_bias():
    """NEUTRAL trend direction applies no bias — same as absent key."""
    engine = _engine()
    signals = [_signal("smc", 0.50), _signal("technical", 0.60)]
    with_neutral = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                                     market_ctx={"trend_direction": "NEUTRAL", "vol_ratio": 1.0, "adx": 30.0},
                                     regime="RANGING")
    without_key = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                                    market_ctx={"vol_ratio": 1.0, "adx": 30.0},
                                    regime="RANGING")
    assert with_neutral.net_score == pytest.approx(without_key.net_score, abs=0.001)


def test_buy_in_bullish_trend_not_damped():
    """BUY net_score in a BULLISH trend should NOT be reduced."""
    engine = _engine()
    signals = [_signal("smc", 0.50), _signal("technical", 0.60)]
    bullish = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                                market_ctx={"trend_direction": "BULLISH", "vol_ratio": 1.0, "adx": 30.0},
                                regime="STRONG_TREND")
    neutral = engine._aggregate(signals, MagicMock(net_score_threshold=0.10),
                                market_ctx={"trend_direction": "NEUTRAL", "vol_ratio": 1.0, "adx": 30.0},
                                regime="STRONG_TREND")
    assert bullish.net_score == pytest.approx(neutral.net_score, abs=0.001)


def test_30pct_reduction_magnitude():
    """Verify the reduction factor is exactly 0.70 (30% reduction)."""
    engine = _engine()
    signals = [_signal("smc", 0.50), _signal("technical", 0.50)]
    bearish = engine._aggregate(signals, MagicMock(net_score_threshold=0.01),
                                market_ctx={"trend_direction": "BEARISH", "vol_ratio": 1.0, "adx": 30.0},
                                regime="RANGING")
    neutral = engine._aggregate(signals, MagicMock(net_score_threshold=0.01),
                                market_ctx={"trend_direction": "NEUTRAL", "vol_ratio": 1.0, "adx": 30.0},
                                regime="RANGING")
    if neutral.net_score > 0:
        ratio = bearish.net_score / neutral.net_score
        assert abs(ratio - 0.70) < 0.05, f"Expected ~0.70 reduction, got ratio={ratio:.3f}"
```

- [ ] **Step 4.2: Run tests — confirm they fail**

```bash
cd /root/cryptobot_v3 && venv/bin/pytest bot/tests/test_ensemble_directional_bias.py -v --tb=short 2>&1 | tail -15
```

Expected: `test_buy_net_reduced_in_bearish_trend` and `test_sell_net_reduced_in_bullish_trend` FAIL.

- [ ] **Step 4.3: Apply the fix to `bot/engine/ensemble.py`**

In `_aggregate()`, find this block (~lines 102–108):

```python
        if total_w > 0:
            net        = net / total_w
            buy_score  = buy_score / total_w
            sell_score = sell_score / total_w

        # ── Hard block: volume so thin price discovery is unreliable ────────────
        ctx = market_ctx or {}
```

Replace with:

```python
        if total_w > 0:
            net        = net / total_w
            buy_score  = buy_score / total_w
            sell_score = sell_score / total_w

        # Directional bias: reduce net_score when fighting the trend
        _bias_ctx = market_ctx or {}
        trend_dir = _bias_ctx.get("trend_direction", "NEUTRAL")
        if trend_dir == "BEARISH" and net > 0:
            net = net * 0.7
            log.debug(f"Ensemble: bearish trend → BUY net reduced to {net:.3f}")
        elif trend_dir == "BULLISH" and net < 0:
            net = net * 0.7
            log.debug(f"Ensemble: bullish trend → SELL net reduced to {net:.3f}")

        # ── Hard block: volume so thin price discovery is unreliable ────────────
        ctx = market_ctx or {}
```

- [ ] **Step 4.4: Run tests — confirm all pass**

```bash
cd /root/cryptobot_v3 && venv/bin/pytest bot/tests/test_ensemble_directional_bias.py -v --tb=short 2>&1 | tail -15
```

Expected: `5 passed`.

- [ ] **Step 4.5: Commit**

```bash
cd /root/cryptobot_v3 && git add bot/engine/ensemble.py bot/tests/test_ensemble_directional_bias.py
git commit -m "fix: ensemble directional bias — damp counter-trend net_score by 30%"
```

---

## Task 5: Full Regression Check

- [ ] **Step 5.1: Run entire test suite**

```bash
cd /root/cryptobot_v3 && venv/bin/pytest bot/tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all previously passing tests still pass, plus 4 new test files all green.

- [ ] **Step 5.2: Smoke-check signal debug log**

```bash
screen -r cryptobot_v3_spot  # or futures
# Attach and watch one cycle, then Ctrl+A D to detach
# OR inspect signal_debug.log for recent entries
tail -50 /root/cryptobot_v3/signal_debug.log | grep -E "sweep|BOS|20EMA|bearish trend|SELL"
```

Verify:
- Any bearish sweep in a bearish STRONG_TREND logs `sweep-` with score contribution (not suppressed to 0)
- Gate 4c rejection appears as `price X below 20EMA Y (bearish)` when triggered
- Ensemble debug shows `bearish trend → BUY net reduced`

---

## Task 6 (Deferred): Lower min_confidence

**Condition:** Defer until Sessions 1–4 have been live for ≥48h and SELL trade frequency has been observed.

**Files:** `config_spot.yaml`, `config_futures.yaml`

- [ ] **Step 6.1: Apply config change**

In both files, find and change:
```yaml
min_confidence: 0.43
```
to:
```yaml
min_confidence: 0.40
```

- [ ] **Step 6.2: Commit**

```bash
cd /root/cryptobot_v3 && git add config_spot.yaml config_futures.yaml
git commit -m "tune: lower min_confidence 0.43 → 0.40 after signal quality fixes"
```
