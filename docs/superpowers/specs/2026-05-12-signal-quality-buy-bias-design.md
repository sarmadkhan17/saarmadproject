# Signal Quality Fixes: BUY Bias in Bearish Strong Trends

**Date:** 2026-05-12
**Status:** Approved
**Branch:** checkpoint/signal-quality-fixes

---

## Problem Summary

The bot has a persistent BUY bias in bearish strong-trend markets. Four confirmed root causes:

1. **SMC agent ignores trend direction** — bearish SMC signals (sweep, BOS) get partial credit (0.15) in *any* trending regime, including bearish ones. Bearish signals should fire at full strength when the trend is bearish.
2. **ML confidence is not penalised for model uncertainty** — when the RF+LGBM ensemble is internally torn (high HOLD probability), `conf` is still reported at the raw BUY/SELL probability, masking genuine uncertainty.
3. **No price-vs-EMA momentum filter** — the risk gate has no check that the current price supports the intended trade direction.
4. **Ensemble net_score has no directional damping** — a counter-trend BUY in a bearish regime gets the same net_score as an aligned BUY.

**Constraint:** The data pipeline produces >70% HOLD labels. The ML model is intentionally forced to return BUY or SELL (not HOLD). We keep this behaviour — we only penalise confidence using the raw HOLD probability.

---

## Session 1 — SMC Agent: Direction-Aware Trend Filter

**File:** `bot/engine/smc_agent.py`
**Lines affected:** 53–90

### Change

Replace the undirected `_trending` boolean with two directional flags:

```python
_regime_str    = ((market_ctx or {}).get("regime") or "").upper()
_trend_dir     = ((market_ctx or {}).get("trend_direction") or "").upper()
_is_bull_trend = "TREND" in _regime_str and _trend_dir == "BULLISH"
_is_bear_trend = "TREND" in _regime_str and _trend_dir == "BEARISH"
```

Update Sweep scoring (was `0.15 if _trending else 0.35`):
```python
if sweep.get("direction") == "bullish" and not _is_bear_trend:
    buy_score += 0.35
elif sweep.get("direction") == "bearish" and not _is_bull_trend:
    sell_score += 0.35
```

Update BOS scoring (was `0.15 if _trending else 0.30`):
```python
if bos.get("direction") == "bullish" and not _is_bear_trend:
    buy_score += 0.30
elif bos.get("direction") == "bearish" and not _is_bull_trend:
    sell_score += 0.30
```

### Behaviour delta

| Scenario | Before | After |
|---|---|---|
| Bearish sweep in STRONG_TREND (bearish) | 0.15 | 0.35 (full credit) |
| Bearish sweep in STRONG_TREND (bullish) | 0.15 | 0.00 (suppressed) |
| Bearish sweep, not trending | 0.35 | 0.35 (unchanged) |

`trend_direction` is set by `bot/risk/manager.py` and is always present in `regime_ctx` / `market_ctx`.

---

## Session 2 — ML Confidence Penalty via HOLD Probability

**File:** `bot/models/ai_strategy.py`
**Method:** `AIStrategyEngine.predict()` (line 637)

### Change

After the `action / conf` assignment block, insert before `conf = round(float(conf), 4)`:

```python
hold_prob = rf_probs[1] * w_rf + lgbm_probs[1] * w_lgbm
conf = conf * (1.0 - hold_prob * 0.5)
conf = max(0.35, min(0.95, conf))
```

`rf_probs`, `lgbm_probs`, `w_rf`, `w_lgbm` are all already in scope at this point.

### Penalty table

| hold_prob | Penalty factor | Example: conf 0.60 → |
|---|---|---|
| 0.30 | ×0.85 | 0.51 |
| 0.50 | ×0.75 | 0.45 |
| 0.70 | ×0.65 | 0.39 (floored to 0.35) |

---

## Session 3 — 20EMA Momentum Gate in Risk Agent

**File:** `bot/engine/risk_agent.py`
**Method:** `evaluate()`
**Insert after:** Gate 4b block ends (~line 242), before Gate 5 comment (~line 244)

### Change

```python
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
```

Fires in **all** regimes (not limited to trending). BTC/USDT is excluded as its own market.
Threshold: 1% deviation — tight enough to block clear counter-momentum, loose enough not to block near-EMA setups.

---

## Session 4 — Ensemble Directional Bias

**File:** `bot/engine/ensemble.py`
**Method:** `_aggregate()`
**Insert after:** `net = net / total_w` block (~line 103), before the `ctx = market_ctx or {}` line (~line 108)

### Change

```python
# Directional bias: reduce net_score when fighting the trend
_bias_ctx = market_ctx or {}
trend_dir = _bias_ctx.get("trend_direction", "NEUTRAL")
if trend_dir == "BEARISH" and net > 0:
    net = net * 0.7
    log.debug(f"Ensemble: bearish trend → BUY net reduced to {net:.3f}")
elif trend_dir == "BULLISH" and net < 0:
    net = net * 0.7
    log.debug(f"Ensemble: bullish trend → SELL net reduced to {net:.3f}")
```

Uses `_bias_ctx` (not `ctx`) to avoid shadowing the `ctx` assigned at line 108.

### Effect

Counter-trend net_score is reduced by 30%. This raises the effective bar for the signal to cross the `threshold` check at lines 128–133, without hard-blocking. Combined with the SMC fix (Session 1), double-counted suppression is avoided because the SMC fix operates on raw agent scores, while this operates on the aggregated weighted net.

---

## Session 5 (Optional) — Lower min_confidence

**Files:** `config_spot.yaml`, `config_futures.yaml`
**Change:** `min_confidence: 0.43 → 0.40`
**Defer until:** Sessions 1–4 are live and the effect on SELL trade frequency is observed (minimum 2–3 days of live data).

---

## Architecture Notes

- All four fixes are additive and independent. They can be applied in any order.
- Sessions 1 and 4 both damp counter-trend signals but at different layers (agent score vs ensemble net), so they are not redundant — each operates on a different variable.
- Session 2 reduces ML confidence independently of Sessions 1/3/4. It will compound with the agent-agreement penalty already in `_aggregate()` (variance-weighted consensus factor).
- Session 3 is the only hard-block gate — it returns `RiskDecision(False)` rather than penalising. It fires late (Gate 4c), after regime and breadth gates have already passed.

---

## Testing Checklist

- [ ] Signal debug log: verify bearish sweep/BOS score is 0.35 (not 0.15) in STRONG_TREND + BEARISH direction
- [ ] Signal debug log: verify HOLD-probability penalty reduces conf when hold_prob > 0.40
- [ ] Risk gate log: verify Gate 4c fires on BUY signals when price < ema20 * 0.99
- [ ] Ensemble log: verify bearish-trend BUY net_score is reduced by 30% before threshold compare
- [ ] Run for 48h in spot mode; compare SELL trade count before/after
