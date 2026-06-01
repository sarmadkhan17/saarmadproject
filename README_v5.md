# CryptoBot v5 — Changes from v3

## What's removed (the ML stack that failed twice)

- HMM regime model — was always 3+ bars late
- LightGBM + Random Forest direction predictors — overfit on noisy crypto data
- DQN / RL position manager — never converged
- AIStrategyEngine, RegimeFusion, RegimeOverlay, RegimeLabeler, GNNCorrelationFilter
- SelfLearner, TrainingFeed, FeatureBuilder, feature_validation
- All retrain pipelines, model files (.pkl, .pt), feature metadata

## What's new

- **MacroContextAgent** — CoinGecko global API, BTC.D + USDT.D rate-of-change. Kills entries when USDT.D rises >0.4%/hr or BTC.D spikes >0.8%/hr. Selects coin universe based on dominance.
- **MicrostructureAgent** — Order book bid/ask imbalance + CVD divergence + funding rate. Confirms or kills the structural bias.
- **DeepSeek reasoning layer** —
  - **Actor** (V3 / `deepseek-chat`) — every cycle, decides whether a setup is worth taking. Reads similar past trades as RAG context.
  - **Judge** (R1 / `deepseek-reasoner`) — after every closed trade, writes a verbal critique. No hindsight allowed.
  - **Meta-Judge** (R1) — every 20 trades, synthesizes rules from accumulated critiques. Updates rules injected into Actor's prompt.
- **TradeMemory** — SQLite store of all closed trades + entry context + Judge critiques. Powers RAG for the Actor.
- **TechnicalAgent** — pure rule-based RSI/MACD/EMA/momentum confluence. Replaces the ML predictor in the ensemble's "technical" slot.

## What stays

Everything that worked in v3: SMCAgent (BOS/CHoCH/FVG), MarketRegimeGate (ADX+breadth), KellyCriterionSizer, ExitEngine (TP1 partial, breakeven, swing trail), CircuitBreaker, CorrelationFilter, ExecutionEngine, StateManager, BinanceWSPriceFeed, CoinScanner, full TelegramNotifier (with all commands), full Dashboard (with all panels).

## Bugs fixed before any new code

1. **`ensemble.py` double trend application** — old code applied `net*0.05` then `net*0.7` = `net*0.035` to counter-trend signals, causing near-permanent HOLD. Now single 30% reduction.
2. **`config_futures.yaml` min_confidence** — was 0.38 (below AGGRESSIVE profile's 0.42 default, removing all filtering). Now 0.45.
3. **`telegram.py` Groq references** — swapped for DeepSeek cost tracking. Health check updated.

## Decision flow

```
Gate 0: macro kill switch       (USDT.D spike, BTC.D spike, drawdown)
Gate 1: macro universe          (BTC.D direction → BTC-only / large-cap / full)
Gate 2: regime gate             (ADX + breadth — CHOPPY = no trade)
Gate 3: ensemble structure      (SMC + Technical + MacroFlow vote)
Gate 4: microstructure          (CVD + order book confirm or kill)
Gate 5: DeepSeek Actor          (LLM endorses with similar-trades RAG)
Gate 6: risk + sizing           (Kelly + ATR + structure-first SL)
        ↓
        Execute on Binance Demo
        ↓
        On close → Judge (R1)
        Every 20 closes → Meta-Judge (R1) → update rules
```

## Deploy

```bash
tar -xzf cryptobot_v5.tar.gz
cd cryptobot_v5
pip install -r requirements.txt

# Run futures bot (recommended for paper trading)
cd bot && BOT_MODE=futures python3 launcher.py

# Or spot
cd bot && BOT_MODE=spot python3 launcher.py

# Dashboard (separate process)
cd dashboard && python3 server.py
# Visit http://localhost:5002
```

## Cost estimate

DeepSeek pricing as of 2026:
- V3 (Actor): ~$0.50/day at active scanning
- R1 (Judge): ~$0.05/day at one judge per closed trade
- R1 (Meta-Judge): ~$0.02 per weekly synthesis

**Total: under $1/day.**

## Paper trade before live capital

- Minimum 2 weeks on Binance Demo
- Minimum 50 closed trades
- Review every Judge critique — these are the real learning signal
- Check Meta-Judge output after the first 20 trades — should produce specific, actionable rules
- Only go live after the Judge consistently rates decision quality "good" or "acceptable" on >70% of trades
