"""
Base Bot - Shared Core Strategy
Both SpotBot and FuturesBot inherit from this.
100% identical strategy — only order direction differs.

Shared:
- ML models (RF + LightGBM + LSTM)
- Agent system (Groq confidence gate)
- Risk management (ATR stops, Kelly, portfolio heat)
- Data feed (validation + caching)
- Coin scanner (autonomous selection)
- Self learner (improves over time)
- Telegram notifier (alerts + commands)
"""

import json
import os
import time
import threading
import logging
import logging.handlers
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys

import numpy as np
import yaml
import dataclasses
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent))

from core.config   import get_exchange_config, DATA_DIR, LOGS_DIR, BOT_ROOT
from data.feed    import DataFeed, TrainingFeed, TrainingDataStore
from data.ws_feed import BinanceWSPriceFeed
from models.ai_strategy  import AIStrategyEngine
from models.hmm import HMMRegimeModel
from models.rl_agent    import RLTradeManager
from models.gnn   import GNNCorrelationFilter
from agents.coordinator       import AgentCoordinator
from risk.manager import RiskManager
from tuning.learner import SelfLearner
from notify.telegram     import TelegramNotifier
from tuning.scanner import CoinScanner

# v5: New 4-layer architecture
from engine.profiles       import TradingProfile
from engine.smc_agent      import SMCAgent
from engine.ensemble       import EnsembleEngine
from engine.risk_agent     import RiskDecisionAgent
from engine.execution_engine import ExecutionEngine

DATA = DATA_DIR


def _compute_class_weights(closed_trades: list) -> dict:
    """
    Precision-adaptive class weights from recent closed trades.
    Returns {SELL(0): w, HOLD(1): 1.0, BUY(2): w}.
    Falls back to {0: 2.0, 1: 1.0, 2: 2.0} if fewer than 20 trades per side.
    """
    import numpy as np
    BUY_SIDES  = {"buy", "long"}
    SELL_SIDES = {"sell", "short"}
    FALLBACK   = {0: 2.0, 1: 1.0, 2: 2.0}

    buy_trades  = [t for t in closed_trades if t.get("side", "") in BUY_SIDES]
    sell_trades = [t for t in closed_trades if t.get("side", "") in SELL_SIDES]

    if len(buy_trades) < 20 or len(sell_trades) < 20:
        return FALLBACK

    buy_prec  = sum(1 for t in buy_trades  if t.get("pnl", 0) > 0) / len(buy_trades)
    sell_prec = sum(1 for t in sell_trades if t.get("pnl", 0) > 0) / len(sell_trades)

    buy_w  = float(np.clip(1.0 / (buy_prec  + 1e-9), 1.0, 4.0))
    sell_w = float(np.clip(1.0 / (sell_prec + 1e-9), 1.0, 4.0))

    return {0: sell_w, 1: 1.0, 2: buy_w}


def setup_logging(config: dict, log_file: str = "bot.log"):
    """Setup rotating log handler. Same for all bot modes."""
    log_cfg  = config.get("logging", {})
    log_path = LOGS_DIR / log_file
    log_path.parent.mkdir(exist_ok=True)
    handler  = logging.handlers.RotatingFileHandler(
        str(log_path),
        maxBytes    = log_cfg.get("max_bytes", 10_485_760),
        backupCount = log_cfg.get("backup_count", 5),
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))
    logging.basicConfig(
        level    = logging.INFO,
        format   = "%(asctime)s [%(levelname)s] %(message)s",
        handlers = [handler],
        force    = True,
    )


@dataclass
class Trade:
    id: str
    symbol: str
    side: str           # "buy"/"sell" for spot | "long"/"short" for futures
    amount: float
    price: float
    timestamp: str
    strategy: str
    timeframe: str
    status: str
    mode: str = "spot"  # "spot" or "futures"
    leverage: int = 1
    pnl: float = 0.0
    close_price: float = 0.0
    close_timestamp: str = ""
    sl_order_id: str = ""


class StateManager:
    """
    Manages trade state.
    Spot and Futures use separate state files so they don't interfere.
    Batches writes to reduce I/O — flushes every 10s or on demand.
    """

    _FLUSH_INTERVAL = 10  # seconds

    def __init__(self, filename: str = "state.json"):
        self.path = DATA / filename
        DATA.mkdir(exist_ok=True)
        self._lock = threading.Lock()
        self._dirty = False
        self._last_flush = time.time()
        self._flush_thread = None
        self._stop_flushing = threading.Event()
        self.state = self._load()
        self._start_flush_thread()

    def _start_flush_thread(self):
        self._flush_thread = threading.Thread(target=self._periodic_flush, daemon=True)
        self._flush_thread.start()

    def _periodic_flush(self):
        while not self._stop_flushing.wait(self._FLUSH_INTERVAL):
            with self._lock:
                if self._dirty:
                    self._dirty = False
                    self._do_save_locked()

    def shutdown(self):
        self._stop_flushing.set()
        with self._lock:
            if self._dirty:
                self._dirty = False
                self._do_save_locked()
        if self._flush_thread:
            self._flush_thread.join(timeout=5)

    def _load(self):
        with self._lock:
            if self.path.exists():
                with open(self.path) as f:
                    return json.load(f)
            return {
                "trades": [], "signals": [],
                "stats": {
                    "total_trades": 0, "wins": 0, "losses": 0,
                    "total_pnl": 0.0,
                    "start_time": datetime.now(timezone.utc).isoformat(),
                },
            }

    def save(self, immediate: bool = False):
        """Mark state dirty for async flush. Set immediate=True to force write now."""
        with self._lock:
            self._dirty = True
            if immediate:
                self._dirty = False
                self._do_save_locked()

    def _do_save_locked(self):
        """Write state to disk. Caller must hold self._lock."""
        archive_path = self.path.with_name(self.path.stem + "_archive.json")
        cutoff = datetime.now(timezone.utc) - timedelta(days=3)
        to_archive = []
        to_keep = []
        for t in self.state.get("trades", []):
            if t.get("status") == "closed" and t.get("close_timestamp"):
                try:
                    ct = datetime.fromisoformat(t["close_timestamp"])
                    if ct.tzinfo is None:
                        ct = ct.replace(tzinfo=timezone.utc)
                    if ct < cutoff:
                        to_archive.append(t)
                        continue
                except (ValueError, TypeError):
                    pass
            to_keep.append(t)
        if to_archive:
            existing = []
            if archive_path.exists():
                try:
                    with open(archive_path) as f:
                        existing = json.load(f)
                except Exception:
                    pass
            arc_tmp = archive_path.with_suffix(".tmp.json")
            with open(arc_tmp, "w") as f:
                json.dump(existing + to_archive, f, indent=2, default=str)
            arc_tmp.replace(archive_path)
            self.state["trades"] = to_keep
        tmp_path = self.path.with_suffix(".tmp.json")
        with open(tmp_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)
        tmp_path.replace(self.path)
        try:
            shutil.copy2(str(self.path),
                         str(self.path.with_suffix(".backup.json")))
        except Exception:
            pass
        self._last_flush = time.time()

    def add_trade(self, trade: Trade):
        self.state["trades"].append(asdict(trade))
        self.state["stats"]["total_trades"] += 1
        self.save()

    def close_trade(self, trade_id, price, pnl):
        for t in self.state["trades"]:
            if t["id"] == trade_id:
                t["status"]          = "closed"
                t["close_price"]     = price
                t["pnl"]             = round(t.get("pnl", 0.0) + pnl, 8)
                t["close_timestamp"] = datetime.now(timezone.utc).isoformat()
                self.state["stats"]["total_pnl"] += pnl
                if t["pnl"] > 0: self.state["stats"]["wins"]   += 1
                else:             self.state["stats"]["losses"] += 1
                break
        self.save()

    def partial_close_trade(self, trade_id, closed_amount, pnl):
        for t in self.state["trades"]:
            if t["id"] == trade_id:
                t["amount"] = max(0.0, round(t["amount"] - closed_amount, 8))
                t["pnl"]    = round(t.get("pnl", 0.0) + pnl, 8)
                self.state["stats"]["total_pnl"] += pnl
                if pnl > 0:
                    self.state["stats"]["wins"] += 1
                else:
                    self.state["stats"]["losses"] += 1
                break
        self.save()

    def update_trade_amount(self, trade_id, new_amount):
        for t in self.state["trades"]:
            if t["id"] == trade_id:
                t["amount"] = round(new_amount, 8)
                break
        self.save()

    def update_trade_sl(self, trade_id, sl_order_id):
        for t in self.state["trades"]:
            if t["id"] == trade_id:
                t["sl_order_id"] = sl_order_id
                break
        self.save()

    def add_signal(self, signal):
        self.state["signals"].append(signal)
        self.state["signals"] = self.state["signals"][-500:]
        self.save()

    def get_open_trades(self):
        return [t for t in self.state["trades"] if t["status"] == "open"]

    def get_all_trades(self):
        trades = list(self.state["trades"])
        archive_path = self.path.with_name(self.path.stem + "_archive.json")
        if archive_path.exists():
            try:
                with open(archive_path) as f:
                    trades.extend(json.load(f))
            except Exception:
                pass
        return trades


class BaseBot:
    """
    Core trading logic shared by SpotBot and FuturesBot.
    Subclasses only override:
    - _setup_exchange()   → spot vs futures connection
    - _place_buy()        → market buy vs open long
    - _place_sell()       → market sell vs open short
    - _place_close()      → spot sell vs reduce-only close
    - _calc_pnl()         → simple pnl vs leveraged pnl
    """

    MODE = "spot"   # Override in subclass

    def __init__(self, config_file: str = "config.yaml", log_file: str = "bot.log"):
        self.config_file = config_file
        cfg_path = BOT_ROOT / config_file
        with open(cfg_path) as f:
            self.config = yaml.safe_load(f)

        setup_logging(self.config, log_file)
        self.log = logging.getLogger(self.__class__.__name__)

        # Exchange setup (overridden by subclass)
        self.exchange = self._setup_exchange()

        # Settings
        risk               = self.config.get("risk", {})
        self.scan_interval = self.config.get("bot", {}).get("scan_interval_seconds", 30)
        self.max_open      = risk.get("max_open_trades", 8)
        self.min_conf      = self.config.get("strategy", {}).get("min_confidence", 0.52)
        self.htf_filter_mode = self.config.get("strategy", {}).get("htf_filter_mode", "strict")
        self._training_in_progress = False   # set True during background retrain

        # State — separate files for spot and futures
        state_file    = "state.json" if self.MODE == "spot" else "futures_state.json"
        self.state    = StateManager(state_file)

        # WebSocket price feed (lower latency than REST polling)
        self.ws_feed  = BinanceWSPriceFeed()
        self.ws_feed.start()

        # Shared core systems — IDENTICAL for both modes
        self.feed       = DataFeed(self.exchange, ws_feed=self.ws_feed)
        self.ai         = AIStrategyEngine()
        self.risk       = RiskManager(self.config)
        self.agents     = AgentCoordinator()
        self.learner    = SelfLearner()
        self.notifier   = TelegramNotifier()
        self.scanner    = CoinScanner(self.config)
        self.hmm_regime = HMMRegimeModel()
        self.rl_agent   = RLTradeManager()
        self.gnn_filter = GNNCorrelationFilter()
        self.training_feed = TrainingFeed()  # v4: real Binance data for training

        self._balance_cache: Optional[float] = None
        self._symbol_loss_cooldown: dict = {}  # symbol → datetime of last loss
        self._symbol_locks: dict = {}           # per-symbol order placement locks
        self._symbol_locks_guard = threading.Lock()

        # ── v5: Four-layer architecture ──────────────────────────────────────
        self.profile = TradingProfile.from_config(self.config)
        self.log.info(f"Trading profile: {self.profile.name} | "
                      f"min_conf={self.profile.min_confidence} | "
                      f"agents={self.profile.min_agent_agreement}/3 | "
                      f"net_threshold={self.profile.net_score_threshold}")

        self.smc_agent   = SMCAgent()
        from engine.smc_agent import AgentSignal
        class MLTechnicalAgent:
            """Fail-fast ML agent: features set via set_features() before analyze().
            Calls build_features → validate → scaler.transform → predict_numpy.
            No fallbacks. FeatureValidationError propagates — no trade for that symbol."""
            def __init__(self, ai, mode_dir):
                self.ai = ai
                self.mode_dir = mode_dir
                self._scaler = None
                self._feature_cols = None
                self._last_X = None  # pre-scaled numpy array, 1 row × N cols
                self._last_symbol = "?"

            def _load_metadata(self):
                if self._scaler is not None:
                    return
                from features.feature_builder import load_feature_metadata
                self._scaler, self._feature_cols = load_feature_metadata(self.mode_dir)

            def set_features(self, symbol: str, feature_row):
                """Compute and validate features for one symbol. Stores pre-scaled X for analyze()."""
                self._load_metadata()
                from features.feature_builder import validate_features, FeatureValidationError
                validate_features(feature_row, self._feature_cols, self._scaler, symbol)
                self._last_X = self._scaler.transform(feature_row.values)
                self._last_symbol = symbol

            def analyze(self, df, profile):
                import numpy as np
                if self._last_X is None:
                    raise RuntimeError("MLTechnicalAgent: set_features() must be called before analyze()")
                try:
                    ml = self.ai.predict_numpy(self._last_X, self._last_symbol)
                finally:
                    self._last_X = None
                ml_action = ml.get("action", "BUY")
                ml_conf   = ml.get("confidence", 0.50)
                buy_score  = ml_conf if ml_action == "BUY" else 0.0
                sell_score = ml_conf if ml_action == "SELL" else 0.0
                net_score  = buy_score - sell_score
                return AgentSignal(
                    agent="technical", buy_score=buy_score, sell_score=sell_score,
                    net_score=net_score, confidence=ml_conf,
                    reasoning=f"ML:{ml_action} prob={ml_conf:.2f}",
                )

        self.ml_agent = MLTechnicalAgent(self.ai, DATA_DIR / self.MODE)
        self.ensemble = EnsembleEngine(
            self.smc_agent, self.ml_agent, None  # Macro/Flow deferred
        )
        self.risk_agent  = RiskDecisionAgent(self.risk, self.gnn_filter, self.hmm_regime)
        self.execution   = ExecutionEngine(
            self.exchange, self.state, self.notifier, self.MODE,
            get_leverage_fn=self._get_leverage,
        )

        self._with_training_heartbeat(lambda: self._train(quick=True), source="startup")

    def _setup_exchange(self):
        """Override in subclass to set spot or futures."""
        raise NotImplementedError

    def _place_buy(self, symbol, amount) -> Optional[dict]:
        """Override in subclass. Spot=market buy, Futures=open long."""
        raise NotImplementedError

    def _place_sell(self, symbol, amount) -> Optional[dict]:
        """Override in subclass. Spot=market sell, Futures=open short."""
        raise NotImplementedError

    def _place_close(self, symbol, amount, side) -> Optional[dict]:
        """Override in subclass. Closes an open position."""
        raise NotImplementedError

    def _calc_pnl(self, trade, close_price) -> float:
        """Override in subclass. Spot=simple, Futures=leveraged."""
        raise NotImplementedError

    def _post_scan(self, symbols):
        """Override in subclass. Called after scanner returns new watchlist."""

    def _reload_config(self):
        """Reload strategy config from YAML. Called before each retrain so dashboard changes apply."""
        try:
            cfg_path = BOT_ROOT / self.config_file
            if cfg_path.exists():
                with open(cfg_path) as f:
                    self.config = yaml.safe_load(f)
                risk = self.config.get("risk", {})
                strat = self.config.get("strategy", {})
                self.min_conf = strat.get("min_confidence", self.min_conf)
                self.htf_filter_mode = strat.get("htf_filter_mode", self.htf_filter_mode)
                self.scan_interval = self.config.get("bot", {}).get("scan_interval_seconds", self.scan_interval)
                self.max_open = risk.get("max_open_trades", self.max_open)
                new_profile_name = strat.get("trading_profile", self.profile.name)
                if new_profile_name != self.profile.name:
                    from engine.profiles import TradingProfile
                    self.profile = dataclasses.replace(TradingProfile.load(new_profile_name))
                    self.log.info(f"Profile switched → {self.profile.name}")
                self.log.info(f"Config reloaded: min_conf={self.min_conf}, htf={self.htf_filter_mode}, max_open={self.max_open}, profile={self.profile.name}")
        except Exception as e:
            self.log.warning(f"Config reload failed: {e}")

    def _maybe_swap_profile(self):
        """Hot-swap trading profile if the config YAML changed without a full retrain."""
        try:
            cfg_path = BOT_ROOT / self.config_file
            if not cfg_path.exists():
                return
            with open(cfg_path) as f:
                raw = yaml.safe_load(f) or {}
            new_name = raw.get("strategy", {}).get("trading_profile", self.profile.name)
            if new_name != self.profile.name:
                from engine.profiles import TradingProfile
                self.profile = dataclasses.replace(TradingProfile.load(new_name))
                self.log.info(f"Profile hot-swapped → {self.profile.name}")
        except Exception as e:
            self.log.warning(f"Profile hot-swap check failed: {e}")

    def _train_with_pipeline(self, quick: bool = False, force: bool = False):
        """v5 training: build multi-TF dataset via build_features(), save FeatureScaler
        for predict-time use. Single feature function for both train and predict.
        quick=True + force=False skips if models already exist."""
        # Quick skip: check cached models first
        if quick and not force:
            model_dir = DATA_DIR / self.MODE
            if (model_dir / "rf_model.pkl").exists() and (model_dir / "lgbm_model.pkl").exists():
                self.log.info(f"Quick pipeline retrain SKIPPED — models already exist for {self.MODE}")
                return
        import pandas as pd
        from features.pipeline import build_training_dataset, load_dataset
        from features.feature_builder import build_features, save_feature_metadata
        from models.ai_strategy import make_labels

        training_cfg = self.config.get("training", {})
        ml_cfg       = self.config.get("ml", {})
        tf           = ml_cfg.get("timeframe", "15m")
        fb           = ml_cfg.get("forward_bars", 2)
        atr_k        = training_cfg.get("atr_k", 0.5)
        mc           = self.config.get("strategy", {}).get("min_confidence", 0.52)
        mv           = self.config.get("strategy", {}).get("min_votes", 1)
        n_jobs       = training_cfg.get("n_jobs", 4)
        train_tfs    = training_cfg.get("timeframes", ["1h", "4h", "1d"])

        # Build pipeline dataset (for stats + caching only — not used for training features)
        symbols = training_cfg.get("symbols", [])
        if self.scanner.top_coins:
            extra = [c for c in self.scanner.top_coins[:training_cfg.get("top_n", 10)] if c not in symbols]
            symbols = symbols + extra

        dataset_stats = build_training_dataset(
            self.training_feed, self.config, symbols=symbols,
        )
        if "error" in dataset_stats:
            raise RuntimeError(f"Dataset build failed: {dataset_stats}")

        self.log.info(
            f"v5 dataset: {dataset_stats['n_rows']} rows, "
            f"{dataset_stats['n_features']} features, "
            f"{dataset_stats['n_symbols']} symbols, "
            f"{dataset_stats['build_time_sec']}s build"
        )

        # ── SINGLE feature extraction: build_features() for ALL symbols ──
        primary_tf = training_cfg.get("primary_timeframe", "1h")
        self.log.info(f"Building per-symbol multi-TF features ({train_tfs}) using parallel workers...")

        def _build_symbol_features(sym_idx):
            sym = dataset_stats["symbols"][sym_idx]
            try:
                sym_api = sym.replace("_", "/", 1) if "_" in sym else sym
                dfs = {}
                for t in train_tfs:
                    df = self.training_feed.fetch_ohlcv(sym_api, t, limit=0)
                    if df is not None and len(df) >= 100:
                        dfs[t] = df
                if len(dfs) < 2:
                    self.log.warning(f"  {sym}: insufficient TFs ({len(dfs)}) — skip")
                    return None

                f = build_features(dfs, prediction_mode=False)
                primary_df = self.training_feed.fetch_ohlcv(sym_api, primary_tf, limit=0)
                if primary_df is None or len(primary_df) < 50:
                    return None
                labels = make_labels(primary_df, forward_bars=fb, atr_k=atr_k)
                common = f.index.intersection(labels.index)
                if len(common) < 50:
                    self.log.warning(f"  {sym_api}: {len(common)} clean rows (<50) — skip")
                    return None
                f = f.loc[common].reset_index(drop=True)
                labels = labels.loc[common].reset_index(drop=True)
                valid = labels.notna() & f.notna().all(axis=1)
                f = f[valid].reset_index(drop=True)
                labels = labels[valid].reset_index(drop=True)

                if len(f) < 50:
                    self.log.warning(f"  {sym}: {len(f)} clean rows (<50) — skip")
                    return None

                self.log.info(f"  {sym_api}: {len(f)} rows, {f.shape[1]} features")
                return (f, labels)
            except Exception as e:
                self.log.warning(f"  {sym}: feature build error: {e}")
                return None

        feat_parts, label_parts = [], []
        n_workers = min(8, len(dataset_stats["symbols"]))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            for result in ex.map(_build_symbol_features, range(len(dataset_stats["symbols"]))):
                if result is not None:
                    feat_parts.append(result[0])
                    label_parts.append(result[1])

        if not feat_parts:
            raise RuntimeError("No clean feature rows — training aborted")

        combined_feats  = pd.concat(feat_parts,  ignore_index=True).dropna()
        combined_labels = pd.concat(label_parts, ignore_index=True).loc[combined_feats.index]

        # ── HOLD undersampling: cap HOLD at 40% of original row count ──
        hold_mask = combined_labels == 1
        hold_frac = hold_mask.sum() / len(combined_labels)
        if hold_frac > 0.40:
            target_hold = int(len(combined_labels) * 0.40)
            hold_idx    = combined_labels[hold_mask].index
            rng         = np.random.default_rng(42)
            drop_idx    = rng.choice(hold_idx, size=len(hold_idx) - target_hold, replace=False)
            combined_feats  = combined_feats.drop(index=drop_idx).reset_index(drop=True)
            combined_labels = combined_labels.drop(index=drop_idx).reset_index(drop=True)
            self.log.info(
                f"HOLD undersampled: {hold_frac:.1%} → "
                f"{(combined_labels==1).mean():.1%} of {len(combined_labels)} rows"
            )

        # Cap dataset size
        max_rows = training_cfg.get("max_rows", 200000)
        if len(combined_feats) > max_rows:
            combined_feats  = combined_feats.iloc[-max_rows:]
            combined_labels = combined_labels.iloc[-max_rows:]

        self.log.info(
            f"Training on {len(combined_feats)} rows from {len(feat_parts)} coins "
            f"(forward_bars={fb}, n_jobs={n_jobs})"
        )

        # ── Fit + save FeatureScaler ─────────────────────────────────
        self._write_training_status("running", source="pipeline", progress=15)
        from sklearn.preprocessing import StandardScaler
        feature_cols = combined_feats.columns.tolist()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(combined_feats.values)

        mode_dir = DATA_DIR / self.MODE
        mode_dir.mkdir(parents=True, exist_ok=True)
        save_feature_metadata(scaler, feature_cols, mode_dir)

        # ── Train models on scaled features ──────────────────────────
        self._write_training_status("running", source="pipeline", progress=20)
        closed_trades = [t for t in self.state.get_all_trades() if t.get("status") == "closed"]
        closed_trades = closed_trades[-50:]
        class_weights = _compute_class_weights(closed_trades)
        self.log.info(f"Class weights: {class_weights} (from {len(closed_trades)} closed trades)")
        results = self.ai.train_all(
            combined_feats,
            feat_df=pd.DataFrame(X_scaled, columns=feature_cols),
            labels_s=combined_labels,
            btc_rows=0,
            forward_bars=fb,
            timeframe=tf,
            min_confidence=mc,
            min_votes=mv,
            quick=quick,
            n_jobs=n_jobs,
            atr_k=atr_k,
            class_weights=class_weights,
            progress_fn=lambda p: self._write_training_status("running", source="pipeline", progress=p),
        )
        self.log.info(f"Training complete: {results}")
        self._check_model_health(results)
        self.agents.invalidate_cache()

    def _train(self, quick: bool = False, force: bool = False):
        self._reload_config()
        # Quick skip: if models already exist and not forced, skip retrain entirely
        if quick and not force:
            model_dir = DATA_DIR / self.MODE
            if (model_dir / "rf_model.pkl").exists() and (model_dir / "lgbm_model.pkl").exists():
                self.log.info(f"Quick retrain SKIPPED — models already exist for {self.MODE}")
                self._write_training_status("completed", source="startup", duration_seconds=0,
                                            note="models already exist, skipped")
                return

        training_type = "QUICK (RF+LGBM only)" if quick else "FULL (all models)"
        self.log.info("=" * 50)
        self.log.info(f"TRAINING AI MODELS v4 [{self.MODE.upper()}] — {training_type}")
        self.log.info("Training source: real Binance public API (api.binance.com)")
        self.log.info("=" * 50)

        training_cfg = self.config.get("training", {})

        # ── v5 pipeline: unified multi-timeframe dataset ──────────────────────
        if training_cfg.get("use_dataset_pipeline"):
            try:
                self._train_with_pipeline(quick, force)
                return
            except Exception as e:
                self.log.warning(f"Dataset pipeline failed: {e} — falling back to legacy training")
                # Fall through to legacy pipeline below

        try:
            import pandas as pd
            from models.ai_strategy import make_features, make_labels

            train_symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT",
                             "XRP/USDT", "DOGE/USDT", "ADA/USDT", "LINK/USDT"]
            if self.scanner.top_coins:
                for c in self.scanner.top_coins[:4]:
                    if c not in train_symbols:
                        train_symbols.append(c)

            tf = self.config.get("ml", {}).get("timeframe") or self.config.get("scanner", {}).get("timeframe", "15m")
            training_cfg = self.config.get("training", {})
            min_bars = training_cfg.get("min_bars_per_coin", 3000)

            self.log.info(f"Fetching training data from real Binance (limit=5000, min_bars={min_bars})...")
            fetched = self.training_feed.fetch_training_data(train_symbols, timeframe=tf, limit=5000, min_bars=100)

            raw_dfs = []
            btc_bars = 0
            for sym in train_symbols:
                df = fetched.get(sym)
                if df is not None and len(df) >= 100:
                    raw_dfs.append(df)
                    self.log.info(f"Training data: {sym} ({len(df)} bars @ {tf})")
                    if sym == "BTC/USDT":
                        btc_bars = len(df)

            if not raw_dfs:
                self.log.error("Training skipped — no data available from real Binance")
                return

            feat_parts, label_parts = [], []
            fb = self.config.get("ml", {}).get("forward_bars", 2)
            mc = self.config.get("strategy", {}).get("min_confidence", 0.52)
            mv = self.config.get("strategy", {}).get("min_votes", 2)

            for sym_df in raw_dfs:
                try:
                    sym_reset = sym_df.reset_index(drop=True)
                    f = make_features(sym_reset)
                    l = make_labels(sym_reset, forward_bars=fb).reindex(f.index).dropna()
                    f = f.loc[l.index]
                    if len(f) < 50:
                        self.log.warning(f"Feature skip: coin had {len(sym_df)} bars, {len(f)} clean rows (<50)")
                        continue
                    feat_parts.append(f.reset_index(drop=True))
                    label_parts.append(l.reset_index(drop=True))
                except Exception as e:
                    self.log.warning(f"Feature/label error: {e}")

            if not feat_parts:
                self.log.error("Training skipped — no clean feature rows")
                return

            combined_feats  = pd.concat(feat_parts,  ignore_index=True)
            combined_labels = pd.concat(label_parts, ignore_index=True)
            combined = pd.concat(raw_dfs, ignore_index=True)

            self.log.info(
                f"Training on {len(combined_feats)} clean rows "
                f"from {len(feat_parts)} coins (forward_bars={fb}, BTC={btc_bars} bars, source=real Binance)"
            )

            results = self.ai.train_all(
                combined,
                feat_df=combined_feats,
                labels_s=combined_labels,
                btc_rows=btc_bars if btc_bars > 0 else len(raw_dfs[0]) if raw_dfs else 0,
                forward_bars=fb,
                timeframe=tf,
                min_confidence=mc,
                min_votes=mv,
                quick=quick,
            )
            self.log.info(f"Training complete: {results}")
            self._check_model_health(results)
            self.agents.invalidate_cache()

            try:
                btc_df = fetched.get("BTC/USDT") or self.training_feed.fetch_training_data(
                    ["BTC/USDT"], timeframe="1h", limit=1000, min_bars=100
                ).get("BTC/USDT")
                if btc_df is not None and len(btc_df) >= 200:
                    hmm_result = self.hmm_regime.train(btc_df)
                    self.log.info(f"HMM regime trained: {hmm_result}")
                else:
                    self.log.warning("HMM skipped: insufficient BTC data for regime training")
            except Exception as e:
                self.log.warning(f"HMM training skipped: {e}")

            manifest = TrainingDataStore.get_manifest()
            self.log.info(f"Training cache: {len(manifest.get('coins',[]))} coins cached to disk")

        except Exception as e:
            self.log.error(f"Training failed: {e}", exc_info=True)
            try:
                self.notifier.send_error_alert(
                    f"{type(e).__name__}: {e}", context=f"Training failed [{self.MODE.upper()}]"
                )
            except Exception:
                pass

    def _check_model_health(self, results: dict):
        """Log if any model trained with suspiciously low accuracy."""
        for model_key in ("rf", "lgbm", "lstm"):
            r = results.get(model_key, {})
            acc = r.get("accuracy", None)
            status = r.get("status", "")
            if status == "below_floor":
                new_acc = r.get("new_accuracy", 0)
                self.log.warning(
                    f"Model health: {model_key.upper()} new training ({new_acc:.1%}) was below floor — keeping old"
                )
            elif acc is not None and acc < 0.40 and status not in ("kept_old", "below_floor"):
                self.log.warning(f"Model health: {model_key.upper()} accuracy {acc:.1%} is low — check training data")

    def place_order_with_confirmation(
        self, symbol, side, amount, params=None, max_retries=3
    ) -> Optional[dict]:
        """
        Place order with confirmation loop.
        Shared by both spot and futures.
        Retries on failure, verifies fill.
        """
        for attempt in range(max_retries):
            try:
                if params:
                    order = self.exchange.create_market_order(
                        symbol, side, amount, params=params
                    )
                else:
                    order = self.exchange.create_market_order(symbol, side, amount)

                order_id = order.get("id")
                if not order_id:
                    continue

                time.sleep(1)
                try:
                    filled = self.exchange.fetch_order(order_id, symbol)
                    status = filled.get("status", "unknown")
                    if status in ["closed", "filled"]:
                        self.log.info(
                            f"Order confirmed: {side.upper()} {amount:.6f} {symbol}"
                        )
                        return filled

                    elif status == "open":
                        return filled

                except Exception:
                    return order  # Testnet may not support fetch_order

            except Exception as e:
                err_str = str(e).lower()
                if "insufficient" in err_str or "balance" in err_str:
                    self.log.error(f"Insufficient funds: {symbol}")
                    return None
                if "invalid" in err_str and "order" in err_str:
                    self.log.error(f"Invalid order {symbol}: {e}")
                    return None
                self.log.error(f"Order error {symbol} attempt {attempt+1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2)
        return None

    def _get_symbol_lock(self, symbol: str) -> threading.Lock:
        with self._symbol_locks_guard:
            if symbol not in self._symbol_locks:
                self._symbol_locks[symbol] = threading.Lock()
            return self._symbol_locks[symbol]

    def get_price(self, symbol) -> float:
        price = self.feed.get_live_price(symbol)
        return price or 0.0

    def get_atr(self, symbol) -> float:
        tf = self.config.get("scanner", {}).get("timeframe", "15m")
        return self.feed.get_atr(symbol, tf, 14)

    def get_usdt_balance(self) -> float:
        try:
            bal = self.exchange.fetch_balance()
            value = float(bal["total"].get("USDT", 0.0))
            return value
        except Exception as e:
            self.log.error(f"Balance error: {e}")
            return self._balance_cache or 0.0

    def get_htf_bias(self, dfs) -> str:
        """Higher timeframe bias — prevents trading against major trend."""
        votes = []
        for tf in ["4h", "1d"]:
            df = dfs.get(tf)
            if df is None or len(df) < 50:
                continue
            close = df["close"]
            ema20 = close.ewm(span=20).mean().iloc[-1]
            ema50 = close.ewm(span=50).mean().iloc[-1]
            price = float(close.iloc[-1])
            if price > ema20 > ema50:   votes.append("BUY")
            elif price < ema20 < ema50: votes.append("SELL")
            else:                       votes.append("NEUTRAL")

        if votes.count("BUY") >= 1 and "SELL" not in votes:   return "BUY"
        elif votes.count("SELL") >= 1 and "BUY" not in votes: return "SELL"
        return "NEUTRAL"

    def _resolve_close_pnl(self, trade: dict, symbol: str, detection_price: float, fraction: float) -> tuple:
        """
        After a close order is placed, look up the actual fill to get the real exit
        price and PnL. For futures fills that carry realizedPnl, use it directly.
        Falls back to detection_price if no matching fill is found.
        Returns (pnl, actual_price).
        """
        try:
            closing_side = "sell" if trade["side"] == "long" else "buy"
            fills = self.exchange.fetch_my_trades(symbol, limit=10)
            closing_fills = [f for f in fills if f.get("side", "").lower() == closing_side]
            if closing_fills:
                best = max(closing_fills, key=lambda x: x["time"])
                # Bot enforces one open trade per symbol (risk manager correlation filter),
                # so the most-recent closing fill unambiguously belongs to this trade.
                actual_price = float(best.get("price", detection_price))
                raw_rpnl = best.get("realizedPnl")
                if raw_rpnl is not None:
                    return float(raw_rpnl) * fraction, actual_price
                return self._calc_pnl(trade, actual_price) * fraction, actual_price
        except Exception as e:
            self.log.debug(f"_resolve_close_pnl fallback to detection_price for {symbol}: {e}")
        return self._calc_pnl(trade, detection_price) * fraction, detection_price

    def check_exits(self):
        """
        Check open trades for exit conditions.
        Pre-fetches ATR values in parallel to avoid sequential API calls.
        """
        trades = self.state.get_open_trades()
        if not trades:
            return
        symbols = list(set(t["symbol"] for t in trades))
        atr_cache = {}
        def _fetch_atr(s):
            return s, self.get_atr(s)
        with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as ex:
            for s, atr in ex.map(_fetch_atr, symbols):
                atr_cache[s] = atr
        def _get_atr_cached(symbol):
            return atr_cache.get(symbol, 0.0)
        exits = self.risk.check_exits(trades, self.get_price, _get_atr_cached)
        for trade, price, reason, fraction in exits:
            close_amount = trade["amount"] * fraction
            _sym_lk = self._get_symbol_lock(trade["symbol"])
            with _sym_lk:
                if trade["id"] not in {t["id"] for t in self.state.get_open_trades()}:
                    continue  # already closed by RL or another path
                order = self._place_close(
                    trade["symbol"], close_amount, trade["side"]
                )
            if order:
                pnl, price = self._resolve_close_pnl(trade, trade["symbol"], price, fraction)
                if fraction >= 1.0:
                    self.state.close_trade(trade["id"], price, pnl)
                    self.risk.cleanup_trade(trade["id"])
                    self._cancel_exchange_stop_loss(
                        trade["symbol"], trade.get("sl_order_id", "")
                    )
                else:
                    self.state.partial_close_trade(trade["id"], close_amount, pnl)
                    if self.MODE == "futures" and trade.get("sl_order_id"):
                        remaining = trade["amount"] - close_amount
                        if remaining > 0:
                            self._cancel_exchange_stop_loss(
                                trade["symbol"], trade.get("sl_order_id", "")
                            )
                            atr = _get_atr_cached(trade["symbol"])
                            if atr and atr > 0:
                                new_sl_id = self.execution._place_sl(
                                    trade["symbol"], trade["side"], remaining,
                                    float(trade["price"]), atr,
                                )
                                if new_sl_id:
                                    self.state.update_trade_sl(trade["id"], new_sl_id)
                                    self.log.info(f"SL re-placed for {trade['symbol']}: {remaining:.4f} remaining")
                self.ai.record_trade_result(trade["symbol"], pnl)
                self.agents.record_trade_result(trade["symbol"], pnl)
                self.risk.record_trade_result(pnl, self.get_usdt_balance())
                self.rl_agent.record_external_close(trade["id"], pnl)
                if pnl < 0:
                    self._symbol_loss_cooldown[trade["symbol"]] = datetime.now(timezone.utc)
                icon    = "TP" if pnl > 0 else "SL"
                pct_str = f" ({fraction*100:.0f}%)" if fraction < 1.0 else ""
                self.log.info(f"{icon} EXIT{pct_str} {trade['symbol']} | PnL={pnl:+.4f} | {reason}")
                self.notifier.send_alert(
                    f"{icon} EXIT{pct_str} {trade['symbol']}\n"
                    f"PnL: ${pnl:+.4f} USDT\n"
                    f"Reason: {reason}"
                )

    def _rl_manage_trades(self, regime_ctx=None):
        """
        Apply RL agent to open trades for execution decisions.
        Runs AFTER check_exits() so ATR safety stops always fire first.
        Fallback: HOLD (existing ATR logic handles everything).
        """
        open_trades = self.state.get_open_trades()
        if not open_trades:
            self.rl_agent.prune_pending([])
            return

        self.rl_agent.prune_pending([t["id"] for t in open_trades])

        regime  = (regime_ctx or {}).get("hmm_regime", "RANGING")
        balance = self.get_usdt_balance()

        # Price for momentum context (use BTC as proxy for market direction)
        price_ago = 0.0
        try:
            tf = self.config.get("scanner", {}).get("timeframe", "15m")
            btc_df = self.feed.fetch_ohlcv("BTC/USDT", tf, limit=3)
            if btc_df is not None and len(btc_df) >= 2:
                price_ago = float(btc_df["close"].iloc[-2])
        except Exception:
            pass

        for trade in list(open_trades):
            try:
                symbol        = trade["symbol"]
                current_price = self.get_price(symbol)
                if not current_price:
                    continue
                atr = self.get_atr(symbol)

                # Record step reward from previous cycle's decision
                self.rl_agent.record_step(
                    trade_id    = trade["id"],
                    next_price  = current_price,
                    next_atr    = atr,
                    done        = False,
                )

                action, rl_conf = self.rl_agent.decide(
                    trade         = trade,
                    current_price = current_price,
                    atr           = atr,
                    regime        = regime,
                    price_1h_ago  = price_ago,
                )

                if action == "HOLD":
                    continue

                elif action == "CLOSE":
                    # RL CLOSE suppressed — check_exits() owns all full exits.
                    self.log.debug(f"RL CLOSE {symbol} → suppressed (exits owned by check_exits)")
                    continue

                elif action == "SCALE_OUT":
                    close_amt = trade["amount"] * 0.25
                    if close_amt > 0:
                        order = self._place_close(symbol, close_amt, trade["side"])
                        if order:
                            full_pnl = self._calc_pnl(trade, current_price)
                            pnl      = full_pnl * 0.25
                            self.state.partial_close_trade(trade["id"], close_amt, pnl)
                            try:
                                self.ai.record_trade_result(symbol, pnl)
                            except Exception:
                                pass
                            try:
                                self.agents.record_trade_result(symbol, pnl)
                            except Exception:
                                pass
                            try:
                                self.risk.record_trade_result(pnl, balance)
                            except Exception:
                                pass
                            try:
                                self.rl_agent.record_external_close(trade["id"], pnl)
                            except Exception:
                                pass
                            self.log.info(f"RL SCALE_OUT {symbol} 25% | PnL={pnl:+.4f}")

                elif action == "SCALE_IN":
                    entry_usdt   = trade["amount"] * float(trade.get("price", current_price) or current_price)
                    extra_usdt   = entry_usdt * 0.25
                    extra_amount = extra_usdt / current_price
                    ok, reason   = self.risk.can_open_trade(
                        symbol=symbol, open_trades=open_trades,
                        balance=balance, new_usdt=extra_usdt,
                        get_price_fn=self.get_price,
                    )
                    if ok and extra_usdt >= 5:
                        if trade["side"] in ("buy", "long"):
                            order = self._place_buy(symbol, extra_amount)
                        else:
                            order = self._place_sell(symbol, extra_amount)
                        if order:
                            new_amount = trade["amount"] + extra_amount
                            trade["amount"] = new_amount
                            self.state.update_trade_amount(trade["id"], new_amount)
                            self.log.info(f"RL SCALE_IN {symbol} +25% @ {current_price:.4f}")

            except Exception as e:
                self.log.error(f"RL manage error {trade.get('symbol','?')}: {e}")

    def analyze_symbol(self, symbol, balance, open_trades, regime_ctx=None, btc_1h_return=None,
                        pre_multi=None, pre_train=None):
        """Decision pipeline: Ensemble → Risk Agent → Execution. v5 layered architecture."""
        try:
            # Per-symbol cooldown: skip 30 min after a loss to prevent consecutive entries
            cooldown_until = self._symbol_loss_cooldown.get(symbol)
            if cooldown_until:
                elapsed = (datetime.now(timezone.utc) - cooldown_until).total_seconds()
                if elapsed < 1800:
                    self.log.debug(f"[{symbol}] cooldown {(1800-elapsed)/60:.0f}m remaining after loss")
                    return
                else:
                    del self._symbol_loss_cooldown[symbol]

            # Use pre-fetched data if available, otherwise fetch
            if pre_multi:
                dfs = pre_multi
            else:
                tf = (self.config.get("ml", {}).get("timeframe")
                      or self.config.get("scanner", {}).get("timeframe", "15m"))
                train_tfs = self.config.get("training", {}).get("timeframes", ["1h", "4h", "1d"])
                multi_tfs = [(t, 500 if t == tf else (300 if t in ("1h","30m") else 200)) for t in train_tfs]
                dfs = self.feed.fetch_multi_timeframe(symbol, timeframes=multi_tfs)
            if not dfs:
                return
            tf = (self.config.get("ml", {}).get("timeframe")
                  or self.config.get("scanner", {}).get("timeframe", "15m"))
            if tf not in dfs:
                return

            df_tf = dfs[tf]
            df_1h = dfs.get("1h", df_tf)
            self.ai.ingest_new_data(symbol, df_tf)

            # ── Layer 1: Ensemble (SMC + Technical + Macro/Flow) ─────────
            hmm_regime = (regime_ctx or {}).get("hmm_regime", "RANGING")

            # Fail-fast: compute multi-TF features for ML agent via build_features()
            try:
                from features.feature_builder import build_features, SUPPORTED_TFS, FeatureValidationError
                if pre_train:
                    tf_dfs = pre_train
                else:
                    tf_dfs = {}
                    for t in SUPPORTED_TFS:
                        tdf = self.training_feed.fetch_ohlcv(symbol, t, limit=2000)
                        if tdf is not None and len(tdf) >= 50:
                            tf_dfs[t] = tdf
                if len(tf_dfs) >= 2:
                    feature_row = build_features(tf_dfs, prediction_mode=True)
                    self.ml_agent.set_features(symbol, feature_row)
            except FeatureValidationError as e:
                self.log.error(f"[{symbol}] Feature validation FAILED — no ML signal: {e}")
                # Don't trade this symbol on ML signal — let SMC decide alone
                pass
            except Exception as e:
                self.log.error(f"[{symbol}] Feature build FAILED — no ML signal: {e}")
                # ccxt BadSymbol: "binance does not have market symbol X/USDT" — permanently invalid
                if "does not have market symbol" in str(e):
                    self.feed.mark_invalid(symbol)  # mark_invalid() logs its own warning

            self._current_regime = (regime_ctx or {}).get("hmm_regime", "RANGING")
            ensemble = self.ensemble.run(symbol, df_1h, self.profile, market_ctx=regime_ctx)

            # Log per-agent detail at DEBUG only
            for s in ensemble.signals:
                self.log.debug(f"AGENT {symbol} | {s.agent}: net={s.net_score:+.3f} buy={s.buy_score:.2f} sell={s.sell_score:.2f} | {s.reasoning[:90]}")
            self.log.debug(f"ENSEMBLE {symbol} | action={ensemble.action} net={ensemble.net_score:+.3f} conf={ensemble.confidence:.2f} agree={ensemble.agents_agreeing}/{ensemble.agents_total}")

            if ensemble.action == "HOLD":
                self.state.add_signal({
                    "symbol": symbol, "action": "HOLD", "confidence": ensemble.confidence,
                    "status": "hold", "reason": f"net={ensemble.net_score:+.3f}",
                    "strategy": f"ensemble:{ensemble.net_score:+.3f}",
                    "timeframe": "AUTO",
                    "indicators": {"buy_score": ensemble.buy_score, "sell_score": ensemble.sell_score,
                                   "agents_agree": ensemble.agents_agreeing},
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                self.log.info(f"SIGNAL {symbol} → HOLD | conf={ensemble.confidence:.2f} agree={ensemble.agents_agreeing}/{ensemble.agents_total} | HOLD (net={ensemble.net_score:+.3f} regime={hmm_regime})")
                return

            # HTF bias for risk agent
            htf_bias = self.get_htf_bias(dfs)
            price = self.get_price(symbol)

            # ── Layer 2: Risk Decision ──────────────────────────────────
            decision = self.risk_agent.evaluate(
                ensemble=ensemble, symbol=symbol, df_1h=df_1h,
                profile=self.profile, regime_ctx=regime_ctx,
                btc_return=btc_1h_return or 0.0,
                open_trades=open_trades, balance=balance,
                get_price_fn=self.get_price, get_atr_fn=self.get_atr,
                htf_bias=htf_bias,
                all_trades=self.state.get_all_trades(),
            )

            if not decision.approved:
                self.log.info(f"SIGNAL {symbol} → {ensemble.action} | conf={ensemble.confidence:.2f} agree={ensemble.agents_agreeing}/{ensemble.agents_total} | REJECTED: {' | '.join(decision.reasons)}")
                self.state.add_signal({
                    "symbol": symbol, "action": ensemble.action,
                    "confidence": ensemble.confidence,
                    "status": "rejected", "reason": " | ".join(decision.reasons),
                    "strategy": f"ensemble:{ensemble.net_score:+.3f}",
                    "timeframe": "AUTO",
                    "indicators": {
                        "buy_score": ensemble.buy_score, "sell_score": ensemble.sell_score,
                        "agents_agree": ensemble.agents_agreeing, "profile": self.profile.name,
                        "adx":           round((regime_ctx or {}).get("adx", 0.0), 1),
                        "quality_score": round(getattr(decision, "quality_score", 0.0), 3),
                        "regime":        (regime_ctx or {}).get("regime", "?"),
                    },
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                return

            self.state.add_signal({
                "symbol": symbol, "action": ensemble.action,
                "confidence": decision.adjusted_conf,
                "status": "taken",
                "strategy": f"ensemble:{ensemble.net_score:+.3f}",
                "timeframe": "AUTO",
                "indicators": {
                    "buy_score": ensemble.buy_score, "sell_score": ensemble.sell_score,
                    "agents_agree": ensemble.agents_agreeing, "profile": self.profile.name,
                    "adx":           round((regime_ctx or {}).get("adx", 0.0), 1),
                    "quality_score": round(getattr(decision, "quality_score", 0.0), 3),
                    "regime":        (regime_ctx or {}).get("regime", "?"),
                    "vol_ratio":     round((regime_ctx or {}).get("vol_ratio", 1.0), 2),
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # ── Layer 3: Execution ─────────────────────────────────────
            if not price or price <= 0:
                return

            # Close opposing position first (if any)
            holding = self._find_position(symbol, open_trades)
            opp_side = {"BUY": "long", "SELL": "short"}.get(ensemble.action)
            if holding and holding["side"] != opp_side:
                order = self._place_close(symbol, holding["amount"], holding["side"])
                if order:
                    close_price = self.get_price(symbol) or price
                    pnl = self._calc_pnl(holding, close_price)
                    self.state.close_trade(holding["id"], close_price, pnl)
                    self.ai.record_trade_result(symbol, pnl)
                    self.agents.record_trade_result(symbol, pnl)
                    self.risk.record_trade_result(pnl, balance)
                    self.risk.cleanup_trade(holding["id"])
                    self._cancel_exchange_stop_loss(symbol, holding.get("sl_order_id", ""))
                    self.rl_agent.record_external_close(holding["id"], pnl)
                    self.log.info(f"CLOSE OPPOSITE {symbol} | PnL={pnl:+.4f}")
                    self.notifier.send_alert(f"CLOSE OPPOSITE {symbol}\nPnL: ${pnl:+.4f}")
                # Re-check open trades after close
                open_trades = self.state.get_open_trades()

            # Dedup: don't open if already holding same side
            holding = self._find_position(symbol, open_trades)
            if holding and holding["side"] == opp_side:
                self.log.info(f"SKIP {symbol}: already open {holding['side']}")
                return
            if ensemble.action == "SELL" and self.MODE == "spot" and not holding:
                return

            trade = self.execution.execute_entry(
                decision=decision, symbol=symbol, action=ensemble.action,
                price=price, get_atr_fn=self.get_atr,
                place_buy_fn=self._place_buy, place_sell_fn=self._place_sell,
                strat=f"ensemble:{ensemble.net_score:+.3f}",
            )

        except Exception as e:
            self.log.error(f"Error analyzing {symbol}: {e}", exc_info=True)

    def _passes_trade_filters(self, df_1h, symbol: str) -> bool:
        """
        Step 10: Pre-entry quality filters.
        Scoring: need >= 1/2 conditions (vol spike OR ATR expansion).
        Both pass = high conviction entry.
        """
        try:
            close = df_1h["close"]
            high  = df_1h["high"]
            low   = df_1h["low"]
            vol   = df_1h["volume"]

            vol_ma   = vol.rolling(20).mean()
            vol_ratio = float(vol.iloc[-1]) / (float(vol_ma.iloc[-1]) + 1e-9)
            vol_ok   = vol_ratio > 1.5

            import ta as _ta
            atr      = _ta.volatility.AverageTrueRange(high, low, close, 14).average_true_range()
            atr_ma   = atr.rolling(20).mean()
            atr_ratio = float(atr.iloc[-1]) / (float(atr_ma.iloc[-1]) + 1e-9)
            atr_ok   = atr_ratio > 1.0

            score = int(vol_ok) + int(atr_ok)
            if score == 0:
                self.log.info(
                    f"FILTER SKIP {symbol}: vol={vol_ratio:.2f}x "
                    f"atr={atr_ratio:.2f}x (need >= 1)"
                )
                return False
            elif score == 2:
                self.log.info(f"FILTER PASS {symbol}: vol={vol_ratio:.2f}x atr={atr_ratio:.2f}x (both)")
            else:
                self.log.info(
                    f"FILTER PASS {symbol}: score=1/2 "
                    f"vol={'OK' if vol_ok else 'skip'}({vol_ratio:.2f}x) "
                    f"atr={'OK' if atr_ok else 'skip'}({atr_ratio:.2f}x)"
                )
            return True
        except Exception as e:
            self.log.warning(f"Trade filter error for {symbol}: {e}")
            return True

    def _find_position(self, symbol, open_trades):
        """Find existing position for a symbol."""
        return next(
            (t for t in open_trades if t["symbol"] == symbol),
            None
        )

    def _get_leverage(self) -> int:
        """Override in futures bot."""
        return 1

    def _place_exchange_stop_loss(self, symbol, side, amount, entry_price, atr):
        """Override in futures bot to place exchange-side SL."""
        return ""

    def _cancel_exchange_stop_loss(self, symbol, sl_order_id):
        """Override in futures bot to cancel exchange-side SL."""
        pass

    def sync_with_exchange(self):
        """Sync state with exchange. Closes local trades not found on exchange per mode."""
        if self.MODE == "futures":
            self._sync_futures()
        elif self.MODE == "spot":
            self._sync_spot()

    def _import_exchange_positions(self, positions: list, d: dict) -> None:
        """Import any exchange position not currently tracked as open in state.
        No symbol blocklists — if the exchange has it, the bot should know about it.
        """
        import time as _time
        our_syms = {t["symbol"] for t in d.get("trades", []) if t.get("status") == "open"}
        for pos in positions:
            sym = pos["symbol"]
            if sym in our_syms or float(pos.get("amount", 0)) == 0:
                continue
            trade = {
                "id":              f"sync_pos_{sym.replace('/','_')}_{int(_time.time())}",
                "symbol":          sym,
                "side":            pos["side"],
                "amount":          pos["amount"],
                "price":           pos["entry_price"],
                "mark_price":      pos.get("mark_price", 0),
                "live_pnl":        round(pos["pnl"], 6),
                "timestamp":       datetime.now(timezone.utc).isoformat(),
                "strategy":        "synced_from_position",
                "timeframe":       f"AUTO-{self.MODE}",
                "status":          "open",
                "mode":            self.MODE,
                "leverage":        pos.get("leverage", 5),
                "sl_order_id":     "",
                "pnl":             0.0,
                "close_price":     0.0,
                "close_timestamp": ""
            }
            d["trades"].append(trade)
            d["stats"]["total_trades"] += 1
            self.log.info(f"Sync: imported {sym} {pos['side']} from exchange")

    def _cleanup_ghost_trades(self, exchange_syms: set, d: dict) -> None:
        """Cancel trades that are open in state but missing from the exchange for >60s.
        Skips trades entered <60s ago (race condition grace).
        Trades missing >60s are immediately closed as ghost trades.
        """
        now = datetime.now(timezone.utc)
        for t in d.get("trades", []):
            if t["status"] != "open":
                continue
            sym = t["symbol"]
            if sym in exchange_syms:
                continue
            ts_str = t.get("timestamp", "")
            age_s = None
            try:
                if ts_str:
                    opened = datetime.fromisoformat(ts_str)
                    if opened.tzinfo is None:
                        opened = opened.replace(tzinfo=timezone.utc)
                    age_s = (now - opened).total_seconds()
            except (ValueError, TypeError):
                pass
            if age_s is not None and age_s < 60:
                self.log.debug(f"Sync: {sym} entered <60s ago, skipping ghost check")
                continue
            # >60s missing from exchange — cancel the ghost trade immediately
            close_price = float(t.get("price", 0))
            raw_rpnl = None
            try:
                # Look for the closing fill: the side opposite to the position
                # (long was closed by a sell fill; short was closed by a buy fill)
                closing_side = "sell" if t["side"] == "long" else "buy"
                fills = self.exchange.fetch_my_trades(sym, limit=20)
                closing_fills = [f for f in fills if f.get("side", "").lower() == closing_side]
                if closing_fills:
                    best = max(closing_fills, key=lambda x: x["time"])
                    close_price = float(best.get("price", close_price))
                    raw_rpnl = best.get("realizedPnl")
                elif fills:
                    # No closing fill found — fall back to ticker
                    close_price = self.exchange.fetch_ticker(sym)["last"]
            except Exception:
                try:
                    close_price = self.exchange.fetch_ticker(sym)["last"]
                except Exception:
                    pass
            pnl = float(raw_rpnl) if raw_rpnl is not None else self._calc_pnl(t, close_price)
            t["status"]          = "closed"
            t["close_price"]     = close_price
            t["pnl"]             = round(pnl, 8)
            t["close_timestamp"] = now.isoformat()
            d["stats"]["total_pnl"] += round(pnl, 6)
            if pnl > 0:   d["stats"]["wins"]   += 1
            elif pnl < 0: d["stats"]["losses"] += 1
            self.log.info(f"Sync: cancelled ghost trade {sym} (missing >60s) pnl={pnl:+.4f}")
            # NOTE: risk.record_trade_result intentionally excluded — ghost PnL is
            # estimated from mark price, not a real fill, and must not trip the circuit breaker.
            for fn in [
                lambda: self._cancel_exchange_stop_loss(sym, t.get("sl_order_id", "")),
                lambda: self.rl_agent.record_external_close(t["id"], pnl),
                lambda: self.ai.record_trade_result(sym, pnl),
                lambda: self.agents.record_trade_result(sym, pnl),
            ]:
                try: fn()
                except Exception: pass

    def _sync_futures(self):
        """Futures sync: update live PnL, import untracked positions, cancel 60s+ ghosts."""
        try:
            self.log.info("Sync running — mode=futures")
            position_fetch_ok = True
            try:
                positions = self.exchange.get_position()
            except Exception as pe:
                self.log.warning(f"Sync: get_position failed ({pe}) — skipping ghost cleanup this cycle")
                positions = []
                position_fetch_ok = False
            if not positions:
                positions = []

            exchange_syms = {p["symbol"] for p in positions if float(p.get("amount", 0)) != 0}
            self.log.info(f"Sync: got {len(exchange_syms)} active positions from exchange")

            bal  = self.exchange.fetch_balance()
            usdt = float(bal["total"].get("USDT", 0))

            d = self.state.state
            with self.state._lock:
                # Update live PnL for tracked positions
                for t in d.get("trades", []):
                    if t["status"] != "open":
                        continue
                    sym = t["symbol"]
                    if sym in exchange_syms:
                        for pos in positions:
                            if pos["symbol"] == sym:
                                t["amount"]     = pos["amount"]
                                t["price"]      = pos["entry_price"]
                                t["leverage"]   = pos.get("leverage", 5)
                                t["live_pnl"]   = round(pos["pnl"], 6)
                                t["mark_price"] = pos.get("mark_price", 0)
                                break

                # Import any exchange position the bot doesn't know about
                self._import_exchange_positions(positions, d)

                # Cancel trades open in state but gone from exchange for >60s
                if position_fetch_ok:
                    self._cleanup_ghost_trades(exchange_syms, d)

            self.state.state["stats"]["balance"]   = round(usdt, 2)
            self.state.state["stats"]["last_sync"] = datetime.now(timezone.utc).isoformat()
            self.log.info(f"Sync saving balance=${usdt:.2f}")
        except Exception as e:
            self.log.warning(f"Sync error: {e}")
        finally:
            self.state.save(immediate=True)

    def _sync_spot(self):
        """Balance-based sync for spot mode."""
        try:
            self.log.info("Sync running — mode=spot")
            bal = self.exchange.fetch_balance()
            totals = bal.get("total", {})

            d = self.state.state
            our_open = {t["symbol"]: t for t in d["trades"] if t["status"] == "open"}
            our_syms = set(our_open.keys())
            all_syms = {t["symbol"] for t in d["trades"]}

            with self.state._lock:
                # RULE 1: Trade in bot, NOT on exchange → close it
                for t in d["trades"]:
                    if t["status"] != "open":
                        continue
                    ts_str = t.get("timestamp", "")
                    recent_entry = False
                    try:
                        if ts_str:
                            opened = datetime.fromisoformat(ts_str)
                            if opened.tzinfo is None:
                                opened = opened.replace(tzinfo=timezone.utc)
                            if (datetime.now(timezone.utc) - opened).total_seconds() < 120:
                                recent_entry = True
                    except (ValueError, TypeError):
                        pass
                    if recent_entry:
                        self.log.debug(f"Sync: skipping orphan check for {t['symbol']} (entered <120s ago)")
                        continue
                    asset = t["symbol"].split("/")[0]
                    held = float(totals.get(asset, 0))
                    min_held = float(t["amount"]) * 0.1
                    trade_value = float(t["amount"]) * float(t["price"])
                    if held < min_held or trade_value < 1.0:
                        close_price = float(t["price"])
                        try:
                            closing_side = "sell"   # spot is always long; closed by a sell
                            fills = self.exchange.fetch_my_trades(t["symbol"], limit=10)
                            closing_fills = [f for f in fills if f.get("side", "").lower() == closing_side]
                            if closing_fills:
                                best = max(closing_fills, key=lambda x: x["time"])
                                close_price = float(best.get("price", close_price))
                            elif fills:
                                close_price = self.exchange.fetch_ticker(t["symbol"])["last"]
                        except Exception:
                            try:
                                close_price = self.exchange.fetch_ticker(t["symbol"])["last"]
                            except Exception:
                                pass
                        pnl = self._calc_pnl(t, close_price)
                        t["status"]          = "closed"
                        t["close_price"]     = close_price
                        t["pnl"]             = round(pnl, 8)
                        t["close_timestamp"] = datetime.now(timezone.utc).isoformat()
                        d["stats"]["total_pnl"] += round(pnl, 6)
                        if pnl > 0:
                            d["stats"]["wins"]   += 1
                        elif pnl < 0:
                            d["stats"]["losses"] += 1
                        self.log.info(f"Sync: closed {t['symbol']} (asset {asset} balance={held:.6f}) pnl={pnl:+.4f}")
                        try:
                            self.risk.record_trade_result(pnl, 5000)
                        except Exception:
                            pass
                        try:
                            self.ai.record_trade_result(t.get("symbol", "unknown"), pnl)
                        except Exception:
                            pass

                # RULE 2: Import watchlist exchange balances with >$50 value
                try:
                    watchlist_coins = set(self.scanner.get_coins(self.exchange, invalid_symbols=self.feed.invalid_symbols))
                except Exception:
                    watchlist_coins = set()
                for asset, free_val in bal.get("free", {}).items():
                    free = float(free_val)
                    if free <= 0:
                        continue
                    sym = f"{asset}/USDT"
                    if asset in ("USDT", "USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDD"):
                        continue
                    if sym in all_syms:
                        continue
                    if sym not in watchlist_coins:
                        continue
                    try:
                        ticker = self.exchange.fetch_ticker(sym)
                        price = float(ticker.get("last", 0))
                    except Exception:
                        continue
                    if price <= 0:
                        continue
                    if free * price < 50.0:
                        self.log.debug(f"Sync: skipping {sym} — dust (${free*price:.2f})")
                        continue
                    import time as _time
                    trade = {
                        "id":              f"sync_{sym.replace('/','_')}_{int(_time.time())}",
                        "symbol":          sym,
                        "side":            "buy",
                        "amount":          round(free, 8),
                        "price":           price,
                        "mark_price":      price,
                        "live_pnl":        0.0,
                        "timestamp":       datetime.now(timezone.utc).isoformat(),
                        "strategy":        "synced_from_exchange",
                        "timeframe":       f"AUTO-{self.MODE}",
                        "status":          "open",
                        "mode":            self.MODE,
                        "leverage":        1,
                        "sl_order_id":     "",
                        "pnl":             0.0,
                        "close_price":     0.0,
                        "close_timestamp": ""
                    }
                    d["trades"].append(trade)
                    d["stats"]["total_trades"] += 1
                    self.log.info(f"Sync: imported {sym} buy {free:.6f} (${free*price:.2f})")

            usdt = float(totals.get("USDT", 0))
            self.log.info(f"Sync saving balance=${usdt:.2f}")
            self.state.state["stats"]["balance"]   = round(usdt, 2)
            self.state.state["stats"]["last_sync"] = datetime.now(timezone.utc).isoformat()
            self.state.state["stats"]["total_live_pnl"] = 0.0
        except Exception as e:
            import traceback
            self.log.warning(f"Sync error: {e}")
            self.log.warning(traceback.format_exc())
        finally:
            self.state.save(immediate=True)

    def _write_training_status(self, status, source="unknown", **extra):
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            p = DATA_DIR / f"retrain_status_{self.MODE}.json"
            data = {"status": status, "source": source, **extra}
            with open(p, "w") as f:
                json.dump(data, f)
        except OSError:
            pass

    def _get_watchlist_hash(self, coins):
        return hash(tuple(sorted(coins)))

    def _retrain_if_watchlist_changed(self, new_coins):
        new_hash = self._get_watchlist_hash(new_coins)
        if hasattr(self, '_last_watchlist_hash') and self._last_watchlist_hash == new_hash:
            return
        if hasattr(self, '_last_watchlist_hash'):
            added   = set(new_coins) - set(getattr(self, '_last_watchlist', []))
            removed = set(getattr(self, '_last_watchlist', [])) - set(new_coins)
            change_count = len(added) + len(removed)
            if change_count >= 5:
                self.log.info(f"Watchlist changed by {change_count} coins Added:{added} Removed:{removed}")
                try:
                    self.notifier.send(
                        f"ℹ️ <b>Watchlist changed [{self.MODE.upper()}]</b> — {change_count} coins\n"
                        f"Added: {added or 'none'}\n"
                        f"Removed: {removed or 'none'}\n"
                        f"Retraining models on new coin set…"
                    )
                except Exception:
                    pass
                try:
                    self._with_training_heartbeat(lambda: self._train(quick=True, force=True), source="watchlist")
                except Exception as e:
                    self.log.error(f"Watchlist retrain error: {e}", exc_info=True)
        self._last_watchlist_hash = new_hash
        self._last_watchlist      = list(new_coins)

    def _check_for_retrain_request(self):
        """Check if dashboard requested a retrain via signal file."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        p = DATA_DIR / f"retrain_requested_{self.MODE}.json"
        if not p.exists():
            return

        req = None
        for attempt in range(3):
            try:
                with open(p) as f:
                    req = json.load(f)
                break
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                if attempt < 2:
                    time.sleep(0.5)
                else:
                    return

        if not req or not req.get("requested"):
            return

        self.log.info(f"Retrain request detected from {req.get('source', 'unknown')}: {req.get('reason', 'manual')}")
        try:
            self.notifier.send(
                f"🔄 <b>Retraining requested [{self.MODE.upper()}]</b>\n"
                f"Source: {req.get('source', 'dashboard')}\n"
                f"Reason: {req.get('reason', 'manual')}"
            )
        except Exception:
            pass

        try:
            self._with_training_heartbeat(lambda: self._train(quick=False), source="dashboard")
        except Exception as e:
            self.log.error(f"Retrain processing error: {e}", exc_info=True)
            try:
                self.notifier.send_error_alert(str(e), context=f"Retrain failed [{self.MODE.upper()}]")
            except Exception:
                pass
        finally:
            try:
                p.unlink()
            except Exception:
                pass

    def is_trading_paused(self) -> bool:
        """Check if trading has been paused via dashboard."""
        p = DATA / "trading_paused.json"
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f).get("paused", False)
            except Exception:
                pass
        return False

    def _close_all_positions(self, open_trades, balance):
        """Close all open positions — triggered by dashboard Close All button."""
        if not open_trades:
            self.log.info("Close All: no open trades to close")
            return
        self.log.info(f"Close All: closing {len(open_trades)} positions")
        self.notifier.send_alert(f"CLOSE ALL: closing {len(open_trades)} positions")
        for trade in list(open_trades):
            sym = trade["symbol"]
            try:
                price = self.get_price(sym)
                if not price or price <= 0:
                    price = float(trade.get("price", 0))
                order = self._place_close(sym, trade["amount"], trade["side"])
                if order:
                    pnl = self._calc_pnl(trade, price)
                    self.state.close_trade(trade["id"], price, pnl)
                    self.risk.cleanup_trade(trade["id"])
                    self._cancel_exchange_stop_loss(sym, trade.get("sl_order_id", ""))
                    try:
                        self.ai.record_trade_result(sym, pnl)
                    except Exception:
                        pass
                    try:
                        self.agents.record_trade_result(sym, pnl)
                    except Exception:
                        pass
                    try:
                        self.risk.record_trade_result(pnl, balance)
                    except Exception:
                        pass
                    try:
                        self.rl_agent.record_external_close(trade["id"], pnl)
                    except Exception:
                        pass
                    self.log.info(f"Close All: {sym} closed pnl={pnl:+.4f}")
                else:
                    self.log.warning(f"Close All: {sym} order failed — marking closed locally")
                    pnl = self._calc_pnl(trade, price)
                    self.state.close_trade(trade["id"], price, pnl)
                    self.risk.cleanup_trade(trade["id"])
                    self._cancel_exchange_stop_loss(sym, trade.get("sl_order_id", ""))
                    try:
                        self.ai.record_trade_result(sym, pnl)
                    except Exception:
                        pass
                    try:
                        self.agents.record_trade_result(sym, pnl)
                    except Exception:
                        pass
                    try:
                        self.risk.record_trade_result(pnl, balance)
                    except Exception:
                        pass
                    try:
                        self.rl_agent.record_external_close(trade["id"], pnl)
                    except Exception:
                        pass
            except Exception as e:
                self.log.error(f"Close All: error closing {sym}: {e}")
        self.state.save(immediate=True)
        self.notifier.send_alert(f"CLOSE ALL: done — {len(open_trades)} positions closed")

    def run_once(self):
        """One scan cycle — identical for both modes."""
        regime_ctx  = None   # initialise early; assigned after symbol scan
        self._maybe_swap_profile()
        self.sync_with_exchange()
        self.check_exits()

        balance     = self.get_usdt_balance()
        open_trades = self.state.get_open_trades()

        # Check for close-all request from dashboard
        close_flag = DATA / "close_all_positions.json"
        if close_flag.exists():
            self._close_all_positions(open_trades, balance)
            try:
                close_flag.unlink()
            except OSError:
                pass
            open_trades = self.state.get_open_trades()
            balance     = self.get_usdt_balance()
            max_reached = len(open_trades) >= self.max_open

        # Check if trading paused via dashboard
        trading_paused = self.is_trading_paused()
        if trading_paused:
            self.log.info("Trading PAUSED via dashboard — monitoring only")

        can_trade, reason = self.risk.breaker.can_trade(balance)
        if not can_trade:
            self.log.warning(f"Circuit breaker: {reason}")

        max_reached = len(open_trades) >= self.max_open
        _vol_ratio  = (regime_ctx or {}).get("vol_ratio", 1.0)
        symbols     = self.scanner.get_coins(self.exchange, invalid_symbols=self.feed.invalid_symbols,
                                             current_atr_ratio=_vol_ratio)
        self._post_scan(symbols)
        self.feed.subscribe_many(symbols)   # ensure WS + REST monitor tracking
        self.log.info(f"[{self.MODE.upper()}] Watching: {symbols}")

        regime_ctx = self.risk.detect_market_regime(self.feed, symbols)

        # HMM overlay + BTC 1h return: single fetch from real Binance
        btc_1h_return = 0.0
        try:
            btc_df = self.training_feed.fetch_ohlcv("BTC/USDT", "1h", limit=150)
            if btc_df is not None and len(btc_df) >= 2:
                btc_1h_return = float(btc_df["close"].pct_change().iloc[-1])
            if btc_df is not None and len(btc_df) >= 50:
                hmm_regime, hmm_adj = self.hmm_regime.get_regime_and_adjustments(btc_df, self.profile.name)
            else:
                hmm_regime, hmm_adj = self.hmm_regime.predict_fallback(btc_df, self.profile.name)
                self.log.info(f"HMM fallback (no BTC data): regime={hmm_regime}")
            if regime_ctx:
                regime_ctx = dict(regime_ctx)  # shallow copy — prevent cache pollution
                old_min_conf  = regime_ctx.get("min_conf", self.min_conf)
                old_size_mult = regime_ctx.get("size_mult", 1.0)
                regime_ctx["min_conf"]   = round(old_min_conf  + hmm_adj["min_conf_delta"], 4)
                regime_ctx["size_mult"]  = round(old_size_mult * hmm_adj["size_mult"], 4)
                regime_ctx["hmm_regime"] = hmm_regime
            self.log.info(
                f"HMM regime: {hmm_regime} | "
                f"min_conf_delta={hmm_adj['min_conf_delta']:+.2f} "
                f"size_mult={hmm_adj['size_mult']:.2f}"
            )
            # Publish live strategy parameters for dashboard
            self.state.state["live_strategy"] = {
                "eff_min_conf":       regime_ctx.get("min_conf", self.min_conf) if regime_ctx else self.min_conf,
                "eff_size_mult":      regime_ctx.get("size_mult", 1.0) if regime_ctx else 1.0,
                "market_regime":      regime_ctx.get("regime", "UNKNOWN") if regime_ctx else "UNKNOWN",
                "hmm_regime":         hmm_regime,
                "profile":            self.profile.name,
                "base_min_conf":      self.min_conf,
                "htf_filter_mode":    self.htf_filter_mode,
                "timeframe":          self.config.get("ml", {}).get("timeframe", "15m"),
                "forward_bars":       self.config.get("ml", {}).get("forward_bars", 2),
                "min_votes":          self.config.get("strategy", {}).get("min_votes", 1),
                "updated_at":         datetime.now(timezone.utc).isoformat(),
            }
            self.state.save()
        except Exception as e:
            self.log.warning(f"HMM overlay failed: {e}")

        # RL execution layer: manages open trades, runs after ATR stops + HMM context ready
        self._rl_manage_trades(regime_ctx)
        open_trades = self.state.get_open_trades()
        balance     = self.get_usdt_balance()
        max_reached = len(open_trades) >= self.max_open

        # Pre-fetch 1h OHLCV for all symbols once — shared between GNN and analysis
        try:
            ohlcv_1h = {}
            def _fetch_1h(s):
                df = self.feed.fetch_ohlcv(s, "1h", limit=168)
                return s, df
            with ThreadPoolExecutor(max_workers=min(10, len(symbols))) as executor:
                for s, df in executor.map(_fetch_1h, symbols):
                    if df is not None and len(df) >= 24:
                        ohlcv_1h[s] = df

            # GNN correlation graph from pre-fetched data
            if len(ohlcv_1h) >= 2:
                self.gnn_filter.update_graph(ohlcv_1h)
                stats = self.gnn_filter.graph_stats()
                self.log.info(
                    f"GNN graph: {stats['n_nodes']} nodes "
                    f"{stats['n_edges']} edges "
                    f"avg_corr={stats['avg_corr']}"
                )
        except Exception as e:
            self.log.warning(f"GNN update failed: {e}")

        # Online learning: retrain with decay weights when enough new bars buffered
        try:
            ol_result = self.ai.incremental_update()
            if ol_result.get("status") == "updated":
                self.log.info(f"Online learning update: {ol_result.get('results', {})}")
        except Exception as e:
            self.log.warning(f"Online learning update failed: {e}")

        # Process symbols — pre-fetch OHLCV data in parallel, then analyze sequentially
        # This avoids 4-8 sequential API fetches per symbol (the main cycle bottleneck)
        from features.feature_builder import SUPPORTED_TFS
        tf_list = list(SUPPORTED_TFS)
        tf = (self.config.get("ml", {}).get("timeframe") or self.config.get("scanner", {}).get("timeframe", "15m"))
        train_tfs = self.config.get("training", {}).get("timeframes", ["15m", "1h", "4h", "1d"])

        # Pre-fetch multi-TF OHLCV for ALL symbols in parallel
        multi_cache = {}
        def _prefetch_multi(symbol):
            try:
                multi_tfs = [(t, 500 if t == tf else (300 if t in ("1h","30m") else 200)) for t in train_tfs]
                dfs = self.feed.fetch_multi_timeframe(symbol, timeframes=multi_tfs)
                return symbol, dfs
            except Exception as e:
                self.log.warning(f"Pre-fetch multi-TF failed for {symbol}: {e}")
                return symbol, None
        with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as ex:
            for s, dfs in ex.map(_prefetch_multi, symbols):
                if dfs:
                    multi_cache[s] = dfs

        # Pre-fetch training-feed OHLCV for ALL symbols and TFs in parallel
        train_cache = {}
        def _prefetch_train(symbol):
            try:
                tf_dfs = {}
                for t in tf_list:
                    tdf = self.training_feed.fetch_ohlcv(symbol, t, limit=2000)
                    if tdf is not None and len(tdf) >= 50:
                        tf_dfs[t] = tdf
                return symbol, tf_dfs
            except Exception as e:
                self.log.warning(f"Pre-fetch train failed for {symbol}: {e}")
                return symbol, None
                tdf = self.training_feed.fetch_ohlcv(symbol, t, limit=2000)
                if tdf is not None and len(tdf) >= 50:
                    tf_dfs[t] = tdf
            return symbol, tf_dfs
        with ThreadPoolExecutor(max_workers=min(8, len(symbols))) as ex:
            for s, tf_dfs in ex.map(_prefetch_train, symbols):
                if tf_dfs:
                    train_cache[s] = tf_dfs

        for symbol in symbols:
            if not can_trade or max_reached or trading_paused:
                try:
                    dfs = multi_cache.get(symbol) or self.feed.fetch_multi_timeframe(symbol)
                    if dfs and "1h" in dfs:
                        ml = self.ai.predict(dfs["1h"], symbol)
                        s  = self.agents.analyze(symbol, dfs["1h"], ml)
                        self.state.add_signal(s)
                except Exception:
                    pass
                continue

            self.analyze_symbol(symbol, balance, open_trades, regime_ctx, btc_1h_return,
                               pre_multi=multi_cache.get(symbol),
                               pre_train=train_cache.get(symbol))
            open_trades = self.state.get_open_trades()
            max_reached = len(open_trades) >= self.max_open


    def run(self):
        self.log.info("=" * 50)
        self.log.info(f"CRYPTOBOT v4 STARTED — MODE: {self.MODE.upper()}")
        self.log.info("=" * 50)

        last_train_week = datetime.now(timezone.utc).isocalendar()[1]
        alert_sent = False

        while True:
            try:
                current_week = datetime.now(timezone.utc).isocalendar()[1]
                if current_week != last_train_week:
                    self.log.info(f"Starting weekly retrain in background (week {current_week})")
                    threading.Thread(
                        target=self._with_training_heartbeat,
                        args=(lambda: self._train(quick=False), "weekly"),
                        daemon=True
                    ).start()
                    last_train_week = current_week

                current_coins = self.scanner.get_coins(self.exchange, invalid_symbols=self.feed.invalid_symbols)
                self._retrain_if_watchlist_changed(current_coins)

                self._check_for_retrain_request()

                if self.learner.should_run():
                    try:
                        self._with_training_heartbeat(lambda: self.learner.run_learning_cycle(), source="ai_agent")
                    except Exception as e:
                        self.log.error(f"AI agent learning error: {e}", exc_info=True)

                # Heartbeat: write liveness flag for dashboard detection
                self._write_heartbeat()

                # Check for command-file from dashboard (fallback when Docker socket unavailable)
                if self._check_bot_control():
                    self.log.info("Bot control stop signal received, shutting down...")
                    self.notifier.send_alert(
                        f"CryptoBot v4 Stopped (dashboard command)\n"
                        f"Mode: {self.MODE.upper()}"
                    )
                    break

                self.run_once()
                self._write_heartbeat()  # also write after cycle — halves gap to dashboard
                # Send startup alert only after first successful cycle — prevents spam during deploys
                if not alert_sent:
                    alert_sent = True
                    self.notifier.send_alert(
                        f"CryptoBot v4 Started\n"
                        f"Mode: {self.MODE.upper()}"
                    )
                self.notifier.exchange = self.exchange
                self.notifier.send_report(self.exchange)

            except Exception as e:
                self.log.error(f"Cycle error: {e}", exc_info=True)

            self.log.info(f"Sleeping {self.scan_interval}s...")
            time.sleep(self.scan_interval)

    def _write_heartbeat(self):
        """Write heartbeat file for dashboard liveness detection."""
        hb = DATA_DIR / f"bot_heartbeat_{self.MODE}.json"
        try:
            with open(hb, "w") as f:
                json.dump({"timestamp": datetime.now(timezone.utc).isoformat()}, f)
        except OSError:
            pass

    def _heartbeat_during_training(self, stop_event):
        while not stop_event.is_set():
            self._write_heartbeat()
            stop_event.wait(30)

    def _with_training_heartbeat(self, fn, source="unknown"):
        stop_event = threading.Event()
        hb_thread = threading.Thread(target=self._heartbeat_during_training, args=(stop_event,), daemon=True)
        hb_thread.start()
        self._training_in_progress = True
        self._write_training_status("running", source=source, started=datetime.now(timezone.utc).isoformat())
        t0 = time.time()
        try:
            fn()
            self._write_training_status("completed", source=source, duration_seconds=round(time.time() - t0, 1),
                                        completed=datetime.now(timezone.utc).isoformat())
        except Exception as e:
            self._write_training_status("error", source=source, error=str(e))
            raise
        finally:
            self._training_in_progress = False
            stop_event.set()
            hb_thread.join(timeout=2)

    def _check_bot_control(self) -> bool:
        """Check for control commands from dashboard. Returns True if should stop."""
        p = DATA_DIR / "bot_control.json"
        if not p.exists():
            return False
        try:
            with open(p) as f:
                cmd = json.load(f)
            target = cmd.get("mode", "all")
            if target not in (self.MODE, "all"):
                return False
            action = cmd.get("command", "")
            if action == "stop":
                p.unlink(missing_ok=True)
                return True
            p.unlink(missing_ok=True)
            return False
        except (json.JSONDecodeError, OSError):
            return False
