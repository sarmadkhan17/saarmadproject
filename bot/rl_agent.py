"""
RL Trade Manager — DQN execution layer.
Manages OPEN trades only. Never decides entries.
Actions: HOLD | SCALE_IN | SCALE_OUT | CLOSE
Falls back to HOLD (existing ATR logic) on any failure or when untrained.
"""

import json
import logging
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
from collections import deque
from datetime import datetime
from typing import Dict, Optional, Tuple

from env_config import DATA_DIR

log  = logging.getLogger("RLAgent")
DATA = DATA_DIR

ACTIONS   = ["HOLD", "SCALE_IN", "SCALE_OUT", "CLOSE"]
N_ACTIONS = 4
STATE_DIM = 8

REGIME_IDX = {"TRENDING": 0, "RANGING": 1, "HIGH_VOL": 2, "CRASH": 3}


# ── State encoding ────────────────────────────────────────────────────────────

def build_state(
    trade: dict,
    current_price: float,
    atr: float,
    regime: str = "RANGING",
    price_1h_ago: float = 0.0,
) -> np.ndarray:
    """
    8-dim state vector, all features in [-1, 1]:
      [pnl_pct, volatility, time_in_trade, regime_oh×4, momentum_1h]
    """
    entry   = float(trade.get("price", current_price) or current_price) or 1.0
    is_long = trade.get("side", "buy") in ("buy", "long")

    raw_pnl = (current_price - entry) / entry
    if not is_long:
        raw_pnl = -raw_pnl
    pnl_pct  = float(np.clip(raw_pnl * 10, -1.0, 1.0))

    vol_feat = float(np.clip((atr / entry) * 50, 0.0, 1.0))

    try:
        ts  = trade.get("timestamp", datetime.utcnow().isoformat())
        hrs = max(0.0, (datetime.utcnow() - datetime.fromisoformat(ts)).total_seconds() / 3600)
    except Exception:
        hrs = 0.0
    time_feat = float(np.clip(np.log1p(hrs) / np.log1p(48), 0.0, 1.0))

    regime_oh          = [0.0, 0.0, 0.0, 0.0]
    regime_oh[REGIME_IDX.get(regime, 1)] = 1.0

    mom = 0.0
    if price_1h_ago and price_1h_ago > 0:
        mom = float(np.clip((current_price - price_1h_ago) / price_1h_ago * 10, -1.0, 1.0))

    return np.array([pnl_pct, vol_feat, time_feat, *regime_oh, mom], dtype=np.float32)


# ── DQN network ───────────────────────────────────────────────────────────────

class DQNNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 64), nn.ReLU(),
            nn.Linear(64, 32),        nn.ReLU(),
            nn.Linear(32, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity: int = 5000):
        self.buf = deque(maxlen=capacity)

    def push(self, s, a, r, s2, done):
        self.buf.append((s, a, r, s2, float(done)))

    def sample(self, n: int):
        batch  = random.sample(self.buf, n)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)),
            torch.LongTensor(np.array(a)),
            torch.FloatTensor(np.array(r)),
            torch.FloatTensor(np.array(s2)),
            torch.FloatTensor(np.array(d)),
        )

    def __len__(self):
        return len(self.buf)


# ── RL Trade Manager ─────────────────────────────────────────────────────────

class RLTradeManager:
    """
    DQN-based execution layer for open trades.
    Learns online from closed-trade PnL. Safe before trained (returns HOLD).
    """

    BATCH_SIZE    = 32
    GAMMA         = 0.99
    LR            = 1e-3
    EPS_START     = 1.0
    EPS_END       = 0.15
    EPS_DECAY     = 0.995
    TARGET_SYNC   = 50       # steps between target-net sync
    MIN_EXPERIENCES = 64     # minimum buffer size before any non-HOLD action

    def __init__(self):
        self.model_path  = DATA / "rl_agent.pt"
        self.meta_path   = DATA / "rl_agent_meta.json"
        self.buffer_path = DATA / "rl_buffer.pkl"

        self.q_net      = DQNNet()
        self.target_net = DQNNet()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.LR)
        self.buffer    = ReplayBuffer()
        self.epsilon   = self.EPS_START
        self.n_steps   = 0
        self.metadata: Dict = {}

        # trade_id → (state_vec, action_idx, entry_price_at_decision)
        self._pending: Dict[str, Tuple[np.ndarray, int, float]] = {}

        self._load()

    # ── Persistence ───────────────────────────────────────────────────────────

    def _load(self):
        if self.model_path.exists():
            try:
                ckpt            = torch.load(str(self.model_path), map_location="cpu",
                                             weights_only=False)
                self.q_net.load_state_dict(ckpt["q_net"])
                self.target_net.load_state_dict(ckpt["target_net"])
                self.epsilon    = ckpt.get("epsilon", self.EPS_END)
                self.n_steps    = ckpt.get("n_steps", 0)
                if self.meta_path.exists():
                    with open(self.meta_path) as f:
                        self.metadata = json.load(f)
                log.info(f"RL agent loaded (eps={self.epsilon:.3f} steps={self.n_steps})")
            except Exception as e:
                log.warning(f"RL agent load failed: {e}")
        if self.buffer_path.exists():
            try:
                self.buffer = joblib.load(self.buffer_path)
                log.info(f"RL replay buffer: {len(self.buffer)} experiences")
            except Exception:
                pass

    def _save(self):
        try:
            torch.save({
                "q_net":      self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon":    self.epsilon,
                "n_steps":    self.n_steps,
            }, str(self.model_path))
            self.metadata.update({
                "epsilon":   round(self.epsilon, 4),
                "n_steps":   self.n_steps,
                "buffer_len": len(self.buffer),
                "saved_at":  datetime.utcnow().isoformat(),
            })
            with open(self.meta_path, "w") as f:
                json.dump(self.metadata, f, indent=2)
            if self.n_steps % 50 == 0:
                joblib.dump(self.buffer, self.buffer_path)
        except Exception as e:
            log.warning(f"RL save failed: {e}")

    # ── Decision ─────────────────────────────────────────────────────────────

    def decide(
        self,
        trade: dict,
        current_price: float,
        atr: float,
        regime: str = "RANGING",
        price_1h_ago: float = 0.0,
    ) -> str:
        """
        Returns action string. Falls back to HOLD on failure or insufficient training.
        SCALE_IN only allowed after MIN_EXPERIENCES.
        """
        try:
            state = build_state(trade, current_price, atr, regime, price_1h_ago)

            if len(self.buffer) < self.MIN_EXPERIENCES:
                self._pending[trade["id"]] = (state, 0, current_price)  # HOLD
                return "HOLD"

            # Reduced epsilon for live trading (don't explore aggressively with real money)
            live_eps = self.epsilon * 0.2
            if random.random() < live_eps:
                action_idx = random.randint(0, N_ACTIONS - 1)
            else:
                self.q_net.eval()
                with torch.no_grad():
                    action_idx = int(
                        self.q_net(torch.FloatTensor(state).unsqueeze(0)).argmax().item()
                    )

            self._pending[trade["id"]] = (state, action_idx, current_price)
            return ACTIONS[action_idx]

        except Exception as e:
            log.error(f"RL decide error: {e}")
            return "HOLD"

    # ── Learning ─────────────────────────────────────────────────────────────

    def record_step(
        self,
        trade_id: str,
        next_price: float,
        next_atr: float,
        done: bool,
        final_pnl: float = 0.0,
    ):
        """
        Record transition for trade_id from last cycle.
        Called each scan cycle per open trade, and on trade close.
        """
        if trade_id not in self._pending:
            return
        prev_state, action_idx, prev_price = self._pending[trade_id]

        if done:
            reward     = float(np.clip(final_pnl / 5.0, -2.0, 2.0))
            next_state = np.zeros(STATE_DIM, dtype=np.float32)
            del self._pending[trade_id]
        else:
            price_chg  = (next_price - prev_price) / (prev_price + 1e-9)
            reward     = float(np.clip(price_chg * 3, -0.1, 0.1))
            next_state = np.zeros(STATE_DIM, dtype=np.float32)  # approximate

        self.buffer.push(prev_state, action_idx, reward, next_state, float(done))
        self.n_steps += 1
        self.epsilon  = max(self.EPS_END, self.epsilon * self.EPS_DECAY)

        if len(self.buffer) >= self.BATCH_SIZE and self.n_steps % 4 == 0:
            self._train_step()

        if done:
            self._save()

    def record_external_close(self, trade_id: str, final_pnl: float):
        """
        Called when a trade is closed by ATR stop / TP (not by RL).
        Provides terminal reward signal even for non-RL-initiated closes.
        """
        if trade_id in self._pending:
            prev_state, action_idx, prev_price = self._pending[trade_id]
            reward     = float(np.clip(final_pnl / 5.0, -2.0, 2.0))
            next_state = np.zeros(STATE_DIM, dtype=np.float32)
            self.buffer.push(prev_state, action_idx, reward, next_state, 1.0)
            del self._pending[trade_id]
            self.n_steps += 1
            if len(self.buffer) >= self.BATCH_SIZE:
                self._train_step()
            self._save()

    def _train_step(self):
        if len(self.buffer) < self.BATCH_SIZE:
            return
        s, a, r, s2, d = self.buffer.sample(self.BATCH_SIZE)

        self.q_net.train()
        current_q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            max_next_q = self.target_net(s2).max(1)[0]
            target_q   = r + self.GAMMA * max_next_q * (1 - d)

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        if self.n_steps % self.TARGET_SYNC == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            self.target_net.eval()
