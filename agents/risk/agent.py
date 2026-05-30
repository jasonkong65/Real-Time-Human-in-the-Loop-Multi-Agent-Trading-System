from __future__ import annotations

import json

import math

import random

import sqlite3

from datetime import datetime, timezone

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

import joblib

import pandas as pd

import torch

import torch.nn as nn

import torch.optim as optim

from .dqn import DQNNetwork
from .config import RiskConfigMixin
from .extractors import RiskExtractionMixin
from .state import RiskStateMixin
from .replay import RiskReplayMixin
from .dqn_policy import RiskDqnPolicyMixin
from .rules import RiskRulesMixin
from .feedback import RiskFeedbackMixin


class RiskAgent(RiskConfigMixin, RiskExtractionMixin, RiskStateMixin, RiskReplayMixin, RiskDqnPolicyMixin, RiskRulesMixin, RiskFeedbackMixin):
    """Strict DQN Risk Agent.

Role:
- Hard rule-based safety layer decides non-negotiable safety constraints.
- PyTorch DQN advisory layer learns risk actions from delayed paper rewards.
- Target network + replay memory make it a real DQN implementation, not only
  a DQN-style sklearn approximator.

Safety design:
- DQN is advisory. It can increase caution for risky buy setups, but it cannot
  create BLOCK_TRADE by itself unless hard safety already triggered a block.
- This keeps the project suitable for paper decision support, not real trading."""


    ACTIONS = ["KEEP_SIGNAL", "DOWNGRADE_TO_HOLD", "BLOCK_TRADE"]


    ACTION_INDEX = {name: idx for idx, name in enumerate(ACTIONS)}


    INDEX_ACTION = {idx: name for name, idx in ACTION_INDEX.items()}


    MODEL_SIGNALS = {
            "BUY_CANDIDATE": 1.0,
            "HOLD": 0.0,
            "SELL_RISK": -1.0,
            "BLOCKED": -0.5,
            "BUY_WATCHLIST_OVERBOUGHT": 0.6,
            "BUY_WATCHLIST_ENTRY_RISK": 0.6,
            "WATCHLIST_BULLISH_ENTRY_RISK": 0.6,
        }


    STATE_DIM = 13


    def __init__(
        self,
        dqn_model_path: str = "models/risk_dqn_model.pt",
        target_model_path: str = "models/risk_dqn_target_model.pt",
        replay_path: str = "data/risk_dqn_replay.csv",
        q_table_path: str = "models/risk_q_table.pkl",
        epsilon: float = 0.08,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        gamma: float = 0.90,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        min_replay_samples: int = 100,
        target_update_steps: int = 25,
        train_epochs_per_update: int = 4,
        hidden_dim: int = 64,
        seed: int = 42,
        config_path: Optional[str] = "config/risk_config.json",
        replay_db_path: str = "data/trading_system.db",
    ):
        config = self._load_runtime_config(config_path)
        epsilon = float(self._cfg(config, "epsilon", epsilon))
        epsilon_min = float(self._cfg(config, "epsilon_min", epsilon_min))
        epsilon_decay = float(self._cfg(config, "epsilon_decay", epsilon_decay))
        gamma = float(self._cfg(config, "gamma", gamma))
        learning_rate = float(self._cfg(config, "learning_rate", learning_rate))
        batch_size = int(self._cfg(config, "batch_size", batch_size))
        min_replay_samples = int(self._cfg(config, "min_replay_samples", min_replay_samples))
        target_update_steps = int(self._cfg(config, "target_update_steps", target_update_steps))
        train_epochs_per_update = int(self._cfg(config, "train_epochs_per_update", train_epochs_per_update))
        hidden_dim = int(self._cfg(config, "hidden_dim", hidden_dim))
        replay_db_path = str(self._cfg(config, "replay_db_path", replay_db_path))

        self.dqn_model_path = Path(dqn_model_path)
        self.target_model_path = Path(target_model_path)
        self.replay_path = Path(replay_path)
        self.q_table_path = Path(q_table_path)  # compatibility for older app/evaluator wording
        self.replay_db_path = Path(replay_db_path)

        self.dqn_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.target_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.replay_path.parent.mkdir(parents=True, exist_ok=True)
        self.q_table_path.parent.mkdir(parents=True, exist_ok=True)
        self.replay_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.loss_history_path = Path("data/risk_dqn_loss_history.csv")
        self.loss_history_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_training_loss: Optional[float] = None

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.min_replay_samples = min_replay_samples
        self.target_update_steps = target_update_steps
        self.train_epochs_per_update = train_epochs_per_update
        self.hidden_dim = hidden_dim
        self.seed = seed

        random.seed(seed)
        torch.manual_seed(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQNNetwork(self.STATE_DIM, len(self.ACTIONS), hidden_dim).to(self.device)
        self.target_net = DQNNetwork(self.STATE_DIM, len(self.ACTIONS), hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()
        self.training_steps = 0

        self.q_table = self._load_q_table_compat()
        self._init_replay_db()
        self._load_or_init_dqn()

