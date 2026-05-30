from __future__ import annotations

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    Small feed-forward Q-network used by RiskAgent.

    Input: numeric risk state vector.
    Output: Q-values for [KEEP_SIGNAL, DOWNGRADE_TO_HOLD, BLOCK_TRADE].
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
