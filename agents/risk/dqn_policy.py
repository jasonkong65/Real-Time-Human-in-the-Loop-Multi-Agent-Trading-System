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


class RiskDqnPolicyMixin:


    def _load_or_init_dqn(self) -> None:
        checkpoint_path = self.dqn_model_path
        if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                self.policy_net.load_state_dict(checkpoint.get("policy_state_dict", checkpoint))
                if "target_state_dict" in checkpoint:
                    self.target_net.load_state_dict(checkpoint["target_state_dict"])
                else:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                if "optimizer_state_dict" in checkpoint:
                    self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self.training_steps = int(checkpoint.get("training_steps", 0))
                self.epsilon = float(checkpoint.get("epsilon", self.epsilon))
                self.policy_net.eval()
                self.target_net.eval()
                return
            except Exception:
                pass
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()


    def _save_dqn(self) -> None:
        checkpoint = {
            "policy_state_dict": self.policy_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_steps": self.training_steps,
            "epsilon": self.epsilon,
            "state_dim": self.STATE_DIM,
            "actions": self.ACTIONS,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        torch.save(checkpoint, self.dqn_model_path)
        torch.save(self.target_net.state_dict(), self.target_model_path)


    def _predict_q_values_from_vector(self, vector: List[float]) -> Dict[str, float]:
        """
        Strict DQN inference.

        Q-values are always produced by the PyTorch policy network.
        We do not use hand-written heuristic Q-values during warm-up.
        If replay memory is still small, the network is simply untrained/warm-up,
        and the hard safety layer remains the final guardrail.
        """
        self.policy_net.eval()
        with torch.no_grad():
            state_tensor = torch.tensor([vector], dtype=torch.float32, device=self.device)
            q_values = self.policy_net(state_tensor).detach().cpu().numpy()[0].tolist()
        return {action: float(q_values[idx]) for action, idx in self.ACTION_INDEX.items()}


    def _predict_q_values(
        self,
        vector: List[float],
        validation_result: Optional[Dict[str, Any]] = None,
        analysis_result: Optional[Dict[str, Any]] = None,
        signal_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        # Keep unused compatibility parameters so old app.py calls still work.
        return self._predict_q_values_from_vector(vector)


    def _choose_dqn_action(self, q_values: Dict[str, float]) -> str:
        if random.random() < self.epsilon:
            # Do not randomly explore BLOCK_TRADE in live paper decision mode.
            return random.choice(["KEEP_SIGNAL", "DOWNGRADE_TO_HOLD"])
        return max(q_values, key=q_values.get)


    def _train_dqn_from_replay(self) -> Dict[str, Any]:
        df = self._read_replay()
        replay_count = len(df)
        if replay_count < self.min_replay_samples:
            return {
                "success": False,
                "trained": False,
                "replay_count": replay_count,
                "min_replay_samples": self.min_replay_samples,
                "summary": f"DQN not trained yet. Need at least {self.min_replay_samples} replay samples.",
            }

        losses = []
        self.policy_net.train()
        self.target_net.eval()

        for _ in range(self.train_epochs_per_update):
            batch = self._sample_replay_batch(df)
            if batch is None:
                continue
            states, actions, rewards, next_states, dones = batch

            q_values = self.policy_net(states).gather(1, actions)
            with torch.no_grad():
                next_q_values = self.target_net(next_states).max(dim=1, keepdim=True)[0]
                target_q = rewards + self.gamma * next_q_values * (1.0 - dones)

            loss = self.loss_fn(q_values, target_q)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            self.training_steps += 1

            if self.training_steps % self.target_update_steps == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.policy_net.eval()
        self.target_net.eval()
        self._save_dqn()

        avg_loss = round(sum(losses) / len(losses), 6) if losses else None
        self.last_training_loss = avg_loss
        self._record_training_loss(avg_loss, replay_count)

        return {
            "success": True,
            "trained": True,
            "replay_count": replay_count,
            "training_steps": self.training_steps,
            "target_update_steps": self.target_update_steps,
            "epsilon": round(self.epsilon, 5),
            "loss": avg_loss,
            "recent_loss": avg_loss,
            "summary": "DQN policy network updated from replay memory.",
        }

