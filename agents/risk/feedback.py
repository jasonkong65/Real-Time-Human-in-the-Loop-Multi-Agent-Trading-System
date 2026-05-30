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


class RiskFeedbackMixin:


    def calculate_reward(self, final_signal: str, future_return: float, volatility_level: str = "Unknown") -> float:
        future_return = self._safe_float(future_return, 0.0) or 0.0
        final_signal = self._normalise_label(final_signal, "HOLD")

        if final_signal == "BUY_CANDIDATE":
            reward = future_return
        elif final_signal == "SELL_RISK":
            reward = -future_return
        elif final_signal == "HOLD":
            reward = -abs(future_return) * 0.15
        elif final_signal == "BLOCKED":
            reward = abs(future_return) * 0.30 if future_return < 0 else -future_return * 0.20
        else:
            reward = 0.0

        if str(volatility_level).title() in ["High", "Critical"]:
            reward -= 0.003
        return float(reward)


    def update_q_value(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Compatibility name for RewardAgent.
        Under strict DQN, this appends a transition to replay memory and trains
        the policy network only when enough replay samples exist.
        """
        if not state:
            return {"success": False, "summary": "Cannot update DQN because state is missing."}

        action = action if action in self.ACTIONS else "KEEP_SIGNAL"
        reward = self._safe_float(reward, 0.0) or 0.0
        state_vector = self._parse_state_string_to_vector(state)
        next_state_vector = self._parse_state_string_to_vector(next_state) if next_state else state_vector

        old_q = self._predict_q_values_from_vector(state_vector).get(action, 0.0)
        replay_count = self._append_replay(
            state_text=state,
            state_vector=state_vector,
            action=action,
            reward=reward,
            next_state_text=next_state,
            next_state_vector=next_state_vector,
            done=next_state is None,
        )
        train_result = self._train_dqn_from_replay()
        new_q = self._predict_q_values_from_vector(state_vector).get(action, old_q)
        q_values = self._predict_q_values(state_vector)
        self._record_state_q_values(state, q_values)

        return {
            "success": True,
            "learning_type": "strict_pytorch_dqn_replay_update",
            "state": state,
            "action": action,
            "reward": round(reward, 6),
            "old_q": round(old_q, 6),
            "new_q": round(new_q, 6),
            "replay_count": replay_count,
            "min_replay_samples": self.min_replay_samples,
            "dqn_model_path": str(self.dqn_model_path),
            "dqn_target_model_path": str(self.target_model_path),
            "dqn_replay_path": str(self.replay_path),
            "dqn_replay_db_path": str(self.replay_db_path),
            "dqn_replay_storage": "sqlite_primary_csv_mirror",
            "train_result": train_result,
            "summary": "Added feedback to DQN replay memory and trained the policy network when enough samples were available.",
        }


    def update_from_feedback(self, risk_result: Dict[str, Any], future_return: float) -> Dict[str, Any]:
        if not isinstance(risk_result, dict):
            return {"success": False, "summary": "Cannot update DQN because risk_result is invalid."}

        final_signal = risk_result.get("final_signal")
        risk_level = risk_result.get("risk_level")
        volatility = risk_result.get("volatility_level") or ("High" if risk_level in ["High", "Critical"] else "Low")
        reward = self.calculate_reward(final_signal, future_return, volatility)

        result = self.update_q_value(
            state=risk_result.get("q_state"),
            action=risk_result.get("risk_action"),
            reward=reward,
        )
        result.update(
            {
                "final_signal": final_signal,
                "risk_level": risk_level,
                "future_return": self._safe_float(future_return, 0.0),
                "calculated_reward": round(reward, 6),
            }
        )
        return result

