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


class RiskRulesMixin:

    """Mixin for implementing hard rule-based safety checks to determine non-negotiable safety actions based on validation results, analysis results, and signal results, as well as methods to combine these with DQN advisory actions and produce final risk assessments and interpretations."""

    def _hard_safety_action(
        self,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any],
    ) -> Tuple[str, List[str]]:
        model_signal = self._model_signal(signal_result)
        validation_score = self._validation_score(validation_result)
        validation_action = self._validation_action(validation_result)
        model_conf = self._model_confidence(signal_result)
        entry_risk = self._entry_risk(analysis_result, signal_result)
        volatility = self._volatility_level(analysis_result)

        reasons = []
        if validation_action == "BLOCK_ANALYSIS":
            reasons.append("data validation blocked the analysis")
            return "BLOCK_TRADE", reasons

        # Only hard-block when the validation layer explicitly blocks the analysis
        # or when the data confidence is extremely weak. A normal "Low" confidence
        # result should usually become a cautious watchlist / hold decision rather
        # than a critical block, especially when the trend and analyst view are still
        # constructive.
        if validation_score < 0.25:
            reasons.append("data confidence is too weak")
            return "BLOCK_TRADE", reasons

        if model_signal == "BUY_CANDIDATE" and validation_score < 0.50:
            reasons.append("buy signal has weak data support")
            return "DOWNGRADE_TO_HOLD", reasons
        if model_signal == "BUY_CANDIDATE" and model_conf < 0.42:
            reasons.append("buy signal has low model confidence")
            return "DOWNGRADE_TO_HOLD", reasons
        if model_signal == "BUY_CANDIDATE" and entry_risk == "High":
            reasons.append("entry timing risk is high")
            return "DOWNGRADE_TO_HOLD", reasons
        if model_signal == "BUY_CANDIDATE" and volatility in ["High", "Critical"]:
            reasons.append("volatility is high")
            return "DOWNGRADE_TO_HOLD", reasons
        reasons.append("no hard safety block was triggered")
        return "KEEP_SIGNAL", reasons


    def _filter_dqn_action(self, dqn_action: str, hard_action: str, model_signal: str) -> str:
        if hard_action == "BLOCK_TRADE":
            return "BLOCK_TRADE"
        if dqn_action == "BLOCK_TRADE":
            # DQN cannot hard-block by itself in the paper decision-support system.
            return "DOWNGRADE_TO_HOLD" if model_signal == "BUY_CANDIDATE" else "KEEP_SIGNAL"
        if dqn_action == "DOWNGRADE_TO_HOLD" and model_signal != "BUY_CANDIDATE":
            return "KEEP_SIGNAL"
        return dqn_action if dqn_action in self.ACTIONS else "KEEP_SIGNAL"


    def _combine_actions(self, hard_action: str, dqn_action: str) -> str:
        priority = {"KEEP_SIGNAL": 0, "DOWNGRADE_TO_HOLD": 1, "BLOCK_TRADE": 2}
        return hard_action if priority.get(hard_action, 0) >= priority.get(dqn_action, 0) else dqn_action


    def _apply_action(self, model_signal: str, action: str) -> str:
        if action == "BLOCK_TRADE":
            return "BLOCKED"
        if action == "DOWNGRADE_TO_HOLD" and model_signal == "BUY_CANDIDATE":
            return "HOLD"
        return model_signal if model_signal in ["BUY_CANDIDATE", "HOLD", "SELL_RISK"] else "HOLD"


    def _risk_level(
        self,
        final_signal: str,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any],
        action: str,
    ) -> Tuple[str, str]:
        if action == "BLOCK_TRADE":
            return "Critical", "Data quality or safety rules blocked the result."

        points = 0
        notes = []
        validation_score = self._validation_score(validation_result)
        model_conf = self._model_confidence(signal_result)
        entry_risk = self._entry_risk(analysis_result, signal_result)
        volatility = self._volatility_level(analysis_result)
        trend = self._trend_direction(analysis_result, signal_result)

        if validation_score < 0.55:
            points += 2
            notes.append("data confidence is limited")
        if model_conf < 0.45:
            points += 1
            notes.append("model confidence is not strong")
        if entry_risk == "High":
            points += 2
            notes.append("entry timing risk is high")
        elif entry_risk == "Medium":
            points += 1
            notes.append("entry timing risk is moderate")
        if volatility == "High":
            points += 2
            notes.append("volatility is high")
        if final_signal == "SELL_RISK":
            points += 2
            notes.append("the final signal points to downside risk")
        if trend == "Positive" and entry_risk in ["Medium", "High"] and final_signal == "HOLD":
            notes.append("main risk is chasing after a strong move")

        if points >= 5:
            level = "High"
        elif points >= 3:
            level = "Medium"
        else:
            level = "Low"
        return level, "; ".join(notes) if notes else "No major risk flag was detected."


    def assess_risk(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        validation_result = validation_result or {}
        analysis_result = analysis_result or {}
        signal_result = signal_result or {}

        symbol = self._get_symbol(signal_result, analysis_result, validation_result)
        model_signal = self._model_signal(signal_result)
        vector = self._state_vector(validation_result, analysis_result, signal_result)
        state = self._state_string(symbol, validation_result, analysis_result, signal_result)

        hard_action, hard_reasons = self._hard_safety_action(validation_result, analysis_result, signal_result)
        q_values = self._predict_q_values(vector, validation_result, analysis_result, signal_result)
        raw_dqn_action = self._choose_dqn_action(q_values)
        filtered_dqn_action = self._filter_dqn_action(raw_dqn_action, hard_action, model_signal)
        final_action = self._combine_actions(hard_action, filtered_dqn_action)
        final_signal = self._apply_action(model_signal, final_action)
        risk_level, risk_interpretation = self._risk_level(final_signal, validation_result, analysis_result, signal_result, final_action)
        self._record_state_q_values(state, q_values)

        replay_count = len(self._read_replay())
        dqn_status = "active" if replay_count >= self.min_replay_samples else "warmup"

        reasoning_steps = [
            f"Model signal: {model_signal}.",
            f"Validation confidence: {self._validation_confidence(validation_result)}.",
            f"Analyst signal: {self._analyst_signal(analysis_result)}.",
            f"Trend direction: {self._trend_direction(analysis_result, signal_result)}.",
            f"Entry timing risk: {self._entry_risk(analysis_result, signal_result)}.",
            f"Hard safety action: {hard_action} ({'; '.join(hard_reasons)}).",
            f"DQN status: {dqn_status}; replay samples: {replay_count}/{self.min_replay_samples}.",
            f"DQN advisory action: {raw_dqn_action}; after safety filter: {filtered_dqn_action}.",
            f"Final risk action: {final_action}; final signal: {final_signal}.",
        ]

        if final_action == "BLOCK_TRADE":
            decision = "The result was blocked because the safety layer found a serious data or risk issue."
        elif final_action == "DOWNGRADE_TO_HOLD":
            decision = "The signal was softened to HOLD because the setup needs more confirmation."
        else:
            decision = "The risk layer kept the model signal."

        return {
            "success": True,
            "agent": "Risk Agent",
            "agent_goal": "Apply hard safety checks and strict DQN risk advisory.",
            "symbol": symbol,
            "original_signal": model_signal,
            "risk_action": final_action,
            "hard_safety_action": hard_action,
            "dqn_action": raw_dqn_action,
            "filtered_dqn_action": filtered_dqn_action,
            "dqn_q_values": {k: round(v, 5) for k, v in q_values.items()},
            "dqn_status": dqn_status,
            "dqn_framework": "PyTorch DQN with replay memory and target network",
            "dqn_model_path": str(self.dqn_model_path),
            "dqn_target_model_path": str(self.target_model_path),
            "dqn_replay_path": str(self.replay_path),
            "dqn_replay_db_path": str(self.replay_db_path),
            "dqn_replay_storage": "sqlite_primary_csv_mirror",
            "dqn_replay_count": replay_count,
            "dqn_min_replay_samples": self.min_replay_samples,
            "epsilon": round(self.epsilon, 5),
            "recent_dqn_loss": self._recent_loss(),
            "dqn_training_ready": replay_count >= self.min_replay_samples,
            "dqn_diagnostics": {
                "q_values": {k: round(v, 5) for k, v in q_values.items()},
                "epsilon": round(self.epsilon, 5),
                "replay_count": replay_count,
                "min_replay_samples": self.min_replay_samples,
                "recent_loss": self._recent_loss(),
                "training_steps": self.training_steps,
                "target_update_steps": self.target_update_steps,
                "status": dqn_status,
            },
            "q_state": state,
            "final_signal": final_signal,
            "risk_level": risk_level,
            "risk_interpretation": risk_interpretation,
            "volatility_level": self._volatility_level(analysis_result),
            "entry_risk_level": self._entry_risk(analysis_result, signal_result),
            "agent_decision": decision,
            "reasoning_steps": reasoning_steps,
            "risk_for_next_agent": {
                "symbol": symbol,
                "original_signal": model_signal,
                "final_signal": final_signal,
                "risk_level": risk_level,
                "risk_action": final_action,
                "dqn_action": filtered_dqn_action,
                "dqn_status": dqn_status,
                "risk_interpretation": risk_interpretation,
                "explanation_for_llm": f"Risk level is {risk_level}. {risk_interpretation}",
            },
            "summary": f"Risk Agent set final signal to {final_signal} with {risk_level} risk.",
        }


    def apply_risk_control(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.assess_risk(signal_result, analysis_result, validation_result)


    def adjust_risk(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.assess_risk(signal_result, analysis_result, validation_result)


    def evaluate_risk(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.assess_risk(signal_result, analysis_result, validation_result)


    def control_risk(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.assess_risk(signal_result, analysis_result, validation_result)


    def run(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.assess_risk(signal_result, analysis_result, validation_result)

