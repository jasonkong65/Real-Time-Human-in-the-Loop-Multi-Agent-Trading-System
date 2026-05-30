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


class RiskStateMixin:


    def _risk_numeric(self, label: str) -> float:
        label = str(label).lower()
        return {
            "low": 0.20,
            "medium": 0.55,
            "moderate": 0.55,
            "high": 0.85,
            "critical": 1.0,
            "unknown": 0.50,
        }.get(label, 0.50)


    def _trend_numeric(self, trend: str) -> float:
        trend = str(trend).lower()
        if "strong positive" in trend:
            return 1.0
        if "positive" in trend:
            return 0.65
        if "strong negative" in trend:
            return -1.0
        if "negative" in trend:
            return -0.65
        return 0.0


    def _state_vector(
        self,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any],
    ) -> List[float]:
        model_signal = self._model_signal(signal_result)
        analyst_signal = self._analyst_signal(analysis_result)
        entry_risk = self._entry_risk(analysis_result, signal_result)
        trend = self._trend_direction(analysis_result, signal_result)
        volatility_level = self._volatility_level(analysis_result)

        rsi = self._feature_value(analysis_result, "rsi_14", 50.0)
        return_5 = self._feature_value(analysis_result, "return_5", 0.0)
        return_20 = self._feature_value(analysis_result, "return_20", 0.0)
        ma_gap = self._feature_value(analysis_result, "ma_gap", 0.0)
        volatility_20 = self._feature_value(analysis_result, "volatility_20", 0.0)

        return [
            self._validation_score(validation_result),
            self._model_confidence(signal_result),
            self._analyst_score(analysis_result),
            self.MODEL_SIGNALS.get(model_signal, 0.0),
            1.0 if ("BULLISH" in analyst_signal or "POSITIVE" in analyst_signal) else 0.0,
            1.0 if ("BEARISH" in analyst_signal or analyst_signal == "SELL_RISK") else 0.0,
            self._risk_numeric(entry_risk),
            self._risk_numeric(volatility_level),
            self._trend_numeric(trend),
            self._clip(rsi / 100.0),
            max(-1.0, min(1.0, return_5)),
            max(-1.0, min(1.0, return_20)),
            max(-1.0, min(1.0, ma_gap * 5.0 + volatility_20)),
        ]


    def _state_string(
        self,
        symbol: str,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any],
    ) -> str:
        parts = {
            "symbol": symbol,
            "validation": self._validation_confidence(validation_result),
            "model": self._model_signal(signal_result),
            "model_conf": self._model_confidence_level(signal_result),
            "analyst": self._analyst_signal(analysis_result),
            "trend": self._trend_direction(analysis_result, signal_result),
            "entry_risk": self._entry_risk(analysis_result, signal_result),
            "volatility": self._volatility_level(analysis_result),
        }
        return "|".join(f"{k}={v}" for k, v in parts.items())


    def _parse_state_string_to_vector(self, state: str) -> List[float]:
        """
        Compatibility path for delayed RewardAgent updates.
        It reconstructs a reasonable vector from the stored q_state string.
        """
        parsed = {}
        for part in str(state or "").split("|"):
            if "=" in part:
                key, value = part.split("=", 1)
                parsed[key.strip()] = value.strip()

        validation = parsed.get("validation", "Medium").title()
        model = parsed.get("model", "HOLD").upper()
        model_conf = parsed.get("model_conf", "Medium").title()
        analyst = parsed.get("analyst", "NEUTRAL").upper()
        trend = parsed.get("trend", "Neutral").title()
        entry_risk = parsed.get("entry_risk", "Medium").title()
        volatility = parsed.get("volatility", "Unknown").title()

        validation_score = {"High": 1.0, "Medium": 0.72, "Low": 0.40}.get(validation, 0.60)
        model_conf_score = {"High": 0.80, "Medium": 0.55, "Low": 0.32}.get(model_conf, 0.50)
        analyst_score = 0.70 if ("BULLISH" in analyst or "POSITIVE" in analyst) else 0.35 if "BEARISH" in analyst else 0.50

        return [
            validation_score,
            model_conf_score,
            analyst_score,
            self.MODEL_SIGNALS.get(model, 0.0),
            1.0 if ("BULLISH" in analyst or "POSITIVE" in analyst) else 0.0,
            1.0 if ("BEARISH" in analyst or analyst == "SELL_RISK") else 0.0,
            self._risk_numeric(entry_risk),
            self._risk_numeric(volatility),
            self._trend_numeric(trend),
            0.50,
            0.0,
            0.0,
            0.0,
        ]

