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


class RiskExtractionMixin:


    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            if isinstance(value, str) and value.lower() in ["none", "nan", "null"]:
                return default
            output = float(value)
            if math.isnan(output) or math.isinf(output):
                return default
            return output
        except Exception:
            return default


    def _clip(self, value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))


    def _get_nested(self, data: Dict[str, Any], keys: List[str], default=None):
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current


    def _normalise_label(self, value: Any, default: str = "UNKNOWN") -> str:
        if value is None:
            return default
        value = str(value).strip().upper()
        return value if value else default


    def _get_symbol(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        validation_result: Dict[str, Any],
    ) -> str:
        candidates = [
            signal_result.get("symbol") if isinstance(signal_result, dict) else None,
            self._get_nested(signal_result, ["signal_for_next_agent", "symbol"]),
            analysis_result.get("symbol") if isinstance(analysis_result, dict) else None,
            validation_result.get("symbol") if isinstance(validation_result, dict) else None,
            self._get_nested(validation_result, ["validation_for_next_agent", "symbol"]),
        ]
        for item in candidates:
            if item:
                return str(item).upper().strip()
        return "UNKNOWN"


    def _validation_score(self, validation_result: Dict[str, Any]) -> float:
        score = self._safe_float(validation_result.get("confidence_score"))
        if score is not None:
            return self._clip(score)
        confidence = str(validation_result.get("confidence", "Medium")).lower()
        return {"high": 1.0, "medium": 0.72, "low": 0.40}.get(confidence, 0.60)


    def _validation_confidence(self, validation_result: Dict[str, Any]) -> str:
        return str(validation_result.get("confidence", "Medium")).title()


    def _validation_action(self, validation_result: Dict[str, Any]) -> str:
        return str(validation_result.get("next_action", "ALLOW_ANALYSIS")).upper()


    def _model_signal(self, signal_result: Dict[str, Any]) -> str:
        value = (
            signal_result.get("model_signal")
            or signal_result.get("final_signal")
            or signal_result.get("display_signal")
            or self._get_nested(signal_result, ["signal_for_next_agent", "signal"])
            or "HOLD"
        )
        return self._normalise_label(value, "HOLD")


    def _model_confidence(self, signal_result: Dict[str, Any]) -> float:
        value = self._safe_float(signal_result.get("prediction_confidence"))
        if value is None:
            value = self._safe_float(self._get_nested(signal_result, ["signal_for_next_agent", "prediction_confidence"]))
        return self._clip(value if value is not None else 0.50)


    def _model_confidence_level(self, signal_result: Dict[str, Any]) -> str:
        level = signal_result.get("confidence_level") or self._get_nested(signal_result, ["signal_for_next_agent", "confidence_level"])
        if level:
            return str(level).title()
        conf = self._model_confidence(signal_result)
        if conf >= 0.66:
            return "High"
        if conf >= 0.45:
            return "Medium"
        return "Low"


    def _analyst_signal(self, analysis_result: Dict[str, Any]) -> str:
        return self._normalise_label(analysis_result.get("analyst_signal"), "NEUTRAL")


    def _analyst_score(self, analysis_result: Dict[str, Any]) -> float:
        return self._clip(self._safe_float(analysis_result.get("analyst_score"), 0.50) or 0.50)


    def _entry_risk(self, analysis_result: Dict[str, Any], signal_result: Dict[str, Any]) -> str:
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        value = (
            analysis_result.get("entry_risk_level")
            or stage2.get("entry_risk_level")
            or self._get_nested(signal_result, ["context_used", "entry_risk_level"])
            or "Medium"
        )
        return str(value).title()


    def _trend_direction(self, analysis_result: Dict[str, Any], signal_result: Dict[str, Any]) -> str:
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        value = (
            analysis_result.get("trend_direction")
            or stage2.get("trend_direction")
            or self._get_nested(signal_result, ["context_used", "trend_direction"])
            or "Neutral"
        )
        return str(value).title()


    def _volatility_level(self, analysis_result: Dict[str, Any]) -> str:
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        value = analysis_result.get("volatility_level") or stage2.get("volatility_level") or "Unknown"
        return str(value).title()


    def _feature_value(self, analysis_result: Dict[str, Any], key: str, default: float = 0.0) -> float:
        features = analysis_result.get("features_for_model", {}) if isinstance(analysis_result, dict) else {}
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        return self._safe_float(features.get(key, stage2.get(key, default)), default) or default

