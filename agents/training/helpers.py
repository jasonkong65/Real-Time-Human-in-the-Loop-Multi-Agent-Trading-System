from __future__ import annotations

from __future__ import annotations

import json

from datetime import datetime, timezone

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

import joblib

import pandas as pd

from sklearn.base import clone

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)


class TrainingHelpersMixin:

    """Mixin for helper methods in the TrainingAgent, including methods for time handling, model path management, safe type conversion, label cleaning, model age calculation, and auto-retraining logic based on model version and age."""

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


    def _now_dt(self) -> datetime:
        return datetime.now(timezone.utc)


    def _set_symbol_model_path(self, symbol: Optional[str] = None) -> None:
        if symbol:
            clean = str(symbol).upper().strip()
            self.model_path = Path("models") / f"signal_model_{clean}.pkl"
        else:
            self.model_path = self.base_model_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)


    def _metadata_path(self) -> Path:
        return self.model_path.with_suffix(".metadata.json")


    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default


    def _round_float(self, value: Any, digits: int = 4):
        try:
            return round(float(value), digits)
        except Exception:
            return value


    def _clean_label_dict(self, values: Dict[Any, Any]) -> Dict[str, Any]:
        return {str(k): int(v) if isinstance(v, (int, float)) and float(v).is_integer() else v for k, v in values.items()}


    def _model_age_days(self, bundle: Optional[Dict[str, Any]]) -> Optional[float]:
        if not bundle:
            return None
        trained_at = bundle.get("trained_at_utc") or bundle.get("created_at_utc")
        if not trained_at:
            return None
        try:
            dt = datetime.strptime(str(trained_at), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            return (self._now_dt() - dt).total_seconds() / 86400
        except Exception:
            return None


    def _should_auto_retrain(self, bundle: Optional[Dict[str, Any]], force_retrain: bool) -> bool:
        if force_retrain:
            return True
        if not bundle:
            return True
        if bundle.get("model_version") != self.MODEL_VERSION:
            return True
        age = self._model_age_days(bundle)
        if age is None:
            return True
        return age >= self.auto_retrain_days

