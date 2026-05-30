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


class TrainingDriftMixin:


    def _feature_snapshot(self, X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Small feature-distribution snapshot for drift checks."""
        snapshot: Dict[str, Dict[str, float]] = {}
        try:
            for col in self.feature_columns:
                if col in X.columns:
                    series = pd.to_numeric(X[col], errors="coerce").dropna()
                    if not series.empty:
                        snapshot[col] = {
                            "mean": float(series.mean()),
                            "std": float(series.std(ddof=0)),
                        }
        except Exception:
            return {}
        return snapshot


    def _feature_drift_report(self, existing_bundle: Optional[Dict[str, Any]], current_snapshot: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compare current feature snapshot with the previous saved model snapshot."""
        previous = {}
        if isinstance(existing_bundle, dict):
            previous = existing_bundle.get("feature_snapshot") or existing_bundle.get("training_feature_snapshot") or {}
        rows = []
        max_abs_z = 0.0
        for feature, cur in current_snapshot.items():
            old = previous.get(feature, {}) if isinstance(previous, dict) else {}
            old_mean = old.get("mean")
            old_std = old.get("std") or 0.0
            cur_mean = cur.get("mean")
            if old_mean is None or cur_mean is None:
                continue
            denom = max(abs(float(old_std)), 1e-6)
            z = float(cur_mean - old_mean) / denom
            max_abs_z = max(max_abs_z, abs(z))
            rows.append({
                "feature": feature,
                "previous_mean": round(float(old_mean), 6),
                "current_mean": round(float(cur_mean), 6),
                "z_like_shift": round(z, 4),
            })
        if not rows:
            label = "No baseline"
        elif max_abs_z >= 2.0:
            label = "High"
        elif max_abs_z >= 1.0:
            label = "Medium"
        else:
            label = "Low"
        return {"drift_level": label, "max_abs_z_like_shift": round(max_abs_z, 4), "details": rows[:10]}


    def _record_training_run_to_storage(self, symbol: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Best-effort SQLite logging; training should never fail because storage failed."""
        try:
            from agents.storage_agent import StorageAgent
            return StorageAgent().record_training_run(run_id=None, symbol=symbol, training_result=result)
        except Exception as exc:
            return {"success": False, "error": str(exc), "summary": "Training result was not written to SQLite."}

