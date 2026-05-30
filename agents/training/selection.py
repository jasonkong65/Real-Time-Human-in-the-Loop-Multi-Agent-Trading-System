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


class TrainingSelectionMixin:

    """Mixin for model selection logic in the TrainingAgent, including methods to define candidate models with specific hyperparameters, perform walk-forward validation to evaluate models on multiple sequential train/test splits, calculate composite scores from multiple metrics, compute baseline scores for comparison, extract feature importance from trained models, and save metadata about training runs and model performance."""

    def _candidate_models(self, pooled_available: bool) -> List[Tuple[str, Any, Dict[str, Any], bool]]:
        candidates: List[Tuple[str, Any, Dict[str, Any], bool]] = [
            (
                "rf_conservative",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=4,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                {"family": "RandomForest", "n_estimators": 120, "max_depth": 4, "min_samples_leaf": 4},
                False,
            ),
            (
                "rf_balanced",
                RandomForestClassifier(
                    n_estimators=180,
                    max_depth=6,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                {"family": "RandomForest", "n_estimators": 180, "max_depth": 6, "min_samples_leaf": 3},
                False,
            ),
            (
                "rf_flexible",
                RandomForestClassifier(
                    n_estimators=240,
                    max_depth=None,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                {"family": "RandomForest", "n_estimators": 240, "max_depth": None, "min_samples_leaf": 5},
                False,
            ),
            (
                "extra_trees_stable",
                ExtraTreesClassifier(
                    n_estimators=220,
                    max_depth=6,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                {"family": "ExtraTrees", "n_estimators": 220, "max_depth": 6, "min_samples_leaf": 4},
                False,
            ),
        ]

        if pooled_available:
            candidates.extend(
                [
                    (
                        "pooled_rf_balanced",
                        RandomForestClassifier(
                            n_estimators=220,
                            max_depth=6,
                            min_samples_leaf=3,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1,
                        ),
                        {"family": "RandomForest", "n_estimators": 220, "max_depth": 6, "min_samples_leaf": 3, "scope": "pooled_local_cache"},
                        True,
                    ),
                    (
                        "pooled_extra_trees_stable",
                        ExtraTreesClassifier(
                            n_estimators=240,
                            max_depth=7,
                            min_samples_leaf=4,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1,
                        ),
                        {"family": "ExtraTrees", "n_estimators": 240, "max_depth": 7, "min_samples_leaf": 4, "scope": "pooled_local_cache"},
                        True,
                    ),
                ]
            )

        return candidates


    def _walk_forward_splits(self, n_rows: int) -> List[Tuple[int, int]]:
        min_train = max(80, int(n_rows * 0.40))
        test_size = max(20, int(n_rows * 0.12))
        splits: List[Tuple[int, int]] = []
        start = min_train
        while start + test_size <= n_rows:
            splits.append((start, start + test_size))
            start += test_size
        if not splits and n_rows > 110:
            split = int(n_rows * 0.75)
            splits.append((split, n_rows))
        return splits[-5:]


    def _score_from_metrics(self, metrics: Dict[str, float]) -> float:
        return (
            0.35 * metrics.get("balanced_accuracy", 0.0)
            + 0.35 * metrics.get("macro_f1", 0.0)
            + 0.20 * metrics.get("accuracy", 0.0)
            + 0.10 * metrics.get("sell_risk_recall", 0.0)
        )


    def _walk_forward_score(
        self,
        model,
        X_current: pd.DataFrame,
        y_current: pd.Series,
        X_pool: Optional[pd.DataFrame] = None,
        y_pool: Optional[pd.Series] = None,
        use_pooled: bool = False,
    ) -> Dict[str, Any]:
        splits = self._walk_forward_splits(len(X_current))
        if not splits:
            raise ValueError("Not enough rows for walk-forward validation.")

        y_true_all: List[str] = []
        y_pred_all: List[str] = []
        fold_rows: List[Dict[str, Any]] = []

        for fold_id, (train_end, test_end) in enumerate(splits, start=1):
            X_train = X_current.iloc[:train_end]
            y_train = y_current.iloc[:train_end]
            X_test = X_current.iloc[train_end:test_end]
            y_test = y_current.iloc[train_end:test_end]

            if use_pooled and X_pool is not None and y_pool is not None and len(X_pool) > 0:
                X_train = pd.concat([X_pool, X_train], ignore_index=True)
                y_train = pd.concat([y_pool, y_train], ignore_index=True)

            if y_train.nunique() < 2 or y_test.empty:
                continue

            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            pred = fold_model.predict(X_test)

            y_true = list(y_test.astype(str))
            y_pred = [str(x) for x in pred]
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

            fold_metrics = {
                "fold": fold_id,
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "sell_risk_recall": float(recall_score(y_true, y_pred, labels=["SELL_RISK"], average="macro", zero_division=0)),
            }
            fold_metrics["score"] = float(self._score_from_metrics(fold_metrics))
            fold_rows.append(fold_metrics)

        if not y_true_all:
            raise ValueError("No valid walk-forward folds were produced.")

        aggregate_metrics = {
            "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
            "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
            "sell_risk_recall": float(recall_score(y_true_all, y_pred_all, labels=["SELL_RISK"], average="macro", zero_division=0)),
            "num_folds": int(len(fold_rows)),
            "total_test_size": int(len(y_true_all)),
        }
        aggregate_metrics["score"] = float(self._score_from_metrics(aggregate_metrics))

        labels = list(self.CORE_SIGNALS)
        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
        report = classification_report(y_true_all, y_pred_all, labels=labels, zero_division=0, output_dict=True)

        return {
            "score": aggregate_metrics["score"],
            "accuracy": aggregate_metrics["accuracy"],
            "balanced_accuracy": aggregate_metrics["balanced_accuracy"],
            "macro_f1": aggregate_metrics["macro_f1"],
            "sell_risk_recall": aggregate_metrics["sell_risk_recall"],
            "num_folds": aggregate_metrics["num_folds"],
            "total_test_size": aggregate_metrics["total_test_size"],
            "folds": fold_rows,
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm.tolist(),
            },
            "classification_report": report,
            "pred_distribution": self._clean_label_dict(pd.Series(y_pred_all).value_counts().to_dict()),
            "true_distribution": self._clean_label_dict(pd.Series(y_true_all).value_counts().to_dict()),
        }


    def _baseline_score(self, y: pd.Series) -> Dict[str, Any]:
        splits = self._walk_forward_splits(len(y))
        y_true_all: List[str] = []
        y_pred_all: List[str] = []
        for train_end, test_end in splits:
            y_train = y.iloc[:train_end]
            y_test = y.iloc[train_end:test_end]
            if y_train.empty or y_test.empty:
                continue
            majority = str(y_train.mode().iloc[0])
            y_true_all.extend(list(y_test.astype(str)))
            y_pred_all.extend([majority] * len(y_test))
        if not y_true_all:
            return {"accuracy": None, "score": None}
        metrics = {
            "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
            "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
            "sell_risk_recall": float(recall_score(y_true_all, y_pred_all, labels=["SELL_RISK"], average="macro", zero_division=0)),
        }
        metrics["score"] = float(self._score_from_metrics(metrics))
        return metrics


    def _feature_importance(self, model) -> List[Dict[str, Any]]:
        if not hasattr(model, "feature_importances_"):
            return []
        rows = []
        for feature, importance in zip(self.feature_columns, model.feature_importances_):
            rows.append({"feature": feature, "importance": round(float(importance), 6)})
        return sorted(rows, key=lambda r: r["importance"], reverse=True)


    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        try:
            with self._metadata_path().open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass

