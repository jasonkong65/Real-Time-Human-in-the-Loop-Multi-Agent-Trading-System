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


class TrainingWorkflowMixin:

    """Mixin for the overall training workflow in the TrainingAgent, including methods to automatically train and select models based on historical data, load existing models, decide when to retrain, and provide various interfaces for training from different data sources. The core method _auto_train orchestrates the entire process of feature engineering, model evaluation, selection, and saving, while other methods handle loading and decision logic."""

    def _auto_train(
        self,
        historical_data: Dict[str, Any],
        symbol: Optional[str],
        validation_confidence_score: float = 0.95,
        existing_bundle: Optional[Dict[str, Any]] = None,
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        symbol_clean = str(symbol or historical_data.get("symbol", "UNKNOWN")).upper().strip()
        X_current, y_current, data_info = self._build_single_dataset(historical_data, validation_confidence_score)
        current_feature_snapshot = self._feature_snapshot(X_current)
        feature_drift = self._feature_drift_report(existing_bundle, current_feature_snapshot)
        X_pool, y_pool, pooled_info = self._load_pooled_local_datasets(symbol_clean, validation_confidence_score)
        pooled_available = X_pool is not None and y_pool is not None and len(X_pool) >= 80

        baseline = self._baseline_score(y_current)
        evaluation_table: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None

        for name, model, params, use_pooled in self._candidate_models(pooled_available=pooled_available):
            try:
                score_info = self._walk_forward_score(
                    model,
                    X_current,
                    y_current,
                    X_pool=X_pool,
                    y_pool=y_pool,
                    use_pooled=use_pooled,
                )
                row = {
                    "candidate": name,
                    "training_scope": "pooled_local_cache" if use_pooled else "single_symbol",
                    "score": round(score_info["score"], 6),
                    "accuracy": round(score_info["accuracy"], 6),
                    "balanced_accuracy": round(score_info["balanced_accuracy"], 6),
                    "macro_f1": round(score_info["macro_f1"], 6),
                    "sell_risk_recall": round(score_info["sell_risk_recall"], 6),
                    "num_folds": score_info["num_folds"],
                    "total_test_size": score_info["total_test_size"],
                    "params": params,
                }
                evaluation_table.append(row)
                if best is None or score_info["score"] > best["score_info"]["score"]:
                    best = {
                        "name": name,
                        "model": model,
                        "params": params,
                        "score_info": score_info,
                        "use_pooled": use_pooled,
                    }
            except Exception as exc:
                evaluation_table.append({"candidate": name, "error": str(exc)})

        if best is None:
            raise ValueError("Automatic model selection failed for every candidate.")

        # Train final candidate on all available rows. Validation remains current-symbol based.
        X_train_final = X_current
        y_train_final = y_current
        if best["use_pooled"] and pooled_available:
            X_train_final = pd.concat([X_pool, X_current], ignore_index=True)
            y_train_final = pd.concat([y_pool, y_current], ignore_index=True)

        final_model = clone(best["model"])
        final_model.fit(X_train_final, y_train_final)
        feature_importance = self._feature_importance(final_model)

        new_score = float(best["score_info"]["score"])
        existing_score = None
        if isinstance(existing_bundle, dict):
            existing_score = self._safe_float(existing_bundle.get("selection_score"), default=None)

        should_save = existing_bundle is None or existing_score is None or new_score >= existing_score + self.min_save_improvement
        save_decision = "saved_new_model" if should_save else "kept_existing_model"

        bundle = {
            "model": final_model,
            "feature_columns": self.feature_columns,
            "model_version": self.MODEL_VERSION,
            "symbol": symbol_clean,
            "trained_at_utc": self._now(),
            "best_candidate": best["name"],
            "best_params": best["params"],
            "training_scope": "pooled_local_cache" if best["use_pooled"] else "single_symbol",
            "selection_score": new_score,
            "model_selection_score": best["score_info"],
            "baseline_score": baseline,
            "label_distribution": data_info["label_distribution"],
            "pooled_data_info": pooled_info,
            "feature_importance": feature_importance,
            "feature_snapshot": current_feature_snapshot,
            "feature_drift": feature_drift,
        }

        if should_save:
            joblib.dump(bundle, self.model_path)
            self.model_bundle = bundle
            self._save_metadata({k: v for k, v in bundle.items() if k != "model"})
        else:
            self.model_bundle = existing_bundle

        best_score = best["score_info"]
        improvement_over_baseline = None
        if baseline.get("score") is not None:
            improvement_over_baseline = round(new_score - float(baseline["score"]), 6)

        result = {
            "success": True,
            "agent": "Training Agent",
            "agent_goal": "Train and select a signal model automatically.",
            "symbol": symbol_clean,
            "model_source": "auto_model_selection",
            "save_decision": save_decision,
            "saved": bool(should_save),
            "model_path": str(self.model_path),
            "metadata_path": str(self._metadata_path()),
            "model_version": self.MODEL_VERSION,
            "best_candidate": best["name"],
            "best_params": best["params"],
            "training_scope": "pooled_local_cache" if best["use_pooled"] else "single_symbol",
            "selection_score": round(new_score, 6),
            "test_accuracy": round(best_score["accuracy"], 6),
            "balanced_accuracy": round(best_score["balanced_accuracy"], 6),
            "macro_f1": round(best_score["macro_f1"], 6),
            "sell_risk_recall": round(best_score["sell_risk_recall"], 6),
            "baseline_accuracy": self._round_float(baseline.get("accuracy"), 6),
            "baseline_score": self._round_float(baseline.get("score"), 6),
            "improvement_over_baseline_score": improvement_over_baseline,
            "num_samples": data_info["num_samples"],
            "label_distribution": data_info["label_distribution"],
            "pooled_data_info": pooled_info,
            "evaluation_table": evaluation_table,
            "optimization_results": evaluation_table,
            "walk_forward_summary": {
                "num_folds": best_score["num_folds"],
                "total_test_size": best_score["total_test_size"],
                "folds": best_score["folds"],
            },
            "confusion_matrix": best_score["confusion_matrix"],
            "classification_report": best_score["classification_report"],
            "feature_importance": feature_importance,
            "feature_drift": feature_drift,
            "feature_snapshot": current_feature_snapshot,
            "model_comparison_reason": (
                "New model saved because no comparable model existed or the score improved enough."
                if should_save
                else "Existing model kept because the new model did not exceed the saved score by the minimum improvement threshold."
            ),
            "agent_decision": (
                "A new model was saved after automatic comparison."
                if should_save
                else "The existing model was kept because the new model was not clearly better."
            ),
            "summary": (
                f"Training Agent selected {best['name']} for {symbol_clean}."
                if should_save
                else f"Training Agent tested new models for {symbol_clean}, but kept the existing model."
            ),
        }
        result["storage_result"] = self._record_training_run_to_storage(symbol_clean, result)
        return result


    def _load_model(self) -> Optional[Dict[str, Any]]:
        if not self.model_path.exists() or self.model_path.stat().st_size == 0:
            return None
        try:
            bundle = joblib.load(self.model_path)
            if isinstance(bundle, dict) and bundle.get("model") is not None:
                self.model_bundle = bundle
                return bundle
            self.model_bundle = {"model": bundle, "feature_columns": self.feature_columns, "model_version": "legacy"}
            return self.model_bundle
        except Exception:
            return None


    def model_exists(self) -> bool:
        return self.model_path.exists()


    def load_existing_model_info(self) -> Dict[str, Any]:
        bundle = self._load_model()
        if not bundle:
            return {"success": False, "summary": "No existing signal model was found."}
        return {
            "success": True,
            "agent": "Training Agent",
            "model_source": "loaded_existing_model",
            "model_path": str(self.model_path),
            "metadata_path": str(self._metadata_path()),
            "model_version": bundle.get("model_version", "unknown"),
            "best_candidate": bundle.get("best_candidate"),
            "best_params": bundle.get("best_params", {}),
            "selection_score": self._round_float(bundle.get("selection_score"), 6),
            "training_scope": bundle.get("training_scope", "unknown"),
            "label_distribution": bundle.get("label_distribution", {}),
            "feature_importance": bundle.get("feature_importance", []),
            "pooled_data_info": bundle.get("pooled_data_info", {}),
            "summary": f"Loaded existing signal model from {self.model_path}.",
        }


    def train_or_load_model(
        self,
        historical_data: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        self._set_symbol_model_path(symbol or (historical_data or {}).get("symbol"))
        existing = self._load_model()

        # No manual optimizer is needed: if the model is missing, old, or forced,
        # the Training Agent automatically runs model selection.
        if existing and not self._should_auto_retrain(existing, force_retrain):
            return self.load_existing_model_info()

        if historical_data and historical_data.get("success"):
            try:
                result = self._auto_train(
                    historical_data,
                    symbol=symbol or historical_data.get("symbol"),
                    existing_bundle=existing,
                    force_retrain=force_retrain,
                )
                self._record_training_result_to_storage(symbol or historical_data.get("symbol"), result)
                return result
            except Exception as exc:
                if existing:
                    self.model_bundle = existing
                    return {
                        "success": True,
                        "agent": "Training Agent",
                        "model_source": "loaded_existing_after_training_failed",
                        "model_path": str(self.model_path),
                        "warning": str(exc),
                        "summary": "Automatic training failed, so the existing model was kept.",
                    }
                return {
                    "success": False,
                    "agent": "Training Agent",
                    "error": str(exc),
                    "summary": "Training Agent could not train a signal model.",
                }

        if existing:
            return self.load_existing_model_info()

        return {
            "success": False,
            "agent": "Training Agent",
            "summary": "No model and no usable historical data were available.",
        }


    def train_or_load_signal_model(self, historical_data=None, symbol=None, force_retrain=False):
        return self.train_or_load_model(historical_data, symbol, force_retrain)


    def load_or_train_model(self, historical_data=None, symbol=None, force_retrain=False):
        return self.train_or_load_model(historical_data, symbol, force_retrain)


    def train_model(self, historical_data=None, symbol=None, force_retrain=True):
        return self.train_or_load_model(historical_data, symbol, force_retrain)


    def run_training(self, historical_data=None, symbol=None, force_retrain=False):
        return self.train_or_load_model(historical_data, symbol, force_retrain)


    def run(self, historical_data=None, symbol=None, force_retrain=False):
        return self.train_or_load_model(historical_data, symbol, force_retrain)


    def train_from_historical_data(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.train_or_load_model(historical_data=historical_data, symbol=historical_data.get("symbol"), force_retrain=True)


    def train_from_csv(self, csv_path: str = "data/historical_data.csv") -> Dict[str, Any]:
        df = pd.read_csv(csv_path)
        historical_data = {"success": True, "symbol": "CSV", "prices": df.to_dict("records")}
        return self.train_or_load_model(historical_data=historical_data, symbol="CSV", force_retrain=True)


    def train_from_price_records(self, price_records: list) -> Dict[str, Any]:
        historical_data = {"success": True, "symbol": "RECORDS", "prices": price_records}
        return self.train_or_load_model(historical_data=historical_data, symbol="RECORDS", force_retrain=True)

