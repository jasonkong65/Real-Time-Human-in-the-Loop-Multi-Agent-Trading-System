from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


class TrainingDiagnosticsMixin:
    """Optional model-diagnostic tools used by TrainingAgent.

    This is not exposed as a separate agent in the UI.  It belongs to the
    Training Agent because it only compares training candidates, writes model
    diagnostics, and optionally updates the saved signal model.
    """

    def _diagnostics_dir(self) -> Path:
        directory = Path("models")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _write_diagnostics_metadata(self, symbol: str, result: Dict[str, Any]) -> str:
        metadata_path = self._diagnostics_dir() / f"training_diagnostics_metadata_{symbol}.json"
        metadata = {
            "symbol": symbol,
            "owned_by": "Training Agent",
            "separate_agent": False,
            "model_path": result.get("model_path"),
            "metadata_path": result.get("metadata_path"),
            "save_decision": result.get("save_decision"),
            "saved": result.get("saved"),
            "best_candidate": result.get("best_candidate"),
            "best_params": result.get("best_params"),
            "training_scope": result.get("training_scope"),
            "selection_score": result.get("selection_score"),
            "test_accuracy": result.get("test_accuracy"),
            "balanced_accuracy": result.get("balanced_accuracy"),
            "macro_f1": result.get("macro_f1"),
            "sell_risk_recall": result.get("sell_risk_recall"),
            "baseline_accuracy": result.get("baseline_accuracy"),
            "baseline_score": result.get("baseline_score"),
            "improvement_over_baseline_score": result.get("improvement_over_baseline_score"),
            "num_samples": result.get("num_samples"),
            "label_distribution": result.get("label_distribution"),
            "pooled_data_info": result.get("pooled_data_info"),
            "confusion_matrix": result.get("confusion_matrix"),
            "classification_report": result.get("classification_report"),
            "feature_importance": result.get("feature_importance"),
        }
        try:
            with metadata_path.open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass
        return str(metadata_path)

    def _build_diagnostics_suggestions(self, result: Dict[str, Any]) -> List[str]:
        suggestions: List[str] = []
        macro_f1 = result.get("macro_f1")
        sell_recall = result.get("sell_risk_recall")
        pooled_info = result.get("pooled_data_info", {}) or {}

        try:
            if macro_f1 is not None and float(macro_f1) < 0.40:
                suggestions.append("Macro F1 is still low. Add more historical data or stronger features before trusting the model too much.")
        except Exception:
            pass

        try:
            if sell_recall is not None and float(sell_recall) < 0.35:
                suggestions.append("SELL_RISK recall is weak. Treat downside-risk outputs cautiously until more data is collected.")
        except Exception:
            pass

        if not pooled_info.get("enabled"):
            suggestions.append("Pooled local training data was limited. Add more symbols for a more stable model comparison.")

        if result.get("save_decision") == "kept_existing_model":
            suggestions.append("The existing model was kept because the new candidate was not clearly better.")
        else:
            suggestions.append("The selected model was saved for the selected diagnostic/main path.")

        suggestions.append("Use these diagnostics for paper decision support only, not live trading.")
        return suggestions

    def optimize_from_historical_data(
        self,
        symbol: str,
        historical_data: Dict[str, Any],
        validation_confidence_score: float = 0.95,
        apply_to_main_model: bool = False,
    ) -> Dict[str, Any]:
        """Run an optional forced model-comparison pass inside TrainingAgent.

        By default this writes to a diagnostic model path, so it does not
        overwrite the normal signal model.  If apply_to_main_model=True, the
        same TrainingAgent model-selection rules may update the saved model.
        """
        symbol = str(symbol or historical_data.get("symbol", "UNKNOWN")).upper().strip()
        model_dir = self._diagnostics_dir()
        main_path = model_dir / f"signal_model_{symbol}.pkl"
        diagnostic_dir = model_dir / "training_diagnostics"
        diagnostic_dir.mkdir(parents=True, exist_ok=True)
        diagnostic_path = diagnostic_dir / f"signal_model_{symbol}_diagnostic_candidate.pkl"
        target_path = main_path if apply_to_main_model else diagnostic_path

        # Use a temporary TrainingAgent instance so diagnostic runs do not mutate
        # the currently loaded pipeline model state.
        diagnostic_agent = self.__class__(
            model_path=str(target_path),
            auto_retrain_days=self.auto_retrain_days,
            min_save_improvement=self.min_save_improvement,
            pooled_data_dir=str(self.pooled_data_dir),
            max_pooled_symbols=self.max_pooled_symbols,
        )
        result = diagnostic_agent.train_or_load_model(
            historical_data=historical_data,
            symbol=symbol,
            force_retrain=True,
        )

        metadata_path = self._write_diagnostics_metadata(symbol, result)
        evaluation_table = result.get("evaluation_table", result.get("optimization_results", []))
        improvement = result.get("improvement_over_baseline_score")
        if improvement is None and result.get("test_accuracy") is not None and result.get("baseline_accuracy") is not None:
            try:
                improvement = round(float(result["test_accuracy"]) - float(result["baseline_accuracy"]), 6)
            except Exception:
                improvement = None

        if result.get("success"):
            performance_comment = "Training Agent ran an optional model-diagnostic comparison."
            if not apply_to_main_model:
                performance_comment += " It used a diagnostic path and did not overwrite the main signal model."
            elif result.get("save_decision") == "kept_existing_model":
                performance_comment += " The existing main model was kept because the new candidate was not clearly better."
            elif result.get("saved"):
                performance_comment += " A stronger candidate was allowed to update the main signal model."
        else:
            performance_comment = "Model diagnostics failed. Check historical data quality and feature availability."

        result.update({
            "agent": "Training Agent",
            "diagnostic_type": "model_comparison",
            "separate_agent": False,
            "manual_optimization_required": False,
            "apply_to_main_model_requested": bool(apply_to_main_model),
            "applied_to_main_model": bool(apply_to_main_model and result.get("saved")),
            "diagnostic_model_path": str(diagnostic_path),
            "main_model_path": str(main_path),
            "saved_model_path": result.get("model_path", str(target_path)),
            "metadata_path": metadata_path,
            "best_test_accuracy": result.get("test_accuracy"),
            "improvement_over_baseline": improvement,
            "performance_comment": performance_comment,
            "evaluation_table": evaluation_table,
            "optimization_results": evaluation_table,
            "confusion_matrix": result.get("confusion_matrix"),
            "classification_report": result.get("classification_report"),
            "feature_importance": result.get("feature_importance", []),
            "suggestions": self._build_diagnostics_suggestions(result),
            "summary": (
                f"Training Agent diagnostics completed for {symbol}. Best candidate: {result.get('best_candidate', 'N/A')}."
                if result.get("success")
                else f"Training Agent diagnostics could not complete for {symbol}."
            ),
        })
        return result

    def run_training_diagnostics(
        self,
        symbol: str,
        historical_data: Dict[str, Any],
        validation_confidence_score: float = 0.95,
        apply_to_main_model: bool = False,
    ) -> Dict[str, Any]:
        return self.optimize_from_historical_data(
            symbol=symbol,
            historical_data=historical_data,
            validation_confidence_score=validation_confidence_score,
            apply_to_main_model=apply_to_main_model,
        )
