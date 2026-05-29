from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from agents.training_agent import TrainingAgent


class TrainingOptimizerAgent:
    """
    Training Optimizer Agent

    The previous optimizer duplicated the Training Agent. In the new design,
    TrainingAgent already performs automatic model selection, walk-forward
    validation, and safe model saving.

    This class is kept for backward compatibility with the existing app.py UI.
    It now acts as a model-comparison dashboard wrapper around TrainingAgent.
    No separate manual optimisation logic is required.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def _write_optimizer_metadata(self, symbol: str, result: Dict[str, Any]) -> str:
        metadata_path = self.model_dir / f"optimizer_metadata_{symbol}.json"
        metadata = {
            "symbol": symbol,
            "merged_with_training_agent": True,
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

    def _build_suggestions(self, result: Dict[str, Any]) -> List[str]:
        suggestions = []
        macro_f1 = result.get("macro_f1")
        sell_recall = result.get("sell_risk_recall")
        pooled_info = result.get("pooled_data_info", {}) or {}

        try:
            if macro_f1 is not None and float(macro_f1) < 0.40:
                suggestions.append("Macro F1 is still low. Add more data, stronger features, or market-regime variables.")
        except Exception:
            pass

        try:
            if sell_recall is not None and float(sell_recall) < 0.35:
                suggestions.append("SELL_RISK recall is weak. Treat downside-risk outputs with caution until more data is collected.")
        except Exception:
            pass

        if not pooled_info.get("enabled"):
            suggestions.append("Pooled local training data was limited. Add more historical symbols for more stable model comparison.")

        if result.get("save_decision") == "kept_existing_model":
            suggestions.append("The existing model was kept because the new candidate was not clearly better.")
        else:
            suggestions.append("The selected model was saved automatically and can be used by the main pipeline.")

        suggestions.append("Use these results for paper decision support only, not live trading.")
        return suggestions

    def optimize_from_historical_data(
        self,
        symbol: str,
        historical_data: Dict[str, Any],
        validation_confidence_score: float = 0.95,
        apply_to_main_model: bool = False,
    ) -> Dict[str, Any]:
        symbol = str(symbol or historical_data.get("symbol", "UNKNOWN")).upper().strip()

        # The optimiser is now merged with TrainingAgent. We force a comparison run,
        # but TrainingAgent still decides whether to save the new model based on quality.
        main_path = self.model_dir / f"signal_model_{symbol}.pkl"
        agent = TrainingAgent(model_path=str(main_path))
        result = agent.train_or_load_model(
            historical_data=historical_data,
            symbol=symbol,
            force_retrain=True,
        )

        metadata_path = self._write_optimizer_metadata(symbol, result)

        evaluation_table = result.get("evaluation_table", result.get("optimization_results", []))
        improvement = result.get("improvement_over_baseline_score")
        if improvement is None and result.get("test_accuracy") is not None and result.get("baseline_accuracy") is not None:
            try:
                improvement = round(float(result["test_accuracy"]) - float(result["baseline_accuracy"]), 6)
            except Exception:
                improvement = None

        if result.get("success"):
            performance_comment = "The optimizer is now merged with Training Agent and uses the same automatic model-selection pipeline."
            if result.get("save_decision") == "kept_existing_model":
                performance_comment += " The existing model was kept because the new model was not clearly better."
            elif result.get("saved"):
                performance_comment += " A stronger model was saved automatically."
        else:
            performance_comment = "Model comparison failed. Check historical data quality and feature availability."

        result.update({
            "agent": "Training Optimizer Agent",
            "merged_with_training_agent": True,
            "manual_optimization_required": False,
            "applied_to_main_model": bool(result.get("saved")),
            "saved_model_path": result.get("model_path", str(main_path)),
            "metadata_path": metadata_path,
            "best_test_accuracy": result.get("test_accuracy"),
            "improvement_over_baseline": improvement,
            "performance_comment": performance_comment,
            "evaluation_table": evaluation_table,
            "optimization_results": evaluation_table,
            "confusion_matrix": result.get("confusion_matrix"),
            "classification_report": result.get("classification_report"),
            "feature_importance": result.get("feature_importance", []),
            "suggestions": self._build_suggestions(result),
            "summary": (
                f"Training Optimizer completed for {symbol}. Best candidate: {result.get('best_candidate', 'N/A')}."
                if result.get("success")
                else f"Training Optimizer could not complete for {symbol}."
            ),
        })
        return result

    def run(
        self,
        symbol: str,
        historical_data: Dict[str, Any],
        validation_confidence_score: float = 0.95,
        apply_to_main_model: bool = False,
    ) -> Dict[str, Any]:
        return self.optimize_from_historical_data(
            symbol,
            historical_data,
            validation_confidence_score,
            apply_to_main_model,
        )
