from pathlib import Path
from typing import Any, Dict
import json
import shutil

import joblib

from agents.training_agent import TrainingAgent


class TrainingOptimizerAgent:
    """
    Training Optimizer Agent

    Uses the same automatic model-selection logic as TrainingAgent. It remains
    available for the optional UI section, but the main Training Agent can now
    optimise itself without manual operation.
    """

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def optimize_from_historical_data(
        self,
        symbol: str,
        historical_data: Dict[str, Any],
        validation_confidence_score: float = 0.95,
        apply_to_main_model: bool = False,
    ) -> Dict[str, Any]:
        symbol = str(symbol or historical_data.get("symbol", "UNKNOWN")).upper().strip()
        temp_path = self.model_dir / f"optimized_signal_model_{symbol}.pkl"
        agent = TrainingAgent(model_path=str(temp_path))
        result = agent.train_or_load_model(historical_data=historical_data, symbol=symbol, force_retrain=True)
        if not result.get("success"):
            result.update({"applied_to_main_model": False, "saved_model_path": str(temp_path)})
            return result

        main_path = self.model_dir / f"signal_model_{symbol}.pkl"
        metadata_path = self.model_dir / f"optimizer_metadata_{symbol}.json"
        if apply_to_main_model and temp_path.exists():
            shutil.copyfile(temp_path, main_path)
            applied = True
            saved_model_path = str(main_path)
        else:
            applied = False
            saved_model_path = str(temp_path)

        metadata = {
            "symbol": symbol,
            "applied_to_main_model": applied,
            "saved_model_path": saved_model_path,
            "best_params": result.get("best_params", {}),
            "selection_score": result.get("selection_score"),
            "test_accuracy": result.get("test_accuracy"),
            "balanced_accuracy": result.get("balanced_accuracy"),
            "macro_f1": result.get("macro_f1"),
            "num_samples": result.get("num_samples"),
        }
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        improvement = None
        if result.get("test_accuracy") is not None and result.get("baseline_accuracy") is not None:
            improvement = round(float(result["test_accuracy"]) - float(result["baseline_accuracy"]), 6)

        performance_comment = "The optimizer selected the strongest candidate from the automatic model search."
        if improvement is not None and improvement <= 0.01:
            performance_comment = "The optimized model was close to the baseline. More data or better features may be needed."
        elif improvement is not None and improvement > 0.01:
            performance_comment = "The optimized model improved over the simple baseline in this validation split."

        result.update({
            "agent": "Training Optimizer Agent",
            "saved_model_path": saved_model_path,
            "metadata_path": str(metadata_path),
            "applied_to_main_model": applied,
            "best_test_accuracy": result.get("test_accuracy"),
            "improvement_over_baseline": improvement,
            "performance_comment": performance_comment,
            "suggestions": [
                "Use the automatically selected model as a paper decision-support model only.",
                "Compare results across several symbols before drawing strong conclusions.",
                "Add market-regime and news features later if more time is available.",
            ],
            "summary": f"Training Optimizer completed for {symbol}. Best model: {result.get('best_candidate', 'selected model')}.",
        })
        return result

    def run(self, symbol: str, historical_data: Dict[str, Any], validation_confidence_score: float = 0.95, apply_to_main_model: bool = False) -> Dict[str, Any]:
        return self.optimize_from_historical_data(symbol, historical_data, validation_confidence_score, apply_to_main_model)
