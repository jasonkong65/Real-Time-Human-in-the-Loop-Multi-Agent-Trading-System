from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


class TrainingOptimizerAgent:
    """
    Training Optimizer Agent:
    Performs a lightweight grid search for the signal model.

    It supports:
    - Training Agent parameter optimization
    - Evaluation-driven improvement
    - Model comparison
    - Optional saving of the optimized model

    This agent does not make trading recommendations directly.
    """

    FEATURE_COLUMNS = [
        "return_1",
        "return_5",
        "return_20",
        "ma_gap",
        "volatility_20",
        "volume_change",
        "rsi_14",
        "validation_confidence_score"
    ]

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Feature engineering helpers
    # --------------------------------------------------
    def _calculate_rsi(self, close_series: pd.Series, window: int = 14) -> pd.Series:
        delta = close_series.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()

        rs = avg_gain / avg_loss.replace(0, 1e-9)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _normalise_price_dataframe(self, historical_data: Dict[str, Any]) -> pd.DataFrame:
        prices = historical_data.get("prices", [])

        if not prices:
            return pd.DataFrame()

        df = pd.DataFrame(prices)

        # Normalise column names
        df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

        if "adj_close" in df.columns and "close" not in df.columns:
            df["close"] = df["adj_close"]

        required_columns = ["open", "high", "low", "close", "volume"]

        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Historical data is missing required column: {col}")

        for col in required_columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date").reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)

        df = df.dropna(subset=required_columns)

        return df

    def _build_training_dataset(
        self,
        historical_data: Dict[str, Any],
        validation_confidence_score: float = 0.95
    ) -> pd.DataFrame:
        df = self._normalise_price_dataframe(historical_data)

        if df.empty:
            return pd.DataFrame()

        df["daily_return"] = df["close"].pct_change()

        df["return_1"] = df["close"].pct_change(1)
        df["return_5"] = df["close"].pct_change(5)
        df["return_20"] = df["close"].pct_change(20)

        df["ma_5"] = df["close"].rolling(window=5).mean()
        df["ma_20"] = df["close"].rolling(window=20).mean()
        df["ma_gap"] = (df["ma_5"] - df["ma_20"]) / df["ma_20"]

        df["volatility_20"] = df["daily_return"].rolling(window=20).std()

        df["volume_change"] = df["volume"].pct_change(5)
        df["rsi_14"] = self._calculate_rsi(df["close"], window=14)

        df["validation_confidence_score"] = validation_confidence_score

        # Future 5-day return becomes the training label
        df["future_return_5"] = df["close"].shift(-5) / df["close"] - 1

        def label_rule(value):
            if value > 0.015:
                return "BUY_CANDIDATE"
            if value < -0.015:
                return "SELL_RISK"
            return "HOLD"

        df["label"] = df["future_return_5"].apply(label_rule)

        df = df.replace([float("inf"), float("-inf")], pd.NA)

        model_df = df.dropna(
            subset=self.FEATURE_COLUMNS + ["label", "future_return_5"]
        ).copy()

        return model_df

    # --------------------------------------------------
    # Optimisation logic
    # --------------------------------------------------
    def _train_and_evaluate(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        n_estimators: int,
        max_depth: Optional[int]
    ) -> Dict[str, Any]:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)

        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0
        )

        return {
            "model": model,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "test_accuracy": float(accuracy),
            "classification_report": report
        }

    def optimize_from_historical_data(
        self,
        symbol: str,
        historical_data: Dict[str, Any],
        validation_confidence_score: float = 0.95,
        apply_to_main_model: bool = False
    ) -> Dict[str, Any]:
        symbol = symbol.upper().strip()

        try:
            dataset = self._build_training_dataset(
                historical_data=historical_data,
                validation_confidence_score=validation_confidence_score
            )

            if dataset.empty or len(dataset) < 60:
                return {
                    "success": False,
                    "agent_goal": "Optimize Training Agent model parameters.",
                    "symbol": symbol,
                    "error": "Not enough historical samples for optimization.",
                    "num_samples": len(dataset),
                    "summary": (
                        f"Training Optimizer could not run for {symbol}: "
                        "not enough historical samples."
                    )
                }

            X = dataset[self.FEATURE_COLUMNS]
            y = dataset["label"]

            split_index = int(len(dataset) * 0.8)

            X_train = X.iloc[:split_index]
            X_test = X.iloc[split_index:]
            y_train = y.iloc[:split_index]
            y_test = y.iloc[split_index:]

            if len(X_test) < 10:
                return {
                    "success": False,
                    "agent_goal": "Optimize Training Agent model parameters.",
                    "symbol": symbol,
                    "error": "Test set is too small for meaningful evaluation.",
                    "num_samples": len(dataset),
                    "summary": (
                        f"Training Optimizer could not run for {symbol}: "
                        "test set is too small."
                    )
                }

            label_distribution = y.value_counts().to_dict()

            grid_n_estimators = [50, 100, 200]
            grid_max_depth = [3, 5, 8, None]

            results: List[Dict[str, Any]] = []
            best_result = None

            for n_estimators in grid_n_estimators:
                for max_depth in grid_max_depth:
                    result = self._train_and_evaluate(
                        X_train=X_train,
                        y_train=y_train,
                        X_test=X_test,
                        y_test=y_test,
                        n_estimators=n_estimators,
                        max_depth=max_depth
                    )

                    clean_result = {
                        "n_estimators": result["n_estimators"],
                        "max_depth": result["max_depth"],
                        "test_accuracy": round(result["test_accuracy"], 6)
                    }

                    results.append(clean_result)

                    if best_result is None:
                        best_result = result
                    elif result["test_accuracy"] > best_result["test_accuracy"]:
                        best_result = result

            baseline_result = self._train_and_evaluate(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                n_estimators=100,
                max_depth=5
            )

            best_accuracy = float(best_result["test_accuracy"])
            baseline_accuracy = float(baseline_result["test_accuracy"])
            improvement = best_accuracy - baseline_accuracy

            if apply_to_main_model:
                model_path = self.model_dir / f"signal_model_{symbol}.pkl"
            else:
                model_path = self.model_dir / f"optimized_signal_model_{symbol}.pkl"

            joblib.dump(best_result["model"], model_path)

            metadata_path = self.model_dir / f"optimizer_metadata_{symbol}.json"

            metadata = {
                "symbol": symbol,
                "trained_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                "feature_columns": self.FEATURE_COLUMNS,
                "best_params": {
                    "n_estimators": best_result["n_estimators"],
                    "max_depth": best_result["max_depth"]
                },
                "best_test_accuracy": best_accuracy,
                "baseline_accuracy": baseline_accuracy,
                "improvement_over_baseline": improvement,
                "label_distribution": label_distribution,
                "num_samples": len(dataset),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "applied_to_main_model": apply_to_main_model,
                "model_path": str(model_path)
            }

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            sorted_results = sorted(
                results,
                key=lambda item: item["test_accuracy"],
                reverse=True
            )

            if improvement > 0.03:
                performance_comment = (
                    "The optimized model improved test accuracy compared with the baseline."
                )
            elif improvement < -0.03:
                performance_comment = (
                    "The optimized model did not improve over the baseline. "
                    "The baseline setting may already be sufficient for this small dataset."
                )
            else:
                performance_comment = (
                    "The optimized model performed similarly to the baseline. "
                    "More training data or additional features may be needed for stronger improvement."
                )

            suggestions = [
                "Use more completed reward records before making strong claims about model performance.",
                "Compare optimized model behaviour across multiple stocks instead of relying on one symbol.",
                "Consider adding market regime features, volatility filters, or fundamental/news features later.",
                "Keep the optimized model as a paper decision-support model rather than a real trading model."
            ]

            summary = (
                f"Training Optimizer completed for {symbol}. "
                f"Best accuracy={best_accuracy:.3f}, "
                f"baseline accuracy={baseline_accuracy:.3f}, "
                f"best params={{n_estimators={best_result['n_estimators']}, "
                f"max_depth={best_result['max_depth']}}}."
            )

            return {
                "success": True,
                "agent_goal": "Optimize Training Agent model parameters using lightweight grid search.",
                "symbol": symbol,
                "num_samples": len(dataset),
                "train_samples": len(X_train),
                "test_samples": len(X_test),
                "feature_columns": self.FEATURE_COLUMNS,
                "label_distribution": label_distribution,
                "best_params": {
                    "n_estimators": best_result["n_estimators"],
                    "max_depth": best_result["max_depth"]
                },
                "best_test_accuracy": round(best_accuracy, 6),
                "baseline_accuracy": round(baseline_accuracy, 6),
                "improvement_over_baseline": round(improvement, 6),
                "optimization_results": sorted_results,
                "best_classification_report": best_result["classification_report"],
                "saved_model_path": str(model_path),
                "metadata_path": str(metadata_path),
                "applied_to_main_model": apply_to_main_model,
                "performance_comment": performance_comment,
                "suggestions": suggestions,
                "summary": summary
            }

        except Exception as e:
            return {
                "success": False,
                "agent_goal": "Optimize Training Agent model parameters.",
                "symbol": symbol,
                "error": str(e),
                "summary": f"Training Optimizer failed for {symbol}: {str(e)}"
            }

    # Compatibility alias
    def run(
        self,
        symbol: str,
        historical_data: Dict[str, Any],
        validation_confidence_score: float = 0.95,
        apply_to_main_model: bool = False
    ) -> Dict[str, Any]:
        return self.optimize_from_historical_data(
            symbol=symbol,
            historical_data=historical_data,
            validation_confidence_score=validation_confidence_score,
            apply_to_main_model=apply_to_main_model
        )