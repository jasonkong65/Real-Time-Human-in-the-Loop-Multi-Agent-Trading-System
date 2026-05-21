from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from utils.features import build_trading_features


class TrainingAgent:
    """
    Training Agent:
    Trains, loads, and uses a lightweight stock signal model.

    The model predicts:
    - BUY_CANDIDATE
    - HOLD
    - SELL_RISK
    """

    def __init__(self, model_path: str = "models/signal_model.pkl"):
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.feature_columns = [
            "return_1",
            "return_5",
            "return_20",
            "ma_gap",
            "volatility_20",
            "volume_change",
            "rsi_14",
            "validation_confidence_score"
        ]

    def model_exists(self) -> bool:
        """
        Check whether a trained model already exists.
        """
        return self.model_path.exists()

    def load_existing_model_info(self) -> dict:
        """
        Load metadata of an existing trained model without retraining.
        """
        if not self.model_path.exists():
            return {
                "success": False,
                "agent_goal": "Load an existing trained signal model.",
                "agent_decision": "No existing model was found.",
                "summary": f"Model not found at {self.model_path}.",
                "model_path": str(self.model_path)
            }

        try:
            model_bundle = joblib.load(self.model_path)

            return {
                "success": True,
                "agent_goal": "Load an existing trained signal model.",
                "agent_decision": "Existing signal model was loaded. Retraining was skipped.",
                "model_path": str(self.model_path),
                "model_type": type(model_bundle.get("model")).__name__,
                "trained_at": model_bundle.get("trained_at"),
                "feature_columns": model_bundle.get("feature_columns"),
                "label_rule": model_bundle.get("label_rule"),
                "summary": f"Loaded existing signal model from {self.model_path}."
            }

        except Exception as e:
            return {
                "success": False,
                "agent_goal": "Load an existing trained signal model.",
                "agent_decision": "Existing model could not be loaded.",
                "summary": f"Model loading failed: {str(e)}",
                "model_path": str(self.model_path)
            }

    def _confidence_level(self, prediction_confidence):
        """
        Convert prediction probability into a readable confidence level.
        """
        if prediction_confidence is None:
            return "Unknown"

        if prediction_confidence >= 0.65:
            return "High"
        elif prediction_confidence >= 0.45:
            return "Medium"
        else:
            return "Low"

    def _make_label(self, future_return: float) -> str:
        """
        Convert future 5-day return into a training label.
        """
        if future_return > 0.015:
            return "BUY_CANDIDATE"
        elif future_return < -0.015:
            return "SELL_RISK"
        else:
            return "HOLD"

    def train_from_historical_data(self, historical_data: dict) -> dict:
        """
        Train signal model using historical_data returned by DataAgent or HistoricalDataAgent.
        """
        if not historical_data.get("success"):
            return {
                "success": False,
                "agent_goal": "Train a lightweight signal model from historical market data.",
                "agent_decision": "Training skipped because historical data is unavailable.",
                "summary": historical_data.get("error", "Historical data unavailable."),
                "model_path": str(self.model_path)
            }

        price_records = historical_data.get("prices", [])
        return self.train_from_price_records(price_records)

    def train_from_csv(self, csv_path: str = "data/historical_data.csv") -> dict:
        """
        Train signal model using a local CSV file.

        Expected columns:
        date, open, high, low, close, volume
        """
        csv_path = Path(csv_path)

        if not csv_path.exists():
            return {
                "success": False,
                "agent_goal": "Train a lightweight signal model from local CSV data.",
                "agent_decision": "Training skipped because local CSV data was not found.",
                "summary": f"Training CSV not found at {csv_path}.",
                "model_path": str(self.model_path)
            }

        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.lower().strip() for c in df.columns]

            required_cols = ["date", "open", "high", "low", "close", "volume"]
            missing_cols = [c for c in required_cols if c not in df.columns]

            if missing_cols:
                return {
                    "success": False,
                    "agent_goal": "Train a lightweight signal model from local CSV data.",
                    "agent_decision": "Training failed because required columns are missing.",
                    "summary": f"Missing columns: {missing_cols}",
                    "model_path": str(self.model_path)
                }

            price_records = df[required_cols].to_dict("records")
            return self.train_from_price_records(price_records)

        except Exception as e:
            return {
                "success": False,
                "agent_goal": "Train a lightweight signal model from local CSV data.",
                "agent_decision": "Training failed while reading CSV data.",
                "summary": f"CSV training error: {str(e)}",
                "model_path": str(self.model_path)
            }

    def train_from_price_records(self, price_records: list) -> dict:
        """
        Train a Random Forest signal model using OHLCV price records.
        """
        agent_goal = "Train a lightweight stock signal model using engineered technical features."

        feature_df = build_trading_features(price_records)

        if feature_df.empty:
            return {
                "success": False,
                "agent_goal": agent_goal,
                "agent_decision": "Training failed because features could not be constructed.",
                "summary": "Feature dataframe is empty.",
                "model_path": str(self.model_path)
            }

        feature_df["validation_confidence_score"] = 1.0

        feature_df["future_return_5"] = feature_df["close"].shift(-5) / feature_df["close"] - 1
        feature_df["target_signal"] = feature_df["future_return_5"].apply(
            lambda x: self._make_label(x) if pd.notna(x) else None
        )

        model_df = feature_df.dropna(subset=self.feature_columns + ["target_signal"]).copy()

        if len(model_df) < 30:
            return {
                "success": False,
                "agent_goal": agent_goal,
                "agent_decision": "Training failed because there are not enough usable samples.",
                "summary": f"Only {len(model_df)} usable samples found. At least 30 are recommended.",
                "model_path": str(self.model_path)
            }

        if model_df["target_signal"].nunique() < 2:
            return {
                "success": False,
                "agent_goal": agent_goal,
                "agent_decision": "Training failed because target labels do not contain enough classes.",
                "summary": "At least two target classes are required to train the model.",
                "model_path": str(self.model_path)
            }

        X = model_df[self.feature_columns]
        y = model_df["target_signal"]

        split_index = int(len(model_df) * 0.8)

        X_train = X.iloc[:split_index]
        y_train = y.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_test = y.iloc[split_index:]

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)

        test_accuracy = None
        if len(X_test) > 0:
            test_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_pred)

        model_bundle = {
            "model": model,
            "feature_columns": self.feature_columns,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "label_rule": "future_return_5 > 1.5% = BUY_CANDIDATE; < -1.5% = SELL_RISK; otherwise HOLD",
            "num_samples": len(model_df),
            "train_accuracy": round(train_accuracy, 3),
            "test_accuracy": round(test_accuracy, 3) if test_accuracy is not None else None
        }

        joblib.dump(model_bundle, self.model_path)

        label_distribution = model_df["target_signal"].value_counts().to_dict()

        return {
            "success": True,
            "agent_goal": agent_goal,
            "agent_decision": "Training completed and the signal model was saved.",
            "model_type": "RandomForestClassifier",
            "model_path": str(self.model_path),
            "num_samples": len(model_df),
            "train_accuracy": round(train_accuracy, 3),
            "test_accuracy": round(test_accuracy, 3) if test_accuracy is not None else None,
            "label_distribution": label_distribution,
            "feature_columns": self.feature_columns,
            "summary": (
                f"Signal model trained with {len(model_df)} samples. "
                f"Train accuracy={train_accuracy:.2f}, "
                f"test accuracy={test_accuracy:.2f}." if test_accuracy is not None
                else f"Signal model trained with {len(model_df)} samples."
            )
        }

    def _fallback_signal(self, analysis_result: dict, reason: str) -> dict:
        """
        Fallback when the trained model is unavailable or cannot be used.
        """
        analyst_signal = analysis_result.get("analyst_signal")
        analyst_score = analysis_result.get("analyst_score", 0.5)
        volatility_level = analysis_result.get("volatility_level")

        if analyst_signal == "BULLISH_WATCH" and analyst_score >= 0.7:
            signal = "BUY_CANDIDATE"
        elif analyst_signal == "BEARISH_RISK" or analyst_score <= 0.35:
            signal = "SELL_RISK"
        else:
            signal = "HOLD"

        if volatility_level == "High" and signal == "BUY_CANDIDATE":
            signal = "HOLD"

        prediction_confidence = round(float(analyst_score), 3) if analyst_score is not None else 0.5
        confidence_level = self._confidence_level(prediction_confidence)

        return {
            "success": True,
            "agent_goal": "Generate a trading signal from analyst features.",
            "signal_source": "fallback_rule",
            "model_signal": signal,
            "raw_model_signal": signal,
            "prediction_confidence": prediction_confidence,
            "confidence_level": confidence_level,
            "agent_decision": f"Used fallback rule because {reason}",
            "signal_for_next_agent": {
                "symbol": analysis_result.get("symbol"),
                "signal": signal,
                "signal_source": "fallback_rule",
                "prediction_confidence": prediction_confidence,
                "confidence_level": confidence_level,
                "raw_model_signal": signal,
                "analyst_score": analyst_score,
                "volatility_level": volatility_level
            },
            "summary": f"Fallback signal generated: {signal} with {confidence_level.lower()} confidence."
        }

    def predict_signal(self, analysis_result: dict) -> dict:
        """
        Predict trading signal using the trained model.
        If the model is unavailable or features are incomplete, use fallback rules.
        """
        if not analysis_result.get("success"):
            return self._fallback_signal(
                analysis_result,
                reason="Analyst Agent did not produce a successful analysis."
            )

        if not self.model_path.exists():
            return self._fallback_signal(
                analysis_result,
                reason="trained signal model was not found."
            )

        features = analysis_result.get("features_for_model") or {}

        missing_features = [
            col for col in self.feature_columns
            if features.get(col) is None
        ]

        if len(missing_features) > 3:
            return self._fallback_signal(
                analysis_result,
                reason=f"too many model features are missing: {missing_features}."
            )

        try:
            model_bundle = joblib.load(self.model_path)
            model = model_bundle["model"]
            feature_columns = model_bundle["feature_columns"]

            model_input = {}

            for col in feature_columns:
                value = features.get(col)
                model_input[col] = 0.0 if value is None else float(value)

            X = pd.DataFrame([model_input])

            prediction = model.predict(X)[0]

            prediction_confidence = None
            probabilities = {}

            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                classes = model.classes_

                probabilities = {
                    str(cls): round(float(prob), 3)
                    for cls, prob in zip(classes, proba)
                }

                prediction_confidence = max(probabilities.values())

            confidence_level = self._confidence_level(prediction_confidence)

            return {
                "success": True,
                "agent_goal": "Generate a trading signal from analyst features using a trained model.",
                "signal_source": "trained_signal_model",
                "model_signal": prediction,
                "raw_model_signal": prediction,
                "prediction_confidence": prediction_confidence,
                "confidence_level": confidence_level,
                "probabilities": probabilities,
                "model_path": str(self.model_path),
                "agent_decision": (
                    "The trained signal model generated a trading signal with "
                    f"{confidence_level.lower()} confidence."
                ),
                "signal_for_next_agent": {
                    "symbol": analysis_result.get("symbol"),
                    "signal": prediction,
                    "signal_source": "trained_signal_model",
                    "prediction_confidence": prediction_confidence,
                    "confidence_level": confidence_level,
                    "raw_model_signal": prediction,
                    "probabilities": probabilities,
                    "analyst_score": analysis_result.get("analyst_score"),
                    "volatility_level": analysis_result.get("volatility_level")
                },
                "summary": f"Signal model prediction: {prediction} with {confidence_level.lower()} confidence."
            }

        except Exception as e:
            return self._fallback_signal(
                analysis_result,
                reason=f"model prediction failed: {str(e)}."
            )