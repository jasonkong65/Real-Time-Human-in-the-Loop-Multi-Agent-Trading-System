from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from utils.features import build_trading_features


class TrainingAgent:
    """
    Training Agent:
    Trains, loads, and uses a lightweight stock signal model.

    Core model labels remain compatible with the existing pipeline:
    - BUY_CANDIDATE
    - HOLD
    - SELL_RISK

    This optimized version adds a post-model signal refinement layer so that
    the system can distinguish:
    - true downside / sell risk
    - positive trend but elevated entry risk / overbought conditions

    Key improvement:
    If the raw model predicts SELL_RISK for a stock with positive momentum,
    positive moving-average structure, and non-high model confidence, the
    signal is softened to HOLD and annotated as BUY_WATCHLIST_OVERBOUGHT.

    This avoids the previous problem where a rising stock such as AAPL could be
    interpreted as a direct SELL_RISK only because the model learned short-term
    pullback risk from future-return labels.
    """

    CORE_SIGNALS = {"BUY_CANDIDATE", "HOLD", "SELL_RISK"}

    def __init__(self, model_path: str = "models/signal_model.pkl"):
        self.base_model_path = Path(model_path)
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
            "validation_confidence_score",
        ]

    # --------------------------------------------------
    # Model path helpers
    # --------------------------------------------------
    def _set_symbol_model_path(self, symbol: Optional[str] = None):
        """
        Use one model per symbol when symbol is available.
        Example: models/signal_model_AAPL.pkl
        """
        if symbol:
            clean_symbol = str(symbol).upper().strip()
            self.model_path = Path("models") / f"signal_model_{clean_symbol}.pkl"
        else:
            self.model_path = self.base_model_path

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def model_exists(self) -> bool:
        return self.model_path.exists()

    # --------------------------------------------------
    # Main app.py-compatible training method
    # --------------------------------------------------
    def train_or_load_model(
        self,
        historical_data: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
        force_retrain: bool = False,
        csv_path: str = "data/historical_data.csv",
    ) -> Dict[str, Any]:
        """
        Main method expected by app.py.

        Logic:
        1. Set per-symbol model path if symbol is provided.
        2. If model exists and force_retrain=False, load model info.
        3. Else train from historical_data if available.
        4. Else train from local CSV.
        5. If training fails but old model exists, load old model instead.
        """

        if symbol is None and isinstance(historical_data, dict):
            symbol = historical_data.get("symbol")

        self._set_symbol_model_path(symbol)

        if self.model_exists() and not force_retrain:
            return self.load_existing_model_info()

        training_result = None

        if isinstance(historical_data, dict) and historical_data.get("success"):
            training_result = self.train_from_historical_data(historical_data)

        if not training_result or not training_result.get("success"):
            csv_training_result = self.train_from_csv(csv_path)

            if csv_training_result.get("success"):
                return csv_training_result

            if training_result is not None:
                return training_result

            return csv_training_result

        return training_result

    # Compatibility aliases
    def train_or_load_signal_model(
        self,
        historical_data: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        return self.train_or_load_model(
            historical_data=historical_data,
            symbol=symbol,
            force_retrain=force_retrain,
        )

    def load_or_train_model(
        self,
        historical_data: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        return self.train_or_load_model(
            historical_data=historical_data,
            symbol=symbol,
            force_retrain=force_retrain,
        )

    def run_training(
        self,
        historical_data: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        return self.train_or_load_model(
            historical_data=historical_data,
            symbol=symbol,
            force_retrain=force_retrain,
        )

    def run(
        self,
        historical_data: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        return self.train_or_load_model(
            historical_data=historical_data,
            symbol=symbol,
            force_retrain=force_retrain,
        )

    # --------------------------------------------------
    # Load existing model metadata
    # --------------------------------------------------
    def load_existing_model_info(self) -> Dict[str, Any]:
        if not self.model_path.exists():
            return {
                "success": False,
                "agent_goal": "Load an existing trained signal model.",
                "agent_decision": "No existing model was found.",
                "summary": f"Model not found at {self.model_path}.",
                "model_path": str(self.model_path),
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
                "post_prediction_refinement": model_bundle.get(
                    "post_prediction_refinement",
                    "enabled_in_current_agent_code",
                ),
                "num_samples": model_bundle.get("num_samples"),
                "train_accuracy": model_bundle.get("train_accuracy"),
                "test_accuracy": model_bundle.get("test_accuracy"),
                "label_distribution": model_bundle.get("label_distribution"),
                "summary": f"Loaded existing signal model from {self.model_path}.",
            }

        except Exception as e:
            return {
                "success": False,
                "agent_goal": "Load an existing trained signal model.",
                "agent_decision": "Existing model could not be loaded.",
                "summary": f"Model loading failed: {str(e)}",
                "model_path": str(self.model_path),
            }

    # --------------------------------------------------
    # Training helpers
    # --------------------------------------------------
    def _confidence_level(self, prediction_confidence):
        if prediction_confidence is None:
            return "Unknown"

        try:
            prediction_confidence = float(prediction_confidence)
        except Exception:
            return "Unknown"

        if prediction_confidence >= 0.65:
            return "High"
        elif prediction_confidence >= 0.45:
            return "Medium"
        else:
            return "Low"

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _make_label(self, future_return: float) -> str:
        """
        Training target label from future 5-day return.

        This target is intentionally simple for the assignment. The optimized
        agent corrects short-term pullback bias at prediction time through
        post-model signal refinement.
        """
        if future_return > 0.015:
            return "BUY_CANDIDATE"
        elif future_return < -0.015:
            return "SELL_RISK"
        else:
            return "HOLD"

    def train_from_historical_data(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(historical_data, dict) or not historical_data.get("success"):
            return {
                "success": False,
                "agent_goal": "Train a lightweight signal model from historical market data.",
                "agent_decision": "Training skipped because historical data is unavailable.",
                "summary": (
                    historical_data.get("error", "Historical data unavailable.")
                    if isinstance(historical_data, dict)
                    else "Historical data unavailable."
                ),
                "model_path": str(self.model_path),
            }

        symbol = historical_data.get("symbol")
        if symbol:
            self._set_symbol_model_path(symbol)

        price_records = historical_data.get("prices", [])
        return self.train_from_price_records(price_records)

    def train_from_csv(self, csv_path: str = "data/historical_data.csv") -> Dict[str, Any]:
        csv_path = Path(csv_path)

        if not csv_path.exists() or csv_path.stat().st_size == 0:
            return {
                "success": False,
                "agent_goal": "Train a lightweight signal model from local CSV data.",
                "agent_decision": "Training skipped because local CSV data was not found or is empty.",
                "summary": f"Training CSV not found or empty at {csv_path}.",
                "model_path": str(self.model_path),
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
                    "model_path": str(self.model_path),
                }

            price_records = df[required_cols].to_dict("records")
            return self.train_from_price_records(price_records)

        except Exception as e:
            return {
                "success": False,
                "agent_goal": "Train a lightweight signal model from local CSV data.",
                "agent_decision": "Training failed while reading CSV data.",
                "summary": f"CSV training error: {str(e)}",
                "model_path": str(self.model_path),
            }

    def train_from_price_records(self, price_records: list) -> Dict[str, Any]:
        agent_goal = "Train a lightweight stock signal model using engineered technical features."

        if not price_records:
            return {
                "success": False,
                "agent_goal": agent_goal,
                "agent_decision": "Training failed because no price records were provided.",
                "summary": "No price records available for training.",
                "model_path": str(self.model_path),
            }

        feature_df = build_trading_features(price_records)

        if feature_df.empty:
            return {
                "success": False,
                "agent_goal": agent_goal,
                "agent_decision": "Training failed because features could not be constructed.",
                "summary": "Feature dataframe is empty.",
                "model_path": str(self.model_path),
            }

        feature_df["validation_confidence_score"] = 1.0

        feature_df["future_return_5"] = (
            feature_df["close"].shift(-5) / feature_df["close"] - 1
        )

        feature_df["target_signal"] = feature_df["future_return_5"].apply(
            lambda x: self._make_label(x) if pd.notna(x) else None
        )

        model_df = feature_df.dropna(
            subset=self.feature_columns + ["target_signal"]
        ).copy()

        if len(model_df) < 30:
            return {
                "success": False,
                "agent_goal": agent_goal,
                "agent_decision": "Training failed because there are not enough usable samples.",
                "summary": f"Only {len(model_df)} usable samples found. At least 30 are recommended.",
                "model_path": str(self.model_path),
            }

        if model_df["target_signal"].nunique() < 2:
            return {
                "success": False,
                "agent_goal": agent_goal,
                "agent_decision": "Training failed because target labels do not contain enough classes.",
                "summary": "At least two target classes are required to train the model.",
                "model_path": str(self.model_path),
            }

        X = model_df[self.feature_columns]
        y = model_df["target_signal"]

        split_index = int(len(model_df) * 0.8)

        X_train = X.iloc[:split_index]
        y_train = y.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_test = y.iloc[split_index:]

        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=5,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        )

        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_pred)

        test_accuracy = None
        if len(X_test) > 0:
            test_pred = model.predict(X_test)
            test_accuracy = accuracy_score(y_test, test_pred)

        label_distribution = model_df["target_signal"].value_counts().to_dict()

        model_bundle = {
            "model": model,
            "feature_columns": self.feature_columns,
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "label_rule": (
                "future_return_5 > 1.5% = BUY_CANDIDATE; "
                "< -1.5% = SELL_RISK; otherwise HOLD. "
                "At prediction time, a post-model refinement layer separates true downside risk "
                "from positive-trend but elevated-entry-risk conditions."
            ),
            "post_prediction_refinement": "enabled",
            "num_samples": len(model_df),
            "train_accuracy": round(train_accuracy, 3),
            "test_accuracy": round(test_accuracy, 3) if test_accuracy is not None else None,
            "label_distribution": label_distribution,
        }

        joblib.dump(model_bundle, self.model_path)

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
            "label_rule": model_bundle["label_rule"],
            "post_prediction_refinement": "enabled",
            "summary": (
                f"Signal model trained with {len(model_df)} samples. "
                f"Train accuracy={train_accuracy:.2f}, "
                f"test accuracy={test_accuracy:.2f}."
                if test_accuracy is not None
                else f"Signal model trained with {len(model_df)} samples."
            ),
        }

    # --------------------------------------------------
    # Market context and signal refinement
    # --------------------------------------------------
    def _get_feature_value(
        self,
        analysis_result: Dict[str, Any],
        features: Dict[str, Any],
        key: str,
        default: float = 0.0,
    ) -> float:
        if key in features and features.get(key) is not None:
            return self._safe_float(features.get(key), default)
        if key in analysis_result and analysis_result.get(key) is not None:
            return self._safe_float(analysis_result.get(key), default)
        return default

    def _market_context(
        self,
        analysis_result: Dict[str, Any],
        features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        features = features or analysis_result.get("features_for_model") or {}

        return_1 = self._get_feature_value(analysis_result, features, "return_1", 0.0)
        return_5 = self._get_feature_value(analysis_result, features, "return_5", 0.0)
        return_20 = self._get_feature_value(analysis_result, features, "return_20", 0.0)
        ma_gap = self._get_feature_value(analysis_result, features, "ma_gap", 0.0)
        volatility_20 = self._get_feature_value(analysis_result, features, "volatility_20", 0.0)
        rsi_14 = self._get_feature_value(analysis_result, features, "rsi_14", 50.0)
        analyst_score = self._safe_float(analysis_result.get("analyst_score"), 0.5)

        analyst_signal = analysis_result.get("analyst_signal", "Unknown")
        volatility_level = analysis_result.get("volatility_level")

        positive_momentum = (return_20 >= 0.04) or (return_5 >= 0.02) or (
            return_20 >= 0.02 and ma_gap >= 0.015
        )
        negative_momentum = (return_20 <= -0.04) or (return_5 <= -0.02) or (
            return_20 <= -0.02 and ma_gap <= -0.015
        )
        positive_structure = ma_gap >= 0.015 or analyst_score >= 0.60
        negative_structure = ma_gap <= -0.015 or analyst_score <= 0.35

        if positive_momentum and positive_structure:
            trend_direction = "Positive"
        elif negative_momentum and negative_structure:
            trend_direction = "Negative"
        else:
            trend_direction = "Mixed_or_Neutral"

        if rsi_14 >= 75:
            rsi_status = "Strongly_Overbought"
        elif rsi_14 >= 70:
            rsi_status = "Overbought"
        elif rsi_14 <= 25:
            rsi_status = "Strongly_Oversold"
        elif rsi_14 <= 30:
            rsi_status = "Oversold"
        else:
            rsi_status = "Normal"

        entry_risk_score = 0
        entry_risk_reasons = []

        if rsi_14 >= 70:
            entry_risk_score += 2
            entry_risk_reasons.append("RSI is high, so chasing risk may be elevated.")

        if return_20 >= 0.08:
            entry_risk_score += 1
            entry_risk_reasons.append("20-day return is strong, so short-term pullback risk may exist.")

        if ma_gap >= 0.06:
            entry_risk_score += 1
            entry_risk_reasons.append("Short-term moving average is far above the medium-term level.")

        if volatility_level == "High" or volatility_20 >= 0.05:
            entry_risk_score += 1
            entry_risk_reasons.append("Volatility is elevated.")

        if entry_risk_score >= 3:
            entry_risk_level = "High"
        elif entry_risk_score >= 1:
            entry_risk_level = "Medium"
        else:
            entry_risk_level = "Low"

        return {
            "return_1": round(return_1, 4),
            "return_5": round(return_5, 4),
            "return_20": round(return_20, 4),
            "ma_gap": round(ma_gap, 4),
            "volatility_20": round(volatility_20, 4),
            "rsi_14": round(rsi_14, 2),
            "analyst_score": round(analyst_score, 3),
            "analyst_signal": analyst_signal,
            "volatility_level": volatility_level,
            "trend_direction": trend_direction,
            "rsi_status": rsi_status,
            "entry_risk_level": entry_risk_level,
            "entry_risk_reasons": entry_risk_reasons,
            "positive_momentum": positive_momentum,
            "negative_momentum": negative_momentum,
            "positive_structure": positive_structure,
            "negative_structure": negative_structure,
        }

    def _core_signal_for_pipeline(self, enriched_signal: str) -> str:
        """
        Convert enriched signal into a core signal if another agent requires
        the original 3-label interface.
        """
        if enriched_signal in self.CORE_SIGNALS:
            return enriched_signal

        if enriched_signal in {"BUY_WATCHLIST_OVERBOUGHT", "WATCHLIST_BULLISH"}:
            return "HOLD"

        if enriched_signal in {"HOLD_MONITOR", "WAIT_FOR_CONFIRMATION"}:
            return "HOLD"

        if enriched_signal in {"DOWNSIDE_RISK_MONITOR"}:
            return "SELL_RISK"

        return "HOLD"

    def _refine_signal(
        self,
        raw_signal: str,
        prediction_confidence: Optional[float],
        probabilities: Dict[str, Any],
        analysis_result: Dict[str, Any],
        features: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Refine raw model output using current market context.

        This does NOT pretend to know the future. It corrects label semantics:
        a model may predict SELL_RISK because the next 5-day target historically
        fell after strong rallies, but that should be interpreted as elevated
        entry/pullback risk, not necessarily a bearish trend.
        """
        context = self._market_context(analysis_result, features)
        confidence_level = self._confidence_level(prediction_confidence)
        analyst_score = context["analyst_score"]
        trend = context["trend_direction"]
        entry_risk = context["entry_risk_level"]
        rsi_status = context["rsi_status"]

        final_signal = raw_signal
        enhanced_signal = raw_signal
        adjustment_applied = False
        adjustment_reason = "No post-model adjustment was applied."

        # Case 1: positive trend but raw model says SELL_RISK.
        # This is the key AAPL-style correction.
        if raw_signal == "SELL_RISK" and trend == "Positive":
            if confidence_level in {"Low", "Medium", "Unknown"} or analyst_score >= 0.58:
                final_signal = "HOLD"
                enhanced_signal = "BUY_WATCHLIST_OVERBOUGHT" if entry_risk in {"Medium", "High"} else "HOLD_MONITOR"
                adjustment_applied = True
                adjustment_reason = (
                    "Raw model predicted SELL_RISK, but current technical context shows a positive trend. "
                    "The signal was softened to HOLD and annotated as positive-trend with elevated entry risk."
                )

        # Case 2: raw BUY_CANDIDATE but entry risk is high.
        elif raw_signal == "BUY_CANDIDATE" and entry_risk == "High":
            final_signal = "HOLD"
            enhanced_signal = "BUY_WATCHLIST_OVERBOUGHT"
            adjustment_applied = True
            adjustment_reason = (
                "Raw model predicted BUY_CANDIDATE, but RSI/momentum conditions suggest elevated entry risk. "
                "The signal was softened to HOLD and annotated as BUY_WATCHLIST_OVERBOUGHT."
            )

        # Case 3: raw HOLD but analyst context is clearly positive.
        elif raw_signal == "HOLD" and trend == "Positive" and analyst_score >= 0.60:
            final_signal = "HOLD"
            enhanced_signal = "BUY_WATCHLIST_OVERBOUGHT" if entry_risk in {"Medium", "High"} else "WATCHLIST_BULLISH"
            adjustment_applied = True
            adjustment_reason = (
                "Raw model predicted HOLD, but analyst and technical context are positive. "
                "The core signal remains HOLD for safety, with a bullish watchlist annotation."
            )

        # Case 4: raw BUY_CANDIDATE but current trend context is negative.
        elif raw_signal == "BUY_CANDIDATE" and trend == "Negative":
            final_signal = "HOLD"
            enhanced_signal = "WAIT_FOR_CONFIRMATION"
            adjustment_applied = True
            adjustment_reason = (
                "Raw model predicted BUY_CANDIDATE, but current trend context is negative. "
                "The signal was softened to HOLD until confirmation improves."
            )

        # Case 5: raw HOLD but context is clearly negative.
        elif raw_signal == "HOLD" and trend == "Negative" and analyst_score <= 0.40:
            final_signal = "SELL_RISK"
            enhanced_signal = "DOWNSIDE_RISK_MONITOR"
            adjustment_applied = True
            adjustment_reason = (
                "Raw model predicted HOLD, but current analyst and trend context are negative. "
                "The signal was tightened to SELL_RISK for risk awareness."
            )

        if enhanced_signal == final_signal and final_signal == "BUY_CANDIDATE" and rsi_status in {"Overbought", "Strongly_Overbought"}:
            enhanced_signal = "BUY_WATCHLIST_OVERBOUGHT"
            adjustment_applied = True
            adjustment_reason = (
                "The core signal remains BUY_CANDIDATE, but RSI is elevated, so the display signal is annotated as overbought."
            )

        core_signal = self._core_signal_for_pipeline(enhanced_signal)

        return {
            "raw_model_signal": raw_signal,
            "model_signal": final_signal,
            "core_signal": core_signal,
            "enhanced_signal": enhanced_signal,
            "display_signal": enhanced_signal,
            "adjustment_applied": adjustment_applied,
            "adjustment_reason": adjustment_reason,
            "market_context": context,
            "prediction_confidence": prediction_confidence,
            "confidence_level": self._confidence_level(prediction_confidence),
            "probabilities": probabilities or {},
        }

    # --------------------------------------------------
    # Signal generation
    # --------------------------------------------------
    def generate_signal(
        self,
        analysis_result: Dict[str, Any],
        training_result: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main method expected by app.py.
        Generate signal after Training Agent has loaded/trained the model.
        """
        if symbol is None and isinstance(analysis_result, dict):
            symbol = analysis_result.get("symbol")

        if symbol:
            self._set_symbol_model_path(symbol)

        if isinstance(training_result, dict):
            model_path = training_result.get("model_path")
            if model_path:
                self.model_path = Path(model_path)

        return self.predict_signal(analysis_result)

    def generate_trading_signal(
        self,
        analysis_result: Dict[str, Any],
        training_result: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.generate_signal(
            analysis_result=analysis_result,
            training_result=training_result,
            symbol=symbol,
        )

    def run_signal_model(
        self,
        analysis_result: Dict[str, Any],
        training_result: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.generate_signal(
            analysis_result=analysis_result,
            training_result=training_result,
            symbol=symbol,
        )

    def predict(
        self,
        analysis_result: Dict[str, Any],
        training_result: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.generate_signal(
            analysis_result=analysis_result,
            training_result=training_result,
            symbol=symbol,
        )

    def _fallback_signal(self, analysis_result: Dict[str, Any], reason: str) -> Dict[str, Any]:
        analysis_result = analysis_result if isinstance(analysis_result, dict) else {}
        analyst_signal = analysis_result.get("analyst_signal")
        analyst_score = analysis_result.get("analyst_score", 0.5)
        volatility_level = analysis_result.get("volatility_level")
        features = analysis_result.get("features_for_model") or {}

        analyst_score_float = self._safe_float(analyst_score, 0.5)

        bullish_signals = [
            "BULLISH",
            "BULLISH_WATCH",
            "QUOTE_BULLISH",
            "HISTORICAL_BULLISH",
            "BUY_CANDIDATE",
            "WATCHLIST_BULLISH",
            "BUY_WATCHLIST_OVERBOUGHT",
        ]

        bearish_signals = [
            "BEARISH",
            "BEARISH_RISK",
            "QUOTE_BEARISH",
            "HISTORICAL_BEARISH",
            "SELL_RISK",
            "DOWNSIDE_RISK_MONITOR",
        ]

        if analyst_signal in bullish_signals or analyst_score_float >= 0.70:
            raw_signal = "BUY_CANDIDATE"
        elif analyst_signal in bearish_signals or analyst_score_float <= 0.35:
            raw_signal = "SELL_RISK"
        else:
            raw_signal = "HOLD"

        if volatility_level == "High" and raw_signal == "BUY_CANDIDATE":
            raw_signal = "HOLD"

        prediction_confidence = round(analyst_score_float, 3)
        probabilities = {}
        refined = self._refine_signal(
            raw_signal=raw_signal,
            prediction_confidence=prediction_confidence,
            probabilities=probabilities,
            analysis_result=analysis_result,
            features=features,
        )

        return {
            "success": True,
            "agent_goal": "Generate a trading signal from analyst features.",
            "signal_source": "fallback_rule_with_context_refinement",
            "model_signal": refined["model_signal"],
            "raw_model_signal": refined["raw_model_signal"],
            "core_signal": refined["core_signal"],
            "enhanced_signal": refined["enhanced_signal"],
            "display_signal": refined["display_signal"],
            "prediction_confidence": refined["prediction_confidence"],
            "confidence_level": refined["confidence_level"],
            "probabilities": probabilities,
            "market_context": refined["market_context"],
            "adjustment_applied": refined["adjustment_applied"],
            "adjustment_reason": refined["adjustment_reason"],
            "agent_decision": (
                f"Used fallback rule because {reason} "
                f"Post-model context refinement: {refined['adjustment_reason']}"
            ),
            "signal_for_next_agent": {
                "symbol": analysis_result.get("symbol"),
                "signal": refined["model_signal"],
                "core_signal": refined["core_signal"],
                "enhanced_signal": refined["enhanced_signal"],
                "display_signal": refined["display_signal"],
                "signal_source": "fallback_rule_with_context_refinement",
                "prediction_confidence": refined["prediction_confidence"],
                "confidence_level": refined["confidence_level"],
                "raw_model_signal": refined["raw_model_signal"],
                "probabilities": probabilities,
                "analyst_score": analyst_score_float,
                "volatility_level": volatility_level,
                "market_context": refined["market_context"],
            },
            "summary": (
                f"Fallback signal generated: {refined['model_signal']} "
                f"({refined['enhanced_signal']}) with {refined['confidence_level'].lower()} confidence."
            ),
        }

    def predict_signal(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict trading signal using the trained model.
        If the model is unavailable or features are incomplete, use fallback rules.
        """
        if not isinstance(analysis_result, dict) or not analysis_result.get("success"):
            return self._fallback_signal(
                analysis_result if isinstance(analysis_result, dict) else {},
                reason="Analyst Agent did not produce a successful analysis.",
            )

        symbol = analysis_result.get("symbol")
        if symbol:
            self._set_symbol_model_path(symbol)

        if not self.model_path.exists():
            return self._fallback_signal(
                analysis_result,
                reason="trained signal model was not found.",
            )

        features = analysis_result.get("features_for_model") or {}

        missing_features = [
            col for col in self.feature_columns
            if features.get(col) is None
        ]

        if len(missing_features) > 3:
            return self._fallback_signal(
                analysis_result,
                reason=f"too many model features are missing: {missing_features}.",
            )

        try:
            model_bundle = joblib.load(self.model_path)
            model = model_bundle["model"]
            feature_columns = model_bundle.get("feature_columns", self.feature_columns)

            model_input = {}

            for col in feature_columns:
                value = features.get(col)
                model_input[col] = 0.0 if value is None else float(value)

            X = pd.DataFrame([model_input])

            raw_prediction = str(model.predict(X)[0])

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

            refined = self._refine_signal(
                raw_signal=raw_prediction,
                prediction_confidence=prediction_confidence,
                probabilities=probabilities,
                analysis_result=analysis_result,
                features=features,
            )

            confidence_level = refined["confidence_level"]

            return {
                "success": True,
                "agent_goal": "Generate a trading signal from analyst features using a trained model.",
                "signal_source": "trained_signal_model_with_context_refinement",
                "model_signal": refined["model_signal"],
                "raw_model_signal": refined["raw_model_signal"],
                "core_signal": refined["core_signal"],
                "enhanced_signal": refined["enhanced_signal"],
                "display_signal": refined["display_signal"],
                "prediction_confidence": refined["prediction_confidence"],
                "confidence_level": confidence_level,
                "probabilities": probabilities,
                "market_context": refined["market_context"],
                "adjustment_applied": refined["adjustment_applied"],
                "adjustment_reason": refined["adjustment_reason"],
                "model_path": str(self.model_path),
                "agent_decision": (
                    "The trained signal model generated a raw trading signal and the Training Agent "
                    "then applied context-aware refinement. "
                    f"Raw signal={refined['raw_model_signal']}; final model signal={refined['model_signal']}; "
                    f"display signal={refined['display_signal']}; confidence={confidence_level}. "
                    f"Reason: {refined['adjustment_reason']}"
                ),
                "signal_for_next_agent": {
                    "symbol": analysis_result.get("symbol"),
                    "signal": refined["model_signal"],
                    "core_signal": refined["core_signal"],
                    "enhanced_signal": refined["enhanced_signal"],
                    "display_signal": refined["display_signal"],
                    "signal_source": "trained_signal_model_with_context_refinement",
                    "prediction_confidence": refined["prediction_confidence"],
                    "confidence_level": confidence_level,
                    "raw_model_signal": refined["raw_model_signal"],
                    "probabilities": probabilities,
                    "analyst_score": analysis_result.get("analyst_score"),
                    "volatility_level": analysis_result.get("volatility_level"),
                    "market_context": refined["market_context"],
                    "adjustment_applied": refined["adjustment_applied"],
                    "adjustment_reason": refined["adjustment_reason"],
                },
                "summary": (
                    f"Signal model prediction: raw={refined['raw_model_signal']}, "
                    f"final={refined['model_signal']}, display={refined['display_signal']} "
                    f"with {confidence_level.lower()} confidence."
                ),
            }

        except Exception as e:
            return self._fallback_signal(
                analysis_result,
                reason=f"model prediction failed: {str(e)}.",
            )
