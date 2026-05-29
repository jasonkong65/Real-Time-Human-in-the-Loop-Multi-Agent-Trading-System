from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


class TrainingAgent:
    """
    Training Agent

    Builds a lightweight signal model automatically. It tries several model
    settings, chooses the best one with time-aware validation, saves the model,
    and then refines the raw model label with the current analyst context.

    The pipeline still returns only the core labels expected by app.py:
    BUY_CANDIDATE, HOLD, SELL_RISK.
    """

    MODEL_VERSION = "auto_rf_v3"
    CORE_SIGNALS = ["BUY_CANDIDATE", "HOLD", "SELL_RISK"]
    FEATURE_COLUMNS = [
        "return_1",
        "return_5",
        "return_20",
        "ma_gap",
        "volatility_20",
        "volume_change",
        "rsi_14",
        "validation_confidence_score",
    ]

    def __init__(self, model_path: str = "models/signal_model.pkl"):
        self.base_model_path = Path(model_path)
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.feature_columns = list(self.FEATURE_COLUMNS)
        self.model_bundle: Optional[Dict[str, Any]] = None

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

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

    def _rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    def _clean_history(self, historical_data: Dict[str, Any]) -> pd.DataFrame:
        prices = historical_data.get("prices", []) if isinstance(historical_data, dict) else []
        if not prices:
            return pd.DataFrame()
        df = pd.DataFrame(prices)
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
        if "close" not in df.columns and "adj_close" in df.columns:
            df["close"] = df["adj_close"]
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date")
        return df.dropna(subset=required).reset_index(drop=True)

    def _make_label(self, future_return: float) -> str:
        # Balanced but cautious thresholds. They are intentionally moderate so
        # the model learns watchlist-style signals rather than extreme actions.
        if future_return >= 0.018:
            return "BUY_CANDIDATE"
        if future_return <= -0.018:
            return "SELL_RISK"
        return "HOLD"

    def _build_dataset(self, historical_data: Dict[str, Any], validation_confidence_score: float = 0.95) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        df = self._clean_history(historical_data)
        if df.empty or len(df) < 80:
            raise ValueError("Not enough historical data for automatic model training.")

        close = df["close"]
        volume = df["volume"]
        features = pd.DataFrame(index=df.index)
        features["return_1"] = close.pct_change(1)
        features["return_5"] = close.pct_change(5)
        features["return_20"] = close.pct_change(20)
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        features["ma_gap"] = (ma20 - ma50) / ma50.replace(0, 1e-9)
        features["volatility_20"] = close.pct_change().rolling(20).std()
        features["volume_change"] = (volume.rolling(5).mean() - volume.rolling(20).mean()) / volume.rolling(20).mean().replace(0, 1e-9)
        features["rsi_14"] = self._rsi(close, 14)
        features["validation_confidence_score"] = validation_confidence_score
        future_return = close.shift(-5) / close - 1
        labels = future_return.apply(lambda x: self._make_label(float(x)) if pd.notna(x) else None)

        dataset = features.copy()
        dataset["label"] = labels
        dataset = dataset.replace([float("inf"), float("-inf")], pd.NA).dropna()
        X = dataset[self.feature_columns].astype(float)
        y = dataset["label"].astype(str)
        label_distribution = y.value_counts().to_dict()
        if y.nunique() < 2:
            raise ValueError("Training labels contain fewer than two classes.")
        return X, y, {"label_distribution": label_distribution, "num_samples": int(len(X))}

    def _candidate_models(self):
        return [
            (
                "random_forest_balanced_small",
                RandomForestClassifier(n_estimators=120, max_depth=4, min_samples_leaf=4, class_weight="balanced", random_state=42),
                {"family": "RandomForest", "n_estimators": 120, "max_depth": 4, "min_samples_leaf": 4},
            ),
            (
                "random_forest_balanced_medium",
                RandomForestClassifier(n_estimators=180, max_depth=6, min_samples_leaf=3, class_weight="balanced", random_state=42),
                {"family": "RandomForest", "n_estimators": 180, "max_depth": 6, "min_samples_leaf": 3},
            ),
            (
                "random_forest_balanced_flexible",
                RandomForestClassifier(n_estimators=220, max_depth=None, min_samples_leaf=5, class_weight="balanced", random_state=42),
                {"family": "RandomForest", "n_estimators": 220, "max_depth": None, "min_samples_leaf": 5},
            ),
            (
                "extra_trees_stable",
                ExtraTreesClassifier(n_estimators=200, max_depth=6, min_samples_leaf=4, class_weight="balanced", random_state=42),
                {"family": "ExtraTrees", "n_estimators": 200, "max_depth": 6, "min_samples_leaf": 4},
            ),
        ]

    def _time_split_score(self, model, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        n = len(X)
        split = max(int(n * 0.75), 50)
        if split >= n - 10:
            split = max(int(n * 0.70), 30)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        bal = balanced_accuracy_score(y_test, pred)
        macro_f1 = f1_score(y_test, pred, average="macro", zero_division=0)
        score = 0.40 * bal + 0.40 * macro_f1 + 0.20 * acc
        return {
            "score": float(score),
            "test_accuracy": float(acc),
            "balanced_accuracy": float(bal),
            "macro_f1": float(macro_f1),
            "test_size": int(len(X_test)),
            "pred_distribution": pd.Series(pred).value_counts().to_dict(),
        }

    def _auto_train(self, historical_data: Dict[str, Any], symbol: Optional[str], validation_confidence_score: float = 0.95) -> Dict[str, Any]:
        X, y, data_info = self._build_dataset(historical_data, validation_confidence_score)
        baseline_label = y.iloc[: int(len(y) * 0.75)].mode().iloc[0]
        split = max(int(len(X) * 0.75), 50)
        if split >= len(X) - 10:
            split = max(int(len(X) * 0.70), 30)
        baseline_acc = accuracy_score(y.iloc[split:], [baseline_label] * len(y.iloc[split:]))

        results = []
        best = None
        for name, model, params in self._candidate_models():
            try:
                score_info = self._time_split_score(model, X, y)
                row = {"candidate": name, **params, **score_info}
                results.append(row)
                if best is None or score_info["score"] > best["score_info"]["score"]:
                    best = {"name": name, "model": model, "params": params, "score_info": score_info}
            except Exception as exc:
                results.append({"candidate": name, "error": str(exc)})

        if best is None:
            raise ValueError("Automatic model selection failed for every candidate.")

        # Refit the best model on all available rows.
        final_model = best["model"]
        final_model.fit(X, y)
        bundle = {
            "model": final_model,
            "feature_columns": self.feature_columns,
            "model_version": self.MODEL_VERSION,
            "symbol": str(symbol).upper().strip() if symbol else None,
            "trained_at_utc": self._now(),
            "best_params": best["params"],
            "model_selection_score": best["score_info"],
            "label_distribution": data_info["label_distribution"],
        }
        joblib.dump(bundle, self.model_path)
        self.model_bundle = bundle

        return {
            "success": True,
            "agent": "Training Agent",
            "agent_goal": "Automatically train and select the best lightweight signal model.",
            "symbol": str(symbol).upper().strip() if symbol else historical_data.get("symbol", "UNKNOWN"),
            "model_source": "auto_optimized_training",
            "model_path": str(self.model_path),
            "model_version": self.MODEL_VERSION,
            "best_params": best["params"],
            "best_candidate": best["name"],
            "selection_score": round(best["score_info"]["score"], 4),
            "test_accuracy": round(best["score_info"]["test_accuracy"], 4),
            "balanced_accuracy": round(best["score_info"]["balanced_accuracy"], 4),
            "macro_f1": round(best["score_info"]["macro_f1"], 4),
            "baseline_accuracy": round(float(baseline_acc), 4),
            "num_samples": data_info["num_samples"],
            "label_distribution": data_info["label_distribution"],
            "optimization_results": results,
            "agent_decision": "The Training Agent selected and saved the strongest model automatically.",
            "summary": f"Training Agent automatically selected {best['name']} for {symbol or historical_data.get('symbol', 'UNKNOWN')}.",
        }

    def _load_model(self) -> Optional[Dict[str, Any]]:
        if not self.model_path.exists() or self.model_path.stat().st_size == 0:
            return None
        try:
            bundle = joblib.load(self.model_path)
            if isinstance(bundle, dict) and bundle.get("model") is not None:
                self.model_bundle = bundle
                return bundle
            # Backward compatibility with old files that stored only the model.
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
            "model_version": bundle.get("model_version", "unknown"),
            "best_params": bundle.get("best_params", {}),
            "label_distribution": bundle.get("label_distribution", {}),
            "summary": f"Loaded existing signal model from {self.model_path}.",
        }

    def train_or_load_model(self, historical_data: Optional[Dict[str, Any]] = None, symbol: Optional[str] = None, force_retrain: bool = False) -> Dict[str, Any]:
        self._set_symbol_model_path(symbol or (historical_data or {}).get("symbol"))
        existing = self._load_model()
        if existing and not force_retrain and existing.get("model_version") == self.MODEL_VERSION:
            return self.load_existing_model_info()
        if historical_data and historical_data.get("success"):
            try:
                return self._auto_train(historical_data, symbol=symbol or historical_data.get("symbol"))
            except Exception as exc:
                if existing:
                    return {
                        "success": True,
                        "agent": "Training Agent",
                        "model_source": "loaded_existing_after_auto_train_failed",
                        "model_path": str(self.model_path),
                        "warning": str(exc),
                        "summary": "Automatic training failed, so the existing model was kept.",
                    }
                return {"success": False, "agent": "Training Agent", "error": str(exc), "summary": "Training Agent could not train a signal model."}
        if existing:
            return self.load_existing_model_info()
        return {"success": False, "agent": "Training Agent", "summary": "No model and no usable historical data were available."}

    # Backward-compatible training aliases
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

    # ------------------------------------------------------------------
    # Prediction and context refinement
    # ------------------------------------------------------------------
    def _feature_from_analysis(self, analysis_result: Dict[str, Any]) -> pd.DataFrame:
        feature_source = analysis_result.get("features_for_model") if isinstance(analysis_result, dict) else {}
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        row = {}
        for col in self.feature_columns:
            row[col] = self._safe_float(
                (feature_source or {}).get(col, stage2.get(col)),
                0.0 if col != "rsi_14" else 50.0,
            )
        if row.get("validation_confidence_score", 0.0) == 0.0:
            row["validation_confidence_score"] = 0.6
        return pd.DataFrame([row], columns=self.feature_columns)

    def _confidence_level(self, confidence: float) -> str:
        if confidence >= 0.66:
            return "High"
        if confidence >= 0.45:
            return "Medium"
        return "Low"

    def _predict_raw(self, X: pd.DataFrame) -> Tuple[str, float, Dict[str, float]]:
        bundle = self.model_bundle or self._load_model()
        if not bundle:
            raise ValueError("No signal model is loaded.")
        model = bundle.get("model")
        raw_signal = str(model.predict(X)[0])
        probabilities: Dict[str, float] = {}
        confidence = 0.50
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = list(model.classes_)
            probabilities = {str(cls): float(p) for cls, p in zip(classes, proba)}
            confidence = max(probabilities.values()) if probabilities else 0.50
        return raw_signal, float(confidence), probabilities

    def _context(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        return {
            "analyst_signal": str(analysis_result.get("analyst_signal", "NEUTRAL")).upper(),
            "analyst_score": self._safe_float(analysis_result.get("analyst_score"), 0.5),
            "trend_direction": analysis_result.get("trend_direction") or stage2.get("trend_direction", "Neutral"),
            "entry_risk_level": analysis_result.get("entry_risk_level") or stage2.get("entry_risk_level", "Medium"),
            "volatility_level": analysis_result.get("volatility_level") or stage2.get("volatility_level", "Unknown"),
            "return_20": self._safe_float(stage2.get("return_20"), 0.0),
            "ma_gap": self._safe_float(stage2.get("ma_gap"), 0.0),
            "rsi_14": self._safe_float(stage2.get("rsi_14"), 50.0),
        }

    def _refine_signal(self, raw_signal: str, confidence: float, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        ctx = self._context(analysis_result)
        trend_positive = ctx["trend_direction"] == "Positive" or (ctx["return_20"] > 0.03 and ctx["ma_gap"] > 0)
        entry_high = ctx["entry_risk_level"] == "High" or ctx["rsi_14"] >= 72
        entry_medium = ctx["entry_risk_level"] in ["Medium", "High"] or ctx["rsi_14"] >= 68
        analyst_bullish = "BULLISH" in ctx["analyst_signal"] or "POSITIVE" in ctx["analyst_signal"] or ctx["analyst_score"] >= 0.62

        final_signal = raw_signal if raw_signal in self.CORE_SIGNALS else "HOLD"
        display_signal = final_signal
        reason = f"Raw model predicted {raw_signal}."

        if final_signal == "BUY_CANDIDATE" and entry_high:
            final_signal = "HOLD"
            display_signal = "BUY_WATCHLIST_OVERBOUGHT"
            reason = "The model was bullish, but the current entry risk is high, so it was softened to HOLD."
        elif final_signal == "SELL_RISK" and trend_positive and analyst_bullish and confidence < 0.70:
            final_signal = "HOLD"
            display_signal = "BUY_WATCHLIST_ENTRY_RISK"
            reason = "The raw model saw pullback risk, but the broader trend is positive, so the output was softened to HOLD."
        elif final_signal == "HOLD" and trend_positive and analyst_bullish:
            display_signal = "BUY_WATCHLIST_OVERBOUGHT" if entry_medium else "BULLISH_WATCHLIST"
            reason = "The model is neutral, while the technical context is positive. This is treated as a watchlist setup."
        elif final_signal == "SELL_RISK" and trend_positive:
            display_signal = "PULLBACK_RISK_IN_UPTREND"
            reason = "The model sees short-term pullback risk inside a positive trend."
        elif final_signal == "BUY_CANDIDATE":
            display_signal = "BUY_RESEARCH_CANDIDATE"
            reason = "The model found a positive setup that needs human review."
        elif final_signal == "SELL_RISK":
            display_signal = "DOWNSIDE_RISK"
            reason = "The model found downside risk."

        return {
            "model_signal": final_signal,
            "display_signal": display_signal,
            "refinement_reason": reason,
            "context": ctx,
        }

    def _fallback_signal(self, analysis_result: Dict[str, Any], reason: str) -> Dict[str, Any]:
        score = self._safe_float(analysis_result.get("analyst_score"), 0.5)
        analyst_signal = str(analysis_result.get("analyst_signal", "NEUTRAL")).upper()
        entry_risk = analysis_result.get("entry_risk_level", "Medium")
        if "BEARISH" in analyst_signal or score <= 0.35:
            signal, display = "SELL_RISK", "DOWNSIDE_RISK"
        elif ("BULLISH" in analyst_signal or "POSITIVE" in analyst_signal or score >= 0.62) and entry_risk != "High":
            signal, display = "BUY_CANDIDATE", "BUY_RESEARCH_CANDIDATE"
        elif "BULLISH" in analyst_signal or "POSITIVE" in analyst_signal or score >= 0.58:
            signal, display = "HOLD", "BUY_WATCHLIST_ENTRY_RISK"
        else:
            signal, display = "HOLD", "HOLD"
        return {
            "success": True,
            "agent": "Training Agent",
            "agent_goal": "Generate a signal from available analyst context.",
            "signal_source": "context_fallback",
            "model_signal": signal,
            "display_signal": display,
            "raw_model_signal": "N/A",
            "prediction_confidence": round(score, 4),
            "confidence_level": "Medium",
            "agent_decision": f"Used analyst-context fallback because {reason}",
            "signal_for_next_agent": {
                "signal": signal,
                "display_signal": display,
                "prediction_confidence": round(score, 4),
                "confidence_level": "Medium",
            },
            "summary": f"Fallback signal generated: {display}.",
        }

    def generate_signal(self, analysis_result: Dict[str, Any], training_result: Optional[Dict[str, Any]] = None, symbol: Optional[str] = None) -> Dict[str, Any]:
        self._set_symbol_model_path(symbol or analysis_result.get("symbol"))
        try:
            X = self._feature_from_analysis(analysis_result)
            raw_signal, confidence, probabilities = self._predict_raw(X)
            refined = self._refine_signal(raw_signal, confidence, analysis_result)
            confidence_level = self._confidence_level(confidence)
            result = {
                "success": True,
                "agent": "Training Agent",
                "agent_goal": "Use the trained model and current context to generate a risk-aware signal.",
                "symbol": str(symbol or analysis_result.get("symbol", "UNKNOWN")).upper(),
                "signal_source": "auto_optimized_model",
                "model_signal": refined["model_signal"],
                "display_signal": refined["display_signal"],
                "raw_model_signal": raw_signal,
                "prediction_confidence": round(confidence, 4),
                "confidence_level": confidence_level,
                "class_probabilities": {k: round(v, 4) for k, v in probabilities.items()},
                "model_path": str(self.model_path),
                "context_used": refined["context"],
                "agent_decision": (
                    f"Raw signal={raw_signal}; final model signal={refined['model_signal']}; "
                    f"display signal={refined['display_signal']}. {refined['refinement_reason']}"
                ),
                "signal_for_next_agent": {
                    "symbol": str(symbol or analysis_result.get("symbol", "UNKNOWN")).upper(),
                    "signal": refined["model_signal"],
                    "display_signal": refined["display_signal"],
                    "raw_model_signal": raw_signal,
                    "prediction_confidence": round(confidence, 4),
                    "confidence_level": confidence_level,
                },
                "summary": f"Signal model output: {refined['display_signal']} with {confidence_level.lower()} confidence.",
            }
            return result
        except Exception as exc:
            return self._fallback_signal(analysis_result, reason=str(exc))

    # Backward-compatible prediction aliases
    def generate_trading_signal(self, analysis_result, training_result=None, symbol=None):
        return self.generate_signal(analysis_result, training_result, symbol)

    def run_signal_model(self, analysis_result, training_result=None, symbol=None):
        return self.generate_signal(analysis_result, training_result, symbol)

    def predict(self, analysis_result, training_result=None, symbol=None):
        return self.generate_signal(analysis_result, training_result, symbol)

    def predict_signal(self, analysis_result, training_result=None, symbol=None):
        return self.generate_signal(analysis_result, training_result, symbol)
