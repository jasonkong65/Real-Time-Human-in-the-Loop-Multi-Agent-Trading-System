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


class TrainingSignalMixin:

    """Mixin for building and generating trading signals in the TrainingAgent, including methods to construct feature vectors from analysis results, predict raw signals using the trained model, refine signals based on contextual factors, generate fallback signals when predictions fail, and produce comprehensive signal outputs that include confidence levels and reasoning for use in downstream agents and diagnostics."""

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
        trend_positive = ctx["trend_direction"] in ["Positive", "Strong Positive"] or (ctx["return_20"] > 0.03 and ctx["ma_gap"] > 0)
        entry_high = ctx["entry_risk_level"] == "High" or ctx["rsi_14"] >= 72
        entry_medium = ctx["entry_risk_level"] in ["Medium", "High"] or ctx["rsi_14"] >= 68
        analyst_bullish = "BULLISH" in ctx["analyst_signal"] or "POSITIVE" in ctx["analyst_signal"] or ctx["analyst_score"] >= 0.62

        final_signal = raw_signal if raw_signal in self.CORE_SIGNALS else "HOLD"
        display_signal = final_signal
        reason = f"Raw model predicted {raw_signal}."

        if final_signal == "BUY_CANDIDATE" and entry_high:
            final_signal = "HOLD"
            display_signal = "BUY_WATCHLIST_OVERBOUGHT"
            reason = "The setup is positive, but entry timing risk is high. The signal is kept as HOLD for paper review."
        elif final_signal == "SELL_RISK" and trend_positive and analyst_bullish and confidence < 0.70:
            final_signal = "HOLD"
            display_signal = "BUY_WATCHLIST_ENTRY_RISK"
            reason = "The model saw pullback risk, but the wider trend is positive. The signal is softened to HOLD."
        elif final_signal == "HOLD" and trend_positive and analyst_bullish:
            display_signal = "BUY_WATCHLIST_OVERBOUGHT" if entry_medium else "BULLISH_WATCHLIST"
            reason = "The model is neutral, while the technical setup is positive. Treat this as a watchlist case."
        elif final_signal == "SELL_RISK" and trend_positive:
            display_signal = "PULLBACK_RISK_IN_UPTREND"
            reason = "The model sees short-term pullback risk inside a positive trend."
        elif final_signal == "BUY_CANDIDATE":
            display_signal = "BUY_RESEARCH_CANDIDATE"
            reason = "The model found a positive setup for human review."
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


    def generate_signal(
        self,
        analysis_result: Dict[str, Any],
        training_result: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
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
                "signal_source": "auto_selected_model",
                "model_signal": refined["model_signal"],
                "display_signal": refined["display_signal"],
                "raw_model_signal": raw_signal,
                "prediction_confidence": round(confidence, 4),
                "confidence_level": confidence_level,
                "class_probabilities": {k: round(v, 4) for k, v in probabilities.items()},
                "model_path": str(self.model_path),
                "context_used": refined["context"],
                "agent_decision": (
                    f"Raw signal={raw_signal}; final signal={refined['model_signal']}; "
                    f"display={refined['display_signal']}. {refined['refinement_reason']}"
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


    def generate_trading_signal(self, analysis_result, training_result=None, symbol=None):
        return self.generate_signal(analysis_result, training_result, symbol)


    def run_signal_model(self, analysis_result, training_result=None, symbol=None):
        return self.generate_signal(analysis_result, training_result, symbol)


    def predict(self, analysis_result, training_result=None, symbol=None):
        return self.generate_signal(analysis_result, training_result, symbol)


    def predict_signal(self, analysis_result, training_result=None, symbol=None):
        return self.generate_signal(analysis_result, training_result, symbol)

