from typing import Any, Dict, List, Optional


class StrategistAgent:
    """
    Strategist Agent

    Turns the risk-controlled signal into a human-readable research plan. It is
    autonomous in the sense that it chooses the next research action, monitoring
    conditions, and review notes without manual parameter tuning.
    """

    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _get_nested(self, data: Dict[str, Any], keys: List[str], default=None):
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current

    def _norm(self, value: Any, default: str = "Unknown") -> str:
        if value is None:
            return default
        value = str(value).strip().upper()
        return value if value else default

    def _title(self, value: Any, default: str = "Unknown") -> str:
        if value is None:
            return default
        value = str(value).strip()
        return value.title() if value else default

    def _symbol(self, *items: Dict[str, Any]) -> str:
        for item in items:
            if isinstance(item, dict) and item.get("symbol"):
                return str(item.get("symbol")).upper().strip()
        return "UNKNOWN"

    def _final_signal(self, risk_result: Dict[str, Any], signal_result: Dict[str, Any]) -> str:
        return self._norm(risk_result.get("final_signal") or self._get_nested(risk_result, ["risk_for_next_agent", "final_signal"]) or signal_result.get("model_signal"), "HOLD")

    def _risk_level(self, risk_result: Dict[str, Any]) -> str:
        return self._title(risk_result.get("risk_level") or self._get_nested(risk_result, ["risk_for_next_agent", "risk_level"]), "Medium")

    def _analyst_signal(self, analysis_result: Dict[str, Any]) -> str:
        return self._norm(analysis_result.get("analyst_signal") or analysis_result.get("display_signal"), "NEUTRAL")

    def _display_signal(self, signal_result: Dict[str, Any], analysis_result: Dict[str, Any]) -> str:
        return self._norm(signal_result.get("display_signal") or analysis_result.get("display_signal") or signal_result.get("model_signal"), "HOLD")

    def _entry_risk(self, analysis_result: Dict[str, Any], risk_result: Dict[str, Any]) -> str:
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        return self._title(risk_result.get("entry_risk_level") or analysis_result.get("entry_risk_level") or stage2.get("entry_risk_level"), "Medium")

    def _trend(self, analysis_result: Dict[str, Any]) -> str:
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        return self._title(analysis_result.get("trend_direction") or stage2.get("trend_direction"), "Neutral")

    def _model_confidence(self, signal_result: Dict[str, Any]) -> str:
        value = signal_result.get("confidence_level") or self._get_nested(signal_result, ["signal_for_next_agent", "confidence_level"])
        return self._title(value, "Medium")

    def _choose_action(self, final_signal: str, risk_level: str, analyst_signal: str, display_signal: str, entry_risk: str, trend: str, model_conf: str) -> str:
        if final_signal == "BLOCKED" or risk_level == "Critical":
            return "NO_ACTION_DATA_OR_RISK_BLOCK"
        if final_signal == "SELL_RISK" or risk_level == "High":
            return "RISK_REDUCTION_REVIEW"
        positive_context = (
            final_signal == "BUY_CANDIDATE"
            or "BULLISH" in analyst_signal
            or "POSITIVE" in analyst_signal
            or "WATCHLIST" in display_signal
        )
        if positive_context and entry_risk == "High":
            return "WAIT_FOR_PULLBACK_OR_CONFIRMATION"
        if positive_context and risk_level in ["Medium", "High"]:
            return "WAIT_FOR_CONFIRMATION"
        if final_signal == "BUY_CANDIDATE" and risk_level == "Low" and model_conf in ["Medium", "High"]:
            return "RESEARCH_FOR_POSSIBLE_ENTRY"
        if trend == "Positive" and positive_context:
            return "MONITOR_POSITIVE_SETUP"
        return "MONITOR_AND_RESEARCH"

    def _level(self, action: str, risk_level: str) -> str:
        if action in ["NO_ACTION_DATA_OR_RISK_BLOCK", "RISK_REDUCTION_REVIEW"]:
            return "Defensive"
        if action in ["WAIT_FOR_PULLBACK_OR_CONFIRMATION", "WAIT_FOR_CONFIRMATION"]:
            return "Cautious"
        if action == "RESEARCH_FOR_POSSIBLE_ENTRY" and risk_level == "Low":
            return "Constructive"
        return "Conservative"

    def _position_guidance(self, action: str, symbol: str) -> str:
        messages = {
            "NO_ACTION_DATA_OR_RISK_BLOCK": "Do not take a paper action from this result. Wait for better data or lower risk.",
            "RISK_REDUCTION_REVIEW": "If this is already on a paper watchlist or paper portfolio, review downside risk and avoid adding exposure from this signal alone.",
            "WAIT_FOR_PULLBACK_OR_CONFIRMATION": "The setup looks positive, but entry timing risk is elevated. Wait for a pullback, cooler RSI, or stronger confirmation before considering a paper entry.",
            "WAIT_FOR_CONFIRMATION": "Keep it on the watchlist. Look for stronger confirmation before treating it as a paper entry candidate.",
            "RESEARCH_FOR_POSSIBLE_ENTRY": "This can be researched as a possible paper-entry candidate, but it still needs human review and source checks.",
            "MONITOR_POSITIVE_SETUP": "Monitor the positive setup. It is not a direct entry signal yet.",
            "MONITOR_AND_RESEARCH": "Maintain a monitoring stance and collect more evidence before making a paper decision.",
        }
        return messages.get(action, "Use this as a research note only.")

    def _leverage_guidance(self, action: str, risk_level: str, entry_risk: str) -> str:
        if risk_level in ["High", "Critical"] or action in ["NO_ACTION_DATA_OR_RISK_BLOCK", "RISK_REDUCTION_REVIEW"]:
            return "Do not use leverage. The risk layer is defensive."
        if entry_risk in ["Medium", "High"] or action.startswith("WAIT"):
            return "Do not use leverage. The setup still has timing risk."
        return "No leverage is assumed in this prototype. Keep the result as paper research only."

    def _conditions(self, action: str, entry_risk: str, trend: str, model_conf: str) -> List[str]:
        conditions = []
        if action in ["WAIT_FOR_PULLBACK_OR_CONFIRMATION", "WAIT_FOR_CONFIRMATION", "MONITOR_POSITIVE_SETUP"]:
            conditions.extend([
                "RSI or entry-risk level cools down from high to medium or low.",
                "Price holds above the short-term moving average after a pullback.",
                "The Analyst Agent stays positive after the next data refresh.",
            ])
        if model_conf == "Low":
            conditions.append("Model confidence improves to medium or high after automatic retraining.")
        if trend != "Positive":
            conditions.append("Trend direction becomes clearer in the historical analysis.")
        if action in ["RISK_REDUCTION_REVIEW", "NO_ACTION_DATA_OR_RISK_BLOCK"]:
            conditions.append("Risk level falls and data confidence remains medium or high.")
        return conditions or ["Re-run the pipeline after new market data is available."]

    def plan_strategy(
        self,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        training_result: Dict[str, Any],
        signal_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        evaluation_result: Optional[Dict[str, Any]] = None,
        optimizer_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        symbol = self._symbol(risk_result, signal_result, analysis_result, validation_result)
        final_signal = self._final_signal(risk_result, signal_result)
        risk_level = self._risk_level(risk_result)
        analyst_signal = self._analyst_signal(analysis_result)
        display_signal = self._display_signal(signal_result, analysis_result)
        entry_risk = self._entry_risk(analysis_result, risk_result)
        trend = self._trend(analysis_result)
        model_conf = self._model_confidence(signal_result)
        action = self._choose_action(final_signal, risk_level, analyst_signal, display_signal, entry_risk, trend, model_conf)
        level = self._level(action, risk_level)
        position_guidance = self._position_guidance(action, symbol)
        leverage_guidance = self._leverage_guidance(action, risk_level, entry_risk)
        conditions = self._conditions(action, entry_risk, trend, model_conf)
        risk_note = risk_result.get("risk_interpretation") or self._get_nested(risk_result, ["risk_for_next_agent", "risk_interpretation"], "")

        reasoning_steps = [
            f"Final risk-controlled signal: {final_signal}.",
            f"Risk level: {risk_level}.",
            f"Analyst signal: {analyst_signal}.",
            f"Display signal: {display_signal}.",
            f"Trend: {trend}; entry risk: {entry_risk}; model confidence: {model_conf}.",
            f"Chosen strategy action: {action}.",
        ]

        summary = f"Strategist Agent selected {action} for {symbol} with {level} strategy level."
        return {
            "success": True,
            "agent": "Strategist Agent",
            "agent_goal": "Convert model and risk outputs into a cautious research plan.",
            "symbol": symbol,
            "strategy_action": action,
            "strategy_level": level,
            "strategy_summary": summary,
            "position_guidance": position_guidance,
            "leverage_guidance": leverage_guidance,
            "watchlist_status": "Watchlist" if action not in ["NO_ACTION_DATA_OR_RISK_BLOCK"] else "No action",
            "conditions_to_reconsider": conditions,
            "risk_note": risk_note,
            "human_review_required": True,
            "reasoning_steps": reasoning_steps,
            "summary": summary,
        }

    def generate_strategy(self, *args, **kwargs) -> Dict[str, Any]:
        return self.plan_strategy(*args, **kwargs)

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        return self.plan_strategy(*args, **kwargs)
