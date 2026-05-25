from typing import Dict, Any, Optional, List


class StrategistAgent:
    """
    Optimized rule-based Strategist Agent.

    Role:
    - Converts Risk Agent outputs into research-oriented strategy guidance.
    - Separates true sell/downside risk from positive-trend-but-entry-risk cases.
    - Does not execute real trades and does not provide personalized financial advice.

    Key improvement:
    - POSITIVE_BUT_ENTRY_RISK + HOLD/Medium risk becomes WAIT_FOR_PULLBACK_OR_CONFIRMATION,
      not a generic MONITOR_AND_RESEARCH.
    """

    ENTRY_RISK_ANALYST_SIGNALS = {
        "POSITIVE_BUT_ENTRY_RISK",
        "WATCHLIST_BULLISH_ENTRY_RISK",
        "BUY_WATCHLIST_OVERBOUGHT",
        "BUY_WATCHLIST_ENTRY_RISK",
        "POSITIVE_BUT_OVERBOUGHT",
    }

    BULLISH_SIGNALS = {
        "BULLISH",
        "BULLISH_MOMENTUM",
        "WATCHLIST_BULLISH",
        "POSITIVE",
        "POSITIVE_BUT_ENTRY_RISK",
        "WATCHLIST_BULLISH_ENTRY_RISK",
        "BUY_WATCHLIST_OVERBOUGHT",
        "BUY_WATCHLIST_ENTRY_RISK",
    }

    BEARISH_SIGNALS = {
        "BEARISH",
        "BEARISH_RISK",
        "DOWNSIDE_RISK",
        "NEGATIVE",
    }

    def __init__(self, auto_load_evaluation: bool = True):
        self.auto_load_evaluation = auto_load_evaluation

    # --------------------------------------------------
    # Safe extraction helpers
    # --------------------------------------------------
    def _get_nested(self, data: Dict[str, Any], keys: List[str], default=None):
        current = data

        for key in keys:
            if not isinstance(current, dict):
                return default

            current = current.get(key)

            if current is None:
                return default

        return current

    def _normalise_signal(self, value: Any, default: str = "Unknown") -> str:
        if value is None:
            return default
        return str(value).strip().upper()

    def _safe_float(self, value, default=None):
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _get_symbol(self, *results: Dict[str, Any]) -> str:
        for result in results:
            if not isinstance(result, dict):
                continue

            symbol = (
                result.get("symbol")
                or self._get_nested(result, ["strategy_for_next_agent", "symbol"])
                or self._get_nested(result, ["risk_for_next_agent", "symbol"])
                or self._get_nested(result, ["signal_for_next_agent", "symbol"])
                or self._get_nested(result, ["analysis_for_next_agent", "symbol"])
                or self._get_nested(result, ["validation_for_next_agent", "symbol"])
            )

            if symbol:
                return str(symbol).upper()

        return "UNKNOWN"

    def _get_final_signal(self, risk_result: Dict[str, Any], signal_result: Dict[str, Any]) -> str:
        return self._normalise_signal(
            risk_result.get("final_signal")
            or self._get_nested(risk_result, ["risk_for_next_agent", "final_signal"])
            or signal_result.get("model_signal")
            or signal_result.get("signal")
            or self._get_nested(signal_result, ["signal_for_next_agent", "model_signal"])
            or self._get_nested(signal_result, ["signal_for_next_agent", "signal"])
            or "HOLD",
            default="HOLD",
        )

    def _get_model_confidence(self, signal_result: Dict[str, Any]) -> str:
        confidence_level = (
            signal_result.get("confidence_level")
            or signal_result.get("model_confidence_level")
            or self._get_nested(signal_result, ["signal_for_next_agent", "confidence_level"])
            or self._get_nested(signal_result, ["signal_for_next_agent", "model_confidence_level"])
        )

        if isinstance(confidence_level, str):
            confidence_level = confidence_level.strip().capitalize()
            if confidence_level in ["High", "Medium", "Low"]:
                return confidence_level

        confidence = (
            signal_result.get("prediction_confidence")
            or signal_result.get("confidence")
            or self._get_nested(signal_result, ["signal_for_next_agent", "prediction_confidence"])
            or self._get_nested(signal_result, ["signal_for_next_agent", "confidence"])
        )

        confidence = self._safe_float(confidence, default=None)

        if confidence is None:
            return "Unknown"
        if confidence >= 0.65:
            return "High"
        if confidence >= 0.45:
            return "Medium"
        return "Low"

    def _get_evaluation_result(self, evaluation_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if isinstance(evaluation_result, dict) and evaluation_result:
            return evaluation_result

        if not self.auto_load_evaluation:
            return {}

        try:
            from agents.evaluator_agent import EvaluatorAgent
            return EvaluatorAgent().evaluate_history()
        except Exception:
            return {}

    def _get_analyst_signal(self, analysis_result: Dict[str, Any]) -> str:
        return self._normalise_signal(
            analysis_result.get("analyst_signal")
            or self._get_nested(analysis_result, ["analysis_for_next_agent", "analyst_signal"])
            or "Unknown"
        )

    def _get_analyst_score(self, analysis_result: Dict[str, Any]):
        return self._safe_float(
            analysis_result.get("analyst_score")
            or self._get_nested(analysis_result, ["analysis_for_next_agent", "analyst_score"]),
            default=None,
        )

    def _get_display_signal(self, analysis_result: Dict[str, Any], signal_result: Dict[str, Any], risk_result: Dict[str, Any]) -> str:
        value = (
            analysis_result.get("display_signal")
            or self._get_nested(analysis_result, ["analysis_for_next_agent", "display_signal"])
            or signal_result.get("display_signal")
            or signal_result.get("enhanced_signal")
            or self._get_nested(signal_result, ["signal_for_next_agent", "display_signal"])
            or self._get_nested(signal_result, ["signal_for_next_agent", "enhanced_signal"])
            or risk_result.get("display_signal")
            or self._get_nested(risk_result, ["risk_for_next_agent", "display_signal"])
            or "Unknown"
        )

        return self._normalise_signal(value)

    def _get_entry_risk_level(self, analysis_result: Dict[str, Any], signal_result: Dict[str, Any], risk_result: Dict[str, Any]) -> str:
        value = (
            risk_result.get("entry_risk_level")
            or self._get_nested(risk_result, ["risk_for_next_agent", "entry_risk_level"])
            or analysis_result.get("entry_risk_level")
            or analysis_result.get("entry_risk")
            or self._get_nested(analysis_result, ["analysis_for_next_agent", "entry_risk_level"])
            or self._get_nested(analysis_result, ["analysis_for_next_agent", "entry_risk"])
            or signal_result.get("entry_risk_level")
            or self._get_nested(signal_result, ["signal_for_next_agent", "entry_risk_level"])
        )

        if value is None:
            return "Unknown"

        return str(value).strip().replace("_", " ").title()

    def _get_trend_direction(self, analysis_result: Dict[str, Any], signal_result: Dict[str, Any], risk_result: Dict[str, Any]) -> str:
        value = (
            risk_result.get("trend_direction")
            or self._get_nested(risk_result, ["risk_for_next_agent", "trend_direction"])
            or analysis_result.get("trend_direction")
            or analysis_result.get("trend")
            or self._get_nested(analysis_result, ["analysis_for_next_agent", "trend_direction"])
            or self._get_nested(analysis_result, ["analysis_for_next_agent", "trend"])
        )

        if value is None and isinstance(signal_result.get("market_context"), dict):
            value = signal_result["market_context"].get("trend_direction")

        if value is None:
            return "Unknown"

        return str(value).strip().title()

    def _has_entry_risk_context(
        self,
        analyst_signal: str,
        analyst_score,
        display_signal: str,
        entry_risk_level: str,
        trend_direction: str,
        risk_result: Dict[str, Any],
    ) -> bool:
        analyst_signal = self._normalise_signal(analyst_signal)
        display_signal = self._normalise_signal(display_signal)
        entry_text = str(entry_risk_level or "").lower()
        trend_text = str(trend_direction or "").lower()

        risk_flag = (
            risk_result.get("positive_entry_risk_context")
            or self._get_nested(risk_result, ["risk_for_next_agent", "positive_entry_risk_context"])
        )

        if risk_flag is True:
            return True

        if analyst_signal in self.ENTRY_RISK_ANALYST_SIGNALS:
            return True

        if "ENTRY_RISK" in display_signal or "OVERBOUGHT" in display_signal:
            return True

        if entry_text in ["elevated", "high", "medium high", "moderate high"]:
            return True

        if "positive" in trend_text and analyst_score is not None and analyst_score >= 0.70:
            return True

        return False

    # --------------------------------------------------
    # Strategy rules
    # --------------------------------------------------
    def _base_strategy_action(
        self,
        validation_confidence: str,
        validation_next_action: str,
        final_signal: str,
        risk_level: str,
        model_confidence: str,
        analyst_signal: str,
        performance_level: str,
        entry_risk_context: bool,
    ) -> str:
        if validation_next_action == "BLOCK_ANALYSIS" or final_signal == "BLOCKED":
            return "NO_ACTION_DATA_OR_RISK_BLOCK"

        if entry_risk_context and final_signal in ["HOLD", "BUY_CANDIDATE"]:
            return "WAIT_FOR_PULLBACK_OR_CONFIRMATION"

        if final_signal == "SELL_RISK" or risk_level in ["High", "Critical"]:
            return "RISK_REDUCTION_REVIEW"

        if analyst_signal in self.BEARISH_SIGNALS:
            return "RISK_REDUCTION_REVIEW"

        if final_signal == "BUY_CANDIDATE":
            if (
                risk_level == "Low"
                and validation_confidence == "High"
                and model_confidence in ["High", "Medium"]
                and performance_level != "Needs improvement"
            ):
                return "RESEARCH_FOR_POSSIBLE_ENTRY"

            return "WAIT_FOR_CONFIRMATION"

        if final_signal == "HOLD":
            return "MONITOR_AND_RESEARCH"

        return "FURTHER_RESEARCH_ONLY"

    def _strategy_level(
        self,
        action: str,
        risk_level: str,
        model_confidence: str,
        performance_level: str,
    ) -> str:
        if action in ["NO_ACTION_DATA_OR_RISK_BLOCK", "RISK_REDUCTION_REVIEW"]:
            return "Defensive"

        if action == "WAIT_FOR_PULLBACK_OR_CONFIRMATION":
            return "Cautious"

        if risk_level in ["High", "Critical"]:
            return "Defensive"

        if performance_level == "Needs improvement":
            return "Conservative"

        if model_confidence == "Low":
            return "Conservative"

        if action == "RESEARCH_FOR_POSSIBLE_ENTRY":
            return "Balanced"

        return "Conservative"

    def _position_guidance(
        self,
        action: str,
        final_signal: str,
        risk_level: str,
        entry_risk_context: bool,
    ) -> str:
        if action == "NO_ACTION_DATA_OR_RISK_BLOCK":
            return (
                "Do not take a paper action from this result. The system should wait until data quality or risk conditions improve."
            )

        if action == "RISK_REDUCTION_REVIEW":
            return (
                "If this stock is already on a paper watchlist or paper portfolio, review exposure and downside risk. "
                "Avoid adding exposure based only on the current signal."
            )

        if action == "WAIT_FOR_PULLBACK_OR_CONFIRMATION":
            return (
                "The stock shows a positive setup, but entry timing risk is elevated. Do not chase aggressively. "
                "Wait for a pullback, cooler RSI/entry-risk conditions, or stronger confirmation before considering a paper entry."
            )

        if action == "WAIT_FOR_CONFIRMATION":
            return (
                "Treat the stock as a watchlist candidate, not an immediate entry. Wait for stronger confirmation from analyst signal, model confidence, and risk level."
            )

        if action == "RESEARCH_FOR_POSSIBLE_ENTRY":
            return (
                "The stock can be reviewed as a candidate for further research. Human review should still check fundamentals, news, valuation, and portfolio risk before any decision."
            )

        if final_signal == "HOLD" or risk_level == "Medium" or entry_risk_context:
            return (
                "Maintain a monitoring stance. Avoid aggressive entry or position increase until stronger evidence appears."
            )

        return "Use the output as research support only and wait for clearer evidence before changing exposure."

    def _leverage_guidance(
        self,
        final_signal: str,
        risk_level: str,
        model_confidence: str,
        performance_level: str,
        entry_risk_context: bool,
    ) -> str:
        if entry_risk_context:
            return "Do not use leverage. The setup has elevated entry timing / chase risk."

        if final_signal != "BUY_CANDIDATE":
            return "Do not use leverage. The final signal is not a buy-candidate signal."

        if risk_level != "Low":
            return "Do not use leverage. Risk level is not low."

        if model_confidence != "High":
            return "Do not use leverage. Model confidence is not high."

        if performance_level == "Needs improvement":
            return "Do not use leverage. Historical evaluation currently suggests the system needs improvement."

        return (
            "Leverage is not supported by this prototype. Any leverage decision would require separate human review, risk limits, and professional assessment."
        )

    def _watchlist_status(self, action: str) -> str:
        mapping = {
            "NO_ACTION_DATA_OR_RISK_BLOCK": "Do not add based on current data",
            "RISK_REDUCTION_REVIEW": "High caution watchlist",
            "WAIT_FOR_PULLBACK_OR_CONFIRMATION": "Positive trend, wait for pullback/confirmation",
            "WAIT_FOR_CONFIRMATION": "Watchlist for confirmation",
            "RESEARCH_FOR_POSSIBLE_ENTRY": "Candidate for further research",
            "MONITOR_AND_RESEARCH": "Monitor list",
            "FURTHER_RESEARCH_ONLY": "Research only",
        }

        return mapping.get(action, "Research only")

    def _conditions_to_reconsider(
        self,
        analyst_signal: str,
        final_signal: str,
        risk_level: str,
        model_confidence: str,
        performance_level: str,
        entry_risk_context: bool,
    ) -> List[str]:
        conditions: List[str] = []

        if entry_risk_context:
            conditions.append("Entry risk cools down, such as RSI/overbought pressure easing or price pulling back to a healthier area.")
            conditions.append("Positive trend remains intact after the pullback or consolidation.")
            conditions.append("Analyst signal remains positive and model confidence improves or stays stable.")
        else:
            if analyst_signal not in self.BULLISH_SIGNALS:
                conditions.append("Analyst signal turns bullish or shows stronger momentum.")

            if final_signal != "BUY_CANDIDATE":
                conditions.append("Signal model changes from HOLD/SELL_RISK to BUY_CANDIDATE.")

        if model_confidence != "High":
            conditions.append("Model confidence improves to High.")

        if risk_level not in ["Low", "Medium"]:
            conditions.append("Risk level decreases after risk control.")

        if performance_level == "Needs improvement":
            conditions.append("Evaluator Agent shows improved average reward or stronger historical performance.")

        conditions.append("New report/news or fundamental evidence supports the technical signal.")

        return conditions

    # --------------------------------------------------
    # Main method
    # --------------------------------------------------
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
        validation_result = validation_result or {}
        analysis_result = analysis_result or {}
        training_result = training_result or {}
        signal_result = signal_result or {}
        risk_result = risk_result or {}
        optimizer_result = optimizer_result or {}

        evaluation = self._get_evaluation_result(evaluation_result)

        symbol = self._get_symbol(risk_result, signal_result, analysis_result, validation_result)

        validation_confidence = validation_result.get("confidence", "Unknown")
        validation_next_action = validation_result.get("next_action", "Unknown")

        analyst_signal = self._get_analyst_signal(analysis_result)
        analyst_score = self._get_analyst_score(analysis_result)
        display_signal = self._get_display_signal(analysis_result, signal_result, risk_result)
        entry_risk_level = self._get_entry_risk_level(analysis_result, signal_result, risk_result)
        trend_direction = self._get_trend_direction(analysis_result, signal_result, risk_result)

        final_signal = self._get_final_signal(risk_result=risk_result, signal_result=signal_result)

        risk_level = (
            risk_result.get("risk_level")
            or self._get_nested(risk_result, ["risk_for_next_agent", "risk_level"])
            or "Unknown"
        )

        risk_action = (
            risk_result.get("risk_action")
            or self._get_nested(risk_result, ["risk_for_next_agent", "risk_action"])
            or "Unknown"
        )

        risk_interpretation = (
            risk_result.get("risk_interpretation")
            or self._get_nested(risk_result, ["risk_for_next_agent", "risk_interpretation"])
            or "Unknown"
        )

        model_confidence = self._get_model_confidence(signal_result)

        performance_level = evaluation.get("performance_level", "Unknown")
        average_reward = evaluation.get("average_reward")
        optimizer_improvement = optimizer_result.get("improvement_over_baseline")

        entry_risk_context = self._has_entry_risk_context(
            analyst_signal=analyst_signal,
            analyst_score=analyst_score,
            display_signal=display_signal,
            entry_risk_level=entry_risk_level,
            trend_direction=trend_direction,
            risk_result=risk_result,
        )

        strategy_action = self._base_strategy_action(
            validation_confidence=validation_confidence,
            validation_next_action=validation_next_action,
            final_signal=final_signal,
            risk_level=risk_level,
            model_confidence=model_confidence,
            analyst_signal=analyst_signal,
            performance_level=performance_level,
            entry_risk_context=entry_risk_context,
        )

        strategy_level = self._strategy_level(
            action=strategy_action,
            risk_level=risk_level,
            model_confidence=model_confidence,
            performance_level=performance_level,
        )

        position_guidance = self._position_guidance(
            action=strategy_action,
            final_signal=final_signal,
            risk_level=risk_level,
            entry_risk_context=entry_risk_context,
        )

        leverage_guidance = self._leverage_guidance(
            final_signal=final_signal,
            risk_level=risk_level,
            model_confidence=model_confidence,
            performance_level=performance_level,
            entry_risk_context=entry_risk_context,
        )

        watchlist_status = self._watchlist_status(strategy_action)

        conditions_to_reconsider = self._conditions_to_reconsider(
            analyst_signal=analyst_signal,
            final_signal=final_signal,
            risk_level=risk_level,
            model_confidence=model_confidence,
            performance_level=performance_level,
            entry_risk_context=entry_risk_context,
        )

        risk_note_parts = [
            f"Risk Agent final signal is {final_signal} with risk level {risk_level}.",
            f"Risk action is {risk_action}.",
            f"Risk interpretation is {risk_interpretation}.",
            f"Analyst signal is {analyst_signal}; display signal is {display_signal}; trend direction is {trend_direction}; entry risk level is {entry_risk_level}.",
            f"Model confidence is {model_confidence}.",
        ]

        if performance_level != "Unknown":
            risk_note_parts.append(f"Evaluator performance level is {performance_level}.")

        if average_reward is not None:
            risk_note_parts.append(f"Average historical reward is {average_reward}.")

        if optimizer_improvement is not None:
            risk_note_parts.append(f"Latest optimizer improvement over baseline is {optimizer_improvement}.")

        risk_note = " ".join(risk_note_parts)

        strategy_summary = (
            f"{symbol} is assigned strategy action {strategy_action}. The strategy level is {strategy_level}. "
            f"This is based on final signal {final_signal}, {risk_level} risk, analyst signal {analyst_signal}, "
            f"entry-risk context {entry_risk_context}, and model confidence {model_confidence}."
        )

        reasoning_steps = [
            f"Read validation confidence: {validation_confidence}; next action: {validation_next_action}.",
            f"Read analyst signal: {analyst_signal}; analyst score: {analyst_score}; display signal: {display_signal}.",
            f"Read trend direction: {trend_direction}; entry risk level: {entry_risk_level}; entry-risk context: {entry_risk_context}.",
            f"Read final risk-controlled signal: {final_signal}; risk level: {risk_level}; risk interpretation: {risk_interpretation}.",
            f"Read model confidence: {model_confidence}.",
            f"Read evaluator performance level: {performance_level}.",
            f"Selected strategy action: {strategy_action}.",
            f"Selected strategy level: {strategy_level}.",
        ]

        human_review_required = True

        strategy_for_next_agent = {
            "symbol": symbol,
            "strategy_action": strategy_action,
            "strategy_level": strategy_level,
            "strategy_summary": strategy_summary,
            "position_guidance": position_guidance,
            "leverage_guidance": leverage_guidance,
            "watchlist_status": watchlist_status,
            "conditions_to_reconsider": conditions_to_reconsider,
            "risk_note": risk_note,
            "entry_risk_context": entry_risk_context,
            "entry_risk_level": entry_risk_level,
            "trend_direction": trend_direction,
            "display_signal": display_signal,
            "human_review_required": human_review_required,
        }

        return {
            "success": True,
            "agent_goal": "Convert risk-controlled signals into research-oriented strategy guidance.",
            "symbol": symbol,
            "strategy_action": strategy_action,
            "strategy_level": strategy_level,
            "strategy_summary": strategy_summary,
            "position_guidance": position_guidance,
            "leverage_guidance": leverage_guidance,
            "watchlist_status": watchlist_status,
            "conditions_to_reconsider": conditions_to_reconsider,
            "risk_note": risk_note,
            "entry_risk_context": entry_risk_context,
            "entry_risk_level": entry_risk_level,
            "trend_direction": trend_direction,
            "display_signal": display_signal,
            "analyst_signal": analyst_signal,
            "analyst_score": analyst_score,
            "risk_level": risk_level,
            "final_signal": final_signal,
            "human_review_required": human_review_required,
            "uses_evaluator_context": bool(evaluation),
            "uses_optimizer_context": bool(optimizer_result),
            "reasoning_steps": reasoning_steps,
            "strategy_for_next_agent": strategy_for_next_agent,
            "summary": (
                f"Strategist Agent selected {strategy_action} for {symbol} with {strategy_level} strategy level."
            ),
        }

    # Compatibility aliases
    def generate_strategy(self, *args, **kwargs) -> Dict[str, Any]:
        return self.plan_strategy(*args, **kwargs)

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        return self.plan_strategy(*args, **kwargs)
