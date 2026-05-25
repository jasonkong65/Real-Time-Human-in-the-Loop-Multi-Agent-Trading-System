from typing import Dict, Any, Optional, List


class StrategistAgent:
    """
    Adaptive rule-based Strategist Agent.

    Role:
    - Converts Risk Agent outputs into research-oriented strategy guidance.
    - Uses validation, analyst, signal, risk, evaluator, and optimizer context.
    - Does not execute trades and does not provide personalized financial advice.
    """

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

    def _get_symbol(self, *results: Dict[str, Any]) -> str:
        for result in results:
            if not isinstance(result, dict):
                continue

            symbol = (
                result.get("symbol")
                or self._get_nested(result, ["risk_for_next_agent", "symbol"])
                or self._get_nested(result, ["signal_for_next_agent", "symbol"])
                or self._get_nested(result, ["validation_for_next_agent", "symbol"])
            )

            if symbol:
                return str(symbol).upper()

        return "UNKNOWN"

    def _get_final_signal(
        self,
        risk_result: Dict[str, Any],
        signal_result: Dict[str, Any]
    ) -> str:
        return (
            risk_result.get("final_signal")
            or self._get_nested(risk_result, ["risk_for_next_agent", "final_signal"])
            or signal_result.get("model_signal")
            or signal_result.get("signal")
            or self._get_nested(signal_result, ["signal_for_next_agent", "signal"])
            or "HOLD"
        )

    def _get_model_confidence(self, signal_result: Dict[str, Any]) -> str:
        confidence_level = (
            signal_result.get("confidence_level")
            or self._get_nested(signal_result, ["signal_for_next_agent", "confidence_level"])
        )

        if confidence_level in ["High", "Medium", "Low"]:
            return confidence_level

        confidence = (
            signal_result.get("prediction_confidence")
            or self._get_nested(signal_result, ["signal_for_next_agent", "prediction_confidence"])
        )

        try:
            confidence = float(confidence)
        except Exception:
            return "Unknown"

        if confidence >= 0.65:
            return "High"
        elif confidence >= 0.45:
            return "Medium"
        else:
            return "Low"

    def _get_evaluation_result(
        self,
        evaluation_result: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        if isinstance(evaluation_result, dict) and evaluation_result:
            return evaluation_result

        if not self.auto_load_evaluation:
            return {}

        try:
            from agents.evaluator_agent import EvaluatorAgent
            return EvaluatorAgent().evaluate_history()
        except Exception:
            return {}

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
        performance_level: str
    ) -> str:
        if validation_next_action == "BLOCK_ANALYSIS" or final_signal == "BLOCKED":
            return "NO_ACTION_DATA_OR_RISK_BLOCK"

        if final_signal == "SELL_RISK" or risk_level in ["High", "Critical"]:
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

        if analyst_signal in ["BEARISH_RISK", "BEARISH"]:
            return "RISK_REDUCTION_REVIEW"

        return "FURTHER_RESEARCH_ONLY"

    def _strategy_level(
        self,
        action: str,
        risk_level: str,
        model_confidence: str,
        performance_level: str
    ) -> str:
        if action in ["NO_ACTION_DATA_OR_RISK_BLOCK", "RISK_REDUCTION_REVIEW"]:
            return "Defensive"

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
        risk_level: str
    ) -> str:
        if action == "NO_ACTION_DATA_OR_RISK_BLOCK":
            return (
                "Do not take a paper action from this result. The system should wait until "
                "data quality or risk conditions improve."
            )

        if action == "RISK_REDUCTION_REVIEW":
            return (
                "If this stock is already on a paper watchlist or paper portfolio, review "
                "exposure and downside risk. Avoid adding exposure based only on the current signal."
            )

        if action == "WAIT_FOR_CONFIRMATION":
            return (
                "Treat the stock as a watchlist candidate, not an immediate entry. Wait for "
                "stronger confirmation from analyst signal, model confidence, and risk level."
            )

        if action == "RESEARCH_FOR_POSSIBLE_ENTRY":
            return (
                "The stock can be reviewed as a candidate for further research. Human review "
                "should still check fundamentals, news, valuation, and portfolio risk before any decision."
            )

        if final_signal == "HOLD" or risk_level == "Medium":
            return (
                "Maintain a monitoring stance. Avoid aggressive entry or position increase until "
                "stronger evidence appears."
            )

        return (
            "Use the output as research support only and wait for clearer evidence before "
            "changing exposure."
        )

    def _leverage_guidance(
        self,
        final_signal: str,
        risk_level: str,
        model_confidence: str,
        performance_level: str
    ) -> str:
        if final_signal != "BUY_CANDIDATE":
            return "Do not use leverage. The final signal is not a buy-candidate signal."

        if risk_level != "Low":
            return "Do not use leverage. Risk level is not low."

        if model_confidence != "High":
            return "Do not use leverage. Model confidence is not high."

        if performance_level == "Needs improvement":
            return (
                "Do not use leverage. Historical evaluation currently suggests the system "
                "needs improvement."
            )

        return (
            "Leverage is not supported by this prototype. Any leverage decision would require "
            "separate human review, risk limits, and professional assessment."
        )

    def _watchlist_status(self, action: str) -> str:
        mapping = {
            "NO_ACTION_DATA_OR_RISK_BLOCK": "Do not add based on current data",
            "RISK_REDUCTION_REVIEW": "High caution watchlist",
            "WAIT_FOR_CONFIRMATION": "Watchlist for confirmation",
            "RESEARCH_FOR_POSSIBLE_ENTRY": "Candidate for further research",
            "MONITOR_AND_RESEARCH": "Monitor list",
            "FURTHER_RESEARCH_ONLY": "Research only"
        }

        return mapping.get(action, "Research only")

    def _conditions_to_reconsider(
        self,
        analyst_signal: str,
        final_signal: str,
        risk_level: str,
        model_confidence: str,
        performance_level: str
    ) -> List[str]:
        conditions = []

        if analyst_signal not in ["BULLISH", "BULLISH_MOMENTUM"]:
            conditions.append("Analyst signal turns bullish or shows stronger momentum.")

        if final_signal != "BUY_CANDIDATE":
            conditions.append("Signal model changes from HOLD/SELL_RISK to BUY_CANDIDATE.")

        if model_confidence != "High":
            conditions.append("Model confidence improves to High.")

        if risk_level != "Low":
            conditions.append("Risk level decreases to Low after risk control.")

        if performance_level == "Needs improvement":
            conditions.append(
                "Evaluator Agent shows improved average reward or stronger historical performance."
            )

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
        optimizer_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        validation_result = validation_result or {}
        analysis_result = analysis_result or {}
        training_result = training_result or {}
        signal_result = signal_result or {}
        risk_result = risk_result or {}
        optimizer_result = optimizer_result or {}

        evaluation = self._get_evaluation_result(evaluation_result)

        symbol = self._get_symbol(
            risk_result,
            signal_result,
            analysis_result,
            validation_result
        )

        validation_confidence = validation_result.get("confidence", "Unknown")
        validation_next_action = validation_result.get("next_action", "Unknown")
        analyst_signal = analysis_result.get("analyst_signal", "Unknown")
        analyst_score = analysis_result.get("analyst_score", "Unknown")

        final_signal = self._get_final_signal(
            risk_result=risk_result,
            signal_result=signal_result
        )

        risk_level = (
            risk_result.get("risk_level")
            or self._get_nested(risk_result, ["risk_for_next_agent", "risk_level"])
            or "Unknown"
        )

        risk_action = risk_result.get("risk_action", "Unknown")
        model_confidence = self._get_model_confidence(signal_result)

        performance_level = evaluation.get("performance_level", "Unknown")
        average_reward = evaluation.get("average_reward")
        optimizer_improvement = optimizer_result.get("improvement_over_baseline")

        strategy_action = self._base_strategy_action(
            validation_confidence=validation_confidence,
            validation_next_action=validation_next_action,
            final_signal=final_signal,
            risk_level=risk_level,
            model_confidence=model_confidence,
            analyst_signal=analyst_signal,
            performance_level=performance_level
        )

        strategy_level = self._strategy_level(
            action=strategy_action,
            risk_level=risk_level,
            model_confidence=model_confidence,
            performance_level=performance_level
        )

        position_guidance = self._position_guidance(
            action=strategy_action,
            final_signal=final_signal,
            risk_level=risk_level
        )

        leverage_guidance = self._leverage_guidance(
            final_signal=final_signal,
            risk_level=risk_level,
            model_confidence=model_confidence,
            performance_level=performance_level
        )

        watchlist_status = self._watchlist_status(strategy_action)

        conditions_to_reconsider = self._conditions_to_reconsider(
            analyst_signal=analyst_signal,
            final_signal=final_signal,
            risk_level=risk_level,
            model_confidence=model_confidence,
            performance_level=performance_level
        )

        risk_note_parts = [
            f"Risk Agent final signal is {final_signal} with risk level {risk_level}.",
            f"Risk action is {risk_action}.",
            f"Model confidence is {model_confidence}."
        ]

        if performance_level != "Unknown":
            risk_note_parts.append(f"Evaluator performance level is {performance_level}.")

        if average_reward is not None:
            risk_note_parts.append(f"Average historical reward is {average_reward}.")

        if optimizer_improvement is not None:
            risk_note_parts.append(
                f"Latest optimizer improvement over baseline is {optimizer_improvement}."
            )

        risk_note = " ".join(risk_note_parts)

        strategy_summary = (
            f"{symbol} is assigned strategy action {strategy_action}. "
            f"The strategy level is {strategy_level}. "
            f"This is based on {final_signal}, {risk_level} risk, analyst signal "
            f"{analyst_signal}, and model confidence {model_confidence}."
        )

        reasoning_steps = [
            f"Read validation confidence: {validation_confidence}; next action: {validation_next_action}.",
            f"Read analyst signal: {analyst_signal}; analyst score: {analyst_score}.",
            f"Read final risk-controlled signal: {final_signal}; risk level: {risk_level}.",
            f"Read model confidence: {model_confidence}.",
            f"Read evaluator performance level: {performance_level}.",
            f"Selected strategy action: {strategy_action}.",
            f"Selected strategy level: {strategy_level}."
        ]

        human_review_required = True

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
            "human_review_required": human_review_required,
            "uses_evaluator_context": bool(evaluation),
            "uses_optimizer_context": bool(optimizer_result),
            "reasoning_steps": reasoning_steps,
            "strategy_for_next_agent": {
                "symbol": symbol,
                "strategy_action": strategy_action,
                "strategy_level": strategy_level,
                "strategy_summary": strategy_summary,
                "position_guidance": position_guidance,
                "leverage_guidance": leverage_guidance,
                "watchlist_status": watchlist_status,
                "conditions_to_reconsider": conditions_to_reconsider,
                "risk_note": risk_note,
                "human_review_required": human_review_required
            },
            "summary": (
                f"Strategist Agent selected {strategy_action} for {symbol} "
                f"with {strategy_level} strategy level."
            )
        }

    # Compatibility aliases
    def generate_strategy(self, *args, **kwargs) -> Dict[str, Any]:
        return self.plan_strategy(*args, **kwargs)

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        return self.plan_strategy(*args, **kwargs)
