from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import random
import joblib


class RiskAgent:
    """
    Optimized Hybrid Risk Agent.

    Role:
    1. Applies hard safety rules to prevent unsafe paper decisions.
    2. Uses a filtered Q-learning advisory layer.
    3. Separates true downside risk from positive-trend-but-entry-risk situations.

    Key improvement:
    - If Analyst Agent reports POSITIVE_BUT_ENTRY_RISK / overbought / elevated entry risk,
      the Risk Agent no longer treats the stock as a simple low-risk HOLD.
    - It outputs Medium risk for timing/entry risk and lets the Strategist Agent give
      WAIT_FOR_PULLBACK_OR_CONFIRMATION guidance.
    """

    ACTIONS = ["KEEP_SIGNAL", "DOWNGRADE_TO_HOLD", "BLOCK_TRADE"]

    ACTION_PRIORITY = {
        "KEEP_SIGNAL": 0,
        "DOWNGRADE_TO_HOLD": 1,
        "BLOCK_TRADE": 2,
    }

    VALID_SIGNALS = [
        "BUY_CANDIDATE",
        "HOLD",
        "SELL_RISK",
        "BLOCKED",
    ]

    ENTRY_RISK_ANALYST_SIGNALS = {
        "POSITIVE_BUT_ENTRY_RISK",
        "WATCHLIST_BULLISH_ENTRY_RISK",
        "BUY_WATCHLIST_OVERBOUGHT",
        "BUY_WATCHLIST_ENTRY_RISK",
        "POSITIVE_BUT_OVERBOUGHT",
    }

    POSITIVE_ANALYST_SIGNALS = {
        "BULLISH",
        "BULLISH_MOMENTUM",
        "WATCHLIST_BULLISH",
        "POSITIVE",
        "POSITIVE_BUT_ENTRY_RISK",
        "WATCHLIST_BULLISH_ENTRY_RISK",
        "BUY_WATCHLIST_OVERBOUGHT",
        "BUY_WATCHLIST_ENTRY_RISK",
        "POSITIVE_BUT_OVERBOUGHT",
    }

    BEARISH_ANALYST_SIGNALS = {
        "BEARISH_RISK",
        "BEARISH",
        "DOWNSIDE_RISK",
        "NEGATIVE",
    }

    def __init__(
        self,
        q_table_path: str = "models/risk_q_table.pkl",
        alpha: float = 0.2,
        gamma: float = 0.9,
        epsilon: float = 0.03,
    ):
        self.q_table_path = Path(q_table_path)
        self.q_table_path.parent.mkdir(parents=True, exist_ok=True)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = self._load_q_table()

    # --------------------------------------------------
    # Generic helpers
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

    def _safe_float(self, value, default: Optional[float] = 0.0):
        try:
            if value is None:
                return default
            return float(value)
        except Exception:
            return default

    def _normalise_signal(self, value: Any, default: str = "Unknown") -> str:
        if value is None:
            return default
        return str(value).strip().upper()

    # --------------------------------------------------
    # Q-table helpers
    # --------------------------------------------------
    def _load_q_table(self) -> dict:
        try:
            if self.q_table_path.exists() and self.q_table_path.stat().st_size > 0:
                table = joblib.load(self.q_table_path)
                if isinstance(table, dict):
                    return table
        except Exception:
            pass
        return {}

    def _save_q_table(self):
        joblib.dump(self.q_table, self.q_table_path)

    def _init_state(self, state: str):
        if not state:
            state = "Unknown|Unknown|HOLD|Unknown|Unknown|NoEntryRisk"

        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.ACTIONS}

        for action in self.ACTIONS:
            self.q_table[state].setdefault(action, 0.0)

    # --------------------------------------------------
    # Input extraction helpers
    # --------------------------------------------------
    def _discretize_confidence(self, validation_result: Dict[str, Any]) -> str:
        if not isinstance(validation_result, dict):
            return "Unknown"

        confidence = (
            validation_result.get("confidence")
            or self._get_nested(validation_result, ["validation_for_next_agent", "confidence"])
        )

        if isinstance(confidence, str):
            confidence = confidence.strip().capitalize()
            if confidence in ["High", "Medium", "Low"]:
                return confidence

        numeric_confidence = self._safe_float(confidence, default=None)

        if numeric_confidence is None:
            return "Unknown"
        if numeric_confidence >= 0.75:
            return "High"
        if numeric_confidence >= 0.45:
            return "Medium"
        return "Low"

    def _get_validation_next_action(self, validation_result: Dict[str, Any]) -> str:
        if not isinstance(validation_result, dict):
            return "Unknown"

        return (
            validation_result.get("next_action")
            or self._get_nested(validation_result, ["validation_for_next_agent", "next_action"])
            or "Unknown"
        )

    def _get_model_signal(self, signal_result: Dict[str, Any]) -> str:
        if not isinstance(signal_result, dict):
            return "HOLD"

        signal = (
            signal_result.get("model_signal")
            or signal_result.get("signal")
            or signal_result.get("final_model_signal")
            or self._get_nested(signal_result, ["signal_for_next_agent", "model_signal"])
            or self._get_nested(signal_result, ["signal_for_next_agent", "signal"])
            or self._get_nested(signal_result, ["signal_for_next_agent", "final_model_signal"])
            or "HOLD"
        )

        signal = self._normalise_signal(signal, default="HOLD")
        return signal if signal in self.VALID_SIGNALS else "HOLD"

    def _get_model_confidence_level(self, signal_result: Dict[str, Any]) -> str:
        if not isinstance(signal_result, dict):
            return "Unknown"

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

        confidence = self._get_prediction_confidence(signal_result)

        if confidence is None:
            return "Unknown"
        if confidence >= 0.65:
            return "High"
        if confidence >= 0.45:
            return "Medium"
        return "Low"

    def _get_prediction_confidence(self, signal_result: Dict[str, Any]) -> Optional[float]:
        if not isinstance(signal_result, dict):
            return None

        confidence = (
            signal_result.get("prediction_confidence")
            or signal_result.get("confidence")
            or self._get_nested(signal_result, ["signal_for_next_agent", "prediction_confidence"])
            or self._get_nested(signal_result, ["signal_for_next_agent", "confidence"])
        )

        return self._safe_float(confidence, default=None)

    def _get_analyst_signal(self, analysis_result: Dict[str, Any]) -> str:
        if not isinstance(analysis_result, dict):
            return "Unknown"

        signal = (
            analysis_result.get("analyst_signal")
            or self._get_nested(analysis_result, ["analysis_for_next_agent", "analyst_signal"])
            or "Unknown"
        )

        return self._normalise_signal(signal, default="Unknown")

    def _get_analyst_score(self, analysis_result: Dict[str, Any]) -> Optional[float]:
        if not isinstance(analysis_result, dict):
            return None

        score = (
            analysis_result.get("analyst_score")
            or self._get_nested(analysis_result, ["analysis_for_next_agent", "analyst_score"])
        )

        return self._safe_float(score, default=None)

    def _get_display_signal(self, analysis_result: Dict[str, Any], signal_result: Dict[str, Any]) -> str:
        display = None

        if isinstance(analysis_result, dict):
            display = (
                analysis_result.get("display_signal")
                or self._get_nested(analysis_result, ["analysis_for_next_agent", "display_signal"])
            )

        if not display and isinstance(signal_result, dict):
            display = (
                signal_result.get("display_signal")
                or signal_result.get("enhanced_signal")
                or self._get_nested(signal_result, ["signal_for_next_agent", "display_signal"])
                or self._get_nested(signal_result, ["signal_for_next_agent", "enhanced_signal"])
            )

        return self._normalise_signal(display, default="Unknown")

    def _get_entry_risk_level(self, analysis_result: Dict[str, Any], signal_result: Dict[str, Any]) -> str:
        value = None

        if isinstance(analysis_result, dict):
            value = (
                analysis_result.get("entry_risk_level")
                or analysis_result.get("entry_risk")
                or analysis_result.get("overbought_risk")
                or self._get_nested(analysis_result, ["analysis_for_next_agent", "entry_risk_level"])
                or self._get_nested(analysis_result, ["analysis_for_next_agent", "entry_risk"])
            )

        if value is None and isinstance(signal_result, dict):
            value = (
                signal_result.get("entry_risk_level")
                or signal_result.get("entry_risk")
                or self._get_nested(signal_result, ["signal_for_next_agent", "entry_risk_level"])
                or self._get_nested(signal_result, ["signal_for_next_agent", "entry_risk"])
            )

        if value is None:
            return "Unknown"

        text = str(value).strip().replace("_", " ").title()

        if text in ["High", "Elevated", "Medium High", "Moderate High", "Medium", "Low"]:
            return text

        return str(value).strip()

    def _get_trend_direction(self, analysis_result: Dict[str, Any], signal_result: Dict[str, Any]) -> str:
        value = None

        if isinstance(analysis_result, dict):
            value = (
                analysis_result.get("trend_direction")
                or analysis_result.get("trend")
                or self._get_nested(analysis_result, ["analysis_for_next_agent", "trend_direction"])
                or self._get_nested(analysis_result, ["analysis_for_next_agent", "trend"])
            )

        if value is None and isinstance(signal_result, dict):
            market_context = signal_result.get("market_context", {})
            if isinstance(market_context, dict):
                value = market_context.get("trend_direction") or market_context.get("trend")

        if value is None:
            return "Unknown"

        return str(value).strip().title()

    def _get_volatility_level(self, analysis_result: Dict[str, Any]) -> str:
        if not isinstance(analysis_result, dict):
            return "Unknown"

        volatility = (
            analysis_result.get("volatility_level")
            or analysis_result.get("historical_volatility_level")
            or self._get_nested(analysis_result, ["analysis_for_next_agent", "volatility_level"])
            or self._get_nested(analysis_result, ["analysis_for_next_agent", "historical_volatility_level"])
            or "Unknown"
        )

        if isinstance(volatility, str):
            text = volatility.strip().capitalize()
            if text in ["High", "Medium", "Low"]:
                return text

        return str(volatility)

    def _get_symbol(
        self,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any],
    ) -> str:
        symbol = None

        for result in [signal_result, analysis_result, validation_result]:
            if not isinstance(result, dict):
                continue

            symbol = (
                result.get("symbol")
                or self._get_nested(result, ["signal_for_next_agent", "symbol"])
                or self._get_nested(result, ["analysis_for_next_agent", "symbol"])
                or self._get_nested(result, ["validation_for_next_agent", "symbol"])
            )

            if symbol:
                break

        return str(symbol or "UNKNOWN").upper()

    def _has_positive_entry_risk_context(
        self,
        analyst_signal: str,
        analyst_score: Optional[float],
        display_signal: str,
        entry_risk_level: str,
        trend_direction: str,
    ) -> bool:
        analyst_signal = self._normalise_signal(analyst_signal, default="Unknown")
        display_signal = self._normalise_signal(display_signal, default="Unknown")
        entry_text = str(entry_risk_level or "").lower()
        trend_text = str(trend_direction or "").lower()

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
    # State construction
    # --------------------------------------------------
    def _build_state(
        self,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any],
    ) -> str:
        validation_confidence = self._discretize_confidence(validation_result)
        volatility = self._get_volatility_level(analysis_result)
        model_signal = self._get_model_signal(signal_result)
        analyst_signal = self._get_analyst_signal(analysis_result)
        analyst_score = self._get_analyst_score(analysis_result)
        model_confidence = self._get_model_confidence_level(signal_result)
        display_signal = self._get_display_signal(analysis_result, signal_result)
        entry_risk_level = self._get_entry_risk_level(analysis_result, signal_result)
        trend_direction = self._get_trend_direction(analysis_result, signal_result)

        entry_context = "EntryRisk" if self._has_positive_entry_risk_context(
            analyst_signal=analyst_signal,
            analyst_score=analyst_score,
            display_signal=display_signal,
            entry_risk_level=entry_risk_level,
            trend_direction=trend_direction,
        ) else "NoEntryRisk"

        return (
            f"{validation_confidence}|"
            f"{volatility}|"
            f"{model_signal}|"
            f"{analyst_signal}|"
            f"{model_confidence}|"
            f"{entry_context}"
        )

    # --------------------------------------------------
    # Rule-based hard safety layer
    # --------------------------------------------------
    def _rule_based_safety_action(
        self,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any],
    ) -> Tuple[str, List[str], bool]:
        validation_confidence = self._discretize_confidence(validation_result)
        validation_next_action = self._get_validation_next_action(validation_result)
        volatility = self._get_volatility_level(analysis_result)
        model_signal = self._get_model_signal(signal_result)
        analyst_signal = self._get_analyst_signal(analysis_result)
        analyst_score = self._get_analyst_score(analysis_result)
        model_confidence = self._get_model_confidence_level(signal_result)
        display_signal = self._get_display_signal(analysis_result, signal_result)
        entry_risk_level = self._get_entry_risk_level(analysis_result, signal_result)
        trend_direction = self._get_trend_direction(analysis_result, signal_result)

        positive_entry_risk = self._has_positive_entry_risk_context(
            analyst_signal=analyst_signal,
            analyst_score=analyst_score,
            display_signal=display_signal,
            entry_risk_level=entry_risk_level,
            trend_direction=trend_direction,
        )

        reasons: List[str] = []

        if validation_next_action == "BLOCK_ANALYSIS":
            reasons.append("Validation Agent blocked downstream analysis.")
            return "BLOCK_TRADE", reasons, True

        if validation_confidence == "Low":
            if model_signal == "BUY_CANDIDATE":
                reasons.append("Validation confidence is Low while model signal is BUY_CANDIDATE.")
                return "BLOCK_TRADE", reasons, True

            reasons.append("Validation confidence is Low, so the signal is downgraded to HOLD.")
            return "DOWNGRADE_TO_HOLD", reasons, False

        if positive_entry_risk and model_signal == "BUY_CANDIDATE":
            reasons.append(
                "Analyst Agent detected positive trend with elevated entry timing risk, so BUY_CANDIDATE is softened to HOLD."
            )
            return "DOWNGRADE_TO_HOLD", reasons, False

        if positive_entry_risk and model_signal == "SELL_RISK":
            reasons.append(
                "Model predicted SELL_RISK, but Analyst Agent indicates a positive trend with entry risk rather than clear downside risk. The signal is softened to HOLD."
            )
            return "DOWNGRADE_TO_HOLD", reasons, False

        if model_confidence == "Low" and model_signal == "BUY_CANDIDATE":
            reasons.append("Model confidence is Low while signal is BUY_CANDIDATE.")
            return "DOWNGRADE_TO_HOLD", reasons, False

        if volatility == "High" and model_signal == "BUY_CANDIDATE":
            reasons.append("High volatility detected while model signal is BUY_CANDIDATE.")
            return "DOWNGRADE_TO_HOLD", reasons, False

        if validation_confidence == "Medium" and model_signal == "BUY_CANDIDATE":
            reasons.append("Validation confidence is only Medium while model signal is BUY_CANDIDATE.")
            return "DOWNGRADE_TO_HOLD", reasons, False

        if model_signal == "SELL_RISK":
            reasons.append("Model detected SELL_RISK, which is already a cautious signal.")
            return "KEEP_SIGNAL", reasons, False

        if analyst_signal in self.BEARISH_ANALYST_SIGNALS:
            reasons.append(f"Analyst Agent detected {analyst_signal}.")
            return "KEEP_SIGNAL", reasons, False

        reasons.append("No hard risk rule was triggered.")
        return "KEEP_SIGNAL", reasons, False

    # --------------------------------------------------
    # Q-learning advisory layer
    # --------------------------------------------------
    def _default_q_action(self, state: str) -> str:
        parts = state.split("|")

        validation_confidence = parts[0] if len(parts) > 0 else "Unknown"
        volatility = parts[1] if len(parts) > 1 else "Unknown"
        model_signal = parts[2] if len(parts) > 2 else "HOLD"
        model_confidence = parts[4] if len(parts) > 4 else "Unknown"
        entry_context = parts[5] if len(parts) > 5 else "NoEntryRisk"

        if validation_confidence == "Low" and model_signal == "BUY_CANDIDATE":
            return "BLOCK_TRADE"

        if model_signal == "BUY_CANDIDATE":
            if model_confidence == "Low" or volatility == "High" or validation_confidence == "Medium":
                return "DOWNGRADE_TO_HOLD"

            if entry_context == "EntryRisk":
                return "DOWNGRADE_TO_HOLD"

        if model_signal == "SELL_RISK" and entry_context == "EntryRisk":
            return "DOWNGRADE_TO_HOLD"

        return "KEEP_SIGNAL"

    def _choose_q_action(self, state: str) -> str:
        self._init_state(state)
        values = self.q_table[state]

        if all(value == 0.0 for value in values.values()):
            return self._default_q_action(state)

        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)

        max_q = max(values.values())
        best_actions = [action for action, value in values.items() if value == max_q]

        best_actions = sorted(
            best_actions,
            key=lambda action: self.ACTION_PRIORITY.get(action, 0),
            reverse=True,
        )

        return best_actions[0]

    def _filter_q_action(
        self,
        raw_q_action: str,
        model_signal: str,
        validation_confidence: str,
        validation_next_action: str,
        volatility_level: str,
        model_confidence: str,
        hard_block_triggered: bool,
        positive_entry_risk: bool,
    ) -> Tuple[str, str]:
        if raw_q_action not in self.ACTIONS:
            return "KEEP_SIGNAL", "Invalid Q-learning action was normalized to KEEP_SIGNAL."

        if raw_q_action == "BLOCK_TRADE":
            if hard_block_triggered:
                return "BLOCK_TRADE", "Q-learning BLOCK_TRADE kept because hard block was triggered."

            if validation_next_action == "BLOCK_ANALYSIS":
                return "BLOCK_TRADE", "Q-learning BLOCK_TRADE kept because Validation Agent blocked analysis."

            if validation_confidence == "Low" and model_signal == "BUY_CANDIDATE":
                return "BLOCK_TRADE", "Q-learning BLOCK_TRADE kept because validation is Low and signal is BUY_CANDIDATE."

            if model_signal == "BUY_CANDIDATE" and (
                volatility_level == "High" or model_confidence == "Low" or positive_entry_risk
            ):
                return "DOWNGRADE_TO_HOLD", (
                    "Q-learning suggested BLOCK_TRADE, but no hard block was triggered. "
                    "For a risky BUY_CANDIDATE, it was softened to DOWNGRADE_TO_HOLD."
                )

            return "KEEP_SIGNAL", (
                "Q-learning suggested BLOCK_TRADE, but no hard block was triggered. "
                "To avoid over-blocking, it was filtered to KEEP_SIGNAL."
            )

        if raw_q_action == "DOWNGRADE_TO_HOLD":
            if model_signal == "BUY_CANDIDATE":
                return "DOWNGRADE_TO_HOLD", "Q-learning DOWNGRADE_TO_HOLD is allowed for BUY_CANDIDATE."

            if model_signal == "SELL_RISK" and positive_entry_risk:
                return "DOWNGRADE_TO_HOLD", (
                    "Q-learning DOWNGRADE_TO_HOLD is allowed because SELL_RISK conflicts with positive-trend entry-risk context."
                )

            return "KEEP_SIGNAL", (
                "Q-learning suggested DOWNGRADE_TO_HOLD, but the signal is not a risky BUY_CANDIDATE or positive-entry-risk SELL_RISK. "
                "It was filtered to KEEP_SIGNAL."
            )

        return "KEEP_SIGNAL", "Q-learning suggested KEEP_SIGNAL."

    # --------------------------------------------------
    # Final action and signal
    # --------------------------------------------------
    def _combine_rule_and_q_action(
        self,
        rule_action: str,
        filtered_q_action: str,
        hard_block_triggered: bool,
    ) -> str:
        if hard_block_triggered or rule_action == "BLOCK_TRADE":
            return "BLOCK_TRADE"

        if rule_action == "DOWNGRADE_TO_HOLD":
            return "DOWNGRADE_TO_HOLD"

        if filtered_q_action == "DOWNGRADE_TO_HOLD":
            return "DOWNGRADE_TO_HOLD"

        return "KEEP_SIGNAL"

    def _apply_risk_action(self, model_signal: str, risk_action: str) -> str:
        if risk_action == "BLOCK_TRADE":
            return "BLOCKED"

        if risk_action == "DOWNGRADE_TO_HOLD":
            if model_signal in ["BUY_CANDIDATE", "SELL_RISK"]:
                return "HOLD"
            return model_signal

        return model_signal

    def _risk_level(
        self,
        validation_confidence: str,
        volatility_level: str,
        model_confidence: str,
        final_signal: str,
        risk_action: str,
        positive_entry_risk: bool,
    ) -> str:
        if risk_action == "BLOCK_TRADE" or final_signal == "BLOCKED":
            return "Critical"

        if final_signal == "SELL_RISK":
            return "High"

        if volatility_level == "High":
            return "High"

        if validation_confidence == "Low":
            return "High"

        # Main optimization: positive trend with elevated entry/chasing risk is not Low risk.
        if positive_entry_risk:
            return "Medium"

        if risk_action == "DOWNGRADE_TO_HOLD":
            return "Medium"

        if validation_confidence == "Medium":
            return "Medium"

        if model_confidence == "Low":
            return "Medium"

        return "Low"

    # --------------------------------------------------
    # Main method
    # --------------------------------------------------
    def assess_risk(
        self,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        agent_goal = (
            "Apply hard safety rules and filtered Q-learning risk adjustment while separating downside risk from entry timing risk."
        )

        validation_result = validation_result if isinstance(validation_result, dict) else {}
        analysis_result = analysis_result if isinstance(analysis_result, dict) else {}
        signal_result = signal_result if isinstance(signal_result, dict) else {}

        state = self._build_state(validation_result, analysis_result, signal_result)
        self._init_state(state)

        symbol = self._get_symbol(validation_result, analysis_result, signal_result)

        validation_confidence = self._discretize_confidence(validation_result)
        validation_next_action = self._get_validation_next_action(validation_result)
        volatility_level = self._get_volatility_level(analysis_result)
        analyst_signal = self._get_analyst_signal(analysis_result)
        analyst_score = self._get_analyst_score(analysis_result)
        display_signal = self._get_display_signal(analysis_result, signal_result)
        entry_risk_level = self._get_entry_risk_level(analysis_result, signal_result)
        trend_direction = self._get_trend_direction(analysis_result, signal_result)

        model_signal = self._get_model_signal(signal_result)
        model_confidence = self._get_model_confidence_level(signal_result)
        prediction_confidence = self._get_prediction_confidence(signal_result)

        positive_entry_risk = self._has_positive_entry_risk_context(
            analyst_signal=analyst_signal,
            analyst_score=analyst_score,
            display_signal=display_signal,
            entry_risk_level=entry_risk_level,
            trend_direction=trend_direction,
        )

        rule_action, rule_reasons, hard_block_triggered = self._rule_based_safety_action(
            validation_result=validation_result,
            analysis_result=analysis_result,
            signal_result=signal_result,
        )

        raw_q_learning_action = self._choose_q_action(state)

        filtered_q_learning_action, q_filter_reason = self._filter_q_action(
            raw_q_action=raw_q_learning_action,
            model_signal=model_signal,
            validation_confidence=validation_confidence,
            validation_next_action=validation_next_action,
            volatility_level=volatility_level,
            model_confidence=model_confidence,
            hard_block_triggered=hard_block_triggered,
            positive_entry_risk=positive_entry_risk,
        )

        final_risk_action = self._combine_rule_and_q_action(
            rule_action=rule_action,
            filtered_q_action=filtered_q_learning_action,
            hard_block_triggered=hard_block_triggered,
        )

        final_signal = self._apply_risk_action(
            model_signal=model_signal,
            risk_action=final_risk_action,
        )

        risk_level = self._risk_level(
            validation_confidence=validation_confidence,
            volatility_level=volatility_level,
            model_confidence=model_confidence,
            final_signal=final_signal,
            risk_action=final_risk_action,
            positive_entry_risk=positive_entry_risk,
        )

        q_values_for_state = self.q_table.get(state, {})

        risk_interpretation = (
            "Entry timing risk / possible chase risk"
            if positive_entry_risk and final_signal == "HOLD"
            else "Downside or data-quality risk"
            if risk_level in ["High", "Critical"] or final_signal in ["SELL_RISK", "BLOCKED"]
            else "Normal monitoring risk"
        )

        reasoning_steps = [
            f"Built Q-learning state: {state}.",
            f"Validation confidence: {validation_confidence}; validation next action: {validation_next_action}.",
            f"Analyst signal: {analyst_signal}; analyst score: {analyst_score}; display signal: {display_signal}.",
            f"Trend direction: {trend_direction}; entry risk level: {entry_risk_level}; positive entry-risk context: {positive_entry_risk}.",
            f"Volatility level: {volatility_level}.",
            f"Original model signal: {model_signal}.",
            f"Model confidence level: {model_confidence}; prediction confidence: {prediction_confidence}.",
            f"Rule-based hard safety layer suggested: {rule_action}.",
            f"Hard block triggered: {hard_block_triggered}.",
            f"Rule reasons: {'; '.join(rule_reasons)}",
            f"Raw Q-learning advisory action: {raw_q_learning_action}.",
            f"Filtered Q-learning action: {filtered_q_learning_action}.",
            f"Q-learning filter reason: {q_filter_reason}",
            f"Final risk action selected: {final_risk_action}.",
            f"Final signal after risk adjustment: {final_signal}.",
            f"Estimated risk level: {risk_level}; interpretation: {risk_interpretation}.",
        ]

        explanation_for_llm = (
            f"The original model signal for {symbol} was {model_signal} with {model_confidence.lower()} model confidence. "
            f"Validation confidence was {validation_confidence}, analyst signal was {analyst_signal}, trend direction was {trend_direction}, "
            f"entry risk level was {entry_risk_level}, and volatility was {volatility_level}. "
            f"Positive entry-risk context was {positive_entry_risk}. The hard safety layer suggested {rule_action}. "
            f"The raw Q-learning advisory action was {raw_q_learning_action}, and after filtering it became {filtered_q_learning_action}. "
            f"The final risk action is {final_risk_action}, so the final signal becomes {final_signal}. "
            f"The estimated risk level is {risk_level}, mainly interpreted as {risk_interpretation}."
        )

        risk_for_next_agent = {
            "symbol": symbol,
            "original_signal": model_signal,
            "final_signal": final_signal,
            "risk_level": risk_level,
            "risk_action": final_risk_action,
            "risk_interpretation": risk_interpretation,
            "q_state": state,
            "validation_confidence": validation_confidence,
            "validation_next_action": validation_next_action,
            "analyst_signal": analyst_signal,
            "analyst_score": analyst_score,
            "display_signal": display_signal,
            "trend_direction": trend_direction,
            "entry_risk_level": entry_risk_level,
            "positive_entry_risk_context": positive_entry_risk,
            "volatility_level": volatility_level,
            "model_confidence_level": model_confidence,
            "prediction_confidence": prediction_confidence,
            "rule_based_action": rule_action,
            "rule_reasons": rule_reasons,
            "hard_block_triggered": hard_block_triggered,
            "raw_q_learning_action": raw_q_learning_action,
            "filtered_q_learning_action": filtered_q_learning_action,
            "q_filter_reason": q_filter_reason,
            "q_values_for_state": q_values_for_state,
            "human_review_required": True,
            "explanation_for_llm": explanation_for_llm,
        }

        return {
            "success": True,
            "agent_goal": agent_goal,
            "symbol": symbol,
            "q_state": state,
            "original_signal": model_signal,
            "model_confidence_level": model_confidence,
            "prediction_confidence": prediction_confidence,
            "validation_confidence": validation_confidence,
            "validation_next_action": validation_next_action,
            "analyst_signal": analyst_signal,
            "analyst_score": analyst_score,
            "display_signal": display_signal,
            "trend_direction": trend_direction,
            "entry_risk_level": entry_risk_level,
            "positive_entry_risk_context": positive_entry_risk,
            "volatility_level": volatility_level,
            "rule_based_action": rule_action,
            "rule_reasons": rule_reasons,
            "hard_block_triggered": hard_block_triggered,
            "raw_q_learning_action": raw_q_learning_action,
            "filtered_q_learning_action": filtered_q_learning_action,
            "q_filter_reason": q_filter_reason,
            "risk_action": final_risk_action,
            "final_signal": final_signal,
            "risk_level": risk_level,
            "risk_interpretation": risk_interpretation,
            "q_values_for_state": q_values_for_state,
            "q_table_path": str(self.q_table_path),
            "human_review_required": True,
            "agent_decision": explanation_for_llm,
            "reasoning_steps": reasoning_steps,
            "explanation_for_llm": explanation_for_llm,
            "risk_for_next_agent": risk_for_next_agent,
            "summary": (
                f"Risk Agent adjusted signal from {model_signal} to {final_signal} "
                f"with risk level {risk_level}."
            ),
        }

    # --------------------------------------------------
    # Compatibility aliases for app.py
    # --------------------------------------------------
    def apply_risk_control(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.assess_risk(
            validation_result=validation_result or {},
            analysis_result=analysis_result or {},
            signal_result=signal_result or {},
        )

    def adjust_risk(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.apply_risk_control(
            signal_result=signal_result,
            analysis_result=analysis_result,
            validation_result=validation_result,
        )

    def evaluate_risk(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.apply_risk_control(
            signal_result=signal_result,
            analysis_result=analysis_result,
            validation_result=validation_result,
        )

    def control_risk(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.apply_risk_control(
            signal_result=signal_result,
            analysis_result=analysis_result,
            validation_result=validation_result,
        )

    def run(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self.apply_risk_control(
            signal_result=signal_result,
            analysis_result=analysis_result,
            validation_result=validation_result,
        )

    # --------------------------------------------------
    # Reward calculation and Q-learning update
    # --------------------------------------------------
    def calculate_reward(
        self,
        final_signal: str,
        future_return: float,
        volatility_level: str = "Unknown",
    ) -> float:
        future_return = self._safe_float(future_return, default=0.0) or 0.0

        if final_signal == "BUY_CANDIDATE":
            reward = future_return
        elif final_signal == "SELL_RISK":
            reward = -future_return
        elif final_signal == "HOLD":
            reward = -abs(future_return) * 0.2
        elif final_signal == "BLOCKED":
            if future_return < -0.015:
                reward = abs(future_return)
            elif future_return > 0.015:
                reward = -future_return * 0.5
            else:
                reward = 0.005
        else:
            reward = 0.0

        if volatility_level in ["High", "Critical"]:
            reward -= 0.005

        return float(reward)

    def update_q_value(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not state:
            return {
                "success": False,
                "summary": "Cannot update Q-table because state is missing.",
            }

        if action not in self.ACTIONS:
            action = "KEEP_SIGNAL"

        reward = self._safe_float(reward, default=0.0) or 0.0
        self._init_state(state)

        current_q = self.q_table[state].get(action, 0.0)

        if next_state:
            self._init_state(next_state)
            max_next_q = max(self.q_table[next_state].values())
        else:
            max_next_q = 0.0

        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
        self._save_q_table()

        return {
            "success": True,
            "state": state,
            "action": action,
            "reward": round(reward, 6),
            "old_q": round(current_q, 6),
            "new_q": round(new_q, 6),
            "q_table_path": str(self.q_table_path),
            "summary": (
                f"Updated Q-value for state={state}, action={action}: "
                f"{current_q:.4f} → {new_q:.4f}."
            ),
        }

    def update_from_feedback(
        self,
        risk_result: Dict[str, Any],
        future_return: float,
    ) -> Dict[str, Any]:
        if not isinstance(risk_result, dict):
            return {
                "success": False,
                "summary": "Cannot update Q-table because risk_result is invalid.",
            }

        state = risk_result.get("q_state")
        action = risk_result.get("risk_action")
        final_signal = risk_result.get("final_signal")
        risk_level = risk_result.get("risk_level")
        volatility_level = risk_result.get("volatility_level")

        if not state or not action:
            return {
                "success": False,
                "summary": "Cannot update Q-table because state or action is missing.",
            }

        if not volatility_level:
            volatility_level = "High" if risk_level in ["High", "Critical"] else "Low"

        reward = self.calculate_reward(
            final_signal=final_signal,
            future_return=future_return,
            volatility_level=volatility_level,
        )

        update_result = self.update_q_value(
            state=state,
            action=action,
            reward=reward,
        )

        update_result["final_signal"] = final_signal
        update_result["risk_level"] = risk_level
        update_result["future_return"] = self._safe_float(future_return, default=0.0)
        update_result["calculated_reward"] = round(reward, 6)

        return update_result
