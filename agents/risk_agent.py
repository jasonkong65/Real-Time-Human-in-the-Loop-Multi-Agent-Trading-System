from pathlib import Path
import random
import joblib


class RiskAgent:
    """
    Hybrid Risk Agent:
    1. Rule-based safety layer
    2. Q-learning risk adjustment layer
    3. Final risk decision

    The Risk Agent does not execute trades.
    It adjusts or blocks trading signals before user confirmation.
    """

    ACTIONS = ["KEEP_SIGNAL", "DOWNGRADE_TO_HOLD", "BLOCK_TRADE"]

    ACTION_PRIORITY = {
        "KEEP_SIGNAL": 0,
        "DOWNGRADE_TO_HOLD": 1,
        "BLOCK_TRADE": 2
    }

    def __init__(
        self,
        q_table_path: str = "models/risk_q_table.pkl",
        alpha: float = 0.2,
        gamma: float = 0.9,
        epsilon: float = 0.05
    ):
        self.q_table_path = Path(q_table_path)
        self.q_table_path.parent.mkdir(parents=True, exist_ok=True)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.q_table = self._load_q_table()

    def _load_q_table(self) -> dict:
        if self.q_table_path.exists():
            try:
                return joblib.load(self.q_table_path)
            except Exception:
                return {}
        return {}

    def _save_q_table(self):
        joblib.dump(self.q_table, self.q_table_path)

    def _init_state(self, state: str):
        if state not in self.q_table:
            self.q_table[state] = {action: 0.0 for action in self.ACTIONS}

    def _discretize_confidence(self, validation_result: dict) -> str:
        confidence = validation_result.get("confidence", "Unknown")

        if confidence in ["High", "Medium", "Low"]:
            return confidence

        score = validation_result.get("confidence_score", 0.5)

        if score >= 0.8:
            return "High"
        elif score >= 0.5:
            return "Medium"
        else:
            return "Low"

    def _get_model_signal(self, signal_result: dict) -> str:
        """
        Expected input from Training Agent / Signal Model:
        model_signal = BUY_CANDIDATE / HOLD / SELL_RISK
        """
        if not signal_result:
            return "HOLD"

        return signal_result.get("model_signal") or signal_result.get("signal") or "HOLD"

    def _get_model_confidence_level(self, signal_result: dict) -> str:
        """
        Get confidence level from Signal Model.
        """
        if not signal_result:
            return "Unknown"

        confidence_level = signal_result.get("confidence_level")

        if confidence_level in ["High", "Medium", "Low"]:
            return confidence_level

        confidence = signal_result.get("prediction_confidence")

        if confidence is None:
            return "Unknown"

        if confidence >= 0.65:
            return "High"
        elif confidence >= 0.45:
            return "Medium"
        else:
            return "Low"

    def _build_state(self, validation_result: dict, analysis_result: dict, signal_result: dict) -> str:
        """
        Build discrete Q-learning state.
        """
        confidence = self._discretize_confidence(validation_result)
        volatility = analysis_result.get("volatility_level", "Unknown")
        model_signal = self._get_model_signal(signal_result)
        analyst_signal = analysis_result.get("analyst_signal", "UNKNOWN")
        model_confidence = self._get_model_confidence_level(signal_result)

        return f"{confidence}|{volatility}|{model_signal}|{analyst_signal}|{model_confidence}"

    def _rule_based_safety_action(
        self,
        validation_result: dict,
        analysis_result: dict,
        signal_result: dict
    ) -> tuple:
        """
        Hard safety rules.
        These rules protect the system even if Q-learning suggests a risky action.
        """
        confidence = self._discretize_confidence(validation_result)
        volatility = analysis_result.get("volatility_level", "Unknown")
        model_signal = self._get_model_signal(signal_result)
        analyst_signal = analysis_result.get("analyst_signal", "UNKNOWN")
        model_confidence = self._get_model_confidence_level(signal_result)

        reasons = []

        if confidence == "Low" or validation_result.get("next_action") == "BLOCK_ANALYSIS":
            reasons.append("Validation confidence is low or validation blocked analysis.")
            return "BLOCK_TRADE", reasons

        if model_confidence == "Low" and model_signal == "BUY_CANDIDATE":
            reasons.append("Model confidence is low while signal is BUY_CANDIDATE.")
            return "DOWNGRADE_TO_HOLD", reasons

        if model_confidence == "Low" and model_signal == "SELL_RISK":
            reasons.append("Model confidence is low, but SELL_RISK is already a cautious signal.")
            return "KEEP_SIGNAL", reasons

        if volatility == "High" and model_signal == "BUY_CANDIDATE":
            reasons.append("High volatility detected while model signal is BUY_CANDIDATE.")
            return "DOWNGRADE_TO_HOLD", reasons

        if model_signal == "SELL_RISK" or analyst_signal == "BEARISH_RISK":
            reasons.append("Model or analyst detected SELL_RISK / BEARISH_RISK.")
            return "KEEP_SIGNAL", reasons

        if confidence == "Medium" and model_signal == "BUY_CANDIDATE":
            reasons.append("Validation confidence is only Medium while model signal is BUY_CANDIDATE.")
            return "DOWNGRADE_TO_HOLD", reasons

        reasons.append("No hard risk rule was triggered.")
        return "KEEP_SIGNAL", reasons

    def _default_q_action(self, state: str) -> str:
        """
        Conservative default action when Q-table has no learning experience.
        """
        parts = state.split("|")

        confidence = parts[0] if len(parts) > 0 else "Unknown"
        volatility = parts[1] if len(parts) > 1 else "Unknown"
        model_signal = parts[2] if len(parts) > 2 else "HOLD"
        model_confidence = parts[4] if len(parts) > 4 else "Unknown"

        if confidence == "Low":
            return "BLOCK_TRADE"

        if model_confidence == "Low" and model_signal == "BUY_CANDIDATE":
            return "DOWNGRADE_TO_HOLD"

        if volatility == "High" and model_signal == "BUY_CANDIDATE":
            return "DOWNGRADE_TO_HOLD"

        if confidence == "Medium" and model_signal == "BUY_CANDIDATE":
            return "DOWNGRADE_TO_HOLD"

        return "KEEP_SIGNAL"

    def _choose_q_action(self, state: str) -> str:
        """
        Choose risk action using epsilon-greedy Q-learning policy.
        """
        self._init_state(state)

        values = self.q_table[state]

        if all(value == 0.0 for value in values.values()):
            return self._default_q_action(state)

        if random.random() < self.epsilon:
            return random.choice(self.ACTIONS)

        return max(values, key=values.get)

    def _normalize_action_for_signal(self, model_signal: str, action: str) -> str:
        """
        Normalize risk actions based on the current model signal.

        DOWNGRADE_TO_HOLD is mainly meaningful for BUY_CANDIDATE.
        If the signal is already SELL_RISK, keeping the cautious signal is clearer
        and more consistent than saying it is downgraded to HOLD.
        """
        if model_signal == "SELL_RISK" and action == "DOWNGRADE_TO_HOLD":
            return "KEEP_SIGNAL"

        if model_signal == "HOLD" and action == "DOWNGRADE_TO_HOLD":
            return "KEEP_SIGNAL"

        return action

    def _more_conservative_action(self, action_a: str, action_b: str) -> str:
        """
        Choose the more conservative action between rule-based and Q-learning actions.
        """
        if self.ACTION_PRIORITY[action_a] >= self.ACTION_PRIORITY[action_b]:
            return action_a
        return action_b

    def _apply_risk_action(self, model_signal: str, risk_action: str) -> str:
        """
        Convert risk action into final signal.
        """
        if risk_action == "BLOCK_TRADE":
            return "BLOCKED"

        if risk_action == "DOWNGRADE_TO_HOLD":
            if model_signal == "BUY_CANDIDATE":
                return "HOLD"
            return model_signal

        return model_signal

    def _risk_level(
        self,
        validation_result: dict,
        analysis_result: dict,
        signal_result: dict,
        final_signal: str,
        risk_action: str
    ) -> str:
        confidence = self._discretize_confidence(validation_result)
        volatility = analysis_result.get("volatility_level", "Unknown")
        model_confidence = self._get_model_confidence_level(signal_result)

        if risk_action == "BLOCK_TRADE" or final_signal == "BLOCKED":
            return "Critical"

        if final_signal == "SELL_RISK":
            return "High"

        if volatility == "High":
            return "High"

        if confidence == "Medium" or risk_action == "DOWNGRADE_TO_HOLD":
            return "Medium"

        if model_confidence == "Low":
            return "Medium"

        return "Low"

    def assess_risk(self, validation_result: dict, analysis_result: dict, signal_result: dict) -> dict:
        """
        Main Risk Agent method.
        """
        agent_goal = "Apply safety rules and Q-learning risk adjustment to the trading signal."

        state = self._build_state(validation_result, analysis_result, signal_result)
        self._init_state(state)

        model_signal = self._get_model_signal(signal_result)
        model_confidence = self._get_model_confidence_level(signal_result)

        rule_action, rule_reasons = self._rule_based_safety_action(
            validation_result=validation_result,
            analysis_result=analysis_result,
            signal_result=signal_result
        )

        raw_q_learning_action = self._choose_q_action(state)

        normalized_rule_action = self._normalize_action_for_signal(
            model_signal=model_signal,
            action=rule_action
        )

        q_learning_action = self._normalize_action_for_signal(
            model_signal=model_signal,
            action=raw_q_learning_action
        )

        final_risk_action = self._more_conservative_action(
            normalized_rule_action,
            q_learning_action
        )

        final_signal = self._apply_risk_action(
            model_signal=model_signal,
            risk_action=final_risk_action
        )

        risk_level = self._risk_level(
            validation_result=validation_result,
            analysis_result=analysis_result,
            signal_result=signal_result,
            final_signal=final_signal,
            risk_action=final_risk_action
        )

        reasoning_steps = [
            f"Built Q-learning state: {state}.",
            f"Model confidence level: {model_confidence}.",
            f"Rule-based safety layer suggested: {rule_action}.",
            f"Rule-based action after signal normalization: {normalized_rule_action}.",
            f"Raw Q-learning layer suggested: {raw_q_learning_action}.",
            f"Q-learning action after signal normalization: {q_learning_action}.",
            f"Final risk action selected: {final_risk_action}.",
            f"Final signal after risk adjustment: {final_signal}."
        ]

        explanation_for_llm = (
            f"The original signal was {model_signal} with {model_confidence.lower()} model confidence. "
            f"The rule-based safety layer suggested {rule_action}, and after normalization this became "
            f"{normalized_rule_action}. The raw Q-learning layer suggested {raw_q_learning_action}, and after "
            f"normalization this became {q_learning_action}. The final risk action is {final_risk_action}, "
            f"so the final signal becomes {final_signal}. The estimated risk level is {risk_level}."
        )

        return {
            "success": True,
            "agent_goal": agent_goal,
            "q_state": state,
            "original_signal": model_signal,
            "model_confidence_level": model_confidence,
            "rule_based_action": normalized_rule_action,
            "raw_rule_based_action": rule_action,
            "rule_reasons": rule_reasons,
            "raw_q_learning_action": raw_q_learning_action,
            "q_learning_action": q_learning_action,
            "risk_action": final_risk_action,
            "final_signal": final_signal,
            "risk_level": risk_level,
            "q_values_for_state": self.q_table.get(state, {}),
            "agent_decision": explanation_for_llm,
            "reasoning_steps": reasoning_steps,
            "explanation_for_llm": explanation_for_llm,
            "risk_for_next_agent": {
                "symbol": analysis_result.get("symbol"),
                "original_signal": model_signal,
                "model_confidence_level": model_confidence,
                "raw_q_learning_action": raw_q_learning_action,
                "q_learning_action": q_learning_action,
                "final_signal": final_signal,
                "risk_level": risk_level,
                "risk_action": final_risk_action,
                "q_state": state,
                "explanation_for_llm": explanation_for_llm
            },
            "summary": f"Risk Agent adjusted signal from {model_signal} to {final_signal} with risk level {risk_level}."
        }

    def calculate_reward(self, final_signal: str, future_return: float, volatility_level: str = "Unknown") -> float:
        """
        Convert future return into a Q-learning reward.
        """
        future_return = float(future_return)

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
                reward = 0.01

        else:
            reward = 0.0

        if volatility_level == "High":
            reward -= 0.005

        return reward

    def update_q_value(self, state: str, action: str, reward: float, next_state: str = None) -> dict:
        """
        Q-learning update:
        Q(s,a) = Q(s,a) + alpha * [reward + gamma * max Q(s') - Q(s,a)]
        """
        self._init_state(state)

        current_q = self.q_table[state][action]

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
            "reward": reward,
            "old_q": round(current_q, 4),
            "new_q": round(new_q, 4),
            "q_table_path": str(self.q_table_path),
            "summary": f"Updated Q-value for state={state}, action={action}: {current_q:.4f} → {new_q:.4f}."
        }

    def update_from_feedback(self, risk_result: dict, future_return: float) -> dict:
        """
        Update Q-table using future return feedback.
        """
        state = risk_result.get("q_state")
        action = risk_result.get("risk_action")
        final_signal = risk_result.get("final_signal")
        risk_level = risk_result.get("risk_level")

        if not state or not action:
            return {
                "success": False,
                "summary": "Cannot update Q-table because state or action is missing."
            }

        volatility_level = "High" if risk_level in ["High", "Critical"] else "Low"

        reward = self.calculate_reward(
            final_signal=final_signal,
            future_return=future_return,
            volatility_level=volatility_level
        )

        return self.update_q_value(
            state=state,
            action=action,
            reward=reward
        )