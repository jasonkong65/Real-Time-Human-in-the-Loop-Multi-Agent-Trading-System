from pathlib import Path
from typing import Dict, Any, Optional
import random
import joblib


class RiskAgent:
    """
    Hybrid Risk Agent:
    1. Rule-based safety layer
    2. Q-learning risk adjustment layer
    3. Final risk decision

    The Risk Agent does not execute real trades.
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

    # --------------------------------------------------
    # Q-table helpers
    # --------------------------------------------------
    def _load_q_table(self) -> dict:
        if self.q_table_path.exists():
            try:
                q_table = joblib.load(self.q_table_path)
                if isinstance(q_table, dict):
                    return q_table
            except Exception:
                return {}
        return {}

    def _save_q_table(self):
        self.q_table_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.q_table, self.q_table_path)

    def _init_state(self, state: str):
        if state not in self.q_table:
            self.q_table[state] = {
                action: 0.0
                for action in self.ACTIONS
            }

        # Make sure old Q-tables still contain all actions
        for action in self.ACTIONS:
            if action not in self.q_table[state]:
                self.q_table[state][action] = 0.0

    # --------------------------------------------------
    # Input extraction helpers
    # --------------------------------------------------
    def _discretize_confidence(self, validation_result: Dict[str, Any]) -> str:
        confidence = validation_result.get("confidence", "Unknown")

        if confidence in ["High", "Medium", "Low"]:
            return confidence

        score = validation_result.get("confidence_score", 0.5)

        try:
            score = float(score)
        except Exception:
            score = 0.5

        if score >= 0.8:
            return "High"
        elif score >= 0.5:
            return "Medium"
        else:
            return "Low"

    def _get_model_signal(self, signal_result: Dict[str, Any]) -> str:
        """
        Expected signals:
        BUY_CANDIDATE / HOLD / SELL_RISK
        """
        if not isinstance(signal_result, dict):
            return "HOLD"

        return (
            signal_result.get("model_signal")
            or signal_result.get("signal")
            or signal_result.get("final_signal")
            or signal_result.get("signal_for_next_agent", {}).get("signal")
            or "HOLD"
        )

    def _get_model_confidence_level(self, signal_result: Dict[str, Any]) -> str:
        if not isinstance(signal_result, dict):
            return "Unknown"

        confidence_level = (
            signal_result.get("confidence_level")
            or signal_result.get("signal_for_next_agent", {}).get("confidence_level")
        )

        if confidence_level in ["High", "Medium", "Low"]:
            return confidence_level

        confidence = (
            signal_result.get("prediction_confidence")
            or signal_result.get("signal_for_next_agent", {}).get("prediction_confidence")
        )

        if confidence is None:
            return "Unknown"

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

    def _get_analyst_signal(self, analysis_result: Dict[str, Any]) -> str:
        if not isinstance(analysis_result, dict):
            return "UNKNOWN"

        return (
            analysis_result.get("analyst_signal")
            or analysis_result.get("analysis_for_next_agent", {}).get("analyst_signal")
            or "UNKNOWN"
        )

    def _get_volatility_level(self, analysis_result: Dict[str, Any]) -> str:
        if not isinstance(analysis_result, dict):
            return "Unknown"

        return (
            analysis_result.get("volatility_level")
            or analysis_result.get("historical_volatility_level")
            or analysis_result.get("analysis_for_next_agent", {}).get("volatility_level")
            or "Unknown"
        )

    # --------------------------------------------------
    # State construction
    # --------------------------------------------------
    def _build_state(
        self,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any]
    ) -> str:
        """
        Build discrete Q-learning state.

        Example:
        High|Medium|SELL_RISK|NEUTRAL|Medium
        """
        confidence = self._discretize_confidence(validation_result)
        volatility = self._get_volatility_level(analysis_result)
        model_signal = self._get_model_signal(signal_result)
        analyst_signal = self._get_analyst_signal(analysis_result)
        model_confidence = self._get_model_confidence_level(signal_result)

        return f"{confidence}|{volatility}|{model_signal}|{analyst_signal}|{model_confidence}"

    # --------------------------------------------------
    # Rule-based safety layer
    # --------------------------------------------------
    def _rule_based_safety_action(
        self,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any]
    ) -> tuple:
        """
        Hard safety rules.
        These rules protect the system even if Q-learning suggests a risky action.
        """
        confidence = self._discretize_confidence(validation_result)
        volatility = self._get_volatility_level(analysis_result)
        model_signal = self._get_model_signal(signal_result)
        analyst_signal = self._get_analyst_signal(analysis_result)
        model_confidence = self._get_model_confidence_level(signal_result)

        reasons = []

        next_action = validation_result.get("next_action", "")

        # Rule 1: low-quality data should block action
        if confidence == "Low" and next_action == "BLOCK_ANALYSIS":
            reasons.append("Validation confidence is low and validation blocked analysis.")
            return "BLOCK_TRADE", reasons

        if next_action == "BLOCK_ANALYSIS":
            reasons.append("Validation Agent blocked downstream analysis.")
            return "BLOCK_TRADE", reasons

        # Rule 2: Low-confidence BUY should be downgraded
        if model_confidence == "Low" and model_signal == "BUY_CANDIDATE":
            reasons.append("Model confidence is low while signal is BUY_CANDIDATE.")
            return "DOWNGRADE_TO_HOLD", reasons

        # Rule 3: SELL_RISK is already cautious
        if model_signal == "SELL_RISK":
            reasons.append("Model detected SELL_RISK, which is already a cautious signal.")
            return "KEEP_SIGNAL", reasons

        # Rule 4: Analyst bearish risk should be preserved
        if analyst_signal == "BEARISH_RISK":
            reasons.append("Analyst Agent detected BEARISH_RISK.")
            return "KEEP_SIGNAL", reasons

        # Rule 5: High volatility + buy candidate is risky
        if volatility == "High" and model_signal == "BUY_CANDIDATE":
            reasons.append("High volatility detected while model signal is BUY_CANDIDATE.")
            return "DOWNGRADE_TO_HOLD", reasons

        # Rule 6: Medium validation + buy candidate should be conservative
        if confidence == "Medium" and model_signal == "BUY_CANDIDATE":
            reasons.append("Validation confidence is only Medium while model signal is BUY_CANDIDATE.")
            return "DOWNGRADE_TO_HOLD", reasons

        reasons.append("No hard risk rule was triggered.")
        return "KEEP_SIGNAL", reasons

    # --------------------------------------------------
    # Q-learning action
    # --------------------------------------------------
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

    # --------------------------------------------------
    # Action normalization and final signal
    # --------------------------------------------------
    def _normalize_action_for_signal(self, model_signal: str, action: str) -> str:
        """
        Normalize risk actions based on the current model signal.

        DOWNGRADE_TO_HOLD is mainly meaningful for BUY_CANDIDATE.
        If the signal is already SELL_RISK, keeping the cautious signal is clearer.
        """
        if model_signal in ["SELL_RISK", "HOLD"] and action == "DOWNGRADE_TO_HOLD":
            return "KEEP_SIGNAL"

        if action not in self.ACTIONS:
            return "KEEP_SIGNAL"

        return action

    def _more_conservative_action(self, action_a: str, action_b: str) -> str:
        if action_a not in self.ACTION_PRIORITY:
            action_a = "KEEP_SIGNAL"

        if action_b not in self.ACTION_PRIORITY:
            action_b = "KEEP_SIGNAL"

        if self.ACTION_PRIORITY[action_a] >= self.ACTION_PRIORITY[action_b]:
            return action_a
        return action_b

    def _apply_risk_action(self, model_signal: str, risk_action: str) -> str:
        if risk_action == "BLOCK_TRADE":
            return "BLOCKED"

        if risk_action == "DOWNGRADE_TO_HOLD":
            if model_signal == "BUY_CANDIDATE":
                return "HOLD"
            return model_signal

        return model_signal

    def _risk_level(
        self,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any],
        final_signal: str,
        risk_action: str
    ) -> str:
        confidence = self._discretize_confidence(validation_result)
        volatility = self._get_volatility_level(analysis_result)
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

    # --------------------------------------------------
    # Main method
    # --------------------------------------------------
    def assess_risk(
        self,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Main Risk Agent method.
        """
        agent_goal = "Apply safety rules and Q-learning risk adjustment to the trading signal."

        if not isinstance(validation_result, dict):
            validation_result = {}

        if not isinstance(analysis_result, dict):
            analysis_result = {}

        if not isinstance(signal_result, dict):
            signal_result = {}

        state = self._build_state(
            validation_result=validation_result,
            analysis_result=analysis_result,
            signal_result=signal_result
        )

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

        symbol = (
            analysis_result.get("symbol")
            or signal_result.get("signal_for_next_agent", {}).get("symbol")
            or validation_result.get("validation_for_next_agent", {}).get("symbol")
            or "UNKNOWN"
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
            f"The original model signal was {model_signal} with {model_confidence.lower()} model confidence. "
            f"The rule-based safety layer suggested {rule_action}, and after normalization this became "
            f"{normalized_rule_action}. The raw Q-learning layer suggested {raw_q_learning_action}, and after "
            f"normalization this became {q_learning_action}. The final risk action is {final_risk_action}, "
            f"so the final signal becomes {final_signal}. The estimated risk level is {risk_level}."
        )

        return {
            "success": True,
            "agent_goal": agent_goal,
            "symbol": symbol,
            "q_state": state,
            "original_signal": model_signal,
            "model_confidence_level": model_confidence,
            "raw_rule_based_action": rule_action,
            "rule_based_action": normalized_rule_action,
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
                "symbol": symbol,
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
            "summary": (
                f"Risk Agent adjusted signal from {model_signal} to {final_signal} "
                f"with risk level {risk_level}."
            )
        }

    # --------------------------------------------------
    # Compatibility aliases for app.py
    # --------------------------------------------------
    def apply_risk_control(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self.assess_risk(
            validation_result=validation_result or {},
            analysis_result=analysis_result or {},
            signal_result=signal_result or {}
        )

    def adjust_risk(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self.apply_risk_control(
            signal_result=signal_result,
            analysis_result=analysis_result,
            validation_result=validation_result
        )

    def evaluate_risk(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self.apply_risk_control(
            signal_result=signal_result,
            analysis_result=analysis_result,
            validation_result=validation_result
        )

    def control_risk(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self.apply_risk_control(
            signal_result=signal_result,
            analysis_result=analysis_result,
            validation_result=validation_result
        )

    def run(
        self,
        signal_result: Dict[str, Any],
        analysis_result: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        return self.apply_risk_control(
            signal_result=signal_result,
            analysis_result=analysis_result,
            validation_result=validation_result
        )

    # --------------------------------------------------
    # Reward calculation and Q-learning update
    # --------------------------------------------------
    def calculate_reward(
        self,
        final_signal: str,
        future_return: float,
        volatility_level: str = "Unknown"
    ) -> float:
        """
        Convert future return into a Q-learning reward.

        This is used by RewardAgent and manual feedback demo.
        """
        try:
            future_return = float(future_return)
        except Exception:
            future_return = 0.0

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

        if volatility_level in ["High", "Critical"]:
            reward -= 0.005

        return float(reward)

    def update_q_value(
        self,
        state: str,
        action: str,
        reward: float,
        next_state: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Q-learning update:
        Q(s,a) = Q(s,a) + alpha * [reward + gamma * max Q(s') - Q(s,a)]
        """
        if not state:
            return {
                "success": False,
                "summary": "Cannot update Q-table because state is missing."
            }

        if action not in self.ACTIONS:
            action = "KEEP_SIGNAL"

        try:
            reward = float(reward)
        except Exception:
            reward = 0.0

        self._init_state(state)

        current_q = self.q_table[state].get(action, 0.0)

        if next_state:
            self._init_state(next_state)
            max_next_q = max(self.q_table[next_state].values())
        else:
            max_next_q = 0.0

        new_q = current_q + self.alpha * (
            reward + self.gamma * max_next_q - current_q
        )

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
            )
        }

    def update_from_feedback(
        self,
        risk_result: Dict[str, Any],
        future_return: float
    ) -> Dict[str, Any]:
        """
        Update Q-table using future return feedback.
        """
        if not isinstance(risk_result, dict):
            return {
                "success": False,
                "summary": "Cannot update Q-table because risk_result is invalid."
            }

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