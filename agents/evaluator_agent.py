from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import joblib


class EvaluatorAgent:
    """
    Evaluator Agent:
    Evaluates historical paper recommendations, delayed rewards,
    pending decisions, and Q-learning risk-control status.

    Important:
    - data_readiness_level measures whether the evaluation pipeline has enough records.
    - performance_level measures whether historical rewards look good or poor.
    - This agent does not make trading decisions.
    """

    def __init__(
        self,
        pending_path: str = "data/pending_rewards.csv",
        history_path: str = "data/reward_history.csv",
        q_table_path: str = "models/risk_q_table.pkl"
    ):
        self.pending_path = Path(pending_path)
        self.history_path = Path(history_path)
        self.q_table_path = Path(q_table_path)

    # --------------------------------------------------
    # Safe loading helpers
    # --------------------------------------------------
    def _safe_load_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()

        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def _safe_load_q_table(self) -> Dict[str, Any]:
        if not self.q_table_path.exists() or self.q_table_path.stat().st_size == 0:
            return {}

        try:
            q_table = joblib.load(self.q_table_path)
            if isinstance(q_table, dict):
                return q_table
            return {}
        except Exception:
            return {}

    def _safe_mean(self, df: pd.DataFrame, column: str) -> Optional[float]:
        if df.empty or column not in df.columns:
            return None

        values = pd.to_numeric(df[column], errors="coerce").dropna()

        if values.empty:
            return None

        return float(values.mean())

    def _value_counts(self, df: pd.DataFrame, column: str) -> Dict[str, int]:
        if df.empty or column not in df.columns:
            return {}

        return df[column].fillna("Unknown").astype(str).value_counts().to_dict()

    def _round_or_none(self, value: Optional[float], digits: int = 6):
        if value is None:
            return None
        return round(float(value), digits)

    # --------------------------------------------------
    # Readiness and performance evaluation
    # --------------------------------------------------
    def _calculate_data_readiness(
        self,
        pending_count: int,
        completed_count: int,
        q_state_count: int,
        best_cases: list,
        worst_cases: list
    ):
        score = 0.0

        if pending_count > 0:
            score += 0.20

        if completed_count > 0:
            score += 0.30

        if completed_count >= 5:
            score += 0.15

        if q_state_count > 0:
            score += 0.20

        if best_cases or worst_cases:
            score += 0.15

        score = round(min(score, 1.0), 3)

        if score >= 0.75:
            level = "Strong"
        elif score >= 0.45:
            level = "Moderate"
        else:
            level = "Early-stage"

        return score, level

    def _calculate_performance_level(
        self,
        completed_count: int,
        average_reward: Optional[float],
        average_future_return: Optional[float]
    ):
        if completed_count == 0 or average_reward is None:
            return {
                "performance_level": "No completed evidence yet",
                "performance_interpretation": (
                    "No completed delayed reward records are available yet, "
                    "so the system cannot evaluate recommendation performance."
                )
            }

        if completed_count < 5:
            evidence_note = (
                "The sample size is still small, so this performance result should be treated cautiously."
            )
        else:
            evidence_note = (
                "There are enough completed records for an early-stage performance check, "
                "but more data is still needed before making strong claims."
            )

        if average_reward > 0.005:
            level = "Positive early performance"
            interpretation = (
                "Average reward is positive, suggesting that recent paper decisions have performed reasonably well. "
                + evidence_note
            )

        elif average_reward < -0.005:
            level = "Needs improvement"
            interpretation = (
                "Average reward is negative, suggesting that some historical paper decisions performed poorly. "
                "This does not mean the system failed, but it shows the value of the Evaluator Agent: "
                "the system should review weak cases and adjust risk thresholds or model features. "
                + evidence_note
            )

        else:
            level = "Neutral / mixed performance"
            interpretation = (
                "Average reward is close to zero, suggesting that the system has not yet shown a clear positive "
                "or negative historical pattern. "
                + evidence_note
            )

        if average_future_return is not None and average_future_return < 0:
            interpretation += (
                " Average future return is also negative, which supports a cautious interpretation."
            )

        return {
            "performance_level": level,
            "performance_interpretation": interpretation
        }

    # --------------------------------------------------
    # Main evaluation logic
    # --------------------------------------------------
    def evaluate_history(self) -> Dict[str, Any]:
        pending_df = self._safe_load_csv(self.pending_path)
        history_df = self._safe_load_csv(self.history_path)
        q_table = self._safe_load_q_table()

        pending_count = len(pending_df)
        completed_count = len(history_df)

        average_reward = self._safe_mean(history_df, "reward")
        average_future_return = self._safe_mean(history_df, "future_return")

        pending_signal_distribution = self._value_counts(pending_df, "final_signal")
        completed_signal_distribution = self._value_counts(history_df, "final_signal")
        risk_action_distribution = self._value_counts(history_df, "risk_action")
        risk_level_distribution = self._value_counts(history_df, "risk_level")

        q_state_count = len(q_table)
        q_action_count = 0
        non_zero_q_values = 0

        for state_values in q_table.values():
            if isinstance(state_values, dict):
                q_action_count += len(state_values)
                non_zero_q_values += sum(
                    1 for value in state_values.values()
                    if isinstance(value, (int, float)) and value != 0
                )

        best_cases = []
        worst_cases = []

        if not history_df.empty and "reward" in history_df.columns:
            temp_df = history_df.copy()
            temp_df["reward_numeric"] = pd.to_numeric(
                temp_df["reward"],
                errors="coerce"
            )

            valid_reward_df = temp_df.dropna(subset=["reward_numeric"])

            if not valid_reward_df.empty:
                best_df = valid_reward_df.sort_values(
                    by="reward_numeric",
                    ascending=False
                ).head(3)

                worst_df = valid_reward_df.sort_values(
                    by="reward_numeric",
                    ascending=True
                ).head(3)

                best_cases = best_df.drop(columns=["reward_numeric"]).to_dict("records")
                worst_cases = worst_df.drop(columns=["reward_numeric"]).to_dict("records")

        data_readiness_score, data_readiness_level = self._calculate_data_readiness(
            pending_count=pending_count,
            completed_count=completed_count,
            q_state_count=q_state_count,
            best_cases=best_cases,
            worst_cases=worst_cases
        )

        performance = self._calculate_performance_level(
            completed_count=completed_count,
            average_reward=average_reward,
            average_future_return=average_future_return
        )

        strengths = []
        limitations = []
        suggestions = []

        if pending_count > 0:
            strengths.append(
                "Reward Agent is recording pending paper decisions for delayed feedback."
            )
        else:
            limitations.append(
                "No pending paper decisions are currently recorded."
            )
            suggestions.append(
                "Run the single-stock pipeline to record pending reward decisions."
            )

        if completed_count > 0:
            strengths.append(
                "Completed delayed reward updates are available for evaluation."
            )
        else:
            limitations.append(
                "No completed delayed reward updates are available yet."
            )
            suggestions.append(
                "Run the system again after a later market close so pending decisions can be updated."
            )

        if q_state_count > 0:
            strengths.append(
                "Q-learning Risk Agent has stored Q-table states."
            )
        else:
            limitations.append(
                "Q-learning table is empty or unavailable."
            )
            suggestions.append(
                "Use delayed reward updates to populate the Q-table."
            )

        if average_reward is not None:
            if average_reward < -0.005:
                limitations.append(
                    "Average reward is negative, suggesting some historical recommendations performed poorly."
                )
                suggestions.append(
                    "Review worst cases and consider adjusting risk thresholds, RSI overbought handling, or model features."
                )
            elif average_reward > 0.005:
                strengths.append(
                    "Average reward is positive in the current completed reward history."
                )
            else:
                limitations.append(
                    "Average reward is close to zero, so performance evidence is mixed."
                )
                suggestions.append(
                    "Collect more completed reward records before making strong claims."
                )

        if completed_count < 10:
            limitations.append(
                "The completed reward sample is still small, so evaluation evidence is early-stage."
            )
            suggestions.append(
                "Collect more paper decisions across different market conditions."
            )

        suggestions.append(
            "Use the Evaluator Agent output to guide future Training Agent parameter optimization."
        )

        summary = (
            f"Evaluator Agent found {pending_count} pending decisions, "
            f"{completed_count} completed reward updates, "
            f"{q_state_count} Q-table states. "
            f"Data readiness is {data_readiness_level}; "
            f"performance level is {performance['performance_level']}."
        )

        return {
            "success": True,
            "agent_goal": (
                "Evaluate historical recommendations, delayed reward feedback, "
                "and Q-learning risk-control status."
            ),

            # New clearer labels
            "data_readiness_score": data_readiness_score,
            "data_readiness_level": data_readiness_level,
            "performance_level": performance["performance_level"],
            "performance_interpretation": performance["performance_interpretation"],

            # Backward-compatible fields
            "evaluation_score": data_readiness_score,
            "evaluation_level": data_readiness_level,

            "pending_count": pending_count,
            "completed_reward_count": completed_count,
            "average_reward": self._round_or_none(average_reward),
            "average_future_return": self._round_or_none(average_future_return),
            "pending_signal_distribution": pending_signal_distribution,
            "completed_signal_distribution": completed_signal_distribution,
            "risk_action_distribution": risk_action_distribution,
            "risk_level_distribution": risk_level_distribution,
            "q_table_summary": {
                "q_table_path": str(self.q_table_path),
                "q_state_count": q_state_count,
                "q_action_count": q_action_count,
                "non_zero_q_values": non_zero_q_values
            },
            "best_cases": best_cases,
            "worst_cases": worst_cases,
            "strengths": strengths,
            "limitations": limitations,
            "suggestions": suggestions,
            "paths": {
                "pending_path": str(self.pending_path),
                "history_path": str(self.history_path),
                "q_table_path": str(self.q_table_path)
            },
            "summary": summary
        }

    def run(self) -> Dict[str, Any]:
        return self.evaluate_history()