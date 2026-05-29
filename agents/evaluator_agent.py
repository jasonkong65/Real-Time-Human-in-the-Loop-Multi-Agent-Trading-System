from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import pandas as pd


class EvaluatorAgent:
    """
    Evaluator Agent

    Reviews paper-decision history, delayed rewards, and the DQN risk-control
    memory. It gives honest early-stage comments instead of overstating results.
    """

    def __init__(
        self,
        pending_path: str = "data/pending_rewards.csv",
        history_path: str = "data/reward_history.csv",
        q_table_path: str = "models/risk_q_table.pkl",
        dqn_replay_path: str = "data/risk_dqn_replay.csv",
        dqn_model_path: str = "models/risk_dqn_model.pkl",
    ):
        self.pending_path = Path(pending_path)
        self.history_path = Path(history_path)
        self.q_table_path = Path(q_table_path)
        self.dqn_replay_path = Path(dqn_replay_path)
        self.dqn_model_path = Path(dqn_model_path)

    def _load_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()

    def _load_q_table(self) -> Dict[str, Any]:
        if not self.q_table_path.exists() or self.q_table_path.stat().st_size == 0:
            return {}
        try:
            data = joblib.load(self.q_table_path)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _mean(self, df: pd.DataFrame, col: str) -> Optional[float]:
        if df.empty or col not in df.columns:
            return None
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            return None
        return float(values.mean())

    def _counts(self, df: pd.DataFrame, col: str) -> Dict[str, int]:
        if df.empty or col not in df.columns:
            return {}
        return df[col].fillna("Unknown").astype(str).value_counts().to_dict()

    def _top_cases(self, df: pd.DataFrame, best: bool = True) -> list:
        if df.empty or "reward" not in df.columns:
            return []
        temp = df.copy()
        temp["reward"] = pd.to_numeric(temp["reward"], errors="coerce")
        temp = temp.dropna(subset=["reward"])
        if temp.empty:
            return []
        temp = temp.sort_values("reward", ascending=not best).head(3)
        cols = [c for c in ["decision_id", "symbol", "entry_price", "latest_close", "future_return", "reward", "risk_action", "final_signal", "risk_level", "status", "updated_at_utc"] if c in temp.columns]
        return temp[cols].to_dict("records")

    def _readiness(self, completed: int, replay: int) -> tuple:
        if completed >= 30 and replay >= 20:
            return "Strong", 1.0
        if completed >= 10 or replay >= 10:
            return "Moderate", 0.7
        if completed > 0 or replay > 0:
            return "Early-stage", 0.4
        return "Not ready", 0.1

    def _performance_level(self, avg_reward: Optional[float], completed: int) -> str:
        if completed < 10:
            return "Early-stage only"
        if avg_reward is None:
            return "Unknown"
        if avg_reward > 0.01:
            return "Promising"
        if avg_reward > -0.005:
            return "Mixed"
        return "Needs improvement"

    def evaluate_history(self) -> Dict[str, Any]:
        pending = self._load_csv(self.pending_path)
        history = self._load_csv(self.history_path)
        replay = self._load_csv(self.dqn_replay_path)
        q_table = self._load_q_table()

        pending_count = int(len(pending[pending.get("status", pd.Series(dtype=str)) == "pending"])) if not pending.empty and "status" in pending.columns else int(len(pending))
        completed_count = int(len(history))
        replay_count = int(len(replay))
        avg_reward = self._mean(history, "reward")
        avg_return = self._mean(history, "future_return")
        readiness, readiness_score = self._readiness(completed_count, replay_count)
        performance = self._performance_level(avg_reward, completed_count)

        limitations = []
        if completed_count < 20:
            limitations.append("Only a small number of delayed outcomes have been completed, so performance evidence is early-stage.")
        if replay_count < 20:
            limitations.append("The DQN risk layer has limited replay memory and should be treated as advisory.")
        if avg_reward is not None and avg_reward < 0:
            limitations.append("Average reward is negative, so the system should keep improving its risk and model rules.")

        suggestions = [
            "Collect more paper decisions across different market conditions.",
            "Review worst cases and adjust entry-risk, RSI, and volatility handling.",
            "Use automatic Training Agent optimisation before relying on a signal model.",
        ]

        return {
            "success": True,
            "agent": "Evaluator Agent",
            "agent_goal": "Review past paper decisions and the DQN risk-control learning loop.",
            "pending_count": pending_count,
            "completed_reward_count": completed_count,
            "dqn_replay_count": replay_count,
            "average_reward": None if avg_reward is None else round(avg_reward, 6),
            "average_future_return": None if avg_return is None else round(avg_return, 6),
            "data_readiness_level": readiness,
            "data_readiness_score": readiness_score,
            "performance_level": performance,
            "signal_distribution": self._counts(history, "final_signal"),
            "risk_action_distribution": self._counts(history, "risk_action"),
            "q_table_summary": {
                "q_state_count": len(q_table),
                "compatibility_note": "This is a compatibility view of DQN state values, not the main learning method.",
            },
            "dqn_summary": {
                "replay_path": str(self.dqn_replay_path),
                "model_path": str(self.dqn_model_path),
                "model_exists": self.dqn_model_path.exists(),
                "replay_count": replay_count,
            },
            "best_cases": self._top_cases(history, best=True),
            "worst_cases": self._top_cases(history, best=False),
            "strengths": [
                "The system records paper decisions for delayed feedback.",
                "The Risk Agent now uses a DQN-style advisory layer with a hard safety filter.",
                "The Evaluator Agent can check whether the learning loop is ready or still early-stage.",
            ],
            "limitations": limitations,
            "suggestions": suggestions,
            "summary": f"Evaluator Agent found {pending_count} pending decisions, {completed_count} completed rewards, and {replay_count} DQN replay samples. Data readiness is {readiness}; performance level is {performance}.",
        }

    def run(self) -> Dict[str, Any]:
        return self.evaluate_history()
