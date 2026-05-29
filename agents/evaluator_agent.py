from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import sqlite3

import joblib
import pandas as pd


class EvaluatorAgent:
    """
    Evaluator Agent

    Role:
    - Reads paper-decision history from SQLite first, then falls back to CSV.
    - Reviews delayed reward results, signal quality, strategy performance, and DQN replay readiness.
    - Provides honest early-stage evaluation instead of overstating performance.

    Main outputs:
    - win_rate
    - directional_win_rate
    - reward_by_strategy_action
    - reward_by_signal_type
    - best_cases / worst_cases
    - DQN replay readiness
    """

    def __init__(
        self,
        db_path: str = "data/trading_system.db",
        pending_path: str = "data/pending_rewards.csv",
        history_path: str = "data/reward_history.csv",
        q_table_path: str = "models/risk_q_table.pkl",
        dqn_replay_path: str = "data/risk_dqn_replay.csv",
        dqn_model_path: str = "models/risk_dqn_model.pt",
        legacy_dqn_model_path: str = "models/risk_dqn_model.pkl",
        neutral_return_band: float = 0.005,
    ):
        self.db_path = Path(db_path)
        self.pending_path = Path(pending_path)
        self.history_path = Path(history_path)
        self.q_table_path = Path(q_table_path)
        self.dqn_replay_path = Path(dqn_replay_path)
        self.dqn_model_path = Path(dqn_model_path)
        self.legacy_dqn_model_path = Path(legacy_dqn_model_path)
        self.neutral_return_band = neutral_return_band

    # --------------------------------------------------
    # Generic helpers
    # --------------------------------------------------
    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None:
                return default
            value = float(value)
            if pd.isna(value):
                return default
            return value
        except Exception:
            return default

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            if value is None:
                return default
            value = int(value)
            return value
        except Exception:
            return default

    def _json_loads(self, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(str(value))
        except Exception:
            return None

    def _load_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
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

    def _normalise_time_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        for col in ["created_at", "created_at_utc", "updated_at", "updated_at_utc", "due_at_utc"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
        return df

    # --------------------------------------------------
    # SQLite helpers
    # --------------------------------------------------
    def _sqlite_available(self) -> bool:
        return self.db_path.exists() and self.db_path.stat().st_size > 0

    def _table_exists(self, conn: sqlite3.Connection, table_name: str) -> bool:
        try:
            query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
            row = conn.execute(query, (table_name,)).fetchone()
            return row is not None
        except Exception:
            return False

    def _read_table(self, conn: sqlite3.Connection, table_name: str) -> pd.DataFrame:
        if not self._table_exists(conn, table_name):
            return pd.DataFrame()
        try:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        except Exception:
            return pd.DataFrame()

    def _load_sqlite_data(self) -> Dict[str, pd.DataFrame]:
        if not self._sqlite_available():
            return {
                "paper_decisions": pd.DataFrame(),
                "reward_updates": pd.DataFrame(),
                "pipeline_runs": pd.DataFrame(),
                "agent_outputs": pd.DataFrame(),
                "training_runs": pd.DataFrame(),
                "screener_runs": pd.DataFrame(),
                "llm_reports": pd.DataFrame(),
                "risk_dqn_replay": pd.DataFrame(),
            }

        try:
            with sqlite3.connect(self.db_path) as conn:
                data = {
                    "paper_decisions": self._read_table(conn, "paper_decisions"),
                    "reward_updates": self._read_table(conn, "reward_updates"),
                    "pipeline_runs": self._read_table(conn, "pipeline_runs"),
                    "agent_outputs": self._read_table(conn, "agent_outputs"),
                    "training_runs": self._read_table(conn, "training_runs"),
                    "screener_runs": self._read_table(conn, "screener_runs"),
                    "llm_reports": self._read_table(conn, "llm_reports"),
                    "risk_dqn_replay": self._read_table(conn, "risk_dqn_replay"),
                }
            return {k: self._normalise_time_columns(v) for k, v in data.items()}
        except Exception:
            return {
                "paper_decisions": pd.DataFrame(),
                "reward_updates": pd.DataFrame(),
                "pipeline_runs": pd.DataFrame(),
                "agent_outputs": pd.DataFrame(),
                "training_runs": pd.DataFrame(),
                "screener_runs": pd.DataFrame(),
                "llm_reports": pd.DataFrame(),
                "risk_dqn_replay": pd.DataFrame(),
            }

    # --------------------------------------------------
    # Data preparation
    # --------------------------------------------------
    def _prepare_history_from_sqlite(self, sqlite_data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        decisions = sqlite_data.get("paper_decisions", pd.DataFrame()).copy()
        updates = sqlite_data.get("reward_updates", pd.DataFrame()).copy()
        runs = sqlite_data.get("pipeline_runs", pd.DataFrame()).copy()

        if decisions.empty and updates.empty:
            return pd.DataFrame(), pd.DataFrame()

        # Completed rewards = rows with numeric reward or completed-like status.
        if not updates.empty:
            updates["reward"] = pd.to_numeric(updates.get("reward"), errors="coerce") if "reward" in updates.columns else None
            updates["future_return"] = pd.to_numeric(updates.get("future_return"), errors="coerce") if "future_return" in updates.columns else None
            if "status" in updates.columns:
                completed_mask = updates["reward"].notna() | updates["status"].fillna("").astype(str).str.upper().str.contains("COMPLETED|UPDATED|DONE")
            else:
                completed_mask = updates["reward"].notna()
            completed = updates[completed_mask].copy()
        else:
            completed = pd.DataFrame()

        if not completed.empty and not decisions.empty and "decision_id" in completed.columns and "decision_id" in decisions.columns:
            suffix_cols = [c for c in decisions.columns if c not in completed.columns or c == "decision_id"]
            completed = completed.merge(decisions[suffix_cols], on="decision_id", how="left")

        if not completed.empty and not runs.empty and "run_id" in completed.columns and "run_id" in runs.columns:
            run_cols = [c for c in ["run_id", "strategy_action", "strategy_level", "analyst_signal", "model_signal", "final_signal", "risk_level"] if c in runs.columns]
            if run_cols:
                completed = completed.merge(runs[run_cols], on="run_id", how="left", suffixes=("", "_run"))
                for col in ["strategy_action", "strategy_level", "analyst_signal", "model_signal", "final_signal", "risk_level"]:
                    run_col = f"{col}_run"
                    if run_col in completed.columns:
                        if col not in completed.columns:
                            completed[col] = completed[run_col]
                        else:
                            completed[col] = completed[col].fillna(completed[run_col])

        # Pending/open decisions from paper_decisions.
        if not decisions.empty:
            status_col = decisions["status"].fillna("").astype(str).str.upper() if "status" in decisions.columns else pd.Series([""] * len(decisions))
            pending_mask = ~status_col.str.contains("COMPLETED|CLOSED|DONE|CANCELLED|EXPIRED")
            pending = decisions[pending_mask].copy()
        else:
            pending = pd.DataFrame()

        return self._normalise_time_columns(pending), self._normalise_time_columns(completed)

    def _prepare_history_from_csv(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        pending = self._load_csv(self.pending_path)
        history = self._load_csv(self.history_path)
        return self._normalise_time_columns(pending), self._normalise_time_columns(history)

    def _load_evaluation_frames(self) -> Dict[str, Any]:
        sqlite_data = self._load_sqlite_data()
        pending_sql, completed_sql = self._prepare_history_from_sqlite(sqlite_data)

        csv_used = False
        if completed_sql.empty and pending_sql.empty:
            pending, completed = self._prepare_history_from_csv()
            csv_used = True
        else:
            pending, completed = pending_sql, completed_sql

        replay = sqlite_data.get("risk_dqn_replay", pd.DataFrame()).copy()
        if replay.empty:
            replay = self._load_csv(self.dqn_replay_path)
        replay = self._normalise_time_columns(replay)

        return {
            "source": "sqlite" if not csv_used else "csv_fallback",
            "sqlite_data": sqlite_data,
            "pending": pending,
            "completed": completed,
            "replay": replay,
        }

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    def _reward_win_rate(self, df: pd.DataFrame) -> Optional[float]:
        if df.empty or "reward" not in df.columns:
            return None
        rewards = pd.to_numeric(df["reward"], errors="coerce").dropna()
        if rewards.empty:
            return None
        return float((rewards > 0).mean())

    def _directional_success(self, row: pd.Series) -> Optional[bool]:
        final_signal = str(row.get("final_signal", row.get("signal", ""))).upper()
        future_return = self._safe_float(row.get("future_return"), None)
        if future_return is None:
            return None

        if final_signal == "BUY_CANDIDATE":
            return future_return > 0
        if final_signal == "SELL_RISK":
            return future_return < 0
        if final_signal in ["HOLD", "BLOCKED"]:
            return abs(future_return) <= self.neutral_return_band
        # For display/enhanced signals such as BUY_WATCHLIST_OVERBOUGHT, treat as cautious watchlist.
        if "BUY" in final_signal and "WATCHLIST" in final_signal:
            return future_return >= -self.neutral_return_band
        if "RISK" in final_signal:
            return future_return <= self.neutral_return_band
        return None

    def _directional_win_rate(self, df: pd.DataFrame) -> Optional[float]:
        if df.empty or "future_return" not in df.columns:
            return None
        results = df.apply(self._directional_success, axis=1).dropna()
        if results.empty:
            return None
        return float(results.mean())

    def _group_metrics(self, df: pd.DataFrame, group_col: str) -> List[Dict[str, Any]]:
        if df.empty or group_col not in df.columns:
            return []

        temp = df.copy()
        temp[group_col] = temp[group_col].fillna("Unknown").astype(str)
        if "reward" in temp.columns:
            temp["reward"] = pd.to_numeric(temp["reward"], errors="coerce")
        if "future_return" in temp.columns:
            temp["future_return"] = pd.to_numeric(temp["future_return"], errors="coerce")

        rows: List[Dict[str, Any]] = []
        for name, group in temp.groupby(group_col):
            avg_reward = self._mean(group, "reward")
            avg_return = self._mean(group, "future_return")
            win_rate = self._reward_win_rate(group)
            directional_rate = self._directional_win_rate(group)
            rows.append({
                group_col: name,
                "count": int(len(group)),
                "avg_reward": None if avg_reward is None else round(avg_reward, 6),
                "avg_future_return": None if avg_return is None else round(avg_return, 6),
                "reward_win_rate": None if win_rate is None else round(win_rate, 4),
                "directional_win_rate": None if directional_rate is None else round(directional_rate, 4),
            })

        rows = sorted(rows, key=lambda x: (x["count"], x.get("avg_reward") or -999), reverse=True)
        return rows

    def _top_cases(self, df: pd.DataFrame, best: bool = True, n: int = 5) -> List[Dict[str, Any]]:
        if df.empty or "reward" not in df.columns:
            return []
        temp = df.copy()
        temp["reward"] = pd.to_numeric(temp["reward"], errors="coerce")
        temp = temp.dropna(subset=["reward"])
        if temp.empty:
            return []
        temp = temp.sort_values("reward", ascending=not best).head(n)
        preferred_cols = [
            "decision_id", "symbol", "reward_horizon", "horizon_label",
            "entry_price", "latest_close", "future_return", "reward",
            "strategy_action", "final_signal", "risk_action", "risk_level",
            "paper_status", "status", "updated_at_utc", "updated_at", "created_at_utc", "created_at",
        ]
        cols = [c for c in preferred_cols if c in temp.columns]
        if not cols:
            cols = list(temp.columns[:10])
        records = temp[cols].copy()
        for col in records.columns:
            if pd.api.types.is_datetime64_any_dtype(records[col]):
                records[col] = records[col].astype(str)
        return records.to_dict("records")

    def _readiness(self, completed: int, replay: int) -> Tuple[str, float]:
        if completed >= 50 and replay >= 100:
            return "Strong", 1.0
        if completed >= 30 and replay >= 50:
            return "Good", 0.85
        if completed >= 10 or replay >= 20:
            return "Moderate", 0.65
        if completed > 0 or replay > 0:
            return "Early-stage", 0.4
        return "Not ready", 0.1

    def _performance_level(self, avg_reward: Optional[float], win_rate: Optional[float], completed: int) -> str:
        if completed < 10:
            return "Early-stage only"
        if avg_reward is None and win_rate is None:
            return "Unknown"
        if (avg_reward is not None and avg_reward > 0.01) and (win_rate is not None and win_rate >= 0.55):
            return "Promising"
        if (avg_reward is not None and avg_reward > -0.005) or (win_rate is not None and win_rate >= 0.45):
            return "Mixed"
        return "Needs improvement"

    def _make_limitations(self, completed: int, replay_count: int, avg_reward: Optional[float], source: str) -> List[str]:
        limitations: List[str] = []
        if completed < 20:
            limitations.append("Only a small number of delayed outcomes have been completed, so the evaluation is early-stage.")
        if replay_count < 100:
            limitations.append("The strict DQN risk layer has limited replay memory; DQN recommendations should remain advisory.")
        if avg_reward is not None and avg_reward < 0:
            limitations.append("Average reward is negative, so the system should keep improving signal, entry-risk, and strategy rules.")
        if source == "csv_fallback":
            limitations.append("SQLite records were unavailable or empty, so the evaluator used CSV fallback data.")
        return limitations

    def _make_suggestions(self, completed: int, replay_count: int) -> List[str]:
        suggestions = [
            "Collect more paper decisions across different market conditions.",
            "Review the worst cases and adjust entry-risk, RSI, volatility, and strategy rules.",
            "Use the automatic Training Agent model selection before relying on new signal outputs.",
            "Compare reward by strategy action and signal type before changing model thresholds.",
        ]
        if completed < 20:
            suggestions.append("Wait for more completed reward horizons before drawing strong performance conclusions.")
        if replay_count < 100:
            suggestions.append("Allow the DQN replay memory to collect at least 100 samples before treating its Q-values as reliable.")
        return suggestions

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def evaluate_history(self) -> Dict[str, Any]:
        frames = self._load_evaluation_frames()
        source = frames["source"]
        pending = frames["pending"]
        completed = frames["completed"]
        replay = frames["replay"]
        sqlite_data = frames["sqlite_data"]
        q_table = self._load_q_table()

        pending_count = int(len(pending))
        completed_count = int(len(completed))
        replay_count = int(len(replay))

        avg_reward = self._mean(completed, "reward")
        avg_return = self._mean(completed, "future_return")
        reward_win_rate = self._reward_win_rate(completed)
        directional_win_rate = self._directional_win_rate(completed)

        readiness, readiness_score = self._readiness(completed_count, replay_count)
        performance = self._performance_level(avg_reward, reward_win_rate, completed_count)

        reward_by_strategy = self._group_metrics(completed, "strategy_action")
        reward_by_signal = self._group_metrics(completed, "final_signal")
        reward_by_risk_action = self._group_metrics(completed, "risk_action")
        reward_by_symbol = self._group_metrics(completed, "symbol")
        reward_by_horizon = self._group_metrics(completed, "horizon_label") or self._group_metrics(completed, "reward_horizon")

        paper_status_distribution = self._counts(pending, "paper_status") or self._counts(pending, "status")
        pending_status_distribution = self._counts(pending, "status")

        dqn_model_exists = self.dqn_model_path.exists() or self.legacy_dqn_model_path.exists()
        dqn_model_path = str(self.dqn_model_path if self.dqn_model_path.exists() else self.legacy_dqn_model_path)

        limitations = self._make_limitations(completed_count, replay_count, avg_reward, source)
        suggestions = self._make_suggestions(completed_count, replay_count)

        db_summary = {
            "db_path": str(self.db_path),
            "db_exists": self._sqlite_available(),
            "data_source_used": source,
            "pipeline_runs": int(len(sqlite_data.get("pipeline_runs", pd.DataFrame()))),
            "agent_outputs": int(len(sqlite_data.get("agent_outputs", pd.DataFrame()))),
            "paper_decisions": int(len(sqlite_data.get("paper_decisions", pd.DataFrame()))),
            "reward_updates": int(len(sqlite_data.get("reward_updates", pd.DataFrame()))),
            "training_runs": int(len(sqlite_data.get("training_runs", pd.DataFrame()))),
            "screener_runs": int(len(sqlite_data.get("screener_runs", pd.DataFrame()))),
            "llm_reports": int(len(sqlite_data.get("llm_reports", pd.DataFrame()))),
            "risk_dqn_replay": int(len(sqlite_data.get("risk_dqn_replay", pd.DataFrame()))),
        }

        summary = (
            f"Evaluator Agent found {pending_count} open paper decisions, "
            f"{completed_count} completed reward records, and {replay_count} DQN replay samples. "
            f"Data readiness is {readiness}; performance level is {performance}."
        )

        if reward_win_rate is not None:
            summary += f" Reward win rate is {reward_win_rate:.1%}."

        return {
            "success": True,
            "agent": "Evaluator Agent",
            "agent_goal": "Evaluate paper-decision outcomes, reward quality, strategy performance, and DQN readiness.",
            "data_source_used": source,
            "pending_count": pending_count,
            "completed_reward_count": completed_count,
            "dqn_replay_count": replay_count,
            "average_reward": None if avg_reward is None else round(avg_reward, 6),
            "average_future_return": None if avg_return is None else round(avg_return, 6),
            "reward_win_rate": None if reward_win_rate is None else round(reward_win_rate, 4),
            "directional_win_rate": None if directional_win_rate is None else round(directional_win_rate, 4),
            "data_readiness_level": readiness,
            "data_readiness_score": readiness_score,
            "performance_level": performance,
            "paper_status_distribution": paper_status_distribution,
            "pending_status_distribution": pending_status_distribution,
            "signal_distribution": self._counts(completed, "final_signal"),
            "strategy_action_distribution": self._counts(completed, "strategy_action"),
            "risk_action_distribution": self._counts(completed, "risk_action"),
            "reward_by_strategy_action": reward_by_strategy,
            "reward_by_signal_type": reward_by_signal,
            "reward_by_risk_action": reward_by_risk_action,
            "reward_by_symbol": reward_by_symbol,
            "reward_by_horizon": reward_by_horizon,
            "evaluation_tables": {
                "reward_by_strategy_action": reward_by_strategy,
                "reward_by_signal_type": reward_by_signal,
                "reward_by_risk_action": reward_by_risk_action,
                "reward_by_symbol": reward_by_symbol,
                "reward_by_horizon": reward_by_horizon,
            },
            "database_summary": db_summary,
            "q_table_summary": {
                "q_state_count": len(q_table),
                "compatibility_note": "This is only a legacy compatibility view. The current Risk Agent should use strict DQN replay/model files.",
            },
            "dqn_summary": {
                "replay_path": str(self.dqn_replay_path),
                "model_path": dqn_model_path,
                "model_exists": dqn_model_exists,
                "replay_count": replay_count,
                "minimum_replay_target": 100,
                "ready_for_training": replay_count >= 100,
            },
            "best_cases": self._top_cases(completed, best=True),
            "worst_cases": self._top_cases(completed, best=False),
            "strengths": [
                "The system records paper decisions for delayed feedback.",
                "SQLite can be used as the primary evaluation source, with CSV kept as fallback.",
                "The evaluator now compares reward by strategy action and signal type instead of only reporting a single average.",
                "The DQN layer is evaluated with replay-memory readiness, not just model existence.",
            ],
            "limitations": limitations,
            "suggestions": suggestions,
            "summary": summary,
        }

    def run(self) -> Dict[str, Any]:
        return self.evaluate_history()
