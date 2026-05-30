from __future__ import annotations

import json

import math

import random

import sqlite3

from datetime import datetime, timezone

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

import joblib

import pandas as pd

import torch

import torch.nn as nn

import torch.optim as optim

from .dqn import DQNNetwork


class RiskReplayMixin:


    def _load_q_table_compat(self) -> Dict[str, Dict[str, float]]:
        if self.q_table_path.exists() and self.q_table_path.stat().st_size > 0:
            try:
                data = joblib.load(self.q_table_path)
                if isinstance(data, dict):
                    return data
            except Exception:
                return {}
        return {}


    def _save_q_table_compat(self) -> None:
        try:
            joblib.dump(self.q_table, self.q_table_path)
        except Exception:
            pass


    def _init_replay_db(self) -> None:
        """Create the SQLite replay-memory table used as the primary DQN memory."""
        try:
            with sqlite3.connect(self.replay_db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS risk_dqn_replay (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at_utc TEXT,
                        state_text TEXT,
                        state_vector_json TEXT,
                        action TEXT,
                        action_index INTEGER,
                        reward REAL,
                        next_state_text TEXT,
                        next_state_vector_json TEXT,
                        done INTEGER,
                        source TEXT DEFAULT 'risk_agent'
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_dqn_replay_time ON risk_dqn_replay(created_at_utc)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_risk_dqn_replay_action ON risk_dqn_replay(action)")
        except Exception:
            # CSV replay remains as a compatibility fallback.
            pass


    def _read_replay_from_sqlite(self) -> pd.DataFrame:
        try:
            if not self.replay_db_path.exists() or self.replay_db_path.stat().st_size == 0:
                return self._empty_replay_df()
            with sqlite3.connect(self.replay_db_path) as conn:
                table = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='risk_dqn_replay'"
                ).fetchone()
                if table is None:
                    return self._empty_replay_df()
                df = pd.read_sql_query(
                    """
                    SELECT created_at_utc, state_text, state_vector_json,
                           action, action_index, reward, next_state_text,
                           next_state_vector_json, done
                    FROM risk_dqn_replay
                    ORDER BY id ASC
                    """,
                    conn,
                )
            for col in self._empty_replay_df().columns:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception:
            return self._empty_replay_df()


    def _append_replay_to_sqlite(self, row: Dict[str, Any]) -> bool:
        try:
            self._init_replay_db()
            with sqlite3.connect(self.replay_db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO risk_dqn_replay (
                        created_at_utc, state_text, state_vector_json,
                        action, action_index, reward, next_state_text,
                        next_state_vector_json, done, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row.get("created_at_utc"),
                        row.get("state_text"),
                        row.get("state_vector_json"),
                        row.get("action"),
                        int(row.get("action_index", 0)),
                        float(row.get("reward", 0.0)),
                        row.get("next_state_text"),
                        row.get("next_state_vector_json"),
                        1 if row.get("done") else 0,
                        "risk_agent",
                    ),
                )
            return True
        except Exception:
            return False


    def _empty_replay_df(self) -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "created_at_utc",
                "state_text",
                "state_vector_json",
                "action",
                "action_index",
                "reward",
                "next_state_text",
                "next_state_vector_json",
                "done",
            ]
        )


    def _read_replay(self) -> pd.DataFrame:
        """
        Read replay memory from SQLite first, then fall back to CSV.
        CSV is kept as a mirror for compatibility with old dashboards.
        """
        sqlite_df = self._read_replay_from_sqlite()
        if not sqlite_df.empty:
            return sqlite_df

        if not self.replay_path.exists() or self.replay_path.stat().st_size == 0:
            return self._empty_replay_df()
        try:
            df = pd.read_csv(self.replay_path)
            for col in self._empty_replay_df().columns:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception:
            return self._empty_replay_df()


    def _append_replay(
        self,
        state_text: str,
        state_vector: List[float],
        action: str,
        reward: float,
        next_state_text: Optional[str] = None,
        next_state_vector: Optional[List[float]] = None,
        done: bool = True,
    ) -> int:
        action = action if action in self.ACTIONS else "KEEP_SIGNAL"
        next_state_vector = next_state_vector if next_state_vector is not None else state_vector
        row = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "state_text": state_text,
            "state_vector_json": json.dumps([float(x) for x in state_vector]),
            "action": action,
            "action_index": self.ACTION_INDEX[action],
            "reward": float(reward),
            "next_state_text": next_state_text or "",
            "next_state_vector_json": json.dumps([float(x) for x in next_state_vector]),
            "done": bool(done),
        }

        # SQLite is the primary replay memory. CSV is only a compatibility mirror.
        self._append_replay_to_sqlite(row)
        df = self._read_replay()
        try:
            csv_df = df.copy()
            csv_df.to_csv(self.replay_path, index=False)
        except Exception:
            pass
        return len(df)


    def _sample_replay_batch(self, df: pd.DataFrame) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if len(df) < self.min_replay_samples:
            return None
        sample_size = min(self.batch_size, len(df))
        batch = df.sample(sample_size, random_state=random.randint(1, 1_000_000))

        states, actions, rewards, next_states, dones = [], [], [], [], []
        for _, row in batch.iterrows():
            try:
                states.append(json.loads(row["state_vector_json"]))
                actions.append(int(row["action_index"]))
                rewards.append(float(row["reward"]))
                next_states.append(json.loads(row["next_state_vector_json"]))
                dones.append(1.0 if str(row.get("done", "True")).lower() in ["true", "1", "yes"] else 0.0)
            except Exception:
                continue

        if not states:
            return None

        return (
            torch.tensor(states, dtype=torch.float32, device=self.device),
            torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32, device=self.device),
            torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1),
        )


    def _record_training_loss(self, loss_value: Optional[float], replay_count: int) -> None:
        """Keep a lightweight local loss log for diagnostics."""
        if loss_value is None:
            return
        row = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "loss": float(loss_value),
            "replay_count": int(replay_count),
            "training_steps": int(self.training_steps),
            "epsilon": float(self.epsilon),
        }
        try:
            df = pd.DataFrame([row])
            if self.loss_history_path.exists() and self.loss_history_path.stat().st_size > 0:
                old_df = pd.read_csv(self.loss_history_path)
                df = pd.concat([old_df, df], ignore_index=True).tail(1000)
            df.to_csv(self.loss_history_path, index=False)
        except Exception:
            pass


    def _recent_loss(self) -> Optional[float]:
        if self.last_training_loss is not None:
            return float(self.last_training_loss)
        try:
            if not self.loss_history_path.exists() or self.loss_history_path.stat().st_size == 0:
                return None
            df = pd.read_csv(self.loss_history_path)
            if df.empty or "loss" not in df.columns:
                return None
            return float(df["loss"].dropna().iloc[-1])
        except Exception:
            return None


    def _record_state_q_values(self, state: str, q_values: Dict[str, float]) -> None:
        self.q_table[state] = {action: float(q_values.get(action, 0.0)) for action in self.ACTIONS}
        self._save_q_table_compat()

