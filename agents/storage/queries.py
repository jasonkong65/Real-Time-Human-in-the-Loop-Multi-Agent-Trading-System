from __future__ import annotations

from __future__ import annotations

import json

import os

import uuid

from datetime import datetime, timezone, timedelta

from pathlib import Path

from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from agents.database_backend import DatabaseBackend
except Exception:
    from database_backend import DatabaseBackend


class StorageQueryMixin:


    def get_recent_pipeline_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self.backend.query(
            f"SELECT * FROM pipeline_runs ORDER BY created_at_utc DESC LIMIT {self.backend.safe_limit(limit, 10)}"
        )


    def get_agent_outputs_for_run(self, run_id: str, parse_json: bool = False) -> List[Dict[str, Any]]:
        rows = self.backend.query(
            "SELECT * FROM agent_outputs WHERE run_id = :run_id ORDER BY created_at_utc ASC",
            {"run_id": run_id},
        )
        if parse_json:
            for row in rows:
                row["output"] = self._from_json(row.get("output_json"), {})
        return rows


    def get_recent_agent_outputs(self, limit: int = 20, parse_json: bool = False) -> List[Dict[str, Any]]:
        rows = self.backend.query(
            f"SELECT * FROM agent_outputs ORDER BY created_at_utc DESC LIMIT {self.backend.safe_limit(limit, 20)}"
        )
        if parse_json:
            for row in rows:
                row["output"] = self._from_json(row.get("output_json"), {})
        return rows


    def get_latest_agent_output(self, agent_name: str, symbol: Optional[str] = None, parse_json: bool = True) -> Optional[Dict[str, Any]]:
        params = {"agent_name": agent_name}
        where = "agent_name = :agent_name"
        if symbol:
            where += " AND symbol = :symbol"
            params["symbol"] = self._normalise_symbol(symbol)
        rows = self.backend.query(
            f"SELECT * FROM agent_outputs WHERE {where} ORDER BY created_at_utc DESC LIMIT 1",
            params,
        )
        if not rows:
            return None
        row = rows[0]
        if parse_json:
            row["output"] = self._from_json(row.get("output_json"), {})
        return row


    def get_recent_screener_runs(self, limit: int = 10, parse_json: bool = False) -> List[Dict[str, Any]]:
        rows = self.backend.query(
            f"SELECT * FROM screener_runs ORDER BY created_at_utc DESC LIMIT {self.backend.safe_limit(limit, 10)}"
        )
        if parse_json:
            for row in rows:
                row["result"] = self._from_json(row.get("result_json"), {})
        return rows


    def get_paper_decisions(self, status: Optional[str] = None, limit: int = 10000) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        where = "1 = 1"
        if status:
            where += " AND status = :status"
            params["status"] = status
        return self.backend.query(
            f"SELECT * FROM paper_decisions WHERE {where} ORDER BY created_at_utc DESC LIMIT {self.backend.safe_limit(limit, 10000)}",
            params,
        )


    def get_reward_updates(self, status: Optional[str] = None, limit: int = 10000) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {}
        where = "1 = 1"
        if status:
            where += " AND status = :status"
            params["status"] = status
        return self.backend.query(
            f"SELECT * FROM reward_updates WHERE {where} ORDER BY updated_at_utc DESC LIMIT {self.backend.safe_limit(limit, 10000)}",
            params,
        )


    def _directional_win(self, row: Dict[str, Any]) -> Optional[int]:
        final_signal = str(row.get("final_signal") or "").upper()
        future_return = self._safe_float(row.get("future_return"))
        if future_return is None:
            return None
        if "BUY" in final_signal:
            return 1 if future_return > 0 else 0
        if "SELL" in final_signal or "RISK" in final_signal:
            return 1 if future_return < 0 else 0
        if "HOLD" in final_signal:
            return 1 if abs(future_return) <= 0.02 else 0
        return 1 if row.get("reward", 0) and float(row.get("reward", 0)) > 0 else 0


    def get_reward_summary(self) -> Dict[str, Any]:
        rows = self.get_reward_updates(limit=100000)
        completed = [r for r in rows if self._safe_float(r.get("reward")) is not None]
        rewards = [self._safe_float(r.get("reward")) for r in completed if self._safe_float(r.get("reward")) is not None]
        future_returns = [self._safe_float(r.get("future_return")) for r in completed if self._safe_float(r.get("future_return")) is not None]
        directional = [self._directional_win(r) for r in completed]
        directional = [d for d in directional if d is not None]
        win_rewards = [1 if r > 0 else 0 for r in rewards]
        return {
            "completed_count": len(completed),
            "avg_reward": sum(rewards) / len(rewards) if rewards else None,
            "avg_future_return": sum(future_returns) / len(future_returns) if future_returns else None,
            "reward_win_rate": sum(win_rewards) / len(win_rewards) if win_rewards else None,
            "directional_win_rate": sum(directional) / len(directional) if directional else None,
        }


    def _group_reward_stats(self, group_field: str) -> List[Dict[str, Any]]:
        rows = self.get_reward_updates(limit=100000)
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            key = str(row.get(group_field) or "UNKNOWN")
            groups.setdefault(key, []).append(row)
        output = []
        for key, items in groups.items():
            rewards = [self._safe_float(r.get("reward")) for r in items if self._safe_float(r.get("reward")) is not None]
            future_returns = [self._safe_float(r.get("future_return")) for r in items if self._safe_float(r.get("future_return")) is not None]
            wins = [1 if r > 0 else 0 for r in rewards]
            directional = [self._directional_win(r) for r in items]
            directional = [d for d in directional if d is not None]
            output.append(
                {
                    group_field: key,
                    "count": len(items),
                    "completed_count": len(rewards),
                    "avg_reward": sum(rewards) / len(rewards) if rewards else None,
                    "avg_future_return": sum(future_returns) / len(future_returns) if future_returns else None,
                    "reward_win_rate": sum(wins) / len(wins) if wins else None,
                    "directional_win_rate": sum(directional) / len(directional) if directional else None,
                }
            )
        return sorted(output, key=lambda x: x.get("completed_count", 0), reverse=True)


    def get_reward_by_strategy_action(self) -> List[Dict[str, Any]]:
        return self._group_reward_stats("strategy_action")


    def get_reward_by_signal_type(self) -> List[Dict[str, Any]]:
        return self._group_reward_stats("final_signal")


    def get_reward_by_horizon(self) -> List[Dict[str, Any]]:
        return self._group_reward_stats("horizon_label")


    def get_evaluator_dataset(self, limit: int = 10000) -> Dict[str, Any]:
        return {
            "storage_backend": self.backend.dialect,
            "paper_decisions": self.get_paper_decisions(limit=limit),
            "reward_updates": self.get_reward_updates(limit=limit),
            "reward_summary": self.get_reward_summary(),
            "reward_by_strategy_action": self.get_reward_by_strategy_action(),
            "reward_by_signal_type": self.get_reward_by_signal_type(),
            "reward_by_horizon": self.get_reward_by_horizon(),
            "dqn_replay_memory": self.get_dqn_replay_memory(limit=limit),
            "recent_pipeline_runs": self.get_recent_pipeline_runs(limit=50),
            "recent_screener_runs": self.get_recent_screener_runs(limit=50, parse_json=True),
        }


    def get_storage_summary(self) -> Dict[str, Any]:
        tables = [
            "historical_prices", "historical_metadata", "market_quotes", "pipeline_runs",
            "agent_outputs", "paper_decisions", "reward_updates", "risk_dqn_replay",
            "training_runs", "screener_runs", "llm_reports",
        ]
        counts = {}
        for table in tables:
            try:
                row = self.backend.query(f"SELECT COUNT(*) AS count FROM {table}")[0]
                counts[table] = row.get("count", 0)
            except Exception:
                counts[table] = None
        return {
            "success": True,
            "database_url": self.database_url,
            "dialect": self.backend.dialect,
            "schema_version": self.SCHEMA_VERSION,
            "table_counts": counts,
        }

