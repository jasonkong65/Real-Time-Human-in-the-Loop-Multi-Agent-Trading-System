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


class StorageSchemaMixin:

    """Mixin for defining the database schema and initialization logic for the StorageAgent, including methods to create necessary tables and indexes, maintain schema versioning, and perform compatibility migrations for reward updates and paper decisions."""

    def _ensure_reward_compatible_schema(self) -> None:
        """
        Keep StorageAgent and RewardAgent compatible when the same SQLite file
        has been created by an older version of the app.

        Earlier StorageAgent tables used fields such as reward_updates.due_at_utc
        and reward_updates.reward_horizon_days. RewardAgent now uses
        target_date_utc and horizon_days. Without this migration, Streamlit can
        crash with: sqlite3.OperationalError: no such column: target_date_utc.
        """
        paper_specs = {
            # Columns used by RewardAgent
            "paper_status": "TEXT",
            "entry_time_utc": "TEXT",
            "q_state": "TEXT",
            "duplicate_group_key": "TEXT",
            "risk_result_json": "TEXT",
            # Columns used by StorageAgent
            "run_id": "TEXT",
            "strategy_action": "TEXT",
            "status": "TEXT",
            "created_at_utc": "TEXT",
            "updated_at_utc": "TEXT",
            "raw_json": "TEXT",
        }
        reward_specs = {
            # Legacy/StorageAgent columns
            "id": "TEXT",
            "reward_horizon_days": "INTEGER",
            "due_at_utc": "TEXT",
            "final_signal": "TEXT",
            "strategy_action": "TEXT",
            "risk_action": "TEXT",
            "raw_json": "TEXT",
            # RewardAgent columns
            "update_id": "TEXT",
            "horizon_display": "TEXT",
            "horizon_days": "INTEGER",
            "target_date_utc": "TEXT",
            "latest_date": "TEXT",
            "dqn_update_json": "TEXT",
            "dqn_update_summary": "TEXT",
            "notes": "TEXT",
            "created_at_utc": "TEXT",
        }

        for column, column_type in paper_specs.items():
            try:
                self.backend.add_column_if_missing("paper_decisions", column, column_type)
            except Exception:
                pass

        for column, column_type in reward_specs.items():
            try:
                self.backend.add_column_if_missing("reward_updates", column, column_type)
            except Exception:
                pass

        # Keep StorageAgent's audit-style replay table compatible with the
        # strict DQN replay table used by RiskAgent. The two agents share the
        # same SQLite file, so the table may have been created by either side.
        dqn_specs = {
            "created_at_utc": "TEXT",
            "state_text": "TEXT",
            "state_vector_json": "TEXT",
            "action_index": "INTEGER",
            "next_state_text": "TEXT",
            "next_state_vector_json": "TEXT",
            "source": "TEXT",
        }
        for column, column_type in dqn_specs.items():
            try:
                self.backend.add_column_if_missing("risk_dqn_replay", column, column_type)
            except Exception:
                pass

        # Best-effort backfill. Each statement is guarded so a partially old DB
        # will not stop the app from opening.
        backfills = [
            """
            UPDATE paper_decisions
            SET paper_status = COALESCE(NULLIF(paper_status, ''), NULLIF(status, ''), 'PAPER_MONITOR_ONLY')
            WHERE paper_status IS NULL OR paper_status = ''
            """,
            """
            UPDATE paper_decisions
            SET entry_time_utc = COALESCE(NULLIF(entry_time_utc, ''), created_at_utc, datetime('now'))
            WHERE entry_time_utc IS NULL OR entry_time_utc = ''
            """,
            """
            UPDATE reward_updates
            SET update_id = COALESCE(NULLIF(update_id, ''), NULLIF(id, ''), lower(hex(randomblob(16))))
            WHERE update_id IS NULL OR update_id = ''
            """,
            """
            UPDATE reward_updates
            SET target_date_utc = COALESCE(NULLIF(target_date_utc, ''), NULLIF(due_at_utc, ''), updated_at_utc, datetime('now'))
            WHERE target_date_utc IS NULL OR target_date_utc = ''
            """,
            """
            UPDATE reward_updates
            SET horizon_days = COALESCE(horizon_days, reward_horizon_days, 1)
            WHERE horizon_days IS NULL
            """,
            """
            UPDATE reward_updates
            SET horizon_display = COALESCE(NULLIF(horizon_display, ''), NULLIF(horizon_label, ''))
            WHERE horizon_display IS NULL OR horizon_display = ''
            """,
            """
            UPDATE reward_updates
            SET created_at_utc = COALESCE(NULLIF(created_at_utc, ''), updated_at_utc, target_date_utc, datetime('now'))
            WHERE created_at_utc IS NULL OR created_at_utc = ''
            """,
        ]
        for sql in backfills:
            try:
                self.backend.execute(sql)
            except Exception:
                pass

        for sql in [
            "CREATE INDEX IF NOT EXISTS idx_reward_updates_status_target ON reward_updates(status, target_date_utc)",
            "CREATE INDEX IF NOT EXISTS idx_paper_decisions_paper_status ON paper_decisions(paper_status)",
        ]:
            try:
                self.backend.execute(sql)
            except Exception:
                pass


    def init_db(self) -> Dict[str, Any]:
        statements = [
            """
            CREATE TABLE IF NOT EXISTS storage_meta (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at_utc TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS historical_prices (
                symbol TEXT NOT NULL,
                period TEXT,
                interval TEXT NOT NULL,
                price_timestamp TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume REAL,
                source TEXT,
                downloaded_at_utc TEXT,
                raw_json TEXT,
                PRIMARY KEY (symbol, interval, price_timestamp)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS historical_metadata (
                symbol TEXT NOT NULL,
                period TEXT NOT NULL,
                interval TEXT NOT NULL,
                latest_timestamp TEXT,
                downloaded_at_utc TEXT,
                record_count INTEGER,
                source TEXT,
                storage_mode TEXT,
                stale_warning INTEGER,
                warnings_json TEXT,
                raw_json TEXT,
                PRIMARY KEY (symbol, period, interval)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                run_id TEXT PRIMARY KEY,
                symbol TEXT,
                selected_price REAL,
                validation_confidence TEXT,
                validation_next_action TEXT,
                analyst_signal TEXT,
                model_signal TEXT,
                model_confidence TEXT,
                final_signal TEXT,
                risk_level TEXT,
                risk_action TEXT,
                strategy_action TEXT,
                strategy_level TEXT,
                reward_decision_id TEXT,
                created_at_utc TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS agent_outputs (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                symbol TEXT,
                agent_name TEXT,
                output_json TEXT,
                created_at_utc TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS market_quotes (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                symbol TEXT,
                source TEXT,
                current_price REAL,
                open_price REAL,
                high_price REAL,
                low_price REAL,
                previous_close REAL,
                quote_timestamp TEXT,
                created_at_utc TEXT,
                raw_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS paper_decisions (
                decision_id TEXT PRIMARY KEY,
                run_id TEXT,
                symbol TEXT,
                entry_price REAL,
                final_signal TEXT,
                risk_action TEXT,
                risk_level TEXT,
                strategy_action TEXT,
                status TEXT,
                created_at_utc TEXT,
                updated_at_utc TEXT,
                raw_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS reward_updates (
                id TEXT PRIMARY KEY,
                decision_id TEXT,
                symbol TEXT,
                entry_price REAL,
                latest_close REAL,
                future_return REAL,
                reward REAL,
                reward_horizon_days INTEGER,
                horizon_label TEXT,
                final_signal TEXT,
                strategy_action TEXT,
                risk_action TEXT,
                status TEXT,
                due_at_utc TEXT,
                updated_at_utc TEXT,
                raw_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS risk_dqn_replay (
                transition_id TEXT PRIMARY KEY,
                symbol TEXT,
                state_json TEXT,
                action TEXT,
                reward REAL,
                next_state_json TEXT,
                done INTEGER,
                source_decision_id TEXT,
                horizon_label TEXT,
                future_return REAL,
                created_at_utc TEXT,
                raw_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                symbol TEXT,
                model_type TEXT,
                best_params_json TEXT,
                accuracy REAL,
                balanced_accuracy REAL,
                macro_f1 REAL,
                sell_risk_recall REAL,
                training_samples INTEGER,
                model_path TEXT,
                metadata_path TEXT,
                save_decision TEXT,
                created_at_utc TEXT,
                raw_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS screener_runs (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                universe_size INTEGER,
                top_n INTEGER,
                period TEXT,
                result_json TEXT,
                created_at_utc TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS llm_reports (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                symbol TEXT,
                report_type TEXT,
                provider TEXT,
                model TEXT,
                source TEXT,
                llm_available INTEGER,
                plain_language_report TEXT,
                created_at_utc TEXT,
                raw_json TEXT
            )
            """,
        ]
        for sql in statements:
            self.backend.execute(sql)

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_hist_symbol_interval_time ON historical_prices(symbol, interval, price_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_market_quotes_symbol_time ON market_quotes(symbol, created_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_agent_outputs_run ON agent_outputs(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_rewards_decision ON reward_updates(decision_id)",
            "CREATE INDEX IF NOT EXISTS idx_rewards_symbol ON reward_updates(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_dqn_created ON risk_dqn_replay(created_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_training_symbol ON training_runs(symbol)",
            "CREATE INDEX IF NOT EXISTS idx_screener_created ON screener_runs(created_at_utc)",
        ]
        for sql in indexes:
            try:
                self.backend.execute(sql)
            except Exception:
                pass

        self._ensure_reward_compatible_schema()

        self.backend.upsert(
            "storage_meta",
            {
                "key": "schema_version",
                "value": self.SCHEMA_VERSION,
                "updated_at_utc": self._now_utc(),
            },
            conflict_cols=["key"],
        )
        self.backend.upsert(
            "storage_meta",
            {
                "key": "database_url_kind",
                "value": "postgresql" if self.backend.dialect == "postgresql" else "sqlite",
                "updated_at_utc": self._now_utc(),
            },
            conflict_cols=["key"],
        )
        return {"success": True, "schema_version": self.SCHEMA_VERSION, "dialect": self.backend.dialect}

