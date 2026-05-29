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


class StorageAgent:
    """
    PostgreSQL-ready persistent memory layer for the multi-agent stock system.

    Default database:
        DATABASE_URL=sqlite:///data/trading_system.db

    Future production database:
        DATABASE_URL=postgresql+psycopg2://user:password@host:5432/trading_system

    This class keeps the existing app/agent-facing method names, while moving the
    storage design toward a database-first architecture:
    - historical_prices and historical_metadata replace per-symbol CSV files as the main store.
    - market quotes, agent outputs, rewards, DQN replay, training metadata, screener runs,
      and LLM reports live in the same persistent memory layer.
    - CSV / Parquet can still be used as optional export or fallback, but not as the main source.
    """

    SCHEMA_VERSION = "2.0_database_first_postgres_ready"

    def __init__(
        self,
        db_path: str = "data/trading_system.db",
        database_url: Optional[str] = None,
        auto_init: bool = True,
    ):
        self.db_path = Path(db_path)
        self.database_url = database_url or os.getenv("DATABASE_URL") or f"sqlite:///{db_path}"
        self.backend = DatabaseBackend(database_url=self.database_url, sqlite_path=db_path)
        if auto_init:
            self.init_db()

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------
    def _now_utc(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _new_id(self, prefix: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{prefix}_{stamp}_{uuid.uuid4().hex[:10]}"

    def _to_json(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return json.dumps({"serialization_error": repr(value)}, ensure_ascii=False)

    def _from_json(self, value: Any, default=None):
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return default

    def _normalise_symbol(self, symbol: Any) -> str:
        return str(symbol or "UNKNOWN").upper().strip()

    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None

    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(float(value))
        except Exception:
            return None

    def _get_nested(self, data: Dict[str, Any], keys: Sequence[str], default=None):
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current

    def _period_to_start_timestamp(self, period: str) -> Optional[str]:
        period = str(period or "").lower().strip()
        if not period or period == "max":
            return None
        days_map = {
            "1d": 1,
            "5d": 5,
            "7d": 7,
            "30d": 30,
            "1mo": 31,
            "3mo": 93,
            "6mo": 186,
            "1y": 366,
            "2y": 732,
            "5y": 1830,
            "10y": 3660,
        }
        days = days_map.get(period)
        if days is None:
            return None
        start = datetime.now(timezone.utc) - timedelta(days=days + 5)
        return start.strftime("%Y-%m-%d %H:%M:%S")

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
            "paper_status": "TEXT",
            "entry_time_utc": "TEXT",
            "q_state": "TEXT",
            "duplicate_group_key": "TEXT",
            "risk_result_json": "TEXT",
            "updated_at_utc": "TEXT",
        }
        reward_specs = {
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

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Historical price store
    # ------------------------------------------------------------------
    def record_historical_prices(
        self,
        symbol: str,
        prices: Any,
        period: str = "1y",
        interval: str = "1d",
        source: str = "yfinance",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        symbol = self._normalise_symbol(symbol)
        downloaded_at = self._now_utc()

        if pd is not None and isinstance(prices, pd.DataFrame):
            records = prices.to_dict("records")
        elif isinstance(prices, dict) and "prices" in prices:
            records = prices.get("prices") or []
        else:
            records = list(prices or [])

        rows = []
        for rec in records:
            date_value = rec.get("date") or rec.get("datetime") or rec.get("price_timestamp") or rec.get("timestamp")
            if date_value is None:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "period": period,
                    "interval": interval,
                    "price_timestamp": str(date_value),
                    "open": self._safe_float(rec.get("open")),
                    "high": self._safe_float(rec.get("high")),
                    "low": self._safe_float(rec.get("low")),
                    "close": self._safe_float(rec.get("close")),
                    "adj_close": self._safe_float(rec.get("adj_close") or rec.get("adjclose")),
                    "volume": self._safe_float(rec.get("volume")),
                    "source": source,
                    "downloaded_at_utc": downloaded_at,
                    "raw_json": self._to_json(rec),
                }
            )

        for row in rows:
            self.backend.upsert(
                "historical_prices",
                row,
                conflict_cols=["symbol", "interval", "price_timestamp"],
            )

        latest = max([r["price_timestamp"] for r in rows], default=None)
        self.record_historical_metadata(
            symbol=symbol,
            period=period,
            interval=interval,
            latest_timestamp=latest,
            record_count=len(rows),
            source=source,
            storage_mode="database",
            metadata=metadata or {},
        )
        return {"success": True, "symbol": symbol, "rows_written": len(rows), "latest_timestamp": latest}

    def record_historical_metadata(
        self,
        symbol: str,
        period: str,
        interval: str,
        latest_timestamp: Optional[str],
        record_count: int,
        source: str,
        storage_mode: str = "database",
        metadata: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        stale_warning: bool = False,
    ) -> Dict[str, Any]:
        row = {
            "symbol": self._normalise_symbol(symbol),
            "period": period,
            "interval": interval,
            "latest_timestamp": latest_timestamp,
            "downloaded_at_utc": self._now_utc(),
            "record_count": int(record_count or 0),
            "source": source,
            "storage_mode": storage_mode,
            "stale_warning": 1 if stale_warning else 0,
            "warnings_json": self._to_json(warnings or []),
            "raw_json": self._to_json(metadata or {}),
        }
        self.backend.upsert("historical_metadata", row, conflict_cols=["symbol", "period", "interval"])
        return {"success": True, "symbol": row["symbol"], "period": period, "interval": interval}

    def get_historical_prices(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        limit: Optional[int] = None,
        as_dataframe: bool = True,
    ) -> Any:
        symbol = self._normalise_symbol(symbol)
        params: Dict[str, Any] = {"symbol": symbol, "interval": interval}
        start_ts = self._period_to_start_timestamp(period)
        where = "symbol = :symbol AND interval = :interval"
        if start_ts:
            where += " AND price_timestamp >= :start_ts"
            params["start_ts"] = start_ts

        sql = f"""
            SELECT price_timestamp AS date, open, high, low, close, adj_close, volume, source, downloaded_at_utc
            FROM historical_prices
            WHERE {where}
            ORDER BY price_timestamp ASC
        """
        if limit:
            sql += f" LIMIT {self.backend.safe_limit(limit)}"
        rows = self.backend.query(sql, params)
        if as_dataframe and pd is not None:
            return pd.DataFrame(rows)
        return rows

    def get_historical_metadata(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[Dict[str, Any]]:
        rows = self.backend.query(
            """
            SELECT * FROM historical_metadata
            WHERE symbol = :symbol AND period = :period AND interval = :interval
            """,
            {"symbol": self._normalise_symbol(symbol), "period": period, "interval": interval},
        )
        if not rows:
            return None
        row = rows[0]
        row["warnings"] = self._from_json(row.get("warnings_json"), [])
        row["metadata"] = self._from_json(row.get("raw_json"), {})
        return row

    # ------------------------------------------------------------------
    # Pipeline and agent output records
    # ------------------------------------------------------------------
    def create_pipeline_run(
        self,
        symbol: str,
        validation_result: Optional[Dict[str, Any]] = None,
        analysis_result: Optional[Dict[str, Any]] = None,
        training_result: Optional[Dict[str, Any]] = None,
        signal_result: Optional[Dict[str, Any]] = None,
        risk_result: Optional[Dict[str, Any]] = None,
        strategy_result: Optional[Dict[str, Any]] = None,
        reward_record_result: Optional[Dict[str, Any]] = None,
        selected_price: Optional[float] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        validation_result = validation_result or {}
        analysis_result = analysis_result or {}
        training_result = training_result or {}
        signal_result = signal_result or {}
        risk_result = risk_result or {}
        strategy_result = strategy_result or {}
        reward_record_result = reward_record_result or {}

        run_id = run_id or self._new_id("run")
        symbol = self._normalise_symbol(symbol)
        row = {
            "run_id": run_id,
            "symbol": symbol,
            "selected_price": self._safe_float(selected_price or validation_result.get("selected_price")),
            "validation_confidence": validation_result.get("confidence"),
            "validation_next_action": validation_result.get("next_action"),
            "analyst_signal": analysis_result.get("analyst_signal") or analysis_result.get("display_signal"),
            "model_signal": signal_result.get("model_signal") or signal_result.get("signal") or training_result.get("model_signal"),
            "model_confidence": signal_result.get("confidence_level") or signal_result.get("model_confidence_level"),
            "final_signal": risk_result.get("final_signal"),
            "risk_level": risk_result.get("risk_level"),
            "risk_action": risk_result.get("risk_action") or risk_result.get("q_action") or risk_result.get("dqn_action"),
            "strategy_action": strategy_result.get("strategy_action"),
            "strategy_level": strategy_result.get("strategy_level"),
            "reward_decision_id": reward_record_result.get("decision_id"),
            "created_at_utc": self._now_utc(),
        }
        self.backend.upsert("pipeline_runs", row, conflict_cols=["run_id"])
        return {"success": True, "run_id": run_id, "symbol": symbol}

    def record_agent_output(self, run_id: Optional[str], symbol: str, agent_name: str, output: Dict[str, Any]) -> Dict[str, Any]:
        row = {
            "id": self._new_id("agent"),
            "run_id": run_id,
            "symbol": self._normalise_symbol(symbol),
            "agent_name": agent_name,
            "output_json": self._to_json(output),
            "created_at_utc": self._now_utc(),
        }
        self.backend.insert("agent_outputs", row)
        return {"success": True, "id": row["id"], "agent_name": agent_name}

    def record_market_quotes(self, multi_quote: Dict[str, Any], run_id: Optional[str] = None, symbol: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(multi_quote, dict):
            return {"success": False, "error": "multi_quote is not a dictionary"}

        symbol = self._normalise_symbol(symbol or multi_quote.get("symbol"))
        possible_sources = []
        for key in ["primary_source", "secondary_source", "finnhub_quote", "alpha_vantage_quote", "quote"]:
            val = multi_quote.get(key)
            if isinstance(val, dict):
                possible_sources.append(val)
        if not possible_sources and "source" in multi_quote:
            possible_sources.append(multi_quote)

        count = 0
        for quote in possible_sources:
            if quote.get("success") is False:
                continue
            row = {
                "id": self._new_id("quote"),
                "run_id": run_id,
                "symbol": self._normalise_symbol(quote.get("symbol") or symbol),
                "source": quote.get("source") or quote.get("provider") or "unknown",
                "current_price": self._safe_float(quote.get("current_price") or quote.get("price") or quote.get("c")),
                "open_price": self._safe_float(quote.get("open_price") or quote.get("open") or quote.get("o")),
                "high_price": self._safe_float(quote.get("high_price") or quote.get("high") or quote.get("h")),
                "low_price": self._safe_float(quote.get("low_price") or quote.get("low") or quote.get("l")),
                "previous_close": self._safe_float(quote.get("previous_close") or quote.get("previous_close_price") or quote.get("pc")),
                "quote_timestamp": str(quote.get("timestamp") or quote.get("latest_trading_day") or quote.get("quote_timestamp") or ""),
                "created_at_utc": self._now_utc(),
                "raw_json": self._to_json(quote),
            }
            self.backend.insert("market_quotes", row)
            count += 1
        return {"success": True, "quotes_recorded": count}

    def record_paper_decision(self, reward_record_result: Dict[str, Any], run_id: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(reward_record_result, dict):
            return {"success": False, "error": "reward_record_result is not a dictionary"}
        if reward_record_result.get("success") is False:
            return {"success": False, "skipped": True, "reason": reward_record_result.get("summary") or reward_record_result.get("error")}

        decision_id = reward_record_result.get("decision_id") or reward_record_result.get("paper_decision_id") or self._new_id("decision")
        row = {
            "decision_id": decision_id,
            "run_id": run_id or reward_record_result.get("run_id"),
            "symbol": self._normalise_symbol(reward_record_result.get("symbol")),
            "entry_price": self._safe_float(reward_record_result.get("entry_price")),
            "final_signal": reward_record_result.get("final_signal"),
            "risk_action": reward_record_result.get("risk_action"),
            "risk_level": reward_record_result.get("risk_level"),
            "strategy_action": reward_record_result.get("strategy_action"),
            "status": reward_record_result.get("paper_status") or reward_record_result.get("status") or "PAPER_MONITOR_ONLY",
            "created_at_utc": reward_record_result.get("created_at_utc") or self._now_utc(),
            "updated_at_utc": self._now_utc(),
            "raw_json": self._to_json(reward_record_result),
        }
        self.backend.upsert("paper_decisions", row, conflict_cols=["decision_id"])
        return {"success": True, "decision_id": decision_id}

    def record_reward_updates(self, auto_reward_update_result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(auto_reward_update_result, dict):
            return {"success": False, "error": "auto_reward_update_result is not a dictionary"}
        updates = auto_reward_update_result.get("updates") or auto_reward_update_result.get("completed_updates") or []
        if isinstance(updates, dict):
            updates = [updates]
        count = 0
        for update in updates:
            update_id = str(update.get("update_id") or update.get("id") or update.get("reward_update_id") or self._new_id("reward"))
            horizon_days = self._safe_int(update.get("horizon_days") or update.get("reward_horizon_days"))
            target_date = update.get("target_date_utc") or update.get("due_at_utc")
            row = {
                # Old StorageAgent schema
                "id": update_id,
                "reward_horizon_days": horizon_days,
                "due_at_utc": target_date,
                # New RewardAgent schema
                "update_id": update_id,
                "horizon_days": horizon_days,
                "horizon_display": update.get("horizon_display") or update.get("horizon_label"),
                "target_date_utc": target_date,
                "latest_date": update.get("latest_date"),
                "dqn_update_json": self._to_json(update.get("dqn_update") or update.get("dqn_update_json")),
                "dqn_update_summary": update.get("dqn_update_summary"),
                "notes": update.get("notes"),
                "created_at_utc": update.get("created_at_utc") or update.get("updated_at_utc") or self._now_utc(),
                # Shared fields
                "decision_id": update.get("decision_id"),
                "symbol": self._normalise_symbol(update.get("symbol")),
                "entry_price": self._safe_float(update.get("entry_price")),
                "latest_close": self._safe_float(update.get("latest_close")),
                "future_return": self._safe_float(update.get("future_return")),
                "reward": self._safe_float(update.get("reward")),
                "horizon_label": update.get("horizon_label"),
                "final_signal": update.get("final_signal"),
                "strategy_action": update.get("strategy_action"),
                "risk_action": update.get("risk_action"),
                "status": update.get("status") or "completed",
                "updated_at_utc": update.get("updated_at_utc") or self._now_utc(),
                "raw_json": self._to_json(update),
            }
            self.backend.upsert("reward_updates", row, conflict_cols=["id"])
            count += 1
        return {"success": True, "updates_recorded": count}

    def record_dqn_replay_transition(self, transition: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(transition, dict):
            return {"success": False, "error": "transition is not a dictionary"}
        transition_id = transition.get("transition_id") or self._new_id("dqn")
        row = {
            "transition_id": transition_id,
            "symbol": self._normalise_symbol(transition.get("symbol")),
            "state_json": self._to_json(transition.get("state") or transition.get("state_vector") or transition.get("q_state")),
            "action": transition.get("action") or transition.get("risk_action") or transition.get("q_action"),
            "reward": self._safe_float(transition.get("reward")),
            "next_state_json": self._to_json(transition.get("next_state") or transition.get("next_state_vector")),
            "done": 1 if transition.get("done") else 0,
            "source_decision_id": transition.get("decision_id") or transition.get("source_decision_id"),
            "horizon_label": transition.get("horizon_label"),
            "future_return": self._safe_float(transition.get("future_return")),
            "created_at_utc": transition.get("created_at_utc") or self._now_utc(),
            "raw_json": self._to_json(transition),
        }
        self.backend.upsert("risk_dqn_replay", row, conflict_cols=["transition_id"])
        return {"success": True, "transition_id": transition_id}

    def get_dqn_replay_memory(self, limit: int = 10000) -> List[Dict[str, Any]]:
        rows = self.backend.query(
            f"""
            SELECT * FROM risk_dqn_replay
            ORDER BY created_at_utc DESC
            LIMIT {self.backend.safe_limit(limit, 10000)}
            """
        )
        for row in rows:
            row["state"] = self._from_json(row.get("state_json"), [])
            row["next_state"] = self._from_json(row.get("next_state_json"), [])
            row["raw"] = self._from_json(row.get("raw_json"), {})
        return rows

    def record_training_run(self, run_id: Optional[str], symbol: str, training_result: Dict[str, Any]) -> Dict[str, Any]:
        training_result = training_result or {}
        metrics = training_result.get("metrics") or training_result.get("best_metrics") or training_result.get("model_metrics") or {}
        row = {
            "id": self._new_id("train"),
            "run_id": run_id,
            "symbol": self._normalise_symbol(symbol or training_result.get("symbol")),
            "model_type": training_result.get("model_type") or training_result.get("best_model_type"),
            "best_params_json": self._to_json(training_result.get("best_params") or training_result.get("params") or {}),
            "accuracy": self._safe_float(metrics.get("accuracy") or training_result.get("accuracy")),
            "balanced_accuracy": self._safe_float(metrics.get("balanced_accuracy") or training_result.get("balanced_accuracy")),
            "macro_f1": self._safe_float(metrics.get("macro_f1") or training_result.get("macro_f1")),
            "sell_risk_recall": self._safe_float(metrics.get("sell_risk_recall") or training_result.get("sell_risk_recall")),
            "training_samples": self._safe_int(training_result.get("training_samples") or training_result.get("num_samples")),
            "model_path": training_result.get("model_path"),
            "metadata_path": training_result.get("metadata_path"),
            "save_decision": training_result.get("save_decision"),
            "created_at_utc": self._now_utc(),
            "raw_json": self._to_json(training_result),
        }
        self.backend.insert("training_runs", row)
        return {"success": True, "id": row["id"]}

    def record_llm_report(self, run_id: Optional[str], symbol: str, report_result: Dict[str, Any]) -> Dict[str, Any]:
        report_result = report_result or {}
        row = {
            "id": self._new_id("llm"),
            "run_id": run_id,
            "symbol": self._normalise_symbol(symbol or report_result.get("symbol")),
            "report_type": report_result.get("report_type"),
            "provider": report_result.get("provider") or "groq",
            "model": report_result.get("model"),
            "source": report_result.get("source"),
            "llm_available": 1 if report_result.get("llm_available") else 0,
            "plain_language_report": report_result.get("plain_language_report") or report_result.get("report") or "",
            "created_at_utc": self._now_utc(),
            "raw_json": self._to_json(report_result),
        }
        self.backend.insert("llm_reports", row)
        return {"success": True, "id": row["id"]}

    def record_screener_run(self, screener_result: Dict[str, Any], top_n: Optional[int] = None, period: Optional[str] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
        screener_result = screener_result or {}
        row = {
            "id": self._new_id("screen"),
            "run_id": run_id,
            "universe_size": self._safe_int(screener_result.get("universe_size") or screener_result.get("scanned_count")),
            "top_n": self._safe_int(top_n or screener_result.get("top_n")),
            "period": period or screener_result.get("period"),
            "result_json": self._to_json(screener_result),
            "created_at_utc": self._now_utc(),
        }
        self.backend.insert("screener_runs", row)
        return {"success": True, "id": row["id"]}

    def record_pipeline_bundle(
        self,
        symbol: str,
        multi_quote: Optional[Dict[str, Any]] = None,
        historical_data: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
        analysis_result: Optional[Dict[str, Any]] = None,
        training_result: Optional[Dict[str, Any]] = None,
        signal_result: Optional[Dict[str, Any]] = None,
        risk_result: Optional[Dict[str, Any]] = None,
        strategy_result: Optional[Dict[str, Any]] = None,
        reward_record_result: Optional[Dict[str, Any]] = None,
        auto_reward_update_result: Optional[Dict[str, Any]] = None,
        llm_report_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run = self.create_pipeline_run(
            symbol=symbol,
            validation_result=validation_result,
            analysis_result=analysis_result,
            training_result=training_result,
            signal_result=signal_result,
            risk_result=risk_result,
            strategy_result=strategy_result,
            reward_record_result=reward_record_result,
        )
        run_id = run["run_id"]
        symbol = self._normalise_symbol(symbol)

        if multi_quote:
            self.record_market_quotes(multi_quote, run_id=run_id, symbol=symbol)
        for name, output in [
            ("Data Agent", multi_quote),
            ("Historical Data Agent", historical_data),
            ("Validation Agent", validation_result),
            ("Analyst Agent", analysis_result),
            ("Training Agent", training_result),
            ("Signal Model", signal_result),
            ("Risk Agent", risk_result),
            ("Strategist Agent", strategy_result),
            ("Reward Agent", reward_record_result),
            ("Reward Update Agent", auto_reward_update_result),
            ("LLM Report Agent", llm_report_result),
        ]:
            if isinstance(output, dict):
                self.record_agent_output(run_id, symbol, name, output)
        if reward_record_result:
            self.record_paper_decision(reward_record_result, run_id=run_id)
        if auto_reward_update_result:
            self.record_reward_updates(auto_reward_update_result)
        if training_result:
            self.record_training_run(run_id, symbol, training_result)
        if llm_report_result:
            self.record_llm_report(run_id, symbol, llm_report_result)
        return {"success": True, "run_id": run_id, "summary": f"Saved pipeline run for {symbol}."}

    # ------------------------------------------------------------------
    # Read methods for dashboards/evaluator
    # ------------------------------------------------------------------
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
