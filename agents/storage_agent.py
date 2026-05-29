from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class StorageAgent:
    """
    SQLite-based persistent memory layer for the multi-agent trading system.

    Role:
    - Store structured outputs from each agent.
    - Store pipeline-level summaries.
    - Store paper decisions and reward updates.
    - Store LLM reports, screener runs, training runs, and market quote snapshots.

    Design:
    - Uses Python's built-in sqlite3, so no extra database server is required.
    - Stores large nested objects as JSON text for auditability.
    - Does not store API keys or real brokerage credentials.
    """

    def __init__(self, db_path: str = "data/trading_system.db", auto_init: bool = True):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if auto_init:
            self.init_db()

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _now_utc(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _new_id(self, prefix: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{prefix}_{stamp}_{uuid.uuid4().hex[:8]}"

    def _to_json(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return json.dumps({"serialization_error": str(value)}, ensure_ascii=False)

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
            return int(value)
        except Exception:
            return None

    def _get_nested(self, data: Dict[str, Any], keys: List[str], default=None):
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _execute(self, sql: str, params: Iterable[Any] = ()):
        with self._connect() as conn:
            conn.execute(sql, tuple(params))

    def _query(self, sql: str, params: Iterable[Any] = ()) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def init_db(self) -> Dict[str, Any]:
        """Create tables if they do not already exist."""
        schema = [
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
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                symbol TEXT,
                agent_name TEXT,
                output_json TEXT,
                created_at_utc TEXT,
                FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS market_quotes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                raw_json TEXT,
                FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
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
                raw_json TEXT,
                FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS reward_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id TEXT,
                symbol TEXT,
                entry_price REAL,
                latest_close REAL,
                future_return REAL,
                reward REAL,
                reward_horizon_days INTEGER,
                updated_at_utc TEXT,
                raw_json TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS training_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                symbol TEXT,
                model_type TEXT,
                best_params_json TEXT,
                accuracy REAL,
                balanced_accuracy REAL,
                macro_f1 REAL,
                training_samples INTEGER,
                model_path TEXT,
                metadata_path TEXT,
                created_at_utc TEXT,
                raw_json TEXT,
                FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS screener_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                symbol TEXT,
                report_type TEXT,
                provider TEXT,
                model TEXT,
                source TEXT,
                llm_available INTEGER,
                plain_language_report TEXT,
                created_at_utc TEXT,
                raw_json TEXT,
                FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
            )
            """,
        ]

        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_pipeline_symbol_time ON pipeline_runs(symbol, created_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_agent_outputs_run ON agent_outputs(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_market_quotes_symbol_time ON market_quotes(symbol, created_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_paper_status ON paper_decisions(status)",
            "CREATE INDEX IF NOT EXISTS idx_reward_decision ON reward_updates(decision_id)",
            "CREATE INDEX IF NOT EXISTS idx_llm_reports_run ON llm_reports(run_id)",
        ]

        with self._connect() as conn:
            for statement in schema:
                conn.execute(statement)
            for statement in indexes:
                conn.execute(statement)

        return {
            "success": True,
            "db_path": str(self.db_path),
            "summary": "SQLite database is ready."
        }

    # ------------------------------------------------------------------
    # Record helpers
    # ------------------------------------------------------------------
    def create_pipeline_run(
        self,
        symbol: str,
        validation_result: Optional[Dict[str, Any]] = None,
        analysis_result: Optional[Dict[str, Any]] = None,
        signal_result: Optional[Dict[str, Any]] = None,
        risk_result: Optional[Dict[str, Any]] = None,
        strategy_result: Optional[Dict[str, Any]] = None,
        reward_record_result: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
    ) -> str:
        validation_result = validation_result or {}
        analysis_result = analysis_result or {}
        signal_result = signal_result or {}
        risk_result = risk_result or {}
        strategy_result = strategy_result or {}
        reward_record_result = reward_record_result or {}

        run_id = run_id or self._new_id("run")
        symbol = (symbol or "UNKNOWN").upper().strip()
        selected_price = self._safe_float(validation_result.get("selected_price"))

        values = (
            run_id,
            symbol,
            selected_price,
            validation_result.get("confidence"),
            validation_result.get("next_action"),
            analysis_result.get("analyst_signal") or analysis_result.get("display_signal"),
            signal_result.get("model_signal") or signal_result.get("signal"),
            signal_result.get("confidence_level") or signal_result.get("model_confidence_level"),
            risk_result.get("final_signal"),
            risk_result.get("risk_level"),
            risk_result.get("risk_action"),
            strategy_result.get("strategy_action"),
            strategy_result.get("strategy_level"),
            reward_record_result.get("decision_id"),
            self._now_utc(),
        )

        self._execute(
            """
            INSERT OR REPLACE INTO pipeline_runs (
                run_id, symbol, selected_price, validation_confidence,
                validation_next_action, analyst_signal, model_signal,
                model_confidence, final_signal, risk_level, risk_action,
                strategy_action, strategy_level, reward_decision_id, created_at_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            values,
        )

        return run_id

    def record_agent_output(
        self,
        run_id: str,
        symbol: str,
        agent_name: str,
        output: Dict[str, Any],
    ) -> Dict[str, Any]:
        self._execute(
            """
            INSERT INTO agent_outputs (run_id, symbol, agent_name, output_json, created_at_utc)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                run_id,
                (symbol or "UNKNOWN").upper().strip(),
                agent_name,
                self._to_json(output or {}),
                self._now_utc(),
            ),
        )

        return {"success": True, "summary": f"Stored {agent_name} output."}

    def record_market_quotes(self, run_id: str, symbol: str, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        multi_quote = multi_quote or {}
        symbol = (symbol or multi_quote.get("symbol") or "UNKNOWN").upper().strip()
        inserted = 0

        source_map = {
            "finnhub": multi_quote.get("finnhub") or multi_quote.get("finnhub_quote"),
            "alpha_vantage": multi_quote.get("alpha_vantage") or multi_quote.get("alpha_vantage_quote"),
            "primary": multi_quote.get("primary"),
            "secondary": multi_quote.get("secondary"),
        }

        seen = set()
        for source_name, quote in source_map.items():
            if not isinstance(quote, dict) or id(quote) in seen:
                continue
            seen.add(id(quote))
            if not quote:
                continue

            self._execute(
                """
                INSERT INTO market_quotes (
                    run_id, symbol, source, current_price, open_price,
                    high_price, low_price, previous_close, quote_timestamp,
                    created_at_utc, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    symbol,
                    quote.get("source") or source_name,
                    self._safe_float(quote.get("current_price") or quote.get("price") or quote.get("c")),
                    self._safe_float(quote.get("open_price") or quote.get("open") or quote.get("o")),
                    self._safe_float(quote.get("high_price") or quote.get("high") or quote.get("h")),
                    self._safe_float(quote.get("low_price") or quote.get("low") or quote.get("l")),
                    self._safe_float(quote.get("previous_close_price") or quote.get("previous_close") or quote.get("pc")),
                    str(quote.get("timestamp") or quote.get("latest_trading_day") or ""),
                    self._now_utc(),
                    self._to_json(quote),
                ),
            )
            inserted += 1

        return {"success": True, "inserted": inserted, "summary": f"Stored {inserted} market quote snapshots."}

    def record_paper_decision(
        self,
        run_id: str,
        symbol: str,
        reward_record_result: Dict[str, Any],
        risk_result: Optional[Dict[str, Any]] = None,
        strategy_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        reward_record_result = reward_record_result or {}
        risk_result = risk_result or {}
        strategy_result = strategy_result or {}

        if not reward_record_result.get("success"):
            return {"success": False, "summary": "No successful paper decision to store."}

        decision_id = reward_record_result.get("decision_id") or self._new_id("decision")
        symbol = (symbol or reward_record_result.get("symbol") or "UNKNOWN").upper().strip()

        self._execute(
            """
            INSERT OR REPLACE INTO paper_decisions (
                decision_id, run_id, symbol, entry_price, final_signal,
                risk_action, risk_level, strategy_action, status,
                created_at_utc, updated_at_utc, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision_id,
                run_id,
                symbol,
                self._safe_float(reward_record_result.get("entry_price")),
                reward_record_result.get("final_signal") or risk_result.get("final_signal"),
                reward_record_result.get("risk_action") or risk_result.get("risk_action"),
                risk_result.get("risk_level"),
                strategy_result.get("strategy_action"),
                "pending",
                self._now_utc(),
                self._now_utc(),
                self._to_json(reward_record_result),
            ),
        )

        return {"success": True, "decision_id": decision_id, "summary": f"Stored paper decision {decision_id}."}

    def record_reward_updates(self, auto_reward_update_result: Dict[str, Any]) -> Dict[str, Any]:
        auto_reward_update_result = auto_reward_update_result or {}
        updates = auto_reward_update_result.get("updates", []) or []
        inserted = 0

        for item in updates:
            if not isinstance(item, dict) or not item.get("updated"):
                continue
            self._execute(
                """
                INSERT INTO reward_updates (
                    decision_id, symbol, entry_price, latest_close,
                    future_return, reward, reward_horizon_days, updated_at_utc, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.get("decision_id"),
                    (item.get("symbol") or "UNKNOWN").upper(),
                    self._safe_float(item.get("entry_price")),
                    self._safe_float(item.get("latest_close")),
                    self._safe_float(item.get("future_return")),
                    self._safe_float(item.get("reward")),
                    None,
                    self._now_utc(),
                    self._to_json(item),
                ),
            )
            inserted += 1

        return {"success": True, "inserted": inserted, "summary": f"Stored {inserted} reward updates."}

    def record_training_run(self, run_id: str, symbol: str, training_result: Dict[str, Any]) -> Dict[str, Any]:
        training_result = training_result or {}
        metrics = training_result.get("metrics", {}) if isinstance(training_result.get("metrics"), dict) else {}
        best_params = training_result.get("best_params") or training_result.get("model_params") or {}

        self._execute(
            """
            INSERT INTO training_runs (
                run_id, symbol, model_type, best_params_json, accuracy,
                balanced_accuracy, macro_f1, training_samples, model_path,
                metadata_path, created_at_utc, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                (symbol or "UNKNOWN").upper(),
                training_result.get("model_type") or training_result.get("model_name") or "unknown",
                self._to_json(best_params),
                self._safe_float(training_result.get("accuracy") or metrics.get("accuracy") or training_result.get("test_accuracy")),
                self._safe_float(training_result.get("balanced_accuracy") or metrics.get("balanced_accuracy")),
                self._safe_float(training_result.get("macro_f1") or metrics.get("macro_f1")),
                self._safe_int(training_result.get("training_samples") or training_result.get("num_samples")),
                training_result.get("model_path") or training_result.get("saved_model_path"),
                training_result.get("metadata_path"),
                self._now_utc(),
                self._to_json(training_result),
            ),
        )
        return {"success": True, "summary": "Stored training run metadata."}

    def record_llm_report(self, run_id: str, symbol: str, report_result: Dict[str, Any]) -> Dict[str, Any]:
        report_result = report_result or {}
        self._execute(
            """
            INSERT INTO llm_reports (
                run_id, symbol, report_type, provider, model, source,
                llm_available, plain_language_report, created_at_utc, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                (symbol or report_result.get("symbol") or "UNKNOWN").upper(),
                report_result.get("report_type"),
                report_result.get("provider"),
                report_result.get("model"),
                report_result.get("source"),
                1 if report_result.get("llm_available") else 0,
                report_result.get("plain_language_report"),
                self._now_utc(),
                self._to_json(report_result),
            ),
        )
        return {"success": True, "summary": "Stored LLM report."}

    def record_screener_run(
        self,
        screener_result: Dict[str, Any],
        run_id: Optional[str] = None,
        top_n: Optional[int] = None,
        period: Optional[str] = None,
    ) -> Dict[str, Any]:
        screener_result = screener_result or {}
        run_id = run_id or self._new_id("screener")
        self._execute(
            """
            INSERT INTO screener_runs (run_id, universe_size, top_n, period, result_json, created_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                self._safe_int(screener_result.get("universe_size")),
                top_n,
                period,
                self._to_json(screener_result),
                self._now_utc(),
            ),
        )
        return {"success": True, "run_id": run_id, "summary": "Stored screener run."}

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
        """Save one full single-stock pipeline run with all structured outputs."""
        run_id = self.create_pipeline_run(
            symbol=symbol,
            validation_result=validation_result,
            analysis_result=analysis_result,
            signal_result=signal_result,
            risk_result=risk_result,
            strategy_result=strategy_result,
            reward_record_result=reward_record_result,
        )

        output_map = {
            "Data Agent": multi_quote or {},
            "Historical Data Agent": historical_data or {},
            "Validation Agent": validation_result or {},
            "Analyst Agent": analysis_result or {},
            "Training Agent": training_result or {},
            "Signal Model": signal_result or {},
            "Risk Agent": risk_result or {},
            "Strategist Agent": strategy_result or {},
            "Reward Agent": reward_record_result or {},
            "Auto Reward Update": auto_reward_update_result or {},
            "Groq Report Agent": llm_report_result or {},
        }

        stored_agents = 0
        for agent_name, output in output_map.items():
            if output:
                self.record_agent_output(run_id, symbol, agent_name, output)
                stored_agents += 1

        quote_result = self.record_market_quotes(run_id, symbol, multi_quote or {})
        paper_result = self.record_paper_decision(
            run_id,
            symbol,
            reward_record_result or {},
            risk_result=risk_result,
            strategy_result=strategy_result,
        )
        reward_updates_result = self.record_reward_updates(auto_reward_update_result or {})
        training_store_result = self.record_training_run(run_id, symbol, training_result or {})

        llm_store_result = {"success": False, "summary": "No LLM report provided."}
        if llm_report_result:
            llm_store_result = self.record_llm_report(run_id, symbol, llm_report_result)

        return {
            "success": True,
            "run_id": run_id,
            "db_path": str(self.db_path),
            "stored_agent_outputs": stored_agents,
            "market_quotes": quote_result,
            "paper_decision": paper_result,
            "reward_updates": reward_updates_result,
            "training_run": training_store_result,
            "llm_report": llm_store_result,
            "summary": f"Stored pipeline run {run_id} in SQLite memory."
        }

    # ------------------------------------------------------------------
    # Read helpers for dashboards / evaluator
    # ------------------------------------------------------------------
    def get_recent_pipeline_runs(self, limit: int = 10) -> List[Dict[str, Any]]:
        return self._query(
            """
            SELECT * FROM pipeline_runs
            ORDER BY created_at_utc DESC
            LIMIT ?
            """,
            (limit,),
        )

    def get_agent_outputs_for_run(self, run_id: str) -> List[Dict[str, Any]]:
        return self._query(
            """
            SELECT * FROM agent_outputs
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        )

    def get_storage_summary(self) -> Dict[str, Any]:
        tables = [
            "pipeline_runs",
            "agent_outputs",
            "market_quotes",
            "paper_decisions",
            "reward_updates",
            "training_runs",
            "screener_runs",
            "llm_reports",
        ]
        summary = {"db_path": str(self.db_path), "tables": {}}
        with self._connect() as conn:
            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                summary["tables"][table] = count
        summary["success"] = True
        summary["summary"] = "Storage summary loaded."
        return summary
