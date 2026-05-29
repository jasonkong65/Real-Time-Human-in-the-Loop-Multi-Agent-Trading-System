from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


class StorageAgent:
    """
    SQLite persistent memory layer for the multi-agent stock decision-support system.

    Main role:
    - Store full pipeline runs and each agent's structured JSON output.
    - Store market quote snapshots, paper decisions, reward updates, training metadata,
      screener runs, and LLM reports.
    - Provide read methods for Evaluator Agent and Streamlit dashboards.

    Design:
    - Uses Python's built-in sqlite3. No database server is required.
    - Stores nested agent outputs as JSON text for auditability.
    - Does not store API keys, broker credentials, or personal financial information.
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
            try:
                return json.dumps({"serialization_error": repr(value)}, ensure_ascii=False)
            except Exception:
                return "{}"

    def _from_json(self, value: Any, default=None):
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return default

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

    def _get_nested(self, data: Dict[str, Any], keys: List[str], default=None):
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current

    def _normalise_symbol(self, symbol: Any) -> str:
        return str(symbol or "UNKNOWN").upper().strip()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
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

    def _table_columns(self, table: str) -> List[str]:
        try:
            with self._connect() as conn:
                rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
            return [row["name"] for row in rows]
        except Exception:
            return []

    def _add_column_if_missing(self, table: str, column: str, column_type: str):
        columns = self._table_columns(table)
        if column not in columns:
            with self._connect() as conn:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------
    def init_db(self) -> Dict[str, Any]:
        """Create and migrate all tables."""
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
                horizon_label TEXT,
                status TEXT,
                updated_at_utc TEXT,
                raw_json TEXT,
                FOREIGN KEY(decision_id) REFERENCES paper_decisions(decision_id)
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
            "CREATE INDEX IF NOT EXISTS idx_agent_outputs_agent ON agent_outputs(agent_name, created_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_market_quotes_symbol_time ON market_quotes(symbol, created_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_paper_status ON paper_decisions(status)",
            "CREATE INDEX IF NOT EXISTS idx_paper_symbol_time ON paper_decisions(symbol, created_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_reward_decision ON reward_updates(decision_id)",
            "CREATE INDEX IF NOT EXISTS idx_reward_symbol_time ON reward_updates(symbol, updated_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_screener_time ON screener_runs(created_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_llm_reports_run ON llm_reports(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_risk_dqn_replay_time ON risk_dqn_replay(created_at_utc)",
            "CREATE INDEX IF NOT EXISTS idx_risk_dqn_replay_action ON risk_dqn_replay(action)",

        ]

        with self._connect() as conn:
            for statement in schema:
                conn.execute(statement)
            for statement in indexes:
                conn.execute(statement)

        # Safe migrations for users with an older db.
        self._add_column_if_missing("reward_updates", "horizon_label", "TEXT")
        self._add_column_if_missing("reward_updates", "status", "TEXT")

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
        symbol = self._normalise_symbol(symbol)

        selected_price = (
            self._safe_float(validation_result.get("selected_price"))
            or self._safe_float(self._get_nested(validation_result, ["validation_for_next_agent", "selected_price"]))
        )

        analyst_signal = (
            analysis_result.get("analyst_signal")
            or analysis_result.get("display_signal")
            or self._get_nested(analysis_result, ["analysis_for_next_agent", "analyst_signal"])
        )

        model_signal = (
            signal_result.get("model_signal")
            or signal_result.get("signal")
            or signal_result.get("display_signal")
            or self._get_nested(signal_result, ["signal_for_next_agent", "signal"])
        )

        model_confidence = (
            signal_result.get("confidence_level")
            or signal_result.get("model_confidence_level")
            or self._get_nested(signal_result, ["signal_for_next_agent", "confidence_level"])
        )

        values = (
            run_id,
            symbol,
            selected_price,
            validation_result.get("confidence"),
            validation_result.get("next_action"),
            analyst_signal,
            model_signal,
            model_confidence,
            risk_result.get("final_signal") or self._get_nested(risk_result, ["risk_for_next_agent", "final_signal"]),
            risk_result.get("risk_level") or self._get_nested(risk_result, ["risk_for_next_agent", "risk_level"]),
            risk_result.get("risk_action") or self._get_nested(risk_result, ["risk_for_next_agent", "risk_action"]),
            strategy_result.get("strategy_action") or self._get_nested(strategy_result, ["strategy_for_next_agent", "strategy_action"]),
            strategy_result.get("strategy_level") or self._get_nested(strategy_result, ["strategy_for_next_agent", "strategy_level"]),
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
                self._normalise_symbol(symbol),
                agent_name,
                self._to_json(output or {}),
                self._now_utc(),
            ),
        )
        return {"success": True, "summary": f"Stored {agent_name} output."}

    def record_market_quotes(
        self,
        run_id: Optional[str] = None,
        symbol: Optional[str] = None,
        multi_quote: Optional[Dict[str, Any]] = None,
        quote_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store market quote snapshots.

        Compatible call styles:
        - record_market_quotes(run_id, symbol, multi_quote)
        - record_market_quotes(symbol="AAPL", quote_result=quote)
        - record_market_quotes(multi_quote=multi_quote)
        """
        multi_quote = multi_quote or quote_result or {}
        symbol = self._normalise_symbol(symbol or multi_quote.get("symbol"))
        inserted = 0

        if not isinstance(multi_quote, dict):
            return {"success": False, "inserted": 0, "summary": "Invalid quote payload."}

        source_map = {
            "finnhub": multi_quote.get("finnhub") or multi_quote.get("finnhub_quote"),
            "alpha_vantage": multi_quote.get("alpha_vantage") or multi_quote.get("alpha_vantage_quote"),
            "primary": multi_quote.get("primary"),
            "secondary": multi_quote.get("secondary"),
        }

        # If a single quote was passed directly.
        if any(k in multi_quote for k in ["current_price", "price", "c"]):
            source_map = {multi_quote.get("source") or "quote": multi_quote}

        seen_signatures = set()
        for source_name, quote in source_map.items():
            if not isinstance(quote, dict) or not quote:
                continue

            quote_source = str(quote.get("source") or source_name)
            quote_price = self._safe_float(quote.get("current_price") or quote.get("price") or quote.get("c"))
            signature = (quote_source, quote_price, str(quote.get("timestamp") or quote.get("latest_trading_day") or ""))

            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

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
                    quote_source,
                    quote_price,
                    self._safe_float(quote.get("open_price") or quote.get("open") or quote.get("o")),
                    self._safe_float(quote.get("high_price") or quote.get("high") or quote.get("h")),
                    self._safe_float(quote.get("low_price") or quote.get("low") or quote.get("l")),
                    self._safe_float(quote.get("previous_close_price") or quote.get("previous_close") or quote.get("pc")),
                    str(quote.get("timestamp") or quote.get("latest_trading_day") or quote.get("quote_timestamp") or ""),
                    self._now_utc(),
                    self._to_json(quote),
                ),
            )
            inserted += 1

        return {
            "success": True,
            "inserted": inserted,
            "summary": f"Stored {inserted} market quote snapshots."
        }

    def record_paper_decision(
        self,
        run_id: Optional[str],
        symbol: str,
        reward_record_result: Dict[str, Any],
        risk_result: Optional[Dict[str, Any]] = None,
        strategy_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        reward_record_result = reward_record_result or {}
        risk_result = risk_result or {}
        strategy_result = strategy_result or {}

        if not reward_record_result.get("success") and not reward_record_result.get("decision_id"):
            return {"success": False, "summary": "No successful paper decision to store."}

        decision_id = reward_record_result.get("decision_id") or self._new_id("decision")
        symbol = self._normalise_symbol(symbol or reward_record_result.get("symbol"))

        status = (
            reward_record_result.get("paper_status")
            or reward_record_result.get("status")
            or "PAPER_PENDING"
        )

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
                self._safe_float(reward_record_result.get("entry_price") or reward_record_result.get("selected_price")),
                reward_record_result.get("final_signal") or risk_result.get("final_signal"),
                reward_record_result.get("risk_action") or risk_result.get("risk_action"),
                reward_record_result.get("risk_level") or risk_result.get("risk_level"),
                reward_record_result.get("strategy_action") or strategy_result.get("strategy_action"),
                status,
                reward_record_result.get("created_at_utc") or self._now_utc(),
                self._now_utc(),
                self._to_json(reward_record_result),
            ),
        )

        return {
            "success": True,
            "decision_id": decision_id,
            "summary": f"Stored paper decision {decision_id}."
        }

    def record_reward_updates(self, auto_reward_update_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store delayed reward updates.

        Compatible with both older updates list and newer multi-horizon update formats.
        """
        auto_reward_update_result = auto_reward_update_result or {}
        updates = auto_reward_update_result.get("updates", []) or auto_reward_update_result.get("reward_updates", []) or []
        inserted = 0

        for item in updates:
            if not isinstance(item, dict):
                continue

            updated = item.get("updated", True)
            status = item.get("status") or ("COMPLETED" if updated else "PENDING")

            if status.upper() in ["PENDING", "NOT_DUE"] and not item.get("future_return"):
                continue

            decision_id = item.get("decision_id") or item.get("paper_decision_id")
            symbol = self._normalise_symbol(item.get("symbol"))

            self._execute(
                """
                INSERT INTO reward_updates (
                    decision_id, symbol, entry_price, latest_close,
                    future_return, reward, reward_horizon_days,
                    horizon_label, status, updated_at_utc, raw_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    symbol,
                    self._safe_float(item.get("entry_price")),
                    self._safe_float(item.get("latest_close") or item.get("close_price")),
                    self._safe_float(item.get("future_return")),
                    self._safe_float(item.get("reward")),
                    self._safe_int(item.get("reward_horizon_days") or item.get("horizon_days")),
                    item.get("horizon_label") or item.get("reward_horizon") or item.get("horizon"),
                    status,
                    item.get("updated_at_utc") or self._now_utc(),
                    self._to_json(item),
                ),
            )
            inserted += 1

            # Keep the parent paper decision status updated if possible.
            if decision_id:
                self._execute(
                    """
                    UPDATE paper_decisions
                    SET updated_at_utc = ?, status =
                        CASE
                            WHEN status LIKE 'COMPLETED%' THEN status
                            ELSE 'PARTIALLY_COMPLETED'
                        END
                    WHERE decision_id = ?
                    """,
                    (self._now_utc(), decision_id),
                )

        return {
            "success": True,
            "inserted": inserted,
            "summary": f"Stored {inserted} reward updates."
        }

    def record_training_run(self, run_id: Optional[str], symbol: str, training_result: Dict[str, Any]) -> Dict[str, Any]:
        training_result = training_result or {}
        metrics = training_result.get("metrics", {}) if isinstance(training_result.get("metrics"), dict) else {}
        best_params = training_result.get("best_params") or training_result.get("model_params") or training_result.get("best_model_params") or {}

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
                self._normalise_symbol(symbol),
                training_result.get("model_type") or training_result.get("model_name") or training_result.get("best_model_name") or "unknown",
                self._to_json(best_params),
                self._safe_float(training_result.get("accuracy") or metrics.get("accuracy") or training_result.get("test_accuracy")),
                self._safe_float(training_result.get("balanced_accuracy") or metrics.get("balanced_accuracy")),
                self._safe_float(training_result.get("macro_f1") or metrics.get("macro_f1")),
                self._safe_int(training_result.get("training_samples") or training_result.get("num_samples") or training_result.get("sample_count")),
                training_result.get("model_path") or training_result.get("saved_model_path"),
                training_result.get("metadata_path"),
                self._now_utc(),
                self._to_json(training_result),
            ),
        )
        return {"success": True, "summary": "Stored training run metadata."}

    def record_llm_report(self, run_id: Optional[str], symbol: str, report_result: Dict[str, Any]) -> Dict[str, Any]:
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
                self._normalise_symbol(symbol or report_result.get("symbol")),
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
        run_id = run_id or screener_result.get("run_id") or self._new_id("screener")

        inferred_top_n = top_n
        if inferred_top_n is None:
            inferred_top_n = (
                len(screener_result.get("top_buy_candidates", []) or [])
                or len(screener_result.get("top_candidates", []) or [])
                or None
            )

        inferred_period = period or screener_result.get("period") or screener_result.get("lookback_period")

        self._execute(
            """
            INSERT INTO screener_runs (run_id, universe_size, top_n, period, result_json, created_at_utc)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                self._safe_int(screener_result.get("universe_size") or screener_result.get("scanned_count")),
                self._safe_int(inferred_top_n),
                inferred_period,
                self._to_json(screener_result),
                self._now_utc(),
            ),
        )

        return {
            "success": True,
            "run_id": run_id,
            "summary": "Stored screener run."
        }

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

        quote_result = self.record_market_quotes(run_id=run_id, symbol=symbol, multi_quote=multi_quote or {})
        paper_result = self.record_paper_decision(
            run_id=run_id,
            symbol=symbol,
            reward_record_result=reward_record_result or {},
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

    def get_agent_outputs_for_run(self, run_id: str, parse_json: bool = False) -> List[Dict[str, Any]]:
        rows = self._query(
            """
            SELECT * FROM agent_outputs
            WHERE run_id = ?
            ORDER BY id ASC
            """,
            (run_id,),
        )

        if parse_json:
            for row in rows:
                row["output"] = self._from_json(row.get("output_json"), default={})
        return rows

    def get_recent_agent_outputs(
        self,
        limit: int = 50,
        agent_name: Optional[str] = None,
        symbol: Optional[str] = None,
        parse_json: bool = False,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []

        if agent_name:
            clauses.append("agent_name = ?")
            params.append(agent_name)

        if symbol:
            clauses.append("symbol = ?")
            params.append(self._normalise_symbol(symbol))

        where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""

        rows = self._query(
            f"""
            SELECT * FROM agent_outputs
            {where_sql}
            ORDER BY created_at_utc DESC, id DESC
            LIMIT ?
            """,
            (*params, limit),
        )

        if parse_json:
            for row in rows:
                row["output"] = self._from_json(row.get("output_json"), default={})
        return rows

    def get_latest_agent_output(
        self,
        agent_name: str,
        symbol: Optional[str] = None,
        parse_json: bool = True,
    ) -> Optional[Dict[str, Any]]:
        rows = self.get_recent_agent_outputs(
            limit=1,
            agent_name=agent_name,
            symbol=symbol,
            parse_json=parse_json,
        )
        return rows[0] if rows else None

    def get_recent_screener_runs(self, limit: int = 10, parse_json: bool = False) -> List[Dict[str, Any]]:
        rows = self._query(
            """
            SELECT * FROM screener_runs
            ORDER BY created_at_utc DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        )

        if parse_json:
            for row in rows:
                row["result"] = self._from_json(row.get("result_json"), default={})
        return rows

    def get_paper_decisions(
        self,
        status: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []

        if status:
            clauses.append("status = ?")
            params.append(status)

        if symbol:
            clauses.append("symbol = ?")
            params.append(self._normalise_symbol(symbol))

        where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""

        return self._query(
            f"""
            SELECT * FROM paper_decisions
            {where_sql}
            ORDER BY created_at_utc DESC
            LIMIT ?
            """,
            (*params, limit),
        )

    def get_reward_updates(
        self,
        symbol: Optional[str] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: List[Any] = []

        if symbol:
            clauses.append("ru.symbol = ?")
            params.append(self._normalise_symbol(symbol))

        where_sql = "WHERE " + " AND ".join(clauses) if clauses else ""

        return self._query(
            f"""
            SELECT
                ru.*,
                pd.final_signal,
                pd.risk_action,
                pd.risk_level,
                pd.strategy_action,
                pd.status AS decision_status
            FROM reward_updates ru
            LEFT JOIN paper_decisions pd ON ru.decision_id = pd.decision_id
            {where_sql}
            ORDER BY ru.updated_at_utc DESC, ru.id DESC
            LIMIT ?
            """,
            (*params, limit),
        )

    def _directional_win(self, row: Dict[str, Any]) -> Optional[int]:
        future_return = self._safe_float(row.get("future_return"))
        final_signal = str(row.get("final_signal") or "").upper()

        if future_return is None:
            return None

        if final_signal in ["BUY_CANDIDATE", "BUY_WATCHLIST_OVERBOUGHT"]:
            return 1 if future_return > 0 else 0

        if final_signal == "SELL_RISK":
            return 1 if future_return < 0 else 0

        if final_signal == "HOLD":
            return 1 if abs(future_return) <= 0.02 else 0

        if final_signal == "BLOCKED":
            return 1 if future_return < 0 else 0

        return None

    def get_reward_summary(self) -> Dict[str, Any]:
        rows = self.get_reward_updates(limit=100000)
        rewards = [self._safe_float(r.get("reward")) for r in rows]
        returns = [self._safe_float(r.get("future_return")) for r in rows]
        rewards = [r for r in rewards if r is not None]
        returns = [r for r in returns if r is not None]

        reward_wins = [1 if r > 0 else 0 for r in rewards]
        directional = [self._directional_win(row) for row in rows]
        directional = [v for v in directional if v is not None]

        open_decisions = self._query(
            """
            SELECT COUNT(*) AS n FROM paper_decisions
            WHERE status NOT LIKE 'COMPLETED%'
            """
        )[0]["n"]

        return {
            "success": True,
            "completed_rewards": len(rewards),
            "open_paper_decisions": open_decisions,
            "average_reward": sum(rewards) / len(rewards) if rewards else None,
            "average_future_return": sum(returns) / len(returns) if returns else None,
            "reward_win_rate": sum(reward_wins) / len(reward_wins) if reward_wins else None,
            "directional_win_rate": sum(directional) / len(directional) if directional else None,
        }

    def _group_reward_stats(self, group_field: str) -> List[Dict[str, Any]]:
        allowed = {
            "strategy_action": "pd.strategy_action",
            "final_signal": "pd.final_signal",
            "risk_action": "pd.risk_action",
            "risk_level": "pd.risk_level",
            "symbol": "ru.symbol",
            "horizon_label": "ru.horizon_label",
        }

        if group_field not in allowed:
            raise ValueError(f"Unsupported group field: {group_field}")

        col = allowed[group_field]

        return self._query(
            f"""
            SELECT
                {col} AS group_name,
                COUNT(*) AS count,
                AVG(ru.reward) AS avg_reward,
                AVG(ru.future_return) AS avg_future_return,
                SUM(CASE WHEN ru.reward > 0 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS reward_win_rate
            FROM reward_updates ru
            LEFT JOIN paper_decisions pd ON ru.decision_id = pd.decision_id
            WHERE ru.reward IS NOT NULL
            GROUP BY {col}
            ORDER BY count DESC, avg_reward DESC
            """
        )

    def get_reward_by_strategy_action(self) -> List[Dict[str, Any]]:
        return self._group_reward_stats("strategy_action")

    def get_reward_by_signal_type(self) -> List[Dict[str, Any]]:
        return self._group_reward_stats("final_signal")

    def get_reward_by_horizon(self) -> List[Dict[str, Any]]:
        return self._group_reward_stats("horizon_label")

    def record_dqn_replay_transition(self, transition: Dict[str, Any]) -> Dict[str, Any]:
        """Store one DQN replay transition in SQLite."""
        transition = transition or {}
        try:
            self._execute(
                """
                INSERT INTO risk_dqn_replay (
                    created_at_utc, state_text, state_vector_json,
                    action, action_index, reward, next_state_text,
                    next_state_vector_json, done, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transition.get("created_at_utc") or self._now_utc(),
                    transition.get("state_text"),
                    transition.get("state_vector_json") or self._to_json(transition.get("state_vector")),
                    transition.get("action"),
                    self._safe_int(transition.get("action_index")),
                    self._safe_float(transition.get("reward")),
                    transition.get("next_state_text"),
                    transition.get("next_state_vector_json") or self._to_json(transition.get("next_state_vector")),
                    1 if transition.get("done", True) else 0,
                    transition.get("source") or "risk_agent",
                ),
            )
            return {"success": True, "summary": "Stored DQN replay transition."}
        except Exception as exc:
            return {"success": False, "error": str(exc), "summary": "Could not store DQN replay transition."}

    def get_dqn_replay_memory(self, limit: int = 10000) -> List[Dict[str, Any]]:
        return self._query(
            """
            SELECT * FROM risk_dqn_replay
            ORDER BY id DESC
            LIMIT ?
            """,
            (int(limit),),
        )

    def get_evaluator_dataset(self, limit: int = 10000) -> Dict[str, Any]:
        """
        Main method for EvaluatorAgent.
        Returns data from SQLite in a single structured payload.
        """
        return {
            "success": True,
            "source": "sqlite",
            "db_path": str(self.db_path),
            "recent_pipeline_runs": self.get_recent_pipeline_runs(limit=50),
            "paper_decisions": self.get_paper_decisions(limit=limit),
            "reward_updates": self.get_reward_updates(limit=limit),
            "reward_summary": self.get_reward_summary(),
            "reward_by_strategy_action": self.get_reward_by_strategy_action(),
            "reward_by_signal_type": self.get_reward_by_signal_type(),
            "reward_by_horizon": self.get_reward_by_horizon(),
            "recent_screener_runs": self.get_recent_screener_runs(limit=20),
            "dqn_replay_memory": self.get_dqn_replay_memory(limit=limit),
            "storage_summary": self.get_storage_summary(),
        }

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
            "risk_dqn_replay",
        ]

        summary = {"db_path": str(self.db_path), "tables": {}}
        with self._connect() as conn:
            for table in tables:
                try:
                    count = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                except Exception:
                    count = 0
                summary["tables"][table] = count

        reward_summary = self.get_reward_summary()
        summary["reward_summary"] = {
            "completed_rewards": reward_summary.get("completed_rewards"),
            "open_paper_decisions": reward_summary.get("open_paper_decisions"),
            "average_reward": reward_summary.get("average_reward"),
            "reward_win_rate": reward_summary.get("reward_win_rate"),
        }
        summary["success"] = True
        summary["summary"] = "Storage summary loaded."
        return summary
