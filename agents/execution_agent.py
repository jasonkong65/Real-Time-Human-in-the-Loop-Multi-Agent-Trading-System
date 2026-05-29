from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class ExecutionAgent:
    """
    Interface Execution / Session Recording Agent.

    This is NOT a real trade execution agent.

    Role in this project:
    - Record what the user asked for in the Streamlit UI.
    - Record the selected portfolio/action context.
    - Record key outputs from the agents.
    - Record chart metadata and a compact chart-data sample.
    - Save a complete audit trail to SQLite and optional JSON artifacts.

    Why this exists:
    The system is a human-in-the-loop paper decision-support prototype.
    The Execution Agent records the paper research session; it does not place
    real orders and does not connect to a broker.
    """

    def __init__(
        self,
        db_path: str = "data/trading_system.db",
        artifact_dir: str = "data/ui_sessions",
        max_chart_rows_to_store: int = 500,
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.max_chart_rows_to_store = int(max_chart_rows_to_store)
        self.init_db()

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def _now(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _json_safe(self, obj: Any) -> Any:
        """Convert Python / pandas / numpy objects into JSON-safe values."""
        try:
            import numpy as np
            numpy_types = (np.integer, np.floating, np.bool_)
        except Exception:
            numpy_types = tuple()

        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        if numpy_types and isinstance(obj, numpy_types):
            try:
                return obj.item()
            except Exception:
                return str(obj)

        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, pd.DataFrame):
            return obj.tail(self.max_chart_rows_to_store).reset_index().to_dict("records")

        if isinstance(obj, pd.Series):
            return obj.to_dict()

        if isinstance(obj, dict):
            return {str(k): self._json_safe(v) for k, v in obj.items()}

        if isinstance(obj, (list, tuple, set)):
            return [self._json_safe(v) for v in obj]

        try:
            return json.loads(json.dumps(obj, default=str))
        except Exception:
            return str(obj)

    def _json_dumps(self, obj: Any) -> str:
        return json.dumps(self._json_safe(obj), ensure_ascii=False, default=str)

    def _normalise_symbol(self, symbol: str) -> str:
        return str(symbol or "").strip().upper()

    # ------------------------------------------------------------------
    # Database schema
    # ------------------------------------------------------------------
    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ui_sessions (
                    session_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    user_intent TEXT,
                    query_modes_json TEXT,
                    has_position INTEGER,
                    shares REAL,
                    average_cost REAL,
                    portfolio_context_json TEXT,
                    event_context_json TEXT,
                    chart_period TEXT,
                    chart_interval TEXT,
                    final_signal TEXT,
                    risk_level TEXT,
                    strategy_action TEXT,
                    strategy_level TEXT,
                    llm_source TEXT,
                    created_at_utc TEXT,
                    artifact_path TEXT,
                    summary_json TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ui_agent_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    symbol TEXT,
                    agent_name TEXT,
                    output_json TEXT,
                    created_at_utc TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ui_chart_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    symbol TEXT,
                    chart_period TEXT,
                    chart_interval TEXT,
                    chart_type TEXT,
                    chart_summary_json TEXT,
                    chart_sample_json TEXT,
                    created_at_utc TEXT
                )
                """
            )

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ui_artifacts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    artifact_type TEXT,
                    file_path TEXT,
                    metadata_json TEXT,
                    created_at_utc TEXT
                )
                """
            )

    # ------------------------------------------------------------------
    # Chart summarisation
    # ------------------------------------------------------------------
    def summarise_chart_data(
        self,
        chart_df: Optional[pd.DataFrame],
        symbol: str,
        period: str,
        interval: str,
    ) -> Dict[str, Any]:
        if chart_df is None or not isinstance(chart_df, pd.DataFrame) or chart_df.empty:
            return {
                "symbol": self._normalise_symbol(symbol),
                "period": period,
                "interval": interval,
                "available": False,
                "summary": "No chart data was available.",
            }

        df = chart_df.copy()
        lower_cols = {str(c).lower(): c for c in df.columns}

        close_col = lower_cols.get("close") or lower_cols.get("adj_close") or lower_cols.get("adj close")
        volume_col = lower_cols.get("volume")

        summary: Dict[str, Any] = {
            "symbol": self._normalise_symbol(symbol),
            "period": period,
            "interval": interval,
            "available": True,
            "rows": int(len(df)),
            "columns": [str(c) for c in df.columns],
        }

        try:
            if close_col:
                close = pd.to_numeric(df[close_col], errors="coerce").dropna()
                if not close.empty:
                    start_price = float(close.iloc[0])
                    end_price = float(close.iloc[-1])
                    price_change = end_price - start_price
                    pct_change = price_change / start_price if start_price else None
                    summary.update(
                        {
                            "start_price": start_price,
                            "end_price": end_price,
                            "price_change": price_change,
                            "pct_change": pct_change,
                            "min_close": float(close.min()),
                            "max_close": float(close.max()),
                        }
                    )
            if volume_col:
                volume = pd.to_numeric(df[volume_col], errors="coerce").dropna()
                if not volume.empty:
                    summary["avg_volume"] = float(volume.mean())
                    summary["latest_volume"] = float(volume.iloc[-1])
        except Exception as exc:
            summary["warning"] = f"Could not fully summarise chart data: {exc}"

        return summary

    def record_chart_snapshot(
        self,
        session_id: str,
        symbol: str,
        chart_df: Optional[pd.DataFrame],
        period: str,
        interval: str,
        chart_type: str = "price_chart",
    ) -> Dict[str, Any]:
        symbol = self._normalise_symbol(symbol)
        summary = self.summarise_chart_data(chart_df, symbol, period, interval)

        sample_json = "[]"
        if isinstance(chart_df, pd.DataFrame) and not chart_df.empty:
            sample = chart_df.tail(self.max_chart_rows_to_store).reset_index()
            sample_json = self._json_dumps(sample.to_dict("records"))

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ui_chart_records (
                    session_id, symbol, chart_period, chart_interval, chart_type,
                    chart_summary_json, chart_sample_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    symbol,
                    period,
                    interval,
                    chart_type,
                    self._json_dumps(summary),
                    sample_json,
                    self._now(),
                ),
            )

        return {"success": True, "chart_summary": summary}

    # ------------------------------------------------------------------
    # Agent output / UI session recording
    # ------------------------------------------------------------------
    def record_agent_outputs(
        self,
        session_id: str,
        symbol: str,
        agent_outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        symbol = self._normalise_symbol(symbol)
        count = 0

        with self._connect() as conn:
            for agent_name, output in (agent_outputs or {}).items():
                if output is None:
                    continue
                conn.execute(
                    """
                    INSERT INTO ui_agent_records (
                        session_id, symbol, agent_name, output_json, created_at_utc
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        session_id,
                        symbol,
                        str(agent_name),
                        self._json_dumps(output),
                        self._now(),
                    ),
                )
                count += 1

        return {"success": True, "recorded_agent_outputs": count}

    def _extract_final_fields(self, pipeline_results: Dict[str, Any]) -> Dict[str, Any]:
        risk = pipeline_results.get("risk_result") or {}
        strategy = pipeline_results.get("strategy_result") or {}
        llm = pipeline_results.get("llm_report_result") or {}

        return {
            "final_signal": risk.get("final_signal") or risk.get("risk_for_next_agent", {}).get("final_signal"),
            "risk_level": risk.get("risk_level") or risk.get("risk_for_next_agent", {}).get("risk_level"),
            "strategy_action": strategy.get("strategy_action"),
            "strategy_level": strategy.get("strategy_level"),
            "llm_source": llm.get("source"),
        }

    def record_interface_session(
        self,
        symbol: str,
        user_context: Optional[Dict[str, Any]] = None,
        chart_context: Optional[Dict[str, Any]] = None,
        pipeline_results: Optional[Dict[str, Any]] = None,
        chart_df: Optional[pd.DataFrame] = None,
        save_artifact: bool = True,
    ) -> Dict[str, Any]:
        symbol = self._normalise_symbol(symbol)
        user_context = user_context or {}
        chart_context = chart_context or {}
        pipeline_results = pipeline_results or {}

        session_id = str(uuid.uuid4())
        final = self._extract_final_fields(pipeline_results)

        agent_outputs = {
            "Data Agent": pipeline_results.get("multi_quote"),
            "Historical Data Agent": pipeline_results.get("historical_data"),
            "Chart Historical Data": pipeline_results.get("chart_historical_data"),
            "Validation Agent": pipeline_results.get("validation_result"),
            "Analyst Agent": pipeline_results.get("analysis_result"),
            "Training Agent": pipeline_results.get("training_result"),
            "Signal Model": pipeline_results.get("signal_result"),
            "Risk Agent": pipeline_results.get("risk_result"),
            "Strategist Agent": pipeline_results.get("strategy_result"),
            "Reward Agent": pipeline_results.get("reward_record_result"),
            "Reward Update Agent": pipeline_results.get("auto_reward_update_result"),
            "LLM Report Agent": pipeline_results.get("llm_report_result"),
            "Screener Agent": pipeline_results.get("screener_result"),
            "Evaluator Agent": pipeline_results.get("evaluation_result"),
            "News / Report Agent": pipeline_results.get("news_report_result"),
            "Storage Agent": pipeline_results.get("storage_result"),
        }

        chart_record = self.record_chart_snapshot(
            session_id=session_id,
            symbol=symbol,
            chart_df=chart_df,
            period=str(chart_context.get("period", "")),
            interval=str(chart_context.get("interval", "")),
        )

        agent_record = self.record_agent_outputs(session_id, symbol, agent_outputs)

        summary = {
            "session_id": session_id,
            "symbol": symbol,
            "user_context": user_context,
            "chart_context": chart_context,
            "final": final,
            "chart_summary": chart_record.get("chart_summary"),
            "recorded_agent_outputs": agent_record.get("recorded_agent_outputs"),
        }

        artifact_path = None
        if save_artifact:
            artifact_path = self.export_session_json(
                session_id=session_id,
                symbol=symbol,
                user_context=user_context,
                chart_context=chart_context,
                pipeline_results=pipeline_results,
                chart_summary=chart_record.get("chart_summary"),
            ).get("artifact_path")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ui_sessions (
                    session_id, symbol, user_intent, query_modes_json,
                    has_position, shares, average_cost, portfolio_context_json,
                    event_context_json, chart_period, chart_interval,
                    final_signal, risk_level, strategy_action, strategy_level,
                    llm_source, created_at_utc, artifact_path, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    symbol,
                    user_context.get("user_intent"),
                    self._json_dumps(user_context.get("query_modes", [])),
                    1 if user_context.get("has_position") else 0,
                    user_context.get("shares"),
                    user_context.get("average_cost"),
                    self._json_dumps(user_context.get("portfolio_context", {})),
                    self._json_dumps(user_context.get("event_context", {})),
                    chart_context.get("period"),
                    chart_context.get("interval"),
                    final.get("final_signal"),
                    final.get("risk_level"),
                    final.get("strategy_action"),
                    final.get("strategy_level"),
                    final.get("llm_source"),
                    self._now(),
                    artifact_path,
                    self._json_dumps(summary),
                ),
            )

        return {
            "success": True,
            "session_id": session_id,
            "artifact_path": artifact_path,
            "summary": summary,
        }

    # Backward-compatible aliases
    def record_session_bundle(self, *args, **kwargs) -> Dict[str, Any]:
        return self.record_interface_session(*args, **kwargs)

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        return self.record_interface_session(*args, **kwargs)

    def export_session_json(
        self,
        session_id: str,
        symbol: str,
        user_context: Dict[str, Any],
        chart_context: Dict[str, Any],
        pipeline_results: Dict[str, Any],
        chart_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        symbol = self._normalise_symbol(symbol)
        filename = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{symbol}_{session_id[:8]}.json"
        path = self.artifact_dir / filename

        payload = {
            "session_id": session_id,
            "symbol": symbol,
            "created_at_utc": self._now(),
            "user_context": self._json_safe(user_context),
            "chart_context": self._json_safe(chart_context),
            "chart_summary": self._json_safe(chart_summary),
            "pipeline_results": self._json_safe(pipeline_results),
            "note": "This is a UI/session audit record for paper decision support only. It is not a real trade record.",
        }

        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ui_artifacts (
                    session_id, artifact_type, file_path, metadata_json, created_at_utc
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    "session_json",
                    str(path),
                    self._json_dumps({"symbol": symbol, "filename": filename}),
                    self._now(),
                ),
            )

        return {"success": True, "artifact_path": str(path)}

    # ------------------------------------------------------------------
    # Read methods for UI dashboards
    # ------------------------------------------------------------------
    def get_recent_ui_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit or 20), 200))
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT * FROM ui_sessions
                ORDER BY created_at_utc DESC
                LIMIT {limit}
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_ui_agent_records(self, session_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM ui_agent_records
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_ui_chart_records(self, session_id: str) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT * FROM ui_chart_records
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [dict(row) for row in rows]
