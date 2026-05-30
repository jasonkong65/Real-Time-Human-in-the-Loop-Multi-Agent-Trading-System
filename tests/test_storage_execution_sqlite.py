from __future__ import annotations

import json
import sqlite3

import pandas as pd


def test_storage_agent_records_and_reads_historical_prices(tmp_path):
    from agents.storage_agent import StorageAgent

    db_path = tmp_path / "trading_system.db"
    storage = StorageAgent(db_path=str(db_path))
    prices = [
        {"date": "2024-01-01", "open": 10, "high": 11, "low": 9, "close": 10.5, "adj_close": 10.5, "volume": 1000},
        {"date": "2024-01-02", "open": 11, "high": 12, "low": 10, "close": 11.5, "adj_close": 11.5, "volume": 1100},
    ]
    written = storage.record_historical_prices("aapl", prices, period="1y", interval="1d", source="test")
    assert written["success"] is True
    assert written["rows_written"] == 2

    read_back = storage.get_historical_prices("AAPL", period="max", interval="1d")
    assert isinstance(read_back, pd.DataFrame)
    assert len(read_back) == 2
    assert list(read_back["close"]) == [10.5, 11.5]

    summary = storage.get_storage_summary()
    assert summary["success"] is True
    assert summary["table_counts"]["historical_prices"] == 2


def test_execution_agent_records_ui_session(tmp_path):
    from agents.execution_agent import ExecutionAgent

    db_path = tmp_path / "ui.db"
    agent = ExecutionAgent(db_path=str(db_path), artifact_dir=str(tmp_path / "ui_sessions"))
    chart_df = pd.DataFrame(
        {"close": [10.0, 11.0], "volume": [100, 110]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    result = agent.record_interface_session(
        symbol="AAPL",
        user_context={"user_intent": "Research only", "query_modes": ["Price chart"]},
        chart_context={"period": "1y", "interval": "1d"},
        pipeline_results={"risk_result": {"final_signal": "HOLD", "risk_level": "Low"}, "strategy_result": {"strategy_action": "MONITOR_POSITIVE_SETUP"}},
        chart_df=chart_df,
        save_artifact=True,
    )
    assert result["success"] is True
    assert result["session_id"]

    sessions = agent.get_recent_ui_sessions(limit=5)
    assert len(sessions) == 1
    assert sessions[0]["symbol"] == "AAPL"

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM ui_chart_records").fetchone()[0]
    assert count == 1
