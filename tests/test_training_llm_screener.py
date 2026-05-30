from __future__ import annotations

import pandas as pd


def test_training_agent_builds_dataset(sample_historical_data, tmp_path):
    from agents.training_agent import TrainingAgent

    agent = TrainingAgent(model_path=str(tmp_path / "models" / "signal_model.pkl"), pooled_data_dir=str(tmp_path / "historical"))
    X, y, meta = agent._build_single_dataset(sample_historical_data, validation_confidence_score=0.9)
    assert not X.empty
    assert len(X) == len(y)
    assert set(agent.feature_columns).issubset(set(X.columns))
    assert meta["num_samples"] == len(X)


def test_llm_report_agent_screener_fallback():
    from agents.llm_report_agent import LLMReportAgent

    result = LLMReportAgent().generate_screener_report(
        screener_result={
            "top_buy_candidates": [{"symbol": "AAPL"}, {"symbol": "MSFT"}],
            "highest_risk_candidates": [{"symbol": "TSLA"}],
        }
    )
    assert result["success"] is True
    assert result["source"] in {"local_fallback", "groq"}
    assert "AAPL" in result["plain_language_report"]


def test_screener_agent_scores_universe_without_network(tmp_path, monkeypatch, sample_historical_data):
    monkeypatch.chdir(tmp_path)
    from agents.screener_agent import ScreenerAgent

    agent = ScreenerAgent(db_path=str(tmp_path / "screen.db"), use_yfinance_metadata=False)
    agent.history_agent.get_or_download_data = lambda symbol, period="1y", interval="1d": {
        **sample_historical_data,
        "symbol": symbol,
    }
    result = agent.screen_universe(["AAPL", "MSFT", "NVDA"], top_n=2, period="1y", interval="1d", save_to_storage=False)
    assert result["success"] is True
    assert result["scanned_count"] >= 1
    assert len(result["top_buy_candidates"]) <= 2
    assert all("symbol" in row for row in result["top_buy_candidates"])
