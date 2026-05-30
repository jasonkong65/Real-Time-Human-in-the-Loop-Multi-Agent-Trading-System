from __future__ import annotations


def test_validation_agent_accepts_consistent_sources(sample_multi_quote, tmp_path):
    from agents.validation_agent import ValidationAgent

    result = ValidationAgent(config_path=str(tmp_path / "validation_config.json")).validate_market_data(sample_multi_quote)
    assert result["success"] is True
    assert result["symbol"] == "AAPL"
    assert result["selected_price"] == 150.0
    assert result["next_action"] in {"ALLOW_ANALYSIS", "ALLOW_ANALYSIS_WITH_CAUTION", "ALLOW_ANALYSIS_WITH_LOW_CONFIDENCE"}
    assert 0 <= result["confidence_score"] <= 1


def test_risk_agent_handles_missing_nested_analysis_sections(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from agents.risk_agent import RiskAgent

    agent = RiskAgent(
        dqn_model_path=str(tmp_path / "models" / "risk_dqn_model.pt"),
        target_model_path=str(tmp_path / "models" / "risk_dqn_target_model.pt"),
        replay_path=str(tmp_path / "data" / "risk_dqn_replay.csv"),
        q_table_path=str(tmp_path / "models" / "risk_q_table.pkl"),
        replay_db_path=str(tmp_path / "data" / "trading_system.db"),
        min_replay_samples=10,
        config_path=str(tmp_path / "missing_config.json"),
        epsilon=0.0,
    )
    validation = {"symbol": "AAPL", "confidence": "High", "confidence_score": 0.95, "next_action": "ALLOW_ANALYSIS"}
    analysis = {"symbol": "AAPL", "analyst_signal": "POSITIVE_BUT_ENTRY_RISK", "stage_2_historical_analysis": None, "features_for_model": None}
    signal = {"symbol": "AAPL", "model_signal": "BUY_CANDIDATE", "prediction_confidence": 0.60, "confidence_level": "Medium"}

    result = agent.assess_risk(signal, analysis, validation)
    assert result["success"] is True
    assert result["symbol"] == "AAPL"
    assert result["final_signal"] in {"BUY_CANDIDATE", "HOLD", "SELL_RISK", "BLOCKED"}
    assert result["dqn_framework"].startswith("PyTorch DQN")
    assert result["dqn_training_ready"] is False


def test_strategist_returns_human_review_plan():
    from agents.strategist_agent import StrategistAgent

    strategy = StrategistAgent().plan_strategy(
        validation_result={"confidence": "High", "symbol": "AAPL"},
        analysis_result={"symbol": "AAPL", "analyst_signal": "POSITIVE_BUT_ENTRY_RISK", "analyst_score": 0.72, "entry_risk_level": "High"},
        training_result={"success": True},
        signal_result={"symbol": "AAPL", "model_signal": "BUY_CANDIDATE", "confidence_level": "Medium"},
        risk_result={"symbol": "AAPL", "final_signal": "HOLD", "risk_level": "Medium", "risk_interpretation": "Entry timing risk is elevated."},
        portfolio_context={"shares": 0},
        event_context={"event_risk": "Unknown"},
    )
    assert strategy["success"] is True
    assert strategy["human_review_required"] is True
    assert strategy["strategy_action"]
    assert isinstance(strategy["checklist"], list) and strategy["checklist"]
