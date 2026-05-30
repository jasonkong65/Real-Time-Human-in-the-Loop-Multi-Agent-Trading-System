from __future__ import annotations


def test_agent_compatibility_imports():
    from agents.analyst_agent import AnalystAgent
    from agents.risk_agent import RiskAgent
    from agents.storage_agent import StorageAgent
    from agents.training_agent import TrainingAgent

    from agents.analysis.agent import AnalystAgent as PackageAnalystAgent
    from agents.risk.agent import RiskAgent as PackageRiskAgent
    from agents.storage.agent import StorageAgent as PackageStorageAgent
    from agents.training.agent import TrainingAgent as PackageTrainingAgent

    assert AnalystAgent is PackageAnalystAgent
    assert RiskAgent is PackageRiskAgent
    assert StorageAgent is PackageStorageAgent
    assert TrainingAgent is PackageTrainingAgent


def test_core_agents_instantiate_with_temp_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    from agents.data_agent import DataAgent
    from agents.validation_agent import ValidationAgent
    from agents.analyst_agent import AnalystAgent
    from agents.training_agent import TrainingAgent
    from agents.risk_agent import RiskAgent
    from agents.storage_agent import StorageAgent
    from agents.evaluator_agent import EvaluatorAgent
    from agents.execution_agent import ExecutionAgent
    from agents.strategist_agent import StrategistAgent
    from agents.llm_report_agent import LLMReportAgent

    storage = StorageAgent(db_path=str(tmp_path / "trading_system.db"))
    risk = RiskAgent(
        dqn_model_path=str(tmp_path / "models" / "risk_dqn_model.pt"),
        target_model_path=str(tmp_path / "models" / "risk_dqn_target_model.pt"),
        replay_path=str(tmp_path / "data" / "risk_dqn_replay.csv"),
        q_table_path=str(tmp_path / "models" / "risk_q_table.pkl"),
        replay_db_path=str(tmp_path / "data" / "trading_system.db"),
        min_replay_samples=5,
        config_path=str(tmp_path / "missing_risk_config.json"),
    )

    assert DataAgent(cache_path=str(tmp_path / "cache" / "quotes.json"), storage_enabled=False)
    assert ValidationAgent(config_path=str(tmp_path / "validation_config.json"))
    assert AnalystAgent(config_path=str(tmp_path / "analyst_config.json"))
    assert TrainingAgent(model_path=str(tmp_path / "models" / "signal_model.pkl"))
    assert risk.policy_net is not None and risk.target_net is not None
    assert storage.backend.dialect == "sqlite"
    assert EvaluatorAgent(db_path=str(tmp_path / "trading_system.db"))
    assert ExecutionAgent(db_path=str(tmp_path / "ui.db"), artifact_dir=str(tmp_path / "ui_sessions"))
    assert StrategistAgent()
    assert LLMReportAgent()._is_available() is False
