"""Regression tests for RewardAgent calendar-date comparisons."""
from __future__ import annotations


def test_reward_date_normalization_avoids_timezone_comparison(tmp_path):
    from agents.reward_agent import RewardAgent

    agent = RewardAgent(
        db_path=str(tmp_path / "trading_system.db"),
        pending_path=str(tmp_path / "pending_rewards.csv"),
        history_path=str(tmp_path / "reward_history.csv"),
        mirror_csv=False,
    )

    latest_date = agent._parse_date("2026-05-31")
    target_date = agent._parse_date("2026-05-31 00:00:00+00:00")

    assert latest_date is not None
    assert target_date is not None
    assert latest_date.tzinfo is None
    assert target_date.tzinfo is None
    assert latest_date == target_date
    assert not (latest_date < target_date)
