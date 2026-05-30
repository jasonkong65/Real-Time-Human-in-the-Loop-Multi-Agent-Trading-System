from __future__ import annotations


def test_build_screener_report_uses_local_fallback_when_llm_absent():
    from app_components.workflows import build_screener_report

    result = build_screener_report(
        agents={"llm": None},
        screener_result={
            "top_buy_candidates": [{"symbol": "AAPL"}],
            "highest_risk_candidates": [{"symbol": "TSLA"}],
        },
    )
    assert result["success"] is True
    assert result["source"] == "local_fallback_no_llm_agent"
    assert "AAPL" in result["plain_language_report"]


def test_call_agent_method_tries_aliases():
    from app_components.helpers import call_agent_method

    class Agent:
        def run(self, symbol):
            return {"symbol": symbol}

    assert call_agent_method(Agent(), ["missing", "run"], "AAPL") == {"symbol": "AAPL"}
