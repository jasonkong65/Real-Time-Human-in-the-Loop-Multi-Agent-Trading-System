from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agents.data_agent import DataAgent
from agents.validation_agent import ValidationAgent
from agents.historical_data_agent import HistoricalDataAgent
from agents.analyst_agent import AnalystAgent
from agents.training_agent import TrainingAgent
from agents.risk_agent import RiskAgent
from agents.strategist_agent import StrategistAgent
from agents.reward_agent import RewardAgent
from agents.storage_agent import StorageAgent
from agents.screener_agent import ScreenerAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.execution_agent import ExecutionAgent

try:
    from agents.llm_report_agent import LLMReportAgent
except Exception:
    LLMReportAgent = None

from app_components.charting import get_live_chart_selection
from app_components.helpers import call_agent_method, historical_to_dataframe
from app_components.schema_repair import repair_sqlite_schema_without_agent_changes

load_dotenv()

def load_agents():
    repair_sqlite_schema_without_agent_changes()
    storage_agent = StorageAgent()
    agents = {
        "data": DataAgent(),
        "validation": ValidationAgent(),
        "historical": HistoricalDataAgent(storage_agent=storage_agent),
        "analyst": AnalystAgent(),
        "training": TrainingAgent(),
        "risk": RiskAgent(),
        "strategist": StrategistAgent(),
        "reward": RewardAgent(),
        "storage": storage_agent,
        "screener": ScreenerAgent(),
        "evaluator": EvaluatorAgent(),
        "execution": ExecutionAgent(),
    }
    if LLMReportAgent is not None:
        try:
            agents["llm"] = LLMReportAgent()
        except Exception:
            agents["llm"] = None
    else:
        agents["llm"] = None
    return agents


def fetch_live_chart_data(
    symbol: str,
    chart_period: str,
    chart_interval: str,
    force_refresh: bool = False,
    refresh_nonce: int = 0,
) -> Tuple[Dict[str, Any], pd.DataFrame, str]:
    """Fetch chart data only, so chart-period changes do not rerun all agents."""
    agents = load_agents()
    chart_historical_data = call_agent_method(
        agents["historical"],
        ["get_or_download_data", "run"],
        symbol,
        chart_period,
        chart_interval,
        symbol=symbol,
        period=chart_period,
        interval=chart_interval,
        force_refresh=force_refresh,
    )
    chart_df = historical_to_dataframe(chart_historical_data)
    fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return chart_historical_data, chart_df, fetched_at


def get_current_chart_for_display(symbol: str, fallback_bundle: Optional[Dict[str, Any]] = None) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """Return chart data using current live chart controls, with safe fallback."""
    chart_label, chart_period, chart_interval, chart_style = get_live_chart_selection()
    fallback_bundle = fallback_bundle or {}
    force_refresh = bool(st.session_state.pop("chart_force_refresh_once", False))
    refresh_nonce = int(st.session_state.get("chart_refresh_nonce", 0))

    metadata = {
        "label": chart_label,
        "period": chart_period,
        "interval": chart_interval,
        "style": chart_style,
        "force_refresh": force_refresh,
        "fetched_at": None,
        "source": "not_loaded",
        "error": None,
    }

    try:
        chart_data, chart_df, fetched_at = fetch_live_chart_data(
            symbol,
            chart_period,
            chart_interval,
            force_refresh=force_refresh,
            refresh_nonce=refresh_nonce,
        )
        metadata.update({
            "fetched_at": fetched_at,
            "source": chart_data.get("source") or "historical_agent",
            "num_records": chart_data.get("num_records"),
            "latest_timestamp": chart_data.get("latest_timestamp") or chart_data.get("latest_date"),
            "is_stale": chart_data.get("is_stale"),
            "warnings": chart_data.get("warnings", []),
        })
        return chart_df, chart_data, metadata
    except Exception as exc:
        fallback_df = fallback_bundle.get("chart_df")
        if fallback_df is None:
            fallback_df = historical_to_dataframe(fallback_bundle.get("chart_historical_data", {}))
        fallback_data = fallback_bundle.get("chart_historical_data", {}) or {}
        metadata.update({
            "source": "fallback_from_last_pipeline_run",
            "error": str(exc),
        })
        return fallback_df, fallback_data, metadata