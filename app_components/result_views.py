import json
from typing import Any, Dict

import pandas as pd
import streamlit as st

from app_components.agent_factory import get_current_chart_for_display
from app_components.charting import render_chart, render_live_chart_controls
from app_components.constants import DEFAULT_CHART_LABEL, DEFAULT_CHART_STYLE
from app_components.helpers import (
    clean_label,
    format_pct,
    format_price,
    get_nested,
    selected_price_from_quote,
)
from app_components.strategy_view import render_strategy_guidance_plain
from app_components.ui_helpers import card, card_variant_from_text, render_status_pills

"""Result rendering functions for displaying the outcomes of the multi-agent stock research pipeline."""

def render_error_if_needed() -> None:
    if "last_error" in st.session_state:
        st.error("The selected workflow crashed.")
        st.code(st.session_state["last_error"]["error"])
        with st.expander("Traceback"):
            st.code(st.session_state["last_error"]["traceback"])
        del st.session_state["last_error"]


def render_start_message() -> None:
    st.info("Enter a stock symbol, choose the analysis modules, then click **Run selected research**.")
    st.markdown(
        """
        #### How to use this system

        1. Enter a stock ticker, such as `AAPL`, `MSFT`, or `NVDA`.
        2. For a normal stock check, keep **Single-stock agent pipeline** and **Price chart** selected.
        3. Click **Run selected research** to let the agents collect data, validate sources, analyse the stock, assess risk, and generate a plain-language report.
        4. Read the **Research Summary** first. It shows the price, analyst signal, model signal, risk level, and strategy suggestion.
        5. Use the **Chart** tab to change the chart period or refresh chart data without rerunning the whole pipeline.
        6. Optional: add portfolio context if you already hold the stock, or run the News / Report Agent and Watchlist Screener when you need extra context.
        7. Check **Agent Responses**, **Evaluator**, and **Storage / Logs** if you want to inspect the technical outputs and saved paper-decision history.
        """
    )


def render_results(bundle: Dict[str, Any], agents: Dict[str, Any]) -> None:
    symbol = bundle.get("symbol", "")
    chart_df, live_chart_historical_data, live_chart_metadata = get_current_chart_for_display(symbol, fallback_bundle=bundle)
    live_chart_label = live_chart_metadata.get("label", DEFAULT_CHART_LABEL)
    live_chart_period = live_chart_metadata.get("period", "1y")
    live_chart_interval = live_chart_metadata.get("interval", "1d")
    live_chart_style = live_chart_metadata.get("style", DEFAULT_CHART_STYLE)

    risk_result = bundle.get("risk_result", {}) or {}
    strategy_result = bundle.get("strategy_result", {}) or {}
    analysis_result = bundle.get("analysis_result", {}) or {}
    signal_result = bundle.get("signal_result", {}) or {}
    validation_result = bundle.get("validation_result", {}) or {}
    multi_quote = bundle.get("multi_quote", {}) or {}
    llm_result = bundle.get("llm_report_result", {}) or {}

    entry_price = selected_price_from_quote(multi_quote, validation_result)

    st.markdown("### Research Summary")

    summary_cols = st.columns(6)
    with summary_cols[0]:
        card("Symbol", symbol, variant="blue")
    with summary_cols[1]:
        card("Price", format_price(entry_price), variant="teal")
    with summary_cols[2]:
        analyst_label = clean_label(analysis_result.get("display_signal") or analysis_result.get("analyst_signal"))
        card("Analyst", analyst_label, variant=card_variant_from_text(analyst_label, "green"))
    with summary_cols[3]:
        model_label = clean_label(signal_result.get("display_signal") or signal_result.get("model_signal") or signal_result.get("signal"))
        card("Model", model_label, variant=card_variant_from_text(model_label, "indigo"))
    with summary_cols[4]:
        risk_label = clean_label(risk_result.get("risk_level"))
        risk_variant = "green" if risk_label.lower() == "low" else "amber" if risk_label.lower() == "medium" else "red"
        card("Risk", risk_label, variant=risk_variant)
    with summary_cols[5]:
        strategy_label = clean_label(strategy_result.get("strategy_action"))
        card("Strategy", strategy_label, variant=card_variant_from_text(strategy_label, "purple"))

    status_items = [
        f"Chart: {live_chart_label} / {live_chart_period} / {live_chart_interval}",
        f"Chart source: {live_chart_metadata.get('source', 'Unknown')}",
        f"Validation: {validation_result.get('confidence', 'Unknown')}",
        f"Final signal: {clean_label(risk_result.get('final_signal'))}",
        f"Strategy level: {clean_label(strategy_result.get('strategy_level'))}",
        f"LLM: {llm_result.get('source', 'not run')}",
    ]
    render_status_pills(status_items)

    tab_overview, tab_chart, tab_agents, tab_news, tab_screener, tab_evaluator, tab_storage = st.tabs(
        [
            "Overview",
            "Chart",
            "Agent Responses",
            "News / Report",
            "Screener",
            "Evaluator",
            "Storage / Logs",
        ]
    )

    with tab_overview:
        left, right = st.columns([1.1, 0.9])
        with left:
            st.markdown("#### Groq / Report Agent Output")
            report_text = (
                llm_result.get("plain_language_report")
                or llm_result.get("report")
                or llm_result.get("summary")
                or "No LLM report was generated for this run."
            )
            st.markdown(report_text)

            st.markdown("#### Strategy Guidance")
            render_strategy_guidance_plain(strategy_result, risk_result)

        with right:
            st.markdown("#### Chart Preview")
            render_chart(chart_df, symbol, chart_style=live_chart_style)
            render_live_chart_controls("overview")
            st.caption(
                f"Displayed period: {live_chart_label} ({live_chart_period}/{live_chart_interval}) · "
                f"source: {live_chart_metadata.get('source', 'Unknown')} · "
                f"latest: {live_chart_metadata.get('latest_timestamp') or 'N/A'} · "
                f"fetched: {live_chart_metadata.get('fetched_at') or 'N/A'}"
            )
            if live_chart_metadata.get("error"):
                st.warning(f"Live chart refresh failed, showing fallback chart: {live_chart_metadata['error']}")

    with tab_chart:
        st.markdown(f"#### {symbol} Price Chart")
        render_chart(chart_df, symbol, chart_style=live_chart_style)
        render_live_chart_controls("chart_tab")
        st.caption(
            f"Displayed period: {live_chart_label} ({live_chart_period}/{live_chart_interval}) · "
            f"source: {live_chart_metadata.get('source', 'Unknown')} · "
            f"latest: {live_chart_metadata.get('latest_timestamp') or 'N/A'} · "
            f"fetched: {live_chart_metadata.get('fetched_at') or 'N/A'}"
        )
        if live_chart_metadata.get("error"):
            st.warning(f"Live chart refresh failed, showing fallback chart: {live_chart_metadata['error']}")
        with st.expander("Chart data preview"):
            if isinstance(chart_df, pd.DataFrame) and not chart_df.empty:
                st.dataframe(chart_df.tail(100), use_container_width=True)
            else:
                st.info("No chart data available.")

    with tab_agents:
        st.markdown("#### Agent Responses")
        agent_outputs = {
            "Data Agent": bundle.get("multi_quote"),
            "Validation Agent": bundle.get("validation_result"),
            "Historical Data Agent": bundle.get("historical_data"),
            "Analyst Agent": bundle.get("analysis_result"),
            "Training Agent": bundle.get("training_result"),
            "Training Agent Diagnostics": bundle.get("training_diagnostics_result"),
            "Signal Model": bundle.get("signal_result"),
            "Risk Agent": bundle.get("risk_result"),
            "Strategist Agent": bundle.get("strategy_result"),
            "Reward Agent": bundle.get("reward_record_result"),
            "Reward Update Agent": bundle.get("auto_reward_update_result"),
            "LLM Report Agent": bundle.get("llm_report_result"),
            "Screener Report Agent": bundle.get("screener_report_result"),
            "Execution / Session Agent": bundle.get("execution_result"),
            "Storage Agent": bundle.get("storage_result"),
        }
        for name, output in agent_outputs.items():
            if output is None:
                continue
            with st.expander(name, expanded=name in ["Risk Agent", "Strategist Agent"]):
                st.json(output)

    with tab_news:
        news_result = bundle.get("news_report_result")
        if not news_result:
            st.info("News/report summarizer was not selected for this run.")
        else:
            st.markdown("#### Financial News / Report Summary")
            st.markdown(news_result.get("plain_language_report") or news_result.get("summary") or "No summary text was returned.")

            company_news = news_result.get("company_specific_news") or get_nested(news_result, ["verified_context", "company_specific_news"]) or []
            excluded_news = news_result.get("excluded_news") or get_nested(news_result, ["verified_context", "excluded_news"]) or []

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("##### Company-specific news used")
                if company_news:
                    st.dataframe(pd.DataFrame(company_news), use_container_width=True)
                else:
                    st.info("No company-specific news items were returned.")
            with col_b:
                st.markdown("##### Excluded / broad news")
                if excluded_news:
                    st.dataframe(pd.DataFrame(excluded_news), use_container_width=True)
                else:
                    st.info("No excluded news items were returned.")

            with st.expander("Full news/report result"):
                st.json(news_result)

    with tab_screener:
        screener_result = bundle.get("screener_result")
        if not screener_result:
            st.info("Screener was not selected for this run.")
        else:
            st.markdown("#### Watchlist Screener")

            screener_report = bundle.get("screener_report_result") or {}
            if screener_report:
                st.markdown("##### Groq / Report Agent Summary")
                st.markdown(
                    screener_report.get("plain_language_report")
                    or screener_report.get("summary")
                    or "No screener explanation text was returned."
                )
                render_status_pills([
                    f"Report source: {screener_report.get('source', 'unknown')}",
                    f"LLM available: {screener_report.get('llm_available', 'N/A')}",
                ])
                if screener_report.get("error") or screener_report.get("llm_error"):
                    st.caption(f"Report warning: {screener_report.get('error') or screener_report.get('llm_error')}")

            buy = screener_result.get("top_buy_candidates") or []
            risk = screener_result.get("highest_risk_candidates") or screener_result.get("top_sell_risk") or []
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("##### Candidates for further research")
                st.dataframe(pd.DataFrame(buy), use_container_width=True)
            with col_b:
                st.markdown("##### Caution candidates")
                st.dataframe(pd.DataFrame(risk), use_container_width=True)
            with st.expander("Full screener result"):
                st.json(screener_result)
            with st.expander("Full screener report result"):
                st.json(screener_report)

    with tab_evaluator:
        evaluation_result = bundle.get("evaluation_result")
        if not evaluation_result:
            try:
                evaluation_result = agents["evaluator"].evaluate_history()
            except Exception:
                evaluation_result = None

        if not evaluation_result:
            st.info("Evaluator data is not available.")
        else:
            st.markdown("#### Evaluator Agent")
            metrics = evaluation_result.get("metrics") or evaluation_result
            cols = st.columns(4)
            with cols[0]:
                card("Reward Win Rate", format_pct(metrics.get("reward_win_rate")), variant="green")
            with cols[1]:
                card("Directional Win Rate", format_pct(metrics.get("directional_win_rate")), variant="teal")
            with cols[2]:
                card("Avg Reward", metrics.get("average_reward", "N/A"), variant="blue")
            with cols[3]:
                dqn_ready = get_nested(evaluation_result, ["dqn_summary", "ready_for_training"], None)
                card("DQN Ready", str(dqn_ready), variant="green" if dqn_ready else "amber")

            completed_count = metrics.get("completed_reward_count") or evaluation_result.get("completed_reward_count")
            pending_count = metrics.get("pending_count") or evaluation_result.get("pending_count")
            dqn_replay_count = get_nested(evaluation_result, ["dqn_summary", "replay_count"], evaluation_result.get("dqn_replay_count"))
            if not completed_count:
                st.info(
                    "Evaluator metrics are N/A because there are no completed delayed reward records yet. "
                    f"Current status: pending decisions = {pending_count or 0}, completed rewards = {completed_count or 0}, "
                    f"DQN replay samples = {dqn_replay_count or 0}. "
                    "Run more paper decisions and wait until reward horizons complete before interpreting win-rate or average reward."
                )
            with st.expander("Full evaluator result"):
                st.json(evaluation_result)

    with tab_storage:
        st.markdown("#### Execution / UI Sessions")
        try:
            sessions = agents["execution"].get_recent_ui_sessions(limit=20)
            if sessions:
                st.dataframe(pd.DataFrame(sessions), use_container_width=True)
                selected_session = st.selectbox(
                    "Inspect UI session",
                    [s["session_id"] for s in sessions],
                    index=0,
                )
                records = agents["execution"].get_ui_agent_records(selected_session)
                with st.expander("Recorded agent outputs for selected UI session", expanded=False):
                    for row in records:
                        st.markdown(f"##### {row.get('agent_name')}")
                        try:
                            st.json(json.loads(row.get("output_json") or "{}"))
                        except Exception:
                            st.code(row.get("output_json"))
            else:
                st.info("No UI sessions have been recorded yet.")
        except Exception as exc:
            st.warning(f"Could not read UI sessions: {exc}")

        st.markdown("#### Storage Summary")
        try:
            st.json(agents["storage"].get_storage_summary())
        except Exception as exc:
            st.warning(f"Could not read storage summary: {exc}")

        st.markdown("#### Recent Pipeline Runs")
        try:
            recent_runs = agents["storage"].get_recent_pipeline_runs(limit=20)
            st.dataframe(pd.DataFrame(recent_runs), use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not read recent pipeline runs: {exc}")