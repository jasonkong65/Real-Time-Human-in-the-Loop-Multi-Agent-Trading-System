from typing import Any, Dict

import streamlit as st

from app_components.charting import ensure_live_chart_state, get_live_chart_selection
from app_components.helpers import clean_symbol

"""Component for rendering the left control panel and returning the selected settings."""

def render_sidebar() -> Dict[str, Any]:
    """Render the left control panel and return the selected settings."""
    with st.sidebar:
        st.header("Research Input")

        symbol = clean_symbol(st.text_input("Stock symbol", value="AAPL", placeholder="AAPL / MSFT / NVDA"))
        user_intent = st.selectbox(
            "What are you trying to do?",
            [
                "Research only",
                "Considering a paper buy",
                "Considering a paper sell",
                "Already holding - review risk",
            ],
            help="This describes the single-stock decision context. News/report and screener agents are enabled separately below.",
        )

        core_query_modes = st.multiselect(
            "Core stock modules",
            [
                "Single-stock agent pipeline",
                "Price chart",
                "Evaluator dashboard",
                "Storage / session logs",
            ],
            default=["Single-stock agent pipeline", "Price chart"],
            key="core_query_modes",
            help="Core modules for the selected stock.",
        )

        ensure_live_chart_state()
        chart_label, chart_period, chart_interval, chart_style = get_live_chart_selection()
        st.caption(
            f"Chart: {chart_label} ({chart_period}/{chart_interval}). "
            "Change the chart period under the chart; it updates without pressing Run."
        )

        st.divider()
        with st.expander("Optional portfolio / event context", expanded=False):
            st.subheader("Portfolio context")
            has_position = st.checkbox(
                "I currently hold this stock",
                value=("holding" in user_intent.lower()),
                help="Enable this when the strategy should consider an existing paper position.",
            )
            shares = st.number_input(
                "Shares / paper quantity",
                min_value=0.0,
                value=0.0,
                step=1.0,
                disabled=not has_position,
            )
            average_cost = st.number_input(
                "Average cost",
                min_value=0.0,
                value=0.0,
                step=1.0,
                disabled=not has_position,
            )

            st.subheader("Event context")
            earnings_date_text = st.text_input("Next earnings date (optional)", placeholder="YYYY-MM-DD")
            event_risk = st.selectbox("Event risk", ["Unknown", "Low", "Medium", "High"], index=0)

        with st.expander("Model / memory options", expanded=False):
            force_retrain = st.checkbox(
                "Force retrain signal model",
                value=False,
                help="Useful for demonstrating the Training Agent, but slower for normal demos.",
            )
            record_paper_decision = st.checkbox(
                "Record paper decision / memory",
                value=True,
                help="Stores the paper decision for later reward/evaluator analysis.",
            )

        # Training diagnostics are now automatic.
        # They run only when the single-stock pipeline is selected, so the user
        # no longer needs to understand or tick extra model-maintenance boxes.
        # If diagnostics finds a clearly stronger model, TrainingAgent's own
        # quality gate decides whether the saved signal model should be updated.
        run_training_diagnostics = "Single-stock agent pipeline" in core_query_modes
        apply_training_diagnostics_to_main_model = run_training_diagnostics

        st.divider()
        st.subheader("News / Report Agent")
        run_news_report = st.checkbox(
            "Run News / Report Agent",
            value=False,
            help="Runs the financial news/report summarizer separately from the stock-decision intent.",
        )
        source_mode = "auto"
        lookback_days = 14
        max_news = 8
        pasted_financial_text = ""
        if run_news_report:
            source_mode = st.selectbox(
                "Source mode",
                ["auto", "news", "financial", "news_and_financial", "pasted_text"],
                index=0,
            )
            lookback_days = st.slider("News lookback days", min_value=3, max_value=60, value=14)
            max_news = st.slider("Max news items", min_value=3, max_value=20, value=8)
            pasted_financial_text = st.text_area(
                "Paste financial news/report text here, not stock symbol",
                placeholder="Example: Apple reported quarterly earnings... Leave blank to fetch source-grounded data.",
                height=120,
            )
        else:
            st.caption("Off. Turn this on only when you want company news or report summarisation.")

        st.divider()
        st.subheader("Watchlist Screener Agent")
        run_screener = st.checkbox(
            "Run Watchlist Screener",
            value=False,
            help="Scans a watchlist and returns Top-N candidates separately from the single-stock workflow.",
        )
        default_universe = "AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, AMD, NFLX, AVGO, JPM, V, MA, WMT, DIS, INTC, QCOM, CSCO, ORCL"
        screener_symbols_text = default_universe
        top_n = 5
        if run_screener:
            screener_symbols_text = st.text_area("Watchlist symbols", value=default_universe, height=90)
            top_n = st.slider("Top N", min_value=3, max_value=10, value=5)
        else:
            st.caption("Off. Turn this on only when you want a Top-N watchlist scan.")

        query_modes = list(core_query_modes)
        if run_news_report:
            query_modes.append("Financial news / report summary")
        if run_screener:
            query_modes.append("Watchlist screener")
        if run_training_diagnostics:
            query_modes.append("Training diagnostics")
        st.session_state["query_modes"] = query_modes
        st.caption("Selected modules: " + (", ".join(query_modes) if query_modes else "None"))

        run_button = st.button("Run selected research", type="primary", use_container_width=True)

    return {
        "symbol": symbol,
        "user_intent": user_intent,
        "query_modes": query_modes,
        "chart_label": chart_label,
        "chart_period": chart_period,
        "chart_interval": chart_interval,
        "chart_style": chart_style,
        "has_position": has_position,
        "shares": shares,
        "average_cost": average_cost,
        "earnings_date_text": earnings_date_text,
        "event_risk": event_risk,
        "force_retrain": force_retrain,
        "record_paper_decision": record_paper_decision,
        "run_training_diagnostics": run_training_diagnostics,
        "apply_training_diagnostics_to_main_model": apply_training_diagnostics_to_main_model,
        "source_mode": source_mode,
        "lookback_days": lookback_days,
        "max_news": max_news,
        "pasted_financial_text": pasted_financial_text,
        "screener_symbols_text": screener_symbols_text,
        "top_n": top_n,
        "run_button": run_button,
    }
