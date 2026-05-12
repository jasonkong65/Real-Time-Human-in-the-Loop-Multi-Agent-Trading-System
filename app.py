import streamlit as st

from agents.data_agent import DataAgent
from agents.validation_agent import ValidationAgent
from agents.analyst_agent import AnalystAgent


st.set_page_config(
    page_title="LLM-Enhanced Multi-Agent Trading System",
    layout="wide"
)


@st.cache_data(ttl=300)
def fetch_market_data(symbol: str):
    """
    Cached market data fetcher.

    The cache lasts for 300 seconds.
    This reduces repeated API calls during testing and demo.
    """
    data_agent = DataAgent()

    multi_quote = data_agent.get_multi_source_quote(symbol)
    historical_data = data_agent.get_historical_daily_prices(symbol)

    return multi_quote, historical_data


st.title("LLM-Enhanced Multi-Agent Trading System")
st.subheader("Step 5: Data Agent + Validation Agent + Two-Stage Analyst Agent")

st.info(
    "This prototype collects market data from Finnhub and Alpha Vantage, "
    "validates multi-source consistency, retrieves historical price data, "
    "and performs two-stage quantitative market analysis before passing results "
    "to later Training, Risk, and LLM agents."
)

symbol = st.text_input("Enter stock symbol", value="AAPL")

if st.button("Run Agent Pipeline"):
    validation_agent = ValidationAgent()
    analyst_agent = AnalystAgent()

    # 1. Data Agent: collect quote data and historical data with cache
    with st.spinner("Data Agent is collecting market data..."):
        multi_quote, historical_data = fetch_market_data(symbol)

    # 2. Validation Agent: validate multi-source quote data
    with st.spinner("Validation Agent is checking multi-source data reliability..."):
        validation_result = validation_agent.validate_multi_source_quote(multi_quote)

    # 3. Analyst Agent: two-stage analysis
    with st.spinner("Analyst Agent is calculating quote-level and historical features..."):
        analysis_result = analyst_agent.analyse_market(
            multi_quote=multi_quote,
            validation_result=validation_result,
            historical_data=historical_data
        )

    # Summary section
    st.subheader("Agent Decision Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Symbol", multi_quote.get("symbol", "N/A"))

    with col2:
        st.metric("Validation", validation_result.get("confidence", "N/A"))

    with col3:
        st.metric("Next Action", validation_result.get("next_action", "N/A"))

    with col4:
        st.metric("Analyst Signal", analysis_result.get("analyst_signal", "N/A"))

    st.write(f"**Validation Decision:** {validation_result.get('agent_decision', 'N/A')}")
    st.write(f"**Analyst Decision:** {analysis_result.get('agent_decision', 'N/A')}")

    # Validation message
    if validation_result.get("is_valid") and validation_result.get("confidence") == "High":
        st.success(validation_result.get("summary", "Validation passed."))
    elif validation_result.get("is_valid"):
        st.warning(validation_result.get("summary", "Validation passed with caution."))
    else:
        st.error(validation_result.get("summary", "Validation failed."))

    # Analyst message
    if analysis_result.get("success"):
        st.success(analysis_result.get("summary", "Analysis completed."))
    else:
        st.error(analysis_result.get("summary", "Analysis failed."))

    # Data Agent outputs
    st.subheader("1. Data Agent Output")

    left, right = st.columns(2)

    with left:
        st.markdown("### Finnhub Quote")
        st.json(multi_quote["finnhub"])

    with right:
        st.markdown("### Alpha Vantage Quote")
        st.json(multi_quote["alpha_vantage"])

    # Historical data summary only, not full raw_data
    with st.expander("Show Historical Data Summary"):
        st.write({
            "source": historical_data.get("source"),
            "success": historical_data.get("success"),
            "symbol": historical_data.get("symbol"),
            "num_price_records": len(historical_data.get("prices", [])),
            "error": historical_data.get("error")
        })

        if historical_data.get("prices"):
            st.write("Latest 5 historical price records:")
            st.dataframe(historical_data["prices"][-5:])

    # Validation Agent output
    st.subheader("2. Validation Agent Output")
    st.json(validation_result)

    # Analyst Agent output
    st.subheader("3. Analyst Agent Output")
    st.json(analysis_result)

    # Reasoning steps
    if validation_result.get("reasoning_steps"):
        with st.expander("Show Validation Reasoning Steps"):
            for step in validation_result["reasoning_steps"]:
                st.write(f"- {step}")

    if analysis_result.get("reasoning_steps"):
        with st.expander("Show Analyst Reasoning Steps"):
            for step in analysis_result["reasoning_steps"]:
                st.write(f"- {step}")

    # Stage-level outputs from two-stage Analyst Agent
    if analysis_result.get("stage_1_quote_analysis"):
        with st.expander("Show Stage 1 Quote-Level Analysis"):
            st.json(analysis_result["stage_1_quote_analysis"])

    if analysis_result.get("stage_2_historical_analysis"):
        with st.expander("Show Stage 2 Historical Analysis"):
            st.json(analysis_result["stage_2_historical_analysis"])

    # Warnings and issues
    if validation_result.get("warnings"):
        st.warning("Validation warnings:")
        for warning in validation_result["warnings"]:
            st.write(f"- {warning}")

    if validation_result.get("issues"):
        st.error("Validation issues:")
        for issue in validation_result["issues"]:
            st.write(f"- {issue}")