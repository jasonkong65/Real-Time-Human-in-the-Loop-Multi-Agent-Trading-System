import streamlit as st

from agents.data_agent import DataAgent
from agents.validation_agent import ValidationAgent
from agents.analyst_agent import AnalystAgent
from agents.training_agent import TrainingAgent
from agents.risk_agent import RiskAgent


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
st.subheader(
    "Data Agent + Validation Agent + Two-Stage Analyst Agent + Training Agent + Q-learning Risk Agent"
)

st.info(
    "This prototype collects market data from Finnhub and Alpha Vantage, "
    "validates multi-source consistency, performs two-stage market analysis, "
    "trains or loads a lightweight signal model, and applies rule-based plus "
    "Q-learning risk control before user confirmation."
)

symbol = st.text_input("Enter stock symbol", value="AAPL")

if st.button("Run Agent Pipeline"):
    validation_agent = ValidationAgent()
    analyst_agent = AnalystAgent()
    training_agent = TrainingAgent()
    risk_agent = RiskAgent()

    # 1. Data Agent
    with st.spinner("Data Agent is collecting market data..."):
        multi_quote, historical_data = fetch_market_data(symbol)

    # 2. Validation Agent
    with st.spinner("Validation Agent is checking multi-source data reliability..."):
        validation_result = validation_agent.validate_multi_source_quote(multi_quote)

    # 3. Analyst Agent
    with st.spinner("Analyst Agent is calculating quote-level and historical features..."):
        analysis_result = analyst_agent.analyse_market(
            multi_quote=multi_quote,
            validation_result=validation_result,
            historical_data=historical_data
        )

    # 4. Training Agent
    with st.spinner("Training Agent is training or loading the signal model..."):
        if historical_data.get("success"):
            training_result = training_agent.train_from_historical_data(historical_data)
        else:
            training_result = training_agent.train_from_csv("data/historical_data.csv")

    # 5. Signal Model
    with st.spinner("Signal Model is generating trading signal..."):
        signal_result = training_agent.predict_signal(analysis_result)

    # 6. Risk Agent
    with st.spinner("Risk Agent is applying safety rules and Q-learning risk adjustment..."):
        risk_result = risk_agent.assess_risk(
            validation_result=validation_result,
            analysis_result=analysis_result,
            signal_result=signal_result
        )

    st.session_state["last_risk_result"] = risk_result

    # Summary section
    st.subheader("Agent Decision Summary")

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Symbol", multi_quote.get("symbol", "N/A"))

    with col2:
        st.metric("Validation", validation_result.get("confidence", "N/A"))

    with col3:
        st.metric("Next Action", validation_result.get("next_action", "N/A"))

    with col4:
        st.metric("Analyst Signal", analysis_result.get("analyst_signal", "N/A"))

    with col5:
        st.metric("Model Signal", signal_result.get("model_signal", "N/A"))

    with col6:
        st.metric("Final Signal", risk_result.get("final_signal", "N/A"))

    st.write(f"**Validation Decision:** {validation_result.get('agent_decision', 'N/A')}")
    st.write(f"**Analyst Decision:** {analysis_result.get('agent_decision', 'N/A')}")
    st.write(f"**Signal Model Decision:** {signal_result.get('agent_decision', 'N/A')}")
    st.write(f"**Risk Decision:** {risk_result.get('agent_decision', 'N/A')}")

    # Main status messages
    if validation_result.get("is_valid") and validation_result.get("confidence") == "High":
        st.success(validation_result.get("summary", "Validation passed."))
    elif validation_result.get("is_valid"):
        st.warning(validation_result.get("summary", "Validation passed with caution."))
    else:
        st.error(validation_result.get("summary", "Validation failed."))

    if analysis_result.get("success"):
        st.success(analysis_result.get("summary", "Analysis completed."))
    else:
        st.error(analysis_result.get("summary", "Analysis failed."))

    if signal_result.get("success"):
        st.success(signal_result.get("summary", "Signal generated."))
    else:
        st.warning(signal_result.get("summary", "Signal generation used fallback or failed."))

    if risk_result.get("success"):
        st.success(risk_result.get("summary", "Risk assessment completed."))
    else:
        st.error(risk_result.get("summary", "Risk assessment failed."))

    # 1. Data Agent Output
    st.subheader("1. Data Agent Output")

    left, right = st.columns(2)

    with left:
        st.markdown("### Finnhub Quote")
        st.json(multi_quote["finnhub"])

    with right:
        st.markdown("### Alpha Vantage Quote")
        st.json(multi_quote["alpha_vantage"])

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

    # 2. Validation Agent Output
    st.subheader("2. Validation Agent Output")
    st.json(validation_result)

    # 3. Analyst Agent Output
    st.subheader("3. Analyst Agent Output")
    st.json(analysis_result)

    # 4. Training Agent Output
    st.subheader("4. Training Agent Output")
    st.json(training_result)

    # 5. Signal Model Output
    st.subheader("5. Signal Model Output")
    st.json(signal_result)

    # 6. Risk Agent Output
    st.subheader("6. Risk Agent Output")
    st.json(risk_result)

    # Reasoning steps
    if validation_result.get("reasoning_steps"):
        with st.expander("Show Validation Reasoning Steps"):
            for step in validation_result["reasoning_steps"]:
                st.write(f"- {step}")

    if analysis_result.get("reasoning_steps"):
        with st.expander("Show Analyst Reasoning Steps"):
            for step in analysis_result["reasoning_steps"]:
                st.write(f"- {step}")

    if risk_result.get("reasoning_steps"):
        with st.expander("Show Risk Agent Reasoning Steps"):
            for step in risk_result["reasoning_steps"]:
                st.write(f"- {step}")

    # Stage-level outputs
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


# Q-learning feedback section
if "last_risk_result" in st.session_state:
    st.subheader("Q-learning Feedback Demo")

    st.info(
        "This section simulates delayed reward feedback. "
        "In a real trading system, future_return would come from later market outcomes. "
        "Here, it is manually entered for demonstration."
    )

    future_return = st.number_input(
        "Enter simulated future return for Q-learning update",
        min_value=-0.20,
        max_value=0.20,
        value=0.00,
        step=0.005,
        format="%.3f"
    )

    if st.button("Update Risk Q-table"):
        feedback_risk_agent = RiskAgent()
        update_result = feedback_risk_agent.update_from_feedback(
            risk_result=st.session_state["last_risk_result"],
            future_return=future_return
        )

        st.subheader("Q-learning Update Output")
        st.json(update_result)