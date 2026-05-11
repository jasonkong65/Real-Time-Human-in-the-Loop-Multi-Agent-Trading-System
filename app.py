import streamlit as st

from agents.data_agent import DataAgent
from agents.validation_agent import ValidationAgent


st.set_page_config(
    page_title="Real-Time Trading Agent",
    layout="wide"
)

st.title("Real-Time Trading Agent")
st.subheader("Step 1: Data Agent + Validation Agent")

st.info(
    "This prototype uses Finnhub API to collect live stock quote data. "
    "The Validation Agent checks whether the data is valid before any trading analysis is performed."
)

symbol = st.text_input("Enter stock symbol", value="AAPL")

if st.button("Get Live Quote"):
    data_agent = DataAgent()
    validation_agent = ValidationAgent()

    with st.spinner("Data Agent is collecting live market data..."):
        quote = data_agent.get_live_quote(symbol)

    st.subheader("1. Data Agent Output")
    st.json(quote)

    with st.spinner("Validation Agent is checking data quality..."):
        validation_result = validation_agent.validate_quote(quote)

    st.subheader("2. Validation Agent Output")
    st.json(validation_result)

    if validation_result["is_valid"]:
        st.success(validation_result["summary"])
    else:
        st.error(validation_result["summary"])

    if validation_result["warnings"]:
        st.warning("Warnings detected:")
        for warning in validation_result["warnings"]:
            st.write(f"- {warning}")

    if validation_result["issues"]:
        st.error("Issues detected:")
        for issue in validation_result["issues"]:
            st.write(f"- {issue}")