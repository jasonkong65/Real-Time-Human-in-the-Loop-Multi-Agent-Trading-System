import streamlit as st

from agents.data_agent import DataAgent
from agents.validation_agent import ValidationAgent


st.set_page_config(
    page_title="Real-Time Trading Agent",
    layout="wide"
)

st.title("Real-Time Trading Agent")
st.subheader("Step 3–4: Finnhub + iTick Multi-Source Validation")

st.info(
    "This prototype uses Finnhub as the primary market data source and iTick as a secondary source. "
    "The Validation Agent compares both sources and adjusts confidence based on data consistency."
)

col1, col2 = st.columns(2)

with col1:
    symbol = st.text_input("Enter stock symbol", value="AAPL")

with col2:
    region = st.selectbox("iTick region", options=["US", "HK", "SH", "SZ"], index=0)

if st.button("Get Multi-Source Quote"):
    data_agent = DataAgent()
    validation_agent = ValidationAgent()

    with st.spinner("Data Agent is collecting data from Finnhub and iTick..."):
        multi_quote = data_agent.get_multi_source_quote(symbol, region)

    st.subheader("1. Data Agent Output")

    left, right = st.columns(2)

    with left:
        st.markdown("### Finnhub")
        st.json(multi_quote["finnhub"])

    with right:
        st.markdown("### iTick")
        st.json(multi_quote["itick"])

    with st.spinner("Validation Agent is performing multi-source validation..."):
        multi_validation = validation_agent.validate_multi_source_quote(multi_quote)

    st.subheader("2. Multi-Source Validation Output")
    st.json(multi_validation)

    if multi_validation["is_valid"] and multi_validation["confidence"] == "High":
        st.success(multi_validation["summary"])
    elif multi_validation["is_valid"]:
        st.warning(multi_validation["summary"])
    else:
        st.error(multi_validation["summary"])

    if multi_validation["warnings"]:
        st.warning("Warnings detected:")
        for warning in multi_validation["warnings"]:
            st.write(f"- {warning}")

    if multi_validation["issues"]:
        st.error("Issues detected:")
        for issue in multi_validation["issues"]:
            st.write(f"- {issue}")