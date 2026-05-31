from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

from app_components.agent_factory import load_agents
from app_components.page import configure_page
from app_components.result_views import render_error_if_needed, render_results, render_start_message
from app_components.sidebar import render_sidebar
from app_components.workflows import run_selected_workflow

load_dotenv()
configure_page()

agents = load_agents()

st.title("📊 Human-in-the-Loop Multi-Agent Stock Research System")
st.caption(
    "Paper decision-support only. The system uses agents for data, validation, analysis, "
    "model signal, DQN risk control, strategy planning, memory, and LLM explanation."
)

controls = render_sidebar()

if controls["run_button"]:
    if not controls["symbol"]:
        st.error("Please enter a stock symbol.")
        st.stop()
    run_selected_workflow(controls, agents)

render_error_if_needed()

bundle = st.session_state.get("last_result_bundle")
if not bundle:
    render_start_message()
    st.stop()

render_results(bundle, agents)
