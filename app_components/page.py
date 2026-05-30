import streamlit as st

"""Main page component for the multi-agent stock research system, responsible for configuring the Streamlit page, applying global styles, and orchestrating the rendering of the sidebar and main content area. This component serves as the entry point for the user interface, setting up the overall layout and visual design while delegating specific functionality to other components like the sidebar and charting modules."""

def configure_page() -> None:
    st.set_page_config(
        page_title="Multi-Agent Stock Research System",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_global_styles()


def apply_global_styles() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.3rem;
            padding-bottom: 2rem;
            max-width: 1450px;
        }
        .soft-card {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.14);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            min-height: 105px;
            color: #ffffff;
            background: linear-gradient(135deg, #334155 0%, #111827 100%);
            box-shadow: 0 10px 22px rgba(0, 0, 0, 0.20);
        }
        .soft-card::after {
            content: "";
            position: absolute;
            right: -30px;
            top: -40px;
            width: 110px;
            height: 110px;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.12);
        }
        .soft-card h4 {
            position: relative;
            z-index: 1;
            margin: 0 0 0.45rem 0;
            font-size: 0.88rem;
            color: rgba(255, 255, 255, 0.78);
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        .soft-card p {
            position: relative;
            z-index: 1;
            margin: 0;
            font-size: 1.08rem;
            line-height: 1.35;
            color: #ffffff;
            font-weight: 800;
            overflow-wrap: anywhere;
        }
        .card-blue { background: linear-gradient(135deg, #2563eb 0%, #172554 100%); border-color: rgba(96, 165, 250, 0.65); }
        .card-green { background: linear-gradient(135deg, #059669 0%, #064e3b 100%); border-color: rgba(52, 211, 153, 0.65); }
        .card-amber { background: linear-gradient(135deg, #d97706 0%, #451a03 100%); border-color: rgba(251, 191, 36, 0.70); }
        .card-red { background: linear-gradient(135deg, #dc2626 0%, #450a0a 100%); border-color: rgba(248, 113, 113, 0.70); }
        .card-purple { background: linear-gradient(135deg, #7c3aed 0%, #2e1065 100%); border-color: rgba(196, 181, 253, 0.70); }
        .card-teal { background: linear-gradient(135deg, #0891b2 0%, #083344 100%); border-color: rgba(103, 232, 249, 0.65); }
        .card-indigo { background: linear-gradient(135deg, #4f46e5 0%, #1e1b4b 100%); border-color: rgba(129, 140, 248, 0.70); }
        .card-neutral { background: linear-gradient(135deg, #475569 0%, #111827 100%); border-color: rgba(148, 163, 184, 0.45); }
        .mini-note {
            position: relative;
            z-index: 1;
            margin-top: 0.35rem;
            font-size: 0.86rem;
            color: rgba(255, 255, 255, 0.76);
        }
        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.55rem;
            margin: 0.1rem 0.2rem 0.1rem 0;
            border-radius: 999px;
            border: 1px solid rgba(255, 255, 255, 0.16);
            font-size: 0.82rem;
            color: rgba(255, 255, 255, 0.90);
            background: rgba(255,255,255,0.12);
        }
        .section-title {
            margin-top: 0.7rem;
            margin-bottom: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )