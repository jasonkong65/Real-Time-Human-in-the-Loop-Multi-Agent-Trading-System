import streamlit as st


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
            border: 1px solid rgba(49, 51, 63, 0.15);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            background: rgba(250, 250, 250, 0.65);
            min-height: 105px;
        }
        .soft-card h4 {
            margin: 0 0 0.4rem 0;
            font-size: 0.9rem;
            color: rgba(49, 51, 63, 0.70);
            font-weight: 650;
        }
        .soft-card p {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 700;
            overflow-wrap: anywhere;
        }
        .mini-note {
            font-size: 0.86rem;
            color: rgba(49, 51, 63, 0.72);
        }
        .status-pill {
            display: inline-block;
            padding: 0.25rem 0.55rem;
            margin: 0.1rem 0.2rem 0.1rem 0;
            border-radius: 999px;
            border: 1px solid rgba(49, 51, 63, 0.18);
            font-size: 0.82rem;
            background: rgba(255,255,255,0.75);
        }
        .section-title {
            margin-top: 0.7rem;
            margin-bottom: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )