from typing import Any, Dict, Tuple

import pandas as pd
import streamlit as st

from app_components.constants import (
    CHART_PERIOD_OPTIONS,
    CHART_STYLE_OPTIONS,
    DEFAULT_CHART_LABEL,
    DEFAULT_CHART_STYLE,
)
from app_components.helpers import first_series, normalise_ohlcv_columns


"""Component for rendering live price charts with user-selectable periods and styles, including utilities to synchronise chart controls across multiple chart instances and trigger lightweight data refreshes without rerunning the full agent pipeline.
The render_chart function takes care of normalising incoming data, handling various column naming conventions, and rendering both price and volume charts with optional moving averages. The chart controls are designed to be independent from the main research pipeline, allowing users to adjust the chart view without triggering expensive data processing or model inference steps."""

def chart_preset(label: str) -> Tuple[str, str]:
    presets = {
        "1 Day": ("1d", "5m"),
        "7 Days": ("7d", "30m"),
        "30 Days": ("30d", "1d"),
        "6 Months": ("6mo", "1d"),
        "1 Year": ("1y", "1d"),
        "2 Years": ("2y", "1d"),
    }
    return presets.get(label, ("1y", "1d"))


def ensure_live_chart_state() -> None:
    """Keep chart controls independent from the full agent pipeline.

    The chart widgets live under the chart. Changing them should trigger only
    a lightweight chart refresh on the next Streamlit rerun, not a full
    Data/Analyst/Training/Risk pipeline run.
    """
    if st.session_state.get("live_chart_label") not in CHART_PERIOD_OPTIONS:
        st.session_state["live_chart_label"] = DEFAULT_CHART_LABEL
    if st.session_state.get("live_chart_style") not in CHART_STYLE_OPTIONS:
        st.session_state["live_chart_style"] = DEFAULT_CHART_STYLE
    st.session_state.setdefault("chart_refresh_nonce", 0)


def get_live_chart_selection() -> Tuple[str, str, str, str]:
    ensure_live_chart_state()
    label = st.session_state["live_chart_label"]
    period, interval = chart_preset(label)
    style = st.session_state["live_chart_style"]
    return label, period, interval, style


def sync_live_chart_value(source_key: str, target_key: str) -> None:
    value = st.session_state.get(source_key)
    if target_key == "live_chart_label" and value in CHART_PERIOD_OPTIONS:
        st.session_state[target_key] = value
    elif target_key == "live_chart_style" and value in CHART_STYLE_OPTIONS:
        st.session_state[target_key] = value


def request_chart_refresh() -> None:
    st.session_state["chart_refresh_nonce"] = int(st.session_state.get("chart_refresh_nonce", 0)) + 1
    st.session_state["chart_force_refresh_once"] = True


def render_live_chart_controls(location_key: str) -> None:
    """Render duplicate-safe chart controls below a chart.

    We use location-specific widget keys and sync them into the shared
    live_chart_* state, so the Overview chart and Chart tab can both have
    controls without Streamlit duplicate-key errors.
    """
    ensure_live_chart_state()
    label_key = f"{location_key}_chart_label_widget"
    style_key = f"{location_key}_chart_style_widget"

    # Synchronise visible widgets to the shared chart state before creation.
    if st.session_state.get(label_key) != st.session_state["live_chart_label"]:
        st.session_state[label_key] = st.session_state["live_chart_label"]
    if st.session_state.get(style_key) != st.session_state["live_chart_style"]:
        st.session_state[style_key] = st.session_state["live_chart_style"]

    c1, c2, c3 = st.columns([1.0, 1.2, 1.0])
    with c1:
        st.selectbox(
            "Chart period",
            CHART_PERIOD_OPTIONS,
            key=label_key,
            on_change=sync_live_chart_value,
            args=(label_key, "live_chart_label"),
            help="Changing this only refreshes the price chart. It does not rerun the full research pipeline.",
        )
    with c2:
        st.selectbox(
            "Chart style",
            CHART_STYLE_OPTIONS,
            key=style_key,
            on_change=sync_live_chart_value,
            args=(style_key, "live_chart_style"),
        )
    with c3:
        st.button(
            "Refresh chart data",
            key=f"{location_key}_refresh_chart_button",
            on_click=request_chart_refresh,
            use_container_width=True,
            help="Force a fresh yfinance chart download without rerunning the full pipeline.",
        )


def render_chart(df: pd.DataFrame, symbol: str, chart_style: str = "Line"):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No chart data is available yet.")
        return

    chart_df = normalise_ohlcv_columns(df)
    if chart_df.empty:
        st.info("No chart data is available after cleaning.")
        return

    # Preserve DatetimeIndex from historical_to_dataframe, but also support raw timestamp columns.
    if "timestamp" in chart_df.columns:
        ts = pd.to_datetime(first_series(chart_df, "timestamp"), errors="coerce")
        chart_df = chart_df.assign(timestamp=ts).dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    lower_cols = {str(c).lower(): c for c in chart_df.columns}
    close_col = lower_cols.get("close") or lower_cols.get("adj_close")
    if close_col is None:
        st.info("Chart data does not contain close prices.")
        return

    close_series = pd.to_numeric(first_series(chart_df, close_col), errors="coerce")
    chart_df = chart_df.assign(Close=close_series)

    if "Line" in chart_style:
        line_cols = ["Close"]
        if len(chart_df) >= 20:
            chart_df["MA20"] = chart_df["Close"].rolling(20).mean()
            line_cols.append("MA20")
        if len(chart_df) >= 50:
            chart_df["MA50"] = chart_df["Close"].rolling(50).mean()
            line_cols.append("MA50")
        st.line_chart(chart_df[line_cols].dropna(how="all"), height=430)
    else:
        st.line_chart(chart_df[["Close"]].dropna(), height=430)

    if "volume" in lower_cols:
        volume_col = lower_cols["volume"]
        vol = pd.to_numeric(first_series(chart_df, volume_col), errors="coerce")
        if vol.notna().any():
            with st.expander("Volume", expanded=False):
                st.bar_chart(vol.dropna(), height=180)