import json
import traceback
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agents.data_agent import DataAgent
from agents.validation_agent import ValidationAgent
from agents.historical_data_agent import HistoricalDataAgent
from agents.analyst_agent import AnalystAgent
from agents.training_agent import TrainingAgent
from agents.risk_agent import RiskAgent
from agents.strategist_agent import StrategistAgent
from agents.reward_agent import RewardAgent
from agents.storage_agent import StorageAgent
from agents.screener_agent import ScreenerAgent
from agents.evaluator_agent import EvaluatorAgent
from agents.execution_agent import ExecutionAgent

try:
    from agents.llm_report_agent import LLMReportAgent
except Exception:
    LLMReportAgent = None


load_dotenv()

st.set_page_config(
    page_title="Multi-Agent Stock Research System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------
def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def clean_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def clean_label(value: Any, fallback: str = "Unknown") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    replacements = {
        "POSITIVE_BUT_ENTRY_RISK": "Positive + Entry Risk",
        "WATCHLIST_BULLISH_ENTRY_RISK": "Bullish Watchlist",
        "BUY_WATCHLIST_OVERBOUGHT": "Bullish Watchlist / High Entry Risk",
        "BUY_WATCHLIST_ENTRY_RISK": "Bullish Watchlist / Entry Risk",
        "WAIT_FOR_PULLBACK_OR_CONFIRMATION": "Wait for Pullback / Confirmation",
        "MONITOR_AND_RESEARCH": "Monitor + Research",
        "RISK_REDUCTION_REVIEW": "Risk Reduction Review",
        "RESEARCH_FOR_POSSIBLE_ENTRY": "Research for Paper Entry",
        "NO_ACTION_DATA_OR_RISK_BLOCK": "No Action / Risk Block",
        "BUY_CANDIDATE": "Research Candidate",
        "SELL_RISK": "Risk Review",
        "HOLD": "Hold / Monitor",
        "BLOCKED": "Blocked",
        "High": "High",
        "Medium": "Medium",
        "Low": "Low",
    }
    return replacements.get(text, text.replace("_", " ").title())


def format_price(value: Any) -> str:
    try:
        value = float(value)
        return f"${value:,.2f}"
    except Exception:
        return "N/A"


def format_pct(value: Any) -> str:
    try:
        value = float(value)
        return f"{value * 100:.2f}%"
    except Exception:
        return "N/A"


def get_nested(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def call_agent_method(agent: Any, method_names: List[str], *args, **kwargs) -> Any:
    errors = []
    for method_name in method_names:
        if not hasattr(agent, method_name):
            continue
        method = getattr(agent, method_name)
        try:
            return method(**kwargs)
        except TypeError as e1:
            errors.append(f"{method_name} kwargs: {e1}")
            try:
                return method(*args)
            except Exception as e2:
                errors.append(f"{method_name} positional: {e2}")
        except Exception as e:
            errors.append(f"{method_name}: {e}")
    raise RuntimeError(f"No working method for {agent.__class__.__name__}. Tried {method_names}. Errors: {errors}")


def selected_price_from_quote(multi_quote: Dict[str, Any], validation_result: Optional[Dict[str, Any]] = None) -> Optional[float]:
    validation_result = validation_result or {}
    candidates = [
        validation_result.get("selected_price"),
        get_nested(validation_result, ["validation_for_next_agent", "selected_price"]),
        get_nested(multi_quote, ["primary_source", "current_price"]),
        get_nested(multi_quote, ["primary_quote", "current_price"]),
        get_nested(multi_quote, ["finnhub", "current_price"]),
        get_nested(multi_quote, ["secondary_source", "current_price"]),
    ]
    for item in candidates:
        try:
            if item is not None and float(item) > 0:
                return float(item)
        except Exception:
            continue
    return None


def historical_to_dataframe(historical_data: Dict[str, Any]) -> pd.DataFrame:
    if not isinstance(historical_data, dict) or not historical_data.get("success"):
        return pd.DataFrame()

    records = (
        historical_data.get("prices")
        or historical_data.get("records")
        or historical_data.get("price_records")
        or []
    )

    if isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        df = pd.DataFrame(records)

    if df.empty:
        return df

    # Normalise column names lightly while preserving original data.
    rename_map = {}
    for col in df.columns:
        lower = str(col).strip().lower().replace(" ", "_")
        if lower in ["datetime", "date", "timestamp", "index"]:
            rename_map[col] = "timestamp"
        elif lower in ["open", "high", "low", "close", "adj_close", "volume"]:
            rename_map[col] = lower
        elif lower == "adj close":
            rename_map[col] = "adj_close"
    df = df.rename(columns=rename_map)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.set_index("timestamp")

    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


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


def card(title: str, value: Any, note: Optional[str] = None):
    note_html = f"<div class='mini-note'>{note}</div>" if note else ""
    st.markdown(
        f"""
        <div class="soft-card">
            <h4>{title}</h4>
            <p>{value}</p>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pills(items: List[str]):
    html = "".join([f"<span class='status-pill'>{item}</span>" for item in items if item])
    st.markdown(html, unsafe_allow_html=True)


def render_chart(df: pd.DataFrame, symbol: str, chart_style: str = "Line"):
    if df.empty:
        st.info("No chart data is available yet.")
        return

    lower_cols = {str(c).lower(): c for c in df.columns}
    close_col = lower_cols.get("close") or lower_cols.get("adj_close")
    if not close_col:
        st.info("Chart data does not contain close prices.")
        return

    chart_df = df.copy()
    chart_df["Close"] = pd.to_numeric(chart_df[close_col], errors="coerce")

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
        vol = pd.to_numeric(chart_df[volume_col], errors="coerce")
        if vol.notna().any():
            with st.expander("Volume", expanded=False):
                st.bar_chart(vol.dropna(), height=180)


@st.cache_resource(show_spinner=False)
def load_agents():
    storage_agent = StorageAgent()
    agents = {
        "data": DataAgent(),
        "validation": ValidationAgent(),
        "historical": HistoricalDataAgent(storage_agent=storage_agent),
        "analyst": AnalystAgent(),
        "training": TrainingAgent(),
        "risk": RiskAgent(),
        "strategist": StrategistAgent(),
        "reward": RewardAgent(),
        "storage": storage_agent,
        "screener": ScreenerAgent(),
        "evaluator": EvaluatorAgent(),
        "execution": ExecutionAgent(),
    }
    if LLMReportAgent is not None:
        try:
            agents["llm"] = LLMReportAgent()
        except Exception:
            agents["llm"] = None
    else:
        agents["llm"] = None
    return agents


def build_portfolio_context(
    has_position: bool,
    shares: float,
    average_cost: float,
    current_price: Optional[float],
    user_intent: str,
) -> Dict[str, Any]:
    market_value = None
    unrealised_return = None
    if has_position and current_price and average_cost:
        try:
            market_value = float(shares) * float(current_price)
            unrealised_return = (float(current_price) - float(average_cost)) / float(average_cost)
        except Exception:
            pass
    return {
        "source": "streamlit_ui",
        "has_position": bool(has_position),
        "current_position": float(shares or 0.0) if has_position else 0.0,
        "shares": float(shares or 0.0) if has_position else 0.0,
        "avg_cost": float(average_cost or 0.0) if has_position else None,
        "market_value": market_value,
        "unrealised_return": unrealised_return,
        "user_intent": user_intent,
    }


def build_event_context(earnings_date_text: str, event_risk: str) -> Dict[str, Any]:
    text = str(earnings_date_text or "").strip()
    days_to_earnings = None
    if text:
        try:
            dt = pd.to_datetime(text).date()
            days_to_earnings = (dt - date.today()).days
        except Exception:
            days_to_earnings = None

    return {
        "source": "streamlit_ui",
        "earnings_date": text or None,
        "days_to_earnings": days_to_earnings,
        "event_risk": event_risk,
    }


def run_single_stock_pipeline(
    symbol: str,
    user_question: str,
    chart_label: str,
    chart_period: str,
    chart_interval: str,
    portfolio_context: Dict[str, Any],
    event_context: Dict[str, Any],
    force_retrain: bool,
    record_paper_decision: bool,
) -> Dict[str, Any]:
    agents = load_agents()

    # 1. Chart data follows the user-selected period.
    chart_historical_data = call_agent_method(
        agents["historical"],
        ["get_or_download_data", "run"],
        symbol,
        chart_period,
        chart_interval,
        symbol=symbol,
        period=chart_period,
        interval=chart_interval,
        force_refresh=False,
    )
    chart_df = historical_to_dataframe(chart_historical_data)

    # 2. Model/analysis data uses a stable daily window.
    model_historical_data = call_agent_method(
        agents["historical"],
        ["get_or_download_data", "run"],
        symbol,
        "1y",
        "1d",
        symbol=symbol,
        period="1y",
        interval="1d",
        force_refresh=False,
    )

    multi_quote = call_agent_method(
        agents["data"],
        ["get_multi_source_quote", "get_multi_source_quotes", "get_market_data", "fetch_market_data", "collect_market_data", "run"],
        symbol,
        symbol=symbol,
    )

    validation_result = call_agent_method(
        agents["validation"],
        ["validate_market_data", "validate_multi_source_data", "validate_quotes", "validate", "run"],
        multi_quote,
        multi_quote=multi_quote,
    )

    analysis_result = call_agent_method(
        agents["analyst"],
        ["analyse_market", "analyze_market", "analyse", "analyze", "run"],
        multi_quote,
        validation_result,
        model_historical_data,
        multi_quote=multi_quote,
        validation_result=validation_result,
        historical_data=model_historical_data,
    )

    training_result = call_agent_method(
        agents["training"],
        ["train_or_load_model", "train_or_load_signal_model", "load_or_train_model", "run"],
        model_historical_data,
        symbol,
        force_retrain,
        historical_data=model_historical_data,
        symbol=symbol,
        force_retrain=force_retrain,
    )

    signal_result = call_agent_method(
        agents["training"],
        ["generate_signal", "generate_trading_signal", "run_signal_model", "predict", "predict_signal"],
        analysis_result,
        training_result,
        symbol,
        analysis_result=analysis_result,
        training_result=training_result,
        symbol=symbol,
    )

    risk_result = call_agent_method(
        agents["risk"],
        ["assess_risk", "apply_risk_control", "adjust_risk", "evaluate_risk", "control_risk", "run"],
        signal_result,
        analysis_result,
        validation_result,
        signal_result=signal_result,
        analysis_result=analysis_result,
        validation_result=validation_result,
    )

    strategy_result = call_agent_method(
        agents["strategist"],
        ["plan_strategy", "generate_strategy", "run"],
        validation_result,
        analysis_result,
        training_result,
        signal_result,
        risk_result,
        None,
        None,
        portfolio_context,
        event_context,
        validation_result=validation_result,
        analysis_result=analysis_result,
        training_result=training_result,
        signal_result=signal_result,
        risk_result=risk_result,
        portfolio_context=portfolio_context,
        event_context=event_context,
    )

    entry_price = selected_price_from_quote(multi_quote, validation_result)
    reward_record_result = {}
    if record_paper_decision and entry_price:
        reward_record_result = call_agent_method(
            agents["reward"],
            ["record_pending_decision", "run"],
            symbol,
            entry_price,
            risk_result,
            symbol=symbol,
            entry_price=entry_price,
            risk_result=risk_result,
        )
    else:
        reward_record_result = {
            "success": True,
            "summary": "Paper decision recording was disabled for this UI run.",
        }

    auto_reward_update_result = call_agent_method(
        agents["reward"],
        ["auto_update_due_rewards"],
    )

    llm_report_result = {}
    if agents.get("llm"):
        try:
            llm_report_result = agents["llm"].generate_single_stock_report(
                user_question=user_question,
                validation_result=validation_result,
                analysis_result=analysis_result,
                training_result=training_result,
                signal_result=signal_result,
                risk_result=risk_result,
                strategy_result=strategy_result,
                reward_record_result=reward_record_result,
                auto_reward_update_result=auto_reward_update_result,
            )
        except Exception as exc:
            llm_report_result = {
                "success": False,
                "source": "error",
                "plain_language_report": f"LLM report failed: {exc}",
                "error": str(exc),
            }

    storage_result = {}
    try:
        storage_result = agents["storage"].record_pipeline_bundle(
            symbol=symbol,
            multi_quote=multi_quote,
            historical_data=model_historical_data,
            validation_result=validation_result,
            analysis_result=analysis_result,
            training_result=training_result,
            signal_result=signal_result,
            risk_result=risk_result,
            strategy_result=strategy_result,
            reward_record_result=reward_record_result,
            auto_reward_update_result=auto_reward_update_result,
            llm_report_result=llm_report_result,
        )
    except Exception as exc:
        storage_result = {"success": False, "error": str(exc)}

    pipeline_results = {
        "multi_quote": multi_quote,
        "historical_data": model_historical_data,
        "chart_historical_data": chart_historical_data,
        "validation_result": validation_result,
        "analysis_result": analysis_result,
        "training_result": training_result,
        "signal_result": signal_result,
        "risk_result": risk_result,
        "strategy_result": strategy_result,
        "reward_record_result": reward_record_result,
        "auto_reward_update_result": auto_reward_update_result,
        "llm_report_result": llm_report_result,
        "storage_result": storage_result,
    }

    execution_result = agents["execution"].record_interface_session(
        symbol=symbol,
        user_context={
            "user_question": user_question,
            "user_intent": portfolio_context.get("user_intent"),
            "has_position": portfolio_context.get("has_position"),
            "shares": portfolio_context.get("shares"),
            "average_cost": portfolio_context.get("avg_cost"),
            "portfolio_context": portfolio_context,
            "event_context": event_context,
            "query_modes": st.session_state.get("query_modes", []),
        },
        chart_context={
            "label": chart_label,
            "period": chart_period,
            "interval": chart_interval,
        },
        pipeline_results=pipeline_results,
        chart_df=chart_df,
        save_artifact=True,
    )
    pipeline_results["execution_result"] = execution_result
    pipeline_results["chart_df"] = chart_df
    return pipeline_results


def run_financial_news_summary(symbol: str, source_mode: str, lookback_days: int, max_news: int, pasted_text: str) -> Dict[str, Any]:
    agents = load_agents()
    llm = agents.get("llm")
    if not llm:
        return {"success": False, "summary": "LLM Report Agent is not available."}

    report_text = (pasted_text or "").strip()
    if not report_text:
        if source_mode == "financial":
            report_text = f"{symbol} financial report"
        elif source_mode == "news_and_financial":
            report_text = f"{symbol} news and financial report"
        else:
            report_text = f"{symbol} news"

    return llm.simplify_financial_text(
        report_text=report_text,
        user_question="Summarise only source-grounded company news or financial information for paper research.",
        source_mode=source_mode,
        ticker_override=symbol,
        lookback_days=lookback_days,
        max_news=max_news,
    )


# ---------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------
agents = load_agents()

st.title("📊 Human-in-the-Loop Multi-Agent Stock Research System")
st.caption(
    "Paper decision-support only. The system uses agents for data, validation, analysis, model signal, DQN risk control, strategy planning, memory, and LLM explanation."
)

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
            "News / report only",
            "Screener / watchlist only",
        ],
    )

    query_modes = st.multiselect(
        "What should the system run?",
        [
            "Single-stock agent pipeline",
            "Price chart",
            "Financial news / report summary",
            "Watchlist screener",
            "Evaluator dashboard",
            "Storage / session logs",
        ],
        default=["Single-stock agent pipeline", "Price chart"],
        key="query_modes",
    )

    chart_label = st.selectbox(
        "Chart period",
        ["1 Day", "7 Days", "30 Days", "6 Months", "1 Year", "2 Years"],
        index=4,
    )
    chart_period, chart_interval = chart_preset(chart_label)
    chart_style = st.selectbox("Chart style", ["Line + moving averages", "Line only"])

    st.divider()
    st.subheader("Portfolio context")
    has_position = st.checkbox("I currently hold this stock", value=("holding" in user_intent.lower()))
    shares = st.number_input("Shares / paper quantity", min_value=0.0, value=0.0, step=1.0, disabled=not has_position)
    average_cost = st.number_input("Average cost", min_value=0.0, value=0.0, step=1.0, disabled=not has_position)

    st.subheader("Event context")
    earnings_date_text = st.text_input("Next earnings date (optional)", placeholder="YYYY-MM-DD")
    event_risk = st.selectbox("Event risk", ["Unknown", "Low", "Medium", "High"], index=0)

    st.divider()
    force_retrain = st.checkbox("Force retrain signal model", value=False)
    record_paper_decision = st.checkbox("Record paper decision / memory", value=True)

    st.subheader("News / report options")
    source_mode = st.selectbox(
        "Source mode",
        ["auto", "news", "financial", "news_and_financial", "pasted_text"],
        index=0,
    )
    lookback_days = st.slider("News lookback days", min_value=3, max_value=60, value=14)
    max_news = st.slider("Max news items", min_value=3, max_value=20, value=8)
    pasted_financial_text = st.text_area(
        "Optional pasted report/news text",
        placeholder="Paste company news, earnings text, or leave blank to fetch source-grounded data.",
        height=120,
    )

    st.subheader("Screener options")
    default_universe = "AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, AMD, NFLX, AVGO, JPM, V, MA, WMT, DIS, INTC, QCOM, CSCO, ORCL"
    screener_symbols_text = st.text_area("Watchlist symbols", value=default_universe, height=90)
    top_n = st.slider("Top N", min_value=3, max_value=10, value=5)

    run_button = st.button("Run selected research", type="primary", use_container_width=True)


if not symbol and run_button:
    st.error("Please enter a stock symbol.")
    st.stop()


if run_button:
    with st.spinner("Running selected agents..."):
        try:
            portfolio_context = build_portfolio_context(
                has_position=has_position,
                shares=shares,
                average_cost=average_cost,
                current_price=None,
                user_intent=user_intent,
            )
            event_context = build_event_context(earnings_date_text, event_risk)

            result_bundle: Dict[str, Any] = {
                "symbol": symbol,
                "query_modes": query_modes,
                "chart_period": chart_period,
                "chart_interval": chart_interval,
                "user_intent": user_intent,
                "portfolio_context": portfolio_context,
                "event_context": event_context,
            }

            if "Single-stock agent pipeline" in query_modes:
                user_question = (
                    f"User intent: {user_intent}. "
                    f"Symbol: {symbol}. "
                    f"Explain the risk-aware paper decision, not a real trade."
                )
                pipeline_results = run_single_stock_pipeline(
                    symbol=symbol,
                    user_question=user_question,
                    chart_label=chart_label,
                    chart_period=chart_period,
                    chart_interval=chart_interval,
                    portfolio_context=portfolio_context,
                    event_context=event_context,
                    force_retrain=force_retrain,
                    record_paper_decision=record_paper_decision,
                )
                result_bundle.update(pipeline_results)

            elif "Price chart" in query_modes:
                chart_historical_data = call_agent_method(
                    agents["historical"],
                    ["get_or_download_data", "run"],
                    symbol,
                    chart_period,
                    chart_interval,
                    symbol=symbol,
                    period=chart_period,
                    interval=chart_interval,
                    force_refresh=False,
                )
                chart_df = historical_to_dataframe(chart_historical_data)
                execution_result = agents["execution"].record_interface_session(
                    symbol=symbol,
                    user_context={
                        "user_intent": user_intent,
                        "query_modes": query_modes,
                        "portfolio_context": portfolio_context,
                        "event_context": event_context,
                    },
                    chart_context={"label": chart_label, "period": chart_period, "interval": chart_interval},
                    pipeline_results={"chart_historical_data": chart_historical_data},
                    chart_df=chart_df,
                )
                result_bundle.update(
                    {
                        "chart_historical_data": chart_historical_data,
                        "chart_df": chart_df,
                        "execution_result": execution_result,
                    }
                )

            if "Financial news / report summary" in query_modes:
                news_result = run_financial_news_summary(
                    symbol=symbol,
                    source_mode=source_mode,
                    lookback_days=lookback_days,
                    max_news=max_news,
                    pasted_text=pasted_financial_text,
                )
                result_bundle["news_report_result"] = news_result

            if "Watchlist screener" in query_modes:
                screener_symbols = [clean_symbol(s) for s in screener_symbols_text.replace("\n", ",").split(",") if clean_symbol(s)]
                screener_result = agents["screener"].screen_universe(
                    symbols=screener_symbols,
                    top_n=top_n,
                    period="1y",
                    interval="1d",
                    save_to_storage=True,
                )
                result_bundle["screener_result"] = screener_result

            if "Evaluator dashboard" in query_modes:
                evaluation_result = agents["evaluator"].evaluate_history()
                result_bundle["evaluation_result"] = evaluation_result

            if "Storage / session logs" in query_modes:
                result_bundle["storage_summary"] = agents["storage"].get_storage_summary()
                result_bundle["recent_ui_sessions"] = agents["execution"].get_recent_ui_sessions(limit=10)
                result_bundle["recent_pipeline_runs"] = agents["storage"].get_recent_pipeline_runs(limit=10)

            st.session_state["last_result_bundle"] = result_bundle

        except Exception as exc:
            st.session_state["last_error"] = {
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }


# ---------------------------------------------------------------------
# Render results
# ---------------------------------------------------------------------
if "last_error" in st.session_state:
    st.error("The selected workflow crashed.")
    st.code(st.session_state["last_error"]["error"])
    with st.expander("Traceback"):
        st.code(st.session_state["last_error"]["traceback"])
    del st.session_state["last_error"]


bundle = st.session_state.get("last_result_bundle")

if not bundle:
    st.info("Enter a stock symbol, choose what you want to query, then click **Run selected research**.")
    st.markdown(
        """
        Suggested demo flow:
        1. Enter `AAPL` or `MSFT`.
        2. Choose **Single-stock agent pipeline** and **Price chart**.
        3. Add holding context if you want a portfolio-aware strategy.
        4. Add **Financial news / report summary** for source-grounded news.
        """
    )
    st.stop()


symbol = bundle.get("symbol", "")
chart_df = bundle.get("chart_df")
if chart_df is None:
    chart_df = historical_to_dataframe(bundle.get("chart_historical_data", {}))

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
    card("Symbol", symbol)
with summary_cols[1]:
    card("Price", format_price(entry_price))
with summary_cols[2]:
    analyst_label = clean_label(analysis_result.get("display_signal") or analysis_result.get("analyst_signal"))
    card("Analyst", analyst_label)
with summary_cols[3]:
    card("Model", clean_label(signal_result.get("display_signal") or signal_result.get("model_signal") or signal_result.get("signal")))
with summary_cols[4]:
    card("Risk", clean_label(risk_result.get("risk_level")))
with summary_cols[5]:
    card("Strategy", clean_label(strategy_result.get("strategy_action")))

status_items = [
    f"Chart: {bundle.get('chart_period', '')} / {bundle.get('chart_interval', '')}",
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
        guidance = {
            "strategy_action": strategy_result.get("strategy_action"),
            "strategy_level": strategy_result.get("strategy_level"),
            "strategy_confidence": strategy_result.get("strategy_confidence"),
            "position_guidance": strategy_result.get("position_guidance"),
            "leverage_guidance": strategy_result.get("leverage_guidance"),
            "risk_interpretation": risk_result.get("risk_interpretation"),
            "checklist": strategy_result.get("checklist") or strategy_result.get("conditions_to_reconsider"),
        }
        st.json(guidance)

    with right:
        st.markdown("#### Chart Preview")
        render_chart(chart_df, symbol, chart_style=chart_style)

with tab_chart:
    st.markdown(f"#### {symbol} Price Chart")
    render_chart(chart_df, symbol, chart_style=chart_style)
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
        "Signal Model": bundle.get("signal_result"),
        "Risk Agent": bundle.get("risk_result"),
        "Strategist Agent": bundle.get("strategy_result"),
        "Reward Agent": bundle.get("reward_record_result"),
        "Reward Update Agent": bundle.get("auto_reward_update_result"),
        "LLM Report Agent": bundle.get("llm_report_result"),
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
            card("Reward Win Rate", format_pct(metrics.get("reward_win_rate")))
        with cols[1]:
            card("Directional Win Rate", format_pct(metrics.get("directional_win_rate")))
        with cols[2]:
            card("Avg Reward", metrics.get("average_reward", "N/A"))
        with cols[3]:
            dqn_ready = get_nested(evaluation_result, ["dqn_summary", "ready_for_training"], None)
            card("DQN Ready", str(dqn_ready))
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
