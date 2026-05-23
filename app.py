import traceback
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agents.data_agent import DataAgent
from agents.validation_agent import ValidationAgent
from agents.historical_data_agent import HistoricalDataAgent
from agents.analyst_agent import AnalystAgent
from agents.training_agent import TrainingAgent
from agents.risk_agent import RiskAgent
from agents.reward_agent import RewardAgent
from agents.screener_agent import ScreenerAgent

try:
    from agents.llm_report_agent import LLMReportAgent
except Exception:
    LLMReportAgent = None


load_dotenv()

st.set_page_config(
    page_title="Human-in-the-Loop Multi-Agent Trading System",
    layout="wide"
)


# -----------------------------
# Helper functions
# -----------------------------
def call_agent_method(agent, method_names, *args, **kwargs):
    """
    Robust helper for calling agent methods.

    It first tries keyword arguments, then positional arguments.
    This makes app.py more tolerant if agent method names are slightly different.
    """
    errors = []

    for method_name in method_names:
        if hasattr(agent, method_name):
            method = getattr(agent, method_name)

            try:
                return method(**kwargs)
            except TypeError as e1:
                errors.append(f"{method_name} kwargs failed: {str(e1)}")

                try:
                    return method(*args)
                except Exception as e2:
                    errors.append(f"{method_name} positional failed: {str(e2)}")

            except Exception as e:
                errors.append(f"{method_name} failed: {str(e)}")

    raise AttributeError(
        f"None of these methods worked for {agent.__class__.__name__}: "
        f"{method_names}. Errors: {errors}"
    )


def get_nested(data, keys, default=None):
    """
    Safely get nested dictionary values.
    """
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


def build_fallback_signal_result(symbol, analysis_result):
    """
    Fallback signal if TrainingAgent signal method is not available.
    """
    analyst_score = analysis_result.get("analyst_score", 0.5)
    volatility_level = analysis_result.get("volatility_level", "Unknown")

    try:
        analyst_score_float = float(analyst_score)
    except Exception:
        analyst_score_float = 0.5

    if analyst_score_float >= 0.7:
        model_signal = "BUY_CANDIDATE"
    elif analyst_score_float <= 0.4:
        model_signal = "SELL_RISK"
    else:
        model_signal = "HOLD"

    return {
        "success": True,
        "agent_goal": "Generate fallback trading signal from analyst features.",
        "signal_source": "fallback_rule",
        "model_signal": model_signal,
        "prediction_confidence": analyst_score_float,
        "confidence_level": "Medium",
        "agent_decision": "Used fallback rule because signal model method was unavailable.",
        "signal_for_next_agent": {
            "symbol": symbol,
            "signal": model_signal,
            "signal_source": "fallback_rule",
            "prediction_confidence": analyst_score_float,
            "confidence_level": "Medium",
            "analyst_score": analyst_score_float,
            "volatility_level": volatility_level
        },
        "summary": f"Fallback signal generated: {model_signal}."
    }


def build_fallback_risk_result(symbol, signal_result):
    """
    Fallback risk result if RiskAgent method is not available.
    """
    model_signal = (
        signal_result.get("model_signal")
        or get_nested(signal_result, ["signal_for_next_agent", "signal"], "HOLD")
    )

    if model_signal == "SELL_RISK":
        risk_level = "High"
    elif model_signal == "BUY_CANDIDATE":
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "success": True,
        "agent_goal": "Fallback risk adjustment.",
        "symbol": symbol,
        "original_signal": model_signal,
        "risk_action": "KEEP_SIGNAL",
        "final_signal": model_signal,
        "risk_level": risk_level,
        "agent_decision": (
            f"Fallback Risk Agent kept the signal as {model_signal}. "
            f"Estimated risk level is {risk_level}."
        ),
        "risk_for_next_agent": {
            "symbol": symbol,
            "original_signal": model_signal,
            "final_signal": model_signal,
            "risk_level": risk_level,
            "risk_action": "KEEP_SIGNAL",
            "explanation_for_llm": (
                f"The fallback risk layer kept the signal as {model_signal}. "
                f"Risk level is {risk_level}."
            )
        },
        "summary": f"Fallback risk result: {model_signal}, risk level {risk_level}."
    }


def make_llm_agent():
    """
    Safely create Groq LLM Report Agent.
    """
    if LLMReportAgent is None:
        return None

    try:
        return LLMReportAgent()
    except Exception:
        return None


def is_direct_trading_question(text: str) -> bool:
    """
    Detect whether the user is asking for direct buy/sell/position/leverage advice.
    The Financial Simplifier should not answer these questions.
    """
    if not text:
        return False

    lowered = text.lower()

    trading_keywords = [
        "should i buy",
        "should i sell",
        "can i buy",
        "can i sell",
        "do i buy",
        "do i sell",
        "buy now",
        "sell now",
        "is it worth buying",
        "worth buying",
        "worth selling",
        "clear position",
        "close position",
        "add position",
        "increase position",
        "reduce position",
        "add leverage",
        "use leverage",
        "margin",
        "加仓",
        "加倉",
        "清仓",
        "清倉",
        "买入",
        "買入",
        "卖出",
        "賣出",
        "加杠杆",
        "加槓桿",
        "加桿",
        "加杆",
        "值得买吗",
        "值得買嗎",
        "该买吗",
        "該買嗎",
        "该卖吗",
        "該賣嗎",
        "要不要买",
        "要不要買",
        "要不要卖",
        "要不要賣"
    ]

    return any(keyword in lowered for keyword in trading_keywords)


# -----------------------------
# Header
# -----------------------------
st.title("Human-in-the-Loop Multi-Agent Trading System")

st.subheader(
    "Data Agent + Validation Agent + Two-Stage Analyst Agent + "
    "Training Agent + Q-learning Risk Agent + Reward Agent + Groq Report Agent"
)

st.info(
    "This prototype collects market data from Finnhub and Alpha Vantage, validates "
    "multi-source consistency, performs two-stage market analysis, trains or loads a "
    "lightweight signal model, applies rule-based plus Q-learning risk control, records "
    "paper decisions for delayed reward updates, and uses Groq to explain results."
)


# -----------------------------
# Single-stock pipeline input
# -----------------------------
symbol = st.text_input("Enter stock symbol", value="AAPL")
clean_symbol = symbol.upper().strip()

user_question = st.text_input(
    "Ask the Groq Report Agent about this stock",
    value="Should I buy this stock now?"
)

force_retrain = st.checkbox("Force retrain signal model", value=False)

run_pipeline = st.button("Run Agent Pipeline")


# -----------------------------
# Single-stock pipeline
# -----------------------------
if run_pipeline:
    if not clean_symbol:
        st.error("Please enter a stock symbol.")
        st.stop()

    try:
        # -----------------------------
        # Agent initialization
        # -----------------------------
        data_agent = DataAgent()
        validation_agent = ValidationAgent()
        historical_data_agent = HistoricalDataAgent()
        analyst_agent = AnalystAgent()
        training_agent = TrainingAgent()
        risk_agent = RiskAgent()
        reward_agent = RewardAgent()

        # -----------------------------
        # 0. Auto delayed reward update
        # -----------------------------
        with st.spinner("Reward Agent is checking delayed rewards..."):
            auto_reward_update_result = reward_agent.auto_update_due_rewards()

        # -----------------------------
        # 1. Data Agent
        # -----------------------------
        with st.spinner("Data Agent is collecting live market data..."):
            multi_quote = call_agent_method(
                data_agent,
                [
                    "get_multi_source_quote",
                    "get_multi_source_quotes",
                    "get_market_data",
                    "fetch_market_data",
                    "collect_market_data",
                    "run"
                ],
                clean_symbol,
                symbol=clean_symbol
            )

        # -----------------------------
        # 2. Validation Agent
        # -----------------------------
        with st.spinner("Validation Agent is validating multi-source data..."):
            validation_result = call_agent_method(
                validation_agent,
                [
                    "validate_market_data",
                    "validate_multi_source_data",
                    "validate_multi_source",
                    "validate_quotes",
                    "validate",
                    "run"
                ],
                multi_quote,
                multi_quote=multi_quote
            )

        # -----------------------------
        # 3. Historical Data Agent
        # -----------------------------
        with st.spinner("Historical Data Agent is loading historical data..."):
            historical_data = historical_data_agent.get_or_download_data(
                symbol=clean_symbol,
                period="1y"
            )

        # -----------------------------
        # 4. Analyst Agent
        # -----------------------------
        with st.spinner("Analyst Agent is performing two-stage analysis..."):
            analysis_result = call_agent_method(
                analyst_agent,
                [
                    "analyse_market",
                    "analyze_market",
                    "run",
                    "analyse",
                    "analyze"
                ],
                multi_quote,
                validation_result,
                historical_data,
                multi_quote=multi_quote,
                validation_result=validation_result,
                historical_data=historical_data
            )

        # -----------------------------
        # 5. Training Agent
        # -----------------------------
        with st.spinner("Training Agent is loading or training signal model..."):
            training_result = call_agent_method(
                training_agent,
                [
                    "train_or_load_model",
                    "train_or_load_signal_model",
                    "load_or_train_model",
                    "train_model",
                    "run_training",
                    "run"
                ],
                historical_data,
                historical_data=historical_data,
                symbol=clean_symbol,
                force_retrain=force_retrain
            )

        # -----------------------------
        # 6. Signal Model
        # -----------------------------
        try:
            with st.spinner("Signal Model is generating a trading signal..."):
                signal_result = call_agent_method(
                    training_agent,
                    [
                        "generate_signal",
                        "predict_signal",
                        "generate_trading_signal",
                        "predict",
                        "run_signal_model"
                    ],
                    analysis_result,
                    training_result,
                    analysis_result=analysis_result,
                    training_result=training_result,
                    symbol=clean_symbol
                )
        except Exception:
            signal_result = build_fallback_signal_result(
                symbol=clean_symbol,
                analysis_result=analysis_result
            )

        # -----------------------------
        # 7. Risk Agent
        # -----------------------------
        try:
            with st.spinner("Risk Agent is applying rule-based and Q-learning risk control..."):
                risk_result = call_agent_method(
                    risk_agent,
                    [
                        "apply_risk_control",
                        "adjust_risk",
                        "evaluate_risk",
                        "run",
                        "control_risk"
                    ],
                    signal_result,
                    analysis_result,
                    validation_result,
                    signal_result=signal_result,
                    analysis_result=analysis_result,
                    validation_result=validation_result
                )
        except Exception:
            risk_result = build_fallback_risk_result(
                symbol=clean_symbol,
                signal_result=signal_result
            )

        # Store latest risk result for manual Q-learning feedback
        st.session_state["last_risk_result"] = risk_result
        st.session_state["last_symbol"] = clean_symbol

        # -----------------------------
        # 8. Reward Agent
        # -----------------------------
        with st.spinner("Reward Agent is recording pending delayed reward decision..."):
            reward_record_result = reward_agent.record_pending_decision(
                symbol=clean_symbol,
                entry_price=validation_result.get("selected_price"),
                risk_result=risk_result
            )

        # -----------------------------
        # 9. Groq LLM Recommendation / Report Agent
        # -----------------------------
        llm_single_stock_report = {
            "success": False,
            "llm_available": False,
            "plain_language_report": (
                "Groq Report Agent is not available. "
                "Please check agents/llm_report_agent.py, groq installation, "
                "and GROQ_API_KEY."
            ),
            "summary": "Groq Report Agent unavailable."
        }

        llm_report_agent = make_llm_agent()

        if llm_report_agent is not None:
            with st.spinner("Groq Report Agent is generating a plain-language explanation..."):
                llm_single_stock_report = llm_report_agent.generate_single_stock_report(
                    user_question=user_question,
                    validation_result=validation_result,
                    analysis_result=analysis_result,
                    training_result=training_result,
                    signal_result=signal_result,
                    risk_result=risk_result,
                    reward_record_result=reward_record_result,
                    auto_reward_update_result=auto_reward_update_result
                )

        # -----------------------------
        # Summary dashboard
        # -----------------------------
        st.header("Agent Decision Summary")

        final_signal = (
            risk_result.get("final_signal")
            or get_nested(risk_result, ["risk_for_next_agent", "final_signal"], "Unknown")
        )

        model_signal = (
            signal_result.get("model_signal")
            or get_nested(signal_result, ["signal_for_next_agent", "signal"], "Unknown")
        )

        risk_level = (
            risk_result.get("risk_level")
            or get_nested(risk_result, ["risk_for_next_agent", "risk_level"], "Unknown")
        )

        summary_cols = st.columns(7)

        with summary_cols[0]:
            st.metric("Symbol", clean_symbol)

        with summary_cols[1]:
            st.metric("Validation", validation_result.get("confidence", "Unknown"))

        with summary_cols[2]:
            st.metric("Next Action", validation_result.get("next_action", "Unknown"))

        with summary_cols[3]:
            st.metric("Analyst Signal", analysis_result.get("analyst_signal", "Unknown"))

        with summary_cols[4]:
            st.metric("Model Signal", model_signal)

        with summary_cols[5]:
            st.metric("Final Signal", final_signal)

        with summary_cols[6]:
            st.metric("Risk Level", risk_level)

        st.write(f"**Validation Decision:** {validation_result.get('agent_decision', 'N/A')}")
        st.write(f"**Analyst Decision:** {analysis_result.get('agent_decision', 'N/A')}")
        st.write(f"**Signal Model Decision:** {signal_result.get('agent_decision', 'N/A')}")
        st.write(f"**Risk Decision:** {risk_result.get('agent_decision', 'N/A')}")
        st.write(f"**Reward Agent:** {reward_record_result.get('summary', 'N/A')}")
        st.write(f"**Groq Report Agent:** {llm_single_stock_report.get('summary', 'N/A')}")

        if validation_result.get("summary"):
            st.success(validation_result.get("summary"))

        if analysis_result.get("summary"):
            st.success(analysis_result.get("summary"))

        if signal_result.get("summary"):
            st.success(signal_result.get("summary"))

        if risk_result.get("summary"):
            st.success(risk_result.get("summary"))

        if reward_record_result.get("summary"):
            if reward_record_result.get("success"):
                st.success(reward_record_result.get("summary"))
            else:
                st.warning(reward_record_result.get("summary"))

        # -----------------------------
        # Detailed outputs
        # -----------------------------
        st.header("1. Data Agent Output")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Finnhub Quote")
            st.json(
                multi_quote.get("finnhub")
                or multi_quote.get("finnhub_quote")
                or multi_quote.get("primary")
                or {}
            )

        with col2:
            st.subheader("Alpha Vantage Quote")
            st.json(
                multi_quote.get("alpha_vantage")
                or multi_quote.get("alpha_vantage_quote")
                or multi_quote.get("secondary")
                or {}
            )

        with st.expander("Show Historical Data Summary"):
            st.json({
                "source": historical_data.get("source"),
                "success": historical_data.get("success"),
                "symbol": historical_data.get("symbol"),
                "num_price_records": len(historical_data.get("prices", [])),
                "error": historical_data.get("error")
            })

            prices = historical_data.get("prices", [])
            if prices:
                st.write("Latest historical price records:")
                st.dataframe(pd.DataFrame(prices).tail(5), use_container_width=True)

        st.header("2. Validation Agent Output")
        st.json(validation_result)

        st.header("3. Analyst Agent Output")
        st.json(analysis_result)

        st.header("4. Training Agent Output")
        st.json(training_result)

        st.header("5. Signal Model Output")
        st.json(signal_result)

        st.header("6. Risk Agent Output")
        st.json(risk_result)

        st.header("7. Reward Agent Output")
        st.json(reward_record_result)

        st.header("8. Auto Delayed Reward Update Output")
        st.json(auto_reward_update_result)

        st.header("9. Groq Recommendation / Report Agent Output")

        if llm_single_stock_report.get("llm_available"):
            st.success("Groq explanation generated successfully.")
        else:
            st.warning(
                "Groq API was not available. The system used a fallback explanation "
                "or skipped LLM output. Check GROQ_API_KEY and groq installation."
            )

        st.markdown(llm_single_stock_report.get("plain_language_report", ""))

        with st.expander("Show Groq Report Agent JSON"):
            st.json(llm_single_stock_report)

        with st.expander("Show Validation Reasoning Steps"):
            for step in validation_result.get("reasoning_steps", []):
                st.write(f"- {step}")

        with st.expander("Show Analyst Reasoning Steps"):
            for step in analysis_result.get("reasoning_steps", []):
                st.write(f"- {step}")

        with st.expander("Show Risk Agent Reasoning Steps"):
            for step in risk_result.get("reasoning_steps", []):
                st.write(f"- {step}")

        if "stage_1_quote_analysis" in analysis_result:
            with st.expander("Show Stage 1 Quote-Level Analysis"):
                st.json(analysis_result.get("stage_1_quote_analysis"))

        if "stage_2_historical_analysis" in analysis_result:
            with st.expander("Show Stage 2 Historical Analysis"):
                st.json(analysis_result.get("stage_2_historical_analysis"))

    except Exception as e:
        st.error("The agent pipeline crashed.")
        st.exception(e)
        st.code(traceback.format_exc())


# -----------------------------
# Manual Q-learning feedback
# -----------------------------
st.divider()

st.header("Manual Q-learning Feedback Demo")

st.info(
    "This is a manual fallback demo for Q-learning feedback. "
    "The system also records paper decisions and can automatically update delayed rewards "
    "when later market prices become available."
)

if "last_risk_result" not in st.session_state:
    st.warning("Run the Agent Pipeline first before using manual Q-learning feedback.")
else:
    last_risk_result = st.session_state["last_risk_result"]
    last_symbol = st.session_state.get("last_symbol", "UNKNOWN")

    st.write(f"Latest stored symbol: **{last_symbol}**")
    st.write(f"Latest final signal: **{last_risk_result.get('final_signal', 'N/A')}**")
    st.write(f"Latest risk action: **{last_risk_result.get('risk_action', 'N/A')}**")
    st.write(f"Latest risk level: **{last_risk_result.get('risk_level', 'N/A')}**")
    st.write(f"Latest Q-state: `{last_risk_result.get('q_state', 'N/A')}`")

    manual_future_return = st.number_input(
        "Enter simulated future return for manual Q-learning update",
        value=0.0,
        step=0.01,
        format="%.3f",
        help=(
            "Example: 0.03 means the stock increased by 3%; "
            "-0.03 means the stock decreased by 3%."
        )
    )

    if st.button("Update Risk Q-table Manually"):
        feedback_risk_agent = RiskAgent()

        update_result = feedback_risk_agent.update_from_feedback(
            risk_result=last_risk_result,
            future_return=manual_future_return
        )

        st.subheader("Manual Q-learning Update Output")

        if update_result.get("success"):
            st.success(update_result.get("summary", "Q-table updated successfully."))
        else:
            st.error(update_result.get("summary", "Q-table update failed."))

        st.json(update_result)


# -----------------------------
# S&P-style Screener
# -----------------------------
st.divider()

st.header("S&P-style Market Screener Prototype")

st.info(
    "This screener ranks a configurable S&P-style stock universe into "
    "Top Buy Candidates and Highest Risk / Caution Candidates. "
    "It is a lightweight prototype and does not scan the entire market."
)

default_universe_text = (
    "AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, AVGO, JPM, V, "
    "MA, UNH, HD, COST, NFLX, AMD, CRM, ADBE, PEP, KO, "
    "BAC, WMT, DIS, MCD, CSCO, INTC, QCOM, TXN, AMAT, ORCL"
)

universe_text = st.text_area(
    "S&P-style stock universe",
    value=default_universe_text,
    height=100
)

top_n = st.slider(
    "Number of candidates to show",
    min_value=3,
    max_value=10,
    value=10
)

screen_period = st.selectbox(
    "Historical period for screening",
    options=["6mo", "1y", "2y"],
    index=1
)

screener_question = st.text_input(
    "Ask the Groq Report Agent about the screener result",
    value="Which stocks look strongest, which stocks need caution, and why?",
    key="screener_question"
)

if st.button("Run S&P-style Screener"):
    symbols = [
        s.strip().upper()
        for s in universe_text.split(",")
        if s.strip()
    ]

    screener_agent = ScreenerAgent()

    with st.spinner("Screener Agent is scanning the S&P-style universe..."):
        screener_result = screener_agent.screen_universe(
            symbols=symbols,
            top_n=top_n,
            period=screen_period
        )

    st.subheader("Screener Summary")
    st.success(screener_result.get("summary", "Screener completed."))

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.metric("Universe Size", screener_result.get("universe_size", 0))

    with col_b:
        st.metric("Scanned Successfully", screener_result.get("scanned_count", 0))

    with col_c:
        st.metric("Failed Symbols", screener_result.get("failed_count", 0))

    st.subheader("Top Buy Candidates for Further Research")

    top_buy_df = pd.DataFrame(screener_result.get("top_buy_candidates", []))

    if not top_buy_df.empty:
        top_buy_df = top_buy_df.sort_values(
            by="buy_score",
            ascending=False
        ).reset_index(drop=True)

        if "rank" not in top_buy_df.columns:
            top_buy_df.insert(0, "rank", range(1, len(top_buy_df) + 1))

        buy_display_cols = [
            "rank",
            "symbol",
            "buy_score",
            "risk_score",
            "screen_signal",
            "return_5",
            "return_20",
            "ma_gap",
            "volatility_20",
            "rsi_14",
            "reason"
        ]

        existing_buy_cols = [
            col for col in buy_display_cols
            if col in top_buy_df.columns
        ]

        st.dataframe(
            top_buy_df[existing_buy_cols],
            use_container_width=True,
            hide_index=True
        )

        if "screen_signal" in top_buy_df.columns:
            overbought_df = top_buy_df[
                top_buy_df["screen_signal"] == "BUY_WATCHLIST_OVERBOUGHT"
            ]

            if not overbought_df.empty:
                overbought_symbols = ", ".join(overbought_df["symbol"].tolist())
                st.warning(
                    "Some high-ranked stocks are marked as BUY_WATCHLIST_OVERBOUGHT "
                    f"because their RSI is high: {overbought_symbols}. "
                    "They may have strong momentum but higher entry risk."
                )
    else:
        st.warning("No buy candidates were generated.")

    st.subheader("Highest Risk / Caution Candidates")

    top_risk_df = pd.DataFrame(
        screener_result.get(
            "highest_risk_candidates",
            screener_result.get("top_sell_risk", [])
        )
    )

    if not top_risk_df.empty:
        top_risk_df = top_risk_df.sort_values(
            by="risk_score",
            ascending=False
        ).reset_index(drop=True)

        if "rank" not in top_risk_df.columns:
            top_risk_df.insert(0, "rank", range(1, len(top_risk_df) + 1))

        risk_display_cols = [
            "rank",
            "symbol",
            "risk_score",
            "buy_score",
            "screen_signal",
            "return_5",
            "return_20",
            "ma_gap",
            "volatility_20",
            "rsi_14",
            "reason"
        ]

        existing_risk_cols = [
            col for col in risk_display_cols
            if col in top_risk_df.columns
        ]

        st.dataframe(
            top_risk_df[existing_risk_cols],
            use_container_width=True,
            hide_index=True
        )

        if "screen_signal" in top_risk_df.columns:
            strong_sell_risk_df = top_risk_df[
                top_risk_df["screen_signal"] == "SELL_RISK"
            ]

            if strong_sell_risk_df.empty:
                st.info(
                    "No strong SELL_RISK signal was detected in this run. "
                    "This table shows the relatively highest-risk stocks within the selected universe."
                )
    else:
        st.warning("No caution candidates were generated.")

    st.subheader("Groq Screener Explanation")

    llm_screener_report = {
        "success": False,
        "llm_available": False,
        "plain_language_report": (
            "Groq Report Agent is not available. "
            "Please check agents/llm_report_agent.py, groq installation, and GROQ_API_KEY."
        ),
        "summary": "Groq Screener Report unavailable."
    }

    llm_report_agent = make_llm_agent()

    if llm_report_agent is not None:
        with st.spinner("Groq Report Agent is explaining the screener result..."):
            llm_screener_report = llm_report_agent.generate_screener_report(
                user_question=screener_question,
                screener_result=screener_result
            )

    if llm_screener_report.get("llm_available"):
        st.success("Groq screener explanation generated successfully.")
    else:
        st.warning(
            "Groq API was not available. The system used a fallback explanation "
            "or skipped LLM output."
        )

    st.markdown(llm_screener_report.get("plain_language_report", ""))

    if screener_result.get("failed_count", 0) > 0:
        with st.expander("Show Failed Symbols"):
            st.json(screener_result.get("failed_symbols", []))

    with st.expander("Show Groq Screener Report JSON"):
        st.json(llm_screener_report)

    with st.expander("Show Full Screener Result JSON"):
        st.json(screener_result)


# -----------------------------
# Financial Report / News Simplifier
# -----------------------------
st.divider()

st.header("Groq Financial Report / News Simplifier")

st.info(
    "This section only simplifies financial reports, earnings news, company announcements, "
    "or market commentary. It does not answer direct buy/sell/clear-position/leverage questions. "
    "For stock decision questions, please use the single-stock pipeline above."
)

st.caption(
    "Example input: Apple reported stronger-than-expected quarterly revenue, "
    "but management warned that China demand remained weak and services growth slowed."
)

financial_text = st.text_area(
    "Paste financial report, earnings news, company announcement, or market commentary",
    height=180,
    placeholder=(
        "Paste actual report/news text here. "
        "Example: Apple reported quarterly revenue growth but warned about weaker China demand..."
    )
)

financial_question = st.text_input(
    "Question for report/news simplification",
    value=(
        "Please simplify this report/news text and identify the main positive signals, "
        "risks, and possible market impact."
    )
)

if st.button("Simplify Financial Report / News"):
    combined_text_for_check = f"{financial_text} {financial_question}"

    if not financial_text.strip():
        st.warning(
            "Please paste a financial report, earnings news, company announcement, "
            "or market commentary before using this section."
        )

    elif is_direct_trading_question(combined_text_for_check):
        st.warning(
            "This section is only for simplifying reports or news. "
            "It does not answer direct buy/sell, clear-position, add-position, or leverage questions. "
            "Please use the single-stock pipeline above for stock decision questions."
        )

        st.info(
            "For example, instead of asking 'Should I buy AAPL now?', paste a news/report paragraph such as: "
            "'Apple reported stronger iPhone sales, but management warned about weaker China demand.'"
        )

    else:
        llm_report_agent = make_llm_agent()

        if llm_report_agent is None:
            st.warning(
                "Groq Report Agent is not available. Please check agents/llm_report_agent.py, "
                "groq installation, and GROQ_API_KEY."
            )

        else:
            with st.spinner("Groq Report Agent is simplifying the report/news text..."):
                simplification_result = llm_report_agent.simplify_financial_text(
                    report_text=financial_text,
                    user_question=financial_question
                )

            if simplification_result.get("llm_available"):
                st.success("Financial report/news text simplified successfully.")
            else:
                st.warning(
                    "Groq API was not available. The system used a fallback explanation "
                    "or skipped LLM output. Please check GROQ_API_KEY and groq installation."
                )

            st.markdown(simplification_result.get("plain_language_report", ""))

            with st.expander("Show Financial Text Simplification JSON"):
                st.json(simplification_result)