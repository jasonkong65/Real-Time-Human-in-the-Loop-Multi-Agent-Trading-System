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
from agents.evaluator_agent import EvaluatorAgent
from agents.training_optimizer_agent import TrainingOptimizerAgent

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
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return default

        current = current.get(key)

        if current is None:
            return default

    return current


def build_fallback_signal_result(symbol, analysis_result):
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
    if LLMReportAgent is None:
        return None

    try:
        return LLMReportAgent()
    except Exception:
        return None


def is_direct_trading_question(text: str) -> bool:
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
        data_agent = DataAgent()
        validation_agent = ValidationAgent()
        historical_data_agent = HistoricalDataAgent()
        analyst_agent = AnalystAgent()
        training_agent = TrainingAgent()
        risk_agent = RiskAgent()
        reward_agent = RewardAgent()

        with st.spinner("Reward Agent is checking delayed rewards..."):
            auto_reward_update_result = reward_agent.auto_update_due_rewards()

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

        with st.spinner("Historical Data Agent is loading historical data..."):
            historical_data = historical_data_agent.get_or_download_data(
                symbol=clean_symbol,
                period="1y"
            )

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

        st.session_state["last_risk_result"] = risk_result
        st.session_state["last_symbol"] = clean_symbol

        with st.spinner("Reward Agent is recording pending delayed reward decision..."):
            reward_record_result = reward_agent.record_pending_decision(
                symbol=clean_symbol,
                entry_price=validation_result.get("selected_price"),
                risk_result=risk_result
            )

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
# Advanced Manual Q-learning feedback
# -----------------------------
st.divider()

with st.expander("Advanced / Debug: Manual Q-learning Feedback", expanded=False):
    st.info(
        "This optional section demonstrates how the Q-learning Risk Agent can update "
        "its Q-table from feedback. In normal use, the Reward Agent records paper "
        "decisions and automatically updates delayed rewards when later market prices "
        "become available."
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
# Evaluator Agent Dashboard
# -----------------------------
st.divider()

st.header("Evaluator Agent Dashboard")

st.info(
    "The Evaluator Agent reviews historical paper decisions, delayed reward updates, "
    "and Q-learning risk-control status. This supports evaluation, testing, and critical reflection."
)

if st.button("Run Evaluator Agent"):
    evaluator_agent = EvaluatorAgent()

    with st.spinner("Evaluator Agent is evaluating historical decisions and reward feedback..."):
        evaluation_result = evaluator_agent.evaluate_history()

    st.subheader("Evaluation Summary")

    if evaluation_result.get("success"):
        st.success(evaluation_result.get("summary", "Evaluation completed."))
    else:
        st.error("Evaluation failed.")

    col_1, col_2, col_3, col_4, col_5 = st.columns(5)

    with col_1:
        st.metric(
            "Data Readiness",
            evaluation_result.get("data_readiness_level", "Unknown")
        )

    with col_2:
        st.metric(
            "Readiness Score",
            evaluation_result.get("data_readiness_score", 0)
        )

    with col_3:
        st.metric(
            "Pending Decisions",
            evaluation_result.get("pending_count", 0)
        )

    with col_4:
        st.metric(
            "Completed Rewards",
            evaluation_result.get("completed_reward_count", 0)
        )

    with col_5:
        st.metric(
            "Q-table States",
            evaluation_result.get("q_table_summary", {}).get("q_state_count", 0)
        )

    st.subheader("Reward Metrics")

    reward_col_1, reward_col_2 = st.columns(2)

    with reward_col_1:
        st.metric(
            "Average Reward",
            evaluation_result.get("average_reward")
            if evaluation_result.get("average_reward") is not None
            else "N/A"
        )

    with reward_col_2:
        st.metric(
            "Average Future Return",
            evaluation_result.get("average_future_return")
            if evaluation_result.get("average_future_return") is not None
            else "N/A"
        )

    st.subheader("Performance Interpretation")

    performance_level = evaluation_result.get("performance_level", "Unknown")
    performance_interpretation = evaluation_result.get(
        "performance_interpretation",
        "No performance interpretation available."
    )

    if performance_level == "Needs improvement":
        st.warning(f"**Performance Level:** {performance_level}")
    elif performance_level == "Positive early performance":
        st.success(f"**Performance Level:** {performance_level}")
    else:
        st.info(f"**Performance Level:** {performance_level}")

    st.write(performance_interpretation)

    st.subheader("Signal and Risk Distributions")

    dist_col_1, dist_col_2 = st.columns(2)

    with dist_col_1:
        st.write("Completed Signal Distribution")
        completed_signal_distribution = evaluation_result.get(
            "completed_signal_distribution",
            {}
        )

        if completed_signal_distribution:
            st.dataframe(
                pd.DataFrame(
                    list(completed_signal_distribution.items()),
                    columns=["Signal", "Count"]
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No completed signal distribution available yet.")

    with dist_col_2:
        st.write("Risk Action Distribution")
        risk_action_distribution = evaluation_result.get(
            "risk_action_distribution",
            {}
        )

        if risk_action_distribution:
            st.dataframe(
                pd.DataFrame(
                    list(risk_action_distribution.items()),
                    columns=["Risk Action", "Count"]
                ),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No risk action distribution available yet.")

    st.subheader("Strengths, Limitations, and Suggestions")

    reflection_col_1, reflection_col_2, reflection_col_3 = st.columns(3)

    with reflection_col_1:
        st.write("Strengths")
        strengths = evaluation_result.get("strengths", [])
        if strengths:
            for item in strengths:
                st.write(f"- {item}")
        else:
            st.info("No strengths identified yet.")

    with reflection_col_2:
        st.write("Limitations")
        limitations = evaluation_result.get("limitations", [])
        if limitations:
            for item in limitations:
                st.write(f"- {item}")
        else:
            st.info("No limitations identified yet.")

    with reflection_col_3:
        st.write("Improvement Suggestions")
        suggestions = evaluation_result.get("suggestions", [])
        if suggestions:
            for item in suggestions:
                st.write(f"- {item}")
        else:
            st.info("No suggestions available yet.")

    st.subheader("Best and Worst Historical Cases")

    case_col_1, case_col_2 = st.columns(2)

    compact_case_columns = [
        "decision_id",
        "symbol",
        "entry_price",
        "latest_close",
        "future_return",
        "reward",
        "risk_action",
        "final_signal",
        "risk_level",
        "status",
        "updated_at_utc"
    ]

    with case_col_1:
        st.write("Best Cases")
        best_cases = evaluation_result.get("best_cases", [])

        if best_cases:
            best_cases_df = pd.DataFrame(best_cases)
            existing_cols = [
                col for col in compact_case_columns
                if col in best_cases_df.columns
            ]
            st.dataframe(
                best_cases_df[existing_cols] if existing_cols else best_cases_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No best cases available yet.")

    with case_col_2:
        st.write("Worst Cases")
        worst_cases = evaluation_result.get("worst_cases", [])

        if worst_cases:
            worst_cases_df = pd.DataFrame(worst_cases)
            existing_cols = [
                col for col in compact_case_columns
                if col in worst_cases_df.columns
            ]
            st.dataframe(
                worst_cases_df[existing_cols] if existing_cols else worst_cases_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No worst cases available yet.")

    with st.expander("Show Full Evaluator Agent JSON"):
        st.json(evaluation_result)


# -----------------------------
# Training Optimizer Dashboard
# -----------------------------
st.divider()

st.header("Training Optimizer Dashboard")

st.info(
    "The Training Optimizer Agent performs a lightweight grid search for the signal model. "
    "It uses historical price-derived features and compares different Random Forest settings. "
    "This responds to Evaluator Agent feedback when historical performance needs improvement."
)

optimizer_symbol = st.text_input(
    "Symbol for Training Optimizer",
    value=clean_symbol if clean_symbol else "AAPL",
    key="optimizer_symbol"
)

optimizer_period = st.selectbox(
    "Historical period for model optimization",
    options=["6mo", "1y", "2y"],
    index=1,
    key="optimizer_period"
)

apply_optimized_model = st.checkbox(
    "Apply optimized model to main signal model path",
    value=False,
    help=(
        "If unchecked, the optimizer saves to optimized_signal_model_SYMBOL.pkl. "
        "If checked, it overwrites signal_model_SYMBOL.pkl used by the main pipeline."
    )
)

if st.button("Run Training Optimizer"):
    optimizer_symbol_clean = optimizer_symbol.upper().strip()

    if not optimizer_symbol_clean:
        st.warning("Please enter a symbol for optimization.")

    else:
        historical_data_agent_for_optimizer = HistoricalDataAgent()
        training_optimizer_agent = TrainingOptimizerAgent()

        with st.spinner("Loading historical data for model optimization..."):
            optimizer_historical_data = historical_data_agent_for_optimizer.get_or_download_data(
                symbol=optimizer_symbol_clean,
                period=optimizer_period
            )

        with st.spinner("Training Optimizer Agent is running grid search..."):
            optimizer_result = training_optimizer_agent.optimize_from_historical_data(
                symbol=optimizer_symbol_clean,
                historical_data=optimizer_historical_data,
                validation_confidence_score=0.95,
                apply_to_main_model=apply_optimized_model
            )

        st.subheader("Training Optimizer Summary")

        if optimizer_result.get("success"):
            st.success(optimizer_result.get("summary", "Training optimization completed."))
        else:
            st.error(optimizer_result.get("summary", "Training optimization failed."))

        opt_col_1, opt_col_2, opt_col_3, opt_col_4 = st.columns(4)

        with opt_col_1:
            st.metric(
                "Best Test Accuracy",
                optimizer_result.get("best_test_accuracy", "N/A")
            )

        with opt_col_2:
            st.metric(
                "Baseline Accuracy",
                optimizer_result.get("baseline_accuracy", "N/A")
            )

        with opt_col_3:
            st.metric(
                "Improvement",
                optimizer_result.get("improvement_over_baseline", "N/A")
            )

        with opt_col_4:
            st.metric(
                "Training Samples",
                optimizer_result.get("num_samples", 0)
            )

        if optimizer_result.get("success"):
            st.subheader("Best Parameters")
            st.json(optimizer_result.get("best_params", {}))

            st.subheader("Performance Comment")
            st.info(optimizer_result.get("performance_comment", "No performance comment available."))

            st.subheader("Label Distribution")
            label_distribution = optimizer_result.get("label_distribution", {})

            if label_distribution:
                st.dataframe(
                    pd.DataFrame(
                        list(label_distribution.items()),
                        columns=["Label", "Count"]
                    ),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No label distribution available.")

            st.subheader("Grid Search Results")
            optimization_results = optimizer_result.get("optimization_results", [])

            if optimization_results:
                st.dataframe(
                    pd.DataFrame(optimization_results),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No grid-search results available.")

            st.subheader("Optimizer Suggestions")

            for suggestion in optimizer_result.get("suggestions", []):
                st.write(f"- {suggestion}")

            st.write(f"**Saved model path:** `{optimizer_result.get('saved_model_path')}`")
            st.write(f"**Metadata path:** `{optimizer_result.get('metadata_path')}`")

            if optimizer_result.get("applied_to_main_model"):
                st.warning(
                    "The optimized model was applied to the main signal model path. "
                    "Run the single-stock pipeline again to use this optimized model."
                )
            else:
                st.info(
                    "The optimized model was saved separately. "
                    "The main pipeline model was not overwritten."
                )

        with st.expander("Show Full Training Optimizer JSON"):
            st.json(optimizer_result)


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

    if not symbols:
        st.warning("Please enter at least one stock symbol for the screener.")

    else:
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