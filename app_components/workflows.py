import traceback
from typing import Any, Dict

import streamlit as st

from app_components.agent_factory import load_agents
from app_components.contexts import build_event_context, build_portfolio_context
from app_components.helpers import (
    call_agent_method,
    clean_symbol,
    historical_to_dataframe,
    selected_price_from_quote,
)

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


def run_selected_workflow(controls: Dict[str, Any], agents: Dict[str, Any]) -> None:
    """Run the modules chosen in the sidebar and store results in session_state."""
    symbol = controls["symbol"]
    query_modes = controls["query_modes"]
    chart_label = controls["chart_label"]
    chart_period = controls["chart_period"]
    chart_interval = controls["chart_interval"]
    user_intent = controls["user_intent"]

    with st.spinner("Running selected agents..."):
        try:
            portfolio_context = build_portfolio_context(
                has_position=controls["has_position"],
                shares=controls["shares"],
                average_cost=controls["average_cost"],
                current_price=None,
                user_intent=user_intent,
            )
            event_context = build_event_context(controls["earnings_date_text"], controls["event_risk"])

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
                    f"User intent: {user_intent}. Symbol: {symbol}. "
                    "Explain the risk-aware paper decision, not a real trade."
                )
                result_bundle.update(
                    run_single_stock_pipeline(
                        symbol=symbol,
                        user_question=user_question,
                        chart_label=chart_label,
                        chart_period=chart_period,
                        chart_interval=chart_interval,
                        portfolio_context=portfolio_context,
                        event_context=event_context,
                        force_retrain=controls["force_retrain"],
                        record_paper_decision=controls["record_paper_decision"],
                    )
                )

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
                result_bundle.update({
                    "chart_historical_data": chart_historical_data,
                    "chart_df": chart_df,
                    "execution_result": execution_result,
                })

            if "Financial news / report summary" in query_modes:
                result_bundle["news_report_result"] = run_financial_news_summary(
                    symbol=symbol,
                    source_mode=controls["source_mode"],
                    lookback_days=controls["lookback_days"],
                    max_news=controls["max_news"],
                    pasted_text=controls["pasted_financial_text"],
                )

            if "Watchlist screener" in query_modes:
                screener_symbols = [
                    clean_symbol(s)
                    for s in controls["screener_symbols_text"].replace("\n", ",").split(",")
                    if clean_symbol(s)
                ]
                try:
                    screener_result = agents["screener"].screen_universe(
                        symbols=screener_symbols,
                        top_n=controls["top_n"],
                        period="1y",
                        interval="1d",
                        save_to_storage=True,
                    )
                except Exception as exc:
                    screener_result = {
                        "success": False,
                        "agent": "Screener Agent",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "summary": "Watchlist screener failed, but the rest of the app can still be reviewed.",
                    }
                result_bundle["screener_result"] = screener_result
                result_bundle["screener_report_result"] = build_screener_report(agents, screener_result)

            if "Evaluator dashboard" in query_modes:
                result_bundle["evaluation_result"] = agents["evaluator"].evaluate_history()

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


def build_screener_report(agents: Dict[str, Any], screener_result: Dict[str, Any]) -> Dict[str, Any]:
    """Explain screener results with Groq when available, otherwise use a local summary."""
    if agents.get("llm") and isinstance(screener_result, dict):
        try:
            return agents["llm"].generate_screener_report(
                user_question=(
                    "Explain the watchlist screener output for paper research. "
                    "Summarise the top candidates, caution candidates, and key risks. "
                    "Do not give direct investment advice."
                ),
                screener_result=screener_result,
            )
        except Exception as exc:
            return {
                "success": False,
                "agent": "Groq Report Agent",
                "report_type": "screener_explanation",
                "source": "error",
                "plain_language_report": f"Screener report generation failed: {exc}",
                "error": str(exc),
            }

    top = screener_result.get("top_buy_candidates") or []
    caution = screener_result.get("highest_risk_candidates") or screener_result.get("top_sell_risk") or []
    top_names = ", ".join([str(x.get("symbol")) for x in top[:5] if isinstance(x, dict) and x.get("symbol")]) or "none"
    caution_names = ", ".join([str(x.get("symbol")) for x in caution[:5] if isinstance(x, dict) and x.get("symbol")]) or "none"
    return {
        "success": True,
        "agent": "Local Report Fallback",
        "report_type": "screener_explanation",
        "source": "local_fallback_no_llm_agent",
        "plain_language_report": (
            f"**Direct answer**\nThe strongest watchlist names for further research are: {top_names}.\n\n"
            f"**Caution names**\nThe names needing more caution are: {caution_names}.\n\n"
            "**Risk note**\nThis is a watchlist screener, not a direct buy/sell instruction."
        ),
    }
