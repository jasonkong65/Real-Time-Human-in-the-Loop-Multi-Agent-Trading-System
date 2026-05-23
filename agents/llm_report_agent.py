import os
import json
from typing import Dict, Any, Optional

from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
except Exception:
    Groq = None


class LLMReportAgent:
    """
    Groq-powered Recommendation / Report Agent.

    Role:
    - Explain structured outputs from other agents in plain language.
    - Answer user questions based on pipeline results.
    - Summarise screener results.
    - Simplify pasted financial/news/report text.

    Safety:
    - The LLM does not execute real trades.
    - The LLM does not override the Signal Model or Risk Agent.
    - The output is framed as paper decision support, not guaranteed financial advice.
    """

    def __init__(self, model: Optional[str] = None):
        self.api_key = os.getenv("GROQ_API_KEY")

        self.model = model or os.getenv(
            "GROQ_MODEL",
            "llama-3.1-8b-instant"
        )

        self.client = None

        if Groq is not None and self.api_key:
            self.client = Groq(api_key=self.api_key)

    # --------------------------------------------------
    # Utility helpers
    # --------------------------------------------------
    def _safe_json(self, data: Any, max_chars: int = 7000) -> str:
        try:
            text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        except Exception:
            text = str(data)

        if len(text) > max_chars:
            return text[:max_chars] + "\n... [truncated]"
        return text

    def _call_llm(self, system_message: str, user_message: str) -> Dict[str, Any]:
        """
        Call Groq API. If the API key is missing or the call fails,
        return a safe fallback instead of crashing Streamlit.
        """

        if self.client is None:
            return {
                "success": False,
                "llm_available": False,
                "provider": "groq",
                "model": self.model,
                "error": (
                    "Groq client is not available. Please check GROQ_API_KEY "
                    "and make sure the groq package is installed."
                ),
                "output_text": None
            }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": system_message
                    },
                    {
                        "role": "user",
                        "content": user_message
                    }
                ],
                temperature=0.2,
                max_tokens=1200
            )

            output_text = response.choices[0].message.content

            return {
                "success": True,
                "llm_available": True,
                "provider": "groq",
                "model": self.model,
                "output_text": output_text
            }

        except Exception as e:
            error_text = str(e)

            if "429" in error_text or "rate limit" in error_text.lower() or "quota" in error_text.lower():
                friendly_error = (
                    "Groq API call failed because the current API quota or rate limit was reached. "
                    "The API key may be valid, but the free-tier limit has been exceeded."
                )
            elif "api_key" in error_text.lower() or "invalid" in error_text.lower() or "unauthorized" in error_text.lower():
                friendly_error = (
                    "Groq API call failed because the API key may be missing or invalid. "
                    "Check GROQ_API_KEY in the .env file."
                )
            else:
                friendly_error = f"Groq API call failed: {error_text}"

            return {
                "success": False,
                "llm_available": False,
                "provider": "groq",
                "model": self.model,
                "error": friendly_error,
                "raw_error": error_text,
                "output_text": None
            }

    def _base_system_prompt(self) -> str:
        return """
You are a Groq-powered Recommendation / Report Agent inside a human-in-the-loop multi-agent trading decision support prototype.

Important safety rules:
1. Do not claim to provide guaranteed financial advice.
2. Do not tell the user to definitely buy, sell, clear a position, add leverage, or enter a real trade.
3. Explain that outputs are for paper decision support and further research.
4. Base your answer only on the provided structured agent outputs.
5. Treat the Risk Agent as the final safety layer.
6. Be clear, practical, and concise.
7. If the signal is risky or confidence is low/medium, recommend caution.
8. For leverage questions, be conservative. Do not recommend leverage under high risk, low confidence, medium confidence, or SELL_RISK.
9. Use simple language suitable for a student demo.

Output format:
- Direct answer
- Evidence from agents
- Risk warning
- Strategy suggestion
- Not financial advice disclaimer
"""

    # --------------------------------------------------
    # Single-stock report
    # --------------------------------------------------
    def generate_single_stock_report(
        self,
        user_question: str,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        training_result: Dict[str, Any],
        signal_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        reward_record_result: Optional[Dict[str, Any]] = None,
        auto_reward_update_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate natural-language answer for one stock based on full pipeline outputs.
        """

        symbol = (
            risk_result.get("risk_for_next_agent", {}).get("symbol")
            or risk_result.get("symbol")
            or signal_result.get("signal_for_next_agent", {}).get("symbol")
            or analysis_result.get("symbol")
            or validation_result.get("validation_for_next_agent", {}).get("symbol")
            or "UNKNOWN"
        )

        structured_context = {
            "symbol": symbol,
            "validation_result": validation_result,
            "analysis_result": analysis_result,
            "training_result": training_result,
            "signal_result": signal_result,
            "risk_result": risk_result,
            "reward_record_result": reward_record_result or {},
            "auto_reward_update_result": auto_reward_update_result or {}
        }

        system_message = self._base_system_prompt()

        user_message = f"""
User question:
{user_question}

Structured multi-agent pipeline output:
{self._safe_json(structured_context)}

Please answer the user's question using only this information.
"""

        llm_result = self._call_llm(system_message, user_message)

        if llm_result.get("success"):
            report_text = llm_result["output_text"]
            source = "groq_llm"
        else:
            report_text = self._fallback_single_stock_report(
                user_question=user_question,
                validation_result=validation_result,
                analysis_result=analysis_result,
                signal_result=signal_result,
                risk_result=risk_result
            )
            source = "fallback_rule"

        return {
            "success": True,
            "agent_goal": "Explain the single-stock multi-agent decision in natural language.",
            "report_type": "single_stock_report",
            "symbol": symbol,
            "source": source,
            "provider": "groq",
            "model": self.model,
            "llm_available": llm_result.get("llm_available", False),
            "llm_error": llm_result.get("error"),
            "plain_language_report": report_text,
            "summary": f"Groq Report Agent generated a single-stock explanation for {symbol}."
        }

    # --------------------------------------------------
    # Screener report
    # --------------------------------------------------
    def generate_screener_report(
        self,
        user_question: str,
        screener_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate natural-language explanation for S&P-style screener result.
        """

        compact_context = {
            "summary": screener_result.get("summary"),
            "universe_size": screener_result.get("universe_size"),
            "scanned_count": screener_result.get("scanned_count"),
            "failed_count": screener_result.get("failed_count"),
            "top_buy_candidates": screener_result.get("top_buy_candidates", [])[:10],
            "highest_risk_candidates": screener_result.get(
                "highest_risk_candidates",
                screener_result.get("top_sell_risk", [])
            )[:10]
        }

        system_message = self._base_system_prompt()

        user_message = f"""
User question:
{user_question}

S&P-style Screener Agent output:
{self._safe_json(compact_context)}

Please explain:
1. Which stocks look strongest for further research.
2. Which stocks need caution.
3. Why overbought stocks should not be treated as direct buy recommendations.
4. A short risk-aware conclusion.
"""

        llm_result = self._call_llm(system_message, user_message)

        if llm_result.get("success"):
            report_text = llm_result["output_text"]
            source = "groq_llm"
        else:
            report_text = self._fallback_screener_report(screener_result)
            source = "fallback_rule"

        return {
            "success": True,
            "agent_goal": "Explain the S&P-style screener result in natural language.",
            "report_type": "screener_report",
            "source": source,
            "provider": "groq",
            "model": self.model,
            "llm_available": llm_result.get("llm_available", False),
            "llm_error": llm_result.get("error"),
            "plain_language_report": report_text,
            "summary": "Groq Report Agent generated a screener explanation."
        }

    # --------------------------------------------------
    # Financial report / news simplifier
    # --------------------------------------------------
    def simplify_financial_text(
        self,
        report_text: str,
        user_question: str = "Please simplify this financial report or news text."
    ) -> Dict[str, Any]:
        """
        Simplify pasted financial report/news/analysis text.
        """

        if not report_text or not report_text.strip():
            return {
                "success": False,
                "agent_goal": "Simplify financial/news/report text in plain language.",
                "report_type": "financial_text_simplification",
                "source": "none",
                "provider": "groq",
                "llm_available": False,
                "plain_language_report": "No report text was provided.",
                "summary": "No report text was provided."
            }

        system_message = """
You are a Groq-powered Report Simplification Agent for an educational trading decision support prototype.

Your task:
- Simplify complex financial, news, or company report text.
- Extract key positive points, negative points, risks, and possible market impact.
- Do not provide guaranteed investment advice.
- Do not recommend real trading or leverage.
- Use clear student-friendly language.

Output format:
- Simple summary
- Positive signals
- Negative signals / risks
- Possible impact
- Cautious conclusion
"""

        user_message = f"""
User question:
{user_question}

Report/news text:
{report_text[:8000]}
"""

        llm_result = self._call_llm(system_message, user_message)

        if llm_result.get("success"):
            report = llm_result["output_text"]
            source = "groq_llm"
        else:
            report = (
                "Groq is not available, so the system used a fallback explanation. "
                "Please check GROQ_API_KEY and groq package installation."
            )
            source = "fallback_rule"

        return {
            "success": True,
            "agent_goal": "Simplify financial/news/report text in plain language.",
            "report_type": "financial_text_simplification",
            "source": source,
            "provider": "groq",
            "model": self.model,
            "llm_available": llm_result.get("llm_available", False),
            "llm_error": llm_result.get("error"),
            "plain_language_report": report,
            "summary": "Groq Report Agent simplified the provided financial text."
        }

    # --------------------------------------------------
    # Fallback reports
    # --------------------------------------------------
    def _fallback_single_stock_report(
        self,
        user_question: str,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any],
        risk_result: Dict[str, Any]
    ) -> str:
        symbol = (
            risk_result.get("risk_for_next_agent", {}).get("symbol")
            or risk_result.get("symbol")
            or signal_result.get("signal_for_next_agent", {}).get("symbol")
            or analysis_result.get("symbol")
            or "this stock"
        )

        validation_confidence = validation_result.get("confidence", "Unknown")
        analyst_signal = analysis_result.get("analyst_signal", "Unknown")
        model_signal = signal_result.get("model_signal", "Unknown")
        model_confidence = signal_result.get("confidence_level", "Unknown")
        final_signal = risk_result.get("final_signal", "Unknown")
        risk_level = risk_result.get("risk_level", "Unknown")
        risk_action = risk_result.get("risk_action", "Unknown")

        if final_signal == "BUY_CANDIDATE":
            direct_answer = (
                f"{symbol} is shown as a buy candidate for further research, "
                "but this should not be treated as a guaranteed buy decision."
            )
            strategy = (
                "A cautious paper strategy is to monitor confirmation and avoid over-sizing the position."
            )

        elif final_signal == "SELL_RISK":
            direct_answer = (
                f"{symbol} is currently classified as SELL_RISK. "
                "The system is cautious about this stock."
            )
            strategy = (
                "A cautious paper strategy is to avoid new entry, reduce exposure, "
                "or monitor until risk improves."
            )

        elif final_signal == "HOLD":
            direct_answer = (
                f"{symbol} is currently closer to HOLD. "
                "The system does not detect a strong directional opportunity."
            )
            strategy = "A cautious paper strategy is to hold or wait for clearer confirmation."

        elif final_signal == "BLOCKED":
            direct_answer = (
                f"The system blocked analysis or action for {symbol}, usually due to low data confidence or high risk."
            )
            strategy = "The safest paper strategy is to take no action."

        else:
            direct_answer = f"The system generated an uncertain signal for {symbol}."
            strategy = "The safest paper strategy is to monitor and avoid aggressive action."

        return f"""
### Direct answer
{direct_answer}

### Evidence from agents
- Validation confidence: {validation_confidence}
- Analyst signal: {analyst_signal}
- Model signal: {model_signal}
- Model confidence: {model_confidence}
- Risk Agent final signal: {final_signal}
- Risk level: {risk_level}
- Risk action: {risk_action}

### Risk warning
This is a paper decision support result. The system may be wrong, and market conditions can change quickly.

### Strategy suggestion
{strategy}

### Disclaimer
This is not financial advice and does not execute real trades.
"""

    def _fallback_screener_report(self, screener_result: Dict[str, Any]) -> str:
        top_buy = screener_result.get("top_buy_candidates", [])[:5]
        top_risk = screener_result.get(
            "highest_risk_candidates",
            screener_result.get("top_sell_risk", [])
        )[:5]

        buy_lines = []
        for row in top_buy:
            buy_lines.append(
                f"- {row.get('symbol')}: buy_score={row.get('buy_score')}, "
                f"signal={row.get('screen_signal')}, reason={row.get('reason')}"
            )

        risk_lines = []
        for row in top_risk:
            risk_lines.append(
                f"- {row.get('symbol')}: risk_score={row.get('risk_score')}, "
                f"signal={row.get('screen_signal')}, reason={row.get('reason')}"
            )

        return f"""
### Screener summary
The Screener Agent scanned an S&P-style stock universe and ranked stocks by technical features.

### Top buy candidates for further research
{chr(10).join(buy_lines) if buy_lines else "No buy candidates available."}

### Highest risk / caution candidates
{chr(10).join(risk_lines) if risk_lines else "No caution candidates available."}

### Risk warning
Stocks marked as BUY_WATCHLIST_OVERBOUGHT may have strong momentum but higher entry risk because RSI is high.

### Disclaimer
This is a prototype screener for paper decision support, not financial advice.
"""