import os
import json
from typing import Dict, Any, Optional

try:
    from groq import Groq
except Exception:
    Groq = None


class LLMReportAgent:
    """
    Groq-powered LLM Report Agent.

    It explains structured outputs from the multi-agent pipeline.
    It does not create the trading decision itself.
    """

    SAFETY_WORDING_RULES = """
Important wording rules:
- Do not say "you should buy", "you should sell", "recommended to buy", or "recommended to sell".
- Use safer decision-support wording:
  - "not identified as a strong buy candidate at this stage"
  - "shows a HOLD-style signal"
  - "requires caution"
  - "candidate for further research"
  - "higher-risk watchlist stock"
  - "not a strong entry signal based on the current agent outputs"
- Do not provide personalized financial advice.
- Do not tell the user to clear position, add position, reduce position, or use leverage.
- Always explain that the output is for paper decision support and further research only.
"""

    FINANCIAL_SIMPLIFIER_RULES = """
Financial report/news simplification rules:
- You must only use the pasted report/news text provided by the user.
- Do not search for news.
- Do not invent company events, numbers, announcements, market reactions, or analyst opinions.
- If the pasted text does not contain enough concrete report/news information, say:
  "Insufficient pasted report/news text was provided. Please paste the actual report or news paragraph."
- Do not answer direct buy/sell/position/leverage questions.
- Do not provide personalized financial advice.
- Use cautious wording such as "may", "could", "suggests", and "potential impact".
"""

    def __init__(self, model: Optional[str] = None, temperature: float = 0.2, max_tokens: int = 1000):
        self.api_key = os.getenv("GROQ_API_KEY")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None

        if Groq is not None and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception:
                self.client = None

    def _safe_json(self, data: Any, max_chars: int = 4500) -> str:
        try:
            text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        except Exception:
            text = str(data)
        if len(text) > max_chars:
            return text[:max_chars] + "\n... [truncated]"
        return text

    def _call_groq(self, prompt: str) -> Dict[str, Any]:
        if self.client is None:
            return {
                "success": False,
                "llm_available": False,
                "llm_error": "Groq client is not available. Check GROQ_API_KEY, GROQ_MODEL, and groq installation.",
                "text": ""
            }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a cautious financial decision-support explanation assistant. "
                            "You explain structured agent outputs in clear language. "
                            "You do not provide personalized financial advice."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return {
                "success": True,
                "llm_available": True,
                "llm_error": "",
                "text": response.choices[0].message.content
            }
        except Exception as e:
            return {
                "success": False,
                "llm_available": False,
                "llm_error": str(e),
                "text": ""
            }

    def _fallback_single_stock_report(
        self,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        signal_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        strategy_result: Optional[Dict[str, Any]] = None
    ) -> str:
        strategy_result = strategy_result or {}

        symbol = (
            strategy_result.get("symbol")
            or risk_result.get("symbol")
            or signal_result.get("symbol")
            or analysis_result.get("symbol")
            or "the stock"
        )

        analyst_signal = analysis_result.get("analyst_signal", "Unknown")
        analyst_score = analysis_result.get("analyst_score", "Unknown")
        model_signal = signal_result.get("model_signal") or signal_result.get("signal") or "Unknown"
        final_signal = (
            risk_result.get("final_signal")
            or risk_result.get("risk_for_next_agent", {}).get("final_signal")
            or model_signal
        )
        risk_level = (
            risk_result.get("risk_level")
            or risk_result.get("risk_for_next_agent", {}).get("risk_level")
            or "Unknown"
        )
        strategy_action = strategy_result.get("strategy_action", "FURTHER_RESEARCH_ONLY")
        strategy_level = strategy_result.get("strategy_level", "Conservative")
        position_guidance = strategy_result.get(
            "position_guidance",
            "Use this result as research support only and wait for clearer evidence."
        )
        leverage_guidance = strategy_result.get("leverage_guidance", "Do not use leverage in this prototype.")

        return f"""
**Direct Answer:** {symbol} is not being presented as a direct buy or sell instruction. Based on the current agent outputs, the risk-controlled signal is **{final_signal}** and the strategy action is **{strategy_action}**.

**Evidence from Agents:**
- Validation confidence: {validation_result.get("confidence", "Unknown")}
- Analyst signal: {analyst_signal}, analyst score: {analyst_score}
- Signal model output: {model_signal}
- Risk Agent final signal: {final_signal}
- Risk level: {risk_level}
- Strategist level: {strategy_level}

**Strategy Guidance:** {position_guidance}

**Leverage Guidance:** {leverage_guidance}

**Risk Warning:** This is paper decision support only. The model may be wrong, and market conditions can change quickly.

**Not Financial Advice Disclaimer:** This output is for paper decision support and further research only. It is not personalized financial advice.
""".strip()

    def _fallback_screener_report(self, screener_result: Dict[str, Any]) -> str:
        top_buy = screener_result.get("top_buy_candidates", [])[:5]
        top_risk = (
            screener_result.get("highest_risk_candidates")
            or screener_result.get("top_sell_risk")
            or []
        )[:5]

        buy_lines = []
        for item in top_buy:
            buy_lines.append(
                f"- {item.get('symbol', 'Unknown')}: buy_score={item.get('buy_score', 'N/A')}, "
                f"risk_score={item.get('risk_score', 'N/A')}, signal={item.get('screen_signal', 'N/A')}"
            )

        risk_lines = []
        for item in top_risk:
            risk_lines.append(
                f"- {item.get('symbol', 'Unknown')}: risk_score={item.get('risk_score', 'N/A')}, "
                f"buy_score={item.get('buy_score', 'N/A')}, signal={item.get('screen_signal', 'N/A')}"
            )

        return f"""
**Direct Answer:** The screener produced candidates for further research, not direct buy or sell recommendations.

**Top Candidates for Further Research:**
{chr(10).join(buy_lines) if buy_lines else "- No candidates available."}

**Higher-Risk / Caution Candidates:**
{chr(10).join(risk_lines) if risk_lines else "- No caution candidates available."}

**Risk Warning:** Some candidates may have high RSI, weak momentum, or higher volatility. These results should be checked with further research.

**Not Financial Advice Disclaimer:** This is a paper decision-support output only. It is not financial advice.
""".strip()

    def _fallback_financial_summary(self) -> str:
        return """
**Summary:** Groq was not available, so the system cannot generate a full LLM summary.

**Positive Signals:** Please review the pasted text for revenue growth, margin improvement, guidance, product progress, or management confidence.

**Negative Signals / Risks:** Please review the pasted text for cost pressure, weaker demand, regulatory concerns, debt risk, or uncertainty.

**Possible Market Impact:** The impact depends on how investors interpret the balance between positive signals and risks.

**Cautious Conclusion:** This section is for report/news simplification only. It does not provide direct buy/sell advice.
""".strip()

    def generate_single_stock_report(
        self,
        user_question: str,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        training_result: Dict[str, Any],
        signal_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        strategy_result: Optional[Dict[str, Any]] = None,
        reward_record_result: Optional[Dict[str, Any]] = None,
        auto_reward_update_result: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        prompt = f"""
You are explaining the result of a human-in-the-loop multi-agent stock decision-support system.

{self.SAFETY_WORDING_RULES}

User question:
{user_question}

Validation Agent output:
{self._safe_json(validation_result)}

Analyst Agent output:
{self._safe_json(analysis_result)}

Training Agent output:
{self._safe_json(training_result)}

Signal Model output:
{self._safe_json(signal_result)}

Risk Agent output:
{self._safe_json(risk_result)}

Strategist Agent output:
{self._safe_json(strategy_result or {})}

Reward Agent output:
{self._safe_json(reward_record_result or {})}

Auto delayed reward update output:
{self._safe_json(auto_reward_update_result or {})}

Please produce a short, clear answer with this structure:

**Direct Answer:**
Use safe wording. Do not say "recommended to buy" or "recommended to sell".

**Evidence from Agents:**
Explain validation, analyst signal, model signal, risk signal, and strategist output.

**Strategy Guidance:**
Use the Strategist Agent output. Do not invent a new strategy.

**Risk Warning:**
Explain risk level and uncertainty.

**Not Financial Advice Disclaimer:**
State that this is paper decision support and not financial advice.
"""
        response = self._call_groq(prompt)

        if response["success"]:
            report = response["text"]
            llm_available = True
        else:
            report = self._fallback_single_stock_report(
                validation_result=validation_result,
                analysis_result=analysis_result,
                signal_result=signal_result,
                risk_result=risk_result,
                strategy_result=strategy_result
            )
            llm_available = False

        symbol = (
            (strategy_result or {}).get("symbol")
            or risk_result.get("symbol")
            or signal_result.get("symbol")
            or analysis_result.get("symbol")
            or "the selected stock"
        )

        return {
            "success": True,
            "agent_goal": "Explain single-stock pipeline output in plain language.",
            "provider": "groq",
            "model": self.model,
            "llm_available": llm_available,
            "llm_error": response.get("llm_error", ""),
            "plain_language_report": report,
            "summary": f"Groq Report Agent generated a single-stock explanation for {symbol}."
        }

    def generate_screener_report(
        self,
        user_question: str,
        screener_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        prompt = f"""
You are explaining the result of a watchlist-based S&P-style stock screener.

{self.SAFETY_WORDING_RULES}

User question:
{user_question}

Screener Agent output:
{self._safe_json(screener_result, max_chars=5000)}

Please produce a clear explanation with this structure:

**Direct Answer:**
Say "top candidates for further research", not "stocks to buy".

**Top Candidates for Further Research:**
Summarize the strongest candidates using buy_score, risk_score, signal, and reason.

**Higher-Risk / Caution Candidates:**
Summarize the caution candidates.

**Evidence from Agents:**
Explain the metrics used, such as momentum, RSI, volatility, moving average gap, and risk score.

**Risk Warning:**
Mention overbought RSI, negative momentum, volatility, and watchlist limitation.

**Not Financial Advice Disclaimer:**
State that this is paper decision support and not financial advice.
"""
        response = self._call_groq(prompt)

        if response["success"]:
            report = response["text"]
            llm_available = True
        else:
            report = self._fallback_screener_report(screener_result)
            llm_available = False

        return {
            "success": True,
            "agent_goal": "Explain screener output in plain language.",
            "provider": "groq",
            "model": self.model,
            "llm_available": llm_available,
            "llm_error": response.get("llm_error", ""),
            "plain_language_report": report,
            "summary": "Groq screener explanation generated successfully."
        }

    def simplify_financial_text(self, report_text: str, user_question: str) -> Dict[str, Any]:
        clean_text = report_text.strip()
        if len(clean_text) < 120 or len(clean_text.split()) < 20:
            report = (
                "**Insufficient pasted report/news text was provided.** "
                "Please paste the actual report or news paragraph. "
                "This section does not search news automatically and should not generate content from a ticker-only query."
            )
            return {
                "success": True,
                "agent_goal": "Simplify pasted financial report/news text.",
                "report_type": "financial_text_simplification",
                "source": "input_validation",
                "provider": "groq",
                "model": self.model,
                "llm_available": False,
                "llm_error": "Input text too short for reliable simplification.",
                "plain_language_report": report,
                "summary": "Financial Simplifier rejected short input to avoid hallucination."
            }

        prompt = f"""
You are a financial report/news simplification assistant.

{self.FINANCIAL_SIMPLIFIER_RULES}

User pasted report/news text:
{clean_text}

User question:
{user_question}

Please produce the answer with this structure:

**Summary:**
Summarize only the pasted text.

**Positive Signals:**
List positive signals only if they are supported by the pasted text.

**Negative Signals / Risks:**
List risks only if they are supported by the pasted text.

**Possible Market Impact:**
Describe possible market impact cautiously, using only the pasted text.

**Cautious Conclusion:**
Do not give buy/sell/position/leverage advice. State that this is only report/news simplification.
"""
        response = self._call_groq(prompt)

        if response["success"]:
            report = response["text"]
            llm_available = True
        else:
            report = self._fallback_financial_summary()
            llm_available = False

        return {
            "success": True,
            "agent_goal": "Simplify pasted financial report/news text in plain language.",
            "report_type": "financial_text_simplification",
            "source": "pasted_user_text_only",
            "provider": "groq",
            "model": self.model,
            "llm_available": llm_available,
            "llm_error": response.get("llm_error", ""),
            "plain_language_report": report,
            "summary": "Financial Report/News Simplifier processed pasted text only."
        }
