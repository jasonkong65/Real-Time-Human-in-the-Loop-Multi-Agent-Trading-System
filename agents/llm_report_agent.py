import os
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
except Exception:
    Groq = None


class LLMReportAgent:
    """
    Groq Report Agent

    Uses Groq only as an explanation layer. It does not override the decisions
    made by the Analyst, Training, Risk, or Strategist agents.
    """

    COMPANY_TERMS = {
        "AAPL": ["apple", "aapl", "iphone", "ipad", "mac", "ios", "app store", "vision pro"],
        "MSFT": ["microsoft", "msft", "azure", "windows", "copilot", "openai", "office", "xbox"],
        "NVDA": ["nvidia", "nvda", "gpu", "cuda", "blackwell", "ai chip"],
        "TSLA": ["tesla", "tsla", "ev", "model y", "model 3", "elon musk"],
        "GOOGL": ["alphabet", "google", "googl", "gemini", "youtube", "search"],
        "META": ["meta", "facebook", "instagram", "whatsapp", "metaverse"],
        "AMZN": ["amazon", "amzn", "aws", "prime", "e-commerce"],
    }

    def __init__(self, model: str = "llama-3.1-8b-instant", temperature: float = 0.2, max_tokens: int = 900):
        self.model = os.getenv("GROQ_MODEL", model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        self.finnhub_key = (os.getenv("FINNHUB_API_KEY") or "").strip()
        self.alpha_vantage_key = (os.getenv("ALPHA_VANTAGE_API_KEY") or "").strip()
        self.client = None
        if Groq is not None and self.api_key:
            try:
                self.client = Groq(api_key=self.api_key)
            except Exception:
                self.client = None

    # ------------------------------------------------------------------
    # General helpers
    # ------------------------------------------------------------------
    def _is_available(self) -> bool:
        return self.client is not None

    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _get_nested(self, data: Dict[str, Any], keys: List[str], default=None):
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current

    def _call_groq(self, system_prompt: str, user_prompt: str, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        if not self._is_available():
            return {"success": False, "error": "Groq client is not available."}
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=max_tokens or self.max_tokens,
            )
            text = response.choices[0].message.content.strip()
            return {"success": True, "text": text}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _clean_markdown(self, text: str) -> str:
        if not text:
            return ""
        # Keep markdown readable but avoid large model rambles.
        return text.strip()

    # ------------------------------------------------------------------
    # Single-stock report
    # ------------------------------------------------------------------
    def generate_single_stock_report(
        self,
        user_question: str,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        training_result: Dict[str, Any],
        signal_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        strategy_result: Dict[str, Any],
        reward_record_result: Optional[Dict[str, Any]] = None,
        auto_reward_update_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        symbol = (
            risk_result.get("symbol")
            or signal_result.get("symbol")
            or analysis_result.get("symbol")
            or validation_result.get("symbol")
            or "UNKNOWN"
        )
        facts = {
            "user_question": user_question,
            "symbol": symbol,
            "validation_confidence": validation_result.get("confidence"),
            "analyst_signal": analysis_result.get("analyst_signal"),
            "analyst_display": analysis_result.get("display_signal"),
            "analyst_score": analysis_result.get("analyst_score"),
            "model_signal": signal_result.get("model_signal"),
            "model_display": signal_result.get("display_signal"),
            "model_confidence": signal_result.get("confidence_level"),
            "risk_final_signal": risk_result.get("final_signal"),
            "risk_level": risk_result.get("risk_level"),
            "risk_interpretation": risk_result.get("risk_interpretation"),
            "strategy_action": strategy_result.get("strategy_action"),
            "strategy_level": strategy_result.get("strategy_level"),
            "position_guidance": strategy_result.get("position_guidance"),
            "leverage_guidance": strategy_result.get("leverage_guidance"),
            "conditions_to_reconsider": strategy_result.get("conditions_to_reconsider", []),
        }
        system_prompt = (
            "You explain a multi-agent stock research system in simple English. "
            "Do not give personalized financial advice. Do not invent facts. "
            "Use only the supplied agent facts. Keep the answer short, practical, and natural. "
            "Do not override the Risk Agent or Strategist Agent."
        )
        user_prompt = (
            "Explain this result for a class demo. Use these headings exactly:\n"
            "Direct answer\nEvidence from agents\nStrategy guidance\nRisk warning\nNot financial advice disclaimer\n\n"
            f"Agent facts:\n{facts}"
        )
        groq = self._call_groq(system_prompt, user_prompt)
        if groq.get("success"):
            return {
                "success": True,
                "agent": "Groq Report Agent",
                "agent_goal": "Explain the single-stock multi-agent decision in natural language.",
                "report_type": "single_stock_recommendation_explanation",
                "source": "groq",
                "provider": "groq",
                "model": self.model,
                "llm_available": True,
                "llm_error": None,
                "symbol": symbol,
                "plain_language_report": self._clean_markdown(groq["text"]),
                "summary": f"Groq Report Agent generated a single-stock explanation for {symbol}.",
            }
        fallback = self._fallback_single_stock(facts, groq.get("error"))
        return fallback

    def _fallback_single_stock(self, facts: Dict[str, Any], error: Optional[str] = None) -> Dict[str, Any]:
        symbol = facts.get("symbol", "UNKNOWN")
        report = f"""
**Direct answer**  
{symbol} is not presented as a direct buy or sell instruction. The risk-controlled signal is **{facts.get('risk_final_signal')}**, and the strategy action is **{facts.get('strategy_action')}**.

**Evidence from agents**  
- Validation confidence: {facts.get('validation_confidence')}  
- Analyst signal: {facts.get('analyst_display') or facts.get('analyst_signal')} with score {facts.get('analyst_score')}  
- Model signal: {facts.get('model_signal')} with {facts.get('model_confidence')} confidence  
- Risk level: {facts.get('risk_level')}  
- Strategy level: {facts.get('strategy_level')}

**Strategy guidance**  
{facts.get('position_guidance') or 'Use this as a research note only.'}

**Risk warning**  
{facts.get('risk_interpretation') or 'Market conditions can change quickly.'}

**Not financial advice disclaimer**  
This output is for paper decision support and class demonstration only. It is not personalized financial advice.
""".strip()
        return {
            "success": True,
            "agent": "Groq Report Agent",
            "source": "local_fallback",
            "provider": "local_fallback",
            "model": self.model,
            "llm_available": False,
            "llm_error": error,
            "symbol": symbol,
            "plain_language_report": report,
            "summary": f"Local fallback generated a single-stock explanation for {symbol}.",
        }

    # ------------------------------------------------------------------
    # Screener report
    # ------------------------------------------------------------------
    def generate_screener_report(self, user_question: str, screener_result: Dict[str, Any]) -> Dict[str, Any]:
        top = screener_result.get("top_buy_candidates", [])[:5]
        risk = screener_result.get("highest_risk_candidates", [])[:5]
        facts = {"question": user_question, "top_buy_candidates": top, "highest_risk_candidates": risk}
        prompt = (
            "Explain the screener output briefly. Only use the supplied rows. "
            "Use simple headings: Direct answer, Stronger watchlist names, Caution names, Risk warning.\n\n"
            f"Facts:\n{facts}"
        )
        groq = self._call_groq(
            "You explain a watchlist screener. Do not provide direct trading advice or invent facts.",
            prompt,
            max_tokens=700,
        )
        if groq.get("success"):
            return {
                "success": True,
                "agent": "Groq Report Agent",
                "report_type": "screener_explanation",
                "source": "groq",
                "llm_available": True,
                "plain_language_report": groq["text"],
                "summary": "Groq Report Agent explained the screener result.",
            }
        top_names = ", ".join([r.get("symbol", "") for r in top if r.get("symbol")]) or "none"
        risk_names = ", ".join([r.get("symbol", "") for r in risk if r.get("symbol")]) or "none"
        return {
            "success": True,
            "agent": "Groq Report Agent",
            "report_type": "screener_explanation",
            "source": "local_fallback",
            "llm_available": False,
            "llm_error": groq.get("error"),
            "plain_language_report": (
                f"**Direct answer**  \nThe strongest watchlist candidates are {top_names}. The caution names are {risk_names}.\n\n"
                "**Risk warning**  \nThis is a watchlist scan, not a full market scan or financial advice."
            ),
            "summary": "Local fallback explained the screener result.",
        }

    # ------------------------------------------------------------------
    # Verified financial/news summarizer
    # ------------------------------------------------------------------
    def _detect_symbol(self, text: str, fallback: Optional[str] = None) -> str:
        text = text or ""
        tickers = re.findall(r"\b[A-Z]{1,5}\b", text.upper())
        common_words = {"NEWS", "REPORT", "FINANCIAL", "THE", "AND", "FOR", "THIS", "WEEK", "LAST", "YEAR"}
        for ticker in tickers:
            if ticker not in common_words:
                return ticker
        text_l = text.lower()
        for ticker, terms in self.COMPANY_TERMS.items():
            if any(term in text_l for term in terms):
                return ticker
        return str(fallback or "").upper().strip()

    def _company_terms(self, symbol: str) -> List[str]:
        symbol = str(symbol or "").upper().strip()
        return self.COMPANY_TERMS.get(symbol, [symbol.lower()]) + [symbol.lower()]

    def _fetch_finnhub_news(self, symbol: str, lookback_days: int = 7, max_news: int = 5) -> Dict[str, Any]:
        if not self.finnhub_key or not symbol:
            return {"success": False, "items": [], "error": "Finnhub key or symbol missing."}
        to_date = datetime.utcnow().date()
        from_date = to_date - timedelta(days=int(lookback_days or 7))
        try:
            response = requests.get(
                "https://finnhub.io/api/v1/company-news",
                params={"symbol": symbol, "from": str(from_date), "to": str(to_date), "token": self.finnhub_key},
                timeout=12,
            )
            if response.status_code != 200:
                return {"success": False, "items": [], "error": f"Finnhub HTTP {response.status_code}"}
            raw = response.json()
            if not isinstance(raw, list):
                return {"success": False, "items": [], "error": "Finnhub did not return a news list."}
            filtered = self._filter_relevant_news(symbol, raw, max_news=max_news)
            return {"success": True, "items": filtered, "raw_count": len(raw)}
        except Exception as exc:
            return {"success": False, "items": [], "error": str(exc)}

    def _score_news_item(self, symbol: str, item: Dict[str, Any]) -> int:
        terms = self._company_terms(symbol)
        text = f"{item.get('headline','')} {item.get('summary','')} {item.get('source','')}".lower()
        score = 0
        reason = []
        if symbol.lower() in text:
            score += 5
            reason.append("ticker mentioned")
        for term in terms:
            if term and term in text:
                score += 3 if term != symbol.lower() else 2
                reason.append(f"company term: {term}")
                break
        product_terms = [t for t in terms if len(t) > 3]
        for term in product_terms:
            if term in text:
                score += 1
        broad_markets = ["s&p", "nasdaq", "market", "etf", "dow", "treasury", "fed", "bubble"]
        if any(term in text for term in broad_markets) and not any(term in text for term in terms):
            score -= 3
            reason.append("broad market only")
        item["relevance_score"] = score
        item["relevance_reason"] = "; ".join(reason) if reason else "weak relevance"
        return score

    def _filter_relevant_news(self, symbol: str, items: List[Dict[str, Any]], max_news: int = 5) -> List[Dict[str, Any]]:
        scored = []
        for item in items:
            if not isinstance(item, dict):
                continue
            score = self._score_news_item(symbol, item)
            if score >= 4:
                scored.append(item)
        scored = sorted(scored, key=lambda x: x.get("relevance_score", 0), reverse=True)
        cleaned = []
        seen = set()
        for item in scored:
            headline = str(item.get("headline", "")).strip()
            if not headline or headline.lower() in seen:
                continue
            seen.add(headline.lower())
            date_value = item.get("datetime")
            try:
                date_text = datetime.utcfromtimestamp(int(date_value)).strftime("%Y-%m-%d") if date_value else ""
            except Exception:
                date_text = ""
            cleaned.append({
                "date": date_text,
                "source": item.get("source", ""),
                "headline": headline,
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "relevance_score": item.get("relevance_score"),
                "relevance_reason": item.get("relevance_reason"),
            })
            if len(cleaned) >= int(max_news or 5):
                break
        return cleaned

    def _fetch_alpha_snapshot(self, symbol: str) -> Dict[str, Any]:
        if not self.alpha_vantage_key or not symbol:
            return {"success": False, "error": "Alpha Vantage key or symbol missing."}
        try:
            response = requests.get(
                "https://www.alphavantage.co/query",
                params={"function": "OVERVIEW", "symbol": symbol, "apikey": self.alpha_vantage_key},
                timeout=12,
            )
            if response.status_code != 200:
                return {"success": False, "error": f"Alpha Vantage HTTP {response.status_code}"}
            data = response.json()
            if not isinstance(data, dict) or not data.get("Symbol"):
                return {"success": False, "error": "No company overview returned."}
            keep = ["Symbol", "Name", "Sector", "Industry", "MarketCapitalization", "PERatio", "ProfitMargin", "QuarterlyRevenueGrowthYOY", "QuarterlyEarningsGrowthYOY", "AnalystTargetPrice"]
            return {"success": True, "snapshot": {k: data.get(k) for k in keep}}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def _infer_mode(self, text: str, source_mode: str) -> str:
        selected = str(source_mode or "auto").lower().strip()
        if selected in ["news", "financial", "news_and_financial", "pasted_text"]:
            # Still correct obvious query mismatch: "MSFT news" should be news.
            if "news" in text.lower() and selected == "financial":
                return "news"
            return selected
        low = text.lower()
        if "news" in low:
            return "news"
        if any(term in low for term in ["financial", "report", "earnings", "revenue", "margin", "guidance"]):
            return "financial"
        return "pasted_text"

    def _build_source_context(self, report_text: str, source_mode: str, symbol: str, lookback_days: int, max_news: int) -> Dict[str, Any]:
        mode = self._infer_mode(report_text, source_mode)
        context = {
            "symbol": symbol,
            "mode": mode,
            "pasted_text": report_text,
            "news": [],
            "financial_snapshot": None,
            "source_status": [],
        }
        should_fetch_news = mode in ["news", "news_and_financial"] or (mode == "financial" and "news" in report_text.lower())
        should_fetch_financial = mode in ["financial", "news_and_financial"]
        if symbol and should_fetch_news:
            news = self._fetch_finnhub_news(symbol, lookback_days, max_news)
            if news.get("success"):
                context["news"] = news.get("items", [])
                context["source_status"].append(f"Fetched {news.get('raw_count', 0)} Finnhub news items and kept {len(context['news'])} relevant company items for {symbol}.")
            else:
                context["source_status"].append(f"Finnhub news was not available: {news.get('error')}")
        if symbol and should_fetch_financial:
            snap = self._fetch_alpha_snapshot(symbol)
            if snap.get("success"):
                context["financial_snapshot"] = snap.get("snapshot")
                context["source_status"].append(f"Fetched a lightweight Alpha Vantage company snapshot for {symbol}.")
            else:
                context["source_status"].append(f"Alpha Vantage company snapshot was not available: {snap.get('error')}")
        if not context["source_status"]:
            context["source_status"].append("Used pasted text only; no external source was added.")
        return context

    def simplify_financial_text(
        self,
        report_text: str,
        question: Optional[str] = None,
        user_question: Optional[str] = None,
        source_mode: str = "auto",
        symbol: Optional[str] = None,
        lookback_days: int = 7,
        max_news: int = 5,
    ) -> Dict[str, Any]:
        report_text = report_text or ""
        question = question or user_question or "Simplify this financial/news text."
        detected_symbol = self._detect_symbol(report_text, fallback=symbol)
        context = self._build_source_context(report_text, source_mode, detected_symbol, lookback_days, max_news)
        system_prompt = (
            "You are a careful financial news/report summarizer for a class project. "
            "Use only the supplied pasted text, retrieved company-specific news, and company snapshot. "
            "Do not mix in unrelated market news. Do not provide trading advice. If evidence is limited, say so. "
            "Write simply and naturally."
        )
        user_prompt = (
            "Use this structure exactly:\n"
            "Summary\nVerified source status\nCompany-specific points\nPositive signals\nRisks\nPossible market impact\nCautious conclusion\n\n"
            f"Question: {question}\n"
            f"Context: {context}"
        )
        groq = self._call_groq(system_prompt, user_prompt, max_tokens=900)
        if groq.get("success"):
            report = groq["text"]
            source = "groq"
            llm_available = True
            llm_error = None
        else:
            report = self._fallback_financial_report(context)
            source = "local_fallback"
            llm_available = False
            llm_error = groq.get("error")
        return {
            "success": True,
            "agent": "Groq Report Agent",
            "agent_goal": "Summarize only verified pasted or API-sourced company information.",
            "report_type": "verified_financial_news_summary",
            "source": source,
            "provider": source,
            "model": self.model,
            "llm_available": llm_available,
            "llm_error": llm_error,
            "symbol": detected_symbol,
            "detected_symbol": detected_symbol,
            "mode": context.get("mode"),
            "source_status": context.get("source_status", []),
            "retrieved_news_items": context.get("news", []),
            "financial_snapshot": context.get("financial_snapshot"),
            "plain_language_report": report,
            "summary": "Groq financial/news summary generated successfully." if llm_available else "Local fallback financial/news summary generated safely.",
        }

    def _fallback_financial_report(self, context: Dict[str, Any]) -> str:
        symbol = context.get("symbol") or "the company"
        status = context.get("source_status", [])
        news = context.get("news", [])
        snapshot = context.get("financial_snapshot")
        lines = [
            "**Summary**  ",
            f"The system used source-grounded information for {symbol}. It did not add outside facts.",
            "",
            "**Verified source status**  ",
        ]
        for item in status:
            lines.append(f"- {item}")
        if news:
            lines += ["", "**Company-specific points**  "]
            for i, item in enumerate(news[:5], 1):
                lines.append(f"{i}. {item.get('date','')} | {item.get('source','')} | {item.get('headline','')}")
        elif context.get("pasted_text"):
            lines += ["", "**Company-specific points**  ", context.get("pasted_text")[:900]]
        else:
            lines += ["", "**Company-specific points**  ", "No company-specific text or verified news item was available."]
        if snapshot:
            lines += ["", "**Positive signals / risks from snapshot**  "]
            lines.append(str(snapshot))
        lines += [
            "",
            "**Positive signals**  ",
            "Only treat positive points as source-limited cues, not as a buy signal.",
            "",
            "**Risks**  ",
            "Headlines and short summaries may be incomplete. Check full filings or full articles before using them.",
            "",
            "**Possible market impact**  ",
            "The direction and size of market impact cannot be concluded from headlines alone.",
            "",
            "**Cautious conclusion**  ",
            "Use this as a verified research summary only. It is not trading advice.",
        ]
        return "\n".join(lines)
