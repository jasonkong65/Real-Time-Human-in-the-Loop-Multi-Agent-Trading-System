import os
import re
import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()

try:
    from groq import Groq
except Exception:
    Groq = None


class LLMReportAgent:
    """
    Groq Report Agent for the multi-agent stock research prototype.

    Roles:
    - Explain structured single-stock agent outputs in simple, safe language.
    - Explain watchlist screener outputs.
    - Summarise pasted financial/news text or fetch source-grounded company news / financial snapshots.

    Safety design:
    - Groq is an explanation layer only; it does not override Risk Agent or Strategist Agent.
    - Prompts and post-processing avoid direct buy/sell wording.
    - Financial/news summaries use only pasted text or verified API context.
    """

    COMPANY_TERMS = {
        "AAPL": ["apple", "aapl", "iphone", "ipad", "mac", "ios", "app store", "vision pro"],
        "MSFT": ["microsoft", "msft", "azure", "windows", "copilot", "openai", "office", "xbox", "linkedin", "github"],
        "NVDA": ["nvidia", "nvda", "gpu", "cuda", "blackwell", "ai chip", "data center"],
        "TSLA": ["tesla", "tsla", "ev", "model y", "model 3", "elon musk", "cybertruck"],
        "GOOGL": ["alphabet", "google", "googl", "googl", "gemini", "youtube", "search", "waymo"],
        "META": ["meta", "facebook", "instagram", "whatsapp", "metaverse", "threads"],
        "AMZN": ["amazon", "amzn", "aws", "prime", "e-commerce"],
        "AMD": ["amd", "advanced micro devices", "ryzen", "epyc"],
        "AVGO": ["broadcom", "avgo", "vmware"],
        "NFLX": ["netflix", "nflx", "streaming"],
        "JPM": ["jpmorgan", "jp morgan", "jpm"],
        "V": ["visa", "visa inc"],
        "MA": ["mastercard"],
        "WMT": ["walmart", "wmt"],
        "DIS": ["disney", "dis"],
        "INTC": ["intel", "intc"],
        "QCOM": ["qualcomm", "qcom"],
        "CSCO": ["cisco", "csco"],
        "ORCL": ["oracle", "orcl"],
    }

    COMPANY_TO_TICKER = {
        "apple": "AAPL",
        "microsoft": "MSFT",
        "nvidia": "NVDA",
        "tesla": "TSLA",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "meta": "META",
        "facebook": "META",
        "amazon": "AMZN",
        "amd": "AMD",
        "broadcom": "AVGO",
        "netflix": "NFLX",
        "jpmorgan": "JPM",
        "jp morgan": "JPM",
        "visa": "V",
        "mastercard": "MA",
        "walmart": "WMT",
        "disney": "DIS",
        "intel": "INTC",
        "qualcomm": "QCOM",
        "cisco": "CSCO",
        "oracle": "ORCL",
    }

    TICKER_STOP_WORDS = {
        "THE", "AND", "FOR", "NEWS", "THIS", "WEEK", "LAST", "YEAR", "YEARS",
        "REPORT", "BUY", "SELL", "HOLD", "AI", "API", "USA", "CEO", "EPS", "Q",
        "A", "AN", "OF", "TO", "IN", "ON", "ABOUT", "LATEST", "RECENT", "FINANCIAL",
        "COMPANY", "MARKET", "ETF", "DOW", "CEO", "CFO", "SEC"
    }

    POSITIVE_TERMS = [
        "beat", "beats", "growth", "strong", "record", "raise", "raised", "upgrade", "upgraded",
        "surge", "surged", "higher revenue", "profit rises", "partnership", "launch", "approval",
        "outperform", "demand", "contract", "win", "wins", "expansion"
    ]

    RISK_TERMS = [
        "miss", "misses", "decline", "declined", "falls", "fell", "drop", "weak", "weaker",
        "warning", "warned", "lawsuit", "probe", "investigation", "antitrust", "regulation",
        "regulatory", "tariff", "layoffs", "cut", "cuts", "downgrade", "downgraded", "loss",
        "slump", "pressure", "cost", "spending", "margin pressure", "risk", "breach"
    ]

    def __init__(self, model: str = "llama-3.1-8b-instant", temperature: float = 0.2, max_tokens: int = 900):
        self.model = os.getenv("GROQ_MODEL", model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        self.finnhub_key = (os.getenv("FINNHUB_API_KEY") or "").strip()
        self.alpha_vantage_key = (
            os.getenv("ALPHA_VANTAGE_API_KEY")
            or os.getenv("ALPHAVANTAGE_API_KEY")
            or ""
        ).strip()
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

    def _safe_json(self, data: Any, max_chars: int = 9000) -> str:
        try:
            text = json.dumps(data, indent=2, ensure_ascii=False, default=str)
        except Exception:
            text = str(data)
        if len(text) > max_chars:
            return text[:max_chars] + "\n... [truncated]"
        return text

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
            return {"success": False, "llm_available": False, "error": "Groq client is not available. Check GROQ_API_KEY and groq package."}
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
            if not text:
                return {"success": False, "llm_available": False, "error": "Groq returned an empty response."}
            return {"success": True, "llm_available": True, "text": self._sanitize_investment_wording(text)}
        except Exception as exc:
            return {"success": False, "llm_available": False, "error": str(exc)}

    def _sanitize_investment_wording(self, text: str) -> str:
        """Post-process Groq output so it stays safely framed as paper decision support."""
        if not text:
            return text
        replacements = [
            (r"\bbefore buying\b", "before considering a paper-trading entry"),
            (r"\bwhen buying\b", "when reviewing a paper-trading entry"),
            (r"\bshould buy\b", "could review as a research candidate"),
            (r"\brecommend buying\b", "mark as a research candidate"),
            (r"\bbuy this stock\b", "review this stock as a paper candidate"),
            (r"\bstrong buy\b", "strong research candidate"),
            (r"\bshould sell\b", "should review risk exposure in paper tracking"),
            (r"\brecommend selling\b", "suggests a paper risk-review stance"),
            (r"\bsell this stock\b", "review this stock's risk in paper tracking"),
            (r"\bclear your position\b", "review the paper position exposure"),
            (r"\badd leverage\b", "avoid leverage in this prototype"),
            (r"\buse leverage\b", "use no leverage in this prototype"),
            (r"\bNO_ACTION_DATA_OR_RISK_BLOCK\b", "No Action / Risk Block"),
            (r"\bWAIT_FOR_PULLBACK_OR_CONFIRMATION\b", "Wait for Pullback / Confirmation"),
            (r"\bWAIT_FOR_CONFIRMATION\b", "Wait for Confirmation"),
            (r"\bMONITOR_POSITIVE_SETUP\b", "Monitor Positive Setup"),
            (r"\bWATCHLIST_BULLISH_ENTRY_RISK\b", "Bullish Watchlist / Entry Risk"),
            (r"\bBUY_WATCHLIST_ENTRY_RISK\b", "Bullish Watchlist / Entry Risk"),
            (r"\bBUY_WATCHLIST_OVERBOUGHT\b", "Bullish Watchlist / High Entry Risk"),
            (r"\bBUY_CANDIDATE\b", "Research Candidate"),
            (r"\bSELL_RISK\b", "Risk Review"),
            (r"\bBLOCKED\b", "Blocked"),
        ]
        clean = text
        for pattern, replacement in replacements:
            clean = re.sub(pattern, replacement, clean, flags=re.IGNORECASE)
        return clean.strip()


    def _clean_label(self, value: Any, fallback: str = "Unknown") -> str:
        """Convert internal enum-like labels into user-facing text."""
        if value is None:
            return fallback
        text = str(value).strip()
        if not text:
            return fallback
        replacements = {
            "POSITIVE_BUT_ENTRY_RISK": "Positive + Entry Risk",
            "WATCHLIST_BULLISH_ENTRY_RISK": "Bullish Watchlist / Entry Risk",
            "BUY_WATCHLIST_ENTRY_RISK": "Bullish Watchlist / Entry Risk",
            "BUY_WATCHLIST_OVERBOUGHT": "Bullish Watchlist / High Entry Risk",
            "WAIT_FOR_PULLBACK_OR_CONFIRMATION": "Wait for Pullback / Confirmation",
            "WAIT_FOR_CONFIRMATION": "Wait for Confirmation",
            "MONITOR_POSITIVE_SETUP": "Monitor Positive Setup",
            "MONITOR_AND_RESEARCH": "Monitor + Research",
            "RISK_REDUCTION_REVIEW": "Risk Reduction Review",
            "RESEARCH_FOR_POSSIBLE_ENTRY": "Research for Paper Entry",
            "NO_ACTION_DATA_OR_RISK_BLOCK": "No Action / Risk Block",
            "BUY_CANDIDATE": "Research Candidate",
            "SELL_RISK": "Risk Review",
            "HOLD": "Hold / Monitor",
            "BLOCKED": "Blocked",
            "LOW": "Low",
            "MEDIUM": "Medium",
            "HIGH": "High",
            "CRITICAL": "Critical",
            "DEFENSIVE": "Defensive",
            "CAUTIOUS": "Cautious",
            "CONSERVATIVE": "Conservative",
        }
        return replacements.get(text, text.replace("_", " ").title())

    def _clean_list(self, items: Any, limit: int = 5) -> List[str]:
        if not isinstance(items, list):
            return []
        clean = []
        for item in items[:limit]:
            if item is None:
                continue
            clean.append(str(item).strip())
        return [x for x in clean if x]

    def _single_stock_human_facts(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare source facts for either Groq or local fallback without raw internal codes."""
        return {
            "symbol": facts.get("symbol", "UNKNOWN"),
            "validation_confidence": self._clean_label(facts.get("validation_confidence")),
            "analyst_view": self._clean_label(facts.get("analyst_display") or facts.get("analyst_signal")),
            "analyst_score": facts.get("analyst_score"),
            "model_view": self._clean_label(facts.get("model_display") or facts.get("model_signal")),
            "model_confidence": self._clean_label(facts.get("model_confidence")),
            "risk_signal": self._clean_label(facts.get("risk_final_signal")),
            "risk_level": self._clean_label(facts.get("risk_level")),
            "risk_interpretation": facts.get("risk_interpretation") or "The system is staying cautious because market and model signals can change quickly.",
            "strategy_action": self._clean_label(facts.get("strategy_action")),
            "strategy_level": self._clean_label(facts.get("strategy_level")),
            "position_guidance": facts.get("position_guidance") or "Use this as a research note only.",
            "leverage_guidance": facts.get("leverage_guidance") or "Do not use leverage in this prototype.",
            "checklist": self._clean_list(facts.get("checklist")),
            "conditions_to_reconsider": self._clean_list(facts.get("conditions_to_reconsider")),
        }

    def _report_quality_score(self, text: str, source_grounded: bool = True) -> Dict[str, Any]:
        """Lightweight quality check for demo/debugging."""
        body = text or ""
        lower = body.lower()
        score = 1.0
        issues = []
        risky_terms = ["should buy", "recommend buying", "should sell", "recommend selling", "use leverage", "before buying"]
        found = [t for t in risky_terms if t in lower]
        if found:
            score -= 0.35
            issues.append("contains direct trading wording: " + ", ".join(found[:3]))
        if len(body.split()) < 35:
            score -= 0.15
            issues.append("very short report")
        if source_grounded and "not financial advice" not in lower and "paper" not in lower:
            score -= 0.10
            issues.append("paper-decision framing is weak")
        return {"score": round(max(0.0, min(1.0, score)), 3), "issues": issues or ["No obvious wording issue detected."]}

    def _short_system_prompt(self, task: str) -> str:
        return (
            f"You are the report writer for a class stock-analysis prototype. {task} "
            "Use only the supplied facts. Keep it short and natural. "
            "Do not give real trading advice. Do not say the user should buy, sell, clear a position, or use leverage. "
            "Use paper-decision wording such as watchlist, research candidate, wait for confirmation, or risk review."
        )

    # ------------------------------------------------------------------
    # Ticker and mode helpers
    # ------------------------------------------------------------------
    def _extract_symbol_from_text(self, text: str) -> Optional[str]:
        text = text or ""
        upper_text = text.upper()
        for token in re.findall(r"\b[A-Z]{1,5}\b", upper_text):
            if token not in self.TICKER_STOP_WORDS:
                return token
        lower_text = text.lower()
        for company, ticker in self.COMPANY_TO_TICKER.items():
            if company in lower_text:
                return ticker
        return None

    def _resolve_symbol(self, report_text: str, symbol: Optional[str] = None, ticker_override: Optional[str] = None) -> Optional[str]:
        # Important: ticker in the actual input wins over UI fallback/override.
        from_text = self._extract_symbol_from_text(report_text or "")
        if from_text:
            return from_text.upper()
        if symbol:
            return str(symbol).strip().upper()
        if ticker_override:
            return str(ticker_override).strip().upper()
        return None

    def _infer_source_mode(self, report_text: str, requested_mode: str = "auto") -> str:
        text = (report_text or "").lower()
        requested = (requested_mode or "auto").lower()
        has_news = any(w in text for w in ["news", "headline", "headlines", "latest", "recent", "this week", "what happened"])
        has_financial = any(w in text for w in ["financial", "report", "earnings", "income statement", "annual", "quarter", "last year", "results"])
        has_pasted = len(re.findall(r"\w+", text)) >= 18

        # User text should override stale UI mode when it clearly asks for news/report.
        if has_news and has_financial:
            return "news_and_financial"
        if has_news:
            return "news"
        if has_financial:
            return "financial"
        if requested in {"news", "financial", "news_and_financial", "both", "pasted_text"}:
            return "news_and_financial" if requested == "both" else requested
        if has_pasted:
            return "pasted_text"
        return "news"

    # ------------------------------------------------------------------
    # News / financial fetch and filtering
    # ------------------------------------------------------------------
    def _fetch_finnhub_news(self, symbol: str, lookback_days: int = 7, scan_limit: int = 40) -> Dict[str, Any]:
        if not self.finnhub_key:
            return {"success": False, "error": "FINNHUB_API_KEY is missing.", "items": [], "raw_count": 0}
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=max(1, int(lookback_days or 7)))
        params = {"symbol": symbol, "from": start.isoformat(), "to": today.isoformat(), "token": self.finnhub_key}
        try:
            response = requests.get("https://finnhub.io/api/v1/company-news", params=params, timeout=12)
            response.raise_for_status()
            raw = response.json()
            if not isinstance(raw, list):
                return {"success": False, "error": "Finnhub returned non-list data.", "items": [], "raw_count": 0}
            items = []
            for row in raw[:scan_limit]:
                dt = row.get("datetime")
                date_text = "Unknown"
                if dt:
                    try:
                        date_text = datetime.fromtimestamp(int(dt), timezone.utc).strftime("%Y-%m-%d")
                    except Exception:
                        date_text = str(dt)
                items.append({
                    "date": date_text,
                    "headline": (row.get("headline") or "").strip(),
                    "summary": (row.get("summary") or "").strip(),
                    "source": row.get("source") or "Unknown",
                    "url": row.get("url") or "",
                })
            return {"success": True, "error": None, "items": items, "raw_count": len(raw), "lookback_days": lookback_days}
        except Exception as exc:
            return {"success": False, "error": str(exc), "items": [], "raw_count": 0}

    def _news_relevance_score(self, symbol: str, item: Dict[str, Any]) -> Tuple[float, str, str]:
        text = f"{item.get('headline', '')} {item.get('summary', '')}".lower()
        terms = self.COMPANY_TERMS.get(symbol.upper(), [symbol.lower()])
        strong_hits = [t for t in terms if t in text]
        score = 0.0
        reasons = []
        if symbol.lower() in text:
            score += 0.60
            reasons.append(f"mentions ticker {symbol}")
        if strong_hits:
            score += min(0.50, 0.15 * len(strong_hits))
            reasons.append("mentions company terms: " + ", ".join(strong_hits[:4]))
        # Penalise obviously broad/ETF/newsletter topics unless they also have strong company mentions.
        broad_terms = ["qqq", "etf", "spacex", "trump", "s&p", "nasdaq", "dow jones", "mega-cap", "market bubble"]
        broad_hits = [t for t in broad_terms if t in text]
        if broad_hits and score < 0.75:
            score -= 0.25
            reasons.append("looks broad/market-related: " + ", ".join(broad_hits[:3]))
        label = "company_specific" if score >= 0.45 else "excluded_broad_or_uncertain"
        return max(0.0, min(1.0, score)), label, "; ".join(reasons) or "no strong company-specific terms found"

    def _filter_company_news(self, symbol: str, items: List[Dict[str, Any]], max_news: int = 5) -> Dict[str, Any]:
        kept = []
        excluded = []
        for item in items:
            score, label, reason = self._news_relevance_score(symbol, item)
            row = dict(item)
            row["relevance_score"] = round(score, 3)
            row["relevance_label"] = label
            row["relevance_reason"] = reason
            if label == "company_specific":
                kept.append(row)
            else:
                excluded.append(row)
        kept = sorted(kept, key=lambda x: x.get("relevance_score", 0), reverse=True)
        return {
            "company_specific_items": kept[:max_news],
            "excluded_items": excluded[:10],
            "kept_count": len(kept[:max_news]),
            "excluded_count": len(excluded),
        }

    def _fetch_alpha_vantage_snapshot(self, symbol: str) -> Dict[str, Any]:
        if not self.alpha_vantage_key:
            return {"success": False, "error": "ALPHA_VANTAGE_API_KEY is missing.", "snapshot": {}}
        snapshot = {}
        errors = []
        try:
            r = requests.get(
                "https://www.alphavantage.co/query",
                params={"function": "INCOME_STATEMENT", "symbol": symbol, "apikey": self.alpha_vantage_key},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            q = (data.get("quarterlyReports") or [])[:1]
            a = (data.get("annualReports") or [])[:1]
            if q:
                snapshot["latest_quarter"] = {k: q[0].get(k) for k in ["fiscalDateEnding", "reportedCurrency", "totalRevenue", "operatingIncome", "netIncome"]}
            if a:
                snapshot["latest_annual"] = {k: a[0].get(k) for k in ["fiscalDateEnding", "reportedCurrency", "totalRevenue", "operatingIncome", "netIncome"]}
        except Exception as exc:
            errors.append(f"income statement error: {exc}")
        try:
            r = requests.get(
                "https://www.alphavantage.co/query",
                params={"function": "EARNINGS", "symbol": symbol, "apikey": self.alpha_vantage_key},
                timeout=15,
            )
            r.raise_for_status()
            data = r.json()
            e = (data.get("quarterlyEarnings") or [])[:1]
            if e:
                snapshot["latest_earnings"] = {k: e[0].get(k) for k in ["fiscalDateEnding", "reportedDate", "reportedEPS", "estimatedEPS", "surprise", "surprisePercentage"]}
        except Exception as exc:
            errors.append(f"earnings error: {exc}")
        return {"success": bool(snapshot), "error": "; ".join(errors) if errors else None, "snapshot": snapshot}

    def _build_source_context(
        self,
        report_text: str,
        source_mode: str,
        symbol: Optional[str],
        ticker_override: Optional[str],
        lookback_days: int,
        max_news: int,
    ) -> Dict[str, Any]:
        resolved_symbol = self._resolve_symbol(report_text, symbol=symbol, ticker_override=ticker_override)
        effective_mode = self._infer_source_mode(report_text, source_mode)
        context = {
            "symbol": resolved_symbol,
            "requested_source_mode": source_mode,
            "effective_source_mode": effective_mode,
            "user_text": report_text,
            "source_status": [],
            "company_specific_news": [],
            "excluded_news": [],
            "verified_news": {},
            "financial_snapshot": {},
        }
        if not resolved_symbol and effective_mode != "pasted_text":
            context["source_status"].append("No ticker or company name was detected, so live source fetching was skipped.")
            return context

        if effective_mode in {"news", "news_and_financial"} and resolved_symbol:
            news = self._fetch_finnhub_news(resolved_symbol, lookback_days=lookback_days, scan_limit=max(40, max_news * 8))
            context["verified_news"] = news
            if news.get("success"):
                filtered = self._filter_company_news(resolved_symbol, news.get("items", []), max_news=max_news)
                context.update(filtered)
                context["source_status"].append(
                    f"Fetched {len(news.get('items', []))} Finnhub items for {resolved_symbol}; kept {filtered['kept_count']} company-specific items; excluded {filtered['excluded_count']} broad or uncertain items."
                )
            else:
                context["source_status"].append(f"Finnhub news was not available: {news.get('error')}")

        if effective_mode in {"financial", "news_and_financial"} and resolved_symbol:
            snapshot = self._fetch_alpha_vantage_snapshot(resolved_symbol)
            context["financial_snapshot"] = snapshot
            if snapshot.get("success"):
                context["source_status"].append(f"Fetched Alpha Vantage financial snapshot for {resolved_symbol}.")
            else:
                context["source_status"].append(f"Alpha Vantage snapshot was not available: {snapshot.get('error')}")

        if effective_mode == "pasted_text":
            context["source_status"].append("Used pasted text only. No live source was fetched.")
        return context

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
        strategy_result: Optional[Dict[str, Any]] = None,
        reward_record_result: Optional[Dict[str, Any]] = None,
        auto_reward_update_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        strategy_result = strategy_result or {}
        symbol = (
            risk_result.get("symbol")
            or self._get_nested(risk_result, ["risk_for_next_agent", "symbol"])
            or signal_result.get("symbol")
            or self._get_nested(signal_result, ["signal_for_next_agent", "symbol"])
            or analysis_result.get("symbol")
            or self._get_nested(validation_result, ["validation_for_next_agent", "symbol"])
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
            "checklist": strategy_result.get("checklist", []),
            "conditions_to_reconsider": strategy_result.get("conditions_to_reconsider", []),
        }
        human_facts = self._single_stock_human_facts(facts)
        system_prompt = self._short_system_prompt("Explain one stock pipeline result for a non-technical user.")
        user_prompt = (
            "Use these headings exactly: Direct answer, Key evidence, Strategy guidance, Risk note, Disclaimer.\n"
            "Write in simple user-facing language. Do not expose raw enum labels or internal codes.\n"
            "Do not say the ticker is 'not a direct trading instruction'. Instead, explain what the system recommends for paper research.\n"
            "Do not give direct buy/sell instructions. Keep the wording cautious and source-grounded.\n\n"
            f"Facts:\n{self._safe_json(human_facts)}"
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
                "plain_language_report": groq["text"],
                "report_quality_score": self._report_quality_score(groq["text"], source_grounded=True),
                "summary": f"Groq Report Agent generated a single-stock explanation for {symbol}.",
            }
        return self._fallback_single_stock(facts, groq.get("error"))

    def _fallback_single_stock(self, facts: Dict[str, Any], error: Optional[str] = None) -> Dict[str, Any]:
        human = self._single_stock_human_facts(facts)
        symbol = human.get("symbol", "UNKNOWN")
        risk_signal = human.get("risk_signal", "Unknown")
        strategy_action = human.get("strategy_action", "Unknown")
        risk_level = human.get("risk_level", "Unknown")
        strategy_level = human.get("strategy_level", "Unknown")

        if risk_signal == "Blocked" or "Block" in strategy_action or risk_level in {"High", "Critical"}:
            direct_answer = (
                f"For **{symbol}**, the system does **not** have enough safe evidence for a paper decision at this stage. "
                f"The risk-controlled signal is **{risk_signal}**, so the safest research action is **{strategy_action}**."
            )
        elif "Research Candidate" in risk_signal:
            direct_answer = (
                f"For **{symbol}**, the system marks this as a **paper research candidate**, "
                "but the user should still review the evidence and risk notes before taking any simulated action."
            )
        else:
            direct_answer = (
                f"For **{symbol}**, the system suggests a **watchlist / monitor** stance rather than an immediate paper decision. "
                f"The current strategy is **{strategy_action}**."
            )

        checklist = human.get("checklist") or []
        checklist_text = "\n".join([f"- {item}" for item in checklist[:4]]) or "- Re-run the pipeline after the next market data refresh."

        report = f"""
**Direct answer**  
{direct_answer}

**Key evidence**  
- Validation confidence: {human.get('validation_confidence')}  
- Analyst view: {human.get('analyst_view')}  
- Model view: {human.get('model_view')} with {human.get('model_confidence')} confidence  
- Risk level: {risk_level}  
- Strategy level: {strategy_level}

**Strategy guidance**  
{human.get('position_guidance')}

{human.get('leverage_guidance')}

**Risk note**  
{human.get('risk_interpretation')}

**Next checks**  
{checklist_text}

**Disclaimer**  
This is for paper decision support and class demonstration only. It is not personalized financial advice.
""".strip()
        report = self._sanitize_investment_wording(report)
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
            "report_quality_score": self._report_quality_score(report, source_grounded=True),
            "summary": f"Local fallback generated a single-stock explanation for {symbol}.",
        }

    # ------------------------------------------------------------------
    # Screener report
    # ------------------------------------------------------------------
    def generate_screener_report(self, user_question: Optional[str] = None, screener_result: Optional[Dict[str, Any]] = None, *args, **kwargs) -> Dict[str, Any]:
        # Backward compatibility: app may call generate_screener_report(screener_result, user_question=...)
        if isinstance(user_question, dict) and screener_result is None:
            screener_result = user_question
            user_question = kwargs.get("user_question") or "Explain the screener result."
        screener_result = screener_result or kwargs.get("screener_result") or {}
        user_question = user_question or kwargs.get("question") or "Explain the screener result."

        top = screener_result.get("top_buy_candidates", [])[:5]
        risk = screener_result.get("highest_risk_candidates", screener_result.get("top_sell_risk", []))[:5]
        facts = {"question": user_question, "top_candidates": top, "caution_candidates": risk}
        prompt = f"Explain this watchlist screener briefly. Use no direct trading advice.\n\nFacts:\n{self._safe_json(facts)}"
        groq = self._call_groq(self._short_system_prompt("Explain a watchlist screener."), prompt, max_tokens=700)
        if groq.get("success"):
            return {
                "success": True,
                "agent": "Groq Report Agent",
                "report_type": "screener_explanation",
                "source": "groq",
                "provider": "groq",
                "model": self.model,
                "llm_available": True,
                "llm_error": None,
                "plain_language_report": groq["text"],
                "report_quality_score": self._report_quality_score(groq["text"], source_grounded=True),
                "summary": "Groq Report Agent explained the screener result.",
            }
        return self._fallback_screener(top, risk, groq.get("error"))

    def _fallback_screener(self, top: List[Dict[str, Any]], risk: List[Dict[str, Any]], error: Optional[str] = None) -> Dict[str, Any]:
        top_names = ", ".join([r.get("symbol", "") for r in top if r.get("symbol")]) or "none"
        risk_names = ", ".join([r.get("symbol", "") for r in risk if r.get("symbol")]) or "none"
        report = (
            f"**Direct answer**\nThe strongest watchlist names for further research are: {top_names}.\n\n"
            f"**Caution names**\nThe names needing more caution are: {risk_names}.\n\n"
            "**Risk note**\nThis is a watchlist screener, not a full-market scan or a direct trading instruction."
        )
        return {
            "success": True,
            "agent": "Groq Report Agent",
            "report_type": "screener_explanation",
            "source": "local_fallback",
            "provider": "local_fallback",
            "llm_available": False,
            "llm_error": error,
            "plain_language_report": report,
            "report_quality_score": self._report_quality_score(report, source_grounded=True),
            "summary": "Local fallback explained the screener result.",
        }

    # ------------------------------------------------------------------
    # Financial report / news summariser
    # ------------------------------------------------------------------
    def simplify_financial_text(
        self,
        report_text: str = "",
        user_question: str = "Please simplify this financial report or news text.",
        question: Optional[str] = None,
        source_mode: str = "auto",
        symbol: Optional[str] = None,
        ticker_override: Optional[str] = None,
        lookback_days: int = 7,
        max_news: int = 5,
        max_news_items: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        report_text = report_text or ""
        user_question = question or user_question or kwargs.get("financial_question") or "Summarise this financial/news input."
        if max_news_items is not None:
            max_news = max_news_items
        context = self._build_source_context(
            report_text=report_text,
            source_mode=source_mode,
            symbol=symbol,
            ticker_override=ticker_override,
            lookback_days=int(lookback_days or 7),
            max_news=int(max_news or 5),
        )
        if not report_text.strip() and not context.get("symbol"):
            return {
                "success": False,
                "agent": "Groq Report Agent",
                "report_type": "financial_news_or_report_simplification",
                "source": "none",
                "llm_available": False,
                "plain_language_report": "No text or ticker was provided.",
                "summary": "No text or ticker was provided.",
            }

        system_prompt = self._short_system_prompt("Summarise source-grounded company news or financial text.")
        user_prompt = (
            "Use these headings: Summary, Sources checked, Relevant company news, Excluded broad news, Financial snapshot, Risks, Cautious conclusion.\n"
            "Only use the context below. If relevant news is limited, say so.\n\n"
            f"User question: {user_question}\n\nContext:\n{self._safe_json(context)}"
        )
        groq = self._call_groq(system_prompt, user_prompt, max_tokens=1000)
        if groq.get("success"):
            report = groq["text"]
            source = "groq"
            llm_available = True
            llm_error = None
            summary = "Groq financial/news summary generated successfully."
        else:
            report = self._fallback_financial_summary(context)
            source = "local_fallback"
            llm_available = False
            llm_error = groq.get("error")
            summary = "Local fallback financial/news summary generated successfully."

        return {
            "success": True,
            "agent": "Groq Report Agent",
            "agent_goal": "Summarise financial/news input with source-grounded context.",
            "report_type": "financial_news_or_report_simplification",
            "source": source,
            "provider": "groq" if source == "groq" else "local_fallback",
            "model": self.model,
            "llm_available": llm_available,
            "llm_error": llm_error,
            "symbol": context.get("symbol"),
            "source_mode": context.get("effective_source_mode"),
            "requested_source_mode": context.get("requested_source_mode"),
            "source_status": context.get("source_status", []),
            "company_specific_news": context.get("company_specific_news", []),
            "excluded_news": context.get("excluded_news", []),
            "verified_news": context.get("verified_news", {}),
            "financial_snapshot": context.get("financial_snapshot", {}),
            "plain_language_report": report,
            "report_quality_score": self._report_quality_score(report, source_grounded=bool(context.get("company_specific_news") or context.get("financial_snapshot"))),
            "summary": summary,
        }

    def _fallback_financial_summary(self, context: Dict[str, Any]) -> str:
        symbol = context.get("symbol") or "the company"
        source_status = context.get("source_status", [])
        relevant = context.get("company_specific_news", [])
        excluded = context.get("excluded_news", [])
        snapshot = (context.get("financial_snapshot") or {}).get("snapshot", {})
        user_text = context.get("user_text", "")

        lines = []
        lines.append("**Summary**")
        if relevant or snapshot:
            lines.append(f"The summary uses source-grounded data for {symbol}. It does not add unstated facts.")
        elif user_text.strip():
            lines.append("The input was treated as pasted text or a short query, but limited source-grounded information was available.")
        else:
            lines.append("No useful input was provided.")

        lines.append("\n**Sources checked**")
        if source_status:
            lines.extend([f"- {s}" for s in source_status])
        else:
            lines.append("- No external source was checked.")

        lines.append("\n**Relevant company news**")
        if relevant:
            for i, item in enumerate(relevant[:5], 1):
                lines.append(f"{i}. {item.get('date')} | {item.get('source')} | {item.get('headline')}  ")
                lines.append(f"   Relevance: {item.get('relevance_score')} — {item.get('relevance_reason')}")
        else:
            lines.append("- No clearly company-specific news passed the relevance filter.")

        lines.append("\n**Excluded broad news**")
        if excluded:
            for i, item in enumerate(excluded[:5], 1):
                lines.append(f"{i}. {item.get('date')} | {item.get('source')} | {item.get('headline')}")
        else:
            lines.append("- No broad or uncertain news was excluded.")

        lines.append("\n**Financial snapshot**")
        if snapshot:
            lines.append(self._safe_json(snapshot, max_chars=1500))
        else:
            lines.append("- No verified financial snapshot was available.")

        lines.append("\n**Risks**")
        lines.append("- Headlines alone are not enough to make a real trading decision.")
        lines.append("- The source data may be incomplete, delayed, or rate-limited.")

        lines.append("\n**Cautious conclusion**")
        lines.append("Use this as a paper research summary only, not personalized financial advice.")
        return "\n".join(lines)

    # Backward-compatible aliases
    def simplify_financial_report(self, *args, **kwargs) -> Dict[str, Any]:
        return self.simplify_financial_text(*args, **kwargs)

    def simplify_report(self, *args, **kwargs) -> Dict[str, Any]:
        return self.simplify_financial_text(*args, **kwargs)

    def summarize_financial_text(self, *args, **kwargs) -> Dict[str, Any]:
        return self.simplify_financial_text(*args, **kwargs)

    def generate_screener_explanation(self, *args, **kwargs) -> Dict[str, Any]:
        return self.generate_screener_report(*args, **kwargs)

    def explain_screener_result(self, *args, **kwargs) -> Dict[str, Any]:
        return self.generate_screener_report(*args, **kwargs)

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        return self.generate_single_stock_report(*args, **kwargs)
