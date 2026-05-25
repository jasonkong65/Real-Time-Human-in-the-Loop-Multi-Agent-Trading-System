import os
import re
import json
import urllib.parse
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


class LLMReportAgent:
    """
    Groq Report Agent for the Human-in-the-Loop Multi-Agent Trading System.

    Main functions:
    1. Explain single-stock multi-agent pipeline results.
    2. Explain S&P-style screener results.
    3. Simplify pasted financial reports / earnings news / company announcements.
    4. For query-like inputs such as "MSFT news" or "Microsoft financial report",
       fetch source-grounded company news and/or financial snapshots when API keys are available.

    Safety design:
    - LLM explains and summarises; it does not execute trades.
    - LLM must not invent unsupported news, earnings, prices, or financial facts.
    - If Groq is unavailable, local fallback still succeeds.
    - If Finnhub / Alpha Vantage is unavailable, the output clearly states that verified data was not retrieved.
    """

    COMPANY_TO_TICKER = {
        "apple": "AAPL",
        "microsoft": "MSFT",
        "tesla": "TSLA",
        "nvidia": "NVDA",
        "amazon": "AMZN",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "meta": "META",
        "facebook": "META",
        "netflix": "NFLX",
        "amd": "AMD",
        "broadcom": "AVGO",
        "walmart": "WMT",
        "visa": "V",
        "mastercard": "MA",
        "costco": "COST",
        "disney": "DIS",
        "intel": "INTC",
        "qualcomm": "QCOM",
        "oracle": "ORCL",
        "cisco": "CSCO",
        "jpmorgan": "JPM",
        "jp morgan": "JPM",
        "home depot": "HD",
        "adobe": "ADBE",
        "pepsico": "PEP",
        "bank of america": "BAC",
        "unitedhealth": "UNH",
        "united health": "UNH",
    }

    SYMBOL_TO_COMPANY_KEYWORDS = {
        # Primary company identifiers and major products/services.
        # These are used to filter API news so that the summarizer does not mix
        # unrelated market headlines into company-specific summaries.
        "AAPL": ["apple", "aapl", "iphone", "ipad", "mac", "macbook", "ios", "app store", "vision pro"],
        "MSFT": ["microsoft", "msft", "azure", "windows", "xbox", "linkedin", "github", "copilot", "office", "teams"],
        "TSLA": ["tesla", "tsla", "elon musk", "model y", "model 3", "cybertruck", "ev"],
        "NVDA": ["nvidia", "nvda", "gpu", "cuda", "blackwell", "ai chip", "chips"],
        "AMZN": ["amazon", "amzn", "aws", "prime", "whole foods"],
        "GOOGL": ["google", "googl", "alphabet", "youtube", "android", "gemini"],
        "META": ["meta", "facebook", "instagram", "whatsapp", "threads", "metaverse"],
        "NFLX": ["netflix", "nflx"],
        "AMD": ["amd", "advanced micro devices", "ryzen", "epyc"],
        "AVGO": ["broadcom", "avgo", "vmware"],
        "WMT": ["walmart", "wmt"],
        "V": ["visa", "visa inc"],
        "MA": ["mastercard"],
        "COST": ["costco"],
        "DIS": ["disney", "espn", "disney+"],
        "INTC": ["intel", "intc"],
        "QCOM": ["qualcomm", "qcom", "snapdragon"],
        "ORCL": ["oracle", "orcl"],
        "CSCO": ["cisco", "csco"],
        "JPM": ["jpmorgan", "jp morgan", "jpm"],
        "HD": ["home depot"],
        "ADBE": ["adobe", "adbe", "creative cloud"],
        "PEP": ["pepsico", "pepsi"],
        "BAC": ["bank of america", "bofa", "bac"],
        "UNH": ["unitedhealth", "united health", "unh", "optum"],
    }

    POSITIVE_KEYWORDS = [
        "beat", "beats", "growth", "gains", "gain", "strong", "record",
        "raise", "raised", "upgrade", "upgraded", "surge", "surged",
        "profit rises", "revenue rises", "higher revenue", "expands",
        "partnership", "launch", "approval", "outperform", "bullish",
        "demand", "contract", "win", "wins"
    ]

    RISK_KEYWORDS = [
        "miss", "misses", "decline", "declined", "falls", "fell",
        "drop", "drops", "weak", "weaker", "warning", "warned",
        "lawsuit", "probe", "investigation", "antitrust", "regulation",
        "regulatory", "tariff", "layoffs", "cut", "cuts", "downgrade",
        "downgraded", "privacy", "breach", "loss", "slump", "pressure",
        "cost", "spending", "margin pressure", "risk"
    ]


    def _clean_api_key(self, key: Optional[str]) -> str:
        """Normalise API keys loaded from .env / Streamlit secrets."""
        if key is None:
            return ""

        text = str(key).strip()

        # Handles accidental values like GROQ_API_KEY=gsk_xxx copied as the value.
        if text.upper().startswith("GROQ_API_KEY="):
            text = text.split("=", 1)[1].strip()
        if text.upper().startswith("GROQ_API="):
            text = text.split("=", 1)[1].strip()
        if text.upper().startswith("GROQ_KEY="):
            text = text.split("=", 1)[1].strip()

        # Remove common quote wrappers from .env / secrets.
        text = text.strip().strip('"').strip("'").strip()
        return text

    def _key_hint(self) -> str:
        if not self.groq_api_key:
            return "not loaded"
        return f"loaded, starts with {self.groq_api_key[:6]}..., length={len(self.groq_api_key)}"

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 15
    ):
        raw_groq_key = (
            groq_api_key
            or os.getenv("GROQ_API_KEY")
            or os.getenv("GROQ_API")
            or os.getenv("GROQ_KEY")
            or ""
        )
        self.groq_api_key = self._clean_api_key(raw_groq_key)

        self.model = (model or os.getenv("GROQ_MODEL") or "llama-3.1-8b-instant").strip()
        self.timeout = timeout
        self.last_llm_error = None
        self.last_llm_raw_error = None

        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY", "")

        self.alpha_vantage_api_key = (
            os.getenv("ALPHA_VANTAGE_API_KEY")
            or os.getenv("ALPHAVANTAGE_API_KEY")
            or ""
        )

    # --------------------------------------------------
    # General helpers
    # --------------------------------------------------
    def _safe_json(self, obj: Any, max_chars: int = 10000) -> str:
        try:
            text = json.dumps(obj, ensure_ascii=False, indent=2, default=str)
        except Exception:
            text = str(obj)

        if len(text) > max_chars:
            return text[:max_chars] + "\n... [truncated]"

        return text

    def _http_get_json(self, url: str) -> Any:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 educational-stock-agent/1.0"
            }
        )

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read().decode("utf-8", errors="replace")
            return json.loads(raw)

    def _call_groq(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1000
    ) -> Dict[str, Any]:
        """
        Call Groq's OpenAI-compatible chat-completions endpoint.

        This version is intentionally diagnostic-friendly: it does not silently
        swallow Groq failures. If the call fails, the returned llm_error includes
        the HTTP status and response body, so Streamlit can show the real reason.
        """
        self.last_llm_error = None
        self.last_llm_raw_error = None

        if not self.groq_api_key:
            error = "GROQ_API_KEY is missing or was not loaded from .env / Streamlit secrets."
            self.last_llm_error = error
            return {
                "success": False,
                "llm_available": False,
                "llm_error": error,
                "llm_debug": {"api_key": self._key_hint(), "model": self.model},
                "text": ""
            }

        system_prompt = str(system_prompt or "").strip()
        user_prompt = str(user_prompt or "").strip()

        if not system_prompt or not user_prompt:
            error = "Groq call skipped because system_prompt or user_prompt was empty."
            self.last_llm_error = error
            return {
                "success": False,
                "llm_available": False,
                "llm_error": error,
                "llm_debug": {"api_key": self._key_hint(), "model": self.model},
                "text": ""
            }

        candidate_models = []
        for m in [self.model, "llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]:
            if m and m not in candidate_models:
                candidate_models.append(m)

        errors = []

        for model_name in candidate_models:
            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
                "stream": False,
            }

            request = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "educational-stock-agent/1.0",
                }
            )

            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8", errors="replace")
                    parsed = json.loads(raw)

                text = (
                    parsed.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                    .strip()
                )

                if not text:
                    error = f"Groq returned an empty response for model {model_name}."
                    errors.append(error)
                    continue

                # Update model to the model that actually worked.
                self.model = model_name

                return {
                    "success": True,
                    "llm_available": True,
                    "llm_error": None,
                    "llm_debug": {
                        "api_key": self._key_hint(),
                        "model_requested": model_name,
                        "model_used": parsed.get("model", model_name),
                        "provider": "groq",
                    },
                    "text": text
                }

            except urllib.error.HTTPError as e:
                try:
                    error_body = e.read().decode("utf-8", errors="replace")
                except Exception:
                    error_body = ""

                error = f"Groq HTTPError {e.code} for model {model_name}: {error_body or str(e)}"
                errors.append(error)

                # If the key/auth/rate limit is the problem, retrying other models will not help.
                if e.code in {401, 403, 408, 409, 413, 429, 500, 502, 503, 504}:
                    break

                # For 400-type model errors, try the next fallback model.
                continue

            except urllib.error.URLError as e:
                error = f"Groq URLError for model {model_name}: {str(e)}"
                errors.append(error)
                break

            except Exception as e:
                error = f"Groq call failed for model {model_name}: {repr(e)}"
                errors.append(error)
                break

        final_error = " | ".join(errors) if errors else "Unknown Groq call failure."
        self.last_llm_error = final_error
        self.last_llm_raw_error = errors

        return {
            "success": False,
            "llm_available": False,
            "llm_error": final_error,
            "llm_raw_errors": errors,
            "llm_debug": {
                "api_key": self._key_hint(),
                "model": self.model,
                "candidate_models": candidate_models,
                "provider": "groq",
            },
            "text": ""
        }

    def _format_markdown_report(
        self,
        direct_answer: str,
        evidence: List[str],
        risk_warning: str,
        strategy_guidance: str,
        leverage_guidance: Optional[str] = None,
        disclaimer: Optional[str] = None
    ) -> str:
        lines = []

        lines.append(f"**Direct Answer:** {direct_answer}")
        lines.append("")
        lines.append("**Evidence from Agents:**")

        for item in evidence:
            lines.append(f"- {item}")

        lines.append("")
        lines.append(f"**Strategy Guidance:** {strategy_guidance}")

        if leverage_guidance:
            lines.append("")
            lines.append(f"**Leverage Guidance:** {leverage_guidance}")

        lines.append("")
        lines.append(f"**Risk Warning:** {risk_warning}")

        lines.append("")
        lines.append(
            "**Not Financial Advice Disclaimer:** "
            + (
                disclaimer
                or "This output is for paper decision support and further research only. It is not personalized financial advice."
            )
        )

        return "\n".join(lines)

    # --------------------------------------------------
    # Text / symbol helpers
    # --------------------------------------------------
    def _extract_symbol(
        self,
        text: str,
        explicit_symbol: Optional[str] = None
    ) -> Optional[str]:
        if explicit_symbol:
            return explicit_symbol.strip().upper()

        text = text or ""

        stop_words = {
            "THE", "AND", "FOR", "NEWS", "THIS", "WEEK", "LAST", "YEAR",
            "YEARS", "REPORT", "BUY", "SELL", "HOLD", "AI", "API", "USA",
            "CEO", "EPS", "Q", "A", "AN", "OF", "TO", "IN", "ON", "ABOUT",
            "LATEST", "RECENT", "FINANCIAL", "COMPANY", "MARKET"
        }

        for token in re.findall(r"\b[A-Z]{1,5}\b", text):
            if token not in stop_words:
                return token

        lower_text = text.lower()

        for company, ticker in self.COMPANY_TO_TICKER.items():
            if company in lower_text:
                return ticker

        return None

    def _is_query_like(self, text: str) -> bool:
        text_l = (text or "").lower().strip()

        query_terms = [
            "news",
            "this week",
            "latest",
            "recent",
            "financial report",
            "annual report",
            "earnings",
            "last year",
            "last years",
            "fetch",
            "search",
            "report for",
            "about",
            "quarterly",
            "income statement",
            "company news",
            "what happened"
        ]

        return any(term in text_l for term in query_terms)

    def _has_concrete_pasted_content(self, text: str) -> bool:
        text_l = (text or "").lower()
        words = re.findall(r"\b\w+\b", text_l)

        concrete_terms = [
            "reported", "announced", "revenue", "profit", "margin",
            "earnings", "eps", "guidance", "growth", "decline",
            "management", "quarter", "annual", "cash flow", "operating",
            "net income", "cloud", "sales", "demand", "cost", "spending",
            "capex", "warning", "forecast", "reported that", "said that"
        ]

        has_concrete_term = any(term in text_l for term in concrete_terms)
        has_number = bool(re.search(r"(\$|%|\d)", text_l))

        return len(words) >= 8 and (has_concrete_term or has_number)

    def _safe_int_or_none(self, value: Any) -> Optional[int]:
        try:
            if value is None:
                return None

            text = str(value).strip()

            if text in ["", "None", "none", "null", "NoneType"]:
                return None

            return int(float(text))

        except Exception:
            return None

    def _format_large_number(self, value: Any) -> str:
        number = self._safe_int_or_none(value)

        if number is None:
            return "not available"

        abs_number = abs(number)

        if abs_number >= 1_000_000_000:
            return f"{number / 1_000_000_000:.2f}B"

        if abs_number >= 1_000_000:
            return f"{number / 1_000_000:.2f}M"

        return f"{number:,}"

    def _normalise_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()

    def _news_relevance_score(
        self,
        symbol: Optional[str],
        item: Dict[str, Any]
    ) -> Tuple[int, List[str]]:
        """
        Score whether a returned API headline is genuinely about the target company.

        Finnhub's company-news endpoint can sometimes return broad-market or ETF
        headlines. This filter keeps only news with explicit company/product
        evidence, so the summarizer does not mix unrelated market headlines into
        a company-specific report.
        """
        if not symbol:
            return 0, ["no target symbol"]

        symbol_u = symbol.upper()
        symbol_l = symbol_u.lower()

        headline = self._normalise_text(item.get("headline", ""))
        summary = self._normalise_text(item.get("summary", ""))
        combined = f"{headline} {summary}".lower()

        keywords = self.SYMBOL_TO_COMPANY_KEYWORDS.get(symbol_u, [symbol_l])
        primary_terms = [symbol_l]

        # Add company names mapped to this ticker.
        for company_name, ticker in self.COMPANY_TO_TICKER.items():
            if ticker == symbol_u:
                primary_terms.append(company_name.lower())

        score = 0
        reasons: List[str] = []

        # Exact ticker match is strong evidence only when it appears as a standalone token.
        if re.search(rf"\b{re.escape(symbol_u)}\b", headline, flags=re.IGNORECASE) or re.search(
            rf"\b{re.escape(symbol_u)}\b", summary, flags=re.IGNORECASE
        ):
            score += 4
            reasons.append("ticker mentioned")

        # Company name / primary identifier.
        for term in primary_terms:
            if term and term in combined:
                score += 4
                reasons.append(f"company term: {term}")
                break

        # Major product/service keywords are weaker but still useful.
        product_hits = []
        for term in keywords:
            term_l = term.lower()
            if term_l in primary_terms:
                continue
            if term_l and term_l in combined:
                product_hits.append(term_l)

        if product_hits:
            score += min(3, len(product_hits))
            reasons.append("product/service terms: " + ", ".join(product_hits[:3]))

        # Headline evidence is more important than a passing mention in the summary.
        headline_l = headline.lower()
        if any(term in headline_l for term in primary_terms if term):
            score += 2
            reasons.append("company term in headline")

        # Penalise broad-market / ETF headlines unless they also contain target evidence.
        broad_terms = [
            "s&p", "sp 500", "s&p 500", "nasdaq", "dow", "qqq", "etf",
            "market bubble", "market rally", "stocks", "index", "indexes",
            "portfolio", "yield", "dividend", "bofa strategist"
        ]
        if any(term in combined for term in broad_terms) and score < 5:
            score -= 3
            reasons.append("broad-market/ETF wording")

        # Penalise unrelated famous-company headlines when target evidence is weak.
        other_company_terms = [
            "spacex", "elon musk", "trump", "nuclear", "italian stocks",
            "bank of america", "bofa", "qqq", "nasdaq-100"
        ]
        if any(term in combined for term in other_company_terms) and score < 5:
            score -= 2
            reasons.append("likely unrelated headline")

        return score, reasons or ["no clear company-specific evidence"]

    def _filter_company_specific_news(
        self,
        symbol: Optional[str],
        items: List[Dict[str, Any]],
        max_items: int = 5
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return filtered company-specific news and excluded broad/uncertain items."""
        kept: List[Dict[str, Any]] = []
        excluded: List[Dict[str, Any]] = []

        for item in items:
            score, reasons = self._news_relevance_score(symbol, item)
            enriched = dict(item)
            enriched["relevance_score"] = score
            enriched["relevance_reason"] = "; ".join(reasons)

            if score >= 4:
                kept.append(enriched)
            else:
                excluded.append(enriched)

        # Preserve API date/order, but keep only the strongest relevant results if there are too many.
        kept = sorted(kept, key=lambda x: x.get("relevance_score", 0), reverse=True)
        return kept[:max_items], excluded[:10]

    # --------------------------------------------------
    # API fetchers
    # --------------------------------------------------
    def _fetch_finnhub_company_news(
        self,
        symbol: str,
        lookback_days: int = 7,
        max_items: int = 5
    ) -> Dict[str, Any]:
        if not self.finnhub_api_key:
            return {
                "success": False,
                "source": "finnhub_company_news",
                "symbol": symbol,
                "error": "FINNHUB_API_KEY is missing.",
                "items": [],
                "raw_items": [],
                "excluded_items": [],
                "raw_count": 0,
                "company_specific_count": 0,
                "excluded_count": 0
            }

        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=max(1, int(lookback_days or 7)))

        params = {
            "symbol": symbol,
            "from": start.isoformat(),
            "to": today.isoformat(),
            "token": self.finnhub_api_key
        }

        url = "https://finnhub.io/api/v1/company-news?" + urllib.parse.urlencode(params)

        try:
            data = self._http_get_json(url)

            if not isinstance(data, list):
                return {
                    "success": False,
                    "source": "finnhub_company_news",
                    "symbol": symbol,
                    "error": "Finnhub returned non-list data.",
                    "items": [],
                    "raw_items": [],
                    "excluded_items": [],
                    "raw_count": 0,
                    "company_specific_count": 0,
                    "excluded_count": 0
                }

            # Scan more than max_items because the first few API results may be broad-market
            # headlines even for a company-news request.
            scan_limit = min(len(data), max(max_items * 8, 40))
            raw_items: List[Dict[str, Any]] = []

            for item in data[:scan_limit]:
                dt = item.get("datetime")
                date_str = "Unknown"

                if dt:
                    try:
                        date_str = datetime.fromtimestamp(
                            int(dt),
                            timezone.utc
                        ).strftime("%Y-%m-%d")
                    except Exception:
                        date_str = str(dt)

                headline = self._normalise_text(item.get("headline", ""))
                summary = self._normalise_text(item.get("summary", ""))

                if not headline and not summary:
                    continue

                raw_items.append(
                    {
                        "date": date_str,
                        "headline": headline,
                        "source": item.get("source", ""),
                        "summary": summary,
                        "url": item.get("url", "")
                    }
                )

            filtered_items, excluded_items = self._filter_company_specific_news(
                symbol=symbol,
                items=raw_items,
                max_items=max_items
            )

            return {
                "success": True,
                "source": "finnhub_company_news",
                "symbol": symbol,
                "lookback_days": lookback_days,
                "count": len(filtered_items),
                "raw_count": len(raw_items),
                "company_specific_count": len(filtered_items),
                "excluded_count": len(excluded_items),
                "items": filtered_items,
                "raw_items": raw_items[:10],
                "excluded_items": excluded_items,
                "filter_note": (
                    "items contains only company-specific news after relevance filtering. "
                    "raw_items/excluded_items are kept for audit/debug only."
                ),
                "error": None
            }

        except Exception as e:
            return {
                "success": False,
                "source": "finnhub_company_news",
                "symbol": symbol,
                "error": str(e),
                "items": [],
                "raw_items": [],
                "excluded_items": [],
                "raw_count": 0,
                "company_specific_count": 0,
                "excluded_count": 0
            }

    def _fetch_alpha_vantage_financial_snapshot(
        self,
        symbol: str
    ) -> Dict[str, Any]:
        if not self.alpha_vantage_api_key:
            return {
                "success": False,
                "source": "alpha_vantage_financial_snapshot",
                "symbol": symbol,
                "error": "ALPHA_VANTAGE_API_KEY is missing.",
                "snapshot": {}
            }

        result = {
            "success": False,
            "source": "alpha_vantage_financial_snapshot",
            "symbol": symbol,
            "error": None,
            "snapshot": {}
        }

        try:
            income_params = {
                "function": "INCOME_STATEMENT",
                "symbol": symbol,
                "apikey": self.alpha_vantage_api_key
            }

            income_url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(income_params)
            income_data = self._http_get_json(income_url)

            quarterly_reports = (
                income_data.get("quarterlyReports", [])
                if isinstance(income_data, dict)
                else []
            )

            annual_reports = (
                income_data.get("annualReports", [])
                if isinstance(income_data, dict)
                else []
            )

            if quarterly_reports:
                q = quarterly_reports[0]
                result["snapshot"]["latest_quarter"] = {
                    "fiscalDateEnding": q.get("fiscalDateEnding"),
                    "reportedCurrency": q.get("reportedCurrency"),
                    "totalRevenue": q.get("totalRevenue"),
                    "grossProfit": q.get("grossProfit"),
                    "operatingIncome": q.get("operatingIncome"),
                    "netIncome": q.get("netIncome")
                }

            if annual_reports:
                a = annual_reports[0]
                result["snapshot"]["latest_annual"] = {
                    "fiscalDateEnding": a.get("fiscalDateEnding"),
                    "reportedCurrency": a.get("reportedCurrency"),
                    "totalRevenue": a.get("totalRevenue"),
                    "grossProfit": a.get("grossProfit"),
                    "operatingIncome": a.get("operatingIncome"),
                    "netIncome": a.get("netIncome")
                }

        except Exception as e:
            result["error"] = f"Income statement fetch failed: {e}"

        try:
            earnings_params = {
                "function": "EARNINGS",
                "symbol": symbol,
                "apikey": self.alpha_vantage_api_key
            }

            earnings_url = "https://www.alphavantage.co/query?" + urllib.parse.urlencode(earnings_params)
            earnings_data = self._http_get_json(earnings_url)

            quarterly_earnings = (
                earnings_data.get("quarterlyEarnings", [])
                if isinstance(earnings_data, dict)
                else []
            )

            if quarterly_earnings:
                e = quarterly_earnings[0]
                result["snapshot"]["latest_earnings"] = {
                    "fiscalDateEnding": e.get("fiscalDateEnding"),
                    "reportedDate": e.get("reportedDate"),
                    "reportedEPS": e.get("reportedEPS"),
                    "estimatedEPS": e.get("estimatedEPS"),
                    "surprise": e.get("surprise"),
                    "surprisePercentage": e.get("surprisePercentage")
                }

        except Exception as e:
            if result["error"]:
                result["error"] += f"; earnings fetch failed: {e}"
            else:
                result["error"] = f"Earnings fetch failed: {e}"

        if result["snapshot"]:
            result["success"] = True

        return result

    # --------------------------------------------------
    # Source-grounded context building
    # --------------------------------------------------
    def _build_verified_context(
        self,
        user_text: str,
        symbol: Optional[str],
        source_mode: str,
        lookback_days: int,
        max_news: int
    ) -> Dict[str, Any]:
        symbol = self._extract_symbol(user_text, explicit_symbol=symbol)

        context = {
            "mode": source_mode,
            "symbol": symbol,
            "user_text": user_text,
            "pasted_content_used": False,
            "verified_news": {},
            "financial_snapshot": {},
            "source_status": []
        }

        if not symbol:
            context["source_status"].append(
                "No ticker/company name was detected, so no live source was fetched."
            )
            return context

        mode_l = (source_mode or "auto").lower()
        text_l = (user_text or "").lower()

        if mode_l == "auto":
            should_fetch_news = any(
                x in text_l
                for x in [
                    "news",
                    "recent",
                    "this week",
                    "latest",
                    "what happened",
                    "company news",
                    "headline",
                    "announcement"
                ]
            )

            should_fetch_financial = any(
                x in text_l
                for x in [
                    "report",
                    "earnings",
                    "financial",
                    "annual",
                    "last year",
                    "quarter",
                    "income statement",
                    "revenue",
                    "profit",
                    "eps"
                ]
            )

            if not should_fetch_news and not should_fetch_financial:
                should_fetch_news = True
                should_fetch_financial = True

        else:
            should_fetch_news = mode_l in [
                "news",
                "news_and_financial",
                "both"
            ]

            should_fetch_financial = mode_l in [
                "financial",
                "news_and_financial",
                "both"
            ]

        if should_fetch_news:
            news_result = self._fetch_finnhub_company_news(
                symbol=symbol,
                lookback_days=lookback_days,
                max_items=max_news
            )

            context["verified_news"] = news_result

            if news_result.get("success"):
                raw_count = news_result.get("raw_count", 0)
                kept_count = news_result.get("company_specific_count", 0)
                excluded_count = news_result.get("excluded_count", 0)

                if kept_count:
                    context["source_status"].append(
                        f"Fetched {raw_count} Finnhub items for {symbol}; kept {kept_count} company-specific items and excluded {excluded_count} broad/uncertain items."
                    )
                else:
                    context["source_status"].append(
                        f"Fetched {raw_count} Finnhub items for {symbol}, but no company-specific items passed the relevance filter. Broad-market headlines were excluded."
                    )
            else:
                context["source_status"].append(
                    f"Finnhub news was not available: {news_result.get('error')}"
                )

        if should_fetch_financial:
            financial_result = self._fetch_alpha_vantage_financial_snapshot(symbol=symbol)
            context["financial_snapshot"] = financial_result

            if financial_result.get("success"):
                context["source_status"].append(
                    f"Fetched financial snapshot from Alpha Vantage for {symbol}."
                )
            else:
                context["source_status"].append(
                    f"Alpha Vantage financial snapshot was not available: {financial_result.get('error')}"
                )

        return context

    # --------------------------------------------------
    # News analysis helpers for fallback
    # --------------------------------------------------
    def _classify_news_items(
        self,
        symbol: Optional[str],
        news_items: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Backward-compatible classifier.

        In the optimized version, _fetch_finnhub_company_news already filters
        items. This method remains for pasted/legacy data and for fallback checks.
        """
        if not symbol:
            return news_items, []

        company_specific, excluded = self._filter_company_specific_news(
            symbol=symbol,
            items=news_items,
            max_items=len(news_items) or 5
        )
        return company_specific, excluded

    def _detect_keyword_items(
        self,
        news_items: List[Dict[str, Any]],
        keywords: List[str]
    ) -> List[Dict[str, Any]]:
        matched = []

        for item in news_items:
            combined = (
                str(item.get("headline", "")) + " "
                + str(item.get("summary", ""))
            ).lower()

            if any(keyword in combined for keyword in keywords):
                matched.append(item)

        return matched

    def _format_news_headline_line(
        self,
        item: Dict[str, Any],
        idx: int
    ) -> str:
        date = item.get("date") or "Unknown date"
        source = item.get("source") or "Unknown source"
        headline = item.get("headline") or "No headline"

        return f"{idx}. {date} | {source} | {headline}"

    def _format_financial_snapshot_lines(
        self,
        snapshot: Dict[str, Any]
    ) -> List[str]:
        lines = []

        latest_quarter = snapshot.get("latest_quarter")
        latest_annual = snapshot.get("latest_annual")
        latest_earnings = snapshot.get("latest_earnings")

        if latest_quarter:
            currency = latest_quarter.get("reportedCurrency") or ""
            lines.append(
                "- Latest quarter "
                f"({latest_quarter.get('fiscalDateEnding', 'unknown date')}): "
                f"revenue {self._format_large_number(latest_quarter.get('totalRevenue'))} {currency}, "
                f"operating income {self._format_large_number(latest_quarter.get('operatingIncome'))} {currency}, "
                f"net income {self._format_large_number(latest_quarter.get('netIncome'))} {currency}."
            )

        if latest_annual:
            currency = latest_annual.get("reportedCurrency") or ""
            lines.append(
                "- Latest annual period "
                f"({latest_annual.get('fiscalDateEnding', 'unknown date')}): "
                f"revenue {self._format_large_number(latest_annual.get('totalRevenue'))} {currency}, "
                f"operating income {self._format_large_number(latest_annual.get('operatingIncome'))} {currency}, "
                f"net income {self._format_large_number(latest_annual.get('netIncome'))} {currency}."
            )

        if latest_earnings:
            lines.append(
                "- Latest earnings "
                f"({latest_earnings.get('reportedDate', 'unknown report date')}): "
                f"reported EPS {latest_earnings.get('reportedEPS', 'not available')}, "
                f"estimated EPS {latest_earnings.get('estimatedEPS', 'not available')}, "
                f"surprise {latest_earnings.get('surprise', 'not available')}, "
                f"surprise percentage {latest_earnings.get('surprisePercentage', 'not available')}."
            )

        return lines

    # --------------------------------------------------
    # Financial report fallback
    # --------------------------------------------------
    def _fallback_financial_simplification(
        self,
        context: Dict[str, Any]
    ) -> str:
        user_text = context.get("user_text", "") or ""
        symbol = context.get("symbol") or "the company"
        source_status = context.get("source_status", []) or []

        news = context.get("verified_news", {}) or {}
        financial = context.get("financial_snapshot", {}) or {}

        # In the optimized fetcher, news["items"] already contains only
        # company-specific, relevance-filtered items.
        news_items = news.get("items", []) if isinstance(news, dict) else []
        excluded_items = news.get("excluded_items", []) if isinstance(news, dict) else []
        raw_count = news.get("raw_count", 0) if isinstance(news, dict) else 0
        snapshot = financial.get("snapshot", {}) if isinstance(financial, dict) else {}

        positive_items = self._detect_keyword_items(news_items, self.POSITIVE_KEYWORDS)
        risk_items = self._detect_keyword_items(news_items, self.RISK_KEYWORDS)

        financial_lines = self._format_financial_snapshot_lines(snapshot)

        lines = []

        lines.append("**Summary:**")

        if news_items or snapshot:
            lines.append(
                f"The system used source-grounded information for {symbol}. "
                "It did not invent unstated facts. The output below is based only on filtered API data and/or pasted text."
            )

        elif raw_count and not news_items:
            lines.append(
                f"Finnhub returned {raw_count} items for {symbol}, but none passed the company-specific relevance filter. "
                "The report therefore avoids drawing conclusions from unrelated market headlines."
            )

        elif self._has_concrete_pasted_content(user_text):
            lines.append(
                f"The pasted text says: {user_text.strip()}"
            )

        else:
            lines.append(
                f"The input is limited: {user_text.strip()!r}. "
                "No reliable financial conclusion can be made unless verified live sources are available or more report/news text is pasted."
            )

        lines.append("")
        lines.append("**Verified Source Status:**")

        if source_status:
            for status in source_status:
                lines.append(f"- {status}")
        else:
            lines.append("- No external source was used.")

        if news_items:
            lines.append("")
            lines.append("**Company-Specific Retrieved News:**")

            for idx, item in enumerate(news_items[:5], start=1):
                line = self._format_news_headline_line(item, idx)
                reason = item.get("relevance_reason")
                if reason:
                    line += f"  \n  Relevance: {reason}."
                lines.append(line)

        elif raw_count:
            lines.append("")
            lines.append("**Company-Specific Retrieved News:**")
            lines.append(
                "- No company-specific headline was kept after filtering. Broad or uncertain headlines were excluded from the plain-language report."
            )

        if excluded_items:
            lines.append("")
            lines.append("**Excluded Broad/Uncertain Items:**")
            lines.append(
                f"- {len(excluded_items)} broad or low-confidence items were excluded from the report to avoid mixing unrelated market news into the company summary."
            )

        if financial_lines:
            lines.append("")
            lines.append("**Financial Snapshot:**")

            for line in financial_lines:
                lines.append(line)

        lines.append("")
        lines.append("**Positive Signals:**")

        if positive_items:
            for item in positive_items[:3]:
                headline = item.get("headline") or "No headline"
                source = item.get("source") or "Unknown source"
                lines.append(
                    f'- Possible positive cue from company-specific retrieved news: "{headline}" ({source}).'
                )
        elif financial_lines:
            lines.append(
                "- A verified financial snapshot was retrieved, but the fallback did not infer whether it is positive without comparison context."
            )
        elif self._has_concrete_pasted_content(user_text):
            lines.append(
                "- The pasted text may contain useful context, but no clear positive signal was identified by fallback keyword review."
            )
        else:
            lines.append(
                "- No specific positive company signal can be confidently identified from the available source-grounded input."
            )

        lines.append("")
        lines.append("**Negative Signals / Risks:**")

        if risk_items:
            for item in risk_items[:3]:
                headline = item.get("headline") or "No headline"
                source = item.get("source") or "Unknown source"
                lines.append(
                    f'- Possible risk cue from company-specific retrieved news: "{headline}" ({source}).'
                )
        else:
            lines.append(
                "- No specific company risk signal was detected by fallback keyword review."
            )

        lines.append(
            "- This report may still be incomplete if API keys are missing, rate-limited, or no recent company-specific news is available."
        )
        lines.append(
            "- Headlines should be checked against full articles and official filings before making any decision."
        )
        lines.append(
            "- This module does not perform valuation, portfolio suitability analysis, or personalized investment advice."
        )

        lines.append("")
        lines.append("**Possible Market Impact:**")

        if news_items:
            lines.append(
                f"The filtered company-specific headlines may affect market interpretation of {symbol}, but the direction and size of market impact cannot be concluded from headlines alone."
            )
        elif financial_lines:
            lines.append(
                "The retrieved financial snapshot can support further review, but market impact requires comparison with expectations and prior periods."
            )
        elif self._has_concrete_pasted_content(user_text):
            lines.append(
                "The pasted text may affect interpretation, but market impact cannot be strongly assessed without fuller context."
            )
        else:
            lines.append(
                "Market impact cannot be assessed from the available input alone."
            )

        lines.append("")
        lines.append("**Cautious Conclusion:**")
        lines.append(
            "Use this as a source-grounded research summary only. "
            "It is not a direct buy/sell instruction and not personalized financial advice."
        )

        return "\n".join(lines)

    # --------------------------------------------------
    # Financial report / news simplifier
    # --------------------------------------------------
    def simplify_financial_text(
        self,
        report_text: str = "",
        question: Optional[str] = None,
        user_question: Optional[str] = None,
        source_mode: str = "auto",
        symbol: Optional[str] = None,
        lookback_days: int = 7,
        max_news: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Simplify financial/news/report text, or fetch source-grounded news/financial snapshot
        for query-like inputs such as "MSFT news".

        Compatible with both:
        - question=
        - user_question=
        """
        report_text = report_text or ""

        if question is None:
            question = (
                user_question
                or kwargs.get("financial_question")
                or kwargs.get("prompt")
                or "Simplify this financial report/news text. Identify the main positive signals, risks, and possible market impact. Do not provide trading advice."
            )

        if not report_text.strip():
            return {
                "success": False,
                "agent": "Groq Report Agent",
                "agent_goal": "Simplify financial/news/report text or source-grounded company snapshot in plain language.",
                "report_type": "financial_news_or_report_simplification",
                "source": "none",
                "provider": "groq",
                "model": self.model,
                "llm_available": False,
                "llm_error": "No input text was provided.",
                "symbol": None,
                "source_mode": source_mode,
                "source_status": [],
                "verified_news": {},
                "financial_snapshot": {},
                "plain_language_report": "No report/news text or ticker query was provided.",
                "summary": "No input text was provided."
            }

        is_concrete_paste = self._has_concrete_pasted_content(report_text)
        is_query = self._is_query_like(report_text)

        if is_concrete_paste and not is_query:
            context = {
                "mode": "pasted_text",
                "symbol": self._extract_symbol(report_text, explicit_symbol=symbol),
                "user_text": report_text,
                "pasted_content_used": True,
                "verified_news": {},
                "financial_snapshot": {},
                "source_status": [
                    "Used pasted text only. No live news or financial API was called."
                ]
            }
        else:
            context = self._build_verified_context(
                user_text=report_text,
                symbol=symbol,
                source_mode=source_mode,
                lookback_days=lookback_days,
                max_news=max_news
            )

        system_prompt = (
            "You are a financial report/news simplification agent for an educational multi-agent stock analysis prototype. "
            "Use only the pasted text or verified API context provided. "
            "Do not invent news, prices, earnings, or financial facts. "
            "Important: verified_news.items has already been filtered to company-specific news. "
            "Do not summarize raw_items or excluded_items as if they were company-specific. "
            "If verified_news.items is empty, clearly say that no company-specific news passed the relevance filter. "
            "Do not give direct buy/sell/clear-position/leverage instructions. "
            "Write in clear sections: Summary, Verified Source Status, Company-Specific Retrieved News if available, "
            "Financial Snapshot if available, Positive Signals, Negative Signals/Risks, Possible Market Impact, Cautious Conclusion."
        )

        user_prompt = (
            f"User question:\n{question}\n\n"
            f"Source-grounded context:\n{self._safe_json(context)}"
        )

        groq_result = self._call_groq(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=1200
        )

        if groq_result.get("success"):
            plain_report = groq_result["text"]
            source = "groq"
            llm_available = True
            llm_error = None
            summary = "Groq financial report/news simplification generated successfully."
        else:
            plain_report = self._fallback_financial_simplification(context)
            source = "local_fallback"
            llm_available = False
            llm_error = groq_result.get("llm_error")
            summary = "Local fallback financial report/news simplification generated successfully."

        return {
            "success": True,
            "agent": "Groq Report Agent",
            "agent_goal": "Simplify financial/news/report text or source-grounded company snapshot in plain language.",
            "report_type": "financial_news_or_report_simplification",
            "source": source,
            "provider": "groq",
            "model": self.model,
            "llm_available": llm_available,
            "llm_error": llm_error,
            "llm_debug": groq_result.get("llm_debug"),
            "symbol": context.get("symbol"),
            "source_mode": source_mode,
            "source_status": context.get("source_status", []),
            "verified_news": context.get("verified_news", {}),
            "financial_snapshot": context.get("financial_snapshot", {}),
            "plain_language_report": plain_report,
            "summary": summary
        }

    def simplify_financial_report(self, *args, **kwargs) -> Dict[str, Any]:
        return self.simplify_financial_text(*args, **kwargs)

    def simplify_report(self, *args, **kwargs) -> Dict[str, Any]:
        return self.simplify_financial_text(*args, **kwargs)

    def summarize_financial_text(self, *args, **kwargs) -> Dict[str, Any]:
        return self.simplify_financial_text(*args, **kwargs)

    # --------------------------------------------------
    # Single-stock pipeline report
    # --------------------------------------------------
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
        validation_result = validation_result or {}
        analysis_result = analysis_result or {}
        training_result = training_result or {}
        signal_result = signal_result or {}
        risk_result = risk_result or {}
        strategy_result = strategy_result or {}
        reward_record_result = reward_record_result or {}
        auto_reward_update_result = auto_reward_update_result or {}

        symbol = (
            risk_result.get("symbol")
            or risk_result.get("risk_for_next_agent", {}).get("symbol")
            or signal_result.get("symbol")
            or signal_result.get("signal_for_next_agent", {}).get("symbol")
            or analysis_result.get("symbol")
            or validation_result.get("validation_for_next_agent", {}).get("symbol")
            or "UNKNOWN"
        )

        structured_context = {
            "symbol": symbol,
            "user_question": user_question,
            "validation_result": validation_result,
            "analysis_result": analysis_result,
            "training_result": training_result,
            "signal_result": signal_result,
            "risk_result": risk_result,
            "strategy_result": strategy_result,
            "reward_record_result": reward_record_result,
            "auto_reward_update_result": auto_reward_update_result
        }

        system_prompt = (
            "You are an LLM report agent for an educational human-in-the-loop stock analysis system. "
            "Explain only the structured agent outputs in plain English. "
            "Do not give personalized financial advice. "
            "Do not create facts outside the provided agent outputs. "
            "Use sections: Direct Answer, Evidence from Agents, Strategy Guidance, "
            "Leverage Guidance, Risk Warning, Not Financial Advice Disclaimer."
        )

        user_prompt = (
            f"User question:\n{user_question}\n\n"
            f"Structured agent outputs:\n{self._safe_json(structured_context)}"
        )

        groq_result = self._call_groq(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=900
        )

        if groq_result.get("success"):
            report = groq_result["text"]
            source = "groq"
            llm_available = True
            llm_error = None
            summary = f"Groq Report Agent generated a single-stock explanation for {symbol}."
        else:
            validation_confidence = validation_result.get("confidence", "Unknown")
            analyst_signal = analysis_result.get("analyst_signal", "Unknown")
            analyst_score = analysis_result.get("analyst_score", "Unknown")

            model_signal = (
                signal_result.get("model_signal")
                or signal_result.get("signal")
                or "Unknown"
            )

            model_confidence = (
                signal_result.get("confidence_level")
                or signal_result.get("model_confidence_level")
                or "Unknown"
            )

            final_signal = risk_result.get("final_signal", model_signal)
            risk_level = risk_result.get("risk_level", "Unknown")

            strategy_action = strategy_result.get(
                "strategy_action",
                "Further research only"
            )

            strategy_level = strategy_result.get(
                "strategy_level",
                "Conservative"
            )

            position_guidance = strategy_result.get(
                "position_guidance",
                "Use this result as research support only and wait for clearer evidence."
            )

            leverage_guidance = strategy_result.get(
                "leverage_guidance",
                "Do not use leverage in this prototype."
            )

            report = self._format_markdown_report(
                direct_answer=(
                    f"{symbol} is not being presented as a direct buy or sell instruction. "
                    f"Based on the current agent outputs, the risk-controlled signal is {final_signal} "
                    f"and the strategy action is {strategy_action}."
                ),
                evidence=[
                    f"Validation confidence: {validation_confidence}",
                    f"Analyst signal: {analyst_signal}, analyst score: {analyst_score}",
                    f"Signal model output: {model_signal}",
                    f"Model confidence: {model_confidence}",
                    f"Risk Agent final signal: {final_signal}",
                    f"Risk level: {risk_level}",
                    f"Strategist level: {strategy_level}"
                ],
                strategy_guidance=position_guidance,
                leverage_guidance=leverage_guidance,
                risk_warning=(
                    "This fallback explanation was generated without a live Groq response. "
                    "The system output is only a paper decision-support interpretation, and market conditions can change quickly."
                )
            )

            source = "local_fallback"
            llm_available = False
            llm_error = groq_result.get("llm_error")
            summary = f"Local fallback generated a single-stock explanation for {symbol}."

        return {
            "success": True,
            "agent": "Groq Report Agent",
            "agent_goal": "Explain the single-stock multi-agent decision in natural language.",
            "report_type": "single_stock_recommendation_explanation",
            "source": source,
            "provider": "groq",
            "model": self.model,
            "llm_available": llm_available,
            "llm_error": llm_error,
            "llm_debug": groq_result.get("llm_debug"),
            "symbol": symbol,
            "plain_language_report": report,
            "summary": summary
        }

    # --------------------------------------------------
    # Screener report
    # --------------------------------------------------
    def generate_screener_report(
        self,
        screener_result: Dict[str, Any],
        user_question: Optional[str] = None
    ) -> Dict[str, Any]:
        screener_result = screener_result or {}

        user_question = (
            user_question
            or "Which stocks look strongest, which stocks need caution, and why?"
        )

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

        system_prompt = (
            "You are a report agent for an educational stock screener prototype. "
            "Explain only the screener result provided. "
            "Do not claim this is a full-market scan. "
            "Do not provide personalized financial advice. "
            "Use sections: Direct Answer, Top Candidates for Further Research, "
            "Higher-Risk / Caution Candidates, Evidence from Agents, Risk Warning, "
            "Not Financial Advice Disclaimer."
        )

        user_prompt = (
            f"User question:\n{user_question}\n\n"
            f"Screener result:\n{self._safe_json(compact_context)}"
        )

        groq_result = self._call_groq(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=900
        )

        if groq_result.get("success"):
            report = groq_result["text"]
            source = "groq"
            llm_available = True
            llm_error = None
            summary = "Groq screener explanation generated successfully."
        else:
            top_buy = compact_context["top_buy_candidates"] or []
            high_risk = compact_context["highest_risk_candidates"] or []

            buy_names = [
                str(item.get("symbol"))
                for item in top_buy[:5]
                if isinstance(item, dict) and item.get("symbol")
            ]

            risk_names = [
                str(item.get("symbol"))
                for item in high_risk[:5]
                if isinstance(item, dict) and item.get("symbol")
            ]

            report = "\n".join(
                [
                    f"**Direct Answer:** The strongest candidates for further research are {', '.join(buy_names) if buy_names else 'not available'}.",
                    "",
                    f"**Higher-Risk / Caution Candidates:** The stocks requiring more caution are {', '.join(risk_names) if risk_names else 'not available'}.",
                    "",
                    "**Evidence from Agents:** The screener ranked the selected watchlist using buy score, risk score, returns, RSI, volatility, and moving-average signals.",
                    "",
                    "**Risk Warning:** This is a watchlist-based screener, not a full-market scan. These outputs are only candidates for further research.",
                    "",
                    "**Not Financial Advice Disclaimer:** This is for paper decision support and educational research only. It is not personalized financial advice."
                ]
            )

            source = "local_fallback"
            llm_available = False
            llm_error = groq_result.get("llm_error")
            summary = "Local fallback screener explanation generated successfully."

        return {
            "success": True,
            "agent": "Groq Report Agent",
            "agent_goal": "Explain the S&P-style screener result in natural language.",
            "report_type": "screener_explanation",
            "source": source,
            "provider": "groq",
            "model": self.model,
            "llm_available": llm_available,
            "llm_error": llm_error,
            "llm_debug": groq_result.get("llm_debug"),
            "plain_language_report": report,
            "summary": summary
        }

    def generate_screener_explanation(
        self,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        return self.generate_screener_report(*args, **kwargs)

    def explain_screener_result(
        self,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        return self.generate_screener_report(*args, **kwargs)

    def run(
        self,
        *args,
        **kwargs
    ) -> Dict[str, Any]:
        return self.generate_single_stock_report(*args, **kwargs)