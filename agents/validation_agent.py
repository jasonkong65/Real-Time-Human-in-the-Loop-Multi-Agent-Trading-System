from datetime import datetime, timezone
from typing import Any, Dict, Optional


class ValidationAgent:
    """
    Validation Agent

    Checks whether the quote data is good enough to use. It compares sources,
    checks staleness where dates exist, and passes a confidence score to later
    agents. The thresholds are gently adjusted when the market looks more
    volatile, so the agent is less brittle than a fixed 1% / 3% rule.
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.01,
        medium_confidence_threshold: float = 0.03,
        stale_date_threshold_days: int = 3,
    ):
        self.high_confidence_threshold = high_confidence_threshold
        self.medium_confidence_threshold = medium_confidence_threshold
        self.stale_date_threshold_days = stale_date_threshold_days

    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _get_source(self, data: Dict[str, Any], names) -> Dict[str, Any]:
        for name in names:
            value = data.get(name)
            if isinstance(value, dict):
                return value
        return {}

    def _valid_source(self, source: Dict[str, Any]) -> bool:
        if not isinstance(source, dict):
            return False
        price = self._extract_price(source)
        return bool(source.get("success")) and price is not None and price > 0

    def _extract_price(self, source: Dict[str, Any]) -> Optional[float]:
        for key in ["current_price", "price", "c", "close", "latest_price"]:
            value = self._safe_float(source.get(key))
            if value is not None and value > 0:
                return value
        raw = source.get("raw_response")
        if isinstance(raw, dict):
            for key in ["c", "05. price"]:
                value = self._safe_float(raw.get(key))
                if value is not None and value > 0:
                    return value
        return None

    def _extract_date(self, source: Dict[str, Any]) -> Optional[str]:
        for key in ["latest_trading_day", "date", "timestamp", "t"]:
            if source.get(key) is not None:
                return str(source.get(key))
        return None

    def _parse_date(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None
        try:
            if isinstance(value, (int, float)) or str(value).isdigit():
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except Exception:
            try:
                return datetime.strptime(str(value)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except Exception:
                return None

    def _date_gap_days(self, left: Optional[str], right: Optional[str]) -> Optional[int]:
        d1 = self._parse_date(left)
        d2 = self._parse_date(right)
        if not d1 or not d2:
            return None
        return abs((d1.date() - d2.date()).days)

    def _dynamic_thresholds(self, primary: Dict[str, Any]) -> tuple:
        high = self.high_confidence_threshold
        medium = self.medium_confidence_threshold
        high_price = self._safe_float(primary.get("high_price"))
        low_price = self._safe_float(primary.get("low_price"))
        price = self._extract_price(primary)
        if price and high_price and low_price and high_price > low_price:
            intraday_range = (high_price - low_price) / price
            if intraday_range > 0.04:
                high *= 1.5
                medium *= 1.5
            elif intraday_range > 0.025:
                high *= 1.25
                medium *= 1.25
        return high, medium

    def _build_result(
        self,
        symbol: str,
        confidence: str,
        confidence_score: float,
        next_action: str,
        selected_price: Optional[float],
        selected_source: Optional[str],
        price_difference_pct: Optional[float],
        source_date_difference_days: Optional[int],
        issues: list,
        warnings: list,
        reasoning_steps: list,
    ) -> Dict[str, Any]:
        if next_action == "BLOCK_ANALYSIS":
            decision = "The data is not reliable enough for analysis."
        elif confidence == "High":
            decision = "The data looks reliable enough to continue."
        elif confidence == "Medium":
            decision = "The data can be used, but later agents should stay cautious."
        else:
            decision = "The data is usable only as a low-confidence reference."

        return {
            "success": next_action != "BLOCK_ANALYSIS",
            "agent": "Validation Agent",
            "agent_goal": "Check market data quality before analysis.",
            "symbol": symbol,
            "confidence": confidence,
            "confidence_score": round(float(confidence_score), 4),
            "next_action": next_action,
            "selected_price": selected_price,
            "selected_source": selected_source,
            "price_difference_pct": None if price_difference_pct is None else round(float(price_difference_pct), 6),
            "source_date_difference_days": source_date_difference_days,
            "issues": issues,
            "warnings": warnings,
            "agent_decision": decision,
            "reasoning_steps": reasoning_steps,
            "validation_for_next_agent": {
                "symbol": symbol,
                "confidence": confidence,
                "confidence_score": round(float(confidence_score), 4),
                "next_action": next_action,
                "selected_price": selected_price,
            },
            "summary": f"Validation completed for {symbol}: {confidence} confidence.",
        }

    def validate_market_data(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        reasoning_steps, issues, warnings = [], [], []
        if not isinstance(multi_quote, dict):
            return self._build_result(
                "UNKNOWN", "Low", 0.0, "BLOCK_ANALYSIS", None, None, None, None,
                ["Input was not a dictionary."], [], []
            )

        symbol = str(multi_quote.get("symbol", "UNKNOWN")).upper().strip()
        primary = self._get_source(multi_quote, ["finnhub", "primary", "finnhub_quote"])
        secondary = self._get_source(multi_quote, ["alpha_vantage", "secondary", "alpha_vantage_quote"])

        primary_valid = self._valid_source(primary)
        secondary_valid = self._valid_source(secondary)
        primary_price = self._extract_price(primary)
        secondary_price = self._extract_price(secondary)

        if primary_valid:
            selected_price = primary_price
            selected_source = primary.get("source", "Finnhub")
        elif secondary_valid:
            selected_price = secondary_price
            selected_source = secondary.get("source", "Alpha Vantage")
        else:
            issues.append("No valid price was found from either source.")
            return self._build_result(
                symbol, "Low", 0.0, "BLOCK_ANALYSIS", None, None, None, None,
                issues, warnings, ["Both data sources failed or returned unusable prices."],
            )

        if primary_valid:
            reasoning_steps.append(f"Primary source price found: {primary_price}.")
        if secondary_valid:
            reasoning_steps.append(f"Secondary source price found: {secondary_price}.")

        price_difference_pct = None
        date_gap = self._date_gap_days(self._extract_date(primary), self._extract_date(secondary))
        if date_gap is not None and date_gap > self.stale_date_threshold_days:
            warnings.append(f"Source dates differ by {date_gap} days.")

        if primary_valid and secondary_valid and primary_price:
            price_difference_pct = abs(primary_price - secondary_price) / primary_price
            high_th, med_th = self._dynamic_thresholds(primary)
            reasoning_steps.append(f"Price difference between sources is {price_difference_pct:.2%}.")
            if price_difference_pct <= high_th and not warnings:
                confidence, score, action = "High", 1.0, "ALLOW_ANALYSIS"
            elif price_difference_pct <= med_th:
                confidence, score, action = "Medium", 0.75, "ALLOW_ANALYSIS_WITH_CAUTION"
            else:
                confidence, score, action = "Low", 0.45, "ALLOW_ANALYSIS_WITH_LOW_CONFIDENCE"
                warnings.append("The two sources are not closely aligned.")
        elif primary_valid:
            confidence, score, action = "Medium", 0.70, "ALLOW_ANALYSIS_WITH_CAUTION"
            warnings.append("Only the primary source was available.")
        else:
            confidence, score, action = "Low", 0.45, "ALLOW_ANALYSIS_WITH_LOW_CONFIDENCE"
            warnings.append("Only the secondary source was available.")

        if date_gap is not None and date_gap > self.stale_date_threshold_days:
            score = min(score, 0.65)
            if confidence == "High":
                confidence = "Medium"
                action = "ALLOW_ANALYSIS_WITH_CAUTION"

        return self._build_result(
            symbol, confidence, score, action, selected_price, selected_source,
            price_difference_pct, date_gap, issues, warnings, reasoning_steps,
        )

    # Backward-compatible aliases
    def validate_multi_source_data(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_multi_source(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_quotes(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def run(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_data(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_sources(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_market_quotes(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_multi_source_quote(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_quote(self, quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data({"symbol": quote.get("symbol", "UNKNOWN"), "finnhub": quote})
