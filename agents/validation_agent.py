from datetime import datetime, timezone
from typing import Dict, Any, Optional


class ValidationAgent:
    """
    Validation Agent:
    Validates multi-source market data from Finnhub and Alpha Vantage.

    Goals:
    1. Check whether primary and secondary market data are valid.
    2. Compare prices from two sources.
    3. Detect stale or mismatched source dates.
    4. Decide whether downstream agents should continue, continue with caution, or block.
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.01,
        medium_confidence_threshold: float = 0.03,
        stale_date_threshold_days: int = 2
    ):
        self.high_confidence_threshold = high_confidence_threshold
        self.medium_confidence_threshold = medium_confidence_threshold
        self.stale_date_threshold_days = stale_date_threshold_days

    # --------------------------------------------------
    # Main method used by app.py
    # --------------------------------------------------
    def validate_market_data(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main validation method.

        Expected input format from DataAgent:

        {
            "symbol": "AAPL",
            "finnhub": {...},
            "alpha_vantage": {...}
        }
        """

        reasoning_steps = []
        issues = []
        warnings = []

        if not isinstance(multi_quote, dict):
            return self._failed_result(
                symbol="UNKNOWN",
                issues=["Input multi_quote is not a dictionary."],
                warnings=[],
                reasoning_steps=[]
            )

        symbol = str(multi_quote.get("symbol", "UNKNOWN")).upper().strip()

        primary = self._get_source(
            multi_quote,
            ["finnhub", "finnhub_quote", "primary", "primary_quote"]
        )

        secondary = self._get_source(
            multi_quote,
            ["alpha_vantage", "alpha_vantage_quote", "secondary", "secondary_quote"]
        )

        primary_valid = self._is_valid_source(primary)
        secondary_valid = self._is_valid_source(secondary)

        primary_price = self._extract_price(primary)
        secondary_price = self._extract_price(secondary)

        primary_date = self._extract_date(primary)
        secondary_date = self._extract_date(secondary)

        source_date_difference_days = self._date_difference_days(
            primary_date,
            secondary_date
        )

        secondary_source_stale = False

        if primary_valid:
            reasoning_steps.append(
                "Checked whether the primary source Finnhub returned valid data."
            )
        else:
            issues.append("Primary source Finnhub did not return valid usable price data.")

        if secondary_valid:
            reasoning_steps.append(
                "Checked whether the secondary source Alpha Vantage returned valid data."
            )
        else:
            warnings.append(
                "Secondary source Alpha Vantage did not return valid usable price data."
            )

        if primary_date or secondary_date:
            reasoning_steps.append(
                f"Checked source dates: Finnhub={primary_date}, Alpha Vantage={secondary_date}."
            )

        if (
            source_date_difference_days is not None
            and source_date_difference_days >= self.stale_date_threshold_days
        ):
            secondary_source_stale = True
            warnings.append(
                f"Source date mismatch detected: difference={source_date_difference_days} days."
            )

        selected_price = None
        selected_source = None
        price_difference = None

        # --------------------------------------------------
        # Case 1: both sources valid
        # --------------------------------------------------
        if primary_valid and secondary_valid:
            selected_price = primary_price
            selected_source = "Finnhub"

            price_difference = abs(primary_price - secondary_price) / max(
                primary_price,
                1e-9
            )

            reasoning_steps.append(
                f"Compared Finnhub price {primary_price} with Alpha Vantage price {secondary_price}."
            )
            reasoning_steps.append(
                f"Calculated relative price difference: {price_difference * 100:.2f}%."
            )

            if price_difference <= self.high_confidence_threshold:
                confidence = "High"
                confidence_score = 0.95
                next_action = "ALLOW_ANALYSIS"
                agent_decision = (
                    "The data is reliable enough to continue to the Analyst Agent."
                )
                summary = (
                    "Multi-source validation passed. Finnhub and Alpha Vantage prices are consistent."
                )

            elif price_difference <= self.medium_confidence_threshold:
                confidence = "Medium"
                confidence_score = 0.65
                next_action = "ALLOW_ANALYSIS_WITH_CAUTION"
                warnings.append(
                    f"Multi-source price mismatch detected: Finnhub={primary_price}, "
                    f"Alpha Vantage={secondary_price}, difference={price_difference * 100:.2f}%."
                )
                agent_decision = (
                    "The primary data is usable, but downstream agents should treat it with caution "
                    "because the two sources differ."
                )
                summary = (
                    "Primary source is valid, but multi-source price difference was detected."
                )

            else:
                confidence = "Low"
                confidence_score = 0.45
                next_action = "ALLOW_ANALYSIS_WITH_HIGH_CAUTION"
                warnings.append(
                    f"Large multi-source price mismatch detected: Finnhub={primary_price}, "
                    f"Alpha Vantage={secondary_price}, difference={price_difference * 100:.2f}%."
                )
                agent_decision = (
                    "The primary source is usable, but the price mismatch is large. "
                    "Downstream agents should treat the result with high caution."
                )
                summary = "Large price mismatch detected between market data sources."

            return self._build_result(
                symbol=symbol,
                is_valid=True,
                issues=issues,
                warnings=warnings,
                confidence=confidence,
                confidence_score=confidence_score,
                price_difference=price_difference,
                selected_price=selected_price,
                selected_source=selected_source,
                next_action=next_action,
                agent_decision=agent_decision,
                reasoning_steps=reasoning_steps,
                secondary_source_stale=secondary_source_stale,
                source_date_difference_days=source_date_difference_days,
                primary_date=primary_date,
                secondary_date=secondary_date,
                summary=summary
            )

        # --------------------------------------------------
        # Case 2: only Finnhub valid
        # --------------------------------------------------
        if primary_valid:
            selected_price = primary_price
            selected_source = "Finnhub"

            warnings.append(
                "Only the primary source Finnhub returned usable price data."
            )

            return self._build_result(
                symbol=symbol,
                is_valid=True,
                issues=issues,
                warnings=warnings,
                confidence="Medium",
                confidence_score=0.70,
                price_difference=None,
                selected_price=selected_price,
                selected_source=selected_source,
                next_action="ALLOW_ANALYSIS_WITH_CAUTION",
                agent_decision=(
                    "The primary source is valid, but secondary confirmation is unavailable. "
                    "Downstream agents should continue with caution."
                ),
                reasoning_steps=reasoning_steps,
                secondary_source_stale=secondary_source_stale,
                source_date_difference_days=source_date_difference_days,
                primary_date=primary_date,
                secondary_date=secondary_date,
                summary=(
                    "Primary source validation passed, but secondary source validation was unavailable."
                )
            )

        # --------------------------------------------------
        # Case 3: only Alpha Vantage valid
        # --------------------------------------------------
        if secondary_valid:
            selected_price = secondary_price
            selected_source = "Alpha Vantage"

            warnings.append(
                "Only the secondary source Alpha Vantage returned usable price data."
            )

            return self._build_result(
                symbol=symbol,
                is_valid=True,
                issues=issues,
                warnings=warnings,
                confidence="Low",
                confidence_score=0.50,
                price_difference=None,
                selected_price=selected_price,
                selected_source=selected_source,
                next_action="ALLOW_ANALYSIS_WITH_HIGH_CAUTION",
                agent_decision=(
                    "The secondary source is usable, but primary confirmation is unavailable. "
                    "Downstream agents should continue with high caution."
                ),
                reasoning_steps=reasoning_steps,
                secondary_source_stale=secondary_source_stale,
                source_date_difference_days=source_date_difference_days,
                primary_date=primary_date,
                secondary_date=secondary_date,
                summary=(
                    "Secondary source validation passed, but primary source validation was unavailable."
                )
            )

        # --------------------------------------------------
        # Case 4: no valid source
        # --------------------------------------------------
        issues.append("No valid price data was available from either source.")

        return self._failed_result(
            symbol=symbol,
            issues=issues,
            warnings=warnings,
            reasoning_steps=reasoning_steps,
            primary_date=primary_date,
            secondary_date=secondary_date,
            source_date_difference_days=source_date_difference_days
        )

    # --------------------------------------------------
    # Compatibility aliases for different app.py versions
    # --------------------------------------------------
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

    # Old method name in your uploaded validation_agent.py
    def validate_multi_source_quote(
        self,
        multi_quote: Dict[str, Any],
        price_diff_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        if price_diff_threshold is not None:
            self.medium_confidence_threshold = price_diff_threshold
        return self.validate_market_data(multi_quote)

    # Old single-source validator support
    def validate_quote(self, quote: Dict[str, Any]) -> Dict[str, Any]:
        price = self._extract_price(quote)

        if not isinstance(quote, dict) or quote.get("success") is False:
            return {
                "is_valid": False,
                "confidence": "Low",
                "confidence_score": 0.0,
                "summary": "Quote validation failed.",
                "issues": ["Quote source returned invalid data."],
                "warnings": [],
                "selected_price": None
            }

        if price is None or price <= 0:
            return {
                "is_valid": False,
                "confidence": "Low",
                "confidence_score": 0.0,
                "summary": "Quote validation failed because price is missing or invalid.",
                "issues": ["Price is missing or invalid."],
                "warnings": [],
                "selected_price": None
            }

        return {
            "is_valid": True,
            "confidence": "High",
            "confidence_score": 0.9,
            "summary": "Single-source quote validation passed.",
            "issues": [],
            "warnings": [],
            "selected_price": price
        }

    # --------------------------------------------------
    # Result builders
    # --------------------------------------------------
    def _build_result(
        self,
        symbol: str,
        is_valid: bool,
        issues: list,
        warnings: list,
        confidence: str,
        confidence_score: float,
        price_difference: Optional[float],
        selected_price: Optional[float],
        selected_source: Optional[str],
        next_action: str,
        agent_decision: str,
        reasoning_steps: list,
        secondary_source_stale: bool,
        source_date_difference_days: Optional[int],
        primary_date: Optional[str],
        secondary_date: Optional[str],
        summary: str
    ) -> Dict[str, Any]:

        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "price_difference": price_difference,
            "selected_price": selected_price,
            "selected_source": selected_source,
            "next_action": next_action,
            "agent_goal": (
                "Validate whether multi-source market data is reliable enough "
                "for trading analysis."
            ),
            "agent_decision": agent_decision,
            "reasoning_steps": reasoning_steps,
            "secondary_source_stale": secondary_source_stale,
            "source_date_difference_days": source_date_difference_days,
            "source_dates": {
                "primary_date": primary_date,
                "secondary_date": secondary_date
            },
            "validation_for_next_agent": {
                "symbol": symbol,
                "selected_price": selected_price,
                "selected_source": selected_source,
                "confidence": confidence,
                "confidence_score": confidence_score,
                "next_action": next_action,
                "secondary_source_stale": secondary_source_stale
            },
            "summary": summary
        }

    def _failed_result(
        self,
        symbol: str,
        issues: list,
        warnings: list,
        reasoning_steps: list,
        primary_date: Optional[str] = None,
        secondary_date: Optional[str] = None,
        source_date_difference_days: Optional[int] = None
    ) -> Dict[str, Any]:

        return self._build_result(
            symbol=symbol,
            is_valid=False,
            issues=issues,
            warnings=warnings,
            confidence="Low",
            confidence_score=0.0,
            price_difference=None,
            selected_price=None,
            selected_source=None,
            next_action="BLOCK_ANALYSIS",
            agent_decision="The data is not reliable enough to continue.",
            reasoning_steps=reasoning_steps,
            secondary_source_stale=False,
            source_date_difference_days=source_date_difference_days,
            primary_date=primary_date,
            secondary_date=secondary_date,
            summary="Validation failed. No reliable market price was available."
        )

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _get_source(self, multi_quote: Dict[str, Any], possible_keys: list) -> Dict[str, Any]:
        if not isinstance(multi_quote, dict):
            return {}

        for key in possible_keys:
            value = multi_quote.get(key)
            if isinstance(value, dict):
                return value

        return {}

    def _is_valid_source(self, source: Dict[str, Any]) -> bool:
        if not isinstance(source, dict) or not source:
            return False

        if source.get("success") is False:
            return False

        price = self._extract_price(source)
        return price is not None and price > 0

    def _extract_price(self, source: Dict[str, Any]) -> Optional[float]:
        if not isinstance(source, dict):
            return None

        for key in ["current_price", "price", "close", "c"]:
            value = source.get(key)
            price = self._to_float(value)
            if price is not None and price > 0:
                return price

        raw_data = source.get("raw_data", {})

        if isinstance(raw_data, dict):
            for key in ["c", "current_price", "price", "close"]:
                price = self._to_float(raw_data.get(key))
                if price is not None and price > 0:
                    return price

            global_quote = raw_data.get("Global Quote", {})
            if isinstance(global_quote, dict):
                for key in ["05. price", "price", "close"]:
                    price = self._to_float(global_quote.get(key))
                    if price is not None and price > 0:
                        return price

        return None

    def _extract_date(self, source: Dict[str, Any]) -> Optional[str]:
        if not isinstance(source, dict):
            return None

        timestamp = source.get("timestamp")

        parsed = self._parse_date_value(timestamp)
        if parsed:
            return parsed

        raw_data = source.get("raw_data", {})

        if isinstance(raw_data, dict):
            parsed = self._parse_date_value(raw_data.get("t"))
            if parsed:
                return parsed

            global_quote = raw_data.get("Global Quote", {})
            if isinstance(global_quote, dict):
                parsed = self._parse_date_value(
                    global_quote.get("07. latest trading day")
                )
                if parsed:
                    return parsed

        return None

    def _parse_date_value(self, value) -> Optional[str]:
        if value is None:
            return None

        if isinstance(value, (int, float)):
            try:
                return datetime.fromtimestamp(
                    float(value),
                    tz=timezone.utc
                ).date().isoformat()
            except Exception:
                return None

        if isinstance(value, str):
            value = value.strip()

            try:
                return datetime.fromisoformat(value).date().isoformat()
            except Exception:
                pass

            try:
                return datetime.strptime(value, "%Y-%m-%d").date().isoformat()
            except Exception:
                pass

        return None

    def _date_difference_days(
        self,
        date_a: Optional[str],
        date_b: Optional[str]
    ) -> Optional[int]:
        if not date_a or not date_b:
            return None

        try:
            parsed_a = datetime.fromisoformat(date_a).date()
            parsed_b = datetime.fromisoformat(date_b).date()
            return abs((parsed_a - parsed_b).days)
        except Exception:
            return None

    def _to_float(self, value) -> Optional[float]:
        try:
            if value is None:
                return None

            if isinstance(value, str):
                value = value.replace(",", "").strip()

            return float(value)

        except Exception:
            return None