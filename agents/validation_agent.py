from datetime import datetime, timezone
from typing import Dict, Any, Optional


class ValidationAgent:
    """
    Validation Agent:
    Validates multi-source market data from Finnhub and Alpha Vantage.

    Main goals:
    - Check whether primary and secondary sources returned valid data.
    - Compare prices between sources.
    - Detect source date mismatch or stale secondary data.
    - Decide whether downstream agents can continue.
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.01,
        medium_confidence_threshold: float = 0.03
    ):
        self.high_confidence_threshold = high_confidence_threshold
        self.medium_confidence_threshold = medium_confidence_threshold

    # --------------------------------------------------
    # Public main method expected by app.py
    # --------------------------------------------------
    def validate_market_data(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main validation method.
        app.py can safely call this method.
        """

        reasoning_steps = []
        issues = []
        warnings = []

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

        if primary_valid:
            reasoning_steps.append("Checked whether the primary source Finnhub returned valid data.")
        else:
            issues.append("Primary source Finnhub did not return valid data.")

        if secondary_valid:
            reasoning_steps.append("Checked whether the secondary source Alpha Vantage returned valid data.")
        else:
            warnings.append("Secondary source Alpha Vantage did not return valid data.")

        primary_price = self._extract_price(primary)
        secondary_price = self._extract_price(secondary)

        selected_price = None
        selected_source = None
        price_difference = None

        primary_date = self._extract_date(primary)
        secondary_date = self._extract_date(secondary)
        source_date_difference_days = None
        secondary_source_stale = False

        if primary_date and secondary_date:
            try:
                source_date_difference_days = abs(
                    (datetime.fromisoformat(primary_date).date()
                     - datetime.fromisoformat(secondary_date).date()).days
                )
                reasoning_steps.append(
                    f"Checked source dates: Finnhub={primary_date}, Alpha Vantage={secondary_date}."
                )

                if source_date_difference_days >= 2:
                    secondary_source_stale = True
                    warnings.append(
                        f"Secondary source may be stale: source date difference is "
                        f"{source_date_difference_days} days."
                    )
            except Exception:
                source_date_difference_days = None

        # -----------------------------
        # Case 1: primary and secondary are both valid
        # -----------------------------
        if primary_valid and secondary_valid and primary_price and secondary_price:
            selected_price = primary_price
            selected_source = "Finnhub"

            price_difference = abs(primary_price - secondary_price) / max(primary_price, 1e-9)

            reasoning_steps.append(
                f"Compared Finnhub price {primary_price} with Alpha Vantage price {secondary_price}."
            )
            reasoning_steps.append(
                f"Calculated relative price difference: {price_difference * 100:.2f}%."
            )

            if price_difference <= self.high_confidence_threshold:
                is_valid = True
                confidence = "High"
                confidence_score = 0.95
                next_action = "ALLOW_ANALYSIS"
                agent_decision = "The data is reliable enough to continue to the Analyst Agent."
                summary = "Multi-source validation passed. Finnhub and Alpha Vantage prices are consistent."

            elif price_difference <= self.medium_confidence_threshold:
                is_valid = True
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
                summary = "Primary source is valid, but multi-source price difference was detected."

            else:
                is_valid = True
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

        # -----------------------------
        # Case 2: only primary source is valid
        # -----------------------------
        elif primary_valid and primary_price:
            selected_price = primary_price
            selected_source = "Finnhub"
            is_valid = True
            confidence = "Medium"
            confidence_score = 0.70
            price_difference = None
            next_action = "ALLOW_ANALYSIS_WITH_CAUTION"
            warnings.append("Only the primary source Finnhub returned usable price data.")
            agent_decision = (
                "The primary source is valid, but secondary confirmation is unavailable. "
                "Downstream agents should continue with caution."
            )
            summary = "Primary source validation passed, but secondary source validation was unavailable."

        # -----------------------------
        # Case 3: only secondary source is valid
        # -----------------------------
        elif secondary_valid and secondary_price:
            selected_price = secondary_price
            selected_source = "Alpha Vantage"
            is_valid = True
            confidence = "Low"
            confidence_score = 0.50
            price_difference = None
            next_action = "ALLOW_ANALYSIS_WITH_HIGH_CAUTION"
            warnings.append("Only the secondary source Alpha Vantage returned usable price data.")
            agent_decision = (
                "The secondary source is usable, but primary confirmation is unavailable. "
                "Downstream agents should continue with high caution."
            )
            summary = "Secondary source validation passed, but primary source validation was unavailable."

        # -----------------------------
        # Case 4: no valid source
        # -----------------------------
        else:
            is_valid = False
            confidence = "Low"
            confidence_score = 0.0
            selected_price = None
            selected_source = None
            price_difference = None
            next_action = "BLOCK_ANALYSIS"
            issues.append("No valid price data was available from either source.")
            agent_decision = "The data is not reliable enough to continue."
            summary = "Validation failed. No reliable market price was available."

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
            "agent_goal": "Validate whether multi-source market data is reliable enough for trading analysis.",
            "agent_decision": agent_decision,
            "reasoning_steps": reasoning_steps,
            "secondary_source_stale": secondary_source_stale,
            "source_date_difference_days": source_date_difference_days,
            "source_dates": {
                "primary_date": primary_date,
                "secondary_date": secondary_date
            },
            "validation_for_next_agent": {
                "symbol": self._extract_symbol(primary, secondary),
                "selected_price": selected_price,
                "selected_source": selected_source,
                "confidence": confidence,
                "confidence_score": confidence_score,
                "next_action": next_action,
                "secondary_source_stale": secondary_source_stale
            },
            "summary": summary
        }

    # --------------------------------------------------
    # Alias methods so app.py will not crash
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

    # Extra aliases, useful if older app.py versions call these
    def validate_data(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_sources(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_market_quotes(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    # --------------------------------------------------
    # Internal helpers
    # --------------------------------------------------
    def _get_source(self, multi_quote: Dict[str, Any], possible_keys) -> Dict[str, Any]:
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

    def _extract_symbol(self, primary: Dict[str, Any], secondary: Dict[str, Any]) -> str:
        for source in [primary, secondary]:
            if not isinstance(source, dict):
                continue

            symbol = source.get("symbol")
            if symbol:
                return str(symbol).upper()

            raw_data = source.get("raw_data", {})
            if isinstance(raw_data, dict):
                global_quote = raw_data.get("Global Quote", {})
                if isinstance(global_quote, dict):
                    symbol = global_quote.get("01. symbol")
                    if symbol:
                        return str(symbol).upper()

        return "UNKNOWN"

    def _extract_price(self, source: Dict[str, Any]) -> Optional[float]:
        if not isinstance(source, dict):
            return None

        possible_keys = [
            "current_price",
            "price",
            "close",
            "c"
        ]

        for key in possible_keys:
            value = source.get(key)
            price = self._to_float(value)
            if price is not None and price > 0:
                return price

        raw_data = source.get("raw_data", {})

        if isinstance(raw_data, dict):
            # Finnhub style raw_data
            for key in ["c", "current_price", "price"]:
                price = self._to_float(raw_data.get(key))
                if price is not None and price > 0:
                    return price

            # Alpha Vantage style raw_data
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

        # Alpha Vantage often gives date string
        if isinstance(timestamp, str):
            parsed = self._parse_date_string(timestamp)
            if parsed:
                return parsed

        # Finnhub often gives epoch seconds
        if isinstance(timestamp, (int, float)):
            try:
                return datetime.fromtimestamp(
                    float(timestamp),
                    tz=timezone.utc
                ).date().isoformat()
            except Exception:
                pass

        raw_data = source.get("raw_data", {})

        if isinstance(raw_data, dict):
            # Finnhub raw timestamp
            raw_timestamp = raw_data.get("t")
            if isinstance(raw_timestamp, (int, float)):
                try:
                    return datetime.fromtimestamp(
                        float(raw_timestamp),
                        tz=timezone.utc
                    ).date().isoformat()
                except Exception:
                    pass

            # Alpha Vantage latest trading day
            global_quote = raw_data.get("Global Quote", {})
            if isinstance(global_quote, dict):
                latest_day = global_quote.get("07. latest trading day")
                parsed = self._parse_date_string(latest_day)
                if parsed:
                    return parsed

        return None

    def _parse_date_string(self, value) -> Optional[str]:
        if not value:
            return None

        try:
            return datetime.fromisoformat(str(value)).date().isoformat()
        except Exception:
            pass

        try:
            return datetime.strptime(str(value), "%Y-%m-%d").date().isoformat()
        except Exception:
            pass

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