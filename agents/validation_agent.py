from datetime import datetime


class ValidationAgent:
    """
    Validation Agent:
    Evaluates data quality, source reliability, and multi-source consistency.

    This agent does not only check whether data exists.
    It also decides whether the system should continue, continue with caution,
    or block later analysis.
    """

    @staticmethod
    def _safe_float(value):
        """
        Safely convert a value to float.
        Returns None if conversion fails.
        """
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    def validate_quote(self, quote: dict) -> dict:
        """
        Validate one source of quote data.
        This is used for single-source checking.
        """
        issues = []
        warnings = []
        reasoning_steps = []

        source = quote.get("source", "Unknown source")
        reasoning_steps.append(f"Checking quote data from {source}.")

        if not quote.get("success"):
            issues.append(quote.get("error", f"{source} failed to fetch data."))

            return {
                "is_valid": False,
                "issues": issues,
                "warnings": warnings,
                "confidence": "Low",
                "confidence_score": 0.2,
                "readable_time": None,
                "next_action": "BLOCK_ANALYSIS",
                "agent_goal": "Validate whether market data is reliable enough for trading analysis.",
                "agent_decision": "The data source failed, so later analysis should be blocked.",
                "reasoning_steps": reasoning_steps,
                "summary": f"{source} validation failed due to API response issues."
            }

        current_price = self._safe_float(quote.get("current_price"))
        previous_close_price = self._safe_float(quote.get("previous_close_price"))
        high_price = self._safe_float(quote.get("high_price"))
        low_price = self._safe_float(quote.get("low_price"))
        open_price = self._safe_float(quote.get("open_price"))
        timestamp = quote.get("timestamp")

        reasoning_steps.append("Checked whether the current price is available and positive.")

        if current_price is None:
            issues.append("Missing current price.")
        elif current_price <= 0:
            issues.append(f"Invalid current price: {current_price}")

        optional_price_fields = {
            "previous_close_price": previous_close_price,
            "high_price": high_price,
            "low_price": low_price,
            "open_price": open_price
        }

        for field, value in optional_price_fields.items():
            if value is not None and value <= 0:
                issues.append(f"Invalid non-positive value: {field} = {value}")

        if high_price is not None and low_price is not None:
            reasoning_steps.append("Checked whether high price is greater than low price.")
            if high_price < low_price:
                issues.append(f"High price {high_price} is lower than low price {low_price}.")

        readable_time = None

        if timestamp is not None:
            try:
                if isinstance(timestamp, (int, float)):
                    readable_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                else:
                    readable_time = str(timestamp)
            except Exception:
                warnings.append(f"Timestamp {timestamp} cannot be converted.")

        if current_price is not None and previous_close_price is not None and previous_close_price > 0:
            daily_change = (current_price - previous_close_price) / previous_close_price
            reasoning_steps.append(f"Calculated daily change: {daily_change:.2%}.")

            if abs(daily_change) > 0.20:
                warnings.append(f"Large daily price change detected: {daily_change:.2%}.")
            elif abs(daily_change) > 0.05:
                warnings.append(f"Moderate daily price change detected: {daily_change:.2%}.")

        is_valid = len(issues) == 0

        if is_valid and not warnings:
            confidence = "High"
            confidence_score = 0.9
            next_action = "ALLOW_ANALYSIS"
            agent_decision = "The single-source data is reliable enough to continue analysis."
            summary = f"{source} validation passed. The quote data is complete and consistent."

        elif is_valid and warnings:
            confidence = "Medium"
            confidence_score = 0.65
            next_action = "ALLOW_ANALYSIS_WITH_CAUTION"
            agent_decision = "The single-source data can be used, but downstream agents should treat it with caution."
            summary = f"{source} validation passed with warnings."

        else:
            confidence = "Low"
            confidence_score = 0.2
            next_action = "BLOCK_ANALYSIS"
            agent_decision = "The single-source data is unreliable, so later analysis should be blocked."
            summary = f"{source} validation failed."

        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "confidence": confidence,
            "confidence_score": confidence_score,
            "readable_time": readable_time,
            "next_action": next_action,
            "agent_goal": "Validate whether market data is reliable enough for trading analysis.",
            "agent_decision": agent_decision,
            "reasoning_steps": reasoning_steps,
            "summary": summary
        }

    def validate_multi_source_quote(self, multi_quote: dict, price_diff_threshold: float = 0.01) -> dict:
        """
        Compare Finnhub and Alpha Vantage prices.

        price_diff_threshold = 0.01 means 1%.
        If the two sources differ by more than 1%, confidence is reduced.
        """
        issues = []
        warnings = []
        reasoning_steps = []

        agent_goal = "Validate whether multi-source market data is reliable enough for trading analysis."

        finnhub = multi_quote.get("finnhub", {})
        alpha_vantage = multi_quote.get("alpha_vantage", {})

        finnhub_valid = finnhub.get("success", False)
        alpha_vantage_valid = alpha_vantage.get("success", False)

        reasoning_steps.append("Checked whether the primary source Finnhub returned valid data.")
        reasoning_steps.append("Checked whether the secondary source Alpha Vantage returned valid data.")

        if not finnhub_valid:
            issues.append(f"Finnhub failed: {finnhub.get('error', 'Unknown error')}")

        if not alpha_vantage_valid:
            warnings.append(f"Alpha Vantage unavailable: {alpha_vantage.get('error', 'Unknown error')}")

        # If primary source fails, block later analysis.
        if not finnhub_valid:
            next_action = "BLOCK_ANALYSIS"
            agent_decision = "The primary data source failed, so later analysis should be blocked."

            return {
                "is_valid": False,
                "issues": issues,
                "warnings": warnings,
                "confidence": "Low",
                "confidence_score": 0.2,
                "price_difference": None,
                "selected_price": None,
                "selected_source": None,
                "next_action": next_action,
                "agent_goal": agent_goal,
                "agent_decision": agent_decision,
                "reasoning_steps": reasoning_steps,
                "validation_for_next_agent": {
                    "symbol": multi_quote.get("symbol"),
                    "selected_price": None,
                    "selected_source": None,
                    "confidence": "Low",
                    "confidence_score": 0.2,
                    "next_action": next_action
                },
                "summary": "Multi-source validation failed because the primary source is unavailable."
            }

        finnhub_price = self._safe_float(finnhub.get("current_price"))
        alpha_vantage_price = self._safe_float(alpha_vantage.get("current_price"))

        if finnhub_price is None or finnhub_price <= 0:
            issues.append("Finnhub current price is missing or invalid.")

        if alpha_vantage_valid:
            if alpha_vantage_price is None or alpha_vantage_price <= 0:
                warnings.append("Alpha Vantage current price is missing or invalid.")
                alpha_vantage_valid = False

        price_difference = None

        if finnhub_price and alpha_vantage_valid and alpha_vantage_price:
            price_difference = abs(finnhub_price - alpha_vantage_price) / finnhub_price

            reasoning_steps.append(
                f"Compared Finnhub price {finnhub_price} with Alpha Vantage price {alpha_vantage_price}."
            )
            reasoning_steps.append(f"Calculated relative price difference: {price_difference:.2%}.")

            if price_difference > price_diff_threshold:
                warnings.append(
                    f"Multi-source price mismatch detected: "
                    f"Finnhub={finnhub_price}, Alpha Vantage={alpha_vantage_price}, "
                    f"difference={price_difference:.2%}."
                )

        # Finnhub is the primary source, so it is selected for downstream agents.
        selected_price = finnhub_price
        selected_source = "Finnhub"

        if issues:
            confidence = "Low"
            confidence_score = 0.2
            is_valid = False
            next_action = "BLOCK_ANALYSIS"
            agent_decision = "Critical data issues were detected, so later analysis should be blocked."
            summary = "Multi-source validation failed because critical data issues were detected."

        elif alpha_vantage_valid and price_difference is not None and price_difference <= price_diff_threshold:
            confidence = "High"
            confidence_score = 0.95
            is_valid = True
            next_action = "ALLOW_ANALYSIS"
            agent_decision = "The data is reliable enough to continue to the Analyst Agent."
            summary = "Multi-source validation passed. Finnhub and Alpha Vantage prices are consistent."

        elif alpha_vantage_valid and price_difference is not None and price_difference > price_diff_threshold:
            confidence = "Medium"
            confidence_score = 0.65
            is_valid = True
            next_action = "ALLOW_ANALYSIS_WITH_CAUTION"
            agent_decision = "The primary data is usable, but downstream agents should treat it with caution because the two sources differ."
            summary = "Primary source is valid, but multi-source price difference was detected."

        else:
            confidence = "Medium"
            confidence_score = 0.6
            is_valid = True
            next_action = "ALLOW_ANALYSIS_WITH_CAUTION"
            agent_decision = "The primary data is usable, but the secondary source is unavailable, so downstream agents should be cautious."
            summary = "Primary source is valid, but secondary source is unavailable. Confidence reduced."

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
            "agent_goal": agent_goal,
            "agent_decision": agent_decision,
            "reasoning_steps": reasoning_steps,
            "validation_for_next_agent": {
                "symbol": multi_quote.get("symbol"),
                "selected_price": selected_price,
                "selected_source": selected_source,
                "confidence": confidence,
                "confidence_score": confidence_score,
                "next_action": next_action
            },
            "summary": summary
        }