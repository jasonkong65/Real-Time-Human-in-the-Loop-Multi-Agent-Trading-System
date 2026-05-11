from datetime import datetime


class ValidationAgent:
    """
    Validation Agent:
    Check the validity of stock symbols, data integrity and multi-source consistency.
    """

    def validate_quote(self, quote: dict) -> dict:
        issues = []
        warnings = []

        if not quote.get("success"):
            issues.append(quote.get("error", "Data Agent failed to fetch data."))
            return {
                "is_valid": False,
                "issues": issues,
                "warnings": warnings,
                "confidence": "Low",
                "readable_time": None,
                "summary": "Data validation failed due to API response issues."
            }

        current_price = quote.get("current_price")
        previous_close_price = quote.get("previous_close_price")
        high_price = quote.get("high_price")
        low_price = quote.get("low_price")
        open_price = quote.get("open_price")
        timestamp = quote.get("timestamp")

        if current_price is None:
            issues.append("Missing current price.")

        if current_price is not None and current_price <= 0:
            issues.append(f"Invalid current price: {current_price}")

        for field, value in {
            "previous_close_price": previous_close_price,
            "high_price": high_price,
            "low_price": low_price,
            "open_price": open_price
        }.items():
            if value is not None and value <= 0:
                issues.append(f"Invalid non-positive value: {field} = {value}")

        if high_price is not None and low_price is not None:
            if high_price < low_price:
                issues.append(f"High price {high_price} is lower than low price {low_price}.")

        readable_time = None
        if timestamp is not None:
            try:
                readable_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                warnings.append(f"Timestamp {timestamp} cannot be converted.")

        if current_price is not None and previous_close_price is not None and previous_close_price > 0:
            daily_change = (current_price - previous_close_price) / previous_close_price

            if abs(daily_change) > 0.20:
                warnings.append(f"Large daily price change detected: {daily_change:.2%}.")
            elif abs(daily_change) > 0.05:
                warnings.append(f"Moderate daily price change detected: {daily_change:.2%}.")

        is_valid = len(issues) == 0

        if is_valid and not warnings:
            confidence = "High"
            summary = "Data validation passed. The quote data is complete and consistent."
        elif is_valid and warnings:
            confidence = "Medium"
            summary = "Data validation passed with warnings."
        else:
            confidence = "Low"
            summary = "Data validation failed."

        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "confidence": confidence,
            "readable_time": readable_time,
            "summary": summary
        }

    def validate_multi_source_quote(self, multi_quote: dict, price_diff_threshold: float = 0.01) -> dict:
        """
        Compare Finnhub and iTick prices.

        price_diff_threshold = 0.01 means 1%.
        If the two sources differ by more than 1%, confidence is reduced.
        """
        issues = []
        warnings = []

        finnhub = multi_quote.get("finnhub", {})
        itick = multi_quote.get("itick", {})

        finnhub_valid = finnhub.get("success", False)
        itick_valid = itick.get("success", False)

        if not finnhub_valid:
            issues.append(f"Finnhub failed: {finnhub.get('error', 'Unknown error')}")

        if not itick_valid:
            warnings.append(f"iTick unavailable: {itick.get('error', 'Unknown error')}")

        if not finnhub_valid:
            return {
                "is_valid": False,
                "issues": issues,
                "warnings": warnings,
                "confidence": "Low",
                "price_difference": None,
                "summary": "Multi-source validation failed because the primary source is unavailable."
            }

        finnhub_price = finnhub.get("current_price")
        itick_price = itick.get("current_price")

        if finnhub_price is None or finnhub_price <= 0:
            issues.append("Finnhub current price is missing or invalid.")

        if itick_valid and (itick_price is None or itick_price <= 0):
            warnings.append("iTick current price is missing or invalid.")
            itick_valid = False

        price_difference = None

        if finnhub_price and itick_valid and itick_price:
            price_difference = abs(finnhub_price - itick_price) / finnhub_price

            if price_difference > price_diff_threshold:
                warnings.append(
                    f"Multi-source price mismatch detected: "
                    f"Finnhub={finnhub_price}, iTick={itick_price}, "
                    f"difference={price_difference:.2%}."
                )

        if issues:
            confidence = "Low"
            is_valid = False
            summary = "Multi-source validation failed because critical data issues were detected."
        elif itick_valid and price_difference is not None and price_difference <= price_diff_threshold:
            confidence = "High"
            is_valid = True
            summary = "Multi-source validation passed. Finnhub and iTick prices are consistent."
        elif itick_valid and price_difference is not None and price_difference > price_diff_threshold:
            confidence = "Medium"
            is_valid = True
            summary = "Primary source is valid, but multi-source price difference was detected."
        else:
            confidence = "Medium"
            is_valid = True
            summary = "Primary source is valid, but secondary source is unavailable. Confidence reduced."

        return {
            "is_valid": is_valid,
            "issues": issues,
            "warnings": warnings,
            "confidence": confidence,
            "price_difference": price_difference,
            "summary": summary
        }