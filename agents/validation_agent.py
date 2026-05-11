from datetime import datetime

class ValidationAgent:

    "Validation Agent: Check the validity of stock symbols and data integrity."

    def validate_quote(self, quote: dict) -> dict:

        "Validates the structure and content of a stock quote."

        issues = []
        Warnings = []

        if not quote.get("success"):
            issues.append(quote.get("error", "Data agent failed to fetch data."))
            return {
                "is_valid": False,
                "issues": issues,
                "warnings": Warnings,
                "confidence": "Low",
                "summary": "Data validation failed due to API response issues."
            }
        
        current_price = quote.get("current_price")
        previous_close_price = quote.get("previous_close_price")
        high_price = quote.get("high_price")
        low_price = quote.get("low_price")
        open_price = quote.get("open_price")
        timestamp = quote.get("timestamp")

        required_fields = {
            "current_price": current_price,
            "previous_close_price": previous_close_price,
            "high_price": high_price,
            "low_price": low_price,
            "open_price": open_price,
            "timestamp": timestamp
        }

        for field, value in required_fields.items():
            if value is None:
                issues.append(f"Missing value for field: {field}")
        
        price_fields = {
            "current_price": current_price,
            "previous_close_price": previous_close_price,
            "high_price": high_price,
            "low_price": low_price,
            "open_price": open_price
        }

        for field, value in price_fields.items():
            if value is not None and value <= 0:
                issues.append(f"Invalid non-positive price value for field: {field} = {value}")

            if high_price is not None and low_price is not None:
                if high_price < low_price:
                    issues.append(f"High price {high_price} is less than low price {low_price}.")

            readable_time = None

            if timestamp is not None:
                try:
                    readable_time = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    Warnings.append(f"Timestamp {timestamp} cannot be converted to readable time format.")    
            
            if current_price is not None and previous_close_price is not None and previous_close_price > 0:
                daily_change = (current_price - previous_close_price) / previous_close_price

                if abs(daily_change) > 0.2:
                    Warnings.append(f"Large daily price change detected: {daily_change:.2%} (current: {current_price}, previous close: {previous_close_price})")
                elif abs(daily_change) > 0.05:
                    Warnings.append(f"Moderate daily price change detected: {daily_change:.2%} (current: {current_price}, previous close: {previous_close_price})")

            is_valid = len(issues) == 0

            if is_valid and len(Warnings) == 0:
                confidence = "High"
                summary = "Data validation passed. The quote data is complete and consistent."
            
            elif is_valid and len(Warnings) > 0:
                confidence = "Medium"
                summary = "Data validation passed with warnings. The quote data is mostly complete and consistent, but some anomalies were detected."
            
            else:
                confidence = "Low"
                summary = "Data validation failed. The quote data has significant issues that may affect its reliability."
            
            return {
                "is_valid": is_valid,
                "issues": issues,
                "warnings": Warnings,
                "confidence": confidence,
                "summary": summary
            }