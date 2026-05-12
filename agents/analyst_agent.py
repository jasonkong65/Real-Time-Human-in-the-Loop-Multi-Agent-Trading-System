from utils.features import build_trading_features


class AnalystAgent:
    """
    Two-Stage Analyst Agent:
    Stage 1: Quote-level analysis using live quote data.
    Stage 2: Historical analysis using OHLCV historical data.
    Stage 3: Combine both stages into a final analyst decision.

    This makes the agent more robust:
    - If historical data is unavailable, it can still perform basic quote-level analysis.
    - If historical data is available, it produces richer quantitative features for the Training Agent.
    """

    @staticmethod
    def _safe_float(value):
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_pct(value):
        if value is None:
            return "N/A"
        return f"{value:.2%}"

    def analyse_quote_level(self, multi_quote: dict, validation_result: dict) -> dict:
        """
        Stage 1:
        Perform fast quote-level analysis using current price, previous close,
        open, high, and low from Finnhub.
        """
        agent_goal = "Perform fast quote-level analysis using validated live market data."

        if validation_result.get("next_action") == "BLOCK_ANALYSIS":
            return {
                "success": False,
                "stage": "Stage 1: Quote-level Analysis",
                "agent_goal": agent_goal,
                "agent_decision": "Quote-level analysis blocked because Validation Agent marked the data as unreliable.",
                "summary": "No quote-level analysis was performed.",
                "quote_features": None,
                "reasoning_steps": []
            }

        finnhub_quote = multi_quote.get("finnhub", {})
        symbol = multi_quote.get("symbol")

        selected_price = self._safe_float(validation_result.get("selected_price"))
        previous_close = self._safe_float(finnhub_quote.get("previous_close_price"))
        open_price = self._safe_float(finnhub_quote.get("open_price"))
        high_price = self._safe_float(finnhub_quote.get("high_price"))
        low_price = self._safe_float(finnhub_quote.get("low_price"))

        issues = []
        reasoning_steps = []

        if selected_price is None:
            issues.append("Selected price is missing.")
        if previous_close is None or previous_close <= 0:
            issues.append("Previous close price is missing or invalid.")

        if issues:
            return {
                "success": False,
                "stage": "Stage 1: Quote-level Analysis",
                "agent_goal": agent_goal,
                "issues": issues,
                "agent_decision": "Quote-level analysis cannot continue because key price fields are missing.",
                "summary": "Quote-level analysis failed.",
                "quote_features": None,
                "reasoning_steps": reasoning_steps
            }

        daily_return = (selected_price - previous_close) / previous_close
        reasoning_steps.append(
            f"Calculated daily return using selected price {selected_price} and previous close {previous_close}."
        )

        open_to_current_return = None
        if open_price is not None and open_price > 0:
            open_to_current_return = (selected_price - open_price) / open_price
            reasoning_steps.append(
                f"Calculated open-to-current return using open price {open_price}."
            )

        intraday_range_pct = None
        if high_price is not None and low_price is not None and previous_close > 0:
            intraday_range_pct = (high_price - low_price) / previous_close
            reasoning_steps.append(
                f"Calculated intraday range using high price {high_price} and low price {low_price}."
            )

        if daily_return >= 0.02:
            quote_trend = "Strong upward"
        elif daily_return >= 0.005:
            quote_trend = "Slight upward"
        elif daily_return <= -0.02:
            quote_trend = "Strong downward"
        elif daily_return <= -0.005:
            quote_trend = "Slight downward"
        else:
            quote_trend = "Neutral"

        if intraday_range_pct is None:
            quote_volatility_level = "Unknown"
        elif intraday_range_pct >= 0.05:
            quote_volatility_level = "High"
        elif intraday_range_pct >= 0.02:
            quote_volatility_level = "Medium"
        else:
            quote_volatility_level = "Low"

        quote_score = 0.5

        if quote_trend in ["Strong upward", "Slight upward"]:
            quote_score += 0.15
        elif quote_trend in ["Strong downward", "Slight downward"]:
            quote_score -= 0.15

        if quote_volatility_level == "High":
            quote_score -= 0.10
        elif quote_volatility_level == "Low":
            quote_score += 0.05

        confidence_score = validation_result.get("confidence_score", 0.5)
        quote_score = max(0, min(1, quote_score * confidence_score))

        if quote_score >= 0.70:
            quote_signal = "QUOTE_BULLISH"
            agent_decision = "The live quote shows positive short-term movement."
        elif quote_score <= 0.35:
            quote_signal = "QUOTE_BEARISH"
            agent_decision = "The live quote shows weak short-term movement."
        elif quote_volatility_level == "High":
            quote_signal = "QUOTE_HIGH_VOLATILITY"
            agent_decision = "The live quote shows elevated intraday volatility."
        else:
            quote_signal = "QUOTE_NEUTRAL"
            agent_decision = "The live quote does not show a strong directional signal."

        quote_features = {
            "daily_return": daily_return,
            "open_to_current_return": open_to_current_return,
            "intraday_range_pct": intraday_range_pct,
            "quote_score": quote_score,
            "quote_trend": quote_trend,
            "quote_volatility_level": quote_volatility_level,
            "quote_signal": quote_signal
        }

        summary = (
            f"{symbol} quote-level analysis: daily return is {self._format_pct(daily_return)}, "
            f"quote trend is {quote_trend}, intraday volatility is {quote_volatility_level}, "
            f"and quote score is {quote_score:.2f}."
        )

        return {
            "success": True,
            "stage": "Stage 1: Quote-level Analysis",
            "agent_goal": agent_goal,
            "symbol": symbol,
            "selected_price": selected_price,
            "daily_return": daily_return,
            "daily_return_pct": self._format_pct(daily_return),
            "open_to_current_return": open_to_current_return,
            "open_to_current_return_pct": self._format_pct(open_to_current_return),
            "intraday_range_pct": intraday_range_pct,
            "intraday_range_pct_text": self._format_pct(intraday_range_pct),
            "quote_trend": quote_trend,
            "quote_volatility_level": quote_volatility_level,
            "quote_score": round(quote_score, 3),
            "quote_signal": quote_signal,
            "agent_decision": agent_decision,
            "quote_features": quote_features,
            "reasoning_steps": reasoning_steps,
            "summary": summary
        }

    def analyse_historical(self, multi_quote: dict, validation_result: dict, historical_data: dict) -> dict:
        """
        Stage 2:
        Perform historical technical analysis using OHLCV data.
        """
        agent_goal = "Perform historical technical analysis and generate quantitative features."

        if not historical_data.get("success"):
            return {
                "success": False,
                "stage": "Stage 2: Historical Analysis",
                "agent_goal": agent_goal,
                "agent_decision": "Historical analysis skipped because historical data is unavailable.",
                "summary": historical_data.get("error", "Historical data request failed."),
                "historical_features": None,
                "reasoning_steps": []
            }

        symbol = multi_quote.get("symbol")
        price_records = historical_data.get("prices", [])
        feature_df = build_trading_features(price_records)

        if feature_df.empty:
            return {
                "success": False,
                "stage": "Stage 2: Historical Analysis",
                "agent_goal": agent_goal,
                "agent_decision": "Historical analysis failed because features could not be constructed.",
                "summary": "Feature construction failed.",
                "historical_features": None,
                "reasoning_steps": []
            }

        latest = feature_df.iloc[-1]

        return_1 = self._safe_float(latest.get("return_1"))
        return_5 = self._safe_float(latest.get("return_5"))
        return_20 = self._safe_float(latest.get("return_20"))
        ma_gap = self._safe_float(latest.get("ma_gap"))
        volatility_20 = self._safe_float(latest.get("volatility_20"))
        volume_change = self._safe_float(latest.get("volume_change"))
        rsi_14 = self._safe_float(latest.get("rsi_14"))

        reasoning_steps = [
            "Loaded historical daily OHLCV data from Alpha Vantage.",
            "Calculated return_1, return_5, return_20, MA gap, volatility, RSI, and volume change.",
            "Converted technical indicators into a historical analyst signal."
        ]

        if return_5 is not None and return_20 is not None:
            if return_5 > 0.02 and return_20 > 0:
                momentum_level = "Positive"
            elif return_5 < -0.02 and return_20 < 0:
                momentum_level = "Negative"
            else:
                momentum_level = "Mixed"
        else:
            momentum_level = "Unknown"

        if ma_gap is None:
            historical_trend = "Unknown"
        elif ma_gap > 0.02:
            historical_trend = "Uptrend"
        elif ma_gap < -0.02:
            historical_trend = "Downtrend"
        else:
            historical_trend = "Sideways"

        if volatility_20 is None:
            historical_volatility_level = "Unknown"
        elif volatility_20 > 0.04:
            historical_volatility_level = "High"
        elif volatility_20 > 0.02:
            historical_volatility_level = "Medium"
        else:
            historical_volatility_level = "Low"

        if rsi_14 is None:
            rsi_signal = "Unknown"
        elif rsi_14 >= 70:
            rsi_signal = "Overbought"
        elif rsi_14 <= 30:
            rsi_signal = "Oversold"
        else:
            rsi_signal = "Neutral"

        historical_score = 0.5

        if momentum_level == "Positive":
            historical_score += 0.15
        elif momentum_level == "Negative":
            historical_score -= 0.15

        if historical_trend == "Uptrend":
            historical_score += 0.15
        elif historical_trend == "Downtrend":
            historical_score -= 0.15

        if historical_volatility_level == "High":
            historical_score -= 0.15
        elif historical_volatility_level == "Low":
            historical_score += 0.05

        if rsi_signal == "Overbought":
            historical_score -= 0.10
        elif rsi_signal == "Oversold":
            historical_score += 0.05

        if volume_change is not None and volume_change > 0.2:
            historical_score += 0.05
        elif volume_change is not None and volume_change < -0.2:
            historical_score -= 0.05

        confidence_score = validation_result.get("confidence_score", 0.5)
        historical_score = max(0, min(1, historical_score * confidence_score))

        if historical_score >= 0.70:
            historical_signal = "HISTORICAL_BULLISH"
            agent_decision = "Historical indicators suggest positive technical conditions."
        elif historical_score <= 0.35:
            historical_signal = "HISTORICAL_BEARISH"
            agent_decision = "Historical indicators suggest weak technical conditions or elevated risk."
        elif historical_volatility_level == "High":
            historical_signal = "HISTORICAL_HIGH_VOLATILITY"
            agent_decision = "Historical volatility is elevated, so downstream agents should apply stricter risk control."
        else:
            historical_signal = "HISTORICAL_NEUTRAL"
            agent_decision = "Historical indicators do not show a strong directional signal."

        historical_features = {
            "return_1": return_1,
            "return_5": return_5,
            "return_20": return_20,
            "ma_gap": ma_gap,
            "volatility_20": volatility_20,
            "volume_change": volume_change,
            "rsi_14": rsi_14,
            "validation_confidence_score": confidence_score
        }

        summary = (
            f"{symbol} historical analysis: momentum is {momentum_level}, trend is {historical_trend}, "
            f"volatility is {historical_volatility_level}, RSI signal is {rsi_signal}, "
            f"and historical score is {historical_score:.2f}."
        )

        return {
            "success": True,
            "stage": "Stage 2: Historical Analysis",
            "agent_goal": agent_goal,
            "symbol": symbol,
            "latest_feature_date": str(latest.get("date")),
            "return_1": return_1,
            "return_1_pct": self._format_pct(return_1),
            "return_5": return_5,
            "return_5_pct": self._format_pct(return_5),
            "return_20": return_20,
            "return_20_pct": self._format_pct(return_20),
            "ma_gap": ma_gap,
            "ma_gap_pct": self._format_pct(ma_gap),
            "volatility_20": volatility_20,
            "volatility_20_pct": self._format_pct(volatility_20),
            "volume_change": volume_change,
            "volume_change_pct": self._format_pct(volume_change),
            "rsi_14": rsi_14,
            "momentum_level": momentum_level,
            "historical_trend": historical_trend,
            "historical_volatility_level": historical_volatility_level,
            "rsi_signal": rsi_signal,
            "historical_score": round(historical_score, 3),
            "historical_signal": historical_signal,
            "agent_decision": agent_decision,
            "historical_features": historical_features,
            "reasoning_steps": reasoning_steps,
            "summary": summary
        }

    def combine_analysis(self, multi_quote: dict, validation_result: dict, quote_result: dict, historical_result: dict) -> dict:
        """
        Stage 3:
        Combine quote-level and historical analysis into final analyst output.
        """
        agent_goal = "Combine quote-level and historical analysis into one analyst decision."

        symbol = multi_quote.get("symbol")
        selected_price = self._safe_float(validation_result.get("selected_price"))
        selected_source = validation_result.get("selected_source")

        reasoning_steps = []

        if not quote_result.get("success") and not historical_result.get("success"):
            return {
                "success": False,
                "agent_goal": agent_goal,
                "agent_decision": "Both quote-level and historical analysis failed, so no analyst decision can be generated.",
                "summary": "Analyst Agent failed.",
                "features_for_model": None,
                "analysis_for_next_agent": None,
                "stage_1_quote_analysis": quote_result,
                "stage_2_historical_analysis": historical_result
            }

        quote_score = quote_result.get("quote_score") if quote_result.get("success") else None
        historical_score = historical_result.get("historical_score") if historical_result.get("success") else None

        if quote_score is not None and historical_score is not None:
            final_score = 0.35 * quote_score + 0.65 * historical_score
            analysis_mode = "quote_and_historical"
            reasoning_steps.append("Combined quote-level score and historical score using weighted average.")
        elif historical_score is not None:
            final_score = historical_score
            analysis_mode = "historical_only"
            reasoning_steps.append("Used historical analysis only because quote-level analysis was unavailable.")
        else:
            final_score = quote_score
            analysis_mode = "quote_only"
            reasoning_steps.append("Used quote-level analysis only because historical analysis was unavailable.")

        final_score = max(0, min(1, final_score))

        # Prefer historical features for model training if available.
        if historical_result.get("success"):
            features_for_model = historical_result.get("historical_features")
            trend = historical_result.get("historical_trend")
            volatility_level = historical_result.get("historical_volatility_level")
            rsi_signal = historical_result.get("rsi_signal")
            momentum_level = historical_result.get("momentum_level")
        else:
            quote_features = quote_result.get("quote_features", {})
            features_for_model = {
                "return_1": quote_features.get("daily_return"),
                "return_5": None,
                "return_20": None,
                "ma_gap": None,
                "volatility_20": quote_features.get("intraday_range_pct"),
                "volume_change": None,
                "rsi_14": None,
                "validation_confidence_score": validation_result.get("confidence_score")
            }
            trend = quote_result.get("quote_trend")
            volatility_level = quote_result.get("quote_volatility_level")
            rsi_signal = "Unknown"
            momentum_level = "Quote-only"

        if final_score >= 0.70:
            analyst_signal = "BULLISH_WATCH"
            agent_decision = "The combined analysis suggests a positive watchlist signal."
        elif final_score <= 0.35:
            analyst_signal = "BEARISH_RISK"
            agent_decision = "The combined analysis suggests downside risk or weak technical conditions."
        elif volatility_level == "High":
            analyst_signal = "HIGH_VOLATILITY_CAUTION"
            agent_decision = "The stock shows elevated volatility, so downstream Risk Agent should be stricter."
        else:
            analyst_signal = "NEUTRAL"
            agent_decision = "The combined analysis does not show a strong directional signal."

        summary = (
            f"{symbol} combined analyst result: mode={analysis_mode}, "
            f"final analyst score={final_score:.2f}, signal={analyst_signal}."
        )

        return {
            "success": True,
            "agent_goal": agent_goal,
            "symbol": symbol,
            "selected_price": selected_price,
            "selected_source": selected_source,
            "analysis_mode": analysis_mode,
            "quote_score": quote_score,
            "historical_score": historical_score,
            "analyst_score": round(final_score, 3),
            "analyst_signal": analyst_signal,
            "trend": trend,
            "momentum_level": momentum_level,
            "volatility_level": volatility_level,
            "rsi_signal": rsi_signal,
            "agent_decision": agent_decision,
            "reasoning_steps": reasoning_steps,
            "features_for_model": features_for_model,
            "analysis_for_next_agent": {
                "symbol": symbol,
                "selected_price": selected_price,
                "analyst_score": round(final_score, 3),
                "analyst_signal": analyst_signal,
                "trend": trend,
                "momentum_level": momentum_level,
                "volatility_level": volatility_level,
                "rsi_signal": rsi_signal,
                "features_for_model": features_for_model,
                "analysis_mode": analysis_mode
            },
            "stage_1_quote_analysis": quote_result,
            "stage_2_historical_analysis": historical_result,
            "summary": summary
        }

    def analyse_market(self, multi_quote: dict, validation_result: dict, historical_data: dict) -> dict:
        """
        Public method used by app.py.

        Runs:
        1. quote-level analysis
        2. historical analysis
        3. combined decision
        """
        quote_result = self.analyse_quote_level(multi_quote, validation_result)
        historical_result = self.analyse_historical(multi_quote, validation_result, historical_data)

        return self.combine_analysis(
            multi_quote=multi_quote,
            validation_result=validation_result,
            quote_result=quote_result,
            historical_result=historical_result
        )