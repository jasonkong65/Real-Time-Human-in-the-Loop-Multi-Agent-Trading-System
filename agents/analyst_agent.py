from typing import Any, Dict, List, Optional, Tuple

from utils.features import build_trading_features


class AnalystAgent:
    """
    Two-Stage Analyst Agent.

    Stage 1: Quote-level analysis using live quote data.
    Stage 2: Historical technical analysis using OHLCV data.
    Stage 3: Combine both stages into a final analyst decision.

    Optimised design:
    - Separates trend direction from entry risk.
    - Avoids treating a strong uptrend near recent highs as SELL_RISK.
    - Produces more useful labels such as WATCHLIST_BULLISH and
      POSITIVE_BUT_ENTRY_RISK.
    - Keeps the original output keys used by app.py and downstream agents.
    """

    # -----------------------------
    # Basic helpers
    # -----------------------------
    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    @staticmethod
    def _format_pct(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        return f"{value:.2%}"

    @staticmethod
    def _get_first_available(record: Dict[str, Any], keys: List[str]) -> Optional[float]:
        for key in keys:
            if key in record:
                try:
                    value = record.get(key)
                    if value is not None:
                        return float(value)
                except (TypeError, ValueError):
                    continue
        return None

    def _confidence_adjust_score(self, raw_score: float, validation_result: Dict[str, Any]) -> float:
        """
        Validation confidence should affect how strongly we trust the score,
        but it should not fully erase the market signal.

        Old version multiplied the score directly by confidence_score. That could
        turn a reasonable bullish technical setup into a neutral result. This
        version applies a milder confidence adjustment.
        """
        confidence_score = self._safe_float(validation_result.get("confidence_score"))
        if confidence_score is None:
            confidence_score = 0.5

        confidence_score = self._clip(confidence_score)
        adjustment = 0.85 + 0.15 * confidence_score
        return self._clip(raw_score * adjustment)

    def _extract_close_series(self, price_records: List[Dict[str, Any]]) -> List[float]:
        closes: List[float] = []
        for record in price_records or []:
            if not isinstance(record, dict):
                continue
            close_price = self._get_first_available(
                record,
                ["close", "c", "Close", "adj_close", "Adj Close"]
            )
            if close_price is not None and close_price > 0:
                closes.append(close_price)
        return closes

    # -----------------------------
    # Market interpretation helpers
    # -----------------------------
    def _classify_rsi(self, rsi_14: Optional[float]) -> str:
        if rsi_14 is None:
            return "Unknown"
        if rsi_14 >= 75:
            return "Strongly Overbought"
        if rsi_14 >= 70:
            return "Overbought"
        if rsi_14 <= 25:
            return "Strongly Oversold"
        if rsi_14 <= 30:
            return "Oversold"
        return "Neutral"

    def _classify_momentum(
        self,
        return_5: Optional[float],
        return_20: Optional[float]
    ) -> str:
        if return_5 is None or return_20 is None:
            return "Unknown"

        if return_5 > 0.02 and return_20 > 0.03:
            return "Strong Positive"
        if return_5 > 0.00 and return_20 > 0.00:
            return "Positive"
        if return_5 < -0.02 and return_20 < -0.03:
            return "Strong Negative"
        if return_5 < 0.00 and return_20 < 0.00:
            return "Negative"
        return "Mixed"

    def _classify_trend(self, ma_gap: Optional[float]) -> str:
        if ma_gap is None:
            return "Unknown"
        if ma_gap > 0.05:
            return "Strong Uptrend"
        if ma_gap > 0.02:
            return "Uptrend"
        if ma_gap < -0.05:
            return "Strong Downtrend"
        if ma_gap < -0.02:
            return "Downtrend"
        return "Sideways"

    def _classify_volatility(self, volatility_20: Optional[float]) -> str:
        if volatility_20 is None:
            return "Unknown"
        if volatility_20 > 0.05:
            return "Very High"
        if volatility_20 > 0.035:
            return "High"
        if volatility_20 > 0.02:
            return "Medium"
        return "Low"

    def _derive_trend_direction(self, momentum_level: str, historical_trend: str) -> str:
        positive_momentum = momentum_level in ["Positive", "Strong Positive"]
        negative_momentum = momentum_level in ["Negative", "Strong Negative"]
        uptrend = historical_trend in ["Uptrend", "Strong Uptrend"]
        downtrend = historical_trend in ["Downtrend", "Strong Downtrend"]

        if positive_momentum and uptrend:
            return "Positive"
        if negative_momentum and downtrend:
            return "Negative"
        if positive_momentum or uptrend:
            return "Mild Positive"
        if negative_momentum or downtrend:
            return "Mild Negative"
        return "Neutral"

    def _assess_entry_risk(
        self,
        rsi_signal: str,
        volatility_level: str,
        return_20: Optional[float],
        ma_gap: Optional[float],
        price_to_period_high: Optional[float]
    ) -> Tuple[str, List[str]]:
        """
        Entry risk is not the same as trend direction. A strong stock can have
        elevated entry risk if it is overbought or extended.
        """
        reasons: List[str] = []
        risk_points = 0

        if rsi_signal == "Strongly Overbought":
            risk_points += 2
            reasons.append("RSI is strongly overbought")
        elif rsi_signal == "Overbought":
            risk_points += 1
            reasons.append("RSI is overbought")

        if volatility_level in ["Very High", "High"]:
            risk_points += 1
            reasons.append("recent volatility is elevated")

        if return_20 is not None and return_20 >= 0.15:
            risk_points += 2
            reasons.append("20-day return is very extended")
        elif return_20 is not None and return_20 >= 0.08:
            risk_points += 1
            reasons.append("20-day return is already strong")

        if ma_gap is not None and ma_gap >= 0.10:
            risk_points += 2
            reasons.append("price is far above the moving-average reference")
        elif ma_gap is not None and ma_gap >= 0.06:
            risk_points += 1
            reasons.append("price is above the moving-average reference")

        if price_to_period_high is not None and price_to_period_high >= 0.98:
            risk_points += 1
            reasons.append("price is close to the recent period high")

        if risk_points >= 3:
            return "High", reasons
        if risk_points >= 1:
            return "Elevated", reasons
        return "Normal", reasons

    def _build_market_context(
        self,
        momentum_level: str,
        historical_trend: str,
        volatility_level: str,
        rsi_signal: str,
        entry_risk_level: str,
        trend_direction: str
    ) -> str:
        if trend_direction in ["Positive", "Mild Positive"] and entry_risk_level in ["Elevated", "High"]:
            return "Positive trend with elevated entry risk"
        if trend_direction in ["Positive", "Mild Positive"]:
            return "Positive technical setup"
        if trend_direction in ["Negative", "Mild Negative"]:
            return "Weak or negative technical setup"
        if volatility_level in ["High", "Very High"]:
            return "Unclear trend with elevated volatility"
        if rsi_signal in ["Oversold", "Strongly Oversold"]:
            return "Possible oversold rebound watch"
        return "Neutral or mixed technical setup"

    # -----------------------------
    # Stage 1: Quote-level analysis
    # -----------------------------
    def analyse_quote_level(self, multi_quote: dict, validation_result: dict) -> dict:
        """
        Perform fast quote-level analysis using validated live quote data.
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

        symbol = multi_quote.get("symbol")
        selected_price = self._safe_float(validation_result.get("selected_price"))

        finnhub = multi_quote.get("finnhub", {}) or {}

        previous_close = self._safe_float(finnhub.get("previous_close_price"))
        open_price = self._safe_float(finnhub.get("open_price"))
        high_price = self._safe_float(finnhub.get("high_price"))
        low_price = self._safe_float(finnhub.get("low_price"))

        if selected_price is None or previous_close is None or previous_close <= 0:
            return {
                "success": False,
                "stage": "Stage 1: Quote-level Analysis",
                "agent_goal": agent_goal,
                "symbol": symbol,
                "agent_decision": "Quote-level analysis cannot continue because key price fields are missing.",
                "summary": "Quote-level analysis failed.",
                "quote_features": None,
                "reasoning_steps": []
            }

        daily_return = (selected_price - previous_close) / previous_close

        open_to_current_return = None
        if open_price is not None and open_price > 0:
            open_to_current_return = (selected_price - open_price) / open_price

        intraday_range_pct = None
        if high_price is not None and low_price is not None and selected_price > 0:
            intraday_range_pct = (high_price - low_price) / selected_price

        reasoning_steps = [
            f"Calculated daily return using selected price {selected_price} and previous close {previous_close}."
        ]

        if open_price is not None:
            reasoning_steps.append(f"Calculated open-to-current return using open price {open_price}.")

        if high_price is not None and low_price is not None:
            reasoning_steps.append(f"Calculated intraday range using high price {high_price} and low price {low_price}.")

        if daily_return > 0.02:
            quote_trend = "Strong upward"
        elif daily_return > 0.005:
            quote_trend = "Slight upward"
        elif daily_return < -0.02:
            quote_trend = "Strong downward"
        elif daily_return < -0.005:
            quote_trend = "Slight downward"
        else:
            quote_trend = "Neutral"

        if intraday_range_pct is None:
            quote_volatility_level = "Unknown"
        elif intraday_range_pct > 0.04:
            quote_volatility_level = "High"
        elif intraday_range_pct > 0.02:
            quote_volatility_level = "Medium"
        else:
            quote_volatility_level = "Low"

        raw_quote_score = 0.5

        if daily_return > 0.02:
            raw_quote_score += 0.20
        elif daily_return > 0.005:
            raw_quote_score += 0.10
        elif daily_return < -0.02:
            raw_quote_score -= 0.20
        elif daily_return < -0.005:
            raw_quote_score -= 0.10

        if open_to_current_return is not None:
            if open_to_current_return > 0.01:
                raw_quote_score += 0.10
            elif open_to_current_return < -0.01:
                raw_quote_score -= 0.10

        if quote_volatility_level == "High":
            raw_quote_score -= 0.10
        elif quote_volatility_level == "Low":
            raw_quote_score += 0.05

        raw_quote_score = self._clip(raw_quote_score)
        quote_score = self._confidence_adjust_score(raw_quote_score, validation_result)

        if quote_score >= 0.68:
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
            "raw_quote_score": raw_quote_score,
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
            "raw_quote_score": round(raw_quote_score, 3),
            "quote_score": round(quote_score, 3),
            "quote_signal": quote_signal,
            "agent_decision": agent_decision,
            "quote_features": quote_features,
            "reasoning_steps": reasoning_steps,
            "summary": summary
        }

    # -----------------------------
    # Stage 2: Historical analysis
    # -----------------------------
    def analyse_historical(
        self,
        multi_quote: dict,
        validation_result: dict,
        historical_data: dict
    ) -> dict:
        """
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
        historical_source = historical_data.get("source", "historical data source")
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

        close_series = self._extract_close_series(price_records)
        latest_close = close_series[-1] if close_series else None
        period_high = max(close_series[-60:]) if close_series else None
        price_to_period_high = None
        distance_from_period_high = None

        if latest_close is not None and period_high is not None and period_high > 0:
            price_to_period_high = latest_close / period_high
            distance_from_period_high = (latest_close - period_high) / period_high

        reasoning_steps = [
            f"Loaded historical daily OHLCV data from {historical_source}.",
            "Calculated return_1, return_5, return_20, MA gap, volatility, RSI, volume change, and recent-high proximity.",
            "Converted technical indicators into direction and entry-risk interpretations."
        ]

        momentum_level = self._classify_momentum(return_5, return_20)
        historical_trend = self._classify_trend(ma_gap)
        historical_volatility_level = self._classify_volatility(volatility_20)
        rsi_signal = self._classify_rsi(rsi_14)
        trend_direction = self._derive_trend_direction(momentum_level, historical_trend)

        entry_risk_level, entry_risk_reasons = self._assess_entry_risk(
            rsi_signal=rsi_signal,
            volatility_level=historical_volatility_level,
            return_20=return_20,
            ma_gap=ma_gap,
            price_to_period_high=price_to_period_high
        )

        market_context = self._build_market_context(
            momentum_level=momentum_level,
            historical_trend=historical_trend,
            volatility_level=historical_volatility_level,
            rsi_signal=rsi_signal,
            entry_risk_level=entry_risk_level,
            trend_direction=trend_direction
        )

        raw_historical_score = 0.5

        if momentum_level == "Strong Positive":
            raw_historical_score += 0.18
        elif momentum_level == "Positive":
            raw_historical_score += 0.12
        elif momentum_level == "Strong Negative":
            raw_historical_score -= 0.18
        elif momentum_level == "Negative":
            raw_historical_score -= 0.12

        if historical_trend == "Strong Uptrend":
            raw_historical_score += 0.18
        elif historical_trend == "Uptrend":
            raw_historical_score += 0.12
        elif historical_trend == "Strong Downtrend":
            raw_historical_score -= 0.18
        elif historical_trend == "Downtrend":
            raw_historical_score -= 0.12

        if historical_volatility_level == "Very High":
            raw_historical_score -= 0.12
        elif historical_volatility_level == "High":
            raw_historical_score -= 0.08
        elif historical_volatility_level == "Low":
            raw_historical_score += 0.04

        # RSI is treated as entry timing risk, not automatic bearish direction.
        if rsi_signal == "Strongly Overbought":
            raw_historical_score -= 0.06
        elif rsi_signal == "Overbought":
            raw_historical_score -= 0.04
        elif rsi_signal in ["Oversold", "Strongly Oversold"]:
            raw_historical_score += 0.04

        if volume_change is not None and volume_change > 0.2:
            raw_historical_score += 0.04
        elif volume_change is not None and volume_change < -0.2:
            raw_historical_score -= 0.04

        raw_historical_score = self._clip(raw_historical_score)
        historical_score = self._confidence_adjust_score(raw_historical_score, validation_result)

        if historical_score >= 0.72 and entry_risk_level not in ["High"]:
            historical_signal = "HISTORICAL_BULLISH"
            agent_decision = "Historical indicators suggest positive technical conditions."
        elif historical_score >= 0.60 and trend_direction in ["Positive", "Mild Positive"] and entry_risk_level in ["Elevated", "High"]:
            historical_signal = "HISTORICAL_POSITIVE_BUT_ENTRY_RISK"
            agent_decision = "Historical indicators are positive, but entry timing risk is elevated."
        elif historical_score <= 0.35:
            historical_signal = "HISTORICAL_BEARISH"
            agent_decision = "Historical indicators suggest downside risk."
        elif historical_volatility_level in ["High", "Very High"]:
            historical_signal = "HISTORICAL_HIGH_VOLATILITY"
            agent_decision = "Historical indicators show elevated volatility."
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
            "validation_confidence_score": validation_result.get("confidence_score"),
            "latest_close": latest_close,
            "period_high_60": period_high,
            "price_to_period_high": price_to_period_high,
            "distance_from_period_high": distance_from_period_high,
            "trend_direction": trend_direction,
            "entry_risk_level": entry_risk_level
        }

        summary = (
            f"{symbol} historical analysis: momentum is {momentum_level}, trend is {historical_trend}, "
            f"trend direction is {trend_direction}, volatility is {historical_volatility_level}, "
            f"RSI signal is {rsi_signal}, entry risk is {entry_risk_level}, "
            f"and historical score is {historical_score:.2f}."
        )

        return {
            "success": True,
            "stage": "Stage 2: Historical Analysis",
            "agent_goal": agent_goal,
            "symbol": symbol,
            "historical_source": historical_source,
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
            "latest_close": latest_close,
            "period_high_60": period_high,
            "price_to_period_high": price_to_period_high,
            "distance_from_period_high": distance_from_period_high,
            "distance_from_period_high_pct": self._format_pct(distance_from_period_high),
            "momentum_level": momentum_level,
            "historical_trend": historical_trend,
            "historical_volatility_level": historical_volatility_level,
            "rsi_signal": rsi_signal,
            "trend_direction": trend_direction,
            "entry_risk_level": entry_risk_level,
            "entry_risk_reasons": entry_risk_reasons,
            "market_context": market_context,
            "raw_historical_score": round(raw_historical_score, 3),
            "historical_score": round(historical_score, 3),
            "historical_signal": historical_signal,
            "agent_decision": agent_decision,
            "historical_features": historical_features,
            "reasoning_steps": reasoning_steps,
            "summary": summary
        }

    # -----------------------------
    # Stage 3: Combine quote and historical results
    # -----------------------------
    def _classify_combined_signal(
        self,
        final_score: float,
        trend_direction: str,
        volatility_level: str,
        rsi_signal: str,
        entry_risk_level: str,
        momentum_level: str
    ) -> Tuple[str, str, str, str]:
        """
        Returns:
        - analyst_signal
        - display_signal
        - agent_decision
        - strategy_hint_for_next_agent
        """
        positive_trend = trend_direction in ["Positive", "Mild Positive"]
        negative_trend = trend_direction in ["Negative", "Mild Negative"]
        high_entry_risk = entry_risk_level in ["Elevated", "High"] or rsi_signal in ["Overbought", "Strongly Overbought"]

        if final_score >= 0.72 and positive_trend and not high_entry_risk:
            return (
                "BULLISH_WATCH",
                "BULLISH_WATCH",
                "The combined analysis suggests a positive watchlist signal.",
                "Constructive setup; downstream agents may consider buy-candidate logic if model confirmation also supports it."
            )

        if final_score >= 0.60 and positive_trend and high_entry_risk:
            return (
                "POSITIVE_BUT_ENTRY_RISK",
                "WATCHLIST_BULLISH_ENTRY_RISK",
                "The combined analysis suggests a positive trend, but entry timing risk is elevated.",
                "Do not treat this as a direct buy. Consider wait-for-pullback or confirmation logic."
            )

        if final_score >= 0.60 and positive_trend:
            return (
                "WATCHLIST_BULLISH",
                "WATCHLIST_BULLISH",
                "The combined analysis is mildly positive, but not strong enough for an aggressive signal.",
                "Monitor as a constructive watchlist candidate."
            )

        if final_score <= 0.35 and negative_trend:
            return (
                "BEARISH_RISK",
                "BEARISH_RISK",
                "The combined analysis suggests downside risk or weak technical conditions.",
                "Downstream agents should apply stricter risk control."
            )

        if final_score <= 0.42 and negative_trend:
            return (
                "WATCHLIST_BEARISH",
                "WATCHLIST_BEARISH",
                "The combined analysis is mildly negative and needs caution.",
                "Avoid aggressive long exposure unless conditions improve."
            )

        if volatility_level in ["High", "Very High"]:
            return (
                "HIGH_VOLATILITY_CAUTION",
                "HIGH_VOLATILITY_CAUTION",
                "The stock shows elevated volatility, so downstream Risk Agent should be stricter.",
                "Use stricter risk limits because volatility is elevated."
            )

        if rsi_signal in ["Oversold", "Strongly Oversold"] and not negative_trend:
            return (
                "OVERSOLD_REBOUND_WATCH",
                "OVERSOLD_REBOUND_WATCH",
                "The stock may be oversold, but confirmation is still needed.",
                "Monitor for rebound confirmation rather than acting immediately."
            )

        return (
            "NEUTRAL",
            "NEUTRAL",
            "The combined analysis does not show a strong directional signal.",
            "Maintain monitoring stance until stronger evidence appears."
        )

    def combine_analysis(
        self,
        multi_quote: dict,
        validation_result: dict,
        quote_result: dict,
        historical_result: dict
    ) -> dict:
        """
        Combine quote-level and historical analysis into final analyst output.
        """
        agent_goal = "Combine quote-level and historical analysis into one analyst decision."

        symbol = multi_quote.get("symbol")
        selected_price = self._safe_float(validation_result.get("selected_price"))
        selected_source = validation_result.get("selected_source")

        reasoning_steps: List[str] = []

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
            final_score = 0.30 * float(quote_score) + 0.70 * float(historical_score)
            analysis_mode = "quote_and_historical"
            reasoning_steps.append("Combined quote-level score and historical score using a 30/70 weighted average.")
        elif historical_score is not None:
            final_score = float(historical_score)
            analysis_mode = "historical_only"
            reasoning_steps.append("Used historical analysis only because quote-level analysis was unavailable.")
        else:
            final_score = float(quote_score)
            analysis_mode = "quote_only"
            reasoning_steps.append("Used quote-level analysis only because historical analysis was unavailable.")

        final_score = self._clip(final_score)

        if historical_result.get("success"):
            features_for_model = historical_result.get("historical_features")
            trend = historical_result.get("historical_trend")
            volatility_level = historical_result.get("historical_volatility_level")
            rsi_signal = historical_result.get("rsi_signal")
            momentum_level = historical_result.get("momentum_level")
            trend_direction = historical_result.get("trend_direction")
            entry_risk_level = historical_result.get("entry_risk_level")
            entry_risk_reasons = historical_result.get("entry_risk_reasons", [])
            market_context = historical_result.get("market_context")
        else:
            quote_features = quote_result.get("quote_features", {}) or {}
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
            trend_direction = "Mild Positive" if quote_result.get("quote_signal") == "QUOTE_BULLISH" else "Neutral"
            entry_risk_level = "Elevated" if volatility_level == "High" else "Normal"
            entry_risk_reasons = ["quote-only volatility check"] if volatility_level == "High" else []
            market_context = "Quote-only technical setup"

        analyst_signal, display_signal, agent_decision, strategy_hint = self._classify_combined_signal(
            final_score=final_score,
            trend_direction=trend_direction,
            volatility_level=volatility_level,
            rsi_signal=rsi_signal,
            entry_risk_level=entry_risk_level,
            momentum_level=momentum_level
        )

        reasoning_steps.append(
            f"Separated trend direction ({trend_direction}) from entry risk ({entry_risk_level})."
        )
        reasoning_steps.append(
            f"Generated analyst signal {analyst_signal} with display signal {display_signal}."
        )

        if entry_risk_reasons:
            reasoning_steps.append("Entry risk reasons: " + "; ".join(entry_risk_reasons) + ".")

        summary = (
            f"{symbol} combined analyst result: mode={analysis_mode}, "
            f"final analyst score={final_score:.2f}, signal={analyst_signal}, "
            f"trend direction={trend_direction}, entry risk={entry_risk_level}."
        )

        analysis_for_next_agent = {
            "symbol": symbol,
            "selected_price": selected_price,
            "analyst_score": round(final_score, 3),
            "analyst_signal": analyst_signal,
            "display_signal": display_signal,
            "trend": trend,
            "trend_direction": trend_direction,
            "momentum_level": momentum_level,
            "volatility_level": volatility_level,
            "rsi_signal": rsi_signal,
            "entry_risk_level": entry_risk_level,
            "entry_risk_reasons": entry_risk_reasons,
            "market_context": market_context,
            "strategy_hint": strategy_hint,
            "features_for_model": features_for_model,
            "analysis_mode": analysis_mode
        }

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
            "display_signal": display_signal,
            "trend": trend,
            "trend_direction": trend_direction,
            "momentum_level": momentum_level,
            "volatility_level": volatility_level,
            "rsi_signal": rsi_signal,
            "entry_risk_level": entry_risk_level,
            "entry_risk_reasons": entry_risk_reasons,
            "market_context": market_context,
            "strategy_hint": strategy_hint,
            "agent_decision": agent_decision,
            "reasoning_steps": reasoning_steps,
            "features_for_model": features_for_model,
            "analysis_for_next_agent": analysis_for_next_agent,
            "stage_1_quote_analysis": quote_result,
            "stage_2_historical_analysis": historical_result,
            "summary": summary
        }

    # -----------------------------
    # Public method used by app.py
    # -----------------------------
    def analyse_market(
        self,
        multi_quote: dict,
        validation_result: dict,
        historical_data: dict
    ) -> dict:
        """
        Public method used by app.py.
        """
        quote_result = self.analyse_quote_level(multi_quote, validation_result)
        historical_result = self.analyse_historical(multi_quote, validation_result, historical_data)

        return self.combine_analysis(
            multi_quote=multi_quote,
            validation_result=validation_result,
            quote_result=quote_result,
            historical_result=historical_result
        )
