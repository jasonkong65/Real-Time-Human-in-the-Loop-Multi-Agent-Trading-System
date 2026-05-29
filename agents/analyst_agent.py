from typing import Any, Dict, List, Optional

import pandas as pd


class AnalystAgent:
    """
    Analyst Agent

    Makes a two-stage technical reading:
    1. quote-level movement from live data
    2. historical trend, momentum, RSI, volatility, and volume

    It separates trend direction from entry timing risk. A strong rising stock
    near a recent high is therefore shown as a bullish watchlist setup, not as a
    simple sell signal.
    """

    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _clip(self, value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    def _source(self, multi_quote: Dict[str, Any], names: List[str]) -> Dict[str, Any]:
        for name in names:
            value = multi_quote.get(name)
            if isinstance(value, dict):
                return value
        return {}

    def _price(self, quote: Dict[str, Any], key: str) -> Optional[float]:
        return self._safe_float(quote.get(key))

    def _to_df(self, historical_data: Dict[str, Any]) -> pd.DataFrame:
        prices = historical_data.get("prices", []) if isinstance(historical_data, dict) else []
        if not prices:
            return pd.DataFrame()
        df = pd.DataFrame(prices)
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date")
        return df.dropna(subset=required).reset_index(drop=True)

    def _rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    def _confidence_score(self, validation_result: Dict[str, Any]) -> float:
        score = self._safe_float(validation_result.get("confidence_score"))
        if score is not None:
            return self._clip(score)
        confidence = str(validation_result.get("confidence", "Medium")).lower()
        return {"high": 1.0, "medium": 0.75, "low": 0.45}.get(confidence, 0.6)

    # ------------------------------------------------------------------
    # Stage 1: live quote
    # ------------------------------------------------------------------
    def analyse_quote_level(self, multi_quote: Dict[str, Any], validation_result: Dict[str, Any]) -> Dict[str, Any]:
        primary = self._source(multi_quote, ["finnhub", "primary", "finnhub_quote"])
        secondary = self._source(multi_quote, ["alpha_vantage", "secondary", "alpha_vantage_quote"])
        quote = primary if primary.get("success") else secondary

        current = self._price(quote, "current_price") or self._safe_float(validation_result.get("selected_price"))
        previous = self._price(quote, "previous_close")
        open_price = self._price(quote, "open_price")
        high = self._price(quote, "high_price")
        low = self._price(quote, "low_price")

        daily_return = (current - previous) / previous if current and previous and previous > 0 else 0.0
        open_return = (current - open_price) / open_price if current and open_price and open_price > 0 else 0.0
        intraday_range = (high - low) / current if current and high and low and current > 0 and high >= low else 0.0

        score = 0.50
        reasons = []
        if daily_return > 0.01:
            score += 0.12
            reasons.append("price is up compared with the previous close")
        elif daily_return < -0.01:
            score -= 0.12
            reasons.append("price is down compared with the previous close")
        if open_return > 0.004:
            score += 0.06
            reasons.append("price is above the open")
        elif open_return < -0.004:
            score -= 0.06
            reasons.append("price is below the open")
        if intraday_range > 0.05:
            score -= 0.10
            volatility = "High"
            reasons.append("intraday range is wide")
        elif intraday_range > 0.025:
            score -= 0.03
            volatility = "Medium"
            reasons.append("intraday range is moderate")
        else:
            score += 0.03
            volatility = "Low"
            reasons.append("intraday range is calm")

        score = self._clip(score * (0.85 + 0.15 * self._confidence_score(validation_result)))
        if score >= 0.62:
            quote_signal = "POSITIVE_QUOTE"
        elif score <= 0.38:
            quote_signal = "NEGATIVE_QUOTE"
        else:
            quote_signal = "NEUTRAL_QUOTE"

        return {
            "quote_score": round(score, 4),
            "quote_signal": quote_signal,
            "daily_return": round(daily_return, 6),
            "open_to_current_return": round(open_return, 6),
            "intraday_range": round(intraday_range, 6),
            "quote_volatility_level": volatility,
            "reason": "; ".join(reasons),
        }

    # ------------------------------------------------------------------
    # Stage 2: historical technical picture
    # ------------------------------------------------------------------
    def analyse_historical(self, multi_quote: Dict[str, Any], validation_result: Dict[str, Any], historical_data: Dict[str, Any]) -> Dict[str, Any]:
        df = self._to_df(historical_data)
        if df.empty or len(df) < 30:
            return {
                "historical_score": 0.5,
                "historical_signal": "NEUTRAL_HISTORY",
                "trend_direction": "Neutral",
                "entry_risk_level": "Medium",
                "volatility_level": "Unknown",
                "reason": "Not enough historical data for a strong technical view.",
            }

        close = df["close"]
        volume = df["volume"]
        ret_1 = close.pct_change(1).iloc[-1]
        ret_5 = close.pct_change(5).iloc[-1]
        ret_20 = close.pct_change(20).iloc[-1]
        ma_20 = close.rolling(20).mean().iloc[-1]
        ma_50 = close.rolling(50).mean().iloc[-1] if len(df) >= 50 else close.rolling(20).mean().iloc[-1]
        ma_gap = (ma_20 - ma_50) / ma_50 if ma_50 and ma_50 > 0 else 0.0
        volatility_20 = close.pct_change().rolling(20).std().iloc[-1]
        rsi_14 = self._rsi(close, 14).iloc[-1]
        volume_change = (volume.rolling(5).mean().iloc[-1] - volume.rolling(20).mean().iloc[-1]) / volume.rolling(20).mean().iloc[-1]
        period_high = close.tail(60).max()
        distance_to_high = (period_high - close.iloc[-1]) / period_high if period_high and period_high > 0 else 0.0

        ret_1 = self._safe_float(ret_1, 0.0) or 0.0
        ret_5 = self._safe_float(ret_5, 0.0) or 0.0
        ret_20 = self._safe_float(ret_20, 0.0) or 0.0
        ma_gap = self._safe_float(ma_gap, 0.0) or 0.0
        volatility_20 = self._safe_float(volatility_20, 0.0) or 0.0
        rsi_14 = self._safe_float(rsi_14, 50.0) or 50.0
        volume_change = self._safe_float(volume_change, 0.0) or 0.0
        distance_to_high = self._safe_float(distance_to_high, 0.0) or 0.0

        score = 0.50
        reasons = []
        if ret_20 > 0.08:
            score += 0.16
            reasons.append("20-day momentum is strong")
        elif ret_20 > 0.02:
            score += 0.09
            reasons.append("20-day momentum is positive")
        elif ret_20 < -0.05:
            score -= 0.16
            reasons.append("20-day momentum is weak")
        elif ret_20 < -0.02:
            score -= 0.08
            reasons.append("20-day momentum is slightly negative")

        if ret_5 > 0.02:
            score += 0.07
            reasons.append("short-term momentum is positive")
        elif ret_5 < -0.02:
            score -= 0.07
            reasons.append("short-term momentum is negative")

        if ma_gap > 0.02:
            score += 0.10
            reasons.append("short-term average is above the medium-term average")
        elif ma_gap < -0.02:
            score -= 0.10
            reasons.append("short-term average is below the medium-term average")

        if volatility_20 > 0.045:
            score -= 0.08
            volatility_level = "High"
            reasons.append("recent volatility is high")
        elif volatility_20 > 0.025:
            score -= 0.02
            volatility_level = "Medium"
            reasons.append("recent volatility is moderate")
        else:
            score += 0.04
            volatility_level = "Low"
            reasons.append("recent volatility is low")

        entry_risk_points = 0
        if rsi_14 >= 78:
            entry_risk_points += 2
            score -= 0.08
            reasons.append("RSI is strongly overbought")
        elif rsi_14 >= 70:
            entry_risk_points += 1
            score -= 0.04
            reasons.append("RSI is overbought")
        elif rsi_14 <= 30:
            entry_risk_points += 1
            reasons.append("RSI is oversold")

        if distance_to_high <= 0.03 and ret_20 > 0:
            entry_risk_points += 1
            reasons.append("price is close to the recent high")
        if ret_20 > 0.12:
            entry_risk_points += 1
            reasons.append("recent move may be stretched")
        if volume_change > 0.4:
            score += 0.03
            reasons.append("volume is above recent average")

        score = self._clip(score)
        if score >= 0.68:
            hist_signal = "BULLISH_HISTORY"
        elif score <= 0.35:
            hist_signal = "BEARISH_HISTORY"
        else:
            hist_signal = "MIXED_HISTORY"

        if ret_20 > 0.03 and ma_gap > 0:
            trend_direction = "Positive"
        elif ret_20 < -0.03 and ma_gap < 0:
            trend_direction = "Negative"
        else:
            trend_direction = "Neutral"

        if entry_risk_points >= 3:
            entry_risk_level = "High"
        elif entry_risk_points >= 1:
            entry_risk_level = "Medium"
        else:
            entry_risk_level = "Low"

        return {
            "historical_score": round(score, 4),
            "historical_signal": hist_signal,
            "return_1": round(ret_1, 6),
            "return_5": round(ret_5, 6),
            "return_20": round(ret_20, 6),
            "ma_gap": round(ma_gap, 6),
            "volatility_20": round(volatility_20, 6),
            "volume_change": round(volume_change, 6),
            "rsi_14": round(rsi_14, 2),
            "distance_to_recent_high": round(distance_to_high, 6),
            "trend_direction": trend_direction,
            "entry_risk_level": entry_risk_level,
            "volatility_level": volatility_level,
            "reason": "; ".join(reasons),
        }

    def combine_analysis(self, multi_quote: Dict[str, Any], validation_result: Dict[str, Any], quote_result: Dict[str, Any], historical_result: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(multi_quote.get("symbol") or validation_result.get("symbol") or "UNKNOWN").upper()
        quote_score = self._safe_float(quote_result.get("quote_score"), 0.5) or 0.5
        historical_score = self._safe_float(historical_result.get("historical_score"), 0.5) or 0.5
        analyst_score = self._clip(0.35 * quote_score + 0.65 * historical_score)
        trend = historical_result.get("trend_direction", "Neutral")
        entry_risk = historical_result.get("entry_risk_level", "Medium")
        volatility = historical_result.get("volatility_level") or quote_result.get("quote_volatility_level", "Unknown")

        if trend == "Positive" and analyst_score >= 0.60:
            if entry_risk == "High":
                analyst_signal = "POSITIVE_BUT_ENTRY_RISK"
                display_signal = "WATCHLIST_BULLISH_ENTRY_RISK"
                decision = "The setup is positive, but entry timing risk is high."
            elif entry_risk == "Medium":
                analyst_signal = "WATCHLIST_BULLISH"
                display_signal = "BULLISH_WATCHLIST"
                decision = "The setup is positive, but it still needs confirmation."
            else:
                analyst_signal = "BULLISH"
                display_signal = "BULLISH"
                decision = "The technical picture is positive."
        elif trend == "Negative" and analyst_score <= 0.42:
            analyst_signal = "BEARISH_RISK"
            display_signal = "DOWNSIDE_RISK"
            decision = "The technical picture points to downside risk."
        elif analyst_score >= 0.66:
            analyst_signal = "WATCHLIST_BULLISH"
            display_signal = "BULLISH_WATCHLIST"
            decision = "The setup leans positive but is not a direct entry signal."
        elif analyst_score <= 0.36:
            analyst_signal = "BEARISH_RISK"
            display_signal = "DOWNSIDE_RISK"
            decision = "The setup leans weak and needs caution."
        else:
            analyst_signal = "NEUTRAL"
            display_signal = "NEUTRAL"
            decision = "The combined analysis does not show a strong directional signal."

        reasoning_steps = [
            f"Quote score: {quote_score:.3f}.",
            f"Historical score: {historical_score:.3f}.",
            f"Combined analyst score: {analyst_score:.3f}.",
            f"Trend direction: {trend}.",
            f"Entry timing risk: {entry_risk}.",
        ]

        features = {
            "return_1": historical_result.get("return_1", 0.0),
            "return_5": historical_result.get("return_5", 0.0),
            "return_20": historical_result.get("return_20", 0.0),
            "ma_gap": historical_result.get("ma_gap", 0.0),
            "volatility_20": historical_result.get("volatility_20", 0.0),
            "volume_change": historical_result.get("volume_change", 0.0),
            "rsi_14": historical_result.get("rsi_14", 50.0),
            "validation_confidence_score": validation_result.get("confidence_score", 0.6),
        }

        return {
            "success": True,
            "agent": "Analyst Agent",
            "agent_goal": "Read market conditions and separate trend from entry risk.",
            "symbol": symbol,
            "analyst_signal": analyst_signal,
            "display_signal": display_signal,
            "analyst_score": round(analyst_score, 4),
            "trend_direction": trend,
            "entry_risk_level": entry_risk,
            "volatility_level": volatility,
            "features_for_model": features,
            "stage_1_quote_analysis": quote_result,
            "stage_2_historical_analysis": historical_result,
            "agent_decision": decision,
            "reasoning_steps": reasoning_steps,
            "summary": f"Analyst Agent result for {symbol}: {display_signal}.",
        }

    def analyse_market(self, multi_quote: Dict[str, Any], validation_result: Dict[str, Any], historical_data: Dict[str, Any]) -> Dict[str, Any]:
        quote = self.analyse_quote_level(multi_quote, validation_result)
        history = self.analyse_historical(multi_quote, validation_result, historical_data)
        return self.combine_analysis(multi_quote, validation_result, quote, history)

    # US spelling and generic aliases
    def analyze_market(self, multi_quote: Dict[str, Any], validation_result: Dict[str, Any], historical_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.analyse_market(multi_quote, validation_result, historical_data)

    def analyse(self, multi_quote: Dict[str, Any], validation_result: Dict[str, Any], historical_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.analyse_market(multi_quote, validation_result, historical_data)

    def analyze(self, multi_quote: Dict[str, Any], validation_result: Dict[str, Any], historical_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.analyse_market(multi_quote, validation_result, historical_data)

    def run(self, multi_quote: Dict[str, Any], validation_result: Dict[str, Any], historical_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.analyse_market(multi_quote, validation_result, historical_data)
