from __future__ import annotations

import json

from datetime import datetime, timezone, timedelta

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from utils.features import build_trading_features
except Exception:
    build_trading_features = None


class AnalystHistoricalMixin:


    def analyse_historical(self, multi_quote: dict, validation_result: dict, historical_data: dict) -> dict:
        agent_goal = "Read historical trend, momentum, volatility, RSI and volume."

        if not isinstance(historical_data, dict) or not historical_data.get("success"):
            return {
                "success": False,
                "stage": "Stage 2: Historical Analysis",
                "agent_goal": agent_goal,
                "agent_decision": "Historical data is unavailable.",
                "summary": historical_data.get("error", "Historical data request failed.") if isinstance(historical_data, dict) else "Historical data request failed.",
                "historical_features": None,
                "reasoning_steps": [],
                "indicator_contributions": [],
            }

        symbol = str(multi_quote.get("symbol") or historical_data.get("symbol") or "UNKNOWN").upper()
        historical_source = historical_data.get("source", "historical data")
        price_records = historical_data.get("prices", [])
        feature_df = self._build_features(price_records)

        if feature_df.empty:
            return {
                "success": False,
                "stage": "Stage 2: Historical Analysis",
                "agent_goal": agent_goal,
                "agent_decision": "Historical features could not be built.",
                "summary": "Feature construction failed.",
                "historical_features": None,
                "reasoning_steps": [],
                "indicator_contributions": [],
            }

        latest = feature_df.iloc[-1]
        hcfg = self.config.get("historical_scoring", {})
        confidence_score = self._confidence_score(validation_result)

        return_1 = self._safe_float(latest.get("return_1"), 0.0) or 0.0
        return_5 = self._safe_float(latest.get("return_5"), 0.0) or 0.0
        return_20 = self._safe_float(latest.get("return_20"), 0.0) or 0.0
        ma_gap = self._safe_float(latest.get("ma_gap"), 0.0) or 0.0
        volatility_20 = self._safe_float(latest.get("volatility_20"), 0.0) or 0.0
        volume_change = self._safe_float(latest.get("volume_change"), 0.0) or 0.0
        rsi_14 = self._safe_float(latest.get("rsi_14"), 50.0) or 50.0

        score = 0.50
        contributions: List[Dict[str, Any]] = []
        reasons: List[str] = []

        if return_20 > hcfg.get("strong_return_20", 0.08):
            delta = 0.16
            momentum_level = "Strong Positive"
            message = "20-day momentum is strong."
        elif return_20 > hcfg.get("positive_return_20", 0.02):
            delta = 0.09
            momentum_level = "Positive"
            message = "20-day momentum is positive."
        elif return_20 < hcfg.get("weak_return_20", -0.05):
            delta = -0.16
            momentum_level = "Negative"
            message = "20-day momentum is weak."
        elif return_20 < hcfg.get("slightly_negative_return_20", -0.02):
            delta = -0.08
            momentum_level = "Slight Negative"
            message = "20-day momentum is slightly weak."
        else:
            delta = 0.0
            momentum_level = "Mixed"
            message = "20-day momentum is mixed."
        score += delta
        reasons.append(message)
        contributions.append(self._contribution("20-day return", self._format_pct(return_20), delta, message, "historical"))

        if return_5 > hcfg.get("positive_return_5", 0.02):
            delta = 0.07
            short_momentum = "Positive"
            message = "5-day momentum is positive."
        elif return_5 < hcfg.get("negative_return_5", -0.02):
            delta = -0.07
            short_momentum = "Negative"
            message = "5-day momentum is negative."
        else:
            delta = 0.0
            short_momentum = "Mixed"
            message = "5-day momentum is mixed."
        score += delta
        contributions.append(self._contribution("5-day return", self._format_pct(return_5), delta, message, "historical"))

        if ma_gap > hcfg.get("uptrend_ma_gap", 0.02):
            delta = 0.10
            historical_trend = "Uptrend"
            trend_direction = "Positive"
            message = "Short-term average is above the medium-term average."
        elif ma_gap < hcfg.get("downtrend_ma_gap", -0.02):
            delta = -0.10
            historical_trend = "Downtrend"
            trend_direction = "Negative"
            message = "Short-term average is below the medium-term average."
        else:
            delta = 0.0
            historical_trend = "Sideways"
            trend_direction = "Neutral"
            message = "Moving-average gap is small."
        score += delta
        contributions.append(self._contribution("Moving-average gap", self._format_pct(ma_gap), delta, message, "historical"))

        if volatility_20 > hcfg.get("high_volatility_20", 0.04):
            delta = -0.08
            historical_volatility_level = "High"
            volatility_risk_level = "High"
            message = "Recent volatility is high."
        elif volatility_20 > hcfg.get("medium_volatility_20", 0.02):
            delta = -0.02
            historical_volatility_level = "Medium"
            volatility_risk_level = "Medium"
            message = "Recent volatility is moderate."
        else:
            delta = 0.04
            historical_volatility_level = "Low"
            volatility_risk_level = "Low"
            message = "Recent volatility is low."
        score += delta
        contributions.append(self._contribution("20-day volatility", self._format_pct(volatility_20), delta, message, "risk"))

        entry_risk_points = 0
        if rsi_14 >= hcfg.get("strong_overbought_rsi", 78):
            delta = -0.08
            rsi_signal = "Strongly Overbought"
            entry_risk_points += 2
            message = "RSI is very high."
        elif rsi_14 >= hcfg.get("overbought_rsi", 70):
            delta = -0.04
            rsi_signal = "Overbought"
            entry_risk_points += 1
            message = "RSI is high."
        elif rsi_14 <= hcfg.get("oversold_rsi", 30):
            delta = 0.04
            rsi_signal = "Oversold"
            entry_risk_points += 1
            message = "RSI is low."
        else:
            delta = 0.0
            rsi_signal = "Neutral"
            message = "RSI is neutral."
        score += delta
        contributions.append(self._contribution("RSI 14", round(rsi_14, 2), delta, message, "risk"))

        # Distance to recent high is calculated directly from price records.
        distance_to_high = None
        try:
            price_df = pd.DataFrame(price_records)
            price_df.columns = [str(c).lower().strip().replace(" ", "_") for c in price_df.columns]
            price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
            close_tail = price_df["close"].dropna().tail(60)
            if len(close_tail) >= 5:
                recent_high = close_tail.max()
                last_close = close_tail.iloc[-1]
                distance_to_high = (recent_high - last_close) / recent_high if recent_high else None
        except Exception:
            distance_to_high = None

        if distance_to_high is not None and distance_to_high <= hcfg.get("near_high_distance", 0.03) and return_20 > 0:
            entry_risk_points += 1
            contributions.append(self._contribution("Distance to recent high", self._format_pct(distance_to_high), -0.02, "Price is close to its recent high.", "risk"))
        if return_20 > hcfg.get("stretched_return_20", 0.12):
            entry_risk_points += 1
            contributions.append(self._contribution("Stretch risk", self._format_pct(return_20), -0.02, "The recent move may be stretched.", "risk"))

        if volume_change > hcfg.get("high_volume_change", 0.2):
            delta = 0.05
            message = "Volume is above recent average."
        elif volume_change < hcfg.get("low_volume_change", -0.2):
            delta = -0.03
            message = "Volume is below recent average."
        else:
            delta = 0.0
            message = "Volume is close to normal."
        score += delta
        contributions.append(self._contribution("Volume change", self._format_pct(volume_change), delta, message, "historical"))

        raw_score = self._clip(score)
        historical_score = self._clip(raw_score * (0.90 + 0.10 * confidence_score))
        contributions.append(self._contribution("Validation confidence", round(confidence_score, 3), historical_score - raw_score, "Data confidence adjusted the historical score.", "data_quality"))

        if entry_risk_points >= 3:
            entry_risk_level = "High"
        elif entry_risk_points >= 2:
            entry_risk_level = "Elevated"
        elif entry_risk_points >= 1:
            entry_risk_level = "Moderate"
        else:
            entry_risk_level = "Low"

        if trend_direction == "Positive" and return_5 > 0:
            trend_direction = "Positive"
        elif trend_direction == "Neutral" and (return_5 > 0.01 or return_20 > 0.03):
            trend_direction = "Mild Positive"
        elif trend_direction == "Neutral" and (return_5 < -0.01 or return_20 < -0.03):
            trend_direction = "Mild Negative"

        thresholds = self.config.get("signal_thresholds", {})
        if historical_score >= thresholds.get("bullish_score", 0.70):
            historical_signal = "BULLISH_HISTORY"
            decision = "Historical trend is positive."
        elif historical_score <= thresholds.get("bearish_score", 0.35):
            historical_signal = "BEARISH_HISTORY"
            decision = "Historical trend is weak."
        elif historical_volatility_level == "High":
            historical_signal = "HIGH_VOLATILITY_HISTORY"
            decision = "Historical trend is mixed with high volatility."
        else:
            historical_signal = "NEUTRAL_HISTORY"
            decision = "Historical trend is mixed."

        historical_features = {
            "return_1": return_1,
            "return_5": return_5,
            "return_20": return_20,
            "ma_gap": ma_gap,
            "volatility_20": volatility_20,
            "volume_change": volume_change,
            "rsi_14": rsi_14,
            "validation_confidence_score": confidence_score,
        }

        return {
            "success": True,
            "stage": "Stage 2: Historical Analysis",
            "agent_goal": agent_goal,
            "symbol": symbol,
            "historical_source": historical_source,
            "latest_feature_date": str(latest.get("date", "Unknown")),
            "return_1": round(return_1, 6),
            "return_1_pct": self._format_pct(return_1),
            "return_5": round(return_5, 6),
            "return_5_pct": self._format_pct(return_5),
            "return_20": round(return_20, 6),
            "return_20_pct": self._format_pct(return_20),
            "ma_gap": round(ma_gap, 6),
            "ma_gap_pct": self._format_pct(ma_gap),
            "volatility_20": round(volatility_20, 6),
            "volatility_20_pct": self._format_pct(volatility_20),
            "volume_change": round(volume_change, 6),
            "volume_change_pct": self._format_pct(volume_change),
            "rsi_14": round(rsi_14, 3),
            "momentum_level": momentum_level,
            "short_momentum": short_momentum,
            "historical_trend": historical_trend,
            "trend_direction": trend_direction,
            "historical_volatility_level": historical_volatility_level,
            "volatility_risk_level": volatility_risk_level,
            "rsi_signal": rsi_signal,
            "entry_risk_level": entry_risk_level,
            "entry_risk_points": entry_risk_points,
            "distance_to_recent_high": round(distance_to_high, 6) if distance_to_high is not None else None,
            "historical_score": round(historical_score, 3),
            "historical_signal": historical_signal,
            "agent_decision": decision,
            "historical_features": historical_features,
            "indicator_contributions": self._top_contributions(contributions, 8),
            "reasoning_steps": ["Calculated momentum, moving-average gap, volatility, RSI, volume and entry timing risk."],
            "summary": f"{symbol}: {trend_direction.lower()} trend, {entry_risk_level.lower()} entry risk.",
        }

