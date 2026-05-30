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


class AnalystQuoteMixin:


    def analyse_quote_level(self, multi_quote: dict, validation_result: dict) -> dict:
        agent_goal = "Read the latest quote and score short-term price action."

        if validation_result.get("next_action") == "BLOCK_ANALYSIS":
            return {
                "success": False,
                "stage": "Stage 1: Quote-level Analysis",
                "agent_goal": agent_goal,
                "agent_decision": "Data quality is too weak for quote analysis.",
                "summary": "Quote analysis was skipped because validation blocked it.",
                "quote_features": None,
                "reasoning_steps": [],
                "indicator_contributions": [],
            }

        symbol = str(multi_quote.get("symbol") or validation_result.get("symbol") or "UNKNOWN").upper()
        primary = self._source(multi_quote, ["finnhub", "primary", "finnhub_quote"])
        secondary = self._source(multi_quote, ["alpha_vantage", "secondary", "alpha_vantage_quote"])
        quote = primary if primary.get("success") else secondary

        selected_price = self._safe_float(validation_result.get("selected_price"))
        current = self._price(quote, "current_price") or selected_price
        previous_close = self._price(quote, "previous_close")
        open_price = self._price(quote, "open_price")
        high_price = self._price(quote, "high_price")
        low_price = self._price(quote, "low_price")

        if current is None or previous_close is None or previous_close <= 0:
            return {
                "success": False,
                "stage": "Stage 1: Quote-level Analysis",
                "agent_goal": agent_goal,
                "symbol": symbol,
                "agent_decision": "Key quote fields are missing.",
                "summary": "Quote analysis could not run because current price or previous close is missing.",
                "quote_features": None,
                "reasoning_steps": [],
                "indicator_contributions": [],
            }

        quote_cfg = self.config.get("quote_scoring", {})
        confidence_score = self._confidence_score(validation_result)
        daily_return = (current - previous_close) / previous_close
        open_return = (current - open_price) / open_price if open_price and open_price > 0 else None
        intraday_range = (high_price - low_price) / current if current and high_price and low_price and high_price >= low_price else None

        score = 0.50
        contributions: List[Dict[str, Any]] = []
        reasons: List[str] = []

        if daily_return > quote_cfg.get("strong_up_return", 0.02):
            delta = 0.20
            message = "Strong day so far."
            quote_trend = "Strong upward"
        elif daily_return > quote_cfg.get("mild_up_return", 0.005):
            delta = 0.10
            message = "Price is modestly higher."
            quote_trend = "Slight upward"
        elif daily_return < quote_cfg.get("strong_down_return", -0.02):
            delta = -0.20
            message = "Strong down day."
            quote_trend = "Strong downward"
        elif daily_return < quote_cfg.get("mild_down_return", -0.005):
            delta = -0.10
            message = "Price is modestly lower."
            quote_trend = "Slight downward"
        else:
            delta = 0.0
            message = "Price is close to the previous close."
            quote_trend = "Neutral"
        score += delta
        reasons.append(message)
        contributions.append(self._contribution("Daily return", self._format_pct(daily_return), delta, message, "quote"))

        if open_return is not None:
            if open_return > quote_cfg.get("open_up_return", 0.01):
                delta = 0.10
                message = "Price is above the open."
            elif open_return < quote_cfg.get("open_down_return", -0.01):
                delta = -0.10
                message = "Price is below the open."
            else:
                delta = 0.0
                message = "Open-to-current move is small."
            score += delta
            reasons.append(message)
            contributions.append(self._contribution("Open-to-current return", self._format_pct(open_return), delta, message, "quote"))

        if intraday_range is None:
            quote_volatility_level = "Unknown"
            delta = 0.0
            message = "Intraday range is unavailable."
        elif intraday_range > quote_cfg.get("high_intraday_range", 0.04):
            quote_volatility_level = "High"
            delta = -0.10
            message = "Intraday movement is wide."
        elif intraday_range > quote_cfg.get("medium_intraday_range", 0.02):
            quote_volatility_level = "Medium"
            delta = -0.03
            message = "Intraday movement is moderate."
        else:
            quote_volatility_level = "Low"
            delta = 0.05
            message = "Intraday movement is calm."
        score += delta
        reasons.append(message)
        contributions.append(self._contribution("Intraday range", self._format_pct(intraday_range), delta, message, "quote"))

        raw_score = self._clip(score)
        adjusted_score = self._clip(raw_score * (0.85 + 0.15 * confidence_score))
        contributions.append(self._contribution("Validation confidence", round(confidence_score, 3), adjusted_score - raw_score, "Data confidence adjusted the quote score.", "data_quality"))

        if adjusted_score >= 0.70:
            quote_signal = "QUOTE_BULLISH"
            decision = "Live quote is supportive."
        elif adjusted_score <= 0.35:
            quote_signal = "QUOTE_BEARISH"
            decision = "Live quote is weak."
        elif quote_volatility_level == "High":
            quote_signal = "QUOTE_HIGH_VOLATILITY"
            decision = "Live quote is volatile."
        else:
            quote_signal = "QUOTE_NEUTRAL"
            decision = "Live quote is mixed."

        quote_features = {
            "daily_return": daily_return,
            "open_to_current_return": open_return,
            "intraday_range_pct": intraday_range,
            "quote_score": adjusted_score,
            "quote_trend": quote_trend,
            "quote_volatility_level": quote_volatility_level,
            "quote_signal": quote_signal,
        }

        return {
            "success": True,
            "stage": "Stage 1: Quote-level Analysis",
            "agent_goal": agent_goal,
            "symbol": symbol,
            "selected_price": current,
            "daily_return": round(daily_return, 6),
            "daily_return_pct": self._format_pct(daily_return),
            "open_to_current_return": round(open_return, 6) if open_return is not None else None,
            "open_to_current_return_pct": self._format_pct(open_return),
            "intraday_range_pct": round(intraday_range, 6) if intraday_range is not None else None,
            "intraday_range_pct_text": self._format_pct(intraday_range),
            "quote_trend": quote_trend,
            "quote_volatility_level": quote_volatility_level,
            "quote_score": round(adjusted_score, 3),
            "quote_signal": quote_signal,
            "agent_decision": decision,
            "reasoning_steps": ["Checked live price move, open-to-current move, intraday range, and data confidence."],
            "indicator_contributions": self._top_contributions(contributions, 5),
            "quote_features": quote_features,
            "summary": f"{symbol}: live quote looks {quote_signal.replace('QUOTE_', '').lower()}.",
        }

