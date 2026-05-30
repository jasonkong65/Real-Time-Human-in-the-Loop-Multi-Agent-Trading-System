from datetime import date
from typing import Any, Dict, Optional

import pandas as pd

"""Context builders for the analyst agent, including portfolio context based on current position and market value, and event context based on upcoming earnings dates and associated risk levels. These contexts can be used to inform the analyst's view of a stock and adjust its analysis and recommendations accordingly."""

def build_portfolio_context(
    has_position: bool,
    shares: float,
    average_cost: float,
    current_price: Optional[float],
    user_intent: str,
) -> Dict[str, Any]:
    market_value = None
    unrealised_return = None
    if has_position and current_price and average_cost:
        try:
            market_value = float(shares) * float(current_price)
            unrealised_return = (float(current_price) - float(average_cost)) / float(average_cost)
        except Exception:
            pass
    return {
        "source": "streamlit_ui",
        "has_position": bool(has_position),
        "current_position": float(shares or 0.0) if has_position else 0.0,
        "shares": float(shares or 0.0) if has_position else 0.0,
        "avg_cost": float(average_cost or 0.0) if has_position else None,
        "market_value": market_value,
        "unrealised_return": unrealised_return,
        "user_intent": user_intent,
    }


def build_event_context(earnings_date_text: str, event_risk: str) -> Dict[str, Any]:
    text = str(earnings_date_text or "").strip()
    days_to_earnings = None
    if text:
        try:
            dt = pd.to_datetime(text).date()
            days_to_earnings = (dt - date.today()).days
        except Exception:
            days_to_earnings = None

    return {
        "source": "streamlit_ui",
        "earnings_date": text or None,
        "days_to_earnings": days_to_earnings,
        "event_risk": event_risk,
    }