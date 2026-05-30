from __future__ import annotations

import pandas as pd


def test_basic_formatting_helpers():
    from app_components.helpers import clean_label, clean_symbol, format_pct, format_price, get_nested, selected_price_from_quote

    assert clean_symbol(" aapl ") == "AAPL"
    assert clean_label("BUY_WATCHLIST_OVERBOUGHT") == "Bullish Watchlist / High Entry Risk"
    assert format_price(1234.5) == "$1,234.50"
    assert format_pct(0.1234) == "12.34%"
    assert get_nested({"a": {"b": 3}}, ["a", "b"]) == 3
    assert selected_price_from_quote({"finnhub": {"current_price": 10}}, {}) == 10.0


def test_dataframe_cleaning_handles_duplicate_and_multiindex_columns():
    from app_components.helpers import first_series, historical_to_dataframe, normalise_ohlcv_columns

    multi = pd.DataFrame(
        [["2024-01-01", 1, 2, 0.5, 1.5, 100], ["2024-01-02", 2, 3, 1.5, 2.5, 110]],
        columns=pd.MultiIndex.from_tuples([
            ("Date", "AAPL"),
            ("Open", "AAPL"),
            ("High", "AAPL"),
            ("Low", "AAPL"),
            ("Close", "AAPL"),
            ("Volume", "AAPL"),
        ]),
    )
    cleaned = normalise_ohlcv_columns(multi)
    assert {"timestamp", "open", "high", "low", "close", "volume"}.issubset(set(cleaned.columns))
    assert not cleaned.columns.duplicated().any()

    duplicate_close = pd.DataFrame([["2024-01-01", 10, 11, 9, 10.5, 10.6, 1000]], columns=["date", "open", "high", "low", "close", "close", "volume"])
    cleaned_duplicate = normalise_ohlcv_columns(duplicate_close)
    assert "close" in cleaned_duplicate.columns
    assert not cleaned_duplicate.columns.duplicated().any()
    assert first_series(cleaned_duplicate, "close").iloc[0] == 10.5

    hist = {"success": True, "prices": [{"date": "2024-01-01", "open": 10, "high": 11, "low": 9, "close": 10.5, "volume": 1000}]}
    df = historical_to_dataframe(hist)
    assert "close" in df.columns
    assert not df.columns.duplicated().any()


def test_context_builders():
    from app_components.contexts import build_event_context, build_portfolio_context

    portfolio = build_portfolio_context(True, shares=10, average_cost=100, current_price=120, user_intent="Research only")
    assert portfolio["has_position"] is True
    assert portfolio["market_value"] == 1200
    assert round(portfolio["unrealised_return"], 4) == 0.2

    no_position = build_portfolio_context(False, shares=10, average_cost=100, current_price=120, user_intent="Research only")
    assert no_position["shares"] == 0.0
    assert no_position["avg_cost"] is None

    event = build_event_context("not-a-date", "High")
    assert event["earnings_date"] == "not-a-date"
    assert event["days_to_earnings"] is None
    assert event["event_risk"] == "High"


def test_chart_preset_mapping():
    from app_components.charting import chart_preset

    assert chart_preset("1 Day") == ("1d", "5m")
    assert chart_preset("6 Months") == ("6mo", "1d")
    assert chart_preset("unknown") == ("1y", "1d")
