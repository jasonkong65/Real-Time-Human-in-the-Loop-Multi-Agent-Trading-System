import json
from typing import Any, Dict, List, Optional

import pandas as pd

def safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    except Exception:
        return str(obj)


def clean_symbol(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def clean_label(value: Any, fallback: str = "Unknown") -> str:
    if value is None:
        return fallback
    text = str(value).strip()
    if not text:
        return fallback
    replacements = {
        "POSITIVE_BUT_ENTRY_RISK": "Positive + Entry Risk",
        "WATCHLIST_BULLISH_ENTRY_RISK": "Bullish Watchlist",
        "BUY_WATCHLIST_OVERBOUGHT": "Bullish Watchlist / High Entry Risk",
        "BUY_WATCHLIST_ENTRY_RISK": "Bullish Watchlist / Entry Risk",
        "WAIT_FOR_PULLBACK_OR_CONFIRMATION": "Wait for Pullback / Confirmation",
        "MONITOR_AND_RESEARCH": "Monitor + Research",
        "RISK_REDUCTION_REVIEW": "Risk Reduction Review",
        "RESEARCH_FOR_POSSIBLE_ENTRY": "Research for Paper Entry",
        "NO_ACTION_DATA_OR_RISK_BLOCK": "No Action / Risk Block",
        "BUY_CANDIDATE": "Research Candidate",
        "SELL_RISK": "Risk Review",
        "HOLD": "Hold / Monitor",
        "BLOCKED": "Blocked",
        "High": "High",
        "Medium": "Medium",
        "Low": "Low",
    }
    return replacements.get(text, text.replace("_", " ").title())


def format_price(value: Any) -> str:
    try:
        value = float(value)
        return f"${value:,.2f}"
    except Exception:
        return "N/A"


def format_pct(value: Any) -> str:
    try:
        value = float(value)
        return f"{value * 100:.2f}%"
    except Exception:
        return "N/A"


def get_nested(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur = data
    for key in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def normalise_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame with safe, unique OHLCV/timestamp columns.

    Some finance downloads/cache tables may return duplicate columns or
    yfinance-style MultiIndex columns. Duplicate labels make expressions like
    df["close"] return a DataFrame instead of a Series, which then causes
    pandas.to_numeric(...) to raise: "arg must be a list, tuple, 1-d array, or Series".
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()

    out = df.copy()

    def clean_col(col: Any) -> str:
        # Prefer the OHLCV/timestamp part when yfinance returns tuple/MultiIndex columns,
        # for example ("Close", "AAPL") -> "close".
        if isinstance(col, tuple):
            parts = [str(x).strip() for x in col if str(x).strip() and str(x).lower() != "nan"]
            canonical = {
                "date", "datetime", "timestamp", "index",
                "open", "high", "low", "close", "adj close", "adj_close", "adjclose", "volume",
            }
            for part in parts:
                cleaned = part.lower().replace(" ", "_")
                if cleaned in {c.replace(" ", "_") for c in canonical}:
                    return cleaned
            return "_".join(parts).lower().replace(" ", "_")
        return str(col).strip().lower().replace(" ", "_")

    out.columns = [clean_col(c) for c in out.columns]

    rename_map = {
        "datetime": "timestamp",
        "date": "timestamp",
        "index": "timestamp",
        "adjclose": "adj_close",
        "adj_close": "adj_close",
        "adj__close": "adj_close",
    }
    out = out.rename(columns={c: rename_map.get(c, c) for c in out.columns})

    # Handle flattened names such as close_aapl or aapl_close by mapping the first
    # matching column to the canonical OHLCV name when the canonical name is absent.
    canonical_cols = ["timestamp", "open", "high", "low", "close", "adj_close", "volume"]
    for target in canonical_cols:
        if target in out.columns:
            continue
        matches = [
            c for c in out.columns
            if c == target or c.startswith(target + "_") or c.endswith("_" + target)
        ]
        if matches:
            out = out.rename(columns={matches[0]: target})

    # After renaming, keep the first occurrence of duplicate columns. This prevents
    # df["close"] from returning a DataFrame.
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()].copy()

    return out


def first_series(df: pd.DataFrame, column: Any) -> pd.Series:
    """Safely return one column as a Series even if duplicate labels exist."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or column is None:
        return pd.Series(dtype="float64")
    try:
        data = df.loc[:, column]
    except Exception:
        try:
            data = df[column]
        except Exception:
            return pd.Series(dtype="float64")
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 0:
            return pd.Series(dtype="float64")
        data = data.iloc[:, 0]
    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    return data


def call_agent_method(agent: Any, method_names: List[str], *args, **kwargs) -> Any:
    errors = []
    for method_name in method_names:
        if not hasattr(agent, method_name):
            continue
        method = getattr(agent, method_name)
        try:
            return method(**kwargs)
        except TypeError as e1:
            errors.append(f"{method_name} kwargs: {e1}")
            try:
                return method(*args)
            except Exception as e2:
                errors.append(f"{method_name} positional: {e2}")
        except Exception as e:
            errors.append(f"{method_name}: {e}")
    raise RuntimeError(f"No working method for {agent.__class__.__name__}. Tried {method_names}. Errors: {errors}")


def selected_price_from_quote(multi_quote: Dict[str, Any], validation_result: Optional[Dict[str, Any]] = None) -> Optional[float]:
    validation_result = validation_result or {}
    candidates = [
        validation_result.get("selected_price"),
        get_nested(validation_result, ["validation_for_next_agent", "selected_price"]),
        get_nested(multi_quote, ["primary_source", "current_price"]),
        get_nested(multi_quote, ["primary_quote", "current_price"]),
        get_nested(multi_quote, ["finnhub", "current_price"]),
        get_nested(multi_quote, ["secondary_source", "current_price"]),
    ]
    for item in candidates:
        try:
            if item is not None and float(item) > 0:
                return float(item)
        except Exception:
            continue
    return None


def historical_to_dataframe(historical_data: Dict[str, Any]) -> pd.DataFrame:
    if not isinstance(historical_data, dict) or not historical_data.get("success"):
        return pd.DataFrame()

    records = (
        historical_data.get("prices")
        or historical_data.get("records")
        or historical_data.get("price_records")
        or []
    )

    if isinstance(records, pd.DataFrame):
        df = records.copy()
    else:
        df = pd.DataFrame(records)

    if df.empty:
        return df

    df = normalise_ohlcv_columns(df)
    if df.empty:
        return df

    if "timestamp" in df.columns:
        ts = pd.to_datetime(first_series(df, "timestamp"), errors="coerce")
        df = df.assign(timestamp=ts)
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.set_index("timestamp")

    # Ensure duplicate labels cannot break pd.to_numeric.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(first_series(df, col), errors="coerce")

    return df