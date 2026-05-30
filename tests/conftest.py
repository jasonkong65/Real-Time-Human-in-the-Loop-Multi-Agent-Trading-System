"""Shared pytest setup for lightweight project tests.

The tests avoid real market/API calls. Optional UI/market packages are stubbed
when they are not installed in the execution environment, so the test suite can
run both on GitHub Actions and in a minimal local Python environment.
"""
from __future__ import annotations

import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_streamlit_stub() -> None:
    if "streamlit" in sys.modules:
        return
    try:
        __import__("streamlit")
        return
    except Exception:
        pass

    st = types.ModuleType("streamlit")
    st.session_state = {}

    def _identity_decorator(*dargs, **dkwargs):
        def decorator(func):
            return func
        if dargs and callable(dargs[0]) and len(dargs) == 1 and not dkwargs:
            return dargs[0]
        return decorator

    class _Context:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False

    def _columns(spec):
        count = len(spec) if isinstance(spec, (list, tuple)) else int(spec)
        return [_Context() for _ in range(count)]

    def _selectbox(label, options, key=None, index=0, **kwargs):
        value = options[index] if options else None
        if key is not None:
            st.session_state[key] = value
        return value

    def _button(*args, **kwargs):
        return False

    def _text_input(*args, value="", **kwargs):
        return value

    def _text_area(*args, value="", **kwargs):
        return value

    def _checkbox(*args, value=False, **kwargs):
        return value

    def _number_input(*args, value=0.0, **kwargs):
        return value

    def _slider(*args, value=None, min_value=None, **kwargs):
        return value if value is not None else min_value

    def _no_op(*args, **kwargs):
        return None

    st.cache_data = _identity_decorator
    st.cache_resource = _identity_decorator
    st.spinner = lambda *args, **kwargs: _Context()
    st.expander = lambda *args, **kwargs: _Context()
    st.columns = _columns
    st.selectbox = _selectbox
    st.button = _button
    st.text_input = _text_input
    st.text_area = _text_area
    st.checkbox = _checkbox
    st.number_input = _number_input
    st.slider = _slider
    st.markdown = _no_op
    st.caption = _no_op
    st.info = _no_op
    st.warning = _no_op
    st.error = _no_op
    st.success = _no_op
    st.json = _no_op
    st.code = _no_op
    st.dataframe = _no_op
    st.line_chart = _no_op
    st.bar_chart = _no_op
    st.set_page_config = _no_op
    st.stop = lambda: (_ for _ in ()).throw(SystemExit())
    st.tabs = lambda names: [_Context() for _ in names]
    st.sidebar = _Context()
    sys.modules["streamlit"] = st


def _install_yfinance_stub() -> None:
    if "yfinance" in sys.modules:
        return
    try:
        __import__("yfinance")
        return
    except Exception:
        pass

    yf = types.ModuleType("yfinance")

    def _fake_download(symbol, period="1y", interval="1d", progress=False, auto_adjust=False, **kwargs):
        dates = pd.date_range("2024-01-01", periods=90, freq="D")
        base = pd.Series(range(90), dtype="float") + 100.0
        return pd.DataFrame(
            {
                "Open": base,
                "High": base + 1.0,
                "Low": base - 1.0,
                "Close": base + 0.5,
                "Adj Close": base + 0.5,
                "Volume": [1_000_000 + i for i in range(90)],
            },
            index=dates,
        )

    class _FastInfo:
        market_cap = 1_000_000_000

    class _Ticker:
        def __init__(self, symbol):
            self.symbol = symbol
            self.fast_info = _FastInfo()
            self.info = {"sector": "Technology", "marketCap": 1_000_000_000, "averageVolume": 1_000_000}

        def history(self, period="1y", interval="1d", **kwargs):
            return _fake_download(self.symbol, period=period, interval=interval)

    yf.download = _fake_download
    yf.Ticker = _Ticker
    sys.modules["yfinance"] = yf


def _install_groq_stub() -> None:
    if "groq" in sys.modules:
        return
    try:
        __import__("groq")
        return
    except Exception:
        pass

    groq = types.ModuleType("groq")

    class Groq:  # pragma: no cover - only used when package is absent
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Groq client is not available in tests")

    groq.Groq = Groq
    sys.modules["groq"] = groq


_install_streamlit_stub()
_install_yfinance_stub()
_install_groq_stub()


@pytest.fixture
def sample_prices():
    # A small but non-monotonic OHLCV series. The changing direction creates
    # multiple label classes for TrainingAgent tests while remaining deterministic.
    import math

    dates = pd.date_range("2024-01-01", periods=140, freq="D")
    rows = []
    for i, d in enumerate(dates):
        close = 100.0 + 8.0 * math.sin(i / 4.0) + 0.04 * i
        open_price = close - 0.2
        rows.append(
            {
                "date": d.strftime("%Y-%m-%d"),
                "open": round(open_price, 4),
                "high": round(close + 1.0, 4),
                "low": round(close - 1.0, 4),
                "close": round(close, 4),
                "adj_close": round(close, 4),
                "volume": 1_000_000 + i * 1000,
            }
        )
    return rows


@pytest.fixture
def sample_historical_data(sample_prices):
    return {"success": True, "symbol": "AAPL", "prices": sample_prices, "source": "test_fixture"}


@pytest.fixture
def sample_multi_quote():
    return {
        "success": True,
        "symbol": "AAPL",
        "finnhub": {
            "success": True,
            "source": "Finnhub",
            "current_price": 150.0,
            "open_price": 148.0,
            "high_price": 151.0,
            "low_price": 147.0,
            "previous_close": 149.0,
            "timestamp": 1_700_000_000,
        },
        "alpha_vantage": {
            "success": True,
            "source": "Alpha Vantage",
            "current_price": 149.5,
            "previous_close": 149.0,
            "latest_trading_day": "2024-01-02",
        },
    }
