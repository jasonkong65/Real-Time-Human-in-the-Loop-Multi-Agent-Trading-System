import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class DataAgent:
    """
    Data Agent

    Collects live market data from Finnhub and Alpha Vantage, standardises the
    fields, and keeps a short local cache so repeated Streamlit runs do not hit
    API limits too quickly.
    """

    def __init__(self, cache_path: str = "data/cache/live_quotes.json", cache_ttl_seconds: int = 60):
        self.finnhub_api_key = (os.getenv("FINNHUB_API_KEY") or "").strip()
        self.alpha_vantage_api_key = (os.getenv("ALPHA_VANTAGE_API_KEY") or "").strip()
        self.finnhub_url = "https://finnhub.io/api/v1/quote"
        self.alpha_vantage_url = "https://www.alphavantage.co/query"
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_ttl_seconds = cache_ttl_seconds
        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Small utility helpers
    # ------------------------------------------------------------------
    def _now(self) -> float:
        return time.time()

    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _read_cache(self) -> Dict[str, Any]:
        if not self.cache_path.exists() or self.cache_path.stat().st_size == 0:
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _write_cache(self, cache: Dict[str, Any]) -> None:
        try:
            with self.cache_path.open("w", encoding="utf-8") as f:
                json.dump(cache, f, indent=2)
        except Exception:
            pass

    def _cache_key(self, source: str, symbol: str) -> str:
        return f"{source}:{symbol.upper().strip()}"

    def _get_cached(self, source: str, symbol: str) -> Optional[Dict[str, Any]]:
        cache = self._read_cache()
        key = self._cache_key(source, symbol)
        item = cache.get(key)
        if not isinstance(item, dict):
            return None
        timestamp = self._safe_float(item.get("cached_at"), 0.0) or 0.0
        if self._now() - timestamp > self.cache_ttl_seconds:
            return None
        payload = item.get("payload")
        return payload if isinstance(payload, dict) else None

    def _set_cached(self, source: str, symbol: str, payload: Dict[str, Any]) -> None:
        cache = self._read_cache()
        cache[self._cache_key(source, symbol)] = {
            "cached_at": self._now(),
            "payload": payload,
        }
        self._write_cache(cache)

    def _request_json(self, url: str, params: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
        last_error = None
        for attempt in range(2):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                if response.status_code != 200:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}",
                        "raw_response": response.text[:300],
                    }
                try:
                    data = response.json()
                except ValueError:
                    return {"success": False, "error": "Response was not valid JSON."}
                return {"success": True, "data": data}
            except Exception as exc:
                last_error = str(exc)
                time.sleep(0.25 * (attempt + 1))
        return {"success": False, "error": last_error or "Request failed."}

    # ------------------------------------------------------------------
    # Source-specific collectors
    # ------------------------------------------------------------------
    def get_finnhub_quote(self, symbol: str) -> Dict[str, Any]:
        symbol = str(symbol or "").upper().strip()
        if not symbol:
            return {"source": "Finnhub", "success": False, "symbol": symbol, "error": "Symbol is empty."}

        cached = self._get_cached("finnhub", symbol)
        if cached:
            cached = dict(cached)
            cached["from_cache"] = True
            return cached

        if not self.finnhub_api_key:
            return {
                "source": "Finnhub",
                "success": False,
                "symbol": symbol,
                "error": "FINNHUB_API_KEY is not configured.",
            }

        result = self._request_json(
            self.finnhub_url,
            params={"symbol": symbol, "token": self.finnhub_api_key},
        )
        if not result.get("success"):
            return {
                "source": "Finnhub",
                "success": False,
                "symbol": symbol,
                "error": result.get("error"),
                "raw_response": result.get("raw_response"),
            }

        data = result.get("data", {})
        current_price = self._safe_float(data.get("c"))
        previous_close = self._safe_float(data.get("pc"))

        if not current_price or current_price <= 0:
            return {
                "source": "Finnhub",
                "success": False,
                "symbol": symbol,
                "error": "Finnhub returned no usable live price.",
                "raw_response": data,
            }

        payload = {
            "source": "Finnhub",
            "success": True,
            "symbol": symbol,
            "current_price": current_price,
            "open_price": self._safe_float(data.get("o")),
            "high_price": self._safe_float(data.get("h")),
            "low_price": self._safe_float(data.get("l")),
            "previous_close": previous_close,
            "timestamp": data.get("t"),
            "raw_response": data,
            "from_cache": False,
        }
        self._set_cached("finnhub", symbol, payload)
        return payload

    def get_alpha_vantage_quote(self, symbol: str) -> Dict[str, Any]:
        symbol = str(symbol or "").upper().strip()
        if not symbol:
            return {"source": "Alpha Vantage", "success": False, "symbol": symbol, "error": "Symbol is empty."}

        cached = self._get_cached("alpha_vantage", symbol)
        if cached:
            cached = dict(cached)
            cached["from_cache"] = True
            return cached

        if not self.alpha_vantage_api_key:
            return {
                "source": "Alpha Vantage",
                "success": False,
                "symbol": symbol,
                "error": "ALPHA_VANTAGE_API_KEY is not configured.",
            }

        result = self._request_json(
            self.alpha_vantage_url,
            params={
                "function": "GLOBAL_QUOTE",
                "symbol": symbol,
                "apikey": self.alpha_vantage_api_key,
            },
        )
        if not result.get("success"):
            return {
                "source": "Alpha Vantage",
                "success": False,
                "symbol": symbol,
                "error": result.get("error"),
                "raw_response": result.get("raw_response"),
            }

        data = result.get("data", {})
        quote = data.get("Global Quote") or data.get("globalQuote") or {}
        current_price = self._safe_float(quote.get("05. price"))

        if not current_price or current_price <= 0:
            note = "Alpha Vantage returned no usable quote."
            if isinstance(data, dict) and data.get("Note"):
                note = data.get("Note")
            return {
                "source": "Alpha Vantage",
                "success": False,
                "symbol": symbol,
                "error": note,
                "raw_response": data,
            }

        payload = {
            "source": "Alpha Vantage",
            "success": True,
            "symbol": symbol,
            "current_price": current_price,
            "open_price": self._safe_float(quote.get("02. open")),
            "high_price": self._safe_float(quote.get("03. high")),
            "low_price": self._safe_float(quote.get("04. low")),
            "previous_close": self._safe_float(quote.get("08. previous close")),
            "latest_trading_day": quote.get("07. latest trading day"),
            "raw_response": data,
            "from_cache": False,
        }
        self._set_cached("alpha_vantage", symbol, payload)
        return payload

    # ------------------------------------------------------------------
    # Main app-compatible method
    # ------------------------------------------------------------------
    def get_multi_source_quote(self, symbol: str) -> Dict[str, Any]:
        symbol = str(symbol or "").upper().strip()
        finnhub = self.get_finnhub_quote(symbol)
        alpha_vantage = self.get_alpha_vantage_quote(symbol)

        available_sources = []
        if finnhub.get("success"):
            available_sources.append("Finnhub")
        if alpha_vantage.get("success"):
            available_sources.append("Alpha Vantage")

        primary_price = finnhub.get("current_price") if finnhub.get("success") else None
        secondary_price = alpha_vantage.get("current_price") if alpha_vantage.get("success") else None

        if primary_price is not None:
            selected_price = primary_price
            selected_source = "Finnhub"
        elif secondary_price is not None:
            selected_price = secondary_price
            selected_source = "Alpha Vantage"
        else:
            selected_price = None
            selected_source = None

        return {
            "success": bool(available_sources),
            "agent": "Data Agent",
            "agent_goal": "Collect live quote data and standardise it for downstream agents.",
            "symbol": symbol,
            "available_sources": available_sources,
            "selected_price": selected_price,
            "selected_source": selected_source,
            "finnhub": finnhub,
            "alpha_vantage": alpha_vantage,
            "summary": (
                f"Collected market data for {symbol} from {', '.join(available_sources)}."
                if available_sources else f"No live market data source was available for {symbol}."
            ),
        }

    # Backward-compatible aliases used by app.py fallback method list
    def get_multi_source_quotes(self, symbol: str) -> Dict[str, Any]:
        return self.get_multi_source_quote(symbol)

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        return self.get_multi_source_quote(symbol)

    def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        return self.get_multi_source_quote(symbol)

    def collect_market_data(self, symbol: str) -> Dict[str, Any]:
        return self.get_multi_source_quote(symbol)

    def run(self, symbol: str) -> Dict[str, Any]:
        return self.get_multi_source_quote(symbol)
