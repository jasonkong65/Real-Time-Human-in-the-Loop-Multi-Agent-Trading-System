import json
import os
import random
import time
import sqlite3
from datetime import datetime, date, time as dt_time, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class DataAgent:
    """
    Data Agent

    Collects live market data from Finnhub and Alpha Vantage, standardises the
    fields, tracks market status, applies dynamic cache TTL, retries API calls
    with exponential backoff, records source time gaps, and can write quote
    snapshots directly into the StorageAgent / SQLite memory layer.

    Design goal:
    - Fast during Streamlit reruns
    - Safer against API rate limits
    - Transparent about stale or delayed data
    - Compatible with the existing app.py method names
    """

    def __init__(
        self,
        cache_path: str = "data/cache/live_quotes.json",
        cache_ttl_seconds: Optional[int] = None,
        max_retries: int = 3,
        backoff_base_seconds: float = 0.5,
        storage_enabled: bool = True,
        db_path: str = "data/trading_system.db",
    ):
        self.finnhub_api_key = (os.getenv("FINNHUB_API_KEY") or "").strip()
        self.alpha_vantage_api_key = (os.getenv("ALPHA_VANTAGE_API_KEY") or "").strip()

        self.finnhub_url = "https://finnhub.io/api/v1/quote"
        self.alpha_vantage_url = "https://www.alphavantage.co/query"

        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        # If None, DataAgent uses automatic TTL based on market status.
        # If an int is passed, it is treated as a fixed override for compatibility.
        self.cache_ttl_seconds = cache_ttl_seconds

        self.max_retries = max(1, int(max_retries))
        self.backoff_base_seconds = float(backoff_base_seconds)
        self.retry_status_codes = {408, 425, 429, 500, 502, 503, 504}

        self.storage_enabled = storage_enabled
        self.db_path = Path(db_path)
        self._storage_agent = None

        self.session = requests.Session()

    # ------------------------------------------------------------------
    # Time, market status, and cache helpers
    # ------------------------------------------------------------------
    def _now(self) -> float:
        return time.time()

    def _utc_now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _safe_int(self, value: Any, default: Optional[int] = None) -> Optional[int]:
        try:
            if value is None or value == "":
                return default
            return int(float(value))
        except Exception:
            return default

    def get_market_status(self) -> Dict[str, Any]:
        """
        Basic US market status using New York time.

        This intentionally avoids extra dependencies. It handles regular
        weekdays and standard pre/post-market windows, but it does not check the
        official NYSE holiday calendar. The output is still useful for cache TTL
        decisions and for warning users about stale quotes.
        """
        ny_tz = ZoneInfo("America/New_York")
        now_ny = datetime.now(ny_tz)
        current_time = now_ny.time()
        weekday = now_ny.weekday()  # Monday=0, Sunday=6

        regular_open = dt_time(9, 30)
        regular_close = dt_time(16, 0)
        premarket_open = dt_time(4, 0)
        postmarket_close = dt_time(20, 0)

        if weekday >= 5:
            status = "weekend_closed"
            is_open = False
            session = "closed"
        elif regular_open <= current_time < regular_close:
            status = "regular_market_open"
            is_open = True
            session = "regular"
        elif premarket_open <= current_time < regular_open:
            status = "premarket"
            is_open = False
            session = "premarket"
        elif regular_close <= current_time < postmarket_close:
            status = "postmarket"
            is_open = False
            session = "postmarket"
        else:
            status = "closed"
            is_open = False
            session = "closed"

        return {
            "market": "US equities",
            "timezone": "America/New_York",
            "local_time": now_ny.isoformat(),
            "status": status,
            "session": session,
            "is_regular_market_open": is_open,
            "regular_open": "09:30",
            "regular_close": "16:00",
            "premarket_open": "04:00",
            "postmarket_close": "20:00",
            "holiday_calendar_checked": False,
            "note": "Weekend and regular trading hours are checked. Exchange holidays are not checked in this lightweight version.",
        }

    def _dynamic_cache_ttl(self, source: str) -> int:
        """
        Adjust cache TTL based on market session and source type.

        Finnhub is treated as the faster live quote source, so it uses shorter
        TTL during regular market hours. Alpha Vantage Global Quote is often
        slower/delayed/rate-limited, so it uses a longer TTL.
        """
        if self.cache_ttl_seconds is not None:
            return int(self.cache_ttl_seconds)

        status = self.get_market_status().get("session")
        source_key = source.lower().strip()

        if source_key == "finnhub":
            if status == "regular":
                return 30
            if status in ["premarket", "postmarket"]:
                return 120
            return 900

        if source_key == "alpha_vantage":
            if status == "regular":
                return 300
            if status in ["premarket", "postmarket"]:
                return 600
            return 1800

        return 120

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

        cached_at = self._safe_float(item.get("cached_at"), 0.0) or 0.0
        ttl = self._dynamic_cache_ttl(source)
        age = self._now() - cached_at

        if age > ttl:
            return None

        payload = item.get("payload")
        if not isinstance(payload, dict):
            return None

        payload = dict(payload)
        payload["from_cache"] = True
        payload["cache_age_seconds"] = round(age, 2)
        payload["cache_ttl_seconds"] = ttl
        return payload

    def _set_cached(self, source: str, symbol: str, payload: Dict[str, Any]) -> None:
        cache = self._read_cache()
        cache[self._cache_key(source, symbol)] = {
            "cached_at": self._now(),
            "ttl_seconds": self._dynamic_cache_ttl(source),
            "payload": payload,
        }
        self._write_cache(cache)

    # ------------------------------------------------------------------
    # API request helper with stronger retry/backoff
    # ------------------------------------------------------------------
    def _request_json(self, url: str, params: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
        last_error = None
        last_status = None
        last_raw = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=timeout)
                last_status = response.status_code
                last_raw = response.text[:500]

                if response.status_code == 200:
                    try:
                        data = response.json()
                    except ValueError:
                        return {
                            "success": False,
                            "error": "Response was not valid JSON.",
                            "attempts": attempt,
                            "raw_response": last_raw,
                        }
                    return {
                        "success": True,
                        "data": data,
                        "attempts": attempt,
                    }

                # Retry only transient/rate-limit/server errors.
                if response.status_code not in self.retry_status_codes:
                    return {
                        "success": False,
                        "error": f"HTTP {response.status_code}",
                        "attempts": attempt,
                        "raw_response": last_raw,
                    }

                last_error = f"HTTP {response.status_code}"

            except requests.Timeout:
                last_error = "Request timed out."
            except requests.RequestException as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = str(exc)

            if attempt < self.max_retries:
                # Exponential backoff with small jitter.
                sleep_seconds = self.backoff_base_seconds * (2 ** (attempt - 1))
                sleep_seconds += random.uniform(0, 0.25)
                time.sleep(sleep_seconds)

        return {
            "success": False,
            "error": last_error or f"Request failed after {self.max_retries} attempts.",
            "http_status": last_status,
            "attempts": self.max_retries,
            "raw_response": last_raw,
        }

    # ------------------------------------------------------------------
    # Data timing helpers
    # ------------------------------------------------------------------
    def _timestamp_to_iso(self, timestamp: Any) -> Optional[str]:
        ts = self._safe_int(timestamp)
        if ts is None or ts <= 0:
            return None
        try:
            return datetime.fromtimestamp(ts, timezone.utc).isoformat()
        except Exception:
            return None

    def _timestamp_age_seconds(self, timestamp: Any) -> Optional[float]:
        ts = self._safe_float(timestamp)
        if ts is None or ts <= 0:
            return None
        return round(self._now() - ts, 2)

    def _parse_alpha_date(self, latest_trading_day: Any) -> Optional[date]:
        if not latest_trading_day:
            return None
        try:
            return datetime.strptime(str(latest_trading_day), "%Y-%m-%d").date()
        except Exception:
            return None

    def _finnhub_trading_date_ny(self, timestamp: Any) -> Optional[date]:
        ts = self._safe_int(timestamp)
        if ts is None or ts <= 0:
            return None
        try:
            ny_tz = ZoneInfo("America/New_York")
            return datetime.fromtimestamp(ts, timezone.utc).astimezone(ny_tz).date()
        except Exception:
            return None

    def _source_time_gap(self, finnhub: Dict[str, Any], alpha_vantage: Dict[str, Any]) -> Dict[str, Any]:
        """
        Measure time/date mismatch between Finnhub and Alpha Vantage.

        Finnhub provides an epoch timestamp. Alpha Vantage Global Quote usually
        provides a latest trading day. We compare their New York trading dates
        rather than pretending both are equally real-time.
        """
        finnhub_date = None
        alpha_date = None

        if finnhub.get("success"):
            finnhub_date = self._finnhub_trading_date_ny(finnhub.get("timestamp"))

        if alpha_vantage.get("success"):
            alpha_date = self._parse_alpha_date(alpha_vantage.get("latest_trading_day"))

        gap_days = None
        warning = None

        if finnhub_date and alpha_date:
            gap_days = abs((finnhub_date - alpha_date).days)
            if gap_days >= 2:
                warning = (
                    f"Large source date gap detected: Finnhub trading date {finnhub_date} "
                    f"vs Alpha Vantage latest trading day {alpha_date}. Treat validation with caution."
                )
            elif gap_days == 1:
                warning = (
                    f"One-day source date gap detected: Finnhub trading date {finnhub_date} "
                    f"vs Alpha Vantage latest trading day {alpha_date}. This may happen outside market hours."
                )

        return {
            "finnhub_timestamp_utc": self._timestamp_to_iso(finnhub.get("timestamp")),
            "finnhub_age_seconds": self._timestamp_age_seconds(finnhub.get("timestamp")),
            "finnhub_trading_date_ny": str(finnhub_date) if finnhub_date else None,
            "alpha_vantage_latest_trading_day": str(alpha_date) if alpha_date else None,
            "source_trading_date_gap_days": gap_days,
            "warning": warning,
        }

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------
    def _get_storage_agent(self):
        if not self.storage_enabled:
            return None

        if self._storage_agent is not None:
            return self._storage_agent

        try:
            from agents.storage_agent import StorageAgent
            self._storage_agent = StorageAgent()
            return self._storage_agent
        except Exception:
            self._storage_agent = None
            return None

    def _record_market_quotes_to_storage(self, symbol: str, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        """
        Best-effort immediate storage. This does not break the pipeline if the
        StorageAgent schema/method signature differs.
        """
        if not self.storage_enabled:
            return {"success": False, "skipped": True, "reason": "storage_disabled"}

        storage = self._get_storage_agent()
        if storage is not None:
            method = getattr(storage, "record_market_quotes", None)
            if callable(method):
                call_attempts = [
                    lambda: method(symbol=symbol, multi_quote=multi_quote),
                    lambda: method(multi_quote=multi_quote),
                    lambda: method(symbol, multi_quote),
                    lambda: method(None, symbol, multi_quote),
                ]
                for call in call_attempts:
                    try:
                        result = call()
                        return result if isinstance(result, dict) else {"success": True, "result": result}
                    except TypeError:
                        continue
                    except Exception as exc:
                        return {"success": False, "error": str(exc)}

        # Fallback: write directly to a minimal market_quotes table if StorageAgent
        # is missing or does not expose record_market_quotes.
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS market_quotes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT,
                        source TEXT,
                        current_price REAL,
                        open_price REAL,
                        high_price REAL,
                        low_price REAL,
                        previous_close REAL,
                        source_timestamp TEXT,
                        latest_trading_day TEXT,
                        from_cache INTEGER,
                        raw_json TEXT,
                        created_at_utc TEXT
                    )
                    """
                )

                for key in ["finnhub", "alpha_vantage"]:
                    quote = multi_quote.get(key, {})
                    if not isinstance(quote, dict) or not quote.get("success"):
                        continue

                    conn.execute(
                        """
                        INSERT INTO market_quotes (
                            symbol, source, current_price, open_price, high_price,
                            low_price, previous_close, source_timestamp,
                            latest_trading_day, from_cache, raw_json, created_at_utc
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            symbol,
                            quote.get("source"),
                            quote.get("current_price"),
                            quote.get("open_price"),
                            quote.get("high_price"),
                            quote.get("low_price"),
                            quote.get("previous_close"),
                            quote.get("timestamp_iso") or quote.get("timestamp"),
                            quote.get("latest_trading_day"),
                            1 if quote.get("from_cache") else 0,
                            json.dumps(quote, ensure_ascii=False, default=str),
                            self._utc_now_iso(),
                        ),
                    )
                conn.commit()

            return {"success": True, "method": "direct_sqlite_fallback", "db_path": str(self.db_path)}
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Source-specific collectors
    # ------------------------------------------------------------------
    def get_finnhub_quote(self, symbol: str) -> Dict[str, Any]:
        symbol = str(symbol or "").upper().strip()
        if not symbol:
            return {"source": "Finnhub", "success": False, "symbol": symbol, "error": "Symbol is empty."}

        cached = self._get_cached("finnhub", symbol)
        if cached:
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
                "http_status": result.get("http_status"),
                "attempts": result.get("attempts"),
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
                "attempts": result.get("attempts"),
                "raw_response": data,
            }

        timestamp = data.get("t")
        payload = {
            "source": "Finnhub",
            "success": True,
            "symbol": symbol,
            "current_price": current_price,
            "open_price": self._safe_float(data.get("o")),
            "high_price": self._safe_float(data.get("h")),
            "low_price": self._safe_float(data.get("l")),
            "previous_close": previous_close,
            "timestamp": timestamp,
            "timestamp_iso": self._timestamp_to_iso(timestamp),
            "data_age_seconds": self._timestamp_age_seconds(timestamp),
            "attempts": result.get("attempts"),
            "raw_response": data,
            "from_cache": False,
            "cache_ttl_seconds": self._dynamic_cache_ttl("finnhub"),
        }
        self._set_cached("finnhub", symbol, payload)
        return payload

    def get_alpha_vantage_quote(self, symbol: str) -> Dict[str, Any]:
        symbol = str(symbol or "").upper().strip()
        if not symbol:
            return {"source": "Alpha Vantage", "success": False, "symbol": symbol, "error": "Symbol is empty."}

        cached = self._get_cached("alpha_vantage", symbol)
        if cached:
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
                "http_status": result.get("http_status"),
                "attempts": result.get("attempts"),
                "raw_response": result.get("raw_response"),
            }

        data = result.get("data", {})

        # Alpha Vantage rate limit messages may be returned as JSON with Note/Information.
        if isinstance(data, dict) and (data.get("Note") or data.get("Information")):
            return {
                "source": "Alpha Vantage",
                "success": False,
                "symbol": symbol,
                "error": data.get("Note") or data.get("Information"),
                "attempts": result.get("attempts"),
                "raw_response": data,
            }

        quote = data.get("Global Quote") or data.get("globalQuote") or {}
        current_price = self._safe_float(quote.get("05. price"))

        if not current_price or current_price <= 0:
            return {
                "source": "Alpha Vantage",
                "success": False,
                "symbol": symbol,
                "error": "Alpha Vantage returned no usable quote.",
                "attempts": result.get("attempts"),
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
            "attempts": result.get("attempts"),
            "raw_response": data,
            "from_cache": False,
            "cache_ttl_seconds": self._dynamic_cache_ttl("alpha_vantage"),
        }
        self._set_cached("alpha_vantage", symbol, payload)
        return payload

    # ------------------------------------------------------------------
    # Main app-compatible method
    # ------------------------------------------------------------------
    def get_multi_source_quote(self, symbol: str) -> Dict[str, Any]:
        symbol = str(symbol or "").upper().strip()
        market_status = self.get_market_status()

        finnhub = self.get_finnhub_quote(symbol)
        alpha_vantage = self.get_alpha_vantage_quote(symbol)
        time_gap = self._source_time_gap(finnhub, alpha_vantage)

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

        warnings = []
        if time_gap.get("warning"):
            warnings.append(time_gap["warning"])
        if not market_status.get("is_regular_market_open"):
            warnings.append(
                f"US regular market is not open now ({market_status.get('status')}). Quotes may be delayed or from the last session."
            )
        if finnhub.get("success") and finnhub.get("data_age_seconds") is not None:
            if finnhub.get("data_age_seconds") > 900 and market_status.get("is_regular_market_open"):
                warnings.append("Finnhub quote timestamp is older than 15 minutes during regular market hours.")

        result = {
            "success": bool(available_sources),
            "agent": "Data Agent",
            "agent_goal": "Collect live quote data, track source freshness, and standardise data for downstream agents.",
            "symbol": symbol,
            "market_status": market_status,
            "available_sources": available_sources,
            "selected_price": selected_price,
            "selected_source": selected_source,
            "source_time_gap": time_gap,
            "warnings": warnings,
            "cache_policy": {
                "mode": "dynamic" if self.cache_ttl_seconds is None else "fixed",
                "finnhub_ttl_seconds": self._dynamic_cache_ttl("finnhub"),
                "alpha_vantage_ttl_seconds": self._dynamic_cache_ttl("alpha_vantage"),
            },
            "finnhub": finnhub,
            "alpha_vantage": alpha_vantage,
            "summary": (
                f"Collected market data for {symbol} from {', '.join(available_sources)}."
                if available_sources else f"No live market data source was available for {symbol}."
            ),
        }

        storage_result = self._record_market_quotes_to_storage(symbol, result)
        result["immediate_storage_result"] = storage_result

        return result

    # Backward-compatible aliases used by app.py fallback method list
    def get_multi_source_quotes(self, symbol: str) -> Dict[str, Any]:
        return self.get_multi_source_quote(symbol)

    def get_market_data(self, symbol: str) -> Dict[str, Any]:
        return self.get_multi_source_quote(symbol)

    def fetch_market_data(self, symbol: str) -> Dict[str, Any]:
        return self.get_multi_source_quote(symbol)

    def collect_market_data(self, symbol: str) -> Dict[str, Any]:
        return self.get_multi_source_quote(symbol)

    def get_live_quote(self, symbol: str) -> Dict[str, Any]:
        return self.get_multi_source_quote(symbol)

    def run(self, symbol: str) -> Dict[str, Any]:
        return self.get_multi_source_quote(symbol)
