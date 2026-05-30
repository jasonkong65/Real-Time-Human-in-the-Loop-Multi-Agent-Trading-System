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


class AnalystMarketContextMixin:
    """Mixin for analyzing market context, including benchmark and sector trends, to adjust the analyst's view of a stock."""

    def _read_context_cache(self) -> Dict[str, Any]:
        if not self.cache_path.exists():
            return {}
        try:
            return json.loads(self.cache_path.read_text(encoding="utf-8"))
        except Exception:
            return {}


    def _write_context_cache(self, cache: Dict[str, Any]) -> None:
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
        except Exception:
            pass


    def _cached_context(self, key: str) -> Optional[Dict[str, Any]]:
        cache = self._read_context_cache()
        item = cache.get(key)
        if not isinstance(item, dict):
            return None
        created_at = item.get("created_at_utc")
        try:
            created_dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        except Exception:
            return None
        ttl_hours = self.config.get("market_context", {}).get("cache_ttl_hours", 6)
        if self._now_utc() - created_dt > timedelta(hours=float(ttl_hours)):
            return None
        return item.get("payload")


    def _save_context_cache(self, key: str, payload: Dict[str, Any]) -> None:
        cache = self._read_context_cache()
        cache[key] = {
            "created_at_utc": self._now_utc().isoformat(),
            "payload": payload,
        }
        self._write_context_cache(cache)


    def _download_context_prices(self, symbol: str) -> Optional[pd.DataFrame]:
        try:
            import yfinance as yf
        except Exception:
            return None
        period = self.config.get("market_context", {}).get("period", "3mo")
        interval = self.config.get("market_context", {}).get("interval", "1d")
        try:
            df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=True)
            if df is None or df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [str(c).lower() for c in df.columns]
            return df.reset_index()
        except Exception:
            return None


    def _summarise_context_symbol(self, symbol: str, label: str) -> Dict[str, Any]:
        key = f"{label}:{symbol}"
        cached = self._cached_context(key)
        if cached:
            return cached

        df = self._download_context_prices(symbol)
        if df is None or df.empty or "close" not in df.columns or len(df) < 25:
            payload = {
                "symbol": symbol,
                "label": label,
                "status": "Unavailable",
                "trend": "Unknown",
                "return_20": None,
                "ma_gap": None,
                "reason": "Context data was not available.",
            }
            self._save_context_cache(key, payload)
            return payload

        close = pd.to_numeric(df["close"], errors="coerce").dropna()
        if len(close) < 25:
            payload = {
                "symbol": symbol,
                "label": label,
                "status": "Unavailable",
                "trend": "Unknown",
                "return_20": None,
                "ma_gap": None,
                "reason": "Not enough context price history.",
            }
            self._save_context_cache(key, payload)
            return payload

        ret20 = float(close.pct_change(20).iloc[-1])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else ma20
        ma_gap = (ma20 - ma50) / ma50 if ma50 else 0.0

        if ret20 > 0.03 and ma_gap > 0:
            trend = "Bullish"
            status = "Supportive"
        elif ret20 < -0.03 and ma_gap < 0:
            trend = "Bearish"
            status = "Weak"
        else:
            trend = "Mixed"
            status = "Neutral"

        payload = {
            "symbol": symbol,
            "label": label,
            "status": status,
            "trend": trend,
            "return_20": round(ret20, 6),
            "ma_gap": round(ma_gap, 6),
            "reason": f"{label} trend is {trend.lower()} based on 20-day return and moving-average gap.",
        }
        self._save_context_cache(key, payload)
        return payload


    def _market_context(self, symbol: str) -> Dict[str, Any]:
        enabled = self.config.get("market_context", {}).get("enabled", True)
        if not enabled:
            return {
                "enabled": False,
                "market_regime": {"trend": "Unknown", "status": "Disabled"},
                "sector_trend": {"trend": "Unknown", "status": "Disabled"},
            }

        benchmark = self.config.get("market_context", {}).get("benchmark_symbol", "SPY")
        sector_map = self.config.get("sector_etf_map", {})
        sector_symbol = sector_map.get(str(symbol).upper())

        market = self._summarise_context_symbol(benchmark, "Market regime")
        if sector_symbol:
            sector = self._summarise_context_symbol(sector_symbol, "Sector trend")
        else:
            sector = {
                "symbol": None,
                "label": "Sector trend",
                "status": "Unavailable",
                "trend": "Unknown",
                "return_20": None,
                "ma_gap": None,
                "reason": "No sector ETF mapping was configured for this symbol.",
            }

        return {
            "enabled": True,
            "benchmark_symbol": benchmark,
            "sector_symbol": sector_symbol,
            "market_regime": market,
            "sector_trend": sector,
        }


    def _context_adjustment(self, context_item: Dict[str, Any], weight: float) -> Tuple[float, Dict[str, Any]]:
        trend = context_item.get("trend")
        if trend == "Bullish":
            adjustment = abs(weight)
            message = f"{context_item.get('label', 'Context')} is supportive."
        elif trend == "Bearish":
            adjustment = -abs(weight)
            message = f"{context_item.get('label', 'Context')} is weak."
        elif trend == "Mixed":
            adjustment = 0.0
            message = f"{context_item.get('label', 'Context')} is mixed."
        else:
            adjustment = 0.0
            message = f"{context_item.get('label', 'Context')} is unavailable."
        contribution = self._contribution(
            indicator=context_item.get("label", "Context"),
            value=trend or "Unknown",
            contribution=adjustment,
            message=message,
            group="context",
        )
        return adjustment, contribution

