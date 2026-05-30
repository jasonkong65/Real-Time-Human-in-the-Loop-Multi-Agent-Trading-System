from __future__ import annotations

from __future__ import annotations

import json

import os

import uuid

from datetime import datetime, timezone, timedelta

from pathlib import Path

from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from agents.database_backend import DatabaseBackend
except Exception:
    from database_backend import DatabaseBackend


class StorageHistoricalMixin:


    def record_historical_prices(
        self,
        symbol: str,
        prices: Any,
        period: str = "1y",
        interval: str = "1d",
        source: str = "yfinance",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        symbol = self._normalise_symbol(symbol)
        downloaded_at = self._now_utc()

        if pd is not None and isinstance(prices, pd.DataFrame):
            records = prices.to_dict("records")
        elif isinstance(prices, dict) and "prices" in prices:
            records = prices.get("prices") or []
        else:
            records = list(prices or [])

        rows = []
        for rec in records:
            date_value = rec.get("date") or rec.get("datetime") or rec.get("price_timestamp") or rec.get("timestamp")
            if date_value is None:
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "period": period,
                    "interval": interval,
                    "price_timestamp": str(date_value),
                    "open": self._safe_float(rec.get("open")),
                    "high": self._safe_float(rec.get("high")),
                    "low": self._safe_float(rec.get("low")),
                    "close": self._safe_float(rec.get("close")),
                    "adj_close": self._safe_float(rec.get("adj_close") or rec.get("adjclose")),
                    "volume": self._safe_float(rec.get("volume")),
                    "source": source,
                    "downloaded_at_utc": downloaded_at,
                    "raw_json": self._to_json(rec),
                }
            )

        for row in rows:
            self.backend.upsert(
                "historical_prices",
                row,
                conflict_cols=["symbol", "interval", "price_timestamp"],
            )

        latest = max([r["price_timestamp"] for r in rows], default=None)
        self.record_historical_metadata(
            symbol=symbol,
            period=period,
            interval=interval,
            latest_timestamp=latest,
            record_count=len(rows),
            source=source,
            storage_mode="database",
            metadata=metadata or {},
        )
        return {"success": True, "symbol": symbol, "rows_written": len(rows), "latest_timestamp": latest}


    def record_historical_metadata(
        self,
        symbol: str,
        period: str,
        interval: str,
        latest_timestamp: Optional[str],
        record_count: int,
        source: str,
        storage_mode: str = "database",
        metadata: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        stale_warning: bool = False,
    ) -> Dict[str, Any]:
        row = {
            "symbol": self._normalise_symbol(symbol),
            "period": period,
            "interval": interval,
            "latest_timestamp": latest_timestamp,
            "downloaded_at_utc": self._now_utc(),
            "record_count": int(record_count or 0),
            "source": source,
            "storage_mode": storage_mode,
            "stale_warning": 1 if stale_warning else 0,
            "warnings_json": self._to_json(warnings or []),
            "raw_json": self._to_json(metadata or {}),
        }
        self.backend.upsert("historical_metadata", row, conflict_cols=["symbol", "period", "interval"])
        return {"success": True, "symbol": row["symbol"], "period": period, "interval": interval}


    def get_historical_prices(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        limit: Optional[int] = None,
        as_dataframe: bool = True,
    ) -> Any:
        symbol = self._normalise_symbol(symbol)
        params: Dict[str, Any] = {"symbol": symbol, "interval": interval}
        start_ts = self._period_to_start_timestamp(period)
        where = "symbol = :symbol AND interval = :interval"
        if start_ts:
            where += " AND price_timestamp >= :start_ts"
            params["start_ts"] = start_ts

        sql = f"""
            SELECT price_timestamp AS date, open, high, low, close, adj_close, volume, source, downloaded_at_utc
            FROM historical_prices
            WHERE {where}
            ORDER BY price_timestamp ASC
        """
        if limit:
            sql += f" LIMIT {self.backend.safe_limit(limit)}"
        rows = self.backend.query(sql, params)
        if as_dataframe and pd is not None:
            return pd.DataFrame(rows)
        return rows


    def get_historical_metadata(self, symbol: str, period: str = "1y", interval: str = "1d") -> Optional[Dict[str, Any]]:
        rows = self.backend.query(
            """
            SELECT * FROM historical_metadata
            WHERE symbol = :symbol AND period = :period AND interval = :interval
            """,
            {"symbol": self._normalise_symbol(symbol), "period": period, "interval": interval},
        )
        if not rows:
            return None
        row = rows[0]
        row["warnings"] = self._from_json(row.get("warnings_json"), [])
        row["metadata"] = self._from_json(row.get("raw_json"), {})
        return row

