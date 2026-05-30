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


class StorageHelpersMixin:

    """Helper methods for the StorageAgent, including ID generation, JSON serialization, symbol normalization, safe type conversion, nested dictionary access, and period parsing."""

    def _now_utc(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


    def _new_id(self, prefix: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
        return f"{prefix}_{stamp}_{uuid.uuid4().hex[:10]}"


    def _to_json(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return json.dumps({"serialization_error": repr(value)}, ensure_ascii=False)


    def _from_json(self, value: Any, default=None):
        if value is None:
            return default
        if isinstance(value, (dict, list)):
            return value
        try:
            return json.loads(value)
        except Exception:
            return default


    def _normalise_symbol(self, symbol: Any) -> str:
        return str(symbol or "UNKNOWN").upper().strip()


    def _safe_float(self, value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None


    def _safe_int(self, value: Any) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(float(value))
        except Exception:
            return None


    def _get_nested(self, data: Dict[str, Any], keys: Sequence[str], default=None):
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current


    def _period_to_start_timestamp(self, period: str) -> Optional[str]:
        period = str(period or "").lower().strip()
        if not period or period == "max":
            return None
        days_map = {
            "1d": 1,
            "5d": 5,
            "7d": 7,
            "30d": 30,
            "1mo": 31,
            "3mo": 93,
            "6mo": 186,
            "1y": 366,
            "2y": 732,
            "5y": 1830,
            "10y": 3660,
        }
        days = days_map.get(period)
        if days is None:
            return None
        start = datetime.now(timezone.utc) - timedelta(days=days + 5)
        return start.strftime("%Y-%m-%d %H:%M:%S")

