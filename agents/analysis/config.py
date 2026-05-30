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


class AnalystConfigMixin:


    def _load_config(self) -> Dict[str, Any]:
        config = json.loads(json.dumps(self.DEFAULT_CONFIG))
        if self.config_path.exists():
            try:
                user_config = json.loads(self.config_path.read_text(encoding="utf-8"))
                config = self._deep_merge(config, user_config)
            except Exception:
                pass
        return config


    def _deep_merge(self, base: Dict[str, Any], extra: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                base[key] = self._deep_merge(base[key], value)
            else:
                base[key] = value
        return base


    @staticmethod
    def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default


    @staticmethod
    def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
        try:
            return max(low, min(high, float(value)))
        except Exception:
            return low


    @staticmethod
    def _format_pct(value: Optional[float]) -> str:
        if value is None:
            return "N/A"
        try:
            return f"{value:.2%}"
        except Exception:
            return "N/A"


    @staticmethod
    def _now_utc() -> datetime:
        return datetime.now(timezone.utc)


    def _confidence_score(self, validation_result: Dict[str, Any]) -> float:
        score = self._safe_float(validation_result.get("confidence_score"))
        if score is not None:
            return self._clip(score)
        confidence = str(validation_result.get("confidence", "Medium")).lower()
        return {"high": 1.0, "medium": 0.75, "low": 0.45}.get(confidence, 0.60)


    def _source(self, multi_quote: Dict[str, Any], names: List[str]) -> Dict[str, Any]:
        for name in names:
            value = multi_quote.get(name)
            if isinstance(value, dict):
                return value
        return {}


    def _price(self, quote: Dict[str, Any], key: str) -> Optional[float]:
        # Support both old and new field names.
        aliases = {
            "previous_close": ["previous_close", "previous_close_price"],
            "current_price": ["current_price", "price", "latest_price"],
            "open_price": ["open_price", "open"],
            "high_price": ["high_price", "high"],
            "low_price": ["low_price", "low"],
        }
        for candidate in aliases.get(key, [key]):
            value = self._safe_float(quote.get(candidate))
            if value is not None:
                return value
        return None

