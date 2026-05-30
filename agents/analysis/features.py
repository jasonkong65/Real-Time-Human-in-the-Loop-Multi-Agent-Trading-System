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


class AnalystFeatureMixin:


    def _fallback_feature_builder(self, price_records: List[Dict[str, Any]]) -> pd.DataFrame:
        if not price_records:
            return pd.DataFrame()
        df = pd.DataFrame(price_records)
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()
            df[col] = pd.to_numeric(df[col], errors="coerce")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date")
        df = df.dropna(subset=required).reset_index(drop=True)
        if len(df) < 30:
            return pd.DataFrame()

        close = df["close"]
        volume = df["volume"]
        df["return_1"] = close.pct_change(1)
        df["return_5"] = close.pct_change(5)
        df["return_20"] = close.pct_change(20)
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean() if len(df) >= 50 else ma20
        df["ma_gap"] = (ma20 - ma50) / ma50.replace(0, pd.NA)
        df["volatility_20"] = close.pct_change().rolling(20).std()
        df["volume_change"] = (volume.rolling(5).mean() - volume.rolling(20).mean()) / volume.rolling(20).mean().replace(0, pd.NA)
        df["rsi_14"] = self._rsi(close, 14)
        return df.dropna(subset=["return_1", "return_5", "return_20", "ma_gap", "volatility_20", "volume_change", "rsi_14"]).reset_index(drop=True)


    def _build_features(self, price_records: List[Dict[str, Any]]) -> pd.DataFrame:
        if build_trading_features is not None:
            try:
                feature_df = build_trading_features(price_records)
                if isinstance(feature_df, pd.DataFrame) and not feature_df.empty:
                    return feature_df
            except Exception:
                pass
        return self._fallback_feature_builder(price_records)


    def _rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))


    def _contribution(
        self,
        indicator: str,
        value: Any,
        contribution: float,
        message: str,
        group: str,
    ) -> Dict[str, Any]:
        if contribution > 0.005:
            direction = "positive"
        elif contribution < -0.005:
            direction = "negative"
        else:
            direction = "neutral"
        return {
            "indicator": indicator,
            "value": value,
            "contribution": round(float(contribution), 4),
            "direction": direction,
            "message": message,
            "group": group,
        }


    def _top_contributions(self, contributions: List[Dict[str, Any]], n: int = 5) -> List[Dict[str, Any]]:
        return sorted(contributions, key=lambda item: abs(item.get("contribution", 0)), reverse=True)[:n]

