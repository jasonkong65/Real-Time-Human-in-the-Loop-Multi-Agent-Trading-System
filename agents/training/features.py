from __future__ import annotations

from __future__ import annotations

import json

from datetime import datetime, timezone

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

import joblib

import pandas as pd

from sklearn.base import clone

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)


class TrainingFeatureMixin:


    def _rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))


    def _clean_history(self, historical_data: Dict[str, Any]) -> pd.DataFrame:
        prices = historical_data.get("prices", []) if isinstance(historical_data, dict) else []
        if not prices:
            return pd.DataFrame()

        df = pd.DataFrame(prices)
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

        if "close" not in df.columns and "adj_close" in df.columns:
            df["close"] = df["adj_close"]

        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()
            df[col] = pd.to_numeric(df[col], errors="coerce")

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.sort_values("date")

        return df.dropna(subset=required).reset_index(drop=True)


    def _make_label(self, future_return: float) -> str:
        # Moderate thresholds keep the model suitable for watchlist-style paper decisions.
        if future_return >= 0.018:
            return "BUY_CANDIDATE"
        if future_return <= -0.018:
            return "SELL_RISK"
        return "HOLD"


    def _build_single_dataset(
        self,
        historical_data: Dict[str, Any],
        validation_confidence_score: float = 0.95,
    ) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
        df = self._clean_history(historical_data)
        if df.empty or len(df) < 80:
            raise ValueError("Not enough historical data for model training.")

        close = df["close"]
        volume = df["volume"]
        features = pd.DataFrame(index=df.index)
        features["return_1"] = close.pct_change(1)
        features["return_5"] = close.pct_change(5)
        features["return_20"] = close.pct_change(20)

        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        features["ma_gap"] = (ma20 - ma50) / ma50.replace(0, 1e-9)
        features["volatility_20"] = close.pct_change().rolling(20).std()
        features["volume_change"] = (
            volume.rolling(5).mean() - volume.rolling(20).mean()
        ) / volume.rolling(20).mean().replace(0, 1e-9)
        features["rsi_14"] = self._rsi(close, 14)
        features["validation_confidence_score"] = validation_confidence_score

        future_return = close.shift(-5) / close - 1
        labels = future_return.apply(
            lambda x: self._make_label(float(x)) if pd.notna(x) else None
        )

        dataset = features.copy()
        dataset["label"] = labels
        dataset = dataset.replace([float("inf"), float("-inf")], pd.NA).dropna()

        X = dataset[self.feature_columns].astype(float)
        y = dataset["label"].astype(str)

        if y.nunique() < 2:
            raise ValueError("Training labels contain fewer than two classes.")

        label_distribution = self._clean_label_dict(y.value_counts().to_dict())
        return X, y, {
            "label_distribution": label_distribution,
            "num_samples": int(len(X)),
        }


    def _read_local_history_file(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            elif path.suffix.lower() == ".csv":
                df = pd.read_csv(path)
            else:
                return None
            if df.empty:
                return None
            symbol = path.stem.split("_")[0].upper()
            return {"success": True, "symbol": symbol, "prices": df.to_dict("records")}
        except Exception:
            return None


    def _load_pooled_local_datasets(
        self,
        current_symbol: Optional[str],
        validation_confidence_score: float = 0.95,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Dict[str, Any]]:
        if not self.pooled_data_dir.exists():
            return None, None, {"enabled": False, "reason": "No local historical data directory."}

        current_symbol = str(current_symbol or "").upper().strip()
        files = []
        files.extend(sorted(self.pooled_data_dir.glob("*.parquet")))
        files.extend(sorted(self.pooled_data_dir.glob("*.csv")))

        X_parts: List[pd.DataFrame] = []
        y_parts: List[pd.Series] = []
        used_symbols: List[str] = []
        skipped: List[str] = []

        seen_symbols = set()
        for path in files:
            symbol = path.stem.split("_")[0].upper()
            if not symbol or symbol == current_symbol or symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)

            local_history = self._read_local_history_file(path)
            if not local_history:
                skipped.append(path.name)
                continue

            try:
                X_local, y_local, _ = self._build_single_dataset(
                    local_history,
                    validation_confidence_score=validation_confidence_score,
                )
                X_parts.append(X_local)
                y_parts.append(y_local)
                used_symbols.append(symbol)
            except Exception:
                skipped.append(path.name)

            if len(used_symbols) >= self.max_pooled_symbols:
                break

        if not X_parts:
            return None, None, {
                "enabled": False,
                "reason": "No usable pooled local datasets found.",
                "skipped_files": skipped[:10],
            }

        X_pool = pd.concat(X_parts, ignore_index=True)
        y_pool = pd.concat(y_parts, ignore_index=True)

        return X_pool, y_pool, {
            "enabled": True,
            "used_symbols": used_symbols,
            "num_pooled_rows": int(len(X_pool)),
            "skipped_files": skipped[:10],
        }

