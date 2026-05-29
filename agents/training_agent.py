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


class TrainingAgent:
    """
    Training Agent

    Builds and maintains a lightweight stock-signal model automatically.

    What this version does:
    - Uses walk-forward validation instead of one static 75/25 split.
    - Tests several model candidates automatically.
    - Can add pooled local historical data as extra training context.
    - Selects models using balanced accuracy, macro F1, accuracy, and SELL_RISK recall.
    - Saves confusion matrix, classification report, feature importance, and evaluation table.
    - Only overwrites an existing model when the new model is meaningfully better.
    - Keeps the core output labels expected by the current app.py:
      BUY_CANDIDATE, HOLD, SELL_RISK.
    """

    MODEL_VERSION = "auto_walkforward_rf_v4"
    CORE_SIGNALS = ["BUY_CANDIDATE", "HOLD", "SELL_RISK"]
    FEATURE_COLUMNS = [
        "return_1",
        "return_5",
        "return_20",
        "ma_gap",
        "volatility_20",
        "volume_change",
        "rsi_14",
        "validation_confidence_score",
    ]

    def __init__(
        self,
        model_path: str = "models/signal_model.pkl",
        auto_retrain_days: int = 7,
        min_save_improvement: float = 0.01,
        pooled_data_dir: str = "data/historical",
        max_pooled_symbols: int = 12,
    ):
        self.base_model_path = Path(model_path)
        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        self.auto_retrain_days = auto_retrain_days
        self.min_save_improvement = min_save_improvement
        self.pooled_data_dir = Path(pooled_data_dir)
        self.max_pooled_symbols = max_pooled_symbols

        self.feature_columns = list(self.FEATURE_COLUMNS)
        self.model_bundle: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _now_dt(self) -> datetime:
        return datetime.now(timezone.utc)

    def _set_symbol_model_path(self, symbol: Optional[str] = None) -> None:
        if symbol:
            clean = str(symbol).upper().strip()
            self.model_path = Path("models") / f"signal_model_{clean}.pkl"
        else:
            self.model_path = self.base_model_path
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def _metadata_path(self) -> Path:
        return self.model_path.with_suffix(".metadata.json")

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _round_float(self, value: Any, digits: int = 4):
        try:
            return round(float(value), digits)
        except Exception:
            return value

    def _clean_label_dict(self, values: Dict[Any, Any]) -> Dict[str, Any]:
        return {str(k): int(v) if isinstance(v, (int, float)) and float(v).is_integer() else v for k, v in values.items()}

    def _model_age_days(self, bundle: Optional[Dict[str, Any]]) -> Optional[float]:
        if not bundle:
            return None
        trained_at = bundle.get("trained_at_utc") or bundle.get("created_at_utc")
        if not trained_at:
            return None
        try:
            dt = datetime.strptime(str(trained_at), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            return (self._now_dt() - dt).total_seconds() / 86400
        except Exception:
            return None

    def _should_auto_retrain(self, bundle: Optional[Dict[str, Any]], force_retrain: bool) -> bool:
        if force_retrain:
            return True
        if not bundle:
            return True
        if bundle.get("model_version") != self.MODEL_VERSION:
            return True
        age = self._model_age_days(bundle)
        if age is None:
            return True
        return age >= self.auto_retrain_days

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Candidate models and walk-forward validation
    # ------------------------------------------------------------------
    def _candidate_models(self, pooled_available: bool) -> List[Tuple[str, Any, Dict[str, Any], bool]]:
        candidates: List[Tuple[str, Any, Dict[str, Any], bool]] = [
            (
                "rf_conservative",
                RandomForestClassifier(
                    n_estimators=120,
                    max_depth=4,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                {"family": "RandomForest", "n_estimators": 120, "max_depth": 4, "min_samples_leaf": 4},
                False,
            ),
            (
                "rf_balanced",
                RandomForestClassifier(
                    n_estimators=180,
                    max_depth=6,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                {"family": "RandomForest", "n_estimators": 180, "max_depth": 6, "min_samples_leaf": 3},
                False,
            ),
            (
                "rf_flexible",
                RandomForestClassifier(
                    n_estimators=240,
                    max_depth=None,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                {"family": "RandomForest", "n_estimators": 240, "max_depth": None, "min_samples_leaf": 5},
                False,
            ),
            (
                "extra_trees_stable",
                ExtraTreesClassifier(
                    n_estimators=220,
                    max_depth=6,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
                {"family": "ExtraTrees", "n_estimators": 220, "max_depth": 6, "min_samples_leaf": 4},
                False,
            ),
        ]

        if pooled_available:
            candidates.extend(
                [
                    (
                        "pooled_rf_balanced",
                        RandomForestClassifier(
                            n_estimators=220,
                            max_depth=6,
                            min_samples_leaf=3,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1,
                        ),
                        {"family": "RandomForest", "n_estimators": 220, "max_depth": 6, "min_samples_leaf": 3, "scope": "pooled_local_cache"},
                        True,
                    ),
                    (
                        "pooled_extra_trees_stable",
                        ExtraTreesClassifier(
                            n_estimators=240,
                            max_depth=7,
                            min_samples_leaf=4,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=-1,
                        ),
                        {"family": "ExtraTrees", "n_estimators": 240, "max_depth": 7, "min_samples_leaf": 4, "scope": "pooled_local_cache"},
                        True,
                    ),
                ]
            )

        return candidates

    def _walk_forward_splits(self, n_rows: int) -> List[Tuple[int, int]]:
        min_train = max(80, int(n_rows * 0.40))
        test_size = max(20, int(n_rows * 0.12))
        splits: List[Tuple[int, int]] = []
        start = min_train
        while start + test_size <= n_rows:
            splits.append((start, start + test_size))
            start += test_size
        if not splits and n_rows > 110:
            split = int(n_rows * 0.75)
            splits.append((split, n_rows))
        return splits[-5:]

    def _score_from_metrics(self, metrics: Dict[str, float]) -> float:
        return (
            0.35 * metrics.get("balanced_accuracy", 0.0)
            + 0.35 * metrics.get("macro_f1", 0.0)
            + 0.20 * metrics.get("accuracy", 0.0)
            + 0.10 * metrics.get("sell_risk_recall", 0.0)
        )

    def _walk_forward_score(
        self,
        model,
        X_current: pd.DataFrame,
        y_current: pd.Series,
        X_pool: Optional[pd.DataFrame] = None,
        y_pool: Optional[pd.Series] = None,
        use_pooled: bool = False,
    ) -> Dict[str, Any]:
        splits = self._walk_forward_splits(len(X_current))
        if not splits:
            raise ValueError("Not enough rows for walk-forward validation.")

        y_true_all: List[str] = []
        y_pred_all: List[str] = []
        fold_rows: List[Dict[str, Any]] = []

        for fold_id, (train_end, test_end) in enumerate(splits, start=1):
            X_train = X_current.iloc[:train_end]
            y_train = y_current.iloc[:train_end]
            X_test = X_current.iloc[train_end:test_end]
            y_test = y_current.iloc[train_end:test_end]

            if use_pooled and X_pool is not None and y_pool is not None and len(X_pool) > 0:
                X_train = pd.concat([X_pool, X_train], ignore_index=True)
                y_train = pd.concat([y_pool, y_train], ignore_index=True)

            if y_train.nunique() < 2 or y_test.empty:
                continue

            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            pred = fold_model.predict(X_test)

            y_true = list(y_test.astype(str))
            y_pred = [str(x) for x in pred]
            y_true_all.extend(y_true)
            y_pred_all.extend(y_pred)

            fold_metrics = {
                "fold": fold_id,
                "train_size": int(len(X_train)),
                "test_size": int(len(X_test)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "sell_risk_recall": float(recall_score(y_true, y_pred, labels=["SELL_RISK"], average="macro", zero_division=0)),
            }
            fold_metrics["score"] = float(self._score_from_metrics(fold_metrics))
            fold_rows.append(fold_metrics)

        if not y_true_all:
            raise ValueError("No valid walk-forward folds were produced.")

        aggregate_metrics = {
            "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
            "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
            "sell_risk_recall": float(recall_score(y_true_all, y_pred_all, labels=["SELL_RISK"], average="macro", zero_division=0)),
            "num_folds": int(len(fold_rows)),
            "total_test_size": int(len(y_true_all)),
        }
        aggregate_metrics["score"] = float(self._score_from_metrics(aggregate_metrics))

        labels = list(self.CORE_SIGNALS)
        cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
        report = classification_report(y_true_all, y_pred_all, labels=labels, zero_division=0, output_dict=True)

        return {
            "score": aggregate_metrics["score"],
            "accuracy": aggregate_metrics["accuracy"],
            "balanced_accuracy": aggregate_metrics["balanced_accuracy"],
            "macro_f1": aggregate_metrics["macro_f1"],
            "sell_risk_recall": aggregate_metrics["sell_risk_recall"],
            "num_folds": aggregate_metrics["num_folds"],
            "total_test_size": aggregate_metrics["total_test_size"],
            "folds": fold_rows,
            "confusion_matrix": {
                "labels": labels,
                "matrix": cm.tolist(),
            },
            "classification_report": report,
            "pred_distribution": self._clean_label_dict(pd.Series(y_pred_all).value_counts().to_dict()),
            "true_distribution": self._clean_label_dict(pd.Series(y_true_all).value_counts().to_dict()),
        }

    def _baseline_score(self, y: pd.Series) -> Dict[str, Any]:
        splits = self._walk_forward_splits(len(y))
        y_true_all: List[str] = []
        y_pred_all: List[str] = []
        for train_end, test_end in splits:
            y_train = y.iloc[:train_end]
            y_test = y.iloc[train_end:test_end]
            if y_train.empty or y_test.empty:
                continue
            majority = str(y_train.mode().iloc[0])
            y_true_all.extend(list(y_test.astype(str)))
            y_pred_all.extend([majority] * len(y_test))
        if not y_true_all:
            return {"accuracy": None, "score": None}
        metrics = {
            "accuracy": float(accuracy_score(y_true_all, y_pred_all)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
            "macro_f1": float(f1_score(y_true_all, y_pred_all, average="macro", zero_division=0)),
            "sell_risk_recall": float(recall_score(y_true_all, y_pred_all, labels=["SELL_RISK"], average="macro", zero_division=0)),
        }
        metrics["score"] = float(self._score_from_metrics(metrics))
        return metrics

    def _feature_importance(self, model) -> List[Dict[str, Any]]:
        if not hasattr(model, "feature_importances_"):
            return []
        rows = []
        for feature, importance in zip(self.feature_columns, model.feature_importances_):
            rows.append({"feature": feature, "importance": round(float(importance), 6)})
        return sorted(rows, key=lambda r: r["importance"], reverse=True)

    def _save_metadata(self, metadata: Dict[str, Any]) -> None:
        try:
            with self._metadata_path().open("w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Training and model management
    # ------------------------------------------------------------------
    def _feature_snapshot(self, X: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Small feature-distribution snapshot for drift checks."""
        snapshot: Dict[str, Dict[str, float]] = {}
        try:
            for col in self.feature_columns:
                if col in X.columns:
                    series = pd.to_numeric(X[col], errors="coerce").dropna()
                    if not series.empty:
                        snapshot[col] = {
                            "mean": float(series.mean()),
                            "std": float(series.std(ddof=0)),
                        }
        except Exception:
            return {}
        return snapshot

    def _feature_drift_report(self, existing_bundle: Optional[Dict[str, Any]], current_snapshot: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compare current feature snapshot with the previous saved model snapshot."""
        previous = {}
        if isinstance(existing_bundle, dict):
            previous = existing_bundle.get("feature_snapshot") or existing_bundle.get("training_feature_snapshot") or {}
        rows = []
        max_abs_z = 0.0
        for feature, cur in current_snapshot.items():
            old = previous.get(feature, {}) if isinstance(previous, dict) else {}
            old_mean = old.get("mean")
            old_std = old.get("std") or 0.0
            cur_mean = cur.get("mean")
            if old_mean is None or cur_mean is None:
                continue
            denom = max(abs(float(old_std)), 1e-6)
            z = float(cur_mean - old_mean) / denom
            max_abs_z = max(max_abs_z, abs(z))
            rows.append({
                "feature": feature,
                "previous_mean": round(float(old_mean), 6),
                "current_mean": round(float(cur_mean), 6),
                "z_like_shift": round(z, 4),
            })
        if not rows:
            label = "No baseline"
        elif max_abs_z >= 2.0:
            label = "High"
        elif max_abs_z >= 1.0:
            label = "Medium"
        else:
            label = "Low"
        return {"drift_level": label, "max_abs_z_like_shift": round(max_abs_z, 4), "details": rows[:10]}

    def _record_training_run_to_storage(self, symbol: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Best-effort SQLite logging; training should never fail because storage failed."""
        try:
            from agents.storage_agent import StorageAgent
            return StorageAgent().record_training_run(run_id=None, symbol=symbol, training_result=result)
        except Exception as exc:
            return {"success": False, "error": str(exc), "summary": "Training result was not written to SQLite."}

    def _auto_train(
        self,
        historical_data: Dict[str, Any],
        symbol: Optional[str],
        validation_confidence_score: float = 0.95,
        existing_bundle: Optional[Dict[str, Any]] = None,
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        symbol_clean = str(symbol or historical_data.get("symbol", "UNKNOWN")).upper().strip()
        X_current, y_current, data_info = self._build_single_dataset(historical_data, validation_confidence_score)
        current_feature_snapshot = self._feature_snapshot(X_current)
        feature_drift = self._feature_drift_report(existing_bundle, current_feature_snapshot)
        X_pool, y_pool, pooled_info = self._load_pooled_local_datasets(symbol_clean, validation_confidence_score)
        pooled_available = X_pool is not None and y_pool is not None and len(X_pool) >= 80

        baseline = self._baseline_score(y_current)
        evaluation_table: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None

        for name, model, params, use_pooled in self._candidate_models(pooled_available=pooled_available):
            try:
                score_info = self._walk_forward_score(
                    model,
                    X_current,
                    y_current,
                    X_pool=X_pool,
                    y_pool=y_pool,
                    use_pooled=use_pooled,
                )
                row = {
                    "candidate": name,
                    "training_scope": "pooled_local_cache" if use_pooled else "single_symbol",
                    "score": round(score_info["score"], 6),
                    "accuracy": round(score_info["accuracy"], 6),
                    "balanced_accuracy": round(score_info["balanced_accuracy"], 6),
                    "macro_f1": round(score_info["macro_f1"], 6),
                    "sell_risk_recall": round(score_info["sell_risk_recall"], 6),
                    "num_folds": score_info["num_folds"],
                    "total_test_size": score_info["total_test_size"],
                    "params": params,
                }
                evaluation_table.append(row)
                if best is None or score_info["score"] > best["score_info"]["score"]:
                    best = {
                        "name": name,
                        "model": model,
                        "params": params,
                        "score_info": score_info,
                        "use_pooled": use_pooled,
                    }
            except Exception as exc:
                evaluation_table.append({"candidate": name, "error": str(exc)})

        if best is None:
            raise ValueError("Automatic model selection failed for every candidate.")

        # Train final candidate on all available rows. Validation remains current-symbol based.
        X_train_final = X_current
        y_train_final = y_current
        if best["use_pooled"] and pooled_available:
            X_train_final = pd.concat([X_pool, X_current], ignore_index=True)
            y_train_final = pd.concat([y_pool, y_current], ignore_index=True)

        final_model = clone(best["model"])
        final_model.fit(X_train_final, y_train_final)
        feature_importance = self._feature_importance(final_model)

        new_score = float(best["score_info"]["score"])
        existing_score = None
        if isinstance(existing_bundle, dict):
            existing_score = self._safe_float(existing_bundle.get("selection_score"), default=None)

        should_save = existing_bundle is None or existing_score is None or new_score >= existing_score + self.min_save_improvement
        save_decision = "saved_new_model" if should_save else "kept_existing_model"

        bundle = {
            "model": final_model,
            "feature_columns": self.feature_columns,
            "model_version": self.MODEL_VERSION,
            "symbol": symbol_clean,
            "trained_at_utc": self._now(),
            "best_candidate": best["name"],
            "best_params": best["params"],
            "training_scope": "pooled_local_cache" if best["use_pooled"] else "single_symbol",
            "selection_score": new_score,
            "model_selection_score": best["score_info"],
            "baseline_score": baseline,
            "label_distribution": data_info["label_distribution"],
            "pooled_data_info": pooled_info,
            "feature_importance": feature_importance,
            "feature_snapshot": current_feature_snapshot,
            "feature_drift": feature_drift,
        }

        if should_save:
            joblib.dump(bundle, self.model_path)
            self.model_bundle = bundle
            self._save_metadata({k: v for k, v in bundle.items() if k != "model"})
        else:
            self.model_bundle = existing_bundle

        best_score = best["score_info"]
        improvement_over_baseline = None
        if baseline.get("score") is not None:
            improvement_over_baseline = round(new_score - float(baseline["score"]), 6)

        result = {
            "success": True,
            "agent": "Training Agent",
            "agent_goal": "Train and select a signal model automatically.",
            "symbol": symbol_clean,
            "model_source": "auto_model_selection",
            "save_decision": save_decision,
            "saved": bool(should_save),
            "model_path": str(self.model_path),
            "metadata_path": str(self._metadata_path()),
            "model_version": self.MODEL_VERSION,
            "best_candidate": best["name"],
            "best_params": best["params"],
            "training_scope": "pooled_local_cache" if best["use_pooled"] else "single_symbol",
            "selection_score": round(new_score, 6),
            "test_accuracy": round(best_score["accuracy"], 6),
            "balanced_accuracy": round(best_score["balanced_accuracy"], 6),
            "macro_f1": round(best_score["macro_f1"], 6),
            "sell_risk_recall": round(best_score["sell_risk_recall"], 6),
            "baseline_accuracy": self._round_float(baseline.get("accuracy"), 6),
            "baseline_score": self._round_float(baseline.get("score"), 6),
            "improvement_over_baseline_score": improvement_over_baseline,
            "num_samples": data_info["num_samples"],
            "label_distribution": data_info["label_distribution"],
            "pooled_data_info": pooled_info,
            "evaluation_table": evaluation_table,
            "optimization_results": evaluation_table,
            "walk_forward_summary": {
                "num_folds": best_score["num_folds"],
                "total_test_size": best_score["total_test_size"],
                "folds": best_score["folds"],
            },
            "confusion_matrix": best_score["confusion_matrix"],
            "classification_report": best_score["classification_report"],
            "feature_importance": feature_importance,
            "feature_drift": feature_drift,
            "feature_snapshot": current_feature_snapshot,
            "model_comparison_reason": (
                "New model saved because no comparable model existed or the score improved enough."
                if should_save
                else "Existing model kept because the new model did not exceed the saved score by the minimum improvement threshold."
            ),
            "agent_decision": (
                "A new model was saved after automatic comparison."
                if should_save
                else "The existing model was kept because the new model was not clearly better."
            ),
            "summary": (
                f"Training Agent selected {best['name']} for {symbol_clean}."
                if should_save
                else f"Training Agent tested new models for {symbol_clean}, but kept the existing model."
            ),
        }
        result["storage_result"] = self._record_training_run_to_storage(symbol_clean, result)
        return result

    def _load_model(self) -> Optional[Dict[str, Any]]:
        if not self.model_path.exists() or self.model_path.stat().st_size == 0:
            return None
        try:
            bundle = joblib.load(self.model_path)
            if isinstance(bundle, dict) and bundle.get("model") is not None:
                self.model_bundle = bundle
                return bundle
            self.model_bundle = {"model": bundle, "feature_columns": self.feature_columns, "model_version": "legacy"}
            return self.model_bundle
        except Exception:
            return None

    def model_exists(self) -> bool:
        return self.model_path.exists()

    def load_existing_model_info(self) -> Dict[str, Any]:
        bundle = self._load_model()
        if not bundle:
            return {"success": False, "summary": "No existing signal model was found."}
        return {
            "success": True,
            "agent": "Training Agent",
            "model_source": "loaded_existing_model",
            "model_path": str(self.model_path),
            "metadata_path": str(self._metadata_path()),
            "model_version": bundle.get("model_version", "unknown"),
            "best_candidate": bundle.get("best_candidate"),
            "best_params": bundle.get("best_params", {}),
            "selection_score": self._round_float(bundle.get("selection_score"), 6),
            "training_scope": bundle.get("training_scope", "unknown"),
            "label_distribution": bundle.get("label_distribution", {}),
            "feature_importance": bundle.get("feature_importance", []),
            "pooled_data_info": bundle.get("pooled_data_info", {}),
            "summary": f"Loaded existing signal model from {self.model_path}.",
        }

    def train_or_load_model(
        self,
        historical_data: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
        force_retrain: bool = False,
    ) -> Dict[str, Any]:
        self._set_symbol_model_path(symbol or (historical_data or {}).get("symbol"))
        existing = self._load_model()

        # No manual optimizer is needed: if the model is missing, old, or forced,
        # the Training Agent automatically runs model selection.
        if existing and not self._should_auto_retrain(existing, force_retrain):
            return self.load_existing_model_info()

        if historical_data and historical_data.get("success"):
            try:
                result = self._auto_train(
                    historical_data,
                    symbol=symbol or historical_data.get("symbol"),
                    existing_bundle=existing,
                    force_retrain=force_retrain,
                )
                self._record_training_result_to_storage(symbol or historical_data.get("symbol"), result)
                return result
            except Exception as exc:
                if existing:
                    self.model_bundle = existing
                    return {
                        "success": True,
                        "agent": "Training Agent",
                        "model_source": "loaded_existing_after_training_failed",
                        "model_path": str(self.model_path),
                        "warning": str(exc),
                        "summary": "Automatic training failed, so the existing model was kept.",
                    }
                return {
                    "success": False,
                    "agent": "Training Agent",
                    "error": str(exc),
                    "summary": "Training Agent could not train a signal model.",
                }

        if existing:
            return self.load_existing_model_info()

        return {
            "success": False,
            "agent": "Training Agent",
            "summary": "No model and no usable historical data were available.",
        }

    # Backward-compatible training aliases
    def train_or_load_signal_model(self, historical_data=None, symbol=None, force_retrain=False):
        return self.train_or_load_model(historical_data, symbol, force_retrain)

    def load_or_train_model(self, historical_data=None, symbol=None, force_retrain=False):
        return self.train_or_load_model(historical_data, symbol, force_retrain)

    def train_model(self, historical_data=None, symbol=None, force_retrain=True):
        return self.train_or_load_model(historical_data, symbol, force_retrain)

    def run_training(self, historical_data=None, symbol=None, force_retrain=False):
        return self.train_or_load_model(historical_data, symbol, force_retrain)

    def run(self, historical_data=None, symbol=None, force_retrain=False):
        return self.train_or_load_model(historical_data, symbol, force_retrain)

    def train_from_historical_data(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.train_or_load_model(historical_data=historical_data, symbol=historical_data.get("symbol"), force_retrain=True)

    def train_from_csv(self, csv_path: str = "data/historical_data.csv") -> Dict[str, Any]:
        df = pd.read_csv(csv_path)
        historical_data = {"success": True, "symbol": "CSV", "prices": df.to_dict("records")}
        return self.train_or_load_model(historical_data=historical_data, symbol="CSV", force_retrain=True)

    def train_from_price_records(self, price_records: list) -> Dict[str, Any]:
        historical_data = {"success": True, "symbol": "RECORDS", "prices": price_records}
        return self.train_or_load_model(historical_data=historical_data, symbol="RECORDS", force_retrain=True)

    # ------------------------------------------------------------------
    # Prediction and context refinement
    # ------------------------------------------------------------------
    def _feature_from_analysis(self, analysis_result: Dict[str, Any]) -> pd.DataFrame:
        feature_source = analysis_result.get("features_for_model") if isinstance(analysis_result, dict) else {}
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        row = {}
        for col in self.feature_columns:
            row[col] = self._safe_float(
                (feature_source or {}).get(col, stage2.get(col)),
                0.0 if col != "rsi_14" else 50.0,
            )
        if row.get("validation_confidence_score", 0.0) == 0.0:
            row["validation_confidence_score"] = 0.6
        return pd.DataFrame([row], columns=self.feature_columns)

    def _confidence_level(self, confidence: float) -> str:
        if confidence >= 0.66:
            return "High"
        if confidence >= 0.45:
            return "Medium"
        return "Low"

    def _predict_raw(self, X: pd.DataFrame) -> Tuple[str, float, Dict[str, float]]:
        bundle = self.model_bundle or self._load_model()
        if not bundle:
            raise ValueError("No signal model is loaded.")
        model = bundle.get("model")
        raw_signal = str(model.predict(X)[0])
        probabilities: Dict[str, float] = {}
        confidence = 0.50
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = list(model.classes_)
            probabilities = {str(cls): float(p) for cls, p in zip(classes, proba)}
            confidence = max(probabilities.values()) if probabilities else 0.50
        return raw_signal, float(confidence), probabilities

    def _context(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        return {
            "analyst_signal": str(analysis_result.get("analyst_signal", "NEUTRAL")).upper(),
            "analyst_score": self._safe_float(analysis_result.get("analyst_score"), 0.5),
            "trend_direction": analysis_result.get("trend_direction") or stage2.get("trend_direction", "Neutral"),
            "entry_risk_level": analysis_result.get("entry_risk_level") or stage2.get("entry_risk_level", "Medium"),
            "volatility_level": analysis_result.get("volatility_level") or stage2.get("volatility_level", "Unknown"),
            "return_20": self._safe_float(stage2.get("return_20"), 0.0),
            "ma_gap": self._safe_float(stage2.get("ma_gap"), 0.0),
            "rsi_14": self._safe_float(stage2.get("rsi_14"), 50.0),
        }

    def _refine_signal(self, raw_signal: str, confidence: float, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        ctx = self._context(analysis_result)
        trend_positive = ctx["trend_direction"] in ["Positive", "Strong Positive"] or (ctx["return_20"] > 0.03 and ctx["ma_gap"] > 0)
        entry_high = ctx["entry_risk_level"] == "High" or ctx["rsi_14"] >= 72
        entry_medium = ctx["entry_risk_level"] in ["Medium", "High"] or ctx["rsi_14"] >= 68
        analyst_bullish = "BULLISH" in ctx["analyst_signal"] or "POSITIVE" in ctx["analyst_signal"] or ctx["analyst_score"] >= 0.62

        final_signal = raw_signal if raw_signal in self.CORE_SIGNALS else "HOLD"
        display_signal = final_signal
        reason = f"Raw model predicted {raw_signal}."

        if final_signal == "BUY_CANDIDATE" and entry_high:
            final_signal = "HOLD"
            display_signal = "BUY_WATCHLIST_OVERBOUGHT"
            reason = "The setup is positive, but entry timing risk is high. The signal is kept as HOLD for paper review."
        elif final_signal == "SELL_RISK" and trend_positive and analyst_bullish and confidence < 0.70:
            final_signal = "HOLD"
            display_signal = "BUY_WATCHLIST_ENTRY_RISK"
            reason = "The model saw pullback risk, but the wider trend is positive. The signal is softened to HOLD."
        elif final_signal == "HOLD" and trend_positive and analyst_bullish:
            display_signal = "BUY_WATCHLIST_OVERBOUGHT" if entry_medium else "BULLISH_WATCHLIST"
            reason = "The model is neutral, while the technical setup is positive. Treat this as a watchlist case."
        elif final_signal == "SELL_RISK" and trend_positive:
            display_signal = "PULLBACK_RISK_IN_UPTREND"
            reason = "The model sees short-term pullback risk inside a positive trend."
        elif final_signal == "BUY_CANDIDATE":
            display_signal = "BUY_RESEARCH_CANDIDATE"
            reason = "The model found a positive setup for human review."
        elif final_signal == "SELL_RISK":
            display_signal = "DOWNSIDE_RISK"
            reason = "The model found downside risk."

        return {
            "model_signal": final_signal,
            "display_signal": display_signal,
            "refinement_reason": reason,
            "context": ctx,
        }

    def _fallback_signal(self, analysis_result: Dict[str, Any], reason: str) -> Dict[str, Any]:
        score = self._safe_float(analysis_result.get("analyst_score"), 0.5)
        analyst_signal = str(analysis_result.get("analyst_signal", "NEUTRAL")).upper()
        entry_risk = analysis_result.get("entry_risk_level", "Medium")
        if "BEARISH" in analyst_signal or score <= 0.35:
            signal, display = "SELL_RISK", "DOWNSIDE_RISK"
        elif ("BULLISH" in analyst_signal or "POSITIVE" in analyst_signal or score >= 0.62) and entry_risk != "High":
            signal, display = "BUY_CANDIDATE", "BUY_RESEARCH_CANDIDATE"
        elif "BULLISH" in analyst_signal or "POSITIVE" in analyst_signal or score >= 0.58:
            signal, display = "HOLD", "BUY_WATCHLIST_ENTRY_RISK"
        else:
            signal, display = "HOLD", "HOLD"
        return {
            "success": True,
            "agent": "Training Agent",
            "agent_goal": "Generate a signal from available analyst context.",
            "signal_source": "context_fallback",
            "model_signal": signal,
            "display_signal": display,
            "raw_model_signal": "N/A",
            "prediction_confidence": round(score, 4),
            "confidence_level": "Medium",
            "agent_decision": f"Used analyst-context fallback because {reason}",
            "signal_for_next_agent": {
                "signal": signal,
                "display_signal": display,
                "prediction_confidence": round(score, 4),
                "confidence_level": "Medium",
            },
            "summary": f"Fallback signal generated: {display}.",
        }

    def generate_signal(
        self,
        analysis_result: Dict[str, Any],
        training_result: Optional[Dict[str, Any]] = None,
        symbol: Optional[str] = None,
    ) -> Dict[str, Any]:
        self._set_symbol_model_path(symbol or analysis_result.get("symbol"))
        try:
            X = self._feature_from_analysis(analysis_result)
            raw_signal, confidence, probabilities = self._predict_raw(X)
            refined = self._refine_signal(raw_signal, confidence, analysis_result)
            confidence_level = self._confidence_level(confidence)
            result = {
                "success": True,
                "agent": "Training Agent",
                "agent_goal": "Use the trained model and current context to generate a risk-aware signal.",
                "symbol": str(symbol or analysis_result.get("symbol", "UNKNOWN")).upper(),
                "signal_source": "auto_selected_model",
                "model_signal": refined["model_signal"],
                "display_signal": refined["display_signal"],
                "raw_model_signal": raw_signal,
                "prediction_confidence": round(confidence, 4),
                "confidence_level": confidence_level,
                "class_probabilities": {k: round(v, 4) for k, v in probabilities.items()},
                "model_path": str(self.model_path),
                "context_used": refined["context"],
                "agent_decision": (
                    f"Raw signal={raw_signal}; final signal={refined['model_signal']}; "
                    f"display={refined['display_signal']}. {refined['refinement_reason']}"
                ),
                "signal_for_next_agent": {
                    "symbol": str(symbol or analysis_result.get("symbol", "UNKNOWN")).upper(),
                    "signal": refined["model_signal"],
                    "display_signal": refined["display_signal"],
                    "raw_model_signal": raw_signal,
                    "prediction_confidence": round(confidence, 4),
                    "confidence_level": confidence_level,
                },
                "summary": f"Signal model output: {refined['display_signal']} with {confidence_level.lower()} confidence.",
            }
            return result
        except Exception as exc:
            return self._fallback_signal(analysis_result, reason=str(exc))

    # Backward-compatible prediction aliases
    def generate_trading_signal(self, analysis_result, training_result=None, symbol=None):
        return self.generate_signal(analysis_result, training_result, symbol)

    def run_signal_model(self, analysis_result, training_result=None, symbol=None):
        return self.generate_signal(analysis_result, training_result, symbol)

    def predict(self, analysis_result, training_result=None, symbol=None):
        return self.generate_signal(analysis_result, training_result, symbol)

    def predict_signal(self, analysis_result, training_result=None, symbol=None):
        return self.generate_signal(analysis_result, training_result, symbol)
