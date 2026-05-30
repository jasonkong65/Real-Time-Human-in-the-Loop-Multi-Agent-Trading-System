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

from .helpers import TrainingHelpersMixin
from .features import TrainingFeatureMixin
from .selection import TrainingSelectionMixin
from .drift import TrainingDriftMixin
from .workflow import TrainingWorkflowMixin
from .signal import TrainingSignalMixin


class TrainingAgent(TrainingHelpersMixin, TrainingFeatureMixin, TrainingSelectionMixin, TrainingDriftMixin, TrainingWorkflowMixin, TrainingSignalMixin):
    """Training Agent

Builds and maintains a lightweight stock-signal model automatically.

What this version does:
- Uses walk-forward validation instead of one static 75/25 split.
- Tests several model candidates automatically.
- Can add pooled local historical data as extra training context.
- Selects models using balanced accuracy, macro F1, accuracy, and SELL_RISK recall.
- Saves confusion matrix, classification report, feature importance, and evaluation table.
- Only overwrites an existing model when the new model is meaningfully better.
- Keeps the core output labels expected by the current app.py:
  BUY_CANDIDATE, HOLD, SELL_RISK."""


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

