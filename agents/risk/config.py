from __future__ import annotations

import json

import math

import random

import sqlite3

from datetime import datetime, timezone

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

import joblib

import pandas as pd

import torch

import torch.nn as nn

import torch.optim as optim

from .dqn import DQNNetwork


class RiskConfigMixin:


    def _load_runtime_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load optional DQN/risk settings from config/risk_config.json.
        This keeps the agent usable without manual code edits while preserving
        backward-compatible constructor arguments.
        """
        candidate_paths = []
        if config_path:
            candidate_paths.append(Path(config_path))
        candidate_paths.append(Path("config/risk_config.json"))

        for path in candidate_paths:
            try:
                if path.exists() and path.stat().st_size > 0:
                    with path.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                    return data if isinstance(data, dict) else {}
            except Exception:
                continue
        return {}


    def _cfg(self, config: Dict[str, Any], key: str, default: Any) -> Any:
        """Read one config value from either the top level or the dqn section."""
        try:
            if key in config:
                return config[key]
            dqn = config.get("dqn", {}) if isinstance(config, dict) else {}
            return dqn.get(key, default) if isinstance(dqn, dict) else default
        except Exception:
            return default

