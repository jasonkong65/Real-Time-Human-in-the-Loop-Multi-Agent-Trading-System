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

from .config import AnalystConfigMixin
from .features import AnalystFeatureMixin
from .market_context import AnalystMarketContextMixin
from .quote import AnalystQuoteMixin
from .historical import AnalystHistoricalMixin
from .combine import AnalystCombineMixin


class AnalystAgent(AnalystConfigMixin, AnalystFeatureMixin, AnalystMarketContextMixin, AnalystQuoteMixin, AnalystHistoricalMixin, AnalystCombineMixin):
    """Two-stage technical Analyst Agent.

Stage 1: Reads live quote movement.
Stage 2: Reads historical trend, momentum, volatility, RSI and volume.
Stage 3: Combines both stages with optional market-regime and sector-trend context.

Design goals:
- Keep trend direction separate from entry timing risk.
- Avoid treating a strong rising stock as a simple sell signal.
- Produce short, natural explanations and an indicator contribution table."""


    DEFAULT_CONFIG = {
            "score_weights": {
                "quote_score_weight": 0.35,
                "historical_score_weight": 0.65,
                "market_regime_adjustment_weight": 0.05,
                "sector_trend_adjustment_weight": 0.04,
            },
            "quote_scoring": {
                "strong_up_return": 0.02,
                "mild_up_return": 0.005,
                "strong_down_return": -0.02,
                "mild_down_return": -0.005,
                "open_up_return": 0.01,
                "open_down_return": -0.01,
                "high_intraday_range": 0.04,
                "medium_intraday_range": 0.02,
            },
            "historical_scoring": {
                "strong_return_20": 0.08,
                "positive_return_20": 0.02,
                "weak_return_20": -0.05,
                "slightly_negative_return_20": -0.02,
                "positive_return_5": 0.02,
                "negative_return_5": -0.02,
                "uptrend_ma_gap": 0.02,
                "downtrend_ma_gap": -0.02,
                "high_volatility_20": 0.04,
                "medium_volatility_20": 0.02,
                "overbought_rsi": 70,
                "strong_overbought_rsi": 78,
                "oversold_rsi": 30,
                "near_high_distance": 0.03,
                "stretched_return_20": 0.12,
                "high_volume_change": 0.2,
                "low_volume_change": -0.2,
            },
            "signal_thresholds": {
                "bullish_score": 0.70,
                "bearish_score": 0.35,
                "watchlist_positive_score": 0.60,
                "entry_risk_score": 2,
            },
            "market_context": {
                "enabled": True,
                "cache_ttl_hours": 6,
                "benchmark_symbol": "SPY",
                "period": "3mo",
                "interval": "1d",
            },
            "sector_etf_map": {},
        }


    def __init__(self, config_path: str = "config/analyst_config.json"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.cache_path = Path("data/cache/analyst_market_context.json")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

