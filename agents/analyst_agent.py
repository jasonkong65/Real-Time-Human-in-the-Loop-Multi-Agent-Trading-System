import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from utils.features import build_trading_features
except Exception:
    build_trading_features = None


class AnalystAgent:
    """
    Two-stage technical Analyst Agent.

    Stage 1: Reads live quote movement.
    Stage 2: Reads historical trend, momentum, volatility, RSI and volume.
    Stage 3: Combines both stages with optional market-regime and sector-trend context.

    Design goals:
    - Keep trend direction separate from entry timing risk.
    - Avoid treating a strong rising stock as a simple sell signal.
    - Produce short, natural explanations and an indicator contribution table.
    """

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

    # ------------------------------------------------------------------
    # Config and safe helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Historical feature helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Indicator contribution helpers
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Market regime and sector trend
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Stage 1: Quote-level analysis
    # ------------------------------------------------------------------
    def analyse_quote_level(self, multi_quote: dict, validation_result: dict) -> dict:
        agent_goal = "Read the latest quote and score short-term price action."

        if validation_result.get("next_action") == "BLOCK_ANALYSIS":
            return {
                "success": False,
                "stage": "Stage 1: Quote-level Analysis",
                "agent_goal": agent_goal,
                "agent_decision": "Data quality is too weak for quote analysis.",
                "summary": "Quote analysis was skipped because validation blocked it.",
                "quote_features": None,
                "reasoning_steps": [],
                "indicator_contributions": [],
            }

        symbol = str(multi_quote.get("symbol") or validation_result.get("symbol") or "UNKNOWN").upper()
        primary = self._source(multi_quote, ["finnhub", "primary", "finnhub_quote"])
        secondary = self._source(multi_quote, ["alpha_vantage", "secondary", "alpha_vantage_quote"])
        quote = primary if primary.get("success") else secondary

        selected_price = self._safe_float(validation_result.get("selected_price"))
        current = self._price(quote, "current_price") or selected_price
        previous_close = self._price(quote, "previous_close")
        open_price = self._price(quote, "open_price")
        high_price = self._price(quote, "high_price")
        low_price = self._price(quote, "low_price")

        if current is None or previous_close is None or previous_close <= 0:
            return {
                "success": False,
                "stage": "Stage 1: Quote-level Analysis",
                "agent_goal": agent_goal,
                "symbol": symbol,
                "agent_decision": "Key quote fields are missing.",
                "summary": "Quote analysis could not run because current price or previous close is missing.",
                "quote_features": None,
                "reasoning_steps": [],
                "indicator_contributions": [],
            }

        quote_cfg = self.config.get("quote_scoring", {})
        confidence_score = self._confidence_score(validation_result)
        daily_return = (current - previous_close) / previous_close
        open_return = (current - open_price) / open_price if open_price and open_price > 0 else None
        intraday_range = (high_price - low_price) / current if current and high_price and low_price and high_price >= low_price else None

        score = 0.50
        contributions: List[Dict[str, Any]] = []
        reasons: List[str] = []

        if daily_return > quote_cfg.get("strong_up_return", 0.02):
            delta = 0.20
            message = "Strong day so far."
            quote_trend = "Strong upward"
        elif daily_return > quote_cfg.get("mild_up_return", 0.005):
            delta = 0.10
            message = "Price is modestly higher."
            quote_trend = "Slight upward"
        elif daily_return < quote_cfg.get("strong_down_return", -0.02):
            delta = -0.20
            message = "Strong down day."
            quote_trend = "Strong downward"
        elif daily_return < quote_cfg.get("mild_down_return", -0.005):
            delta = -0.10
            message = "Price is modestly lower."
            quote_trend = "Slight downward"
        else:
            delta = 0.0
            message = "Price is close to the previous close."
            quote_trend = "Neutral"
        score += delta
        reasons.append(message)
        contributions.append(self._contribution("Daily return", self._format_pct(daily_return), delta, message, "quote"))

        if open_return is not None:
            if open_return > quote_cfg.get("open_up_return", 0.01):
                delta = 0.10
                message = "Price is above the open."
            elif open_return < quote_cfg.get("open_down_return", -0.01):
                delta = -0.10
                message = "Price is below the open."
            else:
                delta = 0.0
                message = "Open-to-current move is small."
            score += delta
            reasons.append(message)
            contributions.append(self._contribution("Open-to-current return", self._format_pct(open_return), delta, message, "quote"))

        if intraday_range is None:
            quote_volatility_level = "Unknown"
            delta = 0.0
            message = "Intraday range is unavailable."
        elif intraday_range > quote_cfg.get("high_intraday_range", 0.04):
            quote_volatility_level = "High"
            delta = -0.10
            message = "Intraday movement is wide."
        elif intraday_range > quote_cfg.get("medium_intraday_range", 0.02):
            quote_volatility_level = "Medium"
            delta = -0.03
            message = "Intraday movement is moderate."
        else:
            quote_volatility_level = "Low"
            delta = 0.05
            message = "Intraday movement is calm."
        score += delta
        reasons.append(message)
        contributions.append(self._contribution("Intraday range", self._format_pct(intraday_range), delta, message, "quote"))

        raw_score = self._clip(score)
        adjusted_score = self._clip(raw_score * (0.85 + 0.15 * confidence_score))
        contributions.append(self._contribution("Validation confidence", round(confidence_score, 3), adjusted_score - raw_score, "Data confidence adjusted the quote score.", "data_quality"))

        if adjusted_score >= 0.70:
            quote_signal = "QUOTE_BULLISH"
            decision = "Live quote is supportive."
        elif adjusted_score <= 0.35:
            quote_signal = "QUOTE_BEARISH"
            decision = "Live quote is weak."
        elif quote_volatility_level == "High":
            quote_signal = "QUOTE_HIGH_VOLATILITY"
            decision = "Live quote is volatile."
        else:
            quote_signal = "QUOTE_NEUTRAL"
            decision = "Live quote is mixed."

        quote_features = {
            "daily_return": daily_return,
            "open_to_current_return": open_return,
            "intraday_range_pct": intraday_range,
            "quote_score": adjusted_score,
            "quote_trend": quote_trend,
            "quote_volatility_level": quote_volatility_level,
            "quote_signal": quote_signal,
        }

        return {
            "success": True,
            "stage": "Stage 1: Quote-level Analysis",
            "agent_goal": agent_goal,
            "symbol": symbol,
            "selected_price": current,
            "daily_return": round(daily_return, 6),
            "daily_return_pct": self._format_pct(daily_return),
            "open_to_current_return": round(open_return, 6) if open_return is not None else None,
            "open_to_current_return_pct": self._format_pct(open_return),
            "intraday_range_pct": round(intraday_range, 6) if intraday_range is not None else None,
            "intraday_range_pct_text": self._format_pct(intraday_range),
            "quote_trend": quote_trend,
            "quote_volatility_level": quote_volatility_level,
            "quote_score": round(adjusted_score, 3),
            "quote_signal": quote_signal,
            "agent_decision": decision,
            "reasoning_steps": ["Checked live price move, open-to-current move, intraday range, and data confidence."],
            "indicator_contributions": self._top_contributions(contributions, 5),
            "quote_features": quote_features,
            "summary": f"{symbol}: live quote looks {quote_signal.replace('QUOTE_', '').lower()}.",
        }

    # ------------------------------------------------------------------
    # Stage 2: Historical analysis
    # ------------------------------------------------------------------
    def analyse_historical(self, multi_quote: dict, validation_result: dict, historical_data: dict) -> dict:
        agent_goal = "Read historical trend, momentum, volatility, RSI and volume."

        if not isinstance(historical_data, dict) or not historical_data.get("success"):
            return {
                "success": False,
                "stage": "Stage 2: Historical Analysis",
                "agent_goal": agent_goal,
                "agent_decision": "Historical data is unavailable.",
                "summary": historical_data.get("error", "Historical data request failed.") if isinstance(historical_data, dict) else "Historical data request failed.",
                "historical_features": None,
                "reasoning_steps": [],
                "indicator_contributions": [],
            }

        symbol = str(multi_quote.get("symbol") or historical_data.get("symbol") or "UNKNOWN").upper()
        historical_source = historical_data.get("source", "historical data")
        price_records = historical_data.get("prices", [])
        feature_df = self._build_features(price_records)

        if feature_df.empty:
            return {
                "success": False,
                "stage": "Stage 2: Historical Analysis",
                "agent_goal": agent_goal,
                "agent_decision": "Historical features could not be built.",
                "summary": "Feature construction failed.",
                "historical_features": None,
                "reasoning_steps": [],
                "indicator_contributions": [],
            }

        latest = feature_df.iloc[-1]
        hcfg = self.config.get("historical_scoring", {})
        confidence_score = self._confidence_score(validation_result)

        return_1 = self._safe_float(latest.get("return_1"), 0.0) or 0.0
        return_5 = self._safe_float(latest.get("return_5"), 0.0) or 0.0
        return_20 = self._safe_float(latest.get("return_20"), 0.0) or 0.0
        ma_gap = self._safe_float(latest.get("ma_gap"), 0.0) or 0.0
        volatility_20 = self._safe_float(latest.get("volatility_20"), 0.0) or 0.0
        volume_change = self._safe_float(latest.get("volume_change"), 0.0) or 0.0
        rsi_14 = self._safe_float(latest.get("rsi_14"), 50.0) or 50.0

        score = 0.50
        contributions: List[Dict[str, Any]] = []
        reasons: List[str] = []

        if return_20 > hcfg.get("strong_return_20", 0.08):
            delta = 0.16
            momentum_level = "Strong Positive"
            message = "20-day momentum is strong."
        elif return_20 > hcfg.get("positive_return_20", 0.02):
            delta = 0.09
            momentum_level = "Positive"
            message = "20-day momentum is positive."
        elif return_20 < hcfg.get("weak_return_20", -0.05):
            delta = -0.16
            momentum_level = "Negative"
            message = "20-day momentum is weak."
        elif return_20 < hcfg.get("slightly_negative_return_20", -0.02):
            delta = -0.08
            momentum_level = "Slight Negative"
            message = "20-day momentum is slightly weak."
        else:
            delta = 0.0
            momentum_level = "Mixed"
            message = "20-day momentum is mixed."
        score += delta
        reasons.append(message)
        contributions.append(self._contribution("20-day return", self._format_pct(return_20), delta, message, "historical"))

        if return_5 > hcfg.get("positive_return_5", 0.02):
            delta = 0.07
            short_momentum = "Positive"
            message = "5-day momentum is positive."
        elif return_5 < hcfg.get("negative_return_5", -0.02):
            delta = -0.07
            short_momentum = "Negative"
            message = "5-day momentum is negative."
        else:
            delta = 0.0
            short_momentum = "Mixed"
            message = "5-day momentum is mixed."
        score += delta
        contributions.append(self._contribution("5-day return", self._format_pct(return_5), delta, message, "historical"))

        if ma_gap > hcfg.get("uptrend_ma_gap", 0.02):
            delta = 0.10
            historical_trend = "Uptrend"
            trend_direction = "Positive"
            message = "Short-term average is above the medium-term average."
        elif ma_gap < hcfg.get("downtrend_ma_gap", -0.02):
            delta = -0.10
            historical_trend = "Downtrend"
            trend_direction = "Negative"
            message = "Short-term average is below the medium-term average."
        else:
            delta = 0.0
            historical_trend = "Sideways"
            trend_direction = "Neutral"
            message = "Moving-average gap is small."
        score += delta
        contributions.append(self._contribution("Moving-average gap", self._format_pct(ma_gap), delta, message, "historical"))

        if volatility_20 > hcfg.get("high_volatility_20", 0.04):
            delta = -0.08
            historical_volatility_level = "High"
            volatility_risk_level = "High"
            message = "Recent volatility is high."
        elif volatility_20 > hcfg.get("medium_volatility_20", 0.02):
            delta = -0.02
            historical_volatility_level = "Medium"
            volatility_risk_level = "Medium"
            message = "Recent volatility is moderate."
        else:
            delta = 0.04
            historical_volatility_level = "Low"
            volatility_risk_level = "Low"
            message = "Recent volatility is low."
        score += delta
        contributions.append(self._contribution("20-day volatility", self._format_pct(volatility_20), delta, message, "risk"))

        entry_risk_points = 0
        if rsi_14 >= hcfg.get("strong_overbought_rsi", 78):
            delta = -0.08
            rsi_signal = "Strongly Overbought"
            entry_risk_points += 2
            message = "RSI is very high."
        elif rsi_14 >= hcfg.get("overbought_rsi", 70):
            delta = -0.04
            rsi_signal = "Overbought"
            entry_risk_points += 1
            message = "RSI is high."
        elif rsi_14 <= hcfg.get("oversold_rsi", 30):
            delta = 0.04
            rsi_signal = "Oversold"
            entry_risk_points += 1
            message = "RSI is low."
        else:
            delta = 0.0
            rsi_signal = "Neutral"
            message = "RSI is neutral."
        score += delta
        contributions.append(self._contribution("RSI 14", round(rsi_14, 2), delta, message, "risk"))

        # Distance to recent high is calculated directly from price records.
        distance_to_high = None
        try:
            price_df = pd.DataFrame(price_records)
            price_df.columns = [str(c).lower().strip().replace(" ", "_") for c in price_df.columns]
            price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
            close_tail = price_df["close"].dropna().tail(60)
            if len(close_tail) >= 5:
                recent_high = close_tail.max()
                last_close = close_tail.iloc[-1]
                distance_to_high = (recent_high - last_close) / recent_high if recent_high else None
        except Exception:
            distance_to_high = None

        if distance_to_high is not None and distance_to_high <= hcfg.get("near_high_distance", 0.03) and return_20 > 0:
            entry_risk_points += 1
            contributions.append(self._contribution("Distance to recent high", self._format_pct(distance_to_high), -0.02, "Price is close to its recent high.", "risk"))
        if return_20 > hcfg.get("stretched_return_20", 0.12):
            entry_risk_points += 1
            contributions.append(self._contribution("Stretch risk", self._format_pct(return_20), -0.02, "The recent move may be stretched.", "risk"))

        if volume_change > hcfg.get("high_volume_change", 0.2):
            delta = 0.05
            message = "Volume is above recent average."
        elif volume_change < hcfg.get("low_volume_change", -0.2):
            delta = -0.03
            message = "Volume is below recent average."
        else:
            delta = 0.0
            message = "Volume is close to normal."
        score += delta
        contributions.append(self._contribution("Volume change", self._format_pct(volume_change), delta, message, "historical"))

        raw_score = self._clip(score)
        historical_score = self._clip(raw_score * (0.90 + 0.10 * confidence_score))
        contributions.append(self._contribution("Validation confidence", round(confidence_score, 3), historical_score - raw_score, "Data confidence adjusted the historical score.", "data_quality"))

        if entry_risk_points >= 3:
            entry_risk_level = "High"
        elif entry_risk_points >= 2:
            entry_risk_level = "Elevated"
        elif entry_risk_points >= 1:
            entry_risk_level = "Moderate"
        else:
            entry_risk_level = "Low"

        if trend_direction == "Positive" and return_5 > 0:
            trend_direction = "Positive"
        elif trend_direction == "Neutral" and (return_5 > 0.01 or return_20 > 0.03):
            trend_direction = "Mild Positive"
        elif trend_direction == "Neutral" and (return_5 < -0.01 or return_20 < -0.03):
            trend_direction = "Mild Negative"

        thresholds = self.config.get("signal_thresholds", {})
        if historical_score >= thresholds.get("bullish_score", 0.70):
            historical_signal = "BULLISH_HISTORY"
            decision = "Historical trend is positive."
        elif historical_score <= thresholds.get("bearish_score", 0.35):
            historical_signal = "BEARISH_HISTORY"
            decision = "Historical trend is weak."
        elif historical_volatility_level == "High":
            historical_signal = "HIGH_VOLATILITY_HISTORY"
            decision = "Historical trend is mixed with high volatility."
        else:
            historical_signal = "NEUTRAL_HISTORY"
            decision = "Historical trend is mixed."

        historical_features = {
            "return_1": return_1,
            "return_5": return_5,
            "return_20": return_20,
            "ma_gap": ma_gap,
            "volatility_20": volatility_20,
            "volume_change": volume_change,
            "rsi_14": rsi_14,
            "validation_confidence_score": confidence_score,
        }

        return {
            "success": True,
            "stage": "Stage 2: Historical Analysis",
            "agent_goal": agent_goal,
            "symbol": symbol,
            "historical_source": historical_source,
            "latest_feature_date": str(latest.get("date", "Unknown")),
            "return_1": round(return_1, 6),
            "return_1_pct": self._format_pct(return_1),
            "return_5": round(return_5, 6),
            "return_5_pct": self._format_pct(return_5),
            "return_20": round(return_20, 6),
            "return_20_pct": self._format_pct(return_20),
            "ma_gap": round(ma_gap, 6),
            "ma_gap_pct": self._format_pct(ma_gap),
            "volatility_20": round(volatility_20, 6),
            "volatility_20_pct": self._format_pct(volatility_20),
            "volume_change": round(volume_change, 6),
            "volume_change_pct": self._format_pct(volume_change),
            "rsi_14": round(rsi_14, 3),
            "momentum_level": momentum_level,
            "short_momentum": short_momentum,
            "historical_trend": historical_trend,
            "trend_direction": trend_direction,
            "historical_volatility_level": historical_volatility_level,
            "volatility_risk_level": volatility_risk_level,
            "rsi_signal": rsi_signal,
            "entry_risk_level": entry_risk_level,
            "entry_risk_points": entry_risk_points,
            "distance_to_recent_high": round(distance_to_high, 6) if distance_to_high is not None else None,
            "historical_score": round(historical_score, 3),
            "historical_signal": historical_signal,
            "agent_decision": decision,
            "historical_features": historical_features,
            "indicator_contributions": self._top_contributions(contributions, 8),
            "reasoning_steps": ["Calculated momentum, moving-average gap, volatility, RSI, volume and entry timing risk."],
            "summary": f"{symbol}: {trend_direction.lower()} trend, {entry_risk_level.lower()} entry risk.",
        }

    # ------------------------------------------------------------------
    # Stage 3: Combine quote, historical, market and sector context
    # ------------------------------------------------------------------
    def combine_analysis(self, multi_quote: dict, validation_result: dict, quote_result: dict, historical_result: dict) -> dict:
        agent_goal = "Combine quote, historical, market and sector context into one analyst view."
        symbol = str(multi_quote.get("symbol") or historical_result.get("symbol") or quote_result.get("symbol") or "UNKNOWN").upper()
        selected_price = self._safe_float(validation_result.get("selected_price"))
        selected_source = validation_result.get("selected_source")

        if not quote_result.get("success") and not historical_result.get("success"):
            return {
                "success": False,
                "agent_goal": agent_goal,
                "agent_decision": "No analyst view was generated because both quote and historical analysis failed.",
                "summary": "Analyst Agent failed.",
                "features_for_model": None,
                "analysis_for_next_agent": None,
                "stage_1_quote_analysis": quote_result,
                "stage_2_historical_analysis": historical_result,
            }

        quote_score = self._safe_float(quote_result.get("quote_score")) if quote_result.get("success") else None
        historical_score = self._safe_float(historical_result.get("historical_score")) if historical_result.get("success") else None
        weights = self.config.get("score_weights", {})
        quote_weight = float(weights.get("quote_score_weight", 0.35))
        hist_weight = float(weights.get("historical_score_weight", 0.65))

        if quote_score is not None and historical_score is not None:
            total = quote_weight + hist_weight
            final_score = (quote_weight * quote_score + hist_weight * historical_score) / total
            analysis_mode = "quote_and_historical"
        elif historical_score is not None:
            final_score = historical_score
            analysis_mode = "historical_only"
        else:
            final_score = quote_score if quote_score is not None else 0.50
            analysis_mode = "quote_only"

        context = self._market_context(symbol)
        context_contributions: List[Dict[str, Any]] = []
        market_adj, market_contrib = self._context_adjustment(
            context.get("market_regime", {}),
            float(weights.get("market_regime_adjustment_weight", 0.05)),
        )
        sector_adj, sector_contrib = self._context_adjustment(
            context.get("sector_trend", {}),
            float(weights.get("sector_trend_adjustment_weight", 0.04)),
        )
        final_score = self._clip(final_score + market_adj + sector_adj)
        context_contributions.extend([market_contrib, sector_contrib])

        if historical_result.get("success"):
            features_for_model = historical_result.get("historical_features")
            trend = historical_result.get("historical_trend")
            trend_direction = historical_result.get("trend_direction", trend)
            volatility_level = historical_result.get("historical_volatility_level")
            volatility_risk_level = historical_result.get("volatility_risk_level", volatility_level)
            rsi_signal = historical_result.get("rsi_signal")
            momentum_level = historical_result.get("momentum_level")
            entry_risk_level = historical_result.get("entry_risk_level", "Medium")
            entry_risk_points = historical_result.get("entry_risk_points", 0)
        else:
            quote_features = quote_result.get("quote_features", {})
            features_for_model = {
                "return_1": quote_features.get("daily_return"),
                "return_5": None,
                "return_20": None,
                "ma_gap": None,
                "volatility_20": quote_features.get("intraday_range_pct"),
                "volume_change": None,
                "rsi_14": None,
                "validation_confidence_score": self._confidence_score(validation_result),
            }
            trend = quote_result.get("quote_trend")
            trend_direction = "Positive" if "up" in str(trend).lower() else "Negative" if "down" in str(trend).lower() else "Neutral"
            volatility_level = quote_result.get("quote_volatility_level")
            volatility_risk_level = volatility_level
            rsi_signal = "Unknown"
            momentum_level = "Quote-only"
            entry_risk_level = "Medium"
            entry_risk_points = 1

        thresholds = self.config.get("signal_thresholds", {})
        bullish_threshold = thresholds.get("bullish_score", 0.70)
        bearish_threshold = thresholds.get("bearish_score", 0.35)
        watchlist_threshold = thresholds.get("watchlist_positive_score", 0.60)
        entry_risk_trigger = thresholds.get("entry_risk_score", 2)

        positive_trend = trend_direction in ["Strong Positive", "Positive", "Mild Positive"] or trend in ["Uptrend", "Strong upward", "Slight upward"]
        negative_trend = trend_direction in ["Negative", "Mild Negative"] or trend in ["Downtrend", "Strong downward", "Slight downward"]
        high_entry_risk = entry_risk_points >= entry_risk_trigger or entry_risk_level in ["Elevated", "High"]

        if positive_trend and high_entry_risk:
            analyst_signal = "POSITIVE_BUT_ENTRY_RISK"
            display_signal = "WATCHLIST_BULLISH_ENTRY_RISK"
            decision = "Positive setup, but entry timing risk is elevated."
        elif final_score >= bullish_threshold and not high_entry_risk:
            analyst_signal = "BULLISH_WATCH"
            display_signal = "BULLISH_WATCHLIST"
            decision = "Positive setup for the watchlist."
        elif final_score >= watchlist_threshold and positive_trend:
            analyst_signal = "WATCHLIST_BULLISH"
            display_signal = "BULLISH_WATCHLIST"
            decision = "Positive but not a direct entry signal."
        elif final_score <= bearish_threshold or negative_trend:
            analyst_signal = "BEARISH_RISK"
            display_signal = "BEARISH_RISK"
            decision = "Weak setup or downside risk."
        elif volatility_risk_level == "High":
            analyst_signal = "HIGH_VOLATILITY_CAUTION"
            display_signal = "HIGH_VOLATILITY_CAUTION"
            decision = "Mixed setup with high volatility."
        else:
            analyst_signal = "NEUTRAL"
            display_signal = "NEUTRAL_MONITOR"
            decision = "Mixed setup. Monitor only."

        quote_contribs = quote_result.get("indicator_contributions", []) if quote_result.get("success") else []
        hist_contribs = historical_result.get("indicator_contributions", []) if historical_result.get("success") else []
        all_contribs = self._top_contributions(quote_contribs + hist_contribs + context_contributions, 10)

        summary = f"{symbol}: {decision} Score {final_score:.2f}."
        reasoning_steps = [
            f"Used {analysis_mode} with configurable weights: quote={quote_weight}, historical={hist_weight}.",
            f"Market regime: {context.get('market_regime', {}).get('trend', 'Unknown')}.",
            f"Sector trend: {context.get('sector_trend', {}).get('trend', 'Unknown')}.",
            f"Final analyst signal: {analyst_signal}.",
        ]

        result = {
            "success": True,
            "agent_goal": agent_goal,
            "symbol": symbol,
            "selected_price": selected_price,
            "selected_source": selected_source,
            "analysis_mode": analysis_mode,
            "quote_score": round(quote_score, 3) if quote_score is not None else None,
            "historical_score": round(historical_score, 3) if historical_score is not None else None,
            "analyst_score": round(final_score, 3),
            "analyst_signal": analyst_signal,
            "display_signal": display_signal,
            "trend": trend,
            "trend_direction": trend_direction,
            "momentum_level": momentum_level,
            "volatility_level": volatility_level,
            "volatility_risk_level": volatility_risk_level,
            "rsi_signal": rsi_signal,
            "entry_risk_level": entry_risk_level,
            "entry_risk_points": entry_risk_points,
            "market_regime": context.get("market_regime"),
            "sector_trend": context.get("sector_trend"),
            "agent_decision": decision,
            "indicator_contributions": all_contribs,
            "reasoning_steps": reasoning_steps,
            "features_for_model": features_for_model,
            "analysis_for_next_agent": {
                "symbol": symbol,
                "selected_price": selected_price,
                "analyst_score": round(final_score, 3),
                "analyst_signal": analyst_signal,
                "display_signal": display_signal,
                "trend": trend,
                "trend_direction": trend_direction,
                "momentum_level": momentum_level,
                "volatility_level": volatility_level,
                "volatility_risk_level": volatility_risk_level,
                "rsi_signal": rsi_signal,
                "entry_risk_level": entry_risk_level,
                "entry_risk_points": entry_risk_points,
                "market_regime": context.get("market_regime"),
                "sector_trend": context.get("sector_trend"),
                "features_for_model": features_for_model,
                "analysis_mode": analysis_mode,
                "indicator_contributions": all_contribs,
            },
            "stage_1_quote_analysis": quote_result,
            "stage_2_historical_analysis": historical_result,
            "summary": summary,
        }
        return result

    # ------------------------------------------------------------------
    # Public methods used by app.py
    # ------------------------------------------------------------------
    def analyse_market(self, multi_quote: dict, validation_result: dict, historical_data: dict) -> dict:
        quote_result = self.analyse_quote_level(multi_quote, validation_result)
        historical_result = self.analyse_historical(multi_quote, validation_result, historical_data)
        return self.combine_analysis(multi_quote, validation_result, quote_result, historical_result)

    def analyze_market(self, *args, **kwargs) -> dict:
        return self.analyse_market(*args, **kwargs)

    def analyse(self, *args, **kwargs) -> dict:
        return self.analyse_market(*args, **kwargs)

    def analyze(self, *args, **kwargs) -> dict:
        return self.analyse_market(*args, **kwargs)

    def run(self, *args, **kwargs) -> dict:
        return self.analyse_market(*args, **kwargs)
