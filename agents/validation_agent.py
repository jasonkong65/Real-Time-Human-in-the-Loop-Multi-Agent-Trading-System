from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


class ValidationAgent:
    """
    Validation Agent

    Role:
    - Check whether market data is reliable enough for downstream agents.
    - Compare primary and secondary sources.
    - Apply source reliability scoring.
    - Penalise delayed Alpha Vantage quotes when timestamps/dates lag behind.
    - Fallback to the primary source when sources disagree, but reduce confidence.
    - Load thresholds from a config file instead of hard-coding all values.

    This agent does not make trading decisions. It only decides whether the data
    is suitable for analysis and how cautious later agents should be.
    """

    DEFAULT_CONFIG: Dict[str, Any] = {
        "thresholds": {
            "high_confidence_price_diff": 0.01,
            "medium_confidence_price_diff": 0.03,
            "large_difference_price_diff": 0.08,
            "stale_date_threshold_days": 3,
        },
        "dynamic_thresholds": {
            "enabled": True,
            "medium_intraday_range": 0.025,
            "high_intraday_range": 0.04,
            "medium_range_multiplier": 1.25,
            "high_range_multiplier": 1.5,
        },
        "source_reliability": {
            "finnhub": 0.95,
            "alpha_vantage": 0.75,
            "yfinance": 0.80,
            "default_primary": 0.85,
            "default_secondary": 0.65,
        },
        "penalties": {
            "alpha_vantage_delayed_quote_penalty": 0.15,
            "date_gap_penalty_per_day": 0.06,
            "max_date_gap_penalty": 0.35,
            "single_secondary_source_penalty": 0.15,
            "large_price_difference_penalty": 0.20,
        },
        "actions": {
            "allow": "ALLOW_ANALYSIS",
            "caution": "ALLOW_ANALYSIS_WITH_CAUTION",
            "low_confidence": "ALLOW_ANALYSIS_WITH_LOW_CONFIDENCE",
            "block": "BLOCK_ANALYSIS",
        },
    }

    def __init__(
        self,
        config_path: str = "config/validation_config.json",
        high_confidence_threshold: Optional[float] = None,
        medium_confidence_threshold: Optional[float] = None,
        stale_date_threshold_days: Optional[int] = None,
        auto_create_config: bool = True,
    ):
        self.config_path = Path(config_path)
        self.config = self._load_config(auto_create_config=auto_create_config)

        # Backward-compatible manual overrides.
        if high_confidence_threshold is not None:
            self.config["thresholds"]["high_confidence_price_diff"] = float(high_confidence_threshold)

        if medium_confidence_threshold is not None:
            self.config["thresholds"]["medium_confidence_price_diff"] = float(medium_confidence_threshold)

        if stale_date_threshold_days is not None:
            self.config["thresholds"]["stale_date_threshold_days"] = int(stale_date_threshold_days)

        self.high_confidence_threshold = float(
            self.config["thresholds"]["high_confidence_price_diff"]
        )
        self.medium_confidence_threshold = float(
            self.config["thresholds"]["medium_confidence_price_diff"]
        )
        self.large_difference_threshold = float(
            self.config["thresholds"].get("large_difference_price_diff", 0.08)
        )
        self.stale_date_threshold_days = int(
            self.config["thresholds"].get("stale_date_threshold_days", 3)
        )

    # --------------------------------------------------
    # Config helpers
    # --------------------------------------------------
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        merged = deepcopy(base)

        for key, value in (override or {}).items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = self._deep_merge(merged[key], value)
            else:
                merged[key] = value

        return merged

    def _load_config(self, auto_create_config: bool = True) -> Dict[str, Any]:
        config = deepcopy(self.DEFAULT_CONFIG)

        if self.config_path.exists():
            try:
                with self.config_path.open("r", encoding="utf-8") as f:
                    user_config = json.load(f)
                config = self._deep_merge(config, user_config)
            except Exception:
                # Keep defaults if the config file is malformed.
                config = deepcopy(self.DEFAULT_CONFIG)

        elif auto_create_config:
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                with self.config_path.open("w", encoding="utf-8") as f:
                    json.dump(config, f, indent=2)
            except Exception:
                pass

        return config

    # --------------------------------------------------
    # Generic helpers
    # --------------------------------------------------
    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _clamp(self, value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    def _get_source(self, data: Dict[str, Any], names) -> Dict[str, Any]:
        for name in names:
            value = data.get(name)
            if isinstance(value, dict):
                return value
        return {}

    def _source_name(self, source: Dict[str, Any], fallback: str) -> str:
        raw = str(source.get("source") or source.get("provider") or fallback or "unknown")
        return raw.strip() or fallback

    def _normalised_source_key(self, source_name: str) -> str:
        name = (source_name or "").lower().replace(" ", "_").replace("-", "_")

        if "finnhub" in name:
            return "finnhub"
        if "alpha" in name or "vantage" in name:
            return "alpha_vantage"
        if "yfinance" in name or "yahoo" in name:
            return "yfinance"

        return name or "unknown"

    def _extract_price(self, source: Dict[str, Any]) -> Optional[float]:
        if not isinstance(source, dict):
            return None

        direct_keys = [
            "current_price",
            "price",
            "c",
            "close",
            "latest_price",
            "latestPrice",
            "05. price",
        ]

        for key in direct_keys:
            value = self._safe_float(source.get(key))
            if value is not None and value > 0:
                return value

        raw = source.get("raw_response") or source.get("raw")
        if isinstance(raw, dict):
            for key in ["c", "05. price", "price", "current_price"]:
                value = self._safe_float(raw.get(key))
                if value is not None and value > 0:
                    return value

            global_quote = raw.get("Global Quote")
            if isinstance(global_quote, dict):
                value = self._safe_float(global_quote.get("05. price"))
                if value is not None and value > 0:
                    return value

        return None

    def _valid_source(self, source: Dict[str, Any]) -> bool:
        if not isinstance(source, dict):
            return False
        price = self._extract_price(source)
        return bool(source.get("success")) and price is not None and price > 0

    def _extract_date_value(self, source: Dict[str, Any]) -> Optional[Any]:
        if not isinstance(source, dict):
            return None

        date_keys = [
            "latest_trading_day",
            "latestTradingDay",
            "date",
            "timestamp",
            "quote_timestamp",
            "source_timestamp",
            "t",
            "datetime",
        ]

        for key in date_keys:
            if source.get(key) is not None:
                return source.get(key)

        raw = source.get("raw_response") or source.get("raw")
        if isinstance(raw, dict):
            for key in ["07. latest trading day", "latest_trading_day", "t", "timestamp"]:
                if raw.get(key) is not None:
                    return raw.get(key)

            global_quote = raw.get("Global Quote")
            if isinstance(global_quote, dict):
                for key in ["07. latest trading day", "latest_trading_day"]:
                    if global_quote.get(key) is not None:
                        return global_quote.get(key)

        return None

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        if value is None:
            return None

        text = str(value).strip()

        if not text:
            return None

        try:
            # Finnhub timestamps are usually Unix seconds.
            if isinstance(value, (int, float)) or text.isdigit():
                number = float(text)
                if number > 10_000_000_000:
                    number = number / 1000.0
                return datetime.fromtimestamp(number, tz=timezone.utc)

            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)

        except Exception:
            try:
                parsed = datetime.strptime(text[:10], "%Y-%m-%d")
                return parsed.replace(tzinfo=timezone.utc)
            except Exception:
                return None

    def _date_gap_days(self, left: Optional[Any], right: Optional[Any]) -> Optional[int]:
        d1 = self._parse_datetime(left)
        d2 = self._parse_datetime(right)

        if not d1 or not d2:
            return None

        return abs((d1.date() - d2.date()).days)

    # --------------------------------------------------
    # Reliability / penalty logic
    # --------------------------------------------------
    def _base_source_reliability(self, source: Dict[str, Any], fallback_role: str) -> float:
        source_name = self._source_name(source, fallback_role)
        key = self._normalised_source_key(source_name)
        reliability_config = self.config.get("source_reliability", {})

        if key in reliability_config:
            return self._clamp(reliability_config[key])

        if fallback_role == "primary":
            return self._clamp(reliability_config.get("default_primary", 0.85))

        return self._clamp(reliability_config.get("default_secondary", 0.65))

    def _source_reliability_score(
        self,
        source: Dict[str, Any],
        fallback_role: str,
        date_gap: Optional[int] = None,
        is_secondary_alpha: bool = False,
    ) -> float:
        if not self._valid_source(source):
            return 0.0

        score = self._base_source_reliability(source, fallback_role)
        score -= self._timestamp_penalty(
            date_gap=date_gap,
            is_secondary_alpha=is_secondary_alpha,
        )

        return self._clamp(score)

    def _timestamp_penalty(self, date_gap: Optional[int], is_secondary_alpha: bool = False) -> float:
        if date_gap is None or date_gap <= 0:
            return 0.0

        penalties = self.config.get("penalties", {})
        penalty = min(
            float(penalties.get("max_date_gap_penalty", 0.35)),
            float(penalties.get("date_gap_penalty_per_day", 0.06)) * date_gap,
        )

        if is_secondary_alpha:
            penalty += float(penalties.get("alpha_vantage_delayed_quote_penalty", 0.15))

        return self._clamp(penalty)

    def _combined_reliability(self, primary_score: float, secondary_score: float) -> float:
        if primary_score and secondary_score:
            return self._clamp((0.65 * primary_score) + (0.35 * secondary_score))
        if primary_score:
            return self._clamp(primary_score)
        if secondary_score:
            return self._clamp(secondary_score)
        return 0.0

    def _dynamic_thresholds(self, primary: Dict[str, Any]) -> Tuple[float, float]:
        high = float(self.high_confidence_threshold)
        medium = float(self.medium_confidence_threshold)

        dynamic_config = self.config.get("dynamic_thresholds", {})
        if not dynamic_config.get("enabled", True):
            return high, medium

        high_price = self._safe_float(primary.get("high_price") or primary.get("h"))
        low_price = self._safe_float(primary.get("low_price") or primary.get("l"))
        price = self._extract_price(primary)

        if price and high_price and low_price and high_price > low_price:
            intraday_range = (high_price - low_price) / max(price, 1e-9)
            if intraday_range > float(dynamic_config.get("high_intraday_range", 0.04)):
                high *= float(dynamic_config.get("high_range_multiplier", 1.5))
                medium *= float(dynamic_config.get("high_range_multiplier", 1.5))
            elif intraday_range > float(dynamic_config.get("medium_intraday_range", 0.025)):
                high *= float(dynamic_config.get("medium_range_multiplier", 1.25))
                medium *= float(dynamic_config.get("medium_range_multiplier", 1.25))

        return high, medium

    # --------------------------------------------------
    # Result builder
    # --------------------------------------------------
    def _score_to_confidence_action(
        self,
        score: float,
        force_low_confidence: bool = False,
    ) -> Tuple[str, str]:
        actions = self.config.get("actions", {})

        if score <= 0:
            return "Low", actions.get("block", "BLOCK_ANALYSIS")

        if force_low_confidence:
            return "Low", actions.get("low_confidence", "ALLOW_ANALYSIS_WITH_LOW_CONFIDENCE")

        if score >= 0.80:
            return "High", actions.get("allow", "ALLOW_ANALYSIS")

        if score >= 0.55:
            return "Medium", actions.get("caution", "ALLOW_ANALYSIS_WITH_CAUTION")

        return "Low", actions.get("low_confidence", "ALLOW_ANALYSIS_WITH_LOW_CONFIDENCE")

    def _build_result(
        self,
        symbol: str,
        confidence: str,
        confidence_score: float,
        next_action: str,
        selected_price: Optional[float],
        selected_source: Optional[str],
        price_difference_pct: Optional[float],
        source_date_difference_days: Optional[int],
        primary_source_reliability: float,
        secondary_source_reliability: float,
        combined_source_reliability: float,
        timestamp_penalty: float,
        fallback_to_primary: bool,
        threshold_info: Dict[str, Any],
        issues: list,
        warnings: list,
        reasoning_steps: list,
    ) -> Dict[str, Any]:
        if next_action == "BLOCK_ANALYSIS":
            decision = "The data is not reliable enough for analysis."
        elif confidence == "High":
            decision = "The data is reliable enough to continue."
        elif confidence == "Medium":
            decision = "The data can be used, but later agents should stay cautious."
        else:
            decision = "The data can be used only as a low-confidence reference."

        confidence_score = self._clamp(confidence_score)

        return {
            "success": next_action != "BLOCK_ANALYSIS",
            "agent": "Validation Agent",
            "agent_goal": "Check market data quality before analysis.",
            "symbol": symbol,
            "confidence": confidence,
            "confidence_score": round(float(confidence_score), 4),
            "next_action": next_action,
            "selected_price": selected_price,
            "selected_source": selected_source,
            "price_difference_pct": None
            if price_difference_pct is None
            else round(float(price_difference_pct), 6),
            "source_date_difference_days": source_date_difference_days,
            "primary_source_reliability": round(float(primary_source_reliability), 4),
            "secondary_source_reliability": round(float(secondary_source_reliability), 4),
            "combined_source_reliability": round(float(combined_source_reliability), 4),
            "timestamp_penalty": round(float(timestamp_penalty), 4),
            "fallback_to_primary": bool(fallback_to_primary),
            "threshold_info": threshold_info,
            "config_path": str(self.config_path),
            "issues": issues,
            "warnings": warnings,
            "agent_decision": decision,
            "reasoning_steps": reasoning_steps,
            "validation_for_next_agent": {
                "symbol": symbol,
                "confidence": confidence,
                "confidence_score": round(float(confidence_score), 4),
                "next_action": next_action,
                "selected_price": selected_price,
                "selected_source": selected_source,
                "source_reliability_score": round(float(combined_source_reliability), 4),
                "timestamp_penalty": round(float(timestamp_penalty), 4),
                "fallback_to_primary": bool(fallback_to_primary),
            },
            "summary": f"Validation completed for {symbol}: {confidence} confidence.",
        }

    # --------------------------------------------------
    # Main validation method
    # --------------------------------------------------
    def validate_market_data(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        reasoning_steps, issues, warnings = [], [], []

        if not isinstance(multi_quote, dict):
            return self._build_result(
                symbol="UNKNOWN",
                confidence="Low",
                confidence_score=0.0,
                next_action="BLOCK_ANALYSIS",
                selected_price=None,
                selected_source=None,
                price_difference_pct=None,
                source_date_difference_days=None,
                primary_source_reliability=0.0,
                secondary_source_reliability=0.0,
                combined_source_reliability=0.0,
                timestamp_penalty=0.0,
                fallback_to_primary=False,
                threshold_info={},
                issues=["Input was not a dictionary."],
                warnings=[],
                reasoning_steps=["Validation stopped because the input was invalid."],
            )

        symbol = str(multi_quote.get("symbol", "UNKNOWN")).upper().strip()

        primary = self._get_source(multi_quote, ["finnhub", "primary", "finnhub_quote"])
        secondary = self._get_source(
            multi_quote,
            ["alpha_vantage", "secondary", "alpha_vantage_quote", "alphavantage"],
        )

        primary_valid = self._valid_source(primary)
        secondary_valid = self._valid_source(secondary)
        primary_price = self._extract_price(primary)
        secondary_price = self._extract_price(secondary)

        primary_source_name = self._source_name(primary, "Finnhub")
        secondary_source_name = self._source_name(secondary, "Alpha Vantage")

        primary_date_value = self._extract_date_value(primary)
        secondary_date_value = self._extract_date_value(secondary)
        date_gap = self._date_gap_days(primary_date_value, secondary_date_value)

        secondary_key = self._normalised_source_key(secondary_source_name)
        is_secondary_alpha = secondary_key == "alpha_vantage"

        primary_reliability = self._source_reliability_score(
            primary,
            fallback_role="primary",
            date_gap=0,
            is_secondary_alpha=False,
        )
        secondary_reliability = self._source_reliability_score(
            secondary,
            fallback_role="secondary",
            date_gap=date_gap,
            is_secondary_alpha=is_secondary_alpha,
        )
        combined_reliability = self._combined_reliability(primary_reliability, secondary_reliability)
        timestamp_penalty = self._timestamp_penalty(date_gap, is_secondary_alpha=is_secondary_alpha)

        if primary_valid:
            reasoning_steps.append(
                f"Primary source ({primary_source_name}) price found: {primary_price}."
            )
        else:
            warnings.append("Primary source was unavailable or invalid.")

        if secondary_valid:
            reasoning_steps.append(
                f"Secondary source ({secondary_source_name}) price found: {secondary_price}."
            )
        else:
            warnings.append("Secondary source was unavailable or invalid.")

        if date_gap is not None:
            reasoning_steps.append(f"Source date gap is {date_gap} day(s).")
            if date_gap > self.stale_date_threshold_days:
                warnings.append(
                    f"Source dates differ by {date_gap} days, so confidence is reduced."
                )

        if is_secondary_alpha and date_gap and date_gap > 0:
            warnings.append(
                "Alpha Vantage quote appears delayed relative to the primary source, so a timestamp penalty was applied."
            )

        if not primary_valid and not secondary_valid:
            issues.append("No valid price was found from either source.")
            return self._build_result(
                symbol=symbol,
                confidence="Low",
                confidence_score=0.0,
                next_action="BLOCK_ANALYSIS",
                selected_price=None,
                selected_source=None,
                price_difference_pct=None,
                source_date_difference_days=date_gap,
                primary_source_reliability=primary_reliability,
                secondary_source_reliability=secondary_reliability,
                combined_source_reliability=combined_reliability,
                timestamp_penalty=timestamp_penalty,
                fallback_to_primary=False,
                threshold_info={},
                issues=issues,
                warnings=warnings,
                reasoning_steps=reasoning_steps
                + ["Both sources failed or returned unusable prices."],
            )

        selected_price = None
        selected_source = None
        price_difference_pct = None
        fallback_to_primary = False
        force_low_confidence = False

        high_threshold, medium_threshold = self._dynamic_thresholds(primary)
        threshold_info = {
            "high_confidence_price_diff": round(high_threshold, 6),
            "medium_confidence_price_diff": round(medium_threshold, 6),
            "large_difference_price_diff": round(self.large_difference_threshold, 6),
            "threshold_source": "config + dynamic adjustment",
        }

        if primary_valid and secondary_valid and primary_price and secondary_price:
            price_difference_pct = abs(primary_price - secondary_price) / max(primary_price, 1e-9)
            reasoning_steps.append(
                f"Price difference between sources is {price_difference_pct:.2%}."
            )

            selected_price = primary_price
            selected_source = primary_source_name

            if price_difference_pct <= high_threshold:
                base_score = 0.96
                warnings_for_price_gap = False
            elif price_difference_pct <= medium_threshold:
                base_score = 0.76
                warnings_for_price_gap = True
                warnings.append("The two sources are close but not perfectly aligned.")
            else:
                fallback_to_primary = True
                force_low_confidence = True
                warnings_for_price_gap = True
                warnings.append(
                    "The two sources differ too much. The agent falls back to the primary source and reduces confidence."
                )
                base_score = 0.52
                base_score -= float(
                    self.config.get("penalties", {}).get("large_price_difference_penalty", 0.20)
                )

            reliability_factor = 0.70 + (0.30 * combined_reliability)
            score = base_score * reliability_factor
            score -= timestamp_penalty

            if warnings_for_price_gap:
                reasoning_steps.append("Validation used a cautious score because the sources were not fully aligned.")

        elif primary_valid:
            selected_price = primary_price
            selected_source = primary_source_name
            base_score = 0.72
            reliability_factor = 0.70 + (0.30 * primary_reliability)
            score = base_score * reliability_factor
            warnings.append("Only the primary source was available, so confidence is reduced.")
            reasoning_steps.append("Validation selected the primary source because the secondary source was unavailable.")

        else:
            selected_price = secondary_price
            selected_source = secondary_source_name
            base_score = 0.52
            reliability_factor = 0.70 + (0.30 * secondary_reliability)
            score = base_score * reliability_factor
            score -= float(
                self.config.get("penalties", {}).get("single_secondary_source_penalty", 0.15)
            )
            score -= timestamp_penalty
            force_low_confidence = True
            warnings.append("Only the secondary source was available, so confidence is low.")
            reasoning_steps.append("Validation selected the secondary source because the primary source was unavailable.")

        # Heavy date gap should not block automatically, but should reduce confidence.
        if date_gap is not None and date_gap > self.stale_date_threshold_days:
            score = min(score, 0.60)
            force_low_confidence = force_low_confidence or score < 0.55

        score = self._clamp(score)
        confidence, action = self._score_to_confidence_action(
            score,
            force_low_confidence=force_low_confidence,
        )

        reasoning_steps.append(
            f"Source reliability scores: primary={primary_reliability:.2f}, secondary={secondary_reliability:.2f}, combined={combined_reliability:.2f}."
        )
        reasoning_steps.append(f"Timestamp penalty applied: {timestamp_penalty:.2f}.")
        reasoning_steps.append(f"Final validation score: {score:.2f}.")

        return self._build_result(
            symbol=symbol,
            confidence=confidence,
            confidence_score=score,
            next_action=action,
            selected_price=selected_price,
            selected_source=selected_source,
            price_difference_pct=price_difference_pct,
            source_date_difference_days=date_gap,
            primary_source_reliability=primary_reliability,
            secondary_source_reliability=secondary_reliability,
            combined_source_reliability=combined_reliability,
            timestamp_penalty=timestamp_penalty,
            fallback_to_primary=fallback_to_primary,
            threshold_info=threshold_info,
            issues=issues,
            warnings=warnings,
            reasoning_steps=reasoning_steps,
        )

    # --------------------------------------------------
    # Backward-compatible aliases
    # --------------------------------------------------
    def validate_multi_source_data(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_multi_source(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_quotes(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def run(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_data(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_sources(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_market_quotes(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_multi_source_quote(self, multi_quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(multi_quote)

    def validate_quote(self, quote: Dict[str, Any]) -> Dict[str, Any]:
        return self.validate_market_data(
            {
                "symbol": quote.get("symbol", "UNKNOWN"),
                "finnhub": quote,
            }
        )
