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


class AnalystCombineMixin:
    """Mixin for combining quote and historical analysis results with market context to produce a final analyst view and signal."""

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

