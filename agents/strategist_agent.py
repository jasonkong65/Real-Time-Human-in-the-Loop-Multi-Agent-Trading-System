import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone


class StrategistAgent:
    """
    Strategist Agent

    Converts the risk-controlled signal into a practical, cautious research plan.

    New features:
    - Optional portfolio context: current paper holding, exposure, average cost.
    - Optional event context: earnings date, days to earnings, event risk.
    - Strategy confidence score.
    - Action checklist.
    - Template-driven wording from config/strategist_templates.json.

    This agent does not execute real trades. It only creates paper decision-support guidance.
    """

    DEFAULT_CONFIG_PATH = "config/strategist_templates.json"

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path or self.DEFAULT_CONFIG_PATH)
        self.templates = self._load_templates()

    # --------------------------------------------------
    # Config and templates
    # --------------------------------------------------
    def _load_templates(self) -> Dict[str, Any]:
        default_templates = {
            "messages": {
                "NO_ACTION_DATA_OR_RISK_BLOCK": {
                    "position_guidance": "Do not make a paper decision from this result. Wait for better data or lower risk.",
                    "leverage_guidance": "Do not use leverage. The data or risk gate is blocking this setup.",
                    "watchlist_status": "No action",
                },
                "RISK_REDUCTION_REVIEW": {
                    "position_guidance": "Review downside risk. If this is already in a paper portfolio, check position size before adding exposure.",
                    "leverage_guidance": "Do not use leverage. The risk layer is defensive.",
                    "watchlist_status": "High caution watchlist",
                },
                "WAIT_FOR_PULLBACK_OR_CONFIRMATION": {
                    "position_guidance": "The setup looks positive, but entry timing risk is elevated. Wait for a pullback or stronger confirmation before any paper entry.",
                    "leverage_guidance": "Do not use leverage. The setup still has timing risk.",
                    "watchlist_status": "Bullish watchlist with entry risk",
                },
                "WAIT_FOR_CONFIRMATION": {
                    "position_guidance": "Keep it on the watchlist. Wait for stronger confirmation before treating it as a paper-entry candidate.",
                    "leverage_guidance": "Do not use leverage. Confirmation is not strong enough yet.",
                    "watchlist_status": "Watchlist for confirmation",
                },
                "RESEARCH_FOR_POSSIBLE_ENTRY": {
                    "position_guidance": "This can be researched as a possible paper-entry candidate. Check valuation, news, and portfolio exposure first.",
                    "leverage_guidance": "No leverage is assumed in this prototype. Keep the decision as paper research only.",
                    "watchlist_status": "Candidate for further research",
                },
                "MONITOR_POSITIVE_SETUP": {
                    "position_guidance": "Monitor the positive setup. It is not a direct entry signal yet.",
                    "leverage_guidance": "Do not use leverage. The signal is still in monitor mode.",
                    "watchlist_status": "Positive monitor list",
                },
                "MONITOR_AND_RESEARCH": {
                    "position_guidance": "Maintain a monitoring stance and collect more evidence before making a paper decision.",
                    "leverage_guidance": "Do not use leverage. The setup is not strong enough for a paper-entry signal.",
                    "watchlist_status": "Monitor list",
                },
            },
            "checklist": {
                "base": [
                    "Check the latest company news.",
                    "Check valuation and earnings context.",
                    "Re-run the pipeline after the next market data refresh.",
                ],
                "entry_risk": [
                    "Check whether RSI or entry-risk level has cooled down.",
                    "Check whether price has pulled back toward a short-term moving average.",
                ],
                "event": [
                    "Check the next earnings date before making any paper decision.",
                    "Avoid over-interpreting signals immediately before major company events.",
                ],
                "portfolio": [
                    "Check current paper exposure before adding more risk.",
                    "Check whether the position is already too large for the paper portfolio.",
                ],
                "risk": [
                    "Check why the Risk Agent raised caution.",
                    "Wait until risk level improves before increasing paper exposure.",
                ],
            },
        }

        try:
            if self.config_path.exists():
                with self.config_path.open("r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    default_templates.update(loaded)
        except Exception:
            pass

        return default_templates

    def _template_for(self, action: str) -> Dict[str, str]:
        messages = self.templates.get("messages", {})
        return messages.get(action, messages.get("MONITOR_AND_RESEARCH", {}))

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

    def _safe_int(self, value: Any, default: Optional[int] = None) -> Optional[int]:
        try:
            if value is None or value == "":
                return default
            return int(float(value))
        except Exception:
            return default

    def _get_nested(self, data: Dict[str, Any], keys: List[str], default=None):
        current = data
        for key in keys:
            if not isinstance(current, dict):
                return default
            current = current.get(key)
            if current is None:
                return default
        return current

    def _norm(self, value: Any, default: str = "Unknown") -> str:
        if value is None:
            return default
        value = str(value).strip().upper()
        return value if value else default

    def _title(self, value: Any, default: str = "Unknown") -> str:
        if value is None:
            return default
        value = str(value).replace("_", " ").strip()
        if not value:
            return default
        return value.title()

    def _symbol(self, *items: Dict[str, Any]) -> str:
        for item in items:
            if not isinstance(item, dict):
                continue
            direct = item.get("symbol")
            nested = (
                self._get_nested(item, ["risk_for_next_agent", "symbol"])
                or self._get_nested(item, ["signal_for_next_agent", "symbol"])
                or self._get_nested(item, ["validation_for_next_agent", "symbol"])
            )
            symbol = direct or nested
            if symbol:
                return str(symbol).upper().strip()
        return "UNKNOWN"

    # --------------------------------------------------
    # Extract context
    # --------------------------------------------------
    def _final_signal(self, risk_result: Dict[str, Any], signal_result: Dict[str, Any]) -> str:
        return self._norm(
            risk_result.get("final_signal")
            or self._get_nested(risk_result, ["risk_for_next_agent", "final_signal"])
            or signal_result.get("model_signal"),
            "HOLD",
        )

    def _risk_level(self, risk_result: Dict[str, Any]) -> str:
        return self._title(
            risk_result.get("risk_level")
            or self._get_nested(risk_result, ["risk_for_next_agent", "risk_level"]),
            "Medium",
        )

    def _analyst_signal(self, analysis_result: Dict[str, Any]) -> str:
        return self._norm(analysis_result.get("analyst_signal") or analysis_result.get("display_signal"), "NEUTRAL")

    def _display_signal(self, signal_result: Dict[str, Any], analysis_result: Dict[str, Any]) -> str:
        return self._norm(
            signal_result.get("display_signal")
            or analysis_result.get("display_signal")
            or signal_result.get("model_signal"),
            "HOLD",
        )

    def _entry_risk(self, analysis_result: Dict[str, Any], risk_result: Dict[str, Any]) -> str:
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        return self._title(
            risk_result.get("entry_risk_level")
            or self._get_nested(risk_result, ["risk_for_next_agent", "entry_risk_level"])
            or analysis_result.get("entry_risk_level")
            or stage2.get("entry_risk_level"),
            "Medium",
        )

    def _trend(self, analysis_result: Dict[str, Any]) -> str:
        stage2 = analysis_result.get("stage_2_historical_analysis", {}) if isinstance(analysis_result, dict) else {}
        return self._title(analysis_result.get("trend_direction") or stage2.get("trend_direction"), "Neutral")

    def _model_confidence(self, signal_result: Dict[str, Any]) -> str:
        value = signal_result.get("confidence_level") or self._get_nested(signal_result, ["signal_for_next_agent", "confidence_level"])
        return self._title(value, "Medium")

    def _analyst_score(self, analysis_result: Dict[str, Any]) -> float:
        return self._safe_float(analysis_result.get("analyst_score"), 0.5) or 0.5

    def _validation_confidence(self, validation_result: Dict[str, Any]) -> str:
        return self._title(validation_result.get("confidence"), "Medium")

    # --------------------------------------------------
    # Portfolio and event context
    # --------------------------------------------------
    def _normalise_portfolio_context(self, portfolio_context: Optional[Dict[str, Any]], symbol: str) -> Dict[str, Any]:
        portfolio_context = portfolio_context or {}

        quantity = self._safe_float(
            portfolio_context.get("quantity")
            or portfolio_context.get("shares")
            or portfolio_context.get("current_position")
            or portfolio_context.get("position_size"),
            0.0,
        ) or 0.0

        exposure_pct = self._safe_float(
            portfolio_context.get("exposure_pct")
            or portfolio_context.get("portfolio_weight")
            or portfolio_context.get("weight_pct"),
            None,
        )

        avg_cost = self._safe_float(portfolio_context.get("avg_cost") or portfolio_context.get("average_cost"), None)
        market_value = self._safe_float(portfolio_context.get("market_value"), None)

        has_position = quantity > 0 or (exposure_pct is not None and exposure_pct > 0) or (market_value is not None and market_value > 0)

        exposure_level = "Unknown"
        if exposure_pct is not None:
            if exposure_pct >= 15:
                exposure_level = "High"
            elif exposure_pct >= 5:
                exposure_level = "Medium"
            elif exposure_pct > 0:
                exposure_level = "Low"
            else:
                exposure_level = "None"
        elif has_position:
            exposure_level = "Unknown Existing Position"
        else:
            exposure_level = "None"

        return {
            "symbol": symbol,
            "has_position": has_position,
            "quantity": quantity,
            "avg_cost": avg_cost,
            "market_value": market_value,
            "exposure_pct": exposure_pct,
            "exposure_level": exposure_level,
            "source": "provided" if portfolio_context else "not_provided",
        }

    def _days_until_date(self, date_value: Any) -> Optional[int]:
        if not date_value:
            return None
        try:
            if isinstance(date_value, datetime):
                dt = date_value
            else:
                text = str(date_value).strip()
                text = text.replace("Z", "+00:00")
                dt = datetime.fromisoformat(text)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            return (dt.date() - now.date()).days
        except Exception:
            return None

    def _normalise_event_context(self, event_context: Optional[Dict[str, Any]], analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        event_context = event_context or {}

        earnings_date = (
            event_context.get("earnings_date")
            or event_context.get("next_earnings_date")
            or analysis_result.get("earnings_date")
            or analysis_result.get("next_earnings_date")
        )

        days_to_earnings = self._safe_int(
            event_context.get("days_to_earnings")
            or event_context.get("days_until_earnings"),
            None,
        )
        if days_to_earnings is None:
            days_to_earnings = self._days_until_date(earnings_date)

        event_risk = self._title(event_context.get("event_risk") or event_context.get("risk_level"), "Unknown")
        if event_risk == "Unknown" and days_to_earnings is not None:
            if 0 <= days_to_earnings <= 7:
                event_risk = "High"
            elif 8 <= days_to_earnings <= 21:
                event_risk = "Medium"
            else:
                event_risk = "Low"

        return {
            "earnings_date": earnings_date,
            "days_to_earnings": days_to_earnings,
            "event_risk": event_risk,
            "source": "provided" if event_context or earnings_date else "not_provided",
        }

    # --------------------------------------------------
    # Strategy selection
    # --------------------------------------------------
    def _choose_action(
        self,
        final_signal: str,
        risk_level: str,
        analyst_signal: str,
        display_signal: str,
        entry_risk: str,
        trend: str,
        model_conf: str,
        portfolio: Dict[str, Any],
        event: Dict[str, Any],
    ) -> str:
        risk_level_n = risk_level.title()
        entry_risk_n = entry_risk.title()
        trend_n = trend.title()
        model_conf_n = model_conf.title()

        if final_signal == "BLOCKED" or risk_level_n == "Critical":
            return "NO_ACTION_DATA_OR_RISK_BLOCK"

        if final_signal == "SELL_RISK" or risk_level_n == "High":
            return "RISK_REDUCTION_REVIEW"

        positive_context = (
            final_signal == "BUY_CANDIDATE"
            or "BULLISH" in analyst_signal
            or "POSITIVE" in analyst_signal
            or "WATCHLIST" in display_signal
        )

        event_risk = event.get("event_risk", "Unknown")
        high_event_risk = event_risk == "High"
        high_exposure = portfolio.get("exposure_level") == "High"

        if positive_context and (entry_risk_n == "High" or high_event_risk or high_exposure):
            return "WAIT_FOR_PULLBACK_OR_CONFIRMATION"

        if positive_context and risk_level_n in ["Medium", "High"]:
            return "WAIT_FOR_CONFIRMATION"

        if final_signal == "BUY_CANDIDATE" and risk_level_n == "Low" and model_conf_n in ["Medium", "High"]:
            return "RESEARCH_FOR_POSSIBLE_ENTRY"

        if trend_n in ["Positive", "Strong Positive"] and positive_context:
            return "MONITOR_POSITIVE_SETUP"

        return "MONITOR_AND_RESEARCH"

    def _level(self, action: str, risk_level: str, portfolio: Dict[str, Any], event: Dict[str, Any]) -> str:
        if action in ["NO_ACTION_DATA_OR_RISK_BLOCK", "RISK_REDUCTION_REVIEW"]:
            return "Defensive"
        if action in ["WAIT_FOR_PULLBACK_OR_CONFIRMATION", "WAIT_FOR_CONFIRMATION"]:
            return "Cautious"
        if risk_level in ["High", "Critical"] or event.get("event_risk") == "High" or portfolio.get("exposure_level") == "High":
            return "Cautious"
        if action == "RESEARCH_FOR_POSSIBLE_ENTRY" and risk_level == "Low":
            return "Constructive"
        return "Conservative"

    def _strategy_confidence(
        self,
        validation_conf: str,
        risk_level: str,
        model_conf: str,
        analyst_score: float,
        entry_risk: str,
        portfolio: Dict[str, Any],
        event: Dict[str, Any],
    ) -> Dict[str, Any]:
        score = 0.55
        reasons = []

        if validation_conf == "High":
            score += 0.12
            reasons.append("data confidence is high")
        elif validation_conf == "Low":
            score -= 0.18
            reasons.append("data confidence is low")

        if model_conf == "High":
            score += 0.10
            reasons.append("model confidence is high")
        elif model_conf == "Low":
            score -= 0.10
            reasons.append("model confidence is low")

        if analyst_score >= 0.70:
            score += 0.08
            reasons.append("technical score is supportive")
        elif analyst_score <= 0.35:
            score -= 0.08
            reasons.append("technical score is weak")

        if risk_level in ["High", "Critical"]:
            score -= 0.20
            reasons.append("risk level is high")
        elif risk_level == "Low":
            score += 0.08
            reasons.append("risk level is low")

        if entry_risk == "High":
            score -= 0.10
            reasons.append("entry timing risk is high")
        elif entry_risk == "Low":
            score += 0.05
            reasons.append("entry timing risk is low")

        if portfolio.get("exposure_level") == "High":
            score -= 0.08
            reasons.append("paper portfolio exposure is high")

        if event.get("event_risk") == "High":
            score -= 0.08
            reasons.append("earnings/event risk is near")

        score = max(0.05, min(0.95, score))
        if score >= 0.72:
            label = "High"
        elif score >= 0.50:
            label = "Medium"
        else:
            label = "Low"

        return {
            "score": round(score, 3),
            "label": label,
            "reasons": reasons or ["mixed evidence"],
        }

    # --------------------------------------------------
    # Guidance and checklist
    # --------------------------------------------------
    def _position_guidance(self, action: str, portfolio: Dict[str, Any], event: Dict[str, Any]) -> str:
        base = self._template_for(action).get("position_guidance", "Use this as a research note only.")

        additions = []
        if portfolio.get("has_position"):
            exposure = portfolio.get("exposure_level", "Unknown")
            additions.append(f"Current paper exposure is {exposure.lower()}, so review position size before adding risk.")
        else:
            additions.append("No existing paper position was provided, so this is treated as a watchlist decision.")

        if event.get("event_risk") in ["High", "Medium"]:
            days = event.get("days_to_earnings")
            if days is not None:
                additions.append(f"Earnings/event risk is {event.get('event_risk').lower()} with about {days} day(s) to the event.")
            else:
                additions.append(f"Earnings/event risk is {event.get('event_risk').lower()}.")

        return " ".join([base] + additions)

    def _leverage_guidance(self, action: str, risk_level: str, entry_risk: str, portfolio: Dict[str, Any], event: Dict[str, Any]) -> str:
        template = self._template_for(action).get("leverage_guidance")
        if template:
            return template

        if risk_level in ["High", "Critical"] or action in ["NO_ACTION_DATA_OR_RISK_BLOCK", "RISK_REDUCTION_REVIEW"]:
            return "Do not use leverage. The risk layer is defensive."
        if entry_risk in ["Medium", "High"] or action.startswith("WAIT"):
            return "Do not use leverage. The setup still has timing risk."
        if portfolio.get("exposure_level") == "High":
            return "Do not use leverage. Paper portfolio exposure is already high."
        if event.get("event_risk") == "High":
            return "Do not use leverage before a near earnings/event window."
        return "No leverage is assumed in this prototype. Keep the result as paper research only."

    def _conditions(self, action: str, entry_risk: str, trend: str, model_conf: str, portfolio: Dict[str, Any], event: Dict[str, Any]) -> List[str]:
        checklist = self.templates.get("checklist", {})
        conditions = list(checklist.get("base", []))

        if action in ["WAIT_FOR_PULLBACK_OR_CONFIRMATION", "WAIT_FOR_CONFIRMATION", "MONITOR_POSITIVE_SETUP"] or entry_risk in ["Medium", "High"]:
            conditions.extend(checklist.get("entry_risk", []))

        if model_conf == "Low":
            conditions.append("Wait for model confidence to improve after automatic retraining.")

        if trend not in ["Positive", "Strong Positive"]:
            conditions.append("Wait for trend direction to become clearer.")

        if action in ["RISK_REDUCTION_REVIEW", "NO_ACTION_DATA_OR_RISK_BLOCK"]:
            conditions.extend(checklist.get("risk", []))

        if portfolio.get("has_position"):
            conditions.extend(checklist.get("portfolio", []))

        if event.get("event_risk") in ["Medium", "High"]:
            conditions.extend(checklist.get("event", []))

        # De-duplicate while preserving order
        seen = set()
        result = []
        for item in conditions:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result

    def _watchlist_status(self, action: str) -> str:
        return self._template_for(action).get("watchlist_status", "Watchlist")

    # --------------------------------------------------
    # Main public method
    # --------------------------------------------------
    def plan_strategy(
        self,
        validation_result: Dict[str, Any],
        analysis_result: Dict[str, Any],
        training_result: Dict[str, Any],
        signal_result: Dict[str, Any],
        risk_result: Dict[str, Any],
        evaluation_result: Optional[Dict[str, Any]] = None,
        optimizer_result: Optional[Dict[str, Any]] = None,
        portfolio_context: Optional[Dict[str, Any]] = None,
        event_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        validation_result = validation_result or {}
        analysis_result = analysis_result or {}
        training_result = training_result or {}
        signal_result = signal_result or {}
        risk_result = risk_result or {}

        symbol = self._symbol(risk_result, signal_result, analysis_result, validation_result)
        final_signal = self._final_signal(risk_result, signal_result)
        risk_level = self._risk_level(risk_result)
        analyst_signal = self._analyst_signal(analysis_result)
        display_signal = self._display_signal(signal_result, analysis_result)
        entry_risk = self._entry_risk(analysis_result, risk_result)
        trend = self._trend(analysis_result)
        model_conf = self._model_confidence(signal_result)
        analyst_score = self._analyst_score(analysis_result)
        validation_conf = self._validation_confidence(validation_result)

        portfolio = self._normalise_portfolio_context(portfolio_context, symbol)
        event = self._normalise_event_context(event_context, analysis_result)

        action = self._choose_action(
            final_signal=final_signal,
            risk_level=risk_level,
            analyst_signal=analyst_signal,
            display_signal=display_signal,
            entry_risk=entry_risk,
            trend=trend,
            model_conf=model_conf,
            portfolio=portfolio,
            event=event,
        )

        level = self._level(action, risk_level, portfolio, event)
        confidence = self._strategy_confidence(
            validation_conf=validation_conf,
            risk_level=risk_level,
            model_conf=model_conf,
            analyst_score=analyst_score,
            entry_risk=entry_risk,
            portfolio=portfolio,
            event=event,
        )

        position_guidance = self._position_guidance(action, portfolio, event)
        leverage_guidance = self._leverage_guidance(action, risk_level, entry_risk, portfolio, event)
        checklist = self._conditions(action, entry_risk, trend, model_conf, portfolio, event)
        watchlist_status = self._watchlist_status(action)

        risk_note = (
            risk_result.get("risk_interpretation")
            or self._get_nested(risk_result, ["risk_for_next_agent", "risk_interpretation"])
            or "Risk interpretation was not provided."
        )

        strategy_summary = (
            f"{symbol}: {action.replace('_', ' ').title()} "
            f"with {level.lower()} stance and {confidence['label'].lower()} strategy confidence."
        )

        reasoning_steps = [
            f"Final risk-controlled signal: {final_signal}.",
            f"Risk level: {risk_level}.",
            f"Analyst signal: {analyst_signal}.",
            f"Display signal: {display_signal}.",
            f"Trend: {trend}; entry risk: {entry_risk}; model confidence: {model_conf}.",
            f"Portfolio context: {portfolio.get('exposure_level')} exposure; has position = {portfolio.get('has_position')}.",
            f"Event context: event risk = {event.get('event_risk')}; days to earnings = {event.get('days_to_earnings')}.",
            f"Chosen strategy action: {action}.",
            f"Strategy confidence: {confidence['score']} ({confidence['label']}).",
        ]

        return {
            "success": True,
            "agent": "Strategist Agent",
            "agent_goal": "Convert model and risk outputs into a cautious research plan.",
            "symbol": symbol,
            "strategy_action": action,
            "strategy_level": level,
            "strategy_confidence": confidence,
            "strategy_summary": strategy_summary,
            "position_guidance": position_guidance,
            "leverage_guidance": leverage_guidance,
            "watchlist_status": watchlist_status,
            "conditions_to_reconsider": checklist,
            "checklist": checklist,
            "portfolio_context": portfolio,
            "event_context": event,
            "risk_note": risk_note,
            "human_review_required": True,
            "reasoning_steps": reasoning_steps,
            "strategy_for_next_agent": {
                "symbol": symbol,
                "strategy_action": action,
                "strategy_level": level,
                "strategy_confidence": confidence,
                "position_guidance": position_guidance,
                "leverage_guidance": leverage_guidance,
                "watchlist_status": watchlist_status,
                "checklist": checklist,
                "portfolio_context": portfolio,
                "event_context": event,
                "human_review_required": True,
            },
            "summary": strategy_summary,
        }

    def generate_strategy(self, *args, **kwargs) -> Dict[str, Any]:
        return self.plan_strategy(*args, **kwargs)

    def run(self, *args, **kwargs) -> Dict[str, Any]:
        return self.plan_strategy(*args, **kwargs)
