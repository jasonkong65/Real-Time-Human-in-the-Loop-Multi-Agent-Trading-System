from __future__ import annotations

from __future__ import annotations

import json

import os

import uuid

from datetime import datetime, timezone, timedelta

from pathlib import Path

from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    from agents.database_backend import DatabaseBackend
except Exception:
    from database_backend import DatabaseBackend


class StorageRecordMixin:

    """Mixin for recording various types of data, including pipeline runs, agent outputs, market quotes, paper decisions, reward updates, DQN replay transitions, training runs, LLM reports, screener runs, and bundled pipeline data. Each method normalizes input data, generates unique IDs and timestamps, and upserts or inserts records into the appropriate database tables."""

    def create_pipeline_run(
        self,
        symbol: str,
        validation_result: Optional[Dict[str, Any]] = None,
        analysis_result: Optional[Dict[str, Any]] = None,
        training_result: Optional[Dict[str, Any]] = None,
        signal_result: Optional[Dict[str, Any]] = None,
        risk_result: Optional[Dict[str, Any]] = None,
        strategy_result: Optional[Dict[str, Any]] = None,
        reward_record_result: Optional[Dict[str, Any]] = None,
        selected_price: Optional[float] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        validation_result = validation_result or {}
        analysis_result = analysis_result or {}
        training_result = training_result or {}
        signal_result = signal_result or {}
        risk_result = risk_result or {}
        strategy_result = strategy_result or {}
        reward_record_result = reward_record_result or {}

        run_id = run_id or self._new_id("run")
        symbol = self._normalise_symbol(symbol)
        row = {
            "run_id": run_id,
            "symbol": symbol,
            "selected_price": self._safe_float(selected_price or validation_result.get("selected_price")),
            "validation_confidence": validation_result.get("confidence"),
            "validation_next_action": validation_result.get("next_action"),
            "analyst_signal": analysis_result.get("analyst_signal") or analysis_result.get("display_signal"),
            "model_signal": signal_result.get("model_signal") or signal_result.get("signal") or training_result.get("model_signal"),
            "model_confidence": signal_result.get("confidence_level") or signal_result.get("model_confidence_level"),
            "final_signal": risk_result.get("final_signal"),
            "risk_level": risk_result.get("risk_level"),
            "risk_action": risk_result.get("risk_action") or risk_result.get("q_action") or risk_result.get("dqn_action"),
            "strategy_action": strategy_result.get("strategy_action"),
            "strategy_level": strategy_result.get("strategy_level"),
            "reward_decision_id": reward_record_result.get("decision_id"),
            "created_at_utc": self._now_utc(),
        }
        self.backend.upsert("pipeline_runs", row, conflict_cols=["run_id"])
        return {"success": True, "run_id": run_id, "symbol": symbol}


    def record_agent_output(self, run_id: Optional[str], symbol: str, agent_name: str, output: Dict[str, Any]) -> Dict[str, Any]:
        row = {
            "id": self._new_id("agent"),
            "run_id": run_id,
            "symbol": self._normalise_symbol(symbol),
            "agent_name": agent_name,
            "output_json": self._to_json(output),
            "created_at_utc": self._now_utc(),
        }
        self.backend.insert("agent_outputs", row)
        return {"success": True, "id": row["id"], "agent_name": agent_name}


    def record_market_quotes(self, multi_quote: Dict[str, Any], run_id: Optional[str] = None, symbol: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(multi_quote, dict):
            return {"success": False, "error": "multi_quote is not a dictionary"}

        symbol = self._normalise_symbol(symbol or multi_quote.get("symbol"))
        possible_sources = []
        for key in ["primary_source", "secondary_source", "finnhub_quote", "alpha_vantage_quote", "quote"]:
            val = multi_quote.get(key)
            if isinstance(val, dict):
                possible_sources.append(val)
        if not possible_sources and "source" in multi_quote:
            possible_sources.append(multi_quote)

        count = 0
        for quote in possible_sources:
            if quote.get("success") is False:
                continue
            row = {
                "id": self._new_id("quote"),
                "run_id": run_id,
                "symbol": self._normalise_symbol(quote.get("symbol") or symbol),
                "source": quote.get("source") or quote.get("provider") or "unknown",
                "current_price": self._safe_float(quote.get("current_price") or quote.get("price") or quote.get("c")),
                "open_price": self._safe_float(quote.get("open_price") or quote.get("open") or quote.get("o")),
                "high_price": self._safe_float(quote.get("high_price") or quote.get("high") or quote.get("h")),
                "low_price": self._safe_float(quote.get("low_price") or quote.get("low") or quote.get("l")),
                "previous_close": self._safe_float(quote.get("previous_close") or quote.get("previous_close_price") or quote.get("pc")),
                "quote_timestamp": str(quote.get("timestamp") or quote.get("latest_trading_day") or quote.get("quote_timestamp") or ""),
                "created_at_utc": self._now_utc(),
                "raw_json": self._to_json(quote),
            }
            self.backend.insert("market_quotes", row)
            count += 1
        return {"success": True, "quotes_recorded": count}


    def record_paper_decision(self, reward_record_result: Dict[str, Any], run_id: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(reward_record_result, dict):
            return {"success": False, "error": "reward_record_result is not a dictionary"}
        if reward_record_result.get("success") is False:
            return {"success": False, "skipped": True, "reason": reward_record_result.get("summary") or reward_record_result.get("error")}

        # Do not create placeholder paper decisions for disabled memory runs,
        # duplicate-control skips, or calls where RewardAgent did not create a
        # real decision/horizon set. Otherwise the Evaluator can show open
        # decisions with zero reward horizons, which looks like a broken N/A.
        decision_id = reward_record_result.get("decision_id") or reward_record_result.get("paper_decision_id")
        entry_price = self._safe_float(reward_record_result.get("entry_price"))
        symbol = self._normalise_symbol(reward_record_result.get("symbol"))
        if (
            not decision_id
            or not symbol
            or entry_price is None
            or reward_record_result.get("skipped_duplicate_control")
            or "disabled" in str(reward_record_result.get("summary", "")).lower()
        ):
            return {
                "success": True,
                "skipped": True,
                "reason": "No new RewardAgent paper decision was created for this run.",
            }

        row = {
            "decision_id": decision_id,
            "run_id": run_id or reward_record_result.get("run_id"),
            "symbol": symbol,
            "entry_price": entry_price,
            "final_signal": reward_record_result.get("final_signal"),
            "risk_action": reward_record_result.get("risk_action"),
            "risk_level": reward_record_result.get("risk_level"),
            "strategy_action": reward_record_result.get("strategy_action"),
            "status": reward_record_result.get("paper_status") or reward_record_result.get("status") or "PAPER_MONITOR_ONLY",
            "paper_status": reward_record_result.get("paper_status") or reward_record_result.get("status") or "PAPER_MONITOR_ONLY",
            "entry_time_utc": reward_record_result.get("entry_time_utc") or reward_record_result.get("created_at_utc") or self._now_utc(),
            "q_state": reward_record_result.get("q_state"),
            "created_at_utc": reward_record_result.get("created_at_utc") or self._now_utc(),
            "updated_at_utc": self._now_utc(),
            "raw_json": self._to_json(reward_record_result),
        }
        self.backend.upsert("paper_decisions", row, conflict_cols=["decision_id"])
        return {"success": True, "decision_id": decision_id}


    def record_reward_updates(self, auto_reward_update_result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(auto_reward_update_result, dict):
            return {"success": False, "error": "auto_reward_update_result is not a dictionary"}
        updates = auto_reward_update_result.get("updates") or auto_reward_update_result.get("completed_updates") or []
        if isinstance(updates, dict):
            updates = [updates]
        count = 0
        for update in updates:
            if not isinstance(update, dict):
                continue

            # RewardAgent already owns the canonical reward_updates table.
            # StorageAgent only mirrors meaningful updates; it should not invent
            # completed rows for skipped/not-yet-due horizons.
            decision_id = update.get("decision_id")
            has_real_outcome = update.get("updated") is True or update.get("reward") is not None or update.get("future_return") is not None
            if not decision_id and not has_real_outcome:
                continue

            update_id = str(update.get("update_id") or update.get("id") or update.get("reward_update_id") or self._new_id("reward"))
            horizon_days = self._safe_int(update.get("horizon_days") or update.get("reward_horizon_days"))
            target_date = update.get("target_date_utc") or update.get("due_at_utc")
            status = update.get("status")
            if not status:
                status = "completed" if has_real_outcome else "pending"
            row = {
                # Old StorageAgent schema
                "id": update_id,
                "reward_horizon_days": horizon_days,
                "due_at_utc": target_date,
                # New RewardAgent schema
                "update_id": update_id,
                "horizon_days": horizon_days,
                "horizon_display": update.get("horizon_display") or update.get("horizon_label"),
                "target_date_utc": target_date,
                "latest_date": update.get("latest_date"),
                "dqn_update_json": self._to_json(update.get("dqn_update") or update.get("dqn_update_json")),
                "dqn_update_summary": update.get("dqn_update_summary"),
                "notes": update.get("notes") or update.get("reason"),
                "created_at_utc": update.get("created_at_utc") or update.get("updated_at_utc") or self._now_utc(),
                # Shared fields
                "decision_id": decision_id,
                "symbol": self._normalise_symbol(update.get("symbol")),
                "entry_price": self._safe_float(update.get("entry_price")),
                "latest_close": self._safe_float(update.get("latest_close")),
                "future_return": self._safe_float(update.get("future_return")),
                "reward": self._safe_float(update.get("reward")),
                "horizon_label": update.get("horizon_label"),
                "final_signal": update.get("final_signal"),
                "strategy_action": update.get("strategy_action"),
                "risk_action": update.get("risk_action"),
                "status": status,
                "updated_at_utc": update.get("updated_at_utc") or self._now_utc(),
                "raw_json": self._to_json(update),
            }
            self.backend.upsert("reward_updates", row, conflict_cols=["id"])
            count += 1
        return {"success": True, "updates_recorded": count}


    def record_dqn_replay_transition(self, transition: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(transition, dict):
            return {"success": False, "error": "transition is not a dictionary"}
        transition_id = transition.get("transition_id") or self._new_id("dqn")
        row = {
            "transition_id": transition_id,
            "symbol": self._normalise_symbol(transition.get("symbol")),
            "state_json": self._to_json(transition.get("state") or transition.get("state_vector") or transition.get("q_state")),
            "action": transition.get("action") or transition.get("risk_action") or transition.get("q_action"),
            "reward": self._safe_float(transition.get("reward")),
            "next_state_json": self._to_json(transition.get("next_state") or transition.get("next_state_vector")),
            "done": 1 if transition.get("done") else 0,
            "source_decision_id": transition.get("decision_id") or transition.get("source_decision_id"),
            "horizon_label": transition.get("horizon_label"),
            "future_return": self._safe_float(transition.get("future_return")),
            "created_at_utc": transition.get("created_at_utc") or self._now_utc(),
            "raw_json": self._to_json(transition),
        }
        self.backend.upsert("risk_dqn_replay", row, conflict_cols=["transition_id"])
        return {"success": True, "transition_id": transition_id}


    def get_dqn_replay_memory(self, limit: int = 10000) -> List[Dict[str, Any]]:
        rows = self.backend.query(
            f"""
            SELECT * FROM risk_dqn_replay
            ORDER BY created_at_utc DESC
            LIMIT {self.backend.safe_limit(limit, 10000)}
            """
        )
        for row in rows:
            row["state"] = self._from_json(row.get("state_json"), [])
            row["next_state"] = self._from_json(row.get("next_state_json"), [])
            row["raw"] = self._from_json(row.get("raw_json"), {})
        return rows


    def record_training_run(self, run_id: Optional[str], symbol: str, training_result: Dict[str, Any]) -> Dict[str, Any]:
        training_result = training_result or {}
        metrics = training_result.get("metrics") or training_result.get("best_metrics") or training_result.get("model_metrics") or {}
        row = {
            "id": self._new_id("train"),
            "run_id": run_id,
            "symbol": self._normalise_symbol(symbol or training_result.get("symbol")),
            "model_type": training_result.get("model_type") or training_result.get("best_model_type"),
            "best_params_json": self._to_json(training_result.get("best_params") or training_result.get("params") or {}),
            "accuracy": self._safe_float(metrics.get("accuracy") or training_result.get("accuracy")),
            "balanced_accuracy": self._safe_float(metrics.get("balanced_accuracy") or training_result.get("balanced_accuracy")),
            "macro_f1": self._safe_float(metrics.get("macro_f1") or training_result.get("macro_f1")),
            "sell_risk_recall": self._safe_float(metrics.get("sell_risk_recall") or training_result.get("sell_risk_recall")),
            "training_samples": self._safe_int(training_result.get("training_samples") or training_result.get("num_samples")),
            "model_path": training_result.get("model_path"),
            "metadata_path": training_result.get("metadata_path"),
            "save_decision": training_result.get("save_decision"),
            "created_at_utc": self._now_utc(),
            "raw_json": self._to_json(training_result),
        }
        self.backend.insert("training_runs", row)
        return {"success": True, "id": row["id"]}


    def record_llm_report(self, run_id: Optional[str], symbol: str, report_result: Dict[str, Any]) -> Dict[str, Any]:
        report_result = report_result or {}
        row = {
            "id": self._new_id("llm"),
            "run_id": run_id,
            "symbol": self._normalise_symbol(symbol or report_result.get("symbol")),
            "report_type": report_result.get("report_type"),
            "provider": report_result.get("provider") or "groq",
            "model": report_result.get("model"),
            "source": report_result.get("source"),
            "llm_available": 1 if report_result.get("llm_available") else 0,
            "plain_language_report": report_result.get("plain_language_report") or report_result.get("report") or "",
            "created_at_utc": self._now_utc(),
            "raw_json": self._to_json(report_result),
        }
        self.backend.insert("llm_reports", row)
        return {"success": True, "id": row["id"]}


    def record_screener_run(self, screener_result: Dict[str, Any], top_n: Optional[int] = None, period: Optional[str] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
        screener_result = screener_result or {}
        row = {
            "id": self._new_id("screen"),
            "run_id": run_id,
            "universe_size": self._safe_int(screener_result.get("universe_size") or screener_result.get("scanned_count")),
            "top_n": self._safe_int(top_n or screener_result.get("top_n")),
            "period": period or screener_result.get("period"),
            "result_json": self._to_json(screener_result),
            "created_at_utc": self._now_utc(),
        }
        self.backend.insert("screener_runs", row)
        return {"success": True, "id": row["id"]}


    def record_pipeline_bundle(
        self,
        symbol: str,
        multi_quote: Optional[Dict[str, Any]] = None,
        historical_data: Optional[Dict[str, Any]] = None,
        validation_result: Optional[Dict[str, Any]] = None,
        analysis_result: Optional[Dict[str, Any]] = None,
        training_result: Optional[Dict[str, Any]] = None,
        signal_result: Optional[Dict[str, Any]] = None,
        risk_result: Optional[Dict[str, Any]] = None,
        strategy_result: Optional[Dict[str, Any]] = None,
        reward_record_result: Optional[Dict[str, Any]] = None,
        auto_reward_update_result: Optional[Dict[str, Any]] = None,
        llm_report_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        run = self.create_pipeline_run(
            symbol=symbol,
            validation_result=validation_result,
            analysis_result=analysis_result,
            training_result=training_result,
            signal_result=signal_result,
            risk_result=risk_result,
            strategy_result=strategy_result,
            reward_record_result=reward_record_result,
        )
        run_id = run["run_id"]
        symbol = self._normalise_symbol(symbol)

        if multi_quote:
            self.record_market_quotes(multi_quote, run_id=run_id, symbol=symbol)
        for name, output in [
            ("Data Agent", multi_quote),
            ("Historical Data Agent", historical_data),
            ("Validation Agent", validation_result),
            ("Analyst Agent", analysis_result),
            ("Training Agent", training_result),
            ("Signal Model", signal_result),
            ("Risk Agent", risk_result),
            ("Strategist Agent", strategy_result),
            ("Reward Agent", reward_record_result),
            ("Reward Update Agent", auto_reward_update_result),
            ("LLM Report Agent", llm_report_result),
        ]:
            if isinstance(output, dict):
                self.record_agent_output(run_id, symbol, name, output)
        if reward_record_result:
            self.record_paper_decision(reward_record_result, run_id=run_id)
        if auto_reward_update_result:
            self.record_reward_updates(auto_reward_update_result)
        if training_result:
            self.record_training_run(run_id, symbol, training_result)
        if llm_report_result:
            self.record_llm_report(run_id, symbol, llm_report_result)
        return {"success": True, "run_id": run_id, "summary": f"Saved pipeline run for {symbol}."}

