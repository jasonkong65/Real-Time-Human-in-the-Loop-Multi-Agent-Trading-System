from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

import pandas as pd


class ExecutionAgent:
    """
    Paper Execution Agent

    This agent does not place real trades. It only records a simulated paper
    action when the Strategy Agent has already produced a research plan.
    """

    COLUMNS = [
        "paper_order_id", "created_at_utc", "symbol", "strategy_action",
        "final_signal", "risk_level", "paper_status", "note",
    ]

    def __init__(self, paper_log_path: str = "data/paper_execution_log.csv"):
        self.paper_log_path = Path(paper_log_path)
        self.paper_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _load(self) -> pd.DataFrame:
        if not self.paper_log_path.exists() or self.paper_log_path.stat().st_size == 0:
            return pd.DataFrame(columns=self.COLUMNS)
        try:
            df = pd.read_csv(self.paper_log_path)
            for col in self.COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception:
            return pd.DataFrame(columns=self.COLUMNS)

    def record_paper_action(self, symbol: str, strategy_result: Dict[str, Any], risk_result: Dict[str, Any], note: Optional[str] = None) -> Dict[str, Any]:
        symbol = str(symbol or strategy_result.get("symbol") or risk_result.get("symbol") or "UNKNOWN").upper().strip()
        strategy_action = strategy_result.get("strategy_action", "MONITOR_AND_RESEARCH")
        final_signal = risk_result.get("final_signal", "HOLD")
        risk_level = risk_result.get("risk_level", "Unknown")
        if strategy_action in ["NO_ACTION_DATA_OR_RISK_BLOCK", "RISK_REDUCTION_REVIEW"]:
            paper_status = "not_recorded_as_entry"
            action_note = "The strategy is defensive, so no paper entry is recorded."
        elif final_signal == "BUY_CANDIDATE":
            paper_status = "paper_research_candidate"
            action_note = "Recorded as a paper research candidate only."
        else:
            paper_status = "monitor_only"
            action_note = "Recorded as monitor-only research."
        row = {
            "paper_order_id": str(uuid.uuid4())[:8],
            "created_at_utc": self._now(),
            "symbol": symbol,
            "strategy_action": strategy_action,
            "final_signal": final_signal,
            "risk_level": risk_level,
            "paper_status": paper_status,
            "note": note or action_note,
        }
        df = self._load()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.paper_log_path, index=False)
        return {
            "success": True,
            "agent": "Paper Execution Agent",
            "symbol": symbol,
            "paper_order_id": row["paper_order_id"],
            "paper_status": paper_status,
            "log_path": str(self.paper_log_path),
            "summary": f"Paper Execution Agent recorded {symbol} as {paper_status}.",
        }

    def run(self, symbol: str, strategy_result: Dict[str, Any], risk_result: Dict[str, Any]) -> Dict[str, Any]:
        return self.record_paper_action(symbol, strategy_result, risk_result)
