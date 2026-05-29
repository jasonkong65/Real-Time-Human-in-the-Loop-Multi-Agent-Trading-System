from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

import pandas as pd
import yfinance as yf

from agents.risk_agent import RiskAgent


class RewardAgent:
    """
    Reward Agent

    Records paper decisions and later updates the Risk Agent's DQN replay memory
    using delayed market outcomes. It never sends real orders.
    """

    PENDING_COLUMNS = [
        "decision_id", "symbol", "entry_price", "entry_time_utc",
        "q_state", "risk_action", "final_signal", "risk_level", "status",
    ]
    HISTORY_COLUMNS = PENDING_COLUMNS + [
        "latest_close", "latest_date", "future_return", "reward", "updated_at_utc", "dqn_update_summary",
    ]

    def __init__(self, pending_path: str = "data/pending_rewards.csv", history_path: str = "data/reward_history.csv"):
        self.pending_path = Path(pending_path)
        self.history_path = Path(history_path)
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def _now(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _load_csv(self, path: Path, columns) -> pd.DataFrame:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame(columns=columns)
        try:
            df = pd.read_csv(path)
            for col in columns:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception:
            return pd.DataFrame(columns=columns)

    def _save_csv(self, df: pd.DataFrame, path: Path, columns) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if df.empty:
            pd.DataFrame(columns=columns).to_csv(path, index=False)
        else:
            df.to_csv(path, index=False)

    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _date(self, value) -> Optional[pd.Timestamp]:
        try:
            return pd.to_datetime(value).normalize()
        except Exception:
            return None

    def _latest_close(self, symbol: str) -> Dict[str, Any]:
        try:
            df = yf.download(symbol, period="10d", interval="1d", auto_adjust=False, progress=False)
            if df.empty:
                return {"success": False, "error": "No recent market data returned."}
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
            df = df.reset_index()
            df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
            if "date" not in df.columns and "datetime" in df.columns:
                df = df.rename(columns={"datetime": "date"})
            if "close" not in df.columns:
                return {"success": False, "error": "Downloaded data has no close column."}
            df = df.dropna(subset=["close"])
            latest = df.iloc[-1]
            return {
                "success": True,
                "latest_close": float(latest["close"]),
                "latest_date": str(pd.to_datetime(latest.get("date", pd.Timestamp.today())).date()),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def record_pending_decision(self, symbol: str, entry_price: float, risk_result: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(symbol or "").upper().strip()
        price = self._safe_float(entry_price)
        if not symbol or price is None or price <= 0:
            return {"success": False, "summary": "Reward Agent could not record this paper decision because the symbol or price is missing."}

        row = {
            "decision_id": str(uuid.uuid4())[:8],
            "symbol": symbol,
            "entry_price": price,
            "entry_time_utc": self._now(),
            "q_state": risk_result.get("q_state"),
            "risk_action": risk_result.get("risk_action"),
            "final_signal": risk_result.get("final_signal"),
            "risk_level": risk_result.get("risk_level"),
            "status": "pending",
        }
        df = self._load_csv(self.pending_path, self.PENDING_COLUMNS)
        # Avoid recording the exact same pending decision too many times in a short demo.
        recent = df[(df.get("symbol", "") == symbol) & (df.get("status", "") == "pending")]
        if not recent.empty and len(recent) >= 5:
            return {
                "success": True,
                "symbol": symbol,
                "skipped_duplicate_control": True,
                "summary": f"Reward Agent already has several pending paper decisions for {symbol}, so it skipped adding another duplicate.",
            }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        self._save_csv(df, self.pending_path, self.PENDING_COLUMNS)
        return {
            "success": True,
            "decision_id": row["decision_id"],
            "symbol": symbol,
            "entry_price": price,
            "final_signal": row["final_signal"],
            "q_state": row["q_state"],
            "risk_action": row["risk_action"],
            "pending_path": str(self.pending_path),
            "summary": f"Recorded a paper decision for {symbol}; the DQN risk layer can learn from it later.",
        }

    def auto_update_due_rewards(self) -> Dict[str, Any]:
        pending = self._load_csv(self.pending_path, self.PENDING_COLUMNS)
        if pending.empty:
            return {"success": True, "updated_count": 0, "skipped_count": 0, "updates": [], "summary": "No pending paper decisions found."}

        risk_agent = RiskAgent()
        updates, completed, still_pending = [], [], []
        for _, row in pending.iterrows():
            item = row.to_dict()
            if str(item.get("status")) != "pending":
                continue
            symbol = str(item.get("symbol", "")).upper().strip()
            entry_price = self._safe_float(item.get("entry_price"))
            entry_date = self._date(item.get("entry_time_utc"))
            latest = self._latest_close(symbol)
            if not latest.get("success") or entry_price is None or entry_price <= 0:
                item["status"] = "pending"
                still_pending.append(item)
                updates.append({"symbol": symbol, "updated": False, "reason": latest.get("error", "Invalid entry price.")})
                continue
            latest_date = self._date(latest.get("latest_date"))
            if latest_date is not None and entry_date is not None and latest_date <= entry_date:
                still_pending.append(item)
                updates.append({"symbol": symbol, "updated": False, "reason": "No later market close is available yet."})
                continue

            latest_close = float(latest["latest_close"])
            future_return = (latest_close - entry_price) / entry_price
            volatility = "High" if item.get("risk_level") in ["High", "Critical"] else "Low"
            reward = risk_agent.calculate_reward(item.get("final_signal"), future_return, volatility)
            dqn_update = risk_agent.update_q_value(item.get("q_state"), item.get("risk_action"), reward)
            item.update({
                "status": "completed",
                "latest_close": latest_close,
                "latest_date": latest.get("latest_date"),
                "future_return": future_return,
                "reward": reward,
                "updated_at_utc": self._now(),
                "dqn_update_summary": dqn_update.get("summary"),
            })
            completed.append(item)
            updates.append({"symbol": symbol, "updated": True, "future_return": round(future_return, 4), "reward": round(reward, 4), "dqn_update": dqn_update})

        self._save_csv(pd.DataFrame(still_pending), self.pending_path, self.PENDING_COLUMNS)
        if completed:
            hist = self._load_csv(self.history_path, self.HISTORY_COLUMNS)
            hist = pd.concat([hist, pd.DataFrame(completed)], ignore_index=True)
            self._save_csv(hist, self.history_path, self.HISTORY_COLUMNS)
        updated_count = sum(1 for u in updates if u.get("updated"))
        skipped_count = sum(1 for u in updates if not u.get("updated"))
        return {
            "success": True,
            "updated_count": updated_count,
            "skipped_count": skipped_count,
            "updates": updates,
            "pending_path": str(self.pending_path),
            "history_path": str(self.history_path),
            "summary": f"Delayed reward check completed: {updated_count} updated, {skipped_count} still pending.",
        }
