from pathlib import Path
from datetime import datetime, timezone
import uuid

import pandas as pd
import yfinance as yf

from agents.risk_agent import RiskAgent


class RewardAgent:
    """
    Reward Agent:
    Records paper-trading decisions and updates the Q-learning table
    when later real market prices become available.

    This creates a delayed reward loop:
    decision now -> future price later -> reward -> Q-table update.

    This does not execute real trades.
    """

    def __init__(
        self,
        pending_path: str = "data/pending_rewards.csv",
        history_path: str = "data/reward_history.csv"
    ):
        self.pending_path = Path(pending_path)
        self.history_path = Path(history_path)

        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def _now_utc(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _load_csv(self, path: Path) -> pd.DataFrame:
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    def _save_csv(self, df: pd.DataFrame, path: Path):
        df.to_csv(path, index=False)

    def _parse_date(self, value):
        try:
            return pd.to_datetime(value).date()
        except Exception:
            return None

    def _get_latest_close(self, symbol: str) -> dict:
        """
        Fetch the latest available daily close using yfinance.
        """
        symbol = symbol.upper().strip()

        try:
            df = yf.download(
                tickers=symbol,
                period="10d",
                interval="1d",
                auto_adjust=False,
                progress=False
            )

            if df.empty:
                return {
                    "success": False,
                    "error": "No latest price data returned."
                }

            df = df.reset_index()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    col[0] if isinstance(col, tuple) else col
                    for col in df.columns
                ]

            df = df.rename(columns={
                "Date": "date",
                "Datetime": "date",
                "Close": "close"
            })

            if "date" not in df.columns or "close" not in df.columns:
                return {
                    "success": False,
                    "error": "Downloaded latest data missing date or close."
                }

            df = df.dropna(subset=["date", "close"])

            if df.empty:
                return {
                    "success": False,
                    "error": "Latest price data is empty after cleaning."
                }

            latest = df.iloc[-1]

            return {
                "success": True,
                "latest_date": str(pd.to_datetime(latest["date"]).date()),
                "latest_close": float(latest["close"])
            }

        except Exception as e:
            return {
                "success": False,
                "error": f"Failed to fetch latest close: {str(e)}"
            }

    def record_pending_decision(
        self,
        symbol: str,
        entry_price: float,
        risk_result: dict
    ) -> dict:
        """
        Record the current paper decision for future reward update.

        This does not place a real trade.
        It only stores the decision so the Q-learning Risk Agent can learn later.
        """
        symbol = symbol.upper().strip()

        if entry_price is None:
            return {
                "success": False,
                "summary": "Cannot record pending reward because entry_price is missing."
            }

        try:
            entry_price = float(entry_price)
        except Exception:
            return {
                "success": False,
                "summary": "Cannot record pending reward because entry_price is invalid."
            }

        if entry_price <= 0:
            return {
                "success": False,
                "summary": "Cannot record pending reward because entry_price is not positive."
            }

        decision_id = str(uuid.uuid4())[:8]

        new_row = {
            "decision_id": decision_id,
            "symbol": symbol,
            "entry_price": entry_price,
            "entry_time_utc": self._now_utc(),
            "q_state": risk_result.get("q_state"),
            "risk_action": risk_result.get("risk_action"),
            "final_signal": risk_result.get("final_signal"),
            "risk_level": risk_result.get("risk_level"),
            "status": "pending"
        }

        pending_df = self._load_csv(self.pending_path)
        pending_df = pd.concat([pending_df, pd.DataFrame([new_row])], ignore_index=True)
        self._save_csv(pending_df, self.pending_path)

        return {
            "success": True,
            "decision_id": decision_id,
            "symbol": symbol,
            "entry_price": entry_price,
            "final_signal": new_row["final_signal"],
            "q_state": new_row["q_state"],
            "risk_action": new_row["risk_action"],
            "pending_path": str(self.pending_path),
            "summary": f"Recorded pending delayed reward decision for {symbol}."
        }

    def auto_update_due_rewards(self) -> dict:
        """
        Automatically update pending rewards using later real market prices.

        A pending decision is only updated when the latest market date is later
        than the recorded entry date. This prevents immediate same-day updates
        from being treated as delayed rewards.
        """
        pending_df = self._load_csv(self.pending_path)

        if pending_df.empty:
            return {
                "success": True,
                "updated_count": 0,
                "skipped_count": 0,
                "updates": [],
                "summary": "No pending reward decisions found."
            }

        required_cols = [
            "decision_id",
            "symbol",
            "entry_price",
            "entry_time_utc",
            "q_state",
            "risk_action",
            "final_signal",
            "risk_level",
            "status"
        ]

        missing_cols = [c for c in required_cols if c not in pending_df.columns]

        if missing_cols:
            return {
                "success": False,
                "updated_count": 0,
                "skipped_count": 0,
                "updates": [],
                "summary": f"Pending reward file missing columns: {missing_cols}"
            }

        risk_agent = RiskAgent()

        updates = []
        completed_rows = []
        still_pending_rows = []

        for _, row in pending_df.iterrows():
            row_dict = row.to_dict()

            if row_dict.get("status") != "pending":
                continue

            symbol = str(row_dict["symbol"]).upper().strip()
            entry_price = float(row_dict["entry_price"])
            entry_date = self._parse_date(row_dict.get("entry_time_utc"))

            latest_result = self._get_latest_close(symbol)

            if not latest_result.get("success"):
                still_pending_rows.append(row_dict)
                updates.append({
                    "decision_id": row_dict["decision_id"],
                    "symbol": symbol,
                    "updated": False,
                    "reason": latest_result.get("error")
                })
                continue

            latest_date = self._parse_date(latest_result.get("latest_date"))
            latest_close = float(latest_result["latest_close"])

            if latest_date is None or entry_date is None:
                still_pending_rows.append(row_dict)
                updates.append({
                    "decision_id": row_dict["decision_id"],
                    "symbol": symbol,
                    "updated": False,
                    "reason": "Cannot compare entry date and latest market date."
                })
                continue

            # Real delayed reward: only update when a later market close is available.
            if latest_date <= entry_date:
                still_pending_rows.append(row_dict)
                updates.append({
                    "decision_id": row_dict["decision_id"],
                    "symbol": symbol,
                    "updated": False,
                    "reason": (
                        f"No later market close yet. "
                        f"Entry date={entry_date}, latest date={latest_date}."
                    )
                })
                continue

            if entry_price <= 0:
                still_pending_rows.append(row_dict)
                updates.append({
                    "decision_id": row_dict["decision_id"],
                    "symbol": symbol,
                    "updated": False,
                    "reason": "Invalid entry price."
                })
                continue

            future_return = (latest_close - entry_price) / entry_price

            volatility_level = (
                "High"
                if row_dict.get("risk_level") in ["High", "Critical"]
                else "Low"
            )

            reward = risk_agent.calculate_reward(
                final_signal=row_dict["final_signal"],
                future_return=future_return,
                volatility_level=volatility_level
            )

            update_result = risk_agent.update_q_value(
                state=row_dict["q_state"],
                action=row_dict["risk_action"],
                reward=reward
            )

            completed_row = row_dict.copy()
            completed_row.update({
                "status": "completed",
                "latest_close": latest_close,
                "latest_date": str(latest_date),
                "future_return": future_return,
                "reward": reward,
                "updated_at_utc": self._now_utc(),
                "q_update_summary": update_result.get("summary")
            })

            completed_rows.append(completed_row)

            updates.append({
                "decision_id": row_dict["decision_id"],
                "symbol": symbol,
                "updated": True,
                "entry_price": entry_price,
                "entry_date": str(entry_date),
                "latest_close": latest_close,
                "latest_date": str(latest_date),
                "future_return": round(future_return, 4),
                "reward": round(reward, 4),
                "q_update": update_result
            })

        # Keep only unresolved pending decisions
        new_pending_df = pd.DataFrame(still_pending_rows)
        self._save_csv(new_pending_df, self.pending_path)

        # Append completed decisions to history
        if completed_rows:
            history_df = self._load_csv(self.history_path)
            history_df = pd.concat(
                [history_df, pd.DataFrame(completed_rows)],
                ignore_index=True
            )
            self._save_csv(history_df, self.history_path)

        updated_count = sum(1 for u in updates if u.get("updated"))
        skipped_count = sum(1 for u in updates if not u.get("updated"))

        return {
            "success": True,
            "updated_count": updated_count,
            "skipped_count": skipped_count,
            "updates": updates,
            "pending_path": str(self.pending_path),
            "history_path": str(self.history_path),
            "summary": (
                f"Auto delayed reward update completed: "
                f"{updated_count} updated, {skipped_count} skipped."
            )
        }