from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

from agents.risk_agent import RiskAgent


class RewardAgent:
    """
    Reward Agent for paper decision feedback.

    Main role:
    - Record paper decisions only; never place real trades.
    - Track delayed outcomes across multiple horizons.
    - Store reward records in SQLite as the primary storage layer.
    - Keep lightweight CSV mirrors for backward compatibility with older dashboards.
    - Feed delayed rewards back into the Risk Agent DQN/Q-learning update interface.

    Supported horizons:
    - 1 day
    - 7 days
    - 30 days
    - 1 month
    - 6 months
    - 1 year
    """

    DEFAULT_HORIZONS = [
        {"label": "1_day", "days": 1, "display": "1 day"},
        {"label": "7_day", "days": 7, "display": "7 days"},
        {"label": "30_day", "days": 30, "display": "30 days"},
        {"label": "1_month", "days": 30, "display": "1 month"},
        {"label": "6_month", "days": 182, "display": "6 months"},
        {"label": "1_year", "days": 365, "display": "1 year"},
    ]

    PENDING_COLUMNS = [
        "decision_id",
        "symbol",
        "entry_price",
        "entry_time_utc",
        "q_state",
        "risk_action",
        "final_signal",
        "risk_level",
        "paper_status",
        "open_horizons",
        "next_target_date_utc",
    ]

    HISTORY_COLUMNS = [
        "update_id",
        "decision_id",
        "symbol",
        "horizon_label",
        "horizon_display",
        "horizon_days",
        "target_date_utc",
        "entry_price",
        "latest_close",
        "latest_date",
        "future_return",
        "reward",
        "status",
        "updated_at_utc",
        "dqn_update_summary",
    ]

    def __init__(
        self,
        db_path: str = "data/trading_system.db",
        pending_path: str = "data/pending_rewards.csv",
        history_path: str = "data/reward_history.csv",
        duplicate_window_hours: int = 6,
        max_open_decisions_per_symbol: int = 3,
        mirror_csv: bool = True,
    ):
        self.db_path = Path(db_path)
        self.pending_path = Path(pending_path)
        self.history_path = Path(history_path)
        self.duplicate_window_hours = duplicate_window_hours
        self.max_open_decisions_per_symbol = max_open_decisions_per_symbol
        self.mirror_csv = mirror_csv

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_db()
        if self.mirror_csv:
            self._sync_csv_mirrors()

    # --------------------------------------------------
    # Time / type helpers
    # --------------------------------------------------
    def _now_dt(self) -> datetime:
        return datetime.now(timezone.utc)

    def _now(self) -> str:
        return self._now_dt().strftime("%Y-%m-%d %H:%M:%S")

    def _dt_to_str(self, value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def _parse_dt(self, value: Any) -> Optional[datetime]:
        try:
            if value is None or value == "":
                return None
            ts = pd.to_datetime(value, utc=True)
            return ts.to_pydatetime()
        except Exception:
            return None

    def _parse_date(self, value: Any) -> Optional[pd.Timestamp]:
        try:
            return pd.to_datetime(value).normalize()
        except Exception:
            return None

    def _safe_float(self, value: Any, default: Optional[float] = None) -> Optional[float]:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _json_dumps(self, value: Any) -> str:
        try:
            return json.dumps(value, ensure_ascii=False, default=str)
        except Exception:
            return json.dumps({"value": str(value)}, ensure_ascii=False)

    def _json_loads(self, value: Any, default: Any = None) -> Any:
        try:
            if value is None or value == "":
                return default
            return json.loads(value)
        except Exception:
            return default

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    # --------------------------------------------------
    # SQLite schema
    # --------------------------------------------------
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_decisions (
                    decision_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    entry_time_utc TEXT NOT NULL,
                    q_state TEXT,
                    risk_action TEXT,
                    final_signal TEXT,
                    risk_level TEXT,
                    paper_status TEXT NOT NULL,
                    duplicate_group_key TEXT,
                    risk_result_json TEXT,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reward_updates (
                    update_id TEXT PRIMARY KEY,
                    decision_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon_label TEXT NOT NULL,
                    horizon_display TEXT,
                    horizon_days INTEGER NOT NULL,
                    target_date_utc TEXT NOT NULL,
                    status TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    latest_close REAL,
                    latest_date TEXT,
                    future_return REAL,
                    reward REAL,
                    updated_at_utc TEXT,
                    dqn_update_json TEXT,
                    dqn_update_summary TEXT,
                    notes TEXT,
                    created_at_utc TEXT NOT NULL,
                    FOREIGN KEY(decision_id) REFERENCES paper_decisions(decision_id)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reward_updates_status_target ON reward_updates(status, target_date_utc)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reward_updates_symbol ON reward_updates(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_decisions_symbol_status ON paper_decisions(symbol, paper_status)")

    # --------------------------------------------------
    # Market data helper
    # --------------------------------------------------
    def _latest_close(self, symbol: str, period: str = "10d") -> Dict[str, Any]:
        try:
            df = yf.download(symbol, period=period, interval="1d", auto_adjust=False, progress=False)
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
            if df.empty:
                return {"success": False, "error": "Downloaded data has no valid close values."}

            latest = df.iloc[-1]
            latest_date = pd.to_datetime(latest.get("date", pd.Timestamp.today())).date()
            return {
                "success": True,
                "latest_close": float(latest["close"]),
                "latest_date": str(latest_date),
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    # --------------------------------------------------
    # Duplicate control and paper status helpers
    # --------------------------------------------------
    def _duplicate_key(self, symbol: str, risk_result: Dict[str, Any]) -> str:
        final_signal = risk_result.get("final_signal", "UNKNOWN")
        risk_action = risk_result.get("risk_action", "UNKNOWN")
        risk_level = risk_result.get("risk_level", "UNKNOWN")
        return f"{symbol}|{final_signal}|{risk_action}|{risk_level}"

    def _paper_status_from_signal(self, final_signal: str, risk_action: str, risk_level: str) -> str:
        final_signal = str(final_signal or "").upper()
        risk_action = str(risk_action or "").upper()
        risk_level = str(risk_level or "").upper()

        if risk_action == "BLOCK_TRADE" or final_signal == "BLOCKED":
            return "PAPER_BLOCKED_NO_ENTRY"
        if final_signal == "BUY_CANDIDATE":
            return "PAPER_WATCHLIST_ENTRY_CANDIDATE"
        if final_signal == "SELL_RISK":
            return "PAPER_RISK_REVIEW"
        if risk_level in ["HIGH", "CRITICAL"]:
            return "PAPER_HIGH_RISK_MONITOR"
        return "PAPER_MONITOR_ONLY"

    def _check_duplicate(self, symbol: str, duplicate_key: str) -> Tuple[bool, Dict[str, Any]]:
        cutoff = self._now_dt() - timedelta(hours=self.duplicate_window_hours)
        cutoff_str = self._dt_to_str(cutoff)

        with self._connect() as conn:
            open_count = conn.execute(
                """
                SELECT COUNT(*) AS n
                FROM paper_decisions
                WHERE symbol = ?
                  AND paper_status NOT IN ('COMPLETED_ALL', 'CANCELLED')
                """,
                (symbol,),
            ).fetchone()["n"]

            recent_same = conn.execute(
                """
                SELECT decision_id, entry_time_utc, paper_status
                FROM paper_decisions
                WHERE symbol = ?
                  AND duplicate_group_key = ?
                  AND entry_time_utc >= ?
                  AND paper_status NOT IN ('COMPLETED_ALL', 'CANCELLED')
                ORDER BY entry_time_utc DESC
                LIMIT 1
                """,
                (symbol, duplicate_key, cutoff_str),
            ).fetchone()

        if recent_same is not None:
            return True, {
                "reason": "A similar paper decision was already recorded recently.",
                "existing_decision_id": recent_same["decision_id"],
                "existing_status": recent_same["paper_status"],
                "duplicate_window_hours": self.duplicate_window_hours,
            }

        if open_count >= self.max_open_decisions_per_symbol:
            return True, {
                "reason": f"There are already {open_count} open paper decisions for {symbol}.",
                "max_open_decisions_per_symbol": self.max_open_decisions_per_symbol,
            }

        return False, {}

    # --------------------------------------------------
    # CSV mirrors for older dashboard/evaluator compatibility
    # --------------------------------------------------
    def _sync_csv_mirrors(self) -> None:
        try:
            pending_rows = []
            history_rows = []

            with self._connect() as conn:
                decisions = conn.execute(
                    """
                    SELECT * FROM paper_decisions
                    WHERE paper_status NOT IN ('COMPLETED_ALL', 'CANCELLED')
                    ORDER BY entry_time_utc DESC
                    """
                ).fetchall()
                for d in decisions:
                    open_updates = conn.execute(
                        """
                        SELECT horizon_label, target_date_utc
                        FROM reward_updates
                        WHERE decision_id = ? AND status = 'pending'
                        ORDER BY horizon_days ASC
                        """,
                        (d["decision_id"],),
                    ).fetchall()
                    pending_rows.append({
                        "decision_id": d["decision_id"],
                        "symbol": d["symbol"],
                        "entry_price": d["entry_price"],
                        "entry_time_utc": d["entry_time_utc"],
                        "q_state": d["q_state"],
                        "risk_action": d["risk_action"],
                        "final_signal": d["final_signal"],
                        "risk_level": d["risk_level"],
                        "paper_status": d["paper_status"],
                        "open_horizons": ",".join([u["horizon_label"] for u in open_updates]),
                        "next_target_date_utc": open_updates[0]["target_date_utc"] if open_updates else None,
                    })

                completed = conn.execute(
                    """
                    SELECT * FROM reward_updates
                    WHERE status = 'completed'
                    ORDER BY updated_at_utc DESC
                    """
                ).fetchall()
                for u in completed:
                    history_rows.append({
                        "update_id": u["update_id"],
                        "decision_id": u["decision_id"],
                        "symbol": u["symbol"],
                        "horizon_label": u["horizon_label"],
                        "horizon_display": u["horizon_display"],
                        "horizon_days": u["horizon_days"],
                        "target_date_utc": u["target_date_utc"],
                        "entry_price": u["entry_price"],
                        "latest_close": u["latest_close"],
                        "latest_date": u["latest_date"],
                        "future_return": u["future_return"],
                        "reward": u["reward"],
                        "status": u["status"],
                        "updated_at_utc": u["updated_at_utc"],
                        "dqn_update_summary": u["dqn_update_summary"],
                    })

            pd.DataFrame(pending_rows, columns=self.PENDING_COLUMNS).to_csv(self.pending_path, index=False)
            pd.DataFrame(history_rows, columns=self.HISTORY_COLUMNS).to_csv(self.history_path, index=False)
        except Exception:
            # CSV mirrors are only compatibility outputs. Do not break the agent if they fail.
            pass

    # --------------------------------------------------
    # Public API: record new paper decision
    # --------------------------------------------------
    def record_pending_decision(self, symbol: str, entry_price: float, risk_result: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(symbol or "").upper().strip()
        price = self._safe_float(entry_price)
        risk_result = risk_result or {}

        if not symbol or price is None or price <= 0:
            return {
                "success": False,
                "summary": "Reward Agent could not record the paper decision because the symbol or entry price is missing.",
            }

        duplicate_key = self._duplicate_key(symbol, risk_result)
        duplicate, duplicate_info = self._check_duplicate(symbol, duplicate_key)
        if duplicate:
            if self.mirror_csv:
                self._sync_csv_mirrors()
            return {
                "success": True,
                "symbol": symbol,
                "skipped_duplicate_control": True,
                "duplicate_info": duplicate_info,
                "summary": f"Reward Agent skipped a duplicate paper decision for {symbol}.",
            }

        decision_id = str(uuid.uuid4())
        now_dt = self._now_dt()
        now_str = self._dt_to_str(now_dt)

        q_state = risk_result.get("q_state") or risk_result.get("risk_for_next_agent", {}).get("q_state")
        risk_action = risk_result.get("risk_action") or risk_result.get("risk_for_next_agent", {}).get("risk_action")
        final_signal = risk_result.get("final_signal") or risk_result.get("risk_for_next_agent", {}).get("final_signal")
        risk_level = risk_result.get("risk_level") or risk_result.get("risk_for_next_agent", {}).get("risk_level")
        paper_status = self._paper_status_from_signal(final_signal, risk_action, risk_level)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO paper_decisions (
                    decision_id, symbol, entry_price, entry_time_utc,
                    q_state, risk_action, final_signal, risk_level,
                    paper_status, duplicate_group_key, risk_result_json,
                    created_at_utc, updated_at_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    decision_id,
                    symbol,
                    price,
                    now_str,
                    q_state,
                    risk_action,
                    final_signal,
                    risk_level,
                    paper_status,
                    duplicate_key,
                    self._json_dumps(risk_result),
                    now_str,
                    now_str,
                ),
            )

            for horizon in self.DEFAULT_HORIZONS:
                update_id = str(uuid.uuid4())
                target_date = now_dt + timedelta(days=int(horizon["days"]))
                conn.execute(
                    """
                    INSERT INTO reward_updates (
                        update_id, decision_id, symbol, horizon_label, horizon_display,
                        horizon_days, target_date_utc, status, entry_price,
                        created_at_utc, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        update_id,
                        decision_id,
                        symbol,
                        horizon["label"],
                        horizon["display"],
                        int(horizon["days"]),
                        self._dt_to_str(target_date),
                        "pending",
                        price,
                        now_str,
                        "Waiting for the target horizon to become due.",
                    ),
                )

        if self.mirror_csv:
            self._sync_csv_mirrors()

        return {
            "success": True,
            "decision_id": decision_id,
            "symbol": symbol,
            "entry_price": price,
            "final_signal": final_signal,
            "risk_level": risk_level,
            "risk_action": risk_action,
            "q_state": q_state,
            "paper_status": paper_status,
            "horizons_created": [h["label"] for h in self.DEFAULT_HORIZONS],
            "db_path": str(self.db_path),
            "pending_path": str(self.pending_path),
            "summary": f"Recorded a paper decision for {symbol} across {len(self.DEFAULT_HORIZONS)} reward horizons.",
        }

    # --------------------------------------------------
    # Public API: update due horizons
    # --------------------------------------------------
    def _due_updates(self, now_str: str) -> List[sqlite3.Row]:
        with self._connect() as conn:
            return conn.execute(
                """
                SELECT
                    u.*,
                    d.q_state,
                    d.risk_action,
                    d.final_signal,
                    d.risk_level
                FROM reward_updates u
                JOIN paper_decisions d ON d.decision_id = u.decision_id
                WHERE u.status = 'pending'
                  AND u.target_date_utc <= ?
                ORDER BY u.target_date_utc ASC
                """,
                (now_str,),
            ).fetchall()

    def _mark_update_skipped(self, update_id: str, reason: str) -> None:
        # Keep it pending, but record why it was not updated this time.
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE reward_updates
                SET notes = ?, updated_at_utc = ?
                WHERE update_id = ?
                """,
                (reason, self._now(), update_id),
            )

    def _refresh_decision_status(self, decision_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_count,
                    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_count
                FROM reward_updates
                WHERE decision_id = ?
                """,
                (decision_id,),
            ).fetchone()
            pending_count = int(row["pending_count"] or 0)
            completed_count = int(row["completed_count"] or 0)

            if pending_count == 0 and completed_count > 0:
                new_status = "COMPLETED_ALL"
            elif completed_count > 0:
                new_status = "PARTIALLY_COMPLETED"
            else:
                # Keep the original paper status for open decisions.
                existing = conn.execute(
                    "SELECT paper_status FROM paper_decisions WHERE decision_id = ?",
                    (decision_id,),
                ).fetchone()
                new_status = existing["paper_status"] if existing else "PENDING"

            conn.execute(
                "UPDATE paper_decisions SET paper_status = ?, updated_at_utc = ? WHERE decision_id = ?",
                (new_status, self._now(), decision_id),
            )
            return new_status

    def auto_update_due_rewards(self) -> Dict[str, Any]:
        now_str = self._now()
        due = self._due_updates(now_str)
        if not due:
            if self.mirror_csv:
                self._sync_csv_mirrors()
            return {
                "success": True,
                "updated_count": 0,
                "skipped_count": 0,
                "updates": [],
                "db_path": str(self.db_path),
                "summary": "No due reward horizons found.",
            }

        risk_agent = RiskAgent()
        updates = []
        market_cache: Dict[str, Dict[str, Any]] = {}

        for row in due:
            item = dict(row)
            symbol = str(item.get("symbol", "")).upper().strip()
            entry_price = self._safe_float(item.get("entry_price"))
            target_dt = self._parse_dt(item.get("target_date_utc"))

            if not symbol or entry_price is None or entry_price <= 0:
                reason = "Invalid symbol or entry price."
                self._mark_update_skipped(item["update_id"], reason)
                updates.append({"symbol": symbol, "horizon": item.get("horizon_label"), "updated": False, "reason": reason})
                continue

            if symbol not in market_cache:
                market_cache[symbol] = self._latest_close(symbol)
            latest = market_cache[symbol]

            if not latest.get("success"):
                reason = latest.get("error", "No recent market data returned.")
                self._mark_update_skipped(item["update_id"], reason)
                updates.append({"symbol": symbol, "horizon": item.get("horizon_label"), "updated": False, "reason": reason})
                continue

            latest_date = self._parse_date(latest.get("latest_date"))
            target_date = pd.to_datetime(target_dt).normalize() if target_dt else None

            # If the horizon is due in calendar time but no market close exists yet,
            # keep it pending until the next available close.
            if latest_date is not None and target_date is not None and latest_date < target_date:
                reason = "Target horizon is due, but no market close at or after the target date is available yet."
                self._mark_update_skipped(item["update_id"], reason)
                updates.append({"symbol": symbol, "horizon": item.get("horizon_label"), "updated": False, "reason": reason})
                continue

            latest_close = float(latest["latest_close"])
            future_return = (latest_close - entry_price) / entry_price
            volatility = "High" if item.get("risk_level") in ["High", "Critical"] else "Low"
            reward = risk_agent.calculate_reward(item.get("final_signal"), future_return, volatility)
            dqn_update = risk_agent.update_q_value(item.get("q_state"), item.get("risk_action"), reward)

            updated_at = self._now()
            with self._connect() as conn:
                conn.execute(
                    """
                    UPDATE reward_updates
                    SET status = 'completed',
                        latest_close = ?,
                        latest_date = ?,
                        future_return = ?,
                        reward = ?,
                        updated_at_utc = ?,
                        dqn_update_json = ?,
                        dqn_update_summary = ?,
                        notes = ?
                    WHERE update_id = ?
                    """,
                    (
                        latest_close,
                        latest.get("latest_date"),
                        future_return,
                        reward,
                        updated_at,
                        self._json_dumps(dqn_update),
                        dqn_update.get("summary"),
                        "Completed and sent delayed reward to the Risk Agent.",
                        item["update_id"],
                    ),
                )

            new_decision_status = self._refresh_decision_status(item["decision_id"])
            updates.append({
                "symbol": symbol,
                "decision_id": item["decision_id"],
                "horizon": item.get("horizon_label"),
                "horizon_display": item.get("horizon_display"),
                "updated": True,
                "future_return": round(future_return, 6),
                "reward": round(float(reward), 6),
                "decision_status": new_decision_status,
                "dqn_update": dqn_update,
            })

        if self.mirror_csv:
            self._sync_csv_mirrors()

        updated_count = sum(1 for u in updates if u.get("updated"))
        skipped_count = sum(1 for u in updates if not u.get("updated"))
        return {
            "success": True,
            "updated_count": updated_count,
            "skipped_count": skipped_count,
            "updates": updates,
            "db_path": str(self.db_path),
            "pending_path": str(self.pending_path),
            "history_path": str(self.history_path),
            "summary": f"Reward horizon check completed: {updated_count} updated, {skipped_count} still pending.",
        }

    # --------------------------------------------------
    # Optional helpers for dashboard/debug
    # --------------------------------------------------
    def get_reward_summary(self) -> Dict[str, Any]:
        with self._connect() as conn:
            decision_counts = conn.execute(
                """
                SELECT paper_status, COUNT(*) AS n
                FROM paper_decisions
                GROUP BY paper_status
                ORDER BY n DESC
                """
            ).fetchall()
            reward_counts = conn.execute(
                """
                SELECT status, COUNT(*) AS n
                FROM reward_updates
                GROUP BY status
                ORDER BY n DESC
                """
            ).fetchall()
            completed_stats = conn.execute(
                """
                SELECT AVG(reward) AS avg_reward, AVG(future_return) AS avg_future_return, COUNT(*) AS n
                FROM reward_updates
                WHERE status = 'completed'
                """
            ).fetchone()

        return {
            "success": True,
            "db_path": str(self.db_path),
            "decision_counts": {row["paper_status"]: row["n"] for row in decision_counts},
            "reward_update_counts": {row["status"]: row["n"] for row in reward_counts},
            "completed_reward_count": int(completed_stats["n"] or 0),
            "average_reward": completed_stats["avg_reward"],
            "average_future_return": completed_stats["avg_future_return"],
            "summary": "Reward Agent summary loaded from SQLite.",
        }

    # Backward-compatible alias
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        return self.auto_update_due_rewards(*args, **kwargs)
