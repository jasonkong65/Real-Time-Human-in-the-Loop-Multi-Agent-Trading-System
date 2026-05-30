import sqlite3
from pathlib import Path

def repair_sqlite_schema_without_agent_changes(db_path: str = "data/trading_system.db") -> None:
    """Repair legacy SQLite columns before agents are instantiated.

    This keeps the agents/ folder unchanged while preventing old local
    trading_system.db files from crashing the current RewardAgent queries.
    """
    path = Path(db_path)
    if not path.exists():
        return

    def table_exists(conn, name):
        return conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
        ).fetchone() is not None

    def cols(conn, table):
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def add_missing(conn, table, specs):
        if not table_exists(conn, table):
            return
        existing = cols(conn, table)
        for name, col_type in specs.items():
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")

    with sqlite3.connect(path) as conn:
        add_missing(conn, "paper_decisions", {
            "symbol": "TEXT", "entry_price": "REAL", "entry_time_utc": "TEXT",
            "q_state": "TEXT", "risk_action": "TEXT", "final_signal": "TEXT",
            "risk_level": "TEXT", "paper_status": "TEXT", "duplicate_group_key": "TEXT",
            "risk_result_json": "TEXT", "created_at_utc": "TEXT", "updated_at_utc": "TEXT",
        })
        add_missing(conn, "reward_updates", {
            "update_id": "TEXT", "decision_id": "TEXT", "symbol": "TEXT",
            "horizon_label": "TEXT", "horizon_display": "TEXT", "horizon_days": "INTEGER",
            "target_date_utc": "TEXT", "status": "TEXT", "entry_price": "REAL",
            "latest_close": "REAL", "latest_date": "TEXT", "future_return": "REAL",
            "reward": "REAL", "updated_at_utc": "TEXT", "dqn_update_json": "TEXT",
            "dqn_update_summary": "TEXT", "notes": "TEXT", "created_at_utc": "TEXT",
        })

        if table_exists(conn, "paper_decisions"):
            p = cols(conn, "paper_decisions")
            if {"paper_status", "status"}.issubset(p):
                conn.execute("""
                    UPDATE paper_decisions
                    SET paper_status = COALESCE(NULLIF(paper_status, ''), NULLIF(status, ''), 'PAPER_MONITOR_ONLY')
                    WHERE paper_status IS NULL OR paper_status = ''
                """)
            if {"entry_time_utc", "created_at_utc"}.issubset(p):
                conn.execute("""
                    UPDATE paper_decisions
                    SET entry_time_utc = COALESCE(NULLIF(entry_time_utc, ''), created_at_utc, datetime('now'))
                    WHERE entry_time_utc IS NULL OR entry_time_utc = ''
                """)
            if {"risk_result_json", "raw_json"}.issubset(p):
                conn.execute("""
                    UPDATE paper_decisions
                    SET risk_result_json = COALESCE(NULLIF(risk_result_json, ''), raw_json)
                    WHERE risk_result_json IS NULL OR risk_result_json = ''
                """)

        if table_exists(conn, "reward_updates"):
            r = cols(conn, "reward_updates")
            if {"update_id", "id"}.issubset(r):
                conn.execute("""
                    UPDATE reward_updates
                    SET update_id = COALESCE(NULLIF(update_id, ''), id, lower(hex(randomblob(16))))
                    WHERE update_id IS NULL OR update_id = ''
                """)
            if {"target_date_utc", "due_at_utc"}.issubset(r):
                conn.execute("""
                    UPDATE reward_updates
                    SET target_date_utc = COALESCE(NULLIF(target_date_utc, ''), due_at_utc, updated_at_utc, datetime('now'))
                    WHERE target_date_utc IS NULL OR target_date_utc = ''
                """)
            elif "target_date_utc" in r:
                conn.execute("""
                    UPDATE reward_updates
                    SET target_date_utc = COALESCE(NULLIF(target_date_utc, ''), updated_at_utc, created_at_utc, datetime('now'))
                    WHERE target_date_utc IS NULL OR target_date_utc = ''
                """)
            if {"horizon_days", "reward_horizon_days"}.issubset(r):
                conn.execute("""
                    UPDATE reward_updates
                    SET horizon_days = COALESCE(horizon_days, reward_horizon_days, 1)
                    WHERE horizon_days IS NULL
                """)
            elif "horizon_days" in r:
                conn.execute("UPDATE reward_updates SET horizon_days = COALESCE(horizon_days, 1) WHERE horizon_days IS NULL")
            if {"horizon_display", "horizon_label"}.issubset(r):
                conn.execute("""
                    UPDATE reward_updates
                    SET horizon_display = COALESCE(NULLIF(horizon_display, ''), horizon_label)
                    WHERE horizon_display IS NULL OR horizon_display = ''
                """)
            if "created_at_utc" in r:
                conn.execute("""
                    UPDATE reward_updates
                    SET created_at_utc = COALESCE(NULLIF(created_at_utc, ''), updated_at_utc, target_date_utc, datetime('now'))
                    WHERE created_at_utc IS NULL OR created_at_utc = ''
                """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reward_updates_status_target ON reward_updates(status, target_date_utc)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reward_updates_symbol ON reward_updates(symbol)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_reward_updates_decision ON reward_updates(decision_id)")
        conn.commit()
