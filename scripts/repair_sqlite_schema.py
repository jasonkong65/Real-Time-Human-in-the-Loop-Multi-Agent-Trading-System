from pathlib import Path
import sqlite3

DB_PATH = Path("data/trading_system.db")

def columns(conn, table):
    try:
        return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except sqlite3.Error:
        return set()

def add_column(conn, table, name, col_type):
    if name not in columns(conn, table):
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")

if not DB_PATH.exists():
    print(f"No database found at {DB_PATH}. The app will create a fresh one on next run.")
    raise SystemExit(0)

with sqlite3.connect(DB_PATH) as conn:
    add_column(conn, "paper_decisions", "paper_status", "TEXT")
    add_column(conn, "paper_decisions", "entry_time_utc", "TEXT")
    add_column(conn, "paper_decisions", "q_state", "TEXT")
    add_column(conn, "paper_decisions", "duplicate_group_key", "TEXT")
    add_column(conn, "paper_decisions", "risk_result_json", "TEXT")
    add_column(conn, "paper_decisions", "updated_at_utc", "TEXT")

    add_column(conn, "reward_updates", "update_id", "TEXT")
    add_column(conn, "reward_updates", "horizon_display", "TEXT")
    add_column(conn, "reward_updates", "horizon_days", "INTEGER")
    add_column(conn, "reward_updates", "target_date_utc", "TEXT")
    add_column(conn, "reward_updates", "latest_date", "TEXT")
    add_column(conn, "reward_updates", "dqn_update_json", "TEXT")
    add_column(conn, "reward_updates", "dqn_update_summary", "TEXT")
    add_column(conn, "reward_updates", "notes", "TEXT")
    add_column(conn, "reward_updates", "created_at_utc", "TEXT")

    statements = [
        """UPDATE paper_decisions SET paper_status = COALESCE(NULLIF(paper_status, ''), NULLIF(status, ''), 'PAPER_MONITOR_ONLY') WHERE paper_status IS NULL OR paper_status = ''""",
        """UPDATE paper_decisions SET entry_time_utc = COALESCE(NULLIF(entry_time_utc, ''), created_at_utc, datetime('now')) WHERE entry_time_utc IS NULL OR entry_time_utc = ''""",
        """UPDATE reward_updates SET update_id = COALESCE(NULLIF(update_id, ''), NULLIF(id, ''), lower(hex(randomblob(16)))) WHERE update_id IS NULL OR update_id = ''""",
        """UPDATE reward_updates SET target_date_utc = COALESCE(NULLIF(target_date_utc, ''), NULLIF(due_at_utc, ''), updated_at_utc, datetime('now')) WHERE target_date_utc IS NULL OR target_date_utc = ''""",
        """UPDATE reward_updates SET horizon_days = COALESCE(horizon_days, reward_horizon_days, 1) WHERE horizon_days IS NULL""",
        """UPDATE reward_updates SET horizon_display = COALESCE(NULLIF(horizon_display, ''), NULLIF(horizon_label, '')) WHERE horizon_display IS NULL OR horizon_display = ''""",
        """UPDATE reward_updates SET created_at_utc = COALESCE(NULLIF(created_at_utc, ''), updated_at_utc, target_date_utc, datetime('now')) WHERE created_at_utc IS NULL OR created_at_utc = ''""",
    ]
    for sql in statements:
        try:
            conn.execute(sql)
        except sqlite3.Error:
            pass
    conn.execute("CREATE INDEX IF NOT EXISTS idx_reward_updates_status_target ON reward_updates(status, target_date_utc)")
    conn.commit()

print("SQLite schema repaired successfully.")
