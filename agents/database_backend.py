from __future__ import annotations

import os
import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


class DatabaseBackend:
    """
    Small database adapter for the project.

    Default mode:
        DATABASE_URL=sqlite:///data/trading_system.db

    PostgreSQL-ready mode:
        DATABASE_URL=postgresql+psycopg2://user:password@host:5432/trading_system

    Design goal:
    - Current coursework/demo can run with SQLite without a server.
    - Long-term deployment can switch to PostgreSQL by changing DATABASE_URL.
    - Agents call StorageAgent methods and do not need to know which database is used.

    Dependency behaviour:
    - SQLite works with Python's built-in sqlite3.
    - PostgreSQL requires SQLAlchemy + psycopg2-binary.
    - SQLAlchemy can also be used for SQLite by setting DB_FORCE_SQLALCHEMY=true.
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        sqlite_path: str = "data/trading_system.db",
        echo: bool = False,
    ):
        self.database_url = (
            database_url
            or os.getenv("DATABASE_URL")
            or f"sqlite:///{sqlite_path}"
        )
        self.echo = echo
        self.sqlite_path = self._sqlite_path_from_url(self.database_url) or Path(sqlite_path)
        self.is_sqlite = self.database_url.startswith("sqlite")
        self.is_postgres = self.database_url.startswith("postgresql") or self.database_url.startswith("postgres")
        self.force_sqlalchemy = os.getenv("DB_FORCE_SQLALCHEMY", "false").lower() in {"1", "true", "yes"}
        self.use_sqlalchemy = self.force_sqlalchemy or self.is_postgres
        self._engine = None

        if self.is_sqlite and not self.use_sqlalchemy:
            self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self._init_sqlalchemy_engine()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------
    def _sqlite_path_from_url(self, url: str) -> Optional[Path]:
        if not url.startswith("sqlite"):
            return None
        # sqlite:///relative/path.db or sqlite:////absolute/path.db
        path = url.replace("sqlite:///", "", 1)
        if path.startswith("/"):
            return Path(path)
        return Path(path)

    def _init_sqlalchemy_engine(self):
        try:
            from sqlalchemy import create_engine
        except Exception as exc:
            raise RuntimeError(
                "SQLAlchemy is required for this DATABASE_URL. "
                "Install dependencies with: pip install SQLAlchemy psycopg2-binary"
            ) from exc

        connect_args = {}
        if self.is_sqlite:
            sqlite_path = self._sqlite_path_from_url(self.database_url)
            if sqlite_path:
                sqlite_path.parent.mkdir(parents=True, exist_ok=True)
            connect_args = {"check_same_thread": False}

        self._engine = create_engine(
            self.database_url,
            echo=self.echo,
            future=True,
            pool_pre_ping=True,
            connect_args=connect_args,
        )

    @property
    def dialect(self) -> str:
        if self.use_sqlalchemy and self._engine is not None:
            return self._engine.dialect.name
        return "sqlite"

    # ------------------------------------------------------------------
    # Low-level execution
    # ------------------------------------------------------------------
    @contextmanager
    def _sqlite_connection(self):
        conn = sqlite3.connect(str(self.sqlite_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def execute(self, sql: str, params: Optional[Dict[str, Any]] = None) -> None:
        params = params or {}
        if self.use_sqlalchemy:
            from sqlalchemy import text

            with self._engine.begin() as conn:
                conn.execute(text(sql), params)
            return

        with self._sqlite_connection() as conn:
            conn.execute(sql, params)

    def executemany(self, sql: str, rows: Sequence[Dict[str, Any]]) -> None:
        if not rows:
            return
        if self.use_sqlalchemy:
            from sqlalchemy import text

            with self._engine.begin() as conn:
                conn.execute(text(sql), list(rows))
            return

        with self._sqlite_connection() as conn:
            conn.executemany(sql, rows)

    def query(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        params = params or {}
        if self.use_sqlalchemy:
            from sqlalchemy import text

            with self._engine.begin() as conn:
                result = conn.execute(text(sql), params)
                return [dict(row._mapping) for row in result.fetchall()]

        with self._sqlite_connection() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

    def insert(self, table: str, row: Dict[str, Any]) -> None:
        clean = {k: v for k, v in row.items() if k and v is not None}
        if not clean:
            return
        cols = list(clean.keys())
        col_sql = ", ".join(cols)
        val_sql = ", ".join([f":{c}" for c in cols])
        self.execute(f"INSERT INTO {table} ({col_sql}) VALUES ({val_sql})", clean)

    def upsert(self, table: str, row: Dict[str, Any], conflict_cols: Sequence[str]) -> None:
        clean = {k: v for k, v in row.items() if k and v is not None}
        if not clean:
            return

        cols = list(clean.keys())
        col_sql = ", ".join(cols)
        val_sql = ", ".join([f":{c}" for c in cols])
        conflict_sql = ", ".join(conflict_cols)
        update_cols = [c for c in cols if c not in conflict_cols]

        if self.dialect == "postgresql":
            if update_cols:
                update_sql = ", ".join([f"{c} = EXCLUDED.{c}" for c in update_cols])
                sql = (
                    f"INSERT INTO {table} ({col_sql}) VALUES ({val_sql}) "
                    f"ON CONFLICT ({conflict_sql}) DO UPDATE SET {update_sql}"
                )
            else:
                sql = (
                    f"INSERT INTO {table} ({col_sql}) VALUES ({val_sql}) "
                    f"ON CONFLICT ({conflict_sql}) DO NOTHING"
                )
            self.execute(sql, clean)
            return

        # SQLite path. This works with composite PRIMARY KEY or UNIQUE constraint.
        sql = f"INSERT OR REPLACE INTO {table} ({col_sql}) VALUES ({val_sql})"
        self.execute(sql, clean)

    def table_exists(self, table: str) -> bool:
        if self.dialect == "postgresql":
            rows = self.query(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'public' AND table_name = :table",
                {"table": table},
            )
            return bool(rows)

        rows = self.query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=:table",
            {"table": table},
        )
        return bool(rows)

    def add_column_if_missing(self, table: str, column: str, column_type: str) -> None:
        if self.dialect == "postgresql":
            sql = (
                f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {column} {column_type}"
            )
            self.execute(sql)
            return

        columns = self.query(f"PRAGMA table_info({table})")
        existing = {row.get("name") for row in columns}
        if column not in existing:
            self.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def safe_limit(self, value: Any, default: int = 100) -> int:
        try:
            value = int(value)
            return max(1, min(value, 100000))
        except Exception:
            return default
