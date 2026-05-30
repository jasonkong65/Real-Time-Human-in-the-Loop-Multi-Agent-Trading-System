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

from .helpers import StorageHelpersMixin
from .schema import StorageSchemaMixin
from .historical import StorageHistoricalMixin
from .records import StorageRecordMixin
from .queries import StorageQueryMixin


class StorageAgent(StorageHelpersMixin, StorageSchemaMixin, StorageHistoricalMixin, StorageRecordMixin, StorageQueryMixin):
    """PostgreSQL-ready persistent memory layer for the multi-agent stock system.

Default database:
    DATABASE_URL=sqlite:///data/trading_system.db

Future production database:
    DATABASE_URL=postgresql+psycopg2://user:password@host:5432/trading_system

This class keeps the existing app/agent-facing method names, while moving the
storage design toward a database-first architecture:
- historical_prices and historical_metadata replace per-symbol CSV files as the main store.
- market quotes, agent outputs, rewards, DQN replay, training metadata, screener runs,
  and LLM reports live in the same persistent memory layer.
- CSV / Parquet can still be used as optional export or fallback, but not as the main source."""


    SCHEMA_VERSION = "2.0_database_first_postgres_ready"


    def __init__(
        self,
        db_path: str = "data/trading_system.db",
        database_url: Optional[str] = None,
        auto_init: bool = True,
    ):
        self.db_path = Path(db_path)
        self.database_url = database_url or os.getenv("DATABASE_URL") or f"sqlite:///{db_path}"
        self.backend = DatabaseBackend(database_url=self.database_url, sqlite_path=db_path)
        if auto_init:
            self.init_db()

