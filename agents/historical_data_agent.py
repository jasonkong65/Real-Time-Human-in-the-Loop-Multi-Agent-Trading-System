from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf

try:
    from agents.storage_agent import StorageAgent
except Exception:
    try:
        from storage_agent import StorageAgent
    except Exception:  # pragma: no cover
        StorageAgent = None


class HistoricalDataAgent:
    """
    Database-first Historical Data Agent.

    Role:
    - Uses SQLite/PostgreSQL-ready StorageAgent as the main store for historical OHLCV data.
    - Avoids creating one new CSV per ticker as the primary storage path.
    - Uses yfinance only when database data is missing, stale, or force_refresh=True.
    - Can still read old CSV/Parquet caches and optionally write file backups.

    Main flow:
        check DB historical_prices
        fresh → return DB data
        missing/stale → download yfinance
        download success → write DB historical_prices + metadata, optional file backup
        download failure + stale DB/file exists → return stale cache with warning
    """

    VALID_INTERVALS = {
        "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
    }

    PERIOD_PRESET_ALIASES = {
        "7d": "7d",
        "30d": "30d",
        "1mo": "1mo",
        "3mo": "3mo",
        "6mo": "6mo",
        "1y": "1y",
        "2y": "2y",
        "5y": "5y",
        "10y": "10y",
    }

    INTERVAL_ALIASES = {
        "1hour": "1h",
        "hourly": "1h",
        "daily": "1d",
        "day": "1d",
        "1day": "1d",
        "week": "1wk",
        "weekly": "1wk",
        "month": "1mo",
        "monthly": "1mo",
    }

    def __init__(
        self,
        data_dir: str = "data/historical",
        metadata_dir: Optional[str] = None,
        stale_days: int = 3,
        prefer_parquet: bool = True,
        storage_agent: Optional[Any] = None,
        database_first: bool = True,
        write_file_backup: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir = Path(metadata_dir) if metadata_dir else self.data_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.stale_days = stale_days
        self.prefer_parquet = prefer_parquet
        self.database_first = database_first
        self.write_file_backup = write_file_backup
        self.storage_agent = storage_agent
        if self.storage_agent is None and StorageAgent is not None:
            try:
                self.storage_agent = StorageAgent()
            except Exception:
                self.storage_agent = None

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _symbol(self, symbol: str) -> str:
        return str(symbol or "").upper().strip()

    def _safe_name(self, text: str) -> str:
        return str(text or "").replace("/", "_").replace(" ", "_").replace(":", "_")

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _normalise_period_interval(self, period: str = "1y", interval: str = "1d") -> Tuple[str, str, List[str]]:
        warnings: List[str] = []
        period = str(period or "1y").strip().lower()
        interval = str(interval or "1d").strip().lower()
        interval = self.INTERVAL_ALIASES.get(interval, interval)

        if interval in self.PERIOD_PRESET_ALIASES:
            period = self.PERIOD_PRESET_ALIASES[interval]
            interval = "1d"
            warnings.append(f"Interpreted interval input as period='{period}' with interval='1d'.")

        if period in {"1h", "hourly"}:
            period = "7d"
            interval = "1h"
            warnings.append("Interpreted period='1h' as period='7d', interval='1h'.")

        if interval == "60m":
            interval = "1h"

        if interval not in self.VALID_INTERVALS:
            warnings.append(f"Unsupported interval '{interval}' was replaced with '1d'.")
            interval = "1d"

        if not period:
            period = "1y"

        return period, interval, warnings

    def _base_stem(self, symbol: str, period: str, interval: str) -> str:
        return f"{self._symbol(symbol)}_{self._safe_name(period)}_{self._safe_name(interval)}"

    def _parquet_path(self, symbol: str, period: str, interval: str) -> Path:
        return self.data_dir / f"{self._base_stem(symbol, period, interval)}.parquet"

    def _csv_path(self, symbol: str, period: str, interval: str) -> Path:
        return self.data_dir / f"{self._base_stem(symbol, period, interval)}.csv"

    def _legacy_csv_path(self, symbol: str) -> Path:
        return self.data_dir / f"{self._symbol(symbol)}.csv"

    def _metadata_path(self, symbol: str, period: str, interval: str) -> Path:
        return self.metadata_dir / f"{self._base_stem(symbol, period, interval)}_metadata.json"

    def _clean_df(self, df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        if "date" not in [str(c).lower() for c in df.columns]:
            df = df.reset_index()

        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        rename_map = {
            "datetime": "date",
            "index": "date",
            "adjclose": "adj_close",
            "adj_close": "adj_close",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if "date" not in df.columns:
            for possible in ["price_date", "timestamp", "price_timestamp"]:
                if possible in df.columns:
                    df = df.rename(columns={possible: "date"})
                    break

        if "close" not in df.columns and "adj_close" in df.columns:
            df["close"] = df["adj_close"]

        required = ["date", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()

        for col in ["open", "high", "low", "close", "adj_close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
        df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"]).sort_values("date")

        intraday = interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
        if intraday:
            df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")

        keep = [c for c in ["date", "open", "high", "low", "close", "adj_close", "volume"] if c in df.columns]
        return df[keep].drop_duplicates(subset=["date"], keep="last")

    def _latest_timestamp(self, df: pd.DataFrame) -> Optional[pd.Timestamp]:
        if df is None or df.empty or "date" not in df.columns:
            return None
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if dates.empty:
            return None
        return dates.max()

    def _stale_threshold_hours(self, interval: str) -> float:
        if interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}:
            return 12.0
        if interval in {"1d", "5d"}:
            return float(self.stale_days * 24)
        if interval == "1wk":
            return 10 * 24.0
        if interval in {"1mo", "3mo"}:
            return 45 * 24.0
        return float(self.stale_days * 24)

    def _age_hours(self, df: pd.DataFrame) -> Optional[float]:
        latest = self._latest_timestamp(df)
        if latest is None:
            return None
        latest = latest.tz_localize(None) if latest.tzinfo is not None else latest
        now = pd.Timestamp.utcnow().tz_localize(None)
        return (now - latest).total_seconds() / 3600

    def _is_stale(self, df: pd.DataFrame, interval: str = "1d") -> bool:
        age = self._age_hours(df)
        if age is None:
            return True
        return age > self._stale_threshold_hours(interval)

    def _write_file_backup(self, symbol: str, period: str, interval: str, df: pd.DataFrame, warnings: Optional[List[str]] = None) -> Tuple[Optional[Path], str, List[str]]:
        warnings = list(warnings or [])
        if not self.write_file_backup:
            return None, "database_only", warnings

        if self.prefer_parquet:
            parquet_path = self._parquet_path(symbol, period, interval)
            try:
                df.to_parquet(parquet_path, index=False)
                return parquet_path, "database_plus_parquet_backup", warnings
            except Exception as exc:
                warnings.append(f"Parquet backup failed; CSV backup was used. Error: {exc}")

        csv_path = self._csv_path(symbol, period, interval)
        df.to_csv(csv_path, index=False)
        return csv_path, "database_plus_csv_backup", warnings

    def _read_file_cache(self, symbol: str, period: str, interval: str) -> Tuple[pd.DataFrame, Optional[Path], str, List[str]]:
        warnings: List[str] = []
        for path, fmt in [
            (self._parquet_path(symbol, period, interval), "parquet"),
            (self._csv_path(symbol, period, interval), "csv"),
            (self._legacy_csv_path(symbol), "legacy_csv"),
        ]:
            if fmt == "legacy_csv" and not (period == "1y" and interval == "1d"):
                continue
            if not path.exists():
                continue
            try:
                if fmt == "parquet":
                    return self._clean_df(pd.read_parquet(path), interval), path, fmt, warnings
                return self._clean_df(pd.read_csv(path), interval), path, fmt, warnings
            except Exception as exc:
                warnings.append(f"Could not read {fmt} cache {path}: {exc}")
        return pd.DataFrame(), None, "none", warnings

    def _write_metadata_file(self, symbol: str, period: str, interval: str, metadata: Dict[str, Any]) -> None:
        try:
            self._metadata_path(symbol, period, interval).write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        except Exception:
            pass

    def _build_result(
        self,
        symbol: str,
        period: str,
        interval: str,
        df: pd.DataFrame,
        source: str,
        storage_format: str,
        file_path: Optional[Path] = None,
        warnings: Optional[List[str]] = None,
        stale_warning: bool = False,
        download_error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        latest = self._latest_timestamp(df)
        warnings = list(warnings or [])
        if stale_warning:
            warnings.append("Fresh download failed, so stale cached historical data was used.")
        return {
            "success": True,
            "symbol": self._symbol(symbol),
            "source": source,
            "period": period,
            "interval": interval,
            "file_path": str(file_path) if file_path else None,
            "storage_format": storage_format,
            "storage_mode": "database_first",
            "metadata_path": str(self._metadata_path(symbol, period, interval)),
            "metadata": metadata or {},
            "prices": df.to_dict("records"),
            "num_records": int(len(df)),
            "latest_date": str(latest.date()) if latest is not None else None,
            "latest_timestamp": str(latest) if latest is not None else None,
            "age_hours": self._age_hours(df),
            "is_stale": self._is_stale(df, interval),
            "stale_warning": bool(stale_warning),
            "download_error": download_error,
            "warnings": warnings,
            "summary": f"Loaded {len(df)} rows for {self._symbol(symbol)} ({period}, {interval}) from {source}.",
        }

    # --------------------------------------------------
    # Database read/write helpers
    # --------------------------------------------------
    def _load_from_database(self, symbol: str, period: str, interval: str) -> Dict[str, Any]:
        if not self.storage_agent:
            return {"success": False, "source": "database", "error": "StorageAgent is not available."}
        try:
            df = self.storage_agent.get_historical_prices(symbol, period=period, interval=interval, as_dataframe=True)
            if df is None or df.empty:
                return {"success": False, "source": "database", "error": "No database historical prices found."}
            df = self._clean_df(df, interval)
            metadata = self.storage_agent.get_historical_metadata(symbol, period=period, interval=interval) or {}
            return self._build_result(
                symbol=symbol,
                period=period,
                interval=interval,
                df=df,
                source="database",
                storage_format="database",
                metadata=metadata,
            )
        except Exception as exc:
            return {"success": False, "source": "database", "error": str(exc)}

    def _save_to_database(self, symbol: str, period: str, interval: str, df: pd.DataFrame, source: str, warnings: Optional[List[str]] = None) -> Dict[str, Any]:
        if not self.storage_agent:
            return {"success": False, "error": "StorageAgent is not available."}
        metadata = {
            "symbol": self._symbol(symbol),
            "period": period,
            "interval": interval,
            "latest_timestamp": str(self._latest_timestamp(df)),
            "downloaded_at_utc": self._now_iso(),
            "num_records": int(len(df)),
            "source": source,
            "storage_mode": "database",
            "warnings": warnings or [],
        }
        result = self.storage_agent.record_historical_prices(
            symbol=symbol,
            prices=df,
            period=period,
            interval=interval,
            source=source,
            metadata=metadata,
        )
        self._write_metadata_file(symbol, period, interval, metadata)
        return result

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def load_local_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        symbol = self._symbol(symbol)
        period, interval, normalise_warnings = self._normalise_period_interval(period, interval)
        if not symbol:
            return {"success": False, "symbol": symbol, "source": "local", "error": "Symbol is empty."}

        if self.database_first:
            db_result = self._load_from_database(symbol, period, interval)
            if db_result.get("success"):
                db_result.setdefault("warnings", [])
                db_result["warnings"] = normalise_warnings + list(db_result.get("warnings", []))
                return db_result

        file_df, path, fmt, file_warnings = self._read_file_cache(symbol, period, interval)
        warnings = normalise_warnings + file_warnings
        if file_df.empty:
            return {
                "success": False,
                "symbol": symbol,
                "source": "local_cache",
                "period": period,
                "interval": interval,
                "error": "No usable local historical data cache was found.",
                "warnings": warnings,
            }

        if self.database_first and self.storage_agent:
            try:
                self._save_to_database(symbol, period, interval, file_df, source=f"imported_{fmt}", warnings=warnings)
                warnings.append("Imported old file cache into database.")
            except Exception as exc:
                warnings.append(f"Could not import file cache into database: {exc}")

        return self._build_result(
            symbol=symbol,
            period=period,
            interval=interval,
            df=file_df,
            source="local_file_cache",
            storage_format=fmt,
            file_path=path,
            warnings=warnings,
        )

    def download_yfinance_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        symbol = self._symbol(symbol)
        period, interval, normalise_warnings = self._normalise_period_interval(period, interval)
        if not symbol:
            return {"success": False, "symbol": symbol, "source": "yfinance", "error": "Symbol is empty."}

        try:
            raw_df = yf.download(
                symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            df = self._clean_df(raw_df, interval)
            if df.empty:
                return {
                    "success": False,
                    "symbol": symbol,
                    "source": "yfinance",
                    "period": period,
                    "interval": interval,
                    "error": "No usable historical data was downloaded.",
                    "warnings": normalise_warnings,
                }

            db_result = self._save_to_database(symbol, period, interval, df, source="yfinance", warnings=normalise_warnings)
            file_path, storage_format, write_warnings = self._write_file_backup(symbol, period, interval, df, warnings=normalise_warnings)
            metadata = {
                "database_result": db_result,
                "downloaded_at_utc": self._now_iso(),
                "storage_mode": "database_first",
                "file_backup": str(file_path) if file_path else None,
            }
            return self._build_result(
                symbol=symbol,
                period=period,
                interval=interval,
                df=df,
                source="yfinance",
                storage_format=storage_format,
                file_path=file_path,
                metadata=metadata,
                warnings=write_warnings,
            )
        except Exception as exc:
            return {
                "success": False,
                "symbol": symbol,
                "source": "yfinance",
                "period": period,
                "interval": interval,
                "error": str(exc),
                "warnings": normalise_warnings,
            }

    def get_or_download_data(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        symbol = self._symbol(symbol)
        period, interval, normalise_warnings = self._normalise_period_interval(period, interval)

        local = self.load_local_data(symbol=symbol, period=period, interval=interval)
        if local.get("success"):
            local.setdefault("warnings", [])
            local["warnings"] = normalise_warnings + list(local.get("warnings", []))

        if local.get("success") and not force_refresh and not local.get("is_stale"):
            local["summary"] = f"Used fresh database/local historical data for {symbol} ({period}, {interval})."
            return local

        downloaded = self.download_yfinance_data(symbol=symbol, period=period, interval=interval)
        if downloaded.get("success"):
            if local.get("success") and local.get("is_stale"):
                downloaded["summary"] = f"Refreshed stale historical data for {symbol} ({period}, {interval}) and saved it to the database."
            return downloaded

        if local.get("success"):
            local["stale_warning"] = True
            local["download_error"] = downloaded.get("error")
            local.setdefault("warnings", [])
            local["warnings"].append("Fresh yfinance download failed; stale cached database/file data was used instead.")
            local["summary"] = f"Used stale historical data for {symbol} ({period}, {interval}) because refresh failed."
            return local

        return downloaded

    def get_available_cache_files(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        symbol_filter = self._symbol(symbol) if symbol else None
        files = []
        for path in sorted(self.data_dir.glob("*")):
            if path.suffix.lower() not in {".csv", ".parquet"}:
                continue
            if symbol_filter and not path.name.startswith(symbol_filter):
                continue
            files.append({"file": str(path), "size_bytes": path.stat().st_size, "format": path.suffix.replace(".", "")})
        db_summary = None
        if self.storage_agent:
            try:
                db_summary = self.storage_agent.get_storage_summary()
            except Exception:
                db_summary = None
        return {"success": True, "count": len(files), "files": files, "database_summary": db_summary}

    def run(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        return self.get_or_download_data(symbol=symbol, period=period, interval=interval)
