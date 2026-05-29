from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yfinance as yf


class HistoricalDataAgent:
    """
    Historical Data Agent

    Role:
    - Acts as the local memory layer for historical OHLCV data.
    - Supports Parquet first, with CSV fallback when a parquet engine is not installed.
    - Stores metadata such as latest data date, download time, period and interval.
    - If fresh download fails, it can safely fall back to stale cached data and mark stale_warning=True.

    Notes:
    - For yfinance, period and interval are different concepts.
      Examples:
        period="7d", interval="1h"
        period="30d", interval="1d"
        period="1y", interval="1d"
    - If the user passes interval="7d", "30d", or "1y", this agent treats that as a period preset
      and uses interval="1d" automatically. This keeps the UI/user experience simple.
    """

    VALID_INTERVALS = {
        "1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"
    }

    PERIOD_PRESET_ALIASES = {
        "7d": "7d",
        "30d": "30d",
        "1y": "1y",
        "1mo": "1mo",
        "3mo": "3mo",
        "6mo": "6mo",
        "2y": "2y",
        "5y": "5y",
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
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_dir = Path(metadata_dir) if metadata_dir else self.data_dir / "metadata"
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

        self.stale_days = stale_days
        self.prefer_parquet = prefer_parquet

    # --------------------------------------------------
    # Path and normalisation helpers
    # --------------------------------------------------
    def _symbol(self, symbol: str) -> str:
        return str(symbol or "").upper().strip()

    def _safe_name(self, text: str) -> str:
        return str(text or "").replace("/", "_").replace(" ", "_").replace(":", "_")

    def _normalise_period_interval(self, period: str = "1y", interval: str = "1d") -> Tuple[str, str, List[str]]:
        warnings: List[str] = []

        period = str(period or "1y").strip().lower()
        interval = str(interval or "1d").strip().lower()
        interval = self.INTERVAL_ALIASES.get(interval, interval)

        # User-friendly shortcut: interval="7d" / "30d" / "1y" means period preset + daily interval.
        if interval in self.PERIOD_PRESET_ALIASES:
            period = self.PERIOD_PRESET_ALIASES[interval]
            interval = "1d"
            warnings.append(
                f"Interpreted interval input as period='{period}' with interval='1d'."
            )

        # Another shortcut: period="1h" means recent hourly data.
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
        # Backward compatibility with older project versions: data/historical/AAPL.csv
        return self.data_dir / f"{self._symbol(symbol)}.csv"

    def _metadata_path(self, symbol: str, period: str, interval: str) -> Path:
        return self.metadata_dir / f"{self._base_stem(symbol, period, interval)}_metadata.json"

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # --------------------------------------------------
    # Data cleaning and metadata helpers
    # --------------------------------------------------
    def _clean_df(self, df: pd.DataFrame, interval: str = "1d") -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        # yfinance often returns Date/Datetime as index.
        if df.index.name is not None or "Date" in df.index.names or "Datetime" in df.index.names:
            df = df.reset_index()
        else:
            df = df.reset_index() if "index" not in df.columns and "date" not in [str(c).lower() for c in df.columns] else df

        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        rename_map = {
            "datetime": "date",
            "date": "date",
            "index": "date",
            "adj_close": "adj_close",
            "adjclose": "adj_close",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if "date" not in df.columns:
            for possible in ["price_date", "timestamp"]:
                if possible in df.columns:
                    df = df.rename(columns={possible: "date"})
                    break

        if "close" not in df.columns and "adj_close" in df.columns:
            df["close"] = df["adj_close"]

        required = ["date", "open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                return pd.DataFrame()

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=False)
        df = df.dropna(subset=required).sort_values("date")

        # Preserve intraday timestamps for intervals such as 1h. Keep daily records simple.
        intraday = interval in {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"}
        if intraday:
            df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
            df["date"] = df["date"].dt.strftime("%Y-%m-%d")

        return df[required].drop_duplicates(subset=["date"], keep="last")

    def _latest_timestamp(self, df: pd.DataFrame) -> Optional[pd.Timestamp]:
        if df.empty or "date" not in df.columns:
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
        if interval in {"1wk"}:
            return 10 * 24.0
        if interval in {"1mo", "3mo"}:
            return 45 * 24.0
        return float(self.stale_days * 24)

    def _is_stale(self, df: pd.DataFrame, interval: str = "1d") -> bool:
        latest = self._latest_timestamp(df)
        if latest is None:
            return True

        latest = latest.tz_localize(None) if latest.tzinfo is not None else latest
        now = pd.Timestamp.utcnow().tz_localize(None)
        age_hours = (now - latest).total_seconds() / 3600
        return age_hours > self._stale_threshold_hours(interval)

    def _age_hours(self, df: pd.DataFrame) -> Optional[float]:
        latest = self._latest_timestamp(df)
        if latest is None:
            return None
        latest = latest.tz_localize(None) if latest.tzinfo is not None else latest
        now = pd.Timestamp.utcnow().tz_localize(None)
        return round((now - latest).total_seconds() / 3600, 3)

    def _write_metadata(
        self,
        symbol: str,
        period: str,
        interval: str,
        df: pd.DataFrame,
        storage_format: str,
        file_path: Path,
        source: str,
        warnings: Optional[List[str]] = None,
        download_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        latest = self._latest_timestamp(df)
        metadata = {
            "symbol": self._symbol(symbol),
            "period": period,
            "interval": interval,
            "source": source,
            "storage_format": storage_format,
            "file_path": str(file_path),
            "num_records": int(len(df)) if df is not None else 0,
            "latest_date": str(latest.date()) if latest is not None else None,
            "latest_timestamp": str(latest) if latest is not None else None,
            "downloaded_at_utc": self._now_iso(),
            "age_hours_at_write": self._age_hours(df) if df is not None and not df.empty else None,
            "warnings": warnings or [],
            "download_error": download_error,
        }

        path = self._metadata_path(symbol, period, interval)
        try:
            path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

        return metadata

    def _read_metadata(self, symbol: str, period: str, interval: str) -> Dict[str, Any]:
        path = self._metadata_path(symbol, period, interval)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    # --------------------------------------------------
    # Storage helpers
    # --------------------------------------------------
    def _read_cached_df(self, symbol: str, period: str, interval: str) -> Tuple[pd.DataFrame, Optional[Path], str, List[str]]:
        warnings: List[str] = []
        parquet_path = self._parquet_path(symbol, period, interval)
        csv_path = self._csv_path(symbol, period, interval)
        legacy_path = self._legacy_csv_path(symbol)

        if parquet_path.exists():
            try:
                return self._clean_df(pd.read_parquet(parquet_path), interval), parquet_path, "parquet", warnings
            except Exception as exc:
                warnings.append(f"Could not read parquet cache: {exc}")

        if csv_path.exists():
            try:
                return self._clean_df(pd.read_csv(csv_path), interval), csv_path, "csv", warnings
            except Exception as exc:
                warnings.append(f"Could not read CSV cache: {exc}")

        # Legacy fallback only for default daily data.
        if period == "1y" and interval == "1d" and legacy_path.exists():
            try:
                warnings.append("Loaded legacy cache file. It will be migrated on next successful download.")
                return self._clean_df(pd.read_csv(legacy_path), interval), legacy_path, "legacy_csv", warnings
            except Exception as exc:
                warnings.append(f"Could not read legacy CSV cache: {exc}")

        return pd.DataFrame(), None, "none", warnings

    def _write_cached_df(self, symbol: str, period: str, interval: str, df: pd.DataFrame, warnings: Optional[List[str]] = None) -> Tuple[Path, str, List[str]]:
        warnings = list(warnings or [])

        if self.prefer_parquet:
            parquet_path = self._parquet_path(symbol, period, interval)
            try:
                df.to_parquet(parquet_path, index=False)
                return parquet_path, "parquet", warnings
            except Exception as exc:
                warnings.append(
                    "Parquet save failed, so CSV fallback was used. "
                    "Install pyarrow for faster Parquet support. "
                    f"Error: {exc}"
                )

        csv_path = self._csv_path(symbol, period, interval)
        df.to_csv(csv_path, index=False)
        return csv_path, "csv", warnings

    def _build_success_result(
        self,
        symbol: str,
        period: str,
        interval: str,
        df: pd.DataFrame,
        source: str,
        file_path: Optional[Path],
        storage_format: str,
        metadata: Optional[Dict[str, Any]] = None,
        warnings: Optional[List[str]] = None,
        stale_warning: bool = False,
        download_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        latest = self._latest_timestamp(df)
        is_stale = self._is_stale(df, interval)
        warnings = list(warnings or [])

        if stale_warning:
            warnings.append("Fresh download failed, so stale local cache was used.")

        return {
            "success": True,
            "symbol": self._symbol(symbol),
            "source": source,
            "period": period,
            "interval": interval,
            "file_path": str(file_path) if file_path else None,
            "storage_format": storage_format,
            "metadata_path": str(self._metadata_path(symbol, period, interval)),
            "metadata": metadata or self._read_metadata(symbol, period, interval),
            "prices": df.to_dict("records"),
            "num_records": int(len(df)),
            "latest_date": str(latest.date()) if latest is not None else None,
            "latest_timestamp": str(latest) if latest is not None else None,
            "age_hours": self._age_hours(df),
            "is_stale": is_stale,
            "stale_warning": bool(stale_warning),
            "download_error": download_error,
            "warnings": warnings,
            "summary": (
                f"Loaded {len(df)} rows for {self._symbol(symbol)} "
                f"({period}, {interval}) from {source}."
                + (" Stale cache was used because refresh failed." if stale_warning else "")
            ),
        }

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def load_local_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        symbol = self._symbol(symbol)
        period, interval, normalise_warnings = self._normalise_period_interval(period, interval)

        if not symbol:
            return {"success": False, "symbol": symbol, "source": "local_cache", "error": "Symbol is empty."}

        df, path, storage_format, read_warnings = self._read_cached_df(symbol, period, interval)
        warnings = normalise_warnings + read_warnings

        if df.empty or path is None:
            return {
                "success": False,
                "symbol": symbol,
                "source": "local_cache",
                "period": period,
                "interval": interval,
                "error": "No usable local historical data cache was found.",
                "warnings": warnings,
            }

        return self._build_success_result(
            symbol=symbol,
            period=period,
            interval=interval,
            df=df,
            source="local_cache",
            file_path=path,
            storage_format=storage_format,
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

            file_path, storage_format, write_warnings = self._write_cached_df(
                symbol=symbol,
                period=period,
                interval=interval,
                df=df,
                warnings=normalise_warnings,
            )

            metadata = self._write_metadata(
                symbol=symbol,
                period=period,
                interval=interval,
                df=df,
                storage_format=storage_format,
                file_path=file_path,
                source="yfinance",
                warnings=write_warnings,
            )

            return self._build_success_result(
                symbol=symbol,
                period=period,
                interval=interval,
                df=df,
                source="yfinance",
                file_path=file_path,
                storage_format=storage_format,
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
            local["warnings"] = list(normalise_warnings) + list(local.get("warnings", []))

        if local.get("success") and not force_refresh and not local.get("is_stale"):
            local["summary"] = f"Used fresh local cache for {symbol} ({period}, {interval})."
            return local

        downloaded = self.download_yfinance_data(symbol=symbol, period=period, interval=interval)

        if downloaded.get("success"):
            if local.get("success") and local.get("is_stale"):
                downloaded["summary"] = f"Refreshed stale local data for {symbol} ({period}, {interval})."
            return downloaded

        # Important safety fallback: use stale cache if download fails.
        if local.get("success"):
            local["stale_warning"] = True
            local["download_error"] = downloaded.get("error")
            local.setdefault("warnings", [])
            local["warnings"].append("Fresh yfinance download failed; stale cache was used instead.")
            local["summary"] = (
                f"Used stale local historical data for {symbol} ({period}, {interval}) because refresh failed."
            )
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

        return {"success": True, "count": len(files), "files": files}

    def run(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        return self.get_or_download_data(symbol=symbol, period=period, interval=interval)
