from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yfinance as yf


class HistoricalDataAgent:
    """
    Historical Data Agent

    Acts as a local memory layer for historical OHLCV data. It refreshes stale
    files automatically and returns a clean list of price records for the
    Analyst and Training agents.
    """

    def __init__(self, data_dir: str = "data/historical", stale_days: int = 3):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.stale_days = stale_days

    def _path(self, symbol: str) -> Path:
        return self.data_dir / f"{str(symbol).upper().strip()}.csv"

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df.reset_index() if "Date" in df.index.names or "Datetime" in df.index.names else df.copy()
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        rename_map = {"adj_close": "adj_close", "datetime": "date"}
        df = df.rename(columns=rename_map)
        if "date" not in df.columns:
            for possible in ["index", "price_date"]:
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
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=required).sort_values("date")
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")
        return df[required].drop_duplicates(subset=["date"], keep="last")

    def _latest_date(self, df: pd.DataFrame) -> Optional[pd.Timestamp]:
        if df.empty or "date" not in df.columns:
            return None
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if dates.empty:
            return None
        return dates.max()

    def _is_stale(self, df: pd.DataFrame) -> bool:
        latest = self._latest_date(df)
        if latest is None:
            return True
        now = pd.Timestamp(datetime.now(timezone.utc).date())
        return (now - latest.normalize()).days > self.stale_days

    def load_local_data(self, symbol: str) -> Dict[str, Any]:
        symbol = str(symbol or "").upper().strip()
        path = self._path(symbol)
        if not path.exists():
            return {"success": False, "symbol": symbol, "source": "local_cache", "error": f"No local file: {path}"}
        try:
            df = self._clean_df(pd.read_csv(path))
            if df.empty:
                return {"success": False, "symbol": symbol, "source": "local_cache", "error": "Local file could not be cleaned."}
            return {
                "success": True,
                "symbol": symbol,
                "source": "local_cache",
                "file_path": str(path),
                "prices": df.to_dict("records"),
                "num_records": int(len(df)),
                "latest_date": str(self._latest_date(df).date()),
                "is_stale": self._is_stale(df),
                "summary": f"Loaded {len(df)} historical rows for {symbol} from local cache.",
            }
        except Exception as exc:
            return {"success": False, "symbol": symbol, "source": "local_cache", "error": str(exc)}

    def download_yfinance_data(self, symbol: str, period: str = "1y", interval: str = "1d") -> Dict[str, Any]:
        symbol = str(symbol or "").upper().strip()
        if not symbol:
            return {"success": False, "symbol": symbol, "source": "yfinance", "error": "Symbol is empty."}
        try:
            df = yf.download(symbol, period=period, interval=interval, auto_adjust=False, progress=False)
            df = self._clean_df(df)
            if df.empty:
                return {"success": False, "symbol": symbol, "source": "yfinance", "error": "No usable historical data was downloaded."}
            path = self._path(symbol)
            df.to_csv(path, index=False)
            return {
                "success": True,
                "symbol": symbol,
                "source": "yfinance",
                "file_path": str(path),
                "prices": df.to_dict("records"),
                "num_records": int(len(df)),
                "latest_date": str(self._latest_date(df).date()),
                "is_stale": False,
                "summary": f"Downloaded and saved {len(df)} historical rows for {symbol}.",
            }
        except Exception as exc:
            return {"success": False, "symbol": symbol, "source": "yfinance", "error": str(exc)}

    def get_or_download_data(self, symbol: str, period: str = "1y", interval: str = "1d", force_refresh: bool = False) -> Dict[str, Any]:
        local = self.load_local_data(symbol)
        if local.get("success") and not force_refresh and not local.get("is_stale"):
            return local

        downloaded = self.download_yfinance_data(symbol, period=period, interval=interval)
        if downloaded.get("success"):
            if local.get("success") and local.get("is_stale"):
                downloaded["summary"] = f"Refreshed stale local data for {str(symbol).upper().strip()}."
            return downloaded

        if local.get("success"):
            local["warning"] = "Fresh download failed, so stale local data was used."
            local["summary"] = f"Used local historical data for {str(symbol).upper().strip()} because refresh failed."
            return local
        return downloaded

    def run(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        return self.get_or_download_data(symbol=symbol, period=period)
