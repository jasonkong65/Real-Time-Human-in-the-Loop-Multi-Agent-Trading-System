from pathlib import Path
import pandas as pd
import yfinance as yf


class HistoricalDataAgent:
    """
    Historical Data Agent:
    Automatically finds or downloads historical OHLCV data for a given stock symbol.

    It works as a local data memory/cache layer for the Training Agent.
    """

    def __init__(self, data_dir: str = "data/historical"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_file_path(self, symbol: str) -> Path:
        symbol = symbol.upper().strip()
        return self.data_dir / f"{symbol}.csv"

    def load_local_data(self, symbol: str) -> dict:
        """
        Load local historical data if it already exists.
        """
        symbol = symbol.upper().strip()
        file_path = self._get_file_path(symbol)

        if not file_path.exists():
            return {
                "success": False,
                "symbol": symbol,
                "error": f"Local historical data not found: {file_path}"
            }

        try:
            df = pd.read_csv(file_path)
            df.columns = [c.lower().strip() for c in df.columns]

            required_cols = ["date", "open", "high", "low", "close", "volume"]
            missing_cols = [c for c in required_cols if c not in df.columns]

            if missing_cols:
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": f"Missing columns in local data: {missing_cols}"
                }

            price_records = df[required_cols].to_dict("records")

            return {
                "success": True,
                "source": "local_cache",
                "symbol": symbol,
                "prices": price_records,
                "file_path": str(file_path),
                "num_records": len(price_records)
            }

        except Exception as e:
            return {
                "success": False,
                "symbol": symbol,
                "error": f"Failed to load local historical data: {str(e)}"
            }

    def download_yfinance_data(
        self,
        symbol: str,
        period: str = "2y",
        interval: str = "1d"
    ) -> dict:
        """
        Download historical OHLCV data from yfinance and cache it locally.
        """
        symbol = symbol.upper().strip()
        file_path = self._get_file_path(symbol)

        try:
            df = yf.download(
                tickers=symbol,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False
            )

            if df.empty:
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": "Downloaded historical data is empty."
                }

            df = df.reset_index()

            # yfinance may return MultiIndex columns in some cases
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    col[0] if isinstance(col, tuple) else col
                    for col in df.columns
                ]

            df = df.rename(columns={
                "Date": "date",
                "Datetime": "date",
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Volume": "volume"
            })

            required_cols = ["date", "open", "high", "low", "close", "volume"]

            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": f"Downloaded data missing columns: {missing_cols}"
                }

            df = df[required_cols].copy()
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            df = df.dropna()

            if df.empty:
                return {
                    "success": False,
                    "symbol": symbol,
                    "error": "Historical data became empty after cleaning."
                }

            df.to_csv(file_path, index=False)

            return {
                "success": True,
                "source": "yfinance",
                "symbol": symbol,
                "file_path": str(file_path),
                "num_records": len(df),
                "prices": df.to_dict("records")
            }

        except Exception as e:
            return {
                "success": False,
                "symbol": symbol,
                "error": f"yfinance download failed: {str(e)}"
            }

    def get_or_download_data(self, symbol: str, period: str = "2y") -> dict:
        """
        Main method:
        1. Try local cache first.
        2. If not found, download from yfinance.
        """
        local_result = self.load_local_data(symbol)

        if local_result.get("success"):
            return local_result

        return self.download_yfinance_data(symbol, period=period)