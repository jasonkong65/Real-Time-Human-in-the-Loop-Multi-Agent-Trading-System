import os
import requests
from pathlib import Path
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class DataAgent:
    """
    Data Agent:
    Collects and standardizes financial data from external APIs.
    Finnhub is used as the primary live market data source.
    Alpha Vantage is used as a secondary reference source for multi-source validation.
    """

    def __init__(self):
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        self.alpha_vantage_api_key = os.getenv("ALPHA_VANTAGE_API_KEY")

        if self.finnhub_api_key:
            self.finnhub_api_key = self.finnhub_api_key.strip()

        if self.alpha_vantage_api_key:
            self.alpha_vantage_api_key = self.alpha_vantage_api_key.strip()

        self.finnhub_url = "https://finnhub.io/api/v1/quote"
        self.alpha_vantage_url = "https://www.alphavantage.co/query"

    def get_finnhub_quote(self, symbol: str) -> dict:
        """
        Get live quote data from Finnhub.
        """
        symbol = symbol.upper().strip()

        if not symbol:
            return {
                "source": "Finnhub",
                "success": False,
                "symbol": symbol,
                "error": "Stock symbol is empty."
            }

        if not self.finnhub_api_key:
            return {
                "source": "Finnhub",
                "success": False,
                "symbol": symbol,
                "error": "Finnhub API key not found. Please set FINNHUB_API_KEY in .env."
            }

        try:
            response = requests.get(
                self.finnhub_url,
                params={
                    "symbol": symbol,
                    "token": self.finnhub_api_key
                },
                timeout=10
            )

            if response.status_code != 200:
                return {
                    "source": "Finnhub",
                    "success": False,
                    "symbol": symbol,
                    "error": f"Finnhub returned status code {response.status_code}.",
                    "raw_response": response.text[:300]
                }

            try:
                data = response.json()
            except ValueError:
                return {
                    "source": "Finnhub",
                    "success": False,
                    "symbol": symbol,
                    "error": "Finnhub response is not valid JSON.",
                    "raw_response": response.text[:300]
                }

            return {
                "source": "Finnhub",
                "success": True,
                "symbol": symbol,
                "current_price": data.get("c"),
                "high_price": data.get("h"),
                "low_price": data.get("l"),
                "open_price": data.get("o"),
                "previous_close_price": data.get("pc"),
                "timestamp": data.get("t"),
                "raw_data": data
            }

        except requests.exceptions.Timeout:
            return {
                "source": "Finnhub",
                "success": False,
                "symbol": symbol,
                "error": "Finnhub request timed out."
            }

        except requests.exceptions.RequestException as e:
            return {
                "source": "Finnhub",
                "success": False,
                "symbol": symbol,
                "error": f"Finnhub request failed: {str(e)}"
            }

        except Exception as e:
            return {
                "source": "Finnhub",
                "success": False,
                "symbol": symbol,
                "error": f"Unexpected Finnhub error: {str(e)}"
            }

    def get_alpha_vantage_quote(self, symbol: str) -> dict:
        """
        Get quote data from Alpha Vantage as a secondary reference source.
        """
        symbol = symbol.upper().strip()

        if not symbol:
            return {
                "source": "Alpha Vantage",
                "success": False,
                "symbol": symbol,
                "error": "Stock symbol is empty."
            }

        if not self.alpha_vantage_api_key:
            return {
                "source": "Alpha Vantage",
                "success": False,
                "symbol": symbol,
                "error": "Alpha Vantage API key not found. Please set ALPHA_VANTAGE_API_KEY in .env."
            }

        try:
            response = requests.get(
                self.alpha_vantage_url,
                params={
                    "function": "GLOBAL_QUOTE",
                    "symbol": symbol,
                    "apikey": self.alpha_vantage_api_key
                },
                timeout=10
            )

            if response.status_code != 200:
                return {
                    "source": "Alpha Vantage",
                    "success": False,
                    "symbol": symbol,
                    "error": f"Alpha Vantage returned status code {response.status_code}.",
                    "raw_response": response.text[:300]
                }

            try:
                data = response.json()
            except ValueError:
                return {
                    "source": "Alpha Vantage",
                    "success": False,
                    "symbol": symbol,
                    "error": "Alpha Vantage response is not valid JSON.",
                    "raw_response": response.text[:300]
                }

            if "Note" in data:
                return {
                    "source": "Alpha Vantage",
                    "success": False,
                    "symbol": symbol,
                    "error": "Alpha Vantage API rate limit reached.",
                    "raw_data": data
                }

            if "Information" in data:
                return {
                    "source": "Alpha Vantage",
                    "success": False,
                    "symbol": symbol,
                    "error": data.get("Information"),
                    "raw_data": data
                }

            if "Error Message" in data:
                return {
                    "source": "Alpha Vantage",
                    "success": False,
                    "symbol": symbol,
                    "error": data.get("Error Message"),
                    "raw_data": data
                }

            quote = data.get("Global Quote", {})

            if not quote:
                return {
                    "source": "Alpha Vantage",
                    "success": False,
                    "symbol": symbol,
                    "error": "Alpha Vantage returned empty quote data.",
                    "raw_data": data
                }

            current_price = quote.get("05. price")
            previous_close = quote.get("08. previous close")
            latest_trading_day = quote.get("07. latest trading day")

            return {
                "source": "Alpha Vantage",
                "success": True,
                "symbol": symbol,
                "current_price": float(current_price) if current_price else None,
                "previous_close_price": float(previous_close) if previous_close else None,
                "timestamp": latest_trading_day,
                "raw_data": data
            }

        except requests.exceptions.Timeout:
            return {
                "source": "Alpha Vantage",
                "success": False,
                "symbol": symbol,
                "error": "Alpha Vantage request timed out."
            }

        except requests.exceptions.RequestException as e:
            return {
                "source": "Alpha Vantage",
                "success": False,
                "symbol": symbol,
                "error": f"Alpha Vantage request failed: {str(e)}"
            }

        except Exception as e:
            return {
                "source": "Alpha Vantage",
                "success": False,
                "symbol": symbol,
                "error": f"Unexpected Alpha Vantage error: {str(e)}"
            }


    def get_historical_daily_prices(self, symbol: str, outputsize: str = "compact") -> dict:
        """
        Get historical daily OHLCV price data from Alpha Vantage.

        outputsize:
            compact = latest 100 data points
            full = full-length historical daily data
        """
        symbol = symbol.upper().strip()

        if not symbol:
            return {
                "source": "Alpha Vantage",
                "success": False,
                "symbol": symbol,
                "error": "Stock symbol is empty."
            }

        if not self.alpha_vantage_api_key:
            return {
                "source": "Alpha Vantage",
                "success": False,
                "symbol": symbol,
                "error": "Alpha Vantage API key not found. Please set ALPHA_VANTAGE_API_KEY in .env."
            }

        try:
            response = requests.get(
                self.alpha_vantage_url,
                params={
                    "function": "TIME_SERIES_DAILY",
                    "symbol": symbol,
                    "outputsize": outputsize,
                    "apikey": self.alpha_vantage_api_key
                },
                timeout=10
            )

            if response.status_code != 200:
                return {
                    "source": "Alpha Vantage",
                    "success": False,
                    "symbol": symbol,
                    "error": f"Alpha Vantage historical API returned status code {response.status_code}.",
                    "raw_response": response.text[:300]
                }

            try:
                data = response.json()
            except ValueError:
                return {
                    "source": "Alpha Vantage",
                    "success": False,
                    "symbol": symbol,
                    "error": "Alpha Vantage historical response is not valid JSON.",
                    "raw_response": response.text[:300]
                }

            if "Note" in data:
                return {
                    "source": "Alpha Vantage",
                    "success": False,
                    "symbol": symbol,
                    "error": "Alpha Vantage API rate limit reached.",
                    "raw_data": data
                }

            if "Information" in data:
                return {
                    "source": "Alpha Vantage",
                    "success": False,
                    "symbol": symbol,
                    "error": data.get("Information"),
                    "raw_data": data
                }

            if "Error Message" in data:
                return {
                    "source": "Alpha Vantage",
                    "success": False,
                    "symbol": symbol,
                    "error": data.get("Error Message"),
                    "raw_data": data
                }

            time_series = data.get("Time Series (Daily)", {})

            if not time_series:
                return {
                    "source": "Alpha Vantage",
                    "success": False,
                    "symbol": symbol,
                    "error": "Alpha Vantage returned empty historical daily data.",
                    "raw_data": data
                }

            prices = []

            for date, values in time_series.items():
                prices.append({
                    "date": date,
                    "open": float(values.get("1. open")),
                    "high": float(values.get("2. high")),
                    "low": float(values.get("3. low")),
                    "close": float(values.get("4. close")),
                    "volume": float(values.get("5. volume"))
                })

            # Alpha Vantage returns newest first, so sort oldest to newest
            prices = sorted(prices, key=lambda x: x["date"])

            return {
                "source": "Alpha Vantage",
                "success": True,
                "symbol": symbol,
                "prices": prices,
                "raw_data": data
            }

        except requests.exceptions.Timeout:
            return {
                "source": "Alpha Vantage",
                "success": False,
                "symbol": symbol,
                "error": "Alpha Vantage historical request timed out."
            }

        except requests.exceptions.RequestException as e:
            return {
                "source": "Alpha Vantage",
                "success": False,
                "symbol": symbol,
                "error": f"Alpha Vantage historical request failed: {str(e)}"
            }

        except Exception as e:
            return {
                "source": "Alpha Vantage",
                "success": False,
                "symbol": symbol,
                "error": f"Unexpected Alpha Vantage historical error: {str(e)}"
            }

    def get_multi_source_quote(self, symbol: str) -> dict:
        """
        Get quote data from both Finnhub and Alpha Vantage.

        This method allows the Validation Agent to compare different sources.
        """
        symbol = symbol.upper().strip()

        finnhub_quote = self.get_finnhub_quote(symbol)
        alpha_vantage_quote = self.get_alpha_vantage_quote(symbol)

        return {
            "symbol": symbol,
            "primary_source": "Finnhub",
            "secondary_source": "Alpha Vantage",
            "finnhub": finnhub_quote,
            "alpha_vantage": alpha_vantage_quote
        }

    def get_live_quote(self, symbol: str) -> dict:
        """
        Keep this method for simple single-source testing.
        """
        return self.get_finnhub_quote(symbol)