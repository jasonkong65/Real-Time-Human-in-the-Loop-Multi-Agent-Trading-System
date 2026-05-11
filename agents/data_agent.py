import os
import requests
from pathlib import Path
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


class DataAgent:
    """
    Data Agent: Collects and processes financial data from APIs.
    Finnhub is the primary source.
    iTick is the secondary source for multi-source validation.
    """

    def __init__(self):
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        self.itick_api_key = os.getenv("ITICK_API_KEY")

        self.finnhub_url = "https://finnhub.io/api/v1/quote"
        self.itick_url = "https://api.itick.org/stock/quote"

    def get_finnhub_quote(self, symbol: str) -> dict:
        symbol = symbol.upper().strip()

        if not self.finnhub_api_key:
            return {
                "source": "Finnhub",
                "success": False,
                "symbol": symbol,
                "error": "Finnhub API key not found. Please set FINNHUB_API_KEY in your environment variables."
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
                    "error": f"Finnhub returned status code {response.status_code}",
                    "raw_response": response.text[:300]
                }

            data = response.json()

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

        except Exception as e:
            return {
                "source": "Finnhub",
                "success": False,
                "symbol": symbol,
                "error": f"Finnhub request failed: {str(e)}"
            }

    def get_itick_quote(self, symbol: str, region: str = "US") -> dict:
        """
        Get iTick stock quote.

        For US stocks:
            region = "US"
            code = "AAPL"

        For HK stocks:
            region = "HK"
            code = "700"

        For A-shares:
            region = "SH" or "SZ"
            code = stock code, for example "600519"
        """
        symbol = symbol.upper().strip()
        region = region.upper().strip()

        if not self.itick_api_key:
            return {
                "source": "iTick",
                "success": False,
                "symbol": symbol,
                "region": region,
                "error": "iTick API key not found. Please set ITICK_API_KEY in your environment variables."
            }

        try:
            headers = {
                "accept": "application/json",
                "Content-Type": "application/json",
                "token": self.itick_api_key.strip()
            }

            response = requests.get(
                self.itick_url,
                params={
                    "region": region,
                    "code": symbol
                },
                headers=headers,
                timeout=10
            )

            if response.status_code != 200:
                return {
                    "source": "iTick",
                    "success": False,
                    "symbol": symbol,
                    "region": region,
                    "error": f"iTick returned status code {response.status_code}",
                    "raw_response": response.text[:300]
                }

            data = response.json()

            if data.get("code") != 0:
                return {
                    "source": "iTick",
                    "success": False,
                    "symbol": symbol,
                    "region": region,
                    "error": f"iTick API error: {data.get('msg')}",
                    "raw_data": data
                }

            quote_data = data.get("data", {})

            timestamp = quote_data.get("t")
            if timestamp and timestamp > 10_000_000_000:
                timestamp = int(timestamp / 1000)

            return {
                "source": "iTick",
                "success": True,
                "symbol": symbol,
                "region": region,
                "current_price": quote_data.get("ld"),
                "timestamp": timestamp,
                "raw_data": data
            }

        except Exception as e:
            return {
                "source": "iTick",
                "success": False,
                "symbol": symbol,
                "region": region,
                "error": f"iTick request failed: {str(e)}"
            }

    def get_multi_source_quote(self, symbol: str, region: str = "US") -> dict:
        finnhub_quote = self.get_finnhub_quote(symbol)
        itick_quote = self.get_itick_quote(symbol, region)

        return {
            "symbol": symbol.upper().strip(),
            "region": region.upper().strip(),
            "finnhub": finnhub_quote,
            "itick": itick_quote
        }

    def get_live_quote(self, symbol: str) -> dict:
        return self.get_finnhub_quote(symbol)