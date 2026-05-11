import os
import requests
from dotenv import load_dotenv

load_dotenv()

class DataAgent:

    "Data Agent: Collects and processes financial data from APIs."

    def __init__(self):
        self.finnhub_api_key = os.getenv("FINNHUB_API_KEY")
        self.base_url = "https://finnhub.io/api/v1"

    def get_live_quote(self, symbol: str) -> dict:
        
        """Fetches live stock quote for a given symbol."""
        
        symbol = symbol.upper().strip()

        if not self.finnhub_api_key:
            return{
                "success": False,
                "symbol": symbol,
                "error": "Finnhub API key not found. Please set FINNHUB_API_KEY in your environment variables."
            }
        
        if not symbol:
            return {
                "success": False,
                "symbol": symbol,
                "error": "Invalid symbol provided. Please provide a non-empty stock symbol."
            }
        
        try:
            reponse = requests.get(
                self.base_url,
                params={
                    "symbol": symbol,
                    "token": self.finnhub_api_key
                },
                timeout=10
            )

            reponse.raise_for_status()
            data = reponse.json()

            return {
                "success": True,
                "symbol": symbol,
                "data": data,
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
                "success": False,
                "symbol": symbol,
                "error": "Request timed out. Please try again later."
            }
        
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "symbol": symbol,
                "error": f"An error occurred while fetching data: {str(e)}"
            }