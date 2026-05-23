from typing import List, Dict, Any

from agents.historical_data_agent import HistoricalDataAgent
from utils.features import build_trading_features


class ScreenerAgent:
    """
    S&P-style Screener Agent:
    Screens a configurable large-cap stock universe and ranks stocks into:
    - Top Buy Candidates for further research
    - Highest Risk / Caution Candidates

    This is a lightweight market screener prototype.
    It does not scan the entire market and does not run the full agent pipeline
    for every stock.
    """

    DEFAULT_UNIVERSE = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
        "META", "TSLA", "AVGO", "JPM", "V",
        "MA", "UNH", "HD", "COST", "NFLX",
        "AMD", "CRM", "ADBE", "PEP", "KO",
        "BAC", "WMT", "DIS", "MCD", "CSCO",
        "INTC", "QCOM", "TXN", "AMAT", "ORCL"
    ]

    def __init__(self):
        self.historical_data_agent = HistoricalDataAgent()

    @staticmethod
    def _safe_float(value, default=None):
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, value))

    def _score_rsi(self, rsi: float) -> float:
        """
        RSI score:
        - 45–65 is healthy
        - 65–75 is strong but should be monitored
        - >75 may be overbought
        - <30 may indicate weakness or oversold condition
        """
        if rsi is None:
            return 0.5

        if 45 <= rsi <= 65:
            return 0.85
        elif 65 < rsi <= 75:
            return 0.65
        elif 35 <= rsi < 45:
            return 0.60
        elif 30 <= rsi < 35:
            return 0.45
        elif rsi > 75:
            return 0.35
        else:
            return 0.30

    def _classify_signal(
        self,
        buy_score: float,
        risk_score: float,
        rsi_14: float,
        volatility_20: float
    ) -> str:
        """
        Convert buy/risk scores into a more risk-aware screen signal.

        This avoids calling a stock a direct BUY_CANDIDATE when it is already
        technically overbought.
        """
        if risk_score >= 0.65:
            return "SELL_RISK"

        if buy_score >= 0.70 and rsi_14 <= 75 and volatility_20 <= 0.05:
            return "BUY_CANDIDATE"

        if buy_score >= 0.70 and rsi_14 > 75:
            return "BUY_WATCHLIST_OVERBOUGHT"

        if buy_score >= 0.65:
            return "WATCHLIST_BUY_MONITOR"

        return "WATCHLIST_HOLD"

    def _score_one_stock(self, symbol: str, period: str = "1y") -> Dict[str, Any]:
        """
        Score one stock using historical OHLCV features.
        """
        symbol = symbol.upper().strip()

        historical_data = self.historical_data_agent.get_or_download_data(
            symbol=symbol,
            period=period
        )

        if not historical_data.get("success"):
            return {
                "success": False,
                "symbol": symbol,
                "error": historical_data.get("error", "Historical data unavailable.")
            }

        price_records = historical_data.get("prices", [])
        feature_df = build_trading_features(price_records)

        if feature_df.empty:
            return {
                "success": False,
                "symbol": symbol,
                "error": "Feature construction failed or not enough historical data."
            }

        latest = feature_df.iloc[-1]

        return_5 = self._safe_float(latest.get("return_5"), 0.0)
        return_20 = self._safe_float(latest.get("return_20"), 0.0)
        ma_gap = self._safe_float(latest.get("ma_gap"), 0.0)
        volatility_20 = self._safe_float(latest.get("volatility_20"), 0.03)
        rsi_14 = self._safe_float(latest.get("rsi_14"), 50.0)
        volume_change = self._safe_float(latest.get("volume_change"), 0.0)

        momentum_score = self._clip(0.50 + return_5 * 3.0 + return_20 * 1.5)
        trend_score = self._clip(0.50 + ma_gap * 4.0 + return_20 * 1.2)
        volatility_score = self._clip(1.0 - volatility_20 * 15.0)
        rsi_score = self._score_rsi(rsi_14)
        volume_score = self._clip(0.50 + volume_change * 0.4)

        buy_score = (
            0.30 * momentum_score
            + 0.30 * trend_score
            + 0.20 * volatility_score
            + 0.15 * rsi_score
            + 0.05 * volume_score
        )

        negative_momentum_score = self._clip(0.50 - return_5 * 3.0 - return_20 * 1.5)
        downtrend_score = self._clip(0.50 - ma_gap * 4.0 - return_20 * 1.2)
        volatility_risk_score = self._clip(volatility_20 * 15.0)

        if rsi_14 > 75:
            rsi_risk_score = 0.80
        elif rsi_14 < 30:
            rsi_risk_score = 0.70
        else:
            rsi_risk_score = 0.35

        risk_score = (
            0.30 * negative_momentum_score
            + 0.30 * downtrend_score
            + 0.25 * volatility_risk_score
            + 0.15 * rsi_risk_score
        )

        screen_signal = self._classify_signal(
            buy_score=buy_score,
            risk_score=risk_score,
            rsi_14=rsi_14,
            volatility_20=volatility_20
        )

        reason_parts = []

        if return_20 > 0.05:
            reason_parts.append("positive 20-day momentum")
        elif return_20 < -0.05:
            reason_parts.append("negative 20-day momentum")

        if ma_gap > 0.03:
            reason_parts.append("short-term moving average above medium-term average")
        elif ma_gap < -0.03:
            reason_parts.append("short-term moving average below medium-term average")

        if volatility_20 > 0.04:
            reason_parts.append("high volatility")
        elif volatility_20 < 0.02:
            reason_parts.append("low volatility")

        if rsi_14 > 75:
            reason_parts.append("RSI may be overbought")
        elif rsi_14 < 30:
            reason_parts.append("RSI may be oversold")

        if screen_signal == "BUY_WATCHLIST_OVERBOUGHT":
            reason_parts.append("strong trend but higher entry risk due to overbought RSI")

        if screen_signal == "WATCHLIST_BUY_MONITOR":
            reason_parts.append("positive setup but not strong enough for direct buy candidate")

        if not reason_parts:
            reason_parts.append("mixed technical signals")

        return {
            "success": True,
            "symbol": symbol,
            "source": historical_data.get("source"),
            "latest_feature_date": str(latest.get("date")),
            "buy_score": round(buy_score, 3),
            "risk_score": round(risk_score, 3),
            "screen_signal": screen_signal,
            "return_5": round(return_5, 4),
            "return_20": round(return_20, 4),
            "ma_gap": round(ma_gap, 4),
            "volatility_20": round(volatility_20, 4),
            "rsi_14": round(rsi_14, 2),
            "volume_change": round(volume_change, 4),
            "reason": "; ".join(reason_parts)
        }

    def _add_rank(self, rows: List[Dict[str, Any]], score_key: str) -> List[Dict[str, Any]]:
        ranked_rows = sorted(
            rows,
            key=lambda x: x.get(score_key, 0),
            reverse=True
        )

        output = []
        for idx, row in enumerate(ranked_rows, start=1):
            row_copy = row.copy()
            row_copy["rank"] = idx
            output.append(row_copy)

        return output

    def screen_universe(
        self,
        symbols: List[str] = None,
        top_n: int = 10,
        period: str = "1y"
    ) -> Dict[str, Any]:
        """
        Screen a stock universe and return top buy candidates and caution candidates.
        """
        if symbols is None:
            symbols = self.DEFAULT_UNIVERSE

        clean_symbols = []
        for symbol in symbols:
            symbol = symbol.upper().strip()
            if symbol and symbol not in clean_symbols:
                clean_symbols.append(symbol)

        results = []
        failed = []

        for symbol in clean_symbols:
            result = self._score_one_stock(symbol, period=period)

            if result.get("success"):
                results.append(result)
            else:
                failed.append(result)

        ranked_buy = self._add_rank(results, "buy_score")
        ranked_risk = self._add_rank(results, "risk_score")

        top_buy_candidates = ranked_buy[:top_n]
        highest_risk_candidates = ranked_risk[:top_n]

        return {
            "success": True,
            "agent_goal": (
                "Screen an S&P-style stock universe and rank buy candidates "
                "and higher-risk caution candidates."
            ),
            "universe_size": len(clean_symbols),
            "scanned_count": len(results),
            "failed_count": len(failed),
            "failed_symbols": failed,
            "top_buy_candidates": top_buy_candidates,
            "highest_risk_candidates": highest_risk_candidates,
            "top_sell_risk": highest_risk_candidates,
            "full_results": results,
            "summary": (
                f"Screener Agent scanned {len(results)}/{len(clean_symbols)} stocks. "
                f"Generated Top {top_n} Buy Candidates and Top {top_n} "
                f"Highest Risk / Caution Candidates."
            )
        }