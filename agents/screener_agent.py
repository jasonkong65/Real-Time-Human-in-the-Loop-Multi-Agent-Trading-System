from typing import Any, Dict, List

import pandas as pd

from agents.historical_data_agent import HistoricalDataAgent


class ScreenerAgent:
    """
    S&P-style Screener Agent

    Scans a user-provided watchlist and ranks stocks by simple technical
    strength and caution risk. It is not a full-market scan.
    """

    def __init__(self):
        self.history_agent = HistoricalDataAgent()

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def _clip(self, value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    def _rsi(self, close: pd.Series, window: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    def _score_rsi_buy(self, rsi: float) -> float:
        if rsi < 30:
            return 0.55
        if 40 <= rsi <= 65:
            return 1.0
        if 65 < rsi <= 72:
            return 0.75
        if rsi > 72:
            return 0.45
        return 0.70

    def _score_one(self, symbol: str, period: str) -> Dict[str, Any]:
        hist = self.history_agent.get_or_download_data(symbol, period=period)
        if not hist.get("success") or not hist.get("prices"):
            return {"success": False, "symbol": symbol, "error": hist.get("error", "No historical data.")}
        df = pd.DataFrame(hist["prices"])
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]
        for col in ["close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        if len(df) < 60:
            return {"success": False, "symbol": symbol, "error": "Not enough history for screening."}
        close = df["close"]
        volume = df["volume"]
        ret5 = self._safe_float(close.pct_change(5).iloc[-1])
        ret20 = self._safe_float(close.pct_change(20).iloc[-1])
        ma20 = close.rolling(20).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma_gap = self._safe_float((ma20 - ma50) / ma50 if ma50 else 0.0)
        vol20 = self._safe_float(close.pct_change().rolling(20).std().iloc[-1])
        rsi = self._safe_float(self._rsi(close).iloc[-1], 50.0)
        volume_change = self._safe_float((volume.rolling(5).mean().iloc[-1] - volume.rolling(20).mean().iloc[-1]) / volume.rolling(20).mean().iloc[-1])

        momentum_score = self._clip((ret20 + 0.10) / 0.22)
        trend_score = self._clip((ma_gap + 0.06) / 0.12)
        volatility_score = self._clip(1 - vol20 / 0.07)
        rsi_buy_score = self._score_rsi_buy(rsi)
        volume_score = self._clip((volume_change + 0.20) / 0.70)
        buy_score = self._clip(0.32 * momentum_score + 0.28 * trend_score + 0.18 * volatility_score + 0.15 * rsi_buy_score + 0.07 * volume_score)

        negative_momentum = self._clip((-ret20 + 0.02) / 0.12)
        downtrend = self._clip((-ma_gap + 0.02) / 0.08)
        volatility_risk = self._clip(vol20 / 0.07)
        overbought_or_oversold = 1.0 if rsi >= 78 or rsi <= 30 else (0.65 if rsi >= 72 else 0.25)
        risk_score = self._clip(0.35 * negative_momentum + 0.25 * downtrend + 0.25 * volatility_risk + 0.15 * overbought_or_oversold)

        reasons = []
        if ret20 > 0.03:
            reasons.append("positive 20-day momentum")
        if ma_gap > 0:
            reasons.append("short-term average above medium-term average")
        if rsi >= 72:
            reasons.append("RSI is high")
        if vol20 > 0.04:
            reasons.append("volatility is high")
        if ret20 < -0.03:
            reasons.append("negative 20-day momentum")
        if not reasons:
            reasons.append("mixed technical signals")

        if buy_score >= 0.72 and risk_score < 0.35 and rsi < 75:
            signal = "BUY_CANDIDATE"
        elif buy_score >= 0.68 and rsi >= 72:
            signal = "BUY_WATCHLIST_OVERBOUGHT"
        elif buy_score >= 0.62:
            signal = "WATCHLIST_BUY_MONITOR"
        elif risk_score >= 0.65:
            signal = "SELL_RISK"
        else:
            signal = "WATCHLIST_HOLD"

        return {
            "success": True,
            "symbol": symbol,
            "buy_score": round(buy_score, 3),
            "risk_score": round(risk_score, 3),
            "screen_signal": signal,
            "return_5": round(ret5, 4),
            "return_20": round(ret20, 4),
            "ma_gap": round(ma_gap, 4),
            "volatility_20": round(vol20, 4),
            "rsi_14": round(rsi, 2),
            "volume_change": round(volume_change, 4),
            "reason": "; ".join(reasons),
        }

    def _rank(self, rows: List[Dict[str, Any]], key: str, reverse: bool = True) -> List[Dict[str, Any]]:
        ranked = sorted(rows, key=lambda r: r.get(key, 0), reverse=reverse)
        for i, row in enumerate(ranked, start=1):
            row["rank"] = i
        return ranked

    def screen_universe(self, symbols: List[str], top_n: int = 10, period: str = "1y") -> Dict[str, Any]:
        clean_symbols = []
        for symbol in symbols or []:
            s = str(symbol).upper().strip()
            if s and s not in clean_symbols:
                clean_symbols.append(s)
        results, failed = [], []
        for symbol in clean_symbols:
            row = self._score_one(symbol, period)
            if row.get("success"):
                results.append(row)
            else:
                failed.append(row)
        top_buy = self._rank(results, "buy_score", True)[:top_n]
        highest_risk = self._rank(results, "risk_score", True)[:top_n]
        return {
            "success": True,
            "agent": "Screener Agent",
            "agent_goal": "Rank a watchlist into research candidates and caution candidates.",
            "universe_size": len(clean_symbols),
            "scanned_count": len(results),
            "failed_count": len(failed),
            "failed_symbols": failed,
            "top_buy_candidates": top_buy,
            "highest_risk_candidates": highest_risk,
            "all_results": results,
            "summary": f"Screener Agent scanned {len(results)}/{len(clean_symbols)} stocks and built watchlist rankings.",
        }

    def run(self, symbols: List[str], top_n: int = 10, period: str = "1y") -> Dict[str, Any]:
        return self.screen_universe(symbols, top_n, period)
