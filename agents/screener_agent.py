from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from agents.historical_data_agent import HistoricalDataAgent

try:
    import yfinance as yf
except Exception:  # yfinance is optional for market-cap/sector enrichment
    yf = None


class ScreenerAgent:
    """
    Watchlist-based Screener Agent.

    Role:
    - Rank a user-provided stock universe into research candidates and caution candidates.
    - Apply sector diversification so one hot sector does not dominate the list.
    - Apply a simple liquidity / market-cap filter where data is available.
    - Prepare a handoff object for the single-stock pipeline.
    - Save screener runs to SQLite / StorageAgent when available.

    Important safety design:
    - This is not a full-market scanner.
    - It does not provide direct investment advice.
    - It only produces candidates for further research and paper decision support.
    """

    DEFAULT_SECTOR_MAP = {
        # Technology
        "AAPL": "Technology",
        "MSFT": "Technology",
        "NVDA": "Technology",
        "AMD": "Technology",
        "AVGO": "Technology",
        "INTC": "Technology",
        "QCOM": "Technology",
        "TXN": "Technology",
        "AMAT": "Technology",
        "ADBE": "Technology",
        "ORCL": "Technology",
        "CSCO": "Technology",
        # Communication Services
        "GOOGL": "Communication Services",
        "GOOG": "Communication Services",
        "META": "Communication Services",
        "NFLX": "Communication Services",
        "DIS": "Communication Services",
        # Consumer
        "AMZN": "Consumer Discretionary",
        "TSLA": "Consumer Discretionary",
        "HD": "Consumer Discretionary",
        "MCD": "Consumer Discretionary",
        "COST": "Consumer Staples",
        "WMT": "Consumer Staples",
        "PEP": "Consumer Staples",
        "KO": "Consumer Staples",
        # Financials
        "JPM": "Financials",
        "BAC": "Financials",
        "V": "Financials",
        "MA": "Financials",
        # Healthcare
        "UNH": "Healthcare",
        "JNJ": "Healthcare",
        "PFE": "Healthcare",
    }

    DEFAULT_CONFIG = {
        "min_history_rows": 60,
        "min_avg_dollar_volume": 10_000_000,
        "min_market_cap": 0,
        "max_per_sector": 2,
        "candidate_queue_size": 3,
        "weights": {
            "momentum": 0.30,
            "trend": 0.25,
            "volatility": 0.18,
            "rsi": 0.15,
            "liquidity": 0.07,
            "volume": 0.05,
        },
        "risk_weights": {
            "negative_momentum": 0.32,
            "downtrend": 0.25,
            "volatility": 0.23,
            "rsi_extreme": 0.13,
            "liquidity": 0.07,
        },
    }

    def __init__(
        self,
        config_path: str = "config/screener_config.json",
        db_path: str = "data/trading_system.db",
        use_yfinance_metadata: bool = True,
    ):
        self.history_agent = HistoricalDataAgent()
        self.config_path = Path(config_path)
        self.db_path = Path(db_path)
        self.use_yfinance_metadata = use_yfinance_metadata
        self.config = self._load_config()
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}

    # --------------------------------------------------
    # Config and safe helpers
    # --------------------------------------------------
    def _load_config(self) -> Dict[str, Any]:
        config = json.loads(json.dumps(self.DEFAULT_CONFIG))

        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    user_config = json.load(f)

                for key, value in user_config.items():
                    if isinstance(value, dict) and isinstance(config.get(key), dict):
                        config[key].update(value)
                    else:
                        config[key] = value
            except Exception:
                pass

        return config

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        try:
            if value is None or value == "":
                return default
            result = float(value)
            if pd.isna(result):
                return default
            return result
        except Exception:
            return default

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            if value is None or value == "":
                return default
            result = int(float(value))
            return result
        except Exception:
            return default

    def _clip(self, value: float, low: float = 0.0, high: float = 1.0) -> float:
        return max(low, min(high, float(value)))

    def _now_utc(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # --------------------------------------------------
    # Technical indicators
    # --------------------------------------------------
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

    def _score_liquidity(self, avg_dollar_volume: float) -> float:
        if avg_dollar_volume <= 0:
            return 0.45
        if avg_dollar_volume >= 100_000_000:
            return 1.00
        if avg_dollar_volume >= 50_000_000:
            return 0.85
        if avg_dollar_volume >= 20_000_000:
            return 0.70
        if avg_dollar_volume >= 10_000_000:
            return 0.55
        return 0.25

    def _score_liquidity_risk(self, avg_dollar_volume: float) -> float:
        return 1.0 - self._score_liquidity(avg_dollar_volume)

    # --------------------------------------------------
    # Metadata / filters
    # --------------------------------------------------
    def _sector_for_symbol(self, symbol: str) -> str:
        symbol = str(symbol).upper().strip()

        if symbol in self.DEFAULT_SECTOR_MAP:
            return self.DEFAULT_SECTOR_MAP[symbol]

        metadata = self._get_yfinance_metadata(symbol)
        sector = metadata.get("sector")

        if sector:
            return str(sector)

        return "Unknown"

    def _get_yfinance_metadata(self, symbol: str) -> Dict[str, Any]:
        symbol = str(symbol).upper().strip()

        if symbol in self._metadata_cache:
            return self._metadata_cache[symbol]

        result = {
            "symbol": symbol,
            "sector": self.DEFAULT_SECTOR_MAP.get(symbol, "Unknown"),
            "market_cap": None,
            "average_volume": None,
            "source": "default_map",
            "error": None,
        }

        if not self.use_yfinance_metadata or yf is None:
            self._metadata_cache[symbol] = result
            return result

        try:
            ticker = yf.Ticker(symbol)

            # fast_info is cheaper than full info where available.
            fast_info = getattr(ticker, "fast_info", None)
            if fast_info:
                try:
                    result["market_cap"] = self._safe_int(getattr(fast_info, "market_cap", None), None)
                except Exception:
                    pass

            # info may be slower but gives sector.
            info = {}
            try:
                info = ticker.info or {}
            except Exception:
                info = {}

            if info.get("sector"):
                result["sector"] = info.get("sector")
            if info.get("marketCap"):
                result["market_cap"] = self._safe_int(info.get("marketCap"), None)
            if info.get("averageVolume"):
                result["average_volume"] = self._safe_int(info.get("averageVolume"), None)

            result["source"] = "yfinance"
        except Exception as e:
            result["error"] = str(e)

        self._metadata_cache[symbol] = result
        return result

    def _passes_filters(
        self,
        avg_dollar_volume_20: float,
        market_cap: Optional[int],
        min_avg_dollar_volume: float,
        min_market_cap: float,
    ) -> Tuple[bool, List[str]]:
        warnings = []
        passed = True

        if avg_dollar_volume_20 < min_avg_dollar_volume:
            passed = False
            warnings.append(
                f"Average dollar volume is below filter: {avg_dollar_volume_20:,.0f} < {min_avg_dollar_volume:,.0f}."
            )

        if min_market_cap > 0 and market_cap is not None and market_cap < min_market_cap:
            passed = False
            warnings.append(
                f"Market cap is below filter: {market_cap:,.0f} < {min_market_cap:,.0f}."
            )

        if min_market_cap > 0 and market_cap is None:
            warnings.append("Market cap was not available; market-cap filter could not be fully verified.")

        return passed, warnings

    # --------------------------------------------------
    # Per-symbol scoring
    # --------------------------------------------------
    def _first_series(self, df: pd.DataFrame, column: str) -> pd.Series:
        """Return one column as a Series even when duplicate labels exist.

        Historical data from yfinance/database caches can contain both close and
        adj_close, or MultiIndex columns. After aliasing, duplicate column names
        make df[column] return a DataFrame. pd.to_numeric then crashes with:
        "arg must be a list, tuple, 1-d array, or Series".
        """
        if df is None or df.empty or column not in df.columns:
            return pd.Series(dtype="float64")
        data = df.loc[:, column]
        if isinstance(data, pd.DataFrame):
            if data.shape[1] == 0:
                return pd.Series(dtype="float64")
            data = data.iloc[:, 0]
        return data if isinstance(data, pd.Series) else pd.Series(data)

    def _normalise_history_result(self, hist: Dict[str, Any]) -> pd.DataFrame:
        prices = hist.get("prices") or hist.get("price_records") or hist.get("records")

        if prices is None:
            return pd.DataFrame()

        if isinstance(prices, pd.DataFrame):
            df = prices.copy()
        else:
            try:
                if len(prices) == 0:
                    return pd.DataFrame()
            except Exception:
                return pd.DataFrame()
            df = pd.DataFrame(prices)

        if df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

        # Some historical agents use adj_close, close_price, or datetime aliases.
        # Do NOT rename adj_close to close when close already exists; that creates
        # duplicate close columns and crashes pd.to_numeric.
        rename_map = {
            "close_price": "close",
            "adjclose": "adj_close",
            "date_time": "date",
            "datetime": "date",
            "timestamp": "date",
            "price_timestamp": "date",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()].copy()

        if "close" not in df.columns and "adj_close" in df.columns:
            df["close"] = self._first_series(df, "adj_close")

        required = ["close", "volume"]
        for col in required:
            if col not in df.columns:
                df[col] = None
            df[col] = pd.to_numeric(self._first_series(df, col), errors="coerce")

        if "date" in df.columns:
            df["date"] = pd.to_datetime(self._first_series(df, "date"), errors="coerce")
            df = df.sort_values("date")

        df = df.dropna(subset=["close", "volume"])
        return df

    def _score_one(
        self,
        symbol: str,
        period: str,
        interval: str = "1d",
        min_avg_dollar_volume: Optional[float] = None,
        min_market_cap: Optional[float] = None,
    ) -> Dict[str, Any]:
        symbol = str(symbol).upper().strip()
        min_history_rows = int(self.config.get("min_history_rows", 60))
        min_avg_dollar_volume = self._safe_float(
            min_avg_dollar_volume,
            self._safe_float(self.config.get("min_avg_dollar_volume"), 10_000_000),
        )
        min_market_cap = self._safe_float(
            min_market_cap,
            self._safe_float(self.config.get("min_market_cap"), 0),
        )

        try:
            hist = self.history_agent.get_or_download_data(symbol, period=period, interval=interval)
        except TypeError:
            hist = self.history_agent.get_or_download_data(symbol, period=period)

        if not hist.get("success"):
            return {
                "success": False,
                "symbol": symbol,
                "error": hist.get("error", "Historical data request failed."),
            }

        df = self._normalise_history_result(hist)

        if len(df) < min_history_rows:
            return {
                "success": False,
                "symbol": symbol,
                "error": f"Not enough history for screening. Need at least {min_history_rows} rows; got {len(df)}.",
            }

        close = df["close"]
        volume = df["volume"]

        ret5 = self._safe_float(close.pct_change(5).iloc[-1])
        ret20 = self._safe_float(close.pct_change(20).iloc[-1])
        ma20 = self._safe_float(close.rolling(20).mean().iloc[-1])
        ma50 = self._safe_float(close.rolling(50).mean().iloc[-1])
        ma_gap = self._safe_float((ma20 - ma50) / ma50 if ma50 else 0.0)
        vol20 = self._safe_float(close.pct_change().rolling(20).std().iloc[-1])
        rsi = self._safe_float(self._rsi(close).iloc[-1], 50.0)
        volume_ma5 = self._safe_float(volume.rolling(5).mean().iloc[-1])
        volume_ma20 = self._safe_float(volume.rolling(20).mean().iloc[-1])
        volume_change = self._safe_float((volume_ma5 - volume_ma20) / volume_ma20 if volume_ma20 else 0.0)
        avg_dollar_volume_20 = self._safe_float((close * volume).rolling(20).mean().iloc[-1])

        metadata = self._get_yfinance_metadata(symbol)
        sector = metadata.get("sector") or self._sector_for_symbol(symbol)
        market_cap = metadata.get("market_cap")

        filter_passed, filter_warnings = self._passes_filters(
            avg_dollar_volume_20=avg_dollar_volume_20,
            market_cap=market_cap,
            min_avg_dollar_volume=min_avg_dollar_volume,
            min_market_cap=min_market_cap,
        )

        momentum_score = self._clip((ret20 + 0.10) / 0.22)
        trend_score = self._clip((ma_gap + 0.06) / 0.12)
        volatility_score = self._clip(1 - vol20 / 0.07)
        rsi_buy_score = self._score_rsi_buy(rsi)
        volume_score = self._clip((volume_change + 0.20) / 0.70)
        liquidity_score = self._score_liquidity(avg_dollar_volume_20)

        weights = self.config.get("weights", self.DEFAULT_CONFIG["weights"])
        buy_score = self._clip(
            weights.get("momentum", 0.30) * momentum_score
            + weights.get("trend", 0.25) * trend_score
            + weights.get("volatility", 0.18) * volatility_score
            + weights.get("rsi", 0.15) * rsi_buy_score
            + weights.get("liquidity", 0.07) * liquidity_score
            + weights.get("volume", 0.05) * volume_score
        )

        negative_momentum = self._clip((-ret20 + 0.02) / 0.12)
        downtrend = self._clip((-ma_gap + 0.02) / 0.08)
        volatility_risk = self._clip(vol20 / 0.07)
        rsi_extreme = 1.0 if rsi >= 78 or rsi <= 30 else (0.65 if rsi >= 72 else 0.25)
        liquidity_risk = self._score_liquidity_risk(avg_dollar_volume_20)

        risk_weights = self.config.get("risk_weights", self.DEFAULT_CONFIG["risk_weights"])
        risk_score = self._clip(
            risk_weights.get("negative_momentum", 0.32) * negative_momentum
            + risk_weights.get("downtrend", 0.25) * downtrend
            + risk_weights.get("volatility", 0.23) * volatility_risk
            + risk_weights.get("rsi_extreme", 0.13) * rsi_extreme
            + risk_weights.get("liquidity", 0.07) * liquidity_risk
        )

        if not filter_passed:
            buy_score = min(buy_score, 0.58)
            risk_score = max(risk_score, 0.55)

        reasons = self._build_reasons(
            ret20=ret20,
            ma_gap=ma_gap,
            rsi=rsi,
            vol20=vol20,
            avg_dollar_volume_20=avg_dollar_volume_20,
            filter_warnings=filter_warnings,
        )

        contributions = self._indicator_contributions(
            momentum_score=momentum_score,
            trend_score=trend_score,
            volatility_score=volatility_score,
            rsi_buy_score=rsi_buy_score,
            volume_score=volume_score,
            liquidity_score=liquidity_score,
            weights=weights,
        )

        signal = self._signal_from_scores(
            buy_score=buy_score,
            risk_score=risk_score,
            rsi=rsi,
            filter_passed=filter_passed,
        )

        return {
            "success": True,
            "symbol": symbol,
            "sector": sector,
            "buy_score": round(buy_score, 3),
            "risk_score": round(risk_score, 3),
            "screen_signal": signal,
            "passes_liquidity_filter": bool(filter_passed),
            "filter_warnings": filter_warnings,
            "return_5": round(ret5, 4),
            "return_20": round(ret20, 4),
            "ma_gap": round(ma_gap, 4),
            "volatility_20": round(vol20, 4),
            "rsi_14": round(rsi, 2),
            "volume_change": round(volume_change, 4),
            "avg_dollar_volume_20": round(avg_dollar_volume_20, 2),
            "market_cap": market_cap,
            "metadata_source": metadata.get("source"),
            "reason": "; ".join(reasons),
            "indicator_contributions": contributions,
            "handoff_hint": self._handoff_hint(signal, risk_score, rsi),
        }

    def _signal_from_scores(self, buy_score: float, risk_score: float, rsi: float, filter_passed: bool) -> str:
        if not filter_passed:
            return "WATCHLIST_LIQUIDITY_CAUTION"
        if buy_score >= 0.72 and risk_score < 0.35 and rsi < 75:
            return "BUY_CANDIDATE"
        if buy_score >= 0.68 and rsi >= 72:
            return "BUY_WATCHLIST_OVERBOUGHT"
        if buy_score >= 0.62:
            return "WATCHLIST_BUY_MONITOR"
        if risk_score >= 0.65:
            return "SELL_RISK"
        return "WATCHLIST_HOLD"

    def _build_reasons(
        self,
        ret20: float,
        ma_gap: float,
        rsi: float,
        vol20: float,
        avg_dollar_volume_20: float,
        filter_warnings: List[str],
    ) -> List[str]:
        reasons = []
        if ret20 > 0.03:
            reasons.append("positive 20-day momentum")
        if ma_gap > 0:
            reasons.append("short-term average above medium-term average")
        if rsi >= 72:
            reasons.append("RSI is high; entry timing risk may be elevated")
        if vol20 > 0.04:
            reasons.append("volatility is high")
        if avg_dollar_volume_20 < self.config.get("min_avg_dollar_volume", 10_000_000):
            reasons.append("liquidity is limited")
        if ret20 < -0.03:
            reasons.append("negative 20-day momentum")
        for warning in filter_warnings:
            reasons.append(warning)
        if not reasons:
            reasons.append("mixed technical signals")
        return reasons

    def _indicator_contributions(
        self,
        momentum_score: float,
        trend_score: float,
        volatility_score: float,
        rsi_buy_score: float,
        volume_score: float,
        liquidity_score: float,
        weights: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        rows = [
            ("Momentum", momentum_score, weights.get("momentum", 0.30)),
            ("Trend", trend_score, weights.get("trend", 0.25)),
            ("Volatility", volatility_score, weights.get("volatility", 0.18)),
            ("RSI", rsi_buy_score, weights.get("rsi", 0.15)),
            ("Liquidity", liquidity_score, weights.get("liquidity", 0.07)),
            ("Volume", volume_score, weights.get("volume", 0.05)),
        ]
        result = []
        for name, score, weight in rows:
            result.append({
                "indicator": name,
                "score": round(score, 3),
                "weight": round(weight, 3),
                "weighted_contribution": round(score * weight, 3),
            })
        return sorted(result, key=lambda r: r["weighted_contribution"], reverse=True)

    def _handoff_hint(self, signal: str, risk_score: float, rsi: float) -> str:
        if signal == "BUY_CANDIDATE":
            return "Send to single-stock pipeline for full validation, risk control, and strategy review."
        if signal == "BUY_WATCHLIST_OVERBOUGHT":
            return "Send to single-stock pipeline, but expect entry-timing risk review."
        if signal == "WATCHLIST_BUY_MONITOR":
            return "Optional handoff for further review."
        if signal == "WATCHLIST_LIQUIDITY_CAUTION":
            return "Do not prioritise unless liquidity filters are relaxed."
        if risk_score >= 0.65:
            return "Review as a caution candidate, not an entry candidate."
        return "Monitor only."

    # --------------------------------------------------
    # Ranking, diversification, and handoff
    # --------------------------------------------------
    def _rank(self, rows: List[Dict[str, Any]], key: str, reverse: bool = True) -> List[Dict[str, Any]]:
        ranked = sorted(rows, key=lambda r: r.get(key, 0), reverse=reverse)
        for i, row in enumerate(ranked, start=1):
            row["rank"] = i
        return ranked

    def _diversified_top(
        self,
        ranked_rows: List[Dict[str, Any]],
        top_n: int,
        max_per_sector: int,
    ) -> List[Dict[str, Any]]:
        selected = []
        sector_counts: Dict[str, int] = {}

        for row in ranked_rows:
            sector = row.get("sector", "Unknown")
            if sector_counts.get(sector, 0) < max_per_sector:
                selected.append(row)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
            if len(selected) >= top_n:
                break

        if len(selected) < top_n:
            selected_symbols = {row.get("symbol") for row in selected}
            for row in ranked_rows:
                if row.get("symbol") not in selected_symbols:
                    selected.append(row)
                    selected_symbols.add(row.get("symbol"))
                if len(selected) >= top_n:
                    break

        for i, row in enumerate(selected, start=1):
            row["diversified_rank"] = i

        return selected

    def _build_pipeline_handoff(
        self,
        top_buy_candidates: List[Dict[str, Any]],
        auto_handoff: bool,
        candidate_queue_size: int,
    ) -> Dict[str, Any]:
        eligible_signals = {"BUY_CANDIDATE", "BUY_WATCHLIST_OVERBOUGHT", "WATCHLIST_BUY_MONITOR"}
        queue = [
            {
                "symbol": row.get("symbol"),
                "screen_signal": row.get("screen_signal"),
                "buy_score": row.get("buy_score"),
                "risk_score": row.get("risk_score"),
                "sector": row.get("sector"),
                "handoff_hint": row.get("handoff_hint"),
            }
            for row in top_buy_candidates
            if row.get("screen_signal") in eligible_signals
        ][:candidate_queue_size]

        selected = queue[0] if queue else None

        return {
            "enabled": bool(auto_handoff),
            "should_run_single_stock_pipeline": bool(auto_handoff and selected),
            "selected_symbol": selected.get("symbol") if selected else None,
            "selected_candidate": selected,
            "candidate_queue": queue,
            "instruction": (
                "Use selected_symbol as the next symbol for the single-stock pipeline. "
                "This handoff is for research review only, not a trade instruction."
            ),
        }

    # --------------------------------------------------
    # Storage
    # --------------------------------------------------
    def _save_screener_run_to_storage(self, result: Dict[str, Any]) -> Dict[str, Any]:
        # Try project StorageAgent first.
        try:
            from agents.storage_agent import StorageAgent

            storage = StorageAgent()
            try:
                saved = storage.record_screener_run(
                    universe_size=result.get("universe_size"),
                    top_n=result.get("top_n"),
                    period=result.get("period"),
                    result_json=result,
                )
                return {"success": True, "method": "StorageAgent.record_screener_run", "result": saved}
            except TypeError:
                saved = storage.record_screener_run(result)
                return {"success": True, "method": "StorageAgent.record_screener_run", "result": saved}
        except Exception as e:
            storage_error = str(e)

        # Fallback: write a minimal SQLite record directly.
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS screener_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT,
                    universe_size INTEGER,
                    top_n INTEGER,
                    period TEXT,
                    result_json TEXT,
                    created_at TEXT
                )
                """
            )
            cur.execute(
                """
                INSERT INTO screener_runs (
                    run_id, universe_size, top_n, period, result_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    result.get("run_id"),
                    result.get("universe_size"),
                    result.get("top_n"),
                    result.get("period"),
                    json.dumps(result, ensure_ascii=False, default=str),
                    self._now_utc(),
                ),
            )
            conn.commit()
            conn.close()
            return {"success": True, "method": "direct_sqlite", "db_path": str(self.db_path)}
        except Exception as e:
            return {"success": False, "method": "none", "error": str(e), "storage_agent_error": storage_error}

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------
    def screen_universe(
        self,
        symbols: List[str],
        top_n: int = 10,
        period: str = "1y",
        interval: str = "1d",
        diversify_by_sector: bool = True,
        max_per_sector: Optional[int] = None,
        min_avg_dollar_volume: Optional[float] = None,
        min_market_cap: Optional[float] = None,
        auto_handoff: bool = True,
        save_to_storage: bool = True,
        pipeline_callback: Optional[Callable[[str], Any]] = None,
    ) -> Dict[str, Any]:
        clean_symbols = []
        for symbol in symbols or []:
            s = str(symbol).upper().strip()
            if s and s not in clean_symbols:
                clean_symbols.append(s)

        max_per_sector = int(max_per_sector or self.config.get("max_per_sector", 2))
        candidate_queue_size = int(self.config.get("candidate_queue_size", 3))
        run_id = f"SCR-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

        results = []
        failed = []
        filtered_out = []

        for symbol in clean_symbols:
            row = self._score_one(
                symbol=symbol,
                period=period,
                interval=interval,
                min_avg_dollar_volume=min_avg_dollar_volume,
                min_market_cap=min_market_cap,
            )
            if not row.get("success"):
                failed.append(row)
            else:
                results.append(row)
                if not row.get("passes_liquidity_filter", True):
                    filtered_out.append(row)

        ranked_by_buy = self._rank(results, "buy_score", True)
        ranked_by_risk = self._rank(results, "risk_score", True)

        if diversify_by_sector:
            top_buy = self._diversified_top(ranked_by_buy, top_n, max_per_sector)
        else:
            top_buy = ranked_by_buy[:top_n]

        highest_risk = ranked_by_risk[:top_n]

        sector_summary = self._sector_summary(results)
        pipeline_handoff = self._build_pipeline_handoff(
            top_buy_candidates=top_buy,
            auto_handoff=auto_handoff,
            candidate_queue_size=candidate_queue_size,
        )

        callback_result = None
        if pipeline_callback and pipeline_handoff.get("should_run_single_stock_pipeline"):
            try:
                callback_result = pipeline_callback(pipeline_handoff["selected_symbol"])
            except Exception as e:
                callback_result = {"success": False, "error": str(e)}

        result = {
            "success": True,
            "agent": "Screener Agent",
            "agent_goal": "Rank a watchlist into research candidates and caution candidates.",
            "run_id": run_id,
            "created_at_utc": self._now_utc(),
            "period": period,
            "interval": interval,
            "top_n": top_n,
            "universe_size": len(clean_symbols),
            "scanned_count": len(results),
            "failed_count": len(failed),
            "filtered_out_count": len(filtered_out),
            "failed_symbols": failed,
            "filtered_out": filtered_out,
            "sector_diversification_enabled": diversify_by_sector,
            "max_per_sector": max_per_sector,
            "sector_summary": sector_summary,
            "top_buy_candidates": top_buy,
            "highest_risk_candidates": highest_risk,
            "all_results": results,
            "single_stock_pipeline_handoff": pipeline_handoff,
            "pipeline_callback_result": callback_result,
            "summary": (
                f"Screener scanned {len(results)}/{len(clean_symbols)} stocks, "
                f"built a diversified candidate list, and prepared a single-stock review handoff."
            ),
        }

        if save_to_storage:
            result["storage_result"] = self._save_screener_run_to_storage(result)
        else:
            result["storage_result"] = {"success": False, "reason": "save_to_storage=False"}

        return result

    def _sector_summary(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not rows:
            return []
        df = pd.DataFrame(rows)
        if "sector" not in df.columns:
            return []
        grouped = (
            df.groupby("sector")
            .agg(
                count=("symbol", "count"),
                avg_buy_score=("buy_score", "mean"),
                avg_risk_score=("risk_score", "mean"),
            )
            .reset_index()
            .sort_values("avg_buy_score", ascending=False)
        )
        return [
            {
                "sector": row["sector"],
                "count": int(row["count"]),
                "avg_buy_score": round(float(row["avg_buy_score"]), 3),
                "avg_risk_score": round(float(row["avg_risk_score"]), 3),
            }
            for _, row in grouped.iterrows()
        ]

    def run(self, symbols: List[str], top_n: int = 10, period: str = "1y", **kwargs) -> Dict[str, Any]:
        return self.screen_universe(symbols, top_n=top_n, period=period, **kwargs)
