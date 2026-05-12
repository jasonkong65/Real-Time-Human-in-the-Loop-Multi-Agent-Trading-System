import pandas as pd


def compute_rsi(close_series: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index.
    """
    delta = close_series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def build_trading_features(price_records: list) -> pd.DataFrame:
    """
    Build trading features from historical OHLCV records.

    Expected input:
    [
        {"date": "...", "open": ..., "high": ..., "low": ..., "close": ..., "volume": ...}
    ]
    """
    df = pd.DataFrame(price_records)

    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["return_1"] = df["close"].pct_change(1)
    df["return_5"] = df["close"].pct_change(5)
    df["return_20"] = df["close"].pct_change(20)

    df["ma_5"] = df["close"].rolling(window=5).mean()
    df["ma_20"] = df["close"].rolling(window=20).mean()
    df["ma_gap"] = (df["ma_5"] - df["ma_20"]) / df["ma_20"]

    df["volatility_20"] = df["return_1"].rolling(window=20).std()

    df["volume_ma_20"] = df["volume"].rolling(window=20).mean()
    df["volume_change"] = (df["volume"] - df["volume_ma_20"]) / df["volume_ma_20"]

    df["rsi_14"] = compute_rsi(df["close"], window=14)

    df = df.dropna().reset_index(drop=True)

    return df