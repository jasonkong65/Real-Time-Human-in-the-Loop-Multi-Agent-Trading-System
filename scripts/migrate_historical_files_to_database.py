from __future__ import annotations

import argparse
from pathlib import Path
import re

import pandas as pd

# Run from project root:
# python scripts/migrate_historical_files_to_database.py --data-dir data/historical

try:
    from agents.storage_agent import StorageAgent
except Exception:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from agents.storage_agent import StorageAgent


def parse_name(path: Path):
    stem = path.stem
    # Supported new format: AAPL_1y_1d.csv / AAPL_30d_1d.parquet
    m = re.match(r"^([A-Za-z.\-]+)_([^_]+)_([^_]+)$", stem)
    if m:
        return m.group(1).upper(), m.group(2), m.group(3)
    # Legacy format: AAPL.csv
    return stem.upper(), "1y", "1d"


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    if "date" not in [str(c).lower() for c in df.columns]:
        df = df.reset_index()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df = df.rename(columns={"datetime": "date", "index": "date", "adjclose": "adj_close"})
    required = ["date", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    for col in ["open", "high", "low", "close", "adj_close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=required)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/historical")
    parser.add_argument("--database-url", default=None)
    args = parser.parse_args()

    storage = StorageAgent(database_url=args.database_url) if args.database_url else StorageAgent()
    data_dir = Path(args.data_dir)
    files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.parquet"))

    migrated = 0
    failed = []
    for path in files:
        try:
            symbol, period, interval = parse_name(path)
            if path.suffix.lower() == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            df = clean_df(df)
            result = storage.record_historical_prices(
                symbol=symbol,
                prices=df,
                period=period,
                interval=interval,
                source=f"migrated_{path.suffix.lower().replace('.', '')}",
                metadata={"original_file": str(path)},
            )
            migrated += int(result.get("rows_written", 0))
            print(f"OK {path.name}: {result.get('rows_written', 0)} rows")
        except Exception as exc:
            failed.append((str(path), str(exc)))
            print(f"FAIL {path.name}: {exc}")

    print("\nMigration complete")
    print(f"Rows migrated: {migrated}")
    print(f"Failed files: {len(failed)}")
    if failed:
        for item in failed:
            print(item)


if __name__ == "__main__":
    main()
