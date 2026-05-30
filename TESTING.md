# Project tests

This project uses `pytest` for lightweight automated checks. The tests are designed for a coursework/demo environment, so they do not call real broker APIs, do not place trades, and do not require live market data.

## What the tests cover

- Import compatibility for the refactored agent packages.
- Basic agent construction with temporary SQLite/model paths.
- App helper functions, ticker cleaning, label formatting, price formatting, and nested dictionary access.
- OHLCV dataframe cleaning, including duplicate `close` columns and yfinance-style MultiIndex columns.
- Portfolio and event context builders.
- Validation Agent behaviour on consistent multi-source quotes.
- Risk Agent DQN warm-up behaviour and safe handling of missing nested analysis sections.
- Strategist Agent human-review output.
- SQLite storage for historical prices and UI session logs.
- Training Agent feature dataset creation.
- LLM Report Agent local fallback when Groq is not configured.
- Screener Agent scoring with mocked historical data.
- Screener report fallback when no LLM agent is available.

## Run locally

```bash
pip install -r requirements.txt
pip install pytest
pytest
```

The tests include small stubs for optional packages such as Streamlit, yfinance, and Groq when those packages are not installed in the current environment. In a normal project environment, the real packages from `requirements.txt` will be used.

## Notes

The tests are smoke and component tests. They check that the project structure, agents, SQLite storage, DQN risk layer, and UI helper logic are wired correctly. They are not intended to verify financial performance or real trading decisions.
