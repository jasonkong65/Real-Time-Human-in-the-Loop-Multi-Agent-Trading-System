# Testing

This project includes a lightweight pytest suite for the Streamlit multi-agent stock research system.

The tests are designed to run without real market/API calls. The pytest setup stubs optional external packages when needed and removes local API/database environment variables during tests, so results should be the same on a local machine and in GitHub Actions.

## Run tests locally

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

For a basic syntax check, run:

```bash
python -m compileall -q .
```

## What the tests cover

The current test suite checks:

- compatibility imports from the old agent entry files and the new package-style agent folders
- basic agent initialization using temporary paths
- Streamlit helper functions and chart dataframe cleaning
- portfolio and event context builders
- Validation Agent data quality logic
- Risk Agent initialization and DQN warm-up path
- Strategist Agent output structure
- SQLite historical price storage and reading
- Execution Agent UI-session recording
- Training Agent dataset construction and signal generation helpers
- LLM Report Agent fallback behavior when Groq is not available
- Screener Agent basic scoring path
- workflow helper behavior

## Notes

The tests do not try to prove the financial model is profitable. They are smoke and integration checks for project reliability, import safety, SQLite isolation, and basic agent behavior.
