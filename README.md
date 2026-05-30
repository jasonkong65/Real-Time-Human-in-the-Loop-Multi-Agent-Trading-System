# Human-in-the-Loop Multi-Agent Stock Research System

This project is a Streamlit stock-research prototype built for paper decision support. It does not connect to a broker and it does not place real trades. The goal is to show how several specialised software agents can work together to collect market data, check data quality, analyse a stock, control risk, explain the result, and save the full research session for later evaluation.

The project is designed as a coursework/demo system. It is useful for showing agent coordination, human-in-the-loop decision making, memory, delayed feedback, and risk-aware reporting. It should not be treated as financial advice.

---

## Project purpose

The system answers questions such as:

- What does the current data say about a stock such as `AAPL`?
- Is the stock only worth monitoring, or is it a possible paper-research candidate?
- Are there any data-quality or timing risks?
- Which stocks in a small watchlist look stronger or riskier?
- How can the result be explained to a normal user in plain English?

The main design idea is that no single model makes the whole decision. Each agent has a small responsibility:

1. collect data;
2. validate the data;
3. analyse technical conditions;
4. train or load a signal model;
5. apply risk control;
6. create a cautious paper strategy;
7. explain the result;
8. store the session and evaluate delayed outcomes later.

The human user stays in control. The app gives research guidance, not trade execution.

---

## Installation

### 1. Clone or unzip the project

```bash
git clone <your-repository-url>
cd Real-Time-Human-in-the-Loop-Multi-Agent-Trading-System
```

If you are using the submitted zip file, unzip it first and then open the project folder.

### 2. Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The main dependencies are:

- `streamlit` for the web UI;
- `pandas` for data handling;
- `yfinance` for historical price downloads;
- `requests` for external API calls;
- `scikit-learn` and `joblib` for the signal model;
- `torch` for the DQN risk layer;
- `groq` for optional LLM explanations;
- `python-dotenv` for local API keys.

### 4. Create the environment file

Copy the example file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
copy .env.example .env
```

Then edit `.env` and add the keys you want to use:

```text
FINNHUB_API_KEY=your_finnhub_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
GROQ_API_KEY=your_groq_key
```

The app can still open without every key, but some live-data or LLM features may use fallback behaviour.

### 5. Repair or initialise SQLite if needed

If you have an older local database from a previous run, repair the schema before starting the app:

```bash
python scripts/repair_sqlite_schema.py
```

This is safe to run multiple times. It only adds missing compatibility columns and keeps existing records.

### 6. Start the Streamlit app

```bash
streamlit run app.py
```

If the UI behaves strangely after code changes, clear Streamlit cache first:

```bash
streamlit cache clear
streamlit run app.py
```

---

## Workflow

The normal single-stock workflow is:

```text
User input
  |
  |-- stock symbol
  |-- selected modules
  |-- optional holding / event context
  v
Data Agent
  |
  |-- fetches live quote data
  |-- uses cache to reduce API pressure
  v
Validation Agent
  |
  |-- checks price consistency, stale data, and source reliability
  v
Historical Data Agent
  |
  |-- loads or downloads OHLCV history
  |-- stores historical data through SQLite-first storage
  v
Analyst Agent
  |
  |-- quote-level analysis
  |-- historical trend, momentum, volatility, RSI, and volume analysis
  |-- combined analyst score and entry-risk label
  v
Training Agent
  |
  |-- trains or loads a signal model
  |-- produces a model signal such as HOLD, BUY_CANDIDATE, or SELL_RISK
  v
Risk Agent
  |
  |-- applies hard safety rules
  |-- uses DQN as an advisory risk layer
  |-- keeps the final output cautious
  v
Strategist Agent
  |
  |-- creates a paper decision plan
  |-- explains whether to monitor, wait, reduce risk, or research further
  v
LLM Report Agent
  |
  |-- writes a plain-language report for the user
  v
Storage / Execution / Reward / Evaluator
  |
  |-- records the run
  |-- saves agent outputs
  |-- tracks delayed paper rewards
  |-- evaluates results when enough future data is available
```

There are also two optional branches:

```text
News / Report Agent
  -> summarises company news, financial context, or pasted report text.

Watchlist Screener Agent
  -> ranks a small user-provided watchlist into research candidates and caution candidates.
```

---

## How to use

### Single-stock analysis

1. Open the app with `streamlit run app.py`.
2. Enter a ticker such as `AAPL`, `MSFT`, `NVDA`, or `TSLA`.
3. Choose the core modules, usually:

```text
Single-stock agent pipeline
Price chart
```

4. Click **Run selected research**.
5. Read the cards at the top for the quick result:

```text
Symbol
Price
Analyst
Model
Risk
Strategy
```

6. Read the **Groq / Report Agent Output** for the plain-English explanation.
7. Open **Agent Responses** if you want to inspect the structured output from each agent.

### Chart period

The chart period is controlled below the chart. Changing the period refreshes the chart without rerunning the whole agent pipeline.

Supported periods:

```text
1 Day
7 Days
30 Days
6 Months
1 Year
2 Years
```

### Portfolio context

Use portfolio context only when you want the strategy to consider an existing paper position. For example, you can tell the app that you currently hold the stock, with a paper quantity and average cost. The Strategist Agent then gives more relevant guidance such as monitoring exposure or avoiding additional risk.

### News / report mode

Turn on the News / Report Agent only when you want news or report explanation. You can either let the system fetch company-specific context, or paste your own financial/news text.

This input box is for report or news text, not for another ticker symbol.

### Watchlist screener

Turn on the Watchlist Screener Agent when you want to compare a group of stocks. Enter a comma-separated list such as:

```text
AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, AMD, NFLX
```

The screener returns:

- candidates for further research;
- caution candidates;
- buy score;
- risk score;
- sector;
- liquidity filter status.

The screener is a watchlist ranking tool. It is not a full-market scanner.

---

## Main features

### Multi-agent research pipeline

The app separates the workflow into specialised agents. This makes the system easier to explain and easier to debug than a single all-in-one model.

### Plain-language user output

The app does not only show raw JSON. The main result is written as a user-facing explanation. More technical outputs are still available in expandable sections for inspection.

### Watchlist screening

The screener can rank a small universe of user-provided tickers. It is useful for a demo because it shows that the project can handle more than one symbol, while still staying within a safe and understandable scope.

### SQLite-first memory

Most important results are stored in SQLite, including historical prices, pipeline runs, agent outputs, paper decisions, reward updates, DQN replay samples, screener runs, LLM reports, and UI sessions.

### Paper decision evaluation

The system can record paper decisions and evaluate them later when the reward horizon is complete. This provides a feedback loop for the Evaluator Agent and the DQN replay layer.

### Refactored agent structure

The largest agents have been split into category folders so the root `agents/` directory stays readable. Compatibility wrappers are kept, so old imports still work.

```text
agents/
  analyst_agent.py      # compatibility wrapper
  training_agent.py     # compatibility wrapper
  risk_agent.py         # compatibility wrapper
  storage_agent.py      # compatibility wrapper

  analysis/
    agent.py
    config.py
    features.py
    market_context.py
    quote.py
    historical.py
    combine.py

  training/
    agent.py
    helpers.py
    features.py
    selection.py
    drift.py
    workflow.py
    signal.py

  risk/
    agent.py
    dqn.py
    config.py
    extractors.py
    state.py
    replay.py
    dqn_policy.py
    rules.py
    feedback.py

  storage/
    agent.py
    helpers.py
    schema.py
    historical.py
    records.py
    queries.py
```

Existing code can still import agents in the old way:

```python
from agents.risk_agent import RiskAgent
from agents.storage_agent import StorageAgent
from agents.analyst_agent import AnalystAgent
from agents.training_agent import TrainingAgent
```

The implementation now lives in the category folders.

---

## Agents and responsibilities

| Agent | Responsibility |
|---|---|
| `DataAgent` | Fetches live quote data, handles API caching, and tracks market-session status. |
| `ValidationAgent` | Checks source reliability, price differences, stale dates, and whether analysis should continue. |
| `HistoricalDataAgent` | Loads or downloads OHLCV history and stores/reuses it through the database layer. |
| `AnalystAgent` | Scores quote movement, historical trend, momentum, volatility, RSI, volume, and entry timing risk. |
| `TrainingAgent` | Trains or loads the signal model using walk-forward validation and model comparison. |
| `TrainingOptimizerAgent` | Backward-compatible wrapper around the Training Agent's automatic model-selection process. |
| `RiskAgent` | Applies hard risk rules and uses a PyTorch DQN layer as an advisory risk-control component. |
| `StrategistAgent` | Converts the final risk-controlled signal into a cautious paper strategy. |
| `RewardAgent` | Records paper decisions and creates delayed reward horizons. |
| `EvaluatorAgent` | Reviews completed paper rewards, strategy performance, and DQN replay readiness. |
| `ScreenerAgent` | Ranks a watchlist into research candidates and caution candidates. |
| `LLMReportAgent` | Explains single-stock, screener, news, or report results in plain language. |
| `StorageAgent` | Main persistent memory layer using SQLite by default. |
| `ExecutionAgent` | Records UI sessions and audit snapshots. It is not a real execution agent. |
| `DatabaseBackend` | Small database adapter used by StorageAgent. SQLite is default; PostgreSQL-style URLs are supported for later deployment. |

---

## DQN

The project uses DQN inside the Risk Agent. The DQN is not the only decision maker and it is not allowed to override the hard safety rules.

The risk layer has two parts:

1. **Hard safety rules** for non-negotiable conditions such as weak data confidence, high risk, or unsafe model signals.
2. **DQN advisory layer** using PyTorch to learn from delayed paper reward feedback.

The DQN-related files are organised under:

```text
agents/risk/
  dqn.py
  replay.py
  dqn_policy.py
  state.py
  feedback.py
```

The DQN components include:

- a small neural network for Q-values;
- a policy network and target network;
- replay memory;
- reward feedback from completed paper decisions;
- a minimum replay-sample requirement before treating DQN learning as meaningful.

The final output remains cautious. If the DQN suggests something risky, the hard safety layer can still downgrade or block the action.

---

## SQLite

SQLite is the primary storage layer for the project. The default database file is:

```text
data/trading_system.db
```

The database stores:

- historical prices;
- historical metadata;
- live market quotes;
- pipeline runs;
- agent outputs;
- paper decisions;
- delayed reward updates;
- DQN replay samples;
- training runs;
- screener runs;
- LLM reports;
- UI sessions and session artifacts.

Some CSV files are still kept for seed data, fallback, or backward compatibility. The main project memory, however, is SQLite-first.

If the local database was created by an older version of the project, run:

```bash
python scripts/repair_sqlite_schema.py
```

This repairs missing compatibility columns such as reward horizon fields without deleting old data.

---

## The price may be slightly different from real finance pages

The price shown in the app may be slightly different from Google Finance, Yahoo Finance, or a broker page. This is expected.

Possible reasons include:

- different data providers;
- delayed quotes;
- caching inside the app;
- regular market vs pre-market/post-market handling;
- different refresh timestamps;
- adjusted close vs live quote differences.

For this project, a small difference is acceptable because the app is a paper research prototype, not a real-time trading execution system. The UI also shows chart source and fetch time to make the data timing clearer.

---

## Evaluator shows N/A

`Reward Win Rate`, `Directional Win Rate`, and `Average Reward` can show `N/A` at the start. This usually means the system has recorded paper decisions, but the future reward windows have not completed yet.

For example, if a paper decision has a 7-day or 30-day horizon, the system cannot fairly judge the outcome immediately. It needs later price data first.

To get evaluator metrics:

1. run the single-stock pipeline several times;
2. keep **Record paper decision / memory** enabled;
3. wait until at least one reward horizon is complete;
4. run the Evaluator tab again.

Until then, `N/A` is a correct early-stage result, not a crash.

---

## DQN Ready is False

`DQN Ready: False` is also normal in early runs.

The DQN needs enough replay samples before its training output is useful. A replay sample is created only after a paper decision has a completed reward result. If there are no completed rewards, there is no meaningful DQN replay memory yet.

The project intentionally avoids pretending the DQN is ready before enough feedback exists. This is safer and more honest for a coursework prototype.

---

## Safety statement

This app is for education, coursework, and paper decision support only.

It does not:

- place trades;
- connect to a broker;
- guarantee returns;
- provide financial advice;
- recommend leverage;
- replace the user's own judgement.

All outputs should be read as research notes. A user should always check the original data source, company news, valuation, event risk, and their own risk tolerance before making any real financial decision outside this prototype.
