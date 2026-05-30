# Human-in-the-Loop Multi-Agent Stock Research System

A Streamlit-based stock research prototype that shows how several specialised agents can work together to support **paper trading research**.

The system does not connect to a broker and it does not place real trades. Instead, it collects market data, checks data quality, analyses a selected stock, applies risk controls, explains the result in plain language, and stores the session so later paper outcomes can be reviewed.

This project is built mainly for coursework and demonstration. It is useful for showing agent coordination, human-in-the-loop decision making, persistent memory, delayed feedback, model diagnostics, and risk-aware reporting.

> **Important:** This project is not financial advice. All outputs should be treated as research notes for demonstration only.

---

## What this project does

The app helps a user explore questions such as:

- What does the current data suggest about a stock such as `AAPL`, `MSFT`, `NVDA`, or `TSLA`?
- Is the stock only worth monitoring, or does it look like a possible paper-research candidate?
- Are there data-quality issues, stale prices, entry-timing risks, or weak model-confidence signals?
- Which stocks in a small watchlist look stronger or riskier?
- How can the system explain the result in clear, non-technical language?

The main design idea is simple: **no single agent makes the whole decision alone**. Each agent handles one part of the workflow, and the final output is still reviewed by the human user.

---

## Main workflow

The normal single-stock workflow is:

```text
User input
  |
  |-- ticker symbol
  |-- selected modules
  |-- optional holding or event context
  v
Data Agent
  |
  |-- fetches live quote data
  |-- uses caching to reduce API pressure
  v
Validation Agent
  |
  |-- checks source reliability
  |-- checks stale data and price consistency
  v
Historical Data Agent
  |
  |-- loads or downloads OHLCV history
  |-- stores and reuses data through SQLite
  v
Analyst Agent
  |
  |-- analyses trend, momentum, volatility, RSI, and volume
  |-- creates an analyst score and entry-risk label
  v
Training Agent
  |
  |-- trains or loads the signal model
  |-- runs automatic model diagnostics inside the stock pipeline
  |-- updates the saved model only if the quality gate accepts a better candidate
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
  |-- turns the risk-controlled signal into a paper decision plan
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

Optional branches:

```text
News / Report Agent
  -> summarises company news, financial context, or pasted report text.

Watchlist Screener Agent
  -> ranks a small watchlist into research candidates and caution candidates.

Training Agent diagnostics
  -> runs automatically as part of the Training Agent.
     It is not a separate user-facing agent and does not need a manual checkbox.
```

---

## Key features

- Multi-agent stock research pipeline
- Human-in-the-loop paper decision support
- Live quote and historical OHLCV data handling
- Data validation before analysis
- Technical analysis using trend, momentum, volatility, RSI, and volume signals
- Automatic Training Agent diagnostics and model-quality checks
- Risk control with hard safety rules and a DQN advisory layer
- Plain-language LLM report generation when a Groq API key is available
- Watchlist screener for comparing a small group of tickers
- Coloured summary cards for easier interpretation
- SQLite-first memory for prices, runs, agent outputs, training metadata, reports, and paper rewards
- Delayed reward evaluation for paper decisions
- Clear safety framing: education and demonstration only

---

## Installation

### 1. Clone or unzip the project

```bash
git clone https://github.com/jasonkong65/Real-Time-Human-in-the-Loop-Multi-Agent-Trading-System.git
cd Real-Time-Human-in-the-Loop-Multi-Agent-Trading-System
```

If you are using a submitted zip file, unzip it first and then open the project folder.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Main packages used by the project include:

- `streamlit` for the web interface
- `pandas` for data handling
- `yfinance` for historical price data
- `requests` for API calls
- `scikit-learn` and `joblib` for the signal model
- `torch` for the DQN risk layer
- `groq` for optional LLM explanations
- `python-dotenv` for local environment variables

### 4. Create the environment file

Copy the example environment file:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
copy .env.example .env
```

Then add your local keys:

```text
FINNHUB_API_KEY=your_finnhub_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
GROQ_API_KEY=your_groq_key
```

The app can still open without every key. If a key is missing, the related live-data or LLM feature may fall back to a limited mode.

### 5. Repair or initialise SQLite if needed

If you have an older local database from a previous run, run:

```bash
python scripts/repair_sqlite_schema.py
```

This script is safe to run more than once. It adds missing compatibility columns without deleting existing records.

### 6. Start the app

```bash
streamlit run app.py
```

If Streamlit still shows old UI behaviour after code changes, clear its cache first:

```bash
streamlit cache clear
streamlit run app.py
```

---

## How to use the app

### Single-stock analysis

1. Start the app with `streamlit run app.py`.
2. Enter a ticker, for example:

```text
AAPL
MSFT
NVDA
TSLA
```

3. Keep the usual core modules selected:

```text
Single-stock agent pipeline
Price chart
```

4. Click **Run selected research**.
5. Review the coloured summary cards at the top:

```text
Symbol
Price
Analyst
Model
Risk
Strategy
```

The colours are only there to make the result easier to scan:

- green usually means supportive or lower risk;
- amber means watchlist or caution;
- red means higher risk;
- blue or teal usually means data context;
- purple usually means strategy guidance.

6. Read **Groq / Report Agent Output** for the plain-language explanation.
7. Open **Agent Responses** only if you want to inspect the structured output from each agent.

### Chart period

The chart period selector sits below the chart. Changing it refreshes the chart without rerunning the full agent pipeline.

Supported periods:

```text
1 Day
7 Days
30 Days
6 Months
1 Year
2 Years
```

### Portfolio and event context

The portfolio and event fields are optional. Use them when you want the strategy to consider an existing paper position, average cost, earnings date, or other known event risk.

For example, if the user already holds a paper position, the Strategist Agent can respond more carefully by discussing exposure, monitoring, or avoiding additional risk.

### News / report mode

Use the News / Report Agent when you want a company-news or report explanation. You can either let the system fetch company-specific context or paste your own news/report text.

The pasted text box is for news or report content. It is not for entering a second ticker symbol.

### Watchlist screener

Use the Watchlist Screener Agent when you want to compare several stocks at once. Enter a comma-separated list such as:

```text
AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, AMD, NFLX
```

The screener returns:

- research candidates;
- caution candidates;
- buy score;
- risk score;
- sector;
- liquidity filter status.

This is a watchlist-ranking tool, not a full-market scanner and not a buy/sell recommendation.

---

## Training Agent diagnostics

The Training Agent now handles model diagnostics automatically during the single-stock pipeline.

There is no separate sidebar checkbox for this because model maintenance is an internal system task, not something a normal user should need to manage manually.

During this step, the Training Agent can:

- compare model candidates;
- run walk-forward validation;
- record feature-importance information;
- check model-quality metrics;
- decide whether the saved signal model should be updated.

The saved model is updated only when the internal quality gate decides that a new candidate is clearly better. If diagnostics fail, the main stock research result can still be reviewed.

This keeps the app practical for a demo: the model can be maintained automatically, while the user only sees the final research output.

---

## Agents and responsibilities

| Agent | Main responsibility |
|---|---|
| `DataAgent` | Fetches live quote data from configured sources and caches results. |
| `ValidationAgent` | Checks source reliability, stale data, price differences, and whether analysis should continue. |
| `HistoricalDataAgent` | Loads or downloads OHLCV history and works with the SQLite-first storage layer. |
| `AnalystAgent` | Builds quote-level and historical technical analysis using trend, momentum, volatility, RSI, and volume. |
| `TrainingAgent` | Trains or loads the signal model, generates model signals, and runs automatic diagnostics during the stock pipeline. |
| `RiskAgent` | Applies hard safety rules and uses DQN as an advisory risk-control layer. |
| `StrategistAgent` | Converts the risk-controlled signal into a cautious paper decision plan. |
| `LLMReportAgent` | Explains structured outputs in normal language and summarises news or report context. |
| `ScreenerAgent` | Scores a watchlist and returns research candidates and caution candidates. |
| `RewardAgent` | Records paper decisions and prepares delayed reward updates. |
| `EvaluatorAgent` | Reviews completed rewards, paper-decision history, and DQN readiness. |
| `StorageAgent` | Saves historical prices, pipeline runs, agent outputs, rewards, training metadata, and reports. |
| `ExecutionAgent` | Records UI sessions and saves an audit trail of what the user ran. |

The code is organised so larger agents can be split into folders such as:

```text
agents/risk/
agents/storage/
agents/analysis/
agents/training/
```

Small compatibility files such as `agents/risk_agent.py`, `agents/storage_agent.py`, `agents/analyst_agent.py`, and `agents/training_agent.py` can still remain at the top level of `agents/` so imports stay simple.

The important point is that **Training Optimizer is treated as part of the Training Agent workflow**, not as a separate standalone agent in the user-facing design.

---

## DQN risk layer

The Risk Agent includes a DQN-style advisory layer. It uses PyTorch components such as a Q-network, target network, replay memory, and feedback from delayed paper rewards.

The DQN is not allowed to override the hard safety rules. The risk system has two layers:

1. **Hard safety rules**  
   These handle non-negotiable conditions such as weak data confidence, high risk, or unsafe model signals.

2. **DQN advisory layer**  
   This learns from delayed paper reward feedback after enough completed reward samples exist.

The DQN-related files are organised under:

```text
agents/risk/
  dqn.py
  replay.py
  dqn_policy.py
  state.py
  feedback.py
```

The final output remains cautious. If the DQN suggests a risky action, the hard safety rules can still downgrade or block the decision.

---

## SQLite memory

SQLite is the main local memory layer for the project. The default database file is:

```text
data/trading_system.db
```

Important stored records include:

```text
historical_prices
historical_metadata
market_quotes
pipeline_runs
agent_outputs
paper_decisions
reward_updates
risk_dqn_replay
training_runs
screener_runs
llm_reports
ui_sessions
ui_agent_records
ui_chart_records
```

CSV files may still be kept for seed data, fallback, or backward compatibility, but the main project design is SQLite-first.

If the local database was created by an older version, run:

```bash
python scripts/repair_sqlite_schema.py
```

This repairs missing compatibility columns, such as reward-horizon fields, without deleting old data.

---

## Why the displayed price may differ from finance websites

The app may show a price that is slightly different from Google Finance, Yahoo Finance, or a broker page. This is normal for a research prototype.

Common reasons include:

- different data providers;
- delayed quotes;
- cached values inside the app;
- regular-market versus pre-market or post-market handling;
- different refresh timestamps;
- adjusted close versus latest quote differences.

For this project, a small difference is acceptable because the system is not a real-time trading execution platform. The UI shows source and timestamp information so the user can understand where the displayed price came from.

---

## Why evaluator metrics may show `N/A`

The Evaluator may show `N/A` for reward win rate, directional win rate, or average reward.

This usually means the system has recorded paper decisions, but the reward horizons have not completed yet. For example, if a paper decision has a 7-day or 30-day horizon, the system needs future price data before it can fairly calculate the result.

To generate evaluator metrics:

1. run the single-stock pipeline several times;
2. keep paper decision recording enabled;
3. wait until at least one reward horizon is complete;
4. run the Evaluator again.

Until enough future data exists, `N/A` is the correct honest result. It is not a crash.

---

## Why `DQN Ready` may be false

`DQN Ready = False` is normal during early use.

The DQN needs enough completed replay samples before its training output becomes meaningful. A replay sample is created only after a paper decision has a completed reward result.

Before enough replay samples exist, the DQN should be treated as early-stage and advisory only. The system intentionally avoids pretending the DQN is ready before there is enough feedback.

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

All outputs should be read as research notes. Before making any real financial decision outside this prototype, a user should check the original data source, company news, valuation, event risk, and their own risk tolerance.
