# Multi-Agent Trading Advisory System

This project is a Streamlit prototype for a human-in-the-loop stock research assistant. It is designed for coursework and demonstration, not for real trading. The system reads market data, checks data quality, runs several specialised agents, applies risk control, explains the output in plain language, and records the session for later paper-decision review.

The main idea is that the software agent does more than answer a prompt. It perceives market inputs, runs a multi-step workflow, uses tools and APIs, stores memory in SQLite, applies safety rules, and gives the human user a final research summary to review.

This is a **paper decision-support system only**. It does not connect to a broker, does not place orders, and does not provide financial advice.

---

## Project purpose

The project was built to demonstrate an intelligent software agent prototype that can:

- collect stock data from external tools and local storage;
- validate whether the data is reliable enough to analyse;
- combine live quote movement with historical technical signals;
- train or load a lightweight signal model;
- apply hard risk rules and a DQN advisory risk layer;
- generate a cautious strategy plan for paper research;
- explain the result in user-friendly language;
- record runs, agent outputs, and paper decisions in SQLite;
- evaluate delayed paper outcomes when enough future data becomes available.

The system is intentionally human-in-the-loop. It does not make a final real-world investment decision for the user. It produces structured evidence and a cautious research note that the user can inspect.

---

## Installation

### 1. Clone or unzip the project

```bash
git clone https://github.com/jasonkong65/Real-Time-Human-in-the-Loop-Multi-Agent-Trading-System.git
cd Real-Time-Human-in-the-Loop-Multi-Agent-Trading-System
```

For a submitted zip file, unzip the folder and open the project root.

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
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main packages are `streamlit`, `pandas`, `yfinance`, `requests`, `scikit-learn`, `joblib`, `torch`, `groq`, `python-dotenv`, and `pytest`.

Using `python -m ...` is recommended because it runs the command from the active virtual environment. This avoids common Windows PowerShell issues where `streamlit` or `pytest` is installed but not recognised as a direct command.

### 4. Add environment variables

Copy the example file:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
copy .env.example .env
```

Then add the API keys you want to use:

```text
FINNHUB_API_KEY=your_finnhub_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
GROQ_API_KEY=your_groq_key
```

The app can still open if some keys are missing, but the related live quote or LLM feature may run in a limited mode.

### 5. Repair SQLite schema if needed

If you already have an older local database from previous runs, run:

```bash
python scripts/repair_sqlite_schema.py
```

This is safe to run more than once. It adds missing compatibility columns without deleting existing records.

### 6. Run the app

```bash
python -m streamlit run app.py
```

If Streamlit keeps showing an old cached interface, run:

```bash
python -m streamlit cache clear
python -m streamlit run app.py
```

### 7. Run tests

```bash
python -m pytest -q
```

The test suite checks imports, helper functions, SQLite storage, workflow helpers, Risk Agent behaviour, Training Agent basics, and several agent-level smoke tests.

### 8. Demo video and report

The 2-minute demo video is available through Google Drive:

[Watch the 2-minute demo video](https://drive.google.com/file/d/1nJ9XhS1o1XWVcbZdoMbMJ789qv5aLfGT/view?usp=drive_link)

A copy of the report PDF is included at `docs/report/Report.pdf`. For formal submission, the report should also include the GitHub repository link and this demo video link.

---

## Workflow

The normal single-stock workflow is:

```text
User input
  |
  |-- ticker symbol
  |-- selected modules
  |-- optional portfolio or event context
  v
Data Agent
  |
  |-- fetches quote data
  |-- uses local cache to reduce API pressure
  v
Validation Agent
  |
  |-- checks source reliability
  |-- checks stale data and price consistency
  v
Historical Data Agent
  |
  |-- loads or downloads OHLCV history
  |-- reuses SQLite/local historical data where possible
  v
Analyst Agent
  |
  |-- analyses trend, momentum, volatility, RSI, and volume
  |-- produces an analyst score and entry-risk label
  v
Training Agent
  |
  |-- trains or loads the signal model
  |-- runs automatic model diagnostics internally
  |-- updates the saved model only if the quality gate accepts it
  v
Risk Agent
  |
  |-- applies hard risk-control rules
  |-- uses DQN as an advisory layer
  v
Strategist Agent
  |
  |-- creates a cautious paper-research plan
  v
LLM Report Agent
  |
  |-- writes a plain-language explanation when Groq is available
  v
Storage / Execution / Reward / Evaluator
  |
  |-- records the run
  |-- stores agent outputs
  |-- tracks paper decisions
  |-- evaluates delayed rewards later
```

There are also optional branches:

```text
News / Report Agent
  -> summarises company-specific news, financial context, or pasted report text.

Watchlist Screener Agent
  -> ranks a small watchlist into research candidates and caution candidates.
```

---

## How to use

### Single-stock research

1. Enter a stock ticker such as `AAPL`, `MSFT`, `NVDA`, or `TSLA`.
2. Keep the core modules selected:

```text
Single-stock agent pipeline
Price chart
```

3. Click **Run selected research**.
4. Start with the coloured summary cards at the top of the page:

```text
Symbol | Price | Analyst | Model | Risk | Strategy
```

5. Read **Groq / Report Agent Output** for the plain-language explanation.
6. Read **Strategy Guidance** for the suggested paper-research action and next checks.
7. Use the **Chart** tab to change the chart period. The chart period can be changed without rerunning the full pipeline.
8. Use **Agent Responses** only when you want to inspect the raw structured output from each agent.

### Portfolio context

Portfolio context is optional. Use it only if the user already has a paper position and wants the strategy to consider current quantity and average cost. This helps the Strategist Agent give more cautious position-aware guidance.

### News / report mode

Use the News / Report Agent when you want the system to summarise company news or pasted financial text. The pasted text box is for news, earnings, or report content. It is not for entering another ticker symbol.

### Watchlist screener

Use the Watchlist Screener Agent to compare a small list of stocks. For example:

```text
AAPL, MSFT, NVDA, TSLA, GOOGL, AMZN, META, AMD, NFLX
```

The screener returns research candidates, caution candidates, buy score, risk score, sector, and liquidity-filter status. It is a watchlist-ranking tool, not a full-market scanner.

### Evaluator and logs

The Evaluator tab is used to review completed paper outcomes. The Storage / Logs tab shows recorded UI sessions, pipeline runs, and stored agent outputs. These pages are useful for demonstrating memory and evaluation, even if the evaluator is still in an early stage.

---

## Main features

- Streamlit interface for interactive stock research
- Multi-agent pipeline with separated responsibilities
- Live quote collection and historical OHLCV handling
- Data validation before downstream analysis
- Technical analysis using trend, momentum, volatility, RSI, and volume
- Training Agent with automatic model diagnostics
- Risk control with hard rules and a DQN advisory layer
- Plain-language report generation with Groq when available
- Watchlist screener for comparing several tickers
- Coloured summary cards for fast interpretation
- SQLite-first memory layer
- Paper decision recording and delayed reward tracking
- Pytest-based project checks
- Clear safety framing for coursework and paper research

---

## Agents and responsibilities

| Agent | Responsibility |
|---|---|
| `DataAgent` | Fetches live quote data from configured sources and caches results. |
| `ValidationAgent` | Checks source reliability, stale data, price gaps, and whether analysis should continue. |
| `HistoricalDataAgent` | Loads or downloads OHLCV history and works with the SQLite-first storage layer. |
| `AnalystAgent` | Analyses live quote movement and historical technical indicators. |
| `TrainingAgent` | Trains or loads the signal model, generates model signals, and runs automatic diagnostics. |
| `RiskAgent` | Applies hard safety rules and uses DQN as an advisory risk-control layer. |
| `StrategistAgent` | Converts the risk-controlled signal into a cautious paper decision plan. |
| `LLMReportAgent` | Explains agent outputs in normal language and summarises news/report context. |
| `ScreenerAgent` | Scores a watchlist and returns research and caution candidates. |
| `RewardAgent` | Records paper decisions and prepares delayed reward updates. |
| `EvaluatorAgent` | Reviews completed rewards, paper-decision history, and DQN readiness. |
| `StorageAgent` | Stores historical prices, pipeline runs, agent outputs, rewards, model metadata, and reports. |
| `ExecutionAgent` | Records UI sessions and creates an audit trail of what the user ran. |

The larger agents are organised into folders so the project is easier to maintain:

```text
agents/risk/
agents/storage/
agents/analysis/
agents/training/
```

Small wrapper files such as `agents/risk_agent.py`, `agents/storage_agent.py`, `agents/analyst_agent.py`, and `agents/training_agent.py` remain in the root of `agents/` so existing imports stay simple.

---

## DQN

The DQN component belongs to the Risk Agent. It is implemented with PyTorch and uses a policy network, target network, replay memory, and delayed paper reward feedback.

The DQN is advisory only. It does not replace the hard safety rules. The risk system works in two layers:

1. **Hard safety rules** handle weak data, unsafe risk levels, and blocked signals.
2. **DQN advisory control** learns from delayed paper rewards after enough replay samples exist.

This design keeps the system safer for a coursework prototype. The final risk output remains cautious, and the DQN cannot turn an unsafe setup into a direct trading recommendation.

---

## SQLite

SQLite is the main local memory layer. The default database path is:

```text
data/trading_system.db
```

Important tables include:

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

CSV files are kept only for seed data, fallback, or compatibility. The project design is SQLite-first.

---

## The price may be slightly different from finance websites

The displayed price may not exactly match Google Finance, Yahoo Finance, or a broker page. This is expected.

Small differences can happen because of:

- different data providers;
- delayed quotes;
- app caching;
- regular-market versus pre-market or post-market handling;
- different refresh timestamps;
- adjusted close versus latest quote differences.

This project is a research prototype, not a real-time order execution system. A small difference between displayed prices is acceptable for the assignment demo.

---

## Evaluator shows `N/A`

The Evaluator may show `N/A` for reward win rate, directional win rate, or average reward. This usually means that paper decisions have been recorded, but their reward horizons have not completed yet.

For example, a 7-day paper reward needs future price data after 7 days. Until that data exists, the honest output is `N/A`, not a guessed score.

To generate evaluator metrics:

1. run the single-stock pipeline several times;
2. keep paper-decision memory enabled;
3. wait until at least one reward horizon is complete;
4. run the Evaluator again.

---

## `DQN Ready` is false

`DQN Ready = False` is normal during early use. The DQN needs enough completed replay samples before it can train meaningfully.

A replay sample is created only after a paper decision has a completed reward result. Before enough replay samples exist, the DQN remains early-stage and advisory.

---

## Common command issues

If PowerShell says `pytest` is not recognised, run:

```powershell
python -m pip install pytest
python -m pytest -q
```

If PowerShell says `streamlit` is not recognised, run:

```powershell
python -m pip install streamlit
python -m streamlit run app.py
```

If several packages are missing, reinstall the full requirements file inside the activated virtual environment:

```powershell
python -m pip install -r requirements.txt
```

---

## Safety statement

This project is for education, coursework, and paper decision support only.

It does not:

- place trades;
- connect to a broker;
- guarantee returns;
- provide financial advice;
- recommend leverage;
- replace the user's own judgement.

All outputs should be treated as research notes for a prototype. Before making any real financial decision outside this project, a user should check the original market data, company news, valuation, event risk, and their own risk tolerance.
