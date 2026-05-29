# Human-in-the-Loop Multi-Agent Stock Research System

This project is a Streamlit-based stock research prototype. It is designed for paper decision support only. It does not place real trades, does not connect to a broker, and should not be treated as financial advice.

The main idea is simple: the user chooses a stock or a watchlist, the system runs several specialised agents, and the final output is written in plain language so a non-technical user can understand why the system is cautious or interested.

## Project purpose

The project demonstrates an intelligent software agent system with:

- multi-source market data collection;
- data validation before analysis;
- historical price analysis and technical scoring;
- automatic signal-model training or loading;
- strict risk control with a PyTorch DQN advisory layer;
- strategy planning for paper decisions;
- LLM/Groq explanation for users;
- SQLite memory for runs, outputs, rewards, UI sessions, and evaluation.

The human remains in control. The system produces research notes, watchlist candidates, and risk-aware strategy guidance, but it never makes a real trade.

## High-level workflow

```text
User input
  |
  |-- Stock symbol, selected modules, optional portfolio/event context
  v
Data Agent
  |
  |-- Collects live quote data from Finnhub / Alpha Vantage or cache
  v
Validation Agent
  |
  |-- Checks source consistency, confidence, stale data, and next action
  v
Historical Data Agent
  |
  |-- Loads or downloads historical OHLCV data
  |-- Stores/reuses historical prices through SQLite-first storage
  v
Analyst Agent
  |
  |-- Stage 1: quote-level movement
  |-- Stage 2: historical trend, momentum, volatility, RSI, volume
  |-- Stage 3: combined technical score and entry-risk label
  v
Training Agent / Signal Model
  |
  |-- Trains or loads a Random Forest / ExtraTrees-style signal model
  |-- Produces BUY_CANDIDATE, HOLD, or SELL_RISK-style signals
  v
Risk Agent
  |
  |-- Applies hard safety rules
  |-- Uses PyTorch DQN as an advisory risk layer
  |-- Final guardrail stays rule-based and cautious
  v
Strategist Agent
  |
  |-- Converts risk-controlled signal into a practical paper strategy
  v
LLM Report Agent
  |
  |-- Turns structured agent outputs into a clear user-facing explanation
  v
Reward / Evaluator / Storage / Execution Agents
  |
  |-- Save paper decisions, update delayed rewards, evaluate outcomes,
      and keep a full audit trail in SQLite.
```

Optional branches:

```text
News / Report Agent
  -> Summarises company news, pasted financial text, or source-grounded context.

Watchlist Screener Agent
  -> Ranks a watchlist into research candidates and caution candidates.
  -> LLM Report Agent can explain the screener result in simple language.
```

## Main features

### 1. Single-stock research

For a stock such as `AAPL`, the app can run the full pipeline:

```text
Data -> Validation -> Historical Data -> Analyst -> Training -> Risk -> Strategy -> LLM Report
```

The output includes:

- current selected price;
- analyst signal;
- model signal;
- risk level;
- strategy action;
- Groq / Report Agent explanation;
- chart preview with selectable period;
- technical agent outputs for inspection.

### 2. Live chart controls

The chart period is controlled under the chart, not only from the sidebar. Changing the chart period refreshes chart data without rerunning the full research pipeline.

Supported chart periods:

- 1 Day;
- 7 Days;
- 30 Days;
- 6 Months;
- 1 Year;
- 2 Years.

### 3. News / report summary

The News / Report Agent is separated from the stock-decision intent. It can be turned on only when needed.

It supports:

- latest company news mode;
- financial-report mode;
- combined news and financial mode;
- pasted text mode.

The app is designed to avoid mixing broad market news with company-specific evidence.

### 4. Watchlist screener

The Watchlist Screener Agent scans a user-provided list of symbols and returns:

- candidates for further research;
- caution candidates;
- buy score;
- risk score;
- sector;
- liquidity filter status;
- technical reason.

The screener is not a full-market scanner. It only ranks the watchlist provided by the user.

### 5. Paper decision memory and evaluation

When memory is enabled, the system records paper decisions and delayed reward horizons. The Evaluator Agent then reports whether there are enough completed outcomes to calculate performance.

At an early stage, it is normal to see:

```text
Reward Win Rate: N/A
Directional Win Rate: N/A
DQN Ready: False
```

That means the system has recorded paper decisions, but the future reward windows have not completed yet.

## Agents and responsibilities

| Agent | Main responsibility |
|---|---|
| `DataAgent` | Fetches live market quotes, uses cache, tracks market session and source timing. |
| `ValidationAgent` | Checks source agreement, reliability, data freshness, and whether downstream analysis should proceed. |
| `HistoricalDataAgent` | Loads/downloads historical OHLCV data and works with database-first storage. |
| `AnalystAgent` | Performs quote-level and historical technical analysis, then produces an analyst score and entry-risk label. |
| `TrainingAgent` | Trains or loads the signal model using walk-forward validation and model comparison. |
| `TrainingOptimizerAgent` | Backward-compatible wrapper around Training Agent's model-selection logic. |
| `RiskAgent` | Applies strict safety rules and a PyTorch DQN advisory layer with replay memory and target network. |
| `StrategistAgent` | Converts the risk-controlled signal into user-facing paper strategy guidance. |
| `RewardAgent` | Records paper decisions and delayed reward horizons. |
| `EvaluatorAgent` | Evaluates completed paper-decision rewards and DQN replay readiness. |
| `ScreenerAgent` | Ranks a watchlist into research candidates and caution candidates. |
| `LLMReportAgent` | Uses Groq when available, or local fallback text, to explain results clearly. |
| `StorageAgent` | Main persistent memory layer. Stores historical prices, runs, outputs, rewards, DQN replay, screener runs, and reports. |
| `ExecutionAgent` | Records UI sessions and audit snapshots. It is not a broker or real execution system. |
| `DatabaseBackend` | Database adapter. SQLite is used by default; PostgreSQL-style URLs are supported for future deployment. |

## Does the project use DQN?

Yes. The DQN implementation is in `agents/risk_agent.py`.

The Risk Agent uses:

- PyTorch neural network (`DQNNetwork`);
- policy network;
- target network;
- replay memory;
- Huber/SmoothL1 loss;
- epsilon-greedy style advisory action;
- SQLite replay table `risk_dqn_replay`;
- CSV compatibility fallback for older workflows.

Important design choice: the DQN is advisory. It can make the system more cautious, but hard risk rules remain the final guardrail. This is safer for a paper decision-support project.

The DQN will usually show as not ready at first because it needs enough delayed reward samples before training becomes meaningful. In the default configuration, the minimum replay target is 100 samples.

## Does the project use SQLite?

Yes. SQLite is the default persistent storage layer:

```text
DATABASE_URL=sqlite:///data/trading_system.db
```

SQLite stores the main memory tables, including:

- `historical_prices`;
- `historical_metadata`;
- `market_quotes`;
- `pipeline_runs`;
- `agent_outputs`;
- `paper_decisions`;
- `reward_updates`;
- `risk_dqn_replay`;
- `training_runs`;
- `screener_runs`;
- `llm_reports`;
- UI session tables created by `ExecutionAgent`.

There are still CSV files in the project. They are used as seed data, mirrors, or backward-compatible fallbacks. The current design is SQLite-first, not CSV-only.

## Folder structure

```text
.
├── app.py                         # Streamlit UI and workflow orchestration
├── agents/                        # Agent classes
│   ├── data_agent.py
│   ├── validation_agent.py
│   ├── historical_data_agent.py
│   ├── analyst_agent.py
│   ├── training_agent.py
│   ├── risk_agent.py
│   ├── strategist_agent.py
│   ├── reward_agent.py
│   ├── evaluator_agent.py
│   ├── screener_agent.py
│   ├── llm_report_agent.py
│   ├── storage_agent.py
│   ├── execution_agent.py
│   └── database_backend.py
├── config/                        # Agent configuration files
├── data/                          # Local data, historical CSV seed data, generated SQLite DB
├── docs/                          # Extra design notes and audit notes
├── models/                        # Saved model metadata / trained models when generated
├── scripts/                       # Helper scripts
├── utils/                         # Shared feature engineering utilities
├── requirements.txt
└── .env.example
```

## Installation

### 1. Download or clone the project

```bash
git clone <your-repository-url>
cd Real-Time-Human-in-the-Loop-Multi-Agent-Trading-System-main
```

Or download the ZIP file and unzip it, then open the project folder in VS Code or a terminal.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The main dependencies are:

- Streamlit;
- pandas;
- yfinance;
- requests;
- scikit-learn;
- joblib;
- torch;
- groq;
- python-dotenv.

### 4. Set environment variables

Copy `.env.example` to `.env`:

Windows PowerShell:

```powershell
copy .env.example .env
```

macOS / Linux:

```bash
cp .env.example .env
```

Then edit `.env`:

```text
FINNHUB_API_KEY=your_finnhub_api_key_here
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.1-8b-instant
DATABASE_URL=sqlite:///data/trading_system.db
```

The app can still show some local/fallback behaviour without all keys, but live quotes and Groq explanations need valid keys.

### 5. Repair or initialise the SQLite schema

If you already have an older local database, run:

```bash
python scripts/repair_sqlite_schema.py
```

This is safe to run before starting the app. It adds missing compatibility columns without deleting existing data.

### 6. Start the app

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## How to use the app

### Basic single-stock demo

1. Enter a stock symbol, for example `AAPL`.
2. Keep `What are you trying to do?` as `Research only`.
3. Keep `Single-stock agent pipeline` and `Price chart` selected.
4. Click `Run selected research`.
5. Read the `Groq / Report Agent Output` and `Strategy Guidance` in the Overview tab.
6. Open `Agent Responses` if you want to inspect each agent's raw output.

### Portfolio-aware review

1. Open `Optional portfolio / event context`.
2. Tick `I currently hold this stock`.
3. Enter shares and average cost.
4. Run the pipeline.
5. The Strategy Agent will include position guidance.

### News/report summary

1. Turn on `Run News / Report Agent`.
2. Select source mode or paste report/news text.
3. Run the workflow.
4. Open the `News / Report` tab.

### Watchlist screener

1. Turn on `Run Watchlist Screener`.
2. Edit the watchlist symbols if needed.
3. Choose `Top N`.
4. Run the workflow.
5. Open the `Screener` tab to view:
   - Groq / Report Agent Summary;
   - Candidates for further research;
   - Caution candidates;
   - Full technical screener result.

### Evaluator

1. Keep `Record paper decision / memory` on.
2. Run several paper decisions over time.
3. Wait for the reward horizons to mature.
4. Open the `Evaluator` tab.

Early N/A values are expected until there are completed reward records.

## Current audit status

The current project was checked for syntax and basic agent readiness.

Confirmed:

- `app.py` and all `agents/*.py` files compile successfully.
- All agent classes instantiate successfully in a clean local run after installing dependencies.
- Screener duplicate-column crash has been guarded against.
- Strategy output is displayed in plain language first, with raw JSON kept in an expander.
- Screener results can now be explained by the LLM Report Agent or local fallback.
- SQLite compatibility repair is included.
- Risk Agent uses strict PyTorch DQN with replay memory and a target network.

Known limitations:

- Live quote quality depends on API keys and market hours.
- Groq output depends on `GROQ_API_KEY`; otherwise local fallback text is used.
- Evaluator metrics are N/A until delayed paper-decision outcomes are completed.
- DQN training is not meaningful until enough replay samples have been collected.
- Some classes are large and should be refactored if this becomes a long-term project.

## Refactoring notes

The project is acceptable for a coursework prototype, but several files are now large:

- `app.py` contains UI, workflow orchestration, chart utilities, and rendering logic.
- `RiskAgent` contains safety rules, DQN model code, replay storage, and DQN training.
- `StorageAgent` contains schema setup, database writes, reads, and compatibility migration.
- `AnalystAgent` contains quote analysis, historical analysis, scoring, and market-context logic.
- `TrainingAgent` contains feature building, walk-forward validation, model selection, saving, and signal generation.

Recommended future split:

```text
app.py
  -> app_state.py
  -> app_sidebar.py
  -> app_rendering.py
  -> app_workflows.py

risk_agent.py
  -> risk_rules.py
  -> dqn_network.py
  -> dqn_replay.py
  -> risk_agent.py

storage_agent.py
  -> schema.py
  -> repositories.py
  -> storage_agent.py

analyst_agent.py
  -> indicators.py
  -> scoring.py
  -> analyst_agent.py
```

This split is not required for the demo, but it would make the code easier to test and maintain.

## Troubleshooting

### The app crashes with a SQLite column error

Run:

```bash
python scripts/repair_sqlite_schema.py
streamlit cache clear
streamlit run app.py
```

### The screener fails on a DataFrame numeric conversion

This version includes a fix for duplicate `close` / `adj_close` columns. If you still see an error, clear Streamlit cache and rerun:

```bash
streamlit cache clear
streamlit run app.py
```

### The price is slightly different from Google Finance

That is expected. Different sources refresh at different times, and this project uses third-party API data for paper research, not exchange-grade real-time execution.

### Evaluator shows N/A

That means no completed delayed reward records exist yet. It is normal during early testing.

### DQN Ready is False

That means replay memory has not collected enough completed reward samples yet. The default minimum target is 100 replay samples.

## Safety statement

This project is for educational and paper decision-support purposes only. It is not a trading bot, not an investment advisor, and not a system for live order execution.
