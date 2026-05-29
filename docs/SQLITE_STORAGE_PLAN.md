# SQLite StorageAgent Integration Plan

## Goal
Add a lightweight persistent memory layer to the multi-agent stock decision-support system without adding a heavy database server.

## Why SQLite
- No server setup required.
- Python has built-in `sqlite3` support.
- Good for coursework/demo scale.
- Produces one portable file: `data/trading_system.db`.

## New File
- `agents/storage_agent.py`

## Database File
- `data/trading_system.db`

Do not commit large production-like databases to GitHub. For demo, committing a tiny sample DB is optional, but normally add this to `.gitignore`:

```gitignore
data/*.db
data/*.db-journal
```

## Tables

### pipeline_runs
One row per single-stock pipeline run. Stores compact summary fields such as symbol, validation confidence, analyst signal, model signal, final signal, risk level, and strategy action.

### agent_outputs
Stores full JSON output from each agent. This provides auditability and reproducibility.

### market_quotes
Stores Finnhub / Alpha Vantage quote snapshots used by the pipeline.

### paper_decisions
Stores paper decision records created by RewardAgent.

### reward_updates
Stores delayed reward updates when future market data becomes available.

### training_runs
Stores model training / optimization metadata.

### screener_runs
Stores screener results.

### llm_reports
Stores Groq or local fallback report outputs.

## Recommended Integration
In `app.py`, after all agent outputs are generated, call:

```python
storage_result = storage_agent.record_pipeline_bundle(
    symbol=clean_symbol,
    multi_quote=multi_quote,
    historical_data=historical_data_summary,
    validation_result=validation_result,
    analysis_result=analysis_result,
    training_result=training_result,
    signal_result=signal_result,
    risk_result=risk_result,
    strategy_result=strategy_result,
    reward_record_result=reward_record_result,
    auto_reward_update_result=auto_reward_update_result,
    llm_report_result=llm_single_stock_report,
)
```

## Assignment Description
You can describe this module as:

> The Storage Agent provides local persistent memory using SQLite. It stores market snapshots, structured outputs from each agent, paper decisions, delayed reward updates, training metadata, screener results, and LLM reports. This improves auditability, reproducibility, and future autonomous evaluation.

## Future Improvements
- Let EvaluatorAgent read directly from SQLite instead of CSV.
- Let RewardAgent write pending decisions directly to SQLite.
- Add automatic database cleanup / retention policy.
- Add a small dashboard to inspect historical decisions and agent outputs.
