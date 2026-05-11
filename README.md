# Real-Time Human-in-the-Loop Multi-Agent Trading System

## Project Description
This project builds a real-time multi-agent trading decision system using live market APIs. The system collects market data, validates it, analyses stock signals, checks risk, and generates trade proposals. The final trading decision remains with the user.

## Agent Architecture
- Data Agent
- Validation Agent
- Analyst Agent
- Strategist Agent
- Risk Agent
- Execution Agent
- Evaluator Agent
- Training Agent

## Setup Instructions
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt