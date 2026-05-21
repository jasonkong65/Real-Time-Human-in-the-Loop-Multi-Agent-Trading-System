# LLM-Enhanced Human-in-the-Loop Multi-Agent Trading System

## Project Overview
This project is a prototype intelligent software agent system for real-time stock decision support. It collects live market data, validates multi-source consistency, performs two-stage quantitative analysis, trains or loads a lightweight signal model, and applies rule-based plus Q-learning risk control. The final decision remains with the user.

## System Architecture
User Input  
→ Data Agent  
→ Validation Agent  
→ Two-Stage Analyst Agent  
→ Training Agent / Signal Model  
→ Q-learning Risk Agent  
→ Human Confirmation  

## Agents
- Data Agent: collects Finnhub and Alpha Vantage data.
- Validation Agent: checks multi-source consistency and produces confidence + next_action.
- Analyst Agent: performs quote-level and historical analysis, with fallback.
- Training Agent: trains a Random Forest signal model or uses fallback rules.
- Risk Agent: applies hard safety rules and Q-learning risk adjustment.

## Setup
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt