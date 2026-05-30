"""Compatibility import for the refactored RiskAgent.

The implementation now lives in agents/risk/. Existing code can still use:
    from agents.risk_agent import RiskAgent
"""

from agents.risk import RiskAgent

__all__ = ['RiskAgent']
