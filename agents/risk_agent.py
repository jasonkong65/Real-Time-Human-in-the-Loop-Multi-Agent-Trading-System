from agents.risk import RiskAgent

"""Compatibility import for the refactored RiskAgent.

The implementation now lives in agents/risk/. Existing code can still use:
    from agents.risk_agent import RiskAgent
"""
__all__ = ['RiskAgent']
