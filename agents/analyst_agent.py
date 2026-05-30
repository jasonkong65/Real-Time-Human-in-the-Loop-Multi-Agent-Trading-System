"""Compatibility import for the refactored AnalystAgent.

The implementation now lives in agents/analysis/. Existing code can still use:
    from agents.analyst_agent import AnalystAgent
"""

from agents.analysis import AnalystAgent

__all__ = ['AnalystAgent']
