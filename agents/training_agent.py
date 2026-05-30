from agents.training import TrainingAgent

"""Compatibility import for the refactored TrainingAgent.

The implementation now lives in agents/training/. Existing code can still use:
    from agents.training_agent import TrainingAgent
"""

__all__ = ['TrainingAgent']
