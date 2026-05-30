from agents.storage import StorageAgent

"""Compatibility import for the refactored StorageAgent.

The implementation now lives in agents/storage/. Existing code can still use:
    from agents.storage_agent import StorageAgent
"""

__all__ = ['StorageAgent']
