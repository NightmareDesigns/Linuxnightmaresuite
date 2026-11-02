"""ModelMemory 10,000 (v2.0) — Model memory, local only.

This makes the model_memory folder a real package, so `import model_memory`
works when the folder sits next to the link (nightmare_memory.py
imports it that way). The CLI imports the model_memory.py module
directly and is unaffected.
"""

from .model_memory import (ModelMemory, ModelMemoryEngine, Knowledge, Node, Session,
                      get_model_memory, DEFAULT_DB_NAMES)

__all__ = ["ModelMemory", "ModelMemoryEngine", "Knowledge", "Node", "Session",
           "get_model_memory", "DEFAULT_DB_NAMES"]
