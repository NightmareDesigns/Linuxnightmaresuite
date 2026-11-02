"""nightmare_memory — ModelMemory 10,000 engine, link-compatible.

Drop this file next to nightmare_link.py AND copy the model_memory/
folder next to it (so model_memory/model_memory.py sits beside this file),
or make sure `model_memory` is importable some other way. The link
does `from nightmare_memory import get_model_memory` and expects a
ModelMemoryEngine with knowledge.search/stats, start_session,
end_session, teach, quiz, correct, get_context_for_prompt and
get_stats. This file provides exactly that, backed by the ModelMemory
10,000 engine in model_memory/model_memory.py.

Wiring (no link changes needed):
  1. Copy nightmare_memory.py into the folder with
     nightmare_link.py (v13_work/ or release/).
  2. Copy the model_memory/ folder into the same place, so the layout is:
         nightmare_link.py
         nightmare_memory.py
         model_memory/model_memory.py
         model_memory.json   (the live ModelMemory store)
     (On the Matrix the live store is
     D:\\advanced-ai-chat-with-soul-and-memory\\
     model_memory.json — get_model_memory() looks there too.)
  3. Start the link. /memory/status then reports available: true
     and the /memory/* endpoints work instead of 503.

Local only. No web server. Stdlib only. Original code.
"""

import json
import os
import sys
from pathlib import Path

__all__ = ["get_model_memory", "ModelMemoryEngine", "ModelMemory"]

def _find_model_memory_package():
    """Make `model_memory` importable. Returns True on success."""
    try:
        import model_memory  # noqa: F401
        return True
    except Exception:
        pass
    here = Path(__file__).resolve().parent
    candidates = [
        here,                       # model_memory/ next to this file
        here.parent,                # model_memory/ one up
        here.parent.parent,         # model_memory/ two up (Suite root)
        Path.home() / "workspace" / "suite",
    ]
    for cand in candidates:
        if (Path(cand) / "model_memory" / "model_memory.py").is_file():
            sys.path.insert(0, str(Path(cand)))
            try:
                import model_memory  # noqa: F401
                return True
            except Exception:
                continue
        if (Path(cand) / "model_memory.py").is_file():
            # The single module file sits next to the link.
            sys.path.insert(0, str(Path(cand)))
            try:
                import model_memory  # noqa: F401
                return True
            except Exception:
                continue
    return False


def _find_db_path():
    """Where the live ModelMemory store is, or None."""
    env = os.environ.get("NTMRE_MEMORY_DB")
    if env and Path(env).is_file():
        return Path(env)
    here = Path(__file__).resolve().parent
    for cand in (
        here / "model_memory.json",
        here.parent / "model_memory.json",
    ):
        if Path(cand).is_file():
            return Path(cand)
    return None


_ENGINE = None


def _fresh_store_doc():
    """Fresh empty store, same schema the seal script writes."""
    return {
        "chats": [],
        "memories": [],
        "messagesByChat": {},
        "meta": "model_memory",
        "personas": [],
        "retrieved_at": None,
    }


def _default_db_path():
    """Where a missing store should be created, not silently lost."""
    env = os.environ.get("NTMRE_MEMORY_DB")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent / "model_memory.json"


def get_model_memory(db_path=None):
    """Return the ModelMemoryEngine the link expects (singleton).

    Loads the store if one is found (autosave on, so teach /
    correct / reinforce persist). If no store file exists, a fresh
    empty model_memory.json is CREATED next to this file (or at
    the explicit db_path / NTMRE_MEMORY_DB path) and used with
    autosave on — never an in-memory autosave=False engine that
    silently loses everything on exit. Never returns None.
    """
    global _ENGINE
    if _ENGINE is not None and db_path is None:
        # If the store file was deleted while the engine was alive,
        # recreate it from the in-memory state instead of silently
        # losing the next save.
        try:
            if _ENGINE.db_path is not None and not Path(_ENGINE.db_path).is_file():
                _ENGINE.save()
        except Exception:
            pass
        return _ENGINE
    if not _find_model_memory_package():
        raise RuntimeError(
            "model_memory package not found — copy the model_memory/ folder "
            "next to nightmare_memory.py")
    from model_memory import ModelMemory, ModelMemoryEngine  # noqa: F401
    path = Path(db_path) if db_path else _find_db_path()
    if path is None:
        path = _default_db_path()
    if not Path(path).is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_fresh_store_doc(), f, ensure_ascii=False, indent=2)
            f.write("\n")
    store = ModelMemory.load(path)
    _ENGINE = ModelMemoryEngine(store, db_path=path, autosave=True)
    return _ENGINE


# Re-exported for callers that want the classes directly.
try:
    if _find_model_memory_package():
        from model_memory import ModelMemory, ModelMemoryEngine  # noqa: F401
except Exception:
    ModelMemory = None
    ModelMemoryEngine = None
