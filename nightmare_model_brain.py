"""nightmare_model_brain — plain model memory in place of Brain.

Sale edition brain: this is NOT Brain, NOT the assistant, and NOT the token transport
token system. It is the separate, plain memory system written for the
Suite — model memory persistence — wrapped so the Suite main can use
it with the same small interface the old brain used, without importing
any Brain/the assistant code:

    brain.is_enabled() -> bool
    brain.store.count() -> int
    brain.store.all() -> list of memory dicts
    brain.store.remember(fact, category=..., weight=...) -> bool
    brain.store.save(force=...) -> None
    brain.observe_turn(user_text, ai_text) -> {"learned": int, ...}

Design, on purpose, to keep context cheap:
- Memories are stored on disk (model_memory.json, autosave on) and
  recalled on demand. They are NOT injected into every prompt.
  The main only adds a short hint ("You have N memories stored.
  Check them when relevant") when there is anything to check.
- The workbench tool catalogue is taught here the same way: one
  memory per domain, plus an index memory, plus how-to-summon.
  The model recalls the tool list when a task actually needs it,
  instead of the whole catalogue eating context every turn.

Original code, stdlib only. Local only, no cloud, no API key.
"""

class _ModelBrainStore:
    """Store interface over a ModelMemoryEngine.

    remember() stores the fact verbatim. teach(concept="fact",
    definition=fact) is the verbatim path in the engine: a concept
    of "fact" means the definition alone becomes the stored fact,
    so no "fact:" prefix or lesson wording is added on top.
    """

    def __init__(self, engine):
        self._engine = engine

    def count(self):
        try:
            return len(self._engine.store.memories)
        except Exception:
            return 0

    def all(self):
        try:
            return list(self._engine.store.memories)
        except Exception:
            return []

    def remember(self, fact, category="", weight=None):
        fact = str(fact or "").strip()
        if not fact:
            return False
        try:
            # Dedup: an identical fact already stored is NOT taught
            # again, so repeated workbench starts do not grow the
            # count or make "K new entries" lie. Return False for
            # a duplicate, True only for a genuinely new memory.
            # weight is a Brain-era tie-break hint; model memory
            # does not use it, so it is accepted and ignored.
            for mem in self._engine.store.memories:
                if (mem.get("fact") or "").strip() == fact:
                    return False
            self._engine.teach("fact", fact, category=category or "taught")
            return True
        except Exception:
            return False

    def save(self, force=False):
        try:
            self._engine.save()
        except Exception:
            # Autosave already persists on teach; a save failure
            # here must never abort a reply.
            pass


class ModelBrain:
    """Plain model memory brain for the sale Suite."""

    def __init__(self, engine):
        self._engine = engine
        self.store = _ModelBrainStore(engine)
        self.enabled = engine is not None

    def is_enabled(self):
        return bool(self.enabled and self._engine is not None)

    def observe_turn(self, user_text, ai_text, learn=True):
        """Remember a completed turn, cheaply.

        Only turns worth remembering are stored, and only as one
        concise memory — not the whole transcript. Trivial turns
        ("ok", "thanks", short replies) are skipped, so memory does
        not fill with junk. Returns {"learned": 0 or 1, ...}.
        """
        if not self.is_enabled() or not learn:
            return {"learned": 0}
        user = str(user_text or "").strip()
        if len(user) < 25:
            return {"learned": 0}
        low = user.lower().strip(".!?, ")
        if low in ("ok", "thanks", "thank you", "yes", "no", "yep",
                   "nope", "cool", "nice", "lol", "haha", "sure",
                   "got it", "done"):
            return {"learned": 0}
        fact = ("Conversation: user asked: %s Assistant replied: %s"
                % (user[:500], str(ai_text or "")[:500]))
        if self.store.remember(fact, category="conversation"):
            return {"learned": 1, "reinforced": 0, "froze": 0}
        return {"learned": 0}

    def get_context_for_prompt(self, message, max_chars=1500):
        """Memories relevant to a message, on demand only."""
        try:
            return self._engine.get_context_for_prompt(message,
                                                      max_chars=max_chars)
        except Exception:
            return ""


def get_model_brain(db_path=None):
    """Build a ModelBrain from nightmare_memory, or raise.

    db_path defaults to the model_memory.json next to this file
    (the sealed sale root), which nightmare_memory already looks
    for. A fresh empty store is used if no file exists yet, so the
    brain is available even on first launch.
    """
    from nightmare_memory import get_model_memory

    engine = get_model_memory(db_path=db_path)
    return ModelBrain(engine)
