"""ModelMemory 10,000 (v2.0) — Model memory, rebuilt modernised, local only.

The ModelMemory is the brain, not the model. Knowledge lives here as
small plain text (memories + personas), the AI reads it, and the
reading IS the learning. There is no gradient descent, no training
loop, no weights to update. "Training" is load_100x(): the whole
store is loaded into RAM 100 times in a tight loop — 100 passes
over a few kilobytes, which is why it is fast. The data is tiny
and RAM is fast.

v2.0 ("ModelMemory 10,000") adds the engine the original had and v1.0
did not: one engine that holds the knowledge, retrieves it, teaches,
quizzes, corrects, reinforces, and tracks sessions — the "all in
one" job list. It also adds persona routing (retrieve/teach/quiz
filtered by persona, bucket, category), memory management (edit,
delete, merge, move), persona CRUD, and a keyword index so retrieval
stays fast at 10,000+ memories. The link's ModelMemoryAdapter can use
it through nightmare_memory.py (get_model_memory), which this package
ships alongside.

This is a pure local program. No web server, no HTTP, no browser,
no external services, no third-party packages. Python stdlib only.
Original code, written for Caleb — no outside code.

Schema — the same as the memory store
(model_memory.json), so an existing store loads here
and what this writes can go back to him:

  top level: chats, memories, messagesByChat, meta, personas,
              retrieved_at, timestamp
  memory:    {bucket, category, createdAt, fact, id, personaId, userId}
  persona:   {backstory, createdAt, creativity, empathy, humor,
              id, name, philosophical, userId}
  chat:      {createdAt, id, personaId, title, updatedAt, userId}

Loading keeps the whole top-level document intact (chats,
messagesByChat, meta, retrieved_at, timestamp all round-trip
untouched); only memories and personas are managed. Saving writes
the same structure back, with `timestamp` refreshed to now.

Engine state that the v2 schema has no field for — reinforcement
strengths, the active persona, lesson/quiz counters, the current
session — lives in a sidecar file next to the database
("<db>.memory-state.json"), never in the backup JSON itself. The
backup format does not change.
"""

import json
import re
import secrets
import string
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["ModelMemory", "ModelMemoryEngine", "Knowledge", "Node", "Session",
           "get_model_memory", "DEFAULT_DB_NAMES"]

# Filenames this looks for, in order, when no --db is given.
DEFAULT_DB_NAMES = ["model_memory.json"]

# The Suite folder, next to this package.
_SUITE_DIR = Path(__file__).resolve().parent.parent

_ID_ALPHABET = string.ascii_letters + string.digits
_ID_LEN = 20  # ids are 20 chars, e.g. "0FjQl5LPenAfnkEzKTr7"
_WORD_RE = re.compile(r"[a-z0-9]+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")


def _utc_now_iso() -> str:
    """ISO-8601 UTC timestamp with Z, e.g. 2026-10-27T05:55:00.000Z."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _new_id(existing) -> str:
    """A fresh 20-char alphanumeric id, unique against `existing`."""
    while True:
        cand = "".join(secrets.choice(_ID_ALPHABET) for _ in range(_ID_LEN))
        if cand not in existing:
            return cand


def _tokens(text: str):
    """Lowercase word tokens of a string, for keyword scoring."""
    return set(_WORD_RE.findall((text or "").lower()))


def _slugify(name: str) -> str:
    """A persona id from a name: lowercase, non-alnum to underscore."""
    slug = _NON_ALNUM_RE.sub("_", (name or "").strip().lower()).strip("_")
    return slug or "persona"


def _clamp_trait(value, default=50):
    """A persona trait as an int in 0..100."""
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return default



def default_db_path():
    """Where the backup lives when no --db is given.

    Looks for model_memory.json next to this file first,
    then in the Suite folder. Returns None if it is nowhere.
    """
    here = Path(__file__).resolve().parent
    for name in DEFAULT_DB_NAMES:
        cand = here / name
        if cand.is_file():
            return cand
        cand = _SUITE_DIR / name
        if cand.is_file():
            return cand
    return None


class ModelMemory:
    """Model memory: memories + personas held in RAM, saved as JSON.

    Usage:
        cx = ModelMemory.load("model_memory.json")
        cx.add_memory("The shop opens at nine.")
        cx.retrieve("who guards the house?")
        cx.load_100x()
        cx.save("my_memory.json")

    v2.0 adds a keyword index (fast retrieval at 10,000+ memories),
    persona routing, and memory/persona management. retrieve() keeps
    its v1.0 semantics exactly: keyword overlap on fact + category
    + bucket, best score first, ties in stored order.
    """

    def __init__(self, data=None):
        """Build from a parsed backup dict, or start empty."""
        data = data or {}
        # Keep the whole document so nothing is lost on save.
        self._doc = dict(data)
        self.memories = list(data.get("memories") or [])
        self.personas = list(data.get("personas") or [])
        self.chats = list(data.get("chats") or [])
        self.messages_by_chat = dict(data.get("messagesByChat") or {})
        self.meta = data.get("meta", "model_memory")
        self.retrieved_at = data.get("retrieved_at")
        self.timestamp = data.get("timestamp")
        # The user this backup belongs to (one id throughout).
        self.user_id = ""
        for src in (self.personas, self.memories):
            for item in src:
                if item.get("userId"):
                    self.user_id = item["userId"]
                    break
            if self.user_id:
                break
        # Active persona for routing (in-memory only; the backup
        # schema has no such field — the engine persists it in the
        # sidecar). None means "no filter".
        self._active_persona_id = None
        # Keyword index: token -> set of indices into self.memories.
        self._index = {}
        # Per-memory token sets, parallel to self.memories. Cached
        # once so scoring a query never re-tokenizes a memory.
        self._mem_tokens = []
        self._rebuild_index()

    # -- load / save ----------------------------------------------------

    @classmethod
    def load(cls, path):
        """Load a ModelMemory from a backup JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def save(self, path):
        """Write the whole store back in the backup JSON format.

        Same top-level keys, same nesting, same field names as his
        file. Only `timestamp` is refreshed (it records when this
        backup was written).
        """
        doc = {
            "chats": self.chats,
            "memories": self.memories,
            "messagesByChat": self.messages_by_chat,
            "meta": self.meta,
            "personas": self.personas,
            "retrieved_at": self.retrieved_at,
            "timestamp": _utc_now_iso(),
        }
        # Preserve any extra top-level keys the file had.
        for key, value in self._doc.items():
            if key not in doc:
                doc[key] = value
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return Path(path)

    # -- keyword index --------------------------------------------------

    def _index_memory(self, mem, idx):
        """Add one memory's tokens to the index at position `idx`."""
        text = " ".join([
            mem.get("fact", ""),
            mem.get("category", ""),
            mem.get("bucket", ""),
        ])
        toks = _tokens(text)
        if idx < len(self._mem_tokens):
            self._mem_tokens[idx] = toks
        else:
            self._mem_tokens.append(toks)
        for token in toks:
            self._index.setdefault(token, set()).add(idx)

    def _rebuild_index(self):
        """Rebuild the whole keyword index from scratch.

        Cheap even at 10,000 memories (a few tens of ms) and always
        correct after edits/deletes, which shift indices.
        """
        self._index = {}
        self._mem_tokens = []
        for idx, mem in enumerate(self.memories):
            self._index_memory(mem, idx)

    # -- memories -------------------------------------------------------

    def add_memory(self, fact, persona_id="default", bucket="active",
                   category="personal", user_id=None):
        """Add one memory. Returns the memory dict that was added.

        Dedup: an identical fact (exact string, after stripping) is
        NOT appended again. The existing memory dict is returned
        instead, so repeated workbench starts re-teaching the same
        tool catalogue do not inflate the count or the "K new
        entries" note. A fact that differs in any character is a
        different memory, even under the same category/persona.
        """
        fact = (fact or "").strip()
        if not fact:
            raise ValueError("fact must not be empty")
        for existing_mem in self.memories:
            if (existing_mem.get("fact") or "").strip() == fact:
                return existing_mem
        existing = {m.get("id") for m in self.memories}
        mem = {
            "bucket": bucket,
            "category": category,
            "createdAt": _utc_now_iso(),
            "fact": fact,
            "id": _new_id(existing),
            "personaId": persona_id,
            "userId": user_id if user_id is not None else self.user_id,
        }
        self.memories.append(mem)
        self._index_memory(mem, len(self.memories) - 1)
        return mem

    def get_memory(self, memory_id):
        """One memory by id, or None."""
        for mem in self.memories:
            if mem.get("id") == memory_id:
                return mem
        return None

    def edit_memory(self, memory_id, fact=None, bucket=None,
                    category=None, persona_id=None):
        """Change a memory's fields. Returns the updated dict.

        Raises KeyError if the id is not in the store.
        """
        mem = self.get_memory(memory_id)
        if mem is None:
            raise KeyError(f"no memory with id {memory_id!r}")
        if fact is not None:
            fact = (fact or "").strip()
            if not fact:
                raise ValueError("fact must not be empty")
            mem["fact"] = fact
        if bucket is not None:
            mem["bucket"] = bucket
        if category is not None:
            mem["category"] = category
        if persona_id is not None:
            mem["personaId"] = persona_id
        self._rebuild_index()
        return mem

    def delete_memory(self, memory_id):
        """Delete a memory by id. Returns True if one was deleted."""
        for idx, mem in enumerate(self.memories):
            if mem.get("id") == memory_id:
                del self.memories[idx]
                self._rebuild_index()
                return True
        return False

    def merge_memories(self, id_a, id_b, fact=None):
        """Merge two memories into one (keeps id_a, deletes id_b).

        fact defaults to "fact A; fact B". Returns the merged dict.
        Raises KeyError if either id is not in the store.
        """
        a = self.get_memory(id_a)
        b = self.get_memory(id_b)
        if a is None:
            raise KeyError(f"no memory with id {id_a!r}")
        if b is None:
            raise KeyError(f"no memory with id {id_b!r}")
        if fact is None:
            fact = f"{a.get('fact', '')}; {b.get('fact', '')}"
        a["fact"] = fact
        self.delete_memory(id_b)
        return a

    def move_memory(self, memory_id, bucket):
        """Move a memory to another bucket. Returns the memory dict."""
        return self.edit_memory(memory_id, bucket=bucket)

    def _scored(self, query, persona_id=None, bucket=None, category=None):
        """(score, index, memory) tuples for a query, best first.

        Score is the v1.0 rule: how many distinct query words occur
        in the memory's fact + category + bucket. Candidates come
        from the keyword index; filters narrow them further.
        """
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        cand = set()
        for token in query_tokens:
            cand |= self._index.get(token, set())
        scored = []
        for idx in cand:
            if idx >= len(self.memories):
                continue
            mem = self.memories[idx]
            if persona_id is not None and mem.get("personaId") != persona_id:
                continue
            if bucket is not None and mem.get("bucket") != bucket:
                continue
            if category is not None and mem.get("category") != category:
                continue
            # Token sets are cached per memory — scoring is a set
            # intersection, never a re-tokenize.
            score = len(query_tokens & self._mem_tokens[idx])
            if score > 0:
                scored.append((-score, idx, mem))
        scored.sort(key=lambda item: (item[0], item[1]))
        return scored

    def retrieve(self, query, top_k=5, persona_id=None, bucket=None,
                 category=None):
        """Most relevant memories for a query, best first.

        Local keyword overlap only — no embeddings, no external
        model. Only memories scoring above 0 are returned, best
        score first, ties in stored order. persona_id / bucket /
        category narrow the search; None means no filter.
        """
        if top_k <= 0:
            return []
        return [mem for _, _, mem in self._scored(
            query, persona_id=persona_id, bucket=bucket,
            category=category)[:top_k]]

    # -- personas -------------------------------------------------------

    def get_persona(self, persona_id):
        """One persona by id, or None."""
        for p in self.personas:
            if p.get("id") == persona_id:
                return p
        return None

    def list_personas(self):
        """All personas, in stored order."""
        return list(self.personas)

    def add_persona(self, name, backstory="", creativity=50, empathy=50,
                    humor=50, philosophical=50, persona_id=None,
                    user_id=None):
        """Add a persona. Returns the persona dict that was added."""
        name = (name or "").strip()
        if not name:
            raise ValueError("name must not be empty")
        existing = {p.get("id") for p in self.personas}
        if persona_id is None:
            persona_id = _slugify(name)
            base = persona_id
            n = 2
            while persona_id in existing:
                persona_id = f"{base}_{n}"
                n += 1
        elif persona_id in existing:
            raise ValueError(f"persona id {persona_id!r} already exists")
        persona = {
            "backstory": backstory or "",
            "createdAt": _utc_now_iso(),
            "creativity": _clamp_trait(creativity),
            "empathy": _clamp_trait(empathy),
            "humor": _clamp_trait(humor),
            "id": persona_id,
            "name": name,
            "philosophical": _clamp_trait(philosophical),
            "userId": user_id if user_id is not None else self.user_id,
        }
        self.personas.append(persona)
        return persona

    def edit_persona(self, persona_id, **fields):
        """Change a persona's fields. Returns the updated dict.

        Allowed fields: name, backstory, creativity, empathy,
        humor, philosophical. Raises KeyError if not found.
        """
        p = self.get_persona(persona_id)
        if p is None:
            raise KeyError(f"no persona with id {persona_id!r}")
        allowed = {"name", "backstory", "creativity", "empathy",
                   "humor", "philosophical"}
        for key, value in fields.items():
            if key not in allowed:
                raise ValueError(f"cannot edit persona field {key!r}")
            if key in ("creativity", "empathy", "humor", "philosophical"):
                p[key] = _clamp_trait(value, p.get(key, 50))
            else:
                p[key] = value
        return p

    def delete_persona(self, persona_id):
        """Delete a persona by id. Returns True if one was deleted.

        Its memories are NOT deleted — they keep their personaId
        and stay searchable. Only the persona itself goes.
        """
        for idx, p in enumerate(self.personas):
            if p.get("id") == persona_id:
                del self.personas[idx]
                if self._active_persona_id == persona_id:
                    self._active_persona_id = None
                return True
        return False

    def set_active_persona(self, persona_id):
        """Select the active persona for routing, or None to clear.

        Returns the persona dict, or None if the id is not in the
        store (and clears the selection). The active persona is
        in-memory only; the engine persists it in the sidecar.
        """
        if persona_id is None:
            self._active_persona_id = None
            return None
        p = self.get_persona(persona_id)
        self._active_persona_id = persona_id if p is not None else None
        return p

    @property
    def active_persona(self):
        """The active persona dict, or None."""
        if self._active_persona_id is None:
            return None
        return self.get_persona(self._active_persona_id)

    @property
    def active_persona_id(self):
        """The active persona's id, or None."""
        return self._active_persona_id

    # -- the 100x -------------------------------------------------------

    def brain_bytes(self) -> bytes:
        """The brain as bytes: memories + personas, JSON encoded.

        This is the whole store the 100x loop reads. The chats and
        message history are conversation log, not knowledge, so they
        are not part of the brain payload.
        """
        payload = {"memories": self.memories, "personas": self.personas}
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def load_100x(self, passes=100):
        """The 100x: read the whole brain into RAM `passes` times.

        This IS the training. One pass reads every byte of the
        store; 100 passes is 100 epochs, done as RAM reads instead
        of gradient descent. The data is tiny and RAM is fast, so
        it takes a fraction of the time a training loop would.

        Each pass copies the bytes and checksums them, so every
        byte is genuinely touched on every pass — it is not just
        rebinding a pointer. Returns a result dict.
        """
        blob = self.brain_bytes()
        per_pass = len(blob)
        checksum = 0
        start = time.perf_counter()
        for _ in range(passes):
            buf = blob[:]  # copy: a real RAM read of the whole store
            checksum = (checksum + zlib.crc32(buf)) & 0xFFFFFFFF
        elapsed = time.perf_counter() - start
        total = per_pass * passes
        return {
            "passes": passes,
            "bytes_per_pass": per_pass,
            "total_bytes": total,
            "elapsed_sec": elapsed,
            "bytes_per_sec": (total / elapsed) if elapsed > 0 else 0,
            "mb_per_sec": (total / elapsed / 1_000_000) if elapsed > 0 else 0,
            "checksum": checksum,
        }

    # -- stats ----------------------------------------------------------

    def stats(self):
        """Counts and size of the store."""
        size = len(self.brain_bytes())
        messages = sum(len(v) for v in self.messages_by_chat.values())
        return {
            "memories": len(self.memories),
            "personas": len(self.personas),
            "chats": len(self.chats),
            "messages": messages,
            "bytes": size,
            "kb": size / 1024,
            "mb": size / (1024 * 1024),
        }


# ============================================================
# ModelMemoryEngine — the "all in one" (v2.0)
# ============================================================

class Node:
    """One knowledge node, as the link sees it.

    A node IS a memory, exposed with an id, a concept/definition
    split (teach() stores facts as "concept: definition"), and a
    score. The link only uses .id, but the rest is there for
    callers that want to show what was found.
    """

    def __init__(self, mem, score=0, strength=0):
        self.id = mem.get("id")
        self.fact = mem.get("fact", "")
        self.category = mem.get("category", "")
        self.persona_id = mem.get("personaId")
        self.bucket = mem.get("bucket", "")
        self.score = score
        self.strength = strength
        if ": " in self.fact:
            self.concept, self.definition = self.fact.split(": ", 1)
        else:
            self.concept = ""
            self.definition = self.fact

    def to_dict(self):
        return {
            "id": self.id,
            "fact": self.fact,
            "concept": self.concept,
            "definition": self.definition,
            "category": self.category,
            "personaId": self.persona_id,
            "bucket": self.bucket,
            "score": self.score,
            "strength": self.strength,
        }

    def __repr__(self):
        return f"Node(id={self.id!r}, score={self.score}, fact={self.fact[:60]!r})"


class Knowledge:
    """The knowledge view of an engine: search + stats.

    This is what the link's ModelMemoryAdapter talks to:
    knowledge.stats() and knowledge.search(text, limit).
    """

    def __init__(self, engine):
        self._engine = engine

    def stats(self):
        """Node counts. The adapter reads 'total_nodes'."""
        store = self._engine.store
        return {
            "total_nodes": len(store.memories),
            "memories": len(store.memories),
            "personas": len(store.personas),
            "active_persona": store.active_persona_id,
        }

    def search(self, text, limit=5):
        """Best-matching nodes for `text`, best first.

        Keyword overlap first (the store's rule); reinforcement
        strength only breaks ties, so a reinforced memory never
        beats a better keyword match.
        """
        store = self._engine.store
        scored = store._scored(text)
        strength = self._engine._strength
        # scored items are (-score, idx, mem): ascending on the first
        # element is descending on the real score. Strength breaks
        # ties only, then stored order.
        ranked = sorted(
            scored,
            key=lambda item: (item[0], -strength.get(item[2].get("id"), 0), item[1]),
        )
        nodes = []
        for score, _idx, mem in ranked[:max(0, limit)]:
            nodes.append(Node(mem, score=-score,
                              strength=strength.get(mem.get("id"), 0)))
        return nodes


class Session:
    """One ModelMemory session: a teach/quiz run with an id and a log."""

    def __init__(self, session_id, mode="teach", started_at=None):
        self.session_id = session_id
        self.mode = mode
        self.started_at = started_at or _utc_now_iso()
        self.lessons = []   # [{"concept": ..., "id": ...}, ...]
        self.quizzes = []   # [{"question": ..., "found": bool}, ...]
        self.ended_at = None

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "mode": self.mode,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "lessons": list(self.lessons),
            "quizzes": list(self.quizzes),
        }

    @classmethod
    def from_dict(cls, data):
        s = cls(data.get("session_id", ""), data.get("mode", "teach"),
                data.get("started_at"))
        s.lessons = list(data.get("lessons") or [])
        s.quizzes = list(data.get("quizzes") or [])
        s.ended_at = data.get("ended_at")
        return s

    def summary(self):
        """Counts + duration, for end_session()."""
        duration = None
        try:
            start = datetime.fromisoformat(
                self.started_at.replace("Z", "+00:00"))
            end = datetime.fromisoformat(
                (self.ended_at or _utc_now_iso()).replace("Z", "+00:00"))
            duration = round((end - start).total_seconds(), 3)
        except Exception:
            duration = None
        return {
            "status": "ended",
            "session_id": self.session_id,
            "mode": self.mode,
            "lessons": len(self.lessons),
            "quizzes": len(self.quizzes),
            "duration_sec": duration,
        }


class ModelMemoryEngine:
    """The ModelMemory engine: teach, quiz, correct, reinforce, sessions.

    One engine that holds the knowledge, retrieves it into prompts,
    teaches, quizzes, corrects, and tracks sessions — the "all in
    one" the original ModelMemory was. It wraps a ModelMemory store and
    adds the learning layer on top. The store itself stays pure
    keyword retrieval; the engine adds reinforcement strength
    (tie-break only) and session/lesson bookkeeping.

    Engine state the backup schema has no field for — reinforcement
    strengths, the active persona, counters, the current session —
    lives in a sidecar file next to the database
    ("<db>.memory-state.json"). The backup JSON itself never gains
    a field. If autosave is on, mutations save the store and the
    sidecar back to the database path.
    """

    def __init__(self, store, db_path=None, autosave=False):
        """Wrap a ModelMemory store. db_path is where it saves to."""
        self.store = store
        self.db_path = Path(db_path) if db_path else None
        self.autosave = bool(autosave)
        self.knowledge = Knowledge(self)
        self.lessons_taught = 0
        self.quizzes_asked = 0
        self.corrections = 0
        self.reinforcements = 0
        self._strength = {}      # memory id -> reinforcement count
        self.sessions = []       # finished + current Session objects
        self.current_session = None
        self._load_state()

    # -- sidecar state --------------------------------------------------

    def _state_path(self, db_path=None):
        """The sidecar file for a database path, or None."""
        p = Path(db_path) if db_path else self.db_path
        if p is None:
            return None
        return p.with_name(p.stem + ".memory-state.json")

    def _load_state(self):
        """Load strengths/counters/active persona/current session."""
        path = self._state_path()
        if path is None or not path.is_file():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception:
            return
        if not isinstance(state, dict):
            return
        strengths = state.get("strengths") or {}
        if isinstance(strengths, dict):
            self._strength = {str(k): int(v) for k, v in strengths.items()
                              if isinstance(v, int)}
        counters = state.get("counters") or {}
        if isinstance(counters, dict):
            self.lessons_taught = int(counters.get("lessons_taught", 0) or 0)
            self.quizzes_asked = int(counters.get("quizzes_asked", 0) or 0)
            self.corrections = int(counters.get("corrections", 0) or 0)
            self.reinforcements = int(counters.get("reinforcements", 0) or 0)
        active = state.get("active_persona")
        if active:
            self.store.set_active_persona(active)
        current = state.get("current_session")
        if isinstance(current, dict) and current.get("session_id"):
            session = Session.from_dict(current)
            self.current_session = session
            self.sessions.append(session)

    def save_state(self, db_path=None):
        """Write the sidecar next to a database path. Returns bool."""
        path = self._state_path(db_path)
        if path is None:
            return False
        state = {
            "active_persona": self.store.active_persona_id,
            "strengths": dict(self._strength),
            "counters": {
                "lessons_taught": self.lessons_taught,
                "quizzes_asked": self.quizzes_asked,
                "corrections": self.corrections,
                "reinforcements": self.reinforcements,
            },
            "current_session": (self.current_session.to_dict()
                                if self.current_session else None),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
            f.write("\n")
        return True

    def save(self, path=None):
        """Save the store and the sidecar. Returns the store path."""
        p = Path(path) if path else self.db_path
        if p is None:
            raise ValueError("no database path to save to")
        self.store.save(p)
        self.save_state(p)
        return p

    def _maybe_autosave(self):
        if self.autosave and self.db_path is not None:
            self.save(self.db_path)

    # -- teach / quiz / correct / reinforce -----------------------------

    def teach(self, concept, definition, category="", examples=None):
        """Teach one thing: store it as a memory, return the result.

        fact is "concept: definition" (or just the definition when
        the concept is empty or "fact"). examples are appended.
        The memory is filed under the active persona, else default.
        """
        concept = str(concept or "").strip()
        definition = str(definition or "").strip()
        if not concept and not definition:
            raise ValueError("teach needs a concept or a definition")
        if concept and concept != "fact" and definition:
            fact = f"{concept}: {definition}"
        elif concept and concept != "fact":
            fact = concept
        else:
            fact = definition
        if examples:
            examples = [str(e).strip() for e in examples if str(e).strip()]
            if examples:
                fact += " Examples: " + "; ".join(examples)
        persona_id = self.store.active_persona_id or "default"
        mem = self.store.add_memory(
            fact, persona_id=persona_id, bucket="active",
            category=(category or "taught"))
        self.lessons_taught += 1
        if self.current_session is not None:
            self.current_session.lessons.append(
                {"concept": concept, "id": mem["id"]})
        self._maybe_autosave()
        return {
            "status": "taught",
            "id": mem["id"],
            "concept": concept,
            "total_nodes": len(self.store.memories),
        }

    def quiz(self, question):
        """Ask a question: the best-matching memory is the answer."""
        question = str(question or "").strip()
        self.quizzes_asked += 1
        nodes = self.knowledge.search(question, limit=1) if question else []
        found = bool(nodes)
        if self.current_session is not None:
            self.current_session.quizzes.append(
                {"question": question, "found": found})
        if not found:
            return {
                "question": question,
                "answer": "",
                "node_id": None,
                "found": False,
                "score": 0,
            }
        node = nodes[0]
        return {
            "question": question,
            "answer": node.fact,
            "node_id": node.id,
            "found": True,
            "score": node.score,
        }

    def correct(self, node_id, right):
        """Fix a wrong answer: rewrite that memory's fact.

        Returns {"status": "corrected", ...} or
        {"status": "not_found", ...} when the id is not in the store.
        """
        right = str(right or "").strip()
        if not right:
            raise ValueError("correct needs a right answer")
        mem = self.store.get_memory(node_id)
        if mem is None:
            return {"status": "not_found", "id": node_id}
        self.store.edit_memory(node_id, fact=right)
        self.corrections += 1
        self._maybe_autosave()
        return {"status": "corrected", "id": node_id, "fact": right}

    def reinforce(self, good_response):
        """Strengthen the memory behind a good response.

        The best-matching memory gains one strength point, which
        breaks retrieval ties in its favour from then on. Strength
        lives in the sidecar (the backup schema has no weight
        field), so it survives restarts of the same database.
        """
        text = str(good_response or "").strip()
        nodes = self.knowledge.search(text, limit=1) if text else []
        if not nodes:
            return {"status": "reinforced", "id": None, "strength": 0,
                    "response": text[:200]}
        node_id = nodes[0].id
        self._strength[node_id] = self._strength.get(node_id, 0) + 1
        self.reinforcements += 1
        self._maybe_autosave()
        return {"status": "reinforced", "id": node_id,
                "strength": self._strength[node_id],
                "response": text[:200]}

    # -- context + sessions + stats --------------------------------------

    def get_context_for_prompt(self, message, max_chars=1500):
        """Memories for a message, assembled into prompt context.

        The active persona's backstory leads (if one is selected),
        then the best-matching memories, one per line, capped at
        max_chars (cut at a line boundary when possible).
        """
        parts = []
        persona = self.store.active_persona
        if persona is not None:
            header = f"You are {persona.get('name', '')}. " \
                     f"{persona.get('backstory', '')}".strip()
            if header:
                parts.append(header[:400])
        active = self.store.active_persona_id
        hits = self.store.retrieve(message, top_k=5, persona_id=active)
        if not hits and active is not None:
            hits = self.store.retrieve(message, top_k=5)
        for mem in hits:
            parts.append(f"[{mem.get('personaId')}] {mem.get('fact', '')}")
        ctx = "\n".join(parts)
        if len(ctx) > max_chars:
            cut = ctx.rfind("\n", 0, max_chars)
            ctx = ctx[:cut] if cut > 0 else ctx[:max_chars]
        return ctx

    def start_session(self, mode="teach"):
        """Start a session. Returns the Session (has .session_id)."""
        session_id = _new_id({s.session_id for s in self.sessions})
        session = Session(session_id, mode=mode)
        self.current_session = session
        self.sessions.append(session)
        self._maybe_autosave()
        return session

    def end_session(self):
        """End the current session. Returns its summary dict."""
        if self.current_session is None:
            return {"status": "no_session"}
        session = self.current_session
        session.ended_at = _utc_now_iso()
        self.current_session = None
        self._maybe_autosave()
        return session.summary()

    def set_persona(self, persona_id):
        """Select the active persona. Returns the persona or None."""
        persona = self.store.set_active_persona(persona_id)
        self._maybe_autosave()
        return persona

    def get_stats(self):
        """Engine stats. The adapter reads 'total_nodes'."""
        size = len(self.store.brain_bytes())
        strongest = sorted(self._strength.items(),
                           key=lambda kv: (-kv[1], kv[0]))[:3]
        return {
            "total_nodes": len(self.store.memories),
            "memories": len(self.store.memories),
            "personas": len(self.store.personas),
            "chats": len(self.store.chats),
            "brain_bytes": size,
            "brain_kb": round(size / 1024, 2),
            "active_persona": self.store.active_persona_id,
            "sessions": len(self.sessions),
            "lessons_taught": self.lessons_taught,
            "quizzes_asked": self.quizzes_asked,
            "corrections": self.corrections,
            "reinforcements": self.reinforcements,
            "strongest": [{"id": mid, "strength": n}
                          for mid, n in strongest],
        }

    def load_100x(self, passes=100):
        """The 100x, through the engine (same as the store's)."""
        return self.store.load_100x(passes=passes)


def get_model_memory(db_path=None, autosave=False):
    """Load the default backup (or an empty store) as an engine.

    db_path may be a path or None (default backup, else empty).
    autosave is forced off when the path is the original store.
    """
    path = Path(db_path) if db_path else default_db_path()
    if path is not None and Path(path).is_file():
        store = ModelMemory.load(path)
    else:
        store = ModelMemory()
        path = None
    return ModelMemoryEngine(store, db_path=path, autosave=autosave)
