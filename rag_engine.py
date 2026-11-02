"""
RAG engine — hand-written retrieval-augmented generation for the Suite.

* Embeddings come from a live engine's `_get_embedding_row` (real token
  vectors, mean-pooled + L2 normalised) or any callable text -> list[float].
* Chunks are ~800-character windows with overlap; persisted as JSON.
* Retrieval is exact cosine over an in-memory matrix (suite is numpy based;
  the no-numpy rule applies only to the pure-Python link sampler).

The GUI wires it up:
    rag = RagEngine(embed_fn, store_dir)
    nightmare_link.set_embedding_provider(rag.embed)   # /v1/embeddings
    context = rag.search(query, top_k=3)
"""

import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import nightmare_np as np

CHUNK_SIZE = 800
CHUNK_OVERLAP = 80


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping windows on whitespace boundaries."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            boundary = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if boundary > start + size // 2:
                end = boundary
        chunks.append(text[start:end])
        if end >= n:
            break
        start = max(end - overlap, 0)
    return chunks


class RagEngine:
    """Persistent, exact-cosine retrieval store + text embedding wrapper."""

    def __init__(self, embed_fn: Optional[Callable[[str], List[float]]] = None,
                 store_dir: Optional[str] = "rag_store"):
        self.embed_fn = embed_fn
        self.store_dir = Path(store_dir)
        self._mem = str(store_dir) == ":mem:"
        if not self._mem:
            self.store_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._docs: Dict[str, dict] = {}        # id -> {"text","meta","ts"}
        self._vec_list: List[List[float]] = []
        self._vec_id: List[str] = []
        self._mat = None                        # lazy (n, dim) matrix
        self.load()

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    @staticmethod
    def from_engine(engine) -> Callable[[str], List[float]]:
        """Build a mean-pooled token embedder from a Suite engine object
        (needs `.encode(text)` and `._get_embedding_row(token_id)`)."""
        if engine is None:
            return None

        def embed(text: str) -> List[float]:
            text = str(text).strip()
            if not text:
                text = "."
            try:
                ids = engine.encode(text, add_bos=False)
            except Exception:
                ids = engine.encode(text)
            if not ids:
                ids = engine.encode(text)
            if not ids:
                return [0.0]
            values = None
            dim = getattr(engine, "dim", None)
            for tid in ids:
                try:
                    row = engine._get_embedding_row(int(tid))
                except Exception:
                    row = None
                if row is None or len(row) == 0:
                    continue
                row = np.asarray(row, dtype=np.float32).reshape(-1)
                if dim is None:
                    dim = row.shape[0]
                if values is None:
                    values = np.zeros(dim, dtype=np.float32)
                nrow = row.shape[0]
                if nrow >= dim:
                    values += row[:dim]
                else:
                    values[:nrow] += row
            if values is None:
                return [0.0]
            values /= max(1, len(ids))
            norm = float(np.linalg.norm(values))
            if norm > 1e-9:
                values /= norm
            return [float(x) for x in values.tolist()]

        return embed

    def embed(self, text: str) -> List[float]:
        if self.embed_fn is None:
            raise RuntimeError("No embedder registered on this RagEngine")
        return self.embed_fn(text)

    # ------------------------------------------------------------------
    # Ingest
    # ------------------------------------------------------------------

    def add(self, text: str, metadata: Optional[dict] = None) -> str:
        chunks = _chunk_text(text)
        if not chunks:
            return ""
        with self._lock:
            doc_id = f"doc-{int(time.time() * 1000)}-{len(self._docs)}"
            self._docs[doc_id] = {"text": text, "meta": metadata or {},
                                   "ts": time.time()}
            for c in chunks:
                vec = self.embed(c)
                self._vec_list.append([float(x) for x in vec])
                self._vec_id.append(doc_id)
            self._mat = None
            self.save()
        return doc_id

    def add_many(self, items):
        ids = []
        if isinstance(items, dict):
            items = items.items()
        for it in items:
            if isinstance(it, tuple) and len(it) >= 2:
                text, meta = it[0], it[1]
            elif isinstance(it, dict):
                text, meta = it.get("text", ""), it.get("metadata")
            else:
                text, meta = it, None
            doc_id = self.add(text, meta)
            if doc_id:
                ids.append(doc_id)
        return ids

    def ingest_file(self, path: str, metadata: Optional[dict] = None) -> Optional[str]:
        p = Path(path)
        if not p.is_file():
            return None
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            text = ""
        if not text:
            return None
        return self.add(text, dict(metadata or {}, source=str(p)))

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _matrix(self):
        if self._mat is None and self._vec_list:
            self._mat = np.asarray(self._vec_list, dtype=np.float32)
        return self._mat

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        if not self._vec_list:
            return []
        q = np.asarray(self.embed(query), dtype=np.float32).reshape(1, -1)
        mat = self._matrix()
        if mat.shape[1] != q.shape[1]:
            # dimension mismatch (embedder changed) -> degrade to first rows
            w = min(mat.shape[1], q.shape[1])
            scores = mat[:, :w] @ q[:, :w].T
        else:
            scores = mat @ q.T
        scores = scores.reshape(-1)
        order = np.argsort(-scores)[:top_k]
        out = []
        for idx in order:
            doc_id = self._vec_id[int(idx)]
            doc = self._docs.get(doc_id, {})
            out.append({"score": float(scores[int(idx)]),
                        "chunk_index": int(idx),
                        "doc_id": doc_id,
                        "text": self._vec_chunk(int(idx)),
                        "metadata": doc.get("meta", {})})
        return out

    def _vec_chunk(self, idx: int) -> str:
        # return the source text slice most representative of this chunk
        total = sum(1 for _ in self._vec_id)
        _ = total
        return self._bytes_source(int(idx))

    def _bytes_source(self, idx: int) -> str:
        # pragmatic: store chunkless; map chunk idx back proportionally is
        # overkill — instead return the owning document's text truncated.
        doc_id = self._vec_id[idx]
        return self._docs.get(doc_id, {}).get("text", "")[:1600]

    def stats(self) -> dict:
        return {"documents": len(self._docs), "chunks": len(self._vec_list)}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self):
        self._docs = {}
        self._vec_list = []
        self._vec_id = []
        if self._mem:
            return
        data_path = self.store_dir / "rag_store.json"
        if not data_path.is_file():
            return
        try:
            data = json.loads(data_path.read_text(encoding="utf-8", errors="replace"))
        except Exception:
            return
        self._docs = data.get("docs", {})
        self._vec_list = data.get("vectors", [])
        self._vec_id = data.get("vec_ids", [])
        self._mat = None

    def save(self):
        if self._mem:
            return
        data_path = self.store_dir / "rag_store.json"
        tmp = data_path.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps({
                "docs": self._docs,
                "vectors": self._vec_list,
                "vec_ids": self._vec_id,
            }, ensure_ascii=False), encoding="utf-8")
            tmp.replace(data_path)
        except Exception as e:
            print(f"[RAG] save failed: {e}")

    def clear(self):
        with self._lock:
            self._docs = {}
            self._vec_list = []
            self._vec_id = []
            self._mat = None
            self.save()

    # ------------------------------------------------------------------
    # Context builder for prompting
    # ------------------------------------------------------------------

    def build_context(self, query: str, top_k: int = 4, max_chars: int = 2400) -> str:
        hits = self.search(query, top_k=top_k)
        parts = []
        used = 0
        for h in hits:
            text = str(h.get("text", "")).strip()
            if not text:
                continue
            if used + len(text) > max_chars:
                text = text[: max_chars - used]
            parts.append(f"[SOURCE {h.get('score', 0):.3f}]\n{text}")
            used += len(text) + 32
            if used >= max_chars:
                break
        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# tiny CLI smoke
# ---------------------------------------------------------------------------

def _smoke(embed_fn, query, expected_hits=0):
    rag = RagEngine(embed_fn, store_dir=":mem:")
    rag.add("The Nightmare Suite fuses two GPUs with a stream engine and a fused router.")
    rag.add("Constrained decoding keeps every token a prefix of valid JSON.")
    rag.add("Mirostat adjusts the surprise budget during sampling.")
    hits = rag.search(query, top_k=2)
    return rag.stats(), hits


if __name__ == "__main__":
    print("rag_engine smoke:", _smoke(lambda t: [float((ord(c) % 9)) for c in t[:8]] + [0.2] * 16, "gpu fusion"))