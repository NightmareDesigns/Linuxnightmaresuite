# -*- coding: utf-8 -*-
"""
cross_model_router.py — Cross-Model Routing Engine
======================================================

Lets the Suite run several DIFFERENT models side by side (any mix of
architectures/formats — GGUF, SafeTensors, NTMRE, ONNX; llama, qwen35,
gemma4, etc.) and automatically route each prompt to whichever registered
model is the best fit, without requiring the models to share tensor
shapes/vocab (unlike weight-level fusion in fusion_engine/).

Design:
  - Loading is delegated to ntmre_router.create_engine(), so every engine
    the Suite already knows how to run (GPU-native, streaming, Qwen3.5
    hybrid, Bonsai, NTMRE C-runtime, ...) works here unchanged.
  - Only a small number of models are kept resident at once (max_loaded);
    least-recently-used models are unloaded automatically to respect the
    Suite's RAM/VRAM budget.
  - Routing is decided per-request from: an explicit model name (OpenAI
    `model` field), keyword/tag scoring against the prompt text, then a
    configured default model.
  - Generation runs against the common "suite model interface" every
    engine already exposes: encode(text) -> ids, forward(token, pos) ->
    logits, decode(ids) -> text, plus .eos_id / .vocab_size. This mirrors
    the sampling loop in nightmare_link_server_13.py so behavior across
    engines stays consistent.

Nightmare Designs — Levi Gates
"""

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import ntmre_router


@dataclass
class ModelRoute:
    name: str
    path: str
    tags: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    is_default: bool = False
    force_cpu: bool = False


class _LoadedModel:
    __slots__ = ("route", "engine", "last_used")

    def __init__(self, route: ModelRoute, engine: Any):
        self.route = route
        self.engine = engine
        self.last_used = time.time()


class CrossModelRouter:
    """Registry + router + lazy loader for multiple simultaneously-available models."""

    def __init__(self, max_loaded: int = 2, status: Callable[[str], None] = print):
        self.routes: Dict[str, ModelRoute] = {}
        self._loaded: Dict[str, _LoadedModel] = {}
        self.max_loaded = max(1, int(max_loaded))
        self.status = status
        self._default_name: Optional[str] = None

    # ---------- registration ----------

    def register(
        self,
        name: str,
        path: str,
        tags: Optional[List[str]] = None,
        keywords: Optional[List[str]] = None,
        is_default: bool = False,
        force_cpu: bool = False,
    ) -> ModelRoute:
        route = ModelRoute(
            name=name,
            path=str(path),
            tags=list(tags or []),
            keywords=[k.lower() for k in (keywords or [])],
            is_default=is_default,
            force_cpu=force_cpu,
        )
        self.routes[name] = route
        if is_default or self._default_name is None:
            self._default_name = name
        return route

    def unregister(self, name: str):
        self.routes.pop(name, None)
        loaded = self._loaded.pop(name, None)
        if loaded is not None:
            self._close_engine(loaded.engine)
        if self._default_name == name:
            self._default_name = next(iter(self.routes), None)

    def list_models(self) -> List[Dict[str, Any]]:
        """OpenAI /v1/models-style listing across every registered model."""
        return [
            {
                "id": name,
                "path": r.path,
                "tags": r.tags,
                "loaded": name in self._loaded,
                "default": name == self._default_name,
            }
            for name, r in self.routes.items()
        ]

    # ---------- routing decision ----------

    def route(self, prompt: str, explicit_model: Optional[str] = None) -> str:
        """Pick which registered model should answer `prompt`."""
        if explicit_model and explicit_model in self.routes:
            return explicit_model

        text = (prompt or "").lower()
        best_name, best_score = None, 0
        for name, r in self.routes.items():
            score = sum(1 for kw in r.keywords if kw in text)
            score += sum(1 for tag in r.tags if tag.lower() in text)
            if score > best_score:
                best_name, best_score = name, score

        if best_name:
            return best_name
        if self._default_name:
            return self._default_name
        if self.routes:
            return next(iter(self.routes))
        raise ValueError("CrossModelRouter has no registered models.")

    # ---------- engine lifecycle (lazy load + LRU eviction) ----------

    def _ensure_loaded(self, name: str):
        if name in self._loaded:
            lm = self._loaded[name]
            lm.last_used = time.time()
            return lm.engine

        route = self.routes.get(name)
        if route is None:
            raise KeyError(f"No such registered model: {name}")

        if len(self._loaded) >= self.max_loaded:
            self._evict_lru()

        self.status(f"[ROUTER] Loading '{name}' ({route.path})...")
        engine = ntmre_router.create_engine(route.path, force_cpu=route.force_cpu, status=self.status)
        self._loaded[name] = _LoadedModel(route, engine)
        return engine

    def _evict_lru(self):
        if not self._loaded:
            return
        lru_name = min(self._loaded, key=lambda n: self._loaded[n].last_used)
        lm = self._loaded.pop(lru_name)
        self.status(f"[ROUTER] Unloading '{lru_name}' (LRU, freeing for new model)...")
        self._close_engine(lm.engine)

    @staticmethod
    def _close_engine(engine: Any):
        for method in ("close", "unload"):
            fn = getattr(engine, method, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass
                return

    def unload_all(self):
        for lm in list(self._loaded.values()):
            self._close_engine(lm.engine)
        self._loaded.clear()

    def unload(self, name: str):
        """Unload one specific model if currently loaded (keeps its registration)."""
        lm = self._loaded.pop(name, None)
        if lm is not None:
            self._close_engine(lm.engine)

    # ---------- generation (uniform across every engine kind) ----------

    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        stop: Optional[List[str]] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ):
        """
        Route `prompt` to the best model and run generation using the common
        encode/forward/decode interface every suite engine exposes.
        Returns (model_name, generated_text, tokens_generated).
        """
        name = model if (model and model in self.routes) else self.route(prompt, model)
        engine = self._ensure_loaded(name)

        ids = engine.encode(prompt, add_bos=True)
        if not ids:
            return name, "", 0
        eos_id = getattr(engine, "eos_id", 2)

        pos = 0
        for tok in ids[:-1]:
            engine.forward(tok, pos)
            pos += 1
        next_tok = ids[-1]

        generated_text = ""
        n_generated = 0
        for _ in range(max_tokens):
            logits = engine.forward(next_tok, pos)
            pos += 1
            logits = logits if isinstance(logits, list) else list(logits)
            next_tok = _sample(logits, temperature, top_p, top_k)
            if next_tok == eos_id:
                break
            piece = engine.decode([next_tok])
            generated_text += piece
            n_generated += 1
            if on_token:
                on_token(piece)
            if stop and any(s in generated_text for s in stop):
                break

        return name, generated_text, n_generated


def build_router_from_discovered_models(
    suite_dir: str, max_loaded: int = 2, status: Callable[[str], None] = print
) -> CrossModelRouter:
    """
    Convenience factory: auto-discover every loadable model via
    ntmre_router.discover_models() and register each with tags guessed from
    its filename, so cross-model routing works out of the box.
    """
    router = CrossModelRouter(max_loaded=max_loaded, status=status)
    found = ntmre_router.discover_models(suite_dir)

    _TAG_HINTS = {
        "code": ["code", "coder", "codellama", "starcoder", "deepseek-coder"],
        "vision": ["vision", "clip", "llava", "vl-"],
        "math": ["math", "wizardmath"],
        "creative": ["story", "writer", "creative", "roleplay"],
    }

    for label, path in found.items():
        low = label.lower()
        tags = [tag for tag, hints in _TAG_HINTS.items() if any(h in low for h in hints)]
        router.register(name=Path(path).stem, path=path, tags=tags, keywords=tags)

    return router


def _sample(logits: List[float], temperature: float = 0.7, top_p: float = 0.9, top_k: int = 0) -> int:
    """Pure-Python nucleus sampler shared by every routed engine (mirrors
    nightmare_link_server_13.sample_token so behavior stays consistent)."""
    n = len(logits)
    if temperature is None or temperature <= 1e-6:
        best, bi = logits[0], 0
        for i in range(1, n):
            if logits[i] > best:
                best, bi = logits[i], i
        return bi
    mx = max(logits)
    inv_t = 1.0 / temperature
    exps = [math.exp((v - mx) * inv_t) for v in logits]
    total = sum(exps)
    order = sorted(range(n), key=lambda i: exps[i], reverse=True)
    if top_k and top_k > 0:
        order = order[:top_k]
    kept, acc = [], 0.0
    limit = top_p if (top_p and 0.0 < top_p < 1.0) else 1.0
    for i in order:
        p = exps[i] / total
        kept.append((i, p))
        acc += p
        if acc >= limit:
            break
    r = random.random() * acc
    run = 0.0
    for i, p in kept:
        run += p
        if r <= run:
            return i
    return kept[-1][0]
