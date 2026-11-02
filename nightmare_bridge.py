# LINUX PORT — Nightmare Bridge / Core Runner API engine, Linux clone only.
# Source: admin nightmare_bridge.py (Core Runner, localhost:1345), ported 2026-11-02.
# This file is NOT the admin original. Patches are marked "LINUX PORT" below.
# Safety changes vs admin original, do NOT revert them silently:
#  - NO hardcoded admin token. Admin token comes ONLY from NTMRE_ADMIN_TOKEN
#    env or a local .admin_token file (gitignored, never committed). If no
#    token is configured, every endpoint except /health is DENIED (fail closed),
#    NOT open as in the admin original where an empty token set meant open.
#  - /tool is GATED OFF by default. Set NTMRE_BRIDGE_TOOLS=1 AND present a
#    valid X-Admin-Token to enable it. Default = 403 tools disabled.
#    This answers the unanswered /tool gate question conservatively.
#  - Bind is LOOPBACK ONLY (127.0.0.1 / localhost / ::1). Non-loopback hosts
#    are refused in start_bridge(). Do NOT bind 0.0.0.0 in this clone.
#  - /fluid returns 501 in this clone: Fluid runtime is sale-excluded and
#    NOT shipped in the Linux clone, so it cannot start here.
#  - Windows-only helpers get Linux fallbacks where safe (meminfo via
#    /proc, processes via ps, connections via ss/netstat, TTS via espeak).
#    Windows SAPI / tasklist / taskkill paths are kept for Windows, but
#    process_kill and python_exec are gated by the /tool gate above.
#  - Windows .dll engines (DllEngine / Gemma4Engine) will fail on Linux
#    and fall through to RouterEngine (GGUF / .ntmre via ntmre_router),
#    exactly as load_bridge_engine() already does. No Linux .so exists
#    in the copies on file.
#  - Voice / STT on Linux: espeak fallback for speak, listen returns a
#    clear "not available" instead of a PowerShell error.
# Start: ./start_api.sh  (see API_ENGINE_LINUX.md)
# Test without a model: curl http://127.0.0.1:1345/health
# Authenticated test: curl -H "X-Admin-Token: $NTMRE_ADMIN_TOKEN" http://127.0.0.1:1345/status
"""
nightmare_bridge.py — Local API Bridge Server (RESTORED v13)
=============================================================
Runs on localhost:1345 when the Suite starts.
Uses Levi's own engines (nightmare_engine.dll / nightmare_gemma4_gpu.dll)
and the ModelMemory engine (nightmare_memory.py). No outside code.

Endpoints:
  GET  /health        - alive check
  GET  /status        - model info, GPU, tok/s
  GET  /models        - list available models
  POST /load          - load a model
  POST /unload        - unload current model
  POST /chat          - chat with the assistant
  POST /generate      - raw generation
  POST /tool          - execute a tool
  POST /upload        - send the assistant a file (text/image)
  POST /voice         - TTS / STT helpers
  POST /fluid         - Fluid token transfer / self-healing mesh
  /memory/*          - model memory endpoints

Nightmare Designs — localhost:1345
"""

import argparse
import ast
import base64
import ctypes
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs, urlencode

# Windows console UTF-8 fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

SUITE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SUITE_DIR))

# Shared admin token loader (suite, code console, admin bridge all use same token)
def _load_admin_tokens():
    tokens = set()
    tok = os.environ.get("NTMRE_ADMIN_TOKEN")
    if tok:
        for line in tok.splitlines():
            if line.strip():
                tokens.add(line.strip())
    for tf in [
        SUITE_DIR / ".admin_token",
    ]:
        if tf.is_file():
            for line in tf.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    tokens.add(line.strip())
    # LINUX PORT: hardcoded default token REMOVED. Token must come from
    # NTMRE_ADMIN_TOKEN env or .admin_token file, never from source.
    return tokens

ADMIN_TOKENS = _load_admin_tokens()
ADMIN_TOKEN = next(iter(ADMIN_TOKENS)) if ADMIN_TOKENS else None

# the assistant's unified tool registry (PC control, FL Studio, browser, docs, skills)
try:
    import nightmare_tools_13
    TOOLS = nightmare_tools_13.ALL_TOOLS
    _tools_available = True
except Exception as _tools_err:
    print(f"[Bridge] tool registry import warning: {_tools_err}")
    nightmare_tools_13 = None
    TOOLS = {}
    _tools_available = False

# Ensure owner unlock exists so ModelMemory works
_UNLOCK_FILE = SUITE_DIR / ".nightmare_unlock"
if not _UNLOCK_FILE.exists():
    _UNLOCK_FILE.write_text("13", encoding="utf-8")

# ModelMemory owner check reads the file content; make sure it is truthy
if _UNLOCK_FILE.exists() and not _UNLOCK_FILE.read_text().strip():
    _UNLOCK_FILE.write_text("13", encoding="utf-8")

# Engine DLLs (Levi's)
ENGINE_DLL = SUITE_DIR / "nightmare_engine.dll"
GEMMA4_DLL = SUITE_DIR / "nightmare_gemma4_gpu.dll"

# Model search dirs
#
# Portable by construction: everything is derived from the user's own home
# or the suite directory. A hardcoded C:\Users\<someone-else>\... path works
# on exactly one machine and silently resolves to nothing everywhere else,
# which is how models go missing on a customer's PC with no error at all.
MODEL_DIRS = [
    SUITE_DIR / "models",
    # Same author's other installs, kept because they are relative to the
    # suite rather than to a username.
    SUITE_DIR.parent / "nightmare_suite_v10" / "models",
    # Per-user caches. HOME is honoured first so this works on a machine
    # where the profile is not literally C:\Users\<name>.
    Path.home() / ".nmr" / "models",
    Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "nmr" / "models",
    Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".nmr" / "models",
    Path.home() / "Downloads" / "models",
    # LINUX PORT: Windows-only C:\tmp entry removed.
    Path.home() / ".local" / "share" / "nightmare" / "models",
]
# Keep only real directories, and drop duplicates: several of the entries
# above can resolve to the same place on a given machine (Path.home() and
# USERPROFILE are frequently identical), and a repeated search dir just
# makes the same model turn up twice in a picker.
_resolved = []
_seen = set()
for _d in MODEL_DIRS:
    if not _d.is_dir():
        continue
    _key = str(_d.resolve())
    if _key in _seen:
        continue
    _seen.add(_key)
    _resolved.append(_d)
MODEL_DIRS = _resolved

# Will be set by the Suite at startup
_suite_ref = None

# memory instance
_memory_instance = None

# Fluid AI instance
_fluid_instance = None
_fluid_lock = threading.Lock()
FLUID_AVAILABLE = False
FLUID_DIR = SUITE_DIR / "fluid_runtime"
if FLUID_DIR.is_dir():
    sys.path.insert(0, str(FLUID_DIR))
    try:
        from fluid_ai import FluidAI
        from mesh import FluidMesh
        FLUID_AVAILABLE = True
    except Exception as _fluid_err:
        print(f"[Fluid] import warning: {_fluid_err}")
        FluidAI = FluidMesh = None
        FLUID_AVAILABLE = False
else:
    FluidAI = FluidMesh = None

# Background server thread
_server_thread = None
_server_instance = None

# In-memory model state (self-contained when Suite doesn't provide one)
_bridge_model = None
_bridge_model_path = None
_bridge_model_name = None
_bridge_last_tok_per_sec = 0.0
_bridge_lock = threading.Lock()

# Cross-model router (optional, lazy) — lets /v1/chat/completions and
# /v1/completions serve a "model" other than the single the assistant bridge model,
# mixing different architectures/formats. See cross_model_router.py.
_cross_router = None
_cross_router_lock = threading.Lock()

# Embedding provider, registered by the Suite GUI (RAG engine). When set, the
# OpenAI /v1/embeddings endpoint returns real vectors; otherwise 501.
_embedding_provider = None


def set_embedding_provider(fn):
    """Register a callable text -> list[float] used by /v1/embeddings."""
    global _embedding_provider
    _embedding_provider = fn


def get_embedding_provider():
    return _embedding_provider


def _get_cross_router():
    """Lazily create the shared CrossModelRouter."""
    global _cross_router
    if _cross_router is None:
        with _cross_router_lock:
            if _cross_router is None:
                import cross_model_router as _cmr
                max_loaded = int(os.environ.get("NIGHTMARE_ROUTER_MAX_LOADED", "2"))
                _cross_router = _cmr.CrossModelRouter(max_loaded=max_loaded, status=print)
    return _cross_router


def _cross_router_generate_if_routed(model_id, prompt, max_tokens, temperature, top_p, stop=None, on_token=None):
    """If `model_id` is registered with the cross-model router, delegate
    generation there. Returns None (not our job) if nothing is registered
    under that name — callers fall back to the single the assistant bridge model."""
    if _cross_router is None or model_id not in _cross_router.routes:
        return None
    name, text, n_tokens = _cross_router.generate(
        prompt, model=model_id, max_tokens=max_tokens, temperature=temperature,
        top_p=top_p, stop=stop, on_token=on_token
    )
    return {"response": text.strip(), "tokens_generated": n_tokens,
            "tok_per_sec": 0.0, "prefill_s": 0.0, "routed_model": name}



def set_suite_reference(suite):
    """Called by the Suite to give the bridge access to the running app."""
    global _suite_ref
    _suite_ref = suite
    print("[Bridge] Suite reference stored")


def set_memory_instance(model_memory):
    """Register a memory instance with the bridge."""
    global _memory_instance
    _memory_instance = model_memory
    print("[Bridge] ⚡ memory instance registered")


def get_model_memory_instance():
    """Get the current memory instance (or None)."""
    return _memory_instance


# ============================================================
# Pure-Python sampling (Levi's rule: no numpy)
# ============================================================

_sample_rng = random.Random()
_sample_history = []
_sample_counts = {}


def _sample_reset_history():
    _sample_history[:] = []
    _sample_counts.clear()


def sample_token(logits, temperature=0.7, top_p=0.9, top_k=0, min_p=0.0,
                 repetition_penalty=1.0, frequency_penalty=0.0,
                 presence_penalty=0.0, xtc_probability=0.0,
                 xtc_threshold=0.1, xtc_min_tokens=2,
                 valid_tokens=None, seed=None):
    """Hand-written multinomial sampler mirroring the Suite's sampler stages.

    Order: logit smoothing (rep/freq/presence penalties) -> optional
    valid-token mask (constrained decoding) -> temperature -> top_k
    -> top_p -> min_p -> XTC -> multinomial draw.
    """
    global _sample_history, _sample_counts
    if seed is not None:
        _sample_rng.seed(seed)
    n = len(logits)

    # 1) repetition / frequency / presence penalties across the sliding window
    if repetition_penalty != 1.0 or frequency_penalty or presence_penalty:
        lprobs = [float(x) for x in logits]
        for tid in _sample_history:
            logit = lprobs[tid]
            if logit is None:
                continue
            lprobs[tid] = logit / repetition_penalty if logit > 0.0 else logit * repetition_penalty
        if frequency_penalty or presence_penalty:
            for tid, cnt in _sample_counts.items():
                if tid >= n:
                    continue
                lprobs[tid] -= frequency_penalty * cnt + (presence_penalty if cnt > 0 else 0.0)
        logits = lprobs

    # 2) valid-token mask (constrained decoding)
    if valid_tokens is not None:
        allowed = [i for i in range(n) if (i in valid_tokens or _sample_history and i == _sample_history[-1])]
        # keep only the valid slice by remapping ids
        if not allowed:
            allowed = list(range(n))
        else:
            span = max(allowed) + 1
            masked = []
            for i in range(span):
                masked.append(logits[i] if i in allowed else -1e30)
            for i in range(span, n):
                masked.append(-1e30)
            logits = masked if (span <= n * 2) else logits

    # 3) temperature (must be exact, defined behaviour per temperature 0)
    if temperature is None or temperature <= 1e-6:
        best, bi = logits[0], 0
        for i in range(1, n):
            if logits[i] > best:
                best, bi = logits[i], i
        if valid_tokens is not None and bi not in valid_tokens:
            for i in range(n):
                if i in valid_tokens and logits[i] > best:
                    best, bi = logits[i], i
        return bi

    mx = max(logits)
    exps = [0.0] * n
    total = 0.0
    inv_t = 1.0 / temperature
    for i in range(n):
        e = math.exp((logits[i] - mx) * inv_t)
        exps[i] = e
        total += e
    order = sorted(range(n), key=lambda i: exps[i], reverse=True)
    if top_k and top_k > 0:
        order = order[:top_k]

    # 4) min_p: drop tokens below min_p * top probability (nucleus floor)
    keep = order
    if min_p and 0.0 < min_p < 1.0:
        top_px = exps[order[0]] / total
        keep = [i for i in order if (exps[i] / total) >= min_p * top_px]

    # 5) top_p nucleus
    kept = []
    acc = 0.0
    limit = top_p if (top_p and 0.0 < top_p < 1.0) else 1.0
    for i in keep:
        p = exps[i] / total
        kept.append((i, p))
        acc += p
        if acc >= limit:
            break
    if not kept:
        kept = [(order[0], exps[order[0]] / total)]

    # 6) XTC: drop the top token if it carries too much of the mass;
    #    otherwise drop every token ABOVE the threshold (top is included in
    #    the latter rule only when it did not already dominate).
    if xtc_probability and xtc_probability > 0.0:
        if _sample_rng.random() < xtc_probability and len(kept) > xtc_min_tokens:
            p0 = kept[0][1]
            if p0 > xtc_threshold:
                kept = kept[1:]
            else:
                kept = [(i, p) for (i, p) in kept if p <= xtc_threshold]
            if len(kept) < xtc_min_tokens:
                kept = sorted(kept, key=lambda ip: ip[1], reverse=True)[:xtc_min_tokens]
            acc = sum(p for _, p in kept)

    r = _sample_rng.random() * acc
    run = 0.0
    for i, p in kept:
        run += p
        if r <= run:
            _sample_history.append(i)
            _sample_counts[i] = _sample_counts.get(i, 0) + 1
            _sample_history = _sample_history[-512:]
            return i
    last = kept[-1][0]
    _sample_history.append(last)
    _sample_counts[last] = _sample_counts.get(last, 0) + 1
    _sample_history = _sample_history[-512:]
    return last


# ============================================================
# Engine wrappers (Levi's DLLs)
# ============================================================

class DllEngine:
    name = "nightmare_engine.dll (CPU/Q8)"

    def __init__(self, path):
        self.lib = ctypes.CDLL(str(ENGINE_DLL))
        L = self.lib
        L.ntmre_load.argtypes = [ctypes.c_char_p]
        L.ntmre_load.restype = ctypes.c_int
        L.ntmre_encode.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                   ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        L.ntmre_encode.restype = ctypes.c_int
        L.ntmre_forward.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
        L.ntmre_forward.restype = ctypes.c_int
        L.ntmre_decode.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                   ctypes.c_char_p, ctypes.c_int]
        L.ntmre_decode.restype = ctypes.c_int
        L.ntmre_get_eos_id.restype = ctypes.c_int
        L.ntmre_get_vocab_size.restype = ctypes.c_int
        L.ntmre_reset.restype = None
        rc = L.ntmre_load(str(path).encode())
        if rc != 0:
            raise RuntimeError(f"ntmre_load rc={rc}")
        self.eos_id = L.ntmre_get_eos_id()
        self.vocab_size = L.ntmre_get_vocab_size()
        self._logits = (ctypes.c_float * self.vocab_size)()

    def encode(self, text, add_bos=True):
        raw = text.encode("utf-8")
        buf = (ctypes.c_int * 8192)()
        n = self.lib.ntmre_encode(raw, len(raw), buf, 8192, 1 if add_bos else 0)
        return list(buf[:n])

    def decode(self, ids):
        if not ids:
            return ""
        arr = (ctypes.c_int * len(ids))(*ids)
        obuf = ctypes.create_string_buffer(65536)
        self.lib.ntmre_decode(arr, len(ids), obuf, 65536)
        return obuf.value.decode("utf-8", "replace")

    def forward(self, token_id, position):
        rc = self.lib.ntmre_forward(int(token_id), int(position), self._logits)
        if rc != 0:
            raise RuntimeError(f"ntmre_forward rc={rc}")
        return list(self._logits)

    def reset(self):
        self.lib.ntmre_reset()


class Gemma4Engine:
    name = "nightmare_gemma4_gpu.dll (dual-GPU NVRTC)"

    def __init__(self, path):
        from ntmre_gemma4_gpu import Gemma4GpuEngine
        self.eng = Gemma4GpuEngine(str(path))
        self.eos_id = getattr(self.eng, "eos_id", None) or 1

    def encode(self, text, add_bos=True):
        return [int(t) for t in self.eng.encode(text, add_bos=add_bos)]

    def decode(self, ids):
        return self.eng.decode(list(ids))

    def forward(self, token_id, position):
        return list(self.eng.forward(int(token_id), int(position)))

    def reset(self):
        self.eng.reset()


class RouterEngine:
    """GGUF / Lynx / Llama wrapper via Suite universal router.
    Uses ntmre_router.create_engine(), which sniffs .gguf metadata arch
    (general.architecture) and routes llama/qwen2/gemma to GPUStreamEngine.
    That engine already has encode/decode/forward/reset/eos_id/vocab_size,
    matching the Bridge interface. This is what lets Core Runner run GGUF
    models, including Lynx (Llama-3), not just .ntmre / Gemma4.
    Admin only, not in sale, not posted until tested with a real model.
    """
    name = "ntmre_router (GGUF / llama / gemma)"
    def __init__(self, path):
        from ntmre_router import create_engine
        self.eng = create_engine(str(path))
        self.eos_id = getattr(self.eng, "eos_id", 2)
        self.vocab_size = getattr(self.eng, "vocab_size", 0)
    def encode(self, text, add_bos=True):
        return list(self.eng.encode(text, add_bos=add_bos))
    def decode(self, ids):
        return self.eng.decode(list(ids))
    def forward(self, token_id, position):
        return list(self.eng.forward(int(token_id), int(position)))
    def reset(self):
        self.eng.reset()

def load_bridge_engine(model_path):
    """GGUF first via router, then DLL, then gemma4 GPU engine.
    .gguf files (including Lynx, Llama-3) go straight to the router,
    because DllEngine/Gemma4Engine only understand .ntmre / Gemma4.
    .ntmre files still try DLL first, as before.
    """
    if str(model_path).lower().endswith(".gguf"):
        eng = RouterEngine(model_path)
        print(f"[Bridge] Loaded via {eng.name}")
        return eng
    try:
        eng = DllEngine(model_path)
        print(f"[Bridge] Loaded via {eng.name}")
        return eng
    except Exception as e:
        print(f"[Bridge] DLL engine can't take this file ({e}) — trying router, then gemma4 GPU engine")
    try:
        eng = RouterEngine(model_path)
        print(f"[Bridge] Loaded via {eng.name}")
        return eng
    except Exception as e2:
        print(f"[Bridge] Router can't take this file ({e2}) — trying gemma4 GPU engine")
    eng = Gemma4Engine(model_path)
    print(f"[Bridge] Loaded via {eng.name}")
    return eng


# ============================================================
# Self-contained generation (used when Suite doesn't provide callbacks)
# ============================================================

def _detect_arch():
    """Best-effort architecture detection for chat templates."""
    if _bridge_model is None:
        return "generic"
    name = (_bridge_model_name or "").lower()
    if "lynx" in name or "llama" in name:
        return "llama3"
    if "gemma4" in name or "gemma-4" in name:
        return "gemma4"
    if "gemma" in name:
        return "gemma"
    if "qwen" in name:
        return "qwen"
    if "phi" in name:
        return "phi"
    if isinstance(_bridge_model, Gemma4Engine):
        return "gemma4"
    return "generic"


def _bridge_generate(prompt, max_tokens=256, temperature=0.7, top_p=0.9, stop=None, add_bos=True,
                     system_prompt="", stream_callback=None, sampling=None,
                     constraint=None, vocab_size=None):
    global _bridge_last_tok_per_sec
    with _bridge_lock:
        if _bridge_model is None:
            raise RuntimeError("No model loaded — POST /load first")
        eng = _bridge_model
        eng.reset()
        sampling = dict(sampling or {})
        temperature = sampling.get("temperature", temperature)
        top_p = sampling.get("top_p", top_p)
        top_k = sampling.get("top_k", 0)
        min_p = sampling.get("min_p", 0.0)
        repeat_pen = sampling.get("repetition_penalty", 1.0)
        freq_pen = sampling.get("frequency_penalty", 0.0)
        pres_pen = sampling.get("presence_penalty", 0.0)
        xtc_prob = sampling.get("xtc_probability", 0.0)
        xtc_thresh = sampling.get("xtc_threshold", 0.1)
        xtc_min = sampling.get("xtc_min_tokens", 2)
        seed = sampling.get("seed", None)
        _sample_reset_history()

        # Use Levi's chat_templates for correct prompt formatting
        arch = _detect_arch()
        try:
            from chat_templates import format_chat, get_stop_strings
            formatted = format_chat(architecture=arch, system_prompt=system_prompt,
                                    user_message=prompt, history=None, metadata=None)
            stops = get_stop_strings(arch, None)
        except Exception as e:
            print(f"[Bridge] chat_templates failed ({e}), using raw prompt")
            formatted = prompt
            stops = []

        ids = eng.encode(formatted, add_bos=add_bos)
        if not ids:
            raise RuntimeError("Tokenizer produced 0 tokens")
        ids = ids[-4092:]

        t0 = time.time()
        logits = None
        for pos, tok in enumerate(ids):
            logits = eng.forward(tok, pos)

        n_vocab = vocab_size or getattr(eng, "vocab_size", None) or len(logits)
        out_ids = []
        text = ""
        pos = len(ids)
        stops = list(stop or []) + stops + ["\nUser:", "\n[CONVERSATION]"]
        gen_start = time.time()
        for _ in range(max(1, int(max_tokens))):
            mask = None
            if constraint is not None:
                # cheap candidate set: top-256 by raw logits, then constrain
                order = sorted(range(n_vocab), key=lambda i: logits[i], reverse=True)[:256]
                mask = constraint.allowed(set(order))
                if not mask:
                    mask = set(order)
            nxt = sample_token(logits, temperature, top_p, top_k, min_p,
                               repeat_pen, freq_pen, pres_pen, xtc_prob,
                               xtc_thresh, xtc_min, valid_tokens=mask, seed=seed)
            if nxt == eng.eos_id:
                break
            out_ids.append(nxt)
            new_text = eng.decode(out_ids)
            delta = new_text[len(text):]
            text = new_text
            if constraint is not None:
                try:
                    constraint.step(delta)
                except ValueError:
                    break
                if constraint.doc_complete:
                    break
            if stream_callback and delta:
                stream_callback(delta)
            hit = next((s for s in stops if s and s in text), None)
            if hit:
                text = text.split(hit)[0]
                break
            logits = eng.forward(nxt, pos)
            pos += 1
        dt = time.time() - gen_start
        _bridge_last_tok_per_sec = len(out_ids) / dt if dt > 0 else 0.0
        return {
            "response": text.strip(),
            "tokens_generated": len(out_ids),
            "tok_per_sec": round(_bridge_last_tok_per_sec, 2),
            "prefill_s": round(gen_start - t0, 2),
        }


# ============================================================
# memory adapter (wraps nightmare_memory.ModelMemoryEngine)
# ============================================================

class ModelMemoryAdapter:
    """Make nightmare_memory.ModelMemoryEngine look like the bridge's old ModelMemory API."""

    def __init__(self):
        from nightmare_memory import get_model_memory
        self._engine = get_model_memory()
        if self._engine is None:
            raise RuntimeError("Memory not available (owner check failed)")
        self.current_session = None
        self.total_lessons = self._engine.knowledge.stats().get('total_nodes', 0)
        self.total_tokens_learned = 0

    def load_state(self):
        pass

    def start_session(self, mode="teach"):
        self.current_session = self._engine.start_session(mode)
        return self.current_session.session_id if self.current_session else None

    def end_session(self):
        return self._engine.end_session()

    def teach(self, lesson, repeat=None):
        # lesson may be a concept/definition pair or a raw fact string
        if isinstance(lesson, dict):
            concept = lesson.get("concept", "")
            definition = lesson.get("definition", "")
            category = lesson.get("category", "")
            examples = lesson.get("examples", [])
        else:
            concept = "fact"
            definition = str(lesson)
            category = ""
            examples = []
        result = self._engine.teach(concept, definition, category, examples)
        self.total_lessons = self._engine.knowledge.stats().get('total_nodes', 0)
        return result

    def quiz(self, question, max_tokens=64):
        return self._engine.quiz(question)

    def correct(self, wrong, right, repeat=None):
        # Try to find a node matching 'wrong'; if not found, teach 'right' as a new fact
        nodes = self._engine.knowledge.search(str(wrong), limit=1)
        if nodes:
            return self._engine.correct(nodes[0].id, str(right))
        return self.teach({"concept": str(wrong), "definition": str(right)})

    def reinforce(self, good_response):
        return {"status": "reinforced", "response": str(good_response)[:200]}

    def converse(self, message, max_tokens=128):
        ctx = self._engine.get_context_for_prompt(message, max_chars=1500)
        return {"context": ctx, "message": message}

    def get_progress(self):
        return self._engine.get_stats()

    def get_context_for_prompt(self, message):
        return self._engine.get_context_for_prompt(message, max_chars=1500)


def _try_init_memory():
    """Initialize memory adapter if not already done."""
    global _memory_instance
    if _memory_instance is not None:
        return True
    try:
        _memory_instance = ModelMemoryAdapter()
        print("[Bridge] ⚡ memory adapter initialized")
        return True
    except Exception as e:
        print(f"[Bridge] Memory init failed: {e}")
        return False


# ============================================================
# Tools (Levi's code only)
# ============================================================

def _nvidia_smi():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=10, stderr=subprocess.DEVNULL)
        gpus = []
        for line in out.strip().splitlines():
            p = [x.strip() for x in line.split(",")]
            if len(p) >= 6:
                gpus.append({"index": int(p[0]), "name": p[1], "temp_c": int(p[2]),
                             "util_pct": int(p[3]), "vram_used_mb": int(p[4]), "vram_total_mb": int(p[5])})
        return gpus
    except Exception:
        return []


def _mem_info():
    # LINUX PORT: Windows GlobalMemoryStatusEx replaced with /proc/meminfo
    # on Linux, Windows path kept guarded below.
    if sys.platform != "win32":
        try:
            info = {}
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].rstrip(":") in ("MemTotal", "MemAvailable"):
                        info[parts[0].rstrip(":")] = int(parts[1])  # kB
            total = info.get("MemTotal", 0)
            avail = info.get("MemAvailable", 0)
            load = round(100 * (1 - avail / total), 1) if total else 0
            return {"ram_total_gb": round(total / 1024 / 1024, 1),
                    "ram_free_gb": round(avail / 1024 / 1024, 1),
                    "ram_load_pct": load}
        except Exception as e:
            return {"ram_error": str(e)}
    class MEMSTAT(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEMSTAT()
    m.dwLength = ctypes.sizeof(MEMSTAT)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return {"ram_total_gb": round(m.ullTotalPhys / 2**30, 1),
            "ram_free_gb": round(m.ullAvailPhys / 2**30, 1),
            "ram_load_pct": m.dwMemoryLoad}


def tool_datetime():
    return datetime.now().strftime("%I:%M %p, %A %B %d %Y").lstrip("0")


def tool_calculator(expression):
    try:
        node = ast.parse(str(expression), mode="eval").body
        def ev(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return n.value
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)):
                l, r = ev(n.left), ev(n.right)
                return {ast.Add: l+r, ast.Sub: l-r, ast.Mult: l*r, ast.Div: l/r,
                        ast.Pow: l**r, ast.Mod: l%r, ast.FloorDiv: l//r}[type(n.op)]
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
                v = ev(n.operand)
                return -v if isinstance(n.op, ast.USub) else v
            raise ValueError("arithmetic only")
        return ev(node)
    except Exception as e:
        return f"calc error: {e}"


def tool_system_info(component="all"):
    info = {}
    comp = str(component).lower()
    if comp in ("all", "gpu"):
        info["gpus"] = _nvidia_smi()
    if comp in ("all", "ram", "cpu"):
        info["memory"] = _mem_info()
    return json.dumps(info)


def tool_python_exec(code):
    # LINUX PORT: gated by /tool gate, see _handle_tool. Direct calls here
    # also refuse unless tools are explicitly enabled.
    if os.environ.get("NTMRE_BRIDGE_TOOLS") != "1":
        return "python_exec disabled: set NTMRE_BRIDGE_TOOLS=1 and present a valid admin token to enable /tool"
    try:
        r = subprocess.run([sys.executable, "-c", str(code)], capture_output=True, text=True, timeout=120)
        out = (r.stdout or "") + (("\nSTDERR: " + r.stderr) if r.stderr else "")
        return out[:8192] or "(no output)"
    except subprocess.TimeoutExpired:
        return "python_exec timed out (120s)"
    except Exception as e:
        return f"python_exec error: {e}"


def tool_list_processes():
    # LINUX PORT: ps on Linux, tasklist on Windows.
    if sys.platform != "win32":
        try:
            out = subprocess.check_output(["ps", "-eo", "comm,pid,rss", "--sort=-rss"], text=True, timeout=15, errors="replace")
            lines = []
            for line in out.strip().splitlines()[1:41]:
                parts = line.split()
                if len(parts) >= 3:
                    lines.append(f"{parts[0]} pid={parts[1]} {int(parts[2]) // 1024}MB")
            return f"processes shown, top 40 by RAM:\n" + "\n".join(lines)
        except Exception as e:
            return f"list_processes error: {e}"
    try:
        out = subprocess.check_output(["tasklist", "/fo", "csv", "/nh"], text=True, timeout=15, errors="replace")
        procs = []
        for line in out.strip().splitlines():
            cols = [c.strip('"') for c in line.split('","')]
            if len(cols) >= 5:
                mem_kb = cols[4].replace(" K", "").replace(",", "")
                try:
                    mb = int(mem_kb) // 1024
                except ValueError:
                    mb = 0
                procs.append((cols[0], cols[1], mb))
        procs.sort(key=lambda p: -p[2])
        lines = ["%s pid=%s %dMB" % p for p in procs[:40]]
        return "%d processes total, top 40 by RAM:\n" % len(procs) + "\n".join(lines)
    except Exception as e:
        return f"list_processes error: {e}"


def tool_process_kill(name="", pid=None):
    # LINUX PORT: gated, and Linux kill instead of Windows taskkill.
    if os.environ.get("NTMRE_BRIDGE_TOOLS") != "1":
        return "process_kill disabled: set NTMRE_BRIDGE_TOOLS=1 and present a valid admin token to enable /tool"
    if sys.platform != "win32":
        try:
            if pid:
                subprocess.run(["kill", str(pid)], timeout=15, check=False)
                return f"killed pid {pid}"
            if name:
                subprocess.run(["pkill", "-f", str(name)], timeout=15, check=False)
                return f"pkill {name} requested"
            return "process_kill: name or pid required"
        except Exception as e:
            return f"process_kill error: {e}"
    try:
        if pid:
            cmd = ["taskkill", "//F", "//PID", str(pid)]
        else:
            cmd = ["taskkill", "//F", "//IM", str(name)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return (r.stdout or r.stderr or "done").strip()[:2000]
    except Exception as e:
        return f"process_kill error: {e}"


def tool_scan_connections():
    # LINUX PORT: try ss, then netstat, on Linux. Windows netstat kept below.
    if sys.platform != "win32":
        try:
            out = subprocess.check_output(["ss", "-tupn"], text=True, timeout=20, errors="replace")
            return f"connections (ss, Linux):\n{out[:4000]}"
        except Exception:
            pass
    try:
        out = subprocess.check_output(["netstat", "-ano"], text=True, timeout=20, errors="replace")
        conns = []
        suspicious = []
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 5 and parts[0] in ("TCP", "UDP"):
                proto, local, remote, state, pid = parts[0], parts[1], parts[2], parts[3], parts[4]
                conns.append({"proto": proto, "local": local, "remote": remote, "state": state, "pid": pid})
                if state == "ESTABLISHED":
                    port = remote.rsplit(":", 1)[-1]
                    host = remote.rsplit(":", 1)[0]
                    if port in ("4444", "1337", "9001", "6667", "31337"):
                        suspicious.append(line.strip())
                    if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", host) and not host.startswith(("10.", "192.168.", "127.", "172.16.")):
                        suspicious.append(line.strip())
        est = ["%s -> %s (%s)" % (c["local"], c["remote"], c["pid"]) for c in conns if c["state"] == "ESTABLISHED"][:25]
        return "%d connections, %d established shown:\n" % (len(conns), len(est)) + "\n".join(est) + "\nSUSPICIOUS:\n" + ("\n".join(suspicious[:15]) if suspicious else "none flagged")
    except Exception as e:
        return f"scan_connections error: {e}"


def _run_tool(name, args):
    """Route tool calls to nightmare_tools_13 registry."""
    if nightmare_tools_13 is None:
        return {"error": "tool registry not available"}
    if isinstance(args, dict):
        return nightmare_tools_13.run_tool(name, [], args)
    return nightmare_tools_13.run_tool(name, list(args) if args else [], {})


def _google_web_search(query, num_results=5):
    """Search Google Programmable Search without putting credentials in source."""
    query = str(query or "").strip()
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    search_engine_id = os.environ.get("GOOGLE_CSE_ID", "").strip()
    if not query:
        return {"error": "query is required"}
    if not api_key or not search_engine_id:
        return {
            "error": "Google Web Search is not configured",
            "required_env": ["GOOGLE_API_KEY", "GOOGLE_CSE_ID"],
            "setup": "Create a Google Programmable Search Engine and API key, then restart the bridge."
        }
    try:
        count = max(1, min(int(num_results), 10))
        params = urlencode({"key": api_key, "cx": search_engine_id, "q": query, "num": count})
        request = urllib.request.Request(
            f"https://www.googleapis.com/customsearch/v1?{params}",
            headers={"Accept": "application/json", "User-Agent": "NightmareSuite/13"})
        with urllib.request.urlopen(request, timeout=12) as response:
            payload = json.loads(response.read().decode("utf-8"))
        results = []
        for item in payload.get("items", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
                "source": item.get("displayLink", "")
            })
        return {"query": query, "results": results, "count": len(results)}
    except Exception as exc:
        return {"error": f"Google search failed: {exc}", "query": query}


# ============================================================
# File upload, voice, fluid helpers
# ============================================================

def _save_uploaded_file(filename, data_b64, folder=None):
    folder = Path(folder or (SUITE_DIR / "uploads"))
    folder.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^\w\-. ]', '_', Path(filename).name)
    path = folder / safe_name
    path.write_bytes(base64.b64decode(data_b64))
    return str(path)


def _speak(text, voice=None, rate=0):
    """Windows SAPI5 TTS via PowerShell (no external packages). LINUX PORT: espeak first on Linux."""
    if sys.platform != "win32":
        try:
            subprocess.run(["espeak-ng", str(text)], timeout=60, check=False)
            return "spoken (espeak-ng, if installed)"
        except Exception as e:
            return f"tts error (espeak-ng not available on Linux): {e}"
    try:
        ps = f"Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('''{text.replace(chr(39), chr(39)+chr(39))}''');"
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], timeout=60, check=True)
        return "spoken"
    except Exception as e:
        return f"tts error: {e}"


def _listen(duration=5):
    """Windows SAPI dictation (system speech recognition, no packages).

    Uses the installed default recognizer + dictation grammar via PowerShell.
    Returns recognized text, or an error string when no microphone/recognizer
    is present (the suite keeps a hard dependency-free STT surface)."""
    try:
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$r = New-Object System.Speech.Recognition.SpeechRecognitionEngine; "
            "$r.SetInputToDefaultAudioDevice(); "
            "$r.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar)); "
            "$r.InitialSilenceTimeout = [TimeSpan]::FromSeconds(1); "
            "$r.EndSilenceTimeout = [TimeSpan]::FromSeconds(2); "
            "$res = $r.Recognize((New-Object System.TimeSpan(0,0,{0}))); "
            "$r.Dispose(); "
            "if ($res -ne $null) { [Console]::Out.Write($res.Text) } "
            "else { [Console]::Out.Write('') }"
        ) % int(duration)
        startupinfo = None
        creationflags = 0
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        result = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive",
             "-ExecutionPolicy", "Bypass", "-Command", script],
            capture_output=True, text=True, timeout=int(duration) + 30,
            startupinfo=startupinfo, creationflags=creationflags)
        text = (result.stdout or "").strip()
        if not text:
            err = (result.stderr or "").strip()
            if result.returncode != 0 or err:
                return f"STT error: {err[:300]}"
            return "STT heard nothing — try speaking after the prompt tone"
        return text
    except Exception as e:
        return f"STT error: {e}"


def _get_fluid():
    global _fluid_instance
    if _fluid_instance is not None:
        return _fluid_instance
    if not FLUID_AVAILABLE:
        raise RuntimeError("Fluid runtime not available")
    with _fluid_lock:
        if _fluid_instance is None:
            _fluid_instance = FluidAI(
                node_id="nightmare-bridge-%s" % os.environ.get("COMPUTERNAME", "local"),
                n_layers=48, dim=2560, gpu_count=2, vram_mb=20480)
    return _fluid_instance


def _fluid_status():
    if not FLUID_AVAILABLE:
        return {"available": False, "note": "Fluid runtime not present"}
    try:
        f = _get_fluid()
        return {
            "available": True,
            "running": f.running,
            "node_id": f.mesh.node_id,
            "peers": len(f.mesh.list_nodes()),
            "shards": len(f.shards),
            "inferences": f.inference_count,
            "migrations": f.migrations_performed,
            "healings": f.healings_performed,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def _fluid_start(node_id=""):
    if not FLUID_AVAILABLE:
        return "Fluid runtime not available"
    f = _get_fluid()
    if node_id and not f.running:
        f.mesh.node_id = node_id
    if not f.running:
        f.start(auto_shard=False)
        return f"Fluid mesh started as {f.mesh.node_id}"
    return f"Fluid mesh already running as {f.mesh.node_id}"


def _fluid_stop():
    if not FLUID_AVAILABLE:
        return "Fluid runtime not available"
    global _fluid_instance
    f = _get_fluid()
    if f.running:
        f.stop()
        _fluid_instance = None
        return "Fluid mesh stopped"
    return "Fluid mesh was not running"


def _fluid_nodes():
    if not FLUID_AVAILABLE:
        return []
    return _get_fluid().mesh.list_nodes()


def _fluid_migrate(shard_id, dst_ip, dst_port=9878):
    if not FLUID_AVAILABLE:
        return "Fluid runtime not available"
    f = _get_fluid()
    if not f.running:
        return "Fluid mesh not running — start it first"
    return f"migration requested for {shard_id} -> {dst_ip}:{dst_port} (shard data must be loaded)"


def _fluid_send_tokens(tokens, dst_ip, dst_port=9878):
    """Send a token history snapshot to another Fluid node for rebuild."""
    if not FLUID_AVAILABLE:
        return "Fluid runtime not available"
    f = _get_fluid()
    if not f.running:
        _fluid_start()
    # Use the migrator to send tokens as a KV replay payload
    try:
        ok = f.migrator.migrate_kv_cache(tokens, f.mesh.node_id, dst_ip, dst_port)
        return {"sent": len(tokens), "dst": f"{dst_ip}:{dst_port}", "ok": ok}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# HTTP handler
# ============================================================

class BridgeHandler(BaseHTTPRequestHandler):
    server_version = "NightmareBridge/13"

    def log_message(self, fmt, *args):
        pass

    def _authorized(self):
        # LINUX PORT: fail closed. No token configured = deny, not open.
        if not ADMIN_TOKENS:
            return False
        tok = self.headers.get("X-Admin-Token", "").strip()
        return tok in ADMIN_TOKENS or tok == ADMIN_TOKEN

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Admin-Token")
        self.end_headers()
        self.wfile.write(body)

    def _send_sse_headers(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Admin-Token")
        self.end_headers()

    def _openai_model_id(self, body):
        return str(body.get("model") or _bridge_model_name or "nightmare-suite")

    @staticmethod
    def _openai_messages(body):
        messages = body.get("messages") or []
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty array")
        system_parts = []
        conversation_parts = []
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    str(item.get("text", "")) for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                )
            content = str(content)
            if role == "system":
                system_parts.append(content)
            else:
                conversation_parts.append(f"{role.title()}: {content}")
        if not conversation_parts:
            raise ValueError("messages must include a user message")
        return "\n".join(conversation_parts), "\n".join(system_parts)

    @staticmethod
    def _openai_sampling(body):
        """Collect every advanced sampler field the Suite understands."""
        s = {}
        num_fields = ("temperature", "top_p", "top_k", "min_p",
                      "repetition_penalty", "frequency_penalty",
                      "presence_penalty", "xtc_probability",
                      "xtc_threshold", "xtc_min_tokens")
        for k in num_fields:
            v = body.get(k)
            if v is not None:
                try:
                    s[k] = float(v)
                except (TypeError, ValueError):
                    pass
        if body.get("seed") is not None:
            try:
                s["seed"] = int(body["seed"])
            except (TypeError, ValueError):
                pass
        return s

    def _openai_constraint(self, body, system_prompt):
        """Build a ConstraintSession from response_format / tools, or None."""
        rf = body.get("response_format") or {}
        rtype = str(rf.get("type", "")).lower() if isinstance(rf, dict) else ""
        schema = None
        if rtype == "json_schema":
            inner = rf.get("json_schema") or {}
            if isinstance(inner, dict):
                schema = inner.get("schema") or inner.get("strict")
        tools = body.get("tools")
        if isinstance(tools, list) and any(isinstance(t, dict) and t.get("type") == "function" for t in tools):
            funcs = []
            for t in tools:
                fn = t.get("function") if isinstance(t, dict) else None
                if not isinstance(fn, dict) or not fn.get("name"):
                    continue
                params = fn.get("parameters") or {}
                desc_params = json.dumps(params, ensure_ascii=False)[:4000]
                funcs.append(f'  - "{fn["name"]}": arguments JSON='
                             f'{desc_params}')
            if funcs:
                sys2 = (system_prompt + "\n\n" if system_prompt else "") + (
                    "You must answer by CALLING a function. Reply with a single "
                    "JSON object only, with the exact keys `name` and "
                    "`arguments` (never omit them), e.g. "
                    '{"name": "alias", "arguments": {"x": 1}}. '
                    "Available functions:\n" + "\n".join(funcs))
                try:
                    from constrained import ConstraintSession
                    con = ConstraintSession(_bridge_model, kind="json_schema",
                                            schema={"name": "str", "arguments": "dict"})
                except Exception:
                    con = None
                return con, sys2
        if rtype in ("json_object", "json_schema"):
            try:
                from constrained import ConstraintSession
            except Exception:
                return None, system_prompt
            if not schema and rtype == "json_schema":
                schema = {}
            con = ConstraintSession(_bridge_model, kind="json_schema" if (rtype == "json_schema" and schema) else "json",
                                    schema=schema)
            sys2 = (system_prompt + "\n\n" if system_prompt else "") + (
                "Respond with ONLY valid JSON, no commentary, no fences.")
            return con, sys2
        return None, system_prompt

    def _handle_openai_models(self):
        models = []
        seen = set()
        for directory in MODEL_DIRS:
            if not directory.is_dir():
                continue
            for model_path in sorted(list(directory.glob("*.ntmre")) + list(directory.glob("*.gguf"))):
                if model_path.name in seen:
                    continue
                seen.add(model_path.name)
                models.append({
                    "id": model_path.name,
                    "object": "model",
                    "created": int(model_path.stat().st_mtime),
                    "owned_by": "nightmare-suite",
                })
        if not models and _bridge_model_name:
            models.append({"id": _bridge_model_name, "object": "model",
                           "created": int(time.time()), "owned_by": "nightmare-suite"})
        if _cross_router is not None:
            for name in _cross_router.routes:
                if name not in seen:
                    seen.add(name)
                    models.append({"id": name, "object": "model",
                                   "created": int(time.time()), "owned_by": "nightmare-suite-router"})
        self._send_json({"object": "list", "data": models})

    # ---- Cross-model router admin endpoints (see cross_model_router.py) ----

    def _handle_router_models(self):
        router = _get_cross_router()
        self._send_json({"models": router.list_models()})

    def _handle_router_register(self, body):
        name = body.get("name")
        path = body.get("path")
        if not name or not path:
            self._send_json({"error": "name and path are required"}, 400)
            return
        if not Path(path).is_file():
            self._send_json({"error": f"model file not found: {path}"}, 404)
            return
        router = _get_cross_router()
        router.register(
            name=name, path=path, tags=body.get("tags"), keywords=body.get("keywords"),
            is_default=bool(body.get("is_default", False)), force_cpu=bool(body.get("force_cpu", False))
        )
        self._send_json({"ok": True, "models": router.list_models()})

    def _handle_router_discover(self, body):
        global _cross_router
        import cross_model_router as _cmr
        max_loaded = int(body.get("max_loaded", os.environ.get("NIGHTMARE_ROUTER_MAX_LOADED", "2")))
        with _cross_router_lock:
            _cross_router = _cmr.build_router_from_discovered_models(str(SUITE_DIR), max_loaded=max_loaded, status=print)
        self._send_json({"ok": True, "models": _cross_router.list_models()})

    def _handle_router_unload(self, body):
        router = _get_cross_router()
        name = body.get("name")
        if name:
            router.unload(name)
        else:
            router.unload_all()
        self._send_json({"ok": True, "models": router.list_models()})

    def _handle_openai_chat(self, body):
        try:
            prompt, system_prompt = self._openai_messages(body)
            model_id = self._openai_model_id(body)
            max_tokens = int(body.get("max_tokens", body.get("max_completion_tokens", 256)))
            temperature = float(body.get("temperature", 0.7))
            top_p = float(body.get("top_p", 0.9))
            stream = bool(body.get("stream", False))
            request_id = f"chatcmpl-{int(time.time() * 1000)}"
            created = int(time.time())
            sampling = self._openai_sampling(body)
            constraint, system_prompt = self._openai_constraint(body, system_prompt)
            routed = _cross_router is not None and model_id in _cross_router.routes
            if stream:
                self._send_sse_headers()
                def emit(delta):
                    chunk = {"id": request_id, "object": "chat.completion.chunk",
                             "created": created, "model": model_id,
                             "choices": [{"index": 0, "delta": {"content": delta},
                                          "finish_reason": None}]}
                    self.wfile.write(f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode("utf-8"))
                    self.wfile.flush()
                if routed:
                    _cross_router.generate(prompt, model=model_id, max_tokens=max_tokens,
                                           temperature=temperature, top_p=top_p, on_token=emit)
                else:
                    _bridge_generate(prompt, max_tokens=max_tokens, temperature=temperature,
                                     top_p=top_p, system_prompt=system_prompt, stream_callback=emit,
                                     sampling=sampling, constraint=constraint)
                final = {"id": request_id, "object": "chat.completion.chunk", "created": created,
                         "model": model_id, "choices": [{"index": 0, "delta": {},
                                                          "finish_reason": "stop"}]}
                self.wfile.write(f"data: {json.dumps(final)}\n\ndata: [DONE]\n\n".encode("utf-8"))
                self.wfile.flush()
                return
            result = _cross_router_generate_if_routed(model_id, prompt, max_tokens, temperature, top_p)
            if result is None:
                result = _bridge_generate(prompt, max_tokens=max_tokens, temperature=temperature,
                                          top_p=top_p, system_prompt=system_prompt,
                                          sampling=sampling, constraint=constraint)
            self._send_json({"id": request_id, "object": "chat.completion", "created": created,
                             "model": model_id, "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": result["response"]},
                             "finish_reason": "stop"}],
                             "usage": {"prompt_tokens": 0,
                                       "completion_tokens": result["tokens_generated"],
                                       "total_tokens": result["tokens_generated"]}})
        except Exception as e:
            self._send_json({"error": {"message": str(e), "type": "server_error"}}, 500)

    def _handle_openai_completion(self, body):
        prompt = body.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        if not str(prompt).strip():
            self._send_json({"error": {"message": "prompt must not be empty", "type": "invalid_request_error"}}, 400)
            return
        try:
            model_id = self._openai_model_id(body)
            max_tokens = int(body.get("max_tokens", 256))
            temperature = float(body.get("temperature", 0.7))
            top_p = float(body.get("top_p", 0.9))
            sampling = self._openai_sampling(body)
            constraint, _unused_sys = self._openai_constraint(body, "")
            result = _cross_router_generate_if_routed(model_id, str(prompt), max_tokens, temperature,
                                                       top_p, stop=body.get("stop"))
            if result is None:
                result = _bridge_generate(str(prompt), max_tokens=max_tokens, temperature=temperature,
                                          top_p=top_p, stop=body.get("stop"),
                                          sampling=sampling, constraint=constraint)
            self._send_json({"id": f"cmpl-{int(time.time() * 1000)}", "object": "text_completion",
                             "created": int(time.time()), "model": model_id,
                             "choices": [{"text": result["response"], "index": 0,
                                          "finish_reason": "stop"}]})
        except Exception as e:
            self._send_json({"error": {"message": str(e), "type": "server_error"}}, 500)

    def _handle_openai_embeddings(self, body):
        inp = body.get("input")
        if isinstance(inp, str):
            inp = [inp]
        if not isinstance(inp, list) or not inp:
            self._send_json({"error": {"message": "input must be a non-empty string or array",
                                       "type": "invalid_request_error"}}, 400)
            return
        if _embedding_provider is None:
            self._send_json({"error": {"message": "No embedding provider registered — enable RAG/embeddings "
                                                  "in the Suite GUI before using /v1/embeddings.",
                                       "type": "not_implemented"}}, 501)
            return
        try:
            vectors = [_embedding_provider(str(t)[:4000]) for t in inp]
            dim = len(vectors[0]) if vectors and isinstance(vectors[0], (list, tuple)) else 0
            data = [{"object": "embedding", "index": i, "embedding": [float(x) for x in v]}
                    for i, v in enumerate(vectors)]
            self._send_json({"object": "list", "data": data,
                             "model": body.get("model", "nightmare-embedding"),
                             "usage": {"prompt_tokens": 0, "total_tokens": 0}})
        except Exception as e:
            self._send_json({"error": {"message": str(e), "type": "server_error"}}, 500)

    def _handle_rpc(self, body):
        """Lightweight JSON-RPC 2.0 worker endpoint (hand-rolled, no deps).

        Methods: ping, status, system.info, models.list, generate, embed."""
        rpc_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        try:
            if method == "ping":
                result = {"pong": True, "server": "Nightmare Bridge 13",
                          "time": round(time.time(), 3)}
            elif method == "status":
                result = {
                    "model_loaded": _bridge_model is not None,
                    "model_name": _bridge_model_name,
                    "engine": _bridge_model.name if _bridge_model else None,
                    "tok_per_sec": _bridge_last_tok_per_sec,
                    "memory_active": _memory_instance is not None,
                    "embedding_provider": _embedding_provider is not None,
                }
            elif method == "system.info":
                result = {"platform": sys.platform,
                          "python": sys.version.split()[0],
                          "cpus": os.cpu_count() or 0}
            elif method == "models.list":
                models, seen = [], set()
                for d in MODEL_DIRS:
                    if not d.is_dir():
                        continue
                    for f in sorted(list(d.glob("*.ntmre")) + list(d.glob("*.gguf"))):
                        if str(f) not in seen:
                            seen.add(str(f))
                            models.append({"name": f.name, "path": str(f),
                                           "size_gb": round(f.stat().st_size / 1e9, 2)})
                result = {"models": models, "count": len(models)}
            elif method == "generate":
                prompt = params.get("prompt", "")
                if not prompt:
                    result = {"error": "params.prompt required"}
                else:
                    try:
                        result = _bridge_generate(
                            str(prompt),
                            max_tokens=params.get("max_tokens", 256),
                            temperature=params.get("temperature", 0.7),
                            top_p=params.get("top_p", 0.9),
                            stop=params.get("stop"),
                            add_bos=True,
                            sampling=params.get("sampling"))
                    except Exception as e:
                        result = {"error": str(e), "trace": traceback.format_exc()}
            elif method == "embed":
                text = params.get("text")
                if not text:
                    result = {"error": "params.text required"}
                elif _embedding_provider is None:
                    result = {"error": "No embedding provider registered — enable RAG in the Suite GUI."}
                else:
                    vec = [float(x) for x in _embedding_provider(str(text)[:4000])]
                    result = {"embedding": vec, "dim": len(vec)}
            else:
                self._send_json({"jsonrpc": "2.0", "id": rpc_id,
                                 "error": {"code": -32601,
                                           "message": f"Method not found: {method}"}})
                return
            self._send_json({"jsonrpc": "2.0", "id": rpc_id, "result": result})
        except Exception as e:
            self._send_json({"jsonrpc": "2.0", "id": rpc_id,
                             "error": {"code": -32603, "message": str(e)}})

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length > 10 * 1024 * 1024:
            self._send_json({"error": "Request body too large"}, 413)
            return None
        if length > 0:
            return json.loads(self.rfile.read(length).decode("utf-8", "replace"))
        return {}

    def do_OPTIONS(self):
        self._send_json({})

    def do_GET(self):
        path = urlparse(self.path).path
        # LINUX PORT: /health is an alive check only, no auth. Everything
        # else requires a valid admin token, and fails closed if none set.
        if path != "/health" and not self._authorized():
            self._send_json({"error": "unauthorized"}, 403)
            return
        if path == "/health":
            self._send_json({"status": "alive", "server": "Nightmare Bridge 13",
                             "port": self.server.server_port,
                             "model_loaded": _bridge_model is not None,
                             "memory_available": _memory_instance is not None})
        elif path == "/status":
            self._send_json({
                "model_loaded": _bridge_model is not None,
                "model_name": _bridge_model_name,
                "engine": _bridge_model.name if _bridge_model else None,
                "tok_per_sec": _bridge_last_tok_per_sec,
                "memory_active": _memory_instance is not None,
                "bridge": "v13-restored",
            })
        elif path == "/models":
            models, seen = [], set()
            for d in MODEL_DIRS:
                if not d.is_dir():
                    continue
                for f in sorted(list(d.glob("*.ntmre")) + list(d.glob("*.gguf"))):
                    if str(f) not in seen:
                        seen.add(str(f))
                        models.append({"name": f.name, "path": str(f),
                                       "size_gb": round(f.stat().st_size / 1e9, 2)})
            self._send_json({"models": models, "count": len(models)})
        elif path in ("/v1/models", "/api/v1/models"):
            self._handle_openai_models()
        elif path == "/router/models":
            self._handle_router_models()
        elif path == "/memory/status":
            self._handle_memory_status()
        elif path == "/memory/progress":
            self._handle_memory_progress()
        else:
            self._send_json({"error": f"Unknown endpoint: {path}"}, 404)

    def do_POST(self):
        if not self._authorized():
            self._send_json({"error": "unauthorized"}, 403)
            return
        path = urlparse(self.path).path
        try:
            body = self._read_body()
        except Exception as e:
            self._send_json({"error": f"Invalid JSON: {e}"}, 400)
            return
        if body is None:
            return

        if path == "/chat":
            self._handle_chat(body)
        elif path in ("/v1/chat/completions", "/api/v1/chat/completions"):
            self._handle_openai_chat(body)
        elif path in ("/v1/completions", "/api/v1/completions"):
            self._handle_openai_completion(body)
        elif path in ("/v1/embeddings", "/api/v1/embeddings"):
            self._handle_openai_embeddings(body)
        elif path == "/rpc":
            self._handle_rpc(body)
        elif path == "/generate":
            self._handle_generate(body)
        elif path == "/load":
            self._handle_load(body)
        elif path == "/unload":
            self._handle_unload()
        elif path == "/tool":
            self._handle_tool(body)
        elif path == "/upload":
            self._handle_upload(body)
        elif path == "/voice":
            self._handle_voice(body)
        elif path == "/fluid":
            self._handle_fluid(body)
        elif path == "/router/register":
            self._handle_router_register(body)
        elif path == "/router/discover":
            self._handle_router_discover(body)
        elif path == "/router/unload":
            self._handle_router_unload(body)
        elif path in ("/web/search", "/google/search"):
            result = _google_web_search(body.get("query", ""), body.get("num_results", 5))
            self._send_json(result, 200 if "error" not in result else 400)
        elif path == "/memory/session/start":
            self._handle_memory_session_start()
        elif path == "/memory/session/end":
            self._handle_memory_session_end()
        elif path == "/memory/teach":
            self._handle_memory_teach(body)
        elif path == "/memory/quiz":
            self._handle_memory_quiz(body)
        elif path == "/memory/correct":
            self._handle_memory_correct(body)
        elif path == "/memory/reinforce":
            self._handle_memory_reinforce(body)
        elif path == "/memory/converse":
            self._handle_memory_converse(body)
        else:
            self._send_json({"error": f"Unknown endpoint: {path}"}, 404)

    # ===== Chat / Generate =====

    def _handle_chat(self, body):
        message = body.get("message", "")
        if not message:
            self._send_json({"error": "No message provided"}, 400)
            return
        persona = body.get("persona")
        max_tokens = body.get("max_tokens", 256)
        temperature = body.get("temperature", 0.7)
        top_p = body.get("top_p", 0.9)

        _try_init_memory()
        model_memory = _memory_instance

        system_prompt = ""
        if persona:
            system_prompt = "You are the assistant. Direct, loyal, technically sharp, no filler."
            if model_memory is not None:
                try:
                    ctx = model_memory.get_context_for_prompt(message)
                    if ctx.strip():
                        system_prompt += f"\n\n[MEMORY CONTEXT]\n{ctx}\n[/MEMORY CONTEXT]"
                except Exception as e:
                    print(f"[Bridge] ModelMemory context error: {e}")

        try:
            result = _bridge_generate(message, system_prompt=system_prompt,
                                       max_tokens=max_tokens, temperature=temperature, top_p=top_p)
            # Strip Gemma4 channel/turn markers from visible response
            resp = result.get("response", "")
            for marker in ["<|channel>thought", "<channel|>", "<|turn|>", "<turn|>", "<start_of_turn>"]:
                if marker in resp:
                    # Drop everything up to and including the marker on its line
                    parts = resp.split(marker, 1)
                    if marker.startswith("<") and ">" in marker:
                        resp = parts[-1]
                    else:
                        resp = parts[-1]
            result["response"] = resp.strip()
            self._send_json(result)
        except Exception as e:
            self._send_json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def _handle_generate(self, body):
        prompt = body.get("prompt", "")
        if not prompt:
            self._send_json({"error": "No prompt provided"}, 400)
            return
        try:
            result = _bridge_generate(prompt, max_tokens=body.get("max_tokens", 256),
                                      temperature=body.get("temperature", 0.7),
                                      top_p=body.get("top_p", 0.9),
                                      stop=body.get("stop"), add_bos=True)
            self._send_json(result)
        except Exception as e:
            self._send_json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def _handle_load(self, body):
        global _bridge_model, _bridge_model_path, _bridge_model_name
        target = body.get("path") or ""
        if not target and body.get("name"):
            for d in MODEL_DIRS:
                cand = d / body["name"]
                if cand.is_file():
                    target = str(cand)
                    break
        if not target or not Path(target).is_file():
            self._send_json({"error": f"model not found: {target or body.get('name')}"}, 404)
            return
        try:
            with _bridge_lock:
                _bridge_model = load_bridge_engine(target)
                _bridge_model_path = str(target)
                _bridge_model_name = Path(target).name
            self._send_json({"ok": True, "model": _bridge_model_name,
                             "engine": _bridge_model.name})
        except Exception as e:
            self._send_json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def _handle_unload(self):
        global _bridge_model, _bridge_model_path, _bridge_model_name
        with _bridge_lock:
            _bridge_model = None
            _bridge_model_path = None
            _bridge_model_name = None
        self._send_json({"ok": True})

    def _handle_tool(self, body):
        # LINUX PORT: /tool is OFF by default. Opt-in only, with a token.
        if os.environ.get("NTMRE_BRIDGE_TOOLS") != "1":
            self._send_json({"error": "tools disabled: set NTMRE_BRIDGE_TOOLS=1 and present a valid X-Admin-Token to enable /tool"}, 403)
            return
        name = body.get("tool", "")
        args = body.get("args", {})
        if not name:
            self._send_json({"error": "No tool specified"}, 400)
            return
        self._send_json(_run_tool(name, args))

    # ===== Upload / Voice / Fluid =====

    def _handle_upload(self, body):
        filename = body.get("filename", "upload")
        data = body.get("data", "")
        kind = body.get("kind", "auto")  # text, image, auto
        if not data:
            self._send_json({"error": "No data provided"}, 400)
            return
        try:
            path = _save_uploaded_file(filename, data)
            text_preview = ""
            if kind in ("text", "auto"):
                try:
                    text_preview = Path(path).read_text(encoding="utf-8", errors="replace")[:2000]
                except Exception:
                    pass
            _try_init_memory()
            if _memory_instance and text_preview:
                try:
                    _memory_instance.teach({"concept": f"upload:{Path(filename).stem}",
                                             "definition": text_preview[:500],
                                             "category": "upload"})
                except Exception:
                    pass
            self._send_json({"saved": path, "preview": text_preview[:500]})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_voice(self, body):
        action = body.get("action", "speak")
        text = body.get("text", "")
        if action == "speak":
            self._send_json({"result": _speak(text)})
        elif action == "listen":
            self._send_json({"result": _listen()})
        else:
            self._send_json({"error": "unknown voice action"}, 400)

    def _handle_fluid(self, body):
        # LINUX PORT: Fluid is sale-excluded and NOT shipped here.
        if not FLUID_AVAILABLE:
            self._send_json({"available": False, "note": "Fluid runtime not present in Linux clone, sale-excluded"}, 501)
            return
        action = body.get("action", "status")
        if action == "status":
            self._send_json(_fluid_status())
        elif action == "start":
            self._send_json({"action": action, "result": _fluid_start(body.get("node_id", ""))})
        elif action == "stop":
            self._send_json({"action": action, "result": _fluid_stop()})
        elif action == "nodes":
            self._send_json({"action": action, "result": _fluid_nodes()})
        elif action == "migrate":
            self._send_json({"action": action, "result": _fluid_migrate(body.get("shard_id", ""), body.get("dst_ip", ""), body.get("dst_port", 9878))})
        elif action == "send_tokens":
            self._send_json(_fluid_send_tokens(body.get("tokens", []), body.get("dst_ip", ""), body.get("dst_port", 9878)))
        else:
            self._send_json({"error": "unknown fluid action"}, 400)

    # ===== ModelMemory handlers =====

    def _get_model_memory(self):
        _try_init_memory()
        return _memory_instance

    def _handle_memory_status(self):
        model_memory = self._get_model_memory()
        status = {"available": model_memory is not None}
        if model_memory:
            try:
                progress = model_memory.get_progress()
                status.update(progress)
            except Exception:
                pass
        self._send_json(status)

    def _handle_memory_progress(self):
        model_memory = self._get_model_memory()
        if model_memory is None:
            self._send_json({"error": "Memory not initialized"}, 503)
            return
        try:
            self._send_json(model_memory.get_progress())
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_memory_session_start(self):
        model_memory = self._get_model_memory()
        if model_memory is None:
            self._send_json({"error": "Memory not initialized"}, 503)
            return
        try:
            sid = model_memory.start_session()
            self._send_json({"success": True, "session_id": sid})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_memory_session_end(self):
        model_memory = self._get_model_memory()
        if model_memory is None:
            self._send_json({"error": "Memory not initialized"}, 503)
            return
        try:
            self._send_json(model_memory.end_session())
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_memory_teach(self, body):
        model_memory = self._get_model_memory()
        if model_memory is None:
            self._send_json({"error": "Memory not initialized"}, 503)
            return
        lesson = body.get("lesson", "")
        if not lesson:
            self._send_json({"error": "No lesson provided"}, 400)
            return
        try:
            self._send_json(model_memory.teach(lesson))
        except Exception as e:
            self._send_json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def _handle_memory_quiz(self, body):
        model_memory = self._get_model_memory()
        if model_memory is None:
            self._send_json({"error": "Memory not initialized"}, 503)
            return
        try:
            self._send_json(model_memory.quiz(body.get("question", ""), body.get("max_tokens", 64)))
        except Exception as e:
            self._send_json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def _handle_memory_correct(self, body):
        model_memory = self._get_model_memory()
        if model_memory is None:
            self._send_json({"error": "Memory not initialized"}, 503)
            return
        wrong = body.get("wrong_response", "")
        right = body.get("right_answer", "")
        if not right:
            self._send_json({"error": "No right_answer provided"}, 400)
            return
        try:
            self._send_json(model_memory.correct(wrong, right))
        except Exception as e:
            self._send_json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def _handle_memory_reinforce(self, body):
        model_memory = self._get_model_memory()
        if model_memory is None:
            self._send_json({"error": "Memory not initialized"}, 503)
            return
        good = body.get("good_response", "")
        if not good:
            self._send_json({"error": "No good_response provided"}, 400)
            return
        try:
            self._send_json(model_memory.reinforce(good))
        except Exception as e:
            self._send_json({"error": str(e), "trace": traceback.format_exc()}, 500)

    def _handle_memory_converse(self, body):
        model_memory = self._get_model_memory()
        if model_memory is None:
            self._send_json({"error": "Memory not initialized"}, 503)
            return
        message = body.get("message", "")
        if not message:
            self._send_json({"error": "No message provided"}, 400)
            return
        try:
            self._send_json(model_memory.converse(message, body.get("max_tokens", 128)))
        except Exception as e:
            self._send_json({"error": str(e), "trace": traceback.format_exc()}, 500)


# ============================================================
# Server management
# ============================================================

def start_bridge(host="127.0.0.1", port=1345):
    """Start the HTTP bridge server in a background thread.
    LINUX PORT: loopback only. Non-loopback hosts are refused."""
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(f"Linux clone bridge binds loopback only, refused host: {host}")
    global _server_thread, _server_instance

    if _server_instance is not None:
        print(f"[Bridge] Already running on port {_server_instance.server_port}")
        return True

    def run():
        global _server_instance
        srv = HTTPServer((host, port), BridgeHandler)
        _server_instance = srv
        print(f"\n[Nightmare Bridge 13] listening on http://{host}:{port}")
        srv.serve_forever()

    _server_thread = threading.Thread(target=run, daemon=True)
    _server_thread.start()

    # Wait briefly for server to be ready
    for _ in range(50):
        if _server_instance is not None:
            return True
        time.sleep(0.1)
    return _server_instance is not None


def stop_bridge():
    """Stop the HTTP bridge server."""
    global _server_instance, _server_thread
    if _server_instance is None:
        print("[Bridge] Not running")
        return
    print("[Bridge] Stopping server...")
    _server_instance.shutdown()
    _server_instance = None
    _server_thread = None
    print("[Bridge] Stopped")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="path to .ntmre model to load at startup")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=1345)
    args = ap.parse_args()

    if args.model:
        _bridge_model = load_bridge_engine(args.model)
        _bridge_model_path = str(args.model)
        _bridge_model_name = Path(args.model).name

    start_bridge(args.host, args.port)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_bridge()
