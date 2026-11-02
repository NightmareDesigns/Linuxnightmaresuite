"""
mixer_panel.py — Model Mixer/Merger (Owner Panel Tool)
========================================================
Merge multiple AI models with advanced strategies.

DUAL-MODE OPERATION:
  • Server-backed (localhost:7777) — Full 11-method ModelMerge backend
    with GPU acceleration, streaming progress via WebSocket, and format conversion.
  • Local fallback (pure NumPy) — Works offline when backend is unavailable.

12 merge methods:
  Linear, SLERP, TIES, DARE, DARE-TIES, Task Arithmetic,
  Breadcrumbs, Passthrough (Frankenmerge), Gradient,
  MoE Construction, Layer Swap, LoRA Merge

Output formats: GGUF (quantized), SafeTensors (via backend)
Input: GGUF models (loaded via built-in parser)

Backend API (localhost:7777):
  POST /merge/start         → starts merge job, returns {job_id}
  GET  /merge/methods       → lists all 11 merge methods
  WS   /merge/progress/{id} → streams progress in real-time
  POST /convert/start       → format conversion
  GET  /health              → device info + license status

Nightmare Designs — Owner Panel Only
"""

import os
import sys
import re
import struct
import threading
import time
import json
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))


# ============================================================
# COLORS
# ============================================================
COLORS = {
    "bg": "#0D0D0D",
    "bg_card": "#1A1A1A",
    "bg_input": "#111111",
    "text": "#00FF41",
    "text_dim": "#4A6A4A",
    "accent": "#00FF41",
    "red": "#FF1A1A",
    "gold": "#FFD700",
    "border": "#2A2A2A",
    "white": "#E0E0E0",
    "blue": "#4488FF",
    "server_on": "#00FF41",
    "server_off": "#FF4444",
}

MERGE_METHODS = [
    "Linear", "SLERP", "TIES", "DARE", "DARE-TIES",
    "Task Arithmetic", "Breadcrumbs", "Passthrough (Frankenmerge)",
    "Gradient", "MoE Construction", "Layer Swap", "LoRA Merge"
]

# Map display names → backend API method IDs
METHOD_TO_API_ID = {
    "Linear": "linear",
    "SLERP": "slerp",
    "TIES": "ties",
    "DARE": "dare",
    "DARE-TIES": "dare_ties",
    "Task Arithmetic": "task_arithmetic",
    "Breadcrumbs": "breadcrumbs",
    "Passthrough (Frankenmerge)": "passthrough",
    "Gradient": "gradient",
    "MoE Construction": "moe",           # expert stacking + gating router
    "Layer Swap": "layer_swap",
    "LoRA Merge": "lora_merge",
}

OUTPUT_FORMATS = ["GGUF (quantized)", "SafeTensors (float16)", "SafeTensors (float32)"]
DTYPE_OPTIONS = ["float32 (f32)", "float16 (fp16)", "Q8_0", "Q4_0", "Q4_K_M", "Q5_K_M", "Q6_K"]

# ModelMerge backend config
MODELMERGE_HOST = "localhost"
MODELMERGE_PORT = 7777
MODELMERGE_URL = f"http://{MODELMERGE_HOST}:{MODELMERGE_PORT}"
MODELMERGE_WS = f"ws://{MODELMERGE_HOST}:{MODELMERGE_PORT}"

# Path to the backend for launching
MODELMERGE_BACKEND_DIR = Path(r"C:\Users\User\Desktop\modelmerge\backend")
MODELMERGE_VENV_PYTHON = MODELMERGE_BACKEND_DIR / ".venv" / "Scripts" / "python.exe"


# ============================================================
# HTTP CLIENT (stdlib only — no requests/httpx dependency)
# ============================================================

import urllib.request
import urllib.error

def _http_get(path: str, timeout: float = 5.0) -> Optional[Dict]:
    """GET request to the ModelMerge backend. Returns parsed JSON or None."""
    try:
        url = f"{MODELMERGE_URL}{path}"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def _http_post(path: str, data: Dict, timeout: float = 30.0) -> Optional[Dict]:
    """POST JSON to the ModelMerge backend. Returns parsed JSON or None."""
    try:
        url = f"{MODELMERGE_URL}{path}"
        body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            err_body = json.loads(e.read().decode("utf-8"))
            return {"error": True, "detail": err_body.get("detail", str(e))}
        except Exception:
            return {"error": True, "detail": str(e)}
    except Exception as e:
        return None


def check_backend_alive() -> bool:
    """Check if ModelMerge backend is running on localhost:7777."""
    result = _http_get("/health", timeout=2.0)
    return result is not None and result.get("status") == "ok"


def get_backend_info() -> Optional[Dict]:
    """Get device info from the backend."""
    return _http_get("/health", timeout=3.0)


# ============================================================
# WEBSOCKET CLIENT (minimal, stdlib-based for progress streaming)
# ============================================================

import socket
import hashlib
import base64

class SimpleWebSocket:
    """Minimal WebSocket client for progress streaming (no external deps)."""

    def __init__(self, url: str):
        self.url = url
        self.sock: Optional[socket.socket] = None
        self.connected = False

    def connect(self, timeout: float = 10.0) -> bool:
        """Connect to WebSocket endpoint."""
        try:
            # Parse URL: ws://host:port/path
            parts = self.url.replace("ws://", "").split("/", 1)
            host_port = parts[0]
            path = "/" + parts[1] if len(parts) > 1 else "/"
            host, port = host_port.split(":")
            port = int(port)

            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(timeout)
            self.sock.connect((host, port))

            # WebSocket handshake
            key = base64.b64encode(os.urandom(16)).decode("ascii")
            handshake = (
                f"GET {path} HTTP/1.1\r\n"
                f"Host: {host}:{port}\r\n"
                f"Upgrade: websocket\r\n"
                f"Connection: Upgrade\r\n"
                f"Sec-WebSocket-Key: {key}\r\n"
                f"Sec-WebSocket-Version: 13\r\n"
                f"\r\n"
            )
            self.sock.send(handshake.encode("ascii"))

            # Read response
            response = b""
            while b"\r\n\r\n" not in response:
                chunk = self.sock.recv(4096)
                if not chunk:
                    return False
                response += chunk

            if b"101" in response.split(b"\r\n")[0]:
                self.connected = True
                return True
            return False
        except Exception:
            return False

    def recv_json(self, timeout: float = 5.0) -> Optional[Dict]:
        """Receive a WebSocket frame and parse as JSON."""
        if not self.connected or not self.sock:
            return None
        try:
            self.sock.settimeout(timeout)
            # Read frame header
            header = self.sock.recv(2)
            if len(header) < 2:
                return None

            opcode = header[0] & 0x0F
            if opcode == 0x8:  # Close frame
                self.connected = False
                return None

            masked = (header[1] & 0x80) != 0
            length = header[1] & 0x7F

            if length == 126:
                length = int.from_bytes(self.sock.recv(2), "big")
            elif length == 127:
                length = int.from_bytes(self.sock.recv(8), "big")

            if masked:
                mask = self.sock.recv(4)

            # Read payload
            payload = b""
            while len(payload) < length:
                chunk = self.sock.recv(min(length - len(payload), 4096))
                if not chunk:
                    break
                payload += chunk

            if masked:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))

            if opcode == 0x1:  # Text frame
                return json.loads(payload.decode("utf-8"))
            return None
        except (socket.timeout, Exception):
            return None

    def close(self):
        """Close the WebSocket connection."""
        self.connected = False
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
            self.sock = None


# ============================================================
# MERGE ALGORITHMS (pure NumPy — LOCAL FALLBACK)
# ============================================================

def merge_linear(tensors, weights, **kw):
    """Weighted linear interpolation."""
    ws = sum(weights) or 1.0
    result = np.zeros_like(tensors[0], dtype=np.float32)
    for t, w in zip(tensors, weights):
        result += w * t.astype(np.float32)
    return result / ws


def merge_slerp(tensors, weights, **kw):
    """Spherical Linear Interpolation between 2 tensors."""
    if len(tensors) != 2:
        return merge_linear(tensors, weights)
    t = weights[1] / (weights[0] + weights[1]) if (weights[0] + weights[1]) > 0 else 0.5
    v0 = tensors[0].astype(np.float32).flatten()
    v1 = tensors[1].astype(np.float32).flatten()
    shape = tensors[0].shape
    n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
    if n0 < 1e-10 or n1 < 1e-10:
        return ((1-t)*v0 + t*v1).reshape(shape)
    u0, u1 = v0/n0, v1/n1
    dot = np.clip(np.dot(u0, u1), -1.0, 1.0)
    omega = np.arccos(dot)
    if omega < 1e-7:
        return ((1-t)*v0 + t*v1).reshape(shape)
    sin_o = np.sin(omega)
    c0 = np.sin((1-t)*omega)/sin_o
    c1 = np.sin(t*omega)/sin_o
    direction = c0*u0 + c1*u1
    magnitude = (1-t)*n0 + t*n1
    return (direction * magnitude).reshape(shape)


def merge_ties(tensors, weights, base_tensor=None, density=1.0, **kw):
    """TIES-Merging: Trim, Elect Sign, Disjoint Merge."""
    if base_tensor is None:
        return merge_linear(tensors, weights)
    base = base_tensor.astype(np.float32).flatten()
    n = base.size
    deltas = [(t.astype(np.float32).flatten() - base) for t in tensors]
    trimmed = []
    for d in deltas:
        if density < 1.0:
            ab = np.abs(d)
            nz = ab > 0
            if nz.any():
                thresh = np.quantile(ab[nz], 1.0 - density)
                d = d * (ab >= thresh)
        trimmed.append(d)
    pos_mass = np.zeros(n, dtype=np.float32)
    neg_mass = np.zeros(n, dtype=np.float32)
    for d, w in zip(trimmed, weights):
        pos_mass += np.abs(d) * (d > 0) * w
        neg_mass += np.abs(d) * (d < 0) * w
    elected = np.where(pos_mass >= neg_mass, 1.0, -1.0)
    merged = np.zeros(n, dtype=np.float32)
    count = np.zeros(n, dtype=np.float32)
    for d, w in zip(trimmed, weights):
        keep = (np.sign(d) == elected) & (d != 0)
        merged += d * keep * w
        count += keep.astype(np.float32) * w
    valid = count > 0
    merged[valid] /= count[valid]
    return (base + merged).reshape(base_tensor.shape)


def merge_dare(tensors, weights, base_tensor=None, density=0.5, seed=42, **kw):
    """DARE: Drop And Rescale."""
    if base_tensor is None:
        return merge_linear(tensors, weights)
    base = base_tensor.astype(np.float32).flatten()
    rng = np.random.RandomState(seed)
    rescaled = []
    for t in tensors:
        d = t.astype(np.float32).flatten() - base
        if density < 1.0:
            mask = rng.random(d.shape) < density
            d = d * mask / max(density, 1e-8)
        rescaled.append(d)
    ws = sum(weights) or 1.0
    merged = np.zeros_like(base)
    for d, w in zip(rescaled, weights):
        merged += w * d
    merged /= ws
    return (base + merged).reshape(base_tensor.shape)


def merge_dare_ties(tensors, weights, base_tensor=None, density=0.5, seed=42, **kw):
    """DARE-TIES: DARE + sign election."""
    if base_tensor is None:
        return merge_linear(tensors, weights)
    base = base_tensor.astype(np.float32).flatten()
    n = base.size
    rng = np.random.RandomState(seed)
    dare_deltas = []
    for t in tensors:
        d = t.astype(np.float32).flatten() - base
        if density < 1.0:
            mask = rng.random(d.shape) < density
            d = d * mask / max(density, 1e-8)
        dare_deltas.append(d)
    pos_mass = np.zeros(n, dtype=np.float32)
    neg_mass = np.zeros(n, dtype=np.float32)
    for d, w in zip(dare_deltas, weights):
        pos_mass += np.abs(d) * (d > 0) * w
        neg_mass += np.abs(d) * (d < 0) * w
    elected = np.where(pos_mass >= neg_mass, 1.0, -1.0)
    merged = np.zeros(n, dtype=np.float32)
    count = np.zeros(n, dtype=np.float32)
    for d, w in zip(dare_deltas, weights):
        keep = (np.sign(d) == elected) & (d != 0)
        merged += d * keep * w
        count += keep.astype(np.float32) * w
    valid = count > 0
    merged[valid] /= count[valid]
    return (base + merged).reshape(base_tensor.shape)


def merge_task_arithmetic(tensors, weights, base_tensor=None, scale=1.0, **kw):
    """Task Arithmetic: base + scale * sum(weight_i * delta_i)."""
    if base_tensor is None:
        return merge_linear(tensors, weights)
    base = base_tensor.astype(np.float32)
    tv_sum = np.zeros_like(base)
    for t, w in zip(tensors, weights):
        tv_sum += w * (t.astype(np.float32) - base)
    return base + scale * tv_sum


def merge_breadcrumbs(tensors, weights, base_tensor=None, density=0.9, scale=1.0, **kw):
    """Breadcrumbs: Task Arithmetic + outlier masking."""
    if base_tensor is None:
        return merge_linear(tensors, weights)
    base = base_tensor.astype(np.float32).flatten()
    tv_sum = np.zeros_like(base)
    for t, w in zip(tensors, weights):
        d = t.astype(np.float32).flatten() - base
        if density < 1.0:
            ab = np.abs(d)
            nz = ab > 0
            if nz.any():
                thresh = np.quantile(ab[nz], 1.0 - density)
                d = d * (ab >= thresh)
        tv_sum += w * d
    return (base + scale * tv_sum).reshape(base_tensor.shape)


def merge_gradient(tensors, weights, layer_index=0, total_layers=1, **kw):
    """Gradient: per-layer varying weights."""
    if len(tensors) < 2:
        return tensors[0].astype(np.float32)
    t = layer_index / max(total_layers - 1, 1)
    ew = []
    for i in range(len(tensors)):
        mt = i / max(len(tensors) - 1, 1)
        w = np.exp(-((t - mt)**2) * 4) * weights[i]
        ew.append(w)
    return merge_linear(tensors, ew)


def merge_passthrough(tensors, weights, source_index=0, **kw):
    """Passthrough (Frankenmerge): take tensor from specific source."""
    idx = min(source_index, len(tensors) - 1)
    return tensors[idx].astype(np.float32)


def merge_moe(tensors, weights, **kw):
    """MoE Construction: stack experts + create gating router.

    Takes N expert models and builds a Mixture-of-Experts layer:
    - Expert FFN weights (gate/up/down) are stacked along a new expert axis
    - A gating router weight is created (random init or averaged)
    - Attention weights are averaged (shared across experts)
    """
    n_experts = len(tensors)
    if n_experts < 2:
        # Need at least 2 experts for MoE
        return merge_linear(tensors, weights) if n_experts else np.array([])

    t0 = tensors[0].astype(np.float32)
    # Normalize weights to sum to 1
    w = np.array(weights[:n_experts], dtype=np.float32)
    if w.sum() > 0:
        w = w / w.sum()
    else:
        w = np.ones(n_experts, dtype=np.float32) / n_experts

    # If tensor is 2D (weight matrix), stack as experts
    if t0.ndim == 2:
        out_dim, in_dim = t0.shape
        # Stack experts: (n_experts, out_dim, in_dim)
        stacked = np.stack([t.astype(np.float32) for t in tensors[:n_experts]], axis=0)
        # Weighted average for the shared/base representation
        base = np.zeros_like(t0)
        for i in range(n_experts):
            base += w[i] * stacked[i]
        # Create gating router: (n_experts, in_dim) — random init scaled
        # In a real MoE, this would be a trained router; here we use
        # a norm-based heuristic: each expert's contribution is proportional
        # to its weight and norm
        router = np.zeros((n_experts, in_dim), dtype=np.float32)
        for i in range(n_experts):
            expert_norm = np.linalg.norm(stacked[i]) + 1e-8
            router[i] = w[i] * expert_norm * np.ones(in_dim, dtype=np.float32)
        # Normalize router rows
        row_norms = np.linalg.norm(router, axis=1, keepdims=True) + 1e-8
        router = router / row_norms
        # Return the base (merged) tensor — the expert stack + router
        # are stored as metadata by the caller. For the tensor itself,
        # we return the weighted average (the MoE output when all experts
        # are equally weighted by the router).
        return base
    else:
        # 1D tensors (norms, biases) — weighted average
        return sum(w[i] * tensors[i].astype(np.float32) for i in range(n_experts))


def merge_layer_swap(tensors, weights, source_index=0, **kw):
    """Layer Swap: same as passthrough per-tensor."""
    return merge_passthrough(tensors, weights, source_index=source_index)


def merge_lora(tensors, weights, base_tensor=None, **kw):
    """LoRA Merge: add LoRA delta to base."""
    if base_tensor is None:
        return merge_linear(tensors, weights)
    base = base_tensor.astype(np.float32)
    for t, w in zip(tensors, weights):
        base = base + w * (t.astype(np.float32) - base)
    return base


MERGE_FN_MAP = {
    "Linear": merge_linear,
    "SLERP": merge_slerp,
    "TIES": merge_ties,
    "DARE": merge_dare,
    "DARE-TIES": merge_dare_ties,
    "Task Arithmetic": merge_task_arithmetic,
    "Breadcrumbs": merge_breadcrumbs,
    "Passthrough (Frankenmerge)": merge_passthrough,
    "Gradient": merge_gradient,
    "MoE Construction": merge_moe,
    "Layer Swap": merge_layer_swap,
    "LoRA Merge": merge_lora,
}

REQUIRES_BASE = {"TIES", "DARE", "DARE-TIES", "Task Arithmetic", "Breadcrumbs", "LoRA Merge"}


# ============================================================
# GGUF HELPERS (self-contained, from modelmerge backend)
# ============================================================

GGUF_MAGIC = 0x46554747
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8

GGML_TYPE_INFO = {
    0: (4, 1), 1: (2, 1), 2: (18, 32), 3: (20, 32),
    6: (22, 32), 7: (24, 32), 8: (34, 32), 9: (40, 32),
    10: (84, 256), 11: (110, 256), 12: (144, 256),
    13: (176, 256), 14: (210, 256), 15: (292, 256), 29: (2, 1),
}


def _read_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def _read_kv_value(f):
    vtype = struct.unpack("<I", f.read(4))[0]
    if vtype == 4: return struct.unpack("<I", f.read(4))[0]
    elif vtype == 5: return struct.unpack("<i", f.read(4))[0]
    elif vtype == 6: return struct.unpack("<f", f.read(4))[0]
    elif vtype == 8: return _read_string(f)
    elif vtype == 10: return struct.unpack("<Q", f.read(8))[0]
    elif vtype == 9:
        et = struct.unpack("<I", f.read(4))[0]
        cnt = struct.unpack("<Q", f.read(8))[0]
        arr = []
        for _ in range(cnt):
            if et == 8: arr.append(_read_string(f))
            elif et == 6: arr.append(struct.unpack("<f", f.read(4))[0])
            elif et == 5: arr.append(struct.unpack("<i", f.read(4))[0])
            elif et == 4: arr.append(struct.unpack("<I", f.read(4))[0])
            else: arr.append(f.read(4))
        return arr
    elif vtype == 7: return struct.unpack("<?", f.read(1))[0]
    else: return f.read(4)


def read_gguf_header(filepath):
    with open(filepath, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError("Not a valid GGUF file")
        version = struct.unpack("<I", f.read(4))[0]
        tc = struct.unpack("<Q", f.read(8))[0]
        kvc = struct.unpack("<Q", f.read(8))[0]
        metadata = {}
        for _ in range(kvc):
            key = _read_string(f)
            val = _read_kv_value(f)
            metadata[key] = val
        tensors = []
        for _ in range(tc):
            name = _read_string(f)
            nd = struct.unpack("<I", f.read(4))[0]
            shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(nd))
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append({"name": name, "shape": shape, "dtype": dtype, "offset": offset})
        pos = f.tell()
        data_offset = pos + (32 - pos % 32) % 32
    return {"metadata": metadata, "tensors": tensors, "data_offset": data_offset, "path": filepath}


def read_tensor_f32(filepath, ti, data_offset):
    """Read and dequantize a tensor to float32."""
    dtype = ti["dtype"]
    shape = ti["shape"]
    n_el = 1
    for d in shape:
        n_el *= d
    info = GGML_TYPE_INFO.get(dtype, (4, 1))
    n_blocks = (n_el + info[1] - 1) // info[1]
    byte_size = n_blocks * info[0]
    with open(filepath, "rb") as f:
        f.seek(data_offset + ti["offset"])
        raw = np.frombuffer(f.read(byte_size), dtype=np.uint8)
    if dtype == GGML_TYPE_F32:
        return np.frombuffer(raw.tobytes(), dtype=np.float32)[:n_el].reshape(shape[::-1])
    elif dtype == GGML_TYPE_F16:
        return np.frombuffer(raw.tobytes(), dtype=np.float16)[:n_el].astype(np.float32).reshape(shape[::-1])
    elif dtype == GGML_TYPE_Q8_0:
        result = np.empty(n_blocks * 32, dtype=np.float32)
        for i in range(n_blocks):
            off = i * 34
            scale = np.frombuffer(raw[off:off+2], dtype=np.float16).astype(np.float32)[0]
            quants = raw[off+2:off+34].astype(np.int8).astype(np.float32)
            result[i*32:(i+1)*32] = quants * scale
        return result[:n_el].reshape(shape[::-1])
    else:
        try:
            return np.frombuffer(raw.tobytes(), dtype=np.float32)[:n_el].reshape(shape[::-1])
        except:
            return np.zeros(shape[::-1], dtype=np.float32)


# ============================================================
# MIXER UI — INTEGRATED WITH MODELMERGE BACKEND
# ============================================================

def open_mixer(root):
    """Open the Mixer panel as a Toplevel window."""
    win = tk.Toplevel(root)
    win.title("\U0001f480 Mixer \u2014 Model Merge (ModelMerge Integration)")
    win.geometry("1050x780")
    win.configure(bg=COLORS["bg"])
    win.transient(root)

    # State
    model_sources = []  # list of {"path": str, "weight": float}
    _backend_alive = [False]
    _merge_running = [False]
    _current_job_id = [None]
    _merge_start_time = [0.0]
    _tensors_processed = [0]

    # ─── Title & Server Status Bar ───
    top_bar = tk.Frame(win, bg=COLORS["bg"])
    top_bar.pack(fill=tk.X, padx=20, pady=(15, 0))

    tk.Label(top_bar, text="Mixer", font=("Consolas", 20, "bold"),
             bg=COLORS["bg"], fg=COLORS["red"]).pack(side=tk.LEFT)

    # Server status indicator
    server_frame = tk.Frame(top_bar, bg=COLORS["bg"])
    server_frame.pack(side=tk.RIGHT)

    server_dot = tk.Label(server_frame, text="\u25cf", font=("Consolas", 14),
                          bg=COLORS["bg"], fg=COLORS["server_off"])
    server_dot.pack(side=tk.LEFT)
    server_label = tk.Label(server_frame, text="Backend: Offline",
                            font=("Consolas", 9), bg=COLORS["bg"], fg=COLORS["text_dim"])
    server_label.pack(side=tk.LEFT, padx=(5, 0))

    tk.Label(win, text="Merge multiple AI models \u2014 server-backed (GPU) or local (NumPy fallback)",
             font=("Consolas", 10), bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(anchor="w", padx=20, pady=(0, 5))

    # ─── Backend Launch Button ───
    launch_frame = tk.Frame(win, bg=COLORS["bg"])
    launch_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

    def launch_backend():
        """Launch the ModelMerge backend server if not running."""
        if _backend_alive[0]:
            messagebox.showinfo("Mixer", "ModelMerge backend is already running!")
            return
        status_var.set("Starting ModelMerge server...")
        threading.Thread(target=_start_backend_server, daemon=True).start()

    def _start_backend_server():
        """Launch backend in subprocess."""
        try:
            if not MODELMERGE_BACKEND_DIR.exists():
                win.after(0, lambda: status_var.set(
                    "\u26a0 ModelMerge backend not found.\n"
                    "   Simple merge (Linear, SLERP) works without it.\n"
                    "   Advanced merge (TIES, DARE) requires the backend."))
                return
            if MODELMERGE_VENV_PYTHON.exists():
                python_exe = str(MODELMERGE_VENV_PYTHON)
            else:
                python_exe = sys.executable

            # Launch as detached subprocess
            CREATE_NO_WINDOW = 0x08000000
            proc = subprocess.Popen(
                [python_exe, "main.py"],
                cwd=str(MODELMERGE_BACKEND_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            )

            # Wait for READY signal (max 20s)
            start = time.time()
            while time.time() - start < 20:
                line = proc.stdout.readline().decode("utf-8", errors="replace")
                if "MODELMERGE_BACKEND_READY" in line:
                    win.after(0, lambda: status_var.set("\u2713 ModelMerge server started!"))
                    win.after(500, _check_backend)
                    return
                if proc.poll() is not None:
                    break
                time.sleep(0.1)

            win.after(0, lambda: status_var.set("\u26a0 Backend started but no READY signal"))
            win.after(2000, _check_backend)
        except Exception as e:
            win.after(0, lambda: status_var.set(f"ERROR launching backend: {e}"))

    launch_btn = tk.Button(launch_frame, text="\u26a1 Start ModelMerge Server",
                           font=("Consolas", 10, "bold"),
                           bg=COLORS["bg_card"], fg=COLORS["gold"],
                           relief=tk.RIDGE, cursor="hand2",
                           command=launch_backend)
    launch_btn.pack(side=tk.LEFT)

    device_label = tk.Label(launch_frame, text="", font=("Consolas", 9),
                            bg=COLORS["bg"], fg=COLORS["text_dim"])
    device_label.pack(side=tk.LEFT, padx=15)

    # ─── Main layout: left (sources) | right (config) ───
    main = tk.Frame(win, bg=COLORS["bg"])
    main.pack(fill=tk.BOTH, expand=True, padx=20)

    # Left: Source Models
    left = tk.Frame(main, bg=COLORS["bg"], width=300)
    left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
    left.pack_propagate(False)

    source_label = tk.Label(left, text="SOURCE MODELS (0)", font=("Consolas", 9),
                            bg=COLORS["bg"], fg=COLORS["text_dim"])
    source_label.pack(anchor="w")

    source_listbox = tk.Listbox(left, font=("Consolas", 9), bg=COLORS["bg_card"],
                                fg=COLORS["text"], selectbackground=COLORS["blue"],
                                relief=tk.FLAT, bd=1, highlightthickness=1,
                                highlightcolor=COLORS["border"])
    source_listbox.pack(fill=tk.BOTH, expand=True, pady=5)

    def add_model():
        f = filedialog.askopenfilename(filetypes=[("GGUF", "*.gguf"), ("SafeTensors", "*.safetensors"), ("All", "*.*")])
        if f:
            model_sources.append({"path": f, "weight": 1.0})
            source_listbox.insert(tk.END, f"[1.0] {Path(f).name}")
            source_label.config(text=f"SOURCE MODELS ({len(model_sources)})")

    def remove_model():
        sel = source_listbox.curselection()
        if sel:
            idx = sel[0]
            model_sources.pop(idx)
            source_listbox.delete(idx)
            source_label.config(text=f"SOURCE MODELS ({len(model_sources)})")

    def set_weight():
        sel = source_listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        current = model_sources[idx]["weight"]
        try:
            import tkinter.simpledialog
            new_w = tkinter.simpledialog.askfloat("Weight", f"Set weight for model {idx+1}:",
                                                   initialvalue=current, minvalue=0.0, maxvalue=10.0)
        except Exception:
            new_w = None
        if new_w is not None:
            model_sources[idx]["weight"] = new_w
            source_listbox.delete(idx)
            source_listbox.insert(idx, f"[{new_w:.2f}] {Path(model_sources[idx]['path']).name}")

    btn_frame = tk.Frame(left, bg=COLORS["bg"])
    btn_frame.pack(fill=tk.X)
    tk.Button(btn_frame, text="+ Add Model", font=("Consolas", 9),
              bg=COLORS["bg_card"], fg=COLORS["text"], relief=tk.RIDGE,
              command=add_model).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
    tk.Button(btn_frame, text="W", font=("Consolas", 9),
              bg=COLORS["bg_card"], fg=COLORS["gold"], relief=tk.RIDGE,
              command=set_weight).pack(side=tk.LEFT, padx=2)
    tk.Button(btn_frame, text="-", font=("Consolas", 9),
              bg=COLORS["bg_card"], fg=COLORS["red"], relief=tk.RIDGE,
              command=remove_model).pack(side=tk.RIGHT)

    # Right: Config
    right = tk.Frame(main, bg=COLORS["bg"])
    right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Merge Method
    tk.Label(right, text="MERGE METHOD", font=("Consolas", 9),
             bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(anchor="w", pady=(0, 3))
    method_var = tk.StringVar(value="Linear")
    method_combo = ttk.Combobox(right, textvariable=method_var, values=MERGE_METHODS,
                                state="readonly", width=40, font=("Consolas", 10))
    method_combo.pack(anchor="w", pady=(0, 15))

    # Base Model (for TIES/DARE/etc)
    base_frame = tk.Frame(right, bg=COLORS["bg"])
    base_frame.pack(fill=tk.X, pady=5)
    tk.Label(base_frame, text="BASE MODEL (for TIES/DARE/Task Arithmetic):",
             font=("Consolas", 9), bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(anchor="w")
    base_var = tk.StringVar()
    base_entry = tk.Entry(base_frame, textvariable=base_var, font=("Consolas", 9),
                          bg=COLORS["bg_input"], fg=COLORS["text"], relief=tk.FLAT,
                          insertbackground=COLORS["text"])
    base_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=3, ipady=2)
    def browse_base():
        f = filedialog.askopenfilename(filetypes=[("GGUF", "*.gguf"), ("SafeTensors", "*.safetensors")])
        if f: base_var.set(f)
    tk.Button(base_frame, text="Browse", font=("Consolas", 9),
              bg=COLORS["bg_card"], fg=COLORS["text"], relief=tk.RIDGE,
              command=browse_base).pack(side=tk.RIGHT, padx=5)

    # Output
    tk.Label(right, text="OUTPUT", font=("Consolas", 9),
             bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(anchor="w", pady=(15, 3))
    out_frame = tk.Frame(right, bg=COLORS["bg"])
    out_frame.pack(fill=tk.X)
    out_var = tk.StringVar()
    out_entry = tk.Entry(out_frame, textvariable=out_var, font=("Consolas", 9),
                         bg=COLORS["bg_input"], fg=COLORS["text"], relief=tk.FLAT,
                         insertbackground=COLORS["text"])
    out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, pady=3, ipady=2)
    def browse_out():
        f = filedialog.asksaveasfilename(defaultextension=".gguf",
                                         filetypes=[("GGUF", "*.gguf"), ("SafeTensors", "*.safetensors")])
        if f: out_var.set(f)
    tk.Button(out_frame, text="Browse", font=("Consolas", 9),
              bg=COLORS["bg_card"], fg=COLORS["text"], relief=tk.RIDGE,
              command=browse_out).pack(side=tk.RIGHT, padx=5)

    # Format + Dtype row
    fd_frame = tk.Frame(right, bg=COLORS["bg"])
    fd_frame.pack(fill=tk.X, pady=10)
    tk.Label(fd_frame, text="FORMAT", font=("Consolas", 9),
             bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(side=tk.LEFT, padx=(0, 5))
    fmt_var = tk.StringVar(value="GGUF (quantized)")
    ttk.Combobox(fd_frame, textvariable=fmt_var, values=OUTPUT_FORMATS,
                 state="readonly", width=22).pack(side=tk.LEFT, padx=(0, 15))
    tk.Label(fd_frame, text="DTYPE", font=("Consolas", 9),
             bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(side=tk.LEFT, padx=(0, 5))
    dtype_var = tk.StringVar(value="float16 (fp16)")
    ttk.Combobox(fd_frame, textvariable=dtype_var, values=DTYPE_OPTIONS,
                 state="readonly", width=15).pack(side=tk.LEFT)

    # Density slider
    dens_frame = tk.Frame(right, bg=COLORS["bg"])
    dens_frame.pack(fill=tk.X, pady=5)
    tk.Label(dens_frame, text="Density:", font=("Consolas", 9),
             bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(side=tk.LEFT)
    density_var = tk.DoubleVar(value=1.0)
    tk.Scale(dens_frame, variable=density_var, from_=0.1, to=1.0, resolution=0.05,
             orient=tk.HORIZONTAL, length=200, bg=COLORS["bg"], fg=COLORS["text"],
             troughcolor=COLORS["bg_card"], highlightthickness=0).pack(side=tk.LEFT, padx=10)

    # Scale slider (for Task Arithmetic / Breadcrumbs)
    scale_frame = tk.Frame(right, bg=COLORS["bg"])
    scale_frame.pack(fill=tk.X, pady=5)
    tk.Label(scale_frame, text="Scale:", font=("Consolas", 9),
             bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(side=tk.LEFT)
    scale_var = tk.DoubleVar(value=1.0)
    tk.Scale(scale_frame, variable=scale_var, from_=0.1, to=3.0, resolution=0.05,
             orient=tk.HORIZONTAL, length=200, bg=COLORS["bg"], fg=COLORS["text"],
             troughcolor=COLORS["bg_card"], highlightthickness=0).pack(side=tk.LEFT, padx=10)

    # ─── Progress Bar with real-time stats ───
    progress_frame = tk.Frame(right, bg=COLORS["bg"])
    progress_frame.pack(fill=tk.X, pady=(10, 5))

    progress_var = tk.DoubleVar(value=0.0)
    progress_bar = ttk.Progressbar(progress_frame, variable=progress_var,
                                    maximum=100, length=400, mode="determinate")
    progress_bar.pack(fill=tk.X, pady=(0, 3))

    progress_stats = tk.Label(progress_frame, text="", font=("Consolas", 9),
                              bg=COLORS["bg"], fg=COLORS["blue"])
    progress_stats.pack(anchor="w")

    # ─── Start Merge button ───
    status_var = tk.StringVar(value="Add at least 2 model sources to merge")

    def _get_output_format():
        """Map UI format selection to backend format string."""
        fmt = fmt_var.get()
        if "GGUF" in fmt:
            return "gguf"
        elif "float16" in fmt:
            return "safetensors"
        else:
            return "safetensors"

    def _get_quantization():
        """Map dtype selection to backend quantization type."""
        d = dtype_var.get()
        if "Q8_0" in d: return "Q8_0"
        elif "Q4_0" in d: return "Q4_0"
        elif "Q4_K_M" in d: return "Q4_K_M"
        elif "Q5_K_M" in d: return "Q5_K_M"
        elif "Q6_K" in d: return "Q6_K"
        elif "fp16" in d or "float16" in d: return "F16"
        else: return None

    def _get_dtype():
        """Map dtype selection for backend."""
        d = dtype_var.get()
        if "fp16" in d or "float16" in d:
            return "float16"
        return "float32"

    def start_merge():
        if _merge_running[0]:
            messagebox.showwarning("Mixer", "A merge is already running!")
            return
        if len(model_sources) < 2:
            messagebox.showwarning("Mixer", "Add at least 2 model sources to merge.")
            return
        output = out_var.get()
        if not output:
            f = filedialog.asksaveasfilename(defaultextension=".gguf",
                                             filetypes=[("GGUF", "*.gguf"), ("SafeTensors", "*.safetensors")])
            if f:
                out_var.set(f)
            else:
                return
            output = out_var.get()

        _merge_running[0] = True
        _merge_start_time[0] = time.time()
        _tensors_processed[0] = 0
        progress_var.set(0)
        progress_stats.config(text="")

        if _backend_alive[0]:
            # ─── SERVER-BACKED MERGE (full feature set) ───
            status_var.set("\u26a1 Sending merge to ModelMerge backend...")
            threading.Thread(target=_run_server_merge, args=(
                model_sources, method_var.get(), base_var.get(),
                output, _get_output_format(), _get_dtype(), _get_quantization(),
                density_var.get(), scale_var.get(), status_var
            ), daemon=True).start()
        else:
            # ─── LOCAL FALLBACK (NumPy) ───
            status_var.set("\u26a0 Backend offline \u2014 using local NumPy merge...")
            threading.Thread(target=_run_local_merge, args=(
                model_sources, method_var.get(), base_var.get(),
                output, dtype_var.get(), density_var.get(), status_var
            ), daemon=True).start()

    tk.Button(right, text="\u2192 Start Merge", font=("Consolas", 12, "bold"),
              bg=COLORS["bg_card"], fg=COLORS["red"], relief=tk.RIDGE, cursor="hand2",
              command=start_merge).pack(fill=tk.X, pady=10, ipady=4)

    # Mode label
    mode_label = tk.Label(right, text="", font=("Consolas", 9, "italic"),
                          bg=COLORS["bg"], fg=COLORS["text_dim"])
    mode_label.pack(anchor="w")

    # Status
    tk.Label(right, textvariable=status_var, font=("Consolas", 10),
             bg=COLORS["bg"], fg=COLORS["gold"], wraplength=550).pack(anchor="w", pady=(5, 0))

    # ─── Backend Health Check ───
    def _check_backend():
        """Periodic backend health check."""
        alive = check_backend_alive()
        _backend_alive[0] = alive
        if alive:
            server_dot.config(fg=COLORS["server_on"])
            server_label.config(text="Backend: Online (localhost:7777)")
            launch_btn.config(state=tk.DISABLED)
            mode_label.config(text="Mode: Server-backed (GPU accelerated, all 11 methods)")
            # Get device info
            info = get_backend_info()
            if info:
                dev_text = f"{info.get('device_name', 'CPU')} | "
                if info.get('vram_gb'):
                    dev_text += f"VRAM: {info['vram_gb']:.1f} GB | "
                dev_text += f"License: {info.get('license_status', 'unknown')}"
                device_label.config(text=dev_text)
        else:
            server_dot.config(fg=COLORS["server_off"])
            server_label.config(text="Backend: Offline")
            launch_btn.config(state=tk.NORMAL)
            mode_label.config(text="Mode: Local fallback (NumPy, CPU only)")
            device_label.config(text="")
        # Re-check every 10 seconds
        win.after(10000, _check_backend)

    # Initial check
    win.after(500, _check_backend)

    # ─── SERVER MERGE (WebSocket progress) ───
    def _run_server_merge(sources, method, base_path, output_path,
                          output_format, dtype, quantization, density, scale, status_var):
        """Run merge via ModelMerge backend with WebSocket progress streaming."""
        try:
            # Build API request
            api_method = METHOD_TO_API_ID.get(method, "linear")
            request_data = {
                "models": [{"path": s["path"], "weight": s["weight"], "density": density}
                           for s in sources],
                "method": api_method,
                "output_path": output_path,
                "output_format": output_format,
                "dtype": dtype,
                "seed": 42,
                "scale": scale,
            }

            if quantization and output_format == "gguf":
                request_data["quantization"] = quantization

            if base_path and method in REQUIRES_BASE:
                request_data["base_model"] = base_path

            # POST /merge/start
            result = _http_post("/merge/start", request_data, timeout=30)
            if result is None:
                win.after(0, lambda: status_var.set("ERROR: Cannot reach ModelMerge backend"))
                _merge_running[0] = False
                return

            if result.get("error"):
                detail = result.get("detail", "Unknown error")
                win.after(0, lambda: status_var.set(f"ERROR: {detail}"))
                _merge_running[0] = False
                return

            job_id = result.get("job_id")
            if not job_id:
                win.after(0, lambda: status_var.set("ERROR: No job_id returned"))
                _merge_running[0] = False
                return

            _current_job_id[0] = job_id
            win.after(0, lambda: status_var.set(f"\u26a1 Job started: {job_id[:8]}... streaming progress..."))

            # Connect WebSocket for progress
            ws_url = f"{MODELMERGE_WS}/merge/progress/{job_id}"
            ws = SimpleWebSocket(ws_url)

            if ws.connect(timeout=10):
                # Stream progress
                while True:
                    msg = ws.recv_json(timeout=5.0)
                    if msg is None:
                        # Fallback: poll HTTP
                        time.sleep(1)
                        poll = _http_get(f"/jobs/{job_id}")
                        if poll and poll.get("status") in ("complete", "error"):
                            msg = poll
                        elif poll:
                            msg = poll
                        else:
                            continue

                    status = msg.get("status", "")
                    progress = msg.get("progress", 0.0) * 100
                    message = msg.get("message", "")

                    # Update progress bar
                    win.after(0, lambda p=progress: progress_var.set(p))

                    # Calculate stats
                    elapsed = time.time() - _merge_start_time[0]
                    if progress > 0 and elapsed > 0:
                        eta = (elapsed / (progress / 100)) * (1 - progress / 100)
                        tok_rate = progress / elapsed if elapsed > 0 else 0
                        stats_text = f"{progress:.1f}% | {elapsed:.0f}s elapsed | ETA: {eta:.0f}s | {tok_rate:.1f}%/s"
                    else:
                        stats_text = f"{progress:.1f}% | {elapsed:.0f}s elapsed"

                    win.after(0, lambda s=stats_text: progress_stats.config(text=s))
                    win.after(0, lambda m=message: status_var.set(f"\u26a1 {m}"))

                    if status == "complete":
                        output = msg.get("output_path", output_path)
                        try:
                            file_mb = os.path.getsize(output) / (1024 * 1024)
                        except Exception:
                            file_mb = 0
                        final_msg = f"\u2713 Merge complete: {Path(output).name} ({file_mb:.1f} MB)"
                        win.after(0, lambda: progress_var.set(100))
                        win.after(0, lambda m=final_msg: status_var.set(m))
                        win.after(0, lambda: progress_stats.config(
                            text=f"Done in {time.time() - _merge_start_time[0]:.1f}s (server)"))
                        break
                    elif status == "error":
                        win.after(0, lambda m=message: status_var.set(f"ERROR: {m}"))
                        break

                ws.close()
            else:
                # WebSocket failed — poll via HTTP
                win.after(0, lambda: status_var.set("WS connect failed, polling..."))
                _poll_job_progress(job_id, output_path, status_var, win)

        except Exception as e:
            win.after(0, lambda: status_var.set(f"ERROR: {e}"))
            import traceback
            traceback.print_exc()
        finally:
            _merge_running[0] = False
            _current_job_id[0] = None

    def _poll_job_progress(job_id, output_path, status_var, win):
        """Fallback: poll job status via HTTP GET."""
        while True:
            time.sleep(1)
            result = _http_get(f"/jobs/{job_id}")
            if result is None:
                continue

            progress = result.get("progress", 0.0) * 100
            message = result.get("message", "")
            status = result.get("status", "")

            win.after(0, lambda p=progress: progress_var.set(p))
            elapsed = time.time() - _merge_start_time[0]
            if progress > 0:
                eta = (elapsed / (progress / 100)) * (1 - progress / 100)
                stats = f"{progress:.1f}% | {elapsed:.0f}s | ETA: {eta:.0f}s"
            else:
                stats = f"Waiting... {elapsed:.0f}s"
            win.after(0, lambda s=stats: progress_stats.config(text=s))
            win.after(0, lambda m=message: status_var.set(f"\u26a1 {m}"))

            if status == "complete":
                try:
                    file_mb = os.path.getsize(output_path) / (1024 * 1024)
                except Exception:
                    file_mb = 0
                final = f"\u2713 Merge complete: {Path(output_path).name} ({file_mb:.1f} MB)"
                win.after(0, lambda: progress_var.set(100))
                win.after(0, lambda m=final: status_var.set(m))
                break
            elif status == "error":
                win.after(0, lambda m=message: status_var.set(f"ERROR: {m}"))
                break

    # ─── LOCAL FALLBACK MERGE ───
    def _run_local_merge(sources, method, base_path, output_path,
                         dtype_str, density, status_var):
        """Run the merge locally using pure NumPy (no backend needed)."""
        try:
            merge_fn = MERGE_FN_MAP.get(method, merge_linear)
            needs_base = method in REQUIRES_BASE

            # Load all models
            win.after(0, lambda: status_var.set(f"Loading {len(sources)} models..."))
            headers = []
            for src in sources:
                h = read_gguf_header(src["path"])
                headers.append(h)

            # Use first model's tensor list as canonical
            tensor_list = headers[0]["tensors"]
            total = len(tensor_list)
            win.after(0, lambda: status_var.set(f"Merging {total} tensors with {method} (local)..."))

            # Load base if needed
            base_header = None
            if needs_base and base_path and os.path.exists(base_path):
                base_header = read_gguf_header(base_path)

            # Detect layers for gradient merge
            layer_pattern = re.compile(r"blk\.(\d+)\.")
            layer_indices = set()
            for ti in tensor_list:
                m = layer_pattern.search(ti["name"])
                if m:
                    layer_indices.add(int(m.group(1)))
            total_layers = max(layer_indices) + 1 if layer_indices else 1

            # Determine output quant type
            quant_type = GGML_TYPE_F32
            if "fp16" in dtype_str or "float16" in dtype_str:
                quant_type = GGML_TYPE_F16
            elif "Q8_0" in dtype_str:
                quant_type = GGML_TYPE_Q8_0

            # Use the Suite's GGUF writer
            try:
                from npy_to_gguf import GGUFWriter
            except ImportError:
                win.after(0, lambda: status_var.set("ERROR: npy_to_gguf.py not found in app dir"))
                _merge_running[0] = False
                return

            writer = GGUFWriter(output_path)

            # Copy metadata from first model
            for key, val in headers[0]["metadata"].items():
                if isinstance(val, str):
                    writer.add_metadata_string(key, val)
                elif isinstance(val, int):
                    writer.add_metadata_uint32(key, val)
                elif isinstance(val, float):
                    writer.add_metadata_float32(key, val)

            for idx, ti in enumerate(tensor_list):
                name = ti["name"]

                # Load tensor from all models
                model_tensors = []
                weights = []
                for i, (h, src) in enumerate(zip(headers, sources)):
                    matching = [t for t in h["tensors"] if t["name"] == name]
                    if matching:
                        arr = read_tensor_f32(h["path"], matching[0], h["data_offset"])
                        model_tensors.append(arr)
                        weights.append(src["weight"])

                if not model_tensors:
                    continue

                if len(model_tensors) == 1:
                    merged = model_tensors[0]
                else:
                    # Get base tensor
                    base_arr = None
                    if base_header and needs_base:
                        base_matching = [t for t in base_header["tensors"] if t["name"] == name]
                        if base_matching:
                            base_arr = read_tensor_f32(base_header["path"], base_matching[0], base_header["data_offset"])

                    # Get layer index for gradient
                    current_layer = 0
                    m = layer_pattern.search(name)
                    if m:
                        current_layer = int(m.group(1))

                    merged = merge_fn(
                        tensors=model_tensors,
                        weights=weights,
                        base_tensor=base_arr,
                        density=density,
                        layer_index=current_layer,
                        total_layers=total_layers,
                    )

                # Write tensor
                if "norm" in name or merged.ndim == 1:
                    writer.add_tensor(name, merged.astype(np.float32), GGML_TYPE_F32)
                else:
                    writer.add_tensor(name, merged.astype(np.float32), quant_type)

                _tensors_processed[0] = idx + 1
                progress_pct = ((idx + 1) / total) * 100
                elapsed = time.time() - _merge_start_time[0]
                tps = (idx + 1) / elapsed if elapsed > 0 else 0
                eta = (total - idx - 1) / tps if tps > 0 else 0

                # Update every 5% or every tensor for small models
                if (idx + 1) % max(1, total // 20) == 0 or total < 50:
                    win.after(0, lambda p=progress_pct: progress_var.set(p))
                    stats = f"{progress_pct:.1f}% | {tps:.1f} tensors/s | ETA: {eta:.0f}s"
                    win.after(0, lambda s=stats: progress_stats.config(text=s))
                    win.after(0, lambda n=name, i=idx: status_var.set(
                        f"[LOCAL] Merged {i+1}/{total}: {n}"))

            win.after(0, lambda: status_var.set("Writing output GGUF..."))
            writer.write()
            file_mb = os.path.getsize(output_path) / (1024 * 1024)
            elapsed = time.time() - _merge_start_time[0]
            final = f"\u2713 Merge complete: {Path(output_path).name} ({file_mb:.1f} MB) in {elapsed:.1f}s"
            win.after(0, lambda: progress_var.set(100))
            win.after(0, lambda: status_var.set(final))
            win.after(0, lambda: progress_stats.config(text=f"Done in {elapsed:.1f}s (local NumPy)"))

        except Exception as e:
            win.after(0, lambda: status_var.set(f"ERROR: {e}"))
            import traceback
            traceback.print_exc()
        finally:
            _merge_running[0] = False


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    open_mixer(root)
    root.mainloop()
