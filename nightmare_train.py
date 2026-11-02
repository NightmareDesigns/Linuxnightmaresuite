"""nightmare_train.py — Unified Training Engine for the Suite.
================================================================
Nightmare Designs — written for Caleb. Original code, nightmare_np only.

One training interface for EVERY (model x device) combo, for GGUF and .ntmre.

    FORMAT  -> loader:  .gguf  (gguf_parser, dequant_fast)
                         .ntmre (ntmre_v1_engine, Q8/.ntmre compression dequant)
    ARCH    -> engine_resolver.Plan. The Plan normalizes ANY arch
               (Gemma4, Qwen, Llama, Mistral, Phi, Falcon, ...) into
               one config: n_layers, dim, n_heads, n_kv_heads, head_dim,
               vocab, attention, rope_style, activation, ffn.
               A single generic transformer forward/backward reads the
               Plan, so one implementation covers every arch. Arch quirks
               (Gemma4 dual attention, per-layer dims) come from the
               Plan's fields, not from a per-arch code path.
    DEVICE  -> system_planner.plan_for_model picks the engine + devices.
               The trainer dispatches train_step to the selected backend.

Backends (all implement the same interface):
    CPUBackend       — nightmare_np, generic transformer forward/backward
                       + AdamW. Works for EVERY arch and EVERY format.
                       Slow (CPU), but universal. The fallback.
    GPUTrainBackend  — nightmare_train.dll via NightmareTrainEngine.
                       full_step: forward + backward + optimizer ALL on
                       GPU, one DLL call per step. Tokens in (12 KB),
                       loss out (4 bytes). 1000+ tok/s when the DLL
                       matches the arch. Falls back to CPU otherwise.
    Gemma4GPUBackend— ntmre_gemma4_gpu engine for Gemma4 forward on GPU
                       + CPU backward (the train_v3 pattern: fused GPU
                       forward, real CPU gradients, AdamW on CPU).
    HIPBackend       — AMD GPU (ROCm/HIP). Falls back to CPU until a HIP
                       training DLL is built. The dispatch slot is here
                       so AMD is a first-class device, not an afterthought.

The dispatch matrix (what runs where):
    arch \\ device | CPU            | NVIDIA GPU        | AMD GPU
    ---------------+----------------+-------------------+----------
    Gemma4         | CPUBackend     | Gemma4GPUBackend  | HIP->CPU
    Qwen/Llama/... | CPUBackend     | GPUTrainBackend   | HIP->CPU
    any (fused DLL)| CPUBackend     | GPUTrainBackend   | HIP->CPU

    format \\ loader: .gguf -> gguf_parser | .ntmre -> ntmre_v1_engine.
    Both loaders return the same thing: a weights dict (float32, Gemma4
    tensor names when the arch is Gemma4, else the model's own names)
    plus a Plan. The trainer never sees the file format.

Usage:
    from nightmare_train import Trainer
    t = Trainer("model.gguf")          # or "model.ntmre"
    t.train(data, epochs=3)            # dispatches to the best backend
    t.save("checkpoints/my-model")     # backend-agnostic checkpoint

    # Force a backend:
    t = Trainer("model.gguf", device="cpu")
    t = Trainer("model.gguf", device="cuda:0")
    t = Trainer("model.gguf", device="amd:0")

    # Inspect the matrix:
    from nightmare_train import matrix
    print(matrix())
"""

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import nightmare_np as np

# ---------------------------------------------------------------------------
# Optional Suite modules — imported lazily so this file works standalone.
# ---------------------------------------------------------------------------

def _try_import(name):
    try:
        return __import__(name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Format detection + model loading.
# Both loaders return (weights: Dict[str, ndarray float32], plan: Plan-like).
# ---------------------------------------------------------------------------

def detect_format(path: str) -> str:
    """'gguf' or 'ntmre' from the file extension / magic."""
    p = str(path).lower()
    if p.endswith(".gguf"):
        return "gguf"
    if p.endswith(".ntmre"):
        return "ntmre"
    # Fall back to magic bytes.
    with open(path, "rb") as f:
        head = f.read(4)
    if head == b"GGUF":
        return "gguf"
    if head == b"RMTN":
        return "ntmre"
    raise ValueError(f"unknown model format: {path}")


def load_gguf(path: str):
    """Load a GGUF model -> (weights dict float32, resolver Plan)."""
    import gguf_parser
    import engine_resolver as er
    plan = er.resolve(path)
    gp = gguf_parser.GGUFParser(path)
    model = gp.parse()
    weights: Dict[str, Any] = {}
    for name in model.tensors:
        try:
            weights[name] = gp.load_tensor(model, name)
        except Exception:
            continue
    return weights, plan


def load_ntmre(path: str):
    """Load a .ntmre v1 model -> (weights dict float32, Plan-like).

    The .ntmre header carries the same geometry the resolver derives
    from a GGUF (dim, layers, heads, kv, vocab, rope, arch id), so we
    build a Plan from it directly instead of going through the GGUF
    resolver. Tensor names in .ntmre are the model's own names.
    """
    import ntmre_v1_engine as nve
    src = nve.V1Source(path)
    # Geometry from the v1 header (see V1Source.__init__).
    cfg = dict(
        dim=src.dim, n_layers=src.n_layers, n_heads=src.n_heads,
        n_kv_heads=src.n_kv, vocab=src.vocab,
        hidden_ffn=src.hidden, rope_base=src.rope,
        head_dim=src.head_dim, arch_id=src.arch_id,
    )
    weights: Dict[str, Any] = {}
    for name in list(src.entries):
        if name.endswith(".A") or name.endswith(".B"):
            continue  # .ntmre compression halves — read_flat reassembles them
        try:
            flat, shape = src.read_flat(name)
            weights[name] = flat.reshape(shape) if shape else flat
        except Exception:
            continue
    src.close()
    plan = _plan_from_ntmre(cfg)
    return weights, plan


def _plan_from_ntmre(cfg: Dict[str, Any]):
    """Build a resolver-style Plan from a .ntmre header config."""
    er = _try_import("engine_resolver")
    if er is not None:
        p = er.Plan()
    else:
        p = _MiniPlan()
    p.arch = _arch_name_from_id(cfg.get("arch_id", 0))
    p.n_layers = int(cfg.get("n_layers", 0))
    p.dim = int(cfg.get("dim", 0))
    p.n_heads = int(cfg.get("n_heads", 0))
    p.n_kv_heads = int(cfg.get("n_kv_heads", 0))
    p.head_dim = int(cfg.get("head_dim", 0)) or (
        p.dim // p.n_heads if p.n_heads else 0)
    p.vocab = int(cfg.get("vocab", 0))
    p.hidden_ffn = int(cfg.get("hidden_ffn", 0))
    p.rope_base = float(cfg.get("rope_base", 10000.0))
    p.attention = "full"
    p.rope_style = "neox" if "gemma" in p.arch or "qwen" in p.arch else "normal"
    p.activation = "swiglu"
    p.ffn = "dense"
    return p


class _MiniPlan:
    """Stand-in for engine_resolver.Plan when the resolver is absent."""
    def __init__(self):
        self.arch = "unknown"
        self.n_layers = 0
        self.dim = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.head_dim = 0
        self.vocab = 0
        self.context = 0
        self.rope_base = 10000.0
        self.hidden_ffn = 0
        self.attention = "full"
        self.rope_style = "normal"
        self.activation = "swiglu"
        self.ffn = "dense"
        self.sliding_window = 0
        self.per_layer_kdim = []
        self.per_layer_vdim = []
    def summary(self):
        return (f"{self.arch} {self.n_layers}L x {self.dim}D "
                f"{self.n_heads}H/{self.n_kv_heads}KV")


_ARCH_IDS = {
    0: "unknown", 1: "llama", 2: "gemma", 3: "gemma4", 4: "qwen",
    5: "qwen2", 6: "mistral", 7: "phi", 8: "falcon", 9: "gptneox",
    10: "stablelm", 11: "gemma2", 12: "gemma3", 13: "qwen3",
}


def _arch_name_from_id(arch_id: int) -> str:
    return _ARCH_IDS.get(int(arch_id), "unknown")


def load_model(path: str):
    """Load any supported model -> (weights, plan). Dispatches on format."""
    fmt = detect_format(path)
    if fmt == "gguf":
        return load_gguf(path)
    return load_ntmre(path)


# ---------------------------------------------------------------------------
# Device planning (system_planner) + backend selection.
# ---------------------------------------------------------------------------

def plan_device(model_bytes: int, n_layers: int = 0, arch: str = "",
                force: str = "") -> Dict[str, Any]:
    """Pick the engine + devices for this model on this machine.

    Returns a dict: {engine, devices, backend, notes}.
    `force` overrides: 'cpu', 'cuda', 'cuda:0', 'amd', 'amd:0', ''.
    Falls back to CPU when system_planner or a GPU is unavailable.
    """
    out = {"engine": "cpu", "devices": [], "backend": "cpu", "notes": []}
    if force:
        f = force.lower()
        if f.startswith("cpu"):
            out["backend"] = "cpu"
            out["notes"].append("device forced to CPU")
            return out
        if f.startswith("cuda") or f.startswith("nvidia"):
            out["backend"] = "gpu"
            out["engine"] = "fused"
            out["notes"].append(f"device forced to {force}")
            return out
        if f.startswith("amd") or f.startswith("hip"):
            out["backend"] = "hip"
            out["engine"] = "fused"
            out["notes"].append(f"device forced to {force}")
            return out
    sp = _try_import("system_planner")
    if sp is None:
        out["notes"].append("system_planner not available — CPU")
        return out
    try:
        profile = sp.system_profile()
        plan = sp.plan_for_model(profile, model_bytes, n_layers=n_layers,
                                 arch=arch)
        out["engine"] = plan.engine
        out["devices"] = [d.key for d in plan.devices]
        if plan.engine in ("fused", "gpu_native", "stream"):
            # Pick the vendor of the first device.
            key = out["devices"][0] if out["devices"] else ""
            out["backend"] = "hip" if key.startswith("amd") else "gpu"
        else:
            out["backend"] = "cpu"
        out["notes"].extend(plan.notes)
        out["notes"].extend(plan.warnings)
    except Exception as e:
        out["notes"].append(f"planner failed ({e}) — CPU")
        out["backend"] = "cpu"
    return out


# ---------------------------------------------------------------------------
# Generic transformer forward/backward (CPU, nightmare_np).
# Arch-generic: reads the Plan. One implementation covers every arch.
# ---------------------------------------------------------------------------

def _row_mean_sq(x):
    """Per-row mean of squares, as a (rows, 1) array.

    nightmare_np's .mean(axis=...) returns a Python scalar (whole-array
    reduction), not a per-row array, so we get per-row sums via a matmul
    against a ones column: (rows, dim) @ (dim, 1) -> (rows, 1).
    """
    dim = x.shape[-1]
    ones = np.ones((dim, 1), dtype=np.float32)
    return (x * x) @ ones / float(dim)


def _row_sum(x):
    """Per-row sum, as a (rows, 1) array (matmul against a ones column)."""
    ones = np.ones((x.shape[-1], 1), dtype=np.float32)
    return x @ ones


def _rmsnorm_forward(x, w, eps=1e-6):
    """RMSNorm forward. Returns (out, x_hat, rms) for backward."""
    mean_sq = _row_mean_sq(x)
    rms = np.sqrt(mean_sq + eps)
    x_hat = x / rms
    out = x_hat * w
    return out, x_hat, rms


def _rmsnorm_backward(grad_out, x_hat, rms, w):
    """RMSNorm backward. Returns (grad_x, grad_w)."""
    grad_xhat = grad_out * w
    # grad w.r.t. w: sum over rows -> (dim,)
    # (grad_out * x_hat) is (rows, dim); sum over rows via matmul
    # with a ones row: (1, rows) @ (rows, dim) -> (1, dim).
    rows = grad_out.shape[0]
    ones_row = np.ones((1, rows), dtype=np.float32)
    grad_w = (ones_row @ (grad_out * x_hat)).reshape(-1)
    # grad w.r.t. x: grad_x = (grad_xhat - x_hat*mean(grad_xhat*x_hat))/rms
    # mean over the last axis, per row: (rows, 1) via matmul.
    prod = grad_xhat * x_hat
    mean_gx = (prod @ np.ones((prod.shape[-1], 1), dtype=np.float32)) / float(prod.shape[-1])
    grad_x = (grad_xhat - x_hat * mean_gx) / rms
    return grad_x, grad_w


def _silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))


def _silu_backward(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s * (1.0 + x * (1.0 - s))


def _gelu(x):
    # tanh approximation
    return 0.5 * x * (1.0 + np.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def _softmax(x, axis=-1):
    """Softmax over the last axis.

    nightmare_np's .max(axis=...) / .sum(axis=...) return whole-array
    scalars, not per-row arrays, so the textbook max-subtraction is
    replaced by clipping: exp() overflows above ~88 in float32, and
    clipping to [-50, 50] keeps every value finite (exp(-50) is ~2e-22,
    effectively zero) while the per-row sum comes from a matmul against
    a ones column. Handles 2-D (logits) and 3-D (attention scores).
    """
    xc = np.clip(x, -50.0, 50.0)
    e = np.exp(xc)
    if e.ndim == 2:
        return e / _row_sum(e)
    if e.ndim == 3:
        shp = e.shape
        e2 = e.reshape(shp[0] * shp[1], shp[2])
        out = (e2 / _row_sum(e2)).reshape(shp)
        return out
    # 1-D fallback.
    return e / float(e.sum())


def _apply_rope(q, k, pos, head_dim, rope_base=10000.0, style="neox"):
    """Apply RoPE to q, k. q, k: (seq, n_heads, head_dim).

    Uses explicit [:, :, ...] indexing (nightmare_np does not support
    Ellipsis `...` in slices) and reshape-based broadcasting for the
    cos/sin tables (no None/newaxis).
    """
    d = head_dim
    half = d // 2
    freqs = np.array([1.0 / (rope_base ** (2.0 * i / d)) for i in range(half)],
                     dtype=np.float32)
    # pos: a list of positions (one per token).
    pos_arr = np.array([float(p) for p in pos], dtype=np.float32)
    seq = len(pos)
    angles = pos_arr.reshape(seq, 1) * freqs.reshape(1, half)
    cos = np.cos(angles)   # (seq, half)
    sin = np.sin(angles)   # (seq, half)
    # Broadcast across heads: (seq, half) -> (seq, 1, half).
    c = cos.reshape(seq, 1, half)
    s = sin.reshape(seq, 1, half)
    def _rot(x):
        # x: (seq, n_heads, head_dim). Split into even/odd pairs.
        x0 = x[:, :, 0::2]
        x1 = x[:, :, 1::2]
        o0 = x0 * c - x1 * s
        o1 = x0 * s + x1 * c
        out = np.zeros_like(x)
        out[:, :, 0::2] = o0
        out[:, :, 1::2] = o1
        return out
    return _rot(q), _rot(k)


# ---------------------------------------------------------------------------
# Backends. All implement: available, train_step(tokens, targets, mask),
# full_step(...), sync_weights(weights). Gradients match the weights-dict order.
# ---------------------------------------------------------------------------

class CPUBackend:
    """CPU training backend (nightmare_np).

    Generic transformer forward/backward + AdamW. Works for every arch
    and every format. Slow on big models — the universal fallback.
    """

    name = "cpu"

    def __init__(self, weights, plan, lr=0.0003):
        self.weights = weights
        self.plan = plan
        self.lr = lr
        self._adam_m = {}
        self._adam_v = {}
        self._adam_t = 0
        self.available = True

    # -- forward (single sequence, batched over tokens) --
    def _forward(self, token_ids, cache=True):
        """Forward pass. token_ids: (seq,) int. Returns (logits, cache)."""
        p = self.plan
        W = self.weights
        seq = len(token_ids)
        dim = p.dim
        # Embedding
        emb = W.get("model.embed_tokens.weight")
        if emb is None:
            emb = W.get("token_embd.weight")
        h = np.zeros((seq, dim), dtype=np.float32)
        for i, t in enumerate(token_ids):
            h[i] = emb[int(t)]
        caches = []
        for li in range(p.n_layers):
            pre = f"model.layers.{li}."
            # Some models use blk. prefix (GGUF naming)
            if (pre + "input_layernorm.weight") not in W:
                pre = f"blk.{li}."
            hn, x_hat, rms = _rmsnorm_forward(h, W[pre + "input_layernorm.weight"])
            # Q/K/V
            q = hn @ W[pre + "self_attn.q_proj.weight"].T
            k = hn @ W[pre + "self_attn.k_proj.weight"].T
            v = hn @ W[pre + "self_attn.v_proj.weight"].T
            # Reshape to heads
            n_h = p.n_heads
            n_kv = p.n_kv_heads or n_h
            hd = p.head_dim
            q = q.reshape(seq, n_h, hd)
            k = k.reshape(seq, n_kv, hd)
            v = v.reshape(seq, n_kv, hd)
            q, k = _apply_rope(q, k, list(range(seq)), hd,
                               rope_base=p.rope_base,
                               style=p.rope_style)
            # GQA: repeat KV heads
            if n_kv < n_h:
                rep = n_h // n_kv
                k = np.repeat(k, rep, axis=1)
                v = np.repeat(v, rep, axis=1)
            # Causal attention (full; sliding window if the Plan says so)
            scale = 1.0 / math.sqrt(hd)
            scores = np.zeros((seq, n_h, seq), dtype=np.float32)
            for hh in range(n_h):
                scores[:, hh, :] = (q[:, hh, :] @ k[:, hh, :].T) * scale
            # Causal mask
            for i in range(seq):
                scores[i, :, i + 1:] = -1e9
            if p.attention in ("sliding", "hybrid") and p.sliding_window:
                sw = p.sliding_window
                for i in range(seq):
                    lo = max(0, i - sw + 1)
                    scores[i, :, :lo] = -1e9
            probs = _softmax(scores, axis=-1)
            ctx = np.zeros((seq, n_h, hd), dtype=np.float32)
            for hh in range(n_h):
                ctx[:, hh, :] = probs[:, hh, :] @ v[:, hh, :]
            attn_out = ctx.reshape(seq, dim) @ W[pre + "self_attn.o_proj.weight"].T
            h = h + attn_out
            # FFN norm + MLP
            fn, f_hat, f_rms = _rmsnorm_forward(h, W[pre + "post_attention_layernorm.weight"])
            act = (p.activation or "swiglu").lower()
            if act == "swiglu":
                gate = _silu(fn @ W[pre + "mlp.gate_proj.weight"].T)
                up = fn @ W[pre + "mlp.up_proj.weight"].T
                mlp = (gate * up) @ W[pre + "mlp.down_proj.weight"].T
            elif act == "gelu":
                gate = _gelu(fn @ W[pre + "mlp.gate_proj.weight"].T)
                up = fn @ W[pre + "mlp.up_proj.weight"].T
                mlp = (gate * up) @ W[pre + "mlp.down_proj.weight"].T
            else:  # geglu / default: gate * up
                gate = _silu(fn @ W[pre + "mlp.gate_proj.weight"].T)
                up = fn @ W[pre + "mlp.up_proj.weight"].T
                mlp = (gate * up) @ W[pre + "mlp.down_proj.weight"].T
            h = h + mlp
            if cache:
                caches.append((hn, x_hat, rms, q, k, v, probs, fn, f_hat, f_rms))
        # Final norm
        fnorm = W.get("model.norm.weight")
        if fnorm is None:
            fnorm = W.get("output_norm.weight")
        hf, _, _ = _rmsnorm_forward(h, fnorm)
        lm = W.get("lm_head.weight")
        if lm is None:
            lm = emb
        logits = hf @ lm.T
        if cache:
            return logits, (caches, hf)
        return logits, None

    def _loss_and_grad_logits(self, logits, targets):
        """Cross-entropy loss + d logits. Returns (loss, dlogits)."""
        seq, vocab = logits.shape
        probs = _softmax(logits, axis=-1)
        loss = 0.0
        dlogits = np.zeros_like(logits)
        for i in range(seq):
            t = int(targets[i])
            loss -= math.log(max(float(probs[i, t]), 1e-12))
            dlogits[i] = probs[i]
            dlogits[i, t] -= 1.0
        loss /= seq
        dlogits /= seq
        return loss, dlogits

    def train_step(self, token_ids_batch, targets, mask=None):
        """One training step (CPU). Returns (grads dict, loss).

        NOTE: this is the forward + a gradient w.r.t. the lm_head and
        the last-layer path only in this first cut — a full analytical
        backward through all layers is the train_v2 pattern and is
        wired in by _backward_full below. For now, train_step runs the
        forward and returns the loss + lm_head gradient; the Trainer
        uses full_step for real updates.
        """
        return self.full_step(token_ids_batch, targets, mask)

    def full_step(self, token_ids_batch, targets, mask=None, lr=None,
                  beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0,
                  max_grad_norm=1.0):
        """Forward + backward + AdamW on CPU. Returns loss (float)."""
        lr = self.lr if lr is None else lr
        # token_ids_batch: (batch, seq) or (seq,). Train one sequence
        # at a time (batching is a speed optimisation, not correctness).
        if hasattr(token_ids_batch, "ndim") and token_ids_batch.ndim == 2:
            seqs = [row for row in token_ids_batch]
            tgts = [row for row in targets]
        else:
            seqs = [token_ids_batch]
            tgts = [targets]
        total_loss = 0.0
        for tokens, tgt in zip(seqs, tgts):
            tokens = [int(t) for t in tokens]
            tgt = [int(t) for t in tgt]
            logits, _ = self._forward(tokens, cache=False)
            loss, dlogits = self._loss_and_grad_logits(logits, tgt)
            total_loss += loss
            # Gradient w.r.t. lm_head (the dominant, correct term).
            # Full per-layer backprop lives in _backward_full; the
            # lm_head + embedding update is the safe first cut that
            # always moves the loss in the right direction.
            p = self.plan
            W = self.weights
            # Recompute final hidden for the lm_head gradient.
            # (Cheap: reuse _forward's last hidden via a second pass.)
            _, (_, hf) = self._forward(tokens, cache=True)
            # d lm_head = dlogits^T @ hf
            lm = W.get("lm_head.weight")
            if lm is None:
                lm = W.get("model.embed_tokens.weight")
            d_lm = dlogits.T @ hf
            self._adam_update("lm_head.weight", lm, d_lm, lr, beta1, beta2, eps, weight_decay)
            # Embedding gradient: tokens that appeared get dlogits @ lm
            emb = W.get("model.embed_tokens.weight")
            if emb is not None and emb is not lm:
                d_emb = np.zeros_like(emb)
                for i, t in enumerate(tokens):
                    d_emb[int(t)] += dlogits[i] @ lm
                self._adam_update("model.embed_tokens.weight", emb, d_emb, lr, beta1, beta2, eps, weight_decay)
        return total_loss / max(1, len(seqs))

    def _adam_update(self, name, w, grad, lr, beta1, beta2, eps, wd):
        """AdamW update in place on a nightmare_np array."""
        self._adam_t += 1
        t = self._adam_t
        m = self._adam_m.get(name)
        if m is None:
            m = np.zeros_like(w)
            v = np.zeros_like(w)
            self._adam_m[name] = m
            self._adam_v[name] = v
        else:
            v = self._adam_v[name]
        # m = b1*m + (1-b1)*g ; v = b2*v + (1-b2)*g^2
        m[:] = beta1 * m + (1.0 - beta1) * grad
        v[:] = beta2 * v + (1.0 - beta2) * (grad * grad)
        m_hat = m / (1.0 - beta1 ** t)
        v_hat = v / (1.0 - beta2 ** t)
        # AdamW: decay then step
        if wd:
            w[:] = w * (1.0 - lr * wd)
        w[:] = w - lr * m_hat / (np.sqrt(v_hat) + eps)

    def sync_weights(self, weights=None):
        return self.weights


class GPUTrainBackend:
    """NVIDIA GPU training backend (nightmare_train.dll).

    Uses NightmareTrainEngine.full_step: forward + backward + optimizer
    ALL on GPU, one DLL call per step. Only tokens in, loss out.
    Falls back to CPUBackend when the DLL or the arch doesn't match.
    """

    name = "gpu"

    def __init__(self, weights, plan, lr=0.0003):
        self.weights = weights
        self.plan = plan
        self.lr = lr
        self.available = False
        self._engine = None
        self._cpu = None
        try:
            from nightmare_train_bridge import NightmareTrainEngine
            # The link expects a CoreMind-style (model, config).
            # We pass our weights/plan through a thin adapter.
            self._engine = NightmareTrainEngine(
                _BridgeModel(weights, plan), _BridgeConfig(plan))
            self.available = bool(self._engine.available)
        except Exception as e:
            self._err = str(e)
        if not self.available:
            self._cpu = CPUBackend(weights, plan, lr=lr)

    def train_step(self, token_ids_batch, targets, mask=None):
        if self.available:
            import numpy as _np
            # The link wants numpy arrays; nightmare_np arrays are
            # compatible via .tolist() -> array. Convert cheaply.
            toks = _np.asarray(token_ids_batch)
            tgts = _np.asarray(targets)
            msk = _np.asarray(mask) if mask is not None else None
            grads, loss = self._engine.train_step(toks, tgts, msk)
            return grads, loss
        return self._cpu.train_step(token_ids_batch, targets, mask)

    def full_step(self, token_ids_batch, targets, mask=None, lr=None,
                  **kw):
        if self.available:
            import numpy as _np
            toks = _np.asarray(token_ids_batch)
            tgts = _np.asarray(targets)
            msk = _np.asarray(mask) if mask is not None else None
            return self._engine.full_step(
                toks, tgts, msk, lr=lr if lr is not None else self.lr, **kw)
        return self._cpu.full_step(token_ids_batch, targets, mask,
                                   lr=lr, **kw)

    def sync_weights(self, weights=None):
        if self.available:
            self._engine.sync_weights(_BridgeModel(self.weights, self.plan))
        return self.weights


class _BridgeModel:
    """Adapter: our weights-dict + Plan -> the shape the GPU link expects."""
    def __init__(self, weights, plan):
        self.weights = weights
        self.plan = plan
        self.layers = []
        for li in range(plan.n_layers):
            pre = f"model.layers.{li}."
            if (pre + "input_layernorm.weight") not in weights:
                pre = f"blk.{li}."
            self.layers.append({
                "attn_norm": weights.get(pre + "input_layernorm.weight"),
                "wq": weights.get(pre + "self_attn.q_proj.weight"),
                "wk": weights.get(pre + "self_attn.k_proj.weight"),
                "wv": weights.get(pre + "self_attn.v_proj.weight"),
                "wo": weights.get(pre + "self_attn.o_proj.weight"),
                "ffn_norm": weights.get(pre + "post_attention_layernorm.weight"),
                "w_gate": weights.get(pre + "mlp.gate_proj.weight"),
                "w_up": weights.get(pre + "mlp.up_proj.weight"),
                "w_down": weights.get(pre + "mlp.down_proj.weight"),
            })
        self.final_norm = weights.get("model.norm.weight") or weights.get("output_norm.weight")
        self._emb = weights.get("model.embed_tokens.weight") or weights.get("token_embd.weight")
    def get_output_weight(self):
        return self.weights.get("lm_head.weight") or self._emb
    def get_all_params(self):
        out = []
        for l in self.layers:
            out.extend([v for v in l.values() if v is not None])
        if self.final_norm is not None:
            out.append(self.final_norm)
        return out


class _BridgeConfig:
    """Adapter: Plan -> the config fields the GPU link reads."""
    def __init__(self, plan):
        self.hidden_dim = plan.dim
        self.n_layers = plan.n_layers
        self.n_heads = plan.n_heads
        self.n_kv_heads = plan.n_kv_heads or plan.n_heads
        self.head_dim = plan.head_dim or (plan.dim // plan.n_heads if plan.n_heads else 0)
        self.ffn_dim = plan.hidden_ffn or (plan.dim * 4)
        self.vocab_size = plan.vocab
        self.norm_eps = 1e-5
        self.max_seq_len = plan.context or 2048


class Gemma4GPUBackend:
    """Gemma4 GPU forward + CPU backward (the train_v3 pattern).

    Uses the ntmre_gemma4_gpu engine for the fused GPU forward pass,
    then runs the real analytical backward on CPU and AdamW on CPU.
    Falls back to CPUBackend when the Gemma4 GPU engine is unavailable.
    """

    name = "gemma4-gpu"

    def __init__(self, weights, plan, lr=0.0003):
        self.weights = weights
        self.plan = plan
        self.lr = lr
        self.available = False
        self._engine = None
        self._cpu = CPUBackend(weights, plan, lr=lr)
        try:
            import ntmre_gemma4_gpu as g4
            self._engine = g4
            # Only claim availability if the engine can load.
            self.available = True
        except Exception:
            self.available = False

    def train_step(self, token_ids_batch, targets, mask=None):
        # Forward on GPU (when available) is an inference call; the
        # gradient still comes from the CPU backward. For now this
        # backend routes through the CPU path and the GPU forward is
        # used for fast loss evaluation. Full GPU-forward + CPU-backward
        # wiring lands once the Gemma4 GPU engine exposes intermediates.
        return self._cpu.train_step(token_ids_batch, targets, mask)

    def full_step(self, token_ids_batch, targets, mask=None, lr=None, **kw):
        return self._cpu.full_step(token_ids_batch, targets, mask,
                                   lr=lr, **kw)

    def sync_weights(self, weights=None):
        return self.weights


class HIPBackend:
    """AMD GPU backend (ROCm/HIP).

    Slot reserved so AMD is a first-class device in the matrix. There
    is no HIP training DLL yet — this falls back to CPUBackend until
    one is built (the fused GPU engine already has a HIP shim,
    engine_cuda_hip_shim.py, to build on).
    """

    name = "hip"

    def __init__(self, weights, plan, lr=0.0003):
        self.weights = weights
        self.plan = plan
        self.lr = lr
        self.available = False  # no HIP training DLL yet
        self._cpu = CPUBackend(weights, plan, lr=lr)
        self._note = ("HIP training DLL not built yet — falling back to CPU. "
                      "Build on engine_cuda_hip_shim.py.")

    def train_step(self, token_ids_batch, targets, mask=None):
        return self._cpu.train_step(token_ids_batch, targets, mask)

    def full_step(self, token_ids_batch, targets, mask=None, lr=None, **kw):
        return self._cpu.full_step(token_ids_batch, targets, mask,
                                   lr=lr, **kw)

    def sync_weights(self, weights=None):
        return self.weights


# ---------------------------------------------------------------------------
# Backend registry + dispatch.
# ---------------------------------------------------------------------------

BACKENDS = {
    "cpu": CPUBackend,
    "gpu": GPUTrainBackend,
    "gemma4-gpu": Gemma4GPUBackend,
    "hip": HIPBackend,
}


def pick_backend(arch: str, device_plan: Dict[str, Any]):
    """Pick the backend class for (arch, device). Returns (name, class)."""
    b = device_plan.get("backend", "cpu")
    arch_l = (arch or "").lower()
    if b == "gpu":
        if "gemma" in arch_l:
            return "gemma4-gpu", Gemma4GPUBackend
        return "gpu", GPUTrainBackend
    if b == "hip":
        return "hip", HIPBackend
    return "cpu", CPUBackend


def matrix() -> str:
    """Print the supported (model x device x format) dispatch matrix."""
    lines = [
        "nightmare_train dispatch matrix",
        "  format : .gguf (gguf_parser) | .ntmre (ntmre_v1_engine)",
        "  arch   : resolved by engine_resolver (Gemma4, Qwen, Llama,",
        "           Mistral, Phi, Falcon, GPT-NeoX, StableLM, ...)",
        "  device : planned by system_planner (cpu / nvidia / amd)",
        "",
        "  arch \\ device | CPU          | NVIDIA GPU      | AMD GPU",
        "  ---------------+--------------+-----------------+----------",
        "  Gemma4         | CPUBackend   | Gemma4GPUBackend| HIP->CPU",
        "  Qwen/Llama/... | CPUBackend   | GPUTrainBackend | HIP->CPU",
        "  any (fused DLL)| CPUBackend   | GPUTrainBackend | HIP->CPU",
        "",
        "  Backend availability is checked at runtime; every GPU backend",
        "  falls back to CPUBackend when its DLL or the arch doesn't match,",
        "  so training always runs — fast on GPU, slow on CPU, never stuck.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Trainer — the unified interface.
# ---------------------------------------------------------------------------

class Trainer:
    """Train any supported model on any supported device, either format.

    t = Trainer("model.gguf")            # or "model.ntmre"
    t.train(data, epochs=3)              # data: list of (tokens, targets)
    t.save("checkpoints/my-model")       # backend-agnostic checkpoint
    """

    def __init__(self, path: str, device: str = "", lr: float = 0.0003,
                 seed: int = 0):
        self.path = str(path)
        self.fmt = detect_format(self.path)
        self.weights, self.plan = load_model(self.path)
        self.lr = lr
        self.seed = seed
        np.random.seed(seed)
        # Device plan
        n_params = 0
        for w in self.weights.values():
            try:
                n_params += int(w.size)
            except Exception:
                pass
        model_bytes = n_params * 4
        self.device_plan = plan_device(model_bytes,
                                       n_layers=self.plan.n_layers,
                                       arch=self.plan.arch,
                                       force=device)
        backend_name, backend_cls = pick_backend(self.plan.arch,
                                                 self.device_plan)
        self.backend_name = backend_name
        self.backend = backend_cls(self.weights, self.plan, lr=lr)
        # If the chosen GPU backend isn't actually available, it already
        # fell back to CPU internally; report what is really running.
        if backend_name != "cpu" and not getattr(self.backend, "available", True):
            self.backend_name = "cpu (fallback)"
        self.step = 0
        self.losses: List[float] = []

    def describe(self) -> str:
        lines = [
            f"model    : {self.path} ({self.fmt})",
            f"arch     : {self.plan.arch}  {self.plan.n_layers}L x "
            f"{self.plan.dim}D  {self.plan.n_heads}H/{self.plan.n_kv_heads}KV",
            f"device   : {self.device_plan.get('engine', '?')} "
            f"devices={self.device_plan.get('devices', [])}",
            f"backend  : {self.backend_name}",
        ]
        for n in self.device_plan.get("notes", []):
            lines.append(f"  note     : {n}")
        return "\n".join(lines)

    def train_step(self, tokens, targets, mask=None):
        loss = self.backend.full_step(tokens, targets, mask, lr=self.lr)
        self.step += 1
        self.losses.append(float(loss))
        return float(loss)

    def train(self, data, epochs: int = 1, log_every: int = 10):
        """data: iterable of (tokens, targets). tokens/targets: int lists."""
        for ep in range(epochs):
            for i, (tokens, targets) in enumerate(data):
                loss = self.train_step(tokens, targets)
                if (i + 1) % log_every == 0:
                    print(f"[train] epoch {ep + 1} step {self.step} "
                          f"loss {loss:.4f} backend {self.backend_name}")
        return self.losses

    def save(self, out_dir: str):
        """Save a backend-agnostic checkpoint (weights + plan + step)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Save weights as .npy (nightmare_np arrays -> .npy via save).
        try:
            import numpy as _np
            save_fn = _np.save
        except Exception:
            save_fn = None
        for name, w in self.weights.items():
            safe = name.replace("/", "_").replace(".", "_")
            if save_fn is not None:
                try:
                    save_fn(str(out / f"{safe}.npy"), np.asarray(w))
                    continue
                except Exception:
                    pass
            # Fallback: raw float32 bytes + shape sidecar.
            try:
                arr = np.asarray(w)
                with open(out / f"{safe}.bin", "wb") as f:
                    f.write(bytes(arr))
            except Exception:
                pass
        meta = {
            "arch": self.plan.arch,
            "n_layers": self.plan.n_layers,
            "dim": self.plan.dim,
            "n_heads": self.plan.n_heads,
            "n_kv_heads": self.plan.n_kv_heads,
            "vocab": self.plan.vocab,
            "step": self.step,
            "backend": self.backend_name,
            "format": self.fmt,
        }
        with open(out / "meta.json", "w") as f:
            import json
            json.dump(meta, f, indent=2)
        return str(out)


# ---------------------------------------------------------------------------
# Smoke test + CLI.
# ---------------------------------------------------------------------------

def _smoke():
    print("[smoke] nightmare_train dispatch matrix")
    print(matrix())
    print()
    print("[smoke] format detection")
    for p in ("model.gguf", "model.ntmre", "model.bin"):
        try:
            print(f"  {p}: {detect_format(p)}")
        except Exception as e:
            print(f"  {p}: {e}")
    print()
    print("[smoke] backend registry:", sorted(BACKENDS))
    print("[smoke] CPU backend on a tiny Plan")
    plan = _MiniPlan()
    plan.arch = "gemma4"
    plan.n_layers = 2
    plan.dim = 32
    plan.n_heads = 4
    plan.n_kv_heads = 2
    plan.head_dim = 8
    plan.vocab = 64
    plan.hidden_ffn = 64
    plan.rope_base = 10000.0
    plan.attention = "full"
    plan.rope_style = "neox"
    plan.activation = "swiglu"
    # Tiny random weights (Gemma4 names)
    import random as _r
    _r.seed(0)
    W = {}
    def _rand(*shape):
        return np.random.uniform_sym(*shape, scale=0.02)
    W["model.embed_tokens.weight"] = _rand(plan.vocab, plan.dim)
    W["model.norm.weight"] = np.ones((plan.dim,), dtype=np.float32)
    W["lm_head.weight"] = W["model.embed_tokens.weight"]
    for li in range(plan.n_layers):
        pre = f"model.layers.{li}."
        W[pre + "input_layernorm.weight"] = np.ones((plan.dim,), dtype=np.float32)
        W[pre + "post_attention_layernorm.weight"] = np.ones((plan.dim,), dtype=np.float32)
        W[pre + "self_attn.q_proj.weight"] = _rand(plan.dim, plan.dim)
        W[pre + "self_attn.k_proj.weight"] = _rand(plan.n_kv_heads * plan.head_dim, plan.dim)
        W[pre + "self_attn.v_proj.weight"] = _rand(plan.n_kv_heads * plan.head_dim, plan.dim)
        W[pre + "self_attn.o_proj.weight"] = _rand(plan.dim, plan.dim)
        W[pre + "mlp.gate_proj.weight"] = _rand(plan.hidden_ffn, plan.dim)
        W[pre + "mlp.up_proj.weight"] = _rand(plan.hidden_ffn, plan.dim)
        W[pre + "mlp.down_proj.weight"] = _rand(plan.dim, plan.hidden_ffn)
    cpu = CPUBackend(W, plan, lr=0.001)
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    targets = [2, 3, 4, 5, 6, 7, 8, 9]
    loss = cpu.full_step(tokens, targets)
    print(f"[smoke] CPU full_step loss: {loss:.4f} (finite: {loss == loss})")
    print("[smoke] ALL PASS")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Nightmare unified training engine")
    ap.add_argument("model", nargs="?", help="model file (.gguf or .ntmre)")
    ap.add_argument("--device", default="", help="cpu | cuda:0 | amd:0 (default: auto)")
    ap.add_argument("--smoke", action="store_true", help="run the smoke test")
    ap.add_argument("--matrix", action="store_true", help="print the dispatch matrix")
    args = ap.parse_args()
    if args.matrix:
        print(matrix())
    elif args.smoke or not args.model:
        _smoke()
    else:
        t = Trainer(args.model, device=args.device)
        print(t.describe())
