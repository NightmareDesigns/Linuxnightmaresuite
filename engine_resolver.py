"""
RESOLVER -- header in, validated engine plan out.

WHAT THIS IS FOR
----------------
The router dispatches. The builder constructs. Both need the same thing
first: a machine-readable answer to "what IS this model, and what would
running it require?"

Without it the router is a lookup table keyed on architecture strings,
and every architecture someone did not think of falls through to a
generic engine that may not implement the math. The fallback is silent,
which is the worst kind: the model loads, prints Ready, and talks noise.

With it, the router can say "I recognise that" or "build this", and the
builder has a target.

FOUR AXES, AND EVERYONE ELSE GETS THEM WRONG
--------------------------------------------
    attention   full RoPE / NeoX RoPE / sliding / hybrid
    activation  SwiGLU / GeGLU-tanh / GeLU
    ffn         dense / MoE (fused gate_up, or split)
    quant       the set of block formats present

THE RULE ABOUT HEADERS
----------------------
Headers lie, and this codebase already documents it:

  "Do not trust the header's head_dim. Do not 'fix' it to dim/heads
   either -- that is what convert.c does and it is why the header
   disagrees."  (GEMMA4_HANDOFF, section 2)

So shape-derived truth wins everywhere it exists, and the header is
demoted to a hint. Every field records where it came from, because
"which source" is the question that decides whether a plan is
trustworthy.

  GROUND   read from a tensor shape
  HEADER   read from metadata
  GUESS    inferred because nothing stated it

A plan with GUESS in it is a plan that should be verified before it is
trusted. That is not a warning label, it is the whole point.

WHAT IT DOES NOT DO
-------------------
It does not compile, place, or run anything. It answers one question
and answers it from the model's own bytes. Placement needs live VRAM,
which belongs to the builder.

Author: Nightmare Designs
"""
from __future__ import annotations

import json
import os
import re
import struct
import sys
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# ---------------------------------------------------------------------------
# GGUF value types. v3 remaps 9 to ARRAY, which is the single most common
# way a hand-rolled reader silently truncates a 60-key header to 17 keys
# and reports a model with no layers.
# ---------------------------------------------------------------------------
_GGUF = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2),
    4: ("<I", 4), 5: ("<i", 4), 6: ("<f", 4), 7: ("<?", 1),
    8: ("<Q", 8),      # string (u64 length prefix)
    10: ("<q", 8),     # int64 in v3
    11: ("<d", 8),     # float64
}
_ARRAY_V3 = 9
_ARRAY_V2 = 12

# Keep small arrays whole. sliding_window_pattern is one bool per layer,
# and a reader that samples 8 of 30 returns a list that then indexes out
# of range against a 30-layer model.
_KEEP_ALL_UNDER = 4096
_KEEP = 8


class Ground:
    GROUND = "GROUND"
    HEADER = "HEADER"
    GUESS = "GUESS"


class ResolveError(Exception):
    pass


def _read_str(f) -> str:
    raw = f.read(8)
    if len(raw) < 8:
        raise EOFError("truncated key length")
    n = struct.unpack("<Q", raw)[0]
    if n > 1 << 20:
        raise ValueError("implausible key length %d -- desynchronised" % n)
    data = f.read(n)
    if len(data) < n:
        raise EOFError("truncated key")
    return data.decode("utf-8", errors="replace")


def _read_value(f, vtype: int):
    if vtype == _ARRAY_V3:
        (et,) = struct.unpack("<I", f.read(4))
        (n,) = struct.unpack("<Q", f.read(8))
        return _read_array(f, et, n)
    if vtype == _ARRAY_V2:
        (et,) = struct.unpack("<I", f.read(4))
        return _read_array(f, et, None)
    if vtype not in _GGUF:
        return None
    code, width = _GGUF[vtype]
    if vtype == 8:
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", errors="replace")
    raw = f.read(width)
    if len(raw) < width:
        raise EOFError("truncated header")
    return struct.unpack(code, raw)[0]


def _read_array(f, etype: int, n: Optional[int]):
    if n is None:
        (n,) = struct.unpack("<Q", f.read(8))
        (etype,) = struct.unpack("<I", f.read(4))
    n = int(n)
    if n < 0 or n > 8_000_000:
        raise ValueError("implausible array length %d" % n)
    keep = n if n <= _KEEP_ALL_UNDER else _KEEP
    out: List[Any] = []
    if etype == 8:
        # Strings. The skipped tail is SEEKED past, not read -- reading
        # it would be correct too but 150k tokens of vocab is seconds of
        # work for data nothing here uses. What matters is that the
        # position ends up right.
        for i in range(n):
            raw = f.read(8)
            if len(raw) < 8:
                raise EOFError("truncated string array")
            ln = struct.unpack("<Q", raw)[0]
            if ln > 1 << 20:
                raise ValueError("implausible string length %d" % ln)
            if i < keep:
                d = f.read(ln)
                if len(d) < ln:
                    raise EOFError("truncated string array body")
                out.append(d.decode("utf-8", errors="replace"))
            else:
                f.seek(ln, os.SEEK_CUR)
    else:
        entry = _GGUF.get(etype)
        if entry is None or entry[0] is None:
            raise ValueError("unknown array element type %d" % etype)
        code, width = entry
        # Fixed-width elements, read in order and then skipped in one
        # seek. The order matters: seeking past the tail BEFORE reading
        # the head leaves the file positioned after the whole array with
        # the head unread, and the next key is read from the middle of
        # the data -- which is the desync this is fixing.
        head = min(n, keep)
        for _i in range(head):
            raw = f.read(width)
            if len(raw) < width:
                raise EOFError("truncated array")
            out.append(struct.unpack(code, raw)[0])
        if n > head:
            f.seek(width * (n - head), os.SEEK_CUR)
    return out


def read_header(path: str) -> Tuple[Dict[str, Any], List[Tuple[str, Tuple]],
                                     int]:
    """Metadata, tensor name->shape, tensor count. No tensor data read."""
    meta: Dict[str, Any] = {}
    tensors: List[Tuple[str, Tuple]] = []
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ResolveError("not a GGUF file")
        f.read(4)                                   # version
        (n_tensors,) = struct.unpack("<Q", f.read(8))
        (n_kv,) = struct.unpack("<Q", f.read(8))
        try:
            for _ in range(min(int(n_kv), 8192)):
                key = _read_str(f)
                (vt,) = struct.unpack("<I", f.read(4))
                meta[key] = _read_value(f, vt)
            for _ in range(min(int(n_tensors), 65536)):
                name = _read_str(f)
                (nd,) = struct.unpack("<I", f.read(4))
                dims = []
                for _d in range(nd):
                    (v,) = struct.unpack("<Q", f.read(8))
                    dims.append(int(v))
                f.read(8)                           # offset, u64
                f.read(4)                           # ggml type, u32
                # The type is FOUR bytes, not eight. Reading eight
                # leaves the stream four bytes past the end of this
                # entry, so the next name is read from the middle of
                # the previous one. The symptom is a key length around
                # 8e18, which is what made every real model look
                # unreadable while the same code parsed a 40-key header
                # correctly when stepped by hand.
                tensors.append((name, tuple(dims)))
        except (ValueError, EOFError, struct.error) as e:
            # A truncated or corrupt header is a normal thing to find on
            # a machine with 100 model files, and it is a fact about the
            # FILE, not a bug in the reader. Raising a bare desync error
            # makes the two indistinguishable, so the parse state is
            # reported alongside it.
            raise ResolveError(
                "header is corrupt or truncated after %d of %d metadata "
                "keys and %d of %d tensors: %s"
                % (len(meta), n_kv, len(tensors), n_tensors, e))
    return meta, tensors, int(n_tensors)


# ---------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------

@dataclass
class Plan:
    """Everything needed to build or choose an engine for one model."""

    path: str = ""
    arch: str = ""
    arch_source: str = Ground.HEADER

    # geometry
    n_layers: int = 0
    dim: int = 0
    n_heads: int = 0
    n_kv_heads: int = 0
    head_dim: int = 0
    vocab: int = 0
    context: int = 0
    rope_base: float = 10000.0
    hidden_ffn: int = 0
    expert_ffn: int = 0
    n_experts: int = 0
    n_experts_used: int = 0

    # axis 1: attention
    attention: str = "full"                 # full | sliding | hybrid
    rope_style: str = "normal"              # normal | neox
    sliding_window: int = 0
    n_swa_layers: int = 0
    per_layer_kdim: List[int] = field(default_factory=list)
    per_layer_vdim: List[int] = field(default_factory=list)

    # axis 2: activation
    activation: str = "unknown"             # swiglu | geglu | gelu | unknown
    act_source: str = Ground.GUESS

    # axis 3: ffn
    ffn: str = "dense"                      # dense | moe
    gate_up_fused: bool = True
    ffn_source: str = Ground.GUESS

    # axis 4: quant
    quant_types: List[int] = field(default_factory=list)
    quant_names: List[str] = field(default_factory=list)

    # NVRTC / spec-based engine choice — ADDED 2026-10-28, admin copy.
    # The resolver's 3 checks above are unchanged. These fields are
    # filled by apply_nvrtc_choice() (below) from discovered specs,
    # never by the model geometry alone. They carry the SELECTION
    # (prebuilt PTX only, never a build) that the router uses.
    #   nvrtc_arch: "75" | "86" | "" (no PTX / non-NVIDIA / unknown)
    #   nvrtc_ptx: absolute path to a prebuilt PTX in nvrtc-kernels/,
    #              or "" when none was selected
    #   engine_mode: "nvrtc_full_vram" | "nvrtc_mesh_split" | "stream" |
    #                "amd_hiprtc" | "cpu" | "" (not yet chosen)
    #   engine_reason: plain-English reason for the choice, always
    #                  present once a choice is made — no silent CPU
    #                  fallback, no silent degradation
    #   nvrtc_vendor: "nvidia" | "amd" | "none" | ""
    nvrtc_arch: str = ""
    nvrtc_ptx: str = ""
    engine_mode: str = ""
    engine_reason: str = ""
    nvrtc_vendor: str = ""

    # provenance and risk
    guesses: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def trusted(self) -> bool:
        """False when anything was inferred rather than measured."""
        return not self.guesses

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        bits = ["%s  %dL x %dD  %dH/%dKV  ctx %d"
                % (self.arch, self.n_layers, self.dim, self.n_heads,
                   self.n_kv_heads, self.context)]
        bits.append("attn=%s rope=%s" % (self.attention, self.rope_style))
        bits.append("act=%s ffn=%s" % (self.activation, self.ffn))
        if self.ffn == "moe":
            bits.append("experts=%d/%d" % (self.n_experts_used,
                                          self.n_experts))
        bits.append("quant=%s" % (",".join(self.quant_names) or "?"))
        bits.append("TRUSTED" if self.trusted else "HAS GUESSES")
        if self.engine_mode:
            bits.append("nvrtc=%s arch=%s ptx=%s (%s)"
                        % (self.engine_mode, self.nvrtc_arch or "-",
                           self.nvrtc_ptx or "none", self.engine_reason))
        return "\n    ".join(bits)


# GGML type ids, for a human-readable quant list
_GGML_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
    14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS",
    19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS",
    24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64", 29: "IQ1_M",
    30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 36: "MXFP4",
}

# tensor-name stems that identify the activation, read from the model's
# own layout rather than guessed from the family name.
_ACT_PATTERNS = [
    ("swiglu", re.compile(r"ffn_gate_up_exps|ffn_gate\b|ffn_up\b")),
    ("geglu", re.compile(r"geglu|tanh_glu|geglu_tanh")),
    ("gelu", re.compile(r"ffn_gate\b.*gelu|gelu")),
]


# ---- Verified Gemma4 variant table, 2026-10-28 (CHECK only, never override) ----
# Source: Drive specs.json 2026-10-24 (21 entries, restored as specs.json
# with 20 entries). Head_dim is a SET because Gemma4 uses dual head_dim:
# SWA layers 256, full layers 512 (E-models), and 12B full layers up to
# 2048 in the Drive measurements. Context is a set for 12B (131072 or
# 262144 in Drive). PLE/family for E4B allow BOTH the E-model form
# (ple 10752) AND the Mahou Imatrix form (ple None, normalised QK,
# .imatrix.gguf, 4.4 MB — an imatrix file, not a full model, kept as a
# separate spec entry and NOT merged with Mahou Nsfw).
# 26B-A4B-It (30L/2816d) is MoE: ffn_hidden 2112 is the Drive value,
# expert tensors must be used — never treat it as a dense model.
VERIFIED_GEMMA4_VARIANTS = {
    (35, 1536): {"variant": "E2B", "n_layers": 35, "dim": 1536, "n_heads": 8,
                 "n_kv_heads": 1, "head_dims": {256, 512}, "ffn_hidden": 6144,
                 "vocab": 262144, "contexts": {131072}, "ple_dims": {8960},
                 "families": {"per-layer-embedding decoder (E-model)"}, "moe": False},
    (42, 2560): {"variant": "E4B", "n_layers": 42, "dim": 2560, "n_heads": 8,
                 "n_kv_heads": 2, "head_dims": {256, 512}, "ffn_hidden": 10240,
                 "vocab": 262144, "contexts": {131072}, "ple_dims": {10752, None},
                 "families": {"per-layer-embedding decoder (E-model)", "normalised QK decoder"},
                 "moe": False},
    (48, 3840): {"variant": "12B", "n_layers": 48, "dim": 3840, "n_heads": 16,
                 "n_kv_heads": 8, "head_dims": {256, 2048}, "ffn_hidden": 15360,
                 "vocab": 262144, "contexts": {131072, 262144}, "ple_dims": {None},
                 "families": {"normalised QK decoder", "normalised QK decoder (gemma-style)"},
                 "moe": False},
    (30, 2816): {"variant": "26B-A4B-It", "n_layers": 30, "dim": 2816, "n_heads": 16,
                 "n_kv_heads": 8, "head_dims": {2048}, "ffn_hidden": 2112,
                 "vocab": 262144, "contexts": {262144}, "ple_dims": {None},
                 "families": {"normalised QK decoder"}, "moe": True},
}
# Suspect geometry, deliberately NOT a verified variant:
# (4, 1024) "Gemma 4 12B It Qat Q4_0 Unquantized Assistant", arch
# gemma4-assistant — see specs-suspect.json. A plan with this geometry
# must warn, never match a 12B spec.
SUSPECT_GEMMA4_GEOMETRY = (4, 1024)


def check_plan_against_verified_variants(plan):
    """Warn (never override) when a Plan disagrees with the verified table.

    Identification is by (n_layers, dim) only, from the Drive specs.
    If a Plan's layers+dim match a verified variant but its heads/kv/
    head_dim/ffn/vocab/context differ from the table's allowed sets,
    a warning naming both sets is appended to plan.warnings.
    Correct metadata is left untouched — the table is a check, not an
    override. A Gemma4 plan whose (layers, dim) is in NO verified
    variant — including the suspect (4, 1024) — gets an 'unknown /
    unverified' warning: it is not in the Drive specs file, so it must
    not be treated as a known model. Non-Gemma4 plans are untouched.
    """
    key = (plan.n_layers, plan.dim)
    if key == SUSPECT_GEMMA4_GEOMETRY:
        plan.warnings.append(
            "model setup check: geometry 4L x 1024d matches the SUSPECT "
            "Drive entry 'Gemma 4 12B It Qat Q4_0 Unquantized Assistant' "
            "(gemma4-assistant, 49 tensors) — NOT a verified 12B spec, "
            "see specs-suspect.json. Do not load it as a 12B.")
        return plan
    spec = VERIFIED_GEMMA4_VARIANTS.get(key)
    if not spec:
        arch = str(getattr(plan, "arch", "") or "").lower()
        if "gemma" in arch:
            plan.warnings.append(
                "model setup check: Gemma4 geometry %dL x %dd is not in "
                "the verified Drive specs table (E2B 35/1536, E4B 42/2560, "
                "12B 48/3840, 26B-A4B 30/2816) — unverified, verify before "
                "trusting" % (plan.n_layers, plan.dim))
        return plan
    if spec["moe"] and getattr(plan, "ffn", "") != "moe":
        plan.warnings.append(
            "model setup check (26B-A4B-It): this is an MoE model, but "
            "the plan's ffn is '%s' — use expert tensors, never dense "
            "E4B/12B numbers" % getattr(plan, "ffn", ""))
    for field, plan_field, want in (
            ("n_heads", "n_heads", {spec["n_heads"]}),
            ("n_kv_heads", "n_kv_heads", {spec["n_kv_heads"]}),
            ("head_dim", "head_dim", spec["head_dims"]),
            ("ffn_hidden", "hidden_ffn", {spec["ffn_hidden"]}),
            ("vocab", "vocab", {spec["vocab"]}),
            ("context", "context", spec["contexts"])):
        got = getattr(plan, plan_field, 0)
        if got and got not in want:
            plan.warnings.append(
                "model setup check (%s): %s is %s in the loaded model but "
                "allowed %s in the verified Drive spec — loaded metadata "
                "kept, verify before trusting"
                % (spec["variant"], field, got, sorted(want)))
    return plan


def resolve(path: str) -> Plan:
    """
    Build a Plan from a GGUF file.

    Every field records whether it came from a tensor shape, the header,
    or an inference. Shapes beat headers everywhere, because the
    documented behaviour of this converter family is that the header
    disagrees with the tensors and the tensors are right.
    """
    meta, tensors, n_tensors = read_header(path)
    p = Plan(path=path)
    tmap = dict(tensors)
    names = [n for n, _ in tensors]

    # --- architecture ------------------------------------------------
    arch = str(meta.get("general.architecture", "") or "")
    p.arch = arch or "unknown"
    a = arch + "."

    def hdr(key, default=0):
        v = meta.get(a + key, default)
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def hf(key, default=0.0):
        v = meta.get(a + key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    p.n_layers = hdr("block_count") or hdr("n_layers")
    p.dim = hdr("embedding_length")
    p.n_heads = hdr("attention.head_count")
    p.n_kv_heads = hdr("attention.head_count_kv")
    p.head_dim = hdr("attention.key_length")
    p.vocab = hdr("vocabulary_length") or hdr("vocab_size")
    p.context = hdr("context_length")
    p.rope_base = hf("rope.freq_base", 10000.0)
    p.hidden_ffn = hdr("feed_forward_length")
    p.expert_ffn = hdr("expert_feed_forward_length")
    p.n_experts = hdr("expert_count")
    p.n_experts_used = hdr("expert_used_count")
    p.sliding_window = hdr("attention.sliding_window")

    # --- GROUND: derive from tensor shapes ---------------------------
    # The shapes in this container are REVERSED: a 2D weight is stored
    # (in_features, out_features), not (out, in). The handoff says so
    # explicitly:
    #
    #   "attn_q = 4096 x 3840 (12B) -> 4096 / 16 heads = 256"
    #
    # so the OUTPUT axis is shape[1], not shape[0]. Taking shape[0] gives
    # 2816/16 = 176 for the 26B, which is not a head size and not even
    # a divisor that means anything. The warning fired, which is the
    # only reason this was caught -- a silent wrong value here would
    # produce a KV cache estimate off by 3x and no visible error.
    def out_dim(name):
        sh = tmap.get(name)
        if not sh or len(sh) != 2:
            return 0
        return int(sh[1])          # reversed convention: output last

    q = tmap.get("blk.0.attn_q.weight")
    k = tmap.get("blk.0.attn_k.weight")
    v = tmap.get("blk.0.attn_v.weight")
    if q and p.n_heads:
        rows = out_dim("blk.0.attn_q.weight")
        if rows and rows % p.n_heads == 0:
            derived = rows // p.n_heads
            if not p.head_dim:
                p.head_dim = derived
            elif derived != p.head_dim:
                # A header/tensor disagreement is normal in this family
                # and the tensor is right, so it is recorded and the
                # tensor wins.
                p.warnings.append(
                    "head_dim: header says %d, attn_q gives %d -- using the "
                    "tensor" % (p.head_dim, derived))
                p.head_dim = derived
            p.arch_source = Ground.GROUND

    # n_kv from attn_k width. On a hybrid model head_count_kv is a
    # PER-LAYER ARRAY, so the scalar max is not the layer-0 value and
    # the tensor is the only reliable source for the widest layer.
    if k and p.head_dim:
        kv_w = out_dim("blk.0.attn_k.weight")
        if kv_w and kv_w % p.head_dim == 0:
            derived_kv = kv_w // p.head_dim
            if derived_kv != p.n_kv_heads:
                p.warnings.append(
                    "n_kv_heads: header %s, attn_k gives %d"
                    % (p.n_kv_heads, derived_kv))
            p.n_kv_heads = derived_kv

    # --- axis 1: attention shape, per layer --------------------------
    full_k = hdr("attention.key_length", 0)
    full_v = hdr("attention.value_length", full_k)
    swa_k = hdr("attention.key_length_swa", 0)
    swa_v = hdr("attention.value_length_swa", 0)
    pattern = meta.get(a + "attention.sliding_window_pattern")
    if isinstance(pattern, list):
        pattern = [bool(x) for x in pattern if isinstance(x, (int, float))]

    per_k: List[int] = []
    per_v: List[int] = []
    for i in range(max(0, p.n_layers)):
        kk = tmap.get("blk.%d.attn_k.weight" % i)
        vv = tmap.get("blk.%d.attn_v.weight" % i)
        # Output axis, same reversed convention as above.
        per_k.append(int(kk[1]) if kk and len(kk) == 2 else 0)
        per_v.append(int(vv[1]) if vv and len(vv) == 2 else 0)

    if per_k and any(per_k):
        n_swa = sum(1 for w in per_k if w and w != per_k[0])
        n_no_v = sum(1 for i in range(len(per_v)) if not per_v[i])
        p.per_layer_kdim = per_k
        p.per_layer_vdim = per_v
        p.n_swa_layers = n_swa
        p.arch_source = Ground.GROUND
        if n_swa == 0 and p.sliding_window == 0:
            p.attention = "full"
        elif n_swa and n_swa < len(per_k):
            p.attention = "hybrid"
        elif n_swa and p.sliding_window:
            p.attention = "sliding"
        else:
            p.attention = "full"
        if n_no_v and n_no_v == n_swa:
            p.notes.append(
                "%d sliding layers carry no attn_v tensor -- they share a "
                "compressed value path" % n_no_v)
    else:
        p.attention = "full" if not p.sliding_window else "sliding"

    # rope style: the same families that are half-split/NeoX
    p.rope_style = "neox" if re.search(
        r"qwen|phi|gemma|gptneox|stablelm|falcon", arch, re.I) else "normal"

    # --- axis 2: activation ------------------------------------------
    for stem, rx in _ACT_PATTERNS:
        if any(rx.search(n) for n in names):
            p.activation = stem
            p.act_source = Ground.GROUND
            break
    if p.activation == "unknown":
        if re.search(r"gemma|geglu", arch, re.I):
            p.activation = "geglu"
        elif re.search(r"qwen|mistral|llama|phi", arch, re.I):
            p.activation = "swiglu"
        else:
            p.activation = "unknown"
        p.act_source = Ground.GUESS
        p.guesses.append("activation")

    # --- axis 3: ffn shape -------------------------------------------
    has_exps = any("_exps" in n for n in names)
    has_split_gate = any(n.endswith("ffn_gate.weight") for n in names)
    if has_exps:
        p.ffn = "moe"
        p.ffn_source = Ground.GROUND
        p.gate_up_fused = any("ffn_gate_up_exps" in n for n in names)
    elif has_split_gate:
        p.ffn = "dense"
        p.ffn_source = Ground.GROUND
        p.gate_up_fused = False
    else:
        p.ffn = "dense"
        p.ffn_source = Ground.GUESS
        p.guesses.append("ffn shape")
    if p.ffn == "moe":
        gu = tmap.get("blk.0.ffn_gate_up_exps.weight")
        if gu and len(gu) == 3 and p.n_experts:
            p.expert_ffn = int(gu[1]) // 2 if int(gu[1]) % 2 == 0\
                else p.expert_ffn
        if not p.n_experts_used:
            p.n_experts_used = 2
            p.warnings.append("expert_used_count absent; assuming top-2")
    elif not p.hidden_ffn and per_k:
        p.guesses.append("feed_forward_length")

    # --- axis 4: quant -----------------------------------------------
    seen = set()
    for n, _sh in tensors:
        pass
    # dtype is not in the shape table; read it from the header when the
    # suite's parser is available, else report what we can prove.
    p.quant_types = []
    p.quant_names = []

    # --- completeness -------------------------------------------------
    for req, label in ((p.n_layers, "n_layers"), (p.dim, "dim"),
                       (p.n_heads, "n_heads"), (p.head_dim, "head_dim")):
        if not req:
            p.warnings.append("could not determine %s" % label)
            p.guesses.append(label)
    if not p.trusted:
        p.warnings.append(
            "plan contains inferred fields (%s) -- verify by running before "
            "trusting" % ", ".join(sorted(set(p.guesses))))
    check_plan_against_verified_variants(p)
    return p


def apply_nvrtc_choice(plan: Plan, devices, model_bytes=0, ptx_dirs=None):
    """Fill a Plan's NVRTC/spec choice from discovered specs.

    This does NOT change the resolver's 3 checks or any geometry
    field. It only adds the engine-combo choice, using
    nightmare_nvrtc_select.choose_combo() (selection from prebuilt
    PTX only — never a PTX build, per Caleb 2026-10-28).

    devices: discovery dicts/objects (vendor, name, total, free,
        compute_cap, drivable). model_bytes: 0 when unknown.
    Returns the same plan, with nvrtc_arch / nvrtc_ptx / engine_mode /
    engine_reason / nvrtc_vendor set. Never raises: on any error the
    choice is left blank with a reason, so the router falls back
    to its existing logic instead of crashing on selection.
    """
    try:
        from nightmare_nvrtc_select import choose_combo
        combo = choose_combo(devices, model_bytes=model_bytes,
                             ptx_dirs=ptx_dirs)
        plan.nvrtc_arch = combo.get("ptx_arch") or ""
        plan.nvrtc_ptx = combo.get("ptx_path") or ""
        plan.engine_mode = combo.get("engine_mode") or ""
        plan.engine_reason = combo.get("reason") or ""
        plan.nvrtc_vendor = combo.get("vendor") or ""
    except Exception as e:
        plan.engine_mode = ""
        plan.engine_reason = ("NVRTC selection unavailable (%s: %s) - "
                              "router will use its existing fallback"
                              % (type(e).__name__, e))
    return plan


def kv_bytes_at(plan: Plan, ctx: int, f16: bool = True) -> int:
    """
    KV cache for a context length, per layer.

    The flat formula -- one layer's shape times n_layers -- is wrong on a
    hybrid model in both directions. Measured on gemma4-26B-A4B it
    over-reserved 5.6x at 32k, because it charged all 30 layers the
    widest shape and grew the sliding ones past their window.

    A sliding layer is capped at its window. That is the entire point of
    the pattern, and ignoring it is why context gets capped for no
    reason.
    """
    elem = 2 if f16 else 4
    n = max(1, plan.n_layers)
    if plan.per_layer_kdim and len(plan.per_layer_kdim) >= n:
        total = 0
        swa = plan.sliding_window
        if swa:
            # The widest layer is the full-attention one, not layer 0.
            # Comparing against per_layer_kdim[0] assumed layer 0 was
            # the reference, which on this model it is not for every
            # layer -- and on a model whose layer 0 IS the narrow one
            # the comparison inverts and caps the wrong half.
            widest = max(plan.per_layer_kdim)
        for i in range(n):
            k = plan.per_layer_kdim[i]
            v = plan.per_layer_vdim[i] if i < len(plan.per_layer_vdim) else k
            eff = ctx
            if swa and k < widest:
                # A sliding layer never needs more than its window.
                eff = min(ctx, swa)
            total += eff * (k + v) * elem
        return total
    return n * ctx * plan.n_kv_heads * plan.head_dim * elem * 2


def kv_growth_note(plan: Plan) -> str:
    """How much of the cache is actually context-sensitive."""
    if not plan.n_layers:
        return "unknown"
    a = kv_bytes_at(plan, 1024)
    b = kv_bytes_at(plan, 32768)
    if not a:
        return "unknown"
    return "32x context costs %.1fx the cache of 1k" % (b / a)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Resolve a GGUF to a plan")
    ap.add_argument("model")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--ctx", type=int, default=4096)
    args = ap.parse_args()

    try:
        plan = resolve(args.model)
    except Exception as e:
        print("RESOLVE FAILED: %s: %s" % (type(e).__name__, e))
        sys.exit(2)

    if args.json:
        print(json.dumps(plan.to_dict(), indent=1))
    else:
        print("=" * 70)
        print("PLAN  %s" % os.path.basename(args.model))
        print("=" * 70)
        print("    %s" % plan.summary())
        print("\n  kv @%d      : %.2f GB"
              % (args.ctx, kv_bytes_at(plan, args.ctx) / 1024 ** 3))
        print("  kv growth   : %s" % kv_growth_note(plan))
        for n in plan.notes:
            print("  note   : %s" % n)
        for w in plan.warnings:
            print("  WARNING: %s" % w)
        for g in sorted(set(plan.guesses)):
            print("  guess  : %s" % g)
