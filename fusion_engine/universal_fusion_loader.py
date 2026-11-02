# -*- coding: utf-8 -*-
"""
universal_fusion_loader.py — Format-agnostic model loader for Model Fusion
============================================================================
Supports: .gguf / .ntmre / .ntmr / .safetensors / .bin / .pt / .pth
CPU RAM capped at 90% via fusion_engine.mem_budget.

Entrypoint:
    from fusion_engine.universal_fusion_loader import load_model_for_fusion
"""

import gc
import sys
from pathlib import Path
from typing import Dict, List, Any

PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


try:
    from fusion_engine.mem_budget import (
        ensure_budget, file_fits_budget, allowed_bytes,
        clamp_rank_for_budget, MemoryBudgetExceeded,
    )
except Exception:  # standalone import
    from mem_budget import (  # type: ignore
        ensure_budget, file_fits_budget, allowed_bytes,
        clamp_rank_for_budget, MemoryBudgetExceeded,
    )

MAX_RAM_FRAC = 0.90

SUPPORTED_EXTENSIONS = (".gguf", ".ntmre", ".ntmr", ".safetensors", ".bin", ".pt", ".pth")


def detect_format(path: str) -> str:
    """Return one of: 'gguf', 'ntmre', 'safetensors', 'pytorch'."""
    ext = Path(path).suffix.lower()
    if ext == ".gguf":
        return "gguf"
    if ext in (".ntmre", ".ntmr"):  # .ntmr = short alias for .ntmre
        return "ntmre"
    if ext == ".safetensors":
        return "safetensors"
    if ext in (".bin", ".pt", ".pth"):
        return "pytorch"
    raise ValueError(
        "Unsupported model format '%s' for fusion. Supported: %s"
        % (ext, ", ".join(SUPPORTED_EXTENSIONS))
    )


def load_model_for_fusion(
    path: str,
    include_norms: bool = True,
    include_attn: bool = False,
    include_ffn: bool = True,
    include_embeddings: bool = True,
    max_ram_frac: float = MAX_RAM_FRAC,
) -> Dict[str, Any]:
    """Load any supported model file into the FusionEngine-ready structure."""
    fmt = detect_format(path)

    if fmt == "gguf":
        from fusion_engine.gguf_fusion_loader import load_gguf_for_fusion
        return load_gguf_for_fusion(path=path, include_attn=include_attn,
                                    max_ram_frac=max_ram_frac)

    if fmt == "ntmre":
        return _load_ntmre_for_fusion(
            path, include_norms, include_attn, include_ffn,
            include_embeddings, max_ram_frac)

    if fmt == "safetensors":
        return SafeTensorsFusionLoader().load_model_for_fusion(
            path=path, include_norms=include_norms, include_attn=include_attn,
            include_ffn=include_ffn, include_embeddings=include_embeddings,
            max_ram_frac=max_ram_frac)

    if fmt == "pytorch":
        return PyTorchFusionLoader().load_model_for_fusion(
            path=path, include_attn=include_attn, max_ram_frac=max_ram_frac)

    raise ValueError("Unhandled format '%s'." % fmt)


# ============================================================
# NTMRE LOADER (.ntmre / .ntmr) — index-first, budget-capped
# ============================================================

def _load_ntmre_for_fusion(path, include_norms=True, include_attn=False,
                           include_ffn=True, include_embeddings=True,
                           max_ram_frac=MAX_RAM_FRAC):
    from ntmre_format_py import read_ntmre
    import re

    # 1) index only — cheap, then estimate before touching weights.
    cfg, tok, index = read_ntmre(path, load_weights=False)
    total_stored = 0
    for _n, _dt, _sh, _data in index:
        # size unknown without loading; estimate from shape*4 (f32 upper bound)
        try:
            import math
            total_stored += int(math.prod(_sh)) * 4 if _sh else 4
        except Exception:
            total_stored += 0
    ensure_budget(int(total_stored * 1.3), frac=max_ram_frac,
                  what="ntmre model %s" % Path(path).name)

    # 2) load weights (format reader streams from disk; still one model at a time)
    cfg, tok, tensors = read_ntmre(path, load_weights=True)

    meta = {
        "architecture": cfg.get("arch", cfg.get("general.architecture", "llama"))
        if isinstance(cfg.get("arch", None), str) else "llama",
        "embedding_length": cfg.get("dim", 4096),
        "dim": cfg.get("dim", 4096),
        "block_count": cfg.get("n_layers", 32),
        "head_count": cfg.get("n_heads", 32),
        "head_count_kv": cfg.get("n_kv_heads", cfg.get("n_heads", 32)),
        "feed_forward_length": cfg.get("hidden_dim", 11008),
        "vocab_size": cfg.get("vocab_size", 32000),
        "context_length": cfg.get("max_seq_len", 4096),
    }
    # arch_type int -> name
    try:
        at = cfg.get("arch_type", 0)
        meta["architecture"] = {0: "llama", 1: "gemma4"}.get(int(at), "llama")
    except Exception:
        pass

    layers_dict: Dict[str, Dict[str, Any]] = {}

    for name, dtype, shape, data in tensors:
        if not data:
            continue
        lname = _ntmre_layer_name(name)
        if lname not in layers_dict:
            layers_dict[lname] = {"name": lname, "weights": {}}
        try:
            tdict = _ntmre_tensor_dict(dtype, data, shape, max_ram_frac)
        except MemoryBudgetExceeded:
            raise
        except Exception:
            continue  # skip unsupported dtype (Q4 etc.) rather than aborting
        # Filter by role like the GGUF path so flags actually do something.
        role = _ntmre_role(name)
        if role in ("ffn_gate", "ffn_up", "ffn_down") and not include_ffn:
            continue
        if role.startswith("attn_") and role not in ("attn_norm",) and not include_attn:
            continue
        if role in ("attn_norm", "ffn_norm") and not include_norms:
            continue
        if role in ("token_embd", "output_norm", "output_proj") and not include_embeddings:
            # embeddings live in their own layer; filter here too
            continue
        layers_dict[lname]["weights"][role if role else name] = tdict

    # Drop empty layers (from filtering) and sort layer_N numerically.
    def _sort_key(k):
        m = re.match(r"layer_(\d+)", k)
        if m:
            return (0, int(m.group(1)))
        return (1, k)
    layers = [layers_dict[k] for k in sorted(layers_dict.keys(), key=_sort_key)
              if layers_dict[k]["weights"]]
    gc.collect()
    return {"id": Path(path).name, "type": "llm", "meta": meta, "layers": layers}


def _ntmre_layer_name(tensor_name: str) -> str:
    import re
    m = re.search(r"(?:blk|layer)[._-]?(\d+)", tensor_name)
    if m:
        return "layer_%d" % int(m.group(1))
    if tensor_name in ("token_embd.weight", "output_norm.weight", "output.weight",
                       "token_embd", "output_norm", "output_proj"):
        return "embed_and_output"
    # bare 1-D / unprefixed tensors -> embed layer
    return "embed_and_output"


def _ntmre_role(tensor_name: str) -> str:
    n = tensor_name
    mapping = {
        "ffn_gate": "ffn_gate", "ffn_up": "ffn_up", "ffn_down": "ffn_down",
        "ffn_gate_proj": "ffn_gate", "ffn_up_proj": "ffn_up", "ffn_down_proj": "ffn_down",
        "attn_q": "attn_q", "attn_k": "attn_k", "attn_v": "attn_v",
        "attn_output": "attn_output", "wo": "attn_output",
        "attn_norm": "attn_norm", "ffn_norm": "ffn_norm",
        "token_embd": "token_embd", "output_norm": "output_norm",
        "output.weight": "output_proj", "output_proj": "output_proj",
    }
    for key, role in mapping.items():
        if key in n:
            return role
    # fallback: last dotted component without .weight
    base = n.split(".")[-1]
    return base.replace(".weight", "") or n


def _ntmre_tensor_dict(dtype, data, shape, max_ram_frac):
    """Dequant any NTMRE dtype -> {"bytes": f32, "shape": tuple}."""
    import numpy as np
    shp = tuple(int(d) for d in shape)
    n = 1
    for d in shp:
        n *= d
    ensure_budget(int(n * 4 * 1.5), frac=max_ram_frac, what="ntmre tensor")
    from ntmre_format_py import F32, F16, Q8, Q4, SVD_Q8, dequant_q8, unpack_svd
    if dtype == F32:
        arr = np.frombuffer(data, dtype=np.float32, count=n).reshape(shp).copy()
    elif dtype == F16:
        arr = np.frombuffer(data, dtype=np.float16, count=n).astype(np.float32).reshape(shp)
    elif dtype == Q8:
        arr = dequant_q8(data, n).reshape(shp)
    elif dtype == SVD_Q8:
        m, n_ = shp
        U, S, Vt = unpack_svd(data, m, n_)
        arr = ((U * S[np.newaxis, :]) @ Vt).astype(np.float32)
    elif dtype == Q4:
        arr = _dequant_q4_f16scale(data, n).reshape(shp)
    else:
        raise ValueError("unsupported ntmre dtype %s" % dtype)
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    return {"bytes": arr.tobytes(), "shape": tuple(int(d) for d in arr.shape)}


def _dequant_q4_f16scale(data, n):
    """Q4: f16 scale + packed int4 nibbles (matches ntmre_format_py Q4 layout)."""
    import numpy as np
    import struct
    out = np.zeros(n, dtype=np.float32)
    pos, i = 0, 0
    # layout: per 32-elem block -> 2B f16 scale + 16B packed nibbles
    while i < n and pos + 18 <= len(data):
        scale = struct.unpack_from("<e", data, pos)[0]
        pos += 2
        packed = data[pos:pos + 16]
        pos += 16
        for b in packed:
            for nib in (b & 0xF, (b >> 4) & 0xF):
                if i >= n:
                    break
                q = nib - 8  # int4 signed-ish centered
                out[i] = float(q) * float(scale)
                i += 1
    return out


# ============================================================
# SAFETENSORS LOADER
# ============================================================

class SafeTensorsFusionLoader:
    """Loads .safetensors via the suite's safetensors_parser.py."""

    def __init__(self):
        from safetensors_parser import SafeTensorsParser  # type: ignore
        self.SafeTensorsParser = SafeTensorsParser

    def load_model_for_fusion(
        self, path: str, include_norms: bool = True, include_attn: bool = False,
        include_ffn: bool = True, include_embeddings: bool = True,
        max_ram_frac: float = MAX_RAM_FRAC,
    ) -> Dict[str, Any]:
        ok, est, budget = file_fits_budget(path, frac=max_ram_frac, multiplier=1.4)
        if not ok:
            raise MemoryBudgetExceeded(
                "safetensors too large: %s needs ~%.2f GiB, 90%% cap leaves %.2f GiB free."
                % (Path(path).name, est / 1024 ** 3, budget / 1024 ** 3))
        parser = self.SafeTensorsParser(path)
        model = parser.parse()

        arch = model.metadata.get("general.architecture", "llama")

        def meta_get(suffix, default):
            return model.metadata.get("%s.%s" % (arch, suffix), default)

        meta = {
            "architecture": arch,
            "embedding_length": meta_get("embedding_length", 4096),
            "block_count": meta_get("block_count", 32),
            "head_count": meta_get("attention.head_count", 32),
            "head_count_kv": meta_get("attention.head_count_kv", meta_get("attention.head_count", 32)),
            "feed_forward_length": meta_get("feed_forward_length", 11008),
            "vocab_size": meta_get("vocabulary_length", 32000),
            "context_length": meta_get("context_length", 2048),
        }

        layers: List[Dict[str, Any]] = []
        for L in range(int(meta["block_count"])):
            layer_weights: Dict[str, Any] = {}
            prefix = "blk.%d" % L
            if include_ffn:
                for role, key in [
                    ("ffn_gate", "%s.ffn_gate.weight" % prefix),
                    ("ffn_up", "%s.ffn_up.weight" % prefix),
                    ("ffn_down", "%s.ffn_down.weight" % prefix),
                ]:
                    if key in model.tensors:
                        layer_weights[role] = self._tensor_dict(parser, model, key, max_ram_frac)
            if include_attn:
                for role, key in [
                    ("attn_q", "%s.attn_q.weight" % prefix),
                    ("attn_k", "%s.attn_k.weight" % prefix),
                    ("attn_v", "%s.attn_v.weight" % prefix),
                    ("attn_output", "%s.attn_output.weight" % prefix),
                ]:
                    if key in model.tensors:
                        layer_weights[role] = self._tensor_dict(parser, model, key, max_ram_frac)
            if include_norms:
                for tag, key in [
                    ("attn_norm", "%s.attn_norm.weight" % prefix),
                    ("ffn_norm", "%s.ffn_norm.weight" % prefix),
                ]:
                    if key in model.tensors:
                        layer_weights[tag] = self._tensor_dict(parser, model, key, max_ram_frac)
            if layer_weights:
                layers.append({"name": "layer_%d" % L, "weights": layer_weights})
            gc.collect()

        emb_layer_weights: Dict[str, Any] = {}
        if include_embeddings:
            for role, key in [
                ("token_embd", "token_embd.weight"),
                ("output_norm", "output_norm.weight"),
                ("output_proj", "output.weight"),
            ]:
                if key in model.tensors:
                    emb_layer_weights[role] = self._tensor_dict(parser, model, key, max_ram_frac)
        if emb_layer_weights:
            layers.append({"name": "embed_and_output", "weights": emb_layer_weights})

        parser.close()
        gc.collect()
        return {"id": Path(path).name, "type": "llm", "meta": meta, "layers": layers}

    @staticmethod
    def _tensor_dict(parser, model, key: str, max_ram_frac=MAX_RAM_FRAC) -> Dict[str, Any]:
        arr = parser.load_tensor(model, key)
        try:
            import numpy as np
            n = int(np.prod(arr.shape))
            ensure_budget(int(n * 4 * 1.5), frac=max_ram_frac, what="st tensor %s" % key)
            arr = np.ascontiguousarray(arr, dtype=np.float32)
            return {"bytes": arr.tobytes(), "shape": tuple(int(d) for d in arr.shape)}
        except MemoryBudgetExceeded:
            raise
        except Exception:
            import struct as _st
            flat = list(arr.flat) if hasattr(arr, "flat") else list(arr)
            return {"bytes": _st.pack("f" * len(flat), *[float(x) for x in flat]),
                    "shape": tuple(int(d) for d in getattr(arr, "shape", (len(flat),)))}


# ============================================================
# LEGACY PYTORCH CHECKPOINT LOADER (.bin / .pt / .pth)
# ============================================================

class PyTorchFusionLoader:
    """Best-effort loader for plain PyTorch state_dict checkpoints."""

    def load_model_for_fusion(self, path: str, include_attn: bool = False,
                              max_ram_frac: float = MAX_RAM_FRAC) -> Dict[str, Any]:
        ok, est, budget = file_fits_budget(path, frac=max_ram_frac, multiplier=2.0)
        if not ok:
            raise MemoryBudgetExceeded(
                "checkpoint too large for 90%% RAM cap: need ~%.2f GiB, have %.2f GiB free."
                % (est / 1024 ** 3, budget / 1024 ** 3))
        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "Loading .bin/.pt/.pth checkpoints requires PyTorch. "
                "Install torch, or convert the model to .gguf/.safetensors first."
            ) from e

        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            raise ValueError("Unexpected checkpoint contents in %s: expected a state_dict." % path)

        try:
            from safetensors_parser import _map_tensor_name  # type: ignore
        except Exception:
            _map_tensor_name = None

        layers_by_index: Dict[int, Dict[str, Any]] = {}
        emb_weights: Dict[str, Any] = {}
        max_layer = -1

        import re
        for hf_name, tensor in state_dict.items():
            gguf_name = _map_tensor_name(hf_name) if _map_tensor_name else hf_name
            if gguf_name is None:
                continue
            arr = tensor.detach().to(torch.float32).contiguous().cpu().numpy()
            ensure_budget(int(arr.nbytes * 1.5), frac=max_ram_frac, what="pt tensor")
            tdict = {"bytes": arr.tobytes(), "shape": tuple(int(d) for d in arr.shape)}
            del arr

            m = re.match(r"blk\.(\d+)\.(.+)\.weight$", gguf_name)
            if m:
                idx = int(m.group(1))
                role_raw = m.group(2)
                role = {
                    "ffn_gate": "ffn_gate", "ffn_up": "ffn_up", "ffn_down": "ffn_down",
                    "attn_q": "attn_q", "attn_k": "attn_k", "attn_v": "attn_v",
                    "attn_output": "attn_output", "attn_norm": "attn_norm", "ffn_norm": "ffn_norm",
                }.get(role_raw)
                if role and (include_attn or not role.startswith("attn_") or role.endswith("_norm")):
                    layers_by_index.setdefault(idx, {})[role] = tdict
                    max_layer = max(max_layer, idx)
            elif gguf_name in ("token_embd.weight", "output_norm.weight", "output.weight"):
                key = {"token_embd.weight": "token_embd",
                       "output_norm.weight": "output_norm",
                       "output.weight": "output_proj"}[gguf_name]
                emb_weights[key] = tdict
            gc.collect()

        layers = [{"name": "layer_%d" % i, "weights": layers_by_index[i]}
                  for i in sorted(layers_by_index.keys())]
        if emb_weights:
            layers.append({"name": "embed_and_output", "weights": emb_weights})
        gc.collect()
        return {"id": Path(path).name, "type": "llm",
                "meta": {"architecture": "llama",
                         "block_count": max_layer + 1 if max_layer >= 0 else 0},
                "layers": layers}
