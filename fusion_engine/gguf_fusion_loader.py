"""
gguf_fusion_loader.py — GGUF Loader Adapter for Model Fusion
=============================================================
- Uses gguf_parser.GGUFParser.load_tensor (mmap + dequant_fast) per
  selected tensor only — no full-model duplicate read.
- Returns fusion-core native {"bytes": f32, "shape": (M,N)} dicts.
- CPU RAM capped at 90% via fusion_engine.mem_budget.
"""

import gc
import sys
from pathlib import Path
from typing import Dict, List, Any


PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import dequant_fast  # noqa: E402 — block sizes for budget estimates

try:
    from fusion_engine.mem_budget import (
        ensure_budget, file_fits_budget, allowed_bytes, MemoryBudgetExceeded,
    )
except Exception:  # standalone use without package import
    from mem_budget import (  # type: ignore
        ensure_budget, file_fits_budget, allowed_bytes, MemoryBudgetExceeded,
    )

MAX_RAM_FRAC = 0.90


class GGUFFusionLoader:
    """
    Loads a GGUF model into a structured representation for fusion.

    Output format (per model):
        {
            "id": str,
            "type": "llm",
            "meta": { ... },
            "layers": [
                {"name": "layer_0", "weights": {"ffn_gate": {"bytes":..., "shape":...}, ...}},
                ...
            ]
        }
    """

    def __init__(self):
        from gguf_parser import GGUFParser  # type: ignore
        self.GGUFParser = GGUFParser

    def load_model_for_fusion(
        self,
        path: str,
        include_norms: bool = True,
        include_attn: bool = False,
        include_ffn: bool = True,
        include_embeddings: bool = True,
        max_ram_frac: float = MAX_RAM_FRAC,
    ) -> Dict[str, Any]:
        parser = self.GGUFParser(path)
        model = parser.parse()

        meta = {
            "architecture": model.architecture,
            "embedding_length": model.embedding_length,
            "block_count": model.block_count,
            "head_count": model.head_count,
            "head_count_kv": model.head_count_kv_scalar,
            "feed_forward_length": model.feed_forward_length,
            "vocab_size": model.vocab_size,
            "context_length": model.context_length,
        }

        # Pre-check: file size vs 90% RAM cap (dequant inflates quants to f32).
        ok, est, budget = file_fits_budget(path, frac=max_ram_frac, multiplier=1.6)
        if not ok:
            parser.close()
            raise MemoryBudgetExceeded(
                "GGUF too large for CPU budget: %s needs ~%.2f GiB, "
                "90%% RAM cap leaves %.2f GiB free. Disable attn/embeddings "
                "or fuse smaller models." % (
                    Path(path).name, est / 1024 ** 3, budget / 1024 ** 3))

        names = set(model.tensors.keys())

        def pick(*cands):
            for c in cands:
                if c in names:
                    return c
            return None

        # Resolve the single common prefix style once (not per layer).
        probe = next(iter(names))
        common_prefix = "blk." if "blk." in probe else (
            "model.layers." if "model.layers." in probe else "blk.")

        layers: List[Dict[str, Any]] = []

        def load_one(key: str) -> Dict[str, Any]:
            info = model.tensors[key]
            shape = getattr(info, "shape", ())
            import math as _m
            ne = int(_m.prod(shape)) if shape else 1
            # need: f32 output + temp headroom
            ensure_budget(int(ne * 4 * 1.5), frac=max_ram_frac,
                          what="gguf tensor %s" % key)
            arr = parser.load_tensor(model, key)  # float32 ndarray, correct shape
            try:
                import numpy as _np
                arr = _np.asarray(arr, dtype=_np.float32)
                shp = tuple(int(d) for d in arr.shape)
                # Fusion core only fuses 2D; keep 1D norms as (N,) — core skips them.
                return {"bytes": arr.tobytes(), "shape": shp}
            except Exception:
                import struct as _st
                flat = [float(x) for x in arr.flat] if hasattr(arr, "flat") else [float(x) for x in arr]
                return {"bytes": _st.pack("f" * len(flat), *flat),
                        "shape": tuple(int(d) for d in getattr(arr, "shape", (len(flat),)))}

        for L in range(int(meta["block_count"])):
            p = "%s%d" % (common_prefix, L) if "blk." in common_prefix or "model.layers." in common_prefix else "blk.%d" % L
            # normalize: gguf names are "blk.N.xxx"
            if common_prefix.startswith("model.layers"):
                p = "model.layers.%d" % L
            else:
                p = "blk.%d" % L
            layer_weights: Dict[str, Any] = {}

            if include_ffn:
                k = pick("%s.ffn_gate_proj.weight" % p, "%s.ffn_gate.weight" % p,
                         "%s.ffn_up_proj.weight" % p, "%s.ffn_up.weight" % p)
                # gate and up are separate tensors; resolve each role independently
                kg = pick("%s.ffn_gate_proj.weight" % p, "%s.ffn_gate.weight" % p)
                ku = pick("%s.ffn_up_proj.weight" % p, "%s.ffn_up.weight" % p)
                # legacy files sometimes pack gate+up naming differently; keep both roles
                if kg:
                    layer_weights["ffn_gate"] = load_one(kg)
                if ku and ku != kg:
                    # store up-proj under ffn_up when distinct; else gate already covers it
                    layer_weights["ffn_up"] = load_one(ku)
                elif k and "ffn_gate" not in layer_weights:
                    layer_weights["ffn_gate"] = load_one(k)
                kd = pick("%s.ffn_down_proj.weight" % p, "%s.ffn_down.weight" % p)
                if kd:
                    layer_weights["ffn_down"] = load_one(kd)

            if include_attn:
                for role, key in [
                    ("attn_q", "%s.attn_q.weight" % p),
                    ("attn_k", "%s.attn_k.weight" % p),
                    ("attn_v", "%s.attn_v.weight" % p),
                    ("attn_output", "%s.attn_output.weight" % p),
                ]:
                    if key in names:
                        layer_weights[role] = load_one(key)

            if include_norms:
                for key in ["%s.attn_norm.weight" % p, "%s.ffn_norm.weight" % p]:
                    if key in names:
                        tag = "attn_norm" if "attn" in key else "ffn_norm"
                        layer_weights[tag] = load_one(key)

            if layer_weights:
                layers.append({"name": "layer_%d" % L, "weights": layer_weights})
            # Keep RSS under 90%: drop numpy temporaries each layer.
            gc.collect()

        emb_layer_weights: Dict[str, Any] = {}
        if include_embeddings:
            for k in ["token_embd.weight"]:
                if k in names:
                    emb_layer_weights["token_embd"] = load_one(k)
            for k in ["output_norm.weight", "model.norm.weight"]:
                if k in names:
                    emb_layer_weights["output_norm"] = load_one(k)
                    break
            for k in ["output.weight"]:
                if k in names:
                    emb_layer_weights["output_proj"] = load_one(k)

        if emb_layer_weights:
            layers.append({"name": "embed_and_output", "weights": emb_layer_weights})

        parser.close()
        gc.collect()

        return {"id": Path(path).name, "type": "llm", "meta": meta, "layers": layers}

    # ---------- legacy helpers kept for compat ----------

    def _tensor_byte_size(self, tinfo: Any, nelements: int) -> int:
        dtype = tinfo.dtype
        if dtype in dequant_fast.BLOCK_SIZES:
            block_elems, block_bytes = dequant_fast.BLOCK_SIZES[dtype]
            blocks = (nelements + block_elems - 1) // block_elems
            return blocks * block_bytes
        return nelements * 4


def load_gguf_for_fusion(path: str, include_attn: bool = False,
                         max_ram_frac: float = MAX_RAM_FRAC) -> Dict[str, Any]:
    """Primary entrypoint from the GUI panel / universal loader."""
    loader = GGUFFusionLoader()
    return loader.load_model_for_fusion(
        path=path, include_norms=True, include_attn=include_attn,
        include_ffn=True, include_embeddings=True, max_ram_frac=max_ram_frac)
