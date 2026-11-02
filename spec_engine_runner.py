# -*- coding: utf-8 -*-
"""
spec_engine_runner.py — Runner & Link for Brain / OpenChamber Spec-Driven Engines
====================================================================================
Wires the spec-measured C/PTX/HIP model engines generated from specs.json into
the Nightmare AI Suite.

Matched Models (Drive specs.json 2026-10-24, restored 2026-10-28):
  20 verified specs in specs.json — E4B variants (42L/2560d/8/2/10240,
  ple 10752, E-model, except Mahou Imatrix: ple None, normalised QK),
  E2B variants (35L/1536d/8/1/6144, ple 8960, E-model),
  12B variants (48L/3840d/16/8/15360, normalised QK),
  26B-A4B-It (30L/2816d/16/8/2112, MoE, normalised QK).
  1 SUSPECT entry excluded from specs.json and flagged in
  specs-suspect.json: 'Gemma 4 12B It Qat Q4_0 Unquantized Assistant'
  (gemma4-assistant, 4L/1024d, 49 tensors) — not a full 12B.

"""
import os
import sys
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any

from nightmare_engines import ArchEngine

try:
    import nightmare_hardware
    get_cpu_compute_threads = nightmare_hardware.get_cpu_compute_threads
except Exception:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        import nightmare_hardware
        get_cpu_compute_threads = nightmare_hardware.get_cpu_compute_threads
    except Exception:
        def get_cpu_compute_threads():
            return int(os.environ.get("NIGHTMARE_CPU_THREADS") or os.cpu_count() or 4)

_HERE = Path(__file__).resolve().parent
_SPECS_FILE = _HERE / "specs.json"
_FUSED_DIR = _HERE / "fused"
_NVRTC_DIR = _HERE / "nvrtc" / "ptx"
_GEN_DIR = _HERE / "gen"


def load_specs() -> list:
    """Load the measured model specs from specs.json."""
    if not _SPECS_FILE.exists():
        return []
    try:
        with open(_SPECS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[SPEC] Warning loading specs.json: {e}")
        return []


_CACHED_SPECS = load_specs()


# ---- Model-setup fix, 2026-10-28 (Caleb: "they have to be fixed right") ----
# specs.json above is the ONLY verified spec source. There is no E4B
# default anywhere in this file anymore. A model whose essential GGUF
# keys are missing is NOT synthesised as E4B: it is reported as an
# unknown setup (see _LAST_UNKNOWN_SPEC_ERROR) and find_matching_spec
# returns None, so the router can fall back to another engine instead
# of silently running the wrong geometry.
_LAST_UNKNOWN_SPEC_ERROR = ""


def get_last_unknown_spec_error():
    """Return the last unknown-model-setup error, or ''."""
    return _LAST_UNKNOWN_SPEC_ERROR


def find_matching_spec(model_path: str, metadata: Optional[Dict[str, Any]] = None) -> Optional[dict]:
    """Check if model matches one of the verified specs in specs.json.

    Pass 1 (high confidence): exact path/filename match, or a geometry
    signature (n_layers + dim) that matches a verified spec.
    Pass 2 (low confidence, name only): keyword matches are returned
    ONLY with a printed warning and match_confidence='low', and NEVER
    if supplied metadata geometry disagrees with that spec.
    If nothing matches, the JIT path is tried; if that also fails for
    a Gemma GGUF, _LAST_UNKNOWN_SPEC_ERROR names the model and the
    missing keys. Nothing here invents an E4B spec.
    """
    global _LAST_UNKNOWN_SPEC_ERROR
    _LAST_UNKNOWN_SPEC_ERROR = ""
    p_str = str(model_path).lower().replace('\\', '/')
    filename = Path(model_path).name.lower()

    # Metadata sniff, once, so geometry can gate name-only matches.
    if metadata is None and filename.endswith('.gguf'):
        try:
            import gguf_parser
            gp = gguf_parser.GGUFParser(model_path)
            metadata = gp.parse().metadata
        except Exception:
            metadata = None

    # Pass 1a: exact path / filename.
    for spec in _CACHED_SPECS:
        s_path = spec.get("path", "").lower().replace('\\', '/')
        s_filename = Path(s_path).name.lower() if s_path else ""
        s_name = spec.get("name", "").lower()
        if s_filename and s_filename in filename:
            return spec
        if s_name and s_name in p_str and len(s_name) > 12:
            # Full verified name in the path is a strong match, not a keyword.
            return spec

    # Pass 1b: geometry signature.
    if metadata:
        try:
            nl = int(metadata.get('gemma4.block_count', metadata.get('general.block_count', 0)) or 0)
            dm = int(metadata.get('gemma4.embedding_length', metadata.get('general.embedding_length', 0)) or 0)
        except Exception:
            nl, dm = 0, 0
        if nl and dm:
            for spec in _CACHED_SPECS:
                if spec.get("n_layers") == nl and spec.get("dim") == dm:
                    return spec

    # Pass 2: keyword matches, gated by geometry, marked low-confidence.
    keyword_rules = [
        ("abliterated", "e4b"), ("claude", None), ("hauhaucs", None),
        ("deckard", "v2"), ("deckard", None), ("mahou", None), ("e2b", None),
    ]
    for spec in _CACHED_SPECS:
        s_name = spec.get("name", "").lower()
        s_path = spec.get("path", "").lower()
        hit = False
        if ("abliterated" in filename and "e4b" in filename) and "abliterated" in s_name:
            hit = True
        elif "claude" in filename and "claude" in s_name:
            hit = True
        elif "hauhaucs" in filename and "hauhaucs" in s_name:
            hit = True
        elif "deckard" in filename and "v2" in filename and "v2" in s_name:
            hit = True
        elif "deckard" in filename and "v2" not in filename and "v2" not in s_name and "deckard" in s_name:
            hit = True
        elif "mahou" in filename and "mahou" in s_name:
            hit = True
        elif "e2b" in filename and "e2b" in s_path:
            hit = True
        if not hit:
            continue
        if metadata:
            try:
                nl = int(metadata.get('gemma4.block_count', metadata.get('general.block_count', 0)) or 0)
                dm = int(metadata.get('gemma4.embedding_length', metadata.get('general.embedding_length', 0)) or 0)
            except Exception:
                nl, dm = 0, 0
            if nl and dm and (spec.get("n_layers") != nl or spec.get("dim") != dm):
                print(f"[SPEC] WARNING: name match '{spec.get('name')}' for "
                      f"{filename} REJECTED — metadata geometry {nl}L x {dm}d "
                      f"disagrees with spec {spec.get('n_layers')}L x {spec.get('dim')}d")
                continue
        out = dict(spec)
        out["match_confidence"] = "low"
        out["match_reason"] = "name-only match, geometry not verified"
        print(f"[SPEC] WARNING: low-confidence name-only match for {filename} "
              f"-> {spec.get('name')} (geometry not verified)")
        return out

    # JIT path: only for a real GGUF with all essential keys present
    # (or fillable from a verified spec). Otherwise it returns None and
    # sets _LAST_UNKNOWN_SPEC_ERROR — it never synthesises E4B.
    jit_spec = jit_synthesize_spec(model_path, metadata=metadata)
    if jit_spec:
        return jit_spec
    if filename.endswith('.gguf') and 'gemma' in p_str:
        _LAST_UNKNOWN_SPEC_ERROR = (
            f"Unknown model setup for {model_path}: no verified spec in "
            f"specs.json matches it and its GGUF metadata is incomplete — "
            f"it was NOT synthesised from a guessed geometry. Add a "
            f"verified spec or fix the model metadata.")
        print(f"[SPEC] ERROR: {_LAST_UNKNOWN_SPEC_ERROR}")
    return None


def jit_synthesize_spec(model_path: str, status=print, metadata=None) -> Optional[dict]:
    """Build a spec ONLY from real GGUF metadata/tensors.

    No E4B defaults. Essential keys are block_count, embedding_length,
    attention.head_count, attention.head_count_kv, vocab size,
    feed_forward_length and context_length. If any are missing:
      - if layers+dim match a verified spec in specs.json, fill the
        missing keys from that verified spec (and say so);
      - otherwise return None with an explicit error — never fill
        them with 42/2560/8/2/10240.
    A JIT spec is in-memory only (is_draft=True) and is NEVER written
    to specs.json: only verified specs live in specs.json.
    """
    global _LAST_UNKNOWN_SPEC_ERROR
    p = Path(model_path)
    if not p.exists() or not p.name.lower().endswith('.gguf'):
        return None
    try:
        import gguf_parser
        gp = gguf_parser.GGUFParser(str(p))
        parsed = gp.parse()
        meta = parsed.metadata or {} if metadata is None else metadata
        tensors = parsed.tensors or {}

        arch = str(meta.get('general.architecture') or ('gemma4' if 'gemma4' in p.name.lower() else 'llama')).lower()
        if 'gemma' not in arch and 'gemma' not in p.name.lower():
            return None

        def _sc(v):
            if isinstance(v, (list, tuple)):
                v = v[0] if v else None
            return v if v not in (None, '') else None

        raw = {
            "n_layers": _sc(meta.get(f'{arch}.block_count', meta.get('general.block_count'))),
            "dim": _sc(meta.get(f'{arch}.embedding_length', meta.get('general.embedding_length'))),
            "vocab": _sc(meta.get(f'{arch}.vocab_size', meta.get('general.vocab_size'))),
            "n_heads": _sc(meta.get(f'{arch}.attention.head_count', meta.get('general.attention.head_count'))),
            "n_kv_heads": _sc(meta.get(f'{arch}.attention.head_count_kv', meta.get('general.attention.head_count_kv'))),
            "head_dim": _sc(meta.get(f'{arch}.attention.key_length')),
            "ffn_hidden": _sc(meta.get(f'{arch}.feed_forward_length', meta.get('general.feed_forward_length'))),
            "context": _sc(meta.get(f'{arch}.context_length', meta.get('general.context_length'))),
        }
        missing = [k for k, v in raw.items() if v is None]
        if missing:
            # Fill ONLY from a verified spec with matching layers+dim.
            filled_from = None
            try:
                nl0 = int(raw["n_layers"]) if raw["n_layers"] is not None else 0
                dm0 = int(raw["dim"]) if raw["dim"] is not None else 0
            except Exception:
                nl0, dm0 = 0, 0
            for spec in _CACHED_SPECS:
                if nl0 and dm0 and spec.get("n_layers") == nl0 and spec.get("dim") == dm0:
                    for k in missing:
                        if k in spec:
                            raw[k] = spec[k]
                    filled_from = spec.get("name")
                    break
            still = [k for k, v in raw.items() if v is None]
            if still:
                _LAST_UNKNOWN_SPEC_ERROR = (
                    f"Unknown model setup for {p.name}: missing GGUF keys "
                    f"{still} (originally missing {missing}) — not "
                    f"synthesised from a guessed geometry. Add a verified "
                    f"spec for this model or repair its metadata.")
                status(f"[SPEC-JIT] ERROR: {_LAST_UNKNOWN_SPEC_ERROR}")
                return None
            status(f"[SPEC-JIT] Filled missing keys {missing} from verified "
                   f"spec '{filled_from}' (layers/dim matched)")

        n_layers = int(raw["n_layers"]); dim = int(raw["dim"])
        vocab = int(raw["vocab"]); n_heads = int(raw["n_heads"])
        n_kv_heads = int(raw["n_kv_heads"]); head_dim = int(raw["head_dim"])
        ffn_hidden = int(raw["ffn_hidden"]); context = int(raw["context"])
        norm_eps = float(_sc(meta.get(f'{arch}.attention.layer_norm_rms_epsilon', meta.get('general.layer_norm_rms_epsilon', 1e-6))) or 1e-6)
        sliding_window = int(_sc(meta.get(f'{arch}.attention.sliding_window_length', meta.get('general.sliding_window_length', 512))) or 512)

        q_rows_by_layer = {}
        kv_rows_by_layer = {}
        for _tname, _ti in (tensors or {}).items():
            if not hasattr(_ti, "shape") or not _ti.shape:
                continue
            _m = re.match(r"blk\.(\d+)\.attn_([qkv])\.weight$", str(_tname))
            if not _m:
                continue
            _layer, _which = int(_m.group(1)), _m.group(2)
            try:
                _rows = int(_ti.shape[-1])
            except Exception:
                continue
            if _which == "q":
                q_rows_by_layer[_layer] = _rows
            else:
                kv_rows_by_layer[_layer] = _rows
        dual_map = {}
        if kv_rows_by_layer:
            _counts = {}
            for _r in kv_rows_by_layer.values():
                _counts[_r] = _counts.get(_r, 0) + 1
            _base_rows = max(_counts, key=_counts.get)
            if head_dim != _base_rows and _counts[_base_rows] > 0:
                head_dim = _base_rows
            for _l, _r in sorted(kv_rows_by_layer.items()):
                if _r != _base_rows:
                    dual_map[_l] = q_rows_by_layer.get(_l, _r)
        if dual_map:
            _pairs = ", ".join(f"{_l}:{_r}" for _l, _r in sorted(dual_map.items()))
            dual_desc = f"True  ({len(dual_map)} differ: {_pairs})"
            head_dims_desc = f"{head_dim}, varying {_base_rows}->{max(dual_map.values())}"
        else:
            dual_desc = False
            head_dims_desc = f"{head_dim}"

        ple_dim = None
        if 'per_layer_token_embd.weight' in tensors:
            sh = tensors['per_layer_token_embd.weight'].shape
            ple_dim = int(sh[0]) if len(sh) >= 1 else None
        elif f'{arch}.ple_dimension' in meta:
            ple_dim = int(meta[f'{arch}.ple_dimension'])

        model_name = str(meta.get('general.name') or p.stem.replace('-', ' ').replace('_', ' ').title())
        spec = {
            "path": str(p.resolve()), "name": model_name, "arch": arch,
            "family": "per-layer-embedding decoder (E-model)" if ple_dim else "normalised QK decoder",
            "n_layers": n_layers, "dim": dim, "vocab": vocab,
            "n_heads": n_heads, "n_kv_heads": n_kv_heads, "head_dim": head_dim,
            "head_dims_desc": head_dims_desc, "ffn_hidden": ffn_hidden,
            "quant": "GGUF", "norm_eps": norm_eps,
            "rope": "base=1e+06, swa_base=10000, dim_count=512, swa_dim_count=256",
            "context": context, "ple_dim": ple_dim, "is_draft": True,
            "features": {
                "per-layer q norm": "blk.0.attn_q_norm.weight" in tensors,
                "per-layer k norm": "blk.0.attn_k_norm.weight" in tensors,
                "per-layer v norm": "blk.0.attn_v_norm.weight" in tensors,
                "qkv fused": False, "q/k/v bias": False, "gate+up packed": False,
                "post-attention norm": "blk.0.post_attention_norm.weight" in tensors,
                "post-ffn norm": "blk.0.post_ffw_norm.weight" in tensors,
                "post-PLE norm": "blk.0.post_norm.weight" in tensors,
                "norm after matrix": True,
                "per-layer embeddings (E)": bool(ple_dim),
                "PLE input gate": bool(ple_dim),
                "PLE projection": bool(ple_dim),
                "layer output scale": "blk.0.layer_output_scale.weight" in tensors,
                "dual head_dim": dual_desc,
                "shared kv across layers": False,
                "tied embeddings": "output.weight" not in tensors,
                "rope_freqs tensor": "rope_freqs.weight" in tensors,
                "draft projections": False, "partial rotary factor": 0.25,
            },
            "tensors": len(tensors), "file_size": p.stat().st_size,
            "partial_rotary_factor": 0.25, "sliding_window": sliding_window,
        }
        status(f"[SPEC-JIT] Built in-memory (draft, NOT written to specs.json) "
               f"spec for: {model_name} ({n_layers}L x {dim}d)")
        return spec
    except Exception as e:
        status(f"[SPEC-JIT] Warning during JIT spec generation: {e}")
        return None


class SpecGemma4Engine(ArchEngine):
    """Dedicated Spec-Driven Gemma 4 Engine with verified GeGLU-tanh, split-half RoPE,
    per-layer embeddings (PLE), and dual head_dim geometry.
    """
    family = "gemma4"
    engine_name = "Gemma4 Spec-Compiled Engine"
    activation = "geglu"
    native_compatible = False

    def __init__(self, model_path, spec=None, max_seq_len=None, **kwargs):
        self.spec = spec or find_matching_spec(model_path)
        if not self.spec:
            raise ValueError(
                "Unknown model setup for %s: no verified spec in "
                "specs.json matches it (%s) — it was NOT synthesised "
                "as E4B. Add a verified spec or repair the model metadata."
                % (model_path, get_last_unknown_spec_error()))
        self.spec_name = self.spec.get("name", "Gemma 4")

        # Resolve the context BEFORE super().__init__(), because that call
        # allocates the KV cache from max_seq_len. Raising it afterwards would
        # leave buffers sized for the old value.
        #
        # The old default of 2048 was passed straight into super().__init__,
        # overriding whatever the model file declares -- so a model with
        # 262,144 context ran at 2,048 no matter what the spec said. The spec
        # value is the measured, verified one, so it wins when present.
        if max_seq_len is None:
            max_seq_len = int(self.spec.get("context", 2048)) if self.spec else 2048

        super().__init__(model_path, max_seq_len=max_seq_len, **kwargs)
        self._spec_ok = True
        self._spec_key = self.spec_name
        self._engine_tag = f"Spec Engine: {self.spec_name}"
        self._ffn_activation = "geglu"

        # Apply spec-measured geometry. Context is resolved above, before
        # super().__init__() allocates the KV cache -- raising it here would
        # leave buffers sized for the old value.
        if self.spec:
            spec_ple = self.spec.get("ple_dim", None)
            if spec_ple and not getattr(self, '_ple_dim', 0):
                nl = getattr(self, 'n_layers', 42)
                if spec_ple > 1024 and nl > 0 and spec_ple % nl == 0:
                    self._ple_dim = spec_ple // nl
                else:
                    self._ple_dim = spec_ple

        # Bind the suite's OpenMP fused kernels DLL pinned to physical cores
        self._fused_lib = None
        target_threads = get_cpu_compute_threads()
        dll_candidates = [
            _HERE / "fused" / "gemma4_fused_omp.dll",
            _HERE.parent.parent / "gemma4_fused_omp.dll",
            Path("gemma4_fused_omp.dll").resolve(),
        ]
        for p in dll_candidates:
            if p.exists():
                try:
                    import ctypes
                    os.add_dll_directory(str(p.parent))
                    lib = ctypes.CDLL(str(p))
                    lib.g4_set_num_threads(target_threads)
                    self._fused_lib = lib
                    self._fused_matvec = self._call_fused_matvec
                    self._engine_tag = f"Spec Fused Engine: {self.spec_name} (OpenMP {target_threads}T Physical)"
                    print(f"[SPEC-FUSED] ⚡ {target_threads}-Thread Physical Fused Kernels Active ({p.name})")
                    break
                except Exception as e:
                    print(f"[SPEC-FUSED] Warning binding {p.name}: {e}")

    def _call_fused_matvec(self, out, W, x, m, k):
        import ctypes
        if self._fused_lib is not None:
            self._fused_lib.g4_fused_matvec(
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                W.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(m), ctypes.c_int(k)
            )
        else:
            out[:] = W @ x

    def _layer_forward(self, x, layer, position, weights):
        return self._layer_forward_gemma4(x, layer, position, weights)

    def _needs_special_math(self):
        return False


def create_spec_engine(path: str, status=print) -> Optional[SpecGemma4Engine]:
    """Attempt to instantiate a SpecGemma4Engine if the model matches a spec."""
    spec = find_matching_spec(path)
    if spec:
        name = spec.get("name", "Gemma 4")
        target_threads = get_cpu_compute_threads()
        status(f"[SPEC-ENGINE] Matched verified engine spec: {name}")
        status(f"  Geometry: {spec.get('n_layers')}L x {spec.get('dim')}d | Heads: {spec.get('n_heads')}/{spec.get('n_kv_heads')}")
        status(f"  Fusions: GeGLU-tanh, Split-half RoPE, OpenMP {target_threads}T Physical, PLE {spec.get('ple_dim') or 'N/A'}")
        return SpecGemma4Engine(path, spec=spec)
    return None
