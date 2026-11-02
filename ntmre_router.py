# -*- coding: utf-8 -*-
"""
ntmre_router.py — universal model sniffer & engine router
==========================================================
Sniffs any model file (.ntmre / .gguf) and instantiates the correct
Nightmare Suite engine for it.  No filename heuristics — the decision
is made from the actual file contents:

  .gguf   -> GGUF metadata arch (general.architecture)
  .ntmre  -> NSVD/NTMR/RMTN JSON container  -> arch from JSON config
             NTMRE v16 binary (RMTN v16)     -> v16 engine
             NTMRE binary v1/v2 (RMTN)       -> C DLL runtime

Engines (all suite's own):
  qwen35 / qwen3next hybrid  -> Qwen35HybridEngine (nightmare_qwen35)
  llama / qwen2 / gemma ...  -> GPUStreamEngine (nightmare_stream)
  qwen35 GGUF Q1_0           -> BonsaiSuiteModel (bonsai_suite)
  NTMRE binary               -> NtmreRuntime (ntmre_runtime, C DLL)

Author: Nightmare Designs 💀
"""
import os
import sys
import json
import struct
import time
from pathlib import Path


# ── NTMRE container versions ───────────────────────────────────────────────
# Canonical/final version is 13 (see ntmre_format_py). These are the versions
# this build still READS; writers emit 13 only.
RMTN_V13 = 13
RMTN_READ_VERSIONS = (1, 2, 13, 16)



if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Spec-driven model engines & dedicated family engines
try:
    from engine.model_engines import spec_engine_runner
except Exception:
    try:
        import spec_engine_runner
    except Exception:
        spec_engine_runner = None

try:
    import nightmare_engines
except Exception:
    nightmare_engines = None

try:
    import model_bridge
except Exception:
    model_bridge = None

# Extra folders where models may reside (dynamically resolved per user)
EXTRA_MODEL_DIRS = [
    str(Path.home() / "Downloads"),
    str(Path.home() / ".lmstudio" / "models"),
    str(Path.home() / ".cache" / "lm-studio" / "models"),
    str(Path.home() / ".ollama" / "models"),
    str(Path.home() / ".cache" / "huggingface" / "hub"),
    r"I:\downloads",
    r"I:\models",
]
if os.environ.get("NIGHTMARE_MODEL_DIRS"):
    for _d in os.environ["NIGHTMARE_MODEL_DIRS"].split(os.pathsep):
        if _d.strip():
            EXTRA_MODEL_DIRS.append(_d.strip())

_QWEN35_ARCHES = ('qwen35', 'qwen3next', 'qwen3_next')


# ════════════════════════════════════════════════════════════════
# SHORTCUT RESOLUTION & IMAGE DIFFUSION FILTERING
# ════════════════════════════════════════════════════════════════

IMAGE_DIFFUSION_KEYWORDS = (
    'sdxl', 'sd_xl', 'sd-xl',
    'stablediffusion', 'stable-diffusion', 'stable_diffusion',
    'sd14', 'sd1.4', 'sd15', 'sd1.5', 'sd20', 'sd2.0', 'sd21', 'sd2.1', 'sd3', 'sd3.5',
    'flux', 'flux1', 'flux.1',
    'pony', 'illustrious',
    'diffusion', 'diffusers',
    'pornmaster', 'dreamshaper', 'deliberate', 'realistic_vision', 'cyberrealistic',
    'controlnet', 'inpainting', 'lora', 'svd-xt', 'pixart', 'auraflow',
    'hunyuan', 'kolors', 'mochi', 'cogvideo', 'ltx-video', 'wan2', 'chroma'
)

# "svd" on its own is NOT an image-model keyword here. It was matching
# Stable Video Diffusion, but this suite's own container format is called
# .ntmre compression, so a bare 'svd' substring rejected every NTMRE .ntmre compression file by name
# (e.g. "svd.ntmre", "qwen_svd_q8.ntmre") and refused to route it. Only
# the explicit Stable Video Diffusion spellings count.
IMAGE_DIFFUSION_SVD_KEYWORDS = (
    'stable-video-diffusion', 'stable_video_diffusion',
    'stablevideodiffusion', 'svd-xt', 'svd_xt',
)

_SHORTCUT_CACHE = {}

# discover_models() runs on every UI refresh but walks all drives; cache its
# result for this many seconds so refreshes are instant.
_DISCOVER_CACHE = None
_DISCOVER_TTL = 20.0

# Directories never worth walking when hunting for model files.  Hidden dirs
# are NOT skipped wholesale because LM Studio (.lmstudio/models) and
# HuggingFace (.cache/huggingface) store real models under dotted names.
_WALK_SKIP_DIRS = {
    'windows', 'program files', 'program files (x86)', 'node_modules',
    '$recycle.bin', 'appdata', 'programdata', 'windowsapps',
    'system volume information', 'perflogs', 'msocache',
    '.git', '.svn', '.hg', '.venv', 'venv', '.idea', '.vscode',
    '__pycache__', '.mypy_cache', '.pytest_cache', 'site-packages',
}


def resolve_shortcut_target(path) -> str:
    """Resolve a .lnk Windows shortcut to its actual target file path."""
    p_str = str(path)
    if not p_str.lower().endswith('.lnk'):
        return p_str
    if p_str in _SHORTCUT_CACHE:
        return _SHORTCUT_CACHE[p_str]
    resolved = None
    if sys.platform == 'win32':
        try:
            import subprocess
            cmd = f"(New-Object -ComObject WScript.Shell).CreateShortcut('{Path(p_str).resolve()}').TargetPath"
            res = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd],
                capture_output=True, text=True, timeout=3
            )
            out = res.stdout.strip()
            if out and os.path.exists(out):
                resolved = out
        except Exception:
            pass
    if not resolved:
        resolved = p_str
    _SHORTCUT_CACHE[p_str] = resolved
    return resolved


def resolve_shortcuts_batch(lnk_paths):
    """Resolve many .lnk files in ONE PowerShell invocation.

    The per-shortcut resolver spun up a PowerShell process per file (~88 ms
    each).  A model folder with ~60 shortcuts therefore cost several seconds
    on every discover_models() call, which runs at UI startup.  This resolves
    the whole batch in a single process and populates _SHORTCUT_CACHE."""
    todo = [str(p) for p in lnk_paths
            if str(p).lower().endswith('.lnk') and str(p) not in _SHORTCUT_CACHE]
    if not todo or sys.platform != 'win32':
        return
    try:
        import subprocess
        lines = []
        for i, p in enumerate(todo):
            lines.append(
                "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%s'); "
                "Write-Output ('%d' + [char]9 + $s.TargetPath)"
                % (Path(p).resolve().as_posix().replace("'", "''"), i))
        script = "; ".join(lines)
        res = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=30)
        got = {}
        for line in (res.stdout or '').splitlines():
            if '\t' in line:
                idx, tgt = line.split('\t', 1)
                try:
                    got[int(idx.strip())] = tgt.strip()
                except ValueError:
                    pass
        for i, p in enumerate(todo):
            tgt = got.get(i)
            _SHORTCUT_CACHE[p] = tgt if (tgt and os.path.exists(tgt)) else p
    except Exception:
        for p in todo:
            _SHORTCUT_CACHE.setdefault(p, p)


def is_image_diffusion_model(path_or_name) -> bool:
    """Detect if a model file is an SDXL/Stable Diffusion/Flux image model rather than an LLM."""
    resolved_path = resolve_shortcut_target(path_or_name)
    name_low = os.path.basename(str(resolved_path)).lower()

    if any(k in name_low for k in IMAGE_DIFFUSION_KEYWORDS):
        return True
    # Stable Video Diffusion, but never a bare "svd" (that is our own
    # format name -- see IMAGE_DIFFUSION_SVD_KEYWORDS).
    if any(k in name_low for k in IMAGE_DIFFUSION_SVD_KEYWORDS):
        return True

    if name_low.endswith('.gguf') and os.path.isfile(resolved_path):
        try:
            with open(resolved_path, 'rb') as f:
                head = f.read(16384).lower()
            if any(k in head for k in (b'diffusion', b'sdxl', b'flux', b'sd3', b'model.diffusion_model')):
                return True
        except Exception:
            pass
    return False


# Architectures that are real GGUF models but are NOT autoregressive chat
# models.  Loading these into the chat engine produced an empty reply with no
# error at all (nomic-embed-text -> logits all NaN, nsfw_wan_14b -> no
# tokenizer), which reads as "the suite is broken" rather than "wrong panel".
_NON_CHAT_ARCHS = {
    'nomic-bert', 'bert', 'jina-bert', 'roberta', 'minilm',
    'wan', 'moe', 't5', 'mt5', 'umt5', 'clip', 'rn-bert',
}
_NON_CHAT_HINT = {
    'nomic-bert': 'an embedding model (use the RAG/embeddings panel)',
    'bert': 'an encoder/embedding model, not a chat model',
    'jina-bert': 'an embedding model, not a chat model',
    'roberta': 'an encoder model, not a chat model',
    'wan': 'a video diffusion model (use the image/video panel)',
    't5': 'an encoder-decoder model, not a chat model',
    'mt5': 'an encoder-decoder model, not a chat model',
    'clip': 'a vision encoder, not a chat model',
}


def non_chat_model_reason(path):
    """Return why `path` cannot be used by the chat engine, else None.

    Cheap: peeks only the GGUF header, never a tensor payload.
    """
    resolved = resolve_shortcut_target(path)
    if not resolved or not os.path.isfile(resolved):
        return None
    if not resolved.lower().endswith('.gguf'):
        return None
    try:
        with open(resolved, 'rb') as f:
            head = f.read(4)
        if head != b'GGUF':
            return None
        import gguf_parser
        g = gguf_parser.GGUFParser(resolved)
        m = g.parse()
        arch = str(m.metadata.get('general.architecture', '')).lower()
        tensors = set(m.tensors or {})
        g.close()
        # A causal LM has per-block attention + FFN projections. Anything
        # without them cannot be run by the transformer engines at all.
        has_blocks = any(t.startswith('blk.') for t in tensors)
        if arch in _NON_CHAT_ARCHS:
            return f"'{os.path.basename(resolved)}' is " + _NON_CHAT_HINT.get(
                arch, f"a '{arch}' model, which is not a chat model")
        if tensors and not has_blocks:
            return (f"'{os.path.basename(resolved)}' (arch '{arch}') has no "
                    "transformer blocks - it is not a language model")
    except Exception:
        return None
    return None


_NON_CHAT_NAME_PATTERNS = (
    'nomic-embed', 'nomic_embed', 'umt5', 'clip-vit', 'mmproj',
    # Adapter/projector/ViT companions that ship next to a real GGUF but are
    # not standalone chat models (some earlier scans listed them).
    'mtp.gguf', 'cursor_projector', 'text_encoder', 'vision_encoder',
    'ggml-vocab', 'vae.', 'vae_', 'lora',
)


def _looks_non_chat_by_name(path) -> bool:
    """Cheap filename-only pre-filter for the model list.

    discover_models() runs on every UI refresh, so it must not parse GGUF
    headers; non_chat_model_reason() (which does) is the authoritative check
    and runs at load time. This only keeps the obvious offenders out of the
    chat dropdown.
    """
    name = os.path.basename(str(path)).lower()
    return any(k in name for k in _NON_CHAT_NAME_PATTERNS)


def sniff_model(path):
    """Inspect a model file and return a dict describing it:
        kind:   'gguf' | 'nsvd' | 'v16' | 'binary' | 'unknown'
        arch:   architecture string ('qwen35', 'llama', ...) or ''
        config: parsed JSON config (nsvd) or {} 
        error:  human-readable problem, or None
    """
    path = resolve_shortcut_target(path)
    info = {'kind': 'unknown', 'arch': '', 'config': {}, 'error': None}
    try:
        with open(path, 'rb') as f:
            head = f.read(64)
    except OSError as e:
        info['error'] = str(e)
        return info

    if len(head) < 8:
        info['error'] = 'file too small'
        return info

    magic = head[:4]

    # ONNX: detect by extension
    if str(path).lower().endswith('.onnx'):
        info['kind'] = 'onnx'
        try:
            import onnx_parser
            op = onnx_parser.ONNXParser()
            tensors = op.parse(str(path))
            info['arch'] = 'onnx'
            info['dtypes'] = sorted({str(t.dtype) for t in tensors.values()})
            op.close()
        except Exception as e:
            info['error'] = f'ONNX parse failed: {e}'
        return info

    # SafeTensors: no magic number — detect by extension + header validation
    if str(path).lower().endswith('.safetensors'):
        info['kind'] = 'safetensors'
        try:
            import safetensors_parser
            sp = safetensors_parser.SafeTensorsParser(path)
            m = sp.parse()
            info['arch'] = str(m.metadata.get('general.architecture', 'llama')).lower()
            info['config'] = m.metadata
            info['dtypes'] = sorted({t.dtype for t in m.tensors.values()})
            sp.close()
        except Exception as e:
            info['error'] = f'SafeTensors parse failed: {e}'
        return info

    if magic == b'GGUF':
        info['kind'] = 'gguf'
        try:
            import gguf_parser
            g = gguf_parser.GGUFParser(path)
            m = g.parse()
            info['arch'] = str(m.metadata.get('general.architecture', 'llama')).lower()
            info['config'] = m.metadata
            info['dtypes'] = sorted({t.dtype for t in m.tensors.values()})
        except Exception as e:
            info['error'] = f'GGUF parse failed: {e}'
        return info

    if magic in (b'NSVD', b'NTMR', b'RMTN'):
        version = struct.unpack_from('<I', head, 4)[0]
        # NTMRE v16 binary: RMTN magic, version 16 (or v1 in v16 layout)
        if magic == b'RMTN' and version in RMTN_READ_VERSIONS:
            from ntmre_v16_engine import is_v16_file
            if is_v16_file(path):
                info['kind'] = 'v16'
                return info
        # JSON container: u32 n_tensors @8, u32 cfg_len @12, JSON @16
        if len(head) >= 17 and head[16:17] == b'{':
            info['kind'] = 'nsvd'
            try:
                cfg_len = struct.unpack_from('<I', head, 12)[0]
                with open(path, 'rb') as f:
                    f.seek(16)
                    cfg = json.loads(f.read(cfg_len).decode('utf-8'))
                info['config'] = cfg
                info['arch'] = str(cfg.get('arch', cfg.get('general.architecture', ''))).lower()
                # truncation check: header + all tensor data must fit
                n_tensors = struct.unpack_from('<I', head, 8)[0]
                fsize = os.path.getsize(path)
                with open(path, 'rb') as f:
                    f.seek(16 + cfg_len)
                    tok_len = struct.unpack('<I', f.read(4))[0]
                    f.seek(tok_len, 1)
                    idx_len = struct.unpack('<I', f.read(4))[0]
                    idx = f.read(idx_len)
                p = 0
                total = 0
                for _ in range(n_tensors):
                    nl = idx[p]; p += 1 + nl
                    nd = idx[p + 1]; p += 2 + 4 * nd
                    total += struct.unpack_from('<Q', idx, p)[0]; p += 8
                data_start = (16 + cfg_len + 4 + tok_len + 4 + idx_len + 63) & ~63
                if data_start + total > fsize:
                    info['error'] = ('file is truncated (conversion did not finish): '
                                     f'needs {data_start + total >> 20} MB, '
                                     f'only {fsize >> 20} MB on disk')
            except Exception as e:
                info['error'] = f'NTMRE container parse failed: {e}'
            return info
        # otherwise: binary NTMRE (C engine format, RMTN v1)
        info['kind'] = 'binary'
        if magic == b'RMTN' and version in (1, 2, 13):
            _v1_integrity_check(path, info)
        return info

    info['error'] = f'unrecognized magic: {magic!r}'
    return info


def _v1_integrity_check(path, info):
    """RMTN v1 (C-layout) files: the converter writes a zeroed placeholder
    index, streams tensor data, then seeks back to fill the index.  If the
    run was interrupted the index stays all-zero and the file can never
    load.  Detect that here and say exactly what happened."""
    try:
        fsize = os.path.getsize(path)
        with open(path, 'rb') as f:
            f.seek(568)
            tok_size = struct.unpack('<I', f.read(4))[0]
            vocab = struct.unpack('<I', f.read(4))[0]
            if not (0 < vocab <= 10_000_000 and 0 < tok_size < fsize):
                return  # not the C layout we know; leave as plain binary
            # tokenizer: vocab x (u32 len + bytes + f32 score), then u32 n_merges
            o = 572 + 4
            for _ in range(vocab):
                f.seek(o)
                ln = struct.unpack('<I', f.read(4))[0]
                if ln > 1 << 20:
                    return
                o += 4 + ln + 4
            f.seek(o)
            n_merges = struct.unpack('<I', f.read(4))[0]
            o += 4
            for _ in range(min(n_merges, 10_000_000)):
                f.seek(o)
                ln = struct.unpack('<I', f.read(4))[0]
                if ln > 1 << 20:
                    return
                o += 4 + ln
            f.seek(o)
            n_tensors = struct.unpack('<I', f.read(4))[0]
            if not (0 < n_tensors <= 200_000):
                return
            # only the start of the index: tensor data follows it and IS
            # non-zero, so never read past a few KB here
            idx = f.read(min(4096, fsize - o - 4))
        if idx and not any(idx):
            info['error'] = (
                'conversion incomplete: the tensor index was never written '
                f'({n_tensors} tensor slots are all zero).  The converter '
                'writes the index last, so this run was interrupted.  '
                'Re-convert from the source GGUF (e.g. ntmre_convert_gpu.exe) '
                'and keep the source GGUF until the .ntmre loads.')
    except Exception:
        pass  # sniffing must never crash


# Known Gemma 4 architecture variants (exact names + hyphen/underscore prefixes).
# Add new variants here instead of using a blind startswith.
_GEMMA4_ARCHES = (
    'gemma4',
    'gemma4-assistant',
    'gemma4_mtp',
    'gemma4-assistant-mtp',
)

# Architectures that need special engines (not the generic GPU-native/streaming path)
_SPECIAL_ARCHS = _QWEN35_ARCHES + _GEMMA4_ARCHES + ('clip',)

def _is_gemma4_variant(arch):
    a = str(arch).lower()
    return any(a == base or a.startswith(base + '-') or a.startswith(base + '_')
               for base in _GEMMA4_ARCHES)

def _is_special_arch(arch):
    a = str(arch).lower()
    return _is_gemma4_variant(arch) or any(a == s for s in _SPECIAL_ARCHS)


def _is_fused_arch(arch):
    """Whitelist for the fused dual-GPU engine (dense, full RoPE, SiLU,
    RMSNorm families).  Families with special kernels use their own engine."""
    try:
        from nightmare_fused_gpu import _FUSED_ARCHS
    except Exception:
        return False
    a = str(arch).lower()
    return any(a == base or a.startswith(base + '-') or a.startswith(base + '_')
               for base in _FUSED_ARCHS)


def _try_fused(path, arch, status, plan=None):
    """Fused resident engine — layers split across the accelerators the
    planner chose.  When `plan` is supplied it is authoritative: if the
    planner says the model fits on one card we do NOT split across two just
    because a second card exists (the cross-device hop per layer boundary
    costs more than the extra parallelism for a small model)."""
    if os.environ.get('NIGHTMARE_FORCE_CPU') == '1':
        return None
    try:
        from nightmare_gpu_discovery import has_gpu_accelerator
        if not has_gpu_accelerator():
            return None
    except Exception:
        pass
    arch = str(arch).lower()
    if not arch or _is_special_arch(arch):
        return None
    if not _is_fused_arch(arch):
        return None

    # Honour the plan's device count.
    want = None
    if plan is not None:
        eng = getattr(plan, "engine", None)
        if eng == "cpu":
            return None
        if eng == "gpu_native":
            return None        # single card is enough; not a fused problem
        n = len(getattr(plan, "devices", []) or [])
        want = max(2, n) if n else None

    try:
        from nightmare_gpu_discovery import usable_accelerators
        accs = usable_accelerators(limit=want or 2)
        if not accs:
            return None
        label = f"{len(accs)} GPU" + ("s" if len(accs) > 1 else "")
        if want and len(accs) < want:
            label += f" (wanted {want})"
        from nightmare_fused_gpu import FusedDualGpuEngine
        status(f"[FUSED] Routing to Fused Engine ({label}, full-resident, {arch})...")
        return FusedDualGpuEngine(path)
    except Exception as e:
        status(f"  Fused engine unavailable ({e}), falling back...")
        return None


def _try_gpu_native(path, arch, status, plan=None):
    """Try the full-VRAM GPU-native engine. Falls back to None if the model
    doesn't fit in VRAM, the quant type has no GPU kernel, or CUDA is
    unavailable."""
    if os.environ.get('NIGHTMARE_FORCE_CPU') == '1':
        return None
    try:
        from nightmare_gpu_discovery import has_gpu_accelerator
        if not has_gpu_accelerator():
            return None
    except Exception:
        pass
    if _is_special_arch(arch):
        return None  # gemma4 has its own engine; qwen35 has its own engine
    # Respect each family engine's declared GPU-native compatibility.  The
    # generic GPU-native kernel implements only SwiGLU + full rotary + RMSNorm;
    # GELU/partial-rope/LayerNorm families (gpt2, phi2, starcoder2, falcon,
    # mpt, bloom, gptj, ...) mark native_compatible=False and must not be
    # routed here or they KeyError on ffn_gate / produce wrong math.
    if nightmare_engines is not None:
        try:
            if not nightmare_engines.native_compatible(arch):
                return None
        except Exception:
            pass
    # The planner may have decided this model needs more VRAM than one card
    # can give (plan.engine == 'stream'), in which case full-resident on a
    # single device is the wrong choice.
    if plan is not None and getattr(plan, "engine", None) == "stream":
        return None
    # NVRTC selection (2026-10-28, admin copy): an AMD-only or CPU
    # plan must not be routed to an NVIDIA GPU-native engine. The
    # planner already sets engine="stream" for AMD, but check the
    # NVRTC vendor/mode here too so a stale plan cannot misroute.
    if plan is not None:
        _vendor = getattr(plan, "nvrtc_vendor", "") or ""
        _mode = getattr(plan, "engine_mode", "") or ""
        if _vendor == "amd" or _mode in ("amd_hiprtc", "cpu"):
            status("  GPU-Native skipped: %s"
                   % (getattr(plan, "engine_reason", "") or _mode))
            return None
        _ptx = getattr(plan, "nvrtc_ptx", "") or ""
        if _ptx:
            status("  [NVRTC] selected prebuilt PTX: %s (sm%s) - %s"
                   % (_ptx, getattr(plan, "nvrtc_arch", "") or "?",
                      getattr(plan, "engine_reason", "") or ""))
    try:
        from nightmare_gpu_native import GpuNativeEngine
        status(f"[GPU] Routing to GPU-Native Engine (full VRAM, no numpy)...")
        return GpuNativeEngine(path)
    except RuntimeError as e:
        msg = str(e)
        if ('VRAM' in msg or 'CUDA' in msg or 'streaming' in msg
                or 'quant' in msg.lower() or 'no kernel' in msg.lower()):
            status(f"  GPU-Native unavailable ({msg[:160]}), "
                   "falling back to streaming...")
            return None
        raise
    except Exception as e:
        status(f"  GPU-Native failed ({e}), falling back to streaming...")
        return None


def _try_gemma4_gpu_native(path, status, nvrtc_ptx=None, nvrtc_arch=None,
                             nvrtc_reason=None):
    """Try the Gemma4 full-VRAM GPU-native engine (NVRTC, no nvcc).

    SELECTION ONLY (2026-10-28, admin copy): nvrtc_ptx, when given,
    is a PREBUILT PTX file already in nvrtc-kernels/ (selected by
    the planner from the person's specs). This function never
    builds PTX at load time — building PTX from .cu is hard to
    auto-compute and is an offline developer job (nvrtc_build.c),
    not part of the load path (Caleb 2026-10-28).
    """
    if os.environ.get('NIGHTMARE_FORCE_CPU') == '1':
        return None
    if nvrtc_ptx:
        status("  [NVRTC] selected prebuilt PTX: %s (sm%s) - %s"
               % (nvrtc_ptx, nvrtc_arch or "?",
                  nvrtc_reason or ""))
    try:
        from nightmare_gpu_discovery import has_gpu_accelerator
        if not has_gpu_accelerator():
            return None
    except Exception:
        pass
    try:
        # Verify the file is an RMTN v1 binary before passing to the C DLL
        with open(path, 'rb') as f:
            head = f.read(16)
        if len(head) < 8 or head[:4] != b'RMTN':
            return None
        version = struct.unpack_from('<I', head, 4)[0]
        if version not in (1, 2, 13):
            return None
        from ntmre_gemma4_gpu import Gemma4GpuEngine
        status(f"[LOAD] Routing to Gemma4 GPU-Native Engine (full VRAM, NVRTC)...")
        return Gemma4GpuEngine(path)
    except RuntimeError as e:
        msg = str(e)
        if 'VRAM' in msg or 'CUDA' in msg or 'failed' in msg.lower() or 'not found' in msg.lower():
            status(f"  Gemma4 GPU-Native unavailable ({e}), falling back to streaming...")
            return None
        raise
    except Exception as e:
        status(f"  Gemma4 GPU-Native failed ({e}), falling back to streaming...")
        return None


def _try_gemma4_spec(path, status):
    """Route Gemma 4 models to the verified spec-compiled engines from the verified spec-compiled engines."""
    if spec_engine_runner is None:
        return None
    try:
        eng = spec_engine_runner.create_spec_engine(path, status=status)
        if eng is not None:
            return eng
    except Exception as e:
        status(f"  Spec engine init failed ({e}), falling back...")
    return None


def _try_dedicated_family(path, arch, status):
    """Route to one of the 22 dedicated family engines (Gemma2/3, Mistral, Phi2,
    DeepSeek, Falcon, StarCoder, etc.) from nightmare_engines."""
    if nightmare_engines is None:
        return None
    try:
        cls = nightmare_engines.engine_class_for(arch)
        if cls is not None and cls is not nightmare_engines.GenericFallbackEngine:
            name = nightmare_engines.engine_name_for(arch)
            status(f"[LOAD] Routing to Dedicated Engine: {name} ({cls.__name__})...")
            return cls(path)
    except Exception as e:
        status(f"  Dedicated engine unavailable ({e}), falling back...")
    return None


def _read_rmtn_v1_arch(path):
    """Read the architecture tag from an RMTN v1 file before routing it."""
    from ntmre_v1_engine import V1Source
    source = V1Source(path)
    try:
        return str(source.config.get('arch', 'llama')).lower()
    finally:
        source.close()


# ─────────────────────────────────────────────────────────────────────────────
# System planning: profile the machine once, then pick the best combination.
# ─────────────────────────────────────────────────────────────────────────────

_PROFILE_CACHE = None


def system_profile(force=False):
    """Cached SystemProfile for this machine."""
    global _PROFILE_CACHE
    if _PROFILE_CACHE is None or force:
        try:
            from system_planner import SystemProfile
            _PROFILE_CACHE = SystemProfile.probe(force=force)
        except Exception:
            _PROFILE_CACHE = None
    return _PROFILE_CACHE


def plan_for(path, info=None, force_cpu=False, max_devices=None):
    """Ask the planner which engine + devices this model should use."""
    try:
        from system_planner import plan_for_model
        prof = system_profile()
        if prof is None:
            return None
        try:
            size = os.path.getsize(path)
        except OSError:
            size = 0
        n_layers = int((info or {}).get('n_layers', 0) or 0)
        arch = str((info or {}).get('arch', ''))
        return plan_for_model(prof, size, n_layers=n_layers, arch=arch,
                              force_cpu=force_cpu, max_devices=max_devices)
    except Exception:
        return None


def nvrtc_selection_for(devices, model_bytes=0):
    """Spec-based NVRTC selection for external callers (Suite app, tests).

    Returns the nightmare_nvrtc_select.choose_combo() dict for the
    given discovery devices and model size. Selection from prebuilt
    PTX only — never a PTX build. Returns a CPU fallback dict with
    a reason if the selector is unavailable, instead of raising.
    """
    try:
        from nightmare_nvrtc_select import choose_combo
        return choose_combo(devices, model_bytes=model_bytes)
    except Exception as e:
        return {
            "vendor": "none", "engine_mode": "cpu", "engine": "cpu",
            "ptx_path": None, "ptx_arch": None,
            "reason": "NVRTC selection unavailable (%s: %s)" % (type(e).__name__, e),
            "devices_used": [], "dual_gpu": False, "fits": False,
        }


def _model_has_gpu(eng):
    """True if `eng` actually holds weights/buffers on an accelerator.

    The old router trusted the route it picked: _try_fused printed
    "[FUSED] Routing to Fused Engine" and returned an engine that had
    already fallen back to CPU -- which is why the failure was invisible.
    """
    if getattr(eng, "_fused_ok", False):
        return True
    if getattr(eng, "_fused_devs", None):
        return True
    if getattr(eng, "_gpu_workers", None):
        return True
    if getattr(eng, "_gpumm", None) is not None:
        return True
    if getattr(eng, "use_gpu", False):
        return True
    return False


def probe_logits(eng, status=None):
    """Run one real forward pass and judge whether the output is USABLE.

    Hardware verification is not enough.  An engine can engage a GPU, print
    "fully resident", hit no exception, and still emit an all-zero or
    NaN logit vector -- which decodes to arbitrary tokens with no warning.
    That is exactly the failure this suite shipped for months.

    Returns (ok, detail).  ok is False only for output that is provably
    degenerate: all NaN/inf, or constant across the whole vocabulary.
    A merely low-confidence distribution is fine -- small models really do
    produce flat logits on some prompts.
    """
    try:
        import nightmare_np as _np
    except Exception:
        return True, "numpy unavailable - skipped"
    try:
        fwd = getattr(eng, "forward", None)
        if fwd is None:
            return True, "engine has no forward() - skipped"
        tok = 0
        bos = getattr(eng, "bos_id", None)
        if isinstance(bos, int) and bos > 0:
            tok = bos
        lg = _np.asarray(fwd(int(tok), 0)).ravel()
    except Exception as e:
        return True, "probe could not run (%s: %s) - skipped" % (
            type(e).__name__, e)

    if lg.size == 0:
        return False, "forward() returned an empty logit vector"

    finite = _np.isfinite(lg)
    if not finite.any():
        return False, "every logit is NaN/inf"
    if not finite.all():
        return False, "%d of %d logits are NaN/inf" % (
            int((~finite).sum()), int(lg.size))

    good = lg[finite]
    spread = float(good.max() - good.min())
    std = float(good.std())
    if spread == 0.0 and std == 0.0:
        return False, ("all %d logits are exactly identical (constant "
                       "vector) - the forward pass produced no signal"
                       % int(lg.size))

    p = _np.exp(good - good.max())
    p /= p.sum()
    ent = float(-(p * _np.log(p + 1e-30)).sum())
    return True, "ok  n=%d  std=%.4f  range=%.3f  top1p=%.4f  entropy=%.2f" % (
        int(lg.size), std, spread, float(p.max()), ent)


def verify_engine(eng, plan=None, status=print):
    """Report loudly if the chosen engine did not get the planned hardware,
    or if it engaged hardware but produced degenerate output."""
    want_gpu = os.environ.get('NIGHTMARE_FORCE_CPU') != '1'
    if plan is not None and getattr(plan, "engine", "cpu") == "cpu":
        want_gpu = False

    hardware_ok = True
    if want_gpu and not _model_has_gpu(eng):
        status(f"[PLAN] WARNING: routed to {type(eng).__name__} but NO "
               "accelerator was engaged - running on CPU. Check the "
               "kernel/VRAM messages above.")
        try:
            eng._nightmare_cpu_fallback = True
        except Exception:
            pass
        hardware_ok = False

    # Second gate: engaging the GPU is not the same as producing output.
    ok, detail = probe_logits(eng)
    if ok:
        if status is not None and os.environ.get('NIGHTMARE_VERBOSE_VERIFY') == '1':
            status("[PLAN] output probe: %s" % detail)
    else:
        status(f"[PLAN] SEVERE: {type(eng).__name__} produced degenerate "
               f"output - {detail}. Tokens from this engine are meaningless. "
               "This is a silent-corruption failure, not a slow path.")
        try:
            eng._nightmare_degenerate_output = True
        except Exception:
            pass
        return False
    return hardware_ok


def _derive_ctx_for(plan, path, info):
    """
    Derive this model's context ceiling on THIS machine.

    Divides the memory this machine can actually give to a KV cache by
    the measured cost of one token, then holds the result to the
    model's own trained length.

    Every input is measured:
      n_layers / n_kv_heads / head_dim   from the model's TENSORS
      trained context                    from the model header
      free VRAM + host RAM               from the system profile

    Nothing is a default. If the geometry cannot be read, this returns
    None and the caller says so rather than substituting a number --
    a silent fallback is the exact defect this replaces.
    """
    try:
        import engine_resolver as ER
        plan_er = ER.resolve(path)
    except Exception:
        plan_er = None

    n_layers = 0
    kv_width = 0
    trained = 0
    kv_pixels = 0
    if plan_er is not None:
        n_layers = int(getattr(plan_er, 'n_layers', 0) or 0)
        trained = int(getattr(plan_er, 'context', 0) or 0)

    # KV width per layer, summed over all layers, taken from the
    # ATTENTION TENSORS.
    #
    # This is fiddlier than it looks and three different derivations
    # gave three different answers, so the reasoning is recorded:
    #
    #   * the header's gemma4.attention.key_length says 512
    #   * the K tensors are (dim, 2048), which with 8 kv heads reads
    #     as head_dim 256
    #   * but the ENGINE allocates n_layers*seq*kv_dim*2*2 with
    #     kv_dim = n_kv_heads * head_dim, and its own log reports
    #     491,520 bytes/token -- which is 8 * 512, i.e. head_dim 512
    #
    # So the engine is sizing KV from the header's key_length, and the
    # tensors describe something else (the packed projection, not the
    # per-head cache width). Deriving from the tensors understated the
    # cost by exactly 2x, which doubled every ceiling computed from it.
    #
    # The engine's allocation is the thing that must not overflow, so
    # this matches the engine rather than the metadata: kv_dim = 4096
    # reproduces the measured bytes/token exactly.
    try:
        import engine_resolver as _ER2
        _meta, _tens, _n = _ER2.read_header(path)
        _d = dict(_tens)
        n_kv = 0
        for _l in range(max(1, n_layers)):
            _k = _d.get('blk.%d.attn_k.weight' % _l)
            if _k and len(_k) == 2 and n_kv == 0:
                n_kv = int(getattr(plan_er, 'n_kv_heads', 0) or 0)
        # head_dim from the engine's own header field, which is what the
        # engine uses when it allocates. Fall back to the tensor only if
        # the header is absent.
        hd = int(_meta.get('%s.attention.key_length' %
                           (str(_meta.get('%s.architecture' % 'x', '')) or 'x'), 0) or 0)
        if not hd:
            for _k2, _v2 in _meta.items():
                if _k2.endswith('.attention.key_length'):
                    hd = int(_v2 or 0)
                    break
        if n_kv and hd:
            kv_pixels = n_kv * hd
        else:
            kv_pixels = 0
    except Exception:
        kv_pixels = 0

    if not (n_layers and kv_pixels):
        return None, "could not measure KV width", 0

    # K and V, F16, for every layer.
    per_token = n_layers * kv_pixels * 2 * 2
    if per_token <= 0:
        return None, "bad per-token cost", 0

    # Budget against the LARGEST SINGLE CARD, not the sum of all of them.
    #
    # This was summing every card's usable VRAM and then adding half the
    # host RAM, which produced 64,816 on a 3060 Ti + 3060 pair. That is
    # the wrong question: _best_kv_vram() places every layer of the cache
    # on ONE device, and nightmare_stream.py allocates it eagerly with
    # np.zeros() at load. So the ceiling describes a single allocation and
    # must be divided by a single card.
    #
    # Summing made it worse than useless -- it tripled a budget that was
    # already ~3x too large, and the result could not be allocated at all:
    #
    #     ctx 64816 -> K 15.93 GB + V 15.93 GB = 31.86 GB of zeros
    #     machine   -> 31.9 GB RAM, of which 15.78 GB is model weights
    #
    # So the cache wanted more RAM than the machine had free once the
    # model was resident. It thrashed to the pagefile and produced no
    # tokens at all.
    #
    # Host RAM is NOT added here. It is a fallback, not a bonus: if no
    # card can hold the cache, the engine spills to host memory, and at
    # that point the ceiling is whatever RAM is left AFTER the weights
    # load. Counting it as extra capacity is what made the number
    # unreachable in the first place.
    best_gb = 0.0
    try:
        from system_planner import VRAM_HEADROOM
        for d in (plan.devices or []):
            g = d.usable_gb(VRAM_HEADROOM)
            if g > best_gb:
                best_gb = g
    except Exception:
        pass

    basis_extra = ""
    if best_gb <= 0:
        # No drivable card. Fall back to half of host RAM, because a CPU
        # or single-unknown-device run genuinely has to spill there.
        try:
            import psutil
            best_gb = (psutil.virtual_memory().available / 1024 ** 3) * 0.5
            basis_extra = " (no drivable card; host RAM fallback)"
        except Exception:
            pass
    if best_gb <= 0:
        return None, "no free memory reported", 0

    by_mem = int((best_gb * 1024 ** 3) // per_token)
    if trained and by_mem > trained:
        return int(trained), ("model's trained limit %d; largest card could hold %d"
                              % (trained, by_mem)), int(per_token)
    return max(512, by_mem), ("%.1f GB on largest card / %d B per token%s"
                              % (best_gb, per_token, basis_extra)), int(per_token)


def apply_plan(plan, status=print):
    """Push the plan's device + CPU-thread decisions into the process."""
    if plan is None:
        return
    try:
        if plan.devices:
            # Ascending vendor-local index order keeps CUDA ordinals aligned
            # with the physical cards, so gpu_info.get_free_vram(n) describes
            # the same card the context was actually created on.
            devs = sorted(plan.devices, key=lambda d: d.index)
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                str(d.index) for d in devs)
        from system_planner import apply_cpu_settings
        if apply_cpu_settings(plan):
            status(f"[PLAN] Pinned to {plan.cpu_threads} physical cores "
                   "(SMT siblings excluded)")
    except Exception as e:
        status(f"[PLAN] could not apply settings: {e}")


def _create_engine_routed(path, force_cpu=False, status=print):
    """Routing body. Use create_engine() -- it adds post-route verification."""
    path = resolve_shortcut_target(path)
    _nonchat = non_chat_model_reason(path)
    if _nonchat:
        raise ValueError(_nonchat)
    if is_image_diffusion_model(path):
        raise ValueError(
            f"'{Path(path).name}' is an Image Diffusion / SDXL model. "
            "It belongs in the Picture Generator panel, not the LLM chat engine."
        )
    info = sniff_model(path)
    if info['error']:
        raise ValueError(info['error'])
    kind = info['kind']
    arch = info['arch']

    # Profile the machine and decide the target combination up front.
    _plan = plan_for(path, info, force_cpu=force_cpu)
    if _plan is not None:
        status("[PLAN] " + _plan.describe().replace("\n", "\n       "))
        apply_plan(_plan, status)

    # Publish the plan so every engine built below picks it up.
    #
    # The streaming engine asks for it when deciding how much context
    # this machine can hold. Without this the context ceiling is a
    # typed-in 12000 that is wrong on both a small box and a large one,
    # and someone running this on their own hardware has no way to see
    # that the number was a guess rather than a measurement.
    #
    # Derived HERE because only this point has both the model and the
    # system profile: the engine's own tensors give the per-token KV
    # cost, and the profile gives the free memory to divide by it.
    if _plan is not None:
        try:
            _ctx, _basis, _pertok = _derive_ctx_for(_plan, path, info or {})
            if _ctx and _pertok:
                _plan.max_ctx = int(_ctx)
                _plan.ctx_basis = _basis
                # The same measurement, kept as a byte budget so the
                # engine's own KV fitting reads the plan instead of
                # re-deriving from a typed default and disagreeing.
                _plan.kv_bytes_per_token = int(_pertok)
                _plan.kv_budget_bytes = int(_ctx * _pertok)
                status("[PLAN] context %d (%s)" % (int(_ctx), _basis))
        except Exception as _ce:
            status("[PLAN] context not derived: %s" % _ce)
    try:
        import nightmare_stream as _ns
        _ns.PLAN_FOR_NEXT_LOAD = _plan
    except Exception:
        pass

    # RMTN v1 carries its authoritative architecture in the fixed header.
    # Resolve Gemma4 here before any generic or Qwen-specific route can see it.
    if kind == 'binary':
        try:
            arch_v1 = _read_rmtn_v1_arch(path)
        except Exception:
            arch_v1 = ''
        if arch_v1 == 'gemma4':
            eng = _try_gemma4_spec(path, status)
            if eng is not None:
                return eng
            if force_cpu:
                from nightmare_stream import GPUStreamEngine
                status("[LOAD] Routing Gemma4 RMTN v1 to CPU-compatible streaming engine...")
                return GPUStreamEngine(path)
            eng = _try_gemma4_gpu_native(
                path, status,
                nvrtc_ptx=(getattr(_plan, 'nvrtc_ptx', '') or None) if _plan is not None else None,
                nvrtc_arch=(getattr(_plan, 'nvrtc_arch', '') or None) if _plan is not None else None,
                nvrtc_reason=(getattr(_plan, 'engine_reason', '') or None) if _plan is not None else None)
            if eng is not None:
                return eng
            from nightmare_stream import GPUStreamEngine
            status("[LOAD] Routing Gemma4 RMTN v1 to Streaming Engine...")
            return GPUStreamEngine(path)

    if kind == 'gguf':
        if arch in _QWEN35_ARCHES and not force_cpu:
            dtypes = set(info.get('dtypes', []))
            if dtypes and dtypes <= {0, 1, 41}:   # F32/F16 + Q1_0 only
                from bonsai_suite import BonsaiSuiteModel
                status(f"[LOAD] Routing to Bonsai Qwen3.5 Hybrid Engine (Q1_0 GPU)...")
                return BonsaiSuiteModel(path)
            from nightmare_qwen35 import Qwen35HybridEngine
            status(f"[LOAD] Routing to Qwen3.5 Hybrid Engine (NTMRE math, GGUF source)...")
            return Qwen35HybridEngine(path)
        if _is_gemma4_variant(arch):
            eng = _try_gemma4_spec(path, status)
            if eng is not None:
                return eng
        if force_cpu:
            os.environ['NIGHTMARE_FORCE_CPU'] = '1'
            eng = _try_dedicated_family(path, arch, status)
            if eng is not None:
                return eng
            from nightmare_stream import GPUStreamEngine
            status("[LOAD] Routing to CPU Streaming Engine...")
            return GPUStreamEngine(path)
        os.environ['NIGHTMARE_FORCE_CPU'] = '0'
        # Fused dual-GPU (full-resident) when two GPUs are drivable
        eng = _try_fused(path, arch, status, _plan)
        if eng is not None:
            return eng
        # Try full-VRAM GPU-native first (fastest — no numpy, no streaming)
        eng = _try_gpu_native(path, arch, status, _plan)
        if eng is not None:
            return eng
        # Try dedicated per-family engine (exact activation, partial RoPE, logit softcap)
        eng = _try_dedicated_family(path, arch, status)
        if eng is not None:
            return eng
        # Fall back to streaming (for models too big for VRAM)
        from nightmare_stream import GPUStreamEngine
        status(f"[LOAD] Routing to Streaming Engine (layer-by-layer, for big models)...")
        return GPUStreamEngine(path)

    if kind == 'nsvd':
        if arch in _QWEN35_ARCHES:
            from nightmare_qwen35 import Qwen35HybridEngine
            status(f"[LOAD] Routing to Qwen3.5 Hybrid Engine (NTMRE SSM+Attention)...")
            return Qwen35HybridEngine(path)
        if _is_gemma4_variant(arch):
            eng = _try_gemma4_spec(path, status)
            if eng is not None:
                return eng
            from nightmare_stream import GPUStreamEngine
            status(f"[LOAD] Routing to NTMRE Streaming Engine (gemma4 dual-attention)...")
            return GPUStreamEngine(path)
        if arch == 'mlp':
            raise ValueError('This .ntmre is a small test artifact (arch=mlp), '
                             'not a chat model.')
        # llama and other classic arches in an NTMRE container
        # Fused dual-GPU (full-resident) when two GPUs are drivable
        eng = _try_fused(path, arch, status, _plan)
        if eng is not None:
            return eng
        # Try full-VRAM GPU-native first
        eng = _try_gpu_native(path, arch, status, _plan)
        if eng is not None:
            return eng
        eng = _try_dedicated_family(path, arch, status)
        if eng is not None:
            return eng
        from nightmare_stream import GPUStreamEngine
        status(f"[LOAD] Routing to NTMRE Streaming Engine ({arch or 'llama'})...")
        return GPUStreamEngine(path)

    if kind == 'onnx':
        # ONNX: convert to NTMRE first, then load
        status("[LOAD] Converting ONNX → NTMRE .ntmre compression...")
        try:
            from nightmare_convert import convert_model
            ntmre_path = str(Path(path).with_suffix('.ntmre'))
            convert_model(Path(path), Path(ntmre_path))
            status(f"[OK] Converted → {ntmre_path}")
            return create_engine(ntmre_path, status=status, force_cpu=force_cpu)
        except Exception as e:
            raise RuntimeError(f'ONNX conversion failed: {e}')

    if kind == 'safetensors':
        if force_cpu:
            raise ValueError('CPU-only mode not yet supported for SafeTensors')
        os.environ['NIGHTMARE_FORCE_CPU'] = '0'
        # Try full-VRAM GPU-native first (fastest)
        eng = _try_gpu_native(path, arch, status, _plan)
        if eng is not None:
            return eng
        eng = _try_dedicated_family(path, arch, status)
        if eng is not None:
            return eng
        # Fall back to streaming (for large unquantized models)
        from nightmare_stream import GPUStreamEngine
        status(f"[LOAD] Routing to Streaming Engine (SafeTensors, layer-by-layer)...")
        return GPUStreamEngine(path)

    if kind == 'v16':
        from ntmre_v16_engine import create_v16_engine
        status("[LOAD] Routing to NTMRE v16 Engine...")
        return create_v16_engine(path, status=status)

    if kind == 'binary':
        # v1 C-layout binaries: the prebuilt DLL predates this layout and
        # crashes on it — the suite's own Python reader handles them.
        from ntmre_v1_engine import V1Source, is_v1_readable
        if is_v1_readable(path):
            # Detect the real architecture from the v1 binary header
            # (arch_type 1 = gemma4).  Gemma4 must use the streaming engine
            # because the GPU-native engine only handles standard dense
            # transformers; llama-layout binaries can try full-VRAM first.
            arch_v1 = 'llama'
            try:
                arch_v1 = _read_rmtn_v1_arch(path)
            except Exception:
                pass
            if arch_v1 == 'gemma4':
                eng = _try_gemma4_spec(path, status)
                if eng is not None:
                    return eng
                eng = _try_gemma4_gpu_native(
                    path, status,
                    nvrtc_ptx=(getattr(_plan, 'nvrtc_ptx', '') or None) if _plan is not None else None,
                    nvrtc_arch=(getattr(_plan, 'nvrtc_arch', '') or None) if _plan is not None else None,
                    nvrtc_reason=(getattr(_plan, 'engine_reason', '') or None) if _plan is not None else None)
                if eng is not None:
                    return eng
                from nightmare_stream import GPUStreamEngine
                status("[LOAD] Routing to NTMRE Streaming Engine (gemma4 v1 binary)...")
                return GPUStreamEngine(path)
            eng = _try_fused(path, arch_v1, status, _plan)
            if eng is not None:
                return eng
            eng = _try_gpu_native(path, arch_v1, status, _plan)
            if eng is not None:
                return eng
            eng = _try_dedicated_family(path, arch_v1, status)
            if eng is not None:
                return eng
            from nightmare_stream import GPUStreamEngine
            status("[LOAD] Routing to NTMRE Streaming Engine (v1 binary)...")
            return GPUStreamEngine(path)
        # If the file IS a fixed-slot v1 binary but damaged (truncated or
        # never-finished conversion), surface that exact reason instead of
        # falling into the legacy DLL's confusing failure.
        try:
            V1Source(path)
        except Exception as ve:
            msg = str(ve)
            if 'truncated' in msg or 'never written' in msg:
                raise ValueError(msg)
        from ntmre_runtime import NtmreRuntime
        eng = NtmreRuntime(path)
        if not getattr(eng, 'loaded', False):
            raise RuntimeError('NTMRE binary engine failed to load model')
        return eng

    raise ValueError(f"Unsupported model file: {path}")


def create_engine(path, force_cpu=False, status=print):
    """Create the right engine for a model file, then VERIFY it got the
    hardware the planner intended.

    Every route goes through here so a silent CPU fallback can never again
    be reported as a successful GPU load.
    """
    target = resolve_shortcut_target(path)
    plan = plan_for(target, force_cpu=force_cpu)
    eng = _create_engine_routed(path, force_cpu=force_cpu, status=status)
    try:
        verify_engine(eng, plan=plan, status=status)
    except Exception:
        pass
    return eng


def describe_system(status=print):
    """Human-readable capability report for this machine."""
    prof = system_profile(force=True)
    if prof is None:
        status("[PLAN] system profiling unavailable")
        return None
    status("[PLAN] System profile:\n       " +
           prof.describe().replace("\n", "\n       "))
    status("       devices:\n       " +
           prof.device_summary().replace("\n", "\n       "))
    return prof


def create_shortcut(lnk_path, target_path) -> bool:
    """Create a Windows .lnk shortcut pointing to target_path."""
    if sys.platform != 'win32':
        return False
    try:
        import subprocess
        lnk_str = str(Path(lnk_path).resolve())
        tgt_str = str(Path(target_path).resolve())
        ps_cmd = f"$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('{lnk_str}'); $s.TargetPath = '{tgt_str}'; $s.Save()"
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd], capture_output=True, timeout=5)
        _SHORTCUT_CACHE[lnk_str] = tgt_str
        return Path(lnk_str).exists()
    except Exception:
        return False


def create_shortcuts_batch(links) -> int:
    """Create multiple Windows .lnk shortcuts in a single fast PowerShell invocation."""
    if not links or sys.platform != 'win32':
        return 0
    needed = []
    for lnk, tgt in links:
        lnk_p = Path(lnk).resolve()
        tgt_p = Path(tgt).resolve()
        _SHORTCUT_CACHE[str(lnk_p)] = str(tgt_p)
        if not lnk_p.exists():
            needed.append((str(lnk_p), str(tgt_p)))
    if not needed:
        return 0
    try:
        import subprocess
        commands = ["$ws = New-Object -ComObject WScript.Shell;"]
        for lnk_str, tgt_str in needed:
            commands.append(f"$s = $ws.CreateShortcut('{lnk_str}'); $s.TargetPath = '{tgt_str}'; $s.Save();")
        ps_cmd = " ".join(commands)
        subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd], capture_output=True, timeout=10)
        return sum(1 for lnk_str, _ in needed if Path(lnk_str).exists())
    except Exception:
        return 0


def discover_models(suite_dir):
    """Find all loadable text LLM models: suite models/, user downloads, desktop, documents,
    and common LLM storage directories (LM Studio, Ollama, HuggingFace).
    Strictly excludes Image Diffusion models (SDXL, SD3, Flux, etc.).
    Returns {display_name: full_path}.

    Walks every drive (bounded depth) so a library stored outside the named
    folders - e.g. D:\\New folder\\... or G:\\ntmre_backup\\... - is still
    listed.  Results are cached for _DISCOVER_TTL seconds because this runs on
    every UI refresh."""
    global _DISCOVER_CACHE
    _now = time.time()
    _ck = str(suite_dir)
    if _DISCOVER_CACHE and _DISCOVER_CACHE[0] == _ck and (_now - _DISCOVER_CACHE[1]) < _DISCOVER_TTL:
        return dict(_DISCOVER_CACHE[2])

    found = {}
    seen_paths = set()

    suite_path = Path(suite_dir).resolve()
    search_dirs = [
        suite_path / 'models',
        suite_path,
        Path.home() / 'Downloads',
        Path.home() / 'Documents' / 'models',
        Path.home() / 'Desktop',
        Path.home() / '.lmstudio' / 'models',
        Path.home() / '.cache' / 'lm-studio' / 'models',
        Path.home() / '.ollama' / 'models',
        Path.home() / '.cache' / 'huggingface' / 'hub',
    ]
    # Add any extra configured directories
    for d in EXTRA_MODEL_DIRS:
        search_dirs.append(Path(d))

    # Also pick up drive-root AI folders (e.g. E:\models, E:\.lmstudio\models),
    # AND the drive roots themselves (bounded walk), so a library outside the
    # named folders is still found.  Without this a model only showed up if a
    # shortcut already existed in suite/models/, so a newly-added drive-level
    # model was invisible until the user ran a full scan.
    _extra_scan = []          # dirs to walk recursively (bounded depth)
    _drive_roots = []
    if sys.platform == 'win32':
        try:
            import ctypes as _ct
            _bitmask = _ct.windll.kernel32.GetLogicalDrives()
            for _letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                if _bitmask & 1:
                    _d = Path(f"{_letter}:\\")
                    try:
                        if _d.exists():
                            _drive_roots.append(_d)
                            for _name in ('models', 'ai', 'llm', 'llms', 'gguf', 'lmstudio'):
                                _cand = _d / _name
                                try:
                                    if _cand.is_dir():
                                        _extra_scan.append(_cand)
                                except Exception:
                                    pass
                            _lm = _d / '.lmstudio' / 'models'
                            try:
                                if _lm.is_dir():
                                    _extra_scan.append(_lm)
                            except Exception:
                                pass
                    except Exception:
                        pass
                _bitmask >>= 1
        except Exception:
            pass

    def _iter_model_files(_dir, _recursive):
        """Yield candidate model files in `_dir`.  Drive-level AI folders are
        walked (bounded depth) because models often sit one folder down, e.g.
        E:\\models\\<repo>\\model.gguf."""
        try:
            if not _recursive:
                for _pat in ('*.gguf', '*.ntmre', '*.safetensors', '*.onnx', '*.lnk'):
                    for _p in sorted(_dir.glob(_pat), key=lambda x: x.name):
                        yield _p
                return
            _maxdepth = 5 if str(_dir) == _dir.anchor else 3
            for _root, _dirs, _files in os.walk(_dir):
                # Skip OS/vendor noise, but NOT dot-directories in general:
                # LM Studio and HuggingFace legitimately store models under
                # hidden dirs like '.lmstudio/models' and '.cache/...'.
                _dirs[:] = [x for x in _dirs
                            if x.lower() not in _WALK_SKIP_DIRS and not x.startswith('$')]
                try:
                    if len(Path(_root).relative_to(_dir).parts) >= _maxdepth:
                        _dirs[:] = []
                except Exception:
                    pass
                for _f in _files:
                    if _f.lower().endswith(('.gguf', '.ntmre', '.safetensors', '.onnx', '.lnk')):
                        yield Path(_root) / _f
        except Exception:
            return

    # Collect (candidate, label-context) pairs first so every .lnk is
    # resolved in ONE batched PowerShell call, not one process per file.
    _cands = []
    for d in search_dirs:
        try:
            if not d.is_dir():
                continue
            for p in _iter_model_files(d, False):
                _cands.append((p, d, False))
        except Exception:
            continue
    for d in _extra_scan:
        try:
            if not d.is_dir():
                continue
            for p in _iter_model_files(d, True):
                _cands.append((p, d, True))
        except Exception:
            continue
    for d in _drive_roots:
        try:
            if not d.is_dir():
                continue
            for p in _iter_model_files(d, True):
                _cands.append((p, d, True))
        except Exception:
            continue
    resolve_shortcuts_batch([p for (p, _d, _x) in _cands])

    for p, d, _recursive in _cands:
        try:
            target_path = resolve_shortcut_target(p)
            if not os.path.exists(target_path):
                continue
            if not target_path.lower().endswith(('.gguf', '.ntmre', '.safetensors', '.onnx')):
                continue
            res_path = str(Path(target_path).resolve())
            if res_path in seen_paths:
                continue

            # STRICT EXCLUSION: Image diffusion models belong exclusively in Picture Generator
            if is_image_diffusion_model(res_path):
                continue
            # ...as are embedding encoders and video diffusion models.
            if _looks_non_chat_by_name(res_path):
                continue

            seen_paths.add(res_path)

            low = Path(res_path).name.lower()
            if 'mmproj' in low or low.endswith('-mmp.gguf'):
                continue

            clean_name = p.name[:-4] if p.name.lower().endswith('.lnk') else p.name
            if d == suite_path / 'models' or d == suite_path:
                label = clean_name
            elif _recursive:
                label = f"{clean_name}  ({Path(res_path).parent.name})"
            else:
                label = f"{clean_name}  ({d.name})"
            found[label] = res_path
        except Exception:
            continue
    _DISCOVER_CACHE = (_ck, _now, dict(found))
    return found



def discover_picture_models(suite_dir=None):
    """Find all image diffusion models (SDXL, SD3, Flux, .ntmre compression, etc.) across
    the suite, downloads, pictures, desktop, and known AI folders.
    Returns {display_name: full_path}."""
    found = {}
    seen_paths = set()
    if suite_dir is None:
        suite_path = Path(__file__).parent.resolve()
    else:
        suite_path = Path(suite_dir).resolve()

    search_dirs = [
        suite_path / 'models',
        suite_path,
        Path.home() / 'Downloads',
        Path.home() / 'Pictures',
        Path.home() / 'Desktop',
        Path.home() / 'Documents' / 'models',
        Path.home() / '.cache' / 'lm-studio' / 'models',
        Path.home() / '.lmstudio' / 'models',
        Path.home() / 'stable-diffusion-webui' / 'models',
        Path.home() / 'ComfyUI' / 'models',
    ]
    # Walk every drive (bounded) so image models stored outside the named
    # folders are still found, exactly like discover_models() does for LLMs.
    if sys.platform == 'win32':
        try:
            import ctypes as _ct
            _bm = _ct.windll.kernel32.GetLogicalDrives()
            for _letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                if _bm & 1:
                    _d = Path(f"{_letter}:\\")
                    try:
                        if _d.exists():
                            search_dirs.append(_d)
                    except Exception:
                        pass
                _bm >>= 1
        except Exception:
            pass

    def _iter_pic_files(_dir):
        try:
            if str(_dir) == _dir.anchor:
                for _root, _dirs, _files in os.walk(_dir):
                    _dirs[:] = [x for x in _dirs
                                if x.lower() not in _WALK_SKIP_DIRS and not x.startswith('$')]
                    try:
                        if len(Path(_root).relative_to(_dir).parts) >= 5:
                            _dirs[:] = []
                    except Exception:
                        pass
                    for _f in _files:
                        if _f.lower().endswith(('.gguf', '.ntmre', '.safetensors', '.lnk')):
                            yield Path(_root) / _f
            else:
                for _pat in ('*.gguf', '*.ntmre', '*.safetensors', '*.lnk'):
                    for _p in sorted(_dir.glob(_pat), key=lambda x: x.name):
                        yield _p
        except Exception:
            return

    _pic_cands = []
    for d in search_dirs:
        try:
            if not d.is_dir():
                continue
            for p in _iter_pic_files(d):
                _pic_cands.append((p, d))
        except Exception:
            continue
    resolve_shortcuts_batch([c[0] for c in _pic_cands])

    for p, d in _pic_cands:
        try:
            target_path = resolve_shortcut_target(p)
            if not os.path.exists(target_path):
                continue
            if not target_path.lower().endswith(('.gguf', '.ntmre', '.safetensors')):
                continue
            res_path = str(Path(target_path).resolve())
            if res_path in seen_paths:
                continue

            if is_image_diffusion_model(res_path):
                seen_paths.add(res_path)
                clean_name = p.name[:-4] if p.name.lower().endswith('.lnk') else p.name
                if d == suite_path / 'models' or d == suite_path:
                    label = clean_name
                else:
                    label = f"{clean_name}  ({d.name})"
                found[label] = res_path
        except Exception:
            continue
    return found


def full_system_quick_scan(suite_dir=None, create_shortcuts=True):
    """Rapidly scan the system for LLMs and Image models:
    - Finds all .gguf, .ntmre, .safetensors models across all drives and AI directories.
    - Excludes image diffusion models from LLM list (routes them to picture list).
    - If create_shortcuts=True: generates .lnk shortcuts in suite models/ folder for text LLMs.
    Returns:
        (llm_models: dict[label, path], picture_models: dict[label, path], num_shortcuts_created: int)
    """
    if suite_dir is None:
        suite_path = Path(__file__).parent.resolve()
    else:
        suite_path = Path(suite_dir).resolve()

    models_dir = suite_path / 'models'
    models_dir.mkdir(exist_ok=True)

    drives = []
    if sys.platform == 'win32':
        try:
            import ctypes
            bitmask = ctypes.windll.kernel32.GetLogicalDrives()
            for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
                if bitmask & 1:
                    d = f"{letter}:\\"
                    if os.path.exists(d):
                        drives.append(Path(d))
                bitmask >>= 1
        except Exception:
            drives = [Path("C:\\")]
    else:
        drives = [Path("/")]

    search_dirs = [
        models_dir,
        suite_path,
        Path.home() / 'Downloads',
        Path.home() / 'Desktop',
        Path.home() / 'Documents',
        Path.home() / 'Documents' / 'models',
        Path.home() / '.lmstudio' / 'models',
        Path.home() / '.cache' / 'lm-studio' / 'models',
        Path.home() / '.ollama' / 'models',
        Path.home() / '.cache' / 'huggingface' / 'hub',
        Path.home() / 'Pictures',
        Path.home() / 'stable-diffusion-webui' / 'models',
        Path.home() / 'ComfyUI' / 'models',
    ]
    for d in EXTRA_MODEL_DIRS:
        search_dirs.append(Path(d))

    ai_folder_names = {'models', 'ai', 'llm', 'llms', 'gguf', 'lmstudio', 'ollama', 'text-generation-webui', 'comfyui', 'stablediffusion'}
    for d in drives:
        try:
            for entry in d.iterdir():
                try:
                    if entry.is_dir() and entry.name.lower() in ai_folder_names:
                        search_dirs.append(entry)
                except Exception:
                    pass
        except Exception:
            pass

    # ALSO walk each drive root itself.  The AI-folder-name heuristic above
    # only catches folders literally called "models"/"ai"/..., so a library
    # laid out as D:\New folder\... or G:\ntmre_backup\... was never scanned
    # and those models were invisible to the chat picker.  Bounded depth keeps
    # this a few seconds even on a large drive.
    for d in drives:
        try:
            if d.exists():
                search_dirs.append(d)
        except Exception:
            pass

    found_llms = {}
    found_pics = {}
    seen = set()
    links_to_create = []
    _walk_cands = []

    for sdir in search_dirs:
        if not sdir.exists():
            continue
        _is_drive_root = (str(sdir) == sdir.anchor)
        _maxdepth = 5 if _is_drive_root else 3
        try:
            for root, dirs, files in os.walk(sdir):
                dirs[:] = [d for d in dirs if d.lower() not in _WALK_SKIP_DIRS and not d.startswith('$')]
                try:
                    rel_parts = Path(root).relative_to(sdir).parts
                    if len(rel_parts) > _maxdepth:
                        dirs[:] = []
                        continue
                except Exception:
                    pass
                for f in files:
                    low = f.lower()
                    if low.endswith(('.gguf', '.ntmre', '.safetensors', '.lnk')):
                        if 'mmproj' in low or low.endswith('-mmp.gguf'):
                            continue
                        _walk_cands.append((Path(root) / f, root))
        except Exception:
            pass

    # Resolve every shortcut in one PowerShell call instead of one per file.
    resolve_shortcuts_batch([c[0] for c in _walk_cands])

    for fpath, root in _walk_cands:
        try:
            resolved = resolve_shortcut_target(fpath)
            if resolved in seen or not os.path.exists(resolved):
                continue
            if not resolved.lower().endswith(('.gguf', '.ntmre', '.safetensors', '.onnx')):
                continue
            seen.add(resolved)

            base_name = Path(resolved).name
            if is_image_diffusion_model(resolved):
                label = f"{base_name}  ({Path(root).name})" if Path(root).resolve() != models_dir.resolve() else base_name
                found_pics[label] = resolved
            else:
                label = f"{base_name}  ({Path(root).name})" if Path(root).resolve() != models_dir.resolve() else base_name
                found_llms[label] = resolved
                if Path(root).resolve() != models_dir.resolve() and sys.platform == 'win32':
                    lnk_file = models_dir / f"{base_name}.lnk"
                    links_to_create.append((lnk_file, Path(resolved)))
        except Exception:
            pass

    num_created = 0
    if create_shortcuts and links_to_create:
        num_created = create_shortcuts_batch(links_to_create)

    # Prune shortcuts whose model is gone. Without this the directory only
    # ever grows: links_to_create is a create-list, nothing ever removed,
    # and the scan resolves every .lnk it finds, so each rescan paid for
    # a growing pile of links that could not be followed. 34 had
    # accumulated -- 29 at C: files that no longer exist, 5 on a Z: drive
    # that was not connected.
    #
    # Only the "dead" class is removable. A link this system cannot read,
    # or one pointing at something that is not a model, is left alone:
    # a scan is not entitled to delete something a person made by hand.
    prune_report = {}
    if create_shortcuts:
        try:
            from model_shortcut_prune import prune as _prune_shortcuts
            prune_report = _prune_shortcuts(models_dir, dry_run=False,
                                            keep_foreign=True)
        except Exception as _pe:
            print("[SHORTCUTS] prune skipped: %s" % _pe, flush=True)

    return found_llms, found_pics, num_created

