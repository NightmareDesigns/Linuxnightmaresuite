"""
💀 NIGHTMARE GPU STREAMING - Layer-by-Layer GPU Inference
==========================================================
Streams model layers through GPU one at a time.
Only 1-2 layers in VRAM at once - runs ANY size model.

How it works:
  1. Model stays on disk (NVMe) or RAM
  2. For each token:
     - Load layer 0 -> GPU -> compute -> evict
     - Load layer 1 -> GPU -> compute -> evict
     - ... repeat all layers
  3. Double-buffer: prefetch layer N+1 while computing layer N
  4. Dual-GPU pipeline: GPU 0 computes, GPU 1 prefetches (alternating)

This lets you run 70B, 700B, or ANY model on consumer hardware.
Speed depends on PCIe bandwidth and layer size.

NTMRE .ntmre compression advantage:
  - Rank-compressed layers are SMALLER = faster PCIe transfer
  - .ntmre compression factors compute faster on GPU (cache-friendly)
  - Same model, smaller layers, faster streaming

Usage:
  From the GUI: Model menu -> "⚡ Stream Mode" 
  CLI: python nightmare_stream.py model.ntmre "prompt"

Author: Nightmare Designs 💀
"""

import os
import sys
import time
import struct
import ctypes
import threading

# === CPU compute thread budget ===
# Must run BEFORE numpy imports: OpenBLAS/MKL read these once at section init.
# Saturating extra threads on BLAS matmuls hurts tok/s (cache thrash), so pin
# to the PHYSICAL core count instead of letting BLAS/OpenMP guess at the SMT
# logical count.  Optionally restrict the process to one logical processor per
# physical core (NIGHTMARE_PHYSICAL_ONLY=1) so work can never spill onto a
# shared SMT sibling.

def _physical_cpu_plan():
    """Return (physical_cores, affinity_mask) for this machine using the
    Win32 GetLogicalProcessorInformation API.  affinity_mask has exactly one
    bit per physical core (the first logical processor of each), or None when
    the API is unavailable / not Windows (then logical count is used)."""
    try:
        import ctypes as _ct
        from ctypes import wintypes
        if os.name != 'nt':
            return getattr(os, 'cpu_count', lambda: None)(), None
        _k32 = _ct.WinDLL('kernel32', use_last_error=False)
        _glpi = _k32.GetLogicalProcessorInformation
        _glpi.restype = wintypes.BOOL
        _glpi.argtypes = [_ct.c_void_p, _ct.POINTER(wintypes.DWORD)]

        class _SLPI(_ct.Structure):
            _fields_ = [('ProcessorMask', _ct.c_size_t),
                        ('Relationship', _ct.c_int),
                        ('Reserved', _ct.c_ubyte * 16)]

        _need = wintypes.DWORD(0)
        _glpi(None, _ct.byref(_need))
        if not _need.value:
            return getattr(os, 'cpu_count', lambda: None)(), None
        _buf = _ct.create_string_buffer(_need.value + 32)
        if not _glpi(_buf, _ct.byref(_need)):
            return getattr(os, 'cpu_count', lambda: None)(), None
        _stride = _ct.sizeof(_SLPI)
        _cores = 0
        _mask = 0
        _n = _need.value // _stride
        for i in range(_n):
            _off = i * _stride
            _rel = int.from_bytes(_buf.raw[_off + _ct.sizeof(_ct.c_size_t):
                                           _off + _ct.sizeof(_ct.c_size_t) + 4],
                                  'little')
            _pm = int.from_bytes(
                _buf.raw[_off:_off + _ct.sizeof(_ct.c_size_t)], 'little')
            if _rel == 0:            # RelationProcessorCore
                _cores += 1
                if _pm:
                    _low = _pm & -_pm   # lowest set bit (first SMT thread)
                    _mask |= _low
        if not _cores:
            return getattr(os, 'cpu_count', lambda: None)(), None
        return _cores, _mask
    except Exception:
        return getattr(os, 'cpu_count', lambda: None)(), None


def _apply_cpu_thread_env():
    _physical, _aff = _physical_cpu_plan()
    try:
        _cores = int(os.environ.get('NIGHTMARE_CPU_THREADS')
                     or _physical
                     or getattr(os, 'cpu_count', lambda: 4)() or 4)
    except Exception:
        _cores = 4
    if not os.environ.get('NIGHTMARE_CPU_THREADS'):
        os.environ['NIGHTMARE_CPU_THREADS'] = str(_cores)
    for _k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
               'NUMEXPR_NUM_THREADS'):
        if not os.environ.get(_k):
            os.environ[_k] = str(_cores)
    # Restrict execution to one logical processor per physical core unless the
    # user opted out (or asked for a specific thread count / all SMT threads).
    if (_aff and _physical
            and os.environ.get('NIGHTMARE_PHYSICAL_ONLY', '1') == '1'
            and not os.environ.get('NIGHTMARE_CPU_THREADS')):
        try:
            import ctypes as _ct
            from ctypes import wintypes
            _k32 = _ct.WinDLL('kernel32', use_last_error=True)
            _k32.GetCurrentProcess.restype = wintypes.HANDLE
            _k32.SetProcessAffinityMask.restype = wintypes.BOOL
            _k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, _ct.c_size_t]
            _k32.SetProcessAffinityMask(
                _k32.GetCurrentProcess(), _ct.c_size_t(_aff))
            print(f"[CPU] Pinned to {_physical} physical core(s) "
                  f"(affinity 0x{_aff:X}); SMT siblings excluded")
        except Exception:
            pass

_cores_default = getattr(os, 'cpu_count', lambda: 4)() or 4
_PHYSICAL_CORES_DEFAULT = _physical_cpu_plan()[0] # uses SMT logical count on non-Windows
_apply_cpu_thread_env()

import nightmare_np as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import deque

import gpu_info


def _is_gemma4_arch(arch) -> bool:
    """Robust matcher for all Gemma 4 architecture variants."""
    a = str(arch or '').lower().strip()
    return any(a == base or a.startswith(base + '-') or a.startswith(base + '_')
               for base in ('gemma4', 'gemma-4', 'gemma_4'))

# RESIDENT Q8_0 CPU MATMUL  (native AVX2 kernel, no f32 layer cache)
# ════════════════════════════════════════════════════════════════
# GGML Q8_0 blocks: 34 bytes / 32 elems = [f16 scale][32 x int8].
# Keeping layer weights RESIDENT in Q8_0 form uses 4x less RAM than
# the f32 dequant cache, so big Q8_0 GGUFs (e.g. a 12B) fit entirely
# in the layer-cache budget and every token is a pure AVX2 GEMV over
# resident bytes (no per-token dequant, no DRAM-f32 re-read).

_Q8DLL = None
_Q8DLL_TRIED = False


def _load_q8_dll():
    """Load the AVX2 Q8_0 GEMV DLL (bundled with the exe or beside this
    script).  Returns the ctypes handle, or None on any failure."""
    global _Q8DLL, _Q8DLL_TRIED
    if _Q8DLL_TRIED:
        return _Q8DLL
    _Q8DLL_TRIED = True
    cands = []
    base = getattr(sys, '_MEIPASS', None)
    if base:
        cands.append(os.path.join(base, 'nightmare_q8_avx2.dll'))
    cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              'nightmare_q8_avx2.dll'))
    for p in cands:
        try:
            if not os.path.exists(p):
                continue
            lib = ctypes.CDLL(p)
            lib.q8_gemv.argtypes = [ctypes.POINTER(ctypes.c_float),
                                    ctypes.POINTER(ctypes.c_uint8),
                                    ctypes.POINTER(ctypes.c_float),
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int]
            lib.q8_gemv.restype = None
            lib.q8_gemm.argtypes = [ctypes.POINTER(ctypes.c_float),
                                    ctypes.POINTER(ctypes.c_uint8),
                                    ctypes.POINTER(ctypes.c_float),
                                    ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int]
            lib.q8_gemm.restype = None
            _Q8DLL = lib
            return lib
        except Exception:
            continue
    return None


def _q8_threads():
    """Thread pool size for the AVX2 Q8_0 GEMV kernel.  Defaults to the
    PHYSICAL core count (SMT siblings share execution units, so spawning
    threads on logical siblings adds contention, not throughput).  Override
    with NIGHTMARE_CPU_Q8_THREADS."""
    try:
        return int(os.environ.get('NIGHTMARE_CPU_Q8_THREADS')
                   or _PHYSICAL_CORES_DEFAULT
                   or min(os.cpu_count() or 4, 8))
    except Exception:
        return 4


def _f32_to_q8_0(arr):
    """Convert a (n, m) f32 ndarray (m % 32 == 0) into GGML Q8_0 raw bytes
    (n * (m//32) * 34).  Mirrors GGML's quantizer semantics: d = amax/127,
    qs = round(x/d) clipped to [-127,127], scale stored as f16 (the kernel
    dequantizes with the f16-rounded d like ggml_fp32_to_fp16)."""
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    n, m = arr.shape
    if m % 32 != 0:
        raise ValueError(f"_f32_to_q8_0 needs in_dim%32==0, got {m}")
    a = arr.reshape(-1, 32)
    nb = a.shape[0]
    amax = np.max(np.abs(a), axis=1)
    d = amax / 127.0
    id_ = np.divide(1.0, d, out=np.zeros_like(d), where=d != 0)
    q = np.clip(np.round(a * id_[:, None]), -127, 127).astype(np.int8)
    q = np.where((d == 0)[:, None], 0, q).astype(np.int8)
    d16 = d.astype(np.float16)  # GGML_FP32_TO_FP16 round-to-nearest-even
    raw = np.empty((nb, 34), dtype=np.uint8)
    raw[:, 0:2] = d16.view(np.uint8).reshape(nb, 2)
    raw[:, 2:] = q.view(np.uint8).reshape(nb, 32)
    return np.ascontiguousarray(raw.reshape(-1)), d


class Q8W:
    """Resident Q8_0 weight descriptor.  Takes ~1 byte/elem of RAM (vs 4
    for f32) and runs decode matvecs through the native AVX2 kernel.
    Fallbacks (`to_f32`, numpy `@`) stay available for batched prefill,
    which dequantizes one layer at a time (same cost as before)."""
    __slots__ = ('raw', 'n_out', 'n_in', 'layer', 'name', '_prs', '_m')

    def __init__(self, raw, n_out, n_in, layer=None, name=None,
                 prs=None, m=None):
        self.raw = np.ascontiguousarray(raw, dtype=np.uint8)
        self.n_out = int(n_out)
        self.n_in = int(n_in)
        self.layer = layer
        self.name = name
        self._prs = prs
        self._m = m

    @property
    def nbytes(self):
        return self.raw.nbytes

    def to_f32(self):
        """Dequantize this weight to f32 from the RESIDENT Q8_0 bytes (used
        by batched prefill GEMMs).  Matches the AVX2 kernel exactly (same
        stored block scales).  NOT cached - the layer cache keeps only the
        1-byte/elem bytes so big models stay resident instead of thrashing."""
        if self.raw is None or self.n_in % 32 != 0:
            return None
        try:
            b = self.raw.reshape(-1, 34)
            d = b[:, 0:2].view('<f2').astype(np.float32)[:, 0]
            q = b[:, 2:].view(np.int8).astype(np.float32)
            return (d[:, None] * q).reshape(self.n_out, self.n_in)
        except Exception:
            return None

    def matvec(self, x):
        """Single-row matvec over resident Q8_0 bytes via the AVX2 kernel.
        Returns (1, n_out).  Falls back to numpy f32 dequant when the DLL
        is unavailable or the width isn't 32-aligned."""
        lib = _load_q8_dll()
        if (lib is not None and self.raw is not None
                and self.n_in % 32 == 0):
            x1 = np.ascontiguousarray(x.reshape(-1), dtype=np.float32)
            if x1.size == self.n_in:
                out = np.empty(self.n_out, dtype=np.float32)
                lib.q8_gemv(
                    x1.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    self.raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    self.n_in, self.n_out, _q8_threads())
                return out.reshape(1, -1)
        wf = self.to_f32()
        return x @ wf.T if wf is not None else x

    def bmm(self, x):
        """Batched over M rows: out[M, n_out] = X @ W^T via the AVX2 kernel
        over resident Q8_0 bytes (only supports n_out % 32 == 0; falls back
        to per-row matvec otherwise)."""
        lib = _load_q8_dll()
        if (lib is not None and self.raw is not None
                and self.n_in % 32 == 0):
            xc = np.ascontiguousarray(x, dtype=np.float32)
            out = np.empty((xc.shape[0], self.n_out), dtype=np.float32)
            lib.q8_gemm(
                xc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.raw.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                self.n_in, self.n_out, xc.shape[0], _q8_threads())
            return out
        wf = self.to_f32()
        return x @ wf.T if wf is not None else x


# ════════════════════════════════════════════════════════════════
# GPU STREAM ENGINE
# ════════════════════════════════════════════════════════════════

class GPUStreamEngine:
    """
    Layer streaming inference engine.
    
    Loads one layer at a time into GPU VRAM, computes, evicts.
    Double-buffers to overlap PCIe transfers with compute.
    Supports dual-GPU pipeline for 2x throughput.
    """
    
    def __init__(self, model_path: str, max_seq_len: int = 2048):
        self.model_path = model_path
        self.max_seq_len = max_seq_len
        self.loaded = False

        # The router publishes the plan it built for THIS machine before
        # constructing the engine. Picking it up here is what makes the
        # context ceiling a measurement of the hardware rather than a
        # number typed into this file -- which is what makes the engine
        # behave sensibly on someone else's machine instead of being
        # capped wherever the author happened to be standing.
        #
        # It is read here rather than deep in the loader because this is
        # the only point guaranteed to run for every constructor path.
        try:
            _p = globals().get('PLAN_FOR_NEXT_LOAD')
            self._ntmre_plan = _p
            if _p is not None and globals().get('PLAN_FOR_NEXT_LOAD') is _p:
                # one-shot: a later load without a new plan must not
                # inherit this one
                globals()['PLAN_FOR_NEXT_LOAD'] = None
        except Exception:
            self._ntmre_plan = None
        
        # Model config (read from header)
        self.dim = 0
        self.n_layers = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.hidden_dim = 0
        self.vocab_size = 0
        self.head_dim = 0
        self.rope_base = 10000.0
        self._norm_eps = 1e-5

        # MoE (Mixture of Experts) config - 0 means dense (no experts)
        self._n_experts = 0
        self._n_experts_used = 0      # top-k experts per token
        self._moe_norm_eps = 1e-5     # eps for expert routing softmax
        # LayerNorm vs RMSNorm: True = standard LayerNorm (with mean subtraction)
        self._use_layernorm = False
        self._kv_f16 = True          # F16 KV cache (half the RAM)
        self._kv_disk_path = None    # set to a path prefix for disk-backed KV cache

        # GPU-resident streaming buffers (allocated when GpuMM is active)
        self._gpu_buf = None       # dict of device pointers for x, xn, q, k, v, etc.
        self._gpu_norm_ptrs = {}   # (layer, 'attn'|'ffn') -> device ptr for norm weight
        self._gpu_kv = None        # ('gpu', k_ptrs, v_ptrs) or ('cpu', None, None)
        self._gpu_logits_ptr = None
        # CUDA graph state (re-captured after each reset)
        self._graph_tried = False
        self._graph_logits_ptr = None
        self._g_pos = None
        self._g_seq = None
        self._g_k_dst = None
        self._g_v_dst = None

        # gemma4 extras: cross-layer KV sharing (last N layers reuse the
        # last same-type layer's K/V) and Per-Layer Embeddings (PLE)
        self._g4_shared_kv = 0     # num_kv_shared_layers (0 = disabled)
        self._g4_share = None      # layer -> source layer map (lazy)
        self._ple_dim = 0          # hidden_size_per_layer_input (0 = no PLE)
        self._ple_model_proj = None
        self._ple_proj_norm = None
        self._ple_tok = None       # (path, data_off, dtype, row_elems)
        self._ple_tok_table = None # whole-table fallback (NTMRE sources)
        self._ple_current = None   # (n_layers, ple_dim) for the live token

        # Streaming state
        self._file = None           # Memory-mapped or open file handle
        self._layer_offsets = []    # File offset for each layer's weights
        self._layer_sizes = []     # Size in bytes for each layer
        self._embedding = None     # Embedding kept in RAM (small enough)
        self._output_weight = None # Output projection in RAM
        self._output_norm = None   # Final norm in RAM
        
        # GPU engine
        self._gpu = None
        self._gpu_count = 0
        self._active_gpu = 0
        
        # Double-buffer: two VRAM slots
        self._buffer_a = None  # GPU pointer for buffer A
        self._buffer_b = None  # GPU pointer for buffer B
        self._prefetch_thread = None
        self._prefetch_ready = threading.Event()
        self._prefetch_layer = -1
        
        # KV cache (stays in RAM - streamed to GPU per-layer)
        self.kv_k = None
        self.kv_v = None
        self.pos = 0
        
        # Performance tracking
        self._times = {'load': 0, 'compute': 0, 'total': 0}
        self._tokens_generated = 0
        # CPU cold-warmup progress: how many of n_layers have been
        # requantized/resident-loaded so far.  Polled by the suite UI so
        # the first (slow) pass shows live progress instead of silence.
        self._warmup_done = 0
        self._warmup_total = 0
        self._warmup_detail = ''
        
        # Initialize
        self._init_gpu()
        if model_path:
            self._load_model_header(model_path)
    
    def _init_gpu(self):
        """Initialize GPU engine (nightmare_engine.dll)."""
        # Raw-quant GPU accelerator (qwen35_gpu.GpuMM): weights stay
        # quantized and are dequantized on the GPU per token.  This is
        # what makes multi-GB GGUF/NTMRE files fast - the CPU dequant +
        # f32 layer cache thrashes once the model exceeds the RAM budget.
        # Created here (before header load) so _load_embedding_and_output
        # can register the logits matrix for GPU-chunked matvec.
        self._gpumm = None
        self._gpumm2 = None      # second GpuMM (device 0) for overflow weights
        self._weight_gpu = {}   # tensor key -> GpuMM that holds it (1 or 2)
        self._raw_store = {}     # tensor key -> raw quantized bytes (RAM)
        self._output_gpu = None  # descriptor for GPU-chunked logits

        is_cpu_only = (os.environ.get('NIGHTMARE_FORCE_CPU') == '1')
        if not is_cpu_only:
            try:
                from nightmare_gpu_discovery import has_gpu_accelerator
                is_cpu_only = not has_gpu_accelerator()
            except Exception:
                pass

        if is_cpu_only:
            self._gpumm = None
            self._gpumm2 = None
            self._gpu = None
            self.use_gpu = False
            self.cu = None
            _t = os.environ.get('NIGHTMARE_CPU_THREADS', '6')
            print(f"[STREAM] 🖥️ CPU Mode Confirmed (AVX2+FMA {_t}-Thread OpenMP Engine)")
        else:
            self._gpu_workers = []
            try:
                from qwen35_gpu import GpuMM
                from nightmare_gpu_discovery import usable_accelerators
                discovered = usable_accelerators()
                if discovered:
                    for d in discovered:
                        dev_idx = d.get('index', 0)
                        try:
                            w = GpuMM.try_create(device=dev_idx)
                            if w is not None:
                                self._gpu_workers.append(w)
                        except Exception:
                            pass
                else:
                    w = GpuMM.try_create()
                    if w is not None:
                        self._gpu_workers.append(w)
            except Exception:
                self._gpu_workers = []

            self._gpumm = self._gpu_workers[0] if self._gpu_workers else None
            self._gpumm2 = self._gpu_workers[1] if len(self._gpu_workers) > 1 else None

            if len(self._gpu_workers) > 1:
                dev_str = ", ".join(f"GPU{w.device}" for w in self._gpu_workers)
                print(f"[STREAM] Multi-GpuMM Active ({len(self._gpu_workers)} GPUs: {dev_str})")
            elif self._gpumm is not None:
                print(f"[STREAM] Single-GpuMM Active (GPU{self._gpumm.device})")
            try:
                from nightmare_engine import NightmareEngine
                self._gpu = NightmareEngine()
                # device_count() was lost in the admin NightmareEngine
                # rewrite, so the old code crashed HERE with
                # AttributeError as soon as the DLL was present again:
                # the getattr() guard on the line above returned 1,
                # then the next line called .device_count() directly.
                # Never let a GPU-probe failure kill a model load;
                # fall back to CPU streaming instead.
                _dc = getattr(self._gpu, 'device_count', None)
                _count = _dc() if callable(_dc) else (1 if self._gpu.available else 0)
                if self._gpu.available and _count > 0:
                    self._gpu_count = _count
                    print(f"[STREAM] GPU engine ready ({self._gpu_count} device{'s' if self._gpu_count > 1 else ''})")
                else:
                    self._gpu = None
                    print("[STREAM] GPU not available - CPU streaming mode")
            except Exception:
                self._gpu = None
                print("[STREAM] GPU engine unavailable - CPU streaming mode")
    
    def _load_model_header(self, path: str):
        """Load model header and build layer offset table."""
        ext = Path(path).suffix.lower()
        
        if ext == '.ntmre':
            self._load_ntmre_header(path)
        elif ext == '.gguf':
            self._load_gguf_header(path)
        elif ext == '.safetensors':
            self._load_safetensors_header(path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        is_cpu_only = (os.environ.get('NIGHTMARE_FORCE_CPU') == '1')
        if not is_cpu_only:
            try:
                from nightmare_gpu_discovery import has_gpu_accelerator
                is_cpu_only = not has_gpu_accelerator()
            except Exception:
                pass

        if is_cpu_only:
            self.use_gpu = False
            self.cu = None
        else:
            try:
                from nvrtc_engine import NvrtcEngine
                self.cu = NvrtcEngine(device=0)
                self.use_gpu = True
            except Exception as e:
                print(f"[STREAM] GPU init failed ({e}), falling back to CPU streaming")
                self.use_gpu = False
    
    @staticmethod
    def _is_v16_layout(path, version):
        """v16 always qualifies; v1 only when it carries a sane v16-style
        u64 config block @128 (e.g. smollm2-135m_svdq8.ntmre)."""
        if version == 16:
            return True
        from ntmre_v16_engine import _u64_cfg_sane
        import os
        with open(path, 'rb') as f:
            head = f.read(4096)
        return _u64_cfg_sane(head, os.path.getsize(path))

    def _load_ntmre_header(self, path: str):
        """Parse NTMRE file header. Detects JSON container vs raw NTMRE binary."""
        with open(path, 'rb') as f:
            head = f.read(17)
        magic_bytes = head[:4]
        version = struct.unpack_from('<I', head, 4)[0] if len(head) >= 8 else 0
        # JSON container: NSVD/NTMR/RMTN magic, then u32 n_tensors, u32
        # cfg_len, and the JSON config starting with '{' at offset 16.
        is_json_container = (magic_bytes in (b'NSVD', b'NTMR', b'RMTN')
                             and len(head) == 17 and head[16:17] == b'{')

        if magic_bytes == b'RMTN' and version in (1, 16) and self._is_v16_layout(path, version):
            # NTMRE v16 binary - V16Source (classic + NEO variants,
            # including v1 files written in the v16 layout)
            from ntmre_v16_engine import V16Source
            self._nsvd = V16Source(path)
            cfg = self._nsvd.config
            arch = str(cfg.get('arch', 'llama')).lower()
            self._architecture = arch
            self.dim = int(cfg['dim'])
            self.n_layers = int(cfg['n_layers'])
            self.n_heads = int(cfg['n_heads'])
            self.n_kv_heads = int(cfg['n_kv_heads'])
            self.hidden_dim = int(cfg['hidden_dim'])
            self.vocab_size = int(cfg['vocab_size'])
            self.max_seq_len = int(cfg['max_seq_len'])
            self.rope_base = float(cfg['rope_theta'])
            self._norm_eps = float(cfg.get('norm_eps', 1e-6 if arch.startswith('qwen') else 1e-5))
            self.head_dim = int(cfg.get('head_dim') or self.dim // max(self.n_heads, 1))
            self._n_experts = int(cfg.get('n_experts', 0))
            self._n_experts_used = int(cfg.get('n_experts_used', 0))
            if self._n_experts and not self._n_experts_used:
                self._n_experts_used = 2
            if any(a in arch for a in ('gpt2', 'gptj', 'gpt-neox', 'command-r', 'falcon', 'refact')):
                self._use_layernorm = True
        elif magic_bytes == b'RMTN' and version in (1, 2, 13):
            # NTMRE v1 binary (C converter layout) - V1Source
            from ntmre_v1_engine import V1Source
            self._nsvd = V1Source(path)
            cfg = self._nsvd.config
            arch = str(cfg.get('arch', 'llama')).lower()
            self._architecture = arch
            self.dim = int(cfg['dim'])
            self.n_layers = int(cfg['n_layers'])
            self.n_heads = int(cfg['n_heads'])
            self.n_kv_heads = int(cfg['n_kv_heads'])
            self.hidden_dim = int(cfg['hidden_dim'])
            self.vocab_size = int(cfg['vocab_size'])
            self.max_seq_len = int(cfg['max_seq_len'])
            self.rope_base = float(cfg['rope_theta'])
            self._norm_eps = float(cfg.get('norm_eps', 1e-6 if arch.startswith('qwen') else 1e-5))
            self.head_dim = int(cfg.get('head_dim') or self.dim // max(self.n_heads, 1))
            self._n_experts = int(cfg.get('n_experts', 0))
            self._n_experts_used = int(cfg.get('n_experts_used', 0))
            if self._n_experts and not self._n_experts_used:
                self._n_experts_used = 2
            if any(a in arch for a in ('gpt2', 'gptj', 'gpt-neox', 'command-r', 'falcon', 'refact')):
                self._use_layernorm = True
        elif is_json_container:
            # NTMRE JSON container - use NSVDSource (universal weight rule)
            from nightmare_qwen35 import NSVDSource
            self._nsvd = NSVDSource(path)
            cfg = self._nsvd.config
            arch = str(cfg.get("arch", cfg.get("general.architecture", "llama"))).lower()
            self._architecture = arch
            self.dim = int(cfg.get(f"{arch}.embedding_length", cfg.get("dim", 4096)))
            self.n_layers = int(cfg.get(f"{arch}.block_count", cfg.get("n_layers", 32)))
            self.n_heads = int(cfg.get(f"{arch}.attention.head_count", cfg.get("n_heads", 32)))
            self.n_kv_heads = int(cfg.get(f"{arch}.attention.head_count_kv", cfg.get("n_kv_heads", self.n_heads)))
            self.hidden_dim = int(cfg.get(f"{arch}.feed_forward_length", cfg.get("hidden_dim", cfg.get("ffn_dim", 11008))))
            self.vocab_size = int(cfg.get(f"{arch}.vocabulary_length", cfg.get("vocab_size", 32000)))
            self.max_seq_len = int(cfg.get(f"{arch}.context_length", cfg.get("max_seq_len", 2048)))
            self.rope_base = float(cfg.get(f"{arch}.rope.freq_base", cfg.get("rope_base", cfg.get("rope_theta", 10000.0))))
            self._norm_eps = float(cfg.get(f"{arch}.attention.layer_norm_rms_epsilon", cfg.get("norm_eps", 1e-6 if arch.startswith('qwen') else 1e-5)))
            self.head_dim = int(cfg.get(f"{arch}.attention.key_length", cfg.get("head_dim", 0))) or self.dim // self.n_heads
            self._n_experts = int(cfg.get(f"{arch}.expert_count", cfg.get("n_experts", 0)))
            self._n_experts_used = int(cfg.get(f"{arch}.expert_used_count", cfg.get("n_experts_used", 0)))
            if self._n_experts and not self._n_experts_used:
                self._n_experts_used = 2
            if any(a in arch for a in ('gpt2', 'gptj', 'gpt-neox', 'command-r', 'falcon', 'refact')):
                self._use_layernorm = True
        else:
            # Raw NTMRE binary format - header is 16 bytes, then config immediately follows
            with open(path, 'rb') as f:
                magic, version, flags, header_size = struct.unpack('<IIII', f.read(16))
                if magic != 0x4E544D52:
                    raise ValueError(f"Not an NTMRE file (magic: 0x{magic:08X})")
                
                # Config: '<IIIIIIIf I' = 36 bytes
                cfg = struct.unpack('<IIIIIIIf I', f.read(36))
                self.dim = cfg[0]
                self.n_layers = cfg[1]
                self.n_heads = cfg[2]
                self.n_kv_heads = cfg[3]
                self.hidden_dim = cfg[4]
                self.vocab_size = cfg[5]
                self.max_seq_len = cfg[6]
                self.rope_base = cfg[7]
                self.head_dim = cfg[8] if cfg[8] != 0 else self.dim // max(self.n_heads, 1)
            
            self._architecture = 'llama'
            self._estimate_layer_offsets()
            self._file = open(path, 'rb')
        
        print(f"[STREAM] NTMRE Model: {self.dim}d x {self.n_layers}L x {self.n_heads}H")
        
        # The context ceiling comes from the ROUTER PLAN when there is one,
        # and otherwise from what memory can actually hold. It used to be a
        # literal 12000 typed in here, which silently capped a machine with
        # 8 cards at a fraction of its capacity while doing nothing to
        # protect a small one -- wrong in both directions, invisibly.
        self.max_seq_len = self._derive_ctx_limit()
        if not self.max_seq_len or int(self.max_seq_len) <= 0:
            # _derive_ctx_limit returns None when it cannot measure
            # anything. Falling back to a literal here would reintroduce
            # exactly the typed constant this replaced -- so fall back to
            # what the caller asked for, which is a real number from a
            # real caller, and say so.
            self.max_seq_len = int(max_seq_len) if max_seq_len else 2048
            print("[STREAM] context %d (caller's value; nothing measurable "
                  "on this machine)" % self.max_seq_len, flush=True)

        # Allocate KV cache
        _nkv = self.n_kv_heads
        if isinstance(_nkv, list):
            _nkv = max(_nkv) if _nkv else self.n_heads
        kv_dim = _nkv * self.head_dim
        if _is_gemma4_arch(getattr(self, '_architecture', '')):
            # dual attention: per-layer KV width varies (256-SWA / 512-full)
            kv_dim = max([kv_dim] + [self._gemma4_layer_info(l)[1]
                                     for l in range(self.n_layers)])
        _kv_dtype = np.float16 if self._kv_f16 else np.float32
        self._fit_kv_cache_to_ram(kv_dim, np.dtype(_kv_dtype).itemsize)
        _kv_shape = (self.n_layers, self.max_seq_len, kv_dim)
        if getattr(self, '_kv_disk_path', None):
            # Disk-backed KV cache via mmap - OS pages hot data to RAM,
            # cold data to SSD. Enables very long conversations without OOM.
            import os
            kp = self._kv_disk_path
            os.makedirs(os.path.dirname(kp) or '.', exist_ok=True)
            self.kv_k = np.memmap(kp + '_k.bin', dtype=_kv_dtype, mode='w+', shape=_kv_shape)
            self.kv_v = np.memmap(kp + '_v.bin', dtype=_kv_dtype, mode='w+', shape=_kv_shape)
            _kv_gb = (_kv_shape[0] * _kv_shape[1] * _kv_shape[2] * (2 if self._kv_f16 else 4)) / 1e9
            print(f'[STREAM] Disk-backed KV cache (mmap): {_kv_gb:.2f} GB at {kp}')
        else:
            self.kv_k = np.zeros(_kv_shape, dtype=_kv_dtype)
            self.kv_v = np.zeros(_kv_shape, dtype=_kv_dtype)
        
        self.loaded = True
        
        # Load embedding and output into RAM
        self._load_embedding_and_output()

        # Populate the built-in tokenizer from the container's tokenizer JSON
        tb = getattr(getattr(self, '_nsvd', None), 'tokenizer_bytes', None)
        if tb:
            try:
                import json as _json
                d = _json.loads(tb.decode('utf-8'))
                self.tokens = d.get('tokens', [])
                self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
                # _apply_bpe expects ID pairs; the JSON stores string pairs
                self.merges = []
                for m in d.get('merges', []):
                    parts = m.split(' ') if isinstance(m, str) else list(m)
                    if len(parts) == 2 and parts[0] in self.token_to_id and parts[1] in self.token_to_id:
                        self.merges.append((self.token_to_id[parts[0]], self.token_to_id[parts[1]]))
                self._bos_id = int(d.get('bos', 1))
                self._eos_id = int(d.get('eos', 2))
            except Exception as e:
                print(f"[STREAM] container tokenizer parse failed: {e}")

        print(f"[STREAM] Ready - streaming {self.n_layers} layers through GPU")
    
    def _estimate_layer_offsets(self):
        """Estimate byte offsets for each layer in the file."""
        # Each layer has: attn_norm, ffn_norm (F32), Q/K/V/O/gate/up/down (Q8 or .ntmre compression)
        # This is approximate - real implementation reads the tensor index
        
        # Q8 block: 36 bytes per 32 elements
        def q8_size(n_elements):
            return ((n_elements + 31) // 32) * 36
        
        # Per-layer weight sizes
        q_size = q8_size(self.dim * self.n_heads * self.head_dim)
        k_size = q8_size(self.dim * self.n_kv_heads * self.head_dim)
        v_size = k_size
        o_size = q8_size(self.n_heads * self.head_dim * self.dim)
        gate_size = q8_size(self.dim * self.hidden_dim)
        up_size = gate_size
        down_size = q8_size(self.hidden_dim * self.dim)
        norm_size = self.dim * 4 * 2  # Two F32 norms
        
        layer_total = q_size + k_size + v_size + o_size + gate_size + up_size + down_size + norm_size
        
        # Embedding offset (after header + tokenizer, approximate)
        emb_start = 1024 + self.vocab_size * 10  # rough estimate for tokenizer
        emb_size = q8_size(self.vocab_size * self.dim)
        
        offset = emb_start + emb_size
        self._layer_offsets = []
        self._layer_sizes = []
        
        for l in range(self.n_layers):
            # Align to 4096
            offset = ((offset + 4095) // 4096) * 4096
            self._layer_offsets.append(offset)
            self._layer_sizes.append(layer_total)
            offset += layer_total
    
    def _load_gguf_header(self, path: str):
        """Parse GGUF file header and map tensors using gguf_parser."""
        import gguf_parser
        self._parser = gguf_parser.GGUFParser(path)
        self.m = self._parser.parse()
        
        # Extract metadata
        meta = self.m.metadata
        arch = meta.get("general.architecture", "llama")
        
        # Handle prefix fallback for architecture
        for prefix in [arch, "llama", "qwen2", "gemma", "mistral", "phi3"]:
            if f"{prefix}.embedding_length" in meta:
                arch = prefix
                break
                
        self._architecture = arch
        self.dim = meta.get(f"{arch}.embedding_length", 4096)
        self.n_layers = meta.get(f"{arch}.block_count", 32)
        # old-style key fallbacks for custom exports (llama.head_count ...)
        self.n_heads = meta.get(f"{arch}.attention.head_count",
                                meta.get(f"{arch}.head_count", 32))
        self.n_kv_heads = meta.get(f"{arch}.attention.head_count_kv",
                                   meta.get(f"{arch}.head_count_kv", self.n_heads))
        # Gemma4 variants sometimes store head counts as lists.
        if isinstance(self.n_heads, (list, tuple)):
            self.n_heads = max(self.n_heads) if self.n_heads else 1
        if isinstance(self.n_kv_heads, (list, tuple)):
            self.n_kv_heads = max(self.n_kv_heads) if self.n_kv_heads else self.n_heads
        self.hidden_dim = meta.get(f"{arch}.feed_forward_length", 11008)
        # Gemma4 (and any mixed-width family) stores feed_forward_length as a
        # PER-LAYER list, e.g. [6144 x15, 12288 x ...]. Leaving that as a list
        # made every `int(self.hidden_dim)` and every
        # np.zeros((self.hidden_dim, self.dim)) raise
        #   TypeError: 'list' object cannot be interpreted as an integer
        # which surfaced as a GPU-alloc failure and a dead CPU load.
        # Keep the per-layer vector, and expose the widest width as the
        # scalar used for buffer sizing.
        self._hidden_dims = None
        if isinstance(self.hidden_dim, (list, tuple)):
            vals = [int(v) for v in self.hidden_dim if v is not None]
            self._hidden_dims = list(self.hidden_dim)
            self.hidden_dim = max(vals) if vals else 11008
        else:
            self.hidden_dim = int(self.hidden_dim)
        self.vocab_size = meta.get(f"{arch}.vocabulary_length", meta.get("tokenizer.ggml.vocab_size", meta.get(f"{arch}.vocab_size", 32000)))
        self.max_seq_len = meta.get(f"{arch}.context_length", 2048)
        # Gemma4 setup check, fixed 2026-10-28: never silently use the
        # 4096-dim / 32-layer Llama defaults for a Gemma4 model. If the
        # embedding length is missing but the layer count identifies a
        # verified variant, fill dim from that verified table and warn.
        # Otherwise fail loudly — do not synthesise it as E4B.
        if str(arch).lower().startswith("gemma"):
            _by_layers = {35: 1536, 42: 2560, 48: 3840, 30: 2816}
            if f"{arch}.embedding_length" not in meta and self.n_layers in _by_layers:
                print(f"[MODEL-SETUP] WARNING: {arch}.embedding_length missing; "
                      f"using verified dim {_by_layers[self.n_layers]} for "
                      f"{self.n_layers}-layer Gemma4 (not 4096 Llama default)")
                self.dim = _by_layers[self.n_layers]
            elif f"{arch}.embedding_length" not in meta or f"{arch}.block_count" not in meta:
                raise ValueError(
                    f"Unknown Gemma4 model setup: missing "
                    f"{[k for k in (f'{arch}.embedding_length', f'{arch}.block_count') if k not in meta]} "
                    f"— not synthesised as E4B. Add a verified spec or repair the model metadata.")

        rope_keys = [
            f"{arch}.rope.freq_base",
            f"{arch}.rope.frequency_base",
            f"{arch}.attention.rope.freq_base",
            f"{arch}.rope_theta",
            "llama.rope.freq_base",
            "llama.rope.frequency_base",
            "rope.freq_base",
            "general.rope.freq_base"
        ]
        rope_base_val = None
        for k in rope_keys:
            if k in meta:
                rope_base_val = meta[k]
                break
        if rope_base_val is not None:
            self.rope_base = float(rope_base_val)
        elif self.vocab_size == 128256 or self.vocab_size == 128000:
            self.rope_base = 500000.0
        else:
            self.rope_base = 10000.0

        self._norm_eps = float(meta.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6 if arch.startswith('qwen') else 1e-5))
        # MoE (Mixture of Experts) detection - Mixtral, DeepSeek MoE, Qwen2 MoE, etc.
        self._n_experts = int(self.m._safe_int(meta.get(f"{arch}.expert_count", 0), 0))
        self._n_experts_used = int(self.m._safe_int(meta.get(f"{arch}.expert_used_count", 0), 0))
        if self._n_experts > 0:
            if self._n_experts_used == 0:
                self._n_experts_used = 2  # default top-2
            print(f"[STREAM] MoE detected: {self._n_experts} experts, top-{self._n_experts_used} routing")
        # LayerNorm detection: some architectures (gpt2, command-r, falcon) use
        # standard LayerNorm instead of RMSNorm. Check metadata key or known archs.
        _ln_archs = ('gpt2', 'gptj', 'gpt-neox', 'command-r', 'falcon', 'refact',
                     'bert', 'nomic-bert', 'jina-bert', 'mini-bert')
        if any(a in arch for a in _ln_archs):
            self._use_layernorm = True
            self._norm_eps = float(meta.get(f"{arch}.attention.layer_norm_epsilon", self._norm_eps))
            print(f"[STREAM] Using LayerNorm (arch={arch})")
        # gemma4 dual-rope: local (SWA) layers use a separate theta
        self._rope_theta_swa = float(meta.get(f"{arch}.rope.freq_base_swa", 10000.0))
        self._rope_theta_full = float(self.rope_base)
        if str(arch).startswith('gemma4'):
            # cross-layer KV sharing: the last N layers reuse the K/V of
            # the last same-type layer before the shared region (their own
            # k/v tensors in the file are ignored - HF/llama.cpp semantics)
            self._g4_shared_kv = int(self.m._safe_int(
                meta.get(f"{arch}.attention.shared_kv_layers", 0), 0))
            # Per-Layer Embeddings (E-models): per-token residual signal
            self._ple_dim = int(self.m._safe_int(
                meta.get(f"{arch}.embedding_length_per_layer_input", 0), 0))
            if self._g4_shared_kv:
                print(f"[STREAM] gemma4 shared KV: last {self._g4_shared_kv} layers reuse earlier K/V")
            if self._ple_dim:
                print(f"[STREAM] gemma4 PLE: {self._ple_dim}d per-layer embeddings")

        # Gemma uses head_dim != dim // n_heads (e.g. 3072d with 256d heads)
        self.head_dim = int(meta.get(f"{arch}.attention.key_length", 0)) or self.dim // self.n_heads

        # Trust the tensors over the metadata: custom exports sometimes
        # report wrong head counts.  Derive the true Q/KV widths from
        # blk.0 shapes (single head_dim archs only - gemma4 differs per
        # layer type and is handled by _gemma4_layer_info).
        if not _is_gemma4_arch(arch):
            rev = getattr(self._parser, 'shape_reversed', True)
            tq = self.m.tensors.get('blk.0.attn_q.weight')
            tk = self.m.tensors.get('blk.0.attn_k.weight')
            if tq is not None and tk is not None and len(tk.shape) == 2:
                q_w = int(tq.shape[1] if rev else tq.shape[0])
                kv_w = int(tk.shape[1] if rev else tk.shape[0])
                if self.n_heads > 0 and q_w % self.n_heads == 0:
                    self.head_dim = q_w // self.n_heads
                if self.head_dim > 0 and kv_w % self.head_dim == 0:
                    self.n_kv_heads = kv_w // self.head_dim

        self.max_seq_len = self._derive_ctx_limit()
        if not self.max_seq_len or int(self.max_seq_len) <= 0:
            self.max_seq_len = int(max_seq_len) if max_seq_len else 2048
            print("[STREAM] context %d (caller's value; nothing measurable "
                  "on this machine)" % self.max_seq_len, flush=True)

        print(f"[STREAM] GGUF Model: {self.dim}d x {self.n_layers}L x {self.n_heads}H")
        
        _nkv = self.n_kv_heads
        if isinstance(_nkv, list):
            _nkv = max(_nkv) if _nkv else self.n_heads
        kv_dim = _nkv * self.head_dim
        if _is_gemma4_arch(getattr(self, '_architecture', '')):
            # dual attention: per-layer KV width varies (256-SWA / 512-full)
            kv_dim = max([kv_dim] + [self._gemma4_layer_info(l)[1]
                                     for l in range(self.n_layers)])
        _kv_dtype = np.float16 if self._kv_f16 else np.float32
        self._fit_kv_cache_to_ram(kv_dim, np.dtype(_kv_dtype).itemsize)
        _kv_shape = (self.n_layers, self.max_seq_len, kv_dim)
        if getattr(self, '_kv_disk_path', None):
            # Disk-backed KV cache via mmap - OS pages hot data to RAM,
            # cold data to SSD. Enables very long conversations without OOM.
            import os
            kp = self._kv_disk_path
            os.makedirs(os.path.dirname(kp) or '.', exist_ok=True)
            self.kv_k = np.memmap(kp + '_k.bin', dtype=_kv_dtype, mode='w+', shape=_kv_shape)
            self.kv_v = np.memmap(kp + '_v.bin', dtype=_kv_dtype, mode='w+', shape=_kv_shape)
            _kv_gb = (_kv_shape[0] * _kv_shape[1] * _kv_shape[2] * (2 if self._kv_f16 else 4)) / 1e9
            print(f'[STREAM] Disk-backed KV cache (mmap): {_kv_gb:.2f} GB at {kp}')
        else:
            self.kv_k = np.zeros(_kv_shape, dtype=_kv_dtype)
            self.kv_v = np.zeros(_kv_shape, dtype=_kv_dtype)
        
        self.loaded = True
        self._load_embedding_and_output()

    def _load_safetensors_header(self, path: str):
        """Parse a HuggingFace SafeTensors file header and build layer table.

        Uses safetensors_parser which presents a GGUF-compatible interface,
        so the rest of the streaming engine works unchanged."""
        import safetensors_parser
        self._parser = safetensors_parser.SafeTensorsParser(path)
        self.m = self._parser.parse()

        meta = self.m.metadata
        arch = str(meta.get('general.architecture', 'llama')).lower()
        self._architecture = arch

        self.dim = int(meta.get(f'{arch}.embedding_length', 4096))
        self.n_layers = int(meta.get(f'{arch}.block_count', 32))
        _nh = meta.get(f'{arch}.attention.head_count', 32)
        _nkv = meta.get(f'{arch}.attention.head_count_kv', _nh)
        if isinstance(_nh, (list, tuple)):
            _nh = max(_nh) if _nh else 1
        if isinstance(_nkv, (list, tuple)):
            _nkv = max(_nkv) if _nkv else _nh
        self.n_heads = int(_nh)
        self.n_kv_heads = int(_nkv)
        self.hidden_dim = int(meta.get(f'{arch}.feed_forward_length', 11008))
        self.vocab_size = int(meta.get(f'{arch}.vocabulary_length',
                                     meta.get(f'{arch}.vocab_size',
                                              meta.get('tokenizer.ggml.vocab_size', 32000))))
        self.max_seq_len = int(meta.get(f'{arch}.context_length', 2048))
        rope_keys = [
            f"{arch}.rope.freq_base",
            f"{arch}.rope.frequency_base",
            f"{arch}.rope_theta",
            "llama.rope.freq_base",
            "llama.rope.frequency_base",
            "rope.freq_base",
            "general.rope.freq_base"
        ]
        rope_base_val = None
        for k in rope_keys:
            if k in meta:
                rope_base_val = meta[k]
                break
        if rope_base_val is not None:
            self.rope_base = float(rope_base_val)
        elif self.vocab_size == 128256 or self.vocab_size == 128000:
            self.rope_base = 500000.0
        else:
            self.rope_base = 10000.0

        self._norm_eps = float(meta.get(f'{arch}.attention.layer_norm_rms_epsilon', 1e-5))
        self.head_dim = int(meta.get(f'{arch}.attention.key_length', 0)) or self.dim // self.n_heads

        # MoE
        self._n_experts = int(self.m._safe_int(meta.get(f'{arch}.expert_count', 0), 0))
        self._n_experts_used = int(self.m._safe_int(meta.get(f'{arch}.expert_used_count', 0), 0))
        if self._n_experts > 0 and self._n_experts_used == 0:
            self._n_experts_used = 2
            print(f"[STREAM] MoE detected: {self._n_experts} experts, top-{self._n_experts_used}")

        # LayerNorm detection
        _ln_archs = ('gpt2', 'gptj', 'gpt-neox', 'command-r', 'falcon', 'refact',
                     'bert', 'nomic-bert', 'jina-bert', 'mini-bert')
        if any(a in arch for a in _ln_archs):
            self._use_layernorm = True
            print(f"[STREAM] Using LayerNorm (arch={arch})")

        # gemma4 RoPE theta (not typically in SafeTensors but handle if present)
        self._rope_theta_swa = float(meta.get(f'{arch}.rope.freq_base_swa', 10000.0))
        self._rope_theta_full = float(self.rope_base)
        self._g4_shared_kv = 0
        self._ple_dim = 0

        # Trust tensor shapes for head_dim (SafeTensors shapes are NOT reversed)
        if not _is_gemma4_arch(arch):
            rev = getattr(self._parser, 'shape_reversed', False)
            tq = self.m.tensors.get('blk.0.attn_q.weight')
            tk = self.m.tensors.get('blk.0.attn_k.weight')
            if tq is not None and tk is not None and len(tk.shape) == 2:
                q_w = int(tq.shape[1] if rev else tq.shape[0])
                kv_w = int(tk.shape[1] if rev else tk.shape[0])
                if self.n_heads > 0 and q_w % self.n_heads == 0:
                    self.head_dim = q_w // self.n_heads
                if self.head_dim > 0 and kv_w % self.head_dim == 0:
                    self.n_kv_heads = kv_w // self.head_dim

        print(f"[STREAM] SafeTensors Model: {self.dim}d x {self.n_layers}L x {self.n_heads}H")

        # The context ceiling comes from the ROUTER PLAN when there is one,
        # and otherwise from what memory can actually hold. It used to be a
        # literal 12000 typed in here, which silently capped a machine with
        # 8 cards at a fraction of its capacity while doing nothing to
        # protect a small one -- wrong in both directions, invisibly.
        self.max_seq_len = self._derive_ctx_limit()
        if not self.max_seq_len or int(self.max_seq_len) <= 0:
            self.max_seq_len = int(max_seq_len) if max_seq_len else 2048
            print("[STREAM] context %d (caller's value; nothing measurable "
                  "on this machine)" % self.max_seq_len, flush=True)

        _nkv = self.n_kv_heads
        if isinstance(_nkv, list):
            _nkv = max(_nkv) if _nkv else self.n_heads
        kv_dim = _nkv * self.head_dim
        _kv_dtype = np.float16 if self._kv_f16 else np.float32
        self._fit_kv_cache_to_ram(kv_dim, np.dtype(_kv_dtype).itemsize)
        _kv_shape = (self.n_layers, self.max_seq_len, kv_dim)
        if getattr(self, '_kv_disk_path', None):
            # Disk-backed KV cache via mmap - OS pages hot data to RAM,
            # cold data to SSD. Enables very long conversations without OOM.
            import os
            kp = self._kv_disk_path
            os.makedirs(os.path.dirname(kp) or '.', exist_ok=True)
            self.kv_k = np.memmap(kp + '_k.bin', dtype=_kv_dtype, mode='w+', shape=_kv_shape)
            self.kv_v = np.memmap(kp + '_v.bin', dtype=_kv_dtype, mode='w+', shape=_kv_shape)
            _kv_gb = (_kv_shape[0] * _kv_shape[1] * _kv_shape[2] * (2 if self._kv_f16 else 4)) / 1e9
            print(f'[STREAM] Disk-backed KV cache (mmap): {_kv_gb:.2f} GB at {kp}')
        else:
            self.kv_k = np.zeros(_kv_shape, dtype=_kv_dtype)
            self.kv_v = np.zeros(_kv_shape, dtype=_kv_dtype)

        self.loaded = True
        self._load_embedding_and_output()

    def _get_embedding_row(self, token_id: int) -> np.ndarray:
        """Get single token embedding vector (float32, shape (dim,))."""
        if getattr(self, '_emb_lazy_name', None) and hasattr(self, '_parser') and hasattr(self, 'm'):
            tid = int(token_id)
            if tid in self._emb_cache:
                return self._emb_cache[tid].copy()
            row = self._parser.load_tensor_row(self.m, self._emb_lazy_name, tid)
            if len(self._emb_cache) < 8192:
                self._emb_cache[tid] = row
            return row.copy()
        if self._embedding is not None:
            tid = min(max(0, int(token_id)), len(self._embedding) - 1)
            return self._embedding[tid].copy()
        return np.zeros(self.dim, dtype=np.float32)

    def _load_embedding_and_output(self):
        """Load embedding and output weights into RAM."""
        self._emb_lazy_name = None
        self._emb_cache = {}
        if hasattr(self, 'm') and hasattr(self, '_parser'):
            # GGUF format - embedding name varies across writers:
            # llama.cpp uses token_embd.weight, some Nightmare GGUFs use
            # token_emb.weight, older llama used tok_embeddings.weight.
            self._embedding = None
            for emb_name in ("token_embd.weight", "token_emb.weight", "tok_embeddings.weight"):
                if emb_name in self.m.tensors:
                    info = self.m.tensors[emb_name]
                    shape = info.shape[::-1] if getattr(self._parser, 'shape_reversed', True) else info.shape
                    n_elems = np.prod(shape)
                    # If embedding is large (>100MB f32), use fast lazy row-level dequantization
                    if n_elems * 4 > 100 * 1024 * 1024 and len(shape) == 2:
                        self._emb_lazy_name = emb_name
                        self._embedding = None
                        v_size = int(shape[0])
                        if v_size != self.vocab_size:
                            print(f"[STREAM] vocab_size corrected from embedding: {self.vocab_size} -> {v_size}")
                            self.vocab_size = v_size
                        print(f"[STREAM] Instant lazy embedding active for {emb_name} ({self.vocab_size} vocab, {self.dim}d) - 0 RAM startup")
                    else:
                        try:
                            self._embedding = self._parser.load_tensor(self.m, emb_name).astype(np.float32)
                        except Exception:
                            self._embedding = None
                    break
            if self._embedding is not None:
                # Mixed-convention files (e.g. the "90M Llama" export) store
                # 2D layer weights in standard order but token_embd in
                # llama.cpp (dim, vocab) order - orient by row width.
                if (self._embedding.ndim == 2
                        and self._embedding.shape[1] != self.dim
                        and self._embedding.shape[0] == self.dim):
                    self._embedding = np.ascontiguousarray(self._embedding.T)
                # Trust the tensor, not the metadata: some GGUFs report a
                # smaller vocab than the embedding actually holds (Gemma).
                if self._embedding.shape[0] != self.vocab_size:
                    print(f"[STREAM] vocab_size corrected from embedding: {self.vocab_size} -> {self._embedding.shape[0]}")
                    self.vocab_size = int(self._embedding.shape[0])
            elif not getattr(self, '_emb_lazy_name', None):
                self._embedding = np.zeros((self.vocab_size, self.dim), dtype=np.float32)
                
            self._try_gpu_logits()   # register before any f32 dequant
            self._output_q8 = None
            if self._output_gpu is not None:
                self._output_weight = None
            else:
                self._output_weight = None
                # Resident Q8_0 output head: the LM projection is a single
                # (vocab x dim) matvec per token - keeping it quantized and
                # running it through the AVX2 GEMV kernel avoids a ~1GB f32
                # copy and cuts that projection cost dramatically on CPU.
                oq = self._build_output_q8()
                if oq is not None:
                    self._output_q8 = oq
                else:
                    try:
                        self._output_weight = self._parser.load_tensor(self.m, "output.weight").astype(np.float32)
                    except KeyError:
                        self._output_weight = self._embedding  # Tied weights
                # Fast Load: keep tied output projection lazy.
                # _logits materializes on demand only when the first token is sampled.

            try:
                self._output_norm = self._parser.load_tensor(self.m, f"output_norm.weight").astype(np.float32)
            except KeyError:
                self._output_norm = np.ones(self.dim, dtype=np.float32)

            # gemma4 precomputed RoPE frequencies (optional)
            self._rope_freqs = None
            try:
                self._rope_freqs = self._parser.load_tensor(
                    self.m, "rope_freqs.weight").astype(np.float32).ravel()
            except KeyError:
                pass
            self._load_ple()
        elif hasattr(self, '_nsvd'):
            # NTMRE JSON container - universal rule: embedding rows are
            # contiguous in the flat stream under both storage conventions.
            try:
                flat, shape = self._nsvd.read_flat("token_embd.weight")
                n_rows = flat.size // self.dim
                self._embedding = flat.reshape((n_rows, self.dim))
                if n_rows != self.vocab_size:
                    print(f"[STREAM] vocab_size corrected from embedding: {self.vocab_size} -> {n_rows}")
                    self.vocab_size = n_rows
            except Exception:
                self._embedding = np.zeros((self.vocab_size, self.dim), dtype=np.float32)

            self._try_gpu_logits()   # register before any f32 dequant
            if self._output_gpu is not None:
                self._output_weight = None
            else:
                try:
                    flat, shape = self._nsvd.read_flat("output.weight")
                    self._output_weight = flat.reshape((flat.size // self.dim, self.dim))
                except Exception:
                    self._output_weight = self._embedding  # Tied weights

            try:
                self._output_norm, _ = self._nsvd.read_flat("output_norm.weight")
            except Exception:
                self._output_norm = np.ones(self.dim, dtype=np.float32)

            # gemma4 precomputed RoPE frequencies (optional)
            self._rope_freqs = None
            try:
                self._rope_freqs, _ = self._nsvd.read_flat("rope_freqs.weight")
            except Exception:
                pass
            self._load_ple()
        else:
            # Fallback
            self._embedding = np.zeros((self.vocab_size, self.dim), dtype=np.float32)
            self._output_weight = self._embedding
            self._output_norm = np.ones(self.dim, dtype=np.float32)

        # Try to allocate GPU-resident buffers for the streaming forward pass
        self._alloc_gpu_buffers()

    def _alloc_gpu_buffers(self):
        """Allocate persistent GPU buffers for the forward pass so the
        hidden state stays on VRAM between matmuls (no PCIe round-trips).
        Falls back gracefully - if VRAM is too tight, _gpu_buf stays None
        and the engine uses the existing CPU/numpy path."""
        gpu = getattr(self, '_gpumm', None)
        if gpu is None:
            return
        try:
            cu = gpu.cu
            _nkv = self.n_kv_heads
            if isinstance(_nkv, list):
                _nkv = max(_nkv) if _nkv else self.n_heads
            q_dim = int(self.n_heads) * int(self.head_dim)
            kv_dim = int(_nkv) * int(self.head_dim)
            if _is_gemma4_arch(getattr(self, '_architecture', '')):
                # Gemma4 mixes SWA (head_dim 256) and full (head_dim 512) layers.
                # Allocate buffers for the largest layer configuration.
                max_q = max(self._gemma4_layer_info(l)[0] for l in range(self.n_layers))
                max_k = max(self._gemma4_layer_info(l)[1] for l in range(self.n_layers))
                q_dim = max(q_dim, max_q)
                kv_dim = max(kv_dim, max_k)
            self._gpu_buf = {
                'x':    cu.alloc(int(self.dim) * 4),
                'xn':   cu.alloc(int(self.dim) * 4),
                'q':    cu.alloc(int(q_dim) * 4),
                'k':    cu.alloc(int(kv_dim) * 4),
                'v':    cu.alloc(int(kv_dim) * 4),
                'attn': cu.alloc(int(q_dim) * 4),
                'tmp':  cu.alloc(int(max(self.dim, q_dim)) * 4),
                'gate': cu.alloc(int(self.hidden_dim) * 4),
                'up':   cu.alloc(int(self.hidden_dim) * 4),
                'silu': cu.alloc(int(self.hidden_dim) * 4),
            }
            # GPU-resident MoE scratch (gemma4): 'moe' accumulates the routed
            # experts, 'attn_out' holds x after the attention residual so the
            # experts read their input from VRAM instead of the CPU path's xn.
            self._gpu_buf['moe'] = cu.alloc(int(self.dim) * 4)
            self._gpu_buf['moe_in'] = cu.alloc(int(self.dim) * 4)
            self._gpu_buf['attn_out'] = cu.alloc(int(self.dim) * 4)

            # Per-worker scratch for multi-GPU MoE.
            #
            # A CUDA pointer is only valid on the device that allocated it.
            # The buffers above all belong to self._gpumm's context, so
            # handing one to a kernel launched on the SECOND worker is a
            # cross-device dereference: it does not fault, it quietly
            # returns garbage. That is worse than a crash -- the model
            # would load, print Ready, and talk nonsense.
            #
            # So every worker that may hold an expert gets its own copy of
            # the scratch an expert computation needs, on its own device.
            # An expert is then computed start-to-finish on one card and
            # the result is copied back, with no pointer ever crossing a
            # device boundary.
            self._gpu_buf_w = {0: self._gpu_buf}
            for _wi, _w in enumerate(self._expert_workers()):
                if _wi == 0:
                    continue
                try:
                    _c = _w.cu
                    _mb = int(self.dim) * 4
                    self._gpu_buf_w[_wi] = {
                        'moe':     _c.alloc(_mb),
                        'moe_in':  _c.alloc(_mb),
                        'gate':    _c.alloc(int(self.hidden_dim) * 4),
                        'up':      _c.alloc(int(self.hidden_dim) * 4),
                        'tmp':     _c.alloc(int(max(self.dim, self.hidden_dim)) * 4),
                        'silu':    _c.alloc(int(self.hidden_dim) * 4),
                        'attn_out': _c.alloc(_mb),
                    }
                except Exception:
                    self._gpu_buf_w[_wi] = None
            if len(self._gpu_buf_w) > 1:
                _live = sorted(k for k, v in self._gpu_buf_w.items() if v)
                print("[STREAM] per-worker MoE scratch on GPU%s"
                      % ", GPU".join(str(i) for i in _live), flush=True)
            # Upload output norm to GPU
            self._gpu_output_norm = gpu.to_gpu_ptr(
                np.asarray(self._output_norm, dtype=np.float32))
            # Upload per-layer norm weights to GPU (small: dim x 4 bytes each)
            self._gpu_norm_ptrs = {}
            for layer in range(self.n_layers):
                for ntype in ('attn_norm', 'ffn_norm'):
                    name = f'blk.{layer}.{ntype}.weight'
                    w = None
                    if hasattr(self, 'm') and hasattr(self, '_parser'):
                        try:
                            w = self._parser.load_tensor(self.m, name).astype(np.float32)
                        except KeyError:
                            w = None
                    elif hasattr(self, '_nsvd'):
                        try:
                            w, _ = self._nsvd.read_flat(name)
                            w = np.asarray(w, dtype=np.float32)
                        except Exception:
                            w = None
                    if w is None:
                        w = np.ones(self.dim, dtype=np.float32)
                    self._gpu_norm_ptrs[(layer, ntype)] = gpu.to_gpu_ptr(w)

            # KV cache on GPU if it fits, else stay on CPU (numpy).
            # F16 KV cache (default) halves the VRAM cost vs F32, leaving more
            # room for resident weights -> fewer streamed matvecs -> faster.
            kv_bpe = 2 if self._kv_f16 else 4
            kv_bytes = (self.n_layers * self.max_seq_len * kv_dim * kv_bpe * 2)
            try:
                free = self._best_kv_vram()
                self._use_gpu_kv = bool(
                    free > kv_bytes + 100 * 1024 * 1024)   # 100MB headroom
            except Exception:
                free, self._use_gpu_kv = 0, False
            try:
                if self._use_gpu_kv:
                    k_ptrs = [cu.alloc(int(self.max_seq_len) * int(kv_dim) * kv_bpe)
                              for _ in range(int(self.n_layers))]
                    v_ptrs = [cu.alloc(int(self.max_seq_len) * int(kv_dim) * kv_bpe)
                              for _ in range(int(self.n_layers))]
                    self._gpu_kv = ('gpu', k_ptrs, v_ptrs, kv_dim, kv_bpe)
                    print(f"[STREAM] KV cache on GPU ({'F16' if kv_bpe == 2 else 'F32'}, {kv_bytes / 1e9:.2f} GB) "
                          f"[best card had {free / 1e9:.2f} GB free]")
                else:
                    self._gpu_kv = ('cpu', None, None, kv_dim, kv_bpe)
                    print(f"[STREAM] KV cache on CPU (no single card had {free / 1e9:.2f} GB "
                          f"free for {kv_bytes / 1e9:.2f} GB)")
            except Exception:
                self._gpu_kv = ('cpu', None, None, kv_dim, kv_bpe)
                self._use_gpu_kv = False

            # Now that our own buffers + KV cache are allocated, refresh the
            # GpuMM resident weight budget to the *remaining* free VRAM so
            # weights grab as much VRAM as possible before falling back to RAM.
            # Use a tiny 8 MB headroom; users can dial back with the env var.
            _frac = float(os.environ.get('NTMRE_RESIDENT_FRACTION', '0.995'))
            _head = int(os.environ.get('NTMRE_RESIDENT_HEADROOM_MB', '8')) * 1024 * 1024
            workers = getattr(self, '_gpu_workers', None) or ([self._gpumm] if self._gpumm else [])
            for w in workers:
                if w is not None:
                    bw = w._refresh_budget(_frac, _head)
                    print(f"[STREAM] GPU{w.device} resident weight budget refreshed -> {bw / 1e9:.2f} GB")

            # Always pre-upload as many layer weights into VRAM as possible:
            # fills GPU1 first, then GPU0 next, and RAM/disk third for whatever remains.
            self._preload_weights_to_gpu()

            print(f"[STREAM] GPU-resident buffers allocated - hidden state stays on VRAM")
        except Exception as e:
            print(f"[STREAM] GPU buffer alloc failed ({e}) - using CPU path")
            self._gpu_buf = None
            self._gpu_kv = None

    def _preload_weights_to_gpu(self):
        """Eagerly upload layer weights to GPU VRAM after buffers are allocated.
        This turns streaming from a per-layer PCIe stream into a VRAM-resident
        engine for models that fit, which is much faster."""
        if getattr(self, '_gpumm', None) is None:
            return
        print("[STREAM] Preloading layer weights into VRAM...")
        preloaded = 0
        streamed = 0
        skipped = 0

        _nh = self.n_heads
        _nkv = self.n_kv_heads
        if isinstance(_nh, (list, tuple)):
            _nh = max(_nh) if _nh else 1
        if isinstance(_nkv, (list, tuple)):
            _nkv = max(_nkv) if _nkv else _nh
        q_dim = int(_nh) * int(self.head_dim)
        kv_dim = int(_nkv) * int(self.head_dim)
        is_g4 = _is_gemma4_arch(getattr(self, '_architecture', ''))

        for layer_idx in range(int(self.n_layers)):
            if is_g4:
                q_dim, kv_dim, _, _, _, _ = self._gemma4_layer_info(layer_idx)

            weight_specs = [
                (f"blk.{layer_idx}.attn_q.weight", q_dim, self.dim),
                (f"blk.{layer_idx}.attn_k.weight", kv_dim, self.dim),
                (f"blk.{layer_idx}.attn_v.weight", kv_dim, self.dim),
                (f"blk.{layer_idx}.attn_output.weight", self.dim, q_dim),
                (f"blk.{layer_idx}.ffn_gate.weight", self.hidden_dim, self.dim),
                (f"blk.{layer_idx}.ffn_up.weight", self.hidden_dim, self.dim),
                (f"blk.{layer_idx}.ffn_down.weight", self.dim, self.hidden_dim),
            ]
            if self._ple_dim:
                weight_specs += [
                    (f"blk.{layer_idx}.inp_gate.weight", self._ple_dim, self.dim),
                    (f"blk.{layer_idx}.proj.weight", self.dim, self._ple_dim),
                ]

            for name, out_dim, in_dim in weight_specs:
                try:
                    desc = self._try_gpu_weight(name, int(out_dim), int(in_dim))
                    if desc is not None:
                        preloaded += 1
                    else:
                        skipped += 1
                except Exception:
                    streamed += 1

        print(f"[STREAM] Preload done: {preloaded} resident, {streamed} streamed, {skipped} skipped")

    def _try_gpu_logits(self):
        """Register the output matrix (vocab, dim) with the GpuMM
        accelerator for chunked GPU logits.  On success the big f32 copy
        is dropped and forward() uses _logits() instead."""
        if getattr(self, '_gpumm', None) is None or self._output_gpu is not None:
            return
        cands = []
        if getattr(self, '_parser', None) is not None and hasattr(self, 'm'):
            names = [n for n in ("output.weight", "token_embd.weight",
                                 "token_emb.weight", "tok_embeddings.weight")
                     if n in self.m.tensors]
            for n in names:
                sh = self.m.tensors[n].shape
                if len(sh) == 2 and int(sh[0]) == self.dim:
                    cands.append((n, int(sh[1]), int(sh[0])))
        elif getattr(self, '_nsvd', None) is not None:
            ent = self._nsvd.entries
            for n in ("output.weight", "token_embd.weight"):
                if n not in ent:
                    continue
                for rows in dict.fromkeys([self.vocab_size,
                                           int(getattr(self._embedding, 'shape', [0])[0])]):
                    if rows > 0:
                        cands.append((n, rows, self.dim))
        for name, out_dim, in_dim in cands:
            d = self._try_gpu_weight(name, out_dim, in_dim)
            if d is not None:
                self._output_gpu = d
                if name.startswith('output'):
                    self._output_weight = None   # free the big f32 copy
                print(f"[STREAM] Logits on GPU ({name}, {out_dim}x{in_dim})")
                return

    def _build_output_q8(self):
        """Build a resident Q8_0 descriptor for the LM head (output.weight)
        on CPU engines, or None when unavailable / unsuitable.  Q8_0 GGUFs
        are used directly; other GGUF quant types are converted to Q8_0
        once so the AVX2 GEMV kernel serves the whole head."""
        if getattr(self, '_parser', None) is None or not hasattr(self, 'm'):
            return None
        if not (not getattr(self, 'use_gpu', False)
                and getattr(self, '_gpu_buf', None) is None
                and os.environ.get('NIGHTMARE_CPU_Q8', '1') != '0'
                and _load_q8_dll() is not None):
            return None

        # 1. Fast Load from disk cache if present
        cache_path = Path(str(getattr(self, 'model_path', '')) + ".lmhead.q8")
        expected_size = self.vocab_size * (self.dim // 32 * 34)
        if cache_path.exists() and cache_path.stat().st_size == expected_size:
            try:
                with open(cache_path, "rb") as f:
                    qb = np.frombuffer(f.read(), dtype=np.uint8)
                q = Q8W(qb, self.vocab_size, self.dim, None, "output.weight (disk cached)",
                        self._parser, self.m)
                print(f"[STREAM] ⚡ Fast Load: Resident Q8_0 output head ({self.vocab_size}x{self.dim}, "
                      f"{q.nbytes//(1024*1024)} MB) loaded from disk in <0.1s")
                return q
            except Exception:
                pass

        # 2. Check for native Q8_0 tensor in model
        for cand in ("output.weight", getattr(self, '_emb_lazy_name', ''), "token_embd.weight"):
            if not cand or cand not in self.m.tensors:
                continue
            try:
                raw, qtype, shape, n = self._parser.load_tensor_raw(self.m, cand)
                out_dim, in_dim = self.vocab_size, self.dim
                if n == out_dim * in_dim and in_dim % 32 == 0 and qtype == 8:
                    qb = np.ascontiguousarray(raw, dtype=np.uint8)
                    q = Q8W(qb, out_dim, in_dim, None, cand, self._parser, self.m)
                    print(f"[STREAM] Resident Q8_0 output head ({out_dim}x{in_dim}, "
                          f"{q.nbytes//(1024*1024)} MB raw)")
                    return q
            except Exception:
                continue
        return None

    def _ensure_output_weights(self):
        """Build the output projection (LM head).  Safe to call from _logits.

        ORDER MATTERS -- do not "simplify" this back to always using the
        embedding.

        A GGUF that ships a distinct 'output.weight' has an UNTIED head:
        token_embd is the input-embedding matrix and output.weight is a
        separately trained matrix.  Substituting the embedding for the head
        is not an approximation -- measured on this suite's own GGUFs:

            Qwen2.5-1.5B-Instruct-Q4_K_M   corr(output, token_embd) = 0.997
            Adult_ExtEval_Mistral-Q4_K_S   corr(output, token_embd) = 0.019

        A near-1.0 correlation hides the substitution, which is exactly why
        the 1.5B model was the only one that appeared to work.  On an untied
        model the engine multiplies the residual stream by an unrelated
        matrix, and the result is fluent-looking garbage with entirely
        plausible logit statistics -- no NaN, no warning, no exception.

        So: use the file's own output.weight whenever it exists, and only
        fall back to the embedding when the head is genuinely tied.
        """
        if self._output_weight is not None:
            return
        _m = getattr(self, 'm', None)
        _tensors = getattr(_m, 'tensors', None) or {}
        if 'output.weight' in _tensors and getattr(self, '_parser', None) is not None:
            try:
                ow = self._parser.load_tensor(self.m, "output.weight").astype(np.float32)
                if (ow.ndim == 2 and ow.shape[1] != self.dim
                        and ow.shape[0] == self.dim):
                    ow = np.ascontiguousarray(ow.T)
                self._output_weight = np.ascontiguousarray(ow)
                print(f"[STREAM] LM head from output.weight {getattr(ow, 'shape', None)}"
                      f" - head is NOT tied to token_embd")
                return
            except Exception as e:
                print(f"[STREAM] output.weight present but unreadable ({e}); "
                      "falling back to the tied embedding")
        if hasattr(self, '_emb_lazy_name') and self._emb_lazy_name and getattr(self, '_parser', None) is not None:
            try:
                ow = self._parser.load_tensor(self.m, self._emb_lazy_name).astype(np.float32)
                if (ow.ndim == 2 and ow.shape[1] != self.dim
                        and ow.shape[0] == self.dim):
                    ow = np.ascontiguousarray(ow.T)
                self._output_weight = ow
                print(f"[STREAM] Materialized tied output projection from lazy embedding ({getattr(ow, 'shape', None)})")
                self._embedding = ow
                return
            except Exception:
                pass
        if getattr(self, '_embedding', None) is not None:
            self._output_weight = self._embedding

    def _logits(self, x):
        """Output projection: GPU-chunked when registered, else CPU."""
        d = getattr(self, '_output_gpu', None)
        if d is not None:
            try:
                if d[0] == 'gpu':
                    _, key, _od, _id = d
                    gpu = self._gpu_for(key)
                    raw = (self._raw_store.get(key)
                           if gpu._raw[key][0] == 'stream' else None)
                    return gpu.matvec_chunked(x, key, raw=raw)
                return self._gpu_mm_desc(x, d)
            except Exception as e:
                print(f"[STREAM] GPU logits failed ({e}) - CPU fallback")
                self._output_gpu = None
                if self._output_weight is None:
                    # rebuild the f32 output matrix once
                    if getattr(self, '_parser', None) is not None:
                        try:
                            ow = self._parser.load_tensor(
                                self.m, "output.weight").astype(np.float32)
                            if (ow.ndim == 2 and ow.shape[1] != self.dim
                                    and ow.shape[0] == self.dim):
                                ow = np.ascontiguousarray(ow.T)
                            self._output_weight = ow
                        except KeyError:
                            self._output_weight = self._embedding
                    else:
                        flat, _s = self._nsvd.read_flat("output.weight")
                        self._output_weight = flat.reshape((flat.size // self.dim, self.dim))
        oq = getattr(self, '_output_q8', None)
        if oq is not None:
            return oq.matvec(x.reshape(1, -1))[0]
        if self._output_weight is None:
            self._ensure_output_weights()
        if self._output_weight is None:
            raise RuntimeError("Output projection unavailable (no output.weight and tied embedding unreadable).")
        fused_mv = getattr(self, '_fused_matvec', None)
        if fused_mv is not None:
            out = np.empty(self.vocab_size, dtype=np.float32)
            fused_mv(out, self._output_weight, x, self.vocab_size, self.dim)
            return out
        return self._output_weight @ x
    
    # ════════════════════════════════════════════════════════════
    # STREAMING FORWARD PASS
    # ════════════════════════════════════════════════════════════
    
    def forward(self, token_id: int, position: int) -> np.ndarray:
        """
        Full forward pass with layer streaming.
        
        Each layer is loaded -> computed -> evicted from VRAM.
        Double-buffered for overlapped transfer.
        """
        t_start = time.time()
        # Robustness: ensure ctypes is available (some sealed exe runtimes
        # fail to keep module-level imports in globals).
        if 'ctypes' not in globals():
            import ctypes as _ct
            globals()['ctypes'] = _ct
        
        # ── GPU-RESIDENT PATH: hidden state stays on VRAM, no PCIe round-trips ──
        _is_g4 = _is_gemma4_arch(getattr(self, '_architecture', ''))
        if self._gpu_buf is not None and not _is_g4:
            gpu = self._gpumm
            buf = self._gpu_buf

            # ── CUDA Graph fast path ──
            # After warmup, the entire layer loop + output projection is
            # captured into a CUDA graph and replayed each token.  This
            # eliminates ~384 Python->driver calls and the GPU-side
            # per-launch overhead (~2 ms).  Position-dependent kernel args
            # use persistent ctypes objects whose .value is updated in
            # place before each replay.
            _can_graph = (getattr(gpu.cu, '_graph_exec', None) is not None
                         or getattr(gpu.cu, '_capturing', False))
            if _can_graph and not getattr(gpu.cu, '_capturing', False):
                # ── Replay path ──
                # 1. Upload embedding to the fixed device buffer
                x = self._get_embedding_row(token_id)
                if str(getattr(self, '_architecture', '')).startswith('gemma'):
                    x = x * np.float32(np.sqrt(self.dim))
                gpu.cu.cu.cuMemcpyHtoD_v2(buf['x'], ctypes.c_void_p(x.ctypes.data),
                                       self.dim * 4)
                # 2. Update position-dependent persistent args
                self._update_graph_args(position)
                # 3. Replay the captured graph
                gpu.cu.replay_graph()
                gpu.cu.sync()
                # 4. Download logits from the fixed output buffer
                logits = gpu.from_gpu_ptr(self._graph_logits_ptr,
                                          self.vocab_size)
                self._times['total'] += time.time() - t_start
                self._tokens_generated += 1
                return logits

            # 1. Embedding lookup -> upload row to GPU
            x = self._get_embedding_row(token_id)
            if str(getattr(self, '_architecture', '')).startswith('gemma'):
                x = x * np.float32(np.sqrt(self.dim))
            gpu.cu.cu.cuMemcpyHtoD_v2(buf['x'], ctypes.c_void_p(x.ctypes.data), self.dim * 4)

            # 2. Stream through layers (weights loaded per-layer, compute on GPU)
            #    Prefetch layer N+1 while computing layer N (overlaps I/O with compute)
            _cache = getattr(self, '_layer_cache', None)
            for layer in range(self.n_layers):
                # Skip prefetch thread creation when the next layer is already
                # cached in RAM (resident weights) - saves ~5 ms/token of
                # thread-creation overhead on warm tokens.
                if layer + 1 < self.n_layers:
                    if _cache is None or (layer + 1) not in _cache:
                        self._start_prefetch(layer + 1)
                    else:
                        self._prefetch_ready.clear()
                        self._prefetch_layer = -1
                t_load_start = time.time()
                layer_weights = self._get_or_load_layer(layer)
                self._times['load'] += time.time() - t_load_start
                t_comp_start = time.time()
                self._layer_forward_gpu(buf['x'], layer, position, layer_weights)
                self._times['compute'] += time.time() - t_comp_start
                del layer_weights

            # 3. Final norm + logits (GPU -> download once)
            if self._use_layernorm:
                gpu.layernorm_gpu(buf['x'], self._gpu_output_norm, self.dim,
                                  self._norm_eps, buf['xn'])
            else:
                gpu.rmsnorm_gpu(buf['x'], self._gpu_output_norm, self.dim,
                                self._norm_eps, buf['xn'])
            # Output projection via GPU-chunked logits or regular matvec
            if self._output_gpu is not None:
                d = self._output_gpu
                if d[0] == 'gpu':
                    _, key, _od, _id = d
                    is_streamed = gpu._raw.get(key, [None])[0] == 'stream'
                    raw = self._raw_store.get(key) if is_streamed else None
                    if not is_streamed:
                        # Resident weight: fused/2-pass matvec on GPU (1-2
                        # kernels, no CPU round-trip for the hidden state).
                        # Far faster than chunked for large vocab heads.
                        out_ptr, _od = gpu.matvec_gpu(buf['xn'], key, raw=raw)
                        logits = gpu.from_gpu_ptr(out_ptr, self.vocab_size)
                        gpu.cu.free(out_ptr)
                    else:
                        logits = gpu.matvec_chunked(gpu.from_gpu_ptr(buf['xn'], self.dim), key, raw=raw)
                else:
                    logits = self._gpu_mm_desc(gpu.from_gpu_ptr(buf['xn'], self.dim), d)
            else:
                xn_cpu = gpu.from_gpu_ptr(buf['xn'], self.dim)
                logits = self._output_weight @ xn_cpu
            if _is_gemma4_arch(getattr(self, '_architecture', '')):
                logits = np.tanh(logits / 30.0) * 30.0
            # After the first warmup pass (all weights resident), try to
            # capture a CUDA graph for subsequent decode tokens.
            if not getattr(self, '_graph_tried', False):
                self._graph_tried = True
                if self._cuda_graphs_enabled():
                    if self._output_gpu is not None and self._gpu_kv is not None:
                        try:
                            self._setup_graph(token_id, position)
                        except Exception:
                            pass
            self._times['total'] += time.time() - t_start
            self._tokens_generated += 1
            return logits

        # ── CPU FALLBACK PATH (original) ──
        # 1. Embedding lookup (from RAM)
        x = self._get_embedding_row(token_id)
        # Gemma scales embeddings by sqrt(dim)
        if str(getattr(self, '_architecture', '')).startswith('gemma'):
            x = x * np.float32(np.sqrt(self.dim))
        # gemma4 Per-Layer Embeddings: per-token signal injected per layer
        self._ple_current = None
        if getattr(self, '_ple_dim', 0) and getattr(self, '_ple_model_proj', None) is not None:
            self._ple_current = self._g4_ple_vector(token_id)

        # 2. Stream through all layers
        for layer in range(self.n_layers):
            t_load_start = time.time()
            
            # Load layer weights (from disk/RAM -> VRAM or RAM)
            layer_weights = self._load_layer(layer)
            self._times['load'] += time.time() - t_load_start
            
            t_comp_start = time.time()
            
            # Compute layer forward
            x = self._layer_forward(x, layer, position, layer_weights)
            self._times['compute'] += time.time() - t_comp_start
            
            # Evict layer (free VRAM)
            del layer_weights
        
        # 3. Final norm + output projection (from RAM)
        x = self._norm(x, self._output_norm)
        logits = self._logits(x)
        if _is_gemma4_arch(getattr(self, '_architecture', '')):
            # gemma4 final logit softcapping (default cap 30)
            logits = np.tanh(logits / 30.0) * 30.0
        
        self._times['total'] += time.time() - t_start
        self._tokens_generated += 1
        
        return logits
    
    def _mm_batch_gpu(self, x_batch, w, layer, wkey, out_batch, M):
        """Batched matmul for prefill: Y[M,out] = X[M,in] @ W[out,in].T.
        Keeps result on GPU.  Handles GPU-resident, .ntmre compression, and numpy weights.
        Falls back to per-token matvec for .ntmre compression (no batched .ntmre compression kernel)."""
        gpu = self._gpumm
        if isinstance(w, tuple) and w and isinstance(w[0], str) and w[0].startswith('g'):
            kind = w[0]
            _, name, out_dim, in_dim = w
            if kind == 'gpu':
                raw = self._raw_store.get(name) if gpu._raw.get(name, [None])[0] == 'stream' else None
                gpu.matmul_batch_gpu(x_batch, name, out_batch, M, raw=raw)
                return
            elif kind in ('gsvd', 'gsvd_nsvd'):
                # Per-token .ntmre compression matvec fallback (no batched .ntmre compression kernel yet)
                for m in range(M):
                    xrow = ctypes.c_ulonglong(x_batch.value + m * in_dim * 4)
                    yrow = ctypes.c_ulonglong(out_batch.value + m * out_dim * 4)
                    if kind == 'gsvd':
                        ra = self._raw_store.get(name + '.A') if gpu._raw.get(name + '.A', [None])[0] == 'stream' else None
                        rb = self._raw_store.get(name + '.B') if gpu._raw.get(name + '.B', [None])[0] == 'stream' else None
                        gpu.matvec_svd_gpu(xrow, name + '.A', name + '.B', yrow, raw_a=ra, raw_b=rb)
                    else:
                        blob = self._raw_store.get(name) if gpu._raw.get(name, [None])[0] == 'svd_stream' else None
                        gpu.matvec_svd_reinterpret_gpu(xrow, name, out_dim, in_dim, yrow, blob=blob)
                return
        if isinstance(w, np.ndarray):
            w_f32 = np.ascontiguousarray(w, dtype=np.float32)
            raw = w_f32.tobytes()
            tname = f'_dyn_l{layer}_{wkey}'
            if not gpu.has(tname):
                gpu.upload_f32(tname, w_f32, w.shape[0], w.shape[1])
            gpu.matmul_batch_gpu(x_batch, tname, out_batch, M, raw=raw)
            return

        # fallback
        tname = f'blk.{layer}.{self._WT_NAMES.get(wkey, wkey)}'
        if not gpu.has(tname):
            gpu.upload_f32(tname, np.asarray(w, dtype=np.float32), w.shape[0], w.shape[1])
        raw = self._raw_store.get(tname) if gpu._raw.get(tname, [None])[0] == 'stream' else None
        gpu.matmul_batch_gpu(x_batch, tname, out_batch, M, raw=raw)

    def prefill_batch(self, token_ids, start_position=0, return_all_logits=False):
        """Batched prefill: process M tokens at once using batched GPU kernels.

        Replaces the per-token prefill loop - the big win is amortizing the
        weight read across all M tokens (weight dequanted once per matmul, not
        once per token).  KV cache is filled for all M tokens.  Returns logits
        for the LAST token (for sampling), or a (M, vocab) matrix of logits for
        ALL tokens when return_all_logits=True (used by speculative decoding
        verification).  Falls back to per-token forward() when the batched path
        is unavailable (CPU, gemma4, LayerNorm, MoE)."""
        # Robustness: ensure ctypes is available
        if 'ctypes' not in globals():
            import ctypes as _ct
            globals()['ctypes'] = _ct
        M = len(token_ids)
        if M == 0:
            return None
        if M == 1:
            _t0 = token_ids[0]
            if hasattr(_t0, 'reshape') and not np.isscalar(_t0):
                _t0 = int(np.asarray(_t0).reshape(-1)[0])
            return self.forward(int(_t0), start_position)

        # Fallback to per-token when the batched path cannot be used
        _is_g4 = _is_gemma4_arch(getattr(self, '_architecture', ''))
        if (self._gpu_buf is None
                or _is_g4
                or getattr(self, '_use_layernorm', False)
                or getattr(self, '_n_experts', 0) > 0):
            if self._gpu_buf is None:
                # CPU batched prefill: one pass over layers for all M tokens,
                # falls back to per-token on any mismatch (never raises).
                bl = (self._prefill_cpu_batched_gemma4(token_ids, start_position)
                      if _is_g4 else
                      self._prefill_cpu_batched(token_ids, start_position))
                if bl is not None:
                    return bl
            try:
                logits = None
                for i, tok in enumerate(token_ids):
                    logits = self.forward(tok, start_position + i)
                    if hasattr(self, 'kv_cache'):
                        self.kv_cache.advance()
                return logits
            except Exception:
                # Never let a single bad forward take down the suite: the
                # caller falls back to its own per-token prefill.
                return None

        t_start = time.time()
        gpu = self._gpumm
        cu = gpu.cu
        buf = self._gpu_buf
        dim = int(self.dim)
        _nkv = self.n_kv_heads
        if isinstance(_nkv, list):
            _nkv = max(_nkv) if _nkv else self.n_heads
        q_dim = int(self.n_heads) * int(self.head_dim)
        kv_dim = int(_nkv) * int(self.head_dim)
        if _is_gemma4_arch(getattr(self, '_architecture', '')):
            max_q = max(self._gemma4_layer_info(l)[0] for l in range(self.n_layers))
            max_k = max(self._gemma4_layer_info(l)[1] for l in range(self.n_layers))
            q_dim = max(q_dim, max_q)
            kv_dim = max(kv_dim, max_k)
        hidden = int(self.hidden_dim)
        eps = self._norm_eps
        arch = getattr(self, '_architecture', '')

        def ab(n):
            return cu.alloc(int(n) * 4)
        x_b    = ab(M * dim)
        xn_b   = ab(M * dim)
        q_b    = ab(M * q_dim)
        k_b    = ab(M * kv_dim)
        v_b    = ab(M * kv_dim)
        attn_b = ab(M * q_dim)
        tmp_b  = ab(M * dim)
        gate_b = ab(M * hidden)
        up_b   = ab(M * hidden)
        silu_b = ab(M * hidden)
        batch_ptrs = [x_b, xn_b, q_b, k_b, v_b, attn_b, tmp_b, gate_b, up_b, silu_b]

        try:
            # 1. Upload all M embeddings as a contiguous (M, dim) block
            rows = [self._get_embedding_row(t) for t in token_ids]
            emb = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
            if str(arch).startswith('gemma'):
                emb = emb * np.float32(np.sqrt(dim))
            cu.cu.cuMemcpyHtoD_v2(x_b, ctypes.c_void_p(emb.ctypes.data), M * dim * 4)

            # 2. Stream through layers (weights loaded per-layer, compute on GPU)
            for layer in range(self.n_layers):
                if layer + 1 < self.n_layers:
                    self._start_prefetch(layer + 1)
                t_load = time.time()
                layer_weights = self._get_or_load_layer(layer)
                self._times['load'] += time.time() - t_load
                t_comp = time.time()

                # ── Attention norm (batched RMSNorm) ──
                norm_ptr = self._gpu_norm_ptrs.get((layer, 'attn_norm'))
                if norm_ptr is not None:
                    gpu.batch_rmsnorm_gpu(x_b, norm_ptr, dim, M, eps, xn_b)
                else:
                    cu.cu.cuMemcpyDtoD_v2(xn_b, x_b, M * dim * 4)

                # ── QKV batched matmul ──
                self._mm_batch_gpu(xn_b, layer_weights['wq'], layer, 'wq', q_b, M)
                self._mm_batch_gpu(xn_b, layer_weights['wk'], layer, 'wk', k_b, M)
                self._mm_batch_gpu(xn_b, layer_weights['wv'], layer, 'wv', v_b, M)

                # ── QKV biases (per-token add - bias is a single vector) ──
                if layer_weights.get('bq') is not None:
                    for bkey, bbuf, bdim in (('bq', q_b, q_dim),
                                            ('bk', k_b, kv_dim),
                                            ('bv', v_b, kv_dim)):
                        bname = f'blk.{layer}.{bkey}'
                        if not gpu.has(bname):
                            gpu.upload_f32(bname, np.asarray(layer_weights[bkey], dtype=np.float32), bdim, 1)
                        bptr = gpu._raw[bname][0]
                        for m in range(M):
                            gpu.add_gpu(ctypes.c_ulonglong(bbuf.value + m * bdim * 4), bptr, bdim)

                # ── Batched RoPE (positions start_position + m) ──
                gpu.batch_rope_gpu(q_b, k_b, self.n_heads, self.n_kv_heads,
                                   self.head_dim, start_position, M, self.rope_base)

                # ── KV cache store + per-token attention ──
                if self._gpu_kv[0] == 'gpu':
                    kv_stride = self._gpu_kv[3]
                    kv_bpe = self._gpu_kv[4] if len(self._gpu_kv) > 4 else 4
                    k_cache_l = self._gpu_kv[1][layer]
                    v_cache_l = self._gpu_kv[2][layer]
                    q1 = buf['q']
                    attn1 = buf['attn']
                    for m in range(M):
                        pos = start_position + m
                        kv_off = pos * kv_stride * kv_bpe
                        if kv_bpe == 2:
                            k_src = ctypes.c_ulonglong(k_b.value + m * kv_dim * 4)
                            v_src = ctypes.c_ulonglong(v_b.value + m * kv_dim * 4)
                            k_dst = ctypes.c_ulonglong(k_cache_l.value + kv_off)
                            v_dst = ctypes.c_ulonglong(v_cache_l.value + kv_off)
                            gpu.store_f16_gpu(k_src, k_dst, kv_dim)
                            gpu.store_f16_gpu(v_src, v_dst, kv_dim)
                        else:
                            cu.cu.cuMemcpyDtoD_v2(
                                ctypes.c_ulonglong(k_cache_l.value + kv_off),
                                ctypes.c_ulonglong(k_b.value + m * kv_dim * 4), kv_dim * 4)
                            cu.cu.cuMemcpyDtoD_v2(
                                ctypes.c_ulonglong(v_cache_l.value + kv_off),
                                ctypes.c_ulonglong(v_b.value + m * kv_dim * 4), kv_dim * 4)
                        # Attention: Q[m] attends to positions 0..pos
                        cu.cu.cuMemcpyDtoD_v2(q1, ctypes.c_ulonglong(q_b.value + m * q_dim * 4), q_dim * 4)
                        seq_len = pos + 1
                        if seq_len > 4096:
                            if kv_bpe == 2:
                                gpu.flash_attention_f16_gpu(q1, k_cache_l, v_cache_l, attn1,
                                                           self.n_heads, self.n_kv_heads,
                                                           self.head_dim, seq_len, kv_stride)
                            else:
                                gpu.flash_attention_gpu(q1, k_cache_l, v_cache_l, attn1,
                                                       self.n_heads, self.n_kv_heads,
                                                       self.head_dim, seq_len, kv_stride)
                        elif kv_bpe == 2:
                            gpu.attention_f16_gpu(q1, k_cache_l, v_cache_l, attn1,
                                                  self.n_heads, self.n_kv_heads,
                                                  self.head_dim, seq_len, kv_stride)
                        else:
                            gpu.attention_gpu(q1, k_cache_l, v_cache_l, attn1,
                                             self.n_heads, self.n_kv_heads,
                                             self.head_dim, seq_len, kv_stride)
                        cu.cu.cuMemcpyDtoD_v2(
                            ctypes.c_ulonglong(attn_b.value + m * q_dim * 4), attn1, q_dim * 4)
                else:
                    # CPU KV cache - download all K/V, store, run numpy attention
                    k_all = gpu.from_gpu_ptr(k_b, M * kv_dim).reshape(M, kv_dim)
                    v_all = gpu.from_gpu_ptr(v_b, M * kv_dim).reshape(M, kv_dim)
                    for m in range(M):
                        self.kv_k[layer, start_position + m] = k_all[m]
                        self.kv_v[layer, start_position + m] = v_all[m]
                    q_all = gpu.from_gpu_ptr(q_b, M * q_dim).reshape(M, q_dim)
                    attn_all = np.empty((M, q_dim), dtype=np.float32)
                    n_rep = self.n_heads // self.n_kv_heads
                    inv_scale = np.float32(1.0 / np.sqrt(self.head_dim))
                    for m in range(M):
                        pos = start_position + m
                        seq_len = pos + 1
                        q_heads = q_all[m].reshape(self.n_heads, self.head_dim)
                        k_heads = self.kv_k[layer, :seq_len].reshape(seq_len, self.n_kv_heads, self.head_dim)
                        v_heads = self.kv_v[layer, :seq_len].reshape(seq_len, self.n_kv_heads, self.head_dim)
                        if n_rep > 1:
                            k_heads = np.repeat(k_heads, n_rep, axis=1)
                            v_heads = np.repeat(v_heads, n_rep, axis=1)
                        scores = np.einsum('hd,phd->hp', q_heads, k_heads, optimize=True)
                        scores *= inv_scale
                        scores -= np.max(scores, axis=1, keepdims=True)
                        np.exp(scores, out=scores)
                        scores /= np.sum(scores, axis=1, keepdims=True)
                        attn_all[m] = np.einsum('hp,phd->hd', scores, v_heads, optimize=True).reshape(-1)
                    cu.cu.cuMemcpyHtoD_v2(attn_b, ctypes.c_void_p(attn_all.ctypes.data), M * q_dim * 4)

                # ── Output projection (batched) + residual ──
                self._mm_batch_gpu(attn_b, layer_weights['wo'], layer, 'wo', tmp_b, M)
                gpu.batch_add_gpu(x_b, tmp_b, dim, M)

                # ── FFN norm (batched RMSNorm) ──
                norm_ptr = self._gpu_norm_ptrs.get((layer, 'ffn_norm'))
                if norm_ptr is not None:
                    gpu.batch_rmsnorm_gpu(x_b, norm_ptr, dim, M, eps, xn_b)
                else:
                    cu.cu.cuMemcpyDtoD_v2(xn_b, x_b, M * dim * 4)

                # ── FFN (batched: gate, up, SiLU, down) + residual ──
                self._mm_batch_gpu(xn_b, layer_weights['wg'], layer, 'wg', gate_b, M)
                self._mm_batch_gpu(xn_b, layer_weights['wu'], layer, 'wu', up_b, M)
                gpu.batch_silu_mul_gpu(gate_b, up_b, hidden, M, silu_b)
                self._mm_batch_gpu(silu_b, layer_weights['wd'], layer, 'wd', tmp_b, M)
                gpu.batch_add_gpu(x_b, tmp_b, dim, M)

                self._times['compute'] += time.time() - t_comp
                del layer_weights

            # 3. Final norm + logits
            if return_all_logits:
                # Speculative-decoding verification: return logits for ALL M tokens
                gpu.batch_rmsnorm_gpu(x_b, self._gpu_output_norm, dim, M, eps, xn_b)
                if self._output_gpu is not None:
                    d = self._output_gpu
                    all_logits = np.empty((M, self.vocab_size), dtype=np.float32)
                    if d[0] == 'gpu':
                        _, key, _od, _id = d
                        is_streamed = gpu._raw.get(key, [None])[0] == 'stream'
                        raw = self._raw_store.get(key) if is_streamed else None
                        if not is_streamed:
                            # Resident weight: M fused matvecs on GPU with a
                            # single batched download - no CPU round-trip for
                            # the hidden state, Mx fewer kernel launches than
                            # the per-token chunked path.
                            logits_buf = gpu.cu.alloc(int(M) * int(self.vocab_size) * 4)
                            try:
                                for m in range(M):
                                    x_ptr = ctypes.c_ulonglong(
                                        xn_b.value + m * dim * 4)
                                    o_ptr = ctypes.c_ulonglong(
                                        logits_buf.value + m * self.vocab_size * 4)
                                    gpu.matvec_gpu(x_ptr, key, out_ptr=o_ptr, raw=raw)
                                all_logits = gpu.from_gpu_ptr(
                                    logits_buf, M * self.vocab_size
                                ).reshape(M, self.vocab_size)
                            finally:
                                gpu.cu.free(logits_buf)
                        else:
                            xn_all = gpu.from_gpu_ptr(xn_b, M * dim).reshape(M, dim)
                            for m in range(M):
                                all_logits[m] = gpu.matvec_chunked(
                                    np.ascontiguousarray(xn_all[m]), key, raw=raw)
                    else:
                        xn_all = gpu.from_gpu_ptr(xn_b, M * dim).reshape(M, dim)
                        for m in range(M):
                            all_logits[m] = self._gpu_mm_desc(
                                np.ascontiguousarray(xn_all[m]), d)
                else:
                    xn_all = gpu.from_gpu_ptr(xn_b, M * dim).reshape(M, dim)
                    all_logits = (xn_all @ self._output_weight.T).astype(np.float32)
                self.pos = start_position + M
                self._times['total'] += time.time() - t_start
                self._tokens_generated += M
                return all_logits

            # Default: logits for the LAST token only
            cu.cu.cuMemcpyDtoD_v2(buf['x'],
                              ctypes.c_ulonglong(x_b.value + (M - 1) * dim * 4), dim * 4)
            gpu.rmsnorm_gpu(buf['x'], self._gpu_output_norm, dim, eps, buf['xn'])

            if self._output_gpu is not None:
                d = self._output_gpu
                if d[0] == 'gpu':
                    _, key, _od, _id = d
                    is_streamed = gpu._raw.get(key, [None])[0] == 'stream'
                    raw = self._raw_store.get(key) if is_streamed else None
                    if not is_streamed:
                        out_ptr, _od = gpu.matvec_gpu(buf['xn'], key, raw=raw)
                        logits = gpu.from_gpu_ptr(out_ptr, self.vocab_size)
                        gpu.cu.free(out_ptr)
                    else:
                        logits = gpu.matvec_chunked(gpu.from_gpu_ptr(buf['xn'], dim), key, raw=raw)
                else:
                    logits = self._gpu_mm_desc(gpu.from_gpu_ptr(buf['xn'], dim), d)
            else:
                xn_cpu = gpu.from_gpu_ptr(buf['xn'], dim)
                logits = self._output_weight @ xn_cpu

            self.pos = start_position + M
            self._times['total'] += time.time() - t_start
            self._tokens_generated += M
            return logits

        except Exception as e:
            # Fall back to per-token prefill
            if return_all_logits:
                all_logits = np.empty((M, self.vocab_size), dtype=np.float32)
                for m, tok in enumerate(token_ids):
                    all_logits[m] = self.forward(tok, start_position + m)
                return all_logits
            else:
                for m, tok in enumerate(token_ids[:-1]):
                    self.forward(tok, start_position + m)
                return self.forward(token_ids[-1], start_position + M - 1)

        finally:
            for p in batch_ptrs:
                try:
                    cu.free(p)
                except Exception:
                    pass

    def verify_batch(self, token_ids, start_position=0):
        """Batched verification for speculative decoding.
        Processes M tokens through all layers and returns a (M, vocab) matrix
        of logits - one row per token.  KV cache is filled for all M tokens.
        This is the target-model verification step in speculative decoding."""
        return self.prefill_batch(token_ids, start_position, return_all_logits=True)

    def rewind_kv(self, n_tokens):
        """Rewind the KV cache position by n_tokens (for speculative decoding
        rollback).  Stale cache entries beyond the new position are simply
        overwritten on the next forward - no need to zero them since attention
        uses seq_len = pos + 1 and never reads beyond the current position."""
        self.pos = max(0, self.pos - n_tokens)

    def _load_layer(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """
        Load a single layer's weights from file.
        Returns dict of weight name -> numpy array.

        Dequantized layers are cached in RAM (LRU, byte-budgeted) so
        tokens after the first don't pay the dequant cost again.
        """
        cache = getattr(self, '_layer_cache', None)
        lock = getattr(self, '_cache_lock', None)
        if lock is None:
            self._cache_lock = threading.Lock()
            lock = self._cache_lock
        if cache is None:
            cache = self._layer_cache = {}
            self._layer_cache_bytes = 0
            self._layer_cache_order = []
            self._warmup_total = int(getattr(self, 'n_layers', 0) or 0)
            # Enterprise Dynamic Layer Cache Budgeting
            # Guarantees OS safety floor and tailors memory footprint to host capacity
            try:
                import nightmare_hardware
                _nl = int(getattr(self, 'n_layers', 42) or 42)
                _b_bytes, _b_mb, _mode = nightmare_hardware.get_layer_cache_budget_bytes(n_layers=_nl)
                self._layer_cache_budget = _b_bytes
                print(f"[STREAM] Dynamic RAM Cache Budget: {_b_mb / 1024:.2f} GB ({_mode})")
            except Exception:
                self._layer_cache_budget = 4096 * 1024 * 1024  # Safe 4 GB fallback
            cache = self._layer_cache
        with lock:
            if layer_idx in cache:
                return cache[layer_idx]
        weights = self._load_layer_uncached(layer_idx)
        self._warmup_done = max(self._warmup_done,
                                min(int(layer_idx) + 1, int(getattr(self, '_warmup_total', 0) or 0)))
        try:
            wb = sum(int(w.nbytes) for w in weights.values() if hasattr(w, 'nbytes'))
            if wb < self._layer_cache_budget:
                with lock:
                    while (self._layer_cache_bytes + wb > self._layer_cache_budget
                           and self._layer_cache_order):
                        old = self._layer_cache_order.pop(0)
                        old_w = self._layer_cache.pop(old, None)
                        if old_w:
                            self._layer_cache_bytes -= sum(int(w.nbytes) for w in old_w.values() if hasattr(w, 'nbytes'))
                    cache[layer_idx] = weights
                    self._layer_cache_order.append(layer_idx)
                    self._layer_cache_bytes += wb
        except Exception:
            pass
        return weights

    def warmup_progress(self):
        """(done, total, detail) readable by the suite UI: layers resident so
        far vs n_layers.  Used to animate the first slow pass (Q4_K/Q6_K ->
        resident Q8_0 requant) instead of showing a frozen prefill bar."""
        total = int(getattr(self, '_warmup_total', 0) or 0)
        done = int(getattr(self, '_warmup_done', 0) or 0)
        if not total or done >= total:
            return (total, total, '')
        arch = str(getattr(self, '_architecture', ''))
        detail = getattr(self, '_warmup_detail', '') or f'warming {arch}'
        return (done, total, detail)

    def prewarm_layers(self):
        """CPU Turbo warm-up: dequant every layer into the resident cache in a
        background thread so the first prompt doesn't stall on disk + dequant.
        Harmless no-op on GPU-resident setups (cache budget already full)."""
        def _warm():
            try:
                for i in range(int(getattr(self, 'n_layers', 0) or 0)):
                    self._load_layer(i)
            except Exception:
                pass
        try:
            t = threading.Thread(target=_warm, daemon=True)
            t.start()
        except Exception:
            pass
        return t

    def _prefill_cpu_batched(self, token_ids, start_position=0):
        """CPU batched prefill for dense classic arches (llama/qwen2/gemma3).
        Processes M prompt tokens through one set of per-layer GEMMs instead of
        M per-token matvecs, so dequant + BLAS setup is paid once per layer.

        Numerically mirrors _layer_forward semantics: RMSNorm, split-half RoPE
        from the decoded-position tables, causal attention over every KV
        position (wo sliding-window masking, exactly like decode), GQA rep,
        SwiGLU.  Returns logits of the LAST token (like prefill_batch).

        Returns None when the batched path can't be used (gemma4, LayerNorm,
        MoE, no KV arrays, or any shape/weight-descriptor mismatch) so the
        caller can fall back to the per-token loop.  Raises never escape."""
        try:
            M = len(token_ids)
            if M < 2:
                return None
            arch = str(getattr(self, '_architecture', '')).lower()
            if (self._gpu_buf is not None
                    or _is_gemma4_arch(arch)
                    or getattr(self, '_use_layernorm', False)
                    or getattr(self, '_n_experts', 0) > 0
                    or not hasattr(self, 'kv_k') or self.kv_k is None):
                return None
            nkv = self.n_kv_heads
            if isinstance(nkv, (list, tuple)):
                nkv = max(nkv) if nkv else self.n_heads
            if not nkv or not self.n_heads:
                return None
            hd = int(self.head_dim)
            half = hd // 2
            nh = int(self.n_heads)
            n_rep = nh // nkv
            if n_rep < 1:
                n_rep = 1
            if half < 1:
                return None
            eps = getattr(self, '_norm_eps', 1e-5)

            x = np.stack([self._get_embedding_row(t) for t in token_ids], axis=0).astype(np.float32)
            if arch.startswith('gemma'):
                x = x * np.float32(np.sqrt(self.dim))

            if not hasattr(self, '_rope_cos_tab'):
                _freqs = 1.0 / (self.rope_base ** (np.arange(half, dtype=np.float32) * 2.0 / hd))
                _pos = np.arange(self.max_seq_len, dtype=np.float32)
                _ang = np.outer(_pos, _freqs)
                self._rope_cos_tab = np.cos(_ang).astype(np.float32)
                self._rope_sin_tab = np.sin(_ang).astype(np.float32)
            _c = self._rope_cos_tab[start_position:start_position + M]  # (M, half)
            _s = self._rope_sin_tab[start_position:start_position + M]

            for layer in range(self.n_layers):
                w = self._load_layer(layer)
                # guard: batched path needs plain numpy weights or resident
                # Q8W (GPU descriptor tuples bail)
                for wk in ('wq', 'wk', 'wv', 'wo', 'wg', 'wu', 'wd'):
                    if isinstance(w.get(wk), tuple):
                        return None

                # batched RMSNorm (axis=-1; scalar means would be wrong for 2-D)
                def _rms(xx, ww, ee):
                    ms = np.mean(xx * xx, axis=-1, keepdims=True) + ee
                    return xx / np.sqrt(ms) * ww

                xn = _rms(x, w['attn_norm'], eps)                     # (M, dim)
                q = self._bm(xn, w['wq'])                          # (M, q_dim)
                k = self._bm(xn, w['wk'])                          # (M, kv_dim)
                v = self._bm(xn, w['wv']) if w.get('wv') is not None else k.copy()
                if w.get('bq') is not None:
                    q = q + w['bq']
                    k = k + w['bk']
                    v = v + w['bv']

                q_r = q.reshape(M, nh, hd)
                k_r = k.reshape(M, nkv, hd)
                q0 = q_r[:, :, :half].copy(); q1 = q_r[:, :, half:].copy()
                q_r[:, :, :half] = q0 * _c[:, None, :] - q1 * _s[:, None, :]
                q_r[:, :, half:] = q0 * _s[:, None, :] + q1 * _c[:, None, :]
                k0 = k_r[:, :, :half].copy(); k1 = k_r[:, :, half:].copy()
                k_r[:, :, :half] = k0 * _c[:, None, :] - k1 * _s[:, None, :]
                k_r[:, :, half:] = k0 * _s[:, None, :] + k1 * _c[:, None, :]

                lo = start_position
                self.kv_k[layer, lo:lo + M] = k_r.reshape(M, -1)
                self.kv_v[layer, lo:lo + M] = v.reshape(M, -1)

                seq = lo + M
                K = self.kv_k[layer, :seq].reshape(seq, nkv, hd)
                V = self.kv_v[layer, :seq].reshape(seq, nkv, hd)
                if n_rep > 1:
                    K = np.repeat(K, n_rep, axis=1)
                    V = np.repeat(V, n_rep, axis=1)
                # causal scores (M q rows, seq keys); p <= lo+m is allowed
                scores = np.einsum('mhd,phd->mhp', q_r, K, optimize=True)  # (M, nh, seq)
                scores = scores * np.float32(1.0 / np.sqrt(hd))
                score_max = np.max(scores, axis=-1, keepdims=True)
                scores = scores - score_max
                np.exp(scores, out=scores)
                # causal mask always: token m may only attend p < lo+m+1
                mask = np.arange(seq)[None, :] < (lo + np.arange(M) + 1)[:, None]
                scores = scores * mask[:, None, :]
                probs = scores / (np.sum(scores, axis=-1, keepdims=True) + 1e-30)
                attn = np.einsum('mhp,phd->mhd', probs, V, optimize=True)  # (M, nh, hd)
                x = x + self._bm(attn.reshape(M, -1), w['wo'])

                xn = _rms(x, w['ffn_norm'], eps)
                if self._n_experts > 0:
                    return None  # MoE stays per-token
                gate = self._bm(xn, w['wg'])
                up = self._bm(xn, w['wu'])
                gate_act = gate / (1.0 + np.exp(-np.clip(gate, -88, 88)))
                x = x + self._bm(gate_act * up, w['wd'])

            x_last = self._norm(x[-1], self._output_norm, eps)
            return self._logits(x_last)
        except Exception:
            return None

    def _g4_ple_vector_batch(self, token_ids):
        """(M, n_layers, ple_dim) PLE signals for a batch of tokens - same
        math as _g4_ple_vector but vectorized across the batch."""
        nl = self.n_layers
        pd = self._ple_dim
        if pd > 1024 and nl > 0 and pd % nl == 0:
            pd = pd // nl
            self._ple_dim = pd
        if hasattr(self._ple_model_proj, 'shape') and nl > 0:
            proj_out = self._ple_model_proj.shape[0]
            if proj_out % nl == 0:
                pd = proj_out // nl
                self._ple_dim = pd
        embs = np.stack([self._get_embedding_row(t) for t in token_ids], axis=0).astype(np.float32)
        ctx = embs @ self._ple_model_proj.T                    # (M, nl*pd)
        ctx = ctx.reshape(-1, nl, pd)
        ms = np.mean(ctx * ctx, axis=-1, keepdims=True) + self._norm_eps
        ctx = ctx / np.sqrt(ms) * self._ple_proj_norm
        if self._ple_tok is not None:
            tok = np.stack([self._ple_tok_row(t).reshape(nl, pd) for t in token_ids])
        else:
            tok = self._ple_tok_table[np.asarray(token_ids)].reshape(-1, nl, pd)
        return (ctx + tok * np.float32(np.sqrt(pd))) * np.float32(1.0 / np.sqrt(2.0))

    def _prefill_cpu_batched_gemma4(self, token_ids, start_position=0):
        """Batched CPU prefill for gemma4 (mirrors _layer_forward_gemma4):
        per-layer head dims (SWA 256 / full 512), per-head QK-/V-norms,
        split-half RoPE per layer type, attention scale 1.0, causal mask,
        sandwich post-norms, GELU-tanh FFN, PLE injection, layer_output_scale,
        sqrt(dim)-scaled embeddings and logit softcap.  Returns logits of the
        LAST token; None on any mismatch so the caller stays on the per-token
        loop.  Raises never escape."""
        try:
            M = len(token_ids)
            if M < 2:
                return None
            arch = str(getattr(self, '_architecture', '')).lower()
            if (self._gpu_buf is not None
                    or not _is_gemma4_arch(arch)
                    or not hasattr(self, 'kv_k') or self.kv_k is None):
                return None
            eps = getattr(self, '_norm_eps', 1e-6)
            scale_d = np.float32(np.sqrt(self.dim))

            x = np.stack([self._get_embedding_row(t) for t in token_ids], axis=0).astype(np.float32)
            x = x * scale_d

            ple_all = None
            if getattr(self, '_ple_dim', 0) and getattr(self, '_ple_model_proj', None) is not None:
                ple_all = self._g4_ple_vector_batch(token_ids)

            def _rms(xx, ww, ee):
                ms = np.mean(xx * xx, axis=-1, keepdims=True) + ee
                return xx / np.sqrt(ms) * ww

            share = self._g4_share_map()
            lo = start_position
            for layer in range(self.n_layers):
                w = self._load_layer(layer)
                for wk in ('wq', 'wk', 'wv', 'wo', 'wg', 'wu', 'wd'):
                    v2 = w.get(wk)
                    if isinstance(v2, tuple):
                        return None
                _qo, _ko, _has_v, hd, nh, nkv = self._gemma4_layer_info(layer)
                kvd = nkv * hd
                n_rep = max(1, nh // nkv)
                share_src = share.get(layer)

                xn = _rms(x, w['attn_norm'], eps)
                q = self._bm(xn, w['wq'])
                k_r = None
                v_r = None
                if share_src is None:
                    k = self._bm(xn, w['wk'])
                    wv = w.get('wv')
                    if wv is not None and not (isinstance(wv, np.ndarray) and not wv.any()):
                        v = self._bm(xn, wv)
                    else:
                        v = k.copy()                 # k_eq_v (V=K, pre-norm/pre-rope)
                    v_r = v.reshape(M, nkv, hd)
                    ms_v = np.mean(v_r * v_r, axis=-1, keepdims=True) + eps
                    v_r = v_r / np.sqrt(ms_v)
                    k_r = k.reshape(M, nkv, hd)
                    wkn = w.get('attn_k_norm')
                    if wkn is not None:
                        ms_k = np.mean(k_r * k_r, axis=-1, keepdims=True) + eps
                        k_r = k_r / np.sqrt(ms_k) * wkn

                q_r = q.reshape(M, nh, hd)
                wqn = w.get('attn_q_norm')
                if wqn is not None:
                    ms_q = np.mean(q_r * q_r, axis=-1, keepdims=True) + eps
                    q_r = q_r / np.sqrt(ms_q) * wqn

                cos_t, sin_t = self._gemma4_rope_tables(hd)
                half = hd // 2
                _c = cos_t[lo:lo + M]
                _s = sin_t[lo:lo + M]
                q0 = q_r[:, :, :half].copy(); q1 = q_r[:, :, half:].copy()
                q_r[:, :, :half] = q0 * _c[:, None, :] - q1 * _s[:, None, :]
                q_r[:, :, half:] = q0 * _s[:, None, :] + q1 * _c[:, None, :]

                if share_src is None:
                    k0 = k_r[:, :, :half].copy(); k1 = k_r[:, :, half:].copy()
                    k_r[:, :, :half] = k0 * _c[:, None, :] - k1 * _s[:, None, :]
                    k_r[:, :, half:] = k0 * _s[:, None, :] + k1 * _c[:, None, :]
                    self.kv_k[layer, lo:lo + M, :kvd] = k_r.reshape(M, -1)
                    self.kv_v[layer, lo:lo + M, :kvd] = v_r.reshape(M, -1)
                    src = layer
                else:
                    src = share_src

                seq = lo + M
                K = self.kv_k[src, :seq, :kvd].reshape(seq, nkv, hd)
                V = self.kv_v[src, :seq, :kvd].reshape(seq, nkv, hd)
                q3 = q_r.reshape(M, nkv, n_rep, hd)
                scores = np.einsum('mgrd,pgd->mgrp', q3, K, optimize=True)
                score_max = np.max(scores, axis=-1, keepdims=True)
                scores = scores - score_max
                np.exp(scores, out=scores)
                mask = (np.arange(seq)[None, None, None, :]
                        < (lo + np.arange(M) + 1)[:, None, None, None])
                scores = scores * mask
                probs = scores / (np.sum(scores, axis=-1, keepdims=True) + 1e-30)
                attn = np.einsum('mgrp,pgd->mgrd', probs, V, optimize=True)
                a = self._bm(attn.reshape(M, -1), w['wo'])
                pan = w.get('post_attention_norm')
                if pan is not None:
                    a = _rms(a, pan, eps)
                x = x + a

                xn = _rms(x, w['ffn_norm'], eps)
                gate = self._bm(xn, w['wg'])
                up = self._bm(xn, w['wu'])
                act = 0.5 * gate * (1.0 + np.tanh(np.sqrt(2.0 / np.pi)
                                                  * (gate + 0.044715 * gate ** 3))) * up
                f = self._bm(act, w['wd'])
                pfn = w.get('post_ffw_norm')
                if pfn is not None:
                    f = _rms(f, pfn, eps)
                x = x + f

                if ple_all is not None and w.get('ple_inp_gate') is not None:
                    g = self._bm(x, w['ple_inp_gate']) if hasattr(self, '_bm') else (x @ w['ple_inp_gate'].T)
                    g = 0.5 * g * (1.0 + np.tanh(np.sqrt(2.0 / np.pi)
                                                 * (g + 0.044715 * g ** 3)))
                    g = g * ple_all[:, layer, :]
                    s = self._bm(g, w['ple_proj']) if hasattr(self, '_bm') else (g @ w['ple_proj'].T)
                    s = _rms(s, w['ple_post_norm'], eps)
                    x = x + s

                ls = w.get('layer_output_scale')
                if ls is not None:
                    x = x * float(np.ravel(ls)[0])

            x_last = self._norm(x[-1], self._output_norm, eps)
            logits = self._logits(x_last)
            if _is_gemma4_arch(arch):
                logits = np.tanh(logits / 30.0) * 30.0
            return logits
        except Exception as e:
            return None

    def _load_layer_uncached(self, layer_idx: int) -> Dict[str, np.ndarray]:
        """Actual per-layer load + dequant (see _load_layer wrapper)."""
        if hasattr(self, 'm') and hasattr(self, '_parser'):
            def _get(*names):
                for name in names:
                    try:
                        w = self._parser.load_tensor(self.m, f"blk.{layer_idx}.{name}")
                        return w.astype(np.float32)
                    except KeyError:
                        continue
                return None

            def _or(arr, default):
                return arr if arr is not None else default

            # 2D weights prefer raw-quant GPU descriptors (no CPU dequant,
            # no f32 cache pressure); anything unusual falls back to CPU.
            _nh = self.n_heads
            _nkv = self.n_kv_heads
            if isinstance(_nh, (list, tuple)):
                _nh = max(_nh) if _nh else 1
            if isinstance(_nkv, (list, tuple)):
                _nkv = max(_nkv) if _nkv else _nh
            q_dim = _nh * self.head_dim
            kv_dim2 = _nkv * self.head_dim
            if _is_gemma4_arch(getattr(self, '_architecture', '')):
                # gemma4 dual attention: per-layer head_dim (256 SWA / 512
                # full) and shared-KV layers - dims from the actual tensors
                q_dim, kv_dim2, _hv, _hd, _nq, _nkv = self._gemma4_layer_info(layer_idx)

            # Resident Q8_0 weights: enabled on CPU-only engines for GGUF
            # matmul tensors.  Keeps 1 byte/elem in the layer cache (4x less
            # RAM than f32) and runs every decode token through the AVX2
            # GEMV kernel.  Opt out with NIGHTMARE_CPU_Q8=0.
            q8_cpu = (not getattr(self, 'use_gpu', False)
                      and getattr(self, '_gpu_buf', None) is None
                      and os.environ.get('NIGHTMARE_CPU_Q8', '1') != '0'
                      and _load_q8_dll() is not None)

            def _q8w(name, out_dim, in_dim):
                """Return a resident Q8W descriptor for this matmul weight,
                or None (caller falls back to f32).  Q8_0 GGUFs are used
                directly; any other GGUF quant type is converted to Q8_0
                once at load (numpy) so one AVX2 kernel serves everything."""
                try:
                    raw, qtype, shape, n = self._parser.load_tensor_raw(
                        self.m, f"blk.{layer_idx}.{name}")
                except Exception:
                    return None
                if in_dim % 32 != 0 or (n != out_dim * in_dim):
                    return None
                if qtype == 8:
                    qb = np.ascontiguousarray(raw, dtype=np.uint8)
                    return Q8W(qb, out_dim, in_dim, layer_idx, name,
                               self._parser, self.m)
                # Non-Q8_0 models: do not waste CPU converting to Q8_0 via Python;
                # return None so _get2d returns high-speed float32 for 12-thread OpenMP math.
                return None

            def _get2d(name, out_dim, in_dim):
                d = self._try_gpu_weight(f"blk.{layer_idx}.{name}", out_dim, in_dim)
                if d is not None:
                    return d
                if q8_cpu:
                    q = _q8w(name, out_dim, in_dim)
                    if q is not None:
                        return q
                return _get(name)

            w_q = _get2d("attn_q.weight", q_dim, self.dim)
            w_k = _get2d("attn_k.weight", kv_dim2, self.dim)
            w_v = _get2d("attn_v.weight", kv_dim2, self.dim)
            _hd = self._hidden_dim_for(layer_idx)
            
            # If QKV is fused in GGUF
            if w_q is None:
                qkv = _get("attn_qkv.weight")
                if qkv is not None:
                    # split qkv
                    hd = self.head_dim
                    q_dim = self.n_heads * hd
                    k_dim = self.n_kv_heads * hd
                    v_dim = self.n_kv_heads * hd
                    w_q = qkv[:q_dim]
                    w_k = qkv[q_dim:q_dim+k_dim]
                    w_v = qkv[q_dim+k_dim:]
            
            return {
                'attn_norm': _or(_get("attn_norm.weight", "attn_output_norm.weight"),
                                 np.ones(self.dim, dtype=np.float32)),
                'ffn_norm': _or(_get("ffn_norm.weight", "layer_output_norm.weight"),
                                np.ones(self.dim, dtype=np.float32)),
                'wq': _or(w_q, np.zeros((q_dim, self.dim), dtype=np.float32)),
                'wk': _or(w_k, np.zeros((kv_dim2, self.dim), dtype=np.float32)),
                'wv': _or(w_v, np.zeros((kv_dim2, self.dim), dtype=np.float32)),
                'wo': _or(_get2d("attn_output.weight", self.dim, q_dim), np.zeros((self.dim, q_dim), dtype=np.float32)),
                'wg': _or(_get2d("ffn_gate.weight", _hd, self.dim), np.zeros((_hd, self.dim), dtype=np.float32)),
                'wu': _or(_get2d("ffn_up.weight", _hd, self.dim), np.zeros((_hd, self.dim), dtype=np.float32)),
                'wd': _or(_get2d("ffn_down.weight", self.dim, _hd), np.zeros((self.dim, _hd), dtype=np.float32)),
                # qwen2-family QKV biases (None for llama/gemma)
                'bq': _get("attn_q.bias"),
                'bk': _get("attn_k.bias"),
                'bv': _get("attn_v.bias"),
                # gemma4 extras (None when absent)
                'attn_q_norm': _get("attn_q_norm.weight"),
                'attn_k_norm': _get("attn_k_norm.weight"),
                'post_attention_norm': _get("post_attention_norm.weight"),
                'post_ffw_norm': _get("post_ffw_norm.weight"),
                'layer_output_scale': _get("layer_output_scale.weight"),
                # gemma4 MoE: the dense FFN is a SHARED EXPERT and is SUMMED
                # with the routed experts, each through its own norm pair.
                # llama.cpp src/models/gemma4.cpp: ffn_norm -> ffn_mlp ->
                # ffn_post_norm_1, then ffn_pre_norm_2 -> ffn_moe ->
                # ffn_post_norm_2, then cur = cur_mlp + cur_moe.
                'post_ffw_norm_1': _get("post_ffw_norm_1.weight"),
                'pre_ffw_norm_2': _get("pre_ffw_norm_2.weight"),
                'post_ffw_norm_2': _get("post_ffw_norm_2.weight"),
                'moe_gate_inp_scale': _get("ffn_gate_inp.scale") if self._n_experts else None,
                # gemma4 PLE (None when absent / not an E-model)
                'ple_inp_gate': _get2d("inp_gate.weight", self._ple_dim, self.dim) if self._ple_dim else None,
                'ple_proj': _get2d("proj.weight", self.dim, self._ple_dim) if self._ple_dim else None,
                'ple_post_norm': _get("post_norm.weight") if self._ple_dim else None,
                # MoE router weight (None when dense / not MoE)
                'moe_router': _get("ffn_gate_inp.weight") if self._n_experts else None,
            }
        elif hasattr(self, '_nsvd'):
            # Universal NTMRE rule: true weight W_pt (out, in) is always
            # flat.reshape((expected_out, expected_in)) - dims come from the
            # architecture, never from the (ggml-order) shape label.
            def _get(name, out_dim=None, in_dim=None):
                if out_dim is not None and in_dim is not None:
                    d = self._try_gpu_weight(f"blk.{layer_idx}.{name}", out_dim, in_dim)
                    if d is not None:
                        return d
                try:
                    flat, shape = self._nsvd.read_flat(f"blk.{layer_idx}.{name}")
                except Exception:
                    return None
                if out_dim is None:
                    return flat  # 1D vector (norms etc.)
                return flat.reshape((out_dim, in_dim))

            _nh = self.n_heads
            _nkv = self.n_kv_heads
            if isinstance(_nh, (list, tuple)):
                _nh = max(_nh) if _nh else 1
            if isinstance(_nkv, (list, tuple)):
                _nkv = max(_nkv) if _nkv else _nh
            q_dim = _nh * self.head_dim
            kv_dim2 = _nkv * self.head_dim
            if _is_gemma4_arch(getattr(self, '_architecture', '')):
                # gemma4 dual attention: per-layer dims from the tensors
                q_dim, kv_dim2, _hv, _hd, _nq, _nkv = self._gemma4_layer_info(layer_idx)
            w_q = _get("attn_q.weight", q_dim, self.dim)
            w_k = _get("attn_k.weight", kv_dim2, self.dim)
            w_v = _get("attn_v.weight", kv_dim2, self.dim)
            _hd = self._hidden_dim_for(layer_idx)
            
            if w_q is None:
                qkv = _get("attn_qkv.weight", q_dim + 2 * kv_dim2, self.dim)
                if qkv is not None:
                    w_q = qkv[:q_dim]
                    w_k = qkv[q_dim:q_dim+kv_dim2]
                    w_v = qkv[q_dim+kv_dim2:]

            def _or(arr, default):
                return arr if arr is not None else default

            return {
                'attn_norm': _or(_get("attn_norm.weight"), np.ones(self.dim, dtype=np.float32)),
                'ffn_norm': _or(_get("ffn_norm.weight"), np.ones(self.dim, dtype=np.float32)),
                'wq': _or(w_q, np.zeros((q_dim, self.dim), dtype=np.float32)),
                'wk': _or(w_k, np.zeros((kv_dim2, self.dim), dtype=np.float32)),
                'wv': _or(w_v, np.zeros((kv_dim2, self.dim), dtype=np.float32)),
                'wo': _or(_get("attn_output.weight", self.dim, q_dim), np.zeros((self.dim, q_dim), dtype=np.float32)),
                'wg': _or(_get("ffn_gate.weight", _hd, self.dim), np.zeros((_hd, self.dim), dtype=np.float32)),
                'wu': _or(_get("ffn_up.weight", _hd, self.dim), np.zeros((_hd, self.dim), dtype=np.float32)),
                'wd': _or(_get("ffn_down.weight", self.dim, _hd), np.zeros((self.dim, _hd), dtype=np.float32)),
                # qwen2-family QKV biases (None when absent)
                'bq': _get("attn_q.bias"),
                'bk': _get("attn_k.bias"),
                'bv': _get("attn_v.bias"),
                # gemma4 extras (None when absent)
                'attn_q_norm': _get("attn_q_norm.weight"),
                'attn_k_norm': _get("attn_k_norm.weight"),
                'post_attention_norm': _get("post_attention_norm.weight"),
                'post_ffw_norm': _get("post_ffw_norm.weight"),
                'layer_output_scale': _get("layer_output_scale.weight"),
                # gemma4 MoE: the dense FFN is a SHARED EXPERT and is SUMMED
                # with the routed experts, each through its own norm pair.
                # llama.cpp src/models/gemma4.cpp: ffn_norm -> ffn_mlp ->
                # ffn_post_norm_1, then ffn_pre_norm_2 -> ffn_moe ->
                # ffn_post_norm_2, then cur = cur_mlp + cur_moe.
                'post_ffw_norm_1': _get("post_ffw_norm_1.weight"),
                'pre_ffw_norm_2': _get("pre_ffw_norm_2.weight"),
                'post_ffw_norm_2': _get("post_ffw_norm_2.weight"),
                'moe_gate_inp_scale': _get("ffn_gate_inp.scale") if self._n_experts else None,
                # gemma4 PLE (None when absent / not an E-model)
                'ple_inp_gate': _get("inp_gate.weight", self._ple_dim, self.dim) if self._ple_dim else None,
                'ple_proj': _get("proj.weight", self.dim, self._ple_dim) if self._ple_dim else None,
                'ple_post_norm': _get("post_norm.weight") if self._ple_dim else None,
                # MoE router weight (None when dense / not MoE)
                'moe_router': _get("ffn_gate_inp.weight") if self._n_experts else None,
            }
        else:
            raise RuntimeError(
                "No model source available - neither GGUF parser nor NSVD "
                "container is loaded. Cannot stream layer weights."
            )
    
    def _upload_weight(self, name, raw, gtag, out_dim, in_dim):
        """Upload a raw weight across GPUs in priority order.
        Fills GPU 0 (or 1) first; when budget is exhausted, spills into
        next available GPU(s). Records which GpuMM holds the weight."""
        workers = getattr(self, '_gpu_workers', None)
        if not workers:
            workers = [w for w in (self._gpumm, getattr(self, '_gpumm2', None)) if w is not None]
        if not workers:
            return None

        # Try to fit resident in each GPU sequentially
        for gpu in workers:
            gpu.upload_raw(name, raw, gtag, out_dim, in_dim)
            if gpu._raw.get(name, (None,))[0] != 'stream':
                self._weight_gpu[name] = gpu
                return gpu

        # If none could hold it resident, stream from primary GPU
        primary = workers[0]
        self._weight_gpu[name] = primary
        return primary

    def _gpu_for(self, name):
        """Return the GpuMM that actually holds this weight (GPU 1 or 2)."""
        return self._weight_gpu.get(name, self._gpumm)

    def _try_gpu_weight(self, name, out_dim, in_dim):
        """Register a 2D weight's RAW quantized bytes with the GpuMM
        accelerator and return a lightweight descriptor for _mm, or None
        (caller falls back to CPU dequant).  Handles GGUF sources, NSVD
        JSON containers and v16 binaries (incl. .ntmre compression factors)."""
        gpu = getattr(self, '_gpumm', None)
        if gpu is None or out_dim is None or in_dim is None:
            return None
        # Defensive: some architectures (gemma4 variants) report head counts
        # as lists.  Use the largest value so the shape check still works.
        if isinstance(out_dim, (list, tuple)):
            out_dim = max(out_dim) if out_dim else 0
        if isinstance(in_dim, (list, tuple)):
            in_dim = max(in_dim) if in_dim else 0
        if out_dim <= 0 or in_dim <= 0:
            return None
        try:
            from qwen35_gpu import (G_F32, G_F16, G_Q1_0, G_Q8_34, G_Q8_36,
                                    G_Q4_0, G_Q5_0, G_Q6_0, G_Q4_K, G_Q5_K, G_Q6_K,
                                    G_BF16, _GGML_BLOCKS)

            def _expect_bytes(gtag):
                if gtag == G_F32:
                    return out_dim * in_dim * 4
                if gtag == G_F16:
                    return out_dim * in_dim * 2
                if gtag == G_BF16:
                    return out_dim * in_dim * 2
                if gtag == G_Q1_0:
                    return (out_dim * in_dim // 128) * 18
                if gtag == G_Q8_34:
                    return (out_dim * in_dim // 32) * 34
                if gtag == G_Q8_36:
                    return (out_dim * in_dim // 32) * 36
                eb, bb = _GGML_BLOCKS[gtag]
                return (out_dim * in_dim // eb) * bb

            # GGUF source
            if getattr(self, '_parser', None) is not None and hasattr(self, 'm'):
                info = self.m.tensors.get(name)
                if info is None:
                    # Check for fused QKV or fused FFN gate/up in GGUF models
                    from safetensors_parser import quantize_to_q8_34
                    if name.endswith("attn_q.weight") or name.endswith("attn_k.weight") or name.endswith("attn_v.weight"):
                        base_pfx = name.rsplit('.', 1)[0]
                        qkv_name = base_pfx + ".attn_qkv.weight"
                        if qkv_name in self.m.tensors:
                            q_dim = self.n_heads * self.head_dim
                            k_dim = self.n_kv_heads * self.head_dim
                            v_dim = self.n_kv_heads * self.head_dim
                            full_arr = self._parser.load_tensor(self.m, qkv_name).astype(np.float32)
                            wq = np.ascontiguousarray(full_arr[:q_dim, :])
                            wk = np.ascontiguousarray(full_arr[q_dim:q_dim + k_dim, :])
                            wv = np.ascontiguousarray(full_arr[q_dim + k_dim:q_dim + k_dim + v_dim, :])
                            q_raw = quantize_to_q8_34(wq)
                            k_raw = quantize_to_q8_34(wk)
                            v_raw = quantize_to_q8_34(wv)
                            self._upload_weight(base_pfx + ".attn_q.weight", q_raw, G_Q8_34, q_dim, self.dim)
                            self._upload_weight(base_pfx + ".attn_k.weight", k_raw, G_Q8_34, k_dim, self.dim)
                            self._upload_weight(base_pfx + ".attn_v.weight", v_raw, G_Q8_34, v_dim, self.dim)
                            self._raw_store[base_pfx + ".attn_q.weight"] = q_raw
                            self._raw_store[base_pfx + ".attn_k.weight"] = k_raw
                            self._raw_store[base_pfx + ".attn_v.weight"] = v_raw
                            return ('gpu', name, int(out_dim), int(in_dim))
                    elif name.endswith("ffn_gate.weight") or name.endswith("ffn_up.weight"):
                        base_pfx = name.rsplit('.', 1)[0]
                        fused_gate_up = base_pfx + ".ffn_gate_up.weight"
                        fused_up = base_pfx + ".ffn_up.weight"
                        target_fused = fused_gate_up if fused_gate_up in self.m.tensors else (fused_up if fused_up in self.m.tensors else None)
                        if target_fused and self.m.tensors[target_fused].shape[1 if getattr(self._parser, 'shape_reversed', True) else 0] == 2 * self.hidden_dim:
                            full_arr = self._parser.load_tensor(self.m, target_fused).astype(np.float32)
                            w_gate = np.ascontiguousarray(full_arr[:self.hidden_dim, :])
                            w_up = np.ascontiguousarray(full_arr[self.hidden_dim:2 * self.hidden_dim, :])
                            gate_raw = quantize_to_q8_34(w_gate)
                            up_raw = quantize_to_q8_34(w_up)
                            self._upload_weight(base_pfx + ".ffn_gate.weight", gate_raw, G_Q8_34, self.hidden_dim, self.dim)
                            self._upload_weight(base_pfx + ".ffn_up.weight", up_raw, G_Q8_34, self.hidden_dim, self.dim)
                            self._raw_store[base_pfx + ".ffn_gate.weight"] = gate_raw
                            self._raw_store[base_pfx + ".ffn_up.weight"] = up_raw
                            return ('gpu', name, int(out_dim), int(in_dim))
                    return None
                if len(info.shape) != 2:
                    return None
                # GGUF stores shapes reversed (in, out); SafeTensors stores
                # natural (out, in).  Respect the parser's shape_reversed flag.
                rev = getattr(self._parser, 'shape_reversed', True)
                if rev:
                    if (int(info.shape[1]), int(info.shape[0])) != (int(out_dim), int(in_dim)):
                        return None
                else:
                    if (int(info.shape[0]), int(info.shape[1])) != (int(out_dim), int(in_dim)):
                        return None
                _ggml = {0: G_F32, 1: G_F16, 2: G_Q4_0, 6: G_Q5_0, 8: G_Q8_34,
                         12: G_Q4_K, 13: G_Q5_K, 14: G_Q6_K, 30: G_BF16, 41: G_Q1_0}
                gtag = _ggml.get(int(info.dtype))
                if gtag is None:
                    return None
                raw, _dt, _sh, _ne = self._parser.load_tensor_raw(self.m, name)
                raw = raw.tobytes()
                if len(raw) != _expect_bytes(gtag):
                    return None
                self._upload_weight(name, raw, gtag, out_dim, in_dim)
                self._raw_store[name] = raw
                return ('gpu', name, int(out_dim), int(in_dim))

            src = getattr(self, '_nsvd', None)
            if src is None or not hasattr(src, 'read_raw'):
                return None
            ttype, raw, shape = src.read_raw(name)
            # v16 .ntmre compression pair: factors uploaded separately
            if ttype == 'svd':
                (a_dt, raw_a), (b_dt, raw_b), (rank, out, in_) = src.read_raw_svd(name)
                if (int(out), int(in_)) != (int(out_dim), int(in_dim)):
                    return None
                # The Q8 dialect follows the source container, exactly like every other
                # weight path. Hardcoding G_Q8_36 here made a v13 .ntmre compression pair
                # upload as f32-scale/36B when the bytes on disk are
                # f16-scale/34B, so every factor decoded to noise (NaNs).
                _q8t = G_Q8_36 if getattr(src, '_q8_is_36B', False) else G_Q8_34
                ga = _q8t if a_dt == 1 else (G_F32 if a_dt == 0 else None)
                gb = _q8t if b_dt == 1 else (G_F32 if b_dt == 0 else None)
                if ga is None or gb is None:
                    return None
                self._upload_weight(name + '.A', raw_a, ga, out, rank)
                self._upload_weight(name + '.B', raw_b, gb, rank, in_)
                self._raw_store[name + '.A'] = raw_a
                self._raw_store[name + '.B'] = raw_b
                # The singular values are part of the factorization: the
                # container stores W = (U*S) @ Vt. Uploading only U and Vt
                # silently reconstructs U @ Vt, which is wrong by the whole
                # S vector, so the scales must be uploaded and applied.
                _sname = name + '.__S'
                if _sname in getattr(src, '_plain', {}):
                    _sdt, _sr, _sc, _soff, _snb = src._plain[_sname]
                    if _sdt == 0:
                        self._upload_weight(_sname,
                                            bytes(src._buf(name)[_soff:_soff + _snb]),
                                            G_F32, int(_sr), 1)
                return ('gsvd', name, int(out_dim), int(in_dim))
            # NSVD container tensor types
            from engine.bonsai_pc_app import (
                TENS_F32, TENS_Q8, TENS_SVD_Q8, TENS_Q1_0,
                TENS_Q4_0, TENS_Q5_0, TENS_Q6_0, TENS_Q4_K, TENS_Q5_K, TENS_Q6_K)
            if ttype == TENS_SVD_Q8:
                m, n = int(shape[0]), int(shape[1])
                if (m, n) != (int(in_dim), int(out_dim)):
                    return None   # numpy-layout .ntmre compression: CPU path
                # .ntmre compression-NSVD blobs go to GPU 1 (the reinterpret matvec is
                # only on the primary GpuMM); record it for matvec lookup.
                gpu.upload_svd_nsvd(name, raw, m, n)
                self._weight_gpu[name] = gpu
                self._raw_store[name] = raw
                return ('gsvd_nsvd', name, int(out_dim), int(in_dim))
            _nsvd_map = {
                TENS_F32: G_F32, TENS_Q8: G_Q8_34, TENS_Q1_0: G_Q1_0,
                TENS_Q4_0: G_Q4_0, TENS_Q5_0: G_Q5_0, TENS_Q6_0: G_Q6_0,
                TENS_Q4_K: G_Q4_K, TENS_Q5_K: G_Q5_K, TENS_Q6_K: G_Q6_K,
            }
            # v16 plain dtypes (0=F32, 1=Q8-36B) arrive as small ints;
            # NSVD TENS_* overlap (0=F32, 1=Q8-34B) - tell them apart by
            # the source class.
            from ntmre_v16_engine import V16Source
            if isinstance(src, V16Source) or getattr(src, '_q8_is_36B', False):
                gtag = {0: G_F32, 1: G_Q8_36}.get(ttype)
            else:
                gtag = _nsvd_map.get(ttype)
            if gtag is None:
                return None
            if len(raw) != _expect_bytes(gtag):
                return None
            self._upload_weight(name, raw, gtag, out_dim, in_dim)
            self._raw_store[name] = raw
            return ('gpu', name, int(out_dim), int(in_dim))
        except Exception:
            return None

    def _gpu_mm_desc(self, x1, w):
        """Run a _try_gpu_weight descriptor.  x1: (in,) f32 -> (out,)."""
        kind = w[0]
        if kind == 'gpu':
            _, key, _od, _id = w
            gpu = self._gpu_for(key)
            raw = self._raw_store.get(key) if gpu._raw[key][0] == 'stream' else None
            try:
                return gpu.matvec(x1, key, raw=raw)
            except Exception as e:
                print(f"[GPU MM ERROR] key={key} od={_od} id={_id} "
                      f"x.shape={getattr(x1,'shape','?')} x.size={getattr(x1,'size','?')}: {e}")
                raise
        if kind == 'gsvd':
            _, name, _od, _id = w
            gpu = self._gpu_for(name + '.A')
            ra = rb = None
            if gpu._raw[name + '.A'][0] == 'stream' or gpu._raw[name + '.B'][0] == 'stream':
                ra = self._raw_store.get(name + '.A')
                rb = self._raw_store.get(name + '.B')
            return gpu.matvec_svd(x1, name + '.A', name + '.B', raw_a=ra, raw_b=rb)
        if kind == 'gsvd_nsvd':
            _, name, od, id_ = w
            gpu = self._gpu_for(name)
            blob = self._raw_store.get(name) if gpu._raw[name][0] == 'svd_stream' else None
            return gpu.matvec_svd_reinterpret(x1, name, od, id_, blob=blob)
        raise ValueError(w[0])

    def _mm(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        if isinstance(w, tuple) and w and isinstance(w[0], str) and w[0].startswith('g'):
            try:
                return self._gpu_mm_desc(x.reshape(-1).astype(np.float32), w).reshape(1, -1)
            except Exception as e:
                # Some quantized descriptors fail on the GPU path (e.g. a
                # few custom-order GGUFs hit an HtoD error).  Fall back to
                # a one-time CPU dequant and cache it.
                key = w[1] if len(w) > 1 else None
                if key and key not in getattr(self, '_cpu_fallback_weights', {}):
                    if getattr(self, '_parser', None) is not None:
                        wf = self._parser.load_tensor(self.m, key).astype(np.float32)
                    elif getattr(self, '_nsvd', None) is not None:
                        flat, _sh = self._nsvd.read_flat(key)
                        wf = flat.reshape(-1, self.dim).astype(np.float32)
                    else:
                        raise
                    if not hasattr(self, '_cpu_fallback_weights'):
                        self._cpu_fallback_weights = {}
                    self._cpu_fallback_weights[key] = wf
                wf = self._cpu_fallback_weights[key]
                return x.reshape(1, -1) @ wf.T
        if isinstance(w, Q8W):
            return w.matvec(x)
        if getattr(self, 'use_gpu', False) and w is not None:
            # NvrtcEngine takes x (1, in) and w (out, in). We need to transpose w so it is (in, out)
            # Actually NvrtcEngine expects (seq, k) and (m, n). 
            # Our weights are typically (out_features, in_features). 
            return self.cu.matmul(x, w.T)
        else:
            return x @ w.T if w is not None else x

    def _np(self, w):
        """Resolve a layer weight to a plain f32 ndarray (dequantizing a
        resident Q8W on demand for batched/GEMM paths).  Q8W matvecs in
        single-token decode use the AVX2 kernel via `_mm` - no dequant."""
        if isinstance(w, Q8W):
            wf = w.to_f32()
            return wf if wf is not None else w
        return w

    def _bm(self, x, w):
        """Batched (M-row) matmul x @ w.T.  Resident Q8W weights go through
        the AVX2 q8_gemm kernel so prefill never dequants to f32; other
        weights (plain ndarray) use numpy as before."""
        if isinstance(w, Q8W):
            return w.bmm(x)
        return x @ w.T if w is not None else None

    # Weight dict key -> GGUF tensor name suffix (for numpy fallback upload)
    _WT_NAMES = {
        'wq': 'attn_q.weight', 'wk': 'attn_k.weight', 'wv': 'attn_v.weight',
        'wo': 'attn_output.weight', 'wg': 'ffn_gate.weight',
        'wu': 'ffn_up.weight', 'wd': 'ffn_down.weight',
    }

    def _mm_gpu(self, x_ptr, w, layer, wkey, out_ptr=None):
        """GPU-resident matvec: keeps result on GPU 1 (no CPU download).
        The hidden state lives on GPU 1, so every matvec MUST run there -
        a weight that overflowed to GPU 0 is 'stream' on GPU 1 and its raw
        bytes are re-supplied per matvec (safe, no cross-GPU pointers).
        Handles regular weights, .ntmre compression, and numpy fallback."""
        gpu = self._gpumm
        if isinstance(w, tuple) and w and isinstance(w[0], str) and w[0].startswith('g'):
            kind = w[0]
            _, name, out_dim, in_dim = w
            if kind == 'gpu':
                raw = self._raw_store.get(name) if gpu._raw.get(name, [None])[0] == 'stream' else None
                return gpu.matvec_gpu(x_ptr, name, out_ptr, raw=raw)
            elif kind == 'gsvd':
                ra = self._raw_store.get(name + '.A') if gpu._raw.get(name + '.A', [None])[0] == 'stream' else None
                rb = self._raw_store.get(name + '.B') if gpu._raw.get(name + '.B', [None])[0] == 'stream' else None
                return gpu.matvec_svd_gpu(x_ptr, name + '.A', name + '.B', out_ptr, raw_a=ra, raw_b=rb)
        if hasattr(w, 'A') and hasattr(w, 'B'):
            # SVDWeight object: compute matvec and copy to out_ptr
            in_dim = getattr(w, 'in_dim', self.dim)
            out_dim = getattr(w, 'out_dim', self.dim)
            x_host = gpu.from_gpu_ptr(x_ptr, in_dim)
            y_host = w.matvec(x_host).astype(np.float32)
            if out_ptr is not None:
                gpu.cu.cu.cuMemcpyHtoD_v2(out_ptr, ctypes.c_void_p(y_host.ctypes.data), y_host.nbytes)
                return out_ptr
            return gpu.to_gpu_ptr(y_host)

        if isinstance(w, np.ndarray):
            w_f32 = np.ascontiguousarray(w, dtype=np.float32)
            raw = w_f32.tobytes()
            tname = f'_dyn_l{layer}_{wkey}'
            if not gpu.has(tname):
                gpu.upload_f32(tname, w_f32, w.shape[0], w.shape[1])
            return gpu.matvec_gpu(x_ptr, tname, out_ptr, raw=raw)

        # fallback
        tname = f'blk.{layer}.{self._WT_NAMES.get(wkey, wkey)}'
        if not gpu.has(tname):
            gpu.upload_f32(tname, np.asarray(w, dtype=np.float32), w.shape[0], w.shape[1])
        raw = self._raw_store.get(tname) if gpu._raw.get(tname, [None])[0] == 'stream' else None
        return gpu.matvec_gpu(x_ptr, tname, out_ptr, raw=raw)

    def _layer_forward_gpu(self, x_ptr, layer: int, position: int,
                           weights: Dict) -> object:
        """GPU-resident layer forward: hidden state stays on VRAM.
        No PCIe round-trips for matmuls. Returns x_ptr (still on GPU)."""
        gpu = self._gpumm
        buf = self._gpu_buf
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim

        # Attention norm (GPU)
        norm_ptr = self._gpu_norm_ptrs.get((layer, 'attn_norm'))
        if norm_ptr is not None:
            if self._use_layernorm:
                gpu.layernorm_gpu(x_ptr, norm_ptr, self.dim, self._norm_eps, buf['xn'])
            else:
                gpu.rmsnorm_gpu(x_ptr, norm_ptr, self.dim, self._norm_eps, buf['xn'])
        else:
            gpu.cu.cu.cuMemcpyDtoD_v2(buf['xn'], x_ptr, self.dim * 4)

        # QKV projections (stay on GPU - no download)
        self._mm_gpu(buf['xn'], weights['wq'], layer, 'wq', buf['q'])
        self._mm_gpu(buf['xn'], weights['wk'], layer, 'wk', buf['k'])
        self._mm_gpu(buf['xn'], weights['wv'], layer, 'wv', buf['v'])

        # QKV biases (qwen2 family) - add on GPU (upload bias once, use add_gpu kernel)
        if weights.get('bq') is not None:
            for bkey, bbuf, bdim in (('bq', 'q', q_dim), ('bk', 'k', kv_dim), ('bv', 'v', kv_dim)):
                bname = f'blk.{layer}.{bkey}'
                if not gpu.has(bname):
                    gpu.upload_f32(bname, np.asarray(weights[bkey], dtype=np.float32), bdim, 1)
                bptr = gpu._raw[bname][0]
                gpu.add_gpu(buf[bbuf], bptr, bdim)

        # RoPE on GPU (in-place)
        # In graph mode, use persistent c_int so the graph can update position.
        _graph = getattr(gpu.cu, '_graph_exec', None) is not None or getattr(gpu.cu, '_capturing', False)
        if _graph:
            gpu.rope_gpu(buf['q'], buf['k'], self.n_heads, self.n_kv_heads,
                         self.head_dim, self._g_pos, self.rope_base)
        else:
            gpu.rope_gpu(buf['q'], buf['k'], self.n_heads, self.n_kv_heads,
                         self.head_dim, position, self.rope_base)

        # KV cache + attention
        if self._gpu_kv[0] == 'gpu':
            kv_stride = self._gpu_kv[3]
            kv_bpe = self._gpu_kv[4] if len(self._gpu_kv) > 4 else 4
            kv_off = position * kv_stride * kv_bpe
            if kv_bpe == 2:
                # F16 KV: convert F32 K/V -> F16 in the cache (half the VRAM),
                # then run the F16 attention kernel.
                if _graph:
                    # Persistent c_ulonglong - updated in place before replay
                    gpu.store_f16_gpu(buf['k'], self._g_k_dst[layer], kv_dim)
                    gpu.store_f16_gpu(buf['v'], self._g_v_dst[layer], kv_dim)
                    gpu.attention_f16_gpu(buf['q'], self._gpu_kv[1][layer], self._gpu_kv[2][layer],
                                          buf['attn'], self.n_heads, self.n_kv_heads,
                                          self.head_dim, self._g_seq, kv_stride)
                else:
                    k_dst = ctypes.c_ulonglong(self._gpu_kv[1][layer].value + kv_off)
                    v_dst = ctypes.c_ulonglong(self._gpu_kv[2][layer].value + kv_off)
                    gpu.store_f16_gpu(buf['k'], k_dst, kv_dim)
                    gpu.store_f16_gpu(buf['v'], v_dst, kv_dim)
                    gpu.attention_f16_gpu(buf['q'], self._gpu_kv[1][layer], self._gpu_kv[2][layer],
                                          buf['attn'], self.n_heads, self.n_kv_heads,
                                          self.head_dim, position + 1, kv_stride)
            else:
                kv_bytes = kv_dim * 4
                gpu.cu.cu.cuMemcpyDtoD_v2(ctypes.c_ulonglong(self._gpu_kv[1][layer].value + kv_off),
                                      buf['k'], kv_bytes)
                gpu.cu.cu.cuMemcpyDtoD_v2(ctypes.c_ulonglong(self._gpu_kv[2][layer].value + kv_off),
                                      buf['v'], kv_bytes)
                gpu.attention_gpu(buf['q'], self._gpu_kv[1][layer], self._gpu_kv[2][layer],
                                  buf['attn'], self.n_heads, self.n_kv_heads,
                                  self.head_dim, position + 1, kv_stride)
        else:
            # CPU KV cache - download q/k/v, vectorized numpy attention, upload result
            q_cpu = gpu.from_gpu_ptr(buf['q'], q_dim)
            k_cpu = gpu.from_gpu_ptr(buf['k'], kv_dim)
            v_cpu = gpu.from_gpu_ptr(buf['v'], kv_dim)
            self.kv_k[layer, position] = k_cpu
            self.kv_v[layer, position] = v_cpu
            seq_len = position + 1
            n_rep = self.n_heads // self.n_kv_heads
            q_heads = q_cpu.reshape(self.n_heads, self.head_dim)
            k_heads = self.kv_k[layer, :seq_len].reshape(seq_len, self.n_kv_heads, self.head_dim)
            v_heads = self.kv_v[layer, :seq_len].reshape(seq_len, self.n_kv_heads, self.head_dim)
            if n_rep > 1:
                k_heads = np.repeat(k_heads, n_rep, axis=1)
                v_heads = np.repeat(v_heads, n_rep, axis=1)
            scores = np.einsum('hd,phd->hp', q_heads, k_heads, optimize=True)
            scores *= np.float32(1.0 / np.sqrt(self.head_dim))
            scores -= np.max(scores, axis=1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= np.sum(scores, axis=1, keepdims=True)
            attn_out = np.einsum('hp,phd->hd', scores, v_heads, optimize=True).astype(np.float32)
            gpu.cu.cu.cuMemcpyHtoD_v2(buf['attn'], ctypes.c_void_p(attn_out.ctypes.data), q_dim * 4)

        # Output projection + residual (GPU)
        self._mm_gpu(buf['attn'], weights['wo'], layer, 'wo', buf['tmp'])
        gpu.add_gpu(x_ptr, buf['tmp'], self.dim)

        # FFN norm (GPU)
        norm_ptr = self._gpu_norm_ptrs.get((layer, 'ffn_norm'))
        if norm_ptr is not None:
            if self._use_layernorm:
                gpu.layernorm_gpu(x_ptr, norm_ptr, self.dim, self._norm_eps, buf['xn'])
            else:
                gpu.rmsnorm_gpu(x_ptr, norm_ptr, self.dim, self._norm_eps, buf['xn'])
        else:
            gpu.cu.cu.cuMemcpyDtoD_v2(buf['xn'], x_ptr, self.dim * 4)

        # FFN: gate, up, SiLU, down (all on GPU)
        if self._n_experts > 0 and weights.get('moe_router') is not None:
            # MoE on GPU - xn stays on VRAM, expert matmuls via matvec_gpu
            self._moe_ffn_gpu(buf['xn'], layer, weights, buf['tmp'])
        else:
            self._mm_gpu(buf['xn'], weights['wg'], layer, 'wg', buf['gate'])
            self._mm_gpu(buf['xn'], weights['wu'], layer, 'wu', buf['up'])
            gpu.silu_mul_gpu(buf['gate'], buf['up'],
                             self._hidden_dim_for(layer), buf['silu'])
            self._mm_gpu(buf['silu'], weights['wd'], layer, 'wd', buf['tmp'])
        gpu.add_gpu(x_ptr, buf['tmp'], self.dim)

        return x_ptr

    def _layer_forward(self, x: np.ndarray, layer: int, position: int,
                       weights: Dict[str, np.ndarray]) -> np.ndarray:
        """Single transformer layer forward (math with GPU matmul fallback)."""
        if _is_gemma4_arch(getattr(self, '_architecture', '')):
            return self._layer_forward_gemma4(x, layer, position, weights)
        x = x.reshape(1, -1)
        xn = self._norm(x, weights['attn_norm'])
        
        q = self._mm(xn, weights['wq'])[0]
        k = self._mm(xn, weights['wk'])[0]
        v = self._mm(xn, weights['wv'])[0]

        # qwen2-family QKV biases
        if weights.get('bq') is not None:
            q = q + weights['bq']
            k = k + weights['bk']
            v = v + weights['bv']

        # RoPE - vectorized with precomputed cos/sin table (no per-token trig)
        n_rep = self.n_heads // self.n_kv_heads
        half = self.head_dim // 2
        if not hasattr(self, '_rope_cos_tab'):
            _freqs = 1.0 / (self.rope_base ** (np.arange(half, dtype=np.float32) * 2.0 / self.head_dim))
            _pos = np.arange(self.max_seq_len, dtype=np.float32)
            _ang = np.outer(_pos, _freqs)
            self._rope_cos_tab = np.cos(_ang).astype(np.float32)
            self._rope_sin_tab = np.sin(_ang).astype(np.float32)
        _cos = self._rope_cos_tab[position]  # (half,)
        _sin = self._rope_sin_tab[position]
        # Q: (n_heads, head_dim) -> rotate each head (copy first - views would alias)
        q_r = q.reshape(self.n_heads, self.head_dim)
        q0, q1 = q_r[:, :half].copy(), q_r[:, half:].copy()
        q_r[:, :half] = q0 * _cos - q1 * _sin
        q_r[:, half:] = q0 * _sin + q1 * _cos
        q = q_r.reshape(-1)
        # K: (n_kv_heads, head_dim) -> same rotation
        k_r = k.reshape(self.n_kv_heads, self.head_dim)
        k0, k1 = k_r[:, :half].copy(), k_r[:, half:].copy()
        k_r[:, :half] = k0 * _cos - k1 * _sin
        k_r[:, half:] = k0 * _sin + k1 * _cos
        k = k_r.reshape(-1)

        # KV cache
        self.kv_k[layer, position] = k
        self.kv_v[layer, position] = v

        # Attention - vectorized (no Python loops): batch matmul + softmax
        seq_len = position + 1
        q_heads = q.reshape(self.n_heads, self.head_dim)
        k_slice = self.kv_k[layer, :seq_len]  # (seq, n_kv_heads*head_dim)
        v_slice = self.kv_v[layer, :seq_len]
        k_heads = k_slice.reshape(seq_len, self.n_kv_heads, self.head_dim)
        v_heads = v_slice.reshape(seq_len, self.n_kv_heads, self.head_dim)
        if n_rep > 1:
            k_heads = np.repeat(k_heads, n_rep, axis=1)  # (seq, n_heads, head_dim)
            v_heads = np.repeat(v_heads, n_rep, axis=1)
        # scores: (n_heads, seq) = einsum('hd,phd->hp')
        scores = np.einsum('hd,phd->hp', q_heads, k_heads, optimize=True)
        scores *= np.float32(1.0 / np.sqrt(self.head_dim))
        # stable softmax over positions (axis=1)
        scores -= np.max(scores, axis=1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= np.sum(scores, axis=1, keepdims=True)
        # weighted sum: (n_heads, head_dim) = einsum('hp,phd->hd')
        attn_out = np.einsum('hp,phd->hd', scores, v_heads, optimize=True).reshape(1, -1)
        x = x + self._mm(attn_out, weights['wo'])
        
        xn = self._norm(x, weights['ffn_norm'])
        if self._n_experts > 0 and weights.get('moe_router') is not None:
            x = x + self._moe_ffn(xn, layer, weights)
        else:
            gate = self._mm(xn, weights['wg'])
            up = self._mm(xn, weights['wu'])
            gate_act = gate / (1.0 + np.exp(-np.clip(gate, -88, 88)))
            x = x + self._mm(gate_act * up, weights['wd'])
        
        return x[0]
    
    def _moe_ffn(self, xn, layer_idx, weights):
        """Mixture-of-Experts FFN: route the token through top-k experts.

        Expert weights are loaded on demand from the source (GGUF parser or
        NSVD reader) and cached per-expert in a small LRU cache so repeated
        tokens don't re-dequantize.  Only the selected top-k experts are
        loaded - the full 3D expert tensor is never held in RAM at once."""
        router = weights['moe_router']  # (n_experts, dim) f32
        xn_flat = xn.reshape(-1).astype(np.float32)
        # Gate logits + softmax over all experts
        gate_logits = router @ xn_flat  # (n_experts,)
        # Numerically stable softmax
        gl_max = np.max(gate_logits)
        exp_gl = np.exp(gate_logits - gl_max)
        probs = exp_gl / np.sum(exp_gl)
        # Select top-k experts
        top_k = min(self._n_experts_used, self._n_experts)
        top_idx = np.argsort(probs)[-top_k:]
        top_weights = probs[top_idx]
        # Renormalize the selected expert weights
        top_weights = top_weights / np.sum(top_weights)

        out = np.zeros(self.dim, dtype=np.float32)
        for ei, w in zip(top_idx, top_weights):
            ei = int(ei)
            gate_e, up_e, down_e = self._load_expert(layer_idx, ei)
            if gate_e is None:
                continue
            # SwiGLU: silu(gate @ xn) * (up @ xn), then down @ result
            g = gate_e @ xn_flat          # (hidden_dim,)
            u = up_e @ xn_flat             # (hidden_dim,)
            act = self._expert_act(g, u)
            expert_out = down_e @ act  # (dim,)
            out += float(w) * expert_out
        return out.reshape(1, -1)

    def _expert_act(self, g, u):
        """Gated activation for a MoE expert.

        Was hardcoded to SwiGLU (silu), which is right for mixtral/qwen2moe
        and wrong for every gated-GELU family. gemma4 uses the tanh
        approximation of GeGLU:

            0.5 * g * (1 + tanh(sqrt(2/pi) * (g + 0.044715 g^3))) * u

        so a gemma4 MoE routed through the SwiGLU branch produces a plausible
        but wrong residual stream -- no exception, no NaN, wrong tokens.
        """
        if self._uses_geglu_expert():
            a = np.sqrt(2.0 / np.pi) * (g + 0.044715 * g ** 3)
            return 0.5 * g * (1.0 + np.tanh(a)) * u
        act = g / (1.0 + np.exp(-np.clip(g, -88, 88)))
        return act * u

    def _uses_geglu_expert(self):
        """True when this family's experts use GeGLU-tanh, not SwiGLU."""
        if str(getattr(self, '_ffn_activation', '') or '').lower() in ('geglu', 'geglu-tanh'):
            return True
        return str(getattr(self, '_architecture', '')).lower().startswith('gemma')

    def _moe_ffn_gpu(self, xn_ptr, layer_idx, weights, out_ptr=None):
        """GPU-resident MoE FFN: xn stays on VRAM, expert matmuls on GPU.
        Only the tiny router logits (n_experts x 4 bytes) and the final
        output (dim x 4 bytes) cross PCIe. Expert weights uploaded to
        GpuMM on first access and cached."""
        gpu = self._gpumm
        buf = self._gpu_buf
        n_exp = self._n_experts
        top_k = min(self._n_experts_used, n_exp)

        # 1. Upload router to GpuMM (cached) and compute gate logits on GPU
        router_name = f'blk.{layer_idx}.moe_router'
        if not gpu.has(router_name):
            router = weights['moe_router']
            gpu.upload_f32(router_name, np.asarray(router, dtype=np.float32),
                            n_exp, self.dim)
        gl_ptr, _ = gpu.matvec_gpu(xn_ptr, router_name, buf['tmp'])
        # Download gate logits (tiny: n_experts x 4 bytes)
        gate_logits = gpu.from_gpu_ptr(gl_ptr, n_exp)
        gl_max = np.max(gate_logits)
        exp_gl = np.exp(gate_logits - gl_max)
        probs = exp_gl / np.sum(exp_gl)
        top_idx = np.argsort(probs)[-top_k:]
        top_weights = probs[top_idx]
        top_weights = top_weights / np.sum(top_weights)

        # 2. For each selected expert: matmuls on GPU, combine on CPU
        out_cpu = np.zeros(self.dim, dtype=np.float32)
        for ei, w in zip(top_idx, top_weights):
            ei = int(ei)
            gate_e, up_e, down_e = self._load_expert(layer_idx, ei)
            if gate_e is None:
                continue
            # Upload expert weights to GpuMM (cached by name)
            gn = f'blk.{layer_idx}.exp.{ei}.gate'
            un = f'blk.{layer_idx}.exp.{ei}.up'
            dn = f'blk.{layer_idx}.exp.{ei}.down'
            if not gpu.has(gn):
                gpu.upload_f32(gn, gate_e, gate_e.shape[0], gate_e.shape[1])
                gpu.upload_f32(un, up_e, up_e.shape[0], up_e.shape[1])
                gpu.upload_f32(dn, down_e, down_e.shape[0], down_e.shape[1])
            # Expert FFN on GPU (xn stays on VRAM)
            gpu.matvec_gpu(xn_ptr, gn, buf['gate'])
            gpu.matvec_gpu(xn_ptr, un, buf['up'])
            gpu.silu_mul_gpu(buf['gate'], buf['up'], self.hidden_dim, buf['silu'])
            gpu.matvec_gpu(buf['silu'], dn, buf['tmp'])
            # Download expert output (dim floats), scale, accumulate
            expert_out = gpu.from_gpu_ptr(buf['tmp'], self.dim)
            out_cpu += float(w) * expert_out

        # 3. Upload combined output to GPU
        if out_ptr is None:
            out_ptr = buf['tmp']
        gpu.cu.cu.cuMemcpyHtoD_v2(out_ptr, ctypes.c_void_p(out_cpu.ctypes.data), self.dim * 4)
        return out_ptr

    def _expert_ffn_dim(self):
        """Width of one expert's FFN (its gate/up output size).

        NOT self.hidden_dim. gemma4-26B-A4B reports feed_forward_length=2112
        but expert_feed_forward_length=704, so anything that reshapes expert
        tensors with hidden_dim is wrong for it.
        """
        m = getattr(self, 'm', None)
        meta = getattr(m, 'metadata', None) or {}
        arch = str(getattr(self, '_architecture', '') or '')
        for key in (f"{arch}.expert_feed_forward_length",
                    f"{arch}.feed_forward_length_expert",
                    "llama.expert_feed_forward_length"):
            if key in meta:
                raw = meta[key]
                if isinstance(raw, dict):
                    raw = raw.get('value')
                try:
                    v = int(np.asarray(raw).ravel()[0]) if np.ndim(raw) else int(raw)
                except Exception:
                    continue
                if v > 0:
                    return v
        return int(self.hidden_dim)

    def _load_expert_fast(self, layer_idx, expert_idx):
        """Dequantise ONE expert by reading only its contiguous byte range.

        Why this exists: gemma4-26B-A4B stores

            ffn_gate_up_exps.weight  ne=[2816, 1408, 128]  Q4_K
            ffn_down_exps.weight     ne=[704, 2816, 128]  Q5_1
            ffn_down_exps.scale      ne=[128]             F32

        Two things break the naive path:
          * gate and up are FUSED into one tensor (ne[1] == 2*expert_ffn),
            so the three-tensor lookup for ffn_gate_exps / ffn_up_exps finds
            nothing and the FFN silently contributes zero;
          * ne[-1] is the expert axis and every expert is block-aligned, so a
            single expert is a contiguous slice -- 2.23 MB + 1.49 MB raw
            instead of dequantising a ~1 GB f32 stack.

        Returns (gate, up, down) as (expert_ffn, dim) / (expert_ffn, dim) /
        (dim, expert_ffn), or None if this layout is not recognised.
        """
        try:
            import dequant_fast as _dq
            md = self.m.metadata
            arch = getattr(self, '_architecture', '')
            n_exp = int(self.m._safe_int(md.get(f"{arch}.expert_count", 0), 0))
            e_ffn = self._expert_ffn_dim()
            if not n_exp or not e_ffn:
                return None
            base = f"blk.{layer_idx}."
            gu = self.m.tensors.get(base + "ffn_gate_up_exps.weight")
            dn = self.m.tensors.get(base + "ffn_down_exps.weight")
            if gu is None or dn is None:
                return None
            dim = int(self.dim)
            if list(gu.shape) != [dim, 2 * e_ffn, n_exp]:
                return None
            if list(dn.shape) != [e_ffn, dim, n_exp]:
                return None

            mm = getattr(self._parser, '_mmap', None)
            if mm is None:
                return None
            tdo = int(getattr(self.m, 'tensor_data_offset', 0))

            def _read(info, n_elems_total, per_expert):
                eb, bb = _dq.BLOCK_SIZES[int(info.dtype)]
                nb = (per_expert // eb) * bb
                off = tdo + int(info.offset) + int(expert_idx) * nb
                raw = np.frombuffer(mm[off:off + nb], dtype=np.uint8)
                if raw.size != nb:
                    return None
                fn = _dq.DEQUANT_FUNCTIONS.get(int(info.dtype))
                if fn is None:
                    return None
                return fn(raw.tobytes(), per_expert)

            gu_e = _read(gu, 0, dim * 2 * e_ffn)
            if gu_e is None:
                return None
            gu_e = np.ascontiguousarray(gu_e, dtype=np.float32).reshape(2 * e_ffn, dim)
            gate_e = np.ascontiguousarray(gu_e[:e_ffn])
            up_e = np.ascontiguousarray(gu_e[e_ffn:])

            dn_e = _read(dn, 0, e_ffn * dim)
            if dn_e is None:
                return None
            down_e = np.ascontiguousarray(dn_e, dtype=np.float32).reshape(dim, e_ffn)

            # Optional per-expert scale on the down projection.
            sc = self.m.tensors.get(base + "ffn_down_exps.scale")
            if sc is not None:
                try:
                    sfull = self._parser.load_tensor(self.m, base + "ffn_down_exps.scale")
                    sf = np.asarray(sfull, dtype=np.float32).ravel()
                    if sf.size == n_exp:
                        down_e = down_e * float(sf[int(expert_idx)])
                except Exception:
                    pass
            return gate_e, up_e, down_e
        except Exception:
            return None

    def _expert_cache_budget_bytes(self):
        """Size the expert LRU so one token's routing set actually fits.

        A top-k MoE layer touches n_experts_used experts, so a single token
        costs n_layer * n_experts_used * (gate+up+down) f32 values. For
        gemma4-26B-A4B that is 30 * 8 * 3 * 704 * 2816 * 4 = 5.7 GB, against a
        flat 2 GB default -- which guaranteed near-total eviction on every
        token and re-dequantised ~4.5 GB of experts per token. That, not the
        matmuls, is what pinned it at 0.05 tok/s.

        NIGHTMARE_EXPERT_CACHE_MB still overrides.
        """
        env = os.environ.get('NIGHTMARE_EXPERT_CACHE_MB')
        if env:
            try:
                return int(env) * 1024 * 1024
            except ValueError:
                pass
        e_ffn = 0
        try:
            e_ffn = int(self._expert_ffn_dim() or 0)
        except Exception:
            e_ffn = 0
        n_layer = int(getattr(self, 'n_layers', 0) or 0)
        n_top = int(getattr(self, '_n_experts_used', 0) or 0)
        if not (e_ffn and n_layer and n_top):
            return 2 * 1024 * 1024 * 1024
        per = 3 * e_ffn * int(self.dim) * 4      # gate + up + down, f32
        working = per * n_top * n_layer
        try:
            from nightmare_hardware import get_system_ram
            free = int(get_system_ram().get('avail_bytes', 0))
        except Exception:
            free = 0
        cap = int(free * 0.5) if free > 4 * 1024 ** 3 else 12 * 1024 ** 3
        return max(int(working * 1.15), per * n_top * min(n_layer, 4))\
            if working * 1.15 < cap else cap

    def _load_expert(self, layer_idx, expert_idx):
        """Load one expert's (gate, up, down) weight matrices.
        Uses a per-expert LRU cache to avoid re-dequantizing on every token."""
        if not hasattr(self, '_expert_cache'):
            self._expert_cache = {}
            self._expert_cache_order = []
            self._expert_cache_bytes = 0
            self._expert_cache_budget = self._expert_cache_budget_bytes()
        key = (layer_idx, expert_idx)
        if key in self._expert_cache:
            return self._expert_cache[key]

        gate_e = up_e = down_e = None
        if hasattr(self, 'm') and hasattr(self, '_parser'):
            expert = self._load_expert_fast(layer_idx, expert_idx)
            if expert is not None:
                gate_e, up_e, down_e = expert
            if gate_e is None and up_e is None and down_e is None:
                # Fallback: whole-tensor dequant, one expert sliced out.
                # Correct but brutal on big MoE models -- it materialises the
                # entire (n_experts, out, in) stack as f32 just to keep one
                # slice. For gemma4-26B-A4B that is ~1 GB per tensor.
                def _slice3d(name, out_d, in_d):
                    try:
                        t = self._parser.load_tensor(self.m, f"blk.{layer_idx}.{name}")
                        if t is None:
                            return None
                        t = t.astype(np.float32)
                        if t.ndim == 3:
                            return t[expert_idx]           # (out, in)
                        if t.ndim == 2:
                            return t                        # single-expert fallback
                        return None
                    except (KeyError, Exception):
                        return None
                # Routed experts use n_ff_exp (704 on gemma4-26B-A4B), NOT the
                # dense ffn_feed_forward_length -- using the dense one sliced the
                # wrong window and produced garbage expert outputs.
                _ef = self._expert_ffn_dim() or self.hidden_dim
                gate_e = _slice3d("ffn_gate_exps.weight", _ef, self.dim)
                up_e   = _slice3d("ffn_up_exps.weight",   _ef, self.dim)
                down_e = _slice3d("ffn_down_exps.weight", self.dim, _ef)
        elif hasattr(self, '_nsvd'):
            def _slice3d(name, out_d, in_d):
                try:
                    flat, shape = self._nsvd.read_flat(f"blk.{layer_idx}.{name}")
                    arr = flat.reshape((self._n_experts, out_d, in_d))
                    return arr[expert_idx].astype(np.float32)
                except Exception:
                    return None
            _ef = self._expert_ffn_dim() or self.hidden_dim
            gate_e = _slice3d("ffn_gate_exps.weight", _ef, self.dim)
            up_e   = _slice3d("ffn_up_exps.weight",   _ef, self.dim)
            down_e = _slice3d("ffn_down_exps.weight", self.dim, _ef)

        if gate_e is not None:
            result = (gate_e, up_e, down_e)
            # Cache it (LRU with byte budget)
            try:
                wb = sum(int(a.nbytes) for a in result if a is not None)
                while (self._expert_cache_bytes + wb > self._expert_cache_budget
                       and self._expert_cache_order):
                    old = self._expert_cache_order.pop(0)
                    old_r = self._expert_cache.pop(old, None)
                    if old_r:
                        self._expert_cache_bytes -= sum(
                            int(a.nbytes) for a in old_r if a is not None)
                self._expert_cache[key] = result
                self._expert_cache_order.append(key)
                self._expert_cache_bytes += wb
            except Exception:
                pass
            return result
        return (None, None, None)

    def _rms_norm(self, x, weight, eps=None):
        if eps is None:
            eps = getattr(self, '_norm_eps', 1e-5)
        ms = np.mean(x * x) + eps
        return x / np.sqrt(ms) * weight

    def _layer_norm(self, x, weight, eps=None):
        """Standard LayerNorm (with mean subtraction) - used by GPT-2,
        Command-R, Falcon, etc. instead of RMSNorm."""
        if eps is None:
            eps = getattr(self, '_norm_eps', 1e-5)
        mean = np.mean(x, axis=-1, keepdims=True)
        x = x - mean
        var = np.mean(x * x, axis=-1, keepdims=True) + eps
        return x / np.sqrt(var) * weight

    def _norm(self, x, weight, eps=None):
        """Dispatch to RMSNorm or LayerNorm based on model config."""
        if getattr(self, '_use_layernorm', False):
            return self._layer_norm(x, weight, eps)
        return self._rms_norm(x, weight, eps)

    # ── gemma4 (qk-norm + sandwich norms + layer scale + shared KV) ──
    def _load_ple(self):
        """gemma4 Per-Layer Embeddings (E-models): global PLE weights.
        The big per_layer_token_embd table stays on disk for GGUF - one
        row is read + dequantized per token (a few KB)."""
        if getattr(self, '_ple_dim', 0) <= 0:
            return
        try:
            if getattr(self, '_parser', None) is not None and hasattr(self, 'm'):
                if 'per_layer_model_proj.weight' not in self.m.tensors:
                    self._ple_dim = 0
                    return
                self._ple_model_proj = self._parser.load_tensor(
                    self.m, 'per_layer_model_proj.weight').astype(np.float32)
                self._ple_proj_norm = self._parser.load_tensor(
                    self.m, 'per_layer_proj_norm.weight').astype(np.float32).ravel()
                info = self.m.tensors['per_layer_token_embd.weight']
                # ggml ne[0] = ple_dim * n_layers (row length, contiguous)
                self._ple_tok = (str(self._parser.file_path),
                                 self.m.tensor_data_offset + int(info.offset),
                                 int(info.dtype), int(info.shape[0]))
            elif getattr(self, '_nsvd', None) is not None:
                flat, _shape = self._nsvd.read_flat('per_layer_model_proj.weight')
                self._ple_model_proj = flat.reshape((flat.size // self.dim, self.dim))
                pn, _ = self._nsvd.read_flat('per_layer_proj_norm.weight')
                self._ple_proj_norm = pn.ravel()
                flat, _shape = self._nsvd.read_flat('per_layer_token_embd.weight')
                row = self._ple_dim * self.n_layers
                self._ple_tok_table = flat.reshape((flat.size // row, row))
            else:
                self._ple_dim = 0
                return
            print(f"[STREAM] PLE loaded ({self._ple_dim}d x {self.n_layers} layers)")
        except Exception as e:
            print(f"[STREAM] PLE disabled ({e})")
            self._ple_dim = 0
            self._ple_model_proj = None
            self._ple_tok = None
            self._ple_tok_table = None

    def _ple_tok_row(self, token_id):
        """Read + dequantize one row of the packed PLE token table (GGUF)."""
        import dequant_fast
        path, off, dt, row_elems = self._ple_tok
        eb, bb = dequant_fast.BLOCK_SIZES[dt]
        row_bytes = (row_elems // eb) * bb
        with open(path, 'rb') as f:
            f.seek(off + int(token_id) * row_bytes)
            raw = f.read(row_bytes)
        return dequant_fast.DEQUANT_FUNCTIONS[dt](raw, row_elems).astype(np.float32)

    def _g4_ple_vector(self, token_id):
        """(n_layers, ple_dim) PLE signal for one token (HF/llama.cpp):
        token-identity row (x sqrt(ple_dim)) + context projection of the
        raw embedding (model_proj @ emb, RMS-normed per layer slice with
        per_layer_proj_norm), combined x 1/sqrt(2).  The projection input
        in the reference is the sqrt(dim)-scaled embedding re-scaled by
        1/sqrt(dim), i.e. simply the raw embedding row."""
        nl = self.n_layers
        pd = self._ple_dim
        if pd > 1024 and nl > 0 and pd % nl == 0:
            pd = pd // nl
            self._ple_dim = pd
        if hasattr(self._ple_model_proj, 'shape') and nl > 0:
            proj_out = self._ple_model_proj.shape[0]
            if proj_out % nl == 0:
                pd = proj_out // nl
                self._ple_dim = pd
        ctx = (self._ple_model_proj @ self._get_embedding_row(token_id)).reshape(nl, pd)
        ms = np.mean(ctx * ctx, axis=1, keepdims=True) + self._norm_eps
        ctx = ctx / np.sqrt(ms) * self._ple_proj_norm
        if self._ple_tok is not None:
            tok = self._ple_tok_row(token_id).reshape(nl, pd)
        else:
            tok = self._ple_tok_table[token_id].reshape(nl, pd)
        return (ctx + tok * np.float32(np.sqrt(pd))) * np.float32(1.0 / np.sqrt(2.0))

    def _g4_share_map(self):
        """gemma4 KV sharing: layer -> source layer (the last same-type
        layer before the shared region).  Type is identified by head_dim
        (e.g. 256 = sliding/local, 512 = full/global)."""
        m = getattr(self, '_g4_share', None)
        if m is None:
            m = {}
            n_shared = getattr(self, '_g4_shared_kv', 0)
            if n_shared > 0 and _is_gemma4_arch(getattr(self, '_architecture', '')):
                first = self.n_layers - n_shared
                for il in range(first, self.n_layers):
                    hd_il = self._gemma4_layer_info(il)[3]
                    for j in range(first - 1, -1, -1):
                        if self._gemma4_layer_info(j)[3] == hd_il:
                            m[il] = j
                            break
            self._g4_share = m
        return m

    def _cuda_graphs_enabled(self) -> bool:
        """CUDA-graph decode acceleration is OFF by default.

        Do not turn this back on without reading this.

        The engine "updates" graph parameters by mutating host-side ctypes
        objects in place -- `self._g_pos.value = position`,
        `self._g_seq.value = position + 1`, `self._g_k_dst[layer].value = ...` --
        and relies on the replay picking the new values up. It does not.
        `replay_graph()` (engine/bonsai_pc_app.py) only calls `cuGraphLaunch`;
        there is no `cuGraphExecUpdate` and no `cuGraphExecKernelNodeSetParams`.
        CUDA snapshots kernel arguments when the graph is instantiated, so every
        one of those writes is discarded and each replay re-runs the graph with
        the position, sequence length and KV write offset that happened to be
        live during capture.

        Observed on Qwen2.5-1.5B-Instruct-Q4_K_M, prompt "The capital of France
        is": greedy decode produced

            'Parisentimes / DeltaTimeaver `atters] ...' (multilingual soup)

        and two runs from reset() returned DIFFERENT garbage, because the slot
        the replay scribbled its KV into depended on when capture happened.
        Prefill was unaffected (it never replays a graph), which is why the first
        token kept coming out right and then everything after it derailed.

        Fixing this properly means moving position / seq_len / KV offset into
        DEVICE memory that the kernels dereference at replay time (the kernels in
        qwen35_gpu.KERNELS would need a `const int*` parameter), or doing a real
        `cuGraphExecKernelNodeSetParams` update per token. Neither is done.

        Until one of those lands, eager decode is the only correct path. If you
        enable this, `_verify_graph_replay()` still gates it: a graph that does
        not reproduce the eager result is destroyed and decoding falls back.
        """
        return os.environ.get("NIGHTMARE_CUDA_GRAPHS", "0") == "1"


    def _all_free_vram(self):
        """
        {device_index: free_bytes} for every GPU, or {} if unknown.

        Goes through gpu_info.get_all_free_vram() rather than guessing
        at attributes on self._gpumm -- which is a GPU *worker*, not a
        device manager, and has no device list to read. An earlier
        version of this helper looked for `_gpumm.devices`, found
        nothing, and silently fell back to the default budget on every
        single call, which would have looked like the fix did nothing.
        """
        try:
            import gpu_info
            v = gpu_info.get_all_free_vram()
            if isinstance(v, dict):
                return {int(k): int(x) for k, x in v.items()}
        except Exception:
            pass
        return {}

    def _best_kv_vram(self):
        """
        Free VRAM on the GPU best suited to holding the KV cache.

        The original code asked only `self._gpumm.device` -- whichever
        card the weights happened to land on. On a two-GPU box that
        ignores an entirely idle second card, which is how a 3060 Ti
        could be handed an 8.24 GB budget and then hold 0.39 GB.

        Returns the largest single-card free figure, because the KV
        allocator still places every layer on one device. Picking the
        emptier card is the whole win here; splitting across cards is
        separate work.
        """
        free_by_dev = self._all_free_vram()
        if free_by_dev:
            best = max(free_by_dev.values())
            if best > 0:
                return best
        try:
            import gpu_info
            return int(gpu_info.get_free_vram(getattr(self._gpumm, 'device', 0)))
        except Exception:
            return 0

    def _gpu_kv_budget_bytes(self, default=None):
        """
        Total free VRAM available to the KV cache across every card.

        Falls back to `default` (the host budget) when no card can be
        queried, so the caller still has a sane number to cap against.
        """
        if default is None:
            default = 2.0 * (1024 ** 3)
        total = sum(self._all_free_vram().values())
        return total if total > 0 else default

    def _expert_workers(self):
        """Every GPU worker that can hold an expert, indexed and tagged.

        GpuMM binds a single CUDA device at construction, so each worker
        owns exactly one card. The index is attached to the worker so a
        tensor's cache key can record which card holds it.
        """
        out = []
        for i, w in enumerate(getattr(self, '_gpu_workers', None) or []):
            if w is None:
                continue
            try:
                w._worker_index = i
            except Exception:
                pass
            out.append(w)
        if not out:
            w = getattr(self, '_gpumm', None)
            if w is not None:
                try:
                    w._worker_index = 0
                except Exception:
                    pass
                out.append(w)
        return out

    def _worker_by_index(self, idx):
        """The GPU worker tagged with this index, or None."""
        try:
            i = int(idx)
        except Exception:
            return None
        for w in self._expert_workers():
            if getattr(w, '_worker_index', 0) == i:
                return w
        return self._gpumm if i == 0 else None

    def _pick_expert_gpu(self, need):
        """
        The worker with the most room for a tensor of `need` bytes.

        Prefers a worker that can take it outright, so a token's expert
        set is not split across cards (each split would add a host
        round-trip, which is the thing being avoided). Falls back to the
        roomiest worker so the LRU eviction path still has somewhere to
        evict into. Returns None only if there are no workers at all.
        """
        workers = self._expert_workers()
        if not workers:
            return None
        best = None
        for w in workers:
            try:
                room = int(getattr(w, '_budget', 0))
            except Exception:
                room = 0
            if best is None or room > (int(getattr(best, '_budget', 0))):
                best = w
        return best

    def _total_free_vram(self):
        """
        Free VRAM summed across every card, for sizing decisions.

        Only used to report a budget the workers can collectively
        satisfy. Allocation still happens per worker -- see
        _pick_expert_gpu -- because a GpuMM cannot allocate on a card it
        was not constructed for.
        """
        free = self._all_free_vram()
        return sum(free.values()) if free else 0

    def _derive_ctx_limit(self):
        """
        The largest context this machine can hold, derived not declared.

        A literal 12000 was typed in at three loaders. That number is
        wrong on both sides: on a machine with eight cards it holds a
        26B back to a fraction of what the hardware could carry, and on
        a small one it sits above what is safe, so it protects nobody.
        Whoever runs this on their own hardware cannot see that the
        ceiling was a guess.

        Order of preference:
          1. the router's plan, if it derived one from this machine
          2. NIGHTMARE_MAX_CTX, if the user set it deliberately
          3. free memory / measured bytes-per-token
          4. the model's OWN trained context, and nothing more

        Step 4 matters: past its trained length a model has no learned
        behaviour for attention, so however much memory is free it
        produces noise rather than context. Returning None below means
        "the caller decides", which is honest.

        kv_dim is taken from the TENSORS, not the header. gemma4
        advertises head_dim 512 in its metadata and stores 256, which
        made an earlier estimate of this number wrong by 2x.
        """
        # 1. the plan
        plan = getattr(self, '_ntmre_plan', None)
        if plan is not None:
            got = getattr(plan, 'max_ctx', None)
            if got:
                print("[STREAM] context %d from router plan (%s)"
                      % (int(got), getattr(plan, 'ctx_basis', '') or 'planned'),
                      flush=True)
                return int(got)

        # 2. an explicit override
        try:
            env = os.environ.get('NIGHTMARE_MAX_CTX', '').strip()
            if env:
                v = int(env)
                if v > 0:
                    print("[STREAM] context %d from NIGHTMARE_MAX_CTX"
                          % v, flush=True)
                    return v
        except Exception:
            pass

        # 3. memory / measured bytes per token
        try:
            n_layers = int(self.n_layers)
            kv_heads = int(self.n_kv_heads)
            head_dim = int(self.head_dim)
            bpe = 2 if getattr(self, '_kv_f16', True) else 4
            if n_layers and kv_heads and head_dim:
                per_token = n_layers * kv_heads * head_dim * bpe * 2
                free = 0
                try:
                    import gpu_info
                    free = int(sum(gpu_info.get_all_free_vram().values()))
                except Exception:
                    free = 0
                if free <= 0:
                    try:
                        free = int(psutil.virtual_memory().available)
                    except Exception:
                        free = 0
                if per_token > 0 and free > 0:
                    by_mem = int((free * 0.85) // per_token)
                    trained = int(getattr(self, 'context_length', 0) or 0)
                    if trained and by_mem > trained:
                        ctx, why = trained, "model's trained limit"
                    else:
                        ctx, why = by_mem, ("%.1f GB free / %d B per token"
                                            % (free / 1024 ** 3, per_token))
                    ctx = max(512, ctx)
                    print("[STREAM] context %d derived from %s"
                          % (ctx, why), flush=True)
                    return ctx
        except Exception:
            pass

        # 4. the model's trained length, uncapped by us
        trained = int(getattr(self, 'context_length', 0) or 0)
        if trained:
            print("[STREAM] context %d (model's trained limit; no memory "
                  "measurement available)" % trained, flush=True)
            return trained

        # Nothing known: leave the caller's value alone rather than
        # replacing it with another guess.
        return None

    def _fit_kv_cache_to_ram(self, kv_dim, itemsize):
        """Shrink max_seq_len so the HOST KV cache stays inside a RAM budget.

        The streaming engine keeps a dense
        (n_layers, max_seq_len, kv_dim) fp32/fp16 array in RAM.  With the
        fixed 12000-token cap that is not a rounding error:

            Mistral 7B   32 x 12000 x 1024 x 4 x 2 = 3.1 GB
            12B llama    40 x 12000 x 1280 x 4 x 2 = 4.9 GB

        which is what exhausted host RAM and pushed the pagefile to ~18 GB,
        stalling the machine.  Conversations do not need 12000 tokens of
        preallocated cache up front, so budget it instead: keep the full
        context when it fits, otherwise use the largest context that does.

        Override with NIGHTMARE_KV_BUDGET_GB (0 disables the cap).

        WHERE THE CACHE ACTUALLY LIVES
        --------------------------------
        This function runs BEFORE the GPU-vs-CPU KV decision is made
        (the placement is at ~L1201, this is called at ~L647/863/980).
        It used to assume "host" unconditionally, so on a machine where
        the KV ends up on VRAM it capped against a RAM budget that did
        not apply. Measured on gemma4-26B-A4B with 20 GB free VRAM:

            Host KV cache would need 5.9 GB at ctx=12000; capping
            context to 4369 to stay inside the 2.0 GB budget

        and then, three lines later:

            KV cache on GPU (F16, 2.15 GB)

        The cache was never on the host. It was capped by a RAM budget,
        paid for in VRAM, and cost thousands of tokens of context for
        nothing. So: if the cache is going to land on VRAM, budget
        against total free VRAM, not against host RAM. The host budget
        still applies when the cache really is host-resident.
        """
        # The host budget used to default to a typed 2.0 GB. That is a
        # guess about someone else's RAM, and it was measured doing
        # damage: on this machine it capped a VRAM-resident cache to
        # 4369 tokens until the placement prediction below was added.
        #
        # Order of preference:
        #   1. NIGHTMARE_KV_BUDGET_GB, if deliberately set
        #   2. the plan the router built for this machine
        #   3. measured free host RAM
        # and never a constant. If none of those can be read, the cap is
        # simply not applied rather than applied with a made-up number.
        budget_bytes = None
        budget_label = "host RAM"
        try:
            env = os.environ.get("NIGHTMARE_KV_BUDGET_GB", "").strip()
        except Exception:
            env = ""
        if env:
            try:
                v = float(env)
                if v <= 0:
                    return                     # explicit "no cap"
                budget_bytes = int(v * (1024 ** 3))
            except ValueError:
                pass

        if budget_bytes is None:
            # From the router's plan: it already measured free memory.
            plan = getattr(self, '_ntmre_plan', None)
            got = getattr(plan, 'kv_budget_bytes', None) if plan else None
            if got:
                budget_bytes = int(got)

        if budget_bytes is None:
            # Measured free host RAM, with a quarter held back so the
            # machine does not page itself to death on the way there.
            try:
                import psutil
                budget_bytes = int(psutil.virtual_memory().available * 0.25)
            except Exception:
                budget_bytes = None

        if budget_bytes is None or budget_bytes <= 0:
            # Nothing measured. Leaving the context alone is honest;
            # capping it by a number invented here is not.
            return

        per_tok = max(1, int(self.n_layers)) * max(1, int(kv_dim)) * int(itemsize) * 2
        need = per_tok * int(self.max_seq_len)

        # Budget against wherever this cache will actually be allocated.
        # This runs before the placement code, so it PREDICTS rather than
        # reads a flag: if the best card can hold the whole window, it
        # goes to VRAM, and the host-RAM budget must not throttle it.
        # The prediction is conservative -- it requires the full window
        # to fit, so the worst case is the old host-RAM behaviour.
        on_gpu = bool(getattr(self, '_use_gpu_kv', False))
        if not on_gpu:
            try:
                probe = max(1, int(self.n_layers)) * max(1, int(kv_dim))\
                    * int(itemsize) * 2 * int(self.max_seq_len)
                on_gpu = self._best_kv_vram() > probe + 100 * (1024 ** 2)
            except Exception:
                on_gpu = False
        if on_gpu:
            budget_bytes = self._gpu_kv_budget_bytes(default=budget_bytes)
            budget_label = "VRAM"

        if need <= budget_bytes:
            return
        allowed = int(budget_bytes // per_tok)
        # Never go below something usable; a tiny cache would just re-cap.
        allowed = max(512, min(int(self.max_seq_len), allowed))
        if allowed < int(self.max_seq_len):
            print(f"[STREAM] KV cache would need {need / 1e9:.1f} GB at "
                  f"ctx={self.max_seq_len}; capping context to {allowed} to stay "
                  f"inside the {budget_bytes / (1024 ** 3):.1f} GB {budget_label} "
                  "budget (NIGHTMARE_KV_BUDGET_GB to change)", flush=True)
            self.max_seq_len = allowed

    def _hidden_dim_for(self, layer):
        """FFN width for one specific layer.

        Scalar for every normal architecture; per-layer for families that
        store feed_forward_length as a vector (gemma4 and friends), where a
        single scalar would either overflow the narrow layers or truncate the
        wide ones.
        """
        dims = getattr(self, '_hidden_dims', None)
        if not dims:
            return int(self.hidden_dim)
        try:
            i = int(layer)
        except Exception:
            return int(self.hidden_dim)
        if 0 <= i < len(dims):
            try:
                return int(dims[i])
            except Exception:
                pass
        return int(self.hidden_dim)

    def _gemma4_layer_info(self, layer):
        """Per-layer attention dims for gemma4: (q_out, k_out, has_v,
        head_dim, n_q_heads, n_kv_heads).  Gemma 4 mixes SWA layers
        (head_dim 256) with full-attention layers (head_dim 512, V=K
        shared, fewer KV heads), so the config-level dims are only a
        fallback - the actual tensor shapes are the ground truth."""
        cache = getattr(self, '_g4_info', None)
        if cache is None:
            cache = self._g4_info = {}
        if layer in cache:
            return cache[layer]
        q_out = self.n_heads * self.head_dim
        k_out = self.n_kv_heads * self.head_dim
        hd = self.head_dim

        def _shape(name):
            n = f"blk.{layer}.{name}"
            ent = getattr(getattr(self, '_nsvd', None), 'entries', None)
            if ent and n in ent:
                return tuple(ent[n][1])
            m = getattr(self, 'm', None)
            if m is not None and n in m.tensors:
                return tuple(m.tensors[n].shape)
            return None

        qn = _shape("attn_q_norm.weight")
        if qn is not None and len(qn) >= 1 and int(qn[0]) in (64, 128, 256, 512):
            hd = int(qn[0])          # QK-norm vector length == head_dim
        qs = _shape("attn_q.weight")
        if qs is not None and len(qs) == 2:
            q_out = int(qs[1])       # all sources label 2D as (in, out)
        ks = _shape("attn_k.weight")
        if ks is not None and len(ks) == 2:
            k_out = int(ks[1])
        has_v = _shape("attn_v.weight") is not None
        n_q = max(1, q_out // hd)
        n_kv = max(1, k_out // hd)
        info = (q_out, k_out, has_v, hd, n_q, n_kv)
        cache[layer] = info
        return info

    def _gemma4_rope_tables(self, hd):
        """cos/sin tables for one gemma4 layer type (keyed by head_dim).
        Local/SWA layers (layer-0 head_dim) use theta_swa (default 1e4)
        and rotate every pair.  Full-attention layers use theta_full
        (default 1e6) with p-RoPE: rope_freqs.weight acts as a per-
        frequency divisor - its 1e30 entries effectively disable rotation
        for the last 75% of pairs (only the first quarter rotate)."""
        cache = getattr(self, '_g4_rope', None)
        if cache is None or not isinstance(cache, dict):
            cache = self._g4_rope = {}
        if hd in cache:
            return cache[hd]
        pos = np.arange(self.max_seq_len, dtype=np.float32)
        half = hd // 2
        hd_swa = self._gemma4_layer_info(0)[3]   # layer 0 is always local
        if hd == hd_swa:
            theta = getattr(self, '_rope_theta_swa', 0.0) or 10000.0
            f = 1.0 / (theta ** (2.0 * np.arange(half, dtype=np.float32) / hd))
        else:
            theta = getattr(self, '_rope_theta_full', 0.0) or self.rope_base or 1000000.0
            f = 1.0 / (theta ** (2.0 * np.arange(half, dtype=np.float32) / hd))
            freqs = getattr(self, '_rope_freqs', None)
            if freqs is not None and len(freqs) >= half:
                f = f / np.asarray(freqs[:half], dtype=np.float32)
        ang = np.outer(pos, f)
        cache[hd] = (np.cos(ang).astype(np.float32),
                     np.sin(ang).astype(np.float32))
        return cache[hd]

    def _g4_moe_gpu_enabled(self):
        """True when gemma4 routed experts can run fully on the GPU.

        Requires the raw single-tensor expert layout (ffn_gate_up_exps /
        ffn_down_exps), the GPU buffers, and a GpuMM.  NIGHTMARE_G4_MOE_GPU=0
        forces the (exact, slow) CPU dequant path for A/B testing."""
        if getattr(self, '_g4_moe_gpu', None) is not None:
            return self._g4_moe_gpu
        ok = False
        if (os.environ.get('NIGHTMARE_G4_MOE_GPU', '1') != '0'
                and getattr(self, '_gpu_buf', None) is not None
                and getattr(self, '_gpumm', None) is not None
                and int(getattr(self, '_n_experts', 0)) > 0):
            try:
                tens = getattr(getattr(self, 'm', None), 'tensors', None) or {}
                ok = bool('blk.0.ffn_gate_up_exps.weight' in tens
                          and 'blk.0.ffn_down_exps.weight' in tens
                          and 'blk.0.ffn_gate_inp.weight' in tens)
            except Exception:
                ok = False
        self._g4_moe_gpu = ok
        if ok:
            print("[STREAM] gemma4 MoE: GPU-resident experts (fused Q4_K/Q5_1)", flush=True)
        elif os.environ.get('NIGHTMARE_G4_MOE_GPU', '1') == '0':
            print("[STREAM] gemma4 MoE: CPU expert path (NIGHTMARE_G4_MOE_GPU=0)", flush=True)
        return ok

    def _g4_expert_gpu(self, layer_idx, ei):
        """Resident GPU descriptor for one expert's (gate_up, down).

        Uploads the expert's raw quantized bytes once (no CPU dequant - the
        fused GPU kernels read the quant blocks directly) and caches the
        device pointers so repeat routing costs no PCIe traffic.  On a cache
        miss only the ~3.7 MB of raw bytes cross the bus (~1 ms), versus the
        ~41 ms full f32 NumPy dequant the CPU path pays per miss.

        Backed by an LRU with its own VRAM byte budget so a long generation
        never runs the card out of memory: the oldest experts are evicted
        (their device memory returned to the GpuMM pool) when the working set
        outgrows the budget.  A token's working set is ~0.9 GB, so with the
        ~18 GB free on this rig the hot set stays resident indefinitely."""
        from collections import OrderedDict
        cache = getattr(self, '_g4_exp_gcache', None)
        if cache is None:
            cache = self._g4_exp_gcache = OrderedDict()
            self._g4_exp_gbytes = 0
            self._g4_exp_gbudget = None
            self._g4_exp_hit = 0
            self._g4_exp_miss = 0
        key = (layer_idx, ei)
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
            self._g4_exp_hit += 1
            return hit
        self._g4_exp_miss += 1
        try:
            import dequant_fast as _dq
            from qwen35_gpu import (G_F32, G_F16, G_Q4_0, G_Q5_0, G_Q5_1,
                                    G_Q8_34, G_Q4_K, G_Q5_K, G_Q6_K, G_BF16)
            _GGML = {0: G_F32, 1: G_F16, 2: G_Q4_0, 6: G_Q5_0, 7: G_Q5_1,
                     8: G_Q8_34, 12: G_Q4_K, 13: G_Q5_K, 14: G_Q6_K, 30: G_BF16}
            tens = self.m.tensors
            info_g = tens[f'blk.{layer_idx}.ffn_gate_up_exps.weight']
            info_d = tens[f'blk.{layer_idx}.ffn_down_exps.weight']
            gt_g = _GGML.get(int(info_g.dtype))
            gt_d = _GGML.get(int(info_d.dtype))
            if gt_g is None or gt_d is None:
                return None
            mm2 = getattr(self._parser, '_mmap', None)
            if mm2 is None:
                return None
            tdo = int(getattr(self.m, 'tensor_data_offset', 0))
            e_ffn = int(self._expert_ffn_dim())
            dim = int(self.dim)

            eb_g, bb_g = _dq.BLOCK_SIZES[int(info_g.dtype)]
            eb_d, bb_d = _dq.BLOCK_SIZES[int(info_d.dtype)]
            nb_g = ((2 * e_ffn * dim) // eb_g) * bb_g   # whole gate_up expert
            nb_d = ((e_ffn * dim) // eb_d) * bb_d
            off_g = tdo + int(info_g.offset) + ei * nb_g
            off_d = tdo + int(info_d.offset) + ei * nb_d
            raw_g = np.ascontiguousarray(np.frombuffer(mm2[off_g:off_g + nb_g], dtype=np.uint8))
            raw_d = np.ascontiguousarray(np.frombuffer(mm2[off_d:off_d + nb_d], dtype=np.uint8))
            if raw_g.size != nb_g or raw_d.size != nb_d:
                return None
            need = int(raw_g.nbytes + raw_d.nbytes)

            # Route to whichever worker has room. GpuMM binds ONE device
            # at construction (qwen35_gpu.py __init__:
            # self.cu = Cuda(device=device)), so self._gpumm can only ever
            # allocate on the card the weights landed on. _gpumm2 -- the
            # second card -- existed and was announced as
            # "Multi-GpuMM Active (2 GPUs)" but was never handed anything
            # to hold. On this machine that meant an 8 GB RTX 3060 Ti sat
            # at 0.39 GB used while the MoE ran 2.93 GiB of expert cache
            # against a 6.11 GB working set.
            gpu = self._pick_expert_gpu(need)
            if gpu is None:
                return None
            wi = getattr(gpu, '_worker_index', 0)
            # Evict LRU experts until the worker can actually accept this
            # one. upload_raw() marks a tensor 'stream' when its own
            # resident budget is exhausted; returning evicted bytes to that
            # budget is what keeps long generations on the GPU instead of
            # silently falling back to the CPU expert path.
            while int(getattr(gpu, '_budget', 0)) < need and len(cache) > 0:
                _ok, _od = cache.popitem(last=False)
                self._evict_g4_expert(_ok, _od)

            # The worker index is part of the name: the same expert may be
            # resident on one card in one generation and on the other in
            # the next, and a name without it would collide in the LRU and
            # in the per-worker _raw table.
            gn = f'g4e.{wi}.{layer_idx}.{ei}.gu'
            dn = f'g4e.{wi}.{layer_idx}.{ei}.d'
            if not gpu.has(gn):
                gpu.upload_raw(gn, raw_g.tobytes(), gt_g, 2 * e_ffn, dim)
                gpu.upload_raw(dn, raw_d.tobytes(), gt_d, dim, e_ffn)
            # 'stream' means even after eviction there is no room (the card is
            # full of resident layer weights) - fall back for this expert.
            if gpu._raw.get(gn, [None])[0] == 'stream' or\
               gpu._raw.get(dn, [None])[0] == 'stream':
                gpu._raw.pop(gn, None)
                gpu._raw.pop(dn, None)
                return None
            # The worker is carried in the descriptor, not looked up again.
            # An expert can be resident on either card, and freeing it
            # through self._gpumm when GPU1 actually holds it would pop a
            # key that is not in GPU1's table and leak the real allocation.
            desc = (gn, dn, 2 * e_ffn, e_ffn, dim, need, wi)
            cache[key] = desc
            self._g4_exp_gbytes += need
            return desc
        except Exception:
            return None

    def _evict_g4_expert(self, key, desc):
        """Free one cached GPU expert's device memory and return its bytes to
        the GpuMM resident-weight budget so the slot can be reused."""
        nbytes = desc[5] if len(desc) > 5 else 0
        try:
            # Free on the card that actually holds it (index at desc[6]),
            # not always the primary worker.
            wi = int(desc[6]) if len(desc) > 6 else 0
            gpu = self._gpumm
            for w in self._expert_workers():
                if getattr(w, '_worker_index', 0) == wi:
                    gpu = w
                    break
            gn, dn = desc[0], desc[1]
            for nm in (gn, dn):
                e = gpu._raw.pop(nm, None)
                if e is not None and not isinstance(e[0], str):
                    try:
                        gpu.cu.free(e[0])
                    except Exception:
                        pass
            gpu._budget += int(nbytes)
        except Exception:
            pass
        self._g4_exp_gbytes = max(0, getattr(self, '_g4_exp_gbytes', 0) - int(nbytes))

    def _g4_down_scale(self, layer_idx):
        """Per-expert ffn_down_exps.scale (n_experts,) f32, or None.

        gemma4 stores a per-expert correction on the down projection; the CPU
        path applies it in _load_expert_fast, so the raw-GPU path must too."""
        cache = getattr(self, '_g4_dscale', None)
        if cache is None:
            cache = self._g4_dscale = {}
        if layer_idx in cache:
            return cache[layer_idx]
        sc = None
        try:
            name = f'blk.{layer_idx}.ffn_down_exps.scale'
            if getattr(self, '_parser', None) is not None and name in self.m.tensors:
                a = self._parser.load_tensor(self.m, name)
                a = np.asarray(a, dtype=np.float32).ravel()
                if a.size == int(self._n_experts):
                    sc = a
        except Exception:
            sc = None
        cache[layer_idx] = sc
        return sc

    def _moe_ffn_g4_gpu(self, attn_out, moe_in, weights, layer, eps):
        """gemma4 routed-expert MoE, fully GPU-resident.

        Router and the pre-expert norm stay on the CPU (128 logits + one
        2816-vector norm are negligible); the expensive per-expert work - 3
        matvecs x top-8 - runs as fused dequant+matvec kernels over raw
        quantized weights on VRAM.  Returns cur_moe as a CPU f32 (dim,)
        array, or None so the caller can finish the layer on the CPU path."""
        if not self._g4_moe_gpu_enabled():
            return None
        gpu = self._gpumm
        gpu._bind()   # generation runs on a worker thread; bind this thread
        buf = self._gpu_buf
        dim = int(self.dim)

        # Router (CPU, exact - matches the reference CPU path bit-for-bit).
        rtmp = np.asarray(attn_out, dtype=np.float32)
        rtmp = rtmp / np.sqrt(np.mean(rtmp * rtmp) + eps)
        rtmp = rtmp * np.float32(1.0 / np.sqrt(float(dim)))
        gs = weights.get('moe_gate_inp_scale')
        if gs is not None:
            rtmp = rtmp * np.asarray(gs, dtype=np.float32)
        router = np.asarray(weights['moe_router'], dtype=np.float32)
        logits = router @ rtmp
        mx = logits.max()
        probs = np.exp(logits - mx)
        probs /= probs.sum()
        k = min(int(self._n_experts_used), int(self._n_experts))
        # Top-k goes through expert module.select() so an expert band
        # can restrict WHICH experts the router picks from.
        #
        # This used to be the bare `np.argsort(probs)[-k:]` below. The
        # previous band implementation wrapped this whole function
        # instead, looking for `probs` among the call arguments -- but
        # probs is a LOCAL here, so it never found one, silently called
        # through unmasked, and printed "routing restricted" while
        # changing nothing. verify_moe_band.py measured 107,520 escapes.
        #
        # select() with no band installed is byte-identical to the line
        # it replaced, so an unbanded run costs one function call.
        try:
            raise ImportError("sealed: expert band not shipped")
            top, tw = _band.select(probs, k)
        except Exception:
            top = np.argsort(probs)[-k:]
            tw = probs[top]
            tw = tw / tw.sum()

        # Upload the normalized expert input to every card that can hold an
        # expert. Which card an expert lands on is not known until the
        # layer routes it, so each card gets its own copy rather than
        # sharing one pointer across devices.
        mi = np.ascontiguousarray(np.asarray(moe_in, dtype=np.float32))
        for _wi, _wb in (self._gpu_buf_w or {}).items():
            if not _wb:
                continue
            _w = self._worker_by_index(_wi)
            if _w is None:
                continue
            try:
                _w._bind()
                _w.cu.cu.cuMemcpyHtoD_v2(
                    _wb['moe_in'], ctypes.c_void_p(mi.ctypes.data), dim * 4)
            except Exception:
                continue
        try:
            gpu._bind()
        except Exception:
            pass
        dscale = self._g4_down_scale(layer)
        cur_moe = None

        for ei, w in zip(top, tw):
            desc = self._g4_expert_gpu(layer, int(ei))
            if desc is None:
                return None
            gn, dn, gu_od, e_ffn, in_dim = desc[:5]
            # The expert lives on exactly one card. Compute it entirely on
            # that card, using THAT card's scratch, then bring the result
            # back. No pointer crosses a device boundary: moe_in was
            # uploaded to every card, and the weighted sum is accumulated
            # on the host, which is where the caller expects a CPU array.
            wi = int(desc[6]) if len(desc) > 6 else 0
            egpu = self._worker_by_index(wi)
            if egpu is None:
                return None
            ebuf = self._gpu_buf_w.get(wi) or buf
            try:
                egpu._bind()
            except Exception:
                pass
            _w = float(w)
            if dscale is not None:
                _w *= float(dscale[int(ei)])

            # Fused path: the whole expert in ONE launch, with the
            # activation staged in shared memory so it never round-trips
            # through global memory. Replaces the four launches below
            # (matvec -> geglu -> matvec -> add_scaled), which is where
            # most of the wall-clock went.
            part = None
            try:
                part = self._g4_fused_expert(egpu, wi, ebuf, gn, dn,
                                             e_ffn, dim, _w)
            except Exception:
                part = None
            if part is None:
                # Fall back to the split path. Correctness first: a wrong
                # number here propagates into every layer above it.
                try:
                    egpu.cu.cu.cuMemsetD32_v2(ebuf['moe'], 0, dim)
                except Exception:
                    return None
                try:
                    egpu.matvec_gpu(ebuf['moe_in'], gn, ebuf['gate'])
                    up_ptr = ctypes.c_ulonglong(ebuf['gate'].value + e_ffn * 4)
                    egpu.geglu_mul_gpu(ebuf['gate'], up_ptr, e_ffn, ebuf['up'])
                    egpu.matvec_gpu(ebuf['up'], dn, ebuf['tmp'])
                    egpu.add_scaled_gpu(ebuf['moe'], ebuf['tmp'], _w, dim)
                    part = egpu.from_gpu_ptr(ebuf['moe'], dim)
                    part = np.asarray(part, dtype=np.float32).reshape(-1)
                except Exception:
                    return None
            if cur_moe is None:
                cur_moe = np.zeros(dim, dtype=np.float32)
            cur_moe += part

        if cur_moe is None:
            cur_moe = np.zeros(dim, dtype=np.float32)
        return cur_moe

    def _g4_fused_expert(self, gpu, wi, ebuf, gn, dn, e_ffn, dim, weight):
        """
        Run one expert through the fused kernel, or return None to fall
        back to the split path.

        Returning None is always safe. The split path is slower but it
        is what has been running, so any doubt about the fused kernel
        should resolve to the old behaviour rather than to a wrong
        number -- a model that loads and talks noise is the worst
        outcome available.

        The kernel is loaded lazily and cached on the worker. It lives
        in g4_moe_kernels.py and is verified by verify_moe_kernel.py,
        which checks it against a numpy reference to 6.75e-07.
        """
        # The fused kernel is CORRECT (verify_moe_kernel.py: 6.75e-07 vs
        # numpy) but MEASURED 3x SLOWER than the split path it replaces:
        #   split  3.17 tok/s    fused  1.05 tok/s
        # Its phase 2 reads W_down column-wise, so adjacent threads walk
        # addresses ~100KB apart and nothing coalesces, while the split
        # kernel maps one thread per output row and does. Correctness
        # was proven; coalescing was not checked, and it is the whole
        # cost. Default OFF until the access pattern is fixed.
        # Set NIGHTMARE_FUSED_MOE=1 to force it on.
        if os.environ.get('NIGHTMARE_FUSED_MOE', '0') != '1':
            return None
        fns = getattr(gpu, '_g4_fused_fns', None)
        if fns is None:
            try:
                import g4_moe_kernels as G4K
                mod = gpu.cu.compile(G4K.kernel_source())
                fns = {'fused': gpu.cu.func(mod, 'g4_moe_expert_fused'),
                       'topk': gpu.cu.func(mod, 'g4_topk_partial')}
                gpu._g4_fused_fns = fns
                gpu._g4_fused_ready = True
                print("[STREAM] fused MoE expert kernel active", flush=True)
            except Exception as e:
                gpu._g4_fused_fns = None
                gpu._g4_fused_ready = False
                print("[STREAM] fused MoE kernel unavailable (%s); using the "
                      "split path" % e, flush=True)
                return None
        if not fns:
            return None

        # The expert bytes live in the worker's _raw table as
        # (ptr, gtype, out_dim, in_dim) -- the POINTER is element 0.
        # Taking element 1 would pass the gtype integer as an address,
        # which is a wild pointer rather than an error, so it is checked
        # explicitly rather than trusted.
        try:
            gp = gpu._raw[gn][0]
            dp = gpu._raw[dn][0]
        except Exception:
            return None
        if not isinstance(gp, ctypes.c_ulonglong) or\
           not isinstance(dp, ctypes.c_ulonglong):
            return None
        if gp.value == 0 or dp.value == 0:
            return None

        acc = ebuf.get('moe_out')
        if acc is None:
            acc = ebuf['moe_out'] = gpu.cu.alloc(int(dim) * 4)
        # grid.x tiles the `dim` outputs of the down projection;
        # block 256 covers the 2*e_ffn gate/up rows.
        gpu.cu.launch(fns['fused'], ((dim + 255) // 256, 1), (256, 1),
                      [ctypes.c_ulonglong(gp.value),
                       ctypes.c_ulonglong(dp.value),
                       ctypes.c_ulonglong(ebuf['moe_in'].value),
                       ctypes.c_ulonglong(acc.value),
                       float(weight), ctypes.c_int(e_ffn),
                       ctypes.c_int(dim), ctypes.c_int(0), ctypes.c_int(0),
                       ctypes.c_float(1.0)],
                      shared_mem=int(2 * e_ffn) * 4)
        return np.asarray(gpu.from_gpu_ptr(acc, dim),
                          dtype=np.float32).reshape(-1)

    def _layer_forward_gemma4(self, x, layer, position, weights):
        """Gemma 4 layer (matches the official HF/llama.cpp semantics):
        RMSNorm -> QKV -> per-head QK-norm -> per-head weight-less V-norm
        -> split-half RoPE (theta 1e4 local / 1e6 + p-RoPE mask global)
        -> attention with scale 1.0 -> post-attn norm -> residual ->
        FFN (GELU-tanh) -> post-FFN norm -> residual -> PLE gated residual
        (E-models) -> whole-layer output multiplied by layer_output_scale.
        Global k_eq_v layers have no attn_v and share V=K (V taken
        pre-norm/pre-rope, then V-normed).  Layers in the shared-KV range
        (last num_kv_shared_layers) compute no K/V at all - they attend
        over the last same-type layer's cached K/V."""
        eps = getattr(self, '_norm_eps', 1e-6)
        x = x.reshape(-1)
        # per-layer dims: SWA layers run head_dim 256, full-attention
        # layers run head_dim 512 (dims from the actual tensors)
        _qo, _ko, _has_v, hd, nh, nkv = self._gemma4_layer_info(layer)
        kvd = nkv * hd
        share_src = self._g4_share_map().get(layer)

        xn = self._rms_norm(x, weights['attn_norm'], eps)
        q = self._mm(xn.reshape(1, -1), weights['wq'])[0].reshape(nh, hd)

        if share_src is None:
            k = self._mm(xn.reshape(1, -1), weights['wk'])[0].reshape(nkv, hd)
            if weights['wv'] is not None and not (isinstance(weights['wv'], np.ndarray)
                                                  and not weights['wv'].any()):
                v = self._mm(xn.reshape(1, -1), weights['wv'])[0].reshape(nkv, hd)
            else:
                v = k.copy()   # k_eq_v layer (V=K, pre-norm/pre-rope)

            # per-head V RMSNorm WITHOUT weight (vectorized - no Python loop)
            ms_v = np.mean(v * v, axis=1, keepdims=True) + eps
            v = v / np.sqrt(ms_v)

            # per-head K RMSNorm (vectorized)
            wkn = weights.get('attn_k_norm')
            if wkn is not None:
                ms_k = np.mean(k * k, axis=1, keepdims=True) + eps
                k = k / np.sqrt(ms_k) * wkn

        # per-head Q RMSNorm (vectorized - no Python loop)
        wqn = weights.get('attn_q_norm')
        if wqn is not None:
            ms_q = np.mean(q * q, axis=1, keepdims=True) + eps
            q = q / np.sqrt(ms_q) * wqn

        # RoPE (split-half NEOX), per layer type: local layers rotate all
        # pairs with theta_swa; global layers theta_full + p-RoPE mask
        cos_t, sin_t = self._gemma4_rope_tables(hd)
        half = hd // 2
        c = cos_t[position]
        s = sin_t[position]
        q0 = q[:, :half].copy(); q1 = q[:, half:].copy()
        q[:, :half] = q0 * c - q1 * s
        q[:, half:] = q0 * s + q1 * c

        if share_src is None:
            k0 = k[:, :half].copy(); k1 = k[:, half:].copy()
            k[:, :half] = k0 * c - k1 * s
            k[:, half:] = k0 * s + k1 * c
            self.kv_k[layer, position, :kvd] = k.reshape(-1)
            self.kv_v[layer, position, :kvd] = v.reshape(-1)
            src = layer
        else:
            # shared-KV layer: reuse the source layer's K/V (post-norm,
            # post-rope), exactly as cached - no cache write of our own
            src = share_src

        # attention (vectorized - no Python loop, grouped einsum)
        # gemma4 uses attention scaling 1.0 (QK norms carry the magnitude)
        n_rep = max(1, nh // nkv)
        K = self.kv_k[src, :position + 1, :kvd].reshape(position + 1, nkv, hd)
        V = self.kv_v[src, :position + 1, :kvd].reshape(position + 1, nkv, hd)
        # group query heads under their shared KV head: (nkv, n_rep, hd)
        # -> einsum against (seq, nkv, hd) avoids materializing K/V repeats
        q3 = q.reshape(nkv, n_rep, hd)
        scores = np.einsum('grd,pgd->grp', q3, K, optimize=True)     # (nkv, n_rep, seq)
        scores_shape = scores.shape
        scores = scores.reshape(nh, position + 1)
        scores -= np.max(scores, axis=1, keepdims=True)
        np.exp(scores, out=scores)
        scores /= np.sum(scores, axis=1, keepdims=True)
        scores = scores.reshape(scores_shape)
        attn = np.einsum('grp,pgd->grd', scores, V, optimize=True)   # (nkv, n_rep, hd)
        attn = attn.reshape(nh, hd)

        a = self._mm(attn.reshape(1, -1), weights['wo'])[0]
        pan = weights.get('post_attention_norm')
        if pan is not None:
            a = self._rms_norm(a, pan, eps)
        x = x + a

        # ---- FFN. gemma4 MoE layers are a SUM of a shared dense expert and the
        # routed experts, each with its own norm pair. Verified against
        # llama.cpp src/models/gemma4.cpp (build_moe_ffn / build_norm calls):
        #   attn_out  = x + attn
        #   cur_mlp   = rms(ffn_norm, attn_out) -> GeGLU dense -> rms(post_ffw_norm_1)
        #   cur_moe   = rms(pre_ffw_norm_2, attn_out) -> routed experts -> rms(post_ffw_norm_2)
        #   router    = ffn_gate_inp @ (rms(attn_out) * 1/sqrt(dim) * ffn_gate_inp_s)
        #   f         = rms(post_ffw_norm, cur_mlp + cur_moe)
        #   x         = attn_out + f
        _is_moe = (self._n_experts > 0 and weights.get('moe_router') is not None)
        attn_out = x

        def _geglu(v):
            return 0.5 * v * (1.0 + np.tanh(np.sqrt(2.0 / np.pi)
                                           * (v + 0.044715 * v ** 3)))

        if _is_moe:
            # shared (dense) expert
            mlp_in = self._rms_norm(attn_out, weights['ffn_norm'], eps)
            g_ = self._mm(mlp_in.reshape(1, -1), weights['wg'])[0]
            u_ = self._mm(mlp_in.reshape(1, -1), weights['wu'])[0]
            cur_mlp = self._mm((_geglu(g_) * u_).reshape(1, -1),
                               weights['wd'])[0]
            if weights.get('post_ffw_norm_1') is not None:
                cur_mlp = self._rms_norm(cur_mlp, weights['post_ffw_norm_1'], eps)

            # routed experts. The router reads attn_out (NOT the normalised
            # expert input) and applies ffn_gate_inp_s before the matmul.
            moe_in = self._rms_norm(attn_out, weights.get('pre_ffw_norm_2'), eps)
            rtmp = attn_out.astype(np.float32)
            rtmp = rtmp / np.sqrt(np.mean(rtmp * rtmp) + eps)
            rtmp = rtmp * np.float32(1.0 / np.sqrt(float(self.dim)))
            gs = weights.get('moe_gate_inp_scale')
            if gs is not None:
                rtmp = rtmp * np.asarray(gs, dtype=np.float32)
            _router = np.asarray(weights['moe_router'], dtype=np.float32)
            _logits = _router @ rtmp
            _mx = _logits.max()
            _probs = np.exp(_logits - _mx)
            _probs /= _probs.sum()
            _k = min(int(self._n_experts_used), int(self._n_experts))
            _top = np.argsort(_probs)[-_k:]
            _tw = _probs[_top]
            _tw = _tw / _tw.sum()

            cur_moe = np.zeros(self.dim, dtype=np.float32)
            _moe_gpu = None
            if self._g4_moe_gpu_enabled():
                _moe_gpu = self._moe_ffn_g4_gpu(attn_out, moe_in, weights, layer, eps)
            if _moe_gpu is not None:
                cur_moe = _moe_gpu
            else:
                for _ei, _w in zip(_top, _tw):
                    _e = self._load_expert(layer, int(_ei))
                    if _e is None or _e[0] is None:
                        continue
                    _gx, _ux, _dx = _e
                    _mf = moe_in.astype(np.float32)
                    _a = self._expert_act(_gx @ _mf, _ux @ _mf)
                    cur_moe += float(_w) * (_dx @ _a)
            if weights.get('post_ffw_norm_2') is not None:
                cur_moe = self._rms_norm(cur_moe, weights['post_ffw_norm_2'], eps)
            f = cur_mlp + cur_moe
        else:
            xn = self._rms_norm(attn_out, weights['ffn_norm'], eps)
            gate = self._mm(xn.reshape(1, -1), weights['wg'])[0]
            up = self._mm(xn.reshape(1, -1), weights['wu'])[0]
            f = self._mm((_geglu(gate) * up).reshape(1, -1), weights['wd'])[0]

        pfn = weights.get('post_ffw_norm')
        if pfn is not None:
            f = self._rms_norm(f, pfn, eps)
        x = attn_out + f

        # gemma4 Per-Layer Embedding (E-models): gated residual signal,
        # applied after the FFN residual and before the layer scale
        if self._ple_current is not None and weights.get('ple_inp_gate') is not None:
            g = self._mm(x.reshape(1, -1), weights['ple_inp_gate'])[0]
            g = 0.5 * g * (1.0 + np.tanh(np.sqrt(2.0 / np.pi)
                                         * (g + 0.044715 * g ** 3)))
            g = g * self._ple_current[layer]
            s = self._mm(g.reshape(1, -1), weights['ple_proj'])[0]
            s = self._rms_norm(s, weights['ple_post_norm'], eps)
            x = x + s

        # gemma4 layer_output_scale multiplies the WHOLE layer output
        ls = weights.get('layer_output_scale')
        if ls is not None:
            x = x * float(np.ravel(ls)[0])
        return x
    
    # ════════════════════════════════════════════════════════════
    # DUAL-GPU PIPELINE
    # ════════════════════════════════════════════════════════════
    
    def forward_dual_gpu(self, token_id: int, position: int) -> np.ndarray:
        """
        Dual-GPU pipelined forward pass.
        GPU 0 computes layer N while GPU 1 prefetches layer N+1.
        ~2x throughput vs single GPU streaming.
        """
        if self._gpu_count < 2:
            return self.forward(token_id, position)
        
        # Same logic as forward() but with overlapped load/compute
        x = self._get_embedding_row(token_id)
        
        for layer in range(self.n_layers):
            # Start prefetch of next layer on alternate GPU
            if layer + 1 < self.n_layers:
                self._start_prefetch(layer + 1)
            
            # Load current layer (may already be prefetched)
            layer_weights = self._get_or_load_layer(layer)
            
            # Compute on current GPU
            x = self._layer_forward(x, layer, position, layer_weights)
            
            # Evict
            del layer_weights
            
            # Swap active GPU
            self._active_gpu = 1 - self._active_gpu
        
        x = self._norm(x, self._output_norm)
        logits = self._logits(x)
        return logits
    
    def _start_prefetch(self, layer_idx: int):
        """Start loading a layer in background thread."""
        self._prefetch_ready.clear()
        self._prefetch_layer = layer_idx
        
        def _do_prefetch():
            # Load layer data (this is the PCIe transfer)
            self._prefetch_data = self._load_layer(layer_idx)
            self._prefetch_ready.set()
        
        self._prefetch_thread = threading.Thread(target=_do_prefetch, daemon=True)
        self._prefetch_thread.start()
    
    def _get_or_load_layer(self, layer_idx: int) -> Dict:
        """Get layer from prefetch buffer or load directly."""
        if self._prefetch_layer == layer_idx and self._prefetch_ready.is_set():
            data = self._prefetch_data
            self._prefetch_data = None
            return data
        else:
            return self._load_layer(layer_idx)
    
    # ════════════════════════════════════════════════════════════
    # CUDA GRAPH SUPPORT
    # ════════════════════════════════════════════════════════════

    def _setup_graph(self, token_id, position):
        """Capture the forward pass into a CUDA graph for replay.
        Called after warmup (all weights resident, caches hot)."""
        gpu = self._gpumm
        cu = gpu.cu
        if cu is None or cu.stream is None:
            return False
        if self._gpu_kv is None or self._gpu_kv[0] != 'gpu':
            return False  # graph only works with GPU KV cache
        # F16 KV cache required: the F32 path uses cuMemcpyDtoD_v2 on the
        # NULL stream, which can't be captured into the graph.
        kv_bpe = self._gpu_kv[4] if len(self._gpu_kv) > 4 else 4
        if kv_bpe != 2:
            print('[GRAPH] Skipping - F32 KV cache not capturable', flush=True)
            return False
        # Guard: every matvec weight must be GPU-resident (not 'stream').
        # Streamed weights alloc+upload per call, which can't be captured.
        _wt_keys = ('wq', 'wk', 'wv', 'wo', 'wg', 'wu', 'wd')
        for layer in range(self.n_layers):
            lw = self._get_or_load_layer(layer)
            for wk in _wt_keys:
                w = lw.get(wk)
                if isinstance(w, tuple) and w and w[0].startswith('g'):
                    name = w[1]
                    if gpu._raw.get(name, [None])[0] == 'stream':
                        print('[GRAPH] Skipping - weight %s is streamed' % name, flush=True)
                        return False
        kv_stride = self._gpu_kv[3]
        kv_bpe = self._gpu_kv[4] if len(self._gpu_kv) > 4 else 4
        # Persistent position-dependent args (updated in place before replay)
        self._g_pos = ctypes.c_int(position)
        self._g_seq = ctypes.c_int(position + 1)
        self._g_k_dst = [ctypes.c_ulonglong(0) for _ in range(self.n_layers)]
        self._g_v_dst = [ctypes.c_ulonglong(0) for _ in range(self.n_layers)]
        self._update_graph_args(position)
        # Pre-allocate a fixed output buffer for the logits matvec
        self._graph_logits_ptr = cu.alloc(int(self.vocab_size) * 4)
        # Ensure all warmup kernels on the stream are done before we
        # touch buf['x'] or begin capture.
        cu.sync()
        # Upload embedding BEFORE capture (it's done outside the graph
        # in the replay path, so don't capture it).
        x = self._get_embedding_row(token_id)
        if str(getattr(self, '_architecture', '')).startswith('gemma'):
            x = x * np.float32(np.sqrt(self.dim))
        cu.cu.cuMemcpyHtoD_v2(self._gpu_buf['x'],
                           ctypes.c_void_p(x.ctypes.data), self.dim * 4)
        cu.sync()
        # Run one forward pass in capture mode (relaxed - allows
        # non-captured stream ops like weight uploads without error).
        try:
            cu._capturing = True
            cu._graph_keep = []
            r = cu.cu.cuStreamBeginCapture(cu.stream, 2)  # RELAXED mode
            if r != 0:
                cu._capturing = False
                print('[GRAPH] BeginCapture failed: %d' % r, flush=True)
                return False
            # Layer loop
            for layer in range(self.n_layers):
                lw = self._get_or_load_layer(layer)
                self._layer_forward_gpu(self._gpu_buf['x'], layer, position, lw)
                del lw
            # Final norm
            if self._use_layernorm:
                gpu.layernorm_gpu(self._gpu_buf['x'], self._gpu_output_norm,
                                  self.dim, self._norm_eps, self._gpu_buf['xn'])
            else:
                gpu.rmsnorm_gpu(self._gpu_buf['x'], self._gpu_output_norm,
                                self.dim, self._norm_eps, self._gpu_buf['xn'])
            # Output projection into the fixed buffer
            d = self._output_gpu
            if d is not None and d[0] == 'gpu':
                _, key, _od, _id = d
                is_streamed = gpu._raw.get(key, [None])[0] == 'stream'
                raw = self._raw_store.get(key) if is_streamed else None
                if not is_streamed:
                    gpu.matvec_gpu(self._gpu_buf['xn'], key,
                                  out_ptr=self._graph_logits_ptr, raw=raw)
                else:
                    # Streamed output - can't graph-capture; abort
                    cu._capturing = False
                    cu.cu.cuStreamEndCapture(cu.stream, ctypes.byref(ctypes.c_void_p()))
                    return False
            else:
                cu._capturing = False
                cu.cu.cuStreamEndCapture(cu.stream, ctypes.byref(ctypes.c_void_p()))
                return False
            ok = cu.end_capture()
            if not ok:
                print('[GRAPH] Capture failed after %d launches' %
                      len(cu._graph_keep), flush=True)
                return False
            # Verify: replay must reproduce what an eager pass produces.
            # "it replayed without error" is NOT verification -- a graph
            # with stale baked-in position/seq/KV-offset args replays
            # perfectly happily and returns nonsense. Compare numbers.
            if not self._verify_graph_replay(token_id, position):
                cu.destroy_graph()
                print('[GRAPH] replay disagreed with the eager pass - graph '
                      'DISCARDED, staying on eager decode', flush=True)
                return False
            print('[GRAPH] Replay verified against eager pass - graph mode active',
                  flush=True)
            return True
        except Exception as _e:
            cu._capturing = False
            print('[GRAPH] Exception during capture: %s' % _e, flush=True)
            try:
                cu.cu.cuStreamEndCapture(cu.stream, ctypes.byref(ctypes.c_void_p()))
            except Exception:
                pass
            return False

    def _eager_logits_for_verify(self, token_id, position):
        """Eager (non-graph) logits for one token, for graph verification."""
        buf = self._gpu_buf
        if buf is None:
            return None
        cu = self._gpumm.cu
        x = self._get_embedding_row(token_id)
        if str(getattr(self, '_architecture', '')).startswith('gemma'):
            x = x * np.float32(np.sqrt(self.dim))
        try:
            cu.cu.cuMemcpyHtoD_v2(buf['x'], ctypes.c_void_p(x.ctypes.data),
                                  self.dim * 4)
        except Exception:
            return None
        cu.sync()
        for layer in range(self.n_layers):
            self._layer_forward_gpu(buf['x'], layer, position,
                                    self._get_or_load_layer(layer))
        gpu = self._gpumm
        if self._use_layernorm:
            gpu.layernorm_gpu(buf['x'], self._gpu_output_norm, self.dim,
                              self._norm_eps, buf['xn'])
        else:
            gpu.rmsnorm_gpu(buf['x'], self._gpu_output_norm, self.dim,
                            self._norm_eps, buf['xn'])
        xn = gpu.from_gpu_ptr(buf['xn'], self.dim)
        d = self._output_gpu
        if d is not None and d[0] == 'gpu':
            return self._gpu_mm_desc(xn, d)
        if self._output_weight is not None:
            return self._output_weight @ xn
        return None

    def _verify_graph_replay(self, token_id, position, tol=2e-2):
        """True only if replaying the captured graph reproduces the eager pass.

        Guards against the failure mode where a CUDA graph is captured with
        arguments that go stale: it replays without error and returns
        confident, wrong logits.
        """
        cu = self._gpumm.cu
        try:
            eager = self._eager_logits_for_verify(token_id, position)
            if eager is None:
                return False
            eager = np.asarray(eager, dtype=np.float64).ravel()
            eager = eager[np.isfinite(eager)]
            if eager.size == 0:
                return False

            self._update_graph_args(position)
            buf = self._gpu_buf
            x = self._get_embedding_row(token_id)
            if str(getattr(self, '_architecture', '')).startswith('gemma'):
                x = x * np.float32(np.sqrt(self.dim))
            cu.cu.cuMemcpyHtoD_v2(buf['x'], ctypes.c_void_p(x.ctypes.data),
                                  self.dim * 4)
            cu.sync()
            if not cu.replay_graph():
                return False
            cu.sync()
            got = np.empty(int(self.vocab_size), dtype=np.float32)
            r = cu.cu.cuMemcpyDtoH_v2(got.ctypes.data, self._graph_logits_ptr,
                                      got.nbytes)
            if r != 0:
                return False
            got = got[np.isfinite(got)]
            if got.size != eager.size:
                return False
            scale = max(1e-6, float(np.abs(eager).max()))
            return bool(float(np.abs(got - eager).max()) <= tol * scale)
        except Exception:
            return False

    def _update_graph_args(self, position):
        """Update persistent graph args for the current position."""
        kv_stride = self._gpu_kv[3]
        kv_bpe = self._gpu_kv[4] if len(self._gpu_kv) > 4 else 4
        kv_off = position * kv_stride * kv_bpe
        self._g_pos.value = position
        self._g_seq.value = position + 1
        for layer in range(self.n_layers):
            self._g_k_dst[layer].value = self._gpu_kv[1][layer].value + kv_off
            self._g_v_dst[layer].value = self._gpu_kv[2][layer].value + kv_off

    # ════════════════════════════════════════════════════════════
    # STATS & CONTROL
    # ════════════════════════════════════════════════════════════
    
    def reset(self):
        """Reset KV cache for new generation."""
        # Robustness: ensure ctypes is available
        if 'ctypes' not in globals():
            import ctypes as _ct
            globals()['ctypes'] = _ct
        # Destroy any captured CUDA graph - it's invalid after reset
        if self._gpumm is not None and self._gpumm.cu is not None:
            self._gpumm.cu.destroy_graph()
        # Allow graph re-capture on the next warmup pass
        self._graph_tried = False
        # Free the fixed logits buffer from the previous graph
        if getattr(self, '_graph_logits_ptr', None) is not None and self._gpumm is not None:
            try:
                self._gpumm.cu.free(self._graph_logits_ptr)
            except Exception:
                pass
        self._graph_logits_ptr = None
        self._g_pos = None
        self._g_seq = None
        self._g_k_dst = None
        self._g_v_dst = None
        if self.kv_k is not None:
            self.kv_k[:] = 0
            self.kv_v[:] = 0
        # Clear GPU KV cache if active (F16 or F32)
        if self._gpu_kv is not None and self._gpu_kv[0] == 'gpu':
            kv_bpe = self._gpu_kv[4] if len(self._gpu_kv) > 4 else 4
            nbytes = self.max_seq_len * self._gpu_kv[3] * kv_bpe
            words = (nbytes + 3) // 4   # cuMemsetD32_v2 writes 4-byte words
            for ptr in self._gpu_kv[1] + self._gpu_kv[2]:
                try:
                    self._gpumm.cu.cu.cuMemsetD32_v2(ptr, 0, words)
                except Exception:
                    pass
        self.pos = 0
    
    def get_stats(self) -> Dict:
        """Get streaming performance stats."""
        total = self._times['total']
        if total == 0:
            return {'tok/s': 0, 'load_pct': 0, 'compute_pct': 0}
        
        return {
            'tokens': self._tokens_generated,
            'tok/s': self._tokens_generated / total if total > 0 else 0,
            'total_time': total,
            'load_time': self._times['load'],
            'compute_time': self._times['compute'],
            'load_pct': self._times['load'] / total * 100,
            'compute_pct': self._times['compute'] / total * 100,
            'per_layer_ms': (total / max(1, self._tokens_generated) / self.n_layers) * 1000,
        }
    
    def close(self):
        """Release ALL resources: GPU VRAM (both GpuMMs), GPU KV cache /
        buffers, file handles, and every large RAM cache.  Called when the
        user switches models so the previous engine's footprint is fully
        reclaimed before the next model loads."""
        # Free CUDA graph resources before freeing the GpuMMs
        if getattr(self, '_gpumm', None) is not None and self._gpumm.cu is not None:
            try:
                self._gpumm.cu.destroy_graph()
            except Exception:
                pass
        if getattr(self, '_graph_logits_ptr', None) is not None and getattr(self, '_gpumm', None) is not None and self._gpumm.cu is not None:
            try:
                self._gpumm.cu.free(self._graph_logits_ptr)
            except Exception:
                pass
            self._graph_logits_ptr = None

        # Free GPU KV cache & forward-pass buffers BEFORE closing GpuMM
        if getattr(self, '_gpumm', None) is not None and self._gpumm.cu is not None:
            cu1 = self._gpumm.cu
            if getattr(self, '_gpu_kv', None) is not None and self._gpu_kv[0] == 'gpu':
                for ptrs in (self._gpu_kv[1], self._gpu_kv[2]):
                    for ptr in (ptrs or []):
                        if ptr is not None:
                            try:
                                cu1.free(ptr)
                            except Exception:
                                pass
                self._gpu_kv = None

            if getattr(self, '_gpu_buf', None) is not None:
                for ptr in self._gpu_buf.values():
                    if ptr is not None:
                        try:
                            cu1.free(ptr)
                        except Exception:
                            pass
                self._gpu_buf = None

            if getattr(self, '_gpu_norm_ptrs', None) is not None:
                for ptr in self._gpu_norm_ptrs.values():
                    if ptr is not None:
                        try:
                            cu1.free(ptr)
                        except Exception:
                            pass
                self._gpu_norm_ptrs = None

            if getattr(self, '_gpu_output_norm', None) is not None:
                try:
                    cu1.free(self._gpu_output_norm)
                except Exception:
                    pass
                self._gpu_output_norm = None

        # GPU-resident weight VRAM + scratch (all GPUs)
        workers = getattr(self, '_gpu_workers', None) or []
        for w in workers:
            try:
                w.close()
            except Exception:
                pass
        self._gpu_workers = []
        if getattr(self, '_gpumm', None) is not None:
            try:
                self._gpumm.close()
            except Exception:
                pass
            self._gpumm = None
        if getattr(self, '_gpumm2', None) is not None:
            try:
                self._gpumm2.close()
            except Exception:
                pass
            self._gpumm2 = None
        # File handle (mmap'd weight data)
        if getattr(self, '_file', None) is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
        # Large RAM caches - these can be many GB
        self._raw_store = None
        self._layer_cache = None
        self._layer_cache_order = None
        self._layer_cache_bytes = 0
        self._expert_cache = None
        self._expert_cache_order = None
        self._expert_cache_bytes = 0
        # gemma4 GPU-resident expert cache: free device memory explicitly
        # before the GpuMM context goes away.
        _gc = getattr(self, '_g4_exp_gcache', None)
        if _gc:
            for _k, _d in list(_gc.items()):
                self._evict_g4_expert(_k, _d)
            _gc.clear()
        self._g4_exp_gcache = None
        self._embedding = None
        self._output_weight = None
        self._output_gpu = None
        self.kv_k = None
        self.kv_v = None
        # Source readers (GGUF parser / NSVD / V1 / V16) - hold mmap data
        self._parser = None
        self._nsvd = None
        self.m = None
        import gc
        gc.collect()
    
    @property
    def name(self):
        return Path(self.model_path).stem
    
    @property
    def architecture(self):
        return getattr(self, '_architecture', 'llama')
    
    @architecture.setter
    def architecture(self, value):
        self._architecture = value
    @property
    def bos_id(self):
        return getattr(self, '_bos_id', 1)
    
    @bos_id.setter
    def bos_id(self, val):
        self._bos_id = val
    
    @property
    def eos_id(self):
        return getattr(self, '_eos_id', 2)
    
    @eos_id.setter
    def eos_id(self, val):
        self._eos_id = val
    
    def _suite_tokenizer(self):
        """The canonical Tokenizer for this model, built once.

        The engine used to carry its own `encode()`: a per-character lookup
        into `token_to_id` followed by a hand-rolled BPE merge loop. That
        cannot work for a byte-level BPE vocabulary. Single characters are
        not in the vocab at all (the entries are things like 'Th', 'Ġc'),
        so every character missed, fell through to `<0xNN>` byte tokens that
        are also absent, and got id 0. The result was [bos, 0, 0, 0, ...]
        regardless of the input -- and the merge loop then had nothing real
        to merge.

        `tokenizer.Tokenizer` handles both the GGUF string-merge form and the
        NTMRE id-pair merge form, applies the GPT-2 byte encoder, and reads
        the specials from the container. It is the verified implementation,
        so delegate to it rather than maintaining a second one that silently
        disagrees.
        """
        t = getattr(self, '_suite_tok', None)
        if t is not None:
            return t
        try:
            from tokenizer import Tokenizer as _T
        except Exception:
            return None

        # GGUF: prefer the parser's own metadata, which is authoritative.
        meta = dict(getattr(self, 'metadata', {}) or {})
        if not meta.get('tokenizer.ggml.tokens'):
            p = getattr(self, '_parser', None)
            m = getattr(p, 'm', None)
            mm = getattr(m, 'metadata', None)
            if isinstance(mm, dict):
                meta.update(mm)
        try:
            t = _T(meta, model_path=getattr(self, 'model_path', None))
        except Exception:
            t = None
        if t is not None and not getattr(t, 'tokens', None):
            try:
                from ntmre_v1_engine import V1Source as _V1
                src = _V1(self.model_path)
                tb = getattr(src, 'tokenizer_bytes', None)
                if tb:
                    import json as _json
                    d = _json.loads(tb.decode('utf-8'))
                    t = _T({'tokenizer.ggml.tokens': d.get('tokens', []),
                            'tokenizer.ggml.merges': d.get('merges', []),
                            'tokenizer.ggml.bos_token_id': d.get('bos', 1),
                            'tokenizer.ggml.eos_token_id': d.get('eos', 2)},
                           model_path=self.model_path)
                src.close()
            except Exception:
                pass
        if t is not None and getattr(t, 'tokens', None):
            self._suite_tok = t
            # Keep the engine's own ids in step so bos_id/eos_id agree.
            try:
                if isinstance(getattr(t, 'bos_id', None), int):
                    self._bos_id = t.bos_id
                if isinstance(getattr(t, 'eos_id', None), int):
                    self._eos_id = t.eos_id
            except Exception:
                pass
            return t
        return None

    def encode(self, text: str, add_bos: bool = True) -> list:
        t = self._suite_tokenizer()
        if t is not None:
            try:
                return t.encode(text, add_bos=add_bos)
            except Exception:
                pass

        # No usable Tokenizer: fall back to the old path, but say so loudly
        # rather than silently returning a stream of token 0.
        print(f"[STREAM] WARNING: no usable tokenizer for "
              f"{getattr(self, 'model_path', '?')}; falling back to "
              f"per-character encoding, which is byte-exact only for "
              f"character-level vocabs.")
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)

        token_to_id = getattr(self, 'token_to_id', {})
        for ch in text:
            token_id = token_to_id.get(ch)
            if token_id is not None:
                tokens.append(token_id)
            else:
                for byte in ch.encode('utf-8'):
                    byte_token = f"<0x{byte:02X}>"
                    bid = token_to_id.get(byte_token, 0)
                    tokens.append(bid)

        if getattr(self, 'merges', []):
            tokens = self._apply_bpe(tokens)

        return tokens
    
    def _apply_bpe(self, tokens: list) -> list:
        merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        while True:
            best_pair = None
            best_rank = len(self.merges)
            best_idx = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                rank = merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
                    best_idx = i
            if best_pair is None:
                break
            a_str = self.tokens[best_pair[0]] if best_pair[0] < len(self.tokens) else ""
            b_str = self.tokens[best_pair[1]] if best_pair[1] < len(self.tokens) else ""
            merged_str = a_str + b_str
            merged_id = self.token_to_id.get(merged_str)
            if merged_id is None:
                del merge_rank[best_pair]
                continue
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(merged_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens
    
    def decode(self, token_ids: list) -> str:
        # Prefer the canonical Tokenizer so decode matches encode exactly.
        t = self._suite_tokenizer()
        if t is not None:
            try:
                return t.decode(list(token_ids))
            except Exception:
                pass
        tok_list = getattr(self, 'tokens', [])
        pieces = []
        for tid in token_ids:
            if not (0 <= tid < len(tok_list)):
                continue
            tok = tok_list[tid]
            if self._is_control_like(tok):
                continue
            pieces.append(self._clean_token(tok))
        return ''.join(pieces)
    
    def decode_single(self, token_id: int) -> str:
        t = self._suite_tokenizer()
        if t is not None and hasattr(t, 'decode_single'):
            try:
                return t.decode_single(int(token_id))
            except Exception:
                pass
        tok_list = getattr(self, 'tokens', [])
        if 0 <= token_id < len(tok_list):
            tok = tok_list[token_id]
            if self._is_control_like(tok):
                return ""
            return self._clean_token(tok)
        return ""
    
    def _clean_token(self, token: str) -> str:
        token = token.replace("▁", " ").replace("<0x0A>", "\n")
        if token.startswith("<0x") and token.endswith(">"):
            try:
                return chr(int(token[3:-1], 16))
            except (ValueError, OverflowError):
                return token
        return token
    
    def _is_control_like(self, token: str) -> bool:
        return token in ("<s>", "</s>", "<pad>", "<unk>", "<|endoftext|>", 
                         "<|im_start|>", "<|im_end|>", "<bos>", "<eos>")
    
    @property
    def metadata(self):
        md = {
            'general.architecture': self.architecture,
            'general.name': self.name,
        }
        # GGUF: expose the full parsed metadata (tokenizer included)
        m = getattr(self, 'm', None)
        if m is not None and getattr(m, 'metadata', None):
            return m.metadata
        # NTMRE containers: expose the embedded tokenizer so the suite's
        # Tokenizer works exactly like with a GGUF file
        tb = getattr(getattr(self, '_nsvd', None), 'tokenizer_bytes', None)
        if tb:
            try:
                import json as _json
                d = _json.loads(tb.decode('utf-8'))
                md['tokenizer.ggml.tokens'] = d.get('tokens', [])
                md['tokenizer.ggml.merges'] = d.get('merges', [])
                md['tokenizer.ggml.bos_token_id'] = d.get('bos', 1)
                md['tokenizer.ggml.eos_token_id'] = d.get('eos', 2)
                md['tokenizer.ggml.padding_token_id'] = d.get('pad', 0)
                md['tokenizer.ggml.unknown_token_id'] = d.get('unk', 3)
                if d.get('model'):
                    md['tokenizer.ggml.model'] = d['model']
            except Exception:
                pass
        return md
    
    # KV cache interface (compatible with unified app)
    class _KVCache:
        def __init__(self, parent):
            self._parent = parent
        def advance(self):
            self._parent.pos += 1
    
    @property
    def kv_cache(self):
        return self._KVCache(self)


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("💀 NIGHTMARE GPU STREAMING ENGINE")
    print("="*50)
    
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python nightmare_stream.py model.ntmre")
        print("  python nightmare_stream.py model.gguf \"prompt\"")
        print()
        print("Streams model through GPU layer-by-layer.")
        print("Runs ANY size model on consumer hardware.")
        sys.exit(0)
    
    model_path = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Hello"
    
    engine = GPUStreamEngine(model_path)
    
    if engine.loaded:
        print(f"\nGenerating from: \"{prompt}\"")
        engine.reset()
        
        # Tokenize prompt and generate text
        tokens = engine.encode(prompt, add_bos=True)
        print(f"  Tokens: {len(tokens)}")
        
        # Prefill prompt tokens
        generated = []
        for i, tok in enumerate(tokens[:-1]):
            logits = engine.forward(tok, i)
        
        # Generate from last prompt token
        next_tok = tokens[-1] if tokens else engine.bos_id
        max_new = 128
        for step in range(max_new):
            logits = engine.forward(next_tok, len(generated) + len(tokens) - 1)
            # Greedy sampling
            next_tok = int(np.argmax(logits))
            if next_tok == engine.eos_id:
                break
            generated.append(next_tok)
            piece = engine.decode_single(next_tok)
            if piece:
                print(piece, end="", flush=True)
        
        print(f"\n\nStats: {engine.get_stats()}")
        
        engine.close()
