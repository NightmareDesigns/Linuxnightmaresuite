# -*- coding: utf-8 -*-
"""
nightmare_fused_gpu.py — FusedDualGpuEngine: fully resident, no streaming
=================================================================================
The "fused engine": the whole model runs RESIDENT on whatever accelerator(s)
the machine actually has.  Every matvec is a fused dequant+matvec kernel on
the GPU owning the layer; nothing is ever evicted and re-streamed.

  Both GPUs  -> layer split across the two biggest cards:
                  GPU0 -> layers [0, split)
                  GPU1 -> layers [split, n_layers) + final norm + logits
  One GPU    -> all layers resident on that single card (incl. AMD 9700 XT)

The hidden state crosses the bus ONCE per token (one ~4-8 KB vector), and
only between two cards.  This is deliberately the best resident option for
whatever the user has: 2 GPUs, 1 NVIDIA GPU, or 1 AMD Radeon GPU.

NVIDIA runs on the proven qwen35_gpu.GpuMM (NVRTC, our own kernels).
AMD runs on HipGpuMM here in the admin copy only; the sealed edition uses prebuilt engines only.  Intel Level-Zero devices are detected by
nightmare_gpu_discovery but kernels for them land later; they're excluded
from fusion until then.  Only OS driver DLLs, ctypes, and our own source:
zero external packages, zero downloaded code.

If the model can't fit the resident VRAM, or the family needs math the
fused cards can't do (MoE / .ntmre compression / partial-RoPE / LayerNorm), the engine
transparently falls back to the proven GPUStreamEngine path — a machine
with any GPU is never slower than before.
"""
import ctypes
import os
import time
import numpy as np

try:
    from nightmare_stream import GPUStreamEngine
except Exception:  # pragma: no cover - import guard only
    GPUStreamEngine = object

from nightmare_gpu_discovery import probe_all, topology as _disp_topo

FUSED_TAG = "[FUSED]"

# Families the fused card runs correctly: dense, full RoPE, SwiGLU FFN,
# RMSNorm.  gemma4 -> Gemma4GpuEngine, qwen35 -> Qwen35HybridEngine,
# partial-RoPE / GELU / MoE families -> the parent streaming path.
# Note: the fused card runs exactly: dense, FULL RoPE, SiLU-gated FFN,
# RMSNorm.  gemma/baichuan/phi2 are intentionally absent (GeGLU/ALiBi/
# LayerNorm); gemma4 -> Gemma4GpuEngine, qwen35 -> Qwen35HybridEngine,
# partial-RoPE / MoE / LayerNorm families -> parent streaming path.
_FUSED_ARCHS = (
    "llama", "llama2", "llama3", "llama3.1", "llama3.2", "llama4",
    "qwen2", "qwen2.5",
    "mistral", "mixtral",
    "internlm", "internlm2",
    "command-r",
    "granite",
)


def _is_cuda(mm):
    """True for the NVIDIA GpuMM backend (has .cu.cu driver CDLL)."""
    return getattr(mm, "vendor", "") != "amd" and getattr(mm, "cu", None) is not None and hasattr(getattr(mm, "cu", None), "cu")


# ═══════════════════════════════════════════════════════════════════
# HIP (AMD) backend — the same embedded kernels from qwen35_gpu.KERNELS,
# Sealed edition: prebuilt engines only; no hiprtc Prebuilt GPU engine.
# ═══════════════════════════════════════════════════════════════════

_HIP_MEMCPY_H2D = 1
_HIP_MEMCPY_D2H = 2
_HIP_MEMCPY_D2D = 3


class HipCuda:
    """Bare-metal HIP runtime on amdhip64.dll.  ctypes only, no toolkit."""

    def __init__(self, device=0):
        self.vendor = "hip"
        self.hip = ctypes.CDLL("amdhip64.dll")
        if self.hip.hipInit(0) != 0:
            raise RuntimeError("hipInit failed")
        self.dev = ctypes.c_int(int(device))
        self.hip.hipSetDevice.argtypes = [ctypes.c_int]
        self.hip.hipSetDevice(self.dev)
        self.hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        self.hip.hipFree.argtypes = [ctypes.c_void_p]
        self.hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        try:
            self.hip.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        except Exception:
            pass
        self.hip.hipStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.hip.hipStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.hip.hipModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        self.hip.hipModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        self.hip.hipModuleLaunchKernel.argtypes = [
            ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        try:
            self.hip.hipModuleUnload.argtypes = [ctypes.c_void_p]
            self.hip.hipStreamDestroy.argtypes = [ctypes.c_void_p]
        except Exception:
            pass
        self.stream = ctypes.c_void_p()
        if self.hip.hipStreamCreate(ctypes.byref(self.stream)) != 0:
            raise RuntimeError("hipStreamCreate failed")

    def alloc(self, nbytes):
        ptr = ctypes.c_void_p()
        r = self.hip.hipMalloc(ctypes.byref(ptr), int(nbytes))
        if r != 0:
            raise RuntimeError("hipMalloc %d for %.1fMB" % (r, nbytes / 1e6))
        return ctypes.c_ulonglong(ptr.value)

    def free(self, ptr):
        self.hip.hipFree(ctypes.c_void_p(int(ptr.value or 0)))

    def htod(self, ptr, host_arr):
        arr = np.ascontiguousarray(host_arr, dtype=np.float32)
        self.hip.hipMemcpy(ctypes.c_void_p(int(ptr.value)),
                           ctypes.c_void_p(arr.ctypes.data), arr.nbytes, _HIP_MEMCPY_H2D)
        self.hip.hipStreamSynchronize(self.stream)

    def dtoh(self, ptr, n_elems):
        out = np.empty(int(n_elems), dtype=np.float32)
        self.hip.hipMemcpy(ctypes.c_void_p(out.ctypes.data),
                           ctypes.c_void_p(int(ptr.value)), out.nbytes, _HIP_MEMCPY_D2H)
        self.hip.hipStreamSynchronize(self.stream)
        return out

    def d2d(self, dst, src, nbytes):
        self.hip.hipMemcpy(ctypes.c_void_p(int(dst.value)),
                           ctypes.c_void_p(int(src.value)), int(nbytes), _HIP_MEMCPY_D2D)
        self.hip.hipStreamSynchronize(self.stream)

    def memset(self, ptr, value, nbytes):
        self.hip.hipMemset(ctypes.c_void_p(int(ptr.value)), int(value), int(nbytes))
        self.hip.hipStreamSynchronize(self.stream)

    def upload_raw(self, ptr, raw):
        buf = ctypes.create_string_buffer(raw, len(raw))
        self.hip.hipMemcpy(ctypes.c_void_p(int(ptr.value)), ctypes.cast(buf, ctypes.c_void_p),
                           len(raw), _HIP_MEMCPY_H2D)
        self.hip.hipStreamSynchronize(self.stream)

    def compile(self, src):
        # Sealed edition: prebuilt GPU engines only.
        raise RuntimeError("Sealed edition: prebuilt GPU engines only — hiprtc JIT compilation is not included. Use Select Prebuilt GPU Engine.")

    def _detect_arch(self):
        try:
            self.hip.hipGetDeviceProperties.argtypes = [ctypes.c_void_p, ctypes.c_int]
            buf = ctypes.create_string_buffer(4096)
            self.hip.hipGetDeviceProperties(buf, int(self.dev.value))
            idx = buf.raw.find(b"gfx")
            if idx >= 0:
                return buf.raw[idx:idx + 8].split(b"\x00")[0].decode(errors="replace")
        except Exception:
            pass
        return "gfx1100"

    def func(self, mod, name):
        fn = ctypes.c_void_p()
        r = self.hip.hipModuleGetFunction(ctypes.byref(fn), mod, name.encode())
        if r != 0 or not fn.value:
            raise RuntimeError("hipModuleGetFunction %s failed (%d)" % (name, r))
        return fn

    def launch(self, fn, grid, block, args, shared_mem=0):
        params = []
        for a in args:
            if isinstance(a, ctypes.c_ulonglong):
                params.append(ctypes.c_void_p(ctypes.addressof(a)))
            elif isinstance(a, ctypes.c_float):
                params.append(ctypes.c_void_p(ctypes.addressof(a)))
            elif isinstance(a, float):
                cf = ctypes.c_float(a)
                params.append(ctypes.c_void_p(ctypes.addressof(cf)))
            else:
                ci = ctypes.c_int(int(a))
                params.append(ctypes.c_void_p(ctypes.addressof(ci)))
        args_arr = (ctypes.c_void_p * len(params))(*params)
        r = self.hip.hipModuleLaunchKernel(
            fn, grid[0], grid[1], 1, block[0], block[1], 1, shared_mem,
            self.stream, args_arr, None)
        if r != 0:
            raise RuntimeError("hipModuleLaunchKernel failed (%d)" % r)
        self.hip.hipStreamSynchronize(self.stream)

    def sync(self):
        self.hip.hipStreamSynchronize(self.stream)

    def close(self):
        try:
            if getattr(self, "mod", None) is not None and self.mod and self.mod.value:
                self.hip.hipModuleUnload(self.mod)
                self.mod = None
            if self.stream and self.stream.value:
                self.hip.hipStreamDestroy(self.stream)
                self.stream = None
        except Exception:
            pass


class HipGpuMM:
    """AMD twin of qwen35_gpu.GpuMM: same embedded kernels, resident
    weights, fused dequant+matvec.  try_create() returns None on any AMD
    failure so fusion keeps working on NVIDIA / CPU."""

    _GTYPE2KERNEL = {
        "G_Q8_34": "matvec_q8_34", "G_Q8_36": "matvec_q8_36",
        "G_Q4_0": "matvec_q4_0", "G_Q5_0": "matvec_q5_0",
        "G_Q6_0": "matvec_q6_0", "G_Q4_K": "matvec_q4_k",
        "G_Q5_K": "matvec_q5_k", "G_Q6_K": "matvec_q6_k",
        "G_F16": "matvec_f16", "G_BF16": "matvec_bf16",
    }

    def __init__(self, device=0):
        import qwen35_gpu as q
        self.q = q
        self.device = device
        self.vendor = "amd"
        self.cu = HipCuda(device=device)
        self._budget = 1 << 62
        try:
            mod = self.cu.compile(q.KERNELS)
            def _f(n): return self.cu.func(mod, n)
            self.f_attn = _f("kernel_attention")
            self.f_attn_f16 = _f("kernel_attention_f16")
            self.f_flash = _f("kernel_flash_attention")
            self.f_flash_f16 = _f("kernel_flash_attention_f16")
            self.f_rmsnorm = _f("kernel_rmsnorm")
            self.f_layernorm = _f("kernel_layernorm")
            self.f_rope = _f("kernel_rope")
            self.f_silu = _f("kernel_silu_mul")
            self.f_add = _f("kernel_add")
            self.f_f16 = _f("kernel_f32_to_f16")
            self.f_matvec = _f("matvec_t")
            self._fused = {v: _f(v) for v in self._GTYPE2KERNEL.values()}
        except Exception as e:
            try:
                self.cu.close()
            except Exception:
                pass
            raise RuntimeError("AMD backend unavailable: %s" % e)
        self._raw = {}
        self._gtype_consts = {}
        for name in ("G_F32", "G_F16", "G_BF16", "G_Q1_0", "G_Q4_0", "G_Q5_0",
                     "G_Q6_0", "G_Q4_K", "G_Q5_K", "G_Q6_K", "G_Q8_34", "G_Q8_36"):
            self._gtype_consts[int(getattr(q, name))] = name

    @staticmethod
    def try_create(device=None):
        try:
            return HipGpuMM(device=0 if device is None else device)
        except Exception as e:
            print(f"{FUSED_TAG} AMD device {device} skipped: {e}", flush=True)
            return None

    # ── resident weights ──────────────────────────────────────────
    def has(self, name):
        return name in self._raw

    def upload_raw(self, name, raw, gtype, out_dim, in_dim):
        if name in self._raw:
            return
        nbytes = len(raw)
        ptr = self.cu.alloc(nbytes if nbytes else 4)
        if nbytes:
            self.cu.upload_raw(ptr, raw)
        self._raw[name] = (ptr, int(gtype), int(out_dim), int(in_dim))

    def upload_f32(self, name, arr, out_dim, in_dim):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        self.upload_raw(name, arr.tobytes(), 0, out_dim, in_dim)

    def to_gpu_ptr(self, arr):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        ptr = self.cu.alloc(arr.nbytes)
        self.cu.htod(ptr, arr)
        return ptr

    def from_gpu_ptr(self, ptr, n_elems):
        return self.cu.dtoh(ptr, n_elems)

    # ── compute ───────────────────────────────────────────────────
    def matvec_gpu(self, x_ptr, name, out_ptr=None, raw=None):
        if name in self._raw:
            ptr, gtype, out_dim, in_dim = self._raw[name]
            temp = None
        elif raw is not None:
            ptr = self.cu.alloc(len(raw))
            self.cu.upload_raw(ptr, raw)
            gtype, out_dim, in_dim = 0, 0, 0
            temp = ptr
        else:
            raise KeyError(name)
        try:
            if out_ptr is None:
                out_ptr = self.cu.alloc(out_dim * 4)
            gname = self._gtype_consts.get(int(gtype), "G_F32")
            fname = self._GTYPE2KERNEL.get(gname)
            if fname is None:
                self.cu.launch(self.f_matvec, ((out_dim + 255) // 256, 1), (256, 1),
                               [ctypes.c_ulonglong(x_ptr.value), ptr, out_ptr, in_dim, out_dim])
            else:
                self.cu.launch(self._fused[fname], ((out_dim + 127) // 128, 1), (128, 1),
                               [ptr, ctypes.c_ulonglong(x_ptr.value), out_ptr, in_dim, out_dim])
            return out_ptr, out_dim
        finally:
            if temp is not None:
                self.cu.free(temp)

    def rmsnorm_gpu(self, x_ptr, w_ptr, dim, eps=1e-5, out_ptr=None):
        if out_ptr is None:
            out_ptr = self.cu.alloc(dim * 4)
        self.cu.launch(self.f_rmsnorm, (1, 1), (min(int(dim), 256), 1),
                       [out_ptr, ctypes.c_ulonglong(x_ptr.value),
                        ctypes.c_ulonglong(w_ptr.value), dim, eps])
        return out_ptr

    def layernorm_gpu(self, x_ptr, w_ptr, dim, eps=1e-5, out_ptr=None):
        if out_ptr is None:
            out_ptr = self.cu.alloc(dim * 4)
        self.cu.launch(self.f_layernorm, (1, 1), (min(int(dim), 256), 1),
                       [out_ptr, ctypes.c_ulonglong(x_ptr.value),
                        ctypes.c_ulonglong(w_ptr.value), dim, eps])
        return out_ptr

    def rope_gpu(self, q_ptr, k_ptr, n_heads, n_kv_heads, head_dim, pos, rope_base=10000.0):
        total = max(int(n_heads), int(n_kv_heads)) * (int(head_dim) // 2)
        self.cu.launch(self.f_rope, ((total + 255) // 256, 1), (256, 1),
                       [ctypes.c_ulonglong(q_ptr.value), ctypes.c_ulonglong(k_ptr.value),
                        n_heads, n_kv_heads, head_dim, pos, rope_base])

    def _attn(self, flash_fn, plain_fn, q_ptr, k_ptr, v_ptr, out_ptr,
              n_heads, n_kv_heads, head_dim, seq_len, kv_stride):
        block = min(int(head_dim), 256)
        seq = seq_len.value if hasattr(seq_len, "value") else seq_len
        fn = flash_fn if int(seq) > 4096 else plain_fn
        self.cu.launch(fn, (int(n_heads), 1), (block, 1),
                       [out_ptr, ctypes.c_ulonglong(q_ptr.value),
                        ctypes.c_ulonglong(k_ptr.value), ctypes.c_ulonglong(v_ptr.value),
                        n_heads, n_kv_heads, head_dim, seq, kv_stride])
        return out_ptr

    def attention_gpu(self, q_ptr, k_ptr, v_ptr, out_ptr, n_heads, n_kv_heads,
                      head_dim, seq_len, kv_stride):
        return self._attn(self.f_flash, self.f_attn, q_ptr, k_ptr, v_ptr, out_ptr,
                          n_heads, n_kv_heads, head_dim, seq_len, kv_stride)

    def attention_f16_gpu(self, q_ptr, k_ptr, v_ptr, out_ptr, n_heads, n_kv_heads,
                          head_dim, seq_len, kv_stride):
        return self._attn(self.f_flash_f16, self.f_attn_f16, q_ptr, k_ptr, v_ptr, out_ptr,
                          n_heads, n_kv_heads, head_dim, seq_len, kv_stride)

    def silu_mul_gpu(self, gate_ptr, up_ptr, dim, out_ptr=None):
        if out_ptr is None:
            out_ptr = self.cu.alloc(dim * 4)
        self.cu.launch(self.f_silu, ((dim + 255) // 256, 1), (256, 1),
                       [out_ptr, ctypes.c_ulonglong(gate_ptr.value),
                        ctypes.c_ulonglong(up_ptr.value), dim])
        return out_ptr

    def add_gpu(self, a_ptr, b_ptr, dim):
        self.cu.launch(self.f_add, ((dim + 255) // 256, 1), (256, 1),
                       [ctypes.c_ulonglong(a_ptr.value), ctypes.c_ulonglong(b_ptr.value), dim])
        return a_ptr

    def store_f16_gpu(self, f32_ptr, f16_ptr, n_elems):
        a = f32_ptr if isinstance(f32_ptr, ctypes.c_ulonglong) else ctypes.c_ulonglong(f32_ptr)
        b = f16_ptr if isinstance(f16_ptr, ctypes.c_ulonglong) else ctypes.c_ulonglong(f16_ptr)
        self.cu.launch(self.f_f16, ((n_elems + 255) // 256, 1), (256, 1), [a, b, n_elems])
        return f16_ptr

    def close(self):
        for name, (ptr, *_rest) in list(getattr(self, "_raw", {}).items()):
            try:
                self.cu.free(ptr)
            except Exception:
                pass
        self._raw.clear()
        try:
            self.cu.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# FusedDualGpuEngine
# ═══════════════════════════════════════════════════════════════════

def _free_bytes(mm):
    try:
        import gpu_info
        free = gpu_info.get_free_vram(int(getattr(mm, "device", 0)))
        if free > 0:
            return int(free)
    except Exception:
        pass
    return int(getattr(mm, "_budget", 0))


class FusedDualGpuEngine(GPUStreamEngine):
    """A model running fully resident on whatever accelerators exist:
    2 GPUs -> layer split across both; 1 GPU -> everything resident on
    that one card (no per-token streaming).  Falls back to the parent
    streaming engine whenever fusion cannot apply (no GPU, unsupported
    family, model too big for the resident VRAM)."""

    # ── GPU backend selection (called from __init__) ────────────────
    def _init_gpu(self):
        super()._init_gpu()
        self._fused_devs = []
        self._fused_ok = False
        self._layer_owner = {}
        self._fused_bufs = {}
        self._fused_norms = {}
        self._fused_kv = {}
        self._fused_outnorm = {}
        self._fused_split = 0
        self._fused_upload_active = False
        try:
            pair = []
            seen = set()
            for mm in (getattr(self, "_gpumm", None), getattr(self, "_gpumm2", None)):
                if mm is None:
                    continue
                mm.vendor = "nvidia"
                k = ("nvidia", int(getattr(mm, "device", 0)))
                if k in seen:
                    continue       # single-GPU box: _gpumm2 == same card
                seen.add(k)
                pair.append(mm)
            # Fill to two independent cards with AMD (HIP) backends.
            if len(pair) < 2:
                for d in probe_all():
                    if d["runtime"] != "hip" or not d["drivable"]:
                        continue
                    if ("amd", int(d["index"])) in seen:
                        continue
                    mm = HipGpuMM.try_create(device=int(d["index"]))
                    if mm is not None:
                        seen.add(("amd", int(d["index"])))
                        pair.append(mm)
                    if len(pair) >= 2:
                        break
            if not pair:
                print(f"{FUSED_TAG} fused engine needs at least one GPU "
                      f"(found none) - using CPU path", flush=True)
                return
            # biggest first = device 0
            pair.sort(key=_free_bytes, reverse=True)
            self._fused_devs = pair[:2]
            self._close_extra(pair[2:])
            gb = " + ".join(f"{int(mm._budget / 1e9)}GB" if getattr(mm, "_budget", 0)
                            else f"?GB" for mm in self._fused_devs)
            label = "DUAL-GPU" if len(self._fused_devs) > 1 else "SINGLE-GPU"
            print(f"{FUSED_TAG} {label} FUSED ENGINE READY "
                  f"({', '.join(getattr(mm, 'vendor', '?') for mm in self._fused_devs)}, "
                  f"budget ~{gb})", flush=True)
        except Exception as e:
            print(f"{FUSED_TAG} fused setup failed ({e}) - streaming path", flush=True)

    def _close_extra(self, devs):
        kept = {id(getattr(self, "_gpumm", None)), id(getattr(self, "_gpumm2", None))}
        for mm in devs:
            if id(mm) in kept:
                continue
            try:
                mm.close()
            except Exception:
                pass

    # ── model plumbing ─────────────────────────────────────────────
    def _load_model_header(self, path):
        super()._load_model_header(path)   # also calls _alloc_gpu_buffers()
        if getattr(self, "_fused_ok", False):
            self._graph_tried = True       # CUDA graphs are single-device

    def _fused_arch_ok(self):
        arch = str(getattr(self, "_architecture", "")).lower().lstrip("-").lstrip("_")
        if self._n_experts and int(self._n_experts) > 0:
            return False
        if getattr(self, "_use_layernorm", False):
            return False
        for base in _FUSED_ARCHS:
            if arch == base or arch.startswith(base + "-") or arch.startswith(base + "_"):
                return True
        return False

    def _alloc_gpu_buffers(self):
        """Fused path first; any failure falls straight through to the
        parent's single-GPU path (never slower than before)."""
        if not getattr(self, "_fused_devs", None):
            super()._alloc_gpu_buffers()
            return
        try:
            if not self._fused_arch_ok():
                arch = getattr(self, "_architecture", "?")
                print(f"{FUSED_TAG} arch '{arch}' not on fused card "
                      f"- streaming path", flush=True)
                self._fused_disable()
                super()._alloc_gpu_buffers()
                return
            if self._build_fused():
                self._gpu_buf = None     # parent GPU-resident loop stays off
                print(f"{FUSED_TAG} fused buffers live; model is fully "
                      f"resident on {len(self._fused_devs)} GPUs", flush=True)
                return
        except Exception as e:
            print(f"{FUSED_TAG} fused alloc failed ({e}) - streaming path", flush=True)
        self._fused_disable()
        super()._alloc_gpu_buffers()

    def _fused_disable(self):
        self._fused_ok = False
        self._layer_owner = {}
        self._fused_bufs = {}
        self._fused_norms = {}
        self._fused_kv = {}
        self._fused_outnorm = {}
        self._fused_split = 0

    def _build_fused(self):
        _nkv = self.n_kv_heads
        if isinstance(_nkv, (list, tuple)):
            _nkv = max(_nkv) if _nkv else self.n_heads
        q_dim = int(self.n_heads) * int(self.head_dim)
        kv_dim = int(_nkv) * int(self.head_dim)
        kv_bpe = 2 if self._kv_f16 else 4
        kv_total = int(self.n_layers) * int(self.max_seq_len) * kv_dim * kv_bpe * 2

        # Working buffers on each GPU (hidden state stays on VRAM).
        per_dev_bufs = (int(self.dim) * 4) * 5 + (q_dim * 4) * 2 + (int(self.hidden_dim) * 4) * 3
        heads = [max(0, _free_bytes(mm) - per_dev_bufs) for mm in self._fused_devs]

        # Per-layer bytes: streaming layer sizes when available.
        if getattr(self, "_layer_sizes", None) and len(self._layer_sizes) == int(self.n_layers):
            per_layer = [int(s) for s in self._layer_sizes]
        else:
            fsize = 0
            try:
                fsize = os.path.getsize(self.model_path)
            except Exception:
                pass
            per_layer = [max(1, int(fsize * 0.9 // max(1, int(self.n_layers))))] * int(self.n_layers)

        # Find the split so each GPU's [layers + its KV] fits.  Greedy
        # over all splits for the best-balanced fit (1 GPU: all layers
        # resident on that one card, so split = n_layers only if it fits).
        dev_count = len(self._fused_devs)
        best = None
        if dev_count == 1:
            b0 = sum(per_layer)
            kv0 = kv_total
            if b0 + kv0 <= heads[0]:
                best = (b0, 0)
                self._fused_split = int(self.n_layers)
        else:
            for split in range(int(self.n_layers) + 1):
                b0 = sum(per_layer[:split])
                b1 = sum(per_layer[split:])
                kv0 = kv_total * split // max(1, int(self.n_layers))
                kv1 = kv_total - kv0
                if b0 + kv0 <= heads[0] and b1 + kv1 <= heads[1]:
                    if best is None or abs(b0 - b1) < abs(best[0] - best[1]):
                        best = (b0, b1)
                        self._fused_split = split
        if best is None:
            need = sum(per_layer) / 1e9
            have = (heads[0] + (heads[1] if dev_count > 1 else 0)) / 1e9
            print(f"{FUSED_TAG} model ~{need:.1f}GB > resident VRAM "
                  f"~{have:.1f}GB - streaming path", flush=True)
            return False

        # Owner map + per-device KV ("cpu" keeps RAM kv_k/kv_v path).
        for layer in range(int(self.n_layers)):
            self._layer_owner[layer] = 0 if layer < self._fused_split else\
                (1 if dev_count > 1 else 0)
        for dev_id in range(dev_count):
            self._fused_kv[dev_id] = {
                "kv": None, "lo": (0 if dev_id == 0 else self._fused_split),
                "hi": (self._fused_split if dev_id == 0 else int(self.n_layers)),
            }

        # Per-device buffers / norms / KV / output norm.
        for dev_id, mm in enumerate(self._fused_devs):
            cu = getattr(mm, "cu", None)
            if cu is None:
                return False
            buf = {
                "x": cu.alloc(int(self.dim) * 4),
                "xn": cu.alloc(int(self.dim) * 4),
                "q": cu.alloc(int(q_dim) * 4),
                "k": cu.alloc(int(kv_dim) * 4),
                "v": cu.alloc(int(kv_dim) * 4),
                "attn": cu.alloc(int(q_dim) * 4),
                "tmp": cu.alloc(max(int(self.dim), int(q_dim)) * 4),
                "gate": cu.alloc(int(self.hidden_dim) * 4),
                "up": cu.alloc(int(self.hidden_dim) * 4),
                "silu": cu.alloc(int(self.hidden_dim) * 4),
            }
            self._fused_bufs[dev_id] = buf
            norms = {}
            for layer in range(self._fused_kv[dev_id]["lo"], self._fused_kv[dev_id]["hi"]):
                for ntype in ("attn_norm", "ffn_norm"):
                    w = np.ones(self.dim, dtype=np.float32)
                    try:
                        if hasattr(self, "m") and hasattr(self, "_parser"):
                            w = self._parser.load_tensor(
                                self.m, f"blk.{layer}.{ntype}.weight").astype(np.float32)
                        elif hasattr(self, "_nsvd"):
                            flat, _s = self._nsvd.read_flat(f"blk.{layer}.{ntype}.weight")
                            w = np.asarray(flat, dtype=np.float32)
                    except Exception:
                        pass
                    norms[(layer, ntype)] = mm.to_gpu_ptr(w)
            self._fused_norms[dev_id] = norms
            self._fused_outnorm[dev_id] = mm.to_gpu_ptr(
                np.asarray(self._output_norm, dtype=np.float32))
            try:
                k_ptrs = {l: cu.alloc(int(self.max_seq_len) * kv_dim * kv_bpe)
                          for l in range(self._fused_kv[dev_id]["lo"],
                                         self._fused_kv[dev_id]["hi"])}
                v_ptrs = {l: cu.alloc(int(self.max_seq_len) * kv_dim * kv_bpe)
                          for l in range(self._fused_kv[dev_id]["lo"],
                                         self._fused_kv[dev_id]["hi"])}
                self._fused_kv[dev_id]["kv"] = ("gpu", k_ptrs, v_ptrs, kv_dim, kv_bpe)
            except Exception:
                self._fused_kv[dev_id]["kv"] = ("cpu", None, None, kv_dim, kv_bpe)

        # Preload every layer's raw weights straight onto its owning GPU
        # (no streaming, no eviction).  Route uploads to the owner.
        self._layer_cache = {}
        self._layer_cache_order = []
        self._layer_cache_bytes = 0
        self._layer_cache_budget = int(os.environ.get(
            "NIGHTMARE_LAYER_CACHE_MB", "131072")) * 1024 * 1024
        self._fused_upload_active = True
        try:
            for layer in range(int(self.n_layers)):
                self._load_layer(layer)
        finally:
            self._fused_upload_active = False

        # Every GPU weight must have landed resident on its owner.
        for layer in range(int(self.n_layers)):
            owner = self._layer_owner[layer]
            mm = self._fused_devs[owner]
            lw = self._layer_cache.get(layer, {})
            for key in ("wq", "wk", "wv", "wo", "wg", "wu", "wd"):
                w = lw.get(key)
                if isinstance(w, tuple) and w and isinstance(w[0], str):
                    kind, name = w[0], w[1]
                    if kind == "gpu":
                        entry = mm._raw.get(name)
                        if entry is None or entry[0] == "stream":
                            print(f"{FUSED_TAG} '{name}' does not fit on dev{owner} "
                                  f"- streaming path", flush=True)
                            return False
                    elif kind.startswith("gsvd"):
                        print(f"{FUSED_TAG} .ntmre compression weights present - streaming path", flush=True)
                        return False

        # Logits: fused path runs the output projection on CPU unless the
        # tail GPU already holds it, so make sure _output_weight is resident
        # in RAM (the parent skips it when _output_gpu is a GPU descriptor).
        if getattr(self, "_output_weight", None) is None:
            try:
                if hasattr(self, "m") and hasattr(self, "_parser"):
                    self._output_weight = self._parser.load_tensor(
                        self.m, "output.weight").astype(np.float32)
                elif hasattr(self, "_nsvd"):
                    flat, _sh = self._nsvd.read_flat("output.weight")
                    self._output_weight = flat.reshape(
                        (flat.size // self.dim, self.dim)).astype(np.float32)
            except Exception:
                self._output_weight = None
        if self._output_weight is None:
            self._output_weight = getattr(self, "_embedding", None)
        if self._output_weight is None or self._output_weight.shape[0] != self.vocab_size:
            self._output_weight = np.zeros(
                (self.vocab_size, self.dim), dtype=np.float32)
        self._fused_ok = True
        n_ln = int(self.n_layers)
        if len(self._fused_devs) > 1:
            print(f"{FUSED_TAG} layer split: dev0=[0,{self._fused_split}) "
                  f"dev1=[{self._fused_split},{n_ln}) - resident", flush=True)
        else:
            print(f"{FUSED_TAG} all {n_ln} layers resident on the "
                  f"single GPU - resident", flush=True)
        return True

    # ── weight routing to the owning GPU ─────────────────────────────
    def _upload_weight(self, name, raw, gtag, out_dim, in_dim):
        active = (getattr(self, "_fused_upload_active", False)
                  and getattr(self, "_fused_devs", None)
                  and hasattr(self, "_layer_owner"))
        if not active:
            return super()._upload_weight(name, raw, gtag, out_dim, in_dim)
        layer = -1
        if name.startswith("blk."):
            try:
                layer = int(name.split(".")[1])
            except Exception:
                layer = -1
        owner = self._layer_owner.get(layer, 0)
        mm = self._fused_devs[owner]
        mm.upload_raw(name, raw, gtag, out_dim, in_dim)
        self._weight_gpu[name] = mm
        self._raw_store[name] = raw if isinstance(raw, (bytes, bytearray)) else bytes(raw)
        return mm

    # ── fused forward ───────────────────────────────────────────────
    def _memcpy_h2d(self, mm, ptr, arr):
        if _is_cuda(mm):
            mm.cu.cu.cuMemcpyHtoD_v2(ptr, ctypes.c_void_p(arr.ctypes.data), self.dim * 4)
        else:
            mm.cu.htod(ptr, arr)

    def _memcpy_d2d(self, mm, dst, src, nbytes):
        if _is_cuda(mm):
            mm.cu.cu.cuMemcpyDtoD_v2(dst, src, nbytes)
        else:
            mm.cu.d2d(dst, src, nbytes)

    def _fused_mm_gpu(self, x_ptr, w, layer, wkey, mm, buf, out_ptr=None):
        if isinstance(w, tuple) and w and isinstance(w[0], str) and w[0].startswith("g"):
            kind, name = w[0], w[1]
            if kind == "gpu":
                return mm.matvec_gpu(x_ptr, name, out_ptr, raw=None)
            if kind in ("gsvd", "gsvd_nsvd"):
                return mm.matvec_svd_gpu(x_ptr, name, out_ptr)
        if isinstance(w, np.ndarray):
            tname = f"_dyn_l{layer}_{wkey}"
            if not mm.has(tname):
                mm.upload_f32(tname, np.ascontiguousarray(w, dtype=np.float32),
                              w.shape[0], w.shape[1])
            return mm.matvec_gpu(x_ptr, tname, out_ptr)
        tname = f"blk.{layer}.{self._WT_NAMES.get(wkey, wkey)}"
        if not mm.has(tname):
            mm.upload_f32(tname, np.asarray(w, dtype=np.float32), w.shape[0], w.shape[1])
        return mm.matvec_gpu(x_ptr, tname, out_ptr)

    def _fused_layer_forward(self, x_ptr, layer, position, weights, mm, buf):
        """One layer, fully on its owning GPU: norm -> QKV -> RoPE ->
        KV cache -> attention -> output proj -> FFN (fused matvecs)."""
        _nkv = self.n_kv_heads
        if isinstance(_nkv, (list, tuple)):
            _nkv = max(_nkv) if _nkv else self.n_heads
        q_dim = int(self.n_heads) * int(self.head_dim)
        kv_dim = int(_nkv) * int(self.head_dim)
        dev_id = self._layer_owner.get(layer, 0)
        norms = self._fused_norms[dev_id]
        kv = self._fused_kv[dev_id]["kv"]

        # Attention norm
        norm_ptr = norms.get((layer, "attn_norm"))
        if norm_ptr is not None:
            if self._use_layernorm:
                mm.layernorm_gpu(x_ptr, norm_ptr, self.dim, self._norm_eps, buf["xn"])
            else:
                mm.rmsnorm_gpu(x_ptr, norm_ptr, self.dim, self._norm_eps, buf["xn"])
        else:
            self._memcpy_d2d(mm, buf["xn"], x_ptr, self.dim * 4)

        # QKV projections (resident on this device)
        self._fused_mm_gpu(buf["xn"], weights["wq"], layer, "wq", mm, buf, buf["q"])
        self._fused_mm_gpu(buf["xn"], weights["wk"], layer, "wk", mm, buf, buf["k"])
        self._fused_mm_gpu(buf["xn"], weights["wv"], layer, "wv", mm, buf, buf["v"])
        if weights.get("bq") is not None:
            for bkey, bbuf, bdim in (("bq", "q", q_dim), ("bk", "k", kv_dim), ("bv", "v", kv_dim)):
                bname = f"blk.{layer}.{bkey}"
                if not mm.has(bname):
                    mm.upload_f32(bname, np.asarray(weights[bkey], dtype=np.float32), bdim, 1)
                mm.add_gpu(buf[bbuf], mm._raw[bname][0], bdim)

        # RoPE (full-head, in-place)
        mm.rope_gpu(buf["q"], buf["k"], self.n_heads, self.n_kv_heads,
                    self.head_dim, position, self.rope_base)

        # KV cache + attention on the owning device
        if kv is not None and kv[0] == "gpu":
            kv_stride = kv[3]
            kv_bpe = kv[4] if len(kv) > 4 else 4
            kv_off = position * kv_stride * kv_bpe
            if kv_bpe == 2:
                k_dst = ctypes.c_ulonglong(kv[1][layer].value + kv_off)
                v_dst = ctypes.c_ulonglong(kv[2][layer].value + kv_off)
                mm.store_f16_gpu(buf["k"], k_dst, kv_dim)
                mm.store_f16_gpu(buf["v"], v_dst, kv_dim)
                mm.attention_f16_gpu(buf["q"], kv[1][layer], kv[2][layer],
                                     buf["attn"], self.n_heads, self.n_kv_heads,
                                     self.head_dim, position + 1, kv_stride)
            else:
                self._memcpy_d2d(mm,
                                 ctypes.c_ulonglong(kv[1][layer].value + kv_off),
                                 buf["k"], kv_dim * 4)
                self._memcpy_d2d(mm,
                                 ctypes.c_ulonglong(kv[2][layer].value + kv_off),
                                 buf["v"], kv_dim * 4)
                mm.attention_gpu(buf["q"], kv[1][layer], kv[2][layer],
                                 buf["attn"], self.n_heads, self.n_kv_heads,
                                 self.head_dim, position + 1, kv_stride)
        else:
            # CPU KV cache (RAM): download, numpy attention, upload result
            q_cpu = mm.from_gpu_ptr(buf["q"], q_dim)
            k_cpu = mm.from_gpu_ptr(buf["k"], kv_dim)
            v_cpu = mm.from_gpu_ptr(buf["v"], kv_dim)
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
            scores = np.einsum("hd,phd->hp", q_heads, k_heads, optimize=True)
            scores *= np.float32(1.0 / np.sqrt(self.head_dim))
            scores -= np.max(scores, axis=1, keepdims=True)
            np.exp(scores, out=scores)
            scores /= np.sum(scores, axis=1, keepdims=True)
            attn_out = np.einsum("hp,phd->hd", scores, v_heads, optimize=True).astype(np.float32)
            if _is_cuda(mm):
                mm.cu.cu.cuMemcpyHtoD_v2(buf["attn"],
                                         ctypes.c_void_p(attn_out.ctypes.data), q_dim * 4)
            else:
                mm.cu.htod(buf["attn"], attn_out)

        # Output projection + residual
        self._fused_mm_gpu(buf["attn"], weights["wo"], layer, "wo", mm, buf, buf["tmp"])
        mm.add_gpu(x_ptr, buf["tmp"], self.dim)

        # FFN norm
        norm_ptr = norms.get((layer, "ffn_norm"))
        if norm_ptr is not None:
            if self._use_layernorm:
                mm.layernorm_gpu(x_ptr, norm_ptr, self.dim, self._norm_eps, buf["xn"])
            else:
                mm.rmsnorm_gpu(x_ptr, norm_ptr, self.dim, self._norm_eps, buf["xn"])
        else:
            self._memcpy_d2d(mm, buf["xn"], x_ptr, self.dim * 4)

        # FFN gate / up / SiLU / down
        self._fused_mm_gpu(buf["xn"], weights["wg"], layer, "wg", mm, buf, buf["gate"])
        self._fused_mm_gpu(buf["xn"], weights["wu"], layer, "wu", mm, buf, buf["up"])
        mm.silu_mul_gpu(buf["gate"], buf["up"], self.hidden_dim, buf["silu"])
        self._fused_mm_gpu(buf["silu"], weights["wd"], layer, "wd", mm, buf, buf["tmp"])
        mm.add_gpu(x_ptr, buf["tmp"], self.dim)
        return x_ptr

    def forward(self, token_id, position):
        if not getattr(self, "_fused_ok", False) or not getattr(self, "_fused_bufs", None):
            return super().forward(token_id, position)
        t0 = time.time()
        devs = self._fused_devs
        bufs = self._fused_bufs
        n_layers = int(self.n_layers)
        mm0, b0 = devs[0], bufs[0]

        # 1. embedding -> first device
        x = self._get_embedding_row(token_id)
        if str(getattr(self, "_architecture", "")).startswith("gemma"):
            x = x * np.float32(np.sqrt(self.dim))
        self._memcpy_h2d(mm0, b0["x"], x)

        # 2. one pass over every layer, hopping devices at split points
        cur = 0
        for layer in range(n_layers):
            owner = self._layer_owner.get(layer, cur)
            if owner != cur:
                hx = devs[cur].from_gpu_ptr(bufs[cur]["x"], self.dim)
                self._memcpy_h2d(devs[owner], bufs[owner]["x"], hx)
                cur = owner
            mm, buf = devs[cur], bufs[cur]
            lw = self._layer_cache.get(layer) if getattr(self, "_layer_cache", None) else None
            if lw is None:
                lw = self._load_layer(layer)
            self._fused_layer_forward(buf["x"], layer, position, lw, mm, buf)

        # 3. final norm + logits on the last layer's device
        tail = self._layer_owner.get(n_layers - 1, 0)
        mmt, bt = devs[tail], bufs[tail]
        if self._use_layernorm:
            mmt.layernorm_gpu(bt["x"], self._fused_outnorm[tail], self.dim,
                              self._norm_eps, bt["xn"])
        else:
            mmt.rmsnorm_gpu(bt["x"], self._fused_outnorm[tail], self.dim,
                            self._norm_eps, bt["xn"])
        xnorm = mmt.from_gpu_ptr(bt["xn"], self.dim)
        logits = self._output_weight @ xnorm

        self._times["total"] += time.time() - t0
        self._tokens_generated += 1
        return logits

    # ── lifecycle ───────────────────────────────────────────────────
    def _memset(self, mm, arr, nbytes):
        if _is_cuda(mm):
            mm.cu.cu.cuMemsetD8_v2(arr, 0, nbytes)
        else:
            mm.cu.memset(arr, 0, nbytes)

    def reset(self):
        super().reset()
        kv = getattr(self, "_fused_kv", None) or {}
        for dev_id in (0, 1):
            if dev_id >= len(self._fused_devs):
                break
            mm = self._fused_devs[dev_id]
            info = kv.get(dev_id)
            if info and info["kv"] and info["kv"][0] == "gpu":
                try:
                    n = int(self.max_seq_len) * info["kv"][3] * info["kv"][4]
                    for arr in list(info["kv"][1].values()) + list(info["kv"][2].values()):
                        self._memset(mm, arr, n)
                except Exception:
                    pass

    def free_gpu(self):
        try:
            self._fused_disable()
            kept = getattr(self, "_gpumm", None)
            kept2 = getattr(self, "_gpumm2", None)
            for mm in list(getattr(self, "_fused_devs", []) or []):
                if mm is kept or mm is kept2:
                    continue
                try:
                    mm.close()
                except Exception:
                    pass
            self._fused_devs = []
        except Exception:
            pass
        super().free_gpu()


if __name__ == "__main__":
    print(_disp_topo())
    print()
    print("Run through ntmre_router.create_engine(path); the fused engine "
          "is picked automatically for any usable GPU (NVIDIA or AMD).")