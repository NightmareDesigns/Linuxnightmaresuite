"""
NIGHTMARE ENGINE — Python Link
==================================
Loads compiled CUDA DLL, reads NSVD/NTMRE model files, uploads weights
(.ntmre compression stays factored — never decompresses), runs full transformer
inference on GPU or CPU for comparison.

Usage:
    from nightmare_engine import NightmareEngine
    engine = NightmareEngine(backend="gpu")  # or "cpu"
    engine.load_model("model.ntmre")
    logits = engine.forward(token_id=1, position=0)

Author: Nightmare Designs
"""

import ctypes
import numpy as np
import os
import sys
import struct
import json
import subprocess
import time
from pathlib import Path


ENGINE_DIR = Path(__file__).parent
CUDA_SOURCE = ENGINE_DIR / "nightmare_engine.cu"
DLL_PATH = ENGINE_DIR / "nightmare_engine.dll"

# NSVD magic bytes
MAGIC_NMRV = 0x4E544D52  # NTMRE v2
MAGIC_NSVD = b'NSVD'
MAGIC_NTMX = b'NTMR'

# Weight index constants (must match nightmare_engine.cu)
W_EMB = 0
W_OUT = 1
W_OUT_NORM = 2
W_LAYER_BASE = 3
W_LAYER_STRIDE = 13

def W_ATTN_NORM(i):      return W_LAYER_BASE + W_LAYER_STRIDE * i + 0
def W_FFN_NORM(i):       return W_LAYER_BASE + W_LAYER_STRIDE * i + 1
def W_ATTN_Q(i):         return W_LAYER_BASE + W_LAYER_STRIDE * i + 2
def W_ATTN_K(i):         return W_LAYER_BASE + W_LAYER_STRIDE * i + 3
def W_ATTN_V(i):         return W_LAYER_BASE + W_LAYER_STRIDE * i + 4
def W_ATTN_OUT(i):       return W_LAYER_BASE + W_LAYER_STRIDE * i + 5
def W_FFN_GATE(i):       return W_LAYER_BASE + W_LAYER_STRIDE * i + 6
def W_FFN_DOWN(i):       return W_LAYER_BASE + W_LAYER_STRIDE * i + 7
def W_FFN_UP(i):         return W_LAYER_BASE + W_LAYER_STRIDE * i + 8
def W_ATTN_Q_NORM(i):    return W_LAYER_BASE + W_LAYER_STRIDE * i + 9   # per-head Q norm
def W_ATTN_K_NORM(i):    return W_LAYER_BASE + W_LAYER_STRIDE * i + 10  # per-head K norm
def W_POST_ATTN_NORM(i): return W_LAYER_BASE + W_LAYER_STRIDE * i + 11  # post-attention norm
def W_POST_FFN_NORM(i):  return W_LAYER_BASE + W_LAYER_STRIDE * i + 12  # post-ffn norm

# Architecture feature flags (must match nightmare_engine.cu)
ARCH_FLAG_QK_NORM    = 1 << 0
ARCH_FLAG_POST_NORMS = 1 << 1
ARCH_FLAG_EMB_SCALE  = 1 << 2
ARCH_FLAG_GELU_TANH  = 1 << 3
ARCH_FLAG_ROPE_SPLIT = 1 << 4
ARCH_FLAG_SWA        = 1 << 5

BLOCK_SIZE = 32
Q8_BLOCK_BYTES = 34

# ============================================================================
# Dequantization (from dequant_fast.py / ntmre_format_py.py / bonsai_pc_app.py)
# ============================================================================

def _fp16_to_fp32_vec(h):
    h = h.astype(np.int32)
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    v = np.where(e == 0, m * 2.0 ** -24,
                 np.where(e == 31, np.where(m == 0, np.inf, np.nan),
                          (1 + m / 1024.0) * 2.0 ** (e - 15)))
    return np.where(s == 1, -v, v).astype(np.float32)


def dequant_q8(data, n):
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    raw = np.frombuffer(data, dtype=np.uint8, count=n_blocks * Q8_BLOCK_BYTES)
    blocks = raw.reshape(n_blocks, Q8_BLOCK_BYTES)
    scales = blocks[:, :2].view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    quants = blocks[:, 2:].view(np.int8).astype(np.float32)
    return (scales * quants).ravel()[:n]


def dequant_q4_0(data, n):
    n_blocks = n // 32
    raw = np.frombuffer(data, dtype=np.uint8)
    blk = raw.reshape(n_blocks, 18)
    d = _fp16_to_fp32_vec(blk[:, 0].astype(np.uint16) | (blk[:, 1].astype(np.uint16) << 8))
    qs = blk[:, 2:18].astype(np.int32)
    lo = qs & 0xF
    hi = qs >> 4
    q = np.concatenate([lo, hi], axis=1).astype(np.float32) - 8.0
    return (d[:, None] * q).ravel()[:n]


def dequant_svd_q8(data, m, n):
    """Dequantize .ntmre compression -> (U, S, Vt) with .ntmre compression weight W = (U * S) @ Vt.

    Handles both dialects written by converters:
      - A/B (NSVD, nightmare_convert.svd_compress):
            rank(u32) + A(q8, m x rank) + B(q8, rank x n); S already folded
            into A (A=U*sqrt(S), B=sqrt(S)*Vt).  Returns U=A, S=ones, Vt=B.
      - U/S/Vt (NTMRE v2, ntmre_format_py.pack_svd):
            rank(u32) + U(q8, m x rank) + S(f32, rank) + Vt(q8, rank x n).
    Dialect is detected by data length, mirroring
    bonsai_pc_app._svd_unpack_factored.
    """
    rank = struct.unpack_from('<I', data, 0)[0]
    if rank == 0:
        pos = 4
        w = dequant_q8(data[pos:], m * n).reshape(m, n)
        return w.astype(np.float32), np.ones(n, dtype=np.float32), np.eye(n, dtype=np.float32)
    pos = 4
    u_len = Q8_BLOCK_BYTES * ((m * rank + BLOCK_SIZE - 1) // BLOCK_SIZE)
    v_len = Q8_BLOCK_BYTES * ((rank * n + BLOCK_SIZE - 1) // BLOCK_SIZE)
    u_data = data[pos:pos + u_len]; pos += u_len
    if len(data) >= pos + 4 * rank + v_len:
        # U/S/Vt dialect
        s_data = data[pos:pos + 4 * rank]; pos += 4 * rank
        v_data = data[pos:]
        U = dequant_q8(u_data, m * rank).reshape(m, rank)
        S = np.frombuffer(s_data, dtype=np.float32).copy()
        if U.shape[1] != rank or S.size != rank:
            raise ValueError(f".ntmre compression tensor corrupt: rank={rank}, U={U.shape}, S={S.size}")
        if not np.all(np.isfinite(S)) or (S.size and float(np.median(np.abs(S))) > 1e4):
            raise ValueError(".ntmre compression tensor is corrupt (singular values non-finite or ~1e6)")
        Vt = dequant_q8(v_data, rank * n).reshape(rank, n)
    else:
        # A/B dialect (S pre-folded) — W = A @ B, S is identity
        b_data = data[pos:]
        A = dequant_q8(u_data, m * rank).reshape(m, rank)
        B = dequant_q8(b_data, rank * n).reshape(rank, n)
        return A.astype(np.float32), np.ones(rank, dtype=np.float32), B.astype(np.float32)
    return U.astype(np.float32), S.astype(np.float32), Vt.astype(np.float32)


def dequant_tensor(dtype, data, shape):
    """NTMRE v2 dtype codes: 0=F32, 1=F16, 2=Q8, 3=Q4, 4=.ntmre compression."""
    n = int(np.prod(shape))
    if dtype == 0:  # F32
        return np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
    elif dtype == 1:  # F16
        return np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(shape).copy()
    elif dtype == 2:  # Q8
        return dequant_q8(data, n).reshape(shape)
    elif dtype == 4:  # .ntmre compression
        U, S, Vt = dequant_svd_q8(data, shape[0], shape[1])
        return ((U * S[np.newaxis, :]) @ Vt).astype(np.float32)
    else:
        raise ValueError(f"Unsupported dtype {dtype}")


def dequant_nsvd(dtype, data, shape):
    """NSVD dtype codes (bonsai): 0=F32, 1=Q8, 2=.ntmre compression."""
    n = int(np.prod(shape))
    if dtype == 0:  # F32
        return np.frombuffer(data, dtype=np.float32).reshape(shape).copy()
    elif dtype == 1:  # Q8
        return dequant_q8(data, n).reshape(shape)
    elif dtype == 2:  # .ntmre compression
        U, S, Vt = dequant_svd_q8(data, shape[0], shape[1])
        return ((U * S[np.newaxis, :]) @ Vt).astype(np.float32)
    else:
        raise ValueError(f"Unsupported NSVD dtype {dtype}")


# ============================================================================
# NSVD Reader (bonsai-style format: magic NSVD/NTMR)
# ============================================================================

def read_nsvd(path, return_svd=False):
    """Read NSVD format -> (config_dict, tensor_dict, tokenizer, svd_dict).
    svd_dict: {name: (U, S, Vt)} for .ntmre compression tensors (kept factored)."""
    import mmap

    with open(path, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        if mm[:4] not in (MAGIC_NSVD, MAGIC_NTMX):
            raise ValueError("Not an NSVD file")

        pos = 4
        version = struct.unpack_from('<I', mm, pos)[0]; pos += 4
        n_tensors = struct.unpack_from('<I', mm, pos)[0]; pos += 4
        cfg_len = struct.unpack_from('<I', mm, pos)[0]; pos += 4

        cfg_json = mm[pos:pos + cfg_len].decode('utf-8')
        cfg = json.loads(cfg_json)
        pos += cfg_len

        tok_len = struct.unpack_from('<I', mm, pos)[0]; pos += 4
        tokenizer_bytes = bytes(mm[pos:pos + tok_len]); pos += tok_len

        idx_len = struct.unpack_from('<I', mm, pos)[0]; pos += 4
        index = mm[pos:pos + idx_len]; pos += idx_len
        data_start = (pos + 63) & ~63

        entries = {}
        p = 0
        for _ in range(n_tensors):
            nl = index[p]; p += 1
            name = index[p:p + nl].decode('utf-8'); p += nl
            ttype = index[p]; p += 1
            nd = index[p]; p += 1
            shape = struct.unpack_from('<%dI' % nd, index, p); p += 4 * nd
            data_len = struct.unpack_from('<Q', index, p)[0]; p += 8
            entries[name] = (ttype, tuple(shape), data_len)

        offsets = {}
        off = data_start
        for name, (ttype, shape, data_len) in entries.items():
            offsets[name] = off
            off += data_len
            off = (off + 63) & ~63

        tensors = {}
        svd_tensors = {}
        for name, (ttype, shape, data_len) in entries.items():
            raw = bytes(mm[offsets[name]:offsets[name] + data_len])
            try:
                if ttype == 2 and return_svd:  # NSVD .ntmre compression — keep factored
                    U, S, Vt = dequant_svd_q8(raw, shape[0], shape[1])
                    svd_tensors[name] = (U, S, Vt)
                    tensors[name] = (U * S[np.newaxis, :]) @ Vt
                else:
                    tensors[name] = dequant_nsvd(ttype, raw, shape)
            except Exception as e:
                print(f"  Warning: failed to dequant {name}: {e}")

        mm.close()
        try:
            tokenizer = json.loads(tokenizer_bytes.decode('utf-8'))
        except Exception:
            tokenizer = {}

    return cfg, tensors, tokenizer, svd_tensors


# ============================================================================
# NTMRE v2 Reader (ntmre_format_py.py format: magic 0x4E544D52)
# ============================================================================

def read_ntmre_v2(path, return_svd=False):
    """Read NTMRE v2 format -> (config_dict, tensor_dict, svd_dict)."""
    with open(path, 'rb') as f:
        magic, version, flags, header_size = struct.unpack('<IIII', f.read(16))
        if magic != MAGIC_NMRV:
            raise ValueError(f"Not an NTMRE v2 file: magic=0x{magic:08X}")

        base = struct.unpack('<IIIIIIIfI', f.read(36))
        ext = struct.unpack('<IIIfIIII', f.read(32))
        per = list(struct.unpack('<64I', f.read(256)))

        cfg = {
            'dim': base[0], 'n_layers': base[1], 'n_heads': base[2],
            'n_kv_heads': base[3], 'hidden_dim': base[4], 'vocab_size': base[5],
            'max_seq_len': base[6], 'rope_base': base[7], 'head_dim': base[8],
            'arch_type': ext[0], 'q_output_dim': ext[1], 'sliding_window': ext[2],
            'softcap': ext[3], 'shared_kv_layers': ext[4],
        }

        th = struct.unpack('<10I', f.read(40))
        n_tokens, n_merges = th[0], th[1]
        eos_id = th[3]

        for _ in range(n_tokens):
            slen = struct.unpack('<H', f.read(2))[0]
            f.read(slen)
        f.read(4 * n_tokens)
        f.read(8 * n_merges)

        n_tensors = struct.unpack('<I', f.read(4))[0]
        entries = []
        for _ in range(n_tensors):
            ti = f.read(91)
            nl = ti[0]
            name = ti[1:1 + nl].decode('utf-8', 'replace')
            dtype = ti[65]
            nd = ti[66]
            d0 = struct.unpack('<I', ti[67:71])[0]
            d1 = struct.unpack('<I', ti[71:75])[0]
            off = struct.unpack('<Q', ti[75:83])[0]
            sz = struct.unpack('<Q', ti[83:91])[0]
            shape = (d0,) if nd == 1 else (d0, d1)
            entries.append((name, dtype, shape, off, sz))

        data_start = (f.tell() + 63) & ~63

        tensors = {}
        svd_tensors = {}
        for name, dtype, shape, off, sz in entries:
            f.seek(data_start + off)
            raw = f.read(sz)
            try:
                if dtype == 4 and return_svd:  # .ntmre compression
                    U, S, Vt = dequant_svd_q8(raw, shape[0], shape[1])
                    svd_tensors[name] = (U, S, Vt)
                    tensors[name] = (U * S[np.newaxis, :]) @ Vt
                else:
                    tensors[name] = dequant_tensor(dtype, raw, shape)
            except Exception as e:
                print(f"  Warning: failed to dequant {name}: {e}")

    cfg['eos_id'] = eos_id
    return cfg, tensors, svd_tensors


# ============================================================================
# GGUF Reader (basic, for common quant types)
# ============================================================================

def read_gguf(path):
    """Read a GGUF file -> (config_dict, tensor_dict).
    Supports F32 and F16 tensors. Q4/Q8 quantized tensors loaded via
    numpy dequant. Limited to common Llama/Qwen/Gemma architectures."""
    import struct

    GGUF_MAGIC = 0x46554747
    GGUF_VERSION = 3

    with open(path, 'rb') as f:
        magic, version = struct.unpack('<II', f.read(8))
        if magic != GGUF_MAGIC:
            raise ValueError("Not a GGUF file")

        n_tensors_q = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]

        # Read metadata KV pairs
        cfg = {}
        for _ in range(n_kv):
            klen = struct.unpack('<Q', f.read(8))[0]
            key = f.read(klen).decode('utf-8')
            val_type = struct.unpack('<I', f.read(4))[0]

            if val_type == 0:  # UINT8
                val = struct.unpack('<B', f.read(1))[0]
            elif val_type == 1:  # INT8
                val = struct.unpack('<b', f.read(1))[0]
            elif val_type == 2:  # UINT16
                val = struct.unpack('<H', f.read(2))[0]
            elif val_type == 3:  # INT16
                val = struct.unpack('<h', f.read(2))[0]
            elif val_type == 4:  # UINT32
                val = struct.unpack('<I', f.read(4))[0]
            elif val_type == 5:  # INT32
                val = struct.unpack('<i', f.read(4))[0]
            elif val_type == 6:  # FLOAT32
                val = struct.unpack('<f', f.read(4))[0]
            elif val_type == 7:  # BOOL
                val = bool(struct.unpack('<B', f.read(1))[0])
            elif val_type == 8:  # STRING
                slen = struct.unpack('<Q', f.read(8))[0]
                val = f.read(slen).decode('utf-8', errors='replace')
            elif val_type == 9:  # ARRAY
                atype = struct.unpack('<I', f.read(4))[0]
                alen = struct.unpack('<Q', f.read(8))[0]
                val = None  # skip complex arrays
            else:
                val = None

            cfg[key] = val

        # Map GGUF keys to our config
        mapped = {}
        mapping = {
            'embedding_length': 'dim',
            'block_count': 'n_layers',
            'attention.head_count': 'n_heads',
            'attention.head_count_kv': 'n_kv_heads',
            'feed_forward_length': 'ffn_dim',
            'context_length': 'max_seq_len',
            'vocab_size': 'vocab_size',
            'rope.freq_base': 'rope_base',
        }
        for gguf_key, our_key in mapping.items():
            if gguf_key in cfg and cfg[gguf_key] is not None:
                mapped[our_key] = cfg[gguf_key]

        # Defaults
        dim = mapped.get('dim', 0)
        n_heads = mapped.get('n_heads', 1)
        mapped.setdefault('head_dim', dim // n_heads if n_heads else 0)
        mapped.setdefault('n_kv_heads', n_heads)
        mapped.setdefault('ffn_dim', 4 * dim)
        mapped.setdefault('vocab_size', 32000)
        mapped.setdefault('max_seq_len', 4096)
        mapped.setdefault('rope_base', 10000.0)

        # Read tensor info
        tensor_info = []
        for _ in range(n_tensors_q):
            nlen = struct.unpack('<Q', f.read(8))[0]
            name = f.read(nlen).decode('utf-8')
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            dtype_gguf = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensor_info.append((name, dims, dtype_gguf, offset))

        # GGUF data offset (after alignment)
        data_start = f.tell()
        data_start = (data_start + 31) & ~31

        # Load F32 tensors only (skip quantized for now - too many types)
        tensors = {}
        for name, dims, dtype_gguf, offset in tensor_info:
            f.seek(data_start + offset)
            n_elements = 1
            for d in dims:
                n_elements *= d

            if dtype_gguf == 0:  # F32
                data = f.read(n_elements * 4)
                arr = np.frombuffer(data, dtype=np.float32).reshape(dims).copy()
                tensors[name] = arr
            elif dtype_gguf == 1:  # F16
                data = f.read(n_elements * 2)
                arr = np.frombuffer(data, dtype=np.float16).astype(np.float32).reshape(dims).copy()
                tensors[name] = arr

    return mapped, tensors


# ============================================================================
# NightmareEngine
# ============================================================================

class NightmareEngine:
    """
    GPU/CPU inference engine. Reads model files, uploads weights to CUDA DLL,
    runs full transformer forward pass. .ntmre compression stays factored (never decompress).

    Args:
        backend: "gpu", "cpu", or "auto" (default: "auto")
        dll_path: path to nightmare_engine.dll
    """

    def __init__(self, backend="auto", dll_path=None):
        self.dll_path = Path(dll_path) if dll_path else DLL_PATH
        self.lib = None
        self._available = False
        self._model_loaded = False
        self._pos = 0
        self._config = {}
        self._backend_name = backend

        if not self.dll_path.exists():
            print(f"[NTMRE] DLL not found: {self.dll_path}")
            return

        try:
            self.lib = ctypes.CDLL(str(self.dll_path))
            self._setup_functions()
            # Set backend BEFORE init so CPU mode works without GPU
            backend_map = {"gpu": 1, "cpu": 2, "auto": 0}
            self.lib.ntmre_set_backend(backend_map.get(self._backend_name, 0))
            ret = self.lib.ntmre_init()
            if ret == 0:
                self._available = True
            else:
                print("[NTMRE] CUDA init failed — GPU not available, CPU backend may still work")
        except Exception as e:
            print(f"[NTMRE] Failed to load DLL: {e}")

    def _setup_functions(self):
        self.lib.ntmre_init.restype = ctypes.c_int
        self.lib.ntmre_init.argtypes = []

        self.lib.ntmre_set_backend.restype = ctypes.c_int
        self.lib.ntmre_set_backend.argtypes = [ctypes.c_int]

        self.lib.ntmre_configure.restype = ctypes.c_int
        self.lib.ntmre_configure.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int, ctypes.c_uint, ctypes.c_float,
            ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ]

        self.lib.ntmre_upload_weight.restype = ctypes.c_int
        self.lib.ntmre_upload_weight.argtypes = [
            ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int
        ]

        self.lib.ntmre_upload_svd.restype = ctypes.c_int
        self.lib.ntmre_upload_svd.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ]

        self.lib.ntmre_forward.restype = ctypes.c_int
        self.lib.ntmre_forward.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
        ]

        self.lib.ntmre_reset.restype = None
        self.lib.ntmre_reset.argtypes = []

        self.lib.ntmre_unload.restype = None
        self.lib.ntmre_unload.argtypes = []

        self.lib.ntmre_get_vocab_size.restype = ctypes.c_int
        self.lib.ntmre_get_vocab_size.argtypes = []

        self.lib.ntmre_get_eos_id.restype = ctypes.c_int
        self.lib.ntmre_get_eos_id.argtypes = []

        self.lib.ntmre_get_n_layers.restype = ctypes.c_int
        self.lib.ntmre_get_n_layers.argtypes = []

        self.lib.ntmre_get_dim.restype = ctypes.c_int
        self.lib.ntmre_get_dim.argtypes = []

        self.lib.ntmre_get_device_vram_mb.restype = ctypes.c_int
        self.lib.ntmre_get_device_vram_mb.argtypes = []

        self.lib.ntmre_get_device_free_mb.restype = ctypes.c_int
        self.lib.ntmre_get_device_free_mb.argtypes = []

        self.lib.ntmre_get_seq_len.restype = ctypes.c_int
        self.lib.ntmre_get_seq_len.argtypes = []

        self.lib.ntmre_get_backend.restype = ctypes.c_int
        self.lib.ntmre_get_backend.argtypes = []

    @property
    def available(self):
        return self._available

    def device_count(self):
        """Number of CUDA GPUs (0 if none).

        Restored 2026-11-01: the admin rewrite dropped this method
        that nightmare_stream._init_gpu() calls, so a load crashed
        with AttributeError the moment the DLL was present again.
        Real count from GPU discovery when it works, else 1 if the
        DLL loaded, else 0 — same fallback as the v13 engine.
        """
        if not self._available:
            return 0
        try:
            from nightmare_gpu_discovery import usable_accelerators
            return len(usable_accelerators())
        except Exception:
            return 1

    @property
    def model_loaded(self):
        return self._model_loaded

    @property
    def vocab_size(self):
        return self.lib.ntmre_get_vocab_size() if self.lib else 0

    @property
    def eos_id(self):
        return self.lib.ntmre_get_eos_id() if self.lib else 2

    @property
    def config(self):
        return self._config

    @property
    def max_seq_len(self):
        return self.lib.ntmre_get_seq_len() if (self.lib and self._model_loaded) else 2048

    @property
    def metadata(self):
        return {
            "arch": self._config.get("arch", "llama"),
            "name": self._config.get("name", "ntmre_model"),
            "dim": self._config.get("dim", 0),
            "n_layers": self._config.get("n_layers", 0),
            "n_heads": self._config.get("n_heads", 0),
            "vocab_size": self._config.get("vocab_size", 0),
        }

    @property
    def name(self):
        return self._config.get("name", "ntmre_model")

    @property
    def architecture(self):
        return self._config.get("arch", "llama")

    class _KVCache:
        def __init__(self, lib):
            self.lib = lib
        def advance(self):
            pass

    @property
    def kv_cache(self):
        if self.lib and self._model_loaded:
            return self._KVCache(self.lib)
        return None

    def close(self):
        self.unload()

    def prefill_batch(self, tokens, start_pos):
        for i, t in enumerate(tokens):
            self.forward(int(t), start_pos + i)
        return None

    prefill = prefill_batch

    def _upload_weight(self, index, array):
        arr = np.ascontiguousarray(array, dtype=np.float32).ravel()
        n = arr.shape[0]
        ptr = arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        ret = self.lib.ntmre_upload_weight(index, ptr, n)
        if ret != 0:
            raise RuntimeError(f"ntmre_upload_weight failed for index {index}")
        return n

    def _upload_svd(self, index, U, S, Vt):
        """Upload .ntmre compression-factored weight.  Caller passes factored (U, S, Vt)
        with full weight W = (U * S) @ Vt, where U is (out, rank) and Vt is
        (rank, in).  The engine kernel computes W[col,i] = sum_r U[r,i]*S[r]*Vt[r,col]
        (U stored (rank, in), Vt stored (rank, out)), so we transpose here."""
        U = np.ascontiguousarray(U, dtype=np.float32)
        S = np.ascontiguousarray(S, dtype=np.float32)
        Vt = np.ascontiguousarray(Vt, dtype=np.float32)
        u_eng = Vt                      # (rank, in)  -> engine U slot
        vt_eng = np.ascontiguousarray(U.T)  # (rank, out) -> engine Vt slot
        u_rows, u_cols = u_eng.shape
        s_len = S.shape[0]
        vt_rows, vt_cols = vt_eng.shape
        ret = self.lib.ntmre_upload_svd(
            index,
            u_eng.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), u_rows, u_cols,
            S.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), s_len,
            vt_eng.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), vt_rows, vt_cols,
        )
        if ret != 0:
            raise RuntimeError(f"ntmre_upload_svd failed for index {index}")
        return u_rows * u_cols + s_len + vt_rows * vt_cols

    def load_model(self, path) -> bool:
        """Load a model file (.ntmre, .nsvd, or .gguf) into GPU/CPU.
        .ntmre compression tensors stay factored — never decompress to full matrix."""
        print(f"[NTMRE] Loading {path}...")
        t0 = time.time()

        # Detect format and read
        try:
            if str(path).lower().endswith('.gguf'):
                cfg, tensors = read_gguf(path)
                tokenizer = {}
                svd_tensors = {}  # GGUF doesn't have .ntmre compression
            else:
                # Try NTMRE v2 first (magic 0x4E544D52)
                with open(path, 'rb') as f:
                    magic_bytes = f.read(4)
                if magic_bytes == MAGIC_NSVD or magic_bytes == MAGIC_NTMX:
                    cfg, tensors, tokenizer, svd_tensors = read_nsvd(path, return_svd=True)
                else:
                    cfg, tensors, svd_tensors = read_ntmre_v2(path, return_svd=True)
                    tokenizer = {}
        except Exception as e:
            print(f"[NTMRE] Failed to read model: {e}")
            import traceback; traceback.print_exc()
            return False

        self._config = cfg

        dim = int(cfg.get('dim', 0))
        n_layers = int(cfg.get('n_layers', 0))
        n_heads = int(cfg.get('n_heads', 1))
        n_kv_heads = int(cfg.get('n_kv_heads', n_heads))
        vocab_size = int(cfg.get('vocab_size', 0))
        ffn_dim = int(cfg.get('ffn_dim', cfg.get('hidden_dim', 4 * dim)))
        rope_theta = float(cfg.get('rope_base', cfg.get('rope_theta', 10000.0)))
        max_seq_len = int(cfg.get('max_seq_len', 2048))
        eos_id = int(cfg.get('eos_id', 2))

        if dim == 0 or n_layers == 0 or vocab_size == 0:
            print(f"[NTMRE] Invalid config: dim={dim}, layers={n_layers}, vocab={vocab_size}")
            return False

        # ---- Architecture detection (gemma3 / qk-norm / post-norm models) ----
        all_names = set(tensors.keys()) | set(svd_tensors.keys())
        arch_hint = (str(cfg.get('arch', '')) + ' ' +
                     str(cfg.get('arch_type', '')) + ' ' +
                     str(cfg.get('name', cfg.get('model_name', '')))).lower()
        is_gemma3 = 'gemma3' in arch_hint

        head_dim = int(cfg.get('head_dim', 0) or 0)
        if head_dim <= 0 and n_heads > 0:
            for cand in (f'layers.0.wq', f'blk.0.attn_q.weight',
                         f'layers.0.attention.wq.weight'):
                q = tensors.get(cand)
                if q is not None and q.ndim == 2:
                    head_dim = q.shape[0] // n_heads
                    break
        if head_dim <= 0:
            head_dim = dim // max(n_heads, 1)

        norm_eps = float(cfg.get('norm_eps', 0.0) or 1e-5)
        rope_theta_local = float(cfg.get('rope_theta_local',
                                  cfg.get('rope_local_base_freq', 0.0)) or 0.0)
        sliding_window = int(cfg.get('sliding_window', 0) or 0)
        swa_pattern = int(cfg.get('sliding_window_pattern', 0) or 0)

        arch_flags = 0
        if is_gemma3:
            arch_flags |= (ARCH_FLAG_QK_NORM | ARCH_FLAG_POST_NORMS |
                           ARCH_FLAG_EMB_SCALE | ARCH_FLAG_GELU_TANH |
                           ARCH_FLAG_ROPE_SPLIT)
            if head_dim <= 0:
                head_dim = 256  # gemma3-1b fixed head_dim
            if rope_theta_local <= 0.0:
                rope_theta_local = 10000.0  # gemma3 local RoPE base
            # gemma3 configs carry window/pattern; default to the 1b values
            if sliding_window <= 0:
                sliding_window = 512
            if swa_pattern <= 0:
                swa_pattern = 6
            if sliding_window > 0 and swa_pattern > 0:
                arch_flags |= ARCH_FLAG_SWA
        else:
            # Generic opt-in from extra per-layer norm tensors
            if any('wq_output_norm' in n or 'attn_q_norm' in n or
                   'attn_qk_norm' in n for n in all_names):
                arch_flags |= ARCH_FLAG_QK_NORM
            if any('post_attention' in n or 'post_ffw' in n for n in all_names):
                arch_flags |= ARCH_FLAG_POST_NORMS

        print(f"[NTMRE] Config: dim={dim}, layers={n_layers}, heads={n_heads}, "
              f"kv_heads={n_kv_heads}, vocab={vocab_size}, ffn={ffn_dim}")
        if is_gemma3 or arch_flags:
            print(f"[NTMRE] Arch: flags=0x{arch_flags:X}, head_dim={head_dim}, "
                  f"norm_eps={norm_eps:.1e}, local_rope={rope_theta_local:.0f}, "
                  f"swa={sliding_window}/{swa_pattern}")

        # Set backend
        backend_map = {"gpu": 1, "cpu": 2, "auto": 0}
        self.lib.ntmre_set_backend(backend_map.get(self._backend_name, 0))

        # Configure DLL
        ret = self.lib.ntmre_configure(
            dim, n_layers, n_heads, n_kv_heads, vocab_size,
            ffn_dim, ctypes.c_float(rope_theta), max_seq_len, eos_id,
            head_dim, ctypes.c_uint(arch_flags),
            ctypes.c_float(rope_theta_local), sliding_window, swa_pattern,
            ctypes.c_float(norm_eps)
        )
        if ret != 0:
            print("[NTMRE] Configure failed")
            return False

        backend = self.lib.ntmre_get_backend()
        print(f"[NTMRE] Active backend: {'GPU' if backend == 1 else 'CPU'}")

        # Upload weights — .ntmre compression stays factored, full weights go as-is
        uploaded = 0
        svd_count = 0
        full_count = 0

        def upload(name, index, allow_svd=True):
            nonlocal uploaded, svd_count, full_count
            # Check for .ntmre compression factored version first
            if allow_svd and name in svd_tensors:
                U, S, Vt = svd_tensors[name]
                self._upload_svd(index, U, S, Vt)
                svd_count += 1
                uploaded += 1
                return True
            # Fall back to full weight
            if name in tensors:
                self._upload_weight(index, tensors[name])
                full_count += 1
                uploaded += 1
                return True
            return False

        # 0: embed (GGUF: token_embd.weight; NSVD converter: embed)
        upload('token_embd.weight', W_EMB, allow_svd=False) or\
            upload('embed', W_EMB, allow_svd=False)

        # 1: output.weight (tie with embedding if missing)
        if not upload('output.weight', W_OUT, allow_svd=False) and\
           not upload('output', W_OUT, allow_svd=False):
            print("  Tied output.weight with embeddings")
            self._upload_weight(W_OUT, tensors.get('token_embd.weight',
                                 tensors.get('embed',
                                 np.zeros((vocab_size, dim), dtype=np.float32))))

        # 2: final norm
        for cand in ['output_norm.weight', 'norm.weight', 'final_norm.weight',
                     'output_output_norm', 'output.output_norm']:
            if upload(cand, W_OUT_NORM, allow_svd=False):
                break
        else:
            self._upload_weight(W_OUT_NORM, np.ones(dim, dtype=np.float32))

        # Per-layer weights
        for i in range(n_layers):
            prefix = f'blk.{i}.'
            lpre = f'layers.{i}.'
            slots = [
                (W_ATTN_NORM(i), [f'{prefix}attn_norm.weight', f'{prefix}attention_norm.weight',
                                  f'{lpre}attn_norm', f'{lpre}attn_output_norm']),
                (W_FFN_NORM(i), [f'{prefix}ffn_norm.weight', f'{prefix}feed_forward_norm.weight',
                                 f'{lpre}ffn_norm', f'{lpre}ffn_output_norm']),
                (W_ATTN_Q(i), [f'{prefix}attn_q.weight', f'{prefix}attention.wq.weight',
                               f'{lpre}wq']),
                (W_ATTN_K(i), [f'{prefix}attn_k.weight', f'{prefix}attention.wk.weight',
                               f'{lpre}wk']),
                (W_ATTN_V(i), [f'{prefix}attn_v.weight', f'{prefix}attention.wv.weight',
                               f'{lpre}wv']),
                (W_ATTN_OUT(i), [f'{prefix}attn_output.weight', f'{prefix}attention.wo.weight',
                                 f'{lpre}wo']),
                (W_FFN_GATE(i), [f'{prefix}ffn_gate.weight', f'{prefix}feed_forward.w1.weight',
                                 f'{lpre}w_gate']),
                (W_FFN_DOWN(i), [f'{prefix}ffn_down.weight', f'{prefix}feed_forward.w2.weight',
                                 f'{lpre}w_down']),
                (W_FFN_UP(i), [f'{prefix}ffn_up.weight', f'{prefix}feed_forward.w3.weight',
                               f'{lpre}w_up']),
            ]
            # gemma3 per-head Q/K norms + post-attention/post-ffw norms
            if arch_flags & ARCH_FLAG_QK_NORM:
                slots += [
                    (W_ATTN_Q_NORM(i), [f'{lpre}wq_output_norm',
                                        f'{prefix}attn_q_norm.weight']),
                    (W_ATTN_K_NORM(i), [f'{lpre}wk_output_norm',
                                        f'{prefix}attn_k_norm.weight']),
                ]
            if arch_flags & ARCH_FLAG_POST_NORMS:
                slots += [
                    (W_POST_ATTN_NORM(i), [f'{lpre}post_attention_output_norm',
                                           f'{prefix}post_attention_norm.weight']),
                    (W_POST_FFN_NORM(i), [f'{lpre}post_ffw_output_norm',
                                          f'{prefix}post_ffw_norm.weight']),
                ]
            for w_idx, w_names in slots:
                found = False
                for cand in w_names:
                    if cand in svd_tensors or cand in tensors:
                        upload(cand, w_idx)
                        found = True
                        break
                if not found:
                    print(f"  Warning: layer {i} missing weight for index {w_idx}")

            if (i + 1) % 4 == 0 or i == n_layers - 1:
                print(f"  Layer {i+1}/{n_layers} uploaded")

        elapsed = time.time() - t0
        backend_str = "GPU" if backend == 1 else "CPU"
        print(f"[NTMRE] Model loaded: {full_count} full + {svd_count} .ntmre compression-factored "
              f"= {uploaded} weights in {elapsed:.1f}s [{backend_str}]")

        if backend == 1:
            free_mb = self.lib.ntmre_get_device_free_mb()
            print(f"[NTMRE] GPU free VRAM: {free_mb} MB")

        self._model_loaded = True
        self._pos = 0
        return True

    def forward(self, token_id: int, position: int = None) -> np.ndarray:
        """Run one token forward. Returns logits (vocab_size,) float32."""
        if not self._model_loaded:
            raise RuntimeError("No model loaded")

        pos = self._pos if position is None else int(position)
        vocab = self.lib.ntmre_get_vocab_size()
        logits = np.empty(vocab, dtype=np.float32)

        ret = self.lib.ntmre_forward(
            int(token_id), pos,
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        if ret != 0:
            raise RuntimeError(f"ntmre_forward failed (code {ret})")

        self._pos = pos + 1
        return logits

    def reset(self):
        if self.lib and self._model_loaded:
            self.lib.ntmre_reset()
        self._pos = 0

    def unload(self):
        if self.lib and self._model_loaded:
            self.lib.ntmre_unload()
        self._model_loaded = False
        self._pos = 0
        self._config = {}

    def shutdown(self):
        self.unload()


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    if "--compile" in sys.argv:
        print("\n  NIGHTMARE ENGINE — COMPILING")
        print("  " + "=" * 40)

        nvcc = None
        cuda_paths = [
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin\nvcc.exe",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin\nvcc.exe",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin\nvcc.exe",
            r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin\nvcc.exe",
        ]
        for p in cuda_paths:
            if os.path.exists(p):
                nvcc = p
                break
        if nvcc is None:
            try:
                r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
                if r.returncode == 0: nvcc = "nvcc"
            except FileNotFoundError:
                pass

        if nvcc is None:
            print("  NVCC not found!")
            sys.exit(1)

        cmd = [nvcc, "-shared", "-o", str(DLL_PATH), str(CUDA_SOURCE),
               "-O3", "--use_fast_math", "-Xcompiler", "/MD",
               "-gencode=arch=compute_75,code=sm_75",
               "-gencode=arch=compute_86,code=sm_86",
               "-gencode=arch=compute_89,code=sm_89"]

        print(f"  Compiling...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  FAILED:\n{result.stderr[:2000]}")
            sys.exit(1)
        print(f"  OK: {DLL_PATH}")

    elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        # Direct inference: python nightmare_engine.py model.ntmre "Hello" [--gpu|--cpu] [--bench]
        model_path = sys.argv[1]
        prompt = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else "Hello"
        backend = "gpu" if "--gpu" in sys.argv else "cpu" if "--cpu" in sys.argv else "auto"
        do_bench = "--bench" in sys.argv

        engine = NightmareEngine(backend=backend)
        if not engine.available and backend == "gpu":
            print("GPU not available — use --cpu for CPU-only mode")
            sys.exit(1)

        if not engine.load_model(model_path):
            print("Failed to load model")
            sys.exit(1)

        # Tokenize prompt (simple ASCII fallback)
        tokens = [ord(c) for c in prompt[:128]]
        eos = engine.eos_id

        if do_bench:
            # Benchmark mode: measure tokens/sec
            print(f"\n[BENCH] Backend: {'GPU' if engine.lib.ntmre_get_backend() == 1 else 'CPU'}")
            print(f"[BENCH] Warming up...")

            # Warmup
            engine.reset()
            for t in tokens:
                engine.forward(t)
            for _ in range(10):
                engine.forward(tokens[-1] if tokens else 1)

            # Timed run
            engine.reset()
            for t in tokens:
                engine.forward(t)

            n_gen = 100
            t0 = time.time()
            for _ in range(n_gen):
                engine.forward(tokens[-1] if tokens else 1)
            elapsed = time.time() - t0

            tok_per_sec = n_gen / elapsed
            ms_per_tok = elapsed / n_gen * 1000
            print(f"[BENCH] {n_gen} tokens in {elapsed:.2f}s")
            print(f"[BENCH] {tok_per_sec:.1f} tok/s  ({ms_per_tok:.1f} ms/token)")
            engine.unload()
        else:
            # Interactive generation
            print(f"\n[GEN] backend={'GPU' if engine.lib.ntmre_get_backend() == 1 else 'CPU'}")
            engine.reset()
            for t in tokens:
                engine.forward(t)

            generated = []
            for step in range(256):
                logits = engine.forward(tokens[-1] if tokens else 1)
                next_token = int(np.argmax(logits))
                if next_token == eos:
                    break
                generated.append(next_token)
                tokens.append(next_token)
                c = chr(next_token) if 32 <= next_token < 127 else f'[{next_token}]'
                print(c, end='', flush=True)

            print(f"\n[GEN] Generated {len(generated)} tokens")
            engine.unload()

    else:
        print("\n  NIGHTMARE ENGINE")
        print("  Usage:")
        print("    python nightmare_engine.py --compile                      (build DLL)")
        print("    python nightmare_engine.py model.ntmre \"Hello\"            (inference, auto backend)")
        print("    python nightmare_engine.py model.ntmre \"Hello\" --gpu     (force GPU)")
        print("    python nightmare_engine.py model.ntmre \"Hello\" --cpu     (force CPU)")
        print("    python nightmare_engine.py model.ntmre --bench --gpu     (benchmark GPU)")
        print("    python nightmare_engine.py model.ntmre --bench --cpu     (benchmark CPU)")
        print()
        if DLL_PATH.exists():
            print(f"  DLL: {DLL_PATH} ({DLL_PATH.stat().st_size / 1024:.0f} KB)")
            engine = NightmareEngine()
            if engine.available:
                print("  GPU: ONLINE")
                vram = engine.lib.ntmre_get_device_vram_mb()
                free = engine.lib.ntmre_get_device_free_mb()
                print(f"  VRAM: {free} MB free / {vram} MB total")
            else:
                print("  GPU: OFFLINE (CPU backend available)")
        else:
            print(f"  DLL not found. Run: python nightmare_engine.py --compile")
