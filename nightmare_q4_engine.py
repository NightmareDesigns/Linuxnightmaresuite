"""
NIGHTMARE Q4 GPU ENGINE — Zero Dependencies
=============================================
Pure Python + ctypes + struct. No numpy. No torch. No bullshit.
Loads GGUF, uploads Q4 weights to your 3060, runs inference on GPU.

Uses:
  - nightmare_engine.dll (your compiled CUDA kernels)
  - nightmare_matmul.dll (CPU fallback if needed)
  - struct module (GGUF parsing)
  - ctypes (DLL calls + memory buffers)
  - array module (typed arrays without numpy)

Author: Nightmare Designs
"""

import ctypes
import struct
import os
import sys
import time
import math
import array
from pathlib import Path
from ctypes import (
    c_float, c_int, c_uint8, c_size_t, c_void_p,
    POINTER, cast, byref, memmove, sizeof
)

# ============================================================
# DLL LOADING
# ============================================================

ENGINE_DIR = Path(__file__).parent
GPU_DLL = ENGINE_DIR / "nightmare_engine.dll"
CPU_DLL = ENGINE_DIR / "nightmare_matmul.dll"


def _load_gpu_dll():
    """Load the CUDA engine DLL with all capabilities."""
    if not GPU_DLL.exists():
        return None
    try:
        lib = ctypes.CDLL(str(GPU_DLL))
        
        # Select GPU with most VRAM before init
        try:
            lib.nightmare_set_device.restype = c_int
            lib.nightmare_set_device.argtypes = [c_int]
            lib.nightmare_get_device_count.restype = c_int
            lib.nightmare_get_vram_mb.restype = c_int
            lib.nightmare_get_vram_mb.argtypes = [c_int]
            n = lib.nightmare_get_device_count()
            if n > 1:
                vrams = [lib.nightmare_get_vram_mb(i) for i in range(n)]
                best = vrams.index(max(vrams))
                lib.nightmare_set_device(best)
                print(f"[ENGINE] Selected GPU {best} ({max(vrams)} MB VRAM)")
        except AttributeError:
            pass  # Older DLL without set_device
        
        # Init CUDA
        lib.nightmare_init.restype = c_int
        ret = lib.nightmare_init()
        if ret != 0:
            print("[ENGINE] CUDA init failed")
            return None
        
        # Q4 native path
        lib.nightmare_upload_q4.restype = c_void_p
        lib.nightmare_upload_q4.argtypes = [POINTER(c_uint8), c_int, c_int]
        lib.nightmare_q4_init_buffers.restype = c_int
        lib.nightmare_q4_init_buffers.argtypes = [c_int]
        lib.nightmare_q4_forward.restype = c_int
        lib.nightmare_q4_forward.argtypes = [POINTER(c_float), c_void_p, POINTER(c_float), c_int, c_int]
        lib.nightmare_free_q4.restype = None
        lib.nightmare_free_q4.argtypes = [c_void_p]
        lib.nightmare_q4_free_buffers.restype = None
        
        # Kernels
        lib.nightmare_rmsnorm.restype = c_int
        lib.nightmare_rmsnorm.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int, c_float]
        lib.nightmare_silu_mul.restype = c_int
        lib.nightmare_silu_mul.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), c_int]
        lib.nightmare_attention.restype = c_int
        lib.nightmare_attention.argtypes = [
            POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float),
            c_int, c_int, c_int, c_int
        ]
        
        lib.nightmare_get_device_count.restype = c_int
        lib.nightmare_get_vram_mb.restype = c_int
        lib.nightmare_get_vram_mb.argtypes = [c_int]
        
        # Persistent F32 weight upload (for Q8/Q6/Q5/F16 dequant -> GPU resident)
        try:
            lib.nightmare_upload_weight.restype = c_void_p
            lib.nightmare_upload_weight.argtypes = [POINTER(c_float), c_int, c_int]
            lib.nightmare_upload_any.restype = c_void_p
            lib.nightmare_upload_any.argtypes = [POINTER(c_uint8), c_int, c_int, c_int]
            lib.nightmare_matmul_persistent.restype = c_int
            lib.nightmare_matmul_persistent.argtypes = [POINTER(c_float), c_void_p, POINTER(c_float), c_int, c_int, c_int]
            lib.nightmare_free_weight.restype = None
            lib.nightmare_free_weight.argtypes = [c_void_p]
        except AttributeError:
            pass  # Older DLL without persistent support - Q4 only
        
        n_gpus = lib.nightmare_get_device_count()
        for i in range(n_gpus):
            vram = lib.nightmare_get_vram_mb(i)
            print(f"[ENGINE] GPU {i}: {vram} MB VRAM")
        print(f"[ENGINE] CUDA loaded. {n_gpus} GPU(s)")
        return lib
    except Exception as e:
        print(f"[ENGINE] GPU DLL failed: {e}")
        return None


# ============================================================
# CTYPES FLOAT ARRAY HELPERS (no numpy)
# ============================================================

def make_float_buf(size):
    """Create a ctypes float array."""
    return (c_float * size)()

def make_byte_buf(size):
    """Create a ctypes byte array."""
    return (c_uint8 * size)()

def float_ptr(buf):
    """Get a c_float pointer from a ctypes array."""
    return cast(buf, POINTER(c_float))

def byte_ptr(buf):
    """Get a c_uint8 pointer from a ctypes array."""
    return cast(buf, POINTER(c_uint8))


# ============================================================
# GGUF PARSER (pure struct, no numpy)
# ============================================================

def _read_string(f):
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')

def _skip_value(f, vtype):
    """Skip a GGUF KV value without fully parsing."""
    if vtype == 0: f.read(1)
    elif vtype == 1: f.read(1)
    elif vtype == 2: f.read(2)
    elif vtype == 3: f.read(2)
    elif vtype == 4: f.read(4)
    elif vtype == 5: f.read(4)
    elif vtype == 6: f.read(4)
    elif vtype == 7: f.read(1)
    elif vtype == 8: _read_string(f)
    elif vtype == 9:
        elem_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        for _ in range(arr_len):
            _skip_value(f, elem_type)
    elif vtype == 10: f.read(8)
    elif vtype == 11: f.read(8)
    elif vtype == 12: f.read(8)

def _read_kv_value(f, vtype):
    """Read a simple KV value."""
    if vtype == 4: return struct.unpack('<I', f.read(4))[0]
    elif vtype == 5: return struct.unpack('<i', f.read(4))[0]
    elif vtype == 8: return _read_string(f)
    elif vtype == 10: return struct.unpack('<Q', f.read(8))[0]
    elif vtype == 6: return struct.unpack('<f', f.read(4))[0]
    elif vtype == 9:
        # Array — skip it but note length
        elem_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        for _ in range(arr_len):
            _skip_value(f, elem_type)
        return f"[array:{arr_len}]"
    else:
        _skip_value(f, vtype)
        return None


def parse_gguf_header(filepath):
    """Parse GGUF file — returns config, tensor_infos, data_offset."""
    with open(filepath, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0x46554747:
            raise ValueError(f"Not a GGUF file: 0x{magic:08X}")
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        
        # Read metadata
        config = {}
        for _ in range(n_kv):
            key = _read_string(f)
            vtype = struct.unpack('<I', f.read(4))[0]
            value = _read_kv_value(f, vtype)
            if value is not None and not str(value).startswith('[array'):
                config[key] = value
        
        # Read tensor infos
        tensor_infos = {}
        for _ in range(n_tensors):
            name = _read_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            # GGUF stores dims reversed; reverse to get logical (out, in) shape.
            shape = tuple(reversed(dims))
            n_elements = 1
            for d in shape:
                n_elements *= d
            tensor_infos[name] = {
                'shape': shape,
                'dtype': dtype,
                'offset': offset,
                'n_elements': n_elements,
            }
        
        # Align to 32 bytes — tensor data starts here
        pos = f.tell()
        pad = (32 - pos % 32) % 32
        data_offset = pos + pad
    
    return config, tensor_infos, data_offset


# ============================================================
# Q4 DEQUANT (pure Python — for F32 norm weights and embedding)
# ============================================================

def dequant_q4_to_floats(raw_bytes, n_elements):
    """Dequantize Q4_0 raw bytes to a ctypes float array. For small tensors only."""
    buf = make_float_buf(n_elements)
    n_blocks = n_elements // 32
    idx = 0
    for b in range(n_blocks):
        block_start = b * 18
        # Scale is f16 (2 bytes)
        scale_bits = raw_bytes[block_start] | (raw_bytes[block_start + 1] << 8)
        # Convert f16 to f32
        sign = (scale_bits >> 15) & 1
        exp = (scale_bits >> 10) & 0x1F
        frac = scale_bits & 0x3FF
        if exp == 0:
            scale = ((-1)**sign) * (2**-14) * (frac / 1024.0)
        elif exp == 31:
            scale = float('inf') if frac == 0 else float('nan')
        else:
            scale = ((-1)**sign) * (2**(exp - 15)) * (1.0 + frac / 1024.0)
        
        # Unpack 16 bytes of nibbles (GGML sequential: lows first, highs second)
        for j in range(16):
            byte_val = raw_bytes[block_start + 2 + j]
            low = (byte_val & 0x0F)
            high = (byte_val >> 4)
            buf[idx + j] = (low - 8) * scale
            buf[idx + j + 16] = (high - 8) * scale
        idx += 32
    return buf


def read_f16_to_floats(raw_bytes, n_elements):
    """Convert F16 raw bytes to ctypes float array."""
    buf = make_float_buf(n_elements)
    for i in range(n_elements):
        bits = raw_bytes[i*2] | (raw_bytes[i*2 + 1] << 8)
        sign = (bits >> 15) & 1
        exp = (bits >> 10) & 0x1F
        frac = bits & 0x3FF
        if exp == 0:
            val = ((-1)**sign) * (2**-14) * (frac / 1024.0)
        elif exp == 31:
            val = float('inf') if frac == 0 else float('nan')
        else:
            val = ((-1)**sign) * (2**(exp - 15)) * (1.0 + frac / 1024.0)
        buf[i] = val
    return buf


def read_f32_to_floats(raw_bytes, n_elements):
    """Read F32 raw bytes into ctypes float array."""
    buf = make_float_buf(n_elements)
    memmove(buf, raw_bytes, n_elements * 4)
    return buf


def read_f16_row(raw_bytes, row_index, row_width):
    """Dequant a SINGLE row from F16 raw bytes. Fast — only 'row_width' elements."""
    buf = make_float_buf(row_width)
    offset = row_index * row_width * 2  # 2 bytes per f16
    for i in range(row_width):
        pos = offset + i * 2
        bits = raw_bytes[pos] | (raw_bytes[pos + 1] << 8)
        sign = (bits >> 15) & 1
        exp = (bits >> 10) & 0x1F
        frac = bits & 0x3FF
        if exp == 0:
            val = ((-1)**sign) * (2**-14) * (frac / 1024.0)
        elif exp == 31:
            val = 0.0  # inf/nan → 0 for safety
        else:
            val = ((-1)**sign) * (2**(exp - 15)) * (1.0 + frac / 1024.0)
        buf[i] = val
    return buf


def dequant_q4_row(raw_bytes, row_index, row_width):
    """Dequant a SINGLE row from Q4_0 raw bytes."""
    blocks_per_row = row_width // 32
    row_byte_size = blocks_per_row * 18
    offset = row_index * row_byte_size
    buf = make_float_buf(row_width)
    idx = 0
    for b in range(blocks_per_row):
        block_start = offset + b * 18
        scale_bits = raw_bytes[block_start] | (raw_bytes[block_start + 1] << 8)
        sign = (scale_bits >> 15) & 1
        exp = (scale_bits >> 10) & 0x1F
        frac = scale_bits & 0x3FF
        if exp == 0:
            scale = ((-1)**sign) * (2**-14) * (frac / 1024.0)
        elif exp == 31:
            scale = 0.0
        else:
            scale = ((-1)**sign) * (2**(exp - 15)) * (1.0 + frac / 1024.0)
        for j in range(16):
            byte_val = raw_bytes[block_start + 2 + j]
            low = (byte_val & 0x0F)
            high = (byte_val >> 4)
            # GGML sequential: lows at 0..15, highs at 16..31
            buf[idx + j] = (low - 8) * scale
            buf[idx + j + 16] = (high - 8) * scale
        idx += 32
    return buf


def dequant_q5_row(raw_bytes, row_index, row_width, dtype=6):
    """Dequant a SINGLE row from Q5_0 (dtype 6) or Q5_1 (dtype 7) raw bytes.

    Q5_0 block (32 weights, 22 bytes): [scale f16][high_bits u32][16 quants]
    Q5_1 block (32 weights, 24 bytes): [scale f16][min f16][high_bits u32][16 quants]
    """
    block_bytes = 22 if dtype == 6 else 24
    blocks_per_row = row_width // 32
    row_byte_size = blocks_per_row * block_bytes
    offset = row_index * row_byte_size
    buf = make_float_buf(row_width)
    idx = 0
    for b in range(blocks_per_row):
        bs = offset + b * block_bytes
        scale = _f16_to_f32(raw_bytes[bs] | (raw_bytes[bs + 1] << 8))
        if dtype == 6:
            mins = 0.0
            hb_off = bs + 2
        else:
            mins = _f16_to_f32(raw_bytes[bs + 2] | (raw_bytes[bs + 3] << 8))
            hb_off = bs + 4
        # 32 high bits, one per weight
        high_bits = int.from_bytes(raw_bytes[hb_off:hb_off + 4], 'little')
        q_off = hb_off + 4
        # low nibbles -> weights 0..15, high nibbles -> weights 16..31
        for j in range(16):
            byte_val = raw_bytes[q_off + j]
            lo = (byte_val & 0x0F) | (((high_bits >> j) & 1) << 4)
            hi = ((byte_val >> 4) & 0x0F) | (((high_bits >> (j + 16)) & 1) << 4)
            buf[idx + j] = scale * lo + mins
            buf[idx + j + 16] = scale * hi + mins
        idx += 32
    return buf


# ============================================================
# ROPE (pure Python — precompute cos/sin table)
# ============================================================

def precompute_rope(head_dim, max_seq_len, base=10000.0):
    """Precompute RoPE cos/sin tables. Returns (cos_table, sin_table) as ctypes arrays."""
    # cos_table[pos] and sin_table[pos] are each (head_dim//2) floats
    half = head_dim // 2
    cos_table = []
    sin_table = []
    for pos in range(max_seq_len):
        cos_row = make_float_buf(half)
        sin_row = make_float_buf(half)
        for i in range(half):
            freq = 1.0 / (base ** (2.0 * i / head_dim))
            angle = pos * freq
            cos_row[i] = math.cos(angle)
            sin_row[i] = math.sin(angle)
        cos_table.append(cos_row)
        sin_table.append(sin_row)
    return cos_table, sin_table


def apply_rope_inplace(vec, cos_pos, sin_pos, n_heads, head_dim):
    """Apply RoPE to a vector in-place. vec is (n_heads * head_dim) floats.
    Uses contiguous half-split (LLaMA/Mistral/Gemma standard)."""
    half = head_dim // 2
    for h in range(n_heads):
        base = h * head_dim
        for i in range(half):
            # Contiguous half-split: pair (x[i], x[i + half])
            x0 = vec[base + i]
            x1 = vec[base + i + half]
            c = cos_pos[i]
            s = sin_pos[i]
            vec[base + i] = x0 * c - x1 * s
            vec[base + i + half] = x0 * s + x1 * c


# ============================================================
# MAIN ENGINE
# ============================================================



# ============================================================
# DEQUANTIZATION HELPERS (for non-Q4 weights → F32)
# ============================================================

def _f16_to_f32(h):
    """Convert a 16-bit float (as uint16) to Python float."""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    frac = h & 0x3FF
    if exp == 0:
        return ((-1)**sign) * (2**-14) * (frac / 1024.0)
    elif exp == 31:
        return 0.0  # treat inf/nan as 0 for safety
    return ((-1)**sign) * (2**(exp - 15)) * (1.0 + frac / 1024.0)


def dequant_full_tensor(raw_bytes, n_elements, dtype):
    """Dequant raw GGUF bytes to a ctypes float array. No numpy."""
    import struct as _struct
    
    buf = make_float_buf(n_elements)
    
    if dtype == 0:  # F32
        memmove(buf, raw_bytes, n_elements * 4)
    
    elif dtype == 1:  # F16
        for i in range(n_elements):
            half_bytes = raw_bytes[i*2:i*2+2]
            # F16 to F32 conversion
            h = int.from_bytes(half_bytes, 'little')
            sign = (h >> 15) & 1
            exp = (h >> 10) & 0x1F
            frac = h & 0x3FF
            if exp == 0:
                val = ((-1)**sign) * (2**-14) * (frac / 1024.0)
            elif exp == 31:
                val = float('inf') if frac == 0 else float('nan')
            else:
                val = ((-1)**sign) * (2**(exp - 15)) * (1 + frac / 1024.0)
            _struct.pack_into('f', buf, i * 4, val)
    
    elif dtype == 8:  # Q8_0: 34 bytes per 32 elements
        block_bytes = 34
        n_blocks = n_elements // 32
        idx = 0
        for b in range(n_blocks):
            offset = b * block_bytes
            d_bytes = raw_bytes[offset:offset+2]
            d_h = int.from_bytes(d_bytes, 'little')
            # Convert f16 scale
            sign = (d_h >> 15) & 1
            exp = (d_h >> 10) & 0x1F
            frac = d_h & 0x3FF
            if exp == 0:
                d = ((-1)**sign) * (2**-14) * (frac / 1024.0)
            elif exp == 31:
                d = 0.0
            else:
                d = ((-1)**sign) * (2**(exp - 15)) * (1 + frac / 1024.0)
            for i in range(32):
                q = raw_bytes[offset + 2 + i]
                if q > 127:
                    q -= 256  # signed int8
                _struct.pack_into('f', buf, idx * 4, d * q)
                idx += 1
    
    elif dtype == 2:  # Q4_0: 18 bytes per 32 elements
        block_bytes = 18
        n_blocks = n_elements // 32
        idx = 0
        for b in range(n_blocks):
            offset = b * block_bytes
            d_bytes = raw_bytes[offset:offset+2]
            d_h = int.from_bytes(d_bytes, 'little')
            sign = (d_h >> 15) & 1
            exp = (d_h >> 10) & 0x1F
            frac = d_h & 0x3FF
            if exp == 0:
                d = ((-1)**sign) * (2**-14) * (frac / 1024.0)
            elif exp == 31:
                d = 0.0
            else:
                d = ((-1)**sign) * (2**(exp - 15)) * (1 + frac / 1024.0)
            for i in range(16):
                byte_val = raw_bytes[offset + 2 + i]
                lo = (byte_val & 0x0F) - 8
                hi = ((byte_val >> 4) & 0x0F) - 8
                _struct.pack_into('f', buf, idx * 4, d * lo)
                idx += 1
                _struct.pack_into('f', buf, idx * 4, d * hi)
                idx += 1

    elif dtype in (6, 7):  # Q5_0 / Q5_1
        block_bytes = 22 if dtype == 6 else 24
        n_blocks = n_elements // 32
        idx = 0
        for b in range(n_blocks):
            off = b * block_bytes
            d = _f16_to_f32(raw_bytes[off] | (raw_bytes[off + 1] << 8))
            if dtype == 6:
                m = 0.0
                hb_off = off + 2
            else:
                m = _f16_to_f32(raw_bytes[off + 2] | (raw_bytes[off + 3] << 8))
                hb_off = off + 4
            high_bits = int.from_bytes(raw_bytes[hb_off:hb_off + 4], 'little')
            q_off = hb_off + 4
            for j in range(16):
                byte_val = raw_bytes[q_off + j]
                lo = (byte_val & 0x0F) | (((high_bits >> j) & 1) << 4)
                hi = ((byte_val >> 4) & 0x0F) | (((high_bits >> (j + 16)) & 1) << 4)
                _struct.pack_into('f', buf, idx * 4, d * lo + m)
                idx += 1
                _struct.pack_into('f', buf, idx * 4, d * hi + m)
                idx += 1

    else:
        # Q4_K (12), Q5_K (13), Q6_K (14), Q4_1 (3) — handwritten dequant
        import struct as _s
        
        if dtype == 3:
            # Q4_1: 20 bytes per 32 elements [f16 d][f16 m][16 bytes quants]
            n_blocks = n_elements // 32
            idx = 0
            for b in range(n_blocks):
                off = b * 20
                d = _f16_to_f32(raw_bytes[off] | (raw_bytes[off+1] << 8))
                m = _f16_to_f32(raw_bytes[off+2] | (raw_bytes[off+3] << 8))
                for i in range(16):
                    byte_val = raw_bytes[off + 4 + i]
                    buf[idx] = d * (byte_val & 0x0F) + m; idx += 1
                    buf[idx] = d * (byte_val >> 4) + m; idx += 1
        
        elif dtype == 12:
            # Q4_K: 256 elements per block, 144 bytes
            # [f16 d][f16 dmin][12 bytes scales][128 bytes quants]
            n_blocks = n_elements // 256
            idx = 0
            for b in range(n_blocks):
                off = b * 144
                d = _f16_to_f32(raw_bytes[off] | (raw_bytes[off+1] << 8))
                dmin = _f16_to_f32(raw_bytes[off+2] | (raw_bytes[off+3] << 8))
                sc_raw = raw_bytes[off+4:off+16]
                qs = raw_bytes[off+16:off+144]
                
                # Decode 8 sub-block scales and mins (matches llama.cpp get_scale_min_k4)
                scales = [0]*8
                mins = [0]*8
                for j in range(4):
                    scales[j] = sc_raw[j] & 0x3F
                    mins[j] = sc_raw[j+4] & 0x3F
                for j in range(4):
                    scales[4+j] = (sc_raw[j+8] & 0x0F) | ((sc_raw[j] >> 6) << 4)
                    mins[4+j] = (sc_raw[j+8] >> 4) | ((sc_raw[j+4] >> 6) << 4)
                
                # Process 4 pairs of sub-blocks (32 bytes each → 64 values)
                for pair in range(4):
                    sb_lo = pair * 2
                    sb_hi = pair * 2 + 1
                    sc_lo = d * scales[sb_lo]
                    mn_lo = dmin * mins[sb_lo]
                    sc_hi = d * scales[sb_hi]
                    mn_hi = dmin * mins[sb_hi]
                    for k in range(32):
                        byte_val = qs[pair * 32 + k]
                        buf[idx + sb_lo*32 + k] = sc_lo * (byte_val & 0x0F) - mn_lo
                        buf[idx + sb_hi*32 + k] = sc_hi * ((byte_val >> 4) & 0x0F) - mn_hi
                idx += 256
        
        elif dtype == 13:
            # Q5_K: 256 elements per block, 176 bytes
            # [f16 d][f16 dmin][12 bytes scales][32 bytes qh][128 bytes qs]
            n_blocks = n_elements // 256
            idx = 0
            for b in range(n_blocks):
                off = b * 176
                d = _f16_to_f32(raw_bytes[off] | (raw_bytes[off+1] << 8))
                dmin = _f16_to_f32(raw_bytes[off+2] | (raw_bytes[off+3] << 8))
                sc_raw = raw_bytes[off+4:off+16]
                qh = raw_bytes[off+16:off+48]
                qs = raw_bytes[off+48:off+176]
                
                # Decode scales and mins (matches llama.cpp get_scale_min_k4)
                scales = [0]*8
                mins = [0]*8
                for j in range(4):
                    scales[j] = sc_raw[j] & 0x3F
                    mins[j] = sc_raw[j+4] & 0x3F
                for j in range(4):
                    scales[4+j] = (sc_raw[j+8] & 0x0F) | ((sc_raw[j] >> 6) << 4)
                    mins[4+j] = (sc_raw[j+8] >> 4) | ((sc_raw[j+4] >> 6) << 4)
                
                # Process 4 pairs of sub-blocks
                for pair in range(4):
                    sb_lo = pair * 2
                    sb_hi = pair * 2 + 1
                    sc_lo = d * scales[sb_lo]
                    mn_lo = dmin * mins[sb_lo]
                    sc_hi = d * scales[sb_hi]
                    mn_hi = dmin * mins[sb_hi]
                    
                    for k in range(32):
                        # Low sub-block: low nibble + high bit
                        qi = pair * 32 + k
                        q_low = qs[qi] & 0x0F
                        bit_idx = sb_lo * 32 + k
                        qh_bit = (qh[bit_idx // 8] >> (bit_idx % 8)) & 1
                        q = q_low | (qh_bit << 4)
                        buf[idx] = sc_lo * q - mn_lo
                        idx += 1
                    
                    for k in range(32):
                        # High sub-block: high nibble + high bit
                        qi = pair * 32 + k
                        q_high = qs[qi] >> 4
                        bit_idx = sb_hi * 32 + k
                        qh_bit = (qh[bit_idx // 8] >> (bit_idx % 8)) & 1
                        q = q_high | (qh_bit << 4)
                        buf[idx] = sc_hi * q - mn_hi
                        idx += 1
        
        elif dtype == 14:
            # Q6_K: 256 elements per block, 210 bytes
            # [128 bytes ql][64 bytes qh][16 bytes scales (int8)][f16 d]
            n_blocks = n_elements // 256
            idx = 0
            for b in range(n_blocks):
                off = b * 210
                ql = raw_bytes[off:off+128]
                qh = raw_bytes[off+128:off+192]
                scales_raw = raw_bytes[off+192:off+208]
                d = _f16_to_f32(raw_bytes[off+208] | (raw_bytes[off+209] << 8))
                
                for j in range(16):
                    sc = scales_raw[j]
                    if sc > 127: sc -= 256
                    sc_f = d * sc
                    for k in range(16):
                        elem = j * 16 + k
                        # Low 4 bits from ql
                        ql_byte = elem // 2
                        if elem % 2 == 0:
                            q_low = ql[ql_byte] & 0x0F
                        else:
                            q_low = ql[ql_byte] >> 4
                        # High 2 bits from qh
                        qh_byte = elem // 4
                        qh_shift = (elem % 4) * 2
                        q_high = (qh[qh_byte] >> qh_shift) & 0x03
                        q = (q_low | (q_high << 4)) - 32
                        buf[idx] = sc_f * q
                        idx += 1
        
        else:
            print(f"[DEQUANT] WARNING: Unsupported dtype {dtype} — zeros!")
    
    return buf

class NightmareQ4Engine:
    """
    Full GPU inference engine. Zero dependencies.
    Loads GGUF, uploads Q4 to VRAM, runs forward on your cards.
    """
    
    def __init__(self):
        self.lib = _load_gpu_dll()
        self.available = self.lib is not None
        self.weights_gpu = {}   # tensor_name -> device_ptr (void*)
        self.weights_cpu = {}   # tensor_name -> ctypes float array (norms, embedding)
        self.config = {}
        self._emb_raw = None    # raw embedding bytes (read one row at a time)
        self._emb_dtype = 0     # embedding quant type
        self.kv_k = []          # per-layer K cache (list of float arrays)
        self.kv_v = []          # per-layer V cache
        self.rope_cos = []
        self.rope_sin = []
        
        # Reusable buffers (allocated once)
        self._x = None
        self._out = None
    
    def load(self, filepath):
        """Load any GGUF model. Auto-detects architecture via HuggingFace model_config."""
        if not self.available:
            print("[ENGINE] No GPU available!")
            return False

        print(f"[ENGINE] Loading: {filepath}")
        t0 = time.time()

        config, tensor_infos, data_offset = parse_gguf_header(filepath)
        self._tensor_list = [{'name': k, **v} for k, v in tensor_infos.items()]

        # Auto-detect architecture via model_config (HuggingFace-aware)
        try:
            from model_config import config_from_gguf
            mcfg = config_from_gguf(filepath)
            arch = mcfg.architecture
            dim = mcfg.dim
            n_layers = mcfg.n_layers
            n_heads = mcfg.n_heads
            hidden_dim = mcfg.ffn_dim
            vocab_size = mcfg.vocab_size
            max_seq = min(mcfg.max_seq_len, 4096)
            rope_base = mcfg.rope_theta
            rms_eps = mcfg.norm_eps
            head_dim = mcfg.head_dim
            n_kv_heads = mcfg.n_kv_heads
            self._logit_cap = mcfg.logit_softcap or 0.0
            self._embedding_scale = mcfg.embedding_scale or 0.0
            self._chat_template = mcfg.chat_template
            print(f"[ENGINE] Architecture: {arch} (via model_config)")
            print(f"[ENGINE] rope_theta={rope_base}, norm_eps={rms_eps}, chat={self._chat_template}")
        except Exception as _e:
            # Fallback to manual GGUF key parsing
            arch = config.get('general.architecture', 'llama')
            p = arch
            print(f"[ENGINE] Architecture: {arch} (fallback: {_e})")
            dim = config.get(f'{p}.embedding_length', config.get('llama.embedding_length', 4096))
            n_layers = config.get(f'{p}.block_count', config.get('llama.block_count', 32))
            n_heads = config.get(f'{p}.attention.head_count', config.get('llama.attention.head_count', 32))
            hidden_dim = config.get(f'{p}.feed_forward_length', config.get('llama.feed_forward_length', 11008))
            vocab_size = config.get(f'{p}.vocab_size', config.get('llama.vocab_size', 32000))
            max_seq = min(config.get(f'{p}.context_length', config.get('llama.context_length', 4096)), 4096)
            rope_base = config.get(f'{p}.rope.freq_base', 10000.0)
            rms_eps = config.get(f'{p}.attention.layer_norm_rms_epsilon', 1e-5)
            n_kv_heads = config.get(f'{p}.attention.head_count_kv', config.get('llama.attention.head_count_kv', n_heads))
            head_dim = dim // n_heads
            self._logit_cap = config.get(f'{p}.final_logit_softcapping', 0.0)
            self._embedding_scale = 0.0
            self._chat_template = ""

        # Override vocab from tensor shape if metadata wrong
        if 'output.weight' in tensor_infos:
            oshape = tensor_infos['output.weight']['shape']
            if len(oshape) > 1:
                # Internal shape convention is (in_features, out_features).
                # Vocab is out_features for output projection.
                vocab_size = int(oshape[1])
            else:
                vocab_size = int(oshape[0])
        elif 'token_embd.weight' in tensor_infos:
            vocab_size = tensor_infos['token_embd.weight']['shape'][0]

        # Override head_dim from Q weight shape if available
        q_key = 'blk.0.attn_q.weight'
        if q_key in tensor_infos:
            q_shape = tensor_infos[q_key]['shape']
            q_out_dim = q_shape[1] if len(q_shape) > 1 else dim
            head_dim = q_out_dim // n_heads

        # Normalize FFN dims (some models expose per-layer FFN as a list)
        if isinstance(hidden_dim, (list, tuple)):
            ffn_dims = [int(x) for x in hidden_dim]
            if len(ffn_dims) < n_layers:
                ffn_dims.extend([ffn_dims[-1]] * (n_layers - len(ffn_dims)))
            elif len(ffn_dims) > n_layers:
                ffn_dims = ffn_dims[:n_layers]
        else:
            h = int(hidden_dim) if isinstance(hidden_dim, (int, float)) else 11008
            ffn_dims = [h] * n_layers
        hidden_dim = max(ffn_dims) if ffn_dims else 11008

        # KV heads per-layer (Gemma 4 has variable)
        p = config.get('general.architecture', 'llama')
        kv_raw = config.get(f'{p}.attention.head_count_kv', config.get('llama.attention.head_count_kv', n_heads))
        if isinstance(kv_raw, list):
            self._kv_heads_per_layer = [int(x) for x in kv_raw]
            n_kv_heads = max(self._kv_heads_per_layer)
        else:
            n_kv_heads = int(kv_raw) if isinstance(kv_raw, (int, float)) else n_kv_heads
            self._kv_heads_per_layer = [n_kv_heads] * n_layers

        self.config = {
            'dim': dim, 'n_layers': n_layers, 'n_heads': n_heads,
            'n_kv_heads': n_kv_heads, 'head_dim': head_dim,
            'hidden_dim': hidden_dim, 'hidden_dims': ffn_dims, 'vocab_size': vocab_size,
            'max_seq_len': max_seq, 'arch': arch, 'rms_eps': rms_eps,
            'rope_base': rope_base, 'chat_template': getattr(self, '_chat_template', ''),
        }

        print(f"[ENGINE] dim={dim} layers={n_layers} heads={n_heads} kv={n_kv_heads} "
              f"head_dim={head_dim} ffn={hidden_dim} vocab={vocab_size}")

        # Per-layer attention geometry (Gemma4 may vary by layer)
        per_layer_q_heads = []
        per_layer_head_dims = []
        per_layer_kv_heads = []
        for layer in range(n_layers):
            q_key = f"blk.{layer}.attn_q.weight"
            k_key = f"blk.{layer}.attn_k.weight"
            kv_l = self._kv_heads_per_layer[layer] if layer < len(self._kv_heads_per_layer) else n_kv_heads
            if q_key not in tensor_infos or k_key not in tensor_infos or kv_l <= 0:
                per_layer_q_heads.append(int(n_heads))
                per_layer_head_dims.append(int(head_dim))
                per_layer_kv_heads.append(int(max(1, kv_l)))
                continue
            q_shape = tensor_infos[q_key]['shape']
            k_shape = tensor_infos[k_key]['shape']
            # shape is logical (out, in): shape[0] = out dim
            q_out = int(q_shape[0]) if len(q_shape) > 1 else dim
            k_out = int(k_shape[0]) if len(k_shape) > 1 else dim
            if k_out % kv_l != 0:
                per_layer_q_heads.append(int(n_heads))
                per_layer_head_dims.append(int(head_dim))
                per_layer_kv_heads.append(int(max(1, kv_l)))
                continue
            hd_l = k_out // kv_l
            if hd_l <= 0 or q_out % hd_l != 0:
                per_layer_q_heads.append(int(n_heads))
                per_layer_head_dims.append(int(head_dim))
                per_layer_kv_heads.append(int(max(1, kv_l)))
                continue
            per_layer_head_dims.append(hd_l)
            per_layer_q_heads.append(q_out // hd_l)
            per_layer_kv_heads.append(int(kv_l))

        self._head_dim_per_layer = per_layer_head_dims if per_layer_head_dims else [int(head_dim)] * n_layers
        self._q_heads_per_layer = per_layer_q_heads if per_layer_q_heads else [int(n_heads)] * n_layers
        self._kv_heads_per_layer = per_layer_kv_heads if per_layer_kv_heads else [int(n_kv_heads)] * n_layers
        self._kv_dim_per_layer = [
            self._kv_heads_per_layer[i] * self._head_dim_per_layer[i] for i in range(n_layers)
        ]

        if len(set(self._head_dim_per_layer)) > 1 or len(set(self._q_heads_per_layer)) > 1:
            print(f"[GPU] Variable per-layer attention geometry detected — enabling adaptive GPU path "
                  f"(q_heads {min(self._q_heads_per_layer)}..{max(self._q_heads_per_layer)}, "
                  f"head_dim {min(self._head_dim_per_layer)}..{max(self._head_dim_per_layer)})")

        # Init GPU buffers
        max_q_dim = max(self._q_heads_per_layer[i] * self._head_dim_per_layer[i] for i in range(n_layers))
        max_kv_dim = max(self._kv_dim_per_layer) if self._kv_dim_per_layer else (n_kv_heads * head_dim)
        max_head_dim = max(self._head_dim_per_layer) if self._head_dim_per_layer else head_dim
        # Q4 kernel scratch buffers only need attention/FFN dims, not full vocab.
        runtime_dim = max(dim, hidden_dim, max_q_dim, max_kv_dim)
        _safe_max_dim = 65536
        self._gpu_safe_q4_dim = _safe_max_dim
        if runtime_dim > _safe_max_dim:
            print(f"[GPU] Disabled for this model: runtime_dim={runtime_dim} exceeds safe DLL limit {_safe_max_dim} — using CPU path")
            return False
        ret = self.lib.nightmare_q4_init_buffers(runtime_dim)
        self._gpu_max_dim = runtime_dim  # Track for bounds checking in _gpu_matmul
        if ret != 0:
            print("[ENGINE] Failed to init GPU buffers!")
            return False
        
        # Init persistent norm/silu buffers (fixes illegal memory access!)
        try:
            ret2 = self.lib.nightmare_init_norm_buffers(runtime_dim)
            if ret2 != 0:
                print("[ENGINE] WARNING: Norm buffer init failed")
        except (AttributeError, OSError):
            print("[ENGINE] WARNING: norm buffers not in DLL — needs recompile")

        # Allocate CPU-side reusable buffers
        self._x = make_float_buf(runtime_dim)
        self._out = make_float_buf(runtime_dim)
        self._q_buf = make_float_buf(max_q_dim)
        self._k_buf = make_float_buf(max_kv_dim)
        self._v_buf = make_float_buf(max_kv_dim)
        self._attn_out = make_float_buf(max_q_dim)
        self._ffn_buf = make_float_buf(hidden_dim)
        self._ffn_buf2 = make_float_buf(hidden_dim)

        # Block size table: (elements_per_block, bytes_per_block)
        BLOCK_INFO = {
            0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20),
            6: (32, 22), 7: (32, 24),
            8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210), 30: (1, 2),
        }

        # Load tensors from file
        uploaded_gpu = 0
        loaded_cpu = 0
        total_gpu_bytes = 0

        with open(filepath, 'rb') as f:
            for tname, info in tensor_infos.items():
                dtype = info['dtype']
                n_el = info['n_elements']
                shape = info['shape']

                if dtype not in BLOCK_INFO:
                    continue

                el_per_block, bytes_per_block = BLOCK_INFO[dtype]
                n_blocks = n_el // el_per_block
                raw_size = n_blocks * bytes_per_block

                # Read raw bytes
                f.seek(data_offset + info['offset'])
                raw = f.read(raw_size)

                is_norm = 'norm' in tname
                is_emb = 'emb' in tname
                is_bias = 'bias' in tname
                key = tname.replace('.weight', '')

                if is_norm:
                    # Norms stay CPU (tiny)
                    if dtype == 0:
                        self.weights_cpu[key] = read_f32_to_floats(raw, n_el)
                    elif dtype == 1:
                        self.weights_cpu[key] = read_f16_to_floats(raw, n_el)
                    else:
                        self.weights_cpu[key] = read_f32_to_floats(raw, n_el)
                    loaded_cpu += 1

                elif is_bias:
                    # Biases stay CPU (tiny 1D vectors, added after GPU matmul)
                    if dtype == 0:
                        self.weights_cpu[key] = read_f32_to_floats(raw, n_el)
                    elif dtype == 1:
                        self.weights_cpu[key] = read_f16_to_floats(raw, n_el)
                    else:
                        self.weights_cpu[key] = dequant_full_tensor(raw, n_el, dtype)
                    loaded_cpu += 1

                elif is_emb:
                    # Embedding — store raw, dequant one row per token
                    self._emb_raw = bytearray(raw)
                    self._emb_dtype = dtype
                    self._emb_dim = shape[1] if len(shape) > 1 else shape[0]
                    if dtype == 1:
                        self._emb_rows = len(raw) // (self._emb_dim * 2)
                    elif dtype == 0:
                        self._emb_rows = len(raw) // (self._emb_dim * 4)
                    else:
                        self._emb_rows = n_el // self._emb_dim
                    print(f"  [EMB] {len(raw)/1e6:.1f} MB, dtype={dtype}, dim={self._emb_dim}, rows={self._emb_rows}")
                    loaded_cpu += 1

                    # If no separate output.weight exists, store embedding as output projection too
                    # (tied embeddings — token_embd IS the output projection)
                    if 'output.weight' not in [t['name'] for t in self._tensor_list]:
                        # Dequant full embedding for output matmul
                        emb_f32 = dequant_full_tensor(raw, n_el, dtype)
                        self.weights_cpu["output"] = emb_f32
                        self.weights_cpu["output__shape"] = (self._emb_rows, self._emb_dim)
                        print(f"  [OUTPUT] Tied to embedding ({self._emb_rows}x{self._emb_dim})")


                elif dtype == 2:
                    # Q4_0 — native GPU upload (fastest path)
                    # shape is logical (out, in): n_in = in = shape[1], n_out = out = shape[0]
                    n_in = shape[1] if len(shape) > 1 else n_el
                    n_out = shape[0] if len(shape) > 1 else 1
                    if n_in > self._gpu_max_dim or n_out > self._gpu_max_dim:
                        # Keep GPU enabled for the rest of the model; run oversized
                        # projections (usually giant vocab outputs) on CPU fallback.
                        self.weights_cpu[key] = dequant_full_tensor(raw, n_el, dtype)
                        self.weights_cpu[f"{key}__shape"] = (n_out, n_in)
                        loaded_cpu += 1
                        print(f"[GPU] Q4 fallback to CPU for {key} ({n_in}x{n_out}) > q4_limit={self._gpu_max_dim}")
                    else:
                        raw_buf = make_byte_buf(raw_size)
                        memmove(raw_buf, raw, raw_size)
                        device_ptr = self.lib.nightmare_upload_q4(byte_ptr(raw_buf), c_int(n_in), c_int(n_out))
                        if device_ptr:
                            self.weights_gpu[key] = (device_ptr, n_in, n_out)
                            uploaded_gpu += 1
                            total_gpu_bytes += raw_size
                        else:
                            loaded_cpu += 1

                elif dtype in (6, 7, 8, 12, 13, 14, 1):
                    # Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, F16 — GPU dequant via DLL
                    # shape is logical (out, in): n_in = in = shape[1], n_out = out = shape[0]
                    n_in = shape[1] if len(shape) > 1 else n_el
                    n_out = shape[0] if len(shape) > 1 else 1
                    has_upload_any = hasattr(self.lib, 'nightmare_upload_any')
                    if has_upload_any:
                        raw_buf = make_byte_buf(raw_size)
                        memmove(raw_buf, raw, raw_size)
                        device_ptr = self.lib.nightmare_upload_any(
                            byte_ptr(raw_buf), c_int(n_el), c_int(dtype), c_int(raw_size))
                        if device_ptr:
                            self.weights_gpu[key] = (device_ptr, n_in, n_out, 'f32')
                            uploaded_gpu += 1
                            total_gpu_bytes += n_el * 4
                        else:
                            # GPU full — fall back to CPU dequant
                            self.weights_cpu[key] = dequant_full_tensor(raw, n_el, dtype)
                            self.weights_cpu[f"{key}__shape"] = (n_out, n_in)
                            loaded_cpu += 1
                    else:
                        # Old DLL — Python dequant + persistent upload
                        f32_buf = dequant_full_tensor(raw, n_el, dtype)
                        has_persistent = hasattr(self.lib, 'nightmare_upload_weight')
                        if has_persistent:
                            # Internal shape convention is (n_in, n_out).
                            # Persistent kernel expects B as (n_in, n_out) row-major.
                            # Raw GGUF tensor bytes are effectively (n_out, n_in) in memory,
                            # so transpose before upload to preserve x @ W semantics.
                            if n_out > 1 and n_in > 1:
                                # In-place transpose in the ctypes buffer
                                transposed = make_float_buf(n_el)
                                for row in range(n_out):
                                    for col in range(n_in):
                                        transposed[col * n_out + row] = f32_buf[row * n_in + col]
                                f32_buf = transposed
                            device_ptr = self.lib.nightmare_upload_weight(
                                float_ptr(f32_buf), c_int(n_in), c_int(n_out))
                            if device_ptr:
                                self.weights_gpu[key] = (device_ptr, n_in, n_out, 'f32')
                                uploaded_gpu += 1
                                total_gpu_bytes += n_el * 4
                            else:
                                self.weights_cpu[key] = f32_buf
                                self.weights_cpu[f"{key}__shape"] = (n_out, n_in)
                                loaded_cpu += 1
                        else:
                            self.weights_cpu[key] = f32_buf
                            self.weights_cpu[f"{key}__shape"] = (n_out, n_in)
                            loaded_cpu += 1

                elif dtype == 0:
                    # F32 — upload persistently
                    # shape is logical (out, in): n_in = in = shape[1], n_out = out = shape[0]
                    n_in = shape[1] if len(shape) > 1 else n_el
                    n_out = shape[0] if len(shape) > 1 else 1
                    f32_buf = dequant_full_tensor(raw, n_el, dtype)
                    has_persistent = hasattr(self.lib, 'nightmare_upload_weight')
                    if has_persistent:
                        # Transpose (n_out, n_in) → (n_in, n_out) for kernel
                        if n_out > 1 and n_in > 1:
                            transposed = make_float_buf(n_el)
                            for row in range(n_out):
                                for col in range(n_in):
                                    transposed[col * n_out + row] = f32_buf[row * n_in + col]
                            f32_buf = transposed
                        device_ptr = self.lib.nightmare_upload_weight(
                            float_ptr(f32_buf), c_int(n_in), c_int(n_out))
                        if device_ptr:
                            self.weights_gpu[key] = (device_ptr, n_in, n_out, 'f32')
                            uploaded_gpu += 1
                            total_gpu_bytes += n_el * 4
                        else:
                            self.weights_cpu[key] = f32_buf
                            self.weights_cpu[f"{key}__shape"] = (n_out, n_in)
                            loaded_cpu += 1
                    else:
                        self.weights_cpu[key] = f32_buf
                        self.weights_cpu[f"{key}__shape"] = (n_out, n_in)
                        loaded_cpu += 1

                if (uploaded_gpu + loaded_cpu) % 50 == 0:
                    print(f"  [{uploaded_gpu + loaded_cpu}/{len(tensor_infos)}] loaded...")

        # Precompute RoPE for max head dim; smaller layers use prefix dims
        self.rope_cos, self.rope_sin = precompute_rope(max_head_dim, max_seq, rope_base)

        # Allocate per-layer KV cache
        for layer in range(n_layers):
            kv_size = self._kv_dim_per_layer[layer] * max_seq
            self.kv_k.append(make_float_buf(kv_size))
            self.kv_v.append(make_float_buf(kv_size))

        elapsed = time.time() - t0
        print(f"[ENGINE] Loaded in {elapsed:.1f}s")
        print(f"[ENGINE] GPU: {uploaded_gpu} tensors ({total_gpu_bytes/1e9:.2f} GB)")
        print(f"[ENGINE] CPU: {loaded_cpu} tensors")

        # Handle tied embeddings: if no 'output' weight was loaded, use embedding
        if 'output' not in self.weights_gpu and 'output' not in self.weights_cpu:
            if hasattr(self, '_emb_raw') and self._emb_raw:
                from dequant_fast import dequantize_tensor
                n_el = self._emb_rows * self._emb_dim
                emb_f32 = dequantize_tensor(bytes(self._emb_raw), self._emb_dtype, n_el)
                self.weights_cpu["output"] = emb_f32
                self.weights_cpu["output__shape"] = (self._emb_rows, self._emb_dim)
                print(f"[ENGINE] Output tied to embedding ({self._emb_rows}x{self._emb_dim})")

        print(f"[ENGINE] Ready!")
        return True
    def reset(self):
        """Clear KV cache for new generation."""
        for i in range(len(self.kv_k)):
            ctypes.memset(self.kv_k[i], 0, sizeof(self.kv_k[i]))
            ctypes.memset(self.kv_v[i], 0, sizeof(self.kv_v[i]))
    
    def _gpu_matmul(self, x_buf, weight_key, out_buf, out_size=0):
        """Matmul via GPU — handles Q4 native and F32 persistent paths.
        
        Returns True on success, False on failure.
        On failure, out_buf is ZEROED to prevent gibberish propagation.
        """
        if weight_key not in self.weights_gpu:
            if weight_key in self.weights_cpu:
                # CPU fallback — dequanted weights in Python float array
                cpu_w = self.weights_cpu[weight_key]
                shape_key = f"{weight_key}__shape"
                if shape_key in self.weights_cpu:
                    n_out, n_in = self.weights_cpu[shape_key]
                else:
                    # Guess: weight is flat, use out_size if given
                    n_in = self.config.get('dim', 4096)
                    n_out = len(cpu_w) // n_in if n_in > 0 else 0
                if n_out <= 0 or n_in <= 0:
                    print(f"[ENGINE WARNING] CPU fallback failed for {weight_key}: bad dims ({n_in}x{n_out})")
                    if out_size > 0:
                        ctypes.memset(out_buf, 0, out_size * sizeof(c_float))
                    return False
                # Manual matmul: out[i] = sum(x[j] * w[i*n_in + j])
                for i in range(n_out):
                    val = 0.0
                    for j in range(n_in):
                        val += x_buf[j] * cpu_w[i * n_in + j]
                    out_buf[i] = val
                return True
            print(f"[ENGINE WARNING] Weight '{weight_key}' not found on GPU or CPU!")
            if out_size > 0:
                ctypes.memset(out_buf, 0, out_size * sizeof(c_float))
            return False
        
        entry = self.weights_gpu[weight_key]
        if len(entry) == 4:
            # F32 persistent path (dequanted Q8/Q5/Q6/F16 or raw F32)
            device_ptr, n_in, n_out, _ = entry
            if n_in <= 0 or n_out <= 0:
                print(f"[ENGINE WARNING] Bad dims for {weight_key}: n_in={n_in}, n_out={n_out}")
                if out_size > 0:
                    ctypes.memset(out_buf, 0, out_size * sizeof(c_float))
                return False
            if hasattr(self.lib, 'nightmare_matmul_persistent'):
                # Weight stored TRANSPOSED on GPU as (n_in, n_out).
                # Kernel: C[1, n_out] = A[1, n_in] @ B[n_in, n_out]
                # A=x (host), B=weight (device), C=out (host)
                ret = self.lib.nightmare_matmul_persistent(
                    float_ptr(x_buf), c_void_p(device_ptr), float_ptr(out_buf),
                    c_int(1), c_int(n_in), c_int(n_out))
                if ret != 0:
                    print(f"[ENGINE ERROR] nightmare_matmul_persistent FAILED for {weight_key} ({n_in}x{n_out})")
                    if out_size > 0:
                        ctypes.memset(out_buf, 0, out_size * sizeof(c_float))
                return ret == 0
            return False
        else:
            # Q4 native path
            device_ptr, n_in, n_out = entry
            # Bounds check against pre-allocated GPU buffers
            max_dim = getattr(self, '_gpu_max_dim', 0)
            if max_dim > 0 and (n_in > max_dim or n_out > max_dim):
                print(f"[ENGINE ERROR] Q4 forward WOULD CRASH: {weight_key} n_in={n_in} n_out={n_out} > max_dim={max_dim}")
                if out_size > 0:
                    ctypes.memset(out_buf, 0, out_size * sizeof(c_float))
                return False
            ret = self.lib.nightmare_q4_forward(
                float_ptr(x_buf), device_ptr, float_ptr(out_buf), c_int(n_in), c_int(n_out))
            if ret != 0:
                print(f"[ENGINE ERROR] nightmare_q4_forward FAILED for {weight_key} ({n_in}x{n_out})")
                if out_size > 0:
                    ctypes.memset(out_buf, 0, out_size * sizeof(c_float))
            return ret == 0

    def _rmsnorm(self, x_buf, norm_key, out_buf, dim):
        """RMSNorm on GPU."""
        norm_w = self.weights_cpu.get(norm_key)
        if norm_w is None:
            # Copy through
            memmove(out_buf, x_buf, dim * sizeof(c_float))
            return
        eps = float(self.config.get('rms_eps', 1e-5))
        self.lib.nightmare_rmsnorm(
            float_ptr(x_buf), float_ptr(norm_w), float_ptr(out_buf),
            c_int(dim), c_float(eps)
        )
    
    def forward(self, token_id, position):
        """Forward pass for one token. Returns logits as ctypes float array or None on fatal error."""
        dim = self.config['dim']
        n_heads = self.config['n_heads']
        n_kv_heads = self.config['n_kv_heads']
        head_dim = self.config['head_dim']
        hidden_dim = self.config['hidden_dim']
        hidden_dims = self.config.get('hidden_dims')
        hidden_dims = self.config.get('hidden_dims')
        n_layers = self.config['n_layers']
        
        # 1. Embedding lookup
        emb_dim = self._emb_dim
        if token_id >= self._emb_rows:
            token_id = 0  # Clamp to pad — vocab padded beyond embedding size
        if self._emb_dtype == 1:  # F16
            row = read_f16_row(self._emb_raw, token_id, emb_dim)
        elif self._emb_dtype == 2:  # Q4
            row = dequant_q4_row(self._emb_raw, token_id, emb_dim)
        elif self._emb_dtype in (6, 7):  # Q5_0 / Q5_1
            row = dequant_q5_row(self._emb_raw, token_id, emb_dim, self._emb_dtype)
        else:  # F32
            row = make_float_buf(emb_dim)
            offset = token_id * emb_dim * 4
            memmove(row, bytes(self._emb_raw[offset:offset + emb_dim*4]), emb_dim * 4)
        
        # Copy row into x buffer
        memmove(self._x, row, emb_dim * sizeof(c_float))
        if getattr(self, '_embedding_scale', 0.0):
            escale = float(self._embedding_scale)
            for i in range(dim):
                self._x[i] *= escale
        
        # 2. Transformer layers
        norm_buf = make_float_buf(dim)
        
        for layer in range(n_layers):
            prefix = f"blk.{layer}"
            n_heads_l = self._q_heads_per_layer[layer] if hasattr(self, '_q_heads_per_layer') else n_heads
            n_kv_heads_l = self._kv_heads_per_layer[layer] if hasattr(self, '_kv_heads_per_layer') else n_kv_heads
            head_dim_l = self._head_dim_per_layer[layer] if hasattr(self, '_head_dim_per_layer') else head_dim
            kv_dim_l = self._kv_dim_per_layer[layer] if hasattr(self, '_kv_dim_per_layer') else (n_kv_heads_l * head_dim_l)
            q_dim_l = n_heads_l * head_dim_l
            
            # Pre-attention norm
            self._rmsnorm(self._x, f"{prefix}.attn_norm", norm_buf, dim)
            
            # Q K V projections (GPU Q4 matmul) — CHECK RETURN VALUES
            if not self._gpu_matmul(norm_buf, f"{prefix}.attn_q", self._q_buf, q_dim_l):
                print(f"[ENGINE FATAL] Layer {layer} Q projection failed — aborting forward pass")
                return None
            # Add Q bias if model has it (Qwen, Phi, etc.)
            q_bias = self.weights_cpu.get(f"{prefix}.attn_q.bias")
            if q_bias is not None:
                for i in range(q_dim_l):
                    self._q_buf[i] += q_bias[i]
            
            if not self._gpu_matmul(norm_buf, f"{prefix}.attn_k", self._k_buf, kv_dim_l):
                print(f"[ENGINE FATAL] Layer {layer} K projection failed — aborting forward pass")
                return None
            # Add K bias
            k_bias = self.weights_cpu.get(f"{prefix}.attn_k.bias")
            if k_bias is not None:
                for i in range(kv_dim_l):
                    self._k_buf[i] += k_bias[i]
            
            if not self._gpu_matmul(norm_buf, f"{prefix}.attn_v", self._v_buf, kv_dim_l):
                print(f"[ENGINE FATAL] Layer {layer} V projection failed — aborting forward pass")
                return None
            # Add V bias
            v_bias = self.weights_cpu.get(f"{prefix}.attn_v.bias")
            if v_bias is not None:
                for i in range(kv_dim_l):
                    self._v_buf[i] += v_bias[i]
            
            # RoPE
            apply_rope_inplace(self._q_buf, self.rope_cos[position], self.rope_sin[position], n_heads_l, head_dim_l)
            apply_rope_inplace(self._k_buf, self.rope_cos[position], self.rope_sin[position], n_kv_heads_l, head_dim_l)
            
            # Store K, V in cache
            kv_offset = position * kv_dim_l
            for i in range(kv_dim_l):
                self.kv_k[layer][kv_offset + i] = self._k_buf[i]
                self.kv_v[layer][kv_offset + i] = self._v_buf[i]
            
            # Attention (GPU)
            seq_len = position + 1

            # Try GPU attention kernel first; fall back to Python on NaN.
            # The CUDA kernel source (kernel_attention in nightmare_engine.cu)
            # is correct (uses shared-memory reduction, not atomicMax), but
            # older compiled DLLs may still have the atomicMax NaN bug.
            _use_gpu_attn = True
            try:
                self.lib.nightmare_attention(
                    float_ptr(self._q_buf),
                    float_ptr(self.kv_k[layer]),
                    float_ptr(self.kv_v[layer]),
                    float_ptr(self._attn_out),
                    c_int(n_heads_l), c_int(n_kv_heads_l), c_int(seq_len), c_int(head_dim_l)
                )
                # NaN check — if any output is NaN, the GPU kernel is buggy
                _attn_size = n_heads_l * head_dim_l
                for _i in range(_attn_size):
                    if self._attn_out[_i] != self._attn_out[_i]:
                        _use_gpu_attn = False
                        break
            except Exception:
                _use_gpu_attn = False

            if not _use_gpu_attn:
                # Pure Python attention fallback (GPU kernel produced NaN or failed)
                import math
                kv_group = max(1, n_heads_l // max(1, n_kv_heads_l))
                scale = 1.0 / math.sqrt(max(1, head_dim_l))
                for h in range(n_heads_l):
                    kv_h = h // kv_group
                    if kv_h >= n_kv_heads_l:
                        kv_h = n_kv_heads_l - 1
                    q_off = h * head_dim_l
                    # Compute scores
                    scores = []
                    for s in range(seq_len):
                        dot = 0.0
                        k_off = s * n_kv_heads_l * head_dim_l + kv_h * head_dim_l
                        for d in range(head_dim_l):
                            dot += self._q_buf[q_off + d] * self.kv_k[layer][k_off + d]
                        scores.append(dot * scale)
                    # Softmax
                    max_s = max(scores)
                    exp_s = [math.exp(s - max_s) for s in scores]
                    sum_s = sum(exp_s) + 1e-10
                    # Weighted sum of values
                    for d in range(head_dim_l):
                        val = 0.0
                        for s in range(seq_len):
                            v_off = s * n_kv_heads_l * head_dim_l + kv_h * head_dim_l
                            val += (exp_s[s] / sum_s) * self.kv_v[layer][v_off + d]
                        self._attn_out[q_off + d] = val
            
            # Output projection
            if not self._gpu_matmul(self._attn_out, f"{prefix}.attn_output", self._out, dim):
                print(f"[ENGINE FATAL] Layer {layer} output projection failed — aborting")
                return None
            
            # Residual add
            for i in range(dim):
                self._x[i] += self._out[i]
            
            # FFN
            # Pre-FFN norm
            self._rmsnorm(self._x, f"{prefix}.ffn_norm", norm_buf, dim)
            
            # Gate + Up projections
            layer_hidden_dim = hidden_dims[layer] if isinstance(hidden_dims, list) and layer < len(hidden_dims) else hidden_dim
            if not self._gpu_matmul(norm_buf, f"{prefix}.ffn_gate", self._ffn_buf, layer_hidden_dim):
                print(f"[ENGINE FATAL] Layer {layer} FFN gate failed — aborting")
                return None
            if not self._gpu_matmul(norm_buf, f"{prefix}.ffn_up", self._ffn_buf2, layer_hidden_dim):
                print(f"[ENGINE FATAL] Layer {layer} FFN up failed — aborting")
                return None
            
            # SiLU(gate) * up (GPU)
            self.lib.nightmare_silu_mul(
                float_ptr(self._ffn_buf), float_ptr(self._ffn_buf2),
                float_ptr(self._ffn_buf), c_int(layer_hidden_dim)
            )
            
            # Down projection
            if not self._gpu_matmul(self._ffn_buf, f"{prefix}.ffn_down", self._out, dim):
                print(f"[ENGINE FATAL] Layer {layer} FFN down failed — aborting")
                return None
            
            # Residual add
            for i in range(dim):
                self._x[i] += self._out[i]
        
        # Final norm
        self._rmsnorm(self._x, "output_norm", norm_buf, dim)
        
        # Output logits (vocab projection)
        logits = make_float_buf(self.config['vocab_size'])
        if not self._gpu_matmul(norm_buf, "output", logits, self.config['vocab_size']):
            print(f"[ENGINE FATAL] Output projection failed — no logits")
            return None
        out_bias = self.weights_cpu.get("output.bias")
        if out_bias is not None:
            n = min(self.config['vocab_size'], len(out_bias))
            for i in range(n):
                logits[i] += out_bias[i]
        if getattr(self, '_logit_cap', 0.0):
            cap = float(self._logit_cap)
            if cap > 0:
                for i in range(self.config['vocab_size']):
                    v = logits[i] / cap
                    logits[i] = math.tanh(v) * cap
        
        return logits
    
    def generate(self, token_ids, max_tokens=128, temperature=0.8, top_p=0.9):
        """Generate tokens. token_ids = list of ints (prompt)."""
        self.reset()
        
        # Prefill
        print(f"[Q4-ENGINE] Prefill {len(token_ids)} tokens...")
        t0 = time.time()
        for i, tok in enumerate(token_ids[:-1]):
            result = self.forward(tok, i)
            if result is None:
                print(f"[Q4-ENGINE] FATAL: Prefill failed at token {i}")
                return []
        prefill_time = time.time() - t0
        print(f"[Q4-ENGINE] Prefill done ({prefill_time:.1f}s)")
        
        # Generate
        pos = len(token_ids) - 1
        next_tok = token_ids[-1]
        generated = []
        
        t0 = time.time()
        for step in range(max_tokens):
            logits = self.forward(next_tok, pos)
            if logits is None:
                print(f"[Q4-ENGINE] FATAL: Forward failed at step {step}, pos {pos}")
                break
            
            # Sample (temperature + top-p)
            next_tok = self._sample(logits, temperature, top_p)
            generated.append(next_tok)
            pos += 1
            
            # EOS check
            if next_tok == 2:  # </s>
                break
        
        elapsed = time.time() - t0
        tok_per_sec = len(generated) / elapsed if elapsed > 0 else 0
        print(f"[Q4-ENGINE] Generated {len(generated)} tokens @ {tok_per_sec:.1f} tok/s")
        
        return generated
    
    def _sample(self, logits, temperature, top_p):
        """Sample from logits with temperature and top-p."""
        vocab_size = self.config['vocab_size']
        
        if temperature < 0.01:
            # Greedy
            best = 0
            best_val = logits[0]
            for i in range(1, vocab_size):
                if logits[i] > best_val:
                    best_val = logits[i]
                    best = i
            return best
        
        # Temperature scaling + softmax
        max_val = logits[0]
        for i in range(1, vocab_size):
            if logits[i] > max_val:
                max_val = logits[i]
        
        exp_sum = 0.0
        probs = []
        for i in range(vocab_size):
            p = math.exp((logits[i] - max_val) / temperature)
            probs.append(p)
            exp_sum += p
        
        # Normalize
        for i in range(vocab_size):
            probs[i] /= exp_sum
        
        # Top-p filtering
        sorted_indices = sorted(range(vocab_size), key=lambda i: probs[i], reverse=True)
        cumulative = 0.0
        cutoff = -1
        for idx in sorted_indices:
            cumulative += probs[idx]
            if cumulative >= top_p:
                cutoff = idx
                break
        
        # Random sample from top-p set
        import random
        r = random.random() * cumulative
        cumsum = 0.0
        for idx in sorted_indices:
            cumsum += probs[idx]
            if cumsum >= r:
                return idx
            if idx == cutoff:
                return idx
        
        return sorted_indices[0]
    
    def shutdown(self):
        """Free all GPU memory."""
        for key, entry in self.weights_gpu.items():
            ptr = entry[0]
            if len(entry) == 4 and hasattr(self.lib, 'nightmare_free_weight'):
                self.lib.nightmare_free_weight(c_void_p(ptr))
            else:
                self.lib.nightmare_free_q4(ptr)
        self.weights_gpu.clear()
        self.lib.nightmare_q4_free_buffers()
        print("[Q4-ENGINE] Shutdown complete.")


# ============================================================
# ENTRY POINT — standalone test
# ============================================================

if __name__ == '__main__':
    engine = NightmareQ4Engine()
    if not engine.available:
        print("No GPU. Install CUDA and recompile nightmare_engine.dll")
        sys.exit(1)
    
    # Find model
    paths = [
        'models/nightmare_7b_q4_best.gguf',
        r'C:\Users\User\Desktop\nightmare_7b_q4_best.gguf',
    ]
    model_path = None
    for p in paths:
        if os.path.exists(p):
            model_path = p
            break
    
    if not model_path:
        print("Model not found!")
        sys.exit(1)
    
    if engine.load(model_path):
        # Quick test — greedy generate from BOS
        print("\nTest generation (greedy, 10 tokens):")
        logits = engine.forward(1, 0)  # BOS
        for i in range(10):
            # Greedy
            best = 0
            best_val = logits[0]
            for j in range(1, engine.config['vocab_size']):
                if logits[j] > best_val:
                    best_val = logits[j]
                    best = j
            print(f"  Token {i}: {best}")
            logits = engine.forward(best, i + 1)
        
        engine.shutdown()
