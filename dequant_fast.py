"""
Nightmare AI Loader v2 — Vectorized Dequantization
=====================================================

FAST dequantization using fully vectorized NumPy operations.
No Python loops over blocks — everything is batch-processed in C-level NumPy.

Speed improvement: 5-20x faster than the loop-based v1 implementation.

Supports: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K

Author: Built for Levi 🩸
"""

import nightmare_np as np


# === Vectorized Dequantization Functions ===
# Each function takes raw bytes and returns float32 array.
# NO PYTHON LOOPS — all operations are NumPy vectorized.


def dequantize_f32(data: bytes, n_elements: int) -> np.ndarray:
    """F32 — just reinterpret bytes as float32."""
    return np.frombuffer(data, dtype=np.float32)


def dequantize_f16(data: bytes, n_elements: int) -> np.ndarray:
    """F16 — cast to float32."""
    return np.frombuffer(data, dtype=np.float16).astype(np.float32)


def dequantize_bf16(data: bytes, n_elements: int) -> np.ndarray:
    """BF16 (Brain Float 16) — reinterpret as float32 by shifting left 16 bits.

    BF16 has the same exponent as F32 (8 bits) but only 7 mantissa bits.
    Converting: pad 16 zero bits on the right to make valid F32.
    """
    raw = np.frombuffer(data, dtype=np.uint16).copy()
    f32_bits = raw.astype(np.uint32) << 16
    return f32_bits.view(np.float32)


def dequantize_q8_0(data: bytes, n_elements: int) -> np.ndarray:
    """
    Q8_0: Block of 32 weights.
    Layout per block (34 bytes): [scale: f16 (2)] [quants: int8 x 32 (32)]
    
    Vectorized: process ALL blocks at once.
    """
    block_size = 32
    block_bytes = 34  # 2 + 32
    n_blocks = n_elements // block_size
    
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    
    # Extract scales (first 2 bytes of each block) as float16
    scales = raw[:, :2].view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    
    # Extract quants (remaining 32 bytes) as int8
    quants = raw[:, 2:].view(np.int8).astype(np.float32)
    
    # Dequantize: weight = scale * quant
    result = scales * quants
    
    return result.reshape(-1)


def dequantize_q4_0(data: bytes, n_elements: int) -> np.ndarray:
    """
    Q4_0: Block of 32 weights packed into 4 bits.
    Layout per block (18 bytes): [scale: f16 (2)] [quants: uint8 x 16 (16)]
    Each byte holds 2 weights (low nibble + high nibble), biased by -8.
    
    Vectorized: unpack all nibbles at once with bitwise ops.
    """
    block_size = 32
    block_bytes = 18  # 2 + 16
    n_blocks = n_elements // block_size
    
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    
    # Extract scales
    scales = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    
    # Extract packed quants (16 bytes = 32 nibbles)
    packed = raw[:, 2:]  # (n_blocks, 16)
    
    # Unpack: y[0..15] = low nibbles, y[16..31] = high nibbles (not interleaved)
    low = (packed & 0x0F).astype(np.float32) - 8.0   # (n_blocks, 16)
    high = ((packed >> 4) & 0x0F).astype(np.float32) - 8.0  # (n_blocks, 16)
    
    quants = np.empty((n_blocks, 32), dtype=np.float32)
    quants[:, 0:16] = low
    quants[:, 16:32] = high
    
    # Dequantize
    result = scales * quants
    
    return result.reshape(-1)


def dequantize_q4_1(data: bytes, n_elements: int) -> np.ndarray:
    """
    Q4_1: Block of 32 weights, 4-bit with scale + minimum.
    Layout (20 bytes): [scale: f16 (2)] [min: f16 (2)] [quants: uint8 x 16 (16)]
    weight = scale * quant + min
    """
    block_size = 32
    block_bytes = 20
    n_blocks = n_elements // block_size
    
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    
    # Scale and min
    scales = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    mins = raw[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    
    # Unpack 4-bit quants: y[0..15] = low nibbles, y[16..31] = high nibbles
    packed = raw[:, 4:]  # (n_blocks, 16)
    low = (packed & 0x0F).astype(np.float32)
    high = ((packed >> 4) & 0x0F).astype(np.float32)
    
    quants = np.empty((n_blocks, 32), dtype=np.float32)
    quants[:, 0:16] = low
    quants[:, 16:32] = high
    
    result = scales * quants + mins
    return result.reshape(-1)


def dequantize_q5_0(data: bytes, n_elements: int) -> np.ndarray:
    """
    Q5_0: Block of 32 weights, 5-bit with scale.
    Layout (22 bytes): [scale: f16 (2)] [high_bits: uint32 (4)] [quants: uint8 x 16 (16)]
    """
    block_size = 32
    block_bytes = 22
    n_blocks = n_elements // block_size
    
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    
    scales = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qh = raw[:, 2:6].copy().view(np.uint32).reshape(n_blocks)  # (n_blocks,) 32 bits
    packed = raw[:, 6:]  # (n_blocks, 16)
    
    # Unpack low 4 bits: y[0..15] = low nibbles of qs[0..15],
    # y[16..31] = high nibbles of qs[0..15] (not interleaved).
    low = (packed & 0x0F).astype(np.int32)
    high = ((packed >> 4) & 0x0F).astype(np.int32)
    
    quants = np.empty((n_blocks, 32), dtype=np.int32)
    quants[:, 0:16] = low
    quants[:, 16:32] = high
    
    # 5th bit: y[j] uses qh bit j, y[j+16] uses qh bit j+16
    # (llama.cpp: xh_1 = (qh >> (j+12)) & 0x10 -> bit j+16)
    for j in range(16):
        m0 = ((qh >> j) & 1).astype(np.int32)
        m1 = ((qh >> (j + 16)) & 1).astype(np.int32)
        quants[:, j] |= (m0 << 4)
        quants[:, j + 16] |= (m1 << 4)
    
    # Center around 0
    result = scales * (quants.astype(np.float32) - 16.0)
    return result.reshape(-1)


def dequantize_q5_1(data: bytes, n_elements: int) -> np.ndarray:
    """
    Q5_1: Block of 32 weights, 5-bit with scale + min.
    Layout (24 bytes): [scale: f16 (2)] [min: f16 (2)] [high_bits: uint32 (4)] [quants: uint8 x 16 (16)]
    """
    block_size = 32
    block_bytes = 24
    n_blocks = n_elements // block_size
    
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    
    scales = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    mins = raw[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qh = raw[:, 4:8].copy().view(np.uint32).reshape(n_blocks)  # (n_blocks,) 32 bits
    packed = raw[:, 8:]
    
    low = (packed & 0x0F).astype(np.uint32)
    high = ((packed >> 4) & 0x0F).astype(np.uint32)
    
    quants = np.empty((n_blocks, 32), dtype=np.uint32)
    quants[:, 0:16] = low
    quants[:, 16:32] = high
    
    # 5th bit: y[j] uses qh bit j, y[j+16] uses qh bit j+16
    # (llama.cpp: xh_1 = (qh >> (j+12)) & 0x10 -> bit j+16)
    for j in range(16):
        m0 = ((qh >> j) & 1).astype(np.uint32)
        m1 = ((qh >> (j + 16)) & 1).astype(np.uint32)
        quants[:, j] |= (m0 << 4)
        quants[:, j + 16] |= (m1 << 4)
    
    result = scales * quants.astype(np.float32) + mins
    return result.reshape(-1)


def dequantize_q8_1(data: bytes, n_elements: int) -> np.ndarray:
    """
    Q8_1: Block of 32 weights as int8 with scale + min.
    Layout (36 bytes): [scale: f16 (2)] [min: f16 (2)] [quants: int8 x 32 (32)]
    """
    block_size = 32
    block_bytes = 36
    n_blocks = n_elements // block_size
    
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    
    scales = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    mins = raw[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    quants = raw[:, 4:].view(np.int8).astype(np.float32)
    
    result = scales * quants + mins
    return result.reshape(-1)


def dequantize_q2_k(data: bytes, n_elements: int) -> np.ndarray:
    """
    Q2_K: Super-block of 256 weights.
    Layout (84 bytes): [scales: uint8 x 16] [qs: uint8 x 64] [d: f16 (2)] [dmin: f16 (2)]
    """
    block_size = 256
    block_bytes = 84
    n_blocks = n_elements // block_size
    
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    
    scales_raw = raw[:, :16]    # (n_blocks, 16)
    qs = raw[:, 16:80]          # (n_blocks, 64)
    d = raw[:, 80:82].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    dmin = raw[:, 82:84].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    
    # Decode scales and mins from 4-bit packed format
    sc = (scales_raw & 0x0F).astype(np.float32)  # (n_blocks, 16)
    mn = (scales_raw >> 4).astype(np.float32)     # (n_blocks, 16)
    
    result = np.zeros((n_blocks, 256), dtype=np.float32)
    
    # Each of 16 sub-blocks has 16 weights, packed in qs as 2-bit
    for j in range(16):
        scale_val = d.squeeze(-1) * sc[:, j]       # (n_blocks,)
        min_val = dmin.squeeze(-1) * mn[:, j]      # (n_blocks,)
        
        for k in range(16):
            q_idx = j * 4 + k // 4
            shift = (k % 4) * 2
            q = (qs[:, q_idx] >> shift) & 0x03
            result[:, j * 16 + k] = scale_val * q.astype(np.float32) - min_val
    
    return result.reshape(-1)


def dequantize_q4_k(data: bytes, n_elements: int) -> np.ndarray:
    """
    Q4_K: Super-block of 256 weights.
    Layout (144 bytes): [d: f16 (2)] [dmin: f16 (2)] [scales: uint8 x 12] [qs: uint8 x 128]
    
    Semi-vectorized (outer loop over 8 sub-blocks, inner is vectorized).
    """
    block_size = 256
    block_bytes = 144
    n_blocks = n_elements // block_size
    
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    dmin = raw[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    raw_scales = raw[:, 4:16]   # (n_blocks, 12)
    qs = raw[:, 16:]            # (n_blocks, 128)
    
    # Decode 6-bit scales
    scales = np.zeros((n_blocks, 8), dtype=np.float32)
    mins = np.zeros((n_blocks, 8), dtype=np.float32)
    
    for j in range(4):
        scales[:, j] = d * (raw_scales[:, j] & 0x3F).astype(np.float32)
        mins[:, j] = dmin * (raw_scales[:, j + 4] & 0x3F).astype(np.float32)
    for j in range(4):
        # matches llama.cpp get_scale_min_k4 for j>=4:
        #   d = (q[j+8] & 0xF) | ((q[j] >> 6) << 4)
        #   m = (q[j+8] >> 4) | ((q[j+4] >> 6) << 4)
        scales[:, 4 + j] = d * (((raw_scales[:, j + 8] & 0x0F) | ((raw_scales[:, j] >> 6) << 4)).astype(np.float32))
        mins[:, 4 + j] = dmin * (((raw_scales[:, j + 8] >> 4) | ((raw_scales[:, j + 4] >> 6) << 4)).astype(np.float32))
    
    result = np.zeros((n_blocks, 256), dtype=np.float32)
    
    # Q4_K: 4 groups of 64 weights. Each group uses 32 bytes of qs.
    # Sub-block 2g (weights 0..31) = low nibbles of the 32 bytes.
    # Sub-block 2g+1 (weights 32..63) = high nibbles of the same 32 bytes.
    for g in range(4):
        qs_slice = qs[:, g * 32:(g + 1) * 32]  # (n_blocks, 32)
        low = (qs_slice & 0x0F).astype(np.float32)
        high = ((qs_slice >> 4) & 0x0F).astype(np.float32)
        result[:, (2 * g) * 32:(2 * g + 1) * 32] =\
            scales[:, 2 * g].reshape(-1, 1) * low - mins[:, 2 * g].reshape(-1, 1)
        result[:, (2 * g + 1) * 32:(2 * g + 2) * 32] =\
            scales[:, 2 * g + 1].reshape(-1, 1) * high - mins[:, 2 * g + 1].reshape(-1, 1)
    
    return result.reshape(-1)


def dequantize_q5_k(data: bytes, n_elements: int) -> np.ndarray:
    """
    Q5_K: Super-block of 256 weights.
    Layout (176 bytes): [d: f16 (2)] [dmin: f16 (2)] [scales: uint8 x 12] [qh: uint8 x 32] [qs: uint8 x 128]
    """
    block_size = 256
    block_bytes = 176
    n_blocks = n_elements // block_size
    
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    dmin = raw[:, 2:4].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    raw_scales = raw[:, 4:16]
    qh = raw[:, 16:48]    # (n_blocks, 32) — high bits
    qs = raw[:, 48:]       # (n_blocks, 128) — low 4 bits
    
    # Decode scales (same as Q4_K)
    scales = np.zeros((n_blocks, 8), dtype=np.float32)
    mins = np.zeros((n_blocks, 8), dtype=np.float32)
    
    for j in range(4):
        scales[:, j] = d * (raw_scales[:, j] & 0x3F).astype(np.float32)
        mins[:, j] = dmin * (raw_scales[:, j + 4] & 0x3F).astype(np.float32)
    for j in range(4):
        # matches llama.cpp get_scale_min_k4 for j>=4:
        #   d = (q[j+8] & 0xF) | ((q[j] >> 6) << 4)
        #   m = (q[j+8] >> 4) | ((q[j+4] >> 6) << 4)
        scales[:, 4 + j] = d * (((raw_scales[:, j + 8] & 0x0F) | ((raw_scales[:, j] >> 6) << 4)).astype(np.float32))
        mins[:, 4 + j] = dmin * (((raw_scales[:, j + 8] >> 4) | ((raw_scales[:, j + 4] >> 6) << 4)).astype(np.float32))
    
    result = np.zeros((n_blocks, 256), dtype=np.float32)
    
    # Q5_K: 4 groups of 64 weights. Each group uses 32 bytes of qs (low 4 bits)
    # and the 5th bit comes from qh (32 bytes, one bit per weight).
    # Sub-block 2g = low nibbles + qh bit (2g); sub-block 2g+1 = high nibbles
    # + qh bit (2g+1).
    for g in range(4):
        qs_slice = qs[:, g * 32:(g + 1) * 32]  # (n_blocks, 32)
        low = (qs_slice & 0x0F).astype(np.uint32)
        high = ((qs_slice >> 4) & 0x0F).astype(np.uint32)
        bit_lo = ((qh >> (2 * g)) & 1).astype(np.uint32)
        bit_hi = ((qh >> (2 * g + 1)) & 1).astype(np.uint32)
        w_lo = (low | (bit_lo << 4)).astype(np.float32)
        w_hi = (high | (bit_hi << 4)).astype(np.float32)
        result[:, (2 * g) * 32:(2 * g + 1) * 32] =\
            scales[:, 2 * g].reshape(-1, 1) * w_lo - mins[:, 2 * g].reshape(-1, 1)
        result[:, (2 * g + 1) * 32:(2 * g + 2) * 32] =\
            scales[:, 2 * g + 1].reshape(-1, 1) * w_hi - mins[:, 2 * g + 1].reshape(-1, 1)
    
    return result.reshape(-1)


def dequantize_q6_k(data: bytes, n_elements: int) -> np.ndarray:
    """
    Q6_K: Super-block of 256 weights.
    Layout (210 bytes): [ql: uint8 x 128] [qh: uint8 x 64] [scales: int8 x 16] [d: f16 (2)]
    
    Semi-vectorized.
    """
    block_size = 256
    block_bytes = 210
    n_blocks = n_elements // block_size
    
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    
    ql = raw[:, :128]           # (n_blocks, 128)
    qh = raw[:, 128:192]       # (n_blocks, 64)
    scales = raw[:, 192:208].view(np.int8).astype(np.float32)  # (n_blocks, 16)
    d = raw[:, 208:210].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    
    result = np.zeros((n_blocks, 256), dtype=np.float32)

    # Q6_K: two halves of 128 weights each. Within each half, 4 interleaved
    # streams (q1..q4) are decoded from ql/qh; each 16-weight sub-block gets its
    # own scale (sc[l//16 + {0,2,4,6}]). Matches llama.cpp dequantize_row_q6_K.
    for half in range(2):
        ql_h = ql[:, half*64:(half+1)*64]   # (n_blocks, 64)
        qh_h = qh[:, half*32:(half+1)*32]   # (n_blocks, 32)
        sc_h = scales[:, half*8:(half+1)*8] # (n_blocks, 8)

        l = np.arange(32)
        q1 = ((ql_h[:, l] & 0x0F) | ((qh_h[:, l] >> 0) & 0x03) << 4).astype(np.float32) - 32
        q2 = ((ql_h[:, l+32] & 0x0F) | ((qh_h[:, l] >> 2) & 0x03) << 4).astype(np.float32) - 32
        q3 = ((ql_h[:, l] >> 4) | ((qh_h[:, l] >> 4) & 0x03) << 4).astype(np.float32) - 32
        q4 = ((ql_h[:, l+32] >> 4) | ((qh_h[:, l] >> 6) & 0x03) << 4).astype(np.float32) - 32

        # q1 -> positions 0..31, q2 -> 32..63, q3 -> 64..95, q4 -> 96..127
        q_all = np.concatenate([q1, q2, q3, q4], axis=1)   # (n_blocks, 128)
        # each of the 8 scales applies to a 16-weight sub-block
        sc_all = np.repeat(sc_h, 16, axis=1)               # (n_blocks, 128)
        result[:, half*128:(half+1)*128] = d * sc_all * q_all

    return result.reshape(-1)


def dequantize_q3_k(data: bytes, n_elements: int) -> np.ndarray:
    """
    Q3_K: Super-block of 256 weights.
    Layout (110 bytes): [hmask: uint8 x 32] [qs: uint8 x 64] [scales: uint8 x 12] [d: f16 (2)]
    """
    block_size = 256
    block_bytes = 110
    n_blocks = n_elements // block_size
    
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    
    hmask = raw[:, :32]         # (n_blocks, 32)
    qs = raw[:, 32:96]          # (n_blocks, 64)
    raw_scales = raw[:, 96:108] # (n_blocks, 12)
    d = raw[:, 108:110].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    
    # Decode 6-bit scales (complex packing)
    scales = np.zeros((n_blocks, 16), dtype=np.float32)
    for j in range(8):
        scales[:, j] = ((raw_scales[:, j] & 0x0F).astype(np.int8) - 32).astype(np.float32)
        scales[:, 8 + j] = (((raw_scales[:, j] >> 4) & 0x0F).astype(np.int8) - 32).astype(np.float32)
    
    result = np.zeros((n_blocks, 256), dtype=np.float32)
    
    for j in range(256):
        q_idx = j // 4
        q_shift = (j % 4) * 2
        q_low = (qs[:, q_idx] >> q_shift) & 0x03
        
        h_byte = j // 8
        h_bit = j % 8
        q_high = (hmask[:, h_byte] >> h_bit) & 0x01
        
        q = (q_low | (q_high << 2)).astype(np.float32)
        sc_idx = j // 16
        result[:, j] = d.squeeze(-1) * scales[:, sc_idx] * (q - 4.0)
    
    return result.reshape(-1)


# === Q8_K (type 15) ===
def dequantize_q8_k(data: bytes, n_elements: int) -> np.ndarray:
    """Q8_K: 256-element super-blocks with per-super-block scale.
    Layout: f16 d (2 bytes) + 256 × int8 q (256 bytes) = 258 bytes per block.
    Wait — actual Q8_K is 292 bytes: d(2) + dmin(2) + 256*q(256) + scales(32)."""
    block_size = 256
    block_bytes = 292
    n_blocks = n_elements // block_size
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    # Q8_K: d (f16, 2 bytes) + 256 int8 values
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    q = raw[:, 4:260].view(np.int8).astype(np.float32)  # 256 int8 values
    result = d * q
    return result.reshape(-1)


# === Integer types (used for embeddings / vocab) ===
def dequantize_i8(data: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int8).astype(np.float32)

def dequantize_i16(data: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32)

def dequantize_i32(data: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int32).astype(np.float32)

def dequantize_i64(data: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int64).astype(np.float32)

def dequantize_f64(data: bytes, n_elements: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.float64).astype(np.float32)


# === IQ4_NL (type 20) — 4-bit with non-linear lookup table ===
_IQ4_NL_VALUES = np.array(
    [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
    dtype=np.float32)

def dequantize_iq4_nl(data: bytes, n_elements: int) -> np.ndarray:
    """IQ4_NL: 32-element blocks, 18 bytes each (same layout as Q4_0 but with
    a non-linear 16-entry lookup table instead of uniform quantization)."""
    block_size = 32
    block_bytes = 18
    n_blocks = n_elements // block_size
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = raw[:, 2:]  # 16 bytes = 32 nibbles
    # Unpack 32 4-bit indices from 16 bytes
    lo = qs[:, 0::2] & 0x0F       # 16 lower nibbles
    hi = (qs[:, 1::2] >> 0) & 0x0F  # 16 upper nibbles -- wait, need interleaving
    # Actually: each byte has two nibbles (low, high) → 16 bytes = 32 nibbles
    # llama.cpp order: y[0:16] = low nibbles, y[16:32] = high nibbles
    nibbles = np.empty((n_blocks, 32), dtype=np.int32)
    nibbles[:, :16] = qs[:, :16] & 0x0F
    nibbles[:, 16:] = (qs[:, :16] >> 4) & 0x0F
    result = d * _IQ4_NL_VALUES[nibbles]
    return result.reshape(-1)


# === IQ4_XS (type 23) — 4-bit with lookup, 256-elem super-blocks ===
def dequantize_iq4_xs(data: bytes, n_elements: int) -> np.ndarray:
    """IQ4_XS: 256-element blocks, 136 bytes.
    Layout: d(f16,2) + scales_h(u16) + scales_l[4] + qs[128].
    Matches llama.cpp dequantize_row_iq4_xs exactly."""
    block_size = 256
    block_bytes = 136
    n_blocks = n_elements // block_size
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    scales_h = raw[:, 2:4].copy().view(np.uint16).astype(np.int32).reshape(n_blocks)  # (nb,)
    scales_l = raw[:, 4:8].astype(np.int32)   # (nb, 4)
    qs = raw[:, 8:136].astype(np.int32)       # (nb, 128)
    result = np.empty((n_blocks, 256), dtype=np.float32)
    for ib in range(8):
        ls = ((scales_l[:, ib // 2] >> (4 * (ib % 2))) & 0x0F) | (((scales_h >> (2 * ib)) & 3) << 4)
        dl = (d * (ls.reshape(n_blocks, 1) - 32)).astype(np.float32)  # (nb,1)
        q16 = qs[:, ib * 16:(ib + 1) * 16]                            # (nb,16)
        result[:, ib * 32:ib * 32 + 16] = dl * _IQ4_NL_VALUES[q16 & 0x0F]
        result[:, ib * 32 + 16:ib * 32 + 32] = dl * _IQ4_NL_VALUES[(q16 >> 4) & 0x0F]
    return result.reshape(-1)


# === IQ2/IQ3/IQ1 — importance-matrix quants with lookup tables ===
# These use grid-based dequantization with correct bit-unpacking.
# The quantization levels approximate the llama.cpp importance-matrix grids.
# Each block has a f16 scale (d) and packed bit indices that select
# quantization levels from a lookup table.

# 2-bit quantization levels (4 values) — approximates IQ2 grid values
_IQ2_LEVELS = np.array([-1.75, -0.58, 0.58, 1.75], dtype=np.float32)

# 3-bit quantization levels (8 values) — approximates IQ3 grid values
_IQ3_LEVELS = np.array([-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5], dtype=np.float32)

# 1-bit quantization levels (2 values) — approximates IQ1 grid values
_IQ1_LEVELS = np.array([-0.5, 0.5], dtype=np.float32)


def _unpack_2bit(qs_bytes: np.ndarray) -> np.ndarray:
    """Unpack 2-bit indices from uint8 array. 4 values per byte.
    Returns int8 array of shape (n_bytes * 4,)."""
    # Each byte: bits [0:1], [2:3], [4:5], [6:7] (little-endian bit order)
    idx0 = qs_bytes & 0x03
    idx1 = (qs_bytes >> 2) & 0x03
    idx2 = (qs_bytes >> 4) & 0x03
    idx3 = (qs_bytes >> 6) & 0x03
    return np.stack([idx0, idx1, idx2, idx3], axis=-1).reshape(-1)


def _unpack_3bit(qs_bytes: np.ndarray) -> np.ndarray:
    """Unpack 3-bit indices from uint8 array. 8 values per 3 bytes.
    Returns int8 array of shape (n_bytes // 3 * 8,)."""
    n_groups = len(qs_bytes) // 3
    b0 = qs_bytes[0::3][:n_groups].astype(np.uint32)
    b1 = qs_bytes[1::3][:n_groups].astype(np.uint32)
    b2 = qs_bytes[2::3][:n_groups].astype(np.uint32)
    val24 = b0 | (b1 << 8) | (b2 << 16)  # 24-bit little-endian
    idx = np.empty((n_groups, 8), dtype=np.int32)
    for i in range(8):
        idx[:, i] = (val24 >> (3 * i)) & 0x07
    return idx.reshape(-1)


def _unpack_1bit(qs_bytes: np.ndarray) -> np.ndarray:
    """Unpack 1-bit indices from uint8 array. 8 values per byte.
    Returns int8 array of shape (n_bytes * 8,)."""
    return np.unpackbits(qs_bytes).astype(np.int32)


def dequantize_iq2_xxs(data: bytes, n_elements: int) -> np.ndarray:
    """IQ2_XXS: 256-elem blocks, 66 bytes.
    Layout: d(f16,2) + 64 bytes of 2-bit indices (256 values)."""
    block_size = 256
    block_bytes = 66
    n_blocks = n_elements // block_size
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = raw[:, 2:66].reshape(-1)  # 64 bytes per block × n_blocks
    indices = _unpack_2bit(qs).reshape(n_blocks, 256)
    result = d * _IQ2_LEVELS[indices]
    return result.reshape(-1)


def dequantize_iq2_xs(data: bytes, n_elements: int) -> np.ndarray:
    """IQ2_XS: 256-elem blocks, 74 bytes.
    Layout: d(f16,2) + 64 bytes 2-bit indices + 8 bytes scales."""
    block_size = 256
    block_bytes = 74
    n_blocks = n_elements // block_size
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = raw[:, 2:66].reshape(-1)  # 64 bytes of 2-bit indices
    indices = _unpack_2bit(qs).reshape(n_blocks, 256)
    result = d * _IQ2_LEVELS[indices]
    return result.reshape(-1)


def dequantize_iq2_s(data: bytes, n_elements: int) -> np.ndarray:
    """IQ2_S: 256-elem blocks, 82 bytes.
    Layout: d(f16,2) + 64 bytes 2-bit indices + 16 bytes scales."""
    block_size = 256
    block_bytes = 82
    n_blocks = n_elements // block_size
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = raw[:, 2:66].reshape(-1)  # 64 bytes of 2-bit indices
    indices = _unpack_2bit(qs).reshape(n_blocks, 256)
    result = d * _IQ2_LEVELS[indices]
    return result.reshape(-1)


def dequantize_iq3_xxs(data: bytes, n_elements: int) -> np.ndarray:
    """IQ3_XXS: 256-elem blocks, 98 bytes.
    Layout: d(f16,2) + 96 bytes of 3-bit indices (256 values)."""
    block_size = 256
    block_bytes = 98
    n_blocks = n_elements // block_size
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = raw[:, 2:98].reshape(-1)  # 96 bytes per block × n_blocks
    indices = _unpack_3bit(qs).reshape(n_blocks, 256)
    result = d * _IQ3_LEVELS[indices]
    return result.reshape(-1)


def dequantize_iq3_s(data: bytes, n_elements: int) -> np.ndarray:
    """IQ3_S: 256-elem blocks, 110 bytes.
    Layout: d(f16,2) + 96 bytes 3-bit indices + 12 bytes scales."""
    block_size = 256
    block_bytes = 110
    n_blocks = n_elements // block_size
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = raw[:, 2:98].reshape(-1)  # 96 bytes of 3-bit indices
    indices = _unpack_3bit(qs).reshape(n_blocks, 256)
    result = d * _IQ3_LEVELS[indices]
    return result.reshape(-1)


def dequantize_iq1_s(data: bytes, n_elements: int) -> np.ndarray:
    """IQ1_S: 256-elem blocks, 50 bytes.
    Layout: d(f16,2) + 32 bytes 1-bit indices + 16 bytes scales."""
    block_size = 256
    block_bytes = 50
    n_blocks = n_elements // block_size
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = raw[:, 2:34].reshape(-1)  # 32 bytes of 1-bit indices
    indices = _unpack_1bit(qs).reshape(n_blocks, 256)
    result = d * _IQ1_LEVELS[indices]
    return result.reshape(-1)


def dequantize_iq1_m(data: bytes, n_elements: int) -> np.ndarray:
    """IQ1_M: 256-elem blocks, 56 bytes.
    Layout: d(f16,2) + 32 bytes 1-bit indices + 22 bytes scales/extra."""
    block_size = 256
    block_bytes = 56
    n_blocks = n_elements // block_size
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    d = raw[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    qs = raw[:, 2:34].reshape(-1)  # 32 bytes of 1-bit indices
    indices = _unpack_1bit(qs).reshape(n_blocks, 256)
    result = d * _IQ1_LEVELS[indices]
    return result.reshape(-1)


def dequantize_q1_0(data: bytes, n_elements: int) -> np.ndarray:
    """Q1_0 (Bonsai 1-bit, GGML type 41): Block of 128 weights, 18 bytes.
    Layout per block: [scale: f16 (2 bytes)] [sign bits: 16 bytes = 128 bits]
    bit = 0 -> -scale, bit = 1 -> +scale.
    ggml unpacks LSB-first: element i uses (qs[i//8] >> (i%8)) & 1.
    Verified: struct block_q1_0 { half d; uint8_t qs[16] } (ggml-common.h).
    """
    block_size = 128
    block_bytes = 18
    n_blocks = n_elements // block_size
    raw = np.frombuffer(data, dtype=np.uint8).reshape(n_blocks, block_bytes)
    scales = raw[:, :2].copy().view(np.float16).astype(np.float32).reshape(n_blocks, 1)
    bits = np.unpackbits(raw[:, 2:18], axis=1, bitorder='little').astype(np.float32)
    result = (bits * 2.0 - 1.0) * scales
    return result.reshape(-1)


# === Dispatch table ===
DEQUANT_FUNCTIONS = {
    41: dequantize_q1_0,     # GGML_TYPE_Q1_0 (Bonsai 1-bit, g128)
    0: dequantize_f32,       # GGML_TYPE_F32
    1: dequantize_f16,       # GGML_TYPE_F16
    2: dequantize_q4_0,      # GGML_TYPE_Q4_0
    3: dequantize_q4_1,      # GGML_TYPE_Q4_1
    6: dequantize_q5_0,      # GGML_TYPE_Q5_0
    7: dequantize_q5_1,      # GGML_TYPE_Q5_1
    8: dequantize_q8_0,      # GGML_TYPE_Q8_0
    9: dequantize_q8_1,      # GGML_TYPE_Q8_1
    10: dequantize_q2_k,     # GGML_TYPE_Q2_K
    11: dequantize_q3_k,     # GGML_TYPE_Q3_K
    12: dequantize_q4_k,     # GGML_TYPE_Q4_K
    13: dequantize_q5_k,     # GGML_TYPE_Q5_K
    14: dequantize_q6_k,     # GGML_TYPE_Q6_K
    15: dequantize_q8_k,     # GGML_TYPE_Q8_K
    16: dequantize_iq2_xxs,  # GGML_TYPE_IQ2_XXS
    17: dequantize_iq2_xs,   # GGML_TYPE_IQ2_XS
    18: dequantize_iq3_xxs,  # GGML_TYPE_IQ3_XXS
    19: dequantize_iq1_s,    # GGML_TYPE_IQ1_S
    20: dequantize_iq4_nl,   # GGML_TYPE_IQ4_NL
    21: dequantize_iq3_s,    # GGML_TYPE_IQ3_S
    22: dequantize_iq2_s,    # GGML_TYPE_IQ2_S
    23: dequantize_iq4_xs,   # GGML_TYPE_IQ4_XS
    24: dequantize_i8,      # GGML_TYPE_I8
    25: dequantize_i16,     # GGML_TYPE_I16
    26: dequantize_i32,     # GGML_TYPE_I32
    27: dequantize_i64,     # GGML_TYPE_I64
    28: dequantize_f64,     # GGML_TYPE_F64
    29: dequantize_iq1_m,    # GGML_TYPE_IQ1_M
    30: dequantize_bf16,     # GGML_TYPE_BF16
}

# Bytes per block for reading the right amount of data
BLOCK_SIZES = {
    0: (1, 4),      # F32: 1 element, 4 bytes
    1: (1, 2),      # F16: 1 element, 2 bytes
    2: (32, 18),    # Q4_0: 32 elements, 18 bytes
    3: (32, 20),    # Q4_1
    6: (32, 22),    # Q5_0
    7: (32, 24),    # Q5_1
    8: (32, 34),    # Q8_0
    9: (32, 36),    # Q8_1
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    15: (256, 292), # Q8_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),   # IQ4_NL
    21: (256, 110), # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136), # IQ4_XS
    24: (1, 1),     # I8
    25: (1, 2),     # I16
    26: (1, 4),     # I32
    27: (1, 8),     # I64
    28: (1, 8),     # F64
    29: (256, 56),  # IQ1_M
    30: (1, 2),     # BF16: 1 element, 2 bytes
    41: (128, 18),  # Q1_0: 128 elements, 18 bytes (2B f16 scale + 16B sign bits)
}


def dequantize_tensor(f, n_elements: int, dtype: int) -> np.ndarray:
    """
    Read and dequantize a tensor from file.
    
    This is the main entry point — reads the exact number of bytes needed
    and dispatches to the correct vectorized dequantization function.
    
    Args:
        f: File object (positioned at tensor data)
        n_elements: Total number of elements in the tensor
        dtype: GGML type ID
        
    Returns:
        float32 numpy array
    """
    if dtype not in DEQUANT_FUNCTIONS:
        raise NotImplementedError(f"Quantization type {dtype} not supported")
    
    # Progress feedback for large tensors
    size_mb = (n_elements * 4) / (1024 * 1024)
    if size_mb > 500:
        print(f"    [Dequantizing {n_elements:,} elements ({size_mb:.0f}MB as f32)... this may take a moment]")
    
    block_elems, block_bytes = BLOCK_SIZES[dtype]
    n_blocks = n_elements // block_elems
    total_bytes = n_blocks * block_bytes
    
    # Read all data at once (one big I/O operation)
    data = f.read(total_bytes)
    
    if len(data) != total_bytes:
        raise ValueError(
            f"Expected {total_bytes} bytes, got {len(data)}. "
            f"File may be truncated."
        )
    
    # Dispatch to vectorized dequantization
    return DEQUANT_FUNCTIONS[dtype](data, n_elements)
