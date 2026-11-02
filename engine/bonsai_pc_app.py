# -*- coding: utf-8 -*-
"""
bonsai_pc_app.py  —  Bonsai-27B PC app (standalone, NTMRE-style GPU engine)
============================================================================
A fully SELF-CONTAINED PC app for fun. It has ZERO imports from the Nightmare
app code — everything is inlined here — but it USES the same system:
  * NTMRE .ntmre compression (NSVD) format
  * NVRTC-compiled CUDA kernels for the matmuls (faster on PC)
  * factored .ntmre compression matmul (never materializes full weights)

It runs Bonsai-27B, a Qwen3.5 hybrid (Gated DeltaNet SSM + Gated Attention +
partial RoPE).

Run:
    python bonsai_pc_app.py Bonsai-27B.ntmre            # Gradio UI
    python bonsai_pc_app.py Bonsai-27B.ntmre --cli      # terminal chat
"""
import os
import sys
import json
import struct
import ctypes
import mmap
import argparse
import threading
import nightmare_np as np
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════
# NSVD (NTMRE .ntmre compression) format — self-contained reader
# ═══════════════════════════════════════════════════════════════════
MAGIC = b'NSVD'
TENS_F32 = 0
TENS_Q8 = 1
TENS_SVD_Q8 = 2
TENS_SVD_REC = 3
TENS_SVD_TERNARY = 4
TENS_Q1_0 = 5
TENS_Q4_0 = 6
TENS_Q5_0 = 7
TENS_Q6_0 = 8
TENS_Q4_K = 9
TENS_Q5_K = 10
TENS_Q6_K = 11
BLOCK_SIZE = 32

# Quant block geometry (matches ggml / q35_gguf.py exactly)
QK_K = 256
Q4_0_BLOCK_ELEMS = 32
Q4_0_BLOCK_BYTES = 18      # fp16 d + 16 bytes (32 x 4-bit)
Q5_0_BLOCK_ELEMS = 32
Q5_0_BLOCK_BYTES = 22      # fp16 d + 4 bytes qh + 16 bytes qs
Q6_0_BLOCK_ELEMS = 32
Q6_0_BLOCK_BYTES = 26      # 16 bytes ql + 8 bytes qh + fp16 d
Q4_K_BLOCK_BYTES = 144     # d + dmin + scales[12] + qs[128]
Q5_K_BLOCK_BYTES = 176     # d + dmin + scales[12] + qh[32] + qs[128]
Q6_K_BLOCK_BYTES = 210     # ql[128] + qh[64] + scales[16] + d


_CTYPES_OK = False


def _ensure_ctypes():
    """Workaround: some sealed exe runtimes drop ctypes from module globals."""
    global _CTYPES_OK
    if _CTYPES_OK:
        return
    if 'ctypes' not in globals():
        import ctypes as _ct
        globals()['ctypes'] = _ct
    _CTYPES_OK = True


def _dequant_q8(data, n, offset=0):
    """Vectorized Q8 dequant using structured dtype — ~100x faster."""
    n_blocks = (n + BLOCK_SIZE - 1) // BLOCK_SIZE
    total_bytes = n_blocks * (2 + BLOCK_SIZE)
    raw = np.frombuffer(data, dtype=np.uint8, count=total_bytes, offset=offset)
    # reshape to blocks: each block is [fp16_scale(2B), int8_data(32B)]
    blocks = raw.reshape(n_blocks, 2 + BLOCK_SIZE)
    scales_raw = blocks[:, :2].reshape(-1, 2).astype(np.uint8)
    # decode fp16 scales vectorized
    h = scales_raw[:, 0].astype(np.uint16) | (scales_raw[:, 1].astype(np.uint16) << 8)
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    scales = np.where(e == 0, m * 2.0 ** -24,
                      np.where(e == 31, np.where(m == 0, np.inf, np.nan),
                               (1.0 + m / 1024.0) * 2.0 ** (e.astype(np.int32) - 15)))
    scales = np.where(s, -scales, scales)
    # extract int8 data (must view as signed int8, not uint8!)
    i8 = blocks[:, 2:].view(np.int8).astype(np.float32)  # (n_blocks, 32)
    # dequant: i8 * scale_per_block
    out = (i8 * scales[:, None]).ravel()[:n]
    return out.astype(np.float32)


def _svd_unpack_factored(data, m, n):
    rank = struct.unpack_from('<I', data, 0)[0]
    if rank == 0:
        # Fallback: stored as plain Q8 (no .ntmre compression factorization)
        pos = 4
        w = _dequant_q8(data[pos:], m * n).reshape(m, n)
        return w.astype(np.float32), np.eye(n, dtype=np.float32)  # identity on right side
    pos = 4
    u_len = (2 + BLOCK_SIZE) * ((m * rank + BLOCK_SIZE - 1) // BLOCK_SIZE)
    v_len = (2 + BLOCK_SIZE) * ((rank * n + BLOCK_SIZE - 1) // BLOCK_SIZE)
    if len(data) >= 4 + u_len + 4 * rank + v_len:
        u_data = data[pos:pos + u_len]; pos += u_len
        s_data = data[pos:pos + 4 * rank]; pos += 4 * rank
        v_data = data[pos:]
        U = _dequant_q8(u_data, m * rank).reshape(m, rank)
        S = np.frombuffer(s_data, dtype=np.float32).copy()
        Vt = _dequant_q8(v_data, rank * n).reshape(rank, n)
        # Sanity: real weight matrices have singular values in the O(1..100)
        # range.  Non-finite or absurdly large S means the converter .ntmre compression'd
        # garbage (bad dequant upstream) — the file cannot run.
        if not np.all(np.isfinite(S)) or (S.size and float(np.median(np.abs(S))) > 1e4):
            raise ValueError(
                '.ntmre compression tensor is corrupt (singular values non-finite or ~1e6) '
                '— this file came from a broken conversion; re-convert from GGUF')
        # Dialect detection: some converters (bonsai_convert, rebuild_fast)
        # store U already scaled by S (pre-folded); others (nsvd_upgrade,
        # ntmre_format_py) store raw orthonormal U.  Column norms of U are
        # ~S for pre-folded and ~1 for raw.
        cn_med = float(np.median(np.linalg.norm(U, axis=0)))
        s_med = float(np.median(np.abs(S))) if S.size else 1.0
        if cn_med > 3.0 and cn_med / max(s_med, 1e-30) < 3.0:
            # pre-folded: S already folded into U — do not apply it again
            return U.astype(np.float32), Vt.astype(np.float32)
        return (U * S[np.newaxis, :]).astype(np.float32), Vt.astype(np.float32)
    a_data = data[pos:pos + u_len]; pos += u_len
    b_data = data[pos:]
    A = _dequant_q8(a_data, m * rank).reshape(m, rank)
    B = _dequant_q8(b_data, rank * n).reshape(rank, n)
    return A.astype(np.float32), B.astype(np.float32)


def _dequant_q4(data, n):
    """fp16 scale + 32 x 4-bit per 32 elems (Q4_0 style)."""
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    i = 0
    while i < n:
        scale = struct.unpack_from('<e', data, pos)[0]
        pos += 2
        packed = np.frombuffer(data, dtype=np.uint8, count=16, offset=pos)
        pos += 16
        lo = packed & 15
        hi = (packed >> 4) & 15
        q = np.stack([lo, hi], axis=1).ravel().astype(np.int8) - 8
        cnt = min(32, n - i)
        out[i:i + cnt] = q[:cnt].astype(np.float32) * float(scale)
        i += 32
    return out


def _unpack_ternary(data, n):
    """packed 2-bit {-1,0,+1} -> int8 array of length n."""
    raw = np.frombuffer(data, dtype=np.uint8, count=(n + 3) // 4)
    code = np.empty((len(raw), 4), dtype=np.uint8)
    code[:, 0] = raw & 3
    code[:, 1] = (raw >> 2) & 3
    code[:, 2] = (raw >> 4) & 3
    code[:, 3] = (raw >> 6) & 3
    code = code.ravel()[:n]
    return np.where(code == 0, -1, np.where(code == 2, 1, 0)).astype(np.int8)


def _fp16_to_fp32_vec(h):
    """Vectorized fp16 (uint16 array) -> fp32. h is cast to signed int so
    `e - 15` does not underflow."""
    h = h.astype(np.int32)
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    m = h & 0x3FF
    v = np.where(e == 0, m * 2.0 ** -24,
                 np.where(e == 31, np.where(m == 0, np.inf, np.nan),
                          (1 + m / 1024.0) * 2.0 ** (e - 15)))
    return np.where(s == 1, -v, v)


def _dequant_q1_0(data, n):
    """Q1_0_g128 dequant: 128 elems/block, 18 bytes/block.
    Layout: fp16 scale[0:2] + bits[2:18]. value = (2*bit - 1) * scale.
    (scale is at byte offset 0 — matches q35_gguf.py and the C engine.)"""
    Q1_BLOCK = 128
    Q1_BYTES = 18
    n_blocks = (n + Q1_BLOCK - 1) // Q1_BLOCK
    raw = np.frombuffer(data, dtype=np.uint8, count=n_blocks * Q1_BYTES)
    blk = raw.reshape(n_blocks, Q1_BYTES)
    h = blk[:, 0:2].astype(np.uint16)
    d = _fp16_to_fp32_vec(h[:, 0] | (h[:, 1] << 8)).astype(np.float32)
    packed = blk[:, 2:18]
    unpacked = np.unpackbits(packed.ravel(), bitorder='little')
    bits_u = unpacked.reshape(n_blocks, Q1_BLOCK).astype(np.float32)
    out = (2.0 * bits_u - 1.0) * d[:, np.newaxis]
    return out.ravel()[:n]


def _dequant_q4_0(data, n):
    """Q4_0: 32-elem blocks, fp16 d[0:2] + 16 bytes (32 x 4-bit).
    y = d * (q - 8). ggml split nibble ordering (low 0..15, high 16..31)."""
    n_blocks = n // Q4_0_BLOCK_ELEMS
    raw = np.frombuffer(data, dtype=np.uint8)
    blk = raw.reshape(n_blocks, Q4_0_BLOCK_BYTES)
    d = _fp16_to_fp32_vec(
        blk[:, 0].astype(np.uint16) | (blk[:, 1].astype(np.uint16) << 8)
    ).astype(np.float32)
    qs = blk[:, 2:18].astype(np.int32)
    lo = qs & 0xF
    hi = qs >> 4
    q = np.concatenate([lo, hi], axis=1).astype(np.float32) - 8.0
    return (d[:, None] * q).ravel()[:n]


def _dequant_q5_0(data, n):
    """Q5_0: 32-elem blocks, fp16 d[0:2] + qh[4] + qs[16].
    y = d * (q4 + high*16 - 16)."""
    n_blocks = n // Q5_0_BLOCK_ELEMS
    raw = np.frombuffer(data, dtype=np.uint8)
    blk = raw.reshape(n_blocks, Q5_0_BLOCK_BYTES)
    d = _fp16_to_fp32_vec(
        blk[:, 0].astype(np.uint16) | (blk[:, 1].astype(np.uint16) << 8)
    ).astype(np.float32)
    qh = blk[:, 2:6]
    qs = blk[:, 6:22]
    lo = qs & 0xF
    hi = qs >> 4
    q4 = np.concatenate([lo, hi], axis=1)
    high = np.unpackbits(qh.ravel(), bitorder='little').reshape(n_blocks, 32)
    q = (q4 + high * 16).astype(np.float32) - 16.0
    return (d[:, None] * q).ravel()[:n]


def _dequant_q6_0(data, n):
    """Q6_0: 32-elem blocks, ql[16] + qh[8] + fp16 d[24:26].
    y = d * (q - 32)."""
    n_blocks = n // Q6_0_BLOCK_ELEMS
    raw = np.frombuffer(data, dtype=np.uint8)
    blk = raw.reshape(n_blocks, Q6_0_BLOCK_BYTES)
    ql = blk[:, 0:16]
    qh = blk[:, 16:24]
    d = _fp16_to_fp32_vec(
        blk[:, 24].astype(np.uint16) | (blk[:, 25].astype(np.uint16) << 8)
    ).astype(np.float32)
    lo = ql & 0xF
    hi = ql >> 4
    q4 = np.concatenate([lo, hi], axis=1)
    bits = np.unpackbits(qh.ravel(), bitorder='little').reshape(n_blocks, 64)
    high = bits[:, 0::2] + 2 * bits[:, 1::2]
    q = (q4 + high * 16).astype(np.float32) - 32.0
    return (d[:, None] * q).ravel()[:n]


def _get_scale_min_k4(j, scales):
    """ggml get_scale_min_k4: j<4 -> 6-bit (scales[j]&63, scales[j+4]&63);
    j>=4 -> 4-bit packed in high bits of scales[j-4] and nibbles of scales[j+4]."""
    if j < 4:
        return scales[:, j] & 63, scales[:, j + 4] & 63
    sc = (scales[:, j + 4] & 0xF) | ((scales[:, j - 4] >> 6) << 4)
    m = (scales[:, j + 4] >> 4) | ((scales[:, j] >> 6) << 4)
    return sc, m


def _dequant_q4_k(data, n):
    """Q4_K: 256-elem super-blocks. d[0:2] dmin[2:4] scales[12] qs[128].
    y = d*scale*q - dmin*minscale."""
    n_blocks = n // QK_K
    raw = np.frombuffer(data, dtype=np.uint8)
    blk = raw.reshape(n_blocks, Q4_K_BLOCK_BYTES)
    d = _fp16_to_fp32_vec(
        blk[:, 0].astype(np.uint16) | (blk[:, 1].astype(np.uint16) << 8)
    ).astype(np.float32)
    dmin = _fp16_to_fp32_vec(
        blk[:, 2].astype(np.uint16) | (blk[:, 3].astype(np.uint16) << 8)
    ).astype(np.float32)
    scales = blk[:, 4:16]
    qs = blk[:, 16:144]

    sc = np.zeros((n_blocks, 8), dtype=np.float32)
    m = np.zeros((n_blocks, 8), dtype=np.float32)
    for j in range(8):
        sc[:, j], m[:, j] = _get_scale_min_k4(j, scales)

    d_scaled = d[:, None] * sc
    m_scaled = dmin[:, None] * m
    qs_low = qs & 0xF
    qs_high = qs >> 4

    result = np.zeros((n_blocks, QK_K), dtype=np.float32)
    for s in range(8):
        group = s // 2
        q_vals = (qs_low[:, group * 32:group * 32 + 32] if s % 2 == 0
                  else qs_high[:, group * 32:group * 32 + 32]).astype(np.float32)
        result[:, s * 32:(s + 1) * 32] = d_scaled[:, s:s + 1] * q_vals - m_scaled[:, s:s + 1]
    return result.ravel()[:n]


def _dequant_q5_k(data, n):
    """Q5_K: 256-elem super-blocks. d[0:2] dmin[2:4] scales[12] qh[32] qs[128].
    y = d*scale*(q4 + high*16) - dmin*minscale."""
    n_blocks = n // QK_K
    raw = np.frombuffer(data, dtype=np.uint8)
    blk = raw.reshape(n_blocks, Q5_K_BLOCK_BYTES)
    d = _fp16_to_fp32_vec(
        blk[:, 0].astype(np.uint16) | (blk[:, 1].astype(np.uint16) << 8)
    ).astype(np.float32)
    dmin = _fp16_to_fp32_vec(
        blk[:, 2].astype(np.uint16) | (blk[:, 3].astype(np.uint16) << 8)
    ).astype(np.float32)
    scales = blk[:, 4:16]
    qh = blk[:, 16:48]
    qs = blk[:, 48:176]

    sc = np.zeros((n_blocks, 8), dtype=np.float32)
    m = np.zeros((n_blocks, 8), dtype=np.float32)
    for j in range(8):
        sc[:, j], m[:, j] = _get_scale_min_k4(j, scales)

    d_scaled = d[:, None] * sc
    m_scaled = dmin[:, None] * m
    qs_low = qs & 0xF
    qs_high = qs >> 4

    result = np.zeros((n_blocks, QK_K), dtype=np.float32)
    l = np.arange(32)
    for s in range(8):
        group = s // 2
        is_high = s % 2
        q4 = qs_low[:, group * 32:group * 32 + 32] if not is_high else qs_high[:, group * 32:group * 32 + 32]
        bit = 2 * group + is_high
        high = ((qh[:, l] >> bit) & 1) * 16
        q5 = (q4 + high).astype(np.float32)
        result[:, s * 32:(s + 1) * 32] = d_scaled[:, s:s + 1] * q5 - m_scaled[:, s:s + 1]
    return result.ravel()[:n]


def _dequant_q6_k(data, n):
    """Q6_K: 256-elem super-blocks. ql[128] qh[64] scales[16] d[208:210].
    y = d*scale*(q - 32)."""
    n_blocks = n // QK_K
    raw = np.frombuffer(data, dtype=np.uint8)
    blk = raw.reshape(n_blocks, Q6_K_BLOCK_BYTES)
    ql = blk[:, 0:128]
    qh = blk[:, 128:192]
    sc = blk[:, 192:208].astype(np.int8).astype(np.float32)
    d = _fp16_to_fp32_vec(
        blk[:, 208].astype(np.uint16) | (blk[:, 209].astype(np.uint16) << 8)
    ).astype(np.float32)

    result = np.zeros((n_blocks, QK_K), dtype=np.float32)
    is_idx = np.arange(32) // 16
    for n2 in range(2):
        ql_g = ql[:, n2 * 64:(n2 + 1) * 64]
        qh_g = qh[:, n2 * 32:(n2 + 1) * 32]
        sc_g = sc[:, n2 * 8:(n2 + 1) * 8]

        q1 = ((ql_g[:, 0:32] & 0xF) | ((qh_g & 3) << 4)).astype(np.float32) - 32.0
        q2 = ((ql_g[:, 32:64] & 0xF) | (((qh_g >> 2) & 3) << 4)).astype(np.float32) - 32.0
        q3 = ((ql_g[:, 0:32] >> 4) | (((qh_g >> 4) & 3) << 4)).astype(np.float32) - 32.0
        q4 = ((ql_g[:, 32:64] >> 4) | (((qh_g >> 6) & 3) << 4)).astype(np.float32) - 32.0

        y1 = d[:, None] * sc_g[:, is_idx + 0] * q1
        y2 = d[:, None] * sc_g[:, is_idx + 2] * q2
        y3 = d[:, None] * sc_g[:, is_idx + 4] * q3
        y4 = d[:, None] * sc_g[:, is_idx + 6] * q4

        result[:, n2 * 128:n2 * 128 + 32] = y1
        result[:, n2 * 128 + 32:n2 * 128 + 64] = y2
        result[:, n2 * 128 + 64:n2 * 128 + 96] = y3
        result[:, n2 * 128 + 96:n2 * 128 + 128] = y4
    return result.ravel()[:n]


def _ratio_decode_s(data, r):
    """ratio-encoded singular values: f32 sigma_1 + r x uint8 ratios."""
    s0 = struct.unpack_from('<f', data, 0)[0]
    ratios = np.frombuffer(data, dtype=np.uint8, count=r, offset=4).astype(np.float32) / 255.0
    S = np.empty(r, dtype=np.float32)
    S[0] = s0
    for i in range(1, r):
        S[i] = S[i - 1] * ratios[i]
    return S


def _svd_unpack_recursive(data, m, n):
    """Recursive residual .ntmre compression (Q8 + Q4 + ternary) -> single factored (A, B).

    Folds the three terms into one pair so the existing factored matmul
    (A @ B) works unchanged:  A = [U1*S1 | U2*S2 | U3*S3], B = [V1; V2; V3].
    """
    r1, r2, r3, _m, _n = struct.unpack_from('<IIIII', data, 0)
    pos = 20
    q8_len = lambda k: (2 + BLOCK_SIZE) * ((k + BLOCK_SIZE - 1) // BLOCK_SIZE)
    q4_len = lambda k: 18 * ((k + 31) // 32)
    tern_len = lambda k: (k + 3) // 4

    u1 = _dequant_q8(data[pos:pos + q8_len(m * r1)], m * r1).reshape(m, r1)
    pos += q8_len(m * r1)
    s1 = _ratio_decode_s(data[pos:pos + 4 + r1], r1); pos += 4 + r1
    v1 = _dequant_q8(data[pos:pos + q8_len(r1 * n)], r1 * n).reshape(r1, n)
    pos += q8_len(r1 * n)

    u2 = _dequant_q4(data[pos:pos + q4_len(m * r2)], m * r2).reshape(m, r2)
    pos += q4_len(m * r2)
    s2 = _ratio_decode_s(data[pos:pos + 4 + r2], r2); pos += 4 + r2
    v2 = _dequant_q4(data[pos:pos + q4_len(r2 * n)], r2 * n).reshape(r2, n)
    pos += q4_len(r2 * n)

    u3 = _unpack_ternary(data[pos:pos + tern_len(m * r3)], m * r3).reshape(m, r3).astype(np.float32)
    pos += tern_len(m * r3)
    s3 = _ratio_decode_s(data[pos:pos + 4 + r3], r3); pos += 4 + r3
    v3 = _unpack_ternary(data[pos:pos + tern_len(r3 * n)], r3 * n).reshape(r3, n).astype(np.float32)

    A = np.concatenate([u1 * s1[np.newaxis, :], u2 * s2[np.newaxis, :], u3 * s3[np.newaxis, :]], axis=1)
    B = np.concatenate([v1, v2, v3], axis=0)
    return A.astype(np.float32), B.astype(np.float32)


def _svd_unpack_ternary(data, m, n):
    """ternary factors: rank + packed U (2-bit) + f32 S' + packed Vt (2-bit)."""
    rank = struct.unpack_from('<I', data, 0)[0]
    pos = 4
    u_len = (m * rank + 3) // 4
    u = _unpack_ternary(data[pos:pos + u_len], m * rank).reshape(m, rank).astype(np.float32)
    pos += u_len
    S = np.frombuffer(data[pos:pos + 4 * rank], dtype=np.float32).copy(); pos += 4 * rank
    v_len = (rank * n + 3) // 4
    v = _unpack_ternary(data[pos:pos + v_len], rank * n).reshape(rank, n).astype(np.float32)
    return (u * S[np.newaxis, :]).astype(np.float32), v.astype(np.float32)


class NSVDStream:
    """mmap + load-on-demand NSVD reader (the streaming system)."""

    def __init__(self, path):
        self.path = str(path)
        self._f = open(self.path, 'rb')
        self._mmap = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self._parse_header()
        self._compute_offsets()
        self._cache = {}

    def _parse_header(self):
        m = self._mmap
        # Accept all NTMRE container magics: NSVD (standard), NTMR/RMTN
        # (legacy writers) — the layout after the magic is identical.
        if m[:4] not in (MAGIC, b'NTMR', b'RMTN'):
            raise ValueError("Not an NTMRE .ntmre compression file")
        pos = 4
        self.version, self.n_tensors = struct.unpack_from('<II', m, pos); pos += 8
        cfg_len = struct.unpack_from('<I', m, pos)[0]; pos += 4
        self.config = json.loads(m[pos:pos + cfg_len]); pos += cfg_len
        tok_len = struct.unpack_from('<I', m, pos)[0]; pos += 4
        self.tokenizer_bytes = bytes(m[pos:pos + tok_len]); pos += tok_len
        idx_len = struct.unpack_from('<I', m, pos)[0]; pos += 4
        index = m[pos:pos + idx_len]; pos += idx_len
        self._index_end = pos
        self.entries = {}
        p = 0
        for _ in range(self.n_tensors):
            nl = index[p]; p += 1
            name = index[p:p + nl].decode('utf-8'); p += nl
            ttype = index[p]; p += 1
            nd = index[p]; p += 1
            shape = struct.unpack_from('<%dI' % nd, index, p); p += 4 * nd
            data_len = struct.unpack_from('<Q', index, p)[0]; p += 8
            self.entries[name] = (ttype, tuple(shape), data_len)

    def _compute_offsets(self):
        self._data_start = (self._index_end + 63) & ~63
        self.offsets = {}
        off = self._data_start
        for name, (ttype, shape, data_len) in self.entries.items():
            self.offsets[name] = off
            # Writer aligns START of each tensor to 64B, so advance by
            # raw data_len then round up for the next tensor's alignment.
            off += data_len
            off = (off + 63) & ~63

    def load_tensor(self, name):
        if name not in self.entries:
            return None
        if name in self._cache:
            return self._cache[name]
        ttype, shape, data_len = self.entries[name]
        off = self.offsets[name]
        data = bytes(self._mmap[off:off + data_len])
        if ttype == TENS_F32:
            w = np.frombuffer(data, dtype=np.float32).reshape(shape)
        elif ttype == TENS_Q8:
            w = _dequant_q8(data, int(np.prod(shape))).reshape(shape)
        elif ttype == TENS_SVD_Q8:
            w = _svd_unpack_factored(data, int(shape[0]), int(shape[1]))
        elif ttype == TENS_SVD_REC:
            w = _svd_unpack_recursive(data, int(shape[0]), int(shape[1]))
        elif ttype == TENS_SVD_TERNARY:
            w = _svd_unpack_ternary(data, int(shape[0]), int(shape[1]))
        elif ttype == TENS_Q1_0:
            w = _dequant_q1_0(data, int(np.prod(shape))).reshape(shape)
        elif ttype == TENS_Q4_0:
            w = _dequant_q4_0(data, int(np.prod(shape))).reshape(shape)
        elif ttype == TENS_Q5_0:
            w = _dequant_q5_0(data, int(np.prod(shape))).reshape(shape)
        elif ttype == TENS_Q6_0:
            w = _dequant_q6_0(data, int(np.prod(shape))).reshape(shape)
        elif ttype == TENS_Q4_K:
            w = _dequant_q4_k(data, int(np.prod(shape))).reshape(shape)
        elif ttype == TENS_Q5_K:
            w = _dequant_q5_k(data, int(np.prod(shape))).reshape(shape)
        elif ttype == TENS_Q6_K:
            w = _dequant_q6_k(data, int(np.prod(shape))).reshape(shape)
        else:
            raise ValueError("Unknown tensor type %d for %s" % (ttype, name))
        self._cache[name] = w
        return w

    def close(self):
        try:
            self._mmap.close()
        except Exception:
            pass
        try:
            self._f.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# NVRTC CUDA wrapper — self-contained (no torch, no nvcc)
# ═══════════════════════════════════════════════════════════════════
# Resolve NVRTC DLL path: try bundled (exe/script dir) first,
# then CUDA_PATH, then system search.  This lets the hardened
# exe run on machines without the CUDA toolkit installed.
#
# ALSO: NVRTC dlopens its own "nvrtc-builtins64_<ver>.dll" by *bare name*
# at compile() time.  The bundled nvrtc64_120_0.dll is really NVRTC 12.6
# (nvrtcVersion reports 12.6) so it wants nvrtc-builtins64_126.dll, which
# ships next to it in the suite root.  Bare-name LoadLibrary does NOT search
# the suite root -- it searches python.exe's dir and System32 -- so without
# this registration every nvrtcCompileProgram fails with
#   "failed to open nvrtc-builtins64_126.dll"
# and the whole engine silently degrades to the CPU path.
# Keep a strong reference to the handles: if they are garbage collected,
# Windows removes those directories from the process DLL search path again.
_DLL_DIR_HANDLES = []


def _register_dll_dir(path):
    """Add `path` to the process DLL search path for the process lifetime."""
    if not path or not os.path.isdir(path):
        return
    if os.name == 'nt' and hasattr(os, 'add_dll_directory'):
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(path))
            return
        except (OSError, AttributeError):
            pass
    try:  # legacy fallback: make bare-name loads find it too
        if path.lower() not in os.environ.get('PATH', '').lower().split(os.pathsep):
            os.environ['PATH'] = path + os.pathsep + os.environ.get('PATH', '')
    except Exception:
        pass


def _suite_root():
    """Directory that holds the bundled DLLs (suite root, or exe dir if frozen)."""
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # parent of engine/


def _find_nvrtc_dll():
    # 1. Same directory as the main script/exe, plus common subdirs
    base = _suite_root()
    for cand in (base, os.path.join(base, '_internal'), os.path.join(base, 'engine')):
        # NVRTC's builtins must be resolvable by BARE name, so register every
        # directory we might load an nvrtc DLL from.
        _register_dll_dir(cand)
        p = os.path.join(cand, 'nvrtc64_120_0.dll')
        if os.path.isfile(p):
            return p
    # 2. CUDA_PATH environment variable
    cpath = os.environ.get("CUDA_PATH", r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
    p = os.path.join(cpath, 'bin', 'nvrtc64_120_0.dll')
    if os.path.isfile(p):
        return p
    # 3. Let ctypes search the system PATH
    return 'nvrtc64_120_0.dll'

CUDA_PATH = os.environ.get("CUDA_PATH", r"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
NVRTC = _find_nvrtc_dll()
DRIVER = "C:/Windows/System32/nvcuda.dll"

KERNEL_SRC = r"""
extern "C" __global__ void matmul(
    const float* A, const float* B, float* C,
    int M, int K, int N)
{
    __shared__ float As[16][17];
    __shared__ float Bs[16][17];
    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < (K + 15) / 16; t++) {
        int a_col = t * 16 + threadIdx.x;
        int b_row = t * 16 + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row*K + a_col] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row*N + col] : 0.0f;
        __syncthreads();
        for (int i = 0; i < 16; i++) sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row*N + col] = sum;
}
"""


# CUDA's "current context" is per-THREAD global state shared by every Cuda
# object living on that thread.  Caching it per-object is wrong: with two
# engines on one thread (dual-GPU / fused) whichever bound last wins, and the
# other engine's allocations land on the WRONG card.  That shows up as
# cuMemAlloc_v2 failing with error 700 while nvidia-smi still reports plenty
# of free VRAM.  Track the bound context per thread instead.
_CTX_BOUND = {}
_get_ident = threading.get_ident
_CTX_VALS = {}


def _bind_ctx(cu, ctx, force=False, tid=None):
    """Make `ctx` current on this thread if it isn't already. True if bound."""
    if tid is None:
        tid = _get_ident()
    want = ctx.value if ctx is not None else None
    if force or _CTX_BOUND.get(tid) != want:
        cu.cuCtxSetCurrent(ctx)
        _CTX_BOUND[tid] = want
        return True
    return False


class Cuda:
    def __init__(self, device=0):
        self.nvrtc = ctypes.CDLL(NVRTC)
        self.cu = ctypes.CDLL(DRIVER)
        r = self.cu.cuInit(0)
        if r != 0:
            raise RuntimeError("cuInit failed: %d" % r)
        self.dev = ctypes.c_int(device)
        self.ctx = ctypes.c_void_p()
        # Use the _v2 driver API entry points.  The legacy cuCtxCreate_v2 /
        # cuMemAlloc_v2 create a 32-bit-address context capped at ~4 GB even on
        # 64-bit Windows, which is why the suite could only allocate ~3.4 GB
        # of a 12 GB GPU.  The _v2 variants enable the full 64-bit address
        # space and report correct free/total memory.
        r = self.cu.cuCtxCreate_v2(ctypes.byref(self.ctx), 0, self.dev)
        if r != 0:
            raise RuntimeError("cuCtxCreate_v2 failed: %d" % r)
        self.cu.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self.cu.cuCtxSetCurrent.restype = ctypes.c_int
        _bind_ctx(self.cu, self.ctx, force=True)
        self.cu.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t]
        self.cu.cuMemcpyHtoD_v2.argtypes = [ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_size_t]
        self.cu.cuMemcpyDtoH_v2.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t]
        self.cu.cuMemcpyDtoD_v2.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_size_t]
        self.cu.cuMemFree_v2.argtypes = [ctypes.c_ulonglong]
        self._ctx_tid = -1  # thread id that has the context bound
        self._ctx_val = self.ctx.value  # bound context handle, for the fast path
        # Create a dedicated CUDA stream (required for graph capture —
        # the default stream (NULL) does not support stream capture).
        self.stream = ctypes.c_void_p()
        self.cu.cuStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        self.cu.cuStreamCreate(ctypes.byref(self.stream), 0)
        # Graph API bindings
        self._graph_exec = None
        self._graph_keep = []  # keep ctypes args alive for graph lifetime
        self.cu.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.cu.cuStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.cu.cuStreamEndCapture.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.cu.cuGraphInstantiate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulonglong]
        self.cu.cuGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.cu.cuGraphDestroy.argtypes = [ctypes.c_void_p]
        self.cu.cuGraphExecDestroy.argtypes = [ctypes.c_void_p]
        self.mod = self.compile(KERNEL_SRC)
        self.f_matmul = self.func(self.mod, "matmul")

    def compile(self, src):
        prog = ctypes.c_void_p()
        self.nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src.encode(), b"k.cu", 0, None, None)
        # Detect the device's compute capability so NVRTC generates code
        # targeted at the actual GPU arch (better instruction scheduling,
        # register allocation, and FMA use than generic PTX + driver JIT).
        arch_flag = None
        try:
            major = ctypes.c_int(0)
            minor = ctypes.c_int(0)
            self.cu.cuDeviceGetAttribute.argtypes = [
                ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
            self.cu.cuDeviceGetAttribute(ctypes.byref(major), 75, self.dev)   # COMPUTE_CAPABILITY_MAJOR
            self.cu.cuDeviceGetAttribute(ctypes.byref(minor), 76, self.dev)   # COMPUTE_CAPABILITY_MINOR
            if major.value > 0:
                arch_flag = ("--gpu-architecture=compute_%d%d"
                             % (major.value, minor.value)).encode()
        except Exception:
            pass
        opt_list = [b"--std=c++17", b"-use_fast_math", b"--fmad=true", b"-extra-device-vectorization"]
        if arch_flag:
            opt_list.append(arch_flag)
        n = len(opt_list)
        opts = (ctypes.c_char_p * n)(*opt_list)
        r = self.nvrtc.nvrtcCompileProgram(prog, n, opts)
        if r != 0:
            sz = ctypes.c_size_t()
            self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(sz))
            buf = ctypes.create_string_buffer(sz.value)
            self.nvrtc.nvrtcGetProgramLog(prog, buf)
            raise RuntimeError("NVRTC compile failed:\n" + buf.value.decode())
        psz = ctypes.c_size_t()
        self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(psz))
        ptx = ctypes.create_string_buffer(psz.value)
        self.nvrtc.nvrtcGetPTX(prog, ptx)
        mod = ctypes.c_void_p()
        self.cu.cuModuleLoadDataEx(ctypes.byref(mod), ptx, 0, None, None)
        return mod

    def func(self, mod, name):
        fn = ctypes.c_void_p()
        self.cu.cuModuleGetFunction(ctypes.byref(fn), mod, name.encode())
        return fn

    def _ensure_ctx(self):
        # Contexts are thread-local (the GUI builds the engine on the main
        # thread but generates on a worker), AND per-thread state is shared by
        # every engine on that thread -- so bind through _bind_ctx rather than
        # caching the thread id on this object.
        #
        # Fast path first. This runs on every single GPU call -- ~4300 times a
        # token -- and the answer changes only when another engine on this
        # thread has taken the context since we last checked. `_CTX_BOUND` is
        # exactly that record, so one dict lookup replaces the import, the
        # second get_ident, and the nested call this used to do every time.
        tid = _get_ident()
        if self._ctx_tid == tid and _CTX_BOUND.get(tid) == self._ctx_val:
            return
        if _bind_ctx(self.cu, self.ctx, tid=tid):
            self._ctx_tid = tid

    def alloc(self, nbytes):
        _ensure_ctypes()
        self._ensure_ctx()
        ptr = ctypes.c_ulonglong()
        r = self.cu.cuMemAlloc_v2(ctypes.byref(ptr), nbytes)
        if r != 0:
            raise RuntimeError("cuMemAlloc_v2 failed (%d) for %.0fMB" % (r, nbytes / 1e6))
        return ptr

    def to_gpu(self, arr):
        _ensure_ctypes()
        self._ensure_ctx()
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        ptr = self.alloc(arr.nbytes)
        r = self.cu.cuMemcpyHtoD_v2(ptr, ctypes.c_void_p(arr.ctypes.data), arr.nbytes)
        if r != 0:
            raise RuntimeError("cuMemcpyHtoD_v2 failed (%d)" % r)
        return ptr, arr.shape

    def from_gpu(self, ptr, shape):
        _ensure_ctypes()
        self._ensure_ctx()
        out = np.empty(int(np.prod(shape)), dtype=np.float32)
        r = self.cu.cuMemcpyDtoH_v2(ctypes.c_void_p(out.ctypes.data), ptr, out.nbytes)
        if r != 0:
            raise RuntimeError("cuMemcpyDtoH_v2 failed (%d)" % r)
        return out.reshape(shape)

    def free(self, ptr):
        _ensure_ctypes()
        self._ensure_ctx()
        self.cu.cuMemFree_v2(ptr)

    def launch(self, fn, grid, block, args, shared_mem=0):
        _ensure_ctypes()
        # Cheap no-op when this engine's context is already current on this
        # thread; rebinds when another engine on the same thread took over.
        self._ensure_ctx()
        params = []
        keep = []
        c_ulonglong = ctypes.c_ulonglong
        c_float = ctypes.c_float
        c_int = ctypes.c_int
        addressof = ctypes.addressof
        for a in args:
            ta = type(a)
            if ta is c_ulonglong:
                # device pointer (already c_ulonglong).  MUST be kept alive
                # for graph capture — the graph stores a pointer to this
                # object, so if it is GC'd the replay reads freed memory.
                keep.append(a)
                params.append(ctypes.c_void_p(addressof(a)))
            elif ta is c_float:
                keep.append(a)
                params.append(ctypes.c_void_p(addressof(a)))
            elif ta is c_int:
                # pre-created persistent c_int (for graph replay updates)
                keep.append(a)
                params.append(ctypes.c_void_p(addressof(a)))
            elif ta is float:
                # kernel scalar float (eps, rope_base, scale, ...) — must be
                # passed as c_float, NOT c_int, or the bit pattern is wrong
                # (e.g. rope_base=10000.0 as int bits → 1.4e-41 → powf → NaN)
                cf = c_float(a)
                keep.append(cf)
                params.append(ctypes.c_void_p(addressof(cf)))
            else:
                ci = c_int(int(a))
                keep.append(ci)
                params.append(ctypes.c_void_p(addressof(ci)))
        args_arr = (ctypes.c_void_p * len(params))(*params)
        # Use the dedicated stream (required for CUDA graph capture).
        r = self.cu.cuLaunchKernel(fn, grid[0], grid[1], 1, block[0], block[1], 1,
                                    shared_mem, self.stream, args_arr, None)
        if r != 0 and getattr(self, '_capturing', False):
            # A failed launch during capture invalidates the graph.
            self._capturing = False
            raise RuntimeError('cuLaunchKernel failed during capture: %d' % r)
        # In capture mode, keep all ctypes objects alive for graph lifetime.
        if self._graph_exec is None and getattr(self, '_capturing', False):
            self._graph_keep.append((args_arr, keep))

    def begin_capture(self, mode=2):
        """Begin CUDA graph capture on the dedicated stream.
        mode: 0=GLOBAL, 1=THREAD_LOCAL, 2=RELAXED (default — most permissive)."""
        self._capturing = True
        self._graph_keep = []
        r = self.cu.cuStreamBeginCapture(self.stream, mode)
        if r != 0:
            self._capturing = False
            raise RuntimeError('cuStreamBeginCapture failed: %d' % r)

    def end_capture(self):
        """End capture, instantiate the graph.  Returns True on success."""
        graph = ctypes.c_void_p()
        r = self.cu.cuStreamEndCapture(self.stream, ctypes.byref(graph))
        self._capturing = False
        if r != 0 or not graph.value:
            print('[GRAPH] cuStreamEndCapture failed: r=%d graph=%s keep=%d' %
                  (r, graph.value, len(self._graph_keep)), flush=True)
            return False
        self._graph_exec = ctypes.c_void_p()
        r = self.cu.cuGraphInstantiate(ctypes.byref(self._graph_exec),
                                       graph, None, None, 0)
        if r != 0:
            print('[GRAPH] cuGraphInstantiate failed: r=%d' % r, flush=True)
            self.cu.cuGraphDestroy(graph)
            self._graph_exec = None
            return False
        self.cu.cuGraphDestroy(graph)
        print('[GRAPH] Captured %d launches, graph exec=%s' %
              (len(self._graph_keep), self._graph_exec.value), flush=True)
        return True

    def replay_graph(self):
        """Replay the captured graph on the dedicated stream."""
        if self._graph_exec is None:
            return False
        r = self.cu.cuGraphLaunch(self._graph_exec, self.stream)
        return r == 0

    def sync(self):
        """Synchronize the dedicated stream."""
        self.cu.cuStreamSynchronize(self.stream)

    def destroy_graph(self):
        """Free the graph exec and its kept args."""
        if self._graph_exec is not None:
            self.cu.cuGraphExecDestroy(self._graph_exec)
            self._graph_exec = None
        self._graph_keep = []

    def close(self):
        """Destroy CUDA context, stream, and graph exec to completely free VRAM."""
        self.destroy_graph()
        if getattr(self, 'stream', None) is not None and getattr(self.stream, 'value', 0):
            try:
                self.cu.cuStreamDestroy(self.stream)
            except Exception:
                pass
            self.stream = None
        if getattr(self, 'ctx', None) is not None and getattr(self.ctx, 'value', 0):
            try:
                self.cu.cuCtxDestroy_v2(self.ctx)
            except Exception:
                pass
            self.ctx = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# Math helpers
# ═══════════════════════════════════════════════════════════════════
def _rmsnorm(x, w, eps):
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * w.astype(np.float32)

def _softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)

def _silu(x):
    return x / (1.0 + np.exp(-x))

def _softplus(x):
    return np.log1p(np.exp(x))


# ═══════════════════════════════════════════════════════════════════
# Minimal tokenizer (self-contained)
# ═══════════════════════════════════════════════════════════════════
class Tokenizer:
    def __init__(self, tokenizer_bytes):
        self.tokens = []
        self.bos = 1
        self.eos = 2
        self.pad = 0
        if tokenizer_bytes:
            try:
                d = json.loads(tokenizer_bytes.decode('utf-8'))
                self.tokens = list(d.get('tokens', []))
                self.bos = int(d.get('bos', 1))
                self.eos = int(d.get('eos', 2))
                self.pad = int(d.get('pad', 0))
            except Exception:
                pass
        self._id = {t: i for i, t in enumerate(self.tokens)}

    def encode(self, text):
        # greedy longest-match over token list (best-effort)
        ids = []
        i = 0
        n = len(text)
        while i < n:
            matched = None
            for j in range(min(32, n - i), 0, -1):
                piece = text[i:i + j]
                if piece in self._id:
                    matched = piece
                    break
            if matched is None:
                matched = text[i]
                i += 1
            else:
                i += len(matched)
            ids.append(self._id.get(matched, 0))
        return ids

    def decode(self, ids):
        return "".join(self.tokens[t] if 0 <= t < len(self.tokens) else "" for t in ids)


# ═══════════════════════════════════════════════════════════════════
# Qwen3.5 hybrid engine (self-contained, GPU matmul)
# ═══════════════════════════════════════════════════════════════════
class BonsaiEngine:
    def __init__(self, model_path, device=0, use_gpu=True, devices=None, layer_split=None):
        self._s = NSVDStream(model_path)
        self.config = self._s.config
        self.use_gpu = use_gpu
        # multi-GPU: one Cuda context per device
        self.devices = list(devices) if devices else [device]
        self.cudas = [Cuda(d) for d in self.devices] if use_gpu else []
        self.layer_split = layer_split
        self.kv_cache = {}
        self.ssm_state = {}
        self._gpu_cache = {}
        self._gpu_cuda = {}   # layer -> cuda index

        self.dim = int(self.config.get('dim', 4096))
        self.n_layers = int(self.config.get('n_layers', 32))
        self.n_heads = int(self.config.get('n_heads', 32))
        self.n_kv = int(self.config.get('n_kv_heads', self.n_heads))
        self.head_dim = int(self.config.get('head_dim', self.dim // self.n_heads))
        self.ffn_dim = int(self.config.get('ffn_dim', 4 * self.dim))
        self.vocab_size = int(self.config.get('vocab_size', 32000))
        self.rope_theta = float(self.config.get('rope_theta', 10000.0))
        self.norm_eps = float(self.config.get('norm_eps', 1e-5))

        self.full_attn_interval = int(self.config.get('full_attn_interval', 4))
        self.rope_dim = int(self.config.get('rope_dim', self.head_dim))
        self.ssm_conv_kernel = int(self.config.get('ssm_conv_kernel', 4))
        self.ssm_state_size = int(self.config.get('ssm_state_size', 128))
        self.ssm_n_group = int(self.config.get('ssm_n_group', 16))
        self.ssm_dt_rank = int(self.config.get('ssm_dt_rank', 48))
        self.ssm_inner = int(self.config.get('ssm_inner', 6144))
        self.ssm_head_v_dim = self.ssm_inner // self.ssm_dt_rank
        # detect layer types: True = full attention, False = SSM
        self._is_attn = [(i + 1) % self.full_attn_interval == 0 for i in range(self.n_layers)]
        # assign each layer to a GPU. If layer_split given (counts per GPU),
        # assign contiguously (GPU0 gets the first N0 layers, etc.); else
        # round-robin across devices.
        if self.layer_split and len(self.layer_split) == len(self.devices):
            self.layer_gpu = {}
            gpu = 0
            for i in range(self.n_layers):
                while gpu < len(self.layer_split) and i >= sum(self.layer_split[:gpu + 1]):
                    gpu += 1
                self.layer_gpu[i] = gpu
        else:
            self.layer_gpu = {i: i % len(self.devices) for i in range(self.n_layers)}

        # load all tensors (factored .ntmre compression stays low-rank)
        self.fact = {}
        self.np_w = {}
        self._emb_raw = None  # raw NSVD offset for token_embd (Q8, lazy dequant)
        self._out_raw = None  # raw NSVD offset for output weight (Q8, lazy dequant)
        for name in self._s.entries:
            if name in ('token_embd.weight', 'output.weight'):
                # don't dequant yet — store offset and shape for lazy access
                ttype, shape, data_len = self._s.entries[name]
                off = self._s.offsets[name]
                if name == 'token_embd.weight':
                    self._emb_raw = (off, data_len, shape)
                else:
                    self._out_raw = (off, data_len, shape)
                continue
            t = self._s.load_tensor(name)
            if isinstance(t, tuple):
                # Check if .ntmre compression fallback (rank=0 stored as full Q8 matrix)
                if t[1].shape[0] == t[1].shape[1] and t[0].shape[1] != min(t[0].shape):
                    # rank=0 fallback: P is the full weight, Q is identity
                    # Treat as regular dense weight for efficiency
                    full_w = t[0] @ t[1] if t[1].shape[0] < 1000 else t[0]
                    self.np_w[name] = full_w.astype(np.float32)
                else:
                    # .ntmre compression-factored: keep as (P,Q) for efficient (x@P)@Q matmul
                    self.fact[name] = ('svd', t[0], t[1])
            elif t.ndim == 2:
                self.np_w[name] = t.astype(np.float32)
            else:
                self.np_w[name] = t.astype(np.float32)
        # lazily materialize output head on first use
        self._lm = None
        self.tokenizer = Tokenizer(self._s.tokenizer_bytes)

    # ─── GPU factored matmul ───
    def _upload(self, name, cu):
        """Upload weight for x@W: (seq,in_dim) @ (in_dim,out_dim) = (seq,out_dim)."""
        kind = self.fact[name][0]
        if kind == 'svd':
            _, P, Q = self.fact[name]  # P(in_dim,rank), Q(rank,out_dim)
            m, r = P.shape  # in_dim, rank
            n = Q.shape[1]  # out_dim
            p_gpu, _ = cu.to_gpu(np.ascontiguousarray(P, dtype=np.float32))
            q_gpu, _ = cu.to_gpu(np.ascontiguousarray(Q, dtype=np.float32))
            return ('svd', p_gpu, q_gpu, r, m, n)
        _, W = self.fact[name]
        m, n = W.shape  # in_dim, out_dim
        w_gpu, _ = cu.to_gpu(np.ascontiguousarray(W, dtype=np.float32))
        return ('full', w_gpu, m, n)

    def _mm_np(self, x_np, name):
        """x @ W where W is stored as (in_dim, out_dim). .ntmre compression: (x@P)@Q."""
        t = self.fact.get(name)
        if t is None:
            w = self.np_w.get(name)
            return x_np @ w if w is not None else None
        if t[0] == 'svd':
            _, P, Q = t  # P(in_dim, rank), Q(rank, out_dim)
            return (x_np @ P) @ Q
        _, W = t
        return x_np @ W

    def _mm_gpu(self, x_np, name):
        if not self.use_gpu:
            return self._mm_np(x_np, name)
        # Fall back to numpy for non-.ntmre compression weights
        if name not in self.fact:
            return self._mm_np(x_np, name)
        layer = int(name.split('.')[1]) if name.startswith('blk.') else -1
        gpu_idx = self.layer_gpu[layer] if layer >= 0 else 0
        cu = self.cudas[gpu_idx]
        if layer not in self._gpu_cache:
            for prev in self._gpu_cache:
                for ref in self._gpu_cache[prev].values():
                    self._free_ref(ref, self.cudas[self._gpu_cuda[prev]])
            self._gpu_cache = {}
            self._gpu_cuda = {}
            cache = {}
            for lname in self.fact:
                if lname.startswith('blk.%d.' % layer):
                    cache[lname] = self._upload(lname, cu)
            self._gpu_cache[layer] = cache
            self._gpu_cuda[layer] = gpu_idx
        ref = self._gpu_cache[layer][name]
        seq = x_np.shape[0]
        x_ptr, _ = cu.to_gpu(x_np)
        if ref[0] == 'svd':
            # (x @ P) @ Q:  P(in_dim,rank), Q(rank,out_dim)
            _, p_gpu, q_gpu, r, m, n = ref
            # step1: t = x @ P  (seq,m) @ (m,r) = (seq,r)
            t_ptr = cu.alloc(seq * r * 4)
            cu.launch(cu.f_matmul, ((r + 15) // 16, (seq + 15) // 16), (16, 16),
                      [x_ptr, p_gpu, t_ptr, seq, m, r])
            temp = cu.from_gpu(t_ptr, (seq, r))
            cu.free(t_ptr)
            # step2: out = t @ Q  (seq,r) @ (r,n) = (seq,n)
            t2_ptr, _ = cu.to_gpu(temp)
            c_ptr = cu.alloc(seq * n * 4)
            cu.launch(cu.f_matmul, ((n + 15) // 16, (seq + 15) // 16), (16, 16),
                      [t2_ptr, q_gpu, c_ptr, seq, r, n])
            out = cu.from_gpu(c_ptr, (seq, n))
            cu.free(t2_ptr); cu.free(c_ptr)
        else:
            # x @ W  (seq,m) @ (m,n) = (seq,n)
            _, w_gpu, m, n = ref
            c_ptr = cu.alloc(seq * n * 4)
            cu.launch(cu.f_matmul, ((n + 15) // 16, (seq + 15) // 16), (16, 16),
                      [x_ptr, w_gpu, c_ptr, seq, m, n])
            out = cu.from_gpu(c_ptr, (seq, n))
            cu.free(c_ptr)
        cu.free(x_ptr)
        return out

    def _free_ref(self, ref, cu):
        if ref[0] == 'svd':
            cu.free(ref[1]); cu.free(ref[2])
        else:
            cu.free(ref[1])

    # ─── partial RoPE ───
    def _rope_partial(self, x, pos):
        n_heads, seq, hd = x.shape
        rd = self.rope_dim
        inv_freq = 1.0 / (self.rope_theta ** (np.arange(0, rd, 2) / rd))
        freqs = np.outer(np.arange(pos, pos + seq), inv_freq)
        cos = np.cos(freqs).astype(np.float32)
        sin = np.sin(freqs).astype(np.float32)
        out = x.copy()
        xr = x[..., :rd]
        x1 = xr[..., :rd // 2]
        x2 = xr[..., rd // 2:]
        out[..., :rd] = np.concatenate([x1 * cos[None] - x2 * sin[None],
                                        x1 * sin[None] + x2 * cos[None]], axis=-1)
        return out

    # ─── gated attention layer ───
    def _full_attention(self, x, i, position):
        seq = x.shape[0]
        hd = self.head_dim
        n_heads = self.n_heads
        n_kv = self.n_kv
        dim = self.dim
        h = _rmsnorm(x, self.np_w['blk.%d.attn_norm.weight' % i], self.norm_eps)
        q_full = self._mm_gpu(h, 'blk.%d.attn_q.weight' % i).reshape(seq, hd * 2, n_heads)
        query = q_full[:, :hd, :].transpose(2, 0, 1).copy()
        gate = q_full[:, hd:, :].transpose(2, 0, 1).copy()
        key = self._mm_gpu(h, 'blk.%d.attn_k.weight' % i).reshape(seq, hd, n_kv).transpose(2, 0, 1).copy()
        value = self._mm_gpu(h, 'blk.%d.attn_v.weight' % i).reshape(seq, hd, n_kv).transpose(2, 0, 1).copy()
        qn = self.np_w.get('blk.%d.attn_q_norm.weight' % i)
        kn = self.np_w.get('blk.%d.attn_k_norm.weight' % i)
        if qn is not None:
            query = _rmsnorm(query, qn, self.norm_eps)
        if kn is not None:
            key = _rmsnorm(key, kn, self.norm_eps)
        query = self._rope_partial(query, position)
        key = self._rope_partial(key, position)
        if i not in self.kv_cache:
            self.kv_cache[i] = (key, value)
        else:
            kk, vv = self.kv_cache[i]
            self.kv_cache[i] = (np.concatenate([kk, key], axis=1), np.concatenate([vv, value], axis=1))
        k_all, v_all = self.kv_cache[i]
        rep = n_heads // n_kv
        if rep > 1:
            k_all = np.repeat(k_all, rep, axis=0)
            v_all = np.repeat(v_all, rep, axis=0)
        scores = (query @ k_all.transpose(0, 2, 1)) / np.sqrt(hd)
        total = k_all.shape[1]
        mask = np.triu(np.ones((seq, total), dtype=np.float32), k=total - seq + 1) * -1e9
        scores = scores + mask[None, :, :]
        attn = _softmax(scores, axis=-1)
        out = (attn @ v_all).transpose(1, 0, 2).reshape(seq, n_heads * hd)
        gate_flat = gate.transpose(1, 0, 2).reshape(seq, n_heads * hd)
        out = out * (1.0 / (1.0 + np.exp(-gate_flat)))
        return self._mm_gpu(out, 'blk.%d.attn_output.weight' % i)

    # ─── Gated DeltaNet (SSM) layer ───
    def _ssm_layer(self, x, i, position):
        seq = x.shape[0]
        hk = self.ssm_state_size
        nk = self.ssm_n_group
        nv = self.ssm_dt_rank
        hv = self.ssm_head_v_dim
        ck = self.ssm_conv_kernel
        qkv_dim = hk * nk * 2 + hv * nv
        h = _rmsnorm(x, self.np_w['blk.%d.attn_norm.weight' % i], self.norm_eps)
        qkv = self._mm_gpu(h, 'blk.%d.attn_qkv.weight' % i)
        z = self._mm_gpu(h, 'blk.%d.attn_gate.weight' % i)
        beta = self._mm_gpu(h, 'blk.%d.ssm_beta.weight' % i)
        alpha = self._mm_gpu(h, 'blk.%d.ssm_alpha.weight' % i)
        dt_bias = self.np_w['blk.%d.ssm_dt.bias' % i]
        a = self.np_w['blk.%d.ssm_a' % i]
        gate = _softplus(alpha + dt_bias) * a
        conv_w = self.np_w['blk.%d.ssm_conv1d.weight' % i]
        if i not in self.ssm_state:
            conv_state = np.zeros((ck - 1, qkv_dim), dtype=np.float32)
        else:
            conv_state = self.ssm_state[i][0]
        conv_in = np.concatenate([conv_state, qkv], axis=0)
        self.ssm_state[i] = (conv_in[-(ck - 1):].copy(), None)
        conv_out = np.zeros_like(qkv)
        for t in range(seq):
            for k in range(ck):
                conv_out[t] += conv_w[k] * conv_in[t + (ck - 1) - k]
        conv_out = _silu(conv_out)
        q = conv_out[:, :hk * nk].reshape(seq, hk, nk)
        k = conv_out[:, hk * nk:2 * hk * nk].reshape(seq, hk, nk)
        v = conv_out[:, 2 * hk * nk:].reshape(seq, hv, nv)
        if nk != nv:
            rep = nv // nk
            q = np.repeat(q, rep, axis=2)
            k = np.repeat(k, rep, axis=2)
        q = q / (np.sqrt(np.sum(q * q, axis=-1, keepdims=True)) + 1e-6)
        k = k / (np.sqrt(np.sum(k * k, axis=-1, keepdims=True)) + 1e-6)
        q = q * (1.0 / np.sqrt(hv))
        beta = 1.0 / (1.0 + np.exp(-beta))
        gate = np.exp(gate)
        if i not in self.ssm_state or self.ssm_state[i][1] is None:
            state = np.zeros((nv, hv, hk), dtype=np.float32)
        else:
            state = self.ssm_state[i][1]
        out = np.zeros((seq, hv * nv), dtype=np.float32)
        for t in range(seq):
            # per-V-head GatedDeltaNet step
            for h_idx in range(nv):
                # state[h_idx]: (hv, hk),  k: (hk, nv), v: (hv, nv), q: (hk, nv)
                st_h = state[h_idx]  # (hv, hk)
                k_h = k[t, :, h_idx]   # (hk,)
                v_h = v[t, :, h_idx]   # (hv,)
                q_h = q[t, :, h_idx]   # (hk,)
                # forget & input gates
                g = gate[t, h_idx]      # scalar
                b = beta[t, h_idx]      # scalar
                # delta rule
                kv_mem = st_h @ k_h     # (hv,)
                delta = (v_h - kv_mem) * b
                st_h = g * st_h + np.outer(delta, k_h)  # (hv, hk) + (hv,) ⊗ (hk,)
                state[h_idx] = st_h
                # output: y = state @ q
                y_h = st_h @ q_h  # (hv,)
                out[t, h_idx * hv:(h_idx + 1) * hv] = y_h
        z = z.reshape(seq, hv, nv)
        core = out.reshape(seq, hv, nv)
        rms = np.sqrt(np.mean(core * core, axis=-2, keepdims=True) + self.norm_eps)
        sn = self.np_w['blk.%d.ssm_norm.weight' % i].astype(np.float32)
        core_norm = (core / rms) * sn[None, :, None]
        core_gated = core_norm * _silu(z)
        core_flat = core_gated.reshape(seq, hv * nv)
        self.ssm_state[i] = (self.ssm_state[i][0], state)
        return self._mm_gpu(core_flat, 'blk.%d.ssm_out.weight' % i)

    # ─── Q8 block helpers (never materialize full matrix) ───
    _BLK = 32  # Q8 block size

    def _q8_row_bytes(self, n_cols):
        return ((n_cols + self._BLK - 1) // self._BLK) * (2 + self._BLK)

    def _get_emb(self):
        """Dequant full embedding matrix once, cache. Returns (vocab_sz, dim)."""
        if hasattr(self, '_emb_cached'):
            return self._emb_cached
        off, dlen, (d0, d1) = self._emb_raw
        data = self._s._mmap
        print(f'Dequantizing embedding ({d0} x {d1}) — one-time...', flush=True)
        import time; t0 = time.time()
        flat = _dequant_q8(data, d0 * d1, offset=off)
        emb = flat.reshape(d0, d1)
        # NSVD stores as (dim, vocab_sz). Transpose to (vocab_sz, dim) for lookup.
        if emb.shape[1] != self.dim and emb.shape[0] == self.dim:
            emb = emb.T  # (dim, vocab) -> (vocab, dim)
        self._emb_cached = emb
        print(f'Embedding ready: {emb.shape} in {time.time()-t0:.1f}s', flush=True)
        return emb

    def _load_lm(self):
        """Dequant full output projection matrix once, cache as float32."""
        if self._lm is not None:
            return self._lm
        off, dlen, (vocab_sz, dim) = self._out_raw
        data = self._s._mmap
        print(f'Dequantizing output head ({vocab_sz} x {dim}) — one-time...', flush=True)
        import time; t0 = time.time()
        lm = _dequant_q8(data, vocab_sz * dim, offset=off).reshape(vocab_sz, dim)
        self._lm = lm
        print(f'Output head ready: {lm.shape} in {time.time()-t0:.1f}s', flush=True)
        return lm

    def _lm_project(self, x):
        """Output projection using cached or lazily dequantized lm_head."""
        lm = self._load_lm()
        if lm.shape[1] == self.dim:
            return (x @ lm.T).ravel()
        else:
            return (x @ lm).ravel()

    # ─── forward ───
    def forward(self, token_ids, position=0):
        ids = np.asarray(token_ids, dtype=np.int64)
        if ids.ndim == 0:
            ids = ids[None]
        seq = len(ids)
        dim = self.dim
        # token embedding — lazy full dequant, cached
        emb = self._get_emb()
        if emb.shape[1] == dim:
            x = emb[ids].astype(np.float32)
        else:
            x = emb[:, ids].T.astype(np.float32)
        for i in range(self.n_layers):
            if self._is_attn[i]:
                out = self._full_attention(x, i, position)
            else:
                out = self._ssm_layer(x, i, position)
            x = x + out
            # post_attention_norm IS the FFN norm
            h = _rmsnorm(x, self.np_w['blk.%d.post_attention_norm.weight' % i], self.norm_eps)
            gate = _silu(self._mm_gpu(h, 'blk.%d.ffn_gate.weight' % i))
            up = self._mm_gpu(h, 'blk.%d.ffn_up.weight' % i)
            down = self._mm_gpu(gate * up, 'blk.%d.ffn_down.weight' % i)
            x = x + down
        x = _rmsnorm(x, self.np_w['output_norm.weight'], self.norm_eps)
        # output projection — chunked Q8 without full materialization
        logits = self._lm_project(x[-1:])
        return logits

    def reset_kv(self):
        self.kv_cache = {}
        self.ssm_state = {}

    def generate(self, input_ids, max_new_tokens=128, temperature=0.7, top_k=40, top_p=0.9):
        self.reset_kv()
        input_ids = list(input_ids)
        generated = []
        logits = self.forward(input_ids, position=0)
        for _ in range(max_new_tokens):
            probs = _softmax(logits / temperature if temperature else logits)
            if top_k and top_k > 0 and top_k < len(probs):
                idx = np.argsort(probs)[-top_k:]
                m = np.full_like(probs, 0.0); m[idx] = probs[idx]; probs = m
                probs = probs / probs.sum()
            if top_p and top_p < 1.0:
                order = np.argsort(probs)[::-1]
                cum = np.cumsum(probs[order])
                keep = order[cum <= top_p]
                if len(keep) == 0:
                    keep = order[:1]
                m = np.full_like(probs, 0.0); m[keep] = probs[keep]
                probs = m / m.sum()
            nxt = int(np.random.choice(len(probs), p=probs))
            generated.append(nxt)
            if nxt in (2, 0):
                break
            input_ids.append(nxt)
            logits = self.forward([nxt], position=len(input_ids) - 1)
        return generated


# ═══════════════════════════════════════════════════════════════════
# Chat + UI
# ═══════════════════════════════════════════════════════════════════
def _auto_balance(devices):
    """Query VRAM per GPU (nvidia-smi) and split layers proportionally.
    Bigger-VRAM card gets more layers. Falls back to round-robin if smi
    is unavailable."""
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            text=True)
        mems = [int(x.strip()) for x in out.strip().split('\n') if x.strip()]
        if len(mems) >= len(devices):
            mems = [mems[d] for d in devices]
            total = sum(mems)
            split = [int(round(m / total * 64)) for m in mems]
            # fix rounding so it sums to 64
            diff = 64 - sum(split)
            split[split.index(max(split))] += diff
            return split
    except Exception:
        pass
    return None


def format_prompt(history, soul=None):
    parts = []
    if soul:
        parts.append("<|im_start|>system\n%s<|im_end|>\n" % soul)
    for role, text in history:
        parts.append("<|im_start|>%s\n%s<|im_end|>\n" % (role, text))
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser(description="Bonsai-27B PC app (standalone NTMRE GPU)")
    ap.add_argument("model", help="path to NTMRE .ntmre compression model (.ntmre)")
    ap.add_argument("--cli", action="store_true", help="terminal chat instead of Gradio UI")
    ap.add_argument("--device", type=int, default=0, help="first GPU device id")
    ap.add_argument("--devices", default="", help="comma-separated GPU ids, e.g. 0,1 (multi-GPU)")
    ap.add_argument("--layer-split", default="", help="layers per GPU, e.g. 28,36 (auto-balance by VRAM if empty)")
    ap.add_argument("--cpu", action="store_true", help="use numpy matmul instead of CUDA")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--soul", default="soul.md", help="path to soul/persona file to inject as system prompt ('' disables)")
    args = ap.parse_args()

    soul = None
    if args.soul:
        sp = Path(args.soul)
        if sp.exists():
            soul = sp.read_text(encoding='utf-8', errors='replace').strip()
            print("Soul injected from %s (%d chars)" % (sp, len(soul)))
        else:
            print("Soul file not found: %s (running without soul)" % args.soul)

    devices = [int(d) for d in args.devices.split(',') if d.strip()] or [args.device]
    layer_split = [int(x) for x in args.layer_split.split(',') if x.strip()] or None
    if layer_split is None and len(devices) > 1:
        layer_split = _auto_balance(devices)
    print("Loading Bonsai-27B (standalone NTMRE engine) on GPUs %s..." % devices)
    eng = BonsaiEngine(args.model, device=args.device, use_gpu=not args.cpu,
                       devices=devices, layer_split=layer_split)
    tok = eng.tokenizer
    print("Ready. dim=%d layers=%d heads=%d on %d GPU(s): %s" % (
        eng.dim, eng.n_layers, eng.n_heads, len(eng.devices), eng.devices))
    if layer_split:
        print("Layer split (per GPU): %s" % layer_split)

    if args.cli:
        print("Type 'exit' to quit.\n")
        while True:
            try:
                prompt = input("You: ")
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.strip().lower() in ("exit", "quit"):
                break
            history = [("user", prompt)]
            ids = tok.encode(format_prompt(history, soul))
            gen = eng.generate(ids, max_new_tokens=args.max_tokens, temperature=args.temperature)
            print("Bonsai:", tok.decode(gen).encode('ascii', 'replace').decode())
        return

    import gradio as gr

    def respond(message, history):
        history = history or []
        msgs = [("user", m[0]) for m in history] + [("user", message)]
        ids = tok.encode(format_prompt(msgs, soul))
        gen = eng.generate(ids, max_new_tokens=args.max_tokens, temperature=args.temperature)
        return tok.decode(gen)

    gr.ChatInterface(
        respond,
        title="Bonsai-27B (NTMRE GPU)",
        description="Qwen3.5 hybrid on the NTMRE .ntmre compression GPU engine — standalone",
    ).launch()


if __name__ == "__main__":
    main()


# Workaround: some sealed exe runtimes fail to keep ctypes in module globals.
# Re-import it at module load if missing.
if 'ctypes' not in globals():
    import ctypes
