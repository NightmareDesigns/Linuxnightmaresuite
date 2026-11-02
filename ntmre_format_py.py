"""ntmre_format_py — Python reader/writer for the .ntmre container.

Canonical format, from the ONLY authoritative source in the tree:
    engine/ntmre_format.h  (NTMRE FORMAT v2, container version 13)

Layout, little-endian, packed (no padding inside structs):
    [0]   file header, 16 bytes:
            magic u32 = 0x4E544D52 ("NTMR"), version u32 = 13,
            flags u32 = 0, header_size u32 = data_offset
    [16]  config, 324 bytes:
            base, 36 bytes: dim, n_layers, n_heads, n_kv_heads,
                hidden_dim, vocab_size, max_seq_len u32s, rope_base f32,
                head_dim u32
            extensions, 32 bytes: arch_type, q_output_dim,
                sliding_window u32s, softcap f32, shared_kv_layers,
                per_layer_hidden_count, _pad_v2[0] (BPE pre-type),
                _pad_v2[1] u32s
            per_layer_hidden_dims, 256 bytes: 64 x u32
    [340] tokenizer header, 40 bytes: n_tokens, n_merges, bos_id,
            eos_id, pad_id, unk_id, tokens_offset, scores_offset,
            merges_offset, tokens_size u32s
            (tokens_offset = 0, scores_offset = tokens_size,
             merges_offset = tokens_size + n_tokens * 4, as in the
             engine converter's writer)
    [...] token strings: u16 length + bytes each, then scores f32 each,
          then merges (u32, u32) each
    [...] u32 n_tensors, then 91-byte tensor infos:
            name_len u8, name 64 bytes (NOT null-terminated, use
            name_len), dtype u8, n_dims u8, dim0 u32, dim1 u32,
            offset u64 (from data-section start), size_bytes u64
    [...] padding to 64-byte alignment -> data section (data_offset)
    [data_offset] tensor data, each tensor start aligned to 64 bytes
          within the data section (as in engine/nightmare_convert.py)

Dtype codes (ntmre_format.h, NOT the engine converter's internal
type codes, and NOT the old writer's codes):
    F32 = 0 (4 bytes/elem), F16 = 1 (2 bytes/elem),
    Q8 = 2 (34 bytes / 32 elems: f16 scale + 32 x int8),
    Q4 = 3 (18 bytes / 32 elems: f16 scale + 16 packed bytes),
    .ntmre compression = 4 (same Q8 block layout when flat; see below)

Reader rules, following the C reader:
    - magic must be 0x4E544D52, version must be <= 13 (writers emit
      13 only; versions 1..13 are accepted, version < 2 forces
      arch_type = LLaMA and per_layer_hidden_count = 0).
    - tensor count must be <= 4096 (NTMRE_MAX_TENSORS).
    - data_offset is recomputed as align_up(position after the index),
      exactly as the C reader does; header_size is written as that
      value but is not trusted on read.
    - tensor data is returned as raw stored bytes by read_ntmre();
      dequantisation is a separate, explicit step (dequant_tensor).

.ntmre compression dialects — read this before touching .ntmre compression tensors:
    There are TWO .ntmre compression storages in this tree, and they are NOT
    interchangeable:
      (a) packed factors (root nightmare_convert.svd_compress via
          pack_svd): u32 rank + Q8(U[:, :rank]) + f32 S[:rank] +
          Q8(Vt[:rank, :]).  This is what fusion_engine expects
          (unpack_svd) and what nightmare_engine.dequant_svd_q8
          detects by length.  The C flat loader SKIPS these, because
          their size is not (n_elements / 32) * 34 — the header
          itself warns about legacy factor files.
      (b) flat denoised Q8 (engine/nightmare_convert.svd_compress):
          the factors are folded back into a full matrix and that
          matrix is Q8-quantised flat, size (n_elements / 32) * 34.
          This is what the C loader and the v13 size guard accept.
    write_ntmre() accepts either storage as-is for .ntmre compression and does
    NOT rewrite one into the other.  dequant_tensor() for .ntmre compression
    detects the storage by length: packed layout first, else flat
    Q8 if the size matches exactly, else a clear ValueError.
    unpack_svd() itself handles packed storage only (plus the
    A/B variant, S folded into the factors, detected by length).

Q4 nibble order — header is authoritative, other code disagrees:
    ntmre_format.h dequant_q4 and fusion_engine's own Q4 helper both
    emit, per packed byte, low nibble first then high nibble, in
    element order (element 2*i = low - 8, 2*i + 1 = high - 8).
    engine/nightmare_convert._dequant_q4_0 and
    nightmare_engine.dequant_q4_0 instead emit all low nibbles
    first, then all high nibbles.  This module follows the header
    (and fusion).  Q4 tensors produced by the engine-converter
    order are therefore UNVERIFIED against this reader — see the
    UNVERIFIED section at the end of this docstring and in the
    test report.  Do not silently re-order Q4 data.

Half-float conversion follows ntmre_format.h exactly, including its
non-standard cases: exponent 0 -> subnormal value, exponent 31 ->
0.0 (not inf/nan).  That matters for Q8/Q4 scales and for F16.

UNVERIFIED — no real .ntmre weights exist in this tree to test
against, so the following are NOT proven by this module:
    - Byte-compatibility with any real .ntmre file produced by the
      original writer, a GPU converter exe, or the C engine.
      Synthetic round-trips (write -> read -> dequant) pass, and
      the layout matches ntmre_format.h and
      engine/nightmare_convert.py field-for-field, but a real
      file has never been read by this code.
    - Q4 cross-compatibility (see Q4 section above).
    - .ntmre compression packed files under the C flat loader (it skips them
      by design; see dialect (a)).
    - Tokenizer merges supplied as "left right" strings: they are
      resolved through the token list and unresolvable merges are
      skipped, matching engine/nightmare_convert.py's extractor;
      a real GGUF merge list has not been round-tripped here.
    - Any behaviour on Windows, on GPU, or under a live model load.

Original code, stdlib only at import time.  numpy is imported
lazily inside the functions that return arrays, because the
consumers (nightmare_convert, fusion loaders) already require it;
if numpy is absent those functions raise a clear ImportError
instead of returning a wrong type.
"""

import struct

__all__ = [
    "MAGIC", "NTMRE_MAGIC", "VERSION", "NTMRE_VERSION",
    "VERSION_MIN_READ", "ALIGN", "NTMRE_ALIGN",
    "MAX_TENSORS", "NTMRE_MAX_TENSORS",
    "MAX_NAME_LEN", "NTMRE_MAX_NAME_LEN",
    "MAX_LAYERS", "NTMRE_MAX_LAYERS",
    "ARCH_LLAMA", "ARCH_GEMMA4", "NTMRE_ARCH_LLAMA", "NTMRE_ARCH_GEMMA4",
    "F32", "F16", "Q8", "Q4", "SVD_Q8",
    "NtmreError",
    "dtype_size", "f16_to_f32", "f32_to_f16",
    "quantize_q8", "dequant_q8", "dequant_q4", "dequant_f16",
    "pack_svd", "unpack_svd", "dequant_tensor",
    "read_ntmre", "write_ntmre",
    "_pack_config", "_pack_tokenizer",
]

# --- constants: ntmre_format.h -------------------------------------------
MAGIC = 0x4E544D52
NTMRE_MAGIC = MAGIC
VERSION = 13
NTMRE_VERSION = VERSION
VERSION_MIN_READ = 1
ALIGN = 64
NTMRE_ALIGN = ALIGN
MAX_TENSORS = 4096
NTMRE_MAX_TENSORS = MAX_TENSORS
MAX_NAME_LEN = 64
NTMRE_MAX_NAME_LEN = MAX_NAME_LEN
MAX_LAYERS = 64
NTMRE_MAX_LAYERS = MAX_LAYERS

ARCH_LLAMA = 0
ARCH_GEMMA4 = 1
NTMRE_ARCH_LLAMA = ARCH_LLAMA
NTMRE_ARCH_GEMMA4 = ARCH_GEMMA4

F32 = 0
F16 = 1
Q8 = 2
Q4 = 3
SVD_Q8 = 4

_BLOCK = 32
_Q8_BLOCK_BYTES = 34
_Q4_BLOCK_BYTES = 18
_CONFIG_SIZE = 324
_TOKENIZER_HEADER_SIZE = 40
_TENSOR_INFO_SIZE = 91


class NtmreError(ValueError):
    """Raised for bad magic, version, layout, or truncated data."""


def _align_up(n, align=ALIGN):
    return ((n + align - 1) // align) * align


def _require_numpy():
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - numpy is present in suite
        raise ImportError(
            "ntmre_format_py array functions require numpy, which the "
            "consumers (nightmare_convert, fusion loaders) already require"
        ) from exc
    return np


# --- half-float conversion, exactly as ntmre_format.h --------------------
def f16_to_f32(bits):
    """One f16 bit pattern -> Python float, C-header semantics.

    exp == 0: subnormal, exp == 31: 0.0 (not inf/nan).
    """
    bits = int(bits) & 0xFFFF
    sign = (bits >> 15) & 1
    exp = (bits >> 10) & 0x1F
    mant = bits & 0x3FF
    if exp == 0:
        value = (mant / 1024.0) * (1.0 / 16384.0)
        return -value if sign else value
    if exp == 31:
        return 0.0
    f32_bits = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13)
    return struct.unpack("<f", struct.pack("<I", f32_bits))[0]


def f32_to_f16(value):
    """Python float -> f16 bit pattern, C quantizer semantics.

    Truncates the mantissa (mant >> 13), clamps exponent > 15 to the
    inf pattern and exponent < -14 to zero, matching
    ntmre_quantize_q8 in ntmre_format.h.  This is NOT round-to-nearest
    (numpy float16); the difference is at most one mantissa bit on
    the Q8/Q4 scale, which is inside Q8 quantisation error.
    """
    value = float(value)
    if value == 0.0:
        return 0
    f32_bits = struct.unpack("<I", struct.pack("<f", value))[0]
    sign_bit = (f32_bits >> 31) & 1
    exp_val = int((f32_bits >> 23) & 0xFF) - 127
    mant_val = f32_bits & 0x7FFFFF
    if exp_val > 15:
        return (sign_bit << 15) | (0x1F << 10)
    if exp_val < -14:
        return sign_bit << 15
    return (sign_bit << 15) | ((exp_val + 15) << 10) | (mant_val >> 13)


def dtype_size(dtype, n_elements):
    """Raw byte size for dtype and element count, per ntmre_format.h.

    Q8/.ntmre compression flat size is (n // 32) * 34; packed .ntmre compression factor data
    deliberately does NOT follow this (see module docstring).
    Unknown dtype -> 0, as in the C function.
    """
    n = int(n_elements)
    if dtype == F32:
        return n * 4
    if dtype == F16:
        return n * 2
    if dtype in (Q8, SVD_Q8):
        return (n // _BLOCK) * _Q8_BLOCK_BYTES
    if dtype == Q4:
        return (n // _BLOCK) * _Q4_BLOCK_BYTES
    return 0


# --- Q8 / Q4 / F16 quantisation and dequantisation -----------------------
def _quantize_q8_blocks(flat):
    """float32 sequence (already padded to a multiple of 32) -> Q8 bytes."""
    out = bytearray()
    for b in range(len(flat) // _BLOCK):
        blk = flat[b * _BLOCK:(b + 1) * _BLOCK]
        amax = 0.0
        for v in blk:
            av = -v if v < 0 else v
            if av > amax:
                amax = av
        scale = amax / 127.0 if amax > 0 else 0.0
        inv = 1.0 / scale if scale > 0 else 0.0
        out += struct.pack("<H", f32_to_f16(scale))
        for v in blk:
            q = v * inv
            qi = int(q + (0.5 if q >= 0 else -0.5))
            if qi > 127:
                qi = 127
            elif qi < -127:
                qi = -127
            out += struct.pack("<b", qi)
    return bytes(out)


def quantize_q8(data):
    """float32 array-like -> Q8_0 bytes, padding the last block with 0.

    Returns bytes only.  The caller is responsible for remembering
    the true element count; dequant_q8() truncates padding by count.
    """
    np = _require_numpy()
    flat = np.asarray(data, dtype=np.float32).ravel().tolist()
    rem = len(flat) % _BLOCK
    if rem:
        flat = flat + [0.0] * (_BLOCK - rem)
    return _quantize_q8_blocks(flat)


def _dequant_q8_blocks(data, n_blocks):
    """Q8 blocks -> list of floats (padded length, n_blocks * 32)."""
    out = []
    for b in range(n_blocks):
        base = b * _Q8_BLOCK_BYTES
        if base + _Q8_BLOCK_BYTES > len(data):
            raise NtmreError("truncated Q8 data: block %d missing" % b)
        scale = f16_to_f32(struct.unpack_from("<H", data, base)[0])
        for i in range(_BLOCK):
            q = struct.unpack_from("<b", data, base + 2 + i)[0]
            out.append(scale * float(q))
    return out


def dequant_q8(data, n):
    """Q8_0 bytes -> float32 numpy array of length n (header layout)."""
    np = _require_numpy()
    data = bytes(data)
    n = int(n)
    if n < 0:
        raise NtmreError("dequant_q8: negative element count")
    n_blocks = (n + _BLOCK - 1) // _BLOCK
    if len(data) < n_blocks * _Q8_BLOCK_BYTES:
        raise NtmreError(
            "dequant_q8: need %d bytes for %d elements, got %d"
            % (n_blocks * _Q8_BLOCK_BYTES, n, len(data)))
    return np.asarray(_dequant_q8_blocks(data, n_blocks),
                      dtype=np.float32)[:n]


def dequant_q4(data, n):
    """Q4_0 bytes -> float32 numpy array of length n (header layout).

    Per 32-element block: f16 scale, then 16 packed bytes.  Per packed
    byte, element 2*i = (low nibble - 8) * scale, element 2*i + 1 =
    (high nibble - 8) * scale.  This is the ntmre_format.h order;
    see the module docstring for the conflicting engine order
    (UNVERIFIED for cross-reader Q4).
    """
    np = _require_numpy()
    data = bytes(data)
    n = int(n)
    n_blocks = (n + _BLOCK - 1) // _BLOCK
    if len(data) < n_blocks * _Q4_BLOCK_BYTES:
        raise NtmreError(
            "dequant_q4: need %d bytes for %d elements, got %d"
            % (n_blocks * _Q4_BLOCK_BYTES, n, len(data)))
    out = []
    for b in range(n_blocks):
        base = b * _Q4_BLOCK_BYTES
        scale = f16_to_f32(struct.unpack_from("<H", data, base)[0])
        packed = data[base + 2:base + _Q4_BLOCK_BYTES]
        for byte in packed:
            out.append(scale * float((byte & 0x0F) - 8))
            out.append(scale * float((byte >> 4) - 8))
    return np.asarray(out, dtype=np.float32)[:n]


def dequant_f16(data, n):
    """F16 bytes -> float32 numpy array of length n, header semantics."""
    np = _require_numpy()
    data = bytes(data)
    n = int(n)
    if len(data) < n * 2:
        raise NtmreError(
            "dequant_f16: need %d bytes for %d elements, got %d"
            % (n * 2, n, len(data)))
    bits = struct.unpack_from("<%dH" % n, data, 0)
    return np.asarray([f16_to_f32(b) for b in bits], dtype=np.float32)


# --- .ntmre compression factor packing ----------------------------------------------------
def pack_svd(U, S, Vt, rank):
    """Pack .ntmre compression factors as root nightmare_convert expects.

    Layout: u32 rank + Q8(U[:, :rank]) + f32 S[:rank] + Q8(Vt[:rank, :]).
    U, S, Vt are array-likes from an .ntmre compression (full_matrices=False).
    Each factor is Q8-padded independently, so unpack_svd() can
    recover each factor by its own padded length.
    """
    np = _require_numpy()
    U = np.asarray(U, dtype=np.float32)
    S = np.asarray(S, dtype=np.float32).ravel()
    Vt = np.asarray(Vt, dtype=np.float32)
    rank = int(rank)
    if rank < 1:
        raise NtmreError("pack_svd: rank must be >= 1, got %d" % rank)
    if rank > len(S):
        raise NtmreError(
            "pack_svd: rank %d exceeds singular values %d" % (rank, len(S)))
    if U.ndim != 2 or Vt.ndim != 2:
        raise NtmreError("pack_svd: U and Vt must be 2-D")
    if U.shape[1] < rank or Vt.shape[0] < rank:
        raise NtmreError(
            "pack_svd: rank %d exceeds U %s / Vt %s"
            % (rank, U.shape, Vt.shape))
    out = bytearray()
    out += struct.pack("<I", rank)
    out += quantize_q8(U[:, :rank])
    out += struct.pack("<%df" % rank, *[float(v) for v in S[:rank]])
    out += quantize_q8(Vt[:rank, :])
    return bytes(out)


def unpack_svd(data, m, n):
    """Unpack packed .ntmre compression factors -> (U, S, Vt) float32 arrays.

    Packed layout only (see module docstring).  Two variants are
    detected by length, mirroring nightmare_engine.dequant_svd_q8:
      - U/S/Vt: rank + Q8(U) + f32 S + Q8(Vt)
      - A/B (S folded): rank + Q8(A) + Q8(B), returned as
        (A, ones(rank), B) so (U * S) @ Vt still reconstructs.
      - rank == 0: the remaining bytes are a flat Q8 matrix;
        returned as (W, ones(n), eye(n)), as in nightmare_engine.
    Anything else raises ValueError instead of silently returning
    a wrong factorisation.  Flat Q8 .ntmre compression data from the engine
    converter is NOT unpacked here — dequant_tensor() handles
    that dialect by length, unpack_svd() alone does not.
    """
    np = _require_numpy()
    data = bytes(data)
    m = int(m)
    n = int(n)
    if m < 1 or n < 1:
        raise NtmreError("unpack_svd: bad shape (%d, %d)" % (m, n))
    if len(data) < 4:
        raise NtmreError("unpack_svd: data too short for a rank field")
    rank = struct.unpack_from("<I", data, 0)[0]
    if rank == 0:
        # Flat Q8 matrix dialect, rank field is a sentinel in the
        # nightmare_engine reader.  Reconstruct as (W, ones, eye).
        w = dequant_q8(data[4:], m * n).reshape(m, n)
        return (w.astype(np.float32),
                np.ones(n, dtype=np.float32),
                np.eye(n, dtype=np.float32))
    if rank > min(m, n):
        raise NtmreError(
            "unpack_svd: rank %d exceeds shape (%d, %d)" % (rank, m, n))
    u_len = ((m * rank + _BLOCK - 1) // _BLOCK) * _Q8_BLOCK_BYTES
    v_len = ((rank * n + _BLOCK - 1) // _BLOCK) * _Q8_BLOCK_BYTES
    pos = 4
    if len(data) < pos + u_len:
        raise NtmreError("unpack_svd: truncated U factor")
    U = dequant_q8(data[pos:pos + u_len], m * rank).reshape(m, rank)
    pos += u_len
    if len(data) >= pos + 4 * rank + v_len and\
            len(data) == pos + 4 * rank + v_len:
        # U/S/Vt dialect: exact length match required, otherwise the
        # trailing bytes would be silently ignored.
        S = np.frombuffer(data[pos:pos + 4 * rank],
                          dtype=np.float32).astype(np.float32).copy()
        pos += 4 * rank
        Vt = dequant_q8(data[pos:pos + v_len], rank * n).reshape(rank, n)
        return U.astype(np.float32), S, Vt.astype(np.float32)
    if len(data) == pos + v_len:
        # A/B dialect: S already folded into the factors.
        B = dequant_q8(data[pos:pos + v_len], rank * n).reshape(rank, n)
        return (U.astype(np.float32),
                np.ones(rank, dtype=np.float32),
                B.astype(np.float32))
    raise NtmreError(
        "unpack_svd: data length %d does not match packed U/S/Vt (%d) "
        "or A/B (%d) layout for shape (%d, %d), rank %d"
        % (len(data), 4 + u_len + 4 * rank + v_len, 4 + u_len + v_len,
           m, n, rank))


def dequant_tensor(dtype, data, shape):
    """Dequantise one stored tensor -> float32 numpy array, given shape.

    dtype is a v13 code (F32/F16/Q8/Q4/.ntmre compression).  data is raw stored
    bytes, shape is a tuple.  .ntmre compression is detected by length: packed
    factors first, flat Q8 if the size matches exactly, else error.
    """
    np = _require_numpy()
    shape = tuple(int(d) for d in shape)
    if not shape:
        raise NtmreError("dequant_tensor: empty shape")
    n = 1
    for d in shape:
        if d < 0:
            raise NtmreError("dequant_tensor: negative dimension")
        n *= d
    data = bytes(data)
    if dtype == F32:
        if len(data) < n * 4:
            raise NtmreError("dequant_tensor F32: truncated data")
        return np.frombuffer(data[:n * 4], dtype=np.float32).reshape(shape).copy()
    if dtype == F16:
        return dequant_f16(data, n).reshape(shape)
    if dtype == Q8:
        return dequant_q8(data, n).reshape(shape)
    if dtype == Q4:
        return dequant_q4(data, n).reshape(shape)
    if dtype == SVD_Q8:
        if len(shape) != 2:
            raise NtmreError(
                "dequant_tensor .ntmre compression: need a 2-D shape, got %s" % (shape,))
        m, n2 = shape
        # Packed first (raises only if it is neither packed nor flat).
        try:
            U, S, Vt = unpack_svd(data, m, n2)
            return ((U * S[np.newaxis, :]) @ Vt).astype(np.float32)
        except NtmreError:
            pass
        if len(data) == (n // _BLOCK) * _Q8_BLOCK_BYTES and n % _BLOCK == 0:
            # Flat denoised Q8 dialect (engine converter).
            return dequant_q8(data, n).reshape(shape)
        raise NtmreError(
            "dequant_tensor .ntmre compression: data length %d matches neither packed "
            "nor flat Q8 layout for shape %s" % (len(data), shape))
    raise NtmreError("dequant_tensor: unknown dtype %d" % dtype)


# --- config / tokenizer packing -------------------------------------------
def _pack_config(config):
    """Config dict -> 324 bytes, exactly as ntmre_format.h lays out."""
    cfg = config or {}
    dim = int(cfg.get("dim", 0))
    n_layers = int(cfg.get("n_layers", 0))
    n_heads = int(cfg.get("n_heads", 0))
    n_kv_heads = int(cfg.get("n_kv_heads", 0))
    hidden_dim = int(cfg.get("hidden_dim", 0))
    vocab_size = int(cfg.get("vocab_size", 0))
    max_seq_len = int(cfg.get("max_seq_len", 1024))
    rope_base = float(cfg.get("rope_base", 10000.0))
    head_dim = int(cfg.get("head_dim", 0)) or\
        (dim // n_heads if dim and n_heads else 0)
    base = struct.pack("<IIIIIIIfI", dim, n_layers, n_heads, n_kv_heads,
                       hidden_dim, vocab_size, max_seq_len, rope_base,
                       head_dim)
    arch_type = int(cfg.get("arch_type", 0))
    q_output_dim = int(cfg.get("q_output_dim", 0)) or\
        (n_heads * head_dim if head_dim else dim)
    sliding_window = int(cfg.get("sliding_window", 0))
    softcap = float(cfg.get("softcap", 0.0))
    shared_kv_layers = int(cfg.get("shared_kv_layers", 0))
    per_dims = cfg.get("per_layer_hidden_dims") or []
    per_dims = [int(v) for v in per_dims][:MAX_LAYERS]
    per_count = int(cfg.get("per_layer_hidden_count", len(per_dims)))
    per_count = max(0, min(per_count, MAX_LAYERS))
    pad = cfg.get("_pad_v2") or []
    pre_type = int(cfg.get("pre_type", pad[0] if pad else 0))
    pad1 = int(pad[1]) if len(pad) > 1 else 0
    ext = struct.pack("<IIIfIIII", arch_type, q_output_dim, sliding_window,
                      softcap, shared_kv_layers, per_count, pre_type, pad1)
    per = list(per_dims) + [0] * (MAX_LAYERS - len(per_dims))
    ext += struct.pack("<%dI" % MAX_LAYERS, *per)
    out = base + ext
    assert len(out) == _CONFIG_SIZE, len(out)
    return out


def _config_from_bytes(raw):
    if len(raw) != _CONFIG_SIZE:
        raise NtmreError("config block is %d bytes, expected 324" % len(raw))
    base = struct.unpack_from("<IIIIIIIfI", raw, 0)
    ext = struct.unpack_from("<IIIfIIII", raw, 36)
    per = list(struct.unpack_from("<%dI" % MAX_LAYERS, raw, 68))
    per_count = ext[5]
    cfg = {
        "dim": base[0], "n_layers": base[1], "n_heads": base[2],
        "n_kv_heads": base[3], "hidden_dim": base[4],
        "vocab_size": base[5], "max_seq_len": base[6],
        "rope_base": base[7], "head_dim": base[8],
        "arch_type": ext[0], "q_output_dim": ext[1],
        "sliding_window": ext[2], "softcap": ext[3],
        "shared_kv_layers": ext[4],
        "per_layer_hidden_count": per_count,
        "per_layer_hidden_dims": per[:per_count] if per_count else [],
        "_pad_v2": [ext[6], ext[7]],
        "pre_type": ext[6],
        "arch": "gemma4" if ext[0] == ARCH_GEMMA4 else "llama",
    }
    return cfg


def _pack_tokenizer(tokenizer):
    """Tokenizer dict -> tokenizer section bytes (header + strings + ...).

    Accepts tokens/scores/merges, bos/eos/pad/unk or bos_id/eos_id/
    pad_id/unk_id.  Merges may be (a, b) int pairs, or "left right"
    strings resolved through the token list (unresolvable merges are
    skipped, as in engine/nightmare_convert.py's extractor).
    """
    tok = tokenizer or {}
    tokens = [str(t) for t in (tok.get("tokens") or [])]
    scores = tok.get("scores") or [0.0] * len(tokens)
    scores = [float(s) for s in scores[:len(tokens)]]
    scores += [0.0] * (len(tokens) - len(scores))
    merges_in = tok.get("merges") or []
    tok_to_id = {}
    for i, t in enumerate(tokens):
        if t and t not in tok_to_id:
            tok_to_id[t] = i
    merges = []
    for m in merges_in:
        if isinstance(m, (list, tuple)) and len(m) == 2:
            merges.append((int(m[0]), int(m[1])))
        elif isinstance(m, str):
            parts = m.split(" ", 1)
            if len(parts) == 2 and parts[0] in tok_to_id\
                    and parts[1] in tok_to_id:
                merges.append((tok_to_id[parts[0]], tok_to_id[parts[1]]))
            # else: skipped, documented above
        # any other shape is skipped, not guessed
    token_bytes = bytearray()
    for t in tokens:
        tb = t.encode("utf-8")
        if len(tb) > 0xFFFF:
            raise NtmreError("token too long for u16 length field")
        token_bytes += struct.pack("<H", len(tb)) + tb
    tokens_size = len(token_bytes)
    scores_offset = tokens_size
    merges_offset = tokens_size + len(tokens) * 4
    hdr = struct.pack(
        "<IIIIIIIIII",
        len(tokens), len(merges),
        int(tok.get("bos_id", tok.get("bos", 0))),
        int(tok.get("eos_id", tok.get("eos", 0))),
        int(tok.get("pad_id", tok.get("pad", 0))),
        int(tok.get("unk_id", tok.get("unk", 0))),
        0, scores_offset, merges_offset, tokens_size)
    out = bytearray(hdr)
    out += token_bytes
    if tokens:
        out += struct.pack("<%df" % len(tokens), *scores)
    for a, b in merges:
        out += struct.pack("<II", a, b)
    return bytes(out)


def _tokenizer_from_bytes(f, th):
    """Read tokenizer section from an open binary file, given its header.

    Follows the C reader: sequential strings, scores, merges.  The
    header offsets are not trusted for positioning.
    """
    n_tokens, n_merges = th[0], th[1]
    tokens = []
    for _ in range(n_tokens):
        raw = f.read(2)
        if len(raw) != 2:
            raise NtmreError("truncated tokenizer: token length")
        slen = struct.unpack("<H", raw)[0]
        tb = f.read(slen)
        if len(tb) != slen:
            raise NtmreError("truncated tokenizer: token bytes")
        tokens.append(tb.decode("utf-8", "replace"))
    scores = []
    if n_tokens:
        raw = f.read(4 * n_tokens)
        if len(raw) != 4 * n_tokens:
            raise NtmreError("truncated tokenizer: scores")
        scores = list(struct.unpack("<%df" % n_tokens, raw))
    merges = []
    for _ in range(n_merges):
        raw = f.read(8)
        if len(raw) != 8:
            raise NtmreError("truncated tokenizer: merges")
        merges.append(struct.unpack("<II", raw))
    return {
        "tokens": tokens, "scores": scores, "merges": merges,
        "bos": th[2], "eos": th[3], "pad": th[4], "unk": th[5],
        "bos_id": th[2], "eos_id": th[3], "pad_id": th[4], "unk_id": th[5],
        "n_tokens": n_tokens, "n_merges": n_merges,
    }


# --- file writer / reader ---------------------------------------------------
def write_ntmre(output_path, config, tokenizer, tensors_packed):
    """Write a .ntmre file in NTMR v13 layout.

    tensors_packed: list of (name, dtype, shape, data_bytes) where
    dtype is a v13 code (F32/F16/Q8/Q4/.ntmre compression) — the codes root
    nightmare_convert uses (F32=0, Q8=2, .ntmre compression=4), NOT the engine
    converter's internal 0/1/2 codes.  Shape may be 1-D or 2-D;
    higher-rank shapes are flattened to (shape[0], rest), as in
    engine/nightmare_convert.py, and that flattening is visible
    on read-back.  F32/F16/Q8 data sizes are validated exactly;
    Q4 data must be ceil(n/32)*18 bytes; .ntmre compression data is accepted
    as packed factors or as flat Q8, and anything else raises
    instead of writing a file a reader will skip or misread.
    Returns the output path as a string.
    """
    entries = []
    for item in tensors_packed:
        if len(item) != 4:
            raise NtmreError(
                "tensors_packed entries must be (name, dtype, shape, data)")
        name, dtype, shape, data = item
        dtype = int(dtype)
        if dtype not in (F32, F16, Q8, Q4, SVD_Q8):
            raise NtmreError("unknown dtype %d for tensor %r" % (dtype, name))
        shape = tuple(int(d) for d in shape)
        if not shape or any(d < 0 for d in shape):
            raise NtmreError("bad shape %r for tensor %r" % (shape, name))
        n_elements = 1
        for d in shape:
            n_elements *= d
        if len(shape) == 1:
            n_dims, dim0, dim1 = 1, shape[0], 0
        elif len(shape) == 2:
            n_dims, dim0, dim1 = 2, shape[0], shape[1]
        else:
            n_dims, dim0, dim1 = 2, shape[0], n_elements // shape[0]\
                if shape[0] else 0
        data = bytes(data)
        if dtype == F32 and len(data) != n_elements * 4:
            raise NtmreError(
                "%s: F32 data is %d bytes but %d elements need %d"
                % (name, len(data), n_elements, n_elements * 4))
        if dtype == F16 and len(data) != n_elements * 2:
            raise NtmreError(
                "%s: F16 data is %d bytes but %d elements need %d"
                % (name, len(data), n_elements, n_elements * 2))
        if dtype == Q8:
            if n_elements % _BLOCK != 0:
                raise NtmreError(
                    "%s: %d elements is not a multiple of 32; Q8 blocks "
                    "are 32 elements and the C loader rejects tensors "
                    "whose size != (n_elements / 32) * 34"
                    % (name, n_elements))
            if len(data) != (n_elements // _BLOCK) * _Q8_BLOCK_BYTES:
                raise NtmreError(
                    "%s: Q8 data is %d bytes but v13 needs %d"
                    % (name, len(data),
                       (n_elements // _BLOCK) * _Q8_BLOCK_BYTES))
        if dtype == Q4:
            need = ((n_elements + _BLOCK - 1) // _BLOCK) * _Q4_BLOCK_BYTES
            if len(data) != need:
                raise NtmreError(
                    "%s: Q4 data is %d bytes but %d elements need %d"
                    % (name, len(data), n_elements, need))
        if dtype == SVD_Q8:
            # Accept packed or flat, reject anything else (see docstring).
            if len(shape) != 2 and len(shape) != 1:
                raise NtmreError("%s: .ntmre compression needs a 1-D or 2-D shape" % name)
            flat_ok = (n_elements % _BLOCK == 0 and
                       len(data) == (n_elements // _BLOCK) * _Q8_BLOCK_BYTES)
            packed_ok = False
            if len(shape) == 2:
                try:
                    unpack_svd(data, shape[0], shape[1])
                    packed_ok = True
                except Exception:
                    packed_ok = False
            if not (flat_ok or packed_ok):
                raise NtmreError(
                    "%s: .ntmre compression data length %d matches neither packed "
                    "nor flat Q8 layout for shape %s"
                    % (name, len(data), shape))
        name_bytes = str(name).encode("utf-8")
        if len(name_bytes) > MAX_NAME_LEN:
            raise NtmreError(
                "tensor name too long for v13 (max 64 bytes): %r" % name)
        if len(name_bytes) == 0:
            raise NtmreError("tensor name must not be empty")
        entries.append((name_bytes, dtype, n_dims, dim0, dim1, data))
    if len(entries) > MAX_TENSORS:
        raise NtmreError(
            "tensor count %d exceeds NTMRE_MAX_TENSORS %d"
            % (len(entries), MAX_TENSORS))

    config_bytes = _pack_config(config or {})
    tok_bytes = _pack_tokenizer(tokenizer)
    header_len = 16 + len(config_bytes) + len(tok_bytes) + 4 +\
        len(entries) * _TENSOR_INFO_SIZE
    data_offset = _align_up(header_len)
    offsets = []
    pos = 0
    for (_, _, _, _, _, data) in entries:
        pos = _align_up(pos)
        offsets.append(pos)
        pos += len(data)

    with open(output_path, "wb") as f:
        f.write(struct.pack("<IIII", MAGIC, VERSION, 0, data_offset))
        f.write(config_bytes)
        f.write(tok_bytes)
        f.write(struct.pack("<I", len(entries)))
        for (name_bytes, dtype, n_dims, dim0, dim1, data), offset in\
                zip(entries, offsets):
            f.write(struct.pack("<B", len(name_bytes)))
            f.write(name_bytes + b"\x00" * (MAX_NAME_LEN - len(name_bytes)))
            f.write(struct.pack("<BBIIQQ", dtype, n_dims, dim0, dim1,
                                offset, len(data)))
        pad = data_offset - f.tell()
        if pad < 0:
            raise NtmreError("internal layout error: header overran data_offset")
        if pad:
            f.write(b"\x00" * pad)
        for (_, _, _, _, _, data), offset in zip(entries, offsets):
            want = data_offset + offset
            cur = f.tell()
            if cur < want:
                f.write(b"\x00" * (want - cur))
            f.write(data)
    return str(output_path)


def read_ntmre(path, load_weights=True):
    """Read a .ntmre file -> (config_dict, tokenizer_dict, tensors).

    tensors is a list of (name, dtype, shape, data) where data is
    raw stored bytes when load_weights is True, and None when False
    (index-only, as fusion_engine uses it).  Shape is (dim0,) for
    1-D tensors and (dim0, dim1) for 2-D tensors.  Raises NtmreError
    (a ValueError) for bad magic, unsupported version, tensor
    count over 4096, unknown n_dims, or truncated data — never
    returns a silently partial index.
    """
    with open(path, "rb") as f:
        raw = f.read(16)
        if len(raw) != 16:
            raise NtmreError("file too short for an NTMRE header")
        magic, version, _flags, header_size = struct.unpack("<IIII", raw)
        if magic != MAGIC:
            raise NtmreError(
                "bad magic 0x%08X, expected 0x%08X (NTMR)" % (magic, MAGIC))
        if version > VERSION:
            raise NtmreError(
                "version %d not supported (max %d)" % (version, VERSION))
        if version < VERSION_MIN_READ:
            raise NtmreError(
                "version %d below minimum readable %d" % (version, VERSION_MIN_READ))
        cfg_raw = f.read(_CONFIG_SIZE)
        if len(cfg_raw) != _CONFIG_SIZE:
            raise NtmreError("truncated config block")
        config = _config_from_bytes(cfg_raw)
        if version < 2:
            config["arch_type"] = ARCH_LLAMA
            config["per_layer_hidden_count"] = 0
            config["per_layer_hidden_dims"] = []
            config["arch"] = "llama"
        th_raw = f.read(_TOKENIZER_HEADER_SIZE)
        if len(th_raw) != _TOKENIZER_HEADER_SIZE:
            raise NtmreError("truncated tokenizer header")
        th = struct.unpack("<10I", th_raw)
        tokenizer = _tokenizer_from_bytes(f, th)
        raw_n = f.read(4)
        if len(raw_n) != 4:
            raise NtmreError("truncated tensor count")
        n_tensors = struct.unpack("<I", raw_n)[0]
        if n_tensors > MAX_TENSORS:
            raise NtmreError(
                "tensor count %d out of range (max %d) — corrupt or "
                "incompatible file" % (n_tensors, MAX_TENSORS))
        infos = []
        for _ in range(n_tensors):
            ti = f.read(_TENSOR_INFO_SIZE)
            if len(ti) != _TENSOR_INFO_SIZE:
                raise NtmreError("truncated tensor index")
            name_len = ti[0]
            if name_len > MAX_NAME_LEN:
                raise NtmreError("tensor name_len %d exceeds 64" % name_len)
            name = ti[1:1 + name_len].decode("utf-8", "replace")
            dtype = ti[65]
            n_dims = ti[66]
            dim0, dim1 = struct.unpack_from("<II", ti, 67)
            offset, size_bytes = struct.unpack_from("<QQ", ti, 75)
            if n_dims == 1:
                shape = (dim0,)
            elif n_dims == 2:
                shape = (dim0, dim1)
            else:
                raise NtmreError(
                    "tensor %r: unsupported n_dims %d" % (name, n_dims))
            infos.append((name, dtype, shape, offset, size_bytes))
        # C reader: data_offset = align_up(current position).
        data_offset = _align_up(f.tell())
        _ = header_size  # written value; C reader recomputes, so do we
        tensors = []
        for name, dtype, shape, offset, size_bytes in infos:
            if not load_weights:
                tensors.append((name, dtype, shape, None))
                continue
            f.seek(data_offset + offset)
            data = f.read(size_bytes)
            if len(data) != size_bytes:
                raise NtmreError(
                    "short read for tensor %r: got %d/%d bytes"
                    % (name, len(data), size_bytes))
            tensors.append((name, dtype, shape, data))
    return config, tokenizer, tensors
