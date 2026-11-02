"""
nightmare_np_svd.py -- .ntmre compression for the suite, from the suite's own C backend
=====================================================================
Gives `convert_to_ntmre.py` a truncated .ntmre compression without numpy.linalg, without
LAPACK, and without installing anything.

Why this exists
---------------
The .ntmre compression converter needs a truncated .ntmre compression of every weight matrix.
nightmare_np has matmul, einsum and rmsnorm, but eigendecomposition is a
different algorithm class -- power iteration cannot reach a rank-512
truncation of a 4096x4096 matrix in reasonable time. So the decomposition
itself lives in C (csrc/nightmare_np_svd.c) and this is the thin binding.

Three things it replaces at once, which is the point:

  * np.linalg.svd       -> np_svd_decompose
  * safetensors.torch   -> load_safetensors (pure stdlib)
  * torch.load          -> handled by the same reader

so the converter runs on a machine with nothing installed, like the rest of
the suite.

Falls back to a pure-Python Jacobi when the C DLL is unavailable, so a build
never dies just because a compiler was missing. Slower, same answer.

Author: Nightmare Designs
"""

import ctypes
import glob
import json
import os
import struct
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
NP_DIR = os.path.join(HERE, "nightmare_np") if os.path.basename(
    HERE) != "nightmare_np" else HERE


# ---------------------------------------------------------------------------
# C backend
# ---------------------------------------------------------------------------

_LIB = None
_TRIED = False


def _load():
    global _LIB, _TRIED
    if _TRIED:
        return _LIB
    _TRIED = True

    # 1. a DLL we build next to this module
    for name in ("nightmare_np_svd.dll", "nightmare_np_svd.so"):
        p = os.path.join(HERE, name)
        if os.path.isfile(p):
            try:
                _LIB = ctypes.CDLL(p)
                break
            except OSError:
                pass

    # 2. the runtime build next to the C source
    if _LIB is None:
        for pat in (os.path.join(NP_DIR, "csrc", "nightmare_np_svd.dll"),
                    os.path.join(NP_DIR, "nightmare_np_svd.dll"),
                    os.path.join(HERE, "nightmare_np", "nightmare_np_svd.dll")):
            if os.path.isfile(pat):
                try:
                    _LIB = ctypes.CDLL(pat)
                    break
                except OSError:
                    pass

    # 3. anywhere under the suite, one level up
    if _LIB is None:
        suite = os.path.abspath(os.path.join(HERE, os.pardir))
        for p in glob.glob(os.path.join(suite, "**", "nightmare_np_svd.dll"),
                           recursive=True):
            try:
                _LIB = ctypes.CDLL(p)
                break
            except OSError:
                pass

    if _LIB is not None:
        P = ctypes.POINTER(ctypes.c_float)
        D = ctypes.POINTER(ctypes.c_double)
        I = ctypes.c_int
        _LIB.np_svd_decompose.argtypes = [P, I, I, P, D, P, I, I]
        _LIB.np_svd_decompose.restype = I
        _LIB.np_svd_rank_for_energy.argtypes = [D, I, ctypes.c_double]
        _LIB.np_svd_rank_for_energy.restype = I
        _LIB.np_svd_reconstruct_error.argtypes = [P, I, I, P, D, P, I]
        _LIB.np_svd_reconstruct_error.restype = ctypes.c_double
    return _LIB


def backend() -> str:
    return "C" if _load() else "python"


# ---------------------------------------------------------------------------
# Pure-python fallback: same algorithm, same layout, much slower.
# Only used so a missing DLL degrades speed instead of breaking the build.
# ---------------------------------------------------------------------------

def _svd_python(rows: List[List[float]], m: int, n: int,
                max_rank: int, max_sweeps: int):
    import math
    a = [list(map(float, r)) for r in rows]
    u = [list(r) for r in a]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for _sweep in range(max(6, max_sweeps)):
        off = 0.0
        for p in range(n - 1):
            for q in range(p + 1, n):
                alpha = beta = gamma = 0.0
                for i in range(m):
                    up, uq = u[i][p], u[i][q]
                    alpha += up * up
                    beta += uq * uq
                    gamma += up * uq
                if abs(gamma) < 1e-18:
                    continue
                off += abs(gamma)
                zeta = (beta - alpha) / (2.0 * gamma)
                if zeta >= 0:
                    t = 1.0 / (zeta + math.sqrt(1.0 + zeta * zeta))
                else:
                    t = -1.0 / (-zeta + math.sqrt(1.0 + zeta * zeta))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = c * t
                for i in range(m):
                    up, uq = u[i][p], u[i][q]
                    u[i][p] = c * up - s * uq
                    u[i][q] = s * up + c * uq
                for i in range(n):
                    vp, vq = v[i][p], v[i][q]
                    v[i][p] = c * vp - s * vq
                    v[i][q] = s * vp + c * vq
        if off < 1e-14:
            break

    s_all = []
    for j in range(n):
        acc = 0.0
        for i in range(m):
            acc += u[i][j] * u[i][j]
        s_all.append(math.sqrt(acc))

    order = sorted(range(n), key=lambda c: -s_all[c])
    s_sorted = [s_all[c] for c in order]
    u = [[u[i][c] for c in order] for i in range(m)]
    v = [[v[r][c] for c in order] for r in range(n)]

    k = max_rank if (0 < max_rank < n) else n
    U = []
    for j in range(k):
        col = []
        s = s_sorted[j]
        for i in range(m):
            col.append(u[i][j] / s if s > 1e-300 else 0.0)
        U.append(col)
    Vt = [[v[i][j] for i in range(n)] for j in range(k)]
    return U, s_sorted[:k], Vt


def np_svd(matrix, max_rank: int = 0, max_sweeps: int = 12
           ) -> Tuple[List[List[float]], List[float], List[List[float]]]:
    """
    Truncated .ntmre compression of a 2-D sequence-of-rows. Returns (U, S, Vt) with

        matrix ~= U[k][m] * diag(S[k]) * Vt[k][n]

    U is k rows of length m, Vt is k rows of length n, S is descending.
    That is the layout svd_weight.py expects: U @ diag(S) @ Vt.

    Accepts anything indexable -- nested lists, numpy arrays, memoryviews --
    so it drops into existing code without reshaping the call sites.
    """
    rows = list(matrix)
    m = len(rows)
    if m == 0:
        return [], [], []
    n = len(rows[0])
    if n == 0:
        return [], [], []
    k = max_rank if (0 < max_rank < n) else n

    lib = _load()
    if lib is None:
        return _svd_python(rows, m, n, k, max_sweeps)

    # Flatten row-major into the float32 buffer the C side expects.
    flat = (ctypes.c_float * (m * n))()
    idx = 0
    for r in rows:
        for value in r:
            flat[idx] = float(value)
            idx += 1

    U = (ctypes.c_float * (k * m))()
    S = (ctypes.c_double * k)()
    V = (ctypes.c_float * (k * n))()

    rc = lib.np_svd_decompose(flat, m, n, U, S, V, k, max_sweeps)
    if rc != 1:
        return _svd_python(rows, m, n, k, max_sweeps)

    Uu = [[U[t * m + i] for i in range(m)] for t in range(k)]
    Ss = [S[t] for t in range(k)]
    Vv = [[V[t * n + i] for i in range(n)] for t in range(k)]
    return Uu, Ss, Vv


def find_energy_rank(S, threshold: float = 0.999) -> int:
    """Smallest k whose leading k singular values hold `threshold` energy."""
    s = list(S)
    if not s:
        return 0
    total = sum(x * x for x in s)
    if total <= 0:
        return len(s)
    acc = 0.0
    for i, x in enumerate(s):
        acc += x * x
        if acc / total >= threshold:
            return i + 1
    return len(s)


def reconstruct_error(matrix, U, S, Vt) -> float:
    """Relative Frobenius error of the approximation. The honesty check."""
    rows = list(matrix)
    m, n = len(rows), len(rows[0])
    num = den = 0.0
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for t in range(len(S)):
                acc += U[t][i] * S[t] * Vt[t][j]
            d = acc - rows[i][j]
            num += d * d
            den += rows[i][j] * rows[i][j]
    if den <= 0:
        return 0.0
    return (num / den) ** 0.5


# ---------------------------------------------------------------------------
# Weight readers -- replaces safetensors.torch and torch.load
# ---------------------------------------------------------------------------

def load_safetensors(path: str) -> Dict[str, List[List[float]]]:
    """
    Read a .safetensors file with the standard library only.

    Format, which is genuinely simple:
        u64  header length N
        N    JSON header: {name: {dtype, shape, data_offsets}}
        rest raw tensor bytes

    Sharded models are handled by the caller; this reads one shard and
    returns every tensor it holds.
    """
    tensors: Dict[str, List[List[float]]] = {}
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n).decode("utf-8"))
        base = 8 + n

        for name, info in header.items():
            if name == "__metadata__":
                continue
            dtype = info.get("dtype", "F32")
            shape = info.get("shape", [])
            start, end = info["data_offsets"]
            fh.seek(base + start)
            raw = fh.read(end - start)

            if dtype in ("F32", "F16", "BF16"):
                vals = _decode_floats(raw, dtype)
            elif dtype in ("I64", "I32"):
                fmt = "<q" if dtype == "I64" else "<i"
                vals = list(struct.unpack("<%d%s" % (len(raw) // 8, fmt[1]),
                                         raw[:len(raw) // 8 * 8]))
            else:
                # Unsupported dtype: skip rather than produce garbage.
                continue

            tensors[name] = _reshape(vals, shape)
    return tensors


def _decode_floats(raw: bytes, dtype: str) -> List[float]:
    if dtype == "F32":
        return list(struct.unpack("<%df" % (len(raw) // 4), raw))
    if dtype == "F16":
        out = []
        for (h,) in struct.unpack("<%dH" % (len(raw) // 2), raw):
            out.append(_f16_to_f32(h))
        return out
    # BF16: the top 16 bits of a float32.
    out = []
    for (h,) in struct.unpack("<%dH" % (len(raw) // 2), raw):
        (f,) = struct.unpack("<f", struct.pack("<I", h << 16))
        out.append(f)
    return out


def _f16_to_f32(h: int) -> float:
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        return float((-1.0 if sign else 1.0) * (mant / 1024.0)
                     * (1.0 / 16384.0))
    if exp == 31:
        return 0.0
    (f,) = struct.unpack("<f", struct.pack(
        "<I", (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13)))
    return f


def _reshape(flat: List[Any], shape: List[int]) -> List[Any]:
    if not shape:
        return flat
    if len(shape) == 1:
        return list(flat)
    stride = 1
    for d in shape[1:]:
        stride *= d
    out = []
    for i in range(0, len(flat), stride) if stride else []:
        chunk = flat[i:i + stride]
        out.append(_reshape(chunk, shape[1:]) if len(shape) > 2
                   else list(chunk))
    return out


def load_gguf(path: str) -> Dict[str, Any]:
    """
    Read a single-tensor-per-entry GGUF.

    GGUF is universal across the rest of the suite, so the diffusion
    converter should accept it too rather than being the one tool that
    needs a different file. Returns
        {"tensors": {name: rows}, "meta": {...}}

    Quantized tensor types are reported in meta["unsupported"] rather than
    silently dequantised -- a diffusion unet shipped as Q8 would come out
    wrong, and a loud list beats a quiet bad image.
    """
    meta: Dict[str, Any] = {}
    tensors: Dict[str, List[List[float]]] = {}
    unsupported: List[str] = []

    with open(path, "rb") as fh:
        if fh.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, = struct.unpack("<I", fh.read(4))
        n_tensors, n_kv = struct.unpack("<QQ", fh.read(16))

        # --- key/value metadata -------------------------------------
        pos = 0
        for _ in range(n_kv):
            key, pos = _read_string(fh, pos)
            vtype, pos = _read_u32(fh, pos)
            val, pos = _read_meta(fh, pos, vtype)
            meta[key] = val

        # --- tensor directory, then a bulk data block ---------------
        info = []
        for _ in range(n_tensors):
            name, pos = _read_string(fh, pos)
            n_dims, pos = _read_u32(fh, pos)
            dims = []
            for _d in range(n_dims):
                d, pos = _read_u64(fh, pos)
                dims.append(d)
            ttype, pos = _read_u32(fh, pos)
            toff, pos = _read_u64(fh, pos)
            info.append((name, dims, ttype, toff))

        base = fh.tell()

        GGML_F32 = 0
        GGML_F16 = 1
        type_size = {GGML_F32: 4, GGML_F16: 2}

        for name, dims, ttype, toff in info:
            count = 1
            for d in dims:
                count *= d
            if ttype not in type_size:
                unsupported.append(name)
                continue
            nbytes = count * type_size[ttype]
            fh.seek(base + toff)
            raw = fh.read(nbytes)
            if ttype == GGML_F32:
                vals = list(struct.unpack("<%df" % count, raw))
            else:
                vals = []
                for (h,) in struct.unpack("<%dH" % count, raw):
                    vals.append(_f16_to_f32(h))
            tensors[name] = _reshape(vals, dims)

    meta["unsupported"] = unsupported
    meta["version"] = version
    return {"tensors": tensors, "meta": meta}


def _read_u32(fh, pos):
    fh.seek(pos)
    v, = struct.unpack("<I", fh.read(4))
    return v, pos + 4


def _read_u64(fh, pos):
    fh.seek(pos)
    v, = struct.unpack("<Q", fh.read(8))
    return v, pos + 8


def _read_string(fh, pos) -> Tuple[str, int]:
    (ln,) = struct.unpack("<Q", fh.read(8))
    s = fh.read(ln).decode("utf-8", errors="replace")
    return s, pos + 8 + ln


def _read_meta(fh, pos, vtype):
    if vtype == 8:                      # string
        return _read_string(fh, pos)
    if vtype == 9:                      # array
        (arr_type,) = struct.unpack("<I", fh.read(4))
        (n,) = struct.unpack("<Q", fh.read(8))
        out = []
        p = pos
        for _ in range(n):
            val, p = _read_meta(fh, p, arr_type)
            out.append(val)
        return out, p
    if vtype in (0, 1, 2, 3):           # u8 i8 u16 i16
        size = {0: 1, 1: 1, 2: 2, 3: 2}[vtype]
        fh.seek(pos)
        return int.from_bytes(fh.read(size), "little", signed=vtype in (1, 3)), pos + size
    if vtype == 4:                      # u32
        v, p = _read_u32(fh, pos)
        return v, p
    if vtype == 5:                      # i32
        v, p = _read_u32(fh, pos)
        return v - (1 << 32) if v >= (1 << 31) else v, p
    if vtype in (6, 7):                 # f32 f64
        size = 4 if vtype == 6 else 8
        fmt = "<f" if vtype == 6 else "<d"
        fh.seek(pos)
        (v,) = struct.unpack(fmt, fh.read(size))
        return v, pos + size
    if vtype == 10:                     # bool
        v, p = _read_u32(fh, pos)
        return bool(v), p
    if vtype == 11:                     # string array
        v, p = _read_meta(fh, pos, 9)
        return v, p
    fh.seek(pos)
    return None, pos + 1


def load_any(path: str) -> Dict[str, Any]:
    """
    One entry point for the converter, whatever the container.

    .safetensors -> the stdlib reader
    .gguf        -> the GGUF reader
    .pt/.pth/.bin -> refused with a clear reason

    A torch pickle is a Python object graph, not a tensor dump. Reading one
    without torch is not a format problem, it is an interpreter problem, and
    pretending otherwise would be the exact failure mode this whole module
    exists to prevent.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".safetensors":
        return {"tensors": load_safetensors(path), "meta": {}}
    if ext == ".gguf":
        return load_gguf(path)
    if ext in (".pt", ".pth", ".bin"):
        raise ValueError(
            "%s is a torch pickle. Convert it to safetensors or gguf first; "
            "reading one without torch is not supported and will not be "
            "guessed at." % os.path.basename(path))
    # unknown extension: sniff the magic
    with open(path, "rb") as fh:
        magic = fh.read(4)
    if magic == b"GGUF":
        return load_gguf(path)
    return {"tensors": load_safetensors(path), "meta": {}}


if __name__ == "__main__":
    print("backend:", backend())

    # known-answer test
    A = [[3.0, 0.0], [0.0, 2.0]]
    U, S, Vt = np_svd(A, 2, 20)
    print("diag(3,2) -> S =", ["%.4f" % s for s in S])
    print("reconstruct error: %.3e" % reconstruct_error(A, U, S, Vt))

    B = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]]      # rank 1, wide
    U, S, Vt = np_svd(B, 2, 20)
    print("rank-1 wide -> S =", ["%.4f" % s for s in S],
          "(expect sqrt(70)=8.3666)")
    print("reconstruct error: %.3e" % reconstruct_error(B, U, S, Vt))

    print("energy rank @0.999 for [10,5,3,2,.1]:",
          find_energy_rank([10.0, 5.0, 3.0, 2.0, 0.1], 0.999))
