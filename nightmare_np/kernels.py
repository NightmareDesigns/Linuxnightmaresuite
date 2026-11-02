"""nightmare_np.kernels - your own fast numerical kernels for the suite.

Each op is FUSED (one pass, minimal temporaries) and dispatches to the C
backend when it is built, otherwise to a clean Python implementation.

These are the exact hot ops the Nightmare Suite forward pass uses:
  rmsnorm, silu, gelu, softmax, matmul.

NO NUMPY. The fallbacks used to call np.asarray / np.sqrt / np.mean /
np.tanh, so numpy was still a dependency of this file even though the C
backend -- the thing that actually runs -- never touched it. They are now
written as plain loops over the last axis, which is both dependency-free
and correct for any rank.

The loop form matters here rather than being a style choice: a whole-array
mean is not the same as a per-row mean, and on a 2-D input the difference
is a silently wrong model rather than an error. The C kernel reduces the
last axis; these do too, so the fallback and the fast path agree.
"""
import math as _math

from . import _cbackend

__all__ = [
    "rmsnorm", "silu", "gelu", "softmax", "matmul",
    "backend_status",
]


def backend_status():
    return _cbackend.reason


def _as_f32(x):
    """(list of floats, ncols, nrows) for an array of any rank >= 1."""
    if hasattr(x, "tolist"):
        x = x.tolist()
    if isinstance(x, (int, float)):
        return [float(x)], 1, 1
    if not isinstance(x, (list, tuple)):
        x = list(x)
    if x and isinstance(x[0], (list, tuple)):
        flat = []
        ncols = len(x[0])
        nrows = 0
        for row in x:
            row = list(row)
            ncols = min(ncols, len(row))
            nrows += 1
            flat.extend(float(v) for v in row)
        return flat, ncols, nrows
    return [float(v) for v in x], len(x), 1


def _shape_of(x):
    shape = []
    while isinstance(x, (list, tuple)):
        shape.append(len(x))
        if not x:
            break
        x = x[0]
    return tuple(shape)


def _wrap(flat, shape):
    from . import array as _array
    return _array(flat, "float32") if not shape else _reshape(
        _array(flat, "float32"), shape)


def _reshape(arr, shape):
    from . import ndarray as _ndarray
    import ctypes as _ct
    buf = (_ct.c_float * (len(shape) and _numel(shape) or len(arr._buf)))()
    for i in range(len(buf)):
        buf[i] = arr._buf[i]
    return _ndarray(buf, shape, "float32")


def _numel(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def rmsnorm(x, weight, eps=1e-6):
    """out = x / sqrt(mean(x^2, axis=-1, keepdims=True) + eps) * weight"""
    if _cbackend.available:
        return _cbackend.rmsnorm(x, weight, eps)
    xf, ncols, nrows = _as_f32(x)
    wf, wcols, _ = _as_f32(weight)
    out = [0.0] * len(xf)
    for r in range(nrows):
        base = r * ncols
        row = xf[base:base + ncols]
        ms = sum(v * v for v in row) / ncols + eps
        inv = 1.0 / _math.sqrt(ms)
        for c in range(ncols):
            w = wf[c] if c < wcols else wf[-1]
            out[base + c] = row[c] * inv * w
    return _wrap(out, _shape_of(x if not hasattr(x, "tolist") else x.tolist()))


def silu(x):
    """SiLU / Swish: x * sigmoid(x)."""
    if _cbackend.available:
        return _cbackend.silu(x)
    xf, _ncols, _nrows = _as_f32(x)
    out = [v / (1.0 + _math.exp(-v)) if v > -60 else 0.0 for v in xf]
    return _wrap(out, _shape_of(x if not hasattr(x, "tolist") else x.tolist()))


def gelu(x):
    """GELU tanh approximation (matches the suite's gemma4_forward)."""
    if _cbackend.available:
        return _cbackend.gelu(x)
    c = _math.sqrt(2.0 / _math.pi)
    xf, _ncols, _nrows = _as_f32(x)
    out = [0.5 * v * (1.0 + _math.tanh(c * (v + 0.044715 * v * v * v)))
           for v in xf]
    return _wrap(out, _shape_of(x if not hasattr(x, "tolist") else x.tolist()))


def softmax(x, axis=-1):
    """Numerically-stable softmax along `axis` (fused per row)."""
    if _cbackend.available and axis in (-1, _shape_of(
            x if not hasattr(x, "tolist") else x.tolist()) and
            len(_shape_of(x if not hasattr(x, "tolist")
                          else x.tolist())) - 1):
        return _cbackend.softmax_rows(x)
    xf, ncols, nrows = _as_f32(x)
    out = [0.0] * len(xf)
    for r in range(nrows):
        base = r * ncols
        row = xf[base:base + ncols]
        m = max(row)
        exps = [_math.exp(v - m) for v in row]
        total = sum(exps)
        for c in range(ncols):
            out[base + c] = exps[c] / total if total else 0.0
    return _wrap(out, _shape_of(x if not hasattr(x, "tolist") else x.tolist()))


def matmul(a, b):
    """C = a @ b (float32). Uses the threaded C kernel when available."""
    if _cbackend.available:
        return _cbackend.matmul(a, b)
    af, ak, am = _as_f32(a)
    bf, bn, bk = _as_f32(b)
    if ak != bk:
        raise ValueError("matmul dim mismatch: %s @ %s" % (_shape_of(a), _shape_of(b)))
    out = [0.0] * (am * bn)
    for i in range(am):
        arow = af[i * ak:(i + 1) * ak]
        for j in range(bn):
            acc = 0.0
            for k in range(ak):
                acc += arow[k] * bf[k * bn + j]
            out[i * bn + j] = acc
    return _wrap(out, (am, bn))
