"""ctypes loader for the optional C backend (nightmare_np_c.dll).

If the DLL is present and loads, `available` is True and the kernels use it.
Otherwise everything falls back to the pure-Python path in `kernels.py`.

NO NUMPY. This file used to `import numpy as np` for six marshalling
calls -- ascontiguousarray, empty_like, prod, empty -- which meant the one
module that is on every hot path was the one place numpy could not be
removed from. None of those calls needed numpy: what the C functions
actually want is a contiguous float32 buffer, its length, and a place to
write the result.

The marshalling is now _to_f32() below. It accepts:

  * this package's ndarray            -- no copy when already float32
  * any nested list/tuple of numbers
  * any foreign array that exposes .tolist()  (duck typed: if a caller
    hands us a numpy array it still works, without this file importing
    numpy or naming it anywhere)

Duck typing rather than isinstance is deliberate. Checking for numpy's
type would mean importing it. Checking for .tolist() means anything that
behaves like an array works, and numpy is just one more thing that does.
"""
import ctypes
import os

_lib = None
available = False
reason = ""

_DLL_NAME = "nightmare_np_c.dll"


def _load():
    global _lib, available, reason
    dll_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _DLL_NAME)
    if not os.path.exists(dll_path):
        reason = "DLL not built (run build_c.bat for the fast C backend)"
        return
    try:
        lib = ctypes.CDLL(dll_path)
        # --- declare signatures ---
        lib.rmsnorm_f32.restype = None
        lib.rmsnorm_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_float,
        ]
        lib.silu_f32.restype = None
        lib.silu_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.gelu_f32.restype = None
        lib.gelu_f32.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        lib.softmax_rows_f32.restype = None
        lib.softmax_rows_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ]
        lib.matmul_f32.restype = None
        lib.matmul_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.einsum_f32.restype = None
        lib.einsum_f32.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int,
        ]
        _lib = lib
        available = True
        reason = "C backend active"
    except Exception as e:  # pragma: no cover
        _lib = None
        available = False
        reason = f"DLL failed to load: {e}"


_load()


# ---------------------------------------------------------------------------
# marshalling
# ---------------------------------------------------------------------------

def _is_c_float(buf):
    return type(buf).__name__.startswith("c_float")


def _as_sequence(a):
    """
    Turn `a` into something array() can consume.

    Lists and tuples go through untouched. Anything with .tolist() -- this
    package's ndarray and any foreign array alike -- is asked for one.
    Nothing here names numpy; .tolist() is just how arrays answer.
    """
    if isinstance(a, (list, tuple)):
        return a
    t = getattr(a, "tolist", None)
    if callable(t):
        return t()
    return a


def _prod(seq):
    n = 1
    for s in seq:
        n *= s
    return n


def _as_array(x):
    """
    Normalise `x` to (something a raw address can be taken of, shape).

    Three cases, in the order they matter:

      1. This package's ndarray whose buffer is already c_float -- used
         directly, no copy.

      2. A foreign array that is already CONTIGUOUS float32 -- used
         directly, no copy. This is the one that matters. It used to go
         through .tolist(), which turns a 128x4096 tensor into 524288
         Python floats and then a ctypes array one element at a time:
         6.3 seconds to marshal a 34 GB/s-worth of memcpy. The C kernels
         were never the bottleneck; the copy in was.

      3. Anything else -- nested lists, non-contiguous arrays, other
         dtypes -- is converted element by element. Rare, and correct.
    """
    buf = getattr(x, "_buf", None)
    shape = getattr(x, "shape", None)
    if buf is not None and shape is not None and _is_c_float(buf):
        return buf, tuple(shape)

    # already a usable float32 buffer in memory: take its address
    ct = getattr(x, "ctypes", None)
    dt = str(getattr(x, "dtype", ""))
    if ct is not None and shape is not None and "float32" in dt:
        flags = getattr(x, "flags", None)
        contiguous = True
        if flags is not None:
            try:
                contiguous = bool(flags["C_CONTIGUOUS"])
            except Exception:
                contiguous = True
        if contiguous:
            return int(ct.data), tuple(shape)

    from . import array as _array
    arr = _array(_as_sequence(x), "float32")
    return arr._buf, tuple(arr.shape)


def _new(shape, like=None):
    """
    A zeroed contiguous float32 destination of `shape`.

    Returns (buffer, shape, array) where `buffer` is something _ptr() can
    turn into an address and `array` is the object the caller sees.

    `like` is an input whose shape the result shares. When that is a foreign
    array, `like * 0.0` gives a zeroed array of the right shape and dtype in
    one vectorised step, and the array ITSELF is the buffer -- returning
    None here instead sent the kernel a null pointer and it wrote to
    address zero.
    """
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    if like is not None and hasattr(like, "ctypes") and not hasattr(like, "_buf"):
        out = like * 0.0
        return out, shape, out
    from . import ndarray as _ndarray
    n = _prod(shape)
    buf = (ctypes.c_float * n)()
    return buf, shape, _ndarray(buf, shape, "float32")


def _rows_cols(shape):
    """(product of all but the last axis, length of the last axis)."""
    if len(shape) > 1:
        return _prod(shape[:-1]), int(shape[-1])
    return 1, int(shape[0]) if shape else 0


def _ptr(buf):
    """
    Raw address for the c_void_p argument.

    Accepts three things, because _as_array and _new between them produce
    all three:

      * a ctypes array          -> cast
      * an int address          -> wrap as-is
      * a foreign array object  -> take .ctypes.data (so a zeroed result
                                   allocated with `like * 0.0` can be
                                   written into directly)

    ctypes.cast() does NOT accept an int, which is why an int address has
    to be handled explicitly rather than falling through.
    """
    if isinstance(buf, int):
        return ctypes.c_void_p(buf)
    ct = getattr(buf, "ctypes", None)
    if ct is not None and not _is_c_float(buf):
        return ctypes.c_void_p(int(ct.data))
    return ctypes.cast(buf, ctypes.c_void_p)


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------

def _like_of(x):
    """
    `x` if it is a foreign array we can write a same-shaped result into,
    otherwise None.

    Elementwise kernels get their output shape from their input, so the
    caller's own array type can be reused as the destination and nothing has
    to be copied back element by element afterwards.
    """
    if hasattr(x, "ctypes") and not hasattr(x, "_buf"):
        return x
    return None


def rmsnorm(x, w, eps=1e-6):
    xb, xs = _as_array(x)
    wb, _ = _as_array(w)
    out_buf, _shape, out = _new(xs, _like_of(x))
    nrows, ncols = _rows_cols(xs)
    _lib.rmsnorm_f32(_ptr(out_buf), _ptr(xb), _ptr(wb),
                     nrows, ncols, ctypes.c_float(eps))
    return out


def silu(x):
    xb, xs = _as_array(x)
    out_buf, _shape, out = _new(xs, _like_of(x))
    _lib.silu_f32(_ptr(out_buf), _ptr(xb), _prod(xs))
    return out


def gelu(x):
    xb, xs = _as_array(x)
    out_buf, _shape, out = _new(xs, _like_of(x))
    _lib.gelu_f32(_ptr(out_buf), _ptr(xb), _prod(xs))
    return out


def softmax_rows(x):
    xb, xs = _as_array(x)
    out_buf, _shape, out = _new(xs, _like_of(x))
    nrows, ncols = _rows_cols(xs)
    _lib.softmax_rows_f32(_ptr(out_buf), _ptr(xb), nrows, ncols)
    return out


def matmul(a, b):
    """
    C = a @ b

    The output shape (m, n) is not derivable from either input's shape, so
    the result stays in one of our own ctypes buffers -- a single malloc,
    no per-element work. It is converted only if a caller asks for it.
    """
    ab, ashape = _as_array(a)
    bb, bshape = _as_array(b)
    if len(ashape) != 2 or len(bshape) != 2:
        raise ValueError("matmul needs 2-D operands, got %s and %s"
                         % (ashape, bshape))
    m, k = ashape
    k2, n = bshape
    if k != k2:
        raise ValueError(f"matmul dim mismatch: {ashape} @ {bshape}")
    out_buf, _shape, out = _new((m, n))
    _lib.matmul_f32(_ptr(out_buf), _ptr(ab), _ptr(bb), m, k, n)
    return out


def einsum(out, ops, ndims, shapes, subscripts, n_ops):
    """
    Raw call into einsum_f32.

    Kept byte-compatible with the previous signature so __init__.einsum can
    pass its already-built ctypes arrays straight through. The pointer cast
    that made this function work at all lives in __init__.einsum; the same
    helper is used here so both paths cannot drift apart.
    """
    _lib.einsum_f32(_ptr(out), _ptr(ops), _ptr(ndims), _ptr(shapes),
                    subscripts, n_ops)
    return out
