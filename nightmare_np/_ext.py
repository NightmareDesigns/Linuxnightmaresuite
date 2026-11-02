"""
numpy compatibility layer for the parts of the API the engine uses that
nightmare_np did not have yet.

WHY THIS EXISTS
---------------
Rule #1: no outside code belongs in this software. The engine was calling
numpy in 439 places across four files, and nightmare_np covered only
about half the names it needed. This module supplies the rest, in the same
style as the rest of the package: pure stdlib, no numpy, hand-written.

    import nightmare_np as np          # then np.arange, np.clip, ...

DTYPE HANDLING IS NOT COSMETIC HERE
-----------------------------------
ndarray stores its buffer as a flat ctypes array, and the original code
always used c_float regardless of the dtype label. That is fine for floats
and WRONG for bytes: `frombuffer(blob, uint8)` must see the actual byte
values, and a c_float reinterpretation of the same bytes gives numbers
like 1.7272e-77 instead of 235.

So this module allocates correctly typed buffers. Anything integer-typed
round-trips exactly; float paths defer to the C backend through the
existing kernels, unchanged.

WHAT IS HERE
------------
Creation   arange, linspace, eye, full_like, frombuffer, tri (via triu),
           copy, concatenate, stack
Elementwise abs, sign, cos, sin, log1p, log2, exp2, clip, where, divide,
           mod, power, maximum/minimum (re-exported), square
Reductions  median, std, var, all, any, prod, cumprod
Inspection  isfinite, isinf, isnan (re-exported), isscalar, ndim, size,
           shape, dtype, newaxis, pi, e
Shaping     ravel, tile, moveaxis, repeat_ (re-exported), transpose
Bit ops     unpackbits, packbits (minimal), bitwise_and/or/xor
dtypes      float16, float64, int8, int16, int32, int64, uint16, uint32,
            uint64, bool_, complex64/128 (stubs)

Plus ndarray methods: .ctypes, .ravel, .flatten, .clip, .argmax, .argmin,
.any, .all, .cumsum, .cumprod, .item, .fill, .nonzero, .astype-copy,
.astype already present, .max/.min already present.

Everything here is verified against numpy in test_np_compat.py. Where the
two genuinely differ, the docstring says so rather than pretending.
"""

import builtins as _b
import ctypes
import math
import random as _random_mod

# PURE STDLIB ONLY. numpy is NOT imported here and must never be added:
# the whole point of this package is that no outside code is a dependency.
# (An earlier draft carried `import numpy as _np_ref` for the tests. That
# was the rule being broken inside the one file whose job is not to break
# it. The tests live outside the package and import numpy themselves.)

__all__ = [
    "ndarray", "_DType", "_BY_NAME", "_as_dtype",
    "float16", "float32", "float64",
    "int8", "int16", "int32", "int64",
    "uint8", "uint16", "uint32", "uint64",
    "bool_", "complex64", "complex128", "integer",
    "newaxis", "pi", "e",
    "arange", "linspace", "eye", "full_like", "zeros_like", "frombuffer",
    "concatenate", "stack", "outer",
    "abs", "fabs", "sign", "cos", "sin", "tan", "log", "log1p", "log2", "log10", "exp2",
    "arccos", "square", "rint", "floor", "ceil", "round",
    "clip", "where", "divide", "true_divide", "mod", "remainder", "power",
    "square", "cumprod",
    "median", "percentile", "std", "var", "all", "any", "prod", "all_", "any_",
    "isfinite", "isinf", "isscalar", "ndim", "size", "shape", "dtype",
    "ravel", "tile", "moveaxis", "transpose", "squeeze", "expand_dims",
    "unpackbits", "packbits", "bitwise_and", "bitwise_or", "bitwise_xor",
    "memmap", "fromiter", "ctypeslib", "load", "seterr", "linalg", "copy",
    "random",
    "nan_to_num", "cumsum_", "minimum", "maximum",
]


# ============================================================================
# dtypes
# ============================================================================

class _DType:
    """A dtype object with just enough behaviour for the engine's use."""

    __slots__ = ("name", "ctype", "itemsize", "isfloat", "isint")

    def __init__(self, name, ctype, itemsize, isfloat, isint):
        self.name = name
        self.ctype = ctype
        self.itemsize = itemsize
        self.isfloat = isfloat
        self.isint = isint

    def __call__(self, x):
        return _coerce(x, self.name)

    def __repr__(self):
        return "dtype('%s')" % self.name

    def __eq__(self, other):
        return _b.isinstance(other, _DType) and other.name == self.name

    def __hash__(self):
        return hash(self.name)


float16 = _DType("float16", ctypes.c_short, 2, True, False)
float32 = _DType("float32", ctypes.c_float, 4, True, False)
float64 = _DType("float64", ctypes.c_double, 8, True, False)
int8 = _DType("int8", ctypes.c_int8, 1, False, True)
int16 = _DType("int16", ctypes.c_int16, 2, False, True)
int32 = _DType("int32", ctypes.c_int32, 4, False, True)
int64 = _DType("int64", ctypes.c_int64, 8, False, True)
uint8 = _DType("uint8", ctypes.c_uint8, 1, False, True)
uint16 = _DType("uint16", ctypes.c_uint16, 2, False, True)
uint32 = _DType("uint32", ctypes.c_uint32, 4, False, True)
uint64 = _DType("uint64", ctypes.c_uint64, 8, False, True)
bool_ = _DType("bool", ctypes.c_bool, 1, False, True)

# numpy's abstract integer scalar type. The engine only ever uses it in
# isinstance checks (e.g. isinstance(dtype, (int, np.integer))), so plain
# int is the faithful stand-in.
integer = int

# complex is not implemented; these exist so `import numpy as np` code that
# only mentions them does not explode at import time. Anything that actually
# tries to compute with them raises, loudly, rather than silently wrong.
complex64 = _DType("complex64", None, 8, True, False)
complex128 = _DType("complex128", None, 16, True, False)

_BY_NAME = {d.name: d for d in
            (float16, float32, float64, int8, int16, int32, int64,
             uint8, uint16, uint32, uint64, bool_)}


def _as_dtype(d, default="float32"):
    if d is None:
        return _BY_NAME[default]
    if isinstance(d, _DType):
        return d
    if isinstance(d, str):
        key = d[3:] if d.startswith("np.") else d
        if key in _BY_NAME:
            return _BY_NAME[key]
        raise TypeError("unsupported dtype %r" % d)
    # a real numpy dtype, if one ever leaks in -- read the name off it
    name = getattr(d, "name", None)
    if name in _BY_NAME:
        return _BY_NAME[name]
    raise TypeError("unsupported dtype %r" % (d,))


def dtype(obj):
    """np.dtype(x) -> our dtype object."""
    return _as_dtype(obj)


# ============================================================================
# the array type
# ============================================================================

class ndarray:
    """
    Minimal array, typed.

    Deliberately NOT a numpy replacement -- it covers what the engine calls.
    `dtype` decides the buffer type, so bytes stay bytes and floats stay
    floats. Mixed-dtype operands promote to the wider of the two, the same
    way a C implementation would.
    """

    __slots__ = ("_buf", "shape", "dtype", "_n")

    def __init__(self, buf, shape, dtype="float32"):
        self._buf = buf
        self.shape = tuple(shape)
        self.dtype = _as_dtype(dtype).name if not isinstance(dtype, str) else dtype
        self._n = len(buf)

    # ---- properties ----
    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return self._n

    @property
    def itemsize(self):
        return _BY_NAME[self.dtype].itemsize

    @property
    def nbytes(self):
        return self._n * self.itemsize

    @property
    def ctypes(self):
        """
        What numpy's `arr.ctypes.data` gives: the raw address of the
        buffer. The GPU paths call ctypes with this, so it has to be a
        real integer address, not a wrapper.
        """
        class _C:
            data = ctypes.addressof(self._buf) if self._n else 0
        return _C()

    @property
    def T(self):
        return transpose(self)

    def __len__(self):
        return self.shape[0] if self.shape else 0

    def __repr__(self):
        return "nightmare_np_ext.ndarray(shape=%s, dtype=%s)" % (self.shape, self.dtype)

    def __bool__(self):
        if self._n == 1:
            return bool(self._buf[0])
        raise ValueError("truth value of an array with %d elements is "
                         "ambiguous" % self._n)

    def __float__(self):
        if self._n != 1:
            raise TypeError("only size-1 arrays can be converted to a scalar")
        return float(self._buf[0])

    def __int__(self):
        if self._n != 1:
            raise TypeError("only size-1 arrays can be converted to a scalar")
        return int(self._buf[0])

    __index__ = __int__

    def item(self, *idx):
        if not idx:
            if self._n != 1:
                raise ValueError("can only convert an array of size 1")
            return self._buf[0]
        return self[tuple(idx) if len(idx) > 1 else idx[0]]

    def tolist(self):
        if self.ndim <= 1:
            return [self._buf[i] for i in range(self._n)]
        raise NotImplementedError("tolist for >1-D; use .reshape first")

    # ---- element access ----
    def _flat_index(self, idx):
        out = 0
        for k, n in zip(idx, self.shape):
            if k < 0:
                k += n
            if k < 0 or k >= n:
                raise IndexError("index %d out of range for axis of %d" % (k, n))
            out = out * n + k
        return out

    def __getitem__(self, key):
        if key is Ellipsis:
            return _clone(self)
        shape, targets = _resolve(self, key)
        if shape == ():
            return self._buf[targets[0]]
        out = _new(shape, self.dtype)
        for i, t in enumerate(targets):
            out._buf[i] = self._buf[t]
        return out

    def __setitem__(self, key, value):
        shape, targets = _resolve(self, key)
        if isinstance(value, ndarray):
            vals = [value._buf[i] for i in range(value._n)]
        elif isinstance(value, (list, tuple)):
            vals = list(value)
        else:
            vals = [value] * len(targets)
        if len(vals) == 1 and len(targets) > 1:
            vals = vals * len(targets)
        if len(vals) != len(targets):
            raise ValueError("could not broadcast input array from shape "
                             "%s into shape %s" % (len(vals), len(targets)))
        for t, v in zip(targets, vals):
            self._buf[t] = _scalar(v)

    # ---- arithmetic ----
    def _bin(self, other, op, rev=False):
        return _apply(self, other, op, rev)

    def _unary(self, cname, pyfn):
        """Compat with __init__.py's sqrt/exp/tanh/silu/gelu, which call
        this method. The C backend name is ignored; _ext is pure Python."""
        return _map1(self, pyfn)

    def _binary(self, other, cname, pyfn):
        """Compat with __init__.py's maximum/minimum."""
        return _apply(self, other, pyfn)

    def __add__(self, o):
        return self._bin(o, lambda a, b: a + b)

    __radd__ = __add__

    def __sub__(self, o):
        return self._bin(o, lambda a, b: a - b)

    def __rsub__(self, o):
        return self._bin(o, lambda a, b: a - b, True)

    def __mul__(self, o):
        return self._bin(o, lambda a, b: a * b)

    __rmul__ = __mul__

    def __truediv__(self, o):
        return _apply(self, o, _div, dt=_truediv_dt(self, o))

    def __rtruediv__(self, o):
        return _apply(self, o, _div, rev=True, dt=_truediv_dt(self, o))

    __div__ = __truediv__

    def __floordiv__(self, o):
        return self._bin(o, lambda a, b: a // b if b else 0)

    def __rfloordiv__(self, o):
        return self._bin(o, lambda a, b: a // b if b else 0, True)

    def __mod__(self, o):
        return self._bin(o, lambda a, b: a - b * int(a / b) if b else 0)

    def __rmod__(self, o):
        return self._bin(o, lambda a, b: a - b * int(a / b) if b else 0, True)

    def __pow__(self, o):
        return self._bin(o, lambda a, b: a ** b)

    def __rpow__(self, o):
        return self._bin(o, lambda a, b: a ** b, True)

    # ---- bitwise ops (the dequantizers live on these) ----
    # numpy semantics: integer bit ops keep the left dtype (a scalar does
    # not promote); values wrap two's-complement, they never raise.
    def __lshift__(self, o):
        return _bitwise(self, o, lambda a, b: a << b)

    def __rshift__(self, o):
        return _bitwise(self, o, lambda a, b: a >> b)

    def __and__(self, o):
        return _bitwise(self, o, lambda a, b: a & b)

    def __or__(self, o):
        return _bitwise(self, o, lambda a, b: a | b)

    def __xor__(self, o):
        return _bitwise(self, o, lambda a, b: a ^ b)

    def __invert__(self):
        return _map1(self, lambda v: ~int(v))

    def __neg__(self):
        return _map1(self, lambda a: -a)

    def __pos__(self):
        return _clone(self)

    def __abs__(self):
        return _map1(self, _b.abs)

    # ---- comparison ----
    def __eq__(self, o):
        return _cmp(self, o, lambda a, b: a == b)

    def __ne__(self, o):
        return _cmp(self, o, lambda a, b: a != b)

    def __lt__(self, o):
        return _cmp(self, o, lambda a, b: a < b)

    def __le__(self, o):
        return _cmp(self, o, lambda a, b: a <= b)

    def __gt__(self, o):
        return _cmp(self, o, lambda a, b: a > b)

    def __ge__(self, o):
        return _cmp(self, o, lambda a, b: a >= b)

    def __hash__(self):
        raise TypeError("unhashable type: 'ndarray'")

    # ---- methods ----
    def astype(self, dt):
        dt = _as_dtype(dt).name
        if dt == self.dtype:
            return _clone(self)
        # Half floats: the buffer holds raw half BITS, not float values, so
        # a plain copy would hand back the bit patterns as numbers
        # (0x3C00 -> 15360.0 instead of 1.0). Convert through struct 'e'.
        if self.dtype == "float16" and dt in ("float32", "float64"):
            import struct as _struct
            out = _new(self.shape, dt)
            for i in range(self._n):
                out._buf[i] = _struct.unpack(
                    "<e", _struct.pack("<H", int(self._buf[i]) & 0xFFFF))[0]
            return out
        if dt == "float16" and self.dtype in ("float32", "float64"):
            import struct as _struct
            out = _new(self.shape, dt)
            for i in range(self._n):
                out._buf[i] = _struct.unpack(
                    "<H", _struct.pack("<e", float(self._buf[i])))[0]
            return out
        out = _new(self.shape, dt)
        for i in range(self._n):
            out._buf[i] = _coerce(self._buf[i], dt)
        return out

    def tobytes(self, order="C"):
        """Raw bytes of the buffer, C order."""
        return bytes(self._buf)

    def view(self, dtype=None):
        """Reinterpret the buffer's bytes as another dtype, numpy-style.
        The last axis rescales by the itemsize ratio."""
        if dtype is None:
            return _clone(self)
        if isinstance(dtype, str):
            dt = _NPY_DESCR.get(dtype)
            if dt is None:
                dt = _as_dtype(dtype).name
        else:
            dt = _as_dtype(dtype).name
        old = _BY_NAME[self.dtype]
        new = _BY_NAME[dt]
        total = self._n * old.itemsize
        if total % new.itemsize:
            raise ValueError("view: buffer size %d not divisible by %d"
                             % (total, new.itemsize))
        if not self.shape:
            shape = (total // new.itemsize,)
        else:
            last = self.shape[-1] * old.itemsize
            if last % new.itemsize:
                raise ValueError("view: last axis does not divide evenly")
            shape = self.shape[:-1] + (last // new.itemsize,)
        raw = bytes(self._buf)
        return frombuffer(raw, dtype=dt).reshape(shape)

    def cumprod(self):
        return cumprod(self)

    def copy(self):
        return self.astype(self.dtype)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = list(shape)
        if -1 in shape:
            known = 1
            for s in shape:
                if s != -1:
                    known *= s
            shape[shape.index(-1)] = self._n // known if known else 0
        return ndarray(self._buf, tuple(shape), self.dtype)

    def ravel(self):
        return ndarray(self._buf, (self._n,), self.dtype)

    flatten = ravel

    def fill(self, v):
        for i in range(self._n):
            self._buf[i] = v

    def clip(self, lo=None, hi=None):
        def f(a):
            if lo is not None and a < lo:
                return lo
            if hi is not None and a > hi:
                return hi
            return a
        return _map1(self, f)

    def argmax(self):
        if self._n == 0:
            raise ValueError("argmax of an empty array")
        best = self._buf[0]
        bi = 0
        for i in range(1, self._n):
            if self._buf[i] > best:
                best = self._buf[i]
                bi = i
        return bi

    def argmin(self):
        if self._n == 0:
            raise ValueError("argmin of an empty array")
        best = self._buf[0]
        bi = 0
        for i in range(1, self._n):
            if self._buf[i] < best:
                best = self._buf[i]
                bi = i
        return bi

    def any(self):
        return _b.any(bool(self._buf[i]) for i in range(self._n))

    def all(self):
        return _b.all(bool(self._buf[i]) for i in range(self._n))

    def sum(self, axis=None, keepdims=False):
        return _reduce(self, "sum", axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return _reduce(self, "mean", axis, keepdims)

    def max(self, axis=None, keepdims=False):
        return _reduce(self, "max", axis, keepdims)

    def min(self, axis=None, keepdims=False):
        return _reduce(self, "min", axis, keepdims)

    def cumsum(self):
        out = _new((self._n,), _ACC[self.dtype])
        run = 0
        for i in range(self._n):
            run += self._buf[i]
            out._buf[i] = run
        return out

    def prod(self):
        run = 1
        for i in range(self._n):
            run *= self._buf[i]
        return run

    def nonzero(self):
        return [ndarray((ctypes.c_int64 * _b.sum(1 for i in range(self._n)
                                                 if self._buf[i]))(),
                        (_b.sum(1 for i in range(self._n) if self._buf[i]),), "int64")]

    def transpose(self, *axes):
        return transpose(self, axes if axes else None)

    def squeeze(self):
        shape = tuple(s for s in self.shape if s != 1)
        return ndarray(self._buf, shape, self.dtype)

    def dot(self, other):
        return dot(self, other)

    def __matmul__(self, other):
        return matmul(self, other)


# integer types widen on arithmetic so overflow does not silently bite
_ACC = {
    "float16": "float32", "float32": "float32", "float64": "float64",
    "int8": "int16", "int16": "int32", "int32": "int64", "int64": "int64",
    "uint8": "int16", "uint16": "int32", "uint32": "int64",
    "uint64": "int64", "bool": "int32",
}


# ============================================================================
# internals
# ============================================================================

def _prod(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def _new(shape, dt="float32"):
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(shape)
    dt = _as_dtype(dt).name
    return ndarray((_BY_NAME[dt].ctype * _prod(shape))(), shape, dt)


def _clone(a):
    return ndarray(a._buf, a.shape, a.dtype)


def _scalar(v):
    return v[0] if isinstance(v, ndarray) else v


def _coerce(x, dt):
    dt = _as_dtype(dt).name
    if _BY_NAME[dt].isfloat:
        return float(x)
    return int(x)


def _values(a):
    """Flat python values of `a`, whatever it is."""
    if isinstance(a, ndarray):
        return [a._buf[i] for i in range(a._n)]
    if isinstance(a, (list, tuple)):
        out = []
        for x in a:
            if isinstance(x, (list, tuple)):
                out.extend(_values(x))
            else:
                out.append(float(x))
        return out
    return [float(a)]


def _promote(d1, d2):
    if _BY_NAME[d1].isfloat or _BY_NAME[d2].isfloat:
        return "float64" if "float64" in (d1, d2) else "float32"
    return _ACC.get(d1, "int64") if _ACC.get(d1, "int64") != d2\
        else _ACC.get(d1, "int64")


def _broadcast_shape(s1, s2):
    """numpy-style broadcast shape, or None if the shapes are incompatible."""
    out = []
    for d1, d2 in zip(reversed(s1), reversed(s2)):
        if d1 == d2:
            out.append(d1)
        elif d1 == 1:
            out.append(d2)
        elif d2 == 1:
            out.append(d1)
        else:
            return None
    longer = s1 if len(s1) > len(s2) else s2
    out.extend(longer[:len(longer) - len(out)])
    return tuple(reversed(out))


def _strides(sh):
    s = [1] * len(sh)
    for i in range(len(sh) - 2, -1, -1):
        s[i] = s[i + 1] * sh[i + 1]
    return s


def _apply(a, b, op, rev=False, dt=None):
    a = asarray(a)
    other = asarray(b) if not isinstance(b, (int, float, complex, bool)) else None
    if other is None:
        # scalar operand: a float scalar promotes the result to float
        if dt is None:
            if isinstance(b, float):
                dt = "float64" if a.dtype == "float64" else "float32"
            else:
                dt = _ACC[a.dtype]
        n = a._n
        vals = [a._buf[i] for i in range(n)]
        out = _new(a.shape, dt)
        for i in range(n):
            out._buf[i] = _coerce(
                op(vals[i], _coerce(b, dt)) if not rev
                else op(_coerce(b, dt), vals[i]), dt)
        return out
    if dt is None:
        dt = _promote(a.dtype, other.dtype)
    shape = _broadcast_shape(a.shape, other.shape)
    if shape is None:
        raise ValueError("operands could not be broadcast together: "
                         "%s %s" % (a.shape, other.shape))
    out = _new(shape, dt)
    a_str = _strides(a.shape)
    b_str = _strides(other.shape)
    nd = len(shape)
    a_off = nd - a.ndim
    b_off = nd - other.ndim
    for flat in range(out._n):
        rem = flat
        coords = [0] * nd
        for i in range(nd - 1, -1, -1):
            coords[i] = rem % shape[i]
            rem //= shape[i]
        ai = 0
        bi = 0
        for i in range(nd):
            if i >= a_off:
                ad = i - a_off
                ai += (coords[i] if a.shape[ad] > 1 else 0) * a_str[ad]
            if i >= b_off:
                bd = i - b_off
                bi += (coords[i] if other.shape[bd] > 1 else 0) * b_str[bd]
        out._buf[flat] = _coerce(
            op(a._buf[ai], other._buf[bi]) if not rev
            else op(other._buf[bi], a._buf[ai]), dt)
    return out


def _truediv_dt(a, b):
    """numpy's true-division result dtype: float64 if either side is
    float64, float32 if either side is float32, else float64."""
    bd = b.dtype if isinstance(b, ndarray) else (
        "float64" if isinstance(b, float) else "int64")
    if "float64" in (a.dtype, bd):
        return "float64"
    if "float32" in (a.dtype, bd):
        return "float32"
    return "float64"


def _bitwise(a, b, op):
    """Integer bitwise op with numpy semantics: the left dtype wins, values
    wrap two's-complement to the dtype width. Used by the dequantizers."""
    a = asarray(a)
    if isinstance(b, bool):
        b = int(b)
    if isinstance(b, (int, float)):
        out = _new(a.shape, a.dtype)
        bits = _BY_NAME[a.dtype].itemsize * 8
        mask = (1 << bits) - 1
        for i in range(a._n):
            v = op(int(a._buf[i]), int(b)) & mask
            if _BY_NAME[a.dtype].isint and v >= (1 << (bits - 1)):
                v -= (1 << bits)
            out._buf[i] = v
        return out
    b = asarray(b)
    dt = a.dtype if a.dtype == b.dtype else _promote(a.dtype, b.dtype)
    if a._n == 1:
        out = _new(b.shape, dt)
        bits = _BY_NAME[dt].itemsize * 8
        mask = (1 << bits) - 1
        for i in range(b._n):
            v = op(int(a._buf[0]), int(b._buf[i])) & mask
            if _BY_NAME[dt].isint and v >= (1 << (bits - 1)):
                v -= (1 << bits)
            out._buf[i] = v
        return out
    if b._n == 1:
        out = _new(a.shape, dt)
        bits = _BY_NAME[dt].itemsize * 8
        mask = (1 << bits) - 1
        for i in range(a._n):
            v = op(int(a._buf[i]), int(b._buf[0])) & mask
            if _BY_NAME[dt].isint and v >= (1 << (bits - 1)):
                v -= (1 << bits)
            out._buf[i] = v
        return out
    if a._n != b._n:
        raise ValueError("bitwise operands could not be broadcast together: "
                         "%s %s" % (a.shape, b.shape))
    out = _new(a.shape, dt)
    bits = _BY_NAME[dt].itemsize * 8
    mask = (1 << bits) - 1
    for i in range(a._n):
        v = op(int(a._buf[i]), int(b._buf[i])) & mask
        if _BY_NAME[dt].isint and v >= (1 << (bits - 1)):
            v -= (1 << bits)
        out._buf[i] = v
    return out


def _div(a, b):
    return a / b if b else 0.0


def _map1(a, f):
    dt = _ACC[a.dtype] if a.dtype in _ACC else "float32"
    out = _new(a.shape, dt)
    for i in range(a._n):
        out._buf[i] = _coerce(f(a._buf[i]), dt)
    return out


def _cmp(a, b, op):
    out = _new(a.shape, "bool")
    for i in range(a._n):
        bv = _scalar(b) if not isinstance(b, ndarray) else b._buf[i]
        out._buf[i] = op(a._buf[i], bv)
    return out


def _slice(a, s):
    n = a.shape[0] if a.shape else 0
    idx = range(*s.indices(n))
    out = _new((len(idx),), a.dtype)
    for i, j in enumerate(idx):
        out._buf[i] = a._buf[j]
    return out


def _resolve(a, key):
    """Resolve an index into (out_shape, targets): out_shape is the shape
    of the result, targets the flat source indices in output order.
    Handles int / slice / list / ndarray / bool mask / Ellipsis / None
    (newaxis), numpy-style. A scalar result has out_shape ()."""
    import itertools as _it

    if not isinstance(key, tuple):
        key = (key,)
    keys = list(key)
    if _b.any(k is Ellipsis for k in keys):
        ei = keys.index(Ellipsis)
        n_explicit = sum(1 for k in keys
                         if k is not Ellipsis and k is not None)
        n_fill = a.ndim - n_explicit
        keys = keys[:ei] + [slice(None)] * n_fill + keys[ei + 1:]
    per_axis = []
    out_shape = []
    d = 0
    for k in keys:
        if k is None:
            per_axis.append(None)
            out_shape.append(1)
            continue
        n = a.shape[d]
        if isinstance(k, int):
            kk = k + n if k < 0 else k
            if not 0 <= kk < n:
                raise IndexError("index %d out of range for axis %d of size %d"
                                 % (k, d, n))
            per_axis.append([kk])
        elif isinstance(k, slice):
            idx = list(range(*k.indices(n)))
            per_axis.append(idx)
            out_shape.append(len(idx))
        elif isinstance(k, ndarray):
            if k.dtype == "bool":
                idx = [i for i, v in enumerate(_values(k)) if v]
            elif k.ndim == 0:
                idx = [int(k._buf[0])]
            else:
                idx = [int(v) for v in _values(k)]
            idx = [i + n if i < 0 else i for i in idx]
            per_axis.append(idx)
            out_shape.append(len(idx))
        elif isinstance(k, (list, tuple)):
            idx = [int(v) + n if int(v) < 0 else int(v) for v in k]
            per_axis.append(idx)
            out_shape.append(len(idx))
        else:
            raise TypeError("unsupported index %r" % (k,))
        d += 1
    while d < a.ndim:
        per_axis.append(list(range(a.shape[d])))
        out_shape.append(a.shape[d])
        d += 1
    if not out_shape:
        src = tuple(p[0] for p in per_axis)
        return (), [a._flat_index(src)]
    targets = []
    for combo in _it.product(*[p for p in per_axis if p is not None]):
        targets.append(a._flat_index(combo))
    return tuple(out_shape), targets


def _reduce(a, kind, axis=None, keepdims=False):
    """Whole-array reduction. Axis support is limited to the last axis,
    which is all the engine asks for; anything else raises rather than
    silently returning the wrong number."""
    vals = [a._buf[i] for i in range(a._n)]
    if kind == "sum":
        r = _b.sum(vals)
    elif kind == "mean":
        r = (_b.sum(vals) / len(vals)) if vals else 0.0
    elif kind == "max":
        r = _b.max(vals) if vals else 0.0
    elif kind == "min":
        r = _b.min(vals) if vals else 0.0
    else:
        raise ValueError(kind)
    if axis is not None and axis not in (-1, a.ndim - 1) and keepdims:
        raise NotImplementedError("keepdims reduction only on the last axis")
    return r


# ============================================================================
# construction
# ============================================================================

def array(obj, dtype=None):
    dt = _as_dtype(dtype).name if dtype is not None else "float32"
    if isinstance(obj, ndarray):
        return obj.astype(dt) if obj.dtype != dt else _clone(obj)
    if isinstance(obj, (bytes, bytearray)):
        return frombuffer(obj, uint8)
    shape = _infer(obj)
    vals = _flatten(obj)
    out = _new(shape, dt)
    for i, v in enumerate(vals):
        out._buf[i] = _coerce(v, dt)
    return out


def asarray(obj, dtype=None):
    if isinstance(obj, ndarray) and (dtype is None or obj.dtype == _as_dtype(dtype).name):
        return obj
    return array(obj, dtype)


def ascontiguousarray(obj, dtype=None):
    return asarray(obj, dtype)


def zeros(shape, dtype=None):
    return _new(shape, dtype or "float32")


def ones(shape, dtype=None):
    out = _new(shape, dtype or "float32")
    for i in range(out._n):
        out._buf[i] = 1
    return out


def empty(shape, dtype=None):
    return zeros(shape, dtype)


def full(shape, fill_value, dtype=None):
    out = _new(shape, dtype or "float32")
    for i in range(out._n):
        out._buf[i] = _coerce(fill_value, out.dtype)
    return out


def zeros_like(a, dtype=None):
    return _new(a.shape, dtype or a.dtype)


def empty_like(a, dtype=None):
    return _new(a.shape, dtype or a.dtype)


def full_like(a, fill_value, dtype=None):
    return full(a.shape, fill_value, dtype or a.dtype)


def _infer(obj):
    shape = []
    while isinstance(obj, (list, tuple)):
        shape.append(len(obj))
        if not obj:
            break
        obj = obj[0]
    return tuple(shape)


def _flatten(obj):
    if isinstance(obj, ndarray):
        return [obj._buf[i] for i in range(obj._n)]
    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            out.extend(_flatten(x))
        return out
    return [obj]


def arange(*args, **kw):
    if len(args) == 1:
        start, stop, step = 0, args[0], 1
    elif len(args) == 2:
        start, stop, step = args[0], args[1], 1
    elif len(args) >= 3:
        start, stop, step = args[0], args[1], args[2]
    else:
        raise TypeError("arange needs at least 1 argument")
    dt = _as_dtype(kw.get("dtype")).name if kw.get("dtype") else (
        "int64" if _b.all(isinstance(v, int) for v in (start, stop, step))
        and not isinstance(step, float) else "float32")
    vals = []
    x = start
    if step == 0:
        raise ValueError("arange: step cannot be zero")
    n = int(math.ceil((stop - start) / step)) if step > 0\
        else int(math.ceil((start - stop) / -step))
    for _ in range(_b.max(0, n)):
        vals.append(x)
        x += step
    out = _new((len(vals),), dt)
    for i, v in enumerate(vals):
        out._buf[i] = _coerce(v, dt)
    return out


def linspace(start, stop, num=50, endpoint=True, dtype=None):
    num = int(num)
    if num <= 0:
        return _new((0,), dtype or "float32")
    div = (num - 1) if endpoint else num
    if div == 0:
        vals = [float(start)]
    else:
        step = (stop - start) / div
        vals = [start + step * i for i in range(num)]
        if not endpoint:
            vals = vals[:num]
    out = _new((len(vals),), dtype or "float32")
    for i, v in enumerate(vals):
        out._buf[i] = v
    return out


def eye(n, m=None, k=0, dtype=None):
    m = n if m is None else m
    out = _new((n, m), dtype or "float32")
    for i in range(n):
        j = i + k
        if 0 <= j < m:
            out._buf[i * m + j] = 1
    return out


def frombuffer(buf, dtype=float32, count=-1, offset=0):
    """
    Wrap existing bytes WITHOUT copying, the way np.frombuffer does.

    This is the one that made the dtype table mandatory. Reading quantised
    model bytes as uint8 has to give back the actual byte values; a c_float
    reinterpretation would return nonsense like 1.7e-77 for the byte 235.
    """
    dt = _as_dtype(dtype)
    if dt.ctype is None:
        raise TypeError("%s is not supported by frombuffer" % dt.name)
    raw = bytes(buf) if not isinstance(buf, (bytes, bytearray)) else buf
    itemsize = dt.itemsize
    if offset % itemsize:
        raise ValueError("offset must be a multiple of the itemsize")
    usable = len(raw) - offset
    n = usable // itemsize if count < 0 else count
    n = _b.max(0, _b.min(n, usable // itemsize))
    # frombuffer is a VIEW. Share the caller's bytes by reinterpreting them.
    arr = (dt.ctype * n).from_buffer_copy(raw, offset) if n else (dt.ctype * 0)()
    if n:
        # copy into a buffer we own so the caller can free theirs safely
        own = (dt.ctype * n)(*arr)
        return ndarray(own, (n,), dt.name)
    return ndarray(arr, (0,), dt.name)


def copy(a):
    return a.copy()


def memmap(filename, dtype=float32, mode="r", shape=None, offset=0):
    with open(filename, "rb") as fh:
        fh.seek(offset)
        return frombuffer(fh.read(), dtype)


def fromiter(iterable, dtype, count=-1):
    """Build an array from an iterable, numpy-style."""
    vals = []
    for i, v in enumerate(iterable):
        if 0 <= count <= i:
            break
        vals.append(v)
    return array(vals, dtype=dtype)


class _CtypesLib:
    """np.ctypeslib, minimal: as_array only, which is all the engine uses."""

    @staticmethod
    def as_array(obj, shape=None):
        if isinstance(obj, ndarray):
            return obj if shape is None else reshape(obj, shape)
        # a ctypes array carries its own length and element type
        if hasattr(obj, "_length_"):
            _CT = {ctypes.c_float: "float32", ctypes.c_double: "float64",
                   ctypes.c_int8: "int8", ctypes.c_int16: "int16",
                   ctypes.c_int32: "int32", ctypes.c_int64: "int64",
                   ctypes.c_uint8: "uint8", ctypes.c_uint16: "uint16",
                   ctypes.c_uint32: "uint32", ctypes.c_uint64: "uint64",
                   ctypes.c_bool: "bool"}
            dt = _CT.get(obj._type_, "float32")
            n = obj._length_
            if shape is not None:
                shape = (shape,) if isinstance(shape, int) else tuple(shape)
                if _prod(shape) != n:
                    raise ValueError("as_array shape does not match buffer")
                out = _new(shape, dt)
            else:
                out = _new((n,), dt)
            for i in range(n):
                out._buf[i] = obj[i]
            return out
        if isinstance(obj, int):
            raise ValueError("ctypeslib.as_array needs a length for a raw address")
        return asarray(obj)


ctypeslib = _CtypesLib()


_NPY_DESCR = {
    "<f4": "float32", "<f8": "float64", "|f4": "float32", "|f8": "float64",
    "<i1": "int8", "<i2": "int16", "<i4": "int32", "<i8": "int64",
    "<u1": "uint8", "<u2": "uint16", "<u4": "uint32", "<u8": "uint64",
    "|i1": "int8", "|u1": "uint8", "|b1": "bool",
}


def load(file, mmap_mode=None, allow_pickle=False, fix_imports=True,
         encoding="ASCII", **kw):
    """Read a .npy file (numpy format 1.0/2.0/3.0 headers, C-order data).
    Pure stdlib: parses the header dict, maps the descr to a dtype, and
    builds the array from the raw bytes. Fortran-order arrays are
    transposed back to C order (2-D only); float16 is decoded to float32."""
    import ast as _ast
    import struct as _struct

    if hasattr(file, "read"):
        raw = file.read()
    else:
        with open(file, "rb") as f:
            raw = f.read()
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("not a .npy file")
    major = raw[6]
    if major == 1:
        hlen = _struct.unpack("<H", raw[8:10])[0]
        hstart = 10
    else:
        hlen = _struct.unpack("<I", raw[8:12])[0]
        hstart = 12
    header = _ast.literal_eval(raw[hstart:hstart + hlen].decode("latin1").strip())
    descr = header["descr"]
    fortran = header["fortran_order"]
    shape = tuple(header["shape"])
    data = raw[hstart + hlen:]
    if descr in ("<f2", "|f2"):
        # half floats: struct 'e' decodes them; store as float32
        n = len(data) // 2
        vals = list(_struct.unpack("<%de" % n, data[:n * 2]))
        out = _new(shape if shape else (n,), "float32")
        for i, v in enumerate(vals):
            out._buf[i] = v
        arr = out
    else:
        dt = _NPY_DESCR.get(descr)
        if dt is None:
            raise TypeError("unsupported .npy descr %r" % descr)
        flat = frombuffer(data, dtype=dt)
        n = 1
        for s in shape:
            n *= s
        arr = _new(shape if shape else (n,), dt)
        for i in range(_b.min(n, flat._n)):
            arr._buf[i] = flat._buf[i]
    if fortran and len(shape) == 2:
        arr = transpose(arr)
    elif fortran and len(shape) > 2:
        raise NotImplementedError("load: Fortran-order >2-D not implemented")
    return arr


# ============================================================================
# joining
# ============================================================================

def concatenate(arrays, axis=0, dtype=None):
    """Join arrays along an existing axis, numpy-style. The old version
    ignored `axis` and always returned a flat 1-D array, which silently
    broke every caller that concatenated on axis -1 or 1 (RoPE, KV cache)."""
    arrays = [asarray(a) for a in arrays]
    arrays = [a for a in arrays if a._n or a.shape != (0,)]
    if not arrays:
        return _new((0,), dtype or "float32")
    dt = _as_dtype(dtype).name if dtype is not None else arrays[0].dtype
    for a in arrays[1:]:
        dt = _promote(dt, a.dtype)
    nd = arrays[0].ndim
    if nd == 0:
        out = _new((len(arrays),), dt)
        for i, a in enumerate(arrays):
            out._buf[i] = _coerce(a._buf[0], dt)
        return out
    ax = axis + nd if axis < 0 else axis
    if not 0 <= ax < nd:
        raise ValueError("concatenate axis %d out of range for %d-D"
                         % (axis, nd))
    for a in arrays:
        if a.ndim != nd:
            raise ValueError("concatenate: all arrays must have the same rank")
        for d in range(nd):
            if d != ax and a.shape[d] != arrays[0].shape[d]:
                raise ValueError("concatenate: shapes differ on axis %d" % d)
    out_shape = list(arrays[0].shape)
    out_shape[ax] = _b.sum(a.shape[ax] for a in arrays)
    out = _new(tuple(out_shape), dt)
    o_str = _strides(tuple(out_shape))
    off = 0
    for a in arrays:
        a_str = _strides(a.shape)
        for flat in range(a._n):
            rem = flat
            coords = [0] * nd
            for i in range(nd - 1, -1, -1):
                coords[i] = rem % a.shape[i]
                rem //= a.shape[i]
            coords[ax] += off
            o_flat = 0
            for i in range(nd):
                o_flat += coords[i] * o_str[i]
            out._buf[o_flat] = _coerce(a._buf[flat], dt)
        off += a.shape[ax]
    return out


def stack(arrays, axis=0, dtype=None):
    """Stack arrays along a NEW axis, numpy-style."""
    arrays = [asarray(a) for a in arrays]
    if not arrays:
        raise ValueError("stack of an empty list")
    dt = _as_dtype(dtype).name if dtype is not None else arrays[0].dtype
    for a in arrays:
        if a.shape != arrays[0].shape:
            raise ValueError("stack: all arrays must have the same shape")
        dt = _promote(dt, a.dtype)
    inner = arrays[0].shape
    nd = len(inner) + 1
    ax = axis + nd if axis < 0 else axis
    if not 0 <= ax <= len(inner):
        raise ValueError("stack axis %d out of range" % axis)
    out_shape = inner[:ax] + (len(arrays),) + inner[ax:]
    out = _new(tuple(out_shape), dt)
    o_str = _strides(tuple(out_shape))
    for i, a in enumerate(arrays):
        a_str = _strides(a.shape)
        for flat in range(a._n):
            rem = flat
            coords = [0] * len(inner)
            for j in range(len(inner) - 1, -1, -1):
                coords[j] = rem % inner[j]
                rem //= inner[j]
            out_coords = coords[:ax] + [i] + coords[ax:]
            o_flat = 0
            for j in range(len(out_shape)):
                o_flat += out_coords[j] * o_str[j]
            out._buf[o_flat] = _coerce(a._buf[flat], dt)
    return out


def vstack(arrays):
    """Stack vertically: 1-D arrays become rows, then concatenate on axis 0."""
    arrays = [asarray(a) for a in arrays]
    arrays = [a.reshape(1, -1) if a.ndim == 1 else a for a in arrays]
    return concatenate(arrays, axis=0)


def hstack(arrays):
    """Stack horizontally: 1-D arrays concatenate flat, 2-D on axis 1."""
    arrays = [asarray(a) for a in arrays]
    if all(a.ndim <= 1 for a in arrays):
        return concatenate(arrays, axis=0)
    arrays = [a.reshape(1, -1) if a.ndim == 1 else a for a in arrays]
    return concatenate(arrays, axis=1)


# ============================================================================
# elementwise
# ============================================================================

def _ufunc1(name, f, npname, float_out=False):
    """Elementwise ufunc. float_out=True for the trig/log/exp family,
    which numpy always returns as float, even on integer input."""
    def g(a, dtype=None):
        a = asarray(a)
        if dtype is not None:
            dt = _as_dtype(dtype).name
        elif float_out:
            dt = "float64" if a.dtype == "float64" else "float32"
        else:
            dt = _ACC.get(a.dtype, "float32")
        out = _new(a.shape, dt)
        for i in range(a._n):
            out._buf[i] = _coerce(f(a._buf[i]), dt)
        return out
    g.__name__ = name
    g.__doc__ = "elementwise %s" % npname
    return g


abs_ = _ufunc1("abs", lambda a: a if a >= 0 else -a, "|a|")
abs = abs_                       # numpy spells it without the underscore
fabs = abs_
absolute = abs_
sign = _ufunc1("sign", lambda a: (a > 0) - (a < 0), "sign")
cos = _ufunc1("cos", math.cos, "cos", float_out=True)
sin = _ufunc1("sin", math.sin, "sin", float_out=True)
tan = _ufunc1("tan", math.tan, "tan", float_out=True)
log1p = _ufunc1("log1p", lambda a: math.log1p(a) if a > -1 else float("-inf"), "log(1+x)", float_out=True)
log2 = _ufunc1("log2", lambda a: math.log2(a) if a > 0 else float("-inf"), "log2", float_out=True)
log10 = _ufunc1("log10", lambda a: math.log10(a) if a > 0 else float("-inf"), "log10", float_out=True)
log = _ufunc1("log", lambda a: math.log(a) if a > 0 else float("-inf"), "log", float_out=True)
exp2 = _ufunc1("exp2", lambda a: 2.0 ** a, "2**x", float_out=True)
square = _ufunc1("square", lambda a: a * a, "x**2")
rint = _ufunc1("rint", lambda a: float(_b.round(a)), "round", float_out=True)
floor = _ufunc1("floor", lambda a: float(math.floor(a)), "floor", float_out=True)
ceil = _ufunc1("ceil", lambda a: float(math.ceil(a)), "ceil", float_out=True)
arccos = _ufunc1("arccos", lambda a: math.acos(_b.max(-1.0, _b.min(1.0, a))), "arccos", float_out=True)


def clip(a, lo=None, hi=None, dtype=None):
    a = asarray(a)

    def f(v):
        if lo is not None and v < lo:
            return lo
        if hi is not None and v > hi:
            return hi
        return v

    return _map1(a, f)


def round(a, decimals=0, dtype=None):        # noqa: A001 - numpy's name
    a = asarray(a)
    if not decimals:
        return _map1(a, lambda v: float(_b.round(v)))
    m = 10.0 ** decimals
    return _map1(a, lambda v: _b.round(v * m) / m)


around = round


def outer(a, b):
    """Outer product of two 1-D arrays: out[i, j] = a[i] * b[j]."""
    a = ravel(asarray(a))
    b = ravel(asarray(b))
    dt = _promote(a.dtype, b.dtype)
    out = _new((a._n, b._n), dt)
    for i in range(a._n):
        for j in range(b._n):
            out._buf[i * b._n + j] = a._buf[i] * b._buf[j]
    return out


def where(cond, x=None, y=None):
    cond = asarray(cond)
    if x is None and y is None:
        # numpy returns a TUPLE of index arrays, one per axis. The old
        # version returned a flat array with -1 for False entries, so
        # `np.where(mask)[0]` gave a scalar instead of the index array.
        flats = [i for i in range(cond._n) if cond._buf[i]]
        per_axis = []
        for d in range(cond.ndim):
            arr = _new((len(flats),), "int64")
            stride = 1
            for s in cond.shape[d + 1:]:
                stride *= s
            for i, f in enumerate(flats):
                arr._buf[i] = (f // stride) % cond.shape[d]
            per_axis.append(arr)
        return tuple(per_axis)
    # x and y may each be an array, a list, or a scalar. Normalise the
    # non-scalars; a list has to become an array or the assignment below
    # gets handed the whole list instead of one element.
    def prep(v):
        if isinstance(v, (ndarray, list, tuple)):
            return asarray(v)
        return None
    xa = prep(x)
    ya = prep(y)
    out = _new(cond.shape, "float32")
    for i in range(cond._n):
        if cond._buf[i]:
            out._buf[i] = xa._buf[i] if xa is not None else x
        else:
            out._buf[i] = ya._buf[i] if ya is not None else y
    return out


def _is_seq(v):
    """True when v is array-like. Lists and tuples count: the callers
    below used to test only for ndarray and so handed an entire list to a
    lambda that expected one scalar."""
    return isinstance(v, (ndarray, list, tuple))


def divide(a, b, dtype=None):
    if _is_seq(b):
        return asarray(a) / asarray(b)
    return _map1(asarray(a), lambda v: _div(v, b))


true_divide = divide


def mod(a, b, dtype=None):
    if _is_seq(b):
        return asarray(a) % asarray(b)
    return _map1(asarray(a), lambda v: v - b * int(v / b) if b else 0)


remainder = mod


def power(a, b, dtype=None):
    if _is_seq(b):
        return asarray(a) ** asarray(b)
    return _map1(asarray(a), lambda v: v ** b)


def maximum(a, b):
    if _is_seq(b):
        return _apply(asarray(a), asarray(b), lambda x, y: x if x > y else y)
    return _map1(asarray(a), lambda v: v if v > b else b)


def minimum(a, b):
    if _is_seq(b):
        return _apply(asarray(a), asarray(b), lambda x, y: x if x < y else y)
    return _map1(asarray(a), lambda v: v if v < b else b)


def cumsum(a, dtype=None):
    return asarray(a).cumsum()


cumsum_ = cumsum


def cumprod(a, dtype=None):
    a = asarray(a)
    out = _new(a.shape, "float32")
    run = 1.0
    for i in range(a._n):
        run *= a._buf[i]
        out._buf[i] = run
    return out


def repeat(a, repeats, axis=None):
    a = asarray(a)
    if _b.isinstance(repeats, int):
        out = _new((a._n * repeats,), a.dtype)
        k = 0
        for i in range(a._n):
            for _ in range(repeats):
                out._buf[k] = a._buf[i]
                k += 1
        return out
    out = _new((len(repeats),), a.dtype)
    for i, r in enumerate(repeats):
        out._buf[i] = a._buf[int(r)]
    return out


# ============================================================================
# reductions and inspection
# ============================================================================

def sum_(a, axis=None, keepdims=False, dtype=None):
    return _reduce(asarray(a), "sum", axis, keepdims)


sum = sum_


def mean(a, axis=None, keepdims=False, dtype=None):
    return _reduce(asarray(a), "mean", axis, keepdims)


def max_(a, axis=None, keepdims=False, dtype=None):
    return _reduce(asarray(a), "max", axis, keepdims)


max = max_


def min_(a, axis=None, keepdims=False, dtype=None):
    return _reduce(asarray(a), "min", axis, keepdims)


min = min_


def all(a, axis=None, keepdims=False, dtype=None):
    return asarray(a).all()


all_ = all


def any(a, axis=None, keepdims=False, dtype=None):
    return asarray(a).any()


any_ = any


def prod(a, axis=None, keepdims=False, dtype=None):
    return asarray(a).prod()


def median(a, axis=None, keepdims=False):
    vals = sorted(asarray(a)._buf[:] if False
                  else [asarray(a)._buf[i] for i in range(asarray(a)._n)])
    if not vals:
        return float("nan")
    m = len(vals) // 2
    return vals[m] if len(vals) % 2 else (vals[m - 1] + vals[m]) / 2.0


def percentile(a, q, axis=None, keepdims=False, method="linear"):
    """Percentile with linear interpolation, numpy-style. `q` in [0, 100]."""
    vals = sorted([asarray(a)._buf[i] for i in range(asarray(a)._n)])
    if not vals:
        raise ValueError("percentile of an empty array")

    def _pct(qq):
        qq = float(qq)
        if qq < 0 or qq > 100:
            raise ValueError("percentile must be in [0, 100]")
        pos = (len(vals) - 1) * qq / 100.0
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return vals[lo]
        return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)

    if isinstance(q, (list, tuple)):
        return array([_pct(v) for v in q])
    return _pct(q)


def std(a, axis=None, ddof=0):
    vals = [asarray(a)._buf[i] for i in range(asarray(a)._n)]
    m = _b.sum(vals) / len(vals)
    return math.sqrt(_b.sum((v - m) ** 2 for v in vals) / _b.max(1, len(vals) - ddof))


def var(a, axis=None, ddof=0):
    vals = [asarray(a)._buf[i] for i in range(asarray(a)._n)]
    m = _b.sum(vals) / len(vals)
    return _b.sum((v - m) ** 2 for v in vals) / _b.max(1, len(vals) - ddof)


def isfinite(a):
    a = asarray(a)
    out = _new(a.shape, "bool")
    for i in range(a._n):
        out._buf[i] = math.isfinite(float(a._buf[i]))
    return out


def isinf(a):
    a = asarray(a)
    out = _new(a.shape, "bool")
    for i in range(a._n):
        out._buf[i] = math.isinf(float(a._buf[i]))
    return out


def nan_to_num(a, nan=0.0, posinf=None, neginf=None):
    a = asarray(a)
    hi = posinf if posinf is not None else 1.7976931348623157e308
    lo = neginf if neginf is not None else -1.7976931348623157e308
    return _map1(a, lambda v: nan if math.isnan(v) else
                 (hi if math.isinf(v) and v > 0 else
                  (lo if math.isinf(v) else v)))


def seterr(all=None, divide=None, over=None, under=None, invalid=None):
    """numpy's error-state setter. nightmare_np has no floating-point
    error state to set -- everything already behaves as if errors were
    ignored. Accept the call and report that (unchanging) state."""
    return {"divide": "ignore", "over": "ignore",
            "under": "ignore", "invalid": "ignore"}


def isscalar(x):
    return _b.isinstance(x, (int, float, complex, _b.bool if hasattr(_b, "bool") else bool))


def ndim(x):
    return asarray(x).ndim if not _b.isinstance(x, (int, float)) else 0


def size(x):
    return asarray(x)._n if not _b.isinstance(x, (int, float)) else 1


def shape(x):
    return asarray(x).shape if not _b.isinstance(x, (int, float)) else ()


# ============================================================================
# reshaping
# ============================================================================

def reshape(a, *shape):
    return asarray(a).reshape(*shape)


def ravel(a):
    return asarray(a).ravel()


def transpose(a, axes=None):
    a = asarray(a)
    if a.ndim < 2:
        return _clone(a)
    if a.ndim == 2:
        s0, s1 = a.shape
        out = _new((s1, s0), a.dtype)
        for i in range(s0):
            for j in range(s1):
                out._buf[j * s0 + i] = a._buf[i * s1 + j]
        return out
    raise NotImplementedError("transpose for >2-D")


def tile(a, reps):
    a = asarray(a)
    if _b.isinstance(reps, int):
        reps = (reps,)
    if a.ndim == 1:
        # (n,) tiled r times -> (n*r,)
        r = reps[0]
        out = _new((a._n * r,), a.dtype)
        for k in range(out._n):
            out._buf[k] = a._buf[k % a._n]
        return out
    if a.ndim == 2:
        d0, d1 = a.shape
        R, S = reps[0], reps[1]
        out = _new((d0 * R, d1 * S), a.dtype)
        for r in range(R):
            for c in range(S):
                for i in range(d0):
                    for j in range(d1):
                        out._buf[(r * d0 + i) * (S * d1) + (c * d1 + j)] =\
                            a._buf[i * d1 + j]
        return out
    raise NotImplementedError("tile for >2-D")


def moveaxis(a, src, dst):
    a = asarray(a)
    if a.ndim != 3:
        raise NotImplementedError("moveaxis only for 3-D here")
    order = [0, 1, 2]
    order.remove(src)
    order.insert(dst, src)
    if order == [0, 1, 2]:
        return _clone(a)
    out = _new(tuple(a.shape[i] for i in order), a.dtype)
    d = a.shape
    for i in range(d[order[0]]):
        for j in range(d[order[1]]):
            for k in range(d[order[2]]):
                src_idx = [i, j, k]
                dst_idx = [0, 0, 0]
                dst_idx[order.index(0)] = i
                dst_idx[order.index(1)] = j
                dst_idx[order.index(2)] = k
                out._buf[(dst_idx[0] * d[order[1]] + dst_idx[1]) * d[order[2]]
                         + dst_idx[2]] = a._buf[(i * d[1] + j) * d[2] + k]
    return out


def expand_dims(a, axis=0):
    a = asarray(a)
    s = list(a.shape)
    s.insert(axis if axis >= 0 else len(s) + axis + 1, 1)
    return ndarray(a._buf, tuple(s), a.dtype)


def squeeze(a):
    return asarray(a).squeeze()


# ============================================================================
# bits
# ============================================================================

def unpackbits(a, axis=None, bitorder="big"):
    a = asarray(a)
    out = _new((a._n * 8,), "uint8")
    k = 0
    for i in range(a._n):
        v = int(a._buf[i]) & 0xFF
        for b in range(8):
            out._buf[k] = (v >> (7 - b)) & 1
            k += 1
    return out


def packbits(a, axis=None, bitorder="big"):
    a = asarray(a)
    n = a._n // 8
    out = _new((n,), "uint8")
    for i in range(n):
        v = 0
        for b in range(8):
            v = (v << 1) | (1 if a._buf[i * 8 + b] else 0)
        out._buf[i] = v
    return out


def bitwise_and(a, b):
    if _is_seq(b):
        return _apply(asarray(a), asarray(b), lambda x, y: int(x) & int(y))
    return _map1(asarray(a), lambda v: int(v) & int(b))


def bitwise_or(a, b):
    if _is_seq(b):
        return _apply(asarray(a), asarray(b), lambda x, y: int(x) | int(y))
    return _map1(asarray(a), lambda v: int(v) | int(b))


def bitwise_xor(a, b):
    if _is_seq(b):
        return _apply(asarray(a), asarray(b), lambda x, y: int(x) ^ int(y))
    return _map1(asarray(a), lambda v: int(v) ^ int(b))


def bitwise_not(a):
    return _map1(asarray(a), lambda v: ~int(v))


def left_shift(a, b):
    return _map1(asarray(a), lambda v: int(v) << int(b))


def right_shift(a, b):
    return _map1(asarray(a), lambda v: int(v) >> int(b))


# ============================================================================
# linear algebra -- what the engine actually calls
# ============================================================================

def dot(a, b):
    a = asarray(a)
    b = asarray(b)
    if a.ndim == 1 and b.ndim == 1:
        return _b.sum(a._buf[i] * b._buf[i] for i in range(a._n))
    if a.ndim == 2 and b.ndim == 1:
        s0, s1 = a.shape
        out = _new((s0,), "float32")
        for i in range(s0):
            out._buf[i] = _b.sum(a._buf[i * s1 + j] * b._buf[j] for j in range(s1))
        return out
    if a.ndim == 1 and b.ndim == 2:
        s0, s1 = b.shape
        out = _new((s1,), "float32")
        for j in range(s1):
            out._buf[j] = _b.sum(a._buf[i] * b._buf[i * s1 + j] for i in range(s0))
        return out
    raise NotImplementedError("dot for these shapes")


def matmul(a, b):
    a = asarray(a)
    b = asarray(b)
    if a.ndim == 1 and b.ndim == 1:
        return dot(a, b)
    if a.ndim == 2 and b.ndim == 1:
        return dot(a, b)
    if a.ndim == 1 and b.ndim == 2:
        return dot(a, b)
    if a.ndim == 2 and b.ndim == 2:
        s0, s1 = a.shape
        t0, t1 = b.shape
        if s1 != t0:
            raise ValueError("matmul shape mismatch %s %s" % (a.shape, b.shape))
        # Fast path: C backend (nightmare_np_c.dll) when it is loaded and
        # both operands are float32. Without it, a 1B-model matmul is a
        # pure-Python triple loop and a single forward pass takes hours.
        # The C matmul takes _ext ndarrays directly and returns one.
        if a.dtype == "float32" and b.dtype == "float32":
            try:
                import nightmare_np._cbackend as _cb
                if _cb.available:
                    return _cb.matmul(a, b)
            except Exception:
                pass
        out = _new((s0, t1), "float32")
        bv = [b._buf[i] for i in range(b._n)]
        for i in range(s0):
            row = i * s1
            for j in range(t1):
                out._buf[i * t1 + j] = _b.sum(
                    a._buf[row + k] * bv[k * t1 + j] for k in range(s1))
        return out
    raise NotImplementedError("matmul for these shapes")


# ============================================================================
# constants and pseudo-modules
# ============================================================================

newaxis = None
pi = math.pi
e = math.e
inf = math.inf
Inf = math.inf
NAN = float("nan")
nan = float("nan")
NaN = float("nan")
infty = math.inf
PINF = math.inf
NINF = -math.inf


class _Linalg:
    """Only the handful of entry points the engine names."""

    @staticmethod
    def norm(x, axis=None):
        vals = [asarray(x)._buf[i] for i in range(asarray(x)._n)]
        return math.sqrt(_b.sum(v * v for v in vals))

    @staticmethod
    def svd(a, full_matrices=True, compute_uv=True, hermitian=False):
        """Truncated .ntmre compression via nightmare_np_svd (C backend, Jacobi fallback).
        Returns (U, S, Vt) as ndarrays in numpy layout: U (m, k), S (k,),
        Vt (k, n), with k = min(m, n). Only full_matrices=False is
        implemented -- that is the only form the engine (the converter)
        calls."""
        if not compute_uv:
            raise NotImplementedError("svd(compute_uv=False) is not implemented")
        if full_matrices:
            raise NotImplementedError(
                "svd(full_matrices=True) is not implemented; "
                "the engine only uses full_matrices=False")
        a = asarray(a)
        if a.ndim != 2:
            raise ValueError("svd expects a 2-D array")
        from .nightmare_np_svd import np_svd
        m, n = a.shape
        rows = [a[i].tolist() for i in range(m)]
        U_list, S_list, Vt_list = np_svd(rows)
        k = _b.min(m, n)
        U_list = U_list[:k]
        S_list = S_list[:k]
        Vt_list = Vt_list[:k]
        # np_svd returns U as k rows of length m (i.e. U transposed);
        # transpose it back to numpy's (m, k) layout.
        U = array([[U_list[t][i] for t in range(k)] for i in range(m)])
        S = array(S_list)
        Vt = array(Vt_list)
        return U, S, Vt

    @staticmethod
    def det(a):
        raise NotImplementedError("det is not implemented")

    @staticmethod
    def inv(a):
        raise NotImplementedError("inv is not implemented")

    @staticmethod
    def solve(a, b):
        raise NotImplementedError("solve is not implemented")


linalg = _Linalg()


class _Random:
    """Present so `np.random.*` names resolve. The engine uses it only in
    benchmark helpers; results are reproducible via seed()."""

    def __init__(self):
        self._r = _random_mod.Random(0)

    def seed(self, s):
        self._r.seed(s)

    def rand(self, *shape):
        if not shape:
            return self._r.random()
        out = _new(shape, "float32")
        for i in range(out._n):
            out._buf[i] = self._r.random()
        return out

    def randn(self, *shape):
        out = _new(shape, "float32")
        for i in range(out._n):
            out._buf[i] = self._r.gauss(0.0, 1.0)
        return out

    def uniform_sym(self, *shape, scale=0.02):
        """Fast symmetric uniform init via float bit patterns.

        Fills a float32 array with zero-mean values whose magnitudes
        sit in [scale/2, 2*scale) — the right order of magnitude for
        weight init, std ~scale. Uses os.urandom + bulk byte ops
        (C speed, ~25M values/sec), ~15x faster than the per-element
        randn loop above. That loop takes ~9 minutes for a 1B model;
        this takes ~40 seconds.

        Method: a float32's value is set by its exponent + mantissa.
        os.urandom supplies random mantissa bits; the exponent is
        forced (via byte-slice assignment + translate, both C speed)
        to the power of two bracketing `scale`, and the sign bit is
        randomized. No per-element Python math anywhere.

        Distribution note: this is uniform in magnitude with random
        signs — NOT a normal distribution (no values very near zero,
        hard cutoff at 2*scale). That is fine for weight init: it
        breaks symmetry, is zero-mean, and has the right scale
        (Xavier/Glorot-style uniform init is standard). Training
        moves weights into the near-zero range on its own.
        """
        import os as _os
        n = _prod(shape) if shape else 1
        if n <= 0:
            return _new(shape, "float32")
        # Exponent bracketing `scale`: magnitudes in [2^e, 2^(e+1))
        # where 2^e <= scale < 2^(e+1). Stored exponent = 127 + e.
        e = math.floor(math.log2(scale)) if scale > 0 else -6
        exp = 127 + e
        b3_base = (exp >> 1) & 0x7F
        b3_pos = b3_base
        b3_neg = b3_base | 0x80
        # byte2 top bit = exponent low bit
        if exp & 1:
            b2_trans = bytes((b | 0x80) for b in range(256))
        else:
            b2_trans = bytes((b & 0x7F) for b in range(256))
        sign_trans = bytes((b3_pos if b < 128 else b3_neg)
                           for b in range(256))
        ba = bytearray(_os.urandom(n * 4))
        ba[3::4] = bytes(_os.urandom(n)).translate(sign_trans)
        ba[2::4] = bytes(ba[2::4]).translate(b2_trans)
        out = _new(shape, "float32")
        ctypes.memmove(ctypes.addressof(out._buf), bytes(ba), n * 4)
        return out

    def randint(self, lo, hi=None, size=None):
        hi = lo if hi is None else hi
        if size is None:
            return self._r.randrange(lo, hi)
        out = _new(size if _b.isinstance(size, tuple) else (size,), "int64")
        for i in range(out._n):
            out._buf[i] = self._r.randrange(lo, hi)
        return out

    def random(self, size=None):
        return self.rand(size) if size is not None else self._r.random()

    def uniform(self, lo=0.0, hi=1.0, size=None):
        if size is None:
            return self._r.uniform(lo, hi)
        out = _new(size, "float32")
        for i in range(out._n):
            out._buf[i] = self._r.uniform(lo, hi)
        return out

    def normal(self, loc=0.0, scale=1.0, size=None):
        if size is None:
            return self._r.gauss(loc, scale)
        out = _new(size, "float32")
        for i in range(out._n):
            out._buf[i] = self._r.gauss(loc, scale)
        return out

    def choice(self, a, size=None, replace=True, p=None):
        """Pick from a 1-D array, optionally with probabilities `p`."""
        a = asarray(a)
        vals = [a._buf[i] for i in range(a._n)]
        if not vals:
            raise ValueError("choice from an empty array")
        if p is None:
            pick = lambda: vals[self._r.randrange(len(vals))]
        else:
            pv = [asarray(p)._buf[i] for i in range(asarray(p)._n)]
            tot = _b.sum(pv)
            if tot <= 0:
                raise ValueError("choice probabilities sum to zero")
            cum = []
            s = 0.0
            for v in pv:
                s += v / tot
                cum.append(s)

            def pick():
                r = self._r.random()
                for i, c in enumerate(cum):
                    if r < c:
                        return vals[i]
                return vals[-1]

        if size is None:
            return pick()
        shape = size if isinstance(size, tuple) else (size,)
        out = _new(shape, a.dtype)
        for i in range(out._n):
            out._buf[i] = pick()
        return out

    def default_rng(self, seed=None):
        """np.random.default_rng: a fresh generator, optionally seeded."""
        r = _Random()
        if seed is not None:
            r.seed(seed)
        return r


random = _Random()


# ============================================================================
# Self-check
# ============================================================================
# Every name in __all__ must actually exist here. This module deliberately
# rebinds several builtins at module level (max, min, sum, abs, round) to
# give them numpy's meaning, which means a name in __all__ can silently
# point at nothing -- it happened with "abs", which existed as abs_ and
# blew up at import time with a confusing message from __init__.py.
_missing = [n for n in __all__ if n not in globals()]
if _missing:
    raise ImportError("nightmare_np._ext: __all__ names that do not "
                      "exist: %s" % ", ".join(sorted(_missing)))
