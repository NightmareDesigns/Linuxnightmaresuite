"""
nightmare_np — a hand-written numpy replacement for the Nightmare Suite.

PURE STDLIB. No numpy. No external code. The numeric work is done by the
hand-written C DLL (nightmare_np_c.dll) via ctypes; if the DLL is not built
yet, everything falls back to pure-Python so your code still runs.

Use it exactly like numpy:

    import nightmare_np as np
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.matmul(a, a)
    y = np.einsum('hd,shd->hs', q, k)

Because this is YOUR library, no AI agent can "pip install" or break it the
way it can the real numpy.
"""
import builtins as _b
import ctypes
import itertools as _it
import math
import os
import random as _random

__version__ = "1.0.0"

# ============================================================================
# C backend loading (stdlib ctypes only)
# ============================================================================
_DLL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nightmare_np_c.dll")


class _C:
    pass


_c = _C()
_c.available = False
_c.reason = "C backend not built (run build_c.bat)"


def _load_dll():
    if not os.path.exists(_DLL_PATH):
        return
    try:
        lib = ctypes.CDLL(_DLL_PATH)
        _c.lib = lib

        def bind(name, restype, argtypes):
            fn = getattr(lib, name)
            fn.restype = restype
            fn.argtypes = argtypes
            setattr(_c, name, fn)

        P = ctypes.c_void_p
        I = ctypes.c_int
        F = ctypes.c_float
        bind("neg_f32", None, [P, P, I])
        bind("abs_f32", None, [P, P, I])
        bind("sqrt_f32", None, [P, P, I])
        bind("exp_f32", None, [P, P, I])
        bind("tanh_f32", None, [P, P, I])
        bind("silu_f32", None, [P, P, I])
        bind("gelu_f32", None, [P, P, I])
        bind("add_f32", None, [P, P, P, I])
        bind("sub_f32", None, [P, P, P, I])
        bind("mul_f32", None, [P, P, P, I])
        bind("div_f32", None, [P, P, P, I])
        bind("max_f32", None, [P, P, P, I])
        bind("min_f32", None, [P, P, P, I])
        bind("sum_last_f32", None, [P, P, I, I])
        bind("mean_last_f32", None, [P, P, I, I])
        bind("max_last_f32", None, [P, P, I, I])
        bind("min_last_f32", None, [P, P, I, I])
        bind("reduce_all_f32", F, [P, I, I])
        bind("rmsnorm_f32", None, [P, P, P, I, I, F])
        bind("softmax_rows_f32", None, [P, P, I, I])
        bind("matmul_f32", None, [P, P, P, I, I, I])
        bind("einsum_f32", None, [P, ctypes.POINTER(P), ctypes.POINTER(I),
                                  ctypes.POINTER(I), ctypes.c_char_p, I])
        _c.available = True
        _c.reason = "C backend active (nightmare_np_c.dll)"
    except Exception as e:  # pragma: no cover
        _c.available = False
        _c.reason = f"C backend failed to load: {e}"


_load_dll()


# ============================================================================
# Small helpers
# ============================================================================
def _prod(shape):
    p = 1
    for s in shape:
        p *= s
    return p


def _flat_index(idx, shape):
    n = 0
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        n += idx[i] * stride
        stride *= shape[i]
    return n


def _unravel(flat, shape):
    idx = [0] * len(shape)
    for i in range(len(shape) - 1, -1, -1):
        idx[i] = flat % shape[i]
        flat //= shape[i]
    return tuple(idx)


def _broadcast_shapes(s1, s2):
    r = []
    for a, b in _it.zip_longest(reversed(s1), reversed(s2), fillvalue=1):
        if a == b:
            r.append(a)
        elif a == 1:
            r.append(b)
        elif b == 1:
            r.append(a)
        else:
            raise ValueError(f"cannot broadcast {s1} and {s2}")
    return tuple(reversed(r))


def _map_index(op_shape, out_shape, out_idx):
    off = len(out_shape) - len(op_shape)
    return tuple(0 if d == 1 else out_idx[i + off] for i, d in enumerate(op_shape))


def _broadcast_to(a, shape):
    if a.shape == shape:
        return a
    n = _prod(shape)
    out = (ctypes.c_float * n)()
    for flat in range(n):
        idx = _unravel(flat, shape)
        src = _map_index(a.shape, shape, idx)
        out[flat] = a._buf[_flat_index(src, a.shape)]
    return ndarray(out, shape, a.dtype)


# ============================================================================
# ndarray class
# ============================================================================
class ndarray:
    __slots__ = ("_buf", "shape", "dtype", "_n")

    def __init__(self, buf, shape, dtype="float32"):
        self._buf = buf          # ctypes.c_float * n
        self.shape = tuple(shape)
        self.dtype = dtype
        self._n = len(buf)

    # ---- basic properties ----
    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return self._n

    @property
    def T(self):
        if self.ndim <= 2:
            if self.ndim == 1:
                return self
            s0, s1 = self.shape
            out = (ctypes.c_float * self._n)()
            for i in range(s0):
                for j in range(s1):
                    out[j * s0 + i] = self._buf[i * s1 + j]
            return ndarray(out, (s1, s0), self.dtype)
        raise NotImplementedError("T for >2D not implemented")

    def __len__(self):
        return self.shape[0] if self.ndim else 0

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f"nightmare_np.ndarray(shape={self.shape}, dtype={self.dtype})"

    # ---- indexing ----
    def _resolve(self, key):
        if not isinstance(key, tuple):
            key = (key,)
        items = list(key)
        n_explicit = _b.sum(1 for it in items if it is not None and it is not Ellipsis)
        n_fill = len(self.shape) - n_explicit
        norm = []
        for it in items:
            if it is Ellipsis:
                norm.extend([slice(None)] * n_fill)
            elif it is None:
                norm.append(None)
            else:
                norm.append(it)
        while len([x for x in norm if x is not None]) < len(self.shape):
            norm.append(slice(None))
        resolved = []
        dim = 0
        for it in norm:
            if it is None:
                resolved.append((1, None, True))
            elif isinstance(it, int):
                i = it if it >= 0 else it + self.shape[dim]
                if not (0 <= i < self.shape[dim]):
                    raise IndexError(f"index {it} out of bounds for dim {dim} size {self.shape[dim]}")
                resolved.append((1, [i], False))  # int dims are consumed, dropped from shape
                dim += 1
            else:
                start, stop, step = it.indices(self.shape[dim])
                resolved.append((len(range(start, stop, step)), list(range(start, stop, step)), True))
                dim += 1
        out_shape = tuple(ol for ol, _, keep in resolved if keep)
        targets = []
        for combo in _it.product(*[range(ol) for ol, _, _ in resolved]):
            src = []
            for (ol, sids, _), c in zip(resolved, combo):
                if sids is not None:
                    src.append(sids[c])
            if not src:
                targets.append(0)
            else:
                targets.append(_flat_index(tuple(src), self.shape))
        return out_shape, targets

    def __getitem__(self, key):
        out_shape, targets = self._resolve(key)
        if out_shape == ():
            return self._buf[targets[0]]
        out = (ctypes.c_float * len(targets))()
        for k, t in enumerate(targets):
            out[k] = self._buf[t]
        return ndarray(out, out_shape, self.dtype)

    def __setitem__(self, key, value):
        out_shape, targets = self._resolve(key)
        if isinstance(value, (int, float)):
            for t in targets:
                self._buf[t] = float(value)
            return
        v = asarray(value)
        if v._n != len(targets):
            raise ValueError("shape mismatch on assignment")
        for k, t in enumerate(targets):
            self._buf[t] = v._buf[k]

    # ---- arithmetic ----
    def _binary(self, other, cname, pyfn):
        if isinstance(other, (int, float)):
            n = self._n
            out = (ctypes.c_float * n)()
            if getattr(_c, cname, None) is not None:
                b = (ctypes.c_float * n)(*([float(other)] * n))
                getattr(_c, cname)(out, self._buf, b, n)
            else:
                for i in range(n):
                    out[i] = pyfn(self._buf[i], float(other))
            return ndarray(out, self.shape, self.dtype)
        other = asarray(other)
        shape = _broadcast_shapes(self.shape, other.shape)
        A = _broadcast_to(self, shape)
        B = _broadcast_to(other, shape)
        n = A._n
        out = (ctypes.c_float * n)()
        if getattr(_c, cname, None) is not None:
            getattr(_c, cname)(out, A._buf, B._buf, n)
        else:
            for i in range(n):
                out[i] = pyfn(A._buf[i], B._buf[i])
        return ndarray(out, shape, self.dtype)

    def __add__(self, o): return self._binary(o, "add_f32", lambda a, b: a + b)
    def __sub__(self, o): return self._binary(o, "sub_f32", lambda a, b: a - b)
    def __mul__(self, o): return self._binary(o, "mul_f32", lambda a, b: a * b)
    def __truediv__(self, o): return self._binary(o, "div_f32", lambda a, b: a / b)
    def __radd__(self, o): return self._binary(o, "add_f32", lambda a, b: a + b)
    def __rsub__(self, o): return self._binary(o, "sub_f32", lambda a, b: b - a)
    def __rmul__(self, o): return self._binary(o, "mul_f32", lambda a, b: a * b)
    def __rdiv__(self, o): return self._binary(o, "div_f32", lambda a, b: b / a)
    def __neg__(self): return self._unary("neg_f32", lambda v: -v)
    def __abs__(self): return self._unary("abs_f32", lambda v: _b.abs(v))
    def __matmul__(self, o): return matmul(self, o)
    def __eq__(self, o): return self._binary(o, None, lambda a, b: 1.0 if a == b else 0.0)
    def __ne__(self, o): return self._binary(o, None, lambda a, b: 1.0 if a != b else 0.0)

    def _unary(self, cname, pyfn):
        n = self._n
        out = (ctypes.c_float * n)()
        if getattr(_c, cname, None) is not None:
            getattr(_c, cname)(out, self._buf, n)
        else:
            for i in range(n):
                out[i] = pyfn(self._buf[i])
        return ndarray(out, self.shape, self.dtype)

    # ---- transforms ----
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        shape = list(shape)
        if -1 in shape:
            known = _prod([s for s in shape if s != -1])
            shape = [self._n // known if s == -1 else s for s in shape]
        shape = tuple(shape)
        if _prod(shape) != self._n:
            raise ValueError(f"cannot reshape {self.shape} into {shape}")
        return ndarray(self._buf, shape, self.dtype)

    def astype(self, dtype):
        return self.copy()

    def copy(self):
        out = (ctypes.c_float * self._n)()
        for i in range(self._n):
            out[i] = self._buf[i]
        return ndarray(out, self.shape, self.dtype)

    def tolist(self):
        def rec(shape, off):
            if not shape:
                return self._buf[off]
            step = _prod(shape[1:]) if len(shape) > 1 else 1
            return [rec(shape[1:], off + i * step) for i in range(shape[0])]
        return rec(self.shape, 0)

    def mean(self, axis=None, keepdims=False): return mean(self, axis, keepdims)
    def sum(self, axis=None, keepdims=False): return sum_(self, axis, keepdims)
    def max(self, axis=None, keepdims=False): return max_(self, axis, keepdims)
    def min(self, axis=None, keepdims=False): return min_(self, axis, keepdims)


# ============================================================================
# Construction
# ============================================================================
def _flatten(obj):
    if isinstance(obj, (list, tuple)):
        out = []
        for x in obj:
            out.extend(_flatten(x))
        return out
    return [float(obj)]


def _infer_shape(obj):
    shape = []
    while isinstance(obj, (list, tuple)):
        shape.append(len(obj))
        if len(obj) == 0:
            break
        obj = obj[0]
    return tuple(shape)


def array(obj, dtype=None):
    if isinstance(obj, ndarray):
        return obj.copy() if dtype is None else obj.astype(dtype)
    shape = _infer_shape(obj)
    data = _flatten(obj)
    buf = (ctypes.c_float * len(data))(*data)
    return ndarray(buf, shape, dtype or "float32")


def asarray(obj, dtype=None):
    if isinstance(obj, ndarray):
        return obj
    return array(obj, dtype)


def ascontiguousarray(obj, dtype=None):
    return asarray(obj, dtype)


def zeros(shape, dtype=None):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    buf = (ctypes.c_float * _prod(shape))()
    return ndarray(buf, shape, dtype or "float32")


def ones(shape, dtype=None):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    n = _prod(shape)
    buf = (ctypes.c_float * n)(*([1.0] * n))
    return ndarray(buf, shape, dtype or "float32")


def empty(shape, dtype=None):
    return zeros(shape, dtype)


def full(shape, fill_value, dtype=None):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    n = _prod(shape)
    buf = (ctypes.c_float * n)(*([float(fill_value)] * n))
    return ndarray(buf, shape, dtype or "float32")


def empty_like(a, dtype=None):
    return zeros(a.shape, dtype or a.dtype)


def zeros_like(a, dtype=None):
    return zeros(a.shape, dtype or a.dtype)


# ============================================================================
# Unary math
# ============================================================================
def _unary_fn(cname, pyfn):
    def f(a):
        a = asarray(a)
        return a._unary(cname, pyfn)
    return f


def sqrt(a):
    a = asarray(a)
    return a._unary("sqrt_f32", lambda v: math.sqrt(v) if v >= 0.0 else float("nan"))
exp = _unary_fn("exp_f32", math.exp)
tanh = _unary_fn("tanh_f32", math.tanh)
silu = _unary_fn("silu_f32", lambda v: v / (1.0 + math.exp(-v)))
gelu = _unary_fn("gelu_f32", lambda v: 0.5 * v * (1.0 + math.tanh(
    0.7978845608028654 * (v + 0.044715 * v * v * v))))


def maximum(a, b):
    return asarray(a)._binary(b, "max_f32", lambda x, y: x if x > y else y)


def minimum(a, b):
    return asarray(a)._binary(b, "min_f32", lambda x, y: x if x < y else y)


def isnan(a):
    a = asarray(a)
    out = (ctypes.c_float * a._n)()
    for i in range(a._n):
        out[i] = 1.0 if math.isnan(a._buf[i]) else 0.0
    return ndarray(out, a.shape, a.dtype)


def any(a):
    a = asarray(a)
    return _b.any(x != 0.0 for x in a._buf)


def all(a):
    a = asarray(a)
    return _b.all(x != 0.0 for x in a._buf)


# ============================================================================
# Reductions
# ============================================================================
def _apply_reduce(vals, op):
    if op == "sum":
        return _b.sum(vals)
    if op == "mean":
        return _b.sum(vals) / len(vals)
    if op == "max":
        return _b.max(vals)
    return _b.min(vals)


def _reduce_axis(a, axis, op, keepdims):
    other = [d for d in range(a.ndim) if d != axis]
    out_shape = tuple(a.shape[d] for d in other)
    if keepdims:
        out_shape = tuple(1 if d == axis else a.shape[d] for d in range(a.ndim))
    out = []
    for combo in _it.product(*[range(a.shape[d]) for d in other]):
        vals = []
        for v in range(a.shape[axis]):
            idx = [0] * a.ndim
            for pos, d in enumerate(other):
                idx[d] = combo[pos]
            idx[axis] = v
            vals.append(a._buf[_flat_index(tuple(idx), a.shape)])
        out.append(_apply_reduce(vals, op))
    buf = (ctypes.c_float * len(out))(*out)
    return ndarray(buf, out_shape, a.dtype)


def _reduce(a, op, axis, keepdims, c_last, c_all):
    a = asarray(a)
    if axis is None:
        if getattr(_c, c_all, None) is not None:
            r = getattr(_c, c_all)(a._buf, a._n, {"sum": 0, "mean": 1, "max": 2, "min": 3}[op])
        else:
            r = _apply_reduce(list(a._buf), op)
        if keepdims:
            buf = (ctypes.c_float * 1)(r)
            return ndarray(buf, (1,) * a.ndim, a.dtype)
        return r
    if isinstance(axis, tuple):
        # reduce over multiple axes
        for ax in sorted(axis, reverse=True):
            a = _reduce_axis(a, ax % a.ndim, op, False)
        if keepdims:
            shape = list(a.shape)
            for ax in axis:
                shape.insert(ax, 1)
            a = a.reshape(shape)
        return a
    axis = axis % a.ndim
    # fast path: last axis via C
    if getattr(_c, c_last, None) is not None and axis == a.ndim - 1:
        nrows = _prod(a.shape[:-1]) if a.ndim > 1 else 1
        ncols = a.shape[-1]
        out = (ctypes.c_float * nrows)()
        getattr(_c, c_last)(out, a._buf, nrows, ncols)
        if keepdims:
            shape = tuple(1 if d == axis else a.shape[d] for d in range(a.ndim))
        else:
            shape = tuple(a.shape[d] for d in range(a.ndim) if d != axis)
        return ndarray(out, shape, a.dtype)
    return _reduce_axis(a, axis, op, keepdims)


def sum_(a, axis=None, keepdims=False): return _reduce(a, "sum", axis, keepdims, "sum_last_f32", "reduce_all_f32")
def mean(a, axis=None, keepdims=False): return _reduce(a, "mean", axis, keepdims, "mean_last_f32", "reduce_all_f32")
def max_(a, axis=None, keepdims=False): return _reduce(a, "max", axis, keepdims, "max_last_f32", "reduce_all_f32")
def min_(a, axis=None, keepdims=False): return _reduce(a, "min", axis, keepdims, "min_last_f32", "reduce_all_f32")


# ============================================================================
# matmul / dot
# ============================================================================
def matmul(a, b):
    a = asarray(a)
    b = asarray(b)
    # 1D @ 1D -> scalar
    if a.ndim == 1 and b.ndim == 1:
        return _b.sum(x * y for x, y in zip(a._buf, b._buf))
    # promote to 2D for the C kernel
    a2 = a
    b2 = b
    squeeze = None
    if a.ndim == 1:
        a2 = a.reshape((1, a.shape[0]))
        squeeze = "a"
    if b.ndim == 1:
        b2 = b.reshape((b.shape[0], 1))
        squeeze = "b" if squeeze is None else "ab"
    m, k = a2.shape
    k2, n = b2.shape
    if k != k2:
        raise ValueError(f"matmul dim mismatch: {a.shape} @ {b.shape}")
    out = (ctypes.c_float * (m * n))()
    if getattr(_c, "matmul_f32", None) is not None:
        _c.matmul_f32(out, a2._buf, b2._buf, m, k, n)
    else:
        for i in range(m):
            for j in range(n):
                s = 0.0
                for p in range(k):
                    s += a2._buf[i * k + p] * b2._buf[p * n + j]
                out[i * n + j] = s
    if squeeze == "ab":
        return out[0]
    if squeeze == "a":
        return ndarray(out, (n,), a.dtype)
    if squeeze == "b":
        return ndarray(out, (m,), a.dtype)
    return ndarray(out, (m, n), a.dtype)


def dot(a, b):
    return matmul(a, b)


# ============================================================================
# einsum
# ============================================================================
def _einsum_parse(subscripts, ops):
    s = subscripts.replace(" ", "")
    if "->" in s:
        ins, out = s.split("->")
    else:
        ins = s
        counts = {}
        for spec in ins.split(","):
            for ch in spec:
                counts[ch] = counts.get(ch, 0) + 1
        out = "".join(ch for ch in counts if counts[ch] == 1)
    ins_specs = ins.split(",")
    sizes = {}
    for spec, op in zip(ins_specs, ops):
        for ch, d in zip(spec, op.shape):
            # A label shared by two operands may be length 1 on one side;
            # numpy broadcasts that, so the effective size is the other one.
            # Without this the size depends on operand order and the whole
            # result is wrong when the length-1 operand comes second.
            if ch in sizes and sizes[ch] == 1 and d != 1:
                sizes[ch] = d
            else:
                sizes[ch] = d if ch not in sizes else sizes[ch]
    return ins_specs, out, sizes


def _einsum_labels_agree(ins_specs, ops):
    """
    Can the C kernel be trusted for these operands?

    The kernel records ONE size per label and reuses it everywhere. If a
    label appears in two operands with different lengths, it reads one
    operand past its buffer instead of broadcasting, and at large shapes
    that is not a wrong number -- it is an access violation:

        access violation reading 0x00000236CA078000

    numpy broadcasts a length-1 dimension here, so the mismatch is legal
    input. The kernel cannot express it, so those cases go to the
    (correct, slower) Python path rather than into a read of unmapped
    memory. Everything the suite actually runs has agreeing labels, so the
    fast path is still the one that gets used.
    """
    seen = {}
    for spec, op in zip(ins_specs, ops):
        for ch, d in zip(spec, op.shape):
            if ch in seen and seen[ch] != d:
                return False
            seen[ch] = d
    return True


def einsum(subscripts, *ops):
    ops = [asarray(o) for o in ops]
    ins_specs, out, sizes = _einsum_parse(subscripts, ops)
    out_shape = tuple(sizes[ch] for ch in out)
    n = _prod(out_shape)
    out_buf = (ctypes.c_float * n)()

    if getattr(_c, "einsum_f32", None) is not None and _einsum_labels_agree(ins_specs, ops):
        n_ops = len(ops)
        # c_void_p wants an ADDRESS. o._buf is the ctypes ARRAY object, not
        # a pointer, so passing it straight through raises
        #     TypeError: incompatible types, c_float_Array_3 instance
        #                 instead of c_void_p instance
        # which is how einsum was dead on this path: every call raised, and
        # because the raise happened before the fallback below, nothing
        # ever computed. cast() is what turns the buffer into its address.
        op_ptrs = (ctypes.c_void_p * n_ops)(
            *[ctypes.cast(o._buf, ctypes.c_void_p) for o in ops])
        ndims = (ctypes.c_int * n_ops)(*[o.ndim for o in ops])
        flat_shapes = [d for o in ops for d in o.shape]
        shapes = (ctypes.c_int * len(flat_shapes))(*flat_shapes)
        _c.einsum_f32(out_buf, op_ptrs, ndims, shapes, subscripts.encode(), n_ops)
        return ndarray(out_buf, out_shape, "float32")

    # pure-Python fallback
    labels = list(sizes.keys())
    for combo in _it.product(*[range(sizes[ch]) for ch in labels]):
        idx = dict(zip(labels, combo))
        val = 1.0
        for spec, op in zip(ins_specs, ops):
            # A length-1 axis is broadcast: any index into it is 0. numpy
            # does this too, and skipping it here raised "invalid index"
            # on the very inputs the C kernel could not safely take.
            oi = []
            for k, ch in enumerate(spec):
                j = idx[ch]
                if op.shape[k] == 1 and j != 0:
                    j = 0
                oi.append(j)
            val *= op._buf[_flat_index(oi, op.shape)]
        oo = tuple(idx[ch] for ch in out)
        out_buf[_flat_index(oo, out_shape)] += val
    return ndarray(out_buf, out_shape, "float32")


# ============================================================================
# misc numpy-like helpers
# ============================================================================
def reshape(a, *shape):
    return asarray(a).reshape(*shape)


def repeat(a, repeats, axis=None):
    a = asarray(a)
    if axis is None:
        data = []
        for v in a._buf:
            data.extend([v] * repeats)
        return ndarray((ctypes.c_float * len(data))(*data), (len(data),), a.dtype)
    axis = axis % a.ndim
    out_shape = list(a.shape)
    out_shape[axis] *= repeats
    out_shape = tuple(out_shape)
    n = _prod(out_shape)
    out = (ctypes.c_float * n)()
    for flat in range(n):
        idx = _unravel(flat, out_shape)
        src = list(idx)
        src[axis] = idx[axis] // repeats
        out[flat] = a._buf[_flat_index(tuple(src), a.shape)]
    return ndarray(out, out_shape, a.dtype)


def triu(a, k=0):
    a = asarray(a)
    out = a.copy()
    m, n = a.shape
    for i in range(m):
        for j in range(n):
            if j < i + k:
                out._buf[i * n + j] = 0.0
    return out


def cumsum(a, axis=None):
    a = asarray(a)
    if axis is None:
        data = [0.0] * a._n
        s = 0.0
        for i in range(a._n):
            s += a._buf[i]
            data[i] = s
        return ndarray((ctypes.c_float * a._n)(*data), a.shape, a.dtype)
    axis %= a.ndim
    out = a.copy()
    other = [d for d in range(a.ndim) if d != axis]
    for combo in _it.product(*[range(a.shape[d]) for d in other]):
        s = 0.0
        for v in range(a.shape[axis]):
            idx = [0] * a.ndim
            for pos, d in enumerate(other):
                idx[d] = combo[pos]
            idx[axis] = v
            f = _flat_index(tuple(idx), a.shape)
            s += a._buf[f]
            out._buf[f] = s
    return out


def argsort(a, axis=-1):
    a = asarray(a)
    axis %= a.ndim
    other = [d for d in range(a.ndim) if d != axis]
    out = a.copy()
    for combo in _it.product(*[range(a.shape[d]) for d in other]):
        pairs = []
        for v in range(a.shape[axis]):
            idx = [0] * a.ndim
            for pos, d in enumerate(other):
                idx[d] = combo[pos]
            idx[axis] = v
            pairs.append((a._buf[_flat_index(tuple(idx), a.shape)], v))
        pairs.sort()
        for pos, (_, orig) in enumerate(pairs):
            idx = [0] * a.ndim
            for p2, d in enumerate(other):
                idx[d] = combo[p2]
            idx[axis] = pos
            out._buf[_flat_index(tuple(idx), a.shape)] = float(orig)
    return out


def argpartition(a, kth, axis=-1):
    return argsort(a, axis)


def argmax(a, axis=None):
    a = asarray(a)
    if axis is None:
        return int(_b.max(range(a._n), key=lambda i: a._buf[i]))
    axis %= a.ndim
    other = [d for d in range(a.ndim) if d != axis]
    out_shape = tuple(a.shape[d] for d in other)
    out = []
    for combo in _it.product(*[range(a.shape[d]) for d in other]):
        best = 0
        bv = None
        for v in range(a.shape[axis]):
            idx = [0] * a.ndim
            for pos, d in enumerate(other):
                idx[d] = combo[pos]
            idx[axis] = v
            val = a._buf[_flat_index(tuple(idx), a.shape)]
            if bv is None or val > bv:
                bv = val
                best = v
        out.append(float(best))
    return ndarray((ctypes.c_float * len(out))(*out), out_shape, a.dtype)


def searchsorted(sorted_a, v):
    a = asarray(sorted_a)
    lo, hi = 0, a._n
    while lo < hi:
        mid = (lo + hi) // 2
        if a._buf[mid] < v:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ============================================================================
# dtypes / constants
# ============================================================================
float32 = float
float_ = float
int_ = int
inf = float("inf")
nan = float("nan")


# ============================================================================
# random
# ============================================================================
class _Random:
    def seed(self, s=None):
        _random.seed(s)

    def _make(self, shape, gen):
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        n = _prod(shape)
        buf = (ctypes.c_float * n)(*[gen() for _ in range(n)])
        return ndarray(buf, shape, "float32")

    def randn(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return self._make(shape, lambda: _random.gauss(0.0, 1.0))

    def rand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return self._make(shape, lambda: _random.random())

    def normal(self, loc=0.0, scale=1.0, size=None):
        shape = size if size is not None else ()
        return self._make(shape, lambda: _random.gauss(loc, scale))

    def uniform(self, low=0.0, high=1.0, size=None):
        shape = size if size is not None else ()
        return self._make(shape, lambda: low + (high - low) * _random.random())


random = _Random()


# ============================================================================
# rmsnorm / softmax convenience (fused)
# ============================================================================
def rmsnorm(x, weight, eps=1e-6):
    x = asarray(x)
    w = asarray(weight)
    nrows = _prod(x.shape[:-1]) if x.ndim > 1 else 1
    ncols = x.shape[-1]
    out = (ctypes.c_float * x._n)()
    if getattr(_c, "rmsnorm_f32", None) is not None:
        _c.rmsnorm_f32(out, x._buf, w._buf, nrows, ncols, eps)
    else:
        for r in range(nrows):
            base = r * ncols
            s = 0.0
            for i in range(ncols):
                s += x._buf[base + i] ** 2
            inv = 1.0 / math.sqrt(s / ncols + eps)
            for i in range(ncols):
                out[base + i] = x._buf[base + i] * inv * w._buf[i]
    return ndarray(out, x.shape, x.dtype)


def softmax(x, axis=-1):
    x = asarray(x)
    if axis in (-1, x.ndim - 1):
        nrows = _prod(x.shape[:-1]) if x.ndim > 1 else 1
        ncols = x.shape[-1]
        out = (ctypes.c_float * x._n)()
        if getattr(_c, "softmax_rows_f32", None) is not None:
            _c.softmax_rows_f32(out, x._buf, nrows, ncols)
        else:
            for r in range(nrows):
                base = r * ncols
                mx = _b.max(x._buf[base:base + ncols])
                s = 0.0
                for i in range(ncols):
                    e = math.exp(x._buf[base + i] - mx)
                    out[base + i] = e
                    s += e
                for i in range(ncols):
                    out[base + i] /= s
        return ndarray(out, x.shape, x.dtype)
    return _softmax_general(x, axis)


def _softmax_general(x, axis):
    axis %= x.ndim
    other = [d for d in range(x.ndim) if d != axis]
    out = x.copy()
    for combo in _it.product(*[range(x.shape[d]) for d in other]):
        vals = []
        for v in range(x.shape[axis]):
            idx = [0] * x.ndim
            for pos, d in enumerate(other):
                idx[d] = combo[pos]
            idx[axis] = v
            vals.append((_flat_index(tuple(idx), x.shape), x._buf[_flat_index(tuple(idx), x.shape)]))
        mx = _b.max(v for _, v in vals)
        s = _b.sum(math.exp(v - mx) for _, v in vals)
        for f, v in vals:
            out._buf[f] = math.exp(v - mx) / s
    return out


# ============================================================================
# public surface
# ============================================================================
def backend_status():
    return _c.reason


# numpy-style aliases (defined last so they don't shadow builtins above)
sum = sum_
max = max_
min = min_


__all__ = [
    "ndarray", "array", "asarray", "ascontiguousarray",
    "zeros", "ones", "empty", "full", "empty_like", "zeros_like",
    "sqrt", "exp", "tanh", "silu", "gelu", "maximum", "minimum",
    "isnan", "any", "all",
    "sum_", "mean", "max_", "min_", "sum", "max", "min",
    "matmul", "dot", "einsum",
    "reshape", "repeat", "triu", "cumsum", "argsort", "argpartition",
    "argmax", "searchsorted",
    "float32", "float_", "int_", "inf", "nan", "random",
    "rmsnorm", "softmax", "backend_status",
]


# ============================================================================
# compatibility layer (_ext.py)
# ============================================================================
# The engine calls numpy names that live in _ext (dtypes, frombuffer,
# arange, clip, linalg, ...). Wire them in so `import nightmare_np as np`
# covers the full API. _ext's __all__ is the allow-list: names __init__
# already defines well (matmul, einsum, rmsnorm, softmax, ...) are not in
# it, so they stay put.
try:
    from ._ext import *
    from ._ext import __all__ as _ext_all
    __all__ = __all__ + [n for n in _ext_all if n not in __all__]
except ImportError:
    pass
