# -*- coding: utf-8 -*-
"""
ntmre_v16_engine.py — NTMRE version-16 binary support for the suite
===================================================================
Two v16 on-disk variants exist (both written by the user's own tools):

  CLASSIC (e.g. qwen15b_q4km.ntmre):
    [0:4] "RMTN", [4:8] version=16
    [128:]  config as u64/f64 fields, [224:] arch_type (u64)
    [1064:] tokenizer: vocab_size tokens, each u16 len + bytes
    [after tokenizer] tensor index: n_tensors (u32) + 256-byte entries
    [aligned 4096] tensor data (Q8: f32 scale + 32 x int8 per 32 elems)
    .ntmre compression tensors: rank>0, factor P at off_a (rows x rank), Q at off_b
    (rank x cols), offsets relative to the data region.

  NEO (e.g. Qwen3.6-27B-NEO-...ntmre) — qwen35 hybrid weights:
    same header/config, but the tensor index lives at the END of the
    file, starting at the config's tensor_data_offset field.
    Entries are 184 bytes: name[128] + 7 x u64:
      f0: lo32 = dtype (0=F32, 1=Q8), hi32 = n_dims
      f1: cols (n for 1D), f2: rows (0 for 1D)
      f5: absolute data offset, f6: byte size
    .ntmre compression stored as separate "<name>.A" (out x rank) / "<name>.B"
    (rank x in) Q8 tensors.

Both variants store 2D weights row-major as (rows, cols) = (out, in),
so the suite's universal rule applies: W_pt = flat.reshape((out, in)).

The V16Source class exposes the same interface as NSVDSource
(config / tokenizer_bytes / entries / read_flat / read_rows / close)
so both Qwen35HybridEngine and GPUStreamEngine can consume it.

Author: Nightmare Designs 💀
"""
import json
import os
import struct
from pathlib import Path

import nightmare_np as np

MAGIC = b'RMTN'
CONFIG_OFFSET = 128
ARCH_OFFSET = 224
TOKENIZER_OFFSET = 1064
INDEX_ENTRY_CLASSIC = 256
INDEX_ENTRY_NEO = 184
Q8_BLOCK = 32
Q8_BLOCK_BYTES = 4 + 32   # f32 scale + 32 x int8

# entries[name] dtype tags
V16_F32 = 0
V16_Q8 = 1
V16_SVD = 2     # virtual, materialized from .A/.B pair

_ARCH_MAP = {0: 'llama', 1: 'gemma', 2: 'mistral', 3: 'qwen'}

# Default BPE merges for v16 files (the format stores tokens only).
#
# Portable: a list searched in order rather than one baked-in absolute path.
# The old value pointed into one person's Downloads folder, so on every other
# machine -- including a customer's -- this resolved to nothing.
_DEFAULT_MERGES_CANDIDATES = [
    str(Path(__file__).resolve().parent / "qwen2.5_merges.json"),
    str(Path.home() / "Downloads" / "qwen2.5_merges.json"),
    str(Path(os.environ.get("USERPROFILE", str(Path.home())))
        / "Downloads" / "bio 2" / "qwen2.5_merges.json"),
]


def _find_default_merges():
    """First merges file that actually exists, else None."""
    for p in _DEFAULT_MERGES_CANDIDATES:
        try:
            if os.path.isfile(p):
                return p
        except Exception:
            continue
    return None


_DEFAULT_MERGES = _find_default_merges()


def _u64(buf, off):
    return struct.unpack_from('<Q', buf, off)[0]


def _f64(buf, off):
    return struct.unpack_from('<d', buf, off)[0]


def _dequant_q8(data, n):
    """v16 Q8 blocks: f32 scale + 32 int8 per 32 elements -> f32."""
    n_blocks = (n + Q8_BLOCK - 1) // Q8_BLOCK
    raw = np.frombuffer(data, dtype=np.uint8, count=n_blocks * Q8_BLOCK_BYTES)
    blk = raw.reshape(n_blocks, Q8_BLOCK_BYTES)
    scales = np.frombuffer(blk[:, 0:4].tobytes(), dtype=np.float32)
    vals = blk[:, 4:4 + Q8_BLOCK].astype(np.int8)
    return (vals * scales[:, None]).astype(np.float32).ravel()[:n]


def _u64_cfg_sane(head, fsize):
    """True if the u64 config block @128 looks like a real v16-style config.
    Used to accept RMTN version-1 files that were written with the v16
    (NEO) layout (e.g. smollm2-135m_svdq8.ntmre)."""
    try:
        dim, n_layers, _h, _kv, _hid, vocab = struct.unpack_from('<6Q', head, 128)
        tdo = struct.unpack_from('<Q', head, 200)[0]
        nt = struct.unpack_from('<Q', head, 208)[0]
    except struct.error:
        return False
    return (32 <= dim <= 65536 and 1 <= n_layers <= 512
            and 100 <= vocab <= 10_000_000
            and 1 <= nt <= 200_000 and 0 < tdo < fsize)


def is_v16_file(path):
    """True if the file is an NTMRE v16 binary (RMTN magic, version 16),
    or a version-1 file written in the v16 layout."""
    try:
        fsize = os.path.getsize(path)
        with open(path, 'rb') as f:
            head = f.read(4096)
    except OSError:
        return False
    if len(head) < 8 or head[:4] != MAGIC:
        return False
    version = struct.unpack_from('<I', head, 4)[0]
    if version == 16:
        return True
    if version == 1 and len(head) >= 232:
        return _u64_cfg_sane(head, fsize)
    return False


class V16Source:
    """mmap reader for NTMRE v16 binaries (classic + NEO variants)."""

    def __init__(self, path):
        self.path = str(path)
        with open(path, 'rb') as f:
            head = f.read(4096)
        if head[:4] != MAGIC:
            raise ValueError(f"Not a v16 NTMRE file (magic={head[:4]!r})")
        version = struct.unpack_from('<I', head, 4)[0]
        if version == 16:
            pass
        elif version == 1 and _u64_cfg_sane(head, os.path.getsize(path)):
            pass   # v1 file written in the v16 (NEO) layout
        else:
            raise ValueError(f"Not a version-16 NTMRE file (version={version})")

        # ── config @128 (u64/f64 fields) ──
        o = CONFIG_OFFSET
        dim = _u64(head, o); o += 8
        n_layers = _u64(head, o); o += 8
        n_heads = _u64(head, o); o += 8
        n_kv_heads = _u64(head, o); o += 8
        hidden_dim = _u64(head, o); o += 8
        vocab_size = _u64(head, o); o += 8
        max_seq_len = _u64(head, o); o += 8
        rope_theta = _f64(head, o); o += 8
        head_dim = _u64(head, o); o += 8
        tensor_data_offset = _u64(head, o); o += 8
        n_tensors = _u64(head, o); o += 8
        arch_type = _u64(head, ARCH_OFFSET)

        self.config = {
            'dim': int(dim), 'n_layers': int(n_layers),
            'n_heads': int(n_heads), 'n_kv_heads': int(n_kv_heads),
            'ffn_dim': int(hidden_dim), 'hidden_dim': int(hidden_dim),
            'vocab_size': int(vocab_size), 'max_seq_len': int(max_seq_len),
            'rope_theta': float(rope_theta),
            'head_dim': int(head_dim) or int(dim) // max(int(n_heads), 1),
            'arch': _ARCH_MAP.get(int(arch_type), 'llama'),
        }

        # ── tokenizer @1064 (tokens only; merges supplied separately) ──
        self._tokens = []
        tok_end = TOKENIZER_OFFSET
        with open(path, 'rb') as f:
            f.seek(TOKENIZER_OFFSET)
            for _ in range(int(vocab_size)):
                ln = struct.unpack('<H', f.read(2))[0]
                self._tokens.append(f.read(ln).decode('utf-8', 'replace'))
            tok_end = f.tell()

        # ── tensor index: NEO (at tensor_data_offset) or CLASSIC (after tok) ──
        self._fsize = os.path.getsize(path)
        self.entries = {}       # name -> (dtype, shape, nbytes)
        self._plain = {}        # name -> (dtype, rows, cols, abs_off, nbytes)
        self._svd = {}          # base  -> (rank, out, in)
        self._neo = False

        region = self._fsize - tensor_data_offset
        if (0 < tensor_data_offset < self._fsize
                and region % INDEX_ENTRY_NEO == 0
                and region // INDEX_ENTRY_NEO == int(n_tensors)):
            self._parse_index_neo(path, tensor_data_offset, int(n_tensors))
        elif (0 < tensor_data_offset < self._fsize
                and (region - 4) % INDEX_ENTRY_NEO == 0
                and (region - 4) // INDEX_ENTRY_NEO == int(n_tensors)):
            # v1-layout variant: u32 n_tensors header before the entries
            with open(path, 'rb') as f:
                f.seek(tensor_data_offset)
                hdr = struct.unpack('<I', f.read(4))[0]
            if hdr == int(n_tensors):
                self._parse_index_neo(path, tensor_data_offset + 4, int(n_tensors))
            else:
                self._parse_index_classic(path, tok_end, int(n_tensors))
        else:
            self._parse_index_classic(path, tok_end, int(n_tensors))

        # qwen35 hybrid detection: SSM tensors present
        if any(n.startswith('blk.0.ssm_') for n in self.entries):
            self.config['arch'] = 'qwen35'
            self._infer_qwen35_config()

        # Interrupted conversions leave pre-allocated zero holes where
        # tensor data was never written (e.g. Qwen3.6-27B-NEO: 833/1331
        # tensors zero-filled).  Detect and report instead of generating
        # garbage tokens from zero weights.
        self._verify_data()

        # ── tokenizer JSON (tokens + optional merges file) ──
        merges = []
        mp = os.environ.get('NTMRE_V16_MERGES', _DEFAULT_MERGES)
        try:
            with open(mp, 'r', encoding='utf-8') as f:
                merges = json.load(f)
        except Exception:
            pass
        # detect special tokens by name (qwen2/3 layout)
        t2i = {t: i for i, t in enumerate(self._tokens)}
        eot = t2i.get(chr(60) + chr(124) + 'endoftext' + chr(124) + chr(62), 0)
        im_end = t2i.get(chr(60) + chr(124) + 'im_end' + chr(124) + chr(62), eot)
        self.tokenizer_bytes = json.dumps({
            'tokens': self._tokens, 'merges': merges,
            'bos': eot, 'eos': im_end, 'pad': eot, 'unk': 0,
            'model': 'qwen' if self.config['arch'].startswith('qwen') else '',
        }).encode('utf-8')

        self._mmap = None

    # ── index parsing ─────────────────────────────────────────────
    def _parse_index_neo(self, path, idx_off, n_tensors):
        self._neo = True
        with open(path, 'rb') as f:
            f.seek(idx_off)
            ibuf = f.read(n_tensors * INDEX_ENTRY_NEO)
        raw = {}
        for i in range(n_tensors):
            e = ibuf[i * INDEX_ENTRY_NEO:(i + 1) * INDEX_ENTRY_NEO]
            name = e[:128].split(b'\x00')[0].decode('utf-8', 'replace')
            f0, f1, f2, _f3, _f4, f5, f6 = struct.unpack_from('<7Q', e, 128)
            dtype = f0 & 0xFFFFFFFF
            nd = f0 >> 32
            cols, rows = int(f1), int(f2)
            raw[name] = (int(dtype), int(nd), rows, cols, int(f5), int(f6))

        # .ntmre compression pairs exist only as <base>.A / <base>.B entries — detect them
        for name in raw:
            if name.endswith('.A') and name[:-2] + '.B' in raw:
                base = name[:-2]
                a = raw[name]          # (out, rank)
                b = raw[base + '.B']   # (rank, in)
                out, rank, in_ = a[2], a[3], b[3]
                self._svd[base] = (rank, out, in_)
                self.entries[base] = (V16_SVD, (in_, out), (out + in_) * rank)

        for name, (dtype, nd, rows, cols, off, nbytes) in raw.items():
            if name.endswith('.A') or name.endswith('.B'):
                # keep .ntmre compression factors reachable for read_flat, but do not
                # expose them as standalone tensors
                self._plain[name] = (dtype, rows, cols, off, nbytes)
                continue
            if name in self._svd:
                continue
            if nd <= 1:
                n = rows if rows else cols
                self._plain[name] = (dtype, n, 0, off, nbytes)
                self.entries[name] = (dtype, (n,), nbytes)
            else:
                self._plain[name] = (dtype, rows, cols, off, nbytes)
                self.entries[name] = (dtype, (cols, rows), nbytes)

    def _parse_index_classic(self, path, tok_end, _n_tensors_cfg):
        with open(path, 'rb') as f:
            f.seek(tok_end)
            n = struct.unpack('<I', f.read(4))[0]
            idx_start = tok_end + 4
            ibuf = f.read(n * INDEX_ENTRY_CLASSIC)
        data_offset = (idx_start + n * INDEX_ENTRY_CLASSIC + 4095) & ~4095
        raw = {}
        for i in range(n):
            o = i * INDEX_ENTRY_CLASSIC
            name_len = struct.unpack_from('<I', ibuf, o)[0]; o += 4
            name = ibuf[o:o + name_len].decode('utf-8', 'replace'); o += name_len
            n_dims = struct.unpack_from('<I', ibuf, o)[0]; o += 4
            rows = struct.unpack_from('<I', ibuf, o)[0]; o += 4
            cols = struct.unpack_from('<I', ibuf, o)[0] if n_dims >= 2 else 0
            o += 4 if n_dims >= 2 else 0
            rank = struct.unpack_from('<I', ibuf, o)[0]; o += 4
            off_a = struct.unpack_from('<Q', ibuf, o)[0]; o += 8
            off_b = struct.unpack_from('<Q', ibuf, o)[0]; o += 8
            nba = struct.unpack_from('<I', ibuf, o)[0]; o += 4
            nbb = struct.unpack_from('<I', ibuf, o)[0]; o += 4
            is_1d = ibuf[o]
            raw[name] = (int(n_dims), int(rows), int(cols), int(rank),
                         data_offset + off_a, data_offset + off_b,
                         int(nba), int(nbb), int(is_1d))

        for name, (nd, rows, cols, rank, oa, ob, nba, nbb, is_1d) in raw.items():
            if nd <= 1 or is_1d:
                self._plain[name] = (V16_Q8, rows, 0, oa, nba * Q8_BLOCK_BYTES)
                self.entries[name] = (V16_Q8, (rows,), nba * Q8_BLOCK_BYTES)
            elif rank > 0:
                self._svd[name] = (rank, rows, cols)
                self.entries[name] = (V16_SVD, (cols, rows),
                                      (rows + cols) * rank)
                self._plain[name + '.__A'] = (V16_Q8, rows * rank, 0, oa,
                                               nba * Q8_BLOCK_BYTES)
                self._plain[name + '.__B'] = (V16_Q8, rank * cols, 0, ob,
                                               nbb * Q8_BLOCK_BYTES)
            else:
                self._plain[name] = (V16_Q8, rows, cols, oa,
                                     nba * Q8_BLOCK_BYTES)
                self.entries[name] = (V16_Q8, (cols, rows),
                                      nba * Q8_BLOCK_BYTES)

    def _infer_qwen35_config(self):
        """Fill qwen35 SSM hyperparams from tensor shapes (NEO files)."""
        e = self.entries
        dt = e.get('blk.0.ssm_dt.bias', (None, (48,), 0))[1]
        dt_rank = int(dt[0]) if dt else 48
        sn = e.get('blk.0.ssm_norm.weight', (None, (128,), 0))[1]
        head_v = int(sn[0]) if sn else 128
        conv = e.get('blk.0.ssm_conv1d.weight', (None, (4, 10240), 0))[1]
        conv_dim = int(conv[-1]) if conv else 10240
        ck = int(conv[0]) if len(conv) > 1 else 4
        inner = head_v * dt_rank
        # conv_dim = state*n_group*2 + inner  (qwen35 standard: 128*16)
        state, n_group = 128, 16
        rem = conv_dim - inner
        if rem > 0 and rem % (2 * state) == 0:
            n_group = rem // (2 * state)
        self.config.update({
            'ssm_dt_rank': dt_rank,
            'ssm_inner': inner,
            'ssm_state_size': state,
            'ssm_n_group': n_group,
            'ssm_conv_kernel': ck,
            'rope_dim': 64,
            'full_attn_interval': 4,
            'norm_eps': 1e-6,
        })

    def _verify_data(self):
        """Sample large tensors' data regions; an all-zero region means
        the conversion was interrupted after pre-allocating the file."""
        zero = []
        with open(self.path, 'rb') as f:
            for name, (_dt, rows, cols, off, nbytes) in self._plain.items():
                if nbytes < (1 << 20):   # only big weights can't be legit-zero
                    continue
                nz = 0
                for k in range(3):
                    pos = off + (nbytes * k) // 3
                    f.seek(pos)
                    buf = f.read(min(4096, off + nbytes - pos))
                    nz += sum(1 for b in buf if b)
                if nz == 0:
                    zero.append(name)
        if zero:
            raise ValueError(
                f'conversion incomplete: {len(zero)} tensor(s) are zero-filled '
                f'(the file was pre-allocated but the converter did not finish '
                f'writing).  First: {zero[0]}.  Re-run the conversion to '
                f'produce a usable model.')

    # ── data access ───────────────────────────────────────────────
    def _mm(self):
        if self._mmap is None:
            import mmap
            self._fh = open(self.path, 'rb')
            self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmap

    def _read_typed(self, dtype, off, nbytes, n):
        data = bytes(self._mm()[off:off + nbytes])
        if dtype == V16_F32:
            return np.frombuffer(data, dtype=np.float32).astype(np.float32)[:n]
        return _dequant_q8(data, n)

    def read_flat(self, name):
        """(flat f32 in storage order, index shape).  .ntmre compression pairs are
        materialized as (A @ B).ravel() = W_pt.ravel()."""
        if name in self._svd:
            rank, out, in_ = self._svd[name]
            if self._neo:
                a_dt, _ar, _ac, a_off, a_nb = self._plain[name + '.A']
                b_dt, _br, _bc, b_off, b_nb = self._plain[name + '.B']
            else:
                a_dt, _ar, _ac, a_off, a_nb = self._plain[name + '.__A']
                b_dt, _br, _bc, b_off, b_nb = self._plain[name + '.__B']
            A = self._read_typed(a_dt, a_off, a_nb, out * rank).reshape(out, rank)
            B = self._read_typed(b_dt, b_off, b_nb, rank * in_).reshape(rank, in_)
            flat = (A @ B).ravel()
            return np.ascontiguousarray(flat, dtype=np.float32), (in_, out)
        dtype, rows, cols, off, nbytes = self._plain[name]
        if cols == 0:
            flat = self._read_typed(dtype, off, nbytes, rows)
            return np.ascontiguousarray(flat, dtype=np.float32), (rows,)
        flat = self._read_typed(dtype, off, nbytes, rows * cols)
        return np.ascontiguousarray(flat, dtype=np.float32), (cols, rows)

    def read_raw(self, name):
        """(dtype, raw bytes, shape) for GPU upload.  .ntmre compression virtual names
        return ('svd', None, (in_, out)) — use read_raw_svd for those."""
        if name in self._svd:
            rank, out, in_ = self._svd[name]
            return ('svd', None, (in_, out))
        dtype, rows, cols, off, nbytes = self._plain[name]
        raw = bytes(self._mm()[off:off + nbytes])
        shape = (rows,) if cols == 0 else (cols, rows)
        return (dtype, raw, shape)

    def read_raw_svd(self, name):
        """Raw factor blobs for an .ntmre compression pair: ((dtA, rawA), (dtB, rawB),
        (rank, out, in_))."""
        rank, out, in_ = self._svd[name]
        if self._neo:
            a_dt, _ar, _ac, a_off, a_nb = self._plain[name + '.A']
            b_dt, _br, _bc, b_off, b_nb = self._plain[name + '.B']
        else:
            a_dt, _ar, _ac, a_off, a_nb = self._plain[name + '.__A']
            b_dt, _br, _bc, b_off, b_nb = self._plain[name + '.__B']
        raw_a = bytes(self._mm()[a_off:a_off + a_nb])
        raw_b = bytes(self._mm()[b_off:b_off + b_nb])
        return (a_dt, raw_a), (b_dt, raw_b), (rank, out, in_)

    def read_rows(self, name, ids, row_len):
        """Dequantize only the given rows of a 2D tensor."""
        if name in self._svd:
            flat, _ = self.read_flat(name)
            _rank, out, in_ = self._svd[name]
            w = flat.reshape(out, in_)
            return [w[int(t)].astype(np.float32) for t in ids]
        dtype, rows, cols, off, nbytes = self._plain[name]
        n_rows = rows if cols else (rows // row_len)
        if dtype == V16_F32:
            bpr = row_len * 4
            return [np.frombuffer(
                bytes(self._mm()[off + int(t) * bpr: off + (int(t) + 1) * bpr]),
                dtype=np.float32).astype(np.float32) for t in ids]
        if dtype == V16_Q8 and row_len % Q8_BLOCK == 0:
            bpr = (row_len // Q8_BLOCK) * Q8_BLOCK_BYTES
            return [_dequant_q8(
                bytes(self._mm()[off + int(t) * bpr: off + (int(t) + 1) * bpr]),
                row_len) for t in ids]
        flat, _ = self.read_flat(name)
        w = flat.reshape(n_rows, row_len)
        return [w[int(t)].astype(np.float32) for t in ids]

    def close(self):
        if self._mmap is not None:
            try:
                self._mmap.close()
                self._fh.close()
            except Exception:
                pass
            self._mmap = None


# ═══════════════════════════════════════════════════════════════════
# Suite engine for non-hybrid v16 models (llama/qwen/gemma/mistral)
# ═══════════════════════════════════════════════════════════════════
def NTMREV16SuiteEngine(model_path, *args, **kwargs):
    """GPUStreamEngine handles v16 files natively via V16Source (see
    nightmare_stream.GPUStreamEngine._load_ntmre_header)."""
    from nightmare_stream import GPUStreamEngine
    return GPUStreamEngine(model_path)


def create_v16_engine(model_path, status=print):
    """Route a v16 file to the right suite engine based on its contents."""
    src = V16Source(model_path)
    arch = src.config['arch']
    src.close()
    if arch == 'qwen35':
        from nightmare_qwen35 import Qwen35HybridEngine
        status("⏳ v16 qwen35 hybrid detected — routing to Qwen3.5 Hybrid Engine...")
        return Qwen35HybridEngine(model_path)
    status(f"⏳ v16 {arch} detected — routing to NTMRE Streaming Engine...")
    return NTMREV16SuiteEngine(model_path)
