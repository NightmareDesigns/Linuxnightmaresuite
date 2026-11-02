"""
ntmre_v1_engine.py — NTMRE version-1 binary support (C converter layout)
=========================================================================
Files written by convert.c / ntmre_convert_gpu.exe (RMTN magic, version 1):

  [0:4]   "RMTN"
  [4:8]   u32 version = 1
  [16:]   u32 config: dim, n_layers, n_heads, n_kv_heads, hidden, vocab, seq
          f32 rope_base, u32 head_dim, u32 arch (0=llama, 1=gemma4),
          then padding (per_layer_hidden_dims area) up to the tokenizer
  tokenizer: u32 tok_data_size, u32 vocab_size, then per token
          u32 len + bytes + f32 score, then u32 n_merges + merge strings
  index:    u32 n_tensors, then n_tensors FIXED 256-byte slots:
          u32 name_len + name + u32 n_dims + u32 rows [+ u32 cols]
          + u32 rank + u64 offset_a + u64 offset_b
          + u32 num_blocks_a + u32 num_blocks_b + u8 is_1d + zero padding
  data:     aligned to 4096; Q8 blocks of 36 bytes (f32 scale + 32 x int8).
          rank > 0: .ntmre compression factors A (rows x rank) at offset_a, B (rank x cols)
          at offset_b;  W = A @ B.  rank == 0: direct Q8 (rows x cols).
          Offsets are relative to the aligned data start.

The converter writes placeholder (zeroed) index slots first and seeks back
to fill them at the very end, so an interrupted conversion leaves a zeroed
index — such files are rejected here with a clear error.
"""

import json
import os
import struct

import numpy as np

MAGIC = b'RMTN'
INDEX_ENTRY = 256
Q8_BLOCK = 32
Q8_BYTES = 36          # f32 scale + 32 x int8  (same as the v16 Q8 dialect)
ALIGN = 4096

# dtype tags mirror V16Source so the GPU upload paths can treat both alike
V1_F32 = 0
V1_Q8 = 1            # 36B/32, f32 scale
V1_SVD = 'svd'


def _dequant_q8_36(data, n):
    """36-byte blocks: f32 scale + 32 int8.  Handles a partial tail block
    (e.g. 1-element tensors like layer_output_scale)."""
    nb = (n + Q8_BLOCK - 1) // Q8_BLOCK
    raw = np.frombuffer(data, dtype=np.uint8)[:nb * Q8_BYTES]
    blk = raw.reshape(nb, Q8_BYTES)
    scales = blk[:, :4].copy().view(np.float32).reshape(nb)
    vals = blk[:, 4:4 + Q8_BLOCK].astype(np.int8)
    return (vals * scales[:, None]).astype(np.float32).ravel()[:n]


def _find_tokenizer(head, vocab):
    """Locate the tokenizer: u32 vocab_size followed by u32 len + token."""
    for o in range(56, len(head) - 16, 4):
        if struct.unpack_from('<I', head, o)[0] == vocab:
            ln = struct.unpack_from('<I', head, o + 4)[0]
            if 1 <= ln <= 64:
                t = head[o + 8:o + 8 + ln]
                if all(32 <= c or c >= 9 for c in t):
                    return o
    return None


def is_v1_readable(path):
    """True for RMTN v1 files with an intact (non-zero) tensor index."""
    try:
        src = V1Source(path)
        src.close()
        return True
    except Exception:
        return False


class V1Source:
    """mmap reader for NTMRE v1 binaries (C converter layout)."""

    # marks Q8 as the 36-byte dialect for GPUStreamEngine._try_gpu_weight
    _q8_is_36B = True

    def __init__(self, path):
        self.path = str(path)
        self._fsize = os.path.getsize(path)
        with open(path, 'rb') as f:
            head = f.read(1 << 20)
        if head[:4] != MAGIC or struct.unpack_from('<I', head, 4)[0] != 1:
            raise ValueError("Not a version-1 NTMRE file")

        (dim, n_layers, n_heads, n_kv, hidden, vocab, seq) = struct.unpack_from('<7I', head, 16)
        rope = struct.unpack_from('<f', head, 44)[0]
        head_dim_cfg = struct.unpack_from('<I', head, 48)[0]
        arch_id = struct.unpack_from('<I', head, 52)[0]
        if not (32 <= dim <= 65536 and 1 <= n_layers <= 512 and 100 <= vocab <= 10_000_000):
            raise ValueError("v1 config block is not sane")

        # ── tokenizer (offset varies between converter builds) ──
        tok_at = _find_tokenizer(head, vocab)
        if tok_at is None:
            raise ValueError("v1 tokenizer not found")
        self._tokens = []
        merges = []
        with open(path, 'rb') as f:
            f.seek(tok_at + 4)
            o = tok_at + 4
            for _ in range(vocab):
                ln = struct.unpack('<I', f.read(4))[0]
                if ln > 1 << 16:
                    raise ValueError("v1 tokenizer corrupt")
                self._tokens.append(f.read(ln).decode('utf-8', 'replace'))
                f.seek(4, 1)          # f32 score
                o += 4 + ln + 4
            f.seek(o)
            n_merges = struct.unpack('<I', f.read(4))[0]
            o += 4
            for _ in range(min(n_merges, 5_000_000)):
                f.seek(o)
                ln = struct.unpack('<I', f.read(4))[0]
                if ln > 1 << 16:
                    break
                merges.append(f.read(ln).decode('utf-8', 'replace'))
                o += 4 + ln
            f.seek(o)
            n_tensors = struct.unpack('<I', f.read(4))[0]
            idx_off = o + 4
            if not (0 < n_tensors <= 200_000):
                raise ValueError("v1 tensor count not sane")
            f.seek(idx_off)
            ibuf = f.read(n_tensors * INDEX_ENTRY)
        if len(ibuf) < n_tensors * INDEX_ENTRY:
            raise ValueError("v1 index truncated")
        if not any(ibuf):
            raise ValueError(
                'conversion incomplete: the tensor index was never written '
                '(all zero).  Re-convert from the source GGUF.')

        # ── tensor index (fixed 256B slots) ──
        self.entries = {}       # name -> (dtype, shape, nbytes)
        self._plain = {}        # name -> (dtype, rows, cols, abs_off, nbytes)
        self._svd = {}          # base -> (rank, out, in)
        data_start = (idx_off + n_tensors * INDEX_ENTRY + ALIGN - 1) & ~(ALIGN - 1)
        self._data_start = data_start
        for i in range(n_tensors):
            e = ibuf[i * INDEX_ENTRY:(i + 1) * INDEX_ENTRY]
            nl = struct.unpack_from('<I', e, 0)[0]
            if nl == 0 or nl > 127:
                raise ValueError(f"v1 index entry {i} corrupt (name_len={nl})")
            name = e[4:4 + nl].decode('utf-8', 'replace')
            o = 4 + nl
            nd, rows = struct.unpack_from('<II', e, o)
            o += 8
            cols = 0
            if nd >= 2:
                cols = struct.unpack_from('<I', e, o)[0]
                o += 4
            rank, off_a, off_b, nba, nbb = struct.unpack_from('<IQQII', e, o)
            is_1d = e[o + 20]
            if rank > 0 and nd >= 2:
                # .ntmre compression: A (rows=out, rank) @ B (rank, cols=in)
                self._svd[name] = (int(rank), int(rows), int(cols))
                self._plain[name + '.A'] = (V1_Q8, int(rows), int(rank),
                                            data_start + off_a, nba * Q8_BYTES)
                self._plain[name + '.B'] = (V1_Q8, int(rank), int(cols),
                                            data_start + off_b, nbb * Q8_BYTES)
                self.entries[name] = (V1_SVD, (int(cols), int(rows)),
                                      (nba + nbb) * Q8_BYTES)
            elif nd >= 2:
                self._plain[name] = (V1_Q8, int(rows), int(cols),
                                     data_start + off_a, nba * Q8_BYTES)
                self.entries[name] = (V1_Q8, (int(cols), int(rows)), nba * Q8_BYTES)
            else:
                self._plain[name] = (V1_Q8, int(rows), 0,
                                     data_start + off_a, nba * Q8_BYTES)
                self.entries[name] = (V1_Q8, (int(rows),), nba * Q8_BYTES)

        # ── integrity: all tensor data must fit inside the file ──
        max_end = max(off + nb for _dt, _r, _c, off, nb in self._plain.values())
        if max_end > self._fsize:
            raise ValueError(
                f'file is truncated (conversion did not finish): tensor data '
                f'ends at {max_end >> 20} MB but only {self._fsize >> 20} MB '
                f'on disk.  Re-convert from the source GGUF.')

        # ── architecture ──
        arch = 'gemma4' if arch_id == 1 else 'llama'
        # derive the REAL head dims from the tensors (config head_dim is
        # unreliable in these files: q rows / n_heads is ground truth)
        head_dim = int(head_dim_cfg)
        n_heads = int(n_heads)
        n_kv = int(n_kv)
        q = self._svd.get('blk.0.attn_q.weight') or self._plain.get('blk.0.attn_q.weight')
        if q is not None:
            q_rows = q[1] if 'blk.0.attn_q.weight' in self._svd else q[1]
            if n_heads > 0 and q_rows % n_heads == 0:
                head_dim = q_rows // n_heads
        k = self._svd.get('blk.0.attn_k.weight') or self._plain.get('blk.0.attn_k.weight')
        if k is not None and head_dim > 0:
            k_rows = k[1]
            if k_rows % head_dim == 0:
                n_kv = k_rows // head_dim
        self.config = {
            'dim': int(dim), 'n_layers': int(n_layers),
            'n_heads': n_heads, 'n_kv_heads': n_kv,
            'ffn_dim': int(hidden), 'hidden_dim': int(hidden),
            'vocab_size': int(vocab), 'max_seq_len': int(seq),
            'rope_theta': float(rope) if rope > 0 else 10000.0,
            'head_dim': head_dim,
            'norm_eps': 1e-6 if arch.startswith('gemma') else 1e-5,
            'arch': arch,
        }

        # ── tokenizer JSON (same shape the v16 engine emits) ──
        t2i = {t: i for i, t in enumerate(self._tokens)}
        self.tokenizer_bytes = json.dumps({
            'tokens': self._tokens, 'merges': merges,
            'bos': t2i.get('<bos>', 2), 'eos': t2i.get('<eos>', 1),
            'pad': t2i.get('<pad>', 0), 'unk': t2i.get('<unk>', 3),
            'model': '',
        }).encode('utf-8')

        # Optional tokenizer sidecar: convert.c sometimes writes the wrong
        # tokenizer type (e.g. llama SPM for a gemma4 BPE source).  If a
        # .tokpatch file exists, replace the embedded tokenizer with it so
        # the engine uses the source's real tokenization.
        tokpatch = self.path + '.tokpatch'
        if os.path.exists(tokpatch):
            try:
                with open(tokpatch, 'r', encoding='utf-8') as f:
                    patch = json.load(f)
                self.tokenizer_bytes = json.dumps({
                    'tokens': patch.get('tokens', self._tokens),
                    'merges': patch.get('merges', merges),
                    'bos': patch.get('bos', 2),
                    'eos': patch.get('eos', 1),
                    'pad': patch.get('pad', 0),
                    'unk': patch.get('unk', 3),
                    'model': patch.get('model', ''),
                }).encode('utf-8')
            except Exception as e:
                print(f"[V1Source] warning: failed to load tokenizer sidecar: {e}")

        self._mmap = None
        self._fh = None
        self._patch_mm = None
        self._patch_fh = None
        self._load_f32_patch()

    # ── optional patch sidecar (<model>.ntmre.f32patch) ────────
    # The C converter damages two kinds of tensors:
    #   * 1D tensors (norm weights, rope_freqs, layer_output_scale):
    #     re-quantized to Q8 (36B/32).  With gemma4's extreme dynamic
    #     range (1e-4..193 in one 32-wide block) small weights quantize
    #     to 0 — gibberish-grade loss.
    #   * attention q/k/v/o matrices: stored as rank-1024 .ntmre compression factors
    #     (~86% correlation with the true weights) — also gibberish-grade.
    # A sidecar written by ntmre_f32_patch.py overrides those tensors
    # with exact data recovered from the source GGUF (F32 for small
    # tensors, the v1 Q8 dialect for matrices).  The .ntmre itself is
    # never modified.  Format (magic "RFP2"):
    #   [0:4] "RFP2"  [4:8] u32 count
    #   per entry: u32 name_len, name, u8 dtype (0=F32, 1=Q8 36B/32),
    #              u32 rows, u32 cols (0 for 1D), u64 nbytes, data
    def _load_f32_patch(self):
        patch_path = self.path + '.f32patch'
        if not os.path.exists(patch_path):
            return
        import mmap
        self._patch_fh = open(patch_path, 'rb')
        self._patch_mm = mmap.mmap(self._patch_fh.fileno(), 0,
                                   access=mmap.ACCESS_READ)
        mm = self._patch_mm
        if mm[:4] == b'RF32':
            raise ValueError('old RF32 sidecar format — rebuild it with '
                             'ntmre_f32_patch.py (format is now RFP2)')
        if mm[:4] != b'RFP2':
            raise ValueError('bad f32patch sidecar magic')
        self._in_patch = set()
        count = struct.unpack_from('<I', mm, 4)[0]
        o = 8
        patched = 0
        for _ in range(count):
            nl = struct.unpack_from('<I', mm, o)[0]
            name = mm[o + 4:o + 4 + nl].decode('utf-8')
            o += 4 + nl
            dtype = mm[o]
            rows, cols = struct.unpack_from('<II', mm, o + 1)
            nbytes = struct.unpack_from('<Q', mm, o + 9)[0]
            data_off = o + 17
            o = data_off + nbytes
            if name in self._svd:
                # replace the lossy .ntmre compression factors with a direct tensor
                rank, out, in_ = self._svd.pop(name)
                self._plain.pop(name + '.A', None)
                self._plain.pop(name + '.B', None)
            if name not in self._plain and name not in self.entries:
                continue
            n = rows * (cols if cols else 1)
            if dtype == 0 and nbytes != n * 4:
                raise ValueError(f'f32patch size mismatch for {name}')
            self._plain[name] = (V1_F32 if dtype == 0 else V1_Q8,
                                 int(rows), int(cols), data_off, int(nbytes))
            shape = (int(rows),) if cols == 0 else (int(cols), int(rows))
            self.entries[name] = (V1_F32 if dtype == 0 else V1_Q8,
                                  shape, int(nbytes))
            self._in_patch.add(name)
            patched += 1
        self._f32_patched = patched

    def _buf(self, name):
        return self._patch_mm if name in getattr(self, '_in_patch', ()) else self._mm()

    # ── io ────────────────────────────────────────────────────────
    def _mm(self):
        if self._mmap is None:
            import mmap
            self._fh = open(self.path, 'rb')
            self._mmap = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        return self._mmap

    def read_flat(self, name):
        """(flat f32 in storage order, index shape)."""
        if name in self._svd:
            rank, out, in_ = self._svd[name]
            _dt, _r, _c, off_a, nb_a = self._plain[name + '.A']
            _dt, _r, _c, off_b, nb_b = self._plain[name + '.B']
            mm = self._mm()
            A = _dequant_q8_36(mm[off_a:off_a + nb_a], out * rank).reshape(out, rank)
            B = _dequant_q8_36(mm[off_b:off_b + nb_b], rank * in_).reshape(rank, in_)
            return np.ascontiguousarray((A @ B).ravel(), dtype=np.float32), (in_, out)
        dtype, rows, cols, off, nbytes = self._plain[name]
        n = rows * (cols if cols else 1)
        if dtype == V1_F32:
            flat = np.frombuffer(self._buf(name)[off:off + nbytes],
                                 dtype='<f4', count=n).astype(np.float32)
        else:
            flat = _dequant_q8_36(self._buf(name)[off:off + nbytes], n)
        shape = (rows,) if cols == 0 else (cols, rows)
        return flat, shape

    def read_raw(self, name):
        """(dtype, raw bytes, shape) — 'svd' for factorized tensors."""
        if name in self._svd:
            rank, out, in_ = self._svd[name]
            return (V1_SVD, None, (in_, out))
        dtype, rows, cols, off, nbytes = self._plain[name]
        raw = bytes(self._buf(name)[off:off + nbytes])
        shape = (rows,) if cols == 0 else (cols, rows)
        return (dtype, raw, shape)

    def read_raw_svd(self, name):
        """((dtA, rawA), (dtB, rawB), (rank, out, in_))."""
        rank, out, in_ = self._svd[name]
        _dt, _r, _c, off_a, nb_a = self._plain[name + '.A']
        _dt, _r, _c, off_b, nb_b = self._plain[name + '.B']
        mm = self._mm()
        return ((V1_Q8, bytes(mm[off_a:off_a + nb_a])),
                (V1_Q8, bytes(mm[off_b:off_b + nb_b])),
                (rank, out, in_))

    def read_rows(self, name, ids, row_len):
        """Dequantize only the given rows of a 2D tensor."""
        if name in self._svd:
            flat, _ = self.read_flat(name)
            _rank, out, in_ = self._svd[name]
            w = flat.reshape(out, in_)
            return [w[int(t)].astype(np.float32) for t in ids]
        dtype, rows, cols, off, nbytes = self._plain[name]
        if dtype == V1_F32:
            buf = self._buf(name)
            return [np.frombuffer(buf[off + int(t) * row_len * 4:
                                      off + (int(t) + 1) * row_len * 4],
                                  dtype='<f4').astype(np.float32)
                    for t in ids]
        bpr = (row_len // Q8_BLOCK) * Q8_BYTES
        mm = self._buf(name)
        return [_dequant_q8_36(mm[off + int(t) * bpr: off + (int(t) + 1) * bpr],
                               row_len) for t in ids]

    def close(self):
        if self._mmap is not None:
            try:
                self._mmap.close()
                self._fh.close()
            except Exception:
                pass
            self._mmap = None
        if self._patch_mm is not None:
            try:
                self._patch_mm.close()
                self._patch_fh.close()
            except Exception:
                pass
            self._patch_mm = None
