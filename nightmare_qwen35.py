# -*- coding: utf-8 -*-
"""
nightmare_qwen35.py — Qwen3.5 hybrid engine for the Nightmare Suite
====================================================================
Runs Qwen3.5 / Qwen3-Next hybrid models (Gated DeltaNet SSM + Gated
Attention + partial RoPE) from NTMRE containers (NSVD / NTMR / RMTN
magics — same JSON-container layout), using the suite's own NSVDStream
reader (engine/bonsai_pc_app.py) which supports every NTMRE tensor type:
F32, Q8, .ntmre compression, .ntmre compression-REC, .ntmre compression-TERNARY, Q1_0, Q4_0, Q5_0, Q6_0,
Q4_K, Q5_K, Q6_K.

Math follows llama.cpp qwen35.cpp (same as the suite's bonsai_gpu.py):
  * SiLU after the SSM conv1d
  * SSM GQA head mapping by TILING (head v uses head v % nk)
  * decay applied to the SSM state BEFORE retrieval
  * FFN pre-norm is post_attention_norm (residual untouched)
  * gated full attention: Q tensor is [q|gate] per head (head-major)
  * partial RoPE on the first rope_dim dims, pairs (i, i+rd/2)

Weight layout (verified against the actual files):
  NTMRE tensor flat data is stored in ggml element order (input dim
  fastest) with index shape labels (in, out).  The logical PyTorch-style
  weight is therefore  W_pt = flat.reshape((out, in))  and a projection
  is  y = x @ W_pt.T.  .ntmre compression factors (P, Q) approximate the matrix the
  converter .ntmre compression'd whose flat bytes are the same ggml stream, so
  W_pt = (P @ Q).ravel().reshape((out, in)).  Synthetic test files
  written from numpy (out, in) arrays (e.g. tiny_bonsai) satisfy the
  same rule, so one rule covers every file.

Suite model interface:
    forward(token_id, position) -> logits (np.ndarray, vocab_size)
    reset()                      -> clear KV/SSM/conv state
    kv_cache.advance()           -> no-op (state managed internally)
    name / architecture / metadata / dim / n_layers / vocab_size /
    max_seq_len / loaded / tok

Author: Nightmare Designs 💀
"""
import json
import os
import struct
import nightmare_np as np
from pathlib import Path

from engine.bonsai_pc_app import (
    NSVDStream,
    TENS_F32, TENS_Q8, TENS_SVD_Q8, TENS_SVD_REC, TENS_SVD_TERNARY,
    TENS_Q1_0, TENS_Q4_0, TENS_Q5_0, TENS_Q6_0, TENS_Q4_K, TENS_Q5_K, TENS_Q6_K,
    _dequant_q8, _dequant_q1_0, _dequant_q4_0, _dequant_q5_0, _dequant_q6_0,
    _dequant_q4_k, _dequant_q5_k, _dequant_q6_k,
    _svd_unpack_factored, _svd_unpack_recursive, _svd_unpack_ternary,
)


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
    return np.log1p(np.exp(np.clip(x, -30, 30)))


class _KVAdvance:
    """The suite calls kv_cache.advance() after each token; this engine
    tracks its own recurrent/attention state, so this is a no-op."""
    def advance(self):
        pass


# ═══════════════════════════════════════════════════════════════════
# Tensor sources — NTMRE container or GGUF, same flat-stream interface
# ═══════════════════════════════════════════════════════════════════
class NSVDSource:
    """NTMRE JSON container (NSVD/NTMR/RMTN) via the suite's NSVDStream."""

    def __init__(self, path):
        self._s = NSVDStream(path)
        self.config = self._s.config
        self.tokenizer_bytes = self._s.tokenizer_bytes
        self.entries = self._s.entries

    def read_flat(self, name):
        """(flat f32 in storage order, index shape).  .ntmre compression types are
        materialized: (P @ Q).ravel() — the same flat byte stream the
        converter quantized."""
        ttype, shape, data_len = self._s.entries[name]
        off = self._s.offsets[name]
        data = bytes(self._s._mmap[off:off + data_len])
        n = int(np.prod(shape))
        if ttype == TENS_F32:
            flat = np.frombuffer(data, dtype=np.float32).astype(np.float32)
        elif ttype == TENS_Q8:
            flat = _dequant_q8(data, n)
        elif ttype == TENS_SVD_Q8:
            P, Q = _svd_unpack_factored(data, int(shape[0]), int(shape[1]))
            flat = (P @ Q).ravel()
        elif ttype == TENS_SVD_REC:
            P, Q = _svd_unpack_recursive(data, int(shape[0]), int(shape[1]))
            flat = (P @ Q).ravel()
        elif ttype == TENS_SVD_TERNARY:
            P, Q = _svd_unpack_ternary(data, int(shape[0]), int(shape[1]))
            flat = (P @ Q).ravel()
        elif ttype == TENS_Q1_0:
            flat = _dequant_q1_0(data, n)
        elif ttype == TENS_Q4_0:
            flat = _dequant_q4_0(data, n)
        elif ttype == TENS_Q5_0:
            flat = _dequant_q5_0(data, n)
        elif ttype == TENS_Q6_0:
            flat = _dequant_q6_0(data, n)
        elif ttype == TENS_Q4_K:
            flat = _dequant_q4_k(data, n)
        elif ttype == TENS_Q5_K:
            flat = _dequant_q5_k(data, n)
        elif ttype == TENS_Q6_K:
            flat = _dequant_q6_k(data, n)
        else:
            raise ValueError(f"Unknown tensor type {ttype} for {name}")
        return np.ascontiguousarray(flat, dtype=np.float32), shape

    def read_raw(self, name):
        """(ttype, raw bytes, index shape) straight from the container."""
        ttype, shape, data_len = self._s.entries[name]
        off = self._s.offsets[name]
        return ttype, bytes(self._s._mmap[off:off + data_len]), shape

    def read_rows(self, name, ids, row_len):
        """Dequantize only the given rows of a 2D tensor (rows are
        contiguous in the flat stream).  Falls back to a full read for
        types without block-aligned rows."""
        ttype, shape, data_len = self._s.entries[name]
        off = self._s.offsets[name]
        rows = []
        if ttype == TENS_F32:
            bpr = row_len * 4
            for t in ids:
                data = bytes(self._s._mmap[off + t * bpr: off + (t + 1) * bpr])
                rows.append(np.frombuffer(data, dtype=np.float32).astype(np.float32))
        elif ttype == TENS_Q8 and row_len % 32 == 0:
            bpr = (row_len // 32) * 34
            for t in ids:
                data = bytes(self._s._mmap[off + t * bpr: off + (t + 1) * bpr])
                rows.append(_dequant_q8(data, row_len))
        elif ttype == TENS_Q1_0 and row_len % 128 == 0:
            bpr = (row_len // 128) * 18
            for t in ids:
                data = bytes(self._s._mmap[off + t * bpr: off + (t + 1) * bpr])
                rows.append(_dequant_q1_0(data, row_len))
        else:
            flat, _ = self.read_flat(name)
            emb = flat.reshape((int(np.prod(shape)) // row_len, row_len))
            rows = [emb[t].astype(np.float32) for t in ids]
        return rows

    def close(self):
        self._s.close()


class GGUFSource:
    """GGUF file via the suite's gguf_parser (qwen35 hybrid GGUFs).
    load_tensor returns the logical (out, in) weight; its ravel IS the
    storage-order flat stream, so the engine's universal reshape rule
    applies unchanged."""

    def __init__(self, path):
        import gguf_parser
        self._g = gguf_parser.GGUFParser(path)
        self._m = self._g.parse()
        if not getattr(self._g, 'shape_reversed', True):
            raise ValueError('non-standard GGUF shape order — unsupported for qwen35')
        md = self._m.metadata
        self.config = {
            'dim': int(md.get('qwen35.embedding_length', 4096)),
            'n_layers': int(md.get('qwen35.block_count', 32)),
            'n_heads': int(md.get('qwen35.attention.head_count', 32)),
            'n_kv_heads': int(md.get('qwen35.attention.head_count_kv', 0)),
            'head_dim': int(md.get('qwen35.attention.key_length', 0)),
            'ffn_dim': int(md.get('qwen35.feed_forward_length', 0)),
            'vocab_size': int(md.get('tokenizer.ggml.vocab_size', 0)),
            'max_seq_len': int(md.get('qwen35.context_length', 4096)),
            'rope_theta': float(md.get('qwen35.rope.freq_base', 10000000.0)),
            'norm_eps': float(md.get('qwen35.attention.layer_norm_rms_epsilon', 1e-6)),
            'rope_dim': int(md.get('qwen35.rope.dimension_count', 64)),
            'full_attn_interval': int(md.get('qwen35.full_attention_interval', 4)),
            'ssm_conv_kernel': int(md.get('qwen35.ssm.conv_kernel', 4)),
            'ssm_state_size': int(md.get('qwen35.ssm.state_size', 128)),
            'ssm_n_group': int(md.get('qwen35.ssm.group_count', 16)),
            'ssm_dt_rank': int(md.get('qwen35.ssm.time_step_rank', 48)),
            'ssm_inner': int(md.get('qwen35.ssm.inner_size', 6144)),
            'arch': 'qwen35',
        }
        if not self.config['vocab_size']:
            self.config['vocab_size'] = int(self._m.vocab_size)
        # tokenizer JSON in the same shape the NTMRE containers carry
        self.tokenizer_bytes = json.dumps({
            'tokens': md.get('tokenizer.ggml.tokens', []),
            'merges': md.get('tokenizer.ggml.merges', []),
            'bos': md.get('tokenizer.ggml.bos_token_id', 1),
            'eos': md.get('tokenizer.ggml.eos_token_id', 2),
            'pad': md.get('tokenizer.ggml.padding_token_id', 0),
            'unk': md.get('tokenizer.ggml.unknown_token_id', 3),
            'model': md.get('tokenizer.ggml.model', ''),
        }).encode('utf-8')
        self.entries = {name: (info.dtype, tuple(info.shape), 0)
                        for name, info in self._m.tensors.items()}

    def read_flat(self, name):
        info = self._m.tensors[name]
        w = self._g.load_tensor(self._m, name)   # logical layout, f32
        return np.ascontiguousarray(w, dtype=np.float32).ravel(), tuple(info.shape)

    def read_raw(self, name):
        """(ggml dtype int, raw bytes, index shape) for GPU upload."""
        raw, dtype, shape, _ne = self._g.load_tensor_raw(self._m, name)
        return int(dtype), raw.tobytes(), tuple(shape)

    def read_rows(self, name, ids, row_len):
        if not hasattr(self, '_row_cache'):
            self._row_cache = {}
        out = []
        for t in ids:
            t = int(t)
            if t in self._row_cache:
                out.append(self._row_cache[t])
            else:
                try:
                    row = self._g.load_tensor_row(self._m, name, t).astype(np.float32)
                except Exception:
                    if getattr(self, '_rows_full', None) is None:
                        w = self._g.load_tensor(self._m, name)
                        self._rows_full = np.ascontiguousarray(w, dtype=np.float32)
                    row = self._rows_full[t]
                if len(self._row_cache) < 8192:
                    self._row_cache[t] = row
                out.append(row)
        return out

    def close(self):
        pass


def _open_source(path):
    with open(path, 'rb') as f:
        magic = f.read(8)
    if magic[:4] == b'GGUF':
        return GGUFSource(path)
    # NTMRE v16 binary (RMTN, version 16) — e.g. Qwen3.6-27B-NEO
    if magic[:4] == b'RMTN' and struct.unpack_from('<I', magic, 4)[0] == 16:
        from ntmre_v16_engine import V16Source
        return V16Source(path)
    return NSVDSource(path)


# ═══════════════════════════════════════════════════════════════════
# Qwen3.5 hybrid engine — Suite edition
# ═══════════════════════════════════════════════════════════════════
class Qwen35HybridEngine:
    def __init__(self, model_path):
        self._src = _open_source(model_path)
        self.config = self._src.config
        self._kv = {}
        self.ssm_state = {}
        self.kv_pos = 0
        self.kv_cache = _KVAdvance()

        self.dim = int(self.config.get('dim', 4096))
        self.n_layers = int(self.config.get('n_layers', 32))
        self.n_heads = int(self.config.get('n_heads', 32))
        self.n_kv_heads = int(self.config.get('n_kv_heads', self.n_heads))
        self.head_dim = int(self.config.get('head_dim', self.dim // self.n_heads))
        self.ffn_dim = int(self.config.get('ffn_dim', 4 * self.dim))
        self.vocab_size = int(self.config.get('vocab_size', 32000))
        self.rope_theta = float(self.config.get('rope_theta', 10000.0))
        self.norm_eps = float(self.config.get('norm_eps', 1e-5))

        # Qwen3.5 hybrid / SSM config
        self.full_attn_interval = int(self.config.get('full_attn_interval', 4))
        self.rope_dim = int(self.config.get('rope_dim', self.head_dim))
        self.ssm_conv_kernel = int(self.config.get('ssm_conv_kernel', 4))
        self.ssm_state_size = int(self.config.get('ssm_state_size', 128))
        self.ssm_n_group = int(self.config.get('ssm_n_group', 16))
        self.ssm_dt_rank = int(self.config.get('ssm_dt_rank', 48))
        self.ssm_inner = int(self.config.get('ssm_inner', 6144))
        self.ssm_head_v_dim = self.ssm_inner // self.ssm_dt_rank
        self.conv_dim = (self.ssm_state_size * self.ssm_n_group * 2
                         + self.ssm_head_v_dim * self.ssm_dt_rank)

        # Layer types detected from the actual tensors (robust vs the
        # interval formula): full-attention layers have attn_q.weight.
        self._is_attn = [f'blk.{i}.attn_q.weight' in self._src.entries
                         for i in range(self.n_layers)]

        # Fail fast on corrupt .ntmre compression conversions (insane singular values)
        # instead of generating garbage tokens at runtime.
        if isinstance(self._src, NSVDSource):
            for name, (ttype, shape, _dl) in self._src.entries.items():
                if ttype == TENS_SVD_Q8 and len(shape) >= 2:
                    self._src.read_flat(name)
                    break

        # Data-layout convention: real conversions keep the ggml flat
        # element order (labels read (dim, vocab) for the embedding);
        # numpy-written test files label the embedding (vocab, dim).
        eshape = self._src.entries.get('token_embd.weight', (None, (0, 0)))[1]
        self._ggml_layout = not (len(eshape) == 2 and eshape[0] == self.vocab_size)

        self.max_seq_len = int(self.config.get('max_seq_len', 2048))
        self.loaded = True
        self.no_bos = True

        # Weight cache policy: cache materialized weights only when the
        # whole model fits comfortably in RAM as f32.  1D vectors and the
        # embedding rows are always cheap enough to cache.
        total_elems = sum(int(np.prod(sh)) for _, sh, _ in self._src.entries.values())
        self._cache_weights = (total_elems * 4) < 2_000_000_000
        self._wcache = {}
        self._vcache = {}
        self._embcache = {}

        # GPU accelerator (NVRTC, suite's own) — weights stay quantized
        # on the GPU; per-token dequant + matvec on device.
        self._gpu = None
        self._gpu2 = None
        self._dual = False
        self._split_layer = self.n_layers
        self._layer_gpu = [1] * self.n_layers
        self._gpu_dispatch = {}
        if os.environ.get('NIGHTMARE_FORCE_CPU') != '1':
            try:
                from qwen35_gpu import GpuMM
                self._gpu = GpuMM.try_create()
                if self._gpu is not None:
                    print("[Qwen35] GPU acceleration ENABLED (NVRTC matvec)")
            except Exception as e:
                print(f"[Qwen35] GPU init failed ({e}) — CPU path")
                self._gpu = None

            # Second GPU (device 0) for dual-GPU weight residence
            if self._gpu is not None:
                try:
                    self._gpu2 = GpuMM.try_create(device=0)
                    if self._gpu2 is not None:
                        print("[Qwen35] Dual-GPU detected: GPU1 (primary) + GPU0 (secondary)")
                except Exception as e:
                    print(f"[Qwen35] GPU2 init failed ({e})")

        # Pre-upload all weights to GPU(s), filling GPU1 first then GPU0
        self._preupload_weights()

        # Suite-facing identity
        self._name = Path(model_path).stem
        self._metadata = self._build_metadata()
        self.tok = self._build_tokenizer()

    # ─── GPU dispatch ───
    def _target_gpu(self, name):
        """Return the GpuMM instance that should hold this weight."""
        if not self._dual or self._gpu2 is None:
            return self._gpu
        if name.startswith("blk."):
            try:
                layer = int(name.split(".")[1])
                if 0 <= layer < self.n_layers:
                    dev = self._layer_gpu[layer]
                    if dev == 1:
                        return self._gpu
                    elif dev == 0:
                        return self._gpu2 if self._gpu2 is not None else self._gpu
                    # dev == -1: streaming / unassigned, use primary
                    return self._gpu
            except (ValueError, IndexError):
                pass
        # Embedding stays on primary GPU
        if name.startswith(("token_embd", "token_emb", "tok_embeddings")):
            return self._gpu
        # Output projection on secondary GPU in dual mode (unless tied)
        if name == "output.weight":
            return self._gpu2 if self._gpu2 is not None else self._gpu
        return self._gpu

    def _estimate_weight_bytes(self, name):
        """Estimate raw byte size of a tensor for VRAM budgeting."""
        entry = self._src.entries.get(name)
        if entry is None:
            return 0
        dtype, shape, data_len = entry
        if data_len:
            return int(data_len)
        nelems = int(np.prod(shape))
        dtype_int = int(dtype) if isinstance(dtype, (int, np.integer)) else 0
        sizes = {
            0: 4, 1: 2, 2: 18 / 32, 6: 22 / 32, 8: 34 / 32,
            12: 144 / 256, 13: 176 / 256, 14: 210 / 256, 30: 2, 41: 18 / 128,
        }
        return int(nelems * sizes.get(dtype_int, 4))

    def _guess_weight_dims(self, name):
        """Infer (out_dim, in_dim) from tensor shape for a weight name."""
        entry = self._src.entries.get(name)
        if entry is None:
            raise KeyError(name)
        dtype, shape, _ = entry
        # For 1D tensors (biases/norms), upload as (n, 1)
        if len(shape) == 1:
            return int(shape[0]), 1
        # GGUF: shape in GGUF is (in_dim, out_dim), so out_dim=shape[1], in_dim=shape[0]
        if isinstance(self._src, GGUFSource):
            return int(shape[1]), int(shape[0])
        # NSVD / V16: shape labels are (in, out)
        return int(shape[1]), int(shape[0])

    def _preupload_weights(self):
        """Pre-upload all layer weights to GPU(s), filling GPU1 first, then GPU0."""
        if self._gpu is None:
            return

        # Refresh per-GPU budgets from current free VRAM
        self._gpu._refresh_budget(0.97, 128 * 1024 * 1024)
        if self._gpu2 is not None:
            self._gpu2._refresh_budget(0.97, 128 * 1024 * 1024)
            print(f"[Qwen35] GPU1 budget {self._gpu._budget / 1e9:.2f} GB, "
                  f"GPU0 budget {self._gpu2._budget / 1e9:.2f} GB")

        # Group weights by layer
        layer_weights = [[] for _ in range(self.n_layers)]
        other_weights = []
        for name in self._src.entries:
            if name.startswith("blk."):
                try:
                    layer = int(name.split(".")[1])
                    if 0 <= layer < self.n_layers:
                        layer_weights[layer].append(name)
                except (ValueError, IndexError):
                    other_weights.append(name)
            else:
                other_weights.append(name)

        # Estimate per-layer bytes
        layer_bytes = [sum(self._estimate_weight_bytes(n) for n in lw) for lw in layer_weights]

        # Assign layers greedily: fill GPU1 first, then GPU0
        gpu1_budget = self._gpu._budget
        gpu0_budget = self._gpu2._budget if self._gpu2 is not None else 0
        used1 = 0
        used0 = 0
        assigned = [1] * self.n_layers
        for i in range(self.n_layers):
            if used1 + layer_bytes[i] <= gpu1_budget:
                assigned[i] = 1
                used1 += layer_bytes[i]
            elif self._gpu2 is not None and used0 + layer_bytes[i] <= gpu0_budget:
                assigned[i] = 0
                used0 += layer_bytes[i]
            else:
                assigned[i] = -1  # leave for streaming

        self._layer_gpu = assigned
        if self._gpu2 is not None and any(g == 0 for g in assigned):
            self._dual = True
            self._split_layer = next((i for i, g in enumerate(assigned) if g == 0), self.n_layers)
            print(f"[Qwen35] Dual-GPU weight split: layers 0-{self._split_layer - 1} on GPU1, "
                  f"{self._split_layer}-{self.n_layers - 1} on GPU0")

        # Upload layer weights
        for i, lw in enumerate(layer_weights):
            if assigned[i] == -1:
                continue
            for name in lw:
                self._upload_to_gpu(name, self._target_gpu(name))

        # Upload other weights according to _target_gpu
        for name in other_weights:
            if self._gpu.has(name) or (self._gpu2 and self._gpu2.has(name)):
                continue
            self._upload_to_gpu(name, self._target_gpu(name))

    def _upload_to_gpu(self, name, gpu):
        """Upload a single weight to the specified GpuMM (best-effort)."""
        try:
            out_dim, in_dim = self._guess_weight_dims(name)
            self._gpu_prepare(name, out_dim, in_dim, gpu=gpu)
        except Exception as e:
            print(f"[Qwen35-GPU] pre-upload {name} failed ({e})")

    def _gpu_prepare(self, name, out_dim, in_dim, gpu=None):
        """Upload a weight's raw bytes to the chosen GPU once.  Returns the
        dispatch kind ('plain' / 'svd_factored' / 'svd_reinterp') or
        False when the tensor type has no GPU path."""
        from qwen35_gpu import (G_F32, G_F16, G_Q1_0, G_Q8_34, G_Q8_36,
                                G_Q4_0, G_Q5_0, G_Q6_0, G_Q4_K, G_Q5_K, G_Q6_K)
        if gpu is None:
            gpu = self._target_gpu(name)
        src = self._src
        if not hasattr(src, 'read_raw'):
            return False
        ttype, raw, shape = src.read_raw(name)
        if isinstance(src, NSVDSource):
            if ttype == TENS_F32:
                gpu.upload_raw(name, raw, G_F32, out_dim, in_dim)
                return ('plain',)
            if ttype == TENS_Q1_0:
                gpu.upload_raw(name, raw, G_Q1_0, out_dim, in_dim)
                return ('plain',)
            if ttype == TENS_Q8:
                gpu.upload_raw(name, raw, G_Q8_34, out_dim, in_dim)
                return ('plain',)
            _nsvd_ggml = {
                TENS_Q4_0: G_Q4_0, TENS_Q5_0: G_Q5_0, TENS_Q6_0: G_Q6_0,
                TENS_Q4_K: G_Q4_K, TENS_Q5_K: G_Q5_K, TENS_Q6_K: G_Q6_K,
            }
            if ttype in _nsvd_ggml:
                gpu.upload_raw(name, raw, _nsvd_ggml[ttype], out_dim, in_dim)
                return ('plain',)
            if ttype == TENS_SVD_Q8:
                m, n = int(shape[0]), int(shape[1])
                if (m, n) != (in_dim, out_dim):
                    return False   # numpy-layout .ntmre compression: CPU path
                gpu.upload_svd_nsvd(name, raw, m, n)
                return ('svd_reinterp',)
            return False
        # GGUFSource (ggml dtype ints)
        if isinstance(src, GGUFSource):
            if ttype == 0:    # F32
                gpu.upload_raw(name, raw, G_F32, out_dim, in_dim)
                return ('plain',)
            if ttype == 8:    # Q8_0: fp16 scale + 32 x int8 == G_Q8_34
                gpu.upload_raw(name, raw, G_Q8_34, out_dim, in_dim)
                return ('plain',)
            if ttype == 41:   # Q1_0 (suite's custom ggml type)
                gpu.upload_raw(name, raw, G_Q1_0, out_dim, in_dim)
                return ('plain',)
            _ggml_gpu = {
                1: G_F16, 2: G_Q4_0, 6: G_Q5_0,
                12: G_Q4_K, 13: G_Q5_K, 14: G_Q6_K,
            }
            if ttype in _ggml_gpu:
                gpu.upload_raw(name, raw, _ggml_gpu[ttype], out_dim, in_dim)
                return ('plain',)
            return False       # Q4_1/Q5_1/Q2_K/... stay on CPU for now
        # V16Source
        if ttype == 'svd':
            (a_dt, raw_a), (b_dt, raw_b), (rank, out, in_) = src.read_raw_svd(name)
            ga = G_Q8_36 if a_dt == 1 else G_F32
            gb = G_Q8_36 if b_dt == 1 else G_F32
            gpu.upload_raw(name + '.A', raw_a, ga, out, rank)
            gpu.upload_raw(name + '.B', raw_b, gb, rank, in_)
            return ('svd_factored',)
        if ttype == 0:
            gpu.upload_raw(name, raw, G_F32, out_dim, in_dim)
            return ('plain',)
        if ttype == 1:
            gpu.upload_raw(name, raw, G_Q8_36, out_dim, in_dim)
            return ('plain',)
        return False

    def _gpu_mm(self, name, x1, out_dim, in_dim):
        """GPU y = x @ W.T for one token, or None to fall back to CPU."""
        dispatch = self._gpu_dispatch.get(name)
        if dispatch is None:
            try:
                gpu = self._target_gpu(name)
                info = self._gpu_prepare(name, out_dim, in_dim, gpu=gpu)
            except Exception as e:
                print(f"[Qwen35-GPU] upload {name} failed ({e}) — CPU")
                info = False
                gpu = None
            self._gpu_dispatch[name] = (gpu, info)
        else:
            gpu, info = dispatch
        if info is False or gpu is None:
            return None
        try:
            kind = info[0]
            if kind == 'plain':
                raw = None
                if gpu._raw[name][0] == 'stream':
                    raw = self._src.read_raw(name)[1]
                return gpu.matvec(x1, name, raw=raw)
            if kind == 'svd_factored':
                ra = rb = None
                a_stream = gpu._raw[name + '.A'][0] == 'stream'
                b_stream = gpu._raw[name + '.B'][0] == 'stream'
                if a_stream or b_stream:
                    (_da, _ra), (_db, _rb), _dims = self._src.read_raw_svd(name)
                    if a_stream:
                        ra = _ra
                    if b_stream:
                        rb = _rb
                return gpu.matvec_svd(x1, name + '.A', name + '.B',
                                      raw_a=ra, raw_b=rb)
            if kind == 'svd_reinterp':
                blob = None
                if gpu._raw[name][0] == 'svd_stream':
                    blob = self._src.read_raw(name)[1]
                return gpu.matvec_svd_reinterpret(x1, name, out_dim, in_dim, blob=blob)
        except Exception as e:
            print(f"[Qwen35-GPU] {name} failed ({e}) — CPU fallback")
            self._gpu_dispatch[name] = (gpu, False)
        return None

    def _gpu_logits(self, x1):
        """Chunked GPU logits over the (vocab, dim) output matrix."""
        name = 'output.weight' if 'output.weight' in self._src.entries else 'token_embd.weight'
        dispatch = self._gpu_dispatch.get(name)
        if dispatch is None:
            try:
                gpu = self._target_gpu(name)
                info = self._gpu_prepare(name, self.vocab_size, self.dim, gpu=gpu)
            except Exception as e:
                print(f"[Qwen35-GPU] logits upload failed ({e}) — CPU")
                info = False
                gpu = None
            self._gpu_dispatch[name] = (gpu, info)
        else:
            gpu, info = dispatch
        if info is False or gpu is None or info[0] != 'plain':
            return None
        try:
            raw = None
            if gpu._raw[name][0] == 'stream':
                raw = self._src.read_raw(name)[1]
            return gpu.matvec_chunked(x1, name, raw=raw)
        except Exception as e:
            print(f"[Qwen35-GPU] logits failed ({e}) — CPU fallback")
            self._gpu_dispatch[name] = (gpu, False)
            return None

    # ─── Suite interface ───
    @property
    def name(self):
        return self._name

    @property
    def architecture(self):
        return 'qwen35'

    @property
    def metadata(self):
        return self._metadata

    def _build_metadata(self):
        """GGUF-style metadata so the suite's Tokenizer/chat templates work."""
        md = {
            'general.architecture': 'qwen35',
            'general.name': self._name,
            'qwen35.embedding_length': self.dim,
            'qwen35.block_count': self.n_layers,
            'qwen35.attention.head_count': self.n_heads,
            'qwen35.attention.head_count_kv': self.n_kv_heads,
            'qwen35.feed_forward_length': self.ffn_dim,
            'qwen35.context_length': self.max_seq_len,
            'qwen35.rope.freq_base': self.rope_theta,
            'qwen35.attention.layer_norm_rms_epsilon': self.norm_eps,
        }
        if self._src.tokenizer_bytes:
            try:
                d = json.loads(self._src.tokenizer_bytes.decode('utf-8'))
                md['tokenizer.ggml.tokens'] = d.get('tokens', [])
                md['tokenizer.ggml.merges'] = d.get('merges', [])
                md['tokenizer.ggml.bos_token_id'] = d.get('bos', 1)
                md['tokenizer.ggml.eos_token_id'] = d.get('eos', 2)
                md['tokenizer.ggml.padding_token_id'] = d.get('pad', 0)
                md['tokenizer.ggml.unknown_token_id'] = d.get('unk', 3)
                md['tokenizer.ggml.model'] = d.get('model', '')
            except Exception as e:
                print(f"[Qwen35] Tokenizer JSON parse failed: {e}")
        return md

    def _build_tokenizer(self):
        from tokenizer import Tokenizer
        return Tokenizer(self._metadata)

    def reset(self):
        """Clear all recurrent/attention state for a fresh message."""
        self._kv = {}
        self.ssm_state = {}
        self.kv_pos = 0

    def close(self):
        """Release GPU VRAM + all RAM caches when switching models."""
        self.loaded = False
        if getattr(self, '_gpu', None) is not None:
            try:
                self._gpu.close()
            except Exception:
                pass
            self._gpu = None
        if getattr(self, '_gpu2', None) is not None:
            try:
                self._gpu2.close()
            except Exception:
                pass
            self._gpu2 = None
        if getattr(self, '_src', None) is not None:
            try:
                self._src.close()
            except Exception:
                pass
            self._src = None
        self._cache_weights = False
        self._embcache = {}
        self._vcache = {}
        self._wcache = {}
        self._kv = {}
        self.ssm_state = {}
        self.tok = None
        self._metadata = None
        import gc
        gc.collect()

    # ─── tensor access ───
    def _flat(self, name):
        """Dequantized flat f32 elements in STORAGE order + index shape."""
        if getattr(self, '_src', None) is None:
            return np.zeros(1, dtype=np.float32), (1,)
        return self._src.read_flat(name)

    def _vec(self, name):
        """1D tensor (norms, ssm_a, dt bias) as f32 (always cached)."""
        if getattr(self, '_src', None) is None or name not in getattr(self._src, 'entries', {}):
            return None
        if self._vcache is not None and name in self._vcache:
            return self._vcache[name]
        flat, _ = self._flat(name)
        if self._vcache is not None:
            self._vcache[name] = flat
        return flat

    def _weight(self, name, out_dim, in_dim):
        """Logical PyTorch-style weight (out_dim, in_dim) f32.

        The flat stream is reshaped with the EXPECTED dims, which is
        correct for both ggml-ordered real conversions and numpy-written
        test files (see module docstring)."""
        key = (name, out_dim, in_dim)
        if self._wcache is not None and key in self._wcache:
            return self._wcache[key]
        if getattr(self, '_src', None) is None:
            return np.zeros((out_dim, in_dim), dtype=np.float32)
        flat, shape = self._flat(name)
        if int(np.prod(shape)) != out_dim * in_dim:
            raise ValueError(f"{name}: shape {shape} doesn't match expected ({out_dim}, {in_dim})")
        w = flat.reshape((out_dim, in_dim))
        if self._cache_weights and self._wcache is not None:
            self._wcache[key] = w
        return w

    def _mm(self, x, name, out_dim, in_dim):
        """y = x @ W.T with W the logical (out, in) weight."""
        if getattr(self, '_src', None) is None:
            return np.zeros((x.shape[0], out_dim), dtype=np.float32)
        if self._gpu is not None and x.shape[0] == 1:
            y = self._gpu_mm(name, x[0], out_dim, in_dim)
            if y is not None:
                return y[None, :]
        w = self._weight(name, out_dim, in_dim)
        return x @ w.T

    def _conv_weight(self, name):
        """SSM conv1d weight as (kernel, conv_dim)."""
        ck = self.ssm_conv_kernel
        cd = self.conv_dim
        key = (name, 'conv')
        if key in self._wcache:
            return self._wcache[key]
        flat, shape = self._flat(name)
        if self._ggml_layout:
            # ggml element order: (kernel, channel) with kernel fastest
            w = flat.reshape((cd, ck)).T.copy()
        else:
            w = flat.reshape((ck, cd))
        w = np.ascontiguousarray(w, dtype=np.float32)
        if self._cache_weights:
            self._wcache[key] = w
        return w

    def _embed(self, ids):
        """Embedding rows for token ids -> (seq, dim) f32.
        Embedding rows are contiguous in the flat stream for both layout
        conventions, so only the needed rows are dequantized."""
        ids = np.asarray(ids, dtype=np.int64)
        out = np.zeros((len(ids), self.dim), dtype=np.float32)
        missing = []
        for j, t in enumerate(ids):
            t = int(t)
            if t in self._embcache:
                out[j] = self._embcache[t]
            else:
                missing.append((j, t))
        if missing:
            rows = self._src.read_rows('token_embd.weight',
                                       [t for _, t in missing], self.dim)
            for (j, t), row in zip(missing, rows):
                self._embcache[t] = row
                out[j] = row
        return out

    # ─── partial RoPE (rotate first rope_dim of head_dim) ───
    def _rope_partial(self, x, position):
        """x: (seq, n_heads, head_dim) — rotate first rope_dim dims,
        pairs (i, i + rope_dim/2), matching llama.cpp MRoPE (text)."""
        seq = x.shape[0]
        rd = self.rope_dim
        half = rd // 2
        inv_freq = 1.0 / (self.rope_theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / rd))
        ang = (position + np.arange(seq, dtype=np.float64))[:, None] * inv_freq[None, :]
        cos = np.cos(ang).astype(np.float32)[:, None, :]
        sin = np.sin(ang).astype(np.float32)[:, None, :]
        out = x.copy()
        x1 = x[..., :half]
        x2 = x[..., half:rd]
        out[..., :half] = x1 * cos - x2 * sin
        out[..., half:rd] = x1 * sin + x2 * cos
        return out

    # ─── gated full attention layer ───
    def _full_attention(self, x, i, position):
        dim = self.dim
        nh = self.n_heads
        nkv = self.n_kv_heads
        hd = self.head_dim
        seq = x.shape[0]

        h = _rmsnorm(x, self._vec(f'blk.{i}.attn_norm.weight'), self.norm_eps)
        # Q tensor holds [query | gate] per head, head-major (llama.cpp)
        q_full = self._mm(h, f'blk.{i}.attn_q.weight', nh * hd * 2, dim)
        q_full = q_full.reshape(seq, nh, hd * 2)
        query = q_full[:, :, :hd]                     # (seq, nh, hd)
        gate = q_full[:, :, hd:]                      # (seq, nh, hd)
        key = self._mm(h, f'blk.{i}.attn_k.weight', nkv * hd, dim).reshape(seq, nkv, hd)
        value = self._mm(h, f'blk.{i}.attn_v.weight', nkv * hd, dim).reshape(seq, nkv, hd)

        qn = self._vec(f'blk.{i}.attn_q_norm.weight')
        kn = self._vec(f'blk.{i}.attn_k_norm.weight')
        if qn is not None:
            query = _rmsnorm(query, qn, self.norm_eps)
        if kn is not None:
            key = _rmsnorm(key, kn, self.norm_eps)

        query = self._rope_partial(query, position)
        key = self._rope_partial(key, position)

        # KV cache stored as (nkv, total, hd)
        k_new = key.transpose(1, 0, 2)
        v_new = value.transpose(1, 0, 2)
        if i not in self._kv:
            self._kv[i] = (k_new, v_new)
        else:
            kk, vv = self._kv[i]
            self._kv[i] = (np.concatenate([kk, k_new], axis=1),
                           np.concatenate([vv, v_new], axis=1))
        k_all, v_all = self._kv[i]
        rep = nh // nkv
        if rep > 1:
            # GQA: query head hh uses KV head hh // rep  == np.repeat
            k_all = np.repeat(k_all, rep, axis=0)
            v_all = np.repeat(v_all, rep, axis=0)
        q = query.transpose(1, 0, 2)                  # (nh, seq, hd)
        scores = (q @ k_all.transpose(0, 2, 1)) / np.sqrt(hd)
        total = k_all.shape[1]
        mask = np.triu(np.ones((seq, total), dtype=np.float32), k=total - seq + 1) * -1e9
        scores = scores + mask[None, :, :]
        attn = _softmax(scores, axis=-1)
        out = (attn @ v_all).transpose(1, 0, 2).reshape(seq, nh * hd)
        gate_flat = gate.reshape(seq, nh * hd)
        out = out * (1.0 / (1.0 + np.exp(-gate_flat)))
        return self._mm(out, f'blk.{i}.attn_output.weight', dim, nh * hd)

    # ─── Gated DeltaNet (SSM) layer ───
    def _ssm_layer(self, x, i, position):
        hk = self.ssm_state_size       # key head dim
        nk = self.ssm_n_group          # key/query heads
        nv = self.ssm_dt_rank          # value heads
        hv = self.ssm_head_v_dim       # value head dim
        ck = self.ssm_conv_kernel
        seq = x.shape[0]
        conv_dim = self.conv_dim

        h = _rmsnorm(x, self._vec(f'blk.{i}.attn_norm.weight'), self.norm_eps)
        qkv = self._mm(h, f'blk.{i}.attn_qkv.weight', conv_dim, self.dim)
        z = self._mm(h, f'blk.{i}.attn_gate.weight', hv * nv, self.dim)
        beta = self._mm(h, f'blk.{i}.ssm_beta.weight', nv, self.dim)
        alpha = self._mm(h, f'blk.{i}.ssm_alpha.weight', nv, self.dim)
        dt_bias = self._vec(f'blk.{i}.ssm_dt.bias')
        a = self._vec(f'blk.{i}.ssm_a')
        gate = _softplus(alpha + dt_bias) * a          # (seq, nv)
        decay = np.exp(np.clip(gate, -30.0, 0.0))
        beta_sig = 1.0 / (1.0 + np.exp(-beta))

        # short causal conv (SiLU afterwards), tap 0 = oldest
        conv_w = self._conv_weight(f'blk.{i}.ssm_conv1d.weight')   # (ck, conv_dim)
        if i in self.ssm_state:
            conv_mem, state = self.ssm_state[i]
        else:
            conv_mem = np.zeros((ck - 1, conv_dim), dtype=np.float32)
            state = np.zeros((hk, hv, nv), dtype=np.float32)
        conv_in = np.concatenate([conv_mem, qkv.astype(np.float32)], axis=0)
        conv_mem = conv_in[-(ck - 1):].copy()
        conv_out = np.zeros((seq, conv_dim), dtype=np.float32)
        for t in range(seq):
            for kk in range(ck):
                conv_out[t] += conv_w[kk] * conv_in[t + kk]
        conv_out = _silu(conv_out)

        rep = nv // nk
        core_all = np.zeros((seq, hv, nv), dtype=np.float32)
        for t in range(seq):
            # llama.cpp views conv out head-major: (head, head_dim)
            q = conv_out[t, :hk * nk].reshape(nk, hk)
            k = conv_out[t, hk * nk:2 * hk * nk].reshape(nk, hk)
            v = conv_out[t, 2 * hk * nk:].reshape(nv, hv)
            if rep > 1:
                # GQA: value head v uses key head v % nk -> TILE, not repeat
                q = np.tile(q, (rep, 1))
                k = np.tile(k, (rep, 1))
            q = q / (np.sqrt(np.sum(q * q, axis=1, keepdims=True)) + 1e-6)
            k = k / (np.sqrt(np.sum(k * k, axis=1, keepdims=True)) + 1e-6)
            q = q * (1.0 / np.sqrt(hv))

            # gated delta rule: decay first, then retrieve/update
            state = state * decay[t][None, None, :]
            kv_mem = np.einsum('khv,vk->hv', state, k)       # (hv, nv)
            delta = (v.T - kv_mem) * beta_sig[t][None, :]    # (hv, nv)
            state = state + np.einsum('vk,hv->khv', k, delta)
            core_all[t] = np.einsum('khv,vk->hv', state, q)  # (hv, nv)
        self.ssm_state[i] = (conv_mem, state)

        # per-head RMS norm over head_v_dim, then sigmoid gate z.
        # z is stored head_v_dim-fastest: z[v*hv + h]
        rms = np.sqrt(np.mean(core_all * core_all, axis=1, keepdims=True) + self.norm_eps)
        sn = self._vec(f'blk.{i}.ssm_norm.weight').astype(np.float32)   # (hv,)
        core_norm = (core_all / rms) * sn[None, :, None]
        zz = z.reshape(seq, nv, hv).transpose(0, 2, 1)                  # (seq, hv, nv)
        core_gated = core_norm * _silu(zz)
        # final output flattened head_v_dim-fastest to match ssm_out
        out = core_gated.transpose(0, 2, 1).reshape(seq, hv * nv)
        return self._mm(out, f'blk.{i}.ssm_out.weight', self.dim, hv * nv)

    # ─── forward (suite interface: single token or list) ───
    def forward(self, token_id, position=0):
        if not getattr(self, 'loaded', True) or getattr(self, '_src', None) is None:
            return None
        ids = np.asarray(token_id if isinstance(token_id, (list, tuple, np.ndarray)) else [token_id],
                         dtype=np.int64)
        seq = len(ids)
        dim = self.dim

        x = self._embed(ids)

        for i in range(self.n_layers):
            if self._is_attn[i]:
                out = self._full_attention(x, i, position)
            else:
                out = self._ssm_layer(x, i, position)
            x = x + out
            # FFN pre-norm is post_attention_norm (residual untouched)
            pan = self._vec(f'blk.{i}.post_attention_norm.weight')
            if pan is None:
                pan = self._vec(f'blk.{i}.ffn_norm.weight')
            h = _rmsnorm(x, pan, self.norm_eps)
            gate = _silu(self._mm(h, f'blk.{i}.ffn_gate.weight', self.ffn_dim, dim))
            up = self._mm(h, f'blk.{i}.ffn_up.weight', self.ffn_dim, dim)
            down = self._mm(gate * up, f'blk.{i}.ffn_down.weight', dim, self.ffn_dim)
            x = x + down

        x = _rmsnorm(x, self._vec('output_norm.weight'), self.norm_eps)
        if self._gpu is not None and x.shape[0] == 1:
            lg = self._gpu_logits(x[-1])
            if lg is not None:
                self.kv_pos = position + seq
                return lg
        if 'output.weight' in self._src.entries:
            w = self._weight('output.weight', self.vocab_size, dim)
            logits = x @ w.T
        else:
            # tied embeddings
            flat, _ = self._flat('token_embd.weight')
            emb = flat.reshape((self.vocab_size, dim))
            logits = x @ emb.T
        self.kv_pos = position + seq
        return logits[-1]

    def prefill_batch(self, token_ids, start_position=0):
        """Batched GPU prefill across all prompt tokens in one fused pass."""
        if not getattr(self, 'loaded', True) or getattr(self, '_src', None) is None or not token_ids:
            return None
        return self.forward(token_ids, position=start_position)

    def prefill(self, token_ids, start_position=0):
        return self.prefill_batch(token_ids, start_position)
