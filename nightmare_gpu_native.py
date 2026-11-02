"""
NIGHTMARE GPU-NATIVE ENGINE - Full VRAM Resident Inference
==========================================================
Loads ALL weights into VRAM and runs the entire forward pass on GPU.
No numpy in the hot path. No streaming. No PCIe per-token overhead.

This is the FAST path for models that fit in VRAM.
The streaming engine (nightmare_stream.py) handles models too big for VRAM.

Routing: ntmre_router picks this engine when the model's quantized
weight size fits in available VRAM (minus KV cache + buffers).

All GPU code is suite-internal (qwen35_gpu.GpuMM + NVRTC JIT kernels).
No third-party libraries. No numpy in production inference.

Author: Built for Levi 🩸
"""

import os
import sys
import time
import struct
import ctypes
import nightmare_np as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import gpu_info


# ── NTMRE container versions ───────────────────────────────────────────────
# Canonical/final version is 13 (see ntmre_format_py). These are the versions
# this build still READS; writers emit 13 only.
RMTN_V13 = 13
RMTN_READ_VERSIONS = (1, 2, 13, 16)




class GpuNativeEngine:
    """Full VRAM-resident GPU inference engine.

    All weights stay quantized on GPU (via GpuMM).  The forward pass
    runs entirely on GPU: matvec -> RoPE -> attention -> norm -> SiLU ->
    residual - all via NVRTC JIT kernels.  Only the final logits are
    downloaded to CPU.

    Falls back to GPUStreamEngine if the model doesn't fit in VRAM.
    """

    def __init__(self, model_path: str, max_seq_len: int = 4096):
        self.model_path = model_path
        self.max_seq_len = min(max_seq_len, 4096)  # attention kernel caps at 4096
        self.loaded = False

        # Config (filled by header load)
        self.dim = 0
        self.n_layers = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.hidden_dim = 0
        self.vocab_size = 0
        self.head_dim = 0
        self.rope_base = 10000.0
        self._norm_eps = 1e-5
        self._architecture = 'llama'
        self._use_layernorm = False

        # MoE
        self._n_experts = 0
        self._n_experts_used = 0

        # GPU accelerator
        self._gpu = None
        self._parser = None
        self._nsvd = None
        self.m = None

        # GPU buffers (allocated at load time)
        self._x_ptr = None         # hidden state (dim,)
        self._xn_ptr = None         # normed hidden state (dim,)
        self._q_ptr = None          # query (n_heads * head_dim,)
        self._k_ptr = None          # key (n_kv_heads * head_dim,)
        self._v_ptr = None          # value (n_kv_heads * head_dim,)
        self._attn_out_ptr = None   # attention output (n_heads * head_dim,)
        self._gate_ptr = None       # FFN gate (hidden_dim,)
        self._up_ptr = None         # FFN up (hidden_dim,)
        self._silu_ptr = None       # SiLU output (hidden_dim,)
        self._kv_k_ptrs = None      # per-layer K cache (list of GPU ptrs)
        self._kv_v_ptrs = None      # per-layer V cache (list of GPU ptrs)
        self._logits_ptr = None     # output logits (vocab_size,)

        # Embedding + output (kept on GPU via GpuMM)
        self._embedding_name = None
        self._output_name = None
        self._output_norm_ptr = None

        # Tokenizer (filled from model)
        self.tokens = []
        self.token_to_id = {}
        self.merges = []
        self._bos_id = 1
        self._eos_id = 2

        # Stats
        self._tokens_generated = 0
        self._total_time = 0.0

        # Dual-GPU support
        self._gpu2 = None           # second GpuMM (device 0, 8 GB card)
        self._dual = False          # True when weights are split across 2 GPUs
        self._split_layer = 0      # layers [0, split) on GPU1, [split, n) on GPU2
        self._bufs1 = None         # buffer dict on GPU1
        self._bufs2 = None         # buffer dict on GPU2
        self._cur_gpu = None       # active GpuMM for current layer
        self._cur_bufs = None      # active buffer dict for current layer
        self._logits2_ptr = None   # logits buffer on GPU2 (dual mode)
        self._output_norm2_ptr = None  # output norm on GPU2 (dual mode)
        self._kv_f16 = True        # F16 KV cache (half the VRAM)
        self._use_flash = False    # flash attention for >4096 context
        self._auto_quantize = False  # on-the-fly F16/BF16 -> Q8 for SafeTensors

        # Try to init primary GPU (device 1, 12 GB card)
        try:
            from qwen35_gpu import GpuMM
            self._gpu = GpuMM.try_create()
            if self._gpu is None:
                raise RuntimeError("GpuMM unavailable")
        except Exception as e:
            raise RuntimeError(f"GPU-Native requires CUDA: {e}")

        # Try to init secondary GPU (device 0, 8 GB card)
        try:
            from qwen35_gpu import GpuMM
            self._gpu2 = GpuMM.try_create(device=0)
            if self._gpu2 is not None:
                print("[GPU-NATIVE] Dual-GPU detected: GPU1 (primary) + GPU2 (device 0)")
        except Exception:
            self._gpu2 = None

        # Load model
        self._load_model(model_path)

    # ════════════════════════════════════════════════════════════════
    # MODEL LOADING
    # ════════════════════════════════════════════════════════════════

    def _load_model(self, path: str):
        ext = Path(path).suffix.lower()
        if ext == '.gguf':
            self._load_gguf(path)
        elif ext == '.ntmre':
            self._load_ntmre(path)
        elif ext == '.safetensors':
            self._load_safetensors(path)
        else:
            raise ValueError(f"Unsupported format: {ext}")

        # GPU-native only supports standard dense transformer families.
        # gemma4 needs dual-attention / shared-KV handling; qwen35 has its own
        # hybrid engine; clip is a vision encoder.  Bail out cleanly so the
        # router falls back to the streaming/specialized engine.
        # Known Gemma 4 variants must use the specialized gemma4 engine or
        # streaming; the generic dense GPU-native path cannot handle them.
        _gemma4_variants = ('gemma4', 'gemma4-assistant', 'gemma4_mtp',
                            'gemma4-assistant-mtp')
        _unsupported = ('qwen35', 'qwen3next', 'qwen3_next', 'clip')
        arch = str(self._architecture).lower()
        is_g4 = any(arch == base or arch.startswith(base + '-') or arch.startswith(base + '_')
                    for base in _gemma4_variants)
        if is_g4 or arch in _unsupported:
            raise RuntimeError(
                f"GPU-native engine does not support architecture '{self._architecture}' "
                f"- use the streaming engine instead")

        # Correct vocab_size from the actual output/embedding tensor shape -
        # many GGUFs (qwen2, etc.) omit the vocab metadata key, so the fallback
        # (32000) is wrong and the logits buffer would be the wrong size.
        if self.m is not None and self._parser is not None:
            for tname in ("output.weight", "token_embd.weight", "token_emb.weight"):
                t = self.m.tensors.get(tname)
                if t is not None and len(t.shape) == 2:
                    rev = getattr(self._parser, 'shape_reversed', True)
                    rows = int(t.shape[1] if rev else t.shape[0])
                    if rows > 0 and rows != self.vocab_size:
                        print(f"[GPU-NATIVE] vocab_size corrected from {tname}: "
                              f"{self.vocab_size} -> {rows}")
                        self.vocab_size = rows
                    break

        # Check VRAM budget
        vram_needed = self._estimate_vram()
        free_vram = self._get_free_vram()
        free_vram2 = self._get_free_vram2() if self._gpu2 is not None else 0
        if free_vram == 0:
            print("[GPU-NATIVE] WARNING: GPU1 reports 0 free VRAM - another app may be using it")
        if free_vram2 == 0 and self._gpu2 is not None:
            print("[GPU-NATIVE] WARNING: GPU0 reports 0 free VRAM - another app may be using it or context is bad")
        if free_vram2 > 0:
            print(f"[GPU-NATIVE] VRAM needed: {vram_needed / 1e9:.2f} GB, "
                  f"GPU1 free: {free_vram / 1e9:.2f} GB, "
                  f"GPU0 free: {free_vram2 / 1e9:.2f} GB, "
                  f"combined: {(free_vram + free_vram2) / 1e9:.2f} GB")
        else:
            print(f"[GPU-NATIVE] VRAM needed: {vram_needed / 1e9:.2f} GB, "
                  f"free: {free_vram / 1e9:.2f} GB")
        # Fit threshold: how much of reported free VRAM we are willing to
        # use before falling back to streaming.  Default 0.95; users with
        # clean systems can raise to 0.99 so models that "should fit" do.
        # Default fit threshold raised to 0.99 so models that "should fit"
        # actually route to the fast GPU-native path instead of streaming.
        _fit_frac = float(os.environ.get('NTMRE_FIT_FRACTION', '0.99'))
        if vram_needed > free_vram * _fit_frac:
            # Model doesn't fit - try auto-quantize (F16/BF16 -> Q8, ~2x smaller)
            is_st = isinstance(self._parser, type(None)) is False and\
                    self._parser is not None and\
                    hasattr(self._parser, '__class__') and\
                    self._parser.__class__.__name__ == 'SafeTensorsParser'
            has_f16 = any(int(t.dtype) in (1, 30) for t in self.m.tensors.values())\
                     if self.m is not None else False
            if is_st and has_f16:
                # Re-estimate with Q8 (roughly half the weight bytes)
                q8_vram = vram_needed * 0.55  # weights ~half, KV+bufs unchanged
                if q8_vram <= free_vram * _fit_frac:
                    self._auto_quantize = True
                    vram_needed = q8_vram
                    print(f"[GPU-NATIVE] Auto-quantize: F16/BF16 -> Q8 "
                          f"({vram_needed / 1e9:.1f} GB, fits in VRAM)")
                elif self._gpu2 is not None:
                    free_vram2 = self._get_free_vram2()
                    combined = (free_vram + free_vram2) * _fit_frac
                    if q8_vram <= combined:
                        self._auto_quantize = True
                        self._dual = True
                        # Split layers proportional to free VRAM so the 12 GB
                        # card carries more layers than the 8 GB card.
                        total_free = free_vram + free_vram2
                        self._split_layer = self.n_layers // 2
                        if total_free > 0:
                            self._split_layer = max(1, min(self.n_layers - 1,
                                                           int(self.n_layers * free_vram / total_free)))
                        vram_needed = q8_vram
                        print(f"[GPU-NATIVE] Auto-quantize + dual-GPU: "
                              f"{vram_needed / 1e9:.1f} GB across 2 GPUs")
                    else:
                        raise RuntimeError(
                            f"Model needs {q8_vram / 1e9:.1f} GB even at Q8, "
                            f"combined free {combined / 1e9:.1f} GB - "
                            f"use streaming engine instead")
            elif self._gpu2 is not None:
                free_vram2 = self._get_free_vram2()
                combined_budget = (free_vram + free_vram2) * _fit_frac
                if vram_needed <= combined_budget:
                    self._dual = True
                    # Split layers proportional to free VRAM so the 12 GB
                    # card carries more layers than the 8 GB card.
                    total_free = free_vram + free_vram2
                    self._split_layer = self.n_layers // 2
                    if total_free > 0:
                        self._split_layer = max(1, min(self.n_layers - 1,
                                                       int(self.n_layers * free_vram / total_free)))
                    print(f"[GPU-NATIVE] Dual-GPU split enabled: "
                          f"layers 0-{self._split_layer - 1} on GPU1, "
                          f"{self._split_layer}-{self.n_layers - 1} on GPU2")
                else:
                    raise RuntimeError(
                        f"Model needs {vram_needed / 1e9:.1f} GB VRAM, "
                        f"combined free {combined_budget / 1e9:.1f} GB - "
                        f"use streaming engine instead")
            else:
                raise RuntimeError(
                    f"Model needs {vram_needed / 1e9:.1f} GB VRAM but only "
                    f"{free_vram / 1e9:.1f} GB free - use streaming engine instead")

        # Allocate KV cache + forward-pass buffers FIRST so their memory
        # is guaranteed before GpuMM fills the remaining VRAM with weights.
        self._alloc_buffers()

        # Refresh resident weight budgets before upload so the GpuMMs use
        # the current free VRAM reported by nvidia-smi.  Allow user override
        # via env vars; default is aggressive (99.5% / 8 MB headroom).
        _frac = float(os.environ.get('NTMRE_RESIDENT_FRACTION', '0.995'))
        _head = int(os.environ.get('NTMRE_RESIDENT_HEADROOM_MB', '8')) * 1024 * 1024
        if hasattr(self._gpu, '_refresh_budget'):
            self._gpu._refresh_budget(_frac, _head)
        if self._gpu2 is not None and hasattr(self._gpu2, '_refresh_budget'):
            self._gpu2._refresh_budget(_frac, _head)

        # Upload all weights to GPU
        self._upload_weights()
        streamed = [k for k, v in self._gpu._raw.items() if v[0] == 'stream']
        if self._gpu2 is not None:
            streamed += [k for k, v in self._gpu2._raw.items() if v[0] == 'stream']
        if streamed:
            raise RuntimeError(f"VRAM ran out during upload ({len(streamed)} tensors could not fit) - use streaming engine instead")
        self.loaded = True
        mode = "dual-GPU" if self._dual else "single-GPU"
        print(f"[GPU-NATIVE] Ready ({mode}) - {self.n_layers} layers, "
              f"{self.dim}d, all weights on VRAM")

    def _load_gguf(self, path: str):
        import gguf_parser
        self._parser = gguf_parser.GGUFParser(path)
        self.m = self._parser.parse()
        meta = self.m.metadata
        arch = meta.get("general.architecture", "llama")

        for prefix in [arch, "llama", "qwen2", "qwen", "gemma", "gemma2", "gemma3", "mistral", "phi3", "phi", "deepseek", "deepseek2", "deepseek3", "starcoder", "starcoder2", "falcon", "baichuan", "internlm2", "minicpm", "smollm", "granite", "mpt", "olmo", "bloom"]:
            if f"{prefix}.embedding_length" in meta:
                arch = prefix
                break

        self._architecture = arch
        self.dim = meta.get(f"{arch}.embedding_length", 4096)
        self.n_layers = meta.get(f"{arch}.block_count", 32)
        self.n_heads = meta.get(f"{arch}.attention.head_count",
                                meta.get(f"{arch}.head_count", 32))
        self.n_kv_heads = meta.get(f"{arch}.attention.head_count_kv",
                                   meta.get(f"{arch}.head_count_kv", self.n_heads))
        # Some GGUFs (gemma4 variants) store head counts as lists.
        if isinstance(self.n_heads, (list, tuple)):
            self.n_heads = max(self.n_heads) if self.n_heads else 1
        if isinstance(self.n_kv_heads, (list, tuple)):
            self.n_kv_heads = max(self.n_kv_heads) if self.n_kv_heads else self.n_heads
        self.hidden_dim = meta.get(f"{arch}.feed_forward_length", 11008)
        # GGUF vocab key varies across writers: vocabulary_length (llama.cpp),
        # vocab_size (some Nightmare GGUFs), or tokenizer.ggml.vocab_size.
        self.vocab_size = meta.get(f"{arch}.vocabulary_length",
                                    meta.get(f"{arch}.vocab_size",
                                             meta.get("tokenizer.ggml.vocab_size", 32000)))
        # Gemma4 setup check, fixed 2026-10-28: never silently use the
        # 4096-dim / 32-layer Llama defaults for a Gemma4 model. If the
        # embedding length is missing but the layer count identifies a
        # verified variant, fill dim from that verified table and warn.
        # Otherwise fail loudly — do not synthesise it as E4B.
        if str(arch).lower().startswith("gemma"):
            _by_layers = {35: 1536, 42: 2560, 48: 3840, 30: 2816}
            if f"{arch}.embedding_length" not in meta and self.n_layers in _by_layers:
                print(f"[MODEL-SETUP] WARNING: {arch}.embedding_length missing; "
                      f"using verified dim {_by_layers[self.n_layers]} for "
                      f"{self.n_layers}-layer Gemma4 (not 4096 Llama default)")
                self.dim = _by_layers[self.n_layers]
            elif f"{arch}.embedding_length" not in meta or f"{arch}.block_count" not in meta:
                raise ValueError(
                    f"Unknown Gemma4 model setup: missing "
                    f"{[k for k in (f'{arch}.embedding_length', f'{arch}.block_count') if k not in meta]} "
                    f"— not synthesised as E4B. Add a verified spec or repair the model metadata.")

        self.max_seq_len = min(self.max_seq_len,
                               meta.get(f"{arch}.context_length", 2048))
        # Comprehensive RoPE frequency base lookup (supports LLaMA 3, 3.1, 3.2, Gemma, Qwen)
        rope_keys = [
            f"{arch}.rope.freq_base",
            f"{arch}.rope.frequency_base",
            f"{arch}.rope_theta",
            "llama.rope.freq_base",
            "llama.rope.frequency_base",
            "rope.freq_base",
            "general.rope.freq_base"
        ]
        rope_base_val = None
        for k in rope_keys:
            if k in meta:
                rope_base_val = meta[k]
                break
        if rope_base_val is not None:
            self.rope_base = float(rope_base_val)
        elif self.vocab_size == 128256 or self.vocab_size == 128000 or (isinstance(self.tokens, list) and len(self.tokens) >= 128000):
            self.rope_base = 500000.0
        else:
            self.rope_base = 10000.0

        self._norm_eps = float(meta.get(f"{arch}.attention.layer_norm_rms_epsilon", meta.get(f"{arch}.attention.layer_norm_epsilon", 1e-5)))
        self.head_dim = int(meta.get(f"{arch}.attention.key_length", 0)) or self.dim // self.n_heads

        # MoE
        self._n_experts = int(self.m._safe_int(meta.get(f"{arch}.expert_count", 0), 0))
        self._n_experts_used = int(self.m._safe_int(meta.get(f"{arch}.expert_used_count", 0), 0))
        if self._n_experts and not self._n_experts_used:
            self._n_experts_used = 2

        # LayerNorm detection
        _ln_archs = ('gpt2', 'gptj', 'gpt-neox', 'command-r', 'falcon', 'refact')
        if any(a in arch for a in _ln_archs):
            self._use_layernorm = True

        # Trust tensor shapes for head_dim
        if arch != 'gemma4':
            rev = getattr(self._parser, 'shape_reversed', True)
            tq = self.m.tensors.get('blk.0.attn_q.weight')
            tk = self.m.tensors.get('blk.0.attn_k.weight')
            if tq is not None and tk is not None and len(tk.shape) == 2:
                q_w = int(tq.shape[1] if rev else tq.shape[0])
                kv_w = int(tk.shape[1] if rev else tk.shape[0])
                if self.n_heads > 0 and q_w % self.n_heads == 0:
                    self.head_dim = q_w // self.n_heads
                if self.head_dim > 0 and kv_w % self.head_dim == 0:
                    self.n_kv_heads = kv_w // self.head_dim

        # Load tokenizer
        self._load_tokenizer_gguf()

    def _load_ntmre(self, path: str):
        with open(path, 'rb') as f:
            head = f.read(17)
        magic = head[:4]
        version = struct.unpack_from('<I', head, 4)[0] if len(head) >= 8 else 0
        is_json = (magic in (b'NSVD', b'NTMR', b'RMTN')
                   and len(head) == 17 and head[16:17] == b'{')

        if magic == b'RMTN' and version in RMTN_READ_VERSIONS:
            from ntmre_v16_engine import is_v16_file, V16Source
            if is_v16_file(path):
                self._nsvd = V16Source(path)
            else:
                from ntmre_v1_engine import V1Source
                self._nsvd = V1Source(path)
        elif is_json:
            from nightmare_qwen35 import NSVDSource
            self._nsvd = NSVDSource(path)
        else:
            raise ValueError("Unsupported NTMRE variant for GPU-native")

        cfg = self._nsvd.config
        arch = str(cfg.get('arch', cfg.get('general.architecture', 'llama'))).lower()
        self._architecture = arch
        self.dim = int(cfg.get('dim', cfg.get(f"{arch}.embedding_length", 4096)))
        self.n_layers = int(cfg.get('n_layers', cfg.get(f"{arch}.block_count", 32)))
        self.n_heads = int(cfg.get('n_heads', cfg.get(f"{arch}.attention.head_count", 32)))
        self.n_kv_heads = int(cfg.get('n_kv_heads', cfg.get(f"{arch}.attention.head_count_kv", self.n_heads)))
        self.hidden_dim = int(cfg.get('hidden_dim', cfg.get(f"{arch}.feed_forward_length", 11008)))
        self.vocab_size = int(cfg.get('vocab_size', 32000))
        self.max_seq_len = min(self.max_seq_len, int(cfg.get('max_seq_len', 2048)))
        self.rope_base = float(cfg.get('rope_theta', cfg.get('rope_base', 10000.0)))
        self._norm_eps = float(cfg.get('norm_eps', 1e-5))
        self.head_dim = int(cfg.get('head_dim', 0)) or self.dim // self.n_heads
        self._n_experts = int(cfg.get('n_experts', 0))
        self._n_experts_used = int(cfg.get('n_experts_used', 0))
        if self._n_experts and not self._n_experts_used:
            self._n_experts_used = 2
        if any(a in arch for a in ('gpt2', 'gptj', 'command-r', 'falcon', 'refact')):
            self._use_layernorm = True

        # Populate the built-in tokenizer from the container's tokenizer JSON
        # (mirrors GPUStreamEngine so encode()/decode_single() work for .ntmre)
        self._load_container_tokenizer()

    def _load_safetensors(self, path: str):
        """Load a HuggingFace SafeTensors model via safetensors_parser."""
        import safetensors_parser
        self._parser = safetensors_parser.SafeTensorsParser(path)
        self.m = self._parser.parse()
        meta = self.m.metadata
        arch = str(meta.get('general.architecture', 'llama')).lower()
        self._architecture = arch

        self.dim = int(meta.get(f'{arch}.embedding_length', 4096))
        self.n_layers = int(meta.get(f'{arch}.block_count', 32))
        self.n_heads = int(meta.get(f'{arch}.attention.head_count', 32))
        self.n_kv_heads = int(meta.get(f'{arch}.attention.head_count_kv', self.n_heads))
        self.hidden_dim = int(meta.get(f'{arch}.feed_forward_length', 11008))
        self.vocab_size = int(meta.get(f'{arch}.vocabulary_length', 32000))
        self.max_seq_len = min(self.max_seq_len, int(meta.get(f'{arch}.context_length', 2048)))
        rope_keys = [
            f"{arch}.rope.freq_base",
            f"{arch}.rope.frequency_base",
            f"{arch}.rope_theta",
            "llama.rope.freq_base",
            "llama.rope.frequency_base",
            "rope.freq_base",
            "general.rope.freq_base"
        ]
        rope_base_val = None
        for k in rope_keys:
            if k in meta:
                rope_base_val = meta[k]
                break
        if rope_base_val is not None:
            self.rope_base = float(rope_base_val)
        elif self.vocab_size == 128256 or self.vocab_size == 128000:
            self.rope_base = 500000.0
        else:
            self.rope_base = 10000.0

        self._norm_eps = float(meta.get(f'{arch}.attention.layer_norm_rms_epsilon', 1e-5))
        self.head_dim = int(meta.get(f'{arch}.attention.key_length', 0)) or self.dim // self.n_heads

        # MoE
        self._n_experts = int(self.m._safe_int(meta.get(f'{arch}.expert_count', 0), 0))
        self._n_experts_used = int(self.m._safe_int(meta.get(f'{arch}.expert_used_count', 0), 0))
        if self._n_experts and not self._n_experts_used:
            self._n_experts_used = 2

        # LayerNorm detection
        _ln_archs = ('gpt2', 'gptj', 'gpt-neox', 'command-r', 'falcon', 'refact')
        if any(a in arch for a in _ln_archs):
            self._use_layernorm = True

        # Trust tensor shapes for head_dim (SafeTensors shapes are NOT reversed)
        if arch != 'gemma4':
            rev = getattr(self._parser, 'shape_reversed', False)
            tq = self.m.tensors.get('blk.0.attn_q.weight')
            tk = self.m.tensors.get('blk.0.attn_k.weight')
            if tq is not None and tk is not None and len(tk.shape) == 2:
                q_w = int(tq.shape[1] if rev else tq.shape[0])
                kv_w = int(tk.shape[1] if rev else tk.shape[0])
                if self.n_heads > 0 and q_w % self.n_heads == 0:
                    self.head_dim = q_w // self.n_heads
                if self.head_dim > 0 and kv_w % self.head_dim == 0:
                    self.n_kv_heads = kv_w // self.head_dim

        # Load tokenizer (populated by safetensors_parser from tokenizer.json)
        self._load_tokenizer_gguf()

    def _load_tokenizer_gguf(self):
        meta = self.m.metadata
        tokens = meta.get("tokenizer.ggml.tokens", [])
        if isinstance(tokens, list) and tokens:
            self.tokens = [str(t) for t in tokens]
            self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
            self._bos_id = int(meta.get("tokenizer.ggml.bos_token_id", 1))
            self._eos_id = int(meta.get("tokenizer.ggml.eos_token_id", 2))
            merges = meta.get("tokenizer.ggml.merges", [])
            self.merges = []
            for m in merges:
                parts = m.split(' ') if isinstance(m, str) else list(m)
                if len(parts) == 2 and parts[0] in self.token_to_id and parts[1] in self.token_to_id:
                    self.merges.append((self.token_to_id[parts[0]], self.token_to_id[parts[1]]))

    def _load_container_tokenizer(self):
        """Load the built-in tokenizer from an NTMRE container's tokenizer JSON.
        Mirrors GPUStreamEngine so this engine can encode/decode on its own."""
        tb = getattr(getattr(self, '_nsvd', None), 'tokenizer_bytes', None)
        if not tb:
            return
        try:
            import json as _json
            d = _json.loads(tb.decode('utf-8'))
            self.tokens = [str(t) for t in d.get('tokens', [])]
            self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
            # _apply_bpe expects ID pairs; the JSON stores string pairs
            self.merges = []
            for m in d.get('merges', []):
                parts = m.split(' ') if isinstance(m, str) else list(m)
                if len(parts) == 2 and parts[0] in self.token_to_id and parts[1] in self.token_to_id:
                    self.merges.append((self.token_to_id[parts[0]], self.token_to_id[parts[1]]))
            self._bos_id = int(d.get('bos', 1))
            self._eos_id = int(d.get('eos', 2))
        except Exception as e:
            print(f"[GPU-NATIVE] container tokenizer parse failed: {e}")

    # ════════════════════════════════════════════════════════════════
    # VRAM ESTIMATION & WEIGHT UPLOAD
    # ════════════════════════════════════════════════════════════════

    def _estimate_vram(self):
        """Estimate total VRAM needed: weights + KV cache + buffers."""
        # Weight size: sum of all tensor raw bytes
        weight_bytes = 0
        if self._parser is not None and self.m is not None:
            for name, info in self.m.tensors.items():
                # GGUF TensorInfo has shape but no n_elements - compute it
                nelems = int(np.prod(info.shape)) if hasattr(info, 'shape') else 0
                weight_bytes += nelems * self._dtype_size(int(info.dtype))
        elif self._nsvd is not None:
            for name, entry in self._nsvd.entries.items():
                weight_bytes += int(entry[2])  # data_len
        # KV cache: n_layers * max_seq * n_kv_heads * head_dim * bytes * 2 (K+V)
        kv_elem_bytes = 2 if getattr(self, '_kv_f16', True) else 4  # F16 or F32
        kv_bytes = self.n_layers * self.max_seq_len * self.n_kv_heads * self.head_dim * kv_elem_bytes * 2
        # Buffers: x, xn, q, k, v, attn_out, gate, up, silu, logits
        buf_bytes = (self.dim * 4 * 4 +  # x, xn, attn_out(=n_heads*hd), output_norm
                     self.n_heads * self.head_dim * 4 +  # q
                     self.n_kv_heads * self.head_dim * 4 * 2 +  # k, v
                     self.hidden_dim * 4 * 3 +  # gate, up, silu
                     self.vocab_size * 4)  # logits
        return weight_bytes + kv_bytes + buf_bytes

    @staticmethod
    def _dtype_size(dtype):
        sizes = {0: 4, 1: 2, 2: 18/32, 3: 20/32, 6: 22/32, 7: 24/32,
                 8: 34/32, 9: 36/32, 10: 84/256, 11: 110/256, 12: 144/256,
                 13: 176/256, 14: 210/256, 30: 2}
        return sizes.get(dtype, 4)

    def _get_free_vram(self):
        """Query free/total VRAM on the GPU GpuMM is using."""
        try:
            free = gpu_info.get_free_vram(self._gpu.device)
            if free > 0:
                return free
            print("[GPU-NATIVE] nvidia-smi reported zero free VRAM for GPU1")
        except Exception as e:
            print(f"[GPU-NATIVE] VRAM query GPU1 exception: {e}")
        return 0

    def _get_free_vram2(self):
        """Query free/total VRAM on the secondary GPU (device 0)."""
        if self._gpu2 is None:
            return 0
        try:
            free = gpu_info.get_free_vram(self._gpu2.device)
            if free > 0:
                return free
            print("[GPU-NATIVE] nvidia-smi reported zero free VRAM for GPU0")
        except Exception as e:
            print(f"[GPU-NATIVE] VRAM query GPU0 exception: {e}")
        return 0

    def _target_gpu(self, name):
        """Return the GpuMM that should hold this tensor in dual-GPU mode."""
        if not self._dual:
            return self._gpu
        # Layer tensors: route by layer number
        if name.startswith("blk."):
            try:
                layer = int(name.split(".")[1])
                return self._gpu if layer < self._split_layer else self._gpu2
            except (ValueError, IndexError):
                return self._gpu
        # Embedding on GPU1, output + output_norm on GPU2
        if name.startswith(("token_embd", "token_emb", "tok_embeddings")):
            return self._gpu
        return self._gpu2

    def _upload_weights(self):
        """Upload all layer weights + embedding + output to GpuMM (GPU)."""
        from qwen35_gpu import (G_F32, G_F16, G_Q4_0, G_Q5_0, G_Q8_34,
                                G_Q4_K, G_Q5_K, G_Q6_K, G_BF16, G_Q1_0)

        _ggml_map = {0: G_F32, 1: G_F16, 2: G_Q4_0, 6: G_Q5_0, 8: G_Q8_34,
                     12: G_Q4_K, 13: G_Q5_K, 14: G_Q6_K, 30: G_BF16, 41: G_Q1_0}

        if self._parser is not None and self.m is not None:
            # GGUF: upload each tensor
            from safetensors_parser import quantize_to_q8_34
            for name, info in self.m.tensors.items():
                if "attn_qkv.weight" in name:
                    # Fused QKV (e.g. Phi-3): slice into Q, K, V and quantize to Q8_34
                    full_arr = self._parser.load_tensor(self.m, name).astype(np.float32)
                    q_dim = self.n_heads * self.head_dim
                    k_dim = self.n_kv_heads * self.head_dim
                    v_dim = self.n_kv_heads * self.head_dim
                    wq = np.ascontiguousarray(full_arr[:q_dim, :])
                    wk = np.ascontiguousarray(full_arr[q_dim:q_dim + k_dim, :])
                    wv = np.ascontiguousarray(full_arr[q_dim + k_dim:q_dim + k_dim + v_dim, :])
                    base_name = name.replace("attn_qkv.weight", "")
                    raw_q = quantize_to_q8_34(wq)
                    raw_k = quantize_to_q8_34(wk)
                    raw_v = quantize_to_q8_34(wv)
                    self._target_gpu(name).upload_raw(base_name + "attn_q.weight", raw_q, G_Q8_34, q_dim, self.dim)
                    self._target_gpu(name).upload_raw(base_name + "attn_k.weight", raw_k, G_Q8_34, k_dim, self.dim)
                    self._target_gpu(name).upload_raw(base_name + "attn_v.weight", raw_v, G_Q8_34, v_dim, self.dim)
                    continue

                if "ffn_up.weight" in name and (info.shape[1] if getattr(self._parser, 'shape_reversed', True) else info.shape[0]) == 2 * self.hidden_dim:
                    # Fused Gate/Up in ffn_up.weight (Phi-3 style)
                    full_arr = self._parser.load_tensor(self.m, name).astype(np.float32)
                    w_gate = np.ascontiguousarray(full_arr[:self.hidden_dim, :])
                    w_up = np.ascontiguousarray(full_arr[self.hidden_dim:2 * self.hidden_dim, :])
                    base_name = name.replace("ffn_up.weight", "")
                    raw_gate = quantize_to_q8_34(w_gate)
                    raw_up = quantize_to_q8_34(w_up)
                    self._target_gpu(name).upload_raw(base_name + "ffn_gate.weight", raw_gate, G_Q8_34, self.hidden_dim, self.dim)
                    self._target_gpu(name).upload_raw(base_name + "ffn_up.weight", raw_up, G_Q8_34, self.hidden_dim, self.dim)
                    continue

                if "ffn_gate_up.weight" in name:
                    # Fused Gate/Up (Qwen / other style)
                    full_arr = self._parser.load_tensor(self.m, name).astype(np.float32)
                    w_gate = np.ascontiguousarray(full_arr[:self.hidden_dim, :])
                    w_up = np.ascontiguousarray(full_arr[self.hidden_dim:2 * self.hidden_dim, :])
                    base_name = name.replace("ffn_gate_up.weight", "")
                    raw_gate = quantize_to_q8_34(w_gate)
                    raw_up = quantize_to_q8_34(w_up)
                    self._target_gpu(name).upload_raw(base_name + "ffn_gate.weight", raw_gate, G_Q8_34, self.hidden_dim, self.dim)
                    self._target_gpu(name).upload_raw(base_name + "ffn_up.weight", raw_up, G_Q8_34, self.hidden_dim, self.dim)
                    continue

                raw, _dt, _sh, _ne = self._parser.load_tensor_raw(self.m, name)
                raw = raw.tobytes()
                dtype = int(info.dtype)
                # Never fall back to G_F32 for an unmapped quant type: the
                # bytes are packed blocks, not floats, so the GPU would
                # reinterpret them as float32 and emit garbage logits with no
                # error at all.  Refuse instead, and let the router fall back
                # to an engine that can actually dequantize this type.
                gtag = _ggml_map.get(dtype)
                if gtag is None:
                    raise RuntimeError(
                        f"GPU-Native has no kernel for GGUF quant type "
                        f"{dtype} (tensor '{name}'). Supported: "
                        f"{sorted(_ggml_map)}. Refusing rather than "
                        f"misreading packed bytes as float32.")
                shape = info.shape
                if len(shape) == 2:
                    rev = getattr(self._parser, 'shape_reversed', True)
                    out_d = int(shape[1] if rev else shape[0])
                    in_d = int(shape[0] if rev else shape[1])
                elif len(shape) == 1:
                    out_d, in_d = int(shape[0]), 1
                elif len(shape) == 3:
                    # MoE expert weights: logical (n_experts, out, in).
                    # Slice into per-expert 2D weights so the existing
                    # fused matvec kernels can run them directly.
                    self._upload_moe_experts(name, gtag)
                    continue
                else:
                    out_d, in_d = int(shape[-1]), int(shape[-2] if len(shape) > 1 else 1)
                # On-the-fly quantize F16/BF16 -> Q8_34 (half the VRAM)
                if self._auto_quantize and dtype in (1, 30):
                    try:
                        from safetensors_parser import quantize_to_q8_34
                        f32_arr = self._parser.load_tensor(self.m, name).astype(np.float32)
                        raw = quantize_to_q8_34(f32_arr)
                        gtag = G_Q8_34
                    except Exception:
                        pass  # fallback to original dtype
                self._target_gpu(name).upload_raw(name, raw, gtag, out_d, in_d)
        elif self._nsvd is not None:
            # NTMRE: upload via read_raw
            from engine.bonsai_pc_app import (
                TENS_F32, TENS_Q8, TENS_SVD_Q8, TENS_Q1_0, TENS_Q4_0, TENS_Q5_0,
                TENS_Q6_0, TENS_Q4_K, TENS_Q5_K, TENS_Q6_K)
            _nsvd_map = {TENS_F32: G_F32, TENS_Q8: G_Q8_34, TENS_Q1_0: G_Q1_0,
                         TENS_Q4_0: G_Q4_0, TENS_Q5_0: G_Q5_0, TENS_Q6_0: G_Q6_K,
                         TENS_Q4_K: G_Q4_K, TENS_Q5_K: G_Q5_K, TENS_Q6_K: G_Q6_K}
            for name in self._nsvd.entries:
                try:
                    ttype, raw, shape = self._nsvd.read_raw(name)
                    if ttype is None or raw is None:
                        continue
                    # .ntmre compression-compressed tensors (TENS_.ntmre compression) are rank-reduced
                    # factor blobs, NOT flat quantized matrices.  The GPU-native
                    # engine only handles flat weights - uploading an .ntmre compression blob
                    # as a flat tensor makes the matvec read past the buffer and
                    # corrupt the CUDA context.  Bail out cleanly so the router
                    # falls back to the streaming engine (which supports .ntmre compression).
                    if ttype == TENS_SVD_Q8:
                        raise RuntimeError(
                            ".ntmre compression-compressed NTMRE weights are not supported "
                            "by the GPU-native engine - use the streaming engine")
                    gtag = _nsvd_map.get(ttype, G_F32)
                    # MoE expert weights: slice into per-expert 2D uploads
                    if '_exps' in name:
                        self._upload_moe_experts(name, gtag)
                        continue
                    if len(shape) >= 2:
                        out_d, in_d = int(shape[0]), int(shape[1])
                    else:
                        out_d, in_d = int(shape[0]) if shape else 1, 1
                    self._target_gpu(name).upload_raw(name, raw, gtag, out_d, in_d)
                except RuntimeError:
                    raise  # propagate .ntmre compression detection to the router fallback
                except Exception:
                    pass

        # Upload output norm as f32 (on GPU2 in dual mode, GPU1 otherwise)
        norm_gpu = self._target_gpu("output_norm.weight")
        if self._parser is not None and self.m is not None:
            try:
                on = self._parser.load_tensor(self.m, "output_norm.weight").astype(np.float32)
                self._output_norm_ptr = norm_gpu.to_gpu_ptr(on)
            except KeyError:
                self._output_norm_ptr = norm_gpu.to_gpu_ptr(np.ones(self.dim, dtype=np.float32))
        elif self._nsvd is not None:
            try:
                on, _ = self._nsvd.read_flat("output_norm.weight")
                self._output_norm_ptr = norm_gpu.to_gpu_ptr(on.astype(np.float32))
            except Exception:
                self._output_norm_ptr = norm_gpu.to_gpu_ptr(np.ones(self.dim, dtype=np.float32))
        # In dual mode, also keep a copy on GPU1 for single-GPU fallback paths
        if self._dual and norm_gpu is self._gpu2:
            self._output_norm2_ptr = self._output_norm_ptr  # already on GPU2
            # Also upload to GPU1 in case the output projection is shared
            try:
                if self._parser is not None and self.m is not None:
                    on = self._parser.load_tensor(self.m, "output_norm.weight").astype(np.float32)
                else:
                    on, _ = self._nsvd.read_flat("output_norm.weight")
                self._output_norm_ptr = self._gpu.to_gpu_ptr(on.astype(np.float32))
            except Exception:
                self._output_norm_ptr = self._gpu.to_gpu_ptr(np.ones(self.dim, dtype=np.float32))

    def _upload_moe_experts(self, name, gtag):
        """Slice a 3D MoE expert weight tensor into per-expert 2D weights
        and upload each to the target GPU.  Names follow the convention
        blk.L.exp.E.<kind> so _layer_forward_gpu can dispatch them.

        We load the full dequantized tensor once, slice it, and upload
        each expert separately.  This avoids having to reverse-engineer
        the raw byte layout per quantization type."""
        # Map the tensor name to a per-expert basename.
        # e.g. blk.0.ffn_gate_exps.weight -> blk.0.exp.{e}.gate
        base = name.replace('.weight', '')
        kind = None
        if '_gate_exps' in base:
            kind = 'gate'
        elif '_up_exps' in base:
            kind = 'up'
        elif '_down_exps' in base:
            kind = 'down'
        else:
            # Unknown 3D tensor - skip (router is 2D and handled above)
            return

        # Load full 3D tensor from GGUF or NTMRE source
        full = None
        if self._parser is not None and self.m is not None:
            try:
                full = self._parser.load_tensor(self.m, name).astype(np.float32)
            except Exception:
                pass
        elif self._nsvd is not None:
            try:
                flat, shape = self._nsvd.read_flat(name)
                full = flat.reshape(shape).astype(np.float32)
            except Exception:
                pass
        if full is None or full.ndim != 3:
            return
        n_exp, out_d, in_d = full.shape
        gpu = self._target_gpu(name)
        for e in range(n_exp):
            exp_arr = np.ascontiguousarray(full[e], dtype=np.float32)
            exp_name = base.replace(f'ffn_{kind}_exps', f'exp.{e}.{kind}')
            gpu.upload_f32(exp_name, exp_arr, out_d, in_d)

    def _alloc_buf_set(self, gpu):
        """Allocate a full set of forward-pass buffers on one GpuMM."""
        cu = gpu.cu
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        bufs = {
            'x': cu.alloc(self.dim * 4),
            'xn': cu.alloc(self.dim * 4),
            'q': cu.alloc(q_dim * 4),
            'k': cu.alloc(kv_dim * 4),
            'v': cu.alloc(kv_dim * 4),
            'attn_out': cu.alloc(q_dim * 4),
            'gate': cu.alloc(self.hidden_dim * 4),
            'up': cu.alloc(self.hidden_dim * 4),
            'silu': cu.alloc(self.hidden_dim * 4),
        }
        return bufs, q_dim, kv_dim

    def _alloc_buffers(self):
        """Allocate persistent GPU buffers for the forward pass.

        In dual-GPU mode, allocates a full buffer set on each GPU and
        splits the KV cache across devices."""
        self._bufs1, q_dim, kv_dim = self._alloc_buf_set(self._gpu)
        # Keep legacy attribute names as aliases into _bufs1 for compatibility
        self._x_ptr = self._bufs1['x']
        self._xn_ptr = self._bufs1['xn']
        self._q_ptr = self._bufs1['q']
        self._k_ptr = self._bufs1['k']
        self._v_ptr = self._bufs1['v']
        self._attn_out_ptr = self._bufs1['attn_out']
        self._gate_ptr = self._bufs1['gate']
        self._up_ptr = self._bufs1['up']
        self._silu_ptr = self._bufs1['silu']
        self._logits_ptr = self._gpu.cu.alloc(self.vocab_size * 4)

        if self._dual and self._gpu2 is not None:
            self._bufs2, _, _ = self._alloc_buf_set(self._gpu2)
            self._logits2_ptr = self._gpu2.cu.alloc(self.vocab_size * 4)
        else:
            self._bufs2 = None

        # KV cache: one GPU buffer per layer (split across GPUs in dual mode)
        # F16 KV cache = 2 bytes/element (was 4) - half the VRAM
        self._kv_k_ptrs = []
        self._kv_v_ptrs = []
        kv_stride = kv_dim
        kv_bytes = 2 if self._kv_f16 else 4  # F16 or F32
        for layer in range(self.n_layers):
            if self._dual and layer >= self._split_layer:
                self._kv_k_ptrs.append(self._gpu2.cu.alloc(self.max_seq_len * kv_stride * kv_bytes))
                self._kv_v_ptrs.append(self._gpu2.cu.alloc(self.max_seq_len * kv_stride * kv_bytes))
            else:
                self._kv_k_ptrs.append(self._gpu.cu.alloc(self.max_seq_len * kv_stride * kv_bytes))
                self._kv_v_ptrs.append(self._gpu.cu.alloc(self.max_seq_len * kv_stride * kv_bytes))
        self._kv_stride = kv_stride
        if self._kv_f16:
            print(f"[GPU-NATIVE] F16 KV cache: {(self.n_layers * self.max_seq_len * kv_stride * 2) / 1e9:.2f} GB (half of F32)")

    # ════════════════════════════════════════════════════════════════
    # FORWARD PASS - ALL ON GPU
    # ════════════════════════════════════════════════════════════════

    def _get_emb_table(self):
        if getattr(self, '_cached_emb_arr', None) is not None:
            return self._cached_emb_arr
        emb_name = self._get_emb_name()
        if not emb_name:
            return None
        if self._parser is not None and self.m is not None:
            try:
                emb_arr = self._parser.load_tensor(self.m, emb_name).astype(np.float32)
                if emb_arr.ndim == 2 and emb_arr.shape[1] != self.dim and emb_arr.shape[0] == self.dim:
                    emb_arr = np.ascontiguousarray(emb_arr.T)
                self._cached_emb_arr = emb_arr
            except Exception:
                self._cached_emb_arr = None
        elif self._nsvd is not None:
            try:
                flat, _ = self._nsvd.read_flat(emb_name)
                self._cached_emb_arr = flat.reshape((-1, self.dim)).astype(np.float32)
            except Exception:
                self._cached_emb_arr = None
        else:
            self._cached_emb_arr = None
        return self._cached_emb_arr

    def _get_emb_name(self):
        if getattr(self, '_emb_name_cached', None) is not None:
            return self._emb_name_cached
        gpu = self._gpu
        emb_name = None
        for n in ("token_embd.weight", "token_emb.weight", "tok_embeddings.weight"):
            if (gpu and gpu.has(n)) or (self.m and n in self.m.tensors) or (self._nsvd and n in getattr(self._nsvd, 'entries', {})):
                emb_name = n
                break
        self._emb_name_cached = emb_name
        return emb_name

    def _embedding_row(self, token_id):
        """Look up the embedding row for a token using lazy row-dequantization."""
        token_id = int(token_id)
        if not hasattr(self, '_emb_row_cache'):
            self._emb_row_cache = {}
        if token_id in self._emb_row_cache:
            row = self._emb_row_cache[token_id]
        else:
            emb_name = self._get_emb_name()
            if self._parser is not None and self.m is not None and emb_name:
                try:
                    row = self._parser.load_tensor_row(self.m, emb_name, token_id).astype(np.float32)
                except Exception:
                    emb_arr = self._get_emb_table()
                    row = emb_arr[token_id].astype(np.float32) if (emb_arr is not None and 0 <= token_id < len(emb_arr)) else np.zeros(self.dim, dtype=np.float32)
            elif self._nsvd is not None:
                emb_arr = self._get_emb_table()
                row = emb_arr[token_id].astype(np.float32) if (emb_arr is not None and 0 <= token_id < len(emb_arr)) else np.zeros(self.dim, dtype=np.float32)
            else:
                row = np.zeros(self.dim, dtype=np.float32)
            if len(self._emb_row_cache) < 4096:
                self._emb_row_cache[token_id] = row

        # Gemma scales embeddings by sqrt(dim)
        if str(self._architecture).startswith('gemma'):
            row = row * np.float32(np.sqrt(self.dim))
        return row

    def _embedding_rows(self, token_ids):
        """Look up embedding rows for multiple tokens (CPU-side)."""
        rows = np.empty((len(token_ids), self.dim), dtype=np.float32)
        for i, t in enumerate(token_ids):
            rows[i] = self._embedding_row(t)
        return np.ascontiguousarray(rows)

    @staticmethod
    def _ptr_off(ptr, offset):
        """Add a byte offset to a CUdeviceptr (c_ulonglong).  ctypes doesn't
        support c_ulonglong + int, so unwrap, add, and re-wrap."""
        return ctypes.c_ulonglong(ptr.value + int(offset))

    def forward(self, token_id: int, position: int, _skip_output: bool = False) -> np.ndarray:
        """Full GPU forward pass. Returns logits as numpy (vocab_size,).

        When _skip_output=True (prefill), the expensive output projection
        (vocab_size x dim matvec) is skipped - only the hidden state is
        updated.  The caller downloads logits only for the last token."""
        t0 = time.time()

        if self._dual:
            return self._forward_dual(token_id, position, t0, _skip_output)

        gpu = self._gpu
        bufs = self._bufs1
        self._cur_gpu = gpu
        self._cur_bufs = bufs

        # 1. Embedding lookup + upload to GPU
        gpu.cu._ensure_ctx()
        row = self._embedding_row(token_id)
        r = gpu.cu.cu.cuMemcpyHtoD_v2(bufs['x'], ctypes.c_void_p(row.ctypes.data), row.nbytes)
        if r != 0:
            raise RuntimeError(f"GPU-Native HtoD embedding failed: {r}")

        # 2. Stream through layers
        for layer in range(self.n_layers):
            self._layer_forward_gpu(layer, position)

        if _skip_output:
            self._tokens_generated += 1
            self._total_time += time.time() - t0
            return None

        # 3. Final norm + output projection
        if self._use_layernorm:
            gpu.layernorm_gpu(bufs['x'], self._output_norm_ptr, self.dim,
                              self._norm_eps, bufs['xn'])
        else:
            gpu.rmsnorm_gpu(bufs['x'], self._output_norm_ptr, self.dim,
                            self._norm_eps, bufs['xn'])

        out_name = None
        for n in ("output.weight", "token_embd.weight", "token_emb.weight"):
            if gpu.has(n):
                out_name = n
                break
        gpu.matvec_gpu(bufs['xn'], out_name, self._logits_ptr)

        # 4. Download logits
        logits = gpu.from_gpu_ptr(self._logits_ptr, self.vocab_size)
        if str(self._architecture) == 'gemma4':
            logits = np.tanh(logits / 30.0) * 30.0

        self._tokens_generated += 1
        self._total_time += time.time() - t0
        return logits

    def _forward_dual(self, token_id: int, position: int, t0: float, _skip_output: bool = False) -> np.ndarray:
        """Dual-GPU forward: first half layers on GPU1, second half on GPU2.

        The hidden state is transferred once at the split point (a single
        dimx4-byte PCIe copy).  KV cache is split across devices."""
        gpu1 = self._gpu
        gpu2 = self._gpu2
        bufs1 = self._bufs1
        bufs2 = self._bufs2

        # 1. Embedding on GPU1
        row = self._embedding_row(token_id)
        gpu1.cu.cu.cuMemcpyHtoD_v2(bufs1['x'], ctypes.c_void_p(row.ctypes.data), row.nbytes)

        # 2a. First half layers on GPU1
        self._cur_gpu = gpu1
        self._cur_bufs = bufs1
        for layer in range(self._split_layer):
            self._layer_forward_gpu(layer, position)

        # 2b. Transfer hidden state GPU1 -> GPU2 (one PCIe round-trip)
        x_host = gpu1.from_gpu_ptr(bufs1['x'], self.dim)
        gpu2.cu._ensure_ctx()
        r = gpu2.cu.cu.cuMemcpyHtoD_v2(bufs2['x'], ctypes.c_void_p(x_host.ctypes.data),
                               x_host.nbytes)
        if r != 0:
            raise RuntimeError(f"GPU-Native dual HtoD transfer failed: {r}")

        # 2c. Second half layers on GPU2
        self._cur_gpu = gpu2
        self._cur_bufs = bufs2
        for layer in range(self._split_layer, self.n_layers):
            self._layer_forward_gpu(layer, position)

        if _skip_output:
            self._tokens_generated += 1
            self._total_time += time.time() - t0
            return None

        # 3. Final norm + output projection on GPU2
        norm_ptr = self._output_norm2_ptr or self._output_norm_ptr
        if self._use_layernorm:
            gpu2.layernorm_gpu(bufs2['x'], norm_ptr, self.dim,
                              self._norm_eps, bufs2['xn'])
        else:
            gpu2.rmsnorm_gpu(bufs2['x'], norm_ptr, self.dim,
                            self._norm_eps, bufs2['xn'])

        out_name = None
        for n in ("output.weight", "token_embd.weight", "token_emb.weight"):
            if gpu2.has(n):
                out_name = n
                break
        if out_name is None:
            # output.weight might be on GPU1 (tied embeddings) - transfer xn
            xn_host = gpu2.from_gpu_ptr(bufs2['xn'], self.dim)
            gpu1.cu._ensure_ctx()
            r = gpu1.cu.cu.cuMemcpyHtoD_v2(bufs1['xn'], ctypes.c_void_p(xn_host.ctypes.data),
                                   xn_host.nbytes)
            if r != 0:
                raise RuntimeError(f"GPU-Native dual output HtoD transfer failed: {r}")
            for n in ("output.weight", "token_embd.weight", "token_emb.weight"):
                if gpu1.has(n):
                    out_name = n
                    break
            gpu1.matvec_gpu(bufs1['xn'], out_name, self._logits_ptr)
            logits = gpu1.from_gpu_ptr(self._logits_ptr, self.vocab_size)
        else:
            gpu2.matvec_gpu(bufs2['xn'], out_name, self._logits2_ptr)
            logits = gpu2.from_gpu_ptr(self._logits2_ptr, self.vocab_size)

        if str(self._architecture) == 'gemma4':
            logits = np.tanh(logits / 30.0) * 30.0

        self._tokens_generated += 1
        self._total_time += time.time() - t0
        return logits

    def speculative_generate(self, draft_model, token_ids, n_tokens=64,
                                start_pos=0, temp=0.0, top_k=0):
        """Speculative decoding: draft model proposes tokens, main model verifies.

        The draft model (e.g., 90M) generates candidate tokens one at a time.
        The main model verifies each via forward().  When the draft is correct,
        we accept the token without an extra main-model forward for the next
        position (the main model's logits already contain the prediction).

        Returns list of generated token ids."""
        # Prefill both models on the prompt
        self.prefill(token_ids, start_pos)
        try:
            draft_model.prefill(token_ids, start_pos)
        except Exception:
            pass  # draft may not have prefill

        pos = start_pos + len(token_ids)
        generated = []
        last_token = token_ids[-1] if token_ids else self._bos_id

        for _ in range(n_tokens):
            # Draft model proposes next token
            try:
                draft_logits = draft_model.forward(last_token, pos - 1)
                draft_token = int(np.argmax(draft_logits))
            except Exception:
                draft_token = -1

            # Main model verifies (produces logits for the current position)
            main_logits = self.forward(last_token, pos - 1)
            main_token = int(np.argmax(main_logits))

            if draft_token == main_token and draft_token != -1:
                # Draft was right - accept (main model already has the state)
                generated.append(main_token)
                last_token = main_token
                pos += 1
            else:
                # Draft wrong (or unavailable) - use main model's token
                generated.append(main_token)
                last_token = main_token
                pos += 1

            if main_token == self._eos_id:
                break

        return generated

    def prefill(self, token_ids: list, start_pos: int = 0) -> np.ndarray:
        """Process a prompt batch efficiently.

        Uses true batched GPU prefill when available (all tokens processed
        in parallel via batch_matvec + batched causal attention). Falls back
        to per-token forward for dual-GPU or LayerNorm models.
        Returns logits for the last token only."""
        if not token_ids:
            return None
        if len(token_ids) == 1:
            return self.forward(token_ids[0], start_pos)
        # Use batched prefill for single-GPU RMSNorm dense models.
        # MoE models fall back to per-token because each token may route
        # to different experts, making batched expert dispatch complex.
        if not self._dual and not self._use_layernorm and self._n_experts == 0:
            try:
                return self._prefill_batch_gpu(token_ids, start_pos)
            except Exception as e:
                print(f"[GPU-NATIVE] Batched prefill failed ({e}), falling back to per-token")
        # Fallback: per-token forward (skip output for all but last)
        for i in range(len(token_ids) - 1):
            self.forward(token_ids[i], start_pos + i, _skip_output=True)
        return self.forward(token_ids[-1], start_pos + len(token_ids) - 1)

    def prefill_batch(self, token_ids: list, start_pos: int = 0) -> np.ndarray:
        """Alias for prefill — full GPU VRAM batch processing."""
        return self.prefill(token_ids, start_pos)

    def _alloc_batch_bufs(self, batch):
        """Allocate (or grow) batch-sized GPU buffers for prefill."""
        gpu = self._gpu
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        old = getattr(self, '_batch_bufs', None)
        old_batch = getattr(self, '_batch_n', 0)
        if old is not None and old_batch >= batch:
            return old
        # Free old buffers if growing
        if old is not None:
            for ptr in old.values():
                try: gpu.cu.free(ptr)
                except: pass
        bufs = {
            'x': gpu.cu.alloc(batch * self.dim * 4),
            'xn': gpu.cu.alloc(batch * self.dim * 4),
            'q': gpu.cu.alloc(batch * q_dim * 4),
            'k': gpu.cu.alloc(batch * kv_dim * 4),
            'v': gpu.cu.alloc(batch * kv_dim * 4),
            'attn_out': gpu.cu.alloc(batch * q_dim * 4),
            'gate': gpu.cu.alloc(batch * self.hidden_dim * 4),
            'up': gpu.cu.alloc(batch * self.hidden_dim * 4),
            'silu': gpu.cu.alloc(batch * self.hidden_dim * 4),
        }
        self._batch_bufs = bufs
        self._batch_n = batch
        return bufs

    def _prefill_batch_gpu(self, token_ids: list, start_pos: int = 0) -> np.ndarray:
        """True batched GPU prefill - all prompt tokens in parallel.

        Uses batch_matvec for all weight projections and batched causal
        attention for the attention layer. Only the last token gets the
        output projection (vocab x dim matvec)."""
        t0 = time.time()
        gpu = self._gpu
        batch = len(token_ids)
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        kv_elems = self.n_kv_heads * self.head_dim
        kv_bytes = 2 if self._kv_f16 else 4

        bufs = self._alloc_batch_bufs(batch)

        # 1. Embedding lookup + upload batch to GPU
        gpu.cu._ensure_ctx()
        rows = self._embedding_rows(token_ids)
        r = gpu.cu.cu.cuMemcpyHtoD_v2(bufs['x'], ctypes.c_void_p(rows.ctypes.data), rows.nbytes)
        if r != 0:
            raise RuntimeError(f"GPU-Native batch prefill HtoD failed: {r}")

        # 2. Stream through layers (batched)
        for layer in range(self.n_layers):
            pfx = f"blk.{layer}."

            # --- Attention norm (batched RMSNorm) ---
            norm_name = pfx + "attn_norm.weight"
            if gpu.has(norm_name):
                w_ptr = self._get_weight_ptr(norm_name, gpu)
                gpu.batch_rmsnorm_gpu(bufs['x'], w_ptr, self.dim, batch,
                                      self._norm_eps, bufs['xn'])
            else:
                gpu.cu.cu.cuMemcpyDtoD_v2(bufs['xn'], bufs['x'], batch * self.dim * 4)

            # --- QKV projections (batched matmul - fused GPU kernels) ---
            wq_name = pfx + "attn_q.weight"
            wk_name = pfx + "attn_k.weight"
            wv_name = pfx + "attn_v.weight"
            if gpu.has(wq_name) and gpu.has(wk_name) and gpu.has(wv_name):
                gpu.matmul_batch_gpu(bufs['xn'], wq_name, bufs['q'], batch)
                gpu.matmul_batch_gpu(bufs['xn'], wk_name, bufs['k'], batch)
                gpu.matmul_batch_gpu(bufs['xn'], wv_name, bufs['v'], batch)
            else:
                raise RuntimeError(f"Missing QKV weights for layer {layer}")

            # --- QKV biases (per-token add; Qwen2/Qwen2.5 family) ---
            for bname, bbuf, bdim in ((pfx + 'attn_q.bias', 'q', q_dim),
                                      (pfx + 'attn_k.bias', 'k', kv_dim),
                                      (pfx + 'attn_v.bias', 'v', kv_dim)):
                if gpu.has(bname):
                    bptr = self._get_weight_ptr(bname, gpu)
                    if bptr is not None:
                        for b in range(batch):
                            row_off = ctypes.c_ulonglong(bufs[bbuf].value + b * bdim * 4)
                            gpu.add_gpu(row_off, bptr, bdim)

            # --- Q/K normalization if present ---
            qn_name = pfx + "attn_q_norm.weight"
            if gpu.has(qn_name):
                w_qn = self._get_weight_ptr(qn_name, gpu)
                gpu.batch_rmsnorm_gpu(bufs['q'], w_qn, q_dim, batch, self._norm_eps, bufs['q'])
            kn_name = pfx + "attn_k_norm.weight"
            if gpu.has(kn_name):
                w_kn = self._get_weight_ptr(kn_name, gpu)
                gpu.batch_rmsnorm_gpu(bufs['k'], w_kn, kv_dim, batch, self._norm_eps, bufs['k'])

            # --- Batched RoPE (each token at position start_pos + b) ---
            gpu.batch_rope_gpu(bufs['q'], bufs['k'], self.n_heads, self.n_kv_heads,
                              self.head_dim, start_pos, batch, self.rope_base)

            # --- Store K, V into KV cache (per-token copy) ---
            for b in range(batch):
                pos = start_pos + b
                if self._kv_f16:
                    kv_off = pos * self._kv_stride * 2
                    # Offset into batch buffer for this token's K
                    k_off = ctypes.c_ulonglong(bufs['k'].value + b * kv_dim * 4)
                    v_off = ctypes.c_ulonglong(bufs['v'].value + b * kv_dim * 4)
                    gpu.store_f16_gpu(k_off, self._ptr_off(self._kv_k_ptrs[layer], kv_off), kv_elems)
                    gpu.store_f16_gpu(v_off, self._ptr_off(self._kv_v_ptrs[layer], kv_off), kv_elems)
                else:
                    kv_off = pos * self._kv_stride * 4
                    kv_sz = kv_elems * 4
                    k_off = ctypes.c_ulonglong(bufs['k'].value + b * kv_dim * 4)
                    v_off = ctypes.c_ulonglong(bufs['v'].value + b * kv_dim * 4)
                    gpu.cu.cu.cuMemcpyDtoD_v2(self._ptr_off(self._kv_k_ptrs[layer], kv_off), k_off, kv_sz)
                    gpu.cu.cu.cuMemcpyDtoD_v2(self._ptr_off(self._kv_v_ptrs[layer], kv_off), v_off, kv_sz)

            # --- Batched causal attention ---
            if self._kv_f16:
                gpu.batch_attention_f16_gpu(bufs['q'], self._kv_k_ptrs[layer],
                                           self._kv_v_ptrs[layer], bufs['attn_out'],
                                           self.n_heads, self.n_kv_heads,
                                           self.head_dim, batch, self._kv_stride)
            else:
                gpu.batch_attention_gpu(bufs['q'], self._kv_k_ptrs[layer],
                                        self._kv_v_ptrs[layer], bufs['attn_out'],
                                        self.n_heads, self.n_kv_heads,
                                        self.head_dim, batch, self._kv_stride)

            # --- Post-attention norm if present ---
            post_attn_name = pfx + "post_attention_norm.weight"
            if gpu.has(post_attn_name):
                w_pan = self._get_weight_ptr(post_attn_name, gpu)
                gpu.batch_rmsnorm_gpu(bufs['attn_out'], w_pan, q_dim, batch, self._norm_eps, bufs['attn_out'])

            # --- Output projection + residual (batched) ---
            gpu.matmul_batch_gpu(bufs['attn_out'], pfx + "attn_output.weight",
                                bufs['xn'], batch)
            gpu.batch_add_gpu(bufs['x'], bufs['xn'], self.dim, batch)

            # --- FFN norm (batched RMSNorm) ---
            ffn_norm_name = pfx + "ffn_norm.weight"
            if gpu.has(ffn_norm_name):
                w_ptr = self._get_weight_ptr(ffn_norm_name, gpu)
                gpu.batch_rmsnorm_gpu(bufs['x'], w_ptr, self.dim, batch,
                                      self._norm_eps, bufs['xn'])
            else:
                gpu.cu.cu.cuMemcpyDtoD_v2(bufs['xn'], bufs['x'], batch * self.dim * 4)

            # --- FFN gate + up (batched matmul) ---
            gpu.matmul_batch_gpu(bufs['xn'], pfx + "ffn_gate.weight",
                                bufs['gate'], batch)
            gpu.matmul_batch_gpu(bufs['xn'], pfx + "ffn_up.weight",
                                bufs['up'], batch)

            # --- Batched SiLU * up ---
            gpu.batch_silu_mul_gpu(bufs['gate'], bufs['up'], self.hidden_dim,
                                  batch, bufs['silu'])

            # --- FFN down + residual (batched) ---
            gpu.matmul_batch_gpu(bufs['silu'], pfx + "ffn_down.weight",
                                bufs['xn'], batch)

            # --- Post-FFW norm if present ---
            post_ffw_name = pfx + "post_ffw_norm.weight"
            if gpu.has(post_ffw_name):
                w_pfn = self._get_weight_ptr(post_ffw_name, gpu)
                gpu.batch_rmsnorm_gpu(bufs['xn'], w_pfn, self.dim, batch, self._norm_eps, bufs['xn'])

            gpu.batch_add_gpu(bufs['x'], bufs['xn'], self.dim, batch)

            # --- Layer output scale if present ---
            scale_name = pfx + "layer_output_scale.weight"
            if gpu.has(scale_name):
                sptr = self._get_weight_ptr(scale_name, gpu)
                if sptr is not None:
                    for b in range(batch):
                        row_off = ctypes.c_ulonglong(bufs['x'].value + b * self.dim * 4)
                        gpu.mul_gpu(row_off, sptr, self.dim)

        # 3. Copy last token's hidden state to the single-token buffer
        #    for the output projection (reuse existing forward path)
        last_off = ctypes.c_ulonglong(bufs['x'].value + (batch - 1) * self.dim * 4)
        gpu.cu.cu.cuMemcpyDtoD_v2(self._bufs1['x'], last_off, self.dim * 4)

        # 4. Final norm + output projection (single token - the last one)
        gpu.rmsnorm_gpu(self._bufs1['x'], self._output_norm_ptr, self.dim,
                        self._norm_eps, self._bufs1['xn'])
        out_name = None
        for n in ("output.weight", "token_embd.weight", "token_emb.weight"):
            if gpu.has(n):
                out_name = n
                break
        gpu.matvec_gpu(self._bufs1['xn'], out_name, self._logits_ptr)

        # 5. Download logits
        logits = gpu.from_gpu_ptr(self._logits_ptr, self.vocab_size)
        if str(self._architecture) == 'gemma4':
            logits = np.tanh(logits / 30.0) * 30.0

        self._tokens_generated += batch
        self._total_time += time.time() - t0
        return logits

    def _layer_forward_gpu(self, layer: int, position: int):
        """Single transformer layer, entirely on the active GPU.

        Uses self._cur_gpu and self._cur_bufs which are set by forward()
        or _forward_dual() before calling this method."""
        gpu = self._cur_gpu
        bufs = self._cur_bufs
        # Ensure the correct CUDA context is current for this GPU before
        # any device memory touches (defensive — fixes silent HtoD/DtoD
        # failures when the worker thread bounces between GPUs).
        gpu.cu._ensure_ctx()
        pfx = f"blk.{layer}."

        # Attention norm
        norm_name = pfx + "attn_norm.weight"
        if gpu.has(norm_name):
            if self._use_layernorm:
                gpu.layernorm_gpu(bufs['x'], self._get_weight_ptr(norm_name, gpu),
                                  self.dim, self._norm_eps, bufs['xn'])
            else:
                gpu.rmsnorm_gpu(bufs['x'], self._get_weight_ptr(norm_name, gpu),
                                self.dim, self._norm_eps, bufs['xn'])
        else:
            gpu.cu.cu.cuMemcpyDtoD_v2(bufs['xn'], bufs['x'], self.dim * 4)

        # QKV projections - fused (one kernel launch instead of three)
        q_dim = self.n_heads * self.head_dim
        kv_dim = self.n_kv_heads * self.head_dim
        wq_name = pfx + "attn_q.weight"
        wk_name = pfx + "attn_k.weight"
        wv_name = pfx + "attn_v.weight"
        if gpu.has(wq_name) and gpu.has(wk_name) and gpu.has(wv_name):
            # Fused QKV (one kernel launch) only helps for F32 weights.
            # For quantized weights, three separate fused dequant+matvec
            # calls avoid the full F32 scratch materialization and are
            # much faster (the fused kernel dequants on-the-fly).
            from qwen35_gpu import G_F32
            wq_gtype = gpu._raw[wq_name][1]
            if wq_gtype == G_F32:
                try:
                    gpu.fused_qkv_gpu(bufs['xn'], wq_name, wk_name, wv_name,
                                      bufs['q'], bufs['k'], bufs['v'],
                                      self.dim, q_dim, kv_dim, kv_dim)
                except Exception:
                    gpu.matvec_gpu(bufs['xn'], wq_name, bufs['q'])
                    gpu.matvec_gpu(bufs['xn'], wk_name, bufs['k'])
                    gpu.matvec_gpu(bufs['xn'], wv_name, bufs['v'])
            else:
                gpu.matvec_gpu(bufs['xn'], wq_name, bufs['q'])
                gpu.matvec_gpu(bufs['xn'], wk_name, bufs['k'])
                gpu.matvec_gpu(bufs['xn'], wv_name, bufs['v'])
        else:
            gpu.matvec_gpu(bufs['xn'], wq_name, bufs['q'])
            gpu.matvec_gpu(bufs['xn'], wk_name, bufs['k'])
            gpu.matvec_gpu(bufs['xn'], wv_name, bufs['v'])

        # QKV biases (Qwen2/Qwen2.5 family).  Upload once per layer on first
        # use, then add in-place on GPU.  LLaMA/Gemma models have no biases
        # so this block is a no-op there.
        for bname, bbuf, bdim in ((pfx + 'attn_q.bias', 'q', q_dim),
                                  (pfx + 'attn_k.bias', 'k', kv_dim),
                                  (pfx + 'attn_v.bias', 'v', kv_dim)):
            if gpu.has(bname):
                bptr = self._get_weight_ptr(bname, gpu)
                if bptr is not None:
                    gpu.add_gpu(bufs[bbuf], bptr, bdim)

        # Q/K normalization if present (Gemma 2, Gemma 3, Command-R, StarCoder2, etc.)
        qn_name = pfx + "attn_q_norm.weight"
        if gpu.has(qn_name):
            gpu.rmsnorm_gpu(bufs['q'], self._get_weight_ptr(qn_name, gpu), q_dim, self._norm_eps, bufs['q'])
        kn_name = pfx + "attn_k_norm.weight"
        if gpu.has(kn_name):
            gpu.rmsnorm_gpu(bufs['k'], self._get_weight_ptr(kn_name, gpu), kv_dim, self._norm_eps, bufs['k'])

        # RoPE on GPU
        gpu.rope_gpu(bufs['q'], bufs['k'],
                     self.n_heads, self.n_kv_heads, self.head_dim,
                     position, self.rope_base)

        # Store K, V into KV cache at this position
        kv_elems = self.n_kv_heads * self.head_dim
        if self._kv_f16:
            # F16 KV cache: convert F32 -> F16 on GPU, store at 2 bytes/element
            kv_offset = position * self._kv_stride * 2
            gpu.store_f16_gpu(bufs['k'], self._ptr_off(self._kv_k_ptrs[layer], kv_offset), kv_elems)
            gpu.store_f16_gpu(bufs['v'], self._ptr_off(self._kv_v_ptrs[layer], kv_offset), kv_elems)
        else:
            kv_offset = position * self._kv_stride * 4
            kv_bytes = kv_elems * 4
            gpu.cu.cu.cuMemcpyDtoD_v2(self._ptr_off(self._kv_k_ptrs[layer], kv_offset), bufs['k'], kv_bytes)
            gpu.cu.cu.cuMemcpyDtoD_v2(self._ptr_off(self._kv_v_ptrs[layer], kv_offset), bufs['v'], kv_bytes)

        # Attention on GPU - F16 cache, flash (long ctx), or standard
        seq_len = position + 1
        if self._kv_f16 and seq_len > 4096:
            gpu.flash_attention_f16_gpu(bufs['q'], self._kv_k_ptrs[layer], self._kv_v_ptrs[layer],
                                       bufs['attn_out'], self.n_heads, self.n_kv_heads,
                                       self.head_dim, seq_len, self._kv_stride)
        elif self._kv_f16:
            gpu.attention_f16_gpu(bufs['q'], self._kv_k_ptrs[layer], self._kv_v_ptrs[layer],
                                  bufs['attn_out'], self.n_heads, self.n_kv_heads,
                                  self.head_dim, seq_len, self._kv_stride)
        elif seq_len > 4096:
            gpu.flash_attention_gpu(bufs['q'], self._kv_k_ptrs[layer], self._kv_v_ptrs[layer],
                                   bufs['attn_out'], self.n_heads, self.n_kv_heads,
                                   self.head_dim, seq_len, self._kv_stride)
        else:
            gpu.attention_gpu(bufs['q'], self._kv_k_ptrs[layer], self._kv_v_ptrs[layer],
                              bufs['attn_out'], self.n_heads, self.n_kv_heads,
                              self.head_dim, seq_len, self._kv_stride)

        # Output projection + residual
        post_attn_name = pfx + "post_attention_norm.weight"
        if gpu.has(post_attn_name):
            gpu.rmsnorm_gpu(bufs['attn_out'], self._get_weight_ptr(post_attn_name, gpu), q_dim, self._norm_eps, bufs['attn_out'])

        gpu.matvec_gpu(bufs['attn_out'], pfx + "attn_output.weight", bufs['xn'])
        gpu.add_gpu(bufs['x'], bufs['xn'], self.dim)

        # FFN norm
        ffn_norm_name = pfx + "ffn_norm.weight"
        if gpu.has(ffn_norm_name):
            if self._use_layernorm:
                gpu.layernorm_gpu(bufs['x'], self._get_weight_ptr(ffn_norm_name, gpu),
                                  self.dim, self._norm_eps, bufs['xn'])
            else:
                gpu.rmsnorm_gpu(bufs['x'], self._get_weight_ptr(ffn_norm_name, gpu),
                                self.dim, self._norm_eps, bufs['xn'])
        else:
            gpu.cu.cu.cuMemcpyDtoD_v2(bufs['xn'], bufs['x'], self.dim * 4)

        # FFN: dense or MoE
        router_name = pfx + "ffn_gate_inp.weight"
        if self._n_experts > 0 and gpu.has(router_name):
            self._moe_ffn_gpu_native(gpu, bufs, layer)
        else:
            gpu.matvec_gpu(bufs['xn'], pfx + "ffn_gate.weight", bufs['gate'])
            gpu.matvec_gpu(bufs['xn'], pfx + "ffn_up.weight", bufs['up'])
            gpu.silu_mul_gpu(bufs['gate'], bufs['up'], self.hidden_dim, bufs['silu'])
            gpu.matvec_gpu(bufs['silu'], pfx + "ffn_down.weight", bufs['xn'])

        # Post-FFW norm (Gemma 2/3)
        post_ffw_name = pfx + "post_ffw_norm.weight"
        if gpu.has(post_ffw_name):
            gpu.rmsnorm_gpu(bufs['xn'], self._get_weight_ptr(post_ffw_name, gpu), self.dim, self._norm_eps, bufs['xn'])

        gpu.add_gpu(bufs['x'], bufs['xn'], self.dim)

        # Layer output scaling if present
        scale_name = pfx + "layer_output_scale.weight"
        if gpu.has(scale_name):
            sptr = self._get_weight_ptr(scale_name, gpu)
            if sptr is not None:
                gpu.mul_gpu(bufs['x'], sptr, self.dim)

    def _moe_ffn_gpu_native(self, gpu, bufs, layer):
        """MoE FFN for the GPU-native engine.  xn stays on VRAM; only the
        tiny router logits and the final combined output cross PCIe."""
        pfx = f"blk.{layer}."
        n_exp = self._n_experts
        top_k = min(self._n_experts_used, n_exp)

        # 1. Router logits on GPU
        router_name = pfx + "ffn_gate_inp.weight"
        if not gpu.has(router_name):
            # Fallback to dense FFN if router is missing
            gpu.matvec_gpu(bufs['xn'], pfx + "ffn_gate.weight", bufs['gate'])
            gpu.matvec_gpu(bufs['xn'], pfx + "ffn_up.weight", bufs['up'])
            gpu.silu_mul_gpu(bufs['gate'], bufs['up'], self.hidden_dim, bufs['silu'])
            gpu.matvec_gpu(bufs['silu'], pfx + "ffn_down.weight", bufs['xn'])
            return

        # Use silu buffer as scratch for router logits (n_experts <= hidden_dim)
        gl_ptr, _ = gpu.matvec_gpu(bufs['xn'], router_name, bufs['silu'])
        gate_logits = gpu.from_gpu_ptr(gl_ptr, n_exp)

        # 2. Softmax + top-k on CPU (n_experts is small)
        gl_max = np.max(gate_logits)
        exp_gl = np.exp(gate_logits - gl_max)
        probs = exp_gl / np.sum(exp_gl)
        top_idx = np.argsort(probs)[-top_k:]
        top_weights = probs[top_idx]
        top_weights = top_weights / np.sum(top_weights)

        # 3. Run selected experts and combine
        out_cpu = np.zeros(self.dim, dtype=np.float32)
        for ei, w in zip(top_idx, top_weights):
            ei = int(ei)
            gn = pfx + f"exp.{ei}.gate"
            un = pfx + f"exp.{ei}.up"
            dn = pfx + f"exp.{ei}.down"
            if not (gpu.has(gn) and gpu.has(un) and gpu.has(dn)):
                continue
            gpu.matvec_gpu(bufs['xn'], gn, bufs['gate'])
            gpu.matvec_gpu(bufs['xn'], un, bufs['up'])
            gpu.silu_mul_gpu(bufs['gate'], bufs['up'], self.hidden_dim, bufs['silu'])
            # Use attn_out buffer as scratch for the expert output (dim floats)
            gpu.matvec_gpu(bufs['silu'], dn, bufs['attn_out'])
            expert_out = gpu.from_gpu_ptr(bufs['attn_out'], self.dim)
            out_cpu += float(w) * expert_out

        # 4. Upload combined output back to GPU (into bufs['xn'])
        r = gpu.cu.cu.cuMemcpyHtoD_v2(bufs['xn'], ctypes.c_void_p(out_cpu.ctypes.data), self.dim * 4)
        if r != 0:
            raise RuntimeError(f"MoE HtoD failed: {r}")

    def _get_weight_ptr(self, name, gpu=None):
        """Get the GPU device pointer for a resident weight on the given GpuMM."""
        if gpu is None:
            gpu = self._gpu
        entry = gpu._raw.get(name)
        if entry is None:
            return None
        if entry[0] == 'stream':
            return None
        return entry[0]

    # ════════════════════════════════════════════════════════════════
    # INTERFACE (matches GPUStreamEngine)
    # ════════════════════════════════════════════════════════════════

    def reset(self):
        """Clear KV cache on all GPUs."""
        # Ensure CUDA context is current before touching device memory
        if self._gpu is not None:
            self._gpu.cu._ensure_ctx()
        # cuMemsetD32_v2 writes 4-byte words.  For F32 cache each element is
        # 4 bytes, so kv_words == element count.  For F16 cache each element
        # is 2 bytes, so we need half as many 32-bit words as F16 elements.
        kv_words = self.max_seq_len * self._kv_stride
        if self._kv_f16:
            kv_words = kv_words // 2
        for layer, ptr in enumerate(self._kv_k_ptrs or []):
            try:
                gpu = self._gpu2 if (self._dual and layer >= self._split_layer) else self._gpu
                gpu.cu._ensure_ctx()
                gpu.cu.cu.cuMemsetD32_v2(ptr, 0, kv_words)
            except Exception:
                pass
        for layer, ptr in enumerate(self._kv_v_ptrs or []):
            try:
                gpu = self._gpu2 if (self._dual and layer >= self._split_layer) else self._gpu
                gpu.cu._ensure_ctx()
                gpu.cu.cu.cuMemsetD32_v2(ptr, 0, kv_words)
            except Exception:
                pass
        self._tokens_generated = 0
        self._total_time = 0.0

    def get_stats(self):
        total = self._total_time
        return {
            'tokens': self._tokens_generated,
            'tok/s': self._tokens_generated / total if total > 0 else 0,
            'total_time': total,
            'engine': 'gpu-native-dual' if self._dual else 'gpu-native',
        }

    def close(self):
        """Free GPU resources on all GPUs."""
        # Free GPU1 buffers
        cu1 = self._gpu.cu if self._gpu else None
        if cu1 is not None:
            for ptr in [self._x_ptr, self._xn_ptr, self._q_ptr, self._k_ptr,
                        self._v_ptr, self._attn_out_ptr, self._gate_ptr,
                        self._up_ptr, self._silu_ptr, self._logits_ptr,
                        self._output_norm_ptr]:
                if ptr is not None:
                    try:
                        cu1.free(ptr)
                    except Exception:
                        pass
            if getattr(self, '_batch_bufs', None) is not None:
                for ptr in self._batch_bufs.values():
                    if ptr is not None:
                        try:
                            cu1.free(ptr)
                        except Exception:
                            pass
                self._batch_bufs = None
        # Free GPU2 buffers (dual mode)
        cu2 = self._gpu2.cu if self._gpu2 else None
        if cu2 is not None and self._bufs2 is not None:
            for ptr in list(self._bufs2.values()) + [self._logits2_ptr, self._output_norm2_ptr]:
                if ptr is not None:
                    try:
                        cu2.free(ptr)
                    except Exception:
                        pass
        # Free KV cache (split across GPUs)
        for layer, ptr in enumerate(self._kv_k_ptrs or []):
            try:
                gpu = self._gpu2 if (self._dual and layer >= self._split_layer) else self._gpu
                gpu.cu.free(ptr)
            except Exception:
                pass
        for layer, ptr in enumerate(self._kv_v_ptrs or []):
            try:
                gpu = self._gpu2 if (self._dual and layer >= self._split_layer) else self._gpu
                gpu.cu.free(ptr)
            except Exception:
                pass
        if self._gpu is not None:
            try:
                self._gpu.close()
            except Exception:
                pass
        if self._gpu2 is not None:
            try:
                self._gpu2.close()
            except Exception:
                pass
        # Drop source readers + parsed model (hold mmap'd file data + RAM)
        self._parser = None
        self._nsvd = None
        self.m = None
        self.loaded = False
        import gc
        gc.collect()

    @property
    def name(self):
        return Path(self.model_path).stem

    @property
    def architecture(self):
        return self._architecture

    @property
    def bos_id(self):
        return self._bos_id

    @property
    def eos_id(self):
        return self._eos_id

    @bos_id.setter
    def bos_id(self, v):
        self._bos_id = v

    @eos_id.setter
    def eos_id(self, v):
        self._eos_id = v

    # ════════════════════════════════════════════════════════════════
    # TOKENIZER + METADATA INTERFACE (matches GPUStreamEngine)
    # ════════════════════════════════════════════════════════════════

    def encode(self, text: str, add_bos: bool = True) -> list:
        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        token_to_id = getattr(self, 'token_to_id', {})
        for ch in text:
            token_id = token_to_id.get(ch)
            if token_id is not None:
                tokens.append(token_id)
            else:
                for byte in ch.encode('utf-8'):
                    byte_token = f"<0x{byte:02X}>"
                    bid = token_to_id.get(byte_token, 0)
                    tokens.append(bid)
        if getattr(self, 'merges', []):
            tokens = self._apply_bpe(tokens)
        return tokens

    def _apply_bpe(self, tokens: list) -> list:
        merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        while True:
            best_pair = None
            best_rank = len(self.merges)
            best_idx = -1
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                rank = merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
                    best_idx = i
            if best_pair is None:
                break
            a_str = self.tokens[best_pair[0]] if best_pair[0] < len(self.tokens) else ""
            b_str = self.tokens[best_pair[1]] if best_pair[1] < len(self.tokens) else ""
            merged_str = a_str + b_str
            merged_id = self.token_to_id.get(merged_str)
            if merged_id is None:
                del merge_rank[best_pair]
                continue
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(merged_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode_single(self, token_id: int) -> str:
        tok_list = getattr(self, 'tokens', [])
        if 0 <= token_id < len(tok_list):
            tok = tok_list[token_id]
            if self._is_control_like(tok):
                return ""
            return self._clean_token(tok)
        return ""

    def _clean_token(self, token: str) -> str:
        token = token.replace("▁", " ").replace("<0x0A>", "\n")
        if token.startswith("<0x") and token.endswith(">"):
            try:
                return chr(int(token[3:-1], 16))
            except (ValueError, OverflowError):
                return token
        return token

    def _is_control_like(self, token: str) -> bool:
        return token in ("<s>", "</s>", "<pad>", "<unk>", "\ufffd",
                         "<|im_start|>", "<|im_end|>", "<bos>", "<eos>")

    @property
    def metadata(self):
        # GGUF / SafeTensors: expose the full parsed metadata (tokenizer included)
        m = getattr(self, 'm', None)
        if m is not None and getattr(m, 'metadata', None):
            return m.metadata
        # NTMRE containers: expose the embedded tokenizer so the suite's
        # Tokenizer works exactly like with a GGUF file
        md = {
            'general.architecture': self.architecture,
            'general.name': self.name,
        }
        tb = getattr(getattr(self, '_nsvd', None), 'tokenizer_bytes', None)
        if tb:
            try:
                import json as _json
                d = _json.loads(tb.decode('utf-8'))
                md['tokenizer.ggml.tokens'] = d.get('tokens', [])
                md['tokenizer.ggml.merges'] = d.get('merges', [])
                md['tokenizer.ggml.bos_token_id'] = d.get('bos', 1)
                md['tokenizer.ggml.eos_token_id'] = d.get('eos', 2)
                md['tokenizer.ggml.padding_token_id'] = d.get('pad', 0)
                md['tokenizer.ggml.unknown_token_id'] = d.get('unk', 3)
                if d.get('model'):
                    md['tokenizer.ggml.model'] = d['model']
            except Exception:
                pass
        return md

    # KV cache interface (compatible with unified app). GpuNativeEngine
    # takes the position explicitly in forward(), so advance() is a no-op.
    class _KVCache:
        def __init__(self, parent):
            self._parent = parent
        def advance(self):
            pass

    @property
    def kv_cache(self):
        return self._KVCache(self)
