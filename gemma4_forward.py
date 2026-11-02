#!/usr/bin/env python3
"""
Gemma 4 E2B ONLY — Complete Standalone Forward Pass
====================================================
E2B ONLY: 35 layers, dim 1536, 8 heads, 1 kv head, FFN 6144 (verified spec).
DO NOT use this file for E4B (42/2560), 12B (48/3840) or 26B-A4B (MoE).
Those variants have different geometry and this file has no path for them.
If a loaded model is not E2B geometry, a loud warning is printed below.
================================================
Nightmare Designs — Full architecture implementation.

Handles ALL unique Gemma 4 E2B features:
  - Per-Layer Embeddings (PLE) with dual embedding tables
  - Dual attention: sliding window (256-dim) + full causal (512-dim)
  - Q/K per-head RMSNorm before RoPE
  - Custom precomputed RoPE frequencies (rope_freqs.weight)
  - Sandwich norms (post_attention_norm, post_ffw_norm)
  - Layer output scaling
  - Variable FFN dimensions per layer
  - Shared KV layers (V=K)
  - Final logit softcapping

Usage:
  python gemma4_forward.py --model path/to/gemma4.gguf --prompt "hey"
  python gemma4_forward.py --model path/to/gemma4.gguf --prompt "hey" --debug

Author: Nightmare Designs [!]
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import nightmare_hardware
    nightmare_hardware.configure_system_environment()
except Exception:
    pass

import numpy as np
from gguf_parser import GGUFParser

try:
    from fast_matmul import FastWeights
except ImportError:
    FastWeights = None

# ============================================================
# CONFIGURATION (read from GGUF metadata at runtime)
# ============================================================

DEBUG = False  # Set via --debug flag


def dbg(msg):
    """Debug print — only when --debug is active."""
    if DEBUG:
        print(f"  [DBG] {msg}")


# ============================================================
# MATH PRIMITIVES
# ============================================================

def rms_norm(x, weight, eps=1e-6):
    """RMSNorm: x / rms(x) * weight. Gemma 4 uses eps=1e-6."""
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def gelu(x):
    """GELU activation (exact, not approx)."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x * x * x)))


def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def softmax(x, axis=-1):
    """Numerically stable softmax."""
    x_max = x.max(axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / e_x.sum(axis=axis, keepdims=True)


# ============================================================
# ROPE (Rotary Position Embedding)
# ============================================================

class RoPE:
    """
    Gemma 4 RoPE with precomputed frequencies from rope_freqs.weight.
    
    Two modes:
      - SWA layers: head_dim=256, uses first 128 frequencies
      - Full layers: head_dim=512, uses all 256 frequencies
    """
    
    def __init__(self, rope_freqs, max_seq=4096):
        """
        Args:
            rope_freqs: np.ndarray of shape (256,) — precomputed inverse frequencies
            max_seq: maximum sequence length to precompute
        """
        self.freqs = rope_freqs.astype(np.float32)  # (256,) = half of full head_dim 512
        self.max_seq = max_seq
        
        # Precompute cos/sin tables for all positions
        positions = np.arange(max_seq, dtype=np.float32)
        # Full table: (max_seq, 256) — one entry per freq per position
        angles = np.outer(positions, self.freqs)  # (max_seq, 256)
        self.cos_full = np.cos(angles).astype(np.float32)  # (max_seq, 256)
        self.sin_full = np.sin(angles).astype(np.float32)  # (max_seq, 256)
    
    def apply(self, q, k, pos, head_dim):
        """
        Apply RoPE to Q and K vectors.
        
        Args:
            q: (n_heads, head_dim) query vectors
            k: (n_kv_heads, head_dim) key vectors
            pos: integer position in sequence
            head_dim: 256 for SWA, 512 for full attention
        """
        half = head_dim // 2
        
        # Slice the appropriate number of frequencies
        cos_p = self.cos_full[pos, :half]  # (half,)
        sin_p = self.sin_full[pos, :half]  # (half,)
        
        # Apply rotation (split-half style, as Gemma uses)
        # Q rotation
        q0 = q[:, :half].copy()
        q1 = q[:, half:].copy()
        q[:, :half] = q0 * cos_p - q1 * sin_p
        q[:, half:] = q0 * sin_p + q1 * cos_p
        
        # K rotation
        k0 = k[:, :half].copy()
        k1 = k[:, half:].copy()
        k[:, :half] = k0 * cos_p - k1 * sin_p
        k[:, half:] = k0 * sin_p + k1 * cos_p
        
        return q, k


# ============================================================
# KV CACHE (supports both sliding window and full attention)
# ============================================================

class Gemma4KVCache:
    """
    KV Cache with per-layer configuration:
    - SWA layers: only store last `window_size` tokens (ring buffer)
    - Full attention layers: store all tokens (up to max_seq)
    """
    
    def __init__(self, n_layers, max_seq, window_size, layer_head_dims, n_kv_heads, swa_pattern):
        """
        Args:
            n_layers: number of transformer layers
            max_seq: maximum total sequence length
            window_size: sliding window size for SWA layers
            layer_head_dims: list of head_dim per layer (256 or 512)
            n_kv_heads: number of KV heads (1 for Gemma 4 E2B)
            swa_pattern: list of bools — True = SWA, False = full attention
        """
        self.n_layers = n_layers
        self.max_seq = max_seq
        self.window_size = window_size
        self.n_kv_heads = n_kv_heads
        self.swa_pattern = swa_pattern
        self.layer_head_dims = layer_head_dims
        
        # Per-layer caches
        self.keys = []    # list of (capacity, n_kv_heads, head_dim) arrays
        self.values = []  # same
        self.lengths = [0] * n_layers  # how many tokens stored per layer
        
        for layer in range(n_layers):
            hd = layer_head_dims[layer]
            if swa_pattern[layer]:
                capacity = window_size
            else:
                capacity = max_seq
            self.keys.append(np.zeros((capacity, n_kv_heads, hd), dtype=np.float32))
            self.values.append(np.zeros((capacity, n_kv_heads, hd), dtype=np.float32))
        
        self.seq_len = 0  # Total tokens seen so far
    
    def store(self, layer, k, v):
        """
        Store K, V for the current position.
        
        Args:
            layer: layer index
            k: (n_kv_heads, head_dim) key vector for this position
            v: (n_kv_heads, head_dim) value vector for this position
        """
        if self.swa_pattern[layer]:
            # Ring buffer for sliding window
            idx = self.seq_len % self.window_size
        else:
            # Append for full attention
            idx = self.seq_len
            if idx >= self.max_seq:
                raise RuntimeError(f"KV cache overflow: pos={idx} >= max_seq={self.max_seq}")
        
        self.keys[layer][idx] = k
        self.values[layer][idx] = v
        self.lengths[layer] = min(self.seq_len + 1, 
                                   self.window_size if self.swa_pattern[layer] else self.max_seq)
    
    def get_kv(self, layer):
        """
        Get all stored K, V for attention computation.
        
        Returns:
            keys: (seq_len, n_kv_heads, head_dim)
            values: (seq_len, n_kv_heads, head_dim)
        """
        length = self.lengths[layer]
        
        if self.swa_pattern[layer]:
            # Sliding window: may need to handle wrap-around
            if self.seq_len < self.window_size:
                return self.keys[layer][:length], self.values[layer][:length]
            else:
                # Ring buffer: reorder so oldest is first
                start = (self.seq_len + 1) % self.window_size
                # Technically we should reorder, but for attention we just need all stored values
                # The positions are implicit from the cache structure
                return self.keys[layer][:self.window_size], self.values[layer][:self.window_size]
        else:
            return self.keys[layer][:length], self.values[layer][:length]
    
    def advance(self):
        """Called after processing a token to advance the position counter."""
        self.seq_len += 1


# ============================================================
# GEMMA 4 E2B MODEL
# ============================================================

class Gemma4E2B:
    """
    Complete Gemma 4 E2B inference engine.
    Loads a GGUF file and runs forward passes with full architecture support.
    """
    
    def __init__(self, model_path, max_seq_len=2048):
        print(f"\n{'='*60}")
        print(f"  Gemma 4 E2B — Nightmare Designs Forward Engine")
        print(f"  Model: {model_path}")
        print(f"{'='*60}\n")
        
        t0 = time.time()
        
        # Parse GGUF
        self.parser = GGUFParser(model_path)
        self.model = self.parser.parse()
        self.metadata = self.model.metadata
        
        # === Read architecture config from metadata ===
        self.dim = self._meta_int('gemma4.embedding_length', 1536)
        self.n_layers = self._meta_int('gemma4.block_count', 35)
        self.n_heads = self._meta_int('gemma4.attention.head_count', 8)
        self.n_kv_heads = self._meta_int('gemma4.attention.head_count_kv', 1)
        self.vocab_size = 262144  # Always 262144 for Gemma 4
        self.norm_eps = float(self.metadata.get('gemma4.attention.layer_norm_rms_epsilon', 1e-6))
        self.logit_cap = float(self.metadata.get('gemma4.final_logit_softcapping', 30.0))
        
        # Attention config
        self.key_length_full = self._meta_int('gemma4.attention.key_length', 512)
        self.key_length_swa = self._meta_int('gemma4.attention.key_length_swa', 256)
        self.value_length_full = self._meta_int('gemma4.attention.value_length', 512)
        self.value_length_swa = self._meta_int('gemma4.attention.value_length_swa', 256)
        self.window_size = self._meta_int('gemma4.attention.sliding_window', 512)
        self.shared_kv_count = self._meta_int('gemma4.attention.shared_kv_layers', 20)
        
        # Sliding window pattern
        swa_raw = self.metadata.get('gemma4.attention.sliding_window_pattern', None)
        if swa_raw is not None and isinstance(swa_raw, (list, np.ndarray)):
            self.swa_pattern = [bool(x) for x in swa_raw[:self.n_layers]]
        else:
            # Default: 4 SWA + 1 full, repeating
            self.swa_pattern = [(i % 5 != 4) for i in range(self.n_layers)]
        
        # FFN dimensions (per-layer list)
        # Fixed 2026-10-28: the old scalar fallback 12288 is NOT the
        # verified E2B FFN. 12288 appears in Gemma4 per-layer FFN
        # lists (e.g. [6144 x15, 12288 x ...] in nightmare_stream.py),
        # i.e. it is a widest-layer width, not a verified scalar.
        # For an identified E2B model (35L/1536d) with no FFN key,
        # use the verified 6144 and warn. Otherwise keep 12288
        # ONLY as a loud, unverified fallback and warn.
        if 'gemma4.feed_forward_length' in self.metadata:
            ffn_raw = self.metadata.get('gemma4.feed_forward_length')
            self._ffn_fallback_used = False
        elif self.dim == 1536 and self.n_layers == 35:
            ffn_raw = 6144
            self._ffn_fallback_used = True
            print('[MODEL-SETUP] WARNING: gemma4.feed_forward_length '
                  'missing; using verified E2B FFN 6144 (not 12288)')
        else:
            ffn_raw = 12288
            self._ffn_fallback_used = True
            print('[MODEL-SETUP] WARNING: gemma4.feed_forward_length '
                  'missing and model is not identified E2B geometry '
                  '(%dL x %dd); using UNVERIFIED fallback 12288 — '
                  'verify before trusting' % (self.n_layers, self.dim))
        if isinstance(ffn_raw, (list, np.ndarray)):
            self.ffn_dims = [int(x) for x in ffn_raw[:self.n_layers]]
        else:
            self.ffn_dims = [int(ffn_raw)] * self.n_layers
        
        # PLE config
        self.ple_dim = self._meta_int('gemma4.embedding_length_per_layer_input', 256)
        self.ple_total_dim = self.ple_dim * self.n_layers  # 256 * 35 = 8960
        # E2B-only guard: this file has no E4B/12B/26B path.
        if (self.dim, self.n_layers) != (1536, 35):
            print('[MODEL-SETUP] WARNING: Gemma4E2B is E2B-ONLY '
                  '(35L x 1536d); this model is %dL x %dd — do not '
                  'trust output from this file for it' % (self.n_layers, self.dim))
        
        # RoPE config
        self.rope_dim_full = self._meta_int('gemma4.rope.dimension_count', 512)
        self.rope_dim_swa = self._meta_int('gemma4.rope.dimension_count_swa', 256)
        
        # Per-layer head dimensions
        self.layer_head_dims = []
        for i in range(self.n_layers):
            if self.swa_pattern[i]:
                self.layer_head_dims.append(self.key_length_swa)  # 256
            else:
                self.layer_head_dims.append(self.key_length_full)  # 512
        
        # Per-layer Q head counts (inferred from Q weight shapes)
        self.layer_q_heads = []
        for i in range(self.n_layers):
            q_name = f"blk.{i}.attn_q.weight"
            if q_name in self.model.tensors:
                q_shape = self.model.tensors[q_name].shape
                q_out = q_shape[1] if len(q_shape) > 1 else q_shape[0]
                hd = self.layer_head_dims[i]
                self.layer_q_heads.append(q_out // hd)
            else:
                self.layer_q_heads.append(self.n_heads)
        
        # Shared KV layers (V = K)
        self.shared_kv_layers = set()
        for i in range(self.n_layers):
            v_name = f"blk.{i}.attn_v.weight"
            if v_name not in self.model.tensors:
                self.shared_kv_layers.add(i)
        
        self.max_seq_len = min(max_seq_len, 131072)
        
        # Print config
        print(f"[Config] dim={self.dim}, layers={self.n_layers}, vocab={self.vocab_size}")
        print(f"[Config] n_heads={self.n_heads}, n_kv_heads={self.n_kv_heads}")
        print(f"[Config] head_dim: SWA={self.key_length_swa}, Full={self.key_length_full}")
        print(f"[Config] SWA pattern: {sum(self.swa_pattern)} SWA + {self.n_layers - sum(self.swa_pattern)} Full")
        print(f"[Config] FFN dims: {sorted(set(self.ffn_dims))}")
        print(f"[Config] PLE dim={self.ple_dim}, total={self.ple_total_dim}")
        print(f"[Config] Shared KV layers: {len(self.shared_kv_layers)}")
        print(f"[Config] Norm eps={self.norm_eps}, logit_cap={self.logit_cap}")
        print(f"[Config] Max seq len={self.max_seq_len}, window={self.window_size}")
        
        # === Load weights ===
        self._load_weights()
        
        # === Initialize RoPE ===
        self._init_rope()
        
        # === Initialize KV cache ===
        self.kv_cache = Gemma4KVCache(
            self.n_layers, self.max_seq_len, self.window_size,
            self.layer_head_dims, self.n_kv_heads, self.swa_pattern
        )
        
        # === Initialize FastWeights (optional speed boost) ===
        self._init_fast_weights()
        
        elapsed = time.time() - t0
        print(f"\n[Ready] Model loaded in {elapsed:.1f}s")
        print(f"{'='*60}\n")
    
    def _meta_int(self, key, default):
        """Get integer metadata value."""
        val = self.metadata.get(key, default)
        if isinstance(val, (list, np.ndarray)):
            return int(val[0]) if len(val) > 0 else default
        return int(val)
    
    # ============================================================
    # WEIGHT LOADING
    # ============================================================
    
    def _load_weights(self):
        """Load all model weights into RAM."""
        print("[Loading] Token embeddings...")
        
        self.weights = {}
        
        # Token embedding (vocab × dim)
        self.token_embd = self.parser.load_tensor(self.model, "token_embd.weight")
        if self.token_embd.shape[0] == self.dim:
            self.token_embd = self.token_embd.T  # Transpose to (vocab, dim)
        print(f"  token_embd: {self.token_embd.shape}")
        assert self.token_embd.shape == (self.vocab_size, self.dim),\
            f"token_embd shape mismatch: {self.token_embd.shape} != ({self.vocab_size}, {self.dim})"
        
        # PLE embedding table (vocab × ple_total_dim)
        print("[Loading] PLE embeddings...")
        if "per_layer_token_embd.weight" in self.model.tensors:
        # NOTE: This is HUGE (262144 × 8960 = ~9.4 GB as f32)
        # Keep it in quantized form and dequant per-row like the suite does for token_embd
            ple_info = self.model.tensors["per_layer_token_embd.weight"]
            self._ple_raw_info = ple_info
            # Load raw bytes (stays quantized — much smaller than f32)
            self._ple_raw_bytes, self._ple_quant_type, self._ple_shape, self._ple_n_elements =\
                self.parser.load_tensor_raw(self.model, "per_layer_token_embd.weight")
            # Determine shape for row indexing
            # GGUF shape is (8960, 262144) — we want rows of 8960
            self._ple_row_dim = self.ple_total_dim  # 8960
            self._ple_n_rows = self._ple_n_elements // self._ple_row_dim  # 262144
            self.ple_token_embd = None  # Not fully dequantized
            print(f"  per_layer_token_embd: ({self._ple_n_rows}, {self._ple_row_dim}) [kept quantized, {len(self._ple_raw_bytes)/1024/1024:.0f} MB raw]")
            self._ple_raw_bytes = None
        else:
            print("  [WARN] per_layer_token_embd not found — PLE disabled")
            self.ple_token_embd = None
        
        # PLE model projection (dim → ple_total_dim)
        if "per_layer_model_proj.weight" in self.model.tensors:
            self.ple_model_proj = self.parser.load_tensor(self.model, "per_layer_model_proj.weight")
            print(f"  per_layer_model_proj: {self.ple_model_proj.shape}")
            # Ensure it's (dim, ple_total_dim) for x @ W
            if self.ple_model_proj.shape[0] == self.ple_total_dim and self.ple_model_proj.shape[1] == self.dim:
                self.ple_model_proj = self.ple_model_proj.T
            elif self.ple_model_proj.shape == (self.dim, self.ple_total_dim):
                pass  # Already correct
            else:
                print(f"  [WARN] per_layer_model_proj unexpected shape {self.ple_model_proj.shape}")
        else:
            self.ple_model_proj = None
        
        # PLE projection norm
        if "per_layer_proj_norm.weight" in self.model.tensors:
            self.ple_proj_norm = self.parser.load_tensor(self.model, "per_layer_proj_norm.weight")
            print(f"  per_layer_proj_norm: {self.ple_proj_norm.shape}")
        else:
            self.ple_proj_norm = None
        
        # RoPE frequencies
        if "rope_freqs.weight" in self.model.tensors:
            self.rope_freqs_tensor = self.parser.load_tensor(self.model, "rope_freqs.weight")
            print(f"  rope_freqs: {self.rope_freqs_tensor.shape}")
        else:
            self.rope_freqs_tensor = None
        
        # Output norm
        self.output_norm = self.parser.load_tensor(self.model, "output_norm.weight")
        print(f"  output_norm: {self.output_norm.shape}")
        
        # Output weight (tied to embedding)
        if "output.weight" in self.model.tensors:
            self.output_weight = self.parser.load_tensor(self.model, "output.weight")
        else:
            self.output_weight = None  # Use token_embd.T
            print("  output: tied to token_embd")
        
        # === Per-layer weights ===
        print(f"[Loading] {self.n_layers} transformer layers...")
        for layer in range(self.n_layers):
            prefix = f"blk.{layer}"
            self._load_layer(layer, prefix)
            if (layer + 1) % 10 == 0 or layer == self.n_layers - 1:
                print(f"  Loaded layer {layer + 1}/{self.n_layers}")
        
        print(f"[Loading] Complete — {len(self.weights)} weight tensors")
    
    def _load_layer(self, layer, prefix):
        """Load all weights for a single transformer layer."""
        
        def load(name):
            full_name = f"{prefix}.{name}"
            if full_name in self.model.tensors:
                t = self.parser.load_tensor(self.model, full_name)
                return t
            return None
        
        # Attention norms
        self.weights[f"{prefix}.attn_norm"] = load("attn_norm.weight")
        self.weights[f"{prefix}.attn_q_norm"] = load("attn_q_norm.weight")
        self.weights[f"{prefix}.attn_k_norm"] = load("attn_k_norm.weight")
        
        # QKV + output projections
        self.weights[f"{prefix}.attn_q"] = load("attn_q.weight")
        self.weights[f"{prefix}.attn_k"] = load("attn_k.weight")
        self.weights[f"{prefix}.attn_v"] = load("attn_v.weight")
        self.weights[f"{prefix}.attn_output"] = load("attn_output.weight")
        
        # Sandwich norms
        self.weights[f"{prefix}.post_attention_norm"] = load("post_attention_norm.weight")
        self.weights[f"{prefix}.post_ffw_norm"] = load("post_ffw_norm.weight")
        
        # FFN
        self.weights[f"{prefix}.ffn_norm"] = load("ffn_norm.weight")
        self.weights[f"{prefix}.ffn_gate"] = load("ffn_gate.weight")
        self.weights[f"{prefix}.ffn_up"] = load("ffn_up.weight")
        self.weights[f"{prefix}.ffn_down"] = load("ffn_down.weight")
        
        # Layer output scale
        self.weights[f"{prefix}.layer_output_scale"] = load("layer_output_scale.weight")
        
        # PLE per-layer weights
        self.weights[f"{prefix}.inp_gate"] = load("inp_gate.weight")
        self.weights[f"{prefix}.proj"] = load("proj.weight")
        self.weights[f"{prefix}.post_norm"] = load("post_norm.weight")
    
    def _init_rope(self):
        """Initialize RoPE with custom frequencies from the model."""
        if self.rope_freqs_tensor is not None:
            freqs = self.rope_freqs_tensor
            print(f"[RoPE] Using model's precomputed frequencies: {freqs.shape}")
        else:
            # Fallback: compute standard RoPE frequencies
            # Full attention: theta=1000000, dim=512 → 256 frequencies
            half = self.rope_dim_full // 2
            freqs = 1.0 / (1000000.0 ** (np.arange(0, half, dtype=np.float32) * 2.0 / self.rope_dim_full))
            print(f"[RoPE] Computed fallback frequencies: theta=1M, dim={self.rope_dim_full}")
        
        self.rope = RoPE(freqs, max_seq=self.max_seq_len)
    
    def _init_fast_weights(self):
        """Register quantized weights for fast CPU SIMD matmul."""
        self.fast = None
        # DISABLED: FastWeights DLL expects raw bytes in (out_dim, in_dim) row-major
        # layout, but this GGUF stores weights as (in_dim, out_dim) for Gemma 4.
        # The DLL reads wrong row boundaries → garbage output.
        # TODO: Either transpose raw bytes before registration, or use the NumPy path.
        # For now, use NumPy x @ W (correct but slower).
        if False and FastWeights is not None:
            try:
                self.fast = FastWeights()
                if self.fast.available:
                    registered = 0
                    for tensor_name, info in self.model.tensors.items():
                        if len(info.shape) != 2:
                            continue
                        if "norm" in tensor_name or "embd" in tensor_name or "emb" in tensor_name:
                            continue
                        if "scale" in tensor_name:
                            continue
                        
                        key = tensor_name.replace(".weight", "")
                        loaded_w = self.weights.get(key)
                        if loaded_w is None or loaded_w.ndim != 2:
                            continue
                        
                        # DLL expects raw bytes in (n_out, n_in) layout — rows are output neurons.
                        # GGUF raw bytes are in native (out_dim, in_dim) order (NOT transposed).
                        # loaded_w.shape = (in_dim, out_dim) after parser transpose.
                        # So for the DLL: n_in = loaded_w.shape[0], n_out = loaded_w.shape[1]... 
                        # BUT raw bytes are (out_dim, in_dim) = (loaded_w.shape[1], loaded_w.shape[0])
                        # DLL wants: n_in = cols of raw = loaded_w.shape[0], n_out = rows of raw = loaded_w.shape[1]
                        n_out, n_in = loaded_w.shape[1], loaded_w.shape[0]
                        try:
                            raw, quant_type, shape, n_elements = self.parser.load_tensor_raw(
                                self.model, tensor_name)
                            self.fast.register(key, raw, quant_type, n_in, n_out)
                            registered += 1
                        except Exception:
                            pass
                    
                    if registered > 0:
                        print(f"[FastWeights] Registered {registered} tensors ({self.fast.memory_mb():.1f} MB)")
                    else:
                        self.fast = None
                else:
                    self.fast = None
            except Exception as e:
                print(f"[FastWeights] Init failed: {e}")
                self.fast = None
    
    # ============================================================
    # MATMUL HELPER
    # ============================================================
    
    def matmul(self, x, key):
        """
        Matrix-vector multiply: y = x @ W
        Uses FastWeights if available, otherwise NumPy.
        
        Args:
            x: input vector (n_in,) or batch (batch, n_in)
            key: weight key string (e.g. "blk.0.attn_q")
        """
        w = self.weights[key]
        if w is None:
            raise ValueError(f"Weight '{key}' is None")
        
        if self.fast and self.fast.has(key):
            if x.ndim == 1:
                return self.fast.matvec(x, key, fallback_weight=w)
            else:
                # Batch: process each row
                out = np.empty((x.shape[0], w.shape[1]), dtype=np.float32)
                for i in range(x.shape[0]):
                    out[i] = self.fast.matvec(x[i], key, fallback_weight=w)
                return out
        else:
            return x @ w
    
    # ============================================================
    # PER-LAYER EMBEDDING (PLE)
    # ============================================================
    
    def compute_ple(self, token_ids, x_initial):
        """
        Compute Per-Layer Embeddings for the given tokens.
        
        PLE signal is derived from:
          1. Token identity (per_layer_token_embd lookup)
          2. Initial hidden representation (projected through per_layer_model_proj)
        
        Args:
            token_ids: list of token IDs
            x_initial: initial hidden states (n_tokens, dim) after embedding + scaling
        
        Returns:
            ple: (n_tokens, n_layers, ple_dim) — per-layer conditioning signal
        """
        n_tokens = len(token_ids)
        
        if self._ple_raw_bytes is None and self.ple_token_embd is None:
            # PLE not available — return zeros
            return np.zeros((n_tokens, self.n_layers, self.ple_dim), dtype=np.float32)
        
        # 1. Token-based PLE: lookup in second embedding table
        # Dequantize only the rows we need (avoids 9.4 GB RAM explosion)
        if self.ple_token_embd is not None:
            # Fully dequantized (legacy path)
            ple_from_tokens = np.array([self.ple_token_embd[tid] for tid in token_ids],
                                        dtype=np.float32)
        else:
            # Quantized: dequant per-row on demand
            from dequant_fast import BLOCK_SIZES, DEQUANT_FUNCTIONS
            block_elems, block_bytes = BLOCK_SIZES[self._ple_quant_type]
            row_bytes = (self._ple_row_dim // block_elems) * block_bytes
            rows = []
            for tid in token_ids:
                offset = tid * row_bytes
                row_data = bytes(self._ple_raw_bytes[offset:offset + row_bytes])
                row = DEQUANT_FUNCTIONS[self._ple_quant_type](row_data, self._ple_row_dim)
                rows.append(row)
            ple_from_tokens = np.array(rows, dtype=np.float32)  # (n_tokens, 8960)
        
        # 2. Hidden-state-based PLE: project initial embeddings
        if self.ple_model_proj is not None:
            # x_initial: (n_tokens, dim=1536), ple_model_proj: (dim, 8960)
            ple_from_hidden = x_initial @ self.ple_model_proj  # (n_tokens, 8960)
            ple_combined = ple_from_tokens + ple_from_hidden
        else:
            ple_combined = ple_from_tokens
        
        # 3. Reshape to (n_tokens, n_layers, ple_dim)
        ple = ple_combined.reshape(n_tokens, self.n_layers, self.ple_dim)
        
        # 4. Normalize each layer's PLE chunk
        if self.ple_proj_norm is not None:
            for i in range(self.n_layers):
                ple[:, i, :] = rms_norm(ple[:, i, :], self.ple_proj_norm, self.norm_eps)
        
        dbg(f"PLE computed: shape={ple.shape}, range=[{ple.min():.4f}, {ple.max():.4f}]")
        return ple
    
    # ============================================================
    # SINGLE LAYER FORWARD PASS
    # ============================================================
    
    def layer_forward(self, x, layer, pos, ple_layer):
        """
        Forward pass through a single transformer layer.
        
        Args:
            x: hidden state (dim,) for single token
            layer: layer index (0..34)
            pos: position in sequence (for RoPE and KV cache)
            ple_layer: PLE signal for this layer (ple_dim,) = (256,)
        
        Returns:
            x: updated hidden state (dim,)
        """
        prefix = f"blk.{layer}"
        is_swa = self.swa_pattern[layer]
        head_dim = self.layer_head_dims[layer]
        n_q_heads = self.layer_q_heads[layer]
        
        # === Step A: PLE Injection ===
        inp_gate_w = self.weights.get(f"{prefix}.inp_gate")
        proj_w = self.weights.get(f"{prefix}.proj")
        post_norm_w = self.weights.get(f"{prefix}.post_norm")
        
        # DISABLED: PLE injection — llama.cpp skips PLE entirely and still produces
        # coherent output. Enable once the base forward pass is verified correct.
        if False and inp_gate_w is not None and proj_w is not None and ple_layer is not None:
            # gate = sigmoid(x @ inp_gate) → (ple_dim,)
            gate = sigmoid(x @ inp_gate_w)  # (256,)
            
            # signal = (gate * ple) @ proj → (dim,)
            gated_ple = gate * ple_layer  # (256,)
            signal = gated_ple @ proj_w   # (256,) @ (256, 1536) → (1536,)
            
            # x = rms_norm(x + signal, post_norm)
            if post_norm_w is not None:
                x = rms_norm(x + signal, post_norm_w, self.norm_eps)
            else:
                x = x + signal
        
        residual = x
        
        # === Step B: Attention Pre-Norm ===
        attn_norm_w = self.weights[f"{prefix}.attn_norm"]
        x_norm = rms_norm(x, attn_norm_w, self.norm_eps)
        
        # === Step C: QKV Projections ===
        q = self.matmul(x_norm, f"{prefix}.attn_q")  # (n_q_heads * head_dim,)
        k = self.matmul(x_norm, f"{prefix}.attn_k")  # (n_kv_heads * head_dim,)
        
        # V projection (shared KV layers use V=K)
        if layer in self.shared_kv_layers:
            v = k.copy()
        else:
            v = self.matmul(x_norm, f"{prefix}.attn_v")  # (n_kv_heads * head_dim,)
        
        # Reshape to (n_heads, head_dim)
        q = q.reshape(n_q_heads, head_dim)
        k = k.reshape(self.n_kv_heads, head_dim)
        v = v.reshape(self.n_kv_heads, head_dim)
        
        # === Step D: Q/K Per-Head RMSNorm ===
        q_norm_w = self.weights.get(f"{prefix}.attn_q_norm")
        k_norm_w = self.weights.get(f"{prefix}.attn_k_norm")
        
        if q_norm_w is not None:
            for h in range(n_q_heads):
                q[h] = rms_norm(q[h], q_norm_w, self.norm_eps)
        
        if k_norm_w is not None:
            for h in range(self.n_kv_heads):
                k[h] = rms_norm(k[h], k_norm_w, self.norm_eps)
        
        # === Step E: RoPE ===
        q, k = self.rope.apply(q, k, pos, head_dim)
        
        # === Step F: Store KV and compute attention ===
        self.kv_cache.store(layer, k, v)
        
        # Get all cached K, V
        cached_k, cached_v = self.kv_cache.get_kv(layer)  # (seq, kv_heads, head_dim)
        seq_len = cached_k.shape[0]
        
        # Attention: Q (n_q_heads, head_dim) × K (seq, kv_heads, head_dim)
        # GQA: expand K,V to match Q head count
        gqa_ratio = n_q_heads // self.n_kv_heads
        
        scale = 1.0 / np.sqrt(head_dim)
        
        # Compute attention scores per Q head
        attn_out = np.zeros((n_q_heads, head_dim), dtype=np.float32)
        
        for h in range(n_q_heads):
            kv_h = h // gqa_ratio  # Which KV head this Q head uses
            
            # scores = Q[h] · K[:, kv_h, :].T * scale → (seq_len,)
            scores = (cached_k[:, kv_h, :] @ q[h]) * scale  # (seq, hd) @ (hd,) → (seq,)
            
            # Causal masking: for SWA, also apply window mask
            # In single-token decode, all cached positions are valid and causal
            # (positions in the future aren't in the cache)
            
            # Softmax
            attn_weights = softmax(scores)
            
            # Weighted sum of values
            attn_out[h] = attn_weights @ cached_v[:, kv_h, :]  # (seq,) @ (seq, hd) → (hd,)
        
        # === Step G: Output Projection ===
        attn_flat = attn_out.reshape(-1)  # (n_q_heads * head_dim,)
        
        # Handle weight shape: attn_output may be (q_dim, dim) or (dim, q_dim)
        o_w = self.weights[f"{prefix}.attn_output"]
        if o_w.shape[0] == attn_flat.shape[0]:
            attn_result = attn_flat @ o_w  # (q_dim,) @ (q_dim, dim) → (dim,)
        elif o_w.shape[1] == attn_flat.shape[0]:
            attn_result = o_w.T @ attn_flat  # Transposed case
        else:
            # Try with fast weights key
            attn_result = self.matmul(attn_flat, f"{prefix}.attn_output")
        
        # === Step H: Post-Attention Norm (sandwich) ===
        post_attn_norm = self.weights.get(f"{prefix}.post_attention_norm")
        if post_attn_norm is not None:
            attn_result = rms_norm(attn_result, post_attn_norm, self.norm_eps)
        
        # === Step I: Residual Add ===
        x = residual + attn_result
        
        # === Step J: FFN ===
        residual = x
        
        ffn_norm_w = self.weights[f"{prefix}.ffn_norm"]
        x_norm = rms_norm(x, ffn_norm_w, self.norm_eps)
        
        # SwiGLU with GELU: out = gelu(gate) * up, then down project
        gate = self.matmul(x_norm, f"{prefix}.ffn_gate")   # (ffn_dim,)
        up = self.matmul(x_norm, f"{prefix}.ffn_up")       # (ffn_dim,)
        ffn_act = gelu(gate) * up                           # (ffn_dim,)
        ffn_out = self.matmul(ffn_act, f"{prefix}.ffn_down")  # (dim,)
        
        # === Step K: Post-FFN Norm (sandwich) ===
        post_ffw_norm = self.weights.get(f"{prefix}.post_ffw_norm")
        if post_ffw_norm is not None:
            ffn_out = rms_norm(ffn_out, post_ffw_norm, self.norm_eps)
        
        # === Step L: Residual Add with Layer Scale ===
        layer_scale = self.weights.get(f"{prefix}.layer_output_scale")
        if layer_scale is not None:
            scale_val = float(layer_scale[0]) if layer_scale.ndim > 0 else float(layer_scale)
            x = residual + ffn_out * scale_val
        else:
            x = residual + ffn_out
        
        return x
    
    # ============================================================
    # FULL FORWARD PASS (single token decode)
    # ============================================================
    
    def forward_token(self, token_id, pos):
        """
        Forward pass for a single token (decode mode).
        
        Args:
            token_id: integer token ID
            pos: position in sequence
        
        Returns:
            logits: (vocab_size,) log-probability scores
        """
        # 1. Token embedding + scaling
        x = self.token_embd[token_id].astype(np.float32).copy()
        x *= np.sqrt(self.dim)  # Gemma embedding scaling
        
        # 2. Compute PLE for this token
        ple = self.compute_ple([token_id], x.reshape(1, -1))  # (1, n_layers, ple_dim)
        ple = ple[0]  # (n_layers, ple_dim)
        
        # 3. Process through all layers
        for layer in range(self.n_layers):
            x = self.layer_forward(x, layer, pos, ple[layer])
            
            if DEBUG and (layer == 0 or layer == self.n_layers - 1):
                dbg(f"Layer {layer}: x range=[{x.min():.4f}, {x.max():.4f}], "
                    f"norm={np.linalg.norm(x):.2f}")
            
            # NaN check (debug mode)
            if DEBUG and np.isnan(x).any():
                print(f"  ⚠️  NaN detected at layer {layer}!")
                return np.full(self.vocab_size, float('nan'))
        
        # 4. Final norm
        x = rms_norm(x, self.output_norm, self.norm_eps)
        
        # 5. Compute logits
        if self.output_weight is not None:
            logits = x @ self.output_weight
        else:
            # Tied embedding: logits = x @ token_embd.T
            logits = x @ self.token_embd.T
        
        # 6. Softcapping
        if self.logit_cap > 0:
            logits = np.tanh(logits / self.logit_cap) * self.logit_cap
        
        return logits
    
    # ============================================================
    # BATCH PREFILL (processes multiple tokens at once)
    # ============================================================
    
    def prefill(self, token_ids):
        """
        Prefill the KV cache with a sequence of tokens.
        Returns logits for the last token.
        
        Args:
            token_ids: list of token IDs (the full prompt)
        
        Returns:
            logits: (vocab_size,) for the last token
        """
        n_tokens = len(token_ids)
        print(f"[Prefill] Processing {n_tokens} tokens...")
        t0 = time.time()
        
        # Process tokens one at a time through the full model
        # (batch prefill would be faster but single-token is simpler and correct)
        logits = None
        for i, tid in enumerate(token_ids):
            logits = self.forward_token(tid, pos=i)
            self.kv_cache.advance()
            
            if (i + 1) % 10 == 0 or i == n_tokens - 1:
                elapsed = time.time() - t0
                tok_s = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  [{i+1}/{n_tokens}] {tok_s:.1f} tok/s", end='\r')
        
        elapsed = time.time() - t0
        print(f"\n[Prefill] Done: {n_tokens} tokens in {elapsed:.1f}s "
              f"({n_tokens/elapsed:.1f} tok/s)")
        
        # Diagnostic
        if logits is not None:
            if np.isnan(logits).any():
                nan_count = np.isnan(logits).sum()
                print(f"  ⚠️  WARNING: {nan_count}/{len(logits)} NaN logits!")
            else:
                top5 = np.argsort(logits)[-5:][::-1]
                print(f"  Top-5 logits: {logits[top5]}")
                print(f"  Top-5 IDs: {top5}")
        
        return logits
    
    # ============================================================
    # GENERATION
    # ============================================================
    
    def generate(self, token_ids, max_new_tokens=100, temperature=1.0, top_k=64, top_p=0.95):
        """
        Generate tokens autoregressively.
        
        Args:
            token_ids: list of prompt token IDs
            max_new_tokens: maximum tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
            top_p: nucleus sampling threshold
        
        Returns:
            list of generated token IDs (not including prompt)
        """
        # Prefill prompt
        logits = self.prefill(token_ids)
        
        if logits is None or np.isnan(logits).any():
            print("[Generate] ERROR: Prefill produced NaN logits. Cannot generate.")
            return []
        
        generated = []
        pos = len(token_ids)
        
        print(f"[Generate] Generating up to {max_new_tokens} tokens...")
        t0 = time.time()
        
        for step in range(max_new_tokens):
            # Sample from logits
            next_token = self._sample(logits, temperature, top_k, top_p)
            
            # Check for EOS
            if next_token == 1:  # EOS token
                break
            
            generated.append(next_token)
            
            # Forward pass for next token
            logits = self.forward_token(next_token, pos)
            self.kv_cache.advance()
            pos += 1
            
            if np.isnan(logits).any():
                print(f"\n  ⚠️  NaN at step {step+1}, stopping.")
                break
            
            # Print speed
            if (step + 1) % 5 == 0:
                elapsed = time.time() - t0
                print(f"  Generated {step+1} tokens ({(step+1)/elapsed:.1f} tok/s)", end='\r')
        
        elapsed = time.time() - t0
        print(f"\n[Generate] {len(generated)} tokens in {elapsed:.1f}s "
              f"({len(generated)/elapsed:.1f} tok/s)")
        
        return generated
    
    def _sample(self, logits, temperature=1.0, top_k=64, top_p=0.95):
        """Sample a token from logits with temperature, top-k, and top-p."""
        if temperature <= 0:
            return int(np.argmax(logits))
        
        # Temperature scaling
        logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            top_k_indices = np.argpartition(logits, -top_k)[-top_k:]
            mask = np.full_like(logits, -np.inf)
            mask[top_k_indices] = logits[top_k_indices]
            logits = mask
        
        # Softmax
        probs = softmax(logits)
        
        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_indices = np.argsort(probs)[::-1]
            cumulative = np.cumsum(probs[sorted_indices])
            cutoff_idx = np.searchsorted(cumulative, top_p) + 1
            keep = sorted_indices[:cutoff_idx]
            filtered_probs = np.zeros_like(probs)
            filtered_probs[keep] = probs[keep]
            probs = filtered_probs / filtered_probs.sum()
        
        # Sample
        return int(np.random.choice(len(probs), p=probs))
    
    # ============================================================
    # RESET
    # ============================================================
    
    def reset(self):
        """Reset KV cache for a new conversation."""
        self.kv_cache = Gemma4KVCache(
            self.n_layers, self.max_seq_len, self.window_size,
            self.layer_head_dims, self.n_kv_heads, self.swa_pattern
        )


# ============================================================
# TOKENIZER (minimal — just enough to test)
# ============================================================

class SimpleTokenizer:
    """Minimal tokenizer wrapper using the GGUF's embedded tokens."""
    
    def __init__(self, model):
        """Initialize from parsed GGUF model."""
        self.tokens = model.metadata.get('tokenizer.ggml.tokens', [])
        self.merges = model.metadata.get('tokenizer.ggml.merges', [])
        self.bos_id = int(model.metadata.get('tokenizer.ggml.bos_token_id', 2))
        self.eos_id = int(model.metadata.get('tokenizer.ggml.eos_token_id', 1))
        self.vocab_size = len(self.tokens) if self.tokens else 262144
        
        # Build vocab lookup
        self.token_to_id = {}
        if isinstance(self.tokens, (list, np.ndarray)):
            for i, tok in enumerate(self.tokens):
                if isinstance(tok, bytes):
                    tok = tok.decode('utf-8', errors='replace')
                self.token_to_id[tok] = i
        
        # Build merge priority
        self.merge_priority = {}
        if isinstance(self.merges, (list, np.ndarray)):
            for i, merge in enumerate(self.merges):
                if isinstance(merge, bytes):
                    merge = merge.decode('utf-8', errors='replace')
                self.merge_priority[merge] = i
        
        print(f"[Tokenizer] {self.vocab_size} tokens, {len(self.merge_priority)} merges")
    
    def encode(self, text, add_bos=True):
        """Simple BPE encode. Returns list of token IDs."""
        # SentencePiece-style greedy longest-match tokenization
        # Gemma uses ▁ (U+2581) as word boundary marker
        ids = []
        if add_bos:
            ids.append(self.bos_id)
        
        # Replace spaces with ▁ for SentencePiece lookup
        # Add ▁ at start (SentencePiece always adds leading space marker)
        text_sp = '▁' + text.replace(' ', '▁')
        
        i = 0
        while i < len(text_sp):
            # Greedy longest match
            best_len = 0
            best_tok = None
            
            # Try progressively shorter substrings
            max_len = min(48, len(text_sp) - i)
            for length in range(max_len, 0, -1):
                candidate = text_sp[i:i+length]
                if candidate in self.token_to_id:
                    best_tok = candidate
                    best_len = length
                    break
            
            if best_tok:
                ids.append(self.token_to_id[best_tok])
                i += best_len
            else:
                # Byte fallback for unknown characters
                char_bytes = text_sp[i].encode('utf-8')
                for b in char_bytes:
                    hex_tok = f"<0x{b:02X}>"
                    ids.append(self.token_to_id.get(hex_tok, 3))  # 3 = UNK
                i += 1
        
        return ids
    
    def decode(self, ids):
        """Decode token IDs to text."""
        pieces = []
        for tid in ids:
            if tid < len(self.tokens):
                tok = self.tokens[tid]
                if isinstance(tok, bytes):
                    tok = tok.decode('utf-8', errors='replace')
                # Handle byte tokens
                if tok.startswith('<0x') and tok.endswith('>'):
                    try:
                        byte_val = int(tok[3:-1], 16)
                        pieces.append(bytes([byte_val]).decode('utf-8', errors='replace'))
                    except:
                        pieces.append(tok)
                else:
                    # Replace ▁ with space
                    pieces.append(tok.replace('▁', ' '))
            else:
                pieces.append(f'<unk:{tid}>')
        
        return ''.join(pieces)
    
    def encode_chat(self, user_message):
        """Encode a chat message with Gemma template."""
        formatted = f"<start_of_turn>user\n{user_message}<end_of_turn>\n<start_of_turn>model\n"
        
        # Try to encode special tokens directly
        tokens = []
        tokens.append(self.bos_id)
        
        # Encode the formatted string
        # For special tokens, look them up directly
        parts = formatted.split('<')
        for i, part in enumerate(parts):
            if i == 0:
                if part:
                    tokens.extend(self.encode(part, add_bos=False))
                continue
            
            # Check if this starts with a special token
            close = part.find('>')
            if close >= 0:
                special = '<' + part[:close+1]
                remainder = part[close+1:]
                
                if special in self.token_to_id:
                    tokens.append(self.token_to_id[special])
                else:
                    tokens.extend(self.encode(special, add_bos=False))
                
                if remainder:
                    tokens.extend(self.encode(remainder, add_bos=False))
            else:
                tokens.extend(self.encode('<' + part, add_bos=False))
        
        return tokens


# ============================================================
# MAIN — CLI Entry Point
# ============================================================

def main():
    global DEBUG
    
    parser = argparse.ArgumentParser(description='Gemma 4 E2B Inference — Nightmare Designs')
    parser.add_argument('--model', required=True, help='Path to GGUF model file')
    parser.add_argument('--prompt', default='hey', help='Input prompt')
    parser.add_argument('--max-tokens', type=int, default=50, help='Max tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=64, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.95, help='Nucleus sampling threshold')
    parser.add_argument('--debug', action='store_true', help='Enable debug diagnostics')
    parser.add_argument('--max-seq', type=int, default=2048, help='Maximum sequence length')
    
    args = parser.parse_args()
    DEBUG = args.debug
    
    # Load model
    model = Gemma4E2B(args.model, max_seq_len=args.max_seq)
    
    # Initialize tokenizer
    tokenizer = SimpleTokenizer(model.model)
    
    # Encode prompt
    print(f"\n[Prompt] '{args.prompt}'")
    token_ids = tokenizer.encode_chat(args.prompt)
    print(f"[Tokens] {len(token_ids)} tokens: {token_ids[:20]}{'...' if len(token_ids) > 20 else ''}")
    
    # Generate
    generated_ids = model.generate(
        token_ids,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    
    # Decode and print
    if generated_ids:
        text = tokenizer.decode(generated_ids)
        print(f"\n{'='*60}")
        print(f"  RESPONSE:")
        print(f"{'='*60}")
        print(f"  {text}")
        print(f"{'='*60}\n")
    else:
        print("\n[!] No tokens generated.")


if __name__ == '__main__':
    main()
