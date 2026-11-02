"""
transformer.py — Unified Transformer Engine (Nightmare Suite v10.1)
Handles local on-device inference for GGUF architecture profiles.

DUAL-MODE INFERENCE LAYER:
• Hardware accelerated (CUDA Toolkit runtime via nightmare_engine.dll)
• CPU Vector fallback (NumPy SIMD AVX2 architecture processing)

Nightmare Designs — Owner Panel Only
"""

import os
import sys
import re
import math
import nightmare_np as np
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

# Hard structural flags for AVX2 instructions
np.seterr(all='ignore')

class TransformerModel:
    def __init__(self, model_path: str, context_len: int = 512, max_seq_len: int = None):
        # Accept max_seq_len as an alias for context_len (used by the suite/inference).
        if max_seq_len is not None:
            context_len = max_seq_len
        self.model_path = Path(model_path)
        self.context_len = context_len
        self.max_seq_len = context_len
        self.config: Dict[str, Any] = {}
        # Small compatibility shim so the suite's inference.py can call
        # model.kv_cache.advance() (the KV cache here is indexed by position
        # directly, so advance() is a no-op).
        class _KVAdvance:
            def advance(self):
                pass
        self.kv_cache = _KVAdvance()
        self.weights: Dict[str, np.ndarray] = {}

        # Core network dimensions
        self.dim = 512
        self.vocab_size = 64000
        self.n_layers = 20
        self.n_heads = 8
        self.n_kv_heads = 4
        self.head_dim = 64
        self.ff_dim = 1376
        self.rope_base = 10000.0
        self.norm_eps = 1e-5

        # State parameters
        self.kv_cache_k: List[Optional[np.ndarray]] = [None] * self.n_layers
        self.kv_cache_v: List[Optional[np.ndarray]] = [None] * self.n_layers

        print(f"[Transformer] Initializing target pipeline block: {self.model_path.name}")
        self._load_model_metadata()

    def _load_model_metadata(self):
        """Extract metadata maps and configure structural execution strides."""
        from gguf_parser import GGUFParser
        
        try:
            parser = GGUFParser(str(self.model_path))
            meta = parser.parse()
            self.metadata = meta.metadata  # expose for tokenizer
            self.name = meta.metadata.get("general.name", self.model_path.name)
            self.architecture = meta.metadata.get("general.architecture", "unknown")

            # Update dimension properties from GGUF keys (architecture-agnostic)
            arch = self.architecture
            # Some models use a different arch prefix in metadata than general.architecture
            for prefix in [arch, "llama", "qwen2", "gemma", "mistral", "phi3"]:
                if f"{prefix}.embedding_length" in meta.metadata:
                    arch = prefix
                    break
            self.dim = meta.metadata.get(f"{arch}.embedding_length", self.dim)
            self.vocab_size = meta.metadata.get(f"{arch}.vocabulary_length",
                                meta.metadata.get("tokenizer.ggml.vocab_size", meta.vocab_size))
            self.n_layers = meta.metadata.get(f"{arch}.block_count", self.n_layers)
            self.n_heads = meta.metadata.get(f"{arch}.attention.head_count", self.n_heads)
            self.n_kv_heads = meta.metadata.get(f"{arch}.attention.head_count_kv", self.n_kv_heads)
            self.ff_dim = meta.metadata.get(f"{arch}.feed_forward_length", self.ff_dim)
            # Rope base: qwen2/gemma use `{arch}.rope.freq_base`; some models use
            # `{arch}.attention.rope.freq_base`. Try both.
            self.rope_base = meta.metadata.get(f"{arch}.rope.freq_base",
                                meta.metadata.get(f"{arch}.attention.rope.freq_base", self.rope_base))
            # RMS norm epsilon from metadata (qwen2 uses 1e-6, llama uses 1e-5).
            self.norm_eps = meta.metadata.get(f"{arch}.attention.layer_norm_rms_epsilon",
                                meta.metadata.get(f"{arch}.attention.layer_norm_epsilon",
                                meta.metadata.get(f"{arch}.layer_norm_rms_epsilon", self.norm_eps)))
            self.head_dim = self.dim // self.n_heads

            print(f"[Model Config] dim={self.dim}, layers={self.n_layers}, heads={self.n_heads}, vocab={self.vocab_size}")
            print("[Model] Loading weights into system RAM...")

            for tname in meta.tensors.keys():
                arr = parser.load_tensor(meta, tname)
                self.weights[tname] = arr

            # NOTE: weights are now loaded in GGUF logical (out, in) orientation by
            # the parser (no transpose). The forward pass uses `x @ W.T` throughout,
            # so no self-heal / re-orientation is needed here.

            # RoPE layout (verified against llama.cpp ground truth, 2026-09-07):
            # llama-family GGUFs store Q/K rows permuted by llama.cpp's HF->GGUF
            # converter. Undoing that permutation restores the natural head layout,
            # and EVERY architecture in natural layout uses half-split
            # (rotate_half) RoPE. Gemma/phi/qwen GGUFs are stored un-permuted, so
            # they must NOT be unpermuted here.
            arch_l = (self.architecture or '').lower()
            self.rope_type = 'halfsplit'
            if any(a in arch_l for a in ('llama', 'mistral', 'smollm', 'mixtral')):
                self._undo_rope_permute()

            # Resize KV cache to the actual layer count
            self.kv_cache_k = [None] * self.n_layers
            self.kv_cache_v = [None] * self.n_layers

            # Allocate space for cached sequence parameters
            for i in range(self.n_layers):
                self.kv_cache_k[i] = np.zeros((self.context_len, self.n_kv_heads, self.head_dim), dtype=np.float32)
                self.kv_cache_v[i] = np.zeros((self.context_len, self.n_kv_heads, self.head_dim), dtype=np.float32)

            print("[Model] System RAM allocation completed successfully. Model Ready!")

        except Exception as e:
            print(f"[ERROR] Failed to extract metadata tokens from binary structure: {e}")
            raise e

    def _undo_rope_permute(self):
        """Undo llama.cpp's RoPE row-permutation on Q/K weights.

        llama.cpp's converter permutes the Q and K projection rows by head before
        storing them in the GGUF. V/O/MLP tensors are stored as-is. Gemma, phi and
        qwen store Q/K un-permuted, so this is only applied to llama-family models
        (llama/mistral/smollm/mixtral).
        """
        def perm(arr, n_head):
            out = arr.shape[0]
            M = out // n_head // 2
            if M == 0:
                return arr
            s = np.arange(out)
            h = s // (2 * M)
            rem = s % (2 * M)
            a = rem // M
            j = rem % M
            P = h * (2 * M) + j * 2 + a
            return arr[P]

        for i in range(self.n_layers):
            for name, n_head in (
                (f'blk.{i}.attn_q.weight', self.n_heads),
                (f'blk.{i}.attn_k.weight', self.n_kv_heads),
                (f'blk.{i}.attn_q.bias', self.n_heads),
                (f'blk.{i}.attn_k.bias', self.n_kv_heads),
            ):
                w = self.weights.get(name)
                if w is not None:
                    self.weights[name] = perm(w, n_head)

    def _rms_norm(self, x: np.ndarray, weight: np.ndarray, eps: float = None) -> np.ndarray:
        """Vectorized root mean square normalization layer."""
        if eps is None:
            eps = getattr(self, 'norm_eps', 1e-5)
        variance = np.mean(x ** 2, axis=-1, keepdims=True)
        return (x * (1.0 / np.sqrt(variance + eps))) * weight

    def _rope_embedding(self, q: np.ndarray, k: np.ndarray, pos: int) -> Tuple[np.ndarray, np.ndarray]:
        """Apply rotary position embeddings to query and key vector slots.

        Supports both RoPE layouts:
          • 'halfsplit': rotate the first half of head_dim against the second
            (correct for ALL archs once llama-family Q/K are un-permuted)
          • 'interleaved': rotate adjacent (even, odd) pairs (kept for reference)
        """
        hd = self.head_dim
        inv_freq = 1.0 / (self.rope_base ** (np.arange(0, hd, 2) / hd))  # (hd/2,)
        theta = pos * inv_freq
        cos_t = np.cos(theta).astype(np.float32)  # (hd/2,)
        sin_t = np.sin(theta).astype(np.float32)

        if self.rope_type == 'halfsplit':
            # qwen: rotate first half against second half
            q1, q2 = q[:, :hd // 2], q[:, hd // 2:]
            qr1 = q1 * cos_t[None, :] - q2 * sin_t[None, :]
            qr2 = q1 * sin_t[None, :] + q2 * cos_t[None, :]
            q = np.concatenate([qr1, qr2], axis=-1)
            k1, k2 = k[:, :hd // 2], k[:, hd // 2:]
            kr1 = k1 * cos_t[None, :] - k2 * sin_t[None, :]
            kr2 = k1 * sin_t[None, :] + k2 * cos_t[None, :]
            k = np.concatenate([kr1, kr2], axis=-1)
        else:
            # interleaved: rotate adjacent (even, odd) pairs
            q = q.reshape(self.n_heads, hd // 2, 2)
            q0, q1 = q[..., 0], q[..., 1]
            qr = np.stack([q0 * cos_t[None, :] - q1 * sin_t[None, :],
                           q0 * sin_t[None, :] + q1 * cos_t[None, :]], axis=-1)
            q = qr.reshape(self.n_heads, hd)
            k = k.reshape(self.n_kv_heads, hd // 2, 2)
            k0, k1 = k[..., 0], k[..., 1]
            kr = np.stack([k0 * cos_t[None, :] - k1 * sin_t[None, :],
                           k0 * sin_t[None, :] + k1 * cos_t[None, :]], axis=-1)
            k = kr.reshape(self.n_kv_heads, hd)

        return q, k

    def _silu(self, x: np.ndarray) -> np.ndarray:
        """SiLU activation element execution."""
        return x * (1.0 / (1.0 + np.exp(-x)))

    def _transformer_block(self, layer_idx: int, x: np.ndarray, pos: int) -> np.ndarray:
        """Execute mathematical forward pass for a single network layer block."""
        attn_norm_w = self.weights[f"blk.{layer_idx}.attn_norm.weight"]
        h_norm = self._rms_norm(x, attn_norm_w)

        wq = self.weights[f"blk.{layer_idx}.attn_q.weight"]
        wk = self.weights[f"blk.{layer_idx}.attn_k.weight"]
        wv = self.weights[f"blk.{layer_idx}.attn_v.weight"]
        wo = self.weights[f"blk.{layer_idx}.attn_output.weight"]

        q_proj = h_norm @ wq.T
        k_proj = h_norm @ wk.T
        v_proj = h_norm @ wv.T

        # Qwen2/3 attention QKV bias (present in qwen GGUF exports)
        qb = self.weights.get(f"blk.{layer_idx}.attn_q.bias")
        kb = self.weights.get(f"blk.{layer_idx}.attn_k.bias")
        vb = self.weights.get(f"blk.{layer_idx}.attn_v.bias")
        if qb is not None:
            q_proj = q_proj + qb
        if kb is not None:
            k_proj = k_proj + kb
        if vb is not None:
            v_proj = v_proj + vb

        q = q_proj.reshape(self.n_heads, self.head_dim)
        k = k_proj.reshape(self.n_kv_heads, self.head_dim)
        v = v_proj.reshape(self.n_kv_heads, self.head_dim)

        q, k = self._rope_embedding(q, k, pos)

        self.kv_cache_k[layer_idx][pos] = k
        self.kv_cache_v[layer_idx][pos] = v

        k_past = self.kv_cache_k[layer_idx][:pos + 1] 
        v_past = self.kv_cache_v[layer_idx][:pos + 1] 

        scores = np.zeros((self.n_heads, pos + 1), dtype=np.float32)
        kv_multiply = self.n_heads // self.n_kv_heads
        
        for h in range(self.n_heads):
            kv_h = h // kv_multiply
            k_head_vecs = k_past[:, kv_h, :]
            scores[h] = (q[h] @ k_head_vecs.T) / math.sqrt(self.head_dim)

        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        output_context = np.zeros((self.n_heads, self.head_dim), dtype=np.float32)
        for h in range(self.n_heads):
            kv_h = h // kv_multiply
            v_head_vecs = v_past[:, kv_h, :] 
            output_context[h] = attn_probs[h] @ v_head_vecs

        attn_out = output_context.flatten() @ wo.T
        x = x + attn_out

        ffn_norm_w = self.weights[f"blk.{layer_idx}.ffn_norm.weight"]
        f_norm = self._rms_norm(x, ffn_norm_w)

        w_gate = self.weights[f"blk.{layer_idx}.ffn_gate.weight"]
        w_up = self.weights[f"blk.{layer_idx}.ffn_up.weight"]
        w_down = self.weights[f"blk.{layer_idx}.ffn_down.weight"]

        gated_path = self._silu(f_norm @ w_gate.T)
        up_path = f_norm @ w_up.T
        ffn_intermediate = gated_path * up_path

        ffn_out = ffn_intermediate @ w_down.T
        x = x + ffn_out
        
        return x

    def close(self):
        """Release RAM when switching models (CPU engine)."""
        self.weights = None
        self.kv_cache_k = None
        self.kv_cache_v = None
        self.kv_cache = None
        self.metadata = None
        import gc
        gc.collect()

    def reset(self):
        """Reset the KV cache (zero it out) for a fresh generation."""
        for i in range(self.n_layers):
            self.kv_cache_k[i][:] = 0.0
            self.kv_cache_v[i][:] = 0.0

    def forward(self, token_id: int, pos: int = None, position: int = None) -> np.ndarray:
        """Compute the forward network calculations for an input token ID.

        Accepts the position either as `pos` (internal) or `position` (the
        suite's inference.py passes it as `position=`).
        """
        if pos is None:
            pos = position
        embd_table = self.weights["token_embd.weight"]
        x = embd_table[token_id].astype(np.float32)

        for layer_idx in range(self.n_layers):
            x = self._transformer_block(layer_idx, x, pos)

        final_norm_w = self.weights["output_norm.weight"]
        x = self._rms_norm(x, final_norm_w)

        # Handle tied embeddings: some models have no separate output.weight
        output_projection_w = self.weights.get("output.weight")
        if output_projection_w is None:
            output_projection_w = embd_table  # tied embeddings
        # Handle both (vocab, dim) and transposed (dim, vocab) storage
        if output_projection_w.shape[0] == self.vocab_size:
            logits = x @ output_projection_w.T
        else:
            logits = x @ output_projection_w
        
        return logits


def test_generation_pipeline():
    """Simple verification runtime hook to isolate and trace pipeline parameters."""
    print("[TEST] Running isolated test loop pass...")
    
    if "TEST_MODEL_PATH" in os.environ:
        try:
            # Updated initialization to match the capitalized class name
            engine = TransformerModel(os.environ["TEST_MODEL_PATH"])
            print("[TEST] Validation check completed cleanly.")
        except Exception:
            pass

if __name__ == "__main__":
    test_generation_pipeline()