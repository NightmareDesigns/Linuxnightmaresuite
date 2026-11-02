"""
💀 NIGHTMARE MOE ENGINE — Mixture of Experts Forward Pass
============================================================
Handles: Mixtral 8x7B/8x22B, DeepSeek v2/v3, any MoE GGUF model.

MoE Architecture:
  Standard transformer layers BUT the FFN is replaced with a router
  that selects top-K experts (each expert = independent FFN).

  Layer forward:
    1. Attention (same as LLaMA)
    2. Router: logits = x @ gate_weight → select top-K experts
    3. For each selected expert: expert_out = FFN_i(x)
    4. Combine: out = sum(softmax(router_scores) * expert_outputs)

Supports:
  • Mixtral-style (8 experts, top-2 routing, no shared expert)
  • DeepSeek-style (N routed experts + 1 shared expert, top-K routing)
  • Variable expert count, variable top-K

Author: Nightmare Designs 💀
"""

import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple


# ════════════════════════════════════════════════════════════════
# MATH PRIMITIVES (same as transformer.py — no external deps)
# ════════════════════════════════════════════════════════════════

def rms_norm(x, weight, eps=1e-6, gemma_style=False):
    """RMSNorm — optionally with (1+weight) for Gemma."""
    ms = np.mean(x * x) + eps
    normed = x / np.sqrt(ms)
    if gemma_style:
        return normed * (1.0 + weight)
    return normed * weight


def silu(x):
    """SiLU/Swish activation."""
    return x / (1.0 + np.exp(-np.clip(x, -88, 88)))


def softmax(x):
    """Numerically stable softmax."""
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)


def apply_rope(q, k, pos, head_dim, base=10000.0):
    """Apply Rotary Position Embeddings to Q and K vectors."""
    half = head_dim // 2
    q_out = q.copy()
    k_out = k.copy()
    for i in range(half):
        freq = 1.0 / (base ** (2.0 * i / head_dim))
        angle = pos * freq
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        # Q
        q_out[i] = q[i] * cos_a - q[i + half] * sin_a
        q_out[i + half] = q[i] * sin_a + q[i + half] * cos_a
        # K
        k_out[i] = k[i] * cos_a - k[i + half] * sin_a
        k_out[i + half] = k[i] * sin_a + k[i + half] * cos_a
    return q_out, k_out


# ════════════════════════════════════════════════════════════════
# MOE ROUTER
# ════════════════════════════════════════════════════════════════

class MoERouter:
    """Top-K expert routing with load balancing."""

    def __init__(self, gate_weight: np.ndarray, n_experts: int, top_k: int = 2):
        """
        gate_weight: (hidden_dim, n_experts) — router projection
        n_experts: total number of experts
        top_k: how many experts to route to per token
        """
        self.gate_weight = gate_weight
        self.n_experts = n_experts
        self.top_k = top_k

    def route(self, x: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """
        Route input x to top-K experts.
        
        Returns:
            expert_ids: list of K expert indices
            expert_weights: normalized weights for each selected expert
        """
        # Router logits
        logits = x @ self.gate_weight  # (n_experts,)

        # Top-K selection
        top_k_ids = np.argsort(logits)[-self.top_k:][::-1]

        # Softmax over selected experts only (normalized routing weights)
        top_k_logits = logits[top_k_ids]
        weights = softmax(top_k_logits)

        return top_k_ids.tolist(), weights


# ════════════════════════════════════════════════════════════════
# MOE FFN (single expert)
# ════════════════════════════════════════════════════════════════

class ExpertFFN:
    """Single expert FFN (gate-up-down with SiLU, same as LLaMA FFN)."""

    def __init__(self, w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray):
        """
        w_gate: (hidden_dim, dim) — gate projection
        w_up:   (hidden_dim, dim) — up projection
        w_down: (dim, hidden_dim) — down projection
        """
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def forward(self, x: np.ndarray) -> np.ndarray:
        """FFN forward: down(silu(gate(x)) * up(x))"""
        gate = x @ self.w_gate.T
        up = x @ self.w_up.T
        return (silu(gate) * up) @ self.w_down.T


# ════════════════════════════════════════════════════════════════
# MOE LAYER
# ════════════════════════════════════════════════════════════════

class MoELayer:
    """
    Full MoE layer: router + experts + optional shared expert.
    
    Mixtral: 8 experts, top-2, no shared expert
    DeepSeek: 64 routed + 1 shared, top-6
    """

    def __init__(self, router: MoERouter, experts: List[ExpertFFN],
                 shared_expert: Optional[ExpertFFN] = None):
        self.router = router
        self.experts = experts
        self.shared_expert = shared_expert

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        MoE FFN forward:
        1. Route x to top-K experts
        2. Run selected experts
        3. Weighted sum of expert outputs
        4. Add shared expert if present (DeepSeek style)
        """
        expert_ids, weights = self.router.route(x)

        # Compute weighted sum of expert outputs
        output = np.zeros_like(x)
        for i, (eid, w) in enumerate(zip(expert_ids, weights)):
            expert_out = self.experts[eid].forward(x)
            output += w * expert_out

        # Shared expert (DeepSeek v2/v3 — always active, no gating)
        if self.shared_expert is not None:
            output += self.shared_expert.forward(x)

        return output


# ════════════════════════════════════════════════════════════════
# MOE TRANSFORMER MODEL
# ════════════════════════════════════════════════════════════════

class MoETransformerModel:
    """
    Full MoE Transformer model.
    
    Architecture: Standard transformer with MoE FFN layers.
    Attention is identical to LLaMA (GQA, RoPE, KV cache).
    Only the FFN is replaced with router + experts.
    
    Loads from GGUF format. Tensor naming:
      Mixtral:  blk.{l}.ffn_gate_exps.{e}.weight
      DeepSeek: blk.{l}.ffn_gate_exps.weight (merged) + shared expert
    """

    def __init__(self, weights: Dict, config: Dict):
        """
        weights: dict of tensor name → numpy array
        config: model configuration dict
        """
        self.dim = config['dim']
        self.n_layers = config['n_layers']
        self.n_heads = config['n_heads']
        self.n_kv_heads = config.get('n_kv_heads', self.n_heads)
        self.head_dim = config.get('head_dim', self.dim // self.n_heads)
        self.hidden_dim = config['hidden_dim']
        self.vocab_size = config['vocab_size']
        self.n_experts = config.get('n_experts', 8)
        self.top_k = config.get('top_k_experts', 2)
        self.has_shared_expert = config.get('has_shared_expert', False)
        self.rope_base = config.get('rope_base', 10000.0)
        self.max_seq_len = config.get('max_seq_len', 4096)
        self.norm_eps = config.get('norm_eps', 1e-5)

        self.weights = weights
        self.n_rep = self.n_heads // self.n_kv_heads

        # KV cache
        kv_dim = self.n_kv_heads * self.head_dim
        self.kv_k = np.zeros((self.n_layers, self.max_seq_len, kv_dim), dtype=np.float32)
        self.kv_v = np.zeros((self.n_layers, self.max_seq_len, kv_dim), dtype=np.float32)
        self.pos = 0

        # Build MoE layers
        self.moe_layers = []
        for l in range(self.n_layers):
            # Router
            gate_w = self._get_weight(f'blk.{l}.ffn_gate_inp.weight')  # (dim, n_experts)
            if gate_w is not None:
                router = MoERouter(gate_w, self.n_experts, self.top_k)

                # Experts
                experts = []
                for e in range(self.n_experts):
                    w_gate = self._get_weight(f'blk.{l}.ffn_gate.{e}.weight')
                    w_up = self._get_weight(f'blk.{l}.ffn_up.{e}.weight')
                    w_down = self._get_weight(f'blk.{l}.ffn_down.{e}.weight')
                    if w_gate is not None and w_up is not None and w_down is not None:
                        experts.append(ExpertFFN(w_gate, w_up, w_down))

                # Shared expert (DeepSeek)
                shared = None
                if self.has_shared_expert:
                    sg = self._get_weight(f'blk.{l}.ffn_gate_shexp.weight')
                    su = self._get_weight(f'blk.{l}.ffn_up_shexp.weight')
                    sd = self._get_weight(f'blk.{l}.ffn_down_shexp.weight')
                    if sg is not None:
                        shared = ExpertFFN(sg, su, sd)

                self.moe_layers.append(MoELayer(router, experts, shared))
            else:
                # Dense layer (some MoE models have dense layers too)
                self.moe_layers.append(None)

        print(f"[MoE] Built: {self.n_layers} layers, {self.n_experts} experts, top-{self.top_k}")
        moe_count = sum(1 for m in self.moe_layers if m is not None)
        print(f"[MoE] MoE layers: {moe_count}, Dense layers: {self.n_layers - moe_count}")

    def _get_weight(self, name: str) -> Optional[np.ndarray]:
        """Get a weight by name, return None if not found."""
        return self.weights.get(name, None)

    def forward(self, token_id: int, position: int) -> np.ndarray:
        """
        Forward pass — single token.
        
        Returns logits (vocab_size,)
        """
        # 1. Embedding
        x = self.weights['token_embd.weight'][token_id].astype(np.float32)

        # 2. Transformer layers
        for l in range(self.n_layers):
            x = self._layer_forward(x, l, position)

        # 3. Final norm
        x = rms_norm(x, self.weights['output_norm.weight'], self.norm_eps)

        # 4. Output projection
        output_w = self.weights.get('output.weight', self.weights['token_embd.weight'])
        logits = x @ output_w.T  # (vocab_size,)

        return logits

    def _layer_forward(self, x: np.ndarray, layer: int, position: int) -> np.ndarray:
        """Single transformer layer with MoE FFN."""
        pf = f'blk.{layer}'

        # ═══ ATTENTION (same as LLaMA — GQA + RoPE) ═══
        x_norm = rms_norm(x, self.weights[f'{pf}.attn_norm.weight'], self.norm_eps)

        # Q, K, V projections
        q = x_norm @ self.weights[f'{pf}.attn_q.weight'].T
        k = x_norm @ self.weights[f'{pf}.attn_k.weight'].T
        v = x_norm @ self.weights[f'{pf}.attn_v.weight'].T

        # Apply RoPE per head
        for h in range(self.n_heads):
            s, e = h * self.head_dim, (h + 1) * self.head_dim
            freq_base = self.rope_base
            half = self.head_dim // 2
            for i in range(half):
                freq = 1.0 / (freq_base ** (2.0 * i / self.head_dim))
                angle = position * freq
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                q0, q1 = q[s + i], q[s + i + half]
                q[s + i] = q0 * cos_a - q1 * sin_a
                q[s + i + half] = q0 * sin_a + q1 * cos_a

        for h in range(self.n_kv_heads):
            s, e = h * self.head_dim, (h + 1) * self.head_dim
            half = self.head_dim // 2
            for i in range(half):
                freq = 1.0 / (self.rope_base ** (2.0 * i / self.head_dim))
                angle = position * freq
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                k0, k1 = k[s + i], k[s + i + half]
                k[s + i] = k0 * cos_a - k1 * sin_a
                k[s + i + half] = k0 * sin_a + k1 * cos_a

        # KV cache
        self.kv_k[layer, position] = k
        self.kv_v[layer, position] = v

        # Multi-head attention with GQA
        attn_out = np.zeros(self.n_heads * self.head_dim, dtype=np.float32)
        for h in range(self.n_heads):
            kv_h = h // self.n_rep
            q_h = q[h * self.head_dim:(h + 1) * self.head_dim]

            # Attention scores
            scores = np.zeros(position + 1, dtype=np.float32)
            for p in range(position + 1):
                k_p = self.kv_k[layer, p, kv_h * self.head_dim:(kv_h + 1) * self.head_dim]
                scores[p] = np.dot(q_h, k_p) / np.sqrt(self.head_dim)

            # Softmax + weighted value sum
            attn_weights = softmax(scores)
            head_out = np.zeros(self.head_dim, dtype=np.float32)
            for p in range(position + 1):
                v_p = self.kv_v[layer, p, kv_h * self.head_dim:(kv_h + 1) * self.head_dim]
                head_out += attn_weights[p] * v_p

            attn_out[h * self.head_dim:(h + 1) * self.head_dim] = head_out

        # Output projection + residual
        x = x + attn_out @ self.weights[f'{pf}.attn_output.weight'].T

        # ═══ FFN (MoE or Dense) ═══
        x_norm = rms_norm(x, self.weights[f'{pf}.ffn_norm.weight'], self.norm_eps)

        if self.moe_layers[layer] is not None:
            # MoE FFN — route to top-K experts
            ffn_out = self.moe_layers[layer].forward(x_norm)
        else:
            # Dense FFN fallback (standard gate-up-down)
            gate = x_norm @ self.weights[f'{pf}.ffn_gate.weight'].T
            up = x_norm @ self.weights[f'{pf}.ffn_up.weight'].T
            ffn_out = (silu(gate) * up) @ self.weights[f'{pf}.ffn_down.weight'].T

        x = x + ffn_out
        return x

    def reset(self):
        """Reset KV cache for new generation."""
        self.kv_k[:] = 0
        self.kv_v[:] = 0
        self.pos = 0


# ════════════════════════════════════════════════════════════════
# GGUF MOE LOADER
# ════════════════════════════════════════════════════════════════

def detect_moe_model(metadata: Dict) -> bool:
    """Check if a GGUF model is MoE based on metadata."""
    n_experts = 0
    for key in metadata:
        if 'expert_count' in key:
            n_experts = int(metadata[key])
        if 'expert_used_count' in key:
            return True  # Definitely MoE
    return n_experts > 1


def get_moe_config(metadata: Dict, architecture: str) -> Dict:
    """Extract MoE-specific config from GGUF metadata."""
    def meta(key, default=0):
        val = metadata.get(f'{architecture}.{key}', None)
        if val is None:
            val = metadata.get(f'llama.{key}', default)
        return val

    config = {
        'dim': int(meta('embedding_length', 0)),
        'n_layers': int(meta('block_count', 0)),
        'n_heads': int(meta('attention.head_count', 0)),
        'n_kv_heads': int(meta('attention.head_count_kv', 0)),
        'hidden_dim': int(meta('feed_forward_length', 0)),
        'vocab_size': int(meta('vocab_size', 0)),
        'n_experts': int(meta('expert_count', 8)),
        'top_k_experts': int(meta('expert_used_count', 2)),
        'rope_base': float(meta('rope.freq_base', 10000.0)),
        'max_seq_len': int(meta('context_length', 4096)),
        'norm_eps': float(meta('attention.layer_norm_rms_epsilon', 1e-5)),
        'has_shared_expert': 'deepseek' in architecture.lower(),
    }

    if config['n_kv_heads'] == 0:
        config['n_kv_heads'] = config['n_heads']
    config['head_dim'] = config['dim'] // config['n_heads'] if config['n_heads'] > 0 else 64

    return config
