# === FORCE CUDA TO SEE BOTH GPUs ===
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

"""
Tensor Operations — From Scratch
All the math needed for transformer inference: matmul, softmax, RMSNorm, RoPE, etc.

Only dependency: NumPy for efficient array math.
"""

import nightmare_np as np
from typing import Optional


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5, add_unit_offset: bool = False) -> np.ndarray:
    """
    Root Mean Square Layer Normalization (used in LLaMA instead of LayerNorm).
    
    Standard (LLaMA/Mistral): RMSNorm(x) = x * weight / sqrt(mean(x^2) + eps)
    Gemma variant:            RMSNorm(x) = x * (1 + weight) / sqrt(mean(x^2) + eps)
    
    Gemma models store norm weights initialized to zeros — the +1 offset makes
    them identity at initialization. Without it, you multiply by ~0 everywhere.
    """
    # Calculate RMS
    rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)
    # Normalize and scale
    if add_unit_offset:
        return (x / rms) * (1.0 + weight)
    return (x / rms) * weight


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax.
    softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
    """
    # Subtract max for numerical stability (prevents overflow)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    """
    SiLU activation (Sigmoid Linear Unit), also called Swish.
    SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    
    Used in LLaMA/Mistral feed-forward networks.
    """
    return x * (1.0 / (1.0 + np.exp(-x)))


def gelu(x: np.ndarray) -> np.ndarray:
    """
    GELU activation (Gaussian Error Linear Unit).
    Used in some transformer variants (GPT-2, Phi, etc.)
    
    Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    """
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


def rope_frequencies(dim: int, seq_len: int, base: float = 10000.0) -> tuple:
    """
    Compute Rotary Position Embedding (RoPE) frequency pairs.
    
    RoPE encodes position by rotating pairs of dimensions using sinusoidal frequencies.
    This gives the model a sense of relative position without explicit position embeddings.
    
    Args:
        dim: Head dimension (must be even)
        seq_len: Maximum sequence length
        base: Base frequency (10000 for most models, 1M+ for long-context)
    
    Returns:
        (cos, sin) arrays of shape (seq_len, dim//2)
    """
    # Frequency for each pair of dimensions
    # theta_i = base^(-2i/dim) for i in [0, dim//2)
    freqs = 1.0 / (base ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    
    # Position indices
    positions = np.arange(seq_len, dtype=np.float32)
    
    # Outer product: (seq_len, dim//2)
    angles = np.outer(positions, freqs)
    
    return np.cos(angles), np.sin(angles)


def apply_rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray, position: int) -> np.ndarray:
    """
    Apply Rotary Position Embedding to a single token's query or key vector.
    
    Rotates pairs of dimensions: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
    
    Args:
        x: Input tensor of shape (n_heads, head_dim) or (head_dim,)
        cos: Cosine frequencies at this position, shape (head_dim//2,)
        sin: Sine frequencies at this position, shape (head_dim//2,)
        position: Current token position in the sequence
    
    Returns:
        Rotated tensor, same shape as input
    """
    # Get cos/sin for this position
    # Clamp position to table size to prevent index out of bounds
    position = min(position, len(cos) - 1)
    cos_pos = cos[position]  # (head_dim//2,)
    sin_pos = sin[position]  # (head_dim//2,)
    
    # Contiguous half-split (LLaMA/Gemma/Mistral style)
    if x.ndim == 2:
        # Shape: (n_heads, head_dim)
        half = x.shape[1] // 2
        x0 = x[:, :half]   # First half
        x1 = x[:, half:]   # Second half
        
        # Rotate
        out = np.empty_like(x)
        out[:, :half] = x0 * cos_pos - x1 * sin_pos
        out[:, half:] = x0 * sin_pos + x1 * cos_pos
    else:
        # Shape: (head_dim,)
        half = x.shape[0] // 2
        x0 = x[:half]
        x1 = x[half:]
        
        out = np.empty_like(x)
        out[:half] = x0 * cos_pos - x1 * sin_pos
        out[half:] = x0 * sin_pos + x1 * cos_pos
    
    return out


def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication wrapper.
    We use NumPy's optimized implementation (which calls BLAS under the hood).
    
    For a truly "from scratch" version you'd write triple-nested loops,
    but that would be ~1000x slower and serve no educational purpose.
    """
    return a @ b


def attention(
    query: np.ndarray,       # (n_heads, head_dim)
    key_cache: np.ndarray,   # (seq_len, n_kv_heads, head_dim)
    value_cache: np.ndarray, # (seq_len, n_kv_heads, head_dim)
    n_heads: int,
    n_kv_heads: int,
    seq_len: int,
    head_dim: int,
    mask: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Multi-Head Attention with Grouped Query Attention (GQA) support.
    
    GQA: Multiple query heads share the same key/value heads.
    If n_heads=32 and n_kv_heads=8, each KV head is shared by 4 query heads.
    
    Args:
        query: Current token's query vectors (n_heads, head_dim)
        key_cache: All cached keys up to current position (seq_len, n_kv_heads, head_dim)
        value_cache: All cached values (seq_len, n_kv_heads, head_dim)
        n_heads: Number of query heads
        n_kv_heads: Number of key/value heads
        seq_len: Current sequence length
        head_dim: Dimension per head
        mask: Optional causal mask
    
    Returns:
        Attention output (n_heads, head_dim)
    """
    # How many query heads per KV head
    kv_group_size = n_heads // n_kv_heads
    scale = 1.0 / np.sqrt(head_dim)
    
    output = np.zeros((n_heads, head_dim), dtype=np.float32)
    
    for h in range(n_heads):
        # Which KV head does this query head use?
        kv_h = h // kv_group_size
        
        # Q @ K^T -> attention scores (seq_len,)
        q = query[h]  # (head_dim,)
        k = key_cache[:seq_len, kv_h, :]  # (seq_len, head_dim)
        
        # Scaled dot-product attention
        scores = (q @ k.T) * scale  # (seq_len,)
        
        # Apply causal mask (optional, for prefill)
        if mask is not None:
            scores = scores + mask[:seq_len]
        
        # Softmax
        attn_weights = softmax(scores)  # (seq_len,)
        
        # Weighted sum of values
        v = value_cache[:seq_len, kv_h, :]  # (seq_len, head_dim)
        output[h] = attn_weights @ v  # (head_dim,)
    
    return output


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create a causal (lower-triangular) attention mask.
    Prevents attending to future tokens during prefill.
    
    Returns mask where future positions are -inf and valid positions are 0.
    """
    mask = np.full((seq_len, seq_len), -np.inf, dtype=np.float32)
    mask = np.triu(mask, k=1)  # Upper triangle = -inf, lower + diagonal = 0
    return mask
