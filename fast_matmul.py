"""
Fast Matmul Integration — Nightmare Suite v6
=============================================
Links the quantized matmul kernel (nightmare_matmul.dll) with the
transformer forward pass. Keeps weights in quantized form in RAM and
dispatches to the DLL for direct computation — no dequant to f32 needed.

Falls back to NumPy when DLL isn't built.

Usage in transformer.py:
    from fast_matmul import FastWeights
    
    # During model load:
    self.fast = FastWeights()
    self.fast.register("blk.0.attn_q", raw_bytes, quant_type=8, n_in=2048, n_out=2048)
    
    # During forward pass:
    q = self.fast.matvec(x_norm, "blk.0.attn_q")
    # Falls back to: x_norm @ self.weights["blk.0.attn_q"] if not registered
"""

import numpy as np
import os

try:
    from nightmare_matmul import NightmareMatmul
except ImportError:
    NightmareMatmul = None


class FastWeights:
    """
    Manages raw quantized weight storage and fast matmul dispatch.
    
    Keeps weights in their native quantized format in RAM (4-8× less memory)
    and computes matvec directly without dequantizing.
    """
    
    def __init__(self):
        self._engine = None
        self._weights = {}  # key -> (raw_bytes, quant_type, n_in, n_out)
        self.available = False
        self.stats = {"registered": 0, "fast_calls": 0, "fallback_calls": 0}
        
        if NightmareMatmul is not None:
            try:
                self._engine = NightmareMatmul()
                self.available = self._engine.available
            except Exception as e:
                print(f"[FastWeights] Engine init failed: {e}")
        
        if self.available:
            print(f"[FastWeights] ⚡ Quantized matmul ACTIVE — {self._engine._lib.supported_quant_types():013b} formats")
        else:
            print(f"[FastWeights] NumPy fallback (build nightmare_matmul.dll for speed)")
    
    def register(self, key: str, raw_bytes: np.ndarray, quant_type: int, n_in: int, n_out: int):
        """
        Register a weight tensor's raw quantized bytes.
        
        Args:
            key: weight name (e.g. "blk.0.attn_q")
            raw_bytes: uint8 numpy array — raw quantized data from GGUF
            quant_type: GGML type ID (2=Q4_0, 8=Q8_0, 12=Q4_K, etc.)
            n_in: input dimension (columns the vector multiplies against)
            n_out: output dimension (number of rows / result elements)
        """
        self._weights[key] = (np.ascontiguousarray(raw_bytes, dtype=np.uint8), 
                              quant_type, n_in, n_out)
        self.stats["registered"] += 1
    
    def matvec(self, x: np.ndarray, key: str, fallback_weight: np.ndarray = None) -> np.ndarray:
        """
        Fast linear projection: y = x @ W
        
        Uses quantized kernel if registered, falls back to NumPy.
        
        Args:
            x: float32 input vector
            key: weight name for raw data lookup
            fallback_weight: dequantized f32 weight for NumPy fallback
        """
        if self.available and key in self._weights:
            raw_bytes, quant_type, n_in, n_out = self._weights[key]
            self.stats["fast_calls"] += 1
            try:
                result = self._engine.matvec(x, raw_bytes, n_in, n_out, quant_type)
                return result
            except Exception:
                # Kernel failed — deregister this weight and fall through to NumPy
                del self._weights[key]
                self.stats["fast_calls"] -= 1
        
        # Fallback
        if fallback_weight is not None:
            self.stats["fallback_calls"] += 1
            # Handle SVDWeight objects (from .ntmre compression models)
            from svd_inference import SVDWeight
            if isinstance(fallback_weight, SVDWeight):
                A = fallback_weight.A
                B = fallback_weight.B
                # Fix orientation: x is (dim,), need A as (dim, rank) for x @ A
                if A.ndim == 2 and x.shape[-1] != A.shape[0]:
                    A = A.T
                temp = x @ A
                # B must be (rank, out_dim) for temp @ B
                if B.ndim == 2 and temp.shape[-1] != B.shape[0]:
                    B = B.T
                return temp @ B
            else:
                return x @ fallback_weight
        
        raise ValueError(f"No weight for '{key}' — neither raw nor dequantized")
    
    def has(self, key: str) -> bool:
        """Check if a weight is registered for fast dispatch."""
        return key in self._weights
    
    def clear(self):
        """Clear all weights (model unload)."""
        self._weights.clear()
        self.stats = {"registered": 0, "fast_calls": 0, "fallback_calls": 0}
    
    def memory_mb(self) -> float:
        """Total raw weight memory in MB."""
        return sum(w[0].nbytes for w in self._weights.values()) / (1024 * 1024)
