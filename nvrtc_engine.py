import os
import sys
import ctypes
import numpy as np
from engine.bonsai_pc_app import Cuda, _dequant_q8, _dequant_q4_0, _dequant_q6_k, _dequant_q4_k, _dequant_q5_k

class NvrtcEngine:
    """
    Pure Python GPU Engine utilizing NVRTC.
    Allocates memory on GPU, streams weights, and computes MatMul.
    """
    def __init__(self, device=0):
        self.cu = Cuda(device=device)
        self.scratch = None
        self.scratch_nbytes = 0
    
    def matmul(self, x: np.ndarray, w: np.ndarray) -> np.ndarray:
        """
        w is (in_dim, out_dim)
        x is (seq, in_dim)
        """
        seq, k = x.shape
        m, n = w.shape
        if k != m:
            raise ValueError(f"Dim mismatch: x {x.shape}, w {w.shape}")
        
        xp, _ = self.cu.to_gpu(x)
        wp, _ = self.cu.to_gpu(w)
        cp = self.cu.alloc(seq * n * 4)
        
        self.cu.launch(self.cu.f_matmul, ((n + 15) // 16, (seq + 15) // 16), (16, 16),
                       [ctypes.c_ulonglong(xp.value), ctypes.c_ulonglong(wp.value),
                        ctypes.c_ulonglong(cp.value), seq, k, n])
        
        out = self.cu.from_gpu(cp, (seq, n))
        self.cu.free(xp)
        self.cu.free(wp)
        self.cu.free(cp)
        return out
