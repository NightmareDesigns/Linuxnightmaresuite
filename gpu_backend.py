"""
GPU Backend — CUDA Acceleration via CuPy
==========================================

Drop-in GPU acceleration for the inference engine.
Uses CuPy (NumPy-compatible CUDA library) for matrix operations.

Supports:
- Auto-detection of NVIDIA GPUs
- Multi-GPU tensor splitting (for your dual-card setup!)
- Automatic fallback to CPU if no GPU available
- Memory-aware model placement

Dependencies:
    pip install cupy-cuda12x    (for CUDA 12.x)
    OR pip install cupy-cuda11x (for CUDA 11.x)
    
    To check your CUDA version: nvidia-smi

Author: Built for Levi's RTX 3060 Ti + GTX 1660 OC setup 🔥
"""

import sys
import nightmare_np as np
import os
import glob
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


# === Auto-detect CUDA PATH before importing CuPy ===
def _auto_detect_cuda():
    """Find and set CUDA_PATH if not already set. Adds CUDA bin to PATH."""
    if os.environ.get("CUDA_PATH") and os.path.isdir(os.environ["CUDA_PATH"]):
        # Already set and valid
        cuda_bin = os.path.join(os.environ["CUDA_PATH"], "bin")
        if cuda_bin not in os.environ.get("PATH", ""):
            os.environ["PATH"] = cuda_bin + os.pathsep + os.environ.get("PATH", "")
        return
    
    # Search common CUDA install locations on Windows
    search_paths = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA",
        r"C:\CUDA",
        os.path.expandvars(r"%LOCALAPPDATA%\NVIDIA\CUDA"),
    ]
    
    for base in search_paths:
        if os.path.isdir(base):
            # Find highest version
            versions = sorted(glob.glob(os.path.join(base, "v*")), reverse=True)
            for cuda_dir in versions:
                bin_dir = os.path.join(cuda_dir, "bin")
                if os.path.isdir(bin_dir) and glob.glob(os.path.join(bin_dir, "curand*.dll")):
                    os.environ["CUDA_PATH"] = cuda_dir
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                    print(f"[GPU] Auto-detected CUDA: {cuda_dir}")
                    return
    
    # Check if nvcc is already on PATH (linux/custom installs)
    import shutil
    nvcc = shutil.which("nvcc")
    if nvcc:
        cuda_dir = os.path.dirname(os.path.dirname(nvcc))
        os.environ["CUDA_PATH"] = cuda_dir
        print(f"[GPU] Found CUDA via nvcc: {cuda_dir}")


# Run CUDA detection BEFORE importing CuPy
_auto_detect_cuda()


# Try to import CuPy — fall back gracefully to CPU
try:
    import cupy as cp
    import cupy.cuda as cuda
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    cp = None


@dataclass
class GPUInfo:
    """Information about a detected GPU."""
    id: int
    name: str
    total_memory: int      # bytes
    free_memory: int       # bytes
    compute_capability: tuple
    
    @property
    def total_memory_gb(self) -> float:
        return self.total_memory / (1024**3)
    
    @property
    def free_memory_gb(self) -> float:
        return self.free_memory / (1024**3)
    
    def __repr__(self):
        return (f"GPU {self.id}: {self.name} | "
                f"{self.free_memory_gb:.1f}/{self.total_memory_gb:.1f} GB free | "
                f"CC {self.compute_capability[0]}.{self.compute_capability[1]}")


class GPUBackend:
    """
    GPU acceleration manager.
    
    Handles:
    - GPU detection and selection
    - Memory management
    - Tensor transfer (CPU <-> GPU)
    - Multi-GPU splitting for large models
    - Automatic fallback to CPU
    
    Usage:
        gpu = GPUBackend()
        gpu.initialize()
        
        # Transfer tensors to GPU
        weight_gpu = gpu.to_gpu(weight_numpy)
        
        # Operations happen on GPU automatically
        result = gpu.matmul(a_gpu, b_gpu)
        
        # Get result back to CPU
        result_cpu = gpu.to_cpu(result)
    """
    
    def __init__(self, preferred_gpu: int = 0, multi_gpu: bool = True):
        """
        Args:
            preferred_gpu: Primary GPU ID (0 = first card)
            multi_gpu: Enable splitting across multiple GPUs
        """
        self.preferred_gpu = preferred_gpu
        self.multi_gpu = multi_gpu
        self.gpus: List[GPUInfo] = []
        self.active_gpu: int = 0
        self.is_initialized = False
        self.use_gpu = False
        
        # Memory tracking
        self.allocated: Dict[int, int] = {}  # gpu_id -> bytes allocated
    
    def initialize(self) -> bool:
        """
        Detect GPUs and initialize CUDA.
        
        Returns True if GPU acceleration is available.
        """
        if not GPU_AVAILABLE:
            print("[GPU] CuPy not found. Running on CPU.")
            print("[GPU] Install with: pip install cupy-cuda12x")
            self.use_gpu = False
            return False
        
        try:
            n_devices = cuda.runtime.getDeviceCount()
        except Exception as e:
            print(f"[GPU] CUDA error: {e}")
            print("[GPU] Falling back to CPU")
            self.use_gpu = False
            return False
        
        if n_devices == 0:
            print("[GPU] No CUDA GPUs detected. Running on CPU.")
            self.use_gpu = False
            return False
        
        # Gather info on each GPU
        print(f"\n[GPU] Detected {n_devices} NVIDIA GPU(s):")
        print(f"{'─'*60}")
        
        for i in range(n_devices):
            with cuda.Device(i):
                props = cuda.runtime.getDeviceProperties(i)
                mem_free, mem_total = cuda.runtime.memGetInfo()
                
                info = GPUInfo(
                    id=i,
                    name=props["name"].decode() if isinstance(props["name"], bytes) else str(props["name"]),
                    total_memory=mem_total,
                    free_memory=mem_free,
                    compute_capability=(props["major"], props["minor"])
                )
                self.gpus.append(info)
                print(f"  {info}")
        
        print(f"{'─'*60}")
        
        # Select primary GPU (prefer the one with more VRAM)
        if self.multi_gpu and len(self.gpus) > 1:
            # Sort by free memory, use the biggest as primary
            sorted_gpus = sorted(self.gpus, key=lambda g: g.free_memory, reverse=True)
            self.active_gpu = sorted_gpus[0].id
            print(f"[GPU] Primary: GPU {self.active_gpu} ({sorted_gpus[0].name})")
            print(f"[GPU] Multi-GPU: Enabled ({len(self.gpus)} cards)")
        else:
            self.active_gpu = self.preferred_gpu
            print(f"[GPU] Using: GPU {self.active_gpu}")
        
        self.use_gpu = True
        self.is_initialized = True
        
        # Set active device
        cp.cuda.Device(self.active_gpu).use()
        
        return True
    
    # === Tensor Operations (GPU-accelerated) ===
    
    def to_gpu(self, array: np.ndarray, gpu_id: Optional[int] = None) -> 'cp.ndarray':
        """Transfer a NumPy array to GPU memory."""
        if not self.use_gpu:
            return array
        
        target_gpu = gpu_id if gpu_id is not None else self.active_gpu
        
        with cp.cuda.Device(target_gpu):
            gpu_array = cp.asarray(array)
            self.allocated[target_gpu] = self.allocated.get(target_gpu, 0) + array.nbytes
            return gpu_array
    
    def to_cpu(self, array) -> np.ndarray:
        """Transfer a CuPy array back to CPU (NumPy)."""
        if not self.use_gpu:
            return array
        if isinstance(array, np.ndarray):
            return array
        return cp.asnumpy(array)
    
    def matmul(self, a, b):
        """Matrix multiplication on GPU."""
        if not self.use_gpu:
            return np.matmul(a, b)
        # Ensure float32 precision for matmuls (weights stored as f16 for memory)
        a32 = a.astype(cp.float32) if a.dtype == cp.float16 else a
        b32 = b.astype(cp.float32) if b.dtype == cp.float16 else b
        return cp.matmul(a32, b32)
    
    def zeros(self, shape, dtype=np.float32):
        """Create zero-filled array on GPU."""
        if not self.use_gpu:
            return np.zeros(shape, dtype=dtype)
        return cp.zeros(shape, dtype=dtype)
    
    def empty(self, shape, dtype=np.float32):
        """Create uninitialized array on GPU."""
        if not self.use_gpu:
            return np.empty(shape, dtype=dtype)
        return cp.empty(shape, dtype=dtype)
    
    def softmax(self, x, axis=-1):
        """GPU-accelerated softmax."""
        if not self.use_gpu:
            from tensor_ops import softmax as cpu_softmax
            return cpu_softmax(x)
        
        x_max = cp.max(x, axis=axis, keepdims=True)
        exp_x = cp.exp(x - x_max)
        return exp_x / cp.sum(exp_x, axis=axis, keepdims=True)
    
    def rms_norm(self, x, weight, eps=1e-5, gemma_norm=False):
        """GPU-accelerated RMS normalization."""
        if not self.use_gpu:
            from tensor_ops import rms_norm as cpu_rms_norm
            return cpu_rms_norm(x, weight, eps)
        
        # Always compute in float32 for precision (weight may be f16)
        x32 = x.astype(cp.float32) if x.dtype != cp.float32 else x
        w32 = weight.astype(cp.float32) if weight.dtype != cp.float32 else weight
        # Gemma stores norm weights as (w-1), add 1 back
        if gemma_norm:
            w32 = w32 + 1.0
        rms = cp.sqrt(cp.mean(x32 ** 2, axis=-1, keepdims=True) + eps)
        return (x32 / rms) * w32
    
    def silu(self, x):
        """GPU-accelerated SiLU activation."""
        if not self.use_gpu:
            from tensor_ops import silu as cpu_silu
            return cpu_silu(x)
        
        return x * (1.0 / (1.0 + cp.exp(-x)))
    
    def rope(self, x, cos, sin, position):
        """GPU-accelerated RoPE application (contiguous half-split, LLaMA/Gemma style)."""
        if not self.use_gpu:
            from tensor_ops import apply_rope
            return apply_rope(x, cos, sin, position)
        
        # Cast to float32 for precision (RoPE tables may be f16)
        cos_pos = cos[position].astype(cp.float32) if cos.dtype != cp.float32 else cos[position]
        sin_pos = sin[position].astype(cp.float32) if sin.dtype != cp.float32 else sin[position]
        
        x32 = x.astype(cp.float32) if x.dtype != cp.float32 else x
        
        # Contiguous half-split (NOT interleaved) — standard for LLaMA, Gemma, Mistral
        if x32.ndim == 2:
            half = x32.shape[1] // 2
            x0 = x32[:, :half]
            x1 = x32[:, half:]
            out = cp.empty_like(x32, dtype=cp.float32)
            out[:, :half] = x0 * cos_pos - x1 * sin_pos
            out[:, half:] = x0 * sin_pos + x1 * cos_pos
        else:
            half = x32.shape[0] // 2
            x0 = x32[:half]
            x1 = x32[half:]
            out = cp.empty_like(x32, dtype=cp.float32)
            out[:half] = x0 * cos_pos - x1 * sin_pos
            out[half:] = x0 * sin_pos + x1 * cos_pos
        
        return out
    
    # === Multi-GPU Utilities ===
    
    def estimate_model_memory(self, n_params: int, quant_bits: int = 4) -> float:
        """Estimate memory needed for a model (in GB)."""
        bytes_per_param = quant_bits / 8
        model_size = n_params * bytes_per_param
        # Add ~20% overhead for KV cache, activations, etc.
        return (model_size * 1.2) / (1024**3)
    
    def plan_model_placement(self, layer_sizes: List[int]) -> Dict[int, List[int]]:
        """
        Plan which layers go on which GPU for multi-GPU inference.
        
        Strategy: Fill the primary GPU first, overflow to secondary.
        This matches your setup — put most layers on the 3060 Ti (8GB),
        overflow to the 1660 (6GB).
        
        Args:
            layer_sizes: Memory per layer in bytes
            
        Returns:
            Dict mapping gpu_id -> list of layer indices
        """
        if not self.multi_gpu or len(self.gpus) <= 1:
            return {self.active_gpu: list(range(len(layer_sizes)))}
        
        placement = {gpu.id: [] for gpu in self.gpus}
        gpu_remaining = {gpu.id: int(gpu.free_memory * 0.85) for gpu in self.gpus}  # 85% safe limit
        
        # Sort GPUs by available memory (largest first)
        gpu_order = sorted(self.gpus, key=lambda g: g.free_memory, reverse=True)
        
        for layer_idx, layer_size in enumerate(layer_sizes):
            placed = False
            for gpu in gpu_order:
                if gpu_remaining[gpu.id] >= layer_size:
                    placement[gpu.id].append(layer_idx)
                    gpu_remaining[gpu.id] -= layer_size
                    placed = True
                    break
            
            if not placed:
                # If no single GPU fits, put on the one with most space
                best_gpu = gpu_order[0]
                placement[best_gpu.id].append(layer_idx)
                print(f"[GPU] Warning: Layer {layer_idx} may exceed VRAM on GPU {best_gpu.id}")
        
        # Print placement plan
        for gpu_id, layers in placement.items():
            if layers:
                gpu_name = next(g.name for g in self.gpus if g.id == gpu_id)
                print(f"[GPU] {gpu_name}: Layers {layers[0]}-{layers[-1]} ({len(layers)} layers)")
        
        return placement
    
    def get_memory_stats(self) -> Dict[int, Dict[str, float]]:
        """Get current memory usage for all GPUs."""
        stats = {}
        if not self.use_gpu:
            return stats
        
        for gpu in self.gpus:
            with cp.cuda.Device(gpu.id):
                mem_free, mem_total = cuda.runtime.memGetInfo()
                stats[gpu.id] = {
                    "name": gpu.name,
                    "total_gb": mem_total / (1024**3),
                    "free_gb": mem_free / (1024**3),
                    "used_gb": (mem_total - mem_free) / (1024**3),
                    "utilization": (mem_total - mem_free) / mem_total * 100
                }
        
        return stats
    
    def print_memory_stats(self):
        """Print a nice memory usage summary."""
        stats = self.get_memory_stats()
        if not stats:
            print("[GPU] No GPU stats available (running on CPU)")
            return
        
        print(f"\n{'─'*60}")
        print(f"  GPU Memory Usage")
        print(f"{'─'*60}")
        for gpu_id, info in stats.items():
            bar_len = 30
            used_bars = int(info["utilization"] / 100 * bar_len)
            bar = "█" * used_bars + "░" * (bar_len - used_bars)
            print(f"  GPU {gpu_id} ({info['name']}):")
            print(f"    [{bar}] {info['utilization']:.1f}%")
            print(f"    {info['used_gb']:.1f} / {info['total_gb']:.1f} GB")
        print(f"{'─'*60}\n")
    
    def cleanup(self):
        """Free all GPU memory."""
        if not self.use_gpu:
            return
        
        for gpu in self.gpus:
            with cp.cuda.Device(gpu.id):
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
        
        print("[GPU] Memory freed")


# === Global singleton ===
_backend: Optional[GPUBackend] = None


def get_gpu_backend(multi_gpu: bool = False) -> GPUBackend:
    """Get or create the global GPU backend instance."""
    global _backend
    if _backend is None:
        _backend = GPUBackend(multi_gpu=multi_gpu)
        _backend.initialize()
    else:
        # Update multi-GPU setting if backend already exists
        _backend.multi_gpu = multi_gpu
    return _backend


def xp():
    """
    Get the active array module (cupy if GPU, numpy if CPU).
    Use this for code that should work on both:
    
        from gpu_backend import xp
        arr = xp().zeros((100, 100))  # GPU if available, else CPU
    """
    if _backend and _backend.use_gpu:
        return cp
    return np
