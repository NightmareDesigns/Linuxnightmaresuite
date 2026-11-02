"""
💀 NIGHTMARE GPU SPLIT — Dual/Multi-GPU Layer Distribution
=============================================================
Splits model layers across multiple GPUs based on VRAM.

Strategy:
  - Detect all GPUs and their available VRAM
  - Assign layers proportionally (more VRAM = more layers)
  - Embedding + output projection always on GPU 0 (biggest)
  - Each GPU holds its layers in VRAM via nightmare_engine

Example (Mistral 7B, 32 layers):
  GPU 0: RTX 3060 Ti (8GB) → Layers 0-19 + embedding + output
  GPU 1: GTX 1660 SUPER (6GB) → Layers 20-31

Author: Nightmare Designs 💀
"""

import ctypes
from pathlib import Path
from typing import List, Dict, Optional, Tuple


class GPUSplitConfig:
    """Configuration for how to split a model across GPUs."""
    
    def __init__(self):
        self.mode = "auto"  # "auto", "gpu0", "gpu1", "split", "cpu"
        self.gpu_count = 0
        self.gpu_info: List[Dict] = []
        self.layer_map: Dict[int, int] = {}  # layer_idx → gpu_idx
        self.embedding_gpu = 0
        self.output_gpu = 0
    
    def detect_gpus(self) -> int:
        """Detect available GPUs via nvidia-smi (reliable, no DLL dependency)."""
        try:
            import subprocess
            # Query GPU names and VRAM via nvidia-smi
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
                creationflags=0x08000000  # CREATE_NO_WINDOW
            )
            if result.returncode != 0:
                self.gpu_count = 0
                return 0
            
            self.gpu_info = []
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = line.split(',')
                if len(parts) >= 2:
                    gpu_id = int(parts[0].strip())
                    vram_mb = int(parts[1].strip())
                    self.gpu_info.append({
                        "id": gpu_id,
                        "vram_mb": vram_mb,
                        "vram_gb": vram_mb / 1024,
                    })
            
            self.gpu_count = len(self.gpu_info)
            
            # Fallback: if nvidia-smi worked but found nothing, try engine
            if self.gpu_count == 0:
                from nightmare_engine import NightmareEngine
                engine = NightmareEngine()
                if engine.available:
                    # Engine initialized = at least 1 GPU
                    self.gpu_count = 1
                    self.gpu_info = [{"id": 0, "vram_mb": 6144, "vram_gb": 6.0}]
            
            return self.gpu_count
        except Exception:
            self.gpu_count = 0
            return 0
    
    def plan_split(self, n_layers: int, model_size_mb: float = 0) -> Dict[int, int]:
        """
        Plan how to split layers across GPUs.
        
        Args:
            n_layers: Number of transformer layers in the model
            model_size_mb: Total model size in MB (for VRAM budget)
        
        Returns:
            Dict mapping layer_idx → gpu_idx
        """
        if self.gpu_count == 0 or self.mode == "cpu":
            # All on CPU
            self.layer_map = {i: -1 for i in range(n_layers)}
            return self.layer_map
        
        if self.mode == "gpu0" or self.gpu_count == 1:
            # All on GPU 0
            self.layer_map = {i: 0 for i in range(n_layers)}
            self.embedding_gpu = 0
            self.output_gpu = 0
            return self.layer_map
        
        if self.mode == "gpu1":
            # All on GPU 1
            self.layer_map = {i: 1 for i in range(n_layers)}
            self.embedding_gpu = 1
            self.output_gpu = 1
            return self.layer_map
        
        # AUTO SPLIT — proportional to VRAM
        total_vram = sum(g["vram_mb"] for g in self.gpu_info)
        
        # Reserve ~1GB on each GPU for overhead (KV cache, activations)
        usable = [(g["vram_mb"] - 1024) for g in self.gpu_info]
        usable = [max(v, 512) for v in usable]  # Minimum 512MB usable
        
        total_usable = sum(usable)
        
        # Assign layers proportionally
        self.layer_map = {}
        layer_idx = 0
        
        for gpu_id in range(self.gpu_count):
            # How many layers this GPU gets (proportional to usable VRAM)
            proportion = usable[gpu_id] / total_usable
            n_layers_this_gpu = max(1, round(n_layers * proportion))
            
            # Don't exceed total
            remaining = n_layers - layer_idx
            n_layers_this_gpu = min(n_layers_this_gpu, remaining)
            
            for _ in range(n_layers_this_gpu):
                if layer_idx < n_layers:
                    self.layer_map[layer_idx] = gpu_id
                    layer_idx += 1
        
        # Assign any remaining layers to last GPU
        while layer_idx < n_layers:
            self.layer_map[layer_idx] = self.gpu_count - 1
            layer_idx += 1
        
        # Embedding and output on GPU 0 (biggest)
        self.embedding_gpu = 0
        self.output_gpu = 0
        
        return self.layer_map
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the split."""
        if not self.layer_map:
            return "No split configured"
        
        gpu_layers: Dict[int, List[int]] = {}
        for layer_idx, gpu_id in self.layer_map.items():
            if gpu_id not in gpu_layers:
                gpu_layers[gpu_id] = []
            gpu_layers[gpu_id].append(layer_idx)
        
        lines = []
        for gpu_id in sorted(gpu_layers.keys()):
            layers = gpu_layers[gpu_id]
            if gpu_id == -1:
                device = "CPU"
            else:
                vram = self.gpu_info[gpu_id]["vram_gb"] if gpu_id < len(self.gpu_info) else 0
                device = f"GPU {gpu_id} ({vram:.0f}GB)"
            
            if layers:
                lines.append(f"  {device}: Layers {min(layers)}-{max(layers)} ({len(layers)} layers)")
        
        lines.append(f"  Embedding: GPU {self.embedding_gpu}")
        lines.append(f"  Output: GPU {self.output_gpu}")
        
        return "\n".join(lines)


# Global config instance
gpu_split = GPUSplitConfig()


def auto_detect_and_split(n_layers: int, mode: str = "auto") -> GPUSplitConfig:
    """
    One-call setup: detect GPUs and plan the split.
    
    Args:
        n_layers: Number of transformer layers
        mode: "auto", "gpu0", "gpu1", "split", "cpu"
    
    Returns:
        Configured GPUSplitConfig
    """
    gpu_split.mode = mode
    gpu_split.detect_gpus()
    gpu_split.plan_split(n_layers)
    
    print(f"\n[GPU Split] Mode: {mode}")
    print(f"[GPU Split] GPUs detected: {gpu_split.gpu_count}")
    for g in gpu_split.gpu_info:
        print(f"  GPU {g['id']}: {g['vram_gb']:.1f} GB VRAM")
    print(f"[GPU Split] Layer assignment:")
    print(gpu_split.get_summary())
    
    return gpu_split
