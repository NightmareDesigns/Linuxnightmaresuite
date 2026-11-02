"""
gpu_info.py — Reliable GPU memory queries for Nightmare Suite
===============================================================

The CUDA driver API's cuMemGetInfo/cuDeviceTotalMem can report clamped
or stale values on some driver stacks (e.g. 4.29 GB total on 8/12 GB
RTX 3060 cards).  This module queries nvidia-smi / NVML directly and
falls back to the CUDA driver only when nvidia-smi is unavailable.

All memory values are returned in BYTES.
"""

import os
import re
import subprocess
import threading
import time


_CACHE = None
_CACHE_TIME = 0.0
_LOCK = threading.Lock()


def _run_nvidia_smi():
    """Run nvidia-smi and parse memory.free / memory.total per GPU."""
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.free,memory.total,name",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=5).decode()
    except Exception:
        return []

    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            free_mb = float(parts[1])
            total_mb = float(parts[2])
            name = parts[3] if len(parts) > 3 else ""
        except ValueError:
            continue
        rows.append({
            "index": idx,
            "free": int(free_mb * 1024 * 1024),
            "total": int(total_mb * 1024 * 1024),
            "name": name,
        })
    return rows


def _fallback_cuda_info():
    """Fallback to the CUDA driver API if nvidia-smi is unavailable."""
    try:
        import ctypes
    except Exception:
        return []
    try:
        cu = ctypes.CDLL("nvcuda.dll")
    except Exception:
        return []
    try:
        if cu.cuInit(0) != 0:
            return []
    except Exception:
        return []

    rows = []
    cu.cuDeviceTotalMem.argtypes = [ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
    for i in range(16):
        dev = ctypes.c_int(i)
        ctx = ctypes.c_void_p()
        try:
            if cu.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) != 0:
                continue
            cu.cuCtxSetCurrent(ctx)
            total = ctypes.c_size_t()
            if cu.cuDeviceTotalMem(ctypes.byref(total), i) != 0:
                cu.cuCtxDestroy_v2(ctx)
                continue
            free = ctypes.c_size_t()
            try:
                cu.cuMemGetInfo.argtypes = [
                    ctypes.POINTER(ctypes.c_size_t),
                    ctypes.POINTER(ctypes.c_size_t),
                ]
                cu.cuMemGetInfo(ctypes.byref(free), ctypes.byref(total))
            except Exception:
                free = total
            rows.append({"index": i, "free": free.value, "total": total.value, "name": ""})
            cu.cuCtxDestroy_v2(ctx)
        except Exception:
            break
    return rows


def get_gpu_info(force=False):
    """Return a list of dicts [{index, free, total, name}] for each GPU."""
    global _CACHE, _CACHE_TIME
    with _LOCK:
        if not force and _CACHE is not None and (time.time() - _CACHE_TIME) < 0.5:
            return _CACHE
        rows = _run_nvidia_smi()
        if not rows:
            rows = _fallback_cuda_info()
        _CACHE = rows
        _CACHE_TIME = time.time()
        return rows


def _find_device(device_index):
    """Look up a CUDA device index in the cached nvidia-smi info."""
    for row in get_gpu_info():
        if row["index"] == device_index:
            return row
    return None


def get_free_vram(device_index):
    """Free VRAM in bytes for the given CUDA device index."""
    row = _find_device(device_index)
    return row["free"] if row else 0


def get_total_vram(device_index):
    """Total VRAM in bytes for the given CUDA device index."""
    row = _find_device(device_index)
    return row["total"] if row else 0


def get_all_free_vram():
    """Return {device_index: free_bytes} for all detected GPUs."""
    return {row["index"]: row["free"] for row in get_gpu_info()}
