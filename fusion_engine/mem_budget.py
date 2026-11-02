# -*- coding: utf-8 -*-
"""
mem_budget.py — CPU RAM budget guard for Model Fusion (90% cap)
================================================================
CPU-only rule: fusion may use at most 90% of physical RAM.
- No third-party deps (ctypes + stdlib only).
- Windows: GlobalMemoryStatusEx + GetProcessMemoryInfo.
- Linux/macOS fallback: /proc/meminfo or sysconf.

Usage:
    from fusion_engine.mem_budget import ensure_budget, clamp_rank_for_budget, max_usable_bytes

Nightmare Designs — Levi Gates
"""

import ctypes
import gc
import os
import sys

DEFAULT_FRAC = 0.90


def get_total_ram_bytes():
    """Total physical RAM in bytes."""
    try:
        if sys.platform == "win32":
            class MS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return int(ms.ullTotalPhys)
        elif os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb * 1024
        # POSIX fallback
        if hasattr(os, "sysconf"):
            try:
                pages = os.sysconf("SC_PHYS_PAGES")
                page_sz = os.sysconf("SC_PAGE_SIZE")
                return int(pages * page_sz)
            except Exception:
                pass
    except Exception:
        pass
    # Last resort: assume 16 GiB so we stay conservative.
    return 16 * 1024 ** 3


def get_avail_ram_bytes():
    """Currently available physical RAM in bytes."""
    try:
        if sys.platform == "win32":
            class MS(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return int(ms.ullAvailPhys)
        elif os.path.exists("/proc/meminfo"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
    except Exception:
        pass
    # Conservative fallback: 25% of total assumed free.
    return int(get_total_ram_bytes() * 0.25)


def get_process_rss_bytes():
    """Current process RSS in bytes (0 if unknown)."""
    try:
        if sys.platform == "win32":
            class PMC(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            h = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
                return int(pmc.WorkingSetSize)
        else:
            import resource
            # ru_maxrss is KiB on Linux, bytes on macOS.
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if sys.platform == "darwin":
                return int(rss)
            return int(rss * 1024)
    except Exception:
        pass
    return 0


def max_usable_bytes(frac=DEFAULT_FRAC):
    """Absolute ceiling: frac * total RAM."""
    return int(get_total_ram_bytes() * float(frac))


def allowed_bytes(frac=DEFAULT_FRAC):
    """
    Bytes we may still allocate without crossing frac*total.
    = min(avail_ram, frac*total - process_rss). Never negative.
    """
    total = get_total_ram_bytes()
    cap = int(total * float(frac))
    rss = get_process_rss_bytes()
    avail = get_avail_ram_bytes()
    remaining_cap = cap - rss if rss else avail
    # Respect both the OS-available figure and our 90% cap.
    budget = min(avail, remaining_cap) if rss else min(avail, cap)
    return max(0, int(budget))


class MemoryBudgetExceeded(RuntimeError):
    pass


def ensure_budget(need_bytes, frac=DEFAULT_FRAC, what="allocation"):
    """
    Raise MemoryBudgetExceeded if need_bytes would cross frac*total.
    Runs gc.collect() once before giving up.
    """
    need = int(need_bytes)
    if need <= 0:
        return True
    if need <= allowed_bytes(frac):
        return True
    gc.collect()
    if need <= allowed_bytes(frac):
        return True
    total_gb = get_total_ram_bytes() / 1024 ** 3
    need_gb = need / 1024 ** 3
    raise MemoryBudgetExceeded(
        "CPU RAM budget exceeded for %s: need ~%.2f GiB but 90%% cap "
        "(%.2f GiB total) is full. Lower .ntmre compression rank, fuse fewer layers, "
        "or enable fewer tensors (skip attn/embeddings)." % (what, need_gb, total_gb)
    )


def clamp_rank_for_budget(M, N, requested_rank, n_models=2, frac=DEFAULT_FRAC):
    """
    Shrink .ntmre compression rank so one tensor fusion step fits in budget.
    Rough working set per tensor: n_models*M*N*4 (inputs)
      + M*R*4 + R*N*4 (factors) + M*N*4 (composite) + margin.
    Returns rank in [8, min(requested, M, N)].
    """
    cap = min(M, N)
    req = max(8, min(int(requested_rank), cap))
    # Quick path: estimate at requested rank.
    est = int(n_models * M * N * 4 + (M * req + req * N) * 4 + M * N * 4)
    est = int(est * 1.25)  # 25% headroom for temporaries
    budget = allowed_bytes(frac)
    if est <= budget or req <= 8:
        return req
    # Solve (M+N)*R*4*1.25 <= budget - n_models*M*N*4*1.25 for R.
    fixed = int(n_models * M * N * 4 * 1.25 + M * N * 4 * 1.25)
    room = budget - fixed
    if room <= 0:
        return 8
    r = int(room // (int((M + N) * 4 * 1.25) + 1))
    return max(8, min(r, req, cap))


def file_fits_budget(path, frac=DEFAULT_FRAC, multiplier=1.6):
    """
    Pre-check: estimated peak = filesize * multiplier (dequant to f32
    inflates quants; plus factor buffers). Returns (ok, est_bytes, budget).
    """
    try:
        sz = os.path.getsize(path)
    except Exception:
        return True, 0, allowed_bytes(frac)
    est = int(sz * float(multiplier))
    return est <= allowed_bytes(frac), est, allowed_bytes(frac)
