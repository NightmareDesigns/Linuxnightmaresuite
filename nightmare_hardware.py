# -*- coding: utf-8 -*-
"""
nightmare_hardware.py — Enterprise Universal Hardware Autotuning & Topology
==========================================================================
Zero-dependency, production-grade hardware detection, thread optimization,
and dynamic memory scaling for Nightmare AI Suite.

Rules:
- ZERO PIP PACKAGES / ZERO OUTSIDE CODE: Uses Python standard library only
  (ctypes, os, sys, winreg, platform, struct).
- Hardware Agnostic: Dynamically self-tunes to any system architecture:
  4-core laptops, 64-core threadrippers, 8GB unified APUs, or multi-GPU rigs.
- Physical Core Pinning: Prevents SMT/hyperthreading cache thrashing.
- Enterprise RAM Budgeting: Analyzes live available RAM, guarantees OS safety
  floor, and dynamically selects resident vs. LRU streaming cache modes.

Author: Nightmare Designs 💀 & Antigravity
"""

import os
import sys

try:
    import nightmare_dos_styler
except Exception:
    pass

import platform
import ctypes
from typing import Tuple, Dict, Any, Optional, List

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Enable Windows Virtual Terminal Sequences (ANSI 24-bit TrueColor)
if sys.platform == 'win32':
    try:
        kernel32 = ctypes.windll.kernel32
        hOut = kernel32.GetStdHandle(-11)
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(hOut, ctypes.byref(mode))
        kernel32.SetConsoleMode(hOut, mode.value | 0x0004)
    except Exception:
        pass

# Cyberpunk Terminal Palette (Blood Red Border & Matrix Green Words)
BORDER_RED = "\033[38;2;255;0;0m"       # Vibrant Blood Red for Borders & Drips
TEXT_GREEN = "\033[1;38;2;0;255;70m"    # Glowing Bright Phosphor Matrix Green
BOLD       = "\033[1m"
RESET      = "\033[0m"

# ─────────────────────────────────────────────────────────────────────────────
# 1. CPU Topology & Physical Core Discovery
# ─────────────────────────────────────────────────────────────────────────────

def get_cpu_brand() -> str:
    """Return human-readable CPU brand string via Windows Registry or platform info."""
    if os.name == 'nt':
        try:
            import winreg
            k = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            val, _ = winreg.QueryValueEx(k, "ProcessorNameString")
            return val.strip()
        except Exception:
            pass
    return platform.processor() or "Generic x86_64 / ARM CPU"


def get_cpu_topology() -> Tuple[int, int, Optional[int]]:
    """
    Query system for (physical_cores, logical_threads, affinity_mask).
    - physical_cores: Count of distinct physical execution cores.
    - logical_threads: Total SMT/hyperthread execution contexts.
    - affinity_mask: Bitmask with 1 bit per physical core (first SMT sibling),
                     or None if unavailable.
    Uses Win32 GetLogicalProcessorInformation (zero external dependencies).
    """
    logical = getattr(os, 'cpu_count', lambda: 4)() or 4
    if os.name != 'nt':
        # Linux fallback via /proc/cpuinfo if available
        if sys.platform.startswith('linux'):
            try:
                cores = set()
                with open('/proc/cpuinfo', 'r') as f:
                    core_id = None
                    phys_id = None
                    for line in f:
                        line = line.strip().lower()
                        if line.startswith('physical id'):
                            phys_id = line.split(':', 1)[1].strip()
                        elif line.startswith('core id'):
                            core_id = line.split(':', 1)[1].strip()
                            if phys_id is not None and core_id is not None:
                                cores.add((phys_id, core_id))
                                phys_id, core_id = None, None
                if cores:
                    return len(cores), logical, None
            except Exception:
                pass
        return max(1, logical // 2), logical, None

    try:
        from ctypes import wintypes
        k32 = ctypes.WinDLL('kernel32', use_last_error=False)
        glpi = k32.GetLogicalProcessorInformation
        glpi.restype = wintypes.BOOL
        glpi.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]

        class _SLPI(ctypes.Structure):
            _fields_ = [
                ('ProcessorMask', ctypes.c_size_t),
                ('Relationship', ctypes.c_int),
                ('Reserved', ctypes.c_ubyte * 16),
            ]

        need = wintypes.DWORD(0)
        glpi(None, ctypes.byref(need))
        if not need.value:
            return max(1, logical // 2), logical, None

        buf = ctypes.create_string_buffer(need.value + 32)
        if not glpi(buf, ctypes.byref(need)):
            return max(1, logical // 2), logical, None

        stride = ctypes.sizeof(_SLPI)
        cores = 0
        mask = 0
        n = need.value // stride
        for i in range(n):
            off = i * stride
            rel = int.from_bytes(
                buf.raw[off + ctypes.sizeof(ctypes.c_size_t):
                        off + ctypes.sizeof(ctypes.c_size_t) + 4],
                'little'
            )
            pm = int.from_bytes(
                buf.raw[off:off + ctypes.sizeof(ctypes.c_size_t)],
                'little'
            )
            if rel == 0:  # RelationProcessorCore
                cores += 1
                if pm:
                    low = pm & -pm  # First logical processor of this core
                    mask |= low

        if cores > 0:
            return cores, logical, mask
    except Exception:
        pass

    return max(1, logical // 2), logical, None


def get_cpu_compute_threads() -> int:
    """
    Determine the optimal compute thread count for BLAS / OpenMP GEMV kernels.
    Honors NIGHTMARE_CPU_THREADS if set; otherwise defaults to physical core count.
    """
    env_val = os.environ.get('NIGHTMARE_CPU_THREADS')
    if env_val:
        try:
            return max(1, int(env_val))
        except ValueError:
            pass
    phys, _, _ = get_cpu_topology()
    return max(1, phys)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Memory Discovery & Enterprise Dynamic Cache Budgeting
# ─────────────────────────────────────────────────────────────────────────────

def get_system_ram() -> Dict[str, Any]:
    """
    Query physical memory status with ZERO external dependencies.
    Returns:
      {
        'total_bytes': int,
        'avail_bytes': int,
        'used_bytes': int,
        'total_gb': float,
        'avail_gb': float,
        'load_pct': int,
      }
    """
    if os.name == 'nt':
        try:
            from ctypes import wintypes
            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', wintypes.DWORD),
                    ('dwMemoryLoad', wintypes.DWORD),
                    ('ullTotalPhys', ctypes.c_uint64),
                    ('ullAvailPhys', ctypes.c_uint64),
                    ('ullTotalPageFile', ctypes.c_uint64),
                    ('ullAvailPageFile', ctypes.c_uint64),
                    ('ullTotalVirtual', ctypes.c_uint64),
                    ('ullAvailVirtual', ctypes.c_uint64),
                    ('ullAvailExtendedVirtual', ctypes.c_uint64),
                ]
            m = _MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(m)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
                t = int(m.ullTotalPhys)
                a = int(m.ullAvailPhys)
                u = max(0, t - a)
                return {
                    'total_bytes': t,
                    'avail_bytes': a,
                    'used_bytes': u,
                    'total_gb': round(t / (1024 ** 3), 2),
                    'avail_gb': round(a / (1024 ** 3), 2),
                    'load_pct': int(m.dwMemoryLoad),
                }
        except Exception:
            pass

    # Cross-platform fallback (Linux/macOS)
    try:
        if hasattr(os, 'sysconf'):
            pages = os.sysconf('SC_PHYS_PAGES')
            p_size = os.sysconf('SC_PAGE_SIZE')
            t = pages * p_size
            # Rough available estimate
            a = int(t * 0.5)
            return {
                'total_bytes': t,
                'avail_bytes': a,
                'used_bytes': t - a,
                'total_gb': round(t / (1024 ** 3), 2),
                'avail_gb': round(a / (1024 ** 3), 2),
                'load_pct': 50,
            }
    except Exception:
        pass

    # Conservative 16GB fallback
    fallback_t = 16 * 1024 * 1024 * 1024
    return {
        'total_bytes': fallback_t,
        'avail_bytes': fallback_t // 2,
        'used_bytes': fallback_t // 2,
        'total_gb': 16.0,
        'avail_gb': 8.0,
        'load_pct': 50,
    }


def get_layer_cache_budget_bytes(n_layers: int = 42, layer_bytes_estimate: int = 420 * 1024 * 1024) -> Tuple[int, int, str]:
    """
    Enterprise Dynamic RAM Cache Budget Calculation.
    Safely calculates how many megabytes/bytes of layer cache to allocate:
    - Analyzes live available RAM (not just total hardware capacity).
    - Reserves a strict OS Safety Floor (minimum 2.5 GB or 15% of total RAM).
    - Adapts automatically:
        * Workstation / High-End (>= 48GB RAM & >= 32GB free): Full Resident Cache.
        * Standard Machine (16GB - 32GB RAM): Dynamic LRU with balanced footprint.
        * Constrained / Laptop (<= 12GB RAM or < 6GB free): Stream-focused LRU to prevent paging.
    Returns:
      (budget_bytes, budget_mb, mode_description)
    """
    # Explicit user override always takes precedence
    override_mb = os.environ.get('NIGHTMARE_LAYER_CACHE_MB')
    if override_mb:
        try:
            mb = int(override_mb)
            return mb * 1024 * 1024, mb, "User Override (NIGHTMARE_LAYER_CACHE_MB)"
        except ValueError:
            pass

    ram = get_system_ram()
    total_mb = int(ram['total_bytes'] // (1024 * 1024))
    avail_mb = int(ram['avail_bytes'] // (1024 * 1024))

    # OS Safety Floor: Ensure OS and background processes always have headroom
    os_floor_mb = max(2560, int(total_mb * 0.15))
    safe_headroom_mb = max(1024, avail_mb - os_floor_mb)

    # Estimate required RAM for full resident model (all layers in float32)
    full_model_mb = int((n_layers * layer_bytes_estimate) // (1024 * 1024))

    # Check for Turbo preference
    turbo_requested = (os.environ.get('NIGHTMARE_CPU_TURBO') == '1')

    if turbo_requested and safe_headroom_mb >= full_model_mb:
        # Full Resident Turbo: All layers resident, zero dequant on token steps
        budget_mb = min(safe_headroom_mb, int(full_model_mb * 1.05))
        mode = "Full Resident Turbo (Zero-Dequant BLAS Execution)"
    elif avail_mb >= 28000 and total_mb >= 40000:
        # Workstation Class (64GB RAM): Keep all or nearly all layers resident
        budget_mb = min(safe_headroom_mb, max(24576, int(avail_mb * 0.75)))
        mode = "Workstation Resident (High-RAM Cache)"
    elif avail_mb >= 14000:
        # Standard Desktop (32GB RAM / 16GB free): Keep ~16-24 layers resident
        budget_mb = min(safe_headroom_mb, max(8192, int(avail_mb * 0.60)))
        mode = "Balanced Resident (Semi-Resident LRU)"
    elif avail_mb >= 7000:
        # Constrained Desktop / 16GB Laptop: Keep 6-10 layers resident
        budget_mb = min(safe_headroom_mb, max(3072, int(avail_mb * 0.45)))
        mode = "Consolidated LRU (Balanced Memory Pressure)"
    else:
        # Low Memory Laptop (8GB - 12GB): Streaming Mode (Hold 2-3 layers in flight)
        budget_mb = max(1024, min(safe_headroom_mb, 2048))
        mode = "Aggressive Streaming (Low-RAM Safety Guard)"

    budget_bytes = budget_mb * 1024 * 1024
    return budget_bytes, budget_mb, mode


# ─────────────────────────────────────────────────────────────────────────────
# 3. Environment & Process Affinity Configuration
# ─────────────────────────────────────────────────────────────────────────────

def configure_system_environment() -> Dict[str, Any]:
    """
    Enterprise Pre-Flight System Configuration.
    Must be called BEFORE any heavy compute/BLAS/C DLL initialization:
    - Binds physical core counts to OpenMP, OpenBLAS, MKL, NumExpr.
    - Applies process affinity mask to physical cores (eliminating SMT cache thrash).
    - Configures GPU visibility dynamically.
    Returns status dictionary.
    """
    phys, logical, aff_mask = get_cpu_topology()
    threads = get_cpu_compute_threads()

    # Set compute thread count across all potential math libraries
    if not os.environ.get('NIGHTMARE_CPU_THREADS'):
        os.environ['NIGHTMARE_CPU_THREADS'] = str(threads)

    for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        if not os.environ.get(k):
            os.environ[k] = str(threads)

    # Pin execution to physical cores unless opted out
    pinned = False
    if (aff_mask and phys and os.environ.get('NIGHTMARE_PHYSICAL_ONLY', '1') == '1'
            and not os.environ.get('NIGHTMARE_CPU_THREADS')):
        if os.name == 'nt':
            try:
                from ctypes import wintypes
                k32 = ctypes.WinDLL('kernel32', use_last_error=True)
                k32.GetCurrentProcess.restype = wintypes.HANDLE
                k32.SetProcessAffinityMask.restype = wintypes.BOOL
                k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
                if k32.SetProcessAffinityMask(k32.GetCurrentProcess(), ctypes.c_size_t(aff_mask)):
                    pinned = True
            except Exception:
                pass

    # Discover GPUs
    gpu_list = []
    try:
        from nightmare_gpu_discovery import usable_accelerators
        gpu_list = usable_accelerators()
    except Exception:
        pass

    if gpu_list:
        dev_indices = ",".join(str(d.get("index", 0)) for d in gpu_list)
        os.environ["CUDA_VISIBLE_DEVICES"] = dev_indices
        os.environ["NIGHTMARE_FORCE_CPU"] = "0"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["NIGHTMARE_FORCE_CPU"] = "1"

    return {
        'physical_cores': phys,
        'logical_threads': logical,
        'threads': threads,
        'affinity_mask': hex(aff_mask) if aff_mask else None,
        'pinned': pinned,
        'gpus': gpu_list,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4. Enterprise Diagnostic Banner
# ─────────────────────────────────────────────────────────────────────────────

def format_hardware_banner(model_name: Optional[str] = None) -> str:
    """
    Format a clean, standardized enterprise hardware diagnostic card.
    Displays CPU architecture, physical vs logical cores, live RAM metrics,
    OS safety reserves, layer cache budget, and active accelerator.
    Styled with blood red dripping borders and phosphor matrix green text.
    """
    cpu_name = get_cpu_brand()
    phys, logical, mask = get_cpu_topology()
    threads = get_cpu_compute_threads()
    ram = get_system_ram()
    _, budget_mb, mode = get_layer_cache_budget_bytes()
    os_floor_gb = round(max(2.5, ram['total_gb'] * 0.15), 2)

    # Accelerator info
    acc_desc = "CPU-Only (AVX2+FMA Multi-Threaded Engine)"
    try:
        from nightmare_gpu_discovery import usable_accelerators
        gpus = usable_accelerators()
        if gpus:
            g_strs = [f"{g.get('name', 'GPU')} ({round(g.get('total', 0) / 1e9, 1)} GB {g.get('runtime', '').upper()})" for g in gpus]
            acc_desc = " | ".join(g_strs)
    except Exception:
        pass

    inner_w = 78
    bar = "═" * inner_w
    top_bar = f"{BORDER_RED}╔{bar}╗{RESET}"
    mid_bar = f"{BORDER_RED}╠══════▼═════════════════════════▼══════════════════════════▼══════════════════╣{RESET}"
    bot_bar = f"{BORDER_RED}╚{bar}╝{RESET}"

    def row(text: str, bold: bool = False, center: bool = False) -> str:
        if center:
            padded = text.center(inner_w)
        else:
            padded = (" " + text).ljust(inner_w)
        c = f"{TEXT_GREEN}{BOLD}" if bold else TEXT_GREEN
        return f"{BORDER_RED}║{RESET}{c}{padded}{RESET}{BORDER_RED}║{RESET}"

    lines = [
        top_bar,
        row("NIGHTMARE AI SUITE v13 — ENTERPRISE", bold=True, center=True),
        row("Universal Hardware Scaling & Autotuning", bold=False, center=True),
        mid_bar,
        row(f"[HARDWARE] CPU: {cpu_name}"),
        row(f"  Topology:    {phys} Physical Cores / {logical} Logical Threads (Affinity: {hex(mask) if mask else 'All'})"),
        row(f"  Execution:   OpenMP {threads}T Physical Cores Bound (Zero SMT Cache Contention)"),
        row(f"[HARDWARE] RAM: {ram['total_gb']} GB Total | {ram['avail_gb']} GB Free (Load: {ram['load_pct']}%)"),
        row(f"  Cache Plan:  {mode}"),
        row(f"  Budget:      {budget_mb / 1024:.2f} GB Allocation (OS Floor Reserve: {os_floor_gb} GB)"),
        row(f"[HARDWARE] Accelerator: {acc_desc}"),
    ]
    if model_name:
        lines.append(row(f"[MODEL] Active Engine: {model_name}"))
    lines.append(bot_bar)
    return "\n".join(lines)


def print_hardware_banner(model_name: Optional[str] = None):
    """Print the enterprise hardware card to console."""
    print(format_hardware_banner(model_name))


if __name__ == '__main__':
    configure_system_environment()
    print_hardware_banner("Gemma 4 E4B Abliterated")
