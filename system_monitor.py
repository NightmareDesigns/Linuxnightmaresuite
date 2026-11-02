"""
💀 NIGHTMARE SYSTEM MONITOR — Hardware Sensors
=================================================
CPU, RAM, GPU stats. No CuPy. No third-party GPU libs.
Uses nvidia-smi for GPU stats (always available with NVIDIA drivers).
Uses psutil for CPU/RAM (stdlib alternative as fallback).

Author: Nightmare Designs 💀
"""

import subprocess
import threading
import time
import os
from typing import Dict, List, Optional


class SystemMonitor:
    """
    Lightweight system monitor — polls hardware stats every second.
    GPU stats via nvidia-smi (no CuPy, no pynvml, just CLI parsing).
    CPU/RAM via psutil or /proc fallback.
    """
    
    def __init__(self, poll_interval: float = 1.0):
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._stats: Dict = {
            "cpu_percent": 0.0,
            "ram_used_gb": 0.0,
            "ram_total_gb": 0.0,
            "ram_percent": 0.0,
            "gpu_count": 0,
            "gpu_names": [],
            "gpu_util": [],
            "gpu_temp": [],
            "gpu_vram_used": [],
            "gpu_vram_total": [],
            "gpu_vram_percent": [],
            "gpu_power": [],
            "gpu_fan": [],
        }
        
        # Detect GPU count on init
        self._detect_gpus()
    
    def _detect_gpus(self):
        """Detect GPUs via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                names = [n.strip() for n in result.stdout.strip().split('\n') if n.strip()]
                self._stats["gpu_count"] = len(names)
                self._stats["gpu_names"] = names
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self._stats["gpu_count"] = 0
    
    def start(self):
        """Start background polling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop polling."""
        self._running = False
    
    def get_stats(self) -> Dict:
        """Get latest stats snapshot (thread-safe)."""
        with self._lock:
            return self._stats.copy()
    
    def _poll_loop(self):
        """Background polling loop."""
        while self._running:
            try:
                with self._lock:
                    self._update_cpu_ram()
                    self._update_gpu()
            except Exception:
                pass
            time.sleep(self.poll_interval)
    
    def _update_cpu_ram(self):
        """Update CPU and RAM stats."""
        try:
            import psutil
            self._stats["cpu_percent"] = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            self._stats["ram_used_gb"] = mem.used / (1024 ** 3)
            self._stats["ram_total_gb"] = mem.total / (1024 ** 3)
            self._stats["ram_percent"] = mem.percent
        except ImportError:
            # Fallback: native Windows API (no psutil, no wmic — wmic is
            # removed on modern Windows).  CPU% via GetSystemTimes deltas,
            # RAM via GlobalMemoryStatusEx.
            self._update_cpu_ram_native()

    def _update_cpu_ram_native(self):
        """CPU/RAM via ctypes Windows API — works without psutil."""
        try:
            import ctypes
            from ctypes import wintypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("len", wintypes.DWORD), ("load", wintypes.DWORD),
                            ("total", ctypes.c_ulonglong), ("avail", ctypes.c_ulonglong),
                            ("total_page", ctypes.c_ulonglong), ("avail_page", ctypes.c_ulonglong),
                            ("total_virt", ctypes.c_ulonglong), ("avail_virt", ctypes.c_ulonglong),
                            ("avail_ext", ctypes.c_ulonglong)]

            ms = MemStatus()
            ms.len = ctypes.sizeof(MemStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
                total_gb = ms.total / (1024 ** 3)
                used_gb = (ms.total - ms.avail) / (1024 ** 3)
                self._stats["ram_total_gb"] = total_gb
                self._stats["ram_used_gb"] = used_gb
                self._stats["ram_percent"] = (ms.load) if total_gb > 0 else 0

            # CPU usage via GetSystemTimes deltas over ~1s.
            class FILETIME(ctypes.Structure):
                _fields_ = [("dwLow", wintypes.DWORD), ("dwHigh", wintypes.DWORD)]

            _idle = FILETIME(); _ker = FILETIME(); _usr = FILETIME()
            if ctypes.windll.kernel32.GetSystemTimes(
                    ctypes.byref(_idle), ctypes.byref(_ker), ctypes.byref(_usr)):

                def _to_t(s):
                    return (s.dwHigh << 32) | s.dwLow

                idle_n = _to_t(_idle); ker_n = _to_t(_ker); usr_n = _to_t(_usr)
                idle_prev = getattr(self, '_cpu_idle_prev', None)
                ker_prev = getattr(self, '_cpu_ker_prev', None)
                usr_prev = getattr(self, '_cpu_usr_prev', None)
                self._cpu_idle_prev, self._cpu_ker_prev, self._cpu_usr_prev = idle_n, ker_n, usr_n
                if idle_prev is not None:
                    d_idle = idle_n - idle_prev
                    # kernel time already includes idle time
                    d_busy = (ker_n + usr_n) - (ker_prev + usr_prev) - d_idle
                    d_tot = d_idle + d_busy
                    if d_tot > 0:
                        self._stats["cpu_percent"] = max(0.0, min(100.0, d_busy * 100.0 / d_tot))
        except Exception:
            pass
    
    def _update_gpu(self):
        """Update GPU stats via nvidia-smi."""
        if self._stats["gpu_count"] == 0:
            return
        
        try:
            # Query all GPU stats in one call
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,fan.speed",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode != 0:
                return
            
            gpu_util = []
            gpu_temp = []
            gpu_vram_used = []
            gpu_vram_total = []
            gpu_vram_pct = []
            gpu_power = []
            gpu_fan = []
            
            for line in result.stdout.strip().split('\n'):
                if not line.strip():
                    continue
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    util = float(parts[0]) if parts[0] not in ('[N/A]', '') else 0
                    temp = float(parts[1]) if parts[1] not in ('[N/A]', '') else 0
                    vram_used_mb = float(parts[2]) if parts[2] not in ('[N/A]', '') else 0
                    vram_total_mb = float(parts[3]) if parts[3] not in ('[N/A]', '') else 0
                    power = float(parts[4]) if len(parts) > 4 and parts[4] not in ('[N/A]', '') else 0
                    fan = float(parts[5]) if len(parts) > 5 and parts[5] not in ('[N/A]', '') else 0
                    
                    gpu_util.append(util)
                    gpu_temp.append(temp)
                    gpu_vram_used.append(vram_used_mb / 1024)  # Convert to GB
                    gpu_vram_total.append(vram_total_mb / 1024)
                    gpu_vram_pct.append((vram_used_mb / vram_total_mb * 100) if vram_total_mb > 0 else 0)
                    gpu_power.append(power)
                    gpu_fan.append(fan)
            
            self._stats["gpu_util"] = gpu_util
            self._stats["gpu_temp"] = gpu_temp
            self._stats["gpu_vram_used"] = gpu_vram_used
            self._stats["gpu_vram_total"] = gpu_vram_total
            self._stats["gpu_vram_percent"] = gpu_vram_pct
            self._stats["gpu_power"] = gpu_power
            self._stats["gpu_fan"] = gpu_fan
            
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
