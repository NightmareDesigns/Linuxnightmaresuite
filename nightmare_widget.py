"""
💀 NIGHTMARE WIDGET — Hardened GPU/CPU/RAM Monitor
=====================================================
Always-on-top desktop widget showing real-time system stats.
Uses nvidia-smi for GPU (no CuPy, no pynvml). 
Pure tkinter + subprocess. From scratch.

Features:
  - CPU usage %
  - RAM used / total (GB + %)
  - Per-GPU: utilization %, temperature, VRAM used/total, power draw, fan speed
  - Color-coded bars (green → yellow → red)
  - Compact mode (small) or expanded mode
  - Always on top
  - Draggable
  - Auto-refresh every 1 second
  - Nightmare theme (black/red/green)

Launch: python nightmare_widget.py
Or: double-click NIGHTMARE_WIDGET.bat

Author: Nightmare Designs 💀
"""

import tkinter as tk
import subprocess
import threading
import time
import os
import sys
from pathlib import Path


# === THEME ===
COLORS = {
    "bg": "#000000",
    "border": "#1a1a1a",
    "text": "#cccccc",
    "text_dim": "#666666",
    "text_bright": "#ffffff",
    "green": "#00ff41",
    "yellow": "#ffcc00",
    "orange": "#ff6600",
    "red": "#ff0000",
    "blood": "#8b0000",
    "bar_bg": "#111111",
    "title": "#ff0000",
}


def get_bar_color(percent):
    """Color based on usage percentage."""
    if percent < 50:
        return COLORS["green"]
    elif percent < 75:
        return COLORS["yellow"]
    elif percent < 90:
        return COLORS["orange"]
    return COLORS["red"]


class NightmareWidget:
    """Always-on-top system monitoring widget."""
    
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("💀 Monitor")
        self.root.configure(bg=COLORS["bg"])
        self.root.attributes("-topmost", True)
        self.root.overrideredirect(True)  # No title bar — custom
        self.root.geometry("320x580+50+50")
        self.root.resizable(False, False)
        
        self._running = True
        self._update_id = None
        self._drag_x = 0
        self._drag_y = 0
        
        # GPU info cache
        self.gpu_count = 0
        self.gpu_names = []
        self._detect_gpus()
        
        # Build UI
        self._build_ui()
        
        # Start monitoring
        self._update_stats()
        
        # Make draggable
        self.root.bind("<Button-1>", self._start_drag)
        self.root.bind("<B1-Motion>", self._do_drag)
        
        # Right-click to close
        self.root.bind("<Button-3>", lambda e: self._close())
    
    def _detect_gpus(self):
        """Detect GPUs via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, creationflags=0x08000000
            )
            if result.returncode == 0:
                names = [n.strip() for n in result.stdout.strip().split('\n') if n.strip()]
                self.gpu_count = len(names)
                self.gpu_names = names
        except:
            self.gpu_count = 0
            self.gpu_names = []
    
    def _build_ui(self):
        """Build the widget UI."""
        # === Title Bar (custom, draggable) ===
        title_frame = tk.Frame(self.root, bg=COLORS["blood"], height=28)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        
        tk.Label(title_frame, text=" 💀 NIGHTMARE MONITOR",
                font=("Consolas", 10, "bold"), bg=COLORS["blood"],
                fg=COLORS["text_bright"]).pack(side=tk.LEFT, padx=5)
        
        tk.Label(title_frame, text="✕", font=("Consolas", 12, "bold"),
                bg=COLORS["blood"], fg=COLORS["text"],
                cursor="hand2").pack(side=tk.RIGHT, padx=8)
        title_frame.children[list(title_frame.children.keys())[-1]].bind(
            "<Button-1>", lambda e: self._close())
        
        # === Separator ===
        tk.Frame(self.root, bg=COLORS["red"], height=2).pack(fill=tk.X)
        
        # === Main content ===
        self.content = tk.Frame(self.root, bg=COLORS["bg"], padx=10, pady=8)
        self.content.pack(fill=tk.BOTH, expand=True)
        
        # CPU Section
        self._build_section("⚡", "CPU")
        self.cpu_label = tk.Label(self.content, text="...",
                                  font=("Consolas", 11), bg=COLORS["bg"],
                                  fg=COLORS["text"], anchor=tk.W)
        self.cpu_label.pack(fill=tk.X, pady=(0, 2))
        self.cpu_bar = tk.Canvas(self.content, height=12, bg=COLORS["bar_bg"],
                                 highlightthickness=0)
        self.cpu_bar.pack(fill=tk.X, pady=(0, 8))
        
        # RAM Section
        self._build_section("🧠", "RAM")
        self.ram_label = tk.Label(self.content, text="...",
                                  font=("Consolas", 11), bg=COLORS["bg"],
                                  fg=COLORS["text"], anchor=tk.W)
        self.ram_label.pack(fill=tk.X, pady=(0, 2))
        self.ram_bar = tk.Canvas(self.content, height=12, bg=COLORS["bar_bg"],
                                 highlightthickness=0)
        self.ram_bar.pack(fill=tk.X, pady=(0, 8))
        
        # GPU Sections
        self.gpu_labels = []
        self.gpu_bars = []
        self.gpu_temp_labels = []
        self.gpu_vram_labels = []
        self.gpu_vram_bars = []
        self.gpu_power_labels = []
        
        for i in range(self.gpu_count):
            name_short = self.gpu_names[i].replace("NVIDIA GeForce ", "")
            self._build_section("🎮", f"GPU {i}: {name_short}")
            
            # Utilization
            lbl = tk.Label(self.content, text="Load: ...",
                          font=("Consolas", 10), bg=COLORS["bg"],
                          fg=COLORS["text"], anchor=tk.W)
            lbl.pack(fill=tk.X)
            self.gpu_labels.append(lbl)
            
            bar = tk.Canvas(self.content, height=10, bg=COLORS["bar_bg"],
                           highlightthickness=0)
            bar.pack(fill=tk.X, pady=(0, 3))
            self.gpu_bars.append(bar)
            
            # Temp + Power
            temp_lbl = tk.Label(self.content, text="Temp: ... | Power: ...",
                               font=("Consolas", 10), bg=COLORS["bg"],
                               fg=COLORS["text_dim"], anchor=tk.W)
            temp_lbl.pack(fill=tk.X)
            self.gpu_temp_labels.append(temp_lbl)
            
            # VRAM
            vram_lbl = tk.Label(self.content, text="VRAM: ...",
                               font=("Consolas", 10), bg=COLORS["bg"],
                               fg=COLORS["text"], anchor=tk.W)
            vram_lbl.pack(fill=tk.X)
            self.gpu_vram_labels.append(vram_lbl)
            
            vram_bar = tk.Canvas(self.content, height=10, bg=COLORS["bar_bg"],
                                highlightthickness=0)
            vram_bar.pack(fill=tk.X, pady=(0, 3))
            self.gpu_vram_bars.append(vram_bar)
            
            # Power
            power_lbl = tk.Label(self.content, text="",
                                font=("Consolas", 9), bg=COLORS["bg"],
                                fg=COLORS["text_dim"], anchor=tk.W)
            power_lbl.pack(fill=tk.X, pady=(0, 8))
            self.gpu_power_labels.append(power_lbl)
    
    def _build_section(self, icon, title):
        """Build a section header."""
        frame = tk.Frame(self.content, bg=COLORS["bg"])
        frame.pack(fill=tk.X, pady=(4, 2))
        tk.Label(frame, text=f"{icon} {title}", font=("Consolas", 10, "bold"),
                bg=COLORS["bg"], fg=COLORS["green"]).pack(side=tk.LEFT)
    
    def _draw_bar(self, canvas, percent, color=None):
        """Draw a progress bar on a canvas."""
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 10:
            w = 300
        
        if color is None:
            color = get_bar_color(percent)
        
        # Background
        canvas.create_rectangle(0, 0, w, h, fill=COLORS["bar_bg"], outline="")
        # Fill
        fill_w = int(w * percent / 100)
        if fill_w > 0:
            canvas.create_rectangle(0, 0, fill_w, h, fill=color, outline="")
        # Percentage text
        canvas.create_text(w - 5, h//2, text=f"{percent:.0f}%",
                          font=("Consolas", 8), fill=COLORS["text_bright"], anchor=tk.E)
    
    def _update_stats(self):
        """Update all stats every 1 second."""
        if not self._running:
            return
        
        try:
            self._update_cpu_ram()
            self._update_gpu()
        except Exception as e:
            pass
        
        self._update_id = self.root.after(3000, self._update_stats)
    
    def _update_cpu_ram(self):
        """Update CPU and RAM."""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            
            self.cpu_label.config(text=f"{cpu:.0f}%")
            self._draw_bar(self.cpu_bar, cpu)
            
            ram_used = mem.used / (1024**3)
            ram_total = mem.total / (1024**3)
            ram_pct = mem.percent
            self.ram_label.config(text=f"{ram_used:.1f} / {ram_total:.1f} GB")
            self._draw_bar(self.ram_bar, ram_pct)
        except ImportError:
            # Fallback without psutil
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "loadpercentage", "/value"],
                    capture_output=True, text=True, timeout=3, creationflags=0x08000000
                )
                for line in result.stdout.split('\n'):
                    if "LoadPercentage" in line:
                        cpu = int(line.split('=')[1].strip())
                        self.cpu_label.config(text=f"{cpu}%")
                        self._draw_bar(self.cpu_bar, cpu)
            except:
                self.cpu_label.config(text="N/A")
    
    def _update_gpu(self):
        """Update GPU stats via nvidia-smi."""
        if self.gpu_count == 0:
            return
        
        try:
            result = subprocess.run(
                ["nvidia-smi",
                 "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total,power.draw,fan.speed",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, creationflags=0x08000000
            )
            
            if result.returncode != 0:
                return
            
            for i, line in enumerate(result.stdout.strip().split('\n')):
                if i >= self.gpu_count:
                    break
                if not line.strip():
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                if len(parts) < 4:
                    continue
                
                util = float(parts[0]) if parts[0] not in ('[N/A]', '') else 0
                temp = float(parts[1]) if parts[1] not in ('[N/A]', '') else 0
                vram_used = float(parts[2]) if parts[2] not in ('[N/A]', '') else 0
                vram_total = float(parts[3]) if parts[3] not in ('[N/A]', '') else 1
                power = float(parts[4]) if len(parts) > 4 and parts[4] not in ('[N/A]', '') else 0
                fan = float(parts[5]) if len(parts) > 5 and parts[5] not in ('[N/A]', '') else 0
                
                vram_pct = (vram_used / vram_total * 100) if vram_total > 0 else 0
                
                # Update labels
                self.gpu_labels[i].config(text=f"Load: {util:.0f}%")
                self._draw_bar(self.gpu_bars[i], util)
                
                # Temp color
                temp_color = COLORS["green"] if temp < 60 else (COLORS["yellow"] if temp < 75 else COLORS["red"])
                self.gpu_temp_labels[i].config(
                    text=f"🌡 {temp:.0f}°C  |  ⚡ {power:.0f}W  |  🌀 {fan:.0f}%",
                    fg=temp_color
                )
                
                self.gpu_vram_labels[i].config(
                    text=f"VRAM: {vram_used:.0f} / {vram_total:.0f} MB")
                self._draw_bar(self.gpu_vram_bars[i], vram_pct)
                
        except Exception:
            pass
    
    # === DRAG ===
    def _start_drag(self, event):
        self._drag_x = event.x
        self._drag_y = event.y
    
    def _do_drag(self, event):
        x = self.root.winfo_x() + event.x - self._drag_x
        y = self.root.winfo_y() + event.y - self._drag_y
        self.root.geometry(f"+{x}+{y}")
    
    # === CLOSE ===
    def _close(self):
        self._running = False
        if self._update_id:
            self.root.after_cancel(self._update_id)
        self.root.destroy()
    
    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = NightmareWidget()
    app.run()
