"""
💀 NIGHTMARE SUITE v5 — TRAINING & GGUF EXPORT PANELS
========================================================
Two slide-in panels:
  1. 🏋️ Training Panel — Train models from the GUI (calls train_full_power.py)
  2. 📦 GGUF Export Panel — Convert checkpoints to GGUF format

RULES:
  - NO external libraries beyond stdlib + numpy
  - Pure black background, Consolas 11+ fonts, red/green accents
  - Same slide-in pattern as existing model_tools_panels.py
  - subprocess with CREATE_NO_WINDOW flag
  - Real-time output capture via PIPE + root.after() polling
  - CUDA_VISIBLE_DEVICES=0,1 always set

Author: Nightmare Designs 💀
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import subprocess
import json
import time
import struct
import re
import sys
from pathlib import Path
from typing import Optional, Dict, List

# === Paths ===
SCRIPT_DIR = Path(__file__).parent
# core_mind is in the suite directory (same as SCRIPT_DIR / core_mind)
CORE_MIND_DIR = SCRIPT_DIR / "core_mind"
# Training scripts are in the suite root, not core_mind
_TRAIN_CANDIDATES = [
    SCRIPT_DIR / "train_v3.py",
    SCRIPT_DIR / "train_v2.py",
    CORE_MIND_DIR / "train_full_power.py",
]
TRAIN_SCRIPT = next((p for p in _TRAIN_CANDIDATES if p.exists()), _TRAIN_CANDIDATES[0])
_EXPORT_CANDIDATES = [
    SCRIPT_DIR / "export_90m.py",
    CORE_MIND_DIR / "export_gguf.py",
]
EXPORT_SCRIPT = next((p for p in _EXPORT_CANDIDATES if p.exists()), _EXPORT_CANDIDATES[0])
CHECKPOINTS_DIR = CORE_MIND_DIR / "checkpoints"
TRAINING_DATA_DIR = CORE_MIND_DIR / "training_data"

# === Windows subprocess flag ===
CREATE_NO_WINDOW = 0x08000000

# Theme colors (same as main suite)
COLORS = {
    "bg": "#000000",
    "bg_panel": "#000000",
    "bg_card": "#0a0a0a",
    "bg_input": "#0d0d0d",
    "border": "#1a1a1a",
    "text": "#cccccc",
    "text_dim": "#666666",
    "text_bright": "#ffffff",
    "blood_red": "#8b0000",
    "blood_bright": "#ff0000",
    "matrix_green": "#00ff41",
    "warning": "#ff6600",
    "gold": "#ffd700",
    "accent": "#ff0000",
    "success": "#00ff41",
}

# Quantization types for GGUF export
QUANT_OPTIONS = [
    ("FP32", "f32", "Full precision — largest, no quality loss"),
    ("FP16", "f16", "Half precision — 50% size, minimal loss"),
    ("Q8_0", "q8_0", "8-bit — ~25% size, near-lossless"),
    ("Q5_K_M", "q5_k_m", "5-bit mixed — good balance"),
    ("Q4_K_M", "q4_k_m", "4-bit mixed — smaller, decent quality"),
    ("Q4_0", "q4_0", "4-bit basic — fast, some quality loss"),
    ("Q3_K", "q3_k", "3-bit — very small, noticeable loss"),
    ("Q2_K", "q2_k", "2-bit — extreme compression, significant loss"),
]

# Bytes per param for file size estimation
QUANT_BYTES_PER_PARAM = {
    "f32": 4.0, "f16": 2.0, "q8_0": 1.1, "q5_k_m": 0.7,
    "q4_k_m": 0.6, "q4_0": 0.55, "q3_k": 0.45, "q2_k": 0.35,
}


# ============================================================
# 🏋️ TRAINING PANEL
# ============================================================

class TrainingPanel:
    """
    Full training GUI panel — configure, launch, monitor, and control
    model training via subprocess. Streams output in real-time.
    """

    def __init__(self, parent_root, colors):
        self.root = parent_root
        self.colors = colors
        self.frame = None
        self.visible = False

        # Training state
        self.process: Optional[subprocess.Popen] = None
        self.training_active = False
        self.training_paused = False
        self.poll_id = None

        # Live stats
        self.current_step = 0
        self.total_steps = 0
        self.current_loss = 0.0
        self.tokens_per_sec = 0.0
        self.eta_str = "N/A"
        self.gpu_temp = "N/A"
        self.vram_usage = "N/A"
        self.loss_history: List[float] = []
        self.step_history: List[int] = []

    def toggle(self):
        if self.visible:
            self.frame.place_forget()
            self.visible = False
        else:
            self._build_if_needed()
            self.frame.place(relx=1.0, y=0, anchor="ne", relheight=1.0, width=520)
            self.visible = True

    def _build_if_needed(self):
        if self.frame:
            return

        self.frame = tk.Frame(self.root, bg=COLORS["bg"], bd=2, relief=tk.RIDGE,
                              highlightbackground=COLORS["blood_red"], highlightthickness=1)

        # Scrollable canvas
        canvas = tk.Canvas(self.frame, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=canvas.yview)
        self.scrollable = tk.Frame(canvas, bg=COLORS["bg"])

        self.scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable, anchor=tk.NW, width=500)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mousewheel
        def _mw(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _mw))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # === HEADER ===
        header = tk.Frame(self.scrollable, bg=COLORS["bg"])
        header.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(header, text="🏋️ TRAINING PANEL", font=("Consolas", 14, "bold"),
                 bg=COLORS["bg"], fg=COLORS["blood_bright"]).pack(side=tk.LEFT)
        tk.Button(header, text="✕", font=("Consolas", 12),
                  bg=COLORS["bg"], fg=COLORS["text_dim"], relief=tk.FLAT,
                  command=self.toggle).pack(side=tk.RIGHT)

        # Separator
        tk.Frame(self.scrollable, height=1, bg=COLORS["blood_red"]).pack(fill=tk.X, padx=10, pady=(0, 8))

        # === MODEL ARCHITECTURE ===
        self._section_header("⚙️ MODEL ARCHITECTURE")

        arch_frame = tk.Frame(self.scrollable, bg=COLORS["bg_card"], bd=1, relief=tk.FLAT)
        arch_frame.pack(fill=tk.X, padx=10, pady=4)

        # Preset selector
        preset_frame = tk.Frame(arch_frame, bg=COLORS["bg_card"])
        preset_frame.pack(fill=tk.X, padx=8, pady=(8, 4))
        tk.Label(preset_frame, text="Preset:", font=("Consolas", 11),
                 bg=COLORS["bg_card"], fg=COLORS["text_dim"]).pack(side=tk.LEFT)
        self.arch_preset = tk.StringVar(value="74.4M")
        preset_menu = tk.OptionMenu(preset_frame, self.arch_preset,
                                    "74.4M", "Custom",
                                    command=self._on_preset_change)
        preset_menu.config(bg=COLORS["bg_input"], fg=COLORS["matrix_green"],
                           font=("Consolas", 11), highlightthickness=0, relief=tk.FLAT)
        preset_menu["menu"].config(bg=COLORS["bg_input"], fg=COLORS["matrix_green"],
                                   font=("Consolas", 11))
        preset_menu.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(8, 0))

        # Architecture fields (2 columns)
        fields_frame = tk.Frame(arch_frame, bg=COLORS["bg_card"])
        fields_frame.pack(fill=tk.X, padx=8, pady=4)

        self.arch_vars = {}
        arch_fields = [
            ("Layers:", "n_layers", "20"),
            ("Dim:", "hidden_dim", "512"),
            ("Heads:", "n_heads", "8"),
            ("KV Heads:", "n_kv_heads", "4"),
            ("FFN Dim:", "ffn_dim", "1376"),
            ("Vocab:", "vocab_size", "32000"),
            ("Seq Len:", "max_seq_len", "2048"),
        ]

        for i, (label, key, default) in enumerate(arch_fields):
            row = i // 2
            col = i % 2
            f = tk.Frame(fields_frame, bg=COLORS["bg_card"])
            f.grid(row=row, column=col, sticky="ew", padx=4, pady=2)
            fields_frame.columnconfigure(col, weight=1)

            tk.Label(f, text=label, font=("Consolas", 11), width=9,
                     bg=COLORS["bg_card"], fg=COLORS["text_dim"], anchor=tk.W).pack(side=tk.LEFT)
            var = tk.StringVar(value=default)
            self.arch_vars[key] = var
            entry = tk.Entry(f, textvariable=var, font=("Consolas", 11), width=8,
                             bg=COLORS["bg_input"], fg=COLORS["matrix_green"],
                             relief=tk.FLAT, insertbackground=COLORS["matrix_green"])
            entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # Param count display
        self.param_label = tk.Label(arch_frame, text="≈ 74,400,000 params",
                                    font=("Consolas", 11, "bold"),
                                    bg=COLORS["bg_card"], fg=COLORS["gold"])
        self.param_label.pack(fill=tk.X, padx=8, pady=(4, 8))

        # === TRAINING DATA ===
        self._section_header("📂 TRAINING DATA")

        data_frame = tk.Frame(self.scrollable, bg=COLORS["bg_card"], bd=1, relief=tk.FLAT)
        data_frame.pack(fill=tk.X, padx=10, pady=4)

        data_path_frame = tk.Frame(data_frame, bg=COLORS["bg_card"])
        data_path_frame.pack(fill=tk.X, padx=8, pady=8)

        self.data_dir_var = tk.StringVar(value=str(TRAINING_DATA_DIR))
        tk.Label(data_path_frame, text="Data Dir:", font=("Consolas", 11),
                 bg=COLORS["bg_card"], fg=COLORS["text_dim"]).pack(side=tk.LEFT)
        tk.Entry(data_path_frame, textvariable=self.data_dir_var, font=("Consolas", 11),
                 bg=COLORS["bg_input"], fg=COLORS["text"],
                 relief=tk.FLAT, insertbackground=COLORS["text"]).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Button(data_path_frame, text="📁", font=("Consolas", 11),
                  bg=COLORS["bg_card"], fg=COLORS["matrix_green"], relief=tk.FLAT,
                  cursor="hand2", command=self._browse_data_dir).pack(side=tk.RIGHT)

        # === TOKENIZER STATUS ===
        self._section_header("🔤 TOKENIZER")

        tok_frame = tk.Frame(self.scrollable, bg=COLORS["bg_card"], bd=1, relief=tk.FLAT)
        tok_frame.pack(fill=tk.X, padx=10, pady=4)

        tok_inner = tk.Frame(tok_frame, bg=COLORS["bg_card"])
        tok_inner.pack(fill=tk.X, padx=8, pady=8)

        self.tok_status_label = tk.Label(tok_inner, text="Checking...",
                                         font=("Consolas", 11),
                                         bg=COLORS["bg_card"], fg=COLORS["text_dim"])
        self.tok_status_label.pack(side=tk.LEFT)

        self.tok_build_btn = tk.Button(tok_inner, text="🔨 Build", font=("Consolas", 11),
                                       bg=COLORS["bg_card"], fg=COLORS["warning"],
                                       relief=tk.FLAT, cursor="hand2",
                                       command=self._build_tokenizer)
        self.tok_build_btn.pack(side=tk.RIGHT)

        self._check_tokenizer_status()

        # === TRAINING CONFIG ===
        self._section_header("⚡ TRAINING CONFIG")

        config_frame = tk.Frame(self.scrollable, bg=COLORS["bg_card"], bd=1, relief=tk.FLAT)
        config_frame.pack(fill=tk.X, padx=10, pady=4)

        # Batch Size slider
        self._add_slider(config_frame, "Batch Size:", "batch_size", 1, 64, 8)
        # Learning Rate slider
        self._add_slider(config_frame, "Learning Rate (×1e-4):", "lr", 1, 100, 3)
        # Epochs
        self._add_slider(config_frame, "Epochs:", "epochs", 1, 100, 10)
        # Sequence Length
        self._add_slider(config_frame, "Seq Length:", "seq_len", 128, 4096, 2048, step=128)
        # Gradient Accumulation
        self._add_slider(config_frame, "Grad Accum:", "grad_accum", 1, 64, 8)

        # === GPU SELECTOR ===
        self._section_header("🖥️ GPU SELECTOR")

        gpu_frame = tk.Frame(self.scrollable, bg=COLORS["bg_card"], bd=1, relief=tk.FLAT)
        gpu_frame.pack(fill=tk.X, padx=10, pady=4)

        self.gpu_vars = {}
        gpu_inner = tk.Frame(gpu_frame, bg=COLORS["bg_card"])
        gpu_inner.pack(fill=tk.X, padx=8, pady=8)

        # Detect GPUs (best effort via nvidia-smi)
        self.detected_gpus = self._detect_gpus()
        if self.detected_gpus:
            for i, gpu_name in enumerate(self.detected_gpus):
                var = tk.BooleanVar(value=True)
                self.gpu_vars[i] = var
                tk.Checkbutton(gpu_inner, text=f"GPU {i}: {gpu_name}",
                               variable=var, font=("Consolas", 11),
                               bg=COLORS["bg_card"], fg=COLORS["matrix_green"],
                               selectcolor=COLORS["bg"], activebackground=COLORS["bg_card"],
                               activeforeground=COLORS["matrix_green"],
                               cursor="hand2").pack(anchor=tk.W, pady=1)
        else:
            tk.Label(gpu_inner, text="No GPUs detected (CPU mode)",
                     font=("Consolas", 11), bg=COLORS["bg_card"],
                     fg=COLORS["warning"]).pack(anchor=tk.W)

        # === CONTROL BUTTONS ===
        self._section_header("🎮 CONTROL")

        ctrl_frame = tk.Frame(self.scrollable, bg=COLORS["bg_card"], bd=1, relief=tk.FLAT)
        ctrl_frame.pack(fill=tk.X, padx=10, pady=4)

        btn_row = tk.Frame(ctrl_frame, bg=COLORS["bg_card"])
        btn_row.pack(fill=tk.X, padx=8, pady=8)

        self.start_btn = tk.Button(btn_row, text="▶ START", font=("Consolas", 12, "bold"),
                                   bg="#002200", fg=COLORS["matrix_green"],
                                   relief=tk.FLAT, cursor="hand2", width=10,
                                   command=self._start_training)
        self.start_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.pause_btn = tk.Button(btn_row, text="⏸ PAUSE", font=("Consolas", 12, "bold"),
                                   bg="#222200", fg=COLORS["warning"],
                                   relief=tk.FLAT, cursor="hand2", width=10,
                                   state=tk.DISABLED, command=self._pause_training)
        self.pause_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.stop_btn = tk.Button(btn_row, text="■ STOP", font=("Consolas", 12, "bold"),
                                  bg="#220000", fg=COLORS["blood_bright"],
                                  relief=tk.FLAT, cursor="hand2", width=10,
                                  state=tk.DISABLED, command=self._stop_training)
        self.stop_btn.pack(side=tk.LEFT)

        # === PROGRESS BAR ===
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = tk.Canvas(self.scrollable, height=20, bg=COLORS["bg_input"],
                                      highlightthickness=0)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=(8, 4))

        self.progress_label = tk.Label(self.scrollable, text="Idle",
                                       font=("Consolas", 11),
                                       bg=COLORS["bg"], fg=COLORS["text_dim"])
        self.progress_label.pack(fill=tk.X, padx=10)

        # === LIVE STATS ===
        self._section_header("📊 LIVE STATS")

        stats_frame = tk.Frame(self.scrollable, bg=COLORS["bg_card"], bd=1, relief=tk.FLAT)
        stats_frame.pack(fill=tk.X, padx=10, pady=4)

        stats_inner = tk.Frame(stats_frame, bg=COLORS["bg_card"])
        stats_inner.pack(fill=tk.X, padx=8, pady=8)

        self.stats_labels = {}
        stat_fields = [
            ("Step:", "step"), ("Loss:", "loss"), ("Tok/s:", "toks"),
            ("ETA:", "eta"), ("GPU Temp:", "gpu_temp"), ("VRAM:", "vram"),
        ]
        for i, (label, key) in enumerate(stat_fields):
            row = i // 2
            col = i % 2
            f = tk.Frame(stats_inner, bg=COLORS["bg_card"])
            f.grid(row=row, column=col, sticky="ew", padx=4, pady=2)
            stats_inner.columnconfigure(col, weight=1)

            tk.Label(f, text=label, font=("Consolas", 11), width=10,
                     bg=COLORS["bg_card"], fg=COLORS["text_dim"], anchor=tk.W).pack(side=tk.LEFT)
            val = tk.Label(f, text="—", font=("Consolas", 11),
                           bg=COLORS["bg_card"], fg=COLORS["matrix_green"], anchor=tk.W)
            val.pack(side=tk.LEFT)
            self.stats_labels[key] = val

        # === LOSS GRAPH (Canvas-based) ===
        self._section_header("📈 LOSS CURVE")

        self.loss_canvas = tk.Canvas(self.scrollable, height=120, bg=COLORS["bg_input"],
                                     highlightthickness=0)
        self.loss_canvas.pack(fill=tk.X, padx=10, pady=4)

        # === CHECKPOINTS ===
        self._section_header("💾 CHECKPOINTS")

        ckpt_frame = tk.Frame(self.scrollable, bg=COLORS["bg_card"], bd=1, relief=tk.FLAT)
        ckpt_frame.pack(fill=tk.X, padx=10, pady=4)

        self.ckpt_listbox = tk.Listbox(ckpt_frame, height=5, bg=COLORS["bg_input"],
                                       fg=COLORS["text"], font=("Consolas", 11),
                                       selectbackground=COLORS["blood_red"],
                                       selectforeground=COLORS["text_bright"],
                                       relief=tk.FLAT, highlightthickness=0)
        self.ckpt_listbox.pack(fill=tk.X, padx=8, pady=(8, 4))

        ckpt_btn_row = tk.Frame(ckpt_frame, bg=COLORS["bg_card"])
        ckpt_btn_row.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Button(ckpt_btn_row, text="🔄 Refresh", font=("Consolas", 11),
                  bg=COLORS["bg_card"], fg=COLORS["text"], relief=tk.FLAT,
                  cursor="hand2", command=self._refresh_checkpoints).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(ckpt_btn_row, text="▶ Resume", font=("Consolas", 11),
                  bg=COLORS["bg_card"], fg=COLORS["matrix_green"], relief=tk.FLAT,
                  cursor="hand2", command=self._resume_from_checkpoint).pack(side=tk.LEFT)

        self._refresh_checkpoints()

        # === TRAINING LOG ===
        self._section_header("📜 TRAINING LOG")

        log_frame = tk.Frame(self.scrollable, bg=COLORS["bg_card"], bd=1, relief=tk.FLAT)
        log_frame.pack(fill=tk.X, padx=10, pady=4)

        self.log_text = tk.Text(log_frame, height=12, bg=COLORS["bg_input"],
                                fg=COLORS["text"], font=("Consolas", 11),
                                relief=tk.FLAT, wrap=tk.WORD, state=tk.DISABLED,
                                insertbackground=COLORS["text"])
        log_scroll = tk.Scrollbar(log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Tag colors for log
        self.log_text.tag_configure("step", foreground=COLORS["matrix_green"])
        self.log_text.tag_configure("loss", foreground=COLORS["gold"])
        self.log_text.tag_configure("error", foreground=COLORS["blood_bright"])
        self.log_text.tag_configure("info", foreground=COLORS["text_dim"])

    # === UI HELPERS ===

    def _section_header(self, text):
        tk.Label(self.scrollable, text=text, font=("Consolas", 12, "bold"),
                 bg=COLORS["bg"], fg=COLORS["blood_bright"], anchor=tk.W).pack(
            fill=tk.X, padx=10, pady=(12, 2))

    def _add_slider(self, parent, label, key, min_val, max_val, default, step=1):
        """Add a labeled slider to the config section."""
        frame = tk.Frame(parent, bg=COLORS["bg_card"])
        frame.pack(fill=tk.X, padx=8, pady=4)

        tk.Label(frame, text=label, font=("Consolas", 11), width=22,
                 bg=COLORS["bg_card"], fg=COLORS["text_dim"], anchor=tk.W).pack(side=tk.LEFT)

        var = tk.IntVar(value=default)
        if not hasattr(self, 'config_vars'):
            self.config_vars = {}
        self.config_vars[key] = var

        val_label = tk.Label(frame, text=str(default), font=("Consolas", 11, "bold"),
                             bg=COLORS["bg_card"], fg=COLORS["matrix_green"], width=6)
        val_label.pack(side=tk.RIGHT)

        scale = tk.Scale(frame, from_=min_val, to=max_val, orient=tk.HORIZONTAL,
                         variable=var, resolution=step, showvalue=False,
                         bg=COLORS["bg_card"], fg=COLORS["matrix_green"],
                         troughcolor=COLORS["bg_input"], highlightthickness=0,
                         sliderrelief=tk.FLAT, length=150,
                         command=lambda v, lbl=val_label: lbl.config(text=str(int(float(v)))))
        scale.pack(side=tk.RIGHT, padx=(4, 4))

    # === GPU DETECTION ===

    def _detect_gpus(self) -> List[str]:
        """Detect NVIDIA GPUs via nvidia-smi."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
                creationflags=CREATE_NO_WINDOW
            )
            if result.returncode == 0:
                gpus = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
                return gpus
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            pass
        return []

    # === TOKENIZER ===

    def _check_tokenizer_status(self):
        """Check if tokenizer.json exists and show status."""
        tok_path = CORE_MIND_DIR / "tokenizer.json"
        if tok_path.exists():
            try:
                with open(tok_path, "r", encoding="utf-8") as f:
                    tok_data = json.load(f)
                vocab_size = len(tok_data.get("vocab", {}))
                self.tok_status_label.config(
                    text=f"✓ Built — {vocab_size:,} tokens",
                    fg=COLORS["matrix_green"])
                self.tok_build_btn.config(text="🔨 Rebuild")
            except Exception:
                self.tok_status_label.config(text="⚠ Corrupt tokenizer.json",
                                            fg=COLORS["warning"])
        else:
            self.tok_status_label.config(text="✗ Not built",
                                        fg=COLORS["blood_bright"])

    def _build_tokenizer(self):
        """Launch tokenizer builder (subprocess in dev mode, clear error in exe)."""
        builder_script = CORE_MIND_DIR / "tokenizer_builder.py"
        if not builder_script.exists() or getattr(sys, 'frozen', False):
            self._log_append(
                "[ERROR] Tokenizer builder not available.\n"
                "   tokenizer_builder.py is missing or running in standalone exe.\n"
                "   Run from source to build the tokenizer.\n", "error")
            return

        self._log_append("[TOKENIZER] Building tokenizer...\n", "info")

        def _run():
            try:
                proc = subprocess.Popen(
                    [sys.executable, str(builder_script)],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    cwd=str(CORE_MIND_DIR), text=True,
                    creationflags=CREATE_NO_WINDOW
                )
                for line in iter(proc.stdout.readline, ''):
                    self.root.after(0, self._log_append, line, "info")
                proc.wait()
                self.root.after(0, self._check_tokenizer_status)
                self.root.after(0, self._log_append, "[TOKENIZER] Done.\n", "step")
            except Exception as e:
                self.root.after(0, self._log_append, f"[ERROR] {e}\n", "error")

        threading.Thread(target=_run, daemon=True).start()

    # === TRAINING CONTROL ===

    def _browse_data_dir(self):
        """Browse for training data directory."""
        path = filedialog.askdirectory(initialdir=str(TRAINING_DATA_DIR),
                                       title="Select Training Data Directory")
        if path:
            self.data_dir_var.set(path)

    def _on_preset_change(self, value):
        """Update arch fields based on preset selection."""
        if value == "74.4M":
            defaults = {"n_layers": "20", "hidden_dim": "512", "n_heads": "8",
                        "n_kv_heads": "4", "ffn_dim": "1376", "vocab_size": "32000",
                        "max_seq_len": "2048"}
            for key, val in defaults.items():
                if key in self.arch_vars:
                    self.arch_vars[key].set(val)
            self.param_label.config(text="≈ 74,400,000 params")

    def _get_gpu_env(self) -> str:
        """Get CUDA_VISIBLE_DEVICES string from checkboxes."""
        selected = [str(i) for i, var in self.gpu_vars.items() if var.get()]
        return ",".join(selected) if selected else "0,1"

    def _start_training(self):
        """Launch training — subprocess in dev mode, in-process in the exe."""
        if self.training_active:
            return

        # Build environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self._get_gpu_env()

        epochs = self.config_vars.get("epochs", tk.IntVar(value=10)).get()

        self._log_append(f"[START] Launching training...\n", "step")
        self._log_append(f"  Script: {TRAIN_SCRIPT}\n", "info")
        self._log_append(f"  GPUs: {env['CUDA_VISIBLE_DEVICES']}\n", "info")
        self._log_append(f"  Epochs: {epochs}\n", "info")
        self._log_append(f"{'═' * 50}\n", "info")

        # Reset stats
        self.current_step = 0
        self.loss_history.clear()
        self.step_history.clear()
        self._update_loss_graph()

        # In the hardened exe, scripts are sealed and sys.executable is the exe itself.
        # Fall back to in-process import + thread with stdout capture.
        if getattr(sys, 'frozen', False) or not TRAIN_SCRIPT.exists():
            self._start_training_inprocess(epochs, env)
            return

        # Dev mode: subprocess with real-time output
        cmd = [sys.executable, str(TRAIN_SCRIPT), "--epochs", str(epochs)]
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(CORE_MIND_DIR),
                text=True,
                bufsize=1,
                env=env,
                creationflags=CREATE_NO_WINDOW
            )
            self.training_active = True
            self.training_paused = False
            self._update_buttons_active()
            self._poll_output()
        except Exception as e:
            self._log_append(f"[ERROR] Failed to start: {e}\n", "error")

    def _start_training_inprocess(self, epochs, env):
        """Run training in-process (for the hardened exe where scripts are sealed)."""
        import io, contextlib, queue as _q
        self._inproc_queue = _q.Queue()
        self._inproc_thread = None
        self._inproc_stop = False

        # Determine module name from script path
        mod_name = TRAIN_SCRIPT.stem  # e.g. 'train_v3'

        def _train_thread():
            old_argv = sys.argv[:]
            sys.argv = [str(TRAIN_SCRIPT), "--epochs", str(epochs)]
            old_env = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0,1")
            buf = io.StringIO()
            try:
                mod = __import__(mod_name)
                if hasattr(mod, 'main'):
                    with contextlib.redirect_stdout(buf):
                        mod.main()
                else:
                    self._inproc_queue.put("[ERROR] Training module has no main() function.\n")
            except Exception as e:
                self._inproc_queue.put(f"[ERROR] Training failed: {e}\n")
            finally:
                sys.argv = old_argv
                if old_env is not None:
                    os.environ["CUDA_VISIBLE_DEVICES"] = old_env
                remaining = buf.getvalue()
                if remaining:
                    for line in remaining.splitlines(True):
                        self._inproc_queue.put(line)
                self._inproc_queue.put(None)  # sentinel: done

        self.training_active = True
        self.training_paused = False
        self._update_buttons_active()
        self._inproc_thread = threading.Thread(target=_train_thread, daemon=True)
        self._inproc_thread.start()
        self._poll_inproc_output()

    def _pause_training(self):
        """Pause/resume training (send signal or toggle flag)."""
        if not self.process:
            return
        if self.training_paused:
            # Resume — on Windows we can't easily SIGCONT, so we just track state
            self.training_paused = False
            self.pause_btn.config(text="⏸ PAUSE", fg=COLORS["warning"])
            self._log_append("[RESUMED] Training resumed.\n", "step")
        else:
            self.training_paused = True
            self.pause_btn.config(text="▶ RESUME", fg=COLORS["matrix_green"])
            self._log_append("[PAUSED] Training paused (subprocess still running).\n", "info")

    def _stop_training(self):
        """Kill the training subprocess or in-process thread."""
        if hasattr(self, '_inproc_thread') and self._inproc_thread and self._inproc_thread.is_alive():
            self._inproc_stop = True
            self._inproc_thread = None  # thread is daemon, will be abandoned
        elif self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

        self.training_active = False
        self.training_paused = False
        self._update_buttons_idle()
        self._log_append("[STOPPED] Training terminated by user.\n", "error")

        if self.poll_id:
            self.root.after_cancel(self.poll_id)
            self.poll_id = None

    def _poll_output(self):
        """Poll subprocess stdout and update GUI. Uses root.after() for thread safety."""
        if not self.process:
            return

        try:
            # Non-blocking read via thread
            import select
            # On Windows, select doesn't work on pipes. Use a thread-based approach.
            self._read_output_chunk()
        except Exception:
            pass

        # Check if process is still running
        retcode = self.process.poll()
        if retcode is not None:
            # Process ended — read remaining output
            remaining = self.process.stdout.read()
            if remaining:
                self._process_output(remaining)
            self.training_active = False
            self._update_buttons_idle()
            status = "completed" if retcode == 0 else f"exited with code {retcode}"
            self._log_append(f"\n[DONE] Training {status}.\n", "step" if retcode == 0 else "error")
            self.process = None
            self._refresh_checkpoints()
            return

        # Schedule next poll (50ms interval for responsive updates)
        self.poll_id = self.root.after(50, self._poll_output)

    def _poll_inproc_output(self):
        """Poll the in-process training queue for output (exe mode)."""
        import queue as _q
        if not hasattr(self, '_inproc_queue'):
            return
        try:
            while True:
                try:
                    chunk = self._inproc_queue.get_nowait()
                    if chunk is None:
                        # sentinel — training done
                        self._log_append("\n[DONE] Training complete.\n", "step")
                        self.training_active = False
                        self.training_paused = False
                        self._update_buttons_idle()
                        self._refresh_checkpoints()
                        self._inproc_thread = None
                        return
                    self._log_append(chunk, "info")
                except _q.Empty:
                    break
        except Exception:
            pass
        if self.training_active:
            self.poll_id = self.root.after(50, self._poll_inproc_output)

    def _read_output_chunk(self):
        """Read available output from subprocess (non-blocking on Windows)."""
        if not self.process or not self.process.stdout:
            return

        # Read one line at a time (readline is blocking but fast for line-buffered output)
        # Use a thread to avoid GUI freeze
        if not hasattr(self, '_output_buffer'):
            self._output_buffer = []
            self._reader_thread_active = False

        if not self._reader_thread_active:
            self._reader_thread_active = True

            def _reader():
                try:
                    while self.process and self.training_active:
                        line = self.process.stdout.readline()
                        if line:
                            self._output_buffer.append(line)
                        else:
                            break
                except Exception:
                    pass
                finally:
                    self._reader_thread_active = False

            threading.Thread(target=_reader, daemon=True).start()

        # Process buffered output on the main thread
        while self._output_buffer:
            line = self._output_buffer.pop(0)
            self._process_output(line)

    def _process_output(self, text):
        """Parse training output and update stats + log."""
        for line in text.split('\n'):
            if not line.strip():
                continue

            # Try to parse step/loss/tok lines
            # Expected formats from train_full_power.py:
            # "Step 100/5000 | Loss: 4.2341 | LR: 0.0003 | Tok/s: 12345 | ..."
            step_match = re.search(r'Step\s+(\d+)[/|](\d+)', line)
            loss_match = re.search(r'Loss:\s*([\d.]+)', line)
            toks_match = re.search(r'Tok/s:\s*([\d.]+)', line)
            eta_match = re.search(r'ETA:\s*([^\s|]+)', line)

            if step_match:
                self.current_step = int(step_match.group(1))
                self.total_steps = int(step_match.group(2))
                self.stats_labels["step"].config(
                    text=f"{self.current_step}/{self.total_steps}")
                # Update progress bar
                if self.total_steps > 0:
                    pct = self.current_step / self.total_steps
                    self._draw_progress(pct)
                    self.progress_label.config(
                        text=f"{pct*100:.1f}% — Step {self.current_step}/{self.total_steps}")

            if loss_match:
                self.current_loss = float(loss_match.group(1))
                self.stats_labels["loss"].config(text=f"{self.current_loss:.4f}")
                self.loss_history.append(self.current_loss)
                self.step_history.append(self.current_step)
                if len(self.loss_history) % 5 == 0:
                    self._update_loss_graph()

            if toks_match:
                self.tokens_per_sec = float(toks_match.group(1))
                self.stats_labels["toks"].config(text=f"{self.tokens_per_sec:.0f}")

            if eta_match:
                self.eta_str = eta_match.group(1)
                self.stats_labels["eta"].config(text=self.eta_str)

            # Detect GPU stats if printed
            temp_match = re.search(r'Temp:\s*(\d+)\s*[°C]?', line)
            vram_match = re.search(r'VRAM:\s*([\d.]+\s*[GM]B)', line)
            if temp_match:
                self.stats_labels["gpu_temp"].config(text=f"{temp_match.group(1)}°C")
            if vram_match:
                self.stats_labels["vram"].config(text=vram_match.group(1))

            # Log the line
            tag = "step" if step_match else ("loss" if loss_match else "info")
            self._log_append(line + "\n", tag)

    def _draw_progress(self, pct):
        """Draw progress bar on canvas."""
        self.progress_bar.delete("all")
        w = self.progress_bar.winfo_width()
        h = self.progress_bar.winfo_height()
        if w < 10:
            w = 480
        if h < 5:
            h = 20

        # Background
        self.progress_bar.create_rectangle(0, 0, w, h, fill=COLORS["bg_input"], outline="")
        # Fill
        fill_w = int(w * pct)
        color = COLORS["matrix_green"] if pct < 1.0 else COLORS["gold"]
        self.progress_bar.create_rectangle(0, 0, fill_w, h, fill=color, outline="")
        # Percentage text
        self.progress_bar.create_text(w // 2, h // 2, text=f"{pct*100:.1f}%",
                                      fill=COLORS["text_bright"], font=("Consolas", 9, "bold"))

    def _update_loss_graph(self):
        """Draw a simple loss curve on the canvas."""
        self.loss_canvas.delete("all")
        w = self.loss_canvas.winfo_width()
        h = self.loss_canvas.winfo_height()
        if w < 10:
            w = 480
        if h < 10:
            h = 120

        if len(self.loss_history) < 2:
            self.loss_canvas.create_text(w // 2, h // 2, text="Waiting for data...",
                                         fill=COLORS["text_dim"], font=("Consolas", 11))
            return

        # Draw grid
        for i in range(5):
            y = int(h * i / 4)
            self.loss_canvas.create_line(0, y, w, y, fill=COLORS["border"], dash=(2, 4))

        # Scale data
        max_loss = max(self.loss_history)
        min_loss = min(self.loss_history)
        loss_range = max_loss - min_loss if max_loss > min_loss else 1.0
        n = len(self.loss_history)

        padding = 5
        graph_w = w - 2 * padding
        graph_h = h - 2 * padding

        points = []
        for i, loss in enumerate(self.loss_history):
            x = padding + int(graph_w * i / (n - 1))
            y = padding + int(graph_h * (1.0 - (loss - min_loss) / loss_range))
            points.append((x, y))

        # Draw loss line
        if len(points) >= 2:
            for i in range(len(points) - 1):
                self.loss_canvas.create_line(
                    points[i][0], points[i][1],
                    points[i + 1][0], points[i + 1][1],
                    fill=COLORS["matrix_green"], width=2)

        # Labels
        self.loss_canvas.create_text(padding, padding, text=f"{max_loss:.3f}",
                                     fill=COLORS["text_dim"], font=("Consolas", 8), anchor=tk.NW)
        self.loss_canvas.create_text(padding, h - padding, text=f"{min_loss:.3f}",
                                     fill=COLORS["text_dim"], font=("Consolas", 8), anchor=tk.SW)

    def _update_buttons_active(self):
        """Set button states for active training."""
        self.start_btn.config(state=tk.DISABLED)
        self.pause_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.NORMAL)

    def _update_buttons_idle(self):
        """Set button states for idle."""
        self.start_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)

    # === CHECKPOINTS ===

    def _refresh_checkpoints(self):
        """Scan checkpoints directory and populate listbox."""
        self.ckpt_listbox.delete(0, tk.END)
        if not CHECKPOINTS_DIR.exists():
            self.ckpt_listbox.insert(tk.END, "(no checkpoints directory)")
            return

        checkpoints = sorted(CHECKPOINTS_DIR.iterdir(), reverse=True)
        if not checkpoints:
            self.ckpt_listbox.insert(tk.END, "(no checkpoints found)")
            return

        for item in checkpoints:
            if item.is_dir():
                # Check for config.json to confirm it's a valid checkpoint
                has_config = (item / "config.json").exists()
                marker = "✓" if has_config else "?"
                self.ckpt_listbox.insert(tk.END, f"  {marker} {item.name}")

    def _resume_from_checkpoint(self):
        """Resume training from selected checkpoint."""
        sel = self.ckpt_listbox.curselection()
        if not sel:
            messagebox.showinfo("Resume", "Select a checkpoint first.")
            return

        entry = self.ckpt_listbox.get(sel[0])
        ckpt_name = entry.strip().lstrip("✓? ").strip()

        if self.training_active:
            messagebox.showwarning("Warning", "Training already running. Stop first.")
            return

        # Start with --resume flag
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = self._get_gpu_env()

        self._log_append(f"[RESUME] Resuming from: {ckpt_name}\n", "step")

        # In the exe, use in-process; in dev mode, use subprocess
        if getattr(sys, 'frozen', False) or not TRAIN_SCRIPT.exists():
            self._start_training_inprocess(0, env)  # epochs=0 means resume
            return

        cmd = [sys.executable, str(TRAIN_SCRIPT), "--resume"]
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(CORE_MIND_DIR),
                text=True,
                bufsize=1,
                env=env,
                creationflags=CREATE_NO_WINDOW
            )
            self.training_active = True
            self.training_paused = False
            self._update_buttons_active()
            self._poll_output()
        except Exception as e:
            self._log_append(f"[ERROR] Failed to resume: {e}\n", "error")

    # === LOG ===

    def _log_append(self, text, tag="info"):
        """Append text to training log with color tag."""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, text, tag)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)


# ============================================================
# 📦 GGUF EXPORT PANEL
# ============================================================

class GGUFExportPanel:
    """
    Convert trained model checkpoints to GGUF format.
    Calls export_gguf.py from core_mind/ with quantization options.
    """

    def __init__(self, parent_root, colors):
        self.root = parent_root
        self.colors = colors
        self.frame = None
        self.visible = False
        self.export_process: Optional[subprocess.Popen] = None
        self.exporting = False

    def toggle(self):
        if self.visible:
            self.frame.place_forget()
            self.visible = False
        else:
            self._build_if_needed()
            self.frame.place(relx=1.0, y=0, anchor="ne", relheight=1.0, width=480)
            self.visible = True

    def _build_if_needed(self):
        if self.frame:
            return

        self.frame = tk.Frame(self.root, bg=COLORS["bg"], bd=2, relief=tk.RIDGE,
                              highlightbackground=COLORS["blood_red"], highlightthickness=1)

        # Scrollable canvas
        canvas = tk.Canvas(self.frame, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=canvas.yview)
        self.scrollable = tk.Frame(canvas, bg=COLORS["bg"])

        self.scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable, anchor=tk.NW, width=460)
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Mousewheel
        def _mw(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _mw))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

        # === HEADER ===
        header = tk.Frame(self.scrollable, bg=COLORS["bg"])
        header.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(header, text="📦 GGUF EXPORT", font=("Consolas", 14, "bold"),
                 bg=COLORS["bg"], fg=COLORS["blood_bright"]).pack(side=tk.LEFT)
        tk.Button(header, text="✕", font=("Consolas", 12),
                  bg=COLORS["bg"], fg=COLORS["text_dim"], relief=tk.FLAT,
                  command=self.toggle).pack(side=tk.RIGHT)

        tk.Frame(self.scrollable, height=1, bg=COLORS["blood_red"]).pack(fill=tk.X, padx=10, pady=(0, 8))

        # === CHECKPOINT SELECTOR ===
        self._section_header("💾 SOURCE CHECKPOINT")

        ckpt_frame = tk.Frame(self.scrollable, bg=COLORS["bg_card"], bd=1, relief=tk.FLAT)
        ckpt_frame.pack(fill=tk.X, padx=10, pady=4)

        # Checkpoint list
        self.export_ckpt_listbox = tk.Listbox(ckpt_frame, height=5, bg=COLORS["bg_input"],
                                              fg=COLORS["text"], font=("Consolas", 11),
                                              selectbackground=COLORS["blood_red"],
                                              selectforeground=COLORS["text_bright"],
                                              relief=tk.FLAT, highlightthickness=0)
        self.export_ckpt_listbox.pack(fill=tk.X, padx=8, pady=(8, 4))
        self.export_ckpt_listbox.bind("<<ListboxSelect>>", self._on_ckpt_select)

        ckpt_btn_frame = tk.Frame(ckpt_frame, bg=COLORS["bg_card"])
        ckpt_btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        tk.Button(ckpt_btn_frame, text="🔄 Refresh", font=("Consolas", 11),
                  bg=COLORS["bg_card"], fg=COLORS["text"], relief=tk.FLAT,
                  cursor="hand2", command=self._refresh_export_checkpoints).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(ckpt_btn_frame, text="📁 Browse", font=("Consolas", 11),
                  bg=COLORS["bg_card"], fg=COLORS["matrix_green"], relief=tk.FLAT,
                  cursor="hand2", command=self._browse_checkpoint).pack(side=tk.LEFT)

        self.selected_ckpt_var = tk.StringVar(value="")
        self.selected_ckpt_label = tk.Label(ckpt_frame, textvariable=self.selected_ckpt_var,
                                            font=("Consolas", 11),
                                            bg=COLORS["bg_card"], fg=COLORS["matrix_green"],
                                            wraplength=430, anchor=tk.W)
        self.selected_ckpt_label.pack(fill=tk.X, padx=8, pady=(0, 8))

        # === OUTPUT PATH ===
        self._section_header("📂 OUTPUT")

        output_frame = tk.Frame(self.scrollable, bg=COLORS["bg_card"], bd=1, relief=tk.FLAT)
        output_frame.pack(fill=tk.X, padx=10, pady=4)

        out_inner = tk.Frame(output_frame, bg=COLORS["bg_card"])
        out_inner.pack(fill=tk.X, padx=8, pady=8)

        tk.Label(out_inner, text="Save to:", font=("Consolas", 11),
                 bg=COLORS["bg_card"], fg=COLORS["text_dim"]).pack(side=tk.LEFT)

        self.output_path_var = tk.StringVar(value=str(CORE_MIND_DIR / "core_mind.gguf"))
        tk.Entry(out_inner, textvariable=self.output_path_var, font=("Consolas", 11),
                 bg=COLORS["bg_input"], fg=COLORS["text"], relief=tk.FLAT,
                 insertbackground=COLORS["text"]).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Button(out_inner, text="📁", font=("Consolas", 11),
                  bg=COLORS["bg_card"], fg=COLORS["matrix_green"], relief=tk.FLAT,
                  cursor="hand2", command=self._browse_output).pack(side=tk.RIGHT)

        # === MODEL NAME ===
        name_frame = tk.Frame(output_frame, bg=COLORS["bg_card"])
        name_frame.pack(fill=tk.X, padx=8, pady=(0, 8))

        tk.Label(name_frame, text="Model Name:", font=("Consolas", 11),
                 bg=COLORS["bg_card"], fg=COLORS["text_dim"]).pack(side=tk.LEFT)
        self.model_name_var = tk.StringVar(value="Core Mind v1")
        tk.Entry(name_frame, textvariable=self.model_name_var, font=("Consolas", 11),
                 bg=COLORS["bg_input"], fg=COLORS["matrix_green"], relief=tk.FLAT,
                 insertbackground=COLORS["matrix_green"]).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # === QUANTIZATION ===
        self._section_header("⚗️ QUANTIZATION")

        quant_frame = tk.Frame(self.scrollable, bg=COLORS["bg_card"], bd=1, relief=tk.FLAT)
        quant_frame.pack(fill=tk.X, padx=10, pady=4)

        quant_inner = tk.Frame(quant_frame, bg=COLORS["bg_card"])
        quant_inner.pack(fill=tk.X, padx=8, pady=8)

        self.quant_var = tk.StringVar(value="f16")
        for name, value, desc in QUANT_OPTIONS:
            f = tk.Frame(quant_inner, bg=COLORS["bg_card"])
            f.pack(fill=tk.X, pady=1)
            tk.Radiobutton(f, text=f"{name:8s}", variable=self.quant_var, value=value,
                           font=("Consolas", 11), bg=COLORS["bg_card"],
                           fg=COLORS["matrix_green"], selectcolor=COLORS["bg"],
                           activebackground=COLORS["bg_card"],
                           activeforeground=COLORS["matrix_green"],
                           cursor="hand2",
                           command=self._update_size_estimate).pack(side=tk.LEFT)
            tk.Label(f, text=desc, font=("Consolas", 9),
                     bg=COLORS["bg_card"], fg=COLORS["text_dim"]).pack(side=tk.LEFT, padx=(4, 0))

        # Size estimate
        self.size_estimate_label = tk.Label(quant_frame, text="Estimated size: ~149 MB (FP16)",
                                            font=("Consolas", 11, "bold"),
                                            bg=COLORS["bg_card"], fg=COLORS["gold"])
        self.size_estimate_label.pack(fill=tk.X, padx=8, pady=(4, 8))

        # === ARCHITECTURE METADATA (auto-fills) ===
        self._section_header("🏗️ ARCHITECTURE (auto-detected)")

        meta_frame = tk.Frame(self.scrollable, bg=COLORS["bg_card"], bd=1, relief=tk.FLAT)
        meta_frame.pack(fill=tk.X, padx=10, pady=4)

        meta_inner = tk.Frame(meta_frame, bg=COLORS["bg_card"])
        meta_inner.pack(fill=tk.X, padx=8, pady=8)

        self.meta_labels = {}
        meta_fields = ["Layers", "Dim", "Heads", "KV Heads", "FFN Dim", "Vocab", "Seq Len", "Params"]
        for i, field in enumerate(meta_fields):
            row = i // 2
            col = i % 2
            f = tk.Frame(meta_inner, bg=COLORS["bg_card"])
            f.grid(row=row, column=col, sticky="ew", padx=4, pady=2)
            meta_inner.columnconfigure(col, weight=1)

            tk.Label(f, text=f"{field}:", font=("Consolas", 11), width=10,
                     bg=COLORS["bg_card"], fg=COLORS["text_dim"], anchor=tk.W).pack(side=tk.LEFT)
            lbl = tk.Label(f, text="—", font=("Consolas", 11),
                           bg=COLORS["bg_card"], fg=COLORS["text"], anchor=tk.W)
            lbl.pack(side=tk.LEFT)
            self.meta_labels[field] = lbl

        # === EXPORT BUTTON ===
        self._section_header("🚀 EXPORT")

        export_btn_frame = tk.Frame(self.scrollable, bg=COLORS["bg_card"], bd=1, relief=tk.FLAT)
        export_btn_frame.pack(fill=tk.X, padx=10, pady=4)

        self.export_btn = tk.Button(export_btn_frame, text="📦 EXPORT TO GGUF",
                                    font=("Consolas", 13, "bold"),
                                    bg="#002200", fg=COLORS["matrix_green"],
                                    relief=tk.FLAT, cursor="hand2",
                                    command=self._start_export)
        self.export_btn.pack(fill=tk.X, padx=8, pady=8, ipady=6)

        # Progress bar
        self.export_progress = tk.Canvas(self.scrollable, height=20, bg=COLORS["bg_input"],
                                         highlightthickness=0)
        self.export_progress.pack(fill=tk.X, padx=10, pady=(4, 2))

        self.export_status_label = tk.Label(self.scrollable, text="Ready",
                                            font=("Consolas", 11),
                                            bg=COLORS["bg"], fg=COLORS["text_dim"])
        self.export_status_label.pack(fill=tk.X, padx=10)

        # === EXPORT LOG ===
        self._section_header("📜 EXPORT LOG")

        log_frame = tk.Frame(self.scrollable, bg=COLORS["bg_card"], bd=1, relief=tk.FLAT)
        log_frame.pack(fill=tk.X, padx=10, pady=4)

        self.export_log = tk.Text(log_frame, height=10, bg=COLORS["bg_input"],
                                  fg=COLORS["text"], font=("Consolas", 11),
                                  relief=tk.FLAT, wrap=tk.WORD, state=tk.DISABLED,
                                  insertbackground=COLORS["text"])
        log_scroll = tk.Scrollbar(log_frame, command=self.export_log.yview)
        self.export_log.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=8)
        self.export_log.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.export_log.tag_configure("success", foreground=COLORS["matrix_green"])
        self.export_log.tag_configure("error", foreground=COLORS["blood_bright"])
        self.export_log.tag_configure("info", foreground=COLORS["text_dim"])

        # Load initial checkpoint list
        self._refresh_export_checkpoints()

    # === UI HELPERS ===

    def _section_header(self, text):
        tk.Label(self.scrollable, text=text, font=("Consolas", 12, "bold"),
                 bg=COLORS["bg"], fg=COLORS["blood_bright"], anchor=tk.W).pack(
            fill=tk.X, padx=10, pady=(12, 2))

    # === CHECKPOINT MANAGEMENT ===

    def _refresh_export_checkpoints(self):
        """Scan checkpoints directory."""
        self.export_ckpt_listbox.delete(0, tk.END)
        if not CHECKPOINTS_DIR.exists():
            self.export_ckpt_listbox.insert(tk.END, "(no checkpoints directory)")
            return

        found = False
        for item in sorted(CHECKPOINTS_DIR.iterdir(), reverse=True):
            if item.is_dir():
                has_config = (item / "config.json").exists()
                if has_config:
                    marker = "✓"
                    found = True
                else:
                    marker = "?"
                self.export_ckpt_listbox.insert(tk.END, f"  {marker} {item.name}")

        if not found:
            self.export_ckpt_listbox.insert(tk.END, "(no valid checkpoints)")

    def _browse_checkpoint(self):
        """Browse for a checkpoint directory."""
        path = filedialog.askdirectory(initialdir=str(CHECKPOINTS_DIR),
                                       title="Select Checkpoint Directory")
        if path:
            self.selected_ckpt_var.set(path)
            self._load_checkpoint_meta(Path(path))

    def _on_ckpt_select(self, event):
        """Handle checkpoint selection from listbox."""
        sel = self.export_ckpt_listbox.curselection()
        if not sel:
            return
        entry = self.export_ckpt_listbox.get(sel[0])
        ckpt_name = entry.strip().lstrip("✓? ").strip()
        ckpt_path = CHECKPOINTS_DIR / ckpt_name
        if ckpt_path.exists():
            self.selected_ckpt_var.set(str(ckpt_path))
            self._load_checkpoint_meta(ckpt_path)

    def _load_checkpoint_meta(self, ckpt_path: Path):
        """Load config.json from checkpoint and populate metadata fields."""
        config_file = ckpt_path / "config.json"
        if not config_file.exists():
            for key in self.meta_labels:
                self.meta_labels[key].config(text="?", fg=COLORS["warning"])
            return

        try:
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            field_map = {
                "Layers": "n_layers", "Dim": "hidden_dim", "Heads": "n_heads",
                "KV Heads": "n_kv_heads", "FFN Dim": "ffn_dim",
                "Vocab": "vocab_size", "Seq Len": "max_seq_len",
                "Params": "total_params",
            }
            for display_name, config_key in field_map.items():
                val = config.get(config_key, "—")
                if isinstance(val, int) and val > 1000:
                    text = f"{val:,}"
                else:
                    text = str(val)
                self.meta_labels[display_name].config(text=text, fg=COLORS["matrix_green"])

            # Update size estimate
            self._update_size_estimate()

            # Auto-set output name
            model_name = config.get("model_name", ckpt_path.name)
            self.model_name_var.set(model_name)

        except Exception as e:
            for key in self.meta_labels:
                self.meta_labels[key].config(text="error", fg=COLORS["blood_bright"])

    def _browse_output(self):
        """Browse for output file location."""
        path = filedialog.asksaveasfilename(
            initialdir=str(CORE_MIND_DIR),
            title="Save GGUF As",
            defaultextension=".gguf",
            filetypes=[("GGUF Models", "*.gguf"), ("All Files", "*.*")]
        )
        if path:
            self.output_path_var.set(path)

    def _update_size_estimate(self):
        """Estimate output file size based on param count and quantization."""
        # Try to get param count from metadata
        params_text = self.meta_labels.get("Params", None)
        params = 74_400_000  # Default to the assistant's count

        if params_text:
            try:
                raw = params_text.cget("text").replace(",", "").replace("—", "0")
                if raw.isdigit():
                    params = int(raw)
            except Exception:
                pass

        quant = self.quant_var.get()
        bytes_per = QUANT_BYTES_PER_PARAM.get(quant, 2.0)
        size_mb = params * bytes_per / (1024 ** 2)

        quant_name = quant.upper().replace("_", " ")
        self.size_estimate_label.config(
            text=f"Estimated size: ~{size_mb:.0f} MB ({quant_name})")

    # === EXPORT ===

    def _start_export(self):
        """Launch export_gguf.py subprocess."""
        if self.exporting:
            return

        ckpt_path = self.selected_ckpt_var.get()
        if not ckpt_path or not Path(ckpt_path).exists():
            messagebox.showerror("Error", "Select a valid checkpoint first.")
            return

        output_path = self.output_path_var.get()
        if not output_path:
            messagebox.showerror("Error", "Set an output path.")
            return

        quant = self.quant_var.get()
        model_name = self.model_name_var.get()

        # Check if export script exists (or if we're in the exe)
        if not EXPORT_SCRIPT.exists() or getattr(sys, 'frozen', False):
            messagebox.showerror("Error",
                f"Export script not available.\n"
                f"The export tool requires running from source.\n"
                f"Expected: {EXPORT_SCRIPT.name}")
            return

        self.exporting = True
        self.export_btn.config(state=tk.DISABLED, text="⏳ Exporting...")
        self._export_log_append(f"[EXPORT] Starting GGUF export...\n", "info")
        self._export_log_append(f"  Checkpoint: {ckpt_path}\n", "info")
        self._export_log_append(f"  Output: {output_path}\n", "info")
        self._export_log_append(f"  Quantization: {quant}\n", "info")
        self._export_log_append(f"  Model Name: {model_name}\n", "info")
        self._export_log_append(f"{'═' * 45}\n", "info")

        # Draw indeterminate progress
        self._draw_export_progress(0.1)
        self.export_status_label.config(text="Exporting...", fg=COLORS["warning"])

        def _run_export():
            try:
                # Build command — handle quantization via dtype flag
                # The existing export_gguf.py supports --dtype f32/f16
                # For extended quant, we pass --quantize
                cmd = [
                    sys.executable, str(EXPORT_SCRIPT),
                    "--checkpoint", ckpt_path,
                    "--output", output_path,
                    "--dtype", quant if quant in ("f32", "f16") else "f16",
                ]

                # For advanced quantization beyond f32/f16, we'd need extended support
                # Pass as extra arg for the updated export script
                if quant not in ("f32", "f16"):
                    cmd.extend(["--quantize", quant])

                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = "0,1"

                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=str(CORE_MIND_DIR),
                    text=True,
                    creationflags=CREATE_NO_WINDOW,
                    env=env
                )

                output_lines = []
                for line in iter(proc.stdout.readline, ''):
                    output_lines.append(line)
                    tag = "success" if "✓" in line else "info"
                    self.root.after(0, self._export_log_append, line, tag)

                proc.wait()

                if proc.returncode == 0:
                    # Get file size
                    out_file = Path(output_path)
                    if out_file.exists():
                        size_mb = out_file.stat().st_size / (1024 ** 2)
                        self.root.after(0, self._export_log_append,
                                        f"\n✓ EXPORT COMPLETE — {size_mb:.1f} MB\n", "success")
                    else:
                        self.root.after(0, self._export_log_append,
                                        f"\n✓ Export finished.\n", "success")
                    self.root.after(0, self._draw_export_progress, 1.0)
                    self.root.after(0, self.export_status_label.config,
                                   {"text": "Export complete!", "fg": COLORS["matrix_green"]})
                else:
                    self.root.after(0, self._export_log_append,
                                    f"\n✗ Export failed (code {proc.returncode})\n", "error")
                    self.root.after(0, self.export_status_label.config,
                                   {"text": "Export failed!", "fg": COLORS["blood_bright"]})

            except Exception as e:
                self.root.after(0, self._export_log_append, f"\n[ERROR] {e}\n", "error")
                self.root.after(0, self.export_status_label.config,
                               {"text": f"Error: {e}", "fg": COLORS["blood_bright"]})
            finally:
                self.exporting = False
                self.root.after(0, self.export_btn.config,
                               {"state": tk.NORMAL, "text": "📦 EXPORT TO GGUF"})

        threading.Thread(target=_run_export, daemon=True).start()

    def _draw_export_progress(self, pct):
        """Draw export progress bar."""
        self.export_progress.delete("all")
        w = self.export_progress.winfo_width()
        h = self.export_progress.winfo_height()
        if w < 10:
            w = 440
        if h < 5:
            h = 20

        self.export_progress.create_rectangle(0, 0, w, h, fill=COLORS["bg_input"], outline="")
        fill_w = int(w * pct)
        color = COLORS["matrix_green"] if pct < 1.0 else COLORS["gold"]
        self.export_progress.create_rectangle(0, 0, fill_w, h, fill=color, outline="")
        self.export_progress.create_text(w // 2, h // 2, text=f"{pct*100:.0f}%",
                                         fill=COLORS["text_bright"], font=("Consolas", 9, "bold"))

    def _export_log_append(self, text, tag="info"):
        """Append to export log."""
        self.export_log.config(state=tk.NORMAL)
        self.export_log.insert(tk.END, text, tag)
        self.export_log.see(tk.END)
        self.export_log.config(state=tk.DISABLED)


# ============================================================
# REGISTRATION FUNCTION
# ============================================================

def register_training_panels(app_instance):
    """
    Call this from NightmareAILoader.__init__ to register training panels.
    Returns a dict of button_text -> toggle_command for sidebar/admin integration.
    """
    root = app_instance.root

    # Create panel instances
    app_instance._training_panel = TrainingPanel(root, COLORS)
    app_instance._gguf_export_panel = GGUFExportPanel(root, COLORS)

    return {
        "🏋️ Train": app_instance._training_panel.toggle,
        "📦 Export": app_instance._gguf_export_panel.toggle,
    }
