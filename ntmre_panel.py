"""
╔══════════════════════════════════════════════════════════════╗
║  NTMRE TOOLS PANEL — Full Pipeline UI                        ║
║  Convert • Train • Compress • Decompress • Export            ║
║  Wraps the existing engine/ scripts — NO rewrites.           ║
║  Nightmare Designs 💀                                        ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import sys
import threading
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

APP_DIR = Path(__file__).parent
ENGINE_DIR = APP_DIR / "engine"

# Import COLORS from the main suite (injected at registration time)
COLORS = None


def register_ntmre_panels(app_instance, colors):
    """Register the NTMRE pipeline panels. Called from NightmareAILoader.__init__."""
    global COLORS
    COLORS = colors

    app_instance._ntmre_panel = NtmreToolsPanel(app_instance.root, colors)

    return {
        "🔄 NTMRE Convert": app_instance._ntmre_panel.show_convert,
        "🔷 Bonsai Convert": app_instance._ntmre_panel.show_bonsai,
        "💎 Gemma Convert": app_instance._ntmre_panel.show_gemma,
        "🏋️ Crucible Train": app_instance._ntmre_panel.show_train,
        "📦 Collapse (.ntmre compression)": app_instance._ntmre_panel.show_collapse,
        "🔓 Decompress": app_instance._ntmre_panel.show_decompress,
        "📤 NTMRE Export": app_instance._ntmre_panel.show_export,
        "☠️ Full Panel": app_instance._ntmre_panel.show_full,
    }


class NtmreToolsPanel:
    """Unified NTMRE pipeline tools panel."""

    def __init__(self, root, colors):
        self.root = root
        self.colors = colors
        self.process = None
        self._window = None

    # ════════════════════════════════════════════════════════
    # FULL PANEL (all tools in one window)
    # ════════════════════════════════════════════════════════

    def show_full(self):
        """Show the full NTMRE tools window."""
        if self._window and self._window.winfo_exists():
            self._window.lift()
            return

        self._window = tk.Toplevel(self.root)
        self._window.title("☠️ NTMRE PIPELINE — Nightmare Engine Tools")
        self._window.geometry("750x650")
        self._window.configure(bg=self.colors["bg"])

        # Title
        tk.Label(self._window, text="☠️ NTMRE PIPELINE",
                 font=("Impact", 20, "bold"), bg=self.colors["bg"],
                 fg=self.colors["matrix_green"]).pack(pady=(12, 2))
        tk.Label(self._window, text="Convert • Train • Collapse • Export",
                 font=("Consolas", 10), bg=self.colors["bg"],
                 fg=self.colors["text_dim"]).pack(pady=(0, 10))

        # Notebook (tabbed interface)
        style = ttk.Style()
        style.theme_use('default')
        style.configure("NTMRE.TNotebook", background=self.colors["bg"])
        style.configure("NTMRE.TNotebook.Tab", background=self.colors["bg_card"],
                        foreground=self.colors["text"], padding=[10, 4])
        style.map("NTMRE.TNotebook.Tab",
                  background=[("selected", self.colors["blood_red"])],
                  foreground=[("selected", "#ffffff")])

        nb = ttk.Notebook(self._window, style="NTMRE.TNotebook")
        nb.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Tab 1: Convert
        convert_frame = tk.Frame(nb, bg=self.colors["bg"])
        nb.add(convert_frame, text=" 🔄 Convert ")
        self._build_convert_tab(convert_frame)

        # Tab 2: Train (Crucible)
        train_frame = tk.Frame(nb, bg=self.colors["bg"])
        nb.add(train_frame, text=" 🏋️ Train ")
        self._build_train_tab(train_frame)

        # Tab 3: Collapse (.ntmre compression Compress)
        collapse_frame = tk.Frame(nb, bg=self.colors["bg"])
        nb.add(collapse_frame, text=" 📦 Collapse ")
        self._build_collapse_tab(collapse_frame)

        # Architecture-specific conversion helpers
        bonsai_frame = tk.Frame(nb, bg=self.colors["bg"])
        nb.add(bonsai_frame, text=" 🔷 Bonsai ")
        self._build_bonsai_tab(bonsai_frame)

        gemma_frame = tk.Frame(nb, bg=self.colors["bg"])
        nb.add(gemma_frame, text=" 💎 Gemma ")
        self._build_gemma_tab(gemma_frame)

        # Tab 4: Export
        export_frame = tk.Frame(nb, bg=self.colors["bg"])
        nb.add(export_frame, text=" 📤 Export ")
        self._build_export_tab(export_frame)

        # Log output (shared)
        tk.Label(self._window, text="OUTPUT", font=("Consolas", 9, "bold"),
                 bg=self.colors["bg"], fg=self.colors["text_dim"]).pack(anchor=tk.W, padx=10)
        self._log = tk.Text(self._window, height=12, bg="#0a0a0a",
                            fg=self.colors["matrix_green"], font=("Consolas", 10),
                            relief=tk.FLAT, bd=0, wrap=tk.WORD)
        self._log.pack(fill=tk.BOTH, expand=True, padx=8, pady=(2, 8))
        self._log.insert("1.0", "[NTMRE] Ready. Select a tool above.\n")

    # ════════════════════════════════════════════════════════
    # CONVERT TAB
    # ════════════════════════════════════════════════════════

    def _build_convert_tab(self, parent):
        """Convert: Any model → .ntmre"""
        f = tk.Frame(parent, bg=self.colors["bg"], padx=16, pady=12)
        f.pack(fill=tk.BOTH, expand=True)

        tk.Label(f, text="Convert any model to .ntmre format",
                 font=("Consolas", 11), bg=self.colors["bg"],
                 fg=self.colors["text"]).pack(anchor=tk.W)
        tk.Label(f, text="Supports: GGUF, SafeTensors, PyTorch .bin, Crucible checkpoints",
                 font=("Consolas", 9), bg=self.colors["bg"],
                 fg=self.colors["text_dim"]).pack(anchor=tk.W, pady=(0, 12))

        # Input file
        input_row = tk.Frame(f, bg=self.colors["bg"])
        input_row.pack(fill=tk.X, pady=4)
        tk.Label(input_row, text="Input:", font=("Consolas", 10),
                 bg=self.colors["bg"], fg=self.colors["text"], width=8).pack(side=tk.LEFT)
        self._convert_input = tk.Entry(input_row, bg=self.colors["bg_input"],
                                        fg=self.colors["text_bright"],
                                        font=("Consolas", 10), relief=tk.FLAT)
        self._convert_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Button(input_row, text="Browse", command=self._browse_convert_input,
                  bg=self.colors["bg_card"], fg=self.colors["matrix_green"],
                  font=("Consolas", 9), relief=tk.FLAT).pack(side=tk.RIGHT)

        # Output file
        out_row = tk.Frame(f, bg=self.colors["bg"])
        out_row.pack(fill=tk.X, pady=4)
        tk.Label(out_row, text="Output:", font=("Consolas", 10),
                 bg=self.colors["bg"], fg=self.colors["text"], width=8).pack(side=tk.LEFT)
        self._convert_output = tk.Entry(out_row, bg=self.colors["bg_input"],
                                         fg=self.colors["text_bright"],
                                         font=("Consolas", 10), relief=tk.FLAT)
        self._convert_output.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # Convert button
        tk.Button(f, text="⚡ CONVERT TO .NTMRE", command=self._run_convert,
                  bg=self.colors["blood_red"], fg="#ffffff",
                  font=("Consolas", 12, "bold"), relief=tk.FLAT,
                  cursor="hand2", height=2).pack(fill=tk.X, pady=(16, 4))

    def _browse_convert_input(self):
        path = filedialog.askopenfilename(
            title="Select model file",
            filetypes=[("All Models", "*.gguf *.safetensors *.bin *.pt *.ntmre"),
                       ("GGUF", "*.gguf"), ("SafeTensors", "*.safetensors"),
                       ("All", "*.*")])
        if path:
            self._convert_input.delete(0, tk.END)
            self._convert_input.insert(0, path)
            # Auto-set output name
            out = str(Path(path).with_suffix('.ntmre'))
            self._convert_output.delete(0, tk.END)
            self._convert_output.insert(0, out)

    def _run_convert(self):
        input_path = self._convert_input.get().strip()
        output_path = self._convert_output.get().strip()
        if not input_path:
            messagebox.showwarning("Convert", "Select an input file first.")
            return

        def do_convert():
            try:
                from nightmare_convert import convert_model
            except ImportError:
                try:
                    from engine.nightmare_convert import convert_model
                except ImportError:
                    self.root.after(0, self._log_msg,
                        "[ERROR] nightmare_convert module not found.\n")
                    return
            convert_model(input_path, output_path if output_path else None)

        self._run_in_process(do_convert, "CONVERT")

    def _build_bonsai_tab(self, parent):
        """Convert Qwen3.5 Q1_0 Bonsai models with the dedicated reader."""
        f = tk.Frame(parent, bg=self.colors["bg"], padx=16, pady=12)
        f.pack(fill=tk.BOTH, expand=True)
        tk.Label(f, text="Bonsai — Qwen3.5 Q1_0 → NTMRE .ntmre compression",
                 font=("Consolas", 11), bg=self.colors["bg"],
                 fg=self.colors["text"]).pack(anchor=tk.W)
        tk.Label(f, text="Uses the Qwen3.5 hybrid SSM/attention converter.",
                 font=("Consolas", 9), bg=self.colors["bg"],
                 fg=self.colors["text_dim"]).pack(anchor=tk.W, pady=(0, 12))
        self._bonsai_input = self._conversion_entry(f, "Input:", "*.gguf")
        self._bonsai_output = self._conversion_entry(f, "Output:", "*.ntmre", browse=False)
        rank_row = tk.Frame(f, bg=self.colors["bg"])
        rank_row.pack(fill=tk.X, pady=8)
        tk.Label(rank_row, text="Rank:", width=8, anchor=tk.W,
                 font=("Consolas", 10), bg=self.colors["bg"],
                 fg=self.colors["text"]).pack(side=tk.LEFT)
        self._bonsai_rank = tk.Entry(rank_row, width=10, bg=self.colors["bg_input"],
                                     fg=self.colors["text_bright"], font=("Consolas", 10),
                                     relief=tk.FLAT)
        self._bonsai_rank.insert(0, "512")
        self._bonsai_rank.pack(side=tk.LEFT, padx=4)
        tk.Button(f, text="🔷 CONVERT BONSAI", command=self._run_bonsai,
                  bg=self.colors["blood_red"], fg="#ffffff",
                  font=("Consolas", 12, "bold"), relief=tk.FLAT,
                  cursor="hand2", height=2).pack(fill=tk.X, pady=(16, 4))

    def _build_gemma_tab(self, parent):
        """Convert Gemma GGUF through the suite's generic GGUF converter."""
        f = tk.Frame(parent, bg=self.colors["bg"], padx=16, pady=12)
        f.pack(fill=tk.BOTH, expand=True)
        tk.Label(f, text="Gemma — GGUF → NTMRE .ntmre compression",
                 font=("Consolas", 11), bg=self.colors["bg"],
                 fg=self.colors["text"]).pack(anchor=tk.W)
        tk.Label(f, text="Uses metadata and tensor-shape detection from nightmare_convert.py.",
                 font=("Consolas", 9), bg=self.colors["bg"],
                 fg=self.colors["text_dim"]).pack(anchor=tk.W, pady=(0, 12))
        self._gemma_input = self._conversion_entry(f, "Input:", "*.gguf")
        self._gemma_output = self._conversion_entry(f, "Output:", "*.ntmre", browse=False)
        tk.Button(f, text="💎 CONVERT GEMMA", command=self._run_gemma,
                  bg=self.colors["blood_red"], fg="#ffffff",
                  font=("Consolas", 12, "bold"), relief=tk.FLAT,
                  cursor="hand2", height=2).pack(fill=tk.X, pady=(16, 4))

    def _conversion_entry(self, parent, label, pattern, browse=True):
        row = tk.Frame(parent, bg=self.colors["bg"])
        row.pack(fill=tk.X, pady=4)
        tk.Label(row, text=label, width=8, anchor=tk.W,
                 font=("Consolas", 10), bg=self.colors["bg"],
                 fg=self.colors["text"]).pack(side=tk.LEFT)
        entry = tk.Entry(row, bg=self.colors["bg_input"], fg=self.colors["text_bright"],
                         font=("Consolas", 10), relief=tk.FLAT)
        entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        if browse:
            def browse_entry():
                path = filedialog.askopenfilename(filetypes=[("Model", pattern), ("All", "*.*")])
                if path:
                    entry.delete(0, tk.END)
                    entry.insert(0, path)
            tk.Button(row, text="Browse", command=browse_entry,
                      bg=self.colors["bg_card"], fg=self.colors["matrix_green"],
                      font=("Consolas", 9), relief=tk.FLAT).pack(side=tk.RIGHT)
        return entry

    def _run_bonsai(self):
        src = self._bonsai_input.get().strip()
        dst = self._bonsai_output.get().strip() or str(Path(src).with_suffix(".ntmre"))
        if not src:
            messagebox.showwarning("Bonsai", "Select a Q1_0 GGUF file first.")
            return
        candidates = [
            APP_DIR.parent / "nightmare_app_final" / "bonsai_convert.py",
            APP_DIR.parent.parent / "bio 2" / "bonsai_convert.py",
            APP_DIR.parent / "bio 2" / "bonsai_convert.py",
        ]
        script = next((path for path in candidates if path.exists()), None)
        if script is None:
            self._log_msg("[ERROR] Bonsai converter not found. Expected bio 2/bonsai_convert.py\n")
            return
        cmd = [sys.executable, str(script), src, dst]
        rank = self._bonsai_rank.get().strip()
        if rank:
            cmd += ["--rank", rank]
        self._run_command(cmd, "BONSAI")

    def _run_gemma(self):
        src = self._gemma_input.get().strip()
        dst = self._gemma_output.get().strip() or str(Path(src).with_suffix(".ntmre"))
        if not src:
            messagebox.showwarning("Gemma", "Select a Gemma GGUF file first.")
            return
        # In-process conversion — works in the hardened exe
        def do_gemma():
            try:
                from nightmare_convert import convert_model
            except ImportError:
                try:
                    from engine.nightmare_convert import convert_model
                except ImportError:
                    self.root.after(0, self._log_msg,
                        "[ERROR] nightmare_convert module not found.\n")
                    return
            convert_model(src, dst)
        self._run_in_process(do_gemma, "GEMMA")

    # ════════════════════════════════════════════════════════
    # TRAIN TAB (Crucible)
    # ════════════════════════════════════════════════════════

    def _build_train_tab(self, parent):
        """Train: Crucible training engine."""
        f = tk.Frame(parent, bg=self.colors["bg"], padx=16, pady=12)
        f.pack(fill=tk.BOTH, expand=True)

        tk.Label(f, text="The Crucible — NTMRE Training Engine",
                 font=("Consolas", 11), bg=self.colors["bg"],
                 fg=self.colors["text"]).pack(anchor=tk.W)
        tk.Label(f, text="Pure CUDA training. Forward + Backward + AdamW. No PyTorch.",
                 font=("Consolas", 9), bg=self.colors["bg"],
                 fg=self.colors["text_dim"]).pack(anchor=tk.W, pady=(0, 12))

        # Config file
        cfg_row = tk.Frame(f, bg=self.colors["bg"])
        cfg_row.pack(fill=tk.X, pady=4)
        tk.Label(cfg_row, text="Config:", font=("Consolas", 10),
                 bg=self.colors["bg"], fg=self.colors["text"], width=8).pack(side=tk.LEFT)
        self._train_config = tk.Entry(cfg_row, bg=self.colors["bg_input"],
                                       fg=self.colors["text_bright"],
                                       font=("Consolas", 10), relief=tk.FLAT)
        self._train_config.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Button(cfg_row, text="Browse", command=self._browse_train_config,
                  bg=self.colors["bg_card"], fg=self.colors["matrix_green"],
                  font=("Consolas", 9), relief=tk.FLAT).pack(side=tk.RIGHT)

        # Mode selection
        mode_row = tk.Frame(f, bg=self.colors["bg"])
        mode_row.pack(fill=tk.X, pady=8)
        tk.Label(mode_row, text="Mode:", font=("Consolas", 10),
                 bg=self.colors["bg"], fg=self.colors["text"], width=8).pack(side=tk.LEFT)
        self._train_mode = tk.StringVar(value="single")
        for mode, label in [("single", "Single GPU"), ("pipeline", "Pipeline Parallel"), ("resume", "Resume")]:
            tk.Radiobutton(mode_row, text=label, variable=self._train_mode, value=mode,
                           bg=self.colors["bg"], fg=self.colors["text"],
                           selectcolor=self.colors["bg_card"],
                           font=("Consolas", 10)).pack(side=tk.LEFT, padx=8)

        # Train button
        tk.Button(f, text="🔥 START TRAINING", command=self._run_train,
                  bg=self.colors["blood_red"], fg="#ffffff",
                  font=("Consolas", 12, "bold"), relief=tk.FLAT,
                  cursor="hand2", height=2).pack(fill=tk.X, pady=(16, 4))

        # Stop button
        tk.Button(f, text="⏹ STOP", command=self._stop_process,
                  bg=self.colors["bg_card"], fg=self.colors["text"],
                  font=("Consolas", 10), relief=tk.FLAT,
                  cursor="hand2").pack(fill=tk.X, pady=2)

    def _browse_train_config(self):
        path = filedialog.askopenfilename(
            title="Select training config",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self._train_config.delete(0, tk.END)
            self._train_config.insert(0, path)

    def _run_train(self):
        crucible = ENGINE_DIR / "crucible_v4.exe"
        if not crucible.exists():
            crucible = ENGINE_DIR / "crucible.exe"
        if not crucible.exists():
            self._log_msg("[ERROR] crucible.exe not found! Compile first:\n"
                         f"  Run: {ENGINE_DIR / 'BUILD_CRUCIBLE_V4.bat'}\n")
            return
        cmd = [str(crucible)]
        config = self._train_config.get().strip()
        if config:
            cmd += ["--config", config]
        mode = self._train_mode.get()
        if mode == "pipeline":
            cmd.append("--pipeline")
        elif mode == "resume":
            cmd.append("--resume")
        self._run_command(cmd, "CRUCIBLE")

    # ════════════════════════════════════════════════════════
    # COLLAPSE TAB (.ntmre compression Compression)
    # ════════════════════════════════════════════════════════

    def _build_collapse_tab(self, parent):
        """Collapse: .ntmre compression compression."""
        f = tk.Frame(parent, bg=self.colors["bg"], padx=16, pady=12)
        f.pack(fill=tk.BOTH, expand=True)

        tk.Label(f, text="Collapse — .ntmre compression Compression",
                 font=("Consolas", 11), bg=self.colors["bg"],
                 fg=self.colors["text"]).pack(anchor=tk.W)
        tk.Label(f, text="25-35× compression at 99.9% energy retention. GPU-accelerated.",
                 font=("Consolas", 9), bg=self.colors["bg"],
                 fg=self.colors["text_dim"]).pack(anchor=tk.W, pady=(0, 12))

        # Input .ntmre or checkpoint
        in_row = tk.Frame(f, bg=self.colors["bg"])
        in_row.pack(fill=tk.X, pady=4)
        tk.Label(in_row, text="Input:", font=("Consolas", 10),
                 bg=self.colors["bg"], fg=self.colors["text"], width=8).pack(side=tk.LEFT)
        self._collapse_input = tk.Entry(in_row, bg=self.colors["bg_input"],
                                         fg=self.colors["text_bright"],
                                         font=("Consolas", 10), relief=tk.FLAT)
        self._collapse_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Button(in_row, text="Browse", command=self._browse_collapse_input,
                  bg=self.colors["bg_card"], fg=self.colors["matrix_green"],
                  font=("Consolas", 9), relief=tk.FLAT).pack(side=tk.RIGHT)

        # Energy slider
        energy_row = tk.Frame(f, bg=self.colors["bg"])
        energy_row.pack(fill=tk.X, pady=8)
        tk.Label(energy_row, text="Energy:", font=("Consolas", 10),
                 bg=self.colors["bg"], fg=self.colors["text"], width=8).pack(side=tk.LEFT)
        self._energy_label = tk.Label(energy_row, text="99.9%",
                                       font=("Consolas", 10), bg=self.colors["bg"],
                                       fg=self.colors["matrix_green"])
        self._energy_label.pack(side=tk.RIGHT)
        self._energy_slider = tk.Scale(energy_row, from_=95.0, to=99.99, resolution=0.1,
                                        orient=tk.HORIZONTAL, bg=self.colors["bg"],
                                        fg=self.colors["text"], highlightthickness=0,
                                        troughcolor=self.colors["bg_input"],
                                        command=lambda v: self._energy_label.config(text=f"{float(v):.1f}%"))
        self._energy_slider.set(99.9)
        self._energy_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        # Compress button
        tk.Button(f, text="💀 COLLAPSE", command=self._run_collapse,
                  bg=self.colors["blood_red"], fg="#ffffff",
                  font=("Consolas", 12, "bold"), relief=tk.FLAT,
                  cursor="hand2", height=2).pack(fill=tk.X, pady=(16, 4))

    def _browse_collapse_input(self):
        path = filedialog.askopenfilename(
            title="Select model/checkpoint",
            filetypes=[("NTMRE/Checkpoint", "*.ntmre *.bin"), ("All", "*.*")])
        if path:
            self._collapse_input.delete(0, tk.END)
            self._collapse_input.insert(0, path)

    def _run_collapse(self):
        input_path = self._collapse_input.get().strip()
        if not input_path:
            messagebox.showwarning("Collapse", "Select an input file first.")
            return
        energy = self._energy_slider.get() / 100.0

        # Try GPU version first (collapse.exe)
        collapse_exe = ENGINE_DIR / "collapse.exe"
        if collapse_exe.exists() and input_path.endswith('.ntmre'):
            out = str(Path(input_path).with_stem(Path(input_path).stem + "_svdq8"))
            cmd = [str(collapse_exe), input_path, out, "--energy", str(energy)]
            self._run_command(cmd, "COLLAPSE (GPU)")
        else:
            # Python fallback — try in-process via convert_model (re-quantize with .ntmre compression)
            def do_collapse():
                try:
                    from nightmare_convert import convert_model
                except ImportError:
                    try:
                        from engine.nightmare_convert import convert_model
                    except ImportError:
                        self.root.after(0, self._log_msg,
                            "[ERROR] Collapse requires collapse.exe (GPU) or nightmare_convert module.\n"
                            "   Compile collapse.exe from collapse.cu, or run from source.\n")
                        return
                out = str(Path(input_path).with_stem(Path(input_path).stem + "_svdq8"))
                convert_model(input_path, out)
            self._run_in_process(do_collapse, "COLLAPSE (Python)")

    # ════════════════════════════════════════════════════════
    # EXPORT TAB
    # ════════════════════════════════════════════════════════

    def _build_export_tab(self, parent):
        """Export: Package as .ntmre"""
        f = tk.Frame(parent, bg=self.colors["bg"], padx=16, pady=12)
        f.pack(fill=tk.BOTH, expand=True)

        tk.Label(f, text="NTMRE Export — Package model for deployment",
                 font=("Consolas", 11), bg=self.colors["bg"],
                 fg=self.colors["text"]).pack(anchor=tk.W)
        tk.Label(f, text="Checkpoint + tokenizer + config → single .ntmre file",
                 font=("Consolas", 9), bg=self.colors["bg"],
                 fg=self.colors["text_dim"]).pack(anchor=tk.W, pady=(0, 12))

        # Checkpoint
        ckpt_row = tk.Frame(f, bg=self.colors["bg"])
        ckpt_row.pack(fill=tk.X, pady=4)
        tk.Label(ckpt_row, text="Checkpoint:", font=("Consolas", 10),
                 bg=self.colors["bg"], fg=self.colors["text"], width=10).pack(side=tk.LEFT)
        self._export_ckpt = tk.Entry(ckpt_row, bg=self.colors["bg_input"],
                                      fg=self.colors["text_bright"],
                                      font=("Consolas", 10), relief=tk.FLAT)
        self._export_ckpt.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Button(ckpt_row, text="Browse",
                  command=lambda: self._browse_to_entry(self._export_ckpt, "*.bin *.ntmre"),
                  bg=self.colors["bg_card"], fg=self.colors["matrix_green"],
                  font=("Consolas", 9), relief=tk.FLAT).pack(side=tk.RIGHT)

        # Config
        cfg_row = tk.Frame(f, bg=self.colors["bg"])
        cfg_row.pack(fill=tk.X, pady=4)
        tk.Label(cfg_row, text="Config:", font=("Consolas", 10),
                 bg=self.colors["bg"], fg=self.colors["text"], width=10).pack(side=tk.LEFT)
        self._export_config = tk.Entry(cfg_row, bg=self.colors["bg_input"],
                                        fg=self.colors["text_bright"],
                                        font=("Consolas", 10), relief=tk.FLAT)
        self._export_config.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Button(cfg_row, text="Browse",
                  command=lambda: self._browse_to_entry(self._export_config, "*.json"),
                  bg=self.colors["bg_card"], fg=self.colors["matrix_green"],
                  font=("Consolas", 9), relief=tk.FLAT).pack(side=tk.RIGHT)

        # Export button
        tk.Button(f, text="📤 EXPORT .NTMRE", command=self._run_export,
                  bg=self.colors["blood_red"], fg="#ffffff",
                  font=("Consolas", 12, "bold"), relief=tk.FLAT,
                  cursor="hand2", height=2).pack(fill=tk.X, pady=(16, 4))

    def _run_export(self):
        ckpt = self._export_ckpt.get().strip()
        config = self._export_config.get().strip()
        if not ckpt:
            messagebox.showwarning("Export", "Select a checkpoint first.")
            return
        # In-process export — works in the hardened exe
        def do_export():
            try:
                from engine.nightmare_compress import compress_checkpoint
            except ImportError:
                try:
                    from nightmare_compress import compress_checkpoint
                except ImportError:
                    self.root.after(0, self._log_msg,
                        "[ERROR] nightmare_compress module not found.\n")
                    return
            compress_checkpoint(ckpt, config or None)
        self._run_in_process(do_export, "EXPORT")

    # ════════════════════════════════════════════════════════
    # SHORTCUT LAUNCHERS
    # ════════════════════════════════════════════════════════

    def show_convert(self):
        self.show_full()

    def show_bonsai(self):
        self.show_full()

    def show_gemma(self):
        self.show_full()

    def show_train(self):
        self.show_full()

    def show_collapse(self):
        self.show_full()

    def show_decompress(self):
        self.show_full()

    def show_export(self):
        self.show_full()

    # ════════════════════════════════════════════════════════
    # UTILITIES
    # ════════════════════════════════════════════════════════

    def _browse_to_entry(self, entry_widget, filetypes_str):
        """Generic file browser that fills an entry widget."""
        ftypes = [(filetypes_str, filetypes_str), ("All", "*.*")]
        path = filedialog.askopenfilename(filetypes=ftypes)
        if path:
            entry_widget.delete(0, tk.END)
            entry_widget.insert(0, path)

    def _log_msg(self, text):
        """Append text to the log output."""
        if hasattr(self, '_log') and self._log.winfo_exists():
            self._log.insert(tk.END, text)
            self._log.see(tk.END)

    def _run_command(self, cmd, label):
        """Run a command in a background thread, streaming output to log."""
        self._log_msg(f"\n[{label}] Running: {' '.join(cmd)}\n{'─'*50}\n")

        def _thread():
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1, cwd=str(ENGINE_DIR))
                self.process = proc
                for line in proc.stdout:
                    self.root.after(0, self._log_msg, line)
                proc.wait()
                self.root.after(0, self._log_msg,
                               f"\n{'─'*50}\n[{label}] Done (exit code: {proc.returncode})\n")
            except Exception as e:
                self.root.after(0, self._log_msg, f"\n[ERROR] {e}\n")
            self.process = None

        threading.Thread(target=_thread, daemon=True).start()

    def _run_in_process(self, func, label):
        """Run a Python function in a background thread, capturing stdout to log."""
        import io, contextlib
        _sep = chr(0x2500) * 50
        self._log_msg("\n[" + label + "] Running in-process...\n" + _sep + "\n")
        self._in_process_running = True

        def _thread():
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    func()
                output = buf.getvalue()
                if output:
                    for line in output.splitlines(True):
                        self.root.after(0, self._log_msg, line)
                self.root.after(0, self._log_msg,
                               "\n" + _sep + "\n[" + label + "] Done (success)\n")
            except Exception as e:
                import traceback
                err = traceback.format_exc()
                output = buf.getvalue()
                if output:
                    for line in output.splitlines(True):
                        self.root.after(0, self._log_msg, line)
                self.root.after(0, self._log_msg, "\n[ERROR] " + str(e) + "\n" + err + "\n")
            self._in_process_running = False

        threading.Thread(target=_thread, daemon=True).start()

    def _stop_process(self):
        """Kill the running process."""
        if self.process:
            self.process.terminate()
            self._log_msg("\n[STOPPED] Process terminated.\n")
