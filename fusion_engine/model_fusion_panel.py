"""
model_fusion_panel.py — Nightmare Suite Model Fusion GUI Panel
================================================================

Full Nightmare Suite skin:
- Black background, matrix green accents, blood-red highlights.
- Consolas fonts throughout.
- Subtle matrix rain header strip.

Features:
- 4 model slots: Specialist 1, Specialist 2, Specialist 3, CORE (always last).
- Fusion modes: Static Subspace / MoE Router / Hybrid /
  Digital Suspension (core-governed floaters, refuse on veto) / Trait Graft.
- .ntmre compression alignment rank control + Q8 quantization toggle.
- Per-model priority sliders.
- Optional Sub-Token Rule Infusion.
- Real-time log and progress bar.

Integration:
- Standalone: python fusion_engine/model_fusion_panel.py
- Embedded: pass a Toplevel as master (used by nightmare_suite_unified admin panel).

Nightmare Designs — Levi Gates
"""

import os
import sys
import time
import random
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional, List


# Ensure we can find project root (nightmare_suite_v10) relative to this file.
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# === NIGHTMARE SUITE THEME ===
COLORS = {
    "bg": "#000000",
    "bg_card": "#0a0a0a",
    "bg_input": "#111111",
    "border": "#1a1a1a",
    "text": "#E0E0E0",
    "text_dim": "#666666",
    "matrix_green": "#00FF41",
    "matrix_dim": "#003b00",
    "blood_red": "#8b0000",
    "blood_bright": "#ff0000",
    "text_bright": "#FFFFFF",
    "gold": "#FFD700",
    "accent": "#ff0000",
}

MATRIX_CHARS = (
    "ア イ ウ エ オ カ キ ク ケ コ サ シ ス セ ソ タ チ ツ テ ト "
    "ナ ニ ヌ ネ ノ ハ ヒ フ ヘ ホ マ ミ ム メ モ ヤ ユ ヨ "
    "ラ リ ル レ ロ ワ ヲ ン 0 1 2 3 4 5 6 7 8 9 "
    "A B C D E F G H I J K L M N O P Q R S T U V W X Y Z "
    "@ # $ % & * + = < > / \\ | { } [ ] ~ ^ ` !"
).split()


class ModelFusionPanel:
    def __init__(self, master=None):
        """
        If master is None, creates its own top-level window.
        Otherwise, embeds into provided Tkinter container (usually a Toplevel).
        """
        self.master = master or tk.Tk()
        if not master:
            self.master.title("NIGHTMARE DESIGNS — Model Fusion")
            self.master.geometry("1200x950")

        # Lazy-import core engine and loader when needed.
        from fusion_engine.model_fusion_core import (
            FUSION_MODE,
            RUNTIME_COMPOSE,
            COMPOSE_EXPERIMENTAL,
            FISSION_MODE,
            fission_clone_mask,
            fission_rank_split,
            build_compose_plan,
            export_compose_plan,
            create_fusion_engine,
            export_fusion_config,
            export_fused_to_ntmre
        )
        try:
            from fusion_engine.rule_infusion import fuse_with_rule_infusion
        except Exception:
            fuse_with_rule_infusion = None  # sealed: rule_infusion not shipped
        from fusion_engine.universal_fusion_loader import load_model_for_fusion

        self.FUSION_MODE = FUSION_MODE
        self.RUNTIME_COMPOSE = RUNTIME_COMPOSE
        self.COMPOSE_EXPERIMENTAL = COMPOSE_EXPERIMENTAL
        self.FISSION_MODE = FISSION_MODE
        self.fission_clone_mask_fn = fission_clone_mask
        self.fission_rank_split_fn = fission_rank_split
        self.build_compose_plan_fn = build_compose_plan
        self.export_compose_plan_fn = export_compose_plan
        self.create_fusion_engine_fn = create_fusion_engine
        self.export_fusion_config_fn = export_fusion_config
        self.export_fused_to_ntmre_fn = export_fused_to_ntmre
        self.fuse_with_rule_infusion_fn = fuse_with_rule_infusion
        self.load_gguf_for_fusion_fn = load_model_for_fusion

        # Internal state
        self.model_paths: List[Optional[str]] = [None, None, None, None]
        self.fusion_engine = None
        self.last_fused_model = None
        self.busy = False

        self._build_ui()

    def _build_ui(self):
        root = self.master
        root.configure(bg=COLORS["bg"])

        # Main container
        main = tk.Frame(root, bg=COLORS["bg"])
        main.pack(fill="both", expand=True, padx=10, pady=10)

        # === MATRIX RAIN HEADER ===
        header_h = 80
        self.header_canvas = tk.Canvas(
            main, height=header_h, bg=COLORS["bg"],
            highlightthickness=1, highlightbackground=COLORS["matrix_green"]
        )
        self.header_canvas.pack(fill="x", pady=(0, 10))

        self._matrix_drops = [random.randint(-20, -1) for _ in range(60)]
        self._matrix_speeds = [random.uniform(0.5, 1.5) for _ in range(60)]
        self._matrix_items = []
        self._animate_matrix()

        # Title overlay
        self.header_canvas.create_text(
            600, 40,
            text="🧬 MODEL FUSION SYSTEM",
            font=("Impact", 24, "bold"),
            fill=COLORS["blood_bright"],
            anchor="center"
        )
        self.header_canvas.create_text(
            600, 62,
            text="Brain-Switch Router  •  .ntmre compression+Q8 Alignment  •  Sub-Token Rule Infusion",
            font=("Consolas", 9),
            fill=COLORS["matrix_green"],
            anchor="center"
        )

        # === LEFT COLUMN: Controls ===
        left = tk.Frame(main, bg=COLORS["bg"], width=580)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))

        # === RIGHT COLUMN: Log + Progress ===
        right = tk.Frame(main, bg=COLORS["bg"])
        right.pack(side="right", fill="both", expand=True, padx=(10, 0))

        self._build_model_slots(left)
        self._build_fusion_settings(left)
        self._build_action_buttons(left)
        self._build_compose_section(left)
        self._build_fission_section(left)
        self._build_right_panel(right)

    # ---------- MATRIX RAIN ANIMATION ----------

    def _animate_matrix(self):
        """Animate subtle matrix rain in header canvas."""
        if not self.header_canvas.winfo_exists():
            return

        # Clean old items periodically
        if len(self._matrix_items) > 80:
            for item_id in self._matrix_items[:20]:
                try:
                    self.header_canvas.delete(item_id)
                except Exception:
                    pass
            self._matrix_items = self._matrix_items[20:]

        w = int(self.header_canvas.winfo_width())
        if w < 100:
            w = 1200
        cols = 60
        col_w = w // cols

        for i in range(cols):
            self._matrix_drops[i] += self._matrix_speeds[i]
            y = int(self._matrix_drops[i] * 14)
            if 0 < y < 80:
                char = random.choice(MATRIX_CHARS)
                item = self.header_canvas.create_text(
                    i * col_w + col_w // 2, y,
                    text=char, font=("Consolas", 8),
                    fill=COLORS["matrix_dim"], anchor="center"
                )
                self._matrix_items.append(item)
            if y > 90:
                self._matrix_drops[i] = random.randint(-20, -1)
                self._matrix_speeds[i] = random.uniform(0.5, 1.5)

        self.master.after(80, self._animate_matrix)

    # ---------- STYLED WIDGET HELPERS ----------

    def _make_label(self, parent, text, fg=None, font_size=9, bold=False):
        return tk.Label(
            parent, text=text,
            fg=fg or COLORS["text"],
            bg=COLORS["bg"],
            font=("Consolas", font_size, "bold" if bold else "normal")
        )

    def _make_button(self, parent, text, command, bg=None, fg=None):
        return tk.Button(
            parent, text=text,
            font=("Consolas", 12, "bold"),
            bg=bg or COLORS["bg_card"],
            fg=fg or COLORS["matrix_green"],
            activebackground=COLORS["blood_bright"],
            activeforeground="#FFFFFF",
            relief=tk.RAISED,
            bd=3,
            cursor="hand2",
            command=command
        )

    def _make_entry(self, parent, textvariable, width=30, show=None):
        return tk.Entry(
            parent,
            textvariable=textvariable,
            width=width,
            show=show or "",
            font=("Consolas", 9),
            bg=COLORS["bg_input"],
            fg=COLORS["text"],
            insertbackground=COLORS["matrix_green"],
            relief=tk.FLAT,
            highlightthickness=1,
            highlightcolor=COLORS["matrix_green"],
            highlightbackground=COLORS["border"]
        )

    def _make_frame(self, parent, title=None):
        frame = tk.LabelFrame(
            parent,
            text=title or "",
            font=("Consolas", 10, "bold"),
            bg=COLORS["bg"],
            fg=COLORS["gold"],
            relief=tk.RIDGE,
            bd=2,
            highlightbackground=COLORS["border"],
            highlightthickness=1
        )
        return frame

    # ---------- MODEL SLOTS ----------

    def _build_model_slots(self, parent):
        frame = self._make_frame(parent, " Model Slots (2, 3, or 4 Models) ")
        frame.pack(fill="x", pady=(0, 10), ipady=4)

        labels = [
            ("Specialist 1", COLORS["text"]),
            ("Specialist 2 (Optional)", COLORS["text"]),
            ("Specialist 3 (Optional)", COLORS["text"]),
            ("CORE Model (Base Identity)", COLORS["blood_bright"]),
        ]

        self.path_vars: List[tk.StringVar] = []
        self.priority_vars: List[tk.DoubleVar] = []

        for i, (lbl, color) in enumerate(labels):
            row = tk.Frame(frame, bg=COLORS["bg"])
            row.pack(fill="x", padx=8, pady=4)

            self._make_label(row, lbl + ":", fg=color, font_size=9).pack(side="left")

            v = tk.StringVar(value=self.model_paths[i] or "")
            self.path_vars.append(v)

            self._make_entry(row, v, width=24).pack(side="left", padx=(6, 4), fill="x", expand=True)

            browse_btn = self._make_button(row, "Browse", lambda idx=i: self._browse_model(idx))
            browse_btn.pack(side="right", padx=(2, 2))

            clear_btn = tk.Button(
                row, text="✕", font=("Consolas", 9, "bold"),
                bg=COLORS["bg_card"], fg=COLORS["blood_bright"],
                activebackground=COLORS["blood_red"], activeforeground="#FFFFFF",
                relief=tk.FLAT, cursor="hand2", bd=1,
                command=lambda idx=i: self._clear_model(idx)
            )
            clear_btn.pack(side="right", padx=(2, 2))

            # Priority slider
            pvar = tk.DoubleVar(value=1.0 if i == 3 else 0.9)
            self.priority_vars.append(pvar)

            scale = tk.Scale(
                row,
                from_=0.0, to=1.0,
                variable=pvar,
                orient="horizontal",
                length=50,
                resolution=0.05,
                showvalue=0,
                sliderlength=10,
                bg=COLORS["bg"],
                fg=COLORS["matrix_green"],
                troughcolor=COLORS["bg_card"],
                highlightthickness=0,
                activebackground=COLORS["blood_red"]
            )
            scale.pack(side="right", padx=(2, 0))

            # Separator so slots are visually distinct
            if i < len(labels) - 1:
                ttk.Separator(frame, orient="horizontal").pack(fill="x", padx=8, pady=2)

        # Hint footer
        hint_lbl = tk.Label(
            frame,
            text="💡 Choose 2, 3, or 4 models. The last filled slot is used as the CORE base model.",
            font=("Consolas", 8),
            bg=COLORS["bg"],
            fg=COLORS["text_dim"]
        )
        hint_lbl.pack(fill="x", padx=8, pady=(4, 2))

    def _clear_model(self, index: int):
        self.model_paths[index] = None
        if index < len(self.path_vars):
            self.path_vars[index].set("")

    def _browse_model(self, index: int):
        try:
            path = filedialog.askopenfilename(
                parent=self.master,
                title=f"Select Model for Slot {index + 1}",
                filetypes=[("Model files", "*.gguf *.bin *.safetensors *.ntmre *.ntmr"), ("All files", "*.*")]
            )
            if path:
                self.model_paths[index] = path
                self.path_vars[index].set(path)
        except Exception as e:
            messagebox.showerror("Browse Error", f"Could not open file browser:\n{e}")

    # ---------- FUSION SETTINGS ----------

    def _build_fusion_settings(self, parent):
        frame = self._make_frame(parent, " Fusion Settings ")
        frame.pack(fill="x", pady=(0, 10), ipady=4)

        # Mode
        mode_row = tk.Frame(frame, bg=COLORS["bg"])
        mode_row.pack(fill="x", padx=8, pady=4)
        self._make_label(mode_row, "Fusion Mode:").pack(side="left")

        self.mode_var = tk.StringVar(value=self.FUSION_MODE["MOE_ROUTER"])
        mode_menu = tk.OptionMenu(
            mode_row, self.mode_var,
            self.FUSION_MODE["STATIC_SUBSPACE"],
            self.FUSION_MODE["MOE_ROUTER"],
            self.FUSION_MODE["HYBRID"],
            self.FUSION_MODE["DIGITAL_SUSPENSION"],
            self.FUSION_MODE["TRAIT_GRAFT"]
        )
        mode_menu.config(
            bg=COLORS["bg_card"],
            fg=COLORS["matrix_green"],
            activebackground=COLORS["blood_red"],
            activeforeground=COLORS["text_bright"],
            font=("Consolas", 9),
            relief=tk.FLAT,
            highlightthickness=0
        )
        mode_menu["menu"].config(
            bg=COLORS["bg_card"],
            fg=COLORS["matrix_green"],
            activebackground=COLORS["blood_red"],
            activeforeground=COLORS["text_bright"],
            font=("Consolas", 9)
        )
        mode_menu.pack(side="left", padx=(8, 0))

        # Rank
        rank_row = tk.Frame(frame, bg=COLORS["bg"])
        rank_row.pack(fill="x", padx=8, pady=4)
        self._make_label(rank_row, ".ntmre compression Target Rank:").pack(side="left")

        self.rank_var = tk.IntVar(value=1024)
        rank_spin = tk.Spinbox(
            rank_row,
            from_=64, to=8192,
            textvariable=self.rank_var,
            width=8,
            font=("Consolas", 9),
            bg=COLORS["bg_input"],
            fg=COLORS["text"],
            relief=tk.FLAT,
            buttonbackground=COLORS["bg_card"]
        )
        rank_spin.pack(side="left", padx=(8, 8))

        # Q8 toggle
        self.q8_var = tk.BooleanVar(value=True)
        q8_chk = tk.Checkbutton(
            rank_row,
            text="Use Q8 Quantization (ntmre)",
            variable=self.q8_var,
            bg=COLORS["bg"],
            fg=COLORS["matrix_green"],
            activebackground=COLORS["bg"],
            activeforeground=COLORS["matrix_green"],
            selectcolor=COLORS["bg_card"],
            font=("Consolas", 9)
        )
        q8_chk.pack(side="left", padx=(8, 0))

        # GPU acceleration toggle
        from fusion_engine.model_fusion_core import gpu_is_available
        gpu_available = gpu_is_available()
        self.use_gpu_var = tk.BooleanVar(value=gpu_available)
        gpu_chk = tk.Checkbutton(
            rank_row,
            text="Use GPU Acceleration (CUDA)" + ("" if gpu_available else " [no CUDA GPU detected]"),
            variable=self.use_gpu_var,
            state=("normal" if gpu_available else "disabled"),
            bg=COLORS["bg"],
            fg=COLORS["matrix_green"] if gpu_available else COLORS["text_dim"],
            activebackground=COLORS["bg"],
            activeforeground=COLORS["matrix_green"],
            selectcolor=COLORS["bg_card"],
            font=("Consolas", 9)
        )
        gpu_chk.pack(side="left", padx=(8, 0))

        # Rule infusion toggle
        rule_row = tk.Frame(frame, bg=COLORS["bg"])
        rule_row.pack(fill="x", padx=8, pady=(6, 4))

        self.rule_enable_var = tk.BooleanVar(value=False)
        rule_chk = tk.Checkbutton(
            rule_row,
            text="Enable Sub-Token Rule Infusion",
            variable=self.rule_enable_var,
            bg=COLORS["bg"],
            fg=COLORS["gold"],
            activebackground=COLORS["bg"],
            activeforeground=COLORS["gold"],
            selectcolor=COLORS["bg_card"],
            font=("Consolas", 9)
        )
        rule_chk.pack(side="left")

        # Rules and secret key
        rules_frame = self._make_frame(frame, " Core Rules & Secret Key (Optional) ")
        rules_frame.pack(fill="x", padx=8, pady=(4, 8))

        self._make_label(rules_frame, "Rules Text (plain English):", font_size=8).pack(anchor="w", padx=6, pady=(4, 0))

        self.rules_text = tk.Text(
            rules_frame,
            height=3,
            font=("Consolas", 8),
            bg=COLORS["bg_input"],
            fg=COLORS["text"],
            insertbackground=COLORS["matrix_green"],
            wrap="word",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightcolor=COLORS["matrix_green"],
            highlightbackground=COLORS["border"]
        )
        self.rules_text.pack(fill="x", padx=6, pady=(2, 4))

        self._make_label(rules_frame, "Secret Key (only your AI uses this):", font_size=8).pack(anchor="w", padx=6, pady=(4, 0))

        self.secret_key_var = tk.StringVar(value="")
        self._make_entry(rules_frame, self.secret_key_var, show="*").pack(fill="x", padx=6, pady=(2, 6))

    # ---------- ACTION BUTTONS ----------

    def _build_action_buttons(self, parent):
        # Outer frame with visible border so the button area cannot be missed.
        outer = tk.LabelFrame(
            parent,
            text=" ▶ START FUSION ",
            font=("Consolas", 11, "bold"),
            bg=COLORS["bg"],
            fg=COLORS["blood_bright"],
            relief=tk.RIDGE,
            bd=2,
            highlightbackground=COLORS["blood_red"],
            highlightthickness=2
        )
        outer.pack(fill="x", pady=(0, 10), ipady=6)

        frame = tk.Frame(outer, bg=COLORS["bg"])
        frame.pack(fill="x", padx=8, pady=4)

        self.run_btn = tk.Button(
            frame,
            text="[ START FUSION ]",
            font=("Impact", 14, "bold"),
            bg=COLORS["blood_red"],
            fg="#FFFFFF",
            activebackground=COLORS["blood_bright"],
            activeforeground="#FFFFFF",
            relief=tk.RAISED,
            bd=4,
            cursor="hand2",
            command=self._on_run_fusion
        )
        self.run_btn.pack(side="left", padx=(0, 10), ipadx=10, ipady=6)

        self.export_ntmre_btn = self._make_button(
            frame,
            "Export .ntmre",
            self._on_export_ntmre,
            fg=COLORS["gold"]
        )
        self.export_ntmre_btn.pack(side="left", padx=(0, 6))

        self.export_cfg_btn = self._make_button(
            frame,
            "Export Config",
            self._on_export_config
        )
        self.export_cfg_btn.pack(side="left")

    # ---------- RUNTIME COMPOSE (no merging) ----------

    def _build_compose_section(self, parent):
        outer = self._make_frame(parent, " Runtime Compose (No Merging) ")
        outer.pack(fill="x", pady=(0, 10), ipady=4)

        row = tk.Frame(outer, bg=COLORS["bg"])
        row.pack(fill="x", padx=8, pady=4)
        self._make_label(row, "Compose Mode:").pack(side="left")

        self.compose_var = tk.StringVar(value=self.RUNTIME_COMPOSE["CASCADE"])
        self.compose_menu = tk.OptionMenu(
            row, self.compose_var,
            self.RUNTIME_COMPOSE["CASCADE"]
        )
        self.compose_menu.config(
            bg=COLORS["bg_card"], fg=COLORS["matrix_green"],
            activebackground=COLORS["blood_red"],
            activeforeground=COLORS["text_bright"],
            font=("Consolas", 9), relief=tk.FLAT, highlightthickness=0
        )
        self.compose_menu["menu"].config(
            bg=COLORS["bg_card"], fg=COLORS["matrix_green"],
            activebackground=COLORS["blood_red"],
            activeforeground=COLORS["text_bright"], font=("Consolas", 9)
        )
        self.compose_menu.pack(side="left", padx=(8, 0))

        self.show_exp_var = tk.BooleanVar(value=False)
        exp_chk = tk.Checkbutton(
            row, text="Show experimental",
            variable=self.show_exp_var, command=self._toggle_experimental,
            bg=COLORS["bg"], fg=COLORS["gold"],
            activebackground=COLORS["bg"], activeforeground=COLORS["gold"],
            selectcolor=COLORS["bg_card"], font=("Consolas", 9)
        )
        exp_chk.pack(side="left", padx=(8, 0))

        exp_btn = self._make_button(outer, "Export Compose Plan",
                                    self._on_export_compose,
                                    fg=COLORS["gold"])
        exp_btn.pack(anchor="w", padx=8, pady=(0, 4))

        hint = tk.Label(
            outer,
            text="Cascade runs one model at a time (fits RAM cap). "
                 "Experimental modes need multi-resident runtimes.",
            font=("Consolas", 8), bg=COLORS["bg"], fg=COLORS["text_dim"]
        )
        hint.pack(fill="x", padx=8, pady=(0, 2))

    def _toggle_experimental(self):
        menu = self.compose_menu["menu"]
        menu.delete(0, "end")
        menu.add_command(label=self.RUNTIME_COMPOSE["CASCADE"],
                         command=lambda v=self.RUNTIME_COMPOSE["CASCADE"]:
                         self.compose_var.set(v))
        if self.show_exp_var.get():
            for m in self.COMPOSE_EXPERIMENTAL:
                menu.add_command(label=m,
                                 command=lambda v=m: self.compose_var.set(v))
        else:
            if self.compose_var.get() in self.COMPOSE_EXPERIMENTAL:
                self.compose_var.set(self.RUNTIME_COMPOSE["CASCADE"])
        if hasattr(self, "fission_frame"):
            if self.show_exp_var.get():
                self.fission_frame.pack(fill="x", pady=(0, 10), ipady=4)
            else:
                self.fission_frame.pack_forget()

    def _on_export_compose(self):
        selected = [v.get().strip() for v in self.path_vars]
        selected = [p for p in selected if p]
        if len(selected) < 2:
            messagebox.showwarning(
                "Compose", "Select at least 2 models in the slots above.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            initialfile="compose_%s.json" % self.compose_var.get(),
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            plan = self.build_compose_plan_fn(
                model_paths=selected, mode=str(self.compose_var.get()))
            saved = self.export_compose_plan_fn(plan, path)
            self.log("Compose plan (%s) written to: %s"
                     % (plan["mode"], saved))
            messagebox.showinfo("Compose Plan",
                                "Plan exported:\n\n%s" % saved)
        except Exception as e:
            messagebox.showerror("Compose Error", str(e))

    # ---------- FISSION (experimental, no merging) ----------

    def _build_fission_section(self, parent):
        self.fission_frame = self._make_frame(parent, " Fission (Experimental) ")
        self.fission_frame.pack(fill="x", pady=(0, 10), ipady=4)

        row = tk.Frame(self.fission_frame, bg=COLORS["bg"])
        row.pack(fill="x", padx=8, pady=4)
        self._make_label(row, "Fission Mode:").pack(side="left")

        self.fission_var = tk.StringVar(value=self.FISSION_MODE["CLONE_MASK"])
        fission_menu = tk.OptionMenu(
            row, self.fission_var,
            self.FISSION_MODE["CLONE_MASK"],
            self.FISSION_MODE["RANK_SPLIT"]
        )
        fission_menu.config(
            bg=COLORS["bg_card"], fg=COLORS["matrix_green"],
            activebackground=COLORS["blood_red"],
            activeforeground=COLORS["text_bright"],
            font=("Consolas", 9), relief=tk.FLAT, highlightthickness=0
        )
        fission_menu["menu"].config(
            bg=COLORS["bg_card"], fg=COLORS["matrix_green"],
            activebackground=COLORS["blood_red"],
            activeforeground=COLORS["text_bright"], font=("Consolas", 9)
        )
        fission_menu.pack(side="left", padx=(8, 0))

        split_btn = self._make_button(self.fission_frame, "Split First Model",
                                      self._on_fission_split,
                                      fg=COLORS["gold"])
        split_btn.pack(anchor="w", padx=8, pady=(0, 4))

        hint = tk.Label(
            self.fission_frame,
            text="Splits the first selected model into 2 runnable parts "
                 "(clone+mask or rank halves). Parts export as .ntmre.",
            font=("Consolas", 8), bg=COLORS["bg"], fg=COLORS["text_dim"]
        )
        hint.pack(fill="x", padx=8, pady=(0, 2))

        if not self.show_exp_var.get():
            self.fission_frame.pack_forget()

    def _on_fission_split(self):
        selected = [v.get().strip() for v in self.path_vars]
        selected = [p for p in selected if p]
        if not selected:
            messagebox.showwarning(
                "Fission", "Select at least 1 model in the slots above.")
            return
        if not os.path.exists(selected[0]):
            messagebox.showerror("Fission", "File does not exist:\n%s" % selected[0])
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".ntmre",
            initialfile="fissionA.ntmre",
            filetypes=[("NTMRE Model Files", "*.ntmre"), ("All files", "*.*")]
        )
        if not path:
            return
        try:
            self.log("Loading for fission: %s" % os.path.basename(selected[0]))
            mdata = self.load_gguf_for_fusion_fn(path=selected[0])
            mode = str(self.fission_var.get())
            if mode == self.FISSION_MODE["RANK_SPLIT"]:
                self.log("Rank-splitting (.ntmre compression halves)...")
                parts = self.fission_rank_split_fn(mdata)
            else:
                self.log("Clone+mask splitting...")
                parts = self.fission_clone_mask_fn(mdata)
            import os.path as _op
            base, _ = _op.splitext(path)
            out_a = self.export_fused_to_ntmre_fn(parts[0], base + ".ntmre")
            out_b = self.export_fused_to_ntmre_fn(
                parts[1], base.replace(".fissionA", ".fissionB")
                if base.endswith(".fissionA") else base + "B.ntmre")
            self.log("Fission parts written:\n  %s\n  %s" % (out_a, out_b))
            messagebox.showinfo("Fission Complete",
                                "Parts exported:\n\n%s\n%s" % (out_a, out_b))
        except Exception as e:
            self.log("Fission error: %s" % e)
            messagebox.showerror("Fission Error", str(e))

    # ---------- RIGHT PANEL: LOGS + PROGRESS ----------

    def _build_right_panel(self, parent):
        # Progress bar label
        self._make_label(parent, "Fusion Progress", fg=COLORS["gold"], font_size=10, bold=True).pack(anchor="w", pady=(0, 4))

        self.progress = ttk.Progressbar(parent, mode="determinate", length=400)
        self.progress.pack(fill="x", pady=(0, 10))
        # Style the progress bar
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Horizontal.TProgressbar",
            troughcolor=COLORS["bg_card"],
            background=COLORS["matrix_green"],
            bordercolor=COLORS["border"],
            lightcolor=COLORS["matrix_green"],
            darkcolor=COLORS["matrix_green"]
        )

        # Log area (plain Text + Scrollbar, no scrolledtext dependency)
        log_frame = self._make_frame(parent, " Fusion Log ")
        log_frame.pack(fill="both", expand=True)

        log_inner = tk.Frame(log_frame, bg=COLORS["bg"])
        log_inner.pack(fill="both", expand=True, padx=6, pady=6)

        self.log_text = tk.Text(
            master=log_inner,
            height=30,
            font=("Consolas", 8),
            bg=COLORS["bg_input"],
            fg=COLORS["text"],
            insertbackground=COLORS["matrix_green"],
            wrap="word",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightcolor=COLORS["matrix_green"],
            highlightbackground=COLORS["border"]
        )
        self.log_text.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(
            log_inner,
            command=self.log_text.yview,
            bg=COLORS["bg_card"],
            troughcolor=COLORS["bg_input"],
            activebackground=COLORS["blood_red"],
            relief=tk.FLAT
        )
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.master.update()

    # ---------- FUSION RUNNER ----------

    def _on_run_fusion(self):
        if self.busy:
            return

        selected = []
        selected_priorities = []
        for i, v in enumerate(self.path_vars):
            p = v.get().strip()
            if p:
                if not os.path.exists(p):
                    messagebox.showerror("Model Not Found", f"File does not exist:\n{p}")
                    return
                selected.append(p)
                selected_priorities.append(float(self.priority_vars[i].get()))

        if len(selected) < 2:
            messagebox.showwarning(
                "Insufficient Models",
                "At least 2 models must be selected (fuse 2, 3, or 4 models; the last filled model is treated as CORE base)."
            )
            return

        if len(selected) > 4:
            messagebox.showwarning(
                "Too Many Models",
                "Maximum 4 models can be fused at once."
            )
            return

        if len(selected) == 2:
            self.log("Note: 2 models selected (1 Specialist + 1 CORE base).")
        elif len(selected) == 3:
            self.log("Note: 3 models selected (2 Specialists + 1 CORE base).")
        elif len(selected) == 4:
            self.log("Note: 4 models selected (3 Specialists + 1 CORE base).")

        mode = str(self.mode_var.get())
        rank = int(self.rank_var.get())
        use_q8 = bool(self.q8_var.get())
        use_gpu = bool(self.use_gpu_var.get())

        rule_enabled = self.rule_enable_var.get() and self.fuse_with_rule_infusion_fn is not None
        rules_text = self.rules_text.get("1.0", "end-1c").strip() if rule_enabled else ""
        secret_key = self.secret_key_var.get().strip() if rule_enabled else ""

        if rule_enabled and (not rules_text or not secret_key):
            messagebox.showwarning(
                "Rule Infusion Incomplete",
                "To enable Sub-Token Rule Infusion, provide both Rules Text and Secret Key."
            )
            return

        core_priority = selected_priorities[-1]
        spec_priorities = selected_priorities[:-1]

        try:
            engine = self.create_fusion_engine_fn(
                mode=mode,
                core_priority=core_priority,
                specialist_priorities=spec_priorities,
                use_gpu=use_gpu
            )
            self.fusion_engine = engine
            if engine.backend.has_native_svd:
                self.log("Backend: native ntmre .ntmre compression/Q8 loaded.")
            elif engine.backend.use_gpu:
                self.log(f"Backend: GPU-accelerated (CUDA, device={engine.backend.device}).")
            else:
                self.log("Backend: pure-Python .ntmre compression/Q8 fallback (slower). Build ntmre_fusion.dll or enable GPU for speed.")
        except Exception as e:
            messagebox.showerror("Fusion Engine Error", str(e))
            return

        models_data = []
        try:
            for i, p in enumerate(selected):
                self.log(f"Loading model {i+1}/{len(selected)}: {os.path.basename(p)}")
                mdata = self.load_gguf_for_fusion_fn(
                    path=p,
                    include_attn=False
                )
                models_data.append(mdata)
        except Exception as e:
            messagebox.showerror("Model Loading Error", str(e))
            return

        self.busy = True
        self.run_btn.configure(state="disabled")

        max_layers = 0
        for m in models_data:
            lc = len(m.get("layers", []))
            if lc > max_layers:
                max_layers = lc
        self.progress["maximum"] = max(1, max_layers)

        def on_progress(layer_idx, total, msg):
            value = (layer_idx + 1) * 100 // total
            self.progress["value"] = value
            self.log(f"Layer {layer_idx+1}/{total}: {msg}")

        try:
            self.log("Starting fusion pipeline...")
            if rule_enabled:
                self.log("Sub-Token Rule Infusion: ENABLED")
                fused_model = self.fuse_with_rule_infusion_fn(
                    engine,
                    models_data=models_data,
                    target_rank=rank,
                    quantize_factors=use_q8,
                    on_progress=on_progress,
                    enable_rules=True,
                    rules_text=rules_text,
                    secret_key=secret_key
                )
            else:
                fused_model = engine.fuse_models(
                    models_data=models_data,
                    target_rank=rank,
                    quantize_factors=use_q8,
                    on_progress=on_progress
                )
            self.last_fused_model = fused_model
            self.log("Fusion complete. Fused model ID: " + fused_model["id"])
            if rule_enabled:
                self.log("Core rules embedded into model weights.")
        except Exception as e:
            self.log(f"Fusion error: {e}")
            messagebox.showerror("Fusion Error", str(e))
        finally:
            self.busy = False
            self.run_btn.configure(state="normal")

    # ---------- EXPORT NTMRE ----------

    def _on_export_ntmre(self):
        if not self.last_fused_model:
            messagebox.showinfo("Export .ntmre", "No fused model available. Please run [ START FUSION ] first.")
            return

        default_name = f"{self.last_fused_model.get('id', 'fused_model')}.ntmre"
        path = filedialog.asksaveasfilename(
            defaultextension=".ntmre",
            initialfile=default_name,
            filetypes=[("NTMRE Model Files", "*.ntmre"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            self.log(f"Exporting fused model to: {path}...")
            def on_export_progress(idx, total, msg):
                val = (idx + 1) * 100 // max(1, total)
                self.progress["value"] = val
                self.log(f"[{val}%] {msg}")

            backend = getattr(self.fusion_engine, "backend", None)
            saved_path = self.export_fused_to_ntmre_fn(
                self.last_fused_model,
                path,
                backend=backend,
                on_progress=on_export_progress
            )
            self.log(f"✅ Fused model successfully written to: {saved_path}")
            messagebox.showinfo("Export Complete", f"Fused model exported successfully!\n\nPath: {saved_path}")
        except Exception as e:
            self.log(f"❌ Export error: {e}")
            messagebox.showerror("Export Error", f"Failed to export .ntmre:\n\n{e}")

    # ---------- EXPORT CONFIG ----------

    def _on_export_config(self):
        if not self.fusion_engine:
            messagebox.showinfo("Export Config", "No fusion engine configured yet.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            self.export_fusion_config_fn(self.fusion_engine, path)
            self.log(f"Config exported to {path}")
        except Exception as e:
            messagebox.showerror("Export Error", str(e))


def run_standalone():
    """
    Run Model Fusion Panel as a standalone window.
    """
    root = tk.Tk()
    ModelFusionPanel(master=root)
    root.mainloop()


if __name__ == "__main__":
    run_standalone()
