# -*- coding: utf-8 -*-
"""
decompiler_panel.py — GGUF & LoRA Decompiler / Model Comparator

Techniques in this file, verified against the code in this tree:
  1. Model analyze — weight statistics per layer, quant-quality / dtype
     summary (same technique as ModelAnalyzerPanel in model_tools_panels.py).
  2. Model compare — identical / modified / only-in-A / only-in-B, with
     cosine similarity and mean absolute difference for changed tensors.
  3. LoRA extract — truncated-.ntmre compression delta between a base and a tuned model,
     saved as .lora_A / .lora_B .npy files.
  4. Layer extract / delete / split — select tensors by blk.N layer range
     (plus embeddings / output head when asked), renumber kept layers
     from 0, and write a new GGUF with the local GGUFWriter. Output is
     F32 only; tensors that cannot be loaded are skipped and logged,
     never silently invented.

NOT in this file, and NOT claimed:
  - No in-place layer editing of a quantized model, no weight editing UI.
  - No auth gate, no .nightmare_unlock check, no tamper hash, no
    obfuscation, no network code, no master-token / the assistant / Brain code.
    This is a Tier 3 Suite tool, not a master-token mechanism.
  - Layer surgery here means extract / delete / split by layer range,
    as implemented below. It does not rewrite individual weights.
"""

import os
import re
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from pathlib import Path
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

_DEFAULT_COLORS = {
    "bg": "#000000", "bg_panel": "#000000", "bg_card": "#0a0a0a",
    "bg_input": "#0d0d0d", "border": "#1a1a1a", "text": "#cccccc",
    "text_dim": "#666666", "text_bright": "#ffffff",
    "blood_red": "#8b0000", "blood_bright": "#ff0000",
    "matrix_green": "#00ff41", "warning": "#ff6600", "gold": "#ffd700",
    "accent": "#ff0000", "error": "#ff4444",
}


def parse_gguf_metadata(filepath):
    """Parse a GGUF file with the CURRENT gguf_parser API.

    Returns (metadata, tensors, data_offset, metadata) where tensors is a
    list of objects with .name, .shape, .dtype, .offset, .size_bytes.
    The old API (p.header / p.tensors as a list / p.data_offset) no
    longer exists in this tree: GGUFParser(filepath).parse() returns a
    GGUFModel with .tensors as a dict and .tensor_data_offset.  The old
    call silently returned ([], 0) for every model; this version does not.
    """
    try:
        from gguf_parser import GGUFParser
        parser = GGUFParser(filepath)
        model = parser.parse()
        tensors = []
        # Best-effort byte size for quantized types, from dequant_fast.
        block_sizes = {}
        try:
            import dequant_fast
            block_sizes = getattr(dequant_fast, "BLOCK_SIZES", {}) or {}
        except Exception:
            block_sizes = {}
        for info in model.tensors.values():
            n_elem = 1
            for dim in info.shape:
                n_elem *= int(dim)
            if info.dtype == 0:
                size_bytes = n_elem * 4
            elif info.dtype == 1:
                size_bytes = n_elem * 2
            elif info.dtype in block_sizes:
                block_elems, block_bytes = block_sizes[info.dtype]
                import math as _math
                size_bytes = _math.ceil(n_elem / block_elems) * block_bytes
            else:
                size_bytes = 0
            # Simple namespace, not a dataclass, so size_bytes can be set.
            tensors.append(type("TensorMeta", (), {
                "name": info.name, "shape": tuple(info.shape),
                "dtype": int(info.dtype), "offset": int(info.offset),
                "size_bytes": int(size_bytes),
                "n_dims": len(info.shape),
            })())
        return model.metadata, tensors, int(model.tensor_data_offset), model.metadata
    except Exception:
        return {}, [], 0, {}


def _thread_safe_update(widget, func):
    try:
        if widget and hasattr(widget, 'after'):
            widget.after(0, func)
        else:
            func()
    except Exception:
        pass


def _layer_index(name):
    m = re.search(r"blk\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def _is_embedding(name):
    return any(k in name for k in ("token_embd", "embed_tokens"))


def _is_output_head(name):
    return any(k in name for k in ("output", "lm_head")) or (
        "norm" in name and "blk" not in name)


def _load_tensor_f32(filepath, tensor_name):
    """Load one tensor, dequantized to float32, via the local parser."""
    from gguf_parser import GGUFParser
    parser = GGUFParser(filepath)
    model = parser.parse()
    return parser.load_tensor(model, tensor_name)


class DecompilerPanel:
    def __init__(self, llog_widget=None, clog_widget=None):
        self._llog = llog_widget
        self._clog = clog_widget

    def _do_lora_extract(self, _bp, _tp, _rk):
        """Background LoRA extraction via truncated .ntmre compression."""
        try:
            _, _bt, _bdo, _ = parse_gguf_metadata(_bp)
            _, _tt, _tdo, _ = parse_gguf_metadata(_tp)

            _bm = {_t.name: _t for _t in _bt}
            _tm = {_t.name: _t for _t in _tt}

            _cw = []
            for _n in _bm:
                if _n in _tm:
                    _b = _bm[_n]
                    _u = _tm[_n]
                    if len(_b.shape) == 2 and _b.dtype == 0 and _u.dtype == 0:
                        if _b.shape == _u.shape:
                            _cw.append(_n)

            if self._llog:
                _thread_safe_update(self._llog, lambda: self._llog.insert(tk.END, f"  Found {len(_cw)} comparable 2D tensors\n"))

            _od = filedialog.askdirectory(title="Save LoRA output to...")
            if not _od:
                return

            Path(_od).mkdir(parents=True, exist_ok=True)

            for _i, _n in enumerate(_cw):
                _b = _bm[_n]
                _u = _tm[_n]

                with open(_bp, "rb") as _f:
                    _f.seek(_bdo + _b.offset)
                    _ba = np.frombuffer(_f.read(_b.size_bytes),
                                       dtype=np.float32).reshape(_b.shape[::-1])

                with open(_tp, "rb") as _f:
                    _f.seek(_tdo + _u.offset)
                    _ua = np.frombuffer(_f.read(_u.size_bytes),
                                       dtype=np.float32).reshape(_u.shape[::-1])

                _delta = _ua - _ba

                if _delta.shape[0] > _rk and _delta.shape[1] > _rk:
                    _U, _S, _Vt = np.linalg.svd(_delta, full_matrices=False)
                    _A = _U[:, :_rk] * np.sqrt(_S[:_rk])
                    _B = _Vt[:_rk, :] * np.sqrt(_S[:_rk])[:, np.newaxis]

                    _sn = _n.replace(".", "_").replace("/", "_")
                    np.save(os.path.join(_od, f"{_sn}.lora_A.npy"),
                           _A.astype(np.float16))
                    np.save(os.path.join(_od, f"{_sn}.lora_B.npy"),
                           _B.astype(np.float16))

                if (_i + 1) % 10 == 0 and self._llog:
                    curr_i = _i + 1
                    _thread_safe_update(self._llog, lambda i=curr_i: self._llog.insert(tk.END, f"  Processed {i}/{len(_cw)}\n"))

            if self._llog:
                _thread_safe_update(self._llog, lambda: self._llog.insert(tk.END, f"\nLoRA extracted to {_od}\n  {len(_cw)} tensor pairs, rank={_rk}\n"))

        except Exception as _e:
            err = str(_e)
            if self._llog:
                _thread_safe_update(self._llog, lambda: self._llog.insert(tk.END, f"\nERROR: {err}\n"))

    def _do_compare(self, _pa, _pb):
        try:
            _, _ta, _oa, _ = parse_gguf_metadata(_pa)
            _, _tb, _ob, _ = parse_gguf_metadata(_pb)

            _ma = {_t.name: _t for _t in _ta}
            _mb = {_t.name: _t for _t in _tb}

            _all = sorted(set(_ma.keys()) | set(_mb.keys()))

            _ident = 0
            _mod = 0
            _miss_a = 0
            _miss_b = 0

            for _n in _all:
                if _n not in _ma:
                    _miss_a += 1
                    if self._clog:
                        name = _n
                        _thread_safe_update(self._clog, lambda n=name: self._clog.insert(tk.END, f"  [ONLY IN B] {n}\n"))
                elif _n not in _mb:
                    _miss_b += 1
                    if self._clog:
                        name = _n
                        _thread_safe_update(self._clog, lambda n=name: self._clog.insert(tk.END, f"  [ONLY IN A] {n}\n"))
                else:
                    _a = _ma[_n]
                    _b = _mb[_n]
                    if _a.shape != _b.shape:
                        if self._clog:
                            msg = f"  [SHAPE DIFF] {_n}: {_a.shape} vs {_b.shape}\n"
                            _thread_safe_update(self._clog, lambda m=msg: self._clog.insert(tk.END, m))
                        _mod += 1
                    elif _a.dtype == 0 and _b.dtype == 0:
                        with open(_pa, "rb") as _f:
                            _f.seek(_oa + _a.offset)
                            _aa = np.frombuffer(_f.read(_a.size_bytes), dtype=np.float32)
                        with open(_pb, "rb") as _f:
                            _f.seek(_ob + _b.offset)
                            _ab = np.frombuffer(_f.read(_b.size_bytes), dtype=np.float32)

                        if np.array_equal(_aa, _ab):
                            _ident += 1
                        else:
                            _dot = np.dot(_aa, _ab)
                            _na = np.linalg.norm(_aa)
                            _nb = np.linalg.norm(_ab)
                            _cos = _dot / (_na * _nb + 1e-10)
                            _mad = np.mean(np.abs(_aa - _ab))
                            if self._clog:
                                msg = f"  [MODIFIED] {_n}: cos={_cos:.6f}, MAD={_mad:.6f}\n"
                                _thread_safe_update(self._clog, lambda m=msg: self._clog.insert(tk.END, m))
                            _mod += 1
                    else:
                        if _a.size_bytes == _b.size_bytes:
                            _ident += 1
                        else:
                            _mod += 1

            if self._clog:
                _sep = "=" * 50
                summary = (
                    f"\n{_sep}\n"
                    f"  SUMMARY\n"
                    f"  Identical:  {_ident}\n"
                    f"  Modified:   {_mod}\n"
                    f"  Only in A:  {_miss_b}\n"
                    f"  Only in B:  {_miss_a}\n"
                    f"  Total:      {len(_all)}\n"
                )
                _thread_safe_update(self._clog, lambda: self._clog.insert(tk.END, summary))

        except Exception as _e:
            err = str(_e)
            if self._clog:
                _thread_safe_update(self._clog, lambda: self._clog.insert(tk.END, f"\nERROR: {err}\n"))


# ---------------------------------------------------------------------------
# Real window — open_decompiler
# ---------------------------------------------------------------------------

class DecompilerWindow:
    """Tk window for the Decompiler. No auth, no unlock file, no network.

    Tabs: Analyze, Compare, LoRA Extract, Layer Extract (surgery).
    Core, testable logic lives in _write_layer_subset() and friends, so
    it can be exercised without clicking through filedialogs.
    """

    def __init__(self, parent, colors=None):
        self.colors = dict(_DEFAULT_COLORS)
        if colors:
            self.colors.update(colors)
        c = self.colors
        self.win = tk.Toplevel(parent)
        self.win.title("💀 Decompiler — Nightmare Designs")
        self.win.geometry("1100x750")
        self.win.configure(bg=c["bg"])
        self._panel = DecompilerPanel()
        self._model_path = tk.StringVar(value="")
        self._base_path = tk.StringVar(value="")
        self._tuned_path = tk.StringVar(value="")
        self._cmp_a = tk.StringVar(value="")
        self._cmp_b = tk.StringVar(value="")
        self._start_var = tk.StringVar(value="0")
        self._end_var = tk.StringVar(value="0")
        self._inc_emb = tk.BooleanVar(value=True)
        self._inc_out = tk.BooleanVar(value=True)
        self._rank_var = tk.StringVar(value="8")

        tk.Label(self.win, text="💀 Decompiler", font=("Consolas", 20, "bold"),
                 bg=c["bg"], fg=c["blood_bright"]).pack(anchor="w", padx=16, pady=(12, 0))
        tk.Label(self.win, text="Analyze, compare, LoRA-extract, and extract / delete / split by layer range. Output is F32 GGUF; tensors that cannot be loaded are skipped and logged.",
                 font=("Consolas", 9), bg=c["bg"], fg=c["text_dim"], wraplength=1050, justify="left").pack(anchor="w", padx=16)

        # Model picker
        row = tk.Frame(self.win, bg=c["bg"])
        row.pack(fill=tk.X, padx=16, pady=8)
        tk.Label(row, text="Model:", font=("Consolas", 10), bg=c["bg"], fg=c["text"]).pack(side=tk.LEFT)
        tk.Entry(row, textvariable=self._model_path, bg=c["bg_input"], fg=c["matrix_green"],
                 font=("Consolas", 10), relief=tk.FLAT).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
        tk.Button(row, text="Browse", command=self._browse_model, bg=c["bg_input"], fg=c["matrix_green"],
                  font=("Consolas", 10), relief=tk.FLAT).pack(side=tk.LEFT)

        # Tab buttons
        tab_row = tk.Frame(self.win, bg=c["bg"])
        tab_row.pack(fill=tk.X, padx=16)
        self._tabs = {}
        self._tab_frames = {}
        for name in ("Analyze", "Compare", "LoRA Extract", "Layer Extract"):
            btn = tk.Button(tab_row, text=name, bg=c["bg_input"], fg=c["matrix_green"],
                            font=("Consolas", 10), relief=tk.FLAT,
                            command=lambda n=name: self._show_tab(n))
            btn.pack(side=tk.LEFT, padx=(0, 4))
            frame = tk.Frame(self.win, bg=c["bg_panel"])
            self._tabs[name] = btn
            self._tab_frames[name] = frame
        self._build_analyze_tab()
        self._build_compare_tab()
        self._build_lora_tab()
        self._build_layer_tab()
        self._show_tab("Analyze")

        self.log = scrolledtext.ScrolledText(self.win, bg=c["bg_input"], fg=c["text"],
                                             font=("Consolas", 10), relief=tk.FLAT, height=14)
        self.log.pack(fill=tk.BOTH, expand=True, padx=16, pady=(8, 16))
        self._log("Decompiler ready. Pick a GGUF model above.\n"
                  "Layer surgery here = extract / delete / split by layer range.\n"
                  "It does not edit individual weights in place.\n")

    # -- helpers ---------------------------------------------------------
    def _log(self, text):
        try:
            self.log.insert(tk.END, text)
            self.log.see(tk.END)
        except Exception:
            pass

    def _show_tab(self, name):
        for frame in self._tab_frames.values():
            frame.pack_forget()
        self._tab_frames[name].pack(fill=tk.X, padx=16, pady=4)

    def _browse_model(self):
        path = filedialog.askopenfilename(title="Select Model",
                                          filetypes=[("GGUF Models", "*.gguf"), ("All Files", "*.*")])
        if path:
            self._model_path.set(path)

    def _pick_into(self, var, title):
        path = filedialog.askopenfilename(title=title,
                                          filetypes=[("GGUF Models", "*.gguf"), ("All Files", "*.*")])
        if path:
            var.set(path)

    def _load_model(self, path):
        """Return (metadata, tensors, data_offset) or raise with a clear message."""
        metadata, tensors, offset, _ = parse_gguf_metadata(path)
        if not tensors:
            raise ValueError(f"No tensors parsed from {path} — is it a GGUF file?")
        return metadata, tensors, offset

    # -- Analyze ---------------------------------------------------------
    def _build_analyze_tab(self):
        frame = self._tab_frames["Analyze"]
        tk.Button(frame, text="Analyze Model", command=self._run_analyze,
                  bg="#0d0d0d", fg="#00ff41", font=("Consolas", 11), relief=tk.FLAT).pack(anchor="w")

    def _run_analyze(self):
        path = self._model_path.get().strip()
        if not path:
            self._log("Analyze: pick a model first.\n")
            return
        threading.Thread(target=self._do_analyze, args=(path,), daemon=True).start()

    def _do_analyze(self, path):
        try:
            metadata, tensors, _ = self._load_model(path)
            lines = [f"=== MODEL WEIGHT ANALYSIS ===", f"File: {Path(path).name}",
                     f"Tensors: {len(tensors)}", ""]
            # Quant / dtype summary, same technique as ModelAnalyzerPanel.
            counts = {}
            for t in tensors:
                counts[t.dtype] = counts.get(t.dtype, 0) + 1
            lines.append("=== DTYPE / QUANT SUMMARY ===")
            for dtype, count in sorted(counts.items()):
                lines.append(f"  dtype {dtype}: {count} tensors")
            lines.append("")
            lines.append("=== WEIGHT STATISTICS (first 20 tensors) ===")
            for info in tensors[:20]:
                n_elem = 1
                for dim in info.shape:
                    n_elem *= int(dim)
                if n_elem > 1000000:
                    lines.append(f"  {info.name[-40:]:<40} [large: {n_elem:,} elements, stats skipped]")
                    continue
                try:
                    arr = _load_tensor_f32(path, info.name).astype(np.float32).ravel()
                    lines.append(f"  {info.name[-40:]:<40} mean={float(np.mean(arr)):>8.4f} "
                                 f"std={float(np.std(arr)):>8.4f} min={float(np.min(arr)):>8.4f} "
                                 f"max={float(np.max(arr)):>8.4f} "
                                 f"sparsity={float(np.sum(np.abs(arr) < 1e-6)) / max(len(arr), 1) * 100:>5.1f}%")
                except Exception as exc:
                    lines.append(f"  {info.name[-40:]:<40} [read error: {exc}]")
            if len(tensors) > 20:
                lines.append(f"\n  ... and {len(tensors) - 20} more tensors")
            if metadata:
                lines.append("\n=== METADATA (first 20) ===")
                for key in sorted(metadata)[:20]:
                    lines.append(f"  {key}: {metadata[key]}")
            self._log("\n".join(lines) + "\n")
        except Exception as exc:
            self._log(f"Analyze error: {exc}\n")

    # -- Compare ---------------------------------------------------------
    def _build_compare_tab(self):
        frame = self._tab_frames["Compare"]
        for label, var in (("Model A:", self._cmp_a), ("Model B:", self._cmp_b)):
            row = tk.Frame(frame, bg="#000000")
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, font=("Consolas", 10), bg="#000000", fg="#cccccc").pack(side=tk.LEFT)
            tk.Entry(row, textvariable=var, bg="#0d0d0d", fg="#00ff41",
                     font=("Consolas", 10), relief=tk.FLAT).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
            tk.Button(row, text="Browse", command=lambda v=var: self._pick_into(v, "Select Model"),
                      bg="#0d0d0d", fg="#00ff41", font=("Consolas", 10), relief=tk.FLAT).pack(side=tk.LEFT)
        tk.Button(frame, text="Compare Models", command=self._run_compare,
                  bg="#0d0d0d", fg="#ff6600", font=("Consolas", 11), relief=tk.FLAT).pack(anchor="w", pady=4)

    def _run_compare(self):
        a, b = self._cmp_a.get().strip(), self._cmp_b.get().strip()
        if not a or not b:
            self._log("Compare: pick both Model A and Model B.\n")
            return
        # Reuse the existing compare logic, writing to this window's log.
        panel = DecompilerPanel(llog_widget=self.log, clog_widget=self.log)
        threading.Thread(target=panel._do_compare, args=(a, b), daemon=True).start()

    # -- LoRA ------------------------------------------------------------
    def _build_lora_tab(self):
        frame = self._tab_frames["LoRA Extract"]
        for label, var in (("Base model:", self._base_path), ("Tuned model:", self._tuned_path)):
            row = tk.Frame(frame, bg="#000000")
            row.pack(fill=tk.X, pady=2)
            tk.Label(row, text=label, font=("Consolas", 10), bg="#000000", fg="#cccccc").pack(side=tk.LEFT)
            tk.Entry(row, textvariable=var, bg="#0d0d0d", fg="#00ff41",
                     font=("Consolas", 10), relief=tk.FLAT).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
            tk.Button(row, text="Browse", command=lambda v=var: self._pick_into(v, "Select Model"),
                      bg="#0d0d0d", fg="#00ff41", font=("Consolas", 10), relief=tk.FLAT).pack(side=tk.LEFT)
        row = tk.Frame(frame, bg="#000000")
        row.pack(fill=tk.X, pady=4)
        tk.Label(row, text="Rank:", font=("Consolas", 10), bg="#000000", fg="#cccccc").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=self._rank_var, width=6, bg="#0d0d0d", fg="#00ff41",
                 font=("Consolas", 10), relief=tk.FLAT).pack(side=tk.LEFT, padx=8)
        tk.Button(frame, text="Extract LoRA (asks for output folder)",
                  command=self._run_lora, bg="#0d0d0d", fg="#00ff41",
                  font=("Consolas", 11), relief=tk.FLAT).pack(anchor="w", pady=4)

    def _run_lora(self):
        base, tuned = self._base_path.get().strip(), self._tuned_path.get().strip()
        if not base or not tuned:
            self._log("LoRA: pick both base and tuned models.\n")
            return
        try:
            rank = int(self._rank_var.get())
        except Exception:
            rank = 8
        panel = DecompilerPanel(llog_widget=self.log, clog_widget=self.log)
        threading.Thread(target=panel._do_lora_extract, args=(base, tuned, rank), daemon=True).start()

    # -- Layer extract / delete / split ----------------------------------
    def _build_layer_tab(self):
        frame = self._tab_frames["Layer Extract"]
        row = tk.Frame(frame, bg="#000000")
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text="Layer Start:", font=("Consolas", 10), bg="#000000", fg="#cccccc").pack(side=tk.LEFT)
        tk.Entry(row, textvariable=self._start_var, width=6, bg="#0d0d0d", fg="#00ff41",
                 font=("Consolas", 10), relief=tk.FLAT).pack(side=tk.LEFT, padx=8)
        tk.Label(row, text="Layer End:", font=("Consolas", 10), bg="#000000", fg="#cccccc").pack(side=tk.LEFT, padx=(16, 0))
        tk.Entry(row, textvariable=self._end_var, width=6, bg="#0d0d0d", fg="#00ff41",
                 font=("Consolas", 10), relief=tk.FLAT).pack(side=tk.LEFT, padx=8)
        tk.Checkbutton(frame, text="Include Embeddings", variable=self._inc_emb,
                       bg="#000000", fg="#cccccc", selectcolor="#0d0d0d",
                       font=("Consolas", 9)).pack(anchor="w")
        tk.Checkbutton(frame, text="Include Output Head / Norm", variable=self._inc_out,
                       bg="#000000", fg="#cccccc", selectcolor="#0d0d0d",
                       font=("Consolas", 9)).pack(anchor="w")
        btn_row = tk.Frame(frame, bg="#000000")
        btn_row.pack(fill=tk.X, pady=4)
        tk.Button(btn_row, text="Extract Layers", command=lambda: self._run_layer("keep"),
                  bg="#0d0d0d", fg="#00ff41", font=("Consolas", 11), relief=tk.FLAT).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="Delete Layers", command=lambda: self._run_layer("delete"),
                  bg="#0d0d0d", fg="#ff0000", font=("Consolas", 11), relief=tk.FLAT).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_row, text="Split Model", command=self._run_split,
                  bg="#0d0d0d", fg="#00ff41", font=("Consolas", 11), relief=tk.FLAT).pack(side=tk.LEFT, padx=4)
        tk.Label(frame, text="Delete = keep every layer outside the range, renumbered from 0. Split = two files, first half and second half. Quantized tensors are dequantized to F32 on write.",
                 font=("Consolas", 9), bg="#000000", fg="#666666", wraplength=1000, justify="left").pack(anchor="w")

    def _run_layer(self, mode):
        path = self._model_path.get().strip()
        if not path:
            self._log("Layer extract: pick a model first.\n")
            return
        try:
            start, end = int(self._start_var.get()), int(self._end_var.get())
        except Exception:
            self._log("Layer extract: start/end must be integers.\n")
            return
        out = filedialog.asksaveasfilename(title="Save Layer Output", defaultextension=".gguf",
                                           filetypes=[("GGUF", "*.gguf")])
        if not out:
            return
        threading.Thread(target=self._layer_job, args=(path, out, start, end, mode), daemon=True).start()

    def _layer_job(self, path, out, start, end, mode):
        try:
            kept, skipped = self._write_layer_subset(
                path, out, start, end, mode,
                include_emb=self._inc_emb.get(), include_out=self._inc_out.get())
            self._log(f"\nLayer {mode}: wrote {kept} tensors to {out} (skipped {skipped}).\n")
        except Exception as exc:
            self._log(f"\nLayer {mode} error: {exc}\n")

    def _run_split(self):
        path = self._model_path.get().strip()
        if not path:
            self._log("Split: pick a model first.\n")
            return
        folder = filedialog.askdirectory(title="Save Split Outputs To...")
        if not folder:
            return
        threading.Thread(target=self._split_job, args=(path, folder), daemon=True).start()

    def _split_job(self, path, folder):
        try:
            _, tensors, _ = self._load_model(path)
            layers = sorted({i for i in (_layer_index(t.name) for t in tensors) if i is not None})
            if len(layers) < 2:
                self._log("Split: fewer than 2 layers found, cannot split.\n")
                return
            mid = layers[len(layers) // 2]
            first = Path(folder) / "split_first_half.gguf"
            second = Path(folder) / "split_second_half.gguf"
            k1, s1 = self._write_layer_subset(path, str(first), layers[0], mid - 1, "keep")
            k2, s2 = self._write_layer_subset(path, str(second), mid, layers[-1], "keep")
            self._log(f"\nSplit at layer {mid}: first half {k1} tensors (skipped {s1}) -> {first}\n"
                      f"second half {k2} tensors (skipped {s2}) -> {second}\n")
        except Exception as exc:
            self._log(f"\nSplit error: {exc}\n")

    def _select_tensors(self, tensors, start, end, mode, include_emb, include_out):
        """Return (kept, renumber_map). Kept blk indices map to 0.. in order."""
        kept_blk = []
        for t in tensors:
            idx = _layer_index(t.name)
            if idx is None:
                if include_emb and _is_embedding(t.name):
                    kept_blk.append(t)
                elif include_out and _is_output_head(t.name):
                    kept_blk.append(t)
            else:
                in_range = start <= idx <= end
                if (mode == "keep" and in_range) or (mode == "delete" and not in_range):
                    kept_blk.append(t)
        kept_layers = sorted({i for i in (_layer_index(t.name) for t in kept_blk) if i is not None})
        renumber = {old: new for new, old in enumerate(kept_layers)}
        return kept_blk, renumber

    def _write_layer_subset(self, path, out_path, start, end, mode,
                            include_emb=True, include_out=True):
        """Write a GGUF containing only the selected tensors. Returns (kept, skipped)."""
        from npy_to_gguf import GGUFWriter, GGML_TYPE_F32
        metadata, tensors, _ = self._load_model(path)
        kept, renumber = self._select_tensors(tensors, start, end, mode, include_emb, include_out)
        if not kept:
            raise ValueError("No tensors selected for that layer range.")
        writer = GGUFWriter(out_path)
        for key, value in metadata.items():
            if isinstance(value, str):
                writer.add_metadata_string(key, value)
            elif isinstance(value, int):
                writer.add_metadata_uint32(key, value)
            elif isinstance(value, float):
                writer.add_metadata_float32(key, value)
        if kept and any(_layer_index(t.name) is not None for t in kept):
            # Update block count to the renumbered layer count, if present.
            try:
                arch = metadata.get("general.architecture", "")
                if arch and f"{arch}.block_count" in metadata:
                    writer.metadata[f"{arch}.block_count"] = (4, len(renumber))
            except Exception:
                pass
        kept_count, skipped = 0, 0
        for info in kept:
            try:
                arr = _load_tensor_f32(path, info.name)
            except Exception:
                skipped += 1
                continue
            new_name = info.name
            idx = _layer_index(info.name)
            if idx is not None and idx in renumber:
                new_name = re.sub(r"blk\.\d+\.", f"blk.{renumber[idx]}.", info.name, count=1)
            writer.add_tensor(new_name, np.asarray(arr, dtype=np.float32), GGML_TYPE_F32)
            kept_count += 1
        if kept_count == 0:
            raise ValueError("No tensors could be loaded; nothing written.")
        writer.write()
        return kept_count, skipped


def open_decompiler(parent_window, colors=None):
    """Open the Decompiler window. Returns the DecompilerWindow.

    This is the entry point the main menu calls. It does not check any
    unlock file, master token, or the assistant/Brain state.
    """
    window = DecompilerWindow(parent_window, colors=colors)
    return window
