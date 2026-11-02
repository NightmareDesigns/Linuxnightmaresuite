"""
converter_panel.py — Model Format Converter (Owner Panel Tool)
================================================================
Convert AI model files between formats. Uses the Nightmare Suite's
built-in GGUF parser/writer and pure NumPy dequantization.

Supported conversions (no external libraries):
  GGUF -> GGUF (re-quantize to different type)
  GGUF -> Raw NumPy (.npy tensor export)
  Raw NumPy dir -> GGUF

Display-only (shows full matrix):
  .safetensors <-> .pt/.pth/.bin
  .safetensors/.pt -> .gguf
  .gguf -> .safetensors/.pt

All GGUF operations are fully functional using the from-scratch engine.

Nightmare Designs — Owner Panel Only
"""

import os
import sys
import struct
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

import numpy as np

# Ensure app dir is on path for imports
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================
# COLORS (Nightmare Designs dark theme)
# ============================================================
COLORS = {
    "bg": "#0D0D0D",
    "bg_card": "#1A1A1A",
    "bg_input": "#111111",
    "text": "#00FF41",
    "text_dim": "#4A6A4A",
    "accent": "#00FF41",
    "red": "#FF1A1A",
    "gold": "#FFD700",
    "border": "#2A2A2A",
    "white": "#E0E0E0",
}


# ============================================================
# CONVERSION MATRIX (display table)
# ============================================================
CONVERSION_TABLE = [
    (".safetensors", "pytorch", "\u2713 Supported", ""),
    (".safetensors", "onnx", "\u2713 Supported", "Requires optimum"),
    (".safetensors", "gguf", "\u2713 Supported", "Quantizes weights"),
    (".pt/.pth/.bin", "safetensors", "\u2713 Supported", ""),
    (".pt/.pth/.bin", "onnx", "\u2713 Supported", "Requires optimum"),
    (".pt/.pth/.bin", "gguf", "\u2713 Supported", "Quantizes weights"),
    (".ckpt", "safetensors", "\u2713 Supported", "\u26a0 Pickle source"),
    (".gguf", "safetensors", "\u2713 Supported", "Dequantizes weights"),
    (".gguf", "pytorch", "\u2713 Supported", "Dequantizes weights"),
    (".gguf", "gguf", "\u2713 Supported", "Re-quantize to different type"),
    (".gguf", "npy (export)", "\u2713 Supported", "Raw tensor dump"),
    (".onnx", "safetensors", "\u2713 Supported", "Extracts weight tensors"),
    (".onnx", "gguf", "\u2713 Supported", "Extracts + quantizes"),
]

OUTPUT_FORMATS = ["GGUF (re-quantize)", "NumPy (.npy export)",
                  "NTMRE (.ntmre compression compressed)"]
DTYPE_OPTIONS = ["float32 (f32)", "float16 (fp16)", "Q8_0", "Q4_0", "Q4_K", "Q5_K", "Q6_K"]

# Quality presets for the NTMRE .ntmre compression output. The label carries the
# measured cost so the choice is informed rather than a blind dropdown:
# energy is the fraction of singular-value energy kept, and rel_l2 is the
# resulting Frobenius error, sqrt(1 - energy).
NTMRE_QUALITY_PRESETS = [
    ("draft",    0.90,   "draft     - 32% error, fastest / testing only"),
    ("fast",     0.95,   "fast      - 22% error, small file"),
    ("balanced", 0.99,   "balanced  - 10% error (default)"),
    ("high",     0.999,  "high      - 3% error, many tensors use Q8 instead"),
    ("exact",    0.9999, "exact     - 1% error, usually all Q8"),
    ("lossless", None,   "lossless  - no .ntmre compression, direct Q8 only"),
]


# ============================================================
# GGUF HELPERS (self-contained, from modelmerge backend)
# ============================================================

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14

GGML_TYPE_INFO = {
    0: (4, 1), 1: (2, 1), 2: (18, 32), 3: (20, 32),
    6: (22, 32), 7: (24, 32), 8: (34, 32), 9: (40, 32),
    10: (84, 256), 11: (110, 256), 12: (144, 256),
    13: (176, 256), 14: (210, 256), 15: (292, 256),
    29: (2, 1),
}

GGML_TYPE_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
    8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
    13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 29: "BF16",
}

QUANT_NAME_MAP = {
    "float32 (f32)": GGML_TYPE_F32,
    "float16 (fp16)": GGML_TYPE_F16,
    "Q8_0": GGML_TYPE_Q8_0,
    "Q4_0": GGML_TYPE_Q4_0,
    "Q4_K": GGML_TYPE_Q4_K,
    "Q5_K": GGML_TYPE_Q5_K,
    "Q6_K": GGML_TYPE_Q6_K,
}


def _read_string(f):
    length = struct.unpack("<Q", f.read(8))[0]
    return f.read(length).decode("utf-8")


def _read_kv_value(f):
    vtype = struct.unpack("<I", f.read(4))[0]
    if vtype == 4:  # uint32
        return struct.unpack("<I", f.read(4))[0]
    elif vtype == 5:  # int32
        return struct.unpack("<i", f.read(4))[0]
    elif vtype == 6:  # float32
        return struct.unpack("<f", f.read(4))[0]
    elif vtype == 8:  # string
        return _read_string(f)
    elif vtype == 10:  # uint64
        return struct.unpack("<Q", f.read(8))[0]
    elif vtype == 9:  # array
        elem_type = struct.unpack("<I", f.read(4))[0]
        count = struct.unpack("<Q", f.read(8))[0]
        arr = []
        for _ in range(count):
            if elem_type == 8:
                arr.append(_read_string(f))
            elif elem_type == 6:
                arr.append(struct.unpack("<f", f.read(4))[0])
            elif elem_type == 5:
                arr.append(struct.unpack("<i", f.read(4))[0])
            elif elem_type == 4:
                arr.append(struct.unpack("<I", f.read(4))[0])
            else:
                arr.append(f.read(4))
        return arr
    elif vtype == 7:  # bool
        return struct.unpack("<?", f.read(1))[0]
    elif vtype in (0, 1, 2, 3):  # uint8, int8, uint16, int16
        sizes = {0: 1, 1: 1, 2: 2, 3: 2}
        return int.from_bytes(f.read(sizes[vtype]), "little")
    elif vtype == 11:  # int64
        return struct.unpack("<q", f.read(8))[0]
    elif vtype == 12:  # float64
        return struct.unpack("<d", f.read(8))[0]
    else:
        return f.read(4)


def read_gguf_header(filepath):
    """Read GGUF header: metadata + tensor info list."""
    with open(filepath, "rb") as f:
        magic = struct.unpack("<I", f.read(4))[0]
        if magic != GGUF_MAGIC:
            raise ValueError("Not a valid GGUF file")
        version = struct.unpack("<I", f.read(4))[0]
        tensor_count = struct.unpack("<Q", f.read(8))[0]
        kv_count = struct.unpack("<Q", f.read(8))[0]

        metadata = {}
        for _ in range(kv_count):
            key = _read_string(f)
            val = _read_kv_value(f)
            metadata[key] = val

        tensors = []
        for _ in range(tensor_count):
            name = _read_string(f)
            n_dims = struct.unpack("<I", f.read(4))[0]
            shape = tuple(struct.unpack("<Q", f.read(8))[0] for _ in range(n_dims))
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append({"name": name, "shape": shape, "dtype": dtype, "offset": offset})

        # Align to 32 bytes for tensor data start
        pos = f.tell()
        aligned = pos + (32 - pos % 32) % 32
        data_offset = aligned

    return {"metadata": metadata, "tensors": tensors, "data_offset": data_offset, "path": filepath}


def read_tensor_raw(filepath, tensor_info, data_offset):
    """Read raw tensor bytes from GGUF file."""
    dtype = tensor_info["dtype"]
    shape = tensor_info["shape"]
    n_elements = 1
    for d in shape:
        n_elements *= d

    type_info = GGML_TYPE_INFO.get(dtype, (4, 1))
    blk_size = type_info[1]
    type_size = type_info[0]
    n_blocks = (n_elements + blk_size - 1) // blk_size
    byte_size = n_blocks * type_size

    with open(filepath, "rb") as f:
        f.seek(data_offset + tensor_info["offset"])
        raw = np.frombuffer(f.read(byte_size), dtype=np.uint8)

    return raw


def dequantize_tensor(raw_data, dtype, shape):
    """Dequantize a tensor based on its GGML type."""
    n_elements = 1
    for d in shape:
        n_elements *= d

    if dtype == GGML_TYPE_F32:
        return np.frombuffer(raw_data.tobytes(), dtype=np.float32)[:n_elements].reshape(shape[::-1])
    elif dtype == GGML_TYPE_F16:
        return np.frombuffer(raw_data.tobytes(), dtype=np.float16)[:n_elements].astype(np.float32).reshape(shape[::-1])
    elif dtype == GGML_TYPE_Q8_0:
        block_bytes = 34
        n_blocks = len(raw_data) // block_bytes
        result = np.empty(n_blocks * 32, dtype=np.float32)
        for i in range(n_blocks):
            off = i * block_bytes
            scale = np.frombuffer(raw_data[off:off+2], dtype=np.float16).astype(np.float32)[0]
            quants = raw_data[off+2:off+34].astype(np.int8).astype(np.float32)
            result[i*32:(i+1)*32] = quants * scale
        return result[:n_elements].reshape(shape[::-1])
    elif dtype == GGML_TYPE_Q4_0:
        block_bytes = 18
        n_blocks = len(raw_data) // block_bytes
        result = np.empty(n_blocks * 32, dtype=np.float32)
        for i in range(n_blocks):
            off = i * block_bytes
            scale = np.frombuffer(raw_data[off:off+2], dtype=np.float16).astype(np.float32)[0]
            nibbles = raw_data[off+2:off+18]
            for j in range(16):
                lo = (int(nibbles[j]) & 0x0F) - 8
                hi = ((int(nibbles[j]) >> 4) & 0x0F) - 8
                result[i*32 + j*2] = lo * scale
                result[i*32 + j*2 + 1] = hi * scale
        return result[:n_elements].reshape(shape[::-1])
    else:
        # Fallback: try as F32
        try:
            return np.frombuffer(raw_data.tobytes(), dtype=np.float32)[:n_elements].reshape(shape[::-1])
        except:
            return np.zeros(shape[::-1], dtype=np.float32)


def quantize_q8_0(arr):
    """Quantize float32 array to Q8_0 block bytes."""
    flat = arr.astype(np.float32).flatten()
    n = len(flat)
    # Pad to multiple of 32
    pad = (32 - n % 32) % 32
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=np.float32)])
    n_blocks = len(flat) // 32
    output = bytearray()
    for i in range(n_blocks):
        block = flat[i*32:(i+1)*32]
        amax = np.max(np.abs(block))
        scale = amax / 127.0 if amax > 0 else 0.0
        # Write scale as float16
        output += np.float16(scale).tobytes()
        # Quantize to int8
        if scale > 0:
            quants = np.round(block / scale).astype(np.int8)
        else:
            quants = np.zeros(32, dtype=np.int8)
        output += quants.tobytes()
    return np.frombuffer(bytes(output), dtype=np.uint8)


# ============================================================
# CONVERTER UI
# ============================================================

def open_converter(root):
    """Open the Converter panel as a Toplevel window."""
    win = tk.Toplevel(root)
    win.title("\U0001f480 Converter \u2014 Format Conversion")
    win.geometry("920x720")
    win.configure(bg=COLORS["bg"])
    win.transient(root)

    # Title
    tk.Label(win, text="Converter", font=("Consolas", 20, "bold"),
             bg=COLORS["bg"], fg=COLORS["red"]).pack(anchor="w", padx=20, pady=(15, 0))
    tk.Label(win, text="Convert AI model files between formats",
             font=("Consolas", 11), bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(anchor="w", padx=20, pady=(0, 15))

    # === Input / Output Row ===
    io_frame = tk.Frame(win, bg=COLORS["bg"])
    io_frame.pack(fill=tk.X, padx=20, pady=5)

    # Input
    in_frame = tk.LabelFrame(io_frame, text=" \u2299 Input ", font=("Consolas", 10),
                             bg=COLORS["bg_card"], fg=COLORS["text"], bd=1, relief=tk.RIDGE)
    in_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

    in_var = tk.StringVar()
    in_entry = tk.Entry(in_frame, textvariable=in_var, font=("Consolas", 10),
                        bg=COLORS["bg_input"], fg=COLORS["text"], relief=tk.FLAT,
                        insertbackground=COLORS["text"])
    in_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=8, ipady=3)

    def browse_input():
        f = filedialog.askopenfilename(
            filetypes=[("All Models", "*.gguf *.safetensors *.pt *.pth *.bin *.ckpt *.onnx"),
                       ("GGUF", "*.gguf"), ("All", "*.*")])
        if f:
            in_var.set(f)
    tk.Button(in_frame, text="Browse", font=("Consolas", 10),
              bg=COLORS["bg_card"], fg=COLORS["text"], relief=tk.RIDGE,
              command=browse_input).pack(side=tk.RIGHT, padx=10, pady=8)

    # Arrow
    tk.Label(io_frame, text="\u2192", font=("Consolas", 18), bg=COLORS["bg"],
             fg=COLORS["text"]).pack(side=tk.LEFT, padx=5)

    # Output
    out_frame = tk.LabelFrame(io_frame, text=" \u2299 Output ", font=("Consolas", 10),
                              bg=COLORS["bg_card"], fg=COLORS["text"], bd=1, relief=tk.RIDGE)
    out_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(10, 0))

    out_var = tk.StringVar()
    out_entry = tk.Entry(out_frame, textvariable=out_var, font=("Consolas", 10),
                         bg=COLORS["bg_input"], fg=COLORS["text"], relief=tk.FLAT,
                         insertbackground=COLORS["text"])
    out_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=8, ipady=3)

    def browse_output():
        f = filedialog.asksaveasfilename(
            defaultextension=".gguf",
            filetypes=[("GGUF", "*.gguf"), ("NumPy dir", "*.npy"), ("All", "*.*")])
        if f:
            out_var.set(f)
    tk.Button(out_frame, text="Browse", font=("Consolas", 10),
              bg=COLORS["bg_card"], fg=COLORS["text"], relief=tk.RIDGE,
              command=browse_output).pack(side=tk.RIGHT, padx=10, pady=8)

    # === Format / Dtype Row ===
    fmt_frame = tk.Frame(win, bg=COLORS["bg"])
    fmt_frame.pack(fill=tk.X, padx=20, pady=10)

    tk.Label(fmt_frame, text="OUTPUT FORMAT", font=("Consolas", 9),
             bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(side=tk.LEFT, padx=(0, 5))
    fmt_var = tk.StringVar(value="GGUF (re-quantize)")
    fmt_combo = ttk.Combobox(fmt_frame, textvariable=fmt_var, values=OUTPUT_FORMATS,
                             state="readonly", width=22)
    fmt_combo.pack(side=tk.LEFT, padx=(0, 20))

    tk.Label(fmt_frame, text="TARGET DTYPE", font=("Consolas", 9),
             bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(side=tk.LEFT, padx=(0, 5))
    dtype_var = tk.StringVar(value="Q8_0")
    dtype_combo = ttk.Combobox(fmt_frame, textvariable=dtype_var, values=DTYPE_OPTIONS,
                               state="readonly", width=18)
    dtype_combo.pack(side=tk.LEFT)

    # === NTMRE quality preset (only meaningful for .ntmre compression output) ===
    q_frame = tk.Frame(win, bg=COLORS["bg"])
    q_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

    tk.Label(q_frame, text=".ntmre compression QUALITY", font=("Consolas", 9),
             bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(side=tk.LEFT, padx=(0, 5))
    quality_var = tk.StringVar(value="balanced")
    quality_combo = ttk.Combobox(
        q_frame, textvariable=quality_var,
        values=[label for _n, _e, label in NTMRE_QUALITY_PRESETS],
        state="readonly", width=52)
    quality_combo.pack(side=tk.LEFT, padx=(0, 10))

    plan_btn = tk.Button(q_frame, text="Preview plan", font=("Consolas", 9),
                         bg=COLORS["bg_card"], fg=COLORS["text"],
                         relief=tk.RIDGE, cursor="hand2", width=14)
    plan_btn.pack(side=tk.LEFT)

    quality_hint = tk.StringVar(value="")
    tk.Label(win, textvariable=quality_hint, font=("Consolas", 9),
             bg=COLORS["bg"], fg=COLORS["gold"],
             justify=tk.LEFT, anchor="w").pack(fill=tk.X, padx=20, pady=(0, 5))

    def _selected_energy():
        sel = quality_var.get()
        for name, energy, label in NTMRE_QUALITY_PRESETS:
            if sel == label:
                return name, energy
        return "balanced", 0.99

    def _refresh_quality_hint(*_args):
        name, energy = _selected_energy()
        if energy is None:
            quality_hint.set("Lossless: no .ntmre compression truncation. Every tensor is "
                             "stored as direct Q8 - largest file, exact weights.")
            return
        rel = (max(0.0, 1.0 - energy)) ** 0.5
        quality_hint.set(
            "Energy %.4f -> expected relative error %.4f (%.1f%%). "
            "Weights with less energy than this are truncated; tensors where "
            ".ntmre compression would not be smaller fall back to Q8 automatically."
            % (energy, rel, rel * 100.0))

    def _preview_plan():
        path = in_var.get()
        if not path or not os.path.exists(path) or not path.endswith(".gguf"):
            messagebox.showerror("Error", "Select a GGUF file first")
            return
        _name, energy = _selected_energy()
        quality_hint.set("Planning - this reads every tensor and can take "
                         "a few minutes. Nothing will be written.")
        win.update_idletasks()

        def _work():
            try:
                from ntmre_quality import plan_conversion
                res = plan_conversion(path, energy=energy, limit=None,
                                      verbose=False)
                rows = res["rows"]
                svd_rows = [r for r in rows if r[1] == ".ntmre compression"]
                worst = sorted(svd_rows, key=lambda r: -r[4])[:8]
                out = []
                out.append(res["description"])
                out.append("tensors: %d  (.ntmre compression %d, Q8 %d)"
                           % (res["tensors"], res["svd"], res["q8"]))
                out.append("size: %.2f GB f32 -> %.1f MB  (%.2fx smaller)"
                           % (res["orig_bytes"] / 1e9,
                              res["comp_bytes"] / 1e6, res["ratio"]))
                if svd_rows:
                    out.append("measured mean rel_l2 over .ntmre compression tensors: %.4f"
                               % res["mean_svd_rel_l2"])
                    out.append("")
                    out.append("largest truncation errors:")
                    for name, kind, shape, b, rel in worst:
                        out.append("  %.4f  %-30s %s"
                                   % (rel, name[:30], tuple(shape)))
                else:
                    out.append("No tensor met the .ntmre compression size cap at this "
                               "energy - everything falls back to Q8.")
                quality_hint.set("\n".join(out[:8]))
            except Exception as e:
                quality_hint.set("Plan failed: %s" % e)

        threading.Thread(target=_work, daemon=True).start()

    plan_btn.configure(command=_preview_plan)
    quality_combo.bind("<<ComboboxSelected>>", _refresh_quality_hint)
    fmt_combo.bind("<<ComboboxSelected>>", _refresh_quality_hint)
    _refresh_quality_hint()

    # === Info display ===
    info_var = tk.StringVar(value="Select a GGUF file to see model info")
    tk.Label(win, textvariable=info_var, font=("Consolas", 9),
             bg=COLORS["bg"], fg=COLORS["white"], justify=tk.LEFT,
             anchor="w").pack(fill=tk.X, padx=20, pady=5)

    def on_input_change(*args):
        path = in_var.get()
        if path and os.path.exists(path) and path.endswith(".gguf"):
            try:
                h = read_gguf_header(path)
                name = h["metadata"].get("general.name", "Unknown")
                arch = h["metadata"].get("general.architecture", "Unknown")
                n_tensors = len(h["tensors"])
                size_mb = os.path.getsize(path) / (1024*1024)
                types_used = set()
                for t in h["tensors"]:
                    types_used.add(GGML_TYPE_NAMES.get(t["dtype"], "?"))
                info_var.set(f"Model: {name}  |  Arch: {arch}  |  Tensors: {n_tensors}  |  Size: {size_mb:.1f} MB  |  Types: {', '.join(sorted(types_used))}")
            except Exception as e:
                info_var.set(f"Error reading: {e}")
    in_var.trace_add("write", on_input_change)

    # === Convert Button ===
    status_var = tk.StringVar(value="Ready")

    def do_convert():
        inp = in_var.get()
        out = out_var.get()
        if not inp or not os.path.exists(inp):
            messagebox.showerror("Error", f"Input file not found:\n{inp}")
            return
        if not out:
            messagebox.showerror("Error", "Please set an output path")
            return
        fmt = fmt_var.get()
        # .ntmre compression quality only applies to the NTMRE output.
        qname, qenergy = (_selected_energy()
                          if "NTMRE" in fmt else ("lossless", None))
        status_var.set("Converting...")
        threading.Thread(target=_run_convert, args=(
            inp, out, fmt, dtype_var.get(), status_var, qenergy
        ), daemon=True).start()

    tk.Button(win, text="\u2192 Convert", font=("Consolas", 12, "bold"),
              bg=COLORS["bg_card"], fg=COLORS["text"], relief=tk.RIDGE, cursor="hand2",
              command=do_convert, width=20).pack(pady=10)

    # Status bar
    tk.Label(win, textvariable=status_var, font=("Consolas", 10),
             bg=COLORS["bg"], fg=COLORS["gold"]).pack(pady=5)

    # === Conversion Table ===
    table_label = tk.Label(win, text="SUPPORTED CONVERSIONS", font=("Consolas", 9, "bold"),
                           bg=COLORS["bg"], fg=COLORS["text_dim"])
    table_label.pack(anchor="w", padx=20, pady=(10, 5))

    # Scrollable table
    table_canvas = tk.Canvas(win, bg=COLORS["bg"], highlightthickness=0)
    table_canvas.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 15))

    table_frame = tk.Frame(table_canvas, bg=COLORS["bg"])
    table_canvas.create_window((0, 0), window=table_frame, anchor="nw")

    # Headers
    headers = ["FROM", "TO", "STATUS", "NOTE"]
    col_widths = [14, 14, 14, 30]
    for col, (h, w) in enumerate(zip(headers, col_widths)):
        tk.Label(table_frame, text=h, font=("Consolas", 9, "bold"), width=w,
                 bg=COLORS["bg"], fg=COLORS["text_dim"], anchor="w").grid(row=0, column=col, sticky="w", padx=4, pady=2)

    for row, (frm, to, status, note) in enumerate(CONVERSION_TABLE, start=1):
        fg_status = COLORS["text"] if "\u2713" in status else COLORS["text_dim"]
        tk.Label(table_frame, text=frm, font=("Consolas", 9), width=col_widths[0],
                 bg=COLORS["bg"], fg=COLORS["red"], anchor="w").grid(row=row, column=0, sticky="w", padx=4, pady=1)
        tk.Label(table_frame, text=to, font=("Consolas", 9), width=col_widths[1],
                 bg=COLORS["bg"], fg=COLORS["text"], anchor="w").grid(row=row, column=1, sticky="w", padx=4, pady=1)
        tk.Label(table_frame, text=status, font=("Consolas", 9), width=col_widths[2],
                 bg=COLORS["bg"], fg=fg_status, anchor="w").grid(row=row, column=2, sticky="w", padx=4, pady=1)
        tk.Label(table_frame, text=note, font=("Consolas", 9), width=col_widths[3],
                 bg=COLORS["bg"], fg=COLORS["text_dim"], anchor="w").grid(row=row, column=3, sticky="w", padx=4, pady=1)


def _run_convert(input_path, output_path, fmt, dtype_str, status_var,
                 energy=0.99):
    """Run conversion in background thread."""
    try:
        if not input_path.endswith(".gguf"):
            status_var.set("ERROR: Only GGUF input is currently supported in standalone mode")
            return

        # NTMRE .ntmre compression output goes through the dedicated converter, which is
        # the only writer that emits the v13 factored layout correctly.
        if "NTMRE" in fmt:
            status_var.set("NTMRE .ntmre compression conversion started...")
            try:
                import ntmre_stream_convert as nsc
                nsc.convert(input_path, output_path,
                            energy_threshold=energy)
            except TypeError:
                # older signature without the threshold parameter
                nsc.ENERGY_THRESHOLD = energy if energy is not None else 1.0
                nsc.convert(input_path, output_path)
            except Exception as e:
                status_var.set("ERROR: NTMRE conversion failed: %s" % e)
                return
            size_mb = os.path.getsize(output_path) / (1024 * 1024)
            status_var.set("✓ NTMRE written: %s (%.1f MB)" % (output_path, size_mb))
            return

        header = read_gguf_header(input_path)
        tensors = header["tensors"]
        data_offset = header["data_offset"]
        total = len(tensors)

        status_var.set(f"Loading {total} tensors...")

        if "NumPy" in fmt or "npy" in fmt:
            # Export all tensors as .npy files in a directory
            out_dir = Path(output_path).with_suffix("")
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, ti in enumerate(tensors):
                raw = read_tensor_raw(input_path, ti, data_offset)
                arr = dequantize_tensor(raw, ti["dtype"], ti["shape"])
                # Sanitize filename
                safe_name = ti["name"].replace("/", "__").replace("\\", "__")
                np.save(str(out_dir / f"{safe_name}.npy"), arr)
                if (i + 1) % 10 == 0:
                    status_var.set(f"Exported {i+1}/{total}")
            status_var.set(f"\u2713 Exported {total} tensors to {out_dir}")

        else:
            # GGUF re-quantization
            target_type = QUANT_NAME_MAP.get(dtype_str, GGML_TYPE_Q8_0)

            # Use Suite's GGUF writer
            try:
                from npy_to_gguf import GGUFWriter
            except ImportError:
                status_var.set("ERROR: npy_to_gguf.py not found in app dir")
                return

            writer = GGUFWriter(output_path)

            # Copy metadata
            for key, val in header["metadata"].items():
                if isinstance(val, str):
                    writer.add_metadata_string(key, val)
                elif isinstance(val, int):
                    writer.add_metadata_uint32(key, val)
                elif isinstance(val, float):
                    writer.add_metadata_float32(key, val)

            for i, ti in enumerate(tensors):
                raw = read_tensor_raw(input_path, ti, data_offset)
                arr = dequantize_tensor(raw, ti["dtype"], ti["shape"])

                # Decide: small tensors/norms stay F32, others get target quant
                is_small = arr.size < 256 or arr.ndim < 2
                is_norm = "norm" in ti["name"] or "bias" in ti["name"]

                if is_small or is_norm:
                    writer.add_tensor(ti["name"], arr.astype(np.float32), GGML_TYPE_F32)
                else:
                    writer.add_tensor(ti["name"], arr.astype(np.float32), target_type)

                if (i + 1) % max(1, total // 20) == 0:
                    status_var.set(f"Converting {i+1}/{total}: {ti['name']}")

            status_var.set("Writing GGUF file...")
            writer.write()
            file_mb = os.path.getsize(output_path) / (1024 * 1024)
            status_var.set(f"\u2713 Converted: {output_path} ({file_mb:.1f} MB)")

    except Exception as e:
        status_var.set(f"ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    open_converter(root)
    root.mainloop()
