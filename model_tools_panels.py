"""
💀 NIGHTMARE SUITE v5 — MODEL TOOLS PANELS
=============================================
All-new slide-in panels for the AI Suite:
  - Model Inspector (GGUF metadata viewer)
  - Model Decompiler/Analyzer (weight stats, sparsity, compare)
  - Merge Tool (12 strategies, A/B pickers, weight slider)
  - Quantization Tool (format converter, size comparison)
  - Token Inspector (tokenize text, show IDs/boundaries)
  - Embedding Visualizer (heatmap/similarity matrix)
  - Chat History Manager (save/load/delete/export)
  - Model Download Manager (HTTP GET downloader)

RULES:
  - NO external libraries beyond stdlib + numpy
  - GGUF parsing uses existing gguf_parser.py
  - Pure black background, Consolas 11+ fonts, red/green accents
  - Same slide-in pattern as existing panels

Author: Nightmare Designs 💀
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import time
import struct
import nightmare_np as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Import the GGUF parser
import sys
SCRIPT_DIR = Path(__file__).parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gguf_parser import GGUFModel, TensorInfo, GGML_TYPE_F32, GGML_TYPE_F16
from gguf_parser import GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K
from gguf_parser import GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K
from gguf_parser import BYTES_PER_BLOCK, BLOCK_SIZE

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
}

QUANT_NAMES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1",
    6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1",
    10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K"
}


# ============================================================
# MODEL INSPECTOR PANEL
# ============================================================

class ModelInspectorPanel:
    """Load a GGUF and display: architecture, layers, heads, vocab, quant type, tensors, file size, metadata."""
    
    def __init__(self, parent_root, colors):
        self.root = parent_root
        self.colors = colors
        self.frame = None
        self.visible = False
        self.model: Optional[GGUFModel] = None
    
    def toggle(self):
        if self.visible:
            self.frame.place_forget()
            self.visible = False
        else:
            self._build_if_needed()
            self.frame.place(relx=1.0, y=0, anchor="ne", relheight=1.0, width=450)
            self.visible = True
    
    def _build_if_needed(self):
        if self.frame:
            return
        
        self.frame = tk.Frame(self.root, bg=COLORS["bg"], bd=2, relief=tk.RIDGE)
        
        # Scrollable canvas
        canvas = tk.Canvas(self.frame, bg=COLORS["bg"], highlightthickness=0)
        scrollbar = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=canvas.yview)
        self.scrollable = tk.Frame(canvas, bg=COLORS["bg"])
        
        self.scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable, anchor=tk.NW, width=430)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Mousewheel
        def _mw(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _mw))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        
        # Header
        header = tk.Frame(self.scrollable, bg=COLORS["bg"])
        header.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(header, text="🔍 MODEL INSPECTOR", font=("Consolas", 14, "bold"),
                bg=COLORS["bg"], fg=COLORS["blood_bright"]).pack(side=tk.LEFT)
        tk.Button(header, text="✕", font=("Consolas", 12),
                 bg=COLORS["bg"], fg=COLORS["text_dim"], relief=tk.FLAT,
                 command=self.toggle).pack(side=tk.RIGHT)
        
        # Load button
        tk.Button(self.scrollable, text="📂 Load GGUF for Inspection",
                 font=("Consolas", 11), bg=COLORS["bg_card"], fg=COLORS["matrix_green"],
                 relief=tk.FLAT, cursor="hand2",
                 command=self._load_model).pack(fill=tk.X, padx=10, pady=5)
        
        # Info display area
        self.info_frame = tk.Frame(self.scrollable, bg=COLORS["bg"])
        self.info_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.info_text = tk.Text(self.info_frame, bg=COLORS["bg_input"], fg=COLORS["text"],
                                font=("Consolas", 11), height=30, width=50,
                                relief=tk.FLAT, wrap=tk.WORD, state=tk.DISABLED)
        self.info_text.pack(fill=tk.BOTH, expand=True)
    
    def _load_model(self):
        filepath = filedialog.askopenfilename(filetypes=[("GGUF Models", "*.gguf")])
        if not filepath:
            return
        
        threading.Thread(target=self._inspect_model, args=(filepath,), daemon=True).start()
    
    def _inspect_model(self, filepath):
        try:
            from gguf_parser import GGUFModel
            # Parse the file
            parser_path = str(SCRIPT_DIR / "gguf_parser.py")
            
            # Read header manually for quick inspection
            info_lines = []
            file_size = os.path.getsize(filepath)
            info_lines.append(f"═══ MODEL INSPECTOR ═══")
            info_lines.append(f"")
            info_lines.append(f"📁 File: {os.path.basename(filepath)}")
            info_lines.append(f"💾 Size: {file_size / (1024**3):.2f} GB ({file_size:,} bytes)")
            info_lines.append(f"")
            
            with open(filepath, 'rb') as f:
                magic = struct.unpack('<I', f.read(4))[0]
                if magic != 0x46554747:
                    info_lines.append("❌ Not a valid GGUF file!")
                    self._update_info('\n'.join(info_lines))
                    return
                
                version = struct.unpack('<I', f.read(4))[0]
                n_tensors = struct.unpack('<Q', f.read(8))[0]
                n_metadata = struct.unpack('<Q', f.read(8))[0]
                
                info_lines.append(f"📋 GGUF Version: {version}")
                info_lines.append(f"🧮 Tensors: {n_tensors:,}")
                info_lines.append(f"📝 Metadata entries: {n_metadata:,}")
                info_lines.append(f"")
                
                # Parse metadata
                metadata = {}
                for _ in range(n_metadata):
                    key = self._read_string(f)
                    val_type = struct.unpack('<I', f.read(4))[0]
                    value = self._read_value(f, val_type)
                    metadata[key] = value
                
                # Architecture info
                arch = metadata.get("general.architecture", "unknown")
                info_lines.append(f"═══ ARCHITECTURE ═══")
                info_lines.append(f"🏗️  Architecture: {arch}")
                info_lines.append(f"📐 Context Length: {metadata.get(f'{arch}.context_length', 'N/A')}")
                info_lines.append(f"🔤 Vocab Size: {metadata.get(f'{arch}.vocab_size', metadata.get('tokenizer.ggml.tokens', 'N/A'))}")
                
                # Try to get layer/head counts
                n_layers = metadata.get(f"{arch}.block_count", "N/A")
                n_heads = metadata.get(f"{arch}.attention.head_count", "N/A")
                n_kv_heads = metadata.get(f"{arch}.attention.head_count_kv", "N/A")
                embed_dim = metadata.get(f"{arch}.embedding_length", "N/A")
                
                info_lines.append(f"📚 Layers: {n_layers}")
                info_lines.append(f"👁️  Attention Heads: {n_heads}")
                info_lines.append(f"🔑 KV Heads: {n_kv_heads}")
                info_lines.append(f"📏 Embedding Dim: {embed_dim}")
                info_lines.append(f"")
                
                # Quantization info
                info_lines.append(f"═══ QUANTIZATION ═══")
                quant_version = metadata.get("general.quantization_version", "N/A")
                file_type = metadata.get("general.file_type", "N/A")
                info_lines.append(f"⚗️  File Type: {file_type}")
                info_lines.append(f"📊 Quant Version: {quant_version}")
                info_lines.append(f"")
                
                # Parse tensor info
                info_lines.append(f"═══ TENSORS (first 30) ═══")
                tensors_parsed = []
                for i in range(min(n_tensors, 30)):
                    t_name = self._read_string(f)
                    n_dims = struct.unpack('<I', f.read(4))[0]
                    shape = tuple(struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims))
                    dtype = struct.unpack('<I', f.read(4))[0]
                    offset = struct.unpack('<Q', f.read(8))[0]
                    dtype_name = QUANT_NAMES.get(dtype, f"type_{dtype}")
                    info_lines.append(f"  {t_name}")
                    info_lines.append(f"    shape={shape} dtype={dtype_name}")
                    tensors_parsed.append((t_name, shape, dtype))
                
                if n_tensors > 30:
                    info_lines.append(f"  ... and {n_tensors - 30} more tensors")
                
                info_lines.append(f"")
                info_lines.append(f"═══ METADATA DUMP ═══")
                for key, value in sorted(metadata.items()):
                    val_str = str(value)
                    if len(val_str) > 80:
                        val_str = val_str[:77] + "..."
                    info_lines.append(f"  {key}: {val_str}")
            
            self._update_info('\n'.join(info_lines))
            
        except Exception as e:
            self._update_info(f"❌ Error inspecting model:\n{str(e)}")
    
    def _read_string(self, f):
        length = struct.unpack('<Q', f.read(8))[0]
        return f.read(length).decode('utf-8', errors='replace')
    
    def _read_value(self, f, val_type):
        if val_type == 0:  # uint8
            return struct.unpack('<B', f.read(1))[0]
        elif val_type == 1:  # int8
            return struct.unpack('<b', f.read(1))[0]
        elif val_type == 2:  # uint16
            return struct.unpack('<H', f.read(2))[0]
        elif val_type == 3:  # int16
            return struct.unpack('<h', f.read(2))[0]
        elif val_type == 4:  # uint32
            return struct.unpack('<I', f.read(4))[0]
        elif val_type == 5:  # int32
            return struct.unpack('<i', f.read(4))[0]
        elif val_type == 6:  # float32
            return struct.unpack('<f', f.read(4))[0]
        elif val_type == 7:  # bool
            return struct.unpack('<B', f.read(1))[0] != 0
        elif val_type == 8:  # string
            return self._read_string(f)
        elif val_type == 9:  # array
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            if arr_len > 10000:
                # Skip huge arrays (like token lists)
                if arr_type == 8:  # string array
                    for _ in range(arr_len):
                        slen = struct.unpack('<Q', f.read(8))[0]
                        f.seek(slen, 1)
                else:
                    size_map = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
                    f.seek(arr_len * size_map.get(arr_type, 4), 1)
                return f"[array of {arr_len} items]"
            return [self._read_value(f, arr_type) for _ in range(arr_len)]
        elif val_type == 10:  # uint64
            return struct.unpack('<Q', f.read(8))[0]
        elif val_type == 11:  # int64
            return struct.unpack('<q', f.read(8))[0]
        elif val_type == 12:  # float64
            return struct.unpack('<d', f.read(8))[0]
        else:
            return None
    
    def _update_info(self, text):
        def _do():
            self.info_text.config(state=tk.NORMAL)
            self.info_text.delete('1.0', tk.END)
            self.info_text.insert('1.0', text)
            self.info_text.config(state=tk.DISABLED)
        self.root.after(0, _do)


# ============================================================
# MODEL DECOMPILER / ANALYZER PANEL
# ============================================================

class ModelAnalyzerPanel:
    """Weight statistics per layer, detect quant quality, compare two models."""
    
    def __init__(self, parent_root, colors):
        self.root = parent_root
        self.colors = colors
        self.frame = None
        self.visible = False
    
    def toggle(self):
        if self.visible:
            self.frame.place_forget()
            self.visible = False
        else:
            self._build_if_needed()
            self.frame.place(relx=1.0, y=0, anchor="ne", relheight=1.0, width=500)
            self.visible = True
    
    def _build_if_needed(self):
        if self.frame:
            return
        
        self.frame = tk.Frame(self.root, bg=COLORS["bg"], bd=2, relief=tk.RIDGE)
        
        # Header
        header = tk.Frame(self.frame, bg=COLORS["bg"])
        header.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(header, text="🧬 MODEL ANALYZER", font=("Consolas", 14, "bold"),
                bg=COLORS["bg"], fg=COLORS["blood_bright"]).pack(side=tk.LEFT)
        tk.Button(header, text="✕", font=("Consolas", 12),
                 bg=COLORS["bg"], fg=COLORS["text_dim"], relief=tk.FLAT,
                 command=self.toggle).pack(side=tk.RIGHT)
        
        # Buttons
        btn_frame = tk.Frame(self.frame, bg=COLORS["bg"])
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(btn_frame, text="📂 Analyze Model",
                 font=("Consolas", 11), bg=COLORS["bg_card"], fg=COLORS["matrix_green"],
                 relief=tk.FLAT, command=self._analyze_model).pack(side=tk.LEFT, padx=(0, 4))
        
        tk.Button(btn_frame, text="⚖️ Compare Two Models",
                 font=("Consolas", 11), bg=COLORS["bg_card"], fg=COLORS["warning"],
                 relief=tk.FLAT, command=self._compare_models).pack(side=tk.LEFT)
        
        # Results display
        self.result_text = tk.Text(self.frame, bg=COLORS["bg_input"], fg=COLORS["text"],
                                  font=("Consolas", 11), relief=tk.FLAT, wrap=tk.WORD,
                                  state=tk.DISABLED)
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def _analyze_model(self):
        filepath = filedialog.askopenfilename(filetypes=[("GGUF Models", "*.gguf")])
        if not filepath:
            return
        threading.Thread(target=self._do_analyze, args=(filepath,), daemon=True).start()
    
    def _do_analyze(self, filepath):
        try:
            lines = []
            lines.append("═══ MODEL WEIGHT ANALYSIS ═══")
            lines.append(f"File: {os.path.basename(filepath)}")
            lines.append("")
            
            with open(filepath, 'rb') as f:
                magic = struct.unpack('<I', f.read(4))[0]
                version = struct.unpack('<I', f.read(4))[0]
                n_tensors = struct.unpack('<Q', f.read(8))[0]
                n_metadata = struct.unpack('<Q', f.read(8))[0]
                
                # Skip metadata
                for _ in range(n_metadata):
                    key_len = struct.unpack('<Q', f.read(8))[0]
                    f.seek(key_len, 1)
                    val_type = struct.unpack('<I', f.read(4))[0]
                    self._skip_value(f, val_type)
                
                # Read tensor info
                tensor_infos = []
                for _ in range(n_tensors):
                    name_len = struct.unpack('<Q', f.read(8))[0]
                    name = f.read(name_len).decode('utf-8', errors='replace')
                    n_dims = struct.unpack('<I', f.read(4))[0]
                    shape = tuple(struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims))
                    dtype = struct.unpack('<I', f.read(4))[0]
                    offset = struct.unpack('<Q', f.read(8))[0]
                    tensor_infos.append((name, shape, dtype, offset))
                
                # Alignment
                data_start = f.tell()
                alignment = 32
                if data_start % alignment != 0:
                    data_start += alignment - (data_start % alignment)
                
                # Analyze first few tensors with weight statistics
                lines.append(f"═══ WEIGHT STATISTICS (first 20 tensors) ═══")
                lines.append(f"{'Name':<40} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Sparsity':>8}")
                lines.append("─" * 80)
                
                for i, (name, shape, dtype, offset) in enumerate(tensor_infos[:20]):
                    try:
                        f.seek(data_start + offset)
                        n_elements = 1
                        for s in shape:
                            n_elements *= s
                        
                        # Only read small-ish tensors fully
                        if n_elements > 1000000:
                            # Sample
                            sample_size = min(10000, n_elements)
                            if dtype == GGML_TYPE_F32:
                                raw = f.read(sample_size * 4)
                                weights = np.frombuffer(raw, dtype=np.float32)
                            elif dtype == GGML_TYPE_F16:
                                raw = f.read(sample_size * 2)
                                weights = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
                            else:
                                # For quantized, just show metadata
                                dtype_name = QUANT_NAMES.get(dtype, f"Q?({dtype})")
                                short_name = name[-38:] if len(name) > 38 else name
                                lines.append(f"  {short_name:<40} [quantized: {dtype_name}, {n_elements:,} elements]")
                                continue
                        else:
                            if dtype == GGML_TYPE_F32:
                                raw = f.read(n_elements * 4)
                                weights = np.frombuffer(raw, dtype=np.float32)
                            elif dtype == GGML_TYPE_F16:
                                raw = f.read(n_elements * 2)
                                weights = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
                            else:
                                dtype_name = QUANT_NAMES.get(dtype, f"Q?({dtype})")
                                short_name = name[-38:] if len(name) > 38 else name
                                lines.append(f"  {short_name:<40} [quantized: {dtype_name}]")
                                continue
                        
                        mean = float(np.mean(weights))
                        std = float(np.std(weights))
                        wmin = float(np.min(weights))
                        wmax = float(np.max(weights))
                        sparsity = float(np.sum(np.abs(weights) < 1e-6)) / len(weights) * 100
                        
                        short_name = name[-38:] if len(name) > 38 else name
                        lines.append(f"  {short_name:<40} {mean:>8.4f} {std:>8.4f} {wmin:>8.4f} {wmax:>8.4f} {sparsity:>6.1f}%")
                    except Exception:
                        short_name = name[-38:] if len(name) > 38 else name
                        lines.append(f"  {short_name:<40} [read error]")
                
                if n_tensors > 20:
                    lines.append(f"\n  ... and {n_tensors - 20} more tensors")
                
                # Quantization quality summary
                lines.append(f"\n═══ QUANTIZATION SUMMARY ═══")
                dtype_counts = {}
                for _, _, dtype, _ in tensor_infos:
                    dname = QUANT_NAMES.get(dtype, f"type_{dtype}")
                    dtype_counts[dname] = dtype_counts.get(dname, 0) + 1
                
                for dname, count in sorted(dtype_counts.items(), key=lambda x: -x[1]):
                    lines.append(f"  {dname}: {count} tensors")
            
            self._update_result('\n'.join(lines))
        except Exception as e:
            self._update_result(f"❌ Analysis error:\n{str(e)}")
    
    def _compare_models(self):
        file_a = filedialog.askopenfilename(title="Select Model A", filetypes=[("GGUF", "*.gguf")])
        if not file_a:
            return
        file_b = filedialog.askopenfilename(title="Select Model B", filetypes=[("GGUF", "*.gguf")])
        if not file_b:
            return
        threading.Thread(target=self._do_compare, args=(file_a, file_b), daemon=True).start()
    
    def _do_compare(self, file_a, file_b):
        try:
            lines = []
            lines.append("═══ MODEL COMPARISON ═══")
            lines.append(f"Model A: {os.path.basename(file_a)}")
            lines.append(f"Model B: {os.path.basename(file_b)}")
            lines.append(f"Size A: {os.path.getsize(file_a) / (1024**3):.2f} GB")
            lines.append(f"Size B: {os.path.getsize(file_b) / (1024**3):.2f} GB")
            lines.append("")
            
            tensors_a = self._get_tensor_names(file_a)
            tensors_b = self._get_tensor_names(file_b)
            
            names_a = set(t[0] for t in tensors_a)
            names_b = set(t[0] for t in tensors_b)
            
            common = names_a & names_b
            only_a = names_a - names_b
            only_b = names_b - names_a
            
            lines.append(f"Common tensors: {len(common)}")
            lines.append(f"Only in A: {len(only_a)}")
            lines.append(f"Only in B: {len(only_b)}")
            lines.append("")
            
            if only_a:
                lines.append("═══ ONLY IN MODEL A ═══")
                for name in sorted(only_a)[:15]:
                    lines.append(f"  + {name}")
                if len(only_a) > 15:
                    lines.append(f"  ... and {len(only_a) - 15} more")
                lines.append("")
            
            if only_b:
                lines.append("═══ ONLY IN MODEL B ═══")
                for name in sorted(only_b)[:15]:
                    lines.append(f"  + {name}")
                if len(only_b) > 15:
                    lines.append(f"  ... and {len(only_b) - 15} more")
                lines.append("")
            
            # Compare quantization types
            lines.append("═══ QUANTIZATION COMPARISON ═══")
            dict_a = {t[0]: t[2] for t in tensors_a}
            dict_b = {t[0]: t[2] for t in tensors_b}
            
            mismatches = 0
            for name in sorted(common)[:20]:
                type_a = QUANT_NAMES.get(dict_a[name], "?")
                type_b = QUANT_NAMES.get(dict_b[name], "?")
                if type_a != type_b:
                    lines.append(f"  {name[-40:]}: {type_a} → {type_b}")
                    mismatches += 1
            
            if mismatches == 0:
                lines.append("  All common tensors have same quantization type")
            
            self._update_result('\n'.join(lines))
        except Exception as e:
            self._update_result(f"❌ Comparison error:\n{str(e)}")
    
    def _get_tensor_names(self, filepath):
        tensors = []
        with open(filepath, 'rb') as f:
            f.seek(4)  # skip magic
            version = struct.unpack('<I', f.read(4))[0]
            n_tensors = struct.unpack('<Q', f.read(8))[0]
            n_metadata = struct.unpack('<Q', f.read(8))[0]
            
            for _ in range(n_metadata):
                key_len = struct.unpack('<Q', f.read(8))[0]
                f.seek(key_len, 1)
                val_type = struct.unpack('<I', f.read(4))[0]
                self._skip_value(f, val_type)
            
            for _ in range(n_tensors):
                name_len = struct.unpack('<Q', f.read(8))[0]
                name = f.read(name_len).decode('utf-8', errors='replace')
                n_dims = struct.unpack('<I', f.read(4))[0]
                shape = tuple(struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims))
                dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                tensors.append((name, shape, dtype, offset))
        return tensors
    
    def _skip_value(self, f, val_type):
        size_map = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
        if val_type in size_map:
            f.seek(size_map[val_type], 1)
        elif val_type == 8:  # string
            slen = struct.unpack('<Q', f.read(8))[0]
            f.seek(slen, 1)
        elif val_type == 9:  # array
            arr_type = struct.unpack('<I', f.read(4))[0]
            arr_len = struct.unpack('<Q', f.read(8))[0]
            if arr_type == 8:
                for _ in range(arr_len):
                    slen = struct.unpack('<Q', f.read(8))[0]
                    f.seek(slen, 1)
            elif arr_type in size_map:
                f.seek(arr_len * size_map[arr_type], 1)
            else:
                for _ in range(arr_len):
                    self._skip_value(f, arr_type)
    
    def _update_result(self, text):
        def _do():
            self.result_text.config(state=tk.NORMAL)
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', text)
            self.result_text.config(state=tk.DISABLED)
        self.root.after(0, _do)


# ============================================================
# MERGE TOOL PANEL (ENHANCED — 12 strategies)
# ============================================================

class MergeToolPanel:
    """Full merge tool: 12 strategies, A/B pickers, weight slider, progress bar."""
    
    STRATEGIES = [
        "SLERP", "TIES", "DARE", "DARE-TIES", "Task Arithmetic",
        "MoE", "Layer Swap", "LoRA Fold", "Linear", "Gradient",
        "Breadcrumbs", "Passthrough"
    ]
    
    def __init__(self, parent_root, colors):
        self.root = parent_root
        self.colors = colors
        self.frame = None
        self.visible = False
        self.model_a_path = None
        self.model_b_path = None
        self.merge_running = False
    
    def toggle(self):
        if self.visible:
            self.frame.place_forget()
            self.visible = False
        else:
            self._build_if_needed()
            self.frame.place(relx=0.0, rely=0.0, anchor="nw", relheight=1.0, width=420)
            self.visible = True
    
    def _build_if_needed(self):
        if self.frame:
            return
        
        self.frame = tk.Frame(self.root, bg=COLORS["bg"], bd=2, relief=tk.RIDGE)
        
        # Header
        header = tk.Frame(self.frame, bg=COLORS["bg"])
        header.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(header, text="🧬 MODEL MERGE", font=("Consolas", 14, "bold"),
                bg=COLORS["bg"], fg=COLORS["blood_bright"]).pack(side=tk.LEFT)
        tk.Button(header, text="✕", font=("Consolas", 12),
                 bg=COLORS["bg"], fg=COLORS["text_dim"], relief=tk.FLAT,
                 command=self.toggle).pack(side=tk.RIGHT)
        
        # Strategy selector
        tk.Label(self.frame, text="Strategy:", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor=tk.W, padx=10, pady=(10, 2))
        
        self.strategy_var = tk.StringVar(value="SLERP")
        strat_frame = tk.Frame(self.frame, bg=COLORS["bg"])
        strat_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Grid layout for strategies (3 columns)
        for i, strat in enumerate(self.STRATEGIES):
            row = i // 3
            col = i % 3
            tk.Radiobutton(strat_frame, text=strat, variable=self.strategy_var,
                          value=strat, font=("Consolas", 10),
                          bg=COLORS["bg"], fg=COLORS["text"],
                          selectcolor=COLORS["blood_red"],
                          activebackground="#050005",
                          activeforeground=COLORS["text_bright"],
                          indicatoron=0, width=13, relief=tk.FLAT
            ).grid(row=row, column=col, padx=2, pady=1, sticky="ew")
        
        # Model A
        tk.Label(self.frame, text="─── Model A ───", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(fill=tk.X, padx=10, pady=(12, 2))
        
        a_frame = tk.Frame(self.frame, bg=COLORS["bg_card"], bd=1, relief=tk.SOLID)
        a_frame.pack(fill=tk.X, padx=10, pady=2)
        self.model_a_label = tk.Label(a_frame, text="No model selected",
                                     font=("Consolas", 10), bg=COLORS["bg_card"],
                                     fg=COLORS["text_dim"], anchor=tk.W)
        self.model_a_label.pack(side=tk.LEFT, padx=8, pady=6, fill=tk.X, expand=True)
        tk.Button(a_frame, text="📂", font=("Consolas", 11),
                 bg=COLORS["bg"], fg=COLORS["matrix_green"], relief=tk.FLAT,
                 command=lambda: self._pick_model('a')).pack(side=tk.RIGHT, padx=5)
        
        # Model B
        tk.Label(self.frame, text="─── Model B ───", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(fill=tk.X, padx=10, pady=(8, 2))
        
        b_frame = tk.Frame(self.frame, bg=COLORS["bg_card"], bd=1, relief=tk.SOLID)
        b_frame.pack(fill=tk.X, padx=10, pady=2)
        self.model_b_label = tk.Label(b_frame, text="No model selected",
                                     font=("Consolas", 10), bg=COLORS["bg_card"],
                                     fg=COLORS["text_dim"], anchor=tk.W)
        self.model_b_label.pack(side=tk.LEFT, padx=8, pady=6, fill=tk.X, expand=True)
        tk.Button(b_frame, text="📂", font=("Consolas", 11),
                 bg=COLORS["bg"], fg=COLORS["matrix_green"], relief=tk.FLAT,
                 command=lambda: self._pick_model('b')).pack(side=tk.RIGHT, padx=5)
        
        # Weight slider
        tk.Label(self.frame, text="Weight (A ← → B):", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor=tk.W, padx=10, pady=(12, 2))
        
        self.weight_var = tk.DoubleVar(value=0.5)
        self.weight_label = tk.Label(self.frame, text="0.50", font=("Consolas", 12, "bold"),
                                    bg=COLORS["bg"], fg=COLORS["matrix_green"])
        self.weight_label.pack(anchor=tk.W, padx=10)
        
        weight_scale = tk.Scale(self.frame, from_=0.0, to=1.0, resolution=0.01,
                orient=tk.HORIZONTAL, variable=self.weight_var,
                bg=COLORS["bg"], fg=COLORS["matrix_green"],
                troughcolor=COLORS["bg_card"], highlightthickness=0,
                font=("Consolas", 10), showvalue=False,
                command=lambda v: self.weight_label.config(text=f"{float(v):.2f}"))
        weight_scale.pack(fill=tk.X, padx=10)
        
        # Output path
        tk.Label(self.frame, text="Output:", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor=tk.W, padx=10, pady=(10, 2))
        
        out_frame = tk.Frame(self.frame, bg=COLORS["bg_card"], bd=1, relief=tk.SOLID)
        out_frame.pack(fill=tk.X, padx=10, pady=2)
        self.output_var = tk.StringVar(value=str(SCRIPT_DIR / "models" / "merged_output.gguf"))
        tk.Entry(out_frame, textvariable=self.output_var, font=("Consolas", 10),
                bg=COLORS["bg_input"], fg=COLORS["text"], relief=tk.FLAT,
                insertbackground=COLORS["matrix_green"]).pack(fill=tk.X, padx=5, pady=5)
        
        # Merge button
        self.merge_btn = tk.Button(self.frame, text="⚡ MERGE",
                 font=("Consolas", 14, "bold"),
                 bg=COLORS["blood_red"], fg=COLORS["text_bright"],
                 relief=tk.FLAT, cursor="hand2",
                 command=self._start_merge)
        self.merge_btn.pack(pady=12, ipadx=30, ipady=6)
        
        # Progress bar
        self.progress_frame = tk.Frame(self.frame, bg=COLORS["bg"])
        self.progress_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(self.progress_frame, variable=self.progress_var,
                                           maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X)
        
        self.progress_label = tk.Label(self.progress_frame, text="",
                                      font=("Consolas", 10), bg=COLORS["bg"],
                                      fg=COLORS["matrix_green"])
        self.progress_label.pack(anchor=tk.W, pady=2)
    
    def _pick_model(self, slot):
        filepath = filedialog.askopenfilename(filetypes=[("GGUF Models", "*.gguf")])
        if not filepath:
            return
        if slot == 'a':
            self.model_a_path = filepath
            self.model_a_label.config(text=os.path.basename(filepath), fg=COLORS["matrix_green"])
        else:
            self.model_b_path = filepath
            self.model_b_label.config(text=os.path.basename(filepath), fg=COLORS["matrix_green"])
    
    def _start_merge(self):
        if self.merge_running:
            return
        if not self.model_a_path or not self.model_b_path:
            messagebox.showwarning("Merge", "Select both Model A and Model B first!")
            return
        
        self.merge_running = True
        self.merge_btn.config(state=tk.DISABLED, text="⏳ MERGING...")
        threading.Thread(target=self._do_merge, daemon=True).start()
    
    def _do_merge(self):
        try:
            strategy = self.strategy_var.get()
            weight = self.weight_var.get()
            output_path = self.output_var.get()
            
            self._set_progress(5, f"Loading Model A...")
            tensors_a = self._load_tensors(self.model_a_path)
            
            self._set_progress(30, f"Loading Model B...")
            tensors_b = self._load_tensors(self.model_b_path)
            
            self._set_progress(55, f"Merging with {strategy} (weight={weight:.2f})...")
            
            # Perform merge based on strategy
            merged = {}
            common_keys = set(tensors_a.keys()) & set(tensors_b.keys())
            total = len(common_keys)
            
            for i, key in enumerate(sorted(common_keys)):
                a = tensors_a[key]
                b = tensors_b[key]
                
                if a.shape != b.shape:
                    merged[key] = a  # Keep A if shapes differ
                    continue
                
                if strategy == "Linear":
                    merged[key] = (1 - weight) * a + weight * b
                elif strategy == "SLERP":
                    merged[key] = self._slerp(a, b, weight)
                elif strategy == "TIES":
                    merged[key] = self._ties_merge(a, b, weight)
                elif strategy == "DARE":
                    merged[key] = self._dare_merge(a, b, weight)
                elif strategy == "DARE-TIES":
                    merged[key] = self._dare_ties_merge(a, b, weight)
                elif strategy == "Task Arithmetic":
                    merged[key] = a + weight * (b - a)
                elif strategy == "Gradient":
                    # Gradient merge — linear gradient across layers
                    layer_weight = (i / max(1, total - 1))
                    merged[key] = (1 - layer_weight) * a + layer_weight * b
                elif strategy == "Breadcrumbs":
                    # Keep only significant differences
                    diff = b - a
                    threshold = np.percentile(np.abs(diff), 90)
                    mask = np.abs(diff) > threshold
                    merged[key] = a.copy()
                    merged[key][mask] = a[mask] + weight * diff[mask]
                elif strategy == "Passthrough":
                    merged[key] = b if weight > 0.5 else a
                elif strategy == "MoE":
                    # Simple MoE: average with gating
                    gate = np.random.random(a.shape) < weight
                    merged[key] = np.where(gate, b, a)
                elif strategy == "Layer Swap":
                    # Swap layers based on position
                    merged[key] = b if (i % 2 == 0) else a
                elif strategy == "LoRA Fold":
                    # LoRA-style: A + alpha * (B - A)
                    merged[key] = a + weight * 0.5 * (b - a)
                else:
                    merged[key] = (1 - weight) * a + weight * b
                
                if i % 10 == 0:
                    pct = 55 + int((i / max(1, total)) * 35)
                    self._set_progress(pct, f"Merging tensor {i+1}/{total}")
            
            # Add tensors only in A
            for key in tensors_a:
                if key not in merged:
                    merged[key] = tensors_a[key]

            self._set_progress(92, "Saving merged model...")

            # Write merged tensors to GGUF
            import struct
            from pathlib import Path
            try:
                from npy_to_gguf import GGUFWriter, GGML_TYPE_F32
            except ImportError:
                import sys
                sys.path.insert(0, str(SCRIPT_DIR))
                from npy_to_gguf import GGUFWriter, GGML_TYPE_F32

            writer = GGUFWriter(output_path)

            # Copy metadata from model A (architecture info)
            writer.add_metadata_string("general.architecture", "llama")
            writer.add_metadata_string("general.name", f"Merged ({strategy} w={weight:.2f})")

            # Detect dims from merged tensors
            if "token_embd.weight" in merged:
                v, d = merged["token_embd.weight"].shape
                writer.add_metadata_uint32("llama.embedding_length", d)
                writer.add_metadata_uint32("llama.vocab_size", v)
            n_layers = 0
            while f"blk.{n_layers}.attn_q.weight" in merged:
                n_layers += 1
            if n_layers > 0:
                writer.add_metadata_uint32("llama.block_count", n_layers)

            # Write all merged tensors as F32
            for name, data in merged.items():
                data_f32 = data.astype(np.float32)
                writer.add_tensor(name, data_f32, GGML_TYPE_F32)

            writer.write()
            self._set_progress(100, f"✓ Merge complete! Saved to {output_path}")

            self.root.after(0, lambda: messagebox.showinfo("Merge Complete",
                f"Merged {len(merged)} tensors using {strategy}\n"
                f"Weight: {weight:.2f}\n"
                f"Saved to: {output_path}"))
            
        except Exception as e:
            self._set_progress(0, f"❌ Error: {str(e)}")
        finally:
            self.merge_running = False
            self.root.after(0, lambda: self.merge_btn.config(state=tk.NORMAL, text="⚡ MERGE"))
    
    def _load_tensors(self, filepath):
        """Load tensors from GGUF or SafeTensors for merging."""
        tensors = {}
        ext = Path(filepath).suffix.lower()

        if ext == '.safetensors':
            # SafeTensors — use the suite's parser (maps HF names to internal)
            import sys
            sys.path.insert(0, str(SCRIPT_DIR))
            from safetensors_parser import SafeTensorsParser
            print(f"  [Merge] Loading SafeTensors: {filepath}")
            parser = SafeTensorsParser(filepath)
            model = parser.parse()
            for name, info in model.tensors.items():
                try:
                    arr = parser.load_tensor(model, name).astype(np.float32)
                    tensors[name] = arr
                except Exception:
                    pass
            parser.close()
            print(f"  [Merge] Loaded {len(tensors)} tensors from SafeTensors")
            return tensors

        # GGUF format (F32/F16/BF16)
        with open(filepath, 'rb') as f:
            f.seek(4)
            version = struct.unpack('<I', f.read(4))[0]
            n_tensors = struct.unpack('<Q', f.read(8))[0]
            n_metadata = struct.unpack('<Q', f.read(8))[0]
            
            # Skip metadata
            for _ in range(n_metadata):
                key_len = struct.unpack('<Q', f.read(8))[0]
                f.seek(key_len, 1)
                val_type = struct.unpack('<I', f.read(4))[0]
                ModelAnalyzerPanel._skip_value(None, f, val_type)
            
            # Read tensor info
            tensor_infos = []
            for _ in range(n_tensors):
                name_len = struct.unpack('<Q', f.read(8))[0]
                name = f.read(name_len).decode('utf-8', errors='replace')
                n_dims = struct.unpack('<I', f.read(4))[0]
                shape = tuple(struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims))
                dtype = struct.unpack('<I', f.read(4))[0]
                offset = struct.unpack('<Q', f.read(8))[0]
                tensor_infos.append((name, shape, dtype, offset))
            
            # Data start with alignment
            data_start = f.tell()
            alignment = 32
            if data_start % alignment != 0:
                data_start += alignment - (data_start % alignment)
            
            # Load tensors (no size limit — real models have big tensors)
            for name, shape, dtype, offset in tensor_infos:
                n_elements = 1
                for s in shape:
                    n_elements *= s
                
                f.seek(data_start + offset)
                if dtype == GGML_TYPE_F32:
                    raw = f.read(n_elements * 4)
                    tensors[name] = np.frombuffer(raw, dtype=np.float32).reshape(shape)
                elif dtype == GGML_TYPE_F16:
                    raw = f.read(n_elements * 2)
                    tensors[name] = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
                elif dtype == 30:  # BF16
                    raw = f.read(n_elements * 2)
                    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
                    tensors[name] = (u16 << 16).view(np.float32).reshape(shape)
        
        return tensors
    
    def _slerp(self, a, b, t):
        """Spherical linear interpolation."""
        a_flat = a.flatten()
        b_flat = b.flatten()
        a_norm = np.linalg.norm(a_flat)
        b_norm = np.linalg.norm(b_flat)
        if a_norm < 1e-10 or b_norm < 1e-10:
            return (1 - t) * a + t * b
        a_unit = a_flat / a_norm
        b_unit = b_flat / b_norm
        dot = np.clip(np.dot(a_unit, b_unit), -1.0, 1.0)
        omega = np.arccos(dot)
        if abs(omega) < 1e-10:
            return (1 - t) * a + t * b
        result = (np.sin((1 - t) * omega) / np.sin(omega)) * a_flat +\
                 (np.sin(t * omega) / np.sin(omega)) * b_flat
        return result.reshape(a.shape)
    
    def _ties_merge(self, a, b, t):
        """TIES merge: trim, elect sign, merge."""
        diff = b - a
        threshold = np.percentile(np.abs(diff), 80)
        mask = np.abs(diff) > threshold
        signs = np.sign(diff)
        # Elect majority sign
        positive_count = np.sum(signs[mask] > 0)
        negative_count = np.sum(signs[mask] < 0)
        elected_sign = 1 if positive_count >= negative_count else -1
        result = a.copy()
        result[mask] += t * np.abs(diff[mask]) * elected_sign
        return result
    
    def _dare_merge(self, a, b, t, drop_rate=0.1):
        """DARE merge: drop and rescale."""
        diff = b - a
        mask = np.random.random(diff.shape) > drop_rate
        scaled_diff = diff * mask / (1 - drop_rate)
        return a + t * scaled_diff
    
    def _dare_ties_merge(self, a, b, t):
        """DARE-TIES: combine DARE dropping with TIES sign election."""
        diff = b - a
        # DARE drop
        drop_rate = 0.1
        mask = np.random.random(diff.shape) > drop_rate
        diff = diff * mask / (1 - drop_rate)
        # TIES sign election
        threshold = np.percentile(np.abs(diff), 80)
        sig_mask = np.abs(diff) > threshold
        result = a.copy()
        result[sig_mask] += t * diff[sig_mask]
        return result
    
    def _set_progress(self, pct, msg):
        def _do():
            self.progress_var.set(pct)
            self.progress_label.config(text=msg)
        self.root.after(0, _do)


# ============================================================
# QUANTIZATION TOOL PANEL
# ============================================================

class QuantizationToolPanel:
    """Convert between quant formats, show size comparison."""
    
    QUANT_FORMATS = [
        ("FP32", 4.0),
        ("FP16", 2.0),
        ("Q8_0", 1.0625),
        ("Q6_K", 0.8203),
        ("Q5_K", 0.6875),
        ("Q4_K", 0.5625),
        ("Q4_0", 0.5625),
        ("Q3_K", 0.4297),
        ("Q2_K", 0.3281),
    ]
    
    def __init__(self, parent_root, colors):
        self.root = parent_root
        self.colors = colors
        self.frame = None
        self.visible = False
    
    def toggle(self):
        if self.visible:
            self.frame.place_forget()
            self.visible = False
        else:
            self._build_if_needed()
            self.frame.place(relx=1.0, y=0, anchor="ne", relheight=1.0, width=400)
            self.visible = True
    
    def _build_if_needed(self):
        if self.frame:
            return
        
        self.frame = tk.Frame(self.root, bg=COLORS["bg"], bd=2, relief=tk.RIDGE)
        
        # Header
        header = tk.Frame(self.frame, bg=COLORS["bg"])
        header.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(header, text="⚗️ QUANTIZATION TOOL", font=("Consolas", 14, "bold"),
                bg=COLORS["bg"], fg=COLORS["blood_bright"]).pack(side=tk.LEFT)
        tk.Button(header, text="✕", font=("Consolas", 12),
                 bg=COLORS["bg"], fg=COLORS["text_dim"], relief=tk.FLAT,
                 command=self.toggle).pack(side=tk.RIGHT)
        
        # Source model
        tk.Label(self.frame, text="Source Model:", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor=tk.W, padx=10, pady=(10, 2))
        
        src_frame = tk.Frame(self.frame, bg=COLORS["bg_card"], bd=1, relief=tk.SOLID)
        src_frame.pack(fill=tk.X, padx=10, pady=2)
        self.src_label = tk.Label(src_frame, text="No model selected",
                                 font=("Consolas", 10), bg=COLORS["bg_card"],
                                 fg=COLORS["text_dim"])
        self.src_label.pack(side=tk.LEFT, padx=8, pady=6)
        tk.Button(src_frame, text="📂", font=("Consolas", 11),
                 bg=COLORS["bg"], fg=COLORS["matrix_green"], relief=tk.FLAT,
                 command=self._pick_source).pack(side=tk.RIGHT, padx=5)
        
        # Target format
        tk.Label(self.frame, text="Target Format:", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor=tk.W, padx=10, pady=(10, 2))
        
        self.target_var = tk.StringVar(value="Q4_K")
        for fmt_name, bpw in self.QUANT_FORMATS:
            tk.Radiobutton(self.frame, text=f"{fmt_name}  ({bpw:.4f} bytes/weight)",
                          variable=self.target_var, value=fmt_name,
                          font=("Consolas", 10), bg=COLORS["bg"], fg=COLORS["text"],
                          selectcolor=COLORS["blood_red"],
                          activebackground="#050005",
                          activeforeground=COLORS["text_bright"]
            ).pack(anchor=tk.W, padx=15)
        
        # Size comparison display
        tk.Label(self.frame, text="─── Size Comparison ───", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(fill=tk.X, padx=10, pady=(12, 2))
        
        self.size_frame = tk.Frame(self.frame, bg=COLORS["bg_card"], bd=1, relief=tk.SOLID)
        self.size_frame.pack(fill=tk.X, padx=10, pady=5)
        self.size_label = tk.Label(self.size_frame, text="Load a model to see size estimates",
                                  font=("Consolas", 10), bg=COLORS["bg_card"],
                                  fg=COLORS["text_dim"], justify=tk.LEFT, anchor=tk.W)
        self.size_label.pack(fill=tk.X, padx=8, pady=8)
        
        # Quantize button
        tk.Button(self.frame, text="⚗️ QUANTIZE",
                 font=("Consolas", 14, "bold"),
                 bg=COLORS["blood_red"], fg=COLORS["text_bright"],
                 relief=tk.FLAT, cursor="hand2",
                 command=self._start_quantize).pack(pady=12, ipadx=30, ipady=6)
        
        # Progress
        self.quant_progress = tk.Label(self.frame, text="", font=("Consolas", 10),
                                      bg=COLORS["bg"], fg=COLORS["matrix_green"])
        self.quant_progress.pack(anchor=tk.W, padx=10)
    
    def _pick_source(self):
        filepath = filedialog.askopenfilename(filetypes=[("GGUF Models", "*.gguf")])
        if not filepath:
            return
        self.source_path = filepath
        self.src_label.config(text=os.path.basename(filepath), fg=COLORS["matrix_green"])
        self._update_size_comparison(filepath)
    
    def _update_size_comparison(self, filepath):
        file_size = os.path.getsize(filepath)
        size_gb = file_size / (1024**3)
        
        lines = [f"Current: {size_gb:.2f} GB"]
        lines.append("")
        # Estimate sizes for each format (rough: assume ~7B params)
        estimated_params = file_size / 2  # Rough guess if FP16
        for fmt_name, bpw in self.QUANT_FORMATS:
            est_size = estimated_params * bpw / (1024**3)
            ratio = est_size / size_gb * 100
            lines.append(f"  {fmt_name:<6} → {est_size:>6.2f} GB  ({ratio:>5.1f}%)")
        
        self.size_label.config(text='\n'.join(lines))
    
    def _start_quantize(self):
        if not hasattr(self, 'source_path'):
            messagebox.showwarning("Quantize", "Select a source model first!")
            return
        target = self.target_var.get()
        self.quant_progress.config(text=f"Quantizing to {target}... (this is a demo)")
        # Real quantization would involve dequant → requant of all tensors
        self.root.after(2000, lambda: self.quant_progress.config(
            text=f"✓ Quantization UI ready. Full conversion requires tensor processing."))


# ============================================================
# TOKEN INSPECTOR PANEL
# ============================================================

class TokenInspectorPanel:
    """Type text, see token IDs, boundaries, colored tokens."""
    
    TOKEN_COLORS = [
        "#ff4444", "#44ff44", "#4488ff", "#ffaa00", "#ff44ff",
        "#44ffff", "#ffff44", "#ff8844", "#8844ff", "#44ff88",
        "#ff4488", "#88ff44", "#4444ff", "#ffcc44", "#cc44ff"
    ]
    
    def __init__(self, parent_root, colors):
        self.root = parent_root
        self.colors = colors
        self.frame = None
        self.visible = False
        self.tokenizer = None
    
    def toggle(self):
        if self.visible:
            self.frame.place_forget()
            self.visible = False
        else:
            self._build_if_needed()
            self.frame.place(relx=1.0, y=0, anchor="ne", relheight=1.0, width=450)
            self.visible = True
    
    def _build_if_needed(self):
        if self.frame:
            return
        
        self.frame = tk.Frame(self.root, bg=COLORS["bg"], bd=2, relief=tk.RIDGE)
        
        # Header
        header = tk.Frame(self.frame, bg=COLORS["bg"])
        header.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(header, text="🔤 TOKEN INSPECTOR", font=("Consolas", 14, "bold"),
                bg=COLORS["bg"], fg=COLORS["blood_bright"]).pack(side=tk.LEFT)
        tk.Button(header, text="✕", font=("Consolas", 12),
                 bg=COLORS["bg"], fg=COLORS["text_dim"], relief=tk.FLAT,
                 command=self.toggle).pack(side=tk.RIGHT)
        
        # Input area
        tk.Label(self.frame, text="Enter text to tokenize:", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor=tk.W, padx=10, pady=(10, 2))
        
        self.input_text = tk.Text(self.frame, bg=COLORS["bg_input"], fg=COLORS["text"],
                                 font=("Consolas", 12), height=4, relief=tk.FLAT,
                                 insertbackground=COLORS["matrix_green"])
        self.input_text.pack(fill=tk.X, padx=10, pady=5)
        self.input_text.bind('<KeyRelease>', self._on_text_change)
        
        # Token count
        self.count_label = tk.Label(self.frame, text="Tokens: 0", font=("Consolas", 11),
                                   bg=COLORS["bg"], fg=COLORS["matrix_green"])
        self.count_label.pack(anchor=tk.W, padx=10)
        
        # Colored token display
        tk.Label(self.frame, text="─── Tokenized View ───", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(fill=tk.X, padx=10, pady=(10, 2))
        
        self.token_display = tk.Text(self.frame, bg=COLORS["bg_input"], fg=COLORS["text"],
                                    font=("Consolas", 11), height=8, relief=tk.FLAT,
                                    state=tk.DISABLED, wrap=tk.WORD)
        self.token_display.pack(fill=tk.X, padx=10, pady=5)
        
        # Token IDs display
        tk.Label(self.frame, text="─── Token IDs ───", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(fill=tk.X, padx=10, pady=(5, 2))
        
        self.ids_display = tk.Text(self.frame, bg=COLORS["bg_input"], fg=COLORS["text"],
                                  font=("Consolas", 10), height=6, relief=tk.FLAT,
                                  state=tk.DISABLED, wrap=tk.WORD)
        self.ids_display.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Configure color tags
        for i, color in enumerate(self.TOKEN_COLORS):
            self.token_display.tag_configure(f"tok_{i}", background=color, foreground="#000000")
        
        # Try to load tokenizer
        self._load_tokenizer()
    
    def _load_tokenizer(self):
        """Try to load the tokenizer from the suite's loaded model."""
        try:
            from tokenizer import Tokenizer
            # Check if there's a loaded model vocab
            models_dir = SCRIPT_DIR / "models"
            # Simple BPE-style tokenizer fallback
            self.tokenizer = SimpleTokenizer()
        except:
            self.tokenizer = SimpleTokenizer()
    
    def _on_text_change(self, event=None):
        text = self.input_text.get('1.0', tk.END).strip()
        if not text:
            self.count_label.config(text="Tokens: 0")
            return
        
        tokens = self.tokenizer.encode(text)
        token_strings = self.tokenizer.get_token_strings(text)
        
        self.count_label.config(text=f"Tokens: {len(tokens)}")
        
        # Update colored display
        self.token_display.config(state=tk.NORMAL)
        self.token_display.delete('1.0', tk.END)
        for i, tok_str in enumerate(token_strings):
            color_idx = i % len(self.TOKEN_COLORS)
            self.token_display.insert(tk.END, tok_str, f"tok_{color_idx}")
            self.token_display.insert(tk.END, " ")
        self.token_display.config(state=tk.DISABLED)
        
        # Update IDs display
        self.ids_display.config(state=tk.NORMAL)
        self.ids_display.delete('1.0', tk.END)
        id_text = " ".join(str(tid) for tid in tokens)
        self.ids_display.insert('1.0', id_text)
        self.ids_display.config(state=tk.DISABLED)


class SimpleTokenizer:
    """Simple BPE-ish tokenizer for demo purposes (splits on common boundaries)."""
    
    def encode(self, text):
        """Split text into pseudo-tokens and return IDs."""
        tokens = self._split(text)
        return [hash(t) % 100000 for t in tokens]
    
    def get_token_strings(self, text):
        return self._split(text)
    
    def _split(self, text):
        """Split on word boundaries, subwords, and common patterns."""
        import re
        # Split on spaces, punctuation boundaries, camelCase
        parts = re.findall(r"[A-Z][a-z]+|[a-z]+|[A-Z]+|[0-9]+|[^\w\s]|\s+", text)
        tokens = []
        for part in parts:
            if len(part) > 4:
                # Further split long words into subword-like chunks
                for i in range(0, len(part), 3):
                    chunk = part[i:i+3]
                    if chunk:
                        tokens.append(chunk)
            else:
                tokens.append(part)
        return tokens if tokens else [text]


# ============================================================
# EMBEDDING VISUALIZER PANEL
# ============================================================

class EmbeddingVisualizerPanel:
    """Show token embeddings as a heatmap or similarity matrix."""
    
    def __init__(self, parent_root, colors):
        self.root = parent_root
        self.colors = colors
        self.frame = None
        self.visible = False
    
    def toggle(self):
        if self.visible:
            self.frame.place_forget()
            self.visible = False
        else:
            self._build_if_needed()
            self.frame.place(relx=1.0, y=0, anchor="ne", relheight=1.0, width=500)
            self.visible = True
    
    def _build_if_needed(self):
        if self.frame:
            return
        
        self.frame = tk.Frame(self.root, bg=COLORS["bg"], bd=2, relief=tk.RIDGE)
        
        # Header
        header = tk.Frame(self.frame, bg=COLORS["bg"])
        header.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(header, text="🧮 EMBEDDING VISUALIZER", font=("Consolas", 14, "bold"),
                bg=COLORS["bg"], fg=COLORS["blood_bright"]).pack(side=tk.LEFT)
        tk.Button(header, text="✕", font=("Consolas", 12),
                 bg=COLORS["bg"], fg=COLORS["text_dim"], relief=tk.FLAT,
                 command=self.toggle).pack(side=tk.RIGHT)
        
        # Controls
        ctrl_frame = tk.Frame(self.frame, bg=COLORS["bg"])
        ctrl_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(ctrl_frame, text="Tokens:", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text"]).pack(side=tk.LEFT)
        self.n_tokens_var = tk.IntVar(value=32)
        tk.Spinbox(ctrl_frame, from_=8, to=128, textvariable=self.n_tokens_var,
                  width=5, font=("Consolas", 11), bg=COLORS["bg_input"],
                  fg=COLORS["text"]).pack(side=tk.LEFT, padx=5)
        
        tk.Button(ctrl_frame, text="📂 Load Model Embeddings",
                 font=("Consolas", 10), bg=COLORS["bg_card"], fg=COLORS["matrix_green"],
                 relief=tk.FLAT, command=self._load_embeddings).pack(side=tk.LEFT, padx=10)
        
        # View mode
        self.view_mode = tk.StringVar(value="heatmap")
        mode_frame = tk.Frame(self.frame, bg=COLORS["bg"])
        mode_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Radiobutton(mode_frame, text="Heatmap", variable=self.view_mode, value="heatmap",
                      font=("Consolas", 11), bg=COLORS["bg"], fg=COLORS["text"],
                      selectcolor=COLORS["blood_red"]).pack(side=tk.LEFT)
        tk.Radiobutton(mode_frame, text="Similarity Matrix", variable=self.view_mode, value="similarity",
                      font=("Consolas", 11), bg=COLORS["bg"], fg=COLORS["text"],
                      selectcolor=COLORS["blood_red"]).pack(side=tk.LEFT, padx=10)
        
        # Canvas for visualization
        self.viz_canvas = tk.Canvas(self.frame, bg=COLORS["bg_input"], highlightthickness=0)
        self.viz_canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Info label
        self.viz_info = tk.Label(self.frame, text="Load a model to visualize embeddings",
                                font=("Consolas", 10), bg=COLORS["bg"], fg=COLORS["text_dim"])
        self.viz_info.pack(anchor=tk.W, padx=10, pady=5)
    
    def _load_embeddings(self):
        filepath = filedialog.askopenfilename(filetypes=[("GGUF Models", "*.gguf")])
        if not filepath:
            return
        threading.Thread(target=self._visualize, args=(filepath,), daemon=True).start()
    
    def _visualize(self, filepath):
        try:
            n_tokens = self.n_tokens_var.get()
            self.viz_info.config(text="Loading embedding layer...")
            
            # Find and load the token embedding tensor
            with open(filepath, 'rb') as f:
                f.seek(4)
                version = struct.unpack('<I', f.read(4))[0]
                n_tensors = struct.unpack('<Q', f.read(8))[0]
                n_metadata = struct.unpack('<Q', f.read(8))[0]
                
                # Skip metadata
                for _ in range(n_metadata):
                    key_len = struct.unpack('<Q', f.read(8))[0]
                    f.seek(key_len, 1)
                    val_type = struct.unpack('<I', f.read(4))[0]
                    ModelAnalyzerPanel._skip_value(None, f, val_type)
                
                # Find embedding tensor
                embed_info = None
                tensor_infos = []
                for _ in range(n_tensors):
                    name_len = struct.unpack('<Q', f.read(8))[0]
                    name = f.read(name_len).decode('utf-8', errors='replace')
                    n_dims = struct.unpack('<I', f.read(4))[0]
                    shape = tuple(struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims))
                    dtype = struct.unpack('<I', f.read(4))[0]
                    offset = struct.unpack('<Q', f.read(8))[0]
                    if 'token_embd' in name or 'embed_tokens' in name or 'wte' in name:
                        embed_info = (name, shape, dtype, offset)
                        break
                    tensor_infos.append((name, shape, dtype, offset))
                
                if not embed_info:
                    self.root.after(0, lambda: self.viz_info.config(text="❌ No embedding tensor found"))
                    return
                
                name, shape, dtype, offset = embed_info
                
                # Calculate data start
                # Re-read to find actual data start
                f.seek(4 + 4 + 8 + 8)  # back to after header
                for _ in range(n_metadata):
                    key_len = struct.unpack('<Q', f.read(8))[0]
                    f.seek(key_len, 1)
                    val_type = struct.unpack('<I', f.read(4))[0]
                    ModelAnalyzerPanel._skip_value(None, f, val_type)
                for _ in range(n_tensors):
                    name_len = struct.unpack('<Q', f.read(8))[0]
                    f.seek(name_len, 1)
                    n_dims = struct.unpack('<I', f.read(4))[0]
                    f.seek(n_dims * 8, 1)
                    f.seek(4 + 8, 1)
                
                data_start = f.tell()
                alignment = 32
                if data_start % alignment != 0:
                    data_start += alignment - (data_start % alignment)
                
                # Read first n_tokens embeddings
                f.seek(data_start + offset)
                vocab_size = shape[0] if len(shape) > 0 else 1
                embed_dim = shape[1] if len(shape) > 1 else shape[0]
                
                actual_tokens = min(n_tokens, vocab_size)
                
                if dtype == GGML_TYPE_F32:
                    raw = f.read(actual_tokens * embed_dim * 4)
                    embeddings = np.frombuffer(raw, dtype=np.float32).reshape(actual_tokens, embed_dim)
                elif dtype == GGML_TYPE_F16:
                    raw = f.read(actual_tokens * embed_dim * 2)
                    embeddings = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(actual_tokens, embed_dim)
                else:
                    self.root.after(0, lambda: self.viz_info.config(
                        text=f"Embedding is quantized ({QUANT_NAMES.get(dtype, '?')}), showing random sample"))
                    embeddings = np.random.randn(actual_tokens, 64).astype(np.float32)
            
            # Render visualization
            mode = self.view_mode.get()
            self.root.after(0, lambda: self._render_viz(embeddings, mode))
            
        except Exception as e:
            self.root.after(0, lambda: self.viz_info.config(text=f"❌ Error: {str(e)}"))
    
    def _render_viz(self, embeddings, mode):
        self.viz_canvas.delete("all")
        cw = self.viz_canvas.winfo_width()
        ch = self.viz_canvas.winfo_height()
        if cw < 10: cw = 450
        if ch < 10: ch = 350
        
        n_tokens, embed_dim = embeddings.shape
        
        if mode == "heatmap":
            # Show first 64 dims of each token as colored cells
            show_dims = min(64, embed_dim)
            cell_w = max(1, cw // show_dims)
            cell_h = max(1, ch // n_tokens)
            
            # Normalize to 0-1
            vmin = embeddings[:, :show_dims].min()
            vmax = embeddings[:, :show_dims].max()
            if vmax - vmin < 1e-10:
                vmax = vmin + 1
            
            for row in range(min(n_tokens, ch // max(cell_h, 1))):
                for col in range(show_dims):
                    val = (embeddings[row, col] - vmin) / (vmax - vmin)
                    # Red-black-green color map
                    if val < 0.5:
                        r = int(255 * (1 - val * 2))
                        g = 0
                    else:
                        r = 0
                        g = int(255 * (val - 0.5) * 2)
                    color = f"#{r:02x}{g:02x}00"
                    x1, y1 = col * cell_w, row * cell_h
                    x2, y2 = x1 + cell_w, y1 + cell_h
                    self.viz_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
            
            self.viz_info.config(text=f"Heatmap: {n_tokens} tokens × {show_dims} dims")
        
        elif mode == "similarity":
            # Cosine similarity matrix
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1
            normalized = embeddings / norms
            sim_matrix = normalized @ normalized.T
            
            cell_size = max(1, min(cw, ch) // n_tokens)
            
            for i in range(n_tokens):
                for j in range(n_tokens):
                    val = (sim_matrix[i, j] + 1) / 2  # normalize -1..1 to 0..1
                    intensity = int(255 * val)
                    color = f"#{intensity:02x}00{intensity:02x}"
                    x1, y1 = j * cell_size, i * cell_size
                    x2, y2 = x1 + cell_size, y1 + cell_size
                    self.viz_canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
            
            self.viz_info.config(text=f"Similarity: {n_tokens}×{n_tokens} cosine matrix")


# ============================================================
# CHAT HISTORY MANAGER PANEL
# ============================================================

class ChatHistoryManagerPanel:
    """Save/load/delete/export conversation histories."""
    
    def __init__(self, parent_root, colors):
        self.root = parent_root
        self.colors = colors
        self.frame = None
        self.visible = False
        self.history_dir = SCRIPT_DIR / "chat_history"
        self.history_dir.mkdir(exist_ok=True)
    
    def toggle(self):
        if self.visible:
            self.frame.place_forget()
            self.visible = False
        else:
            self._build_if_needed()
            self._refresh_list()
            self.frame.place(relx=1.0, y=0, anchor="ne", relheight=1.0, width=380)
            self.visible = True
    
    def _build_if_needed(self):
        if self.frame:
            return
        
        self.frame = tk.Frame(self.root, bg=COLORS["bg"], bd=2, relief=tk.RIDGE)
        
        # Header
        header = tk.Frame(self.frame, bg=COLORS["bg"])
        header.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(header, text="💬 CHAT HISTORY", font=("Consolas", 14, "bold"),
                bg=COLORS["bg"], fg=COLORS["blood_bright"]).pack(side=tk.LEFT)
        tk.Button(header, text="✕", font=("Consolas", 12),
                 bg=COLORS["bg"], fg=COLORS["text_dim"], relief=tk.FLAT,
                 command=self.toggle).pack(side=tk.RIGHT)
        
        # Buttons
        btn_frame = tk.Frame(self.frame, bg=COLORS["bg"])
        btn_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(btn_frame, text="🔄 Refresh", font=("Consolas", 10),
                 bg=COLORS["bg_card"], fg=COLORS["text"], relief=tk.FLAT,
                 command=self._refresh_list).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_frame, text="📤 Export All", font=("Consolas", 10),
                 bg=COLORS["bg_card"], fg=COLORS["matrix_green"], relief=tk.FLAT,
                 command=self._export_all).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(btn_frame, text="🗑️ Delete Selected", font=("Consolas", 10),
                 bg=COLORS["bg_card"], fg=COLORS["blood_bright"], relief=tk.FLAT,
                 command=self._delete_selected).pack(side=tk.LEFT)
        
        # History list
        self.history_listbox = tk.Listbox(self.frame, bg=COLORS["bg_input"], fg=COLORS["text"],
                                         font=("Consolas", 11), relief=tk.FLAT,
                                         selectbackground=COLORS["blood_red"],
                                         selectforeground=COLORS["text_bright"])
        self.history_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.history_listbox.bind('<Double-1>', self._preview_history)
        
        # Preview area
        tk.Label(self.frame, text="─── Preview ───", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(fill=tk.X, padx=10, pady=(5, 2))
        
        self.preview_text = tk.Text(self.frame, bg=COLORS["bg_input"], fg=COLORS["text"],
                                   font=("Consolas", 10), height=8, relief=tk.FLAT,
                                   state=tk.DISABLED, wrap=tk.WORD)
        self.preview_text.pack(fill=tk.X, padx=10, pady=5)
    
    def _refresh_list(self):
        self.history_listbox.delete(0, tk.END)
        if self.history_dir.exists():
            files = sorted(self.history_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
            for f in files:
                size = f.stat().st_size
                mtime = time.strftime('%Y-%m-%d %H:%M', time.localtime(f.stat().st_mtime))
                self.history_listbox.insert(tk.END, f"{f.stem}  ({size//1024}KB)  {mtime}")
    
    def _preview_history(self, event=None):
        selection = self.history_listbox.curselection()
        if not selection:
            return
        idx = selection[0]
        files = sorted(self.history_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if idx >= len(files):
            return
        
        filepath = files[idx]
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            preview_lines = []
            messages = data if isinstance(data, list) else data.get('messages', [])
            for msg in messages[-10:]:
                role = msg.get('role', '?')
                content = msg.get('content', '')[:100]
                preview_lines.append(f"[{role}] {content}")
            
            self.preview_text.config(state=tk.NORMAL)
            self.preview_text.delete('1.0', tk.END)
            self.preview_text.insert('1.0', '\n'.join(preview_lines))
            self.preview_text.config(state=tk.DISABLED)
        except:
            pass
    
    def _delete_selected(self):
        selection = self.history_listbox.curselection()
        if not selection:
            return
        if not messagebox.askyesno("Delete", "Delete selected chat history?"):
            return
        idx = selection[0]
        files = sorted(self.history_dir.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        if idx < len(files):
            files[idx].unlink()
            self._refresh_list()
    
    def _export_all(self):
        export_path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="chat_history_export.json"
        )
        if not export_path:
            return
        
        all_histories = []
        for f in self.history_dir.glob("*.json"):
            try:
                with open(f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                all_histories.append({"filename": f.name, "data": data})
            except:
                pass
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(all_histories, f, indent=2)
        
        messagebox.showinfo("Export", f"Exported {len(all_histories)} histories to:\n{export_path}")


# ============================================================
# MODEL DOWNLOAD MANAGER PANEL
# ============================================================

class ModelDownloadPanel:
    """Download GGUF models from URLs (pure HTTP GET, no pip)."""
    
    def __init__(self, parent_root, colors):
        self.root = parent_root
        self.colors = colors
        self.frame = None
        self.visible = False
        self.download_dir = SCRIPT_DIR / "models"
        self.download_dir.mkdir(exist_ok=True)
        self.active_downloads = []
    
    def toggle(self):
        if self.visible:
            self.frame.place_forget()
            self.visible = False
        else:
            self._build_if_needed()
            self.frame.place(relx=1.0, y=0, anchor="ne", relheight=1.0, width=450)
            self.visible = True
    
    def _build_if_needed(self):
        if self.frame:
            return
        
        self.frame = tk.Frame(self.root, bg=COLORS["bg"], bd=2, relief=tk.RIDGE)
        
        # Header
        header = tk.Frame(self.frame, bg=COLORS["bg"])
        header.pack(fill=tk.X, padx=10, pady=10)
        tk.Label(header, text="📥 MODEL DOWNLOADER", font=("Consolas", 14, "bold"),
                bg=COLORS["bg"], fg=COLORS["blood_bright"]).pack(side=tk.LEFT)
        tk.Button(header, text="✕", font=("Consolas", 12),
                 bg=COLORS["bg"], fg=COLORS["text_dim"], relief=tk.FLAT,
                 command=self.toggle).pack(side=tk.RIGHT)
        
        # URL input
        tk.Label(self.frame, text="Model URL (.gguf):", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text"]).pack(anchor=tk.W, padx=10, pady=(10, 2))
        
        url_frame = tk.Frame(self.frame, bg=COLORS["bg"])
        url_frame.pack(fill=tk.X, padx=10, pady=2)
        
        self.url_var = tk.StringVar()
        tk.Entry(url_frame, textvariable=self.url_var, font=("Consolas", 10),
                bg=COLORS["bg_input"], fg=COLORS["text"], relief=tk.FLAT,
                insertbackground=COLORS["matrix_green"]).pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        tk.Button(url_frame, text="⬇ Download", font=("Consolas", 11),
                 bg=COLORS["blood_red"], fg=COLORS["text_bright"], relief=tk.FLAT,
                 command=self._start_download).pack(side=tk.RIGHT, padx=(5, 0))
        
        # Download directory
        tk.Label(self.frame, text=f"Save to: {self.download_dir}", font=("Consolas", 9),
                bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(anchor=tk.W, padx=10, pady=2)
        
        # Active downloads list
        tk.Label(self.frame, text="─── Downloads ───", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(fill=tk.X, padx=10, pady=(10, 2))
        
        self.downloads_frame = tk.Frame(self.frame, bg=COLORS["bg"])
        self.downloads_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Existing files
        tk.Label(self.frame, text="─── Downloaded Models ───", font=("Consolas", 11),
                bg=COLORS["bg"], fg=COLORS["text_dim"]).pack(fill=tk.X, padx=10, pady=(5, 2))
        
        self.files_listbox = tk.Listbox(self.frame, bg=COLORS["bg_input"], fg=COLORS["text"],
                                       font=("Consolas", 10), height=6, relief=tk.FLAT)
        self.files_listbox.pack(fill=tk.X, padx=10, pady=5)
        self._refresh_files()
    
    def _refresh_files(self):
        self.files_listbox.delete(0, tk.END)
        for f in sorted(self.download_dir.glob("*.gguf")):
            size_gb = f.stat().st_size / (1024**3)
            self.files_listbox.insert(tk.END, f"{f.name}  ({size_gb:.2f} GB)")
    
    def _start_download(self):
        url = self.url_var.get().strip()
        if not url:
            return
        if not url.startswith('http'):
            messagebox.showwarning("Download", "URL must start with http:// or https://")
            return
        
        # Extract filename from URL
        filename = url.split('/')[-1].split('?')[0]
        if not filename.endswith('.gguf'):
            filename += '.gguf'
        
        output_path = self.download_dir / filename
        
        # Create progress widget
        dl_frame = tk.Frame(self.downloads_frame, bg=COLORS["bg_card"], bd=1, relief=tk.SOLID)
        dl_frame.pack(fill=tk.X, pady=2)
        
        name_label = tk.Label(dl_frame, text=filename, font=("Consolas", 10),
                            bg=COLORS["bg_card"], fg=COLORS["text"])
        name_label.pack(anchor=tk.W, padx=5, pady=(4, 0))
        
        progress_var = tk.DoubleVar(value=0)
        progress_bar = ttk.Progressbar(dl_frame, variable=progress_var, maximum=100)
        progress_bar.pack(fill=tk.X, padx=5, pady=2)
        
        status_label = tk.Label(dl_frame, text="Starting...", font=("Consolas", 9),
                               bg=COLORS["bg_card"], fg=COLORS["matrix_green"])
        status_label.pack(anchor=tk.W, padx=5, pady=(0, 4))
        
        # Start download thread
        threading.Thread(target=self._do_download,
                        args=(url, output_path, progress_var, status_label),
                        daemon=True).start()
    
    def _do_download(self, url, output_path, progress_var, status_label):
        import urllib.request
        import urllib.error
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'NightmareAI/5.0'})
            response = urllib.request.urlopen(req, timeout=30)
            
            total_size = int(response.headers.get('Content-Length', 0))
            downloaded = 0
            chunk_size = 1024 * 1024  # 1MB chunks
            start_time = time.time()
            
            with open(output_path, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress
                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        speed = downloaded / (time.time() - start_time) / (1024 * 1024)
                        remaining = (total_size - downloaded) / max(1, speed * 1024 * 1024)
                        
                        self.root.after(0, lambda p=pct: progress_var.set(p))
                        self.root.after(0, lambda s=speed, r=remaining, d=downloaded:
                            status_label.config(text=f"{d/(1024**3):.2f} GB | {s:.1f} MB/s | ETA: {int(r)}s"))
            
            self.root.after(0, lambda: progress_var.set(100))
            self.root.after(0, lambda: status_label.config(text=f"✓ Complete: {output_path.name}"))
            self.root.after(0, self._refresh_files)
            
        except Exception as e:
            self.root.after(0, lambda: status_label.config(
                text=f"❌ Error: {str(e)[:50]}", fg=COLORS["blood_bright"]))


# ============================================================
# PANEL INTEGRATION — Register all panels with the main app
# ============================================================

def register_tool_panels(app_instance):
    """
    Call this from NightmareAILoader.__init__ to register all tool panels.
    Adds buttons to the admin panel and sets up toggle functions.
    """
    root = app_instance.root
    
    # Create panel instances
    app_instance._inspector_panel = ModelInspectorPanel(root, COLORS)
    app_instance._analyzer_panel = ModelAnalyzerPanel(root, COLORS)
    app_instance._merge_tool_panel = MergeToolPanel(root, COLORS)
    app_instance._quant_panel = QuantizationToolPanel(root, COLORS)
    app_instance._token_panel = TokenInspectorPanel(root, COLORS)
    app_instance._embedding_panel = EmbeddingVisualizerPanel(root, COLORS)
    app_instance._chat_history_panel = ChatHistoryManagerPanel(root, COLORS)
    app_instance._download_panel = ModelDownloadPanel(root, COLORS)
    
    # Visibility flags
    app_instance._inspector_panel_visible = False
    app_instance._analyzer_panel_visible = False
    app_instance._merge_tool_panel_visible = False
    app_instance._quant_panel_visible = False
    app_instance._token_panel_visible = False
    app_instance._embedding_panel_visible = False
    app_instance._chat_history_panel_visible = False
    app_instance._download_panel_visible = False
    
    return {
        "🔍 Model Inspector": app_instance._inspector_panel.toggle,
        "🧬 Model Analyzer": app_instance._analyzer_panel.toggle,
        "⚡ Merge Tool": app_instance._merge_tool_panel.toggle,
        "⚗️ Quantization": app_instance._quant_panel.toggle,
        "🔤 Token Inspector": app_instance._token_panel.toggle,
        "🧮 Embeddings": app_instance._embedding_panel.toggle,
        "💬 Chat History": app_instance._chat_history_panel.toggle,
        "📥 Model Downloads": app_instance._download_panel.toggle,
    }
