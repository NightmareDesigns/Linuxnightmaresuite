"""
GGUF File Format Parser — From Scratch
Parses the binary GGUF format used by llama.cpp and most local AI models.

GGUF Spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

No external dependencies beyond struct (stdlib) and numpy.
"""

import struct
import nightmare_np as np
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

# Import the vectorized dequantization module
import dequant_fast

# Suppress overflow warnings in dequantization (minor precision, doesn't affect quality)
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="overflow")

# === GGUF Constants ===
GGUF_MAGIC = 0x46554747  # "GGUF" as little-endian uint32 (G=0x47, G=0x47, U=0x55, F=0x46)
GGUF_VERSION_2 = 2
GGUF_VERSION_3 = 3

# GGML Tensor Types (quantization formats)
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
# IQ (importance-matrix) quantization types
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
# Integer types (used for embeddings / vocab)
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16 = 30

# Metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12

# Bytes per element for each quant type
TYPE_SIZE = {
    GGML_TYPE_F32: 4,
    GGML_TYPE_F16: 2,
    GGML_TYPE_BF16: 2,
    GGML_TYPE_Q4_0: 0.5 + 2/32,
    GGML_TYPE_Q4_1: 0.5 + 4/32,
    GGML_TYPE_Q5_0: 0.625 + 2/32,
    GGML_TYPE_Q5_1: 0.625 + 4/32,
    GGML_TYPE_Q8_0: 1 + 2/32,
    GGML_TYPE_Q8_1: 1 + 4/32,
    GGML_TYPE_Q2_K: 0.25 + 10/256,
    GGML_TYPE_Q3_K: 0.375 + 14/256,
    GGML_TYPE_Q4_K: 0.5 + 16/256,
    GGML_TYPE_Q5_K: 0.625 + 24/256,
    GGML_TYPE_Q6_K: 0.75 + 18/256,
    GGML_TYPE_Q8_K: 1 + 4/256,     # 256-elem blocks, 292 bytes/block
    GGML_TYPE_I8: 1,
    GGML_TYPE_I16: 2,
    GGML_TYPE_I32: 4,
    GGML_TYPE_I64: 8,
    GGML_TYPE_F64: 8,
    GGML_TYPE_IQ4_NL: 0.5 + 2/32,  # 32-elem blocks, 18 bytes/block
    GGML_TYPE_IQ4_XS: 4.25 + 14/256, # 256-elem blocks, ~136 bytes
    GGML_TYPE_IQ2_XXS: 2 + 2/256,   # 256-elem, 66 bytes
    GGML_TYPE_IQ2_XS: 2 + 10/256,    # 256-elem, 74 bytes
    GGML_TYPE_IQ2_S: 2 + 58/256,     # 256-elem, 82 bytes
    GGML_TYPE_IQ3_XXS: 3 + 2/256,   # 256-elem, 98 bytes
    GGML_TYPE_IQ3_S: 3 + 34/256,     # 256-elem, 110 bytes
    GGML_TYPE_IQ1_S: 1 + 50/256,     # 256-elem, 50 bytes
    GGML_TYPE_IQ1_M: 1 + 56/256,     # 256-elem, 56 bytes
}

# Block sizes for quantized types
BLOCK_SIZE = {
    GGML_TYPE_F32: 1,
    GGML_TYPE_F16: 1,
    GGML_TYPE_BF16: 1,
    GGML_TYPE_Q4_0: 32,
    GGML_TYPE_Q4_1: 32,
    GGML_TYPE_Q5_0: 32,
    GGML_TYPE_Q5_1: 32,
    GGML_TYPE_Q8_0: 32,
    GGML_TYPE_Q8_1: 32,
    GGML_TYPE_Q2_K: 256,
    GGML_TYPE_Q3_K: 256,
    GGML_TYPE_Q4_K: 256,
    GGML_TYPE_Q5_K: 256,
    GGML_TYPE_Q6_K: 256,
    GGML_TYPE_Q8_K: 256,
    GGML_TYPE_I8: 1,
    GGML_TYPE_I16: 1,
    GGML_TYPE_I32: 1,
    GGML_TYPE_I64: 1,
    GGML_TYPE_F64: 1,
    GGML_TYPE_IQ4_NL: 32,
    GGML_TYPE_IQ4_XS: 256,
    GGML_TYPE_IQ2_XXS: 256,
    GGML_TYPE_IQ2_XS: 256,
    GGML_TYPE_IQ2_S: 256,
    GGML_TYPE_IQ3_XXS: 256,
    GGML_TYPE_IQ3_S: 256,
    GGML_TYPE_IQ1_S: 256,
    GGML_TYPE_IQ1_M: 256,
}

# Bytes per block for quantized formats
BYTES_PER_BLOCK = {
    GGML_TYPE_F32: 4,
    GGML_TYPE_F16: 2,
    GGML_TYPE_BF16: 2,
    GGML_TYPE_Q4_0: 18,
    GGML_TYPE_Q4_1: 20,
    GGML_TYPE_Q5_0: 22,
    GGML_TYPE_Q5_1: 24,
    GGML_TYPE_Q8_0: 34,
    GGML_TYPE_Q8_1: 36,
    GGML_TYPE_Q2_K: 84,
    GGML_TYPE_Q3_K: 110,
    GGML_TYPE_Q4_K: 144,
    GGML_TYPE_Q5_K: 176,
    GGML_TYPE_Q6_K: 210,
    GGML_TYPE_Q8_K: 292,
    GGML_TYPE_I8: 1,
    GGML_TYPE_I16: 2,
    GGML_TYPE_I32: 4,
    GGML_TYPE_I64: 8,
    GGML_TYPE_F64: 8,
    GGML_TYPE_IQ4_NL: 18,
    GGML_TYPE_IQ4_XS: 136,
    GGML_TYPE_IQ2_XXS: 66,
    GGML_TYPE_IQ2_XS: 74,
    GGML_TYPE_IQ2_S: 82,
    GGML_TYPE_IQ3_XXS: 98,
    GGML_TYPE_IQ3_S: 110,
    GGML_TYPE_IQ1_S: 50,
    GGML_TYPE_IQ1_M: 56,
}


@dataclass
class TensorInfo:
    """Metadata about a tensor stored in the GGUF file."""
    name: str
    n_dims: int
    shape: tuple
    dtype: int  # GGML type
    offset: int  # Offset from start of tensor data section


@dataclass
class GGUFModel:
    """Parsed GGUF model file."""
    version: int
    metadata: dict = field(default_factory=dict)
    tensors: dict = field(default_factory=dict)  # name -> TensorInfo
    tensor_data_offset: int = 0  # File offset where tensor data begins
    file_path: str = ""
    
    # === Safe metadata accessors ===
    # These handle all the weird formats different models use:
    # - Lists vs ints (Gemma 4 per-layer KV heads)
    # - Missing keys (some models omit optional fields)
    # - Different key naming patterns
    
    def _safe_int(self, value, default=0) -> int:
        """Safely convert a metadata value to int (handles lists, floats, etc.)."""
        if value is None:
            return default
        if isinstance(value, list):
            return max(value) if value else default
        if isinstance(value, float):
            return int(value)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default
    
    def _safe_float(self, value, default=0.0) -> float:
        """Safely convert a metadata value to float."""
        if value is None:
            return default
        if isinstance(value, list):
            return float(value[0]) if value else default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
    
    @property
    def architecture(self) -> str:
        return self.metadata.get("general.architecture", "unknown")
    
    @property
    def name(self) -> str:
        return self.metadata.get("general.name", "unknown")
    
    @property
    def context_length(self) -> int:
        arch = self.architecture
        val = self.metadata.get(f"{arch}.context_length", 2048)
        return self._safe_int(val, 2048)
    
    @property
    def embedding_length(self) -> int:
        arch = self.architecture
        val = self.metadata.get(f"{arch}.embedding_length", 4096)
        return self._safe_int(val, 4096)
    
    @property
    def block_count(self) -> int:
        arch = self.architecture
        val = self.metadata.get(f"{arch}.block_count", 32)
        return self._safe_int(val, 32)
    
    @property
    def head_count(self) -> int:
        arch = self.architecture
        val = self.metadata.get(f"{arch}.attention.head_count",
              self.metadata.get(f"{arch}.head_count", 32))  # old-style key
        return self._safe_int(val, 32)
    
    @property
    def head_count_kv(self):
        """
        Get KV head count. Returns int OR list (per-layer).
        
        Models that use per-layer KV heads (list):
        - Gemma 4 (sliding window attention)
        - Some custom merged models
        
        Models that use a single value (int):
        - LLaMA, Mistral, Phi, Qwen, DeepSeek, etc.
        """
        arch = self.architecture
        val = self.metadata.get(f"{arch}.attention.head_count_kv",
              self.metadata.get(f"{arch}.head_count_kv", self.head_count))
        # If it's a single-element list, unwrap it
        if isinstance(val, list) and len(val) == 1:
            return val[0]
        return val
    
    @property
    def head_count_kv_scalar(self) -> int:
        """Get a single KV head count (max value if per-layer list)."""
        val = self.head_count_kv
        if isinstance(val, list):
            return max(val) if val else self.head_count
        return self._safe_int(val, self.head_count)
    
    @property
    def vocab_size(self) -> int:
        val = self.metadata.get("tokenizer.ggml.vocab_size", None)
        if val is not None:
            return self._safe_int(val)
        tokens = self.metadata.get("tokenizer.ggml.tokens", [])
        return len(tokens) if isinstance(tokens, list) else 0
    
    @property
    def feed_forward_length(self) -> int:
        arch = self.architecture
        val = self.metadata.get(f"{arch}.feed_forward_length", 
              self.metadata.get(f"{arch}.feed_forward_dim", 11008))
        return self._safe_int(val, 11008)
    
    @property
    def rope_freq_base(self) -> float:
        arch = self.architecture
        val = self.metadata.get(f"{arch}.rope.freq_base",
              self.metadata.get(f"{arch}.rope.frequency_base", 10000.0))
        return self._safe_float(val, 10000.0)
    
    @property
    def norm_eps(self) -> float:
        """Get layer norm epsilon (different keys for different architectures)."""
        arch = self.architecture
        for key in [
            f"{arch}.attention.layer_norm_rms_epsilon",
            f"{arch}.attention.layer_norm_epsilon",
            f"{arch}.layer_norm_rms_epsilon",
            f"{arch}.layer_norm_epsilon",
            "general.layer_norm_rms_epsilon",
        ]:
            val = self.metadata.get(key)
            if val is not None:
                return self._safe_float(val, 1e-5)
        return 1e-5


class GGUFParser:
    """
    Binary parser for the GGUF file format.
    Reads headers, metadata, and tensor info without loading all tensor data into RAM.
    """
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.f = None
    
    def parse(self) -> GGUFModel:
        """Parse the GGUF file and return a GGUFModel with metadata and tensor info."""
        model = GGUFModel(version=0, file_path=str(self.file_path))
        
        # Verify file exists and is readable
        if not self.file_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.file_path}")
        
        file_size = self.file_path.stat().st_size
        if file_size < 20:
            raise ValueError(f"File too small to be a GGUF model ({file_size} bytes)")
        
        with open(self.file_path, "rb") as f:
            self.f = f
            
            # 1. Read and validate magic number
            magic = self._read_u32()
            if magic != GGUF_MAGIC:
                # Show first bytes for debugging
                f.seek(0)
                first_bytes = f.read(16)
                raise ValueError(
                    f"Not a GGUF file. Expected magic 0x{GGUF_MAGIC:08X} ('GGUF'), "
                    f"got 0x{magic:08X}. First 16 bytes: {first_bytes.hex()} "
                    f"File: {self.file_path} ({file_size:,} bytes)"
                )

            # 2. Read version
            model.version = self._read_u32()
            if model.version not in (1, GGUF_VERSION_2, GGUF_VERSION_3):
                raise ValueError(f"Unsupported GGUF version: {model.version}. Supported: 1, 2, 3")
            
            # GGUF v1 uses uint32 for counts, v2+ uses uint64
            use_64bit = model.version >= GGUF_VERSION_2
            
            # 3. Read counts
            if use_64bit:
                n_tensors = self._read_u64()
                n_metadata_kv = self._read_u64()
            else:
                n_tensors = self._read_u32()
                n_metadata_kv = self._read_u32()
            
            print(f"[GGUF] Version: {model.version}")
            print(f"[GGUF] Tensors: {n_tensors}")
            print(f"[GGUF] Metadata entries: {n_metadata_kv}")
            
            # 4. Read metadata key-value pairs
            meta_start = f.tell()
            try:
                for _ in range(n_metadata_kv):
                    key, value = self._read_metadata_kv()
                    model.metadata[key] = value
            except Exception:
                # Some custom exports (e.g. the "90M Llama Q4_0" file) write a
                # non-standard metadata type enum: 0=u32, 3=f32, 5=string.
                # Rewind and retry with that dialect before giving up.
                f.seek(meta_start)
                model.metadata.clear()
                self._meta_dialect = True
                try:
                    for _ in range(n_metadata_kv):
                        key, value = self._read_metadata_kv()
                        model.metadata[key] = value
                    print("[GGUF] Non-standard metadata dialect "
                          "(0=u32, 3=f32, 5=string) — handled")
                except Exception as e:
                    print(f"[GGUF] Warning: Error reading metadata entry: {e}")
                    print(f"[GGUF] File position: {f.tell()}")
                finally:
                    self._meta_dialect = False
            
            # 5. Read tensor infos
            for _ in range(n_tensors):
                tensor_info = self._read_tensor_info()
                model.tensors[tensor_info.name] = tensor_info

            # Detect the tensor shape convention. llama.cpp GGUFs store shapes in
            # REVERSE order (last dim / contiguous dim first); some custom GGUFs
            # (e.g. the user's tia.gguf) store them in standard order. We detect it
            # from a non-square linear weight: if shape[0] == embedding_length the
            # tensor is stored (in, out) -> reversed; otherwise it is (out, in).
            self.shape_reversed = True
            dim = model.embedding_length
            for name in ('blk.0.attn_k.weight', 'blk.0.attn_v.weight', 'blk.0.ffn_gate.weight'):
                ti = model.tensors.get(name)
                if ti is not None and len(ti.shape) == 2:
                    self.shape_reversed = (ti.shape[0] == dim)
                    break
            print(f"[GGUF] Tensor shape convention: reversed={self.shape_reversed}")

            # 6. Calculate tensor data offset (aligned to 32 bytes)
            current_pos = f.tell()
            alignment = model.metadata.get("general.alignment", 32)
            model.tensor_data_offset = self._align(current_pos, alignment)
            
            print(f"[GGUF] Architecture: {model.architecture}")
            print(f"[GGUF] Context length: {model.context_length}")
            print(f"[GGUF] Embedding size: {model.embedding_length}")
            print(f"[GGUF] Layers: {model.block_count}")
            print(f"[GGUF] Vocab size: {model.vocab_size}")
        
        self.f = None
        # Memory-map the file for fast tensor access (OS handles page
        # caching + prefetching — no per-tensor open/seek/read syscalls)
        try:
            import mmap as _mmap
            self._mmap_file = open(self.file_path, 'rb')
            self._mmap = _mmap.mmap(self._mmap_file.fileno(), 0,
                                   access=_mmap.ACCESS_READ)
        except Exception:
            self._mmap = None
            self._mmap_file = None
        return model
    
    def load_tensor(self, model: GGUFModel, tensor_name: str) -> np.ndarray:
        """Load a single tensor's data from disk and dequantize to float32.

        Uses the vectorized dequant_fast module for all quantization formats.
        """
        if tensor_name not in model.tensors:
            raise KeyError(f"Tensor '{tensor_name}' not found in model")

        info = model.tensors[tensor_name]
        file_offset = model.tensor_data_offset + info.offset
        n_elements = 1
        for dim in info.shape:
            n_elements *= dim

        # Read from mmap (fast — OS page cache) or fall back to file open
        import io
        if getattr(self, '_mmap', None) is not None:
            raw = self._mmap[file_offset:file_offset + n_elements * 4]  # max possible
            f = io.BytesIO(raw)
        else:
            f = open(self.file_path, 'rb')
            f.seek(file_offset)
        try:
            # Use vectorized dequantization for all formats
            data = dequant_fast.dequantize_tensor(f, n_elements, info.dtype)
        finally:
            if not hasattr(f, 'fileno') or f is not getattr(self, '_mmap', None):
                f.close()

        # GGUF stores tensor shapes in REVERSE order (last dim first, matching
        # ggml's ne[] layout where the last dimension is contiguous). So we
        # reshape with shape[::-1] to get the logical (rows, cols) layout.
        # Some custom GGUFs (e.g. the user's tia.gguf) store shapes in standard
        # order instead — detected during parse() and stored in self.shape_reversed.
        # We do NOT transpose 2D weights here: the forward pass uses `x @ W.T`
        # and expects weights in GGUF logical (out, in) orientation.
        if getattr(self, 'shape_reversed', True):
            return data.reshape(info.shape[::-1])
        return data.reshape(info.shape)

    def load_tensor_row(self, model: GGUFModel, tensor_name: str, row_idx: int) -> np.ndarray:
        """Load and dequantize a SINGLE row (token embedding) from a 2D tensor.
        
        Reads only the bytes for that row from memory-map/disk and dequantizes it
        instantly (< 0.01ms), saving multiple gigabytes of RAM during startup.
        """
        if tensor_name not in model.tensors:
            raise KeyError(f"Tensor '{tensor_name}' not found in model")

        info = model.tensors[tensor_name]
        shape = info.shape[::-1] if getattr(self, 'shape_reversed', True) else info.shape
        if len(shape) != 2:
            raise ValueError(f"load_tensor_row expects a 2D tensor, got shape {shape}")
        
        n_rows, n_cols = int(shape[0]), int(shape[1])
        if row_idx < 0 or row_idx >= n_rows:
            return np.zeros(n_cols, dtype=np.float32)

        block_elems, block_bytes = dequant_fast.BLOCK_SIZES[info.dtype]
        row_blocks = n_cols // block_elems
        row_bytes = row_blocks * block_bytes
        file_offset = model.tensor_data_offset + info.offset + row_idx * row_bytes

        import io
        if getattr(self, '_mmap', None) is not None:
            raw = self._mmap[file_offset:file_offset + row_bytes]
            f = io.BytesIO(raw)
        else:
            f = open(self.file_path, 'rb')
            f.seek(file_offset)
        try:
            row_data = dequant_fast.dequantize_tensor(f, n_cols, info.dtype)
        finally:
            if not hasattr(f, 'fileno') or f is not getattr(self, '_mmap', None):
                f.close()
        return row_data.reshape(-1)

    def load_tensor_raw(self, model: GGUFModel, tensor_name: str):
        """Load a tensor's raw quantized bytes without dequantizing.
        
        Returns: (raw_bytes: np.ndarray[uint8], quant_type: int, shape: tuple, n_elements: int)
        """
        if tensor_name not in model.tensors:
            raise KeyError(f"Tensor '{tensor_name}' not found in model")
        
        info = model.tensors[tensor_name]
        file_offset = model.tensor_data_offset + info.offset
        n_elements = 1
        for dim in info.shape:
            n_elements *= dim
        
        block_elems, block_bytes = dequant_fast.BLOCK_SIZES[info.dtype]
        total_bytes = (n_elements // block_elems) * block_bytes
        
        if getattr(self, '_mmap', None) is not None:
            raw = np.frombuffer(
                self._mmap[file_offset:file_offset + total_bytes],
                dtype=np.uint8).copy()
        else:
            with open(self.file_path, 'rb') as f:
                f.seek(file_offset)
                raw = np.frombuffer(f.read(total_bytes), dtype=np.uint8).copy()
        
        return raw, info.dtype, info.shape, n_elements
    
    # === Binary Reading Helpers ===
    
    def _read_u8(self) -> int:
        return struct.unpack("<B", self.f.read(1))[0]
    
    def _read_i8(self) -> int:
        return struct.unpack("<b", self.f.read(1))[0]
    
    def _read_u16(self) -> int:
        return struct.unpack("<H", self.f.read(2))[0]
    
    def _read_i16(self) -> int:
        return struct.unpack("<h", self.f.read(2))[0]
    
    def _read_u32(self) -> int:
        return struct.unpack("<I", self.f.read(4))[0]
    
    def _read_i32(self) -> int:
        return struct.unpack("<i", self.f.read(4))[0]
    
    def _read_u64(self) -> int:
        return struct.unpack("<Q", self.f.read(8))[0]
    
    def _read_i64(self) -> int:
        return struct.unpack("<q", self.f.read(8))[0]
    
    def _read_f32(self) -> float:
        return struct.unpack("<f", self.f.read(4))[0]
    
    def _read_f64(self) -> float:
        return struct.unpack("<d", self.f.read(8))[0]
    
    def _read_bool(self) -> bool:
        return self._read_u8() != 0
    
    def _read_string(self) -> str:
        """Read a length-prefixed string. Length is uint64 in GGUF v2+."""
        length = self._read_u64()
        if length > 1024 * 1024:  # Sanity check: no string > 1MB
            raise ValueError(f"String length too large: {length} — possible file corruption")
        if length < 0:
            raise ValueError(f"Negative string length: {length}")
        return self.f.read(length).decode("utf-8", errors="replace")
    
    def _read_metadata_kv(self) -> tuple:
        """Read a single metadata key-value pair."""
        key = self._read_string()
        value_type = self._read_u32()
        value = self._read_value(value_type)
        return key, value
    
    def _read_value(self, value_type: int) -> Any:
        """Read a typed value from the file."""
        if getattr(self, '_meta_dialect', False):
            # non-standard export dialect: 0=u32, 3=f32, 5=string
            if value_type == 5:
                return self._read_string()
            if value_type == 0:
                return self._read_u32()
            if value_type == 3:
                return self._read_f32()
            raise ValueError(f"Unknown dialect metadata type: {value_type}")
        if value_type == GGUF_TYPE_UINT8:
            return self._read_u8()
        elif value_type == GGUF_TYPE_INT8:
            return self._read_i8()
        elif value_type == GGUF_TYPE_UINT16:
            return self._read_u16()
        elif value_type == GGUF_TYPE_INT16:
            return self._read_i16()
        elif value_type == GGUF_TYPE_UINT32:
            return self._read_u32()
        elif value_type == GGUF_TYPE_INT32:
            return self._read_i32()
        elif value_type == GGUF_TYPE_UINT64:
            return self._read_u64()
        elif value_type == GGUF_TYPE_INT64:
            return self._read_i64()
        elif value_type == GGUF_TYPE_FLOAT32:
            return self._read_f32()
        elif value_type == GGUF_TYPE_FLOAT64:
            return self._read_f64()
        elif value_type == GGUF_TYPE_BOOL:
            return self._read_bool()
        elif value_type == GGUF_TYPE_STRING:
            return self._read_string()
        elif value_type == GGUF_TYPE_ARRAY:
            return self._read_array()
        else:
            raise ValueError(f"Unknown metadata type: {value_type}")
    
    def _read_array(self) -> list:
        """Read an array value (type + length + elements)."""
        elem_type = self._read_u32()
        n_elements = self._read_u64()
        if n_elements > 10_000_000:  # Sanity: no array > 10M elements
            raise ValueError(f"Array too large: {n_elements} elements — possible alignment error")
        if elem_type == GGUF_TYPE_FLOAT32:
            raw = self.f.read(n_elements * 4)
            return list(struct.unpack(f"<{n_elements}f", raw))
        elif elem_type == GGUF_TYPE_INT32:
            raw = self.f.read(n_elements * 4)
            return list(struct.unpack(f"<{n_elements}i", raw))
        elif elem_type == GGUF_TYPE_UINT32:
            raw = self.f.read(n_elements * 4)
            return list(struct.unpack(f"<{n_elements}I", raw))
        elif elem_type == GGUF_TYPE_UINT8:
            raw = self.f.read(n_elements)
            return list(raw)
        elif elem_type == GGUF_TYPE_STRING:
            # Fast string array read
            res = []
            f_read = self.f.read
            unpack_u64 = struct.unpack
            for _ in range(n_elements):
                slen = unpack_u64('<Q', f_read(8))[0]
                res.append(f_read(slen).decode('utf-8', errors='replace'))
            return res
        return [self._read_value(elem_type) for _ in range(n_elements)]
    
    def _read_tensor_info(self) -> TensorInfo:
        """Read tensor metadata (name, shape, type, offset)."""
        name = self._read_string()
        n_dims = self._read_u32()
        shape = tuple(self._read_u64() for _ in range(n_dims))
        dtype = self._read_u32()
        offset = self._read_u64()
        
        return TensorInfo(
            name=name,
            n_dims=n_dims,
            shape=shape,
            dtype=dtype,
            offset=offset
        )
    
    @staticmethod
    def _align(pos: int, alignment: int) -> int:
        """Align position to the given boundary."""
        return pos + (alignment - pos % alignment) % alignment

    def close(self):
        """Close mmap and file handles."""
        try:
            if getattr(self, '_mmap', None) is not None:
                self._mmap.close()
                self._mmap = None
        except Exception:
            pass
        try:
            if getattr(self, '_mmap_file', None) is not None:
                self._mmap_file.close()
                self._mmap_file = None
        except Exception:
            pass
        try:
            if self.f is not None:
                self.f.close()
                self.f = None
        except Exception:
            pass
