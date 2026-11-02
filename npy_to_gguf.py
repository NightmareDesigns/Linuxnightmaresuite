"""
NPY to GGUF Converter — Nightmare Suite v6
============================================
Converts NumPy (.npy) weight checkpoints to GGUF format for inference.

Supports:
  - Single .npy files (one tensor per file)
  - Directories of .npy files (one per layer weight)
  - Optional quantization during export (F32, F16, Q8_0)

Usage:
  python npy_to_gguf.py <input_dir> <output.gguf> [--arch llama] [--quant q8_0]
  
  # Example: convert a directory of .npy weight files
  python npy_to_gguf.py ./my_model_weights ./my_model.gguf --arch llama --quant q8_0

The input directory should contain .npy files named to match GGUF tensor conventions:
  token_embd.weight.npy
  blk.0.attn_q.weight.npy
  blk.0.attn_k.weight.npy
  blk.0.attn_v.weight.npy
  blk.0.attn_output.weight.npy
  blk.0.ffn_gate.weight.npy
  blk.0.ffn_up.weight.npy
  blk.0.ffn_down.weight.npy
  blk.0.attn_norm.weight.npy
  blk.0.ffn_norm.weight.npy
  output_norm.weight.npy
  output.weight.npy  (optional — if absent, uses tied embeddings)

Or PyTorch-style names (auto-mapped):
  model.embed_tokens.weight.npy
  model.layers.0.self_attn.q_proj.weight.npy
  model.layers.0.self_attn.k_proj.weight.npy
  etc.
"""

import os
import sys
import struct
import numpy as np
from pathlib import Path


# ============================================================
# GGUF FORMAT CONSTANTS
# ============================================================

GGUF_MAGIC = 0x46554747  # "GGUF" in little-endian (G=0x47, G=0x47, U=0x55, F=0x46)
GGUF_VERSION = 3

# GGML tensor types
GGML_TYPE_F32  = 0
GGML_TYPE_F16  = 1
GGML_TYPE_Q8_0 = 8

# GGUF metadata value types
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10

# Block sizes for quantization
QUANT_BLOCK_SIZE = {
    GGML_TYPE_F32:  (1, 4),    # 1 element, 4 bytes
    GGML_TYPE_F16:  (1, 2),    # 1 element, 2 bytes
    GGML_TYPE_Q8_0: (32, 34),  # 32 elements, 34 bytes (2 scale + 32 quants)
}


# ============================================================
# QUANTIZATION FUNCTIONS
# ============================================================

def quantize_f16(data: np.ndarray) -> bytes:
    """Convert float32 array to float16 bytes."""
    return data.astype(np.float16).tobytes()


def quantize_q8_0(data: np.ndarray) -> bytes:
    """
    Quantize float32 array to Q8_0 format.
    
    Q8_0 block (34 bytes):
      [scale: float16 (2 bytes)] [quants: int8 x 32 (32 bytes)]
    
    Algorithm:
      scale = max(abs(block)) / 127
      quant[i] = round(block[i] / scale)
    """
    data = data.flatten().astype(np.float32)
    n = len(data)
    
    # Pad to multiple of 32
    if n % 32 != 0:
        pad = 32 - (n % 32)
        data = np.concatenate([data, np.zeros(pad, dtype=np.float32)])
        n = len(data)
    
    n_blocks = n // 32
    blocks = data.reshape(n_blocks, 32)
    
    result = bytearray()
    
    for block in blocks:
        # Compute scale
        amax = np.max(np.abs(block))
        scale = amax / 127.0 if amax > 0 else 0.0
        
        # Quantize
        if scale > 0:
            quants = np.round(block / scale).astype(np.int8)
        else:
            quants = np.zeros(32, dtype=np.int8)
        
        # Write scale as float16
        scale_f16 = np.float16(scale)
        result.extend(scale_f16.tobytes())
        
        # Write quants as int8
        result.extend(quants.tobytes())
    
    return bytes(result)


# ============================================================
# PYTORCH → GGUF NAME MAPPING
# ============================================================

def map_tensor_name(name: str) -> str:
    """
    Map PyTorch/HuggingFace tensor names to GGUF convention.
    
    Examples:
      model.embed_tokens.weight → token_embd.weight
      model.layers.0.self_attn.q_proj.weight → blk.0.attn_q.weight
      model.layers.0.mlp.gate_proj.weight → blk.0.ffn_gate.weight
      model.norm.weight → output_norm.weight
      lm_head.weight → output.weight
    """
    # Remove common prefixes
    name = name.replace("model.", "").replace("transformer.", "")
    
    # Embedding
    if "embed_tokens" in name or "wte" in name or "token_embedding" in name:
        return "token_embd.weight"
    
    # Output norm
    if name in ("norm.weight", "ln_f.weight", "final_layernorm.weight"):
        return "output_norm.weight"
    
    # Output/LM head
    if "lm_head" in name or name == "output.weight":
        return "output.weight"
    
    # Layer weights
    import re
    layer_match = re.search(r'layers?[._](\d+)[._]', name)
    if layer_match:
        layer_idx = layer_match.group(1)
        prefix = f"blk.{layer_idx}"
        
        # Attention
        if "q_proj" in name or "wq" in name:
            return f"{prefix}.attn_q.weight"
        if "k_proj" in name or "wk" in name:
            return f"{prefix}.attn_k.weight"
        if "v_proj" in name or "wv" in name:
            return f"{prefix}.attn_v.weight"
        if "o_proj" in name or "wo" in name:
            return f"{prefix}.attn_output.weight"
        
        # Attention norm
        if "input_layernorm" in name or "attention_norm" in name or "attn_norm" in name:
            return f"{prefix}.attn_norm.weight"
        
        # FFN
        if "gate_proj" in name or "w1" in name or "ffn_gate" in name:
            return f"{prefix}.ffn_gate.weight"
        if "up_proj" in name or "w3" in name or "ffn_up" in name:
            return f"{prefix}.ffn_up.weight"
        if "down_proj" in name or "w2" in name or "ffn_down" in name:
            return f"{prefix}.ffn_down.weight"
        
        # FFN norm
        if "post_attention_layernorm" in name or "ffn_norm" in name:
            return f"{prefix}.ffn_norm.weight"
    
    # If already in GGUF format, return as-is
    if name.startswith("blk.") or name.startswith("token_embd") or name.startswith("output"):
        return name
    
    # Unknown — return cleaned name
    return name


# ============================================================
# GGUF WRITER
# ============================================================

class GGUFWriter:
    """Writes tensors and metadata to a GGUF file."""
    
    def __init__(self, output_path: str):
        self.output_path = output_path
        self.metadata = {}  # key -> (type, value)
        self.tensors = []   # [(name, shape, dtype, data_bytes)]
    
    def add_metadata_string(self, key: str, value: str):
        self.metadata[key] = (GGUF_TYPE_STRING, value)
    
    def add_metadata_uint32(self, key: str, value: int):
        self.metadata[key] = (GGUF_TYPE_UINT32, value)
    
    def add_metadata_int32(self, key: str, value: int):
        self.metadata[key] = (GGUF_TYPE_INT32, value)
    
    def add_metadata_float32(self, key: str, value: float):
        self.metadata[key] = (GGUF_TYPE_FLOAT32, value)
    
    def add_metadata_array_string(self, key: str, values: list):
        self.metadata[key] = (GGUF_TYPE_ARRAY, (GGUF_TYPE_STRING, values))
    
    def add_tensor(self, name: str, data: np.ndarray, quant_type: int = GGML_TYPE_F32):
        """
        Add a tensor to the GGUF file.
        
        Args:
            name: GGUF tensor name
            data: numpy array (float32)
            quant_type: target quantization (F32, F16, Q8_0)
        """
        # Quantize
        if quant_type == GGML_TYPE_F32:
            data_bytes = data.astype(np.float32).tobytes()
        elif quant_type == GGML_TYPE_F16:
            data_bytes = quantize_f16(data)
        elif quant_type == GGML_TYPE_Q8_0:
            data_bytes = quantize_q8_0(data)
        else:
            raise ValueError(f"Unsupported quant type: {quant_type}")
        
        self.tensors.append((name, data.shape, quant_type, data_bytes))
    
    def write(self):
        """Write the complete GGUF file."""
        with open(self.output_path, "wb") as f:
            # === Header ===
            f.write(struct.pack("<I", GGUF_MAGIC))
            f.write(struct.pack("<I", GGUF_VERSION))
            f.write(struct.pack("<Q", len(self.tensors)))    # n_tensors
            f.write(struct.pack("<Q", len(self.metadata)))   # n_metadata
            
            # === Metadata KV pairs ===
            for key, (vtype, value) in self.metadata.items():
                self._write_string(f, key)
                if vtype == GGUF_TYPE_STRING:
                    f.write(struct.pack("<I", GGUF_TYPE_STRING))
                    self._write_string(f, value)
                elif vtype == GGUF_TYPE_UINT32:
                    f.write(struct.pack("<I", GGUF_TYPE_UINT32))
                    f.write(struct.pack("<I", value))
                elif vtype == GGUF_TYPE_INT32:
                    f.write(struct.pack("<I", GGUF_TYPE_INT32))
                    f.write(struct.pack("<i", value))
                elif vtype == GGUF_TYPE_FLOAT32:
                    f.write(struct.pack("<I", GGUF_TYPE_FLOAT32))
                    f.write(struct.pack("<f", value))
                elif vtype == GGUF_TYPE_ARRAY:
                    elem_type, arr_values = value
                    f.write(struct.pack("<I", GGUF_TYPE_ARRAY))
                    f.write(struct.pack("<I", elem_type))
                    f.write(struct.pack("<Q", len(arr_values)))
                    for v in arr_values:
                        if elem_type == GGUF_TYPE_STRING:
                            self._write_string(f, v)
                        elif elem_type == GGUF_TYPE_FLOAT32:
                            f.write(struct.pack("<f", v))
                        elif elem_type == GGUF_TYPE_INT32:
                            f.write(struct.pack("<i", v))
            
            # === Tensor infos ===
            # Calculate offsets (relative to start of tensor data section)
            tensor_data_start = None  # Will be calculated after writing all infos
            offset = 0
            tensor_offsets = []
            
            for name, shape, dtype, data_bytes in self.tensors:
                # Align to 32 bytes
                aligned_offset = self._align(offset, 32)
                tensor_offsets.append(aligned_offset)
                offset = aligned_offset + len(data_bytes)
            
            for i, (name, shape, dtype, data_bytes) in enumerate(self.tensors):
                self._write_string(f, name)
                f.write(struct.pack("<I", len(shape)))  # n_dims
                for dim in reversed(shape):  # GGUF stores dims in reverse (col-major)
                    f.write(struct.pack("<Q", dim))
                f.write(struct.pack("<I", dtype))
                f.write(struct.pack("<Q", tensor_offsets[i]))
            
            # === Tensor data (aligned to 32 bytes from start of data section) ===
            # Align file position to 32 bytes for tensor data start
            current_pos = f.tell()
            aligned_pos = self._align(current_pos, 32)
            f.write(b'\x00' * (aligned_pos - current_pos))
            
            for i, (name, shape, dtype, data_bytes) in enumerate(self.tensors):
                # Pad to aligned offset
                current = f.tell() - aligned_pos
                target = tensor_offsets[i]
                if target > current:
                    f.write(b'\x00' * (target - current))
                f.write(data_bytes)
        
        file_size = os.path.getsize(self.output_path)
        print(f"\n[GGUF] Written: {self.output_path}")
        print(f"[GGUF] Size: {file_size / (1024*1024):.1f} MB")
        print(f"[GGUF] Tensors: {len(self.tensors)}")
    
    def _write_string(self, f, s: str):
        """Write a GGUF string (length-prefixed, UTF-8)."""
        encoded = s.encode("utf-8")
        f.write(struct.pack("<Q", len(encoded)))
        f.write(encoded)
    
    @staticmethod
    def _align(pos: int, alignment: int) -> int:
        return pos + (alignment - pos % alignment) % alignment


# ============================================================
# ARCHITECTURE DETECTION
# ============================================================

def detect_architecture(tensor_names: list, shapes: dict) -> dict:
    """
    Detect model architecture from tensor names and shapes.
    
    Returns dict with: arch, dim, n_layers, n_heads, n_kv_heads, 
                       ff_dim, vocab_size, head_dim, etc.
    """
    config = {}
    
    # Count layers
    layer_indices = set()
    for name in tensor_names:
        import re
        m = re.search(r'blk\.(\d+)\.', name)
        if m:
            layer_indices.add(int(m.group(1)))
    
    config['n_layers'] = max(layer_indices) + 1 if layer_indices else 0
    
    # Get dimensions from known tensors
    if 'token_embd.weight' in shapes:
        vocab, dim = shapes['token_embd.weight']
        config['vocab_size'] = vocab
        config['dim'] = dim
    
    if 'blk.0.attn_q.weight' in shapes:
        shape = shapes['blk.0.attn_q.weight']
        # Shape is (dim, n_heads * head_dim) in our convention
        config['n_heads_x_head_dim'] = shape[1] if len(shape) == 2 else shape[0]
    
    if 'blk.0.attn_k.weight' in shapes:
        shape = shapes['blk.0.attn_k.weight']
        config['n_kv_heads_x_head_dim'] = shape[1] if len(shape) == 2 else shape[0]
    
    if 'blk.0.ffn_gate.weight' in shapes:
        shape = shapes['blk.0.ffn_gate.weight']
        config['ff_dim'] = shape[1] if len(shape) == 2 else shape[0]
    elif 'blk.0.ffn_up.weight' in shapes:
        shape = shapes['blk.0.ffn_up.weight']
        config['ff_dim'] = shape[1] if len(shape) == 2 else shape[0]
    
    # Compute head counts
    dim = config.get('dim', 0)
    if dim > 0:
        q_dim = config.get('n_heads_x_head_dim', dim)
        kv_dim = config.get('n_kv_heads_x_head_dim', dim)
        
        # Try common head_dims
        for hd in [128, 64, 256, 96, 80]:
            if q_dim % hd == 0:
                config['head_dim'] = hd
                config['n_heads'] = q_dim // hd
                if kv_dim % hd == 0:
                    config['n_kv_heads'] = kv_dim // hd
                break
        
        if 'head_dim' not in config:
            config['head_dim'] = dim // 32  # fallback guess
            config['n_heads'] = 32
            config['n_kv_heads'] = 8
    
    return config


# ============================================================
# MAIN CONVERSION
# ============================================================

def convert_npy_to_gguf(input_dir: str, output_path: str, 
                         architecture: str = "llama",
                         quant: str = "q8_0",
                         model_name: str = None,
                         context_length: int = 4096,
                         rope_base: float = 10000.0):
    """
    Convert a directory of .npy files to a GGUF model.
    
    Args:
        input_dir: path to directory containing .npy weight files
        output_path: output .gguf file path
        architecture: model architecture (llama, gemma, mistral)
        quant: quantization level (f32, f16, q8_0)
        model_name: model name for metadata
        context_length: max context length
        rope_base: RoPE base frequency
    """
    input_path = Path(input_dir)
    
    if not input_path.exists():
        print(f"ERROR: Input path not found: {input_dir}")
        return False
    
    # Find all .npy files
    npy_files = sorted(input_path.glob("**/*.npy"))
    
    if not npy_files:
        print(f"ERROR: No .npy files found in {input_dir}")
        return False
    
    print(f"╔══════════════════════════════════════════════════╗")
    print(f"║  NPY → GGUF Converter — Nightmare Suite v6     ║")
    print(f"╚══════════════════════════════════════════════════╝")
    print(f"\n  Input:  {input_dir} ({len(npy_files)} .npy files)")
    print(f"  Output: {output_path}")
    print(f"  Arch:   {architecture}")
    print(f"  Quant:  {quant}")
    print()
    
    # Select quant type
    quant_map = {
        'f32': GGML_TYPE_F32,
        'f16': GGML_TYPE_F16,
        'q8_0': GGML_TYPE_Q8_0,
    }
    
    if quant.lower() not in quant_map:
        print(f"ERROR: Unknown quant type '{quant}'. Use: f32, f16, q8_0")
        return False
    
    quant_type = quant_map[quant.lower()]
    
    # Load all tensors and map names
    print("Loading tensors...")
    tensors = {}  # gguf_name -> numpy array
    
    for npy_file in npy_files:
        # Derive tensor name from filename
        # "blk.0.attn_q.weight.npy" → "blk.0.attn_q.weight"
        # "model.layers.0.self_attn.q_proj.weight.npy" → mapped
        raw_name = npy_file.stem  # removes .npy
        if not raw_name.endswith(".weight"):
            raw_name += ".weight"
        
        gguf_name = map_tensor_name(raw_name)
        
        try:
            data = np.load(str(npy_file))
            tensors[gguf_name] = data.astype(np.float32)
            print(f"  {npy_file.name} → {gguf_name} {data.shape}")
        except Exception as e:
            print(f"  WARNING: Failed to load {npy_file.name}: {e}")
    
    if not tensors:
        print("ERROR: No tensors loaded!")
        return False
    
    # Detect architecture
    shapes = {name: arr.shape for name, arr in tensors.items()}
    config = detect_architecture(list(tensors.keys()), shapes)
    
    print(f"\n  Detected: {config.get('n_layers', '?')} layers, "
          f"dim={config.get('dim', '?')}, "
          f"heads={config.get('n_heads', '?')}, "
          f"kv_heads={config.get('n_kv_heads', '?')}, "
          f"ff_dim={config.get('ff_dim', '?')}, "
          f"vocab={config.get('vocab_size', '?')}")
    
    # Build GGUF
    writer = GGUFWriter(output_path)
    
    # Add metadata
    dim = config.get('dim', 0)
    n_layers = config.get('n_layers', 0)
    n_heads = config.get('n_heads', 32)
    n_kv_heads = config.get('n_kv_heads', n_heads)
    ff_dim = config.get('ff_dim', dim * 4)
    vocab_size = config.get('vocab_size', 32000)
    head_dim = config.get('head_dim', dim // n_heads if n_heads > 0 else 64)
    
    if model_name is None:
        model_name = Path(input_dir).name
    
    writer.add_metadata_string("general.architecture", architecture)
    writer.add_metadata_string("general.name", model_name)
    writer.add_metadata_uint32(f"{architecture}.context_length", context_length)
    writer.add_metadata_uint32(f"{architecture}.embedding_length", dim)
    writer.add_metadata_uint32(f"{architecture}.block_count", n_layers)
    writer.add_metadata_uint32(f"{architecture}.attention.head_count", n_heads)
    writer.add_metadata_uint32(f"{architecture}.attention.head_count_kv", n_kv_heads)
    writer.add_metadata_uint32(f"{architecture}.feed_forward_length", ff_dim)
    writer.add_metadata_float32(f"{architecture}.attention.layer_norm_rms_epsilon", 1e-5)
    writer.add_metadata_float32(f"{architecture}.rope.freq_base", rope_base)
    writer.add_metadata_uint32(f"{architecture}.vocab_size", vocab_size)
    
    # Add tensors with quantization
    print(f"\nQuantizing to {quant.upper()} and writing...")
    
    # Norm weights should stay F32 (they're small and need precision)
    norm_quant = GGML_TYPE_F32
    
    total_params = 0
    for name, data in tensors.items():
        total_params += data.size
        
        # Use F32 for norm weights and small tensors
        if "norm" in name or data.ndim == 1 or data.size < 256:
            writer.add_tensor(name, data, norm_quant)
        else:
            writer.add_tensor(name, data, quant_type)
        
        print(f"  {name}: {data.shape} → {quant.upper() if 'norm' not in name and data.ndim > 1 else 'F32'}")
    
    # Write file
    print(f"\nWriting GGUF...")
    writer.write()
    
    print(f"\n{'='*50}")
    print(f"  DONE! Total parameters: {total_params:,}")
    print(f"  Model: {model_name}")
    print(f"  Architecture: {architecture}")
    print(f"  Quantization: {quant.upper()}")
    print(f"{'='*50}\n")
    
    return True


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Convert NumPy (.npy) weight files to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion with Q8_0 quantization:
  python npy_to_gguf.py ./weights ./model.gguf
  
  # Specify architecture and keep full precision:
  python npy_to_gguf.py ./weights ./model.gguf --arch llama --quant f32
  
  # Custom model name and context length:
  python npy_to_gguf.py ./weights ./model.gguf --name "My Model" --ctx 8192
  
  # With custom RoPE base:
  python npy_to_gguf.py ./weights ./model.gguf --rope-base 500000
        """
    )
    
    parser.add_argument("input", help="Directory containing .npy weight files")
    parser.add_argument("output", help="Output .gguf file path")
    parser.add_argument("--arch", default="llama", 
                       choices=["llama", "gemma", "gemma2", "gemma3", "mistral", "phi"],
                       help="Model architecture (default: llama)")
    parser.add_argument("--quant", default="q8_0",
                       choices=["f32", "f16", "q8_0"],
                       help="Quantization level (default: q8_0)")
    parser.add_argument("--name", default=None,
                       help="Model name for metadata")
    parser.add_argument("--ctx", type=int, default=4096,
                       help="Context length (default: 4096)")
    parser.add_argument("--rope-base", type=float, default=10000.0,
                       help="RoPE frequency base (default: 10000)")
    
    args = parser.parse_args()
    
    success = convert_npy_to_gguf(
        input_dir=args.input,
        output_path=args.output,
        architecture=args.arch,
        quant=args.quant,
        model_name=args.name,
        context_length=args.ctx,
        rope_base=args.rope_base,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
