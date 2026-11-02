"""
╔══════════════════════════════════════════════════════════════╗
║  NIGHTMARE UNIVERSAL CONVERTER                               ║
║  Any model → Nightmare .ntmre compression compressed .ntmre              ║
║                                                              ║
║  Supports:                                                   ║
║    • GGUF (llama.cpp format — any quant)                     ║
║    • SafeTensors (HuggingFace format)                        ║
║    • PyTorch .bin / .pt (state_dict)                         ║
║    • Raw F32 .bin (Crucible checkpoints)                     ║
║    • .npy checkpoint directories                             ║
║                                                              ║
║  Nightmare Designs 💀 — Patent Pending                       ║
╚══════════════════════════════════════════════════════════════╝

Usage:
    python nightmare_convert.py model.gguf
    python nightmare_convert.py model.safetensors --output compressed.ntmre
    python nightmare_convert.py pytorch_model.bin --arch llama --dim 4096 --layers 32
    python nightmare_convert.py checkpoints/step_5000.bin --config config_1.3b.json

Supported input formats (auto-detected by extension):
    .gguf           → GGUF (any quantization — dequantized to F32 first)
    .safetensors    → HuggingFace SafeTensors
    .bin / .pt      → PyTorch state_dict or raw F32 flat checkpoint
    directory/      → .npy checkpoint directory (Crucible/NightmareAI format)
"""

import sys
import os
import json
import struct
import nightmare_np as np
from pathlib import Path

# ============================================================
# .ntmre compression PARAMETERS
# ============================================================

ENERGY_THRESHOLD = 0.999
MAX_RANK_RATIO   = 0.15
MIN_RANK         = 32
BLOCK_SIZE       = 32

# ============================================================
# Q8 QUANTIZATION
# ============================================================

def quantize_q8(data):
    """Quantize float32 → Q8_0 blocks (34 bytes per 32 elements)."""
    data = data.flatten().astype(np.float32)
    remainder = len(data) % BLOCK_SIZE
    if remainder != 0:
        data = np.concatenate([data, np.zeros(BLOCK_SIZE - remainder, dtype=np.float32)])
    
    n_blocks = len(data) // BLOCK_SIZE
    output = bytearray()
    
    for b in range(n_blocks):
        block = data[b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE]
        amax = np.max(np.abs(block))
        scale = amax / 127.0 if amax > 0 else 0.0
        inv_scale = 1.0 / scale if scale > 0 else 0.0
        scale_f16 = np.float16(scale)
        output += struct.pack('<e', scale_f16)
        quants = np.clip(np.round(block * inv_scale), -127, 127).astype(np.int8)
        output += quants.tobytes()
    
    return bytes(output), n_blocks


# ============================================================
# .ntmre compression COMPRESSION
# ============================================================

def compute_optimal_rank(S):
    """Energy-based rank selection (99.9% energy retention)."""
    s2 = S ** 2
    total = np.sum(s2)
    cumulative = np.cumsum(s2)
    energy_rank = int(np.searchsorted(cumulative, ENERGY_THRESHOLD * total)) + 1
    max_rank = max(int(len(S) * MAX_RANK_RATIO), MIN_RANK)
    rank = min(energy_rank, max_rank)
    rank = max(rank, MIN_RANK)
    rank = min(rank, len(S))
    energy = cumulative[rank - 1] / total
    return rank, energy


def svd_compress(W):
    """.ntmre compression compress a 2D weight matrix.
    Returns (q8_data, rank, energy).

    v13 storage: the .ntmre compression is a DENOISING step, not a size step. Both
    readers (the C loader's size guard in ntmre_format.h and the Python
    runtime's _upload_weights) dequantize .ntmre compression as flat Q8 blocks of the
    full matrix, and the C loader REJECTS any .ntmre compression tensor whose
    size_bytes != (n_elements / 32) * 34. The old writer stored the raw
    factors (u32 rank + Q8(A) + Q8(B)), which is smaller but unreadable
    by either reader -- that is the "legacy .ntmre compression" the header warns
    about. So the factors are folded back into a denoised matrix
    (A @ B, truncated to 99.9% energy) and that matrix is Q8-quantized
    flat. The rank is baked into the reconstruction, not stored.
    """
    m, n = W.shape
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    rank, energy = compute_optimal_rank(S)

    sqrt_s = np.sqrt(S[:rank])
    A = U[:, :rank] * sqrt_s[np.newaxis, :]     # (m, rank)
    B = Vt[:rank, :] * sqrt_s[:, np.newaxis]    # (rank, n)
    denoised = A @ B                            # (m, n), noise truncated

    q8_data, _ = quantize_q8(denoised)
    return q8_data, rank, energy


# ============================================================
# FORMAT LOADERS
# ============================================================

def load_gguf(path):
    """Load a GGUF file → dict of {tensor_name: numpy_array(f32)}.
    Handles Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, F16, F32, BF16.
    """
    print("  [GGUF] Parsing header...")
    
    with open(path, 'rb') as f:
        # Magic
        magic = f.read(4)
        if magic != b'GGUF':
            raise ValueError(f"Not a GGUF file (magic: {magic})")
        
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        
        print(f"  [GGUF] Version {version}, {n_tensors} tensors, {n_kv} KV pairs")
        
        # --- Read KV metadata ---
        metadata = {}
        for _ in range(n_kv):
            key = _read_gguf_string(f)
            val_type = struct.unpack('<I', f.read(4))[0]
            val = _read_gguf_value(f, val_type)
            metadata[key] = val
        
        # --- Read tensor info ---
        tensor_infos = []
        for _ in range(n_tensors):
            name = _read_gguf_string(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tensor_infos.append((name, dims, dtype, offset))
        
        # Alignment
        alignment = metadata.get('general.alignment', 32)
        
        # Data section starts after header (aligned)
        header_end = f.tell()
        data_start = header_end + (alignment - (header_end % alignment)) % alignment
        
        # --- Read and dequantize tensors ---
        tensors = {}
        print(f"  [GGUF] Loading {n_tensors} tensors...")
        
        for name, dims, dtype, offset in tensor_infos:
            f.seek(data_start + offset)
            n_elements = 1
            for d in dims:
                n_elements *= d
            
            tensor = _read_gguf_tensor(f, dtype, n_elements, dims)
            tensors[name] = tensor
        
        return tensors, metadata


# GGUF dtype constants
GGUF_TYPE_F32  = 0
GGUF_TYPE_F16  = 1
GGUF_TYPE_Q4_0 = 2
GGUF_TYPE_Q4_1 = 3
GGUF_TYPE_Q5_0 = 6
GGUF_TYPE_Q5_1 = 7
GGUF_TYPE_Q8_0 = 8
GGUF_TYPE_Q8_1 = 9
GGUF_TYPE_Q2_K = 10
GGUF_TYPE_Q3_K = 11
GGUF_TYPE_Q4_K = 12
GGUF_TYPE_Q5_K = 13
GGUF_TYPE_Q6_K = 14
GGUF_TYPE_BF16 = 30


def _read_gguf_string(f):
    """Read a GGUF string (uint64 length + bytes)."""
    length = struct.unpack('<Q', f.read(8))[0]
    return f.read(length).decode('utf-8')


def _read_gguf_value(f, val_type):
    """Read a GGUF metadata value by type."""
    # Types: 0=uint8, 1=int8, 2=uint16, 3=int16, 4=uint32, 5=int32,
    #         6=float32, 7=bool, 8=string, 9=array, 10=uint64, 11=int64, 12=float64
    if val_type == 0: return struct.unpack('<B', f.read(1))[0]
    elif val_type == 1: return struct.unpack('<b', f.read(1))[0]
    elif val_type == 2: return struct.unpack('<H', f.read(2))[0]
    elif val_type == 3: return struct.unpack('<h', f.read(2))[0]
    elif val_type == 4: return struct.unpack('<I', f.read(4))[0]
    elif val_type == 5: return struct.unpack('<i', f.read(4))[0]
    elif val_type == 6: return struct.unpack('<f', f.read(4))[0]
    elif val_type == 7: return bool(f.read(1)[0])
    elif val_type == 8: return _read_gguf_string(f)
    elif val_type == 9:
        arr_type = struct.unpack('<I', f.read(4))[0]
        arr_len = struct.unpack('<Q', f.read(8))[0]
        return [_read_gguf_value(f, arr_type) for _ in range(arr_len)]
    elif val_type == 10: return struct.unpack('<Q', f.read(8))[0]
    elif val_type == 11: return struct.unpack('<q', f.read(8))[0]
    elif val_type == 12: return struct.unpack('<d', f.read(8))[0]
    else:
        return None


def _dequant_q8_0(raw_bytes, n_elements):
    """Dequantize Q8_0 blocks → float32."""
    n_blocks = (n_elements + 31) // 32
    out = np.zeros(n_blocks * 32, dtype=np.float32)
    for b in range(n_blocks):
        base = b * 34
        scale = np.frombuffer(raw_bytes[base:base+2], dtype=np.float16).astype(np.float32)[0]
        quants = np.frombuffer(raw_bytes[base+2:base+34], dtype=np.int8).astype(np.float32)
        if not np.isfinite(scale):
            scale = 0.0
        out[b*32:(b+1)*32] = quants * scale
    return out[:n_elements]


def _dequant_q4_0(raw_bytes, n_elements):
    """Dequantize Q4_0 blocks → float32. 18 bytes per 32 elements."""
    n_blocks = (n_elements + 31) // 32
    out = np.zeros(n_blocks * 32, dtype=np.float32)
    for b in range(n_blocks):
        base = b * 18
        scale = np.frombuffer(raw_bytes[base:base+2], dtype=np.float16).astype(np.float32)[0]
        packed = np.frombuffer(raw_bytes[base+2:base+18], dtype=np.uint8)
        for i in range(16):
            low = int(packed[i] & 0x0F) - 8
            high = int(packed[i] >> 4) - 8
            out[b*32 + i*2] = scale * low
            out[b*32 + i*2 + 1] = scale * high
    return out[:n_elements]


def _dequant_f16(raw_bytes, n_elements):
    """F16 → float32."""
    return np.frombuffer(raw_bytes[:n_elements*2], dtype=np.float16).astype(np.float32)


def _dequant_bf16(raw_bytes, n_elements):
    """BF16 → float32."""
    raw = np.frombuffer(raw_bytes[:n_elements*2], dtype=np.uint16)
    # BF16: just shift left by 16 bits into float32
    f32_int = raw.astype(np.uint32) << 16
    return f32_int.view(np.float32)


def _read_gguf_tensor(f, dtype, n_elements, dims):
    """Read and dequantize a GGUF tensor to float32."""
    if dtype == GGUF_TYPE_F32:
        data = f.read(n_elements * 4)
        arr = np.frombuffer(data, dtype=np.float32).copy()
    elif dtype == GGUF_TYPE_F16:
        data = f.read(n_elements * 2)
        arr = _dequant_f16(data, n_elements)
    elif dtype == GGUF_TYPE_BF16:
        data = f.read(n_elements * 2)
        arr = _dequant_bf16(data, n_elements)
    elif dtype == GGUF_TYPE_Q8_0:
        n_blocks = (n_elements + 31) // 32
        data = f.read(n_blocks * 34)
        arr = _dequant_q8_0(data, n_elements)
    elif dtype == GGUF_TYPE_Q4_0:
        n_blocks = (n_elements + 31) // 32
        data = f.read(n_blocks * 18)
        arr = _dequant_q4_0(data, n_elements)
    else:
        # All other quant types (Q4_K, Q5_K, Q6_K, Q5_0, Q5_1, Q8_1,
        # Q2_K, Q3_K, IQ*, I8, I16, etc.) — use the suite's dequant_fast
        # module which handles all 29 GGUF quant types.
        try:
            from dequant_fast import dequantize_tensor
            arr = dequantize_tensor(f, n_elements, dtype)
        except Exception as e:
            print(f"      Dequant failed for type {dtype}: {e} — using zeros")
            # Fallback: skip the right number of bytes and use zeros
            bytes_per_block = {
                GGUF_TYPE_Q4_1: 20, GGUF_TYPE_Q5_0: 22, GGUF_TYPE_Q5_1: 24,
                GGUF_TYPE_Q8_1: 36, GGUF_TYPE_Q2_K: 256, GGUF_TYPE_Q3_K: 256,
                GGUF_TYPE_Q4_K: 144, GGUF_TYPE_Q5_K: 176, GGUF_TYPE_Q6_K: 210,
            }.get(dtype, 34)
            elements_per_block = 32 if dtype <= 9 else 256
            n_blocks = (n_elements + elements_per_block - 1) // elements_per_block
            f.read(n_blocks * bytes_per_block)  # skip
            arr = np.zeros(n_elements, dtype=np.float32)
    
    # Reshape
    if len(dims) == 1:
        return arr[:dims[0]]
    elif len(dims) == 2:
        return arr[:dims[0]*dims[1]].reshape(dims[1], dims[0]).T  # GGUF stores transposed
    else:
        shape = list(reversed(dims))
        return arr[:n_elements].reshape(shape)


def load_safetensors(path):
    """Load a SafeTensors file → dict of {name: numpy_array(f32)}."""
    print("  [SafeTensors] Loading...")
    
    with open(path, 'rb') as f:
        # Header: uint64 length of JSON header
        header_len = struct.unpack('<Q', f.read(8))[0]
        header_json = f.read(header_len).decode('utf-8')
        header = json.loads(header_json)
        
        data_start = 8 + header_len
        tensors = {}
        
        for name, info in header.items():
            if name == '__metadata__':
                continue
            dtype_str = info['dtype']
            shape = info['shape']
            offsets = info['data_offsets']
            start, end = offsets
            
            f.seek(data_start + start)
            raw = f.read(end - start)
            
            if dtype_str == 'F32':
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shape)
            elif dtype_str == 'F16':
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
            elif dtype_str == 'BF16':
                raw_u16 = np.frombuffer(raw, dtype=np.uint16)
                arr = (raw_u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
            elif dtype_str == 'I32':
                arr = np.frombuffer(raw, dtype=np.int32).reshape(shape)
            elif dtype_str == 'I64':
                arr = np.frombuffer(raw, dtype=np.int64).reshape(shape)
            else:
                print(f"      Unknown dtype '{dtype_str}' for {name}, skipping")
                continue
            
            tensors[name] = arr.copy()
        
        print(f"  [SafeTensors] Loaded {len(tensors)} tensors")
        return tensors


def load_pytorch(path):
    """Load a PyTorch .bin/.pt state_dict → dict of {name: numpy_array(f32)}.
    Uses numpy-only approach — no torch dependency.
    """
    print("  [PyTorch] Attempting to load...")
    
    # Try importing torch (might not be available)
    try:
        import torch
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        if hasattr(state_dict, 'state_dict'):
            state_dict = state_dict.state_dict()
        
        tensors = {}
        for name, tensor in state_dict.items():
            tensors[name] = tensor.float().numpy()
        
        print(f"  [PyTorch] Loaded {len(tensors)} tensors")
        return tensors
    except ImportError:
        print("  [PyTorch] torch not installed  trying pickle fallback...")
        # Basic pickle-based approach for simple state dicts
        import pickle
        import io
        
        # PyTorch .bin files are ZIP archives with pickled tensors
        import zipfile
        if zipfile.is_zipfile(path):
            print("  [PyTorch] ZIP-based format detected")
            raise NotImplementedError(
                "PyTorch ZIP format requires torch. Install: pip install torch\n"
                "Or convert to safetensors first: python -m safetensors.torch convert"
            )
        else:
            raise ValueError(f"Cannot load {path} without PyTorch installed")


def load_npy_checkpoint(directory):
    """Load a .npy checkpoint directory → dict of {name: numpy_array}."""
    print(f"  [NPY] Loading from: {directory}")
    directory = Path(directory)
    tensors = {}
    
    # Load config
    config_path = directory / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    
    # Load top-level tensors
    for npy_file in directory.glob("*.npy"):
        name = npy_file.stem
        tensors[name] = np.load(str(npy_file)).astype(np.float32)
    
    # Load layer tensors
    layers_dir = directory / "layers"
    if layers_dir.exists():
        for layer_dir in sorted(layers_dir.iterdir()):
            if layer_dir.is_dir():
                layer_num = layer_dir.name
                for npy_file in layer_dir.glob("*.npy"):
                    name = f"layers.{layer_num}.{npy_file.stem}"
                    tensors[name] = np.load(str(npy_file)).astype(np.float32)
    
    print(f"  [NPY] Loaded {len(tensors)} tensors")
    return tensors


# ============================================================
# NAME NORMALIZATION
# ============================================================

def normalize_tensor_name(name):
    """Normalize various naming conventions to a standard format.
    
    Handles: HuggingFace, GGUF, NightmareAI, LLaMA naming.
    Output: layers.{N}.{component} format.
    """
    # Common substitutions
    name = name.replace('model.', '')
    name = name.replace('transformer.', '')
    name = name.replace('h.', 'layers.')
    name = name.replace('blk.', 'layers.')
    
    # HuggingFace LLaMA naming
    name = name.replace('self_attn.q_proj', 'wq')
    name = name.replace('self_attn.k_proj', 'wk')
    name = name.replace('self_attn.v_proj', 'wv')
    name = name.replace('self_attn.o_proj', 'wo')
    name = name.replace('mlp.gate_proj', 'w_gate')
    name = name.replace('mlp.up_proj', 'w_up')
    name = name.replace('mlp.down_proj', 'w_down')
    name = name.replace('input_layernorm', 'attn_norm')
    name = name.replace('post_attention_layernorm', 'ffn_norm')
    
    # GGUF naming
    name = name.replace('attn_q', 'wq')
    name = name.replace('attn_k', 'wk')
    name = name.replace('attn_v', 'wv')
    name = name.replace('attn_output', 'wo')
    name = name.replace('ffn_gate', 'w_gate')
    name = name.replace('ffn_up', 'w_up')
    name = name.replace('ffn_down', 'w_down')
    name = name.replace('attn_norm', 'attn_norm')
    name = name.replace('ffn_norm', 'ffn_norm')
    
    # Embeddings
    name = name.replace('embed_tokens', 'embed')
    name = name.replace('token_embd', 'embed')
    name = name.replace('tok_emb', 'embed')
    name = name.replace('lm_head', 'output')
    name = name.replace('output.weight', 'output')
    name = name.replace('norm.weight', 'output_norm')
    name = name.replace('final_norm', 'output_norm')
    
    # Remove trailing .weight
    if name.endswith('.weight'):
        name = name[:-7]
    
    return name


# ============================================================
# ============================================================
# NTMRE v13 WRITER
# ============================================================
# v13 is final (Caleb's call, 2026-10-26): one format, one version.
# Layout per engine/ntmre_format.h (the C definition), matched by
# ntmre_runtime._read_header_and_tokenizer and GPUStreamEngine:
#   [0]   file header    16 bytes: magic NTMR, version 13, flags 0,
#         header_size (= data_offset, bytes before tensor data)
#   [16]  config        324 bytes: 36-byte base + 288-byte extensions
#   [340] tokenizer hdr  40 bytes: counts, ids, section offsets
#   [380] token strings: u16 len + bytes each, then scores (f32 each),
#         then merges (u32, u32 each)
#   [...] u32 n_tensors, then 91-byte tensor infos
#   [...] padding to 64-byte alignment -> data section
#   [data_offset] tensor data, each tensor 64-byte aligned
#
# v13 dtype codes (ntmre_format.h) -- NOT the old writer's codes:
#   F32 = 0, F16 = 1, Q8 = 2, Q4 = 3, .ntmre compression = 4
# The old writer used 0=F32, 1=Q8, 2=.ntmre compression, which v13 readers
# misread as F32/F16/Q8.

NTMRE_MAGIC = b'NTMR'
NTMRE_VERSION = 13
NTMRE_ALIGN = 64

# v13 dtype codes (ntmre_format.h)
DT_F32 = 0
DT_F16 = 1
DT_Q8 = 2
DT_Q4 = 3
DT_SVD_Q8 = 4

# converter's type codes (kept for convert_model and callers)
TENSOR_F32 = 0
TENSOR_Q8 = 1
TENSOR_SVD_Q8 = 2

# converter's type codes -> v13 dtype codes
_TYPE_TO_DTYPE = {TENSOR_F32: DT_F32, TENSOR_Q8: DT_Q8, TENSOR_SVD_Q8: DT_SVD_Q8}


def _align_up(n, align=NTMRE_ALIGN):
    return ((n + align - 1) // align) * align


def _build_config_bytes(config):
    """324-byte v13 config: 36-byte base + 288-byte extensions."""
    dim = int(config.get('dim', 0))
    n_layers = int(config.get('n_layers', 0))
    n_heads = int(config.get('n_heads', 0))
    n_kv_heads = int(config.get('n_kv_heads', 0))
    hidden_dim = int(config.get('hidden_dim', 0))
    vocab_size = int(config.get('vocab_size', 0))
    max_seq_len = int(config.get('max_seq_len', 1024))
    rope_base = float(config.get('rope_base', 10000.0))
    head_dim = int(config.get('head_dim', 0)) or (dim // n_heads if dim and n_heads else 0)
    base = struct.pack('<IIIIIIIfI', dim, n_layers, n_heads, n_kv_heads,
                       hidden_dim, vocab_size, max_seq_len, rope_base, head_dim)
    arch_type = int(config.get('arch_type', 0))
    q_output_dim = int(config.get('q_output_dim', 0)) or (n_heads * head_dim if head_dim else dim)
    sliding_window = int(config.get('sliding_window', 0))
    softcap = float(config.get('softcap', 0.0))
    shared_kv_layers = int(config.get('shared_kv_layers', 0))
    per_layer_hidden_count = 0
    # The C loader reads the BPE pre-tokenizer code from _pad_v2[0].
    pre_type = int(config.get('pre_type', 0))
    ext = struct.pack('<IIIfIIII', arch_type, q_output_dim, sliding_window,
                      softcap, shared_kv_layers, per_layer_hidden_count,
                      pre_type, 0)
    ext += struct.pack('<64I', *([0] * 64))  # per_layer_hidden_dims (unused, count is 0)
    return base + ext


def _build_tokenizer_bytes(tokenizer):
    """Tokenizer section: 40-byte header + strings + scores + merges.
    tokenizer is a dict with tokens (list[str]), scores (list[float]),
    merges (list[(int, int)]), bos_id, eos_id, pad_id, unk_id; or None
    for an empty tokenizer (n_tokens=0, n_merges=0)."""
    tok = tokenizer or {}
    tokens = tok.get('tokens', [])
    scores = tok.get('scores', [0.0] * len(tokens))
    merges = tok.get('merges', [])
    token_bytes = bytearray()
    for t in tokens:
        tb = t.encode('utf-8')
        token_bytes += struct.pack('<H', len(tb)) + tb
    tokens_size = len(token_bytes)
    scores_offset = tokens_size
    merges_offset = tokens_size + len(tokens) * 4
    hdr = struct.pack('<IIIIIIIIII',
                      len(tokens), len(merges),
                      int(tok.get('bos_id', 0)), int(tok.get('eos_id', 0)),
                      int(tok.get('pad_id', 0)), int(tok.get('unk_id', 0)),
                      0, scores_offset, merges_offset, tokens_size)
    out = bytearray(hdr)
    out += token_bytes
    out += struct.pack('<%df' % len(tokens),
                       *[float(s) for s in scores[:len(tokens)]])
    for a, b in merges:
        out += struct.pack('<II', int(a), int(b))
    return bytes(out)


def write_ntmre_svd(output_path, config, tensors_packed, tokenizer=None):
    """Write a v13 .ntmre file (NTMR magic, version 13, binary layout).

    tensors_packed: list of (name, ttype, shape, data) where ttype is the
    converter's code (TENSOR_F32=0, TENSOR_Q8=1, TENSOR_.ntmre compression=2), shape
    is a tuple, and data is bytes. Q8/.ntmre compression data must be flat Q8 blocks
    of the full tensor (see svd_compress); the C loader rejects any
    Q8-family tensor whose size != (n_elements / 32) * 34, so a tensor
    whose element count is not a multiple of 32 is refused here with a
    clear error instead of writing a file the loader will skip.
    """
    entries = []
    for name, ttype, shape, data in tensors_packed:
        if ttype not in _TYPE_TO_DTYPE:
            raise ValueError(f"unknown tensor type {ttype} for {name}")
        dtype = _TYPE_TO_DTYPE[ttype]
        shape = tuple(shape)
        n_elements = 1
        for d in shape:
            n_elements *= d
        if len(shape) == 1:
            n_dims, dim0, dim1 = 1, shape[0], 0
        elif len(shape) == 2:
            n_dims, dim0, dim1 = 2, shape[0], shape[1]
        else:
            # v13 tensor info only carries two dims; flatten the rest
            # (quantize_q8 already flattened the data, so the count agrees)
            n_dims, dim0, dim1 = 2, shape[0], n_elements // shape[0]
        if dtype in (DT_Q8, DT_SVD_Q8):
            if n_elements % 32 != 0:
                raise ValueError(
                    f"{name}: {n_elements} elements is not a multiple of 32; "
                    "Q8 blocks are 32 elements and the C loader rejects "
                    "tensors whose size != (n_elements / 32) * 34")
            need = (n_elements // 32) * 34
            if len(data) != need:
                raise ValueError(
                    f"{name}: Q8 data is {len(data)} bytes but the v13 layout "
                    f"needs {need} bytes for {n_elements} elements")
        elif dtype == DT_F32:
            if len(data) != n_elements * 4:
                raise ValueError(
                    f"{name}: F32 data is {len(data)} bytes but {n_elements} "
                    f"elements need {n_elements * 4}")
        name_bytes = name.encode('utf-8')
        if len(name_bytes) > 64:
            raise ValueError(f"tensor name too long for v13 (max 64 bytes): {name}")
        entries.append((name_bytes, dtype, n_dims, dim0, dim1, bytes(data)))

    config_bytes = _build_config_bytes(config or {})
    tok_bytes = _build_tokenizer_bytes(tokenizer)

    # Lay out the file to find data_offset (header_size).
    header_len = 16 + len(config_bytes) + len(tok_bytes) + 4 + len(entries) * 91
    data_offset = _align_up(header_len)
    offsets = []
    pos = 0
    for (_, _, _, _, _, data) in entries:
        pos = _align_up(pos)
        offsets.append(pos)
        pos += len(data)

    with open(output_path, 'wb') as f:
        f.write(struct.pack('<IIII', 0x4E544D52, NTMRE_VERSION, 0, data_offset))
        f.write(config_bytes)
        f.write(tok_bytes)
        f.write(struct.pack('<I', len(entries)))
        for (name_bytes, dtype, n_dims, dim0, dim1, data), offset in zip(entries, offsets):
            f.write(struct.pack('<B', len(name_bytes)))
            f.write(name_bytes + b'\x00' * (64 - len(name_bytes)))
            f.write(struct.pack('<BBIIQQ', dtype, n_dims, dim0, dim1, offset, len(data)))
        pad = data_offset - f.tell()
        if pad:
            f.write(b'\x00' * pad)
        for (_, _, _, _, _, data), offset in zip(entries, offsets):
            cur = f.tell()
            want = data_offset + offset
            if cur < want:
                f.write(b'\x00' * (want - cur))
            f.write(data)


def extract_tokenizer_from_gguf(metadata):
    """Build a v13 tokenizer dict from GGUF metadata, or None if the
    metadata carries no tokenizer. GGUF merges are "left right" strings
    (SentencePiece pieces, no raw spaces); each side is looked up in the
    token list, and merges that do not resolve are skipped."""
    tokens = metadata.get('tokenizer.ggml.tokens') if metadata else None
    if not tokens:
        return None
    scores = metadata.get('tokenizer.ggml.scores') or [0.0] * len(tokens)
    tok_to_id = {t: i for i, t in enumerate(tokens)}
    merge_pairs = []
    for m in (metadata.get('tokenizer.ggml.merges') or []):
        parts = m.split(' ', 1)
        if len(parts) != 2:
            continue
        a = tok_to_id.get(parts[0])
        b = tok_to_id.get(parts[1])
        if a is None or b is None:
            continue
        merge_pairs.append((a, b))
    return {
        'tokens': list(tokens),
        'scores': [float(s) for s in scores[:len(tokens)]],
        'merges': merge_pairs,
        'bos_id': int(metadata.get('tokenizer.ggml.bos_token_id', 1)),
        'eos_id': int(metadata.get('tokenizer.ggml.eos_token_id', 2)),
        'pad_id': int(metadata.get('tokenizer.ggml.padding_token_id', 0)),
        'unk_id': int(metadata.get('tokenizer.ggml.unknown_token_id', 0)),
    }


# ============================================================
# TOKENIZER FILES (non-GGUF sources)
# ============================================================
# GGUF carries its tokenizer in the metadata. SafeTensors, PyTorch and
# .npy checkpoints do not -- their tokenizer lives in files next to the
# model (the HuggingFace layout). These loaders read those files with
# the stdlib only, and return the same v13 tokenizer dict the writer
# takes: tokens, scores, merges, bos_id, eos_id, pad_id, unk_id.

def _load_tokenizer_json(path):
    """HuggingFace tokenizer.json -> v13 tokenizer dict. Vocab is a
    token->id dict (inverted to an id-ordered list); merges are "left
    right" strings; added tokens (special/chat tokens) live outside the
    vocab and are merged in by id. HF tokenizers carry no per-token
    scores, so scores are 0.0."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    model = data.get('model', {})
    vocab = model.get('vocab', {})
    tokens = [''] * len(vocab)
    for tok, i in vocab.items():
        if isinstance(i, int) and 0 <= i < len(tokens):
            tokens[i] = tok
    tok_to_id = {t: i for i, t in enumerate(tokens) if t}
    merges = []
    for m in (model.get('merges') or []):
        parts = m.split(' ', 1)
        if len(parts) != 2:
            continue
        a = tok_to_id.get(parts[0])
        b = tok_to_id.get(parts[1])
        if a is None or b is None:
            continue
        merges.append((a, b))
    added = data.get('added_tokens', [])
    max_id = max([i for i in vocab.values() if isinstance(i, int)] +
                 [t.get('id', 0) for t in added if isinstance(t.get('id'), int)] + [-1])
    if max_id >= len(tokens):
        tokens.extend([''] * (max_id + 1 - len(tokens)))
    for t in added:
        i = t.get('id')
        if isinstance(i, int) and 0 <= i < len(tokens):
            tokens[i] = t.get('content', '')
    return {'tokens': tokens, 'scores': [0.0] * len(tokens), 'merges': merges}


def _load_gpt2_tokenizer(model_dir):
    """vocab.json + merges.txt (GPT-2 BPE) -> v13 tokenizer dict.
    No scores in this format, so scores are 0.0."""
    vocab_path = os.path.join(model_dir, 'vocab.json')
    if not os.path.exists(vocab_path):
        return None
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    tokens = [''] * len(vocab)
    for tok, i in vocab.items():
        if isinstance(i, int) and 0 <= i < len(tokens):
            tokens[i] = tok
    tok_to_id = {t: i for i, t in enumerate(tokens) if t}
    merges = []
    merges_path = os.path.join(model_dir, 'merges.txt')
    if os.path.exists(merges_path):
        with open(merges_path, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                line = line.rstrip('\n')
                if not line or line.startswith('#'):
                    continue
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    continue
                a = tok_to_id.get(parts[0])
                b = tok_to_id.get(parts[1])
                if a is None or b is None:
                    continue
                merges.append((a, b))
    return {'tokens': tokens, 'scores': [0.0] * len(tokens), 'merges': merges}


def _read_varint(data, pos):
    """Read a protobuf varint at pos. Returns (value, new_pos)."""
    result = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            break
        shift += 7
    return result, pos


def _load_sentencepiece_tokenizer(path):
    """SentencePiece .model (tokenizer.model / spiece.model) -> v13
    tokenizer dict. Minimal protobuf reader, stdlib only: ModelProto
    field 1 is the repeated pieces, each a message with field 1 = piece
    (string), field 2 = score (float32), field 3 = type (varint).
    Merges are empty -- SentencePiece is a unigram model, there are no
    BPE merges to store, and the encoder uses scores instead."""
    data = open(path, 'rb').read()
    tokens = []
    scores = []
    pos = 0
    while pos < len(data):
        tag, pos = _read_varint(data, pos)
        field = tag >> 3
        wire = tag & 0x07
        if field == 1 and wire == 2:
            ln, pos = _read_varint(data, pos)
            piece_data = data[pos:pos + ln]
            pos += ln
            ppos = 0
            piece = None
            score = 0.0
            while ppos < len(piece_data):
                ptag, ppos = _read_varint(piece_data, ppos)
                pfield = ptag >> 3
                pwire = ptag & 0x07
                if pfield == 1 and pwire == 2:
                    ln2, ppos = _read_varint(piece_data, ppos)
                    piece = piece_data[ppos:ppos + ln2].decode('utf-8', errors='replace')
                    ppos += ln2
                elif pfield == 2 and pwire == 5:
                    score = struct.unpack('<f', piece_data[ppos:ppos + 4])[0]
                    ppos += 4
                elif pwire == 0:
                    _, ppos = _read_varint(piece_data, ppos)
                elif pwire == 2:
                    ln2, ppos = _read_varint(piece_data, ppos)
                    ppos += ln2
                elif pwire == 5:
                    ppos += 4
                elif pwire == 1:
                    ppos += 8
                else:
                    break
            if piece is not None:
                tokens.append(piece)
                scores.append(score)
        elif wire == 0:
            _, pos = _read_varint(data, pos)
        elif wire == 2:
            ln, pos = _read_varint(data, pos)
            pos += ln
        elif wire == 5:
            pos += 4
        elif wire == 1:
            pos += 8
        else:
            break
    if not tokens:
        return None
    return {'tokens': tokens, 'scores': scores, 'merges': []}


def _load_wordpiece_tokenizer(path):
    """vocab.txt (BERT WordPiece, one token per line) -> v13 tokenizer
    dict. No scores, no merges in this format."""
    tokens = []
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            tokens.append(line.rstrip('\n'))
    if not tokens:
        return None
    return {'tokens': tokens, 'scores': [0.0] * len(tokens), 'merges': []}


def _special_ids_from_config(model_dir, tokens):
    """Read bos/eos/pad/unk ids from tokenizer_config.json if present.
    Returns a dict with None for anything not found (so the caller can
    fall back to well-known special tokens instead of guessing)."""
    ids = {'bos_id': None, 'eos_id': None, 'pad_id': None, 'unk_id': None}
    cfg_path = os.path.join(model_dir, 'tokenizer_config.json')
    if not os.path.exists(cfg_path):
        return ids
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception:
        return ids
    tok_to_id = {}
    for i, t in enumerate(tokens):
        if t and t not in tok_to_id:
            tok_to_id[t] = i
    for key, field in (('bos_token', 'bos_id'), ('eos_token', 'eos_id'),
                       ('pad_token', 'pad_id'), ('unk_token', 'unk_id')):
        v = cfg.get(key)
        if isinstance(v, dict):
            v = v.get('content')
        if isinstance(v, str) and v in tok_to_id:
            ids[field] = tok_to_id[v]
    return ids


def find_tokenizer(model_dir):
    """Find and load a tokenizer from the files next to a model.
    Priority: tokenizer.json, vocab.json (+merges.txt), tokenizer.model,
    spiece.model, vocab.txt. Returns a v13 tokenizer dict or None."""
    model_dir = str(model_dir)
    tok = None
    if os.path.exists(os.path.join(model_dir, 'tokenizer.json')):
        tok = _load_tokenizer_json(os.path.join(model_dir, 'tokenizer.json'))
    elif os.path.exists(os.path.join(model_dir, 'vocab.json')):
        tok = _load_gpt2_tokenizer(model_dir)
    elif os.path.exists(os.path.join(model_dir, 'tokenizer.model')):
        tok = _load_sentencepiece_tokenizer(os.path.join(model_dir, 'tokenizer.model'))
    elif os.path.exists(os.path.join(model_dir, 'spiece.model')):
        tok = _load_sentencepiece_tokenizer(os.path.join(model_dir, 'spiece.model'))
    elif os.path.exists(os.path.join(model_dir, 'vocab.txt')):
        tok = _load_wordpiece_tokenizer(os.path.join(model_dir, 'vocab.txt'))
    if tok is None:
        return None
    ids = _special_ids_from_config(model_dir, tok['tokens'])
    tok_to_id = {t: i for i, t in enumerate(tok['tokens']) if t}
    for field, names in (('bos_id', ['<s>', '<|startoftext|>']),
                         ('eos_id', ['</s>', '<|endoftext|>']),
                         ('pad_id', ['<pad>', '<|endoftext|>']),
                         ('unk_id', ['<unk>'])):
        if ids[field] is None:
            for n in names:
                if n in tok_to_id:
                    ids[field] = tok_to_id[n]
                    break
        tok[field] = ids[field] if ids[field] is not None else 0
    return tok


# ============================================================
# MAIN CONVERSION PIPELINE
# ============================================================

def convert_model(input_path, output_path=None, config_override=None):
    """Universal model converter.
    
    Auto-detects format, loads, compresses with .ntmre compression, outputs .ntmre.
    """
    print()
    print("  ========================================")
    print("    NIGHTMARE UNIVERSAL CONVERTER")
    print("    Any model -> .ntmre compression .ntmre")
    print("    Nightmare Designs")
    print("  ========================================")
    print()
    
    input_path = Path(input_path)
    metadata = None

    # --- Detect format and load ---
    if input_path.is_dir():
        tensors = load_npy_checkpoint(str(input_path))
        source_format = "npy"
    elif input_path.suffix == '.gguf':
        tensors, metadata = load_gguf(str(input_path))
        source_format = "gguf"
    elif input_path.suffix == '.safetensors':
        tensors = load_safetensors(str(input_path))
        source_format = "safetensors"
    elif input_path.suffix in ('.bin', '.pt', '.pth'):
        # Check if it's a raw flat binary (from Crucible) or PyTorch
        if config_override:
            # Assume flat binary if config is provided
            tensors = None  # Handle below
            source_format = "flat_bin"
        else:
            try:
                tensors = load_pytorch(str(input_path))
                source_format = "pytorch"
            except:
                print("    Cannot determine format. Provide --config for flat binary.")
                return
    else:
        print(f"   Unknown format: {input_path.suffix}")
        print("  Supported: .gguf, .safetensors, .bin, .pt, directory/")
        return
    
    print(f"\n  Source format: {source_format}")
    print(f"  Tensors loaded: {len(tensors) if tensors else 'pending'}")
    
    # --- Infer config from tensors ---
    config = infer_config(tensors, config_override)
    if config:
        print(f"\n  Detected architecture:")
        print(f"    dim:        {config.get('dim', '?')}")
        print(f"    layers:     {config.get('n_layers', '?')}")
        print(f"    heads:      {config.get('n_heads', '?')}")
        print(f"    kv_heads:   {config.get('n_kv_heads', '?')}")
        print(f"    hidden_dim: {config.get('hidden_dim', '?')}")
        print(f"    vocab:      {config.get('vocab_size', '?')}")
    
    # --- Normalize names ---
    normalized = {}
    for name, weight in tensors.items():
        norm_name = normalize_tensor_name(name)
        normalized[norm_name] = weight
    
    # --- Compress ---
    print(f"\n  Compressing {len(normalized)} tensors...")
    print("  " + "-" * 60)
    
    tensors_packed = []
    total_original = 0
    total_compressed = 0
    
    for name, weight in sorted(normalized.items()):
        weight = weight.astype(np.float32)
        original_size = weight.nbytes
        total_original += original_size
        
        if weight.ndim == 1:
            # 1D (norms) — keep F32
            data = weight.tobytes()
            tensors_packed.append((name, TENSOR_F32, weight.shape, data))
            total_compressed += len(data)
            
        elif 'embed' in name or name == 'output':
            # Embedding / output — Q8 only
            data, _ = quantize_q8(weight)
            tensors_packed.append((name, TENSOR_Q8, weight.shape, data))
            total_compressed += len(data)
            ratio = original_size / len(data)
            print(f"    {name:40s} Q8      {original_size/1024**2:7.2f} -> {len(data)/1024**2:7.2f} MB ({ratio:.1f})")
            
        elif weight.ndim == 2:
            # 2D weights — .ntmre compression
            combined, rank, energy = svd_compress(weight)
            tensors_packed.append((name, TENSOR_SVD_Q8, weight.shape, combined))
            total_compressed += len(combined)
            ratio = original_size / len(combined)
            print(f"    {name:40s} .ntmre compression  rank={rank:3d} e={energy:.4f} "
                  f"{original_size/1024**2:6.2f} → {len(combined)/1024**2:5.2f} MB ({ratio:.1f}×)")
        else:
            # Higher-dim (rare) — Q8
            data, _ = quantize_q8(weight)
            tensors_packed.append((name, TENSOR_Q8, weight.shape, data))
            total_compressed += len(data)
    
    print("  " + "-" * 60)
    
    # --- Write output ---
    if output_path is None:
        output_path = str(input_path.with_suffix('.ntmre'))
    
    # --- Tokenizer: GGUF carries it in metadata; every other format
    #     keeps it in files next to the model (tokenizer.json,
    #     vocab.json+merges.txt, tokenizer.model, vocab.txt). ---
    tokenizer = extract_tokenizer_from_gguf(metadata)
    if tokenizer is None:
        model_dir = input_path if input_path.is_dir() else input_path.parent
        tokenizer = find_tokenizer(model_dir)
    if tokenizer:
        print(f"\n  Tokenizer: {len(tokenizer['tokens'])} tokens, "
              f"{len(tokenizer['merges'])} merges")
    else:
        print("\n  Tokenizer: none found -- writing an empty tokenizer")

    print(f"\n  Writing: {output_path}")
    write_ntmre_svd(output_path, config or {}, tensors_packed, tokenizer=tokenizer)
    
    output_size = os.path.getsize(output_path)
    ratio = total_original / output_size
    
    print()
    print(f"  === CONVERSION COMPLETE ===")
    print(f"    Original:   {total_original / 1024**3:.2f} GB")
    print(f"    Compressed: {output_size / 1024**2:.1f} MB")
    print(f"    Ratio:      {ratio:.1f} smaller")
    print(f"    Output:     {output_path}")
    print()
    print("   Compressed. Ship it.")
    print()


def infer_config(tensors, config_override=None):
    """Infer model config from tensor shapes."""
    if config_override:
        if os.path.exists(config_override):
            with open(config_override) as f:
                raw = json.load(f)
            return raw.get('model', raw)
        return None
    
    if not tensors:
        return None
    
    config = {}
    
    # Find embedding to get vocab_size and dim
    for name, weight in tensors.items():
        norm_name = normalize_tensor_name(name)
        if 'embed' in norm_name and weight.ndim == 2:
            config['vocab_size'] = weight.shape[0]
            config['dim'] = weight.shape[1]
            break
    
    # Count layers
    layer_nums = set()
    for name in tensors.keys():
        norm_name = normalize_tensor_name(name)
        parts = norm_name.split('.')
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                try:
                    layer_nums.add(int(parts[i + 1]))
                except ValueError:
                    pass
    if layer_nums:
        config['n_layers'] = max(layer_nums) + 1
    
    # Infer heads from Q projection
    dim = config.get('dim', 0)
    for name, weight in tensors.items():
        norm_name = normalize_tensor_name(name)
        if 'wq' in norm_name and weight.ndim == 2:
            q_out = weight.shape[0] if weight.shape[1] == dim else weight.shape[1]
            config['n_heads'] = q_out // 128  # Assume head_dim=128
            config['head_dim'] = 128
            break
    
    # Infer kv_heads from K projection
    for name, weight in tensors.items():
        norm_name = normalize_tensor_name(name)
        if 'wk' in norm_name and weight.ndim == 2:
            k_out = weight.shape[0] if weight.shape[1] == dim else weight.shape[1]
            config['n_kv_heads'] = k_out // 128
            break
    
    # Infer hidden_dim from gate projection
    for name, weight in tensors.items():
        norm_name = normalize_tensor_name(name)
        if 'w_gate' in norm_name and weight.ndim == 2:
            config['hidden_dim'] = max(weight.shape)
            break
    
    config.setdefault('max_seq_len', 1024)
    config.setdefault('rope_base', 10000.0)
    
    return config


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python nightmare_convert.py <model_file> [options]")
        print()
        print("Supported formats:")
        print("  .gguf          GGUF (any quant  auto-dequantized)")
        print("  .safetensors   HuggingFace SafeTensors")
        print("  .bin / .pt     PyTorch or raw F32 checkpoint")
        print("  directory/     .npy checkpoint folder")
        print()
        print("Options:")
        print("  --output PATH    Output .ntmre path")
        print("  --config PATH    Model config JSON (for flat .bin files)")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = None
    config_override = None
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]; i += 2
        elif sys.argv[i] == "--config" and i + 1 < len(sys.argv):
            config_override = sys.argv[i + 1]; i += 2
        else:
            i += 1
    
    convert_model(input_path, output_path, config_override)
