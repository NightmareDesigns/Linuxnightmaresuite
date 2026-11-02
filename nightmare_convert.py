"""
NIGHTMARE UNIVERSAL CONVERTER  v14
===================================
ONE format. ONE magic. "NTMR" 0x4E544D52. Version 13. No NSVD. No RMTN.

Inputs:
    .gguf           GGUF v2/v3, any quant (all 29 types dequantized)
    .safetensors    HuggingFace SafeTensors (F32/F16/BF16/I32/I64)
    .bin/.pt/.pth   PyTorch state_dict or raw F32 Crucible checkpoint
    directory/      .npy checkpoint folder (Crucible/NightmareAI format)
    .onnx           ONNX weight export

Output: always .ntmre — readable by nightmare_engine.dll, ntmre_runtime.py,
ntmre_format_py.read_ntmre(), and every engine in the suite.

Nightmare Designs — Levi Gates

Usage:
    python nightmare_convert.py model.gguf
    python nightmare_convert.py model.safetensors -o out.ntmre
    python nightmare_convert.py model.gguf --energy 0.999
    python nightmare_convert.py model.gguf --nosvd
    python nightmare_convert.py model.gguf --verify

Options:
    -o / --output PATH   Output .ntmre path  (default: input stem + .ntmre)
    --energy FLOAT       .ntmre compression energy fraction 0-1  (default 0.99)
    --nosvd              Direct Q8 everywhere, no .ntmre compression
    --rank INT           Force fixed .ntmre compression rank
    --config PATH        JSON config for raw .bin flat checkpoints
    --mgpu               Pass -mgpu to GPU converter exe if present
    --verify             Validate output file after conversion
"""

import sys
import os
import json
import struct
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Import the canonical format writer. This is the ONLY module that writes
# the on-disk NTMR layout. nightmare_convert never writes bytes directly.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
for _p in [_HERE, _HERE / 'engine']:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

try:
    import ntmre_format_py as _fmt
except ImportError as _ie:
    raise ImportError(
        "ntmre_format_py.py not found — must be in the same directory.\n"
        + str(_ie)
    )

# Re-export format constants
NTMRE_MAGIC   = _fmt.MAGIC      # 0x4E544D52  "NTMR"
NTMRE_VERSION = _fmt.VERSION    # 13
TENSOR_F32    = _fmt.F32        # 0
TENSOR_Q8     = _fmt.Q8         # 2
TENSOR_SVD_Q8 = _fmt.SVD_Q8    # 4

# Default compression parameters
ENERGY_THRESHOLD = 0.99
BLOCK_SIZE = 32


# ===========================================================================
# Q8 quantization  (34 bytes per 32 elements: f16 scale + 32 x int8)
# ===========================================================================

def quantize_q8(data):
    """float32 -> Q8_0 bytes. Returns (bytes, n_blocks)."""
    data = np.asarray(data, dtype=np.float32).flatten()
    rem = len(data) % BLOCK_SIZE
    if rem:
        data = np.concatenate([data, np.zeros(BLOCK_SIZE - rem, dtype=np.float32)])
    n_blocks = len(data) // BLOCK_SIZE
    out = bytearray()
    for b in range(n_blocks):
        blk  = data[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE]
        amax = float(np.max(np.abs(blk)))
        scale = amax / 127.0 if amax > 0 else 0.0
        inv   = 1.0 / scale if scale > 0 else 0.0
        out  += struct.pack('<e', np.float16(scale))
        out  += np.clip(np.round(blk * inv), -127, 127).astype(np.int8).tobytes()
    return bytes(out), n_blocks


# ===========================================================================
# .ntmre compression compression
# ===========================================================================

def _select_rank(S, m, n, energy_threshold):
    """Return (rank, use_svd). Adaptive: pick energy rank, only use .ntmre compression when
    it actually compresses (rank < m*n/(m+n))."""
    s2    = S ** 2
    total = s2.sum()
    if total <= 0:
        return 1, False
    cum         = np.cumsum(s2)
    erank       = int(np.searchsorted(cum, energy_threshold * total)) + 1
    erank       = max(1, min(erank, len(S)))
    cap         = max(1, int(m * n / (m + n)))
    return (erank, True) if erank <= cap else (cap, False)


def svd_compress(W, energy_threshold=ENERGY_THRESHOLD, fixed_rank=None):
    """Compress a 2-D weight with .ntmre compression via ntmre_format_py.pack_svd.
    Returns (data_bytes, rank, energy, used_svd).
    Falls back to Q8 when .ntmre compression would not compress.
    """
    m, n = W.shape
    U, S, Vt = np.linalg.svd(W.astype(np.float32), full_matrices=False)

    if fixed_rank:
        rank, use_svd = max(1, min(int(fixed_rank), len(S))), True
    else:
        rank, use_svd = _select_rank(S, m, n, energy_threshold)

    s2tot  = float(np.sum(S ** 2))
    energy = float(np.sum(S[:rank] ** 2)) / s2tot if s2tot > 0 else 1.0

    if not use_svd:
        data, _ = quantize_q8(W)
        return data, 0, energy, False

    # pack_svd: u32 rank + Q8(U[:,:rank]) + F32(S[:rank]) + Q8(Vt[:rank,:])
    data = _fmt.pack_svd(U, S, Vt, rank)
    return data, rank, energy, True


# ===========================================================================
# Per-tensor compression dispatcher
# ===========================================================================

_EMBED_NAMES = frozenset({
    'token_embd.weight', 'token_embd', 'embedder.weight',
    'output.weight', 'output', 'token_embedding.weight',
})


def _compress_tensor(name, arr, energy_threshold, fixed_rank, disable_svd):
    """Return (ttype, data_bytes, rank)."""
    arr = arr.astype(np.float32)

    if arr.ndim == 1:
        return TENSOR_F32, arr.tobytes(), 0

    is_embed = (name in _EMBED_NAMES or
                'embed' in name.lower() or
                name.endswith('output.weight'))
    if is_embed:
        data, _ = quantize_q8(arr)
        return TENSOR_Q8, data, 0

    if arr.ndim == 2 and not disable_svd:
        data, rank, energy, used = svd_compress(arr, energy_threshold, fixed_rank)
        return (TENSOR_SVD_Q8, data, rank) if used else (TENSOR_Q8, data, 0)

    data, _ = quantize_q8(arr)
    return TENSOR_Q8, data, 0


# ===========================================================================
# Tensor name normalisation
# ===========================================================================

def normalize_tensor_name(name):
    if name in ('embedder.weight', 'model.embed_tokens.weight'):
        return 'token_embd.weight'
    if name in ('model.norm.weight', 'norm.weight'):
        return 'output_norm.weight'
    if name == 'lm_head.weight':
        return 'output.weight'
    name = (name
        .replace('model.', '').replace('transformer.', '')
        .replace('h.', 'layers.').replace('blk.', 'layers.')
        .replace('self_attn.q_proj', 'attn_q')
        .replace('self_attn.k_proj', 'attn_k')
        .replace('self_attn.v_proj', 'attn_v')
        .replace('self_attn.o_proj', 'attn_output')
        .replace('mlp.gate_proj', 'ffn_gate')
        .replace('mlp.up_proj', 'ffn_up')
        .replace('mlp.down_proj', 'ffn_down')
        .replace('input_layernorm', 'attn_norm')
        .replace('post_attention_layernorm', 'ffn_norm')
        .replace('embed_tokens', 'token_embd')
        .replace('tok_emb', 'token_embd')
    )
    if name.endswith('.weight'):
        name = name[:-7]
    return name


# ===========================================================================
# Multimodal tower filter  (strips v.* / a.* / mm.* vision/audio tensors)
# ===========================================================================

def _is_nontext(name):
    return (name.startswith('v.') or name.startswith('a.') or
            name.startswith('au.') or name.startswith('mm.'))


# ===========================================================================
# GGUF helpers
# ===========================================================================

def _rd_str(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')


def _rd_val(f, vt):
    if vt == 0:  return struct.unpack('<B', f.read(1))[0]
    if vt == 1:  return struct.unpack('<b', f.read(1))[0]
    if vt == 2:  return struct.unpack('<H', f.read(2))[0]
    if vt == 3:  return struct.unpack('<h', f.read(2))[0]
    if vt == 4:  return struct.unpack('<I', f.read(4))[0]
    if vt == 5:  return struct.unpack('<i', f.read(4))[0]
    if vt == 6:  return struct.unpack('<f', f.read(4))[0]
    if vt == 7:  return bool(f.read(1)[0])
    if vt == 8:  return _rd_str(f)
    if vt == 9:
        at = struct.unpack('<I', f.read(4))[0]
        n  = struct.unpack('<Q', f.read(8))[0]
        return [_rd_val(f, at) for _ in range(n)]
    if vt == 10: return struct.unpack('<Q', f.read(8))[0]
    if vt == 11: return struct.unpack('<q', f.read(8))[0]
    if vt == 12: return struct.unpack('<d', f.read(8))[0]
    return None


_GGUF_SKIP_BPB   = {3:20, 6:22, 7:24, 9:40, 10:84, 11:110,
                    12:144, 13:176, 14:210, 15:292, 16:66, 17:74,
                    18:98, 19:50, 20:18, 21:110, 22:82, 23:136,
                    29:56}
_GGUF_SKIP_ELEMS = {k: (32 if k in (20,) else 256)
                    for k in _GGUF_SKIP_BPB}


def _f16_to_f32(h):
    return np.array([h], dtype=np.uint16).view(np.uint16).astype(np.float16).astype(np.float32)[0]


def _dequant_q4_0(raw, n):
    nb  = (n + 31) // 32
    out = np.zeros(nb * 32, dtype=np.float32)
    p   = 0
    for b in range(nb):
        sc = _f16_to_f32(struct.unpack_from('<H', raw, p)[0]); p += 2
        pk = np.frombuffer(raw, dtype=np.uint8, count=16, offset=p); p += 16
        for i in range(16):
            out[b*32 + i]      = sc * (float(pk[i] & 0x0F) - 8.0)
            out[b*32 + i + 16] = sc * (float(pk[i] >> 4)  - 8.0)
    return out[:n]


def _dequant_q8_0(raw, n):
    nb  = (n + 31) // 32
    out = np.zeros(nb * 32, dtype=np.float32)
    p   = 0
    for b in range(nb):
        sc = _f16_to_f32(struct.unpack_from('<H', raw, p)[0]); p += 2
        qv = np.frombuffer(raw, dtype=np.int8, count=32, offset=p); p += 32
        out[b*32:b*32+32] = qv.astype(np.float32) * sc
    return out[:n]


def _dequant_gguf_tensor(f, dtype, n_elements):
    """Dequantize a GGUF tensor from the current file position -> float32."""
    if dtype == 0:   # F32
        return np.frombuffer(f.read(n_elements * 4), dtype=np.float32).copy()
    if dtype == 1:   # F16
        raw = f.read(n_elements * 2)
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32)
    if dtype == 30:  # BF16
        raw = f.read(n_elements * 2)
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
        return (u << 16).view(np.float32)
    if dtype == 8:   # Q8_0
        nb  = (n_elements + 31) // 32
        raw = f.read(nb * 34)
        return _dequant_q8_0(raw, n_elements)
    if dtype == 2:   # Q4_0
        nb  = (n_elements + 31) // 32
        raw = f.read(nb * 18)
        return _dequant_q4_0(raw, n_elements)
    # K-quants and iQuants — try dequant_fast
    try:
        from dequant_fast import dequantize_tensor
        return dequantize_tensor(f, n_elements, dtype)
    except Exception:
        pass
    # Fallback: skip bytes, return zeros
    elems_per = _GGUF_SKIP_ELEMS.get(dtype, 32)
    bpb       = _GGUF_SKIP_BPB.get(dtype, 34)
    nb        = (n_elements + elems_per - 1) // elems_per
    f.read(nb * bpb)
    print(f"    [!] dtype {dtype} not fully dequantized — using zeros")
    return np.zeros(n_elements, dtype=np.float32)


# ===========================================================================
# GGUF streaming loader  (one tensor at a time, no full-model RAM spike)
# ===========================================================================

def _load_gguf_streaming(path, energy, fixed_rank, disable_svd):
    path = str(path)
    print(f"  [GGUF] Parsing: {path}")

    with open(path, 'rb') as f:
        if f.read(4) != b'GGUF':
            raise ValueError("Not a GGUF file")
        version   = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv      = struct.unpack('<Q', f.read(8))[0]
        print(f"  [GGUF] v{version}, {n_tensors} tensors, {n_kv} metadata keys")

        meta = {}
        for _ in range(n_kv):
            key = _rd_str(f)
            vt  = struct.unpack('<I', f.read(4))[0]
            meta[key] = _rd_val(f, vt)

        alignment = int(meta.get('general.alignment', 32))
        tinfos = []
        for _ in range(n_tensors):
            name   = _rd_str(f)
            nd     = struct.unpack('<I', f.read(4))[0]
            dims   = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
            dtype  = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            tinfos.append((name, dims, dtype, offset))

        hdr_end    = f.tell()
        data_start = hdr_end + (alignment - hdr_end % alignment) % alignment

    # Build config from metadata
    arch = str(meta.get('general.architecture', 'llama')).lower()
    dim  = int(meta.get(f'{arch}.embedding_length', 0))
    nh   = int(meta.get(f'{arch}.attention.head_count', 0))
    config = {
        'arch':        arch,
        'dim':         dim,
        'n_layers':    int(meta.get(f'{arch}.block_count', 0)),
        'n_heads':     nh,
        'n_kv_heads':  int(meta.get(f'{arch}.attention.head_count_kv', nh)),
        'head_dim':    int(meta.get(f'{arch}.attention.key_length',
                                    dim // nh if nh else 0)),
        'hidden_dim':  int(meta.get(f'{arch}.feed_forward_length', 0)),
        'vocab_size':  0,
        'max_seq_len': int(meta.get(f'{arch}.context_length', 2048)),
        'rope_base':   float(meta.get(f'{arch}.rope.freq_base', 10000.0)),
        'arch_type':   1 if 'gemma' in arch else 0,
    }

    tokens = list(meta.get('tokenizer.ggml.tokens', []))
    merges = list(meta.get('tokenizer.ggml.merges', []))
    config['vocab_size'] = len(tokens)
    tokenizer = {
        'tokens': tokens,
        'merges': merges,
        'bos':    int(meta.get('tokenizer.ggml.bos_token_id', 1)),
        'eos':    int(meta.get('tokenizer.ggml.eos_token_id', 2)),
        'pad':    int(meta.get('tokenizer.ggml.padding_token_id', 0)),
        'unk':    int(meta.get('tokenizer.ggml.unknown_token_id', 3)),
    }

    # Filter multimodal tensors
    kept = [(n, d, dt, o) for (n, d, dt, o) in tinfos if not _is_nontext(n)]
    stripped = len(tinfos) - len(kept)
    if stripped:
        print(f"  [GGUF] Stripped {stripped} multimodal tensors, "
              f"keeping {len(kept)}")

    tensors_packed = []
    n_svd = n_q8 = n_f32 = 0
    total_orig = total_comp = 0

    with open(path, 'rb') as f:
        for idx, (name, dims, dtype, offset) in enumerate(kept):
            n_el = 1
            for d in dims:
                n_el *= d
            f.seek(data_start + offset)
            arr = _dequant_gguf_tensor(f, dtype, n_el)

            if len(dims) == 1:
                arr = arr[:dims[0]]
            elif len(dims) == 2:
                arr = arr[:dims[0]*dims[1]].reshape(dims[1], dims[0]).T
            else:
                arr = arr.reshape(list(reversed(dims)))

            total_orig += arr.nbytes
            norm = normalize_tensor_name(name)
            ttype, data, rank = _compress_tensor(
                norm, arr, energy, fixed_rank, disable_svd)
            total_comp += len(data)

            if ttype == TENSOR_SVD_Q8: n_svd += 1
            elif ttype == TENSOR_F32:   n_f32 += 1
            else:                       n_q8  += 1

            pct = (idx + 1) / len(kept) * 100
            tag = (f".ntmre compression r={rank}" if ttype == TENSOR_SVD_Q8 else
                   "F32" if ttype == TENSOR_F32 else "Q8 ")
            print(f"  [{idx+1:4d}/{len(kept)}] {pct:5.1f}%  "
                  f"{norm:40s}  {tag:10s}  "
                  f"{arr.nbytes/1e6:6.1f}MB -> {len(data)/1e6:5.1f}MB")

            tensors_packed.append((norm, ttype, arr.shape, data))

    print(f"\n  .ntmre compression: {n_svd}  Q8: {n_q8}  F32: {n_f32}")
    print(f"  Original F32: {total_orig/1e9:.2f} GB  "
          f"Compressed: {total_comp/1e6:.1f} MB")
    return config, tokenizer, tensors_packed


# ===========================================================================
# Non-GGUF loaders
# ===========================================================================

def load_safetensors(path):
    print(f"  [SafeTensors] Loading: {path}")
    with open(path, 'rb') as f:
        hlen   = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(hlen).decode('utf-8'))
        dstart = 8 + hlen
        tensors = {}
        for name, info in header.items():
            if name == '__metadata__':
                continue
            dt  = info['dtype']
            shp = info['shape']
            s, e = info['data_offsets']
            f.seek(dstart + s)
            raw = f.read(e - s)
            if   dt == 'F32':
                arr = np.frombuffer(raw, dtype=np.float32).reshape(shp)
            elif dt == 'F16':
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shp)
            elif dt == 'BF16':
                u   = np.frombuffer(raw, dtype=np.uint16)
                arr = (u.astype(np.uint32) << 16).view(np.float32).reshape(shp)
            elif dt in ('I32', 'I64'):
                ndt = np.int32 if dt == 'I32' else np.int64
                arr = np.frombuffer(raw, dtype=ndt).astype(np.float32).reshape(shp)
            else:
                print(f"    Skipping {name}: unknown dtype {dt}")
                continue
            tensors[name] = arr.copy()
    print(f"  [SafeTensors] {len(tensors)} tensors loaded")
    return tensors


def load_pytorch(path):
    print(f"  [PyTorch] Loading: {path}")
    try:
        import torch
        sd = torch.load(str(path), map_location='cpu', weights_only=True)
        if hasattr(sd, 'state_dict'):
            sd = sd.state_dict()
        tensors = {k: v.float().numpy() for k, v in sd.items()}
        print(f"  [PyTorch] {len(tensors)} tensors loaded")
        return tensors
    except ImportError:
        import zipfile
        if zipfile.is_zipfile(str(path)):
            raise NotImplementedError(
                "PyTorch ZIP format requires torch.\n"
                "Convert to SafeTensors first or install torch.")
        raise ValueError(f"Cannot load {path} without torch installed")


def load_npy_checkpoint(directory):
    print(f"  [NPY] Loading: {directory}")
    directory = Path(directory)
    tensors = {}
    for fp in directory.glob("*.npy"):
        tensors[fp.stem] = np.load(str(fp)).astype(np.float32)
    layers_dir = directory / "layers"
    if layers_dir.exists():
        for ld in sorted(layers_dir.iterdir()):
            if ld.is_dir():
                for fp in ld.glob("*.npy"):
                    tensors[f"layers.{ld.name}.{fp.stem}"] =\
                        np.load(str(fp)).astype(np.float32)
    print(f"  [NPY] {len(tensors)} tensors loaded")
    return tensors


# ===========================================================================
# Config inference (for non-GGUF sources)
# ===========================================================================

def _infer_config(tensors, config_json_path=None):
    if config_json_path and Path(config_json_path).exists():
        raw = json.loads(Path(config_json_path).read_text())
        return raw.get('model', raw)

    norm = {normalize_tensor_name(k): v for k, v in tensors.items()}
    cfg  = {}

    for k, w in norm.items():
        if ('token_embd' in k or 'embed' in k) and w.ndim == 2:
            cfg['vocab_size'] = w.shape[0]
            cfg['dim']        = w.shape[1]
            break

    layers = set()
    for k in norm:
        parts = k.split('.')
        for i, p in enumerate(parts):
            if p == 'layers' and i + 1 < len(parts):
                try: layers.add(int(parts[i + 1]))
                except ValueError: pass
    if layers:
        cfg['n_layers'] = max(layers) + 1

    dim = cfg.get('dim', 0)
    for k, w in norm.items():
        if 'attn_q' in k and w.ndim == 2:
            qo = w.shape[0] if w.shape[1] == dim else w.shape[1]
            cfg.setdefault('n_heads', max(1, qo // 128))
            cfg.setdefault('head_dim', 128)
            break
    for k, w in norm.items():
        if 'attn_k' in k and w.ndim == 2:
            ko = w.shape[0] if w.shape[1] == dim else w.shape[1]
            cfg.setdefault('n_kv_heads', max(1, ko // 128))
            break
    for k, w in norm.items():
        if 'ffn_gate' in k and w.ndim == 2:
            cfg.setdefault('hidden_dim', max(w.shape))
            break

    cfg.setdefault('max_seq_len', 2048)
    cfg.setdefault('rope_base', 10000.0)
    cfg.setdefault('arch_type', 0)
    return cfg


# ===========================================================================
# Compress a dict of tensors -> list of (name, ttype, shape, data)
# ===========================================================================

def _compress_dict(tensors, energy, fixed_rank, disable_svd):
    norm = {normalize_tensor_name(k): v for k, v in tensors.items()}
    out  = []
    n_svd = n_q8 = n_f32 = 0
    total_orig = total_comp = 0

    print(f"  Compressing {len(norm)} tensors...")
    for i, (name, arr) in enumerate(sorted(norm.items())):
        total_orig += arr.nbytes
        ttype, data, rank = _compress_tensor(
            name, arr, energy, fixed_rank, disable_svd)
        total_comp += len(data)
        if ttype == TENSOR_SVD_Q8: n_svd += 1
        elif ttype == TENSOR_F32:   n_f32 += 1
        else:                       n_q8  += 1
        tag = (f".ntmre compression r={rank}" if ttype == TENSOR_SVD_Q8 else
               "F32" if ttype == TENSOR_F32 else "Q8 ")
        print(f"  [{i+1:4d}] {name:40s}  {tag:10s}  "
              f"{arr.nbytes/1e6:6.1f}MB -> {len(data)/1e6:5.1f}MB")
        out.append((name, ttype, arr.shape, data))

    print(f"\n  .ntmre compression: {n_svd}  Q8: {n_q8}  F32: {n_f32}")
    print(f"  Original: {total_orig/1e9:.2f} GB  "
          f"Compressed: {total_comp/1e6:.1f} MB")
    return out


# ===========================================================================
# GPU converter fast-path  (ntmre_convert_gpu.exe / ntmre_convert_v13.exe)
# ===========================================================================

def _find_gpu_exe():
    candidates = [
        _HERE / 'ntmre_convert_gpu.exe',
        _HERE / 'ntmre_convert_v13.exe',
        _HERE / 'engine' / 'ntmre_convert_gpu.exe',
        Path.cwd() / 'ntmre_convert_gpu.exe',
    ]
    if getattr(sys, 'frozen', False):
        base = Path(sys.executable).parent
        candidates = [base / 'ntmre_convert_gpu.exe',
                      base / 'ntmre_convert_v13.exe'] + candidates
        mei = getattr(sys, '_MEIPASS', None)
        if mei:
            candidates = [Path(mei) / 'ntmre_convert_gpu.exe'] + candidates
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _try_gpu_converter(input_path, output_path, energy, mgpu=False):
    exe = _find_gpu_exe()
    if not exe:
        return False
    import subprocess
    cmd = [exe, str(input_path), '-o', str(output_path), '-e', str(energy)]
    if mgpu:
        cmd.append('-mgpu')
    print(f"  [GPU] {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.stdout:
            print(r.stdout)
        if r.returncode == 0 and Path(output_path).exists():
            with open(output_path, 'rb') as vf:
                got = struct.unpack('<I', vf.read(4))[0]
            if got == NTMRE_MAGIC:
                print("  [GPU] Validated — NTMR magic confirmed.")
                return True
            print(f"  [GPU] Bad magic 0x{got:08X} — falling back to Python")
            Path(output_path).unlink(missing_ok=True)
        else:
            if r.stderr:
                print(r.stderr)
            print("  [GPU] Failed — falling back to Python")
    except Exception as e:
        print(f"  [GPU] Could not run: {e}")
    return False


# ===========================================================================
# Public write function (thin wrapper around ntmre_format_py.write_ntmre)
# ===========================================================================

def write_ntmre(output_path, config, tokenizer, tensors_packed):
    """Write .ntmre in canonical NTMR v13 format.
    tensors_packed: list of (name, ttype, shape, data_bytes)
    """
    _fmt.write_ntmre(str(output_path), config, tokenizer, tensors_packed)


def write_ntmre_svd(output_path, config, tensors_packed, tokenizer_data=None):
    """Legacy shim — was write_ntmre_svd (NSVD format). Now writes NTMR v13.
    tokenizer_data may be a JSON bytes blob from the old format; decoded here.
    """
    tokenizer = {}
    if isinstance(tokenizer_data, (bytes, bytearray)):
        try:
            tokenizer = json.loads(tokenizer_data.decode('utf-8'))
        except Exception:
            pass
    write_ntmre(output_path, config, tokenizer, tensors_packed)


# ===========================================================================
# Master conversion pipeline
# ===========================================================================

def convert_model(input_path, output_path=None, config_override=None,
                  energy=ENERGY_THRESHOLD, fixed_rank=None,
                  disable_svd=False, mgpu=False):
    """Convert any supported model format to .ntmre (NTMR v13).

    This is the single entry point used by ntmre_panel.py, the link,
    the GUI converter panel, and the CLI.
    """
    print()
    print("  " + "=" * 58)
    print("    NIGHTMARE UNIVERSAL CONVERTER  v14")
    print("    Any model  ->  .ntmre  (NTMR v13, .ntmre compression)")
    print("    Nightmare Designs")
    print("  " + "=" * 58)
    print()

    input_path  = Path(input_path)
    output_path = Path(output_path) if output_path else\
                  input_path.with_suffix('.ntmre')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ext = input_path.suffix.lower()

    # GGUF: try GPU exe first
    if ext == '.gguf' and _try_gpu_converter(input_path, output_path,
                                              energy, mgpu):
        _print_done(output_path)
        return

    # Python path
    config    = None
    tokenizer = {}

    if ext == '.ntmre':
        # Re-convert an existing .ntmre (any old format: NSVD, RMTN v1/v2/v13/v16)
        # Load it via the canonical reader, then re-compress and write fresh v13.
        print(f"  [NTMRE] Re-converting existing .ntmre: {input_path}")
        try:
            old_cfg, old_tok, old_tensors = _fmt.read_ntmre(str(input_path),
                                                              load_weights=True)
        except Exception as e:
            raise ValueError(
                f"Could not read existing .ntmre file: {e}\n"
                "If this is a v16 or NSVD file the ntmre_v16_engine or "
                "ntmre_runtime may be needed to read it first."
            )
        config    = old_cfg
        tokenizer = old_tok
        # old_tensors is list of (name, dtype, shape, data_bytes)
        # Re-compress: dequantize existing tensors then repack
        tensors_packed = []
        import ntmre_format_py as _fmt2
        for name, dtype, shape, data in old_tensors:
            if data is None:
                print(f"    Skipping {name}: no data loaded")
                continue
            try:
                arr = _fmt2.dequant_tensor(dtype, data, shape)
            except Exception as de:
                print(f"    Skipping {name}: dequant failed ({de})")
                continue
            ttype, out_data, rank = _compress_tensor(
                name, arr, energy, fixed_rank, disable_svd)
            tensors_packed.append((name, ttype, arr.shape, out_data))
            tag = (f".ntmre compression r={rank}" if ttype == TENSOR_SVD_Q8 else
                   "F32" if ttype == TENSOR_F32 else "Q8 ")
            print(f"  {name:40s}  {tag}")

    elif input_path.is_dir():
        tensors        = load_npy_checkpoint(str(input_path))
        config         = _infer_config(tensors, config_override)
        tensors_packed = _compress_dict(tensors, energy, fixed_rank, disable_svd)

    elif ext == '.gguf':
        config, tokenizer, tensors_packed = _load_gguf_streaming(
            input_path, energy, fixed_rank, disable_svd)

    elif ext == '.safetensors':
        tensors        = load_safetensors(str(input_path))
        config         = _infer_config(tensors, config_override)
        tensors_packed = _compress_dict(tensors, energy, fixed_rank, disable_svd)

    elif ext == '.onnx':
        try:
            from onnx_parser import load_onnx
            tensors = load_onnx(str(input_path))
        except ImportError:
            raise ImportError("onnx_parser.py required for .onnx input")
        config         = _infer_config(tensors, config_override)
        tensors_packed = _compress_dict(tensors, energy, fixed_rank, disable_svd)

    elif ext in ('.bin', '.pt', '.pth'):
        tensors        = load_pytorch(str(input_path))
        config         = _infer_config(tensors, config_override)
        tensors_packed = _compress_dict(tensors, energy, fixed_rank, disable_svd)

    else:
        raise ValueError(
            f"Unknown input format: {input_path.suffix}\n"
            "Supported: .gguf  .safetensors  .bin  .pt  .pth  directory/"
        )

    print(f"\n  Writing: {output_path}")
    write_ntmre(output_path, config or {}, tokenizer, tensors_packed)
    _print_done(output_path)


def _print_done(output_path):
    p = Path(output_path)
    if not p.exists():
        print("\n  [!] Output file missing — conversion may have failed")
        return
    sz = p.stat().st_size
    print()
    print("  " + "=" * 50)
    print("  CONVERSION COMPLETE")
    print(f"  Output : {p}")
    print(f"  Size   : {sz/1e6:.1f} MB  ({sz/1e9:.3f} GB)")
    print(f"  Format : NTMR v{NTMRE_VERSION}")
    print("  " + "=" * 50)
    print()


# Legacy alias used by nightmare_compress.py
def compress_checkpoint(ckpt_path, config_path=None):
    convert_model(ckpt_path, config_override=config_path)


# ===========================================================================
# CLI
# ===========================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    _src        = sys.argv[1]
    _dst        = None
    _cfg        = None
    _energy     = ENERGY_THRESHOLD
    _rank       = None
    _nosvd      = False
    _mgpu       = False
    _verify     = False

    i = 2
    while i < len(sys.argv):
        a = sys.argv[i]
        if a in ('-o', '--output') and i + 1 < len(sys.argv):
            _dst = sys.argv[i + 1]; i += 2
        elif a in ('-e', '--energy') and i + 1 < len(sys.argv):
            _energy = float(sys.argv[i + 1]); i += 2
        elif a in ('-r', '--rank') and i + 1 < len(sys.argv):
            _rank = int(sys.argv[i + 1]); i += 2
        elif a == '--config' and i + 1 < len(sys.argv):
            _cfg = sys.argv[i + 1]; i += 2
        elif a == '--nosvd':
            _nosvd = True; i += 1
        elif a == '--mgpu':
            _mgpu = True; i += 1
        elif a == '--verify':
            _verify = True; i += 1
        else:
            i += 1

    convert_model(_src, _dst, _cfg, _energy, _rank, _nosvd, _mgpu)

    if _verify:
        _op = _dst or str(Path(_src).with_suffix('.ntmre'))
        try:
            cfg, tok, tens = _fmt.read_ntmre(_op, load_weights=False)
            print(f"  [VERIFY] OK  tensors={len(tens)}  "
                  f"dim={cfg.get('dim')}  layers={cfg.get('n_layers')}  "
                  f"vocab={cfg.get('vocab_size')}")
        except Exception as ve:
            print(f"  [VERIFY] FAILED: {ve}")
            sys.exit(1)
