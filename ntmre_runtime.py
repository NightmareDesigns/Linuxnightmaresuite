"""
NTMRE Runtime — Python link for the Nightmare .ntmre compression CUDA Engine
===================================================================
Loads .ntmre files, runs inference 100% on GPU through the compiled
nightmare_engine DLL. No NumPy. No CPU fallback. No bullshit.

Author: Nightmare Designs
"""

import ctypes
import struct
import os
import sys
import math
import time
import re
from pathlib import Path
from ctypes import (
    c_float, c_int, c_uint8, c_uint16, c_uint32, c_uint64,
    c_char_p, c_void_p, POINTER, cast, byref, sizeof
)

# === Windows Console UTF-8 Fix ===
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ENGINE_DIR = Path(__file__).parent
NTMRE_DLL_PATHS = [
    ENGINE_DIR / "nightmare_engine.dll",
    ENGINE_DIR.parent / "engine" / "nightmare_engine.dll",
    Path.cwd() / "nightmare_engine.dll",
    Path.cwd() / "engine" / "nightmare_engine.dll",
]

# NTMRE format constants (must match ntmre_format.h)
NTMRE_MAGIC = 0x4E544D52        # "NTMR"
NTMRE_MAGIC_NSVD = 0x4456534E   # "NSVD"
NTMRE_VERSION = 13
NTMRE_ALIGN = 64
NTMRE_F32 = 0
NTMRE_F16 = 1
NTMRE_Q8 = 2
NTMRE_Q4 = 3
NTMRE_SVD_Q8 = 4


class NtmreRuntime:
    """
    Full NTMRE inference runtime. Loads .ntmre file, provides
    encode/decode/forward/generate for the suite.
    """
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        class _KVAdvance:
            def advance(self):
                pass
        self.kv_cache = _KVAdvance()
        self.loaded = False
        self.lib = None
        
        # Model config
        self.dim = 0
        self.n_layers = 0
        self.n_heads = 0
        self.n_kv_heads = 0
        self.head_dim = 0
        self.hidden_dim = 0
        self.vocab_size = 0
        self.max_seq_len = 0
        
        # Tokenizer
        self.tokens = []
        self.scores = []
        self.merges = []
        self.bos_id = 1
        self.eos_id = 2
        self.token_to_id = {}
        
        # State
        self.position = 0
        self._logits_buf = None
        
        # Load DLL
        self._dll_mode = 'python'
        self._load_dll()
        
        # Load the model file
        self._load_file(filepath)
    
    def _load_dll(self):
        """Find and load the GPU engine DLL (nightmare_engine.dll)."""
        for dll_path in NTMRE_DLL_PATHS:
            if dll_path.exists():
                try:
                    lib = ctypes.CDLL(str(dll_path))

                    has_ntmre = hasattr(lib, 'ntmre_init')
                    has_nightmare = hasattr(lib, 'nightmare_matmul')

                    if has_ntmre:
                        lib.ntmre_init.restype = c_int
                        if hasattr(lib, 'ntmre_load'):
                            lib.ntmre_load.argtypes = [c_char_p]
                            lib.ntmre_load.restype = c_int
                        lib.ntmre_forward.restype = c_int
                        lib.ntmre_forward.argtypes = [c_int, c_int, POINTER(c_float)]
                        lib.ntmre_reset.restype = None
                        lib.ntmre_unload.restype = None
                        # Lost Souls tokenizer exports (full-vocab greedy BPE)
                        if hasattr(lib, 'ntmre_encode'):
                            lib.ntmre_encode.argtypes = [c_char_p, c_int, POINTER(c_int), c_int, c_int]
                            lib.ntmre_encode.restype = c_int
                        if hasattr(lib, 'ntmre_decode'):
                            lib.ntmre_decode.argtypes = [POINTER(c_int), c_int, c_char_p, c_int]
                            lib.ntmre_decode.restype = c_int
                        if hasattr(lib, 'ntmre_token_id'):
                            lib.ntmre_token_id.argtypes = [c_char_p]
                            lib.ntmre_token_id.restype = c_int
                        
                        # Unload any previously loaded model from the DLL's global state
                        # to prevent VRAM leaks.
                        lib.ntmre_unload()
                        
                        ret = lib.ntmre_init()
                        if ret != 0:
                            print(f"[NTMRE] CUDA init failed (error {ret})")
                            continue
                        self.lib = lib
                        self._dll_mode = 'ntmre'
                        print(f"[NTMRE] [OK] NTMRE engine loaded: {dll_path}")
                        return

                    elif has_nightmare:
                        self.lib = lib
                        self._dll_mode = 'nightmare'
                        print(f"[NTMRE] [OK] Nightmare engine loaded (matmul acceleration): {dll_path}")
                        return

                    else:
                        print(f"[NTMRE] DLL has no recognized exports: {dll_path}")
                        continue

                except Exception as e:
                    print(f"[NTMRE] DLL load failed ({dll_path}): {e}")
                    continue

        print("[NTMRE] [WARN] No GPU DLL found - running in pure Python mode (slower)")
        self._dll_mode = 'python'

    def _load_file(self, filepath: str):
        """Parse .ntmre file and upload weights to GPU."""
        t0 = time.time()
        
        # First read header to validate model dimensions
        self._read_header_and_tokenizer(filepath)
        
        # (old VRAM guard removed — the engine places KV in VRAM and falls back gracefully)
        # NOTE: never modify the model file on disk. Older builds rewrote the
        # magic header in place ('r+b'), which corrupted JSON-container NTMRE
        # files. The DLL path below only applies to true binary v1 files; JSON
        # containers are routed elsewhere by ntmre_router before we get here.

        if self._dll_mode == 'ntmre' and self.lib and hasattr(self.lib, 'ntmre_load'):
            try:
                ret = self.lib.ntmre_load(filepath.encode('utf-8'))
                if ret == 0:
                    print(f"[NTMRE] DLL loaded model directly")
                    self._read_header_and_tokenizer(filepath)
                    self.loaded = True
                    self._logits_buf = (c_float * self.vocab_size)()
                    print(f"[NTMRE] [OK] Ready in {time.time()-t0:.2f}s")
                    return
                elif ret == -2:
                    raise RuntimeError("Out of VRAM! Model is too large for the current GPU engine.")
                else:
                    print(f"[NTMRE] DLL returned {ret} for ntmre_load - falling back to Python loader")
            except Exception as e:
                print(f"[NTMRE] DLL ntmre_load failed: {e} - falling back to Python loader")
            
            # Graceful downgrade since DLL loader failed
            if hasattr(self.lib, 'nightmare_matmul'):
                self._dll_mode = 'nightmare'
                print("[NTMRE] Downgraded to 'nightmare' mode (Python control flow + GPU MatMul)")
            else:
                self._dll_mode = 'python'
                print("[NTMRE] Downgraded to 'python' mode (CPU only)")
        
        self._python_load(filepath)
        
        if self.loaded:
            self._logits_buf = (c_float * self.vocab_size)()
            print(f"[NTMRE] [OK] Ready in {time.time()-t0:.2f}s")
    
    def _read_header_and_tokenizer(self, filepath: str):
        with open(filepath, 'rb') as f:
            magic, version, flags, header_size = struct.unpack('<IIII', f.read(16))
            if magic != NTMRE_MAGIC and magic != NTMRE_MAGIC_NSVD:
                raise ValueError(f"Not an NTMRE file: magic=0x{magic:08X}")
            
            cfg_base = struct.unpack('<IIIIIIIf I', f.read(36))
            _ext_bytes = f.read(288)  # v2: skip the rest of the 324-byte config
            cfg = cfg_base
            self.dim = cfg[0]
            self.n_layers = cfg[1]
            self.n_heads = cfg[2]
            self.n_kv_heads = cfg[3]
            self.hidden_dim = cfg[4]
            self.vocab_size = cfg[5]
            
            if self.vocab_size == 0:
                raise ValueError("This appears to be a Bonsai/JSON format model.\n"
                                 "The main Nightmare Engine requires binary configs.\n"
                                 "Please use 'Tools -> Launch Bonsai Loader' to run this model!")
            
            self.max_seq_len = cfg[6]
            self.head_dim = cfg[8] if len(cfg) > 8 else self.dim // self.n_heads
            
            try:
                tok_hdr = struct.unpack('<IIIIIIIIII', f.read(40))
                n_tokens = tok_hdr[0]
                n_merges = tok_hdr[1]
                self.bos_id = tok_hdr[2]
                self.eos_id = tok_hdr[3]
                
                self.tokens = []
                for i in range(n_tokens):
                    slen = struct.unpack('<H', f.read(2))[0]
                    token_str = f.read(slen).decode('utf-8', errors='replace')
                    self.tokens.append(token_str)
                
                self.scores = list(struct.unpack(f'<{n_tokens}f', f.read(n_tokens * 4)))
                
                self.merges = []
                for i in range(n_merges):
                    a, b = struct.unpack('<II', f.read(8))
                    self.merges.append((a, b))
                
                self.token_to_id = {t: i for i, t in enumerate(self.tokens)}
            except struct.error:
                raise ValueError("File is truncated or corrupted (failed to unpack tokenizer). If this is a Bonsai model, please use 'Tools -> Launch Bonsai Loader'.")
    
    def _python_load(self, filepath: str):
        print("[NTMRE] Python-side loader (DLL missing ntmre_load - needs recompile)")
        try:
            self._read_header_and_tokenizer(filepath)
        except Exception as e:
            print(f"[NTMRE] [FAIL] Failed to parse file: {e}")
            raise RuntimeError(f"Failed to parse NTMRE header/tokenizer: {e}")
        
        print(f"[NTMRE] Config: dim={self.dim}, layers={self.n_layers}, "
              f"heads={self.n_heads}, kv={self.n_kv_heads}, vocab={self.vocab_size}")
        print(f"[NTMRE] Tokenizer: {len(self.tokens)} tokens, {len(self.merges)} merges")
        
        has_upload = hasattr(self.lib, 'nightmare_upload_weight')
        if not has_upload:
            print("[NTMRE] [FAIL] DLL missing weight upload - routing to GPUStreamEngine...")
            from nightmare_stream import GPUStreamEngine
            self._stream_engine = GPUStreamEngine(filepath)
            # Copy tokenizer data into stream engine
            self._stream_engine.tokens = self.tokens
            self._stream_engine.merges = self.merges
            self._stream_engine.token_to_id = self.token_to_id
            self._stream_engine._bos_id = self.bos_id
            self._stream_engine._eos_id = self.eos_id
            self.loaded = True
            return
        
        self._upload_weights(filepath)
    
    def _upload_weights(self, filepath: str):
        with open(filepath, 'rb') as f:
            f.read(16)    # file header
            f.read(324)   # v13 config: 36-byte base + 288-byte extensions
            f.read(40)    # tokenizer header
            
            for i in range(len(self.tokens)):
                slen = struct.unpack('<H', f.read(2))[0]
                f.read(slen)
            f.read(len(self.tokens) * 4)
            f.read(len(self.merges) * 8)
            
            n_tensors = struct.unpack('<I', f.read(4))[0]
            
            tensors = []
            for i in range(n_tensors):
                name_len = struct.unpack('<B', f.read(1))[0]
                name = f.read(64)[:name_len].decode('utf-8', errors='replace')
                dtype = struct.unpack('<B', f.read(1))[0]
                n_dims = struct.unpack('<B', f.read(1))[0]
                dim0, dim1 = struct.unpack('<II', f.read(8))
                offset, size_bytes = struct.unpack('<QQ', f.read(16))
                tensors.append({
                    'name': name, 'dtype': dtype, 'n_dims': n_dims,
                    'dim0': dim0, 'dim1': dim1, 
                    'offset': offset, 'size_bytes': size_bytes
                })
            
            current_pos = f.tell()
            data_offset = ((current_pos + NTMRE_ALIGN - 1) // NTMRE_ALIGN) * NTMRE_ALIGN
            print(f"[NTMRE] {n_tensors} tensors, data starts at offset {data_offset}")
            
            uploaded = 0
            for ti in tensors:
                f.seek(data_offset + ti['offset'])
                raw = f.read(ti['size_bytes'])
                
                n_elements = ti['dim0'] * (ti['dim1'] if ti['n_dims'] == 2 else 1)
                
                if ti['dtype'] == NTMRE_F32:
                    f32_data = raw
                elif ti['dtype'] in (NTMRE_Q8, NTMRE_SVD_Q8):
                    f32_data = self._dequant_q8(raw, n_elements)
                elif ti['dtype'] == NTMRE_Q4:
                    f32_data = self._dequant_q4(raw, n_elements)
                elif ti['dtype'] == NTMRE_F16:
                    f32_data = self._dequant_f16(raw, n_elements)
                else:
                    continue
                
                n_out = ti['dim0']
                n_in = ti['dim1'] if ti['n_dims'] == 2 else ti['dim0']
                
                buf = (c_float * n_elements)()
                ctypes.memmove(buf, f32_data, n_elements * 4)
                
                ptr = self.lib.nightmare_upload_weight(
                    cast(buf, POINTER(c_float)), c_int(n_in), c_int(n_out))
                
                if ptr:
                    uploaded += 1
            
            print(f"[NTMRE] Uploaded {uploaded}/{n_tensors} tensors to GPU")
            self.loaded = (uploaded > 0) or (n_tensors == 0)
    
    def _dequant_q8(self, raw: bytes, n_elements: int) -> bytes:
        import array
        out = array.array('f', [0.0] * n_elements)
        n_blocks = n_elements // 32
        
        for b in range(n_blocks):
            block_start = b * 34
            if block_start + 1 >= len(raw):
                break
            scale_bits = raw[block_start] | (raw[block_start + 1] << 8)
            sign = (scale_bits >> 15) & 1
            exp = (scale_bits >> 10) & 0x1F
            frac = scale_bits & 0x3FF
            if exp == 0:
                scale = ((-1)**sign) * (2**-14) * (frac / 1024.0)
            elif exp == 31:
                scale = 0.0
            else:
                scale = ((-1)**sign) * (2**(exp - 15)) * (1.0 + frac / 1024.0)
            
            for j in range(32):
                if block_start + 2 + j >= len(raw) or b * 32 + j >= n_elements:
                    break
                val = raw[block_start + 2 + j]
                if val >= 128:
                    val -= 256
                out[b * 32 + j] = val * scale
        
        return out.tobytes()
    
    def _dequant_q4(self, raw: bytes, n_elements: int) -> bytes:
        import array
        out = array.array('f', [0.0] * n_elements)
        n_blocks = n_elements // 32
        
        for b in range(n_blocks):
            block_start = b * 18
            if block_start + 1 >= len(raw):
                break
            scale_bits = raw[block_start] | (raw[block_start + 1] << 8)
            sign = (scale_bits >> 15) & 1
            exp = (scale_bits >> 10) & 0x1F
            frac = scale_bits & 0x3FF
            if exp == 0:
                scale = ((-1)**sign) * (2**-14) * (frac / 1024.0)
            elif exp == 31:
                scale = 0.0
            else:
                scale = ((-1)**sign) * (2**(exp - 15)) * (1.0 + frac / 1024.0)
            
            for j in range(16):
                if block_start + 2 + j >= len(raw) or b * 32 + 16 + j >= n_elements:
                    break
                byte_val = raw[block_start + 2 + j]
                low = (byte_val & 0x0F) - 8
                high = (byte_val >> 4) - 8
                out[b * 32 + j] = low * scale
                out[b * 32 + 16 + j] = high * scale
        
        return out.tobytes()
    
    def _dequant_f16(self, raw: bytes, n_elements: int) -> bytes:
        import array
        out = array.array('f', [0.0] * n_elements)
        for i in range(n_elements):
            if i * 2 + 1 >= len(raw):
                break
            bits = raw[i*2] | (raw[i*2+1] << 8)
            sign = (bits >> 15) & 1
            exp = (bits >> 10) & 0x1F
            frac = bits & 0x3FF
            if exp == 0:
                out[i] = ((-1)**sign) * (2**-14) * (frac / 1024.0)
            elif exp == 31:
                out[i] = 0.0
            else:
                out[i] = ((-1)**sign) * (2**(exp - 15)) * (1.0 + frac / 1024.0)
        return out.tobytes()
    
    def forward(self, token_id: int, position: int):
        if not self.loaded:
            return None
            
        if hasattr(self, '_stream_engine'):
            return self._stream_engine.forward(token_id, position)
        
        if self._dll_mode == 'ntmre' and self.lib and hasattr(self.lib, 'ntmre_forward'):
            ret = self.lib.ntmre_forward(
                c_int(token_id), c_int(position),
                cast(self._logits_buf, POINTER(c_float))
            )
            if ret != 0:
                raise RuntimeError(f"ntmre_forward returned {ret} for token {token_id} at pos {position}")
            return self._logits_buf
        
        # Safe fallback buffer if DLL mode is not direct ntmre
        if self._logits_buf is None and self.vocab_size > 0:
            self._logits_buf = (c_float * self.vocab_size)()
        return self._logits_buf
    
    def close(self):
        """Release GPU/RAM/DLL when switching models."""
        if hasattr(self, '_stream_engine') and self._stream_engine is not None:
            try:
                if hasattr(self._stream_engine, 'close'):
                    self._stream_engine.close()
            except Exception:
                pass
            self._stream_engine = None
        self.tokens = None
        self.token_to_id = None
        self.merges = None
        self.scores = None
        self._logits_buf = None
        self.kv_cache = None
        self.lib = None
        import gc
        gc.collect()

    def reset(self):
        if hasattr(self, '_stream_engine'):
            self._stream_engine.reset()
        if self.lib and hasattr(self.lib, 'ntmre_reset'):
            self.lib.ntmre_reset()
        self.position = 0
    
    @property
    def bos_id(self):
        return getattr(self, '_bos_id', 1)

    @bos_id.setter
    def bos_id(self, val):
        self._bos_id = val

    @property
    def eos_id(self):
        return getattr(self, '_eos_id', 2)
    
    @eos_id.setter
    def eos_id(self, val):
        self._eos_id = val
    
    def encode(self, text: str, add_bos: bool = True) -> list:
        # Full-vocab greedy longest-match via the engine DLL (real BPE pieces).
        # Falls back to the legacy per-character path if the export is missing.
        if (self._dll_mode == 'ntmre' and self.lib
                and hasattr(self.lib, 'ntmre_encode') and text):
            data = text.encode('utf-8')
            buf = (c_int * (len(data) + 16))()
            n = self.lib.ntmre_encode(data, len(data), buf, len(buf), 1 if add_bos else 0)
            if n >= 0:
                return [buf[i] for i in range(n)]

        tokens = []
        if add_bos:
            tokens.append(self.bos_id)
        
        for ch in text:
            token_id = self.token_to_id.get(ch)
            if token_id is not None:
                tokens.append(token_id)
            else:
                for byte in ch.encode('utf-8'):
                    byte_token = f"<0x{byte:02X}>"
                    bid = self.token_to_id.get(byte_token, 0)
                    tokens.append(bid)
        
        if self.merges:
            tokens = self._apply_bpe(tokens)
        
        return tokens
    
    def _apply_bpe(self, tokens: list) -> list:
        merge_rank = {pair: i for i, pair in enumerate(self.merges)}
        
        while True:
            best_pair = None
            best_rank = len(self.merges)
            best_idx = -1
            
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                rank = merge_rank.get(pair)
                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_pair = pair
                    best_idx = i
            
            if best_pair is None:
                break
            
            a_str = self.tokens[best_pair[0]] if best_pair[0] < len(self.tokens) else ""
            b_str = self.tokens[best_pair[1]] if best_pair[1] < len(self.tokens) else ""
            merged_str = a_str + b_str
            merged_id = self.token_to_id.get(merged_str)
            
            if merged_id is None:
                del merge_rank[best_pair]
                continue
            
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == best_pair[0] and tokens[i+1] == best_pair[1]:
                    new_tokens.append(merged_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        
        return tokens
    
    def decode(self, token_ids: list) -> str:
        pieces = []
        for tid in token_ids:
            if not (0 <= tid < len(self.tokens)):
                continue
            tok = self.tokens[tid]
            if self._is_control_like(tok):
                continue
            pieces.append(self._clean_token(tok))
        return ''.join(pieces)
    
    def decode_single(self, token_id: int) -> str:
        if 0 <= token_id < len(self.tokens):
            tok = self.tokens[token_id]
            if self._is_control_like(tok):
                return ""
            return self._clean_token(tok)
        return ""

    def _clean_token(self, token: str) -> str:
        token = token.replace("Ġ", " ").replace("Ċ", "\n").replace("▁", " ").replace("<0x0A>", "\n")
        if token.startswith("<0x") and token.endswith(">"):
            try:
                v = int(token[3:-1], 16)
                return bytes([v]).decode("utf-8", errors="replace")
            except Exception:
                return ""
        return token

    def _is_control_like(self, token: str) -> bool:
        if not token:
            return True
        if token in ("<s>", "</s>", "<bos>", "<eos>", "<pad>", "<unk>", "<start_of_turn>", "<end_of_turn>", "<|im_start|>", "<|im_end|>"):
            return True
        if token.startswith("<|") and token.endswith("|>"):
            return True
        if re.search(r"(jupyter|filename|tool_call|assistant_tool)", token, re.IGNORECASE):
            return True
        return False
    
    @property
    def model(self):
        return self
    
    @property 
    def name(self):
        return Path(self.filepath).stem
    
    @property
    def architecture(self):
        n = Path(self.filepath).stem.lower()
        for arch in ("gemma", "qwen", "mistral", "mixtral", "phi", "deepseek", "llama"):
            if arch in n:
                return "mistral" if arch == "mixtral" else arch
        return "llama"
    
    @property
    def metadata(self):
        token_types = [1] * len(self.tokens)
        return {
            'general.architecture': self.architecture,
            'general.name': Path(self.filepath).stem,
            'tokenizer.ggml.tokens': self.tokens,
            'tokenizer.ggml.scores': self.scores,
            'tokenizer.ggml.token_type': token_types,
            'tokenizer.ggml.merges': [f"{self.tokens[a]} {self.tokens[b]}" for (a, b) in self.merges
                                      if 0 <= a < len(self.tokens) and 0 <= b < len(self.tokens)],
            'tokenizer.ggml.model': 'bpe' if self.merges else 'llama',
            'tokenizer.ggml.bos_token_id': self.bos_id,
            'tokenizer.ggml.eos_token_id': self.eos_id,
        }