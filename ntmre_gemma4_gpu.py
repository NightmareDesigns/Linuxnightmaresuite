"""
ntmre_gemma4_gpu.py — Python link for the Gemma4 GPU-native engine
====================================================================
Loads RMTN v1 NTMRE binaries produced by ntmre_convert_gpu.exe and runs
inference 100% on GPU through nightmare_gemma4_gpu.dll (NVRTC, no nvcc).

Author: Nightmare Designs 💀
"""

import ctypes
import json
import os
import sys
from pathlib import Path
from ctypes import c_float, c_int, c_char_p, POINTER

# === Windows Console UTF-8 Fix ===
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

ENGINE_DIR = Path(__file__).parent
DLL_PATHS = [
    ENGINE_DIR / "nightmare_gemma4_gpu.dll",
    ENGINE_DIR / "engine" / "nightmare_gemma4_gpu.dll",
    Path.cwd() / "nightmare_gemma4_gpu.dll",
    Path.cwd() / "engine" / "nightmare_gemma4_gpu.dll",
]
# In a PyInstaller onefile exe the DLL extracts to sys._MEIPASS (not next to
# the module's __file__ nor cwd). Add the frozen extraction dir + the exe's
# own folder so the GPU engine is actually found inside the packaged suite.
if getattr(sys, 'frozen', False):
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        DLL_PATHS.append(Path(meipass) / "nightmare_gemma4_gpu.dll")
        DLL_PATHS.append(Path(meipass) / "engine" / "nightmare_gemma4_gpu.dll")
    try:
        exe_dir = Path(sys.executable).resolve().parent
    except Exception:
        exe_dir = None
    if exe_dir is not None:
        DLL_PATHS.append(exe_dir / "nightmare_gemma4_gpu.dll")
        DLL_PATHS.append(exe_dir / "engine" / "nightmare_gemma4_gpu.dll")


def _compile_dll():
    """Auto-compile nightmare_gemma4_gpu.dll. Tries gcc, then MSVC cl."""
    import shutil
    import subprocess

    srcs = [
        "ntmre_gemma4_gpu.c", "ntmre_model.c", "ntmre_quant.c", "ntmre_matmul.c",
        "ntmre_tokenizer.c", "ntmre_svd.c", "ntmre_threads.c", "ntmre_kvcache.c"
    ]
    base = Path(__file__).parent
    src_paths = [str(base / s) for s in srcs]
    if not all(Path(p).exists() for p in src_paths):
        print("[Gemma4-GPU] Source files missing - cannot auto-compile DLL")
        return False

    def_file = base / "ntmre_gemma4_gpu.def"
    try:
        def_file.write_text(
            "EXPORTS\n"
            "ntmre_gemma4_gpu_available\n"
            "ntmre_gemma4_gpu_load\n"
            "ntmre_gemma4_gpu_unload\n"
            "ntmre_gemma4_gpu_reset\n"
            "ntmre_gemma4_gpu_forward\n"
        )
    except Exception as e:
        print(f"[Gemma4-GPU] Failed to write .def file: {e}")
        return False

    cuda_path = os.environ.get("CUDA_PATH")
    if not cuda_path:
        for v in ["13.3", "12.9", "12.6", "12.5", "12.4", "12.3", "12.2", "12.1", "12.0", "11.8"]:
            p = Path(rf"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v{v}")
            if (p / "include" / "cuda.h").exists():
                cuda_path = str(p)
                break
    if not cuda_path:
        print("[Gemma4-GPU] CUDA Toolkit not found - cannot auto-compile DLL")
        return False

    out_dll = base / "nightmare_gemma4_gpu.dll"

    # --- Try gcc first (MinGW/w64devkit) ---
    gcc = shutil.which("gcc")
    gcc_works = False
    if gcc:
        try:
            rv = subprocess.run([gcc, "--version"], capture_output=True, text=True, timeout=10)
            gcc_works = rv.returncode == 0 and (rv.stdout.strip() or rv.stderr.strip())
        except Exception:
            pass

    if gcc_works:
        cflags = ["-O3", "-mavx2", "-mfma", "-std=c11", "-ffast-math"]
        cmd = [gcc] + cflags + ["-shared", f"-I{cuda_path}/include", "-o", str(out_dll)] + src_paths + [str(def_file), f"-L{cuda_path}/lib/x64", "-lcuda", "-lnvrtc", "-lm"]
        print("[Gemma4-GPU] Auto-compiling DLL with gcc...")
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if r.returncode == 0 and out_dll.exists():
                print(f"[Gemma4-GPU] Auto-compiled {out_dll}")
                return True
            print("[Gemma4-GPU] gcc compile failed, falling back to cl...")
            print(r.stdout[-2000:] if r.stdout else "")
            print(r.stderr[-2000:] if r.stderr else "")
        except Exception as e:
            print(f"[Gemma4-GPU] gcc compile error: {e}")

    # --- Fall back to MSVC cl ---
    vs_bat = _find_vcvarsall()
    if not vs_bat:
        print("[Gemma4-GPU] No working compiler found (gcc/cl)")
        return False

    cuda_fwd = cuda_path.replace("\\", "/")
    srcs_str = " ".join(src_paths)
    cmd = (f'"{vs_bat}" x64 && cl /O2 /arch:AVX2 /fp:fast /I"{cuda_fwd}/include" /LD {srcs_str} '
           f'/link /DEF:"{def_file}" /LIBPATH:"{cuda_fwd}/lib/x64" cuda.lib nvrtc.lib /OUT:"{out_dll}"')
    print("[Gemma4-GPU] Auto-compiling DLL with cl...")
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and out_dll.exists():
            print(f"[Gemma4-GPU] Auto-compiled {out_dll}")
            return True
        print("[Gemma4-GPU] cl compile failed:")
        print(r.stdout[-3000:] if r.stdout else "")
        print(r.stderr[-1000:] if r.stderr else "")
        return False
    except Exception as e:
        print(f"[Gemma4-GPU] cl compile error: {e}")
        return False


def _find_vcvarsall():
    """Locate Visual Studio vcvarsall.bat."""
    from pathlib import Path
    candidates = [
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"),
        Path(r"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"),
        Path(r"C:\Program Files\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Search by glob
    for root in [Path(r"C:\Program Files (x86)\Microsoft Visual Studio"), Path(r"C:\Program Files\Microsoft Visual Studio")]:
        for p in root.rglob("vcvarsall.bat"):
            return str(p)
    return None


def _load_dll():
    """Find and load the Gemma4 GPU engine DLL; auto-compile if missing."""
    for dll_path in DLL_PATHS:
        if dll_path.exists():
            try:
                lib = ctypes.CDLL(str(dll_path))
                lib.ntmre_gemma4_gpu_available.restype = c_int
                lib.ntmre_gemma4_gpu_load.argtypes = [c_char_p]
                lib.ntmre_gemma4_gpu_load.restype = c_int
                lib.ntmre_gemma4_gpu_unload.restype = None
                lib.ntmre_gemma4_gpu_reset.restype = None
                lib.ntmre_gemma4_gpu_forward.argtypes = [c_int, c_int, POINTER(c_float)]
                lib.ntmre_gemma4_gpu_forward.restype = c_int
                return lib
            except Exception as e:
                print(f"[Gemma4-GPU] DLL load failed ({dll_path}): {e}")
                continue

    # DLL not found - try to auto-compile
    if _compile_dll():
        return _load_dll()
    return None


class Gemma4GpuEngine:
    """
    Gemma 4 E2B GPU-native inference engine.
    Loads .ntmre (RMTN v1) files and runs the full forward pass on GPU.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.name = Path(filepath).stem
        self.architecture = "gemma4"
        self.loaded = False
        self._logits_buf = None
        self._lib = None
        self.tokenizer = None
        self._v1 = None

        class _KVAdvance:
            def advance(self):
                pass
        self.kv_cache = _KVAdvance()

        # Load tokenizer/config from the NTMRE v1 binary
        self._load_metadata(filepath)

        # Load GPU engine DLL
        self._lib = _load_dll()
        if self._lib is None:
            raise RuntimeError("nightmare_gemma4_gpu.dll not found or failed to load")

        if not self._lib.ntmre_gemma4_gpu_available():
            raise RuntimeError("No CUDA device available for Gemma4 GPU engine")

        ret = self._lib.ntmre_gemma4_gpu_load(filepath.encode('utf-8'))
        if ret != 0:
            raise RuntimeError(f"ntmre_gemma4_gpu_load failed ({ret})")

        self._logits_buf = (c_float * self.vocab_size)()
        self.loaded = True
        print(f"[Gemma4-GPU] Engine ready: {self.name}")

    def _load_metadata(self, filepath: str):
        """Parse the NTMRE v1 binary header/tokenizer using V1Source."""
        try:
            from ntmre_v1_engine import V1Source
            self._v1 = V1Source(filepath)
            cfg = self._v1.config
            self.dim = int(cfg.get('dim', 0))
            self.n_layers = int(cfg.get('n_layers', 0))
            self.n_heads = int(cfg.get('n_heads', 0))
            self.n_kv_heads = int(cfg.get('n_kv_heads', 0))
            self.head_dim = int(cfg.get('head_dim', 0))
            self.hidden_dim = int(cfg.get('hidden_dim', 0))
            self.vocab_size = int(cfg.get('vocab_size', 0))
            self.max_seq_len = int(cfg.get('max_seq_len', 0))
            self.rope_base = float(cfg.get('rope_theta', 10000.0))

            # Build tokenizer from the embedded JSON
            tok = json.loads(self._v1.tokenizer_bytes.decode('utf-8'))
            metadata = {
                'general.architecture': 'gemma4',
                'general.name': self.name,
                'tokenizer.ggml.tokens': tok.get('tokens', []),
                'tokenizer.ggml.scores': [0.0] * len(tok.get('tokens', [])),
                'tokenizer.ggml.merges': tok.get('merges', []),
                'tokenizer.ggml.bos_token_id': int(tok.get('bos', 1)),
                'tokenizer.ggml.eos_token_id': int(tok.get('eos', 2)),
                'tokenizer.ggml.model': tok.get('model', ''),
            }
            from tokenizer import Tokenizer
            self.tokenizer = Tokenizer(metadata, model_path=filepath)
        except Exception as e:
            raise RuntimeError(f"Failed to parse NTMRE v1 metadata/tokenizer: {e}")

    @property
    def bos_id(self):
        return self.tokenizer.bos_id if self.tokenizer else 1

    @property
    def eos_id(self):
        return self.tokenizer.eos_id if self.tokenizer else 2

    @property
    def metadata(self):
        return {
            'general.architecture': self.architecture,
            'general.name': self.name,
            'tokenizer.ggml.tokens': self.tokenizer.tokens if self.tokenizer else [],
            'tokenizer.ggml.scores': self.tokenizer.scores if self.tokenizer else [],
            'tokenizer.ggml.bos_token_id': self.bos_id,
            'tokenizer.ggml.eos_token_id': self.eos_id,
        }

    @property
    def model(self):
        return self

    def encode(self, text: str, add_bos: bool = True) -> list:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.encode(text, add_bos=add_bos)

    def decode(self, token_ids: list) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.decode(token_ids)

    def decode_single(self, token_id: int) -> str:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded")
        return self.tokenizer.decode([token_id])

    def forward(self, token_id: int, position: int):
        if not self.loaded or self._lib is None:
            return None
        ret = self._lib.ntmre_gemma4_gpu_forward(token_id, position, self._logits_buf)
        if ret != 0:
            raise RuntimeError(f"ntmre_gemma4_gpu_forward failed ({ret}) at pos {position}")
        return self._logits_buf

    def prefill_batch(self, token_ids: list, start_pos: int = 0):
        """GPU VRAM prefill across all prompt tokens."""
        if not self.loaded or self._lib is None or not token_ids:
            return None
        for i, tok in enumerate(token_ids):
            ret = self._lib.ntmre_gemma4_gpu_forward(int(tok), start_pos + i, self._logits_buf)
            if ret != 0:
                raise RuntimeError(f"ntmre_gemma4_gpu_forward prefill failed ({ret}) at pos {start_pos + i}")
        return self._logits_buf

    def prefill(self, token_ids: list, start_pos: int = 0):
        return self.prefill_batch(token_ids, start_pos)

    def reset(self):
        if self._lib is not None:
            self._lib.ntmre_gemma4_gpu_reset()

    def close(self):
        if self._lib is not None:
            self._lib.ntmre_gemma4_gpu_unload()
            self._lib = None
        self._v1 = None
        self.tokenizer = None
        self._logits_buf = None
        self.loaded = False
        import gc
        gc.collect()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
