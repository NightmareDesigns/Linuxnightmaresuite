"""
Nightmare Matmul — Python wrapper for quantized SIMD kernels.

Loads nightmare_matmul.dll (compiled C) for fast quantized dot products.
Falls back to NumPy when the DLL isn't available.

Usage:
    from nightmare_matmul import NightmareMatmul
    
    nm = NightmareMatmul()
    if nm.available:
        # Direct quantized matvec — no dequant needed!
        output = nm.q8_0_matvec(x_float32, weight_raw_bytes, n_in, n_out)
    else:
        # Fallback: dequant + numpy
        output = x @ dequantized_weight
"""

import os
import ctypes
import numpy as np
from ctypes import c_float, c_int, c_uint8, POINTER


class NightmareMatmul:
    """Fast quantized matrix-vector multiply via compiled C kernels."""
    
    def __init__(self):
        self.available = False
        self._lib = None
        
        # Try to load the compiled DLL
        dll_dir = os.path.dirname(os.path.abspath(__file__))
        dll_names = [
            os.path.join(dll_dir, "nightmare_matmul.dll"),
            os.path.join(dll_dir, "nightmare_matmul.pyd"),
            os.path.join(dll_dir, "nightmare_matmul.so"),
        ]
        
        for dll_path in dll_names:
            if os.path.exists(dll_path):
                try:
                    self._lib = ctypes.CDLL(dll_path)
                    self._setup_functions()
                    self.available = True
                    
                    # Check version
                    ver = self._lib.nightmare_matmul_version()
                    avx2 = self._lib.has_avx2()
                    print(f"[NightmareMatmul] Loaded v{ver} (AVX2: {'yes' if avx2 else 'no'})")
                    break
                except Exception as e:
                    print(f"[NightmareMatmul] Failed to load {dll_path}: {e}")
        
        if not self.available:
            print("[NightmareMatmul] DLL not found — using NumPy fallback")
            print(f"[NightmareMatmul] Build with: cl /O2 /arch:AVX2 /LD /Fe:nightmare_matmul.dll nightmare_matmul.c")
    
    def _setup_functions(self):
        """Set up ctypes function signatures."""
        lib = self._lib
        
        # q8_0_dot(x, w, n) -> float
        lib.q8_0_dot.argtypes = [POINTER(c_float), POINTER(c_uint8), c_int]
        lib.q8_0_dot.restype = c_float
        
        # q8_0_matvec(x, w, out, n_in, n_out)
        lib.q8_0_matvec.argtypes = [POINTER(c_float), POINTER(c_uint8),
                                     POINTER(c_float), c_int, c_int]
        lib.q8_0_matvec.restype = None
        
        # q8_0_matvec_batch4(x, w, out, n_in, n_out)
        lib.q8_0_matvec_batch4.argtypes = [POINTER(c_float), POINTER(c_uint8),
                                            POINTER(c_float), c_int, c_int]
        lib.q8_0_matvec_batch4.restype = None

        # Unified dispatcher: nightmare_matvec(x, w, out, n_in, n_out, quant_type)
        lib.nightmare_matvec.argtypes = [POINTER(c_float), POINTER(c_uint8),
                                          POINTER(c_float), c_int, c_int, c_int]
        lib.nightmare_matvec.restype = None
        
        # Supported types bitmask
        lib.supported_quant_types.argtypes = []
        lib.supported_quant_types.restype = c_int
        
        # q4_0_dot(x, w, n) -> float
        lib.q4_0_dot.argtypes = [POINTER(c_float), POINTER(c_uint8), c_int]
        lib.q4_0_dot.restype = c_float
        
        # q4_0_matvec(x, w, out, n_in, n_out)
        lib.q4_0_matvec.argtypes = [POINTER(c_float), POINTER(c_uint8),
                                     POINTER(c_float), c_int, c_int]
        lib.q4_0_matvec.restype = None
        
        # Version check
        lib.nightmare_matmul_version.argtypes = []
        lib.nightmare_matmul_version.restype = c_int
        lib.has_avx2.argtypes = []
        lib.has_avx2.restype = c_int
    
    def matvec(self, x: np.ndarray, w_raw: np.ndarray, n_in: int, n_out: int, quant_type: int) -> np.ndarray:
        """
        Universal quantized matrix-vector multiply.
        
        Dispatches to the correct kernel based on quant_type.
        Supports ALL GGUF formats: Q4_0(2), Q4_1(3), Q5_0(6), Q5_1(7),
        Q8_0(8), Q8_1(9), Q2_K(10), Q3_K(11), Q4_K(12), Q5_K(13), Q6_K(14), F16(1), F32(0)
        
        Args:
            x: float32 input vector, shape (n_in,)
            w_raw: raw quantized weight bytes as uint8 array
            n_in: input dimension
            n_out: output dimension
            quant_type: GGML type ID (0=F32, 1=F16, 2=Q4_0, 8=Q8_0, 12=Q4_K, etc.)
            
        Returns:
            float32 output vector, shape (n_out,)
        """
        if not self.available:
            raise RuntimeError("NightmareMatmul DLL not loaded")
        
        x = np.ascontiguousarray(x, dtype=np.float32)
        w_raw = np.ascontiguousarray(w_raw, dtype=np.uint8)
        out = np.empty(n_out, dtype=np.float32)
        
        x_ptr = x.ctypes.data_as(POINTER(c_float))
        w_ptr = w_raw.ctypes.data_as(POINTER(c_uint8))
        out_ptr = out.ctypes.data_as(POINTER(c_float))
        
        self._lib.nightmare_matvec(x_ptr, w_ptr, out_ptr, c_int(n_in), c_int(n_out), c_int(quant_type))
        return out
    
    def q8_0_matvec(self, x: np.ndarray, w_raw: np.ndarray, n_in: int, n_out: int) -> np.ndarray:
        """
        Quantized Q8_0 matrix-vector multiply.
        
        Args:
            x: float32 input vector, shape (n_in,)
            w_raw: raw Q8_0 weight bytes, shape (n_out * blocks_per_row * 34,) as uint8
            n_in: input dimension
            n_out: output dimension
            
        Returns:
            float32 output vector, shape (n_out,)
        """
        if not self.available:
            raise RuntimeError("NightmareMatmul DLL not loaded")
        
        x = np.ascontiguousarray(x, dtype=np.float32)
        w_raw = np.ascontiguousarray(w_raw, dtype=np.uint8)
        out = np.empty(n_out, dtype=np.float32)
        
        x_ptr = x.ctypes.data_as(POINTER(c_float))
        w_ptr = w_raw.ctypes.data_as(POINTER(c_uint8))
        out_ptr = out.ctypes.data_as(POINTER(c_float))
        
        self._lib.q8_0_matvec_batch4(x_ptr, w_ptr, out_ptr, c_int(n_in), c_int(n_out))
        return out
    
    def q4_0_matvec(self, x: np.ndarray, w_raw: np.ndarray, n_in: int, n_out: int) -> np.ndarray:
        """
        Quantized Q4_0 matrix-vector multiply.
        
        Args:
            x: float32 input vector, shape (n_in,)
            w_raw: raw Q4_0 weight bytes as uint8
            n_in: input dimension
            n_out: output dimension
            
        Returns:
            float32 output vector, shape (n_out,)
        """
        if not self.available:
            raise RuntimeError("NightmareMatmul DLL not loaded")
        
        x = np.ascontiguousarray(x, dtype=np.float32)
        w_raw = np.ascontiguousarray(w_raw, dtype=np.uint8)
        out = np.empty(n_out, dtype=np.float32)
        
        x_ptr = x.ctypes.data_as(POINTER(c_float))
        w_ptr = w_raw.ctypes.data_as(POINTER(c_uint8))
        out_ptr = out.ctypes.data_as(POINTER(c_float))
        
        self._lib.q4_0_matvec(x_ptr, w_ptr, out_ptr, c_int(n_in), c_int(n_out))
        return out


# Build script helper
BUILD_SCRIPT = """
@echo off
echo Building Nightmare Matmul kernels...
echo.

where cl >nul 2>&1
if %errorlevel%==0 (
    echo [MSVC detected]
    cl /O2 /arch:AVX2 /LD /Fe:nightmare_matmul.dll nightmare_matmul.c
    if exist nightmare_matmul.dll (
        echo.
        echo SUCCESS: nightmare_matmul.dll built!
    )
) else (
    where gcc >nul 2>&1
    if %errorlevel%==0 (
        echo [GCC detected]
        gcc -O3 -mavx2 -shared -o nightmare_matmul.dll nightmare_matmul.c
        if exist nightmare_matmul.dll (
            echo.
            echo SUCCESS: nightmare_matmul.dll built!
        )
    ) else (
        echo ERROR: No C compiler found!
        echo Install Visual Studio Build Tools or MinGW-w64.
        echo.
        echo Visual Studio: https://visualstudio.microsoft.com/visual-cpp-build-tools/
        echo MinGW-w64: https://www.mingw-w64.org/
    )
)
pause
"""

if __name__ == "__main__":
    # Generate build script
    build_path = os.path.join(os.path.dirname(__file__), "BUILD_MATMUL.bat")
    with open(build_path, "w") as f:
        f.write(BUILD_SCRIPT)
    print(f"Build script written to: {build_path}")
    print("Run BUILD_MATMUL.bat from a VS Developer Command Prompt")
    
    # Test if already built
    nm = NightmareMatmul()
    if nm.available:
        # Quick benchmark
        import time
        n_in, n_out = 2048, 2048
        x = np.random.randn(n_in).astype(np.float32)
        # Simulate Q8_0 weight data
        w_bytes = np.random.randint(0, 255, size=n_out * (n_in // 32) * 34, dtype=np.uint8)
        
        # Warm up
        nm.q8_0_matvec(x, w_bytes, n_in, n_out)
        
        # Benchmark
        start = time.perf_counter()
        n_iter = 100
        for _ in range(n_iter):
            nm.q8_0_matvec(x, w_bytes, n_in, n_out)
        elapsed = time.perf_counter() - start
        
        ops = n_in * n_out * n_iter
        gops = ops / elapsed / 1e9
        print(f"\nBenchmark: {n_in}x{n_out} Q8_0 matvec")
        print(f"  {n_iter} iterations in {elapsed:.3f}s")
        print(f"  {elapsed/n_iter*1000:.2f}ms per matvec")
        print(f"  {gops:.2f} GOPS (int8 multiply-accumulates)")
