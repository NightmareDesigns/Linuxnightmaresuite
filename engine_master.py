import os
import sys
import ctypes
from ctypes import c_int, c_void_p, c_size_t, POINTER, byref

class EngineMaster:
    """
    Bare-metal unified loader.
    Communicates directly with nvcuda.dll and amdhip64.dll via ctypes.
    Zero external Python packages required.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EngineMaster, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.nvidia_lib = None
        self.amd_lib = None
        self.devices = []
        self._boot_hardware()
        self._initialized = True

    def _boot_hardware(self):
        print("[EngineMaster] Probing hardware via native drivers...")

        # 1. Probe NVIDIA Driver (Bare-metal CUDA Driver API)
        cuda_path = r"C:\Windows\System32\nvcuda.dll"
        if os.path.exists(cuda_path):
            try:
                self.nvidia_lib = ctypes.CDLL(cuda_path)
                if self.nvidia_lib.cuInit(0) == 0:
                    count = c_int()
                    self.nvidia_lib.cuDeviceGetCount(byref(count))
                    for i in range(count.value):
                        self.devices.append({"vendor": "NVIDIA", "index": i, "handle": None})
                    print(f"[EngineMaster] Found {count.value} NVIDIA GPU(s).")
            except Exception as e:
                print(f"[EngineMaster] NVIDIA driver probe warning: {e}")

        # 2. Probe AMD Driver (Bare-metal HIP Runtime/Driver API)
        hip_path = r"C:\Windows\System32\amdhip64.dll"
        if os.path.exists(hip_path):
            try:
                self.amd_lib = ctypes.CDLL(hip_path)
                if self.amd_lib.hipInit(0) == 0:
                    count = c_int()
                    self.amd_lib.hipGetDeviceCount(byref(count))
                    for i in range(count.value):
                        self.devices.append({"vendor": "AMD", "index": i, "handle": None})
                    print(f"[EngineMaster] Found {count.value} AMD GPU(s).")
            except Exception as e:
                print(f"[EngineMaster] AMD driver probe warning: {e}")

        if not self.devices:
            print("[EngineMaster] No dedicated GPU drivers initialized. CPU fallback ready.")

    def get_topology(self):
        """Returns the detected hardware layout."""
        return self.devices

    def forward_heterogeneous_pipeline(self, input_tokens, split_layer_index=24):
        """
        Routes pipeline stages across devices:
        - NVIDIA handles layers 0 -> (split_layer_index - 1)
        - AMD handles layers split_layer_index -> end
        """
        # Direct dispatch to your custom C kernel functions using pure memory pointers
        pass