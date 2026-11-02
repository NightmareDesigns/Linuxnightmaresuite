# -*- coding: utf-8 -*-
"""
nightmare_gpu_discovery.py — every GPU on the market, one probe
=================================================================
Vendor-agnostic accelerator discovery.  Talks to ONLY the OS driver DLLs
that already exist on the machine — no CUDA toolkit, no ROCm toolkit, no
pip packages, no downloaded sources:

    nvcuda.dll        NVIDIA driver API          -> runtime 'cuda'
    amdhip64.dll      AMD HIP runtime driver     -> runtime 'hip'
    ze_loader.dll     Intel Graphics / Level-ZerO loader (Arc + iGPU)
                                                     -> runtime 'levelzero'

Each probe is fully guarded: a broken/absent DLL just yields zero devices.
Returns a normalized list so callers don't care which vendor answered:

    {vendor, index, total, free, name, runtime, drivable}

  runtime    = 'cuda' | 'hip' | 'levelzero'
  drivable   = True if the Nightmare fused engine can already compile +
               launch kernels on it (cuda/hip), False for levelzero
               until the Level-Zero kernel backend lands.

No GPU off any vendor = empty list; the engine then uses the CPU path.
"""
import ctypes
import os
import re
import struct
import subprocess

# ─────────────────────────────────────────────────────────────────────
# Tiny helpers
# ─────────────────────────────────────────────────────────────────────

def _load_dll(name):
    """Load a DLL by bare name (Windows searches System32). Never raises."""
    try:
        return ctypes.CDLL(name)
    except Exception:
        return None


def _hip_dll_dir():
    """Directory that holds the AMD HIP runtime DLLs.  The Adrenalin driver
    usually drops amdhip64.dll/hiprtc.dll into System32, but ROCm installs
    and older drivers place them under 'C:\\Program Files\\AMD\\ROCm\\<ver>\\bin'
    (or %ROCM_PATH% / %HIP_PATH%).  Returns the dir, or None if bae-named
    loads work from System32."""
    import glob as _glob
    cands = ["C:\\Program Files\\AMD\\ROCm\\bin",
             "%ROCM_PATH%\\bin", "%HIP_PATH%\\bin"]
    try:
        for v in (_glob.glob(r"C:\Program Files\AMD\ROCm\*\bin")):
            cands.append(v)
    except Exception:
        pass
    for raw in cands:
        p = os.path.expandvars(raw)
        if p and os.path.isdir(p):
            has_hip = (os.path.exists(os.path.join(p, "amdhip64.dll"))
                       or any(f.lower().startswith("amdhip") and f.lower().endswith(".dll")
                              for f in os.listdir(p)))
            if has_hip:
                return p
    return None


def _find_hip_dlls():
    """amdhip64.dll + hiprtc.dll as (dll, hiprtc) CDLL handles.

    ROCm installs versioned names (amdhip64_7.dll, hiprtc0702.dll) and
    Adrenalin drivers install unversioned names (amdhip64.dll, hiprtc.dll).
    Try both.  Searches System32 first (bare-name), then the ROCm/AMD bin
    folders.  Adds the ROCm folder to the DLL search path so hiprtc's
    dependencies resolve."""
    hip = _load_dll("amdhip64.dll")
    if hip is None:
        for name in ("amdhip64_7.dll", "amdhip64_6.dll", "amdhip64_5.dll"):
            hip = _load_dll(name)
            if hip is not None:
                break
    hiprtc = _load_dll("hiprtc.dll")
    if hiprtc is None:
        import glob as _g
        for cand in sorted(_g.glob(r"C:\Program Files\AMD\ROCm\*\bin\hiprtc*.dll"), reverse=True):
            try:
                hiprtc = ctypes.CDLL(cand)
                break
            except Exception:
                pass
    if hip is not None and hiprtc is not None:
        return hip, hiprtc
    d = _hip_dll_dir()
    if d is not None:
        try:
            os.add_dll_directory(d)
        except AttributeError:
            pass
        if hip is None:
            import glob as _g
            for cand in sorted(_g.glob(os.path.join(d, "amdhip64*.dll")), reverse=True):
                try:
                    hip = ctypes.CDLL(cand)
                    break
                except Exception:
                    pass
        if hiprtc is None:
            import glob as _g
            for cand in sorted(_g.glob(os.path.join(d, "hiprtc*.dll")), reverse=True):
                try:
                    hiprtc = ctypes.CDLL(cand)
                    break
                except Exception:
                    pass
    return hip, hiprtc


def _result_ok(r):
    """CUDA: 0 == CUDA_SUCCESS.  HIP: 0 == hipSuccess.  Level-Zero:
    0 == ZE_RESULT_SUCCESS."""
    return int(r) == 0


# Some driver stacks report a clamped sentinel instead of real VRAM:
# 0xFFFFFFFF (4294967295) and 0xFFFFFF00 are the usual ones.  Treat any
# 32-bit-looking value as untrustworthy -- no real GPU has 4 GB exactly.
_BAD_TOTALS = {0, 0xFFFFFFFF, 0xFFFFFF00, 0xFFFF0000}


def _plausible_total(v):
    """True if `v` looks like a real VRAM capacity in bytes."""
    try:
        v = int(v)
    except (TypeError, ValueError):
        return False
    if v in _BAD_TOTALS:
        return False
    # 512 MB .. 1 PB
    return (512 * 1024 ** 2) <= v <= (1024 ** 4)


def dev_key(d):
    """Globally unique, vendor-qualified device key.

    `index` is only meaningful WITHIN a vendor's own enumeration: NVIDIA
    device 0 and AMD device 0 are different physical cards.  Any code that
    mixes vendors must key on this instead, or two different GPUs collide on
    the same ordinal.
    """
    return f"{d.get('vendor', '?')}:{d.get('index', 0)}"


def vram_gb(d):
    """Usable VRAM in GiB, or 0.0 when unknown."""
    t = d.get("total", 0) or 0
    return round(t / (1024 ** 3), 2) if _plausible_total(t) else 0.0


def free_gb(d):
    """Free VRAM in GiB, or 0.0 when unknown."""
    f = d.get("free", 0) or 0
    return round(f / (1024 ** 3), 2) if f > 0 else 0.0


def _nvidia_smi():
    """nvidia-smi memory query — best source of free VRAM on NVIDIA."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free,memory.total,name",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, timeout=5).decode(errors="replace")
    except Exception:
        return {}
    rows = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
            free = int(float(parts[1]) * 1024 * 1024)
            total = int(float(parts[2]) * 1024 * 1024)
            rows[idx] = (free, total)
        except ValueError:
            continue
    return rows


# ─────────────────────────────────────────────────────────────────────
# NVIDIA
# ─────────────────────────────────────────────────────────────────────


def _nvidia_smi_cc():
    """nvidia-smi compute_cap query -- fallback for compute capability.

    Returns {index: (major, minor)} or {}. Kept separate from
    _nvidia_smi() so that function's (free, total) shape is unchanged
    for existing callers. The driver-API path in probe_nvidia() is
    tried first; this is only the fallback.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,compute_cap",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, timeout=5).decode(errors="replace")
    except Exception:
        return {}
    rows = {}
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            major, minor = parts[1].split(".", 1)
            rows[idx] = (int(major), int(minor))
        except Exception:
            continue
    return rows


def probe_nvidia():
    """Probe the NVIDIA driver API (nvcuda.dll). Returns device dicts."""
    cu = _load_dll("nvcuda.dll")
    if cu is None:
        return []
    try:
        cu.cuInit.argtypes = [ctypes.c_uint]
        if not _result_ok(cu.cuInit(0)):
            return []
        count = ctypes.c_int()
        cu.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        cu.cuDeviceGetCount(ctypes.byref(count))
        n = int(count.value)
        if n <= 0:
            return []
        smi = _nvidia_smi()
        smi_cc = _nvidia_smi_cc()
        cu.cuDeviceGetAttribute.argtypes = [
            ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        cu.cuDeviceGetName.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        cu.cuDeviceTotalMem.argtypes = [
            ctypes.POINTER(ctypes.c_size_t), ctypes.c_int]
        devs = []
        for i in range(n):
            name = ctypes.create_string_buffer(256)
            cu.cuDeviceGetName(name, 256, i)
            total = ctypes.c_size_t()
            try:
                cu.cuDeviceTotalMem(ctypes.byref(total), i)
            except Exception:
                total = ctypes.c_size_t(0)
            # cuDeviceTotalMem returns a bogus 0xFFFFFFFF sentinel on some
            # driver stacks (it clamps instead of reporting real capacity).
            # nvidia-smi is authoritative, so prefer it and only fall back to
            # the driver value when it looks like a real number.
            drv_total = int(total.value)
            if not _plausible_total(drv_total):
                drv_total = 0
            smi_free, smi_total = smi.get(i, (0, 0)) if smi else (0, 0)
            if _plausible_total(smi_total):
                real_total = int(smi_total)
            else:
                real_total = drv_total
            free = int(smi_free) if smi_free else real_total
            # Compute capability: driver API first (attrs 75=major,
            # 76=minor), nvidia-smi compute_cap as fallback, else
            # None. The NVRTC selector does the name-table fallback
            # and the unknown-arch handling, so discovery never
            # invents a capability here.
            cc = None
            try:
                major = ctypes.c_int()
                minor = ctypes.c_int()
                if _result_ok(cu.cuDeviceGetAttribute(ctypes.byref(major), 75, i)) and _result_ok(cu.cuDeviceGetAttribute(ctypes.byref(minor), 76, i)):
                    cc = (int(major.value), int(minor.value))
            except Exception:
                cc = None
            if cc is None:
                cc = smi_cc.get(i)
            devs.append({
                "vendor": "nvidia", "index": i, "total": real_total,
                "free": free, "name": name.value.decode(errors="replace"),
                "runtime": "cuda", "drivable": True,
                "compute_cap": cc,
            })
        return devs
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────
# AMD
# ─────────────────────────────────────────────────────────────────────

def probe_amd():
    """Probe the AMD HIP runtime (amdhip64.dll) + hiprtc.dll (JIT)."""
    hip, hiprtc = _find_hip_dlls()
    if hip is None:
        return []
    try:
        hip.hipInit.argtypes = [ctypes.c_uint]
        if not _result_ok(hip.hipInit(0)):
            return []
        count = ctypes.c_int()
        hip.hipGetDeviceCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
        hip.hipGetDeviceCount(ctypes.byref(count))
        n = int(count.value)
        if n <= 0:
            return []
        hip.hipSetDevice.argtypes = [ctypes.c_int]
        hip.hipDeviceGetName.argtypes = [
            ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
        hip.hipMemGetInfo.argtypes = [
            ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
        hip.hipDeviceProp_t = None
        devs = []
        for i in range(n):
            hip.hipSetDevice(i)
            name = ctypes.create_string_buffer(256)
            hip.hipDeviceGetName(name, 256, i)
            free = ctypes.c_size_t(0)
            total = ctypes.c_size_t(0)
            try:
                hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
            except Exception:
                free = ctypes.c_size_t(0)
                total = ctypes.c_size_t(0)
            devs.append({
                "vendor": "amd", "index": i, "total": int(total.value),
                "free": int(free.value), "name": name.value.decode(errors="replace"),
                "runtime": "hip", "drivable": hiprtc is not None,
            })
        return devs
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────
# Intel — Level-Zero (Arc discrete + integrated iGPU)
# ─────────────────────────────────────────────────────────────────────

# ze_device_properties_t payload (version 1), the fields we need.
_ZE_DEVICE_PROP_OFFSETS = {
    "type":           8,   # after stype(4) + pNext(4, pointer)
    "vendorId":       12,
    "deviceId":       16,
    "flags":          20,
    "revisionId":     44,
    "name":           48,  # char deviceName[256]
}

_ZE_MEMORY_PROP = struct.Struct("<I I I I 256s Q Q")  # stype,pNext,flags,heapCount,name,physicalSize,reserved


def _ze_loader():
    for name in ("ze_loader.dll", "ze_loader64.dll"):
        dll = _load_dll(name)
        if dll is not None:
            return dll
    return None


def probe_intel():
    """Probe Intel GPUs through the Level-Zero ICD loader that ships with
    the Intel Graphics driver (Arc / Iris Xe iGPU).  Kernels that execute
    on them land with the Level-Zero backend; this probe makes discovery
    complete today."""
    ze = _ze_loader()
    if ze is None:
        return []
    try:
        ze.zeInit.argtypes = [ctypes.c_uint]
        if not _result_ok(ze.zeInit(0)):
            return []
        n_drivers = ctypes.c_uint(0)
        drv = getattr(ze, "zeDriverGet", None) or None
        if drv is None:
            return []
        drv.argtypes = [ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_void_p)]
        r = drv(ctypes.byref(n_drivers), None)
        if not _result_ok(r) or n_drivers.value == 0:
            return []
        drivers = (ctypes.c_void_p * n_drivers.value)()
        r = drv(ctypes.byref(n_drivers), drivers)
        if not _result_ok(r):
            return []
        ze.zeDeviceGet.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_void_p)]
        ze.zeDeviceGetProperties.argtypes = [
            ctypes.c_void_p, ctypes.c_void_p]
        ze.zeDeviceGetMemoryProperties.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint), ctypes.c_void_p]
        devs = []
        idx = 0
        for hdr in drivers:
            n_dev = ctypes.c_uint(0)
            if not _result_ok(ze.zeDeviceGet(hdr, ctypes.byref(n_dev), None)):
                continue
            devices = (ctypes.c_void_p * n_dev.value)()
            if not _result_ok(ze.zeDeviceGet(hdr, ctypes.byref(n_dev), devices)):
                continue
            for hdev in devices:
                try:
                    buf = ctypes.create_string_buffer(1024)
                    ze.zeDeviceGetProperties(hdev, buf)
                    data = buf.raw
                    vendor_id = struct.unpack_from("<I", data, _ZE_DEVICE_PROP_OFFSETS["vendorId"])[0]
                    device_id = struct.unpack_from("<I", data, _ZE_DEVICE_PROP_OFFSETS["deviceId"])[0]
                    name = data[_ZE_DEVICE_PROP_OFFSETS["name"]:_ZE_DEVICE_PROP_OFFSETS["name"] + 256]
                    name = name.split(b"\x00")[0].decode(errors="replace")
                    total = 0
                    try:
                        n_mem = ctypes.c_uint(0)
                        ze.zeDeviceGetMemoryProperties(hdev, ctypes.byref(n_mem), None)
                        mem_props = ctypes.create_string_buffer(256 * max(1, n_mem.value))
                        r2 = ze.zeDeviceGetMemoryProperties(hdev, ctypes.byref(n_mem), mem_props)
                        if _result_ok(r2):
                            for k in range(n_mem.value):
                                tot = struct.unpack_from("<Q", mem_props.raw, 48 + k * 256)[0]
                                total += tot
                    except Exception:
                        total = 0
                    devs.append({
                        "vendor": "intel",
                        "index": idx,
                        "total": total,
                        "free": total,
                        "name": name or f"Intel GPU 0x{device_id:04X}",
                        "runtime": "levelzero",
                        "drivable": False,
                    })
                    idx += 1
                except Exception:
                    continue
        return devs
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────────────
# Combined
# ─────────────────────────────────────────────────────────────────────

_CACHE = None


def probe_all(force=False):
    """Probe every vendor once (memoized). Returns the combined list."""
    global _CACHE
    if _CACHE is not None and not force:
        return _CACHE
    devs = probe_nvidia() + probe_amd() + probe_intel()
    devs.sort(key=lambda d: d["free"], reverse=True)
    _CACHE = devs
    return devs


def usable_accelerators(limit=None):
    """The devices the fused engine can actually drive today
    (cuda/hip — levelzero kernels are still pending). If limit is provided, caps to N."""
    accs = [d for d in probe_all() if d["drivable"]]
    return accs[:limit] if limit is not None else accs


def has_gpu_accelerator() -> bool:
    """Return True if at least one usable GPU accelerator (NVIDIA CUDA or AMD HIP) exists."""
    return len(usable_accelerators()) > 0


def topology():
    """Human-readable summary of every detected GPU."""
    lines = []
    found = probe_all()
    for d in found:
        tag = {
            "cuda": "NVIDIA/NVRTC", "hip": "AMD/HIPRTC",
            "levelzero": "Intel (kernels pending)",
        }.get(d["runtime"], d["runtime"])
        if d["total"]:
            gb = f"{d['total'] / 1e9:.1f} GB"
        else:
            gb = "? GB"
        lines.append(f"  [{d['index']}] {d['vendor'].upper():8s} "
                     f"{d['name']:<40s} {gb:>7s} free={d['free'] / 1e9:.1f} "
                     f"({tag})")
    if not lines:
        lines.append("  none — CPU-only mode")
    return "\n".join(lines)


if __name__ == "__main__":
    print(topology())