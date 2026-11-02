# -*- coding: utf-8 -*-
"""
nightmare_nvrtc_select.py — pick the best prebuilt NVRTC PTX + engine combo
=============================================================================

SELECTION ONLY. This module never builds PTX.

Caleb's correction (2026-10-28): "I don't expect the auto engine to
build the ptx files that's kinda a hard one to auto compute."

So the runtime job is: look at the person's actual specs (vendor, GPU
name, compute capability, VRAM, GPU count, model size), pick the best
PREBUILT PTX file already in nvrtc-kernels/, pick the best engine mode
for those specs, and load it via the NVRTC driver path. If no matching
PTX exists, fall back cleanly (layer streaming, .ntmre compression, CPU, or the AMD
HIPRTC path) with a reason — never crash, never silently degrade to
CPU, and never auto-compute a new PTX at load time.

PTX facts used here (verified from the files in nvrtc-kernels/):
  * The PTX files are model-agnostic: one md5 per arch. The sm75 files
    (gguf, hf, gemma_4_e4b, ...) are all md5 7a2a78db..., the sm86 files
    are all md5 00f54f50.... So a gguf/hf generic file is a valid
    fallback for any model; a model-specific filename is not required.
  * Base .ptx (no __smNN suffix) IS the sm75 build.
  * __sm86.ptx is the Ampere build. Per auto_engine_builder's own
    comment, sm86 PTX JITs upward to Ada (8.9), so sm86 is also the
    pick for 8.9 and 9.0 cards — it is the highest prebuilt arch here.
  * Anything below sm75 (Maxwell, Pascal, Volta, Fermi) has no
    prebuilt PTX here. Those cards do NOT get a PTX built for them
    at load; they fall back to CPU (with a reason).

Lookup order for PTX (selection, not a build step):
  1. <admin-root>/nvrtc-kernels/   (the real location in the admin copy)
  2. <admin-root>/nvrtc/ptx/       (the legacy location auto_engine_builder
                                     originally expected; kept for
                                     compatibility, usually absent)

nvrtc_build.c / nvrtc_build.exe are developer/offline build tools
only. They are NOT required at runtime, and the load path never calls
them. Building a new PTX from .cu is an offline developer job, done
with gcc from nvrtc_build.c, not something the auto engine does
when a user loads a model.

No third-party dependencies. Stdlib only.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# PTX lookup directories, in order. First hit wins.
PTX_DIRS = [
    os.path.join(HERE, "nvrtc-kernels"),
    os.path.join(HERE, "nvrtc", "ptx"),
]

# VRAM headroom: never plan to fill a card completely (matches
# system_planner.VRAM_HEADROOM).
VRAM_HEADROOM = 0.90
LOGITS_HEADROOM = 0.15

# Known NVIDIA cards, by name substring (lowercase, longest first).
# Only used when no compute-capability value was probed. This is a
# name lookup, not a new discovery system — the real source is
# cuDeviceGetAttribute / nvidia-smi compute_cap in
# nightmare_gpu_discovery, when it is available.
_KNOWN_NVIDIA = [
    ("rtx 4090", (8, 9)),
    ("rtx 4080", (8, 9)),
    ("rtx 4070", (8, 9)),
    ("rtx 4060", (8, 9)),
    ("rtx 3060 ti", (8, 6)),
    ("rtx 3060", (7, 5)),
    ("rtx 3070", (8, 6)),
    ("rtx 3080", (8, 6)),
    ("rtx 3090", (8, 6)),
    ("rtx 2080", (7, 5)),
    ("rtx 2070", (7, 5)),
    ("rtx 2060", (7, 5)),
    ("gtx 1660", (7, 5)),
    ("gtx 1070", (6, 1)),
    ("gtx 1080", (6, 1)),
    ("a100", (8, 0)),
    ("h100", (9, 0)),
]


def parse_compute_cap(value):
    """Normalise a compute-cap value to (major, minor) or None.

    Accepts (7, 5), "7.5", "75", 75, "sm75", "compute_75".
    """
    if value is None:
        return None
    try:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        s = str(value).strip().lower()
        s = s.replace("compute_", "").replace("sm", "").strip()
        if "." in s:
            major, minor = s.split(".", 1)
            return (int(major), int(minor))
        if s.isdigit() and len(s) == 2:
            return (int(s[0]), int(s[1]))
        if s.isdigit() and len(s) == 3:
            # e.g. "860" is sometimes written for 8.6 in tooling
            return (int(s[0]), int(s[1:]))
    except Exception:
        return None
    return None


def infer_compute_cap(device):
    """Compute capability for one device dict, or None if unknown.

    Order: probed compute_cap field first, then GPU name table.
    Never guesses a capability for an unknown card — callers treat
    None as 'unknown arch' and pick the lowest prebuilt PTX (sm75,
    which JITs upward) with that reason stated.
    """
    if not isinstance(device, dict):
        # Objects with attributes (system_planner.Device) also work.
        try:
            value = getattr(device, "compute_cap", None)
            if value is not None:
                return parse_compute_cap(value)
            name = str(getattr(device, "name", "") or "")
        except Exception:
            return None
    else:
        value = device.get("compute_cap")
        if value is not None:
            parsed = parse_compute_cap(value)
            if parsed is not None:
                return parsed
        name = str(device.get("name", "") or "")
    low = name.lower()
    for substr, cc in _KNOWN_NVIDIA:
        if substr in low:
            return cc
    return None


def ptx_arch_for_cc(cc):
    """Prebuilt PTX arch for a compute capability, or None if too old.

    (8, 6) and above -> "86" (86 JITs upward to 8.9/9.0, highest
    prebuilt arch here). (7, 5) to (8, 5) -> "75". Below (7, 5) ->
    None: no prebuilt PTX, fall back, do not build one.
    """
    if cc is None:
        return None
    major, minor = cc
    if (major, minor) >= (8, 6):
        return "86"
    if (major, minor) >= (7, 5):
        return "75"
    return None


def find_ptx(ptx_dirs=None, arch=None):
    """Find a prebuilt PTX file for `arch` ("75" or "86"), or None.

    Selection only — this only looks for files that already exist.
    Prefers the generic gguf file, then hf, then any file with the
    matching __smNN suffix, then the base file when arch == "75"
    (base IS sm75). Returns an absolute path or None.
    """
    dirs = list(ptx_dirs) if ptx_dirs else list(PTX_DIRS)
    if arch is None:
        return None
    candidates = []
    for stem in ("gguf__nvidia_kernels__kernels",
                 "hf__nvidia_kernels__kernels"):
        candidates.append("%s__sm%s.ptx" % (stem, arch))
    if arch == "75":
        # Base files are sm75 builds.
        candidates.append("gguf__nvidia_kernels__kernels.ptx")
        candidates.append("hf__nvidia_kernels__kernels.ptx")
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        try:
            names = os.listdir(d)
        except Exception:
            continue
        for cand in candidates:
            if cand in names:
                return os.path.join(d, cand)
        # Any model file with the right suffix — PTX is model-agnostic.
        for name in sorted(names):
            if name.endswith("__sm%s.ptx" % arch):
                return os.path.join(d, name)
        if arch == "75":
            for name in sorted(names):
                if name.endswith("__nvidia_kernels__kernels.ptx"):
                    return os.path.join(d, name)
    return None


def _device_info(device):
    """Normalise a device dict/object to (vendor, name, total, free, cc)."""
    if isinstance(device, dict):
        vendor = str(device.get("vendor", "") or "").lower()
        name = str(device.get("name", "") or "")
        total = int(device.get("total", 0) or 0)
        free = int(device.get("free", 0) or 0)
        cc = infer_compute_cap(device)
        drivable = bool(device.get("drivable", True))
    else:
        vendor = str(getattr(device, "vendor", "") or "").lower()
        name = str(getattr(device, "name", "") or "")
        total = int(getattr(device, "total", 0) or 0)
        free = int(getattr(device, "free", 0) or 0)
        cc = infer_compute_cap(device)
        drivable = bool(getattr(device, "drivable", True))
    return vendor, name, total, free, cc, drivable


def choose_combo(devices, model_bytes=0, ptx_dirs=None):
    """Pick the best engine + PTX combo for these specs.

    devices: list of device dicts/objects from nightmare_gpu_discovery
        (vendor, name, total, free, compute_cap, drivable).
    model_bytes: on-disk / VRAM footprint of the weights (0 = unknown).

    Returns a dict with:
      vendor: "nvidia" | "amd" | "none"
      engine_mode: "nvrtc_full_vram" | "nvrtc_mesh_split" |
                   "stream" | "amd_hiprtc" | "cpu"
      engine: planner-compatible name: "gpu_native" | "fused" |
              "stream" | "cpu" (amd maps to "stream" for the planner,
              with vendor="amd" and engine_mode="amd_hiprtc" carrying
              the real reason — the planner must not assign an
              NVIDIA gpu_native engine to an AMD card)
      ptx_path: absolute path to the selected prebuilt PTX, or None
      ptx_arch: "75" | "86" | None
      reason: plain-English reason, always present — no silent CPU
              fallback, no silent degradation
      devices_used: list of device dicts/objects actually used
      dual_gpu: True when layers are split across 2+ cards
      fits: True when the model fits resident in usable VRAM
    """
    infos = []
    for d in (devices or []):
        vendor, name, total, free, cc, drivable = _device_info(d)
        if not drivable:
            continue
        infos.append((d, vendor, name, total, free, cc))
    if not infos:
        return {
            "vendor": "none", "engine_mode": "cpu", "engine": "cpu",
            "ptx_path": None, "ptx_arch": None,
            "reason": "no drivable accelerator - CPU only",
            "devices_used": [], "dual_gpu": False, "fits": False,
        }

    nvidia = [x for x in infos if x[1] == "nvidia"]
    amd = [x for x in infos if x[1] == "amd"]
    if not nvidia and amd:
        return {
            "vendor": "amd", "engine_mode": "amd_hiprtc",
            "engine": "stream",
            "ptx_path": None, "ptx_arch": None,
            "reason": "AMD card (%s) - NVIDIA PTX does not apply; "
                      "use HIPRTC/AMD path or streaming, not NVIDIA PTX"
                      % (amd[0][2] or "AMD"),
            "devices_used": [x[0] for x in amd],
            "dual_gpu": len(amd) > 1, "fits": False,
        }
    if not nvidia:
        # Intel or unknown vendor only — no PTX, no HIPRTC path here.
        return {
            "vendor": infos[0][1] or "none", "engine_mode": "cpu",
            "engine": "cpu",
            "ptx_path": None, "ptx_arch": None,
            "reason": "no NVIDIA or AMD accelerator with a usable "
                      "engine path - CPU only",
            "devices_used": [x[0] for x in infos],
            "dual_gpu": False, "fits": False,
        }

    # NVIDIA path from here. Usable VRAM per card, 90% headroom.
    def usable(x):
        base = x[4] if x[4] > 0 else x[3]
        return base * VRAM_HEADROOM

    # Best single card first (most free VRAM), then total across cards.
    nvidia_sorted = sorted(nvidia, key=lambda x: usable(x), reverse=True)
    need = 0
    if model_bytes:
        need = float(model_bytes) * (1.0 + LOGITS_HEADROOM)
    total_usable = sum(usable(x) for x in nvidia_sorted)
    single_usable = usable(nvidia_sorted[0]) if nvidia_sorted else 0
    fits_single = (not need) or (single_usable >= need)
    fits_total = (not need) or (total_usable >= need)

    if fits_single:
        chosen = [nvidia_sorted[0]]
        mode = "nvrtc_full_vram"
        engine = "gpu_native"
        fits = True
    elif fits_total and len(nvidia_sorted) > 1:
        chosen = nvidia_sorted
        mode = "nvrtc_mesh_split"
        engine = "fused"
        fits = True
    else:
        # Too big for resident VRAM, or model size unknown-but-large
        # is not assumed here: when need is known and exceeds total,
        # streaming is the honest answer. When need is 0, fits_single
        # above already handled it, so reaching here means need > total.
        chosen = nvidia_sorted
        mode = "stream"
        engine = "stream"
        fits = False

    # PTX arch: lowest common capability across the chosen cards, so
    # one PTX runs on every card in the split. Unknown CC -> sm75,
    # which JITs upward, stated as a reason, not a guess hidden.
    ccs = [x[5] for x in chosen if x[5] is not None]
    if ccs:
        lowest = min(ccs)
        arch = ptx_arch_for_cc(lowest)
        if arch is None:
            return {
                "vendor": "nvidia", "engine_mode": "cpu",
                "engine": "cpu",
                "ptx_path": None, "ptx_arch": None,
                "reason": "NVIDIA card arch sm%d%d is below the "
                          "minimum prebuilt PTX sm75 - no PTX built "
                          "at load; CPU fallback"
                          % (lowest[0], lowest[1]),
                "devices_used": [x[0] for x in chosen],
                "dual_gpu": False, "fits": False,
            }
        arch_reason = "lowest card CC %d.%d -> sm%s" % (
            lowest[0], lowest[1], arch)
    else:
        arch = "75"
        arch_reason = ("unknown arch - using lowest prebuilt PTX sm75, "
                       "which JITs upward; verify on the Matrix")
    ptx = find_ptx(ptx_dirs, arch)
    if ptx is None:
        # No matching prebuilt PTX at all: clean fallback, not a build.
        return {
            "vendor": "nvidia", "engine_mode": "stream",
            "engine": "stream",
            "ptx_path": None, "ptx_arch": None,
            "reason": "no prebuilt PTX sm%s found in nvrtc-kernels/ "
                      "or nvrtc/ptx/ - PTX is not built at load; "
                      "falling back to layer streaming" % arch,
            "devices_used": [x[0] for x in chosen],
            "dual_gpu": len(chosen) > 1, "fits": fits,
        }
    if mode == "stream":
        reason = ("model needs ~%.1f GiB but only %.1f GiB usable "
                  "across %d NVIDIA card(s) - layer streaming "
                  "(.ntmre compression compressed weights where available) "
                  "with PTX sm%s (%s)"
                  % (need / 1024 ** 3 if need else 0,
                     total_usable / 1024 ** 3, len(chosen), arch,
                     arch_reason))
    elif mode == "nvrtc_mesh_split":
        reason = ("model fits across %d NVIDIA cards (%.1f GiB usable) "
                  "but not on one card - dual-GPU mesh split with "
                  "PTX sm%s (%s)"
                  % (len(chosen), total_usable / 1024 ** 3, arch,
                     arch_reason))
    else:
        reason = ("full-VRAM NVRTC on %s with PTX sm%s (%s)"
                  % (chosen[0][2] or "NVIDIA card", arch, arch_reason))
    return {
        "vendor": "nvidia", "engine_mode": mode, "engine": engine,
        "ptx_path": ptx, "ptx_arch": arch, "reason": reason,
        "devices_used": [x[0] for x in chosen],
        "dual_gpu": len(chosen) > 1, "fits": fits,
    }
