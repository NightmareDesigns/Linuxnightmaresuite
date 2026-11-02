"""
ARCH SELECTOR dY-' Pick the right kernels for the actual hardware
=====================================================================
This module answers one question: given this machine, which kernel artifact
should the loader use?

  - NVIDIA present?  read compute capability, take the closest PTX
                      (exact, else nearest lower, else highest available).
  - AMD present?      read gfx arch, take the closest HIP kernel.
  - Neither?          CPU fused engine, which is plain C and always works.

ACTUAL PTX LAYOUT (fixed 2026-10-31; previously this file claimed
nvrtc/ptx, gen, and fused, none of which exist in this tree):
  * PTX lives in <admin-root>/nvrtc-kernels/ — 60 files in this tree.
  * Per model spec, three variants share the pattern
    <spec>__nvidia_kernels__kernels[.ptx | __sm75.ptx | __sm86.ptx].
  * The base .ptx (no __smNN suffix) IS the sm75 build — verified by
    md5: hf base and hf __sm75 are identical (7a2a78db...), while
    __sm86 is a different build (00f54f50...). available_ptx() therefore
    maps the base file to compute 75, and an explicit __sm75 file, if
    present, overrides it.
  * GEN_DIR / FUSED_DIR below are legacy paths. They do not exist in
    this tree and no AMD HIP source or CPU fused .exe is shipped here,
    so the AMD/CPU routes in plan() correctly report "not present"
    instead of finding an artifact. Do not invent those artifacts.

Design notes:
  * Fails soft. A missing or unreadable GPU is not an error; it is a
    routing decision. Returns a reason the caller can show a human.
  * No network, no downloads, no outside code. Reads local artifacts only.
  * Never raises into the caller. Routing must not be able to break a load.
  * NOT wired into nightmare_nvrtc_select.choose_combo (see that file's
    note in the parent report): arch_selector is spec-based, while
    nvrtc_select is model-agnostic, and their unknown/low-arch fallbacks
    differ. Wiring it in would change GPU selection behaviour without
    a Matrix test, so it stays a standalone helper / diagnostic here.

Author: Nightmare Designs
"""

import glob
import os
import re
from typing import Any, Dict, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
# Actual PTX location in this tree. Previously "nvrtc/ptx", which does
# not exist here; the real directory is nvrtc-kernels/ beside this file.
PTX_DIR = os.path.join(HERE, "nvrtc-kernels")
# Legacy only — these directories do not exist in this tree (see header).
GEN_DIR = os.path.join(HERE, "gen")
FUSED_DIR = os.path.join(HERE, "fused")


def _no_nvidia() -> Dict[str, Any]:
    return {"vendor": "none", "arch": None, "name": None,
            "memoryMB": 0, "count": 0, "ok": False,
            "detail": "no NVIDIA device visible"}


def _no_amd() -> Dict[str, Any]:
    return {"vendor": "none", "arch": None, "name": None,
            "memoryMB": 0, "count": 0, "ok": False,
            "detail": "no AMD device visible"}


def probe_nvidia() -> Dict[str, Any]:
    """Read NVIDIA device list via nvidia-smi. Never raises."""
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=name,compute_cap,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=8)
        if r.returncode != 0 or not r.stdout.strip():
            return _no_nvidia()
        devs = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                continue
            name, cap, mem = parts[0], parts[1], parts[2]
            try:
                cap_f = float(cap)
                cap_i = int(round(cap_f * 10))   # 8.6 -> 86
            except ValueError:
                cap_i = None
            try:
                mem_i = int(mem)
            except ValueError:
                mem_i = 0
            devs.append({"name": name, "cap": cap_i,
                         "capText": cap, "memoryMB": mem_i})
        if not devs:
            return _no_nvidia()
        # Prefer the biggest card as the primary target.
        devs.sort(key=lambda d: -d.get("memoryMB", 0))
        primary = devs[0]
        return {
            "vendor": "nvidia", "ok": True,
            "arch": primary["cap"], "name": primary["name"],
            "memoryMB": primary["memoryMB"],
            "count": len(devs),
            "all": devs,
            "detail": "%d device(s), primary %s" % (len(devs), primary["name"]),
        }
    except FileNotFoundError:
        return _no_nvidia()
    except Exception as e:
        d = _no_nvidia()
        d["detail"] = "probe failed: %s" % e
        return d


def probe_amd() -> Dict[str, Any]:
    """Read AMD device list via rocminfo if present. Never raises."""
    try:
        import subprocess
        r = subprocess.run(["rocminfo"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode != 0 or not r.stdout.strip():
            return _no_amd()
        # rocminfo prints 'Name: gfx1030' per agent.
        gfx = re.findall(r"gfx(\w+)", r.stdout)
        if not gfx:
            return _no_amd()
        archs = sorted(set(gfx))
        return {
            "vendor": "amd", "ok": True,
            "arch": archs[0], "name": "AMD %s" % ",".join(archs),
            "memoryMB": 0, "count": len(gfx),
            "detail": "gfx%s" % ",gfx".join(archs),
        }
    except FileNotFoundError:
        return _no_amd()
    except Exception as e:
        d = _no_amd()
        d["detail"] = "probe failed: %s" % e
        return d


def _sm_from_text(text: Optional[str]) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"sm_(\d+)", str(text))
    return int(m.group(1)) if m else None


def available_ptx(spec: str) -> Dict[int, str]:
    """Map compute -> ptx path for one engine spec.

    The base file <spec>__nvidia_kernels__kernels.ptx (no __smNN
    suffix) IS the sm75 build, so it maps to 75. An explicit
    __sm75 file, if present, overrides the base mapping.
    """
    out: Dict[int, str] = {}
    base = os.path.join(PTX_DIR, "%s__nvidia_kernels__kernels.ptx" % spec)
    if os.path.isfile(base):
        out[75] = base
    pattern = os.path.join(PTX_DIR, "%s__nvidia_kernels__kernels__sm*.ptx" % spec)
    for path in glob.glob(pattern):
        m = re.search(r"__sm(\d+)\.ptx$", path)
        if m:
            out[int(m.group(1))] = path
    return out


def pick_ptx(spec: str, device: Dict[str, Any]) -> Dict[str, Any]:
    """
    Choose the best PTX for a device.

    Preference order:
      1. exact compute match
      2. nearest LOWER compute (forward compatible, PTX JIT upward)
      3. highest available (still runs, via JIT)

    Returns a dict describing the choice, never None, never raises.
    """
    avail = available_ptx(spec)
    if not avail:
        return {"ok": False, "vendor": "nvidia", "path": None,
                "reason": "no compiled PTX for %s" % spec,
                "available": []}

    cap = device.get("arch")
    if not isinstance(cap, int):
        # Unknown capability: take the highest we have.
        best = max(avail)
        return {"ok": True, "vendor": "nvidia", "path": avail[best],
                "compute": best, "exact": False,
                "reason": "device capability unknown, using highest",
                "available": sorted(avail)}

    if cap in avail:
        return {"ok": True, "vendor": "nvidia", "path": avail[cap],
                "compute": cap, "exact": True,
                "reason": "exact match", "available": sorted(avail)}

    lower = [c for c in avail if c < cap]
    if lower:
        best = max(lower)
        return {"ok": True, "vendor": "nvidia", "path": avail[best],
                "compute": best, "exact": False,
                "reason": "device sm_%d, using nearest lower sm_%d" % (cap, best),
                "available": sorted(avail)}

    best = min(avail)
    return {"ok": True, "vendor": "nvidia", "path": avail[best],
            "compute": best, "exact": False,
            "reason": "device sm_%d, no lower PTX, using sm_%d" % (cap, best),
            "available": sorted(avail)}


def plan(spec: str) -> Dict[str, Any]:
    """
    Full routing plan for one engine spec on this machine.
    This is the function the loader calls.
    """
    nvidia = probe_nvidia()
    amd = probe_amd()

    if nvidia.get("ok"):
        pick = pick_ptx(spec, nvidia)
        if pick.get("ok"):
            return {
                "route": "nvidia",
                "artifact": pick["path"],
                "compute": pick.get("compute"),
                "exact": pick.get("exact"),
                "reason": pick.get("reason"),
                "device": nvidia,
                "amd": amd,
            }

    # AMD path: HIP source is kept even when HIPRTC is unavailable, so the
    # engine can compile on the machine that has ROCm.
    hip_src = os.path.join(GEN_DIR, "%s__amd_kernels__kernels.hip" % spec)
    if amd.get("ok"):
        if os.path.isfile(hip_src):
            return {
                "route": "amd",
                "artifact": hip_src,
                "compute": amd.get("arch"),
                "exact": None,
                "reason": "AMD device detected, HIP source present",
                "device": amd, "nvidia": nvidia,
            }
        return {
            "route": "amd",
            "artifact": None,
            "compute": amd.get("arch"),
            "exact": None,
            "reason": "AMD device detected but no HIP source for %s" % spec,
            "device": amd, "nvidia": nvidia,
        }

    # CPU is the floor. Plain C, always builds, always runs.
    fused = os.path.join(FUSED_DIR, "%s__cpu_fused__fused.exe" % spec)
    if os.path.isfile(fused):
        reason = "no GPU detected, using CPU fused engine"
    else:
        reason = "no GPU detected and no CPU fused engine for %s" % spec
    return {
        "route": "cpu",
        "artifact": fused if os.path.isfile(fused) else None,
        "compute": None,
        "exact": None,
        "reason": reason,
        "device": {"vendor": "none", "detail": "cpu only"},
        "nvidia": nvidia, "amd": amd,
    }


if __name__ == "__main__":
    import json
    print("=== hardware ===")
    print("nvidia:", json.dumps(probe_nvidia(), indent=1)[:400])
    print("amd   :", json.dumps(probe_amd(), indent=1)[:200])

    print("\n=== routing per engine spec ===")
    specs = sorted({
        os.path.basename(p).split("__nvidia_kernels")[0]
        for p in glob.glob(os.path.join(PTX_DIR, "*__nvidia_kernels__*.ptx"))
    })
    for s in specs:
        pl = plan(s)
        art = os.path.basename(pl["artifact"]) if pl.get("artifact") else None
        print("  %-58s -> %-7s %s" % (
            s[:58], pl["route"], (pl.get("reason") or "")[:44]))
