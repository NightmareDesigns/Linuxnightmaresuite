# -*- coding: utf-8 -*-
"""
auto_engine_builder.py — build an engine for any model, automatically
=========================================================================
When a model is discovered that has no engine, this measures it, generates
the vendor-neutral + per-vendor engine sources, COMPILES the ones this
machine can actually build, and VERIFIES them before trusting them.

The generator (engine/model_engines/generate_engines.py) only ever emitted
source. Nothing compiled it, so 42 generated files sat unused. This closes
that loop and is safe to run on discovery:

  spec  ->  6 renderings  ->  compile what the host supports  ->  self-test

Build targets, in order of preference:
  cpu_fused    gcc -O3 -mavx2 -mfma -fopenmp  -> selftest .exe, run 3x and
               require "RESULT: PASS" (it compares fused kernels against the
               unfused reference and demands bit-identical output)
  nvidia       NVRTC (nvrtc_build.exe) -> .ptx
  amd          HIPRTC -> code object.  Skipped with a clear reason when ROCm
               is absent rather than failing.
  hosts        generated but NOT compiled: they include st_reader.h, which
               does not exist in this distribution, and they export no ABI.

Nothing here raises into the caller.  Every step is reported so the suite can
say exactly what happened.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict

HERE = os.path.dirname(os.path.abspath(__file__))          # engine/model_engines
SUITE = os.path.dirname(os.path.dirname(HERE))             # suite root
GEN_DIR = os.path.join(HERE, "gen")
FUSED_DIR = os.path.join(HERE, "fused")
PTX_DIR = os.path.join(HERE, "nvrtc", "ptx")
# PTX lookup, in order (SELECTION only, 2026-10-28 correction):
#   1. HERE/nvrtc-kernels/  — the real location in the admin copy
#      (60 prebuilt .ptx files, sm75/sm86, model-agnostic)
#   2. HERE/nvrtc/ptx/      — legacy location this builder originally
#      expected; usually absent in the admin copy.
# The load path SELECTS from these files. It does NOT build PTX at
# runtime. nvrtc_build.c / nvrtc_build.exe are developer/offline
# build tools only (see find_nvrtc_build below).
PTX_DIRS = [os.path.join(HERE, "nvrtc-kernels"), PTX_DIR]
OBJ_DIR = os.path.join(HERE, "obj")
SPECS_FILE = os.path.join(HERE, "specs.json")
MANIFEST = os.path.join(HERE, "build_manifest.json")

sys.path.insert(0, HERE)
# spec_engine_runner imports nightmare_engines / gguf_parser from the suite root
if SUITE not in sys.path:
    sys.path.insert(0, SUITE)

# NO -ffast-math. It reassociates float ops, which makes the fused kernels
# disagree with the unfused reference (measured maxdiff ~1e-6, failing the
# bit-exact selftest on mv_headnorm / gate_up_act / qkv.q). Greedy decoding
# turns a 1e-6 logit difference into a different token, so bit-exactness is a
# correctness requirement here, not a nicety. The project's other build
# scripts still pass -ffast-math, which is worth revisiting.
CPU_FLAGS = ["-O3", "-mavx2", "-mfma", "-fopenmp"]

# Candidate C compilers, best first.
GCC_CANDIDATES = [
    r"C:\msys64\ucrt64\bin\gcc.exe",
    r"C:\msys64\mingw64\bin\gcc.exe",
    r"C:\msys64\clang64\bin\clang.exe",
    r"C:\mingw64\bin\gcc.exe",
]


def find_gcc():
    for c in GCC_CANDIDATES:
        if os.path.exists(c):
            return c
    for name in ("gcc", "clang"):
        p = shutil.which(name)
        if p:
            return p
    return None


# gcc finds `as`, `collect2` and `ld` through PATH, not relative to itself.
# With only the compiler directory on PATH it compiles but fails to link with
# "cannot execute collect2.exe: CreateProcess: No such file or directory".
# These are the runtime dirs that make the link step work.
TOOLCHAIN_PATH_DIRS = [
    r"C:\msys64\ucrt64\bin",
    r"C:\msys64\usr\bin",
    r"C:\msys64\mingw64\bin",
]

# NVRTC's build step loads nvrtc-builtins64_<ver>.dll by plain name. Windows
# resolves that against PATH and the current directory, and the builtins DLL
# ships in the suite root rather than next to nvrtc64_*.dll in nvrtc\. Without
# the suite root on PATH every PTX compile fails with
# "failed to open nvrtc-builtins64_126.dll ... NVRTC_ERROR_BUILTIN_OPERATION_FAILURE",
# which reads like a CUDA install problem but is purely a search-path problem.
_SUITE_ROOT = os.path.abspath(os.path.join(HERE, os.pardir, os.pardir))


def toolchain_env(extra_dirs=None):
    """Environment with the toolchain's assembler/linker on PATH.

    Also puts the suite root on PATH so NVRTC can find its builtins DLL.
    See _SUITE_ROOT above.
    """
    env = dict(os.environ)
    dirs = list(extra_dirs or [])
    for d in TOOLCHAIN_PATH_DIRS:
        if os.path.isdir(d):
            dirs.append(d)
    if os.path.isdir(_SUITE_ROOT):
        dirs.append(_SUITE_ROOT)
    if dirs:
        env["PATH"] = os.pathsep.join(dirs + [env.get("PATH", "")])
    return env


def find_nvrtc_build():
    """Find nvrtc_build.exe, if a developer built it offline.

    DEVELOPER/OFFLINE ONLY (2026-10-28 correction, Caleb: the auto
    engine is NOT expected to build PTX files at runtime — that is
    hard to auto-compute). The normal load path never calls this
    and never requires this exe to exist; it selects a prebuilt
    PTX via find_prebuilt_ptx() instead. Build the exe offline
    with gcc from nvrtc-kernels/nvrtc_build.c if a new PTX is ever
    needed, then place the resulting .ptx in nvrtc-kernels/.
    Also looks in nvrtc-kernels/ (where nvrtc_build.c lives), not
    only in the legacy nvrtc/ location.
    """
    for cand in (os.path.join(HERE, "nvrtc", "nvrtc_build.exe"),
                 os.path.join(HERE, "nvrtc-kernels", "nvrtc_build.exe")):
        if os.path.exists(cand):
            return cand
    return None


def find_prebuilt_ptx(arch="75", ptx_dirs=None):
    """Select a prebuilt PTX file for arch ("75" or "86"), or None.

    Selection only — this only looks for files that already exist,
    in nvrtc-kernels/ first, then nvrtc/ptx/. Prefers the generic
    gguf file, then hf, then any file with the matching __smNN
    suffix, then the base file when arch == "75" (base IS sm75).
    PTX here is model-agnostic (one md5 per arch), so a generic
    file is valid for any model. Returns an absolute path or None;
    a None result means the caller must fall back cleanly (stream,
    .ntmre compression, CPU, or AMD path) — it must NOT trigger a PTX build
    at load time.
    """
    dirs = list(ptx_dirs) if ptx_dirs else list(PTX_DIRS)
    candidates = []
    for stem in ("gguf__nvidia_kernels__kernels",
                 "hf__nvidia_kernels__kernels"):
        candidates.append("%s__sm%s.ptx" % (stem, arch))
    if arch == "75":
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
        for name in sorted(names):
            if name.endswith("__sm%s.ptx" % arch):
                return os.path.join(d, name)
        if arch == "75":
            for name in sorted(names):
                if name.endswith("__nvidia_kernels__kernels.ptx"):
                    return os.path.join(d, name)
    return None


def has_hiprtc():
    """True when a HIP runtime compiler is available on this host."""
    try:
        import ctypes
        ctypes.CDLL("hiprtc.dll")
        return True
    except Exception:
        pass
    for d in (r"C:\Program Files\AMD\ROCm\bin",):
        if os.path.exists(os.path.join(d, "hiprtc.dll")):
            return True
    return False


def run(cmd, timeout=1800, cwd=None, env=None):
    """Run a command, return (rc, output)."""
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, cwd=cwd,
                           env=env or toolchain_env())
        out = (r.stdout or b"").decode("utf-8", "replace") +\
              (r.stderr or b"").decode("utf-8", "replace")
        return r.returncode, out
    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"
    except Exception as e:
        return -2, f"{type(e).__name__}: {e}"


def slug(name):
    s = re.sub(r"[^a-z0-9]+", "_", str(name).lower())
    return s.strip("_")


# ── manifest ────────────────────────────────────────────────────────────────

def load_manifest():
    try:
        with open(MANIFEST, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_manifest(m):
    try:
        with open(MANIFEST, "w", encoding="utf-8") as f:
            json.dump(m, f, indent=2)
    except Exception:
        pass


# ── build steps ─────────────────────────────────────────────────────────────

def build_cpu_fused(src, out_exe, gcc, status=print):
    """Compile + self-verify the fused CPU kernels. Returns dict."""
    rc, out = run([gcc] + CPU_FLAGS + ["-DNTMRE_SELFTEST_MAIN", src, "-o",
                                      out_exe, "-lm"])
    if rc != 0:
        return {"ok": False, "stage": "compile", "detail": out.strip()[-400:]}
    # The binary self-tests: fused vs unfused must be bit-identical.
    passes, logs = 0, []
    for i in range(3):
        rc2, o2 = run([out_exe], timeout=300)
        logs.append(o2)
        if "RESULT: PASS" in o2:
            passes += 1
    ok = passes == 3
    tail = [ln for ln in (logs[-1] if logs else "").splitlines()
            if "selftest" in ln or "RESULT" in ln]
    return {"ok": ok, "stage": "verify" if ok else "selftest",
            "runs_passed": passes, "runs": 3,
            "detail": " | ".join(tail)}


# NVIDIA targets to ship PTX for. Emitted per-arch into the same filename
# stem with an __smNN suffix, so one source produces several kernels and the
# loader can pick at runtime (see arch_selector.py).
#
# 75  Turing     -- the original single target
# 80  Ampere A100
# 86  Ampere consumer (RTX 30xx). The most common consumer card still in
#      service, and the oldest arch that gets tensor cores and async copy.
# 89  Ada        -- RTX 40xx. Shares Ampere's PTX ISA, so 86 JITs upward.
# 90  Hopper     -- datacenter, for the enterprise tier.
NVIDIA_TARGETS = ("compute_75", "compute_86")

# Archs deliberately NOT targeted: anything below 7.5 (Maxwell, Pascal, Volta,
# Turing consumer) cannot run 7.5-class PTX efficiently and old Fermi cards
# like the GTX 660 are not NVRTC-capable at all. Those buyers route to the
# CPU engine via arch_selector rather than getting a kernel that would fail to
# load. Shipping a "sm_35" file that JITs through three ISA generations would
# be slower than the CPU path and would look like it worked.


def build_nvidia_ptx(src, out_ptx, nvrtc_build, arch="compute_75",
                     status=print):
    """Compile device kernels to PTX with NVRTC. Returns dict.

    out_ptx may be a directory target; when `archs` is passed instead of a
    single arch, one PTX is emitted per target with an __smNN suffix.
    """
    os.makedirs(os.path.dirname(out_ptx), exist_ok=True)
    rc, out = run([nvrtc_build, src, out_ptx, arch], timeout=600)
    if rc != 0 or not os.path.exists(out_ptx):
        return {"ok": False, "stage": "nvrtc", "detail": out.strip()[-400:]}
    try:
        with open(out_ptx, encoding="utf-8", errors="replace") as f:
            body = f.read()
        entries = len(re.findall(r"\.visible\s+\.entry", body))
    except Exception:
        entries = 0
    return {"ok": entries > 0, "stage": "nvrtc", "kernels": entries,
            "arch": arch}


def build_nvidia_ptx_multi(src, nvrtc_build, archs=None, status=print):
    """
    Compile one .cu to PTX for every NVIDIA target.

    Each artifact is written as
        <stem>__nvidia_kernels__kernels__sm<NN>.ptx
    so several generations coexist instead of overwriting each other.

    Returns a dict keyed by compute number. A single failing target does not
    fail the others -- partial coverage is useful, total absence is not.
    """
    archs = archs or NVIDIA_TARGETS
    out: Dict[str, Any] = {}
    os.makedirs(PTX_DIR, exist_ok=True)

    stem = os.path.basename(src)
    for suffix in (".cu",):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    base = stem[:-len("__nvidia_kernels__kernels")]\
        if stem.endswith("__nvidia_kernels__kernels") else stem

    for arch in archs:
        digits = arch.replace("compute_", "")
        dest = os.path.join(
            PTX_DIR, "%s__nvidia_kernels__kernels__sm%s.ptx" % (base, digits))
        r = build_nvidia_ptx(src, dest, nvrtc_build, arch=arch, status=status)
        out[digits] = r
        status("[BUILD]   nvidia sm_%s: %s" % (
            digits,
            ("%d kernels -> PTX" % r["kernels"]) if r.get("ok")
            else "FAILED: %s" % str(r.get("detail", ""))[:100]))
    return out


def build_amd(src, out_obj, status=print):
    """HIPRTC code object. Needs ROCm; report clearly when absent."""
    if not has_hiprtc():
        return {"ok": False, "skipped": True, "stage": "hiprtc",
                "detail": "no HIPRTC on this host (install ROCm) -- source "
                          "kept, engine will use the CPU or NVIDIA path"}
    return {"ok": False, "skipped": True, "stage": "hiprtc",
            "detail": "HIPRTC present but no hiprtc build harness in this "
                      "distribution; source kept for later"}


# ── orchestration ───────────────────────────────────────────────────────────

# generate_engines.py emits gemma4-shaped hosts: per-layer embeddings (PLE),
# dual head_dim tables, sliding-window RoPE bases and logit softcap.  Applying
# that template to a llama/qwen geometry would produce something that compiles
# and passes its own arithmetic selftest while being wrong for the model, so we
# only auto-build for architectures the generator actually models.
SUPPORTED_ARCH_PREFIXES = ("gemma4", "gemma-4", "gemma_4")


def _arch_cache_path():
    return os.path.join(HERE, "arch_cache.json")


_ARCH_CACHE = None


def _load_arch_cache():
    global _ARCH_CACHE
    if _ARCH_CACHE is None:
        try:
            with open(_arch_cache_path(), encoding="utf-8") as f:
                _ARCH_CACHE = json.load(f)
        except Exception:
            _ARCH_CACHE = {}
    return _ARCH_CACHE


def _save_arch_cache():
    try:
        with open(_arch_cache_path(), "w", encoding="utf-8") as f:
            json.dump(_ARCH_CACHE or {}, f, indent=2)
    except Exception:
        pass


def arch_supported(model_path, spec=None):
    """(bool, arch) - can we honestly generate an engine for this model?

    Results are cached in arch_cache.json keyed by (path, mtime, size), so a
    startup scan over hundreds of models parses each GGUF header at most once
    instead of on every launch."""
    arch = ""
    if spec:
        arch = str(spec.get("arch", "")).lower()
    if not arch:
        try:
            st = os.stat(model_path)
            ckey = "%s|%d|%d" % (os.path.abspath(model_path), int(st.st_mtime),
                                 int(st.st_size))
        except OSError:
            ckey = None
        cache = _load_arch_cache()
        if ckey and ckey in cache:
            return bool(cache[ckey].get("ok")), cache[ckey].get("arch", "")
        try:
            import gguf_parser
            meta = gguf_parser.GGUFParser(str(model_path)).parse().metadata or {}
            arch = str(meta.get("general.architecture", "")).lower()
        except Exception:
            arch = ""
    if not arch:
        low = os.path.basename(model_path).lower()
        arch = "gemma4" if "gemma4" in low or "gemma-4" in low else ""
    ok = any(arch.startswith(p) or p in arch for p in SUPPORTED_ARCH_PREFIXES)
    if ckey:
        cache = _load_arch_cache()
        cache[ckey] = {"ok": bool(ok), "arch": arch}
        _save_arch_cache()
    return ok, arch


def synthesize_spec(path, status=print):
    """Build a ModelSpec from a GGUF by sniffing tensors/metadata."""
    ok, arch = arch_supported(path)
    if not ok:
        return None, arch
    from spec_engine_runner import jit_synthesize_spec
    # forward status so the inner `except Exception` is visible instead of
    # silently returning None
    return jit_synthesize_spec(path, status=status), arch


# Files that are not LLM models and must never be measured as one.
# imatrix = calibration data, not a model. The audio/VAD/diffusion names
# below are weights for other jobs (denoising, voice activity, image
# diffusion) that happen to sit in the same model folders.
_SKIP_BUILD_NAME_PATTERNS = (
    "imatrix",
    "deepfilternet", "rnnoise", "gtcrn", "zipenhancer",
    "marblenet", "silero_vad", "unet.safetensors",
    "sd-turbo", "sd-v1", "sd-v1-5", "tiny-sd",
)
# Paths already measured in this process with no engine built. Without
# this, a startup scan that builds nothing re-measures the same files
# on every UI refresh, because a failed build is never "complete" in
# the manifest and therefore never skipped.
_MEASURED_NO_BUILD = set()


def build_for_model(path, force=False, status=print):
    """Measure, generate, compile and verify engines for one model.

    Returns a manifest dict. Never raises.
    """
    path = os.path.abspath(path)
    name = os.path.basename(path)
    result = {"model": name, "path": path, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "generated": [], "built": {}, "notes": []}

    if not os.path.exists(path):
        result["notes"].append("model file not found")
        return result

    # Not an LLM model: skip before any measuring print or GGUF parse.
    _low = name.lower()
    if any(p in _low for p in _SKIP_BUILD_NAME_PATTERNS):
        result["notes"].append("skipped: not an LLM model file")
        return result

    # NOTE 2026-11-01 (reverted): an earlier fix skipped the whole
    # scan in a sealed copy where generate_engines.py is absent.
    # That silenced the auto engine entirely, which read as "it
    # does not even check models". The auto engine's job is to
    # CHECK models first: measuring/spec synthesis still runs here,
    # and only the generation step below fails cleanly with a note
    # when generate_engines.py is not shipped. Dedupe + session
    # skip + non-LLM skip above are what stop the repeats, not
    # silencing the first check.
    if not force and path in _MEASURED_NO_BUILD:
        result["notes"].append("already measured in this session, no engine built")
        return result

    man = load_manifest()
    key = os.path.basename(path)
    prev = man.get(key)
    if prev and not force and prev.get("complete"):
        result["notes"].append("already built " + prev.get("time", ""))
        result["built"] = prev.get("built", {})
        result["cached"] = True
        return result

    # 1. spec
    status(f"[BUILD] {name}: measuring model...")
    try:
        spec, arch = synthesize_spec(path, status=status)
        result["arch"] = arch
        if not spec:
            if arch and not any(arch.startswith(p) or p in arch
                                for p in SUPPORTED_ARCH_PREFIXES):
                result["notes"].append(
                    f"arch '{arch}' has no engine template in this suite; "
                    "nothing generated (a wrong engine is worse than none)")
            else:
                result["notes"].append("no spec synthesized")
            _MEASURED_NO_BUILD.add(path)
            return result
    except Exception as e:
        result["notes"].append(f"spec synthesis failed: {type(e).__name__}: {e}")
        _MEASURED_NO_BUILD.add(path)
        return result

    mname = spec.get("name") or name
    s = slug(mname)
    result["spec_name"] = mname
    result["arch"] = spec.get("arch")
    result["n_layers"] = spec.get("n_layers")

    # 1b. sidecars, before any kernel is generated.
    #
    # A model with no .tokpatch/.f32patch produces plausible-looking noise,
    # not an error, so kernels compiled against it are wasted work and the
    # manifest would otherwise claim a complete build for a model that
    # cannot talk. Report the repair state alongside the engine state.
    if name.lower().endswith(".ntmre"):
        try:
            import ntmre_sidecars
            sc = ntmre_sidecars.build_sidecars(path, status=status)
            result["sidecars"] = {
                "complete": sc.get("ok"),
                "actions": sc.get("actions", []),
                "notes": sc.get("notes", []),
                "integrity": sc.get("integrity", {}),
            }
            if sc.get("ok"):
                status(f"[BUILD]   sidecars: complete")
            else:
                status(f"[BUILD]   sidecars: INCOMPLETE "
                       f"(engine builds anyway, output will be degraded)")
        except Exception as e:
            result["sidecars"] = {"complete": False,
                                  "notes": [f"{type(e).__name__}: {e}"]}
            status(f"[BUILD]   sidecars: check failed ({e})")

    # 2. generate sources
    if not os.path.exists(os.path.join(HERE, "generate_engines.py")):
        # Sealed/sale edition: spec was measured above, generation
        # itself is a dev tool and is not shipped. Say so plainly
        # instead of a generic "generation failed".
        result["notes"].append(
            "spec measured; engine generation not shipped in this "
            "edition (load uses prebuilt PTX instead)")
        status(f"[BUILD] {name}: spec checked, generation not shipped in this edition")
        _MEASURED_NO_BUILD.add(path)
        return result
    try:
        import generate_engines
        written = generate_engines.generate_for_single_spec(spec, GEN_DIR)
        result["generated"] = [os.path.basename(p) for p, _ in written]
        status(f"[BUILD] {name}: generated {len(written)} engine renderings")
    except Exception as e:
        result["notes"].append(f"generation failed: {type(e).__name__}: {e}")
        _MEASURED_NO_BUILD.add(path)
        return result

    # 3. compile
    os.makedirs(FUSED_DIR, exist_ok=True)
    os.makedirs(OBJ_DIR, exist_ok=True)

    fused_src = os.path.join(GEN_DIR, f"{s}__cpu_fused__fused.c")
    nv_src = os.path.join(GEN_DIR, f"{s}__nvidia_kernels__kernels.cu")
    amd_src = os.path.join(GEN_DIR, f"{s}__amd_kernels__kernels.hip")

    gcc = find_gcc()
    if gcc and os.path.exists(fused_src):
        exe = os.path.join(FUSED_DIR, f"{s}__cpu_fused__fused.exe")
        r = build_cpu_fused(fused_src, exe, gcc, status)
        r["artifact"] = exe
        result["built"]["cpu_fused"] = r
        status(f"[BUILD]   cpu_fused: "
               + ("VERIFIED bit-identical x3" if r["ok"] else
                  f"FAILED ({r.get('stage')}): {r.get('detail','')[:120]}"))
    else:
        result["notes"].append("no C compiler found; CPU fused kernels not built")

    nb = find_nvrtc_build()
    if nb and os.path.exists(nv_src):
        # One compile per NVIDIA target so a single arch file cannot shadow
        # the others. arch_selector.py picks between them at load time.
        per_arch = build_nvidia_ptx_multi(nv_src, nb, status=status)
        built = {k: v for k, v in per_arch.items() if v.get("ok")}
        result["built"]["nvidia_device"] = {
            "ok": bool(built),
            "stage": "nvrtc",
            "targets": sorted(built.keys()),
            "kernels": (list(built.values())[0].get("kernels")
                        if built else 0),
            "detail": "; ".join(
                "sm_%s: %s" % (k, v.get("kernels") if v.get("ok")
                               else str(v.get("detail", ""))[:80])
                for k, v in sorted(per_arch.items())),
        }
        status("[BUILD]   nvidia: %d/%d targets -> PTX" % (
            len(built), len(per_arch)))
    else:
        # Offline/dev only: PTX building is NOT part of the normal
        # load path (Caleb 2026-10-28). Load selects a prebuilt PTX
        # via find_prebuilt_ptx(); a missing nvrtc_build.exe does
        # not block loading. This note only means this offline
        # build step did not produce a new PTX here.
        result["notes"].append(
            "nvrtc_build.exe missing; no new PTX built offline "
            "(load path selects prebuilt PTX instead)")

    if os.path.exists(amd_src):
        r = build_amd(amd_src, os.path.join(OBJ_DIR, f"{s}__amd.hip.o"))
        result["built"]["amd_device"] = r
        status(f"[BUILD]   amd: " + ("built" if r["ok"] else r["detail"][:100]))

    result["complete"] = any(v.get("ok") for v in result["built"].values())
    if not result["complete"]:
        _MEASURED_NO_BUILD.add(path)
    man[key] = result
    save_manifest(man)
    return result


def scan(paths, force=False, status=print):
    """Auto-build for every model in `paths` that has no working engine yet."""
    out = []
    _seen = set()
    for p in paths:
        _ap = os.path.abspath(p) if p else ""
        if not _ap or _ap in _seen:
            continue
        _seen.add(_ap)
        try:
            r = build_for_model(p, force=force, status=status)
            if r.get("notes") or r.get("built"):
                out.append(r)
        except Exception as e:
            status(f"[BUILD] {os.path.basename(p)}: {type(e).__name__}: {e}")
    return out


_AUTO_STATE = {"running": False, "done": 0, "total": 0, "built": 0,
               "last": "", "error": None}


def auto_state():
    """Snapshot of the background auto-build, for the UI to poll."""
    return dict(_AUTO_STATE)


def ensure_engines_async(paths, status=None, on_done=None):
    """Kick off an auto-build for `paths` in a background daemon thread.

    Safe to call on every startup / model refresh: already-built models are
    skipped via the manifest, and an in-flight run is not duplicated.
    """
    import threading
    if _AUTO_STATE["running"]:
        return False
    _seen = set()
    paths = []
    for p in (paths or []):
        if not p or not os.path.exists(p):
            continue
        _ap = os.path.abspath(p)
        if _ap in _seen:
            continue
        _seen.add(_ap)
        paths.append(p)
    if not paths:
        return False

    def log(msg):
        _AUTO_STATE["last"] = msg
        if status:
            try:
                status(msg)
            except Exception:
                pass

    def worker():
        _AUTO_STATE["running"] = True
        _AUTO_STATE["done"] = 0
        _AUTO_STATE["total"] = len(paths)
        _AUTO_STATE["built"] = 0
        _AUTO_STATE["error"] = None
        try:
            for p in paths:
                try:
                    r = build_for_model(p, force=False, status=log)
                    _AUTO_STATE["done"] += 1
                    if r.get("built") and not r.get("cached"):
                        _AUTO_STATE["built"] += 1
                except Exception as e:
                    _AUTO_STATE["error"] = f"{type(e).__name__}: {e}"
                    log(f"[BUILD] {os.path.basename(p)}: {e}")
        finally:
            _AUTO_STATE["running"] = False
            if on_done:
                try:
                    on_done(_AUTO_STATE["built"])
                except Exception:
                    pass

    t = threading.Thread(target=worker, daemon=True,
                         name="nightmare-autobuild")
    t.start()
    return True


def summary():
    man = load_manifest()
    if not man:
        return "no engines built yet"
    lines = [f"auto-built engines: {len(man)}"]
    for k, v in sorted(man.items()):
        bits = []
        for target, r in (v.get("built") or {}).items():
            if r.get("ok"):
                extra = ""
                if target == "cpu_fused":
                    extra = f" x{r.get('runs_passed')}"
                elif target == "nvidia_device":
                    extra = f" {r.get('kernels')}k"
                bits.append(f"{target}=OK{extra}")
            elif r.get("skipped"):
                bits.append(f"{target}=skipped")
            else:
                bits.append(f"{target}=FAIL")
        lines.append(f"  {k[:44]:44s} {' '.join(bits)}")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--summary":
        print(summary())
    elif len(sys.argv) > 1 and sys.argv[1] == "--scan":
        import ntmre_router
        found = ntmre_router.discover_models(SUITE)
        scan(list(found.values()), status=print)
        print()
        print(summary())
    elif len(sys.argv) > 1:
        for p in sys.argv[1:]:
            r = build_for_model(p, status=print)
            print(json.dumps({k: v for k, v in r.items() if k != "generated"},
                             indent=2)[:900])
    else:
        print("usage: auto_engine_builder.py <model> [<model> ...]")
        print("       auto_engine_builder.py --scan")
        print("       auto_engine_builder.py --summary")
        print()
        print(summary())