# -*- coding: utf-8 -*-
"""
system_planner.py — enterprise capability model + device/ engine planner
=========================================================================
Replaces the old "is there a GPU? yes/no" boolean with a real model of the
machine, so the router can pick the best combination per system instead of
hardcoding two GPUs and hoping.

What it knows about:
  * every accelerator, keyed by VENDOR (nvidia:0 != amd:0 -- they are
    different physical cards, and feeding a vendor-local index to a CUDA
    ordinal silently targets the wrong GPU)
  * real VRAM totals, taken from the vendor's own reporting tool, with the
    0xFFFFFFFF clamp sentinel rejected
  * CPU physical cores vs SMT threads, plus runtime ISA detection
  * live RAM and the OS safety floor

What it decides:
  * which devices to use (and how many) for a given model footprint
  * resident-fused vs streaming vs CPU-only
  * how to split layers across devices, proportional to each card's VRAM
  * how many CPU threads, and pins to physical cores only

Zero third-party dependencies: ctypes / os / sys only.
"""
import ctypes
import os
import sys

# ── CPU feature detection (runtime, not compile-time) ────────────────────────

def _cpu_has(feature):
    """True if the CPU supports `feature` ('avx2', 'fma', 'avx512f', ...)."""
    try:
        import platform
        machine = platform.machine().lower()
    except Exception:
        machine = ""
    if os.name == 'nt':
        try:
            import winreg
            k = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            feats, _ = winreg.QueryValueEx(k, "FeatureSet")
            # Windows FeatureSet bit positions (documented, not guessed).
            # Verified against this machine: i7-8700 sets 12(FMA), 28(AVX),
            # 29(AVX2) and leaves 40 clear.
            bits = {
                'sse3': 0, 'ssse3': 9, 'sse4_1': 19, 'sse4_2': 20,
                'fma': 12, 'avx': 28, 'avx2': 29,
            }
            if feature in bits:
                return bool((int(feats) >> bits[feature]) & 1)
            if feature.startswith('avx512'):
                return False
        except Exception:
            pass
    # Non-Windows / registry unavailable: ask GCC/Clang if we can.
    try:
        if feature == 'avx2':
            return _gcc_has('avx2')
        if feature == 'fma':
            return _gcc_has('fma')
        if feature == 'avx':
            return _gcc_has('avx')
    except Exception:
        pass
    return False


def _gcc_has(feat):
    """Ask the C compiler what it thinks, via __builtin_cpu_supports."""
    import subprocess
    import tempfile
    src = ('int main(void){return __builtin_cpu_supports("%s")?0:1;}' % feat)
    with tempfile.NamedTemporaryFile('w', suffix='.c', delete=False) as f:
        f.write(src)
        p = f.name
    for cc in ("gcc", "clang"):
        try:
            r = subprocess.run([cc, "-O1", "-o", p + ".exe", p],
                               capture_output=True, timeout=20)
            if r.returncode == 0:
                x = subprocess.run([p + ".exe"], capture_output=True, timeout=10)
                return x.returncode == 0
        except Exception:
            continue
    return False


def cpu_isa():
    """Runtime CPU capability summary."""
    return {
        'avx': _cpu_has('avx'),
        'avx2': _cpu_has('avx2'),
        'fma': _cpu_has('fma'),
        'avx512f': _cpu_has('avx512f'),
    }


# ── Device model ─────────────────────────────────────────────────────────────

class Device:
    """One physical accelerator, with a vendor-qualified identity."""

    __slots__ = ("vendor", "index", "name", "runtime", "total", "free",
                 "drivable", "key", "compute_cap")

    def __init__(self, d):
        self.vendor = d.get("vendor", "?")
        self.index = int(d.get("index", 0))
        self.name = d.get("name", "")
        self.runtime = d.get("runtime", "")
        self.total = int(d.get("total", 0) or 0)
        self.free = int(d.get("free", 0) or 0)
        self.drivable = bool(d.get("drivable", False))
        self.key = f"{self.vendor}:{self.index}"
        # Compute capability, when probed (nightmare_gpu_discovery
        # now reports it from cuDeviceGetAttribute / nvidia-smi).
        # None = unknown; the NVRTC selector falls back to sm75
        # (JITs upward) with that reason stated, never a hidden guess.
        self.compute_cap = d.get("compute_cap")

    @property
    def total_gb(self):
        return self.total / (1024 ** 3)

    @property
    def free_gb(self):
        return self.free / (1024 ** 3)

    def usable_gb(self, headroom=0.90):
        """Free VRAM we are willing to actually fill."""
        base = self.free if self.free > 0 else self.total
        return base * headroom / (1024 ** 3)

    def __repr__(self):
        return (f"<Device {self.key} {self.name[:26]!r} "
                f"{self.free_gb:.1f}/{self.total_gb:.1f}GiB>")

    def as_dict(self):
        return {
            "vendor": self.vendor, "index": self.index, "key": self.key,
            "name": self.name, "runtime": self.runtime,
            "total": self.total, "free": self.free,
            "total_gb": round(self.total_gb, 2), "free_gb": round(self.free_gb, 2),
            "drivable": self.drivable,
            "compute_cap": self.compute_cap,
        }


# ── System profile ───────────────────────────────────────────────────────────

class SystemProfile:
    """A complete picture of the machine, gathered once."""

    def __init__(self):
        self.devices = []
        self.cpu_brand = ""
        self.phys_cores = 1
        self.logical = 1
        self.affinity_mask = None
        self.isa = {}
        self.ram_total = 0
        self.ram_free = 0

    # -- construction -------------------------------------------------
    @classmethod
    def probe(cls, force=False):
        self = cls()
        self._probe_cpu()
        self._probe_ram()
        self._probe_gpu(force)
        return self

    def _probe_cpu(self):
        try:
            import nightmare_hardware as hw
            self.cpu_brand = hw.get_cpu_brand()
            self.phys_cores, self.logical, self.affinity_mask =\
                hw.get_cpu_topology()
            ram = hw.get_system_ram()
            self.ram_total = ram['total_bytes']
            self.ram_free = ram['avail_bytes']
        except Exception:
            self.cpu_brand = platform_cpu_name()
            self.logical = os.cpu_count() or 1
            self.phys_cores = max(1, self.logical // 2)
        self.isa = cpu_isa()

    def _probe_ram(self):
        if self.ram_total:
            return
        try:
            import nightmare_hardware as hw
            ram = hw.get_system_ram()
            self.ram_total = ram['total_bytes']
            self.ram_free = ram['avail_bytes']
        except Exception:
            self.ram_total = 16 * 1024 ** 3
            self.ram_free = 8 * 1024 ** 3

    def _probe_gpu(self, force):
        try:
            from nightmare_gpu_discovery import probe_all
            self.devices = [Device(d) for d in probe_all(force=force)]
        except Exception:
            self.devices = []

    # -- derived views ------------------------------------------------
    def drivable_devices(self):
        """Devices the engine can actually launch kernels on, best first."""
        out = [d for d in self.devices if d.drivable]
        out.sort(key=lambda d: (d.free if d.free > 0 else d.total), reverse=True)
        return out

    def total_vram_gb(self, headroom=0.90):
        return round(sum(d.usable_gb(headroom) for d in self.drivable_devices()), 2)

    def device_summary(self):
        lines = []
        for d in sorted(self.devices, key=lambda x: x.key):
            tag = "usable" if d.drivable else "detected-only"
            vram = (f"{d.free_gb:.1f}/{d.total_gb:.1f} GiB"
                    if d.total else "VRAM unknown")
            lines.append(f"  {d.key:12s} {d.name[:34]:34s} {vram:>18s}  {tag}")
        if not lines:
            lines.append("  no accelerators detected")
        return "\n".join(lines)

    def describe(self):
        isa = ", ".join(k.upper() for k, v in self.isa.items() if v) or "unknown"
        return (
            f"CPU      : {self.cpu_brand}\n"
            f"            {self.phys_cores} physical cores / {self.logical} "
            f"logical threads\n"
            f"ISA       : {isa}\n"
            f"RAM       : {self.ram_free/1024**3:.1f} GiB free of "
            f"{self.ram_total/1024**3:.1f} GiB\n"
            f"Devices   : {len(self.drivable_devices())} drivable of "
            f"{len(self.devices)} detected\n"
            f"Usable VRAM: {self.total_vram_gb()} GiB\n"
        )


def platform_cpu_name():
    try:
        import platform
        return platform.processor() or "unknown CPU"
    except Exception:
        return "unknown CPU"


# ── Plan ─────────────────────────────────────────────────────────────────────

def derive_max_ctx(profile, plan, n_layers=0, n_kv_heads=0, head_dim=0,
                   kv_bytes=2, host_gb=None):
    """
    Work out the largest context this machine can actually hold.

    WHY THIS EXISTS
    ---------------
    The streaming engine capped context at a literal 12000 in three
    places. A typed-in number is wrong on both sides: on a box with 8
    cards it holds a 26B back to 12000 when the hardware could hold
    3x that, and on a small box it sits ABOVE what is safe, so it
    protects nobody. The user running this on their own hardware cannot
    see the number is a guess.

    So the ceiling is computed from what is actually present:
    free VRAM, the model's own trained context, and the measured cost
    per token of the KV cache.

    THE MEASUREMENT THAT MATTERS
    ----------------------------
    kv_dim must come from the model's tensors, not from a header field.
    gemma4 advertises head_dim 512 in its metadata while the tensors are
    256, and its per-layer KV widths differ between the full-attention
    and sliding-window layers. Guessing this produced a 2x error and a
    set of ceilings that were double what the hardware could do. The
    caller passes what it measured; if it measured nothing, this
    returns None rather than inventing a number.

    Returns (max_ctx, basis) where basis explains the choice.
    """
    if not (n_layers and n_kv_heads and head_dim):
        # Not enough measured to derive a number. Say so instead of
        # falling back to a default -- a silent default is the bug this
        # function exists to remove.
        return None, ""

    # K and V, each n_kv_heads * head_dim, at kv_bytes each.
    per_token = int(n_layers) * int(n_kv_heads) * int(head_dim) *\
        int(kv_bytes) * 2
    if per_token <= 0:
        return None, ""

    # Memory this machine can actually give to the cache.
    budget_gb = 0.0
    for d in (plan.devices or []):
        budget_gb += d.usable_gb(VRAM_HEADROOM)
    if host_gb:
        budget_gb += float(host_gb)
    if budget_gb <= 0:
        return None, ""

    budget_bytes = int(budget_gb * (1024 ** 3))
    by_memory = budget_bytes // per_token

    # The model knows its own trained length; past it, attention has no
    # learned behaviour however much memory is free.
    trained = getattr(plan, "model_trained_ctx", 0) or 0

    if trained and by_memory > trained:
        return int(trained), ("model's trained limit (%d); memory could "
                              "hold %d" % (trained, by_memory))
    return int(by_memory), ("%.1f GB usable / %d bytes per token"
                            % (budget_gb, per_token))


class Plan:
    """The chosen configuration for one model on this machine."""

    def __init__(self):
        self.engine = "cpu"          # cpu | stream | fused | gpu_native
        self.devices = []            # [Device] in placement order
        self.split = []              # [(device, n_layers)] or [] if not fused
        self.cpu_threads = 1
        self.pin_physical = False
        self.notes = []
        self.warnings = []
        # NVRTC / spec-based choice (2026-10-28, admin copy). Filled
        # by plan_for_model() from nightmare_nvrtc_select (selection
        # from prebuilt PTX only, never a PTX build). Empty = not
        # chosen yet; engine_reason always explains a CPU/stream
        # fallback instead of leaving it silent.
        self.nvrtc_arch = ""
        self.nvrtc_ptx = ""
        self.engine_mode = ""
        self.engine_reason = ""
        self.nvrtc_vendor = ""
        # Context length, DERIVED from this machine's free memory and the
        # model's own trained limit. Never a typed-in constant: a fixed
        # number silently under-drives a bigger machine and silently
        # over-drives a smaller one, and neither failure is visible.
        self.max_ctx = None          # None = engine derives it at load
        self.ctx_basis = ""          # why, shown in the plan log
        self.kv_bytes_per_token = 0  # measured, not assumed
        # Bytes the engine may spend on the KV cache, from the same
        # measurement that produced max_ctx. The engine used to default
        # this to a typed 2.0 GB, which is a guess about someone else's
        # RAM; it now reads a number derived from this machine.
        self.kv_budget_bytes = None

    @property
    def device_count(self):
        return len(self.devices)

    def describe(self):
        lines = [f"  engine      : {self.engine}"]
        if self.engine_mode:
            lines.append(f"  engine_mode : {self.engine_mode}")
        if self.nvrtc_ptx:
            lines.append(f"  nvrtc_ptx   : {self.nvrtc_ptx} (sm{self.nvrtc_arch})")
        if self.engine_reason:
            lines.append(f"  reason      : {self.engine_reason}")
        if self.devices:
            lines.append("  devices     : " +
                         ", ".join(f"{d.key}({d.free_gb:.1f}G)" for d in self.devices))
        if self.split:
            lines.append("  layer split : " +
                         " ".join(f"{d.key}:[{a},{b})"
                                  for d, a, b in self.split))
        if self.max_ctx:
            lines.append("  max context : %d%s"
                         % (self.max_ctx,
                            ("  (%s)" % self.ctx_basis) if self.ctx_basis else ""))
        lines.append(f"  cpu threads : {self.cpu_threads}"
                     + ("  (pinned to physical cores)"
                        if self.pin_physical else ""))
        for n in self.notes:
            lines.append(f"  note        : {n}")
        for w in self.warnings:
            lines.append(f"  WARNING     : {w}")
        return "\n".join(lines)


# ── The planner ──────────────────────────────────────────────────────────────

# Layer sizes that are safe to assume for engines that stream by layer.
# Overridable per engine; the caller passes n_layers when known.
VRAM_HEADROOM = 0.90      # never plan to fill a card completely
LOGITS_HEADROOM = 0.15    # extra 15% for logits/embeddings/KV


def plan_for_model(profile, model_bytes, n_layers=0, arch="", force_cpu=False,
                   max_devices=None):
    """Choose the best engine + device combination for `model_bytes`.

    `model_bytes`  on-disk or VRAM footprint of the weights
    `n_layers`     0 when unknown (then layer split is omitted)
    `max_devices`  cap the device count (None = use every card that helps)
    """
    plan = Plan()
    devs = [] if force_cpu else profile.drivable_devices()

    if max_devices is not None:
        devs = devs[:max_devices]

    # NVRTC / spec-based choice (2026-10-28, admin copy). Selection
    # from prebuilt PTX only — never a PTX build at load (Caleb
    # 2026-10-28). Computed here so the planner, the router, and the
    # resolver all read the same choice. It does not replace the
    # VRAM-fit logic below, except for the cases where that logic
    # would assign the wrong vendor/arch engine (AMD-only, or a
    # card below sm75): those override to a clean fallback with a
    # reason, instead of a silent NVIDIA assignment.
    _combo = None
    try:
        from nightmare_nvrtc_select import choose_combo
        _combo = choose_combo(devs, model_bytes=model_bytes)
        plan.nvrtc_arch = _combo.get("ptx_arch") or ""
        plan.nvrtc_ptx = _combo.get("ptx_path") or ""
        plan.engine_mode = _combo.get("engine_mode") or ""
        plan.engine_reason = _combo.get("reason") or ""
        plan.nvrtc_vendor = _combo.get("vendor") or ""
    except Exception as _e:
        plan.engine_reason = ("NVRTC selection unavailable (%s) - "
                              "planner VRAM logic below still applies"
                              % _e)

    need = model_bytes * (1.0 + LOGITS_HEADROOM)

    # No usable accelerator -> CPU path, sized to physical cores.
    if not devs:
        plan.engine = "cpu"
        plan.cpu_threads = profile.phys_cores
        plan.pin_physical = profile.affinity_mask is not None
        plan.warnings.append("no drivable accelerator - CPU only")
        return plan

    # AMD-only: NVIDIA PTX does not apply. Do not assign gpu_native
    # (an NVIDIA engine) to an AMD card — that was the old silent
    # misassignment. Stream / HIPRTC path, with the reason stated.
    if _combo is not None and _combo.get("vendor") == "amd":
        plan.engine = "stream"
        plan.devices = devs
        plan.cpu_threads = profile.phys_cores
        plan.pin_physical = profile.affinity_mask is not None
        plan.warnings.append(plan.engine_reason)
        return plan

    # Card below sm75: no prebuilt PTX. CPU, with the reason stated,
    # not a PTX build and not a failed GPU load.
    if _combo is not None and _combo.get("engine_mode") == "cpu":
        plan.engine = "cpu"
        plan.devices = []
        plan.cpu_threads = profile.phys_cores
        plan.pin_physical = profile.affinity_mask is not None
        plan.warnings.append(plan.engine_reason)
        return plan

    # Accumulate devices until the footprint fits.
    agg = sum(d.usable_gb(VRAM_HEADROOM) for d in devs)
    agg_bytes = agg * (1024 ** 3)

    if agg_bytes < need:
        # Doesn't fit fully resident anywhere. Streaming over every card plus
        # the host RAM cache is still far better than CPU-only.
        plan.engine = "stream"
        plan.devices = devs
        plan.cpu_threads = profile.phys_cores
        plan.pin_physical = profile.affinity_mask is not None
        plan.warnings.append(
            f"model needs ~{need/1024**3:.1f} GiB but only {agg:.1f} GiB "
            f"usable across {len(devs)} device(s) - streaming + host cache")
        return plan

    # It fits. Now the question is throughput, not capacity.
    #
    # Measured on a 2x RTX 3060 box with a 1.04 GB Qwen2.5-1.5B:
    #     1 GPU (gpu_native)  : 6.05 tok/s
    #     2 GPUs (fused)      : 9.74 tok/s   <- +61%
    # Spare VRAM on an idle card is worth far more than the per-layer-boundary
    # device-to-device hop of the hidden state, so: use EVERY card we have.
    # Only a single card can be reported as gpu_native.
    if len(devs) == 1:
        plan.engine = "gpu_native"
        plan.devices = devs
    else:
        plan.engine = "fused"
        plan.devices = devs
        if n_layers > 0:
            plan.split = layer_split(devs, n_layers)
        plan.notes.append(
            f"model fits on one card ({devs[0].free_gb:.1f} GiB free) but is "
            f"split across {len(devs)} for throughput")

    plan.cpu_threads = profile.phys_cores
    plan.pin_physical = profile.affinity_mask is not None
    return plan


def layer_split(devices, n_layers, usable_frac=VRAM_HEADROOM):
    """Split `n_layers` across devices in proportion to usable VRAM.

    Returns [(device, first_layer, end_layer_exclusive)].  Every layer is
    assigned exactly once and the ranges are contiguous, so the fused
    forward pass can walk them in order.
    """
    weights = [max(0.01, d.usable_gb(usable_frac)) for d in devices]
    total = sum(weights)
    counts = [max(1, int(round(n_layers * w / total))) for w in weights]

    # Fix rounding drift: force the sum back to n_layers.
    while sum(counts) > n_layers:
        i = counts.index(max(counts))
        if counts[i] <= 1:
            break
        counts[i] -= 1
    while sum(counts) < n_layers:
        i = counts.index(min(counts))
        counts[i] += 1

    out, cur = [], 0
    for d, c in zip(devices, counts):
        out.append((d, cur, cur + c))
        cur += c
    return out


def apply_cpu_settings(plan):
    """Push the plan's CPU thread/pinning decisions into the environment."""
    os.environ["NIGHTMARE_CPU_THREADS"] = str(plan.cpu_threads)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, str(plan.cpu_threads))

    if plan.pin_physical and os.name == 'nt':
        try:
            from nightmare_hardware import get_cpu_topology
            _, _, mask = get_cpu_topology()
            if mask:
                from ctypes import wintypes
                k32 = ctypes.WinDLL('kernel32', use_last_error=True)
                k32.GetCurrentProcess.restype = wintypes.HANDLE
                k32.SetProcessAffinityMask.restype = wintypes.BOOL
                k32.SetProcessAffinityMask.argtypes = [wintypes.HANDLE,
                                                      ctypes.c_size_t]
                return bool(k32.SetProcessAffinityMask(
                    k32.GetCurrentProcess(), ctypes.c_size_t(mask)))
        except Exception:
            return False
    return False


if __name__ == '__main__':
    p = SystemProfile.probe()
    print("=" * 70)
    print(p.describe())
    print("  device detail:")
    print(p.device_summary())
    print("=" * 70)

    # Example: plan for a 1 GB and a 40 GB model.
    for gb in (1.0, 13.0, 40.0):
        pl = plan_for_model(p, int(gb * 1024 ** 3), n_layers=28, arch="qwen2")
        print(f"\nmodel {gb:.0f} GiB, 28 layers:")
        print(pl.describe())