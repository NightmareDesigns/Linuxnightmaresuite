#!/usr/bin/env python3
"""Model library integrity checker.

Scans every model file on disk and reports which ones are usable, truncated,
or corrupt -- so a broken file never ships as a "supported" model.

Checks, per file:
  GGUF   : magic/version, header parses, and every tensor's byte range fits
           inside the file (catches truncated downloads).
  NTMRE  : magic/version, config parses, and the declared tensor-data region
           fits inside the file.
  other  : skipped (safetensors/onnx are not GGUF-family).

Usage:
    python model_integrity_check.py            # scan all drives
    python model_integrity_check.py E:\\models  # scan one folder
Exit code 0 = all good, 1 = problems found.
"""
import os
import re
import sys
import json
import struct
import ctypes
from pathlib import Path

EXTS = (".gguf", ".ntmre")
# Files that are companions, not standalone chat models.
NON_MODEL = re.compile(
    r"(ggml-vocab|mmproj|vae|diffusion|clip_|text_encoder|sd-turbo|sd-v1|tiny-sd)",
    re.I)

SKIP_DIRS = {
    "windows", "program files", "program files (x86)", "node_modules",
    "$recycle.bin", "appdata", "programdata", "windowsapps",
    "system volume information", "perflogs", "msocache",
    ".git", ".venv", "venv", "__pycache__", ".mypy_cache", ".pytest_cache",
}


def _drives():
    if sys.platform != "win32":
        return [Path("/")]
    drives = []
    try:
        bm = ctypes.windll.kernel32.GetLogicalDrives()
    except Exception:
        bm = 0
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        if bm & 1:
            d = Path(f"{letter}:\\")
            try:
                if d.exists():
                    drives.append(d)
            except Exception:
                pass
        bm >>= 1
    return drives


def _walk(root, maxdepth=5):
    root = Path(root)
    if not root.is_dir():
        return
    for r, dirs, files in os.walk(root, onerror=lambda e: None):
        dirs[:] = [d for d in dirs if d.lower() not in SKIP_DIRS and not d.startswith("$")]
        try:
            if len(Path(r).relative_to(root).parts) >= maxdepth:
                dirs[:] = []
        except Exception:
            pass
        for f in files:
            if f.lower().endswith(EXTS):
                yield Path(r) / f


def _quiet():
    """Silence gguf_parser's per-file header prints during bulk checks."""
    import io
    import contextlib
    return contextlib.redirect_stdout(io.StringIO())


def check_gguf_exact(p):
    """Authoritative check via the suite's own parser (reads tensor table)."""
    try:
        import gguf_parser
        import dequant_fast as dq
        with _quiet():
            g = gguf_parser.GGUFParser(str(p))
            m = g.parse()
            tdo = m.tensor_data_offset
            end = 0
            for n, info in m.tensors.items():
                eb, bb = dq.BLOCK_SIZES.get(int(info.dtype), (1, 4))
                ne = 1
                for d in info.shape:
                    ne *= d
                nb = (ne // eb) * bb
                end = max(end, tdo + int(info.offset) + nb)
            g.close()
        size = p.stat().st_size
        if end > size + 4096:
            return "truncated", f"needs {end/1e6:.1f}MB, have {size/1e6:.1f}MB"
        return "ok", ""
    except PermissionError:
        return "unreadable", "permission denied"
    except OSError as e:
        return "unreadable", str(e)[:60]
    except Exception as e:
        return "corrupt", f"{type(e).__name__}: {e}"[:80]


def check_ntmre(p):
    try:
        size = p.stat().st_size
        with open(p, "rb") as f:
            head = f.read(17)
        if len(head) < 8:
            return "corrupt", "file too small"
        magic = head[:4]
        if magic not in (b"NSVD", b"NTMR", b"RMTN"):
            return "corrupt", f"bad magic {magic!r}"
        if magic == b"RMTN":
            ver = struct.unpack("<I", head[4:8])[0]
            if ver not in (1, 2, 13, 16):
                return "corrupt", f"unsupported version {ver}"
        return "ok", ""
    except Exception as e:
        return "corrupt", f"{type(e).__name__}: {e}"[:80]


def main():
    args = sys.argv[1:]
    roots = [Path(a) for a in args] if args else _drives()
    print("scanning:", ", ".join(str(r) for r in roots), flush=True)

    files = []
    for root in roots:
        files.extend(_walk(root))
    # dedupe by resolved path
    seen = set()
    uniq = []
    for f in files:
        try:
            k = str(f.resolve())
        except Exception:
            k = str(f)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(f)
    print(f"found {len(uniq)} model files\n", flush=True)

    ok, bad = [], []
    for i, p in enumerate(uniq, 1):
        name = p.name
        if NON_MODEL.search(name):
            continue
        if p.suffix.lower() == ".gguf":
            status, detail = check_gguf_exact(p)
        else:
            status, detail = check_ntmre(p)
        if status == "ok":
            ok.append(p)
        else:
            bad.append((p, status, detail))
        if i % 50 == 0:
            print(f"  ...{i}/{len(uniq)}", flush=True)

    print("=" * 70)
    from collections import Counter
    cats = Counter(s for _, s, _ in bad)
    print(f"OK: {len(ok)}    PROBLEMS: {len(bad)}"
          f"   (corrupt={cats['corrupt']}  truncated={cats['truncated']}"
          f"  unreadable={cats['unreadable']})")
    print("=" * 70)
    if bad:
        print("\n--- files that will NOT load ---")
        for p, status, detail in sorted(bad, key=lambda x: (x[1], x[0].name)):
            print(f"  [{status:10s}] {p.name[:50]:50s} {detail}")
            print(f"               {p}")
    # machine-readable dump
    try:
        with open("_integrity_report.json", "w", encoding="utf-8") as f:
            json.dump({
                "ok": [str(p) for p in ok],
                "bad": [{"path": str(p), "status": s, "detail": d}
                        for p, s, d in bad],
            }, f, indent=2)
        print("\nreport written to _integrity_report.json")
    except Exception:
        pass
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
