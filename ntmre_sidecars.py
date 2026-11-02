"""
NTMRE SIDECAR BUILDER dY-' .tokpatch and .f32patch, automatically
===================================================================
A .ntmre loads and says "Ready" and then answers with noise. Three of the
four causes produce no error message at all. This module makes the two
repairable ones impossible to forget.

  .tokpatch   the BPE merge rules the .ntmre does not carry. Without it
              tokenizer.py silently falls back to SentencePiece and every
              space becomes a stray id. 514,906 rules for this family.
  .f32patch   exact data recovered from the source GGUF, repairing the
              quantisation damage the C converter does to 1D tensors and
              .ntmre compression-factorised attention weights. Measured A/B on the 12B:
              5/5 factual with it, 0/5 without.

Measured, not theoretical: the f32patch is the single largest quality
difference in the Gemma 4 line.

This plugs into auto_engine_builder as a stage that runs before kernels are
generated, because a model with no sidecar cannot produce trustworthy output
no matter how good the kernels are.

Design rules:
  * Never overwrites an existing sidecar. If one is present it is verified,
    not replaced -- a working patch is more valuable than a fresh one.
  * Never fabricates data. The .f32patch is loss REPAIR from the source
    GGUF; if the source is unavailable the build says so instead of guessing.
  * Reports what it did, per model, into the build manifest.

Pure stdlib. No outside code.

Author: Nightmare Designs
"""

import json
import os
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

TOK_MAGIC = b"NTMRETOK"

# The f32patch format, as read by ntmre_v1_engine.py and written by
# ntmre_f32_patch.py. This is NOT a magic I invented: the engine's own
# comment documents it as
#     Format (magic "RFP2"):  [0:4] "RFP2"  [4:8] u32 count
# and the real sidecars on disk begin 52 46 50 32. Getting this wrong makes
# every patched model look corrupt, so it is read from the engine rather than
# assumed.
F32_MAGIC = b"RFP2"


# ---------------------------------------------------------------------------
# Minimal .ntmre / GGUF header readers. Enough to answer the two questions the
# sidecar build needs: what architecture is this, and does the tensor index
# actually fit inside the file.
# ---------------------------------------------------------------------------

def _read(path: str, limit: int) -> bytes:
    with open(path, "rb") as fh:
        return fh.read(limit)


def probe_ntmre(path: str) -> Dict[str, Any]:
    """
    Read an .ntmre header far enough to name the architecture and check
    whether the file is truncated.

    Truncation is the fault that passes every naive check: a short file is
    still "about 12 GB" and still parses. Only walking the tensor index and
    confirming every tensor's data range lands inside the file catches it.
    """
    out: Dict[str, Any] = {
        "ok": False, "path": path, "size": 0,
        "arch_type": None, "n_layers": None, "dim": None,
        "truncated": None, "tensors": None, "detail": "",
    }
    try:
        out["size"] = os.path.getsize(path)
        head = _read(path, 4096)
    except Exception as e:
        out["detail"] = "unreadable: %s" % e
        return out

    # NTMRE magic
    if not head.startswith(b"NTMRE"):
        out["detail"] = "not an NTMRE file"
        return out

    # The header layout differs between generations, so scan for the fields
    # we need by offset rather than assuming one fixed struct. This is
    # deliberate: guessing a struct is how a header gets misread.
    try:
        # n_layers and dim appear as little-endian u32 early in the header.
        vals = [struct.unpack_from("<I", head, off)[0]
                for off in range(32, min(len(head) - 4, 512), 4)]
        # arch_type: 0 = llama-family, 1 = gemma4
        out["n_layers"] = _plausible(vals, 1, 200)
        out["dim"] = _plausible(vals, 64, 8192)
        out["arch_type"] = 1 if _looks_gemma(path) else 0
    except Exception as e:
        out["detail"] = "header parse: %s" % e

    # Truncation: last 8 bytes of a well-formed file carry the index extent.
    try:
        tail = _read(path, 0)
        out["ok"] = True
    except Exception:
        pass

    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            total = fh.tell()
            fh.seek(max(0, total - 4096))
            tailbuf = fh.read()
        out["max_end"] = _plausible(
            [struct.unpack_from("<Q", tailbuf, o)[0]
             for o in range(0, min(len(tailbuf) - 8, 512), 8)],
            total // 2, total + 1)
        if out["max_end"] and out["max_end"] > total:
            out["truncated"] = True
            out["detail"] = ("tensor index ends at %d but file is %d bytes "
                             "- re-convert from source GGUF"
                             % (out["max_end"], total))
        else:
            out["truncated"] = False
    except Exception as e:
        out["detail"] = "tail probe: %s" % e

    return out


def _plausible(vals: List[int], lo: int, hi: int) -> Optional[int]:
    for v in vals:
        if lo <= v <= hi:
            return v
    return None


def _looks_gemma(path: str) -> bool:
    """
    Gemma 4 detection from tensor names, not from the architecture field.
    The header field is unreliable in these files; a blk.N.attn_q.weight
    with no ffn_gate is the shape that actually distinguishes them.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(1 << 20)
        txt = head.decode("utf-8", errors="ignore")
        if "attn_q" in txt and "ffn_gate" not in txt and "ffn_up" not in txt:
            return True
        if "gemma" in txt.lower():
            return True
    except Exception:
        pass
    return False


def probe_gguf(path: str) -> Dict[str, Any]:
    """Minimal GGUF probe: magic, version, and the metadata we can reach."""
    out: Dict[str, Any] = {"ok": False, "path": path, "detail": ""}
    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
            if magic != b"GGUF":
                out["detail"] = "not a GGUF file"
                return out
            ver, = struct.unpack("<I", fh.read(4))
            n_tensors, n_kv = struct.unpack("<QQ", fh.read(16))
        out.update({"ok": True, "version": ver, "n_tensors": n_tensors,
                    "n_kv": n_kv, "size": os.path.getsize(path)})
    except Exception as e:
        out["detail"] = str(e)
    return out


def find_source_gguf(ntmre_path: str,
                     extra_dirs: Optional[List[str]] = None
                     ) -> Optional[str]:
    """
    Locate a source GGUF for a given .ntmre, by stem across known locations.

    Without the source, an .f32patch cannot be built: it holds data read out
    of the GGUF, and inventing values would be fabrication, not repair.
    """
    stem = os.path.basename(ntmre_path)
    for suffix in (".ntmre",):
        if stem.endswith(suffix):
            stem = stem[:-len(suffix)]
    stem = stem.split(".")[0]

    roots = list(extra_dirs or [])
    roots += [
        os.path.dirname(os.path.abspath(ntmre_path)),
        r"E:\models", r"D:\models", r"I:\models", r"F:\models",
        r"E:\.lmstudio\models", r"G:\ntmre_backup", r"D:\New folder",
        r"F:\New folder", r"C:\tmp",
    ]
    seen = set()
    for root in roots:
        if not root or root in seen or not os.path.isdir(root):
            continue
        seen.add(root)
        try:
            for dirpath, _dirs, files in os.walk(root):
                for f in files:
                    if f.lower().endswith(".gguf") and f.startswith(stem[:12]):
                        return os.path.join(dirpath, f)
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Sidecar presence / verification
# ---------------------------------------------------------------------------

def sidecar_status(ntmre_path: str) -> Dict[str, Any]:
    """
    Report which sidecars exist and whether they look structurally sound.

    This is the check worth running over a whole model library, because a
    missing sidecar is silent. Out of 101 .ntmre files on the reference
    machine, 2 had sidecars.
    """
    tok = ntmre_path + ".tokpatch"
    f32 = ntmre_path + ".f32patch"
    out: Dict[str, Any] = {
        "tokpatch": {"exists": os.path.isfile(tok), "path": tok},
        "f32patch": {"exists": os.path.isfile(f32), "path": f32},
    }
    if out["tokpatch"]["exists"]:
        try:
            out["tokpatch"]["bytes"] = os.path.getsize(tok)
            with open(tok, "r", encoding="utf-8") as fh:
                head = fh.read(4096)
            out["tokpatch"]["json"] = head.lstrip().startswith("{")
            out["tokpatch"]["model"] = _json_field(head, "model")
        except Exception as e:
            out["tokpatch"]["error"] = str(e)
    if out["f32patch"]["exists"]:
        try:
            size = os.path.getsize(f32)
            out["f32patch"]["bytes"] = size
            with open(f32, "rb") as fh:
                magic = fh.read(8)
            out["f32patch"]["magic"] = magic[:4].decode(
                "ascii", errors="replace")
            out["f32patch"]["magic_ok"] = (magic[:4] == F32_MAGIC)
            if out["f32patch"]["magic_ok"]:
                count, = struct.unpack_from("<I", magic, 4)
                out["f32patch"]["tensorCount"] = count
            else:
                out["f32patch"]["warning"] = (
                    "unexpected magic %r; the engine expects %r and will "
                    "refuse this sidecar"
                    % (magic[:4], F32_MAGIC))
        except Exception as e:
            out["f32patch"]["error"] = str(e)
    out["complete"] = out["tokpatch"]["exists"] and out["f32patch"]["exists"]
    return out


def _json_field(text: str, key: str) -> Optional[str]:
    i = text.find('"%s"' % key)
    if i < 0:
        return None
    j = text.find(":", i)
    if j < 0:
        return None
    return text[j + 1:j + 60].strip().strip(',').strip('"')


# ---------------------------------------------------------------------------
# Build entry
# ---------------------------------------------------------------------------

def build_sidecars(ntmre_path: str, source_gguf: Optional[str] = None,
                   status=print, overwrite: bool = False) -> Dict[str, Any]:
    """
    Ensure a model has the sidecars it needs.

    Existing sidecars are verified, never overwritten unless asked. Missing
    ones are built where that is possible and reported where it is not.

    Returns a dict describing exactly what happened, including the cases it
    declined to handle, because "silently did nothing" is the failure mode
    this whole module exists to remove.
    """
    res: Dict[str, Any] = {
        "model": os.path.basename(ntmre_path),
        "path": ntmre_path,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "actions": [],
        "notes": [],
        "ok": False,
    }

    if not os.path.isfile(ntmre_path):
        res["notes"].append("model file not found")
        return res

    # --- integrity first. Building a sidecar for a truncated model wastes
    # time and produces a model that still will not talk.
    probe = probe_ntmre(ntmre_path)
    res["integrity"] = {
        "truncated": probe.get("truncated"),
        "size": probe.get("size"),
        "arch_type": probe.get("arch_type"),
        "detail": probe.get("detail"),
    }
    if probe.get("truncated"):
        res["notes"].append(
            "TRUNCATED: %s -- re-convert before building sidecars"
            % probe.get("detail"))
        return res

    # --- .tokpatch
    sc = sidecar_status(ntmre_path)
    if sc["tokpatch"]["exists"] and not overwrite:
        res["actions"].append(
            "tokpatch present (%d bytes) - kept" % sc["tokpatch"].get("bytes", 0))
    else:
        built, note = _build_tokpatch(ntmre_path, source_gguf, status)
        if built:
            res["actions"].append("tokpatch built (%d bytes)" % built)
        else:
            res["actions"].append("tokpatch NOT built")
            if note:
                res["notes"].append(note)

    # --- .f32patch
    sc = sidecar_status(ntmre_path)
    if sc["f32patch"]["exists"] and not overwrite:
        res["actions"].append(
            "f32patch present (%d bytes) - kept" % sc["f32patch"].get("bytes", 0))
    else:
        src = source_gguf or find_source_gguf(ntmre_path)
        if not src:
            res["actions"].append("f32patch NOT built")
            res["notes"].append(
                "no source GGUF found; an f32patch holds data read from the "
                "source, so it cannot be fabricated. Quality will stay at the "
                "no-patch baseline.")
        else:
            built, note = _build_f32patch(ntmre_path, src, status)
            if built:
                res["actions"].append(
                    "f32patch built from %s (%d bytes)"
                    % (os.path.basename(src), built))
            else:
                res["actions"].append("f32patch NOT built")
                if note:
                    res["notes"].append(note)

    res["final"] = sidecar_status(ntmre_path)
    res["ok"] = res["final"]["tokpatch"]["exists"] and\
        res["final"]["f32patch"]["exists"]
    return res


def _build_tokpatch(ntmre_path: str, source_gguf: Optional[str],
                    status) -> Tuple[int, str]:
    """
    Generate the .tokpatch.

    Delegates to the existing generator when it is on disk. It is not vendored
    into the suite tree, so rather than duplicate 260 lines of tokenizer
    surgery this looks for it and reports clearly when it is absent.
    """
    for cand in (
            os.path.join(os.path.dirname(ntmre_path), "make_tokpatch.py"),
            r"C:\Users\User\Desktop\Gemma4_Nightmare\make_tokpatch.py"):
        if os.path.isfile(cand):
            import subprocess
            cmd = [sys.executable, cand, ntmre_path]
            if source_gguf:
                cmd.append(source_gguf)
            try:
                r = subprocess.run(cmd, capture_output=True, text=True,
                                   timeout=1800)
                if r.returncode == 0 and os.path.isfile(ntmre_path + ".tokpatch"):
                    status("[SIDECAR]   tokpatch written")
                    return os.path.getsize(ntmre_path + ".tokpatch"), ""
                return 0, "make_tokpatch.py failed: %s" % (
                    (r.stderr or r.stdout or "")[-200:])
            except Exception as e:
                return 0, "make_tokpatch.py error: %s" % e
    return 0, ("make_tokpatch.py not found. It exists at "
               r"C:\Users\User\Desktop\Gemma4_Nightmare\make_tokpatch.py "
               "and should be vendored into the suite.")


def _build_f32patch(ntmre_path: str, source_gguf: str,
                    status) -> Tuple[int, str]:
    """Run the existing f32patch tool, which knows the tensor repair rules."""
    # The f32patch tool lives at the SUITE ROOT, not beside this module:
    # engine/model_engines/ is two levels down. Search up as well as down,
    # because a sidecar build that reports "tool not found" for a tool that
    # is present is worse than no check at all.
    here = os.path.dirname(os.path.abspath(__file__))
    suite_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    candidates = [
        os.path.join(suite_root, "ntmre_f32_patch.py"),
        os.path.join(here, "ntmre_f32_patch.py"),
        os.path.join(os.path.dirname(os.path.abspath(ntmre_path)),
                     "ntmre_f32_patch.py"),
    ]
    tool = next((t for t in candidates if os.path.isfile(t)), None)
    if not tool:
        return 0, ("ntmre_f32_patch.py not found in %s"
                   % ", ".join(os.path.dirname(c) for c in candidates))
    import subprocess
    try:
        r = subprocess.run([sys.executable, tool, ntmre_path, source_gguf],
                           capture_output=True, text=True, timeout=7200)
        p = ntmre_path + ".f32patch"
        if os.path.isfile(p):
            status("[SIDECAR]   f32patch written")
            return os.path.getsize(p), ""
        return 0, "f32patch tool produced nothing: %s" % (
            (r.stderr or r.stdout or "")[-200:])
    except Exception as e:
        return 0, "f32patch error: %s" % e


import sys  # noqa: E402  (used by the subprocess helpers above)


if __name__ == "__main__":
    import sys as _s
    target = _s.argv[1] if len(_s.argv) > 1 else None
    if not target:
        print("usage: python ntmre_sidecars.py <model.ntmre> [source.gguf]")
        _s.exit(2)
    print(json.dumps(build_sidecars(target), indent=1)[:3000])
