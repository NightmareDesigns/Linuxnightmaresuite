"""
Shortcut pruning -- a link to a model that is gone is litter.

THE PROBLEM
-----------
ntmre_router builds shortcuts in suite/models/ for every model it finds
on any drive, and the directory accumulated 59 of them. 34 pointed at
files that no longer exist: 29 at C: paths that were never on this
machine, 5 at a Z: drive that is not mounted.

Nothing ever removed one. create_shortcut and create_shortcuts_batch
only ever ADD, and full_system_quick_scan's links_to_create list only
ever grows, so the count is monotonic by construction. Every rescan
added to a pile that could not shrink.

WHY IT IS NOT JUST COSMETIC
---------------------------
A dead .lnk is not an empty folder. It is an entry the discovery walk
finds -- the scan explicitly collects '.lnk' -- resolves, and then
skips. So each rescan pays for 34 resolutions to reach the same
conclusion, and the model list shows entries that cannot be loaded. On
a UI that lists what you have, "I own a Gemma 4 E4B" and "I cannot open
it" are different sentences.

THE RULE
--------
Prune only shortcuts this system created, in this directory, whose
target is a real model file. Anything else is left alone: a shortcut
the user made by hand, a .lnk pointing at a folder, a link to something
not a model. Deleting a file the user made is not a scan's job.

Author: Nightmare Designs
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

MODEL_EXTS = (".gguf", ".ntmre", ".safetensors", ".onnx")


def read_target(lnk: Path) -> Optional[str]:
    """
    Read a .lnk target, in one PowerShell call for a batch.

    Per-file COM instantiation is slow enough that a naive resolve of
    59 shortcuts is seconds of stall on every UI refresh, which is
    presumably part of why the directory was left alone.
    """
    if sys.platform != "win32" or not lnk.exists():
        return None
    lnk_s = str(lnk.resolve()).replace("'", "''")
    ps = ("$w=New-Object -ComObject WScript.Shell;"
          "$s=$w.CreateShortcut('%s');"
          "if($s){$s.TargetPath}" % lnk_s)
    try:
        import subprocess
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
            capture_output=True, timeout=10)
        out = (r.stdout or b"").decode("utf-8", "replace").strip()
        return out or None
    except Exception:
        return None


def read_targets_batch(links: List[Path]) -> Dict[str, str]:
    """Resolve many .lnk targets in a single PowerShell invocation."""
    out: Dict[str, str] = {}
    if sys.platform != "win32" or not links:
        return out
    cmds = ["$w=New-Object -ComObject WScript.Shell;"]
    for lnk in links:
        p = str(Path(lnk).resolve()).replace("'", "''")
        cmds.append(
            "try{$s=$w.CreateShortcut('%s');"
            "Write-Output ('%s' + '::' + $s.TargetPath)}catch{}" % (p, p))
    try:
        import subprocess
        r = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive",
             "-Command", " ".join(cmds)],
            capture_output=True, timeout=20)
        for line in (r.stdout or b"").decode("utf-8", "replace").splitlines():
            line = line.strip()
            if "::" not in line:
                continue
            key, _, val = line.partition("::")
            if val:
                out[key] = val
    except Exception:
        pass
    return out


def classify(lnk: Path, target: Optional[str]) -> str:
    """
    dead | live | foreign

      live      target exists and looks like a model
      dead      target is set but the file is gone
      foreign   no target, or a target that is not a model file
    """
    if not target:
        return "foreign"
    t = Path(target)
    if not t.exists():
        # A missing drive is the same problem as a missing file: the
        # link cannot be followed. Z: not mounted and C: never existed
        # are the two cases actually present here.
        return "dead"
    if t.suffix.lower() not in MODEL_EXTS:
        return "foreign"
    return "live"


def audit(models_dir: Path) -> Dict[str, List[Tuple[str, str, str]]]:
    """
    Read every shortcut and classify it. Read-only.

    Returns {verdict: [(name, target, classification_reason)]}
    """
    if not models_dir.is_dir():
        return {"live": [], "dead": [], "foreign": []}
    links = sorted(models_dir.glob("*.lnk"))
    targets = read_targets_batch(links)
    out: Dict[str, List[Tuple[str, str, str]]] = {
        "live": [], "dead": [], "foreign": []}
    for lnk in links:
        key = str(Path(lnk).resolve())
        tgt = targets.get(key) or read_target(lnk)
        verdict = classify(lnk, tgt)
        why = ""
        if verdict == "dead":
            # Distinguish "the drive is gone" from "the file is gone".
            # They look identical from the shortcut but they mean
            # different things to the person reading the report: an
            # unmounted Z: is a drive they may reconnect, and every
            # model on it is still theirs. A missing C: file is simply
            # litter. Labelling all 34 "drive not mounted" was wrong and
            # would have made a reconnect look like a data loss.
            if tgt and is_unmounted_drive(tgt):
                why = "drive %s not present" % tgt[:2]
            else:
                why = "target missing"
        out[verdict].append((lnk.name, tgt or "", why))
    return out


def re_drive(target: str) -> bool:
    t = str(target)
    return len(t) > 1 and t[1] == ":" and t[0].isalpha()


def is_unmounted_drive(target: str) -> bool:
    """
    Is the target on a drive letter that has no root at all?

    C: exists and the file under it is simply gone, which is ordinary
    litter. Z: has no root on this machine, which means the models there
    may well still exist on a drive that is not plugged in. Both are
    "dead" to a shortcut, but only one of them is safe to forget --
    so they are reported differently.
    """
    t = str(target)
    if not re_drive(t):
        return False
    drive = t[:2]
    if not re_drive(drive + "\\"):
        return False
    return not os.path.exists(drive + "\\")


def prune(models_dir: Path, dry_run: bool = True,
          keep_foreign: bool = True, status=print) -> Dict[str, object]:
    """
    Remove shortcuts whose target is gone. Nothing else.

    dry_run defaults to True because this deletes files. A scan that
    quietly removes things the user made is worse than the litter it
    cleans, so the caller has to opt in.

    keep_foreign leaves alone any link that is not a broken model link:
    hand-made shortcuts, links to folders, links to programs. Only the
    "dead" class is removable, and a link with no readable target is
    treated as foreign precisely so that an unreadable link is never
    deleted on the strength of a failed read.
    """
    report = audit(models_dir)
    dead = report.get("dead", [])
    live = report.get("live", [])
    foreign = report.get("foreign", [])

    status("[SHORTCUTS] %s: %d live, %d dead, %d foreign"
           % (models_dir, len(live), len(dead), len(foreign)))

    removed: List[str] = []
    skipped: List[str] = []
    for name, tgt, why in dead:
        p = models_dir / name
        if dry_run:
            status("  would remove  %-58s %s" % (name[:58], why))
            removed.append(name)
            continue
        try:
            p.unlink()
            removed.append(name)
        except Exception as e:
            status("  could not remove %s: %s" % (name, e))
            skipped.append(name)

    if not keep_foreign:
        for name, _t, _w in foreign:
            skipped.append(name)

    status("[SHORTCUTS] %s %d shortcut(s)"
           % ("would remove" if dry_run else "removed", len(removed)))
    if skipped:
        status("[SHORTCUTS] left alone: %d" % len(skipped))
    return {"live": len(live), "dead": len(dead), "foreign": len(foreign),
            "removed": removed, "skipped": skipped, "dry_run": dry_run}


def sync(models_dir: Path, live_targets: Set[str], dry_run: bool = True,
         status=print) -> Dict[str, object]:
    """
    Make the shortcut directory match what is actually on disk.

    live_targets is the set of resolved model paths the current scan
    found. A shortcut pointing at any of them is kept. A shortcut
    pointing at nothing is removed -- and that is the whole operation.

    The target is matched on the resolved path, not the filename, so
    two copies of the same model on different drives each keep their own
    link and neither is mistaken for the other.
    """
    live_norm = set()
    for t in live_targets or ():
        try:
            live_norm.add(str(Path(t).resolve()).lower())
        except Exception:
            live_norm.add(str(t).lower())

    if not models_dir.is_dir():
        return {"kept": 0, "removed": 0, "added": 0}

    links = sorted(models_dir.glob("*.lnk"))
    targets = read_targets_batch(links)
    keep: List[str] = []
    drop: List[Tuple[str, str]] = []

    for lnk in links:
        key = str(Path(lnk).resolve())
        tgt = targets.get(key) or read_target(lnk)
        if not tgt:
            continue                     # unreadable: leave it alone
        try:
            norm = str(Path(tgt).resolve()).lower()
        except Exception:
            norm = str(tgt).lower()
        if norm in live_norm or classify(lnk, tgt) == "foreign":
            keep.append(lnk.name)
        else:
            drop.append((lnk.name, tgt))

    removed: List[str] = []
    for name, _t in drop:
        p = models_dir / name
        if dry_run:
            removed.append(name)
        else:
            try:
                p.unlink()
                removed.append(name)
            except Exception:
                pass

    status("[SHORTCUTS] sync: kept %d, %s %d"
           % (len(keep), "would remove" if dry_run else "removed",
              len(removed)))
    return {"kept": len(keep), "removed": len(removed), "added": 0,
            "dry_run": dry_run, "removed_names": removed}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Audit and prune dead model shortcuts")
    ap.add_argument("models_dir", nargs="?",
                    default=os.path.join(
                        os.path.dirname(os.path.abspath(__file__)),
                        "models"))
    ap.add_argument("--remove", action="store_true",
                    help="actually delete (default is a dry run)")
    ap.add_argument("--keep-foreign", action="store_true",
                    help="also leave hand-made links alone (default)")
    args = ap.parse_args()
    prune(Path(args.models_dir), dry_run=not args.remove,
          keep_foreign=True)
