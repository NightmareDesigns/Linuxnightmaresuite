"""nightmare_np.failsafe — snapshot / check / restore your known-good code.

The real failsafe against AI agents (or stray `pip install` runs) overwriting
or breaking your hand-written suite. Pure stdlib, no numpy.

Quick start (run from the suite folder, when everything works):

    # 1) snapshot the known-good state
    python -m nightmare_np.failsafe snapshot --targets inference.py gemma4_forward.py fast_matmul.py nightmare_np --out failsafe_state

    # 2) later, see if anything changed / broke
    python -m nightmare_np.failsafe check --state failsafe_state

    # 3) restore the known-good files
    python -m nightmare_np.failsafe restore --state failsafe_state

    # 4) watch (auto-check; add --auto-restore to heal automatically)
    python -m nightmare_np.failsafe watch --state failsafe_state --interval 60 --auto-restore
"""
import argparse
import hashlib
import json
import os
import shutil
import time

MANIFEST = "manifest.json"
BACKUP = "backup"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _expand_targets(targets):
    files = []
    for t in targets:
        if os.path.isdir(t):
            for root, _, fs in os.walk(t):
                for f in fs:
                    files.append(os.path.join(root, f))
        elif os.path.isfile(t):
            files.append(t)
    return sorted(set(os.path.abspath(f) for f in files))


def _store_path(f):
    """Path stored in the manifest: relative to cwd when possible, else absolute."""
    try:
        return os.path.relpath(f)
    except ValueError:
        return os.path.abspath(f)


def _backup_rel(rel):
    """Safe relative path for storing a copy under the backup dir (handles other drives)."""
    p = rel.replace(":", "_").lstrip("\\/").replace("\\", "/")
    return p


def snapshot(targets, state_dir):
    """Record hashes of the target files + keep copies in state_dir/backup."""
    os.makedirs(state_dir, exist_ok=True)
    backup_dir = os.path.join(state_dir, BACKUP)
    files = _expand_targets(targets)
    manifest = {}
    for f in files:
        rel = _store_path(f)
        manifest[rel] = {"hash": sha256_file(f), "size": os.path.getsize(f)}
        dst = os.path.join(backup_dir, _backup_rel(rel))
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        shutil.copy2(f, dst)
    with open(os.path.join(state_dir, MANIFEST), "w") as fh:
        json.dump(manifest, fh, indent=2)
    return len(manifest)


def _load(state_dir):
    with open(os.path.join(state_dir, MANIFEST)) as fh:
        return json.load(fh)


def check(state_dir):
    """Return (ok, changed, missing) relative paths."""
    manifest = _load(state_dir)
    ok, changed, missing = [], [], []
    for rel, meta in manifest.items():
        if not os.path.exists(rel):
            missing.append(rel)
        elif sha256_file(rel) != meta["hash"]:
            changed.append(rel)
        else:
            ok.append(rel)
    return ok, changed, missing


def restore(state_dir):
    """Copy the known-good copies back over any file that changed/missing."""
    manifest = _load(state_dir)
    backup_dir = os.path.join(state_dir, BACKUP)
    restored = []
    for rel, meta in manifest.items():
        if not os.path.exists(rel) or sha256_file(rel) != meta["hash"]:
            src = os.path.join(backup_dir, _backup_rel(rel))
            if os.path.exists(src):
                os.makedirs(os.path.dirname(rel) or ".", exist_ok=True)
                shutil.copy2(src, rel)
                restored.append(rel)
    return restored


def _print_report(ok, changed, missing):
    print(f"OK      : {len(ok)} files unchanged")
    print(f"CHANGED : {len(changed)} files")
    for c in changed:
        print(f"   ! {c}")
    print(f"MISSING : {len(missing)} files")
    for m in missing:
        print(f"   ? {m}")


def main(argv=None):
    p = argparse.ArgumentParser(description="nightmare_np failsafe")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot")
    s.add_argument("--targets", nargs="+", required=True)
    s.add_argument("--out", default="failsafe_state")

    c = sub.add_parser("check")
    c.add_argument("--state", default="failsafe_state")

    r = sub.add_parser("restore")
    r.add_argument("--state", default="failsafe_state")

    w = sub.add_parser("watch")
    w.add_argument("--state", default="failsafe_state")
    w.add_argument("--interval", type=int, default=60)
    w.add_argument("--auto-restore", action="store_true")

    args = p.parse_args(argv)

    if args.cmd == "snapshot":
        n = snapshot(args.targets, args.out)
        print(f"[failsafe] snapshot saved {n} files to {args.out}")
    elif args.cmd == "check":
        ok, changed, missing = check(args.state)
        _print_report(ok, changed, missing)
        if changed or missing:
            print("\n[failsafe] PROBLEMS FOUND. Run: "
                  f"python -m nightmare_np.failsafe restore --state {args.state}")
    elif args.cmd == "restore":
        restored = restore(args.state)
        print(f"[failsafe] restored {len(restored)} file(s)")
        for r in restored:
            print(f"   + {r}")
    elif args.cmd == "watch":
        print(f"[failsafe] watching {args.state} every {args.interval}s "
              f"(auto-restore={'on' if args.auto_restore else 'off'})")
        while True:
            ok, changed, missing = check(args.state)
            if changed or missing:
                print(f"[failsafe] {time.strftime('%H:%M:%S')} DETECTED: "
                      f"{len(changed)} changed, {len(missing)} missing")
                if args.auto_restore:
                    restored = restore(args.state)
                    print(f"[failsafe] auto-restored {len(restored)} file(s)")
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
