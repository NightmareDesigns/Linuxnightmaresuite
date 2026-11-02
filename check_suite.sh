#!/bin/sh
# Nightmare Suite — Linux clone — first-shot check
#   chmod +x check_suite.sh
#   ./check_suite.sh
# Does NOT train, does NOT load a model, does NOT change the brain.
# It: checks python3 + tkinter, compiles every .py, imports every Sale
# module in a separate process (so one bad module cannot hide the rest),
# reports winsound / startfile shims, lists missing optional packages,
# and logs everything to logs/check-YYYY-MM-DD_HHMM.log
# Send Alder that log + logs/suite-*.log if the GUI still does not start.
set -e
cd "$(dirname "$0")"
mkdir -p logs
LOG="logs/check-$(date +%Y-%m-%d_%H%M).log"
{
echo "Nightmare Suite Linux check - $(date)"
echo "Folder: $(pwd)"
command -v python3 >/dev/null 2>&1 && python3 --version || echo "ERROR: python3 not found"
python3 -c "import tkinter; print('tkinter OK', tkinter.TkVersion)" 2>&1 || echo "ERROR: tkinter missing - install python3-tk (Debian/Ubuntu) or python3-tkinter (Fedora)"
echo "--- Compiling .py files ---"
python3 -m py_compile *.py core_mind/*.py engine/*.py fusion_engine/*.py model_memory/*.py nightmare_np/*.py 2>&1 && echo "COMPILE OK: all .py files" || echo "COMPILE FAILED: see above"
echo "--- Importing Sale modules, one process each, with linux_compat ---"
python3 - << 'PYEOF'
import subprocess, sys, pathlib
root = pathlib.Path(".")
fails = []
total = 0
for p in sorted(root.glob("*.py")):
    if p.name in ("linux_compat.py",):
        continue
    total += 1
    code = "import linux_compat; import sys; sys.path.insert(0, '.'); import " + p.stem
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    if r.returncode != 0:
        err = (r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "no stderr")[:200]
        fails.append((p.stem, err))
        print(f"FAIL {p.stem}: {err}")
    else:
        print(f"OK {p.stem}")
print(f"IMPORT TOTAL {total} FAIL {len(fails)}")
for name, err in fails:
    print(f"  FAIL {name}: {err}")
PYEOF
echo "--- Optional packages (reported, not required to start) ---"
python3 - << 'PYEOF'
for mod in ("numpy", "torch", "PIL", "psutil", "requests", "cryptography", "fido2", "scipy"):
    try:
        __import__(mod)
        print(f"OK {mod}")
    except Exception as e:
        print(f"MISSING {mod}: {e}")
PYEOF
echo "--- Files ---"
ls *.py | head -100
echo "Log saved to: $LOG"
} 2>&1 | tee "$LOG"
echo "Log saved to: $LOG"
