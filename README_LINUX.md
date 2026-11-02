# Nightmare Suite — Linux Clone

Linux clone of the Nightmare AI Suite Sale copy, for troubleshooting
with matching copies on both sides.

- Source: `sealed-sale-2026-10-30/` Sale copy, byte-for-byte for all
  Sale .py files. No Sale code was changed in this clone.
- Sale exclusions preserved: NO Tia, NO Conflux, NO fluid token system,
  NO Bridge, NO licence keys, NO admin-only tools.
- Windows .dll files removed — they cannot load on Linux, and the
  NVIDIA NVRTC .dll alone is 44 MB. GPU / fast-matmul / train paths
  that need a .dll will fall back to CPU / numpy on Linux, or fail
  that tool honestly. A Linux .so build does not exist in the copies
  on file.
- New files in this clone only: `linux_compat.py`, `run_suite.sh`,
  `check_suite.sh`, `requirements.txt`, `.gitignore`,
  `README_LINUX.md`, `SALE_AUDIT_2026-11-02.md`.

## Start

```sh
chmod +x run_suite.sh check_suite.sh
./check_suite.sh   # first, if anything does not start
./run_suite.sh     # GUI
```

Requirements: Python 3.11+ and tkinter.
Debian/Ubuntu: `sudo apt install python3 python3-tk`
Fedora: `sudo dnf install python3 python3-tkinter`

Optional, for full features (reported by check, not required to start):
see `requirements.txt` — numpy, torch, Pillow, psutil, requests, etc.
Missing optional packages stop only the tool that needs them.

## Logs, for troubleshooting from the start

- `logs/suite-YYYY-MM-DD_HHMM.log` — run_suite.sh, stdout/stderr
- `logs/check-YYYY-MM-DD_HHMM.log` — check_suite.sh
- `CRASH_LOG.txt` — written by the Suite itself on a crash, in this folder

If it does not work first shot, send Alder: the exact step, the model
path if any, the full error, and those logs. See SALE_AUDIT_2026-11-02.md
for what was already found: GUI starts on Linux, DAW winsound is shimmed,
locked download ships no DLLs, model loading was NOT tested (no weights
on the Alder VM).

## Matching copies

Windows Sale copy and this Linux clone share the same .py code, so:
- Same failure in both = Sale code, not platform.
- Failure only on one = platform difference, named by the logs from both sides.
- Do NOT edit Sale .py files in only one copy and then compare — that
  breaks the matching, and the diff will lie about what failed.
