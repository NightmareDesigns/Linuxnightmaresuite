#!/bin/sh
# Nightmare Suite — Linux clone — start script
# Double-click equivalent on Linux: ./run_suite.sh  (chmod +x first, see below)
# Usage:
#   chmod +x run_suite.sh check_suite.sh
#   ./run_suite.sh
# What it does:
#   1. cd to this folder (so relative model / data paths work the same as Windows)
#   2. checks python3 exists, logs the check
#   3. imports linux_compat.py FIRST (winsound + os.startfile shims, no Sale code changed)
#   4. runs nightmare_suite_unified.py, logging stdout/stderr to logs/suite-YYYY-MM-DD_HHMM.log
#   5. if the GUI exits with an error, prints the log path and waits for Enter
# Requirements: python3 + tkinter (python3-tk on Debian/Ubuntu, python3-tkinter on Fedora).
#   Optional, for full features, NOT required to start: numpy, torch, Pillow, psutil, requests
#   See requirements.txt and check_suite.sh — missing optional packages are
#   reported, they do NOT stop startup, except where a tool actually needs them.
set -e
cd "$(dirname "$0")"
mkdir -p logs
LOG="logs/suite-$(date +%Y-%m-%d_%H%M).log"
echo "Nightmare Suite Linux starting - $(date)" | tee "$LOG"
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found. Install Python 3.11+ and python3-tk, then run again." | tee -a "$LOG"
  exit 1
fi
python3 --version 2>&1 | tee -a "$LOG"
echo "Folder: $(pwd)" | tee -a "$LOG"
echo "Starting GUI - log: $LOG" | tee -a "$LOG"
set +e
python3 -c "import linux_compat, runpy; runpy.run_path('nightmare_suite_unified.py', run_name='__main__')" 2>&1 | tee -a "$LOG"
CODE=${PIPESTATUS[0]:-$?}
echo "Suite exited with code $CODE - $(date)" | tee -a "$LOG"
if [ "$CODE" -ne 0 ]; then
  echo ""
  echo "Suite exited with an error - see log: $LOG"
  echo "Also run: ./check_suite.sh  and send Alder that log + logs/check-*.log"
  read -r _dummy </dev/tty || true
fi
exit "$CODE"
