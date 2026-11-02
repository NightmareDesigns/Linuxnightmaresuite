#!/bin/sh
# Nightmare Suite — Linux clone — API engine (Core Runner / Bridge) start script
#   chmod +x start_api.sh
#   ./start_api.sh
#   ./start_api.sh --port 1345
#   ./start_api.sh --model models/my-model.gguf   # optional, load at startup
#
# What it runs: nightmare_bridge.py, localhost ONLY, default port 1345.
#   GET  /health        - alive, NO auth (alive check only)
#   GET  /status        - requires admin token
#   GET  /models        - requires admin token, lists *.ntmre + *.gguf in model folders
#   POST /chat /generate /load /unload /v1/* - requires admin token
#   POST /tool          - DISABLED by default, see below
#   POST /fluid         - 501 in this clone, Fluid is sale-excluded and not shipped
#
# Auth (fail closed, no default token in source):
#   Export a token before starting, OR put it in a local .admin_token file
#   (gitignored, NEVER commit it):
#     export NTMRE_ADMIN_TOKEN="your-long-random-token-here"
#     ./start_api.sh
#   Then call with:  curl -H "X-Admin-Token: $NTMRE_ADMIN_TOKEN" http://127.0.0.1:1345/status
#   If no token is configured, every endpoint except /health returns 403.
#
# /tool gate (off by default):
#   export NTMRE_BRIDGE_TOOLS=1
#   export NTMRE_ADMIN_TOKEN="your-long-random-token-here"
#   ./start_api.sh
#   Without BOTH, POST /tool returns 403 tools disabled. python_exec and
#   process_kill are also gated the same way, even if called directly.
#
# Bind: loopback only (127.0.0.1). Non-loopback hosts are refused by
# nightmare_bridge.py start_bridge(). Do NOT edit that to 0.0.0.0 in
# this clone without understanding that /tool can run code on your PC.
#
# Logs: logs/api-YYYY-MM-DD_HHMM.log
# No model? /health still works, /status shows model_loaded false, /models
# shows 0 if no model folder exists. That is NOT a crash — load a model
# with POST /load, or start with --model, on a machine with a real model.
set -e
cd "$(dirname "$0")"
mkdir -p logs
LOG="logs/api-$(date +%Y-%m-%d_%H%M).log"
echo "Nightmare Bridge Linux starting - $(date)" | tee "$LOG"
if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 not found." | tee -a "$LOG"
  exit 1
fi
python3 --version 2>&1 | tee -a "$LOG"
if [ -z "$NTMRE_ADMIN_TOKEN" ] && [ ! -f .admin_token ]; then
  echo "WARNING: No NTMRE_ADMIN_TOKEN and no .admin_token file - /health will work, everything else will return 403 (fail closed, by design)." | tee -a "$LOG"
fi
if [ "$NTMRE_BRIDGE_TOOLS" != "1" ]; then
  echo "Tools: /tool DISABLED by default (set NTMRE_BRIDGE_TOOLS=1 + token to enable)." | tee -a "$LOG"
fi
echo "Listening on loopback only, default port 1345 - log: $LOG" | tee -a "$LOG"
set +e
python3 nightmare_bridge.py "$@" 2>&1 | tee -a "$LOG"
CODE=${PIPESTATUS[0]:-$?}
echo "Bridge exited with code $CODE - $(date)" | tee -a "$LOG"
exit "$CODE"
