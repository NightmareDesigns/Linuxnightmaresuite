# API Engine — Linux Port (Core Runner / Bridge)

Ported 2026-11-02 from admin `nightmare_bridge.py` (Core Runner, localhost:1345)
into this Linux clone. Admin original is NOT changed; this file is
`nightmare_bridge.py` in this repo, with Linux patches marked LINUX PORT.

## Start

```sh
chmod +x start_api.sh
export NTMRE_ADMIN_TOKEN="your-long-random-token-here"
./start_api.sh
# optional: ./start_api.sh --port 1345 --model models/my-model.gguf
```

## Endpoints

- `GET /health` — alive, NO auth. Returns status alive, port, model_loaded bool.
- `GET /status` — auth required. Model, engine, tok/s, memory.
- `GET /models` — auth required. Lists `*.ntmre` + `*.gguf` in model folders.
- `POST /load` — auth required. Body `{"path": "..."}` or `{"name": "..."}`.
  GGUF goes via RouterEngine (GGUF / Llama / Lynx, from the Nov 1 GGUF+Lynx
  fix in admin Bridge). `.ntmre` tries the Windows DLL first, fails on Linux,
  then falls through to Router, then Gemma4 GPU engine — that fall-through
  is already in load_bridge_engine(), not a Linux invention.
- `POST /chat`, `/generate`, `/v1/chat/completions`, `/v1/completions`,
  `/v1/embeddings`, `/rpc`, `/upload`, `/voice`, `/memory/*` — auth required.
- `POST /tool` — **DISABLED by default**. See gate below.
- `POST /fluid` — **501 in this clone**. Fluid runtime is sale-excluded
  and NOT shipped here, so no token transfer / mesh can start. The fluid
  handler code is still in the file (for matching with admin), but it
  returns 501 before it can do anything, because FLUID_AVAILABLE is False
  without fluid_runtime/.
- `POST /router/*` — auth required, same as admin.

## Auth — fail closed, no default token

Admin original had a hardcoded token in source, and an empty token set
meant "open". BOTH are removed in this Linux port:

- Token comes ONLY from `NTMRE_ADMIN_TOKEN` env or a local `.admin_token`
  file. That file is gitignored — NEVER commit it, NEVER paste it in chat.
- If no token is configured, every endpoint except `/health` returns 403.
- All other endpoints require header `X-Admin-Token: <your token>`.

Example:

```sh
curl http://127.0.0.1:1345/health
curl -H "X-Admin-Token: $NTMRE_ADMIN_TOKEN" http://127.0.0.1:1345/status
curl -H "X-Admin-Token: $NTMRE_ADMIN_TOKEN" http://127.0.0.1:1345/models
```

## /tool gate — off by default

This answers the unanswered /tool gate question conservatively:

```sh
export NTMRE_BRIDGE_TOOLS=1
export NTMRE_ADMIN_TOKEN="your-long-random-token-here"
./start_api.sh
```

Without BOTH, `POST /tool` returns 403 tools disabled. `python_exec` and
`process_kill` are gated the same way. Do NOT enable /tool on a machine
you share, and do NOT bind the bridge to anything but loopback — see below.

## Bind — loopback only

`start_bridge()` in this port refuses any host other than 127.0.0.1,
localhost, or ::1. Do NOT change that to 0.0.0.0 in this clone without
understanding that /tool, when enabled, can run code and kill processes
on your PC.

## Linux fallbacks — what changed, what did not

- Memory: `/proc/meminfo` on Linux, Windows GlobalMemoryStatusEx kept guarded.
- Processes: `ps` on Linux, `tasklist` on Windows.
- Connections: `ss` (then netstat) on Linux, `netstat -ano` on Windows.
- TTS: `espeak-ng` on Linux if installed, Windows SAPI via PowerShell on Windows.
- STT / listen: Windows SAPI only in admin code; on Linux it returns a
  clear error, it does not crash, and it does not fake dictation.
- Model folders: Windows-only `C:\tmp` entry removed, replaced with
  `~/.local/share/nightmare/models`. Home-based folders are kept.
- DLL engines: Windows .dll files cannot load on Linux, and are NOT in
  this clone. DllEngine / Gemma4Engine will fail and fall through to
  RouterEngine, as designed. A Linux .so build does not exist in the
  copies on file.

## What was NOT tested — honest, same as admin

- No real model, no GPU, no weights on the Alder VM, so /load /chat
  /generate with a real model were NOT tested here. /health /status
  /models with no model loaded were tested in admin Bridge on Nov 1
  (port 1346/1347, health 200, model_loaded false, models 0) — that is
  NOT Matrix / GPU proof, and NOT Lynx proof. Real test needs a Matrix
  (or Linux + NVIDIA + CUDA) machine with a real GGUF / .ntmre model,
  e.g. Lynx 8B GGUF, and a token set as above.
- Voice on Linux was NOT tested with a real espeak install here.
- /tool, when enabled, was NOT tested with a real tool registry call
  here — it is gated off by default for that reason.

## Logs

- `logs/api-YYYY-MM-DD_HHMM.log` — start_api.sh, bridge stdout/stderr.
- If the bridge does not start, send Alder that log + `./check_suite.sh`
  output. If /load fails, send the exact model path, the exact error
  from /load (it includes a trace), and the model file size / format.
