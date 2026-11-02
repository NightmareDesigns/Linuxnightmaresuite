# Sale Copy Audit — 2026-11-02, Linux VM, Alder

Source audited: `sealed-sale-2026-10-30/` (plaintext Sale copy, 124 .py files
in root + subfolders, 269 files total, 58 MB) and
`locked-sale-2026-11-01-allsealed/` (the website download, 116 .sealed,
60 PTX, 7 .py, 4.9 MB, entry `run_nightmare.py`).

## What actually happens on Linux — tested, not guessed

1. **Both Sale copies START the GUI on Linux under a display.**
   - Sealed-sale plaintext: `python3 nightmare_suite_unified.py` under xvfb,
     ran past startup, GUI stayed open until the 30s test timeout killed it.
   - Locked all-sealed: `python3 run_nightmare.py` under xvfb, same — GUI
     stayed open until the 20s test timeout killed it.
   - So "sale copy failing" is NOT a startup crash on Linux. Startup is
     not the failure. The failures below are what breaks AFTER startup,
     or what breaks on a machine without a display / without the right
     packages / without Windows.

2. **Without a display, both fail at Tk, honestly.**
   - Locked, no DISPLAY: `tkinter.TclError: no display name and no
     $DISPLAY environment variable`, exit 1, after the startup banner.
   - That is normal for a Tk GUI, not a Sale bug — but it means a
     headless Linux server / SSH session with no X will NOT run the Suite
     without xvfb or a desktop. Use `./run_suite.sh` on a desktop, or
     `xvfb-run -a ./run_suite.sh` headless.

3. **Startup notices that are NOT fatal, but look like failures:**
   - `Process scan notice: No module named 'psutil'` — caught, startup
     continues. psutil is optional at startup, used lazily elsewhere.
   - `Working set clear failed: module 'ctypes' has no attribute 'WinDLL'`
     — caught, startup continues. This is the Windows RAM-purge code in
     `clear_ram_on_startup()` running unguarded on Linux, then being
     caught. It does NOT kill startup, but it proves the startup path
     was never run on Linux before this audit.

4. **All Sale .py files compile: 124 / 124, 0 compile fails.**

5. **Module imports, one process each, Linux, no shim: 3 fails out of 101 root .py files.**
   - `edm_panel` FAIL — `ModuleNotFoundError: No module named 'winsound'`.
     winsound is Windows-only stdlib. This is a REAL Linux failure:
     the DAW / EDM panel cannot even be imported on Linux, so that
     panel cannot open. Fixed in this Linux clone by `linux_compat.py`
     (winsound stub, no Sale file changed).
   - `export_90m_fixed` FAIL on import — it runs export code at import
     time and prints `ERROR: No checkpoint found!`. This is a script,
     not a panel, and it should NOT be imported as a module. Do not
     import it; run it only as a script with a checkpoint present.
   - `setup_gr_hash` FAIL on import — it is a setuptools build script,
     `error: no commands supplied` when imported. Again, a script, not
     a runtime module. Build with `python3 setup_gr_hash.py build_ext --inplace`
     if you need the native gr_hash extension, do not import it.

   Every other Sale module, including `nightmare_suite_unified` itself,
   imports OK on Linux.

## What WILL fail on Linux after startup — known, not fixed by a shim

6. **Windows DLLs do not load on Linux, period.**
   - Sealed-sale ships 13 .dll files, including `nvrtc64_120_0.dll`
     (44 MB, NVIDIA NVRTC runtime) and `nightmare_engine.dll`,
     `nightmare_matmul.dll`, `nightmare_train.dll`, etc.
   - Locked all-sealed (the website download) ships **ZERO .dll files**.
     That is a separate, Windows-side failure: the download copy has
     no native engines at all, so any code path that loads
     `nightmare_engine.dll` / `nightmare_matmul.dll` by bare name will
     raise `OSError: Could not locate …` and fall back to CPU, or fail
     that tool, depending on the caller. The sealed-sale plaintext copy
     DOES have the DLLs; the locked download does NOT. Do not assume
     the download and the sealed-sale copy are the same — they are not.
   - Linux clone ships NO .dll files on purpose (they cannot load on
     Linux, and nvrtc64 alone is 44 MB). On Linux, GPU / fast-matmul /
     train paths that need a .dll will fall back to CPU / numpy, or
     fail that tool honestly. A Linux .so build of those engines does
     NOT exist in the copies on file — building one is a separate job,
     not a shim.

7. **Windows-only APIs, unguarded in places, guarded in others.**
   - `ctypes.WinDLL` / `ctypes.windll` in: main (startup, caught),
     `nightmare_hardware.py`, `nightmare_stream.py`, `system_monitor.py`,
     `system_planner.py`, `nightmare_tools_13.py` (mouse / window control,
     memory status), `fusion_engine/mem_budget.py`, `ntmre_router.py`
     (drive-letter scan), `model_integrity_check.py` (guarded by
     `sys.platform != "win32"` — correct).
   - `os.startfile` in: `edm_panel.py`, `nightmare_tools_13.py`.
     On Linux, `os.startfile` does not exist. Some callers catch it,
     some fall back to `explorer.exe` (which does not exist on Linux).
     Fixed in this clone by the `linux_compat.py` startfile shim
     (xdg-open). Window / mouse control in `nightmare_tools_13.py`
     has NO Linux equivalent in this clone — those tools will fail
     honestly on Linux, they are Windows desktop automation, not Suite
     core.
   - Hardcoded Windows paths throughout: `C:\Program Files\NVIDIA…`,
     `C:\Users\User\Desktop\…`, `D:\…`, `G:\ntmre_backup`, etc., in
     GPU discovery, CUDA search, model sidecars, and example model
     paths. On Linux these paths simply find nothing, so GPU discovery
     falls back to CPU, and model search must be given a Linux folder
     explicitly. Do NOT edit those paths in the clone to guess a Linux
     path — use the Browse / model folder picker in the GUI, or set
     `CUDA_PATH` / model folder yourself.

8. **Third-party packages are NOT bundled in the Sale copy.**
   - Sale code imports, lazily or at panel open: numpy, torch, PIL,
     psutil, requests, cryptography / Crypto, fido2, scipy, docx,
     openpyxl, pptx, pypdf, gradio, cupy, and others. None of these
     are in the Sale zip. On a fresh Linux install, only stdlib +
     tkinter are guaranteed. `check_suite.sh` reports which are
     missing; missing packages do NOT stop startup, they stop the
     specific tool that needs them, with an ImportError at that point.
   - On the Alder Linux VM, `check_suite.sh` found: numpy OK, PIL OK,
     scipy OK, requests OK; torch MISSING, psutil MISSING, cryptography
     MISSING, fido2 MISSING. Startup still worked — because startup
     does not need them. Do not claim "Suite needs X" from startup
     alone; check the tool you actually click.

9. **No model, no GPU, no weights on the Alder VM — model loading was NOT tested.**
   - No .ntmre, no .gguf, no real Golden weights are on the Alder VM.
     The Sale copy's model load path, GGUF parser, .ntmre router, and
     dual-GPU split were NOT exercised in this audit, because there
     is nothing to load. If "sale copy failing" means "model fails to
     load on the Matrix", that failure is still unreproduced here —
     send Alder the exact model path, the exact error / CRASH_LOG.txt,
     and the log from `./check_suite.sh`, and we can reproduce it in
     this Linux clone with the same model folder, because the clone
     and the Sale copy share the same .py code.
   - PTX kernels ARE in both Sale copies (60 PTX, sm75 / sm86, Gemma4
     E4B / E2B / 12B variants). PTX alone does not run a model — it
     needs an NVIDIA GPU + CUDA runtime + a model, none of which are
     on the Alder VM.

## What this Linux clone changes — and what it does NOT change

- CHANGED: nothing in any Sale .py file. Sale code is byte-for-byte
  the sealed-sale copy, so a diff between Windows Sale and this clone
  is clean, and a failure in the clone is a failure in the Sale code,
  not a clone-only bug.
- ADDED: `linux_compat.py` (winsound stub + os.startfile shim only),
  `run_suite.sh`, `check_suite.sh`, `requirements.txt`, `.gitignore`,
  `README_LINUX.md`, and this audit file.
- REMOVED: Windows .dll files (cannot load on Linux, 50 MB saved),
  `__pycache__`, runtime folders created by test runs (`chat_history`).
- NOT included, same as Sale: Tia, Conflux, fluid token system, Bridge,
  licence keys, admin-only tools, master tokens. Sale exclusions are
  preserved. If a file in this clone mentions Tia in a comment or a
  model name (chat_templates, gguf_parser), that is a name check in
  Sale code, not Tia herself — Tia is not in this clone.

## How to troubleshoot with matching copies

1. Windows side: run the Sale copy, note the exact step that fails.
2. Linux side: run this clone, `./check_suite.sh` first, then
   `./run_suite.sh`, do the SAME step, with the SAME model folder
   if the failure is model-related.
3. Compare: if it fails the same way in both, it is Sale code, not
   Windows vs Linux. If it fails only on one, the log + CRASH_LOG.txt
   from both sides names the platform difference.
4. Send Alder: the step, the model path (if any), the full error,
   `logs/check-*.log`, `logs/suite-*.log`, and `CRASH_LOG.txt` if the
   Suite wrote one. "It didn't work" alone cannot be reproduced.
