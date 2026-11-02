"""linux_compat — Linux shims for the Nightmare Suite sale copy, Linux clone only.

This file is NEW in the Linux clone. It does NOT change any Sale .py file.
Sale code is copied byte-for-byte from sealed-sale-2026-10-30, so a diff
between the Windows sale copy and this Linux clone shows ONLY this file
plus run_suite.sh / check_suite.sh / README_LINUX.md / SALE_AUDIT file.

Why a shim is needed, found by audit 2026-11-02:
- edm_panel.py imports winsound at top level. winsound exists ONLY on
  Windows. On Linux, `import edm_panel` fails with ModuleNotFoundError,
  so the DAW / EDM panel cannot even open, even though rendering itself
  is pure Python. Shim: install a winsound stub in sys.modules BEFORE
  any Sale module is imported. PlaySound tries aplay / paplay / ffplay
  if present, otherwise it logs and does nothing — it never crashes.
- os.startfile exists ONLY on Windows. Sale code calls it in
  edm_panel.py and nightmare_tools_13.py. On Linux, os.startfile does
  not exist, so those calls raise AttributeError (caught in some places,
  misleading in others). Shim: define os.startfile on Linux as xdg-open,
  falling back to `open` on macOS, and to a plain error message if neither
  exists. No Sale file is edited to do this.
- ctypes.WinDLL / ctypes.windll exist ONLY on Windows. Sale startup
  already catches the WinDLL working-set failure and logs it, so startup
  does NOT crash on Linux. This shim does NOT fake WinDLL, because faking
  a Windows kernel API would lie about RAM / process control. Calls that
  need WinDLL will still fail honestly, in the places that need them,
  and the audit file lists them. Do NOT add a fake WinDLL here.

Usage: run_suite.sh imports this file first, then runs
nightmare_suite_unified.py. Importing this file alone changes nothing
except the two shims above.
"""
import os
import subprocess
import sys
import types


def _install_winsound_stub():
    if "winsound" in sys.modules:
        return
    if sys.platform == "win32":
        return
    try:
        import winsound  # noqa: F401 — real winsound, Windows only, never here
        return
    except ImportError:
        pass

    stub = types.ModuleType("winsound")
    stub.SND_ASYNC = 0x0001
    stub.SND_FILENAME = 0x00020000
    stub.SND_PURGE = 0x0040
    stub.SND_LOOP = 0x0008
    stub.SND_NODEFAULT = 0x0002

    def _play(path, flags=0):
        # Stop: PlaySound(None, SND_PURGE) on Windows.
        if path is None:
            return
        # Try a Linux player, in order, silently. If none exists, do
        # nothing — playback is not worth crashing the DAW panel over.
        for cmd in (["aplay", str(path)], ["paplay", str(path)], ["ffplay", "-nodisp", "-autoexit", str(path)]):
            try:
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return
            except Exception:
                continue
        print(f"[linux_compat] No Linux audio player found (aplay/paplay/ffplay), skipping playback of {path}")

    stub.PlaySound = _play
    stub.MessageBeep = lambda *a, **k: None
    sys.modules["winsound"] = stub


def _install_startfile():
    if hasattr(os, "startfile"):
        return
    if sys.platform.startswith("linux"):
        def _startfile(path):
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return None
    elif sys.platform == "darwin":
        def _startfile(path):
            subprocess.Popen(["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return None
    else:
        def _startfile(path):
            raise OSError(f"os.startfile is not available on {sys.platform} for {path}")
    os.startfile = _startfile


def install():
    _install_winsound_stub()
    _install_startfile()


install()
