"""
nightmare_tools_13.py — Nightmare Suite tool registry
=============================================
All tool wrappers are pure Python + stdlib/ctypes. No numpy.
Actionable skills are wrapped so the assistant can invoke them.
Includes full PC control + FL Studio integration.
"""

import base64
import ctypes
import json
import os
import re
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
FL_DIR = SUITE_DIR / "fl_studio_scripts"
SKILL_CATALOG = SUITE_DIR.parent / "biofolder" / "skill_master_db" / "catalog.json"


def _run(cmd, timeout=60, shell=False):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           shell=shell, errors="replace")
        out = (r.stdout or "") + (("\nSTDERR: " + r.stderr) if r.stderr else "")
        return out[:8192] or "(no output)"
    except subprocess.TimeoutExpired:
        return "timed out"
    except Exception as e:
        return f"error: {e}"


def _skill_path(name):
    # All derived from the running user's profile. A hardcoded
    # C:\Users\<someone>\... path finds nothing on any other machine, which is
    # a silent failure: the skill just is not there, with no indication that
    # the path was ever wrong.
    home = Path.home()
    env_home = Path(os.environ.get("USERPROFILE") or home)
    roots = [
        SUITE_DIR.parent / "biofolder" / "skill_master_db" / "skills",
        env_home / ".lmstudio" / "skills",
        home / ".lmstudio" / "skills",
        home / ".agents" / "skills",
        home / ".claude" / "skills",
    ]
    seen = set()
    for root in roots:
        try:
            key = str(root.resolve())
        except Exception:
            continue
        if key in seen:
            continue
        seen.add(key)
        p = root / name / "SKILL.md"
        if p.is_file():
            return p
    return None


# ============================================================
# Skill lookup
# ============================================================

def tool_skill(name, query=""):
    p = _skill_path(str(name))
    if not p:
        return f"skill '{name}' not found"
    text = p.read_text(encoding="utf-8", errors="replace")
    if len(text) > 4000:
        return text[:4000] + f"\n...[truncated, {len(text)} total chars]"
    return text


# ============================================================
# PC control
# ============================================================

def tool_open_app(name_or_path):
    """Open an application by name or full path."""
    p = Path(str(name_or_path))
    if p.is_file():
        subprocess.Popen([str(p)])
        return f"launched {p}"
    # Try common locations
    candidates = [
        Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")),
        Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)")),
        # os.environ.get does NOT expand %USERNAME% -- the old fallback was
        # the literal string C:\Users\%USERNAME%\AppData\Local, which is a
        # directory that never exists. Fall back to the real profile instead.
        Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local")),
    ]
    for root in candidates:
        if not root.exists():
            continue
        for exe in root.rglob("*.exe"):
            if name_or_path.lower() in exe.stem.lower():
                subprocess.Popen([str(exe)])
                return f"launched {exe}"
    return f"could not find app matching '{name_or_path}'"


def tool_screenshot(output="screenshot.png"):
    """Take a screenshot using PowerShell / .NET (no external packages)."""
    try:
        out_path = Path(output).resolve()
        ps = (f"Add-Type -AssemblyName System.Windows.Forms; "
              f"[System.Windows.Forms.SendKeys]::SendWait('%{{PRTSC}}'); "
              f"$img=[Windows.Forms.Clipboard]::GetImage(); "
              f"if($img){{$img.Save('{out_path}'); Write-Host 'saved {out_path}'}}else{{Write-Host 'clipboard empty'}}")
        r = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=30)
        return (r.stdout or r.stderr or "done").strip()
    except Exception as e:
        return f"screenshot error: {e}"


def tool_type_text(text):
    """Type text into the active window via SendKeys."""
    try:
        safe = str(text).replace("'", "'")
        ps = f"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('{safe}')"
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], timeout=10, check=True)
        return f"typed {len(text)} chars"
    except Exception as e:
        return f"type error: {e}"


def tool_press_key(key):
    """Press a key or key combo in the active window."""
    try:
        key = str(key).upper()
        ps = f"Add-Type -AssemblyName System.Windows.Forms; [System.Windows.Forms.SendKeys]::SendWait('{{{key}}}')"
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], timeout=10, check=True)
        return f"pressed {key}"
    except Exception as e:
        return f"key error: {e}"


def tool_mouse_click(x, y, button="left"):
    """Move mouse and click at screen coordinates."""
    try:
        x, y = int(x), int(y)
        flags = {"left": 0x0002 | 0x0004, "right": 0x0008 | 0x0010, "middle": 0x0020 | 0x0040}.get(button.lower(), 0x0002 | 0x0004)
        ctypes.windll.user32.SetCursorPos(x, y)
        ctypes.windll.user32.mouse_event(flags, 0, 0, 0, 0)
        return f"clicked {button} at {x},{y}"
    except Exception as e:
        return f"mouse error: {e}"


def tool_list_windows():
    """List visible window titles."""
    try:
        titles = []
        def enum(hwnd, ctx):
            if ctypes.windll.user32.IsWindowVisible(hwnd):
                buf = ctypes.create_unicode_buffer(256)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
                if buf.value:
                    titles.append(buf.value)
            return True
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
        ctypes.windll.user32.EnumWindows(EnumWindowsProc(enum), 0)
        return "\n".join(titles[:50]) or "no visible windows"
    except Exception as e:
        return f"window list error: {e}"


def tool_activate_window(title_substring):
    """Activate a window by title substring."""
    try:
        target = str(title_substring).lower()
        found = None
        def enum(hwnd, ctx):
            nonlocal found
            if ctypes.windll.user32.IsWindowVisible(hwnd):
                buf = ctypes.create_unicode_buffer(256)
                ctypes.windll.user32.GetWindowTextW(hwnd, buf, 256)
                if target in buf.value.lower():
                    found = hwnd
            return True
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_int, ctypes.POINTER(ctypes.c_int))
        ctypes.windll.user32.EnumWindows(EnumWindowsProc(enum), 0)
        if found:
            ctypes.windll.user32.SetForegroundWindow(found)
            return f"activated window {found}"
        return f"no window matching '{title_substring}'"
    except Exception as e:
        return f"activate error: {e}"


def tool_run_command(command, timeout=60):
    """Run a shell command and return output."""
    return _run(command, timeout=timeout, shell=True)


def tool_list_processes():
    return _run(["tasklist", "/fo", "csv", "/nh"], timeout=15)


def tool_process_kill(name="", pid=None):
    try:
        if pid:
            cmd = ["taskkill", "//F", "//PID", str(pid)]
        else:
            cmd = ["taskkill", "//F", "//IM", str(name)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return (r.stdout or r.stderr or "done").strip()[:2000]
    except Exception as e:
        return f"process_kill error: {e}"


def tool_system_info(component="all"):
    info = {}
    comp = str(component).lower()
    if comp in ("all", "gpu"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"], text=True, timeout=10, stderr=subprocess.DEVNULL)
            gpus = []
            for line in out.strip().splitlines():
                p = [x.strip() for x in line.split(",")]
                if len(p) >= 6:
                    gpus.append({"index": int(p[0]), "name": p[1], "temp_c": int(p[2]),
                                 "util_pct": int(p[3]), "vram_used_mb": int(p[4]), "vram_total_mb": int(p[5])})
            info["gpus"] = gpus
        except Exception:
            info["gpus"] = []
    if comp in ("all", "ram", "cpu"):
        class MEMSTAT(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        m = MEMSTAT()
        m.dwLength = ctypes.sizeof(MEMSTAT)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        info["memory"] = {"ram_total_gb": round(m.ullTotalPhys / 2**30, 1),
                          "ram_free_gb": round(m.ullAvailPhys / 2**30, 1),
                          "ram_load_pct": m.dwMemoryLoad}
    return json.dumps(info)


# ============================================================
# FL Studio integration
# ============================================================

def _fl_client():
    client = FL_DIR / "fl_bridge_client_13.py"
    if not client.is_file():
        return None
    return str(client)


def tool_fl_studio(action, **kw):
    """Control FL Studio via NightmareLink13."""
    client = _fl_client()
    if not client:
        return "FL bridge client not found"
    action = str(action).lower()

    if action == "open":
        exe = _run([sys.executable, client, "detect"], timeout=10)
        # Try to launch FL
        for root in [Path(os.environ.get("PROGRAMFILES", r"C:\Program Files")),
                     Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))]:
            il = root / "Image-Line"
            if il.is_dir():
                for d in sorted(il.iterdir()):
                    if d.name.startswith("FL Studio"):
                        for exe_name in ("FL64.exe", "FL.exe"):
                            if (d / exe_name).is_file():
                                subprocess.Popen([str(d / exe_name)])
                                return f"launched {d / exe_name}"
        return "FL Studio not found"

    if action == "install_bridge":
        return _run([sys.executable, client, "install"], timeout=30)

    if action == "status":
        return _run([sys.executable, client, "status"], timeout=15)

    if action in ("play", "stop", "pause", "record"):
        return _run([sys.executable, client, "send", action], timeout=10)

    if action == "send":
        op = kw.get("op", "ping")
        args = kw.get("args", {})
        arglist = [f"{k}={v}" for k, v in args.items()]
        return _run([sys.executable, client, "send", op] + arglist, timeout=10)

    if action == "generate_track":
        style = kw.get("style", "house")
        key = kw.get("key", "A")
        scale = kw.get("scale", "minor")
        bpm = kw.get("bpm", 140)
        bars = kw.get("bars", 4)
        name = kw.get("name", "nightmare_track")
        out_dir = kw.get("out_dir", str(SUITE_DIR / "generated_tracks"))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        edm = FL_DIR / "edm_gen_13.py"
        if not edm.is_file():
            return "EDM generator not found"
        cmd = [sys.executable, str(edm), "--style", str(style), "--key", str(key),
               "--scale", str(scale), "--bpm", str(bpm), "--bars", str(bars),
               "--name", str(name), "--out", str(out_dir)]
        return _run(cmd, timeout=30)

    return f"FL Studio action '{action}' not wired"


# ============================================================
# Local paths (folders + files)
# ============================================================
#
# WHY THIS EXISTS
#
# There was no way to open a local folder. The only "open" tool she knew was
# browser_open, which shells out to `agent-browser open` -- so asking her to
# open her models folder launched a web browser. She was doing the only thing
# the tool table allowed, and it was the wrong thing.
#
# tool_open_path dispatches on what the path actually is, which is what
# "open" means to a person: a folder goes to Explorer, a file goes to
# whatever normally handles it, a URL goes to the browser.

_URL_PREFIXES = ("http://", "https://", "file://", "ftp://")


def _looks_like_url(value: str) -> bool:
    v = str(value).strip().lower()
    return v.startswith(_URL_PREFIXES)


def tool_open_path(path, mode="auto"):
    """
    Open a local folder or file the way a person would.

    mode: auto (decide), explorer (always Explorer), app (default handler),
          browser (force the browser).
    """
    raw = str(path or "").strip().strip('"').strip("'")
    if not raw:
        return "Error: no path given."

    mode = str(mode or "auto").strip().lower()

    # A URL asked for explicitly, or clearly one, goes to the browser.
    if mode == "browser" or _looks_like_url(raw):
        return tool_browser_open(raw)

    # Expand the things people actually type: ~, and the env vars that show
    # up in model output as %USERPROFILE% and friends.
    expanded = os.path.expandvars(os.path.expanduser(raw))
    p = Path(expanded)

    if mode == "explorer" or (mode == "auto" and p.is_dir()):
        if not p.is_dir():
            return f"Not a folder: {p}"
        try:
            os.startfile(str(p))          # noqa: S606 - this is the point
            return f"Opened folder: {p}"
        except Exception as e:
            # Fall back to the explicit binary if startfile is unavailable.
            r = _run(["explorer.exe", str(p)], timeout=20)
            return r if r else f"Opened folder: {p} ({e})"

    if not p.exists():
        # A path that does not exist is usually a typo or a missing ~. Say so
        # instead of opening a browser to a file:// URL and showing nothing.
        return f"Path does not exist: {p}"

    try:
        os.startfile(str(p))              # noqa: S606
        return f"Opened: {p}"
    except Exception as e:
        return f"Could not open {p}: {e}"


# ============================================================
# Browser / web
# ============================================================

def tool_browser_open(url):
    """
    Open a URL in the browser.

    Refuses local paths instead of launching a browser at them. A file:// URL
    gives a blank page and looks like the tool is broken; a folder path
    opened as a URL is worse, because it looks like it worked.
    """
    raw = str(url or "").strip()
    if not raw:
        return "Error: no url given."

    expanded = os.path.expandvars(os.path.expanduser(raw))
    if not _looks_like_url(expanded):
        # Not a URL. If it is a real path, open it properly instead.
        if Path(expanded).exists():
            return tool_open_path(expanded)
        return ("'%s' is not a web address. If it is a folder or file, use "
                "open_path instead -- this tool only opens URLs."
                % raw)
    return _run(["agent-browser", "open", str(expanded)], timeout=30)


def tool_browser_screenshot(url, output="screenshot.png"):
    return _run(["agent-browser", "screenshot", str(url), "--output", str(output)], timeout=60)


def tool_browser_scrape(url):
    return _run(["agent-browser", "scrape", str(url)], timeout=60)


# ============================================================
# Document / spreadsheet / PDF
# ============================================================

def tool_docx_create(path, text=""):
    try:
        from docx import Document
        d = Document()
        d.add_heading("Nightmare Document", 0)
        d.add_paragraph(str(text))
        d.save(str(path))
        return f"created {path}"
    except Exception as e:
        return f"docx error: {e}"


def tool_docx_read(path):
    try:
        from docx import Document
        d = Document(str(path))
        return "\n".join([p.text for p in d.paragraphs])[:4000]
    except Exception as e:
        return f"docx read error: {e}"


def tool_pptx_create(path, title="Nightmare Deck", slides=None):
    try:
        from pptx import Presentation
        prs = Presentation()
        prs.slides.add_slide(prs.slide_layouts[0]).shapes.title.text = str(title)
        if slides:
            for s in slides:
                slide = prs.slides.add_slide(prs.slide_layouts[1])
                slide.shapes.title.text = str(s)
        prs.save(str(path))
        return f"created {path}"
    except Exception as e:
        return f"pptx error: {e}"


def tool_pptx_read(path):
    try:
        from pptx import Presentation
        prs = Presentation(str(path))
        out = []
        for i, slide in enumerate(prs.slides, 1):
            texts = []
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    texts.append(shape.text)
            out.append(f"Slide {i}: {' | '.join(texts)}")
        return "\n".join(out)[:4000]
    except Exception as e:
        return f"pptx read error: {e}"


def tool_xlsx_create(path, data=None):
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        if data:
            for row in data:
                ws.append([str(c) for c in row])
        wb.save(str(path))
        return f"created {path}"
    except Exception as e:
        return f"xlsx error: {e}"


def tool_xlsx_read(path):
    try:
        from openpyxl import load_workbook
        wb = load_workbook(str(path), read_only=True, data_only=True)
        ws = wb.active
        rows = []
        for row in ws.iter_rows(values_only=True):
            rows.append("\t".join(str(c) if c is not None else "" for c in row))
        return "\n".join(rows)[:4000]
    except Exception as e:
        return f"xlsx read error: {e}"


def tool_pdf_read(path, pages=None):
    try:
        from pypdf import PdfReader
        r = PdfReader(str(path))
        out = []
        rng = range(len(r.pages)) if pages is None else [int(p)-1 for p in str(pages).split(",")]
        for i in rng:
            if 0 <= i < len(r.pages):
                out.append(r.pages[i].extract_text() or "")
        return "\n".join(out)[:4000]
    except Exception as e:
        return f"pdf read error: {e}"


def tool_pdf_merge(paths, output="merged.pdf"):
    try:
        from pypdf import PdfMerger
        m = PdfMerger()
        for p in paths:
            m.append(str(p))
        m.write(str(output))
        m.close()
        return f"merged into {output}"
    except Exception as e:
        return f"pdf merge error: {e}"


# ============================================================
# Voice / upload / utility
# ============================================================

def tool_voice(action="speak", text=""):
    if action == "speak":
        try:
            safe = str(text).replace("'", "'")
            ps = f"Add-Type -AssemblyName System.Speech; $s=New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Speak('''{safe}''');"
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], timeout=60, check=True)
            return "spoken"
        except Exception as e:
            return f"tts error: {e}"
    return "voice action not supported"


def tool_python_exec(code):
    return _run([sys.executable, "-c", str(code)], timeout=120)


def tool_calculator(expression):
    try:
        import ast
        node = ast.parse(str(expression), mode="eval").body
        def ev(n):
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return n.value
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)):
                l, r = ev(n.left), ev(n.right)
                return {ast.Add: l+r, ast.Sub: l-r, ast.Mult: l*r, ast.Div: l/r,
                        ast.Pow: l**r, ast.Mod: l%r, ast.FloorDiv: l//r}[type(n.op)]
            if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
                v = ev(n.operand)
                return -v if isinstance(n.op, ast.USub) else v
            raise ValueError("arithmetic only")
        return ev(node)
    except Exception as e:
        return f"calc error: {e}"


def tool_datetime():
    from datetime import datetime
    return datetime.now().strftime("%I:%M %p, %A %B %d %Y").lstrip("0")


def tool_file_read(path):
    p = Path(str(path))
    if not p.is_file():
        return f"file not found: {path}"
    try:
        return p.read_text(encoding="utf-8", errors="replace")[:4000]
    except Exception as e:
        return f"read error: {e}"


def tool_file_list(directory):
    p = Path(str(directory))
    if not p.is_dir():
        return f"dir not found: {directory}"
    return "\n".join((("D " if e.is_dir() else "F ") + e.name) for e in sorted(p.iterdir())[:200])


def tool_upload(filename, data_b64, kind="auto"):
    folder = SUITE_DIR / "uploads"
    folder.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r'[^\w\-. ]', '_', Path(filename).name)
    path = folder / safe_name
    try:
        path.write_bytes(base64.b64decode(data_b64))
        preview = ""
        if kind in ("text", "auto"):
            try:
                preview = path.read_text(encoding="utf-8", errors="replace")[:2000]
            except Exception:
                pass
        return {"saved": str(path), "preview": preview[:500]}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Tool registry
# ============================================================

ALL_TOOLS = {
    # PC control
    "open_app": lambda a, k: tool_open_app(a[0] if a else k.get("name", "")),
    "screenshot": lambda a, k: tool_screenshot(k.get("output", a[0] if a else "screenshot.png")),
    "type_text": lambda a, k: tool_type_text(a[0] if a else k.get("text", "")),
    "press_key": lambda a, k: tool_press_key(a[0] if a else k.get("key", "")),
    "mouse_click": lambda a, k: tool_mouse_click(a[0] if a else k.get("x", 0), a[1] if len(a) > 1 else k.get("y", 0), k.get("button", "left")),
    "list_windows": lambda a, k: tool_list_windows(),
    "activate_window": lambda a, k: tool_activate_window(a[0] if a else k.get("title", "")),
    "run_command": lambda a, k: tool_run_command(a[0] if a else k.get("command", ""), k.get("timeout", 60)),
    "list_processes": lambda a, k: tool_list_processes(),
    "process_kill": lambda a, k: tool_process_kill(k.get("name", a[0] if a else ""), k.get("pid")),
    "system_info": lambda a, k: tool_system_info(k.get("component", a[0] if a else "all")),
    "file_read": lambda a, k: tool_file_read(k.get("path", a[0] if a else "")),
    "file_list": lambda a, k: tool_file_list(k.get("directory", a[0] if a else ".")),
    # Open a local folder or file. Registered under several names because
    # "open my models folder" arrives as any of them depending on how she
    # phrased it, and a missing name is what sent her to browser_open.
    "open_path": lambda a, k: tool_open_path(
        k.get("path", a[0] if a else ""), k.get("mode", a[1] if len(a) > 1 else "auto")),
    "open_folder": lambda a, k: tool_open_path(
        k.get("path", a[0] if a else ""), "explorer"),
    "explorer": lambda a, k: tool_open_path(
        k.get("path", a[0] if a else ""), "explorer"),
    # FL Studio
    "fl_studio": lambda a, k: tool_fl_studio(
        a[0] if a else k.get("action", ""),
        **{key: val for key, val in k.items() if key != "action"}
    ),
    "make_track": lambda a, k: tool_fl_studio("generate_track", **{key: val for key, val in k.items() if key != "action"}),
    # Browser
    "browser_open": lambda a, k: tool_browser_open(a[0] if a else k.get("url", "")),
    "browser_screenshot": lambda a, k: tool_browser_screenshot(a[0] if a else k.get("url", ""), k.get("output", "screenshot.png")),
    "browser_scrape": lambda a, k: tool_browser_scrape(a[0] if a else k.get("url", "")),
    # Documents
    "docx_create": lambda a, k: tool_docx_create(k.get("path", a[0] if a else "doc.docx"), k.get("text", "")),
    "docx_read": lambda a, k: tool_docx_read(k.get("path", a[0] if a else "")),
    "pptx_create": lambda a, k: tool_pptx_create(k.get("path", a[0] if a else "deck.pptx"), k.get("title", "Nightmare Deck"), k.get("slides")),
    "pptx_read": lambda a, k: tool_pptx_read(k.get("path", a[0] if a else "")),
    "xlsx_create": lambda a, k: tool_xlsx_create(k.get("path", a[0] if a else "sheet.xlsx"), k.get("data")),
    "xlsx_read": lambda a, k: tool_xlsx_read(k.get("path", a[0] if a else "")),
    "pdf_read": lambda a, k: tool_pdf_read(k.get("path", a[0] if a else ""), k.get("pages")),
    "pdf_merge": lambda a, k: tool_pdf_merge(k.get("paths", a), k.get("output", "merged.pdf")),
    # Voice / utility
    "voice": lambda a, k: tool_voice(k.get("action", "speak"), a[0] if a else k.get("text", "")),
    "python_exec": lambda a, k: tool_python_exec(a[0] if a else k.get("code", "")),
    "calculator": lambda a, k: tool_calculator(a[0] if a else k.get("expression", "0")),
    "datetime": lambda a, k: tool_datetime(),
    "upload": lambda a, k: tool_upload(k.get("filename", "upload"), k.get("data", ""), k.get("kind", "auto")),
    "skill": lambda a, k: tool_skill(a[0] if a else k.get("name", ""), k.get("query", "")),
}

# Add a tool entry for every skill in the catalog
SKILL_NAMES = [
    "agent-browser", "alphafold-database-fetch-and-analyze", "alphagenome-single-variant-analysis",
    "android-cli", "bigquery-ai-ml", "bigquery-bigframes", "bigquery-data-transfer-service",
    "bigquery-graph", "bigquery-sql", "browser-automation", "building-data-apps", "canvas",
    "chembl-database", "claude-api", "clinical-trials-database", "clinvar-database",
    "data-autocleaning", "dataform-bigquery", "dbsnp-database", "dbt-bigquery", "discord",
    "discovering-gcp-data-assets", "doc-coauthoring", "docx", "embl-ebi-ols",
    "encode-ccres-database", "enforcing-resource-attribution", "ensembl-database",
    "federate-lakehouse-catalog", "find-skills", "fl-studio", "foldseek-structural-search",
    "frontend-design", "gcloud-auth-verification", "gcp-composer-troubleshooting",
    "gcp-data-pipelines", "gcp-dataflow", "gcp-managed-airflow-dag-authoring",
    "gcp-managed-airflow-migrations", "gcp-managed-airflow-recommendations",
    "gcp-pipeline-orchestration", "gcp-pipeline-resource-provisioning", "gcp-spark",
    "gcs-security-assessment", "gepeto", "gnomad-database", "google-cloud-storage-basics",
    "gtex-database", "human-protein-atlas-database", "imsg", "interpro-database",
    "jaspar-database", "literature-search-arxiv", "literature-search-biorxiv",
    "literature-search-europepmc", "literature-search-openalex", "managing-python-dependencies",
    "mcp-builder", "ml-best-practices", "ncbi-sequence-fetch", "notebook-guidance",
    "obsidian-vault-maintainer", "openfda-database", "opentargets-database", "pdb-database",
    "pdf", "pinokio", "pptx", "prose", "protein-sequence-msa",
    "protein-sequence-similarity-search", "pubchem-database", "pubmed-database", "pymol",
    "quickgo-database", "reactome-database", "scienceskillscommon", "skill-creator",
    "skill-repair", "string-database", "ucsc-conservation-and-tfbs", "unibind-database",
    "uniprot-database", "uv", "web-design-guidelines", "webapp-testing", "wiki-maintainer",
    "workflow-skill-creator", "xlsx", "accidental-data-loss-prevention"
]

for skill_name in SKILL_NAMES:
    key = f"skill_{skill_name.replace('-', '_')}"
    if key not in ALL_TOOLS:
        ALL_TOOLS[key] = lambda a, k, name=skill_name: tool_skill(name, " ".join(str(x) for x in a) if a else k.get("query", ""))


# --- AI Workbench (260+ tools, reached through one gated tool) -----------
#
# The workbench exposes a single `execute` tool rather than 260 names,
# because two hundred and sixty tool descriptions in a system prompt is
# context spent on every turn describing capabilities the model may not
# currently use.
#
# The handler is nightmare_workbench.WorkbenchLink.execute, which checks
# the tool gate INSIDE itself before spawning anything. Registering it here
# is therefore safe: this file cannot become a path around the gate, because
# the arm check lives in the callable and not at any call site.
#
# Wrapped in try/except because the workbench is optional. A buyer without
# it installed gets the other tools unchanged.
try:
    from nightmare_workbench import get_bridge as _get_wb_bridge, WB_TOOL_NAME
    _WB_BRIDGE = _get_wb_bridge()
    ALL_TOOLS[WB_TOOL_NAME] = _WB_BRIDGE.handler()
except Exception as _wb_err:      # pragma: no cover - optional dependency
    _WB_BRIDGE = None
    WB_TOOL_NAME = "wb.execute"
    # No handler registered, so the name resolves to "unknown tool" rather
    # than to something that silently does nothing.


def run_tool(name, args, kwargs):
    fn = ALL_TOOLS.get(name.lower())
    if fn is None:
        return {"error": f"unknown tool: {name}"}
    try:
        return {"result": fn(args, kwargs)}
    except Exception as e:
        return {"error": str(e), "trace": traceback.format_exc()}
