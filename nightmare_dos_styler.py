# -*- coding: utf-8 -*-
"""
nightmare_dos_styler.py — Universal Cyberpunk DOS Output Hook
============================================================
Intercepts all console output (stdout & stderr) and universally colorizes it:
- Bracketed tags ([Nightmare], [GGUF], [STREAM], [Tokenizer], [MEMORY], [Link], etc.)
  are rendered in Intense Blood Red (\033[38;2;255;0;0m).
- All message payloads and console text are rendered in Glowing Bright Phosphor
  Matrix Green (\033[1;38;2;0;255;70m).
- Bullet symbols (▶, ●, 💀, ⚡, ✓, ℹ, ⚠, ✗, ☠) are rendered in Vibrant Blood Red.
- Divider bars are styled in Blood Red.
- Existing ANSI escape sequences (borders, banners, animations) are preserved 100%.
- Full UTF-8 safety and Windows ANSI 24-bit TrueColor console mode enabled.

Zero external dependencies (pure Python standard library).
"""

import sys
import os
import re

BORDER_RED   = "\033[38;2;255;0;0m"       # Intense Blood Red for Tags & Borders
TAG_RED      = "\033[1;38;2;255;20;20m"   # Bold Vibrant Red for Symbols
TEXT_GREEN   = "\033[1;38;2;0;255;70m"    # Glowing Bright Phosphor Matrix Green
RESET        = "\033[0m"

TAG_PATTERN     = re.compile(r'^(\s*)(\[[^\]\r\n]+\])(.*)$')
BULLET_PATTERN  = re.compile(r'^(\s*)([▶●💀⚡✓ℹ⚠✗☠]+)(\s*)(.*)$')
DIVIDER_PATTERN = re.compile(r'^(\s*)[=\-_*#]{10,}(\s*)$')

def style_line(line: str) -> str:
    """Format a single line in Cyberpunk Blood Red & Bright Matrix Green."""
    # If already ANSI colorized, do not double-wrap
    if "\033[" in line:
        return line

    newline = ""
    if line.endswith("\r\n"):
        newline = "\r\n"
        line = line[:-2]
    elif line.endswith("\n"):
        newline = "\n"
        line = line[:-1]

    if not line.strip():
        return line + newline

    # Check for divider bars (=================, -----------------)
    if DIVIDER_PATTERN.match(line):
        return f"{BORDER_RED}{line}{RESET}{newline}"

    # Check for bullets and status symbols
    bm = BULLET_PATTERN.match(line)
    if bm:
        indent, sym, space, rest = bm.groups()
        tm = TAG_PATTERN.match(rest)
        if tm:
            t_indent, tag, payload = tm.groups()
            return f"{indent}{TAG_RED}{sym}{RESET}{space}{t_indent}{BORDER_RED}{tag}{RESET} {TEXT_GREEN}{payload.strip()}{RESET}{newline}"
        return f"{indent}{TAG_RED}{sym}{RESET}{space}{TEXT_GREEN}{rest}{RESET}{newline}"

    # Check for bracketed tags at line start: [GGUF], [STREAM], [Tokenizer], etc.
    tm = TAG_PATTERN.match(line)
    if tm:
        indent, tag, rest = tm.groups()
        return f"{indent}{BORDER_RED}{tag}{RESET} {TEXT_GREEN}{rest.strip()}{RESET}{newline}"

    # General plain or indented text
    leading_spaces = len(line) - len(line.lstrip())
    indent = line[:leading_spaces]
    content = line[leading_spaces:]
    return f"{indent}{TEXT_GREEN}{content}{RESET}{newline}"

class CyberpunkStreamWrapper:
    """Stream wrapper that buffers lines and applies cyberpunk styling."""
    _cyberpunk_wrapped = True

    def __init__(self, target, is_stderr: bool = False):
        self._target = target
        self._is_stderr = is_stderr
        self._buffer = ""

    def write(self, s):
        if not s:
            return
        if not isinstance(s, str):
            try:
                s = str(s)
            except Exception:
                return

        self._buffer += s
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            try:
                self._target.write(style_line(line + "\n"))
            except Exception:
                try:
                    self._target.write(line + "\n")
                except Exception:
                    pass
        try:
            self._target.flush()
        except Exception:
            pass

    def flush(self):
        if self._buffer:
            try:
                self._target.write(style_line(self._buffer))
            except Exception:
                try:
                    self._target.write(self._buffer)
                except Exception:
                    pass
            self._buffer = ""
        try:
            self._target.flush()
        except Exception:
            pass

    @property
    def buffer(self):
        return getattr(self._target, 'buffer', None)

    def __getattr__(self, name):
        return getattr(self._target, name)


def install():
    """Install the universal Cyberpunk output hook on sys.stdout and sys.stderr."""
    # 1. Force UTF-8 encoding
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    # 2. Enable Windows Virtual Terminal Sequences (ANSI 24-bit TrueColor)
    if sys.platform == 'win32':
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            for fd in (-11, -12):
                h = kernel32.GetStdHandle(fd)
                mode = ctypes.c_ulong()
                if kernel32.GetConsoleMode(h, ctypes.byref(mode)):
                    kernel32.SetConsoleMode(h, mode.value | 0x0004)
        except Exception:
            pass

    # 3. Wrap sys.stdout and sys.stderr
    if sys.stdout is not None and not getattr(sys.stdout, '_cyberpunk_wrapped', False):
        sys.stdout = CyberpunkStreamWrapper(sys.stdout)
    if sys.stderr is not None and not getattr(sys.stderr, '_cyberpunk_wrapped', False):
        sys.stderr = CyberpunkStreamWrapper(sys.stderr, is_stderr=True)


# Auto-install upon import
install()
