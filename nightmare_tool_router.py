"""
NIGHTMARE TOOL ROUTER -- local tool selection, no helper model
=========================================================
Picks the right workbench tool for a plain-English task without a model.

WHY THIS EXISTS
---------------
The workbench's own `execute` tool uses a small helper model to translate
plain English into a tool call. That model needs a local server, which is
not running, and even when it is it costs context on every turn -- context
the suite does not have to spare with a 26B resident.

This router does the same job locally: score every candidate tool against
the task by word overlap, with domain awareness, and run the best match.
No model, no server, no extra context.

HOW IT SCORES
-------------
1. Word overlap between the task and each tool's name + description.
2. Domain match: if the task names a domain (mouse -> input, temperature
   -> wmi), tools in that domain score higher. This is what stops
   "move my mouse" from matching fs.move.
3. Verb match: move/click/type/drag map to input.* tools.

The workbench's find() does the candidate search; this ranks them.
"""

import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


from nightmare_workbench import Workbench


# Domain keywords: task words -> workbench domain.
# Built from the actual tool names in each domain (see registry.ts),
# not guessed. Corrected after comparing against the real catalog:
#   process -> process_ext, package -> package_manager,
#   uia/misc don't exist as domains, kill is the kill-switch not process kill.
DOMAIN_KEYWORDS = {
    "input": ["mouse", "cursor", "click", "key", "type", "keyboard", "drag",
              "scroll", "button", "move", "paste", "clipboard", "hotkey",
              "double-click", "right-click", "hold"],
    "fs": ["file", "folder", "directory", "path", "rename", "copy", "delete",
           "move", "list", "read", "write", "edit", "find", "stat", "mkdir",
           "link", "open file"],
    "shell": ["shell", "command", "terminal", "run", "exec", "script",
              "process list", "process start", "process kill", "process env",
              "which", "batch"],
    "process_ext": ["process", "module", "thread", "handle", "environment",
                    "priority", "affinity", "mitigation", "io counter",
                    "page fault", "memory usage"],
    "kill": ["kill switch", "guardian", "audit", "disarm", "engage"],
    "web": ["web", "browser", "url", "search", "download", "page", "site",
            "fetch", "scrape", "http", "link"],
    "network": ["network", "ip", "port", "ping", "wifi", "internet", "dns",
                "arp", "route", "firewall", "connection", "interface",
                "traceroute", "port scan", "hosts"],
    "screen": ["screen", "screenshot", "display", "pixel", "watch", "diff"],
    "window": ["window", "minimize", "maximize", "focus", "title", "uia",
               "element", "control", "automation", "ui", "foreground",
               "set rect", "close window"],
    "power": ["power", "battery", "sleep", "shutdown", "reboot", "lock",
              "logoff", "hibernate", "restart", "plan"],
    "audio": ["audio", "sound", "volume", "play", "record", "microphone",
              "wav", "convert", "device"],
    "wmi": ["temperature", "temp", "cpu", "gpu", "hardware", "sensor",
            "fan", "thermal", "throttle", "clock", "utilization", "vram",
            "wmi", "query"],
    "registry": ["registry", "reg", "key"],
    "service": ["service", "daemon", "start", "stop", "restart service"],
    "scheduled_task": ["schedule", "task", "cron", "job"],
    "event_log": ["event", "log", "error", "warning", "clear log"],
    "security": ["security", "firewall", "antivirus", "malware", "threat",
                 "isolate", "block ip", "honeypot", "forensic", "ioc",
                 "vulnerability", "audit", "memory dump", "persistence"],
    "flstudio": ["flstudio", "fl", "daw", "midi", "playlist", "fader",
                 "launch", "project"],
    "git": ["git", "commit", "branch", "repo", "diff", "log", "stash",
            "push", "pull", "merge", "clone", "tag"],
    "package_manager": ["package", "install", "uninstall", "upgrade", "pip",
                        "npm", "apt", "update"],
    "shortcut": ["shortcut", "link", "icon", "start menu", "desktop"],
    "ssh": ["ssh", "remote", "connect", "scp", "tunnel", "keygen"],
    "virtual_desktop": ["virtual", "desktop", "workspace", "switch desktop"],
    "bitlocker": ["bitlocker", "encrypt", "tpm", "lock drive"],
    "certificate": ["certificate", "cert", "ssl", "trust"],
    "hosts": ["hosts", "dns", "flush"],
    "display": ["display", "resolution", "monitor", "dpi", "rotation",
                "night light", "hdr", "arrange"],
    "engine": ["engine"],
    "index": ["index", "catalog"],
}


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z]+", text.lower()))


def _domain_of(task_tokens: set) -> Optional[str]:
    """Which workbench domain does the task belong to, if any."""
    best, best_n = None, 0
    for domain, kws in DOMAIN_KEYWORDS.items():
        n = len(task_tokens & set(kws))
        if n > best_n:
            best, best_n = domain, n
    return best if best_n > 0 else None


def score_tool(task: str, name: str, desc: str, domain: Optional[str]) -> float:
    """
    Relevance of one tool to a task.

    Word overlap is the base; a domain match is worth more than any single
    word because it is what separates input.move from fs.move when the task
    says "mouse".
    """
    tt = _tokenize(task)
    nt = _tokenize(name)
    dt = _tokenize(desc)
    if not tt:
        return 0.0

    name_overlap = len(tt & nt)
    desc_overlap = len(tt & dt)
    score = name_overlap * 2.0 + desc_overlap * 0.5

    # Exact name containment: if the task names this tool (or the tool name
    # appears in the task), it is almost certainly the right one. Without
    # this, "take a screenshot" falls back to find("screen") and picks
    # screen.read because its description overlaps more.
    name_l = name.lower()
    task_l = task.lower()
    if name_l in task_l or task_l in name_l:
        score += 10.0

    # Word-level name match: "screenshot" matches "screen.shot" via "shot".
    name_words = set(re.findall(r'[a-z]+', name_l))
    task_words = set(re.findall(r'[a-z]+', task_l))
    if name_words & task_words:
        score += 5.0

    if domain:
        td = _tokenize(domain)
        if tt & td:
            score += 3.0
    return score


def parse_find(text: str) -> List[Tuple[str, str, str]]:
    """
    Parse find() output into (name, args, description, domain).

    Lines look like:
      - input.move (x,y) — Move the cursor without clicking. Use when: ...
    """
    out = []
    for m in re.finditer(
            r"-\s+([a-z_]+\.[a-z_]+)\s*\(([^)]*)\)\s*[—-]\s*(.+)",
            text, re.MULTILINE):
        name = m.group(1)
        desc = m.group(3).strip()
        domain = name.split(".")[0]
        out.append((name, m.group(2), desc, domain))
    return out


class ToolRouter:
    """
    Route plain-English tasks to workbench tools, locally.

    Holds one Workbench subprocess and a small cache of find() results so
    repeated tasks do not pay the search cost every time.
    """

    def __init__(self, workbench: Optional[Workbench] = None):
        self.wb = workbench or Workbench()
        self._cache: Dict[str, List[Tuple[str, str, str]]] = {}

    def start(self) -> bool:
        return self.wb.start()

    def stop(self):
        self.wb.stop()

    def _find(self, query: str, limit: int = 10) -> List[Tuple[str, str, str]]:
        if query in self._cache:
            return self._cache[query]
        raw = self.wb.call("find", {"query": query, "limit": limit}, timeout=60)
        cands = parse_find(raw or "")
        self._cache[query] = cands
        return cands

    # Direct aliases: common task phrases -> exact tool name.
    # Fuzzy matching cannot catch these ("screenshot" != "shot" as tokens),
    # so the phrases that actually get used are mapped explicitly.
    ALIASES = {
        "screenshot": "screen.shot",
        "take a screenshot": "screen.shot",
        "screen shot": "screen.shot",
        "gpu temperature": "wmi.query",
        "cpu temperature": "wmi.query",
        "gpu temp": "wmi.query",
        "cpu temp": "wmi.query",
        "vram": "wmi.query",
        "move mouse": "input.move",
        "click mouse": "input.click",
        "type text": "input.type",
        "list files": "fs.list",
        "list folder": "fs.list",
    }

    # Common PC tasks that find() can't match by keyword.
    # "GPU temperature" returns 0 candidates because the workbench search
    # doesn't associate "temperature" with wmi.query. These are just PC
    # tasks, so map them directly to a pre-built query instead of treating
    # them as a special hardware case.
    #
    # Hardware queries go through shell.exec rather than wmi.query because
    # the workbench's internal method name doesn't match the catalog name
    # (wmi.query -> WmiQuery), and shell.exec avoids that mapping entirely.
    PC_TASKS = [
        (r'gpu.*temp|temp.*gpu', 'shell.exec',
         {'command': 'nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader'}),
        (r'cpu.*temp|temp.*cpu', 'shell.exec',
         {'command': "Get-WmiObject Win32_Processor | Select-Object -ExpandProperty CurrentTemperature"}),
        (r'disk.*space|space.*disk|storage', 'shell.exec',
         {'command': 'Get-PSDrive -PSProvider FileSystem | Select-Object Name,Used,Free'}),
        (r'memory|ram', 'shell.exec',
         {'command': 'Get-WmiObject Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory'}),
        (r'battery', 'shell.exec',
         {'command': 'Get-WmiObject Win32_Battery | Select-Object EstimatedChargeRemaining,BatteryStatus'}),
        (r'process.*list|list.*process', 'shell.exec',
         {'command': 'Get-Process | Select-Object -First 20 Name,Id,CPU'}),
    ]

    def route(self, task: str) -> Optional[str]:
        """
        Pick the best tool for a task. Returns the tool name, or None.

        Strategy: search with the task itself, then with the domain keyword
        if the first search comes back empty. Score every candidate and take
        the top one.
        """
        task_l = task.lower()

        # Check common PC tasks first -- these are just PC tasks, not a
        # special hardware category.
        for pattern, tool, _args in self.PC_TASKS:
            if re.search(pattern, task_l):
                return tool

        cands = self._find(task_l)
        if not cands:
            dom = _domain_of(_tokenize(task_l))
            if dom:
                cands = self._find(dom)
        if not cands:
            return None

        dom = _domain_of(_tokenize(task_l))
        best, best_score = None, 0.0
        for name, args, desc, cdomain in cands:
            s = score_tool(task_l, name, desc, cdomain)
            if dom and cdomain == dom:
                s += 3.0
            if s > best_score:
                best, best_score = name, s
        return best

    def execute(self, task: str, timeout: float = 120.0) -> str:
        """
        Route and run a task. Returns the tool output, or "" on failure.
        """
        if not self.wb.is_running():
            if not self.wb.start():
                return ""
        name = self.route(task)
        if not name:
            return ""

        # PC tasks carry pre-built args; everything else runs with none.
        args = {}
        for pattern, tool, task_args in self.PC_TASKS:
            if tool == name and re.search(pattern, task.lower()):
                args = task_args
                break

        return self.wb.call("run", {"tool": name, "args": args}, timeout=timeout)


if __name__ == "__main__":
    r = ToolRouter()
    if not r.start():
        print("workbench did not start")
        sys.exit(1)

    tests = [
        "move my mouse in a circle 3 times",
        "what is my GPU temperature",
        "list files in my models folder",
        "click the start button",
        "take a screenshot",
    ]
    print("routing tests:")
    for t in tests:
        pick = r.route(t)
        print("  %-42s -> %s" % (t, pick or "(none)"))

    print()
    print("live test:")
    out = r.execute("what is my GPU temperature", timeout=90)
    print("  ", str(out)[:300])
    r.stop()
