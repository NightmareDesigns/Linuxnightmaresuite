"""
NIGHTMARE WORKBENCH BRIDGE -- AI Workbench behind the tool gate
==========================================================
Connects the Nightmare Suite to an external aiworkbench directory,
resolved relative to the user (see _find_workbench_root).

WHAT THIS CHANGES
-----------------
Until now the gate fronted the suite's own 14 tools. This adds the
workbench: 260+ tools across 29 domains, reached through ONE tool called
`execute`. The model says what it wants in plain words and the workbench
decides which underlying tools to use.

That shape is the whole point. Two hundred and sixty tool names in a system
prompt is context the model spends on every turn to describe capabilities it
is not allowed to use. One `execute` costs about forty tokens.

WHY A SUBPROCESS AND NOT AN IMPORT
-----------------------------------
The workbench is TypeScript on Bun. The suite is Python. The only link is
stdio, speaking newline-delimited JSON (the MCP protocol), which is what
python\\aiworkbench.py in the workbench tree already does.

Start lazily. The workbench costs a process and a warm-up, and most turns
never need it. Starting it on suite boot would pay that cost for users who
never ask for a tool.

SECURITY POSTURE
----------------
The workbench reaches shell, filesystem, windows, input and network. The
gate in front of it is the only thing between the model and all of that.

This module therefore:
  * exposes exactly ONE handler, `wb.execute`, so there is no second path in
  * routes every call through gate.call(), never calls a handler directly
  * starts nothing and runs nothing until the gate has armed for the turn
  * does not inject `execute` into the prompt while disarmed, so the model
    cannot go looking for it
  * never starts a subprocess without an explicit user action or an armed
    turn

The workbench's own workbench.toml `policy` is a second, independent layer
and is left at whatever the user configured. Two gates either of which can
refuse.

Author: Nightmare Designs
"""

import json
import os
import re
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional

SUITE_DIR = os.path.dirname(os.path.abspath(__file__))

# Where the workbench lives. Resolved relative to the running user rather
# than baked in, because the previous value named one person's Desktop and
# resolved to nothing on every other machine -- including a customer's.
def _find_workbench_root():
    """First plausible aiworkbench directory that actually exists."""
    home = os.path.expanduser("~")
    user = os.environ.get("USERPROFILE") or home
    candidates = [
        os.path.join(user, "Desktop", "aiworkbench"),
        os.path.join(home, "aiworkbench"),
        os.path.join(SUITE_DIR, "aiworkbench"),
        os.path.join(SUITE_DIR, os.pardir, "aiworkbench"),
    ]
    for c in candidates:
        try:
            if os.path.isdir(c):
                return os.path.abspath(c)
        except Exception:
            continue
    # Nothing on disk yet. Return the conventional location so the error
    # message names where it was expected, rather than a path that never
    # existed on this machine.
    return os.path.abspath(candidates[0])


DEFAULT_WORKBENCH = _find_workbench_root()
DEFAULT_BUN = os.path.join(os.path.expanduser("~"), ".bun", "bin", "bun.exe")

# The MCP entry point. Found by search rather than hardcoded because the
# workbench tree has a nested src\\src\\ layout in at least one copy, and a
# path that is wrong on someone else's machine is worse than a search.
SERVER_CANDIDATES = [
    os.path.join("src", "src", "frontend", "mcp", "server.ts"),
    os.path.join("src", "frontend", "mcp", "server.ts"),
    os.path.join("frontend", "mcp", "server.ts"),
]


class WorkbenchError(Exception):
    """Startup or protocol failure. Distinct from a tool error on purpose."""


class Workbench:
    """
    One workbench subprocess, started on demand.

        wb = Workbench()
        if not wb.is_running():
            wb.start()                       # explicit, or via the gate
        out = wb.call("execute", {"task": "list my model files"})

    Thread-safe for calls: a request/response lock serialises them, because
    the protocol is a single pipe with ids.
    """

    START_TIMEOUT = 45.0          # bun cold-start plus module load
    CALL_TIMEOUT = 180.0          # a task can legitimately take a while
    MAX_RESULT_CHARS = 6000       # do not dump a megabyte into a prompt

    def __init__(self, root: str = DEFAULT_WORKBENCH, bun: str = DEFAULT_BUN):
        # One instance per caller, as originally designed. This was briefly
        # made a singleton so the suite and the WorkbenchLink in
        # nightmare_tools_13 would share one subprocess -- but that also made
        # the two tool systems collapse into one, since a second caller could
        # no longer construct its own bench. Kept separate on purpose.
        self.root = root
        self.bun = bun
        self.proc: Optional[subprocess.Popen] = None
        self._io = threading.RLock()
        self._req_id = 0
        self.started_at = 0.0
        self.last_error = ""
        self.calls = 0

    # ------------------------------------------------------------------
    def server_path(self) -> Optional[str]:
        for rel in SERVER_CANDIDATES:
            p = os.path.join(self.root, rel)
            if os.path.isfile(p):
                return p
        return None

    def is_available(self) -> Dict[str, Any]:
        """Can this be started at all? Reported to the user, not guessed."""
        srv = self.server_path()
        return {
            "installed": os.path.isdir(self.root),
            "server": srv,
            "serverFound": srv is not None,
            "bun": self.bun,
            "bunFound": os.path.isfile(self.bun),
            "ok": bool(srv) and os.path.isfile(self.bun),
        }

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ------------------------------------------------------------------
    def start(self) -> bool:
        """
        Start the subprocess and wait for its ready line.
        Returns True on success. Never raises -- a missing workbench must not
        take the suite down.
        """
        with self._io:
            if self.is_running():
                return True

            avail = self.is_available()
            if not avail["ok"]:
                self.last_error = (
                    "workbench not runnable (installed=%s server=%s bun=%s)"
                    % (avail["installed"], avail["serverFound"],
                       avail["bunFound"]))
                return False

            try:
                # stderr goes to a log file, NOT a pipe. A pipe that nobody
                # drains deadlocks the child, and the server logs its banner to
                # stderr while keeping stdout for JSON-RPC -- so stdout must
                # stay a pipe and stderr must not.
                err_path = os.path.join(self.root, "workbench_stderr.log")
                with open(err_path, "a", encoding="utf-8", errors="replace") as errf:
                    self.proc = subprocess.Popen(
                        [self.bun, "run", avail["server"]],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=errf,
                        cwd=self.root,
                        text=True,
                        bufsize=1,
                        encoding="utf-8",
                        errors="replace",
                    )
            except Exception as e:
                self.last_error = "spawn failed: %s" % e
                self.proc = None
                return False

            # Wait for the ready line. It arrives on STDERR (the server logs
            # its banner there and keeps stdout for JSON-RPC), and stderr now
            # goes to a log file -- so poll that file rather than reading a
            # pipe. Reading stdout here blocked forever, which is the
            # "Starting workbench..." hang.
            deadline = time.time() + self.START_TIMEOUT
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    err = ""
                    try:
                        with open(err_path, "r", encoding="utf-8",
                                  errors="replace") as fh:
                            err = (fh.read() or "")[-300:]
                    except Exception:
                        pass
                    self.last_error = ("workbench exited during startup: %s"
                                       % err.strip())
                    self.proc = None
                    return False

                try:
                    with open(err_path, "r", encoding="utf-8",
                              errors="replace") as fh:
                        log = fh.read()
                except Exception:
                    log = ""
                if "ready" in log.lower():
                    self.started_at = time.time()
                    self.last_error = ""
                    return True
                time.sleep(0.2)

            self.last_error = "workbench did not signal ready in %ds" %\
                self.START_TIMEOUT
            self.stop()
            return False

    def stop(self) -> None:
        with self._io:
            proc = self.proc
            self.proc = None
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _send(self, req: Dict[str, Any], timeout: Optional[float] = None) -> str:
        """Send one JSON-RPC request and return the response text."""
        with self._io:
            if not self.is_running() and not self.start():
                return ""
            self._req_id += 1
            rid = self._req_id
            req["id"] = rid
            try:
                self.proc.stdin.write(json.dumps(req) + "\n")
                self.proc.stdin.flush()
            except Exception as e:
                self.last_error = "write failed: %s" % e
                self.stop()
                return ""

            deadline = time.time() + (timeout or self.CALL_TIMEOUT)
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    self.last_error = "workbench died during a call"
                    self.stop()
                    return ""
                line = self.proc.stdout.readline()
                if not line:
                    continue
                try:
                    resp = json.loads(line)
                except Exception:
                    continue
                if resp.get("id") != rid:
                    continue
                self.calls += 1
                if "error" in resp:
                    self.last_error = json.dumps(resp["error"])[:300]
                    return ""
                result = resp.get("result", {}) or {}
                content = result.get("content") or []
                if content and isinstance(content, list):
                    return self._trim(content[0].get("text", ""))
                return self._trim(json.dumps(result))
            self.last_error = "call timed out"
            return ""

    def _call_tool(self, tool: str, args: Optional[Dict[str, Any]] = None,
                   timeout: Optional[float] = None) -> str:
        return self._send({
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {"name": tool, "arguments": args or {}},
        }, timeout)

    def _extract_tool_names(self, text: str) -> List[str]:
        """Pull tool names out of a find() result."""
        import re
        names = re.findall(r'^\s+([a-z_]+\.[a-z_]+)\s*-', text, re.MULTILINE)
        if not names:
            names = re.findall(r'\b([a-z_]+\.[a-z_]+)\b', text)
        return names[:5]

    def _score(self, task: str, name: str, desc: str) -> float:
        """Relevance score: word overlap between task and tool name+description."""
        import re
        task_words = set(re.findall(r'[a-z]+', task.lower()))
        if not task_words:
            return 0.0
        name_words = set(re.findall(r'[a-z]+', name.lower()))
        desc_words = set(re.findall(r'[a-z]+', desc.lower()))
        # Name matches weigh more than description matches.
        return (len(task_words & name_words) * 2.0
                + len(task_words & desc_words) * 0.5)

    def _execute_task(self, args: Dict[str, Any],
                      timeout: Optional[float] = None) -> str:
        """
        Translate a plain-English task into a tool call.

        `execute` -- the server's own natural-language engine -- is not
        exposed in mcp mode; the server answers `Unknown tool "execute"`.
        So this does the selection here: find() for candidates, then run the
        best one through `run`, which is the universal dispatcher.

        Two hard-won constraints shape it:

        1. Candidates that take NO required arguments are preferred. `run`
           validates against the tool's schema and rejects a missing param,
           so a tool needing `path` or `program` cannot be called from a
           plain-English task without inventing a value. "list all visible
           windows" -> window.list works; "list the files in X" -> fs.list
           cannot, and guessing would just produce BAD_ARGS.
        2. The candidate must go through `run`, never under its own name.
           Sending "flstudio.act" directly gets `Unknown tool -- this server
           exposes only: find, learn, playbook, run`.
        """
        task = str((args or {}).get("task", "")).strip()
        if not task:
            return ""

        find_out = self._call_tool("find", {"query": task, "limit": 8}, timeout)
        if not find_out:
            return ""

        # "name (args) -- description" lines off find()'s output. The parenthesised
        # argument list is OPTIONAL: zero-argument tools render as plain
        # "window.list -- List visible top-level windows", and requiring the
        # parens silently dropped every tool that could actually be called,
        # which is the opposite of what this needs.
        candidates = []
        for m in re.finditer(
                r'-\s+([a-z_]+\.[a-z_]+)\s*(?:\(([^)]*)\))?\s*[—-]\s*(.+)',
                find_out, re.MULTILINE):
            candidates.append((m.group(1), (m.group(2) or "").strip(),
                               m.group(3).strip()))
        if not candidates:
            self.last_error = "no tool found for: %s" % task[:80]
            return ""

        scorable = [(n, sig, d) for n, sig, d in candidates]
        # A zero-argument candidate is callable as-is; anything else needs a
        # value we do not have. Only fall back to those if nothing is free.
        free = [c for c in scorable if not c[1]]
        pool = free or scorable

        best = max(pool, key=lambda c: self._score(task, c[0], c[2]))
        return self._call_tool("run", {"tool": best[0], "args": {}}, timeout)

    def call(self, tool: str, args: Optional[Dict[str, Any]] = None,
             timeout: Optional[float] = None) -> str:
        """
        One tool call. Returns text, or "" on failure.

        Never raises. A workbench that dies mid-session must degrade to "no
        tools", not to a crash in the chat loop.

        `execute` is the worker-mode name; in mcp mode we translate it through
        find -> run so the suite's wb.execute interface keeps working.
        """
        if tool == "execute":
            return self._execute_task(args or {}, timeout)
        return self._call_tool(tool, args, timeout)

    def _trim(self, text: str) -> str:
        t = str(text or "")
        if len(t) <= self.MAX_RESULT_CHARS:
            return t
        return (t[:self.MAX_RESULT_CHARS]
                + "\n...[truncated, %d more chars]" % (len(t)
                                                        - self.MAX_RESULT_CHARS))

    def stats(self) -> Dict[str, Any]:
        return {
            "running": self.is_running(),
            "uptimeSec": round(time.time() - self.started_at, 1)
            if self.started_at else 0,
            "calls": self.calls,
            "lastError": self.last_error,
            **self.is_available(),
        }


# ---------------------------------------------------------------------------
# The single gated handler
# ---------------------------------------------------------------------------

WB_TOOL_NAME = "wb.execute"

PROMPT_BLOCK = """
=== WORKBENCH ARMED ===
You have one tool this turn:

  <tool>wb.execute(task)</tool>

`task` is plain English: what you want done on this PC. For example
"list the files in my models folder", "what is my GPU temperature",
"search the web for the latest Gemma release".

The workbench decides which underlying tools to use. You never name them
and you never guess at arguments. Describe the goal in one sentence.

Use it only when it genuinely helps. If it is not needed, do not call it.
"""


class WorkbenchBridge:
    """
    Adapts the workbench to the suite's tool calling convention and puts it
    behind the gate.

        link = WorkbenchLink(gate, workbench)
        handler = link.handler()          # register this in ALL_TOOLS
        handler(["find my model files"], {}) # gate checks inside
    """

    def __init__(self, gate: Any = None, workbench: Optional[Workbench] = None,
                 root: str = DEFAULT_WORKBENCH):
        from nightmare_tool_gate import get_gate
        self.gate = gate or get_gate()
        self.wb = workbench or Workbench(root=root)
        self._status: Optional[Callable[[str], None]] = None

    # ------------------------------------------------------------------
    def handler(self):
        """The callable the suite registers. Same shape as every other tool."""
        def _handler(args: Any, ctx: Optional[Dict[str, Any]] = None) -> str:
            return self.execute(args, ctx)
        return _handler

    def execute(self, args: Any, ctx: Optional[Dict[str, Any]] = None) -> str:
        """
        Run one task.

        The arm check lives HERE, inside the callable, so registering this
        anywhere in the suite still cannot bypass the gate. That is the
        property that matters: a new call site cannot skip enforcement.
        """
        task = self._first_arg(args)
        if not task:
            return "Error: no task given."

        allowed, why = self.gate.check(WB_TOOL_NAME)
        if not allowed:
            # A refusal is not an execution. report_call would count it as a
            # completed call and inflate the "used" counter in the sidebar,
            # so only the check's own denial counter moves.
            return ("Tool gate: %s. Ask me to use tools, or turn on the "
                    "Tools switch." % why)

        # Start lazily, but only now -- after the arm check. Nothing about
        # the workbench should happen on a turn that has no tools.
        if not self.wb.is_running():
            self.output_status("Starting workbench...")
            if not self.wb.start():
                return "Workbench unavailable: %s" % self.wb.last_error

        out = self.wb.call("execute", {"task": task})
        if out:
            self.gate.report_call(WB_TOOL_NAME, True)
            self.output_status("Workbench finished.")
            return out

        err = self.wb.last_error or "no result"
        self.gate.report_call(WB_TOOL_NAME, False, err)
        # A failed task disarms, same as no task at all.
        try:
            self.gate.note_no_tool_called()
        except Exception:
            pass
        return "Workbench error: %s" % err

    # -- test seams ------------------------------------------------------
    def set_status_sink(self, fn: Optional[Callable[[str], None]]) -> None:
        """Let the suite surface start/finish progress. None = silent."""
        self._status = fn

    def _status_emit(self, msg: str) -> None:
        fn = getattr(self, "_status", None)
        if fn:
            try:
                fn(msg)
            except Exception:
                pass

    @staticmethod
    def _first_arg(args: Any) -> str:
        if isinstance(args, (list, tuple)):
            return str(args[0]).strip() if args else ""
        if isinstance(args, dict):
            vals = list(args.values())
            return str(vals[0]).strip() if vals else ""
        return str(args or "").strip()

    def output_status(self, msg: str) -> None:
        """Hook for the suite to surface progress. Silent by default."""
        self._status_emit(msg)

    # ------------------------------------------------------------------
    def prompt_block(self) -> str:
        """
        What to inject while armed. Empty while disarmed, so the model is not told
        the workbench exists until it may use it.
        """
        if not self.gate.is_armed():
            return ""
        return PROMPT_BLOCK

    def register_into(self, registry: Dict[str, Any]) -> str:
        """Add the one handler to a nightmare_tools_13.ALL_TOOLS dict."""
        registry[WB_TOOL_NAME] = self.handler()
        return WB_TOOL_NAME

    def available(self) -> bool:
        return self.wb.is_available()["ok"]

    def stats(self) -> Dict[str, Any]:
        return {
            "gate": self.gate.stats(),
            "workbench": self.wb.stats(),
            "toolName": WB_TOOL_NAME,
        }


_BRIDGE: Optional[WorkbenchBridge] = None


def get_bridge(gate: Any = None, root: str = DEFAULT_WORKBENCH
               ) -> WorkbenchBridge:
    global _BRIDGE
    if _BRIDGE is None:
        _BRIDGE = WorkbenchBridge(gate=gate, root=root)
    return _BRIDGE


if __name__ == "__main__":
    from nightmare_tool_gate import ToolGate, user_asked_for_tools

    wb = Workbench()
    print("=== availability ===")
    for k, v in wb.is_available().items():
        print("  %-14s %s" % (k, v))
    print("  server path  %s" % (wb.server_path() or "NOT FOUND"))

    gate = ToolGate()
    bridge = WorkbenchBridge(gate=gate, workbench=wb)

    print("\n=== gate is in front of it ===")
    out = bridge.execute(["do something dangerous"], {})
    check_deny = "not armed" in out.lower() or "gate" in out.lower()
    print("  disarmed call refused: %s" % check_deny)
    print("  refusal text: %r" % out[:90])

    print("\n=== prompt injection only while armed ===")
    print("  disarmed block empty: %s" % (bridge.prompt_block() == ""))
    gate.arm("test")
    print("  armed block non-empty: %s" % ("wb.execute" in bridge.prompt_block()))
    gate.disarm()

    print("\n=== ask detection feeds it ===")
    print("  'can you check my files' ->", user_asked_for_tools("can you check my files"))
    print("  'use your tools to look this up' ->",
          user_asked_for_tools("use your tools to look this up"))

    print("\n=== registry shape ===")
    reg: Dict[str, Any] = {}
    name = bridge.register_into(reg)
    print("  registered: %s  callable: %s" % (name, callable(reg[name])))
