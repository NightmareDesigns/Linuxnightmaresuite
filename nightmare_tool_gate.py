"""
NIGHTMARE TOOL GATE -- Arming and enforcement for the tool bench
=============================================================
Gates every tool call in Nightmare Suite v13. The gate lives INSIDE the
dispatcher, not at the call site, so there is no path from a <tool> string
to a handler that skips the check.

Activation, any one of three triggers:
  1. The AI asks for tools  -- detected from the user's own phrasing
  2. The persistent switch is on
  3. The "Enable Tools" checkbox in the sidebar is ticked

Lifetime: an activation arms tools for exactly ONE turn. If the model does
not actually call a tool on that turn, the arm is consumed and disarms
itself. Asking for tools does not leave tools available; it gives the model
one chance to use them.

This is a second, independent layer beneath the workbench's own policy
prompt (config/workbench.toml). Either layer can refuse. Neither trusts
the model to police itself: arming is decided in the host process, and
"did a tool actually run" is verified from the dispatch return value.

Pure stdlib. No outside code.

Author: Nightmare Designs
"""

import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple


# Phrasings that mean "you may use tools this turn". Deliberately explicit --
# a loose matcher here would arm tools on ordinary conversation.
ASK_PATTERNS = [
    r"\buse (?:the |your )?tools?\b",
    r"\buse (?:the |your )?tool ?bench(?:h)?\b",
    r"\bwith (?:the |your )?tools?\b",
    r"\bcall (?:a |the )?tools?\b",
    r"\brun (?:a |the )?tools?\b",
    r"\binvoke (?:a |the )?tools?\b",
    r"\bsearch (?:the )?(?:web|internet|files?|system)\b",
    r"\blook (?:this |that )?up\b",
    r"\bgoogle\b",
    r"\bfetch\b",
    r"\bdownload\b",
    r"\bopen (?:a |the )?(?:file|folder|browser|url)\b",
    r"\bcheck (?:my |the )?(?:files?|disk|system|folder)\b",
    r"\bscreenshot\b",
    r"\bcan you (?:run|execute|check|look|search|find|open|read)\b",
    r"\bgo ahead and (?:run|use|call|check|search)\b",
    r"\bshell\b",
    r"\bterminal\b",
    r"\bexecute\b",
]

_ASK_RE = re.compile("|".join(ASK_PATTERNS), re.IGNORECASE)


def user_asked_for_tools(text: str) -> bool:
    """True when the user explicitly invites tool use.

    Kept for the summon gesture and for anyone who wants a one-turn arm.
    Under the risk model it no longer gates ordinary tool use -- see ToolGate.
    """
    if not text:
        return False
    return bool(_ASK_RE.search(str(text)))


class ToolGate:
    """
    Arm / verify / disarm. One instance per Suite run.

        gate = ToolGate()
        gate.arm("switch")            # persistent
        gate.arm("ask", user_text)    # one turn
        ...
        ok, reason = gate.check("fs.list")
        if ok:
            result = handler(args, ctx)
            gate.report_call("fs.list", True)
    """

    def __init__(self, dispatch: Optional[Callable[..., Any]] = None):
        self._lock = threading.RLock()
        self._armed = False
        self._arm_reason: str = ""
        self._armed_at: float = 0.0
        self._persistent = False
        self._used_this_turn = False
        self._dispatch = dispatch

        # Observability, surfaced in the sidebar.
        self.total_calls = 0
        self.denied_calls = 0
        self.arms = 0
        self.recent: List[Dict[str, Any]] = []

    # -- state -------------------------------------------------------------
    def is_armed(self) -> bool:
        with self._lock:
            return self._armed

    def is_persistent(self) -> bool:
        with self._lock:
            return self._persistent

    def reason(self) -> str:
        with self._lock:
            return self._arm_reason

    def was_used(self) -> bool:
        with self._lock:
            return self._used_this_turn

    # -- arming ------------------------------------------------------------
    def arm(self, reason: str = "manual", persistent: bool = False) -> bool:
        """
        Arm tools. Returns True if this changed anything.

        `persistent` keeps tools available across turns until set_persistent
        is called with False. The sidebar switch and the checkbox both use
        this; the "AI asked" path does not.
        """
        with self._lock:
            was = self._armed
            self._armed = True
            self._arm_reason = reason
            self._armed_at = time.time()
            self._used_this_turn = False
            self._persistent = bool(persistent)
            if not was:
                self.arms += 1
            return not was

    def set_persistent(self, on: bool, reason: str = "switch") -> bool:
        """The sidebar switch / checkbox. Stays on until turned off."""
        if on:
            return self.arm(reason, persistent=True)
        return self.disarm(reason)

    def disarm(self, reason: str = "expired") -> bool:
        with self._lock:
            was = self._armed
            self._armed = False
            self._used_this_turn = False
            if not self._persistent:
                self._persistent = False
            self._arm_reason = reason
            return was

    def end_turn(self) -> bool:
        """
        Called when a turn finishes.

        One-turn arms are consumed here. A persistent arm survives, because
        the user asked for tools to stay available until they say otherwise.
        Returns True if the arm was released by this call.
        """
        with self._lock:
            if not self._persistent and self._armed:
                self._armed = False
                self._used_this_turn = False
                self._arm_reason = "turn ended"
                return True
        return False

    def note_no_tool_called(self) -> bool:
        """
        Explicitly record that a turn produced no tool call.
        Same effect as end_turn for a one-turn arm: disarm immediately
        rather than waiting, so the next turn starts clean.
        """
        return self.end_turn()

    # -- enforcement -------------------------------------------------------
    #
    # Risk model, added after the one-turn arm proved to be the wrong axis.
    #
    # The old design locked EVERY tool behind an arm that expired after one
    # turn, and the real workbench tools never consulted it at all (they
    # resolve through _resolve_workbench_tool, which had no check). So the
    # guard cost friction on `calculator` while `fs.delete` walked straight
    # past it.
    #
    # Now: safe tools run. Dangerous ones need a human. The classification
    # comes from nightmare_tool_risk, which reads the workbench's own `risk` field.
    #
    # `authorize` is the single entry point. Every path -- suite registry,
    # workbench by name, router fallback -- must call it. check() and call()
    # are kept below for the one-turn mode and for existing callers.

    def authorize(self, tool_name: str, args: Any = None,
                  reason: str = "", timeout: Optional[float] = None,
                  ) -> Tuple[bool, str]:
        """
        Decide whether `tool_name` may run. Returns (allowed, reason).

        SAFE  -> allowed, no prompt.
        ASK   -> blocks until a human approves, via the approval broker.
        NEVER -> refused outright; the workbench reserves these names so no
                 future tool can take one by accident.

        In one-turn mode this falls back to the armed check, so a user who
        wants the old strict behaviour can still have it.
        """
        name = str(tool_name or "").strip().lower()

        if self.is_one_turn_mode():
            return self.check(name)

        try:
            from nightmare_tool_risk import classify, NEVER, ASK
        except Exception:
            # Risk module unavailable: refuse rather than run unguarded.
            return False, "risk classification unavailable -- refusing %s" % name

        try:
            verdict = classify(name)
        except Exception:
            return False, "risk classification failed -- refusing %s" % name

        if verdict == NEVER:
            self.denied_calls += 1
            self._log(name, "DENIED - reserved name, never permitted")
            return False, "%s is on the never-permitted list" % name

        if verdict == ASK:
            try:
                from nightmare_approval import get_broker
                broker = get_broker()
            except Exception:
                broker = None
            if broker is None:
                self.denied_calls += 1
                return False, "approval broker unavailable"

            self._log(name, "AWAITING APPROVAL")
            ok, why = broker.request(
                name, args,
                reason or "%s is classified as a risky operation" % name,
                timeout=timeout,
            )
            if not ok:
                self.denied_calls += 1
                self._log(name, "DENIED - %s" % why)
                return False, why
            self._log(name, "approved by user")
            return True, ""

        return True, ""

    def is_one_turn_mode(self) -> bool:
        """
        Legacy strict mode: tools must be armed, and the arm lasts one turn.

        Off by default. Set it in the admin panel if you want the old
        behaviour back.
        """
        with self._lock:
            return bool(getattr(self, "_one_turn", False))

    def set_one_turn_mode(self, on: bool) -> bool:
        """
        Toggle legacy mode. Returns the new state.

        Turning it ON disarms immediately, so switching into strict mode can
        never leave tools live by accident.
        """
        with self._lock:
            self._one_turn = bool(on)
            if on:
                self._armed = False
                self._persistent = False
                self._arm_reason = "one-turn mode on"
            else:
                self._arm_reason = "risk model"
            return self._one_turn

    def check(self, tool_name: str) -> Tuple[bool, str]:
        """
        The gate. Returns (allowed, reason_if_denied).

        Called from inside the dispatcher on every single invocation, so a
        new call site cannot skip it.
        """
        with self._lock:
            if self._armed:
                return True, ""
            self.denied_calls += 1
            self._log(tool_name, "DENIED - tools not armed")
            return False, "tools are not armed"

    def report_call(self, tool_name: str, ok: bool = True,
                    detail: str = "") -> None:
        """Record that a tool actually executed."""
        with self._lock:
            self.total_calls += 1
            self._used_this_turn = True
            self._log(tool_name, "ok" if ok else "error", detail)

    def _log(self, tool_name: str, status: str, detail: str = "") -> None:
        self.recent.insert(0, {
            "tool": tool_name,
            "status": status,
            "detail": detail[:200],
            "at": time.strftime("%H:%M:%S", time.localtime()),
        })
        del self.recent[20:]

    # -- dispatch ----------------------------------------------------------
    def call(self, tool_name: str, handler: Callable[..., Any],
             args: Any = None, ctx: Optional[Dict[str, Any]] = None
             ) -> Dict[str, Any]:
        """
        The single sanctioned way to run a tool.

            res = gate.call("fs.list", handler, ["C:/"], {})
            res["ok"]        -> bool
            res["denied"]    -> True when the gate refused
            res["result"]    -> handler return value

        Everything the suite runs goes through here.
        """
        allowed, why = self.check(tool_name)
        if not allowed:
            return {
                "ok": False,
                "denied": True,
                "result": None,
                "reason": why,
            }

        try:
            out = handler(args if args is not None else [], ctx or {})
            self.report_call(tool_name, True)
            return {"ok": True, "denied": False, "result": out,
                    "reason": ""}
        except Exception as exc:  # a bad tool must not kill the turn
            self.report_call(tool_name, False, str(exc))
            return {"ok": False, "denied": False, "result": None,
                    "reason": "%s: %s" % (type(exc).__name__, exc)}

    # -- prompt support ----------------------------------------------------
    def tool_instructions(self, limit: int = 40) -> str:
        """
        What to inject into the system prompt.

        Under the risk model tools are available every turn, so this is no
        longer conditional -- but it is still deliberately NOT the tool list.
        Dumping 260 names costs context on every turn to describe things the model
        is not using right now. The catalogue is already in the model's memory and
        the model recalls it when a task needs it.

        What the model does need is the exact call syntax, because without it every
        attempt is a guess and the model can only ever say "I can't do that".

        Returns "" while disarmed in legacy one-turn mode.
        """
        if self.is_one_turn_mode() and not self.is_armed():
            return ""

        return (
            "\n\nYou got a new tool bench loaded with goodies. Check your "
            "memories for the tool list and functions. Pretty cool.\n\n"
            "You have ONE tool this turn: wb.execute\n"
            "Call it exactly like this, on one line:\n"
            "<tool>wb.execute(task)</tool>\n"
            "`task` is plain English -- what you want done on this PC. "
            "You never name the underlying tools and you never guess at "
            "arguments. Describe the goal in one sentence, e.g. "
            '"list the files in my models folder", '
            '"what is my GPU temperature". Use it only when it genuinely '
            "helps."
        )

    # -- observability -----------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        with self._lock:
            out = {
                "armed": self._armed,
                "persistent": self._persistent,
                "reason": self._arm_reason,
                "usedThisTurn": self._used_this_turn,
                "totalCalls": self.total_calls,
                "deniedCalls": self.denied_calls,
                "arms": self.arms,
            }
        # Outside the lock: these reach into other modules, and holding this
        # lock across them risks a slow call blocking every tool check.
        try:
            from nightmare_approval import get_broker
            out["approvals"] = get_broker().stats()
        except Exception:
            out["approvals"] = {}
        try:
            from nightmare_tool_risk import get_registry
            out["risk"] = get_registry().stats()
        except Exception:
            out["risk"] = {}
        out["oneTurn"] = self.is_one_turn_mode()
        return out

    def status_line(self) -> str:
        s = self.stats()
        if s.get("oneTurn"):
            if s["persistent"]:
                state = "ON (switch)"
            elif s["armed"]:
                state = "armed 1 turn"
            else:
                state = "off"
        else:
            ap = s.get("approvals") or {}
            state = "ON, %d waiting" % ap.get("pending", 0)\
                if ap.get("pending") else "ON"
        return "Tools: %s | %d calls, %d blocked" % (
            state, s["totalCalls"], s["deniedCalls"])


# Module-level singleton -- the suite imports this, not a class instance.
GATE = ToolGate()


def get_gate() -> ToolGate:
    return GATE


if __name__ == "__main__":
    g = ToolGate()

    def echo(args, ctx):
        return "echo: %s" % (args,)

    # 1. Disarmed by default.
    print("default armed:", g.is_armed())
    r = g.call("echo", echo, ["hi"], {})
    print("call while off -> denied=%s ok=%s" % (r["denied"], r["ok"]))

    # 2. Ask detection.
    print("ask detected ('can you check my files'):",
          user_asked_for_tools("can you check my files please"))
    print("ask NOT detected ('hello how are you'):",
          user_asked_for_tools("hello how are you"))

    # 3. Arm from an ask, use it, then auto-disarm.
    if user_asked_for_tools("please use your tools to look this up"):
        g.arm("ai-asked", persistent=False)
    print("armed after ask:", g.is_armed(), "reason:", g.reason())
    r = g.call("echo", echo, ["one turn"], {})
    print("call -> ok=%s result=%s" % (r["ok"], r["result"]))
    g.end_turn()
    print("armed after turn ended:", g.is_armed())

    # 4. A one-turn arm with no tool called disarms on its own.
    g.arm("ai-asked", persistent=False)
    print("re-armed:", g.is_armed())
    g.note_no_tool_called()
    print("armed after no call:", g.is_armed())

    # 5. Persistent switch survives turn boundaries.
    g.set_persistent(True, "switch")
    print("persistent armed:", g.is_armed(), "persistent:", g.is_persistent())
    g.end_turn()
    print("still armed after turn:", g.is_armed())
    g.set_persistent(False, "switch")
    print("after switch off:", g.is_armed())

    # 6. Injection only while armed.
    print("prompt while off is empty:", g.tool_instructions() == "")
    g.set_persistent(True, "switch")
    print("prompt while on has header:",
          "TOOLS ARMED" in g.tool_instructions())
    g.set_persistent(False)

    print("stats:", g.stats())
