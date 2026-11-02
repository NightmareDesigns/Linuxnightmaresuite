"""
Function calling, the way every other client does it.

Generic converter: works with any model loaded in the Suite, not
tied to any particular model.

THE PROBLEM THIS REPLACES
-------------------------
Tools used to travel as one invented text protocol:

    <tool>name(args)</tool>

parsed by a single regex, with one syntax. A model that writes
`<|tool_call|>` (Qwen, Hermes), a bare JSON object, or the OpenAI
`tool_calls` array gets nothing, because none of those match. Every
client on earth handles those; this one handled one.

WHAT IS HERE NOW
----------------
1. build_tools_array()  - an OpenAI-shaped `tools` list, built from the
   LIVE workbench catalogue. Not a hand-written table: if the workbench
   does not know a tool, the model is not told it exists, and if the workbench
   is down the list is empty rather than fictional.

2. render_tools_prompt() - this engine takes no `tools` field, it is
   prompt-driven, so the array is rendered into the system prompt in
   the same shape, with a token estimate. Which means the cost is
   visible instead of surprising.

3. parse_tool_calls()  - every dialect in one place, longest-form first:
       <|tool_call|>{"name":..,"arguments":..}<|/tool_call|>
       <tool_call>{"name":..,"arguments":{..}}</tool_call>
       <tool_call>name(args)</tool>            (the old syntax, still works)
       <tool_call>{"name":..,"args":..}</tool_call>
       a bare {"tool":..,"args":..} or {"name":..,"arguments":..}
   Each returns {id, name, arguments(dict), syntax} so the caller can
   log which form the model used, and so an unparsable argument string is
   reported rather than silently sent as {}.

4. format_tool_result() - the assistant/tool turn, in OpenAI's shape, so
   the next pass sees what happened and can react.

ORDER MATTERS IN parse_tool_calls. The JSON forms are tried before the
bare-args form because `<tool>x({"a":1})</tool_call>` also matches the bare
form, and the JSON is the right read.

Arguments are ALWAYS a dict when they can be. A model that writes
arguments as a JSON *string* (the single most common malformed case, and
what several trainers emit) is unwrapped rather than dropped.

Author: Nightmare Designs
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "build_tools_array", "render_tools_prompt", "parse_tool_calls",
    "format_tool_result", "render_call_args", "split_call_args",
    "refused_result", "tools_prompt_tokens", "TOOL_CALL_SYNTAXES",
]

# Recorded so a caller can see which dialect a model actually prefers.
# Useful when one model keeps writing a form the parser has to work for.
TOOL_CALL_SYNTAXES = (
    "qwen",        # <|tool_call|>{json}<|/tool_call|>
    "hermes",      # <tool_call>{json}</tool_call>
    "legacy",      # <tool>name(args)</tool>
    "bare_json",   # a bare {"name":..,"arguments":..} object
    "bare_name",   # a bare name(args) call, whole text only
)

_NAME_RE = r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*"


# ── 1. the schema ─────────────────────────────────────────────────────────

# The MCP server has no tools/list. It exposes four meta-tools:
#   find, learn, playbook, run
# and find() is the catalogue. Its answer is text, one line per tool:
#
#   - audio.play (path) — Play an audio file through the default output
#     device. Use when: Auditioning a bounce or export to check it sounds
#     right.
#
# So the schema is parsed out of find()'s prose. That is the only source
# that exists; a hand-written table would go stale the moment a tool was
# added, and would keep claiming tools that had been removed.

_FIND_LINE = re.compile(
    r"^-\s*(?P<name>[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*)"
    r"(?:\s*\((?P<params>[^)]*)\))?"
    r"\s*(?:[—–-]\s*(?P<desc>.*))?$"
)


def parse_find_output(text: str) -> List[Dict[str, Any]]:
    """
    Turn find()'s answer into {name, description, parameters}.

    Two things this has to get right, both of which a naive per-line parse
    gets wrong:

    1. find() WRAPS descriptions. A tool's description arrives as an
       indented continuation:

           - audio.analyze (path) — Measure a WAV's peak and RMS. Use when:
             Verifying a recording is real, not silent and not clipping.

       Reading line by line keeps only the first half and hands the
       model a description that stops mid-sentence at "Use when:".

    2. find() gives parameter NAMES, not a schema: `fs.list (path,
       recursive)`, with no types and nothing marked required. Building a
       `properties` map with invented types would be inventing a contract
       the server never stated, so the names are kept as names. See
       _params_schema.

    Returns [] when there is nothing tool-shaped in the text.
    """
    out: List[Dict[str, Any]] = []
    seen = set()
    cur: Optional[Dict[str, Any]] = None

    def flush() -> None:
        nonlocal cur
        if cur is None:
            return
        cur["description"] = " ".join(" ".join(cur["description"]).split())
        out.append(cur)
        cur = None

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("-"):
            m = _FIND_LINE.match(line)
            if not m:
                continue
            name = m.group("name")
            flush()
            if name in seen:
                cur = None
                continue
            seen.add(name)
            params = [p.strip() for p in (m.group("params") or "").split(",")
                      if p.strip()]
            cur = {
                "name": name,
                "description": [(m.group("desc") or "").strip()],
                "parameters": params,
            }
        elif cur is not None:
            # an indented continuation of the description above
            cur["description"].append(line)
    flush()
    return out



def build_tools_array(workbench: Any, query: str = "", limit: int = 60) -> List[Dict[str, Any]]:
    """
    An OpenAI `tools` array describing what the model can actually call.

    Built from the live workbench via find(). Returns [] when the
    workbench cannot be reached -- an empty list is honest, a stale
    hard-coded list is not, and a fictional tool name gets the model a refusal
    from the server that reads like a broken tool.

    `query` is what the model is likely to want. find() ranks by relevance to
    it, so a task-shaped query returns the tools that fit that task --
    which is the point of `limit`: rendering all ~260 into the system
    prompt would cost more context than the model has. Ask again with a
    different query and the model gets a different, equally real, list.

    Sorted by name so the same query gives the same prompt turn to turn,
    rather than the list shuffling as the server's ranking shifts.
    """
    if workbench is None:
        return []
    raw = ""
    try:
        # _call_tool is the suite's own wrapper and handles start-up.
        # .call is the raw transport. Try both: which one exists depends
        # on which version of the bench is installed.
        if hasattr(workbench, "_call_tool"):
            raw = workbench._call_tool("find", {"query": query or "tools",
                                                 "limit": 400}, timeout=30)
        elif hasattr(workbench, "call"):
            raw = workbench.call("run", {"tool": "find",
                                         "args": {"query": query or "tools",
                                                  "limit": 400}}, timeout=30)
    except Exception:
        return []
    if not raw:
        return []

    found = parse_find_output(str(raw))
    if not found:
        # A JSON tools list, if a future server offers one. Cheap to keep.
        try:
            parsed = json.loads(str(raw))
        except Exception:
            return []
        items = parsed.get("tools") if isinstance(parsed, dict) else parsed
        if not isinstance(items, list):
            return []
        for it in items:
            if not isinstance(it, dict):
                continue
            fn = it.get("function") if isinstance(it.get("function"), dict) else it
            nm = str(fn.get("name") or "").strip()
            if not nm:
                continue
            found.append({
                "name": nm,
                "description": str(fn.get("description") or "").strip(),
                "parameters": fn.get("parameters")
                               or fn.get("inputSchema")
                               or {"type": "object", "properties": {}},
            })

    tools = [{
        "type": "function",
        "function": {
            "name": f["name"],
            "description": f["description"],
            "parameters": _params_schema(f["parameters"]),
        },
    } for f in found]

    tools.sort(key=lambda t: t["function"]["name"])
    return tools[:limit] if limit and len(tools) > limit else tools


def _params_schema(params: Any) -> Dict[str, Any]:
    """
    find() gives names, not types. Keep them as a string-typed map with
    no required list -- every property optional, because the server never
    said which are. An invented `required` list makes a model refuse to
    call a tool that would have worked.
    """
    if isinstance(params, dict):
        return params or {"type": "object", "properties": {}}
    names = [p for p in (params or []) if isinstance(p, str) and p]
    if not names:
        return {"type": "object", "properties": {}}
    return {
        "type": "object",
        "properties": {n: {"type": "string"} for n in names},
        "required": [],
    }



def render_tools_prompt(tools: List[Dict[str, Any]],
                        instructions: str = "") -> str:
    """
    The `tools` array as prompt text, because this engine has no tools field.

    Same JSON, so a model that has seen this shape in training recognises
    it, plus one worked example of the expected reply shape. The example
    is what stops a model inventing a third syntax.
    """
    if not tools:
        return ""
    lines: List[str] = ["=== TOOLS (call one if it helps; otherwise just answer) ==="]
    if instructions:
        lines.append(instructions)
    lines.append(json.dumps({"tools": tools}, ensure_ascii=False))
    lines.append(
        'To call a tool, reply with exactly one tool call and nothing else, '
        'in this shape:\n'
        '<tool_call>{"name": "the.tool.name", '
        '"arguments": {"param": "value"}}</tool_call>'
    )
    return "\n".join(lines)


def tools_prompt_tokens(tools: List[Dict[str, Any]]) -> int:
    """Rough token cost, for the prefill readout. chars/4 is the same
    estimate used elsewhere in the suite."""
    return len(render_tools_prompt(tools)) // 4


# ── 2. the parser ─────────────────────────────────────────────────────────

def _coerce_args(raw: Any) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Get a dict out of whatever the model wrote.

    Returns (args, error). error is set only when the text LOOKS like
    arguments and is not readable -- the caller should surface it rather
    than call the tool with {} and blame the tool.

    Handles the common malformed case where arguments arrive as a JSON
    *string* rather than an object, which several trainers emit.
    """
    if raw is None or raw == "":
        return {}, None
    if isinstance(raw, dict):
        return dict(raw), None
    if isinstance(raw, (list, int, float, bool)):
        return {"input": raw}, None
    if not isinstance(raw, str):
        return {}, None

    text = raw.strip()
    if not text:
        return {}, None

    # Bare key=value pairs, e.g. the old syntax: path="C:\tmp"
    if "=" in text and "{" not in text:
        out: Dict[str, Any] = {}
        for m in re.finditer(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|[^,]+)',
                             text):
            k = m.group(1)
            v = m.group(2).strip()
            if v[:1] in ("'", '"') and v[-1:] == v[:1]:
                v = v[1:-1]
            out[k] = v
        if out:
            return out, None

    # JSON object
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed, None
        return {"input": parsed}, None
    except Exception:
        pass

    # A bare positional value: treat it as the sole argument.
    if " " not in text and "\n" not in text:
        return {"input": text}, None

    return {}, ("could not read the arguments: %s" % (text[:120],))


def _strip_fences(text: str) -> str:
    """Remove ```json fences around a payload, which models add freely."""
    t = text.strip()
    m = re.match(r"^```(?:json|tool_call|xml)?\s*(.*?)\s*```$", t, re.S)
    return m.group(1) if m else t


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """
    Every tool-call dialect this model family is known to emit.

    Returns a list of {id, name, arguments, syntax, error}. `error` is
    non-None only when the name was understood and the arguments were
    not -- the caller should refuse that one call and say why, rather
    than run a tool with no arguments.

    An empty list means the model did not ask for a tool. That is the common
    case and is not an error.
    """
    if not text:
        return []
    out: List[Dict[str, Any]] = []
    n = 0

    def add(name: str, raw_args: Any, syntax: str, raw: str = "") -> None:
        nonlocal n
        name = (name or "").strip()
        if not name or not re.match(r"^%s$" % _NAME_RE, name):
            return
        args, err = _coerce_args(raw_args)
        n += 1
        out.append({
            "id": "call_%d" % n,
            "name": name,
            "arguments": args,
            "syntax": syntax,
            "error": err,
            "raw": raw,
        })

    def from_obj(obj: Any, syntax: str) -> bool:
        """Pull name+arguments out of a decoded object. True if handled."""
        if not isinstance(obj, dict):
            return False
        fn = obj.get("function") if isinstance(obj.get("function"), dict) else obj
        name = fn.get("name") or fn.get("tool") or fn.get("tool_name")
        if not name:
            return False
        raw = fn.get("arguments")
        if raw is None:
            raw = fn.get("parameters")
        if raw is None:
            raw = fn.get("args")
        add(str(name), raw, syntax, json.dumps(obj, ensure_ascii=False)[:400])
        return True

    # -- 1. qwen: <|tool_call|>{...}<|/tool_call|> -------------------------
    qwen = re.findall(r"<\|tool_call\|>(.*?)(?:<\|/tool_call\|>|$)",
                      text, re.S)
    for payload in qwen:
        body = _strip_fences(payload)
        if not from_obj(_try_json(body), "qwen"):
            _name_args_text(body, add, "qwen")

    # -- 2. hermes: <tool_call>{...}</tool_call> ---------------------------------------
    hermes = re.findall(r"<tool_call>(.*?)(?:</tool_call>|$)", text, re.S)
    for payload in hermes:
        body = _strip_fences(payload)
        if not from_obj(_try_json(body), "hermes"):
            _name_args_text(body, add, "hermes")

    # -- 3. the old syntax: <tool>name(args)</tool> and <tool>{json}</tool> --
    legacy = re.findall(r"<tool>(.*?)</tool>", text, re.S)
    for payload in legacy:
        body = payload.strip()
        if not from_obj(_try_json(body), "legacy"):
            _name_args_text(body, add, "legacy")

    # -- 4. a bare JSON object anywhere -------------------------------------
    # Brace-balanced, not a regex. The obvious pattern
    #   \{[^{}]*"name"[^{}]*\}          (no nested braces)
    # cannot match the commonest real case:
    #   {"name":"fs.list","arguments":{"path":"C:/tmp"}}
    # because [^{}]* stops at the inner brace. That is why this had to be
    # written as a scan rather than a pattern.
    if not out:
        for chunk in _iter_json_objects(text):
            if from_obj(_try_json(chunk), "bare_json"):
                break

    # -- 5. a bare name(args) call, with no tags at all -------------------
    # Generic addition (not in the original): a model that writes just
    # `calculator(2+3)` and nothing else is clearly making a tool call,
    # and no other dialect above catches it. Only the whole text, with
    # nothing else around it, is accepted here, so ordinary prose --
    # which never matches ^name(args)$ on its own -- is never converted
    # into a call by this step. Plain prose tasks are handled instead by
    # the suite's guarded plain-English fallback, not by this parser.
    if not out:
        _name_args_text(_strip_fences(text), add, "bare_name")

    return out


def _iter_json_objects(text: str):
    """
    Yield outermost balanced {...} spans, left to right, skipping ones
    nested inside an already-yielded span.

    String-aware: a brace inside a JSON string literal does not open a
    scope, so "{\"a\":1}" does not desynchronise the count. A path
    argument like C:\\{tmp} is exactly why that matters.
    """
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]
                    start = -1


def _try_json(s: str) -> Any:
    try:
        return json.loads(_strip_fences(s))
    except Exception:
        return None


def _name_args_text(body: str, add, syntax: str) -> None:
    """`name(arg, arg)` or a bare `name` with no arguments."""
    m = re.match(r"^\s*(%s)\s*\((.*)\)\s*$" % _NAME_RE, body, re.S)
    if m:
        add(m.group(1), m.group(2).strip(), syntax, body[:400])
        return
    m = re.match(r"^\s*(%s)\s*$" % _NAME_RE, body)
    if m:
        add(m.group(1), "", syntax, body[:400])


# ── 3. the result ─────────────────────────────────────────────────────────

def format_tool_result(calls: List[Dict[str, Any]],
                      results: List[str],
                      model: str = "assistant") -> str:
    """
    The assistant/tool turn, in OpenAI's shape, to append before the
    next pass.

    Every call gets a result even when it failed, including one whose
    arguments could not be read. A call the model cannot see the outcome
    of is a call it will repeat, and the error message is the only thing
    that tells it to stop.
    """
    if not calls:
        return ""
    parts: List[str] = []
    for i, c in enumerate(calls):
        body = results[i] if i < len(results) else "(no result returned)"
        if not str(body).strip():
            body = "(the tool returned nothing)"
        parts.append(
            '<tool_response>{"tool_call_id": "%s", "name": "%s", '
            '"content": %s}</tool_response>'
            % (c.get("id", "call_%d" % (i + 1)),
               c.get("name", "?"),
               json.dumps(str(body), ensure_ascii=False))
        )
    return "\n".join(parts)


def refused_result(reason: str) -> str:
    """What a blocked or unreadable call looks like to the model."""
    return "[refused: %s]" % (reason or "not permitted")


# -- 4. the link back to a string-shaped dispatcher -----------------------

def render_call_args(arguments: Any) -> str:
    """
    Flatten a parsed call's arguments back into `key=value` text.

    The suite's dispatch path predates structured arguments: _parse_tool_args,
    gate.authorize, the router's task string and the positional split all read
    a string like `path="C:\\tmp", mode="rw"`. Rather than rewrite every one
    of those call sites, the dict is rendered back into the shape they expect.

    Nested values are JSON-encoded, because `key=value` cannot carry an object
    and a half-written one is worse than none. A model that passes a dict
    where the tool wants a string gets quoted JSON -- wrong, but visible --
    instead of an empty value that is wrong and silent.

    Returns "" for no arguments, which every call site already reads as
    "nothing to pass".
    """
    if not isinstance(arguments, dict) or not arguments:
        return ""
    parts: List[str] = []
    for k, v in arguments.items():
        if isinstance(v, str):
            parts.append('%s="%s"' % (k, v.replace('"', '\\"')))
        elif v is None:
            parts.append("%s=" % k)
        elif isinstance(v, (int, float, bool)):
            parts.append("%s=%s" % (k, v))
        else:
            parts.append("%s=%s" % (k, json.dumps(v, ensure_ascii=False)))
    return ", ".join(parts)


def split_call_args(argstr: str) -> List[str]:
    """
    Split `a="1", b="2, 3"` on commas that are NOT inside quotes.

    Fixes one specific bug: `argstr.split(",")` turns a single argument
    whose value contains a comma into several arguments. Of the 131
    built-in tools this path serves, file_read, docx_create and
    browser_scrape all take free text, so almost every real call had this
    wrong. The workbench path never had the bug -- _parse_tool_args matches
    quoted runs -- which is why it was not obvious.

    DELIBERATELY PRESERVES the old `.strip("'\"")` on every piece, so
    nothing that worked before stops working. That strip is itself
    suspect -- it turns `path="C:/tmp"` into `path="C:/tmp`, unbalanced --
    but the 131 handlers were written against it and this cannot verify
    what they expect. Left alone rather than guessed at. See the note in
    HANDOFF.md.

    Returns [] for empty input, which the caller already treats as
    "this tool takes no arguments".
    """
    text = (argstr or "").strip()
    if not text:
        return []
    out: List[str] = []
    buf: List[str] = []
    quote = ""
    escaped = False
    for ch in text:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if ch == "\\" and quote:
            buf.append(ch)
            escaped = True
            continue
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = ""
            continue
        if ch in "\"'":
            quote = ch
            buf.append(ch)
            continue
        if ch == ",":
            out.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    if buf:
        out.append("".join(buf))
    return [p.strip().strip("'\"") for p in out if p.strip()]


