"""
Constrained Decoding — hand-written, no external code.

Provides incremental, tokenizer-aware constraints that restrict sampling to
tokens which keep the generated text valid against a small grammar:

  * JSON mode           -> every emitted piece must keep the running document
                            a valid JSON prefix (OpenAI "json_object" mode,
                            tool-call arguments, structured output).
  * JSON-with-schema    -> adds key completion: inside an object, only keys
                           that are prefixes of the allowed schema keys are
                           permitted (function/tool calling).
  * Sequence prefix     -> force a fixed token/text prefix first.

Each constraint exposes an `allowed(candidate_ids)` set per step; callers feed
that set into Sampler.sample(logits, mask) or the link sample_token mask.

The tokenizer contract is: an object with `decode_single(token_id) -> str`
(any of the suite's tokenizers / DllEngine via decode([tid])).
"""

from typing import Optional, Sequence

_WS = " \t\r\n"
_DIGITS = "0123456789"
_HEX = "0123456789abcdefABCDEF"
_KEYWORDS = ("true", "false", "null")


# ---------------------------------------------------------------------------
# JSON prefix state machine
# ---------------------------------------------------------------------------
# A state is a tuple: (stack, pos, in_str, esc, running, keyopen)
#   stack   : tuple of ('{', comma_seen) / ('[', comma_seen) markers
#   pos     : 'value' | 'key' | 'colon' | 'comma'
#   in_str  : inside a string literal
#   esc     : pending backslash
#   running : partial key text (for schema key completion)
#   keyopen : the open string is an object key
#
# advance(state, text) -> next state, or None when `text` can no longer be
# completed into valid JSON.

def _mk(stack=(), pos="value", in_str=False, esc=False, running="", keyopen=False):
    return (stack, pos, in_str, esc, running, keyopen)


_INIT = _mk()


def advance(state, text: str):
    stack, pos, in_str, esc, running, keyopen = state
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]

        if in_str:
            if esc:
                if ch == "u":
                    hex_part = text[i + 1:i + 5]
                    if len(hex_part) < 4:
                        # still typing \uXXXX (or bad char before the 4 hex)
                        if hex_part and any(c not in _HEX for c in hex_part):
                            return None
                        return _mk(stack, pos, True, False, running, keyopen)
                    if any(c not in _HEX for c in hex_part):
                        return None
                    i += 5
                elif ch in '"\\/bfnrt':
                    i += 1
                else:
                    return None
                esc = False
                continue
            if ch == "\\":
                esc = True
                i += 1
                continue
            if ch == '"':
                in_str = False
                i += 1
                if keyopen:
                    pos = "colon"
                    running = ""
                else:
                    pos = "comma"
                    running = ""
                keyopen = False
                continue
            if ord(ch) < 0x20:
                return None
            if keyopen:
                running += ch
            i += 1
            continue

        if ch in _WS:
            i += 1
            continue

        if ch == "{":
            if pos != "value":
                return None
            stack = stack + (("{", False),)
            pos = "key"
            running = ""
            i += 1
            continue
        if ch == "[":
            if pos != "value":
                return None
            stack = stack + (("[", False),)
            pos = "value"
            i += 1
            continue
        if ch == "}":
            if pos == "comma" or (pos == "key" and stack and stack[-1][0] == "{" and not stack[-1][1]):
                if not stack or stack[-1][0] != "{":
                    return None
                stack = stack[:-1]
                pos = "comma"
                i += 1
                continue
            return None
        if ch == "]":
            if pos == "comma" or (pos == "value" and stack and stack[-1][0] == "[" and not stack[-1][1]):
                if not stack or stack[-1][0] != "[":
                    return None
                stack = stack[:-1]
                pos = "comma"
                i += 1
                continue
            return None
        if ch == ":":
            if pos != "colon":
                return None
            pos = "value"
            i += 1
            continue
        if ch == ",":
            if pos != "comma":
                return None
            if stack:
                tag, seen = stack[-1]
                stack = stack[:-1] + ((tag, True),)
            pos = "key" if (stack and stack[-1][0] == "{") else "value"
            running = "" if pos == "key" else running
            i += 1
            continue

        # object key position
        if pos == "key":
            if stack and stack[-1][0] == "{":
                if ch == '"':
                    in_str = True
                    keyopen = True
                    running = ""
                    i += 1
                    continue
            return None

        if ch == '"':
            if pos != "value":
                return None
            in_str = True
            keyopen = False
            running = ""
            i += 1
            continue

        if pos != "value":
            return None

        # number
        if ch in _DIGITS or ch == "-":
            j = i
            if ch == "-":
                j += 1
            seen_dot = False
            seen_e = False
            while j < n:
                c = text[j]
                if c in _DIGITS:
                    j += 1
                elif c == "." and not seen_dot and not seen_e:
                    seen_dot = True
                    j += 1
                elif c in "eE" and not seen_e and j > i:
                    seen_e = True
                    j += 1
                elif c in "+-" and seen_e and j > i and text[j - 1] in "eE":
                    j += 1
                else:
                    break
            if j < n:
                if text[j] in _WS or text[j] in ",}]":
                    return advance(_mk(stack, "comma", in_str, esc, running, keyopen), text[j:])
                return None
            return _mk(stack, "comma", in_str, esc, running, keyopen)

        # keyword: true / false / null
        ate = False
        for kw in _KEYWORDS:
            if text[i:].startswith(kw):
                return advance(_mk(stack, "comma", in_str, esc, running, keyopen), text[i + len(kw):])
            if kw.startswith(text[i:]):
                return _mk(stack, pos, in_str, esc, running, keyopen)
        if not ate:
            return None
    return _mk(stack, pos, in_str, esc, running, keyopen)


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class JsonModeConstraint:
    """
    Constrain generation to a valid JSON document (optionally restricted to a
    set of allowed keys at each object level).

    `tokenizer` must provide `decode_single(token_id) -> str`.
    `schema` is a dict template; at each object depth only its keys are
    permitted. Nested arrays use the "_" key as the item schema.
    """

    def __init__(self, tokenizer, schema: Optional[dict] = None):
        self.tokenizer = tokenizer
        self.schema = schema if schema is not None else {}
        self.state = _INIT

    def reset(self):
        self.state = _INIT

    def _schema_node(self, stack):
        node = self.schema
        nstack = len(stack)
        for idx, (tag, _seen) in enumerate(stack):
            if not isinstance(node, dict):
                node = {}
                continue
            if (nstack - 1 - idx) >= 1:
                # deeper than the schema root's own children -> lenient
                node = {}
                continue
            if tag == "[":
                inner = node.get("_")
                node = inner if isinstance(inner, dict) else {}
        return node if isinstance(node, dict) else {}

    def _probe(self, piece: str) -> bool:
        base = self.state
        before_key = base[4]
        was_open_key = bool(base[2] and base[5])
        st = advance(base, piece)
        if st is None:
            return False
        if not self.schema:
            return True

        keys = {k for k in self._schema_node(base[0]) if isinstance(k, str)}
        if not keys:
            return True

        now_open_key = bool(st[2] and st[5])
        if was_open_key and now_open_key:
            # the key string is still open: it must stay a prefix of a key
            trial = st[4]
            return any(k.startswith(trial) for k in keys)
        if was_open_key:
            # the key string closed during this piece: the completed key
            # (before_key + the part of `piece` up to the closing quote)
            # must be one of the allowed keys.
            closed = before_key + piece.split('"')[0]
            return closed in keys
        # a fresh key may have opened AND closed entirely within this piece
        head = piece.lstrip(_WS)
        if base[1] == "key" and head.startswith('"'):
            inner = head[1:].split('"')[0]
            return inner in keys
        return True

    def allowed(self, candidate_ids: Sequence[int]) -> set:
        out = set()
        seen = set()
        for tid in candidate_ids:
            if tid < 0 or int(tid) in seen:
                continue
            seen.add(int(tid))
            try:
                piece = self.tokenizer.decode_single(int(tid))
            except Exception:
                continue
            if piece is None or piece == "":
                continue
            if self._probe(piece):
                out.add(int(tid))
        return out

    def step(self, piece: str):
        self.state = advance(self.state, piece)
        if self.state is None:
            self.state = _INIT
            raise ValueError(f"constrained state diverged at {piece!r}")

    @property
    def doc_complete(self) -> bool:
        """True when the top-level JSON value has been closed and the machine
        is not inside a string/number — i.e. the model may stop here."""
        s = self.state
        return bool(not s[0] and s[1] == "comma" and not s[2] and not s[3])


class SequenceConstraint:
    """Force generation to begin with a required text prefix (piece by piece)."""

    def __init__(self, tokenizer, required: str):
        self.tokenizer = tokenizer
        self.required = required
        self.buffer = ""

    def allowed(self, candidate_ids: Sequence[int]) -> set:
        out = set()
        need = self.required[len(self.buffer):]
        for tid in candidate_ids:
            try:
                piece = self.tokenizer.decode_single(int(tid))
            except Exception:
                continue
            if piece is None:
                continue
            if need.startswith(piece):
                out.add(int(tid))
        return out

    def step(self, piece: str):
        self.buffer += piece


class ConstraintSession:
    """
    Convenience wrapper used by the GUI / link generate loops:

        con = ConstraintSession(tokenizer, kind="json", schema=...)
        ...
        mask = con.allowed(top_candidate_ids)
        tok = sampler.sample(logits, mask=mask)
        con.step(decode_single(tok))
    """

    def __init__(self, tokenizer, kind: str = "json", schema: Optional[dict] = None,
                 required_prefix: str = ""):
        if kind in ("json", "json_schema"):
            self._impl = JsonModeConstraint(tokenizer, schema=schema)
        elif kind == "sequence":
            self._impl = SequenceConstraint(tokenizer, required_prefix)
        else:
            raise ValueError(f"unknown constraint kind: {kind}")
        self.kind = kind

    def allowed(self, candidate_ids: Sequence[int]) -> set:
        return self._impl.allowed(candidate_ids)

    def step(self, piece: str):
        self._impl.step(piece)

    def reset(self):
        self._impl.reset()

    @property
    def doc_complete(self) -> bool:
        return bool(getattr(self._impl, "doc_complete", False))