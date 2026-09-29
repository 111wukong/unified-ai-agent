"""Noticing that the agent is going in circles, and doing something about it.

Why this exists
---------------
The system prompt already said "do not repeat the same failing call" and
"do not spend steps re-confirming things you have already established". A
real run ignored both: 60 steps, 515k tokens, 120 tool calls of which 38
were distinct, `orders/calc.py` read 16 times -- and not one byte written.
The task was "run the tests and fix what fails".

A prompt is a request. This module is the guarantee.

What it does
------------
Two tiers, and they are deliberately different because reads and writes are
not the same kind of thing:

* **Reads** (`read_file`, `list_directory`, `file_info`, ...) are pure. Asking
  for the same read again cannot produce new information *unless the file
  changed*, and if it did change the result digest differs and nothing is
  suppressed. So a repeated read is safe to (a) show as a pointer instead of
  the same body again, and then (b) refuse outright.

* **Anything else** -- writes, commands, network -- is only *annotated*. The
  same `run_tests({})` before and after an edit is the normal shape of a fix
  loop, not a bug, so refusing it would break real work. The annotation
  ("you have made this exact call 5 times") is what the model needs.

The asymmetry is the whole design: be strict where repetition is provably
useless, be loud but permissive where it might be legitimate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

#: Tools whose repeated invocation cannot yield anything new. Membership is
#: about *purity*, not about being cheap: `run_tests` is a read of the world
#: in spirit but its answer changes the moment you edit a file, so it is not
#: here.
PURE_READ_TOOLS = frozenset(
    {
        "read_file",
        "list_directory",
        "file_info",
        "search_files",
        "grep",
        "recall",
        "list_memory",
        "search_memory",
        "git_status",
        "git_diff",
        "git_log",
    }
)

#: Above this many identical calls the model is told, in the observation
#: itself, that it is repeating itself.
ANNOTATE_AT = 2


def call_signature(tool: str, arguments: dict[str, Any] | None) -> str:
    """A stable identity for "this exact call".

    Key order must not matter, so the dict is dumped sorted -- otherwise
    `{"a":1,"b":2}` and `{"b":2,"a":1}` look like two different calls and the
    repetition is invisible.
    """
    payload = json.dumps(
        {"tool": tool, "args": arguments or {}},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def result_digest(text: str | None) -> str:
    """Digest of what a call returned, for "same call, same answer" checks."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def is_pure_read(tool: str) -> bool:
    return tool in PURE_READ_TOOLS


@dataclass(frozen=True)
class RepeatVerdict:
    """What to do about a call, given how many times it has been made."""

    count: int
    #: Refuse to execute it at all.
    blocked: bool = False
    #: Tell the model, in the observation, that it is repeating itself.
    annotate: bool = False
    note: str = ""

    @property
    def repeated(self) -> bool:
        return self.count > 1


def check(
    *,
    tool: str,
    count: int,
    stop_at: int,
    pure_read: bool | None = None,
) -> RepeatVerdict:
    """Decide what a repeat of `tool` deserves.

    `count` is how many times this exact call has *already* been made in this
    task, so the first invocation arrives with `count == 0`.
    """
    pure = is_pure_read(tool) if pure_read is None else pure_read
    already = count + 1  # counting the call being made now

    if pure and stop_at > 0 and already > stop_at:
        return RepeatVerdict(
            count=already,
            blocked=True,
            note=(
                f"REFUSED: you have already made this exact call {count} time(s) "
                f"in this task, and it is a read -- asking again cannot return "
                f"anything you do not already have. The result is in your log "
                f"above. Stop re-reading and do something with it: edit the file, "
                f"or write your final answer saying what is still unknown."
            ),
        )

    if already >= ANNOTATE_AT:
        return RepeatVerdict(
            count=already,
            annotate=True,
            note=(
                f"NOTE: this is the {_ordinal(already)} time you have made this "
                f"exact call in this task. The earlier result is unchanged and is "
                f"already in your log. If you are looking for something specific, "
                f"say what it is and look somewhere else instead."
            ),
        )

    return RepeatVerdict(count=already)


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"
