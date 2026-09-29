#!/usr/bin/env python3
"""Audit how a task actually spent its budget.

Why this exists
---------------
A task can end in `completed` and still have been useless, and it can end in
`failed` for a reason that has nothing to do with the goal. Reading the final
status tells you nothing about either. This walks the event log and answers
the questions that status does not:

  * Did the agent ever *change* anything, or only read?
  * Did it repeat itself -- the same call, the same cycle?
  * Did it retry a call that had already failed, unchanged?
  * Did it stop because it was done, or because it ran out of steps?

Usage
-----
    python scripts/audit-task.py --home /tmp/uaa-live                 # every task
    python scripts/audit-task.py --home /tmp/uaa-live task_abc123     # one task
    python scripts/audit-task.py --home /tmp/uaa-live --json          # machine-readable

Exit code is 1 when any audited task shows a finding, so this can gate a run.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from unified_agent.config import load_settings  # noqa: E402
from unified_agent.storage.store import Store  # noqa: E402

#: Tools that cannot change the world. A run made only of these read the
#: repository and stopped -- it did not do the task. Used only as a hint;
#: classification comes from the ledger's `effect_class`, which is recorded
#: at call time and does not have to guess.
READ_ONLY_TOOLS = {
    "read_file",
    "list_directory",
    "file_info",
    "search_files",
    "grep",
    "git_status",
    "git_diff",
    "git_log",
    "run_tests",
    "run_linter",
    "web_fetch",
    "web_search",
    "recall",
    "search_memory",
    "list_memory",
}


def _canonical(arguments: dict) -> str:
    """Stable key for "the same call", ignoring key order."""
    return json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)


def _shorten(signature: str, limit: int = 60) -> str:
    """`read_file:{"path": "orders/calc.py"}` -> `read_file(orders/calc.py)`."""
    name, _, raw = signature.partition(":")
    try:
        args = json.loads(raw)
    except json.JSONDecodeError:
        return signature[:limit]
    inner = ", ".join(f"{k}={v}" for k, v in sorted(args.items()))
    out = f"{name}({inner})"
    return out if len(out) <= limit else out[: limit - 1] + "…"


def _ngrams(seq: list[str], n: int) -> Counter:
    return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))


def audit(store: Store, task_id: str) -> dict:
    calls = store.list_tool_calls(task_id, limit=10_000)
    task = store.get_task(task_id) or {}
    events = store.events(task_id, limit=100_000)

    terminal_error = None
    for event in reversed(events):
        payload = event.payload or {}
        if event.type in ("task_failed", "budget_exceeded"):
            terminal_error = payload.get("reason") or payload.get("error") or str(payload)[:300]
            break

    signatures = [f"{c.name}:{_canonical(c.arguments or {})}" for c in calls]

    # Repeated identical calls: same tool, same arguments, more than once.
    sig_counts = Counter(signatures)
    repeated = {s: n for s, n in sig_counts.items() if n > 1}
    wasted = sum(n - 1 for n in repeated.values())

    # Cycles, detected on the *signature* rather than the tool name. Reading
    # three different files in a row is a normal thing to do, and a name-based
    # check calls it a loop -- which is how this script produced a false
    # positive on a run that succeeded. Only the same call coming round again
    # is a loop.
    loop_len, loop_count = 0, 0
    for n in range(2, 13):
        grams = _ngrams(signatures, n)
        top = grams.most_common(1)
        if not top or top[0][1] < 2:
            break
        loop_len, loop_count = n, top[0][1]
    loop_pattern: list[str] = []
    if loop_len:
        grams = _ngrams(signatures, loop_len)
        loop_pattern = [_shorten(s) for s in grams.most_common(1)[0][0]]

    # Failures, and failures that were re-issued with the same arguments.
    failures = [c for c in calls if c.status == "failed"]
    failed_sigs = Counter(f"{c.name}:{_canonical(c.arguments or {})}" for c in failures)
    futile_retries = {s: n for s, n in failed_sigs.items() if n > 1}

    read_calls = [c for c in calls if c.name in READ_ONLY_TOOLS]
    # The ledger records the effect class at the moment of the call, so use
    # that rather than inferring it from the tool name. `run_command` is the
    # case that makes name-based classification wrong: `python -m pytest` and
    # `python fix.py` are the same tool and opposite kinds of act.
    by_effect = Counter(c.effect_class or "unknown" for c in calls)
    write_calls = [c for c in calls if (c.effect_class or "") == "write_local"]

    # A file read repeatedly is the clearest sign the agent is not retaining
    # what it read.
    file_reads = Counter(
        (c.arguments or {}).get("path")
        for c in calls
        if c.name in ("read_file", "file_info") and (c.arguments or {}).get("path")
    )
    hot_files = {p: n for p, n in file_reads.items() if n > 2}

    findings: list[str] = []
    notes: list[str] = []

    # "The model never acted" and "it only read" are only defects when the
    # task did not deliver. A completed task that answered a question without
    # touching a tool is the *correct* shape, and flagging it is how an audit
    # tool gets ignored. So the two are separated: a failure to act is a
    # finding, an answer without action is a note.
    answered = any(
        e.type == "task_completed" and (e.payload or {}).get("answer")
        for e in events
    )
    failed = task.get("status") == "failed"

    if not calls:
        if failed or not answered:
            findings.append("no tool calls at all -- the model never acted")
        else:
            notes.append("answered without using a tool (fine for a question)")
    if calls and not write_calls:
        if failed:
            findings.append(
                f"read-only run that failed: {len(read_calls)} calls, zero "
                f"write_local calls -- nothing was changed"
            )
        else:
            notes.append(
                f"no write_local calls ({len(read_calls)} read-only) -- it "
                f"answered rather than changed anything"
            )
    if loop_len >= 3 and loop_count >= 3:
        findings.append(
            f"loop: the {loop_len}-step cycle {' -> '.join(loop_pattern)} "
            f"repeated {loop_count}x"
        )
    elif loop_len >= 3 and loop_count == 2:
        # Twice is not a loop. Reading a set of files, editing them, then
        # reading them again to check is the normal shape of a fix -- flagging
        # it as a finding is how an audit tool trains its reader to ignore it.
        notes.append(
            f"the {loop_len}-step cycle {' -> '.join(loop_pattern)} ran twice "
            f"(normal if it was a verify pass)"
        )
    if wasted:
        findings.append(f"repeated calls: {wasted} redundant calls across {len(repeated)} signatures")
    if futile_retries:
        findings.append(
            f"unchanged retries after failure: {len(futile_retries)} call(s) re-issued verbatim"
        )
    if hot_files:
        worst = sorted(hot_files.items(), key=lambda kv: -kv[1])[:3]
        findings.append(
            "hot files: " + ", ".join(f"{p} x{n}" for p, n in worst)
        )
    if terminal_error and "budget" in str(terminal_error).lower():
        findings.append(f"stopped on budget, not on completion: {terminal_error}")
    if terminal_error and "exact repeats" in str(terminal_error):
        notes.append("the runtime named the loop in its failure message")

    return {
        "task_id": task_id,
        "status": task.get("status"),
        "goal": (task.get("goal") or "")[:80],
        "steps": task.get("steps"),
        "tokens": task.get("tokens"),
        "tool_calls": len(calls),
        "unique_calls": len(sig_counts),
        "read_only_calls": len(read_calls),
        "write_local_calls": len(write_calls),
        "by_effect": dict(by_effect),
        "failed_calls": len(failures),
        "redundant_calls": wasted,
        "loop_len": loop_len,
        "loop_repeats": loop_count,
        "loop_pattern": loop_pattern,
        "hot_files": hot_files,
        "terminal_error": terminal_error,
        "findings": findings,
        "notes": notes,
    }


def render(report: dict) -> str:
    lines = [
        f"=== {report['task_id']}  [{report['status']}]  {report['goal']}",
        f"    steps={report['steps']}  tokens={report['tokens']}  "
        f"calls={report['tool_calls']} (unique {report['unique_calls']})",
        f"    read-only={report['read_only_calls']}  "
        f"write_local={report['write_local_calls']}  failed={report['failed_calls']}  "
        f"redundant={report['redundant_calls']}",
        "    effects: "
        + ", ".join(f"{k}={v}" for k, v in sorted(report["by_effect"].items())),
    ]
    if report["loop_len"]:
        lines.append(
            f"    loop: {report['loop_len']}-step cycle x{report['loop_repeats']} "
            f"-> {' -> '.join(report['loop_pattern'])}"
        )
    if report["terminal_error"]:
        lines.append(f"    terminal: {report['terminal_error']}")
    if report["findings"]:
        lines.append("    findings:")
        lines.extend(f"      ! {f}" for f in report["findings"])
    else:
        lines.append("    findings: none")
    for note in report.get("notes", []):
        lines.append(f"      - {note}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_ids", nargs="*", help="task ids; default is all recent tasks")
    parser.add_argument("--home", default=None, help="config dir (default ~/.uaa)")
    parser.add_argument("--limit", type=int, default=20, help="how many tasks to audit")
    parser.add_argument("--json", action="store_true", help="emit JSON")
    args = parser.parse_args()

    settings = load_settings(home=Path(args.home) if args.home else None)
    if not settings.db_path.exists():
        print(f"no database at {settings.db_path}", file=sys.stderr)
        return 2

    store = Store.readonly(settings.db_path)
    try:
        if args.task_ids:
            task_ids = args.task_ids
        else:
            task_ids = [t["id"] for t in store.list_tasks(limit=args.limit)]
        reports = [audit(store, tid) for tid in task_ids]
    finally:
        store.close()

    if args.json:
        print(json.dumps(reports, ensure_ascii=False, indent=2, default=str))
    else:
        for report in reports:
            print(render(report))
            print()

    return 1 if any(r["findings"] for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
