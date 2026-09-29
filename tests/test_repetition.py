"""The repetition guard.

The behaviour under test came from a real run: 60 steps, 515k tokens, 120
tool calls of which 38 were distinct, `orders/calc.py` read 16 times, and the
goal was "run the tests and fix what fails" -- which ended with nothing
written. The system prompt already asked the model not to repeat itself and
it did anyway, so these tests are about the guarantee, not the request.
"""

from __future__ import annotations

import pytest

from unified_agent.agent import repetition
from unified_agent.agent.context import ContextBuilder
from unified_agent.agent.repetition import call_signature, result_digest
from unified_agent.agent.state import AgentState, LogEntry, TaskStatus, replay
from unified_agent.observability.events import EventType


# ---------------------------------------------------------------------------
# signature
# ---------------------------------------------------------------------------


def test_signature_ignores_key_order():
    """Otherwise the same call looks like two different calls."""
    assert call_signature("read_file", {"a": 1, "b": 2}) == call_signature(
        "read_file", {"b": 2, "a": 1}
    )


def test_signature_separates_tools_and_arguments():
    base = call_signature("read_file", {"path": "a.py"})
    assert base != call_signature("read_file", {"path": "b.py"})
    assert base != call_signature("file_info", {"path": "a.py"})


def test_signature_handles_missing_arguments():
    assert call_signature("git_status", None) == call_signature("git_status", {})


def test_result_digest_tracks_content():
    assert result_digest("hello") == result_digest("hello")
    assert result_digest("hello") != result_digest("hello ")
    assert result_digest(None) == result_digest("")


# ---------------------------------------------------------------------------
# policy: strict on reads, loud but permissive on everything else
# ---------------------------------------------------------------------------


def test_first_call_is_silent():
    verdict = repetition.check(tool="read_file", count=0, stop_at=3)
    assert not verdict.blocked
    assert not verdict.annotate


def test_second_call_is_annotated_not_blocked():
    verdict = repetition.check(tool="read_file", count=1, stop_at=3)
    assert verdict.annotate
    assert not verdict.blocked
    assert verdict.count == 2


def test_pure_read_is_refused_past_the_limit():
    verdict = repetition.check(tool="read_file", count=3, stop_at=3)
    assert verdict.blocked
    assert "REFUSED" in verdict.note
    # The message has to tell the model what to do instead, or it will just
    # try the same call again.
    assert "edit the file" in verdict.note


def test_non_read_is_never_blocked():
    """`run_tests({})` before and after an edit is a fix loop, not a bug."""
    verdict = repetition.check(tool="run_tests", count=9, stop_at=3)
    assert not verdict.blocked
    assert verdict.annotate


def test_write_tools_are_not_treated_as_pure_reads():
    for tool in ("write_file", "apply_patch", "run_command", "delete_file"):
        assert not repetition.is_pure_read(tool), tool


def test_zero_limit_disables_the_refusal_but_keeps_the_warning():
    verdict = repetition.check(tool="read_file", count=50, stop_at=0)
    assert not verdict.blocked
    assert verdict.annotate


# ---------------------------------------------------------------------------
# state folding
# ---------------------------------------------------------------------------


def _log_event(seq: int, tool: str, arguments: dict, text: str = "body", success: bool = True):
    entry = LogEntry(
        index=seq, step_id="step_1", kind="tool", tool=tool, arguments=arguments,
        success=success, text=text,
    )
    return type(
        "E",
        (),
        {"seq": seq, "type": EventType.LOG_APPENDED, "payload": {"entry": entry.model_dump(mode="json")}},
    )()


def test_replay_counts_identical_calls():
    events = [_log_event(i, "read_file", {"path": "a.py"}) for i in range(1, 5)]
    events.append(_log_event(5, "read_file", {"path": "b.py"}))
    state = replay(events, task_id="t1")
    assert state.call_counts[call_signature("read_file", {"path": "a.py"})] == 4
    assert state.call_counts[call_signature("read_file", {"path": "b.py"})] == 1


def test_system_entries_do_not_count():
    """A refused repeat is logged as `system`; counting it would make the
    refusal escalate itself on every retry."""
    entries = [_log_event(1, "read_file", {"path": "a.py"})]
    system = LogEntry(index=2, kind="system", text="refused", success=False)
    entries.append(
        type("E", (), {"seq": 2, "type": EventType.LOG_APPENDED, "payload": {"entry": system.model_dump(mode="json")}})()
    )
    state = replay(entries, task_id="t1")
    assert state.call_counts[call_signature("read_file", {"path": "a.py"})] == 1


def test_counts_survive_compaction():
    """Compaction trims `state.log`. A counter derived from the log would
    forget exactly the repeats it exists to catch."""
    events = [_log_event(i, "read_file", {"path": "a.py"}) for i in range(1, 5)]
    events.append(
        type(
            "E",
            (),
            {
                "seq": 5,
                "type": EventType.CONTEXT_COMPACTED,
                "payload": {"kept_from": 4, "summary": "s", "dropped_entries": 4},
            },
        )()
    )
    state = replay(events, task_id="t1")
    assert len(state.log) == 0  # the log really was trimmed
    assert state.call_counts[call_signature("read_file", {"path": "a.py"})] == 4


# ---------------------------------------------------------------------------
# the loop, end to end
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_repeated_reads_are_refused_in_a_real_run(scripted, settings, session_id):
    """Five identical reads: three run, the rest are refused, and the task
    still finishes rather than burning the budget."""
    script = [
        {"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]}
        for _ in range(5)
    ]
    script.append({"content": "done"})
    agent, _ = await scripted(script)

    result = await agent.runtime.run("read app.py repeatedly", session_id=session_id)
    task_id = result.task_id

    calls = agent.store.list_tool_calls(task_id, limit=100)
    reads = [c for c in calls if c.name == "read_file"]
    # 3 executed (limit), the other 2 declined before touching the disk.
    assert len(reads) == 3, [c.name for c in calls]

    events = agent.store.events(task_id)
    repeated = [e for e in events if e.type is EventType.TOOL_REPEATED]
    # The escalation is deliberate: calls 2 and 3 run but carry a warning in
    # their observation, calls 4 and 5 are refused before touching the disk.
    annotated = [e for e in repeated if e.payload["action"] == "annotated"]
    refused = [e for e in repeated if e.payload["action"] == "refused"]
    assert [e.payload["count"] for e in annotated] == [2, 3]
    # Both refusals report count 4, not 4 then 5: a refused call is logged as
    # `kind="system"` and does not increment, so it cannot raise its own count
    # and escalate itself further on every retry.
    assert [e.payload["count"] for e in refused] == [4, 4]
    assert result.status == TaskStatus.COMPLETED.value


@pytest.mark.asyncio
async def test_repeat_warning_reaches_the_model(scripted, settings, session_id):
    """The annotation has to be in the observation the model reads, not in a
    side channel it never sees."""
    script = [
        {"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]}
        for _ in range(2)
    ]
    script.append({"content": "done"})
    agent, _ = await scripted(script)

    result = await agent.runtime.run("read app.py twice", session_id=session_id)
    events = agent.store.events(result.task_id)

    annotated = [e for e in events if e.type is EventType.TOOL_REPEATED]
    assert len(annotated) == 1
    assert annotated[0].payload["action"] == "annotated"

    tool_entries = [
        e.payload["entry"]
        for e in events
        if e.type is EventType.LOG_APPENDED and e.payload["entry"]["kind"] == "tool"
    ]
    assert "2nd time" in tool_entries[-1]["text"]


@pytest.mark.asyncio
async def test_budget_failure_names_the_loop(scripted, settings, session_id):
    """`steps budget exhausted` is true and useless. When the cause was a
    loop, the failure has to say so."""
    script = [
        {"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]}
        for _ in range(12)
    ]
    agent, _ = await scripted(script)
    result = await agent.runtime.run("loop on the same read", session_id=session_id)

    assert result.status == TaskStatus.FAILED.value
    assert "budget exhausted" in result.error
    assert "exact repeats" in result.error
    assert "app.py" in result.error, "the failure should name the call that looped"


@pytest.mark.asyncio
async def test_non_read_repeats_still_execute(scripted, settings, session_id):
    """The guard must not break a legitimate re-run."""
    script = [
        {"tool_calls": [{"name": "write_file", "arguments": {"path": "out.txt", "content": "x"}}]}
        for _ in range(4)
    ]
    script.append({"content": "done"})
    agent, _ = await scripted(script)
    result = await agent.runtime.run("write the same file repeatedly", session_id=session_id)

    calls = agent.store.list_tool_calls(result.task_id, limit=100)
    writes = [c for c in calls if c.name == "write_file"]
    assert len(writes) == 4, "writes must never be silently swallowed"


# ---------------------------------------------------------------------------
# context: the warning the model actually reads
# ---------------------------------------------------------------------------


class _StubModel:
    capabilities = type("C", (), {"max_context_tokens": 100_000})()

    def count_messages(self, messages):
        return sum(len(m.content or "") for m in messages) // 4


def _builder(settings):
    return ContextBuilder(
        settings=settings, model=_StubModel(), tool_catalog="tools", skills_index=""
    )


def _state_with(log_entries):
    state = AgentState(task_id="t", session_id="s", goal="g")
    for i, entry in enumerate(log_entries):
        entry.index = i
        state.append_log(entry)
        state.call_counts[call_signature(entry.tool, entry.arguments)] = (
            state.call_counts.get(call_signature(entry.tool, entry.arguments), 0) + 1
        )
    return state


def test_repeat_block_names_the_call_not_just_the_tool(settings):
    """The old block said `read_file x16`, which the model could not act on."""
    state = _state_with(
        [
            LogEntry(index=0, kind="tool", tool="read_file", arguments={"path": "orders/calc.py"}, text="x"),
            LogEntry(index=1, kind="tool", tool="read_file", arguments={"path": "orders/calc.py"}, text="x"),
        ]
    )
    block = _builder(settings)._repeat_block(state)
    assert "read_file" in block
    assert "orders/calc.py" in block
    assert "x2" in block
    assert "REPEATED" in block


def test_repeat_block_lists_single_calls_without_crying_loop(settings):
    state = _state_with(
        [LogEntry(index=0, kind="tool", tool="read_file", arguments={"path": "a.py"}, text="x")]
    )
    block = _builder(settings)._repeat_block(state)
    assert "REPEATED" not in block
    assert "Do not repeat a call" in block


def test_identical_repeat_is_elided_from_the_prompt(settings):
    """The log keeps both copies; the prompt shows the second as a pointer."""
    state = _state_with(
        [
            LogEntry(index=0, kind="tool", tool="read_file", arguments={"path": "a.py"}, text="BODY" * 200),
            LogEntry(index=1, kind="tool", tool="read_file", arguments={"path": "a.py"}, text="BODY" * 200),
        ]
    )
    rendered = _builder(settings)._render_log(state, chars=40_000)
    assert rendered.count("BODY") < 200 * 2
    assert "not repeated" in rendered


def test_changed_content_is_not_elided(settings):
    """Suppression is only safe when the bytes are identical."""
    state = _state_with(
        [
            LogEntry(index=0, kind="tool", tool="read_file", arguments={"path": "a.py"}, text="OLD"),
            LogEntry(index=1, kind="tool", tool="read_file", arguments={"path": "a.py"}, text="NEW"),
        ]
    )
    rendered = _builder(settings)._render_log(state, chars=40_000)
    assert "OLD" in rendered and "NEW" in rendered
    assert "not repeated" not in rendered


def test_failed_repeat_is_not_elided(settings):
    """An error that repeats may still have changed; that difference is the
    signal the model is looking for."""
    state = _state_with(
        [
            LogEntry(index=0, kind="tool", tool="run_command", arguments={"command": "x"}, text="boom", success=False),
            LogEntry(index=1, kind="tool", tool="run_command", arguments={"command": "x"}, text="boom", success=False),
        ]
    )
    rendered = _builder(settings)._render_log(state, chars=40_000)
    assert rendered.count("boom") == 2
