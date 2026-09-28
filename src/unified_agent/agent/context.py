"""Context assembly and compaction.

This is the piece the original spec was missing entirely, and it is the
single most common cause of "the agent worked for 8 steps and then died".
A long task produces tool output faster than any context window can hold;
without an explicit budget and a compaction policy, step 12 returns a 400
and the whole task is lost.

Design:

* The message list is **rebuilt from state each step**, not appended to.
  That is what makes compaction a pure function of state instead of a
  surgery on a growing list.
* Compaction is *proactive* (triggered by a token estimate crossing a
  threshold) rather than reactive (triggered by a provider error). By the
  time the provider errors you have already lost the turn.
* There is always a deterministic fallback. If no summariser model is
  available, compaction degrades to a structured digest rather than
  failing -- a task must not become uncompletable because a cheap model
  alias is missing.
"""

from __future__ import annotations

from typing import Any

from unified_agent.agent.prompts import (
    COMPACTION_PROMPT,
    SYSTEM_PROMPT,
    plan_block,
    workspace_block,
)
from unified_agent.agent.state import AgentState
from unified_agent.config import Settings
from unified_agent.models.base import ChatModel, estimate_tokens
from unified_agent.observability.events import EventType
from unified_agent.types import Message


class ContextBuilder:
    def __init__(
        self,
        *,
        settings: Settings,
        model: ChatModel,
        tool_catalog: str,
        skills_index: str = "",
        store: Any = None,
        summarizer: ChatModel | None = None,
    ) -> None:
        self.settings = settings
        self.model = model
        self.tool_catalog = tool_catalog
        self.skills_index = skills_index
        self.store = store
        self.summarizer = summarizer

    # -- budget -----------------------------------------------------------
    def budget(self) -> int:
        """Tokens available for the prompt.

        `max_tokens` in the agent config is a *cost* ceiling, so the prompt
        budget is the smaller of the model's window (minus output reserve)
        and that ceiling.
        """
        window = self.model.capabilities.max_context_tokens
        reserve = int(window * self.settings.agent.output_reserve_ratio)
        usable = window - reserve
        return max(2_000, min(usable, self.settings.agent.max_tokens))

    def _log_render_chars(self, budget: int) -> int:
        # ~3.6 chars/token for latin; assume the pessimistic 2.0 so CJK-heavy
        # logs do not blow the budget.
        return max(4_000, int(budget * 0.55 * 2.0))

    # -- assembly ---------------------------------------------------------
    def build(self, state: AgentState, *, memory_block: str = "") -> list[Message]:
        budget = self.budget()
        base = SYSTEM_PROMPT + "\n\n" + workspace_block(
            workspace=str(self.settings.workspace),
            tool_catalog=self.tool_catalog,
            skills_index=self.skills_index,
            memory_block=memory_block,
        )

        state_block_parts: list[str] = []
        if state.plan:
            state_block_parts.append(plan_block(state.plan, state.current_step))
        if state.compacted_summary:
            state_block_parts.append(
                "# Earlier work (compacted)\n\n" + state.compacted_summary
            )
        called = [e.tool for e in state.log if e.tool]
        if called:
            # Cheap redundancy guard. Without it, models happily re-list the
            # same directory four times, and every one of those costs a turn.
            counts: dict[str, int] = {}
            for name in called:
                counts[name] = counts.get(name, 0) + 1
            state_block_parts.append(
                "# Tools already called\n\n"
                + ", ".join(f"{name} x{n}" for name, n in counts.items())
                + "\n\nDo not repeat a call whose result you already have."
            )
        if state.approved_effects:
            state_block_parts.append(
                "# Pre-approved effects\n\n"
                + ", ".join(state.approved_effects)
                + " — calls of these effects will run without asking."
            )

        messages: list[Message] = [Message(role="system", content=base, pinned=True)]
        if state_block_parts:
            messages.append(
                Message(role="system", content="\n\n".join(state_block_parts), pinned=True)
            )
        messages.append(Message(role="user", content=f"# Goal\n\n{state.goal}", pinned=True))

        fixed = self.model.count_messages(messages)
        remaining = max(1_000, budget - fixed - 600)
        log_text = self._render_log(state, chars=self._log_render_chars(remaining))
        if log_text:
            messages.append(Message(role="user", content=log_text))
        messages.append(
            Message(
                role="user",
                content=(
                    "Continue. Call a tool, or write the final answer if you are done. "
                    "If you are blocked, say so and explain why."
                ),
            )
        )
        return messages

    def _render_log(self, state: AgentState, *, chars: int) -> str:
        """Two tiers, and the cheap one comes first.

        Recent entries are rendered in full -- they are what the model is
        working from. Entries older than `keep_recent_observations` keep only
        their head line: the tool, its arguments, whether it worked, and how
        much came back.

        That distinction is the whole point. Squeezing old entries to a
        smaller *body* still spends the tokens, and when the budget runs out
        the oldest entries are dropped outright -- so the model loses the
        record of what it already did and re-runs calls whose results it
        cannot see. A digest is a few dozen characters, so the entire action
        history fits, and the fact the model actually needs ("I already read
        app.py and it worked") survives at no cost.
        """
        if not state.log:
            return ""
        header = "# Progress so far\n"
        budget = max(1_000, chars - len(header))
        keep = self.settings.agent.keep_recent_observations

        chunks: list[str] = []
        remaining = budget
        for i, entry in enumerate(reversed(state.log)):
            if i < keep:
                rendered = entry.render(max_chars=max(400, int(budget * 0.7)))
            else:
                rendered = entry.digest()
            if len(rendered) > remaining:
                rendered = rendered[: max(200, remaining)]
            remaining -= len(rendered)
            chunks.append(rendered)
            if remaining <= 0:
                omitted = len(state.log) - len(chunks)
                chunks.append(f"[... {omitted} earlier entries omitted from context]")
                break
        return header + "\n\n" + "\n\n".join(reversed(chunks))

    def estimate(self, state: AgentState) -> int:
        return self.model.count_messages(self.build(state))

    # -- compaction -------------------------------------------------------
    def should_compact(self, state: AgentState) -> bool:
        """Whether to pay for the LLM summary.

        This triggers on *state growth*, not on whether the prompt fits --
        the render already guarantees the prompt fits, for free. What the
        summary buys is a smaller durable state: the log is folded into
        `compacted_summary` and dropped, so `replay` stays fast and the event
        stream stays bounded over a long task.

        Measuring the raw log is therefore correct here, even though the
        rendered form is much smaller. Measuring the rendered form would mean
        never compacting, and a state that grows without limit.
        """
        keep = self.settings.agent.keep_recent_observations
        if len(state.log) <= keep:
            return False
        log_tokens = sum(estimate_tokens(e.render(max_chars=20_000)) for e in state.log)
        return log_tokens > self.budget() * 0.5

    async def maybe_compact(self, state: AgentState) -> bool:
        if not self.should_compact(state):
            return False
        keep = self.settings.agent.keep_recent_observations
        dropped = state.log[:-keep]
        if not dropped:
            return False
        summary = await self._summarize(dropped, state.compacted_summary)
        state.compacted_summary = summary
        state.log_offset += len(dropped)
        state.log = state.log[len(dropped) :]
        if self.store is not None:
            self.store.append(
                state.task_id,
                EventType.CONTEXT_COMPACTED,
                {
                    "kept_from": len(dropped),
                    "summary": summary,
                    "dropped_entries": len(dropped),
                },
            )
        return True

    async def _summarize(self, dropped: list, previous: str | None) -> str:
        digest = _digest(dropped)
        if self.summarizer is None:
            return _merge(previous, digest)
        transcript = "\n\n".join(e.render(max_chars=2_000) for e in dropped)
        if previous:
            transcript = f"Previous summary:\n{previous}\n\nNew entries:\n{transcript}"
        try:
            resp = await self.summarizer.chat(
                [
                    Message(role="system", content=COMPACTION_PROMPT),
                    Message(role="user", content=transcript[:60_000]),
                ]
            )
        except Exception:  # noqa: BLE001 - compaction must never fail a task
            return _merge(previous, digest)
        text = (resp.content or "").strip()
        if not text or len(text) < 20:
            return _merge(previous, digest)
        return _merge(previous, text)


def _digest(entries: list) -> str:
    """Deterministic fallback summary. No model required."""
    lines = [f"{len(entries)} earlier step(s):"]
    tools: dict[str, int] = {}
    failures: list[str] = []
    paths: set[str] = set()
    for entry in entries:
        if entry.tool:
            tools[entry.tool] = tools.get(entry.tool, 0) + 1
            if not entry.success:
                first_line = (entry.text or "").strip().splitlines()
                failures.append(f"{entry.tool}: {first_line[0][:160] if first_line else 'failed'}")
        for token in _path_tokens(entry.text or ""):
            paths.add(token)
    lines.append("tools used: " + ", ".join(f"{k} x{v}" for k, v in sorted(tools.items())))
    if paths:
        lines.append("paths touched: " + ", ".join(sorted(paths)[:25]))
    if failures:
        lines.append("failures: " + " | ".join(failures[:8]))
    return "\n".join(lines)


def _path_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"[\w./\-]+\.(?:py|md|json|toml|txt|js|ts|tsx|yaml|yml|go|rs|sql|css|html)", text)[:10]


def _merge(previous: str | None, new: str) -> str:
    if not previous:
        return new
    return f"{previous}\n\n---\n{new}"
