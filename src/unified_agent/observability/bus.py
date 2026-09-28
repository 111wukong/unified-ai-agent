"""In-process event bus.

Exists because two consumers need the *same* stream of facts for different
reasons, and conflating them would be wrong:

* **durable events** go to SQLite and the JSONL log. They are the source of
  truth and are replayed on resume.
* **ephemeral deltas** (per-token text) go only to live subscribers. Writing
  every token to SQLite would multiply the database size for data that is
  worthless after the turn ends -- the final `MODEL_RESPONSE` event already
  contains the complete text.

So the bus carries both, tagged, and the durable half is still emitted by
`Store.append` (which is why `publish` is synchronous: it is called from
inside the store's write path and must not block on a slow UI).

Backpressure policy: a bounded queue per subscriber, and on overflow the
**oldest** item is dropped rather than blocking the agent. A slow browser
tab must never stall a task. Dropped items are counted and surfaced as a
`CUSTOM` event so the client knows to re-sync from a snapshot instead of
silently rendering a truncated stream.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal

from unified_agent.observability.events import Event

DEFAULT_QUEUE_SIZE = 1024


@dataclass
class BusItem:
    """Either a durable event or an ephemeral stream delta."""

    kind: Literal["event", "text_delta", "notice"]
    task_id: str
    event: Event | None = None
    text: str = ""
    message_id: str = ""
    notice: str = ""
    dropped: int = 0

    @property
    def is_durable(self) -> bool:
        return self.kind == "event"


@dataclass(eq=False)
class _Subscriber:
    """`eq=False` on purpose: subscribers are identity-keyed, and the default
    dataclass `__eq__` sets `__hash__ = None`, which makes them unhashable
    and breaks the `set` they live in."""

    queue: asyncio.Queue[BusItem] = field(default_factory=lambda: asyncio.Queue(DEFAULT_QUEUE_SIZE))
    dropped: int = 0

    def offer(self, item: BusItem) -> None:
        try:
            self.queue.put_nowait(item)
        except asyncio.QueueFull:
            # Drop the oldest and keep going. Losing an early token is
            # recoverable (the final message event has the full text);
            # blocking the agent loop is not.
            with contextlib.suppress(asyncio.QueueEmpty):
                self.queue.get_nowait()
            self.dropped += 1
            with contextlib.suppress(asyncio.QueueFull):
                self.queue.put_nowait(
                    BusItem(
                        kind="notice",
                        task_id=item.task_id,
                        notice="stream fell behind; re-sync from a snapshot",
                        dropped=self.dropped,
                    )
                )


class Subscription:
    """Async iterator over one task's stream. Use as a context manager."""

    def __init__(self, bus: "EventBus", task_id: str) -> None:
        self.bus = bus
        self.task_id = task_id
        self._sub = _Subscriber()

    async def __aenter__(self) -> "Subscription":
        self.bus._subscribers[self.task_id].add(self._sub)
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        subs = self.bus._subscribers.get(self.task_id)
        if subs is not None:
            subs.discard(self._sub)
            if not subs:
                self.bus._subscribers.pop(self.task_id, None)

    async def __aiter__(self) -> AsyncIterator[BusItem]:
        while True:
            item = await self._sub.queue.get()
            if item.kind == "notice" and item.notice == "__close__":
                return
            yield item

    async def get(self, timeout: float | None = None) -> BusItem | None:
        try:
            return await asyncio.wait_for(self._sub.queue.get(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

    def pending(self) -> int:
        """Items waiting to be consumed. Lets a consumer tell "nothing more
        is coming" from "the producer is thinking" without a long timeout."""
        return self._sub.queue.qsize()


class EventBus:
    """Fan-out for one process. Multi-process streaming would need Redis;
    a local-first tool does not have that problem yet."""

    def __init__(self) -> None:
        self._subscribers: dict[str, set[_Subscriber]] = defaultdict(set)

    # -- publisher side (called from sync code) ---------------------------
    def publish(self, event: Event) -> None:
        subs = self._subscribers.get(event.task_id)
        if not subs:
            return
        item = BusItem(kind="event", task_id=event.task_id, event=event)
        for sub in tuple(subs):
            sub.offer(item)

    def emit(self, event: Event) -> None:
        """Alias so the bus can sit in a sink chain.

        `BusSink` fans out to objects with `.emit()`. Without this alias the
        bus is called as `emit`, raises AttributeError, and -- because sink
        errors are swallowed so they cannot break a task -- the bus silently
        receives *nothing*. That is exactly the failure this alias prevents.
        """
        self.publish(event)

    def publish_delta(self, task_id: str, *, message_id: str, text: str) -> None:
        subs = self._subscribers.get(task_id)
        if not subs:
            return
        item = BusItem(kind="text_delta", task_id=task_id, message_id=message_id, text=text)
        for sub in tuple(subs):
            sub.offer(item)

    # -- subscriber side --------------------------------------------------
    def subscribe(self, task_id: str) -> Subscription:
        return Subscription(self, task_id)

    def has_subscribers(self, task_id: str) -> bool:
        return bool(self._subscribers.get(task_id))

    def subscriber_count(self, task_id: str) -> int:
        return len(self._subscribers.get(task_id) or ())

    def close_task(self, task_id: str) -> None:
        for sub in tuple(self._subscribers.pop(task_id, ())):
            with contextlib.suppress(asyncio.QueueFull):
                sub.queue.put_nowait(
                    BusItem(kind="notice", task_id=task_id, notice="__close__")
                )


class BusSink:
    """Adapter so `Store(sink=...)` can fan out to both JSONL and the bus.

    `Store.append` already takes one sink; rather than teach it about the
    bus, compose the sinks.

    Failures are recorded in `problems` rather than swallowed. Swallowing
    them is how a broken sink stays broken for months while every test
    passes -- the events just never arrive, and nothing says so.
    """

    def __init__(self, *sinks: object) -> None:
        self.sinks = sinks
        self.problems: list[str] = []

    def emit(self, event: Event) -> None:
        for sink in self.sinks:
            if sink is None:
                continue
            handler = getattr(sink, "emit", None)
            if handler is None:
                self._record(f"{type(sink).__name__} has no emit()")
                continue
            try:
                handler(event)
            except Exception as exc:  # noqa: BLE001 - observability must not break a task
                self._record(f"{type(sink).__name__}.emit failed: {type(exc).__name__}: {exc}")

    def _record(self, message: str) -> None:
        if message not in self.problems:
            self.problems.append(message)

    def close(self) -> None:
        for sink in self.sinks:
            close = getattr(sink, "close", None)
            if close:
                close()


__all__ = ["EventBus", "Subscription", "BusItem", "BusSink"]
