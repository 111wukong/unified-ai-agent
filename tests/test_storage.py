"""Durability primitives: event store, ledger, artifacts, memory search."""

from __future__ import annotations

import pytest

from wukong.observability.events import EventType
from wukong.observability.redact import Redactor
from wukong.storage.store import Store


@pytest.fixture
def store(tmp_path):  # noqa: ANN001
    s = Store(tmp_path / "wukong.db")
    yield s
    s.close()


class TestEventLog:
    def test_sequence_is_gapless_per_task(self, store: Store) -> None:
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        a = store.create_task(session_id=sid, goal="a")
        b = store.create_task(session_id=sid, goal="b")

        for _ in range(3):
            store.append(a, EventType.STATE_TRANSITION, {"from": "x", "to": "y"})
        for _ in range(2):
            store.append(b, EventType.STATE_TRANSITION, {"from": "x", "to": "y"})

        assert [e.seq for e in store.events(a)] == [1, 2, 3, 4]
        assert [e.seq for e in store.events(b)] == [1, 2, 3]
        assert store.last_seq(a) == 4

    def test_events_can_be_read_incrementally(self, store: Store) -> None:
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        task = store.create_task(session_id=sid, goal="g")
        for i in range(5):
            store.append(task, EventType.MODEL_RESPONSE, {"i": i})
        # TASK_CREATED is seq 1, so the five appends are seqs 2..6.
        assert [e.seq for e in store.events(task, since_seq=3)] == [4, 5, 6]


class TestSessionTurns:
    """A session is a conversation, not a list of unrelated runs.

    Without this the follow-up request in a thread has no idea what was
    already asked, so "expand on the second point" has no referent.
    """

    @staticmethod
    def _finish(store: Store, task_id: str, answer: str) -> None:
        store.save_projection(
            task_id,
            status="completed",
            state={},
            steps_used=1,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            result={"answer": answer},
        )

    def test_a_thread_comes_back_oldest_first(self, store: Store) -> None:
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        first = store.create_task(session_id=sid, goal="第一个问题")
        second = store.create_task(session_id=sid, goal="第二个问题")
        self._finish(store, first, "第一个答案")
        self._finish(store, second, "第二个答案")

        turns = store.session_turns(sid)

        # Oldest first: a conversation reads forwards, and the model needs the
        # order to make sense of "the second point".
        assert [t["goal"] for t in turns] == ["第一个问题", "第二个问题"]
        assert [t["answer"] for t in turns] == ["第一个答案", "第二个答案"]

    def test_an_unanswered_task_is_left_out(self, store: Store) -> None:
        """A question with no answer reads as a debt the model still owes."""
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        answered = store.create_task(session_id=sid, goal="答过的")
        failed = store.create_task(session_id=sid, goal="失败的")
        self._finish(store, answered, "答案")
        store.save_projection(
            failed,
            status="failed",
            state={},
            steps_used=1,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            error="boom",
        )

        assert [t["goal"] for t in store.session_turns(sid)] == ["答过的"]

    def test_another_session_is_a_different_conversation(self, store: Store) -> None:
        a = store.create_session(name="a", working_dir="/tmp", model_alias="mock")
        b = store.create_session(name="b", working_dir="/tmp", model_alias="mock")
        task_a = store.create_task(session_id=a, goal="A 的问题")
        task_b = store.create_task(session_id=b, goal="B 的问题")
        self._finish(store, task_a, "A 的答案")
        self._finish(store, task_b, "B 的答案")

        assert [t["goal"] for t in store.session_turns(a)] == ["A 的问题"]
        assert [t["goal"] for t in store.session_turns(b)] == ["B 的问题"]


class TestTaskListCarriesTheParentLink:
    """The console draws a fan-out as a tree, and it can only do that if the
    list says who spawned whom. A sub-agent shown as a peer of its parent
    makes "fan out to five" read as six unrelated runs."""

    def test_list_tasks_reports_the_parent(self, store: Store) -> None:
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        parent = store.create_task(session_id=sid, goal="扇出")
        child = store.create_task(session_id=sid, goal="子任务", parent_task_id=parent)

        rows = {row["id"]: row for row in store.list_tasks(session_id=sid)}

        assert rows[child]["parent_task_id"] == parent
        assert rows[parent]["parent_task_id"] is None

    def test_list_children_returns_them_oldest_first(self, store: Store) -> None:
        """Oldest first so a fan-out reads in the order it was dispatched."""
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        parent = store.create_task(session_id=sid, goal="扇出")
        kids = [
            store.create_task(session_id=sid, goal=f"子 {i}", parent_task_id=parent)
            for i in range(3)
        ]

        assert [c["id"] for c in store.list_children(parent)] == kids


class TestTaskStats:
    """The summary view.

    Every number is already in `tasks`; the point is to stop making the user
    run `task list` and add up columns by eye.
    """

    @staticmethod
    def _finish(store: Store, task_id: str, *, steps: int, cost: float, tin: int = 100) -> None:
        store.save_projection(
            task_id,
            status="completed",
            state={},
            steps_used=steps,
            tokens_in=tin,
            tokens_out=10,
            cost_usd=cost,
        )

    def test_totals_add_up_across_tasks(self, store: Store) -> None:
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        for i in range(3):
            task = store.create_task(session_id=sid, goal=f"任务 {i}")
            self._finish(store, task, steps=i + 1, cost=0.01)

        summary = store.task_stats()

        assert summary["tasks"] == 3
        assert summary["steps"] == 6  # 1 + 2 + 3
        assert summary["tokens_in"] == 300
        assert summary["cost_usd"] == pytest.approx(0.03)
        assert summary["by_status"] == {"completed": 3}

    def test_the_dearest_come_first(self, store: Store) -> None:
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        cheap = store.create_task(session_id=sid, goal="便宜")
        dear = store.create_task(session_id=sid, goal="贵")
        self._finish(store, cheap, steps=1, cost=0.01)
        self._finish(store, dear, steps=1, cost=0.99)

        assert store.task_stats()["most_expensive"][0]["goal"] == "贵"

    def test_an_empty_store_is_zero_rather_than_an_error(self, store: Store) -> None:
        """A fresh install runs `stats` before it runs anything else."""
        summary = store.task_stats()

        assert summary["tasks"] == 0
        assert summary["cost_usd"] == 0.0
        assert summary["by_status"] == {}
        assert summary["most_expensive"] == []

    def test_appending_updates_the_projection_cursor(self, store: Store) -> None:
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        task = store.create_task(session_id=sid, goal="g")
        store.append(task, EventType.STATE_TRANSITION, {})
        assert store.get_task(task)["last_event_seq"] == 2


class TestRedactionAtTheBoundary:
    def test_secrets_are_masked_on_the_way_into_the_store(self, tmp_path) -> None:  # noqa: ANN001
        secret = "sk-live-abcdefghijklmnopqrstuvwxyz"
        store = Store(tmp_path / "wukong.db", redactor=Redactor([secret]))
        try:
            sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
            task = store.create_task(session_id=sid, goal="g")
            store.append(task, EventType.TOOL_COMPLETED, {"output": f"token={secret}"})

            event = store.events(task)[-1]
            assert secret not in str(event.payload)
            assert "REDACTED" in str(event.payload)
        finally:
            store.close()

    def test_artifacts_are_redacted_before_being_written(self, tmp_path) -> None:  # noqa: ANN001
        """Offloading a 200 KB tool output is pointless if the key lands on disk."""
        secret = "ghp_" + "a" * 36
        store = Store(tmp_path / "wukong.db", redactor=Redactor([secret]))
        try:
            path = store.save_artifact(
                task_id="t",
                tool_call_id=None,
                content=f"env output:\nGITHUB_TOKEN={secret}\n",
                artifact_dir=tmp_path / "artifacts",
                sha256="deadbeef",
            )
            from pathlib import Path

            assert secret not in Path(path).read_text(encoding="utf-8")
        finally:
            store.close()

    def test_pattern_based_redaction_catches_unknown_secrets(self) -> None:
        redactor = Redactor()
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature_part"
        assert "eyJhbGciOiJIUzI1NiJ9" not in redactor(text)
        assert "REDACTED" in redactor(text)

    def test_env_file_content_is_masked(self) -> None:
        redactor = Redactor()
        assert "sk-real" not in redactor("OPENAI_API_KEY=sk-realvalue123456\n")


class TestToolCallLedger:
    def _begin(self, store: Store, task: str, **overrides):  # noqa: ANN202
        kwargs = dict(
            task_id=task,
            step_id="step_1",
            attempt=1,
            name="read_file",
            arguments={"path": "a.py"},
            effect_class="read_only",
            idempotency_key="key1",
        )
        kwargs.update(overrides)
        return store.begin_tool_call(**kwargs)

    def test_started_rows_are_visible_before_completion(self, store: Store) -> None:
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        task = store.create_task(session_id=sid, goal="g")
        call_id = self._begin(store, task)

        unfinished = store.unfinished_calls(task)
        assert [c.id for c in unfinished] == [call_id]
        assert unfinished[0].status == "started"
        assert unfinished[0].arguments == {"path": "a.py"}

    def test_finishing_removes_it_from_unfinished(self, store: Store) -> None:
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        task = store.create_task(session_id=sid, goal="g")
        call_id = self._begin(store, task)
        store.finish_tool_call(call_id, status="succeeded", result={"output": "ok"}, duration_ms=3)

        assert store.unfinished_calls(task) == []
        record = store.tool_call(call_id)
        assert record.status == "succeeded"
        assert record.result == {"output": "ok"}
        assert record.duration_ms == 3

    def test_attempts_share_an_idempotency_key(self, store: Store) -> None:
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        task = store.create_task(session_id=sid, goal="g")
        self._begin(store, task, attempt=1)
        self._begin(store, task, attempt=2)

        rows = store.list_tool_calls(task)
        assert [r.attempt for r in rows] == [1, 2]
        # Same logical call, so both attempts carry the same key -- which is
        # what makes "did this exact call run twice?" answerable from the
        # per-task ledger without a separate key index.
        assert {r.idempotency_key for r in rows} == {"key1"}
        assert len(store.unfinished_calls(task)) == 2

    def test_ambiguous_is_a_terminal_state(self, store: Store) -> None:
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        task = store.create_task(session_id=sid, goal="g")
        call_id = self._begin(store, task, name="run_command", effect_class="execute_local")
        store.finish_tool_call(call_id, status="ambiguous", error="outcome unknown")

        assert store.unfinished_calls(task) == []
        assert store.tool_call(call_id).status == "ambiguous"

    def test_ledger_survives_a_reopen(self, tmp_path) -> None:  # noqa: ANN001
        path = tmp_path / "wukong.db"
        store = Store(path)
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        task = store.create_task(session_id=sid, goal="g")
        self._begin(store, task)
        store.close()

        reopened = Store(path)
        try:
            assert len(reopened.unfinished_calls(task)) == 1
            # Only TASK_CREATED: the ledger is a separate table.
            assert [e.seq for e in reopened.events(task)] == [1]
        finally:
            reopened.close()


class TestMemorySearch:
    def test_chinese_two_character_query_matches(self, store: Store) -> None:
        """FTS5's unicode61 tokenizer treats a whole CJK run as one token, and
        trigram needs >=3 characters. A two-character Chinese query is the
        most common case there is, so the LIKE fallback is load-bearing."""
        store.add_memory(content="认证模块的测试覆盖率不足，需要补充用例", tags=["testing"])
        hits = store.search_memories("认证")
        assert hits, "a 2-char CJK query must still find the memory"
        assert "认证模块" in hits[0]["content"]

    def test_three_character_chinese_query_uses_fts(self, store: Store) -> None:
        store.add_memory(content="部署流程需要通过 CI 的检查才能合并")
        assert store.search_memories("部署流程")

    def test_english_search(self, store: Store) -> None:
        store.add_memory(content="run the test suite with pytest -q")
        assert store.search_memories("pytest")

    def test_scope_filter(self, store: Store) -> None:
        store.add_memory(content="project-scoped note about deploys", scope="project")
        store.add_memory(content="user-scoped note about deploys", scope="user")
        hits = store.search_memories("deploys", scope="user")
        assert len(hits) == 1 and hits[0]["scope"] == "user"

    def test_empty_query_returns_nothing(self, store: Store) -> None:
        store.add_memory(content="something")
        assert store.search_memories("") == []

    def test_fts_query_with_syntax_characters_does_not_crash(self, store: Store) -> None:
        store.add_memory(content="a note about OR and AND operators")
        assert store.search_memories('"OR" AND*') is not None

    def test_delete_removes_from_the_index(self, store: Store) -> None:
        mid = store.add_memory(content="temporary note about penguins")
        assert store.search_memories("penguins")
        assert store.delete_memory(mid)
        assert store.search_memories("penguins") == []


class TestSkillsRegistryTable:
    def test_upsert_is_idempotent(self, store: Store) -> None:
        for _ in range(2):
            store.upsert_skill(
                name="reviewer",
                path="/x/SKILL.md",
                description="Review code.",
                source="local",
                status="active",
                allowed_tools="read_file",
                sha256="abc",
            )
        rows = store.list_skills()
        assert len(rows) == 1
        assert rows[0]["description"] == "Review code."


class TestTaskListing:
    def test_tasks_are_listed_newest_first(self, store: Store) -> None:
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        for goal in ("first", "second", "third"):
            store.create_task(session_id=sid, goal=goal)
        goals = [t["goal"] for t in store.list_tasks(session_id=sid)]
        assert goals[0] == "third"

    def test_listing_can_filter_to_what_is_still_resumable(self, store: Store) -> None:
        """`wukong task list --resumable`.

        The filter lives on the listing rather than on a point lookup: a list
        view is exactly what the projection is for, while `resume` itself
        must decide from the event fold.
        """
        sid = store.create_session(name="s", working_dir="/tmp", model_alias="mock")
        live = store.create_task(session_id=sid, goal="live")
        done = store.create_task(session_id=sid, goal="done")
        store.save_projection(
            done,
            status="completed",
            state={},
            steps_used=1,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
        )

        resumable = store.list_tasks(statuses=["pending", "planning", "running"])
        assert [t["id"] for t in resumable] == [live]

        # An empty filter means "none", not "no filter": the caller asked for
        # an empty set of statuses.
        assert store.list_tasks(statuses=[]) == []
