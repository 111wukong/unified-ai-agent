"""Searching past tasks.

The record already existed -- every goal, every answer, every event -- and none
of it was reachable. An agent asked to "do the thing I did last week" had no way
to find out what that was, and neither did the user.
"""

from __future__ import annotations

from pathlib import Path

from wukong.storage.store import Store
from wukong.tools.base import ToolContext
from wukong.tools.memory_tools import RecallTool


def seed(store: Store, session: str, goal: str, answer: str, *, parent: str | None = None) -> str:
    task_id = store.create_task(session_id=session, goal=goal, parent_task_id=parent)
    store.save_projection(
        task_id,
        status="completed",
        state={},
        steps_used=3,
        tokens_in=10,
        tokens_out=20,
        cost_usd=0.0,
        result={"answer": answer, "error": None},
    )
    return task_id


class TestSearchTasks:
    def test_a_task_is_found_by_its_goal(self, settings) -> None:  # noqa: ANN001
        store = Store(settings.db_path)
        try:
            session = store.create_session(
                name="s", working_dir=str(settings.workspace), model_alias="mock"
            )
            seed(store, session, "修复 orders 包的边界 bug", "改了两处比较")
            seed(store, session, "总结项目结构", "三个模块")

            hits = store.search_tasks("边界")
        finally:
            store.close()
        assert len(hits) == 1
        assert "orders" in hits[0]["goal"]

    def test_a_task_is_found_by_its_answer(self, settings) -> None:  # noqa: ANN001
        """Searching only the goal would miss the case that matters most:
        'what did I conclude about X'."""
        store = Store(settings.db_path)
        try:
            session = store.create_session(
                name="s", working_dir=str(settings.workspace), model_alias="mock"
            )
            seed(store, session, "看一下部署配置", "结论是 staging 用 alembic 迁移")
            hits = store.search_tasks("alembic")
        finally:
            store.close()
        assert len(hits) == 1
        assert "部署" in hits[0]["goal"]

    def test_sub_agent_tasks_are_hidden_by_default(self, settings) -> None:  # noqa: ANN001
        """One multi-agent run writes a task per worker. Including them buries
        the tasks the user actually started under the ones the runtime
        invented."""
        store = Store(settings.db_path)
        try:
            session = store.create_session(
                name="s", working_dir=str(settings.workspace), model_alias="mock"
            )
            parent = seed(store, session, "对比五个模块的取舍", "见报告")
            seed(store, session, "worker: 读 config.py", "读完了", parent=parent)

            visible = store.search_tasks("config")
            everything = store.search_tasks("config", include_children=True)
        finally:
            store.close()
        assert visible == [], "the worker task should not surface on its own"
        assert len(everything) == 1

    def test_a_stray_quote_does_not_raise(self, settings) -> None:  # noqa: ANN001
        """The query reaches FTS5, where `"`, `*` and `NEAR` are syntax. A
        search box that throws on a quote is worse than one that finds
        nothing."""
        store = Store(settings.db_path)
        try:
            session = store.create_session(
                name="s", working_dir=str(settings.workspace), model_alias="mock"
            )
            seed(store, session, 'fix the "quoted" thing', "done")
            for query in ['"', "a*b", "NEAR(a b)", 'unbalanced "quote', "()"]:
                store.search_tasks(query)  # must not raise
            assert store.search_tasks('"quoted"'), "a quoted phrase should still match"
        finally:
            store.close()

    def test_an_empty_query_returns_nothing(self, settings) -> None:  # noqa: ANN001
        store = Store(settings.db_path)
        try:
            session = store.create_session(
                name="s", working_dir=str(settings.workspace), model_alias="mock"
            )
            seed(store, session, "something", "anything")
            assert store.search_tasks("") == []
            assert store.search_tasks("   ") == []
        finally:
            store.close()

    def test_the_limit_is_respected(self, settings) -> None:  # noqa: ANN001
        store = Store(settings.db_path)
        try:
            session = store.create_session(
                name="s", working_dir=str(settings.workspace), model_alias="mock"
            )
            for i in range(6):
                seed(store, session, f"任务编号 {i} 关于缓存", f"结论 {i}")
            assert len(store.search_tasks("缓存", limit=2)) == 2
        finally:
            store.close()


class TestRecallTool:
    def context(self, tmp_path: Path) -> ToolContext:
        return ToolContext(
            task_id="t",
            session_id="s",
            step_id="step_1",
            workspace=tmp_path,
            home=tmp_path,
            artifact_dir=tmp_path,
        )

    async def test_it_lists_what_it_found(self, settings, tmp_path: Path) -> None:  # noqa: ANN001
        store = Store(settings.db_path)
        try:
            session = store.create_session(
                name="s", working_dir=str(settings.workspace), model_alias="mock"
            )
            task_id = seed(store, session, "修复边界 bug", "改了两处")
            result = await RecallTool(store).run({"query": "边界"}, self.context(tmp_path))
        finally:
            store.close()

        assert result.success
        assert task_id in result.output
        assert "修复边界 bug" in result.output
        assert result.metadata["matches"] == 1

    async def test_no_match_says_what_to_try_instead(self, settings, tmp_path: Path) -> None:  # noqa: ANN001
        """'Nothing found' with no next step is a dead end for a model."""
        store = Store(settings.db_path)
        try:
            session = store.create_session(
                name="s", working_dir=str(settings.workspace), model_alias="mock"
            )
            seed(store, session, "修复边界 bug", "改了两处")
            result = await RecallTool(store).run(
                {"query": "完全无关的词汇xyz"}, self.context(tmp_path)
            )
        finally:
            store.close()

        assert result.success, "finding nothing is not an error"
        assert "no past task matches" in result.output
        assert "wording differs" in result.output

    async def test_it_is_read_only(self) -> None:
        from wukong.types import EffectClass

        assert RecallTool.spec.effect_class is EffectClass.READ_ONLY
        assert RecallTool.spec.idempotent is True
