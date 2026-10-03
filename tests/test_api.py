"""Service layer: REST + AG-UI over SSE + WebSocket.

The AG-UI tests assert on the wire format, not on internal state, because
the whole point of adopting the protocol is that an external frontend can
consume it.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("fastapi", reason="install with `pip install -e '.[api]'`")

from fastapi.testclient import TestClient  # noqa: E402

from wukong.api.app import Service, create_app  # noqa: E402
from wukong.agent.factory import build_agent  # noqa: E402

from tests.conftest import ScriptedModel, ScriptedModels  # noqa: E402


def parse_sse(text: str) -> list[dict]:
    """Decode an SSE body into the JSON payloads it carried."""
    events: list[dict] = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload:
                events.append(json.loads(payload))
    return events


def types_of(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


@pytest.fixture
def api(settings, workspace):  # noqa: ANN001
    """A TestClient whose runtime uses the offline scripted model."""
    created: list = []

    def _make(script=None):  # noqa: ANN001
        model = ScriptedModel(settings.models["scripted"], script)
        service = Service(settings)
        # Pre-build the agent with an injected model registry so no API key
        # is ever needed; the lifespan then leaves it alone.
        service.agent = asyncio.run(
            build_agent(settings=settings, models=ScriptedModels(model))
        )
        created.append(service.agent)
        # The guard only accepts loopback Host headers, and the test client
        # sends `testserver`. Declaring it here keeps the allowlist in
        # production loopback-only rather than widening it for tests.
        return TestClient(
            create_app(settings, service=service, allowed_hosts=["testserver", "127.0.0.1"])
        )

    yield _make
    for agent in created:
        agent.close()


# ---------------------------------------------------------------------------


class TestMeta:
    def test_health(self, api) -> None:
        client = api([])
        with client:
            body = client.get("/api/v1/health").json()
        assert body["status"] == "ok"
        assert "version" in body

    def test_sandbox_status_is_reported_honestly(self, api) -> None:
        """A silent fallback to no sandbox is the bug this endpoint prevents."""
        client = api([])
        with client:
            body = client.get("/api/v1/sandbox").json()
        assert body["backend"] in {"none", "seatbelt", "docker"}
        assert "caveats" in body and body["caveats"]
        if body["backend"] == "none":
            assert body["notes"], "falling back to no sandbox must say why"

    def test_tools_are_listed(self, api) -> None:
        client = api([])
        with client:
            tools = client.get("/api/v1/tools").json()
        names = {t["name"] for t in tools}
        assert {"read_file", "run_command", "git_status", "update_plan"} <= names


class TestSkillPromotion:
    """The ladder is the safety mechanism, so it needs an entry point that is
    not "edit the database". A gate reachable only from a terminal is a gate
    that gets worked around."""

    @staticmethod
    def _write_candidate(settings) -> None:  # noqa: ANN001
        from tests.test_skills import write_skill

        candidates = settings.home / "skills-candidates"
        write_skill(
            candidates,
            "invented",
            "name: invented\ndescription: A skill the agent wrote for itself.\n",
        )

    def test_a_candidate_cannot_jump_the_ladder(self, api, settings) -> None:  # noqa: ANN001
        self._write_candidate(settings)
        client = api([])
        with client:
            response = client.post(
                "/api/v1/skills/invented/promote", json={"status": "active"}
            )
        assert response.status_code == 400
        assert "skip" in response.json()["detail"]

    def test_promotion_moves_one_rung_and_persists(self, api, settings) -> None:  # noqa: ANN001
        self._write_candidate(settings)
        client = api([])
        with client:
            response = client.post(
                "/api/v1/skills/invented/promote", json={"status": "validated"}
            )
            body = response.json()
            listed = client.get("/api/v1/skills").json()

        assert response.status_code == 200
        assert body["status"] == "validated"
        stored = {row["name"]: row["status"] for row in listed}
        assert stored["invented"] == "validated", "the promotion did not reach the store"

    def test_an_unknown_skill_is_a_client_error(self, api) -> None:
        client = api([])
        with client:
            response = client.post(
                "/api/v1/skills/nope/promote", json={"status": "validated"}
            )
        assert response.status_code == 400
        assert "unknown skill" in response.json()["detail"]

    def test_an_invalid_status_is_rejected_by_the_schema(self, api) -> None:
        client = api([])
        with client:
            response = client.post(
                "/api/v1/skills/invented/promote", json={"status": "super-active"}
            )
        assert response.status_code == 422

    def test_skill_runs_are_exposed(self, api) -> None:  # noqa: ANN001
        client = api([])
        with client:
            assert client.get("/api/v1/skills/runs").json() == []


class TestRestTasks:
    def test_create_and_wait(self, api) -> None:
        client = api(
            [
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]},
                {"content": "app.py defines add()"},
            ]
        )
        with client:
            created = client.post(
                "/api/v1/tasks", json={"goal": "read app.py", "model": "scripted"}
            ).json()
            assert created["status"] == "pending"
            task = client.post(f"/api/v1/tasks/{created['id']}/wait", params={"timeout_s": 30}).json()

        assert task["status"] == "completed"
        assert "app.py" in (task["answer"] or "")
        assert task["steps"] >= 1
        assert task["plan"], "the plan must be exposed to the client"

    def test_events_are_exposed_in_order(self, api) -> None:
        client = api([{"content": "done"}])
        with client:
            task_id = client.post(
                "/api/v1/tasks", json={"goal": "say done", "model": "scripted"}
            ).json()["id"]
            client.post(f"/api/v1/tasks/{task_id}/wait", params={"timeout_s": 30})
            events = client.get(f"/api/v1/tasks/{task_id}/events").json()

        assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
        assert events[0]["type"] == "task_created"
        assert events[-1]["type"] == "task_completed"

    def test_unknown_task_is_404(self, api) -> None:
        client = api([])
        with client:
            assert client.get("/api/v1/tasks/task_nope").status_code == 404

    def test_approve_without_pending_is_409(self, api) -> None:
        client = api([{"content": "done"}])
        with client:
            task_id = client.post(
                "/api/v1/tasks", json={"goal": "nothing", "model": "scripted"}
            ).json()["id"]
            client.post(f"/api/v1/tasks/{task_id}/wait", params={"timeout_s": 30})
            response = client.post(f"/api/v1/tasks/{task_id}/approve")
        assert response.status_code == 409

    def test_system_admin_cannot_be_pre_approved_over_http(self, api) -> None:
        """A web request must never be able to grant SYSTEM_ADMIN."""
        client = api([])
        with client:
            response = client.post(
                "/api/v1/tasks",
                json={"goal": "x", "model": "scripted", "approve": ["system_admin"]},
            )
        assert response.status_code == 400
        assert "不能通过 HTTP 预授权" in response.json()["detail"]

    def test_unknown_effect_is_400(self, api) -> None:
        client = api([])
        with client:
            response = client.post(
                "/api/v1/tasks", json={"goal": "x", "model": "scripted", "approve": ["nope"]}
            )
        assert response.status_code == 400


class TestAgUiProtocol:
    def test_run_lifecycle_and_message_events(self, api) -> None:
        client = api([{"content": "hello from the agent"}])
        with client:
            response = client.post(
                "/agui",
                json={
                    "threadId": "thread_1",
                    "runId": "run_1",
                    "messages": [{"role": "user", "content": "say hello"}],
                    "model": "scripted",
                },
            )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")

        events = parse_sse(response.text)
        kinds = types_of(events)
        assert kinds[0] == "RUN_STARTED"
        assert "RUN_FINISHED" in kinds
        assert "TEXT_MESSAGE_START" in kinds
        assert "TEXT_MESSAGE_END" in kinds

        started = events[0]
        assert started["threadId"] == "thread_1"
        assert started["runId"] == "run_1"

    def test_tool_call_events_use_the_spec_field_names(self, api) -> None:
        client = api(
            [
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]},
                {"content": "done"},
            ]
        )
        with client:
            response = client.post(
                "/agui",
                json={
                    "messages": [{"role": "user", "content": "read app.py"}],
                    "model": "scripted",
                },
            )
        events = parse_sse(response.text)
        by_type = {}
        for event in events:
            by_type.setdefault(event["type"], []).append(event)

        assert "TOOL_CALL_START" in by_type
        start = by_type["TOOL_CALL_START"][0]
        assert {"toolCallId", "toolCallName"} <= set(start)
        assert start["toolCallName"] == "read_file"

        assert "TOOL_CALL_ARGS" in by_type
        assert "delta" in by_type["TOOL_CALL_ARGS"][0]

        assert "TOOL_CALL_END" in by_type
        assert "TOOL_CALL_RESULT" in by_type
        result = by_type["TOOL_CALL_RESULT"][0]
        assert {"messageId", "toolCallId", "content"} <= set(result)

        # STEP_STARTED/FINISHED bracket the tool, per the lifecycle pattern.
        assert "STEP_STARTED" in by_type and "STEP_FINISHED" in by_type

    def test_plan_is_published_as_an_activity_snapshot(self, api) -> None:
        client = api([{"content": "done"}])
        with client:
            response = client.post(
                "/agui",
                json={"messages": [{"role": "user", "content": "plan something"}], "model": "scripted"},
            )
        events = parse_sse(response.text)
        activities = [e for e in events if e["type"] == "ACTIVITY_SNAPSHOT"]
        assert activities, "the plan must reach the UI"
        plan = activities[0]
        assert plan["activityType"] == "PLAN"
        assert plan["content"]["steps"]
        assert "messageId" in plan

    def test_pause_is_run_finished_with_an_interrupt(self, api) -> None:
        """AG-UI expresses HITL as a terminal event with an interrupt outcome.

        The frontend therefore has exactly one terminal event to handle.
        """
        client = api(
            [
                {"tool_calls": [{"name": "run_command", "arguments": {"command": "echo hi"}}]},
                {"content": "ran it"},
            ]
        )
        with client:
            response = client.post(
                "/agui",
                json={"messages": [{"role": "user", "content": "run echo"}], "model": "scripted"},
            )
        events = parse_sse(response.text)
        finals = [e for e in events if e["type"] == "RUN_FINISHED"]
        assert finals, "an interrupt still ends the run"
        outcome = finals[-1]["outcome"]
        assert outcome["type"] == "interrupt"
        interrupt = outcome["interrupts"][0]
        assert interrupt["tool"] == "run_command"
        assert interrupt["effect"] == "execute_local"
        assert interrupt["id"]
        # No RUN_ERROR: a pause is not a failure.
        assert "RUN_ERROR" not in types_of(events)

    def test_interrupt_can_be_resumed(self, api) -> None:
        client = api(
            [
                {"tool_calls": [{"name": "run_command", "arguments": {"command": "echo hi"}}]},
                {"content": "ran it"},
            ]
        )
        with client:
            first = client.post(
                "/agui",
                json={"messages": [{"role": "user", "content": "run echo"}], "model": "scripted"},
            )
            parse_sse(first.text)
            # This used to read the task id back over REST, with a note that the
            # stream did not carry it. That gap was real and it had teeth: the
            # console could not tell which task a run belonged to, so approving
            # a prompt posted to whatever the list had selected -- a 409 the
            # moment anything had been run before.
            started = parse_sse(first.text)[0]
            assert started["type"] == "RUN_STARTED"
            task_id = started["taskId"]
            assert client.get(f"/api/v1/tasks/{task_id}").json()["status"] == "waiting_confirmation"

            resumed = client.post(f"/api/v1/tasks/{task_id}/approve")
            assert resumed.status_code == 200
            client.post(f"/api/v1/tasks/{task_id}/wait", params={"timeout_s": 30})
            final = client.get(f"/api/v1/tasks/{task_id}").json()

        assert final["status"] == "completed"

    def test_empty_input_is_a_run_error(self, api) -> None:
        client = api([])
        with client:
            response = client.post("/agui", json={"messages": [], "model": "scripted"})
        events = parse_sse(response.text)
        assert types_of(events) == ["RUN_ERROR"]
        assert events[0]["code"] == "empty_input"

    def test_resume_pointing_at_an_unknown_task_errors(self, api) -> None:
        client = api([])
        with client:
            response = client.post(
                "/agui",
                json={"messages": [], "resume": [{"taskId": "task_missing"}], "model": "scripted"},
            )
        events = parse_sse(response.text)
        assert types_of(events) == ["RUN_ERROR"]
        assert events[0]["code"] == "bad_resume"

    def test_every_event_carries_type_and_timestamp(self, api) -> None:
        client = api([{"content": "hi"}])
        with client:
            response = client.post(
                "/agui",
                json={"messages": [{"role": "user", "content": "hi"}], "model": "scripted"},
            )
        for event in parse_sse(response.text):
            assert "type" in event
            assert "timestamp" in event


class TestAttachStream:
    def test_attaching_replays_history_then_streams(self, api) -> None:
        """A late subscriber must not be blind to what already happened."""
        client = api([{"content": "done"}])
        with client:
            task_id = client.post(
                "/api/v1/tasks", json={"goal": "say done", "model": "scripted"}
            ).json()["id"]
            client.post(f"/api/v1/tasks/{task_id}/wait", params={"timeout_s": 30})
            response = client.get(f"/api/v1/tasks/{task_id}/stream")

        events = parse_sse(response.text)
        kinds = types_of(events)
        assert kinds[0] == "RUN_STARTED"
        assert "TEXT_MESSAGE_CONTENT" in kinds, "the finished answer must be replayed"
        assert "RUN_FINISHED" in kinds

    def test_unknown_task_is_404(self, api) -> None:
        client = api([])
        with client:
            assert client.get("/api/v1/tasks/task_nope/stream").status_code == 404


class TestWebSocket:
    def test_streams_the_same_events_as_sse(self, api) -> None:
        client = api(
            [
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]},
                {"content": "done"},
            ]
        )
        with client:
            task_id = client.post(
                "/api/v1/tasks", json={"goal": "read app.py", "model": "scripted"}
            ).json()["id"]
            received: list[dict] = []
            with client.websocket_connect(f"/ws/tasks/{task_id}") as ws:
                while True:
                    try:
                        message = ws.receive_json()
                    except Exception:  # noqa: BLE001 - socket closed
                        break
                    received.append(message)
                    if message["type"] == "RUN_FINISHED":
                        break
            client.post(f"/api/v1/tasks/{task_id}/wait", params={"timeout_s": 30})

        assert received[0]["type"] == "RUN_STARTED"
        assert "RUN_FINISHED" in types_of(received)


class TestSessions:
    def test_session_round_trip(self, api) -> None:
        client = api([])
        with client:
            created = client.post("/api/v1/sessions", json={"name": "my-project"}).json()
            assert created["id"].startswith("session_")
            listed = client.get("/api/v1/sessions").json()
        assert any(s["id"] == created["id"] for s in listed)


class TestWireFormatInvariants:
    """Guards against silent protocol corruption.

    The bug these exist for: a handler returned a single dict where the
    caller expected a list, so `for x in result` iterated the dict's *keys*
    and the wire carried `data: "type"` instead of the event. Every other
    assertion still looked plausible, which is what made it silent.
    """

    def test_every_frame_is_a_json_object(self, api) -> None:
        client = api(
            [
                {"tool_calls": [{"name": "read_file", "arguments": {"path": "app.py"}}]},
                {"content": "done"},
            ]
        )
        with client:
            response = client.post(
                "/agui",
                json={"messages": [{"role": "user", "content": "read app.py"}], "model": "scripted"},
            )
        for line in response.text.splitlines():
            if not line.strip().startswith("data:"):
                continue
            payload = line.strip()[5:].strip()
            if not payload:
                continue
            decoded = json.loads(payload)
            assert isinstance(decoded, dict), f"frame was not an object: {payload[:120]}"
            assert isinstance(decoded.get("type"), str)

    def test_no_frame_carries_a_bare_field_name(self, api) -> None:
        """The exact symptom: dict keys leaking onto the wire as strings."""
        client = api([{"content": "hello"}])
        with client:
            response = client.post(
                "/agui",
                json={"messages": [{"role": "user", "content": "hi"}], "model": "scripted"},
            )
        payloads = [
            line.strip()[5:].strip()
            for line in response.text.splitlines()
            if line.strip().startswith("data:")
        ]
        for payload in payloads:
            assert payload.startswith("{"), f"expected a JSON object, got {payload[:80]}"

    def test_run_finished_actually_reaches_the_wire(self, api) -> None:
        client = api([{"content": "hello"}])
        with client:
            response = client.post(
                "/agui",
                json={"messages": [{"role": "user", "content": "hi"}], "model": "scripted"},
            )
        kinds = types_of(parse_sse(response.text))
        assert kinds[-1] == "RUN_FINISHED", f"last event was {kinds[-1]!r}"

    def test_on_event_always_returns_a_list(self) -> None:
        """Unit-level guard for the normalisation, independent of HTTP."""
        from wukong.api.agui import AgUiEncoder
        from wukong.observability.events import Event, EventType

        encoder = AgUiEncoder(task_id="t", thread_id="th", run_id="r")
        single = encoder.on_event(
            Event(seq=1, task_id="t", type=EventType.TASK_COMPLETED, payload={"answer": "ok"})
        )
        assert isinstance(single, list) and len(single) == 1
        assert single[0]["type"] == "RUN_FINISHED"

        empty = encoder.on_event(Event(seq=2, task_id="t", type=EventType.LOG_APPENDED, payload={}))
        assert empty == []

        many = encoder.on_event(
            Event(
                seq=3,
                task_id="t",
                type=EventType.TOOL_STARTED,
                payload={"call_id": "c1", "name": "read_file", "arguments": {}},
            )
        )
        assert isinstance(many, list) and len(many) >= 3




class TestA2A:
    """The A2A surface over HTTP: card discovery and the JSON-RPC binding."""

    def test_the_card_is_absent_until_a2a_is_enabled(self, api, settings) -> None:  # noqa: ANN001
        """Publishing an endpoint that accepts work from other agents is a
        decision, so the route says so rather than 404-ing silently."""
        client = api([])
        with client:
            response = client.get("/.well-known/agent-card.json")
        assert response.status_code == 404
        assert "a2a.enabled" in response.json()["detail"]

    def test_the_card_describes_the_agent(self, api, settings) -> None:  # noqa: ANN001
        settings.a2a.enabled = True
        settings.a2a.name = "test-agent"
        client = api([])
        with client:
            card = client.get("/.well-known/agent-card.json").json()
        assert card["name"] == "test-agent"
        assert card["protocolVersion"] == "1.0"
        assert card["url"].endswith("/a2a")
        assert card["capabilities"]["streaming"] is True
        # Honest: this build does not send webhooks, so it must not say it does.
        assert card["capabilities"]["pushNotifications"] is False

    def test_message_send_over_http(self, api, settings) -> None:  # noqa: ANN001
        settings.a2a.enabled = True
        client = api([{"content": "hello from the peer"}])
        with client:
            response = client.post(
                "/a2a",
                json={
                    "jsonrpc": "2.0",
                    "id": "1",
                    "method": "message/send",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{"kind": "text", "text": "do the thing"}],
                        }
                    },
                },
            )
        body = response.json()
        assert "error" not in body, body.get("error")
        assert body["result"]["status"]["state"] == "completed"

    def test_a_malformed_body_is_a_json_rpc_parse_error(self, api, settings) -> None:  # noqa: ANN001
        settings.a2a.enabled = True
        client = api([])
        with client:
            response = client.post(
                "/a2a", content=b"not json", headers={"content-type": "application/json"}
            )
        assert response.status_code == 200
        assert response.json()["error"]["code"] == -32700

    def test_message_stream_is_served_as_sse(self, api, settings) -> None:  # noqa: ANN001
        settings.a2a.enabled = True
        client = api([{"content": "streamed"}])
        with client:
            response = client.post(
                "/a2a",
                json={
                    "jsonrpc": "2.0",
                    "id": "1",
                    "method": "message/stream",
                    "params": {
                        "message": {
                            "role": "user",
                            "parts": [{"kind": "text", "text": "go"}],
                        }
                    },
                },
            )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = parse_sse(response.text)
        assert events, "the stream carried nothing"
        assert events[0]["result"]["state"] == "submitted"
        assert events[-1]["result"]["final"] is True


class TestCancelEndpoint:
    """The console puts a Cancel button next to every approval prompt.

    So this is the path a user is most likely to hit first -- and it used to
    answer `"cancelling"` while the task sat in `waiting_confirmation` for
    ever, because a paused task has no loop to read the cancellation marker.
    """

    def test_cancelling_a_paused_task_actually_cancels_it(self, api, settings) -> None:  # noqa: ANN001
        client = api(
            [
                {"tool_calls": [{"name": "run_command", "arguments": {"command": "ls"}}]},
            ]
        )
        with client:
            created = client.post("/api/v1/tasks", json={"goal": "run ls"}).json()
            task_id = created.get("task_id") or created.get("id")
            client.post(f"/api/v1/tasks/{task_id}/wait", json={"timeout_s": 10})

            paused = client.get(f"/api/v1/tasks/{task_id}").json()
            assert paused["status"] == "waiting_confirmation", (
                "the test needs the task paused, or it proves nothing"
            )

            response = client.post(f"/api/v1/tasks/{task_id}/cancel")
            assert response.status_code == 200
            body = response.json()
            assert body["outcome"] == "cancelled"
            assert body["status"] == "cancelled"

            after = client.get(f"/api/v1/tasks/{task_id}").json()
        assert after["status"] == "cancelled"

    def test_an_unknown_task_is_a_404(self, api) -> None:  # noqa: ANN001
        client = api([])
        with client:
            response = client.post("/api/v1/tasks/task_nope/cancel")
        assert response.status_code == 404


def _referenced_assets(html: str) -> list[str]:
    """Stylesheet and script targets, resolved against `/`.

    Only `<link>` and `<script>`: those are what the page cannot render
    without. A footer link to a guarded API route is a navigation, not an
    asset, and asserting it would make this test fail for the wrong reason.

    Relative on purpose -- an absolute URL points somewhere else by
    definition, and the question is only whether the page can load what it
    asks its own origin for.
    """
    import re

    found = re.findall(r'<(?:link|script)\b[^>]*?\b(?:href|src)="([^"]+)"', html)
    return [
        f"/{target.lstrip('/')}"
        for target in found
        if not target.startswith(("http://", "https://", "//", "#", "data:"))
    ]


class TestApprovingTheRightTask:
    """A client must be able to learn which task its run belongs to.

    Reported from use: approving a prompt answered `409 task ... has no pending
    approval`. The console had no way to know the id of the task it had just
    started, so it posted the approval to whichever task the list had selected
    -- correct only when the run happened to be that one.
    """

    def test_run_started_carries_the_task_id(self, api) -> None:  # noqa: ANN001
        client = api([{"content": "hi"}])
        with client:
            events = parse_sse(
                client.post(
                    "/agui",
                    json={"messages": [{"role": "user", "content": "hi"}], "model": "scripted"},
                ).text
            )
            started = events[0]
            assert started["type"] == "RUN_STARTED"
            assert started.get("taskId"), "the client cannot resume without it"
            # And it is the real id, not something derived from the thread.
            task = client.get(f"/api/v1/tasks/{started['taskId']}")
        assert task.status_code == 200
        assert task.json()["goal"] == "hi"

    def test_approving_a_task_that_is_not_waiting_is_a_409(self, api) -> None:  # noqa: ANN001
        """The error the user saw, pinned down: a real task, nothing pending.

        It is the right answer to the wrong request, which is why the fix is
        for the client to know its own task id rather than for the server to
        accept the call.
        """
        client = api([{"content": "hi"}])
        with client:
            events = parse_sse(
                client.post(
                    "/agui",
                    json={"messages": [{"role": "user", "content": "hi"}], "model": "scripted"},
                ).text
            )
            task_id = events[0]["taskId"]
            client.post(f"/api/v1/tasks/{task_id}/wait", params={"timeout_s": 30})

            response = client.post(f"/api/v1/tasks/{task_id}/approve")
        assert response.status_code == 409
        # Asserted on the *reason*, not on the sentence: the wording is
        # user-facing text and moves with the interface language.
        assert "没有待处理的审批" in response.json()["detail"]

    def test_the_run_id_is_not_the_task_id(self, api) -> None:  # noqa: ANN001
        """`threadId` and `runId` are the client's own identifiers. A client
        that confused them would post approvals to a name the server has never
        heard of."""
        client = api([{"content": "hi"}])
        with client:
            events = parse_sse(
                client.post(
                    "/agui",
                    json={
                        "threadId": "thread_1",
                        "runId": "run_1",
                        "messages": [{"role": "user", "content": "hi"}],
                        "model": "scripted",
                    },
                ).text
            )
            started = events[0]
        assert started["threadId"] == "thread_1"
        assert started["runId"] == "run_1"
        assert started["taskId"] not in {"thread_1", "run_1"}


class TestGuard:
    """Host allowlist and session token.

    Two independent checks that stop different attackers, which is why both
    exist:

    * **Host** defeats DNS rebinding. A browser will POST to 127.0.0.1 on
      behalf of any page it loads, but it cannot forge the Host header.
    * **Token** defeats another local process, which can find the port but
      not the per-launch token.
    """

    @pytest.fixture
    def guarded(self, settings):  # noqa: ANN001
        """A client whose app requires a token and only trusts loopback."""
        import asyncio as _asyncio

        from tests.conftest import ScriptedModel, ScriptedModels
        from wukong.agent.factory import build_agent

        created: list = []

        def _make(token=None, hosts=None):  # noqa: ANN001
            model = ScriptedModel(settings.models["scripted"], [{"content": "hi"}])
            service = Service(settings)
            service.agent = _asyncio.run(
                build_agent(settings=settings, models=ScriptedModels(model))
            )
            created.append(service.agent)
            app = create_app(
                settings,
                service=service,
                token=token,
                allowed_hosts=hosts if hosts is not None else ["testserver", "127.0.0.1"],
            )
            return TestClient(app)

        yield _make
        for agent in created:
            agent.close()

    def test_host_not_in_the_allowlist_is_refused(self, guarded) -> None:
        """This is the DNS-rebinding case: Host says something else entirely."""
        client = guarded()
        with client:
            response = client.get("/api/v1/health", headers={"Host": "evil.example"})
        assert response.status_code == 403
        assert "not allowed" in response.json()["detail"]

    def test_allowed_host_passes(self, guarded) -> None:
        client = guarded(hosts=["testserver"])
        with client:
            assert client.get("/api/v1/health").status_code == 200

    def test_the_console_and_its_assets_are_not_guarded(self, guarded) -> None:  # noqa: ANN001
        """The console must load before it can present a token.

        Checks the assets the page *actually asks for*, resolved the way a
        browser resolves them. The previous version of this test asserted a
        hard-coded `/console/app.js` -- which passed, because the static mount
        served that path. The page itself was served from `/`, so the browser
        requested `/app.js` and got a 404: no stylesheet, no script, and every
        other check still green. A test that names the path it expects cannot
        catch a disagreement about the path.
        """
        client = guarded(token="s3cret", hosts=["testserver"])
        with client:
            page = client.get("/", headers={"Host": "testserver"})
            assert page.status_code == 200

            assets = _referenced_assets(page.text)
            assert assets, "the console page references no assets; did it change shape?"
            for asset in assets:
                response = client.get(asset, headers={"Host": "testserver"})
                assert response.status_code == 200, (
                    f"the page references {asset} but the server does not serve it "
                    f"({response.status_code}) -- the console would load unstyled "
                    "and without its script"
                )

            # ...while the API behind it is still refused.
            assert (
                client.get("/api/v1/health", headers={"Host": "testserver"}).status_code == 403
            )

    def test_missing_token_is_refused(self, guarded) -> None:
        client = guarded(token="s3cret")
        with client:
            response = client.get("/api/v1/health")
        assert response.status_code == 403
        assert "session token" in response.json()["detail"]

    def test_wrong_token_is_refused(self, guarded) -> None:
        client = guarded(token="s3cret")
        with client:
            response = client.get("/api/v1/health", headers={"X-WUKONG-Token": "nope"})
        assert response.status_code == 403

    def test_header_token_is_accepted(self, guarded) -> None:
        client = guarded(token="s3cret")
        with client:
            response = client.get("/api/v1/health", headers={"X-WUKONG-Token": "s3cret"})
        assert response.status_code == 200
        assert response.json()["token_required"] is True

    def test_query_token_is_accepted(self, guarded) -> None:
        """For curl and for opening a stream in a plain browser tab."""
        client = guarded(token="s3cret")
        with client:
            assert client.get("/api/v1/health?token=s3cret").status_code == 200

    def test_bearer_token_is_accepted(self, guarded) -> None:
        client = guarded(token="s3cret")
        with client:
            response = client.get(
                "/api/v1/health", headers={"Authorization": "Bearer s3cret"}
            )
        assert response.status_code == 200

    def test_the_agui_endpoint_is_guarded_too(self, guarded) -> None:
        """Otherwise the token is trivially bypassed by running a task."""
        client = guarded(token="s3cret")
        with client:
            response = client.post(
                "/agui",
                json={"messages": [{"role": "user", "content": "hi"}], "model": "scripted"},
            )
        assert response.status_code == 403

    def test_no_token_configured_means_no_token_needed(self, guarded) -> None:
        """`wukong serve` without --token stays usable; the Host check still runs."""
        client = guarded(token=None)
        with client:
            assert client.get("/api/v1/health").status_code == 200
            assert client.get("/api/v1/health", headers={"Host": "evil.example"}).status_code == 403

    def test_the_a2a_surface_is_guarded_like_everything_else(self, guarded, settings) -> None:  # noqa: ANN001
        """Otherwise the token is bypassed by calling the agent through A2A.

        The Agent Card is behind the same check rather than being public
        discovery, which is a deliberate deviation from A2A's convention: the
        card lists this agent's skills, and this is a local-first tool where
        the whole surface sits behind loopback + token.
        """
        settings.a2a.enabled = True
        client = guarded(token="s3cret")
        with client:
            assert client.get("/.well-known/agent-card.json").status_code == 403
            assert (
                client.post("/a2a", json={"jsonrpc": "2.0", "method": "tasks/get"}).status_code
                == 403
            )
            # With the token it gets through to the route, not to the guard.
            assert client.get("/.well-known/agent-card.json?token=s3cret").status_code == 200

    def test_health_reports_whether_a_token_is_required(self, guarded) -> None:
        with guarded(token=None) as client:
            assert client.get("/api/v1/health").json()["token_required"] is False


class TestWorkspaceFiles:
    """What the console's file tree reads, one directory at a time."""

    def test_directories_come_before_files(self, api, workspace) -> None:
        (workspace / "a_dir").mkdir()
        (workspace / "z_file.txt").write_text("x", encoding="utf-8")

        with api([]) as client:
            entries = client.get("/api/v1/files").json()["entries"]

        kinds = [e["type"] for e in entries]
        assert kinds == sorted(kinds, key=lambda k: 0 if k == "dir" else 1)
        assert {"a_dir", "z_file.txt"} <= {e["name"] for e in entries}

    def test_a_path_outside_the_workspace_is_refused(self, api) -> None:
        """Refused, not clamped.

        Clamping means the caller reads a different directory than the one it
        asked for and never finds out -- which is worse than an error, because
        an error can be handled.
        """
        with api([]) as client:
            response = client.get("/api/v1/files", params={"path": "../../../etc"})

        assert response.status_code == 400

    def test_hidden_entries_are_left_out(self, api, workspace) -> None:
        (workspace / ".secret").write_text("x", encoding="utf-8")
        (workspace / "visible.txt").write_text("x", encoding="utf-8")

        with api([]) as client:
            names = {e["name"] for e in client.get("/api/v1/files").json()["entries"]}

        assert "visible.txt" in names
        assert ".secret" not in names

    def test_asking_for_a_file_as_a_directory_is_a_404(self, api, workspace) -> None:
        (workspace / "notes.md").write_text("x", encoding="utf-8")

        with api([]) as client:
            response = client.get("/api/v1/files", params={"path": "notes.md"})

        assert response.status_code == 404

    def test_a_task_reports_the_budgets_it_was_created_with(self, api) -> None:
        """The console draws a progress bar from these, so they have to survive
        the round trip from the TASK_CREATED event to the API."""
        client = api([{"content": "done"}])
        with client:
            task_id = client.post("/api/v1/tasks", json={"goal": "g", "wait": True}).json()["id"]
            body = client.get(f"/api/v1/tasks/{task_id}").json()

        assert body["budgets"]["max_steps"] > 0


class TestFileEditing:
    """The editor, and the rule that makes it safe.

    The rule is not "who wins" -- it is "there is only ever one writer". While
    the agent has the workspace, it has it. The agent holds observations of
    these files in its context, and a file changed underneath it turns those
    observations into a lie it will act on: `apply_patch` fails loudly on a
    mismatch, but a decision already made from stale content does not.
    """

    def test_a_save_lands_and_the_digest_changes(self, api, workspace) -> None:
        (workspace / "notes.md").write_text("before\n", encoding="utf-8")

        with api([]) as client:
            opened = client.get("/api/v1/files/content", params={"path": "notes.md"}).json()
            saved = client.put(
                "/api/v1/files/content",
                json={"path": "notes.md", "text": "after\n", "base_sha": opened["sha"]},
            )

        assert saved.status_code == 200, saved.text
        assert (workspace / "notes.md").read_text(encoding="utf-8") == "after\n"
        assert saved.json()["sha"] != opened["sha"]

    def test_a_write_is_refused_while_a_task_is_running(self, api, workspace) -> None:
        """The whole consistency story, in one assertion."""
        (workspace / "notes.md").write_text("before\n", encoding="utf-8")

        client = api([])
        with client:
            client.app.state.svc.running["task_x"] = object()
            try:
                response = client.put(
                    "/api/v1/files/content",
                    json={"path": "notes.md", "text": "sneaky\n"},
                )
            finally:
                client.app.state.svc.running.clear()

        assert response.status_code == 409
        assert (workspace / "notes.md").read_text(encoding="utf-8") == "before\n"

    def test_a_save_over_a_changed_file_is_refused(self, api, workspace) -> None:
        """The optimistic lock. Without it a save silently discards whatever
        wrote in between -- the agent, another tab, or the user's terminal."""
        target = workspace / "notes.md"
        target.write_text("before\n", encoding="utf-8")

        with api([]) as client:
            opened = client.get("/api/v1/files/content", params={"path": "notes.md"}).json()
            target.write_text("written by someone else\n", encoding="utf-8")
            response = client.put(
                "/api/v1/files/content",
                json={"path": "notes.md", "text": "mine\n", "base_sha": opened["sha"]},
            )

        assert response.status_code == 409
        assert target.read_text(encoding="utf-8") == "written by someone else\n"

    def test_a_path_outside_the_workspace_is_refused(self, api) -> None:
        with api([]) as client:
            response = client.put(
                "/api/v1/files/content",
                json={"path": "../../../tmp/evil.txt", "text": "x"},
            )

        assert response.status_code == 400

    def test_a_directory_is_not_a_file(self, api, workspace) -> None:
        (workspace / "a_dir").mkdir()

        with api([]) as client:
            response = client.put(
                "/api/v1/files/content", json={"path": "a_dir", "text": "x"}
            )

        assert response.status_code == 404

    def test_the_read_hands_back_a_digest_to_send_on_save(self, api, workspace) -> None:
        (workspace / "notes.md").write_text("x\n", encoding="utf-8")

        with api([]) as client:
            body = client.get("/api/v1/files/content", params={"path": "notes.md"}).json()

        assert body["sha"]


class TestFilePreview:
    """The read-only preview pane. It must not be able to change anything --
    the agent may be mid-run holding a file it has already read, and an editor
    here would open a question nobody has answered."""

    def test_it_returns_the_text(self, api, workspace) -> None:
        (workspace / "notes.md").write_text("# hello\n\nbody\n", encoding="utf-8")

        with api([]) as client:
            body = client.get("/api/v1/files/content", params={"path": "notes.md"}).json()

        assert body["text"] == "# hello\n\nbody\n"
        assert body["binary"] is False
        assert body["size"] == 14

    def test_a_path_outside_the_workspace_is_refused(self, api) -> None:
        with api([]) as client:
            response = client.get(
                "/api/v1/files/content", params={"path": "../../../etc/hosts"}
            )

        assert response.status_code == 400

    def test_a_directory_is_not_a_file(self, api, workspace) -> None:
        (workspace / "a_dir").mkdir()

        with api([]) as client:
            response = client.get("/api/v1/files/content", params={"path": "a_dir"})

        assert response.status_code == 404

    def test_binary_content_is_declined_rather_than_mangled(self, api, workspace) -> None:
        """Decoding a PNG as UTF-8 produces replacement characters, which the
        pane would happily render as a file that looks corrupt but is not."""
        (workspace / "image.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR")

        with api([]) as client:
            body = client.get("/api/v1/files/content", params={"path": "image.png"}).json()

        assert body["binary"] is True
        assert body["text"] == ""
        assert body["reason"]

    def test_an_oversized_file_reports_its_size_instead_of_freezing_the_tab(
        self, api, workspace
    ) -> None:
        (workspace / "huge.log").write_text("x" * (600 * 1024), encoding="utf-8")

        with api([]) as client:
            body = client.get("/api/v1/files/content", params={"path": "huge.log"}).json()

        assert body["text"] == ""
        assert "600" in body["reason"] or "614400" in body["reason"]


class TestSkillGate:
    """The console's skill pane, and the ladder behind it.

    `candidate → validated → approved → active` is the safety mechanism: an
    agent able to promote its own skill would have no gate at all. A gate
    reachable only from a terminal is one that gets bypassed -- by editing the
    database, or by moving files around.
    """

    @staticmethod
    def _write_candidate(settings, name: str = "demo") -> None:
        root = settings.home / "skills-candidates" / name
        root.mkdir(parents=True, exist_ok=True)
        (root / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: 演示用技能。当用户说要演示时使用。\n---\n\n# {name}\n",
            encoding="utf-8",
        )

    def test_a_candidate_is_visible_before_it_is_promoted(self, api, settings) -> None:
        """The gap this covers: the list used to read the `skills` table, which
        only holds skills promoted at least once. A fresh candidate was
        therefore invisible -- so the one thing that needed a human decision
        was the one thing the pane could not show."""
        self._write_candidate(settings)

        with api([]) as client:
            listed = {s["name"]: s for s in client.get("/api/v1/skills").json()}

        assert listed["demo"]["status"] == "candidate"
        assert listed["demo"]["source"] == "candidate"

    def test_pending_lists_what_is_waiting_on_a_human(self, api, settings) -> None:
        self._write_candidate(settings, "waiting")

        with api([]) as client:
            pending = client.get("/api/v1/skills/pending").json()

        assert [s["name"] for s in pending] == ["waiting"]

    def test_promotion_moves_one_rung_and_is_visible_afterwards(self, api, settings) -> None:
        self._write_candidate(settings, "climber")

        with api([]) as client:
            response = client.post("/api/v1/skills/climber/promote", json={"status": "validated"})
            assert response.status_code == 200, response.text
            after = {s["name"]: s["status"] for s in client.get("/api/v1/skills").json()}

        assert after["climber"] == "validated"

    def test_skipping_a_rung_is_refused(self, api, settings) -> None:
        """Jumping straight to `active` would make the middle rungs decorative,
        and each rung is supposed to be a separate judgement."""
        self._write_candidate(settings, "jumper")

        with api([]) as client:
            response = client.post("/api/v1/skills/jumper/promote", json={"status": "active"})

        assert response.status_code == 400
