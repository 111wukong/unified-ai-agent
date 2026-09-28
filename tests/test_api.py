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

from unified_agent.api.app import Service, create_app  # noqa: E402
from unified_agent.agent.factory import build_agent  # noqa: E402

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
        assert "cannot be pre-approved" in response.json()["detail"]

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
            # The SSE stream does not carry the task id, so look it up over
            # REST. Noted as a gap: a client that wants to resume needs it.
            tasks = client.get("/api/v1/tasks").json()
            task_id = tasks[0]["id"]
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
        from unified_agent.api.agui import AgUiEncoder
        from unified_agent.observability.events import Event, EventType

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
        from unified_agent.agent.factory import build_agent

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

    def test_the_console_and_its_assets_are_not_guarded(self, guarded) -> None:
        """The console must load before it can present a token."""
        client = guarded(token="s3cret", hosts=["testserver"])
        with client:
            assert client.get("/", headers={"Host": "testserver"}).status_code == 200
            assert (
                client.get("/console/app.js", headers={"Host": "testserver"}).status_code == 200
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
            response = client.get("/api/v1/health", headers={"X-UAA-Token": "nope"})
        assert response.status_code == 403

    def test_header_token_is_accepted(self, guarded) -> None:
        client = guarded(token="s3cret")
        with client:
            response = client.get("/api/v1/health", headers={"X-UAA-Token": "s3cret"})
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
        """`uaa serve` without --token stays usable; the Host check still runs."""
        client = guarded(token=None)
        with client:
            assert client.get("/api/v1/health").status_code == 200
            assert client.get("/api/v1/health", headers={"Host": "evil.example"}).status_code == 403

    def test_health_reports_whether_a_token_is_required(self, guarded) -> None:
        with guarded(token=None) as client:
            assert client.get("/api/v1/health").json()["token_required"] is False
