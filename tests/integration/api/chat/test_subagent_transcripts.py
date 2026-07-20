"""Chat subagent transcript integration tests."""

from __future__ import annotations

from ._support import *  # noqa: F401,F403

def test_session_subagent_api_lists_details_and_appends_guidance(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    runtime_db = sessions_dir / "runtime.db"
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = SessionStore(sessions_dir)
    runtime_service = RuntimeService(engine=object(), store=RuntimeStore(runtime_db))
    app.state.runtime_service = runtime_service

    asyncio.run(
        app.state.session_store.save_event(
            "session-subagent-a",
            {
                "type": "session_meta",
                "event": "created",
                "session_id": "session-subagent-a",
                "timestamp": datetime.now(tz=UTC).isoformat(),
            },
        )
    )
    asyncio.run(
        app.state.session_store.save_event(
            "session-subagent-b",
            {
                "type": "session_meta",
                "event": "created",
                "session_id": "session-subagent-b",
                "timestamp": datetime.now(tz=UTC).isoformat(),
            },
        )
    )
    asyncio.run(
        runtime_service._store.upsert_subagent_transcript(
            subagent_id="subagent-session-1",
            parent_type="session",
            parent_id="session-subagent-a",
            session_id="session-subagent-a",
            role_id="verifier",
            title="Verifier",
            model_id="gpt-5.4",
            status="running",
            system_prompt="System prompt",
            user_prompt="User prompt",
            prompt_preview="User prompt",
            summary="Initial summary.",
            metadata={"lane": "session"},
        )
    )
    asyncio.run(
        runtime_service._store.append_subagent_transcript_event(
            "subagent-session-1",
            {
                "type": "subagent_started",
                "parent_type": "session",
                "parent_id": "session-subagent-a",
                "subagent_id": "subagent-session-1",
                "role_id": "verifier",
                "status": "running",
                "summary": "Initial summary.",
                "created_at": datetime.now(tz=UTC).isoformat(),
            },
        )
    )
    asyncio.run(
        runtime_service._store.upsert_subagent_transcript(
            subagent_id="subagent-session-2",
            parent_type="session",
            parent_id="session-subagent-b",
            session_id="session-subagent-b",
            role_id="reviewer",
            title="Reviewer",
            status="running",
        )
    )

    with TestClient(app) as client:
        list_response = client.get("/v1/sessions/session-subagent-a/subagents")
        assert list_response.status_code == 200
        list_payload = list_response.json()
        assert len(list_payload) == 1
        assert list_payload[0]["subagent_id"] == "subagent-session-1"
        assert list_payload[0]["session_id"] == "session-subagent-a"
        assert list_payload[0]["role_id"] == "verifier"
        assert list_payload[0]["event_count"] == 1

        detail_response = client.get(
            "/v1/sessions/session-subagent-a/subagents/subagent-session-1"
        )
        assert detail_response.status_code == 200
        detail_payload = detail_response.json()
        assert detail_payload["subagent_id"] == "subagent-session-1"
        assert detail_payload["session_id"] == "session-subagent-a"
        assert detail_payload["system_prompt"] == "System prompt"
        assert detail_payload["user_prompt"] == "User prompt"
        assert len(detail_payload["events"]) == 1

        message_response = client.post(
            "/v1/sessions/session-subagent-a/subagents/subagent-session-1/messages",
            json={
                "role": "user",
                "content": "Please pick this up on the next resume.",
                "project_id": "project-session",
                "workspace_dir": str(tmp_path / "workspace-session"),
                "metadata": {"channel": "session-subagent-chat"},
            },
        )
        assert message_response.status_code == 200
        message_payload = message_response.json()
        assert message_payload["subagent_id"] == "subagent-session-1"
        assert message_payload["event_count"] == 2
        assert message_payload["message_id"]
        assert message_payload["delivery_mode"] == "resume_only"
        assert message_payload["delivery_status"] == "accepted"
        guidance_event = message_payload["events"][-1]
        assert guidance_event["type"] == "subagent_progress"
        assert guidance_event["content"] == "Please pick this up on the next resume."
        assert guidance_event["subagent_id"] == "subagent-session-1"
        assert guidance_event["role_id"] == "verifier"
        assert guidance_event["message_id"] == message_payload["message_id"]
        assert guidance_event["delivery_mode"] == "resume_only"
        assert guidance_event["delivery_status"] == "accepted"
        assert guidance_event["metadata"]["source"] == "session_subagent_message_api"
        assert guidance_event["metadata"]["source_type"] == "subagent_message"
        assert guidance_event["metadata"]["resume_requested"] is True
        assert guidance_event["metadata"]["attachments"] == []
        assert guidance_event["metadata"]["message_id"] == message_payload["message_id"]
        assert guidance_event["metadata"]["delivery_mode"] == "resume_only"
        assert guidance_event["metadata"]["delivery_status"] == "accepted"

        interrupt_response = client.post(
            "/v1/sessions/session-subagent-a/subagents/subagent-session-1/messages",
            json={
                "role": "user",
                "content": "Stop the current direction and inspect the cache first.",
                "delivery_mode": "inject_now",
                "interrupt": True,
                "cancel_current_tool": True,
            },
        )
        assert interrupt_response.status_code == 200
        interrupt_payload = interrupt_response.json()
        assert interrupt_payload["event_count"] == 4
        assert interrupt_payload["message_id"]
        assert interrupt_payload["delivery_mode"] == "inject_now"
        assert interrupt_payload["delivery_status"] == "queued"
        assert interrupt_payload["delivery_reason"] == "tool_cancel_pending"
        assert [event["type"] for event in interrupt_payload["events"][-2:]] == [
            "subagent_progress",
            "subagent_tool_cancel_requested",
        ]
        interrupt_event = interrupt_payload["events"][-1]
        assert interrupt_event["content"] == "Stop the current direction and inspect the cache first."
        assert interrupt_event["message_id"] == interrupt_payload["message_id"]
        assert interrupt_event["delivery_mode"] == "inject_now"
        assert interrupt_event["delivery_status"] == "queued"
        assert interrupt_event["delivery_reason"] == "tool_cancel_pending"
        assert interrupt_event["interrupt"] is True
        assert interrupt_event["cancel_current_tool"] is True
        assert interrupt_event["metadata"]["message_id"] == interrupt_payload["message_id"]
        assert interrupt_event["metadata"]["delivery_mode"] == "inject_now"
        assert interrupt_event["metadata"]["delivery_status"] == "queued"
        assert interrupt_event["metadata"]["delivery_reason"] == "tool_cancel_pending"
        assert interrupt_event["metadata"]["interrupt"] is True
        assert interrupt_event["metadata"]["cancel_current_tool"] is True

        wrong_session_detail = client.get(
            "/v1/sessions/session-subagent-b/subagents/subagent-session-1"
        )
        assert wrong_session_detail.status_code == 404

        wrong_session_message = client.post(
            "/v1/sessions/session-subagent-b/subagents/subagent-session-1/messages",
            json={"role": "user", "content": "This should fail."},
        )
        assert wrong_session_message.status_code == 404

def test_delegated_subagent_resume_replays_session_guidance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[Any] = []

    class _GuidanceCapturingOrchestrator:
        def __init__(self, **_: Any) -> None:
            pass

        async def run(self, request: Any) -> MultiAgentRunResult:
            captured_requests.append(request)
            return MultiAgentRunResult(
                run_id=request.run_id,
                protocol="teacher_student_distill",
                state="succeeded",
                task_input=request.task_input,
                candidates=[],
                selected_candidate_id=None,
                evaluation=None,
                artifacts={"final_answer": "Resumed with durable guidance."},
                events=[],
            )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator", _GuidanceCapturingOrchestrator)
    store = RuntimeStore(tmp_path / "runtime-guidance.db")
    service = RuntimeService(engine=object(), store=store)

    async def _run() -> dict[str, Any] | None:
        await store.create_task_run(
            task_id="delegated-guidance-task",
            input_text="Compare two implementations.",
            session_id="session-guidance-replay",
            project_id=None,
            workspace_dir=None,
            project_workspace_dir=None,
            task_workspace_dir=None,
            task_type="delegated_multi_agent",
            metadata={
                "protocol": "teacher_student_distill",
                "delegated_subagent": {
                    "display_name": "Researcher",
                    "role": "researcher",
                    "instruction": "Compare two implementations.",
                    "objective": "Compare two implementations.",
                    "parent_session_id": "session-guidance-replay",
                    "status": "failed",
                },
            },
        )
        await store.update_task_status("delegated-guidance-task", "failed", error="Needs guidance.")
        await store.upsert_subagent_transcript(
            subagent_id="guidance-researcher",
            parent_type="delegated_task",
            parent_id="delegated-guidance-task",
            session_id="session-guidance-replay",
            role_id="researcher",
            title="Researcher",
            status="failed",
            metadata={"task_id": "delegated-guidance-task"},
        )
        await store.append_subagent_transcript_event(
            "guidance-researcher",
            {
                "type": "subagent_message",
                "subagent_id": "guidance-researcher",
                "target_role_id": "researcher",
                "content": "Prefer the simpler implementation unless risk is materially lower.",
                "metadata": {"source": "session_subagent_message_api"},
                "created_at": datetime.now(tz=UTC).isoformat(),
            },
        )
        summary = await service.resume_task(
            "delegated-guidance-task",
            decision="approve_once",
            reason="test resume",
        )
        job = service._active_jobs.get("delegated-guidance-task")  # noqa: SLF001
        if job is not None:
            await job
        return summary

    summary = asyncio.run(_run())

    assert summary is not None
    assert captured_requests
    request = captured_requests[0]
    assert request.guidance_messages == []
    assert request.role_guidance_messages == {
        "researcher": ["Prefer the simpler implementation unless risk is materially lower."]
    }

def test_chat_subagent_stream_synthesizes_and_persists_subagent_events(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"

    class _FakeSubagentEngine(_FakeEngine):
        async def chat(
            self,
            message: str,
            session_id: str | None = None,
            inference_overrides: dict[str, Any] | None = None,
            project_id: str | None = None,
            workspace_dir: str | None = None,
            selected_skill_ids: list[str] | None = None,
            attachments: list[AttachmentRef] | None = None,
        ) -> AsyncIterator[object]:
            _ = (inference_overrides, project_id, workspace_dir, selected_skill_ids, attachments)
            self.chat_calls.append((message, session_id))
            yield ThinkingEvent(content="Planning delegation")
            yield ToolCallRequestEvent(
                call_id="call-subagent-1",
                tool_name="delegate_subagent_task",
                arguments={
                    "objective": "Research perspective A and perspective B, then compare them.",
                    "suggested_roles": ["researcher_a", "researcher_b"],
                    "suggested_models": {"researcher_a": "gpt-5.4", "researcher_b": "gpt-5.4"},
                    "expected_artifacts": ["comparison memo"],
                },
            )
            yield ToolCallResultEvent(
                call_id="call-subagent-1",
                tool_name="delegate_subagent_task",
                result={
                    "status": "queued",
                    "task_id": "subagent-task-1",
                    "task_type": "delegated_multi_agent",
                    "display_name": "Delegated comparison",
                    "parent_session_id": session_id,
                },
                metadata={
                    "status": "queued",
                    "task_id": "subagent-task-1",
                    "task_type": "delegated_multi_agent",
                    "parent_session_id": session_id,
                },
            )
            yield FinalAnswerEvent(content="I delegated the comparison to background subagents.")

    app, engine = _build_app(engine=_FakeSubagentEngine())
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = SessionStore(sessions_dir)
    runtime_service = RuntimeService(engine=object(), store=RuntimeStore(sessions_dir / "runtime.db"))
    app.state.runtime_service = runtime_service

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/stream",
            json={
                "message": "Use subagents to compare approach A and approach B.",
                "session_id": "session-subagent-stream",
            },
        ) as response:
            chunks = [
                line.removeprefix("data: ")
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

        detail_response = client.get(
            "/v1/sessions/session-subagent-stream/subagents/subagent-task-1"
        )

    assert response.status_code == 200
    assert engine.chat_calls == [
        ("Use subagents to compare approach A and approach B.", "session-subagent-stream")
    ]

    events = [json.loads(chunk) for chunk in chunks]
    event_types = [event["type"] for event in events]
    assert event_types == [
        "thinking",
        "tool_call_request",
        "tool_call_result",
        "subagent_started",
        "subagent_prompt",
        "subagent_progress",
        "final_answer",
    ]
    turn_ids = {event["turn_id"] for event in events}
    assert len(turn_ids) == 1
    turn_id = next(iter(turn_ids))
    assert turn_id

    started_event = events[3]
    prompt_event = events[4]
    progress_event = events[5]
    assert started_event["subagent_id"] == "subagent-task-1"
    assert started_event["parent_type"] == "chat_turn"
    assert started_event["parent_id"] == turn_id
    assert started_event["model_id"] == "gpt-5.4"
    assert started_event["status"] == "queued"
    assert prompt_event["type"] == "subagent_prompt"
    assert prompt_event["status"] == "queued"
    assert prompt_event["user_prompt"] == "Research perspective A and perspective B, then compare them."
    assert progress_event["type"] == "subagent_progress"
    assert progress_event["status"] == "queued"

    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    assert detail_payload["subagent_id"] == "subagent-task-1"
    assert detail_payload["session_id"] == "session-subagent-stream"
    assert detail_payload["parent_turn_id"] == turn_id
    assert detail_payload["status"] == "queued"
    assert [event["type"] for event in detail_payload["events"]] == [
        "subagent_started",
        "subagent_prompt",
        "subagent_progress",
    ]

def test_chat_subagent_stream_projects_live_delegated_runtime_events(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    runtime_service = RuntimeService(engine=object(), store=RuntimeStore(sessions_dir / "runtime.db"))

    class _FakeLiveProjectionEngine(_FakeEngine):
        async def chat(
            self,
            message: str,
            session_id: str | None = None,
            inference_overrides: dict[str, Any] | None = None,
            project_id: str | None = None,
            workspace_dir: str | None = None,
            selected_skill_ids: list[str] | None = None,
            attachments: list[AttachmentRef] | None = None,
        ) -> AsyncIterator[object]:
            _ = (inference_overrides, project_id, workspace_dir, selected_skill_ids, attachments)
            self.chat_calls.append((message, session_id))
            yield ThinkingEvent(content="Planning delegation")
            yield ToolCallRequestEvent(
                call_id="call-live-subagent-1",
                tool_name="delegate_subagent_task",
                arguments={"objective": "Track live delegated progress."},
            )
            yield ToolCallResultEvent(
                call_id="call-live-subagent-1",
                tool_name="delegate_subagent_task",
                result={
                    "status": "queued",
                    "task_id": "subagent-live-task-1",
                    "task_type": "delegated_multi_agent",
                    "display_name": "Live delegated task",
                    "parent_session_id": session_id,
                },
                metadata={
                    "status": "queued",
                    "task_id": "subagent-live-task-1",
                    "task_type": "delegated_multi_agent",
                    "parent_session_id": session_id,
                },
            )
            yield FinalAnswerEvent(content="The live delegated task is running.")
            await asyncio.sleep(0.01)
            runtime_service._publish_delegated_subagent_runtime_event(  # noqa: SLF001
                session_id=session_id,
                task_id="subagent-live-task-1",
                event={
                    "type": "subagent_progress",
                    "subagent_id": "live-worker-1",
                    "parent_type": "delegated_task",
                    "parent_id": "subagent-live-task-1",
                    "role_id": "researcher",
                    "title": "Live Researcher",
                    "status": "running",
                    "summary": "Live researcher found the first relevant source.",
                    "metadata": {
                        "source": "delegate_subagent_task_runtime",
                        "task_id": "subagent-live-task-1",
                    },
                },
            )
            runtime_service._publish_delegated_subagent_runtime_event(  # noqa: SLF001
                session_id=session_id,
                task_id="subagent-live-task-1",
                event={
                    "type": "subagent_completed",
                    "subagent_id": "live-worker-1",
                    "parent_type": "delegated_task",
                    "parent_id": "subagent-live-task-1",
                    "role_id": "researcher",
                    "title": "Live Researcher",
                    "status": "completed",
                    "summary": "Live researcher completed the delegated work.",
                    "content": "Live delegated work complete.",
                    "metadata": {
                        "source": "delegate_subagent_task_runtime",
                        "task_id": "subagent-live-task-1",
                    },
                },
            )

    app, engine = _build_app(engine=_FakeLiveProjectionEngine())
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = SessionStore(sessions_dir)
    app.state.runtime_service = runtime_service

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/stream",
            json={
                "message": "Delegate this and keep streaming live status.",
                "session_id": "session-live-projection",
            },
        ) as response:
            chunks = [
                line.removeprefix("data: ")
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

    assert response.status_code == 200
    assert engine.chat_calls == [
        ("Delegate this and keep streaming live status.", "session-live-projection")
    ]
    events = [json.loads(chunk) for chunk in chunks]
    event_types = [event["type"] for event in events]
    assert event_types == [
        "thinking",
        "tool_call_request",
        "tool_call_result",
        "subagent_started",
        "subagent_prompt",
        "subagent_progress",
        "final_answer",
        "subagent_progress",
        "subagent_completed",
    ]
    live_progress = events[-2]
    live_completed = events[-1]
    assert live_progress["subagent_id"] == "live-worker-1"
    assert live_progress["summary"] == "Live researcher found the first relevant source."
    assert live_completed["subagent_id"] == "live-worker-1"
    assert live_completed["status"] == "completed"
    assert live_completed["content"] == "Live delegated work complete."
    assert live_progress["turn_id"] == live_completed["turn_id"]

def test_chat_subagent_stream_projects_live_delegated_control_events(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    runtime_service = RuntimeService(engine=object(), store=RuntimeStore(sessions_dir / "runtime.db"))

    class _FakeLiveControlEngine(_FakeEngine):
        async def chat(
            self,
            message: str,
            session_id: str | None = None,
            inference_overrides: dict[str, Any] | None = None,
            project_id: str | None = None,
            workspace_dir: str | None = None,
            selected_skill_ids: list[str] | None = None,
            attachments: list[AttachmentRef] | None = None,
        ) -> AsyncIterator[object]:
            _ = (inference_overrides, project_id, workspace_dir, selected_skill_ids, attachments)
            self.chat_calls.append((message, session_id))
            yield ToolCallRequestEvent(
                call_id="call-live-control-1",
                tool_name="delegate_subagent_task",
                arguments={"objective": "Track live delegated control events."},
            )
            yield ToolCallResultEvent(
                call_id="call-live-control-1",
                tool_name="delegate_subagent_task",
                result={
                    "status": "queued",
                    "task_id": "subagent-live-control-task-1",
                    "task_type": "delegated_multi_agent",
                    "display_name": "Live delegated control task",
                    "parent_session_id": session_id,
                },
                metadata={
                    "status": "queued",
                    "task_id": "subagent-live-control-task-1",
                    "task_type": "delegated_multi_agent",
                    "parent_session_id": session_id,
                },
            )
            yield FinalAnswerEvent(content="The delegated control task is running.")
            await asyncio.sleep(0.01)
            runtime_service._publish_delegated_subagent_runtime_event(  # noqa: SLF001
                session_id=session_id,
                task_id="subagent-live-control-task-1",
                event={
                    "type": "subagent_tool_cancel_requested",
                    "subagent_id": "live-control-worker-1",
                    "parent_type": "delegated_task",
                    "parent_id": "subagent-live-control-task-1",
                    "role_id": "researcher",
                    "title": "Live Control Researcher",
                    "status": "running",
                    "message_id": "live-control-message-1",
                    "delivery_mode": "inject_now",
                    "interrupt": True,
                    "cancel_current_tool": True,
                    "reason": "operator_request",
                },
            )
            runtime_service._publish_delegated_subagent_runtime_event(  # noqa: SLF001
                session_id=session_id,
                task_id="subagent-live-control-task-1",
                event={
                    "type": "subagent_tool_cancelled",
                    "subagent_id": "live-control-worker-1",
                    "parent_type": "delegated_task",
                    "parent_id": "subagent-live-control-task-1",
                    "role_id": "researcher",
                    "title": "Live Control Researcher",
                    "status": "cancelled",
                    "message_id": "live-control-message-1",
                    "tool_call_id": "tool-call-live-control-1",
                    "tool_name": "exec_command",
                    "delivery_mode": "inject_now",
                    "delivery_reason": "tool_cancelled",
                    "interrupt": True,
                    "cancel_current_tool": True,
                    "reason": "tool_cancelled",
                },
            )

    app, engine = _build_app(engine=_FakeLiveControlEngine())
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = SessionStore(sessions_dir)
    app.state.runtime_service = runtime_service

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/stream",
            json={
                "message": "Delegate and stream live control events.",
                "session_id": "session-subagent-live-control-stream",
            },
        ) as response:
            chunks = [
                line.removeprefix("data: ")
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

    assert response.status_code == 200
    assert engine.chat_calls == [
        ("Delegate and stream live control events.", "session-subagent-live-control-stream")
    ]
    events = [json.loads(chunk) for chunk in chunks]
    live_events = [
        event
        for event in events
        if event["type"] in {"subagent_tool_cancel_requested", "subagent_tool_cancelled"}
    ]
    assert [event["type"] for event in live_events] == [
        "subagent_tool_cancel_requested",
        "subagent_tool_cancelled",
    ]
    assert live_events[1]["tool_name"] == "exec_command"
    assert live_events[1]["delivery_reason"] == "tool_cancelled"

def test_delegated_subagent_runtime_events_persist_session_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SlowFirstUpsertRuntimeStore(RuntimeStore):
        async def upsert_subagent_transcript(self, **kwargs: Any) -> dict[str, Any]:
            metadata = kwargs.get("metadata")
            if (
                kwargs.get("subagent_id") == "live-researcher-1"
                and isinstance(metadata, dict)
                and metadata.get("task_event_seq") == 1
            ):
                await asyncio.sleep(0.02)
            return await super().upsert_subagent_transcript(**kwargs)

    class _FakeLiveOrchestrator:
        def __init__(self, **_: Any) -> None:
            pass

        async def run(self, request: Any) -> MultiAgentRunResult:
            assert request.runtime_event_callback is not None
            events = [
                MultiAgentRunEvent(
                    run_id="delegated-live-task",
                    seq=1,
                    type="subagent_prompt",
                    payload={
                        "subagent_id": "live-researcher-1",
                        "role_id": "researcher",
                        "title": "Researcher",
                        "model_id": "gpt-5.4",
                        "system_prompt": "Inspect the evidence and report only supported claims.",
                        "user_prompt": "Compare approach A and approach B.",
                        "stage": "prompt",
                    },
                    timestamp="2026-06-30T01:00:00Z",
                ),
                MultiAgentRunEvent(
                    run_id="delegated-live-task",
                    seq=2,
                    type="role_started",
                    payload={
                        "subagent_id": "live-researcher-1",
                        "role_id": "researcher",
                        "title": "Researcher",
                        "model_id": "gpt-5.4",
                        "summary": "Researcher started.",
                    },
                    timestamp="2026-06-30T01:00:01Z",
                ),
                MultiAgentRunEvent(
                    run_id="delegated-live-task",
                    seq=3,
                    type="subagent_tool_call",
                    payload={
                        "subagent_id": "live-researcher-1",
                        "role_id": "researcher",
                        "title": "Researcher",
                        "model_id": "gpt-5.4",
                        "tool_call_id": "tool-call-1",
                        "tool_name": "web_search",
                        "arguments_preview": '{"query": "approach A vs approach B"}',
                        "summary": "Researcher requested web_search.",
                    },
                    timestamp="2026-06-30T01:00:02Z",
                ),
                MultiAgentRunEvent(
                    run_id="delegated-live-task",
                    seq=4,
                    type="subagent_tool_result",
                    payload={
                        "subagent_id": "live-researcher-1",
                        "role_id": "researcher",
                        "title": "Researcher",
                        "model_id": "gpt-5.4",
                        "tool_call_id": "tool-call-1",
                        "tool_name": "web_search",
                        "status": "completed",
                        "summary": "web_search completed.",
                    },
                    timestamp="2026-06-30T01:00:03Z",
                ),
                MultiAgentRunEvent(
                    run_id="delegated-live-task",
                    seq=5,
                    type="runtime_blocked",
                    payload={
                        "role_id": "controller",
                        "blocker_type": "approval",
                        "summary": "Controller is waiting for approval.",
                    },
                    timestamp="2026-06-30T01:00:04Z",
                ),
                MultiAgentRunEvent(
                    run_id="delegated-live-task",
                    seq=6,
                    type="role_completed",
                    payload={
                        "subagent_id": "live-researcher-1",
                        "role_id": "researcher",
                        "title": "Researcher",
                        "model_id": "gpt-5.4",
                        "summary": "Approach A has lower setup cost; approach B has lower operational risk.",
                    },
                    timestamp="2026-06-30T01:00:05Z",
                ),
            ]
            await asyncio.gather(
                *[
                    asyncio.create_task(request.runtime_event_callback(event))
                    for event in events
                ]
            )
            return MultiAgentRunResult(
                run_id="delegated-live-task",
                protocol="teacher_student_distill",
                state="succeeded",
                task_input=request.task_input,
                candidates=[],
                selected_candidate_id=None,
                evaluation=None,
                artifacts={"final_answer": "Live delegated comparison complete."},
                events=events,
            )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator", _FakeLiveOrchestrator)
    store = _SlowFirstUpsertRuntimeStore(tmp_path / "runtime-live.db")
    service = RuntimeService(engine=object(), store=store)

    async def _run() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]:
        await store.create_task_run(
            task_id="delegated-live-task",
            input_text="Compare approach A and approach B.",
            session_id="session-live-subagent",
            project_id=None,
            workspace_dir=None,
            project_workspace_dir=None,
            task_workspace_dir=None,
            task_type="delegated_multi_agent",
            metadata={
                "protocol": "teacher_student_distill",
                "delegated_subagent": {
                    "display_name": "Delegated Researcher",
                    "role": "researcher",
                    "instruction": "Compare approach A and approach B.",
                    "objective": "Compare approach A and approach B.",
                    "parent_session_id": "session-live-subagent",
                    "status": "queued",
                },
            },
        )
        await service._run_task(task_id="delegated-live-task")  # noqa: SLF001
        task = await store.get_task_run("delegated-live-task")
        transcripts = await store.list_subagent_transcripts(session_id="session-live-subagent")
        detail = await store.get_subagent_transcript("live-researcher-1")
        return task or {}, transcripts, detail

    task, transcripts, detail = asyncio.run(_run())

    assert task["status"] == "succeeded"
    assert [item["subagent_id"] for item in transcripts] == ["live-researcher-1"]
    summary = transcripts[0]
    assert summary["parent_type"] == "delegated_task"
    assert summary["parent_id"] == "delegated-live-task"
    assert summary["session_id"] == "session-live-subagent"
    assert summary["status"] == "completed"
    assert summary["event_count"] == 5
    assert detail is not None
    assert detail["system_prompt"] == "Inspect the evidence and report only supported claims."
    assert detail["user_prompt"] == "Compare approach A and approach B."
    assert detail["metadata"]["source"] == "delegate_subagent_task_runtime"
    assert detail["metadata"]["task_id"] == "delegated-live-task"
    assert [event["type"] for event in detail["events"]] == [
        "subagent_prompt",
        "subagent_started",
        "subagent_tool_call",
        "subagent_tool_result",
        "subagent_completed",
    ]
    assert detail["events"][2]["tool_name"] == "web_search"
    assert detail["events"][3]["status"] == "completed"

def test_delegated_subagent_runtime_interruption_delivery_applies_queued_session_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _InterruptAwareOrchestrator:
        def __init__(self, **_: Any) -> None:
            pass

        async def run(self, request: Any) -> MultiAgentRunResult:
            assert request.subagent_message_provider is not None
            assert request.runtime_event_callback is not None
            messages = await request.subagent_message_provider.poll_messages(
                task_id="delegated-interrupt-task",
                subagent_id="interrupt-subagent-1",
                role_id="researcher",
                stage="role::researcher",
                safe_point="before_model_invocation",
            )
            assert [item["content"] for item in messages] == ["Narrow the analysis to option B."]
            message_id = messages[0]["message_id"]
            events = [
                MultiAgentRunEvent(
                    run_id="delegated-interrupt-task",
                    seq=1,
                    type="subagent_message_accepted",
                    payload={
                        "subagent_id": "interrupt-subagent-1",
                        "role_id": "researcher",
                        "message_id": message_id,
                        "delivery_mode": "inject_now",
                        "delivery_status": "accepted",
                        "content": messages[0]["content"],
                    },
                    timestamp="2026-06-30T02:00:00Z",
                ),
                MultiAgentRunEvent(
                    run_id="delegated-interrupt-task",
                    seq=2,
                    type="subagent_message_applied",
                    payload={
                        "subagent_id": "interrupt-subagent-1",
                        "role_id": "researcher",
                        "message_id": message_id,
                        "delivery_mode": "inject_now",
                        "delivery_status": "applied",
                        "content": messages[0]["content"],
                    },
                    timestamp="2026-06-30T02:00:01Z",
                ),
            ]
            for event in events:
                await request.runtime_event_callback(event)
            await request.subagent_message_provider.mark_messages_handled(
                message_ids=[message_id],
                status="applied",
            )
            assert await request.subagent_message_provider.poll_messages(
                task_id="delegated-interrupt-task",
                subagent_id="interrupt-subagent-1",
                role_id="researcher",
                stage="role::researcher",
                safe_point="before_model_invocation",
            ) == []
            return MultiAgentRunResult(
                run_id="delegated-interrupt-task",
                protocol="teacher_student_distill",
                state="succeeded",
                task_input=request.task_input,
                candidates=[],
                selected_candidate_id=None,
                evaluation=None,
                artifacts={"final_answer": "Applied queued guidance."},
                events=events,
            )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator", _InterruptAwareOrchestrator)
    store = RuntimeStore(tmp_path / "runtime-interrupt.db")
    service = RuntimeService(engine=object(), store=store)

    async def _run() -> dict[str, Any] | None:
        await store.create_task_run(
            task_id="delegated-interrupt-task",
            input_text="Compare option A and option B.",
            session_id="session-interrupt-subagent",
            project_id=None,
            workspace_dir=None,
            project_workspace_dir=None,
            task_workspace_dir=None,
            task_type="delegated_multi_agent",
            metadata={
                "protocol": "teacher_student_distill",
                "delegated_subagent": {
                    "display_name": "Delegated Researcher",
                    "role": "researcher",
                    "instruction": "Compare option A and option B.",
                    "objective": "Compare option A and option B.",
                    "parent_session_id": "session-interrupt-subagent",
                    "status": "queued",
                },
            },
        )
        await store.upsert_subagent_transcript(
            subagent_id="interrupt-subagent-1",
            parent_type="delegated_task",
            parent_id="delegated-interrupt-task",
            session_id="session-interrupt-subagent",
            role_id="researcher",
            title="Researcher",
            status="running",
            metadata={"task_id": "delegated-interrupt-task"},
        )
        await store.append_subagent_transcript_event(
            "interrupt-subagent-1",
            {
                "type": "subagent_message",
                "message_id": "queued-message-1",
                "subagent_id": "interrupt-subagent-1",
                "target_role_id": "researcher",
                "content": "Narrow the analysis to option B.",
                "delivery_mode": "inject_now",
                "delivery_status": "queued",
                "metadata": {"source": "session_subagent_message_api"},
                "created_at": "2026-06-30T01:59:59Z",
            },
        )
        await service._run_task(task_id="delegated-interrupt-task")  # noqa: SLF001
        return await store.get_subagent_transcript("interrupt-subagent-1")

    detail = asyncio.run(_run())

    assert detail is not None
    assert [event["type"] for event in detail["events"]] == [
        "subagent_message",
        "subagent_message_accepted",
        "subagent_message_applied",
    ]
    assert detail["events"][1]["message_id"] == "queued-message-1"
    assert detail["events"][2]["delivery_status"] == "applied"

def test_collect_delegated_session_subagent_guidance_marks_approval_resume_context(
    tmp_path: Path,
) -> None:
    store = RuntimeStore(tmp_path / "runtime-approval-guidance.db")
    service = RuntimeService(engine=object(), store=store)

    async def _run() -> tuple[list[str], dict[str, list[str]]]:
        await store.upsert_subagent_transcript(
            subagent_id="approval-guidance-subagent-1",
            parent_type="delegated_task",
            parent_id="delegated-approval-guidance-task",
            session_id="session-approval-guidance",
            role_id="researcher",
            title="Researcher",
            status="running",
            metadata={"task_id": "delegated-approval-guidance-task"},
        )
        await store.append_subagent_transcript_event(
            "approval-guidance-subagent-1",
            {
                "type": "subagent_message",
                "message_id": "approval-guidance-message-1",
                "subagent_id": "approval-guidance-subagent-1",
                "target_role_id": "researcher",
                "content": "Inspect the failure mode after approval resumes.",
                "delivery_mode": "inject_now",
                "delivery_status": "queued",
            },
        )
        await store.append_subagent_transcript_event(
            "approval-guidance-subagent-1",
            {
                "type": "subagent_message_deferred",
                "message_id": "approval-guidance-message-1",
                "subagent_id": "approval-guidance-subagent-1",
                "role_id": "researcher",
                "delivery_mode": "inject_now",
                "delivery_status": "deferred",
                "delivery_reason": "approval_pending",
                "reason": "approval_pending",
            },
        )
        return await service._collect_delegated_session_subagent_guidance(  # noqa: SLF001
            session_id="session-approval-guidance",
            task_id="delegated-approval-guidance-task",
        )

    guidance_messages, role_guidance_messages = asyncio.run(_run())

    assert guidance_messages == []
    assert role_guidance_messages == {
        "researcher": [
            "Approval-resume guidance: Inspect the failure mode after approval resumes."
        ]
    }

def test_chat_subagent_stream_preserves_approval_required_metadata(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"

    class _ApprovalSubagentEngine(_FakeEngine):
        async def chat(
            self,
            message: str,
            session_id: str | None = None,
            inference_overrides: dict[str, Any] | None = None,
            project_id: str | None = None,
            workspace_dir: str | None = None,
            selected_skill_ids: list[str] | None = None,
            attachments: list[AttachmentRef] | None = None,
        ) -> AsyncIterator[object]:
            _ = (inference_overrides, project_id, workspace_dir, selected_skill_ids, attachments)
            self.chat_calls.append((message, session_id))
            yield ToolCallRequestEvent(
                call_id="call-subagent-approval",
                tool_name="delegate_subagent_task",
                arguments={"objective": "Run a bounded smoke command after approval."},
            )
            yield ToolCallResultEvent(
                call_id="call-subagent-approval",
                tool_name="delegate_subagent_task",
                result={
                    "status": "awaiting_approval",
                    "task_id": "subagent-task-approval",
                    "task_type": "delegated_multi_agent",
                    "display_name": "Delegated approval probe",
                    "parent_session_id": session_id,
                    "approval_state": {
                        "status": "waiting_approval",
                        "approval_ids": ["exec-approval-chat-1"],
                        "tool_names": ["exec_command"],
                        "pending_approvals": [
                            {
                                "approval_id": "exec-approval-chat-1",
                                "tool_name": "exec_command",
                                "reason": "Exec command requires approval.",
                                "approval_kind": "exec",
                                "approval_scope": "workspace",
                                "replay_safe": False,
                                "security_decision": "require_approval",
                                "policy_source": "runtime_policy",
                                "allowed_decisions": ["approve_once", "reject"],
                            }
                        ],
                    },
                },
                metadata={
                    "status": "awaiting_approval",
                    "task_id": "subagent-task-approval",
                    "task_type": "delegated_multi_agent",
                    "parent_session_id": session_id,
                    "approval_state": {
                        "status": "waiting_approval",
                        "approval_ids": ["exec-approval-chat-1"],
                        "tool_names": ["exec_command"],
                        "pending_approvals": [
                            {
                                "approval_id": "exec-approval-chat-1",
                                "tool_name": "exec_command",
                                "reason": "Exec command requires approval.",
                                "approval_kind": "exec",
                                "approval_scope": "workspace",
                                "replay_safe": False,
                                "security_decision": "require_approval",
                                "policy_source": "runtime_policy",
                                "allowed_decisions": ["approve_once", "reject"],
                            }
                        ],
                    },
                },
            )
            yield FinalAnswerEvent(content="The delegated subagent is waiting for approval.")

    app, _ = _build_app(engine=_ApprovalSubagentEngine())
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = SessionStore(sessions_dir)
    app.state.runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(sessions_dir / "runtime-approval.db"),
    )

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/stream",
            json={
                "message": "Delegate a task that will need approval.",
                "session_id": "session-subagent-approval",
            },
        ) as response:
            chunks = [
                line.removeprefix("data: ")
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

        detail_response = client.get(
            "/v1/sessions/session-subagent-approval/subagents/subagent-task-approval"
        )

    assert response.status_code == 200
    events = [json.loads(chunk) for chunk in chunks]
    event_types = [event["type"] for event in events]
    assert "subagent_tool_result" in event_types
    assert "runtime_blocked" in event_types
    tool_result_event = next(event for event in events if event["type"] == "subagent_tool_result")
    assert tool_result_event["status"] == "approval_required"
    assert tool_result_event["metadata"]["approval_id"] == "exec-approval-chat-1"
    assert tool_result_event["metadata"]["approval_scope"] == "workspace"
    assert tool_result_event["metadata"]["replay_safe"] is False
    assert tool_result_event["metadata"]["security_decision"] == "require_approval"
    runtime_blocked = next(event for event in events if event["type"] == "runtime_blocked")
    assert runtime_blocked["approval_ids"] == ["exec-approval-chat-1"]
    assert runtime_blocked["tool_names"] == ["exec_command"]
    assert runtime_blocked["recommended_action"] == "resolve_approval"
    assert runtime_blocked["pending_approvals"][0]["approval_kind"] == "exec"

    assert detail_response.status_code == 200
    detail_payload = detail_response.json()
    detail_events = detail_payload["events"]
    assert [event["type"] for event in detail_events] == [
        "subagent_started",
        "subagent_prompt",
        "subagent_tool_result",
        "runtime_blocked",
    ]
    assert detail_events[2]["metadata"]["approval_scope"] == "workspace"
    assert detail_events[3]["metadata"]["pending_approvals"][0]["security_decision"] == "require_approval"
