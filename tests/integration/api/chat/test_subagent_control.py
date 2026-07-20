"""Chat subagent control integration tests."""

from __future__ import annotations

from ._support import *  # noqa: F401,F403


def test_session_subagent_api_cancel_resume_actions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
            "session-subagent-actions",
            {
                "type": "session_meta",
                "event": "created",
                "session_id": "session-subagent-actions",
                "timestamp": datetime.now(tz=UTC).isoformat(),
            },
        )
    )
    asyncio.run(
        runtime_service._store.upsert_subagent_transcript(
            subagent_id="subagent-cancel-1",
            parent_type="session",
            parent_id="session-subagent-actions",
            session_id="session-subagent-actions",
            role_id="researcher",
            title="Researcher",
            status="running",
            metadata={"task_id": "delegated-cancel-task"},
        )
    )
    asyncio.run(
        runtime_service._store.upsert_subagent_transcript(
            subagent_id="subagent-resume-1",
            parent_type="delegated_task",
            parent_id="delegated-resume-task",
            session_id="session-subagent-actions",
            role_id="reviewer",
            title="Reviewer",
            status="running",
            metadata={},
        )
    )

    cancel_calls: list[str] = []
    resume_calls: list[dict[str, Any]] = []

    async def _cancel_task(task_id: str) -> dict[str, Any] | None:
        cancel_calls.append(task_id)
        return {"id": task_id, "status": "cancelled"}

    async def _resume_task(
        task_id: str,
        *,
        decision: str,
        reason: str | None = None,
        rule: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        resume_calls.append(
            {
                "task_id": task_id,
                "decision": decision,
                "reason": reason,
                "rule": rule,
            }
        )
        return {"id": task_id, "status": "running"}

    monkeypatch.setattr(runtime_service, "cancel_task", _cancel_task)
    monkeypatch.setattr(runtime_service, "resume_task", _resume_task)

    with TestClient(app) as client:
        cancel_response = client.post(
            "/v1/sessions/session-subagent-actions/subagents/subagent-cancel-1/cancel"
        )
        resume_response = client.post(
            "/v1/sessions/session-subagent-actions/subagents/subagent-resume-1/resume",
            json={
                "role": "user",
                "content": "Resume after applying this guidance.",
                "metadata": {"channel": "session-subagent-action-test"},
            },
        )

    assert cancel_response.status_code == 200
    cancel_payload = cancel_response.json()
    assert cancel_calls == ["delegated-cancel-task"]
    assert cancel_payload["type"] == "session_subagent_action"
    assert cancel_payload["action"] == "cancel"
    assert cancel_payload["task_id"] == "delegated-cancel-task"
    assert cancel_payload["task"]["status"] == "cancelled"
    assert cancel_payload["transcript"]["subagent_id"] == "subagent-cancel-1"

    assert resume_response.status_code == 200
    resume_payload = resume_response.json()
    assert resume_calls == [
        {
            "task_id": "delegated-resume-task",
            "decision": "approve_once",
            "reason": "Session subagent resume requested",
            "rule": None,
        }
    ]
    assert resume_payload["action"] == "resume"
    assert resume_payload["task_id"] == "delegated-resume-task"
    assert resume_payload["task"]["status"] == "running"
    assert resume_payload["transcript"]["event_count"] == 1
    guidance_event = resume_payload["transcript"]["events"][-1]
    assert guidance_event["type"] == "subagent_progress"
    assert guidance_event["content"] == "Resume after applying this guidance."
    assert guidance_event["metadata"]["source"] == "session_subagent_message_api"
    assert guidance_event["metadata"]["resume_requested"] is True

def test_session_subagent_api_cancel_resume_requires_task_link(tmp_path: Path) -> None:
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
            "session-subagent-unlinked",
            {
                "type": "session_meta",
                "event": "created",
                "session_id": "session-subagent-unlinked",
                "timestamp": datetime.now(tz=UTC).isoformat(),
            },
        )
    )
    asyncio.run(
        runtime_service._store.upsert_subagent_transcript(
            subagent_id="subagent-unlinked-1",
            parent_type="session",
            parent_id="session-subagent-unlinked",
            session_id="session-subagent-unlinked",
            role_id="writer",
            title="Writer",
            status="running",
            metadata={},
        )
    )

    with TestClient(app) as client:
        cancel_response = client.post(
            "/v1/sessions/session-subagent-unlinked/subagents/subagent-unlinked-1/cancel"
        )
        resume_response = client.post(
            "/v1/sessions/session-subagent-unlinked/subagents/subagent-unlinked-1/resume"
        )

    assert cancel_response.status_code == 409
    assert cancel_response.json()["detail"] == "Session subagent is not linked to a delegated task"
    assert resume_response.status_code == 409
    assert resume_response.json()["detail"] == "Session subagent is not linked to a delegated task"

def test_subagent_interrupt_cancel_message_emits_protocol_events() -> None:
    class _CancelProvider:
        def __init__(self) -> None:
            self.handled: list[tuple[list[str], str, str | None]] = []

        async def poll_messages(self, **_kwargs: Any) -> list[dict[str, Any]]:
            return [
                {
                    "message_id": "cancel-message-1",
                    "subagent_id": "run-cancel:researcher:role-researcher",
                    "role_id": "researcher",
                    "content": "Stop the current tool and inspect the cache.",
                    "delivery_mode": "inject_now",
                    "delivery_status": "queued",
                    "delivery_reason": "tool_cancel_pending",
                    "interrupt": True,
                    "cancel_current_tool": True,
                }
            ]

        async def mark_messages_handled(
            self,
            *,
            message_ids: list[str],
            status: str,
            reason: str | None = None,
        ) -> None:
            self.handled.append((message_ids, status, reason))

    provider = _CancelProvider()
    emitted: list[tuple[str, dict[str, Any]]] = []
    orchestrator = MultiAgentOrchestrator(engine=object())
    orchestrator._current_run_id = "run-cancel"  # noqa: SLF001
    orchestrator._subagent_message_provider = provider  # noqa: SLF001
    orchestrator._emit_runtime_event = lambda event_type, payload: emitted.append(  # noqa: SLF001
        (event_type, payload)
    )

    applied = asyncio.run(
        orchestrator._poll_subagent_messages_for_role(  # noqa: SLF001
            role_id="researcher",
            subagent_id="run-cancel:researcher:role-researcher",
            stage="role::researcher",
            safe_point="after_model_invocation",
            delivery="defer",
            reason="generation_in_progress",
        )
    )

    assert applied == []
    assert provider.handled == [(["cancel-message-1"], "deferred", "generation_in_progress")]
    assert [event_type for event_type, _payload in emitted] == [
        "subagent_message_accepted",
        "subagent_interrupted",
        "subagent_tool_cancel_requested",
        "subagent_message_deferred",
        "subagent_tool_cancel_deferred",
    ]
    assert emitted[1][1]["interrupt"] is True
    assert emitted[2][1]["cancel_current_tool"] is True
    assert emitted[-1][1]["reason"] == "no_active_tool"

def test_subagent_cancel_current_tool_applies_after_true_cancellation() -> None:
    class _CancellableToolInvocationEngine:
        async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResult:
            controller = request.active_tool_controller
            assert controller is not None
            await controller.activate_tool(
                tool_call_id="tool-call-cancelled",
                tool_name="exec_command",
                cancellable=False,
            )
            cancelled = asyncio.Event()

            async def _cancel() -> ToolCancellationResult:
                cancelled.set()
                return ToolCancellationResult(
                    cancelled=True,
                    reason="tool_cancelled",
                    tool_call_id="tool-call-cancelled",
                    tool_name="exec_command",
                    session_id="exec-session-1",
                )

            await controller.bind_cancel_callback(
                session_id="exec-session-1",
                callback=_cancel,
            )

            applied_messages: list[dict[str, Any]] = []
            for _ in range(100):
                if cancelled.is_set():
                    applied_messages = await controller.consume_post_tool_messages()
                    if applied_messages:
                        break
                await asyncio.sleep(0.01)
            await controller.finish_tool()
            return AgentInvocationResult(
                content=(
                    str(applied_messages[0].get("content") or "")
                    if applied_messages
                    else "Tool was cancelled."
                ),
                events=[
                    ToolCallRequestEvent(
                        call_id="tool-call-cancelled",
                        tool_name="exec_command",
                        arguments={"command": "pytest -q"},
                    ),
                    ToolCallResultEvent(
                        call_id="tool-call-cancelled",
                        tool_name="exec_command",
                        result={"status": "cancelled"},
                        metadata={"status": "cancelled"},
                    ),
                ],
                diagnostics=AgentInvocationDiagnostics(
                    execution_profile="subagent_readonly",
                    tool_mode="auto",
                ),
            )

    class _CancelProvider:
        def __init__(self) -> None:
            self.handled: list[tuple[list[str], str, str | None]] = []
            self._handled_ids: set[str] = set()

        async def poll_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
            if kwargs.get("safe_point") != "during_active_tool":
                return []
            if "cancel-success-1" in self._handled_ids:
                return []
            return [
                {
                    "message_id": "cancel-success-1",
                    "subagent_id": "run-cancel-success:researcher:role-researcher",
                    "role_id": "researcher",
                    "content": "Stop the current command and inspect the cache.",
                    "delivery_mode": "inject_now",
                    "delivery_status": "queued",
                    "delivery_reason": "tool_cancel_pending",
                    "interrupt": True,
                    "cancel_current_tool": True,
                }
            ]

        async def mark_messages_handled(
            self,
            *,
            message_ids: list[str],
            status: str,
            reason: str | None = None,
        ) -> None:
            self._handled_ids.update(message_ids)
            self.handled.append((message_ids, status, reason))

    provider = _CancelProvider()
    emitted: list[tuple[str, dict[str, Any]]] = []
    orchestrator = MultiAgentOrchestrator(engine=_CancellableToolInvocationEngine())
    orchestrator._current_run_id = "run-cancel-success"  # noqa: SLF001
    orchestrator._subagent_message_provider = provider  # noqa: SLF001
    orchestrator._emit_runtime_event = lambda event_type, payload: emitted.append(  # noqa: SLF001
        (event_type, payload)
    )

    content, _diagnostics = asyncio.run(
        orchestrator._invoke_configured_text(  # noqa: SLF001
            model_id="gpt-test",
            system_prompt="System",
            user_prompt="Inspect the current command.",
            temperature=0.2,
            max_tokens=64,
            execution_profile="subagent_readonly",
            tool_mode="auto",
            system_prompt_addendum="System",
            session_scope="role::researcher",
            runtime_role_id="researcher",
            runtime_title="Researcher",
            runtime_stage="role::researcher",
            runtime_subagent_id="run-cancel-success:researcher:role-researcher",
        )
    )

    assert content == "Stop the current command and inspect the cache."
    assert provider.handled == [(["cancel-success-1"], "applied", None)]
    assert [event_type for event_type, _payload in emitted[:5]] == [
        "subagent_message_accepted",
        "subagent_interrupted",
        "subagent_tool_cancel_requested",
        "subagent_tool_cancelled",
        "subagent_message_applied",
    ]
    assert emitted[3][1]["tool_name"] == "exec_command"
    assert emitted[3][1]["tool_call_id"] == "tool-call-cancelled"
    assert orchestrator._role_guidance_messages["researcher"] == [  # noqa: SLF001
        "Stop the current command and inspect the cache."
    ]

def test_subagent_interrupt_restarts_after_true_mid_generation_cancellation() -> None:
    class _InterruptibleInvocationEngine:
        def __init__(self) -> None:
            self.invoke_calls = 0
            self.messages: list[str] = []

        def supports_mid_generation_cancellation(self, *, model_id: str | None = None) -> bool:
            _ = model_id
            return True

        async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResult:
            self.invoke_calls += 1
            self.messages.append(request.message)
            if self.invoke_calls == 1:
                await asyncio.sleep(5)
            assert "Live subagent guidance:" in request.message
            assert "Stop the current direction and inspect the cache first." in request.message
            return AgentInvocationResult(
                content="Restarted after the interrupt.",
                events=[],
                diagnostics=AgentInvocationDiagnostics(
                    execution_profile="subagent_readonly",
                    tool_mode="auto",
                ),
            )

    class _InterruptProvider:
        def __init__(self) -> None:
            self.handled: list[tuple[list[str], str, str | None]] = []
            self._handled_ids: set[str] = set()

        async def poll_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
            if kwargs.get("safe_point") != "during_model_invocation":
                return []
            if "midgen-cancel-1" in self._handled_ids:
                return []
            return [
                {
                    "message_id": "midgen-cancel-1",
                    "subagent_id": "run-midgen-cancel:researcher:role-researcher",
                    "role_id": "researcher",
                    "content": "Stop the current direction and inspect the cache first.",
                    "delivery_mode": "inject_now",
                    "delivery_status": "queued",
                    "delivery_reason": "interrupt_pending",
                    "interrupt": True,
                    "cancel_current_tool": True,
                }
            ]

        async def mark_messages_handled(
            self,
            *,
            message_ids: list[str],
            status: str,
            reason: str | None = None,
        ) -> None:
            self._handled_ids.update(message_ids)
            self.handled.append((message_ids, status, reason))

    engine = _InterruptibleInvocationEngine()
    provider = _InterruptProvider()
    emitted: list[tuple[str, dict[str, Any]]] = []
    orchestrator = MultiAgentOrchestrator(engine=engine)
    orchestrator._current_run_id = "run-midgen-cancel"  # noqa: SLF001
    orchestrator._subagent_message_provider = provider  # noqa: SLF001
    orchestrator._emit_runtime_event = lambda event_type, payload: emitted.append(  # noqa: SLF001
        (event_type, payload)
    )

    content, _diagnostics = asyncio.run(
        orchestrator._invoke_configured_text(  # noqa: SLF001
            model_id="gpt-test",
            system_prompt="System",
            user_prompt="Inspect the current direction.",
            temperature=0.2,
            max_tokens=64,
            execution_profile="subagent_readonly",
            tool_mode="auto",
            system_prompt_addendum="System",
            session_scope="role::researcher",
            runtime_role_id="researcher",
            runtime_title="Researcher",
            runtime_stage="role::researcher",
            runtime_subagent_id="run-midgen-cancel:researcher:role-researcher",
        )
    )

    assert content == "Restarted after the interrupt."
    assert engine.invoke_calls == 2
    assert engine.messages[0] == "Inspect the current direction."
    assert provider.handled == [(["midgen-cancel-1"], "applied", None)]
    assert [event_type for event_type, _payload in emitted[:5]] == [
        "subagent_message_accepted",
        "subagent_interrupted",
        "subagent_tool_cancel_requested",
        "subagent_tool_cancelled",
        "subagent_message_applied",
    ]
    assert emitted[3][1]["reason"] == "model_invocation_cancelled"
    assert emitted[3][1]["status"] == "running"
    assert emitted[4][1]["delivery_status"] == "applied"
    assert orchestrator._role_guidance_messages["researcher"] == [  # noqa: SLF001
        "Stop the current direction and inspect the cache first."
    ]

def test_subagent_interrupt_restarts_mid_generation_without_tool_cancel_event() -> None:
    class _InterruptibleInvocationEngine:
        def __init__(self) -> None:
            self.invoke_calls = 0

        def supports_mid_generation_cancellation(self, *, model_id: str | None = None) -> bool:
            _ = model_id
            return True

        async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResult:
            self.invoke_calls += 1
            if self.invoke_calls == 1:
                await asyncio.sleep(5)
            assert "Live subagent guidance:" in request.message
            assert "Restart with the cache inspection first." in request.message
            return AgentInvocationResult(
                content="Restarted after interrupt-only guidance.",
                events=[],
                diagnostics=AgentInvocationDiagnostics(
                    execution_profile="subagent_readonly",
                    tool_mode="auto",
                ),
            )

    class _InterruptProvider:
        def __init__(self) -> None:
            self.handled: list[tuple[list[str], str, str | None]] = []
            self._handled_ids: set[str] = set()

        async def poll_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
            if kwargs.get("safe_point") != "during_model_invocation":
                return []
            if "midgen-interrupt-only-1" in self._handled_ids:
                return []
            return [
                {
                    "message_id": "midgen-interrupt-only-1",
                    "subagent_id": "run-midgen-interrupt-only:researcher:role-researcher",
                    "role_id": "researcher",
                    "content": "Restart with the cache inspection first.",
                    "delivery_mode": "inject_now",
                    "delivery_status": "queued",
                    "delivery_reason": "interrupt_pending",
                    "interrupt": True,
                    "cancel_current_tool": False,
                }
            ]

        async def mark_messages_handled(
            self,
            *,
            message_ids: list[str],
            status: str,
            reason: str | None = None,
        ) -> None:
            self._handled_ids.update(message_ids)
            self.handled.append((message_ids, status, reason))

    engine = _InterruptibleInvocationEngine()
    provider = _InterruptProvider()
    emitted: list[tuple[str, dict[str, Any]]] = []
    orchestrator = MultiAgentOrchestrator(engine=engine)
    orchestrator._current_run_id = "run-midgen-interrupt-only"  # noqa: SLF001
    orchestrator._subagent_message_provider = provider  # noqa: SLF001
    orchestrator._emit_runtime_event = lambda event_type, payload: emitted.append(  # noqa: SLF001
        (event_type, payload)
    )

    content, _diagnostics = asyncio.run(
        orchestrator._invoke_configured_text(  # noqa: SLF001
            model_id="gpt-test",
            system_prompt="System",
            user_prompt="Inspect the current direction.",
            temperature=0.2,
            max_tokens=64,
            execution_profile="subagent_readonly",
            tool_mode="auto",
            system_prompt_addendum="System",
            session_scope="role::researcher",
            runtime_role_id="researcher",
            runtime_title="Researcher",
            runtime_stage="role::researcher",
            runtime_subagent_id="run-midgen-interrupt-only:researcher:role-researcher",
        )
    )

    assert content == "Restarted after interrupt-only guidance."
    assert engine.invoke_calls == 2
    assert provider.handled == [(["midgen-interrupt-only-1"], "applied", None)]
    assert [event_type for event_type, _payload in emitted[:3]] == [
        "subagent_message_accepted",
        "subagent_interrupted",
        "subagent_message_applied",
    ]
    assert not any(event_type.startswith("subagent_tool_cancel") for event_type, _payload in emitted)

def test_subagent_interrupt_restarts_after_late_mid_generation_cancellation() -> None:
    class _LateCancelInvocationEngine:
        def __init__(self) -> None:
            self.invoke_calls = 0

        def supports_mid_generation_cancellation(self, *, model_id: str | None = None) -> bool:
            _ = model_id
            return True

        async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResult:
            self.invoke_calls += 1
            if self.invoke_calls == 1:
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    await asyncio.sleep(0.35)
                    raise
            assert "Live subagent guidance:" in request.message
            assert "Inspect the cache before continuing." in request.message
            return AgentInvocationResult(
                content="Restarted after late cancellation.",
                events=[],
                diagnostics=AgentInvocationDiagnostics(
                    execution_profile="subagent_readonly",
                    tool_mode="auto",
                ),
            )

    class _InterruptProvider:
        def __init__(self) -> None:
            self.handled: list[tuple[list[str], str, str | None]] = []
            self._handled_ids: set[str] = set()

        async def poll_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
            if kwargs.get("safe_point") != "during_model_invocation":
                return []
            if "midgen-late-cancel-1" in self._handled_ids:
                return []
            return [
                {
                    "message_id": "midgen-late-cancel-1",
                    "subagent_id": "run-midgen-late-cancel:researcher:role-researcher",
                    "role_id": "researcher",
                    "content": "Inspect the cache before continuing.",
                    "delivery_mode": "inject_now",
                    "delivery_status": "queued",
                    "delivery_reason": "interrupt_pending",
                    "interrupt": True,
                    "cancel_current_tool": True,
                }
            ]

        async def mark_messages_handled(
            self,
            *,
            message_ids: list[str],
            status: str,
            reason: str | None = None,
        ) -> None:
            self._handled_ids.update(message_ids)
            self.handled.append((message_ids, status, reason))

    engine = _LateCancelInvocationEngine()
    provider = _InterruptProvider()
    emitted: list[tuple[str, dict[str, Any]]] = []
    orchestrator = MultiAgentOrchestrator(engine=engine)
    orchestrator._current_run_id = "run-midgen-late-cancel"  # noqa: SLF001
    orchestrator._subagent_message_provider = provider  # noqa: SLF001
    orchestrator._emit_runtime_event = lambda event_type, payload: emitted.append(  # noqa: SLF001
        (event_type, payload)
    )

    content, _diagnostics = asyncio.run(
        orchestrator._invoke_configured_text(  # noqa: SLF001
            model_id="gpt-test",
            system_prompt="System",
            user_prompt="Inspect the current direction.",
            temperature=0.2,
            max_tokens=64,
            execution_profile="subagent_readonly",
            tool_mode="auto",
            system_prompt_addendum="System",
            session_scope="role::researcher",
            runtime_role_id="researcher",
            runtime_title="Researcher",
            runtime_stage="role::researcher",
            runtime_subagent_id="run-midgen-late-cancel:researcher:role-researcher",
        )
    )

    assert content == "Restarted after late cancellation."
    assert engine.invoke_calls == 2
    assert provider.handled == [(["midgen-late-cancel-1"], "applied", None)]
    assert [event_type for event_type, _payload in emitted[:5]] == [
        "subagent_message_accepted",
        "subagent_interrupted",
        "subagent_tool_cancel_requested",
        "subagent_tool_cancelled",
        "subagent_message_applied",
    ]
    assert emitted[3][1]["reason"] == "model_invocation_cancelled"

def test_subagent_interrupt_defers_when_mid_generation_cancel_does_not_unwind() -> None:
    class _StickyInvocationEngine:
        def __init__(self) -> None:
            self.invoke_calls = 0
            self.messages: list[str] = []

        def supports_mid_generation_cancellation(self, *, model_id: str | None = None) -> bool:
            _ = model_id
            return True

        async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResult:
            self.invoke_calls += 1
            self.messages.append(request.message)
            try:
                await asyncio.sleep(0.15)
            except asyncio.CancelledError:
                await asyncio.sleep(0.05)
            return AgentInvocationResult(
                content="The original generation completed.",
                events=[],
                diagnostics=AgentInvocationDiagnostics(
                    execution_profile="subagent_readonly",
                    tool_mode="auto",
                ),
            )

    class _InterruptProvider:
        def __init__(self) -> None:
            self.handled: list[tuple[list[str], str, str | None]] = []
            self._handled_ids: set[str] = set()

        async def poll_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
            if kwargs.get("safe_point") != "during_model_invocation":
                return []
            if "midgen-cancel-fallback-1" in self._handled_ids:
                return []
            return [
                {
                    "message_id": "midgen-cancel-fallback-1",
                    "subagent_id": "run-midgen-fallback:researcher:role-researcher",
                    "role_id": "researcher",
                    "content": "Stop the current direction and inspect the cache first.",
                    "delivery_mode": "inject_now",
                    "delivery_status": "queued",
                    "delivery_reason": "interrupt_pending",
                    "interrupt": True,
                    "cancel_current_tool": True,
                }
            ]

        async def mark_messages_handled(
            self,
            *,
            message_ids: list[str],
            status: str,
            reason: str | None = None,
        ) -> None:
            self._handled_ids.update(message_ids)
            self.handled.append((message_ids, status, reason))

    engine = _StickyInvocationEngine()
    provider = _InterruptProvider()
    emitted: list[tuple[str, dict[str, Any]]] = []
    orchestrator = MultiAgentOrchestrator(engine=engine)
    orchestrator._current_run_id = "run-midgen-fallback"  # noqa: SLF001
    orchestrator._subagent_message_provider = provider  # noqa: SLF001
    orchestrator._emit_runtime_event = lambda event_type, payload: emitted.append(  # noqa: SLF001
        (event_type, payload)
    )

    content, _diagnostics = asyncio.run(
        orchestrator._invoke_configured_text(  # noqa: SLF001
            model_id="gpt-test",
            system_prompt="System",
            user_prompt="Inspect the current direction.",
            temperature=0.2,
            max_tokens=64,
            execution_profile="subagent_readonly",
            tool_mode="auto",
            system_prompt_addendum="System",
            session_scope="role::researcher",
            runtime_role_id="researcher",
            runtime_title="Researcher",
            runtime_stage="role::researcher",
            runtime_subagent_id="run-midgen-fallback:researcher:role-researcher",
        )
    )

    assert content == "The original generation completed."
    assert engine.invoke_calls == 1
    assert engine.messages == ["Inspect the current direction."]
    assert provider.handled == [
        (["midgen-cancel-fallback-1"], "deferred", "generation_in_progress")
    ]
    protocol_events = [
        (event_type, payload)
        for event_type, payload in emitted
        if event_type
        in {
            "subagent_message_accepted",
            "subagent_interrupted",
            "subagent_tool_cancel_requested",
            "subagent_message_deferred",
            "subagent_tool_cancel_deferred",
            "subagent_tool_cancelled",
        }
    ]
    assert [event_type for event_type, _payload in protocol_events] == [
        "subagent_message_accepted",
        "subagent_interrupted",
        "subagent_tool_cancel_requested",
        "subagent_message_deferred",
        "subagent_tool_cancel_deferred",
    ]
    assert protocol_events[3][1]["reason"] == "generation_in_progress"
    assert protocol_events[4][1]["reason"] == "no_active_tool"
    assert "researcher" not in orchestrator._role_guidance_messages  # noqa: SLF001

def test_subagent_interrupt_restarts_during_configured_model_fallback() -> None:
    class _FallbackOnlyEngine:
        def __init__(self) -> None:
            self.generate_calls = 0

        def supports_mid_generation_cancellation(self, *, model_id: str | None = None) -> bool:
            _ = model_id
            return True

        async def generate_with_configured_model(
            self,
            *,
            model_id: str,
            messages: list[Any],
            temperature: float = 0.2,
            max_tokens: int = 1024,
            reasoning_effort: str | None = None,
        ) -> GenerationResult:
            _ = (model_id, temperature, max_tokens, reasoning_effort)
            self.generate_calls += 1
            user_prompt = str(messages[-1].content or "")
            if self.generate_calls == 1:
                try:
                    await asyncio.sleep(5)
                except asyncio.CancelledError:
                    raise
            assert "Live subagent guidance:" in user_prompt
            assert "Restart with the cache check first." in user_prompt
            return GenerationResult(content="Fallback restarted after interrupt.", model="gpt-test")

    class _InterruptProvider:
        def __init__(self) -> None:
            self.handled: list[tuple[list[str], str, str | None]] = []
            self._handled_ids: set[str] = set()

        async def poll_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
            if kwargs.get("safe_point") != "during_model_invocation":
                return []
            if "midgen-fallback-restart-1" in self._handled_ids:
                return []
            return [
                {
                    "message_id": "midgen-fallback-restart-1",
                    "subagent_id": "run-midgen-fallback-restart:researcher:role-researcher",
                    "role_id": "researcher",
                    "content": "Restart with the cache check first.",
                    "delivery_mode": "inject_now",
                    "delivery_status": "queued",
                    "delivery_reason": "interrupt_pending",
                    "interrupt": True,
                    "cancel_current_tool": True,
                }
            ]

        async def mark_messages_handled(
            self,
            *,
            message_ids: list[str],
            status: str,
            reason: str | None = None,
        ) -> None:
            self._handled_ids.update(message_ids)
            self.handled.append((message_ids, status, reason))

    provider = _InterruptProvider()
    emitted: list[tuple[str, dict[str, Any]]] = []
    orchestrator = MultiAgentOrchestrator(engine=_FallbackOnlyEngine())
    orchestrator._current_run_id = "run-midgen-fallback-restart"  # noqa: SLF001
    orchestrator._subagent_message_provider = provider  # noqa: SLF001
    orchestrator._emit_runtime_event = lambda event_type, payload: emitted.append(  # noqa: SLF001
        (event_type, payload)
    )

    content, _diagnostics = asyncio.run(
        orchestrator._invoke_configured_text(  # noqa: SLF001
            model_id="gpt-test",
            system_prompt="System",
            user_prompt="Inspect the current direction.",
            temperature=0.2,
            max_tokens=64,
            execution_profile="subagent_readonly",
            tool_mode="disabled",
            system_prompt_addendum="System",
            session_scope="role::researcher",
            runtime_role_id="researcher",
            runtime_title="Researcher",
            runtime_stage="role::researcher",
            runtime_subagent_id="run-midgen-fallback-restart:researcher:role-researcher",
        )
    )

    assert content == "Fallback restarted after interrupt."
    assert provider.handled == [(["midgen-fallback-restart-1"], "applied", None)]
    assert [event_type for event_type, _payload in emitted[:5]] == [
        "subagent_message_accepted",
        "subagent_interrupted",
        "subagent_tool_cancel_requested",
        "subagent_tool_cancelled",
        "subagent_message_applied",
    ]

def test_subagent_cancel_current_tool_defers_for_non_cancellable_active_tool() -> None:
    class _BusyToolInvocationEngine:
        async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResult:
            controller = request.active_tool_controller
            assert controller is not None
            await controller.activate_tool(
                tool_call_id="tool-call-busy",
                tool_name="web_search",
                cancellable=False,
            )
            await asyncio.sleep(0.12)
            await controller.finish_tool()
            return AgentInvocationResult(
                content="Busy tool completed.",
                events=[
                    ToolCallRequestEvent(
                        call_id="tool-call-busy",
                        tool_name="web_search",
                        arguments={"query": "option B"},
                    ),
                    ToolCallResultEvent(
                        call_id="tool-call-busy",
                        tool_name="web_search",
                        result={"ok": True},
                        metadata={"status": "completed"},
                    ),
                ],
                diagnostics=AgentInvocationDiagnostics(
                    execution_profile="subagent_readonly",
                    tool_mode="auto",
                ),
            )

    class _CancelProvider:
        def __init__(self) -> None:
            self.handled: list[tuple[list[str], str, str | None]] = []
            self._handled_ids: set[str] = set()

        async def poll_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
            if kwargs.get("safe_point") != "during_active_tool":
                return []
            if "cancel-busy-1" in self._handled_ids:
                return []
            return [
                {
                    "message_id": "cancel-busy-1",
                    "subagent_id": "run-busy-tool:researcher:role-researcher",
                    "role_id": "researcher",
                    "content": "Stop the current tool and inspect the cache.",
                    "delivery_mode": "inject_now",
                    "delivery_status": "queued",
                    "interrupt": True,
                    "cancel_current_tool": True,
                }
            ]

        async def mark_messages_handled(
            self,
            *,
            message_ids: list[str],
            status: str,
            reason: str | None = None,
        ) -> None:
            self._handled_ids.update(message_ids)
            self.handled.append((message_ids, status, reason))

    provider = _CancelProvider()
    emitted: list[tuple[str, dict[str, Any]]] = []
    orchestrator = MultiAgentOrchestrator(engine=_BusyToolInvocationEngine())
    orchestrator._current_run_id = "run-busy-tool"  # noqa: SLF001
    orchestrator._subagent_message_provider = provider  # noqa: SLF001
    orchestrator._emit_runtime_event = lambda event_type, payload: emitted.append(  # noqa: SLF001
        (event_type, payload)
    )

    asyncio.run(
        orchestrator._invoke_configured_text(  # noqa: SLF001
            model_id="gpt-test",
            system_prompt="System",
            user_prompt="Inspect the current tool.",
            temperature=0.2,
            max_tokens=64,
            execution_profile="subagent_readonly",
            tool_mode="auto",
            system_prompt_addendum="System",
            session_scope="role::researcher",
            runtime_role_id="researcher",
            runtime_title="Researcher",
            runtime_stage="role::researcher",
            runtime_subagent_id="run-busy-tool:researcher:role-researcher",
        )
    )

    assert provider.handled == [(["cancel-busy-1"], "deferred", "tool_in_progress")]
    assert [event_type for event_type, _payload in emitted[:5]] == [
        "subagent_message_accepted",
        "subagent_interrupted",
        "subagent_tool_cancel_requested",
        "subagent_message_deferred",
        "subagent_tool_cancel_deferred",
    ]
    assert emitted[4][1]["reason"] == "tool_in_progress"

def test_delegated_subagent_runtime_persists_interrupt_cancel_protocol_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ProtocolEventOrchestrator:
        def __init__(self, **_: Any) -> None:
            pass

        async def run(self, request: Any) -> MultiAgentRunResult:
            assert request.subagent_message_provider is not None
            assert request.runtime_event_callback is not None
            messages = await request.subagent_message_provider.poll_messages(
                task_id="delegated-cancel-task",
                subagent_id="cancel-subagent-1",
                role_id="researcher",
                stage="role::researcher",
                safe_point="after_model_invocation",
            )
            assert len(messages) == 1
            message = messages[0]
            assert message["interrupt"] is True
            assert message["cancel_current_tool"] is True
            message_id = message["message_id"]
            events = [
                MultiAgentRunEvent(
                    run_id="delegated-cancel-task",
                    seq=1,
                    type="subagent_message_accepted",
                    payload={
                        "subagent_id": "cancel-subagent-1",
                        "role_id": "researcher",
                        "message_id": message_id,
                        "delivery_mode": "inject_now",
                        "delivery_status": "accepted",
                        "content": message["content"],
                    },
                    timestamp="2026-06-30T03:00:00Z",
                ),
                MultiAgentRunEvent(
                    run_id="delegated-cancel-task",
                    seq=2,
                    type="subagent_interrupted",
                    payload={
                        "subagent_id": "cancel-subagent-1",
                        "role_id": "researcher",
                        "message_id": message_id,
                        "delivery_mode": "inject_now",
                        "interrupt": True,
                        "cancel_current_tool": True,
                        "reason": "operator_interrupt",
                        "summary": "Subagent interrupt requested.",
                    },
                    timestamp="2026-06-30T03:00:01Z",
                ),
                MultiAgentRunEvent(
                    run_id="delegated-cancel-task",
                    seq=3,
                    type="subagent_tool_cancel_requested",
                    payload={
                        "subagent_id": "cancel-subagent-1",
                        "role_id": "researcher",
                        "message_id": message_id,
                        "delivery_mode": "inject_now",
                        "interrupt": True,
                        "cancel_current_tool": True,
                        "reason": "operator_request",
                        "summary": "Subagent tool cancellation requested.",
                    },
                    timestamp="2026-06-30T03:00:02Z",
                ),
                MultiAgentRunEvent(
                    run_id="delegated-cancel-task",
                    seq=4,
                    type="subagent_message_deferred",
                    payload={
                        "subagent_id": "cancel-subagent-1",
                        "role_id": "researcher",
                        "message_id": message_id,
                        "delivery_mode": "inject_now",
                        "delivery_status": "deferred",
                        "reason": "generation_in_progress",
                        "content": message["content"],
                    },
                    timestamp="2026-06-30T03:00:03Z",
                ),
                MultiAgentRunEvent(
                    run_id="delegated-cancel-task",
                    seq=5,
                    type="subagent_tool_cancel_deferred",
                    payload={
                        "subagent_id": "cancel-subagent-1",
                        "role_id": "researcher",
                        "message_id": message_id,
                        "delivery_mode": "inject_now",
                        "delivery_reason": "generation_in_progress",
                        "interrupt": True,
                        "cancel_current_tool": True,
                        "reason": "generation_in_progress",
                        "summary": "Subagent tool cancellation deferred.",
                    },
                    timestamp="2026-06-30T03:00:04Z",
                ),
            ]
            for event in events:
                await request.runtime_event_callback(event)
            await request.subagent_message_provider.mark_messages_handled(
                message_ids=[message_id],
                status="deferred",
                reason="generation_in_progress",
            )
            return MultiAgentRunResult(
                run_id="delegated-cancel-task",
                protocol="teacher_student_distill",
                state="succeeded",
                task_input=request.task_input,
                candidates=[],
                selected_candidate_id=None,
                evaluation=None,
                artifacts={"final_answer": "Deferred cancel protocol events."},
                events=events,
            )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator", _ProtocolEventOrchestrator)
    store = RuntimeStore(tmp_path / "runtime-cancel-protocol.db")
    service = RuntimeService(engine=object(), store=store)

    async def _run() -> dict[str, Any] | None:
        await store.create_task_run(
            task_id="delegated-cancel-task",
            input_text="Inspect the current task.",
            session_id="session-cancel-subagent",
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
                    "instruction": "Inspect the current task.",
                    "objective": "Inspect the current task.",
                    "parent_session_id": "session-cancel-subagent",
                    "status": "queued",
                },
            },
        )
        await store.upsert_subagent_transcript(
            subagent_id="cancel-subagent-1",
            parent_type="delegated_task",
            parent_id="delegated-cancel-task",
            session_id="session-cancel-subagent",
            role_id="researcher",
            title="Researcher",
            status="running",
            metadata={"task_id": "delegated-cancel-task"},
        )
        await store.append_subagent_transcript_event(
            "cancel-subagent-1",
            {
                "type": "subagent_message",
                "message_id": "cancel-message-persist-1",
                "subagent_id": "cancel-subagent-1",
                "target_role_id": "researcher",
                "content": "Stop the current command and inspect the cache.",
                "delivery_mode": "inject_now",
                "delivery_status": "queued",
                "delivery_reason": "tool_cancel_pending",
                "interrupt": True,
                "cancel_current_tool": True,
                "metadata": {
                    "source": "session_subagent_message_api",
                    "interrupt": True,
                    "cancel_current_tool": True,
                    "delivery_reason": "tool_cancel_pending",
                },
                "created_at": "2026-06-30T02:59:59Z",
            },
        )
        await service._run_task(task_id="delegated-cancel-task")  # noqa: SLF001
        return await store.get_subagent_transcript("cancel-subagent-1")

    detail = asyncio.run(_run())

    assert detail is not None
    assert [event["type"] for event in detail["events"]] == [
        "subagent_message",
        "subagent_message_accepted",
        "subagent_interrupted",
        "subagent_tool_cancel_requested",
        "subagent_message_deferred",
        "subagent_tool_cancel_deferred",
    ]
    assert detail["events"][2]["interrupt"] is True
    assert detail["events"][3]["cancel_current_tool"] is True
    assert detail["events"][-1]["delivery_reason"] == "generation_in_progress"

def test_subagent_after_current_tool_message_applies_after_tool_result() -> None:
    class _ToolInvocationEngine:
        async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResult:
            assert "Wait for tool completion." not in request.message
            return AgentInvocationResult(
                content="Tool-backed response.",
                events=[
                    ToolCallRequestEvent(
                        call_id="tool-call-after-current",
                        tool_name="web_search",
                        arguments={"query": "option B"},
                    ),
                    ToolCallResultEvent(
                        call_id="tool-call-after-current",
                        tool_name="web_search",
                        result={"ok": True},
                        metadata={"status": "completed"},
                    ),
                ],
                diagnostics=AgentInvocationDiagnostics(
                    execution_profile="subagent_readonly",
                    tool_mode="auto",
                ),
            )

    class _AfterToolProvider:
        def __init__(self) -> None:
            self.polls: list[str | None] = []
            self.handled: list[tuple[list[str], str]] = []

        async def poll_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
            self.polls.append(kwargs.get("safe_point"))
            return [
                {
                    "message_id": "after-tool-message-1",
                    "subagent_id": "run-tool:researcher:role-researcher",
                    "role_id": "researcher",
                    "content": "Wait for tool completion, then bias toward option B.",
                    "delivery_mode": "after_current_tool",
                }
            ]

        async def mark_messages_handled(
            self,
            *,
            message_ids: list[str],
            status: str,
            reason: str | None = None,
        ) -> None:
            _ = reason
            self.handled.append((message_ids, status))

    provider = _AfterToolProvider()
    emitted: list[tuple[str, dict[str, Any]]] = []
    orchestrator = MultiAgentOrchestrator(engine=_ToolInvocationEngine())
    orchestrator._current_run_id = "run-tool"  # noqa: SLF001
    orchestrator._subagent_message_provider = provider  # noqa: SLF001
    orchestrator._emit_runtime_event = lambda event_type, payload: emitted.append(  # noqa: SLF001
        (event_type, payload)
    )

    content, _diagnostics = asyncio.run(
        orchestrator._invoke_configured_text(  # noqa: SLF001
            model_id="gpt-test",
            system_prompt="System",
            user_prompt="Compare options.",
            temperature=0.2,
            max_tokens=64,
            execution_profile="subagent_readonly",
            tool_mode="auto",
            system_prompt_addendum="System",
            session_scope="role::researcher",
            runtime_role_id="researcher",
            runtime_title="Researcher",
            runtime_stage="role::researcher",
            runtime_subagent_id="run-tool:researcher:role-researcher",
        )
    )

    assert content == "Tool-backed response."
    assert provider.polls == ["before_model_invocation", "after_tool_result", "after_model_invocation"]
    assert provider.handled == [(["after-tool-message-1"], "applied")]
    delivery_events = [
        (event_type, payload)
        for event_type, payload in emitted
        if event_type.startswith("subagent_message_")
    ]
    assert [event_type for event_type, _payload in delivery_events] == [
        "subagent_message_accepted",
        "subagent_message_applied",
    ]
    assert delivery_events[1][1]["delivery_status"] == "applied"
    assert orchestrator._role_guidance_messages["researcher"] == [  # noqa: SLF001
        "Wait for tool completion, then bias toward option B."
    ]

def test_subagent_cancel_current_tool_defers_when_approval_is_pending() -> None:
    class _ApprovalToolInvocationEngine:
        async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResult:
            assert "Stop the current tool." not in request.message
            return AgentInvocationResult(
                content="Approval is pending.",
                events=[
                    ToolCallRequestEvent(
                        call_id="tool-call-approval-cancel",
                        tool_name="exec_command",
                        arguments={"command": "pytest -q"},
                    ),
                    ToolCallResultEvent(
                        call_id="tool-call-approval-cancel",
                        tool_name="exec_command",
                        result=None,
                        metadata={
                            "status": "approval_pending",
                            "approval_id": "approval-cancel-1",
                            "requires_approval": True,
                        },
                    ),
                ],
                diagnostics=AgentInvocationDiagnostics(
                    execution_profile="subagent_readonly",
                    tool_mode="auto",
                ),
            )

    class _CancelProvider:
        def __init__(self) -> None:
            self.handled: list[tuple[list[str], str, str | None]] = []

        async def poll_messages(self, **kwargs: Any) -> list[dict[str, Any]]:
            if kwargs.get("safe_point") != "after_model_invocation":
                return []
            return [
                {
                    "message_id": "approval-cancel-message-1",
                    "subagent_id": "run-approval-cancel:researcher:role-researcher",
                    "role_id": "researcher",
                    "content": "Stop the current tool.",
                    "delivery_mode": "inject_now",
                    "interrupt": True,
                    "cancel_current_tool": True,
                }
            ]

        async def mark_messages_handled(
            self,
            *,
            message_ids: list[str],
            status: str,
            reason: str | None = None,
        ) -> None:
            self.handled.append((message_ids, status, reason))

    provider = _CancelProvider()
    emitted: list[tuple[str, dict[str, Any]]] = []
    orchestrator = MultiAgentOrchestrator(engine=_ApprovalToolInvocationEngine())
    orchestrator._current_run_id = "run-approval-cancel"  # noqa: SLF001
    orchestrator._subagent_message_provider = provider  # noqa: SLF001
    orchestrator._emit_runtime_event = lambda event_type, payload: emitted.append(  # noqa: SLF001
        (event_type, payload)
    )

    asyncio.run(
        orchestrator._invoke_configured_text(  # noqa: SLF001
            model_id="gpt-test",
            system_prompt="System",
            user_prompt="Run the approved command.",
            temperature=0.2,
            max_tokens=64,
            execution_profile="subagent_readonly",
            tool_mode="auto",
            system_prompt_addendum="System",
            session_scope="role::researcher",
            runtime_role_id="researcher",
            runtime_title="Researcher",
            runtime_stage="role::researcher",
            runtime_subagent_id="run-approval-cancel:researcher:role-researcher",
        )
    )

    assert provider.handled == [(["approval-cancel-message-1"], "deferred", "approval_pending")]
    protocol_events = [
        (event_type, payload)
        for event_type, payload in emitted
        if event_type
        in {
            "subagent_message_accepted",
            "subagent_interrupted",
            "subagent_tool_cancel_requested",
            "subagent_message_deferred",
            "subagent_tool_cancel_deferred",
        }
    ]
    assert [event_type for event_type, _payload in protocol_events] == [
        "subagent_message_accepted",
        "subagent_interrupted",
        "subagent_tool_cancel_requested",
        "subagent_message_deferred",
        "subagent_tool_cancel_deferred",
    ]
    assert protocol_events[3][1]["reason"] == "approval_pending"
    assert protocol_events[4][1]["reason"] == "approval_pending"

def test_subagent_after_current_tool_message_defers_when_approval_pending() -> None:
    class _ApprovalToolInvocationEngine:
        async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResult:
            assert "Wait for approval resolution." not in request.message
            return AgentInvocationResult(
                content="Approval is pending.",
                events=[
                    ToolCallRequestEvent(
                        call_id="tool-call-approval",
                        tool_name="exec_command",
                        arguments={"command": "pytest -q"},
                    ),
                    ToolCallResultEvent(
                        call_id="tool-call-approval",
                        tool_name="exec_command",
                        result=None,
                        metadata={
                            "status": "approval_pending",
                            "approval_id": "approval-after-tool-1",
                            "requires_approval": True,
                        },
                    ),
                ],
                diagnostics=AgentInvocationDiagnostics(
                    execution_profile="subagent_readonly",
                    tool_mode="auto",
                ),
            )

    class _AfterToolProvider:
        def __init__(self) -> None:
            self.handled: list[tuple[list[str], str, str | None]] = []

        async def poll_messages(self, **_kwargs: Any) -> list[dict[str, Any]]:
            return [
                {
                    "message_id": "after-tool-approval-message-1",
                    "subagent_id": "run-approval-tool:researcher:role-researcher",
                    "role_id": "researcher",
                    "content": "Wait for approval resolution, then inspect the failure mode.",
                    "delivery_mode": "after_current_tool",
                }
            ]

        async def mark_messages_handled(
            self,
            *,
            message_ids: list[str],
            status: str,
            reason: str | None = None,
        ) -> None:
            self.handled.append((message_ids, status, reason))

    provider = _AfterToolProvider()
    emitted: list[tuple[str, dict[str, Any]]] = []
    orchestrator = MultiAgentOrchestrator(engine=_ApprovalToolInvocationEngine())
    orchestrator._current_run_id = "run-approval-tool"  # noqa: SLF001
    orchestrator._subagent_message_provider = provider  # noqa: SLF001
    orchestrator._emit_runtime_event = lambda event_type, payload: emitted.append(  # noqa: SLF001
        (event_type, payload)
    )

    asyncio.run(
        orchestrator._invoke_configured_text(  # noqa: SLF001
            model_id="gpt-test",
            system_prompt="System",
            user_prompt="Run the approved command.",
            temperature=0.2,
            max_tokens=64,
            execution_profile="subagent_readonly",
            tool_mode="auto",
            system_prompt_addendum="System",
            session_scope="role::researcher",
            runtime_role_id="researcher",
            runtime_title="Researcher",
            runtime_stage="role::researcher",
            runtime_subagent_id="run-approval-tool:researcher:role-researcher",
        )
    )

    assert provider.handled == [(["after-tool-approval-message-1"], "deferred", "approval_pending")]
    delivery_events = [
        (event_type, payload)
        for event_type, payload in emitted
        if event_type.startswith("subagent_message_")
    ]
    assert [event_type for event_type, _payload in delivery_events] == [
        "subagent_message_accepted",
        "subagent_message_deferred",
    ]
    assert delivery_events[1][1]["delivery_status"] == "deferred"
    assert delivery_events[1][1]["reason"] == "approval_pending"
    assert "researcher" not in orchestrator._role_guidance_messages  # noqa: SLF001

@pytest.mark.parametrize(('delegate_status', 'expected_event_type', 'expected_status'), [('queued', 'subagent_progress', 'queued'), ('running', 'subagent_progress', 'running'), ('resumed', 'subagent_progress', 'running'), ('completed', 'subagent_completed', 'completed'), ('succeeded', 'subagent_completed', 'completed'), ('done', 'subagent_completed', 'completed'), ('failed', 'subagent_completed', 'failed'), ('error', 'subagent_completed', 'failed'), ('cancelled', 'subagent_completed', 'cancelled')])
def test_build_subagent_lifecycle_events_tracks_delegate_result_status(
    delegate_status: str,
    expected_event_type: str,
    expected_status: str,
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"

    class _StatusEngine(_FakeEngine):
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
            _ = (
                inference_overrides,
                project_id,
                workspace_dir,
                selected_skill_ids,
                attachments,
            )
            self.chat_calls.append((message, session_id))
            yield ToolCallRequestEvent(
                call_id="call-subagent-status",
                tool_name="delegate_subagent_task",
                arguments={"objective": "Compare options."},
            )
            yield ToolCallResultEvent(
                call_id="call-subagent-status",
                tool_name="delegate_subagent_task",
                result={
                    "status": delegate_status,
                    "task_id": "subagent-task-status",
                    "task_type": "delegated_multi_agent",
                    "display_name": "Delegated status probe",
                    "parent_session_id": session_id,
                },
                metadata={
                    "status": delegate_status,
                    "task_id": "subagent-task-status",
                    "task_type": "delegated_multi_agent",
                    "parent_session_id": session_id,
                },
            )
            yield FinalAnswerEvent(content="Delegated.")

    app, _ = _build_app(engine=_StatusEngine())
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = SessionStore(sessions_dir)
    app.state.runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(sessions_dir / "runtime-status.db"),
    )

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/stream",
            json={
                "message": "Delegate the comparison.",
                "session_id": f"session-status-{delegate_status}",
            },
        ) as response:
            chunks = [
                line.removeprefix("data: ")
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]

    assert response.status_code == 200
    events = [json.loads(chunk) for chunk in chunks]
    synthesized = [event for event in events if event["type"].startswith("subagent_")]
    assert [event["type"] for event in synthesized[:2]] == ["subagent_started", "subagent_prompt"]
    assert synthesized[0]["status"] == ("queued" if expected_status == "queued" else "running")
    assert synthesized[1]["status"] == ("queued" if expected_status == "queued" else "running")
    assert synthesized[2]["type"] == expected_event_type
    assert synthesized[2]["status"] == expected_status
