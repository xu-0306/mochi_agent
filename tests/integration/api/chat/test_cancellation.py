"""Chat cancellation and stream teardown integration tests."""

from __future__ import annotations

from ._support import *  # noqa: F401,F403

def test_agent_engine_cancel_chat_run_cancels_active_run() -> None:
    engine = AgentEngine.__new__(AgentEngine)
    started = threading.Event()
    cancelled = threading.Event()

    async def _fake_invoke(
        self: AgentEngine,
        request: AgentInvocationRequest,
        *,
        event_callback=None,
    ) -> AgentInvocationResult:
        del self, request
        if event_callback is not None:
            await event_callback(ThinkingEvent(content="working"))
        started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return AgentInvocationResult(
            content="",
            events=[],
            diagnostics=AgentInvocationDiagnostics(
                execution_profile="chat",
                tool_mode="auto",
            ),
        )

    engine._invoke_shared_runtime = _fake_invoke.__get__(engine, AgentEngine)  # type: ignore[attr-defined]

    async def _exercise() -> dict[str, Any]:
        stream = engine.chat(
            "cancel me",
            session_id="session-cancel",
            turn_id="turn-cancel",
        )
        first = await anext(stream)
        assert isinstance(first, ThinkingEvent)
        assert started.wait(timeout=1.0)
        result = await engine.cancel_chat_run("session-cancel", turn_id="turn-cancel")
        with contextlib.suppress(StopAsyncIteration):
            await anext(stream)
        await stream.aclose()
        return result

    cancel_response = asyncio.run(_exercise())

    assert cancel_response["status"] == "cancel_requested"
    assert cancel_response["run_state"] == "cancelled"
    assert cancel_response["cancel_outcome"] == "cancelled"
    assert cancelled.wait(timeout=1.0)

def test_chat_cancel_endpoint_forwards_to_engine() -> None:
    class _CancelableEngine(_FakeEngine):
        async def cancel_chat_run(
            self,
            session_id: str,
            *,
            turn_id: str | None = None,
        ) -> dict[str, Any]:
            return {
                "status": "cancel_requested",
                "session_id": session_id,
                "turn_id": turn_id,
                "run_state": "cancelled",
                "cancel_outcome": "cancelled",
                "cancel_reason": None,
            }

    app, _ = _build_app(engine=_CancelableEngine())

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/session-endpoint/cancel",
            json={"turn_id": "turn-endpoint"},
        )

    assert response.status_code == 200
    assert response.json() == {
        "status": "cancel_requested",
        "session_id": "session-endpoint",
        "turn_id": "turn-endpoint",
        "run_state": "cancelled",
        "cancel_outcome": "cancelled",
        "cancel_reason": None,
    }

def test_chat_stream_cancel_endpoint_reports_completed_when_final_answer_wins_race() -> None:
    engine = AgentEngine.__new__(AgentEngine)
    cancelled = threading.Event()

    async def _fake_invoke(
        self: AgentEngine,
        request: AgentInvocationRequest,
        *,
        event_callback=None,
    ) -> AgentInvocationResult:
        del self, request
        final = FinalAnswerEvent(content="done")
        if event_callback is not None:
            await event_callback(ThinkingEvent(content="working"))
            await event_callback(final)
        try:
            await asyncio.sleep(0.2)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return AgentInvocationResult(
            content="done",
            events=[final],
            diagnostics=AgentInvocationDiagnostics(
                execution_profile="chat",
                tool_mode="auto",
            ),
        )

    engine._invoke_shared_runtime = _fake_invoke.__get__(engine, AgentEngine)  # type: ignore[attr-defined]

    async def _noop_close(self: AgentEngine) -> None:
        del self

    engine.close = _noop_close.__get__(engine, AgentEngine)  # type: ignore[attr-defined]
    app, _ = _build_app(engine=engine)  # type: ignore[arg-type]

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/stream",
            json={"message": "finish first", "session_id": "session-complete-race"},
        ) as response:
            seen_final = False
            for line in response.iter_lines():
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line.removeprefix("data: "))
                if payload.get("type") == "final_answer":
                    seen_final = True
                    break
            assert seen_final is True
            turn_id = response.headers["x-turn-id"]
            cancel_response = client.post(
                "/v1/chat/session-complete-race/cancel",
                json={"turn_id": turn_id},
            )

    assert cancel_response.status_code == 200
    assert cancel_response.json()["status"] == "already_completed"
    assert cancel_response.json()["run_state"] == "completed"
    assert cancel_response.json()["cancel_outcome"] == "completed"
    assert cancelled.is_set() is False

def test_agent_engine_chat_yields_events_before_invocation_finishes() -> None:
    """`AgentEngine.chat()` should surface intermediate events before the invocation fully completes."""
    engine = AgentEngine.__new__(AgentEngine)

    async def _fake_invoke(
        self: AgentEngine,
        request: AgentInvocationRequest,
        *,
        event_callback=None,
    ) -> AgentInvocationResult:
        del self, request
        thinking = ThinkingEvent(content="streaming-thought")
        thinking.turn_id = "turn-stream"  # type: ignore[attr-defined]
        final = FinalAnswerEvent(content="streaming-answer")
        final.turn_id = "turn-stream"  # type: ignore[attr-defined]

        if event_callback is not None:
            await event_callback(thinking)
        await asyncio.sleep(0.05)
        if event_callback is not None:
            await event_callback(final)
        await asyncio.sleep(0.2)

        return AgentInvocationResult(
            content="streaming-answer",
            events=[thinking, final],
            diagnostics=AgentInvocationDiagnostics(
                execution_profile="chat",
                tool_mode="auto",
            ),
        )

    engine._invoke_shared_runtime = _fake_invoke.__get__(engine, AgentEngine)  # type: ignore[attr-defined]

    async def _collect() -> tuple[list[object], float, float]:
        start = time.perf_counter()
        first_event_elapsed: float | None = None
        events: list[object] = []

        async for event in engine.chat("hello", session_id="session-stream"):
            events.append(event)
            if first_event_elapsed is None:
                first_event_elapsed = time.perf_counter() - start

        total_elapsed = time.perf_counter() - start
        assert first_event_elapsed is not None
        return events, first_event_elapsed, total_elapsed

    events, first_event_elapsed, total_elapsed = asyncio.run(_collect())

    assert [event.type for event in events] == ["thinking", "final_answer"]
    assert first_event_elapsed < 0.15
    assert total_elapsed >= 0.25

def test_agent_engine_chat_teardown_cancels_worker() -> None:
    engine = AgentEngine.__new__(AgentEngine)
    cancelled = threading.Event()

    async def _fake_invoke(
        self: AgentEngine,
        request: AgentInvocationRequest,
        *,
        event_callback=None,
    ) -> AgentInvocationResult:
        del self, request
        thinking = ThinkingEvent(content="streaming-thought")
        if event_callback is not None:
            await event_callback(thinking)
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return AgentInvocationResult(
            content="",
            events=[thinking],
            diagnostics=AgentInvocationDiagnostics(
                execution_profile="chat",
                tool_mode="auto",
            ),
        )

    engine._invoke_shared_runtime = _fake_invoke.__get__(engine, AgentEngine)  # type: ignore[attr-defined]

    async def _consume_and_close() -> None:
        stream = engine.chat(
            "hello",
            session_id="session-stream-cancel",
            turn_id="turn-stream-cancel",
        )
        first_event = await anext(stream)
        assert isinstance(first_event, ThinkingEvent)
        await stream.aclose()

    asyncio.run(_consume_and_close())

    assert cancelled.wait(timeout=1.0)

def test_run_cancellation_context_cancels_generation_when_no_tool_is_active() -> None:
    async def _run() -> tuple[dict[str, Any], dict[str, Any]]:
        context = RunCancellationContext(run_id="run-no-tool")
        task = asyncio.create_task(asyncio.sleep(60))
        await context.bind_generation_cancel_callback(lambda: cancel_asyncio_task(task))
        result = await context.request_run_cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return result.__dict__, await context.snapshot()

    result, snapshot = asyncio.run(_run())

    assert result["cancelled"] is True
    assert result["state"] == "cancelled"
    assert result["boundary"] == "generation"
    assert snapshot["state"] == "cancelled"
    assert snapshot["cancel_confirmed"] is True

def test_run_cancellation_context_cancels_cancellable_active_tool_before_stopping_run() -> None:
    async def _run() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        context = RunCancellationContext(run_id="run-cancellable-tool")
        controller = ActiveToolController()
        await context.bind_active_tool_controller(controller)
        task = asyncio.create_task(asyncio.sleep(60))
        await context.bind_generation_cancel_callback(lambda: cancel_asyncio_task(task))
        await controller.activate_tool(
            tool_call_id="call-tool",
            tool_name="exec_command",
            cancellable=True,
        )
        trace: list[str] = []

        async def _cancel() -> ToolCancellationResult:
            trace.append("tool")
            await controller.finish_tool()
            return ToolCancellationResult(
                cancelled=True,
                reason="tool_cancelled",
                tool_call_id="call-tool",
                tool_name="exec_command",
            )

        await controller.bind_cancel_callback(session_id="session-tool", callback=_cancel)
        result = await context.request_run_cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return result.__dict__, await context.snapshot(), trace

    result, snapshot, trace = asyncio.run(_run())

    assert trace == ["tool"]
    assert result["cancelled"] is True
    assert result["state"] == "cancelled"
    assert result["boundary"] == "generation"
    assert snapshot["state"] == "cancelled"

def test_run_cancellation_context_defers_when_active_tool_is_not_cancellable() -> None:
    async def _run() -> tuple[dict[str, Any], dict[str, Any], list[str]]:
        context = RunCancellationContext(run_id="run-non-cancellable-tool")
        controller = ActiveToolController()
        await context.bind_active_tool_controller(controller)
        generation_calls: list[str] = []

        async def _cancel_generation() -> str:
            generation_calls.append("generation")
            return "cancelled"

        await context.bind_generation_cancel_callback(_cancel_generation)
        await controller.activate_tool(
            tool_call_id="call-tool",
            tool_name="non_cancellable_tool",
            cancellable=False,
        )
        result = await context.request_run_cancel()
        return result.__dict__, await context.snapshot(), generation_calls

    result, snapshot, generation_calls = asyncio.run(_run())

    assert result["cancelled"] is False
    assert result["state"] == "pending"
    assert result["boundary"] == "tool"
    assert result["reason"] == "tool_in_progress"
    assert snapshot["state"] == "cancelling"
    assert generation_calls == []
