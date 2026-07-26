from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mochi.agents.engine import AgentEngine
from mochi.agents.events import ToolCallResultEvent
from mochi.agents.invocation import AgentInvocationRequest
from mochi.backends.types import GenerationResult, Message
from mochi.config.schema import MochiConfig
from mochi.security.file_contract import tool_arguments_digest
from mochi.sessions.store import SessionStore
from mochi.sessions.timeline_coordinator import TimelineCoordinator, TimelineCoordinatorError
from mochi.sessions.turn_timeline import SessionTurnTimelineRepository
from tests.unit.engine._support import FakeBackend


class _BlockingBackend(FakeBackend):
    def __init__(self) -> None:
        super().__init__()
        self.first_generation_started = asyncio.Event()
        self.release_first_generation = asyncio.Event()
        self._generation_count = 0

    async def generate(self, messages: list[Message], **kwargs: object) -> GenerationResult:
        self.calls.append([Message(**message.__dict__) for message in messages])
        self._generation_count += 1
        if self._generation_count == 1:
            self.first_generation_started.set()
            await self.release_first_generation.wait()
        return GenerationResult(content="fake reply")


def _config(tmp_path: Path) -> MochiConfig:
    return MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {
                "require_approval_for_exec": False,
                "require_approval_for_file_write": False,
            },
        }
    )


async def _collect(
    engine: AgentEngine,
    message: str,
    session_id: str,
    *,
    turn_id: str | None = None,
) -> list[object]:
    return [
        event
        async for event in engine.chat(
            message,
            session_id=session_id,
            turn_id=turn_id,
        )
    ]


@pytest.mark.asyncio
async def test_same_session_chat_claims_fifo_before_prompt_and_persists_atomic_transcript(
    tmp_path: Path,
) -> None:
    backend = _BlockingBackend()
    engine = AgentEngine(_config(tmp_path))

    async def load(_: str) -> _BlockingBackend:
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = load  # type: ignore[method-assign]
    first = asyncio.create_task(_collect(engine, "first request", "timeline-session"))
    await asyncio.wait_for(backend.first_generation_started.wait(), timeout=2)

    second = asyncio.create_task(_collect(engine, "second request", "timeline-session"))
    await asyncio.sleep(0.05)
    assert len(backend.calls) == 1

    backend.release_first_generation.set()
    await asyncio.wait_for(first, timeout=4)
    await asyncio.wait_for(second, timeout=4)

    second_prompt = next(
        call
        for call in reversed(backend.calls)
        if any(message.role == "user" and message.content == "second request" for message in call)
    )
    contents = [message.content for message in second_prompt]
    assert contents.index("first request") < contents.index("fake reply") < contents.index("second request")

    store = SessionStore(tmp_path / "sessions")
    snapshot = await store.load_strict_snapshot("timeline-session")
    messages = [event for event in snapshot.events if event.get("type") == "message"]
    assert [(event["role"], event["content"]) for event in messages] == [
        ("user", "first request"),
        ("user", "second request"),
        ("assistant", "fake reply"),
        ("assistant", "fake reply"),
    ]
    assistant_indexes = [
        index
        for index, event in enumerate(snapshot.events)
        if event.get("type") == "message" and event.get("role") == "assistant"
    ]
    assert len(assistant_indexes) == 2
    for index in assistant_indexes:
        assert snapshot.events[index + 1].get("event") == "session_turn_timeline"
    timeline = await SessionTurnTimelineRepository(store).load("timeline-session")
    assert timeline.timeline is not None
    assert timeline.timeline.lane_turn_id is None
    assert all(turn.status == "terminal" for turn in timeline.timeline.turns)
    await engine.close()


@pytest.mark.asyncio
async def test_three_preclaimed_turns_materialize_history_in_fifo_turn_order(
    tmp_path: Path,
) -> None:
    backend = _BlockingBackend()
    engine = AgentEngine(_config(tmp_path))

    async def load(_: str) -> _BlockingBackend:
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = load  # type: ignore[method-assign]
    first = asyncio.create_task(
        _collect(
            engine,
            "first request",
            "three-turn-session",
            turn_id="turn-one",
        )
    )
    await asyncio.wait_for(backend.first_generation_started.wait(), timeout=2)
    second = asyncio.create_task(
        _collect(
            engine,
            "second request",
            "three-turn-session",
            turn_id="turn-two",
        )
    )
    tasks = [first, second]
    try:
        repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
        for _ in range(500):
            loaded = await repository.load("three-turn-session")
            if loaded.timeline is not None and len(loaded.timeline.turns) == 2:
                break
            await asyncio.sleep(0.01)
        assert loaded.timeline is not None
        assert [turn.turn_id for turn in loaded.timeline.turns] == ["turn-one", "turn-two"]

        third = asyncio.create_task(
            _collect(
                engine,
                "third request",
                "three-turn-session",
                turn_id="turn-three",
            )
        )
        tasks.append(third)
        for _ in range(500):
            loaded = await repository.load("three-turn-session")
            if loaded.timeline is not None and len(loaded.timeline.turns) == 3:
                break
            await asyncio.sleep(0.01)
        assert loaded.timeline is not None
        assert [turn.turn_id for turn in loaded.timeline.turns] == [
            "turn-one",
            "turn-two",
            "turn-three",
        ]
        assert len(backend.calls) == 1

        backend.release_first_generation.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=8)

        third_prompt = next(
            call
            for call in reversed(backend.calls)
            if any(message.role == "user" and message.content == "third request" for message in call)
        )
        relevant_contents = [
            message.content
            for message in third_prompt
            if message.content in {"first request", "second request", "third request", "fake reply"}
        ]
        assert relevant_contents == [
            "first request",
            "fake reply",
            "second request",
            "fake reply",
            "third request",
        ]
    finally:
        backend.release_first_generation.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await engine.close()


@pytest.mark.asyncio
async def test_cross_engine_same_session_claims_one_durable_model_lane(
    tmp_path: Path,
) -> None:
    first_backend = _BlockingBackend()
    second_backend = FakeBackend()
    first_engine = AgentEngine(_config(tmp_path))
    second_engine = AgentEngine(_config(tmp_path))

    async def load_first(_: str) -> _BlockingBackend:
        first_engine._router._active = first_backend  # noqa: SLF001
        return first_backend

    async def load_second(_: str) -> FakeBackend:
        second_engine._router._active = second_backend  # noqa: SLF001
        return second_backend

    first_engine._router.load = load_first  # type: ignore[method-assign]
    second_engine._router.load = load_second  # type: ignore[method-assign]
    first = asyncio.create_task(
        _collect(
            first_engine,
            "first cross-engine request",
            "cross-engine-session",
            turn_id="turn-one",
        )
    )
    await asyncio.wait_for(first_backend.first_generation_started.wait(), timeout=2)
    second = asyncio.create_task(
        _collect(
            second_engine,
            "second cross-engine request",
            "cross-engine-session",
            turn_id="turn-two",
        )
    )
    tasks = (first, second)
    try:
        repository = SessionTurnTimelineRepository(SessionStore(tmp_path / "sessions"))
        for _ in range(500):
            loaded = await repository.load("cross-engine-session")
            if loaded.timeline is not None and len(loaded.timeline.turns) == 2:
                break
            await asyncio.sleep(0.01)
        assert loaded.timeline is not None
        assert second_backend.calls == []

        first_backend.release_first_generation.set()
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=8)

        relevant_sequences = [
            [
                message.content
                for message in call
                if message.content
                in {
                    "first cross-engine request",
                    "second cross-engine request",
                    "fake reply",
                }
            ]
            for call in second_backend.calls
            if any(
                message.role == "user"
                and message.content == "second cross-engine request"
                for message in call
            )
        ]
        assert any(
            sequence[:3]
            == [
                "first cross-engine request",
                "fake reply",
                "second cross-engine request",
            ]
            for sequence in relevant_sequences
        )
    finally:
        first_backend.release_first_generation.set()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await first_engine.close()
        await second_engine.close()


@pytest.mark.asyncio
async def test_cancelled_queued_chat_never_reaches_model_and_terminalizes(tmp_path: Path) -> None:
    backend = _BlockingBackend()
    engine = AgentEngine(_config(tmp_path))

    async def load(_: str) -> _BlockingBackend:
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = load  # type: ignore[method-assign]
    first = asyncio.create_task(_collect(engine, "first request", "cancel-session"))
    await asyncio.wait_for(backend.first_generation_started.wait(), timeout=2)
    second = asyncio.create_task(_collect(engine, "queued request", "cancel-session"))
    await asyncio.sleep(0.05)
    turn_id = engine._active_chat_session_index["cancel-session"][-1]  # noqa: SLF001
    result = await engine.cancel_chat_run("cancel-session", turn_id=turn_id)
    assert result["status"] == "cancel_requested"
    backend.release_first_generation.set()
    await asyncio.wait_for(first, timeout=4)
    await asyncio.wait_for(second, timeout=4)

    assert not any(
        any(message.role == "user" and message.content == "queued request" for message in call)
        for call in backend.calls
    )
    timeline = await SessionTurnTimelineRepository(
        SessionStore(tmp_path / "sessions")
    ).load("cancel-session")
    assert timeline.timeline is not None
    queued = next(turn for turn in timeline.timeline.turns if turn.turn_id == turn_id)
    assert queued.status == "terminal"
    assert queued.terminal_outcome == "cancelled"

    await _collect(
        engine,
        "request after cancellation",
        "cancel-session",
        turn_id="turn-after-cancellation",
    )
    successor_prompts = [
        call
        for call in backend.calls
        if any(
            message.role == "user"
            and message.content == "request after cancellation"
            for message in call
        )
    ]
    assert successor_prompts
    assert all(
        message.content != "queued request"
        for call in successor_prompts
        for message in call
    )
    await engine.close()


@pytest.mark.asyncio
async def test_cancelled_running_chat_releases_the_lane_only_after_worker_stops(tmp_path: Path) -> None:
    backend = _BlockingBackend()
    engine = AgentEngine(_config(tmp_path))

    async def load(_: str) -> _BlockingBackend:
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = load  # type: ignore[method-assign]
    running = asyncio.create_task(_collect(engine, "running request", "running-cancel"))
    await asyncio.wait_for(backend.first_generation_started.wait(), timeout=2)
    turn_id = engine._active_chat_session_index["running-cancel"][-1]  # noqa: SLF001
    result = await engine.cancel_chat_run("running-cancel", turn_id=turn_id)
    assert result["status"] == "cancel_requested"
    await asyncio.wait_for(running, timeout=4)

    timeline = await SessionTurnTimelineRepository(
        SessionStore(tmp_path / "sessions")
    ).load("running-cancel")
    assert timeline.timeline is not None
    turn = next(turn for turn in timeline.timeline.turns if turn.turn_id == turn_id)
    assert turn.status == "terminal"
    assert turn.terminal_outcome == "cancelled"
    assert timeline.timeline.lane_turn_id is None
    await engine.close()


async def _engine_terminal_pending_approval(
    engine: AgentEngine,
    *,
    session_id: str,
) -> tuple[dict[str, object], str, str]:
    coordinator = TimelineCoordinator(
        session_store=engine._session_store,  # noqa: SLF001
        session_id=session_id,
        turn_id="turn-one",
    )
    await coordinator.admit_user_message(
        {
            "type": "message",
            "schema_version": 1,
            "session_id": session_id,
            "turn_id": "turn-one",
            "role": "user",
            "content": "write after approval",
        }
    )
    await coordinator.claim()
    arguments = {"path": "approval.txt", "content": "approved"}
    operation_id, arguments_digest = await coordinator.precommit_mutation(
        tool_name="file_write",
        arguments=arguments,
        call_id="approval-call-1",
    )
    await coordinator.persist_approval_pending(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={
            "tool_name": "file_write",
            "metadata": {"timeline_approval_pending": True},
        },
    )
    await coordinator.block_unstarted_turn()
    await coordinator.finish()
    assert arguments_digest == tool_arguments_digest(
        tool_name="file_write",
        arguments=arguments,
    )
    payload: dict[str, object] = {
        "session_id": session_id,
        "tool_name": "file_write",
        "arguments": arguments,
        "operation_id": operation_id,
        "timeline_call_id": "approval-call-1",
        "arguments_digest": arguments_digest,
        "ordinary_chat_checkpoint": {
            "schema_version": 1,
            "source": "ordinary_chat",
            "session_id": session_id,
            "turn_id": "turn-one",
            "operation_id": operation_id,
            "timeline_call_id": "approval-call-1",
            "arguments_digest": arguments_digest,
            "resume_cursor": {"tool_call_id": "approval-call-1"},
        },
    }
    return payload, operation_id, arguments_digest


@pytest.mark.asyncio
async def test_engine_approval_boundary_allows_exact_resume_once_before_runtime_dispatch(
    tmp_path: Path,
) -> None:
    engine = AgentEngine(_config(tmp_path))
    payload, operation_id, _ = await _engine_terminal_pending_approval(
        engine,
        session_id="engine-approval-once",
    )
    runtime_calls = 0

    async def begin_then_dispatch() -> None:
        nonlocal runtime_calls
        try:
            await engine.begin_ordinary_chat_approval_operation(
                approval_id="approval-1",
                approval_payload=payload,
            )
        except ValueError:
            return
        runtime_calls += 1

    await asyncio.gather(begin_then_dispatch(), begin_then_dispatch())
    assert runtime_calls == 1

    await engine.record_ordinary_chat_approval_operation_result(
        approval_id="approval-1",
        approval_payload=payload,
        execution_result={"status": "completed", "output": {"bytes_written": 8}},
        status="succeeded",
    )
    timeline = await SessionTurnTimelineRepository(engine._session_store).load(  # noqa: SLF001
        "engine-approval-once"
    )
    assert timeline.timeline is not None
    descriptor = timeline.timeline.turns[0].operation_descriptors[0]
    assert descriptor.operation_id == operation_id
    assert descriptor.status == "succeeded"
    await engine.close()


@pytest.mark.asyncio
async def test_engine_started_approval_crash_is_unknown_and_blocks_follower_effects(
    tmp_path: Path,
) -> None:
    engine = AgentEngine(_config(tmp_path))
    payload, operation_id, _ = await _engine_terminal_pending_approval(
        engine,
        session_id="engine-approval-unknown",
    )
    await engine.begin_ordinary_chat_approval_operation(
        approval_id="approval-unknown",
        approval_payload=payload,
    )
    await engine.record_ordinary_chat_approval_operation_result(
        approval_id="approval-unknown",
        approval_payload=payload,
        execution_result={"status": "transport_lost"},
        status="unknown",
    )
    timeline = await SessionTurnTimelineRepository(engine._session_store).load(  # noqa: SLF001
        "engine-approval-unknown"
    )
    assert timeline.timeline is not None
    assert timeline.timeline.turns[0].operation_descriptors[0].operation_id == operation_id
    assert timeline.timeline.turns[0].operation_descriptors[0].status == "unknown"

    follower = TimelineCoordinator(
        session_store=engine._session_store,  # noqa: SLF001
        session_id="engine-approval-unknown",
        turn_id="turn-two",
    )
    await follower.admit_user_message(
        {
            "type": "message",
            "schema_version": 1,
            "session_id": "engine-approval-unknown",
            "turn_id": "turn-two",
            "role": "user",
            "content": "try another mutation",
        }
    )
    await follower.claim()
    with pytest.raises(TimelineCoordinatorError, match="unknown prior operation"):
        await follower.precommit_mutation(
            tool_name="file_write",
            arguments={"path": "blocked.txt", "content": "never"},
            call_id="follower-write",
        )
    await follower.finish(failed=True)
    await engine.close()


@pytest.mark.asyncio
async def test_engine_publishes_approval_pending_only_after_timeline_companion_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    engine = AgentEngine(_config(tmp_path))
    monkeypatch.setattr(engine, "_log_agent_event", lambda *_: None)
    coordinator = TimelineCoordinator(
        session_store=engine._session_store,  # noqa: SLF001
        session_id="engine-pending-publication",
        turn_id="turn-one",
    )
    await coordinator.admit_user_message(
        {
            "type": "message",
            "schema_version": 1,
            "session_id": "engine-pending-publication",
            "turn_id": "turn-one",
            "role": "user",
            "content": "request approval",
        }
    )
    await coordinator.claim()
    operation_id, _ = await coordinator.precommit_mutation(
        tool_name="file_write",
        arguments={"path": "pending.txt", "content": "approval"},
        call_id="pending-call",
    )
    event = ToolCallResultEvent(
        call_id="pending-call",
        tool_name="file_write",
        result=None,
        error="File write requires approval.",
        metadata={
            "timeline_operation_id": operation_id,
            "timeline_approval_pending": True,
            "approval_id": "approval-pending",
        },
    )
    observed: list[tuple[str, str, bool]] = []

    async def callback(_: object) -> None:
        timeline = await SessionTurnTimelineRepository(engine._session_store).load(  # noqa: SLF001
            "engine-pending-publication"
        )
        snapshot = await engine._session_store.load_strict_snapshot(  # noqa: SLF001
            "engine-pending-publication"
        )
        assert timeline.timeline is not None
        descriptor = timeline.timeline.turns[0].operation_descriptors[0]
        observed.append(
            (
                descriptor.status,
                descriptor.precommit_boundary,
                any(item.get("event_id") == "turn-one:1" for item in snapshot.events),
            )
        )

    request = AgentInvocationRequest(
        message="request approval",
        session_id="engine-pending-publication",
        execution_profile="chat",
        persist_session=False,
        turn_id="turn-one",
        timeline_coordinator=coordinator,
    )
    next_seq = await engine._record_react_event(  # noqa: SLF001
        event=event,
        trajectory_id="pending-publication",
        tool_exposure_metadata={},
        turn_id="turn-one",
        session_id="engine-pending-publication",
        request=request,
        persist_turn_events=False,
        events=[],
        event_callback=callback,
        turn_event_seq=0,
    )
    assert next_seq == 1
    assert observed == [("precommitted", "not_started", True)]
    await coordinator.block_unstarted_turn()
    await coordinator.finish()
    await engine.close()
