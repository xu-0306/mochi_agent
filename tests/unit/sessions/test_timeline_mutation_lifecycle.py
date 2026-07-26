from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

import mochi.agents.engine as engine_module
from mochi.agents.engine import AgentEngine
from mochi.agents.events import (
    FinalAnswerEvent,
    ToolCallCompletedEvent,
    ToolCallResultEvent,
)
from mochi.agents.invocation import AgentInvocationRequest
from mochi.backends.types import Message
from mochi.config.schema import MochiConfig
from mochi.agents.artifact_verifier import tool_arguments_digest
from mochi.runtime.approvals import InMemoryApprovalStore
from mochi.runtime.exec_sessions import ExecSessionStatus, SessionPollResult
from mochi.runtime.sandbox.base import (
    HostSandboxBackend,
    SandboxResourceLimits,
    create_sandbox_plan,
)
from mochi.sessions.store import SessionStore
from mochi.sessions.timeline_coordinator import (
    TimelineCoordinator,
    TimelineCoordinatorError,
)
from mochi.sessions.turn_timeline import SessionTurnTimelineRepository
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.exec_command import ExecCommandTool
from mochi.tools.execute_code import ExecuteCodeTool
from mochi.tools.execute_code_v2 import ExecuteCodeV2Tool
from mochi.tools.file_ops import FileReadTool, FileWriteTool
from mochi.tools.kill_session import KillSessionTool
from mochi.tools.process_control import ProcessStopTool
from mochi.tools.registry import ToolRegistry
from mochi.tools.write_stdin import WriteStdinTool
from tests.unit.engine._support import FakeBackend


async def _claimed_coordinator(
    tmp_path: Path,
    *,
    session_id: str = "timeline-mutation",
    turn_id: str = "turn-one",
) -> tuple[TimelineCoordinator, SessionStore]:
    store = SessionStore(tmp_path / "sessions")
    coordinator = TimelineCoordinator(
        session_store=store,
        session_id=session_id,
        turn_id=turn_id,
    )
    await coordinator.admit_user_message(
        {
            "type": "message",
            "schema_version": 1,
            "session_id": session_id,
            "turn_id": turn_id,
            "role": "user",
            "content": "mutate a file",
        }
    )
    await coordinator.claim()
    return coordinator, store


def _context(
    tmp_path: Path,
    coordinator: TimelineCoordinator,
    *,
    call_id: str = "file-call-1",
    permission_policy: dict[str, object] | None = None,
) -> ToolExecutionContext:
    return ToolExecutionContext(
        workspace_dir=str(tmp_path),
        permission_policy=permission_policy or {},
        state={
            "timeline_tool_lifecycle": coordinator,
            "timeline_tool_call_id": call_id,
        },
    )


def _descriptor(store: SessionStore, session_id: str, operation_id: str):
    repository = SessionTurnTimelineRepository(store)

    async def load():
        loaded = await repository.load(session_id)
        assert loaded.timeline is not None
        return next(
            item
            for turn in loaded.timeline.turns
            for item in turn.operation_descriptors
            if item.operation_id == operation_id
        )

    return load


class _PreEffectFileWrite(BaseTool):
    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return "Return before reaching a physical write."

    @property
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        }

    @property
    def supports_timeline_side_effect_boundary(self) -> bool:
        return True

    @property
    def supports_timeline_approval_revocation(self) -> bool:
        return True

    async def execute(self, **_: object) -> ToolResult:
        return ToolResult(error="pre-effect validation failed")


class _UnawareFileWriteProbe(BaseTool):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return "Simulates a custom side effect that has no durable boundary hook."

    @property
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        }

    async def execute(self, **_: object) -> ToolResult:
        self.calls += 1
        return ToolResult(output={"exit_code": 0})


class _RecordingExecRuntime:
    def __init__(
        self,
        coordinator: TimelineCoordinator,
        *,
        fail_start: bool = False,
        status: ExecSessionStatus = ExecSessionStatus.COMPLETED,
    ) -> None:
        self._coordinator = coordinator
        self._fail_start = fail_start
        self._status = status
        self.calls = 0
        self.boundary_statuses: list[tuple[str, str]] = []

    def build_sandbox_plan(
        self,
        *,
        mode: str,
        cwd: Path,
        env: dict[str, str] | None,
        timeout_sec: float,
        requested_escalation: str,
        workspace_root: Path,
        **_: object,
    ):
        return create_sandbox_plan(
            mode=mode,  # type: ignore[arg-type]
            executable=sys.executable,
            argv=("-c", ""),
            cwd=cwd,
            read_roots=(workspace_root,),
            write_roots=(workspace_root,),
            network_policy="allow",
            env=env,
            resource_limits=SandboxResourceLimits(
                timeout_milliseconds=max(1, int(timeout_sec * 1000)),
                memory_limit_mb=0,
                max_processes=1,
                output_limit_bytes=1024,
            ),
            requested_escalation=requested_escalation,
            backend=HostSandboxBackend(),
        )

    async def start_command(self, **_: object) -> SessionPollResult:
        self.calls += 1
        loaded = await self._coordinator._repository.load(self._coordinator.session_id)  # noqa: SLF001
        assert loaded.timeline is not None
        descriptor = next(
            item
            for turn in loaded.timeline.turns
            for item in turn.operation_descriptors
        )
        self.boundary_statuses.append((descriptor.status, descriptor.precommit_boundary))
        if self._fail_start:
            raise RuntimeError("runtime start failed")
        return SessionPollResult(
            session_id="timeline-exec-session",
            shell="powershell",
            status=self._status,
            background=False,
            tty=False,
            pid=None,
            exit_code=0,
            timed_out=False,
            approval_state="not_required",
            stdout="timeline complete",
            stderr="",
        )


class _BindingMismatchExecTool(ExecCommandTool):
    def validates_timeline_approval_binding(
        self,
        approval_id: str,
        *,
        operation_id: str,
        arguments_digest: str,
        call_id: str,
    ) -> bool:
        del approval_id, operation_id, arguments_digest, call_id
        return False


def _timeline_exec_tool(
    *,
    runtime: _RecordingExecRuntime,
    workspace_dir: Path,
) -> ExecCommandTool:
    return ExecCommandTool(
        runtime=runtime,  # type: ignore[arg-type]
        workspace_dir=workspace_dir,
        require_approval=False,
        command_rules=[
            {
                "tokens": ["echo", "timeline"],
                "decision": "allow",
                "match": "exact",
                "shells": ["powershell"],
            }
        ],
    )


class _RecordingCoordinator(TimelineCoordinator):
    status_before_start: tuple[str, str] | None = None

    async def mark_mutation_started(self, *, operation_id: str) -> None:
        loaded = await self._repository.load(self.session_id)  # noqa: SLF001
        assert loaded.timeline is not None
        descriptor = next(
            item
            for turn in loaded.timeline.turns
            for item in turn.operation_descriptors
            if item.operation_id == operation_id
        )
        self.status_before_start = (descriptor.status, descriptor.precommit_boundary)
        await super().mark_mutation_started(operation_id=operation_id)


def _session_poll(
    status: ExecSessionStatus = ExecSessionStatus.RUNNING,
) -> SessionPollResult:
    return SessionPollResult(
        session_id="timeline-session",
        shell="powershell",
        status=status,
        background=True,
        tty=False,
        pid=123,
        exit_code=0 if status is ExecSessionStatus.COMPLETED else None,
        timed_out=status is ExecSessionStatus.TIMED_OUT,
        approval_state="not_required",
        stdout="",
        stderr="",
    )


class _RecordingSessionControlRuntime:
    def __init__(
        self,
        coordinator: TimelineCoordinator,
        *,
        preflight: SessionPollResult | None,
        result: SessionPollResult | None = None,
        failure: Exception | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._preflight = preflight
        self._result = result or preflight
        self._failure = failure
        self.write_calls = 0
        self.kill_calls = 0
        self.boundary_statuses: list[tuple[str, str]] = []

    async def inspect_session(self, _: str) -> SessionPollResult | None:
        return self._preflight

    async def _observe_boundary(self) -> None:
        loaded = await self._coordinator._repository.load(self._coordinator.session_id)  # noqa: SLF001
        assert loaded.timeline is not None
        descriptor = next(
            item
            for turn in loaded.timeline.turns
            for item in turn.operation_descriptors
        )
        self.boundary_statuses.append((descriptor.status, descriptor.precommit_boundary))

    async def write_stdin(self, *_: object, **__: object) -> SessionPollResult | None:
        self.write_calls += 1
        await self._observe_boundary()
        if self._failure is not None:
            raise self._failure
        return self._result

    async def kill_session(self, _: str) -> SessionPollResult | None:
        self.kill_calls += 1
        await self._observe_boundary()
        if self._failure is not None:
            raise self._failure
        return self._result


class _RecordingProcessService:
    def __init__(
        self,
        coordinator: TimelineCoordinator,
        *,
        preflight: dict[str, object] | None,
        result: dict[str, object] | None = None,
        failure: Exception | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._preflight = preflight
        self._result = result or preflight
        self._failure = failure
        self.stop_calls = 0
        self.boundary_statuses: list[tuple[str, str]] = []

    async def poll(self, _: str) -> dict[str, object] | None:
        return self._preflight

    async def stop(self, _: str) -> dict[str, object] | None:
        self.stop_calls += 1
        loaded = await self._coordinator._repository.load(self._coordinator.session_id)  # noqa: SLF001
        assert loaded.timeline is not None
        descriptor = next(
            item
            for turn in loaded.timeline.turns
            for item in turn.operation_descriptors
        )
        self.boundary_statuses.append((descriptor.status, descriptor.precommit_boundary))
        if self._failure is not None:
            raise self._failure
        return self._result


@pytest.mark.asyncio
async def test_file_write_precommits_before_writer_and_receipts_exact_result(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions")
    coordinator = _RecordingCoordinator(
        session_store=store,
        session_id="file-write-lifecycle",
        turn_id="turn-one",
    )
    await coordinator.admit_user_message(
        {
            "type": "message",
            "session_id": "file-write-lifecycle",
            "turn_id": "turn-one",
            "role": "user",
            "content": "write report",
        }
    )
    await coordinator.claim()
    observed: list[tuple[bool, tuple[str, str] | None]] = []

    async def writer(path: Path, content: str, append: bool, encoding: str) -> int:
        del append
        observed.append((path.exists(), coordinator.status_before_start))
        path.write_text(content, encoding=encoding)
        return len(content.encode(encoding))

    registry = ToolRegistry(discover_builtin=False)
    registry.register(
        FileWriteTool(workspace_dir=tmp_path, require_approval=False, writer=writer)
    )
    result = await registry.execute(
        "file_write",
        {"path": "report.md", "content": "# Report\n"},
        context=_context(tmp_path, coordinator),
    )

    assert result.error is None
    assert observed == [(False, ("precommitted", "not_started"))]
    operation_id = str(result.metadata["timeline_operation_id"])
    assert result.metadata["call_id"] == "file-call-1"
    assert result.metadata["arguments_digest"] == tool_arguments_digest(
        tool_name="file_write",
        arguments={"path": "report.md", "content": "# Report\n"},
    )
    descriptor = await _descriptor(store, "file-write-lifecycle", operation_id)()
    assert (descriptor.status, descriptor.precommit_boundary) == ("started", "started")

    await coordinator.persist_tool_result(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "file_write", "result": result.output, "error": result.error},
        error=result.error,
    )
    await coordinator.finish()
    descriptor = await _descriptor(store, "file-write-lifecycle", operation_id)()
    assert descriptor.status == "succeeded"
    assert descriptor.receipt_reference == "turn-one:1"


@pytest.mark.asyncio
async def test_duplicate_file_call_does_not_reach_writer_twice(tmp_path: Path) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id="duplicate-call")
    calls = 0

    async def writer(path: Path, content: str, append: bool, encoding: str) -> int:
        nonlocal calls
        del append
        calls += 1
        path.write_text(content, encoding=encoding)
        return len(content.encode(encoding))

    registry = ToolRegistry(discover_builtin=False)
    registry.register(
        FileWriteTool(workspace_dir=tmp_path, require_approval=False, writer=writer)
    )
    context = _context(tmp_path, coordinator, call_id="same-call")
    first = await registry.execute(
        "file_write",
        {"path": "once.txt", "content": "once"},
        context=context,
    )
    second = await registry.execute(
        "file_write",
        {"path": "once.txt", "content": "once"},
        context=context,
    )

    assert first.error is None
    assert second.error is not None
    assert second.metadata["timeline_fail_closed"] is True
    assert calls == 1
    operation_id = str(first.metadata["timeline_operation_id"])
    await coordinator.persist_tool_result(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "file_write", "result": first.output, "error": first.error},
        error=first.error,
    )
    await coordinator.finish(failed=True)
    loaded = await SessionTurnTimelineRepository(store).load("duplicate-call")
    assert loaded.timeline is not None
    assert loaded.timeline.turns[0].terminal_outcome == "blocked"


@pytest.mark.asyncio
async def test_read_only_file_access_creates_no_operation_descriptor(tmp_path: Path) -> None:
    (tmp_path / "readme.txt").write_text("read only", encoding="utf-8")
    coordinator, store = await _claimed_coordinator(tmp_path, session_id="read-only")
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir=tmp_path))

    result = await registry.execute(
        "file_read",
        {"path": "readme.txt"},
        context=_context(tmp_path, coordinator),
    )

    assert result.error is None
    loaded = await SessionTurnTimelineRepository(store).load("read-only")
    assert loaded.timeline is not None
    assert loaded.timeline.turns[0].operation_descriptors == ()
    await coordinator.finish()


@pytest.mark.asyncio
async def test_same_name_unaware_file_write_is_never_dispatched(
    tmp_path: Path,
) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id="unaware-file-write")
    tool = _UnawareFileWriteProbe()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(tool)

    result = await registry.execute(
        "file_write",
        {"path": "never-created.txt"},
        context=_context(tmp_path, coordinator),
    )

    assert result.metadata["status"] == "timeline_boundary_aware_tool_required"
    assert result.metadata["timeline_fail_closed"] is True
    assert tool.calls == 0
    await coordinator.finish()
    loaded = await SessionTurnTimelineRepository(store).load("unaware-file-write")
    assert loaded.timeline is not None
    assert loaded.timeline.turns[0].operation_descriptors == ()
    assert loaded.timeline.turns[0].terminal_outcome == "blocked"


@pytest.mark.asyncio
async def test_exec_command_precommits_and_starts_before_runtime_dispatch(tmp_path: Path) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id="exec-lifecycle")
    runtime = _RecordingExecRuntime(coordinator)
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_timeline_exec_tool(runtime=runtime, workspace_dir=tmp_path))

    result = await registry.execute(
        "exec_command",
        {"command": "echo timeline", "shell": "powershell"},
        context=_context(tmp_path, coordinator, call_id="exec-call-1"),
    )

    assert result.error is None
    assert runtime.calls == 1
    assert runtime.boundary_statuses == [("started", "started")]
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.persist_tool_result(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "exec_command", "result": result.output, "error": result.error},
        error=result.error,
    )
    await coordinator.finish()

    descriptor = await _descriptor(store, "exec-lifecycle", operation_id)()
    assert descriptor.status == "succeeded"
    assert descriptor.receipt_reference == "turn-one:1"


@pytest.mark.asyncio
async def test_exec_runtime_start_failure_is_quarantined_as_unknown(tmp_path: Path) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id="exec-start-failure")
    runtime = _RecordingExecRuntime(coordinator, fail_start=True)
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_timeline_exec_tool(runtime=runtime, workspace_dir=tmp_path))

    result = await registry.execute(
        "exec_command",
        {"command": "echo timeline", "shell": "powershell"},
        context=_context(tmp_path, coordinator, call_id="exec-call-1"),
    )

    assert result.error == "Exec command failed: runtime start failed"
    assert runtime.calls == 1
    assert runtime.boundary_statuses == [("started", "started")]
    assert result.metadata["timeline_result_unknown"] is True
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.persist_tool_result(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "exec_command", "error": result.error},
        error=result.error,
        unknown=True,
    )
    await coordinator.finish()

    descriptor = await _descriptor(store, "exec-start-failure", operation_id)()
    assert (descriptor.status, descriptor.precommit_boundary) == ("unknown", "unknown")


@pytest.mark.asyncio
async def test_post_precommit_exec_approval_preserves_exact_continuation(
    tmp_path: Path,
) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id="exec-approval-orphan")
    runtime = _RecordingExecRuntime(coordinator)
    approvals = InMemoryApprovalStore()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(
        ExecCommandTool(
            runtime=runtime,  # type: ignore[arg-type]
            approval_store=approvals,
            workspace_dir=tmp_path,
            require_approval=False,
        )
    )

    context = _context(tmp_path, coordinator, call_id="exec-approval-1")
    context.state["ordinary_chat_approval_context"] = {
        "source": "ordinary_chat",
        "session_id": "exec-approval-orphan",
        "turn_id": "turn-one",
        "resume_cursor": {"tool_call_id": "exec-approval-1"},
    }
    arguments = {"command": "echo requires approval", "shell": "powershell"}
    result = await registry.execute(
        "exec_command",
        arguments,
        context=context,
    )

    approval_id = str(result.metadata["approval_id"])
    approval = approvals.get(approval_id)
    assert result.metadata["timeline_approval_pending"] is True
    assert result.metadata["timeline_approval_mode"] == "continuable"
    assert runtime.calls == 0
    assert approval is not None
    assert approval.status == "pending"
    assert approval.metadata["timeline_call_id"] == "exec-approval-1"
    assert approval.command_payload["normalized_arguments"] == arguments
    assert approval.command_payload["ordinary_chat_checkpoint"]["timeline_call_id"] == "exec-approval-1"

    operation_id = str(result.metadata["timeline_operation_id"])
    assert result.metadata["arguments_digest"] == tool_arguments_digest(
        tool_name="exec_command",
        arguments=arguments,
    )
    descriptor = await _descriptor(store, "exec-approval-orphan", operation_id)()
    assert (descriptor.status, descriptor.precommit_boundary) == ("precommitted", "not_started")
    await coordinator.finish(failed=True)


@pytest.mark.asyncio
async def test_mismatched_post_precommit_exec_approval_is_superseded(
    tmp_path: Path,
) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id="exec-approval-race")
    runtime = _RecordingExecRuntime(coordinator)
    approvals = InMemoryApprovalStore()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(
        _BindingMismatchExecTool(
            runtime=runtime,  # type: ignore[arg-type]
            approval_store=approvals,
            workspace_dir=tmp_path,
            require_approval=False,
        )
    )

    context = _context(tmp_path, coordinator, call_id="exec-approval-race-1")
    context.state["ordinary_chat_approval_context"] = {
        "source": "ordinary_chat",
        "session_id": "exec-approval-race",
        "turn_id": "turn-one",
        "resume_cursor": {"tool_call_id": "exec-approval-race-1"},
    }
    result = await registry.execute(
        "exec_command",
        {"command": "echo requires approval", "shell": "powershell"},
        context=context,
    )

    assert result.metadata["timeline_approval_invalidated"] is True
    assert runtime.calls == 0
    approval = approvals.get(str(result.metadata["approval_id"]))
    assert approval is not None
    assert approval.status == "superseded"
    assert approvals.list(status="pending") == []
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.abandon_pre_effect_operation(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "exec_command", "error": result.error},
    )
    await coordinator.finish(failed=True)

    descriptor = await _descriptor(store, "exec-approval-race", operation_id)()
    assert (descriptor.status, descriptor.precommit_boundary) == ("abandoned", "not_started")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        ExecSessionStatus.FAILED,
        ExecSessionStatus.TIMED_OUT,
        ExecSessionStatus.KILLED,
    ],
)
async def test_terminal_exec_outcomes_are_known_failures(
    tmp_path: Path,
    status: ExecSessionStatus,
) -> None:
    session_id = f"exec-{status.value}"
    coordinator, store = await _claimed_coordinator(tmp_path, session_id=session_id)
    runtime = _RecordingExecRuntime(coordinator, status=status)
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_timeline_exec_tool(runtime=runtime, workspace_dir=tmp_path))

    result = await registry.execute(
        "exec_command",
        {"command": "echo timeline", "shell": "powershell"},
        context=_context(tmp_path, coordinator, call_id=f"exec-{status.value}"),
    )

    assert result.metadata["timeline_result_disposition"] == "failed"
    if status is ExecSessionStatus.KILLED:
        assert result.error is None
    else:
        assert result.error is not None
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.persist_tool_result(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "exec_command", "result": result.output, "error": result.error},
        error=result.error,
        disposition=str(result.metadata["timeline_result_disposition"]),
    )
    await coordinator.finish(failed=True)

    descriptor = await _descriptor(store, session_id, operation_id)()
    assert descriptor.status == "failed"


@pytest.mark.asyncio
async def test_execute_code_precommits_before_runner_dispatch(tmp_path: Path) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id="execute-code-lifecycle")
    observed: list[tuple[str, str]] = []

    async def runner(_: str, __: Path, ___: int, ____: str) -> tuple[int, str, str]:
        loaded = await coordinator._repository.load(coordinator.session_id)  # noqa: SLF001
        assert loaded.timeline is not None
        descriptor = loaded.timeline.turns[0].operation_descriptors[0]
        observed.append((descriptor.status, descriptor.precommit_boundary))
        return 0, "ok", ""

    registry = ToolRegistry(discover_builtin=False)
    registry.register(
        ExecuteCodeTool(workspace_dir=tmp_path, require_approval=False, runner=runner)
    )
    result = await registry.execute(
        "execute_code",
        {"code": "print('ok')"},
        context=_context(tmp_path, coordinator, call_id="execute-code-call"),
    )

    assert result.error is None
    assert observed == [("started", "started")]
    assert result.metadata["timeline_result_disposition"] == "succeeded"
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.persist_tool_result(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "execute_code", "result": result.output, "error": result.error},
        error=result.error,
        disposition=str(result.metadata["timeline_result_disposition"]),
    )
    await coordinator.finish()
    descriptor = await _descriptor(store, "execute-code-lifecycle", operation_id)()
    assert descriptor.status == "succeeded"


@pytest.mark.asyncio
async def test_execute_code_v2_precommits_before_runner_dispatch(tmp_path: Path) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id="execute-code-v2-lifecycle")
    observed: list[tuple[str, str]] = []

    async def runner(
        _: str,
        __: Path,
        ___: int,
        ____: str,
        _____: Path,
        ______: list[str],
    ) -> dict[str, object]:
        loaded = await coordinator._repository.load(coordinator.session_id)  # noqa: SLF001
        assert loaded.timeline is not None
        descriptor = loaded.timeline.turns[0].operation_descriptors[0]
        observed.append((descriptor.status, descriptor.precommit_boundary))
        return {"stdout": "ok", "result": None, "tool_calls": []}

    registry = ToolRegistry(discover_builtin=False)
    registry.register(
        ExecuteCodeV2Tool(workspace_dir=tmp_path, require_approval=False, runner=runner)
    )
    result = await registry.execute(
        "execute_code_v2",
        {"code": "result = 'ok'"},
        context=_context(tmp_path, coordinator, call_id="execute-code-v2-call"),
    )

    assert result.error is None
    assert observed == [("started", "started")]
    assert result.metadata["timeline_result_disposition"] == "succeeded"
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.persist_tool_result(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "execute_code_v2", "result": result.output, "error": result.error},
        error=result.error,
        disposition=str(result.metadata["timeline_result_disposition"]),
    )
    await coordinator.finish()
    descriptor = await _descriptor(store, "execute-code-v2-lifecycle", operation_id)()
    assert descriptor.status == "succeeded"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["execute_code", "execute_code_v2"])
async def test_execute_code_dispatch_failure_is_unknown(
    tmp_path: Path,
    tool_name: str,
) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id=f"{tool_name}-failure")

    async def code_runner(*_: object) -> tuple[int, str, str]:
        raise RuntimeError("dispatch transport failed")

    async def v2_runner(*_: object) -> dict[str, object]:
        raise RuntimeError("dispatch transport failed")

    registry = ToolRegistry(discover_builtin=False)
    if tool_name == "execute_code":
        registry.register(
            ExecuteCodeTool(workspace_dir=tmp_path, require_approval=False, runner=code_runner)
        )
        arguments = {"code": "print('never confirmed')"}
    else:
        registry.register(
            ExecuteCodeV2Tool(workspace_dir=tmp_path, require_approval=False, runner=v2_runner)
        )
        arguments = {"code": "result = 'never confirmed'"}

    result = await registry.execute(
        tool_name,
        arguments,
        context=_context(tmp_path, coordinator, call_id=f"{tool_name}-failure-call"),
    )

    assert result.error == "Code execution failed: dispatch transport failed"
    assert result.metadata["timeline_result_unknown"] is True
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.persist_tool_result(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": tool_name, "error": result.error},
        error=result.error,
        disposition="unknown",
    )
    await coordinator.finish()
    descriptor = await _descriptor(store, f"{tool_name}-failure", operation_id)()
    assert descriptor.status == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["write_stdin", "kill_session"])
async def test_missing_exec_session_is_abandoned_before_control_dispatch(
    tmp_path: Path,
    tool_name: str,
) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id=f"{tool_name}-missing")
    runtime = _RecordingSessionControlRuntime(coordinator, preflight=None)
    registry = ToolRegistry(discover_builtin=False)
    if tool_name == "write_stdin":
        registry.register(WriteStdinTool(runtime=runtime))  # type: ignore[arg-type]
        arguments = {"session_id": "missing", "chars": "input"}
    else:
        registry.register(KillSessionTool(runtime=runtime))  # type: ignore[arg-type]
        arguments = {"session_id": "missing"}

    result = await registry.execute(
        tool_name,
        arguments,
        context=_context(tmp_path, coordinator, call_id=f"{tool_name}-missing-call"),
    )

    assert result.metadata["timeline_pre_effect_abandoned"] is True
    assert runtime.write_calls + runtime.kill_calls == 0
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.abandon_pre_effect_operation(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": tool_name, "error": result.error},
    )
    await coordinator.finish(failed=True)
    descriptor = await _descriptor(store, f"{tool_name}-missing", operation_id)()
    assert (descriptor.status, descriptor.precommit_boundary) == ("abandoned", "not_started")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "terminal_status"),
    [
        ("write_stdin", ExecSessionStatus.COMPLETED),
        ("kill_session", ExecSessionStatus.KILLED),
    ],
)
async def test_terminal_exec_session_is_abandoned_before_control_dispatch(
    tmp_path: Path,
    tool_name: str,
    terminal_status: ExecSessionStatus,
) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id=f"{tool_name}-preflight-terminal")
    runtime = _RecordingSessionControlRuntime(
        coordinator,
        preflight=_session_poll(terminal_status),
    )
    registry = ToolRegistry(discover_builtin=False)
    if tool_name == "write_stdin":
        registry.register(WriteStdinTool(runtime=runtime))  # type: ignore[arg-type]
        arguments = {"session_id": "timeline-session", "chars": "input"}
    else:
        registry.register(KillSessionTool(runtime=runtime))  # type: ignore[arg-type]
        arguments = {"session_id": "timeline-session"}

    result = await registry.execute(
        tool_name,
        arguments,
        context=_context(tmp_path, coordinator, call_id=f"{tool_name}-preflight-terminal-call"),
    )

    assert result.metadata["timeline_pre_effect_abandoned"] is True
    assert runtime.write_calls + runtime.kill_calls == 0
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.abandon_pre_effect_operation(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": tool_name, "error": result.error},
    )
    await coordinator.finish(failed=True)
    descriptor = await _descriptor(store, f"{tool_name}-preflight-terminal", operation_id)()
    assert (descriptor.status, descriptor.precommit_boundary) == ("abandoned", "not_started")


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["write_stdin", "kill_session"])
async def test_exec_session_control_transport_failure_is_unknown(
    tmp_path: Path,
    tool_name: str,
) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id=f"{tool_name}-transport")
    runtime = _RecordingSessionControlRuntime(
        coordinator,
        preflight=_session_poll(),
        failure=RuntimeError("control transport failed"),
    )
    registry = ToolRegistry(discover_builtin=False)
    if tool_name == "write_stdin":
        registry.register(WriteStdinTool(runtime=runtime))  # type: ignore[arg-type]
        arguments = {"session_id": "timeline-session", "chars": "input"}
    else:
        registry.register(KillSessionTool(runtime=runtime))  # type: ignore[arg-type]
        arguments = {"session_id": "timeline-session"}

    result = await registry.execute(
        tool_name,
        arguments,
        context=_context(tmp_path, coordinator, call_id=f"{tool_name}-transport-call"),
    )

    assert runtime.boundary_statuses == [("started", "started")]
    assert result.metadata["timeline_result_unknown"] is True
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.persist_tool_result(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": tool_name, "error": result.error},
        error=result.error,
        disposition="unknown",
    )
    await coordinator.finish()
    descriptor = await _descriptor(store, f"{tool_name}-transport", operation_id)()
    assert descriptor.status == "unknown"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "runtime_result", "expected_disposition"),
    [
        ("write_stdin", _session_poll(ExecSessionStatus.FAILED), "failed"),
        ("kill_session", _session_poll(ExecSessionStatus.KILLED), "succeeded"),
    ],
)
async def test_exec_session_control_terminal_outcomes_are_known(
    tmp_path: Path,
    tool_name: str,
    runtime_result: SessionPollResult,
    expected_disposition: str,
) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id=f"{tool_name}-terminal")
    runtime = _RecordingSessionControlRuntime(
        coordinator,
        preflight=_session_poll(),
        result=runtime_result,
    )
    registry = ToolRegistry(discover_builtin=False)
    if tool_name == "write_stdin":
        registry.register(WriteStdinTool(runtime=runtime))  # type: ignore[arg-type]
        arguments = {"session_id": "timeline-session", "chars": "input"}
    else:
        registry.register(KillSessionTool(runtime=runtime))  # type: ignore[arg-type]
        arguments = {"session_id": "timeline-session"}

    result = await registry.execute(
        tool_name,
        arguments,
        context=_context(tmp_path, coordinator, call_id=f"{tool_name}-terminal-call"),
    )

    assert runtime.boundary_statuses == [("started", "started")]
    assert result.metadata["timeline_result_disposition"] == expected_disposition
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.persist_tool_result(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": tool_name, "result": result.output, "error": result.error},
        error=result.error,
        disposition=expected_disposition,
    )
    await coordinator.finish(failed=expected_disposition == "failed")
    descriptor = await _descriptor(store, f"{tool_name}-terminal", operation_id)()
    assert descriptor.status == expected_disposition


@pytest.mark.asyncio
async def test_process_stop_preflight_and_transport_failure_are_durable(tmp_path: Path) -> None:
    missing, missing_store = await _claimed_coordinator(tmp_path, session_id="process-stop-missing")
    missing_service = _RecordingProcessService(missing, preflight=None)
    missing_registry = ToolRegistry(discover_builtin=False)
    missing_registry.register(ProcessStopTool(process_service=missing_service))  # type: ignore[arg-type]
    missing_result = await missing_registry.execute(
        "process_stop",
        {"process_id": "missing"},
        context=_context(tmp_path, missing, call_id="process-stop-missing-call"),
    )

    assert missing_result.metadata["timeline_pre_effect_abandoned"] is True
    assert missing_service.stop_calls == 0
    missing_operation = str(missing_result.metadata["timeline_operation_id"])
    await missing.abandon_pre_effect_operation(
        operation_id=missing_operation,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "process_stop", "error": missing_result.error},
    )
    await missing.finish(failed=True)
    missing_descriptor = await _descriptor(missing_store, "process-stop-missing", missing_operation)()
    assert missing_descriptor.status == "abandoned"

    coordinator, store = await _claimed_coordinator(tmp_path, session_id="process-stop-transport")
    service = _RecordingProcessService(
        coordinator,
        preflight={"process_id": "proc-1", "status": "running"},
        failure=RuntimeError("stop transport failed"),
    )
    registry = ToolRegistry(discover_builtin=False)
    registry.register(ProcessStopTool(process_service=service))  # type: ignore[arg-type]
    result = await registry.execute(
        "process_stop",
        {"process_id": "proc-1"},
        context=_context(tmp_path, coordinator, call_id="process-stop-transport-call"),
    )

    assert service.boundary_statuses == [("started", "started")]
    assert result.metadata["timeline_result_unknown"] is True
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.persist_tool_result(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "process_stop", "error": result.error},
        error=result.error,
        disposition="unknown",
    )
    await coordinator.finish()
    descriptor = await _descriptor(store, "process-stop-transport", operation_id)()
    assert descriptor.status == "unknown"


@pytest.mark.asyncio
async def test_process_stop_confirmed_terminal_outcome_is_known_success(tmp_path: Path) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id="process-stop-terminal")
    service = _RecordingProcessService(
        coordinator,
        preflight={"process_id": "proc-1", "status": "running"},
        result={
            "process_id": "proc-1",
            "status": "exited",
            "stopped": True,
            "stop_signal": "terminate",
        },
    )
    registry = ToolRegistry(discover_builtin=False)
    registry.register(ProcessStopTool(process_service=service))  # type: ignore[arg-type]
    result = await registry.execute(
        "process_stop",
        {"process_id": "proc-1"},
        context=_context(tmp_path, coordinator, call_id="process-stop-terminal-call"),
    )

    assert service.boundary_statuses == [("started", "started")]
    assert result.metadata["timeline_result_disposition"] == "succeeded"
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.persist_tool_result(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "process_stop", "result": result.output, "error": result.error},
        error=result.error,
        disposition="succeeded",
    )
    await coordinator.finish()
    descriptor = await _descriptor(store, "process-stop-terminal", operation_id)()
    assert descriptor.status == "succeeded"


@pytest.mark.asyncio
async def test_process_stop_exit_race_is_known_failure(tmp_path: Path) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id="process-stop-exit-race")
    service = _RecordingProcessService(
        coordinator,
        preflight={"process_id": "proc-1", "status": "running"},
        result={
            "process_id": "proc-1",
            "status": "exited",
            "stopped": True,
            "stop_signal": None,
        },
    )
    registry = ToolRegistry(discover_builtin=False)
    registry.register(ProcessStopTool(process_service=service))  # type: ignore[arg-type]
    result = await registry.execute(
        "process_stop",
        {"process_id": "proc-1"},
        context=_context(tmp_path, coordinator, call_id="process-stop-exit-race-call"),
    )

    assert service.boundary_statuses == [("started", "started")]
    assert result.metadata["timeline_result_disposition"] == "failed"
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.persist_tool_result(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "process_stop", "result": result.output, "error": result.error},
        error=result.error,
        disposition="failed",
    )
    await coordinator.finish(failed=True)
    descriptor = await _descriptor(store, "process-stop-exit-race", operation_id)()
    assert descriptor.status == "failed"


@pytest.mark.asyncio
async def test_terminal_process_preflight_is_abandoned_before_stop_dispatch(tmp_path: Path) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id="process-stop-preflight-terminal")
    service = _RecordingProcessService(
        coordinator,
        preflight={"process_id": "proc-1", "status": "exited", "stopped": False},
    )
    registry = ToolRegistry(discover_builtin=False)
    registry.register(ProcessStopTool(process_service=service))  # type: ignore[arg-type]
    result = await registry.execute(
        "process_stop",
        {"process_id": "proc-1"},
        context=_context(tmp_path, coordinator, call_id="process-stop-preflight-terminal-call"),
    )

    assert result.metadata["timeline_pre_effect_abandoned"] is True
    assert service.stop_calls == 0
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.abandon_pre_effect_operation(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "process_stop", "error": result.error},
    )
    await coordinator.finish(failed=True)
    descriptor = await _descriptor(store, "process-stop-preflight-terminal", operation_id)()
    assert (descriptor.status, descriptor.precommit_boundary) == ("abandoned", "not_started")


@pytest.mark.asyncio
async def test_pre_effect_result_is_abandoned_with_its_atomic_receipt(tmp_path: Path) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id="abandoned-operation")
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_PreEffectFileWrite())

    result = await registry.execute(
        "file_write",
        {"path": "never-created.txt"},
        context=_context(tmp_path, coordinator),
    )

    assert result.error == "timeline_effect_boundary_not_reached: mutation was stopped before a durable effect boundary."
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.abandon_pre_effect_operation(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "file_write", "error": result.error},
    )
    await coordinator.finish(failed=True)

    descriptor = await _descriptor(store, "abandoned-operation", operation_id)()
    assert (descriptor.status, descriptor.precommit_boundary) == ("abandoned", "not_started")
    assert descriptor.receipt_reference == "turn-one:1"
    loaded = await SessionTurnTimelineRepository(store).load("abandoned-operation")
    assert loaded.timeline is not None
    assert loaded.timeline.turns[0].terminal_outcome == "blocked"
    snapshot = await store.load_strict_snapshot("abandoned-operation")
    assert any(
        event.get("type") == "turn_event" and event.get("event_id") == "turn-one:1"
        for event in snapshot.events
    )


@pytest.mark.asyncio
async def test_started_writer_failure_quarantines_the_operation_as_unknown(tmp_path: Path) -> None:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id="writer-failure")

    async def failing_writer(_: Path, __: str, ___: bool, ____: str) -> int:
        raise RuntimeError("writer crashed after the effect boundary")

    registry = ToolRegistry(discover_builtin=False)
    registry.register(
        FileWriteTool(
            workspace_dir=tmp_path,
            require_approval=False,
            writer=failing_writer,
        )
    )
    result = await registry.execute(
        "file_write",
        {"path": "unknown.txt", "content": "payload"},
        context=_context(tmp_path, coordinator),
    )

    assert result.error is not None
    assert result.metadata["timeline_result_unknown"] is True
    operation_id = str(result.metadata["timeline_operation_id"])
    await coordinator.persist_tool_result(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "file_write", "error": result.error},
        error=result.error,
        unknown=True,
    )
    await coordinator.finish()

    descriptor = await _descriptor(store, "writer-failure", operation_id)()
    assert (descriptor.status, descriptor.precommit_boundary) == ("unknown", "unknown")
    loaded = await SessionTurnTimelineRepository(store).load("writer-failure")
    assert loaded.timeline is not None
    assert loaded.timeline.turns[0].terminal_outcome == "unknown"


@pytest.mark.asyncio
async def test_file_write_predicted_approval_without_store_fails_closed_before_effect(
    tmp_path: Path,
) -> None:
    first, store = await _claimed_coordinator(tmp_path, session_id="blocked-lane")
    writer_calls = 0

    async def writer(_: Path, __: str, ___: bool, ____: str) -> int:
        nonlocal writer_calls
        writer_calls += 1
        return 0

    registry = ToolRegistry(discover_builtin=False)
    registry.register(
        FileWriteTool(workspace_dir=tmp_path, require_approval=False, writer=writer)
    )
    blocked = await registry.execute(
        "file_write",
        {"path": "approval.txt", "content": "must not write"},
        context=_context(
            tmp_path,
            first,
            permission_policy={"require_approval_for_file_write": True},
        ),
    )
    assert blocked.error == (
        "timeline_effect_boundary_not_reached: mutation was stopped before a durable effect boundary."
    )
    assert blocked.metadata["timeline_pre_effect_abandoned"] is True
    assert writer_calls == 0
    operation_id = str(blocked.metadata["timeline_operation_id"])
    await first.abandon_pre_effect_operation(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "file_write", "error": blocked.error},
    )
    await first.finish(failed=True)
    descriptor = await _descriptor(store, "blocked-lane", operation_id)()
    assert (descriptor.status, descriptor.precommit_boundary) == ("abandoned", "not_started")


@pytest.mark.asyncio
async def test_file_write_predicted_approval_with_store_preserves_exact_continuation(
    tmp_path: Path,
) -> None:
    first, store = await _claimed_coordinator(tmp_path, session_id="blocked-lane-continuable")
    writer_calls = 0

    async def writer(_: Path, __: str, ___: bool, ____: str) -> int:
        nonlocal writer_calls
        writer_calls += 1
        return 0

    approvals = InMemoryApprovalStore()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(
        FileWriteTool(
            workspace_dir=tmp_path,
            require_approval=False,
            approval_store=approvals,
            writer=writer,
        )
    )
    context = _context(
        tmp_path,
        first,
        permission_policy={"require_approval_for_file_write": True},
    )
    context.state["ordinary_chat_approval_context"] = {
        "source": "ordinary_chat",
        "session_id": "blocked-lane-continuable",
        "turn_id": "turn-one",
        "resume_cursor": {
            "turn_id": "turn-one",
            "phase": "tool_call",
            "tool_call_id": "file-call-1",
            "tool_name": "file_write",
        },
    }
    arguments = {"path": "approval.txt", "content": "must not write"}
    pending = await registry.execute("file_write", arguments, context=context)
    assert pending.metadata["timeline_approval_pending"] is True
    assert pending.metadata["timeline_approval_mode"] == "continuable"
    assert writer_calls == 0
    approval_id = str(pending.metadata["approval_id"])
    approval = approvals.get(approval_id)
    assert approval is not None
    assert approval.status == "pending"
    operation_id = str(pending.metadata["timeline_operation_id"])
    assert approval.metadata["operation_id"] == operation_id
    assert approval.metadata["timeline_call_id"] == "file-call-1"
    assert approval.metadata["arguments_digest"] == pending.metadata["arguments_digest"]
    assert approval.command_payload["ordinary_chat_checkpoint"]["operation_id"] == operation_id
    assert approval.command_payload["ordinary_chat_checkpoint"]["timeline_call_id"] == "file-call-1"

    await first.persist_approval_pending(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "file_write", "metadata": dict(pending.metadata)},
    )
    await first.finish(
        companion_events=(
            {
                "type": "message",
                "session_id": "blocked-lane-continuable",
                "turn_id": "turn-one",
                "role": "assistant",
                "content": "approval-blocked transcript",
            },
        )
    )
    descriptor = await _descriptor(store, "blocked-lane-continuable", operation_id)()
    assert (descriptor.status, descriptor.precommit_boundary) == ("precommitted", "not_started")

    second = TimelineCoordinator(
        session_store=store,
        session_id="blocked-lane-continuable",
        turn_id="turn-two",
    )
    await second.admit_user_message(
        {
            "type": "message",
            "session_id": "blocked-lane-continuable",
            "turn_id": "turn-two",
            "role": "user",
            "content": "follow-up",
        }
    )
    history = await asyncio.wait_for(second.claim(), timeout=1)
    assert any(event.get("content") == "approval-blocked transcript" for event in history)
    loaded = await SessionTurnTimelineRepository(store).load("blocked-lane-continuable")
    assert loaded.timeline is not None
    assert loaded.timeline.turns[0].terminal_outcome == "blocked"
    with pytest.raises(TimelineCoordinatorError, match="pending ordinary-Chat continuation"):
        await second.precommit_mutation(
            tool_name="file_write",
            arguments={"path": "later.txt", "content": "blocked"},
            call_id="file-call-2",
        )
    await second.finish()


@pytest.mark.asyncio
async def test_engine_persists_killed_exec_as_known_failure_before_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MochiConfig.model_validate(
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
    engine = AgentEngine(config)
    store = SessionStore(tmp_path / "sessions")
    coordinator = TimelineCoordinator(
        session_store=store,
        session_id="callback-order",
        turn_id="turn-one",
    )
    await coordinator.admit_user_message(
        {
            "type": "message",
            "session_id": "callback-order",
            "turn_id": "turn-one",
            "role": "user",
            "content": "run command",
        }
    )
    history = list(await coordinator.claim())
    operation_id, _ = await coordinator.precommit_mutation(
        tool_name="exec_command",
        arguments={"command": "echo timeline", "shell": "powershell"},
        call_id="exec-1",
    )
    await coordinator.mark_mutation_started(operation_id=operation_id)

    class _ReceiptLoop:
        def __init__(self, **_: object) -> None:
            self.turn_messages = [Message(role="assistant", content="command cancelled")]

        async def run(self, **_: object):
            yield ToolCallResultEvent(
                call_id="exec-1",
                tool_name="exec_command",
                result={"runtime_status": "killed", "exit_code": None},
                metadata={
                    "timeline_operation_id": operation_id,
                    "timeline_result_disposition": "failed",
                },
            )
            yield ToolCallCompletedEvent(
                call_id="exec-1",
                tool_name="exec_command",
                arguments={"command": "echo timeline", "shell": "powershell"},
                result={"runtime_status": "killed", "exit_code": None},
                metadata={"timeline_operation_id": operation_id},
            )
            yield FinalAnswerEvent(content="command cancelled")

    monkeypatch.setattr(engine_module, "AsyncReActLoop", _ReceiptLoop)
    backend = FakeBackend()

    async def load(_: str) -> FakeBackend:
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = load  # type: ignore[method-assign]
    observed: list[tuple[str, str, str | None]] = []

    async def callback(event: object) -> None:
        if not isinstance(event, (ToolCallResultEvent, ToolCallCompletedEvent)):
            return
        descriptor = await _descriptor(store, "callback-order", operation_id)()
        observed.append((event.type, descriptor.status, descriptor.receipt_reference))

    await engine._invoke_shared_runtime(  # noqa: SLF001
        AgentInvocationRequest(
            message="run command",
            session_id="callback-order",
            execution_profile="chat",
            persist_session=False,
            turn_id="turn-one",
            timeline_history_events=history,
            timeline_user_message_admitted=True,
            timeline_coordinator=coordinator,
        ),
        event_callback=callback,
    )

    assert observed == [
        ("tool_call_result", "failed", "turn-one:1"),
        ("tool_call_completed", "failed", "turn-one:1"),
    ]
    await coordinator.finish()
    await engine.close()


async def _terminal_pending_approval_operation(
    tmp_path: Path,
    *,
    session_id: str,
) -> tuple[SessionStore, str, str]:
    coordinator, store = await _claimed_coordinator(tmp_path, session_id=session_id)
    operation_id, arguments_digest = await coordinator.precommit_mutation(
        tool_name="file_write",
        arguments={"path": "approval.txt", "content": "approved"},
        call_id="approval-call-1",
    )
    await coordinator.persist_approval_pending(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={
            "tool_name": "file_write",
            "metadata": {
                "timeline_approval_pending": True,
                "operation_id": operation_id,
                "arguments_digest": arguments_digest,
                "call_id": "approval-call-1",
            },
        },
    )
    await coordinator.block_unstarted_turn()
    await coordinator.finish()
    return store, operation_id, arguments_digest


@pytest.mark.asyncio
async def test_terminal_pending_approval_receipt_releases_fifo_but_blocks_follower_mutation(
    tmp_path: Path,
) -> None:
    store, operation_id, arguments_digest = await _terminal_pending_approval_operation(
        tmp_path,
        session_id="pending-approval-fifo",
    )
    repository = SessionTurnTimelineRepository(store)
    loaded = await repository.load("pending-approval-fifo")
    assert loaded.timeline is not None
    first = loaded.timeline.turns[0]
    descriptor = first.operation_descriptors[0]
    assert first.terminal_outcome == "blocked"
    assert loaded.timeline.lane_turn_id is None
    assert (descriptor.operation_id, descriptor.arguments_digest) == (
        operation_id,
        arguments_digest,
    )
    assert (descriptor.status, descriptor.precommit_boundary) == ("precommitted", "not_started")
    snapshot = await store.load_strict_snapshot("pending-approval-fifo")
    assert any(
        event.get("type") == "turn_event" and event.get("event_id") == "turn-one:1"
        for event in snapshot.events
    )

    (tmp_path / "read-only.txt").write_text("readable", encoding="utf-8")
    follower = TimelineCoordinator(
        session_store=store,
        session_id="pending-approval-fifo",
        turn_id="turn-two",
    )
    await follower.admit_user_message(
        {
            "type": "message",
            "schema_version": 1,
            "session_id": "pending-approval-fifo",
            "turn_id": "turn-two",
            "role": "user",
            "content": "read and then write",
        }
    )
    await follower.claim()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir=tmp_path))
    read = await registry.execute(
        "file_read",
        {"path": "read-only.txt"},
        context=_context(tmp_path, follower, call_id="follower-read"),
    )
    assert read.error is None
    with pytest.raises(TimelineCoordinatorError, match="pending ordinary-Chat continuation"):
        await follower.precommit_mutation(
            tool_name="file_write",
            arguments={"path": "later.txt", "content": "blocked"},
            call_id="follower-write",
        )
    await follower.finish(failed=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("result_status", ["succeeded", "failed", "unknown"])
async def test_terminal_approval_start_has_one_winner_and_late_abandon_cannot_overwrite(
    tmp_path: Path,
    result_status: str,
) -> None:
    store, operation_id, arguments_digest = await _terminal_pending_approval_operation(
        tmp_path,
        session_id="pending-approval-resume",
    )
    repository = SessionTurnTimelineRepository(store)
    loaded = await repository.load("pending-approval-resume")
    assert loaded.history_revision is not None
    start = lambda: repository.mark_terminal_precommitted_operation_started(
        "pending-approval-resume",
        turn_id="turn-one",
        expected_history_revision=loaded.history_revision,
        operation_id=operation_id,
        call_id="approval-call-1",
        arguments_digest=arguments_digest,
    )
    first, second = await asyncio.gather(start(), start())
    assert {first.status, second.status} == {"boundary_updated", "rebase_required"}

    started = await repository.load("pending-approval-resume")
    assert started.history_revision is not None
    assert started.timeline is not None
    assert started.timeline.turns[0].operation_descriptors[0].status == "started"
    duplicate_start = await repository.mark_terminal_precommitted_operation_started(
        "pending-approval-resume",
        turn_id="turn-one",
        expected_history_revision=started.history_revision,
        operation_id=operation_id,
        call_id="approval-call-1",
        arguments_digest=arguments_digest,
    )
    assert duplicate_start.status == "already_started"

    result = await repository.record_terminal_continuation_result(
        "pending-approval-resume",
        turn_id="turn-one",
        expected_history_revision=started.history_revision,
        operation_id=operation_id,
        call_id="approval-call-1",
        arguments_digest=arguments_digest,
        status=result_status,  # type: ignore[arg-type]
        result_digest=(None if result_status == "unknown" else "sha256:" + "a" * 64),
        receipt_reference=(None if result_status == "unknown" else "approval:approval-1"),
    )
    assert result.status == "operation_result"
    completed = await repository.load("pending-approval-resume")
    assert completed.history_revision is not None
    assert completed.timeline is not None
    assert completed.timeline.turns[0].operation_descriptors[0].status == result_status
    late_abandon = await repository.abandon_terminal_precommitted_operation(
        "pending-approval-resume",
        turn_id="turn-one",
        expected_history_revision=completed.history_revision,
        operation_id=operation_id,
        call_id="approval-call-1",
        arguments_digest=arguments_digest,
        result_digest="sha256:" + "b" * 64,
        receipt_reference="approval:approval-1",
    )
    assert late_abandon.status == "invalid"
    final = await repository.load("pending-approval-resume")
    assert final.timeline is not None
    assert final.timeline.turns[0].operation_descriptors[0].status == result_status
