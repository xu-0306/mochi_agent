from __future__ import annotations

import sqlite3
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import pytest

from mochi.agents.engine import AgentEngine
from mochi.config.schema import MochiConfig, SecurityConfig
from mochi.runtime.approval_lifecycle import PersistentApprovalStore
from mochi.runtime.approval_state_machine import ApprovalExpired
from mochi.runtime.approvals import ApprovalConflict
from mochi.runtime.exec_sessions import ExecSessionStatus, SessionPollResult
from mochi.runtime.sandbox.base import (
    HostSandboxBackend,
    SandboxResourceLimits,
    create_sandbox_plan,
)
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore
from mochi.security.policy import EffectivePolicyResolver
from mochi.sessions.timeline_coordinator import TimelineCoordinator
from mochi.sessions.turn_timeline import SessionTurnTimelineRepository
from mochi.tools.base import ToolExecutionContext
from mochi.tools.file_ops import FileWriteTool
from mochi.tools.exec_command import ExecCommandTool
from mochi.tools.registry import ToolRegistry


def _config(tmp_path: Path) -> MochiConfig:
    return MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path / "workspace"),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {
                "require_approval_for_exec": False,
                "require_approval_for_file_write": True,
            },
        }
    )


class _TimelineCallbackEngine:
    def __init__(self, engine: AgentEngine) -> None:
        self._engine = engine
        self.events: list[str] = []
        self.continuation_descriptor_statuses: list[str] = []

    async def begin_ordinary_chat_approval_operation(self, **kwargs: Any) -> None:
        await self._engine.begin_ordinary_chat_approval_operation(**kwargs)
        self.events.append("begin")

    async def record_ordinary_chat_approval_operation_result(self, **kwargs: Any) -> None:
        await self._engine.record_ordinary_chat_approval_operation_result(**kwargs)
        self.events.append(f"result:{kwargs['status']}")

    async def abandon_ordinary_chat_approval_operation(self, **kwargs: Any) -> None:
        await self._engine.abandon_ordinary_chat_approval_operation(**kwargs)
        self.events.append("abandon")

    async def resume_ordinary_chat_approval(
        self,
        *,
        approval_payload: Mapping[str, Any],
        **_: Any,
    ) -> dict[str, str]:
        checkpoint = approval_payload["ordinary_chat_checkpoint"]
        loaded = await SessionTurnTimelineRepository(self._engine._session_store).load(  # noqa: SLF001
            str(checkpoint["session_id"])
        )
        assert loaded.timeline is not None
        descriptor = next(
            item
            for turn in loaded.timeline.turns
            for item in turn.operation_descriptors
            if item.operation_id == checkpoint["operation_id"]
        )
        self.continuation_descriptor_statuses.append(descriptor.status)
        self.events.append("resume")
        return {"status": "continued"}


class _TerminalExecRuntime:
    def __init__(
        self,
        *,
        status: ExecSessionStatus | None = None,
        exit_code: int | None = None,
        dispatch_error: Exception | None = None,
    ) -> None:
        self.status = status
        self.exit_code = exit_code
        self.dispatch_error = dispatch_error
        self.calls = 0

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
    ) -> object:
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
        if self.dispatch_error is not None:
            raise self.dispatch_error
        assert self.status is not None
        return SessionPollResult(
            session_id="timeline-approved-exec",
            shell="powershell",
            status=self.status,
            background=False,
            tty=False,
            pid=None,
            exit_code=(
                self.exit_code
                if self.exit_code is not None
                else 3 if self.status == ExecSessionStatus.FAILED else None
            ),
            timed_out=self.status == ExecSessionStatus.TIMED_OUT,
            approval_state="approved",
            stdout="",
            stderr="terminal failure",
        )


async def _pending_timeline_file_approval(
    tmp_path: Path,
    *,
    session_id: str,
) -> tuple[
    AgentEngine,
    _TimelineCallbackEngine,
    RuntimeService,
    PersistentApprovalStore,
    str,
    str,
]:
    config = _config(tmp_path)
    workspace = Path(config.workspace_dir)
    workspace.mkdir(parents=True)
    core_engine = AgentEngine(config)
    callback_engine = _TimelineCallbackEngine(core_engine)
    security = SecurityConfig(require_approval_for_file_write=True)
    policy = EffectivePolicyResolver().resolve(security).to_dict()
    approvals = PersistentApprovalStore(tmp_path / f"{session_id}-approvals.db")
    runtime_store = RuntimeStore(tmp_path / f"{session_id}-runtime.db")
    await runtime_store.initialize()
    service = RuntimeService(
        engine=callback_engine,
        store=runtime_store,
        exec_approval_store=approvals,
    )
    service.update_security_config(security)

    coordinator = TimelineCoordinator(
        session_store=core_engine._session_store,  # noqa: SLF001
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
    context = ToolExecutionContext(
        workspace_dir=str(workspace),
        project_workspace=str(workspace),
        session_id=session_id,
        permission_policy=policy,
        state={
            "timeline_tool_lifecycle": coordinator,
            "timeline_tool_call_id": "file-call-1",
            "ordinary_chat_approval_context": {
                "schema_version": 1,
                "source": "ordinary_chat",
                "session_id": session_id,
                "turn_id": "turn-one",
                "resume_cursor": {
                    "turn_id": "turn-one",
                    "phase": "tool_call",
                    "tool_call_id": "file-call-1",
                    "tool_name": "file_write",
                },
                "react_continuation": {
                    "schema_version": 1,
                    "messages": [],
                    "callable_tool_names": ["file_write"],
                    "generation": {},
                },
            },
        },
    )
    registry = ToolRegistry(discover_builtin=False)
    registry.register(
        FileWriteTool(
            workspace_dir=workspace,
            require_approval=False,
            approval_store=approvals,
        )
    )
    pending = await registry.execute(
        "file_write",
        {"path": "report.txt", "content": "written once\n"},
        context=context,
    )
    assert pending.metadata["timeline_approval_pending"] is True
    approval_id = str(pending.metadata["approval_id"])
    operation_id = str(pending.metadata["timeline_operation_id"])
    await coordinator.persist_approval_pending(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "file_write", "metadata": dict(pending.metadata)},
    )
    await coordinator.finish()
    return core_engine, callback_engine, service, approvals, approval_id, operation_id


async def _pending_timeline_exec_approval(
    tmp_path: Path,
    *,
    session_id: str,
    runtime: _TerminalExecRuntime,
) -> tuple[
    AgentEngine,
    _TimelineCallbackEngine,
    RuntimeService,
    PersistentApprovalStore,
    str,
    str,
    dict[str, object],
]:
    config = _config(tmp_path)
    workspace = Path(config.workspace_dir)
    workspace.mkdir(parents=True)
    core_engine = AgentEngine(config)
    callback_engine = _TimelineCallbackEngine(core_engine)
    security = SecurityConfig(require_approval_for_exec=True)
    policy = EffectivePolicyResolver().resolve(security).to_dict()
    approvals = PersistentApprovalStore(tmp_path / f"{session_id}-approvals.db")
    runtime_store = RuntimeStore(tmp_path / f"{session_id}-runtime.db")
    await runtime_store.initialize()
    service = RuntimeService(
        engine=callback_engine,
        store=runtime_store,
        exec_approval_store=approvals,
        exec_runtime=runtime,  # type: ignore[arg-type]
    )
    service.update_security_config(security)
    coordinator = TimelineCoordinator(
        session_store=core_engine._session_store,  # noqa: SLF001
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
            "content": "run after approval",
        }
    )
    await coordinator.claim()
    context = ToolExecutionContext(
        workspace_dir=str(workspace),
        project_workspace=str(workspace),
        session_id=session_id,
        permission_policy=policy,
        state={
            "timeline_tool_lifecycle": coordinator,
            "timeline_tool_call_id": "exec-call-1",
            "ordinary_chat_approval_context": {
                "schema_version": 1,
                "source": "ordinary_chat",
                "session_id": session_id,
                "turn_id": "turn-one",
                "resume_cursor": {
                    "turn_id": "turn-one",
                    "phase": "tool_call",
                    "tool_call_id": "exec-call-1",
                    "tool_name": "exec_command",
                },
                "react_continuation": {
                    "schema_version": 1,
                    "messages": [],
                    "callable_tool_names": ["exec_command"],
                    "generation": {},
                },
            },
        },
    )
    registry = ToolRegistry(discover_builtin=False)
    registry.register(
        ExecCommandTool(
            runtime=runtime,  # type: ignore[arg-type]
            approval_store=approvals,
            workspace_dir=workspace,
            require_approval=False,
        )
    )
    arguments: dict[str, object] = {"command": "echo terminal", "shell": "powershell"}
    pending = await registry.execute("exec_command", arguments, context=context)
    assert pending.metadata["timeline_approval_pending"] is True
    approval_id = str(pending.metadata["approval_id"])
    operation_id = str(pending.metadata["timeline_operation_id"])
    await coordinator.persist_approval_pending(
        operation_id=operation_id,
        event_id="turn-one:1",
        sequence=1,
        payload={"tool_name": "exec_command", "metadata": dict(pending.metadata)},
    )
    await coordinator.finish()
    return core_engine, callback_engine, service, approvals, approval_id, operation_id, policy


@pytest.mark.asyncio
async def test_runtime_service_records_timeline_result_before_react_continuation(
    tmp_path: Path,
) -> None:
    core, callback, service, approvals, approval_id, operation_id = (
        await _pending_timeline_file_approval(tmp_path, session_id="service-order")
    )
    policy = EffectivePolicyResolver().resolve(
        SecurityConfig(require_approval_for_file_write=True)
    ).to_dict()

    resolved = await service.resolve_approval(
        approval_id,
        decision="approve_once",
        current_permission_policy=policy,
    )
    assert resolved is not None
    assert resolved["status"] == "consumed"
    assert callback.events == ["begin", "result:succeeded", "resume"]
    assert callback.continuation_descriptor_statuses == ["succeeded"]
    assert (Path(core._config.workspace_dir) / "report.txt").read_text(encoding="utf-8") == "written once\n"  # noqa: SLF001
    approval = approvals.get(approval_id)
    assert approval is not None
    assert approval.execution_result is not None
    assert approval.execution_result["react_continuation"] == {"status": "continued"}
    with pytest.raises(ApprovalConflict):
        await service.resolve_approval(
            approval_id,
            decision="approve_once",
            current_permission_policy=policy,
        )
    timeline = await SessionTurnTimelineRepository(core._session_store).load("service-order")  # noqa: SLF001
    assert timeline.timeline is not None
    assert timeline.timeline.turns[0].operation_descriptors[0].operation_id == operation_id
    assert timeline.timeline.turns[0].operation_descriptors[0].status == "succeeded"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_result_status"),
    [
        (ExecSessionStatus.FAILED, "failed"),
        (ExecSessionStatus.TIMED_OUT, "timed_out"),
    ],
)
async def test_runtime_service_records_confirmed_exec_terminal_outcomes_as_failed(
    tmp_path: Path,
    status: ExecSessionStatus,
    expected_result_status: str,
) -> None:
    runtime = _TerminalExecRuntime(status=status)
    core, callback, service, approvals, approval_id, operation_id, policy = (
        await _pending_timeline_exec_approval(
            tmp_path,
            session_id=f"service-exec-{status.value}",
            runtime=runtime,
        )
    )
    resolved = await service.resolve_approval(
        approval_id,
        decision="approve_once",
        current_permission_policy=policy,
    )
    assert resolved is not None
    assert resolved["status"] == "execution_failed"
    assert runtime.calls == 1
    assert callback.events == ["begin", "result:failed"]
    assert callback.continuation_descriptor_statuses == []
    approval = approvals.get(approval_id)
    assert approval is not None
    assert approval.execution_result is not None
    assert approval.execution_result["status"] == expected_result_status
    timeline = await SessionTurnTimelineRepository(core._session_store).load(  # noqa: SLF001
        f"service-exec-{status.value}"
    )
    assert timeline.timeline is not None
    descriptor = timeline.timeline.turns[0].operation_descriptors[0]
    assert descriptor.operation_id == operation_id
    assert descriptor.status == "failed"


@pytest.mark.asyncio
async def test_runtime_service_records_completed_nonzero_exec_as_failed(
    tmp_path: Path,
) -> None:
    runtime = _TerminalExecRuntime(status=ExecSessionStatus.COMPLETED, exit_code=3)
    core, callback, service, approvals, approval_id, operation_id, policy = (
        await _pending_timeline_exec_approval(
            tmp_path,
            session_id="service-exec-completed-nonzero",
            runtime=runtime,
        )
    )
    resolved = await service.resolve_approval(
        approval_id,
        decision="approve_once",
        current_permission_policy=policy,
    )
    assert resolved is not None
    assert resolved["status"] == "execution_failed"
    assert runtime.calls == 1
    assert callback.events == ["begin", "result:failed"]
    assert callback.continuation_descriptor_statuses == []
    approval = approvals.get(approval_id)
    assert approval is not None
    assert approval.execution_result is not None
    assert approval.execution_result["status"] == "completed"
    assert approval.execution_result["exit_code"] == 3
    timeline = await SessionTurnTimelineRepository(core._session_store).load(  # noqa: SLF001
        "service-exec-completed-nonzero"
    )
    assert timeline.timeline is not None
    descriptor = timeline.timeline.turns[0].operation_descriptors[0]
    assert descriptor.operation_id == operation_id
    assert descriptor.status == "failed"


@pytest.mark.asyncio
async def test_runtime_service_marks_exec_dispatch_exception_as_unknown(
    tmp_path: Path,
) -> None:
    runtime = _TerminalExecRuntime(dispatch_error=RuntimeError("dispatch link dropped"))
    core, callback, service, approvals, approval_id, operation_id, policy = (
        await _pending_timeline_exec_approval(
            tmp_path,
            session_id="service-exec-dispatch-unknown",
            runtime=runtime,
        )
    )
    resolved = await service.resolve_approval(
        approval_id,
        decision="approve_once",
        current_permission_policy=policy,
    )
    assert resolved is not None
    assert resolved["status"] == "execution_failed"
    assert runtime.calls == 1
    assert callback.events == ["begin", "result:unknown"]
    approval = approvals.get(approval_id)
    assert approval is not None
    assert approval.execution_result is not None
    assert approval.execution_result["error_code"] == "exec_dispatch_exception"
    timeline = await SessionTurnTimelineRepository(core._session_store).load(  # noqa: SLF001
        "service-exec-dispatch-unknown"
    )
    assert timeline.timeline is not None
    descriptor = timeline.timeline.turns[0].operation_descriptors[0]
    assert descriptor.operation_id == operation_id
    assert descriptor.status == "unknown"


@pytest.mark.asyncio
async def test_runtime_service_records_known_file_tool_error_as_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core, callback, service, approvals, approval_id, operation_id = (
        await _pending_timeline_file_approval(tmp_path, session_id="service-file-known-failure")
    )

    async def known_file_error(*_: object, **__: object) -> dict[str, object]:
        return {
            "status": "execution_failed",
            "tool_name": "file_write",
            "error": "writer rejected the approved content",
            "metadata": {"timeline_result_disposition": "failed"},
        }

    monkeypatch.setattr(service, "_execute_approved_standalone_request", known_file_error)
    policy = EffectivePolicyResolver().resolve(
        SecurityConfig(require_approval_for_file_write=True)
    ).to_dict()
    resolved = await service.resolve_approval(
        approval_id,
        decision="approve_once",
        current_permission_policy=policy,
    )
    assert resolved is not None
    assert resolved["status"] == "execution_failed"
    assert callback.events == ["begin", "result:failed"]
    approval = approvals.get(approval_id)
    assert approval is not None
    assert approval.execution_result is not None
    assert approval.execution_result["error"] == "writer rejected the approved content"
    timeline = await SessionTurnTimelineRepository(core._session_store).load(  # noqa: SLF001
        "service-file-known-failure"
    )
    assert timeline.timeline is not None
    descriptor = timeline.timeline.turns[0].operation_descriptors[0]
    assert descriptor.operation_id == operation_id
    assert descriptor.status == "failed"


@pytest.mark.asyncio
async def test_runtime_service_rejection_abandons_timeline_operation_for_replanning(
    tmp_path: Path,
) -> None:
    core, callback, service, _, approval_id, operation_id = await _pending_timeline_file_approval(
        tmp_path,
        session_id="service-reject",
    )
    resolved = await service.resolve_approval(approval_id, decision="reject", reason="not now")
    assert resolved is not None
    assert resolved["status"] == "rejected"
    assert callback.events == ["abandon"]
    timeline = await SessionTurnTimelineRepository(core._session_store).load("service-reject")  # noqa: SLF001
    assert timeline.timeline is not None
    descriptor = timeline.timeline.turns[0].operation_descriptors[0]
    assert descriptor.operation_id == operation_id
    assert descriptor.status == "abandoned"

    follower = TimelineCoordinator(
        session_store=core._session_store,  # noqa: SLF001
        session_id="service-reject",
        turn_id="turn-two",
    )
    await follower.admit_user_message(
        {
            "type": "message",
            "schema_version": 1,
            "session_id": "service-reject",
            "turn_id": "turn-two",
            "role": "user",
            "content": "replan",
        }
    )
    await follower.claim()
    next_operation, _ = await follower.precommit_mutation(
        tool_name="file_write",
        arguments={"path": "replacement.txt", "content": "new plan"},
        call_id="replacement-call",
    )
    await follower.abandon_pre_effect_operation(
        operation_id=next_operation,
        event_id="turn-two:1",
        sequence=1,
        payload={"tool_name": "file_write", "error": "replanned"},
    )
    await follower.finish(failed=True)


@pytest.mark.asyncio
async def test_stale_timeline_approval_lease_is_quarantined_as_unknown(
    tmp_path: Path,
) -> None:
    core, callback, service, approvals, approval_id, operation_id = (
        await _pending_timeline_file_approval(tmp_path, session_id="service-stale")
    )
    approval = approvals.get(approval_id)
    assert approval is not None
    approvals.resolve(
        approval_id,
        decision="approve_once",
        requester_id=approval.requester_id,
        request_digest=approval.request_digest,
        context_digest=approval.context_digest,
    )
    claim = approvals.consume(
        approval_id,
        execution_idempotency_key=f"approval-execution:{approval_id}:{operation_id}",
        lease_owner="crashed-worker",
        requester_id=approval.requester_id,
        request_digest=approval.request_digest,
        context_digest=approval.context_digest,
    )
    assert claim.status == "consuming"
    with sqlite3.connect(tmp_path / "service-stale-approvals.db") as conn:
        conn.execute(
            "UPDATE exec_approval_requests SET consume_lease_expires_at=? WHERE approval_id=?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), approval_id),
        )
        conn.commit()

    await service._recover_stale_approval_consumptions()  # noqa: SLF001
    assert callback.events == ["begin", "result:unknown"]
    recovered = approvals.get(approval_id)
    assert recovered is not None
    assert recovered.status == "execution_failed"
    timeline = await SessionTurnTimelineRepository(core._session_store).load("service-stale")  # noqa: SLF001
    assert timeline.timeline is not None
    descriptor = timeline.timeline.turns[0].operation_descriptors[0]
    assert descriptor.operation_id == operation_id
    assert descriptor.status == "unknown"


@pytest.mark.asyncio
async def test_runtime_service_drift_abandons_before_timeline_start(
    tmp_path: Path,
) -> None:
    core, callback, service, approvals, approval_id, operation_id = (
        await _pending_timeline_file_approval(tmp_path, session_id="service-drift")
    )
    # The exact base state changed while the approval waited. This is known
    # no-effect drift, so it must abandon instead of crossing the boundary.
    (Path(core._config.workspace_dir) / "report.txt").write_text("outside change\n", encoding="utf-8")  # noqa: SLF001
    policy = EffectivePolicyResolver().resolve(
        SecurityConfig(require_approval_for_file_write=True)
    ).to_dict()
    resolved = await service.resolve_approval(
        approval_id,
        decision="approve_once",
        current_permission_policy=policy,
    )
    assert resolved is not None
    assert resolved["status"] == "superseded"
    assert resolved["checkpoint_error_code"] == "file_base_drift"
    assert callback.events == ["abandon"]
    approval = approvals.get(approval_id)
    assert approval is not None and approval.status == "superseded"
    timeline = await SessionTurnTimelineRepository(core._session_store).load("service-drift")  # noqa: SLF001
    assert timeline.timeline is not None
    descriptor = timeline.timeline.turns[0].operation_descriptors[0]
    assert descriptor.operation_id == operation_id
    assert descriptor.status == "abandoned"
    assert (Path(core._config.workspace_dir) / "report.txt").read_text(encoding="utf-8") == "outside change\n"  # noqa: SLF001


@pytest.mark.asyncio
async def test_runtime_service_expiry_abandons_before_timeline_start(
    tmp_path: Path,
) -> None:
    core, callback, service, approvals, approval_id, operation_id = (
        await _pending_timeline_file_approval(tmp_path, session_id="service-expiry")
    )
    with sqlite3.connect(tmp_path / "service-expiry-approvals.db") as conn:
        conn.execute(
            "UPDATE exec_approval_requests SET expires_at=? WHERE approval_id=?",
            ((datetime.now(UTC) - timedelta(seconds=1)).isoformat(), approval_id),
        )
        conn.commit()
    policy = EffectivePolicyResolver().resolve(
        SecurityConfig(require_approval_for_file_write=True)
    ).to_dict()
    with pytest.raises(ApprovalExpired):
        await service.resolve_approval(
            approval_id,
            decision="approve_once",
            current_permission_policy=policy,
        )
    assert callback.events == ["abandon"]
    approval = approvals.get(approval_id)
    assert approval is not None and approval.status == "expired"
    timeline = await SessionTurnTimelineRepository(core._session_store).load("service-expiry")  # noqa: SLF001
    assert timeline.timeline is not None
    descriptor = timeline.timeline.turns[0].operation_descriptors[0]
    assert descriptor.operation_id == operation_id
    assert descriptor.status == "abandoned"
