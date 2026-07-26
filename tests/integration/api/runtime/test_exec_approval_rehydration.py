"""Runtime API tests grouped by ownership."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.runtime.approval_state_machine import derive_approval_binding
from mochi.runtime.approvals import (
    APPROVAL_OWNER_TASK_ID_KEY,
    ApprovalConflict,
    ApprovalRequesterMismatch,
    InMemoryApprovalStore,
    PersistentApprovalStore,
)
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.service import RuntimeService, is_successful_exec_approval_result
from mochi.runtime.store import RuntimeStore
from mochi.utils.shell_providers import SubprocessSpec
from tests.support.exec_providers import PythonDirectProvider as _ApiRuntimePythonDirectProvider

from ._support import (
    _build_linked_agent_run_exec_approval_orchestrator,
    _create_agent_run_exec_test_client,
    _RuntimeExecLinkedFakeEngine,
    _RuntimeFakeEngine,
    _wait_agent_run_until,
    _wait_until,
)


def test_linked_agent_run_exec_approval_remains_listed_after_runtime_restart(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-agent-run-restart-list-1"
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(
        "mochi.runtime.service.MultiAgentOrchestrator.run",
        _build_linked_agent_run_exec_approval_orchestrator(
            approval_id=approval_id,
            workdir=tmp_path,
            final_answer="Approval restart listing completed.",
        ),
    )

    with _create_agent_run_exec_test_client(
        sessions_dir=sessions_dir,
        exec_approval_store=InMemoryApprovalStore(),
    ) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Approval restart listing run",
                "topic": "list linked approval after restart",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        waiting = _wait_agent_run_until(client, run_id, {"awaiting_approval"}, timeout_seconds=4.0)
        assert waiting["summary"]["approval_state"]["approval_ids"] == [approval_id]

    with _create_agent_run_exec_test_client(
        sessions_dir=sessions_dir,
        exec_approval_store=InMemoryApprovalStore(),
    ) as restarted_client:
        approvals_response = restarted_client.get("/v1/approvals?status=pending")
        assert approvals_response.status_code == 200
        approvals = approvals_response.json()
        linked = [item for item in approvals if item["approval_id"] == approval_id]
        assert len(linked) == 1
        assert linked[0]["status"] == "pending"
        assert linked[0]["tool_name"] == "exec_command"

def test_approval_resolve_endpoint_after_restart_auto_resumes_linked_agent_run(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-agent-run-restart-resume-1"
    resumed_payloads: list[dict[str, Any]] = []
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(
        "mochi.runtime.service.MultiAgentOrchestrator.run",
        _build_linked_agent_run_exec_approval_orchestrator(
            approval_id=approval_id,
            workdir=tmp_path,
            final_answer="Approval restart auto-resume completed.",
            resumed_payloads=resumed_payloads,
        ),
    )

    with _create_agent_run_exec_test_client(
        sessions_dir=sessions_dir,
        exec_approval_store=InMemoryApprovalStore(),
    ) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Approval restart auto-resume run",
                "topic": "resolve linked approval after restart",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        waiting = _wait_agent_run_until(client, run_id, {"awaiting_approval"}, timeout_seconds=4.0)
        assert waiting["summary"]["approval_state"]["approval_ids"] == [approval_id]

    restarted_exec_approval_store = InMemoryApprovalStore()
    with _create_agent_run_exec_test_client(
        sessions_dir=sessions_dir,
        exec_approval_store=restarted_exec_approval_store,
    ) as restarted_client:
        resolve_response = restarted_client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allow linked restart auto resume"},
        )
        assert resolve_response.status_code == 200
        assert resolve_response.json()["status"] == "consumed"

        completed = _wait_agent_run_until(
            restarted_client,
            run_id,
            {"succeeded"},
            timeout_seconds=4.0,
        )
        assert completed["summary"]["final_answer"] == "Approval restart auto-resume completed."

    assert resumed_payloads
    resolved = restarted_exec_approval_store.get(approval_id)
    assert resolved is not None
    assert resolved.status == "consumed"
    assert resolved.execution_result is not None

def test_approvals_api_persists_standalone_exec_approvals_across_runtime_restart(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    exec_approval_id = "exec-approval-standalone-restart-1"

    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )

    initial_exec_store = PersistentApprovalStore(sessions_dir / "exec-approvals.db")
    initial_exec_store.create(
        approval_id=exec_approval_id,
        command="print('approved exec')",
        shell="test",
        scope="dangerous_command",
        reason="requires review",
        command_payload={
            "command": "print('approved exec')",
            "shell": "test",
            "workdir": str(tmp_path),
            "env": None,
            "timeout_sec": 5.0,
            "background": False,
            "tty": False,
            "approval_state": "approved",
        },
    )
    initial_runtime_service = RuntimeService(
        engine=_RuntimeFakeEngine(),
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=initial_exec_store,
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    app.state.runtime_service = initial_runtime_service

    with TestClient(app) as client:
        pending_response = client.get("/v1/approvals?status=pending")
        assert pending_response.status_code == 200
        pending_items = pending_response.json()
        assert any(item["approval_id"] == exec_approval_id for item in pending_items)

    restarted_app = create_app()
    restarted_app.state.engine_factory = lambda: _RuntimeFakeEngine()
    restarted_app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )
    restarted_runtime_service = RuntimeService(
        engine=_RuntimeFakeEngine(),
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=PersistentApprovalStore(sessions_dir / "exec-approvals.db"),
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    restarted_app.state.runtime_service = restarted_runtime_service

    with TestClient(restarted_app) as restarted_client:
        restarted_pending = restarted_client.get("/v1/approvals?status=pending")
        assert restarted_pending.status_code == 200
        assert any(
            item["approval_id"] == exec_approval_id
            for item in restarted_pending.json()
        )

        resolve_response = restarted_client.post(
            f"/v1/approvals/{exec_approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allowed"},
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        assert resolved["approval_id"] == exec_approval_id
        assert resolved["status"] == "consumed"
        assert resolved["execution_result"]["status"] == "completed"
        assert resolved["exec_session_id"] is not None

    final_app = create_app()
    final_app.state.engine_factory = lambda: _RuntimeFakeEngine()
    final_app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )
    final_runtime_service = RuntimeService(
        engine=_RuntimeFakeEngine(),
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=PersistentApprovalStore(sessions_dir / "exec-approvals.db"),
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    final_app.state.runtime_service = final_runtime_service

    with TestClient(final_app) as final_client:
        approved_response = final_client.get("/v1/approvals?status=consumed")
        assert approved_response.status_code == 200
        approved_items = approved_response.json()
        approved = next(item for item in approved_items if item["approval_id"] == exec_approval_id)
        assert approved["exec_status"] == "consumed"
        assert approved["exec_session_id"] is not None
        assert approved["execution_result"]["status"] == "completed"
        assert "approved exec" in approved["execution_result"]["stdout"]

def test_task_exec_approval_remains_resolvable_after_runtime_restart(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    approval_id: str
    task_id: str

    app = create_app()
    initial_exec_approval_store = InMemoryApprovalStore()
    initial_engine = _RuntimeExecLinkedFakeEngine(exec_approval_store=initial_exec_approval_store)
    initial_runtime_service = RuntimeService(
        engine=initial_engine,
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=initial_exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    initial_runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = initial_runtime_service
    app.state.engine_factory = lambda: initial_engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={"input_message": "run exec command", "workspace_dir": str(tmp_path / "workspace")},
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        approval_id = waiting["pending_approval"]["id"]

    restarted_app = create_app()
    restarted_exec_approval_store = InMemoryApprovalStore()
    restarted_runtime_service = RuntimeService(
        engine=_RuntimeExecLinkedFakeEngine(exec_approval_store=restarted_exec_approval_store),
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=restarted_exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    restarted_runtime_service.set_scheduler_poll_interval(0.05)
    restarted_app.state.runtime_service = restarted_runtime_service
    restarted_app.state.engine_factory = lambda: _RuntimeExecLinkedFakeEngine(
        exec_approval_store=restarted_exec_approval_store
    )
    restarted_app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )

    with TestClient(restarted_app) as restarted_client:
        approvals_response = restarted_client.get("/v1/approvals?status=pending")
        assert approvals_response.status_code == 200
        restarted_approval = next(
            item for item in approvals_response.json() if item["approval_id"] == approval_id
        )
        assert restarted_approval["source"] == "task_runtime"
        assert restarted_approval["exec_approval_id"] == "exec-approval-linked-runtime-1"
        assert restarted_approval["exec_status"] == "pending"

        resolve_response = restarted_client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allow linked exec after restart"},
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        assert resolved["status"] == "consumed"
        assert resolved["execution_result"]["status"] == "completed"
        assert "linked exec approved" in resolved["execution_result"]["stdout"]

        done_payload = _wait_until(restarted_client, task_id, {"succeeded"})
        assert done_payload["status"] == "succeeded"
        assert "linked exec approved" in (done_payload["final_answer"] or "")

    rehydrated = restarted_exec_approval_store.get("exec-approval-linked-runtime-1")
    assert rehydrated is not None
    assert rehydrated.status == "consumed"
    assert rehydrated.execution_result is not None
    assert rehydrated.execution_result["status"] == "completed"

def test_task_exec_approval_result_stays_visible_after_runtime_restart(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"

    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    engine = _RuntimeExecLinkedFakeEngine(exec_approval_store=exec_approval_store)
    runtime_service = RuntimeService(
        engine=engine,
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={"input_message": "run exec command", "workspace_dir": str(tmp_path / "workspace")},
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        approval_id = waiting["pending_approval"]["id"]

        resolve_response = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allow linked exec"},
        )
        assert resolve_response.status_code == 200

        done_payload = _wait_until(client, task_id, {"succeeded"})
        assert done_payload["status"] == "succeeded"

    restarted_app = create_app()
    restarted_runtime_service = RuntimeService(
        engine=_RuntimeExecLinkedFakeEngine(exec_approval_store=InMemoryApprovalStore()),
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=InMemoryApprovalStore(),
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    restarted_app.state.runtime_service = restarted_runtime_service
    restarted_app.state.engine_factory = lambda: _RuntimeExecLinkedFakeEngine(
        exec_approval_store=InMemoryApprovalStore()
    )
    restarted_app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )

    with TestClient(restarted_app) as restarted_client:
        approved_response = restarted_client.get("/v1/approvals?status=consumed")
        assert approved_response.status_code == 200
        approved = next(item for item in approved_response.json() if item["approval_id"] == approval_id)
        assert approved["source"] == "task_runtime"
        assert approved["exec_approval_id"] == "exec-approval-linked-runtime-1"
        assert approved["exec_status"] == "consumed"
        assert approved["exec_session_id"] is not None
        assert approved["execution_result"]["status"] == "completed"
        assert "linked exec approved" in approved["execution_result"]["stdout"]

@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"status": "completed"}, True),
        ({"status": "completed", "exit_code": 0}, True),
        ({"status": "completed", "exit_code": 3}, False),
        ({"status": "completed", "timed_out": True}, False),
        ({"status": "running", "background": True, "session_id": "exec-1"}, True),
        ({"status": "running", "background": True, "session_id": " "}, False),
        ({"status": "running", "background": False, "session_id": "exec-1"}, False),
        ({"status": "failed"}, False),
        ({"status": "timed_out"}, False),
        ({"status": "killed"}, False),
        ({"status": "cancelled"}, False),
        ({"status": "execution_failed"}, False),
    ],
)
def test_exec_approval_success_classification_is_explicit(
    result: dict[str, object],
    expected: bool,
) -> None:
    assert is_successful_exec_approval_result(result) is expected

def test_runtime_service_marks_failed_standalone_exec_approval_execution_failed(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        approvals = InMemoryApprovalStore()
        approvals.create(
            approval_id="exec-approval-failed-standalone",
            command="raise SystemExit(1)",
            shell="test",
            scope="dangerous_command",
            command_payload={
                "command": "raise SystemExit(1)",
                "shell": "test",
                "workdir": str(tmp_path),
                "env": None,
                "timeout_sec": 5.0,
                "background": False,
                "tty": False,
                "approval_state": "approved",
            },
        )
        service = RuntimeService(
            engine=_RuntimeFakeEngine(),
            store=RuntimeStore(tmp_path / "runtime.db"),
            exec_approval_store=approvals,
            exec_runtime=ExecRuntime(
                providers={"test": _ApiRuntimePythonDirectProvider()},
                default_shell="test",
            ),
        )
        resolved = await service.resolve_approval(
            "exec-approval-failed-standalone",
            decision="approve_once",
        )
        assert resolved is not None
        assert resolved["status"] == "execution_failed"
        assert resolved["execution_result"]["status"] == "failed"

    asyncio.run(scenario())

def test_runtime_service_marks_failed_task_linked_exec_execution_failed(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        runtime_store = RuntimeStore(tmp_path / "runtime.db")
        await runtime_store.initialize()
        await runtime_store.create_task_run(
            task_id="failed-linked-task",
            input_text="run failing command",
            session_id=None,
            project_id=None,
            workspace_dir=str(tmp_path),
            project_workspace_dir=None,
            task_workspace_dir=None,
            inference_overrides={},
        )
        approvals = InMemoryApprovalStore()
        approvals.create(
            approval_id="exec-approval-failed-linked",
            command="raise SystemExit(1)",
            shell="test",
            scope="dangerous_command",
            command_payload={
                "command": "raise SystemExit(1)",
                "shell": "test",
                "workdir": str(tmp_path),
                "env": None,
                "timeout_sec": 5.0,
                "background": False,
                "tty": False,
                "approval_state": "approved",
            },
        )
        await runtime_store.create_approval_request(
            approval_id="task-approval-failed-linked",
            task_id="failed-linked-task",
            call_id="call-failed-linked",
            tool_name="exec_command",
            arguments={"command": "raise SystemExit(1)", "shell": "test"},
            metadata={"approval_id": "exec-approval-failed-linked"},
        )
        service = RuntimeService(
            engine=_RuntimeFakeEngine(),
            store=runtime_store,
            exec_approval_store=approvals,
            exec_runtime=ExecRuntime(
                providers={"test": _ApiRuntimePythonDirectProvider()},
                default_shell="test",
            ),
        )
        resolved = await service.resolve_approval(
            "task-approval-failed-linked",
            decision="approve_once",
        )
        assert resolved is not None
        assert resolved["status"] == "execution_failed"
        linked = approvals.get("exec-approval-failed-linked")
        assert linked is not None
        assert linked.status == "execution_failed"
        assert linked.execution_result is not None
        assert linked.execution_result["status"] == "failed"
        task = await runtime_store.get_task_run("failed-linked-task")
        assert task is not None
        assert task["status"] == "failed"
        assert task["final_answer"] is None

    asyncio.run(scenario())


def test_runtime_service_marks_completed_nonzero_task_linked_exec_execution_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def scenario() -> None:
        runtime_store = RuntimeStore(tmp_path / "runtime.db")
        await runtime_store.initialize()
        await runtime_store.create_task_run(
            task_id="completed-nonzero-linked-task",
            input_text="run a command that exits nonzero",
            session_id=None,
            project_id=None,
            workspace_dir=str(tmp_path),
            project_workspace_dir=None,
            task_workspace_dir=None,
            inference_overrides={},
        )
        approvals = InMemoryApprovalStore()
        approvals.create(
            approval_id="exec-approval-completed-nonzero-linked",
            command="exit 3",
            shell="test",
            scope="dangerous_command",
            command_payload={
                "command": "exit 3",
                "shell": "test",
                "workdir": str(tmp_path),
                "env": None,
                "timeout_sec": 5.0,
                "background": False,
                "tty": False,
                "approval_state": "approved",
            },
        )
        await runtime_store.create_approval_request(
            approval_id="task-approval-completed-nonzero-linked",
            task_id="completed-nonzero-linked-task",
            call_id="call-completed-nonzero-linked",
            tool_name="exec_command",
            arguments={"command": "exit 3", "shell": "test"},
            metadata={"approval_id": "exec-approval-completed-nonzero-linked"},
        )
        service = RuntimeService(
            engine=_RuntimeFakeEngine(),
            store=runtime_store,
            exec_approval_store=approvals,
            exec_runtime=ExecRuntime(
                providers={"test": _ApiRuntimePythonDirectProvider()},
                default_shell="test",
            ),
        )

        async def completed_nonzero_exec(*_: object, **__: object) -> dict[str, object]:
            return {
                "status": "completed",
                "exit_code": 3,
                "timed_out": False,
                "background": False,
                "stderr": "command exited with code 3",
            }

        monkeypatch.setattr(service, "_execute_approved_exec_request", completed_nonzero_exec)
        resolved = await service.resolve_approval(
            "task-approval-completed-nonzero-linked",
            decision="approve_once",
        )
        assert resolved is not None
        assert resolved["status"] == "execution_failed"
        linked = approvals.get("exec-approval-completed-nonzero-linked")
        assert linked is not None
        assert linked.status == "execution_failed"
        assert linked.execution_result is not None
        assert linked.execution_result["status"] == "completed"
        assert linked.execution_result["exit_code"] == 3
        task = await runtime_store.get_task_run("completed-nonzero-linked-task")
        assert task is not None
        assert task["status"] == "failed"
        assert task["final_answer"] == "command exited with code 3"

    asyncio.run(scenario())


def test_runtime_service_rejects_tampered_standalone_exec_binding_before_execution(
    tmp_path: Path,
) -> None:
    class _RecordingProvider(_ApiRuntimePythonDirectProvider):
        def __init__(self) -> None:
            self.commands: list[str] = []

        def build_subprocess_spec(self, command: str, *, tty: bool = False) -> SubprocessSpec:
            self.commands.append(command)
            return super().build_subprocess_spec(command, tty=tty)

    async def scenario() -> None:
        approvals = InMemoryApprovalStore()
        command_payload = {
            "command": "print('must not run')",
            "shell": "test",
            "workdir": str(tmp_path),
            "env": None,
            "timeout_sec": 5.0,
            "background": False,
            "tty": False,
            "approval_state": "approved",
        }
        approvals.create(
            approval_id="tampered-standalone-binding",
            command="print('must not run')",
            shell="test",
            scope="dangerous_command",
            command_payload=command_payload,
            requester_id="runtime-service",
            request_digest="0" * 64,
            context_digest="1" * 64,
        )
        provider = _RecordingProvider()
        service = RuntimeService(
            engine=_RuntimeFakeEngine(),
            store=RuntimeStore(tmp_path / "runtime.db"),
            exec_approval_store=approvals,
            exec_runtime=ExecRuntime(providers={"test": provider}, default_shell="test"),
        )

        with pytest.raises(ApprovalConflict):
            await service.resolve_approval(
                "tampered-standalone-binding",
                decision="approve_once",
            )

        stored = approvals.get("tampered-standalone-binding")
        assert stored is not None
        assert stored.status == "pending"
        assert provider.commands == []

    asyncio.run(scenario())

def test_runtime_task_approval_binding_normal_flow_and_sqlite_tamper_conflict(
    tmp_path: Path,
) -> None:
    normal_sessions_dir = tmp_path / "normal-sessions"
    normal_app = create_app()
    normal_engine = _RuntimeFakeEngine()
    normal_app.state.engine_factory = lambda: normal_engine
    normal_app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(normal_sessions_dir)}
    )

    with TestClient(normal_app) as client:
        created = client.post(
            "/v1/tasks",
            json={"input_message": "run bound approval", "workspace_dir": str(tmp_path / "workspace")},
        )
        assert created.status_code == 200
        task_id = created.json()["task_id"]
        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        approval_id = waiting["pending_approval"]["id"]
        stored = asyncio.run(
            RuntimeStore(normal_sessions_dir / "runtime.db").get_approval_request(approval_id)
        )
        assert stored is not None
        assert stored["requester_id"] == f"runtime-task:{task_id}"
        assert len(stored["request_digest"]) == 64
        assert len(stored["context_digest"]) == 64

        resolved = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "normal bound approval"},
        )
        assert resolved.status_code == 200
        assert _wait_until(client, task_id, {"succeeded"})["status"] == "succeeded"

    tampered_sessions_dir = tmp_path / "tampered-sessions"
    tampered_app = create_app()
    tampered_engine = _RuntimeFakeEngine()
    tampered_app.state.engine_factory = lambda: tampered_engine
    tampered_app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tampered_sessions_dir)}
    )

    with TestClient(tampered_app) as client:
        created = client.post(
            "/v1/tasks",
            json={"input_message": "tamper bound approval", "workspace_dir": str(tmp_path / "workspace")},
        )
        assert created.status_code == 200
        task_id = created.json()["task_id"]
        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        approval_id = waiting["pending_approval"]["id"]

        # Deliberately mutate persisted binding columns to simulate database tampering.
        with sqlite3.connect(tampered_sessions_dir / "runtime.db") as conn:
            conn.execute(
                """
                UPDATE approval_requests
                SET request_digest=?, context_digest=?
                WHERE id=?
                """,
                ("0" * 64, "1" * 64, approval_id),
            )

        conflict = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "must not resolve"},
        )
        assert conflict.status_code == 409
        pending = client.get("/v1/approvals?status=pending")
        assert pending.status_code == 200
        assert any(item["approval_id"] == approval_id for item in pending.json())
        assert len(tampered_engine.permission_policy_calls) == 1

def test_runtime_service_marks_foreground_running_task_linked_exec_failed(
    tmp_path: Path,
) -> None:
    class _ForegroundRunningService(RuntimeService):
        async def _execute_approved_exec_request(self, approval: Any) -> dict[str, Any]:
            del approval
            return {
                "status": "running",
                "background": False,
                "session_id": "foreground-session",
                "stdout": "",
                "stderr": "",
                "timed_out": False,
            }

    async def scenario() -> None:
        runtime_store = RuntimeStore(tmp_path / "runtime.db")
        await runtime_store.initialize()
        await runtime_store.create_task_run(
            task_id="foreground-running-linked-task",
            input_text="run foreground command",
            session_id=None,
            project_id=None,
            workspace_dir=str(tmp_path),
            project_workspace_dir=None,
            task_workspace_dir=None,
            inference_overrides={},
        )
        approvals = InMemoryApprovalStore()
        approvals.create(
            approval_id="exec-approval-foreground-running",
            command="print('foreground')",
            shell="test",
            scope="dangerous_command",
            command_payload={
                "command": "print('foreground')",
                "shell": "test",
                "workdir": str(tmp_path),
                "env": None,
                "timeout_sec": 5.0,
                "background": False,
                "tty": False,
                "approval_state": "approved",
            },
        )
        await runtime_store.create_approval_request(
            approval_id="task-approval-foreground-running",
            task_id="foreground-running-linked-task",
            call_id="call-foreground-running",
            tool_name="exec_command",
            arguments={"command": "print('foreground')", "shell": "test"},
            metadata={"approval_id": "exec-approval-foreground-running"},
        )
        service = _ForegroundRunningService(
            engine=_RuntimeFakeEngine(),
            store=runtime_store,
            exec_approval_store=approvals,
            exec_runtime=ExecRuntime(
                providers={"test": _ApiRuntimePythonDirectProvider()},
                default_shell="test",
            ),
        )

        resolved = await service.resolve_approval(
            "task-approval-foreground-running",
            decision="approve_once",
        )
        assert resolved is not None
        assert resolved["status"] == "execution_failed"

        linked = approvals.get("exec-approval-foreground-running")
        assert linked is not None
        assert linked.status == "execution_failed"
        task = await runtime_store.get_task_run("foreground-running-linked-task")
        assert task is not None
        assert task["status"] == "failed"
        assert task["final_answer"] is None
        events = await runtime_store.get_task_events("foreground-running-linked-task")
        result_event = next(event for event in events if event["type"] == "tool_call_result")
        assert result_event["error"] is not None
        assert result_event["metadata"]["status"] == "running"

    asyncio.run(scenario())

@pytest.mark.parametrize(
    ("tampered_requester_id", "tampered_request_digest", "expected_error"),
    [
        pytest.param(None, "0" * 64, ApprovalConflict, id="request-digest"),
        pytest.param(
            "tampered-requester",
            None,
            ApprovalRequesterMismatch,
            id="requester",
        ),
    ],
)
def test_runtime_service_rejects_tampered_task_linked_exec_binding_before_outer_transition(
    tmp_path: Path,
    tampered_requester_id: str | None,
    tampered_request_digest: str | None,
    expected_error: type[ApprovalConflict] | type[ApprovalRequesterMismatch],
) -> None:
    class _RecordingProvider(_ApiRuntimePythonDirectProvider):
        def __init__(self) -> None:
            self.commands: list[str] = []

        def build_subprocess_spec(self, command: str, *, tty: bool = False) -> SubprocessSpec:
            self.commands.append(command)
            return super().build_subprocess_spec(command, tty=tty)

    async def scenario() -> None:
        task_id = "tampered-linked-binding-task"
        approval_id = "task-approval-tampered-linked-binding"
        linked_approval_id = "exec-approval-tampered-linked-binding"
        runtime_store = RuntimeStore(tmp_path / "runtime.db")
        await runtime_store.initialize()
        await runtime_store.create_task_run(
            task_id=task_id,
            input_text="run bound linked command",
            session_id=None,
            project_id=None,
            workspace_dir=str(tmp_path),
            project_workspace_dir=None,
            task_workspace_dir=None,
            inference_overrides={},
        )
        _, task_request_digest, task_context_digest = derive_approval_binding(
            requester_id=f"runtime-task:{task_id}",
            request={
                "tool_name": "exec_command",
                "arguments": {"command": "print('linked')", "shell": "test"},
            },
            authorization_context={
                "source": "runtime_task_approval",
                "task_id": task_id,
                "call_id": "call-tampered-linked-binding",
            },
        )
        await runtime_store.create_approval_request(
            approval_id=approval_id,
            task_id=task_id,
            call_id="call-tampered-linked-binding",
            tool_name="exec_command",
            arguments={"command": "print('linked')", "shell": "test"},
            metadata={"approval_id": linked_approval_id},
            requester_id=f"runtime-task:{task_id}",
            request_digest=task_request_digest,
            context_digest=task_context_digest,
        )
        command_payload = {
            "command": "print('linked')",
            "shell": "test",
            "workdir": str(tmp_path),
            "env": None,
            "timeout_sec": 5.0,
            "background": False,
            "tty": False,
            "approval_state": "approved",
        }
        requester_id, request_digest, context_digest = derive_approval_binding(
            requester_id=f"runtime-task:{task_id}",
            request={
                "command": "print('linked')",
                "shell": "test",
                "scope": "dangerous_command",
                "command_payload": command_payload,
            },
            authorization_context={"source": "exec_command", "owner_task_id": task_id},
        )
        approvals = InMemoryApprovalStore()
        approvals.create(
            approval_id=linked_approval_id,
            command="print('linked')",
            shell="test",
            scope="dangerous_command",
            metadata={APPROVAL_OWNER_TASK_ID_KEY: task_id},
            command_payload=command_payload,
            requester_id=tampered_requester_id or requester_id,
            request_digest=tampered_request_digest or request_digest,
            context_digest=context_digest,
        )
        provider = _RecordingProvider()
        service = RuntimeService(
            engine=_RuntimeFakeEngine(),
            store=runtime_store,
            exec_approval_store=approvals,
            exec_runtime=ExecRuntime(providers={"test": provider}, default_shell="test"),
        )

        with pytest.raises(expected_error):
            await service.resolve_approval(approval_id, decision="approve_once")

        outer = await runtime_store.get_approval_request(approval_id)
        assert outer is not None
        assert outer["status"] == "pending"
        linked = approvals.get(linked_approval_id)
        assert linked is not None
        assert linked.status == "pending"
        assert provider.commands == []

    asyncio.run(scenario())
