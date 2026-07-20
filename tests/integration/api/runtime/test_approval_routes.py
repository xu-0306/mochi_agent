"""Runtime API tests grouped by ownership."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.runtime.approvals import InMemoryApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore
from tests.support.exec_providers import PythonDirectProvider as _ApiRuntimePythonDirectProvider

from ._support import (
    _BACKGROUND_SMOKE_COMMAND,
    _RuntimeExecLinkedBackgroundFakeEngine,
    _RuntimeExecLinkedFakeEngine,
    _RuntimeFakeEngine,
    _wait_until,
)


def test_approvals_api_includes_and_resolves_standalone_exec_approvals(tmp_path: Path) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    exec_approval_store = InMemoryApprovalStore()
    created = exec_approval_store.create(
        approval_id="exec-approval-standalone-1",
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
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    runtime_service = RuntimeService(
        engine=_RuntimeFakeEngine(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service

    with TestClient(app) as client:
        pending_response = client.get("/v1/approvals?status=pending")
        assert pending_response.status_code == 200
        pending_items = pending_response.json()
        standalone = [
            item
            for item in pending_items
            if item["approval_id"] == created.approval_id
        ]
        assert len(standalone) == 1
        assert standalone[0]["tool_name"] == "exec_command"
        assert standalone[0]["task_id"] is None
        assert standalone[0]["source"] == "exec_runtime"
        assert standalone[0]["requires_approval"] is True
        assert standalone[0]["exec_approval_id"] == created.approval_id
        assert standalone[0]["exec_status"] == "pending"
        assert standalone[0]["exec_session_id"] is None
        assert standalone[0]["execution_result"] is None

        resolve_response = client.post(
            f"/v1/approvals/{created.approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allowed"},
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        assert resolved["approval_id"] == created.approval_id
        assert resolved["status"] == "consumed"
        assert resolved["reason"] == "allowed"
        assert resolved["source"] == "exec_runtime"
        assert resolved["execution_result"]["status"] == "completed"
        assert "approved exec" in resolved["execution_result"]["stdout"]
        assert resolved["exec_session_id"] is not None

        approved_response = client.get("/v1/approvals?status=consumed")
        assert approved_response.status_code == 200
        approved_items = approved_response.json()
        assert any(item["approval_id"] == created.approval_id for item in approved_items)

    updated = exec_approval_store.get(created.approval_id)
    assert updated is not None
    assert updated.status == "consumed"
    assert updated.execution_result is not None
    assert updated.execution_result["status"] == "completed"

def test_runtime_task_resolve_syncs_linked_exec_approval(tmp_path: Path) -> None:
    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    engine = _RuntimeExecLinkedFakeEngine(exec_approval_store=exec_approval_store)
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )
    runtime_service = RuntimeService(
        engine=engine,
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={"input_message": "run exec command", "workspace_dir": str(tmp_path / "workspace")},
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        assert waiting["pending_approval"] is not None

        approvals_response = client.get("/v1/approvals?status=pending")
        assert approvals_response.status_code == 200
        approvals = approvals_response.json()
        assert len(approvals) == 1
        runtime_approval = approvals[0]
        assert runtime_approval["tool_name"] == "exec_command"
        assert runtime_approval["source"] == "task_runtime"
        assert runtime_approval["exec_approval_id"] == "exec-approval-linked-runtime-1"
        assert runtime_approval["exec_status"] == "pending"
        assert runtime_approval["exec_session_id"] is None
        assert runtime_approval["execution_result"] is None

        resolve_response = client.post(
            f"/v1/approvals/{runtime_approval['approval_id']}/resolve",
            json={"decision": "approve_once", "reason": "allow linked exec"},
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        assert resolved["status"] == "consumed"
        assert resolved["exec_approval_id"] == "exec-approval-linked-runtime-1"
        assert resolved["execution_result"]["status"] == "completed"
        assert "linked exec approved" in resolved["execution_result"]["stdout"]
        assert resolved["exec_session_id"] is not None

        done_payload = _wait_until(client, task_id, {"succeeded"})
        assert done_payload["status"] == "succeeded"
        assert "linked exec approved" in (done_payload["final_answer"] or "")
        assert done_payload["events"][-1]["type"] == "final_answer"
        assert "linked exec approved" in done_payload["events"][-1]["content"]

    linked = exec_approval_store.get(engine.linked_exec_approval_id or "")
    assert linked is not None
    assert linked.status == "consumed"
    assert linked.execution_result is not None
    assert linked.execution_result["status"] == "completed"
    assert "linked exec approved" in linked.execution_result["stdout"]
    assert engine.second_run_started is False

def test_approval_exec_session_endpoints_support_live_standalone_exec_sessions(tmp_path: Path) -> None:
    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    created = exec_approval_store.create(
        approval_id="exec-approval-live-session-1",
        command=_BACKGROUND_SMOKE_COMMAND,
        shell="test",
        scope="dangerous_command",
        reason="requires review",
        command_payload={
            "command": _BACKGROUND_SMOKE_COMMAND,
            "shell": "test",
            "workdir": str(tmp_path),
            "env": None,
            "timeout_sec": 30.0,
            "background": True,
            "tty": False,
            "approval_state": "approved",
        },
    )
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    runtime_service = RuntimeService(
        engine=_RuntimeFakeEngine(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        resolve_response = client.post(
            f"/v1/approvals/{created.approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allowed"},
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        session_id = resolved["exec_session_id"]
        assert resolved["exec_status"] == "consumed"
        assert session_id is not None

        session_response = client.get(
            f"/v1/approvals/{created.approval_id}/exec-session",
            params={"yield_time_ms": 50},
        )
        assert session_response.status_code == 200
        session_payload = session_response.json()
        assert session_payload["approval_id"] == created.approval_id
        assert session_payload["source"] == "exec_runtime"
        assert session_payload["exec_approval_id"] == created.approval_id
        assert session_payload["session_id"] == session_id
        assert session_payload["status"] == "consumed"
        assert session_payload["live_status"] == "available"
        assert session_payload["session"]["status"] in {"running", "completed"}

        stop_response = client.post(f"/v1/approvals/{created.approval_id}/exec-session/stop")
        assert stop_response.status_code == 200
        stop_payload = stop_response.json()
        assert stop_payload["approval_id"] == created.approval_id
        assert stop_payload["session_id"] == session_id
        assert stop_payload["stop_status"] in {"killed", "completed"}
        assert stop_payload["session"]["status"] in {"killed", "completed"}

        approved_response = client.get("/v1/approvals?status=consumed")
        assert approved_response.status_code == 200
        approved_items = approved_response.json()
        updated = next(item for item in approved_items if item["approval_id"] == created.approval_id)
        assert updated["execution_result"]["status"] == stop_payload["session"]["status"]
        assert updated["exec_session_id"] == session_id

def test_approval_exec_session_endpoints_reject_non_exec_approvals(tmp_path: Path) -> None:
    app = create_app()
    engine = _RuntimeFakeEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={"input_message": "run shell approval", "workspace_dir": str(tmp_path / "workspace")},
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        approval_id = waiting["pending_approval"]["id"]

        session_response = client.get(f"/v1/approvals/{approval_id}/exec-session")
        assert session_response.status_code == 404

        stop_response = client.post(f"/v1/approvals/{approval_id}/exec-session/stop")
        assert stop_response.status_code == 404

def test_approval_exec_session_endpoints_return_conflict_without_live_session(tmp_path: Path) -> None:
    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    engine = _RuntimeExecLinkedFakeEngine(exec_approval_store=exec_approval_store)
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    runtime_service = RuntimeService(
        engine=engine,
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
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

        session_response = client.get(f"/v1/approvals/{approval_id}/exec-session")
        assert session_response.status_code == 409

        stop_response = client.post(f"/v1/approvals/{approval_id}/exec-session/stop")
        assert stop_response.status_code == 409

def test_linked_runtime_approval_stop_persists_latest_exec_state(tmp_path: Path) -> None:
    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    engine = _RuntimeExecLinkedBackgroundFakeEngine(exec_approval_store=exec_approval_store)
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )
    runtime_service = RuntimeService(
        engine=engine,
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={"input_message": "run background exec command", "workspace_dir": str(tmp_path / "workspace")},
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        approval_id = waiting["pending_approval"]["id"]

        resolve_response = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allow linked background exec"},
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        session_id = resolved["exec_session_id"]
        assert resolved["execution_result"]["status"] in {"running", "completed"}
        assert session_id is not None

        stop_response = client.post(f"/v1/approvals/{approval_id}/exec-session/stop")
        assert stop_response.status_code == 200
        stop_payload = stop_response.json()
        assert stop_payload["session_id"] == session_id
        assert stop_payload["session"]["status"] in {"killed", "completed"}

        approved_response = client.get("/v1/approvals?status=consumed")
        assert approved_response.status_code == 200
        approved_items = approved_response.json()
        updated = next(item for item in approved_items if item["approval_id"] == approval_id)
        assert updated["exec_approval_id"] == "exec-approval-linked-runtime-bg-1"
        assert updated["execution_result"]["status"] == stop_payload["session"]["status"]
        assert updated["exec_session_id"] == session_id

def test_standalone_exec_approval_save_rule_without_valid_rule_is_rejected(
    tmp_path: Path,
) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {'sessions_dir': str(tmp_path / 'sessions')}
    )

    exec_approval_store = InMemoryApprovalStore()
    created = exec_approval_store.create(
        approval_id='exec-approval-no-rule-1',
        command='print(''approved exec'')',
        shell='test',
        scope='dangerous_command',
        reason='requires review',
        command_payload={
            'command': 'print(''approved exec'')',
            'shell': 'test',
            'workdir': str(tmp_path),
            'env': None,
            'timeout_sec': 5.0,
            'background': False,
            'tty': False,
            'approval_state': 'approved',
        },
    )
    runtime_service = RuntimeService(
        engine=_RuntimeFakeEngine(),
        store=RuntimeStore(tmp_path / 'sessions' / 'runtime.db'),
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={'test': _ApiRuntimePythonDirectProvider()},
            default_shell='test',
        ),
    )
    app.state.runtime_service = runtime_service

    with TestClient(app) as client:
        response = client.post(
            f'/v1/approvals/{created.approval_id}/resolve',
            json={'decision': 'approve_and_save_rule', 'reason': 'allow but no rule'},
        )
        assert response.status_code == 400
        assert 'valid command rule' in response.json()['detail']

    updated = exec_approval_store.get(created.approval_id)
    assert updated is not None
    assert updated.status == 'pending'
    assert exec_approval_store.list_side_effects(created.approval_id) == []

def test_task_approval_save_rule_without_valid_rule_is_rejected(tmp_path: Path) -> None:
    app = create_app()
    engine = _RuntimeFakeEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {'sessions_dir': str(tmp_path / 'sessions')}
    )

    with TestClient(app) as client:
        create_response = client.post(
            '/v1/tasks',
            json={'input_message': 'run shell approval', 'workspace_dir': str(tmp_path / 'workspace')},
        )
        assert create_response.status_code == 200
        task_id = create_response.json()['task_id']
        waiting = _wait_until(client, task_id, {'awaiting_approval'})
        approval_id = waiting['pending_approval']['id']

        response = client.post(
            f'/v1/approvals/{approval_id}/resolve',
            json={'decision': 'approve_and_save_rule', 'reason': 'allow but no rule'},
        )
        assert response.status_code == 400
        assert 'valid command rule' in response.json()['detail']
        pending = client.get('/v1/approvals?status=pending').json()
        assert any(item['approval_id'] == approval_id for item in pending)
