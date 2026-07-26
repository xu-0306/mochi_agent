"""Runtime API tests grouped by ownership."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.runtime.approvals import InMemoryApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.ordinary_chat_session_gate import OrdinaryChatSessionGateError
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore
from mochi.sessions.store import SessionStore
from tests.support.exec_providers import PythonDirectProvider as _ApiRuntimePythonDirectProvider

from ._support import (
    _BACKGROUND_SMOKE_COMMAND,
    _RuntimeExecLinkedBackgroundFakeEngine,
    _RuntimeExecLinkedFakeEngine,
    _RuntimeFakeEngine,
    _wait_until,
)


class _ReconcileRouteService:
    def __init__(
        self,
        *,
        session_id: str | None = "reconcile-session",
        outcome: dict[str, object] | None = None,
        ordinary_chat: bool = True,
        gate_error: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.outcome = outcome or {"status": "continued"}
        self.ordinary_chat = ordinary_chat
        self.gate_error = gate_error
        self.policy: dict[str, object] | None = None
        self.approval_status = "pending"
        self.engine_calls = 0
        self.resolve_calls = 0

    def update_security_config(self, *_: object) -> None:
        pass

    def update_sandbox_config(self, *_: object) -> None:
        pass

    def bind_app_config(self, **_: object) -> None:
        pass

    async def start(self) -> None:
        pass

    def ordinary_chat_approval_session_id(self, _: str) -> str | None:
        return self.session_id

    def ordinary_chat_approval_owner(self, _: str) -> tuple[bool, str | None]:
        return self.ordinary_chat, self.session_id

    async def resolve_approval(
        self,
        _: str,
        **kwargs: object,
    ) -> dict[str, object]:
        self.resolve_calls += 1
        policy = kwargs.get("current_permission_policy")
        self.policy = dict(policy) if isinstance(policy, dict) else None
        self.approval_status = "consumed"
        self.engine_calls += 1
        return {"status": "consumed"}

    async def reconcile_recovered_ordinary_chat_approval(
        self,
        **kwargs: object,
    ) -> dict[str, object]:
        if self.gate_error is not None:
            raise OrdinaryChatSessionGateError(self.gate_error)
        self.policy = None
        return self.outcome


class _CountingSessionStore(SessionStore):
    def __init__(self, sessions_dir: Path) -> None:
        super().__init__(sessions_dir)
        self.load_calls = 0

    async def load_strict_snapshot(self, session_id: str):  # type: ignore[no-untyped-def]
        self.load_calls += 1
        return await super().load_strict_snapshot(session_id)


def _reconcile_app(
    tmp_path: Path,
    service: _ReconcileRouteService,
    *,
    session_store: SessionStore | None = None,
) -> TestClient:
    app = create_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )
    app.state.runtime_service = service
    if session_store is not None:
        app.state.session_store = session_store
    return TestClient(app)


def _created_session_event(
    session_id: str,
    *,
    security_override: dict[str, object] | None = None,
) -> dict[str, object]:
    event: dict[str, object] = {
        "type": "session_meta",
        "event": "created",
        "session_id": session_id,
        "timestamp": "2026-07-25T00:00:00+00:00",
    }
    if security_override is not None:
        event["security_override"] = security_override
    return event


def test_ordinary_chat_reconcile_route_derives_policy_server_side(tmp_path: Path) -> None:
    service = _ReconcileRouteService()
    sessions = SessionStore(tmp_path / "sessions")
    asyncio.run(
        sessions.save_event(
            "reconcile-session",
            _created_session_event(
                "reconcile-session",
                security_override={"autonomy_mode": "high_autonomy"},
            ),
        )
    )
    with _reconcile_app(tmp_path, service) as client:
        response = client.post(
            "/v1/approvals/a/reconcile",
            json={"current_permission_policy": {"autonomy_mode": "strict"}},
        )
    assert response.status_code == 200
    assert service.policy is None


def test_ordinary_chat_resolve_route_derives_policy_from_one_session_snapshot(
    tmp_path: Path,
) -> None:
    service = _ReconcileRouteService()
    sessions = _CountingSessionStore(tmp_path / "sessions")
    asyncio.run(
        sessions.save_event(
            "reconcile-session",
            _created_session_event(
                "reconcile-session",
                security_override={"autonomy_mode": "high_autonomy"},
            ),
        )
    )

    with _reconcile_app(tmp_path, service, session_store=sessions) as client:
        response = client.post(
            "/v1/approvals/a/resolve",
            json={
                "decision": "approve_once",
                "current_permission_policy": {"autonomy_mode": "strict"},
            },
        )

    assert response.status_code == 200
    assert sessions.load_calls == 1
    assert service.policy is not None
    assert service.policy.get("autonomy_mode") == "high_autonomy"


def test_ordinary_chat_reconcile_route_reads_verified_session_once(tmp_path: Path) -> None:
    service = _ReconcileRouteService()
    sessions = _CountingSessionStore(tmp_path / "sessions")
    asyncio.run(
        sessions.save_event(
            "reconcile-session",
            _created_session_event("reconcile-session"),
        )
    )

    with _reconcile_app(tmp_path, service, session_store=sessions) as client:
        response = client.post("/v1/approvals/a/reconcile")

    assert response.status_code == 200
    assert sessions.load_calls == 0


def test_ordinary_chat_resolve_route_rejects_invalid_session_before_side_effect(
    tmp_path: Path,
) -> None:
    cases = (
        ("missing", "missing", None, None, "ordinary_chat_resolve_session_missing"),
        ("empty", "empty", [], None, "ordinary_chat_resolve_session_invalid"),
        (
            "wrong",
            "wrong",
            [_created_session_event("other")],
            None,
            "ordinary_chat_resolve_session_invalid",
        ),
        (
            "corrupt-log",
            "corrupt-log",
            None,
            "{not-valid-json}\n",
            "ordinary_chat_resolve_session_invalid",
        ),
        ("corrupt-checkpoint", None, None, None, "ordinary_chat_resolve_session_invalid"),
    )
    for name, session_id, events, raw_session, code in cases:
        service = _ReconcileRouteService(session_id=session_id)
        sessions = SessionStore(tmp_path / name / "sessions")
        if events is not None:
            if not events:
                sessions._session_path(name).parent.mkdir(parents=True, exist_ok=True)  # noqa: SLF001
                sessions._session_path(name).touch()  # noqa: SLF001
            for event in events:
                asyncio.run(sessions.save_event(name, event))
        if raw_session is not None:
            path = sessions._session_path(session_id or name)  # noqa: SLF001
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(raw_session, encoding="utf-8")

        with _reconcile_app(tmp_path / name, service) as client:
            response = client.post(
                "/v1/approvals/a/resolve",
                json={"decision": "approve_once"},
            )

        assert response.status_code == 409
        assert response.json()["detail"]["code"] == code
        assert service.approval_status == "pending"
        assert service.resolve_calls == 0
        assert service.engine_calls == 0


def test_ordinary_chat_resolve_route_rejects_malformed_durable_override(
    tmp_path: Path,
) -> None:
    service = _ReconcileRouteService()
    sessions = SessionStore(tmp_path / "sessions")
    asyncio.run(
        sessions.save_event(
            "reconcile-session",
            _created_session_event(
                "reconcile-session",
                security_override={"autonomy_mode": "not-a-valid-mode"},
            ),
        )
    )
    with _reconcile_app(tmp_path, service) as client:
        response = client.post(
            "/v1/approvals/a/resolve",
            json={"decision": "approve_once"},
        )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "ordinary_chat_resolve_session_invalid"
    assert service.resolve_calls == 0
    assert service.engine_calls == 0


def test_nonordinary_resolve_route_skips_session_validation(tmp_path: Path) -> None:
    service = _ReconcileRouteService(session_id=None, ordinary_chat=False)

    with _reconcile_app(tmp_path, service) as client:
        response = client.post(
            "/v1/approvals/a/resolve",
            json={"decision": "approve_once"},
        )

    assert response.status_code == 200
    assert service.resolve_calls == 1
    assert service.policy is None


def test_ordinary_chat_reconcile_route_rejects_invalid_session_logs(tmp_path: Path) -> None:
    for name, events, code in (
        ("missing", None, "ordinary_chat_reconciliation_session_missing"),
        ("empty", [], "ordinary_chat_reconciliation_session_invalid"),
        (
            "wrong",
            [_created_session_event("other")],
            "ordinary_chat_reconciliation_session_invalid",
        ),
        ("corrupt", None, "ordinary_chat_reconciliation_session_invalid"),
    ):
        service = _ReconcileRouteService(
            session_id=name,
            gate_error="missing" if name == "missing" else "invalid",
        )
        sessions = SessionStore(tmp_path / name / "sessions")
        if events is not None:
            if not events:
                sessions._session_path(name).parent.mkdir(parents=True, exist_ok=True)  # noqa: SLF001
                sessions._session_path(name).touch()  # noqa: SLF001
            for event in events:
                asyncio.run(sessions.save_event(name, event))
        if name == "corrupt":
            path = sessions._session_path(name)  # noqa: SLF001
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("{not-valid-json}\n", encoding="utf-8")
        with _reconcile_app(tmp_path / name, service) as client:
            response = client.post("/v1/approvals/a/reconcile")
        assert response.status_code == 409
        assert response.json()["detail"]["code"] == code


def test_ordinary_chat_reconcile_route_rejects_corrupt_checkpoint(tmp_path: Path) -> None:
    service = _ReconcileRouteService(session_id=None, gate_error="invalid")

    with _reconcile_app(tmp_path, service) as client:
        response = client.post("/v1/approvals/a/reconcile")

    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "ordinary_chat_reconciliation_session_invalid"


def test_ordinary_chat_reconcile_route_maps_unavailable_states(tmp_path: Path) -> None:
    sessions = SessionStore(tmp_path / "sessions")
    asyncio.run(
        sessions.save_event(
            "reconcile-session",
            _created_session_event("reconcile-session"),
        )
    )
    cases = [
        (
            _ReconcileRouteService(session_id=None, ordinary_chat=False),
            404,
            "ordinary_chat_reconciliation_not_found",
        ),
        (
            _ReconcileRouteService(
                outcome={
                    "status": "not_available",
                    "reason": "reconciliation_not_required",
                }
            ),
            409,
            "ordinary_chat_reconciliation_not_required",
        ),
        (
            _ReconcileRouteService(
                outcome={
                    "status": "not_available",
                    "reason": "continuation_unknown_outcome",
                }
            ),
            409,
            "ordinary_chat_continuation_unknown_outcome",
        ),
        (
            _ReconcileRouteService(
                outcome={
                    "status": "not_available",
                    "reason": "reconciliation_claim_rejected",
                }
            ),
            409,
            "ordinary_chat_reconciliation_claim_rejected",
        ),
    ]
    for service, status, code in cases:
        with _reconcile_app(tmp_path, service) as client:
            response = client.post("/v1/approvals/a/reconcile")
        assert response.status_code == status
        assert response.json()["detail"]["code"] == code


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
