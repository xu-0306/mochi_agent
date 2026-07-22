from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request

from mochi.config.manager import load_config_snapshot
from mochi.runtime import approval_side_effect_worker as side_effect_worker_module
from mochi.runtime.approval_side_effect_worker import ApprovalSideEffectWorker
from mochi.runtime.approvals import (
    APPROVAL_OWNER_TASK_ID_KEY,
    ApprovalConflict,
    ApprovalExpired,
    ApprovalRequesterMismatch,
    ConsumeRecoveryOutcome,
    InMemoryApprovalStore,
    PersistentApprovalStore,
)
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.models import ApprovalResolution
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore
from mochi.security.file_contract import (
    AuthorizationContext,
    AuthorizationEnvelope,
    EnvVarHash,
    ExecRequest,
    FileIdentity,
    ResourceLimits,
    authorization_request_digest,
)


def _future(seconds: int = 300) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def _past() -> str:
    return (datetime.now(UTC) - timedelta(seconds=1)).isoformat()


def _create_exec(store: PersistentApprovalStore, *, expires_at: str | None = None) -> None:
    store.create(
        approval_id="approval-1",
        command="echo ok",
        shell="powershell",
        scope="dangerous_command",
        requester_id="requester-1",
        request_digest="a" * 64,
        context_digest="b" * 64,
        expires_at=expires_at or _future(),
        command_payload={"command": "echo ok", "shell": "powershell"},
    )


def test_resolve_save_rule_is_single_use_and_outbox_is_atomic(tmp_path: Path) -> None:
    db_path = tmp_path / "approvals.db"
    store = PersistentApprovalStore(db_path)
    _create_exec(store)

    resolved = store.resolve(
        "approval-1",
        decision="approve_and_save_rule",
        requester_id="requester-1",
        request_digest="a" * 64,
        context_digest="b" * 64,
        rule_side_effect={
            "payload": {"action": "allow", "command": "echo ok"},
            "target_config_path": "config.yaml",
        },
    )

    assert resolved is not None
    assert resolved.status == "approved_once"
    assert resolved.resolution_kind == "approve_and_save_rule"
    effects = store.list_side_effects("approval-1")
    assert len(effects) == 1
    assert effects[0]["status"] == "pending"
    assert effects[0]["kind"] == "save_command_rule"

    with pytest.raises(ApprovalConflict):
        store.resolve("approval-1", decision="approve_once")

    with sqlite3.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM approval_side_effects").fetchone()[0] == 1


def test_rule_outbox_delivery_is_restart_safe_and_independent_from_execution(tmp_path: Path) -> None:
    db_path = tmp_path / "approvals.db"
    config_path = tmp_path / "config.yaml"
    store = PersistentApprovalStore(db_path)
    _create_exec(store)
    store.resolve(
        "approval-1",
        decision="approve_and_save_rule",
        requester_id="requester-1",
        request_digest="a" * 64,
        context_digest="b" * 64,
        rule_side_effect={
            "payload": {
                "tokens": ["echo", "ok"],
                "decision": "allow",
                "match": "exact",
                "shells": ["powershell"],
            },
            "target_config_path": str(config_path),
        },
    )
    claim = store.consume(
        "approval-1",
        execution_idempotency_key="execution-1",
        lease_owner="execution-worker",
        requester_id="requester-1",
        request_digest="a" * 64,
        context_digest="b" * 64,
    )
    store.complete_consumption(
        "approval-1",
        execution_idempotency_key="execution-1",
        lease_owner="execution-worker",
        lease_token=claim.consume_lease_token or "",
        execution_result={"status": "completed"},
    )

    first_worker = ApprovalSideEffectWorker([db_path])
    delivered = asyncio.run(first_worker.deliver_available())

    assert delivered[0]["status"] == "delivered"
    snapshot = load_config_snapshot(config_path)
    assert [rule.tokens for rule in snapshot.config.security.command_rules] == [["echo", "ok"]]
    effect = store.list_side_effects("approval-1")[0]
    assert effect["status"] == "delivered"

    # Simulate a crash after config replace but before the delivered DB mark.
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE approval_side_effects
            SET status='retrying',lease_owner=NULL,lease_expires_at=NULL,delivered_at=NULL
            WHERE side_effect_id=?
            """,
            (effect["side_effect_id"],),
        )
        conn.commit()
    restarted_worker = ApprovalSideEffectWorker([db_path])
    replay = asyncio.run(restarted_worker.deliver_available())

    assert replay[0]["status"] == "delivered"
    restarted = load_config_snapshot(config_path).config
    assert [rule.tokens for rule in restarted.security.command_rules] == [["echo", "ok"]]
    assert restarted.security.applied_rule_side_effect_ids == [effect["side_effect_id"]]
    assert store.get("approval-1").status == "consumed"  # type: ignore[union-attr]


def test_two_workers_claim_one_rule_side_effect_once(tmp_path: Path) -> None:
    async def scenario() -> None:
        db_path = tmp_path / "approvals.db"
        config_path = tmp_path / "config.yaml"
        store = PersistentApprovalStore(db_path)
        _create_exec(store)
        store.resolve(
            "approval-1",
            decision="approve_and_save_rule",
            requester_id="requester-1",
            request_digest="a" * 64,
            context_digest="b" * 64,
            rule_side_effect={
                "payload": {
                    "tokens": ["echo", "once"],
                    "decision": "allow",
                    "match": "exact",
                    "shells": ["powershell"],
                },
                "target_config_path": str(config_path),
            },
        )

        first, second = await asyncio.gather(
            ApprovalSideEffectWorker([db_path]).deliver_available(max_items=1),
            ApprovalSideEffectWorker([db_path]).deliver_available(max_items=1),
        )

        delivered = [item for item in [*first, *second] if item["status"] == "delivered"]
        assert len(delivered) == 1
        effect = store.list_side_effects("approval-1")[0]
        assert effect["status"] == "delivered"
        snapshot = load_config_snapshot(config_path)
        assert [rule.tokens for rule in snapshot.config.security.command_rules] == [
            ["echo", "once"]
        ]
        assert snapshot.config.security.applied_rule_side_effect_ids == [
            effect["side_effect_id"]
        ]

    asyncio.run(scenario())


def test_retryable_rule_delivery_uses_backoff_instead_of_hot_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "approvals.db"
    config_path = tmp_path / "config.yaml"
    store = PersistentApprovalStore(db_path)
    _create_exec(store)
    store.resolve(
        "approval-1",
        decision="approve_and_save_rule",
        requester_id="requester-1",
        request_digest="a" * 64,
        context_digest="b" * 64,
        rule_side_effect={
            "payload": {
                "tokens": ["echo", "retry"],
                "decision": "allow",
                "match": "exact",
                "shells": ["powershell"],
            },
            "target_config_path": str(config_path),
        },
    )
    monkeypatch.setattr(
        side_effect_worker_module,
        "save_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("replace busy")),
    )

    results = asyncio.run(ApprovalSideEffectWorker([db_path]).deliver_available(max_items=32))

    assert len(results) == 1
    assert results[0]["status"] == "retrying"
    effect = store.list_side_effects("approval-1")[0]
    assert effect["status"] == "retrying"
    assert effect["attempts"] == 1
    assert datetime.fromisoformat(effect["lease_expires_at"]) > datetime.now(UTC)


def test_invalid_rule_delivery_is_permanently_failed(tmp_path: Path) -> None:
    db_path = tmp_path / "approvals.db"
    store = PersistentApprovalStore(db_path)
    _create_exec(store)
    store.resolve(
        "approval-1",
        decision="approve_and_save_rule",
        requester_id="requester-1",
        request_digest="a" * 64,
        context_digest="b" * 64,
        rule_side_effect={
            "payload": {"tokens": "not-a-token-list"},
            "target_config_path": str(tmp_path / "config.yaml"),
        },
    )

    results = asyncio.run(ApprovalSideEffectWorker([db_path]).deliver_available())

    assert results[0]["status"] == "failed"
    assert store.list_side_effects("approval-1")[0]["status"] == "failed"


def test_malformed_outbox_json_is_permanently_failed(tmp_path: Path) -> None:
    db_path = tmp_path / "approvals.db"
    store = PersistentApprovalStore(db_path)
    _create_exec(store)
    store.resolve(
        "approval-1",
        decision="approve_and_save_rule",
        requester_id="requester-1",
        request_digest="a" * 64,
        context_digest="b" * 64,
        rule_side_effect={
            "payload": {"tokens": ["echo", "ok"], "decision": "allow"},
            "target_config_path": str(tmp_path / "config.yaml"),
        },
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE approval_side_effects SET payload_json='{' ")
        conn.commit()

    results = asyncio.run(ApprovalSideEffectWorker([db_path]).deliver_available())

    assert results[0]["status"] == "failed"
    assert store.list_side_effects("approval-1")[0]["status"] == "failed"


def test_expired_and_requester_mismatch_fail_closed(tmp_path: Path) -> None:
    store = PersistentApprovalStore(tmp_path / "approvals.db")
    _create_exec(store, expires_at=_past())

    with pytest.raises(ApprovalRequesterMismatch):
        store.resolve("approval-1", decision="approve_once", requester_id="attacker")

    with pytest.raises(ApprovalExpired):
        store.resolve("approval-1", decision="approve_once", requester_id="requester-1")

    assert store.get("approval-1").status == "expired"  # type: ignore[union-attr]


def test_consume_once_and_idempotency_key_are_cas_guarded(tmp_path: Path) -> None:
    store = PersistentApprovalStore(tmp_path / "approvals.db")
    _create_exec(store)
    store.resolve("approval-1", decision="approve_once", requester_id="requester-1")

    consuming = store.consume(
        "approval-1",
        execution_idempotency_key="execution-1",
        lease_owner="worker-1",
        requester_id="requester-1",
        request_digest="a" * 64,
        context_digest="b" * 64,
    )
    assert consuming.status == "consuming"

    with pytest.raises(ApprovalConflict):
        store.consume(
            "approval-1",
            execution_idempotency_key="execution-2",
            lease_owner="worker-2",
        )

    consumed = store.complete_consumption(
        "approval-1",
        execution_idempotency_key="execution-1",
        lease_owner="worker-1",
        lease_token=consuming.consume_lease_token or "",
        execution_result={"status": "completed", "stdout": "ok"},
    )
    assert consumed.status == "consumed"
    assert consumed.execution_result == {"status": "completed", "stdout": "ok"}

    with pytest.raises(ApprovalConflict):
        store.complete_consumption(
            "approval-1",
            execution_idempotency_key="execution-1",
            lease_owner="worker-1",
            lease_token=consuming.consume_lease_token or "",
            execution_result={"status": "completed"},
        )


def test_in_memory_approval_payloads_are_immutable_snapshots() -> None:
    store = InMemoryApprovalStore()
    payload: dict[str, Any] = {
        "command": "echo ok",
        "env": {"TOKEN": "secret"},
    }
    metadata: dict[str, Any] = {"nested": {"owner": "original"}}
    created = store.create(
        approval_id="approval-1",
        command="echo ok",
        shell="powershell",
        scope="dangerous_command",
        command_payload=payload,
        metadata=metadata,
    )

    payload["env"]["TOKEN"] = "caller-mutated"
    metadata["nested"]["owner"] = "caller-mutated"
    assert created.command_payload is not None
    created.command_payload["env"]["TOKEN"] = "returned-object-mutated"
    created.metadata["nested"]["owner"] = "returned-object-mutated"

    stored = store.get("approval-1")
    assert stored is not None
    assert stored.command_payload is not None
    assert stored.command_payload["env"]["TOKEN"] == "secret"
    assert stored.metadata["nested"]["owner"] == "original"

    resolved = store.resolve("approval-1", decision="approve_once")
    assert resolved is not None
    assert resolved.command_payload is not None
    resolved.command_payload["env"]["TOKEN"] = "resolved-object-mutated"
    claim = store.consume(
        "approval-1",
        execution_idempotency_key="execution-1",
        lease_owner="worker-1",
    )
    assert claim.command_payload is not None
    assert claim.command_payload["env"]["TOKEN"] == "secret"


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ("applied", "consumed"),
        ("not_started", "approved_once"),
        ("unknown", "execution_failed"),
    ],
)
def test_stale_consume_lease_recovery_is_evidence_driven(
    tmp_path: Path,
    outcome: ConsumeRecoveryOutcome,
    expected: str,
) -> None:
    db_path = tmp_path / f"{outcome}.db"
    store = PersistentApprovalStore(db_path)
    _create_exec(store)
    store.resolve("approval-1", decision="approve_once")
    store.consume(
        "approval-1",
        execution_idempotency_key=f"execution-{outcome}",
        lease_owner="dead-worker",
    )
    with sqlite3.connect(db_path) as conn:  # controlled stale-lease fixture
        conn.execute(
            "UPDATE exec_approval_requests SET consume_lease_expires_at=? WHERE approval_id=?",
            (_past(), "approval-1"),
        )
        conn.commit()

    recovered = store.recover_consumption(
        "approval-1",
        outcome=outcome,
        execution_result={"status": "completed"} if outcome == "applied" else None,
    )
    assert recovered.status == expected


def test_runtime_store_resolve_and_outbox_share_one_transaction(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = RuntimeStore(tmp_path / "runtime.db")
        await store.initialize()
        await store.create_task_run(
            task_id="task-1",
            input_text="run",
            session_id="session-1",
            project_id=None,
            workspace_dir=str(tmp_path),
            project_workspace_dir=None,
            task_workspace_dir=None,
            inference_overrides={},
        )
        await store.create_approval_request(
            approval_id="approval-1",
            task_id="task-1",
            call_id="call-1",
            tool_name="exec_command",
            arguments={"command": "echo ok"},
            requester_id="requester-1",
            request_digest="a" * 64,
            context_digest="b" * 64,
            expires_at=_future(),
        )
        resolved = await store.resolve_approval_request(
            "approval-1",
            decision="approve_and_save_rule",
            requester_id="requester-1",
            request_digest="a" * 64,
            context_digest="b" * 64,
            rule_side_effect={
                "payload": {"command": "echo ok"},
                "target_config_path": "config.yaml",
            },
        )
        assert resolved is not None
        assert resolved["status"] == "approved_once"
        assert resolved["resolution_kind"] == "approve_and_save_rule"
        assert resolved["rule_persistence_status"] == "pending"

        with pytest.raises(ApprovalConflict):
            await store.resolve_approval_request("approval-1", decision="approve_once")

    asyncio.run(scenario())


def _exec_envelope(*, argv: tuple[str, ...] = ("echo", "ok")) -> AuthorizationEnvelope:
    return AuthorizationEnvelope(
        schema_version=2,
        kind="exec",
        context=AuthorizationContext(
            requester_id="requester-1",
            session_id="session-1",
            task_id="task-1",
            workspace_root="C:/workspace",
            workspace_identity=FileIdentity("windows", "volume-1", "file-1", 1, False),
        ),
        policy_version="policy-1",
        file_request=None,
        exec_request=ExecRequest(
            command_utf8_sha256="c" * 64,
            shell="powershell",
            executable="powershell.exe",
            argv=argv,
            resolved_cwd="C:/workspace",
            env=(EnvVarHash("TOKEN", "d" * 64),),
            network_policy="deny",
            resource_limits=ResourceLimits(30, 512, 1024),
            requested_escalation="none",
            sandbox_backend="windows-appcontainer",
            sandbox_capability_plan_digest="e" * 64,
        ),
    )


def test_runtime_recovery_uses_terminal_task_evidence(tmp_path: Path) -> None:
    async def scenario() -> None:
        db_path = tmp_path / "runtime-evidence.db"
        store = RuntimeStore(db_path)
        await store.initialize()
        await store.create_task_run(
            task_id="task-evidence",
            input_text="run",
            session_id=None,
            project_id=None,
            workspace_dir=str(tmp_path),
            project_workspace_dir=None,
            task_workspace_dir=None,
            task_type=None,
            metadata={},
            inference_overrides={},
        )
        await store.create_approval_request(
            approval_id="approval-evidence",
            task_id="task-evidence",
            call_id="call-evidence",
            tool_name="exec_command",
            arguments={"command": "echo ok"},
        )
        await store.resolve_approval_request(
            "approval-evidence",
            decision="approve_once",
        )
        await store.consume_approval_request(
            "approval-evidence",
            execution_idempotency_key="execution-evidence",
            lease_owner="dead-worker",
        )
        await store.update_task_status("task-evidence", "succeeded")
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                "UPDATE approval_requests SET consume_lease_expires_at=? WHERE id=?",
                (_past(), "approval-evidence"),
            )
            conn.commit()

        assert await store.recover_stale_approval_consumptions() == 1
        recovered = await store.get_approval_request("approval-evidence")
        assert recovered is not None
        assert recovered["status"] == "consumed"
        assert recovered["execution_result"]["status"] == "succeeded"

    asyncio.run(scenario())


def test_linked_exec_owner_blocks_direct_resolution_before_outer_row(tmp_path: Path) -> None:
    async def scenario() -> None:
        store = RuntimeStore(tmp_path / "runtime-owner.db")
        await store.initialize()
        exec_store = InMemoryApprovalStore()
        exec_store.create(
            approval_id="linked-exec",
            command="echo ok",
            shell="powershell",
            scope="dangerous_command",
            metadata={APPROVAL_OWNER_TASK_ID_KEY: "task-owner"},
            command_payload={"command": "echo ok", "shell": "powershell"},
        )
        service = RuntimeService(
            engine=object(),
            store=store,
            exec_approval_store=exec_store,
            exec_runtime=ExecRuntime(),
        )
        with pytest.raises(ApprovalConflict):
            await service.resolve_approval(
                "linked-exec",
                decision="approve_once",
            )
        assert exec_store.get("linked-exec").status == "pending"  # type: ignore[union-attr]

    asyncio.run(scenario())


def test_periodic_recovery_sweeps_lease_that_expires_after_startup(tmp_path: Path) -> None:
    async def scenario() -> None:
        exec_store = PersistentApprovalStore(tmp_path / "periodic-approval.db")
        _create_exec(exec_store)
        exec_store.resolve("approval-1", decision="approve_once")
        exec_store.consume(
            "approval-1",
            execution_idempotency_key="execution-periodic",
            lease_owner="dead-worker",
            lease_seconds=1,
        )
        service = RuntimeService(
            engine=object(),
            store=RuntimeStore(tmp_path / "periodic-runtime.db"),
            exec_approval_store=exec_store,
            exec_runtime=ExecRuntime(),
        )
        service.set_scheduler_poll_interval(0.05)
        await service.start()
        assert service._side_effect_task is not None
        assert not service._side_effect_task.done()
        try:
            for _ in range(40):
                if exec_store.get("approval-1").status == "execution_failed":  # type: ignore[union-attr]
                    break
                await asyncio.sleep(0.05)
            assert exec_store.get("approval-1").status == "execution_failed"  # type: ignore[union-attr]
        finally:
            await service.close()
        assert service._side_effect_task is None

    asyncio.run(scenario())


def test_canonical_exec_digest_covers_exact_authorization_contract() -> None:
    baseline = authorization_request_digest(_exec_envelope())
    assert len(baseline) == 64
    assert baseline != authorization_request_digest(
        _exec_envelope(argv=("echo", "changed"))
    )


def test_request_and_context_digest_mismatch_are_conflicts(tmp_path: Path) -> None:
    store = PersistentApprovalStore(tmp_path / 'approvals.db')
    _create_exec(store)
    with pytest.raises(ApprovalConflict):
        store.resolve(
            'approval-1',
            decision='approve_once',
            request_digest='f' * 64,
        )
    with pytest.raises(ApprovalConflict):
        store.resolve(
            'approval-1',
            decision='approve_once',
            context_digest='f' * 64,
        )
    assert store.get('approval-1').status == 'pending'  # type: ignore[union-attr]


def test_concurrent_resolve_has_exactly_one_winner(tmp_path: Path) -> None:
    store = PersistentApprovalStore(tmp_path / 'approvals.db')
    _create_exec(store)

    def attempt(_: int) -> str:
        try:
            store.resolve('approval-1', decision='approve_once')
        except ApprovalConflict:
            return 'conflict'
        return 'resolved'

    with ThreadPoolExecutor(max_workers=16) as pool:
        outcomes = list(pool.map(attempt, range(32)))
    assert outcomes.count('resolved') == 1
    assert outcomes.count('conflict') == 31


def test_invalid_outbox_payload_does_not_resolve(tmp_path: Path) -> None:
    store = PersistentApprovalStore(tmp_path / 'approvals.db')
    _create_exec(store)
    with pytest.raises(ValueError):
        store.resolve(
            'approval-1',
            decision='approve_and_save_rule',
            rule_side_effect={'payload': {'command': 'echo ok'}},
        )
    assert store.get('approval-1').status == 'pending'  # type: ignore[union-attr]
    assert store.list_side_effects('approval-1') == []


@pytest.mark.parametrize(
    ('error', 'status_code'),
    [
        (ApprovalRequesterMismatch('requester mismatch'), 403),
        (ApprovalConflict('conflict'), 409),
        (ApprovalExpired('expired'), 410),
    ],
)
def test_approval_route_maps_typed_lifecycle_errors(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    status_code: int,
) -> None:
    import mochi.api.server as server_module  # ensure route graph is initialized

    _ = server_module
    from mochi.api.routes import approvals as approval_routes
    class _Service:
        async def resolve_approval(self, *args: object, **kwargs: object) -> None:
            raise error

    async def _fake_service(app: FastAPI) -> _Service:
        return _Service()

    monkeypatch.setattr(approval_routes, '_get_runtime_service', _fake_service)

    async def scenario() -> None:
        request = Request({'type': 'http', 'app': FastAPI()})
        with pytest.raises(HTTPException) as caught:
            await approval_routes.resolve_approval(
                request,
                'approval-1',
                ApprovalResolution(decision='approve_once'),
            )
        assert caught.value.status_code == status_code

    asyncio.run(scenario())


def test_supersede_is_a_pending_only_cas_transition(tmp_path: Path) -> None:
    store = PersistentApprovalStore(tmp_path / 'approvals.db')
    _create_exec(store)
    superseded = store.supersede('approval-1', reason='new request replaced it')
    assert superseded.status == 'superseded'
    with pytest.raises(ApprovalConflict):
        store.resolve('approval-1', decision='approve_once')


def test_startup_stale_recovery_fails_unknown_outcome_closed(tmp_path: Path) -> None:
    db_path = tmp_path / "approvals.db"
    store = PersistentApprovalStore(db_path)
    _create_exec(store)
    store.resolve('approval-1', decision='approve_once')
    store.consume(
        'approval-1',
        execution_idempotency_key='execution-1',
        lease_owner='dead-worker',
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            'UPDATE exec_approval_requests SET consume_lease_expires_at=? WHERE approval_id=?',
            (_past(), 'approval-1'),
        )
        conn.commit()
    recovered = store.recover_stale_consumptions()
    assert [item.status for item in recovered] == ['execution_failed']
