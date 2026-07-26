from __future__ import annotations

import asyncio
import json
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
from mochi.security.policy import EffectivePolicyResolver
from mochi.config.schema import SecurityConfig
from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore
from mochi.tools.base import ToolExecutionContext
from mochi.tools.exec_command import ExecCommandTool
from mochi.tools.file_ops import FileWriteTool
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


def test_ordinary_chat_file_approval_is_exactly_once_and_policy_bound(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        security = SecurityConfig(require_approval_for_file_write=True)
        policy = EffectivePolicyResolver().resolve(security).to_dict()
        exec_store = PersistentApprovalStore(tmp_path / "exec-approvals.db")
        context = ToolExecutionContext(
            workspace_dir=str(workspace),
            project_workspace=str(workspace),
            session_id="chat-session-1",
            permission_policy=policy,
            state={
                "ordinary_chat_approval_context": {
                    "schema_version": 1,
                    "source": "ordinary_chat",
                    "session_id": "chat-session-1",
                    "turn_id": "turn-1",
                    "resume_cursor": {
                        "turn_id": "turn-1",
                        "tool_call_id": "call-1",
                        "tool_name": "file_write",
                    },
                }
            },
        )
        tool = FileWriteTool(
            workspace_dir=workspace,
            require_approval=False,
            approval_store=exec_store,
        )
        pending = await tool.execute(
            path="report.txt",
            content="approved once\n",
            context=context,
        )
        approval_id = str(pending.metadata["approval_id"])
        checkpoint = exec_store.get(approval_id)
        assert checkpoint is not None
        assert checkpoint.metadata["approval_source"] == "ordinary_chat"
        assert checkpoint.command_payload is not None
        durable = checkpoint.command_payload["ordinary_chat_checkpoint"]
        assert durable["normalized_arguments"]["path"] == "report.txt"
        assert durable["resume_cursor"]["tool_call_id"] == "call-1"
        assert durable["policy_version"] == policy["policy_version"]
        assert durable["inventory_version"] == checkpoint.metadata["tool_inventory_version"]

        runtime_store = RuntimeStore(tmp_path / "runtime.db")
        await runtime_store.initialize()
        service = RuntimeService(
            engine=object(),
            store=runtime_store,
            exec_approval_store=exec_store,
            exec_runtime=ExecRuntime(),
        )
        service.update_security_config(security)
        resolved = await service.resolve_approval(
            approval_id,
            decision="approve_once",
            current_permission_policy=policy,
        )
        assert resolved is not None
        print(resolved)
        assert resolved["status"] == "consumed", resolved
        assert (workspace / "report.txt").read_text(encoding="utf-8") == "approved once\n"
        assert exec_store.get(approval_id).execution_idempotency_key == (  # type: ignore[union-attr]
            f"approval-execution:{approval_id}:{durable['operation_id']}"
        )
        with pytest.raises(ApprovalConflict):
            await service.resolve_approval(
                approval_id,
                decision="approve_once",
                current_permission_policy=policy,
            )

    asyncio.run(scenario())


def test_ordinary_chat_approval_continues_the_original_react_turn(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        security = SecurityConfig(require_approval_for_file_write=True)
        policy = EffectivePolicyResolver().resolve(security).to_dict()
        exec_store = PersistentApprovalStore(tmp_path / "exec-approvals.db")

        class _ResumeEngine:
            def __init__(self) -> None:
                self.calls: list[dict[str, object]] = []

            async def resume_ordinary_chat_approval(self, **kwargs: object) -> dict[str, object]:
                self.calls.append(dict(kwargs))
                return {"status": "continued", "content": "Saved report.txt"}

        engine = _ResumeEngine()
        context = ToolExecutionContext(
            workspace_dir=str(workspace),
            project_workspace=str(workspace),
            session_id="chat-session-1",
            permission_policy=policy,
            state={
                "ordinary_chat_approval_context": {
                    "schema_version": 1,
                    "source": "ordinary_chat",
                    "session_id": "chat-session-1",
                    "turn_id": "turn-1",
                    "resume_cursor": {
                        "turn_id": "turn-1",
                        "tool_call_id": "call-1",
                        "tool_name": "file_write",
                    },
                    "react_continuation": {
                        "schema_version": 1,
                        "messages": [],
                        "callable_tool_names": ["file_write"],
                        "generation": {},
                    },
                }
            },
        )
        tool = FileWriteTool(
            workspace_dir=workspace,
            require_approval=False,
            approval_store=exec_store,
        )
        pending = await tool.execute(
            path="report.txt",
            content="approved once\n",
            context=context,
        )
        approval_id = str(pending.metadata["approval_id"])
        runtime_store = RuntimeStore(tmp_path / "runtime.db")
        await runtime_store.initialize()
        service = RuntimeService(
            engine=engine,
            store=runtime_store,
            exec_approval_store=exec_store,
            exec_runtime=ExecRuntime(),
        )
        service.update_security_config(security)

        resolved = await service.resolve_approval(
            approval_id,
            decision="approve_once",
            current_permission_policy=policy,
        )

        assert resolved is not None
        assert resolved["status"] == "consumed"
        assert (workspace / "report.txt").read_text(encoding="utf-8") == "approved once\n"
        assert len(engine.calls) == 1
        assert engine.calls[0]["approval_id"] == approval_id
        continuation = resolved["execution_result"]["react_continuation"]
        assert continuation == {"status": "continued", "content": "Saved report.txt"}

    asyncio.run(scenario())


def test_ordinary_chat_approval_policy_drift_is_superseded_before_execution(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        original_security = SecurityConfig(require_approval_for_file_write=True)
        original_policy = EffectivePolicyResolver().resolve(original_security).to_dict()
        changed_security = SecurityConfig(require_approval_for_file_write=False)
        changed_policy = EffectivePolicyResolver().resolve(changed_security).to_dict()
        exec_store = PersistentApprovalStore(tmp_path / "exec-approvals.db")
        context = ToolExecutionContext(
            workspace_dir=str(workspace),
            project_workspace=str(workspace),
            session_id="chat-session-1",
            permission_policy=original_policy,
            state={
                "ordinary_chat_approval_context": {
                    "source": "ordinary_chat",
                    "session_id": "chat-session-1",
                    "turn_id": "turn-1",
                    "resume_cursor": {"turn_id": "turn-1", "tool_call_id": "call-1"},
                }
            },
        )
        tool = FileWriteTool(
            workspace_dir=workspace,
            require_approval=False,
            approval_store=exec_store,
        )
        pending = await tool.execute(path="report.txt", content="must not write", context=context)
        approval_id = str(pending.metadata["approval_id"])
        runtime_store = RuntimeStore(tmp_path / "runtime.db")
        await runtime_store.initialize()
        service = RuntimeService(
            engine=object(),
            store=runtime_store,
            exec_approval_store=exec_store,
            exec_runtime=ExecRuntime(),
        )
        service.update_security_config(changed_security)
        resolved = await service.resolve_approval(
            approval_id,
            decision="approve_once",
            current_permission_policy=changed_policy,
        )
        assert resolved is not None
        assert resolved["status"] == "superseded"
        assert resolved["checkpoint_status"] == "drift"
        assert resolved["checkpoint_error_code"] == "policy_drift"
        assert not (workspace / "report.txt").exists()

    asyncio.run(scenario())


def test_ordinary_chat_approval_target_drift_fails_without_replaying(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        target = workspace / "report.txt"
        security = SecurityConfig(require_approval_for_file_write=True)
        policy = EffectivePolicyResolver().resolve(security).to_dict()
        exec_store = PersistentApprovalStore(tmp_path / "exec-approvals.db")
        context = ToolExecutionContext(
            workspace_dir=str(workspace),
            project_workspace=str(workspace),
            session_id="chat-session-1",
            permission_policy=policy,
            state={
                "ordinary_chat_approval_context": {
                    "source": "ordinary_chat",
                    "session_id": "chat-session-1",
                    "turn_id": "turn-1",
                    "resume_cursor": {"turn_id": "turn-1", "tool_call_id": "call-1"},
                }
            },
        )
        tool = FileWriteTool(
            workspace_dir=workspace,
            require_approval=False,
            approval_store=exec_store,
        )
        pending = await tool.execute(path="report.txt", content="approved content\n", context=context)
        approval_id = str(pending.metadata["approval_id"])
        target.write_text("someone else changed this\n", encoding="utf-8")

        runtime_store = RuntimeStore(tmp_path / "runtime.db")
        await runtime_store.initialize()
        service = RuntimeService(
            engine=object(),
            store=runtime_store,
            exec_approval_store=exec_store,
            exec_runtime=ExecRuntime(),
        )
        service.update_security_config(security)
        resolved = await service.resolve_approval(
            approval_id,
            decision="approve_once",
            current_permission_policy=policy,
        )
        assert resolved is not None
        assert resolved["status"] == "execution_failed"
        result = resolved["execution_result"]
        assert result["error_code"] == "file_base_drift"
        assert target.read_text(encoding="utf-8") == "someone else changed this\n"

    asyncio.run(scenario())


def test_ordinary_chat_exec_approval_persists_the_exact_checkpoint(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        security = SecurityConfig(require_approval_for_exec=True)
        policy = EffectivePolicyResolver().resolve(security).to_dict()
        exec_store = PersistentApprovalStore(tmp_path / "exec-approvals.db")
        context = ToolExecutionContext(
            workspace_dir=str(workspace),
            project_workspace=str(workspace),
            session_id="chat-session-1",
            permission_policy=policy,
            state={
                "ordinary_chat_approval_context": {
                    "source": "ordinary_chat",
                    "session_id": "chat-session-1",
                    "turn_id": "turn-1",
                    "resume_cursor": {
                        "turn_id": "turn-1",
                        "tool_call_id": "call-1",
                        "tool_name": "exec_command",
                    },
                }
            },
        )
        tool = ExecCommandTool(
            runtime=ExecRuntime(),
            approval_store=exec_store,
            workspace_dir=workspace,
            require_approval=False,
        )
        pending = await tool.execute(
            command="echo durable-checkpoint",
            shell="cmd",
            context=context,
        )
        approval_id = str(pending.metadata["approval_id"])
        approval = exec_store.get(approval_id)
        assert approval is not None
        assert approval.metadata["approval_source"] == "ordinary_chat"
        assert approval.requester_id == "runtime-session:chat-session-1"
        payload = approval.command_payload
        assert payload is not None
        checkpoint = payload["ordinary_chat_checkpoint"]
        assert checkpoint["normalized_arguments"]["command"] == "echo durable-checkpoint"
        assert checkpoint["resolved_workspace_dir"] == str(workspace.resolve())
        assert checkpoint["resume_cursor"]["tool_call_id"] == "call-1"
        assert checkpoint["policy_version"] == policy["policy_version"]
        assert checkpoint["inventory_version"] == approval.metadata["tool_inventory_version"]

        runtime_store = RuntimeStore(tmp_path / "runtime.db")
        await runtime_store.initialize()
        service = RuntimeService(
            engine=object(),
            store=runtime_store,
            exec_approval_store=exec_store,
            exec_runtime=ExecRuntime(),
        )
        service.update_security_config(security)
        resolved = await service.resolve_approval(
            approval_id,
            decision="approve_once",
            current_permission_policy=policy,
        )
        assert resolved is not None
        assert resolved["status"] == "consumed", resolved
        assert resolved["execution_result"]["status"] == "completed"
        with pytest.raises(ApprovalConflict):
            await service.resolve_approval(
                approval_id,
                decision="approve_once",
                current_permission_policy=policy,
            )

    asyncio.run(scenario())


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


def test_applied_ordinary_chat_result_recovers_without_replaying_mutation(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        security = SecurityConfig(require_approval_for_file_write=True)
        policy = EffectivePolicyResolver().resolve(security).to_dict()
        approvals = PersistentApprovalStore(tmp_path / "approvals.db")

        class ResumeEngine:
            def __init__(self, started: asyncio.Event | None = None, release: asyncio.Event | None = None) -> None:
                self.calls = 0
                self._started = started
                self._release = release

            async def resume_ordinary_chat_approval(self, **_: object) -> dict[str, str]:
                self.calls += 1
                if self._started is not None and self._release is not None:
                    self._started.set()
                    await self._release.wait()
                return {"status": "continued"}

        context = ToolExecutionContext(
            workspace_dir=str(workspace), project_workspace=str(workspace),
            session_id="recovery-session", permission_policy=policy,
            state={"ordinary_chat_approval_context": {
                "schema_version": 1, "source": "ordinary_chat",
                "session_id": "recovery-session", "turn_id": "turn-recovery",
                "resume_cursor": {"turn_id": "turn-recovery", "tool_call_id": "call-recovery", "tool_name": "file_write"},
                "react_continuation": {"schema_version": 1, "messages": [], "callable_tool_names": ["file_write"], "generation": {}},
            }},
        )
        pending = await FileWriteTool(workspace_dir=workspace, require_approval=False, approval_store=approvals).execute(
            path="report.txt", content="written once\n", context=context,
        )
        approval_id = str(pending.metadata["approval_id"])
        approval = approvals.get(approval_id)
        assert approval is not None
        approvals.resolve(approval_id, decision="approve_once")
        operation_id = str(approval.metadata["operation_id"])
        execution_key = f"approval-execution:{approval_id}:{operation_id}"
        claim = approvals.consume(
            approval_id, execution_idempotency_key=execution_key, lease_owner="crashed-worker",
        )
        service = RuntimeService(engine=ResumeEngine(), store=RuntimeStore(tmp_path / "runtime.db"), exec_approval_store=approvals, exec_runtime=ExecRuntime())
        service.update_security_config(security)
        result = await service._execute_approved_standalone_request(claim, current_permission_policy=policy)
        approvals.record_execution_result(approval_id, execution_result=result)
        assert (workspace / "report.txt").read_text(encoding="utf-8") == "written once\n"
        with sqlite3.connect(tmp_path / "approvals.db") as conn:
            conn.execute("UPDATE exec_approval_requests SET consume_lease_expires_at=? WHERE approval_id=?", (_past(), approval_id))
            conn.commit()

        continuation_started = asyncio.Event()
        release_continuation = asyncio.Event()
        engine = ResumeEngine(continuation_started, release_continuation)
        recovered_service = RuntimeService(engine=engine, store=RuntimeStore(tmp_path / "runtime-recovered.db"), exec_approval_store=PersistentApprovalStore(tmp_path / "approvals.db"), exec_runtime=ExecRuntime())
        recovered_service.update_security_config(security)
        competing_engine = ResumeEngine()
        competing_service = RuntimeService(engine=competing_engine, store=RuntimeStore(tmp_path / "runtime-competing.db"), exec_approval_store=PersistentApprovalStore(tmp_path / "approvals.db"), exec_runtime=ExecRuntime())
        competing_service.update_security_config(security)
        await asyncio.gather(
            recovered_service._recover_stale_approval_consumptions(),
            competing_service._recover_stale_approval_consumptions(),
        )
        recovered = recovered_service._exec_approval_store.get(approval_id)
        assert recovered is not None and recovered.status == "consumed"
        assert recovered.execution_result is not None
        assert recovered.execution_result["recovery_required"] == "ordinary_chat_continuation"
        assert engine.calls == 0
        first_task = asyncio.create_task(
            recovered_service._reconcile_recovered_ordinary_chat_approval_with_policy(
                approval_id=approval_id,
                current_permission_policy=policy,
                enforce_checkpoint_preflight=False,
            )
        )
        await asyncio.wait_for(continuation_started.wait(), timeout=1)
        second = await competing_service._reconcile_recovered_ordinary_chat_approval_with_policy(
            approval_id=approval_id,
            current_permission_policy=policy,
            enforce_checkpoint_preflight=False,
        )
        assert second["status"] == "not_available"
        assert competing_engine.calls == 0
        release_continuation.set()
        first = await first_task
        assert first["status"] == "continued"
        assert engine.calls == 1
        assert (workspace / "report.txt").read_text(encoding="utf-8") == "written once\n"

    asyncio.run(scenario())


def test_continuation_reconciliation_expired_lease_becomes_unknown(tmp_path: Path) -> None:
    store = PersistentApprovalStore(tmp_path / "approvals.db")
    _create_exec(store)
    store.resolve("approval-1", decision="approve_once")
    store.consume("approval-1", execution_idempotency_key="execution-1", lease_owner="worker")
    store.complete_consumption(
        "approval-1", execution_idempotency_key="execution-1", lease_owner="worker",
        lease_token=store.get("approval-1").consume_lease_token or "",  # type: ignore[union-attr]
        execution_result={"status": "completed", "recovery_required": "ordinary_chat_continuation"},
    )
    first = store.claim_continuation_reconciliation("approval-1", lease_owner="first", lease_seconds=1)
    with sqlite3.connect(tmp_path / "approvals.db") as conn:
        conn.execute("UPDATE approval_continuation_reconciliations SET lease_expires_at=? WHERE approval_id=?", (_past(), "approval-1"))
        conn.commit()
    with pytest.raises(ApprovalConflict, match="unknown outcome"):
        store.claim_continuation_reconciliation("approval-1", lease_owner="second")
    with pytest.raises(ApprovalConflict, match="terminal: unknown"):
        store.claim_continuation_reconciliation("approval-1", lease_owner="third")
    with pytest.raises(ApprovalConflict):
        store.complete_continuation_reconciliation(
            "approval-1", lease_token=first.lease_token, execution_result={"status": "completed"},
        )


def _create_recoverable_ordinary_chat_approval(
    tmp_path: Path,
) -> tuple[PersistentApprovalStore, str]:
    store = PersistentApprovalStore(tmp_path / "approvals.db")
    approval = store.create(
        approval_id="ordinary-chat-recovery-1",
        command="write report.txt",
        shell="test",
        scope="file_write",
        metadata={"approval_source": "ordinary_chat"},
        command_payload={"tool_name": "file_write"},
    )
    store.resolve(approval.approval_id, decision="approve_once")
    consuming = store.consume(
        approval.approval_id,
        execution_idempotency_key="ordinary-chat-recovery-execution-1",
        lease_owner="recovery-worker",
    )
    store.complete_consumption(
        approval.approval_id,
        execution_idempotency_key="ordinary-chat-recovery-execution-1",
        lease_owner="recovery-worker",
        lease_token=consuming.consume_lease_token or "",
        execution_result={
            "status": "completed",
            "recovery_required": "ordinary_chat_continuation",
        },
    )
    return store, approval.approval_id


async def _create_dispatchable_ordinary_chat_recovery(
    tmp_path: Path,
    *,
    session_id: str = "dispatcher-session",
    security_override: dict[str, object] | None = None,
) -> tuple[PersistentApprovalStore, str, MochiConfig, SessionStore]:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    sessions_dir = tmp_path / "sessions"
    security = SecurityConfig(require_approval_for_file_write=True)
    config = MochiConfig.model_validate(
        {
            "sessions_dir": str(sessions_dir),
            "security": security.model_dump(mode="python"),
        }
    )
    sessions = SessionStore(sessions_dir)
    event: dict[str, object] = {
        "type": "session_meta",
        "event": "created",
        "session_id": session_id,
    }
    if security_override is not None:
        event["security_override"] = security_override
    await sessions.save_event(session_id, event)
    approval_policy = EffectivePolicyResolver().resolve(security).to_dict()
    policy = EffectivePolicyResolver().resolve(
        security,
        session_overrides=security_override,
    ).to_dict()
    approvals = PersistentApprovalStore(tmp_path / "approvals.db")
    context = ToolExecutionContext(
        workspace_dir=str(workspace),
        project_workspace=str(workspace),
        session_id=session_id,
        permission_policy=approval_policy,
        state={
            "ordinary_chat_approval_context": {
                "schema_version": 1,
                "source": "ordinary_chat",
                "session_id": session_id,
                "turn_id": "dispatcher-turn",
                "resume_cursor": {
                    "turn_id": "dispatcher-turn",
                    "tool_call_id": "dispatcher-call",
                    "tool_name": "file_write",
                },
                "react_continuation": {
                    "schema_version": 1,
                    "messages": [],
                    "callable_tool_names": ["file_write"],
                    "generation": {},
                },
            }
        },
    )
    pending = await FileWriteTool(
        workspace_dir=workspace,
        require_approval=True,
        approval_store=approvals,
    ).execute(path="report.txt", content="must not be replayed\n", context=context)
    approval_id = str(pending.metadata["approval_id"])
    approval = approvals.get(approval_id)
    assert approval is not None and isinstance(approval.command_payload, dict)
    payload = dict(approval.command_payload)
    checkpoint = payload.get("ordinary_chat_checkpoint")
    assert isinstance(checkpoint, dict)
    checkpoint = dict(checkpoint)
    checkpoint["policy_snapshot_id"] = policy["policy_snapshot_id"]
    checkpoint["policy_version"] = policy["policy_version"]
    payload["ordinary_chat_checkpoint"] = checkpoint
    payload["permission_policy"] = policy
    with sqlite3.connect(tmp_path / "approvals.db") as conn:
        conn.execute(
            "UPDATE exec_approval_requests SET command_payload_json=? WHERE approval_id=?",
            (json.dumps(payload), approval_id),
        )
        conn.commit()
    approvals.resolve(approval_id, decision="approve_once")
    approval = approvals.get(approval_id)
    assert approval is not None
    operation_id = str(approval.metadata["operation_id"])
    claim = approvals.consume(
        approval_id,
        execution_idempotency_key=f"approval-execution:{approval_id}:{operation_id}",
        lease_owner="crashed-worker",
    )
    approvals.complete_consumption(
        approval_id,
        execution_idempotency_key=f"approval-execution:{approval_id}:{operation_id}",
        lease_owner="crashed-worker",
        lease_token=claim.consume_lease_token or "",
        execution_result={
            "status": "succeeded",
            "recovery_required": "ordinary_chat_continuation",
        },
    )
    return approvals, approval_id, config, sessions


class _DispatcherResumeEngine:
    def __init__(
        self,
        *,
        started: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
        cancel: bool = False,
    ) -> None:
        self.calls = 0
        self.policies: list[dict[str, object]] = []
        self._started = started
        self._release = release
        self._cancel = cancel

    async def resume_ordinary_chat_approval(self, **kwargs: object) -> dict[str, str]:
        self.calls += 1
        policy = kwargs.get("current_permission_policy")
        self.policies.append(dict(policy) if isinstance(policy, dict) else {})
        if self._started is not None:
            self._started.set()
        if self._release is not None:
            await self._release.wait()
        if self._cancel:
            raise asyncio.CancelledError()
        return {"status": "continued"}


async def _dispatcher_service(
    tmp_path: Path,
    *,
    approvals: PersistentApprovalStore,
    config: MochiConfig,
    engine: object,
    ordinary_chat_session_store: SessionStore | None = None,
) -> RuntimeService:
    runtime_store = RuntimeStore(tmp_path / "runtime.db")
    await runtime_store.initialize()
    service = RuntimeService(
        engine=engine,
        store=runtime_store,
        exec_approval_store=approvals,
        exec_runtime=ExecRuntime(),
        ordinary_chat_session_store=ordinary_chat_session_store,
    )
    service.bind_app_config(config=config, config_path=None)
    return service


def _continuation_status(db_path: Path, approval_id: str) -> str | None:
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT status FROM approval_continuation_reconciliations WHERE approval_id=?",
            (approval_id,),
        ).fetchone()
    return str(row[0]) if row is not None else None


def test_ordinary_chat_public_reconcile_reads_one_strict_session_snapshot(
    tmp_path: Path,
) -> None:
    class CountingSessionStore(SessionStore):
        def __init__(self, sessions_dir: Path) -> None:
            super().__init__(sessions_dir)
            self.strict_reads = 0

        async def load_strict_snapshot(self, session_id: str):  # type: ignore[no-untyped-def]
            self.strict_reads += 1
            return await super().load_strict_snapshot(session_id)

    async def scenario() -> None:
        approvals, approval_id, config, _ = await _create_dispatchable_ordinary_chat_recovery(tmp_path)
        sessions = CountingSessionStore(Path(config.sessions_dir))
        engine = _DispatcherResumeEngine()
        service = await _dispatcher_service(
            tmp_path,
            approvals=approvals,
            config=config,
            engine=engine,
            ordinary_chat_session_store=sessions,
        )
        outcome = await service.reconcile_recovered_ordinary_chat_approval(
            approval_id=approval_id,
        )
        assert outcome["status"] == "continued"
        assert engine.calls == 1
        assert sessions.strict_reads == 1

    asyncio.run(scenario())


def test_ordinary_chat_dispatcher_migrates_legacy_candidate_to_versioned_pending(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        unrelated = PersistentApprovalStore(tmp_path / "approvals.db")
        for index in range(20):
            approval_id = f"unrelated-consumed-{index}"
            unrelated.create(
                approval_id=approval_id,
                command="echo unrelated",
                shell="powershell",
                scope="dangerous_command",
            )
            unrelated.resolve(approval_id, decision="approve_once")
            claim = unrelated.consume(
                approval_id,
                execution_idempotency_key=f"unrelated-execution-{index}",
                lease_owner="worker",
            )
            unrelated.complete_consumption(
                approval_id,
                execution_idempotency_key=f"unrelated-execution-{index}",
                lease_owner="worker",
                lease_token=claim.consume_lease_token or "",
                execution_result={
                    "status": "completed",
                    "note": "ordinary_chat_continuation",
                },
            )
        approvals, approval_id, _, _ = await _create_dispatchable_ordinary_chat_recovery(tmp_path)
        candidates = approvals.list_continuation_reconciliation_candidates(limit=4)
        assert [candidate.approval.approval_id for candidate in candidates] == [approval_id]
        assert candidates[0].status == "pending"
        assert candidates[0].schema_version == 1
        assert _continuation_status(tmp_path / "approvals.db", approval_id) == "pending"

    asyncio.run(scenario())


def test_ordinary_chat_dispatcher_startup_continues_once_without_mutation_executor(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        approvals, approval_id, config, _ = await _create_dispatchable_ordinary_chat_recovery(tmp_path)
        engine = _DispatcherResumeEngine()
        service = await _dispatcher_service(
            tmp_path,
            approvals=approvals,
            config=config,
            engine=engine,
        )
        service.set_scheduler_poll_interval(0.05)
        executions = 0

        async def unexpected_mutation_executor(**_: object) -> dict[str, object]:
            nonlocal executions
            executions += 1
            raise AssertionError("automatic continuation must not execute the approved mutation")

        setattr(service, "_execute_approved_standalone_request", unexpected_mutation_executor)
        await service.start()
        for _ in range(40):
            if engine.calls == 1:
                break
            await asyncio.sleep(0.025)
        await asyncio.sleep(0.15)
        await service.close()
        assert engine.calls == 1
        assert executions == 0
        assert _continuation_status(tmp_path / "approvals.db", approval_id) == "continued"

        approvals, approval_id, config, sessions = await _create_dispatchable_ordinary_chat_recovery(
            tmp_path / "policy-drift",
            security_override={"autonomy_mode": "high_autonomy"},
        )
        await sessions.save_event(
            "dispatcher-session",
            {
                "type": "session_meta",
                "event": "security_override_updated",
                "session_id": "dispatcher-session",
                "security_override": {"autonomy_mode": "strict"},
            },
        )
        drift_engine = _DispatcherResumeEngine()
        drift_service = await _dispatcher_service(
            tmp_path / "policy-drift",
            approvals=approvals,
            config=config,
            engine=drift_engine,
        )
        await drift_service.dispatch_ordinary_chat_approval_recovery_once()
        assert drift_engine.calls == 0
        assert _continuation_status(tmp_path / "policy-drift" / "approvals.db", approval_id) == "unknown"

        approvals, approval_id, config, sessions = await _create_dispatchable_ordinary_chat_recovery(
            tmp_path / "mismatched-session",
        )
        await sessions.replace_session(
            "dispatcher-session",
            [{"type": "session_meta", "event": "created", "session_id": "other-session"}],
        )
        mismatched_engine = _DispatcherResumeEngine()
        mismatched_service = await _dispatcher_service(
            tmp_path / "mismatched-session",
            approvals=approvals,
            config=config,
            engine=mismatched_engine,
        )
        await mismatched_service.dispatch_ordinary_chat_approval_recovery_once()
        assert mismatched_engine.calls == 0
        assert _continuation_status(tmp_path / "mismatched-session" / "approvals.db", approval_id) == "unknown"

        approvals, approval_id, config, sessions = await _create_dispatchable_ordinary_chat_recovery(
            tmp_path / "malformed-override",
        )
        await sessions.save_event(
            "dispatcher-session",
            {
                "type": "session_meta",
                "event": "security_override_updated",
                "session_id": "dispatcher-session",
                "security_override": {"autonomy_mode": "invalid-mode"},
            },
        )
        malformed_engine = _DispatcherResumeEngine()
        malformed_service = await _dispatcher_service(
            tmp_path / "malformed-override",
            approvals=approvals,
            config=config,
            engine=malformed_engine,
        )
        await malformed_service.dispatch_ordinary_chat_approval_recovery_once()
        assert malformed_engine.calls == 0
        assert _continuation_status(tmp_path / "malformed-override" / "approvals.db", approval_id) == "unknown"

    asyncio.run(scenario())


def test_ordinary_chat_dispatcher_invalid_exact_checkpoint_is_terminal_unknown(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        approvals, approval_id, _, _ = await _create_dispatchable_ordinary_chat_recovery(tmp_path)
        approval = approvals.get(approval_id)
        assert approval is not None and isinstance(approval.command_payload, dict)
        payload = dict(approval.command_payload)
        checkpoint = dict(payload["ordinary_chat_checkpoint"])
        checkpoint["operation_id"] = "different-operation"
        payload["ordinary_chat_checkpoint"] = checkpoint
        with sqlite3.connect(tmp_path / "approvals.db") as conn:
            conn.execute(
                "UPDATE exec_approval_requests SET command_payload_json=? WHERE approval_id=?",
                (json.dumps(payload), approval_id),
            )
            conn.commit()
        assert approvals.list_continuation_reconciliation_candidates(limit=4) == []
        assert _continuation_status(tmp_path / "approvals.db", approval_id) == "unknown"

    asyncio.run(scenario())


def test_ordinary_chat_dispatcher_two_services_claim_one_continuation(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        approvals, approval_id, config, _ = await _create_dispatchable_ordinary_chat_recovery(tmp_path)
        started = asyncio.Event()
        release = asyncio.Event()
        first_engine = _DispatcherResumeEngine(started=started, release=release)
        second_engine = _DispatcherResumeEngine()
        first = await _dispatcher_service(
            tmp_path / "first",
            approvals=PersistentApprovalStore(tmp_path / "approvals.db"),
            config=config,
            engine=first_engine,
        )
        second = await _dispatcher_service(
            tmp_path / "second",
            approvals=PersistentApprovalStore(tmp_path / "approvals.db"),
            config=config,
            engine=second_engine,
        )
        first_task = asyncio.create_task(first.dispatch_ordinary_chat_approval_recovery_once())
        await asyncio.wait_for(started.wait(), timeout=1)
        second_outcomes = await second.dispatch_ordinary_chat_approval_recovery_once()
        release.set()
        await first_task
        assert first_engine.calls == 1
        assert second_engine.calls == 0
        assert second_outcomes == []
        assert _continuation_status(tmp_path / "approvals.db", approval_id) == "continued"

    asyncio.run(scenario())


def test_ordinary_chat_dispatcher_uses_server_policy_and_strict_session_gate(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        approvals, approval_id, config, sessions = await _create_dispatchable_ordinary_chat_recovery(
            tmp_path,
            security_override={"autonomy_mode": "high_autonomy"},
        )
        engine = _DispatcherResumeEngine()
        service = await _dispatcher_service(
            tmp_path,
            approvals=approvals,
            config=config,
            engine=engine,
        )
        await service.dispatch_ordinary_chat_approval_recovery_once()
        assert engine.calls == 1
        assert engine.policies[0]["autonomy_mode"] == "high_autonomy"
        assert _continuation_status(tmp_path / "approvals.db", approval_id) == "continued"

        approvals, approval_id, config, sessions = await _create_dispatchable_ordinary_chat_recovery(
            tmp_path / "corrupt",
        )
        session_path = sessions._session_path("dispatcher-session")  # noqa: SLF001
        session_path.write_text("{not-json}\n", encoding="utf-8")
        corrupt_engine = _DispatcherResumeEngine()
        corrupt_service = await _dispatcher_service(
            tmp_path / "corrupt",
            approvals=approvals,
            config=config,
            engine=corrupt_engine,
        )
        await corrupt_service.dispatch_ordinary_chat_approval_recovery_once()
        assert corrupt_engine.calls == 0
        assert _continuation_status(tmp_path / "corrupt" / "approvals.db", approval_id) == "unknown"

    asyncio.run(scenario())


def test_ordinary_chat_dispatcher_cancellation_and_stale_lease_are_terminal_unknown(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        approvals, approval_id, config, _ = await _create_dispatchable_ordinary_chat_recovery(tmp_path)
        cancelling = _DispatcherResumeEngine(cancel=True)
        service = await _dispatcher_service(
            tmp_path,
            approvals=approvals,
            config=config,
            engine=cancelling,
        )
        with pytest.raises(asyncio.CancelledError):
            await service.dispatch_ordinary_chat_approval_recovery_once()
        assert _continuation_status(tmp_path / "approvals.db", approval_id) == "unknown"

        approvals, approval_id, config, _ = await _create_dispatchable_ordinary_chat_recovery(
            tmp_path / "stale",
        )
        approvals.list_continuation_reconciliation_candidates(limit=1)
        claim = approvals.claim_continuation_reconciliation(approval_id, lease_owner="dead", lease_seconds=1)
        with sqlite3.connect(tmp_path / "stale" / "approvals.db") as conn:
            conn.execute(
                "UPDATE approval_continuation_reconciliations SET lease_expires_at=? WHERE approval_id=?",
                (_past(), approval_id),
            )
            conn.commit()
        stale_engine = _DispatcherResumeEngine()
        stale_service = await _dispatcher_service(
            tmp_path / "stale",
            approvals=approvals,
            config=config,
            engine=stale_engine,
        )
        await stale_service.dispatch_ordinary_chat_approval_recovery_once()
        assert claim.lease_token
        assert stale_engine.calls == 0
        assert _continuation_status(tmp_path / "stale" / "approvals.db", approval_id) == "unknown"

    asyncio.run(scenario())


def test_reconciliation_cancellation_becomes_terminal_unknown_before_reraising(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        approvals, approval_id = _create_recoverable_ordinary_chat_approval(tmp_path)
        service = RuntimeService(
            engine=object(),
            store=RuntimeStore(tmp_path / "runtime.db"),
            exec_approval_store=approvals,
            exec_runtime=ExecRuntime(),
        )
        calls = 0

        async def cancelled_resume(**_: object) -> dict[str, str]:
            nonlocal calls
            calls += 1
            raise asyncio.CancelledError()

        setattr(
            service,
            "_resume_ordinary_chat_approval_react_loop",
            cancelled_resume,
        )

        with pytest.raises(asyncio.CancelledError):
            await service._reconcile_recovered_ordinary_chat_approval_with_policy(
                approval_id=approval_id,
                current_permission_policy={},
                enforce_checkpoint_preflight=False,
            )

        assert calls == 1
        with pytest.raises(ApprovalConflict, match="terminal: unknown"):
            approvals.claim_continuation_reconciliation(
                approval_id,
                lease_owner="retry-worker",
            )
        retry = await service._reconcile_recovered_ordinary_chat_approval_with_policy(
            approval_id=approval_id,
            current_permission_policy={},
            enforce_checkpoint_preflight=False,
        )
        assert retry["status"] == "not_available"
        assert calls == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("resume_outcome", "terminal_method"),
    [
        ({"status": "continued"}, "complete_continuation_reconciliation"),
        ({"status": "not_available"}, "fail_continuation_reconciliation"),
    ],
)
def test_reconciliation_terminal_cas_loss_returns_unknown(
    tmp_path: Path,
    resume_outcome: dict[str, str],
    terminal_method: str,
) -> None:
    async def scenario() -> None:
        approvals, approval_id = _create_recoverable_ordinary_chat_approval(
            tmp_path / terminal_method
        )
        service = RuntimeService(
            engine=object(),
            store=RuntimeStore(tmp_path / terminal_method / "runtime.db"),
            exec_approval_store=approvals,
            exec_runtime=ExecRuntime(),
        )
        calls = 0

        async def resumed(**_: object) -> dict[str, str]:
            nonlocal calls
            calls += 1
            with sqlite3.connect(tmp_path / terminal_method / "approvals.db") as conn:
                cursor = conn.execute(
                    "UPDATE approval_continuation_reconciliations "
                    "SET status='unknown',lease_owner=NULL,lease_token=NULL,"
                    "lease_expires_at=NULL,reason=? "
                    "WHERE approval_id=? AND status='reconciling'",
                    ("simulated competing reconciliation", approval_id),
                )
                assert cursor.rowcount == 1
                conn.commit()
            return dict(resume_outcome)

        setattr(service, "_resume_ordinary_chat_approval_react_loop", resumed)

        response = await service._reconcile_recovered_ordinary_chat_approval_with_policy(
            approval_id=approval_id,
            current_permission_policy={},
            enforce_checkpoint_preflight=False,
        )

        assert response == {
            "status": "not_available",
            "reason": "continuation_unknown_outcome",
        }
        retry = await service._reconcile_recovered_ordinary_chat_approval_with_policy(
            approval_id=approval_id,
            current_permission_policy={},
            enforce_checkpoint_preflight=False,
        )
        assert retry["status"] == "not_available"
        assert calls == 1

    asyncio.run(scenario())


def test_normal_continuation_interruption_recovers_as_unknown_without_replay(tmp_path: Path) -> None:
    async def scenario() -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        security = SecurityConfig(require_approval_for_file_write=True)
        policy = EffectivePolicyResolver().resolve(security).to_dict()
        approvals = PersistentApprovalStore(tmp_path / "approvals.db")

        class CrashEngine:
            def __init__(self) -> None:
                self.calls = 0

            async def resume_ordinary_chat_approval(self, **_: object) -> dict[str, str]:
                self.calls += 1
                (workspace / "continuation-side-effect.txt").write_text("started\n", encoding="utf-8")
                raise asyncio.CancelledError()

        context = ToolExecutionContext(
            workspace_dir=str(workspace), project_workspace=str(workspace),
            session_id="seam-session", permission_policy=policy,
            state={"ordinary_chat_approval_context": {
                "schema_version": 1, "source": "ordinary_chat", "session_id": "seam-session", "turn_id": "seam-turn",
                "resume_cursor": {"turn_id": "seam-turn", "tool_call_id": "seam-call", "tool_name": "file_write"},
                "react_continuation": {"schema_version": 1, "messages": [], "callable_tool_names": ["file_write"], "generation": {}},
            }},
        )
        pending = await FileWriteTool(workspace_dir=workspace, require_approval=False, approval_store=approvals).execute(
            path="report.txt", content="exact mutation\n", context=context,
        )
        approval_id = str(pending.metadata["approval_id"])
        service = RuntimeService(engine=CrashEngine(), store=RuntimeStore(tmp_path / "runtime.db"), exec_approval_store=approvals, exec_runtime=ExecRuntime())
        service.update_security_config(security)
        with pytest.raises(asyncio.CancelledError):
            await service.resolve_approval(approval_id, decision="approve_once", current_permission_policy=policy)
        interrupted = approvals.get(approval_id)
        assert interrupted is not None and interrupted.status == "consuming"
        assert interrupted.execution_result is not None
        assert interrupted.execution_result["continuation_started"] is True
        assert (workspace / "continuation-side-effect.txt").exists()
        with sqlite3.connect(tmp_path / "approvals.db") as conn:
            conn.execute("UPDATE exec_approval_requests SET consume_lease_expires_at=? WHERE approval_id=?", (_past(), approval_id))
            conn.commit()

        class SafeEngine:
            def __init__(self) -> None:
                self.calls = 0

            async def resume_ordinary_chat_approval(self, **_: object) -> dict[str, str]:
                self.calls += 1
                return {"status": "continued"}

        safe = SafeEngine()
        restarted = RuntimeService(engine=safe, store=RuntimeStore(tmp_path / "restarted.db"), exec_approval_store=PersistentApprovalStore(tmp_path / "approvals.db"), exec_runtime=ExecRuntime())
        restarted.update_security_config(security)
        await restarted._recover_stale_approval_consumptions()
        recovered = restarted._exec_approval_store.get(approval_id)
        assert recovered is not None and recovered.status == "consumed"
        assert recovered.execution_result is not None
        assert recovered.execution_result["continuation_outcome"] == "unknown"
        assert "recovery_required" not in recovered.execution_result
        response = await restarted._reconcile_recovered_ordinary_chat_approval_with_policy(
            approval_id=approval_id,
            current_permission_policy=policy,
            enforce_checkpoint_preflight=False,
        )
        assert response["status"] == "not_available"
        assert safe.calls == 0

    asyncio.run(scenario())
