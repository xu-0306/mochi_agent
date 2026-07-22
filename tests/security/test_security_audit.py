from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr

from mochi.config.manager import load_config_snapshot
from mochi.config.schema import MochiConfig
from mochi.runtime.approvals import InMemoryApprovalStore, PersistentApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.execution_transcript import normalize_subagent_event
from mochi.runtime.security_audit import (
    KnownSecretRegistry,
    SecurityAuditEvent,
    file_content_observation,
    known_secrets,
    redact_for_persistence,
    register_known_secrets,
)
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore


def test_recursive_redactor_replaces_known_values_and_sensitive_keys() -> None:
    registry = KnownSecretRegistry()
    registry.register("known-secret-marker")

    redacted = redact_for_persistence(
        {
            "tool_args": {"query": "prefix known-secret-marker suffix"},
            "environment": {"SAFE": "known-secret-marker"},
            "nested": [{"api_key": "known-secret-marker"}],
        },
        registry=registry,
    )

    rendered = json.dumps(redacted, ensure_ascii=False)
    assert "known-secret-marker" not in rendered
    assert rendered.count("[REDACTED]") >= 2


def test_loading_config_registers_secret_values_for_general_transcripts(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "config.yaml"
    secret = "registered-config-secret"
    config_path.write_text(
        f"openai_compat:\n  api_key: {secret}\n",
        encoding="utf-8",
    )

    try:
        snapshot = load_config_snapshot(config_path)
        assert isinstance(snapshot.config, MochiConfig)
        payload = normalize_subagent_event(
            {
                "type": "subagent_progress",
                "content": f"tool output included {secret}",
            },
            parent_type="task",
            parent_id="task-secret-registry",
        )
    finally:
        known_secrets.discard(secret)

    assert secret not in json.dumps(payload, ensure_ascii=False)


def test_registering_config_object_secret_redacts_nested_tool_arguments() -> None:
    secret = "object-config-secret"
    config = MochiConfig()
    config.openai_compat.api_key = SecretStr(secret)
    try:
        register_known_secrets(config)
        redacted = redact_for_persistence(
            {"arguments": {"query": f"prefix {secret} suffix"}}
        )
    finally:
        known_secrets.discard(secret)

    assert secret not in json.dumps(redacted, ensure_ascii=False)


def test_file_content_observation_records_digest_without_body() -> None:
    observation = file_content_observation(b"authoritative-secret-bytes", reason_code="mutation")

    assert observation["byte_count"] == len(b"authoritative-secret-bytes")
    assert len(observation["sha256"]) == 64
    assert "authoritative-secret-bytes" not in json.dumps(observation)


def test_execution_transcript_redacts_known_secret_in_general_content() -> None:
    known_secrets.register("transcript-known-secret")
    try:
        payload = normalize_subagent_event(
            {
                "type": "subagent_progress",
                "content": "output transcript-known-secret",
                "metadata": {"note": "transcript-known-secret"},
            },
            parent_type="task",
            parent_id="task-1",
        )
    finally:
        known_secrets.discard("transcript-known-secret")

    assert "transcript-known-secret" not in json.dumps(payload, ensure_ascii=False)


@pytest.mark.asyncio
async def test_runtime_security_audit_redacts_before_sqlite_persistence(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "runtime.db")
    await store.initialize()
    known_secrets.register("audit-known-secret")
    try:
        persisted = await store.append_security_audit_event(
            SecurityAuditEvent(
                event_type="approval_resolved",
                subject_type="approval",
                subject_id="approval-1",
                request_digest="a" * 64,
                outcome="approved_once",
                details={
                    "arguments": {"command": "echo audit-known-secret"},
                    "stdout": "audit-known-secret",
                },
            )
        )
    finally:
        known_secrets.discard("audit-known-secret")

    assert "audit-known-secret" not in json.dumps(persisted, ensure_ascii=False)
    with sqlite3.connect(tmp_path / "runtime.db") as conn:
        stored = conn.execute(
            "SELECT details_json FROM security_audit_events WHERE id=?",
            (persisted["id"],),
        ).fetchone()[0]
    assert "audit-known-secret" not in stored
    assert json.loads(stored)["stdout"] == "[REDACTED]"


def test_security_audit_event_rejects_unknown_event_type() -> None:
    with pytest.raises(ValueError, match="Unsupported security audit event type"):
        SecurityAuditEvent(  # type: ignore[arg-type]
            event_type="arbitrary_event",
            subject_type="approval",
        )


@pytest.mark.asyncio
async def test_runtime_security_audit_redacts_top_level_projection_fields(
    tmp_path: Path,
) -> None:
    store = RuntimeStore(tmp_path / "runtime.db")
    secret = "audit-top-level-secret"
    known_secrets.register(secret)
    try:
        persisted = await store.append_security_audit_event(
            SecurityAuditEvent(
                event_type="review_decided",
                subject_type="approval",
                subject_id=secret,
                outcome=secret,
            )
        )
    finally:
        known_secrets.discard(secret)

    assert secret not in json.dumps(persisted, ensure_ascii=False)


@pytest.mark.asyncio
async def test_security_audit_retention_prunes_expired_and_overflow_rows(
    tmp_path: Path,
) -> None:
    store = RuntimeStore(tmp_path / "runtime.db")
    for index in range(3):
        await store.append_security_audit_event(
            SecurityAuditEvent(
                event_type="review_decided",
                subject_type="approval",
                subject_id=f"approval-{index}",
                outcome="allow",
            )
        )
    with sqlite3.connect(tmp_path / "runtime.db") as conn:
        conn.execute(
            "UPDATE security_audit_events SET created_at=? WHERE subject_id=?",
            (
                (datetime.now(UTC) - timedelta(days=120)).isoformat(),
                "approval-0",
            ),
        )
        conn.commit()

    removed = await store.prune_security_audit_events(
        retention_days=30,
        max_events=1,
    )
    remaining = await store.list_security_audit_events()

    assert removed == 2
    assert [event["subject_id"] for event in remaining] == ["approval-2"]


@pytest.mark.asyncio
async def test_approval_api_projection_redacts_arguments_metadata_and_execution_output(
    tmp_path: Path,
) -> None:
    secret = "approval-api-known-secret"
    approval_store = InMemoryApprovalStore()
    approval_store.create(
        approval_id="approval-secret",
        command=f"echo {secret}",
        shell="powershell",
        scope="dangerous_command",
        metadata={"suggested_rule": {"tokens": [secret]}},
        command_payload={"command": f"echo {secret}", "env": {"SECRET": secret}},
    )
    approval_store.resolve("approval-secret", decision="approve_once")
    claim = approval_store.consume(
        "approval-secret",
        execution_idempotency_key="execution-secret",
        lease_owner="worker",
    )
    known_secrets.register(secret)
    try:
        approval_store.complete_consumption(
            "approval-secret",
            execution_idempotency_key="execution-secret",
            lease_owner="worker",
            lease_token=claim.consume_lease_token or "",
            execution_result={"stdout": secret, "stderr": secret},
        )
        service = RuntimeService(
            engine=object(),
            store=RuntimeStore(tmp_path / "runtime.db"),
            exec_approval_store=approval_store,
            exec_runtime=ExecRuntime(),
        )
        payload = await service.list_approvals()
    finally:
        known_secrets.discard(secret)

    assert secret not in json.dumps(payload, ensure_ascii=False)


@pytest.mark.asyncio
async def test_file_approval_api_projection_uses_body_digest_not_raw_content(
    tmp_path: Path,
) -> None:
    store = RuntimeStore(tmp_path / "runtime.db")
    await store.create_task_run(
        task_id="task-file",
        input_text="write a file",
        session_id="session-file",
        project_id=None,
        workspace_dir=str(tmp_path),
        project_workspace_dir=None,
        task_workspace_dir=None,
        inference_overrides={},
    )
    body = "raw-file-body-that-must-not-be-projected"
    await store.create_approval_request(
        approval_id="approval-file",
        task_id="task-file",
        call_id="call-file",
        tool_name="file_write",
        arguments={"path": "notes.txt", "content": body},
        metadata={
            "security_decision": "require_approval",
            "approval_kind": "file_write",
            "approval_scope": "workspace",
        },
        requester_id="runtime-task:task-file",
        request_digest="a" * 64,
        context_digest="b" * 64,
        expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
    )
    service = RuntimeService(
        engine=object(),
        store=store,
        exec_approval_store=InMemoryApprovalStore(),
        exec_runtime=ExecRuntime(),
    )

    payload = await service.list_approvals()
    rendered = json.dumps(payload, ensure_ascii=False)

    assert body not in rendered
    observations = payload[0]["arguments"]["body_observations"]
    assert observations[0]["field"] == "content"
    assert observations[0]["byte_count"] == len(body.encode("utf-8"))
    assert len(observations[0]["sha256"]) == 64


@pytest.mark.asyncio
async def test_service_records_resolution_and_rule_delivery_audit_events(
    tmp_path: Path,
) -> None:
    runtime_store = RuntimeStore(tmp_path / "runtime.db")
    exec_store = PersistentApprovalStore(tmp_path / "approvals.db")
    exec_store.create(
        approval_id="approval-audit",
        command="echo ok",
        shell="powershell",
        scope="dangerous_command",
        requester_id="requester-audit",
        request_digest="a" * 64,
        context_digest="b" * 64,
        expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        command_payload={"command": "echo ok", "shell": "powershell"},
    )
    exec_store.resolve(
        "approval-audit",
        decision="approve_and_save_rule",
        requester_id="requester-audit",
        request_digest="a" * 64,
        context_digest="b" * 64,
        rule_side_effect={
            "payload": {
                "tokens": ["echo", "ok"],
                "decision": "allow",
                "match": "exact",
                "shells": ["powershell"],
            },
            "target_config_path": str(tmp_path / "config.yaml"),
        },
    )
    service = RuntimeService(
        engine=object(),
        store=runtime_store,
        exec_approval_store=exec_store,
        exec_runtime=ExecRuntime(),
    )

    delivered = await service.deliver_approval_side_effects_once()
    reject_store = InMemoryApprovalStore()
    reject_store.create(
        approval_id="approval-reject",
        command="echo reject",
        shell="powershell",
        scope="dangerous_command",
        command_payload={"command": "echo reject", "shell": "powershell"},
    )
    rejection_service = RuntimeService(
        engine=object(),
        store=runtime_store,
        exec_approval_store=reject_store,
        exec_runtime=ExecRuntime(),
    )
    rejected = await rejection_service.resolve_approval(
        "approval-reject",
        decision="reject",
    )
    events = await runtime_store.list_security_audit_events()

    assert delivered[0]["status"] == "delivered"
    assert rejected is not None
    assert rejected["status"] == "rejected"
    assert {event["event_type"] for event in events} == {
        "approval_resolved",
        "rule_persistence_delivered",
    }
