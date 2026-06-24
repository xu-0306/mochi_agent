"""Runtime store tests."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mochi.runtime.store import RuntimeStore


def test_runtime_store_persists_tasks_events_and_approvals(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "sessions" / "runtime.db")
    task = asyncio.run(
        store.create_task_run(
            task_id="task-1",
            input_text="hello",
            session_id="s1",
            project_id=None,
            workspace_dir=None,
            project_workspace_dir=str(tmp_path / "project-workspace"),
            task_workspace_dir=str(tmp_path / "sessions" / "runtime-tasks" / "task-1" / "workspace"),
            inference_overrides={"temperature": 0.1},
        )
    )
    assert task["id"] == "task-1"
    assert task["status"] == "queued"

    asyncio.run(store.append_task_event("task-1", {"type": "thinking", "content": "..." }))
    events = asyncio.run(store.get_task_events("task-1"))
    assert events == [{"type": "thinking", "content": "..."}]

    approval = asyncio.run(
        store.create_approval_request(
            approval_id="approval-1",
            task_id="task-1",
            call_id="call-1",
            tool_name="shell",
            arguments={"cmd": "ls"},
        )
    )
    assert approval["status"] == "pending"
    assert approval["arguments"] == {"cmd": "ls"}

    resolved = asyncio.run(
        store.resolve_approval_request(
            "approval-1",
            decision="approve_once",
            reason="ok",
        )
    )
    assert resolved is not None
    assert resolved["status"] == "approved_once"
    assert resolved["reason"] == "ok"

    asyncio.run(store.update_task_status("task-1", "succeeded"))
    saved_task = asyncio.run(store.get_task_run("task-1"))
    assert saved_task is not None
    assert saved_task["status"] == "succeeded"
    assert saved_task["project_workspace_dir"] == str(tmp_path / "project-workspace")
    assert saved_task["task_workspace_dir"] == str(
        tmp_path / "sessions" / "runtime-tasks" / "task-1" / "workspace"
    )


def test_runtime_store_persists_goals_and_attempts(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "sessions" / "runtime.db")
    goal = asyncio.run(
        store.create_goal(
            goal_id="goal-1",
            objective="Collect forum conversations for corpus building",
            title="Forum Corpus",
            goal_type="dataset_collection",
            protocol_id="teacher_student_distill",
            topic="forum corpus",
            project_id="proj-1",
            workspace_dir=str(tmp_path / "workspace"),
            run_policy={"max_wall_clock_sec": 18_000},
            capability_policy={"allowed_tools": ["web_search"]},
            source_manifest={"sources": [{"id": "forum-a"}]},
            summary={"phase": "created"},
            metadata={"packet": "A"},
        )
    )
    assert goal["id"] == "goal-1"
    assert goal["status"] == "created"
    assert goal["attempts"] == []

    attempt = asyncio.run(
        store.create_goal_attempt(
            attempt_id="goal-attempt-1",
            goal_id="goal-1",
            attempt_index=1,
            status="queued",
            trigger="manual_start",
            summary={"queued_from_status": "created"},
            metadata={"launcher": "operator"},
        )
    )
    assert attempt["id"] == "goal-attempt-1"
    assert attempt["goal_id"] == "goal-1"
    assert attempt["trigger"] == "manual_start"

    asyncio.run(
        store.update_goal_status(
            "goal-1",
            "queued",
            current_attempt_id="goal-attempt-1",
        )
    )
    asyncio.run(
        store.update_goal_attempt_status(
            "goal-attempt-1",
            "paused",
            summary={"checkpoint": "checkpoint-1"},
            metadata={"reason": "operator_pause"},
            latest_error="paused by operator",
        )
    )
    asyncio.run(
        store.update_goal_metadata(
            "goal-1",
            summary={"phase": "paused"},
            metadata={"owner": "worker-a"},
            latest_error="waiting for operator resume",
        )
    )

    saved_goal = asyncio.run(store.get_goal("goal-1"))
    assert saved_goal is not None
    assert saved_goal["current_attempt_id"] == "goal-attempt-1"
    assert saved_goal["run_policy"]["max_wall_clock_sec"] == 18_000
    assert saved_goal["capability_policy"]["allowed_tools"] == ["web_search"]
    assert saved_goal["summary"]["phase"] == "paused"
    assert saved_goal["metadata"]["owner"] == "worker-a"
    assert saved_goal["latest_error"] == "waiting for operator resume"
    assert len(saved_goal["attempts"]) == 1
    assert saved_goal["attempts"][0]["status"] == "paused"
    assert saved_goal["attempts"][0]["summary"]["checkpoint"] == "checkpoint-1"
    assert saved_goal["attempts"][0]["metadata"]["reason"] == "operator_pause"

    goals = asyncio.run(store.list_goals())
    assert len(goals) == 1
    assert goals[0]["id"] == "goal-1"
    assert goals[0]["attempts"][0]["id"] == "goal-attempt-1"


def test_runtime_store_persists_goal_leases_and_audit_findings(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "sessions" / "runtime.db")
    asyncio.run(
        store.create_goal(
            goal_id="goal-lease-1",
            objective="Supervise a long-running dataset collection goal",
        )
    )

    lease = asyncio.run(
        store.upsert_goal_lease(
            goal_id="goal-lease-1",
            owner_id="runtime-owner-1",
            metadata={"reason": "startup_recovery"},
            acquired_at="2026-06-21T10:00:00+00:00",
            heartbeat_at="2026-06-21T10:00:10+00:00",
            expires_at="2026-06-21T10:02:10+00:00",
        )
    )
    assert lease is not None
    assert lease["goal_id"] == "goal-lease-1"
    assert lease["owner_id"] == "runtime-owner-1"
    assert lease["metadata"]["reason"] == "startup_recovery"

    taken_over = asyncio.run(
        store.upsert_goal_lease(
            goal_id="goal-lease-1",
            owner_id="runtime-owner-2",
            metadata={"reason": "maintenance_takeover"},
            acquired_at="2026-06-21T10:03:00+00:00",
            heartbeat_at="2026-06-21T10:03:05+00:00",
            expires_at="2026-06-21T10:05:05+00:00",
            force_takeover=True,
        )
    )
    assert taken_over is not None
    assert taken_over["owner_id"] == "runtime-owner-2"
    assert taken_over["takeover_count"] == 1

    finding = asyncio.run(
        store.upsert_goal_audit_finding(
            goal_id="goal-lease-1",
            finding_code="stale_running",
            summary="Goal lease was stale and required takeover.",
            details={"previous_owner_id": "runtime-owner-1"},
        )
    )
    assert finding["goal_id"] == "goal-lease-1"
    assert finding["finding_code"] == "stale_running"
    assert finding["status"] == "open"
    assert finding["details"]["previous_owner_id"] == "runtime-owner-1"
    assert finding["resolved_at"] is None
    assert finding["closed_at"] is None

    findings = asyncio.run(store.list_goal_audit_findings("goal-lease-1", status="open"))
    assert len(findings) == 1
    assert findings[0]["summary"] == "Goal lease was stale and required takeover."

    leases = asyncio.run(store.list_goal_leases(owner_id="runtime-owner-2"))
    assert len(leases) == 1
    assert leases[0]["goal_id"] == "goal-lease-1"


def test_runtime_store_persists_goal_audit_finding_resolution_lifecycle(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "sessions" / "runtime.db")
    asyncio.run(
        store.create_goal(
            goal_id="goal-resolution-1",
            objective="Track audit-finding resolution lifecycle",
        )
    )

    finding = asyncio.run(
        store.upsert_goal_audit_finding(
            goal_id="goal-resolution-1",
            finding_code="stale_running",
            summary="Goal lease became stale during supervision.",
            details={"previous_owner_id": "runtime-owner-1"},
        )
    )
    assert finding["status"] == "open"
    assert finding["resolved_at"] is None
    assert finding["closed_at"] is None

    resolved = asyncio.run(store.resolve_goal_audit_finding(finding["id"]))
    assert resolved is not None
    assert resolved["id"] == finding["id"]
    assert resolved["status"] == "resolved"
    assert resolved["resolved_at"] is not None
    assert resolved["closed_at"] is None

    saved_resolved = asyncio.run(store.get_goal_audit_finding(finding["id"]))
    assert saved_resolved is not None
    assert saved_resolved["status"] == "resolved"
    assert saved_resolved["resolved_at"] == resolved["resolved_at"]
    assert asyncio.run(store.list_goal_audit_findings("goal-resolution-1", status="open")) == []

    closed = asyncio.run(store.close_goal_audit_finding(finding["id"]))
    assert closed is not None
    assert closed["id"] == finding["id"]
    assert closed["status"] == "closed"
    assert closed["resolved_at"] == resolved["resolved_at"]
    assert closed["closed_at"] is not None

    closed_findings = asyncio.run(store.list_goal_audit_findings("goal-resolution-1", status="closed"))
    assert len(closed_findings) == 1
    assert closed_findings[0]["id"] == finding["id"]

    reopened = asyncio.run(
        store.upsert_goal_audit_finding(
            goal_id="goal-resolution-1",
            finding_code="stale_running",
            summary="Goal lease became stale again after recovery.",
            details={"previous_owner_id": "runtime-owner-2"},
        )
    )
    assert reopened["id"] != finding["id"]
    assert reopened["status"] == "open"
    assert reopened["resolved_at"] is None
    assert reopened["closed_at"] is None

    all_findings = asyncio.run(store.list_goal_audit_findings("goal-resolution-1"))
    assert [saved_finding["status"] for saved_finding in all_findings] == ["open", "closed"]
    assert [saved_finding["id"] for saved_finding in all_findings] == [
        reopened["id"],
        finding["id"],
    ]


def test_runtime_store_persists_goal_checkpoints_and_memory_snapshots(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "sessions" / "runtime.db")
    asyncio.run(
        store.create_goal(
            goal_id="goal-progress-1",
            objective="Persist durable progress snapshots",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        store.create_goal_attempt(
            attempt_id="goal-progress-attempt-1",
            goal_id="goal-progress-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-progress-run-1",
        )
    )

    checkpoint_1 = asyncio.run(
        store.create_goal_checkpoint(
            goal_id="goal-progress-1",
            attempt_id="goal-progress-attempt-1",
            agent_run_id="linked-progress-run-1",
            checkpoint_index=3,
            stage="evidence_collection",
            source="agent_run_sync",
            payload={
                "checkpoint_index": 3,
                "stage": "evidence_collection",
                "recovery_state": {"status": "awaiting_resources"},
            },
            metadata={"signature": "sig-1"},
            captured_at="2026-06-23T01:00:00+00:00",
        )
    )
    checkpoint_2 = asyncio.run(
        store.create_goal_checkpoint(
            goal_id="goal-progress-1",
            attempt_id="goal-progress-attempt-1",
            agent_run_id="linked-progress-run-1",
            checkpoint_index=4,
            stage="controller_decision",
            source="agent_run_sync",
            payload={
                "checkpoint_index": 4,
                "stage": "controller_decision",
                "approval_state": {"approval_ids": ["exec-approval-1"]},
            },
            metadata={"signature": "sig-2"},
            captured_at="2026-06-23T01:05:00+00:00",
        )
    )
    assert checkpoint_1["goal_id"] == "goal-progress-1"
    assert checkpoint_2["checkpoint_index"] == 4

    memory_snapshot = asyncio.run(
        store.create_goal_memory_snapshot(
            goal_id="goal-progress-1",
            attempt_id="goal-progress-attempt-1",
            checkpoint_id=checkpoint_2["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={
                "checkpoint_index": 4,
                "stage": "controller_decision",
                "pending_approval_ids": ["exec-approval-1"],
                "unfinished_steps": ["resume after approval"],
            },
            metadata={"signature": "mem-sig-2"},
            captured_at="2026-06-23T01:05:01+00:00",
        )
    )
    assert memory_snapshot["goal_id"] == "goal-progress-1"
    assert memory_snapshot["checkpoint_id"] == checkpoint_2["id"]
    assert memory_snapshot["snapshot_kind"] == "compact_recovery_v1"

    latest_checkpoint = asyncio.run(
        store.get_latest_goal_checkpoint(
            "goal-progress-1",
            attempt_id="goal-progress-attempt-1",
        )
    )
    assert latest_checkpoint is not None
    assert latest_checkpoint["id"] == checkpoint_2["id"]
    assert latest_checkpoint["payload"]["approval_state"]["approval_ids"] == ["exec-approval-1"]

    checkpoints = asyncio.run(
        store.list_goal_checkpoints(
            "goal-progress-1",
            attempt_id="goal-progress-attempt-1",
        )
    )
    assert [item["id"] for item in checkpoints] == [checkpoint_2["id"], checkpoint_1["id"]]

    latest_memory_snapshot = asyncio.run(
        store.get_latest_goal_memory_snapshot(
            "goal-progress-1",
            attempt_id="goal-progress-attempt-1",
        )
    )
    assert latest_memory_snapshot is not None
    assert latest_memory_snapshot["id"] == memory_snapshot["id"]
    assert latest_memory_snapshot["snapshot"]["unfinished_steps"] == ["resume after approval"]

    memory_snapshots = asyncio.run(
        store.list_goal_memory_snapshots(
            "goal-progress-1",
            attempt_id="goal-progress-attempt-1",
        )
    )
    assert [item["id"] for item in memory_snapshots] == [memory_snapshot["id"]]


def test_runtime_store_persists_goal_worker_generations(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "sessions" / "runtime.db")
    asyncio.run(
        store.create_goal(
            goal_id="goal-generation-1",
            objective="Persist worker-generation lifecycle state",
        )
    )
    asyncio.run(
        store.create_goal_attempt(
            attempt_id="goal-generation-attempt-1",
            goal_id="goal-generation-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-generation-run-1",
        )
    )

    checkpoint = asyncio.run(
        store.create_goal_checkpoint(
            goal_id="goal-generation-1",
            attempt_id="goal-generation-attempt-1",
            agent_run_id="linked-generation-run-1",
            checkpoint_index=7,
            stage="context_handoff",
            source="supervisor_rollover",
            payload={"checkpoint_index": 7, "stage": "context_handoff"},
            captured_at="2026-06-23T02:05:00+00:00",
        )
    )
    snapshot = asyncio.run(
        store.create_goal_memory_snapshot(
            goal_id="goal-generation-1",
            attempt_id="goal-generation-attempt-1",
            checkpoint_id=checkpoint["id"],
            snapshot_kind="compact_handoff_v1",
            snapshot={"unfinished_steps": ["refresh worker context"]},
            captured_at="2026-06-23T02:05:01+00:00",
        )
    )

    generation_1 = asyncio.run(
        store.create_goal_worker_generation(
            goal_id="goal-generation-1",
            attempt_id="goal-generation-attempt-1",
            agent_run_id="linked-generation-run-1",
            generation_index=1,
            status="running",
            metadata={"mode": "bootstrap"},
            started_at="2026-06-23T02:00:00+00:00",
        )
    )
    generation_2 = asyncio.run(
        store.create_goal_worker_generation(
            goal_id="goal-generation-1",
            attempt_id="goal-generation-attempt-1",
            agent_run_id="linked-generation-run-2",
            generation_index=2,
            status="running",
            rollover_reason="scheduled_refresh",
            parent_generation_id=generation_1["id"],
            resume_source_snapshot_id=snapshot["id"],
            metadata={"mode": "handoff_resume"},
            started_at="2026-06-23T02:06:00+00:00",
        )
    )

    assert generation_1["goal_id"] == "goal-generation-1"
    assert generation_1["metadata"]["mode"] == "bootstrap"
    assert generation_2["generation_index"] == 2
    assert generation_2["parent_generation_id"] == generation_1["id"]
    assert generation_2["resume_source_snapshot_id"] == snapshot["id"]
    assert generation_2["rollover_reason"] == "scheduled_refresh"

    saved_generation = asyncio.run(store.get_goal_worker_generation(generation_2["id"]))
    assert saved_generation is not None
    assert saved_generation["metadata"]["mode"] == "handoff_resume"

    latest_generation = asyncio.run(
        store.get_latest_goal_worker_generation(
            "goal-generation-1",
            attempt_id="goal-generation-attempt-1",
        )
    )
    assert latest_generation is not None
    assert latest_generation["id"] == generation_2["id"]

    generations = asyncio.run(
        store.list_goal_worker_generations(
            "goal-generation-1",
            attempt_id="goal-generation-attempt-1",
        )
    )
    assert [item["id"] for item in generations] == [generation_2["id"], generation_1["id"]]

    asyncio.run(
        store.update_goal_worker_generation(
            generation_1["id"],
            status="rolled_over",
            rollover_reason="scheduled_refresh",
            metadata={"handoff_snapshot_id": snapshot["id"]},
            finished_at="2026-06-23T02:05:59+00:00",
        )
    )
    updated_generation = asyncio.run(store.get_goal_worker_generation(generation_1["id"]))
    assert updated_generation is not None
    assert updated_generation["status"] == "rolled_over"
    assert updated_generation["rollover_reason"] == "scheduled_refresh"
    assert updated_generation["metadata"]["handoff_snapshot_id"] == snapshot["id"]
    assert updated_generation["finished_at"] == "2026-06-23T02:05:59+00:00"


def test_runtime_store_migrates_goal_worker_generation_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions" / "runtime.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE goal_worker_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal_id TEXT NOT NULL
            )
            """
        )
        conn.commit()

    store = RuntimeStore(db_path)
    asyncio.run(store.initialize())

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(goal_worker_generations)")}

    assert {
        "attempt_id",
        "agent_run_id",
        "generation_index",
        "status",
        "rollover_reason",
        "parent_generation_id",
        "resume_source_snapshot_id",
        "metadata_json",
        "started_at",
        "finished_at",
        "created_at",
        "updated_at",
    }.issubset(columns)


def test_runtime_store_claim_goal_attempt_agent_run_id_is_atomic(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "sessions" / "runtime.db")
    asyncio.run(
        store.create_goal(
            goal_id="goal-claim-1",
            objective="Atomically claim a linked agent run id",
        )
    )
    asyncio.run(
        store.create_goal_attempt(
            attempt_id="goal-claim-attempt-1",
            goal_id="goal-claim-1",
            attempt_index=1,
            status="queued",
            trigger="manual_start",
        )
    )

    first_claim = asyncio.run(
        store.claim_goal_attempt_agent_run_id("goal-claim-attempt-1", "run-claim-1")
    )
    second_claim = asyncio.run(
        store.claim_goal_attempt_agent_run_id("goal-claim-attempt-1", "run-claim-2")
    )

    assert first_claim is not None
    assert first_claim["agent_run_id"] == "run-claim-1"
    assert second_claim is not None
    assert second_claim["agent_run_id"] == "run-claim-1"


def test_runtime_store_does_not_overwrite_fresh_foreign_goal_lease_without_force_takeover(
    tmp_path: Path,
) -> None:
    store = RuntimeStore(tmp_path / "sessions" / "runtime.db")
    asyncio.run(
        store.create_goal(
            goal_id="goal-lease-1",
            objective="Keep a fresh foreign lease intact",
        )
    )
    now = datetime.now(UTC)
    acquired_at = now.isoformat()
    future_expires_at = (now + timedelta(minutes=5)).isoformat()

    first_lease = asyncio.run(
        store.upsert_goal_lease(
            goal_id="goal-lease-1",
            owner_id="runtime-a",
            metadata={"reason": "runtime-a"},
            acquired_at=acquired_at,
            heartbeat_at=acquired_at,
            expires_at=future_expires_at,
        )
    )
    assert first_lease is not None
    assert first_lease["owner_id"] == "runtime-a"

    blocked_takeover = asyncio.run(
        store.upsert_goal_lease(
            goal_id="goal-lease-1",
            owner_id="runtime-b",
            metadata={"reason": "runtime-b"},
            acquired_at=acquired_at,
            heartbeat_at=acquired_at,
            expires_at=future_expires_at,
        )
    )
    assert blocked_takeover is not None
    assert blocked_takeover["owner_id"] == "runtime-a"
    assert blocked_takeover["metadata"]["reason"] == "runtime-a"
    assert blocked_takeover["takeover_count"] == 0

    forced_takeover = asyncio.run(
        store.upsert_goal_lease(
            goal_id="goal-lease-1",
            owner_id="runtime-b",
            metadata={"reason": "runtime-b"},
            acquired_at=acquired_at,
            heartbeat_at=acquired_at,
            expires_at=future_expires_at,
            force_takeover=True,
        )
    )
    assert forced_takeover is not None
    assert forced_takeover["owner_id"] == "runtime-b"
    assert forced_takeover["metadata"]["reason"] == "runtime-b"
    assert forced_takeover["takeover_count"] == 1


def test_runtime_store_round_trips_packet_c_runtime_policy_fields(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "sessions" / "runtime.db")
    run_policy = {
        "requested_duration_text": "2 hours",
        "requested_duration_sec": 7_200,
        "requested_duration_source": "objective",
        "runtime_mode": "fixed_duration",
        "soft_deadline_at": "2026-06-22T10:00:00+00:00",
        "hard_stop_at": "2026-06-22T12:00:00+00:00",
        "checkpoint_interval_steps": 5,
        "checkpoint_interval_sec": 60,
        "generation_refresh_interval_sec": 1_800,
        "context_handoff_threshold": 0.85,
        "approval_mode": "default",
        "max_attempt_retries": 4,
    }
    asyncio.run(
        store.create_goal(
            goal_id="goal-packet-c-1",
            objective="Persist Packet C runtime-policy fields",
            run_policy=run_policy,
        )
    )

    saved_goal = asyncio.run(store.get_goal("goal-packet-c-1"))
    assert saved_goal is not None
    assert saved_goal["run_policy"] == run_policy

    goals = asyncio.run(store.list_goals())
    assert len(goals) == 1
    assert goals[0]["run_policy"] == run_policy


def test_runtime_store_reads_legacy_goal_rows_without_packet_c_fields(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "sessions" / "runtime.db")
    asyncio.run(
        store.create_goal(
            goal_id="goal-legacy-1",
            objective="Load a legacy goal row",
            run_policy={"max_wall_clock_sec": 18_000},
        )
    )

    saved_goal = asyncio.run(store.get_goal("goal-legacy-1"))
    assert saved_goal is not None
    assert saved_goal["run_policy"]["max_wall_clock_sec"] == 18_000
    assert "requested_duration_sec" not in saved_goal["run_policy"]
    assert "runtime_mode" not in saved_goal["run_policy"]

    goals = asyncio.run(store.list_goals())
    assert len(goals) == 1
    assert goals[0]["run_policy"]["max_wall_clock_sec"] == 18_000
    assert "requested_duration_sec" not in goals[0]["run_policy"]
