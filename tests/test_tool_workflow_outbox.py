from __future__ import annotations

import asyncio
import copy
import json
import re
import sqlite3
import threading
from collections.abc import Mapping
from pathlib import Path

import pytest

from mochi.api.tool_workflow_aggregate import canonical_sha256_subset_v1
from mochi.api.tool_workflow_outbox import (
    ToolWorkflowOutboxError,
    ToolWorkflowOutboxReconcileResult,
    ToolWorkflowOutboxRepository,
    ToolWorkflowOutboxUnsupportedError,
    ToolWorkflowOutboxVerificationResult,
    ToolWorkflowOutboxVerifierDiagnostics,
    approval_observation_from_request,
)
from mochi.runtime.approval_lifecycle import InMemoryApprovalStore, PersistentApprovalStore
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore
from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore, ToolWorkflowPublicationGate
from mochi.sessions.turn_timeline import SessionTurnTimelineRepository
from mochi.tools.base import ToolExecutionContext
from mochi.tools.exec_command import _observe_ordinary_chat_approval as observe_exec_approval
from mochi.tools.file_ops import _observe_ordinary_chat_approval as observe_file_approval


_FIXTURES = Path(__file__).parent / "fixtures" / "tool_workflow_aggregate" / "v1_cases.json"


def _case() -> dict[str, object]:
    return copy.deepcopy(json.loads(_FIXTURES.read_text(encoding="utf-8"))["complete_verified"])


def _timeline_event(case: dict[str, object]) -> dict[str, object]:
    timeline = copy.deepcopy(case["timeline"])
    assert isinstance(timeline, dict)
    timeline["history_current_revision"] = 1
    return {
        "type": "session_meta",
        "event": "session_turn_timeline",
        "schema_version": 1,
        "session_id": case["session_id"],
        "timeline": timeline,
        "timestamp": case["occurred_at"],
    }


def _approval_payload(case: dict[str, object]) -> dict[str, object]:
    approval = case["approvals"][0]
    assert isinstance(approval, dict)
    return {
        "session_id": case["session_id"],
        "operation_id": approval["operation_id"],
        "timeline_call_id": approval["call_id"],
        "arguments_digest": approval["arguments_digest"],
        "ordinary_chat_checkpoint": {
            "source": "ordinary_chat",
            "session_id": case["session_id"],
            "turn_id": case["turn_id"],
            "operation_id": approval["operation_id"],
            "timeline_call_id": approval["call_id"],
            "arguments_digest": approval["arguments_digest"],
        },
    }


async def _append_timeline(store: SessionStore, case: dict[str, object]) -> None:
    snapshot = await store.load_strict_snapshot(str(case["session_id"]))
    result = await store.append_strict_batch_if_revision(
        str(case["session_id"]),
        expected_history_revision=snapshot.history_revision,
        events=(_timeline_event(case),),
    )
    assert result.status == "appended"


def _ordinary_approval(store: PersistentApprovalStore, case: dict[str, object]):
    source = case["approvals"][0]
    assert isinstance(source, dict)
    return store.create(
        approval_id=str(source["approval_id"]),
        command="redacted command",
        shell="auto",
        scope="workspace",
        metadata={"approval_source": "ordinary_chat"},
        command_payload=_approval_payload(case),
        request_digest=str(source["request_digest"]),
        context_digest=str(source["context_digest"]),
    )


@pytest.mark.asyncio
async def test_outbox_reuses_idempotency_key_and_never_self_triggers_sequence(tmp_path: Path) -> None:
    case = _case()
    store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    await _append_timeline(store, case)
    repository = ToolWorkflowOutboxRepository(store, enabled=True)

    initial = await repository.list(str(case["session_id"]), turn_id=str(case["turn_id"]))
    assert len(initial) == 1
    assert initial[0]["seq"] == 1
    repaired = await repository.rebuild_turn(str(case["session_id"]), str(case["turn_id"]))
    assert repaired == initial[0]
    after_restart = ToolWorkflowOutboxRepository(store, enabled=True)
    replayed = await after_restart.rebuild_turn(str(case["session_id"]), str(case["turn_id"]))
    assert replayed == initial[0]
    assert await after_restart.list(str(case["session_id"]), turn_id=str(case["turn_id"])) == initial


@pytest.mark.asyncio
async def test_outbox_uses_special_session_id_as_opaque_identity_not_filename(tmp_path: Path) -> None:
    case = _case()
    session_id = "opaque:a?b"
    case["session_id"] = session_id
    timeline = case["timeline"]
    assert isinstance(timeline, dict)
    timeline["session_id"] = session_id

    store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    await _append_timeline(store, case)

    records = await ToolWorkflowOutboxRepository(store, enabled=True).list(
        session_id,
        turn_id=str(case["turn_id"]),
    )

    assert records[0]["session_id"] == session_id
    assert session_id not in store._session_path(session_id).name  # noqa: SLF001


@pytest.mark.asyncio
async def test_approval_observation_is_idempotent_and_restart_safe_without_tool_execution(tmp_path: Path) -> None:
    case = _case()
    store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    await _append_timeline(store, case)
    approvals = PersistentApprovalStore(tmp_path / "approvals.sqlite3")
    pending = _ordinary_approval(approvals, case)
    repository = ToolWorkflowOutboxRepository(store, enabled=True)

    first = await repository.observe_approval(pending)
    assert first.status == "observed"
    duplicate = await repository.observe_approval(approvals.get(pending.approval_id))  # type: ignore[arg-type]
    assert duplicate.status == "already_observed"
    records = await repository.list(str(case["session_id"]), turn_id=str(case["turn_id"]))
    assert [record["seq"] for record in records] == [1, 2]
    assert records[-1]["state"]["calls"][0]["review_status"] == "pending"

    rejected = approvals.resolve(pending.approval_id, decision="reject")
    assert rejected is not None
    assert rejected.approval_revision == pending.approval_revision + 1
    after_transition = await ToolWorkflowOutboxRepository(store, enabled=True).observe_approval(rejected)
    assert after_transition.status == "observed"
    records = await repository.list(str(case["session_id"]), turn_id=str(case["turn_id"]))
    assert [record["seq"] for record in records] == [1, 2, 3]
    assert records[-1]["state"]["calls"][0]["review_status"] == "rejected"


@pytest.mark.asyncio
async def test_historical_conflicting_observation_revision_fails_closed_even_after_newer_revision(tmp_path: Path) -> None:
    case = _case()
    store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    await _append_timeline(store, case)
    approvals = PersistentApprovalStore(tmp_path / "approvals.sqlite3")
    pending = _ordinary_approval(approvals, case)
    repository = ToolWorkflowOutboxRepository(store, enabled=True)
    await repository.observe_approval(pending)
    rejected = approvals.resolve(pending.approval_id, decision="reject")
    assert rejected is not None
    await repository.observe_approval(rejected)

    snapshot = await store.load_strict_snapshot(str(case["session_id"]))
    original = next(
        event
        for event in snapshot.events
        if event.get("event") == "tool_workflow_approval_observation"
        and event.get("approval_observation", {}).get("approval_revision") == 1
    )
    conflict = dict(original)
    conflict["approval_observation"] = dict(original["approval_observation"])
    conflict["approval_observation"]["status"] = "rejected"
    # Deliberately emulate already-persisted corrupt history.  Publication is
    # disabled only for the fixture write; the reader must still detect it.
    raw_store = SessionStore(tmp_path / "sessions", tool_observability_v1=False)
    appended = await raw_store.append_strict_batch_if_revision(
        str(case["session_id"]),
        expected_history_revision=snapshot.history_revision,
        events=(conflict,),
    )
    assert appended.status == "appended"
    with pytest.raises(ToolWorkflowOutboxUnsupportedError):
        await repository.rebuild_turn(str(case["session_id"]), str(case["turn_id"]))


@pytest.mark.asyncio
async def test_checkpoint_reconciler_repairs_after_db_to_session_crash_and_excludes_task_runtime(tmp_path: Path) -> None:
    case = _case()
    store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    await _append_timeline(store, case)
    approvals = PersistentApprovalStore(tmp_path / "approvals.sqlite3")
    ordinary = _ordinary_approval(approvals, case)
    task_only = approvals.create(
        approval_id="task-runtime-approval",
        command="redacted",
        shell="auto",
        scope="workspace",
        metadata={"approval_source": "task_runtime"},
    )
    assert task_only.approval_revision == 1
    checkpoint = copy.deepcopy(case["checkpoint"])
    assert isinstance(checkpoint, dict)
    checkpoint["approval_record"] = {"approval_id": ordinary.approval_id}
    snapshot = await store.load_strict_snapshot(str(case["session_id"]))
    appended = await store.append_strict_batch_if_revision(
        str(case["session_id"]),
        expected_history_revision=snapshot.history_revision,
        events=(
            {
                "type": "session_meta",
                "event": "turn_execution_checkpoint",
                "schema_version": 1,
                "session_id": case["session_id"],
                "turn_id": case["turn_id"],
                "checkpoint": checkpoint,
                "timestamp": case["occurred_at"],
            },
        ),
    )
    assert appended.status == "appended"
    # The checkpoint is a new authoritative source, so it produced exactly one
    # extra delivery entry before the later approval observation repair.
    assert [item["seq"] for item in await ToolWorkflowOutboxRepository(store, enabled=True).list(
        str(case["session_id"]), turn_id=str(case["turn_id"])
    )] == [1, 2]

    repository = ToolWorkflowOutboxRepository(store, enabled=True)
    repaired = await repository.reconcile_checkpoint_approvals(str(case["session_id"]), approvals)
    assert [(item.approval_id, item.status) for item in repaired] == [(ordinary.approval_id, "observed")]
    assert approvals.get(ordinary.approval_id).status == "pending"  # type: ignore[union-attr]
    assert approvals.get(ordinary.approval_id).approval_revision == 1  # type: ignore[union-attr]
    assert (await repository.observe_approval(task_only)).status == "not_ordinary_chat"


@pytest.mark.asyncio
async def test_feature_flag_stops_new_writes_but_readers_keep_existing_records(tmp_path: Path) -> None:
    case = _case()
    store = SessionStore(tmp_path / "sessions", tool_observability_v1=False)
    await _append_timeline(store, case)
    disabled = ToolWorkflowOutboxRepository(store, enabled=False)
    assert await disabled.list(str(case["session_id"]), turn_id=str(case["turn_id"])) == ()
    assert await disabled.rebuild_turn(str(case["session_id"]), str(case["turn_id"])) is None

    enabled = ToolWorkflowOutboxRepository(store, enabled=True)
    published = await enabled.rebuild_turn(str(case["session_id"]), str(case["turn_id"]))
    assert published is not None
    rollback = ToolWorkflowOutboxRepository(store, enabled=False)
    assert await rollback.list(str(case["session_id"]), turn_id=str(case["turn_id"])) == (published,)


@pytest.mark.asyncio
async def test_disabled_live_gate_rejects_a_prebuilt_outbox_batch(tmp_path: Path) -> None:
    case = _case()
    gate = ToolWorkflowPublicationGate(True)
    store = SessionStore(
        tmp_path / "sessions",
        tool_observability_v1=True,
        tool_workflow_publication_gate=gate,
    )
    await _append_timeline(store, case)
    before = await store.load_strict_snapshot(str(case["session_id"]))
    prebuilt = next(
        _thaw(event)
        for event in before.events
        if event.get("event") == "tool_workflow_aggregate_outbox"
    )
    assert isinstance(prebuilt, dict)

    await gate.set_enabled_async(False)
    result = await store.append_strict_batch_if_revision(
        str(case["session_id"]),
        expected_history_revision=before.history_revision,
        events=(prebuilt,),
    )

    assert result.status == "unchanged"
    assert result.after.history_revision == before.history_revision
    assert result.after.event_count == before.event_count


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


async def _append_unchecked_outbox_event(
    store: SessionStore,
    session_id: str,
    event: dict[str, object],
) -> None:
    snapshot = await store.load_strict_snapshot(session_id)
    result = await store.append_strict_batch_if_revision(
        session_id,
        expected_history_revision=snapshot.history_revision,
        events=(event,),
    )
    assert result.status == "appended"


@pytest.mark.asyncio
async def test_outbox_replay_verifier_reports_a_matching_cache(tmp_path: Path) -> None:
    case = _case()
    store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    await _append_timeline(store, case)

    result = await ToolWorkflowOutboxRepository(store, enabled=True).verify_session(
        str(case["session_id"])
    )
    assert result.checked == result.matched == 1
    assert result.counters() == {
        "source_mismatch": 0,
        "duplicate": 0,
        "gap": 0,
        "unsupported": 0,
    }


@pytest.mark.asyncio
async def test_outbox_replay_verifier_keeps_prior_cache_positions_for_later_timeline_sources(
    tmp_path: Path,
) -> None:
    case = _case()
    store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    await _append_timeline(store, case)
    timeline = SessionTurnTimelineRepository(store)
    snapshot = await store.load_strict_snapshot(str(case["session_id"]))
    admitted = await timeline.admit(
        str(case["session_id"]),
        turn_id="turn:2",
        expected_history_revision=snapshot.history_revision,
    )
    assert admitted.status == "admitted"
    before = await store.load_strict_snapshot(str(case["session_id"]))

    result = await ToolWorkflowOutboxRepository(store, enabled=True).verify_session(
        str(case["session_id"])
    )
    after = await store.load_strict_snapshot(str(case["session_id"]))
    assert result.checked == result.matched == 2
    assert result.counters() == {
        "source_mismatch": 0,
        "duplicate": 0,
        "gap": 0,
        "unsupported": 0,
    }
    assert after.history_revision == before.history_revision
    assert after.event_count == before.event_count


@pytest.mark.asyncio
async def test_outbox_replay_verifier_reports_payload_mismatch_and_duplicate_conflict(tmp_path: Path) -> None:
    case = _case()
    store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    await _append_timeline(store, case)
    snapshot = await store.load_strict_snapshot(str(case["session_id"]))
    original = next(event for event in snapshot.events if event.get("event") == "tool_workflow_aggregate_outbox")
    conflict = _thaw(original)
    assert isinstance(conflict, dict)
    aggregate = conflict["aggregate"]
    assert isinstance(aggregate, dict)
    aggregate["occurred_at"] = "2026-07-25T12:00:01+00:00"
    raw_store = SessionStore(tmp_path / "sessions", tool_observability_v1=False)
    await _append_unchecked_outbox_event(raw_store, str(case["session_id"]), conflict)

    result = await ToolWorkflowOutboxRepository(store, enabled=True).verify_session(
        str(case["session_id"])
    )
    assert result.source_mismatch == 1
    assert result.duplicate == 1


@pytest.mark.asyncio
async def test_outbox_replay_verifier_reports_sequence_gaps(tmp_path: Path) -> None:
    case = _case()
    store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    await _append_timeline(store, case)
    snapshot = await store.load_strict_snapshot(str(case["session_id"]))
    original = next(event for event in snapshot.events if event.get("event") == "tool_workflow_aggregate_outbox")
    gap_event = _thaw(original)
    assert isinstance(gap_event, dict)
    aggregate = gap_event["aggregate"]
    assert isinstance(aggregate, dict)
    aggregate["seq"] = 3
    from mochi.api.tool_workflow_aggregate import build_tool_workflow_event_id_v1

    aggregate["event_id"] = build_tool_workflow_event_id_v1(
        session_id=str(case["session_id"]),
        turn_id=str(case["turn_id"]),
        seq=3,
    )
    raw_store = SessionStore(tmp_path / "sessions", tool_observability_v1=False)
    await _append_unchecked_outbox_event(raw_store, str(case["session_id"]), gap_event)

    result = await ToolWorkflowOutboxRepository(store, enabled=True).verify_session(
        str(case["session_id"])
    )
    assert result.gap == 1


@pytest.mark.asyncio
async def test_outbox_replay_verifier_reports_unsupported_records(tmp_path: Path) -> None:
    case = _case()
    store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    await _append_timeline(store, case)
    snapshot = await store.load_strict_snapshot(str(case["session_id"]))
    original = next(event for event in snapshot.events if event.get("event") == "tool_workflow_aggregate_outbox")
    unsupported = _thaw(original)
    assert isinstance(unsupported, dict)
    unsupported["schema_version"] = 99
    raw_store = SessionStore(tmp_path / "sessions", tool_observability_v1=False)
    await _append_unchecked_outbox_event(raw_store, str(case["session_id"]), unsupported)

    result = await ToolWorkflowOutboxRepository(store, enabled=True).verify_session(
        str(case["session_id"])
    )
    assert result.unsupported == 1


@pytest.mark.asyncio
async def test_runtime_outbox_verifier_counters_are_wired_and_deduplicate_full_replays() -> None:
    class Approval:
        approval_id = "approval-1"
        metadata = {"approval_source": "ordinary_chat"}

    verification = ToolWorkflowOutboxVerificationResult(
        session_id="session-1",
        checked=1,
        source_mismatch=1,
        messages=("position=2: aggregate does not match durable sources",),
        findings=(("source_mismatch", "position=2"),),
    )

    class Outbox:
        enabled = True

        def __init__(self) -> None:
            self.observe_calls = 0
            self.reconcile_calls = 0
            self.verify_calls = 0

        async def observe_approval(self, approval: Approval) -> ToolWorkflowOutboxReconcileResult:
            self.observe_calls += 1
            return ToolWorkflowOutboxReconcileResult(
                approval_id=approval.approval_id,
                session_id="session-1",
                turn_id="turn-1",
                status="observed",
            )

        async def reconcile_checkpoint_approvals(self, session_id: str, store: object) -> tuple[()]:
            assert session_id == "session-1"
            self.reconcile_calls += 1
            return ()

        async def verify_session(self, session_id: str) -> ToolWorkflowOutboxVerificationResult:
            assert session_id == "session-1"
            self.verify_calls += 1
            return verification

    class ApprovalStore:
        def list(self) -> list[Approval]:
            return [Approval()]

    outbox = Outbox()
    service = object.__new__(RuntimeService)
    service._tool_workflow_outbox = outbox
    service._exec_approval_store = ApprovalStore()
    service._tool_workflow_verifier_diagnostics = ToolWorkflowOutboxVerifierDiagnostics()

    await service.reconcile_tool_workflow_approval_observations()
    assert outbox.observe_calls == 1
    assert outbox.reconcile_calls == 1
    # Per-approval handoff uses the SessionStore post-commit delta verifier;
    # full replay is reserved for the background startup/manual audit.
    assert outbox.verify_calls == 0
    await service._verify_tool_workflow_outbox_session("session-1")
    assert service.tool_workflow_outbox_verifier_counters_snapshot() == {
        "source_mismatch": 1,
        "duplicate": 0,
        "gap": 0,
        "unsupported": 0,
    }

    await service.reconcile_tool_workflow_approval_observations()
    assert outbox.verify_calls == 1
    await service._verify_tool_workflow_outbox_session("session-1")
    # The replay sees the same immutable defect but does not inflate the
    # process-lifetime unique-finding counter.
    assert service.tool_workflow_outbox_verifier_counters_snapshot()["source_mismatch"] == 1


@pytest.mark.asyncio
async def test_outbox_rejects_unsupported_source_before_appending_source_or_cache(tmp_path: Path) -> None:
    case = _case()
    timeline_event = _timeline_event(case)
    timeline = timeline_event["timeline"]
    assert isinstance(timeline, dict)
    timeline["timeline_version"] = "session-turn-timeline-v999"
    store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    snapshot = await store.load_strict_snapshot(str(case["session_id"]))
    with pytest.raises(ToolWorkflowOutboxUnsupportedError):
        await store.append_strict_batch_if_revision(
            str(case["session_id"]),
            expected_history_revision=snapshot.history_revision,
            events=(timeline_event,),
        )
    assert (await store.load_strict_snapshot(str(case["session_id"]))).event_count == 0


@pytest.mark.asyncio
async def test_engine_style_artifact_receipt_uses_timeline_tool_result_reference(tmp_path: Path) -> None:
    case = _case()
    timeline = case["timeline"]
    assert isinstance(timeline, dict)
    descriptor = timeline["turns"][0]["operation_descriptors"][0]
    assert isinstance(descriptor, dict)
    descriptor["receipt_reference"] = "turn-one:1"
    store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    await _append_timeline(store, case)

    checkpoint = copy.deepcopy(case["checkpoint"])
    assert isinstance(checkpoint, dict)
    approval = case["approvals"][0]
    assert isinstance(approval, dict)
    receipt_source = case["receipts"][0]
    assert isinstance(receipt_source, dict)
    receipt = copy.deepcopy(receipt_source["receipt"])
    assert isinstance(receipt, dict)

    async def append(event: dict[str, object]) -> None:
        snapshot = await store.load_strict_snapshot(str(case["session_id"]))
        result = await store.append_strict_batch_if_revision(
            str(case["session_id"]),
            expected_history_revision=snapshot.history_revision,
            events=(event,),
        )
        assert result.status == "appended"

    await append(
        {
            "type": "session_meta",
            "event": "turn_execution_checkpoint",
            "schema_version": 1,
            "session_id": case["session_id"],
            "turn_id": case["turn_id"],
            "checkpoint": checkpoint,
            "timestamp": case["occurred_at"],
        }
    )
    await append(
        {
            "type": "session_meta",
            "event": "tool_workflow_approval_observation",
            "schema_version": 1,
            "session_id": case["session_id"],
            "turn_id": case["turn_id"],
            "approval_observation": {
                key: approval[key]
                for key in (
                    "approval_id",
                    "approval_revision",
                    "status",
                    "request_digest",
                    "context_digest",
                    "call_id",
                    "operation_id",
                    "arguments_digest",
                )
            },
            "timestamp": case["occurred_at"],
        }
    )
    # Engine persistence deliberately supplies no receipt_reference here.
    await append(
        {
            "type": "session_meta",
            "event": "artifact_verification_receipt",
            "schema_version": 1,
            "session_id": case["session_id"],
            "turn_id": case["turn_id"],
            "artifact_receipt": receipt,
            "timestamp": case["occurred_at"],
        }
    )

    aggregate = (await ToolWorkflowOutboxRepository(store, enabled=True).list(
        str(case["session_id"]), turn_id=str(case["turn_id"])
    ))[-1]
    call = aggregate["state"]["calls"][0]
    assert aggregate["state"]["turn_status"] == "completed"
    assert call["verification_status"] == "verified"
    assert aggregate["source_refs"]["receipts"][0]["kind"] == "artifact_receipt"
    assert aggregate["source_refs"]["receipts"][0]["receipt_reference"] == "turn-one:1"


@pytest.mark.asyncio
async def test_artifact_receipt_without_matching_timeline_call_fails_closed(tmp_path: Path) -> None:
    case = _case()
    store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    await _append_timeline(store, case)
    receipt_source = case["receipts"][0]
    assert isinstance(receipt_source, dict)
    receipt = copy.deepcopy(receipt_source["receipt"])
    assert isinstance(receipt, dict)
    receipt["tool_call_ids"] = ["other-call"]
    snapshot = await store.load_strict_snapshot(str(case["session_id"]))
    with pytest.raises(ToolWorkflowOutboxError):
        await store.append_strict_batch_if_revision(
            str(case["session_id"]),
            expected_history_revision=snapshot.history_revision,
            events=(
                {
                    "type": "session_meta",
                    "event": "artifact_verification_receipt",
                    "schema_version": 1,
                    "session_id": case["session_id"],
                    "turn_id": case["turn_id"],
                    "artifact_receipt": receipt,
                    "timestamp": case["occurred_at"],
                },
            ),
        )


@pytest.mark.asyncio
async def test_event_identity_is_opaque_for_special_session_and_turn_ids(tmp_path: Path) -> None:
    case = _case()
    case["session_id"] = "session:alpha/with?special"
    case["turn_id"] = "turn:1:with/slash"
    timeline = case["timeline"]
    assert isinstance(timeline, dict)
    timeline["session_id"] = case["session_id"]
    turns = timeline["turns"]
    assert isinstance(turns, list)
    turns[0]["turn_id"] = case["turn_id"]
    checkpoint = case["checkpoint"]
    assert isinstance(checkpoint, dict)
    checkpoint["session_id"] = case["session_id"]
    checkpoint["turn_id"] = case["turn_id"]

    store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    await _append_timeline(store, case)
    record = (await ToolWorkflowOutboxRepository(store, enabled=True).list(
        str(case["session_id"]), turn_id=str(case["turn_id"])
    ))[0]
    assert re.fullmatch(r"twa:v1:[A-Za-z0-9_-]{43}", str(record["event_id"]))
    assert str(case["session_id"]) not in str(record["event_id"])
    assert str(case["turn_id"]) not in str(record["event_id"])


@pytest.mark.parametrize(
    "factory",
    [
        lambda tmp_path: InMemoryApprovalStore(),
        lambda tmp_path: PersistentApprovalStore(tmp_path / "approval-revisions.sqlite3"),
    ],
)
def test_approval_revision_advances_for_every_observable_lifecycle_transition(tmp_path: Path, factory) -> None:
    approvals = factory(tmp_path)
    created = approvals.create(
        approval_id="revision-main",
        command="redacted",
        shell="auto",
        scope="workspace",
    )
    assert created.approval_revision == 1
    resolved = approvals.resolve("revision-main", decision="approve_once")
    assert resolved is not None and resolved.approval_revision == 2
    claimed = approvals.consume(
        "revision-main",
        execution_idempotency_key="revision-key",
        lease_owner="test",
    )
    assert claimed.approval_revision == 3
    recorded = approvals.record_execution_result(
        "revision-main",
        execution_result={"status": "succeeded"},
    )
    assert recorded is not None and recorded.approval_revision == 4
    # The durable execution-result checkpoint is observable even when its
    # payload is equal, so it receives a new source revision.
    repeated = approvals.record_execution_result(
        "revision-main",
        execution_result={"status": "succeeded"},
    )
    assert repeated is not None and repeated.approval_revision == 5
    completed = approvals.complete_consumption(
        "revision-main",
        execution_idempotency_key="revision-key",
        lease_owner="test",
        lease_token=claimed.consume_lease_token or "",
        execution_result={"status": "succeeded"},
    )
    assert completed.approval_revision == 6

    superseded = approvals.create(
        approval_id="revision-supersede",
        command="redacted",
        shell="auto",
        scope="workspace",
    )
    assert approvals.supersede("revision-supersede").approval_revision == superseded.approval_revision + 1


@pytest.mark.asyncio
async def test_pre_revision_approval_row_keeps_legacy_digest_until_a_transition(tmp_path: Path) -> None:
    """The column migration must not manufacture a monotonic source proof."""

    case = _case()
    db_path = tmp_path / "approval-migration.sqlite3"
    payload = _approval_payload(case)
    with sqlite3.connect(db_path) as conn:
        # This is the exact pre-P2.3 approval table shape: no revision column.
        conn.execute(
            """
            CREATE TABLE exec_approval_requests(
                approval_id TEXT PRIMARY KEY,status TEXT NOT NULL,reason TEXT,
                command TEXT NOT NULL,shell TEXT NOT NULL,scope TEXT NOT NULL,
                created_at TEXT NOT NULL,metadata_json TEXT NOT NULL DEFAULT '{}',
                command_payload_json TEXT,execution_result_json TEXT,resolved_at TEXT,
                updated_at TEXT NOT NULL,requester_id TEXT NOT NULL DEFAULT 'legacy',
                request_digest TEXT NOT NULL DEFAULT '',context_digest TEXT NOT NULL DEFAULT '',
                expires_at TEXT NOT NULL DEFAULT '',resolution_kind TEXT,
                execution_idempotency_key TEXT,consume_lease_owner TEXT,
                consume_lease_token TEXT,consume_lease_expires_at TEXT,consumed_at TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO exec_approval_requests(
                approval_id,status,reason,command,shell,scope,created_at,
                metadata_json,command_payload_json,execution_result_json,resolved_at,
                updated_at,requester_id,request_digest,context_digest,expires_at,
                resolution_kind,execution_idempotency_key,consume_lease_owner,
                consume_lease_token,consume_lease_expires_at,consumed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "legacy-row",
                "pending",
                None,
                "redacted",
                "auto",
                "workspace",
                "2026-01-01T00:00:00+00:00",
                json.dumps({"approval_source": "ordinary_chat"}),
                json.dumps(payload),
                None,
                None,
                "2026-01-01T00:00:00+00:00",
                "legacy",
                str(case["approvals"][0]["request_digest"]),  # type: ignore[index]
                str(case["approvals"][0]["context_digest"]),  # type: ignore[index]
                "2099-01-01T00:00:00+00:00",
                None,
                None,
                None,
                None,
                None,
                None,
            ),
        )
        conn.commit()

    approvals = PersistentApprovalStore(db_path)
    legacy = approvals.get("legacy-row")
    assert legacy is not None and legacy.approval_revision is None
    observation = approval_observation_from_request(legacy)
    assert observation is not None
    assert observation["approval_revision"] is None
    assert isinstance(observation["legacy_digest"], str)

    session_store = SessionStore(tmp_path / "sessions", tool_observability_v1=True)
    await _append_timeline(session_store, case)
    outbox = ToolWorkflowOutboxRepository(session_store, enabled=True)
    assert (await outbox.observe_approval(legacy)).status == "observed"
    legacy_ref = (await outbox.list(str(case["session_id"])))[-1]["source_refs"]["approvals"][0]
    assert legacy_ref["approval_revision"] is None
    assert legacy_ref["legacy_digest"] == observation["legacy_digest"]

    resolved = approvals.resolve("legacy-row", decision="approve_once")
    assert resolved is not None and resolved.approval_revision == 1
    assert (await outbox.observe_approval(resolved)).status == "observed"
    current_ref = (await outbox.list(str(case["session_id"])))[-1]["source_refs"]["approvals"][0]
    assert current_ref["approval_revision"] == 1
    assert current_ref["legacy_digest"] is None


@pytest.mark.asyncio
async def test_observer_failure_keeps_the_pending_approval_handoff_non_fatal() -> None:
    async def broken_observer(_approval: object) -> None:
        raise RuntimeError("outbox unavailable")

    context = ToolExecutionContext(
        state={"tool_workflow_approval_observer": broken_observer}
    )
    approval = object()
    # These helpers run after the approval-store create.  They may be audited
    # and repaired later, but must not replace a caller's pending result.
    await observe_file_approval(context, approval)
    await observe_exec_approval(context, approval)


@pytest.mark.asyncio
async def test_publication_gate_drains_inflight_leases_without_serializing_sessions() -> None:
    gate = ToolWorkflowPublicationGate(enabled=True)
    first_entered = threading.Event()
    second_entered = threading.Event()
    release = threading.Event()

    def publish(entered: threading.Event) -> None:
        with gate.publication_transaction() as enabled:
            assert enabled is True
            entered.set()
            while not release.is_set():
                import time

                time.sleep(0.001)

    first = asyncio.create_task(asyncio.to_thread(publish, first_entered))
    second = asyncio.create_task(asyncio.to_thread(publish, second_entered))
    await asyncio.wait_for(asyncio.to_thread(first_entered.wait), timeout=1)
    await asyncio.wait_for(asyncio.to_thread(second_entered.wait), timeout=1)

    disabled = asyncio.create_task(gate.set_enabled_async(False))
    await asyncio.sleep(0.02)
    assert not disabled.done()
    release.set()
    await asyncio.gather(first, second)
    await disabled
    assert gate.enabled is False
    with gate.publication_transaction() as enabled:
        assert enabled is False


@pytest.mark.asyncio
async def test_stale_outbox_observer_cannot_append_after_gate_rollback(tmp_path: Path) -> None:
    case = _case()
    gate = ToolWorkflowPublicationGate(enabled=True)
    store = SessionStore(
        tmp_path / "sessions",
        tool_observability_v1=True,
        tool_workflow_publication_gate=gate,
    )
    await _append_timeline(store, case)
    approvals = PersistentApprovalStore(tmp_path / "approvals.sqlite3")
    pending = _ordinary_approval(approvals, case)
    stale = ToolWorkflowOutboxRepository(store, enabled=True, publication_gate=gate)

    await gate.set_enabled_async(False)
    result = await stale.observe_approval(pending)
    assert result.status == "disabled"
    snapshot = await store.load_strict_snapshot(str(case["session_id"]))
    assert not any(
        event.get("event") == "tool_workflow_approval_observation"
        for event in snapshot.events
    )
    assert len(await stale.list(str(case["session_id"]))) == 1


def test_verifier_diagnostics_distinguish_replaced_payload_at_same_position() -> None:
    diagnostics = ToolWorkflowOutboxVerifierDiagnostics()
    first = ToolWorkflowOutboxVerificationResult(
        session_id="session-1",
        source_mismatch=1,
        findings=(("source_mismatch", "position=7;raw=sha256:first"),),
    )
    diagnostics.record(first)
    diagnostics.record(first)  # Same full replay must not inflate the counter.
    replacement = ToolWorkflowOutboxVerificationResult(
        session_id="session-1",
        source_mismatch=1,
        findings=(("source_mismatch", "position=7;raw=sha256:replacement"),),
    )
    diagnostics.record(replacement)
    assert diagnostics.snapshot()["source_mismatch"] == 2


def test_legacy_observation_digest_is_recomputed_and_v2_cannot_carry_revision() -> None:
    from mochi.api.tool_workflow_outbox import _parse_observation_event

    source = {
        "session_id": "session-1",
        "turn_id": "turn-1",
        "approval_id": "approval-1",
        "approval_revision": None,
        "status": "pending",
        "request_digest": "a" * 64,
        "context_digest": "b" * 64,
        "call_id": "call-1",
        "operation_id": "operation-1",
        "arguments_digest": "c" * 64,
    }
    event = {
        "type": "session_meta",
        "event": "tool_workflow_approval_observation",
        "schema_version": 2,
        "session_id": "session-1",
        "turn_id": "turn-1",
        "approval_observation": {
            **{key: value for key, value in source.items() if key not in {"session_id", "turn_id"}},
            "legacy_digest": canonical_sha256_subset_v1(source),
        },
        "timestamp": "2026-07-25T00:00:00+00:00",
    }
    assert _parse_observation_event(event, session_id="session-1")["approval_observation"]["legacy_digest"]

    tampered = copy.deepcopy(event)
    tampered["approval_observation"]["legacy_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ToolWorkflowOutboxError):
        _parse_observation_event(tampered, session_id="session-1")

    revision_bearing = copy.deepcopy(event)
    revision_bearing["approval_observation"]["approval_revision"] = 1
    revision_bearing["approval_observation"]["legacy_digest"] = None
    with pytest.raises(ToolWorkflowOutboxError):
        _parse_observation_event(revision_bearing, session_id="session-1")


@pytest.mark.asyncio
async def test_runtime_rebind_off_to_on_starts_a_fresh_background_audit(tmp_path: Path) -> None:
    session_store = SessionStore(tmp_path / "sessions")
    service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "runtime.sqlite3"),
        ordinary_chat_session_store=session_store,
    )
    disabled = MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
            "agent": {"tool_observability_v1": False},
        }
    )
    enabled = disabled.model_copy(
        update={
            "agent": disabled.agent.model_copy(update={"tool_observability_v1": True})
        }
    )

    service.bind_app_config(config=disabled, config_path=None)
    assert service._tool_workflow_outbox_audit_task is None  # noqa: SLF001
    service.bind_app_config(config=enabled, config_path=None)
    audit_task = service._tool_workflow_outbox_audit_task  # noqa: SLF001
    assert audit_task is not None
    await audit_task
    service.bind_app_config(config=enabled, config_path=None)
    assert service._tool_workflow_outbox_audit_task is audit_task  # noqa: SLF001
