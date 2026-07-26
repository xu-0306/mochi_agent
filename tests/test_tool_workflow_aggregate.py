from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from mochi.api.tool_workflow_aggregate import (
    AggregatePayloadError,
    UnsupportedAggregateSourceError,
    adapt_legacy_turn_events_v1,
    build_tool_workflow_idempotency_key_v1,
    canonical_json_subset_v1,
    parse_tool_workflow_aggregate_v1,
    reduce_tool_workflow_aggregate_v1,
)


_FIXTURES = Path(__file__).parent / "fixtures" / "tool_workflow_aggregate" / "v1_cases.json"


def _cases() -> dict[str, dict[str, object]]:
    return json.loads(_FIXTURES.read_text(encoding="utf-8"))


def _reduce(case: dict[str, object]) -> dict[str, object]:
    return reduce_tool_workflow_aggregate_v1(**copy.deepcopy(case))


def test_complete_mutation_with_verified_receipt_is_canonical_and_parseable() -> None:
    aggregate = _reduce(_cases()["complete_verified"])

    assert aggregate["state"]["turn_status"] == "completed"
    assert aggregate["state"]["integrity"] == "complete"
    assert aggregate["state"]["calls"] == [
        {
            "call_id": "call-1",
            "operation_id": "operation-1",
            "tool_name": "file_write",
            "arguments_digest": "7409ff8f6dbce36e2536c8223fa8ca9a9e862ca5ddee7ca18a53e0dfb70ce989",
            "target": None,
            "activation_status": "not_observed",
            "review_status": "consumed",
            "approval_id": "approval-1",
            "execution_status": "succeeded",
            "verification_status": "verified",
            "receipt_reference": "receipt-1",
            "changed_paths": ["README.md"],
            "blocker": None,
        }
    ]
    assert re.fullmatch(r"twa:v1:[A-Za-z0-9_-]{43}", str(aggregate["event_id"]))
    assert aggregate["source_refs"]["timeline"]["timeline_version"] == "session-turn-timeline-v4"
    assert parse_tool_workflow_aggregate_v1(aggregate) == aggregate


def test_raw_success_without_required_receipt_is_partial_and_never_verified() -> None:
    aggregate = _reduce(_cases()["success_without_receipt"])
    call = aggregate["state"]["calls"][0]

    assert aggregate["state"]["turn_status"] == "blocked"
    assert aggregate["state"]["integrity"] == "partial"
    assert call["execution_status"] == "succeeded"
    assert call["verification_status"] == "not_observed"


def test_pending_approval_blocks_terminal_completion_without_erasing_matched_execution() -> None:
    case = _cases()["complete_verified"]
    case["approvals"][0]["status"] = "pending"

    aggregate = _reduce(case)
    call = aggregate["state"]["calls"][0]
    assert aggregate["state"]["turn_status"] == "blocked"
    assert call["review_status"] == "pending"
    assert call["execution_status"] == "succeeded"


def test_abandoned_pre_effect_call_survives_turn_cancellation() -> None:
    aggregate = _reduce(_cases()["abandoned_cancelled"])
    call = aggregate["state"]["calls"][0]

    assert aggregate["state"]["turn_status"] == "cancelled"
    assert call["execution_status"] == "abandoned"
    assert call["verification_status"] == "not_required"


def test_unknown_operation_precedes_cancel_and_never_becomes_abandoned() -> None:
    aggregate = _reduce(_cases()["unknown_operation"])
    call = aggregate["state"]["calls"][0]

    assert aggregate["state"]["turn_status"] == "unknown"
    assert call["execution_status"] == "unknown"


def test_mismatched_receipt_is_partial_and_cannot_complete() -> None:
    case = _cases()["complete_verified"]
    case["receipts"][0]["receipt"]["operation_id"] = "other-operation"

    aggregate = _reduce(case)
    call = aggregate["state"]["calls"][0]
    assert aggregate["state"]["integrity"] == "partial"
    assert aggregate["state"]["turn_status"] == "blocked"
    assert call["verification_status"] == "not_observed"


def test_legacy_raw_events_remain_partial_and_do_not_fabricate_execution_evidence() -> None:
    case = _cases()["legacy_raw_events"]
    aggregate = adapt_legacy_turn_events_v1(**case)
    call = aggregate["state"]["calls"][0]

    assert aggregate["state"]["integrity"] == "partial"
    assert aggregate["state"]["turn_status"] == "unknown"
    assert call["operation_id"] is None
    assert call["execution_status"] == "not_started"
    assert call["verification_status"] == "not_observed"


def test_future_source_and_future_aggregate_fail_closed() -> None:
    source = _cases()["complete_verified"]
    source["timeline"]["timeline_version"] = "session-turn-timeline-v999"
    with pytest.raises(UnsupportedAggregateSourceError):
        _reduce(source)

    aggregate = _reduce(_cases()["complete_verified"])
    aggregate["schema_version"] = 2
    with pytest.raises(AggregatePayloadError):
        parse_tool_workflow_aggregate_v1(aggregate)


def test_idempotency_is_order_independent_and_subset_serializer_rejects_float() -> None:
    first = _reduce(_cases()["complete_verified"])
    second_case = _cases()["complete_verified"]
    second_case["approvals"] = list(reversed(second_case["approvals"]))
    second_case["receipts"] = list(reversed(second_case["receipts"]))
    second = _reduce(second_case)

    assert first["idempotency_key"] == second["idempotency_key"]
    assert canonical_json_subset_v1({"a": 1, "b": [None, True]}) == '{"a":1,"b":[null,true]}'
    with pytest.raises(AggregatePayloadError):
        canonical_json_subset_v1({"unsafe": 1.5})


def test_timeline_result_reference_and_artifact_receipt_reference_are_distinct() -> None:
    case = _cases()["complete_verified"]
    descriptor = case["timeline"]["turns"][0]["operation_descriptors"][0]
    descriptor["receipt_reference"] = "tool-result-event-1"
    case["receipts"][0]["receipt_reference"] = "artifact-receipt-1"

    aggregate = _reduce(case)
    call = aggregate["state"]["calls"][0]
    assert aggregate["state"]["turn_status"] == "completed"
    assert call["receipt_reference"] == "artifact-receipt-1"
    assert aggregate["source_refs"]["receipts"][0]["receipt_reference"] == "artifact-receipt-1"


def test_identical_duplicate_approval_and_receipt_are_deduplicated() -> None:
    first = _reduce(_cases()["complete_verified"])
    case = _cases()["complete_verified"]
    case["approvals"].append(copy.deepcopy(case["approvals"][0]))
    case["receipts"].append(copy.deepcopy(case["receipts"][0]))

    duplicate = _reduce(case)
    assert duplicate["idempotency_key"] == first["idempotency_key"]
    assert len(duplicate["source_refs"]["approvals"]) == 1
    assert len(duplicate["source_refs"]["receipts"]) == 1


def test_conflicting_duplicate_receipt_fails_closed() -> None:
    case = _cases()["complete_verified"]
    conflict = copy.deepcopy(case["receipts"][0])
    conflict["receipt_reference"] = "other-artifact-receipt"
    case["receipts"].append(conflict)

    with pytest.raises(UnsupportedAggregateSourceError):
        _reduce(case)


def test_conflicting_duplicate_approval_and_unknown_approval_status_fail_closed() -> None:
    conflicting = _cases()["complete_verified"]
    conflict = copy.deepcopy(conflicting["approvals"][0])
    conflict["status"] = "rejected"
    conflict["approval_revision"] = 5
    conflicting["approvals"].append(conflict)
    with pytest.raises(UnsupportedAggregateSourceError):
        _reduce(conflicting)

    unknown_status = _cases()["complete_verified"]
    unknown_status["approvals"][0]["status"] = "future_approval_status"
    with pytest.raises(UnsupportedAggregateSourceError):
        _reduce(unknown_status)


def test_current_approval_without_exact_operation_join_cannot_update_review() -> None:
    case = _cases()["complete_verified"]
    del case["approvals"][0]["operation_id"]
    del case["approvals"][0]["arguments_digest"]

    aggregate = _reduce(case)
    call = aggregate["state"]["calls"][0]
    assert aggregate["state"]["integrity"] == "partial"
    assert call["review_status"] == "not_observed"
    assert call["approval_id"] is None


def test_receipt_execution_conflict_is_partial_and_never_applies_verification() -> None:
    case = _cases()["complete_verified"]
    descriptor = case["timeline"]["turns"][0]["operation_descriptors"][0]
    descriptor["status"] = "failed"
    case["timeline"]["turns"][0]["terminal_outcome"] = "blocked"

    aggregate = _reduce(case)
    call = aggregate["state"]["calls"][0]
    assert aggregate["state"]["integrity"] == "partial"
    assert call["execution_status"] == "failed"
    assert call["verification_status"] == "unknown"


def test_unknown_timeline_operation_remains_unknown_when_receipt_claims_success() -> None:
    case = _cases()["unknown_operation"]
    receipt = copy.deepcopy(_cases()["complete_verified"]["receipts"][0])
    receipt["session_id"] = "session:delta"
    receipt["receipt_reference"] = "unknown-conflict-receipt"
    receipt["receipt"]["operation_id"] = "operation-4"
    receipt["receipt"]["turn_id"] = "turn:4"
    receipt["receipt"]["tool_call_ids"] = ["call-4"]
    receipt["receipt"]["scope_evidence"] = {
        "authorized_paths_by_call": {"call-4": ["README.md"]},
        "observed_paths_by_call": {"call-4": ["README.md"]},
        "unexpected_changed_paths": [],
    }
    case["receipts"] = [receipt]

    aggregate = _reduce(case)
    call = aggregate["state"]["calls"][0]
    assert aggregate["state"]["turn_status"] == "unknown"
    assert aggregate["state"]["integrity"] == "partial"
    assert call["execution_status"] == "unknown"
    assert call["verification_status"] == "unknown"


def test_strict_reader_rejects_completed_state_with_unknown_operation() -> None:
    aggregate = _reduce(_cases()["complete_verified"])
    aggregate["state"]["calls"][0]["execution_status"] = "unknown"
    aggregate["idempotency_key"] = build_tool_workflow_idempotency_key_v1(
        source_refs=aggregate["source_refs"], state=aggregate["state"]
    )

    with pytest.raises(AggregatePayloadError):
        parse_tool_workflow_aggregate_v1(aggregate)
