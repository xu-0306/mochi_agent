from __future__ import annotations

import pytest

from mochi.agents.events import (
    ErrorEvent,
    FinalAnswerEvent,
    StatusEvent,
    ToolCallResultEvent,
)
from mochi.agents.failures import (
    FAILURE_ENVELOPE_REQUIRED_FIELDS,
    FailureEnvelope,
    FailureEnvelopeVersionError,
    legacy_failure_envelope,
)


@pytest.mark.parametrize(
    ("event_type", "error", "code", "metadata", "expected_kind", "expected_origin"),
    [
        ("error", "provider unavailable", "AGENT_ERROR", {"error_type": "backend_request_error"}, "backend_error", "backend"),
        ("tool_call_result", "tool failed", None, {}, "tool_error", "tool"),
        ("tool_call_result", None, None, {"status": "approval_pending", "approval_id": "approval-1"}, "tool_denied", "tool"),
        ("status", None, None, {"status": "runtime_steering"}, "runtime_steering", "runtime"),
        ("error", "too large", None, {"error_type": "context_overflow"}, "context_overflow", "runtime"),
        ("final_answer", None, None, {"error_type": "output_truncated"}, "output_truncated", "agent"),
        ("error", "empty", None, {"error_type": "empty_model_response"}, "empty_response", "backend"),
        ("tool_call_result", "bad call", None, {"error_type": "invalid_tool_call"}, "invalid_tool_call", "runtime"),
        ("tool_call_result", None, "CANCELLED", {"status": "cancelled"}, "cancelled", "runtime"),
    ],
)
def test_legacy_failure_mapping_uses_the_canonical_vocabulary(
    event_type: str,
    error: str | None,
    code: str | None,
    metadata: dict[str, object],
    expected_kind: str,
    expected_origin: str,
) -> None:
    envelope = legacy_failure_envelope(
        event_type=event_type,
        error=error,
        code=code,
        metadata=metadata,
    )

    assert envelope is not None
    payload = envelope.to_dict()
    assert set(payload).issuperset(FAILURE_ENVELOPE_REQUIRED_FIELDS)
    assert payload["kind"] == expected_kind
    assert payload["origin"] == expected_origin
    assert payload["telemetry_key"] == f"failure.{expected_kind}"


def test_denial_and_steering_are_not_collapsed_into_tool_error() -> None:
    denied = ToolCallResultEvent(
        call_id="call-denied",
        tool_name="write_file",
        metadata={"status": "approval_pending", "approval_id": "approval-1"},
    )
    steering = StatusEvent(
        content="synthesize now",
        metadata={"status": "runtime_steering", "runtime_category": "runtime_steering"},
    )

    assert denied.failure is not None
    assert denied.failure.kind == "tool_denied"
    assert denied.failure.inject_into_model_context is True
    assert steering.failure is not None
    assert steering.failure.kind == "runtime_steering"
    assert steering.failure.retry_policy == "none"


def test_event_classes_attach_a_typed_failure_without_changing_legacy_metadata() -> None:
    error = ErrorEvent(
        message="raw provider detail",
        metadata={"error_type": "backend_request_error", "recoverability": "retryable_after_repair"},
    )
    truncated = FinalAnswerEvent(
        content="partial answer",
        metadata={"error_type": "output_truncated", "runtime_category": "truncation"},
    )

    assert error.failure is not None
    assert error.failure.kind == "backend_error"
    assert error.failure.terminal is True
    assert error.failure.retry_policy == "automatic"
    assert "failure" not in error.metadata
    assert truncated.failure is not None
    assert truncated.failure.kind == "output_truncated"
    assert truncated.failure.terminal is True


def test_minor_extensions_round_trip_but_unknown_major_is_rejected() -> None:
    envelope = ErrorEvent(message="provider failed").failure
    assert envelope is not None
    payload = envelope.to_dict() | {"schema_version": "1.4", "future_minor_field": {"safe": True}}

    assert FailureEnvelope.from_mapping(payload).to_dict() == payload

    payload["schema_version"] = "2.0"
    with pytest.raises(FailureEnvelopeVersionError):
        FailureEnvelope.from_mapping(payload)
