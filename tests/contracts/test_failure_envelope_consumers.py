from __future__ import annotations

import asyncio

import pytest

from mochi.agents.events import ErrorEvent
from mochi.agents.failures import FailureEnvelope, FailureEnvelopeVersionError
from mochi.sessions.store import SessionStore


def _mock_sse_consumer(payload: dict[str, object]) -> dict[str, object]:
    failure = payload["failure"]
    assert isinstance(failure, dict)
    return FailureEnvelope.from_mapping(failure).to_dict()


def _mock_ui_consumer(payload: dict[str, object]) -> dict[str, object]:
    failure = payload["failure"]
    assert isinstance(failure, dict)
    parsed = FailureEnvelope.from_mapping(failure)
    return {"telemetry_key": parsed.telemetry_key, "ui_hint": parsed.ui_hint, "failure": parsed.to_dict()}


def test_failure_envelope_is_identical_through_sse_persistence_and_ui(tmp_path) -> None:
    event = ErrorEvent(
        message="restricted upstream diagnostics",
        metadata={"error_type": "backend_request_error", "diagnostics_ref": "diag:failure-1"},
    )
    assert event.failure is not None
    envelope = event.failure.to_dict() | {"future_minor_field": {"accepted": True}}
    sse_payload: dict[str, object] = {
        "type": "error",
        "error": "A safe user-facing message.",
        "failure": envelope,
    }

    assert _mock_sse_consumer(sse_payload) == envelope
    store = SessionStore(tmp_path / "sessions")
    asyncio.run(store.save_event("failure-consumer", sse_payload))
    persisted = asyncio.run(store.load_session("failure-consumer"))[0]
    assert persisted["failure"] == envelope
    ui_projection = _mock_ui_consumer(persisted)
    assert ui_projection["failure"] == envelope
    assert ui_projection["telemetry_key"] == "failure.backend_error"
    assert ui_projection["ui_hint"] == "retry"


def test_mock_consumers_reject_an_unknown_failure_major() -> None:
    payload: dict[str, object] = {
        "type": "error",
        "failure": {
            "schema_version": "2.0",
            "kind": "backend_error",
            "origin": "backend",
            "recoverability": "manual_retry",
            "retry_policy": "manual",
            "terminal": True,
            "inject_into_model_context": False,
            "telemetry_key": "failure.backend_error",
            "ui_hint": "retry",
            "diagnostics_ref": None,
        },
    }

    with pytest.raises(FailureEnvelopeVersionError):
        _mock_sse_consumer(payload)
    with pytest.raises(FailureEnvelopeVersionError):
        _mock_ui_consumer(payload)
