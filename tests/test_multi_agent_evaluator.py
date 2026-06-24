from __future__ import annotations

from mochi.agents.multi_agent.evaluator import resolve_evidence_gate_status


def test_resolve_evidence_gate_status_verified() -> None:
    status = resolve_evidence_gate_status(verified=True)

    assert status.status == "verified"
    assert status.reason is None


def test_resolve_evidence_gate_status_skipped() -> None:
    status = resolve_evidence_gate_status(skipped=True)

    assert status.status == "skipped"
    assert status.reason is None


def test_resolve_evidence_gate_status_failed_from_error() -> None:
    status = resolve_evidence_gate_status(error="missing citation")

    assert status.status == "failed"
    assert status.reason == "missing citation"

