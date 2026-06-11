"""Helpers for normalized fallback diagnostics."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


def build_fallback_diagnostic(
    *,
    category: str,
    name: str,
    reason: str,
    kind: str = "fallback",
    severity: str = "warning",
    from_state: str | None = None,
    to_state: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create one normalized fallback diagnostic event."""
    payload: dict[str, Any] = {
        "kind": kind,
        "category": category,
        "name": name,
        "reason": reason,
        "severity": severity,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    if from_state is not None:
        payload["from"] = from_state
    if to_state is not None:
        payload["to"] = to_state
    if metadata:
        payload["metadata"] = dict(metadata)
    return payload


def append_fallback_diagnostic(
    diagnostics: list[dict[str, Any]],
    *,
    category: str,
    name: str,
    reason: str,
    kind: str = "fallback",
    severity: str = "warning",
    from_state: str | None = None,
    to_state: str | None = None,
    metadata: dict[str, Any] | None = None,
    limit: int = 20,
) -> dict[str, Any]:
    """Append one normalized fallback diagnostic event to a list."""
    payload = build_fallback_diagnostic(
        category=category,
        name=name,
        reason=reason,
        kind=kind,
        severity=severity,
        from_state=from_state,
        to_state=to_state,
        metadata=metadata,
    )
    diagnostics.append(payload)
    if limit > 0 and len(diagnostics) > limit:
        del diagnostics[:-limit]
    return payload
