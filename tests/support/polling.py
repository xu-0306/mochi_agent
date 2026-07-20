"""Bounded polling helpers for HTTP integration tests."""

from __future__ import annotations

import time
from collections.abc import Collection
from typing import Any

from fastapi.testclient import TestClient


def wait_for_status(
    client: TestClient,
    path: str,
    statuses: Collection[str],
    *,
    timeout_seconds: float = 4.0,
    poll_interval_seconds: float = 0.05,
    terminal_statuses: Collection[str] = (),
    resource_label: str | None = None,
) -> dict[str, Any]:
    """Poll one JSON endpoint until a desired or unexpected terminal status appears."""
    desired = {str(status) for status in statuses}
    terminals = {str(status) for status in terminal_statuses} - desired
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if poll_interval_seconds < 0:
        raise ValueError("poll_interval_seconds must not be negative")

    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] | None = None
    while True:
        response = client.get(path)
        assert response.status_code == 200, (
            f"Expected {path} to return 200 while polling; got {response.status_code}: "
            f"{response.text}"
        )
        payload = response.json()
        assert isinstance(payload, dict), f"Expected {path} to return a JSON object: {payload!r}"
        last_payload = payload
        status = str(payload.get("status") or "")
        if status in desired:
            return payload
        if status in terminals:
            label = resource_label or path
            raise AssertionError(
                f"{label} reached terminal status {status!r} before one of {sorted(desired)}; "
                f"payload={payload}"
            )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(poll_interval_seconds, remaining))

    label = resource_label or path
    raise AssertionError(
        f"Timed out waiting for {label} statuses {sorted(desired)}; last payload={last_payload}"
    )
