"""Fixtures shared by runtime API integration tests."""

from __future__ import annotations

import pytest


@pytest.fixture(
    params=[
        pytest.param(
            ("observe", "allow_legacy", "would_reject_contract_unavailable", "not_enforced"),
            id="observe-non-blocking",
        ),
        pytest.param(
            ("enforce", "reject_contract_unavailable", None, "configured_unavailable"),
            id="enforce-policy-fail-closed",
        ),
    ]
)
def change_contract_path(
    request: pytest.FixtureRequest,
) -> tuple[str, str, str | None, str]:
    """Reusable observe/enforce contract paths for rollout-gate tests."""
    return request.param
