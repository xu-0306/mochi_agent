from __future__ import annotations

from mochi.api.tool_workflow_observability import (
    build_tool_workflow_observability,
)


def _exposure_metadata() -> dict[str, object]:
    return {
        "tool_exposure": {
            "exposed_tools": ["tool_search", "tool_activate"],
            "diagnostics": {
                "stages": [
                    {
                        "stage": "turn_contract_rollout",
                        "capability_plan": {
                            "plan_version": "capability-plan-v1",
                            "eligible_tools": ["tool_search", "file_write"],
                            "tool_diagnostics": [
                                {
                                    "tool_name": "tool_search",
                                    "status": "exposed",
                                    "include_reasons": ["matches_required_capability"],
                                    "exclude_reasons": [],
                                },
                                {
                                    "tool_name": "file_write",
                                    "status": "eligible",
                                    "include_reasons": ["matches_required_capability"],
                                    "exclude_reasons": ["schema_not_directly_exposable"],
                                },
                            ],
                        },
                    }
                ]
            },
        }
    }


def test_projection_keeps_catalog_activation_review_and_verification_distinct() -> None:
    events = [
        {
            "type": "tool_call_request",
            "call_id": "activate-1",
            "tool_name": "tool_activate",
            "arguments": {"tool_name": "file_write"},
        },
        {
            "type": "tool_call_result",
            "call_id": "activate-1",
            "tool_name": "tool_activate",
            "result": {
                "status": "tool_activated",
                "requested_tool": "file_write",
                "callable_this_turn": True,
                "activation_authorizes_tool_call": False,
            },
            "metadata": {
                **_exposure_metadata(),
                "status": "tool_activated",
            },
        },
        {
            "type": "tool_call_completed",
            "call_id": "write-1",
            "tool_name": "file_write",
            "arguments": {"path": "README.md", "content": "hello"},
            "result": {
                "operation_id": "operation-1",
                "changed_paths": ["README.md"],
                "verification_status": "verified",
            },
            "error": None,
            "metadata": {
                **_exposure_metadata(),
                "auto_review_decision": "allow",
                "auto_review_source": "reviewed_allow",
                "operation_id": "operation-1",
                "changed_paths": ["README.md"],
                "verification_status": "verified",
            },
        },
    ]
    projection = build_tool_workflow_observability(
        events=events,
        effective_policy={
            "policy_snapshot_id": "policy-1",
            "policy_version": "effective-policy:1",
            "source_chain": ["security_config", "session_override"],
            "autonomy_mode": "auto_review",
        },
        expected_policy_version="effective-policy:1",
    )

    inventory = projection["tool_inventory"]
    assert inventory["catalog_scope"] == "policy_eligible"
    assert "catalog" not in inventory
    assert inventory["policy_catalog"] == ["tool_search", "file_write"]
    assert inventory["eligible_tools"] == ["tool_search", "file_write"]
    assert inventory["exposed_tools"] == ["tool_search", "tool_activate"]

    activation = projection["activation"]["calls"][0]
    assert activation["status"] == "tool_activated"
    assert activation["callable_this_turn"] is True
    assert activation["activation_authorizes_tool_call"] is False

    review = projection["call_review"]["calls"][0]
    assert review["auto_review_decision"] == "allow"
    assert review["approval_status"] == "not_observed"
    assert review["target"] == "README.md"

    execution = projection["execution"]["calls"][0]
    assert execution["status"] == "completed"
    assert execution["operation_id"] == "operation-1"
    assert execution["verification_status"] == "verified"
    assert projection["effective_policy"]["expectation_status"] == "matches"


def test_projection_does_not_infer_review_or_verification_from_success() -> None:
    projection = build_tool_workflow_observability(
        events=[
            {
                "type": "tool_call_result",
                "call_id": "read-1",
                "tool_name": "file_read",
                "result": {"content": "hello"},
                "error": None,
                "metadata": _exposure_metadata(),
            }
        ],
        effective_policy={
            "policy_snapshot_id": "policy-2",
            "policy_version": "effective-policy:2",
            "autonomy_mode": "auto_review",
        },
        expected_policy_version="effective-policy:old",
    )

    review = projection["call_review"]["calls"][0]
    execution = projection["execution"]["calls"][0]
    assert review["auto_review_decision"] == "not_observed"
    assert review["approval_status"] == "not_observed"
    assert execution["status"] == "completed"
    assert execution["verification_status"] == "not_observed"
    assert projection["activation"]["status"] == "not_observed"
    assert projection["effective_policy"]["expectation_status"] == "stale"
