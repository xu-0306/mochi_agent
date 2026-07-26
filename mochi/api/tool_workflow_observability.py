"""Read-only projection of ordinary Chat tool workflow state."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def build_tool_workflow_observability(
    *,
    events: Sequence[Mapping[str, Any]],
    effective_policy: Mapping[str, Any],
    expected_policy_version: str | None = None,
) -> dict[str, Any]:
    """Project authoritative runtime facts without changing workflow state."""

    policy_version = _text(effective_policy.get("policy_version"))
    policy_expectation = "not_provided"
    if expected_policy_version:
        policy_expectation = (
            "matches" if expected_policy_version == policy_version else "stale"
        )

    exposure = _latest_tool_exposure(events)
    capability_plan = _capability_plan(exposure)
    tool_diagnostics = _record_list(capability_plan.get("tool_diagnostics"))
    policy_catalog = [
        name
        for item in tool_diagnostics
        if (name := _text(item.get("tool_name"))) is not None
    ]

    request_arguments: dict[str, dict[str, Any]] = {}
    activation_by_call: dict[str, dict[str, Any]] = {}
    execution_by_call: dict[str, dict[str, Any]] = {}
    review_by_call: dict[str, dict[str, Any]] = {}

    for index, event in enumerate(events):
        event_type = _text(event.get("type")) or ""
        call_id = _text(event.get("call_id")) or f"event-{index}"
        tool_name = _text(event.get("tool_name")) or ""
        arguments = _record(event.get("arguments"))
        if arguments:
            request_arguments[call_id] = arguments
        if event_type not in {"tool_call_completed", "tool_call_result"}:
            continue

        metadata = _record(event.get("metadata"))
        result = _record(event.get("result"))
        effective_arguments = arguments or request_arguments.get(call_id, {})
        if tool_name == "tool_activate":
            activation_by_call[call_id] = _activation_projection(
                call_id=call_id,
                metadata=metadata,
                result=result,
                arguments=effective_arguments,
                error=_text(event.get("error")),
            )
            continue

        review_by_call[call_id] = _review_projection(
            call_id=call_id,
            tool_name=tool_name,
            arguments=effective_arguments,
            metadata=metadata,
        )
        execution_by_call[call_id] = _execution_projection(
            call_id=call_id,
            tool_name=tool_name,
            metadata=metadata,
            result=result,
            error=_text(event.get("error")),
        )

    exposed_tools = _string_list(exposure.get("exposed_tools"))
    eligible_tools = _string_list(capability_plan.get("eligible_tools"))
    return {
        "schema_version": 1,
        "effective_policy": {
            **dict(effective_policy),
            "expectation_status": policy_expectation,
            "expected_policy_version": expected_policy_version,
            "review_semantics": "concrete_call_only",
        },
        "tool_inventory": {
            "catalog_scope": "policy_eligible",
            "policy_catalog": policy_catalog,
            "eligible_tools": eligible_tools,
            "exposed_tools": exposed_tools,
            "tool_diagnostics": tool_diagnostics,
        },
        "activation": {
            "status": "observed" if activation_by_call else "not_observed",
            "calls": list(activation_by_call.values()),
        },
        "call_review": {
            "status": "observed" if review_by_call else "not_observed",
            "calls": list(review_by_call.values()),
        },
        "execution": {
            "status": "observed" if execution_by_call else "not_observed",
            "calls": list(execution_by_call.values()),
        },
    }


def _latest_tool_exposure(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for event in reversed(events):
        metadata = _record(event.get("metadata"))
        exposure = _record(metadata.get("tool_exposure"))
        if exposure:
            return exposure
    return {}


def _capability_plan(exposure: Mapping[str, Any]) -> dict[str, Any]:
    diagnostics = _record(exposure.get("diagnostics"))
    for stage in reversed(_record_list(diagnostics.get("stages"))):
        if stage.get("stage") != "turn_contract_rollout":
            continue
        plan = _record(stage.get("capability_plan"))
        if plan:
            return plan
    return {}


def _activation_projection(
    *,
    call_id: str,
    metadata: Mapping[str, Any],
    result: Mapping[str, Any],
    arguments: Mapping[str, Any],
    error: str | None,
) -> dict[str, Any]:
    status = _text(metadata.get("status")) or _text(result.get("status"))
    return {
        "call_id": call_id,
        "requested_tool": (
            _text(metadata.get("requested_tool"))
            or _text(result.get("requested_tool"))
            or _text(arguments.get("tool_name"))
        ),
        "status": status or ("failed" if error else "not_observed"),
        "reason": _text(metadata.get("reason")) or _text(metadata.get("error_type")),
        "callable_this_turn": _optional_bool(
            metadata.get("callable_this_turn"),
            result.get("callable_this_turn"),
        ),
        "activation_authorizes_tool_call": _optional_bool(
            metadata.get("activation_authorizes_tool_call"),
            result.get("activation_authorizes_tool_call"),
        ),
    }


def _review_projection(
    *,
    call_id: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    auto_review_decision = _text(metadata.get("auto_review_decision"))
    approval_status = _text(metadata.get("approval_status")) or _text(
        metadata.get("approval_state")
    )
    review_status = auto_review_decision or approval_status or "not_observed"
    return {
        "call_id": call_id,
        "tool_name": tool_name,
        "arguments": dict(arguments),
        "target": _target_projection(arguments, metadata),
        "status": review_status,
        "approval_id": _text(metadata.get("approval_id")),
        "approval_status": approval_status or "not_observed",
        "auto_review_decision": auto_review_decision or "not_observed",
        "auto_review_source": _text(metadata.get("auto_review_source")),
    }


def _execution_projection(
    *,
    call_id: str,
    tool_name: str,
    metadata: Mapping[str, Any],
    result: Mapping[str, Any],
    error: str | None,
) -> dict[str, Any]:
    verification_status = (
        _text(metadata.get("verification_status"))
        or _text(result.get("verification_status"))
        or "not_observed"
    )
    return {
        "call_id": call_id,
        "tool_name": tool_name,
        "status": "failed" if error else "completed",
        "operation_id": _text(metadata.get("operation_id"))
        or _text(result.get("operation_id")),
        "changed_paths": _string_list(
            metadata.get("changed_paths") or result.get("changed_paths")
        ),
        "verification_status": verification_status,
        "retry_status": _text(metadata.get("retry_status")) or "not_observed",
        "blocker": _text(metadata.get("blocker")) or _text(metadata.get("reason")),
    }


def _target_projection(
    arguments: Mapping[str, Any], metadata: Mapping[str, Any]
) -> str | None:
    for key in ("resolved_target", "target", "path", "file_path", "workdir", "url"):
        value = _text(metadata.get(key)) or _text(arguments.get(key))
        if value:
            return value
    return None


def _record(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _record_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [_record(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _optional_bool(*values: Any) -> bool | None:
    for value in values:
        if isinstance(value, bool):
            return value
    return None
