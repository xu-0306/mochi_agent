"""Apply an authoritative capability plan within deterministic exposure ceilings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mochi.agents.capability_planner import CapabilityPlan
from mochi.agents.tool_exposure import ToolExposurePlan
from mochi.agents.turn_intent_contract import TurnIntentContract

ToolMode = Literal["disabled", "auto", "required"]

CAPABILITY_EXPOSURE_ADAPTER_VERSION = "capability-exposure-adapter-v1"
_MUTATION_TOOLS = frozenset({"file_write", "file_edit", "apply_patch"})
_CLARIFICATION_BLOCKED_TOOLS = _MUTATION_TOOLS | frozenset(
    {
        "exec_command",
        "execute_code",
        "execute_code_v2",
        "write_stdin",
        "kill_session",
        "process_stop",
    }
)
_NEUTRAL_INFRASTRUCTURE_TOOLS = frozenset(
    {"tool_search", "tool_activate", "tool_result_read"}
)


@dataclass(frozen=True)
class ExposurePolicyCeilings:
    """Hard runtime policy bounds that the adapter may only narrow."""

    tool_mode: ToolMode = "auto"
    allowed_tool_names: frozenset[str] | None = None
    denied_tool_names: frozenset[str] = frozenset()
    sandbox_eligible_tool_names: frozenset[str] | None = None


_DEFAULT_EXPOSURE_POLICY_CEILINGS = ExposurePolicyCeilings()


def adapt_capability_plan_to_exposure(
    *,
    baseline_plan: ToolExposurePlan,
    capability_plan: CapabilityPlan,
    contract: TurnIntentContract,
    ceilings: ExposurePolicyCeilings = _DEFAULT_EXPOSURE_POLICY_CEILINGS,
) -> ToolExposurePlan:
    """Apply an authoritative capability plan without bypassing hard policy."""

    if capability_plan.turn_id != contract.turn_id:
        raise ValueError("capability plan turn_id does not match intent contract")
    if capability_plan.contract_version != contract.contract_version:
        raise ValueError("capability plan contract_version does not match intent contract")

    baseline_names = list(dict.fromkeys(baseline_plan.tool_names))
    authoritative_names = _enforced_tool_names(
        baseline_names=baseline_names,
        capability_plan=capability_plan,
        contract=contract,
        ceilings=ceilings,
    )
    candidate_names = _reconcile_schema_budget(
        candidate_names=authoritative_names,
        capability_plan=capability_plan,
        limit=baseline_plan.limit,
    )
    discoverable_tool_names = _enforced_discoverable_tool_names(
        baseline_plan=baseline_plan,
        final_names=candidate_names,
        capability_plan=capability_plan,
        contract=contract,
        ceilings=ceilings,
    )
    deferred_names = _deferred_tool_names(
        final_names=candidate_names,
        discoverable_tool_names=discoverable_tool_names,
    )
    activation_allowed = _activation_allowed_tool_names(
        capability_plan=capability_plan,
        contract=contract,
        ceilings=ceilings,
    )
    if not set(deferred_names).issubset(activation_allowed):
        raise RuntimeError("deferred tools must be activation-authorized")
    broker_required = bool(deferred_names)
    if broker_required:
        candidate_names = _reconcile_schema_budget(
            candidate_names=authoritative_names,
            capability_plan=capability_plan,
            limit=max(baseline_plan.limit - 1, 0),
        )
        discoverable_tool_names = _enforced_discoverable_tool_names(
            baseline_plan=baseline_plan,
            final_names=candidate_names,
            capability_plan=capability_plan,
            contract=contract,
            ceilings=ceilings,
        )
        deferred_names = _deferred_tool_names(
            final_names=candidate_names,
            discoverable_tool_names=discoverable_tool_names,
        )
        if not set(deferred_names).issubset(activation_allowed):
            raise RuntimeError("deferred tools must be activation-authorized")
        broker_required = bool(deferred_names)

    if baseline_plan.limit <= 0:
        candidate_names = []
        discoverable_tool_names = []
        deferred_names = []
        broker_required = False

    baseline_set = set(baseline_names)
    candidate_set = set(candidate_names)
    added = [name for name in candidate_names if name not in baseline_set]
    removed = [name for name in baseline_names if name not in candidate_set]
    reasons = _diff_reasons(
        added=added,
        removed=removed,
        capability_plan=capability_plan,
        contract=contract,
        ceilings=ceilings,
    )
    adapter_diagnostics = {
        "adapter_version": CAPABILITY_EXPOSURE_ADAPTER_VERSION,
        "mode": "enforce",
        "applied": True,
        "plan_version": capability_plan.plan_version,
        "contract_version": contract.contract_version,
        "turn_id": contract.turn_id,
        "added_tools": added,
        "removed_tools": removed,
        "reasons": reasons,
        "artifact_obligation": capability_plan.artifact_obligation.to_dict(),
        "required_capabilities": sorted(capability_plan.required_capabilities),
        "activation_allowed_tool_names": activation_allowed,
        "continuation_candidate_tool_names": [
            name
            for name in baseline_plan.discoverable_tool_names
            if name == "tool_result_read"
            and _within_hard_ceilings(name, ceilings=ceilings)
        ],
        "activation_broker": {
            "required": broker_required,
            "reserved_schema_slots": int(broker_required),
            "deferred_tool_names": deferred_names,
        },
        "schema_budget": {
            "limit": baseline_plan.limit,
            "authoritative_candidate_count": len(authoritative_names),
            "planned_schema_count": len(candidate_names),
            "reserved_runtime_schema_count": int(broker_required),
            "expected_runtime_schema_count": (
                len(candidate_names) + int(broker_required)
            ),
            "evicted_tools": [
                name for name in authoritative_names if name not in candidate_set
            ],
        },
        "unavailable_capabilities": sorted(
            capability_plan.unavailable_capabilities
        ),
        "policy_ceilings": {
            "tool_mode": ceilings.tool_mode,
            "allowed_tool_names": (
                sorted(ceilings.allowed_tool_names)
                if ceilings.allowed_tool_names is not None
                else None
            ),
            "denied_tool_names": sorted(ceilings.denied_tool_names),
            "sandbox_eligible_tool_names": (
                sorted(ceilings.sandbox_eligible_tool_names)
                if ceilings.sandbox_eligible_tool_names is not None
                else None
            ),
        },
    }
    diagnostics = dict(baseline_plan.diagnostics)
    diagnostics["capability_exposure_adapter"] = adapter_diagnostics
    diagnostics["workspace_write_obligation"] = {
        **capability_plan.artifact_obligation.to_dict(),
        "source": "capability_plan",
        "plan_version": capability_plan.plan_version,
    }
    return ToolExposurePlan(
        tool_names=candidate_names,
        matched_groups=list(baseline_plan.matched_groups),
        limit=baseline_plan.limit,
        discoverable_tool_names=discoverable_tool_names,
        workspace_bound=baseline_plan.workspace_bound,
        attachment_count=baseline_plan.attachment_count,
        diagnostics=diagnostics,
    )


def _enforced_tool_names(
    *,
    baseline_names: list[str],
    capability_plan: CapabilityPlan,
    contract: TurnIntentContract,
    ceilings: ExposurePolicyCeilings,
) -> list[str]:
    if ceilings.tool_mode == "disabled":
        return []

    authoritative_names = {
        *capability_plan.exposed_tools,
        *(name for name in baseline_names if name in _NEUTRAL_INFRASTRUCTURE_TOOLS),
    }
    names = [name for name in baseline_names if name in authoritative_names]
    for name in capability_plan.exposed_tools:
        if name in names:
            continue
        if contract.mutation_requirement == "forbidden" and name in _MUTATION_TOOLS:
            continue
        if contract.clarification_needed and name in _CLARIFICATION_BLOCKED_TOOLS:
            continue
        if not _within_hard_ceilings(name, ceilings=ceilings):
            continue
        names.append(name)

    filtered_names = [
        name
        for name in names
        if _within_hard_ceilings(name, ceilings=ceilings)
        and not (
            contract.mutation_requirement == "forbidden"
            and name in _MUTATION_TOOLS
        )
        and not (
            contract.clarification_needed
            and name in _CLARIFICATION_BLOCKED_TOOLS
        )
    ]
    return filtered_names


def _deferred_tool_names(
    *,
    final_names: list[str],
    discoverable_tool_names: list[str],
) -> list[str]:
    final_set = set(final_names)
    return [name for name in discoverable_tool_names if name not in final_set]


def _reconcile_schema_budget(
    *,
    candidate_names: list[str],
    capability_plan: CapabilityPlan,
    limit: int,
) -> list[str]:
    if limit <= 0:
        return []
    if len(candidate_names) <= limit:
        return candidate_names

    candidate_set = set(candidate_names)
    priority_order: list[str] = []
    for name in capability_plan.exposed_tools:
        if name in candidate_set and name not in priority_order:
            priority_order.append(name)
    for name in candidate_names:
        if name in _NEUTRAL_INFRASTRUCTURE_TOOLS and name not in priority_order:
            priority_order.append(name)
    for name in candidate_names:
        if name not in priority_order:
            priority_order.append(name)

    keep = set(priority_order[:limit])
    return [name for name in candidate_names if name in keep]


def _within_hard_ceilings(
    tool_name: str,
    *,
    ceilings: ExposurePolicyCeilings,
) -> bool:
    if ceilings.tool_mode == "disabled":
        return False
    if (
        ceilings.allowed_tool_names is not None
        and tool_name not in ceilings.allowed_tool_names
    ):
        return False
    if tool_name in ceilings.denied_tool_names:
        return False
    return not (
        ceilings.sandbox_eligible_tool_names is not None
        and tool_name not in ceilings.sandbox_eligible_tool_names
    )


def _activation_allowed_tool_names(
    *,
    capability_plan: CapabilityPlan,
    contract: TurnIntentContract,
    ceilings: ExposurePolicyCeilings,
) -> list[str]:
    return [
        name
        for name in capability_plan.eligible_tools
        if _within_hard_ceilings(name, ceilings=ceilings)
        and not (
            contract.mutation_requirement == "forbidden"
            and name in _MUTATION_TOOLS
        )
        and not (
            contract.clarification_needed
            and name in _CLARIFICATION_BLOCKED_TOOLS
        )
    ]


def _enforced_discoverable_tool_names(
    *,
    baseline_plan: ToolExposurePlan,
    final_names: list[str],
    capability_plan: CapabilityPlan,
    contract: TurnIntentContract,
    ceilings: ExposurePolicyCeilings,
) -> list[str]:
    allowed = set(
        _activation_allowed_tool_names(
            capability_plan=capability_plan,
            contract=contract,
            ceilings=ceilings,
        )
    )
    ordered_candidates = [
        *baseline_plan.discoverable_tool_names,
        *capability_plan.eligible_tools,
        *final_names,
    ]
    return list(dict.fromkeys(name for name in ordered_candidates if name in allowed))


def _diff_reasons(
    *,
    added: list[str],
    removed: list[str],
    capability_plan: CapabilityPlan,
    contract: TurnIntentContract,
    ceilings: ExposurePolicyCeilings,
) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = {}
    for name in added:
        reasons[name] = ["capability_plan_required_exposure"]
    for name in removed:
        tool_reasons: list[str] = []
        if ceilings.tool_mode == "disabled":
            tool_reasons.append("tool_mode_disabled")
        if contract.mutation_requirement == "forbidden" and name in _MUTATION_TOOLS:
            tool_reasons.append("mutation_forbidden_by_contract")
        if contract.clarification_needed and name in _CLARIFICATION_BLOCKED_TOOLS:
            tool_reasons.append("clarification_pending_risky_tool_suppressed")
        if (
            not tool_reasons
            and name not in capability_plan.exposed_tools
            and name not in _NEUTRAL_INFRASTRUCTURE_TOOLS
        ):
            tool_reasons.append("not_exposed_by_capability_plan")
        if (
            ceilings.allowed_tool_names is not None
            and name not in ceilings.allowed_tool_names
        ):
            tool_reasons.append("hard_allowlist_excluded")
        if name in ceilings.denied_tool_names:
            tool_reasons.append("hard_denylist_blocked")
        if (
            ceilings.sandbox_eligible_tool_names is not None
            and name not in ceilings.sandbox_eligible_tool_names
        ):
            tool_reasons.append("sandbox_ineligible")
        reasons[name] = tool_reasons or [
            "schema_budget_reconciled_for_capability_plan"
        ]
    for name in capability_plan.exposed_tools:
        if name in added or name in reasons:
            continue
        if not _within_hard_ceilings(name, ceilings=ceilings):
            reasons[name] = ["capability_addition_blocked_by_hard_policy"]
    return {name: reasons[name] for name in sorted(reasons)}
