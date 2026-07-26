from __future__ import annotations

from typing import Any

from mochi.agents.capability_exposure_adapter import (
    CAPABILITY_EXPOSURE_ADAPTER_VERSION,
    ExposurePolicyCeilings,
    adapt_capability_plan_to_exposure,
)
from mochi.agents.capability_planner import ArtifactObligation, CapabilityPlan
from mochi.agents.tool_exposure import ToolExposurePlan
from mochi.agents.turn_intent_contract import (
    ClarificationRequest,
    DeliverableContract,
    TurnIntentContract,
)
from mochi.tools.base import BaseTool, ToolResult
from mochi.tools.registry import ToolRegistry


class _NamedTool(BaseTool):
    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Test tool {self._name}."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    async def execute(self, **kwargs: Any) -> ToolResult:
        del kwargs
        return ToolResult(output={"tool": self._name})


def _contract(
    *,
    operations: frozenset[str],
    mutation_requirement: str,
    clarification_needed: bool = False,
) -> TurnIntentContract:
    deliverables = (
        (
            DeliverableContract(
                kind="workspace_file",
                target_hint="report.md",
                source_turn_ids=("turn-1",),
            ),
        )
        if mutation_requirement == "required"
        else ()
    )
    return TurnIntentContract(
        turn_id="turn-1",
        active_goal_id=None,
        objective="complete the requested operation",
        current_speech_act="request_execution",
        operations=operations,  # type: ignore[arg-type]
        deliverables=deliverables,
        resolved_references=(),
        positive_constraints=(),
        negative_constraints=(),
        mutation_requirement=mutation_requirement,  # type: ignore[arg-type]
        clarification=(
            ClarificationRequest(
                question="Which file should be changed?",
                missing_fields=("deliverable.target_hint",),
                source_turn_ids=("turn-1",),
            )
            if clarification_needed
            else None
        ),
        supersedes_previous_goal=False,
        cancels_active_goal=False,
        modifies_active_task=False,
        confidence=0.95,
        evidence=(),
    )


def _capability_plan(
    *,
    contract: TurnIntentContract,
    required_capabilities: frozenset[str],
    exposed_tools: tuple[str, ...],
    eligible_tools: tuple[str, ...] | None = None,
) -> CapabilityPlan:
    artifact_required = contract.mutation_requirement == "required"
    return CapabilityPlan(
        turn_id=contract.turn_id,
        contract_version=contract.contract_version,
        required_capabilities=required_capabilities,  # type: ignore[arg-type]
        unavailable_capabilities=frozenset(),
        eligible_tools=eligible_tools or exposed_tools,
        exposed_tools=exposed_tools,
        artifact_obligation=ArtifactObligation(
            required=artifact_required,
            ready=artifact_required,
            requirements=(),
            reason_codes=("workspace_artifact_required",) if artifact_required else (),
        ),
        tool_diagnostics=(),
    )


def _baseline_plan(*, tool_names: list[str], limit: int = 8) -> ToolExposurePlan:
    return ToolExposurePlan(
        tool_names=tool_names,
        matched_groups=["baseline_policy"],
        limit=limit,
        discoverable_tool_names=[
            "file_read",
            "file_write",
            "file_edit",
            "apply_patch",
            "exec_command",
            "tool_search",
        ],
        workspace_bound=True,
        diagnostics={
            "workspace_write_obligation": {
                "required": False,
                "source": "baseline_policy",
            }
        },
    )


def test_contract_write_overrides_advisory_inquiry_baseline() -> None:
    contract = _contract(
        operations=frozenset({"workspace_write"}),
        mutation_requirement="required",
    )
    capability_plan = _capability_plan(
        contract=contract,
        required_capabilities=frozenset({"workspace_write"}),
        exposed_tools=("file_write",),
    )

    result = adapt_capability_plan_to_exposure(
        baseline_plan=_baseline_plan(tool_names=["tool_search"]),
        capability_plan=capability_plan,
        contract=contract,
    )

    assert result.tool_names == ["tool_search", "file_write"]
    adapter = result.diagnostics["capability_exposure_adapter"]
    assert adapter["adapter_version"] == CAPABILITY_EXPOSURE_ADAPTER_VERSION
    assert adapter["added_tools"] == ["file_write"]
    assert adapter["removed_tools"] == []
    assert result.diagnostics["workspace_write_obligation"]["required"] is True
    assert result.diagnostics["workspace_write_obligation"]["source"] == (
        "capability_plan"
    )


def test_enforce_composes_read_write_and_execution_capabilities() -> None:
    contract = _contract(
        operations=frozenset({"workspace_read", "workspace_write", "execution"}),
        mutation_requirement="required",
    )
    capability_plan = _capability_plan(
        contract=contract,
        required_capabilities=frozenset(
            {"workspace_read", "workspace_write", "execution"}
        ),
        exposed_tools=("file_read", "file_write", "exec_command"),
    )

    result = adapt_capability_plan_to_exposure(
        baseline_plan=_baseline_plan(tool_names=["file_read", "tool_search"]),
        capability_plan=capability_plan,
        contract=contract,
    )

    assert result.tool_names == [
        "file_read",
        "tool_search",
        "file_write",
        "exec_command",
    ]
    assert result.diagnostics["capability_exposure_adapter"]["added_tools"] == [
        "file_write",
        "exec_command",
    ]


def test_enforce_mutation_forbidden_removes_all_core_mutation_tools() -> None:
    contract = _contract(
        operations=frozenset({"workspace_read"}),
        mutation_requirement="forbidden",
    )
    capability_plan = _capability_plan(
        contract=contract,
        required_capabilities=frozenset({"workspace_read"}),
        exposed_tools=("file_read",),
    )

    result = adapt_capability_plan_to_exposure(
        baseline_plan=_baseline_plan(
            tool_names=["file_read", "file_write", "file_edit", "apply_patch"]
        ),
        capability_plan=capability_plan,
        contract=contract,
    )

    assert result.tool_names == ["file_read"]
    assert result.diagnostics["capability_exposure_adapter"]["removed_tools"] == [
        "file_write",
        "file_edit",
        "apply_patch",
    ]
    reasons = result.diagnostics["capability_exposure_adapter"]["reasons"]
    assert reasons["file_write"] == ["mutation_forbidden_by_contract"]


def test_disabled_tool_mode_cannot_be_expanded_by_capability_plan() -> None:
    contract = _contract(
        operations=frozenset({"workspace_write"}),
        mutation_requirement="required",
    )
    capability_plan = _capability_plan(
        contract=contract,
        required_capabilities=frozenset({"workspace_write"}),
        exposed_tools=("file_write",),
    )

    result = adapt_capability_plan_to_exposure(
        baseline_plan=_baseline_plan(tool_names=[]),
        capability_plan=capability_plan,
        contract=contract,
        ceilings=ExposurePolicyCeilings(tool_mode="disabled"),
    )

    assert result.tool_names == []
    adapter = result.diagnostics["capability_exposure_adapter"]
    assert adapter["added_tools"] == []
    assert adapter["reasons"]["file_write"] == [
        "capability_addition_blocked_by_hard_policy"
    ]


def test_hard_allow_deny_and_sandbox_ceilings_only_narrow() -> None:
    contract = _contract(
        operations=frozenset({"workspace_read", "workspace_write", "execution"}),
        mutation_requirement="required",
    )
    capability_plan = _capability_plan(
        contract=contract,
        required_capabilities=frozenset(
            {"workspace_read", "workspace_write", "execution"}
        ),
        exposed_tools=("file_read", "file_write", "exec_command"),
    )

    result = adapt_capability_plan_to_exposure(
        baseline_plan=_baseline_plan(tool_names=["file_read", "tool_search"]),
        capability_plan=capability_plan,
        contract=contract,
        ceilings=ExposurePolicyCeilings(
            allowed_tool_names=frozenset(
                {"file_read", "file_write", "exec_command", "tool_search"}
            ),
            denied_tool_names=frozenset({"exec_command"}),
            sandbox_eligible_tool_names=frozenset(
                {"file_read", "file_write", "tool_search"}
            ),
        ),
    )

    assert result.tool_names == ["file_read", "tool_search", "file_write"]
    reasons = result.diagnostics["capability_exposure_adapter"]["reasons"]
    assert reasons["exec_command"] == [
        "capability_addition_blocked_by_hard_policy"
    ]


def test_pending_clarification_suppresses_baseline_mutation_and_execution() -> None:
    contract = _contract(
        operations=frozenset({"conversation"}),
        mutation_requirement="unknown",
        clarification_needed=True,
    )
    capability_plan = _capability_plan(
        contract=contract,
        required_capabilities=frozenset(),
        exposed_tools=(),
    )

    result = adapt_capability_plan_to_exposure(
        baseline_plan=_baseline_plan(
            tool_names=["file_read", "file_write", "exec_command", "tool_search"]
        ),
        capability_plan=capability_plan,
        contract=contract,
    )

    assert result.tool_names == ["tool_search"]
    reasons = result.diagnostics["capability_exposure_adapter"]["reasons"]
    assert reasons["file_write"] == [
        "clarification_pending_risky_tool_suppressed"
    ]
    assert reasons["exec_command"] == [
        "clarification_pending_risky_tool_suppressed"
    ]


def test_enforce_removes_baseline_business_tools_outside_capability_plan() -> None:
    contract = _contract(
        operations=frozenset({"workspace_read"}),
        mutation_requirement="unknown",
    )
    capability_plan = _capability_plan(
        contract=contract,
        required_capabilities=frozenset({"workspace_read"}),
        exposed_tools=("file_read",),
    )

    result = adapt_capability_plan_to_exposure(
        baseline_plan=_baseline_plan(
            tool_names=["exec_command", "web_search", "file_read", "tool_search"]
        ),
        capability_plan=capability_plan,
        contract=contract,
    )

    assert result.tool_names == ["file_read", "tool_search"]
    assert result.discoverable_tool_names == ["file_read", "tool_search"]
    adapter = result.diagnostics["capability_exposure_adapter"]
    assert adapter["activation_allowed_tool_names"] == ["file_read"]
    assert adapter["reasons"]["exec_command"] == [
        "not_exposed_by_capability_plan"
    ]
    assert adapter["reasons"]["web_search"] == [
        "not_exposed_by_capability_plan"
    ]


def test_enforce_reconciles_within_baseline_schema_budget() -> None:
    contract = _contract(
        operations=frozenset({"workspace_write"}),
        mutation_requirement="required",
    )
    capability_plan = _capability_plan(
        contract=contract,
        required_capabilities=frozenset({"workspace_write"}),
        exposed_tools=("file_write",),
    )

    result = adapt_capability_plan_to_exposure(
        baseline_plan=_baseline_plan(
            tool_names=["tool_search", "tool_result_read"],
            limit=2,
        ),
        capability_plan=capability_plan,
        contract=contract,
    )

    assert result.limit == 2
    assert result.tool_names == ["tool_search", "file_write"]
    adapter = result.diagnostics["capability_exposure_adapter"]
    assert adapter["added_tools"] == ["file_write"]
    assert adapter["removed_tools"] == ["tool_result_read"]
    assert adapter["reasons"]["tool_result_read"] == [
        "schema_budget_reconciled_for_capability_plan"
    ]


def test_enforce_reserves_broker_slot_in_actual_registry_view() -> None:
    contract = _contract(
        operations=frozenset({"workspace_write"}),
        mutation_requirement="required",
    )
    capability_plan = _capability_plan(
        contract=contract,
        required_capabilities=frozenset({"workspace_write"}),
        exposed_tools=("file_write",),
        eligible_tools=("file_write", "file_edit"),
    )
    result = adapt_capability_plan_to_exposure(
        baseline_plan=_baseline_plan(tool_names=["tool_search"], limit=2),
        capability_plan=capability_plan,
        contract=contract,
    )

    source_registry = ToolRegistry(discover_builtin=False)
    for name in ("tool_search", "file_write", "file_edit"):
        source_registry.register(_NamedTool(name))
    view = source_registry.create_view(
        result.tool_names,
        tool_search_catalog_names=result.discoverable_tool_names,
    )

    schema_names = [schema["function"]["name"] for schema in view.get_schemas()]
    assert "tool_activate" in schema_names
    assert len(schema_names) <= result.limit
    adapter = result.diagnostics["capability_exposure_adapter"]
    assert adapter["activation_broker"]["required"] is True
    assert adapter["schema_budget"]["expected_runtime_schema_count"] == 2
