"""Deterministic capability planning from a resolved turn contract.

This module is intentionally independent from the agent engine.  It translates
one authoritative :class:`TurnIntentContract` into an auditable tool exposure
plan, while policy and environment inputs can only narrow that plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, cast

from mochi.agents.turn_intent_contract import DeliverableContract, TurnIntentContract

PlannerCapability = Literal[
    "open_world_lookup",
    "literature_research",
    "workspace_read",
    "workspace_write",
    "execution",
    "tool_discovery",
]
ToolRisk = Literal["low", "elevated", "high"]
ToolPlanStatus = Literal["excluded", "eligible", "exposed"]

CAPABILITY_PLAN_VERSION = "capability-plan-v1"

_ALL_CAPABILITIES = frozenset(
    {
        "open_world_lookup",
        "literature_research",
        "workspace_read",
        "workspace_write",
        "execution",
        "tool_discovery",
    }
)
_OPERATION_CAPABILITIES: dict[str, frozenset[PlannerCapability]] = {
    "conversation": frozenset(),
    "open_world_lookup": frozenset({"open_world_lookup"}),
    "literature_research": frozenset({"literature_research"}),
    "workspace_read": frozenset({"workspace_read"}),
    "workspace_write": frozenset({"workspace_write"}),
    "execution": frozenset({"execution"}),
    "tool_discovery": frozenset({"tool_discovery"}),
}

# Exact stable tool identifiers bridge current BaseTool metadata that predates
# explicit planner capabilities.  This is catalog normalization, not natural
# language intent routing.  New adapters should pass explicit capabilities.
_LEGACY_TOOL_CAPABILITIES: dict[str, frozenset[PlannerCapability]] = {
    "file_read": frozenset({"workspace_read"}),
    "tool_result_read": frozenset({"workspace_read"}),
    "glob_search": frozenset({"workspace_read"}),
    "grep_search": frozenset({"workspace_read"}),
    "csv_read": frozenset({"workspace_read"}),
    "pdf_read": frozenset({"workspace_read"}),
    "docx_read": frozenset({"workspace_read"}),
    "notebook_read": frozenset({"workspace_read"}),
    "repo_map": frozenset({"workspace_read"}),
    "read_symbol": frozenset({"workspace_read"}),
    "file_write": frozenset({"workspace_write"}),
    "file_edit": frozenset({"workspace_write"}),
    "apply_patch": frozenset({"workspace_write"}),
    "exec_command": frozenset({"execution"}),
    "execute_code": frozenset({"execution"}),
    "execute_code_v2": frozenset({"execution"}),
    "write_stdin": frozenset({"execution"}),
    "kill_session": frozenset({"execution"}),
    "process_poll": frozenset({"execution"}),
    "process_stop": frozenset({"execution"}),
    "tool_search": frozenset({"tool_discovery"}),
}


def _clean_name(value: str, *, field_name: str) -> str:
    cleaned = str(value or "").strip()
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    return cleaned


def _clean_capabilities(
    values: frozenset[PlannerCapability], *, field_name: str
) -> frozenset[PlannerCapability]:
    invalid = {str(value) for value in values if value not in _ALL_CAPABILITIES}
    if invalid:
        raise ValueError(f"unsupported {field_name}: {sorted(invalid)}")
    return frozenset(values)


def _string_values(value: Any) -> frozenset[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(str(item).strip() for item in value if str(item).strip())


@dataclass(frozen=True)
class CatalogToolDescriptor:
    """Planner-facing projection of one catalog tool definition."""

    name: str
    capabilities: frozenset[PlannerCapability]
    domains: frozenset[str] = field(default_factory=frozenset)
    retrieval_modes: frozenset[str] = field(default_factory=frozenset)
    preference_tags: frozenset[str] = field(default_factory=frozenset)
    read_only: bool = False
    destructive: bool = False
    open_world: bool = False
    requires_approval: bool = False
    risk: ToolRisk = "low"
    directly_exposable: bool = True
    exposure_priority: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _clean_name(self.name, field_name="tool.name"))
        object.__setattr__(
            self,
            "capabilities",
            _clean_capabilities(self.capabilities, field_name="tool capabilities"),
        )
        if self.risk not in {"low", "elevated", "high"}:
            raise ValueError(f"unsupported tool risk: {self.risk!r}")
        if self.exposure_priority < 0:
            raise ValueError("exposure_priority must be non-negative")
        for field_name in ("domains", "retrieval_modes", "preference_tags"):
            values = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                frozenset(
                    _clean_name(value, field_name=f"tool.{field_name}")
                    for value in values
                ),
            )

    @property
    def mutating(self) -> bool:
        return self.destructive or "workspace_write" in self.capabilities

    @property
    def risky(self) -> bool:
        return (
            self.risk != "low"
            or self.requires_approval
            or self.mutating
            or "execution" in self.capabilities
        )

    @classmethod
    def from_capability_metadata(
        cls,
        *,
        name: str,
        metadata: Mapping[str, Any] | None = None,
        capabilities: frozenset[PlannerCapability] = frozenset(),
        requires_approval: bool = False,
        directly_exposable: bool = True,
        exposure_priority: int = 100,
        risk: ToolRisk = "low",
    ) -> CatalogToolDescriptor:
        """Normalize the vocabulary currently returned by BaseTool."""

        payload = metadata or {}
        domains = _string_values(payload.get("domains"))
        retrieval_modes = _string_values(payload.get("retrieval_modes"))
        normalized = set(capabilities)
        normalized.update(_LEGACY_TOOL_CAPABILITIES.get(name, frozenset()))

        raw_explicit = payload.get("capabilities", payload.get("operations"))
        for value in _string_values(raw_explicit):
            if value in _ALL_CAPABILITIES:
                normalized.add(cast(PlannerCapability, value))

        read_only = bool(payload.get("read_only", False))
        open_world = bool(payload.get("open_world", False))
        if "workspace" in domains and read_only:
            normalized.add("workspace_read")
        if "web" in domains or open_world:
            normalized.add("open_world_lookup")
        if "literature" in domains:
            normalized.add("literature_research")

        return cls(
            name=name,
            capabilities=frozenset(normalized),
            domains=domains,
            retrieval_modes=retrieval_modes,
            preference_tags=_string_values(payload.get("preference_tags")),
            read_only=read_only,
            destructive=bool(payload.get("destructive", False)),
            open_world=open_world,
            requires_approval=requires_approval,
            directly_exposable=directly_exposable,
            exposure_priority=exposure_priority,
            risk=risk,
        )


@dataclass(frozen=True)
class ExecutionProfile:
    """Capabilities and schema budget supported by the active model/runtime."""

    capabilities: frozenset[PlannerCapability]
    max_exposed_tools: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capabilities",
            _clean_capabilities(self.capabilities, field_name="profile capabilities"),
        )
        if self.max_exposed_tools < 0:
            raise ValueError("max_exposed_tools must be non-negative")


@dataclass(frozen=True)
class ToolEnvironmentEligibility:
    """Environment or sandbox decision for one catalog tool."""

    tool_name: str
    eligible: bool
    reason_code: str = "environment_tool_ineligible"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "tool_name", _clean_name(self.tool_name, field_name="tool_name")
        )
        object.__setattr__(
            self, "reason_code", _clean_name(self.reason_code, field_name="reason_code")
        )


@dataclass(frozen=True)
class EnvironmentEligibility:
    """Observed environment/sandbox capabilities and per-tool restrictions."""

    capabilities: frozenset[PlannerCapability]
    tools: tuple[ToolEnvironmentEligibility, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "capabilities",
            _clean_capabilities(
                self.capabilities, field_name="environment capabilities"
            ),
        )
        names = [item.tool_name for item in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("environment tool eligibility contains duplicate names")


@dataclass(frozen=True)
class ArtifactRequirement:
    kind: str
    target_hint: str | None
    acceptance_criteria: tuple[str, ...]
    source_turn_ids: tuple[str, ...]

    @classmethod
    def from_deliverable(cls, deliverable: DeliverableContract) -> ArtifactRequirement:
        return cls(
            kind=deliverable.kind,
            target_hint=deliverable.target_hint,
            acceptance_criteria=deliverable.acceptance_criteria,
            source_turn_ids=deliverable.source_turn_ids,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_hint": self.target_hint,
            "acceptance_criteria": list(self.acceptance_criteria),
            "source_turn_ids": list(self.source_turn_ids),
        }


@dataclass(frozen=True)
class ArtifactObligation:
    required: bool
    ready: bool
    requirements: tuple[ArtifactRequirement, ...]
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "required": self.required,
            "ready": self.ready,
            "requirements": [item.to_dict() for item in self.requirements],
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class ToolPlanDiagnostic:
    tool_name: str
    status: ToolPlanStatus
    matched_capabilities: frozenset[PlannerCapability]
    include_reasons: tuple[str, ...] = ()
    exclude_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "status": self.status,
            "matched_capabilities": sorted(self.matched_capabilities),
            "include_reasons": list(self.include_reasons),
            "exclude_reasons": list(self.exclude_reasons),
        }


@dataclass(frozen=True)
class CapabilityPlan:
    """Versioned, auditable output for one model iteration."""

    turn_id: str
    contract_version: str
    required_capabilities: frozenset[PlannerCapability]
    unavailable_capabilities: frozenset[PlannerCapability]
    eligible_tools: tuple[str, ...]
    exposed_tools: tuple[str, ...]
    artifact_obligation: ArtifactObligation
    tool_diagnostics: tuple[ToolPlanDiagnostic, ...]
    plan_version: str = CAPABILITY_PLAN_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "turn_id": self.turn_id,
            "contract_version": self.contract_version,
            "required_capabilities": sorted(self.required_capabilities),
            "unavailable_capabilities": sorted(self.unavailable_capabilities),
            "eligible_tools": list(self.eligible_tools),
            "exposed_tools": list(self.exposed_tools),
            "artifact_obligation": self.artifact_obligation.to_dict(),
            "tool_diagnostics": [item.to_dict() for item in self.tool_diagnostics],
        }


class CapabilityPlanner:
    """Build capability/exposure plans without interpreting natural language."""

    def plan(
        self,
        *,
        contract: TurnIntentContract,
        catalog: tuple[CatalogToolDescriptor, ...],
        session_capabilities: frozenset[PlannerCapability],
        execution_profile: ExecutionProfile,
        environment: EnvironmentEligibility,
        allowed_tools: frozenset[str] | None = None,
        denied_tools: frozenset[str] = frozenset(),
    ) -> CapabilityPlan:
        self._validate_inputs(
            catalog=catalog,
            session_capabilities=session_capabilities,
            allowed_tools=allowed_tools,
            denied_tools=denied_tools,
        )
        required = self._required_capabilities(contract)
        available = (
            required
            & session_capabilities
            & execution_profile.capabilities
            & environment.capabilities
        )
        unavailable = required - available
        environment_by_tool = {item.tool_name: item for item in environment.tools}

        eligible: list[CatalogToolDescriptor] = []
        diagnostics: dict[str, ToolPlanDiagnostic] = {}
        for tool in catalog:
            matched = tool.capabilities & required
            reasons = self._exclusion_reasons(
                tool=tool,
                matched=matched,
                required=required,
                session_capabilities=session_capabilities,
                profile_capabilities=execution_profile.capabilities,
                environment_capabilities=environment.capabilities,
                environment_decision=environment_by_tool.get(tool.name),
                allowed_tools=allowed_tools,
                denied_tools=denied_tools,
                contract=contract,
            )
            usable = matched & available
            if reasons or not usable:
                if not reasons:
                    reasons = ("required_capability_unavailable",)
                diagnostics[tool.name] = ToolPlanDiagnostic(
                    tool_name=tool.name,
                    status="excluded",
                    matched_capabilities=matched,
                    exclude_reasons=reasons,
                )
                continue
            eligible.append(tool)
            diagnostics[tool.name] = ToolPlanDiagnostic(
                tool_name=tool.name,
                status="eligible",
                matched_capabilities=usable,
                include_reasons=("matches_required_capability",),
            )

        exposed = self._minimal_exposure(
            eligible=eligible,
            required=available,
            limit=execution_profile.max_exposed_tools,
        )
        exposed_names = {tool.name for tool in exposed}
        artifact_obligation = self._artifact_obligation(contract)
        explicit_artifact = artifact_obligation.required and artifact_obligation.ready
        for tool in eligible:
            prior = diagnostics[tool.name]
            if tool.name in exposed_names:
                include_reasons = [
                    *prior.include_reasons,
                    "selected_minimal_capability_cover",
                ]
                if explicit_artifact and "workspace_write" in tool.capabilities:
                    include_reasons.append(
                        "explicit_workspace_artifact_first_iteration"
                    )
                diagnostics[tool.name] = ToolPlanDiagnostic(
                    tool_name=tool.name,
                    status="exposed",
                    matched_capabilities=prior.matched_capabilities,
                    include_reasons=tuple(include_reasons),
                )
                continue
            exclude_reasons = (
                ("schema_not_directly_exposable",)
                if not tool.directly_exposable
                else ("redundant_for_minimal_exposure",)
            )
            diagnostics[tool.name] = ToolPlanDiagnostic(
                tool_name=tool.name,
                status="eligible",
                matched_capabilities=prior.matched_capabilities,
                include_reasons=prior.include_reasons,
                exclude_reasons=exclude_reasons,
            )

        return CapabilityPlan(
            turn_id=contract.turn_id,
            contract_version=contract.contract_version,
            required_capabilities=required,
            unavailable_capabilities=unavailable,
            eligible_tools=tuple(tool.name for tool in eligible),
            exposed_tools=tuple(tool.name for tool in exposed),
            artifact_obligation=artifact_obligation,
            tool_diagnostics=tuple(diagnostics[tool.name] for tool in catalog),
        )

    @staticmethod
    def _validate_inputs(
        *,
        catalog: tuple[CatalogToolDescriptor, ...],
        session_capabilities: frozenset[PlannerCapability],
        allowed_tools: frozenset[str] | None,
        denied_tools: frozenset[str],
    ) -> None:
        _clean_capabilities(session_capabilities, field_name="session capabilities")
        names = [tool.name for tool in catalog]
        if len(names) != len(set(names)):
            raise ValueError("catalog contains duplicate tool names")
        if allowed_tools is not None:
            for name in allowed_tools:
                _clean_name(name, field_name="allowed tool")
        for name in denied_tools:
            _clean_name(name, field_name="denied tool")

    @staticmethod
    def _required_capabilities(
        contract: TurnIntentContract,
    ) -> frozenset[PlannerCapability]:
        required: set[PlannerCapability] = set()
        for operation in contract.operations:
            required.update(_OPERATION_CAPABILITIES[operation])
        if (
            contract.current_speech_act == "side_question"
            and "tool_discovery" in contract.operations
        ):
            return frozenset({"tool_discovery"})
        if contract.mutation_requirement == "forbidden":
            required.discard("workspace_write")
        return frozenset(required)

    @staticmethod
    def _exclusion_reasons(
        *,
        tool: CatalogToolDescriptor,
        matched: frozenset[PlannerCapability],
        required: frozenset[PlannerCapability],
        session_capabilities: frozenset[PlannerCapability],
        profile_capabilities: frozenset[PlannerCapability],
        environment_capabilities: frozenset[PlannerCapability],
        environment_decision: ToolEnvironmentEligibility | None,
        allowed_tools: frozenset[str] | None,
        denied_tools: frozenset[str],
        contract: TurnIntentContract,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        if not matched:
            reasons.append("capability_not_required")
        if matched and not matched & session_capabilities:
            reasons.append("session_capability_unavailable")
        if matched and not matched & profile_capabilities:
            reasons.append("execution_profile_disallows_capability")
        if matched and not matched & environment_capabilities:
            reasons.append("environment_capability_unavailable")
        if allowed_tools is not None and tool.name not in allowed_tools:
            reasons.append("tool_not_in_allowlist")
        if tool.name in denied_tools:
            reasons.append("tool_explicitly_denied")
        if environment_decision is not None and not environment_decision.eligible:
            reasons.append(environment_decision.reason_code)
        if contract.mutation_requirement == "forbidden" and tool.mutating:
            reasons.append("mutation_forbidden")
        if tool.mutating and "workspace_write" not in required:
            reasons.append("mutation_not_required")
        if "execution" in tool.capabilities and "execution" not in required:
            reasons.append("execution_not_required")
        if contract.clarification_needed and tool.risky:
            reasons.append("clarification_blocks_risky_tool")
        if (
            contract.current_speech_act == "side_question"
            and "tool_discovery" in contract.operations
            and tool.capabilities != frozenset({"tool_discovery"})
        ):
            reasons.append("tool_discovery_side_question_scope")
        return tuple(dict.fromkeys(reasons))

    @staticmethod
    def _minimal_exposure(
        *,
        eligible: list[CatalogToolDescriptor],
        required: frozenset[PlannerCapability],
        limit: int,
    ) -> tuple[CatalogToolDescriptor, ...]:
        uncovered = set(required)
        remaining = [tool for tool in eligible if tool.directly_exposable]
        selected: list[CatalogToolDescriptor] = []
        while uncovered and remaining and len(selected) < limit:
            candidates = [tool for tool in remaining if tool.capabilities & uncovered]
            if not candidates:
                break
            candidates.sort(
                key=lambda tool: (
                    -len(tool.capabilities & uncovered),
                    tool.risky,
                    tool.exposure_priority,
                    tool.name,
                )
            )
            chosen = candidates[0]
            selected.append(chosen)
            uncovered.difference_update(chosen.capabilities)
            remaining.remove(chosen)
        return tuple(selected)

    @staticmethod
    def _artifact_obligation(contract: TurnIntentContract) -> ArtifactObligation:
        discovery_side_question = (
            contract.current_speech_act == "side_question"
            and "tool_discovery" in contract.operations
        )
        deliverables = tuple(
            ArtifactRequirement.from_deliverable(deliverable)
            for deliverable in contract.deliverables
            if deliverable.required and deliverable.status in {"pending", "in_progress"}
        )
        required = (
            bool(deliverables)
            and "workspace_write" in contract.operations
            and contract.mutation_requirement != "forbidden"
            and not discovery_side_question
        )
        if not required:
            reason = (
                "tool_discovery_has_no_mutation_obligation"
                if discovery_side_question
                else "no_workspace_artifact_required"
            )
            return ArtifactObligation(
                required=False,
                ready=False,
                requirements=(),
                reason_codes=(reason,),
            )
        ready = not contract.clarification_needed
        return ArtifactObligation(
            required=True,
            ready=ready,
            requirements=deliverables,
            reason_codes=(
                ("workspace_artifact_required",)
                if ready
                else ("artifact_waiting_for_clarification",)
            ),
        )
