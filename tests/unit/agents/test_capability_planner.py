from __future__ import annotations

from mochi.agents.capability_planner import (
    CAPABILITY_PLAN_VERSION,
    CapabilityPlanner,
    CatalogToolDescriptor,
    EnvironmentEligibility,
    ExecutionProfile,
    ToolEnvironmentEligibility,
)
from mochi.agents.turn_intent_contract import (
    ClarificationRequest,
    DeliverableContract,
    IntentAdvisory,
    TurnIntentContract,
)
from mochi.tools.datetime_tool import DateTimeTool

ALL_CAPABILITIES = frozenset(
    {
        "temporal_lookup",
        "open_world_lookup",
        "literature_research",
        "workspace_read",
        "workspace_write",
        "execution",
        "tool_discovery",
    }
)


def _tool(
    name: str,
    *capabilities: str,
    priority: int = 100,
    directly_exposable: bool = True,
    destructive: bool = False,
    read_only: bool = False,
) -> CatalogToolDescriptor:
    return CatalogToolDescriptor(
        name=name,
        capabilities=frozenset(capabilities),  # type: ignore[arg-type]
        exposure_priority=priority,
        directly_exposable=directly_exposable,
        destructive=destructive,
        read_only=read_only,
    )


def _contract(
    *,
    operations: frozenset[str],
    mutation_requirement: str = "unknown",
    deliverables: tuple[DeliverableContract, ...] = (),
    speech_act: str = "request_execution",
    clarification: ClarificationRequest | None = None,
    advisories: tuple[IntentAdvisory, ...] = (),
) -> TurnIntentContract:
    return TurnIntentContract(
        turn_id="turn-1",
        active_goal_id="goal-1",
        objective="resolved objective",
        current_speech_act=speech_act,  # type: ignore[arg-type]
        operations=operations,  # type: ignore[arg-type]
        deliverables=deliverables,
        resolved_references=(),
        positive_constraints=(),
        negative_constraints=(),
        mutation_requirement=mutation_requirement,  # type: ignore[arg-type]
        clarification=clarification,
        supersedes_previous_goal=False,
        cancels_active_goal=False,
        modifies_active_task=True,
        confidence=0.95,
        evidence=(),
        advisories=advisories,
    )


def _plan(
    contract: TurnIntentContract,
    catalog: tuple[CatalogToolDescriptor, ...],
    *,
    session_capabilities=ALL_CAPABILITIES,  # type: ignore[no-untyped-def]
    profile_capabilities=ALL_CAPABILITIES,  # type: ignore[no-untyped-def]
    environment_capabilities=ALL_CAPABILITIES,  # type: ignore[no-untyped-def]
    environment_tools: tuple[ToolEnvironmentEligibility, ...] = (),
    allowed_tools: frozenset[str] | None = None,
    denied_tools: frozenset[str] = frozenset(),
    semantic_fallback: bool = False,
):  # type: ignore[no-untyped-def]
    return CapabilityPlanner().plan(
        contract=contract,
        catalog=catalog,
        session_capabilities=session_capabilities,
        execution_profile=ExecutionProfile(capabilities=profile_capabilities),
        environment=EnvironmentEligibility(
            capabilities=environment_capabilities,
            tools=environment_tools,
        ),
        allowed_tools=allowed_tools,
        denied_tools=denied_tools,
        semantic_fallback=semantic_fallback,
    )


def test_composite_contract_plans_web_read_write_and_execution_together() -> None:
    report = DeliverableContract(
        kind="workspace_report",
        target_hint="reports/findings.md",
        acceptance_criteria=("contains sourced findings",),
        source_turn_ids=("turn-1",),
    )
    contract = _contract(
        operations=frozenset(
            {"open_world_lookup", "workspace_read", "workspace_write", "execution"}
        ),
        mutation_requirement="required",
        deliverables=(report,),
    )
    catalog = (
        _tool("web_search", "open_world_lookup", priority=20),
        _tool("file_read", "workspace_read", priority=10),
        _tool("file_write", "workspace_write", priority=10),
        _tool("file_edit", "workspace_write", priority=20),
        _tool("exec_command", "execution", priority=10),
        _tool("tool_search", "tool_discovery", read_only=True),
    )

    plan = _plan(contract, catalog)

    assert plan.plan_version == CAPABILITY_PLAN_VERSION
    assert plan.required_capabilities == frozenset(
        {"open_world_lookup", "workspace_read", "workspace_write", "execution"}
    )
    assert plan.exposed_tools == (
        "file_read",
        "web_search",
        "exec_command",
        "file_write",
    )
    assert plan.artifact_obligation.required is True
    assert plan.artifact_obligation.ready is True
    assert plan.artifact_obligation.requirements[0].target_hint == "reports/findings.md"
    write_diagnostic = next(
        item for item in plan.tool_diagnostics if item.tool_name == "file_write"
    )
    assert (
        "explicit_workspace_artifact_first_iteration"
        in write_diagnostic.include_reasons
    )
    assert next(
        item for item in plan.tool_diagnostics if item.tool_name == "file_edit"
    ).exclude_reasons == ("redundant_for_minimal_exposure",)


def test_capability_intersection_and_tool_allow_deny_are_fail_closed() -> None:
    contract = _contract(
        operations=frozenset({"workspace_read", "workspace_write", "execution"}),
        mutation_requirement="required",
        deliverables=(
            DeliverableContract(
                kind="workspace_artifact",
                target_hint="result.txt",
                source_turn_ids=("turn-1",),
            ),
        ),
    )
    catalog = (
        _tool("file_read", "workspace_read"),
        _tool("file_write", "workspace_write"),
        _tool("exec_command", "execution"),
    )

    plan = _plan(
        contract,
        catalog,
        profile_capabilities=frozenset({"workspace_read", "workspace_write"}),
        allowed_tools=frozenset({"file_read", "file_write", "exec_command"}),
        denied_tools=frozenset({"file_write"}),
    )

    assert plan.exposed_tools == ("file_read",)
    assert plan.unavailable_capabilities == frozenset({"execution"})
    diagnostics = {item.tool_name: item for item in plan.tool_diagnostics}
    assert "tool_explicitly_denied" in diagnostics["file_write"].exclude_reasons
    assert (
        "execution_profile_disallows_capability"
        in diagnostics["exec_command"].exclude_reasons
    )


def test_environment_and_sandbox_can_narrow_individual_tools() -> None:
    contract = _contract(operations=frozenset({"open_world_lookup"}))
    catalog = (
        _tool("web_search", "open_world_lookup", priority=10),
        _tool("web_fetch", "open_world_lookup", priority=20),
    )

    plan = _plan(
        contract,
        catalog,
        environment_tools=(
            ToolEnvironmentEligibility(
                tool_name="web_search",
                eligible=False,
                reason_code="network_backend_unavailable",
            ),
        ),
    )

    assert plan.eligible_tools == ("web_fetch",)
    assert plan.exposed_tools == ("web_fetch",)
    diagnostic = next(
        item for item in plan.tool_diagnostics if item.tool_name == "web_search"
    )
    assert diagnostic.exclude_reasons == ("network_backend_unavailable",)


def test_mutation_forbidden_excludes_mutation_even_when_advisory_requests_it() -> None:
    contract = _contract(
        operations=frozenset({"workspace_read"}),
        mutation_requirement="forbidden",
        advisories=(
            IntentAdvisory(
                label="workspace_write",
                confidence=0.99,
                rationale="classifier disagrees",
                recommended_operations=frozenset({"workspace_write"}),
                source_turn_ids=("turn-1",),
            ),
        ),
    )
    catalog = (
        _tool("file_read", "workspace_read"),
        _tool("file_write", "workspace_write"),
        _tool("dangerous_combo", "workspace_read", "workspace_write"),
    )

    plan = _plan(contract, catalog)

    assert plan.exposed_tools == ("file_read",)
    diagnostics = {item.tool_name: item for item in plan.tool_diagnostics}
    assert "mutation_forbidden" in diagnostics["file_write"].exclude_reasons
    assert "mutation_forbidden" in diagnostics["dangerous_combo"].exclude_reasons
    assert plan.artifact_obligation.required is False


def test_clarification_never_exposes_risky_tools() -> None:
    contract = _contract(
        operations=frozenset({"workspace_read", "workspace_write", "execution"}),
        mutation_requirement="unknown",
        deliverables=(
            DeliverableContract(
                kind="workspace_artifact",
                target_hint="unknown target",
                source_turn_ids=("turn-1",),
            ),
        ),
        clarification=ClarificationRequest(
            question="Which target should be changed?",
            missing_fields=("target",),
            source_turn_ids=("turn-1",),
        ),
    )
    catalog = (
        _tool("file_read", "workspace_read"),
        _tool("file_write", "workspace_write"),
        _tool("exec_command", "execution"),
    )

    plan = _plan(contract, catalog)

    assert plan.exposed_tools == ("file_read",)
    assert plan.artifact_obligation.required is True
    assert plan.artifact_obligation.ready is False
    diagnostics = {item.tool_name: item for item in plan.tool_diagnostics}
    assert (
        "clarification_blocks_risky_tool" in diagnostics["file_write"].exclude_reasons
    )
    assert (
        "clarification_blocks_risky_tool" in diagnostics["exec_command"].exclude_reasons
    )


def test_tool_discovery_side_question_is_catalog_only_without_obligation() -> None:
    contract = _contract(
        operations=frozenset({"tool_discovery", "workspace_read"}),
        mutation_requirement="forbidden",
        speech_act="side_question",
    )
    catalog = (
        _tool("tool_search", "tool_discovery", read_only=True),
        _tool("file_read", "workspace_read"),
        _tool("file_write", "workspace_write"),
    )

    plan = _plan(contract, catalog)

    assert plan.required_capabilities == frozenset({"tool_discovery"})
    assert plan.eligible_tools == ("tool_search",)
    assert plan.exposed_tools == ("tool_search",)
    assert plan.artifact_obligation.required is False
    assert plan.artifact_obligation.reason_codes == (
        "tool_discovery_has_no_mutation_obligation",
    )


def test_previously_satisfied_durable_deliverable_is_not_reopened() -> None:
    contract = _contract(
        operations=frozenset({"workspace_write"}),
        mutation_requirement="required",
        deliverables=(
            DeliverableContract(
                kind="workspace_artifact",
                target_hint="report.md",
                status="satisfied",
                source_turn_ids=("turn-prior",),
            ),
        ),
    )

    plan = _plan(contract, (_tool("file_write", "workspace_write"),))

    assert plan.artifact_obligation.required is False
    assert plan.artifact_obligation.reason_codes == (
        "no_workspace_artifact_required",
    )


def test_deferred_tool_is_eligible_but_not_exposed_before_activation() -> None:
    contract = _contract(operations=frozenset({"open_world_lookup"}))
    catalog = (
        _tool(
            "deferred_web_tool",
            "open_world_lookup",
            directly_exposable=False,
        ),
    )

    plan = _plan(contract, catalog)

    assert plan.eligible_tools == ("deferred_web_tool",)
    assert plan.exposed_tools == ()
    assert plan.tool_diagnostics[0].status == "eligible"
    assert plan.tool_diagnostics[0].exclude_reasons == (
        "schema_not_directly_exposable",
    )


def test_semantic_fallback_adds_only_safe_read_only_candidates_to_eligibility() -> None:
    contract = _contract(
        operations=frozenset({"conversation", "tool_discovery"}),
        mutation_requirement="forbidden",
    )
    catalog = (
        _tool("tool_search", "tool_discovery", read_only=True),
        CatalogToolDescriptor(
            name="lookup_anywhere",
            capabilities=frozenset({"open_world_lookup"}),
            read_only=True,
            directly_exposable=False,
        ),
        CatalogToolDescriptor(
            name="unsafe_lookup",
            capabilities=frozenset({"open_world_lookup", "execution"}),
            read_only=True,
        ),
        CatalogToolDescriptor(
            name="approval_lookup",
            capabilities=frozenset({"open_world_lookup"}),
            read_only=True,
            requires_approval=True,
            risk="elevated",
        ),
        CatalogToolDescriptor(
            name="write_tool",
            capabilities=frozenset({"workspace_write"}),
            read_only=False,
        ),
    )

    plan = _plan(contract, catalog, semantic_fallback=True)

    assert plan.required_capabilities == frozenset({"tool_discovery"})
    assert plan.eligible_tools == ("tool_search", "lookup_anywhere")
    assert plan.exposed_tools == ("tool_search",)
    diagnostics = {item.tool_name: item for item in plan.tool_diagnostics}
    assert diagnostics["lookup_anywhere"].include_reasons == (
        "semantic_fallback_safe_candidate",
    )
    assert diagnostics["unsafe_lookup"].status == "excluded"
    assert diagnostics["approval_lookup"].status == "excluded"
    assert diagnostics["write_tool"].status == "excluded"


def test_semantic_fallback_still_honors_tool_and_environment_ceilings() -> None:
    contract = _contract(
        operations=frozenset({"conversation", "tool_discovery"}),
        mutation_requirement="forbidden",
    )
    catalog = (
        _tool("tool_search", "tool_discovery", read_only=True),
        CatalogToolDescriptor(
            name="safe_lookup",
            capabilities=frozenset({"open_world_lookup"}),
            read_only=True,
            directly_exposable=False,
        ),
    )

    plan = _plan(
        contract,
        catalog,
        semantic_fallback=True,
        allowed_tools=frozenset({"tool_search", "safe_lookup"}),
        denied_tools=frozenset({"safe_lookup"}),
    )

    assert plan.eligible_tools == ("tool_search",)
    diagnostic = next(item for item in plan.tool_diagnostics if item.tool_name == "safe_lookup")
    assert "tool_explicitly_denied" in diagnostic.exclude_reasons


def test_semantic_fallback_rejects_unsafe_tools_even_when_capability_is_required() -> None:
    contract = _contract(
        operations=frozenset({"conversation", "tool_discovery"}),
        mutation_requirement="forbidden",
    )
    catalog = (
        _tool("tool_search", "tool_discovery", read_only=True),
        CatalogToolDescriptor(
            name="destructive_discovery",
            capabilities=frozenset({"tool_discovery"}),
            read_only=True,
            destructive=True,
        ),
        CatalogToolDescriptor(
            name="elevated_discovery",
            capabilities=frozenset({"tool_discovery"}),
            read_only=True,
            risk="elevated",
        ),
        CatalogToolDescriptor(
            name="update_plan",
            capabilities=frozenset(),
            read_only=True,
        ),
        CatalogToolDescriptor(
            name="reference_bound_lookup",
            capabilities=frozenset({"workspace_read"}),
            read_only=True,
            activation_requirements=frozenset({"tool_result_reference"}),
        ),
    )

    plan = _plan(contract, catalog, semantic_fallback=True)

    assert plan.eligible_tools == ("tool_search",)
    assert plan.exposed_tools == ("tool_search",)
    diagnostics = {item.tool_name: item for item in plan.tool_diagnostics}
    for tool_name in (
        "destructive_discovery",
        "elevated_discovery",
        "update_plan",
        "reference_bound_lookup",
    ):
        assert diagnostics[tool_name].status == "excluded"
        assert (
            "semantic_fallback_unsafe_tool"
            in diagnostics[tool_name].exclude_reasons
        )


def test_semantic_fallback_rejects_environment_ineligible_and_unlisted_tools() -> None:
    contract = _contract(
        operations=frozenset({"conversation", "tool_discovery"}),
        mutation_requirement="forbidden",
    )
    catalog = (
        _tool("tool_search", "tool_discovery", read_only=True),
        CatalogToolDescriptor(
            name="environment_blocked_lookup",
            capabilities=frozenset({"open_world_lookup"}),
            read_only=True,
            directly_exposable=False,
        ),
        CatalogToolDescriptor(
            name="unlisted_lookup",
            capabilities=frozenset({"open_world_lookup"}),
            read_only=True,
            directly_exposable=False,
        ),
    )

    plan = _plan(
        contract,
        catalog,
        semantic_fallback=True,
        allowed_tools=frozenset({"tool_search", "environment_blocked_lookup"}),
        environment_tools=(
            ToolEnvironmentEligibility(
                tool_name="environment_blocked_lookup",
                eligible=False,
                reason_code="sandbox_ineligible",
            ),
        ),
    )

    assert plan.eligible_tools == ("tool_search",)
    diagnostics = {item.tool_name: item for item in plan.tool_diagnostics}
    assert "sandbox_ineligible" in diagnostics[
        "environment_blocked_lookup"
    ].exclude_reasons
    assert "tool_not_in_allowlist" in diagnostics["unlisted_lookup"].exclude_reasons


def test_activation_requirements_are_normalized_from_tool_metadata() -> None:
    descriptor = CatalogToolDescriptor.from_capability_metadata(
        name="reference_bound_lookup",
        metadata={
            "capabilities": ["workspace_read"],
            "read_only": True,
            "activation_requirements": ["tool_result_reference"],
        },
    )

    assert descriptor.activation_requirements == frozenset(
        {"tool_result_reference"}
    )


def test_classifier_disagreement_or_absence_does_not_change_plan() -> None:
    base = _contract(
        operations=frozenset({"workspace_write"}),
        mutation_requirement="required",
        deliverables=(
            DeliverableContract(
                kind="workspace_artifact",
                target_hint="README.md",
                source_turn_ids=("turn-1",),
            ),
        ),
    )
    disagreement = base.with_advisories(
        (
            IntentAdvisory(
                label="tool_discovery",
                confidence=0.01,
                rationale="advisory outage fallback disagrees",
                recommended_operations=frozenset({"tool_discovery"}),
                source_turn_ids=("turn-1",),
            ),
        )
    )
    catalog = (
        _tool("file_write", "workspace_write"),
        _tool("tool_search", "tool_discovery"),
    )

    without_classifier = _plan(base, catalog)
    with_disagreement = _plan(disagreement, catalog)

    assert without_classifier.to_dict() == with_disagreement.to_dict()
    assert with_disagreement.exposed_tools == ("file_write",)


def test_existing_tool_capability_metadata_is_normalized_without_text_routing() -> None:
    web = CatalogToolDescriptor.from_capability_metadata(
        name="provider_search",
        metadata={
            "domains": ["web"],
            "retrieval_modes": ["search"],
            "preference_tags": ["open_web"],
            "read_only": True,
            "destructive": False,
            "open_world": True,
        },
    )
    write = CatalogToolDescriptor.from_capability_metadata(
        name="file_write",
        metadata={
            "domains": [],
            "retrieval_modes": [],
            "preference_tags": [],
            "read_only": False,
            "destructive": False,
            "open_world": False,
        },
    )

    assert web.capabilities == frozenset({"open_world_lookup"})
    assert web.domains == frozenset({"web"})
    assert write.capabilities == frozenset({"workspace_write"})
    assert write.mutating is True


def test_datetime_metadata_only_satisfies_temporal_lookup_contract() -> None:
    tool = DateTimeTool()
    datetime_descriptor = CatalogToolDescriptor.from_capability_metadata(
        name=tool.name,
        metadata=tool.tool_capabilities,
    )
    temporal_contract = _contract(
        operations=frozenset({"temporal_lookup"}),
        mutation_requirement="forbidden",
    )

    temporal_plan = _plan(
        temporal_contract,
        (datetime_descriptor, _tool("exec_command", "execution")),
    )
    execution_plan = _plan(
        _contract(
            operations=frozenset({"execution"}),
            mutation_requirement="forbidden",
        ),
        (datetime_descriptor, _tool("exec_command", "execution")),
    )

    assert datetime_descriptor.read_only is True
    assert datetime_descriptor.capabilities == frozenset({"temporal_lookup"})
    assert temporal_plan.eligible_tools == ("get_current_time",)
    assert temporal_plan.exposed_tools == ("get_current_time",)
    datetime_diagnostic = next(
        item
        for item in temporal_plan.tool_diagnostics
        if item.tool_name == "get_current_time"
    )
    assert datetime_diagnostic.status == "exposed"
    assert datetime_diagnostic.matched_capabilities == frozenset({"temporal_lookup"})
    assert execution_plan.eligible_tools == ("exec_command",)
    assert execution_plan.exposed_tools == ("exec_command",)
    execution_datetime_diagnostic = next(
        item
        for item in execution_plan.tool_diagnostics
        if item.tool_name == "get_current_time"
    )
    assert execution_datetime_diagnostic.status == "excluded"
    assert execution_datetime_diagnostic.exclude_reasons == (
        "capability_not_required",
    )
