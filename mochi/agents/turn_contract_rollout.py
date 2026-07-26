"""Engine-facing helpers for TurnIntentContract rollout.

The helpers in this module adapt already-bounded prompt context and registered
tool metadata.  They do not interpret natural language, grant permissions, or
read persisted audit contracts as authorization inputs.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from typing import Any, Literal

from mochi.agents.capability_planner import (
    CapabilityPlan,
    CapabilityPlanner,
    CatalogToolDescriptor,
    EnvironmentEligibility,
    ExecutionProfile as CapabilityExecutionProfile,
    PlannerCapability,
)
from mochi.agents.context import PromptContext
from mochi.agents.conversation_resolver import (
    ConversationResolution,
    ConversationSummary,
    ConversationTurn,
)
from mochi.agents.conversation_state_store import ConversationStateLoadDiagnostics
from mochi.backends.types import Message
from mochi.tools.base import BaseTool


# Stable catalog preference for tools that cover the same planner capability.
# This is deliberately independent of user text; explicit skill preferences
# still take precedence over these defaults.
_DEFAULT_TOOL_EXPOSURE_PRIORITY: dict[str, int] = {
    "tool_search": 10,
    "repo_map": 10,
    "web_search": 5,
    "arxiv_search": 10,
    "file_write": 10,
    "exec_command": 10,
    "grep_search": 20,
    "semantic_scholar_search": 20,
    "file_read": 20,
    "file_edit": 20,
    "execute_code": 20,
    "read_symbol": 30,
    "crossref_search": 30,
    "glob_search": 30,
    "apply_patch": 30,
    "execute_code_v2": 30,
    "pubmed_search": 40,
    "csv_read": 40,
    "pdf_read": 50,
    "docx_read": 60,
    "notebook_read": 70,
    "tool_result_read": 80,
    "process_poll": 90,
    "write_stdin": 100,
    "process_stop": 110,
    "kill_session": 120,
}

TurnContractMode = Literal["enforce"]

_ALL_CAPABILITIES: frozenset[PlannerCapability] = frozenset(
    {
        "open_world_lookup",
        "literature_research",
        "workspace_read",
        "workspace_write",
        "execution",
        "tool_discovery",
    }
)
_PROFILE_CAPABILITIES: dict[str, frozenset[PlannerCapability]] = {
    "chat": _ALL_CAPABILITIES,
    "task": _ALL_CAPABILITIES,
    "subagent_readonly": frozenset({"workspace_read", "tool_discovery"}),
    "subagent_research": frozenset(
        {
            "open_world_lookup",
            "literature_research",
            "workspace_read",
            "tool_discovery",
        }
    ),
    "subagent_execution_request": frozenset(
        {
            "open_world_lookup",
            "literature_research",
            "workspace_read",
            "tool_discovery",
        }
    ),
    "controller_exec": frozenset(
        {
            "open_world_lookup",
            "literature_research",
            "workspace_read",
            "execution",
            "tool_discovery",
        }
    ),
    "judge": frozenset(
        {
            "open_world_lookup",
            "literature_research",
            "workspace_read",
            "tool_discovery",
        }
    ),
    "verifier": frozenset(
        {
            "open_world_lookup",
            "literature_research",
            "workspace_read",
            "tool_discovery",
        }
    ),
}
@dataclass(frozen=True)
class TurnContractRolloutResult:
    """One resolved contract and plan plus non-authoritative rollout diagnostics."""

    mode: TurnContractMode
    resolution: ConversationResolution
    capability_plan: CapabilityPlan
    state_load_diagnostics: ConversationStateLoadDiagnostics
    state_persist_error: str | None = None
    state_revision: int | None = None

    def diagnostics(self) -> dict[str, Any]:
        contract = self.resolution.contract
        return {
            "mode": self.mode,
            "resolver": dict(self.resolution.diagnostics),
            "state_load": {
                "status": self.state_load_diagnostics.status,
                "event_schema_version": self.state_load_diagnostics.event_schema_version,
                "event_index": self.state_load_diagnostics.event_index,
                "messages": list(self.state_load_diagnostics.messages),
            },
            "contract": contract.to_dict(),
            "capability_plan": self.capability_plan.to_dict(),
            "state_persist_error": self.state_persist_error,
            "state_revision": self.state_revision,
        }


def conversation_inputs_from_prompt_context(
    *,
    turn_id: str,
    current_message: str,
    prompt_context: PromptContext,
) -> tuple[ConversationTurn, tuple[ConversationTurn, ...], ConversationSummary | None]:
    """Project the already-bounded prompt context into resolver input types."""

    current = ConversationTurn(turn_id=turn_id, role="user", content=current_message)
    history: list[ConversationTurn] = []
    for position, message in enumerate(prompt_context.history):
        content = _message_content(message)
        if not content:
            continue
        history_id = _stable_history_id(position, message.role, content)
        history.append(
            ConversationTurn(
                turn_id=history_id,
                role=message.role,
                content=content,
            )
        )
    summary = (
        ConversationSummary(
            content=prompt_context.summary,
            source_turn_ids=(f"{turn_id}:summary",),
        )
        if isinstance(prompt_context.summary, str) and prompt_context.summary.strip()
        else None
    )
    return current, tuple(history), summary


def build_capability_plan(
    *,
    planner: CapabilityPlanner,
    resolution: ConversationResolution,
    available_tools: list[BaseTool],
    preferred_tool_names: list[str],
    policy_eligible_tool_names: set[str],
    execution_profile: str,
    tool_mode: str,
    workspace_mutation_eligible: bool,
    tool_allowlist: list[str] | None,
    tool_denylist: list[str] | None,
) -> CapabilityPlan:
    """Build a deterministic plan bounded by invocation-level ceilings."""

    preference_order = {
        name: index for index, name in enumerate(dict.fromkeys(preferred_tool_names))
    }
    catalog = tuple(
        _catalog_descriptor(
            tool,
            exposure_priority=preference_order.get(
                tool.name,
                len(preference_order)
                + 100
                + _DEFAULT_TOOL_EXPOSURE_PRIORITY.get(tool.name, 500),
            ),
        )
        for tool in available_tools
        if tool.name in policy_eligible_tool_names
    )
    profile_capabilities = _PROFILE_CAPABILITIES.get(
        execution_profile,
        frozenset(),
    )
    session_capabilities = frozenset() if tool_mode == "disabled" else _ALL_CAPABILITIES
    environment_capabilities = set(_ALL_CAPABILITIES)
    if not workspace_mutation_eligible:
        environment_capabilities.discard("workspace_write")
    return planner.plan(
        contract=resolution.contract,
        catalog=catalog,
        session_capabilities=session_capabilities,
        execution_profile=CapabilityExecutionProfile(
            capabilities=profile_capabilities,
        ),
        environment=EnvironmentEligibility(
            capabilities=frozenset(environment_capabilities)
        ),
        allowed_tools=(
            frozenset(tool_allowlist) if tool_allowlist is not None else None
        ),
        denied_tools=frozenset(tool_denylist or ()),
    )


def _catalog_descriptor(
    tool: BaseTool,
    *,
    exposure_priority: int,
) -> CatalogToolDescriptor:
    risk: Literal["low", "elevated", "high"]
    if tool.is_destructive:
        risk = "high"
    elif tool.requires_approval:
        risk = "elevated"
    else:
        risk = "low"
    return CatalogToolDescriptor.from_capability_metadata(
        name=tool.name,
        metadata=tool.tool_capabilities,
        exposure_priority=exposure_priority,
        requires_approval=tool.requires_approval,
        risk=risk,
    )


def _message_content(message: Message) -> str:
    content = str(message.content or "").strip()
    if content:
        return content
    if not message.tool_calls:
        return ""
    return json.dumps(
        [
            {
                "id": tool_call.id,
                "name": tool_call.name,
                "arguments": tool_call.arguments,
            }
            for tool_call in message.tool_calls
        ],
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def _stable_history_id(position: int, role: str, content: str) -> str:
    digest = hashlib.sha256(
        f"{position}\x00{role}\x00{content}".encode("utf-8", errors="replace")
    ).hexdigest()
    return f"history:{position}:{digest[:24]}"
