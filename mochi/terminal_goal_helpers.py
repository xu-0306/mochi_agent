from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

GoalExecutionMode = Literal["single_agent", "workflow"]
GoalCommandAction = Literal["help", "proposal", "status", "pause", "resume", "stop"]
GoalContinuationAction = Literal[
    "forward_guidance",
    "resume_then_forward",
    "refresh_then_forward",
    "manual_resolution_required",
    "blocked",
]

GOAL_PROPOSAL_REPLY_HELP = (
    "Reply `start`, `go ahead`, `proceed`, `yes`, or `run it` to launch it. "
    "Send another follow-up to revise it, or use `/chat` to step outside the goal lane."
)

_GOAL_TERMINAL_STATUSES = {
    "completed",
    "done",
    "succeeded",
    "cancelled",
    "canceled",
    "failed",
    "error",
}
_LINKED_RUN_TERMINAL_STATUSES = {
    "completed",
    "done",
    "succeeded",
    "failed",
    "cancelled",
    "canceled",
    "partial",
}


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    values: list[str] = []
    for item in value:
        normalized = _string_or_none(item)
        if normalized is not None:
            values.append(normalized)
    return values


def _unique_strings(values: list[str | None]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in values:
        if not isinstance(item, str):
            continue
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


@dataclass(frozen=True)
class ChatModeCommand:
    mode: Literal["workflow", "chat"]
    content: str


@dataclass(frozen=True)
class GoalCommand:
    action: GoalCommandAction
    content: str
    raw: str


@dataclass(frozen=True)
class GoalRoutingDecision:
    mode_command: ChatModeCommand | None
    goal_command: GoalCommand | None
    request_text: str
    workflow_mode_requested: bool
    workflow_proposal_requested: bool
    natural_language_goal_requested: bool
    active_goal_follow_up_requested: bool
    confirmation_requested: bool
    proposal_revision_requested: bool
    should_handle_goal_workflow_routing: bool


@dataclass(frozen=True)
class GoalContinuationDecision:
    action: GoalContinuationAction
    summary: str
    blocking: bool
    approval_ids: list[str]
    run_id: str | None


def empty_goal_session_state() -> dict[str, Any]:
    return {
        "active_goal_id": None,
        "active_goal_status": None,
        "execution_mode": None,
        "default_route": "chat",
        "last_goal_summary": None,
        "pending_proposal": None,
    }


def normalize_goal_session_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    objective = _string_or_none(value.get("objective"))
    execution_mode = normalize_goal_execution_mode(value.get("execution_mode"))
    if objective is None or execution_mode is None:
        return None
    return {
        "goal_id": _string_or_none(value.get("goal_id")),
        "objective": objective,
        "execution_mode": execution_mode,
        "protocol_id": _string_or_none(value.get("protocol_id")),
        "models": _string_list(value.get("models")),
        "role_summary": _string_or_none(value.get("role_summary")),
        "runtime_mode": _string_or_none(value.get("runtime_mode")),
        "risk_note": _string_or_none(value.get("risk_note")),
        "status": _string_or_none(value.get("status")),
    }


def normalize_goal_session_proposal(value: Any) -> dict[str, Any] | None:
    summary = normalize_goal_session_summary(value)
    if summary is None or not isinstance(value, dict):
        return None
    revision_index_raw = value.get("revision_index", 0)
    try:
        revision_index = max(0, int(revision_index_raw))
    except (TypeError, ValueError):
        revision_index = 0
    return {
        **summary,
        "proposal_id": _string_or_none(value.get("proposal_id"))
        or _string_or_none(value.get("id"))
        or f"goal-proposal-{int(datetime.now(tz=UTC).timestamp() * 1000)}",
        "revision_index": revision_index,
        "updated_at": _string_or_none(value.get("updated_at")) or datetime.now(tz=UTC).isoformat(),
    }


def normalize_goal_session_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return empty_goal_session_state()
    pending = normalize_goal_session_proposal(value.get("pending_proposal"))
    last_summary = normalize_goal_session_summary(value.get("last_goal_summary"))
    execution_mode = normalize_goal_execution_mode(value.get("execution_mode"))
    if execution_mode is None and pending is not None:
        execution_mode = pending["execution_mode"]
    if execution_mode is None and last_summary is not None:
        execution_mode = last_summary["execution_mode"]
    default_route = _string_or_none(value.get("default_route"))
    return {
        "active_goal_id": _string_or_none(value.get("active_goal_id")) or _string_or_none(value.get("goal_id")),
        "active_goal_status": _string_or_none(value.get("active_goal_status")) or _string_or_none(value.get("status")),
        "execution_mode": execution_mode,
        "default_route": default_route if default_route in {"chat", "goal", "workflow"} else "chat",
        "last_goal_summary": last_summary,
        "pending_proposal": pending,
    }


def normalize_goal_execution_mode(value: Any) -> GoalExecutionMode | None:
    normalized = _string_or_none(value)
    if normalized in {"single_agent", "workflow"}:
        return normalized
    return None


def is_goal_terminal_status(status: str | None) -> bool:
    return (status or "").strip().lower() in _GOAL_TERMINAL_STATUSES


def parse_chat_mode_command(value: str) -> ChatModeCommand | None:
    match = re.match(r"^/(workflow|chat)(?:\s+([\s\S]*))?$", value.strip(), flags=re.IGNORECASE)
    if match is None:
        return None
    return ChatModeCommand(
        mode=match.group(1).lower(),
        content=(match.group(2) or "").strip(),
    )


def parse_goal_command(value: str) -> GoalCommand | None:
    match = re.match(r"^/goal(?:\s+([\s\S]*))?$", value.strip(), flags=re.IGNORECASE)
    if match is None:
        return None
    content = (match.group(1) or "").strip()
    normalized = content.lower()
    if not normalized:
        return GoalCommand(action="help", content="", raw=value.strip())
    if normalized in {"status", "pause", "resume", "stop"}:
        return GoalCommand(action=normalized, content="", raw=value.strip())
    return GoalCommand(action="proposal", content=content, raw=value.strip())


def is_goal_confirmation_text(value: str) -> bool:
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    return normalized in {"start", "go ahead", "proceed", "yes", "run it"}


def is_natural_language_goal_request(value: str) -> bool:
    normalized = re.sub(r"\s+", " ", value.strip().lower())
    if not normalized:
        return False
    patterns = (
        r"\b(?:in the background|background task|background run)\b",
        r"\b(?:keep working on this|continue working on this|keep going on this)\b",
        r"\b(?:make progress while i(?:'m| am) away|work on this while i(?:'m| am) away)\b",
        r"\b(?:retry until|checkpointed|with checkpoints|save checkpoints)\b",
        r"\b(?:spend|work for|run for|continue for|for the next)\s+\d+\s*(?:min(?:ute)?s?|hours?|hrs?)\b",
        r"\b\d+\s*(?:min(?:ute)?s?|hours?|hrs?)\b.*\b(?:background|keep working|continue working|come back)\b",
        r"\b(?:keep at it|stay on this|come back with progress)\b",
    )
    return any(re.search(pattern, normalized) for pattern in patterns)


def resolve_goal_workflow_routing(
    *,
    text: str,
    has_pending_proposal: bool,
    has_active_goal: bool,
) -> GoalRoutingDecision:
    mode_command = parse_chat_mode_command(text)
    goal_command = parse_goal_command(text)
    request_text = (
        mode_command.content
        if mode_command is not None
        else goal_command.content
        if goal_command is not None and goal_command.action == "proposal"
        else text
    )
    workflow_mode_requested = mode_command is not None and mode_command.mode == "workflow"
    workflow_proposal_requested = workflow_mode_requested and len(request_text) > 0
    natural_language_goal_requested = (
        goal_command is None
        and mode_command is None
        and not has_pending_proposal
        and is_natural_language_goal_request(request_text)
    )
    active_goal_follow_up_requested = (
        goal_command is None
        and mode_command is None
        and not has_pending_proposal
        and has_active_goal
        and len(request_text.strip()) > 0
        and not natural_language_goal_requested
    )
    confirmation_requested = (
        goal_command is None
        and mode_command is None
        and has_pending_proposal
        and is_goal_confirmation_text(text)
    )
    proposal_revision_requested = (
        goal_command is None
        and mode_command is None
        and has_pending_proposal
        and not confirmation_requested
        and len(request_text.strip()) > 0
    )
    return GoalRoutingDecision(
        mode_command=mode_command,
        goal_command=goal_command,
        request_text=request_text,
        workflow_mode_requested=workflow_mode_requested,
        workflow_proposal_requested=workflow_proposal_requested,
        natural_language_goal_requested=natural_language_goal_requested,
        active_goal_follow_up_requested=active_goal_follow_up_requested,
        confirmation_requested=confirmation_requested,
        proposal_revision_requested=proposal_revision_requested,
        should_handle_goal_workflow_routing=(
            goal_command is not None
            or workflow_proposal_requested
            or natural_language_goal_requested
            or active_goal_follow_up_requested
            or confirmation_requested
            or proposal_revision_requested
        ),
    )


def detect_goal_execution_mode_hint(value: str) -> GoalExecutionMode | None:
    normalized = value.lower()
    if any(token in normalized for token in ("single agent", "single-agent", "solo agent", "one agent")):
        return "single_agent"
    if any(
        token in normalized
        for token in ("workflow", "debate", "distill", "planner", "executor", "controller", "evaluator")
    ):
        return "workflow"
    return None


def _suggest_workflow_roles_and_protocol(value: str) -> tuple[list[str], str]:
    normalized = value.lower()
    if any(token in normalized for token in ("compare", "comparison", "evaluate", "evaluation", "debate")):
        return ["debater_a", "debater_b", "judge"], "multi_agent_debate"
    if any(token in normalized for token in ("distill", "compression", "compress", "teacher", "student")):
        return ["teacher", "student", "evaluator"], "teacher_student_distill"
    if any(token in normalized for token in ("research", "investigate", "survey", "literature")):
        return ["planner", "researcher_a", "researcher_b", "synthesizer", "verifier"], "multi_agent_debate"
    return ["planner", "executor", "controller", "evaluator"], "controlled_subagent_execution"


def detect_goal_runtime_hint(value: str) -> str | None:
    duration_match = re.search(r"\b\d+\s*(?:min(?:ute)?s?|hour(?:s)?|hr|hrs)\b", value, flags=re.IGNORECASE)
    if duration_match is not None:
        return f"Requested duration: {duration_match.group(0)}"
    if re.search(r"schedule|cron|interval", value, flags=re.IGNORECASE):
        return "Scheduled goal execution requested"
    return None


def format_goal_role_summary(roles: list[str]) -> str | None:
    if not roles:
        return None
    return ", ".join(role.replace("_", " ") for role in roles)


def build_goal_proposal_state(
    objective: str,
    execution_mode: GoalExecutionMode,
    *,
    current_model: str | None,
    autonomy_mode: str | None,
    previous: dict[str, Any] | None = None,
    revision_text: str | None = None,
) -> dict[str, Any]:
    normalized_objective = objective.strip()
    revision_source = (revision_text or "").strip()
    effective_execution_mode = detect_goal_execution_mode_hint(revision_source) or execution_mode
    workflow_roles, default_protocol = _suggest_workflow_roles_and_protocol(revision_source or normalized_objective)
    runtime_hint = detect_goal_runtime_hint(revision_source or normalized_objective)
    primary_models = _unique_strings([current_model])[:3] if effective_execution_mode == "workflow" else _unique_strings([current_model])[:1]
    role_summary = (
        format_goal_role_summary(workflow_roles)
        if effective_execution_mode == "workflow"
        else "Primary agent continues the task directly with the current chat tools."
    )
    runtime_mode = runtime_hint or (
        "Workflow run starts immediately"
        if effective_execution_mode == "workflow"
        else "Single-agent long-running execution"
    )
    risk_note = (
        "Runtime actions may pause for approval before execution."
        if autonomy_mode == "strict"
        else "Riskier runtime actions may still require approval."
        if autonomy_mode == "trusted_workspace"
        else None
    )
    revision_index = -1
    if isinstance(previous, dict):
        try:
            revision_index = int(previous.get("revision_index", -1))
        except (TypeError, ValueError):
            revision_index = -1
    return {
        "goal_id": None,
        "proposal_id": (
            _string_or_none(previous.get("proposal_id")) if isinstance(previous, dict) else None
        )
        or f"goal-proposal-{int(datetime.now(tz=UTC).timestamp() * 1000)}",
        "objective": normalized_objective,
        "execution_mode": effective_execution_mode,
        "protocol_id": default_protocol if effective_execution_mode == "workflow" else None,
        "models": primary_models,
        "role_summary": role_summary,
        "runtime_mode": runtime_mode,
        "risk_note": risk_note,
        "status": None,
        "revision_index": revision_index + 1,
        "updated_at": datetime.now(tz=UTC).isoformat(),
    }


def build_goal_summary_from_goal(
    goal: dict[str, Any],
    fallback: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "goal_id": _string_or_none(goal.get("goal_id")) or _string_or_none(goal.get("id")) or (fallback or {}).get("goal_id"),
        "objective": _string_or_none(goal.get("objective")) or (fallback or {}).get("objective") or "",
        "execution_mode": normalize_goal_execution_mode(goal.get("execution_mode")) or (fallback or {}).get("execution_mode") or "workflow",
        "protocol_id": _string_or_none(goal.get("protocol_id")) or (fallback or {}).get("protocol_id"),
        "models": list((fallback or {}).get("models") or []),
        "role_summary": (fallback or {}).get("role_summary"),
        "runtime_mode": (fallback or {}).get("runtime_mode"),
        "risk_note": (fallback or {}).get("risk_note"),
        "status": _string_or_none(goal.get("status")) or (fallback or {}).get("status"),
    }


def goal_card_from_summary(
    summary: dict[str, Any],
    *,
    kind: Literal["proposal", "revised_proposal", "started"],
    label: str | None = None,
    goal_id: str | None = None,
    status: str | None = None,
    superseded: bool | None = None,
) -> dict[str, Any]:
    default_label = (
        "Goal started"
        if kind == "started"
        else "Revised goal proposal"
        if kind == "revised_proposal"
        else "Goal proposal"
    )
    return {
        "kind": kind,
        "label": label or default_label,
        "objective": summary.get("objective") or "",
        "executionMode": summary.get("execution_mode") or "workflow",
        "protocolId": summary.get("protocol_id"),
        "models": list(summary.get("models") or []),
        "roleSummary": summary.get("role_summary"),
        "runtimeMode": summary.get("runtime_mode"),
        "riskNote": summary.get("risk_note"),
        "goalId": goal_id if goal_id is not None else summary.get("goal_id"),
        "status": status if status is not None else summary.get("status"),
        "superseded": superseded,
    }


def resolve_goal_continuation_decision(health: dict[str, Any]) -> GoalContinuationDecision:
    recommendation = health.get("recommended_next_action") if isinstance(health, dict) else {}
    if not isinstance(recommendation, dict):
        recommendation = {}
    recommended_action = _string_or_none(recommendation.get("action"))
    recommended_summary = (
        _string_or_none(recommendation.get("summary"))
        or "The active goal needs operator attention before it can continue."
    )
    linked_agent_run = health.get("linked_agent_run") if isinstance(health.get("linked_agent_run"), dict) else {}
    current_attempt = health.get("current_attempt") if isinstance(health.get("current_attempt"), dict) else {}
    linked_run_id = _string_or_none(linked_agent_run.get("run_id")) or _string_or_none(current_attempt.get("agent_run_id"))
    linked_run_status = _string_or_none(linked_agent_run.get("status"))
    approval_state = health.get("approval_state") if isinstance(health.get("approval_state"), dict) else {}
    approval_ids = _string_list(recommendation.get("approval_ids") or approval_state.get("approval_ids"))

    if recommended_action == "resolve_approval":
        return GoalContinuationDecision(
            action="manual_resolution_required",
            summary=recommended_summary,
            blocking=True,
            approval_ids=approval_ids,
            run_id=linked_run_id,
        )
    if recommended_action == "inspect_runtime_budget":
        return GoalContinuationDecision(
            action="blocked",
            summary=recommended_summary,
            blocking=True,
            approval_ids=[],
            run_id=linked_run_id,
        )
    if recommended_action == "refresh_worker_generation":
        return GoalContinuationDecision(
            action="refresh_then_forward",
            summary=recommended_summary,
            blocking=False,
            approval_ids=[],
            run_id=linked_run_id,
        )
    if recommended_action == "resume_goal":
        return GoalContinuationDecision(
            action="resume_then_forward",
            summary=recommended_summary,
            blocking=False,
            approval_ids=[],
            run_id=linked_run_id,
        )
    if recommended_action in {"monitor", "capture_checkpoint", "inspect_collector_shards"}:
        return GoalContinuationDecision(
            action="forward_guidance",
            summary=recommended_summary,
            blocking=False,
            approval_ids=[],
            run_id=linked_run_id,
        )
    if (
        not is_goal_terminal_status(_string_or_none(health.get("status")))
        and (linked_run_id is None or (linked_run_status or "").lower() in _LINKED_RUN_TERMINAL_STATUSES)
    ):
        return GoalContinuationDecision(
            action="resume_then_forward",
            summary=(
                "The current goal attempt is no longer actively running, so a follow-up "
                "should reopen or advance the goal before forwarding guidance."
            ),
            blocking=False,
            approval_ids=[],
            run_id=linked_run_id,
        )
    return GoalContinuationDecision(
        action="forward_guidance",
        summary=recommended_summary,
        blocking=False,
        approval_ids=[],
        run_id=linked_run_id,
    )
