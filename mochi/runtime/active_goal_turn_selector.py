from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Protocol
from uuid import uuid4

from mochi.agents.invocation import AgentInvocationRequest
from mochi.runtime.models import ActiveGoalTurnDecision

_ALLOWED_ACTIVE_GOAL_TURN_KINDS = {
    "answer_question",
    "explain_goal_state",
    "steer",
    "replan",
    "lifecycle",
    "exit_to_chat",
    "clarify",
}

_ACTIVE_GOAL_TURN_SELECTOR_SYSTEM_PROMPT = """
You are an internal classifier for follow-up turns inside an already active Goal.

Classify only the user's latest message. Return strict JSON only using this exact schema:
{"kind":"answer_question|explain_goal_state|steer|replan|lifecycle|exit_to_chat|clarify","confidence":0.0,"selection_reason":"...","requires_confirmation":false}

Allowed kinds:
- answer_question: asks for explanation, progress, reasoning, or a contextual answer without changing the goal
- explain_goal_state: asks what the current blocked, waiting, paused, or running state means
- steer: gives a forward-looking instruction for how the active goal should continue
- replan: asks to revise scope, plan, strategy, or approach
- lifecycle: asks to start, pause, resume, stop, cancel, or otherwise control the goal lifecycle
- exit_to_chat: asks to leave the active-goal lane and continue in normal chat
- clarify: the user's intent is too ambiguous to route safely

Rules:
- Be conservative. Prefer clarify over a false positive mutating decision.
- Questions about progress, blocked state, approvals, or what the goal is doing are explanatory, not lifecycle.
- Only choose steer, replan, or lifecycle when the user clearly intends to change execution.
- Use requires_confirmation=true when a human confirmation is still needed before acting on a mutating interpretation.
- Confidence must be calibrated from 0.0 to 1.0. Reserve >=0.95 for very clear instructions.
""".strip()


class ActiveGoalTurnSemanticInvoker(Protocol):
    async def invoke(self, request: AgentInvocationRequest) -> Any:
        """Run a bounded semantic classification invocation."""


def build_active_goal_turn_selector(engine: object) -> Any | None:
    invoke = getattr(engine, "invoke", None)
    if not callable(invoke):
        return None

    async def _select(context: Any) -> ActiveGoalTurnDecision | None:
        payload = _build_active_goal_turn_payload(context)
        if payload is None:
            return None
        invocation = AgentInvocationRequest(
            message=json.dumps(payload, ensure_ascii=False, indent=2),
            session_id=f"active-goal-turn:{uuid4()}",
            inference_overrides={
                "temperature": 0.0,
                "max_tokens": 200,
            },
            tool_mode="disabled",
            execution_profile="judge",
            system_prompt_addendum=_ACTIVE_GOAL_TURN_SELECTOR_SYSTEM_PROMPT,
            max_iterations_override=1,
            persist_session=False,
            persist_turn_events=False,
            persist_learning=False,
        )
        result = await invoke(invocation)
        content = str(getattr(result, "content", "") or "")
        fallback_decision = getattr(context, "fallback_decision", None)
        if not isinstance(fallback_decision, ActiveGoalTurnDecision):
            return None
        return parse_active_goal_turn_semantic_decision(
            content,
            fallback_decision=fallback_decision,
        )

    return _select


def parse_active_goal_turn_semantic_decision(
    text: str,
    *,
    fallback_decision: ActiveGoalTurnDecision,
) -> ActiveGoalTurnDecision | None:
    payload = _extract_json_object(text)
    if payload is None:
        return None

    lane = str(payload.get("lane") or "active_goal_turn").strip()
    if lane != "active_goal_turn":
        return None

    kind = str(payload.get("kind") or "").strip()
    if kind not in _ALLOWED_ACTIVE_GOAL_TURN_KINDS:
        return None

    confidence = _normalize_confidence(payload.get("confidence"))
    if confidence is None:
        return None

    selection_reason = (
        str(payload.get("selection_reason") or "").strip()
        or str(payload.get("rationale") or "").strip()
    )
    if not selection_reason:
        return None

    requires_confirmation = False
    if "requires_confirmation" in payload:
        normalized_confirmation = _normalize_bool(payload.get("requires_confirmation"))
        if normalized_confirmation is None:
            return None
        requires_confirmation = normalized_confirmation

    return ActiveGoalTurnDecision.model_validate(
        {
            "lane": "active_goal_turn",
            "kind": kind,
            "confidence": confidence,
            "selection_source": "semantic_registry_selector",
            "selection_reason": selection_reason,
            "requires_confirmation": requires_confirmation,
            "goal_status": fallback_decision.goal_status,
            "linked_run_status": fallback_decision.linked_run_status,
            "recommended_action": fallback_decision.recommended_action,
        }
    )


def _build_active_goal_turn_payload(context: Any) -> dict[str, Any] | None:
    fallback_decision = getattr(context, "fallback_decision", None)
    if not isinstance(fallback_decision, ActiveGoalTurnDecision):
        return None
    goal = _mapping_or_empty(getattr(context, "goal", None))
    health = _mapping_or_empty(getattr(context, "health", None))
    linked_run = _mapping_or_empty(health.get("linked_agent_run"))
    recommended_next_action = _mapping_or_empty(health.get("recommended_next_action"))
    approval_state = _mapping_or_empty(health.get("approval_state"))

    return {
        "goal": {
            "goal_id": str(goal.get("goal_id") or ""),
            "objective": str(goal.get("objective") or ""),
            "status": str(goal.get("status") or ""),
            "current_attempt_id": str(goal.get("current_attempt_id") or ""),
        },
        "health": {
            "status": str(health.get("status") or ""),
            "linked_agent_run": {
                "run_id": str(linked_run.get("run_id") or ""),
                "status": str(linked_run.get("status") or ""),
                "latest_error": str(linked_run.get("latest_error") or ""),
            },
            "recommended_next_action": {
                "action": str(recommended_next_action.get("action") or ""),
                "reason": str(recommended_next_action.get("reason") or ""),
            },
            "approval_state": {
                "status": str(approval_state.get("status") or ""),
                "pending_count": approval_state.get("pending_count"),
                "tool_names": approval_state.get("tool_names"),
            },
        },
        "fallback_decision": fallback_decision.model_dump(mode="python"),
        "user_message": str(getattr(context, "message", "") or ""),
    }


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _normalize_confidence(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
    elif isinstance(value, str):
        try:
            numeric = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if numeric < 0.0:
        return 0.0
    if numeric > 1.0:
        return 1.0
    return numeric


def _normalize_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    candidates = [stripped]
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.IGNORECASE)
    if fenced_match is not None:
        candidates.append(fenced_match.group(1).strip())

    brace_match = re.search(r"\{[\s\S]*\}", stripped)
    if brace_match is not None:
        candidates.append(brace_match.group(0).strip())

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None
