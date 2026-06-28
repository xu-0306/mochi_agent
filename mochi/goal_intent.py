from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from uuid import uuid4

from mochi.agents.invocation import AgentInvocationRequest

GoalProposalFollowUpIntent = Literal[
    "confirm_start",
    "revise_proposal",
    "exit_goal_lane",
    "ambiguous",
]

_ALLOWED_INTENTS: set[str] = {
    "confirm_start",
    "revise_proposal",
    "exit_goal_lane",
    "ambiguous",
}
_CONFIRM_START_MIN_CONFIDENCE = 0.72
_SHORT_CONFIRMATION_MAX_LENGTH = 24

_GOAL_PROPOSAL_INTENT_SYSTEM_PROMPT = """
You are an internal intent classifier for Mochi's pending goal proposal flow.

Classify only the user's latest follow-up to a pending goal proposal.

Allowed intents:
- confirm_start: the user clearly wants to launch the pending proposal now
- revise_proposal: the user wants to change, clarify, narrow, question, postpone, or discuss the proposal before launch
- exit_goal_lane: the user wants to step out of goal setup and use normal chat instead
- ambiguous: the intent is not clear enough to safely choose another class

Rules:
- Be conservative. Prefer ambiguous over a false positive launch.
- Any hesitation, "before starting", scope change, question about the plan, or request for adjustment should be revise_proposal.
- A short confirmation in any language can be confirm_start if it clearly means "start now".
- Ignore normal conversational language-matching behavior. This is an internal classifier.
- Return strict JSON only using this exact schema:
  {"intent":"confirm_start|revise_proposal|exit_goal_lane|ambiguous","confidence":0.0,"rationale":"..."}
""".strip()

_DIRECT_CONFIRM_START_REPLIES = {
    "ok",
    "okay",
    "yes",
    "y",
    "yep",
    "sure",
    "confirm",
    "confirmed",
    "approve",
    "approved",
    "go",
    "goahead",
    "start",
    "startit",
    "launch",
    "launchit",
    "doit",
    "looksgood",
    "shipit",
    "\u597d",
    "\u597d\u7684",
    "\u597d\u554a",
    "\u597d\u5594",
    "\u53ef\u4ee5",
    "\u53ef\u4ee5\u4e86",
    "\u884c",
    "\u884c\u5427",
    "\u958b\u59cb",
    "\u958b\u59cb\u5427",
    "\u958b\u59cb\u57f7\u884c",
    "\u958b\u59cb\u4efb\u52d9",
    "\u555f\u52d5",
    "\u555f\u52d5\u5427",
    "\u555f\u52d5\u57f7\u884c",
    "\u8acb\u958b\u59cb",
    "\u76f4\u63a5\u958b\u59cb",
    "\u5c31\u9019\u6a23",
    "\u5c31\u9019\u6a23\u5427",
    "\u78ba\u8a8d\u958b\u59cb",
    "\u5f00\u59cb",
    "\u5f00\u59cb\u5427",
    "\u5f00\u59cb\u6267\u884c",
    "\u5f00\u59cb\u4efb\u52a1",
    "\u542f\u52a8",
    "\u542f\u52a8\u5427",
    "\u542f\u52a8\u6267\u884c",
    "\u8bf7\u5f00\u59cb",
    "\u76f4\u63a5\u5f00\u59cb",
    "\u5c31\u8fd9\u6837",
    "\u5c31\u8fd9\u6837\u5427",
    "\u786e\u8ba4\u5f00\u59cb",
    "\u306f\u3044",
    "\u958b\u59cb\u3057\u3066",
    "\u59cb\u3081\u3066",
    "\u9032\u3081\u3066",
    "\uc2dc\uc791",
    "\uc2dc\uc791\ud574",
    "\uc2dc\uc791\ud574\uc918",
    "\ub124",
    "\uc88b\uc544\uc694",
}


class GoalProposalIntentInvoker(Protocol):
    async def invoke(self, request: AgentInvocationRequest) -> Any:
        """Run a bounded internal invocation."""


@dataclass(frozen=True)
class GoalProposalFollowUpIntentResult:
    intent: GoalProposalFollowUpIntent
    confidence: float | None
    rationale: str


def _normalize_short_follow_up_for_rules(value: str) -> str:
    lowered = value.strip().casefold()
    return re.sub(r"[\s\.\,\!\?\-\_\:\;\'\"\`\~\(\)\[\]\{\}\<\>\/\\\|\u3000\u3001\u3002\uff0c\uff01\uff1f\uff1b\uff1a]+", "", lowered)


def _classify_follow_up_intent_by_rules(
    user_message: str,
) -> GoalProposalFollowUpIntentResult | None:
    trimmed = user_message.strip()
    if not trimmed or len(trimmed) > _SHORT_CONFIRMATION_MAX_LENGTH:
        return None

    normalized = _normalize_short_follow_up_for_rules(trimmed)
    if normalized in _DIRECT_CONFIRM_START_REPLIES:
        return GoalProposalFollowUpIntentResult(
            intent="confirm_start",
            confidence=1.0,
            rationale="Deterministic short confirmation matched a launch-now phrase.",
        )

    return None


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


def parse_goal_proposal_follow_up_intent_result(text: str) -> GoalProposalFollowUpIntentResult:
    payload = _extract_json_object(text)
    if payload is None:
        return GoalProposalFollowUpIntentResult(
            intent="ambiguous",
            confidence=None,
            rationale="Classifier did not return a valid JSON object.",
        )

    raw_intent = str(payload.get("intent") or "").strip()
    intent: GoalProposalFollowUpIntent = (
        raw_intent if raw_intent in _ALLOWED_INTENTS else "ambiguous"
    )  # type: ignore[assignment]
    confidence = _normalize_confidence(payload.get("confidence"))
    rationale = str(payload.get("rationale") or "").strip() or "No classifier rationale was provided."

    if intent == "confirm_start" and (confidence is None or confidence < _CONFIRM_START_MIN_CONFIDENCE):
        return GoalProposalFollowUpIntentResult(
            intent="ambiguous",
            confidence=confidence,
            rationale=(
                "Classifier suggested launch but confidence was below the safe start threshold. "
                f"Original rationale: {rationale}"
            ),
        )

    if intent == "ambiguous" and confidence is None:
        confidence = 0.0

    return GoalProposalFollowUpIntentResult(
        intent=intent,
        confidence=confidence,
        rationale=rationale,
    )


async def classify_goal_proposal_follow_up_intent(
    invoker: GoalProposalIntentInvoker,
    *,
    user_message: str,
    proposal_objective: str,
    execution_mode: str,
) -> GoalProposalFollowUpIntentResult:
    direct_match = _classify_follow_up_intent_by_rules(user_message)
    if direct_match is not None:
        return direct_match

    payload = {
        "pending_proposal": {
            "objective": proposal_objective,
            "execution_mode": execution_mode,
        },
        "user_follow_up": user_message,
    }
    invocation = AgentInvocationRequest(
        message=json.dumps(payload, ensure_ascii=False, indent=2),
        session_id=f"goal-intent:{uuid4()}",
        inference_overrides={
            "temperature": 0.0,
            "max_tokens": 160,
        },
        tool_mode="disabled",
        execution_profile="judge",
        system_prompt_addendum=_GOAL_PROPOSAL_INTENT_SYSTEM_PROMPT,
        max_iterations_override=1,
        persist_session=False,
        persist_turn_events=False,
        persist_learning=False,
    )
    result = await invoker.invoke(invocation)
    content = str(getattr(result, "content", "") or "")
    return parse_goal_proposal_follow_up_intent_result(content)
