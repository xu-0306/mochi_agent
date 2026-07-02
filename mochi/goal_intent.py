from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol
from uuid import uuid4

from mochi.agents.invocation import AgentInvocationRequest
from mochi.runtime.goal_strategy_registry import (
    DEFAULT_GOAL_STRATEGY_ID,
    GoalStrategyRegistryEntryData,
)

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

_CONFIRM_START_PREFIXES = {
    "ok",
    "okay",
    "yes",
    "sure",
    "confirm",
    "confirmed",
    "approve",
    "approved",
    "\u597d",
    "\u597d\u7684",
    "\u597d\u554a",
    "\u597d\u5594",
    "\u53ef\u4ee5",
    "\u53ef\u4ee5\u4e86",
    "\u884c",
    "\u884c\u5427",
    "\u597d\u90a3",
    "\u597d\u5440",
    "\u90a3\u5c31",
    "\u5c31",
}

_CONFIRM_START_LAUNCH_TOKENS = {
    "go",
    "goahead",
    "start",
    "startit",
    "launch",
    "launchit",
    "doit",
    "\u958b\u59cb",
    "\u958b\u59cb\u5427",
    "\u958b\u59cb\u57f7\u884c",
    "\u958b\u59cb\u4efb\u52d9",
    "\u555f\u52d5",
    "\u555f\u52d5\u5427",
    "\u555f\u52d5\u57f7\u884c",
    "\u8acb\u958b\u59cb",
    "\u76f4\u63a5\u958b\u59cb",
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
    "\u786e\u8ba4\u5f00\u59cb",
}


class GoalProposalIntentInvoker(Protocol):
    async def invoke(self, request: AgentInvocationRequest) -> Any:
        """Run a bounded internal invocation."""


@dataclass(frozen=True)
class GoalProposalFollowUpIntentResult:
    intent: GoalProposalFollowUpIntent
    confidence: float | None
    rationale: str


GoalStrategySelectionSource = Literal[
    "explicit_override",
    "semantic_registry_selector",
    "safe_default",
]


@dataclass(frozen=True)
class GoalStrategySelectionResult:
    strategy_id: str
    protocol_id: str
    selection_source: GoalStrategySelectionSource
    selection_reason: str


_SEMANTIC_TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{2,}")
_SEMANTIC_STOP_WORDS = {
    "agent",
    "agents",
    "analysis",
    "and",
    "answer",
    "before",
    "between",
    "can",
    "compare",
    "completion",
    "default",
    "execution",
    "for",
    "from",
    "goal",
    "goals",
    "judge",
    "most",
    "mode",
    "needs",
    "one",
    "ordinary",
    "output",
    "outputs",
    "plan",
    "report",
    "review",
    "role",
    "roles",
    "single",
    "specialized",
    "strategy",
    "task",
    "tasks",
    "that",
    "the",
    "their",
    "this",
    "through",
    "until",
    "use",
    "user",
    "when",
    "where",
    "with",
    "without",
    "workflow",
    "worker",
}


def _normalize_short_follow_up_for_rules(value: str) -> str:
    lowered = value.strip().casefold()
    return re.sub(r"[\s\.\,\!\?\-\_\:\;\'\"\`\~\(\)\[\]\{\}\<\>\/\\\|\u3000\u3001\u3002\uff0c\uff01\uff1f\uff1b\uff1a]+", "", lowered)


def _semantic_tokens(value: str) -> set[str]:
    return {
        token
        for token in _SEMANTIC_TOKEN_PATTERN.findall(value.casefold())
        if token not in _SEMANTIC_STOP_WORDS
    }


def _selection_reason_from_entry(
    entry: GoalStrategyRegistryEntryData,
    *,
    evidence_source: str,
) -> str:
    guidance = (
        (entry.selection_guidance or "").strip()
        or entry.when_to_use.strip()
        or entry.description.strip()
    )
    return (
        f"Selected registry strategy {entry.id} ({entry.display_name}) from {evidence_source}: "
        f"{guidance}"
    )


def _default_goal_strategy_selection() -> GoalStrategySelectionResult:
    return GoalStrategySelectionResult(
        strategy_id=DEFAULT_GOAL_STRATEGY_ID,
        protocol_id=DEFAULT_GOAL_STRATEGY_ID,
        selection_source="safe_default",
        selection_reason="Defaulted to autonomous_single_agent because no explicit strategy was provided.",
    )


def _default_goal_strategy_selection_with_reason(reason: str) -> GoalStrategySelectionResult:
    return GoalStrategySelectionResult(
        strategy_id=DEFAULT_GOAL_STRATEGY_ID,
        protocol_id=DEFAULT_GOAL_STRATEGY_ID,
        selection_source="safe_default",
        selection_reason=reason,
    )


def select_goal_strategy_from_registry(
    *,
    objective: str,
    entries: tuple[GoalStrategyRegistryEntryData, ...] | list[GoalStrategyRegistryEntryData],
) -> GoalStrategySelectionResult:
    objective_tokens = _semantic_tokens(objective)
    if not objective_tokens:
        return _default_goal_strategy_selection()

    best_entry: GoalStrategyRegistryEntryData | None = None
    best_score = 0
    best_evidence_source = ""
    for entry in entries:
        entry_tokens = _semantic_tokens(
            " ".join(
                part
                for part in (
                    entry.id,
                    entry.name,
                    entry.display_name,
                    entry.description,
                    entry.when_to_use,
                    entry.selection_guidance or "",
                )
                if part
            )
        )
        if not entry_tokens:
            continue
        score = len(objective_tokens & entry_tokens)
        if score < 2:
            continue
        if score <= best_score:
            continue
        best_entry = entry
        best_score = score
        if objective_tokens & _semantic_tokens(entry.selection_guidance or ""):
            best_evidence_source = "selection_guidance"
        elif objective_tokens & _semantic_tokens(entry.when_to_use):
            best_evidence_source = "when_to_use"
        else:
            best_evidence_source = "description"

    if best_entry is None:
        return _default_goal_strategy_selection()

    if best_entry.requires_confirmation:
        return _default_goal_strategy_selection_with_reason(
            f"Defaulted to autonomous_single_agent because {best_entry.id} matched semantically but requires explicit confirmation."
        )

    if not best_entry.available or best_entry.deprecated:
        availability_note = (
            (best_entry.availability_reason or "").strip()
            or "the registry entry is unavailable or deprecated"
        )
        return _default_goal_strategy_selection_with_reason(
            (
                f"Defaulted to autonomous_single_agent because {best_entry.id} matched semantically "
                f"but {availability_note}."
            )
        )

    protocol_id = str(best_entry.protocol_id or best_entry.id).strip() or best_entry.id
    return GoalStrategySelectionResult(
        strategy_id=best_entry.id,
        protocol_id=protocol_id,
        selection_source="semantic_registry_selector",
        selection_reason=_selection_reason_from_entry(
            best_entry,
            evidence_source=best_evidence_source,
        ),
    )


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

    for prefix in _CONFIRM_START_PREFIXES:
        if not normalized.startswith(prefix):
            continue
        suffix = normalized[len(prefix) :]
        if suffix and suffix in _CONFIRM_START_LAUNCH_TOKENS:
            return GoalProposalFollowUpIntentResult(
                intent="confirm_start",
                confidence=1.0,
                rationale="Deterministic short confirmation matched an affirmation-plus-start phrase.",
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
