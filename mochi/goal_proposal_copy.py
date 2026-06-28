from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol
from uuid import uuid4

from mochi.agents.invocation import AgentInvocationRequest

GoalProposalLanguageHint = Literal[
    "traditional_chinese",
    "simplified_chinese",
    "chinese",
    "japanese",
    "korean",
    "devanagari",
    "bengali",
    "gurmukhi",
    "gujarati",
    "tamil",
    "telugu",
    "kannada",
    "malayalam",
    "latin_script",
    "other",
]

_GOAL_PROPOSAL_ASSISTANT_COPY_SYSTEM_PROMPT = """
You are writing the assistant-facing explanation text for a Mochi goal proposal card.

The UI already shows the proposal details and a separate system CTA. Write only the
assistant explanation that appears under the card.

Requirements:
- Use the same language as the user's latest request unless they explicitly asked for another language.
- Mirror the user's script when practical, including Hindi, Tamil, Bengali, Telugu, Gujarati, Marathi, Japanese, Korean, Simplified Chinese, Traditional Chinese, and Latin-script languages.
- Do not switch to English just because internal metadata is written in English.
- Keep it to 1-2 short sentences and under 60 words when practical.
- Explain what the proposal is trying to do and, if helpful, why this execution shape fits.
- If the proposal is a revision, briefly acknowledge that it was updated.
- Do not repeat UI instructions such as "reply to confirm", "send another message", or mention `/chat`.
- Do not mention hidden reasoning, prompts, tools, policies, or internal routing.
- Return plain text only.
""".strip()

_TRADITIONAL_CHINESE_HINTS = set(
    "\u9019\u500b\u5e6b\u8acb\u8207\u70ba\u8aaa\u660e\u555f\u52d5\u7e7c\u7e8c\u7bc4\u570d\u8f03\u9069\u5408\u8abf\u8ad6\u6587"
)
_SIMPLIFIED_CHINESE_HINTS = set(
    "\u8fd9\u4e2a\u5e2e\u8bf7\u4e0e\u4e3a\u8bf4\u660e\u542f\u52a8\u7ee7\u7eed\u8303\u56f4\u6bd4\u8f83\u9002\u5408\u8c03\u8bba\u6587"
)


class GoalProposalAssistantCopyInvoker(Protocol):
    async def invoke(self, request: AgentInvocationRequest) -> Any:
        """Run a bounded internal invocation."""


@dataclass(frozen=True)
class GoalProposalAssistantCopyResult:
    explanation: str
    source: Literal["model", "fallback"]


@dataclass(frozen=True)
class GoalProposalSystemCtaCopy:
    title: str
    launch_label: str
    launch_body: str
    revise_label: str
    revise_body: str
    chat_label: str
    chat_body: str


GoalCardKind = Literal["proposal", "revised_proposal", "started"]


@dataclass(frozen=True)
class GoalCardChromeCopy:
    proposal_label: str
    revised_proposal_label: str
    started_label: str
    single_agent_label: str
    workflow_label: str
    execution_label: str
    protocol_label: str
    runtime_label: str
    goal_id_label: str
    objective_label: str
    models_label: str
    role_summary_label: str
    risk_note_label: str
    superseded_label: str
    active_goal_label: str
    most_recent_goal_label: str
    goal_summary_label: str
    goal_blocked_label: str
    goal_status_label: str
    goal_updated_label: str
    goal_paused_label: str
    goal_resumed_label: str
    goal_stopped_label: str
    pending_summary_intro: str
    active_summary_intro: str
    recent_summary_intro: str


GoalLifecycleMessageKind = Literal[
    "goal_started",
    "goal_manage_hint",
    "pending_cleared",
    "no_active_goal",
    "status_fetched",
    "goal_paused",
    "goal_resumed",
    "goal_stopped",
]


def _contains_japanese_kana(text: str) -> bool:
    return any(
        ("\u3040" <= char <= "\u309f")
        or ("\u30a0" <= char <= "\u30ff")
        or ("\u31f0" <= char <= "\u31ff")
        or ("\uff66" <= char <= "\uff9f")
        for char in text
    )


def _contains_hangul(text: str) -> bool:
    return any(
        ("\u1100" <= char <= "\u11ff")
        or ("\u3130" <= char <= "\u318f")
        or ("\uac00" <= char <= "\ud7af")
        for char in text
    )


def _contains_ascii_letters(text: str) -> bool:
    return any(("A" <= char <= "Z") or ("a" <= char <= "z") for char in text)


def _contains_block(text: str, start: str, end: str) -> bool:
    return any(start <= char <= end for char in text)


def detect_goal_proposal_language_hint(message: str) -> GoalProposalLanguageHint:
    text = message.strip()
    if not text:
        return "latin_script"

    if _contains_japanese_kana(text):
        return "japanese"

    if _contains_hangul(text):
        return "korean"

    if any("\u4e00" <= char <= "\u9fff" for char in text):
        traditional_hits = sum(char in _TRADITIONAL_CHINESE_HINTS for char in text)
        simplified_hits = sum(char in _SIMPLIFIED_CHINESE_HINTS for char in text)
        if traditional_hits > simplified_hits:
            return "traditional_chinese"
        if simplified_hits > traditional_hits:
            return "simplified_chinese"
        return "chinese"

    if _contains_block(text, "\u0900", "\u097f"):
        return "devanagari"
    if _contains_block(text, "\u0980", "\u09ff"):
        return "bengali"
    if _contains_block(text, "\u0a00", "\u0a7f"):
        return "gurmukhi"
    if _contains_block(text, "\u0a80", "\u0aff"):
        return "gujarati"
    if _contains_block(text, "\u0b80", "\u0bff"):
        return "tamil"
    if _contains_block(text, "\u0c00", "\u0c7f"):
        return "telugu"
    if _contains_block(text, "\u0c80", "\u0cff"):
        return "kannada"
    if _contains_block(text, "\u0d00", "\u0d7f"):
        return "malayalam"

    if _contains_ascii_letters(text):
        return "latin_script"

    return "other"


def _normalize_explanation(value: str) -> str:
    text = " ".join(line.strip() for line in value.strip().splitlines() if line.strip())
    return text[:600].strip()


def _is_language_aligned(user_message: str, explanation: str) -> bool:
    user_hint = detect_goal_proposal_language_hint(user_message)
    explanation_hint = detect_goal_proposal_language_hint(explanation)

    if user_hint in {"traditional_chinese", "simplified_chinese", "chinese"}:
        return explanation_hint in {"traditional_chinese", "simplified_chinese", "chinese"}
    if user_hint in {
        "japanese",
        "korean",
        "devanagari",
        "bengali",
        "gurmukhi",
        "gujarati",
        "tamil",
        "telugu",
        "kannada",
        "malayalam",
    }:
        return explanation_hint == user_hint
    return True


def build_goal_proposal_assistant_copy_fallback(
    *,
    user_message: str,
    proposal_objective: str,
    execution_mode: str,
    protocol_selection: str | None,
    revision_index: int,
) -> str:
    del proposal_objective

    language_hint = detect_goal_proposal_language_hint(user_message)
    protocol = (protocol_selection or "").strip()
    updated = revision_index > 0

    if language_hint in {"traditional_chinese", "chinese"}:
        prefix = (
            "\u6211\u5df2\u4f9d\u7167\u4f60\u6700\u65b0\u7684\u65b9\u5411\u66f4\u65b0\u9019\u4efd goal \u63d0\u6848\u3002"
            if updated
            else "\u6211\u628a\u4f60\u7684\u9700\u6c42\u6574\u7406\u6210\u4e00\u4efd\u53ef\u4ee5\u76f4\u63a5\u555f\u52d5\u7684 goal \u63d0\u6848\u3002"
        )
        detail = (
            f"\u76ee\u524d\u6703\u4ee5 {protocol} \u4f5c\u70ba\u57f7\u884c\u65b9\u5f0f\u3002"
            if protocol
            else "\u9019\u500b\u7bc4\u570d\u8f03\u9069\u5408\u7528 workflow \u65b9\u5f0f\u57f7\u884c\u3002"
            if execution_mode == "workflow"
            else "\u9019\u500b\u7bc4\u570d\u8f03\u9069\u5408\u7528 single-agent \u9577\u4efb\u52d9\u65b9\u5f0f\u57f7\u884c\u3002"
        )
        return f"{prefix} {detail}".strip()

    if language_hint == "simplified_chinese":
        prefix = (
            "\u6211\u5df2\u6839\u636e\u4f60\u6700\u65b0\u7684\u65b9\u5411\u66f4\u65b0\u8fd9\u4efd goal \u63d0\u6848\u3002"
            if updated
            else "\u6211\u628a\u4f60\u7684\u9700\u6c42\u6574\u7406\u6210\u4e00\u4efd\u53ef\u4ee5\u76f4\u63a5\u542f\u52a8\u7684 goal \u63d0\u6848\u3002"
        )
        detail = (
            f"\u76ee\u524d\u4f1a\u4ee5 {protocol} \u4f5c\u4e3a\u6267\u884c\u65b9\u5f0f\u3002"
            if protocol
            else "\u8fd9\u4e2a\u8303\u56f4\u66f4\u9002\u5408\u7528 workflow \u65b9\u5f0f\u6267\u884c\u3002"
            if execution_mode == "workflow"
            else "\u8fd9\u4e2a\u8303\u56f4\u66f4\u9002\u5408\u7528 single-agent \u957f\u4efb\u52a1\u65b9\u5f0f\u6267\u884c\u3002"
        )
        return f"{prefix} {detail}".strip()

    prefix = (
        "I updated this goal proposal to match your latest direction."
        if updated
        else "I framed your request as a goal proposal that we can launch directly."
    )
    detail = (
        f"The current execution shape is anchored around {protocol}."
        if protocol
        else "This scope fits a workflow run best."
        if execution_mode == "workflow"
        else "This scope fits a single-agent long-running run best."
    )
    return f"{prefix} {detail}".strip()


def build_goal_proposal_system_cta_copy(
    *,
    user_message: str,
) -> GoalProposalSystemCtaCopy:
    language_hint = detect_goal_proposal_language_hint(user_message)

    if language_hint in {"traditional_chinese", "chinese"}:
        return GoalProposalSystemCtaCopy(
            title="\u4e0b\u4e00\u6b65",
            launch_label="\u555f\u52d5",
            launch_body="\u8981\u958b\u59cb\u57f7\u884c\u6642\uff0c\u8acb\u9001\u51fa\u4e00\u5247\u7c21\u77ed\u78ba\u8a8d\u8a0a\u606f\u3002",
            revise_label="\u4fee\u6539\u63d0\u6848",
            revise_body="\u5982\u679c\u8981\u7e2e\u5c0f\u3001\u8abf\u6574\u6216\u64f4\u5927\u7bc4\u570d\uff0c\u518d\u9001\u4e00\u5247\u8a0a\u606f\u5373\u53ef\u3002",
            chat_label="\u4e00\u822c\u804a\u5929",
            chat_body="\u5982\u679c\u4f60\u60f3\u5148\u8a0e\u8ad6\uff0c\u8acb\u7528 `/chat <request>` \u66ab\u6642\u96e2\u958b goal \u8a2d\u5b9a\u3002",
        )

    if language_hint == "simplified_chinese":
        return GoalProposalSystemCtaCopy(
            title="\u4e0b\u4e00\u6b65",
            launch_label="\u542f\u52a8",
            launch_body="\u8981\u5f00\u59cb\u6267\u884c\u65f6\uff0c\u8bf7\u53d1\u9001\u4e00\u6761\u7b80\u77ed\u786e\u8ba4\u6d88\u606f\u3002",
            revise_label="\u4fee\u6539\u63d0\u6848",
            revise_body="\u5982\u679c\u8981\u7f29\u5c0f\u3001\u8c03\u6574\u6216\u6269\u5927\u8303\u56f4\uff0c\u518d\u53d1\u9001\u4e00\u6761\u6d88\u606f\u5373\u53ef\u3002",
            chat_label="\u666e\u901a\u804a\u5929",
            chat_body="\u5982\u679c\u4f60\u60f3\u5148\u8ba8\u8bba\uff0c\u8bf7\u7528 `/chat <request>` \u6682\u65f6\u79bb\u5f00 goal \u8bbe\u7f6e\u3002",
        )

    return GoalProposalSystemCtaCopy(
        title="Next step",
        launch_label="Launch",
        launch_body="Send a short confirmation when you want execution to begin.",
        revise_label="Revise",
        revise_body="Send another message to narrow, change, or expand the draft.",
        chat_label="Chat",
        chat_body="Use `/chat <request>` to step outside goal setup.",
    )


def build_goal_lifecycle_message(
    *,
    user_message: str,
    kind: GoalLifecycleMessageKind,
) -> str:
    language_hint = detect_goal_proposal_language_hint(user_message)

    if language_hint in {"traditional_chinese", "chinese"}:
        mapping = {
            "goal_started": "Goal \u5df2\u555f\u52d5\u3002\u4f60\u53ef\u4ee5\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u4f86\u7ba1\u7406\u5b83\u3002",
            "goal_manage_hint": "\u4f60\u53ef\u4ee5\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u4f86\u7ba1\u7406\u76ee\u524d\u7684 goal\u3002",
            "pending_cleared": "\u5df2\u6e05\u9664\u9019\u4efd\u5f85\u78ba\u8a8d\u7684 goal \u63d0\u6848\u3002\u4f60\u53ef\u4ee5\u7528 `/goal <request>` \u6216 `/workflow <request>` \u91cd\u65b0\u958b\u4e00\u500b\u3002",
            "no_active_goal": "\u9019\u500b\u5c0d\u8a71\u76ee\u524d\u6c92\u6709\u7d81\u5b9a\u4efb\u4f55\u9032\u884c\u4e2d\u7684 goal\u3002\u8acb\u7528 `/goal <request>` \u6216 `/workflow <request>` \u958b\u59cb\u4e00\u500b\u65b0\u7684\u4efb\u52d9\u3002",
            "status_fetched": "\u6211\u5df2\u53d6\u56de\u6700\u65b0\u7684 goal \u72c0\u614b\u3002",
            "goal_paused": "\u6211\u5df2\u66ab\u505c\u9019\u500b\u9032\u884c\u4e2d\u7684 goal\u3002",
            "goal_resumed": "\u6211\u5df2\u6062\u5fa9\u9019\u500b goal \u7684\u57f7\u884c\u3002",
            "goal_stopped": "\u6211\u5df2\u505c\u6b62\u9019\u500b goal\u3002",
        }
        return mapping[kind]

    if language_hint == "simplified_chinese":
        mapping = {
            "goal_started": "Goal \u5df2\u542f\u52a8\u3002\u4f60\u53ef\u4ee5\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u6765\u7ba1\u7406\u5b83\u3002",
            "goal_manage_hint": "\u4f60\u53ef\u4ee5\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u6765\u7ba1\u7406\u5f53\u524d\u7684 goal\u3002",
            "pending_cleared": "\u5df2\u6e05\u9664\u8fd9\u4efd\u5f85\u786e\u8ba4\u7684 goal \u63d0\u6848\u3002\u4f60\u53ef\u4ee5\u7528 `/goal <request>` \u6216 `/workflow <request>` \u91cd\u65b0\u5f00\u4e00\u4e2a\u3002",
            "no_active_goal": "\u8fd9\u4e2a\u5bf9\u8bdd\u76ee\u524d\u6ca1\u6709\u7ed1\u5b9a\u4efb\u4f55\u8fdb\u884c\u4e2d\u7684 goal\u3002\u8bf7\u7528 `/goal <request>` \u6216 `/workflow <request>` \u5f00\u59cb\u4e00\u4e2a\u65b0\u7684\u4efb\u52a1\u3002",
            "status_fetched": "\u6211\u5df2\u53d6\u56de\u6700\u65b0\u7684 goal \u72b6\u6001\u3002",
            "goal_paused": "\u6211\u5df2\u6682\u505c\u8fd9\u4e2a\u8fdb\u884c\u4e2d\u7684 goal\u3002",
            "goal_resumed": "\u6211\u5df2\u6062\u590d\u8fd9\u4e2a goal \u7684\u6267\u884c\u3002",
            "goal_stopped": "\u6211\u5df2\u505c\u6b62\u8fd9\u4e2a goal\u3002",
        }
        return mapping[kind]

    mapping = {
        "goal_started": "Goal started. Use `/goal status`, `/goal pause`, `/goal resume`, or `/goal stop` to manage it.",
        "goal_manage_hint": "Use `/goal status`, `/goal pause`, `/goal resume`, or `/goal stop` to manage the active goal.",
        "pending_cleared": "Cleared the pending goal proposal. Start a new one with `/goal <request>` or `/workflow <request>`.",
        "no_active_goal": "No active goal is bound to this chat. Start one with `/goal <request>` or `/workflow <request>`.",
        "status_fetched": "Fetched the latest goal status.",
        "goal_paused": "Paused the active goal.",
        "goal_resumed": "Resumed the active goal.",
        "goal_stopped": "Stopped the active goal.",
    }
    return mapping[kind]


GoalFollowUpMessageKind = Literal[
    "active_goal_exists",
    "goal_help",
    "manual_resolution_required",
    "blocked",
    "no_live_attempt",
    "refreshed_forwarded",
    "resumed_forwarded",
    "forwarded",
]


def build_goal_card_chrome_copy(
    *,
    user_message: str,
) -> GoalCardChromeCopy:
    language_hint = detect_goal_proposal_language_hint(user_message)

    if language_hint in {"traditional_chinese", "chinese"}:
        return GoalCardChromeCopy(
            proposal_label="Goal \u63d0\u6848",
            revised_proposal_label="\u5df2\u66f4\u65b0\u7684 goal \u63d0\u6848",
            started_label="Goal \u5df2\u555f\u52d5",
            single_agent_label="\u55ae\u4ee3\u7406",
            workflow_label="\u5de5\u4f5c\u6d41",
            execution_label="\u57f7\u884c\u65b9\u5f0f",
            protocol_label="\u5354\u5b9a",
            runtime_label="\u57f7\u884c\u6a21\u5f0f",
            goal_id_label="Goal ID",
            objective_label="\u76ee\u6a19",
            models_label="\u6a21\u578b",
            role_summary_label="\u89d2\u8272\u6458\u8981",
            risk_note_label="\u98a8\u96aa\u63d0\u793a",
            superseded_label="\u5df2\u53d6\u4ee3",
            active_goal_label="\u9032\u884c\u4e2d\u7684 goal",
            most_recent_goal_label="\u6700\u8fd1\u7684 goal",
            goal_summary_label="Goal \u6458\u8981",
            goal_blocked_label="Goal \u53d7\u963b",
            goal_status_label="Goal \u72c0\u614b",
            goal_updated_label="Goal \u5df2\u66f4\u65b0",
            goal_paused_label="Goal \u5df2\u66ab\u505c",
            goal_resumed_label="Goal \u5df2\u6062\u5fa9",
            goal_stopped_label="Goal \u5df2\u505c\u6b62",
            pending_summary_intro="\u6b64\u5c0d\u8a71\u76ee\u524d\u6709\u4e00\u4efd\u5f85\u78ba\u8a8d\u7684 goal \u63d0\u6848\u3002",
            active_summary_intro="\u4ee5\u4e0b\u662f\u9019\u500b\u5c0d\u8a71\u76ee\u524d\u9032\u884c\u4e2d\u7684 goal \u6458\u8981\u3002",
            recent_summary_intro="\u4ee5\u4e0b\u662f\u9019\u500b\u5c0d\u8a71\u6700\u8fd1\u4e00\u6b21\u7684 goal \u6458\u8981\u3002",
        )

    if language_hint == "simplified_chinese":
        return GoalCardChromeCopy(
            proposal_label="Goal \u63d0\u6848",
            revised_proposal_label="\u5df2\u66f4\u65b0\u7684 goal \u63d0\u6848",
            started_label="Goal \u5df2\u542f\u52a8",
            single_agent_label="\u5355\u4ee3\u7406",
            workflow_label="\u5de5\u4f5c\u6d41",
            execution_label="\u6267\u884c\u65b9\u5f0f",
            protocol_label="\u534f\u8bae",
            runtime_label="\u6267\u884c\u6a21\u5f0f",
            goal_id_label="Goal ID",
            objective_label="\u76ee\u6807",
            models_label="\u6a21\u578b",
            role_summary_label="\u89d2\u8272\u6458\u8981",
            risk_note_label="\u98ce\u9669\u63d0\u793a",
            superseded_label="\u5df2\u66ff\u4ee3",
            active_goal_label="\u8fdb\u884c\u4e2d\u7684 goal",
            most_recent_goal_label="\u6700\u8fd1\u7684 goal",
            goal_summary_label="Goal \u6458\u8981",
            goal_blocked_label="Goal \u53d7\u963b",
            goal_status_label="Goal \u72b6\u6001",
            goal_updated_label="Goal \u5df2\u66f4\u65b0",
            goal_paused_label="Goal \u5df2\u6682\u505c",
            goal_resumed_label="Goal \u5df2\u6062\u590d",
            goal_stopped_label="Goal \u5df2\u505c\u6b62",
            pending_summary_intro="\u6b64\u5bf9\u8bdd\u76ee\u524d\u6709\u4e00\u4efd\u5f85\u786e\u8ba4\u7684 goal \u63d0\u6848\u3002",
            active_summary_intro="\u4ee5\u4e0b\u662f\u8fd9\u4e2a\u5bf9\u8bdd\u76ee\u524d\u8fdb\u884c\u4e2d\u7684 goal \u6458\u8981\u3002",
            recent_summary_intro="\u4ee5\u4e0b\u662f\u8fd9\u4e2a\u5bf9\u8bdd\u6700\u8fd1\u4e00\u6b21\u7684 goal \u6458\u8981\u3002",
        )

    return GoalCardChromeCopy(
        proposal_label="Goal proposal",
        revised_proposal_label="Revised goal proposal",
        started_label="Goal started",
        single_agent_label="Single agent",
        workflow_label="Workflow",
        execution_label="Execution",
        protocol_label="Protocol",
        runtime_label="Runtime",
        goal_id_label="Goal ID",
        objective_label="Objective",
        models_label="Models",
        role_summary_label="Role summary",
        risk_note_label="Risk note",
        superseded_label="Superseded",
        active_goal_label="Active goal",
        most_recent_goal_label="Most recent goal",
        goal_summary_label="Goal summary",
        goal_blocked_label="Goal blocked",
        goal_status_label="Goal status",
        goal_updated_label="Goal updated",
        goal_paused_label="Goal paused",
        goal_resumed_label="Goal resumed",
        goal_stopped_label="Goal stopped",
        pending_summary_intro="Pending goal proposal in this session.",
        active_summary_intro="Active goal summary for this session.",
        recent_summary_intro="Most recent goal summary for this session.",
    )


def build_goal_card_kind_label(
    *,
    user_message: str,
    kind: GoalCardKind,
) -> str:
    copy = build_goal_card_chrome_copy(user_message=user_message)
    if kind == "revised_proposal":
        return copy.revised_proposal_label
    if kind == "started":
        return copy.started_label
    return copy.proposal_label


def build_goal_card_execution_mode_label(
    *,
    user_message: str,
    execution_mode: str,
) -> str:
    copy = build_goal_card_chrome_copy(user_message=user_message)
    return copy.single_agent_label if execution_mode == "single_agent" else copy.workflow_label


def build_goal_card_status_label(
    *,
    user_message: str,
    status: str | None,
) -> str | None:
    normalized = (status or "").strip().lower()
    if not normalized:
        return None

    language_hint = detect_goal_proposal_language_hint(user_message)
    if language_hint in {"traditional_chinese", "chinese"}:
        mapping = {
            "completed": "\u5df2\u5b8c\u6210",
            "succeeded": "\u5df2\u5b8c\u6210",
            "done": "\u5df2\u5b8c\u6210",
            "running": "\u57f7\u884c\u4e2d",
            "active": "\u57f7\u884c\u4e2d",
            "started": "\u5df2\u555f\u52d5",
            "in_progress": "\u9032\u884c\u4e2d",
            "waiting_approval": "\u7b49\u5f85\u6838\u51c6",
            "awaiting_approval": "\u7b49\u5f85\u6838\u51c6",
            "blocked": "\u5df2\u53d7\u963b",
            "paused": "\u5df2\u66ab\u505c",
            "awaiting_resources": "\u7b49\u5f85\u8cc7\u6e90",
            "stalled": "\u5df2\u505c\u6eef",
            "partial": "\u90e8\u5206\u5b8c\u6210",
            "failed": "\u5931\u6557",
            "error": "\u932f\u8aa4",
            "cancelled": "\u5df2\u53d6\u6d88",
            "canceled": "\u5df2\u53d6\u6d88",
            "superseded": "\u5df2\u53d6\u4ee3",
        }
        return mapping.get(normalized, normalized.replace("_", " "))

    if language_hint == "simplified_chinese":
        mapping = {
            "completed": "\u5df2\u5b8c\u6210",
            "succeeded": "\u5df2\u5b8c\u6210",
            "done": "\u5df2\u5b8c\u6210",
            "running": "\u6267\u884c\u4e2d",
            "active": "\u6267\u884c\u4e2d",
            "started": "\u5df2\u542f\u52a8",
            "in_progress": "\u8fdb\u884c\u4e2d",
            "waiting_approval": "\u7b49\u5f85\u6279\u51c6",
            "awaiting_approval": "\u7b49\u5f85\u6279\u51c6",
            "blocked": "\u5df2\u53d7\u963b",
            "paused": "\u5df2\u6682\u505c",
            "awaiting_resources": "\u7b49\u5f85\u8d44\u6e90",
            "stalled": "\u5df2\u505c\u6ede",
            "partial": "\u90e8\u5206\u5b8c\u6210",
            "failed": "\u5931\u8d25",
            "error": "\u9519\u8bef",
            "cancelled": "\u5df2\u53d6\u6d88",
            "canceled": "\u5df2\u53d6\u6d88",
            "superseded": "\u5df2\u66ff\u4ee3",
        }
        return mapping.get(normalized, normalized.replace("_", " "))

    return normalized.replace("_", " ")


def build_goal_hidden_models_label(
    *,
    user_message: str,
    hidden_count: int,
) -> str:
    language_hint = detect_goal_proposal_language_hint(user_message)
    if language_hint in {"traditional_chinese", "chinese", "simplified_chinese"}:
        return f"+{max(0, hidden_count)} \u66f4\u591a"
    return f"+{max(0, hidden_count)} more"


def _goal_follow_up_base_summary(
    *,
    user_message: str,
    summary: str | None,
    traditional_default: str,
    simplified_default: str,
    english_default: str,
) -> str:
    trimmed = (summary or "").strip()
    if trimmed and _is_language_aligned(user_message, trimmed):
        return trimmed
    language_hint = detect_goal_proposal_language_hint(user_message)
    if language_hint in {"traditional_chinese", "chinese"}:
        return traditional_default
    if language_hint == "simplified_chinese":
        return simplified_default
    return english_default


def build_goal_command_help_message(
    *,
    user_message: str,
) -> str:
    language_hint = detect_goal_proposal_language_hint(user_message)

    if language_hint in {"traditional_chinese", "chinese"}:
        return (
            "\u4f7f\u7528 `/goal <request>` \u6e96\u5099\u9577\u6642\u9593\u57f7\u884c\u7684 single-agent goal\u3002\n"
            "\u4f7f\u7528 `/workflow <request>` \u6e96\u5099 workflow goal\u3002\n"
            "Goal \u555f\u52d5\u5f8c\uff0c\u53ef\u4ee5\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u4f86\u7ba1\u7406\u3002"
        )

    if language_hint == "simplified_chinese":
        return (
            "\u4f7f\u7528 `/goal <request>` \u51c6\u5907\u957f\u65f6\u95f4\u6267\u884c\u7684 single-agent goal\u3002\n"
            "\u4f7f\u7528 `/workflow <request>` \u51c6\u5907 workflow goal\u3002\n"
            "Goal \u542f\u52a8\u540e\uff0c\u53ef\u4ee5\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u6765\u7ba1\u7406\u3002"
        )

    return (
        "Use `/goal <request>` to prepare a long-running single-agent goal.\n"
        "Use `/workflow <request>` to prepare a workflow goal.\n"
        "Use `/goal status`, `/goal pause`, `/goal resume`, or `/goal stop` after a goal starts."
    )


def build_goal_follow_up_message(
    *,
    user_message: str,
    kind: GoalFollowUpMessageKind,
    summary: str | None = None,
    approval_count: int = 0,
    tool_names: list[str] | None = None,
    operator_control_hint: str | None = None,
) -> str:
    language_hint = detect_goal_proposal_language_hint(user_message)
    tool_names = [item.strip() for item in (tool_names or []) if item.strip()]
    operator_hint = (
        operator_control_hint.strip()
        if operator_control_hint and _is_language_aligned(user_message, operator_control_hint)
        else ""
    )

    if kind == "goal_help":
        return build_goal_command_help_message(user_message=user_message)

    if kind == "active_goal_exists":
        if language_hint in {"traditional_chinese", "chinese"}:
            return (
                "\u9019\u500b\u5c0d\u8a71\u5df2\u7d93\u6709\u4e00\u500b\u9032\u884c\u4e2d\u7684 goal\u3002"
                " \u8acb\u5148\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u4f86\u7ba1\u7406\u5b83\u3002"
            )
        if language_hint == "simplified_chinese":
            return (
                "\u8fd9\u4e2a\u5bf9\u8bdd\u5df2\u7ecf\u6709\u4e00\u4e2a\u8fdb\u884c\u4e2d\u7684 goal\u3002"
                " \u8bf7\u5148\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u6765\u7ba1\u7406\u5b83\u3002"
            )
        return (
            "This chat already has an active goal. Use `/goal status`, `/goal pause`, "
            "`/goal resume`, or `/goal stop` before starting a new one."
        )

    if kind == "manual_resolution_required":
        base = _goal_follow_up_base_summary(
            user_message=user_message,
            summary=summary,
            traditional_default="\u9019\u500b goal \u5728\u7e7c\u7e8c\u524d\u9700\u8981\u4f60\u5148\u8655\u7406\u5f85\u6838\u51c6\u9805\u76ee\u3002",
            simplified_default="\u8fd9\u4e2a goal \u5728\u7ee7\u7eed\u524d\u9700\u8981\u4f60\u5148\u5904\u7406\u5f85\u6279\u51c6\u9879\u76ee\u3002",
            english_default="The active goal needs approval handling before it can continue.",
        )
        if language_hint in {"traditional_chinese", "chinese"}:
            tool_hint = (
                f" \u5f85\u6838\u51c6\u5de5\u5177\uff1a{', '.join(tool_names)}\u3002"
                if tool_names
                else ""
            )
            action = (
                f" \u8acb\u5148\u5f9e goal drawer \u6216 Goal Console \u8655\u7406 {approval_count} \u500b\u5f85\u6838\u51c6\u9805\u76ee\u3002"
                if approval_count > 0
                else " \u8acb\u5148\u6253\u958b Goal Console \u6aa2\u67e5\u76ee\u524d\u7684\u963b\u585e\u6838\u51c6\u72c0\u614b\u3002"
            )
            return f"{base}{tool_hint}{action}".strip()
        if language_hint == "simplified_chinese":
            tool_hint = (
                f" \u5f85\u6279\u51c6\u5de5\u5177\uff1a{', '.join(tool_names)}\u3002"
                if tool_names
                else ""
            )
            action = (
                f" \u8bf7\u5148\u4ece goal drawer \u6216 Goal Console \u5904\u7406 {approval_count} \u4e2a\u5f85\u6279\u51c6\u9879\u76ee\u3002"
                if approval_count > 0
                else " \u8bf7\u5148\u6253\u5f00 Goal Console \u68c0\u67e5\u5f53\u524d\u7684\u963b\u585e\u6279\u51c6\u72b6\u6001\u3002"
            )
            return f"{base}{tool_hint}{action}".strip()
        tool_hint = f" Pending approval for {', '.join(tool_names)}." if tool_names else ""
        action = (
            f" Review the pending approval{'s' if approval_count > 1 else ''} from the goal drawer or Goal Console before continuing."
            if approval_count > 0
            else " Open the Goal Console to inspect the blocking approval state before continuing."
        )
        return f"{base}{tool_hint}{action}".strip()

    if kind == "blocked":
        base = _goal_follow_up_base_summary(
            user_message=user_message,
            summary=summary,
            traditional_default="\u9019\u500b goal \u76ee\u524d\u8655\u65bc\u53d7\u963b\u72c0\u614b\u3002",
            simplified_default="\u8fd9\u4e2a goal \u76ee\u524d\u5904\u4e8e\u53d7\u963b\u72b6\u6001\u3002",
            english_default="The active goal is currently blocked.",
        )
        if language_hint in {"traditional_chinese", "chinese"}:
            extra = f" {operator_hint}" if operator_hint else ""
            return f"{base}{extra} \u8acb\u5148\u5230 Goal Console \u8abf\u6574 goal\uff0c\u518d\u7e7c\u7e8c\u9001\u51fa\u5f8c\u7e8c\u57f7\u884c\u6307\u793a\u3002".strip()
        if language_hint == "simplified_chinese":
            extra = f" {operator_hint}" if operator_hint else ""
            return f"{base}{extra} \u8bf7\u5148\u5230 Goal Console \u8c03\u6574 goal\uff0c\u518d\u7ee7\u7eed\u53d1\u9001\u540e\u7eed\u6267\u884c\u6307\u4ee4\u3002".strip()
        extra = f" {operator_hint}" if operator_hint else ""
        return f"{base}{extra} Adjust the goal from the Goal Console before sending more execution guidance.".strip()

    if kind == "no_live_attempt":
        if language_hint in {"traditional_chinese", "chinese"}:
            return (
                "\u76ee\u524d\u9019\u500b active goal \u9084\u6c92\u6709\u53ef\u4ee5\u63a5\u6536\u5f8c\u7e8c\u6307\u793a\u7684\u6d3b\u52d5 attempt\u3002"
                " \u8acb\u5148\u6253\u958b Goal Console \u6aa2\u67e5\u73fe\u5728\u7684\u6062\u5fa9\u72c0\u614b\u3002"
            )
        if language_hint == "simplified_chinese":
            return (
                "\u76ee\u524d\u8fd9\u4e2a active goal \u8fd8\u6ca1\u6709\u53ef\u4ee5\u63a5\u6536\u540e\u7eed\u6307\u4ee4\u7684\u6d3b\u52a8 attempt\u3002"
                " \u8bf7\u5148\u6253\u5f00 Goal Console \u68c0\u67e5\u73b0\u5728\u7684\u6062\u590d\u72b6\u6001\u3002"
            )
        return (
            "The active goal still does not have a live attempt ready to receive follow-up guidance. "
            "Use the Goal Console to inspect the current recovery state."
        )

    if kind == "refreshed_forwarded":
        if language_hint in {"traditional_chinese", "chinese"}:
            return "\u6211\u5df2\u91cd\u65b0\u6574\u7406 active worker generation\uff0c\u4e26\u628a\u4f60\u7684\u6307\u793a\u8f49\u9001\u5230\u66f4\u65b0\u5f8c\u7684 goal attempt\u3002"
        if language_hint == "simplified_chinese":
            return "\u6211\u5df2\u91cd\u65b0\u6574\u7406 active worker generation\uff0c\u5e76\u628a\u4f60\u7684\u6307\u4ee4\u8f6c\u53d1\u5230\u66f4\u65b0\u540e\u7684 goal attempt\u3002"
        return "Refreshed the active worker generation and forwarded your guidance to the updated goal attempt."

    if kind == "resumed_forwarded":
        if language_hint in {"traditional_chinese", "chinese"}:
            return "\u6211\u5df2\u6062\u5fa9\u9019\u500b active goal\uff0c\u4e26\u628a\u4f60\u7684\u6307\u793a\u8f49\u9001\u5230\u76ee\u524d\u7684 attempt\u3002"
        if language_hint == "simplified_chinese":
            return "\u6211\u5df2\u6062\u590d\u8fd9\u4e2a active goal\uff0c\u5e76\u628a\u4f60\u7684\u6307\u4ee4\u8f6c\u53d1\u5230\u5f53\u524d\u7684 attempt\u3002"
        return "Resumed the active goal and forwarded your guidance to the current attempt."

    if kind == "forwarded":
        if language_hint in {"traditional_chinese", "chinese"}:
            return "\u6211\u5df2\u628a\u4f60\u7684\u6307\u793a\u8f49\u9001\u7d66\u76ee\u524d\u7684 active goal\u3002\u5b83\u6703\u4f9d\u7167\u9019\u500b\u66f4\u65b0\u7684\u65b9\u5411\u7e7c\u7e8c\u57f7\u884c\u3002"
        if language_hint == "simplified_chinese":
            return "\u6211\u5df2\u628a\u4f60\u7684\u6307\u4ee4\u8f6c\u53d1\u7ed9\u5f53\u524d\u7684 active goal\u3002\u5b83\u4f1a\u6309\u7167\u8fd9\u4e2a\u66f4\u65b0\u540e\u7684\u65b9\u5411\u7ee7\u7eed\u6267\u884c\u3002"
        return "Forwarded your guidance to the active goal. It will continue working with this updated direction."

    raise ValueError(f"Unsupported goal follow-up message kind: {kind}")


async def generate_goal_proposal_assistant_copy(
    invoker: GoalProposalAssistantCopyInvoker,
    *,
    user_message: str,
    proposal_objective: str,
    execution_mode: str,
    protocol_selection: str | None = None,
    role_summary: str | None = None,
    runtime_mode: str | None = None,
    revision_index: int = 0,
) -> GoalProposalAssistantCopyResult:
    summary_lines = [
        "Latest user request (use this language for the reply):",
        user_message.strip() or proposal_objective.strip() or "(empty)",
        "",
        "Goal proposal summary:",
        f"- Objective: {proposal_objective.strip() or '(empty)'}",
        f"- Execution mode: {execution_mode.strip() or 'single_agent'}",
    ]
    if protocol_selection and protocol_selection.strip():
        summary_lines.append(f"- Protocol selection: {protocol_selection.strip()}")
    if runtime_mode and runtime_mode.strip():
        summary_lines.append(f"- Runtime mode: {runtime_mode.strip()}")
    if role_summary and role_summary.strip():
        summary_lines.append(f"- Role summary: {role_summary.strip()}")
    summary_lines.append(f"- Revision index: {max(0, revision_index)}")
    summary_lines.append("")
    summary_lines.append("Write the assistant explanation now.")
    message = "\n".join(summary_lines)

    invocation = AgentInvocationRequest(
        message=message,
        session_id=f"goal-proposal-copy:{uuid4()}",
        inference_overrides={
            "temperature": 0.2,
            "max_tokens": 160,
        },
        tool_mode="disabled",
        execution_profile="judge",
        system_prompt_addendum=_GOAL_PROPOSAL_ASSISTANT_COPY_SYSTEM_PROMPT,
        max_iterations_override=1,
        persist_session=False,
        persist_turn_events=False,
        persist_learning=False,
    )

    try:
        result = await invoker.invoke(invocation)
        content = _normalize_explanation(str(getattr(result, "content", "") or ""))
    except Exception:
        content = ""

    if content and _is_language_aligned(user_message, content):
        return GoalProposalAssistantCopyResult(
            explanation=content,
            source="model",
        )

    return GoalProposalAssistantCopyResult(
        explanation=build_goal_proposal_assistant_copy_fallback(
            user_message=user_message,
            proposal_objective=proposal_objective,
            execution_mode=execution_mode,
            protocol_selection=protocol_selection,
            revision_index=revision_index,
        ),
        source="fallback",
    )
