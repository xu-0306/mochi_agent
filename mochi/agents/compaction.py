"""Conversation compaction primitives."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

from mochi.backends.types import Message

SemanticSummaryMode = Literal["deterministic", "hybrid"]

_FILE_PATTERN = re.compile(
    r"(?<![\w/\\.-])(?:[A-Za-z]:)?(?:[\w.-]+[\\/])*[\w.-]+\.(?:py|md|txt|json|ya?ml|toml|ini|cfg|ts|tsx|js|jsx|html|css|sql)"
)
_SYMBOL_PATTERN = re.compile(r"\b(?:class|def|async def|function)\s+([A-Za-z_][A-Za-z0-9_]*)")


@dataclass(frozen=True)
class ConversationStateSummary:
    """語義化對話摘要狀態。"""

    current_task: str = ""
    current_state: str = ""
    important_files: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    errors_and_corrections: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    next_step: str = ""
    recent_user_intent: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "current_task": self.current_task,
            "current_state": self.current_state,
            "important_files": list(self.important_files),
            "decisions": list(self.decisions),
            "errors_and_corrections": list(self.errors_and_corrections),
            "open_questions": list(self.open_questions),
            "next_step": self.next_step,
            "recent_user_intent": self.recent_user_intent,
        }


@dataclass(frozen=True)
class ContextBudget:
    """Prompt context budget inputs for compaction."""

    max_input_tokens: int | None = None
    reserve_output_tokens: int = 0


@dataclass(frozen=True)
class CompactionDiagnostics:
    """Compaction diagnostics exposed to snapshots and prompts."""

    compaction_mode: Literal["legacy", "semantic"]
    summary_mode: SemanticSummaryMode | None
    reason: Literal["history_window", "token_budget"] | None
    compacted_count: int
    history_tokens: int
    retained_tokens: int
    state_tokens: int
    max_input_tokens: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "compaction_mode": self.compaction_mode,
            "summary_mode": self.summary_mode,
            "reason": self.reason,
            "compacted_count": self.compacted_count,
            "history_tokens": self.history_tokens,
            "retained_tokens": self.retained_tokens,
            "state_tokens": self.state_tokens,
            "max_input_tokens": self.max_input_tokens,
        }


@dataclass(frozen=True)
class CompactionPolicy:
    """對話壓縮策略。"""

    trigger_messages: int
    retain_recent_messages: int
    max_summary_chars: int = 1200
    max_points: int = 8
    max_chars_per_point: int = 120
    semantic_compaction_enabled: bool = True
    summary_mode: SemanticSummaryMode = "hybrid"
    max_input_tokens: int | None = None


@dataclass(frozen=True)
class CompactionResult:
    """壓縮結果。"""

    summary: str
    retained_history: list[Message]
    compacted_count: int
    summary_state: ConversationStateSummary | None = None
    diagnostics: CompactionDiagnostics | None = None


class ConversationCompactor:
    """Manage short-term history compaction for prompt context."""

    def __init__(self, policy: CompactionPolicy) -> None:
        self._policy = policy

    @classmethod
    def from_max_messages(cls, max_messages: int) -> ConversationCompactor:
        return cls.from_settings(max_messages=max_messages)

    @classmethod
    def from_settings(
        cls,
        *,
        max_messages: int,
        semantic_compaction_enabled: bool = True,
        summary_mode: SemanticSummaryMode = "hybrid",
        max_input_tokens: int | None = None,
        keep_recent_messages: int | None = None,
    ) -> ConversationCompactor:
        safe_max = max(8, max_messages)
        trigger = max(6, int(safe_max * 0.8))
        retain = keep_recent_messages or max(4, int(safe_max * 0.4))
        if retain >= trigger:
            retain = max(4, trigger // 2)
        return cls(
            CompactionPolicy(
                trigger_messages=trigger,
                retain_recent_messages=retain,
                semantic_compaction_enabled=semantic_compaction_enabled,
                summary_mode=summary_mode,
                max_input_tokens=max_input_tokens,
            )
        )

    def compact(
        self,
        history: list[Message],
        *,
        previous_summary: str | None = None,
        previous_state: ConversationStateSummary | None = None,
        budget: ContextBudget | None = None,
        summarizer: Callable[[ConversationStateSummary, list[Message]], ConversationStateSummary | None]
        | None = None,
    ) -> CompactionResult | None:
        """Compact history into either semantic state or legacy summary text."""

        retain = min(self._policy.retain_recent_messages, len(history))
        if retain <= 0 or len(history) <= retain:
            return None

        reason = self._resolve_compaction_reason(history, retain=retain, budget=budget)
        if reason is None:
            return None

        compacted = history[:-retain]
        retained_history = list(history[-retain:])
        if self._policy.semantic_compaction_enabled:
            summary_state = self._build_state_summary(compacted, previous_state)
            if self._policy.summary_mode == "hybrid" and summarizer is not None:
                refined = summarizer(summary_state, compacted)
                if refined is not None:
                    summary_state = refined
            summary = self._render_state_summary(summary_state)
            diagnostics = CompactionDiagnostics(
                compaction_mode="semantic",
                summary_mode=self._policy.summary_mode,
                reason=reason,
                compacted_count=len(compacted),
                history_tokens=_estimate_messages_tokens(history),
                retained_tokens=_estimate_messages_tokens(retained_history),
                state_tokens=_estimate_text_tokens(summary),
                max_input_tokens=(budget.max_input_tokens if budget is not None else self._policy.max_input_tokens),
            )
            return CompactionResult(
                summary=self._truncate(summary, self._policy.max_summary_chars),
                retained_history=retained_history,
                compacted_count=len(compacted),
                summary_state=summary_state,
                diagnostics=diagnostics,
            )

        summary = self._merge_summary(previous_summary=previous_summary, compacted=compacted)
        diagnostics = CompactionDiagnostics(
            compaction_mode="legacy",
            summary_mode=None,
            reason=reason,
            compacted_count=len(compacted),
            history_tokens=_estimate_messages_tokens(history),
            retained_tokens=_estimate_messages_tokens(retained_history),
            state_tokens=_estimate_text_tokens(summary),
            max_input_tokens=(budget.max_input_tokens if budget is not None else self._policy.max_input_tokens),
        )
        return CompactionResult(
            summary=summary,
            retained_history=retained_history,
            compacted_count=len(compacted),
            summary_state=None,
            diagnostics=diagnostics,
        )

    def _resolve_compaction_reason(
        self,
        history: list[Message],
        *,
        retain: int,
        budget: ContextBudget | None,
    ) -> Literal["history_window", "token_budget"] | None:
        if budget is not None and budget.max_input_tokens is not None:
            max_input_tokens = max(1, budget.max_input_tokens - max(0, budget.reserve_output_tokens))
            history_tokens = _estimate_messages_tokens(history)
            if history_tokens > max_input_tokens:
                return "token_budget"
        elif self._policy.max_input_tokens is not None:
            if _estimate_messages_tokens(history) > self._policy.max_input_tokens:
                return "token_budget"

        if len(history) > self._policy.trigger_messages and len(history) > retain:
            return "history_window"
        return None

    def _build_state_summary(
        self,
        compacted: list[Message],
        previous_state: ConversationStateSummary | None,
    ) -> ConversationStateSummary:
        previous_files = list(previous_state.important_files) if previous_state is not None else []
        previous_decisions = list(previous_state.decisions) if previous_state is not None else []
        previous_errors = (
            list(previous_state.errors_and_corrections) if previous_state is not None else []
        )
        previous_questions = list(previous_state.open_questions) if previous_state is not None else []

        user_messages = [self._normalize_text(message.content) for message in compacted if message.role == "user"]
        assistant_messages = [
            self._normalize_text(message.content) for message in compacted if message.role == "assistant"
        ]
        all_text = "\n".join(text for text in user_messages + assistant_messages if text)

        current_task = self._pick_latest_nonempty(
            user_messages,
            fallback=previous_state.current_task if previous_state is not None else "",
        )
        recent_user_intent = self._pick_latest_nonempty(
            user_messages,
            fallback=previous_state.recent_user_intent if previous_state is not None else "",
        )
        next_step = self._derive_next_step(
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            fallback=previous_state.next_step if previous_state is not None else "",
        )
        current_state = self._derive_current_state(
            assistant_messages=assistant_messages,
            fallback=previous_state.current_state if previous_state is not None else "",
        )

        important_files = _dedupe_preserve_order(
            previous_files + _extract_file_mentions(all_text) + _extract_symbol_mentions(all_text)
        )[:8]
        decisions = _dedupe_preserve_order(
            previous_decisions + self._extract_decision_points(compacted)
        )[:8]
        errors = _dedupe_preserve_order(
            previous_errors + self._extract_error_points(compacted)
        )[:8]
        questions = _dedupe_preserve_order(
            previous_questions + self._extract_open_questions(compacted)
        )[:8]

        return ConversationStateSummary(
            current_task=self._truncate(current_task, 220),
            current_state=self._truncate(current_state, 280),
            important_files=[self._truncate(item, 140) for item in important_files],
            decisions=[self._truncate(item, 180) for item in decisions],
            errors_and_corrections=[self._truncate(item, 180) for item in errors],
            open_questions=[self._truncate(item, 180) for item in questions],
            next_step=self._truncate(next_step, 220),
            recent_user_intent=self._truncate(recent_user_intent, 220),
        )

    def _render_state_summary(self, state: ConversationStateSummary) -> str:
        sections = [
            ("Current task", state.current_task),
            ("Current state", state.current_state),
            ("Important files", "\n".join(f"- {item}" for item in state.important_files)),
            ("Decisions", "\n".join(f"- {item}" for item in state.decisions)),
            (
                "Errors and corrections",
                "\n".join(f"- {item}" for item in state.errors_and_corrections),
            ),
            ("Open questions", "\n".join(f"- {item}" for item in state.open_questions)),
            ("Next step", state.next_step),
            ("Recent user intent", state.recent_user_intent),
        ]
        rendered: list[str] = []
        for title, body in sections:
            text = body.strip() if isinstance(body, str) else ""
            if not text:
                continue
            rendered.append(f"{title}:\n{text}")
        if not rendered:
            return "Conversation summary:\n- Conversation continued with low-signal messages."
        return "\n\n".join(rendered)

    def _merge_summary(
        self,
        *,
        previous_summary: str | None,
        compacted: list[Message],
    ) -> str:
        points: list[str] = []
        if previous_summary and previous_summary.strip():
            points.append(f"Earlier summary: {self._truncate(previous_summary.strip(), 400)}")

        for message in compacted[-self._policy.max_points :]:
            text = self._normalize_text(message.content)
            if not text:
                continue
            role = message.role.capitalize()
            points.append(f"{role}: {self._truncate(text, self._policy.max_chars_per_point)}")

        if not points:
            points.append("Conversation continued with low-signal messages.")

        summary = "\n".join(f"- {point}" for point in points)
        return self._truncate(summary, self._policy.max_summary_chars)

    def _extract_decision_points(self, messages: list[Message]) -> list[str]:
        hits: list[str] = []
        for message in messages:
            normalized = self._normalize_text(message.content)
            lowered = normalized.lower()
            if not normalized:
                continue
            if any(keyword in lowered for keyword in ("decide", "decision", "choose", "use ", "switch", "implement")):
                hits.append(normalized)
        return hits

    def _extract_error_points(self, messages: list[Message]) -> list[str]:
        hits: list[str] = []
        for message in messages:
            normalized = self._normalize_text(message.content)
            lowered = normalized.lower()
            if any(
                keyword in lowered
                for keyword in ("error", "failed", "exception", "traceback", "bug", "fix", "warning")
            ):
                hits.append(normalized)
        return hits

    def _extract_open_questions(self, messages: list[Message]) -> list[str]:
        hits: list[str] = []
        for message in messages:
            normalized = self._normalize_text(message.content)
            if message.role == "user" and "?" in normalized:
                hits.append(normalized)
        return hits

    def _derive_next_step(
        self,
        *,
        user_messages: list[str],
        assistant_messages: list[str],
        fallback: str,
    ) -> str:
        if assistant_messages:
            latest = assistant_messages[-1]
            for sentence in _split_sentences(latest):
                lowered = sentence.lower()
                if any(keyword in lowered for keyword in ("next", "will", "then", "plan", "follow")):
                    return sentence
        if user_messages:
            return user_messages[-1]
        return fallback

    def _derive_current_state(self, *, assistant_messages: list[str], fallback: str) -> str:
        if assistant_messages:
            tail = assistant_messages[-2:]
            return " | ".join(self._truncate(item, 140) for item in tail if item)
        return fallback

    @staticmethod
    def _pick_latest_nonempty(items: list[str], *, fallback: str) -> str:
        for item in reversed(items):
            if item:
                return item
        return fallback

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.strip().split())

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(text) <= max_chars:
            return text
        return f"{text[: max_chars - 3].rstrip()}..."


def _estimate_messages_tokens(messages: list[Message]) -> int:
    rendered = "\n".join(f"<{message.role}>{message.content}</{message.role}>" for message in messages)
    return _estimate_text_tokens(rendered)


def _estimate_text_tokens(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    return max(1, max((len(normalized) + 3) // 4, len(normalized.split())))


def _extract_file_mentions(text: str) -> list[str]:
    return [match.group(0) for match in _FILE_PATTERN.finditer(text)]


def _extract_symbol_mentions(text: str) -> list[str]:
    return [match.group(1) for match in _SYMBOL_PATTERN.finditer(text)]


def _split_sentences(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered
