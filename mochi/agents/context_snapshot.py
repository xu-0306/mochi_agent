"""Utilities for lightweight chat context estimation and snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

from mochi.agents.compaction import CompactionDiagnostics, ConversationStateSummary
from mochi.backends.inference_capabilities import ReasoningEffort
from mochi.backends.types import Message, ModelInfo

try:  # Optional dependency for better token estimates.
    import tiktoken  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    tiktoken = None


@dataclass(frozen=True)
class TokenEstimate:
    """Estimated token count with a roughness flag."""

    tokens: int
    approximate: bool
    source: str


@dataclass(frozen=True)
class ChatContextSnapshot:
    """Summary of the prompt budget for the next assistant turn."""

    type: str
    session_id: str
    model: str
    backend_type: str
    context_length: int
    estimated_prompt_tokens: int
    reserved_output_tokens: int
    remaining_tokens: int
    usage_ratio: float
    summary_tokens: int
    history_tokens: int
    memory_tokens: int
    skills_tokens: int
    tool_tokens: int
    draft_tokens: int
    compaction_triggered: bool
    compaction_reason: str | None
    approximate: bool
    compaction_mode: str = "legacy"
    summary_mode: str | None = None
    state_tokens: int = 0
    recent_raw_tokens: int = 0
    reasoning_effort: ReasoningEffort | None = None
    context_source: str = "fallback_default"
    context_confidence: str = "low"
    context_is_hard_limit: bool = False
    revision: int = 0
    compaction_revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextLifecycleSnapshot:
    """Versioned durable context state for restart-safe prompt assembly."""

    invocation_id: str
    revision: int
    compaction_revision: int
    phase: Literal["pre_prompt", "post_compaction", "post_response"]
    history: tuple[Mapping[str, Any], ...]
    summary: str | None
    summary_state: ConversationStateSummary | None
    compaction_diagnostics: CompactionDiagnostics | None
    schema_version: int = 2
    type: str = "context_snapshot"

    def to_event(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "schema_version": self.schema_version,
            "invocation_id": self.invocation_id,
            "revision": self.revision,
            "compaction_revision": self.compaction_revision,
            "phase": self.phase,
            "history": [dict(message) for message in self.history],
            "summary": self.summary,
            "summary_state": (
                self.summary_state.to_dict() if self.summary_state is not None else None
            ),
            "compaction_diagnostics": (
                self.compaction_diagnostics.to_dict()
                if self.compaction_diagnostics is not None
                else None
            ),
        }

    @classmethod
    def from_event(cls, event: Mapping[str, Any]) -> ContextLifecycleSnapshot | None:
        """Parse only complete v2 snapshots; older events remain compatible."""

        if event.get("type") != "context_snapshot" or event.get("schema_version") != 2:
            return None
        invocation_id = event.get("invocation_id")
        revision = event.get("revision")
        compaction_revision = event.get("compaction_revision")
        phase = event.get("phase")
        history = event.get("history")
        summary = event.get("summary")
        if (
            not isinstance(invocation_id, str)
            or not invocation_id
            or not _positive_int(revision)
            or _nonnegative_int(compaction_revision) is None
            or phase not in {"pre_prompt", "post_compaction", "post_response"}
            or not isinstance(history, list)
            or not all(isinstance(message, Mapping) for message in history)
            or summary is not None
            and not isinstance(summary, str)
        ):
            return None

        summary_state = _summary_state_from_dict(event.get("summary_state"))
        if event.get("summary_state") is not None and summary_state is None:
            return None
        diagnostics = _diagnostics_from_dict(event.get("compaction_diagnostics"))
        if event.get("compaction_diagnostics") is not None and diagnostics is None:
            return None
        return cls(
            invocation_id=invocation_id,
            revision=revision,
            compaction_revision=compaction_revision,
            phase=phase,
            history=tuple(dict(message) for message in history),
            summary=summary,
            summary_state=summary_state,
            compaction_diagnostics=diagnostics,
        )


def _positive_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _nonnegative_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _summary_state_from_dict(value: object) -> ConversationStateSummary | None:
    if not isinstance(value, Mapping):
        return None

    def text(field: str) -> str:
        candidate = value.get(field)
        return candidate if isinstance(candidate, str) else ""

    def text_list(field: str) -> list[str]:
        candidate = value.get(field)
        if not isinstance(candidate, list) or not all(isinstance(item, str) for item in candidate):
            return []
        return list(candidate)

    return ConversationStateSummary(
        current_task=text("current_task"),
        current_state=text("current_state"),
        important_files=text_list("important_files"),
        decisions=text_list("decisions"),
        errors_and_corrections=text_list("errors_and_corrections"),
        open_questions=text_list("open_questions"),
        next_step=text("next_step"),
        recent_user_intent=text("recent_user_intent"),
    )


def _diagnostics_from_dict(value: object) -> CompactionDiagnostics | None:
    if not isinstance(value, Mapping):
        return None
    compaction_mode = value.get("compaction_mode")
    summary_mode = value.get("summary_mode")
    reason = value.get("reason")
    compacted_count = _nonnegative_int(value.get("compacted_count"))
    history_tokens = _nonnegative_int(value.get("history_tokens"))
    retained_tokens = _nonnegative_int(value.get("retained_tokens"))
    state_tokens = _nonnegative_int(value.get("state_tokens"))
    max_input_tokens = value.get("max_input_tokens")
    if (
        compaction_mode not in {"legacy", "semantic"}
        or summary_mode not in {None, "deterministic", "hybrid"}
        or reason not in {None, "history_window", "token_budget"}
        or compacted_count is None
        or history_tokens is None
        or retained_tokens is None
        or state_tokens is None
        or max_input_tokens is not None
        and _positive_int(max_input_tokens) is None
    ):
        return None
    return CompactionDiagnostics(
        compaction_mode=compaction_mode,
        summary_mode=summary_mode,
        reason=reason,
        compacted_count=compacted_count,
        history_tokens=history_tokens,
        retained_tokens=retained_tokens,
        state_tokens=state_tokens,
        max_input_tokens=max_input_tokens,
    )


def estimate_text_tokens(
    text: str,
    *,
    tokenizer: Any | None = None,
    model_name: str | None = None,
) -> TokenEstimate:
    """Estimate token count using tokenizer-backed counts when possible."""

    normalized = text or ""
    if not normalized:
        return TokenEstimate(tokens=0, approximate=False, source="empty")

    if tokenizer is not None:
        exact = _count_tokens_with_tokenizer(normalized, tokenizer)
        if exact is not None:
            return TokenEstimate(tokens=exact, approximate=False, source="tokenizer")

    if model_name and tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except Exception:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                encoding = None
        if encoding is not None:
            try:
                return TokenEstimate(
                    tokens=len(encoding.encode(normalized)),
                    approximate=False,
                    source="tiktoken",
                )
            except Exception:
                pass

    return TokenEstimate(
        tokens=_heuristic_token_estimate(normalized),
        approximate=True,
        source="heuristic",
    )


def estimate_messages_tokens(
    messages: list[Message],
    *,
    tokenizer: Any | None = None,
    model_name: str | None = None,
) -> TokenEstimate:
    """Estimate token usage for structured chat messages."""

    if not messages:
        return TokenEstimate(tokens=0, approximate=False, source="empty")

    rendered = "\n".join(
        f"<{message.role}>\n{message.content}\n</{message.role}>"
        for message in messages
    )
    estimate = estimate_text_tokens(rendered, tokenizer=tokenizer, model_name=model_name)
    return TokenEstimate(
        tokens=max(estimate.tokens, len(messages) * 4),
        approximate=estimate.approximate,
        source=estimate.source,
    )


def estimate_backend_text_tokens(
    text: str,
    *,
    backend: Any | None = None,
    model_info: ModelInfo | None = None,
) -> TokenEstimate:
    """Estimate tokens using backend-specific helpers when available."""

    if backend is not None:
        exact = _estimate_with_backend(text, backend)
        if exact is not None:
            return TokenEstimate(tokens=exact, approximate=False, source=type(backend).__name__)

    model_name = model_info.name if model_info is not None else None
    return estimate_text_tokens(text, model_name=model_name)


def _estimate_with_backend(text: str, backend: Any) -> int | None:
    """Best-effort access to backend-specific token counters."""

    if hasattr(backend, "_count_tokens_with_tokenizer") and hasattr(backend, "_resolve_chat_template_source"):
        try:
            tokenizer = backend._resolve_chat_template_source()  # noqa: SLF001
            count = backend._count_tokens_with_tokenizer(text, tokenizer)  # noqa: SLF001
            if isinstance(count, int):
                return count
        except Exception:
            pass

    if hasattr(backend, "_count_tokens_with_runtime") and hasattr(backend, "_model"):
        try:
            model = getattr(backend, "_model", None)
            if model is not None:
                count = backend._count_tokens_with_runtime(model, text)  # noqa: SLF001
                if isinstance(count, int):
                    return count
        except Exception:
            pass

    return None


def _count_tokens_with_tokenizer(text: str, tokenizer: Any | None) -> int | None:
    if tokenizer is None:
        return None

    if hasattr(tokenizer, "encode"):
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if isinstance(encoded, list):
                return len(encoded)
        except Exception:
            pass

    try:
        encoded_dict = tokenizer(  # type: ignore[misc]
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        if isinstance(encoded_dict, dict):
            input_ids = encoded_dict.get("input_ids")
            if isinstance(input_ids, list):
                if input_ids and isinstance(input_ids[0], list):
                    return len(input_ids[0])
                return len(input_ids)
    except Exception:
        pass

    if hasattr(tokenizer, "tokenize"):
        try:
            pieces = tokenizer.tokenize(text)
            if isinstance(pieces, list):
                return len(pieces)
        except Exception:
            pass

    return None


def _heuristic_token_estimate(text: str) -> int:
    normalized = text.strip()
    if not normalized:
        return 0
    char_estimate = (len(normalized) + 3) // 4
    word_estimate = len(normalized.split())
    return max(1, min(len(normalized), max(char_estimate, word_estimate)))
