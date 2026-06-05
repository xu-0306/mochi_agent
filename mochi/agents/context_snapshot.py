"""Utilities for lightweight chat context estimation and snapshots."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
