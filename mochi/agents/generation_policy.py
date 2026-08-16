"""Transport-neutral terminal and recovery policy for generation results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Final

from mochi.backends.tool_call_parsers import parse_tool_calls


class GenerationTerminal(StrEnum):
    """Canonical terminal meaning before ReAct decides how to emit it."""

    COMPLETE = "complete"
    TOOL_CALLS = "tool_calls"
    OUTPUT_TRUNCATED = "output_truncated"
    INVALID_TOOL_CALL = "invalid_tool_call"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GenerationTerminalDecision:
    """One normalized terminal signal without provider-specific semantics."""

    terminal: GenerationTerminal
    normalized_reason: str
    raw_terminal_signal: str | None
    partial_output: str

    @property
    def requires_recovery(self) -> bool:
        return self.terminal in {
            GenerationTerminal.OUTPUT_TRUNCATED,
            GenerationTerminal.INVALID_TOOL_CALL,
        }


@dataclass(frozen=True)
class ContinuationMerge:
    """A deduplicated continuation with a stable digest for later auditing."""

    content: str
    overlap_chars: int
    merge_digest: str


@dataclass
class RecoveryBudget:
    """Shared bounded budget for textual and native recovery attempts."""

    max_attempts: int = 1
    max_tokens: int | None = None
    max_elapsed_ms: int | None = None
    attempts_used: int = 0
    tokens_used: int = 0
    elapsed_ms: int = 0

    def __post_init__(self) -> None:
        _require_non_negative_int(self.max_attempts, name="max_attempts")
        _require_non_negative_int(self.attempts_used, name="attempts_used")
        _require_non_negative_int(self.tokens_used, name="tokens_used")
        _require_non_negative_int(self.elapsed_ms, name="elapsed_ms")
        if self.max_tokens is not None:
            _require_non_negative_int(self.max_tokens, name="max_tokens")
        if self.max_elapsed_ms is not None:
            _require_non_negative_int(self.max_elapsed_ms, name="max_elapsed_ms")

    @property
    def exhausted_reason(self) -> str | None:
        if self.attempts_used >= self.max_attempts:
            return "attempt_limit"
        if self.max_tokens is not None and self.tokens_used >= self.max_tokens:
            return "token_limit"
        if self.max_elapsed_ms is not None and self.elapsed_ms >= self.max_elapsed_ms:
            return "elapsed_limit"
        return None

    def can_consume(self, *, tokens: int = 0, elapsed_ms: int = 0) -> bool:
        _require_non_negative_int(tokens, name="tokens")
        _require_non_negative_int(elapsed_ms, name="elapsed_ms")
        if self.attempts_used >= self.max_attempts:
            return False
        if self.max_tokens is not None and self.tokens_used + tokens > self.max_tokens:
            return False
        return self.max_elapsed_ms is None or self.elapsed_ms + elapsed_ms <= self.max_elapsed_ms

    def consume(self, *, tokens: int = 0, elapsed_ms: int = 0) -> bool:
        """Reserve one recovery attempt only when every aggregate limit allows it."""

        if not self.can_consume(tokens=tokens, elapsed_ms=elapsed_ms):
            return False
        self.attempts_used += 1
        self.tokens_used += tokens
        self.elapsed_ms += elapsed_ms
        return True

    def snapshot(self) -> dict[str, int | None]:
        return {
            "attempts_used": self.attempts_used,
            "tokens_used": self.tokens_used,
            "elapsed_ms": self.elapsed_ms,
            "max_attempts": self.max_attempts,
            "max_tokens": self.max_tokens,
            "max_elapsed_ms": self.max_elapsed_ms,
        }


_TRUNCATION_SIGNALS: Final[frozenset[str]] = frozenset(
    {
        "context_length",
        "incomplete",
        "length",
        "max_output_tokens",
        "max_tokens",
        "output_truncated",
        "response.incomplete",
        "response_incomplete",
        "token_limit",
        "truncated",
    }
)
_COMPLETE_SIGNALS: Final[frozenset[str]] = frozenset(
    {"complete", "completed", "done", "end_turn", "stop", "success"}
)
_TOOL_CALL_SIGNALS: Final[frozenset[str]] = frozenset(
    {"function_call", "function_calls", "tool_call", "tool_calls"}
)
_TOOL_MARKUP_MARKERS: Final[tuple[str, ...]] = (
    "<tool_call",
    "</tool_call",
    "<function=",
    "</function",
    "<parameter=",
    "</parameter",
)


def normalize_generation_terminal(
    *,
    terminal_signal: object,
    has_structured_tool_calls: bool,
    output: str = "",
) -> GenerationTerminalDecision:
    """Classify a terminal signal without treating unknown metadata as success.

    Structured calls win because their transport is already valid. Truncation
    takes precedence over malformed textual markup so a split tool call can be
    completed by a bounded continuation instead of being rejected prematurely.
    """

    raw_signal = _normalized_signal(terminal_signal)
    if has_structured_tool_calls:
        return GenerationTerminalDecision(
            terminal=GenerationTerminal.TOOL_CALLS,
            normalized_reason="structured_tool_calls",
            raw_terminal_signal=raw_signal,
            partial_output=output,
        )
    if raw_signal in _TRUNCATION_SIGNALS:
        return GenerationTerminalDecision(
            terminal=GenerationTerminal.OUTPUT_TRUNCATED,
            normalized_reason="output_truncated",
            raw_terminal_signal=raw_signal,
            partial_output=output,
        )
    if contains_malformed_tool_markup(output):
        return GenerationTerminalDecision(
            terminal=GenerationTerminal.INVALID_TOOL_CALL,
            normalized_reason="malformed_tool_markup",
            raw_terminal_signal=raw_signal,
            partial_output=output,
        )
    if raw_signal in _TOOL_CALL_SIGNALS:
        return GenerationTerminalDecision(
            terminal=GenerationTerminal.UNKNOWN,
            normalized_reason="missing_structured_tool_calls",
            raw_terminal_signal=raw_signal,
            partial_output=output,
        )
    if raw_signal in _COMPLETE_SIGNALS:
        return GenerationTerminalDecision(
            terminal=GenerationTerminal.COMPLETE,
            normalized_reason="complete",
            raw_terminal_signal=raw_signal,
            partial_output=output,
        )
    return GenerationTerminalDecision(
        terminal=GenerationTerminal.UNKNOWN,
        normalized_reason="unknown_terminal_signal",
        raw_terminal_signal=raw_signal,
        partial_output=output,
    )


def contains_malformed_tool_markup(output: str) -> bool:
    """Identify tool-like text which is not a valid parsed tool-call payload."""

    if not isinstance(output, str):
        raise TypeError("output must be a string")
    lowered = output.lower()
    if not any(marker in lowered for marker in _TOOL_MARKUP_MARKERS):
        return False
    return not parse_tool_calls(output)


def merge_continuation(partial_output: str, continuation: str) -> ContinuationMerge:
    """Join a continuation without removing accidental one-character overlaps."""

    if not isinstance(partial_output, str) or not isinstance(continuation, str):
        raise TypeError("partial_output and continuation must be strings")

    overlap_chars = _continuation_overlap(partial_output, continuation)
    if continuation.startswith(partial_output):
        content = continuation
        overlap_chars = len(partial_output)
    elif partial_output.endswith(continuation):
        content = partial_output
        overlap_chars = len(continuation)
    else:
        content = partial_output + continuation[overlap_chars:]
    return ContinuationMerge(
        content=content,
        overlap_chars=overlap_chars,
        merge_digest=sha256(content.encode("utf-8")).hexdigest(),
    )


def _continuation_overlap(partial_output: str, continuation: str) -> int:
    max_overlap = min(len(partial_output), len(continuation))
    for size in range(max_overlap, 2, -1):
        if partial_output.endswith(continuation[:size]):
            return size
    return 0


def _normalized_signal(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None


def _require_non_negative_int(value: object, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
