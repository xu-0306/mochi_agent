from __future__ import annotations

from hashlib import sha256

import pytest

from mochi.agents.generation_policy import (
    GenerationTerminal,
    RecoveryBudget,
    contains_malformed_tool_markup,
    merge_continuation,
    normalize_generation_terminal,
)


@pytest.mark.parametrize(
    "terminal_signal",
    ["length", "max_tokens", "response.incomplete", "output_truncated"],
)
def test_truncation_signals_remain_canonical_until_recovery(terminal_signal: str) -> None:
    decision = normalize_generation_terminal(
        terminal_signal=terminal_signal,
        has_structured_tool_calls=False,
        output="partial answer",
    )

    assert decision.terminal is GenerationTerminal.OUTPUT_TRUNCATED
    assert decision.normalized_reason == "output_truncated"
    assert decision.requires_recovery is True
    assert decision.partial_output == "partial answer"


def test_structured_tool_calls_win_over_incomplete_transport_metadata() -> None:
    decision = normalize_generation_terminal(
        terminal_signal="incomplete",
        has_structured_tool_calls=True,
        output="",
    )

    assert decision.terminal is GenerationTerminal.TOOL_CALLS
    assert decision.normalized_reason == "structured_tool_calls"
    assert decision.requires_recovery is False


def test_unknown_or_missing_terminal_metadata_is_not_normal_completion() -> None:
    unknown = normalize_generation_terminal(
        terminal_signal="provider_specific_status",
        has_structured_tool_calls=False,
        output="visible answer",
    )
    missing = normalize_generation_terminal(
        terminal_signal=None,
        has_structured_tool_calls=False,
        output="visible answer",
    )

    assert unknown.terminal is GenerationTerminal.UNKNOWN
    assert missing.terminal is GenerationTerminal.UNKNOWN
    assert unknown.normalized_reason == "unknown_terminal_signal"
    assert missing.raw_terminal_signal is None


def test_broken_markup_is_invalid_but_truncated_markup_stays_recoverable() -> None:
    broken = '<tool_call>{"name":"read_file","arguments":'
    invalid = normalize_generation_terminal(
        terminal_signal="stop",
        has_structured_tool_calls=False,
        output=broken,
    )
    truncated = normalize_generation_terminal(
        terminal_signal="length",
        has_structured_tool_calls=False,
        output=broken,
    )

    assert contains_malformed_tool_markup(broken) is True
    assert invalid.terminal is GenerationTerminal.INVALID_TOOL_CALL
    assert invalid.normalized_reason == "malformed_tool_markup"
    assert truncated.terminal is GenerationTerminal.OUTPUT_TRUNCATED


def test_valid_tool_markup_is_not_classified_as_malformed() -> None:
    markup = '<tool_call>{"name":"echo","arguments":{"value":"ok"}}</tool_call>'

    assert contains_malformed_tool_markup(markup) is False


def test_one_budget_is_shared_between_textual_and_native_repairs() -> None:
    budget = RecoveryBudget(max_attempts=1, max_tokens=12, max_elapsed_ms=50)

    assert budget.consume(tokens=8, elapsed_ms=20) is True
    assert budget.consume(tokens=1, elapsed_ms=1) is False
    assert budget.exhausted_reason == "attempt_limit"
    assert budget.snapshot() == {
        "attempts_used": 1,
        "tokens_used": 8,
        "elapsed_ms": 20,
        "max_attempts": 1,
        "max_tokens": 12,
        "max_elapsed_ms": 50,
    }


def test_recovery_budget_enforces_token_and_elapsed_limits() -> None:
    token_limited = RecoveryBudget(max_attempts=2, max_tokens=5)
    elapsed_limited = RecoveryBudget(max_attempts=2, max_elapsed_ms=10)

    assert token_limited.consume(tokens=6) is False
    assert token_limited.attempts_used == 0
    assert elapsed_limited.consume(elapsed_ms=11) is False
    assert elapsed_limited.attempts_used == 0


def test_merge_continuation_removes_real_repeated_prefixes_only() -> None:
    merged = merge_continuation("Partial answer", " answer with the remainder")
    adjacent_words = merge_continuation("hello", "orange")

    assert merged.content == "Partial answer with the remainder"
    assert merged.overlap_chars == len(" answer")
    assert merged.merge_digest == sha256(merged.content.encode("utf-8")).hexdigest()
    assert adjacent_words.content == "helloorange"
    assert adjacent_words.overlap_chars == 0


def test_merge_continuation_keeps_one_copy_when_a_backend_repeats_the_prefix() -> None:
    merged = merge_continuation("Partial answer", "Partial answer with the remainder")

    assert merged.content == "Partial answer with the remainder"
    assert merged.overlap_chars == len("Partial answer")
