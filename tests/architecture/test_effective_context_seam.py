from __future__ import annotations

from mochi.agents.effective_context import (
    ContextConfidence,
    ContextSource,
    select_effective_context,
)


def test_serving_context_wins_over_larger_advertised_context() -> None:
    context = select_effective_context(
        advertised_context=32768,
        serving_context=4096,
    )

    assert context.context_length == 4096
    assert context.source is ContextSource.SERVING
    assert context.confidence is ContextConfidence.HIGH
    assert context.is_hard_limit is True


def test_lower_configured_context_remains_conservative() -> None:
    context = select_effective_context(
        configured_context=2048,
        serving_context=4096,
        advertised_context=32768,
    )

    assert context.context_length == 2048
    assert context.source is ContextSource.CONFIGURED
    assert context.confidence is ContextConfidence.HIGH
    assert context.is_hard_limit is True


def test_advertised_context_is_not_a_hard_guarantee() -> None:
    context = select_effective_context(advertised_context=32768)

    assert context.context_length == 32768
    assert context.source is ContextSource.ADVERTISED
    assert context.confidence is ContextConfidence.MEDIUM
    assert context.is_hard_limit is False


def test_unknown_or_invalid_metadata_uses_low_confidence_fallback() -> None:
    context = select_effective_context(
        configured_context=0,
        serving_context=True,
        advertised_context="32768",  # type: ignore[arg-type]
        fallback_context=2048,
    )

    assert context.context_length == 2048
    assert context.source is ContextSource.FALLBACK_DEFAULT
    assert context.confidence is ContextConfidence.LOW
    assert context.is_hard_limit is False


def test_equal_candidates_use_serving_provenance_deterministically() -> None:
    context = select_effective_context(
        configured_context=4096,
        serving_context=4096,
        advertised_context=4096,
    )

    assert context.context_length == 4096
    assert context.source is ContextSource.SERVING
    assert context.confidence is ContextConfidence.HIGH
    assert context.is_hard_limit is True
