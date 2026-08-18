"""Pure, conservative selection for a model's usable context window."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

DEFAULT_CONTEXT_LENGTH_FALLBACK: Final[int] = 4096


class ContextSource(StrEnum):
    """The category that supplied the selected context limit."""

    SERVING = "serving"
    CONFIGURED = "configured"
    ADVERTISED = "advertised"
    FALLBACK_DEFAULT = "fallback_default"


class ContextConfidence(StrEnum):
    """How safely a selected context limit can act as a hard boundary."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class EffectiveContext:
    """One conservative context limit with its provenance and confidence."""

    context_length: int
    source: ContextSource
    confidence: ContextConfidence
    is_hard_limit: bool


_SOURCE_PRECEDENCE: Final[dict[ContextSource, int]] = {
    ContextSource.SERVING: 0,
    ContextSource.CONFIGURED: 1,
    ContextSource.ADVERTISED: 2,
}


def select_effective_context(
    *,
    configured_context: int | None = None,
    serving_context: int | None = None,
    advertised_context: int | None = None,
    fallback_context: int = DEFAULT_CONTEXT_LENGTH_FALLBACK,
) -> EffectiveContext:
    """Select the smallest valid limit without trusting metadata as a hard cap.

    Serving and configured values are high-confidence hard limits. Advertised
    model capacity still constrains the result conservatively, but never turns
    into a hard request gate on its own. Invalid values are treated as absent.
    """

    candidates = tuple(
        candidate
        for candidate in (
            _candidate(
                serving_context,
                source=ContextSource.SERVING,
                confidence=ContextConfidence.HIGH,
                is_hard_limit=True,
            ),
            _candidate(
                configured_context,
                source=ContextSource.CONFIGURED,
                confidence=ContextConfidence.HIGH,
                is_hard_limit=True,
            ),
            _candidate(
                advertised_context,
                source=ContextSource.ADVERTISED,
                confidence=ContextConfidence.MEDIUM,
                is_hard_limit=False,
            ),
        )
        if candidate is not None
    )
    if candidates:
        return min(
            candidates,
            key=lambda candidate: (
                candidate.context_length,
                _SOURCE_PRECEDENCE[candidate.source],
            ),
        )

    return EffectiveContext(
        context_length=_positive_int_or_none(fallback_context) or DEFAULT_CONTEXT_LENGTH_FALLBACK,
        source=ContextSource.FALLBACK_DEFAULT,
        confidence=ContextConfidence.LOW,
        is_hard_limit=False,
    )


def _candidate(
    value: object,
    *,
    source: ContextSource,
    confidence: ContextConfidence,
    is_hard_limit: bool,
) -> EffectiveContext | None:
    context_length = _positive_int_or_none(value)
    if context_length is None:
        return None
    return EffectiveContext(
        context_length=context_length,
        source=source,
        confidence=confidence,
        is_hard_limit=is_hard_limit,
    )


def _positive_int_or_none(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value
