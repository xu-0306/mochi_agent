from __future__ import annotations

from mochi.agents.context_snapshot import estimate_text_tokens


def test_estimate_text_tokens_uses_heuristic_fallback_without_tokenizer() -> None:
    estimate = estimate_text_tokens("abcdef", tokenizer=None)

    assert estimate.tokens == 2
    assert estimate.approximate is True
