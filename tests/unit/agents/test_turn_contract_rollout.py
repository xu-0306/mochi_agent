from mochi.agents.context import PromptContext
from mochi.agents.turn_contract_rollout import (
    conversation_inputs_from_prompt_context,
)
from mochi.backends.types import Message


def test_prompt_context_projection_preserves_duplicate_content_in_order() -> None:
    prompt_context = PromptContext(
        history=[
            Message(role="user", content="continue"),
            Message(role="assistant", content="working"),
            Message(role="user", content="continue"),
        ]
    )

    _, history, _ = conversation_inputs_from_prompt_context(
        turn_id="turn-now",
        current_message="continue",
        prompt_context=prompt_context,
    )

    assert [turn.content for turn in history] == ["continue", "working", "continue"]
    assert len({turn.turn_id for turn in history}) == 3
    assert history[0].turn_id.startswith("history:0:")
    assert history[2].turn_id.startswith("history:2:")
