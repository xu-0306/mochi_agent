"""Backend tool-calling contract tests."""

from __future__ import annotations

from mochi.backends.simulated_tool_protocol import SimulatedToolProtocol
from mochi.backends.tool_call_contract import validate_tool_turn_result
from mochi.backends.types import GenerationResult, Message, ToolCall, ToolSchema


def test_validate_tool_turn_accepts_structured_tool_calls() -> None:
    result = GenerationResult(
        content="",
        thinking="plan",
        tool_calls=[ToolCall(id="1", name="web_search", arguments={})],
    )

    verdict = validate_tool_turn_result(result=result, tools_requested=True)

    assert verdict.is_valid is True
    assert verdict.reason == "tool_calls"

def test_validate_tool_turn_rejects_thinking_only_output() -> None:
    result = GenerationResult(content="", thinking="planning only", tool_calls=[])

    verdict = validate_tool_turn_result(result=result, tools_requested=True)

    assert verdict.is_valid is False
    assert verdict.reason == "thinking_only"

def test_simulated_tool_protocol_flattens_prior_tool_messages() -> None:
    protocol = SimulatedToolProtocol()
    tools = [
        ToolSchema(
            name="web_search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="1", name="web_search", arguments={"query": "Mochi"})],
        ),
        Message(
            role="tool",
            content="Tool web_search result:\nfound",
            tool_call_id="1",
            name="web_search",
        ),
    ]

    prepared = protocol.prepare_messages(messages=messages, tools=tools)

    assert any(
        message.role == "assistant" and "Tool request: web_search" in message.content
        for message in prepared
    )
    assert any(
        message.role == "user" and message.content.startswith("Tool web_search result:")
        for message in prepared
    )
