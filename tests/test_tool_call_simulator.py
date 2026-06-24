"""ToolCallSimulator 測試。"""

from __future__ import annotations

import re

from mochi.backends.tool_call_simulator import ToolCallSimulator
from mochi.backends.types import ToolSchema


def test_inject_tools_into_prompt_adds_tool_definitions() -> None:
    """應把工具定義注入 system prompt。"""
    simulator = ToolCallSimulator()
    system_prompt = "You are Mochi."
    tools = [
        ToolSchema(
            name="web_search",
            description="Search web pages",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        ),
        ToolSchema(
            name="memory_save",
            description="Save memory",
            parameters={"type": "object", "properties": {"content": {"type": "string"}}},
        ),
    ]

    injected = simulator.inject_tools_into_prompt(system_prompt, tools)

    assert injected.startswith(system_prompt)
    assert "web_search" in injected
    assert "memory_save" in injected
    assert "## Tool Use Instructions" in injected
    assert "Available tools:" in injected
    assert "Parameters:" in injected
    assert "<tool_call>" in injected
    assert re.search(r"[\u4e00-\u9fff]", injected) is None


def test_parse_tool_calls_supports_multiple_blocks_and_ignores_bad_json() -> None:
    """應可解析多個 tool_call，壞 JSON 要被忽略。"""
    simulator = ToolCallSimulator()
    llm_output = """
先呼叫工具。
<tool_call>{"name": "web_search", "arguments": {"query": "Mochi AI"}}</tool_call>
<tool_call>{not-valid-json}</tool_call>
<tool_call>{"name": "memory_save", "arguments": {"content": "done"}}</tool_call>
"""

    tool_calls = simulator.parse_tool_calls(llm_output)

    assert len(tool_calls) == 2
    assert [tc.name for tc in tool_calls] == ["web_search", "memory_save"]
    assert tool_calls[0].arguments == {"query": "Mochi AI"}
    assert tool_calls[1].arguments == {"content": "done"}


def test_parse_tool_calls_supports_function_wrapper_and_string_arguments() -> None:
    """應支援 OpenAI 風格 function 欄位與字串化 arguments。"""
    simulator = ToolCallSimulator()
    llm_output = (
        '<tool_call>{"function":{"name":"file_read","arguments":"{\\"path\\":\\"README.md\\"}"}}'
        "</tool_call>"
    )

    tool_calls = simulator.parse_tool_calls(llm_output)

    assert len(tool_calls) == 1
    assert tool_calls[0].name == "file_read"
    assert tool_calls[0].arguments == {"path": "README.md"}


def test_extract_text_response_removes_tool_call_blocks() -> None:
    """應能抽出不含 tool_call 區塊的純文字回答。"""
    simulator = ToolCallSimulator()
    llm_output = """
我先查一下資料。

<tool_call>{"name":"web_search","arguments":{"query":"python"}}</tool_call>

查完後我會整理重點回覆。
"""

    text = simulator.extract_text_response(llm_output)

    assert "<tool_call>" not in text
    assert "web_search" not in text
    assert "我先查一下資料。" in text
    assert "查完後我會整理重點回覆。" in text
