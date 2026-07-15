"""Tool-call parser fixture tests."""

from __future__ import annotations

from mochi.backends.tool_call_parsers import parse_tool_calls
from mochi.backends.tool_call_simulator import ToolCallSimulator


def test_parse_tool_calls_preserves_json_single_call() -> None:
    tool_calls = parse_tool_calls(
        '<tool_call>{"name": "web_search", "arguments": {"query": "Mochi AI"}}</tool_call>'
    )

    assert len(tool_calls) == 1
    assert tool_calls[0].name == "web_search"
    assert tool_calls[0].arguments == {"query": "Mochi AI"}


def test_parse_tool_calls_preserves_json_multi_call_list() -> None:
    tool_calls = parse_tool_calls(
        """
<tool_call>
[
  {"name": "web_search", "arguments": {"query": "Mochi AI"}},
  {"name": "memory_save", "arguments": {"content": "done"}}
]
</tool_call>
"""
    )

    assert [tc.name for tc in tool_calls] == ["web_search", "memory_save"]
    assert tool_calls[0].arguments == {"query": "Mochi AI"}
    assert tool_calls[1].arguments == {"content": "done"}


def test_parse_tool_calls_supports_qwen_xml_single_call_with_coercion() -> None:
    tool_calls = parse_tool_calls(
        """
<tool_call> <function=arxiv_search>
<parameter=query>"medical imaging"</parameter>
<parameter=max_results>5</parameter>
<parameter=include_abstracts>true</parameter>
<parameter=threshold>1.5</parameter>
<parameter=filters>{"year": 2026, "open_access": false}</parameter>
</function> </tool_call>
"""
    )

    assert len(tool_calls) == 1
    assert tool_calls[0].name == "arxiv_search"
    assert tool_calls[0].arguments == {
        "query": "medical imaging",
        "max_results": 5,
        "include_abstracts": True,
        "threshold": 1.5,
        "filters": {"year": 2026, "open_access": False},
    }


def test_parse_tool_calls_supports_qwen_xml_single_line() -> None:
    tool_calls = parse_tool_calls(
        "<tool_call> <function=arxiv_search> <parameter=query>medical imaging</parameter> </function> </tool_call>"
    )

    assert len(tool_calls) == 1
    assert tool_calls[0].name == "arxiv_search"
    assert tool_calls[0].arguments == {"query": "medical imaging"}


def test_parse_tool_calls_supports_qwen_xml_multiple_functions() -> None:
    tool_calls = parse_tool_calls(
        """
<tool_call>
  <function=web_search>
    <parameter=query>Mochi runtime</parameter>
  </function>
  <function=memory_save>
    <parameter=content>done</parameter>
  </function>
</tool_call>
"""
    )

    assert [tc.name for tc in tool_calls] == ["web_search", "memory_save"]
    assert tool_calls[0].arguments == {"query": "Mochi runtime"}
    assert tool_calls[1].arguments == {"content": "done"}


def test_parse_tool_calls_ignores_malformed_qwen_xml() -> None:
    tool_calls = parse_tool_calls(
        """
<tool_call>
  <function=web_search>
    <parameter=query>Mochi runtime</parameter>
</tool_call>
"""
    )

    assert tool_calls == []


def test_tool_call_simulator_delegates_to_multi_format_parser() -> None:
    simulator = ToolCallSimulator()

    tool_calls = simulator.parse_tool_calls(
        "<tool_call><function=file_read><parameter=path>README.md</parameter></function></tool_call>"
    )

    assert len(tool_calls) == 1
    assert tool_calls[0].name == "file_read"
    assert tool_calls[0].arguments == {"path": "README.md"}
