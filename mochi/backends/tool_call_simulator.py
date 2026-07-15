"""Tool Call Simulator for backends without native function calling support."""

from __future__ import annotations

import json
import re

from mochi.backends.tool_call_parsers import (
    TOOL_CALL_RE,
    parse_tool_calls,
    strip_tool_call_blocks,
)
from mochi.backends.types import ToolCall, ToolSchema


class ToolCallSimulator:
    """Inject tool instructions and parse model-emitted tool call markup."""

    JSON_TOOL_PROMPT_TEMPLATE = (
        "\n\n## Tool Use Instructions\n"
        "The following tools are available for completing the task.\n"
        "To call a tool, output one or more <tool_call> blocks. Each block must contain valid JSON:\n"
        "<tool_call>\n"
        '{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}\n'
        "</tool_call>\n"
        "If multiple tool calls are needed in one turn, output multiple <tool_call> blocks in sequence.\n\n"
        "Available tools:\n{tool_definitions}"
    )
    QWEN_XML_TOOL_PROMPT_TEMPLATE = (
        "\n\n## Tool Use Instructions\n"
        "The following tools are available for completing the task.\n"
        "To call a tool, output one or more <tool_call> blocks. For Qwen-style local models, "
        "the preferred format is XML-ish function markup:\n"
        "<tool_call>\n"
        "<function=tool_name>\n"
        "<parameter=arg1>value1</parameter>\n"
        "</function>\n"
        "</tool_call>\n"
        "Use one <parameter=name>value</parameter> element per argument. Do not include prose inside "
        "a <tool_call> block. JSON <tool_call> blocks are also accepted if that is easier.\n\n"
        "Available tools:\n{tool_definitions}"
    )
    TOOL_PROMPT_TEMPLATE = JSON_TOOL_PROMPT_TEMPLATE

    TOOL_CALL_RE = TOOL_CALL_RE

    def __init__(self, *, tool_prompt_profile: str = "json_tool_call") -> None:
        self.tool_prompt_profile = self._normalize_tool_prompt_profile(tool_prompt_profile)

    def inject_tools_into_prompt(
        self,
        system_prompt: str,
        tools: list[ToolSchema],
        *,
        tool_prompt_profile: str | None = None,
    ) -> str:
        """Inject tool definitions into the system prompt."""
        if not tools:
            return system_prompt

        profile = self._normalize_tool_prompt_profile(tool_prompt_profile or self.tool_prompt_profile)
        definitions = "\n".join(
            f"- **{t.name}**: {t.description}\n"
            f"  Parameters: {json.dumps(t.parameters, ensure_ascii=False)}"
            for t in tools
        )
        template = (
            self.QWEN_XML_TOOL_PROMPT_TEMPLATE
            if profile == "qwen_xml_tool_call"
            else self.JSON_TOOL_PROMPT_TEMPLATE
        )
        injection = template.format(tool_definitions=definitions)
        return system_prompt + injection

    @staticmethod
    def _normalize_tool_prompt_profile(profile: str | None) -> str:
        normalized = str(profile or "").strip().lower().replace("-", "_")
        if normalized in {"qwen", "qwen_xml", "qwen_xml_tool_call", "xml_tool_call"}:
            return "qwen_xml_tool_call"
        return "json_tool_call"

    def parse_tool_calls(self, llm_output: str) -> list[ToolCall]:
        """Extract tool calls from LLM output."""
        return parse_tool_calls(llm_output)

    def extract_text_response(self, llm_output: str) -> str:
        """Return model text with any ``<tool_call>`` blocks removed."""
        if not llm_output.strip():
            return ""

        text_only = strip_tool_call_blocks(llm_output)
        lines = [line.rstrip() for line in text_only.splitlines()]
        collapsed = "\n".join(lines).strip()
        return re.sub(r"\n{3,}", "\n\n", collapsed)
