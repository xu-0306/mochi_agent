"""Tool Call Simulator — 為不支援原生 function calling 的模型模擬工具呼叫。"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from mochi.backends.types import ToolCall, ToolSchema


class ToolCallSimulator:
    """透過 prompt 注入模擬 tool calling。

    當模型不支援原生 function calling 時，
    將工具定義注入 system prompt，解析 LLM 文字輸出中的 JSON tool call 區塊。
    """

    TOOL_PROMPT_TEMPLATE = (
        "\n\n## Tool Use Instructions\n"
        "The following tools are available for completing the task.\n"
        "To call a tool, output one or more <tool_call> blocks. Each block must contain valid JSON:\n"
        "<tool_call>\n"
        '{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}\n'
        "</tool_call>\n"
        "If multiple tool calls are needed in one turn, output multiple <tool_call> blocks in sequence.\n\n"
        "Available tools:\n{tool_definitions}"
    )

    TOOL_CALL_RE = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        re.DOTALL | re.IGNORECASE,
    )

    def inject_tools_into_prompt(self, system_prompt: str, tools: list[ToolSchema]) -> str:
        """將工具定義注入 system prompt。

        Args:
            system_prompt: 原始系統提示詞。
            tools: 工具列表。

        Returns:
            注入工具定義後的系統提示詞。
        """
        if not tools:
            return system_prompt

        definitions = "\n".join(
            f"- **{t.name}**: {t.description}\n"
            f"  Parameters: {json.dumps(t.parameters, ensure_ascii=False)}"
            for t in tools
        )
        injection = self.TOOL_PROMPT_TEMPLATE.format(tool_definitions=definitions)
        return system_prompt + injection

    def parse_tool_calls(self, llm_output: str) -> list[ToolCall]:
        """從 LLM 文字輸出中解析工具呼叫。

        Args:
            llm_output: LLM 原始輸出文字。

        Returns:
            解析到的 ToolCall 列表（可能為空）。
        """
        results: list[ToolCall] = []
        if not llm_output.strip():
            return results

        for match in self.TOOL_CALL_RE.finditer(llm_output):
            payload = match.group(1).strip()
            for item in self._decode_tool_payload(payload):
                tool_call = self._to_tool_call(item)
                if tool_call is not None:
                    results.append(tool_call)
        return results

    def extract_text_response(self, llm_output: str) -> str:
        """抽取 LLM 的純文字回答（移除所有 <tool_call> 區塊）。"""
        if not llm_output.strip():
            return ""

        text_only = self.TOOL_CALL_RE.sub("", llm_output)
        lines = [line.rstrip() for line in text_only.splitlines()]
        collapsed = "\n".join(lines).strip()
        return re.sub(r"\n{3,}", "\n\n", collapsed)

    def _decode_tool_payload(self, payload: str) -> list[dict[str, Any]]:
        """解析單一 <tool_call> 區塊中的 JSON payload。"""
        value: Any
        try:
            value = json.loads(payload)
        except json.JSONDecodeError:
            value = self._try_raw_decode(payload)
            if value is None:
                return []

        if isinstance(value, dict):
            return [value]
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        return []

    def _try_raw_decode(self, text: str) -> Any | None:
        """嘗試從文字中擷取第一個 JSON 值。"""
        object_pos = text.find("{")
        array_pos = text.find("[")
        starts = [pos for pos in (object_pos, array_pos) if pos >= 0]
        if not starts:
            return None

        start = min(starts)
        decoder = json.JSONDecoder()
        try:
            value, _ = decoder.raw_decode(text[start:])
            return value
        except json.JSONDecodeError:
            return None

    def _to_tool_call(self, data: dict[str, Any]) -> ToolCall | None:
        """將解析出的字典轉為 ToolCall。"""
        name: str = ""
        arguments_raw: Any = {}

        if isinstance(data.get("name"), str):
            name = data["name"].strip()
            arguments_raw = data.get("arguments", {})
        elif isinstance(data.get("function"), dict):
            function_data = data["function"]
            if isinstance(function_data.get("name"), str):
                name = function_data["name"].strip()
            arguments_raw = function_data.get("arguments", {})

        if not name:
            return None

        arguments: dict[str, Any]
        if isinstance(arguments_raw, dict):
            arguments = arguments_raw
        elif isinstance(arguments_raw, str):
            try:
                parsed = json.loads(arguments_raw)
                arguments = parsed if isinstance(parsed, dict) else {}
            except json.JSONDecodeError:
                arguments = {}
        else:
            arguments = {}

        return ToolCall(id=str(uuid.uuid4()), name=name, arguments=arguments)
