"""Shared prompt-simulated tool-calling helpers."""

from __future__ import annotations

import json
import uuid
from typing import Any

from mochi.backends.tool_call_simulator import ToolCallSimulator
from mochi.backends.types import Message, ToolCall, ToolSchema


class SimulatedToolProtocol:
    """Wrap prompt injection, message flattening, and simulated output parsing."""

    def __init__(self, simulator: ToolCallSimulator | None = None) -> None:
        self._simulator = simulator or ToolCallSimulator()

    def prepare_messages(
        self,
        *,
        messages: list[Message],
        tools: list[ToolSchema] | None,
    ) -> list[Message]:
        prepared = [self._copy_message(message) for message in messages]
        if not tools:
            return prepared
        return self._inject_tools_into_messages(
            self.flatten_messages(prepared),
            tools,
        )

    def flatten_messages(self, messages: list[Message]) -> list[Message]:
        flattened: list[Message] = []
        for message in messages:
            if message.role == "assistant" and message.tool_calls:
                if message.content:
                    flattened.append(Message(role="assistant", content=message.content))
                for tool_call in message.tool_calls:
                    flattened.append(
                        Message(
                            role="assistant",
                            content=(
                                f"Tool request: {tool_call.name}\n"
                                f"Arguments: {tool_call.arguments}"
                            ),
                        )
                    )
                continue

            if message.role == "tool":
                tool_name = message.name or "tool"
                tool_text = message.content
                if not tool_text.startswith(f"Tool {tool_name} result:") and not tool_text.startswith(
                    f"Tool {tool_name} error:"
                ):
                    tool_text = f"Tool {tool_name} result:\n{tool_text}"
                flattened.append(Message(role="user", content=tool_text))
                continue

            flattened.append(self._copy_message(message))
        return flattened

    def prepare_chat_payload_messages(
        self,
        *,
        raw_messages: list[Any],
        tools: list[ToolSchema] | None,
    ) -> list[dict[str, Any]]:
        messages = [
            self._message_from_payload(raw_message)
            for raw_message in raw_messages
            if isinstance(raw_message, dict)
        ]
        prepared = self.prepare_messages(messages=messages, tools=tools)
        return [self._message_to_payload(message) for message in prepared]

    def inject_tools_into_prompt(self, prompt: str, tools: list[ToolSchema] | None) -> str:
        if not tools:
            return prompt
        return self._simulator.inject_tools_into_prompt(prompt, tools)

    def parse_assistant_content(self, content: str) -> tuple[str, list[ToolCall]]:
        tool_calls = self._simulator.parse_tool_calls(content)
        if not tool_calls:
            return content, []
        return self._simulator.extract_text_response(content), tool_calls

    def _inject_tools_into_messages(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
    ) -> list[Message]:
        injected = False
        prepared = [self._copy_message(message) for message in messages]
        for message in prepared:
            if message.role != "system":
                continue
            message.content = self._simulator.inject_tools_into_prompt(message.content, tools)
            injected = True
            break

        if not injected:
            prepared.insert(
                0,
                Message(
                    role="system",
                    content=self._simulator.inject_tools_into_prompt("", tools).strip(),
                ),
            )
        return prepared

    def _message_from_payload(self, value: dict[str, Any]) -> Message:
        role = value.get("role")
        content = self._coerce_text(value.get("content"))
        if role not in {"system", "user", "assistant", "tool"}:
            role = "user"
        return Message(
            role=role,
            content=content,
            tool_calls=self._parse_payload_tool_calls(value.get("tool_calls")),
            tool_call_id=self._coerce_optional_text(value.get("tool_call_id")),
            name=self._coerce_optional_text(value.get("name") or value.get("tool_name")),
        )

    def _message_to_payload(self, message: Message) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "role": message.role,
            "content": message.content,
        }
        if message.tool_call_id:
            payload["tool_call_id"] = message.tool_call_id
        if message.name:
            if message.role == "tool":
                payload["tool_name"] = message.name
            else:
                payload["name"] = message.name
        if message.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                    },
                }
                for tool_call in message.tool_calls
            ]
        return payload

    @staticmethod
    def _copy_message(message: Message) -> Message:
        return Message(**message.__dict__)

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _coerce_optional_text(value: Any) -> str | None:
        if isinstance(value, str) and value:
            return value
        return None

    def _parse_payload_tool_calls(self, raw_tool_calls: Any) -> list[ToolCall]:
        if not isinstance(raw_tool_calls, list):
            return []

        tool_calls: list[ToolCall] = []
        for raw_item in raw_tool_calls:
            if not isinstance(raw_item, dict):
                continue
            function = raw_item.get("function", {})
            if not isinstance(function, dict):
                function = {}
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                continue

            raw_arguments = function.get("arguments", {})
            arguments: dict[str, Any]
            if isinstance(raw_arguments, dict):
                arguments = raw_arguments
            elif isinstance(raw_arguments, str):
                try:
                    decoded = json.loads(raw_arguments)
                except json.JSONDecodeError:
                    decoded = {}
                arguments = decoded if isinstance(decoded, dict) else {}
            else:
                arguments = {}

            call_id = raw_item.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = str(uuid.uuid4())
            tool_calls.append(ToolCall(id=call_id, name=name.strip(), arguments=arguments))
        return tool_calls
