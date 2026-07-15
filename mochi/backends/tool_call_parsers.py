"""Best-effort parsers for simulated backend tool-call markup."""

from __future__ import annotations

import html
import json
import re
import uuid
from typing import Any

from mochi.backends.types import ToolCall

TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL | re.IGNORECASE,
)
FUNCTION_RE = re.compile(
    r"<function\s*=\s*([A-Za-z_][\w.-]*)\s*>(.*?)</function>",
    re.DOTALL | re.IGNORECASE,
)
PARAMETER_RE = re.compile(
    r"<parameter\s*=\s*([A-Za-z_][\w.-]*)\s*>(.*?)</parameter>",
    re.DOTALL | re.IGNORECASE,
)


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Parse JSON and Qwen XML-ish ``<tool_call>`` blocks into ToolCall values."""
    results: list[ToolCall] = []
    if not text.strip():
        return results

    for match in TOOL_CALL_RE.finditer(text):
        payload = match.group(1).strip()
        for item in _decode_json_payload(payload):
            tool_call = _json_item_to_tool_call(item)
            if tool_call is not None:
                results.append(tool_call)
        results.extend(_decode_qwen_xml_payload(payload))

    return results


def strip_tool_call_blocks(text: str) -> str:
    """Remove tool-call markup blocks from model text."""
    return TOOL_CALL_RE.sub("", text)


def _decode_json_payload(payload: str) -> list[dict[str, Any]]:
    value: Any
    try:
        value = json.loads(payload)
    except json.JSONDecodeError:
        value = _try_raw_json_decode(payload)
        if value is None:
            return []

    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _try_raw_json_decode(text: str) -> Any | None:
    object_pos = text.find("{")
    array_pos = text.find("[")
    starts = [pos for pos in (object_pos, array_pos) if pos >= 0]
    if not starts:
        return None

    decoder = json.JSONDecoder()
    try:
        value, _ = decoder.raw_decode(text[min(starts) :])
        return value
    except json.JSONDecodeError:
        return None


def _json_item_to_tool_call(data: dict[str, Any]) -> ToolCall | None:
    name = ""
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

    arguments = _coerce_arguments(arguments_raw)
    return ToolCall(id=str(uuid.uuid4()), name=name, arguments=arguments)


def _decode_qwen_xml_payload(payload: str) -> list[ToolCall]:
    results: list[ToolCall] = []
    for function_match in FUNCTION_RE.finditer(payload):
        name = function_match.group(1).strip()
        body = function_match.group(2)
        arguments: dict[str, Any] = {}
        for parameter_match in PARAMETER_RE.finditer(body):
            key = parameter_match.group(1).strip()
            raw_value = html.unescape(parameter_match.group(2).strip())
            arguments[key] = coerce_tool_argument(raw_value)
        if name:
            results.append(ToolCall(id=str(uuid.uuid4()), name=name, arguments=arguments))
    return results


def _coerce_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def coerce_tool_argument(value: str) -> Any:
    """Safely coerce simple scalar strings without treating arbitrary text as code."""
    stripped = value.strip()
    lowered = stripped.lower()

    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"[+-]?\d+", stripped):
        try:
            return int(stripped)
        except ValueError:
            return stripped
    if re.fullmatch(r"[+-]?(?:\d+\.\d*|\.\d+)(?:[eE][+-]?\d+)?", stripped) or re.fullmatch(
        r"[+-]?\d+[eE][+-]?\d+",
        stripped,
    ):
        try:
            return float(stripped)
        except ValueError:
            return stripped

    if _looks_like_json(stripped):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return stripped

    return stripped


def _looks_like_json(value: str) -> bool:
    pairs = {("{", "}"), ("[", "]"), ('"', '"')}
    return any(value.startswith(start) and value.endswith(end) for start, end in pairs)
