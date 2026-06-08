"""Shared helper utilities for multi-agent orchestration."""

from __future__ import annotations

import json
from typing import Any


def parse_json_payload(content: str) -> Any:
    """Parse a JSON object or array from free-form model output."""

    text = content.strip()
    if not text:
        return None
    candidates = [text]
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            stripped = part.strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].strip()
            if stripped:
                candidates.append(stripped)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        object_start = text.find("{")
        object_end = text.rfind("}")
        if 0 <= object_start < object_end:
            candidates.append(text[object_start : object_end + 1])
        array_start = text.find("[")
        array_end = text.rfind("]")
        if 0 <= array_start < array_end:
            candidates.append(text[array_start : array_end + 1])
        for candidate in candidates[1:]:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return None
