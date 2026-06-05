"""Security helpers for memory tool content validation."""

from __future__ import annotations

import json
import re
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")

_INSTRUCTION_HIJACK_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore developer instructions",
    "ignore system instructions",
    "system prompt",
    "developer instructions",
)

_EXFILTRATION_ACTION_PATTERNS: tuple[str, ...] = (
    "reveal",
    "disclose",
    "expose",
    "dump",
    "print",
    "show",
)

_SECRET_TARGET_PATTERNS: tuple[str, ...] = (
    "api key",
    "api keys",
    "token",
    "password",
    "secret",
    "secrets",
    "credential",
    "credentials",
    ".env",
    "environment variable",
    "environment variables",
)


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return json.dumps(value, ensure_ascii=False)


def _normalize_text(value: Any) -> str:
    return _WHITESPACE_RE.sub(" ", _flatten_text(value).lower()).strip()


def is_suspicious_memory_payload(
    *,
    content: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> bool:
    """Return True for clear prompt-injection or secret-exfiltration patterns."""
    text = _normalize_text([content or "", metadata or {}])
    if not text:
        return False

    has_instruction_hijack = any(pattern in text for pattern in _INSTRUCTION_HIJACK_PATTERNS)
    has_exfiltration_action = any(pattern in text for pattern in _EXFILTRATION_ACTION_PATTERNS)
    has_secret_target = any(pattern in text for pattern in _SECRET_TARGET_PATTERNS)

    return (has_instruction_hijack and has_secret_target) or (
        "system prompt" in text and has_exfiltration_action
    )
