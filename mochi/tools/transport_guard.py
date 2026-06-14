"""Backend-safe tool result transport guard."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import tempfile
from typing import Any
from uuid import uuid4

from mochi.tools.base import ToolExecutionContext, ToolResult


_WEB_EVIDENCE_TOOL_NAMES = frozenset({"web_fetch", "web_search"})
_WEB_EVIDENCE_ALLOWED_JSON_FLAGS = frozenset(
    {"json_envelope", "large_payload", "structured_payload"}
)


@dataclass
class ToolResultTransportOutcome:
    """Guarded backend-safe tool result plus diagnostics."""

    content: str
    diagnostics: dict[str, Any]


class ToolResultTransportGuard:
    """Normalize tool results before sending them into any backend."""

    def __init__(
        self,
        *,
        preview_chars: int = 480,
        persistence_multiplier: int = 4,
    ) -> None:
        self._preview_chars = max(120, preview_chars)
        self._persistence_multiplier = max(2, persistence_multiplier)

    def guard(
        self,
        *,
        tool_name: str,
        result: ToolResult,
        formatted_content: str,
        context: ToolExecutionContext | None,
        max_chars: int,
        backend_name: str,
        api_mode: str | None = None,
    ) -> ToolResultTransportOutcome:
        raw_payload = self._serialize_raw_payload(result)
        formatted_text = self._coerce_text(formatted_content)
        risk_flags = self._detect_risks(
            formatted_text=formatted_text,
            raw_payload=raw_payload,
            result=result,
        )

        diagnostics: dict[str, Any] = {
            "tool_name": tool_name,
            "backend_name": backend_name,
            "api_mode": api_mode,
            "formatted_length": len(formatted_text),
            "raw_length": len(raw_payload),
            "guarded_length": 0,
            "summary_applied": False,
            "overflow_persisted": False,
            "reference_id": None,
            "artifact_path": None,
            "risk_flags": risk_flags,
            "transport_type": "tool_result_text",
        }

        if self._is_plain_safe_text(
            tool_name=tool_name,
            formatted_text=formatted_text,
            max_chars=max_chars,
            risk_flags=risk_flags,
        ):
            diagnostics["guarded_length"] = len(formatted_text)
            self._collect_diagnostics(context, diagnostics)
            return ToolResultTransportOutcome(content=formatted_text, diagnostics=diagnostics)

        if self._is_safe_web_evidence_text(
            tool_name=tool_name,
            formatted_text=formatted_text,
            max_chars=max_chars,
            risk_flags=risk_flags,
            backend_name=backend_name,
        ):
            diagnostics["guarded_length"] = len(formatted_text)
            diagnostics["transport_type"] = "web_evidence_json"
            self._collect_diagnostics(context, diagnostics)
            return ToolResultTransportOutcome(content=formatted_text, diagnostics=diagnostics)

        candidate = self._summarize_result(tool_name=tool_name, result=result, max_chars=max_chars)
        diagnostics["summary_applied"] = True

        persist_needed = (
            len(candidate) > max_chars
            or len(raw_payload) > max_chars * self._persistence_multiplier
            or "large_payload" in risk_flags
        )
        if persist_needed:
            persisted = self._persist_payload(
                tool_name=tool_name,
                raw_payload=raw_payload,
                context=context,
                result=result,
            )
            if persisted is not None:
                reference_id, persisted_path, persisted_encoding = persisted
                diagnostics["overflow_persisted"] = True
                diagnostics["reference_id"] = reference_id
                diagnostics["artifact_path"] = str(persisted_path)
                candidate = self._build_reference_message(
                    tool_name=tool_name,
                    preview_text=self._truncate_text(
                        self._summarize_value(result.output),
                        self._preview_chars,
                    ),
                    raw_length=len(raw_payload),
                    reference_id=reference_id,
                )
                if context is not None:
                    context.tool_result_references[reference_id] = {
                        "reference_id": reference_id,
                        "artifact_path": str(persisted_path),
                        "tool_name": tool_name,
                        "encoding": persisted_encoding,
                    }

        diagnostics["guarded_length"] = len(candidate)
        self._collect_diagnostics(context, diagnostics)
        return ToolResultTransportOutcome(content=candidate, diagnostics=diagnostics)

    def _is_plain_safe_text(
        self,
        *,
        tool_name: str,
        formatted_text: str,
        max_chars: int,
        risk_flags: list[str],
    ) -> bool:
        if tool_name == "file_read":
            return (
                bool(formatted_text.strip())
                and len(formatted_text) <= max_chars
                and not risk_flags
            )
        return (
            bool(formatted_text.strip())
            and len(formatted_text) <= max_chars
            and not risk_flags
            and not formatted_text.lstrip().startswith("{")
            and not formatted_text.lstrip().startswith("[")
        )

    def _is_safe_web_evidence_text(
        self,
        *,
        tool_name: str,
        formatted_text: str,
        max_chars: int,
        risk_flags: list[str],
        backend_name: str,
    ) -> bool:
        stripped = formatted_text.lstrip()
        if backend_name != "openai_compat":
            return False
        if tool_name not in _WEB_EVIDENCE_TOOL_NAMES:
            return False
        if not stripped or len(formatted_text) > max_chars:
            return False
        if not (stripped.startswith("{") or stripped.startswith("[")):
            return False
        if not set(risk_flags).issubset(_WEB_EVIDENCE_ALLOWED_JSON_FLAGS):
            return False
        return self._try_parse_json(formatted_text) is not None

    def _detect_risks(
        self,
        *,
        formatted_text: str,
        raw_payload: str,
        result: ToolResult,
    ) -> list[str]:
        flags: list[str] = []
        stripped = formatted_text.lstrip()
        if stripped.startswith("{") or stripped.startswith("["):
            flags.append("structured_payload")
            parsed = self._try_parse_json(formatted_text)
            if isinstance(parsed, dict) and "ok" in parsed:
                flags.append("json_envelope")
        if len(raw_payload) > self._preview_chars * self._persistence_multiplier:
            flags.append("large_payload")
        if isinstance(result.output, (dict, list)):
            flags.append("structured_payload")
        return list(dict.fromkeys(flags))

    def _summarize_result(
        self,
        *,
        tool_name: str,
        result: ToolResult,
        max_chars: int,
    ) -> str:
        preview_budget = max(120, min(self._preview_chars, max_chars - 64))
        if result.error:
            body = self._truncate_text(self._normalize_whitespace(result.error), preview_budget)
            return f"Tool {tool_name} error:\n{body}"

        body = self._truncate_text(self._summarize_value(result.output), preview_budget)
        prefix = f"Tool {tool_name} result:"
        if not body:
            return prefix
        return f"{prefix}\n{body}"

    def _summarize_value(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return self._normalize_whitespace(value)
        if isinstance(value, (int, float, bool)):
            return str(value)
        if isinstance(value, dict):
            parts: list[str] = []
            for key, item in value.items():
                key_text = str(key).strip()
                item_text = self._summarize_value(item)
                if not item_text:
                    continue
                sep = "\n" if "\n" in item_text else " "
                parts.append(f"{key_text}:{sep}{item_text}".strip())
            return "\n".join(parts)
        if isinstance(value, list):
            parts = [self._summarize_value(item) for item in value[:8]]
            parts = [part for part in parts if part]
            if len(value) > 8:
                parts.append(f"... ({len(value) - 8} more items)")
            return "\n".join(f"- {part}" for part in parts)
        return self._normalize_whitespace(str(value))

    def _persist_payload(
        self,
        *,
        tool_name: str,
        raw_payload: str,
        context: ToolExecutionContext | None,
        result: ToolResult | None = None,
    ) -> tuple[str, Path, str] | None:
        target_dir = self._resolve_store_dir(context)
        target_dir.mkdir(parents=True, exist_ok=True)
        reference_id = f"{tool_name}-{uuid4().hex[:10]}"
        persisted_text = raw_payload
        persisted_encoding = "utf-8"
        suffix = ".json" if raw_payload.lstrip().startswith(("{", "[")) else ".txt"
        if result is not None and result.error is None and isinstance(result.output, str):
            persisted_text = result.output
            suffix = ".txt"
        target_path = target_dir / f"{reference_id}{suffix}"
        target_path.write_text(persisted_text, encoding=persisted_encoding)
        return reference_id, target_path, persisted_encoding

    def _resolve_store_dir(self, context: ToolExecutionContext | None) -> Path:
        if context is not None and context.tool_result_store_dir:
            return Path(context.tool_result_store_dir)

        session_part = context.session_id if context is not None and context.session_id else "default"
        return Path(tempfile.gettempdir()) / "mochi-tool-results" / session_part

    def _build_reference_message(
        self,
        *,
        tool_name: str,
        preview_text: str,
        raw_length: int,
        reference_id: str,
    ) -> str:
        message = f"Tool {tool_name} result preview (truncated from {raw_length} chars).\n"
        if preview_text:
            message += preview_text + "\n"
        message += (
            f"Reference: {reference_id}\n"
            f'To continue reading, call: file_read(path="tool-result://{reference_id}", '
            'offset=1, limit=200, line_numbers=True)'
        )
        return message.strip()

    @staticmethod
    def _collect_diagnostics(
        context: ToolExecutionContext | None,
        diagnostics: dict[str, Any],
    ) -> None:
        if context is None:
            return
        context.transport_diagnostics.append(dict(diagnostics))

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _serialize_raw_payload(result: ToolResult) -> str:
        payload = {
            "output": result.output,
            "error": result.error,
            "metadata": result.metadata,
            "retryable": result.retryable,
            "suggestion": result.suggestion,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def _normalize_whitespace(value: str) -> str:
        import re

        collapsed = re.sub(r"\r\n?", "\n", value)
        collapsed = re.sub(r"[ \t]+", " ", collapsed)
        collapsed = re.sub(r"\n{3,}", "\n\n", collapsed)
        return collapsed.strip()

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        suffix = "...[truncated]"
        if max_chars <= len(suffix):
            return suffix[:max_chars]
        return value[: max_chars - len(suffix)] + suffix

    @staticmethod
    def _try_parse_json(value: str) -> Any | None:
        try:
            return json.loads(value)
        except Exception:
            return None
