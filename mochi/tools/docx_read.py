"""Read DOCX text from the local filesystem."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
import zipfile

from mochi.config import defaults
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.utils.security import check_file_tool_path, normalize_workspace_dir

_WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def extract_docx_paragraphs(path: Path, max_chars: int | None = None) -> dict[str, Any]:
    """Extract paragraph text from a DOCX file."""
    try:
        with zipfile.ZipFile(path) as archive:
            try:
                document_xml = archive.read("word/document.xml")
            except KeyError as exc:
                raise ValueError("DOCX file is missing word/document.xml.") from exc
    except zipfile.BadZipFile as exc:
        raise ValueError(f"File is not a valid DOCX archive: {path}") from exc

    try:
        root = ET.fromstring(document_xml)
    except ET.ParseError as exc:
        raise ValueError("DOCX document.xml is not valid XML.") from exc

    paragraphs: list[str] = []
    consumed = 0
    truncated = False

    for paragraph in root.findall(".//w:p", _WORD_NAMESPACE):
        text_parts = [node.text or "" for node in paragraph.findall(".//w:t", _WORD_NAMESPACE)]
        paragraph_text = "".join(text_parts).strip()
        if not paragraph_text:
            continue

        if max_chars is not None:
            remaining = max_chars - consumed
            if remaining <= 0:
                truncated = True
                break
            if len(paragraph_text) > remaining:
                paragraphs.append(paragraph_text[:remaining])
                truncated = True
                break

        paragraphs.append(paragraph_text)
        consumed += len(paragraph_text)

    return {
        "paragraphs": paragraphs,
        "paragraph_count": len(paragraphs),
        "truncated": truncated,
    }


class DocxReadTool(BaseTool):
    """Read paragraph text from a DOCX file."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        path_scope: str = "workspace",
    ) -> None:
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._path_scope = path_scope

    @property
    def name(self) -> str:
        return "docx_read"

    @property
    def description(self) -> str:
        return (
            "Read text from a local DOCX Word document and return extracted paragraphs."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local DOCX file path."},
                "max_chars": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum total characters to return across extracted paragraphs.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    @property
    def search_hint(self) -> str | None:
        return "Extract DOCX paragraph text before summarization or review."

    async def execute(
        self,
        *,
        path: str,
        max_chars: int | None = None,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not path.strip():
            return ToolResult(error="`path` must not be empty.")
        if max_chars is not None and max_chars <= 0:
            return ToolResult(error="`max_chars` must be greater than 0.")

        workspace_root = self._resolve_workspace_root(context)
        target, security_decision = check_file_tool_path(
            path,
            workspace_dir=workspace_root,
            scope=self._path_scope,
            access="read",
        )
        if security_decision is not None or target is None:
            return ToolResult(
                error=security_decision.reason if security_decision is not None else "Path denied.",
                metadata=security_decision.to_metadata() if security_decision is not None else {},
            )
        if not target.exists():
            return ToolResult(error=f"File not found: {target}")
        if not target.is_file():
            return ToolResult(error=f"Path is not a file: {target}")

        try:
            payload = await asyncio.to_thread(extract_docx_paragraphs, target, max_chars)
        except ValueError as exc:
            return ToolResult(error=str(exc))

        return ToolResult(
            output={"paragraphs": payload["paragraphs"]},
            metadata={
                "path": str(target),
                "paragraph_count": payload["paragraph_count"],
                "paragraphs_returned": len(payload["paragraphs"]),
                "truncated": payload["truncated"],
                "max_chars": max_chars,
            },
        )

    def _resolve_workspace_root(self, context: ToolExecutionContext | None) -> Path:
        if context is not None:
            for candidate in (
                context.task_sandbox_dir,
                context.project_workspace,
                context.workspace_dir,
            ):
                if candidate:
                    return normalize_workspace_dir(candidate)
        return self._workspace_dir
