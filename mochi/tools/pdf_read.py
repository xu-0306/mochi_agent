"""Read PDF text from the workspace."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Callable

from pypdf import PdfReader

from mochi.config import defaults
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.utils.security import check_file_tool_path, normalize_workspace_dir

PdfReaderFactory = Callable[[Path], Any]


def extract_pdf_pages(
    path: Path,
    *,
    page_range: str | None = None,
    max_chars: int | None = None,
    reader_factory: PdfReaderFactory | None = None,
) -> dict[str, Any]:
    """Extract text from selected PDF pages."""
    active_reader_factory = reader_factory or PdfReadTool._default_reader_factory
    reader = active_reader_factory(path)
    pages = list(getattr(reader, "pages", []))
    page_count = len(pages)
    indices = PdfReadTool._parse_page_range(page_range, page_count)

    extracted: list[dict[str, Any]] = []
    consumed = 0
    truncated = False
    for index in indices:
        text = pages[index - 1].extract_text() or ""
        if max_chars is not None:
            remaining = max_chars - consumed
            if remaining <= 0:
                truncated = True
                break
            if len(text) > remaining:
                extracted.append({"page": index, "text": text[:remaining]})
                truncated = True
                break
        extracted.append({"page": index, "text": text})
        consumed += len(text)

    return {
        "pages": extracted,
        "page_count": page_count,
        "truncated": truncated,
    }


class PdfReadTool(BaseTool):
    """Read text from selected PDF pages."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        path_scope: str = "workspace",
        reader_factory: PdfReaderFactory | None = None,
    ) -> None:
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._path_scope = path_scope
        self._reader_factory = reader_factory or self._default_reader_factory

    @property
    def name(self) -> str:
        return "pdf_read"

    @property
    def description(self) -> str:
        return (
            "Read text from a PDF file in the workspace and return extracted text by page."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "PDF file path inside the workspace."},
                "page_range": {
                    "type": "string",
                    "description": "Optional page selection such as '1-3' or '2,4-5'. Defaults to all pages.",
                },
                "max_chars": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum total characters to return across selected pages.",
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
        return "Extract PDF text by page before deeper analysis or summarization."

    async def execute(
        self,
        *,
        path: str,
        page_range: str | None = None,
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
            payload = await asyncio.to_thread(
                extract_pdf_pages,
                target,
                page_range=page_range,
                max_chars=max_chars,
                reader_factory=self._reader_factory,
            )
        except ValueError as exc:
            return ToolResult(error=str(exc))

        return ToolResult(
            output={"pages": payload["pages"]},
            metadata={
                "path": str(target),
                "page_count": payload["page_count"],
                "pages_returned": len(payload["pages"]),
                "truncated": payload["truncated"],
                "page_range": page_range,
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

    @staticmethod
    def _default_reader_factory(path: Path) -> PdfReader:
        return PdfReader(str(path))

    @staticmethod
    def _parse_page_range(page_range: str | None, page_count: int) -> list[int]:
        if page_count == 0:
            return []
        if page_range is None or not page_range.strip():
            return list(range(1, page_count + 1))

        selected: list[int] = []
        for part in page_range.split(","):
            token = part.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start = int(start_text)
                end = int(end_text)
                if start <= 0 or end <= 0 or end < start:
                    raise ValueError("`page_range` must contain valid positive page intervals.")
                selected.extend(range(start, end + 1))
                continue
            page = int(token)
            if page <= 0:
                raise ValueError("`page_range` must contain only positive page numbers.")
            selected.append(page)

        if not selected:
            raise ValueError("`page_range` did not select any pages.")
        if max(selected) > page_count:
            raise ValueError(
                f"`page_range` references page {max(selected)} but the PDF has {page_count} pages."
            )
        return selected
