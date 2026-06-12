"""Read CSV files from the workspace with preview-friendly output."""

from __future__ import annotations

import asyncio
import csv
from pathlib import Path
from typing import Any

from mochi.config import defaults
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.utils.security import check_file_tool_path, normalize_workspace_dir


class CsvReadTool(BaseTool):
    """Read CSV files as structured preview rows."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        path_scope: str = "workspace",
        default_encoding: str = "utf-8",
        default_row_limit: int = 50,
        max_row_limit: int = 1000,
    ) -> None:
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._path_scope = path_scope
        self._default_encoding = default_encoding
        self._default_row_limit = max(1, int(default_row_limit))
        self._max_row_limit = max(self._default_row_limit, int(max_row_limit))

    @property
    def name(self) -> str:
        return "csv_read"

    @property
    def description(self) -> str:
        return (
            "Read a local CSV file and return a structured preview with "
            "column names and rows."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local CSV file path."},
                "encoding": {"type": "string", "default": "utf-8"},
                "delimiter": {
                    "type": "string",
                    "default": ",",
                    "description": "Single-character CSV delimiter.",
                },
                "row_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self._max_row_limit,
                    "description": "Maximum number of rows to return.",
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
        return "Preview tabular CSV data before filtering or further processing."

    async def execute(
        self,
        *,
        path: str,
        encoding: str | None = None,
        delimiter: str = ",",
        row_limit: int | None = None,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not path.strip():
            return ToolResult(error="`path` must not be empty.")
        if len(delimiter) != 1:
            return ToolResult(error="`delimiter` must be a single character.")

        effective_limit = self._default_row_limit if row_limit is None else int(row_limit)
        if effective_limit <= 0:
            return ToolResult(error="`row_limit` must be greater than 0.")
        effective_limit = min(effective_limit, self._max_row_limit)

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

        active_encoding = encoding or self._default_encoding
        try:
            payload = await asyncio.to_thread(
                self._read_csv,
                target,
                active_encoding,
                delimiter,
                effective_limit,
            )
        except UnicodeDecodeError:
            return ToolResult(error=f"File is not valid {active_encoding} text: {target}")
        except csv.Error as exc:
            return ToolResult(error=f"CSV parse failed: {exc}")

        return ToolResult(
            output={
                "columns": payload["columns"],
                "rows": payload["rows"],
            },
            metadata={
                "path": str(target),
                "row_count": len(payload["rows"]),
                "total_rows": payload["total_rows"],
                "truncated": payload["total_rows"] > len(payload["rows"]),
                "delimiter": delimiter,
                "encoding": active_encoding,
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
    def _read_csv(
        path: Path,
        encoding: str,
        delimiter: str,
        row_limit: int,
    ) -> dict[str, Any]:
        with path.open("r", encoding=encoding, newline="") as handle:
            reader = csv.DictReader(handle, delimiter=delimiter)
            columns = list(reader.fieldnames or [])
            rows: list[dict[str, str]] = []
            total_rows = 0
            for row in reader:
                total_rows += 1
                if len(rows) < row_limit:
                    rows.append(
                        {
                            str(key): "" if value is None else str(value)
                            for key, value in row.items()
                            if key is not None
                        }
                    )
        return {
            "columns": columns,
            "rows": rows,
            "total_rows": total_rows,
        }
