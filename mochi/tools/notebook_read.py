"""Read Jupyter notebooks from the local filesystem."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from mochi.config import defaults
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.utils.security import check_file_tool_path, normalize_workspace_dir


class NotebookReadTool(BaseTool):
    """Read notebook cells as structured records."""

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
        return "notebook_read"

    @property
    def description(self) -> str:
        return (
            "Read a local Jupyter notebook and return selected cells with "
            "their source and optional outputs."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local notebook path."},
                "offset": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Starting cell number using 1-based indexing.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of cells to return.",
                },
                "include_outputs": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to include simplified cell outputs.",
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
        return "Inspect notebook cells without opening the full .ipynb JSON manually."

    async def execute(
        self,
        *,
        path: str,
        offset: int = 1,
        limit: int | None = None,
        include_outputs: bool = False,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not path.strip():
            return ToolResult(error="`path` must not be empty.")
        if offset <= 0:
            return ToolResult(error="`offset` must be greater than 0.")
        if limit is not None and limit <= 0:
            return ToolResult(error="`limit` must be greater than 0.")

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
            payload = await asyncio.to_thread(
                self._read_notebook,
                target,
                offset,
                limit,
                include_outputs,
            )
        except json.JSONDecodeError as exc:
            return ToolResult(error=f"Notebook JSON parse failed: {exc}")

        if payload["total_cells"] > 0 and offset > payload["total_cells"]:
            return ToolResult(
                error=(
                    f"Notebook is shorter than the provided offset ({offset}). "
                    f"The notebook has {payload['total_cells']} cells."
                ),
                metadata={"path": str(target), "total_cells": payload["total_cells"]},
            )

        return ToolResult(
            output={"cells": payload["cells"]},
            metadata={
                "path": str(target),
                "total_cells": payload["total_cells"],
                "start_cell": payload["start_cell"],
                "end_cell": payload["end_cell"],
                "truncated": payload["truncated"],
                "include_outputs": include_outputs,
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
    def _read_notebook(
        path: Path,
        offset: int,
        limit: int | None,
        include_outputs: bool,
    ) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_cells = payload.get("cells")
        if not isinstance(raw_cells, list):
            raw_cells = []

        total_cells = len(raw_cells)
        start_idx = offset - 1
        selected = raw_cells[start_idx:] if limit is None else raw_cells[start_idx : start_idx + limit]
        cells: list[dict[str, Any]] = []
        for index, cell in enumerate(selected, start=offset):
            source = cell.get("source", [])
            if isinstance(source, list):
                source_text = "".join(str(part) for part in source)
            else:
                source_text = str(source)
            item: dict[str, Any] = {
                "index": index,
                "cell_type": str(cell.get("cell_type", "")),
                "source": source_text,
            }
            if include_outputs:
                outputs = cell.get("outputs", [])
                item["outputs"] = NotebookReadTool._simplify_outputs(outputs)
            cells.append(item)

        end_cell = offset + len(cells) - 1 if cells else offset - 1
        return {
            "cells": cells,
            "total_cells": total_cells,
            "start_cell": offset if cells or total_cells == 0 else offset,
            "end_cell": end_cell,
            "truncated": total_cells > end_cell if cells else False,
        }

    @staticmethod
    def _simplify_outputs(outputs: Any) -> list[str]:
        if not isinstance(outputs, list):
            return []
        simplified: list[str] = []
        for output in outputs:
            if not isinstance(output, dict):
                continue
            if "text" in output:
                text = output["text"]
                if isinstance(text, list):
                    simplified.append("".join(str(part) for part in text))
                else:
                    simplified.append(str(text))
                continue
            data = output.get("data")
            if isinstance(data, dict):
                text_value = data.get("text/plain")
                if isinstance(text_value, list):
                    simplified.append("".join(str(part) for part in text_value))
                elif text_value is not None:
                    simplified.append(str(text_value))
        return simplified
