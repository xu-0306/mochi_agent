"""Workspace file discovery by glob pattern."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from mochi.config import defaults
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.utils.security import normalize_workspace_dir, resolve_path_in_workspace


class GlobSearchTool(BaseTool):
    """Find files in the workspace using glob patterns."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        default_limit: int = 100,
        max_limit: int = 1000,
    ) -> None:
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._default_limit = max(1, int(default_limit))
        self._max_limit = max(self._default_limit, int(max_limit))

    @property
    def name(self) -> str:
        return "glob_search"

    @property
    def description(self) -> str:
        return (
            "Find files in the workspace by glob pattern. Use this when you know roughly "
            "what filename or path shape you want, but not the exact location."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, for example 'src/*.py' or '**/*.md'."},
                "path": {
                    "type": "string",
                    "description": "Optional base directory inside the workspace. Defaults to the workspace root.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self._max_limit,
                    "description": "Maximum number of results to return.",
                },
            },
            "required": ["pattern"],
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
        return "Find files by path pattern before reading or editing them."

    async def execute(
        self,
        *,
        pattern: str,
        path: str | None = None,
        limit: int | None = None,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not pattern.strip():
            return ToolResult(error="`pattern` must not be empty.")

        workspace_root = self._resolve_workspace_root(context)
        try:
            search_root = resolve_path_in_workspace(path or ".", workspace_root)
        except ValueError as exc:
            return ToolResult(error=str(exc))

        effective_limit = self._default_limit if limit is None else int(limit)
        if effective_limit <= 0:
            return ToolResult(error="`limit` must be greater than 0.")
        effective_limit = min(effective_limit, self._max_limit)

        matches = await asyncio.to_thread(
            self._collect_matches,
            search_root,
            pattern,
            workspace_root,
            effective_limit,
        )
        truncated = len(matches) >= effective_limit
        return ToolResult(
            output=matches,
            metadata={
                "count": len(matches),
                "limit": effective_limit,
                "truncated": truncated,
                "path": str(search_root),
                "pattern": pattern,
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
    def _collect_matches(
        search_root: Path,
        pattern: str,
        workspace_root: Path,
        limit: int,
    ) -> list[str]:
        results: list[str] = []
        for candidate in sorted(search_root.glob(pattern)):
            if len(results) >= limit:
                break
            if not candidate.exists() or not candidate.is_file():
                continue
            resolved = candidate.resolve(strict=False)
            try:
                relative = resolved.relative_to(workspace_root)
            except ValueError:
                continue
            results.append(relative.as_posix())
        return results
