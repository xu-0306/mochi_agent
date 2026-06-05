"""Workspace content search using ripgrep when available, with Python fallback."""

from __future__ import annotations

import asyncio
from pathlib import Path
import re
import shutil
from typing import Any

from mochi.config import defaults
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.utils.security import normalize_workspace_dir, resolve_path_in_workspace


class GrepSearchTool(BaseTool):
    """Search file contents within the workspace."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        default_head_limit: int = 100,
        max_head_limit: int = 1000,
    ) -> None:
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._default_head_limit = max(1, int(default_head_limit))
        self._max_head_limit = max(self._default_head_limit, int(max_head_limit))

    @property
    def name(self) -> str:
        return "grep_search"

    @property
    def description(self) -> str:
        return (
            "Search file contents in the workspace with a regular expression. Use this "
            "to find matching lines or the set of files containing a pattern."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Regular expression to search for."},
                "path": {
                    "type": "string",
                    "description": "Optional file or directory inside the workspace. Defaults to the workspace root.",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional file glob filter such as '*.py' or '**/*.md'.",
                },
                "output_mode": {
                    "type": "string",
                    "enum": ["content", "files_with_matches"],
                    "default": "files_with_matches",
                },
                "head_limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self._max_head_limit,
                    "description": "Maximum number of matches or files to return.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
                    "description": "Skip the first N matches before returning results.",
                },
                "ignore_case": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to search case-insensitively.",
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
        return "Search file contents before deciding which file to read or edit."

    async def execute(
        self,
        *,
        pattern: str,
        path: str | None = None,
        glob: str | None = None,
        output_mode: str = "files_with_matches",
        head_limit: int | None = None,
        offset: int = 0,
        ignore_case: bool = False,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not pattern.strip():
            return ToolResult(error="`pattern` must not be empty.")
        if output_mode not in {"content", "files_with_matches"}:
            return ToolResult(error="`output_mode` must be 'content' or 'files_with_matches'.")
        if offset < 0:
            return ToolResult(error="`offset` must be greater than or equal to 0.")

        workspace_root = self._resolve_workspace_root(context)
        try:
            search_root = resolve_path_in_workspace(path or ".", workspace_root)
        except ValueError as exc:
            return ToolResult(error=str(exc))

        effective_limit = self._default_head_limit if head_limit is None else int(head_limit)
        if effective_limit <= 0:
            return ToolResult(error="`head_limit` must be greater than 0.")
        effective_limit = min(effective_limit, self._max_head_limit)

        results = await self._search(
            pattern=pattern,
            search_root=search_root,
            workspace_root=workspace_root,
            glob_pattern=glob,
            output_mode=output_mode,
            head_limit=effective_limit,
            offset=offset,
            ignore_case=ignore_case,
        )
        if isinstance(results, ToolResult):
            return results
        return ToolResult(
            output=results,
            metadata={
                "count": len(results),
                "pattern": pattern,
                "path": str(search_root),
                "output_mode": output_mode,
                "head_limit": effective_limit,
                "offset": offset,
                "glob": glob,
                "engine": "rg" if shutil.which("rg") else "python",
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

    async def _search(
        self,
        *,
        pattern: str,
        search_root: Path,
        workspace_root: Path,
        glob_pattern: str | None,
        output_mode: str,
        head_limit: int,
        offset: int,
        ignore_case: bool,
    ) -> list[str] | ToolResult:
        if shutil.which("rg"):
            return await self._search_with_rg(
                pattern=pattern,
                search_root=search_root,
                workspace_root=workspace_root,
                glob_pattern=glob_pattern,
                output_mode=output_mode,
                head_limit=head_limit,
                offset=offset,
                ignore_case=ignore_case,
            )
        return await asyncio.to_thread(
            self._search_with_python,
            pattern,
            search_root,
            workspace_root,
            glob_pattern,
            output_mode,
            head_limit,
            offset,
            ignore_case,
        )

    async def _search_with_rg(
        self,
        *,
        pattern: str,
        search_root: Path,
        workspace_root: Path,
        glob_pattern: str | None,
        output_mode: str,
        head_limit: int,
        offset: int,
        ignore_case: bool,
    ) -> list[str] | ToolResult:
        command = ["rg", "--color", "never", "--with-filename", "--line-number", "--sort", "path"]
        if ignore_case:
            command.append("-i")
        if output_mode == "files_with_matches":
            command.append("--files-with-matches")
        command.append(pattern)
        if glob_pattern:
            command.extend(["--glob", glob_pattern])
        command.append(str(search_root))

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode not in {0, 1}:
            return ToolResult(error=(stderr.decode("utf-8", errors="replace").strip() or "grep_search failed."))

        decoded = stdout.decode("utf-8", errors="replace").splitlines()
        if process.returncode == 1 and not decoded:
            return []

        normalized: list[str] = []
        for line in decoded:
            if not line.strip():
                continue
            if output_mode == "files_with_matches":
                candidate = Path(line.strip()).resolve(strict=False)
                try:
                    normalized.append(candidate.relative_to(workspace_root).as_posix())
                except ValueError:
                    continue
                continue
            match = re.match(r"^(.*):(\d+):(.*)$", line)
            if match is None:
                continue
            candidate = Path(match.group(1)).resolve(strict=False)
            try:
                relative = candidate.relative_to(workspace_root).as_posix()
            except ValueError:
                continue
            normalized.append(f"{relative}:{match.group(2)}: {match.group(3)}")

        return normalized[offset : offset + head_limit]

    @staticmethod
    def _search_with_python(
        pattern: str,
        search_root: Path,
        workspace_root: Path,
        glob_pattern: str | None,
        output_mode: str,
        head_limit: int,
        offset: int,
        ignore_case: bool,
    ) -> list[str]:
        flags = re.IGNORECASE if ignore_case else 0
        regex = re.compile(pattern, flags)
        if search_root.is_file():
            candidates = [search_root]
        else:
            iterator_pattern = glob_pattern or "**/*"
            candidates = sorted(search_root.glob(iterator_pattern))

        matches: list[str] = []
        seen_files: set[str] = set()
        for candidate in candidates:
            if not candidate.exists() or not candidate.is_file():
                continue
            try:
                relative = candidate.resolve(strict=False).relative_to(workspace_root).as_posix()
            except ValueError:
                continue
            try:
                lines = candidate.read_text(encoding="utf-8").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            matched = False
            for line_no, line in enumerate(lines, start=1):
                if regex.search(line) is None:
                    continue
                matched = True
                if output_mode == "content":
                    matches.append(f"{relative}:{line_no}: {line}")
                if output_mode == "files_with_matches":
                    break
            if output_mode == "files_with_matches" and matched and relative not in seen_files:
                seen_files.add(relative)
                matches.append(relative)
        return matches[offset : offset + head_limit]
