"""Lightweight workspace repo mapping and symbol reads."""

from __future__ import annotations

import ast
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

from mochi.config import defaults
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.utils.security import (
    check_file_tool_path,
    normalize_workspace_dir,
    resolve_path_in_workspace,
)

_IGNORED_DIRS = {
    ".git",
    ".mochi",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
}
_PYTHON_EXTENSIONS = {".py"}
_JS_TS_EXTENSIONS = {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}
_TEXT_KIND_BY_SUFFIX = {
    ".md": ("markdown", "document"),
    ".txt": ("text", "document"),
    ".rst": ("rst", "document"),
    ".json": ("json", "data"),
    ".yaml": ("yaml", "config"),
    ".yml": ("yaml", "config"),
    ".toml": ("toml", "config"),
    ".ini": ("ini", "config"),
    ".cfg": ("config", "config"),
    ".ipynb": ("notebook", "notebook"),
}
_JS_DECLARATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "class",
        re.compile(
            r"^(?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+"
            r"(?P<name>[A-Za-z_$][\w$]*)\b"
        ),
    ),
    (
        "function",
        re.compile(
            r"^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+"
            r"(?P<name>[A-Za-z_$][\w$]*)\b"
        ),
    ),
    (
        "function",
        re.compile(
            r"^(?:export\s+)?(?:const|let|var)\s+(?P<name>[A-Za-z_$][\w$]*)\s*="
            r"\s*(?:async\s+)?(?:function\b|\([^)]*\)\s*=>|[A-Za-z_$][\w$]*\s*=>)"
        ),
    ),
    (
        "interface",
        re.compile(r"^(?:export\s+)?interface\s+(?P<name>[A-Za-z_$][\w$]*)\b"),
    ),
    (
        "type",
        re.compile(r"^(?:export\s+)?type\s+(?P<name>[A-Za-z_$][\w$]*)\b"),
    ),
    (
        "enum",
        re.compile(r"^(?:export\s+)?enum\s+(?P<name>[A-Za-z_$][\w$]*)\b"),
    ),
)


@dataclass(frozen=True)
class _SymbolInfo:
    name: str
    kind: str
    start_line: int
    end_line: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
        }


def _resolve_workspace_root(
    context: ToolExecutionContext | None,
    fallback_workspace_dir: Path,
) -> Path:
    if context is not None:
        for candidate in (
            context.task_sandbox_dir,
            context.project_workspace,
            context.workspace_dir,
        ):
            if candidate:
                return normalize_workspace_dir(candidate)
    return fallback_workspace_dir


def _detect_language_and_kind(path: Path) -> tuple[str | None, str | None]:
    suffix = path.suffix.lower()
    if suffix in _PYTHON_EXTENSIONS:
        return "python", "source"
    if suffix == ".js":
        return "javascript", "source"
    if suffix == ".jsx":
        return "jsx", "source"
    if suffix == ".ts":
        return "typescript", "source"
    if suffix == ".tsx":
        return "tsx", "source"
    if suffix in {".mjs", ".cjs"}:
        return "javascript", "source"
    return _TEXT_KIND_BY_SUFFIX.get(suffix, (None, None))


def _extract_symbols(
    path: Path,
    text: str,
    *,
    max_symbols: int,
) -> list[_SymbolInfo]:
    suffix = path.suffix.lower()
    if suffix in _PYTHON_EXTENSIONS:
        return _extract_python_symbols(text, max_symbols=max_symbols)
    if suffix in _JS_TS_EXTENSIONS:
        return _extract_js_ts_symbols(text, max_symbols=max_symbols)
    return []


def _extract_python_symbols(text: str, *, max_symbols: int) -> list[_SymbolInfo]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []

    symbols: list[_SymbolInfo] = []
    for node in tree.body:
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start_line = _python_symbol_start_line(node)
        end_line = max(start_line, getattr(node, "end_lineno", start_line))
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        symbols.append(
            _SymbolInfo(
                name=node.name,
                kind=kind,
                start_line=start_line,
                end_line=end_line,
            )
        )
        if len(symbols) >= max_symbols:
            break
    return symbols


def _python_symbol_start_line(
    node: ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    decorator_lines = [getattr(decorator, "lineno", node.lineno) for decorator in node.decorator_list]
    if decorator_lines:
        return min(min(decorator_lines), node.lineno)
    return node.lineno


def _extract_js_ts_symbols(text: str, *, max_symbols: int) -> list[_SymbolInfo]:
    lines = text.splitlines()
    starts: list[tuple[str, str, int]] = []
    for line_no, line in enumerate(lines, start=1):
        if not line or line[0].isspace():
            continue
        stripped = line.strip()
        for kind, pattern in _JS_DECLARATION_PATTERNS:
            match = pattern.match(stripped)
            if match is None:
                continue
            starts.append((match.group("name"), kind, line_no))
            break
        if len(starts) >= max_symbols:
            break

    symbols: list[_SymbolInfo] = []
    for index, (name, kind, start_line) in enumerate(starts):
        symbols.append(
            _SymbolInfo(
                name=name,
                kind=kind,
                start_line=start_line,
                end_line=_find_js_ts_symbol_end(lines, start_index=start_line - 1),
            )
        )
    return symbols


def _find_js_ts_symbol_end(lines: list[str], *, start_index: int) -> int:
    brace_depth = 0
    paren_depth = 0
    bracket_depth = 0
    saw_structure = False
    last_non_empty = start_index + 1

    for index in range(start_index, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if stripped:
            last_non_empty = index + 1

        open_braces = line.count("{")
        close_braces = line.count("}")
        open_parens = line.count("(")
        close_parens = line.count(")")
        open_brackets = line.count("[")
        close_brackets = line.count("]")

        if open_braces or open_parens or open_brackets:
            saw_structure = True

        brace_depth += open_braces - close_braces
        paren_depth += open_parens - close_parens
        bracket_depth += open_brackets - close_brackets

        if saw_structure and brace_depth <= 0 and paren_depth <= 0 and bracket_depth <= 0:
            if stripped.endswith(("}", "};", ";")):
                return last_non_empty

        if (
            index > start_index
            and stripped
            and not line[:1].isspace()
            and brace_depth <= 0
            and paren_depth <= 0
            and bracket_depth <= 0
        ):
            return max(start_index + 1, index)

    return last_non_empty


def _format_numbered_block(lines: list[str], *, start_line: int, end_line: int) -> str:
    selected = lines[start_line - 1 : end_line]
    return "\n".join(
        f"{line_no}: {line}"
        for line_no, line in enumerate(selected, start=start_line)
    )


def _relative_workspace_path(path: Path, workspace_root: Path) -> Path | None:
    try:
        return path.resolve(strict=False).relative_to(workspace_root)
    except ValueError:
        return None


def _ignored_path_segment(path: Path, workspace_root: Path) -> str | None:
    relative_path = _relative_workspace_path(path, workspace_root)
    if relative_path is None:
        return None
    for segment in relative_path.parts:
        if segment in _IGNORED_DIRS:
            return segment
    return None


class RepoMapTool(BaseTool):
    """Summarize workspace files and top-level symbols for navigation."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        default_max_files: int = 50,
        max_files: int = 200,
        default_max_symbols_per_file: int = 8,
        max_symbols_per_file: int = 20,
        max_scan_files: int = 500,
        max_parse_bytes: int = 200_000,
    ) -> None:
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._default_max_files = max(1, int(default_max_files))
        self._max_files = max(self._default_max_files, int(max_files))
        self._default_max_symbols_per_file = max(1, int(default_max_symbols_per_file))
        self._max_symbols_per_file = max(self._default_max_symbols_per_file, int(max_symbols_per_file))
        self._max_scan_files = max(1, int(max_scan_files))
        self._max_parse_bytes = max(1, int(max_parse_bytes))

    @property
    def name(self) -> str:
        return "repo_map"

    @property
    def description(self) -> str:
        return (
            "Summarize the workspace as a bounded repo map with relative file paths, "
            "detected file kinds, and top-level symbols when they can be extracted."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Optional directory or file path inside the workspace to narrow the repo map.",
                },
                "max_files": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self._max_files,
                    "description": "Maximum number of file entries to return.",
                },
                "max_symbols_per_file": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self._max_symbols_per_file,
                    "description": "Maximum number of top-level symbols to return per file.",
                },
            },
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
        return "Use this to orient in a workspace before targeted file reads or symbol inspection."

    @property
    def tool_capabilities(self) -> dict[str, Any]:
        return {
            "domains": ["workspace"],
            "retrieval_modes": ["repo_map", "symbol_index"],
            "preference_tags": ["repo_navigation", "code_discovery"],
            "read_only": True,
            "destructive": False,
            "open_world": False,
        }

    async def execute(
        self,
        *,
        path: str | None = None,
        max_files: int | None = None,
        max_symbols_per_file: int | None = None,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        workspace_root = _resolve_workspace_root(context, self._workspace_dir)

        try:
            if path is None or path.strip() in {"", "."}:
                search_root = workspace_root
            else:
                checked_root, security_decision = check_file_tool_path(
                    path,
                    workspace_dir=workspace_root,
                    scope="workspace",
                    access="read",
                )
                if security_decision is not None or checked_root is None:
                    return ToolResult(
                        error=security_decision.reason if security_decision is not None else "Path denied.",
                        metadata=security_decision.to_metadata() if security_decision is not None else {},
                    )
                search_root = resolve_path_in_workspace(checked_root, workspace_root)
        except ValueError as exc:
            return ToolResult(error=str(exc))

        ignored_segment = _ignored_path_segment(search_root, workspace_root)
        if ignored_segment is not None:
            return ToolResult(
                error=(
                    f"Path '{path}' is inside ignored directory '{ignored_segment}' "
                    "and is not available for repo navigation."
                )
            )

        effective_max_files = self._default_max_files if max_files is None else int(max_files)
        if effective_max_files <= 0:
            return ToolResult(error="`max_files` must be greater than 0.")
        effective_max_files = min(effective_max_files, self._max_files)

        effective_max_symbols = (
            self._default_max_symbols_per_file
            if max_symbols_per_file is None
            else int(max_symbols_per_file)
        )
        if effective_max_symbols <= 0:
            return ToolResult(error="`max_symbols_per_file` must be greater than 0.")
        effective_max_symbols = min(effective_max_symbols, self._max_symbols_per_file)

        payload = await asyncio.to_thread(
            self._build_repo_map,
            search_root,
            workspace_root,
            effective_max_files,
            effective_max_symbols,
        )
        return ToolResult(
            output=payload,
            metadata={
                "workspace_root": str(workspace_root),
                "path": str(search_root),
                "count": len(payload["files"]),
                "max_files": effective_max_files,
                "max_symbols_per_file": effective_max_symbols,
                "truncated": payload["truncated"],
                "scanned_files": payload["scanned_files"],
            },
        )

    def _build_repo_map(
        self,
        search_root: Path,
        workspace_root: Path,
        max_files: int,
        max_symbols_per_file: int,
    ) -> dict[str, Any]:
        file_entries: list[dict[str, Any]] = []
        scanned_files = 0
        truncated = False

        for candidate in self._iter_files(search_root):
            if scanned_files >= self._max_scan_files or len(file_entries) >= max_files:
                truncated = True
                break

            scanned_files += 1
            try:
                relative_path = candidate.resolve(strict=False).relative_to(workspace_root).as_posix()
            except ValueError:
                continue
            language, kind = _detect_language_and_kind(candidate)
            entry: dict[str, Any] = {"path": relative_path}
            if language is not None:
                entry["language"] = language
            if kind is not None:
                entry["kind"] = kind

            try:
                file_size = candidate.stat().st_size
            except OSError:
                file_entries.append(entry)
                continue

            if file_size > self._max_parse_bytes:
                file_entries.append(entry)
                continue

            if candidate.suffix.lower() in _PYTHON_EXTENSIONS | _JS_TS_EXTENSIONS:
                try:
                    text = candidate.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    file_entries.append(entry)
                    continue
                symbols = _extract_symbols(
                    candidate,
                    text,
                    max_symbols=max_symbols_per_file,
                )
                if symbols:
                    entry["symbols"] = [symbol.to_payload() for symbol in symbols]

            file_entries.append(entry)

        return {
            "root": str(search_root),
            "files": file_entries,
            "scanned_files": scanned_files,
            "truncated": truncated,
        }

    @staticmethod
    def _iter_files(search_root: Path):
        if search_root.is_file():
            yield search_root
            return

        for root, dirs, files in os.walk(search_root, topdown=True):
            dirs[:] = sorted(directory for directory in dirs if directory not in _IGNORED_DIRS)
            for filename in sorted(files):
                yield Path(root) / filename


class ReadSymbolTool(BaseTool):
    """Read a bounded top-level symbol block from a workspace file."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        default_max_lines: int = 200,
        max_lines: int = 400,
        max_parse_bytes: int = 300_000,
    ) -> None:
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._default_max_lines = max(1, int(default_max_lines))
        self._max_lines = max(self._default_max_lines, int(max_lines))
        self._max_parse_bytes = max(1, int(max_parse_bytes))

    @property
    def name(self) -> str:
        return "read_symbol"

    @property
    def description(self) -> str:
        return (
            "Read a bounded top-level symbol block from a workspace file with line numbers. "
            "Use when you know the file and symbol name but do not want the whole file."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "File path inside the workspace.",
                },
                "symbol": {
                    "type": "string",
                    "description": "Exact top-level symbol name to read.",
                },
                "max_lines": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self._max_lines,
                    "description": "Maximum number of lines to return from the symbol block.",
                },
            },
            "required": ["path", "symbol"],
            "additionalProperties": False,
        }

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    @property
    def allow_plain_text_result_for_model(self) -> bool:
        return True

    @property
    def search_hint(self) -> str | None:
        return "Use this after locating the right file when you need one class or function, not a whole file."

    @property
    def tool_capabilities(self) -> dict[str, Any]:
        return {
            "domains": ["workspace"],
            "retrieval_modes": ["symbol_lookup", "targeted_read"],
            "preference_tags": ["repo_navigation", "definition_lookup"],
            "read_only": True,
            "destructive": False,
            "open_world": False,
        }

    async def execute(
        self,
        *,
        path: str,
        symbol: str,
        max_lines: int | None = None,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not path.strip():
            return ToolResult(error="`path` must not be empty.")
        if not symbol.strip():
            return ToolResult(error="`symbol` must not be empty.")

        workspace_root = _resolve_workspace_root(context, self._workspace_dir)
        try:
            checked_target, security_decision = check_file_tool_path(
                path,
                workspace_dir=workspace_root,
                scope="workspace",
                access="read",
            )
            if security_decision is not None or checked_target is None:
                return ToolResult(
                    error=security_decision.reason if security_decision is not None else "Path denied.",
                    metadata=security_decision.to_metadata() if security_decision is not None else {},
                )
            target = resolve_path_in_workspace(checked_target, workspace_root)
        except ValueError as exc:
            return ToolResult(error=str(exc))

        ignored_segment = _ignored_path_segment(target, workspace_root)
        if ignored_segment is not None:
            return ToolResult(
                error=(
                    f"Path '{path}' is inside ignored directory '{ignored_segment}' "
                    "and is not available for symbol reads."
                )
            )

        if not target.exists():
            return ToolResult(error=f"File not found: {target}")
        if not target.is_file():
            return ToolResult(error=f"Path is not a file: {target}")

        effective_max_lines = self._default_max_lines if max_lines is None else int(max_lines)
        if effective_max_lines <= 0:
            return ToolResult(error="`max_lines` must be greater than 0.")
        effective_max_lines = min(effective_max_lines, self._max_lines)

        try:
            file_size = await asyncio.to_thread(lambda: target.stat().st_size)
        except OSError as exc:
            return ToolResult(error=f"Failed to inspect file: {exc}")
        if file_size > self._max_parse_bytes:
            return ToolResult(
                error=(
                    f"File is too large for symbol extraction ({file_size} bytes exceeds "
                    f"{self._max_parse_bytes} bytes). Use file_read for a bounded chunk instead."
                )
            )

        try:
            text = await asyncio.to_thread(target.read_text, encoding="utf-8")
        except UnicodeDecodeError:
            return ToolResult(error=f"File is not valid utf-8 text: {target}")
        except OSError as exc:
            return ToolResult(error=f"Failed to read file: {exc}")

        symbols = _extract_symbols(target, text, max_symbols=10_000)
        if not symbols:
            return ToolResult(
                error=(
                    f"No top-level symbols were extracted from "
                    f"{target.resolve(strict=False).relative_to(workspace_root).as_posix()}."
                )
            )

        match = next((item for item in symbols if item.name == symbol), None)
        if match is None:
            available = ", ".join(item.name for item in symbols[:10])
            details = f" Available symbols: {available}." if available else ""
            return ToolResult(error=f"Symbol '{symbol}' was not found in {path}.{details}")

        lines = text.splitlines()
        bounded_end_line = min(match.end_line, match.start_line + effective_max_lines - 1, len(lines))
        block = _format_numbered_block(
            lines,
            start_line=match.start_line,
            end_line=bounded_end_line,
        )
        language, kind = _detect_language_and_kind(target)
        relative_path = target.resolve(strict=False).relative_to(workspace_root).as_posix()
        return ToolResult(
            output=block,
            metadata={
                "path": relative_path,
                "symbol": match.name,
                "symbol_kind": match.kind,
                "language": language,
                "kind": kind,
                "start_line": match.start_line,
                "end_line": bounded_end_line,
                "symbol_end_line": match.end_line,
                "truncated": bounded_end_line < match.end_line,
                "max_lines": effective_max_lines,
            },
        )
