"""execute_code_v2 tool with subprocess execution and local tool helpers."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mochi.config import defaults
from mochi.security import require_approval_decision
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.utils.security import normalize_workspace_dir, resolve_path_in_workspace

ExecuteCodeV2Runner = Callable[
    [str, Path, int, str, Path, list[str]],
    Awaitable[dict[str, Any]],
]

_SUPPORTED_TOOL_HELPERS: tuple[str, ...] = (
    "file_read",
    "glob_search",
    "grep_search",
    "csv_read",
    "pdf_read",
    "web_search",
    "web_fetch",
)

_BOOTSTRAP = r"""
import asyncio
import contextlib
import io
import json
import sys
import traceback

repo_root = sys.argv[1]
workspace_dir = sys.argv[2]
allowed_tools = json.loads(sys.argv[3])
user_code = sys.argv[4]

sys.path.insert(0, repo_root)

from mochi.tools.csv_read import CsvReadTool
from mochi.tools.file_ops import FileReadTool
from mochi.tools.glob_search import GlobSearchTool
from mochi.tools.grep_search import GrepSearchTool
from mochi.tools.pdf_read import PdfReadTool
from mochi.tools.web_fetch import WebFetchTool
from mochi.tools.web_search import WebSearchTool

tool_cache = {}
tool_calls = []

def _build_tool(name):
    if name == "file_read":
        return FileReadTool(workspace_dir=workspace_dir)
    if name == "glob_search":
        return GlobSearchTool(workspace_dir=workspace_dir)
    if name == "grep_search":
        return GrepSearchTool(workspace_dir=workspace_dir)
    if name == "csv_read":
        return CsvReadTool(workspace_dir=workspace_dir)
    if name == "pdf_read":
        return PdfReadTool(workspace_dir=workspace_dir)
    if name == "web_search":
        return WebSearchTool()
    if name == "web_fetch":
        return WebFetchTool()
    raise RuntimeError(f"Unsupported helper tool: {name}")

def _call_tool(name, **kwargs):
    if name not in allowed_tools:
        raise RuntimeError(f"Tool not allowed: {name}")
    tool = tool_cache.get(name)
    if tool is None:
        tool = _build_tool(name)
        tool_cache[name] = tool
    result = asyncio.run(tool.execute(**kwargs))
    payload = {
        "ok": result.error is None,
        "output": result.output,
        "error": result.error,
        "metadata": result.metadata,
        "retryable": result.retryable,
        "suggestion": result.suggestion,
    }
    tool_calls.append(
        {
            "tool_name": name,
            "ok": payload["ok"],
            "error": payload["error"],
            "metadata": payload["metadata"],
        }
    )
    if result.error is not None:
        raise RuntimeError(result.error)
    return payload

def call_tool(name, **kwargs):
    return _call_tool(name, **kwargs)

def _wrapper(name):
    return lambda **kwargs: _call_tool(name, **kwargs)["output"]

globals_dict = {
    "__name__": "__main__",
    "call_tool": call_tool,
    "json": json,
    "result": None,
}
for helper_name in allowed_tools:
    globals_dict[helper_name] = _wrapper(helper_name)

stdout_buffer = io.StringIO()
error = None
traceback_text = ""
with contextlib.redirect_stdout(stdout_buffer):
    try:
        exec(user_code, globals_dict, globals_dict)
    except Exception as exc:
        error = str(exc)
        traceback_text = traceback.format_exc()

payload = {
    "stdout": stdout_buffer.getvalue(),
    "result": globals_dict.get("result"),
    "tool_calls": tool_calls,
    "error": error,
    "traceback": traceback_text,
}
sys.stdout.write(json.dumps(payload, ensure_ascii=False, default=str))
"""


class ExecuteCodeV2Tool(BaseTool):
    """Run Python code with access to selected Mochi tool helpers."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        require_approval: bool = True,
        default_timeout_sec: int = 20,
        python_executable: str | None = None,
        runner: ExecuteCodeV2Runner | None = None,
    ) -> None:
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._require_approval = require_approval
        self._default_timeout_sec = default_timeout_sec
        self._python_executable = python_executable or sys.executable
        self._runner = runner or self._default_runner

    @property
    def name(self) -> str:
        return "execute_code_v2"

    @property
    def description(self) -> str:
        return (
            "Run Python code in a controlled subprocess with helper access to selected Mochi tools. "
            "Use this for multi-step local workflows that benefit from programmatic tool orchestration."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code string to run."},
                "cwd": {
                    "type": "string",
                    "description": "Working directory. Must be inside the workspace.",
                },
                "timeout_sec": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                    "default": self._default_timeout_sec,
                    "description": "Execution timeout in seconds.",
                },
                "approved": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether user approval has been granted. Required when approval is enabled.",
                },
                "allowed_tools": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(_SUPPORTED_TOOL_HELPERS)},
                    "description": "Subset of helper tools available to the Python code.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        }

    @property
    def requires_approval(self) -> bool:
        return self._require_approval

    async def execute(
        self,
        *,
        code: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        approved: bool = False,
        allowed_tools: list[str] | None = None,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not code.strip():
            return ToolResult(error="`code` must not be empty.")

        if self._require_approval and not approved:
            decision = require_approval_decision(
                reason="Code execution requires explicit approval.",
                approval_kind="other",
                approval_scope="dangerous_command",
                replay_safe=True,
                policy_source="execute_code_policy",
            )
            return ToolResult(
                error="Code execution requires approval.",
                metadata=decision.to_metadata(),
            )

        workspace_root = self._resolve_workspace_root(context)
        try:
            working_dir = (
                resolve_path_in_workspace(cwd, workspace_root)
                if cwd is not None
                else workspace_root
            )
        except ValueError as exc:
            return ToolResult(error=str(exc))

        if not working_dir.exists() or not working_dir.is_dir():
            return ToolResult(error=f"Working directory does not exist: {working_dir}")

        effective_timeout = timeout_sec if timeout_sec is not None else self._default_timeout_sec
        if effective_timeout <= 0:
            return ToolResult(error="`timeout_sec` must be greater than 0.")

        effective_allowed_tools = list(_SUPPORTED_TOOL_HELPERS) if allowed_tools is None else list(allowed_tools)
        invalid_tools = [name for name in effective_allowed_tools if name not in _SUPPORTED_TOOL_HELPERS]
        if invalid_tools:
            return ToolResult(
                error="`allowed_tools` contains unsupported tools.",
                metadata={"invalid_tools": invalid_tools, "supported_tools": list(_SUPPORTED_TOOL_HELPERS)},
            )

        try:
            payload = await self._runner(
                code,
                working_dir,
                effective_timeout,
                self._python_executable,
                workspace_root,
                effective_allowed_tools,
            )
        except Exception as exc:  # pragma: no cover
            return ToolResult(
                error=f"Code execution failed: {exc}",
                metadata={"cwd": str(working_dir)},
            )

        output = {
            "stdout": payload.get("stdout", ""),
            "result": payload.get("result"),
            "tool_calls": payload.get("tool_calls", []),
        }
        metadata = {
            "cwd": str(working_dir),
            "allowed_tools": effective_allowed_tools,
            "tool_call_count": len(output["tool_calls"]) if isinstance(output["tool_calls"], list) else 0,
        }
        traceback_text = payload.get("traceback")
        if isinstance(traceback_text, str) and traceback_text:
            metadata["traceback"] = traceback_text

        error = payload.get("error")
        if isinstance(error, str) and error:
            return ToolResult(error=error, output=output, metadata=metadata)

        return ToolResult(output=output, metadata=metadata)

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
    async def _default_runner(
        code: str,
        cwd: Path,
        timeout_sec: int,
        python_executable: str,
        workspace_dir: Path,
        allowed_tools: list[str],
    ) -> dict[str, Any]:
        try:
            process = await asyncio.create_subprocess_exec(
                python_executable,
                "-c",
                _BOOTSTRAP,
                str(Path(__file__).resolve().parents[2]),
                str(workspace_dir),
                json.dumps(allowed_tools),
                code,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_sec,
                )
            except TimeoutError:
                process.kill()
                await process.communicate()
                return {
                    "stdout": "",
                    "result": None,
                    "tool_calls": [],
                    "error": f"Execution timed out after {timeout_sec} seconds.",
                    "traceback": "",
                }

            if process.returncode not in {0, None}:
                stderr = stderr_bytes.decode("utf-8", errors="replace")
                return {
                    "stdout": "",
                    "result": None,
                    "tool_calls": [],
                    "error": stderr or f"Process exited with non-zero status: {process.returncode}",
                    "traceback": "",
                }

            stdout = stdout_bytes.decode("utf-8", errors="replace")
            return json.loads(stdout or "{}")
        except Exception:
            return await asyncio.to_thread(
                ExecuteCodeV2Tool._run_sync_fallback,
                code,
                cwd,
                timeout_sec,
                python_executable,
                workspace_dir,
                allowed_tools,
            )

    @staticmethod
    def _run_sync_fallback(
        code: str,
        cwd: Path,
        timeout_sec: int,
        python_executable: str,
        workspace_dir: Path,
        allowed_tools: list[str],
    ) -> dict[str, Any]:
        try:
            completed = subprocess.run(
                [
                    python_executable,
                    "-c",
                    _BOOTSTRAP,
                    str(Path(__file__).resolve().parents[2]),
                    str(workspace_dir),
                    json.dumps(allowed_tools),
                    code,
                ],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                check=False,
            )
            if completed.returncode != 0:
                return {
                    "stdout": "",
                    "result": None,
                    "tool_calls": [],
                    "error": completed.stderr or f"Process exited with non-zero status: {completed.returncode}",
                    "traceback": "",
                }
            return json.loads(completed.stdout or "{}")
        except subprocess.TimeoutExpired:
            return {
                "stdout": "",
                "result": None,
                "tool_calls": [],
                "error": f"Execution timed out after {timeout_sec} seconds.",
                "traceback": "",
            }
