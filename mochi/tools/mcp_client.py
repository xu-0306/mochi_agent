"""mcp_call 工具 — 最小可用 MCP 代理呼叫。"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from typing import Any, Protocol

from mochi.tools.base import BaseTool, ToolResult

McpCaller = Callable[[str, str, dict[str, Any]], Any | Awaitable[Any]]


class SupportsMcpAdapter(Protocol):
    """MCP 呼叫適配器介面。"""

    async def call(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        """呼叫 MCP server 的工具。"""
        ...


class MCPCallTool(BaseTool):
    """透過注入 callable/adapter 代理 MCP 呼叫的最小工具。"""

    def __init__(
        self,
        *,
        caller: McpCaller | None = None,
        adapter: SupportsMcpAdapter | None = None,
    ) -> None:
        """初始化 mcp_call 工具。

        Args:
            caller: 可注入呼叫函式（可同步或非同步）。
            adapter: 可注入 adapter，需提供 async call()。
        """
        self._caller = caller
        self._adapter = adapter

    @property
    def name(self) -> str:
        """工具名稱。"""
        return "mcp_call"

    @property
    def description(self) -> str:
        """工具用途描述。"""
        return "Call an MCP server tool. Requires an injected caller or adapter to execute."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema 格式參數。"""
        return {
            "type": "object",
            "properties": {
                "server": {"type": "string", "description": "MCP server name or identifier."},
                "tool": {"type": "string", "description": "MCP tool name."},
                "arguments": {
                    "type": "object",
                    "default": {},
                    "description": "Tool arguments object.",
                },
            },
            "required": ["server", "tool"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        server: str,
        tool: str,
        arguments: dict[str, Any] | None = None,
    ) -> ToolResult:
        """執行 MCP 呼叫。"""
        if not server.strip():
            return ToolResult(error="`server` must not be empty.")
        if not tool.strip():
            return ToolResult(error="`tool` must not be empty.")
        if arguments is not None and not isinstance(arguments, dict):
            return ToolResult(error="`arguments` must be an object.")

        payload = arguments or {}

        try:
            if self._caller is not None:
                result = self._caller(server, tool, payload)
            elif self._adapter is not None:
                result = self._adapter.call(server, tool, payload)
            else:
                return ToolResult(
                    error="MCP caller is not configured. Inject `caller` or `adapter` first.",
                    metadata={"server": server, "tool": tool},
                )

            if inspect.isawaitable(result):
                result = await result
        except Exception as exc:
            return ToolResult(
                error=f"MCP call failed: {exc}",
                metadata={"server": server, "tool": tool},
            )

        return ToolResult(output=result, metadata={"server": server, "tool": tool})
