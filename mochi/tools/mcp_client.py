"""MCP runtime helpers and fallback tool wrappers."""

from __future__ import annotations

import inspect
import json
import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult

McpCaller = Callable[[str, str, dict[str, Any]], Any | Awaitable[Any]]


class SupportsMcpAdapter(Protocol):
    """Compatibility protocol for the legacy adapter path."""

    async def call(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        ...


class McpCatalogRefreshError(RuntimeError):
    """A server catalog could not be refreshed safely."""

    def __init__(self, message: str, *, status: str = "refresh_failed") -> None:
        super().__init__(message)
        self.status = str(status or "").strip() or "refresh_failed"


class McpCatalogTimeoutError(McpCatalogRefreshError):
    """Refresh timed out before a safe catalog snapshot was available."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status="timeout")


@dataclass
class McpToolDefinition:
    """Cached metadata for one MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass
class McpServerConfig:
    """Minimal first-wave MCP server configuration."""

    name: str
    transport: str = "stdio"
    command: str | None = None
    url: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    allow_tools: list[str] = field(default_factory=list)
    deny_tools: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class McpCatalogState:
    """Observed refresh state for one MCP server catalog."""

    server: str
    generation: int
    fingerprint: str | None
    status: str
    error: str | None = None


class McpRuntimeManager:
    """Product-layer MCP runtime manager."""

    def __init__(
        self,
        *,
        servers: dict[str, McpServerConfig] | list[McpServerConfig] | None = None,
        adapters: dict[str, Any] | None = None,
    ) -> None:
        if isinstance(servers, list):
            self._servers = {server.name: server for server in servers}
        else:
            self._servers = dict(servers or {})
        self._adapters = dict(adapters or {})
        self._tool_cache: dict[str, list[McpToolDefinition]] = {}
        self._resource_cache: dict[str, list[dict[str, Any]]] = {}
        self._catalog_generation: dict[str, int] = {}
        self._catalog_fingerprint: dict[str, str] = {}
        self._catalog_status: dict[str, str] = {}
        self._catalog_error: dict[str, str] = {}

    def get_server(self, name: str) -> McpServerConfig | None:
        return self._servers.get(name)

    def list_server_names(self, *, enabled_only: bool = True) -> list[str]:
        names: list[str] = []
        for name, server in self._servers.items():
            if enabled_only and not server.enabled:
                continue
            names.append(name)
        return names

    async def refresh_server(self, name: str) -> None:
        adapter = self._require_adapter(name)
        try:
            tools = await self._list_tools_from_adapter(adapter)
            resources = await self._list_resources_from_adapter(adapter)
        except TimeoutError as exc:
            self._catalog_status[name] = "timeout"
            self._catalog_error[name] = str(exc) or type(exc).__name__
            raise McpCatalogTimeoutError(
                f"MCP catalog refresh timed out for server '{name}': {exc}"
            ) from exc
        except McpCatalogRefreshError as exc:
            self._catalog_status[name] = exc.status
            self._catalog_error[name] = str(exc)
            raise
        except (TypeError, ValueError) as exc:
            self._catalog_status[name] = "malformed_catalog"
            self._catalog_error[name] = str(exc)
            raise McpCatalogRefreshError(
                f"MCP catalog refresh returned malformed data for server '{name}': {exc}",
                status="malformed_catalog",
            ) from exc
        except Exception as exc:
            self._catalog_status[name] = "refresh_failed"
            self._catalog_error[name] = str(exc)
            raise McpCatalogRefreshError(
                f"MCP catalog refresh failed for server '{name}': {exc}"
            ) from exc

        self._tool_cache[name] = tools
        self._resource_cache[name] = resources
        self._catalog_generation[name] = self._catalog_generation.get(name, 0) + 1
        self._catalog_fingerprint[name] = _catalog_fingerprint(tools)
        self._catalog_status[name] = "ready"
        self._catalog_error.pop(name, None)

    async def list_tools(self, server: str) -> list[McpToolDefinition]:
        if server not in self._tool_cache:
            await self.refresh_server(server)
        return list(self._tool_cache.get(server, []))

    async def list_resources(self, server: str) -> list[dict[str, Any]]:
        if server not in self._resource_cache:
            await self.refresh_server(server)
        return list(self._resource_cache.get(server, []))

    async def read_resource(self, server: str, uri: str) -> Any:
        adapter = self._require_adapter(server)
        read_resource = getattr(adapter, "read_resource", None)
        if not callable(read_resource):
            raise RuntimeError(f"MCP server '{server}' does not support resource reads.")
        result = read_resource(uri)
        if inspect.isawaitable(result):
            result = await result
        return result

    async def call_tool(self, server: str, tool: str, arguments: dict[str, Any]) -> Any:
        if not self._is_tool_allowed(server, tool):
            raise PermissionError(f"MCP tool '{server}.{tool}' is denied by policy.")
        adapter = self._require_adapter(server)
        call_tool = getattr(adapter, "call_tool", None)
        if callable(call_tool):
            result = call_tool(tool, arguments)
        else:
            call = getattr(adapter, "call", None)
            if not callable(call):
                raise RuntimeError(f"MCP server '{server}' does not support tool calls.")
            result = call(server, tool, arguments)
        if inspect.isawaitable(result):
            result = await result
        return result

    def materialize_tools(self) -> list[BaseTool]:
        tools: list[BaseTool] = []
        for server_name, definitions in self._tool_cache.items():
            for definition in definitions:
                if self._is_tool_allowed(server_name, definition.name):
                    tools.append(
                        McpDynamicTool(
                            runtime=self,
                            server=server_name,
                            definition=definition,
                        )
                    )
        return tools

    def get_catalog_generation(self, server: str) -> int:
        return int(self._catalog_generation.get(server, 0))

    def get_catalog_fingerprint(self, server: str) -> str | None:
        return self._catalog_fingerprint.get(server)

    def get_catalog_state(self, server: str) -> McpCatalogState:
        return McpCatalogState(
            server=server,
            generation=self.get_catalog_generation(server),
            fingerprint=self.get_catalog_fingerprint(server),
            status=self._catalog_status.get(server, "missing"),
            error=self._catalog_error.get(server),
        )

    def _require_adapter(self, name: str) -> Any:
        server = self._servers.get(name)
        if server is None:
            raise RuntimeError(f"MCP server '{name}' is not configured.")
        if not server.enabled:
            raise RuntimeError(f"MCP server '{name}' is disabled.")
        adapter = self._adapters.get(name)
        if adapter is None:
            raise RuntimeError(f"MCP server '{name}' is not connected.")
        return adapter

    def _is_tool_allowed(self, server: str, tool: str) -> bool:
        config = self._servers.get(server)
        if config is None or not config.enabled:
            return False
        if config.allow_tools and tool not in config.allow_tools:
            return False
        if tool in config.deny_tools:
            return False
        return True

    async def _list_tools_from_adapter(self, adapter: Any) -> list[McpToolDefinition]:
        list_tools = getattr(adapter, "list_tools", None)
        if not callable(list_tools):
            return []
        raw = list_tools()
        if inspect.isawaitable(raw):
            raw = await raw
        if raw is None:
            return []
        if not isinstance(raw, (list, tuple)):
            raise ValueError("MCP tools/list response must be a sequence")

        normalized: list[McpToolDefinition] = []
        seen_names: set[str] = set()
        for item in raw or []:
            if isinstance(item, McpToolDefinition):
                definition = item
            elif not isinstance(item, dict):
                raise ValueError("MCP tool catalog entries must be objects")
            else:
                input_schema = item.get("input_schema") or item.get("inputSchema") or {
                    "type": "object",
                    "properties": {},
                }
                if not isinstance(input_schema, dict):
                    raise ValueError("MCP tool input_schema must be an object")
                annotations = item.get("annotations") or {}
                if not isinstance(annotations, dict):
                    raise ValueError("MCP tool annotations must be an object")
                definition = McpToolDefinition(
                    name=str(item.get("name") or "").strip(),
                    description=str(item.get("description") or "").strip(),
                    input_schema=input_schema,
                    annotations=dict(annotations),
                )

            if not definition.name:
                raise ValueError("MCP tool catalog entries must include a non-empty name")
            if definition.name in seen_names:
                raise ValueError(f"duplicate MCP tool catalog entry: {definition.name}")
            seen_names.add(definition.name)
            if not isinstance(definition.input_schema, dict):
                raise ValueError("MCP tool input_schema must be an object")
            normalized.append(definition)
        return normalized

    async def _list_resources_from_adapter(self, adapter: Any) -> list[dict[str, Any]]:
        list_resources = getattr(adapter, "list_resources", None)
        if not callable(list_resources):
            return []
        raw = list_resources()
        if inspect.isawaitable(raw):
            raw = await raw
        if raw is None:
            return []
        if not isinstance(raw, (list, tuple)):
            raise ValueError("MCP resources/list response must be a sequence")
        resources: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError("MCP resource catalog entries must be objects")
            resources.append(dict(item))
        return resources


def _catalog_fingerprint(definitions: list[McpToolDefinition]) -> str:
    payload = [
        {
            "name": definition.name,
            "description": definition.description,
            "input_schema": definition.input_schema,
            "annotations": definition.annotations,
        }
        for definition in sorted(definitions, key=lambda item: item.name)
    ]
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class McpDynamicTool(BaseTool):
    """Expose one MCP tool as a first-class Mochi tool."""

    def __init__(
        self,
        *,
        runtime: McpRuntimeManager,
        server: str,
        definition: McpToolDefinition,
    ) -> None:
        self._runtime = runtime
        self._server = server
        self._definition = definition

    @property
    def name(self) -> str:
        return f"mcp__{self._server}__{self._definition.name}"

    @property
    def description(self) -> str:
        return self._definition.description or f"MCP tool {self._server}.{self._definition.name}"

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._definition.input_schema

    @property
    def is_read_only(self) -> bool:
        return bool(self._definition.annotations.get("readOnlyHint"))

    @property
    def is_destructive(self) -> bool:
        return bool(self._definition.annotations.get("destructiveHint"))

    @property
    def is_open_world(self) -> bool:
        return bool(self._definition.annotations.get("openWorldHint"))

    @property
    def search_hint(self) -> str | None:
        hint = self._definition.annotations.get("searchHint")
        return str(hint) if isinstance(hint, str) and hint.strip() else None

    async def execute(
        self,
        context: ToolExecutionContext | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        del context
        try:
            output = await self._runtime.call_tool(self._server, self._definition.name, kwargs)
        except PermissionError as exc:
            return ToolResult(error=str(exc), metadata={"server": self._server, "tool": self._definition.name})
        except Exception as exc:
            return ToolResult(
                error=f"MCP call failed: {exc}",
                metadata={"server": self._server, "tool": self._definition.name},
            )
        return ToolResult(
            output=output,
            metadata={"server": self._server, "tool": self._definition.name},
        )


class McpListResourcesTool(BaseTool):
    """List resources exposed by an MCP server."""

    def __init__(self, *, runtime: McpRuntimeManager) -> None:
        self._runtime = runtime

    @property
    def name(self) -> str:
        return "mcp_list_resources"

    @property
    def description(self) -> str:
        return "List resources exposed by a configured MCP server."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {"type": "string", "description": "Configured MCP server name."},
            },
            "required": ["server"],
            "additionalProperties": False,
        }

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_open_world(self) -> bool:
        return True

    async def execute(self, *, server: str, context: ToolExecutionContext | None = None) -> ToolResult:
        del context
        if not server.strip():
            return ToolResult(error="`server` must not be empty.")
        try:
            output = await self._runtime.list_resources(server)
        except Exception as exc:
            return ToolResult(error=f"MCP resource listing failed: {exc}", metadata={"server": server})
        return ToolResult(output=output, metadata={"server": server})


class McpReadResourceTool(BaseTool):
    """Read one MCP resource."""

    def __init__(self, *, runtime: McpRuntimeManager) -> None:
        self._runtime = runtime

    @property
    def name(self) -> str:
        return "mcp_read_resource"

    @property
    def description(self) -> str:
        return "Read a resource exposed by a configured MCP server."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "server": {"type": "string", "description": "Configured MCP server name."},
                "uri": {"type": "string", "description": "Resource URI."},
            },
            "required": ["server", "uri"],
            "additionalProperties": False,
        }

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_open_world(self) -> bool:
        return True

    async def execute(
        self,
        *,
        server: str,
        uri: str,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        del context
        if not server.strip():
            return ToolResult(error="`server` must not be empty.")
        if not uri.strip():
            return ToolResult(error="`uri` must not be empty.")
        try:
            output = await self._runtime.read_resource(server, uri)
        except Exception as exc:
            return ToolResult(
                error=f"MCP resource read failed: {exc}",
                metadata={"server": server, "uri": uri},
            )
        return ToolResult(output=output, metadata={"server": server, "uri": uri})


class MCPCallTool(BaseTool):
    """Low-level MCP call fallback."""

    def __init__(
        self,
        *,
        caller: McpCaller | None = None,
        adapter: SupportsMcpAdapter | None = None,
        runtime: McpRuntimeManager | None = None,
    ) -> None:
        self._caller = caller
        self._adapter = adapter
        self._runtime = runtime

    @property
    def name(self) -> str:
        return "mcp_call"

    @property
    def description(self) -> str:
        return (
            "Call a tool exposed by a configured MCP server through an injected caller, "
            "runtime, or adapter. Use as a low-level fallback when a first-class MCP tool "
            "is not available."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
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
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        del context
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
            elif self._runtime is not None:
                result = self._runtime.call_tool(server, tool, payload)
            elif self._adapter is not None:
                result = self._adapter.call(server, tool, payload)
            else:
                return ToolResult(
                    error="MCP caller is not configured. Inject `caller`, `runtime`, or `adapter` first.",
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
