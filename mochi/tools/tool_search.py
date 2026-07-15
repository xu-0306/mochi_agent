"""Search registered tool metadata locally."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from mochi.tools.base import BaseTool, ToolResult

ToolCatalogProvider = Callable[[], list[BaseTool]]
CallableToolNameProvider = Callable[[], set[str]]


class ToolSearchTool(BaseTool):
    """Search over locally registered tool metadata."""

    def __init__(
        self,
        *,
        catalog_provider: ToolCatalogProvider,
        callable_name_provider: CallableToolNameProvider | None = None,
        default_top_k: int = 5,
        max_top_k: int = 50,
    ) -> None:
        self._catalog_provider = catalog_provider
        self._callable_name_provider = callable_name_provider
        self._default_top_k = max(1, int(default_top_k))
        self._max_top_k = max(self._default_top_k, int(max_top_k))

    @property
    def name(self) -> str:
        return "tool_search"

    @property
    def description(self) -> str:
        return "Search locally available tools by name, description, hint, and parameter metadata."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Tool search query."},
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self._max_top_k,
                    "description": "Maximum number of tool matches to return.",
                },
            },
            "required": ["query"],
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
        return "Use this when the tool you need is not already visible, especially in large local tool sets."

    async def execute(self, *, query: str, top_k: int | None = None) -> ToolResult:
        if not query.strip():
            return ToolResult(error="`query` must not be empty.")

        effective_top_k = self._default_top_k if top_k is None else int(top_k)
        if effective_top_k <= 0:
            return ToolResult(error="`top_k` must be greater than 0.")
        effective_top_k = min(effective_top_k, self._max_top_k)

        matches = self._search_catalog(query, effective_top_k)
        return ToolResult(
            output=matches,
            metadata={"count": len(matches), "top_k": effective_top_k, "query": query},
        )

    def _search_catalog(self, query: str, top_k: int) -> list[dict[str, Any]]:
        tokens = [token for token in query.lower().split() if token]
        ranked: list[tuple[int, str, dict[str, Any]]] = []
        catalog = self._catalog_provider()
        callable_names = self._callable_names_for_catalog(catalog)

        for tool in catalog:
            payload = self._tool_payload(tool, callable_names=callable_names)
            capabilities_text = self._capabilities_text(tool.tool_capabilities).lower()
            haystacks = [
                tool.name.lower(),
                tool.description.lower(),
                (tool.search_hint or "").lower(),
                self._schema_text(tool.parameters_schema).lower(),
                capabilities_text,
            ]
            score = 0
            for token in tokens:
                if token in tool.name.lower():
                    score += 5
                for haystack in haystacks[1:]:
                    score += haystack.count(token)
            ranked.append((score, tool.name, payload))

        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [payload for _, _, payload in ranked[:top_k]]

    def scoped_to_catalog(
        self,
        catalog_provider: ToolCatalogProvider,
        callable_name_provider: CallableToolNameProvider | None = None,
    ) -> ToolSearchTool:
        return ToolSearchTool(
            catalog_provider=catalog_provider,
            callable_name_provider=callable_name_provider,
            default_top_k=self._default_top_k,
            max_top_k=self._max_top_k,
        )

    def _callable_names_for_catalog(self, catalog: list[BaseTool]) -> set[str]:
        if self._callable_name_provider is not None:
            return set(self._callable_name_provider())
        return {tool.name for tool in catalog}

    @staticmethod
    def _schema_text(schema: dict[str, Any]) -> str:
        properties = schema.get("properties", {})
        if not isinstance(properties, dict):
            return ""
        parts: list[str] = []
        for key, value in properties.items():
            parts.append(str(key))
            if isinstance(value, dict):
                description = value.get("description")
                if description is not None:
                    parts.append(str(description))
        return " ".join(parts)

    @staticmethod
    def _capabilities_text(capabilities: dict[str, Any]) -> str:
        parts: list[str] = []
        for value in capabilities.values():
            if isinstance(value, list):
                parts.extend(str(item) for item in value)
                continue
            if isinstance(value, (str, bool)):
                parts.append(str(value))
        return " ".join(parts)

    @staticmethod
    def _tool_payload(tool: BaseTool, *, callable_names: set[str]) -> dict[str, Any]:
        callable_this_turn = tool.name in callable_names
        payload = {
            "name": tool.name,
            "description": tool.description,
            "search_hint": tool.search_hint,
            "read_only": tool.is_read_only,
            "destructive": tool.is_destructive,
            "open_world": tool.is_open_world,
            "capabilities": tool.tool_capabilities,
            "callable_this_turn": callable_this_turn,
            "activation_required": not callable_this_turn,
            "activation_reason": (
                None
                if callable_this_turn
                else "Tool is discoverable but not exposed as callable in this turn."
            ),
        }
        activation_request = ToolSearchTool._activation_request_for_tool(
            tool,
            callable_this_turn=callable_this_turn,
        )
        if activation_request is not None:
            payload["activation_request"] = activation_request
        return payload

    @staticmethod
    def _activation_request_for_tool(
        tool: BaseTool,
        *,
        callable_this_turn: bool,
    ) -> dict[str, str] | None:
        if callable_this_turn or tool.is_read_only:
            return None

        if tool.name in {"file_write", "file_edit", "apply_patch"}:
            return {
                "tool_name": tool.name,
                "required_intent": "workspace_write",
                "policy_check": "required",
            }
        return None
