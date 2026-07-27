"""Search registered tool metadata locally."""

from __future__ import annotations

import hashlib
import inspect
from collections.abc import Awaitable, Callable
from typing import Any

from mochi.tools.base import BaseTool, ToolResult
from mochi.tools.tool_catalog_index import (
    CatalogSearchCandidate,
    ToolCatalogIndex,
    ToolCatalogIndexError,
)

ToolCatalogProvider = Callable[[], list[BaseTool]]
CallableToolNameProvider = Callable[[], set[str]]
CatalogGenerationProvider = Callable[[], int | None]
ToolDiscoveryHook = Callable[[dict[str, Any]], Any | Awaitable[Any]]


class ToolSearchTool(BaseTool):
    """Search over locally registered tool metadata."""

    def __init__(
        self,
        *,
        catalog_provider: ToolCatalogProvider,
        callable_name_provider: CallableToolNameProvider | None = None,
        catalog_generation_provider: CatalogGenerationProvider | None = None,
        discovery_hook: ToolDiscoveryHook | None = None,
        default_top_k: int = 5,
        max_top_k: int = 10,
    ) -> None:
        self._catalog_provider = catalog_provider
        self._callable_name_provider = callable_name_provider
        self._catalog_generation_provider = catalog_generation_provider
        self._discovery_hook = discovery_hook
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

    async def execute(
        self,
        *,
        query: str,
        top_k: int | None = None,
        context: Any | None = None,
    ) -> ToolResult:
        if not query.strip():
            return ToolResult(error="`query` must not be empty.")

        effective_top_k = self._default_top_k if top_k is None else int(top_k)
        if effective_top_k <= 0:
            return ToolResult(error="`top_k` must be greater than 0.")
        effective_top_k = min(effective_top_k, self._max_top_k)

        matches, catalog_fingerprint, catalog_status = self._search_catalog(query, effective_top_k)
        catalog_generation = self._catalog_generation()
        metadata: dict[str, Any] = {
            "count": len(matches),
            "top_k": effective_top_k,
            "query": query,
            "catalog_status": catalog_status,
        }
        if catalog_fingerprint is not None:
            metadata["catalog_fingerprint"] = catalog_fingerprint
        if catalog_generation is not None:
            metadata["catalog_generation"] = catalog_generation
        await self._emit_discovery_hook(
            query=query,
            matches=matches,
            catalog_fingerprint=catalog_fingerprint,
            catalog_generation=catalog_generation,
            catalog_status=catalog_status,
            context=context,
        )
        return ToolResult(output=matches, metadata=metadata)

    def _search_catalog(
        self,
        query: str,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], str | None, str]:
        try:
            catalog = list(self._catalog_provider())
        except TimeoutError:
            return [], None, "timeout"
        except Exception as exc:
            status = getattr(exc, "status", None)
            if status in {"timeout", "malformed_catalog", "refresh_failed"}:
                return [], None, str(status)
            return [], None, "refresh_failed"

        try:
            index = ToolCatalogIndex.from_tools(
                catalog,
                default_top_k=self._default_top_k,
                max_top_k=self._max_top_k,
            )
        except TimeoutError:
            return [], None, "timeout"
        except ToolCatalogIndexError as exc:
            return [], None, exc.kind
        except Exception:
            return [], None, "malformed_catalog"

        callable_names = self._callable_names_for_catalog(catalog)
        tools_by_name = {tool.name: tool for tool in catalog}
        matches: list[dict[str, Any]] = []
        for candidate in index.search(query, top_k):
            tool = tools_by_name.get(candidate.name)
            if tool is None:
                continue
            payload = self._tool_payload(
                tool,
                callable_names=callable_names,
                candidate=candidate,
            )
            matches.append(payload)
        return matches, index.catalog_fingerprint, "ready"

    def scoped_to_catalog(
        self,
        catalog_provider: ToolCatalogProvider,
        callable_name_provider: CallableToolNameProvider | None = None,
    ) -> ToolSearchTool:
        return ToolSearchTool(
            catalog_provider=catalog_provider,
            callable_name_provider=callable_name_provider,
            catalog_generation_provider=self._catalog_generation_provider,
            discovery_hook=self._discovery_hook,
            default_top_k=self._default_top_k,
            max_top_k=self._max_top_k,
        )

    def _callable_names_for_catalog(self, catalog: list[BaseTool]) -> set[str]:
        if self._callable_name_provider is not None:
            return set(self._callable_name_provider())
        return {tool.name for tool in catalog}

    @staticmethod
    def _tool_payload(
        tool: BaseTool,
        *,
        callable_names: set[str],
        candidate: CatalogSearchCandidate,
    ) -> dict[str, Any]:
        callable_this_turn = tool.name in callable_names
        activation_request = ToolSearchTool._activation_request_for_tool(
            tool,
            callable_this_turn=callable_this_turn,
            activation_broker_callable="tool_activate" in callable_names,
        )
        payload = {
            "name": tool.name,
            "description": tool.description,
            "search_hint": tool.search_hint,
            "read_only": tool.is_read_only,
            "destructive": tool.is_destructive,
            "open_world": tool.is_open_world,
            "capabilities": tool.tool_capabilities,
            "rank": candidate.rank,
            "score": candidate.score,
            "catalog_fingerprint": candidate.catalog_fingerprint,
            "callable_this_turn": callable_this_turn,
            "activation_required": not callable_this_turn,
            "activation_authorizes_tool_call": False,
            "activation_reason": (
                None
                if callable_this_turn
                else (
                    "Tool is discoverable but not exposed as callable in this turn. "
                    "Call tool_activate with this tool name to request policy-gated "
                    "activation."
                    if activation_request is not None
                    and activation_request.get("activation_tool") == "tool_activate"
                    else "Tool is discoverable but not exposed as callable in this turn."
                )
            ),
        }
        if activation_request is not None:
            payload["activation_request"] = activation_request
        return payload

    def _catalog_generation(self) -> int | None:
        if self._catalog_generation_provider is None:
            return None
        try:
            value = self._catalog_generation_provider()
        except Exception:
            return None
        if value is None:
            return None
        return int(value)

    async def _emit_discovery_hook(
        self,
        *,
        query: str,
        matches: list[dict[str, Any]],
        catalog_fingerprint: str | None,
        catalog_generation: int | None,
        catalog_status: str,
        context: Any | None,
    ) -> None:
        if (
            self._discovery_hook is None
            or not matches
            or catalog_fingerprint is None
            or catalog_status != "ready"
        ):
            return
        payload = {
            "query": query,
            "source_query_hash": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "catalog_fingerprint": catalog_fingerprint,
            "catalog_generation": catalog_generation,
            "session_id": getattr(context, "session_id", None),
            "turn_id": (
                context.state.get("turn_id")
                if context is not None and isinstance(getattr(context, "state", None), dict)
                else None
            ),
            "matches": [
                {
                    "tool_name": item["name"],
                    "rank": item["rank"],
                    "score": item["score"],
                    "capability_risk_class": self._capability_risk_class(item),
                }
                for item in matches
            ],
        }
        try:
            result = self._discovery_hook(payload)
            if inspect.isawaitable(result):
                await result
        except Exception:
            return

    @staticmethod
    def _capability_risk_class(item: dict[str, Any]) -> str:
        if item.get("destructive"):
            return "destructive"
        if item.get("read_only"):
            return "read_only"
        if item.get("open_world"):
            return "open_world"
        return "general"

    @staticmethod
    def _activation_request_for_tool(
        tool: BaseTool,
        *,
        callable_this_turn: bool,
        activation_broker_callable: bool = False,
    ) -> dict[str, Any] | None:
        if callable_this_turn:
            return None

        request: dict[str, Any] = {
            "tool_name": tool.name,
            "policy_check": "required",
        }
        if tool.name in {"file_write", "file_edit", "apply_patch"}:
            request["required_intent"] = "workspace_write"
        if activation_broker_callable:
            request.update(
                {
                    "activation_tool": "tool_activate",
                    "arguments": {"tool_name": tool.name},
                }
            )
        return request
