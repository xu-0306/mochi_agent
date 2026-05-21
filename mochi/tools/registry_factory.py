"""Built-in tool assembly for one effective workspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import SecretStr

from mochi.tools.base import BaseTool
from mochi.tools.execute_code import ExecuteCodeTool
from mochi.tools.file_ops import FileEditTool, FileReadTool, FileWriteTool
from mochi.tools.literature_search import (
    ArxivSearchTool,
    CrossrefSearchTool,
    PubMedSearchTool,
    SemanticScholarSearchTool,
)
from mochi.tools.mcp_client import MCPCallTool, McpListResourcesTool, McpReadResourceTool, McpRuntimeManager
from mochi.tools.memory_save import MemorySaveTool
from mochi.tools.memory_search import MemorySearchTool
from mochi.tools.process_control import ProcessPollTool, ProcessStopTool
from mochi.tools.process_service import ProcessService
from mochi.tools.registry import ToolRegistry
from mochi.tools.shell import ShellTool
from mochi.tools.web_fetch import WebFetchTool
from mochi.tools.web_search import WebSearchTool

if TYPE_CHECKING:
    from mochi.config.schema import MochiConfig
    from mochi.memory.store import MemoryStore


BindingScope = Literal["shared", "workspace"]


@dataclass(frozen=True)
class BuiltInToolSpec:
    """One built-in tool registration contract."""

    name: str
    binding_scope: BindingScope
    tool_group: str
    factory: Any


class ToolRegistryFactory:
    """Create per-workspace registries without duplicating assembly logic."""

    def __init__(
        self,
        config: MochiConfig,
        *,
        memory_store: MemoryStore,
        mcp_runtime_manager: McpRuntimeManager | None = None,
    ) -> None:
        self._config = config
        self._memory_store = memory_store
        self._mcp_runtime_manager = mcp_runtime_manager
        self._process_service = ProcessService()
        self._builtins = self._build_specs()

    @property
    def tool_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for spec in self._builtins:
            groups.setdefault(spec.tool_group, []).append(spec.name)
        if self._mcp_runtime_manager is not None:
            for tool in self._mcp_runtime_manager.materialize_tools():
                groups.setdefault("mcp", []).append(tool.name)
        return groups

    def create_registry(self, workspace_dir: str) -> ToolRegistry:
        """Create a registry bound to one effective workspace."""
        registry = ToolRegistry(
            extra_dirs=self._config.tools.extra_tools_dirs or None,
            discover_builtin=False,
        )
        for spec in self._builtins:
            if (
                spec.name in {"mcp_list_resources", "mcp_read_resource"}
                and self._mcp_runtime_manager is None
            ):
                continue
            registry.register(spec.factory(self._config, workspace_dir, self._services()))
        if self._mcp_runtime_manager is not None:
            for tool in self._mcp_runtime_manager.materialize_tools():
                registry.register(tool)
        return registry

    def _services(self) -> dict[str, Any]:
        return {
            "memory_store": self._memory_store,
            "mcp_runtime_manager": self._mcp_runtime_manager,
            "process_service": self._process_service,
        }

    def _build_specs(self) -> list[BuiltInToolSpec]:
        return [
            BuiltInToolSpec("web_search", "shared", "web", self._build_web_search),
            BuiltInToolSpec("web_fetch", "shared", "web", self._build_web_fetch),
            BuiltInToolSpec("get_current_time", "shared", "web", self._build_datetime),
            BuiltInToolSpec("calculator", "shared", "web", self._build_calculator),
            BuiltInToolSpec("shell", "workspace", "workspace", self._build_shell),
            BuiltInToolSpec("file_read", "workspace", "workspace", self._build_file_read),
            BuiltInToolSpec("file_write", "workspace", "workspace", self._build_file_write),
            BuiltInToolSpec("file_edit", "workspace", "workspace", self._build_file_edit),
            BuiltInToolSpec("execute_code", "workspace", "workspace", self._build_execute_code),
            BuiltInToolSpec("process_poll", "workspace", "workspace", self._build_process_poll),
            BuiltInToolSpec("process_stop", "workspace", "workspace", self._build_process_stop),
            BuiltInToolSpec("arxiv_search", "shared", "literature", self._build_arxiv),
            BuiltInToolSpec("semantic_scholar_search", "shared", "literature", self._build_semantic_scholar),
            BuiltInToolSpec("crossref_search", "shared", "literature", self._build_crossref),
            BuiltInToolSpec("pubmed_search", "shared", "literature", self._build_pubmed),
            BuiltInToolSpec("memory_search", "workspace", "memory", self._build_memory_search),
            BuiltInToolSpec("memory_save", "workspace", "memory", self._build_memory_save),
            BuiltInToolSpec("mcp_call", "shared", "mcp", self._build_mcp_call),
            BuiltInToolSpec("mcp_list_resources", "shared", "mcp", self._build_mcp_list_resources),
            BuiltInToolSpec("mcp_read_resource", "shared", "mcp", self._build_mcp_read_resource),
        ]

    @staticmethod
    def _secret(value: SecretStr | None) -> str | None:
        return value.get_secret_value() if value is not None else None

    def _build_web_search(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del workspace_dir, services
        tc = config.tools
        return WebSearchTool(
            engine=tc.web_search_engine,
            timeout=tc.http_timeout,
            fallback_engines=tc.web_search_fallback_engines,
            searxng_base_url=tc.web_search_searxng_base_url,
            brave_api_key=self._secret(tc.web_search_brave_api_key),
            tavily_api_key=self._secret(tc.web_search_tavily_api_key),
            serper_api_key=self._secret(tc.web_search_serper_api_key),
            jina_api_key=self._secret(tc.web_search_jina_api_key),
            exa_api_key=self._secret(tc.web_search_exa_api_key),
            language=tc.web_search_language,
            region=tc.web_search_region,
        )

    def _build_web_fetch(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del workspace_dir, services
        tc = config.tools
        jina_key = self._secret(tc.web_fetch_jina_api_key) or self._secret(tc.web_search_jina_api_key)
        return WebFetchTool(
            timeout=tc.http_timeout,
            jina_api_key=jina_key,
            extractor=tc.web_fetch_extractor,
        )

    def _build_shell(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del services
        return ShellTool(
            allowlist=config.security.shell_command_allowlist,
            workspace_dir=workspace_dir,
            require_approval=config.security.require_approval_for_shell,
            process_service=services.get("process_service"),
        )

    def _build_file_read(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del services
        return FileReadTool(
            workspace_dir=workspace_dir,
            path_scope=config.security.file_ops_scope,
        )

    def _build_file_write(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del services
        return FileWriteTool(
            workspace_dir=workspace_dir,
            path_scope=config.security.file_ops_scope,
            require_approval=config.security.require_approval_for_file_write,
            max_write_size_mb=config.security.max_file_write_size_mb,
            undo_max_size_mb=config.security.file_undo_max_size_mb,
        )

    def _build_file_edit(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del services
        return FileEditTool(
            workspace_dir=workspace_dir,
            path_scope=config.security.file_ops_scope,
            require_approval=config.security.require_approval_for_file_write,
            max_write_size_mb=config.security.max_file_write_size_mb,
            undo_max_size_mb=config.security.file_undo_max_size_mb,
        )

    def _build_execute_code(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del services
        return ExecuteCodeTool(
            workspace_dir=workspace_dir,
            require_approval=config.security.require_approval_for_shell,
            process_service=services.get("process_service"),
        )

    def _build_process_poll(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config, workspace_dir
        return ProcessPollTool(process_service=services["process_service"])

    def _build_process_stop(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config, workspace_dir
        return ProcessStopTool(process_service=services["process_service"])

    def _build_arxiv(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del workspace_dir, services
        return ArxivSearchTool(timeout=config.tools.http_timeout)

    def _build_semantic_scholar(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del workspace_dir, services
        return SemanticScholarSearchTool(
            timeout=config.tools.http_timeout,
            api_key=self._secret(config.tools.semantic_scholar_api_key),
        )

    def _build_crossref(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del workspace_dir, services
        return CrossrefSearchTool(
            timeout=config.tools.http_timeout,
            mailto=config.tools.crossref_mailto,
        )

    def _build_pubmed(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del workspace_dir, services
        return PubMedSearchTool(
            timeout=config.tools.http_timeout,
            email=config.tools.pubmed_email,
            api_key=self._secret(config.tools.pubmed_api_key),
        )

    def _build_memory_search(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        return MemorySearchTool(
            memory_store=services["memory_store"],
            workspace_dir=workspace_dir,
            default_top_k=config.memory.fts_top_k,
        )

    def _build_memory_save(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        return MemorySaveTool(
            memory_store=services["memory_store"],
            workspace_dir=workspace_dir,
        )

    def _build_mcp_call(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config, workspace_dir
        runtime = services.get("mcp_runtime_manager")
        if runtime is not None:
            return MCPCallTool(runtime=runtime)
        return MCPCallTool()

    def _build_mcp_list_resources(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config, workspace_dir
        runtime = services.get("mcp_runtime_manager")
        if runtime is None:
            raise RuntimeError("MCP runtime is required for mcp_list_resources.")
        return McpListResourcesTool(runtime=runtime)

    def _build_mcp_read_resource(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config, workspace_dir
        runtime = services.get("mcp_runtime_manager")
        if runtime is None:
            raise RuntimeError("MCP runtime is required for mcp_read_resource.")
        return McpReadResourceTool(runtime=runtime)

    def _build_calculator(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config, workspace_dir, services
        from mochi.tools.calculator import CalculatorTool

        return CalculatorTool()

    def _build_datetime(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config, workspace_dir, services
        from mochi.tools.datetime_tool import DateTimeTool

        return DateTimeTool()
