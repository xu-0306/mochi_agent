"""Built-in tool assembly for one effective workspace."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import SecretStr

from mochi.security.policy import resolve_runtime_permission_policy
from mochi.runtime.approvals import InMemoryApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.tools.base import BaseTool
from mochi.tools.exec_command import ExecCommandTool
from mochi.tools.execute_code import ExecuteCodeTool
from mochi.tools.execute_code_v2 import ExecuteCodeV2Tool
from mochi.tools.csv_read import CsvReadTool
from mochi.tools.delegate_subagent_task import DelegateSubagentTaskTool
from mochi.tools.docx_read import DocxReadTool
from mochi.tools.file_ops import FileEditTool, FileReadTool, FileWriteTool
from mochi.tools.glob_search import GlobSearchTool
from mochi.tools.grep_search import GrepSearchTool
from mochi.tools.kill_session import KillSessionTool
from mochi.tools.literature_search import (
    ArxivSearchTool,
    CrossrefSearchTool,
    PubMedSearchTool,
    SemanticScholarSearchTool,
)
from mochi.tools.list_sessions import ListSessionsTool
from mochi.tools.mcp_client import MCPCallTool, McpListResourcesTool, McpReadResourceTool, McpRuntimeManager
from mochi.tools.memory_save import MemorySaveTool
from mochi.tools.memory_search import MemorySearchTool
from mochi.tools.memory_update import MemoryUpdateTool
from mochi.tools.memory_delete import MemoryDeleteTool
from mochi.tools.memory_export import MemoryExportTool
from mochi.tools.notebook_read import NotebookReadTool
from mochi.tools.pdf_read import PdfReadTool
from mochi.tools.process_control import ProcessPollTool, ProcessStopTool
from mochi.tools.process_service import ProcessService
from mochi.tools.read_session import ReadSessionTool
from mochi.tools.registry import ToolRegistry
from mochi.tools.shell import ShellTool
from mochi.tools.tool_search import ToolSearchTool
from mochi.tools.web_crawl import WebCrawlTool
from mochi.tools.web_fetch import WebFetchTool
from mochi.tools.web_search import WebSearchTool
from mochi.tools.write_stdin import WriteStdinTool

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
        self._exec_runtime = ExecRuntime(
            default_shell=self._resolve_exec_default_shell(),
            output_tail_limit=self._config.security.exec_session_output_limit,
        )
        self._exec_approval_store = InMemoryApprovalStore()
        self._builtins = self._build_specs()

    @property
    def tool_groups(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = {}
        for spec in self._builtins:
            groups.setdefault(spec.tool_group, []).append(spec.name)
        groups.setdefault("workspace", []).append("tool_search")
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
        registry.register(ToolSearchTool(catalog_provider=registry.list_tools))
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
            BuiltInToolSpec("web_crawl", "shared", "web", self._build_web_crawl),
            BuiltInToolSpec("get_current_time", "shared", "web", self._build_datetime),
            BuiltInToolSpec("calculator", "shared", "web", self._build_calculator),
            BuiltInToolSpec("delegate_subagent_task", "workspace", "workspace", self._build_delegate_subagent_task),
            BuiltInToolSpec("exec_command", "workspace", "workspace", self._build_exec_command),
            BuiltInToolSpec("read_session", "workspace", "workspace", self._build_read_session),
            BuiltInToolSpec("write_stdin", "workspace", "workspace", self._build_write_stdin),
            BuiltInToolSpec("kill_session", "workspace", "workspace", self._build_kill_session),
            BuiltInToolSpec("list_sessions", "workspace", "workspace", self._build_list_sessions),
            BuiltInToolSpec("shell", "workspace", "workspace", self._build_shell),
            BuiltInToolSpec("file_read", "workspace", "workspace", self._build_file_read),
            BuiltInToolSpec("glob_search", "workspace", "workspace", self._build_glob_search),
            BuiltInToolSpec("grep_search", "workspace", "workspace", self._build_grep_search),
            BuiltInToolSpec("csv_read", "workspace", "workspace", self._build_csv_read),
            BuiltInToolSpec("docx_read", "workspace", "workspace", self._build_docx_read),
            BuiltInToolSpec("pdf_read", "workspace", "workspace", self._build_pdf_read),
            BuiltInToolSpec("notebook_read", "workspace", "workspace", self._build_notebook_read),
            BuiltInToolSpec("file_write", "workspace", "workspace", self._build_file_write),
            BuiltInToolSpec("file_edit", "workspace", "workspace", self._build_file_edit),
            BuiltInToolSpec("execute_code", "workspace", "workspace", self._build_execute_code),
            BuiltInToolSpec("execute_code_v2", "workspace", "workspace", self._build_execute_code_v2),
            BuiltInToolSpec("process_poll", "workspace", "workspace", self._build_process_poll),
            BuiltInToolSpec("process_stop", "workspace", "workspace", self._build_process_stop),
            BuiltInToolSpec("arxiv_search", "shared", "literature", self._build_arxiv),
            BuiltInToolSpec("semantic_scholar_search", "shared", "literature", self._build_semantic_scholar),
            BuiltInToolSpec("crossref_search", "shared", "literature", self._build_crossref),
            BuiltInToolSpec("pubmed_search", "shared", "literature", self._build_pubmed),
            BuiltInToolSpec("memory_search", "workspace", "memory", self._build_memory_search),
            BuiltInToolSpec("memory_save", "workspace", "memory", self._build_memory_save),
            BuiltInToolSpec("memory_update", "workspace", "memory", self._build_memory_update),
            BuiltInToolSpec("memory_delete", "workspace", "memory", self._build_memory_delete),
            BuiltInToolSpec("memory_export", "workspace", "memory", self._build_memory_export),
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

    def _build_delegate_subagent_task(
        self,
        config: MochiConfig,
        workspace_dir: str,
        services: dict[str, Any],
    ) -> BaseTool:
        del config, workspace_dir, services
        return DelegateSubagentTaskTool()

    def _build_web_fetch(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del workspace_dir, services
        tc = config.tools
        jina_key = self._secret(tc.web_fetch_jina_api_key) or self._secret(tc.web_search_jina_api_key)
        return WebFetchTool(
            timeout=tc.http_timeout,
            jina_api_key=jina_key,
            extractor=tc.web_fetch_extractor,
        )

    def _build_web_crawl(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del workspace_dir, services
        return WebCrawlTool(timeout=config.tools.http_timeout)

    def _build_shell(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        runtime_policy = resolve_runtime_permission_policy(config.security)
        return ShellTool(
            allowlist=config.security.shell_command_allowlist,
            workspace_dir=workspace_dir,
            require_approval=runtime_policy.require_approval_for_shell,
            process_service=services.get("process_service"),
        )

    def _build_exec_command(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del services
        runtime_policy = resolve_runtime_permission_policy(config.security)
        return ExecCommandTool(
            runtime=self._exec_runtime,
            approval_store=self._exec_approval_store,
            workspace_dir=workspace_dir,
            allowlist=config.security.shell_command_allowlist,
            allowed_env_vars=config.security.exec_allowed_env_vars,
            require_approval=runtime_policy.require_approval_for_exec,
            default_timeout_sec=config.security.exec_default_timeout_sec,
        )

    def _build_read_session(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config, workspace_dir, services
        return ReadSessionTool(runtime=self._exec_runtime)

    def _build_write_stdin(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config, workspace_dir, services
        return WriteStdinTool(runtime=self._exec_runtime)

    def _build_kill_session(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config, workspace_dir, services
        return KillSessionTool(runtime=self._exec_runtime)

    def _build_list_sessions(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config, workspace_dir, services
        return ListSessionsTool(runtime=self._exec_runtime)

    def _build_file_read(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del services
        runtime_policy = resolve_runtime_permission_policy(config.security)
        return FileReadTool(
            workspace_dir=workspace_dir,
            path_scope=runtime_policy.file_ops_scope,
        )

    def _build_glob_search(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config, services
        return GlobSearchTool(workspace_dir=workspace_dir)

    def _build_grep_search(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config, services
        return GrepSearchTool(workspace_dir=workspace_dir)

    def _build_csv_read(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del services
        runtime_policy = resolve_runtime_permission_policy(config.security)
        return CsvReadTool(
            workspace_dir=workspace_dir,
            path_scope=runtime_policy.file_ops_scope,
        )

    def _build_docx_read(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del services
        runtime_policy = resolve_runtime_permission_policy(config.security)
        return DocxReadTool(
            workspace_dir=workspace_dir,
            path_scope=runtime_policy.file_ops_scope,
        )

    def _build_pdf_read(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del services
        runtime_policy = resolve_runtime_permission_policy(config.security)
        return PdfReadTool(
            workspace_dir=workspace_dir,
            path_scope=runtime_policy.file_ops_scope,
        )

    def _build_notebook_read(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del services
        runtime_policy = resolve_runtime_permission_policy(config.security)
        return NotebookReadTool(
            workspace_dir=workspace_dir,
            path_scope=runtime_policy.file_ops_scope,
        )

    def _build_file_write(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del services
        runtime_policy = resolve_runtime_permission_policy(config.security)
        return FileWriteTool(
            workspace_dir=workspace_dir,
            path_scope=runtime_policy.file_ops_scope,
            require_approval=runtime_policy.require_approval_for_file_write,
            max_write_size_mb=config.security.max_file_write_size_mb,
            undo_max_size_mb=config.security.file_undo_max_size_mb,
        )

    def _build_file_edit(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del services
        runtime_policy = resolve_runtime_permission_policy(config.security)
        return FileEditTool(
            workspace_dir=workspace_dir,
            path_scope=runtime_policy.file_ops_scope,
            require_approval=runtime_policy.require_approval_for_file_write,
            max_write_size_mb=config.security.max_file_write_size_mb,
            undo_max_size_mb=config.security.file_undo_max_size_mb,
        )

    def _build_execute_code(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        runtime_policy = resolve_runtime_permission_policy(config.security)
        return ExecuteCodeTool(
            workspace_dir=workspace_dir,
            require_approval=runtime_policy.require_approval_for_shell,
            process_service=services.get("process_service"),
        )

    def _build_execute_code_v2(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del services
        runtime_policy = resolve_runtime_permission_policy(config.security)
        return ExecuteCodeV2Tool(
            workspace_dir=workspace_dir,
            require_approval=runtime_policy.require_approval_for_shell,
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

    def _build_memory_update(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config
        return MemoryUpdateTool(
            memory_store=services["memory_store"],
            workspace_dir=workspace_dir,
        )

    def _build_memory_delete(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config
        return MemoryDeleteTool(
            memory_store=services["memory_store"],
            workspace_dir=workspace_dir,
        )

    def _build_memory_export(self, config: MochiConfig, workspace_dir: str, services: dict[str, Any]) -> BaseTool:
        del config
        return MemoryExportTool(
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

    def _resolve_exec_default_shell(self) -> str:
        configured = self._config.security.exec_default_shell
        if configured != "auto":
            return configured
        from mochi.config import defaults

        return "powershell" if defaults.running_on_windows() else "bash"
