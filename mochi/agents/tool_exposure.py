"""Subset-first tool exposure planning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

from mochi.agents.tool_intent_router import ToolIntent, normalize_tool_intent_name
from mochi.backends.base import BaseLLMBackend


@dataclass(frozen=True)
class ToolExposurePlan:
    """Selected tool names for one turn."""

    tool_names: list[str]
    matched_groups: list[str]
    limit: int
    discoverable_tool_names: list[str] = field(default_factory=list)
    workspace_bound: bool = False
    attachment_count: int = 0
    intent_route: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def exposure_metadata(self) -> dict[str, Any]:
        payload = {
            "exposed_tools": list(self.tool_names),
            "workspace_bound": self.workspace_bound,
            "attachment_count": self.attachment_count,
        }
        if self.intent_route is not None:
            payload["intent_route"] = dict(self.intent_route)
        if self.diagnostics:
            payload["diagnostics"] = dict(self.diagnostics)
        return payload


class ToolExposurePlanner:
    """Select a smaller, intent-matched tool subset for one turn."""
    _CORE_WORKSPACE_READ_ONLY_TOOLS: tuple[str, ...] = (
        "file_read",
        "tool_result_read",
        "glob_search",
        "grep_search",
        "csv_read",
        "pdf_read",
        "docx_read",
        "notebook_read",
    )

    _CORE_WORKSPACE_WRITE_TOOLS: tuple[str, ...] = (
        "file_write",
        "file_edit",
        "apply_patch",
    )

    _RISKY_TOOLS: frozenset[str] = frozenset(
        {
            "exec_command",
            "execute_code",
            "execute_code_v2",
            "write_stdin",
            "kill_session",
            "file_write",
            "file_edit",
            "apply_patch",
            "process_stop",
            "mcp_call",
            "delegate_subagent_task",
        }
    )
    _STRICT_BLOCKED_TOOLS: frozenset[str] = frozenset(
        {
            "exec_command",
            "execute_code",
            "execute_code_v2",
            "write_stdin",
            "kill_session",
            "process_stop",
            "mcp_call",
        }
    )
    _CONTEXTUAL_TOOLS: frozenset[str] = frozenset(
        {
            "read_session",
            "write_stdin",
            "kill_session",
            "list_sessions",
            "process_poll",
            "process_stop",
        }
    )
    _AUTONOMY_LIMITS: dict[str, tuple[int, int]] = {
        "strict": (4, 2),
        "trusted_workspace": (6, 3),
        "auto_review": (8, 3),
        "high_autonomy": (10, 4),
    }
    _TOOL_PRIORITY: dict[str, int] = {
        "glob_search": 5,
        "grep_search": 8,
        "repo_map": 9,
        "file_read": 10,
        "read_symbol": 11,
        "csv_read": 11,
        "pdf_read": 12,
        "docx_read": 13,
        "notebook_read": 14,
        "tool_search": 15,
        "delegate_subagent_task": 18,
        "file_write": 20,
        "file_edit": 30,
        "apply_patch": 35,
        "exec_command": 40,
        "execute_code": 50,
        "execute_code_v2": 55,
        "read_session": 60,
        "write_stdin": 70,
        "list_sessions": 80,
        "kill_session": 90,
        "process_poll": 100,
        "process_stop": 110,
        "memory_search": 130,
        "memory_save": 140,
        "arxiv_search": 142,
        "semantic_scholar_search": 144,
        "crossref_search": 146,
        "pubmed_search": 148,
        "web_search": 150,
        "discourse_topic_collect": 155,
        "web_fetch": 160,
        "web_crawl": 165,
        "get_current_time": 170,
        "calculator": 180,
    }
    # Legacy keyword routing only remains as a ranking hint layer.
    # New work must not add more keyword-gating rules here.
    # Future direction: dynamic discovery and capability-first exposure.
    _GROUP_KEYWORDS: dict[str, tuple[str, ...]] = {
        "web": (
            "today",
            "weather",
            "news",
            "latest",
            "search web",
            "crawl",
            "site",
            "url",
            "http",
            "https",
        ),
        "workspace": (
            "code",
            "file",
            "files",
            "grep",
            "glob",
            "todo",
            "test",
            "run",
            "debug",
            "repo",
            "project",
            "workspace",
            "folder",
            "directory",
            "path",
            "local file",
            "exec",
            "session",
            "stdin",
            "subagent",
            "delegate",
            "background task",
            "controlled execution",
        ),
        "literature": (
            "paper",
            "papers",
            "research",
            "academic",
            "scholarly",
            "arxiv",
            "pubmed",
            "doi",
            "citation",
            "citations",
            "journal",
            "abstract",
            "literature",
            "\u8ad6\u6587",
            "\u7814\u7a76",
            "\u6587\u737b",
            "\u5b78\u8853",
            "\u5f15\u7528",
            "\u671f\u520a",
            "\u6458\u8981",
        ),
        "memory": (
            "remember",
            "memory",
            "skill",
        ),
        "mcp": (
            "mcp",
            "resource",
            "server",
        ),
    }
    _CONTEXT_KEYWORDS: tuple[str, ...] = (
        "background",
        "process",
        "poll",
        "tail",
        "stop",
        "session",
        "stdin",
        "pty",
    )
    _TOOL_DISCOVERY_KEYWORDS: tuple[str, ...] = (
        "which tool",
        "what tool",
        "available tools",
        "list tools",
        "find tool",
        "tool should i use",
        "tool to use",
    )
    _ATTACHED_WORKSPACE_FILE_MARKERS: tuple[str, ...] = (
        "attached workspace files:",
        "attached workspace file:",
    )
    _READ_ONLY_FILE_INTENT_KEYWORDS: tuple[str, ...] = (
        "read",
        "inspect",
        "review",
        "summarize",
        "summary",
        "analyze",
        "analyse",
        "extract",
        "file-reading",
    )
    # Fallback recall hints only. Structured routed intent is the source of
    # truth for workspace write exposure; do not expand this into a primary
    # natural-language classifier.
    _ATTACHMENT_MUTATION_INTENT_KEYWORDS: tuple[str, ...] = (
        "edit",
        "edited",
        "editing",
        "update",
        "updated",
        "updating",
        "modify",
        "modified",
        "modifying",
        "change",
        "changed",
        "rewrite",
        "rewritten",
        "revise",
        "revised",
        "patch",
        "save",
        "replace",
        "fix",
        "編輯",
        "编辑",
        "更新",
        "修改",
        "變更",
        "变更",
        "改寫",
        "改写",
        "修補",
        "修复",
        "修復",
        "儲存",
        "保存",
        "寫入",
        "写入",
        "建立",
        "创建",
        "新增",
        "刪除",
        "删除",
        "套用 patch",
    )
    _ATTACHMENT_PROCESSING_INTENT_KEYWORDS: tuple[str, ...] = (
        "translate",
        "translation",
        "transform",
        "convert",
        "reformat",
        "normalize",
        "parse",
        "json",
        "json array",
        "structured output",
        "output json",
        "return json",
        "翻譯",
        "轉換",
        "整理",
    )
    _NON_MUTATING_EXECUTION_TOOLS: frozenset[str] = frozenset(
        {
            "exec_command",
            "execute_code",
            "execute_code_v2",
        }
    )
    _FILE_BROWSE_INTENT_KEYWORDS: tuple[str, ...] = (
        "browse",
        "directory",
        "directories",
        "file",
        "files",
        "find",
        "folder",
        "folders",
        "grep",
        "inspect",
        "list",
        "match",
        "path",
        "paths",
        "pdf",
        "read",
        "repo",
        "review",
        "search",
        "todo",
    )
    _EXECUTION_INTENT_KEYWORDS: tuple[str, ...] = (
        "background",
        "benchmark",
        "build",
        "command",
        "compile",
        "debug",
        "execute",
        "install",
        "launch",
        "run",
        "script",
        "server",
        "session",
        "start",
        "stdin",
        "stop",
        "test",
        "tty",
    )
    _FILETYPE_TOOL_KEYWORDS: dict[str, tuple[str, ...]] = {
        "pdf_read": ("pdf",),
        "csv_read": ("csv", "tsv", "spreadsheet"),
        "docx_read": ("docx", "word document", ".docx"),
        "notebook_read": ("notebook", "ipynb", "jupyter"),
    }
    _REPO_NAVIGATION_INTENT_KEYWORDS: tuple[str, ...] = (
        "repo map",
        "repo structure",
        "project structure",
        "codebase overview",
        "large repo",
        "larger repo",
        "orient",
        "orientation",
        "where to start",
        "where should i start",
    )
    _SYMBOL_LOOKUP_INTENT_KEYWORDS: tuple[str, ...] = (
        "symbol",
        "definition",
        "class",
        "function",
        "method",
        "declaration",
    )
    _INTENT_REQUIRED_TOOL_KEYWORDS: dict[str, tuple[str, ...]] = {
        "glob_search": ("find", "search", "grep", "glob", "todo", "match"),
        "grep_search": ("find", "search", "grep", "todo", "match"),
        "pdf_read": ("pdf",),
        "csv_read": ("csv", "tsv", "spreadsheet"),
        "docx_read": ("docx", "word document", ".docx"),
        "notebook_read": ("notebook", "ipynb", "jupyter"),
        "tool_search": (
            "which tool",
            "what tool",
            "available tools",
            "list tools",
            "find tool",
            "tool should i use",
            "tool to use",
        ),
        "delegate_subagent_task": (
            "subagent",
            "delegate",
            "background task",
            "controlled execution",
        ),
        "web_crawl": ("crawl", "site", "url", "http", "https"),
        "calculator": (
            "calculate",
            "calculator",
            "math",
            "sum",
            "average",
            "percent",
            "percentage",
        ),
    }
    _RESEARCH_KEYWORDS: tuple[str, ...] = (
        *_GROUP_KEYWORDS["literature"],
        "preprint",
        "manuscript",
        "publication",
        "publications",
        "survey paper",
        "survey papers",
        "related work",
        "bibliography",
        "reference",
        "references",
    )
    _CITATION_KEYWORDS: tuple[str, ...] = (
        "doi",
        "citation",
        "citations",
        "cite",
        "cited",
        "reference",
        "references",
        "\u5f15\u7528",
    )
    _BIOMEDICAL_KEYWORDS: tuple[str, ...] = (
        "biomedical",
        "bioinformatics",
        "clinical",
        "medicine",
        "medical",
        "medline",
        "pubmed",
        "\u91ab\u5b78",
        "\u751f\u91ab",
        "\u81e8\u5e8a",
    )
    _RECENT_RESEARCH_KEYWORDS: tuple[str, ...] = (
        "recent",
        "recent years",
        "latest papers",
        "new papers",
        "\u8fd1\u5e7e\u5e74",
        "\u8fd1\u5e74",
        "\u6700\u65b0\u8ad6\u6587",
    )
    _EXPLICIT_SUBAGENT_DELEGATION_KEYWORDS: tuple[str, ...] = (
        "subagent",
        "subagents",
        "\u526f\u4ee3\u7406",
        "\u5b50\u4ee3\u7406",
        "\u5b50\u4ee3\u7406\u4eba",
        "delegate",
        "delegated",
        "\u59d4\u6d3e",
        "\u5206\u6d3e",
        "\u6d3e\u7d66",
        "\u4ea4\u7d66",
        "\u958b subagent",
        "\u958bsubagent",
        "\u958b \u5b50\u4ee3\u7406",
        "\u958b\u5b50\u4ee3\u7406",
        "\u30b5\u30d6\u30a8\u30fc\u30b8\u30a7\u30f3\u30c8",
        "parallel research",
        "independent comparison",
        "multi-perspective",
        "multiple perspectives",
        "research a and b",
        "compare approach a and approach b",
    )
    _DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-z0-9]+\b", re.IGNORECASE)

    def __init__(self, *, tool_groups: dict[str, list[str]]) -> None:
        self._tool_groups = tool_groups

    def plan(
        self,
        *,
        message: str,
        user_intent_message: str | None = None,
        available_tool_names: list[str],
        backend: BaseLLMBackend,
        session_bound_workspace: bool,
        autonomy_mode: str | None = None,
        preferred_tool_names: list[str] | None = None,
        tool_capabilities: dict[str, dict[str, Any]] | None = None,
        attachment_count: int = 0,
        workspace_attachment_count: int = 0,
        tool_mode: Literal["disabled", "auto", "required"] = "auto",
        routed_intent: ToolIntent | None = None,
        intent_confidence: float | None = None,
        intent_source: str | None = None,
        intent_rationale: str | None = None,
    ) -> ToolExposurePlan:
        intent_route_metadata = self._intent_route_metadata(
            routed_intent=routed_intent,
            intent_confidence=intent_confidence,
            intent_source=intent_source,
            intent_rationale=intent_rationale,
        )
        if tool_mode == "disabled":
            return ToolExposurePlan(
                tool_names=[],
                matched_groups=[],
                limit=0,
                discoverable_tool_names=[],
                workspace_bound=session_bound_workspace,
                attachment_count=max(0, attachment_count),
                intent_route=intent_route_metadata,
                diagnostics={
                    "stage": "planner",
                    "disable_reason": "tool_mode_disabled",
                    "available_tool_count": len(available_tool_names),
                    "tool_mode": tool_mode,
                },
            )

        backend_info = backend.get_model_info()
        metadata = backend_info.metadata if isinstance(backend_info.metadata, dict) else {}
        prompt_guided_ollama = (
            backend_info.backend_type == "ollama"
            and (metadata.get("tool_calling_protocol") or metadata.get("tool_calling_style"))
            == "prompt_guided"
        )
        if metadata.get("tool_calling_blocked") is True or (
            metadata.get("tool_call_mode") == "unavailable" and not prompt_guided_ollama
        ):
            return ToolExposurePlan(
                tool_names=[],
                matched_groups=[],
                limit=0,
                discoverable_tool_names=[],
                workspace_bound=session_bound_workspace,
                attachment_count=max(0, attachment_count),
                intent_route=intent_route_metadata,
                diagnostics={
                    "stage": "planner",
                    "disable_reason": (
                        "backend_tool_calling_blocked"
                        if metadata.get("tool_calling_blocked") is True
                        else "backend_tool_calling_unavailable"
                    ),
                    "available_tool_count": len(available_tool_names),
                    "backend": self._backend_diagnostics(backend_info, metadata),
                },
            )

        lowered = message.lower()
        # Only user-authored intent should drive attachment mutation detection.
        # Falling back to the structured planner message would let attachment
        # filenames/paths reintroduce false positives like `updated-report.pdf`.
        lowered_user_intent = (user_intent_message or "").lower()
        lowered_primary_intent = lowered_user_intent or lowered
        normalized_routed_intent = self._normalize_routed_intent(routed_intent)
        routed_workspace_write = normalized_routed_intent == "workspace_write"
        routed_execution_request = normalized_routed_intent == "execution_or_process"
        routed_workspace_request = normalized_routed_intent in {
            "workspace_read",
            "workspace_write",
            "execution_or_process",
        }
        routed_tool_discovery = normalized_routed_intent == "tool_discovery"
        routed_research_request = normalized_routed_intent == "literature_research"
        routed_web_request = normalized_routed_intent == "open_world_lookup"
        normalized_attachment_count = max(0, attachment_count)
        normalized_workspace_attachment_count = max(0, workspace_attachment_count)
        attached_workspace_files = normalized_workspace_attachment_count > 0 or any(
            marker in lowered for marker in self._ATTACHED_WORKSPACE_FILE_MARKERS
        )
        explicit_workspace_target = attached_workspace_files or bool(
            re.search(
                (
                    r"(?:[a-z]:[\\/]|(?:\.\.?[\\/])|"
                    r"\b(?:file|files|path|paths|artifact|workspace)\b|"
                    r"\b[\w.-]+\.(?:py|txt|md|json|ya?ml|csv|ts|tsx|js|jsx|toml|ini|cfg|sql|html|css)\b)"
                ),
                lowered_user_intent,
            )
        )
        route_allows_mutation_fallback = normalized_routed_intent in {None, "ambiguous"}
        attachment_mutation_request = routed_workspace_write or (
            route_allows_mutation_fallback
            and explicit_workspace_target
            and not routed_tool_discovery
            and self._matches_any_keyword(
                lowered_user_intent,
                self._ATTACHMENT_MUTATION_INTENT_KEYWORDS,
            )
        )
        attachment_processing_request = (
            attached_workspace_files
            and (
                routed_execution_request
                or self._matches_any_keyword(
                    lowered_primary_intent,
                    self._ATTACHMENT_PROCESSING_INTENT_KEYWORDS,
                )
            )
        )
        read_only_file_request = (
            attached_workspace_files
            and not attachment_mutation_request
            and not routed_execution_request
            and not self._matches_any_keyword(lowered, self._EXECUTION_INTENT_KEYWORDS)
        )
        file_browse_request = (
            self._matches_any_keyword(lowered, self._FILE_BROWSE_INTENT_KEYWORDS)
            and not routed_execution_request
            and not self._matches_any_keyword(lowered, self._EXECUTION_INTENT_KEYWORDS)
        )

        normalized_capabilities = {
            tool_name: self._normalize_tool_capabilities(
                (tool_capabilities or {}).get(tool_name),
            )
            for tool_name in available_tool_names
        }
        tools_by_group = self._build_tools_by_group(
            available_tool_names,
            normalized_capabilities,
        )

        research_request = routed_research_request or self._is_research_request(lowered)
        citation_request = self._is_citation_request(lowered)
        biomedical_request = self._matches_any_keyword(lowered, self._BIOMEDICAL_KEYWORDS)
        recent_research_request = self._matches_any_keyword(lowered, self._RECENT_RESEARCH_KEYWORDS)
        workspace_request = routed_workspace_request or self._matches_any_keyword(
            lowered,
            self._GROUP_KEYWORDS["workspace"],
        )
        web_request = routed_web_request or self._matches_any_keyword(lowered, self._GROUP_KEYWORDS["web"])
        explicit_workspace_request = (
            routed_workspace_request
            or workspace_request
            or attached_workspace_files
            or file_browse_request
            or attachment_mutation_request
            or attachment_processing_request
        )

        matched_groups: list[str] = []
        for group_name in ("memory", "mcp"):
            if (
                tools_by_group[group_name]
                and self._matches_any_keyword(lowered, self._GROUP_KEYWORDS[group_name])
            ):
                matched_groups.append(group_name)

        if routed_research_request and tools_by_group["literature"]:
            matched_groups.append("literature")
            if tools_by_group["web"]:
                matched_groups.append("web")
        elif routed_web_request and tools_by_group["web"]:
            matched_groups.append("web")
        elif routed_workspace_request and tools_by_group["workspace"]:
            matched_groups.append("workspace")
        elif research_request and tools_by_group["literature"]:
            matched_groups.append("literature")
            if tools_by_group["web"]:
                matched_groups.append("web")
        elif web_request and tools_by_group["web"]:
            matched_groups.append("web")
        elif workspace_request and tools_by_group["workspace"]:
            matched_groups.append("workspace")

        if not matched_groups:
            default_group = "workspace" if session_bound_workspace else "web"
            fallback_group = "web" if default_group == "workspace" else "workspace"
            if tools_by_group[default_group]:
                matched_groups.append(default_group)
            elif tools_by_group[fallback_group]:
                matched_groups.append(fallback_group)

        base_limit = 6 if backend_info.backend_type in {"gguf", "safetensors"} else 10
        effective_mode = autonomy_mode or "trusted_workspace"
        mode_limit, risky_limit = self._AUTONOMY_LIMITS.get(effective_mode, (base_limit, 1))
        limit = min(base_limit, mode_limit)

        available_order = {name: index for index, name in enumerate(available_tool_names)}
        preferred = [
            tool_name
            for tool_name in (preferred_tool_names or [])
            if tool_name in available_order
        ]
        if "tool_search" in available_order and (
            routed_tool_discovery
            or self._matches_any_keyword(
                lowered,
                self._TOOL_DISCOVERY_KEYWORDS,
            )
        ):
            preferred = [
                "tool_search",
                *[tool_name for tool_name in preferred if tool_name != "tool_search"],
            ]
        if attached_workspace_files and "file_read" in available_order:
            preferred = [
                "file_read",
                *[tool_name for tool_name in preferred if tool_name != "file_read"],
            ]
        for tool_name, keywords in self._FILETYPE_TOOL_KEYWORDS.items():
            if tool_name not in available_order:
                continue
            if self._matches_any_keyword(lowered, keywords):
                preferred.append(tool_name)
        if "repo_map" in available_order and self._matches_any_keyword(
            lowered,
            self._REPO_NAVIGATION_INTENT_KEYWORDS,
        ):
            preferred.append("repo_map")
        if "read_symbol" in available_order and self._matches_any_keyword(
            lowered,
            self._SYMBOL_LOOKUP_INTENT_KEYWORDS,
        ):
            preferred.append("read_symbol")
        if attachment_processing_request:
            for tool_name in ("exec_command", "execute_code", "execute_code_v2"):
                if tool_name in available_order:
                    preferred.append(tool_name)
        preferred = list(dict.fromkeys(preferred))

        selected: list[str] = []
        available = set(available_tool_names)
        for tool_name in preferred:
            if tool_name not in selected:
                selected.append(tool_name)
        for tool_name in available_tool_names:
            if tool_name in selected:
                continue
            selected.append(tool_name)

        preferred_set = set(preferred)
        selected.sort(
            key=lambda name: (
                0 if name in preferred_set else 1,
                self._matched_group_rank(name, matched_groups, tools_by_group),
                -self._capability_affinity_score(
                    capabilities=normalized_capabilities.get(name, {}),
                    matched_groups=matched_groups,
                    research_request=research_request,
                    citation_request=citation_request,
                    biomedical_request=biomedical_request,
                    recent_research_request=recent_research_request,
                    web_request=web_request,
                    workspace_request=workspace_request,
                ),
                self._TOOL_PRIORITY.get(name, 1000),
                available_order.get(name, 10_000),
            )
        )

        workspace_focus_request = (
            session_bound_workspace
            and (
                routed_workspace_request
                or (
                    explicit_workspace_request
                    and not web_request
                    and not research_request
                )
            )
        )
        open_world_focus_request = (
            session_bound_workspace
            and (research_request or web_request)
            and not explicit_workspace_request
        )
        non_workspace_attachment_request = (
            normalized_attachment_count > 0
            and normalized_workspace_attachment_count == 0
            and not session_bound_workspace
            and not workspace_request
        )
        filtered: list[str] = []
        risky_count = 0
        explicit_subagent_delegation = self._matches_any_keyword(
            lowered_primary_intent,
            self._EXPLICIT_SUBAGENT_DELEGATION_KEYWORDS,
        )
        for tool_name in selected:
            capabilities = normalized_capabilities.get(tool_name, {})
            domains = set(capabilities.get("domains", ()))
            core_workspace_write_tool = tool_name in self._CORE_WORKSPACE_WRITE_TOOLS
            if (
                normalized_routed_intent
                in {
                    "workspace_read",
                    "open_world_lookup",
                    "literature_research",
                    "tool_discovery",
                    "execution_or_process",
                }
                and core_workspace_write_tool
            ):
                continue
            if tool_name == "delegate_subagent_task" and not explicit_subagent_delegation:
                continue
            if (
                workspace_focus_request
                and bool(capabilities.get("open_world", False))
                and domains & {"web", "literature"}
            ):
                continue
            if routed_tool_discovery and tool_name in self._CORE_WORKSPACE_WRITE_TOOLS:
                continue
            if non_workspace_attachment_request and tool_name in self._CORE_WORKSPACE_READ_ONLY_TOOLS:
                continue
            if open_world_focus_request and tool_name in self._CORE_WORKSPACE_READ_ONLY_TOOLS:
                continue
            if tool_name in self._CONTEXTUAL_TOOLS and not self._matches_any_keyword(
                lowered,
                self._CONTEXT_KEYWORDS,
            ):
                continue
            if (
                read_only_file_request
                and tool_name in self._RISKY_TOOLS
                and (
                    tool_name in self._CORE_WORKSPACE_WRITE_TOOLS
                    or not (
                        attachment_processing_request
                        and tool_name in self._NON_MUTATING_EXECUTION_TOOLS
                    )
                )
            ):
                continue
            if (
                file_browse_request
                and tool_name == "exec_command"
                and not attachment_processing_request
            ):
                continue
            if effective_mode == "strict" and tool_name in self._STRICT_BLOCKED_TOOLS:
                continue
            if tool_name in self._RISKY_TOOLS:
                if risky_count >= risky_limit:
                    continue
                risky_count += 1
            filtered.append(tool_name)

        workspace_baseline = [
            tool_name
            for tool_name in self._CORE_WORKSPACE_READ_ONLY_TOOLS
            if session_bound_workspace
            and tool_name in available
            and (
                workspace_focus_request
                or file_browse_request
                or attached_workspace_files
                or attachment_processing_request
            )
        ]
        workspace_write_baseline = [
            tool_name
            for tool_name in self._CORE_WORKSPACE_WRITE_TOOLS
            if (session_bound_workspace or routed_workspace_write or attachment_mutation_request)
            and tool_name in available
        ]
        general_web_baseline = [
            tool_name
            for tool_name in ("web_search", "web_fetch", "get_current_time")
            if (
                session_bound_workspace
                and not explicit_workspace_request
                and tool_name in available
                and tool_name in filtered
            )
        ]
        final_tool_names = list(filtered[:limit])
        for tool_name in workspace_baseline:
            if tool_name not in final_tool_names:
                final_tool_names.append(tool_name)
        if routed_workspace_write or attachment_mutation_request:
            for tool_name in workspace_write_baseline:
                if tool_name not in final_tool_names:
                    final_tool_names.append(tool_name)
        # A workspace-bound session should not silently lose general web lookup.
        # Keep a small open-world baseline visible unless the user explicitly
        # asked for workspace-local inspection/manipulation.
        for tool_name in general_web_baseline:
            if tool_name not in final_tool_names:
                final_tool_names.append(tool_name)
        if self._should_expose_tool_search(
            available_tool_names=available_tool_names,
            discoverable_tool_names=filtered,
            visible_tool_names=final_tool_names,
        ) and "tool_search" not in final_tool_names:
            if (
                not session_bound_workspace
                and not routed_workspace_write
                and not attachment_mutation_request
                and limit > 0
                and len(final_tool_names) >= limit
            ):
                final_tool_names = [*final_tool_names[: limit - 1], "tool_search"]
            else:
                final_tool_names.append("tool_search")

        final_limit = max(limit, len(final_tool_names))
        discoverable_tool_names = list(filtered)
        for tool_name in workspace_write_baseline:
            if tool_name not in discoverable_tool_names:
                discoverable_tool_names.append(tool_name)
        if routed_workspace_write:
            workspace_write_obligation_source = intent_source or "routed_intent"
            workspace_write_obligation_confidence = intent_confidence
            workspace_write_obligation_rationale = (
                intent_rationale or "Routed workspace write intent."
            )
        elif attachment_mutation_request:
            workspace_write_obligation_source = "fallback_keyword"
            workspace_write_obligation_confidence = None
            workspace_write_obligation_rationale = (
                "Explicit workspace target and mutation keyword fallback."
            )
        else:
            workspace_write_obligation_source = "none"
            workspace_write_obligation_confidence = None
            workspace_write_obligation_rationale = None
        return ToolExposurePlan(
            tool_names=final_tool_names,
            matched_groups=matched_groups,
            limit=final_limit,
            discoverable_tool_names=discoverable_tool_names,
            workspace_bound=session_bound_workspace,
            attachment_count=normalized_attachment_count,
            intent_route=intent_route_metadata,
            diagnostics={
                "stage": "planner",
                "available_tool_count": len(available_tool_names),
                "filtered_tool_count": len(filtered),
                "final_tool_count": len(final_tool_names),
                "workspace_write_obligation": {
                    "required": routed_workspace_write or attachment_mutation_request,
                    "source": workspace_write_obligation_source,
                    "confidence": workspace_write_obligation_confidence,
                    "rationale": workspace_write_obligation_rationale,
                },
                "backend": self._backend_diagnostics(backend_info, metadata),
            },
        )

    @staticmethod
    def _normalize_routed_intent(routed_intent: ToolIntent | str | None) -> ToolIntent | None:
        return normalize_tool_intent_name(routed_intent)

    @staticmethod
    def _backend_diagnostics(backend_info: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "tool_call_mode",
            "tool_calling_protocol",
            "tool_calling_style",
            "tool_calling_blocked",
            "native_tool_calling_status",
            "fallback_validation_status",
        )
        return {
            "backend_type": getattr(backend_info, "backend_type", None),
            "provider": getattr(backend_info, "provider", None),
            "model": getattr(backend_info, "name", None),
            "metadata": {
                key: metadata.get(key)
                for key in keys
                if key in metadata
            },
        }

    @classmethod
    def _intent_route_metadata(
        cls,
        *,
        routed_intent: ToolIntent | None,
        intent_confidence: float | None,
        intent_source: str | None,
        intent_rationale: str | None,
    ) -> dict[str, Any] | None:
        normalized_routed_intent = cls._normalize_routed_intent(routed_intent)
        if (
            normalized_routed_intent is None
            and intent_confidence is None
            and not intent_source
            and not intent_rationale
        ):
            return None
        return {
            "intent": normalized_routed_intent or "ambiguous",
            "confidence": intent_confidence,
            "source": intent_source or "fallback_keyword",
            "rationale": intent_rationale or "",
        }

    @classmethod
    def _should_expose_tool_search(
        cls,
        *,
        available_tool_names: list[str],
        discoverable_tool_names: list[str],
        visible_tool_names: list[str],
    ) -> bool:
        if "tool_search" not in available_tool_names:
            return False
        visible = set(visible_tool_names)
        return any(
            tool_name != "tool_search" and tool_name not in visible
            for tool_name in discoverable_tool_names
        )

    def _build_tools_by_group(
        self,
        available_tool_names: list[str],
        tool_capabilities: dict[str, dict[str, Any]],
    ) -> dict[str, list[str]]:
        groups = {
            "web": [],
            "workspace": [],
            "literature": [],
            "memory": [],
            "mcp": [],
        }
        for tool_name in available_tool_names:
            domains = set(tool_capabilities.get(tool_name, {}).get("domains", ()))
            for group_name in groups:
                if group_name in domains or tool_name in self._tool_groups.get(group_name, []):
                    groups[group_name].append(tool_name)
        return groups

    def _matched_group_rank(
        self,
        tool_name: str,
        matched_groups: list[str],
        tools_by_group: dict[str, list[str]],
    ) -> int:
        for index, group_name in enumerate(matched_groups):
            if tool_name in tools_by_group[group_name]:
                return index
        return len(matched_groups)

    def _capability_affinity_score(
        self,
        *,
        capabilities: dict[str, Any],
        matched_groups: list[str],
        research_request: bool,
        citation_request: bool,
        biomedical_request: bool,
        recent_research_request: bool,
        web_request: bool,
        workspace_request: bool,
    ) -> int:
        domains = set(capabilities.get("domains", ()))
        retrieval_modes = set(capabilities.get("retrieval_modes", ()))
        preference_tags = set(capabilities.get("preference_tags", ()))

        score = 0
        if research_request:
            if "literature" in domains:
                score += 120
            if "web" in domains:
                score += 30
            if "scholarly_index" in preference_tags:
                score += 25
            if "paper_metadata" in preference_tags:
                score += 10
            if "search" in retrieval_modes:
                score += 10
            if "fetch" in retrieval_modes:
                score += 1
            if "crawl" in retrieval_modes:
                score -= 10
            if citation_request:
                if "citation_lookup" in preference_tags or "doi_lookup" in preference_tags:
                    score += 40
                if "bibliographic_metadata" in preference_tags:
                    score += 15
            if biomedical_request and "biomedical" in preference_tags:
                score += 40
            if recent_research_request and "recent_papers" in preference_tags:
                score += 15
            return score

        if "web" in matched_groups or web_request:
            if "web" in domains:
                score += 50
            if "search" in retrieval_modes:
                score += 15
            if "fetch" in retrieval_modes:
                score += 5
            if "crawl" in retrieval_modes:
                score += 3

        if "workspace" in matched_groups or workspace_request:
            if "workspace" in domains:
                score += 40

        if "memory" in matched_groups and "memory" in domains:
            score += 35
        if "mcp" in matched_groups and "mcp" in domains:
            score += 35
        return score

    def _is_research_request(self, lowered_message: str) -> bool:
        return (
            self._matches_any_keyword(lowered_message, self._RESEARCH_KEYWORDS)
            or self._matches_any_keyword(lowered_message, self._CITATION_KEYWORDS)
            or self._matches_any_keyword(lowered_message, self._BIOMEDICAL_KEYWORDS)
            or self._DOI_PATTERN.search(lowered_message) is not None
        )

    def _is_citation_request(self, lowered_message: str) -> bool:
        return (
            self._matches_any_keyword(lowered_message, self._CITATION_KEYWORDS)
            or self._DOI_PATTERN.search(lowered_message) is not None
        )

    @classmethod
    def _matches_any_keyword(cls, lowered_message: str, keywords: tuple[str, ...]) -> bool:
        return any(cls._matches_keyword(lowered_message, keyword) for keyword in keywords)

    @staticmethod
    def _matches_keyword(lowered_message: str, keyword: str) -> bool:
        if not keyword:
            return False
        if keyword[0].isalnum() and keyword[-1].isalnum():
            pattern = rf"(?<![a-z0-9_]){re.escape(keyword)}(?![a-z0-9_])"
            return re.search(pattern, lowered_message) is not None
        return keyword in lowered_message

    @staticmethod
    def _normalize_tool_capabilities(raw: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {
                "domains": (),
                "retrieval_modes": (),
                "preference_tags": (),
                "read_only": False,
                "destructive": False,
                "open_world": False,
            }
        return {
            "domains": ToolExposurePlanner._coerce_string_tuple(raw.get("domains")),
            "retrieval_modes": ToolExposurePlanner._coerce_string_tuple(raw.get("retrieval_modes")),
            "preference_tags": ToolExposurePlanner._coerce_string_tuple(raw.get("preference_tags")),
            "read_only": bool(raw.get("read_only", False)),
            "destructive": bool(raw.get("destructive", False)),
            "open_world": bool(raw.get("open_world", False)),
        }

    @staticmethod
    def _coerce_string_tuple(value: Any) -> tuple[str, ...]:
        if not isinstance(value, (list, tuple)):
            return ()
        items = [str(item).strip() for item in value]
        return tuple(item for item in items if item)
