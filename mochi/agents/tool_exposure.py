"""Subset-first tool exposure planning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mochi.backends.base import BaseLLMBackend


@dataclass(frozen=True)
class ToolExposurePlan:
    """Selected tool names for one turn."""

    tool_names: list[str]
    matched_groups: list[str]
    limit: int


class ToolExposurePlanner:
    """Select a smaller, intent-matched tool subset for one turn."""

    _RISKY_TOOLS: frozenset[str] = frozenset(
        {
            "exec_command",
            "shell",
            "execute_code",
            "execute_code_v2",
            "write_stdin",
            "kill_session",
            "file_write",
            "file_edit",
            "process_stop",
            "mcp_call",
            "delegate_subagent_task",
        }
    )
    _STRICT_BLOCKED_TOOLS: frozenset[str] = frozenset(
        {
            "exec_command",
            "shell",
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
        "file_read": 10,
        "csv_read": 11,
        "pdf_read": 12,
        "docx_read": 13,
        "notebook_read": 14,
        "tool_search": 15,
        "delegate_subagent_task": 18,
        "file_write": 20,
        "file_edit": 30,
        "exec_command": 40,
        "execute_code": 50,
        "execute_code_v2": 55,
        "read_session": 60,
        "write_stdin": 70,
        "list_sessions": 80,
        "kill_session": 90,
        "process_poll": 100,
        "process_stop": 110,
        "shell": 120,
        "memory_search": 130,
        "memory_save": 140,
        "web_search": 150,
        "web_fetch": 160,
        "web_crawl": 165,
        "get_current_time": 170,
        "calculator": 180,
    }
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
            "find",
            "search",
            "grep",
            "glob",
            "todo",
            "test",
            "run",
            "debug",
            "repo",
            "project",
            "workspace",
            "shell",
            "exec",
            "session",
            "stdin",
            "subagent",
            "delegate",
            "background task",
            "controlled execution",
            "子代理",
            "子agent",
            "背景任務",
            "受控執行",
        ),
        "literature": (
            "paper",
            "arxiv",
            "pubmed",
            "doi",
            "citation",
            "literature",
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
    _FILETYPE_TOOL_KEYWORDS: dict[str, tuple[str, ...]] = {
        "pdf_read": ("pdf",),
        "csv_read": ("csv", "tsv", "spreadsheet"),
        "docx_read": ("docx", "word document", ".docx"),
        "notebook_read": ("notebook", "ipynb", "jupyter"),
    }
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
    }

    def __init__(self, *, tool_groups: dict[str, list[str]]) -> None:
        self._tool_groups = tool_groups

    def plan(
        self,
        *,
        message: str,
        available_tool_names: list[str],
        backend: BaseLLMBackend,
        session_bound_workspace: bool,
        autonomy_mode: str | None = None,
        preferred_tool_names: list[str] | None = None,
        tool_mode: Literal["disabled", "auto", "required"] = "auto",
    ) -> ToolExposurePlan:
        if tool_mode == "disabled":
            return ToolExposurePlan(
                tool_names=[],
                matched_groups=[],
                limit=0,
            )
        lowered = message.lower()
        attached_workspace_files = any(
            marker in lowered for marker in self._ATTACHED_WORKSPACE_FILE_MARKERS
        )
        read_only_file_request = attached_workspace_files and any(
            keyword in lowered for keyword in self._READ_ONLY_FILE_INTENT_KEYWORDS
        )
        matched_groups: list[str] = []
        for group_name in ("web", "workspace", "literature", "memory", "mcp"):
            if any(keyword in lowered for keyword in self._GROUP_KEYWORDS[group_name]):
                matched_groups.append(group_name)

        if not matched_groups:
            matched_groups.append("workspace" if session_bound_workspace else "web")

        available_order = {name: index for index, name in enumerate(available_tool_names)}
        grouped_tool_names = {
            tool_name
            for tool_names in self._tool_groups.values()
            for tool_name in tool_names
        }
        preferred = [
            tool_name
            for tool_name in (preferred_tool_names or [])
            if tool_name in available_order
        ]
        if "tool_search" in available_order and any(
            keyword in lowered for keyword in self._TOOL_DISCOVERY_KEYWORDS
        ):
            preferred = ["tool_search", *[tool_name for tool_name in preferred if tool_name != "tool_search"]]
        if attached_workspace_files and "file_read" in available_order:
            preferred = ["file_read", *[tool_name for tool_name in preferred if tool_name != "file_read"]]
        for tool_name, keywords in self._FILETYPE_TOOL_KEYWORDS.items():
            if tool_name not in available_order:
                continue
            if any(keyword in lowered for keyword in keywords):
                preferred.append(tool_name)
        preferred = list(dict.fromkeys(preferred))

        selected: list[str] = []
        available = set(available_tool_names)
        for tool_name in preferred:
            if tool_name not in selected:
                selected.append(tool_name)
        for group_name in matched_groups:
            for tool_name in self._tool_groups.get(group_name, []):
                if tool_name in available and tool_name not in selected:
                    selected.append(tool_name)
        for tool_name in available_tool_names:
            if tool_name in selected or tool_name in grouped_tool_names:
                continue
            selected.append(tool_name)

        preferred_set = set(preferred)
        selected.sort(
            key=lambda name: (
                0 if name in preferred_set else 1,
                self._TOOL_PRIORITY.get(name, 1000),
                available_order.get(name, 10_000),
            )
        )

        base_limit = 6 if backend.get_model_info().backend_type in {"gguf", "safetensors"} else 10
        effective_mode = autonomy_mode or "trusted_workspace"
        mode_limit, risky_limit = self._AUTONOMY_LIMITS.get(effective_mode, (base_limit, 1))
        limit = min(base_limit, mode_limit)
        filtered: list[str] = []
        risky_count = 0
        for tool_name in selected:
            if not self._tool_matches_message(tool_name, lowered):
                continue
            if tool_name in self._CONTEXTUAL_TOOLS and not any(
                keyword in lowered for keyword in self._CONTEXT_KEYWORDS
            ):
                continue
            if read_only_file_request and tool_name in self._RISKY_TOOLS:
                continue
            if effective_mode == "strict" and tool_name in self._STRICT_BLOCKED_TOOLS:
                continue
            if tool_name in self._RISKY_TOOLS:
                if risky_count >= risky_limit:
                    continue
                risky_count += 1
            filtered.append(tool_name)

        return ToolExposurePlan(
            tool_names=filtered[:limit],
            matched_groups=matched_groups,
            limit=limit,
        )

    def _tool_matches_message(self, tool_name: str, lowered_message: str) -> bool:
        keywords = self._INTENT_REQUIRED_TOOL_KEYWORDS.get(tool_name)
        if not keywords:
            return True
        return any(keyword in lowered_message for keyword in keywords)
