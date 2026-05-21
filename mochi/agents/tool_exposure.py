"""Subset-first tool exposure planning."""

from __future__ import annotations

from dataclasses import dataclass

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
            "shell",
            "execute_code",
            "file_write",
            "file_edit",
            "process_stop",
            "mcp_call",
        }
    )
    _STRICT_BLOCKED_TOOLS: frozenset[str] = frozenset(
        {
            "shell",
            "execute_code",
            "process_stop",
            "mcp_call",
        }
    )
    _CONTEXTUAL_TOOLS: frozenset[str] = frozenset({"process_poll", "process_stop"})
    _AUTONOMY_LIMITS: dict[str, tuple[int, int]] = {
        "strict": (4, 2),
        "trusted_workspace": (6, 3),
        "auto_review": (8, 3),
        "high_autonomy": (10, 4),
    }
    _TOOL_PRIORITY: dict[str, int] = {
        "file_read": 10,
        "file_write": 20,
        "file_edit": 30,
        "shell": 40,
        "execute_code": 50,
        "process_poll": 60,
        "process_stop": 70,
        "memory_search": 80,
        "memory_save": 90,
        "web_search": 100,
        "web_fetch": 110,
        "get_current_time": 120,
        "calculator": 130,
    }

    _GROUP_KEYWORDS: dict[str, tuple[str, ...]] = {
        "web": (
            "today", "weather", "news", "latest", "search", "url", "http", "https",
            "今天", "天氣", "新聞", "最新", "查詢", "搜尋", "網址", "網頁",
        ),
        "workspace": (
            "code", "file", "test", "run", "debug", "repo", "project", "shell",
            "程式", "代碼", "檔案", "測試", "執行", "除錯", "專案", "倉庫",
        ),
        "literature": (
            "paper", "arxiv", "pubmed", "doi", "citation", "literature",
            "論文", "文獻", "引用", "doi",
        ),
        "memory": (
            "remember", "memory", "skill", "記住", "記憶", "技能",
        ),
        "mcp": (
            "mcp", "resource", "server", "資源", "伺服器",
        ),
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
    ) -> ToolExposurePlan:
        lowered = message.lower()
        matched_groups: list[str] = []
        for group_name in ("web", "workspace", "literature", "memory", "mcp"):
            if any(keyword in lowered for keyword in self._GROUP_KEYWORDS[group_name]):
                matched_groups.append(group_name)

        if not matched_groups:
            matched_groups.append("workspace" if session_bound_workspace else "web")

        selected: list[str] = []
        available = set(available_tool_names)
        for group_name in matched_groups:
            for tool_name in self._tool_groups.get(group_name, []):
                if tool_name in available and tool_name not in selected:
                    selected.append(tool_name)
        selected.sort(key=lambda name: self._TOOL_PRIORITY.get(name, 1000))

        base_limit = 6 if backend.get_model_info().backend_type in {"gguf", "safetensors"} else 10
        effective_mode = autonomy_mode or "trusted_workspace"
        mode_limit, risky_limit = self._AUTONOMY_LIMITS.get(effective_mode, (base_limit, 1))
        limit = min(base_limit, mode_limit)
        filtered: list[str] = []
        risky_count = 0
        for tool_name in selected:
            if tool_name in self._CONTEXTUAL_TOOLS and not any(
                keyword in lowered for keyword in ("background", "process", "poll", "tail", "stop")
            ):
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
