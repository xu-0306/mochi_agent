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

        limit = 6 if backend.get_model_info().backend_type in {"gguf", "safetensors"} else 10
        return ToolExposurePlan(
            tool_names=selected[:limit],
            matched_groups=matched_groups,
            limit=limit,
        )
