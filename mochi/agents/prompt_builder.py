# Inspired by openclaw/src/agents/pi-embedded-runner design pattern
"""Prompt 組裝器 — 將系統提示、技能、工具定義組合為 LLM 輸入。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mochi.backends.types import ToolSchema
    from mochi.learning.types import Skill


class PromptBuilder:
    """組裝傳遞給 LLM 的 System Prompt 和工具描述。

    負責將靜態系統提示、動態注入的技能指引、工具清單合併為一個
    結構清晰的字串，以最大化 LLM 的工具呼叫準確率。
    """

    def __init__(self, base_system_prompt: str) -> None:
        """初始化 PromptBuilder。

        Args:
            base_system_prompt: 基礎系統提示詞（來自 AgentConfig）。
        """
        self._base_prompt = base_system_prompt

    def build_system_prompt(
        self,
        skills_context: str | None = None,
        memory_context: str | None = None,
        base_prompt: str | None = None,
    ) -> str:
        """建構完整的系統提示詞。

        Args:
            skills_context: 從技能庫檢索到的相關技能描述（Markdown 格式）。
            memory_context: 從長期記憶檢索到的相關內容。
            base_prompt: 覆蓋基礎 system prompt。

        Returns:
            完整系統提示詞字串。
        """
        parts: list[str] = [base_prompt if base_prompt is not None else self._base_prompt]

        if memory_context:
            parts.append(
                "\n\n## Relevant Memory\n"
                "The following prior information may be relevant to the current task:\n"
                f"{memory_context}"
            )

        if skills_context:
            parts.append(
                "\n\n## Reusable Skill Guidance\n"
                "The following skills were loaded from Mochi's skill library. "
                "Use them as optional task-specific operating guidance:\n"
                f"{skills_context}"
            )

        return "\n".join(parts)

    def format_skills_context(
        self,
        skills: list[Skill] | list[dict[str, Any]],
        max_skills: int = 3,
    ) -> str:
        """將檢索到的 Skill 格式化為可注入 system prompt 的 Markdown。

        Args:
            skills: Skill dataclass/model 或 dict 列表。
            max_skills: 最多注入的技能數量。

        Returns:
            Markdown 格式技能內容；沒有技能時回傳空字串。
        """
        if not skills or max_skills <= 0:
            return ""

        lines: list[str] = []
        for index, skill in enumerate(skills[:max_skills], start=1):
            name = self._skill_value(skill, "name", "Unnamed skill")
            description = self._skill_value(skill, "description", "")
            preconditions = self._skill_value(skill, "preconditions", "")
            steps = self._skill_value(skill, "steps", [])
            tools_used = self._skill_value(skill, "tools_used", [])
            version = self._skill_value(skill, "version", 1)
            source_type = self._skill_value(skill, "source_type", "learned")
            source_path = self._skill_value(skill, "source_path", "")
            body = self._skill_value(skill, "body", "")

            lines.append(f"### {index}. {name}")
            lines.append(f"- **version**: {version}")
            lines.append(f"- **source_type**: {source_type}")
            if source_path:
                lines.append(f"- **source_path**: {source_path}")
            lines.append(f"- **description**: {description or '(none)'}")
            if body:
                lines.append("- **instructions**:")
                lines.append(str(body).strip())
            else:
                lines.append(f"- **preconditions**: {self._format_skill_field(preconditions)}")
                lines.append("- **steps**:")
                lines.extend(f"  {line}" for line in self._format_numbered_list(steps))
            lines.append(f"- **tools_used**: {self._format_skill_field(tools_used)}")
            lines.append("")

        return "\n".join(lines).strip()

    def format_tool_definitions(self, tools: list[ToolSchema]) -> str:
        """將工具列表格式化為 Markdown 描述（供 Tool Call Simulator 使用）。

        Args:
            tools: 工具 Schema 列表。

        Returns:
            Markdown 格式的工具清單字串。
        """
        if not tools:
            return ""

        lines: list[str] = ["## Available Tools\n"]
        for tool in tools:
            lines.append(f"### `{tool.name}`")
            lines.append(tool.description)
            lines.append("**Parameters**:")
            lines.append("```json")
            lines.append(json.dumps(tool.parameters, ensure_ascii=False, indent=2))
            lines.append("```\n")

        return "\n".join(lines)

    @staticmethod
    def _skill_value(skill: Skill | dict[str, Any], key: str, default: Any) -> Any:
        """從 Skill 物件或 dict 取得欄位值。"""
        if isinstance(skill, dict):
            return skill.get(key, default)
        return getattr(skill, key, default)

    @staticmethod
    def _format_skill_field(value: Any) -> str:
        """格式化 Skill 的純文字或列表欄位。"""
        if value is None or value == "":
            return "(none)"
        if isinstance(value, str):
            return value
        if isinstance(value, Sequence):
            items = [str(item) for item in value if str(item)]
            return ", ".join(items) if items else "(none)"
        return str(value)

    @staticmethod
    def _format_numbered_list(value: Any) -> list[str]:
        """格式化步驟列表，確保 Markdown 結構穩定。"""
        if value is None or value == "":
            return ["1. (none)"]
        if isinstance(value, str):
            return [f"1. {value}"]
        if isinstance(value, Sequence):
            items = [str(item) for item in value if str(item)]
            if not items:
                return ["1. (none)"]
            return [f"{index}. {item}" for index, item in enumerate(items, start=1)]
        return [f"1. {value}"]
