"""BaseTool 抽象基類 — 所有工具的統一介面。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolResult:
    """工具執行結果。"""

    output: Any = None
    """執行輸出（成功時填入）。"""

    error: str | None = None
    """錯誤訊息（失敗時填入）。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """附加元資料（如執行時間、資源用量）。"""


class BaseTool(ABC):
    """工具抽象基類。

    每個工具檔案 export 一個繼承此類的類，ToolRegistry 會自動發現並注冊。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名稱（英文，用於 LLM tool calling）。"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具用途描述（給 LLM 看的說明）。"""
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema 格式的參數定義。"""
        ...

    @property
    def requires_approval(self) -> bool:
        """此工具是否需要使用者確認才能執行。預設不需要。"""
        return False

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """執行工具。

        Args:
            **kwargs: 對應 parameters_schema 中定義的參數。

        Returns:
            ToolResult 執行結果。
        """
        ...

    def to_schema_dict(self) -> dict[str, Any]:
        """轉為 OpenAI function calling 格式的 schema 字典。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }
