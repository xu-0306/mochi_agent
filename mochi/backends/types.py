"""後端共用型別定義。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# 角色類型
Role = Literal["system", "user", "assistant", "tool"]


@dataclass
class ToolCall:
    """LLM 回覆中的工具呼叫請求。"""

    id: str
    """工具呼叫唯一 ID。"""

    name: str
    """工具名稱。"""

    arguments: dict[str, Any]
    """工具參數（JSON 物件）。"""


@dataclass
class Message:
    """對話訊息。"""

    role: Role
    """發言角色。"""

    content: str
    """訊息內容。"""

    tool_calls: list[ToolCall] = field(default_factory=list)
    """若 role=assistant 且含工具呼叫則填入此欄。"""

    tool_call_id: str | None = None
    """若 role=tool，對應的 ToolCall.id。"""

    name: str | None = None
    """工具名稱（role=tool 時使用）。"""

    def to_dict(self) -> dict[str, Any]:
        """轉為 OpenAI chat message 格式字典。"""
        d: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": str(tc.arguments)},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class ToolSchema:
    """工具的 JSON Schema 描述（OpenAI function calling 格式）。"""

    name: str
    """工具名稱。"""

    description: str
    """工具用途描述。"""

    parameters: dict[str, Any]
    """JSON Schema 格式的參數定義。"""

    def to_dict(self) -> dict[str, Any]:
        """轉為 OpenAI tools 格式字典。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class GenerationResult:
    """非串流模式的完整生成結果。"""

    content: str
    """生成的文字內容。"""

    tool_calls: list[ToolCall] = field(default_factory=list)
    """工具呼叫請求列表。"""

    input_tokens: int = 0
    """輸入 token 數量。"""

    output_tokens: int = 0
    """輸出 token 數量。"""

    model: str = ""
    """實際使用的模型名稱。"""

    finish_reason: str = "stop"
    """停止原因（stop / tool_calls / length）。"""


@dataclass
class StreamChunk:
    """串流模式的單一 chunk。"""

    delta: str = ""
    """本次增量文字。"""

    tool_call_delta: ToolCall | None = None
    """本次增量工具呼叫（部分填充）。"""

    is_final: bool = False
    """是否為最後一個 chunk。"""

    finish_reason: str | None = None
    """最終 chunk 的停止原因。"""


@dataclass
class ModelInfo:
    """模型基本資訊。"""

    name: str
    """模型名稱。"""

    backend_type: str
    """後端類型（ollama / gguf / safetensors / openai_compat）。"""

    context_length: int = 4096
    """最大上下文視窗（tokens）。"""

    supports_tool_calling: bool = False
    """是否原生支援 function calling。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """後端自訂元資料。"""
