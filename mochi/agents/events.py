"""AgentEvent 型別定義 — Agent 推理過程中產生的事件流。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class TextChunkEvent:
    """LLM 串流輸出的文字片段。"""

    type: Literal["text_chunk"] = field(default="text_chunk", init=False)
    content: str = ""
    """本次增量文字。"""


@dataclass
class ThinkingEvent:
    """Agent 思考過程（CoT 步驟）。"""

    type: Literal["thinking"] = field(default="thinking", init=False)
    content: str = ""
    """思考內容。"""


@dataclass
class ToolCallRequestEvent:
    """Agent 請求呼叫工具。"""

    type: Literal["tool_call_request"] = field(default="tool_call_request", init=False)
    call_id: str = ""
    """工具呼叫唯一 ID。"""

    tool_name: str = ""
    """工具名稱。"""

    arguments: dict[str, Any] = field(default_factory=dict)
    """工具參數。"""


@dataclass
class ToolCallResultEvent:
    """工具執行完成的結果。"""

    type: Literal["tool_call_result"] = field(default="tool_call_result", init=False)
    call_id: str = ""
    """對應 ToolCallRequestEvent.call_id。"""

    tool_name: str = ""
    """工具名稱。"""

    result: Any = None
    """工具執行結果。"""

    error: str | None = None
    """若執行失敗，此欄包含錯誤訊息。"""


@dataclass
class FinalAnswerEvent:
    """Agent 最終回答。"""

    type: Literal["final_answer"] = field(default="final_answer", init=False)
    content: str = ""
    """完整回覆文字。"""

    trajectory_id: str | None = None
    """關聯的軌跡 ID（供學習系統使用）。"""


@dataclass
class ErrorEvent:
    """Agent 執行過程中的錯誤。"""

    type: Literal["error"] = field(default="error", init=False)
    message: str = ""
    """錯誤描述。"""

    code: str = "AGENT_ERROR"
    """錯誤代碼。"""


# 聯合型別，涵蓋所有事件類型
AgentEvent = (
    TextChunkEvent
    | ThinkingEvent
    | ToolCallRequestEvent
    | ToolCallResultEvent
    | FinalAnswerEvent
    | ErrorEvent
)
