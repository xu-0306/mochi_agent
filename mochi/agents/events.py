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
    metadata: dict[str, Any] = field(default_factory=dict)
    """思考內容。"""


@dataclass
class StatusEvent:
    """Runtime progress/status diagnostics that should not be shown as model reasoning."""

    type: Literal["status"] = field(default="status", init=False)
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AssistantTruncatedEvent:
    """Explicit event emitted when assistant output hits a length limit."""

    type: Literal["assistant_truncated"] = field(default="assistant_truncated", init=False)
    content: str = ""
    finish_reason: str = "length"
    recovery_attempt: int = 0
    partial_output_chars: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubagentStartedEvent:
    """UI-safe delegated subagent lifecycle start event."""

    type: Literal["subagent_started"] = field(default="subagent_started", init=False)
    subagent_id: str = ""
    parent_type: str | None = None
    parent_id: str | None = None
    role_id: str | None = None
    title: str | None = None
    model_id: str | None = None
    prompt_preview: str | None = None
    status: str | None = None
    summary: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubagentPromptEvent:
    """UI-safe delegated subagent prompt event."""

    type: Literal["subagent_prompt"] = field(default="subagent_prompt", init=False)
    subagent_id: str = ""
    parent_type: str | None = None
    parent_id: str | None = None
    role_id: str | None = None
    title: str | None = None
    model_id: str | None = None
    system_prompt: str | None = None
    user_prompt: str | None = None
    prompt_preview: str | None = None
    status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubagentProgressEvent:
    """UI-safe delegated subagent progress event."""

    type: Literal["subagent_progress"] = field(default="subagent_progress", init=False)
    subagent_id: str = ""
    parent_type: str | None = None
    parent_id: str | None = None
    role_id: str | None = None
    title: str | None = None
    content: str = ""
    status: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SubagentCompletedEvent:
    """UI-safe delegated subagent completion event."""

    type: Literal["subagent_completed"] = field(default="subagent_completed", init=False)
    subagent_id: str = ""
    parent_type: str | None = None
    parent_id: str | None = None
    role_id: str | None = None
    title: str | None = None
    model_id: str | None = None
    status: str | None = None
    summary: str | None = None
    content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallCreatedEvent:
    """Explicit event emitted when a tool call is parsed and ready to execute."""

    type: Literal["tool_call_created"] = field(default="tool_call_created", init=False)
    call_id: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallCompletedEvent:
    """Explicit event emitted when a parsed tool call has completed execution."""

    type: Literal["tool_call_completed"] = field(default="tool_call_completed", init=False)
    call_id: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GoalStateChangedEvent:
    """Explicit event emitted when a goal lifecycle status changes."""

    type: Literal["goal_state_changed"] = field(default="goal_state_changed", init=False)
    goal_id: str = ""
    previous_status: str | None = None
    status: str = ""
    attempt_id: str | None = None
    agent_run_id: str | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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

    metadata: dict[str, Any] = field(default_factory=dict)
    """工具附加元資料。"""


@dataclass
class FinalAnswerEvent:
    """Agent 最終回答。"""

    type: Literal["final_answer"] = field(default="final_answer", init=False)
    content: str = ""
    """完整回覆文字。"""

    trajectory_id: str | None = None
    """關聯的軌跡 ID（供學習系統使用）。"""

    input_tokens: int = 0
    """輸入 token 數量。"""

    output_tokens: int = 0
    """輸出 token 數量。"""

    generation_time_ms: float = 0.0
    """推理耗時（毫秒）。"""

    finish_reason: str = "stop"
    """停止原因。"""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional diagnostics."""


@dataclass
class ErrorEvent:
    """Agent 執行過程中的錯誤。"""

    type: Literal["error"] = field(default="error", init=False)
    message: str = ""
    """錯誤描述。"""

    code: str = "AGENT_ERROR"
    metadata: dict[str, Any] = field(default_factory=dict)
    """錯誤代碼。"""


# 聯合型別，涵蓋所有事件類型
AgentEvent = (
    TextChunkEvent
    | ThinkingEvent
    | StatusEvent
    | AssistantTruncatedEvent
    | SubagentStartedEvent
    | SubagentPromptEvent
    | SubagentProgressEvent
    | SubagentCompletedEvent
    | ToolCallCreatedEvent
    | ToolCallCompletedEvent
    | GoalStateChangedEvent
    | ToolCallRequestEvent
    | ToolCallResultEvent
    | FinalAnswerEvent
    | ErrorEvent
)
