"""Core tool contracts."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from mochi.runtime.cancellation import (
    CancellationCapability,
    CancellationCapabilityRegistry,
    CommitFence,
    CommitFenceDecision,
    CommitOutcome,
    CommitState,
)


@dataclass
class ToolResult:
    """Normalized tool result envelope."""

    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    retryable: bool = False
    suggestion: str | None = None


@dataclass
class FileReadState:
    """Cached file read snapshot used by write guards."""

    path: str
    content: str
    encoding: str
    mtime_ns: int | None
    size_bytes: int
    partial: bool = False


@dataclass
class ToolCancellationResult:
    """Normalized cancellation outcome for one active tool invocation."""

    cancelled: bool
    reason: str | None = None
    tool_call_id: str | None = None
    tool_name: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


TaskCancellationOutcome = Literal["cancelled", "completed", "pending"]


async def cancel_asyncio_task(
    task: asyncio.Task[Any] | None,
) -> TaskCancellationOutcome:
    """Best-effort task cancellation with a truthful immediate outcome."""
    if task is None:
        return "completed"
    if task.done():
        return "cancelled" if task.cancelled() else "completed"
    task.cancel()
    await asyncio.sleep(0)
    if not task.done():
        return "pending"
    return "cancelled" if task.cancelled() else "completed"


@dataclass
class RunCancellationResult:
    """Normalized run-level cancellation outcome."""

    cancelled: bool
    state: Literal["cancelled", "completed", "pending"]
    boundary: Literal["generation", "tool"] | None = None
    reason: str | None = None
    tool_result: ToolCancellationResult | None = None
    capability: str | None = None
    durable_outcome: str | None = None


class RunCancellationContext:
    """Coordinate truthful run-level and tool-level cancellation boundaries."""

    def __init__(self, *, run_id: str) -> None:
        self.run_id = str(run_id or "").strip() or "run"
        self._lock = asyncio.Lock()
        self._state: Literal["running", "cancelling", "cancelled", "completed"] = "running"
        self._cancel_requested = False
        self._cancel_confirmed = False
        self._active_tool_controller: ActiveToolController | None = None
        self._generation_cancel_callback: Callable[[], Awaitable[TaskCancellationOutcome]] | None = None
        self._capabilities = CancellationCapabilityRegistry(
            {
                "generation": CancellationCapability.IMMEDIATE,
                "tool": CancellationCapability.DEFERRED,
            }
        )
        self._commit_fence = CommitFence()

    async def bind_active_tool_controller(
        self,
        controller: ActiveToolController | None,
    ) -> None:
        async with self._lock:
            self._active_tool_controller = controller

    async def bind_generation_cancel_callback(
        self,
        callback: Callable[[], Awaitable[TaskCancellationOutcome]] | None,
    ) -> None:
        async with self._lock:
            self._generation_cancel_callback = callback

    async def mark_completed(self) -> None:
        async with self._lock:
            if self._state != "cancelled" and self._commit_fence.state is not CommitState.CANCELLATION_REQUESTED:
                self._state = "completed"

    async def mark_cancelled(self) -> None:
        decision = self._commit_fence.request_cancellation(safe_point="run_cancelled")
        async with self._lock:
            self._cancel_requested = True
            if decision.durable_outcome is CommitOutcome.ALREADY_COMMITTED:
                self._state = "completed"
            else:
                self._cancel_confirmed = True
                self._state = "cancelled"

    def resolve_cancellation_capability(
        self,
        target: str,
        *,
        safe_point: str,
    ) -> CancellationCapability:
        """Expose the frozen capability classification to orchestration code."""

        return self._capabilities.resolve(target, safe_point=safe_point).capability

    async def try_commit_durable_effect(self, *, safe_point: str) -> CommitFenceDecision:
        """Cross the one-way durable boundary unless cancellation arrived first."""

        decision = self._commit_fence.try_commit(safe_point=safe_point)
        async with self._lock:
            if decision.durable_outcome in {CommitOutcome.COMMITTED, CommitOutcome.ALREADY_COMMITTED}:
                self._state = "completed"
                self._cancel_confirmed = False
        return decision

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            controller = self._active_tool_controller
            state = self._state
            cancel_requested = self._cancel_requested
            cancel_confirmed = self._cancel_confirmed
            generation_bound = self._generation_cancel_callback is not None
        tool_snapshot = await controller.snapshot() if controller is not None else None
        return {
            "run_id": self.run_id,
            "state": state,
            "cancel_requested": cancel_requested,
            "cancel_confirmed": cancel_confirmed,
            "generation_bound": generation_bound,
            "active_tool": tool_snapshot,
            "commit_state": self._commit_fence.state.value,
        }

    async def request_generation_cancel(self) -> RunCancellationResult:
        capability = self.resolve_cancellation_capability(
            "generation",
            safe_point="generation_cancel_requested",
        )
        if capability is CancellationCapability.UNSUPPORTED:
            return RunCancellationResult(
                cancelled=False,
                state="pending",
                boundary="generation",
                reason="generation_cancellation_unsupported",
                capability=capability.value,
            )
        async with self._lock:
            if self._state == "completed":
                return RunCancellationResult(
                    cancelled=False,
                    state="completed",
                    boundary="generation",
                    capability=capability.value,
                    durable_outcome=(
                        CommitOutcome.ALREADY_COMMITTED.value
                        if self._commit_fence.state is CommitState.COMMITTED
                        else None
                    ),
                )
            if self._state == "cancelled":
                return RunCancellationResult(
                    cancelled=True,
                    state="cancelled",
                    boundary="generation",
                    capability=capability.value,
                    durable_outcome=CommitOutcome.CANCELLED_PRE_COMMIT.value,
                )
            self._cancel_requested = True
            self._state = "cancelling"
            cancel_callback = self._generation_cancel_callback
        if cancel_callback is None:
            return RunCancellationResult(
                cancelled=False,
                state="pending",
                boundary="generation",
                reason="generation_not_bound",
                capability=capability.value,
            )
        outcome = await cancel_callback()
        if outcome == "cancelled":
            await self.mark_cancelled()
            return RunCancellationResult(
                cancelled=True,
                state="cancelled",
                boundary="generation",
                capability=capability.value,
                durable_outcome=CommitOutcome.CANCELLED_PRE_COMMIT.value,
            )
        if outcome == "completed":
            await self.mark_completed()
            return RunCancellationResult(
                cancelled=False,
                state="completed",
                boundary="generation",
                capability=capability.value,
            )
        return RunCancellationResult(
            cancelled=False,
            state="pending",
            boundary="generation",
            reason="generation_in_progress",
            capability=capability.value,
        )

    async def request_active_tool_cancel(self) -> RunCancellationResult:
        capability = self.resolve_cancellation_capability(
            "tool",
            safe_point="tool_cancel_requested",
        )
        if capability is CancellationCapability.UNSUPPORTED:
            return RunCancellationResult(
                cancelled=False,
                state="pending",
                boundary="tool",
                reason="tool_cancellation_unsupported",
                capability=capability.value,
            )
        async with self._lock:
            controller = self._active_tool_controller
        if controller is None:
            return RunCancellationResult(
                cancelled=False,
                state="pending",
                boundary="tool",
                reason="no_active_tool",
                capability=capability.value,
            )
        tool_result = await controller.request_cancel()
        if tool_result.cancelled:
            return RunCancellationResult(
                cancelled=True,
                state="cancelled",
                boundary="tool",
                reason=tool_result.reason,
                tool_result=tool_result,
                capability=capability.value,
            )
        return RunCancellationResult(
            cancelled=False,
            state="pending",
            boundary="tool",
            reason=tool_result.reason or "tool_in_progress",
            tool_result=tool_result,
            capability=capability.value,
        )

    async def request_run_cancel(self) -> RunCancellationResult:
        async with self._lock:
            if self._state not in {"completed", "cancelled"}:
                self._cancel_requested = True
                self._state = "cancelling"
        snapshot = await self.snapshot()
        state = str(snapshot.get("state") or "running")
        if state == "completed":
            return RunCancellationResult(
                cancelled=False,
                state="completed",
                durable_outcome=CommitOutcome.ALREADY_COMMITTED.value,
            )
        if state == "cancelled":
            return RunCancellationResult(
                cancelled=True,
                state="cancelled",
                durable_outcome=CommitOutcome.CANCELLED_PRE_COMMIT.value,
            )

        fence_decision = self._commit_fence.request_cancellation(
            safe_point="run_cancel_requested"
        )
        if fence_decision.durable_outcome is CommitOutcome.ALREADY_COMMITTED:
            await self.mark_completed()
            return RunCancellationResult(
                cancelled=False,
                state="completed",
                reason="already_committed",
                durable_outcome=fence_decision.durable_outcome.value,
            )

        active_tool = snapshot.get("active_tool")
        if isinstance(active_tool, Mapping) and bool(active_tool.get("active")):
            tool_result = await self.request_active_tool_cancel()
            if not tool_result.cancelled:
                return tool_result

        generation_result = await self.request_generation_cancel()
        if generation_result.cancelled:
            return generation_result
        if generation_result.state == "completed":
            if isinstance(active_tool, Mapping) and bool(active_tool.get("active")):
                return RunCancellationResult(
                    cancelled=False,
                    state="completed",
                    boundary="tool",
                    reason="run_completed_before_cancellation",
                )
            return generation_result
        return generation_result


class ActiveToolController:
    """Track the currently running tool and coordinate live cancellation/apply hooks."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tool_call_id: str | None = None
        self._tool_name: str | None = None
        self._session_id: str | None = None
        self._active = False
        self._cancellable = False
        self._cancel_requested = False
        self._cancel_callback: Callable[[], Awaitable[ToolCancellationResult]] | None = None
        self._pending_post_tool_messages: list[dict[str, Any]] = []
        self._applied_post_tool_messages: list[dict[str, Any]] = []

    async def activate_tool(
        self,
        *,
        tool_call_id: str,
        tool_name: str,
        cancellable: bool = False,
    ) -> None:
        async with self._lock:
            self._tool_call_id = str(tool_call_id or "").strip() or None
            self._tool_name = str(tool_name or "").strip() or None
            self._session_id = None
            self._active = True
            self._cancellable = bool(cancellable)
            self._cancel_requested = False
            self._cancel_callback = None

    async def bind_cancel_callback(
        self,
        *,
        session_id: str | None,
        callback: Callable[[], Awaitable[ToolCancellationResult]],
    ) -> None:
        async with self._lock:
            if not self._active:
                return
            self._session_id = str(session_id or "").strip() or None
            self._cancel_callback = callback
            self._cancellable = True

    async def finish_tool(self) -> None:
        async with self._lock:
            self._tool_call_id = None
            self._tool_name = None
            self._session_id = None
            self._active = False
            self._cancellable = False
            self._cancel_requested = False
            self._cancel_callback = None

    async def snapshot(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "active": self._active,
                "tool_call_id": self._tool_call_id,
                "tool_name": self._tool_name,
                "session_id": self._session_id,
                "cancellable": self._cancellable and self._cancel_callback is not None,
                "cancel_requested": self._cancel_requested,
            }

    async def request_cancel(self) -> ToolCancellationResult:
        async with self._lock:
            if not self._active:
                return ToolCancellationResult(cancelled=False, reason="no_active_tool")
            tool_call_id = self._tool_call_id
            tool_name = self._tool_name
            session_id = self._session_id
            if not self._cancellable or self._cancel_callback is None:
                return ToolCancellationResult(
                    cancelled=False,
                    reason="tool_in_progress",
                    tool_call_id=tool_call_id,
                    tool_name=tool_name,
                    session_id=session_id,
                )
            self._cancel_requested = True
            cancel_callback = self._cancel_callback

        result = await cancel_callback()
        if result.tool_call_id is None:
            result.tool_call_id = tool_call_id
        if result.tool_name is None:
            result.tool_name = tool_name
        if result.session_id is None:
            result.session_id = session_id
        return result

    async def queue_post_tool_message(self, message: Mapping[str, Any]) -> None:
        payload = {str(key): value for key, value in dict(message).items()}
        async with self._lock:
            self._pending_post_tool_messages.append(payload)

    async def consume_post_tool_messages(self) -> list[dict[str, Any]]:
        async with self._lock:
            if not self._pending_post_tool_messages:
                return []
            messages = [dict(item) for item in self._pending_post_tool_messages]
            self._pending_post_tool_messages.clear()
            self._applied_post_tool_messages.extend(messages)
            return messages

    async def drain_applied_post_tool_messages(self) -> list[dict[str, Any]]:
        async with self._lock:
            if not self._applied_post_tool_messages:
                return []
            messages = [dict(item) for item in self._applied_post_tool_messages]
            self._applied_post_tool_messages.clear()
            return messages


@dataclass
class ToolExecutionContext:
    """Shared execution context for one tool run."""

    workspace_dir: str | None = None
    session_id: str | None = None
    project_workspace: str | None = None
    task_sandbox_dir: str | None = None
    permission_policy: dict[str, Any] = field(default_factory=dict)
    read_state_cache: dict[str, FileReadState] = field(default_factory=dict)
    tool_result_store_dir: str | None = None
    tool_result_references: dict[str, dict[str, Any]] = field(default_factory=dict)
    transport_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    progress_callback: Any | None = None
    cancellation_requested: bool = False
    active_tool_controller: ActiveToolController | None = None
    client_timezone: str | None = None


class BaseTool(ABC):
    """Base class for every tool exposed to the agent."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable tool name used by model tool calling."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """LLM-facing tool description."""
        ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema describing tool arguments."""
        ...

    @property
    def requires_approval(self) -> bool:
        return False

    @property
    def is_read_only(self) -> bool:
        return False

    @property
    def is_destructive(self) -> bool:
        return False

    @property
    def is_concurrency_safe(self) -> bool:
        return False

    @property
    def is_open_world(self) -> bool:
        return False

    @property
    def is_cancellable(self) -> bool:
        return False

    @property
    def supports_timeline_side_effect_boundary(self) -> bool:
        """Whether this tool records its durable effect boundary before acting.

        Ordinary Chat timeline dispatch relies on this explicit contract instead
        of treating a tool name as proof that the implementation is safe.
        """
        return False

    @property
    def supports_timeline_approval_revocation(self) -> bool:
        """Whether post-precommit approvals can be durably revoked."""
        return False

    @property
    def timeline_approval_mode(self) -> str:
        """Declare how a side-effecting tool participates in Chat approvals.

        ``none`` means the tool contract never creates a durable approval.
        ``continuable`` means an ordinary-Chat pending approval preserves the
        precommitted operation for the server-owned continuation. ``revocable``
        is reserved for tools that can prove an unexpected post-precommit
        approval was superseded before any effect is possible.
        """
        return "none"

    def revoke_timeline_approval(self, approval_id: str, *, reason: str) -> bool:
        """Durably prevent a post-precommit approval from being consumed."""
        del approval_id, reason
        return False

    def validates_timeline_approval_binding(
        self,
        approval_id: str,
        *,
        operation_id: str,
        arguments_digest: str,
        call_id: str,
    ) -> bool:
        """Confirm a pending approval is durable and bound to one timeline call."""
        del approval_id, operation_id, arguments_digest, call_id
        return False

    @property
    def search_hint(self) -> str | None:
        return None

    @property
    def allow_plain_text_result_for_model(self) -> bool:
        return False

    @property
    def tool_capabilities(self) -> dict[str, Any]:
        """Internal planner/search metadata for capability-aware routing."""
        return {
            "domains": [],
            "retrieval_modes": [],
            "preference_tags": [],
            "read_only": self.is_read_only,
            "destructive": self.is_destructive,
            "open_world": self.is_open_world,
        }

    def validate_input(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult | None:
        del arguments, context
        return None

    def check_permissions(
        self,
        arguments: dict[str, Any],
        context: ToolExecutionContext | None = None,
    ) -> ToolResult | None:
        del arguments, context
        return None

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a ToolResult."""
        ...

    def format_result_for_model(
        self,
        result: ToolResult,
        *,
        max_chars: int = 2000,
    ) -> str:
        """Serialize tool output for model reinjection."""
        if self._should_preserve_plain_text_result(result=result, max_chars=max_chars):
            return str(result.output)

        if result.error:
            payload: dict[str, Any] = {"ok": False, "error": result.error}
        else:
            payload = {"ok": True, "output": result.output}

        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        if len(serialized) <= max_chars:
            return serialized

        if result.error:
            compact_payload = {
                "ok": False,
                "error": self._truncate_text(result.error, max_chars // 2),
                "truncated": True,
                "original_length": len(result.error),
            }
        elif isinstance(result.output, str):
            compact_payload = {
                "ok": True,
                "output": self._truncate_text(result.output, max_chars // 2),
                "truncated": True,
                "original_length": len(result.output),
            }
        else:
            compact_payload = {
                "ok": True,
                "output_preview": self._truncate_text(serialized, max_chars // 2),
                "truncated": True,
                "original_length": len(serialized),
            }
        return json.dumps(compact_payload, ensure_ascii=False, default=str)

    def summarize_result_for_ui(self, result: ToolResult) -> str:
        """Return a short UI summary."""
        if result.error:
            return result.error
        if isinstance(result.output, str):
            return result.output
        return json.dumps(result.output, ensure_ascii=False, default=str)

    def to_schema_dict(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        suffix = "...[truncated]"
        if max_chars <= len(suffix):
            return suffix[:max_chars]
        return value[: max_chars - len(suffix)] + suffix

    def _should_preserve_plain_text_result(
        self,
        *,
        result: ToolResult,
        max_chars: int,
    ) -> bool:
        output = result.output
        if result.error is not None or not isinstance(output, str):
            return False
        if not self.is_read_only or not self.allow_plain_text_result_for_model:
            return False
        return not (not output.strip() or len(output) > max_chars)
