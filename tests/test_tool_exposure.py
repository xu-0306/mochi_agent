from __future__ import annotations

from collections.abc import AsyncIterator

from mochi.agents.tool_exposure import ToolExposurePlanner
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, StreamChunk, ToolSchema


class _FakeBackend(BaseLLMBackend):
    def __init__(self, backend_type: str = "openai_compat") -> None:
        self._backend_type = backend_type

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        stream: bool = False,
    ) -> GenerationResult | AsyncIterator[StreamChunk]:
        del messages, tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        return GenerationResult(content="")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="fake", backend_type=self._backend_type)

    async def health_check(self) -> bool:
        return True


def test_tool_exposure_strict_mode_filters_risky_tools() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "shell", "execute_code", "file_write"],
        }
    )
    plan = planner.plan(
        message="run shell command in project",
        available_tool_names=["file_read", "shell", "execute_code", "file_write"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="strict",
    )
    assert plan.limit == 4
    assert "shell" not in plan.tool_names
    assert "execute_code" not in plan.tool_names
    assert plan.tool_names == ["file_read", "file_write"]


def test_tool_exposure_auto_review_limits_risky_count() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "shell", "execute_code", "file_write", "process_stop"],
        }
    )
    plan = planner.plan(
        message="debug and run code",
        available_tool_names=["file_read", "shell", "execute_code", "file_write", "process_stop"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )
    risky = {"shell", "execute_code", "file_write", "process_stop", "file_edit", "mcp_call"}
    risky_selected = [name for name in plan.tool_names if name in risky]
    assert plan.limit == 8
    assert len(risky_selected) == 3
    assert plan.tool_names == ["file_read", "file_write", "shell", "execute_code"]
