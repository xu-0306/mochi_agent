from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from mochi.agents.tool_exposure import ToolExposurePlanner
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, StreamChunk, ToolSchema


class _FakeBackend(BaseLLMBackend):
    def __init__(
        self,
        *,
        backend_type: str = "openai_compat",
        metadata: dict[str, object] | None = None,
    ) -> None:
        self._backend_type = backend_type
        self._metadata = dict(metadata or {})

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
        return ModelInfo(
            name="fake",
            backend_type=self._backend_type,
            metadata=dict(self._metadata),
        )

    async def health_check(self) -> bool:
        return True


@pytest.fixture
def planner() -> ToolExposurePlanner:
    return ToolExposurePlanner(tool_groups={})


def test_policy_baseline_is_route_independent_and_deduplicated(
    planner: ToolExposurePlanner,
) -> None:
    plan = planner.plan_contract_baseline(
        available_tool_names=["web_search", "tool_search", "web_search"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        attachment_count=2,
    )

    assert plan.tool_names == ["tool_search"]
    assert plan.discoverable_tool_names == ["web_search", "tool_search"]
    assert plan.limit == 8
    assert plan.workspace_bound is True
    assert plan.attachment_count == 2
    assert plan.diagnostics["stage"] == "contract_policy_baseline"
    assert plan.diagnostics["baseline_policy_tools"] == ["tool_search"]


def test_disabled_tool_mode_closes_policy_catalog(
    planner: ToolExposurePlanner,
) -> None:
    plan = planner.plan_contract_baseline(
        available_tool_names=["tool_search", "file_read"],
        backend=_FakeBackend(),
        session_bound_workspace=False,
        tool_mode="disabled",
    )

    assert plan.tool_names == []
    assert plan.discoverable_tool_names == []
    assert plan.limit == 0
    assert plan.diagnostics["disable_reason"] == "tool_mode_disabled"


@pytest.mark.parametrize(
    ("metadata", "reason"),
    [
        ({"tool_calling_blocked": True}, "backend_tool_calling_blocked"),
        ({"tool_call_mode": "unavailable"}, "backend_tool_calling_unavailable"),
    ],
)
def test_backend_tool_calling_ceiling_fails_closed(
    planner: ToolExposurePlanner,
    metadata: dict[str, object],
    reason: str,
) -> None:
    plan = planner.plan_contract_baseline(
        available_tool_names=["tool_search", "web_search"],
        backend=_FakeBackend(metadata=metadata),
        session_bound_workspace=False,
    )

    assert plan.tool_names == []
    assert plan.discoverable_tool_names == []
    assert plan.limit == 0
    assert plan.diagnostics["disable_reason"] == reason


def test_prompt_guided_ollama_remains_tool_eligible(
    planner: ToolExposurePlanner,
) -> None:
    plan = planner.plan_contract_baseline(
        available_tool_names=["tool_search", "web_search"],
        backend=_FakeBackend(
            backend_type="ollama",
            metadata={
                "tool_call_mode": "unavailable",
                "tool_calling_protocol": "prompt_guided",
            },
        ),
        session_bound_workspace=False,
    )

    assert plan.tool_names == ["tool_search"]
    assert plan.discoverable_tool_names == ["tool_search", "web_search"]


def test_strict_mode_applies_execution_ceiling_without_message_gates(
    planner: ToolExposurePlanner,
) -> None:
    plan = planner.plan_contract_baseline(
        available_tool_names=[
            "tool_search",
            "repo_map",
            "exec_command",
            "process_poll",
            "file_write",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="strict",
    )

    assert plan.limit == 4
    assert plan.discoverable_tool_names == [
        "tool_search",
        "repo_map",
        "file_write",
    ]


@pytest.mark.parametrize(
    ("backend_type", "autonomy_mode", "expected_limit"),
    [
        ("gguf", "high_autonomy", 6),
        ("safetensors", "auto_review", 6),
        ("openai_compat", "trusted_workspace", 6),
        ("openai_compat", "high_autonomy", 10),
    ],
)
def test_schema_limit_is_backend_and_policy_bounded(
    planner: ToolExposurePlanner,
    backend_type: str,
    autonomy_mode: str,
    expected_limit: int,
) -> None:
    plan = planner.plan_contract_baseline(
        available_tool_names=["tool_search"],
        backend=_FakeBackend(backend_type=backend_type),
        session_bound_workspace=False,
        autonomy_mode=autonomy_mode,
    )

    assert plan.limit == expected_limit
