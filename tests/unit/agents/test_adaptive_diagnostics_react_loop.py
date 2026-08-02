"""Focused production-boundary tests for adaptive ReAct diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import pytest

from mochi.agents.adaptive_diagnostics import DIAGNOSTICS_CONTEXT_KEY
from mochi.agents.events import ToolCallResultEvent
from mochi.agents.react_loop import AsyncReActLoop
from mochi.backends.types import GenerationResult, ToolCall
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.registry import ToolRegistry
from tests.unit.engine._support import FakeBackend


class _ScriptedBackend(FakeBackend):
    def __init__(self, results: Sequence[GenerationResult | Exception]) -> None:
        super().__init__()
        self._results = list(results)

    async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(messages)
        self.generation_kwargs.append(dict(kwargs))
        next_result = self._results.pop(0)
        if isinstance(next_result, Exception):
            raise next_result
        return next_result


class _ProbeTool(BaseTool):
    def __init__(self, *, raises: bool = False) -> None:
        self.calls = 0
        self._raises = raises

    @property
    def name(self) -> str:
        return "probe"

    @property
    def description(self) -> str:
        return "Probe the real registry execution boundary."

    @property
    def parameters_schema(self) -> dict[str, object]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    async def execute(self, **_: object) -> ToolResult:
        self.calls += 1
        if self._raises:
            raise RuntimeError("tool exploded")
        return ToolResult(output="ok")


async def _run(loop: AsyncReActLoop) -> None:
    _ = [event async for event in loop.run("system", [], "request")]


@pytest.mark.asyncio
async def test_react_loop_records_successful_model_and_actual_tool_execution() -> None:
    context = ToolExecutionContext()
    tool = _ProbeTool()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(tool)
    backend = _ScriptedBackend(
        [
            GenerationResult(
                content="",
                input_tokens=12,
                output_tokens=4,
                tool_calls=[ToolCall(id="tool-1", name="probe", arguments={})],
                finish_reason="tool_calls",
            ),
            GenerationResult(content="done", input_tokens=7, output_tokens=3),
        ]
    )

    await _run(
        AsyncReActLoop(
            backend=backend, tool_registry=registry, tool_execution_context=context
        )
    )

    assert tool.calls == 1
    counters = context.state[DIAGNOSTICS_CONTEXT_KEY].snapshot()
    assert counters["model_calls"] == 2
    assert counters["tool_calls"] == 1
    assert counters["input_tokens"] == 19
    assert counters["output_tokens"] == 7
    assert counters["model_wall_ms"] >= 0
    assert counters["tool_wall_ms"] >= 0


@pytest.mark.asyncio
async def test_react_loop_records_backend_and_tool_exceptions_after_execution_boundary() -> (
    None
):
    backend_context = ToolExecutionContext()
    await _run(
        AsyncReActLoop(
            backend=_ScriptedBackend([RuntimeError("backend exploded")]),
            tool_execution_context=backend_context,
        )
    )
    assert backend_context.state[DIAGNOSTICS_CONTEXT_KEY].snapshot()["model_calls"] == 1

    tool_context = ToolExecutionContext()
    tool = _ProbeTool(raises=True)
    registry = ToolRegistry(discover_builtin=False)
    registry.register(tool)
    await _run(
        AsyncReActLoop(
            backend=_ScriptedBackend(
                [
                    GenerationResult(
                        content="",
                        tool_calls=[ToolCall(id="tool-1", name="probe", arguments={})],
                    ),
                    GenerationResult(content="done"),
                ]
            ),
            tool_registry=registry,
            tool_execution_context=tool_context,
        )
    )
    assert tool.calls == 1
    assert tool_context.state[DIAGNOSTICS_CONTEXT_KEY].snapshot()["tool_calls"] == 1


@pytest.mark.asyncio
async def test_plan_guard_does_not_count_model_requested_but_unexecuted_tool() -> None:
    context = ToolExecutionContext(
        state={
            "plan_runtime": {
                "enabled": True,
                "required": True,
                "state": "required",
                "preplan_read_calls_used": 0,
                "max_preplan_read_calls": 0,
                "plan_corrections_used": 0,
                "max_plan_prompt_corrections": 1,
            }
        }
    )
    tool = _ProbeTool()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(tool)
    await _run(
        AsyncReActLoop(
            backend=_ScriptedBackend(
                [
                    GenerationResult(
                        content="",
                        tool_calls=[ToolCall(id="tool-1", name="probe", arguments={})],
                    ),
                    GenerationResult(content="blocked"),
                ]
            ),
            tool_registry=registry,
            tool_execution_context=context,
        )
    )

    assert tool.calls == 0
    counters = context.state[DIAGNOSTICS_CONTEXT_KEY].snapshot()
    assert counters["tool_calls"] == 0
    assert counters["effectful_plan_guard_blocks"] == 1


@pytest.mark.asyncio
async def test_react_loop_without_context_is_a_diagnostics_noop() -> None:
    await _run(
        AsyncReActLoop(backend=_ScriptedBackend([GenerationResult(content="done")]))
    )


@pytest.mark.asyncio
async def test_unmarked_cached_context_resets_diagnostics_for_each_new_run() -> None:
    context = ToolExecutionContext()

    await _run(
        AsyncReActLoop(
            backend=_ScriptedBackend([GenerationResult(content="first")]),
            tool_execution_context=context,
        )
    )
    first_accumulator = context.state[DIAGNOSTICS_CONTEXT_KEY]
    await _run(
        AsyncReActLoop(
            backend=_ScriptedBackend([GenerationResult(content="second")]),
            tool_execution_context=context,
        )
    )

    assert context.state[DIAGNOSTICS_CONTEXT_KEY] is not first_accumulator
    assert context.state[DIAGNOSTICS_CONTEXT_KEY].snapshot()["model_calls"] == 1


@pytest.mark.asyncio
async def test_host_recovery_marker_attributes_real_react_model_attempt() -> None:
    context = ToolExecutionContext(
        state={
            "controlled_recovery_budget_runtime": {
                "model_calls_limit": 2,
                "model_calls_used": 0,
                "tool_calls_limit": 1,
                "tool_calls_used": 0,
            }
        }
    )
    await _run(
        AsyncReActLoop(
            backend=_ScriptedBackend(
                [GenerationResult(content="done", input_tokens=3, output_tokens=2)]
            ),
            tool_execution_context=context,
        )
    )
    counters = context.state[DIAGNOSTICS_CONTEXT_KEY].snapshot()
    assert (
        (counters["model_calls"], counters["input_tokens"], counters["output_tokens"])
        == (
            counters["recovery_model_calls"],
            counters["recovery_input_tokens"],
            counters["recovery_output_tokens"],
        )
        == (1, 3, 2)
    )


@pytest.mark.asyncio
async def test_real_tool_search_records_trusted_candidate_count() -> None:
    class SearchTool(_ProbeTool):
        @property
        def name(self) -> str:
            return "tool_search"

        async def execute(self, **_: object) -> ToolResult:
            self.calls += 1
            return ToolResult(output="ok", metadata={"count": 2})

    context = ToolExecutionContext()
    tool = SearchTool()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(tool)
    await _run(
        AsyncReActLoop(
            backend=_ScriptedBackend(
                [
                    GenerationResult(
                        content="",
                        tool_calls=[ToolCall(id="s", name="tool_search", arguments={})],
                    ),
                    GenerationResult(content="done"),
                ]
            ),
            tool_registry=registry,
            tool_execution_context=context,
        )
    )
    counters = context.state[DIAGNOSTICS_CONTEXT_KEY].snapshot()
    assert (
        counters["retrieval_search_queries"],
        counters["retrieval_candidates"],
        counters["retrieval_zero_match_queries"],
        counters["tool_calls"],
    ) == (1, 2, 0, 1)


@pytest.mark.asyncio
async def test_real_tool_search_records_zero_match() -> None:
    class SearchTool(_ProbeTool):
        @property
        def name(self) -> str:
            return "tool_search"

        async def execute(self, **_: object) -> ToolResult:
            self.calls += 1
            return ToolResult(output="ok", metadata={"count": 0})

    context = ToolExecutionContext()
    tool = SearchTool()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(tool)
    await _run(
        AsyncReActLoop(
            backend=_ScriptedBackend(
                [
                    GenerationResult(
                        content="",
                        tool_calls=[ToolCall(id="s", name="tool_search", arguments={})],
                    ),
                    GenerationResult(content="done"),
                ]
            ),
            tool_registry=registry,
            tool_execution_context=context,
        )
    )
    c = context.state[DIAGNOSTICS_CONTEXT_KEY].snapshot()
    assert (
        c["retrieval_search_queries"],
        c["retrieval_zero_match_queries"],
        c["retrieval_candidates"],
        c["tool_calls"],
    ) == (1, 1, 0, 1)


@pytest.mark.asyncio
async def test_real_registry_activation_counts_only_new_callable_schema() -> None:
    class OtherTool(_ProbeTool):
        @property
        def name(self) -> str:
            return "other_probe"

    source = ToolRegistry(discover_builtin=False)
    source.register(_ProbeTool())
    source.register(OtherTool())
    registry = source.create_view(
        [],
        tool_search_catalog_names=["probe", "other_probe"],
    )
    context = ToolExecutionContext(
        state={
            "tool_activation_policy": {
                "capability_enforcement_mode": "enforce",
                "activation_allowed_tool_names": ["probe"],
                "discoverable_tool_names": ["probe", "other_probe"],
                "execution_profile": "chat",
                "tool_mode": "auto",
                "tool_allowlist": None,
                "tool_denylist": None,
            }
        }
    )
    events = [
        event
        async for event in AsyncReActLoop(
            backend=_ScriptedBackend(
                [
                    GenerationResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="activate-new",
                                name="tool_activate",
                                arguments={"tool_name": "probe"},
                            )
                        ],
                    ),
                    GenerationResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="activate-existing",
                                name="tool_activate",
                                arguments={"tool_name": "probe"},
                            )
                        ],
                    ),
                    GenerationResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="probe",
                                name="probe",
                                arguments={},
                            )
                        ],
                    ),
                    GenerationResult(content="done"),
                ]
            ),
            tool_registry=registry,
            tool_execution_context=context,
        ).run("system", [], "request")
    ]

    counters = context.state[DIAGNOSTICS_CONTEXT_KEY].snapshot()
    activation_results = [
        event
        for event in events
        if isinstance(event, ToolCallResultEvent)
        and event.tool_name == "tool_activate"
    ]
    assert [event.metadata["status"] for event in activation_results] == [
        "tool_activated",
        "tool_already_callable",
    ]
    assert counters["retrieval_activations"] == 1
    assert counters["retrieval_schema_count_before_total"] == 1
    assert counters["retrieval_schema_count_after_total"] == 2
    assert counters["retrieval_schema_token_estimate_after_total"] > (
        counters["retrieval_schema_token_estimate_before_total"]
    )
    assert counters["tool_calls"] == 3


@pytest.mark.asyncio
async def test_tool_search_exception_records_attempt_without_inventing_matches() -> None:
    class ExplodingSearchTool(_ProbeTool):
        @property
        def name(self) -> str:
            return "tool_search"

    context = ToolExecutionContext()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(ExplodingSearchTool(raises=True))

    await _run(
        AsyncReActLoop(
            backend=_ScriptedBackend(
                [
                    GenerationResult(
                        content="",
                        tool_calls=[
                            ToolCall(id="search-error", name="tool_search", arguments={})
                        ],
                    ),
                    GenerationResult(content="done"),
                ]
            ),
            tool_registry=registry,
            tool_execution_context=context,
        )
    )

    counters = context.state[DIAGNOSTICS_CONTEXT_KEY].snapshot()
    assert (
        counters["retrieval_search_queries"],
        counters["retrieval_candidates"],
        counters["retrieval_zero_match_queries"],
        counters["tool_calls"],
    ) == (1, 0, 0, 1)


@pytest.mark.parametrize(
    "metadata",
    [
        cast(dict[str, Any], "not-a-mapping"),
        {"count": True},
        {"count": -1},
        {"count": 1.5},
        {"count": "2"},
    ],
)
@pytest.mark.asyncio
async def test_tool_search_malformed_metadata_fails_closed(
    metadata: dict[str, Any],
) -> None:
    class MalformedSearchTool(_ProbeTool):
        @property
        def name(self) -> str:
            return "tool_search"

        async def execute(self, **_: object) -> ToolResult:
            self.calls += 1
            return ToolResult(output="untrusted", metadata=metadata)

    context = ToolExecutionContext()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(MalformedSearchTool())

    await _run(
        AsyncReActLoop(
            backend=_ScriptedBackend(
                [
                    GenerationResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="search-malformed",
                                name="tool_search",
                                arguments={},
                            )
                        ],
                    ),
                    GenerationResult(content="done"),
                ]
            ),
            tool_registry=registry,
            tool_execution_context=context,
        )
    )

    counters = context.state[DIAGNOSTICS_CONTEXT_KEY].snapshot()
    assert (
        counters["retrieval_search_queries"],
        counters["retrieval_candidates"],
        counters["retrieval_zero_match_queries"],
        counters["tool_calls"],
    ) == (1, 0, 0, 1)


@pytest.mark.asyncio
async def test_forged_activation_metadata_without_schema_transition_is_not_counted() -> (
    None
):
    class ForgedActivationTool(_ProbeTool):
        @property
        def name(self) -> str:
            return "tool_activate"

        @property
        def parameters_schema(self) -> dict[str, object]:
            return {
                "type": "object",
                "properties": {"tool_name": {"type": "string"}},
                "required": ["tool_name"],
                "additionalProperties": False,
            }

        async def execute(self, **_: object) -> ToolResult:
            self.calls += 1
            return ToolResult(
                metadata={
                    "status": "tool_activated",
                    "requested_tool": "probe",
                    "callable_this_turn": True,
                }
            )

    context = ToolExecutionContext()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(ForgedActivationTool())

    await _run(
        AsyncReActLoop(
            backend=_ScriptedBackend(
                [
                    GenerationResult(
                        content="",
                        tool_calls=[
                            ToolCall(
                                id="forged-activation",
                                name="tool_activate",
                                arguments={"tool_name": "probe"},
                            )
                        ],
                    ),
                    GenerationResult(content="done"),
                ]
            ),
            tool_registry=registry,
            tool_execution_context=context,
        )
    )

    counters = context.state[DIAGNOSTICS_CONTEXT_KEY].snapshot()
    assert counters["retrieval_activations"] == 0
    assert counters["retrieval_schema_count_before_total"] == 0
    assert counters["retrieval_schema_count_after_total"] == 0
    assert counters["tool_calls"] == 1
