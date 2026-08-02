"""AgentEngine Phase 2 整合測試。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mochi.agents import engine as engine_module
from mochi.agents.conversation_resolver import (
    BoundedConversationContext,
    ConversationResolver,
    IntentInterpretation,
)
from mochi.agents.engine import AgentEngine
from mochi.agents.events import (
    AgentEvent,
    FinalAnswerEvent,
    StatusEvent,
    ToolCallResultEvent,
)
from mochi.agents.invocation import AgentInvocationDiagnostics, AgentInvocationRequest
from mochi.agents.plan_ledger import PLAN_LEDGER_VERSION, PlanItem, PlanLedger
from mochi.agents.tool_exposure import ToolExposurePlan
from mochi.agents.turn_intent_contract import DeliverableContract
from mochi.backends.base import BackendRequestError
from mochi.backends.types import (
    AttachmentRef,
    GenerationResult,
    Message,
    ModelInfo,
    ToolCall,
)
from mochi.config.schema import MochiConfig
from mochi.learning.runtime import FailureAdvisoryHint
from mochi.tools.base import ToolExecutionContext
from mochi.tools.registry import ToolRegistry
from mochi.tools.update_plan import UpdatePlanTool
from tests.unit.engine._support import (
    EchoTool,
    FakeBackend,
)


class _OperationInterpreter:
    def __init__(self, *operations: str) -> None:
        self._operations = frozenset(operations)

    async def interpret(
        self,
        context: BoundedConversationContext,
    ) -> IntentInterpretation:
        return IntentInterpretation(
            current_speech_act="request_information",
            task_relation="standalone",
            objective=context.current_turn.content,
            operations=self._operations,  # type: ignore[arg-type]
            confidence=0.99,
        )


def _resolver_factory(*operations: str):  # type: ignore[no-untyped-def]
    return lambda backend: ConversationResolver(
        interpreter=_OperationInterpreter(*operations)
    )


def _unavailable_resolver_factory():  # type: ignore[no-untyped-def]
    class _UnavailableInterpreter:
        async def interpret(self, context: BoundedConversationContext) -> IntentInterpretation:
            del context
            raise BackendRequestError("interpreter unavailable")

    return lambda backend: ConversationResolver(interpreter=_UnavailableInterpreter())


async def _bind_fake_backend(engine: AgentEngine, backend: FakeBackend) -> None:
    async def fake_load(model_spec: str) -> FakeBackend:
        del model_spec
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = fake_load  # type: ignore[method-assign]


class _FallbackActivationBackend(FakeBackend):
    def __init__(self, *, activation_target: str, invoke_datetime: bool) -> None:
        super().__init__()
        self._activation_target = activation_target
        self._invoke_datetime = invoke_datetime
        self.schema_history: list[list[str]] = []

    async def generate(  # type: ignore[override]
        self,
        messages: list[Message],
        tools: list | None = None,
        **kwargs: object,
    ) -> GenerationResult:
        self.calls.append(messages)
        tool_names = [tool.name for tool in tools or []]
        self.tool_calls_seen.append(tool_names)
        self.schema_history.append(tool_names)
        self.generation_kwargs.append(dict(kwargs))
        call_number = len(self.calls)
        if call_number == 1:
            assert "tool_activate" in tool_names
            assert "get_current_time" not in tool_names
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="activate-1",
                        name="tool_activate",
                        arguments={"tool_name": self._activation_target},
                    )
                ],
                finish_reason="tool_calls",
            )
        if self._invoke_datetime and call_number == 2:
            assert "get_current_time" in tool_names
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="datetime-1",
                        name="get_current_time",
                        arguments={"timezone": "UTC"},
                    )
                ],
                finish_reason="tool_calls",
            )
        return GenerationResult(content="completed", finish_reason="stop")


@pytest.mark.parametrize(
    "message",
    [
        "請回報現在的 UTC 時間。",
        "Report the current UTC timestamp using an available read-only tool.",
    ],
)
@pytest.mark.asyncio
async def test_interpreter_outage_safe_read_only_activation(
    tmp_path: Path,
    message: str,
) -> None:
    backend = _FallbackActivationBackend(
        activation_target="get_current_time",
        invoke_datetime=True,
    )
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config, conversation_resolver_factory=_unavailable_resolver_factory())
    await _bind_fake_backend(engine, backend)

    result = await engine.invoke(
        AgentInvocationRequest(message=message, persist_session=False)
    )

    adapter = result.diagnostics.tool_exposure["diagnostics"]["capability_exposure_adapter"]
    allowed = set(adapter["activation_allowed_tool_names"])
    direct = set(result.diagnostics.exposed_tools)
    assert adapter["required_capabilities"] == ["tool_discovery"]
    assert "tool_search" in direct
    assert {"web_search", "get_current_time"} & allowed
    assert not ({"web_search", "get_current_time"} & direct)
    deferred = set(adapter["activation_broker"]["deferred_tool_names"])
    assert deferred <= allowed
    tool_results = [
        event for event in result.events if isinstance(event, ToolCallResultEvent)
    ]
    assert [event.tool_name for event in tool_results] == [
        "tool_activate",
        "get_current_time",
    ]
    assert tool_results[0].error is None
    assert tool_results[0].metadata["status"] == "tool_activated"
    assert tool_results[1].error is None
    assert tool_results[1].result["timezone"] == "UTC"
    assert tool_results[1].result["iso"]
    assert "get_current_time" not in backend.schema_history[0]
    assert "get_current_time" in backend.schema_history[1]
    assert result.content == "completed"
    await engine.close()


@pytest.mark.parametrize(
    "unsafe_tool",
    ["file_write", "exec_command", "update_plan", "tool_result_read"],
)
@pytest.mark.asyncio
async def test_interpreter_outage_denies_unsafe_activation(
    tmp_path: Path,
    unsafe_tool: str,
) -> None:
    backend = _FallbackActivationBackend(
        activation_target=unsafe_tool,
        invoke_datetime=False,
    )
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config, conversation_resolver_factory=_unavailable_resolver_factory())
    await _bind_fake_backend(engine, backend)

    result = await engine.invoke(
        AgentInvocationRequest(message="do the task", persist_session=False)
    )

    adapter = result.diagnostics.tool_exposure["diagnostics"]["capability_exposure_adapter"]
    allowed = set(adapter["activation_allowed_tool_names"])
    assert not allowed & {
        "update_plan",
        "file_write",
        "exec_command",
        "tool_result_read",
    }
    assert adapter["activation_broker"]["deferred_tool_names"]
    tool_results = [
        event for event in result.events if isinstance(event, ToolCallResultEvent)
    ]
    assert len(tool_results) == 1
    assert tool_results[0].tool_name == "tool_activate"
    assert tool_results[0].error is not None
    assert tool_results[0].metadata["error_type"] == "tool_activation_denied"
    assert tool_results[0].metadata["requested_tool"] == unsafe_tool
    assert all(unsafe_tool not in schemas for schemas in backend.schema_history)
    assert not any(
        event.tool_name == unsafe_tool
        for event in result.events
        if isinstance(event, ToolCallResultEvent)
    )
    assert not (tmp_path / "denied-side-effect.txt").exists()
    await engine.close()


@pytest.mark.asyncio
async def test_invoke_failure_hints_are_advisory_and_do_not_expand_tools(tmp_path: Path) -> None:
    def config(enabled: bool) -> MochiConfig:
        return MochiConfig.model_validate({
            "model": "ollama:test", "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / ("on" if enabled else "off")),
            "memory": {"db_path": str(tmp_path / ("on.db" if enabled else "off.db"))},
            "agent": {"ordinary_chat_adaptive_runtime": {"failure_learning": {"hint_injection_enabled": enabled}}},
        })
    async def bind(engine: AgentEngine, backend: FakeBackend) -> None:
        async def load(_: str) -> FakeBackend:
            engine._router._active = backend  # noqa: SLF001
            return backend
        engine._router.load = load  # type: ignore[method-assign]

    off_backend = FakeBackend()
    off = AgentEngine(config(False), conversation_resolver_factory=_resolver_factory())
    off.learning_runtime.advisory_hint_selections = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            FailureAdvisoryHint(
                candidate_id="candidate-off",
                text="SHOULD NOT APPEAR",
            ),
        )
    )
    await bind(off, off_backend)
    await off.invoke(AgentInvocationRequest(message="hello", session_id="off", persist_session=False))
    assert off.learning_runtime.advisory_hint_selections.await_count == 0
    assert "SHOULD NOT APPEAR" not in str(off_backend.calls[0][0].content)

    on_backend = FakeBackend()
    on = AgentEngine(config(True), conversation_resolver_factory=_resolver_factory())
    on.learning_runtime.advisory_hint_selections = AsyncMock(  # type: ignore[method-assign]
        return_value=(
            FailureAdvisoryHint(
                candidate_id="candidate-on",
                text="bounded hint",
            ),
        )
    )
    on.learning_runtime.record_hint_selections = AsyncMock(  # type: ignore[method-assign]
        return_value=1
    )
    await bind(on, on_backend)
    await on.invoke(
        AgentInvocationRequest(
            message="hello",
            session_id="on",
            turn_id="turn-hint",
            persist_session=True,
        )
    )
    assert on.learning_runtime.advisory_hint_selections.await_count >= 1
    on.learning_runtime.record_hint_selections.assert_awaited_once()
    hint_attribution = (
        on.learning_runtime.record_hint_selections.await_args.kwargs
    )
    assert hint_attribution["session_id"] == "on"
    assert hint_attribution["turn_id"] == "turn-hint"
    system = str(on_backend.calls[0][0].content)
    assert "bounded hint" in system and "grant no tools or authority" in system
    assert on_backend.tool_calls_seen[0] == off_backend.tool_calls_seen[0]
    await off.close()
    await on.close()


def _plan_item(
    *,
    item_id: str,
    status: str,
    dependencies: tuple[str, ...] = (),
    evidence_refs: tuple[str, ...] = (),
) -> PlanItem:
    return PlanItem(
        item_id=item_id,
        title=f"Title for {item_id}",
        status=status,  # type: ignore[arg-type]
        dependencies=dependencies,
        success_criteria=("done",),
        source_turn_ids=("turn-1",),
        evidence_refs=evidence_refs,
    )


def _rollout(goal_id: str) -> object:
    contract = SimpleNamespace(
        objective="Build the selected artifact",
        active_goal_id=goal_id,
        cancels_active_goal=False,
        supersedes_previous_goal=False,
        modifies_active_task=True,
    )
    return SimpleNamespace(
        resolution=SimpleNamespace(
            contract=contract,
            context=SimpleNamespace(
                active_task=SimpleNamespace(goal_id=goal_id),
            ),
            next_active_task=SimpleNamespace(goal_id=goal_id),
        )
    )


@pytest.mark.asyncio
async def test_engine_weather_prompt_exposes_only_web_subset_for_local_backend(
    tmp_path: Path,
) -> None:
    """Weather queries should expose only the web tool subset, capped for local backends."""
    fake_backend = FakeBackend()

    async def fake_health_check() -> bool:
        return True

    fake_backend.health_check = fake_health_check  # type: ignore[method-assign]
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="local.gguf",
        backend_type="gguf",
        context_length=32768,
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory("open_world_lookup"),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    _ = [event async for event in engine.chat("請幫我查詢今天台中天氣", session_id="s1")]

    exposed = fake_backend.tool_calls_seen[-1]
    assert {"web_search", "tool_search", "tool_activate"} <= set(exposed)
    assert "file_read" not in exposed
    assert len(exposed) <= 6

    await engine.close()


@pytest.mark.asyncio
async def test_engine_temporal_lookup_contract_allows_read_only_datetime_activation(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory("temporal_lookup"),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        del model_spec
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    result = await engine.invoke(
        AgentInvocationRequest(
            message="Use get_current_time to report the current time.",
            session_id="read-only-datetime-exposure",
            persist_session=False,
        )
    )

    adapter = result.diagnostics.tool_exposure["diagnostics"][
        "capability_exposure_adapter"
    ]
    assert "get_current_time" in adapter["activation_allowed_tool_names"]
    rollout_stage = next(
        stage
        for stage in result.diagnostics.tool_exposure["diagnostics"]["stages"]
        if stage.get("stage") == "turn_contract_rollout"
    )
    assert "get_current_time" in rollout_stage["capability_plan"]["eligible_tools"]
    datetime_diagnostic = next(
        diagnostic
        for diagnostic in rollout_stage["capability_plan"]["tool_diagnostics"]
        if diagnostic["tool_name"] == "get_current_time"
    )
    assert datetime_diagnostic["status"] == "exposed"

    await engine.close()


@pytest.mark.asyncio
async def test_same_contract_with_different_wording_has_identical_tool_plan(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        context_length=32768,
        supports_tool_calling=True,
    )
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory("open_world_lookup"),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        del model_spec
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    first = await engine.invoke(
        AgentInvocationRequest(
            message="What is the weather today?",
            session_id="contract-wording-a",
            persist_session=False,
        )
    )
    second = await engine.invoke(
        AgentInvocationRequest(
            message="Find recent external sources about model evaluation.",
            session_id="contract-wording-b",
            persist_session=False,
        )
    )

    assert first.diagnostics.exposed_tools == second.diagnostics.exposed_tools
    assert (
        first.diagnostics.tool_exposure["diagnostics"][
            "capability_exposure_adapter"
        ]["activation_allowed_tool_names"]
        == second.diagnostics.tool_exposure["diagnostics"][
            "capability_exposure_adapter"
        ]["activation_allowed_tool_names"]
    )

    await engine.close()


@pytest.mark.asyncio
async def test_engine_coding_prompt_exposes_workspace_subset(
    tmp_path: Path,
) -> None:
    """Coding prompts should expose workspace tools instead of the full registry."""
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        context_length=32768,
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {
                "require_approval_for_exec": False,
                "require_approval_for_file_write": False,
                "command_rules": [{"tokens": ["echo"], "decision": "allow", "match": "prefix"}],
                "max_file_write_size_mb": 1,
            },
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory("workspace_read"),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    _ = [event async for event in engine.chat("請幫我 debug 這個 repo 的 test failure", session_id="s1")]

    exposed = fake_backend.tool_calls_seen[-1]
    assert {"repo_map", "tool_search", "tool_activate"} <= set(exposed)
    assert "web_search" not in exposed

    await engine.close()


@pytest.mark.asyncio
async def test_engine_chinese_workspace_contract_exposes_minimal_read_and_broker(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        context_length=32768,
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory("workspace_read"),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    _ = [
        event
        async for event in engine.chat(
            "請檢查這個工作區，找出包含 TODO 的地方，並查看相關內容",
            session_id="workspace-zh",
            workspace_dir=str(tmp_path / "scoped-workspace"),
        )
    ]

    exposed = fake_backend.tool_calls_seen[-1]
    assert {"repo_map", "tool_search", "tool_activate"} <= set(exposed)
    assert not ({"file_write", "file_edit", "apply_patch"} & set(exposed))

    await engine.close()


@pytest.mark.asyncio
async def test_engine_attachment_prompt_context_distinguishes_attachment_sources(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)

    attachments = [
        AttachmentRef(
            name="brief.md",
            path=str(tmp_path / "brief.md"),
            source="upload",
        ),
        AttachmentRef(
            name="app.py",
            path=str(tmp_path / "app.py"),
            source="workspace_file",
        ),
        AttachmentRef(
            name="engine.py",
            path=str(tmp_path / "engine.py"),
            source="workspace_selection",
            line_start=10,
            line_end=14,
            quote="def execute(...):",
            note="Investigate this branch.",
        ),
        AttachmentRef(
            name="error.png",
            path=str(tmp_path / "error.png"),
            source="image",
        ),
    ]

    prompt_context = engine._build_attachment_prompt_context(  # noqa: SLF001
        attachments=attachments,
        available_tool_names=["file_read", "image_view"],
    )

    assert "uploads, workspace references, selections, or images" in prompt_context
    assert "quote: \"def execute(...):\"" in prompt_context
    assert "note: Investigate this branch." in prompt_context
    assert "[workspace selection]" in prompt_context
    assert "[image]" in prompt_context

    await engine.close()


@pytest.mark.asyncio
async def test_engine_invoke_exposes_tool_exposure_metadata_from_final_plan(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        context_length=32768,
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory("workspace_read"),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    result = await engine.invoke(
        AgentInvocationRequest(
            message="請檢查這個工作區，找出包含 TODO 的地方，並查看相關內容",
            session_id="diagnostics-worker",
            workspace_dir=str(tmp_path / "scoped-workspace"),
            attachments=[
                AttachmentRef(
                    name="brief.md",
                    path=str(tmp_path / "brief.md"),
                    source="workspace_file",
                ),
                AttachmentRef(
                    name="notes.pdf",
                    path=str(tmp_path / "notes.pdf"),
                    source="workspace_selection",
                ),
            ],
            tool_mode="auto",
            execution_profile="chat",
            persist_session=False,
        )
    )

    tool_exposure = result.diagnostics.to_dict()["tool_exposure"]
    assert tool_exposure["exposed_tools"] == result.diagnostics.exposed_tools
    assert tool_exposure["workspace_bound"] is True
    assert tool_exposure["attachment_count"] == 2
    assert tool_exposure.get("intent_route") is None
    adapter = tool_exposure["diagnostics"]["capability_exposure_adapter"]
    assert adapter["required_capabilities"] == ["workspace_read"]

    await engine.close()


def test_agent_invocation_diagnostics_to_dict_serializes_tool_exposure() -> None:
    diagnostics = AgentInvocationDiagnostics(
        execution_profile="chat",
        tool_mode="auto",
        exposed_tools=["file_read"],
        matched_tool_groups=["workspace"],
        tool_exposure={
            "exposed_tools": ["file_read"],
            "workspace_bound": True,
            "attachment_count": 2,
        },
        adaptive_runtime={
            "complexity": {},
            "plan": {},
            "retrieval": {},
            "verification": {},
            "recovery": {},
            "failure_learning": {},
        },
    )

    assert diagnostics.to_dict()["tool_exposure"] == diagnostics.tool_exposure
    assert diagnostics.to_dict()["adaptive_runtime"] == diagnostics.adaptive_runtime


@pytest.mark.asyncio
async def test_configure_plan_runtime_exposes_update_plan_for_active_chat_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "agent": {
                "ordinary_chat_adaptive_runtime": {
                    "complexity": {"mode": "enforce"},
                }
            },
        }
    )
    engine = AgentEngine(config)
    ledger = PlanLedger(
        ledger_version=PLAN_LEDGER_VERSION,
        ledger_id="plan-goal-plan",
        session_id="plan-session",
        goal_id="goal-plan",
        revision=0,
        status="active",
        objective="Build the selected artifact",
        reason_codes=("multiple_deliverables",),
        items=(
            _plan_item(
                item_id="item-1",
                status="completed",
                evidence_refs=("receipt-1",),
            ),
            _plan_item(
                item_id="item-2",
                status="in_progress",
                dependencies=("item-1",),
            ),
        ),
        created_turn_id="turn-0",
        updated_turn_id="turn-0",
    )
    saved = await engine._plan_ledger_repository.save(  # noqa: SLF001
        ledger,
        expected_revision=0,
        turn_id="turn-0",
        idempotency_key="plan-update:seed",
    )
    assert saved.status == "saved"
    async def _continue_existing_plan(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "kind": "continue_existing_plan",
            "hard_reason_codes": ["multiple_deliverables"],
        }

    monkeypatch.setattr(
        engine,
        "_resolve_complexity_decision",
        _continue_existing_plan,
    )
    tool_execution_context = ToolExecutionContext(session_id="plan-session")
    exposure_plan = ToolExposurePlan(
        tool_names=["file_read"],
        matched_groups=["workspace"],
        limit=4,
        discoverable_tool_names=["file_read", "update_plan"],
    )

    updated_plan, complexity_decision, task_plan_context = await engine._configure_plan_runtime(  # noqa: SLF001
        session_id="plan-session",
        turn_id="turn-1",
        request=AgentInvocationRequest(
            message="Continue the existing plan",
            session_id="plan-session",
            tool_mode="auto",
            execution_profile="chat",
            persist_session=True,
        ),
        rollout=_rollout("goal-plan"),  # type: ignore[arg-type]
        available_tools=[EchoTool(), UpdatePlanTool()],
        exposure_plan=exposure_plan,
        tool_execution_context=tool_execution_context,
    )

    plan_runtime = tool_execution_context.state["plan_runtime"]
    assert complexity_decision["kind"] == "continue_existing_plan"
    assert "update_plan" in updated_plan.tool_names
    assert plan_runtime["enabled"] is True
    assert plan_runtime["state"] == "active"
    assert plan_runtime["required"] is True
    assert plan_runtime["exposed"] is True
    assert plan_runtime["mutable"] is True
    assert plan_runtime["ledger_status"] == "active"
    assert plan_runtime["current_revision"] == 1
    assert plan_runtime["current_item_id"] == "item-2"
    assert plan_runtime["completed_item_ids"] == ["item-1"]
    assert tool_execution_context.state["plan_ledger_snapshot"]["ledger_id"] == "plan-goal-plan"
    assert tool_execution_context.state["recognized_plan_evidence_refs"] == {"receipt-1"}
    assert "update_plan_controller" in tool_execution_context.state
    assert task_plan_context is not None
    assert "Use `update_plan` to create or update the durable task plan." in task_plan_context
    assert "Current in-progress item: item-2" in task_plan_context

    view = await UpdatePlanTool().execute(
        action="view",
        expected_revision=0,
        items=[],
        item_id=None,
        status=None,
        evidence_refs=[],
        blocker_reason=None,
        context=tool_execution_context,
    )
    assert view.error is None
    assert view.output["status"] == "loaded"
    assert view.output["ledger"]["ledger_id"] == "plan-goal-plan"
    await engine.close()


@pytest.mark.asyncio
async def test_configure_plan_runtime_keeps_update_plan_hidden_when_chat_plan_runtime_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "agent": {
                "ordinary_chat_adaptive_runtime": {
                    "complexity": {"mode": "enforce"},
                }
            },
        }
    )
    engine = AgentEngine(config)
    async def _plan_required(**kwargs: object) -> dict[str, object]:
        del kwargs
        return {
            "kind": "plan_required",
            "hard_reason_codes": ["multiple_deliverables"],
        }

    monkeypatch.setattr(
        engine,
        "_resolve_complexity_decision",
        _plan_required,
    )
    tool_execution_context = ToolExecutionContext(session_id="plan-session")
    exposure_plan = ToolExposurePlan(
        tool_names=["file_read"],
        matched_groups=["workspace"],
        limit=4,
        discoverable_tool_names=["file_read", "update_plan"],
    )

    updated_plan, _, task_plan_context = await engine._configure_plan_runtime(  # noqa: SLF001
        session_id="plan-session",
        turn_id="turn-1",
        request=AgentInvocationRequest(
            message="Plan this task",
            session_id="plan-session",
            tool_mode="disabled",
            execution_profile="chat",
            persist_session=True,
        ),
        rollout=_rollout("goal-plan"),  # type: ignore[arg-type]
        available_tools=[EchoTool(), UpdatePlanTool()],
        exposure_plan=exposure_plan,
        tool_execution_context=tool_execution_context,
    )

    plan_runtime = tool_execution_context.state["plan_runtime"]
    assert updated_plan.tool_names == ["file_read"]
    assert plan_runtime["enabled"] is False
    assert plan_runtime["state"] == "unavailable"
    assert plan_runtime["unavailable_reason"] == "planning_unavailable_tool_mode"
    assert "update_plan_controller" not in tool_execution_context.state
    assert "update_plan_runtime" not in tool_execution_context.state
    assert task_plan_context is None
    await engine.close()


@pytest.mark.asyncio
async def test_engine_invoke_exposes_diagnostics_and_honors_disabled_tool_mode(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory(),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]
    started_trajectories: list[str] = []
    engine._start_trajectory = lambda message: started_trajectories.append(message) or "traj"  # type: ignore[method-assign]  # noqa: SLF001

    result = await engine.invoke(
        AgentInvocationRequest(
            message="review the attached design note",
            session_id="worker-a",
            tool_mode="disabled",
            execution_profile="subagent_readonly",
            system_prompt_addendum="Role identity: Reviewer",
            persist_session=False,
        )
    )

    assert result.content == "fake reply"
    assert result.diagnostics.execution_profile == "subagent_readonly"
    assert result.diagnostics.tool_mode == "disabled"
    assert result.diagnostics.exposed_tools == []
    assert result.diagnostics.adaptive_runtime is not None
    assert "complexity" in result.diagnostics.adaptive_runtime
    assert "verification" in result.diagnostics.adaptive_runtime
    assert fake_backend.tool_calls_seen[-1] == []
    assert started_trajectories == []

    await engine.close()


@pytest.mark.asyncio
async def test_engine_subagent_research_profile_keeps_tools_read_only(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        context_length=32768,
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {"autonomy_mode": "auto_review"},
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory("workspace_read"),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    result = await engine.invoke(
        AgentInvocationRequest(
            message="search repo files, inspect csv data, then run training with exec_command and execute code",
            session_id="research-worker",
            tool_mode="auto",
            execution_profile="subagent_research",
            persist_session=False,
        )
    )

    risky = {
        "exec_command",
        "execute_code",
        "execute_code_v2",
        "file_write",
        "file_edit",
        "write_stdin",
        "kill_session",
        "process_stop",
        "mcp_call",
    }
    assert not (set(result.diagnostics.exposed_tools) & risky)
    assert not (set(fake_backend.tool_calls_seen[-1]) & risky)
    assert {"repo_map", "grep_search", "csv_read"} & set(
        result.diagnostics.exposed_tools
    )

    await engine.close()


@pytest.mark.asyncio
async def test_engine_controlled_execution_profiles_gate_risky_tools(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {"autonomy_mode": "auto_review"},
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory("workspace_read"),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    executor_result = await engine.invoke(
        AgentInvocationRequest(
            message="delegate subagent execution, inspect files, run command, and write results",
            session_id="controlled-executor",
            tool_mode="auto",
            execution_profile="subagent_execution_request",
            persist_session=False,
        )
    )
    executor_tools = set(executor_result.diagnostics.exposed_tools)
    assert "exec_command" not in executor_tools
    assert "file_write" not in executor_tools
    assert "delegate_subagent_task" not in executor_tools

    controller_result = await engine.invoke(
        AgentInvocationRequest(
            message="review command and run command in background",
            session_id="controlled-controller",
            tool_mode="auto",
            execution_profile="controller_exec",
            persist_session=False,
        )
    )
    controller_tools = set(controller_result.diagnostics.exposed_tools)
    assert "exec_command" in controller_tools
    assert "process_poll" in controller_tools
    assert "file_write" not in controller_tools

    await engine.close()


@pytest.mark.asyncio
async def test_engine_restricted_profiles_use_hard_readonly_allowlists(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        context_length=32768,
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {"autonomy_mode": "auto_review"},
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory("workspace_read"),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    blocked_memory_tools = {"memory_save", "memory_update", "memory_delete"}
    for profile in ("subagent_readonly", "judge", "verifier"):
        session_id = f"profile-{profile}"
        context = engine._get_tool_execution_context(  # noqa: SLF001
            session_id=session_id,
            workspace_dir=str(tmp_path),
        )
        context.tool_result_references["file_read-profile-ref"] = {
            "reference_id": "file_read-profile-ref",
            "artifact_path": str(tmp_path / f"{profile}-artifact.txt"),
            "source_path": str(tmp_path / f"{profile}-source.txt"),
        }
        result = await engine.invoke(
            AgentInvocationRequest(
                message="inspect repo files, search memory, and continue reading prior tool output if needed",
                session_id=session_id,
                tool_mode="auto",
                execution_profile=profile,  # type: ignore[arg-type]
                persist_session=False,
            )
        )
        exposed = set(result.diagnostics.exposed_tools)
        assert "repo_map" in exposed
        assert "tool_result_read" in exposed
        assert not (exposed & blocked_memory_tools)

    await engine.close()


@pytest.mark.asyncio
async def test_engine_followup_open_world_turn_preserves_tool_result_read_only_with_references(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        context_length=32768,
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {"autonomy_mode": "auto_review"},
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory("open_world_lookup"),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    session_id = "continuation-open-world"
    context = engine._get_tool_execution_context(  # noqa: SLF001
        session_id=session_id,
        workspace_dir=str(tmp_path),
    )
    context.tool_result_references["file_read-abc123"] = {
        "reference_id": "file_read-abc123",
        "artifact_path": str(tmp_path / "artifact.txt"),
        "source_path": str(tmp_path / "source.txt"),
    }

    result = await engine.invoke(
        AgentInvocationRequest(
            message="find recent papers about weather forecast model fine-tuning",
            session_id=session_id,
            workspace_dir=str(tmp_path),
            tool_mode="auto",
            execution_profile="subagent_research",
            persist_session=False,
        )
    )

    exposed = set(result.diagnostics.exposed_tools)
    assert "tool_result_read" in exposed
    assert "tool_result_read" in set(fake_backend.tool_calls_seen[-1])
    assert {"arxiv_search", "semantic_scholar_search", "web_search"} & exposed

    await engine.close()


@pytest.mark.asyncio
async def test_engine_followup_open_world_turn_does_not_preserve_tool_result_read_without_references(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {"autonomy_mode": "auto_review"},
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    result = await engine.invoke(
        AgentInvocationRequest(
            message="find recent papers about weather forecast model fine-tuning",
            session_id="no-continuation-open-world",
            workspace_dir=str(tmp_path),
            tool_mode="auto",
            execution_profile="subagent_research",
            persist_session=False,
        )
    )

    exposed = set(result.diagnostics.exposed_tools)
    assert "tool_result_read" not in exposed
    assert "tool_result_read" not in set(fake_backend.tool_calls_seen[-1])

    await engine.close()


@pytest.mark.asyncio
async def test_engine_invocation_tool_overrides_are_limited_by_profile(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {"autonomy_mode": "auto_review"},
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory("workspace_read"),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    result = await engine.invoke(
        AgentInvocationRequest(
            message="search memory and files",
            session_id="override-worker",
            tool_mode="auto",
            execution_profile="subagent_readonly",
            tool_names_override=[
                "file_read",
                "memory_save",
                "web_search",
                "exec_command",
                "file_read",
            ],
            tool_allowlist=["file_read", "memory_save", "web_search", "exec_command"],
            tool_denylist=["web_search"],
            persist_session=False,
        )
    )

    assert result.diagnostics.exposed_tools == ["file_read"]
    assert fake_backend.tool_calls_seen[-1] == ["file_read"]

    await engine.close()


@pytest.mark.asyncio
async def test_engine_blocks_unavailable_contract_write_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = FakeBackend(metadata={"effective_context_length": 32768})
    seen_requires_file_mutation: list[bool] = []
    seen_tool_names: list[list[str]] = []

    class SpyReActLoop:
        def __init__(
            self,
            *args: object,
            tool_registry: ToolRegistry | None = None,
            requires_file_mutation: bool = False,
            **kwargs: object,
        ) -> None:
            del args, kwargs
            seen_requires_file_mutation.append(requires_file_mutation)
            seen_tool_names.append(
                [schema["function"]["name"] for schema in tool_registry.get_schemas()]
                if tool_registry is not None
                else []
            )

        async def run(self, *args: object, **kwargs: object) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            yield FinalAnswerEvent(content="spy reply")

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    class _Interpreter:
        async def interpret(
            self,
            context: BoundedConversationContext,
        ) -> IntentInterpretation:
            if "weather" in context.current_turn.content.lower():
                return IntentInterpretation(
                    current_speech_act="request_information",
                    task_relation="standalone",
                    operations=frozenset({"open_world_lookup"}),
                    confidence=0.99,
                )
            return IntentInterpretation(
                current_speech_act="request_execution",
                task_relation="standalone",
                operations=frozenset({"workspace_write"}),
                deliverables=(
                    DeliverableContract(
                        kind="workspace_file",
                        target_hint="report.md",
                        source_turn_ids=(context.current_turn.turn_id,),
                    ),
                ),
                mutation_requirement="required",
                confidence=0.99,
            )

    engine = AgentEngine(
        config,
        conversation_resolver_factory=lambda backend: ConversationResolver(
            interpreter=_Interpreter()
        ),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        del model_spec
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    result = await engine.invoke(
        AgentInvocationRequest(
            message="create report.md",
            session_id="workspace-write-obligation-worker",
            tool_mode="auto",
            execution_profile="chat",
            tool_denylist=["file_write", "file_edit", "apply_patch"],
            persist_session=False,
        )
    )

    assert "required capabilities are unavailable" in result.content
    assert result.diagnostics.fallback_reason == (
        "turn_contract_required_capability_unavailable"
    )
    assert seen_requires_file_mutation == []

    fallback_result = await engine.invoke(
        AgentInvocationRequest(
            message="save report.md",
            session_id="ambiguous-workspace-write-obligation-worker",
            tool_mode="auto",
            execution_profile="chat",
            tool_denylist=["file_write", "file_edit", "apply_patch"],
            persist_session=False,
        )
    )

    assert "required capabilities are unavailable" in fallback_result.content
    assert seen_requires_file_mutation == []

    non_write_result = await engine.invoke(
        AgentInvocationRequest(
            message="Update me on the latest weather in Taipei",
            session_id="open-world-obligation-worker",
            tool_mode="auto",
            execution_profile="chat",
            tool_denylist=["file_write", "file_edit", "apply_patch"],
            persist_session=False,
        )
    )

    assert non_write_result.content == "spy reply"
    assert seen_requires_file_mutation == [False]

    await engine.close()

@pytest.mark.asyncio
async def test_engine_invocation_max_iterations_override_reaches_react_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = FakeBackend()
    seen_iterations: list[int] = []

    class SpyReActLoop:
        def __init__(self, *args: object, max_iterations: int = 10, **kwargs: object) -> None:
            del args, kwargs
            seen_iterations.append(max_iterations)

        async def run(self, *args: object, **kwargs: object) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            yield FinalAnswerEvent(content="spy reply")

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory(),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    result = await engine.invoke(
        AgentInvocationRequest(
            message="quick review",
            session_id="iteration-worker",
            tool_mode="disabled",
            execution_profile="subagent_readonly",
            max_iterations_override=2,
            persist_session=False,
        )
    )

    assert result.content == "spy reply"
    assert seen_iterations == [2]

    await engine.close()


@pytest.mark.asyncio
async def test_engine_chat_binds_turn_id_into_tool_execution_context_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = FakeBackend()
    seen_turn_ids: list[str | None] = []

    class SpyReActLoop:
        def __init__(
            self,
            *args: object,
            tool_execution_context: object | None = None,
            **kwargs: object,
        ) -> None:
            del args, kwargs
            state = getattr(tool_execution_context, "state", None)
            seen_turn_ids.append(state.get("turn_id") if isinstance(state, dict) else None)

        async def run(self, *args: object, **kwargs: object) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            yield FinalAnswerEvent(content="spy reply")

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory("workspace_read"),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    result = await engine.invoke(
        AgentInvocationRequest(
            message="inspect repo files",
            session_id="turn-id-worker",
            tool_mode="auto",
            execution_profile="chat",
            persist_session=False,
        )
    )

    assert result.content == "spy reply"
    assert len(seen_turn_ids) == 1
    assert isinstance(seen_turn_ids[0], str) and seen_turn_ids[0]

    await engine.close()


@pytest.mark.asyncio
async def test_engine_tool_search_hook_persists_discovery_state(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)
    session_id = "tool-discovery-session"
    turn_id = "turn-1"

    await engine._persist_session_message(  # noqa: SLF001
        session_id,
        Message(role="user", content="find the file read tool"),
        turn_id=turn_id,
    )
    context = engine._get_tool_execution_context(  # noqa: SLF001
        session_id=session_id,
        workspace_dir=str(tmp_path),
    )
    context.state["turn_id"] = turn_id
    registry = engine._tool_registry_factory.create_registry(str(tmp_path))  # noqa: SLF001

    result = await registry.execute(
        "tool_search",
        {"query": "file_read"},
        context=context,
    )
    loaded = await engine._tool_discovery_state_repository.load(session_id)  # noqa: SLF001

    assert result.error is None
    assert loaded.status == "loaded"
    assert loaded.state is not None
    assert loaded.state.catalog_generation == 0
    file_read_entry = next(
        entry for entry in loaded.state.entries if entry.tool_name == "file_read"
    )
    assert file_read_entry.discovered_turn_id == turn_id
    assert file_read_entry.last_used_turn_id == turn_id
    assert file_read_entry.discovered_turn_index == 1
    assert len(file_read_entry.source_query_hash) == 64
    assert file_read_entry.capability_risk_class == "read_only"

    await engine.close()


@pytest.mark.asyncio
async def test_engine_uses_higher_default_max_iterations_for_local_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = FakeBackend(backend_type="ollama")
    seen_iterations: list[int] = []

    class SpyReActLoop:
        def __init__(self, *args: object, max_iterations: int = 10, **kwargs: object) -> None:
            del args, kwargs
            seen_iterations.append(max_iterations)

        async def run(self, *args: object, **kwargs: object) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            yield FinalAnswerEvent(content="spy reply")

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "agent": {"max_react_iterations": 10},
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory(),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    result = await engine.invoke(
        AgentInvocationRequest(
            message="quick review",
            session_id="iteration-worker",
            tool_mode="disabled",
            execution_profile="subagent_readonly",
            persist_session=False,
        )
    )

    assert result.content == "spy reply"
    assert seen_iterations == [15]

    await engine.close()


@pytest.mark.asyncio
async def test_engine_keeps_contract_rollout_when_preflight_overflow_is_unreliable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = FakeBackend()
    activation_policies: list[dict[str, object]] = []

    class SpyReActLoop:
        def __init__(
            self,
            *args: object,
            tool_execution_context: ToolExecutionContext,
            **kwargs: object,
        ) -> None:
            del args, kwargs
            activation_policies.append(
                dict(tool_execution_context.state["tool_activation_policy"])
            )

        async def run(
            self,
            *args: object,
            **kwargs: object,
        ) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            yield FinalAnswerEvent(content="spy reply")

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(
        config,
        conversation_resolver_factory=_resolver_factory("execution"),
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        del model_spec
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]
    original_estimate_prompt_budget = engine._estimate_prompt_budget  # noqa: SLF001

    def unreliable_preflight_budget(**kwargs: object) -> dict[str, object]:
        budget = original_estimate_prompt_budget(**kwargs)  # type: ignore[arg-type]
        if kwargs.get("tool_schemas") == []:
            return {
                **budget,
                "hard_gate_enabled": False,
                "hard_overflow": True,
                "overflow": True,
            }
        return budget

    monkeypatch.setattr(
        engine,
        "_estimate_prompt_budget",
        unreliable_preflight_budget,
    )
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    result = await engine.invoke(
        AgentInvocationRequest(
            message="Run one read-only execution tool.",
            session_id="unreliable-preflight-overflow",
            persist_session=False,
        )
    )

    adapter = result.diagnostics.tool_exposure["diagnostics"][
        "capability_exposure_adapter"
    ]
    assert adapter["applied"] is True
    assert "exec_command" in adapter["activation_allowed_tool_names"]
    assert result.diagnostics.tool_exposure["diagnostics"]["tool_mode"] == "auto"
    assert len(activation_policies) == 1
    activation_policy = activation_policies[0]
    assert activation_policy["tool_mode"] == "auto"
    assert activation_policy["capability_enforcement_mode"] == "enforce"
    assert "exec_command" in activation_policy["activation_allowed_tool_names"]

    await engine.close()


@pytest.mark.asyncio
async def test_engine_blocks_invocation_when_prompt_exceeds_conservative_fallback_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = FakeBackend(
        metadata={
            "effective_context_length": 128,
            "effective_context_length_source": "auto_num_ctx.fallback_default",
            "context_length_source": "unknown",
            "context_length_fallback": 128,
        }
    )
    react_loop_called = False

    class SpyReActLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            nonlocal react_loop_called
            react_loop_called = True

        async def run(self, *args: object, **kwargs: object) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            yield FinalAnswerEvent(content="should not run")

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> FakeBackend:
        del model_spec
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    result = await engine.invoke(
        AgentInvocationRequest(
            message=" ".join(["oversized"] * 1000),
            session_id="fallback-context-overflow-worker",
            inference_overrides={
                "max_output_tokens": 64,
                "reserve_output_tokens": 64,
            },
            tool_mode="disabled",
            execution_profile="subagent_readonly",
            persist_session=False,
        )
    )

    assert react_loop_called is False
    assert fake_backend.calls == []
    assert result.diagnostics.fallback_reason == "context_overflow"
    final_events = [event for event in result.events if isinstance(event, FinalAnswerEvent)]
    assert final_events[-1].finish_reason == "context_overflow"
    assert final_events[-1].metadata["context_length_source"] == "auto_num_ctx.fallback_default"
    assert final_events[-1].metadata["hard_overflow"] is True
    assert final_events[-1].metadata["runtime_category"] == "context_budget"

    await engine.close()


@pytest.mark.asyncio
async def test_engine_blocks_invocation_when_prompt_exceeds_effective_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = FakeBackend(
        metadata={
            "effective_context_length": 128,
            "effective_context_length_source": "test.effective_context",
        }
    )
    react_loop_called = False

    class SpyReActLoop:
        def __init__(self, *args: object, **kwargs: object) -> None:
            del args, kwargs
            nonlocal react_loop_called
            react_loop_called = True

        async def run(self, *args: object, **kwargs: object) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            yield FinalAnswerEvent(content="should not run")

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> FakeBackend:
        del model_spec
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    result = await engine.invoke(
        AgentInvocationRequest(
            message=" ".join(["oversized"] * 1000),
            session_id="context-overflow-worker",
            inference_overrides={
                "max_output_tokens": 64,
                "reserve_output_tokens": 64,
            },
            tool_mode="disabled",
            execution_profile="subagent_readonly",
            persist_session=False,
        )
    )

    assert react_loop_called is False
    assert fake_backend.calls == []
    assert result.diagnostics.fallback_reason == "context_overflow"
    assert "context window" in result.content
    assert any(
        isinstance(event, StatusEvent)
        and event.metadata.get("reason") == "context_overflow"
        for event in result.events
    )
    final_events = [event for event in result.events if isinstance(event, FinalAnswerEvent)]
    assert final_events[-1].finish_reason == "context_overflow"
    assert final_events[-1].metadata["estimated_prompt_tokens"] > final_events[-1].metadata[
        "context_length"
    ]
    assert final_events[-1].metadata["hard_overflow"] is True
    assert final_events[-1].metadata["runtime_category"] == "context_budget"
    assert final_events[-1].metadata["error_type"] == "context_overflow"
    assert final_events[-1].metadata["recoverability"] == "requires_user_input"

    await engine.close()
