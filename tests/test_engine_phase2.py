from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from mochi.agents.engine import AgentEngine
from mochi.agents.invocation import AgentInvocationDiagnostics, AgentInvocationRequest
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import AttachmentRef, GenerationResult, Message, ModelInfo, StreamChunk
from mochi.config.schema import MochiConfig


class FakeBackend(BaseLLMBackend):
    def __init__(self, backend_type: str = "test", metadata: dict | None = None) -> None:
        self.calls: list[list[Message]] = []
        self.tool_calls_seen: list[list[str]] = []
        self.closed = False
        self.backend_type = backend_type
        self.metadata = metadata or {}

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
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
        self.calls.append(messages)
        self.tool_calls_seen.append([tool.name for tool in tools or []])
        return GenerationResult(content="fake reply")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="fake", backend_type=self.backend_type, metadata=dict(self.metadata))

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


def test_agent_invocation_diagnostics_to_dict_serializes_tool_exposure() -> None:
    diagnostics = AgentInvocationDiagnostics(
        execution_profile="chat",
        tool_mode="auto",
        exposed_tools=["file_read"],
        matched_tool_groups=["workspace"],
    )
    diagnostics.tool_exposure = {
        "exposed_tools": ["file_read"],
        "workspace_bound": True,
        "attachment_count": 2,
    }

    assert diagnostics.to_dict()["tool_exposure"] == diagnostics.tool_exposure


@pytest.mark.asyncio
async def test_engine_invoke_serializes_tool_exposure_metadata_in_diagnostics(
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
    engine = AgentEngine(config)

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

    diagnostics = result.diagnostics.to_dict()
    assert diagnostics["tool_exposure"] == {
        "exposed_tools": result.diagnostics.exposed_tools,
        "workspace_bound": True,
        "attachment_count": 2,
    }

    await engine.close()
