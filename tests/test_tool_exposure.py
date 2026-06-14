from __future__ import annotations

from collections.abc import AsyncIterator

from mochi.agents.engine import AgentEngine
from mochi.agents.tool_exposure import ToolExposurePlanner
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import AttachmentRef, GenerationResult, Message, ModelInfo, StreamChunk, ToolSchema
from mochi.config.schema import MochiConfig


class _FakeBackend(BaseLLMBackend):
    def __init__(self, backend_type: str = "openai_compat", metadata: dict | None = None) -> None:
        self._backend_type = backend_type
        self._metadata = metadata or {}

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
        return ModelInfo(name="fake", backend_type=self._backend_type, metadata=dict(self._metadata))

    async def health_check(self) -> bool:
        return True


def test_tool_exposure_includes_workspace_baseline_for_chinese_prompt() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["glob_search", "grep_search", "file_read", "file_write"],
        }
    )
    message = "請檢查目前工作區，找出包含 TODO 的地方，並查看相關內容"

    plan = planner.plan(
        message=message,
        available_tool_names=["glob_search", "grep_search", "file_read", "file_write"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )

    assert "read" not in message.lower()
    assert "file" not in message.lower()
    assert {"file_read", "glob_search", "grep_search"} <= set(plan.tool_names)


def test_tool_exposure_attachment_bias_works_with_engine_structured_attachment_header() -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": ".",
            "sessions_dir": "./.tmp-test-sessions",
            "memory": {"db_path": "./.tmp-test-memory.db", "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "exec_command",
                "file_read",
                "docx_read",
                "pdf_read",
                "file_write",
                "file_edit",
                "execute_code",
            ],
        }
    )
    planner_message = engine._build_tool_planner_message(  # noqa: SLF001
        "請先檢查附件內容再回答",
        [
            AttachmentRef(
                name="report.docx",
                path=".mochi/workspace/browser-imports/report.docx",
                source="workspace_file",
            )
        ],
    )

    plan = planner.plan(
        message=planner_message,
        available_tool_names=[
            "exec_command",
            "file_read",
            "docx_read",
            "pdf_read",
            "file_write",
            "file_edit",
            "execute_code",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="trusted_workspace",
        attachment_count=1,
    )

    assert "Structured attachments:" in planner_message
    assert {"file_read", "pdf_read", "docx_read"} <= set(plan.tool_names)
    assert not {"exec_command", "file_write", "file_edit", "execute_code"} & set(plan.tool_names)


def test_tool_exposure_attached_docx_edit_intent_keeps_write_tools_available() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "file_read",
                "docx_read",
                "file_write",
                "file_edit",
                "apply_patch",
            ],
        }
    )

    plan = planner.plan(
        message="Update the attached docx and save the revised version in the workspace.",
        available_tool_names=["file_read", "docx_read", "file_write", "file_edit", "apply_patch"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="trusted_workspace",
        attachment_count=1,
    )

    assert {"file_read", "docx_read", "file_write", "file_edit", "apply_patch"} <= set(plan.tool_names)
