"""AgentEngine Phase 5 learning integration tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from mochi.agents.engine import AgentEngine
from mochi.agents.events import FinalAnswerEvent
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, StreamChunk, ToolCall
from mochi.config.schema import MochiConfig
from mochi.learning.types import Skill
from mochi.tools.base import BaseTool, ToolResult


class _LearningBackend(BaseLLMBackend):
    """Fake backend that supports one tool-using turn plus skill extraction."""

    def __init__(self) -> None:
        self.chat_calls: list[list[Message]] = []
        self.extraction_calls = 0

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        **_: object,
    ) -> GenerationResult | AsyncIterator[StreamChunk] | str:
        del tools, temperature, max_tokens, stream
        if messages and messages[0].content.startswith("Extract one reusable skill"):
            self.extraction_calls += 1
            return """
            {
              "skill_id": "skill-debug",
              "name": "Debug with fake tool",
              "description": "Use fake_tool to inspect the problem before answering.",
              "trigger_keywords": ["debug", "fake_tool"],
              "preconditions": "A debugging task is requested.",
              "steps": ["Call fake_tool", "Use the observation in the final answer"],
              "tools_used": ["fake_tool"]
            }
            """

        self.chat_calls.append(messages)
        if len(self.chat_calls) == 1:
            return GenerationResult(
                content="I need to inspect this.",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="fake_tool",
                        arguments={"query": "debug"},
                    )
                ],
            )
        return GenerationResult(content="debug complete")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="learning-fake", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _MultiToolLearningBackend(BaseLLMBackend):
    """Fake backend that requires two tool calls before answering."""

    def __init__(self) -> None:
        self.chat_calls: list[list[Message]] = []
        self.extraction_calls = 0

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        **_: object,
    ) -> GenerationResult | str:
        del tools, temperature, max_tokens, stream
        if messages and messages[0].content.startswith("Extract one reusable skill"):
            self.extraction_calls += 1
            return """
            {
              "skill_id": "skill-debug-multi",
              "name": "Debug with repeated inspection",
              "description": "Use fake_tool multiple times before answering.",
              "trigger_keywords": ["debug", "inspection"],
              "preconditions": "A debugging task is requested.",
              "steps": ["Call fake_tool twice", "Use the observations in the final answer"],
              "tools_used": ["fake_tool"]
            }
            """

        self.chat_calls.append(messages)
        if len(self.chat_calls) == 1:
            return GenerationResult(
                content="Need a first inspection.",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="fake_tool",
                        arguments={"query": "phase-1"},
                    )
                ],
            )
        if len(self.chat_calls) == 2:
            return GenerationResult(
                content="Need a second inspection.",
                tool_calls=[
                    ToolCall(
                        id="call-2",
                        name="fake_tool",
                        arguments={"query": "phase-2"},
                    )
                ],
            )
        return GenerationResult(content="debug complete")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="learning-fake-multi", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _PromptCaptureBackend(BaseLLMBackend):
    """Fake backend that records prompts and returns a final answer."""

    def __init__(self) -> None:
        self.chat_calls: list[list[Message]] = []

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        **_: object,
    ) -> GenerationResult:
        del tools, temperature, max_tokens, stream
        self.chat_calls.append(messages)
        return GenerationResult(content="ok")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="prompt-capture", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _ConversationOnlyBackend(BaseLLMBackend):
    """Fake backend that only returns a final answer with no tool usage."""

    def __init__(self) -> None:
        self.chat_calls: list[list[Message]] = []
        self.extraction_calls = 0

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        **_: object,
    ) -> GenerationResult | str:
        del tools, temperature, max_tokens, stream
        if messages and messages[0].content.startswith("Extract one reusable skill"):
            self.extraction_calls += 1
            return '{"name":"Should not be learned","steps":["Do not store this."]}'
        self.chat_calls.append(messages)
        return GenerationResult(content="Here is a direct answer.")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="conversation-only", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _FakeTool(BaseTool):
    @property
    def name(self) -> str:
        return "fake_tool"

    @property
    def description(self) -> str:
        return "Inspect fake debugging context."

    @property
    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {"query": {"type": "string"}}}

    async def execute(self, **kwargs) -> ToolResult:  # noqa: ANN003
        return ToolResult(output=f"observed:{kwargs.get('query', '')}")


def _build_skill(
    *,
    skill_id: str,
    name: str,
    description: str,
    tools_used: list[str] | None = None,
    body: str = "",
) -> Skill:
    return Skill(
        skill_id=skill_id,
        name=name,
        description=description,
        trigger_keywords=[],
        preconditions="",
        steps=["Use the selected skill."],
        tools_used=tools_used or [],
        source_trajectory_id="",
        body=body,
    )


@pytest.mark.asyncio
async def test_engine_does_not_extract_skill_when_tool_threshold_is_not_met(tmp_path: Path) -> None:
    backend = _LearningBackend()
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path / "workspace"),
            "sessions_dir": str(tmp_path / "sessions"),
            "skills_dir": str(tmp_path / "skills"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "security": {
                "require_approval_for_exec": False,
                "require_approval_for_file_write": False,
            },
            "learning": {"min_steps_for_extraction": 3},
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> _LearningBackend:
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = fake_load  # type: ignore[method-assign]
    engine._tool_registry.register(_FakeTool())  # noqa: SLF001

    first_events = [event async for event in engine.chat("debug this", session_id="learn-s1")]
    first_final = next(event for event in first_events if isinstance(event, FinalAnswerEvent))

    assert first_final.trajectory_id is not None
    assert backend.extraction_calls == 0
    assert (tmp_path / "workspace" / "trajectories.jsonl").exists()
    learned_skills = [
        skill for skill in await engine.list_skills() if getattr(skill, "source_type", "learned") == "learned"
    ]
    assert learned_skills == []

    second_events = [event async for event in engine.chat("debug again", session_id="learn-s1")]
    assert any(isinstance(event, FinalAnswerEvent) for event in second_events)
    second_system_prompt = backend.chat_calls[-1][0].content
    assert "Debug with fake tool" not in second_system_prompt

    await engine.close()


@pytest.mark.asyncio
async def test_engine_extracts_skill_when_tool_call_threshold_is_met(tmp_path: Path) -> None:
    backend = _MultiToolLearningBackend()
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path / "workspace"),
            "sessions_dir": str(tmp_path / "sessions"),
            "skills_dir": str(tmp_path / "skills"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "security": {
                "require_approval_for_exec": False,
                "require_approval_for_file_write": False,
            },
            "learning": {
                "min_steps_for_extraction": 3,
                "min_tool_calls_for_extraction": 2,
            },
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> _MultiToolLearningBackend:
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = fake_load  # type: ignore[method-assign]
    engine._tool_registry.register(_FakeTool())  # noqa: SLF001

    events = [event async for event in engine.chat("debug this thoroughly", session_id="learn-s4")]

    assert any(isinstance(event, FinalAnswerEvent) for event in events)
    assert backend.extraction_calls == 1
    skill_names = [skill.name for skill in await engine.list_skills()]
    assert "Debug with repeated inspection" in skill_names

    await engine.close()


@pytest.mark.asyncio
async def test_engine_auto_syncs_filesystem_skill_into_prompt(tmp_path: Path) -> None:
    backend = _PromptCaptureBackend()
    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "skill-installer"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: skill-installer
description: Install skills without manual DB import.
tags: [skills, install]
---

# Skill Installer

Custom filesystem skill unique phrase.
""",
        encoding="utf-8",
    )
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path / "workspace"),
            "sessions_dir": str(tmp_path / "sessions"),
            "skills_dir": str(skills_dir),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "learning": {"auto_extract_skills": False},
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> _LearningBackend:
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    events = [event async for event in engine.chat("How do I install skills?", session_id="learn-s2")]

    assert any(isinstance(event, FinalAnswerEvent) for event in events)
    system_prompt = backend.chat_calls[-1][0].content
    assert "skill-installer" in system_prompt
    assert "# Skill Installer" in system_prompt
    assert "Custom filesystem skill unique phrase." in system_prompt

    await engine.close()


@pytest.mark.asyncio
async def test_engine_explicit_selected_skill_is_injected_into_prompt(tmp_path: Path) -> None:
    backend = _PromptCaptureBackend()
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path / "workspace"),
            "sessions_dir": str(tmp_path / "sessions"),
            "skills_dir": str(tmp_path / "skills"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "learning": {
                "auto_extract_skills": False,
                "auto_sync_filesystem_skills": False,
            },
        }
    )
    engine = AgentEngine(config)
    await engine._skill_library.add(  # noqa: SLF001
        _build_skill(
            skill_id="skill-selected",
            name="Selected Exec Skill",
            description="Use exec_command carefully for deterministic inspection.",
            tools_used=["exec_command"],
            body="# Selected Exec Skill\n\nExplicitly selected skill unique phrase.",
        )
    )

    async def fake_load(model_spec: str) -> _LearningBackend:
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    events = [
        event
        async for event in engine.chat(
            "Tell me a joke about turtles.",
            session_id="learn-s5",
            selected_skill_ids=["skill-selected"],
        )
    ]

    assert any(isinstance(event, FinalAnswerEvent) for event in events)
    system_prompt = backend.chat_calls[-1][0].content
    assert "Selected Exec Skill" in system_prompt
    assert "Explicitly selected skill unique phrase." in system_prompt

    await engine.close()


@pytest.mark.asyncio
async def test_engine_does_not_extract_skill_for_conversation_only_success(tmp_path: Path) -> None:
    backend = _ConversationOnlyBackend()
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path / "workspace"),
            "sessions_dir": str(tmp_path / "sessions"),
            "skills_dir": str(tmp_path / "skills"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "learning": {
                "min_steps_for_extraction": 1,
            },
            "security": {
                "require_approval_for_exec": False,
                "require_approval_for_file_write": False,
            },
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> _ConversationOnlyBackend:
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    events = [event async for event in engine.chat("What is recursion?", session_id="learn-s3")]

    assert any(isinstance(event, FinalAnswerEvent) for event in events)
    assert backend.extraction_calls == 0
    learned_skills = [
        skill for skill in await engine.list_skills() if getattr(skill, "source_type", "learned") == "learned"
    ]
    assert learned_skills == []

    await engine.close()
