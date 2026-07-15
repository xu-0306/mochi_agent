from __future__ import annotations

from pathlib import Path

import pytest

from mochi.agents.engine import AgentEngine
from mochi.agents.invocation import AgentInvocationRequest
from mochi.agents.events import FinalAnswerEvent, StatusEvent, ThinkingEvent, ToolCallResultEvent
from mochi.agents.react_loop import AsyncReActLoop
from mochi.backends.base import BackendRequestError, BaseLLMBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, ToolCall, ToolSchema
from mochi.config.schema import MochiConfig
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.file_ops import FileReadTool
from mochi.tools.literature_search import ArxivSearchTool
from mochi.tools.registry import ToolRegistry
from mochi.tools.tool_result_read import ToolResultReadTool


class _JsonLookingFileReadBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

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
    ) -> GenerationResult:
        del tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        self.calls.append(messages)
        tool_message = next((message for message in messages if message.role == "tool"), None)
        if tool_message is not None:
            assert tool_message.content == '{"name": "mochi"}'
            assert not tool_message.content.startswith('{"ok":')
            return GenerationResult(content="done")
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-file-read-json",
                    name="file_read",
                    arguments={"path": "sample.json", "line_numbers": False},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="json-file-read-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _LargeFileReadBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

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
    ) -> GenerationResult:
        del tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        self.calls.append(messages)
        tool_messages = [message for message in messages if message.role == "tool"]
        tool_message = tool_messages[-1] if tool_messages else None
        if tool_message is not None:
            if "Reference: " in tool_message.content and 'tool_result_read(reference_id="' in tool_message.content:
                reference_id = (
                    tool_message.content.split('tool_result_read(reference_id="', 1)[1].split('"', 1)[0]
                )
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-tool-result-read-continue",
                            name="tool_result_read",
                            arguments={"reference_id": reference_id, "offset": 61, "limit": 3, "line_numbers": True},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            assert tool_message.content == "61: line 61\n62: line 62\n63: line 63"
            return GenerationResult(content="done")
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-file-read-large",
                    name="file_read",
                    arguments={"path": "large.txt", "offset": 1, "limit": 60, "line_numbers": True},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="large-file-read-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _UnavailableToolBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

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
    ) -> GenerationResult:
        del tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        self.calls.append(messages)
        tool_message = next((message for message in messages if message.role == "tool"), None)
        if tool_message is not None:
            assert "exec_command" in tool_message.content
            assert "not available in this turn" in tool_message.content
            assert "file_read" in tool_message.content
            return GenerationResult(content="Recovered with visible tools only.")
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-hidden-exec",
                    name="exec_command",
                    arguments={"cmd": "python transform.py"},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="unavailable-tool-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _ChannelMarkerBackend(BaseLLMBackend):
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
    ) -> GenerationResult:
        del messages, tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        return GenerationResult(content="<|channel|>thought<channel|>Visible final answer.")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="channel-marker-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _AnalysisTagBackend(BaseLLMBackend):
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
    ) -> GenerationResult:
        del messages, tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        return GenerationResult(content="<analysis>Inspect attachment schema first.</analysis>Visible final answer.")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="analysis-tag-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _RoleSentinelBackend(BaseLLMBackend):
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
    ) -> GenerationResult:
        del messages, tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        return GenerationResult(
            content="<|im_start|>assistant<|im_end|>\nVisible final answer.",
            thinking="<｜Assistant｜>Review the attached records first.",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="role-sentinel-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _ThinkingOnlyAfterToolBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

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
    ) -> GenerationResult:
        del tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        self.calls.append(messages)

        if any(
            message.role == "user"
            and "no user-visible final answer" in message.content
            for message in messages
        ):
            return GenerationResult(content="Recovered visible answer.")

        if any(message.role == "tool" for message in messages):
            return GenerationResult(
                content="",
                thinking="I have enough information to answer now.",
            )

        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-file-read-thinking-only",
                    name="file_read",
                    arguments={"path": "sample.txt", "line_numbers": False},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="thinking-only-after-tool-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _InvalidToolTurnAfterToolBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[list[Message], bool]] = []

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
    ) -> GenerationResult:
        del temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        self.calls.append((messages, bool(tools)))

        if any(
            message.role == "user"
            and "no user-visible final answer" in message.content
            for message in messages
        ):
            assert not tools
            return GenerationResult(content="Recovered visible answer.")

        if any(message.role == "tool" for message in messages):
            raise BackendRequestError(
                "Prompt-simulated tool calling returned an invalid tool-eligible turn.",
                metadata={
                    "backend_name": "test",
                    "tool_turn_reason": "thinking_only",
                    "rejected_thinking": "I have enough information to answer now.",
                },
            )

        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-file-read-invalid-tool-turn",
                    name="file_read",
                    arguments={"path": "sample.txt", "line_numbers": False},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="invalid-tool-turn-after-tool-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _InvalidInitialToolTurnOllamaBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[list[Message], bool]] = []

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
    ) -> GenerationResult:
        del temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        self.calls.append((messages, bool(tools)))

        if any(message.role == "tool" for message in messages):
            return GenerationResult(content="Recovered with tool call.")

        if any(
            message.role == "user"
            and "invalid for a tool-capable turn" in message.content
            for message in messages
        ):
            assert tools
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-file-read-repair",
                        name="file_read",
                        arguments={"path": "sample.txt", "line_numbers": False},
                    )
                ],
                finish_reason="tool_calls",
            )

        raise BackendRequestError(
            "Ollama returned an invalid tool-eligible turn.",
            metadata={
                "backend_name": "ollama",
                "tool_turn_reason": "thinking_only",
                "rejected_thinking": "I should inspect the file before answering.",
            },
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="invalid-initial-tool-turn-ollama-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _InvalidInitialToolTurnOllamaLengthBackend(_InvalidInitialToolTurnOllamaBackend):
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
    ) -> GenerationResult:
        del temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        self.calls.append((messages, bool(tools)))

        if any(message.role == "tool" for message in messages):
            return GenerationResult(content="Recovered with length-limited tool call.")

        if any(
            message.role == "user"
            and "invalid for a tool-capable turn" in message.content
            for message in messages
        ):
            assert tools
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-file-read-length-repair",
                        name="file_read",
                        arguments={"path": "sample.txt", "line_numbers": False},
                    )
                ],
                finish_reason="tool_calls",
            )

        raise BackendRequestError(
            "Ollama returned an invalid native tool-eligible turn.",
            metadata={
                "backend_name": "ollama",
                "tool_turn_reason": "thinking_only",
                "rejected_finish_reason": "length",
                "rejected_thinking": "I should inspect the file before answering.",
            },
        )

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="invalid-initial-tool-turn-ollama-length-backend", backend_type="test")


class _EmptyFinalAfterToolBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.calls: list[tuple[list[Message], bool]] = []

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
    ) -> GenerationResult:
        del temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        self.calls.append((messages, bool(tools)))

        if any(
            message.role == "user"
            and "no user-visible final answer" in message.content
            for message in messages
        ):
            assert not tools
            return GenerationResult(content="Recovered visible answer.")

        if any(message.role == "tool" for message in messages):
            return GenerationResult(content="")

        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-file-read-empty-final",
                    name="file_read",
                    arguments={"path": "sample.txt", "line_numbers": False},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="empty-final-after-tool-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _TwoStepLiteratureBackend(BaseLLMBackend):
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
    ) -> GenerationResult:
        del tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        tool_messages = [message for message in messages if message.role == "tool"]
        if len(tool_messages) >= 2:
            return GenerationResult(content="Synthesized from two searches.")
        if tool_messages:
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-semantic-scholar",
                        name="semantic_scholar_search",
                        arguments={"query": "meteorological forecast model fine tuning"},
                    )
                ],
                finish_reason="tool_calls",
            )
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-arxiv",
                    name="arxiv_search",
                    arguments={"query": "weather forecast model fine-tuning"},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="two-step-literature-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _OvereagerLiteratureBackend(BaseLLMBackend):
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
    ) -> GenerationResult:
        del tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        tool_messages = [message for message in messages if message.role == "tool"]
        if len(tool_messages) >= 3:
            return GenerationResult(content="Synthesized after runtime steering.")
        if len(tool_messages) == 2:
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-pubmed-after-ready",
                        name="pubmed_search",
                        arguments={"query": "weather model fine-tuning clinical literature"},
                    )
                ],
                finish_reason="tool_calls",
            )
        if len(tool_messages) == 1:
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-semantic-scholar",
                        name="semantic_scholar_search",
                        arguments={"query": "meteorological forecast model fine tuning"},
                    )
                ],
                finish_reason="tool_calls",
            )
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-arxiv",
                    name="arxiv_search",
                    arguments={"query": "weather forecast model fine-tuning"},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="overeager-literature-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _FakeLiteratureSearchTool(BaseTool):
    def __init__(self, name: str, results: list[dict[str, str]]) -> None:
        self._name = name
        self._results = results

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "Search literature."

    @property
    def parameters_schema(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }

    async def execute(self, **kwargs: object) -> ToolResult:
        del kwargs
        return ToolResult(output=list(self._results))


class _FollowupSchemaBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.tool_names_seen: list[list[str]] = []

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
    ) -> GenerationResult:
        del messages, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        self.tool_names_seen.append([tool.name for tool in tools or []])
        return GenerationResult(content="follow-up complete")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="followup-schema-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _LiteratureReinjectionBackend(BaseLLMBackend):
    def __init__(self) -> None:
        self.tool_messages: list[str] = []

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
    ) -> GenerationResult:
        del tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        tool_message = next((message for message in messages if message.role == "tool"), None)
        if tool_message is not None:
            self.tool_messages.append(tool_message.content)
            assert not tool_message.content.lstrip().startswith("{")
            assert '"ok": true' not in tool_message.content
            assert "Attention Is All You Need" in tool_message.content
            assert "Ashish Vaswani" in tool_message.content
            assert "http://arxiv.org/abs/1706.03762v7" in tool_message.content
            return GenerationResult(content="citation-first reinjection confirmed")
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-arxiv-citation-format",
                    name="arxiv_search",
                    arguments={"query": "attention is all you need"},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="literature-reinjection-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_react_loop_preserves_small_json_looking_file_read_text(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.json").write_text('{"name": "mochi"}', encoding="utf-8")

    backend = _JsonLookingFileReadBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir=tmp_path))
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=ToolExecutionContext(workspace_dir=str(tmp_path)),
        max_iterations=3,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="read the json file",
        )
    ]

    result_event = next(event for event in events if isinstance(event, ToolCallResultEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert result_event.result == '{"name": "mochi"}'
    assert final_event.content == "done"


@pytest.mark.asyncio
async def test_react_loop_persists_oversized_file_read_before_dead_end_preview(
    tmp_path: Path,
) -> None:
    (tmp_path / "large.txt").write_text(
        "".join(f"line {idx}\n" for idx in range(1, 80)),
        encoding="utf-8",
    )

    backend = _LargeFileReadBackend()
    context = ToolExecutionContext(
        workspace_dir=str(tmp_path),
        session_id="session-file-read-large",
        tool_result_store_dir=str(tmp_path / "tool-results"),
    )
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir=tmp_path))
    registry.register(ToolResultReadTool(workspace_dir=tmp_path))
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=context,
        max_iterations=3,
        max_tool_message_chars=220,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="read the large file",
        )
    ]

    result_event = next(event for event in events if isinstance(event, ToolCallResultEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))
    transport = result_event.metadata["transport"]
    reference_id = transport["reference_id"]

    assert reference_id
    assert transport["overflow_persisted"] is True
    assert context.tool_result_references[reference_id]["source_path"] == str(
        (tmp_path / "large.txt").resolve(strict=False)
    )
    assert context.tool_result_references[reference_id]["reference_id"] == reference_id
    assert final_event.content == "done"


@pytest.mark.asyncio
async def test_react_loop_converts_hidden_tool_calls_into_recoverable_tool_errors() -> None:
    backend = _UnavailableToolBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir="."))
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=ToolExecutionContext(workspace_dir="."),
        max_iterations=3,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="Process the attached file.",
        )
    ]

    result_event = next(event for event in events if isinstance(event, ToolCallResultEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert result_event.error is not None
    assert "not available in this turn" in result_event.error
    assert "file_read" in result_event.error
    assert final_event.content == "Recovered with visible tools only."


@pytest.mark.asyncio
async def test_react_loop_strips_channel_reasoning_markers_from_final_answer() -> None:
    backend = _ChannelMarkerBackend()
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=ToolRegistry(discover_builtin=False),
        tool_execution_context=ToolExecutionContext(workspace_dir="."),
        max_iterations=1,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="Answer directly.",
        )
    ]

    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert final_event.content == "Visible final answer."


@pytest.mark.asyncio
async def test_react_loop_extracts_analysis_tags_into_reasoning_events() -> None:
    backend = _AnalysisTagBackend()
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=ToolRegistry(discover_builtin=False),
        tool_execution_context=ToolExecutionContext(workspace_dir="."),
        max_iterations=1,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="Answer directly.",
        )
    ]

    thinking_event = next(event for event in events if isinstance(event, ThinkingEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert thinking_event.content == "Inspect attachment schema first."
    assert final_event.content == "Visible final answer."


@pytest.mark.asyncio
async def test_react_loop_strips_role_sentinels_from_visible_and_reasoning_text() -> None:
    backend = _RoleSentinelBackend()
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=ToolRegistry(discover_builtin=False),
        tool_execution_context=ToolExecutionContext(workspace_dir="."),
        max_iterations=1,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="Answer directly.",
        )
    ]

    thinking_event = next(event for event in events if isinstance(event, ThinkingEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert thinking_event.content == "Review the attached records first."
    assert final_event.content == "Visible final answer."


@pytest.mark.asyncio
async def test_react_loop_recovers_when_backend_returns_thinking_only_after_tool(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.txt").write_text("tool payload", encoding="utf-8")

    backend = _ThinkingOnlyAfterToolBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir=tmp_path))
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=ToolExecutionContext(workspace_dir=str(tmp_path)),
        max_iterations=4,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="read the file and answer",
        )
    ]

    thinking_event = next(event for event in events if isinstance(event, ThinkingEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert thinking_event.content == "I have enough information to answer now."
    assert final_event.content == "Recovered visible answer."
    assert len(backend.calls) == 3
    assert any(
        message.role == "user" and "no user-visible final answer" in message.content
        for message in backend.calls[-1]
    )


@pytest.mark.asyncio
async def test_react_loop_recovers_when_backend_returns_empty_final_after_tool(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.txt").write_text("tool payload", encoding="utf-8")

    backend = _EmptyFinalAfterToolBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir=tmp_path))
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=ToolExecutionContext(workspace_dir=str(tmp_path)),
        max_iterations=4,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="read the file and answer",
        )
    ]

    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert final_event.content == "Recovered visible answer."
    assert len(backend.calls) == 3
    assert backend.calls[-1][1] is False
    assert any(
        message.role == "user" and "no user-visible final answer" in message.content
        for message in backend.calls[-1][0]
    )


@pytest.mark.asyncio
async def test_react_loop_recovers_when_backend_raises_invalid_tool_turn_after_tool(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.txt").write_text("tool payload", encoding="utf-8")

    backend = _InvalidToolTurnAfterToolBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir=tmp_path))
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=ToolExecutionContext(workspace_dir=str(tmp_path)),
        max_iterations=4,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="read the file and answer",
        )
    ]

    thinking_event = next(event for event in events if isinstance(event, ThinkingEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert thinking_event.content == "I have enough information to answer now."
    assert final_event.content == "Recovered visible answer."
    assert len(backend.calls) == 3
    assert backend.calls[-1][1] is False
    assert any(
        message.role == "user" and "no user-visible final answer" in message.content
        for message in backend.calls[-1][0]
    )


@pytest.mark.asyncio
async def test_react_loop_repairs_initial_invalid_ollama_tool_turn_without_disabling_tools(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.txt").write_text("tool payload", encoding="utf-8")

    backend = _InvalidInitialToolTurnOllamaBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir=tmp_path))
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=ToolExecutionContext(workspace_dir=str(tmp_path)),
        max_iterations=4,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="read the file and answer",
        )
    ]

    thinking_event = next(event for event in events if isinstance(event, ThinkingEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert thinking_event.content == "I should inspect the file before answering."
    assert final_event.content == "Recovered with tool call."
    assert len(backend.calls) == 3
    assert backend.calls[1][1] is True
    assert any(
        message.role == "user" and "invalid for a tool-capable turn" in message.content
        for message in backend.calls[1][0]
    )


@pytest.mark.asyncio
async def test_react_loop_repairs_initial_length_limited_ollama_tool_turn_with_tools(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.txt").write_text("tool payload", encoding="utf-8")

    backend = _InvalidInitialToolTurnOllamaLengthBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir=tmp_path))
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=ToolExecutionContext(workspace_dir=str(tmp_path)),
        max_iterations=4,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="read the file and answer",
        )
    ]

    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert final_event.content == "Recovered with length-limited tool call."
    assert len(backend.calls) == 3
    assert backend.calls[1][1] is True
    assert any(
        message.role == "user" and "invalid for a tool-capable turn" in message.content
        for message in backend.calls[1][0]
    )


@pytest.mark.asyncio
async def test_literature_guard_allows_second_search_after_single_arxiv_batch() -> None:
    backend = _TwoStepLiteratureBackend()
    arxiv_results = [
        {
            "title": f"Space weather validation paper {index}",
            "summary": "A paper about space weather forecast validation.",
        }
        for index in range(10)
    ]
    semantic_results = [
        {
            "title": "Fine-tuning meteorological forecast models",
            "abstract": "A relevant paper about model adaptation for weather prediction.",
        }
    ]
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_FakeLiteratureSearchTool("arxiv_search", arxiv_results))
    registry.register(_FakeLiteratureSearchTool("semantic_scholar_search", semantic_results))
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=ToolExecutionContext(workspace_dir="."),
        max_iterations=4,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="Find papers about weather forecast model fine-tuning.",
        )
    ]

    tool_results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert [event.tool_name for event in tool_results] == ["arxiv_search", "semantic_scholar_search"]
    assert all(event.error is None for event in tool_results)
    assert final_event.content == "Synthesized from two searches."


@pytest.mark.asyncio
async def test_literature_guard_reports_runtime_steering_without_tool_error() -> None:
    backend = _OvereagerLiteratureBackend()
    arxiv_results = [
        {
            "title": f"Space weather validation paper {index}",
            "summary": "A paper about space weather forecast validation.",
        }
        for index in range(10)
    ]
    semantic_results = [
        {
            "title": "Fine-tuning meteorological forecast models",
            "abstract": "A relevant paper about model adaptation for weather prediction.",
        }
    ]
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_FakeLiteratureSearchTool("arxiv_search", arxiv_results))
    registry.register(_FakeLiteratureSearchTool("semantic_scholar_search", semantic_results))
    registry.register(_FakeLiteratureSearchTool("pubmed_search", []))
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=ToolExecutionContext(workspace_dir="."),
        max_iterations=5,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="Find papers about weather forecast model fine-tuning.",
        )
    ]

    tool_results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    steering_events = [
        event
        for event in events
        if isinstance(event, StatusEvent)
        and event.metadata.get("reason") == "runtime_steering"
        and event.metadata.get("guard") == "literature_summary_ready"
    ]
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert [event.tool_name for event in tool_results] == [
        "arxiv_search",
        "semantic_scholar_search",
        "pubmed_search",
    ]
    guarded_result = tool_results[-1]
    assert guarded_result.error is None
    assert guarded_result.metadata["status"] == "runtime_steering"
    assert guarded_result.metadata["steering_reason"] == "evidence_sufficient"
    assert steering_events[0].metadata["runtime_category"] == "runtime_steering"
    assert steering_events[0].metadata["error_type"] == "runtime_steering"
    assert steering_events[0].metadata["recoverability"] == "recovered"
    assert "Sufficient literature evidence" in str(guarded_result.result)
    assert steering_events
    assert steering_events[0].metadata["tool_name"] == "pubmed_search"
    assert final_event.content == "Synthesized after runtime steering."


@pytest.mark.asyncio
async def test_engine_followup_backend_schema_includes_tool_result_read_after_overflow_reference(
    tmp_path: Path,
) -> None:
    backend = _FollowupSchemaBackend()
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

    async def fake_load(model_spec: str) -> _FollowupSchemaBackend:
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    session_id = "followup-schema"
    context = engine._get_tool_execution_context(  # noqa: SLF001
        session_id=session_id,
        workspace_dir=str(tmp_path),
    )
    context.tool_result_references["file_read-overflow"] = {
        "reference_id": "file_read-overflow",
        "artifact_path": str(tmp_path / "overflow.txt"),
        "source_path": str(tmp_path / "large.txt"),
    }

    await engine.invoke(
        AgentInvocationRequest(
            message="find recent papers about weather forecast model fine-tuning",
            session_id=session_id,
            workspace_dir=str(tmp_path),
            tool_mode="auto",
            execution_profile="subagent_research",
            persist_session=False,
        )
    )

    assert backend.tool_names_seen
    assert "tool_result_read" in backend.tool_names_seen[-1]

    await engine.close()


@pytest.mark.asyncio
async def test_react_loop_reinjects_literature_results_as_citation_first_plain_text() -> None:
    backend = _LiteratureReinjectionBackend()
    registry = ToolRegistry(discover_builtin=False)
    tool = ArxivSearchTool()
    registry.register(tool)
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=ToolExecutionContext(workspace_dir="."),
        max_iterations=3,
    )

    original_execute = tool.execute

    async def fake_execute(**kwargs: object) -> ToolResult:
        del kwargs
        return ToolResult(
            output=[
                {
                    "id": "1706.03762v7",
                    "title": "Attention Is All You Need",
                    "authors": ["Ashish Vaswani", "Noam Shazeer"],
                    "summary": "We propose a new simple network architecture.",
                    "published": "2017-06-12T17:57:34Z",
                    "updated": "2023-08-02T00:00:00Z",
                    "categories": ["cs.CL", "cs.LG"],
                    "primary_category": "cs.CL",
                    "url": "http://arxiv.org/abs/1706.03762v7",
                    "pdf_url": "http://arxiv.org/pdf/1706.03762v7",
                }
            ]
        )

    tool.execute = fake_execute  # type: ignore[method-assign]
    try:
        events = [
            event
            async for event in react_loop.run(
                system_prompt="system",
                history=[],
                user_message="find the paper and summarize it",
            )
        ]
    finally:
        tool.execute = original_execute  # type: ignore[method-assign]
        await tool.close()

    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))
    assert backend.tool_messages
    assert final_event.content == "citation-first reinjection confirmed"
