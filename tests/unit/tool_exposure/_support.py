from __future__ import annotations

from collections.abc import AsyncIterator

from mochi.agents.tool_intent_router import (
    ToolIntentRoute,
)
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, StreamChunk, ToolSchema
from mochi.tools.base import BaseTool, ToolResult


def _tool_capabilities(*tool_names: str) -> dict[str, dict]:
    capabilities: dict[str, dict] = {}
    for tool_name in tool_names:
        if tool_name == "web_search":
            capabilities[tool_name] = {
                "domains": ["web"],
                "retrieval_modes": ["search"],
                "preference_tags": ["open_web", "source_discovery"],
                "read_only": True,
                "open_world": True,
            }
        elif tool_name == "web_fetch":
            capabilities[tool_name] = {
                "domains": ["web"],
                "retrieval_modes": ["fetch"],
                "preference_tags": ["open_web", "source_reading"],
                "read_only": True,
                "open_world": True,
            }
        elif tool_name == "arxiv_search":
            capabilities[tool_name] = {
                "domains": ["literature"],
                "retrieval_modes": ["search"],
                "preference_tags": ["scholarly_index", "paper_metadata", "recent_papers"],
                "read_only": True,
                "open_world": True,
            }
        elif tool_name == "semantic_scholar_search":
            capabilities[tool_name] = {
                "domains": ["literature"],
                "retrieval_modes": ["search"],
                "preference_tags": ["scholarly_index", "paper_metadata", "citations", "recent_papers"],
                "read_only": True,
                "open_world": True,
            }
        elif tool_name == "crossref_search":
            capabilities[tool_name] = {
                "domains": ["literature"],
                "retrieval_modes": ["search"],
                "preference_tags": ["citation_lookup", "doi_lookup", "bibliographic_metadata"],
                "read_only": True,
                "open_world": True,
            }
        elif tool_name == "pubmed_search":
            capabilities[tool_name] = {
                "domains": ["literature"],
                "retrieval_modes": ["search"],
                "preference_tags": ["scholarly_index", "paper_metadata", "biomedical"],
                "read_only": True,
                "open_world": True,
            }
        elif tool_name == "repo_map":
            capabilities[tool_name] = {
                "domains": ["workspace"],
                "retrieval_modes": ["repo_map", "symbol_index"],
                "preference_tags": ["repo_navigation", "code_discovery"],
                "read_only": True,
                "open_world": False,
            }
        elif tool_name == "read_symbol":
            capabilities[tool_name] = {
                "domains": ["workspace"],
                "retrieval_modes": ["symbol_lookup", "targeted_read"],
                "preference_tags": ["repo_navigation", "definition_lookup"],
                "read_only": True,
                "open_world": False,
            }
    return capabilities


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


class _FakeToolIntentClassifier:
    def __init__(self, result: ToolIntentRoute | Exception) -> None:
        self._result = result

    async def classify(
        self,
        *,
        user_message: str,
        session_bound_workspace: bool,
        attachment_count: int,
        workspace_attachment_count: int,
    ) -> ToolIntentRoute:
        del user_message, session_bound_workspace, attachment_count, workspace_attachment_count
        if isinstance(self._result, Exception):
            raise self._result
        return self._result


class _DummyTool(BaseTool):
    def __init__(self, name: str, description: str, *, search_hint: str | None = None) -> None:
        self._name = name
        self._description = description
        self._search_hint = search_hint

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict[str, object]:
        return {"type": "object", "properties": {}, "additionalProperties": False}

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def search_hint(self) -> str | None:
        return self._search_hint

    async def execute(self, **kwargs: object) -> ToolResult:
        del kwargs
        return ToolResult(output=self._name)
