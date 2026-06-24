"""GGUFBackend runtime 測試。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mochi.backends.gguf import GGUFBackend
from mochi.backends.types import Message, ToolCall, ToolSchema


class _FakeGGUFModel:
    """測試用 GGUF 模型假物件。"""

    def __init__(
        self,
        content: str = "hello",
        usage: dict[str, int] | None = None,
    ) -> None:
        self.content = content
        self.usage = usage if usage is not None else {"prompt_tokens": 9, "completion_tokens": 4}
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(
                [
                    {"choices": [{"delta": {"content": "Hel"}, "finish_reason": None}]},
                    {"choices": [{"delta": {"content": "lo"}, "finish_reason": "stop"}]},
                ]
            )
        return {
            **({"usage": self.usage} if self.usage is not None else {}),
            "model": "fake-gguf",
            "choices": [
                {
                    "message": {"content": self.content},
                    "finish_reason": "stop",
                }
            ],
        }


class _FakeGGUFModelWithRuntimeTokenizer(_FakeGGUFModel):
    """提供 runtime tokenize 計數能力的假模型。"""

    def tokenize(self, payload: bytes, add_bos: bool = False) -> list[str]:  # noqa: ARG002
        text = payload.decode("utf-8")
        normalized = " ".join(text.split())
        if not normalized:
            return []
        return normalized.split(" ")


class _FakeGGUFModelWithChatFormat(_FakeGGUFModel):
    """Test double with an explicit chat_format attribute."""

    def __init__(
        self,
        *,
        chat_format: str,
        content: str = "ok",
        usage: dict[str, int] | None = None,
        message: dict | None = None,
        finish_reason: str = "stop",
    ) -> None:
        super().__init__(content=content, usage=usage)
        self.chat_format = chat_format
        self._message = message
        self._finish_reason = finish_reason

    def create_chat_completion(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        return {
            **({"usage": self.usage} if self.usage is not None else {}),
            "model": "fake-gguf",
            "choices": [
                {
                    "message": self._message if self._message is not None else {"content": self.content},
                    "finish_reason": self._finish_reason,
                }
            ],
        }


TOOLS = [
    ToolSchema(
        name="web_search",
        description="search",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
]


def _tool_history_messages() -> list[Message]:
    return [
        Message(role="system", content="You are a helpful assistant."),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="web_search",
                    arguments={"query": "Mochi AI"},
                )
            ],
        ),
        Message(
            role="tool",
            name="web_search",
            tool_call_id="call-1",
            content='{"ok": true, "output": "found: Mochi AI"}',
        ),
        Message(role="user", content="continue"),
    ]


@pytest.mark.asyncio
async def test_gguf_generate_with_injected_model_loader(tmp_path: Path) -> None:
    """GGUFBackend 應可透過注入 loader 執行最小 non-stream 推理。"""
    model_path = tmp_path / "model.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    fake_model = _FakeGGUFModel("GGUF says hi")

    backend = GGUFBackend(
        model_path=str(model_path),
        model_loader=lambda: fake_model,
    )
    backend._dependency_error = None  # noqa: SLF001

    result = await backend.generate([Message(role="user", content="hello")], stream=False)

    assert result.content == "GGUF says hi"
    assert result.model == "fake-gguf"
    assert result.input_tokens == 9
    assert result.output_tokens == 4
    assert fake_model.calls[0]["messages"][0]["content"] == "hello"


@pytest.mark.asyncio
async def test_gguf_generate_fallbacks_to_runtime_tokenize_when_usage_missing(tmp_path: Path) -> None:
    """usage 缺失時應優先使用 runtime tokenize。"""
    model_path = tmp_path / "model.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    fake_model = _FakeGGUFModelWithRuntimeTokenizer(
        content="alpha beta",
        usage={},
    )

    backend = GGUFBackend(
        model_path=str(model_path),
        model_loader=lambda: fake_model,
    )
    backend._dependency_error = None  # noqa: SLF001

    result = await backend.generate([Message(role="user", content="one two three")], stream=False)

    assert result.input_tokens == 3
    assert result.output_tokens == 2


@pytest.mark.asyncio
async def test_gguf_generate_fallbacks_to_deterministic_heuristic_without_usage_or_tokenize(
    tmp_path: Path,
) -> None:
    """usage 與 runtime tokenize 都缺失時應使用可重現啟發式。"""
    model_path = tmp_path / "model.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    fake_model = _FakeGGUFModel(
        content="abcdefghij",
        usage={},
    )

    backend = GGUFBackend(
        model_path=str(model_path),
        model_loader=lambda: fake_model,
    )
    backend._dependency_error = None  # noqa: SLF001

    result = await backend.generate([Message(role="user", content="abcd")], stream=False)

    assert result.input_tokens == 1
    assert result.output_tokens == 3


@pytest.mark.asyncio
async def test_gguf_generate_reports_dependency_missing(tmp_path: Path) -> None:
    """缺依賴時應回報一致錯誤語義。"""
    model_path = tmp_path / "model.gguf"
    model_path.write_text("placeholder", encoding="utf-8")

    backend = GGUFBackend(model_path=str(model_path))
    backend._dependency_error = "missing dependency"  # noqa: SLF001

    with pytest.raises(RuntimeError, match=r"gguf generate unavailable \[dependency_missing\]"):
        await backend.generate([Message(role="user", content="hi")])


@pytest.mark.asyncio
async def test_gguf_stream_mode_not_implemented(tmp_path: Path) -> None:
    """stream 模式應回傳最小可用的 StreamChunk。"""
    model_path = tmp_path / "model.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    fake_model = _FakeGGUFModel()

    backend = GGUFBackend(
        model_path=str(model_path),
        model_loader=lambda: fake_model,
    )
    backend._dependency_error = None  # noqa: SLF001

    stream_iter = await backend.generate([Message(role="user", content="hi")], stream=True)
    chunks = [chunk async for chunk in stream_iter]

    assert [chunk.delta for chunk in chunks] == ["Hel", "lo"]
    assert chunks[-1].is_final is True
    assert chunks[-1].finish_reason == "stop"


def test_gguf_default_model_loader_passes_performance_kwargs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """預設 loader 應將關鍵效能參數傳給 llama-cpp-python。"""
    model_path = tmp_path / "model.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeLlama:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)

    fake_runtime = SimpleNamespace(llama_supports_gpu_offload=lambda: True, _base_path=None)
    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=_FakeLlama, llama_cpp=fake_runtime))
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", fake_runtime)

    backend = GGUFBackend(
        model_path=str(model_path),
        n_ctx=8192,
        n_gpu_layers=99,
        n_threads=6,
        n_batch=1024,
        n_ubatch=512,
        n_threads_batch=4,
        flash_attn=True,
        offload_kqv=True,
        use_mmap=False,
        use_mlock=True,
    )

    backend._default_model_loader()  # noqa: SLF001

    assert captured["model_path"] == str(model_path)
    assert captured["n_ctx"] == 8192
    assert captured["n_gpu_layers"] == 99
    assert captured["n_threads"] == 6
    assert captured["n_batch"] == 1024
    assert captured["n_ubatch"] == 512
    assert captured["n_threads_batch"] == 4
    assert captured["flash_attn"] is True
    assert captured["offload_kqv"] is True
    assert captured["use_mmap"] is False
    assert captured["use_mlock"] is True


def test_gguf_default_model_loader_ignores_external_runtime_library_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GGUF inference must not override the wheel's bundled llama runtime with external DLLs."""
    model_path = tmp_path / "model.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    runtime_dir = tmp_path / "llama-runtime"
    runtime_dir.mkdir()
    (runtime_dir / "llama.dll").write_text("dll", encoding="utf-8")
    captured: dict[str, object] = {}

    class _FakeLlama:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]
            captured.update(kwargs)

    fake_runtime = SimpleNamespace(
        llama_supports_gpu_offload=lambda: True,
        _base_path=runtime_dir,
    )
    fake_llama_cpp_module = SimpleNamespace(Llama=_FakeLlama, llama_cpp=fake_runtime)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp_module)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", fake_runtime)

    backend = GGUFBackend(
        model_path=str(model_path),
        llama_cpp_lib_path=str(runtime_dir),
    )

    backend._default_model_loader()  # noqa: SLF001

    assert os.environ.get("LLAMA_CPP_LIB_PATH") != str(runtime_dir)
    assert captured["model_path"] == str(model_path)


def test_gguf_default_model_loader_rejects_cpu_only_wheel_runtime_when_gpu_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """GPU offload requests must fail fast when the installed llama-cpp-python runtime is CPU-only."""
    model_path = tmp_path / "model.gguf"
    model_path.write_text("placeholder", encoding="utf-8")

    class _FakeLlama:
        def __init__(self, **kwargs):  # type: ignore[no-untyped-def]  # noqa: ARG002
            return None

    fake_runtime = SimpleNamespace(
        llama_supports_gpu_offload=lambda: False,
        _base_path=None,
    )
    fake_llama_cpp_module = SimpleNamespace(Llama=_FakeLlama, llama_cpp=fake_runtime)
    monkeypatch.setitem(sys.modules, "llama_cpp", fake_llama_cpp_module)
    monkeypatch.setitem(sys.modules, "llama_cpp.llama_cpp", fake_runtime)

    backend = GGUFBackend(
        model_path=str(model_path),
        n_gpu_layers=-1,
    )

    with pytest.raises(RuntimeError, match="Installed llama-cpp-python runtime does not support GPU offload"):
        backend._default_model_loader()  # noqa: SLF001


@pytest.mark.asyncio
async def test_gguf_uses_structured_native_for_allowlisted_chat_format(tmp_path: Path) -> None:
    """Allowlisted chat format keeps OpenAI-style tool messages."""
    model_path = tmp_path / "model.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    fake_model = _FakeGGUFModelWithChatFormat(chat_format="functionary-v2")

    backend = GGUFBackend(
        model_path=str(model_path),
        model_loader=lambda: fake_model,
    )
    backend._dependency_error = None  # noqa: SLF001

    await backend.generate(_tool_history_messages(), tools=TOOLS, stream=False)

    sent_messages = fake_model.calls[0]["messages"]
    assert fake_model.calls[0]["tools"] == [tool.to_dict() for tool in TOOLS]
    assert sent_messages[0]["content"] == "You are a helpful assistant."
    assert "## Tool Use Instructions" not in sent_messages[0]["content"]
    assert any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and message.get("name") == "web_search"
        for message in sent_messages
    )
    assert any(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and message.get("tool_calls")
        for message in sent_messages
    )

    info = backend.get_model_info()
    assert info.metadata["tool_call_strategy"] == "structured_native"
    assert info.metadata["detected_chat_format"] == "functionary-v2"
    assert "allowlist" in info.metadata["tool_call_strategy_reason"]


@pytest.mark.asyncio
async def test_gguf_falls_back_to_flattened_text_for_unknown_chat_format(tmp_path: Path) -> None:
    """Unknown chat format should force flattened text strategy."""
    model_path = tmp_path / "model.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    fake_model = _FakeGGUFModelWithChatFormat(chat_format="chatml")

    backend = GGUFBackend(
        model_path=str(model_path),
        model_loader=lambda: fake_model,
    )
    backend._dependency_error = None  # noqa: SLF001

    await backend.generate(_tool_history_messages(), tools=TOOLS, stream=False)

    sent_messages = fake_model.calls[0]["messages"]
    assert "tools" not in fake_model.calls[0]
    assert sent_messages[0]["role"] == "system"
    assert "## Tool Use Instructions" in sent_messages[0]["content"]
    assert all(
        not (isinstance(message, dict) and message.get("role") == "tool")
        for message in sent_messages
    )
    assert all(
        not (isinstance(message, dict) and message.get("tool_calls"))
        for message in sent_messages
    )
    assert any(
        isinstance(message, dict)
        and message.get("role") == "user"
        and "Tool web_search result:" in str(message.get("content", ""))
        for message in sent_messages
    )
    assert any(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and "Tool request: web_search" in str(message.get("content", ""))
        for message in sent_messages
    )

    info = backend.get_model_info()
    assert info.metadata["tool_call_strategy"] == "flattened_text"
    assert info.metadata["detected_chat_format"] == "chatml"
    assert "not in allowlist" in info.metadata["tool_call_strategy_reason"]


@pytest.mark.asyncio
async def test_gguf_falls_back_to_flattened_text_when_chat_format_cannot_be_detected(
    tmp_path: Path,
) -> None:
    """Missing chat_format inspection should force flattened text strategy."""
    model_path = tmp_path / "model.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    fake_model = _FakeGGUFModel()

    backend = GGUFBackend(
        model_path=str(model_path),
        model_loader=lambda: fake_model,
    )
    backend._dependency_error = None  # noqa: SLF001

    await backend.generate(_tool_history_messages(), tools=TOOLS, stream=False)

    sent_messages = fake_model.calls[0]["messages"]
    assert all(
        not (isinstance(message, dict) and message.get("role") == "tool")
        for message in sent_messages
    )
    assert all(
        not (isinstance(message, dict) and message.get("tool_calls"))
        for message in sent_messages
    )

    info = backend.get_model_info()
    assert info.metadata["tool_call_strategy"] == "flattened_text"
    assert info.metadata["detected_chat_format"] == "unknown"
    assert "unable to inspect chat format" in info.metadata["tool_call_strategy_reason"]


@pytest.mark.asyncio
async def test_gguf_parses_native_tool_calls_for_allowlisted_chat_format(tmp_path: Path) -> None:
    """Allowlisted native mode should parse OpenAI-style message.tool_calls."""
    model_path = tmp_path / "model.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    fake_model = _FakeGGUFModelWithChatFormat(
        chat_format="functionary",
        message={
            "content": "I'll call the tool now.",
            "tool_calls": [
                {
                    "id": "",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({"query": "Mochi AI"}, ensure_ascii=False),
                    },
                },
                {
                    "function": {
                        "name": "web_search",
                        "arguments": {"query": "Mochi Framework"},
                    }
                },
            ],
        },
        finish_reason="stop",
    )

    backend = GGUFBackend(
        model_path=str(model_path),
        model_loader=lambda: fake_model,
    )
    backend._dependency_error = None  # noqa: SLF001

    result = await backend.generate(
        [Message(role="user", content="search Mochi")],
        tools=TOOLS,
        stream=False,
    )

    assert result.content == "I'll call the tool now."
    assert len(result.tool_calls) == 2
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "Mochi AI"}
    assert result.tool_calls[0].id
    assert result.tool_calls[1].name == "web_search"
    assert result.tool_calls[1].arguments == {"query": "Mochi Framework"}
    assert result.tool_calls[1].id
    assert result.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_gguf_generate_without_tools_keeps_plain_chat_messages(tmp_path: Path) -> None:
    """No-tools requests should stay as plain chat and avoid simulator injection."""
    model_path = tmp_path / "model.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    fake_model = _FakeGGUFModel(content="plain")

    backend = GGUFBackend(
        model_path=str(model_path),
        model_loader=lambda: fake_model,
    )
    backend._dependency_error = None  # noqa: SLF001

    await backend.generate(
        [
            Message(role="system", content="baseline system"),
            Message(role="user", content="hello"),
        ],
        tools=None,
        stream=False,
    )

    sent_messages = fake_model.calls[0]["messages"]
    assert sent_messages[0]["content"] == "baseline system"
    assert "## Tool Use Instructions" not in sent_messages[0]["content"]
    assert all(not message.get("tool_calls") for message in sent_messages if isinstance(message, dict))
