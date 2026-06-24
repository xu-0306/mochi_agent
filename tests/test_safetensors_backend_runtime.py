"""SafetensorsBackend runtime 測試。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mochi.backends.safetensors import SafetensorsBackend
from mochi.backends.types import Message


class _FakePipeline:
    """測試用 pipeline 假物件。"""

    def __init__(
        self,
        generated_text: str = "hello",
        usage: dict[str, int] | None = None,
    ) -> None:
        self.generated_text = generated_text
        self.usage = usage
        self.calls: list[dict] = []
        self.tokenizer = None

    def __call__(self, prompt: str, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"prompt": prompt, **kwargs})
        payload = {"generated_text": prompt + self.generated_text}
        if self.usage is not None:
            payload["usage"] = self.usage
        return [payload]


class _FakeTokenizer:
    """測試用 chat-template tokenizer。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def apply_chat_template(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"messages": messages, **kwargs})
        return "<chat-template>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens
        return list(range(len(text.split())))


@pytest.mark.asyncio
async def test_safetensors_generate_with_injected_pipeline(tmp_path: Path) -> None:
    """SafetensorsBackend 應可透過注入 pipeline 執行最小 non-stream 推理。"""
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()
    fake_pipeline = _FakePipeline("transformers says hi")

    backend = SafetensorsBackend(
        model_dir=str(model_dir),
        pipeline_factory=lambda: fake_pipeline,
    )
    backend._dependency_error = None  # noqa: SLF001

    result = await backend.generate([Message(role="user", content="hello")], stream=False)

    assert result.content == "transformers says hi"
    assert result.model == str(model_dir)
    assert result.input_tokens == 6
    assert result.output_tokens == 5
    assert "user: hello" in fake_pipeline.calls[0]["prompt"]


@pytest.mark.asyncio
async def test_safetensors_prefers_chat_template_when_available(tmp_path: Path) -> None:
    """若 tokenizer 支援 apply_chat_template，應優先使用。"""
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()
    fake_pipeline = _FakePipeline("template result")
    fake_tokenizer = _FakeTokenizer()
    fake_pipeline.tokenizer = fake_tokenizer

    backend = SafetensorsBackend(
        model_dir=str(model_dir),
        pipeline_factory=lambda: fake_pipeline,
    )
    backend._dependency_error = None  # noqa: SLF001

    result = await backend.generate([Message(role="user", content="hello")], stream=False)

    assert result.content == "template result"
    assert fake_pipeline.calls[0]["prompt"] == "<chat-template>"
    assert fake_tokenizer.calls[0]["add_generation_prompt"] is True
    assert result.input_tokens == 1
    assert result.output_tokens == 2


@pytest.mark.asyncio
async def test_safetensors_prefers_usage_metadata_for_token_accounting(tmp_path: Path) -> None:
    """若 pipeline 回傳 usage，應優先使用。"""
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()
    fake_pipeline = _FakePipeline(
        generated_text="from usage",
        usage={"prompt_tokens": 33, "completion_tokens": 7},
    )

    backend = SafetensorsBackend(
        model_dir=str(model_dir),
        pipeline_factory=lambda: fake_pipeline,
    )
    backend._dependency_error = None  # noqa: SLF001

    result = await backend.generate([Message(role="user", content="hello")], stream=False)

    assert result.input_tokens == 33
    assert result.output_tokens == 7


@pytest.mark.asyncio
async def test_safetensors_generate_reports_dependency_missing(tmp_path: Path) -> None:
    """缺依賴時應回報一致錯誤語義。"""
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()

    backend = SafetensorsBackend(model_dir=str(model_dir))
    backend._dependency_error = "missing dependency"  # noqa: SLF001

    with pytest.raises(
        RuntimeError,
        match=r"safetensors generate unavailable \[dependency_missing\]",
    ):
        await backend.generate([Message(role="user", content="hi")])


@pytest.mark.asyncio
async def test_safetensors_stream_mode_not_implemented(tmp_path: Path) -> None:
    """stream 模式應回傳 pseudo-stream 結果。"""
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()
    fake_pipeline = _FakePipeline("stream result")

    backend = SafetensorsBackend(
        model_dir=str(model_dir),
        pipeline_factory=lambda: fake_pipeline,
    )
    backend._dependency_error = None  # noqa: SLF001

    stream_iter = await backend.generate([Message(role="user", content="hi")], stream=True)
    chunks = [chunk async for chunk in stream_iter]

    assert chunks[0].delta == "stream result"
    assert chunks[-1].is_final is True
    assert chunks[-1].finish_reason == "stop"


def test_safetensors_summarize_device_map_limits_log_size() -> None:
    """device map 摘要應限制輸出大小，避免巨大 log。"""
    backend = SafetensorsBackend(model_dir="/models/demo")
    device_map = {f"layer_{index}": f"cuda:{index % 2}" for index in range(10)}

    summary = backend._summarize_device_map(device_map)  # noqa: SLF001

    assert "layer_0" in summary
    assert "+2 more" in summary
