from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest

from mochi.backends.llama_cpp_server import LlamaCppServerBackend
from mochi.backends.types import GenerationResult, Message, StreamChunk


class _FakeProcess:
    def __init__(self) -> None:
        self.pid = 1234
        self.returncode: int | None = None
        self.stdout = None
        self.stderr = None
        self.terminated = False
        self.killed = False

    async def wait(self) -> int:
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


class _FakeDelegate:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def generate(self, **kwargs: Any) -> GenerationResult | AsyncIterator[StreamChunk]:
        self.calls.append(kwargs)
        if kwargs.get("stream"):
            async def _iterator() -> AsyncIterator[StreamChunk]:
                yield StreamChunk(delta="hi")
                yield StreamChunk(is_final=True, finish_reason="stop")

            return _iterator()
        return GenerationResult(
            content="ok",
            input_tokens=1,
            output_tokens=1,
            model="server-model",
            finish_reason="stop",
        )

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


def test_llama_cpp_server_backend_reports_missing_runtime(tmp_path: Path) -> None:
    model_path = tmp_path / "demo.gguf"
    model_path.write_text("x", encoding="utf-8")

    backend = LlamaCppServerBackend(
        model_path=str(model_path),
        runtime_root=None,
    )

    assert backend.get_model_info().metadata["dependency_error"] is not None


@pytest.mark.asyncio
async def test_llama_cpp_server_backend_starts_runtime_and_delegates_generate(tmp_path: Path) -> None:
    model_path = tmp_path / "demo.gguf"
    model_path.write_text("x", encoding="utf-8")
    runtime_root = tmp_path / "llama-runtime"
    runtime_root.mkdir()
    (runtime_root / "llama-server.exe").write_text("bin", encoding="utf-8")

    launched: list[list[str]] = []
    fake_process = _FakeProcess()
    fake_delegate = _FakeDelegate()

    async def _fake_launcher(*command: str, **_: Any) -> _FakeProcess:
        launched.append(list(command))
        return fake_process

    async def _ready_probe(*, base_url: str, timeout_seconds: float) -> tuple[bool, str | None]:
        assert base_url.startswith("http://127.0.0.1:")
        assert timeout_seconds > 0
        return True, None

    async def _resolve_model_name(*, base_url: str, timeout_seconds: float) -> str:
        assert base_url.startswith("http://127.0.0.1:")
        assert timeout_seconds > 0
        return "server-model"

    backend = LlamaCppServerBackend(
        model_path=str(model_path),
        runtime_root=str(runtime_root),
        n_ctx=8192,
        n_gpu_layers=-1,
        n_threads=6,
        flash_attn=True,
        launcher=_fake_launcher,
        readiness_probe=_ready_probe,
        model_name_resolver=_resolve_model_name,
        openai_backend_factory=lambda **_: fake_delegate,
        port_resolver=lambda: 18080,
    )

    result = await backend.generate([Message(role="user", content="hello")], stream=False)

    assert result.content == "ok"
    assert launched
    assert launched[0][0].endswith("llama-server.exe")
    assert "--model" in launched[0]
    assert str(model_path) in launched[0]
    assert "--ctx-size" in launched[0]
    assert "8192" in launched[0]
    assert "--n-gpu-layers" in launched[0]
    assert "--flash-attn" in launched[0]
    assert fake_delegate.calls[0]["messages"][0].content == "hello"

    await backend.close()
    assert fake_process.terminated is True
    assert fake_delegate.closed is True
