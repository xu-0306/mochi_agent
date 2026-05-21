"""GGUF backend powered by an external llama.cpp runtime."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
import socket
import time
from typing import Any, Protocol

import httpx

from mochi.backends.base import BaseLLMBackend
from mochi.backends.openai_compat import OpenAICompatBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, StreamChunk, ToolSchema


DEFAULT_LLAMA_SERVER_HOST = "127.0.0.1"
DEFAULT_LLAMA_SERVER_STARTUP_TIMEOUT_SECONDS = 180.0


class _ManagedProcess(Protocol):
    pid: int | None
    returncode: int | None
    stdout: asyncio.StreamReader | None
    stderr: asyncio.StreamReader | None

    async def wait(self) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


ProcessLauncher = Callable[..., Awaitable[_ManagedProcess]]
ReadinessProbe = Callable[..., Awaitable[tuple[bool, str | None]]]
ModelNameResolver = Callable[..., Awaitable[str]]
PortResolver = Callable[[], int]
OpenAIBackendFactory = Callable[..., OpenAICompatBackend]


@dataclass(slots=True)
class _RuntimeState:
    process: _ManagedProcess | None = None
    delegate: OpenAICompatBackend | None = None
    base_url: str | None = None
    model_name: str | None = None
    command: list[str] | None = None
    stdout_task: asyncio.Task[None] | None = None
    stderr_task: asyncio.Task[None] | None = None


async def _default_readiness_probe(
    *,
    base_url: str,
    timeout_seconds: float,
) -> tuple[bool, str | None]:
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(f"{base_url.rstrip('/')}/models")
        if response.status_code >= 400:
            return False, f"GET {base_url.rstrip('/')}/models returned HTTP {response.status_code}"
        return True, None
    except httpx.HTTPError as exc:
        return False, f"GET {base_url.rstrip('/')}/models failed: {exc}"


async def _default_model_name_resolver(
    *,
    base_url: str,
    timeout_seconds: float,
) -> str:
    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        response = await client.get(f"{base_url.rstrip('/')}/models")
        response.raise_for_status()
    payload = response.json()
    data = payload.get("data", [])
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            model_id = first.get("id")
            if isinstance(model_id, str) and model_id.strip():
                return model_id.strip()
    return "default"


def _default_port_resolver() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((DEFAULT_LLAMA_SERVER_HOST, 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


async def _drain_stream_tail(
    stream: asyncio.StreamReader | None,
    buffer: deque[str],
) -> None:
    if stream is None:
        return

    while True:
        line = await stream.readline()
        if not line:
            break
        text = line.decode("utf-8", errors="replace").rstrip()
        if text:
            buffer.append(text)


class LlamaCppServerBackend(BaseLLMBackend):
    """GGUF backend that proxies requests through external `llama-server`."""

    def __init__(
        self,
        model_path: str,
        runtime_root: str | None,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        n_batch: int = 512,
        n_ubatch: int = 512,
        n_threads_batch: int | None = None,
        flash_attn: bool = False,
        offload_kqv: bool = True,
        use_mmap: bool = True,
        use_mlock: bool = False,
        *,
        host: str = DEFAULT_LLAMA_SERVER_HOST,
        startup_timeout_seconds: float = DEFAULT_LLAMA_SERVER_STARTUP_TIMEOUT_SECONDS,
        launcher: ProcessLauncher | None = None,
        readiness_probe: ReadinessProbe | None = None,
        model_name_resolver: ModelNameResolver | None = None,
        openai_backend_factory: OpenAIBackendFactory | None = None,
        port_resolver: PortResolver | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.model_path = model_path
        self.runtime_root = runtime_root
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.n_ubatch = n_ubatch
        self.n_threads_batch = n_threads_batch
        self.flash_attn = flash_attn
        self.offload_kqv = offload_kqv
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.host = host
        self.startup_timeout_seconds = startup_timeout_seconds
        self._launcher = launcher or asyncio.create_subprocess_exec
        self._readiness_probe = readiness_probe or _default_readiness_probe
        self._model_name_resolver = model_name_resolver or _default_model_name_resolver
        self._openai_backend_factory = openai_backend_factory or self._default_openai_backend_factory
        self._port_resolver = port_resolver or _default_port_resolver
        self._sleep = sleep
        self._monotonic = monotonic
        self._lock = asyncio.Lock()
        self._stdout_tail: deque[str] = deque(maxlen=120)
        self._stderr_tail: deque[str] = deque(maxlen=120)
        self._state = _RuntimeState()
        self._server_binary_path = self._resolve_server_binary_path()
        self._dependency_error = self._probe_dependency_error()

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
        delegate = await self._ensure_server_ready()
        return await delegate.generate(
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            stream=stream,
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model_path,
            backend_type="gguf",
            context_length=self.n_ctx,
            supports_tool_calling=True,
            metadata={
                "model_path": self.model_path,
                "loaded": self._state.delegate is not None,
                "dependency_error": self._dependency_error,
                "runtime_root": self.runtime_root,
                "server_binary_path": str(self._server_binary_path) if self._server_binary_path else None,
                "base_url": self._state.base_url,
                "pid": self._state.process.pid if self._state.process is not None else None,
                "startup_timeout_seconds": self.startup_timeout_seconds,
            },
        )

    async def health_check(self) -> bool:
        if self._dependency_error is not None:
            return False
        if not Path(self.model_path).is_file():
            return False
        if self._state.delegate is None:
            return True
        return await self._state.delegate.health_check()

    async def close(self) -> None:
        async with self._lock:
            await self._stop_server_locked()

    async def _ensure_server_ready(self) -> OpenAICompatBackend:
        if self._dependency_error is not None:
            raise self._build_generate_error("dependency_missing", self._dependency_error)

        model_file = Path(self.model_path)
        if not model_file.is_file():
            raise self._build_generate_error(
                "model_path_missing",
                f"GGUF model file not found: {self.model_path}",
            )

        async with self._lock:
            if self._state.delegate is not None and self._state.process is not None and self._state.process.returncode is None:
                return self._state.delegate

            await self._stop_server_locked()

            port = self._port_resolver()
            base_url = f"http://{self.host}:{port}/v1"
            command = self._build_server_command(port)

            try:
                process = await self._launcher(
                    *command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except Exception as exc:
                raise self._build_generate_error(
                    "runtime_start_failed",
                    f"Failed to launch llama-server: {exc}",
                ) from exc

            self._stdout_tail.clear()
            self._stderr_tail.clear()
            self._state.process = process
            self._state.command = command
            self._state.base_url = base_url
            self._state.stdout_task = asyncio.create_task(_drain_stream_tail(process.stdout, self._stdout_tail))
            self._state.stderr_task = asyncio.create_task(_drain_stream_tail(process.stderr, self._stderr_tail))

            deadline = self._monotonic() + self.startup_timeout_seconds
            last_error: str | None = None
            while self._monotonic() < deadline:
                if process.returncode is not None:
                    detail = self._build_startup_error_detail(
                        f"llama-server exited with code {process.returncode} before readiness."
                    )
                    await self._stop_server_locked()
                    raise self._build_generate_error("runtime_start_failed", detail)

                ready, probe_error = await self._readiness_probe(
                    base_url=base_url,
                    timeout_seconds=2.0,
                )
                if ready:
                    model_name = await self._model_name_resolver(
                        base_url=base_url,
                        timeout_seconds=5.0,
                    )
                    delegate = self._openai_backend_factory(
                        base_url=base_url,
                        model=model_name,
                    )
                    self._state.delegate = delegate
                    self._state.model_name = model_name
                    return delegate

                if probe_error:
                    last_error = probe_error
                await self._sleep(0.5)

            detail = self._build_startup_error_detail(
                "llama-server did not become ready within the startup timeout."
                if not last_error
                else f"llama-server did not become ready within the startup timeout: {last_error}"
            )
            await self._stop_server_locked()
            raise self._build_generate_error("runtime_start_failed", detail)

    async def _stop_server_locked(self) -> None:
        delegate = self._state.delegate
        if delegate is not None:
            await delegate.close()
        self._state.delegate = None
        self._state.model_name = None

        process = self._state.process
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        if self._state.stdout_task is not None:
            await self._state.stdout_task
        if self._state.stderr_task is not None:
            await self._state.stderr_task

        self._state = _RuntimeState()

    def _resolve_server_binary_path(self) -> Path | None:
        if not self.runtime_root:
            return None

        root = Path(self.runtime_root).expanduser().resolve(strict=False)
        candidates = [
            root / "llama-server.exe",
            root / "llama-server",
            root / "build" / "bin" / "llama-server.exe",
            root / "build" / "bin" / "llama-server",
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _probe_dependency_error(self) -> str | None:
        if not self.runtime_root:
            return "Configured llama.cpp runtime is unavailable for GGUF inference: runtime root is not set."
        if self._server_binary_path is None:
            return (
                "Configured llama.cpp runtime does not contain `llama-server`; "
                f"runtime root: {self.runtime_root}"
            )
        return None

    def _build_server_command(self, port: int) -> list[str]:
        if self._server_binary_path is None:
            raise RuntimeError("llama-server binary path is unavailable.")

        command = [
            str(self._server_binary_path),
            "--model",
            self.model_path,
            "--host",
            self.host,
            "--port",
            str(port),
            "--ctx-size",
            str(self.n_ctx),
            "--n-gpu-layers",
            str(self.n_gpu_layers),
        ]
        if self.n_threads is not None:
            command.extend(["--threads", str(self.n_threads)])
        if self.flash_attn:
            command.extend(["--flash-attn", "on"])
        if not self.use_mmap:
            command.append("--no-mmap")
        if self.use_mlock:
            command.append("--mlock")
        return command

    def _default_openai_backend_factory(self, *, base_url: str, model: str) -> OpenAICompatBackend:
        return OpenAICompatBackend(base_url=base_url, model=model)

    def _build_startup_error_detail(self, prefix: str) -> str:
        details = [prefix]
        if self._state.command:
            details.append("command: " + " ".join(self._state.command))
        if self._stderr_tail:
            details.append("stderr tail:\n" + "\n".join(self._stderr_tail))
        if self._stdout_tail:
            details.append("stdout tail:\n" + "\n".join(self._stdout_tail))
        return " | ".join(details)

    def _build_generate_error(self, code: str, detail: str) -> RuntimeError:
        return RuntimeError(f"gguf generate unavailable [{code}]: {detail}")
