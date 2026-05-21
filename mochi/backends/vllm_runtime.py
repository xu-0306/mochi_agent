"""Managed vLLM runtime metadata and planning helpers."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
import os
from shlex import quote
import time
from typing import Any, Literal, Protocol
from urllib.parse import urlparse

import httpx

from mochi.config.schema import MochiConfig

DEFAULT_MANAGED_VLLM_HOST = "127.0.0.1"
DEFAULT_MANAGED_VLLM_PORT = 8000


RuntimeState = Literal["disabled", "not_ready", "starting", "ready", "stopped", "failed"]
LaunchMode = Literal["external", "managed"]


@dataclass(slots=True)
class ManagedVLLMProcessMetadata:
    """In-memory process metadata for a managed vLLM runtime."""

    pid: int | None = None
    command: list[str] = field(default_factory=list)
    env_overrides: dict[str, str] = field(default_factory=dict)
    host: str = DEFAULT_MANAGED_VLLM_HOST
    port: int | None = None
    model: str | None = None
    started_at: str | None = None
    exit_code: int | None = None
    error_message: str | None = None

    @property
    def base_url(self) -> str | None:
        """Return `http://host:port/v1` when host and port are available."""
        return build_vllm_base_url(self.host, self.port)


@dataclass(slots=True)
class ManagedVLLMCommandPlan:
    """Serializable command plan for launching managed vLLM later."""

    command: list[str]
    env_overrides: dict[str, str]
    host: str
    port: int
    base_url: str
    warnings: list[str] = field(default_factory=list)

    def command_preview(self) -> str:
        """Return shell-quoted command preview for logs/UI."""
        return " ".join(quote(part) for part in self.command)


@dataclass(slots=True)
class ManagedVLLMRuntimeStatus:
    """Stable status payload for route-level orchestration."""

    state: RuntimeState
    enabled: bool
    launch_mode: LaunchMode | None
    host: str
    port: int | None
    base_url: str | None
    startup_timeout_seconds: int
    pid: int | None
    model: str | None
    last_error: str | None = None
    warnings: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


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
SleepCallable = Callable[[float], Awaitable[None]]
MonotonicCallable = Callable[[], float]


def build_vllm_base_url(host: str, port: int | None) -> str | None:
    """Build OpenAI-compatible base URL for vLLM."""
    normalized_host = host.strip()
    if not normalized_host or port is None:
        return None
    return f"http://{normalized_host}:{port}/v1"


def resolve_managed_vllm_port(
    preferred_port: int | None,
    *,
    fallback_port: int = DEFAULT_MANAGED_VLLM_PORT,
) -> int:
    """Resolve port with deterministic fallback."""
    return preferred_port if preferred_port is not None else fallback_port


def build_managed_vllm_command_plan(
    *,
    model: str,
    host: str = DEFAULT_MANAGED_VLLM_HOST,
    port: int | None = None,
    tensor_parallel_size: int = 1,
    dtype: str = "auto",
    gpu_memory_utilization: float = 0.9,
    max_model_len: int | None = None,
    trust_remote_code: bool = False,
    quantization: str | None = None,
    api_key: str | None = None,
    cuda_visible_devices: str | None = None,
    python_executable: str = "python",
    module: str = "vllm.entrypoints.openai.api_server",
    extra_args: tuple[str, ...] = (),
) -> ManagedVLLMCommandPlan:
    """Create a bounded command plan without launching any process."""
    normalized_model = model.strip()
    if not normalized_model:
        raise ValueError("model must be non-empty")

    normalized_host = host.strip() or DEFAULT_MANAGED_VLLM_HOST
    resolved_port = resolve_managed_vllm_port(port)
    command = [
        python_executable,
        "-m",
        module,
        "--model",
        normalized_model,
        "--host",
        normalized_host,
        "--port",
        str(resolved_port),
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--dtype",
        dtype,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if max_model_len is not None:
        command.extend(["--max-model-len", str(max_model_len)])
    if trust_remote_code:
        command.append("--trust-remote-code")
    if quantization:
        command.extend(["--quantization", quantization])
    command.extend(extra_args)

    env_overrides: dict[str, str] = {}
    if api_key:
        env_overrides["VLLM_API_KEY"] = api_key
    if cuda_visible_devices:
        env_overrides["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    warnings: list[str] = []
    if port is None:
        warnings.append(
            f"Port is not configured; fallback {resolved_port} is used for planning."
        )

    return ManagedVLLMCommandPlan(
        command=command,
        env_overrides=env_overrides,
        host=normalized_host,
        port=resolved_port,
        base_url=build_vllm_base_url(normalized_host, resolved_port) or "",
        warnings=warnings,
    )


def create_process_metadata(
    *,
    command: list[str],
    env_overrides: dict[str, str] | None = None,
    host: str = DEFAULT_MANAGED_VLLM_HOST,
    port: int | None = None,
    model: str | None = None,
    pid: int | None = None,
) -> ManagedVLLMProcessMetadata:
    """Create process metadata snapshot with current UTC timestamp."""
    return ManagedVLLMProcessMetadata(
        pid=pid,
        command=list(command),
        env_overrides=dict(env_overrides or {}),
        host=host,
        port=port,
        model=model,
        started_at=datetime.now(UTC).isoformat(),
    )


def build_managed_vllm_runtime_status(
    *,
    enabled: bool,
    launch_mode: LaunchMode | None,
    host: str = DEFAULT_MANAGED_VLLM_HOST,
    port: int | None = None,
    startup_timeout_seconds: int = 180,
    model: str | None = None,
    process: ManagedVLLMProcessMetadata | None = None,
) -> ManagedVLLMRuntimeStatus:
    """Compute runtime status from config + optional process metadata only."""
    base_url = build_vllm_base_url(host, port)
    warnings: list[str] = []
    actions: list[str] = []

    if not enabled:
        return ManagedVLLMRuntimeStatus(
            state="disabled",
            enabled=False,
            launch_mode=launch_mode,
            host=host,
            port=port,
            base_url=base_url,
            startup_timeout_seconds=startup_timeout_seconds,
            pid=None,
            model=model,
            actions=["enable_vllm_runtime"],
        )

    if base_url is None:
        warnings.append("vLLM base URL is incomplete because port is missing.")
        actions.append("configure_vllm_port")

    if launch_mode == "external":
        actions.append("verify_external_runtime")
        return ManagedVLLMRuntimeStatus(
            state="ready" if base_url else "not_ready",
            enabled=True,
            launch_mode=launch_mode,
            host=host,
            port=port,
            base_url=base_url,
            startup_timeout_seconds=startup_timeout_seconds,
            pid=None,
            model=model,
            warnings=warnings,
            actions=actions,
        )

    if process is None:
        actions.append("plan_managed_runtime")
        return ManagedVLLMRuntimeStatus(
            state="not_ready",
            enabled=True,
            launch_mode=launch_mode,
            host=host,
            port=port,
            base_url=base_url,
            startup_timeout_seconds=startup_timeout_seconds,
            pid=None,
            model=model,
            warnings=warnings,
            actions=actions,
        )

    if process.exit_code is not None:
        state: RuntimeState = "stopped" if process.exit_code == 0 else "failed"
        actions.append("restart_managed_runtime")
        return ManagedVLLMRuntimeStatus(
            state=state,
            enabled=True,
            launch_mode=launch_mode,
            host=host,
            port=port,
            base_url=base_url,
            startup_timeout_seconds=startup_timeout_seconds,
            pid=process.pid,
            model=model or process.model,
            last_error=process.error_message,
            warnings=warnings,
            actions=actions,
        )

    state = "ready" if process.pid is not None and base_url else "starting"
    if state == "starting":
        actions.append("wait_for_runtime_ready")
    return ManagedVLLMRuntimeStatus(
        state=state,
        enabled=True,
        launch_mode=launch_mode,
        host=host,
        port=port,
        base_url=base_url,
        startup_timeout_seconds=startup_timeout_seconds,
        pid=process.pid,
        model=model or process.model,
        last_error=process.error_message,
        warnings=warnings,
        actions=actions,
    )


@dataclass(slots=True)
class _ResolvedManagerConfig:
    enabled: bool = False
    host: str = DEFAULT_MANAGED_VLLM_HOST
    port: int | None = None
    startup_timeout_seconds: int = 180
    tensor_parallel_size: int = 1
    dtype: str = "auto"
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    trust_remote_code: bool = False
    quantization: str | None = None
    api_key: str | None = None
    cuda_visible_devices: str | None = None


async def _default_readiness_probe(
    *,
    base_url: str,
    timeout_seconds: float,
) -> tuple[bool, str | None]:
    models_endpoint = f"{base_url.rstrip('/')}/models"
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds) as client:
            response = await client.get(models_endpoint)
        if response.status_code >= 400:
            body = response.text.strip()
            detail = f"GET {models_endpoint} returned HTTP {response.status_code}"
            if body:
                detail = f"{detail}: {body[:400]}"
            return False, detail
        return True, None
    except httpx.HTTPError as exc:
        return False, f"GET {models_endpoint} failed: {exc}"


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


def _tail_text(buffer: deque[str]) -> str:
    return "\n".join(buffer)


def _extract_secret(value: Any) -> str | None:
    if value is None:
        return None
    get_secret_value = getattr(value, "get_secret_value", None)
    if callable(get_secret_value):
        secret = get_secret_value()
        if isinstance(secret, str) and secret.strip():
            return secret
        return None
    if isinstance(value, str) and value.strip():
        return value
    return None


def _resolve_manager_config(config: MochiConfig | None) -> _ResolvedManagerConfig:
    if config is None:
        return _ResolvedManagerConfig()

    vllm_config = getattr(config, "vllm", None)
    if vllm_config is None:
        return _ResolvedManagerConfig()

    return _ResolvedManagerConfig(
        enabled=bool(getattr(vllm_config, "enabled", False)),
        host=str(getattr(vllm_config, "host", DEFAULT_MANAGED_VLLM_HOST)),
        port=getattr(vllm_config, "port", None),
        startup_timeout_seconds=int(getattr(vllm_config, "startup_timeout_seconds", 180)),
        tensor_parallel_size=int(getattr(vllm_config, "tensor_parallel_size", 1)),
        dtype=str(getattr(vllm_config, "dtype", "auto")),
        gpu_memory_utilization=float(getattr(vllm_config, "gpu_memory_utilization", 0.9)),
        max_model_len=getattr(vllm_config, "max_model_len", None),
        trust_remote_code=bool(getattr(vllm_config, "trust_remote_code", False)),
        quantization=getattr(vllm_config, "quantization", None),
        api_key=_extract_secret(getattr(vllm_config, "api_key", None)),
        cuda_visible_devices=getattr(vllm_config, "cuda_visible_devices", None),
    )


def _resolve_host_port(
    *,
    base_url: str | None,
    fallback_host: str,
    fallback_port: int | None,
) -> tuple[str, int]:
    host = fallback_host.strip() or DEFAULT_MANAGED_VLLM_HOST
    port = fallback_port

    raw_base_url = (base_url or "").strip()
    if raw_base_url:
        parsed = urlparse(raw_base_url)
        if parsed.hostname:
            host = parsed.hostname
        if parsed.port is not None:
            port = parsed.port

    return host, resolve_managed_vllm_port(port)


class ManagedVLLMRuntimeManager:
    """Managed vLLM subprocess runtime manager with readiness probing."""

    def __init__(
        self,
        *,
        launcher: ProcessLauncher | None = None,
        readiness_probe: ReadinessProbe | None = None,
        sleep: SleepCallable = asyncio.sleep,
        monotonic: MonotonicCallable = time.monotonic,
        poll_interval_seconds: float = 1.0,
        shutdown_timeout_seconds: float = 10.0,
        tail_lines: int = 120,
    ) -> None:
        self._launcher = launcher or asyncio.create_subprocess_exec
        self._readiness_probe = readiness_probe or _default_readiness_probe
        self._sleep = sleep
        self._monotonic = monotonic
        self._poll_interval_seconds = poll_interval_seconds
        self._shutdown_timeout_seconds = shutdown_timeout_seconds

        self._lock = asyncio.Lock()
        self._process: _ManagedProcess | None = None
        self._process_metadata: ManagedVLLMProcessMetadata | None = None
        self._stdout_task: asyncio.Task[None] | None = None
        self._stderr_task: asyncio.Task[None] | None = None
        self._stdout_tail: deque[str] = deque(maxlen=tail_lines)
        self._stderr_tail: deque[str] = deque(maxlen=tail_lines)

        self._state: str = "stopped"
        self._message: str | None = None
        self._last_error: str | None = None
        self._active_model_id: str | None = None
        self._active_model_spec: str | None = None
        self._active_base_url: str | None = None

    async def status(self, *, config: MochiConfig | None = None, **_: Any) -> dict[str, Any]:
        async with self._lock:
            self._refresh_process_exit_state_locked()
            return self._status_payload_locked(config=config)

    async def start(
        self,
        *,
        model_id: str | None = None,
        model_spec: str,
        base_url: str | None = None,
        launch_mode: str = "managed",
        config: MochiConfig | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if launch_mode != "managed":
            raise ValueError("Only managed launch mode is supported for vLLM runtime start.")

        normalized_model_spec = model_spec.strip()
        if not normalized_model_spec:
            raise ValueError("Managed vLLM model spec is required.")

        async with self._lock:
            self._refresh_process_exit_state_locked()
            resolved = _resolve_manager_config(config)
            host, port = _resolve_host_port(
                base_url=base_url,
                fallback_host=resolved.host,
                fallback_port=resolved.port,
            )
            effective_base_url = build_vllm_base_url(host, port) or ""

            target_changed = (
                self._active_model_spec != normalized_model_spec
                or self._active_base_url != effective_base_url
                or self._active_model_id != model_id
            )

            if self._has_live_process_locked() and not target_changed:
                return self._status_payload_locked(config=config)

            if self._has_live_process_locked():
                await self._stop_locked(clear_target=False)

            plan = build_managed_vllm_command_plan(
                model=normalized_model_spec,
                host=host,
                port=port,
                tensor_parallel_size=resolved.tensor_parallel_size,
                dtype=resolved.dtype,
                gpu_memory_utilization=resolved.gpu_memory_utilization,
                max_model_len=resolved.max_model_len,
                trust_remote_code=resolved.trust_remote_code,
                quantization=resolved.quantization,
                api_key=resolved.api_key,
                cuda_visible_devices=resolved.cuda_visible_devices,
            )
            env = os.environ.copy()
            env.update(plan.env_overrides)
            process = await self._launcher(
                *plan.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            self._stdout_tail.clear()
            self._stderr_tail.clear()
            self._process = process
            self._process_metadata = create_process_metadata(
                command=plan.command,
                env_overrides=plan.env_overrides,
                host=plan.host,
                port=plan.port,
                model=normalized_model_spec,
                pid=process.pid,
            )
            self._stdout_task = asyncio.create_task(_drain_stream_tail(process.stdout, self._stdout_tail))
            self._stderr_task = asyncio.create_task(_drain_stream_tail(process.stderr, self._stderr_tail))
            self._state = "starting"
            self._message = f"Starting managed vLLM runtime for {normalized_model_spec}."
            self._last_error = None
            self._active_model_id = model_id
            self._active_model_spec = normalized_model_spec
            self._active_base_url = plan.base_url

            timeout_seconds = max(1, resolved.startup_timeout_seconds)
            deadline = self._monotonic() + timeout_seconds
            last_probe_error: str | None = None
            probe_timeout = max(0.5, min(self._poll_interval_seconds, 3.0))

            while True:
                self._refresh_process_exit_state_locked()
                if not self._has_live_process_locked():
                    await self._stop_locked(clear_target=False)
                    detail = self._build_startup_error_detail(
                        prefix="Managed vLLM process exited before readiness probe completed.",
                    )
                    self._state = "failed"
                    self._message = detail
                    self._last_error = detail
                    raise RuntimeError(detail)

                ready, probe_error = await self._readiness_probe(
                    base_url=plan.base_url,
                    timeout_seconds=probe_timeout,
                )
                if ready:
                    self._state = "running"
                    self._message = f"Managed vLLM runtime ready at {plan.base_url}."
                    self._last_error = None
                    return self._status_payload_locked(config=config)

                if probe_error:
                    last_probe_error = probe_error

                if self._monotonic() >= deadline:
                    break
                await self._sleep(self._poll_interval_seconds)

            await self._stop_locked(clear_target=False)
            self._last_error = self._build_startup_error_detail(
                prefix=(
                    f"Managed vLLM runtime did not become ready within {timeout_seconds} seconds."
                    if not last_probe_error
                    else f"Managed vLLM runtime did not become ready within {timeout_seconds} seconds: {last_probe_error}"
                ),
            )
            self._state = "failed"
            self._message = self._last_error
            raise RuntimeError(self._last_error)

    async def stop(self, **_: Any) -> dict[str, Any]:
        async with self._lock:
            await self._stop_locked(clear_target=True)
            self._state = "stopped"
            self._message = "Managed vLLM runtime stopped."
            self._last_error = None
            return self._status_payload_locked(config=None)

    def _has_live_process_locked(self) -> bool:
        return self._process is not None and self._process.returncode is None

    def _refresh_process_exit_state_locked(self) -> None:
        process = self._process
        if process is None or process.returncode is None:
            return

        if self._process_metadata is not None:
            self._process_metadata.exit_code = process.returncode

        if process.returncode == 0:
            if self._state not in {"stopped", "failed"}:
                self._state = "stopped"
                self._message = "Managed vLLM process exited."
            self._process = None
            return

        detail = self._build_startup_error_detail(
            prefix=f"Managed vLLM process exited with code {process.returncode}.",
        )
        self._state = "failed"
        self._message = detail
        self._last_error = detail
        if self._process_metadata is not None:
            self._process_metadata.error_message = detail
        self._process = None

    async def _stop_locked(self, *, clear_target: bool) -> None:
        process = self._process
        if process is not None and process.returncode is None:
            process.terminate()
            try:
                await asyncio.wait_for(process.wait(), timeout=self._shutdown_timeout_seconds)
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

        if process is not None and self._process_metadata is not None:
            self._process_metadata.exit_code = process.returncode

        if self._stdout_task is not None:
            await self._stdout_task
            self._stdout_task = None
        if self._stderr_task is not None:
            await self._stderr_task
            self._stderr_task = None

        self._process = None
        if clear_target:
            self._active_model_id = None
            self._active_model_spec = None
            self._active_base_url = None

    def _status_payload_locked(self, *, config: MochiConfig | None) -> dict[str, Any]:
        resolved = _resolve_manager_config(config)
        fallback_host = resolved.host
        fallback_port = resolve_managed_vllm_port(resolved.port)
        base_url = self._active_base_url or build_vllm_base_url(fallback_host, fallback_port)
        state = self._state
        running = self._has_live_process_locked() and state not in {"failed", "stopped"}
        return {
            "state": state,
            "running": running,
            "launch_mode": "managed",
            "active_model_id": self._active_model_id,
            "active_model_spec": self._active_model_spec,
            "base_url": base_url,
            "message": self._message,
            "last_error": self._last_error,
            "pid": self._process.pid if self._process is not None else None,
            "stdout_tail": _tail_text(self._stdout_tail),
            "stderr_tail": _tail_text(self._stderr_tail),
            "command": list(self._process_metadata.command) if self._process_metadata is not None else [],
        }

    def _build_startup_error_detail(self, *, prefix: str) -> str:
        details: list[str] = [prefix]
        if self._process_metadata is not None and self._process_metadata.command:
            details.append("command: " + " ".join(quote(part) for part in self._process_metadata.command))
        stderr_tail = _tail_text(self._stderr_tail)
        stdout_tail = _tail_text(self._stdout_tail)
        if stderr_tail:
            details.append(f"stderr tail:\n{stderr_tail[-2000:]}")
        if stdout_tail:
            details.append(f"stdout tail:\n{stdout_tail[-2000:]}")
        return " | ".join(details)
