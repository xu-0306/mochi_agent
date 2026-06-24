from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

from mochi.backends.vllm_runtime import ManagedVLLMRuntimeManager
from mochi.config.schema import MochiConfig


def _build_config(*, startup_timeout_seconds: int = 5) -> MochiConfig:
    return MochiConfig.model_validate(
        {
            "vllm": {
                "enabled": True,
                "host": "127.0.0.1",
                "port": 8000,
                "startup_timeout_seconds": startup_timeout_seconds,
            }
        }
    )


class _FakeProcess:
    def __init__(
        self,
        *,
        pid: int,
        returncode: int | None = None,
        stdout_lines: Sequence[str] = (),
        stderr_lines: Sequence[str] = (),
    ) -> None:
        self.pid = pid
        self.returncode = returncode
        self.terminate_calls = 0
        self.kill_calls = 0
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        for line in stdout_lines:
            self.stdout.feed_data((line + "\n").encode("utf-8"))
        for line in stderr_lines:
            self.stderr.feed_data((line + "\n").encode("utf-8"))
        self.stdout.feed_eof()
        self.stderr.feed_eof()
        self._waiter: asyncio.Future[int] = asyncio.get_running_loop().create_future()
        if self.returncode is not None:
            self._waiter.set_result(self.returncode)

    async def wait(self) -> int:
        return await self._waiter

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = 0
        if not self._waiter.done():
            self._waiter.set_result(0)

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9
        if not self._waiter.done():
            self._waiter.set_result(-9)


@pytest.mark.asyncio
async def test_managed_vllm_runtime_start_launches_and_becomes_ready() -> None:
    launch_calls: list[dict[str, Any]] = []
    process = _FakeProcess(pid=4321)
    probe_results = iter([(False, "booting"), (True, None)])
    sleep_calls: list[float] = []

    async def _launcher(*cmd: str, stdout: object, stderr: object, env: dict[str, str]) -> _FakeProcess:
        launch_calls.append({"cmd": list(cmd), "stdout": stdout, "stderr": stderr, "env": env})
        return process

    async def _probe(*, base_url: str, timeout_seconds: float) -> tuple[bool, str | None]:
        _ = timeout_seconds
        result = next(probe_results)
        return result[0], result[1] if result[0] else f"{result[1]} @ {base_url}"

    async def _sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    manager = ManagedVLLMRuntimeManager(
        launcher=_launcher,
        readiness_probe=_probe,
        sleep=_sleep,
        poll_interval_seconds=0.01,
    )

    status = await manager.start(
        model_id="vllm-managed-qwen",
        model_spec="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1",
        config=_build_config(),
    )

    assert len(launch_calls) == 1
    launched_cmd = launch_calls[0]["cmd"]
    assert launched_cmd[0].endswith("python")
    assert launched_cmd[1:4] == ["-m", "vllm.entrypoints.openai.api_server", "--model"]
    assert status["running"] is True
    assert status["state"] == "running"
    assert status["active_model_id"] == "vllm-managed-qwen"
    assert status["active_model_spec"] == "Qwen/Qwen2.5-7B-Instruct"
    assert status["base_url"] == "http://localhost:8000/v1"
    assert sleep_calls


@pytest.mark.asyncio
async def test_managed_vllm_runtime_start_replaces_existing_target() -> None:
    process_a = _FakeProcess(pid=101)
    process_b = _FakeProcess(pid=202)
    processes = [process_a, process_b]
    launch_count = 0

    async def _launcher(*_cmd: str, stdout: object, stderr: object, env: dict[str, str]) -> _FakeProcess:
        nonlocal launch_count
        _ = (stdout, stderr, env)
        launch_count += 1
        return processes.pop(0)

    async def _probe(*, base_url: str, timeout_seconds: float) -> tuple[bool, str | None]:
        _ = (base_url, timeout_seconds)
        return True, None

    manager = ManagedVLLMRuntimeManager(launcher=_launcher, readiness_probe=_probe)

    await manager.start(
        model_id="first",
        model_spec="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1",
        config=_build_config(),
    )
    await manager.start(
        model_id="second",
        model_spec="meta-llama/Llama-3.1-8B-Instruct",
        base_url="http://localhost:8000/v1",
        config=_build_config(),
    )
    status = await manager.status(config=_build_config())

    assert launch_count == 2
    assert process_a.terminate_calls == 1
    assert status["running"] is True
    assert status["active_model_id"] == "second"
    assert status["active_model_spec"] == "meta-llama/Llama-3.1-8B-Instruct"


@pytest.mark.asyncio
async def test_managed_vllm_runtime_startup_timeout_sets_failed_status() -> None:
    process = _FakeProcess(pid=333)
    clock = {"now": 0.0}

    async def _launcher(*_cmd: str, stdout: object, stderr: object, env: dict[str, str]) -> _FakeProcess:
        _ = (stdout, stderr, env)
        return process

    async def _probe(*, base_url: str, timeout_seconds: float) -> tuple[bool, str | None]:
        _ = timeout_seconds
        return False, f"probe not ready @ {base_url}"

    async def _sleep(seconds: float) -> None:
        clock["now"] += seconds

    manager = ManagedVLLMRuntimeManager(
        launcher=_launcher,
        readiness_probe=_probe,
        sleep=_sleep,
        monotonic=lambda: clock["now"],
        poll_interval_seconds=1.0,
    )

    with pytest.raises(RuntimeError, match="did not become ready"):
        await manager.start(
            model_id="timeout-model",
            model_spec="Qwen/Qwen2.5-7B-Instruct",
            base_url="http://localhost:8000/v1",
            config=_build_config(startup_timeout_seconds=2),
        )

    status = await manager.status(config=_build_config(startup_timeout_seconds=2))
    assert status["running"] is False
    assert status["state"] == "failed"
    assert "probe not ready" in (status["message"] or "")
    assert process.terminate_calls == 1


@pytest.mark.asyncio
async def test_managed_vllm_runtime_stop_clears_active_target() -> None:
    process = _FakeProcess(pid=987)

    async def _launcher(*_cmd: str, stdout: object, stderr: object, env: dict[str, str]) -> _FakeProcess:
        _ = (stdout, stderr, env)
        return process

    async def _probe(*, base_url: str, timeout_seconds: float) -> tuple[bool, str | None]:
        _ = (base_url, timeout_seconds)
        return True, None

    manager = ManagedVLLMRuntimeManager(launcher=_launcher, readiness_probe=_probe)
    await manager.start(
        model_id="managed-qwen",
        model_spec="Qwen/Qwen2.5-7B-Instruct",
        base_url="http://localhost:8000/v1",
        config=_build_config(),
    )

    stopped = await manager.stop()

    assert process.terminate_calls == 1
    assert stopped["running"] is False
    assert stopped["state"] == "stopped"
    assert stopped["active_model_id"] is None
    assert stopped["active_model_spec"] is None
