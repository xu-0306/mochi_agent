from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from mochi.agents.multi_agent.execution_coordinator import (
    _collect_detached_exec_jobs,
    _prepare_detached_exec_layout,
)
from mochi.tools import exec_command as exec_command_module
from mochi.tools.exec_command import ExecCommandTool, get_shared_exec_runtime
from mochi.utils.shell_providers import BaseShellProvider, SubprocessSpec


class _PythonDirectProvider(BaseShellProvider):
    @property
    def canonical_name(self) -> str:
        return "test"

    @property
    def aliases(self) -> tuple[str, ...]:
        return ("test",)

    def build_subprocess_spec(self, command: str, *, tty: bool = False) -> SubprocessSpec:
        del tty
        script = _SCRIPT_BY_COMMAND.get(command)
        if script is None:
            raise ValueError(f"Unsupported test command: {command}")
        return SubprocessSpec(executable=sys.executable, args=("-c", script))


_SCRIPT_BY_COMMAND = {
    "bg": "import time;print('started', flush=True);time.sleep(5)",
}


def test_prepare_detached_exec_layout_returns_recovery_paths(tmp_path: Path) -> None:
    layout = _prepare_detached_exec_layout(str(tmp_path), "req-123")

    assert layout is not None
    assert Path(layout["root_dir"]).is_dir()
    assert Path(layout["checkpoint_dir"]).is_dir()
    assert Path(layout["log_path"]).name == "session.log"
    assert Path(layout["session_log_path"]) == Path(layout["log_path"])
    assert Path(layout["manifest_path"]).name == "manifest.json"
    assert Path(layout["stdout_log_path"]).name == "stdout.log"
    assert Path(layout["stderr_log_path"]).name == "stderr.log"
    assert Path(layout["runtime_state_root"]).name == "exec-runtime"


def test_collect_detached_exec_jobs_marks_recoverable_and_reattachable() -> None:
    execution_results = [
        {
            "request_id": "req-1",
            "status": "running",
            "command": "python worker.py",
            "shell": "powershell",
            "workdir": "H:/_python/agent_mochi",
            "timeout": 30,
            "background": True,
            "log_path": "H:/tmp/session.log",
            "checkpoint_dir": "H:/tmp/checkpoints",
            "metadata": {
                "session_id": "exec-1",
                "pid": 123,
                "approval_state": "not_required",
                "detached": True,
                "restored": False,
                "supports_stdin": False,
                "recovery_supported": True,
                "session_log_path": "H:/tmp/session.log",
                "runtime_state_root": "H:/.mochi/exec-runtime",
                "detached_layout": {
                    "root_dir": "H:/tmp",
                    "log_path": "H:/tmp/session.log",
                    "session_log_path": "H:/tmp/session.log",
                    "checkpoint_dir": "H:/tmp/checkpoints",
                    "manifest_path": "H:/tmp/manifest.json",
                    "stdout_log_path": "H:/tmp/stdout.log",
                    "stderr_log_path": "H:/tmp/stderr.log",
                    "runtime_state_root": "H:/.mochi/exec-runtime",
                },
            },
        }
    ]

    payload = _collect_detached_exec_jobs(execution_results)

    assert payload["count"] == 1
    assert payload["reattachable_count"] == 1
    assert payload["recoverable_count"] == 1
    item = payload["items"][0]
    assert item["session_id"] == "exec-1"
    assert item["reattach_supported"] is True
    assert item["recoverable"] is True
    assert item["detached"] is True
    assert item["supports_stdin"] is False
    assert item["manifest_path"].endswith("manifest.json")
    assert item["stdout_log_path"].endswith("stdout.log")
    assert item["stderr_log_path"].endswith("stderr.log")
    assert item["runtime_state_root"].endswith("exec-runtime")


@pytest.mark.asyncio
async def test_exec_command_background_metadata_includes_detached_layout(tmp_path: Path) -> None:
    detached_layout = _prepare_detached_exec_layout(str(tmp_path), "req-bg")
    runtime = exec_command_module.ExecRuntime(
        providers={"test": _PythonDirectProvider()},
        default_shell="test",
        state_root=Path(detached_layout["runtime_state_root"]),
    )
    tool = ExecCommandTool(runtime=runtime, require_approval=False, workspace_dir=str(tmp_path))

    result = await tool.execute(
        command="bg",
        shell="test",
        background=True,
        detached_layout=detached_layout,
    )

    assert result.error is None
    assert result.metadata["status"] == "running"
    assert result.metadata["detached"] is True
    assert result.metadata["reattach_supported"] is True
    assert result.metadata["recovery_supported"] is True
    assert result.metadata["supports_stdin"] is False
    assert result.metadata["manifest_path"].endswith("manifest.json")
    assert result.metadata["stdout_log_path"].endswith("stdout.log")
    assert result.metadata["stderr_log_path"].endswith("stderr.log")
    assert isinstance(result.output, dict)
    assert result.output["detached_layout"]["root_dir"] == detached_layout["root_dir"]
    await runtime.kill_session(result.metadata["session_id"])


def test_get_shared_exec_runtime_uses_fixed_state_root_when_supported(monkeypatch: Any) -> None:
    original_runtime = exec_command_module._SHARED_RUNTIME
    original_exec_runtime = exec_command_module.ExecRuntime

    captured: dict[str, Any] = {}

    class _FakeRuntime:
        def __init__(self, *, state_root: Path | None = None) -> None:
            captured["state_root"] = state_root

    monkeypatch.setattr(exec_command_module, "ExecRuntime", _FakeRuntime)
    exec_command_module._SHARED_RUNTIME = None
    try:
        runtime = get_shared_exec_runtime()
    finally:
        exec_command_module._SHARED_RUNTIME = original_runtime
        monkeypatch.setattr(exec_command_module, "ExecRuntime", original_exec_runtime)

    assert isinstance(runtime, _FakeRuntime)
    assert captured["state_root"] == exec_command_module._shared_exec_runtime_state_root()
