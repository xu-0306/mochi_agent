"""Background process tools tests."""

from __future__ import annotations

import asyncio

import pytest

from mochi.tools.process_control import ProcessPollTool, ProcessStopTool
from mochi.tools.process_service import ProcessService
from mochi.tools.shell import ShellTool


@pytest.mark.asyncio
async def test_shell_background_launch_poll_stop(tmp_path: Path) -> None:
    service = ProcessService()
    shell = ShellTool(
        allowlist=["ping"],
        workspace_dir=tmp_path,
        require_approval=False,
        process_service=service,
    )
    poll_tool = ProcessPollTool(process_service=service)
    stop_tool = ProcessStopTool(process_service=service)

    start = await shell.execute(
        command="ping 127.0.0.1 -n 6",
        background=True,
        process_label="bg-shell",
    )
    assert start.error is None
    assert start.metadata["background"] is True
    assert start.metadata["status"] == "running"
    process_id = start.metadata["process_id"]

    poll = await poll_tool.execute(process_id=process_id)
    assert poll.error is None
    assert poll.metadata["process_id"] == process_id

    stop = await stop_tool.execute(process_id=process_id)
    assert stop.error is None
    assert stop.metadata["status"] == "exited"
    assert stop.metadata["stopped"] is True


@pytest.mark.asyncio
async def test_process_poll_and_stop_not_found() -> None:
    service = ProcessService()
    poll_tool = ProcessPollTool(process_service=service)
    stop_tool = ProcessStopTool(process_service=service)

    poll = await poll_tool.execute(process_id="proc-404")
    assert poll.error is not None

    stop = await stop_tool.execute(process_id="proc-404")
    assert stop.error is not None


def test_process_service_records_exited_process(tmp_path: Path) -> None:
    async def _run() -> None:
        service = ProcessService()
        payload = await service.start_shell(command='python -c "print(1)"', cwd=tmp_path)
        process_id = payload["process_id"]
        await asyncio.sleep(0.2)
        polled = await service.poll(process_id)
        assert polled is not None
        assert polled["status"] in {"running", "exited"}

    asyncio.run(_run())


def test_process_service_exposes_output_tails(tmp_path: Path) -> None:
    async def _run() -> None:
        service = ProcessService()
        payload = await service.start_shell(
            command='python -c "import sys; print(\'hello\'); sys.stderr.write(\'oops\\n\')"',
            cwd=tmp_path,
        )
        process_id = payload["process_id"]
        await asyncio.sleep(0.2)
        polled = await service.poll(process_id)
        assert polled is not None
        assert polled["status"] == "exited"
        assert "hello" in polled["stdout_tail"]
        assert "oops" in polled["stderr_tail"]

    asyncio.run(_run())
