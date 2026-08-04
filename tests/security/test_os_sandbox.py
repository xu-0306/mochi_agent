"""Security invariants for digest-bound operating-system sandbox plans."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from mochi.runtime.sandbox.base import (
    HostSandboxBackend,
    SandboxCapabilities,
    SandboxPlan,
    SandboxPlanMismatch,
    SandboxResourceLimits,
    SandboxUnavailableError,
    canonical_path,
    create_sandbox_plan,
    env_hashes,
)
from mochi.runtime.sandbox.broker_protocol import (
    BROKER_PROTOCOL_VERSION,
    BrokerProtocolError,
    decode_frame,
    run_frame,
)
from mochi.runtime.sandbox.linux import BubblewrapSandboxBackend
from mochi.runtime.sandbox.windows import WindowsSandboxBackend

NONCE = "0123456789abcdef0123456789abcdef"


def _host_plan(
    tmp_path: Path,
    *,
    mode: str = "preferred",
    argv: tuple[str, ...] = ("-c", "print('ok')"),
    capabilities: SandboxCapabilities | None = None,
) -> SandboxPlan:
    backend = HostSandboxBackend(degraded_reason="OS sandbox backend unavailable")
    observed = capabilities or backend.probe()
    return SandboxPlan(
        mode=mode,  # type: ignore[arg-type]
        executable=sys.executable,
        argv=argv,
        resolved_cwd=str(tmp_path.resolve()),
        read_roots=(str(tmp_path.resolve()),),
        write_roots=(str(tmp_path.resolve()),),
        network_policy="deny",
        env=env_hashes({"TOKEN": "top-secret"}),
        resource_limits=SandboxResourceLimits(
            timeout_milliseconds=30_000,
            memory_limit_mb=512,
            max_processes=8,
            output_limit_bytes=1_000_000,
        ),
        requested_escalation="none",
        backend=observed.backend,
        backend_version=observed.version,
        capabilities=observed,
        request_nonce=NONCE,
    )


@pytest.mark.security
def test_plan_digest_is_stable_and_secret_free(tmp_path: Path) -> None:
    first = _host_plan(tmp_path)
    second = _host_plan(tmp_path)

    assert first.digest == second.digest
    encoded = json.dumps(first.to_dict())
    assert "top-secret" not in encoded
    assert first.env[0].name == "TOKEN"
    assert len(first.env[0].value_sha256) == 64


@pytest.mark.security
def test_plan_digest_changes_when_launch_facts_change(tmp_path: Path) -> None:
    original = _host_plan(tmp_path)
    changed_argv = _host_plan(tmp_path, argv=("-c", "print('changed')"))
    other_cwd = tmp_path / "other"
    other_cwd.mkdir()
    changed_cwd = _host_plan(other_cwd)

    assert original.digest != changed_argv.digest
    assert original.digest != changed_cwd.digest


@pytest.mark.security
def test_plan_rejects_tampering_and_unknown_fields(tmp_path: Path) -> None:
    serialized = _host_plan(tmp_path).to_dict()
    serialized["argv"] = ["-c", "print('tampered')"]
    with pytest.raises(SandboxPlanMismatch):
        SandboxPlan.from_dict(serialized)

    serialized = _host_plan(tmp_path).to_dict()
    serialized["unexpected"] = True
    with pytest.raises(ValueError, match="fields"):
        SandboxPlan.from_dict(serialized)


@pytest.mark.security
def test_broker_protocol_binds_version_nonce_and_exact_fields(tmp_path: Path) -> None:
    frame = run_frame(_host_plan(tmp_path))
    decoded = decode_frame(frame.encode(), expected_nonce=NONCE)
    assert decoded.protocol_version == BROKER_PROTOCOL_VERSION

    value = json.loads(frame.encode())
    value["protocol_version"] = 99
    with pytest.raises(BrokerProtocolError, match="version"):
        decode_frame(json.dumps(value))

    with pytest.raises(BrokerProtocolError, match="nonce mismatch"):
        decode_frame(frame.encode(), expected_nonce="fedcba9876543210fedcba9876543210")

    value = json.loads(frame.encode())
    value["extra"] = "rejected"
    with pytest.raises(BrokerProtocolError, match="fields"):
        decode_frame(json.dumps(value))


@pytest.mark.security
def test_required_mode_rejects_host_or_incomplete_capabilities(tmp_path: Path) -> None:
    backend = HostSandboxBackend(degraded_reason="backend unavailable")
    plan = _host_plan(tmp_path, mode="required", capabilities=backend.probe())

    with pytest.raises(SandboxUnavailableError, match="filesystem, process, and network"):
        backend.prepare_launch(plan, env=None)


@pytest.mark.security
def test_preferred_mode_degrades_explicitly_to_host(tmp_path: Path) -> None:
    backend = HostSandboxBackend(degraded_reason="OS backend unavailable")
    plan = _host_plan(tmp_path, capabilities=backend.probe())

    launch = backend.prepare_launch(plan, env={"SAFE": "1"})

    assert launch.backend == "host"
    assert launch.degraded_reason == "OS backend unavailable"
    assert launch.plan_digest == plan.digest


@pytest.mark.security
def test_capability_change_after_approval_fails_closed(tmp_path: Path) -> None:
    approved_backend = HostSandboxBackend(degraded_reason="first probe")
    launch_backend = HostSandboxBackend(degraded_reason="changed probe")
    plan = _host_plan(tmp_path, capabilities=approved_backend.probe())

    with pytest.raises(SandboxPlanMismatch, match="changed"):
        launch_backend.prepare_launch(plan, env=None)


@pytest.mark.security
def test_bubblewrap_launch_is_argument_only_and_denies_network(tmp_path: Path) -> None:
    capabilities = SandboxCapabilities(
        backend="bubblewrap",
        version="bubblewrap 1.2.3",
        available=True,
        filesystem=True,
        process=True,
        network=True,
    )
    backend = BubblewrapSandboxBackend(binary="bwrap")
    backend._cached_probe = capabilities
    plan = replace(
        _host_plan(tmp_path),
        backend=capabilities.backend,
        backend_version=capabilities.version,
        capabilities=capabilities,
    )

    launch = backend.prepare_launch(plan, env={"TOKEN": "top-secret"})

    assert launch.executable == "bwrap"
    assert "--unshare-net" in launch.args
    assert "--bind" in launch.args
    assert (
        "--chmod",
        "0555",
        canonical_path(tmp_path.parent),
    ) in tuple(zip(launch.args, launch.args[1:], launch.args[2:]))
    assert launch.args[-4:] == ("--", canonical_path(sys.executable), "-c", "print('ok')")
    assert launch.env is None


@pytest.mark.security
def test_windows_handshake_never_promotes_incomplete_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    helper = tmp_path / "mochi-sandbox-windows.exe"
    helper.touch()
    capabilities = SandboxCapabilities(
        backend="windows-appcontainer",
        version="scaffold-1",
        available=False,
        filesystem=False,
        process=False,
        network=False,
        degraded_reason="not_implemented",
    )

    def _probe(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        del kwargs
        nonce = args[-1]
        from mochi.runtime.sandbox.broker_protocol import hello_frame

        frame = hello_frame(
            nonce=nonce,
            backend=capabilities.backend,
            version=capabilities.version,
            capabilities=capabilities.to_dict(),
        )
        return subprocess.CompletedProcess(args, 0, stdout=frame.encode(), stderr=b"")

    monkeypatch.setattr("mochi.runtime.sandbox.windows.sys.platform", "win32")
    monkeypatch.setattr("mochi.runtime.sandbox.windows.subprocess.run", _probe)
    observed = WindowsSandboxBackend(helper_path=helper).probe()

    assert observed.available is False
    assert observed.complete is False
    assert observed.degraded_reason == "not_implemented"


@pytest.mark.real_linux_sandbox
@pytest.mark.skipif(
    not sys.platform.startswith("linux") or os.environ.get("MOCHI_RUN_REAL_SANDBOX_TESTS") != "1",
    reason="real bubblewrap release gate is opt-in",
)
def test_real_bubblewrap_blocks_write_outside_workspace(tmp_path: Path) -> None:
    backend = BubblewrapSandboxBackend()
    capabilities = backend.probe()
    assert capabilities.complete, capabilities.degraded_reason
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    plan = create_sandbox_plan(
        mode="required",
        executable="/bin/sh",
        argv=("-c", f"printf escaped > '{outside}'"),
        cwd=workspace,
        read_roots=(workspace,),
        write_roots=(workspace,),
        network_policy="deny",
        env=None,
        resource_limits=SandboxResourceLimits(
            timeout_milliseconds=5_000,
            memory_limit_mb=128,
            max_processes=4,
            output_limit_bytes=100_000,
        ),
        requested_escalation="none",
        backend=backend,
    )
    launch = backend.prepare_launch(plan, env=None)

    completed = subprocess.run(
        [launch.executable, *launch.args],
        cwd=launch.cwd,
        env=launch.env,
        check=False,
        capture_output=True,
        timeout=10,
    )

    assert completed.returncode != 0
    assert not outside.exists()
