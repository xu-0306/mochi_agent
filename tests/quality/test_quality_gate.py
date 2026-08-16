from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.quality_gate import run


def _dirty_state(entries: list[str]) -> dict[str, object]:
    tracked_diff_sha256 = "a" * 64
    untracked_files: dict[str, dict[str, object]] = {}
    return {
        "entries": entries,
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked_files": untracked_files,
        "sha256": run.dirty_state_digest(
            entries=entries,
            tracked_diff_sha256=tracked_diff_sha256,
            untracked_files=untracked_files,
        ),
    }


def test_impact_selection_is_conservative() -> None:
    impacts = run.select_impacts(
        [
            "mochi/agents/engine.py",
            "web/src/app/page.tsx",
            "docs/architecture/contracts/quality-report-v1.json",
        ]
    )

    assert impacts == ("contracts", "diff", "python", "ruff", "web")


def test_web_task_uses_the_local_typescript_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MOCHI_NODE_EXECUTABLE", raising=False)
    monkeypatch.setattr(run.shutil, "which", lambda _name: None)
    executable = "web/node_modules/.bin/tsc.cmd" if run.os.name == "nt" else "web/node_modules/.bin/tsc"

    assert run._tasks()["web"].command == (
        executable,
        "-p",
        "web/tsconfig.json",
        "--noEmit",
        "--incremental",
        "false",
    )


def test_web_task_can_use_an_explicit_existing_node_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    node_runtime = tmp_path / ("node.exe" if run.os.name == "nt" else "node")
    node_runtime.write_bytes(b"")
    monkeypatch.setenv("MOCHI_NODE_EXECUTABLE", str(node_runtime.resolve()))

    assert run._tasks()["web"].command == (
        str(node_runtime.resolve()),
        "web/node_modules/typescript/bin/tsc",
        "-p",
        "web/tsconfig.json",
        "--noEmit",
        "--incremental",
        "false",
    )


def test_ruff_task_uses_declared_executable_product_roots() -> None:
    assert run._tasks()["ruff"].command == (
        run.sys.executable,
        "-m",
        "ruff",
        "check",
        "--no-cache",
        "--force-exclude",
        *run.RUFF_TARGETS,
    )


def test_expired_quarantine_is_rejected() -> None:
    with pytest.raises(run.QualityGateError, match="expired"):
        run.validate_quarantines(
            [
                {
                    "id": "expired-provider-smoke",
                    "owner": "release",
                    "reason": "fixture unavailable",
                    "expires_at": "2020-01-01T00:00:00Z",
                    "lanes": ["nightly"],
                }
            ],
            now=datetime(2026, 8, 5, tzinfo=UTC),
        )


def test_overlapping_quarantines_are_rejected() -> None:
    record = {
        "id": "provider-smoke-2026-08",
        "owner": "release",
        "reason": "fixture unavailable",
        "expires_at": "2026-09-01T00:00:00Z",
        "lanes": ["nightly"],
    }

    with pytest.raises(run.QualityGateError, match="multiple quarantines"):
        run.validate_quarantines([record, {**record, "id": "provider-smoke-duplicate"}])


def test_baseline_report_records_every_lane_without_claiming_release_pass() -> None:
    report = run.build_report(
        repo_root=Path.cwd(),
        mode="baseline",
        execute=False,
        git_sha="f45a0c631ec9c18b13849ddf18a93a5dbb0f63bc",
        dirty_state=_dirty_state([" M docs/testing/README.md"]),
    )

    assert [lane["lane"] for lane in report["lanes"]] == list(run.LANES)
    assert {lane["result"] for lane in report["lanes"]} == {"conditional"}
    assert report["dirty_state"]["entries"] == [" M docs/testing/README.md"]


def test_clean_checkout_uses_all_components_when_no_impact_source_is_available() -> None:
    report = run.build_report(
        repo_root=Path.cwd(),
        mode="baseline",
        execute=False,
        git_sha="f45a0c631ec9c18b13849ddf18a93a5dbb0f63bc",
        dirty_state=_dirty_state([]),
    )

    pr_lane = next(lane for lane in report["lanes"] if lane["lane"] == "pr")
    assert pr_lane["capability"]["components"] == sorted(run.COMPONENTS)


def test_changed_paths_between_uses_the_explicit_ci_revision_range(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, ...]] = []

    def fake_git_output(_repo_root: Path, *arguments: str) -> str:
        observed.append(arguments)
        return "mochi/agents/engine.py\0tests/test_engine.py\0"

    monkeypatch.setattr(run, "_git_output", fake_git_output)

    assert run.changed_paths_between(Path.cwd(), "base-sha", "head-sha") == (
        "mochi/agents/engine.py",
        "tests/test_engine.py",
    )
    assert observed == [("diff", "--name-only", "-z", "base-sha", "head-sha")]


def test_main_lane_always_selects_full_required_components() -> None:
    report = run.build_report(
        repo_root=Path.cwd(),
        mode="baseline",
        execute=False,
        git_sha="f45a0c631ec9c18b13849ddf18a93a5dbb0f63bc",
        dirty_state=_dirty_state([" M docs/testing/README.md"]),
    )

    main_lane = next(lane for lane in report["lanes"] if lane["lane"] == "main")
    assert main_lane["capability"]["components"] == sorted(run.COMPONENTS)


def test_quarantined_lane_carries_an_expiring_waiver() -> None:
    quarantine = {
        "id": "provider-smoke-2026-08",
        "owner": "release",
        "reason": "provider credentials unavailable in CI",
        "expires_at": "2026-09-01T00:00:00Z",
        "lanes": ["nightly"],
    }
    report = run.build_report(
        repo_root=Path.cwd(),
        mode="baseline",
        execute=False,
        quarantines=[quarantine],
        git_sha="f45a0c631ec9c18b13849ddf18a93a5dbb0f63bc",
        dirty_state=_dirty_state([]),
    )

    nightly = next(lane for lane in report["lanes"] if lane["lane"] == "nightly")
    assert nightly["result"] == "quarantined"
    assert nightly["waiver"] == quarantine


def test_report_validator_rejects_missing_lane() -> None:
    report = run.build_report(
        repo_root=Path.cwd(),
        mode="baseline",
        execute=False,
        git_sha="f45a0c631ec9c18b13849ddf18a93a5dbb0f63bc",
        dirty_state=_dirty_state([]),
    )
    report["lanes"].pop()

    with pytest.raises(run.QualityGateError, match="every qualification lane"):
        run.validate_report(report)


def test_dirty_state_digest_binds_untracked_file_contents() -> None:
    first = run.dirty_state_digest(
        entries=["?? artifacts/quality/baseline.json"],
        tracked_diff_sha256="a" * 64,
        untracked_files={"artifacts/quality/baseline.json": {"sha256": "b" * 64, "size": 1}},
    )
    second = run.dirty_state_digest(
        entries=["?? artifacts/quality/baseline.json"],
        tracked_diff_sha256="a" * 64,
        untracked_files={"artifacts/quality/baseline.json": {"sha256": "c" * 64, "size": 2}},
    )

    assert first != second


def test_report_validator_rejects_dirty_state_content_mismatch() -> None:
    report = run.build_report(
        repo_root=Path.cwd(),
        mode="baseline",
        execute=False,
        git_sha="f45a0c631ec9c18b13849ddf18a93a5dbb0f63bc",
        dirty_state=_dirty_state([]),
    )
    report["dirty_state"]["tracked_diff_sha256"] = "b" * 64

    with pytest.raises(run.QualityGateError, match="does not match"):
        run.validate_report(report)


def test_missing_optional_dependency_is_environment_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = run.Task("optional-live-provider", ("definitely-not-an-installed-executable",))
    monkeypatch.setattr(run, "_tasks", lambda: {"optional-live-provider": missing})

    result, artifacts = run._execute_components(Path.cwd(), ("optional-live-provider",))

    assert result == "environment-blocked"
    assert artifacts[0]["result"] == "environment-blocked"


def test_windows_cmd_shim_is_run_through_cmd_with_tokenized_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []
    shim = r"C:\\Program Files\\pnpm\\pnpm.cmd"
    command = ("pnpm", "--dir", "web folder", "run", "typecheck", "--label=a&b")

    def fake_run(arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((arguments, kwargs))
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(run.shutil, "which", lambda executable: shim if executable == "pnpm" else None)
    monkeypatch.setattr(run.subprocess, "run", fake_run)

    completed = run._run(command, cwd=Path("D:/workspace"))

    assert completed.returncode == 0
    assert Path(calls[0][0][0]).name.lower() == "cmd.exe"
    assert calls[0][0][1:4] == ["/d", "/s", "/c"]
    assert calls[0][0][4] == subprocess.list2cmdline([shim, *command[1:]])


def test_run_decodes_console_output_as_utf8_with_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(_arguments: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(kwargs)
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr(run.subprocess, "run", fake_run)

    run._run(("git", "status"), cwd=Path("D:/workspace"))

    assert calls == [
        {
            "cwd": Path("D:/workspace"),
            "check": False,
            "capture_output": True,
            "text": True,
            "encoding": "utf-8",
            "errors": "replace",
        }
    ]


def test_live_lane_requires_execute_flag(capsys: pytest.CaptureFixture[str]) -> None:
    result = run.main(["--mode", "release", "--report", "artifacts/quality/release.json"])

    assert result == 2
    assert "require --execute" in capsys.readouterr().err


def test_report_validator_rejects_passed_lane_with_a_blocked_component() -> None:
    report = run.build_report(
        repo_root=Path.cwd(),
        mode="baseline",
        execute=False,
        git_sha="f45a0c631ec9c18b13849ddf18a93a5dbb0f63bc",
        dirty_state=_dirty_state([]),
    )
    pr_lane = next(lane for lane in report["lanes"] if lane["lane"] == "pr")
    pr_lane["result"] = "passed"
    pr_lane["capability"]["execution_requested"] = True
    pr_lane["artifacts"] = [
        {"component": component, "result": "environment-blocked", "detail": "fixture unavailable"}
        for component in pr_lane["capability"]["components"]
    ]

    with pytest.raises(run.QualityGateError, match="non-passing component"):
        run.validate_report(report)
