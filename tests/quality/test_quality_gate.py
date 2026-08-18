from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath

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


def test_component_diagnostics_stream_to_console_and_durable_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    command = (
        run.sys.executable,
        "-c",
        "import sys; print('standard output'); print('standard error', file=sys.stderr); raise SystemExit(3)",
    )
    log_path = tmp_path / "artifacts" / "quality" / "logs" / "sample.log"

    completed, summary = run._run_with_diagnostics(command, cwd=tmp_path, log_path=log_path)

    captured = capsys.readouterr()
    assert completed.returncode == 3
    assert "standard output" in captured.out
    assert "standard error" in captured.err
    assert "[stdout] standard output" in log_path.read_text(encoding="utf-8")
    assert "[stderr] standard error" in log_path.read_text(encoding="utf-8")
    assert "standard output" in summary
    assert "standard error" in summary


def test_unicode_console_failure_does_not_stop_diagnostic_pipe_drain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Cp950Console:
        def write(self, _text: str) -> int:
            raise UnicodeEncodeError("cp950", "🦊", 0, 1, "cannot encode")

        def flush(self) -> None:
            return None

    log_path = tmp_path / "diagnostics.log"
    monkeypatch.setattr(run.sys, "stdout", Cp950Console())

    completed, summary = run._run_with_diagnostics(
        (run.sys.executable, "-c", "import sys; sys.stdout.buffer.write('🦊 unicode output\\n'.encode('utf-8'))"),
        cwd=tmp_path,
        log_path=log_path,
    )

    assert completed.returncode == 0
    assert summary == ""
    assert "🦊 unicode output" in log_path.read_text(encoding="utf-8")


def test_failed_component_keeps_diagnostics_and_runs_remaining_components(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    failing = run.Task("first", ("first",))
    passing = run.Task("second", ("second",))
    monkeypatch.setattr(run, "_tasks", lambda: {"first": failing, "second": passing})
    monkeypatch.setattr(run, "_task_available", lambda _task: (True, None))
    called: list[str] = []

    def fake_run(command: list[str], *, cwd: Path, log_path: Path) -> tuple[subprocess.CompletedProcess[str], str]:
        called.append(command[0])
        if command[0] == "first":
            print("FAILED tests/test_first.py")
            print("collection warning", file=run.sys.stderr)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text("[stdout] FAILED tests/test_first.py\n[stderr] collection warning\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 1, "", ""), "[stdout] FAILED tests/test_first.py\n[stderr] collection warning"
        return subprocess.CompletedProcess(command, 0, "", ""), ""

    monkeypatch.setattr(run, "_run_with_diagnostics", fake_run)

    result, artifacts = run._execute_components(tmp_path, ("first", "second"))

    diagnostics = capsys.readouterr().err
    assert result == "regression"
    assert called == ["first", "second"]
    assert artifacts[0]["exit_code"] == 1
    assert artifacts[0]["log_path"] == "artifacts/quality/local/logs/first.log"
    assert artifacts[0]["duration_seconds"] >= 0
    assert artifacts[0]["failure_summary"] == "[stdout] FAILED tests/test_first.py\n[stderr] collection warning"
    assert "quality component 'first' failed with exit code 1" in diagnostics
    assert "FAILED tests/test_first.py" in diagnostics
    assert "collection warning" in diagnostics


def test_generated_artifact_exclusions_are_limited_to_selected_components(tmp_path: Path) -> None:
    report_target = tmp_path / "artifacts" / "quality" / "pr.json"
    invocation_root = tmp_path / "artifacts" / "quality" / "runs" / "pr-unique"
    user_artifact = tmp_path / "artifacts" / "quality" / "logs" / "user-note.log"
    user_artifact.parent.mkdir(parents=True)
    user_artifact.write_text("preserve me", encoding="utf-8")

    paths = run._generated_artifact_paths(("diff", "python"), report_target, invocation_root)

    assert paths == (
        report_target,
        invocation_root / "logs" / "diff.log",
        invocation_root / "logs" / "python.log",
        invocation_root / "junit" / "python.xml",
    )
    assert user_artifact not in paths
    assert user_artifact.read_text(encoding="utf-8") == "preserve me"
    assert "artifacts/quality/logs/user-note.log" not in run._excluded_relative_paths(tmp_path, paths)


def test_pytest_component_records_a_junit_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task = run.Task("python", (run.sys.executable, "-m", "pytest", "-q"), "pytest")
    monkeypatch.setattr(run, "_tasks", lambda: {"python": task})
    monkeypatch.setattr(run, "_task_available", lambda _task: (True, None))
    seen_commands: list[list[str]] = []

    def fake_run(command: list[str], *, cwd: Path, log_path: Path) -> tuple[subprocess.CompletedProcess[str], str]:
        seen_commands.append(command)
        return subprocess.CompletedProcess(command, 0, "", ""), ""

    monkeypatch.setattr(run, "_run_with_diagnostics", fake_run)

    result, artifacts = run._execute_components(tmp_path, ("python",))

    assert result == "passed"
    assert seen_commands[0][-2:] == ["--junitxml", "artifacts/quality/local/junit/python.xml"]
    assert artifacts[0]["junit_path"] == "artifacts/quality/local/junit/python.xml"


def test_component_artifact_root_must_stay_inside_repository(tmp_path: Path) -> None:
    outside_root = tmp_path.parent / "outside-artifacts"

    with pytest.raises(run.QualityGateError, match="inside the repository"):
        run._execute_components(tmp_path, (), artifact_root=outside_root)


def test_windows_cmd_shim_is_run_through_cmd_with_tokenized_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shim = r"C:\\Program Files\\pnpm\\pnpm.cmd"
    command = ("pnpm", "--dir", "web folder", "run", "typecheck", "--label=a&b")

    monkeypatch.setattr(run.shutil, "which", lambda executable: shim if executable == "pnpm" else None)

    converted = run._command_for_subprocess(command, host_os_name="nt")

    assert PureWindowsPath(converted[0]).name.lower() == "cmd.exe"
    assert converted[1:4] == ["/d", "/s", "/c"]
    assert converted[4] == subprocess.list2cmdline([shim, *command[1:]])


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
