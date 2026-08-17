"""Produce a revision-bound, machine-readable qualification report.

The runner deliberately accepts no arbitrary command input.  Qualification
commands are defined in this module so a report cannot represent an unrelated
or silently skipped command as a passing release gate.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPORT_SCHEMA_VERSION = "1.0"
LANES = ("pr", "main", "nightly", "release")
RESULT_VALUES = ("passed", "regression", "conditional", "environment-blocked", "quarantined")
RESULTS = frozenset(RESULT_VALUES)
COMPONENTS = frozenset({"contracts", "diff", "python", "ruff", "web"})
RUFF_TARGETS = ("mochi", "tests", "scripts/quality_gate")
CONTRACT_SCHEMA_NAME = "mochi.contract"
CONTRACT_SCHEMA_VERSION = "1.0"
CONTRACT_REQUIRED_FIELDS = frozenset(
    {
        "schema_name",
        "schema_version",
        "contract_id",
        "producer_package",
        "git_revision",
        "status",
        "symbols",
        "inputs",
        "outputs",
        "errors",
        "compatibility",
        "consumers",
        "gate_ids",
    }
)


class QualityGateError(ValueError):
    """Raised when report input would make a qualification result ambiguous."""


@dataclass(frozen=True)
class Task:
    """A fixed qualification command and its deterministic capability check."""

    name: str
    command: tuple[str, ...]
    python_module: str | None = None


def _command_for_subprocess(
    command: Sequence[str],
    *,
    host_os_name: str | None = None,
) -> list[str]:
    """Return a Windows-runnable command without broadening task inputs.

    The quality task catalog is fixed, but its `pnpm` executable can resolve to
    a `.cmd` shim.  `CreateProcess` cannot execute that shim directly when
    `shell=False`, so invoke only that resolved shim through the system command
    interpreter with token-preserving quoting.
    """

    normalized = list(command)
    if (host_os_name or os.name) != "nt":
        return normalized

    configured = Path(normalized[0])
    resolved = str(configured) if configured.is_file() else shutil.which(normalized[0])
    if not resolved or Path(resolved).suffix.lower() not in {".bat", ".cmd"}:
        return normalized

    return [
        os.environ.get("COMSPEC", "cmd.exe"),
        "/d",
        "/s",
        "/c",
        subprocess.list2cmdline([resolved, *normalized[1:]]),
    ]


def _run(command: Sequence[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _command_for_subprocess(command),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _git_output(repo_root: Path, *arguments: str) -> str:
    completed = _run(("git", "-C", str(repo_root), *arguments), cwd=repo_root)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise QualityGateError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def _git_bytes(repo_root: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).decode("utf-8", errors="replace").strip()
        raise QualityGateError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout


def current_revision(repo_root: Path) -> str:
    """Return the exact checked-out Git revision without changing the worktree."""

    return _git_output(repo_root, "rev-parse", "HEAD").strip()


def _fingerprint_untracked_file(repo_root: Path, relative_path: str) -> dict[str, Any]:
    root = repo_root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise QualityGateError(f"untracked path escapes repository: {relative_path}") from error
    if not candidate.is_file():
        raise QualityGateError(f"untracked path is not a regular file: {relative_path}")
    payload = candidate.read_bytes()
    return {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}


def dirty_state_digest(
    *,
    entries: Sequence[str],
    tracked_diff_sha256: str,
    untracked_files: dict[str, dict[str, Any]],
) -> str:
    """Return the canonical digest for a content-bound dirty-state snapshot."""

    canonical = json.dumps(
        {
            "entries": list(entries),
            "tracked_diff_sha256": tracked_diff_sha256,
            "untracked_files": untracked_files,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _excluded_relative_paths(repo_root: Path, paths: Iterable[Path]) -> frozenset[str]:
    root = repo_root.resolve()
    excluded: set[str] = set()
    for path in paths:
        candidate = path if path.is_absolute() else root / path
        try:
            excluded.add(candidate.resolve().relative_to(root).as_posix())
        except ValueError as error:
            raise QualityGateError("excluded dirty-state path must be inside the repository") from error
    return frozenset(excluded)


def current_dirty_state(
    repo_root: Path, *, excluded_paths: Iterable[Path] = ()
) -> dict[str, Any]:
    """Return an exact content-bound snapshot without changing the worktree."""

    excluded = _excluded_relative_paths(repo_root, excluded_paths)
    raw_status = _git_output(repo_root, "status", "--porcelain=v1", "--untracked-files=all")
    entries = [
        line.rstrip("\n")
        for line in raw_status.splitlines()
        if line and line[3:] not in excluded
    ]
    tracked_diff_sha256 = hashlib.sha256(_git_bytes(repo_root, "diff", "--binary", "HEAD")).hexdigest()
    untracked_names = _git_output(repo_root, "ls-files", "--others", "--exclude-standard", "-z").split("\0")
    untracked_files = {
        path: _fingerprint_untracked_file(repo_root, path)
        for path in sorted(name for name in untracked_names if name and name not in excluded)
    }
    return {
        "entries": entries,
        "tracked_diff_sha256": tracked_diff_sha256,
        "untracked_files": untracked_files,
        "sha256": dirty_state_digest(
            entries=entries,
            tracked_diff_sha256=tracked_diff_sha256,
            untracked_files=untracked_files,
        ),
    }


def select_impacts(paths: Iterable[str]) -> tuple[str, ...]:
    """Map changed paths to conservative qualification components.

    A component is selected whenever a path could affect it.  Unknown paths
    retain the always-on diff check rather than being interpreted as a pass.
    """

    impacts = {"diff"}
    normalized = tuple(path.replace("\\", "/") for path in paths)
    if any(path.startswith(("mochi/", "tests/")) or path in {"pyproject.toml", "uv.lock"} for path in normalized):
        impacts.update({"python", "ruff"})
    if any(path.startswith("web/") for path in normalized):
        impacts.add("web")
    if any(
        path.startswith(("docs/architecture/contracts/", "scripts/quality_gate/", ".github/workflows/"))
        for path in normalized
    ):
        impacts.add("contracts")
    return tuple(sorted(impacts))


def _changed_paths(dirty_state: dict[str, Any]) -> tuple[str, ...]:
    status_paths = [entry[3:] for entry in dirty_state["entries"] if len(entry) > 3]
    return tuple(sorted({*status_paths, *dirty_state["untracked_files"]}))


def changed_paths_between(repo_root: Path, start_revision: str, end_revision: str = "HEAD") -> tuple[str, ...]:
    """Return the exact Git paths changed between two revisions for CI impact selection."""

    raw_paths = _git_output(repo_root, "diff", "--name-only", "-z", start_revision, end_revision)
    return tuple(sorted({path for path in raw_paths.split("\0") if path}))


def _parse_timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise QualityGateError(f"invalid quarantine expiry {value!r}") from error
    if parsed.tzinfo is None:
        raise QualityGateError("quarantine expiry must include a timezone")
    return parsed.astimezone(UTC)


def validate_quarantines(
    quarantines: Iterable[dict[str, Any]], *, now: datetime | None = None
) -> tuple[dict[str, Any], ...]:
    """Validate expiring, owned quarantine records before they affect a lane."""

    observed_now = now or datetime.now(UTC)
    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    covered_lanes: set[str] = set()
    for entry in quarantines:
        required = {"id", "owner", "reason", "expires_at", "lanes"}
        missing = sorted(required.difference(entry))
        if missing:
            raise QualityGateError(f"quarantine missing required fields: {', '.join(missing)}")
        if not isinstance(entry["lanes"], list) or not entry["lanes"]:
            raise QualityGateError("quarantine lanes must be a non-empty list")
        unknown_lanes = sorted(set(entry["lanes"]).difference(LANES))
        if unknown_lanes:
            raise QualityGateError(f"quarantine references unknown lanes: {', '.join(unknown_lanes)}")
        if not all(isinstance(entry[field], str) and entry[field].strip() for field in required - {"lanes"}):
            raise QualityGateError("quarantine id, owner, reason, and expires_at must be non-empty strings")
        if _parse_timestamp(entry["expires_at"]) <= observed_now:
            raise QualityGateError(f"quarantine {entry['id']} is expired")
        if entry["id"] in seen_ids:
            raise QualityGateError(f"duplicate quarantine id {entry['id']}")
        overlapping_lanes = covered_lanes.intersection(entry["lanes"])
        if overlapping_lanes:
            raise QualityGateError(
                f"multiple quarantines cover lanes: {', '.join(sorted(overlapping_lanes))}"
            )
        seen_ids.add(entry["id"])
        covered_lanes.update(entry["lanes"])
        validated.append(dict(entry))
    return tuple(validated)


def _load_quarantines(path: Path | None) -> tuple[dict[str, Any], ...]:
    if path is None:
        return ()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise QualityGateError(f"unable to read quarantine file {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise QualityGateError(f"invalid quarantine JSON in {path}: {error}") from error
    records = payload.get("quarantines", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise QualityGateError("quarantine file must contain a list of objects")
    return validate_quarantines(records)


def _web_typecheck_command() -> tuple[str, ...]:
    """Run the already-installed local compiler without package-manager mutation."""

    explicit_node = os.environ.get("MOCHI_NODE_EXECUTABLE")
    if explicit_node:
        node_path = Path(explicit_node)
        if not node_path.is_absolute() or not node_path.is_file():
            raise QualityGateError(
                "MOCHI_NODE_EXECUTABLE must be an absolute path to an existing Node executable"
            )
        return (
            str(node_path),
            "web/node_modules/typescript/bin/tsc",
            "-p",
            "web/tsconfig.json",
            "--noEmit",
            "--incremental",
            "false",
        )

    discovered_node = shutil.which("node")
    if discovered_node:
        return (
            discovered_node,
            "web/node_modules/typescript/bin/tsc",
            "-p",
            "web/tsconfig.json",
            "--noEmit",
            "--incremental",
            "false",
        )

    executable = "web/node_modules/.bin/tsc.cmd" if os.name == "nt" else "web/node_modules/.bin/tsc"
    return (executable, "-p", "web/tsconfig.json", "--noEmit", "--incremental", "false")


def _tasks() -> dict[str, Task]:
    return {
        "diff": Task("diff", ("git", "diff", "--check")),
        "contracts": Task(
            "contracts",
            (sys.executable, "-m", "pytest", "tests/contracts/test_frozen_contract_artifacts.py", "-q"),
            "pytest",
        ),
        "python": Task("python", (sys.executable, "-m", "pytest", "-q"), "pytest"),
        "ruff": Task(
            "ruff",
            (
                sys.executable,
                "-m",
                "ruff",
                "check",
                "--no-cache",
                "--force-exclude",
                *RUFF_TARGETS,
            ),
            "ruff",
        ),
        "web": Task("web", _web_typecheck_command()),
    }


def _components_for_lane(lane: str, impacts: Sequence[str]) -> tuple[str, ...]:
    selected = set(impacts)
    if lane in {"main", "nightly", "release"}:
        selected.update(COMPONENTS)
    return tuple(sorted(selected))


def _matching_quarantine(lane: str, quarantines: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    return next((entry for entry in quarantines if lane in entry["lanes"]), None)


def _task_available(task: Task) -> tuple[bool, str | None]:
    if task.python_module and importlib.util.find_spec(task.python_module) is None:
        return False, f"Python module {task.python_module!r} is unavailable"
    executable = task.command[0]
    if Path(executable).is_file() or shutil.which(executable):
        return True, None
    return False, f"Executable {executable!r} is unavailable"


def _execute_components(repo_root: Path, components: Sequence[str]) -> tuple[str, list[dict[str, Any]]]:
    task_results: list[dict[str, Any]] = []
    any_regression = False
    any_environment_block = False
    for component in components:
        task = _tasks()[component]
        available, reason = _task_available(task)
        if not available:
            any_environment_block = True
            task_results.append(
                {
                    "component": component,
                    "command": list(task.command),
                    "result": "environment-blocked",
                    "detail": reason,
                }
            )
            continue
        completed = _run(task.command, cwd=repo_root)
        result = "passed" if completed.returncode == 0 else "regression"
        any_regression = any_regression or result == "regression"
        if result == "regression":
            print(
                f"quality component {component!r} failed with exit code "
                f"{completed.returncode}",
                file=sys.stderr,
            )
            if completed.stdout:
                print(completed.stdout.rstrip(), file=sys.stderr)
            if completed.stderr:
                print(completed.stderr.rstrip(), file=sys.stderr)
        task_results.append(
            {
                "component": component,
                "command": list(task.command),
                "result": result,
                "exit_code": completed.returncode,
            }
        )
    if any_regression:
        return "regression", task_results
    if any_environment_block:
        return "environment-blocked", task_results
    return "passed", task_results


def build_report(
    *,
    repo_root: Path,
    mode: str,
    execute: bool,
    quarantines: Sequence[dict[str, Any]] = (),
    git_sha: str | None = None,
    dirty_state: dict[str, Any] | None = None,
    changed_paths: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Build a report for all release lanes, executing only the requested lane."""

    if mode not in {"baseline", *LANES}:
        raise QualityGateError(f"unknown mode {mode!r}")
    if mode == "baseline" and execute:
        raise QualityGateError("baseline mode records qualification state and cannot execute lanes")

    validated_quarantines = validate_quarantines(quarantines)
    observed_sha = git_sha or current_revision(repo_root)
    observed_dirty_state = dirty_state or current_dirty_state(repo_root)
    dirty_paths = _changed_paths(observed_dirty_state)
    if changed_paths is None:
        observed_paths = dirty_paths
        impacts = select_impacts(observed_paths) if observed_paths else tuple(sorted(COMPONENTS))
    else:
        observed_paths = tuple(sorted({*dirty_paths, *changed_paths}))
        impacts = select_impacts(observed_paths)
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
    }
    lane_results: list[dict[str, Any]] = []

    for lane in LANES:
        components = _components_for_lane(lane, impacts)
        waiver = _matching_quarantine(lane, validated_quarantines)
        task_results: list[dict[str, Any]] = []
        if waiver is not None:
            result = "quarantined"
            detail = "active quarantine validated before lane selection"
        elif mode == "baseline":
            result = "conditional"
            detail = "baseline records lane eligibility without executing qualification commands"
        elif lane != mode:
            result = "conditional"
            detail = f"lane not requested by mode {mode!r}"
        elif not execute:
            result = "conditional"
            detail = "execution was not requested"
        else:
            result, task_results = _execute_components(repo_root, components)
            detail = "all selected components passed" if result == "passed" else "one or more selected components did not pass"

        lane_results.append(
            {
                "lane": lane,
                "capability": {
                    "impacts": list(impacts),
                    "changed_paths": list(observed_paths),
                    "components": list(components),
                    "execution_requested": execute and lane == mode,
                },
                "environment": environment,
                "result": result,
                "detail": detail,
                "artifacts": task_results,
                "waiver": waiver,
            }
        )

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "mode": mode,
        "git_sha": observed_sha,
        "dirty_state": observed_dirty_state,
        "lanes": lane_results,
    }
    validate_report(report)
    return report


def validate_report(
    report: dict[str, Any],
    *,
    repo_root: Path | None = None,
    excluded_paths: Iterable[Path] = (),
) -> None:
    """Validate the subset of JSON Schema needed by every gate and consumer."""

    required = {"schema_version", "generated_at", "mode", "git_sha", "dirty_state", "lanes"}
    missing = sorted(required.difference(report))
    if missing:
        raise QualityGateError(f"report missing required fields: {', '.join(missing)}")
    if report["schema_version"] != REPORT_SCHEMA_VERSION:
        raise QualityGateError(f"unsupported report schema version {report['schema_version']!r}")
    if report["mode"] not in {"baseline", *LANES}:
        raise QualityGateError(f"report has unknown mode {report['mode']!r}")
    if not isinstance(report["git_sha"], str) or not report["git_sha"]:
        raise QualityGateError("report git_sha must be a non-empty string")
    dirty_state = report["dirty_state"]
    dirty_required = {"entries", "tracked_diff_sha256", "untracked_files", "sha256"}
    if not isinstance(dirty_state, dict) or not dirty_required.issubset(dirty_state):
        raise QualityGateError("report dirty_state must contain entries and content fingerprints")
    if not isinstance(dirty_state["entries"], list) or not all(
        isinstance(entry, str) for entry in dirty_state["entries"]
    ):
        raise QualityGateError("report dirty_state entries must be strings")
    if not isinstance(dirty_state["tracked_diff_sha256"], str) or len(dirty_state["tracked_diff_sha256"]) != 64:
        raise QualityGateError("report dirty_state must contain a tracked diff SHA-256 digest")
    untracked_files = dirty_state["untracked_files"]
    if not isinstance(untracked_files, dict):
        raise QualityGateError("report dirty_state untracked_files must be an object")
    for path, fingerprint in untracked_files.items():
        if not isinstance(path, str) or not path or not isinstance(fingerprint, dict):
            raise QualityGateError("report dirty_state contains an invalid untracked file fingerprint")
        if not isinstance(fingerprint.get("sha256"), str) or len(fingerprint["sha256"]) != 64:
            raise QualityGateError("untracked file fingerprints require SHA-256 digests")
        if not isinstance(fingerprint.get("size"), int) or isinstance(fingerprint["size"], bool) or fingerprint["size"] < 0:
            raise QualityGateError("untracked file fingerprints require non-negative byte sizes")
    expected_digest = dirty_state_digest(
        entries=dirty_state["entries"],
        tracked_diff_sha256=dirty_state["tracked_diff_sha256"],
        untracked_files=untracked_files,
    )
    if dirty_state.get("sha256") != expected_digest:
        raise QualityGateError("report dirty_state SHA-256 digest does not match its content fingerprints")
    if repo_root is not None:
        if report["git_sha"] != current_revision(repo_root):
            raise QualityGateError("report git_sha does not match the checked-out revision")
        if dirty_state != current_dirty_state(repo_root, excluded_paths=excluded_paths):
            raise QualityGateError("report dirty_state does not match the current working tree")
    lanes = report["lanes"]
    if not isinstance(lanes, list) or [entry.get("lane") for entry in lanes] != list(LANES):
        raise QualityGateError("report must contain exactly one ordered result for every qualification lane")
    lane_required = {"lane", "capability", "environment", "result", "detail", "artifacts", "waiver"}
    for lane in lanes:
        lane_missing = sorted(lane_required.difference(lane))
        if lane_missing:
            raise QualityGateError(f"lane result missing required fields: {', '.join(lane_missing)}")
        if lane["result"] not in RESULTS:
            raise QualityGateError(f"lane {lane['lane']} has an unknown result {lane['result']!r}")
        capability = lane["capability"]
        if not isinstance(capability, dict):
            raise QualityGateError(f"lane {lane['lane']} capability must be an object")
        capability_required = {"impacts", "changed_paths", "components", "execution_requested"}
        if not capability_required.issubset(capability):
            raise QualityGateError(f"lane {lane['lane']} capability is incomplete")
        for field in ("impacts", "changed_paths", "components"):
            if not isinstance(capability[field], list) or not all(
                isinstance(value, str) for value in capability[field]
            ):
                raise QualityGateError(f"lane {lane['lane']} capability {field} must be a string list")
        if set(capability["components"]).difference(COMPONENTS):
            raise QualityGateError(f"lane {lane['lane']} references an unknown component")
        if not isinstance(capability["execution_requested"], bool):
            raise QualityGateError(f"lane {lane['lane']} execution_requested must be boolean")
        if not isinstance(lane["artifacts"], list):
            raise QualityGateError(f"lane {lane['lane']} artifacts must be a list")
        if lane["result"] == "quarantined" and lane["waiver"] is None:
            raise QualityGateError(f"lane {lane['lane']} is quarantined without a waiver")
        if lane["result"] != "quarantined" and lane["waiver"] is not None:
            raise QualityGateError(f"lane {lane['lane']} has a waiver without a quarantined result")
        if lane["result"] == "quarantined":
            validate_quarantines([lane["waiver"]])
            if lane["lane"] not in lane["waiver"]["lanes"]:
                raise QualityGateError(f"lane {lane['lane']} waiver does not cover that lane")
            if lane["artifacts"]:
                raise QualityGateError(f"lane {lane['lane']} quarantined result cannot include executed artifacts")
            continue
        if lane["result"] == "conditional":
            if capability["execution_requested"] or lane["artifacts"]:
                raise QualityGateError(f"lane {lane['lane']} conditional result must not represent execution")
            continue
        if not capability["execution_requested"]:
            raise QualityGateError(f"lane {lane['lane']} executed result lacks an execution request")
        artifact_results = [artifact.get("result") for artifact in lane["artifacts"] if isinstance(artifact, dict)]
        if len(artifact_results) != len(lane["artifacts"]):
            raise QualityGateError(f"lane {lane['lane']} contains an invalid component artifact")
        if set(artifact.get("component") for artifact in lane["artifacts"]) != set(capability["components"]):
            raise QualityGateError(f"lane {lane['lane']} component artifacts do not match its selected components")
        if lane["result"] == "passed":
            if not lane["artifacts"] or any(result != "passed" for result in artifact_results):
                raise QualityGateError(f"lane {lane['lane']} passed result contains a non-passing component")
            if any(artifact.get("exit_code") != 0 for artifact in lane["artifacts"]):
                raise QualityGateError(f"lane {lane['lane']} passed result is missing a zero exit code")
        elif lane["result"] == "regression":
            if "regression" not in artifact_results:
                raise QualityGateError(f"lane {lane['lane']} regression result lacks a failed component")
        elif lane["result"] == "environment-blocked" and "environment-blocked" not in artifact_results:
            raise QualityGateError(f"lane {lane['lane']} environment-blocked result lacks a blocked component")


def write_report(report: dict[str, Any], *, repo_root: Path, destination: Path) -> None:
    """Atomically write a report below the repository without altering user WIP."""

    root = repo_root.resolve()
    target = (destination if destination.is_absolute() else root / destination).resolve()
    try:
        target.relative_to(root)
    except ValueError as error:
        raise QualityGateError("report destination must be inside the repository") from error
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def validate_frozen_contract_artifact(payload: dict[str, Any], *, expected_revision: str) -> None:
    """Validate the common, dispatch-consumable frozen-contract envelope."""

    missing = sorted(CONTRACT_REQUIRED_FIELDS.difference(payload))
    if missing:
        raise QualityGateError(f"contract missing required fields: {', '.join(missing)}")
    if payload["schema_name"] != CONTRACT_SCHEMA_NAME:
        raise QualityGateError("contract schema_name is unsupported")
    if payload["schema_version"] != CONTRACT_SCHEMA_VERSION:
        raise QualityGateError("contract schema_version is unsupported")
    if payload["status"] != "frozen":
        raise QualityGateError("contract status must be frozen")
    if payload["git_revision"] != expected_revision:
        raise QualityGateError("contract revision does not match the checked-out revision")
    non_empty_strings = ("contract_id", "producer_package")
    if not all(isinstance(payload[field], str) and payload[field].strip() for field in non_empty_strings):
        raise QualityGateError("contract id and producer package must be non-empty strings")
    non_empty_collections = ("symbols", "errors", "consumers", "gate_ids")
    if not all(isinstance(payload[field], list) and payload[field] for field in non_empty_collections):
        raise QualityGateError("contract symbols, errors, consumers, and gate_ids must be non-empty lists")
    if not isinstance(payload["inputs"], dict) or not isinstance(payload["outputs"], dict):
        raise QualityGateError("contract inputs and outputs must be objects")
    if not isinstance(payload["compatibility"], dict) or not payload["compatibility"]:
        raise QualityGateError("contract compatibility must be a non-empty object")


def validate_quality_report_contract(payload: dict[str, Any], *, expected_revision: str) -> None:
    """Validate the frozen quality-report contract against the production runner."""

    validate_frozen_contract_artifact(payload, expected_revision=expected_revision)
    if payload["contract_id"] != "quality-report-v1":
        raise QualityGateError("quality report contract_id is unsupported")
    if payload["producer_package"] != "PKG-MAIN-00-BASELINE-RELEASE":
        raise QualityGateError("quality report contract producer is unsupported")
    report_schema = payload.get("report_schema")
    if not isinstance(report_schema, dict):
        raise QualityGateError("quality report contract schema must be an object")
    expected_schema = {
        "required": ["schema_version", "generated_at", "mode", "git_sha", "dirty_state", "lanes"],
        "dirty_state_required": ["entries", "tracked_diff_sha256", "untracked_files", "sha256"],
        "lane_required": [
            "lane",
            "capability",
            "environment",
            "result",
            "detail",
            "artifacts",
            "waiver",
        ],
        "lane_capability_required": ["impacts", "changed_paths", "components", "execution_requested"],
        "lane_results": list(RESULT_VALUES),
    }
    for field, expected in expected_schema.items():
        if report_schema.get(field) != expected:
            raise QualityGateError(f"quality report contract {field} does not match the runner")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("baseline", *LANES), required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="run the fixed commands for the requested lane")
    parser.add_argument("--quarantine", type=Path, help="JSON file containing owned, expiring quarantine records")
    parser.add_argument(
        "--changed-from",
        help="Git revision used to derive impact paths in a clean CI checkout",
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        if args.mode != "baseline" and not args.execute:
            raise QualityGateError("live qualification lanes require --execute")
        report_target = args.report if args.report.is_absolute() else repo_root / args.report
        starting_revision = current_revision(repo_root)
        starting_dirty_state = current_dirty_state(repo_root, excluded_paths=(report_target,))
        changed_paths = (
            changed_paths_between(repo_root, args.changed_from) if args.changed_from is not None else None
        )
        report = build_report(
            repo_root=repo_root,
            mode=args.mode,
            execute=args.execute,
            quarantines=_load_quarantines(args.quarantine),
            dirty_state=starting_dirty_state,
            changed_paths=changed_paths,
        )
        if args.execute:
            if current_revision(repo_root) != starting_revision:
                raise QualityGateError("qualification commands changed the checked-out revision")
            if current_dirty_state(repo_root, excluded_paths=(report_target,)) != starting_dirty_state:
                raise QualityGateError("qualification commands changed the pre-existing working tree")
        write_report(report, repo_root=repo_root, destination=args.report)
        validate_report(report, repo_root=repo_root, excluded_paths=(report_target,))
    except QualityGateError as error:
        print(f"quality gate error: {error}", file=sys.stderr)
        return 2

    selected_lane = next((lane for lane in report["lanes"] if lane["lane"] == args.mode), None)
    if selected_lane is not None and selected_lane["result"] in {"regression", "environment-blocked"}:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
