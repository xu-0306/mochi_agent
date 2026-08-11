# Main-90 Release-Remediation Inventory (2026-08-09)

## Scope and status

This is a bounded working inventory for `PKG-MAIN-90-INTEGRATE-QUALIFY-RELEASE`.
It preserves the diagnostic history and records the authorized completion.

The canonical plan remains `in_progress` / `resolved` because the separately
governed `GATE-FAILURE-UI` remains `not_run`. Main-90 is now `verified` and
`GATE-RELEASE` is `passed`. The shared working tree remains dirty. The earlier
blocked integration record is historical and unchanged; the final release
evidence is a new package-scoped immutable copy.

## Python failure inventory

The direct post-Ruff full-suite result before the two fixes below was `8
failed, 2679 passed, 12 skipped, 3 warnings`.  It consisted of five IANA
timezone cases, the Windows symlink capability case, the sessions range
regression, and an intermittent voice playback failure.

The later capture-oriented rerun produced `12 failed, 2675 passed, 12 skipped,
3 warnings` in 334.64 seconds.  Five of its extra results are test-invocation
artifacts: the capture child did not inherit the normal Git environment, so the
frozen-contract tests failed at `git rev-parse HEAD` with exit 128.  The sixth,
the Projects timestamp collision, was a real regression and is remediated
below.  The normal-environment rerun of the contract module plus the Projects
test passed (`8 passed`).

| Node IDs | Classification | Evidence and disposition |
| --- | --- | --- |
| `tests/contracts/test_frozen_contract_artifacts.py::{test_frozen_contract_artifacts_are_revision_bound_and_machine_readable,test_quality_report_contract_matches_the_runner_validator,test_contract_validator_rejects_empty_compatibility,test_quality_report_contract_validator_rejects_lane_schema_drift,test_contract_validator_cli_validates_the_declared_revision}` | Test-invocation environment | The capture runner lacked the normal Git environment. The ordinary `rtk proxy` rerun passed all five; do not record these as product failures. |
| `tests/test_api_projects.py::test_projects_crud_round_trip` | Current regression, remediated | Consecutive create/update calls could receive the same timestamp. `ProjectStore.update_project()` now ensures the new UTC ISO timestamp is strictly later than the stored one; a deterministic regression test was added. |
| `tests/test_datetime_tool.py::{test_datetime_with_timezone,test_datetime_accepts_etc_utc_and_unlisted_iana_timezone,test_datetime_uses_request_timezone_when_argument_is_omitted,test_datetime_explicit_timezone_overrides_isolated_request_contexts,test_datetime_iana_timezone_preserves_dst_rules}` | Host prerequisite, remediated | The exact `uv.lock` wheel for `tzdata==2026.2` (SHA-256 `bbe9af844f658da81a5f95019480da3a89415801f6cc966806612cc7169bffe7`) is now installed in `.venv`; all 12 datetime-tool tests pass. No IANA fallback table or test weakening was used. |
| `tests/test_local_model_discovery.py::test_discover_local_models_does_not_follow_symlink` | Host prerequisite | Windows account lacks symlink privilege (`WinError 1314`). Enable Developer Mode or grant the account symbolic-link privilege; preserve the filesystem-security assertion. |
| `tests/unit/sessions/test_sessions_dir_binding.py::test_tool_workflow_snapshot_range_and_sse_are_storage_scoped` | Current regression, remediated | `zip(selected, selected[1:], strict=True)` rejected the intentionally shorter adjacent sequence. It is explicitly `strict=False`; the focused regression now passes. |
| `tests/test_voice_cli.py::test_voice_async_continuous_mode_queues_next_turn_until_current_turn_done` | Current regression, remediated | The normal completion path waited for generation tasks but not their queued playback. It now waits for the active-turn/playback idle event. The former failure reproduced 3/10 times; the repaired target passed 20 consecutive runs. |

Focused verification after the product fixes:

```powershell
rtk proxy D:\mochi_agent\.venv\Scripts\python.exe -m pytest `
  tests\unit\sessions\test_sessions_dir_binding.py `
  tests\test_voice_cli.py `
  tests\test_api_projects.py -q
# 29 passed, 6 skipped, 1 warning
```

### Latest local verification

The three-preclaimed-turn FIFO test exposed a further Windows scheduling
regression: `msvcrt.LK_LOCK` retries lock contention only once per second.
Concurrent strict-timeline reads therefore could consume the test's eight
second completion budget without violating the FIFO order. `SessionStore` now
uses `LK_NBLCK` with a 10 ms retry interval and the same 10 second overall
timeout. A Windows-specific unit test covers the retry behavior.

```powershell
rtk proxy D:\mochi_agent\.venv\Scripts\python.exe -m pytest `
  tests\test_session_store.py `
  tests\unit\engine\test_timeline_chat_integration.py -q -p no:cacheprovider
# 30 passed; the former FIFO target completed in 2.02 s (previously 8.10 s)

rtk proxy D:\mochi_agent\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider `
  --deselect=tests/test_datetime_tool.py::test_datetime_with_timezone `
  --deselect=tests/test_datetime_tool.py::test_datetime_accepts_etc_utc_and_unlisted_iana_timezone `
  --deselect=tests/test_datetime_tool.py::test_datetime_uses_request_timezone_when_argument_is_omitted `
  --deselect=tests/test_datetime_tool.py::test_datetime_explicit_timezone_overrides_isolated_request_contexts `
  --deselect=tests/test_datetime_tool.py::test_datetime_iana_timezone_preserves_dst_rules `
  --deselect=tests/test_local_model_discovery.py::test_discover_local_models_does_not_follow_symlink
# 2683 passed, 12 skipped, 6 deselected, 3 warnings
```

The latter command is diagnostic-only: its six deselections are host
prerequisites, not a release waiver or an acceptance result.

After that diagnostic run, the explicitly authorized lockfile wheel restore
installed `tzdata==2026.2` into `.venv` without changing project files or
`uv.lock`. `tests\\test_datetime_tool.py -q -p no:cacheprovider` then passed
all 12 tests. The only remaining host test failure is the symlink capability
node.

The next Python qualification must run from the ordinary inherited developer or
CI environment, after the remaining symlink prerequisite is restored. Its expected
outcome must be a zero-exit complete suite; the capture-run result above is a
diagnostic record, not release evidence.

## Static and Web lanes

| Lane | Current evidence | Remaining condition |
| --- | --- | --- |
| Ruff | `rtk proxy D:\mochi_agent\.venv\Scripts\python.exe -m ruff check --no-cache --force-exclude mochi tests scripts\quality_gate` now exits 0 (`All checks passed!`). | Re-run it as part of the authorized release qualification; this local result neither overwrites the historical report nor passes `GATE-RELEASE`. |
| TypeScript | The command using cached Node passed: `C:\Users\xu\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe web\node_modules\typescript\bin\tsc --project web\tsconfig.json --noEmit --incremental false`. | `node` is still absent from the normal host `PATH`. Restore an approved Node runtime path; do not mutate `web/node_modules` or run a package-manager repair. |
| Contracts and diff | The historical Main-90 record is valid only for its original run. | Re-run both under the actual release environment as part of one authorized release qualification. |

## Required authorization and acceptance evidence

Before invoking the canonical release entrypoint, enable/grant the Windows
symlink capability and make the already verified cached Node runtime available
to the invoking shell. The exact locked `tzdata` restore and the formal release
execution have explicit authorization. No ignore, waiver, package-manager
repair, or project/lockfile change may substitute for the remaining capability.

Only after those conditions are complete may Main run:

```powershell
rtk proxy D:\mochi_agent\.venv\Scripts\python.exe `
  scripts\quality_gate\run.py --mode release --execute `
  --report artifacts\quality\release.json
```

Before the authorized, complete, zero-regression execution described below,
Main-90 was not release-qualified.

## Authorized completion (2026-08-09)

The user explicitly authorized installation of the lockfile-pinned
`tzdata==2026.2` into `.venv`, enabled Windows Developer Mode, and authorized
the formal release execution. The restored environment passed the full Python
suite (`2690 passed, 11 skipped, 3 warnings`).

During qualification, the browser harness was made release-safe: the test now
writes its transient screenshots, manifest, and fixture-ready file below a
bounded temporary directory rather than overwriting the historical Foreman
evidence directory. The quality runner now invokes the local compiler with
`-p web/tsconfig.json --noEmit --incremental false`, which both selects the
actual Web project and prevents incremental-cache mutation.

The canonical release command then exited 0. Its `release` lane has
`result="passed"`, `waiver=null`, and `execution_requested=true`; contracts,
diff, Python, Ruff, and Web all passed with exit code 0. The immutable report
copy is
`artifacts/foreman/PKG-MAIN-90-INTEGRATE-QUALIFY-RELEASE/release-20260809.json`
with SHA-256
`578cc1cf2b76f408fb34d47dd1c9ebc6570994db6d1b81ea18e922b19958138d`.

`integration-release-20260809.json` and the append-only
`continuation-20260807-evidence-ledger.jsonl` entry record this deterministic
pass. The existing historical `integration.json` is deliberately untouched.

The preliminary gate attempts exposed that the prior browser test rewrote
`PKG-WORKER-12-FAILURE-WEB/browser-harness/fixture-server.json` and
`manifest.json` during ordinary test execution. The temporary-artifact repair
prevents future writes there. The final report fingerprints the then-current
worktree; no prior blocked integration result was upgraded from those mutable
files.
