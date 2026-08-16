# Mochi Agent Maturity - Next-Agent Handoff (2026-08-09)

## Mission

This handoff records the actual post-release state and tells the receiving
agent exactly where to resume. It does not authorize cleanup, staging, commit,
or a rewrite of immutable historical evidence.

The user-authorized host restoration and canonical Main-90 release
qualification are complete. The new result is package-scoped and immutable;
the earlier failed report and integration record remain historical evidence.

## Read First

1. `AGENTS.md`.
2. `.claude/skills/agent-memory/memories/INDEX.md`.
3. `.claude/skills/agent-memory/memories/project-status/current-status.md`.
4. `.claude/skills/agent-memory/memories/project-status/agent-maturity-review-remediation-2026-08-09.md`.
5. [`2026-08-09-agent-maturity-review-handoff.md`](2026-08-09-agent-maturity-review-handoff.md).
6. [`2026-08-04-agent-maturity-subagent-execution-plan.json`](2026-08-04-agent-maturity-subagent-execution-plan.json) - the only semantic plan source of truth.
7. [`integration-release-20260809.json`](../../artifacts/foreman/PKG-MAIN-90-INTEGRATE-QUALIFY-RELEASE/integration-release-20260809.json) and [`release-20260809.json`](../../artifacts/foreman/PKG-MAIN-90-INTEGRATE-QUALIFY-RELEASE/release-20260809.json). The older `integration.json` is historical blocked evidence.

## Current Snapshot

- Repository: `D:\mochi_agent`; `HEAD` is `f45a0c631ec9c18b13849ddf18a93a5dbb0f63bc`.
- The shared working tree is intentionally broad and dirty. All pre-existing/unrelated changes are user-owned.
- Canonical plan: `in_progress` / `resolved`; it is not globally complete because `GATE-FAILURE-UI` remains `not_run`.
- Package state: 20 `verified`, 10 `integrated`, 0 `blocked`.
- Gate state: 34 `passed`, `GATE-FAILURE-UI` `not_run`.
- `PKG-MAIN-90-INTEGRATE-QUALIFY-RELEASE` is `verified`. The formal `GATE-RELEASE` lane passed with no waiver and all five selected components at exit code 0.

### 2026-08-09 remediation update

- The authorized exact lockfile restore installed `tzdata==2026.2`; Windows
  Developer Mode restored the symlink capability; and cached Node was injected
  into the release shell without changing `web/node_modules`.
- The release runner now invokes the local compiler as
  `tsc -p web/tsconfig.json --noEmit --incremental false`. Browser tests use a
  bounded temporary artifact directory, so their harness evidence cannot
  mutate historical Foreman artifacts during the full Python lane.
- The authorized canonical command exited 0. Its release lane records
  `result=passed`, `waiver=null`, `execution_requested=true`, and passing
  contracts, diff, Python, Ruff, and Web components. The immutable report copy
  is
  `artifacts/foreman/PKG-MAIN-90-INTEGRATE-QUALIFY-RELEASE/release-20260809.json`
  with SHA-256
  `578cc1cf2b76f408fb34d47dd1c9ebc6570994db6d1b81ea18e922b19958138d`.
- `integration-release-20260809.json` and an append-only ledger entry record
  the new deterministic pass. Do not modify the historical blocked
  `integration.json`.

- [`2026-08-09-main-90-release-remediation-inventory.md`](2026-08-09-main-90-release-remediation-inventory.md)
  now records the bounded diagnosis and remediation work.
- The Windows strict-session sidecar lock used `msvcrt.LK_LOCK`, whose
  one-second contention retries delayed three FIFO turns to the edge of their
  eight-second completion budget. It now uses bounded 10 ms `LK_NBLCK` retries;
  the focused SessionStore/timeline suite passed 30 tests and the former target
  completed in 2.02 seconds.
- The original diagnostic Python suite passed with six host nodes deselected:
  `2683 passed, 12 skipped, 6 deselected, 3 warnings`. The explicitly
  authorized exact lockfile wheel restore subsequently installed
  `tzdata==2026.2`, and all 12 datetime-tool tests now pass. The sole remaining
  host test blocker is the Windows symlink-privilege test; this is still not a
  release acceptance.
- Full Ruff now passes for `mochi`, `tests`, and `scripts/quality_gate`; the
  cached-Node TypeScript command also passes. Neither result supersedes the
  historical Main-90 report.

## What Is Already Complete

### Agent-maturity implementation and evidence

- Provider adapter/matrix packages (Worker-21/22/23), cancellation seam/Worker-41, Main-60 and restart Workers-61/62, and Skill/Memory Workers-71/72 are verified.
- Session Search (Worker-31) and Session API (Worker-32) are integrated; the historical lineage blocker was reconciled without rewriting the original blocked record.
- Main-70 memory/skill governance is integrated; downstream Worker-71/72 qualification does not change that package to `verified`.
- Main-50 and Worker-51 selective file-approval flow are integrated. The server derives/persists the child manifest, atomically supersedes the parent approval, rejects client patch/digest projection, keeps dependency groups indivisible, and does not auto-retry conflicts.
- Worker-12 failure Web mapping and Worker-11 failure SSE evidence exist; the narrower web-mapping gate passes, while aggregate `GATE-FAILURE-UI` is still not run.

### Findings remediated after the independent review

- `SessionStore.append_event_if()` now preserves the conditional candidate mutation before normalization, fixing the ConversationState durable CAS regression.
- The file-workflow browser test now exercises the production Next route, `TaskPanel`, Zustand store, and API client through Playwright instead of a handwritten `page.setContent()` imitation.
- The locked Web dependency tree was restored without changing `web/package.json` or `web/package-lock.json`. In the restored environment, `tsc -p web/tsconfig.json --noEmit` exits 0.
- Focused evidence: ConversationState repository 15 passed; SessionStore/failure normalization 33 passed; related ConversationState consumers 49 passed (four known generation-status oracle-drift failures remain); server selective approval 12 passed; real aggregate file workflow 13 passed; focused Ruff, Node syntax, and `git diff --check` passed.

The remediation originally preserved the canonical plan, Foreman evidence,
historical ledger, package/gate statuses, and Main-90 result. The later
authorized release pass added a new immutable Main-90 evidence record and
updated only the current plan projection. `PKG-WORKER-51-FILE-WEB` remains
`integrated` because it still lacks independent package qualification.

## Remaining Work

### Main-90 release qualification (completed)

The host prerequisites have been restored and the authorized formal release
lane passed. The report's release result is deterministic and bound to the
current revision and dirty-state fingerprint; it is preserved under the
package-scoped immutable path listed above. No deselection, waiver,
package-manager repair, or dependency-tree mutation substituted for the pass.

The command with six Python deselections remains diagnostic-only. It must never
be reported as a release pass or converted into a permanent deselection/waiver.

### Governed but non-release-completing follow-up

- `GATE-FAILURE-UI` remains `not_run`; do not infer it from Worker-12's narrower mapping gate.
- Main-40 awaits a fresh evaluator with an evaluator-only writable artifact directory. Its deterministic Main probe passes, but it remains `integrated`.
- Worker-12, Worker-31, Worker-32, Worker-51, Main-50, Main-55/56/57, and Main-70 remain deliberately `integrated` rather than being upgraded by inference. Any upgrade needs the package's declared independent qualification, not a later aggregate result.

## Required Preflight

Run read-only checks before planning or editing:

```powershell
$env:CLAUDE_CONFIG_DIR='D:\mochi_agent\.claude'

rtk git rev-parse HEAD
rtk git status --short
rtk git diff --check

rtk proxy D:\mochi_agent\.venv\Scripts\python.exe C:\Users\xu\.codex\skills\agent-foreman\scripts\validate_plan.py docs\architecture\2026-08-04-agent-maturity-subagent-execution-plan.json --ledger artifacts\foreman\continuation-20260807-evidence-ledger.jsonl

rtk proxy D:\mochi_agent\.venv\Scripts\python.exe -m pytest tests\quality\test_quality_gate.py -q
rtk proxy D:\mochi_agent\.venv\Scripts\python.exe -m ruff check scripts\quality_gate\run.py tests\quality\test_quality_gate.py

Get-Command node
```

Known final results: canonical plan+ledger validation passed; quality-gate and
browser regression coverage passed; the complete Python lane passed 2690 tests
with 11 skips; and the formal release report passed every selected component.

After the release-remediation scope is explicitly authorized and completed, the only canonical release command is:

```powershell
rtk proxy D:\mochi_agent\.venv\Scripts\python.exe scripts\quality_gate\run.py --mode release --execute --report artifacts\quality\release.json
```

The older 2026-08-07 continuation handoff omits `--execute`; it is historical and not an equivalent release command.

## Evidence and Historical Caveats

- The continuation ledger is append-only: `artifacts/foreman/continuation-20260807-evidence-ledger.jsonl`. Four old records use `independent_evaluator`; this supported legacy alias must not be normalized by rewriting history.
- Older aggregate evidence paths caused five historical hash mismatches after later overwrites. Preserve those lines and use only package-scoped immutable artifacts for new evidence.
- The Main-00 canonical review has old rendered-plan hashes and old `19 passed / 16 not run` gate counts. It is historical; use the current canonical plan and its validator instead.
- `artifacts/quality/release.json` was overwritten only by the explicitly authorized full release rerun. Its immutable Main-90 copy and new integration record are the current deterministic evidence; the old blocked `integration.json` remains historical-and-unchanged.
- [`2026-08-07-agent-maturity-terra-continuation-handoff.md`](2026-08-07-agent-maturity-terra-continuation-handoff.md) reports obsolete plan counts. Use it only for historical governance/protocol context.

## Do Not Do These Things

- Do not edit `mochi/runtime/sandbox/linux.py` or `tests/security/test_os_sandbox.py`.
- Do not rewrite, delete, normalize, or regenerate immutable historical ledger/evidence to make it look current.
- Do not clean/reset/stage/commit/push the dirty shared workspace.
- Do not mutate `web/node_modules`, use package-manager repair, or delete `.pnpm-store` without a distinct user-authorized dependency scope.
- Do not report release success from focused tests, skipped/unavailable lanes, missing `node`, old contract/diff evidence, or a stale release report.

## Handoff Completion Criteria for the Next Agent

Before returning control, the receiving agent should either:

1. preserve the completed Main-90 release evidence and continue only the independently governed `GATE-FAILURE-UI` or other integrated follow-up packages; or
2. if asked to rerun release, create another new package-scoped immutable evidence record rather than replacing the 2026-08-09 copy.

Every conclusion must name the relevant invariant, exact path/symbol, command and exit code, immutable artifact path, and one bounded next action.
