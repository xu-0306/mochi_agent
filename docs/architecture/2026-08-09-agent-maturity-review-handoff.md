# Mochi Agent Maturity - Review Handoff (2026-08-09)

## Purpose and Review Boundary

This is a read-first audit checklist for an independent reviewing agent. It records the implementation and governance state reached in this workspace; it does **not** authorize a release, a cleanup, dependency repair, or a change to historical evidence.

The reviewer should begin with the semantic source of truth, [`2026-08-04-agent-maturity-subagent-execution-plan.json`](2026-08-04-agent-maturity-subagent-execution-plan.json), then reconcile every conclusion against package-scoped immutable artifacts under `artifacts/foreman/`. Deterministic gate evidence is authoritative over implementation, Main, worker, or evaluator narrative.

Current governed-plan state is `in_progress` / `resolved`:

| Status | Count | Meaning for this review |
| --- | ---: | --- |
| `verified` | 19 packages | Deterministic package evidence and the required qualification currently support the stated package boundary. |
| `integrated` | 10 packages | The seam is accepted into the working tree, but the plan deliberately withholds a `verified` claim. |
| `blocked` | 1 package | Main-90 release qualification is failed and must remain so. |
| Gates | 33 passed, 1 not run, 1 failed | `GATE-FAILURE-UI` is not run; `GATE-RELEASE` failed. |

The workspace is intentionally dirty. Treat unrelated or pre-existing changes as user-owned. Do not use a clean-tree operation to obtain a different result.

## Completed and Integrated Work Inventory

### Verified packages

The canonical plan lists these as `verified`; validate the associated `integration.json` before treating the implementation claim as accepted:

- `PKG-MAIN-00-BASELINE-RELEASE`, `PKG-MAIN-05-BROWSER-TEST-INFRA`, and `PKG-MAIN-10-FAILURE-CONTRACT`.
- `PKG-MAIN-15-EFFECTIVE-CONTEXT-SEAM`, `PKG-MAIN-20-CONTEXT-LIFECYCLE`, `PKG-MAIN-25-GENERATION-POLICY-SEAM`, and `PKG-MAIN-30-GENERATION-TOOL-POLICY`.
- `PKG-MAIN-35-SESSION-INDEX-SEAM` and `PKG-WORKER-11-FAILURE-SSE`.
- `PKG-WORKER-21-OPENAI-ADAPTER`, `PKG-WORKER-22-OLLAMA-ADAPTER`, and `PKG-WORKER-23-PROVIDER-MATRIX`.
- `PKG-MAIN-38-CANCELLATION-SEAM` and `PKG-WORKER-41-CANCELLATION-E2E`.
- `PKG-MAIN-60-SUPERVISOR-EXEC-RESOURCE`, `PKG-WORKER-61-SUPERVISOR-RESTART`, and `PKG-WORKER-62-EXEC-RESTART`.
- `PKG-WORKER-71-SKILL-API` and `PKG-WORKER-72-MEMORY-RESTORE`.

### Accepted seams that remain only integrated

| Package | Why it remains `integrated` | Primary review artifact |
| --- | --- | --- |
| `PKG-WORKER-12-FAILURE-WEB` | Browser/failure mapping is integrated; do not conflate this narrower evidence with the unrun aggregate failure-UI gate. | `artifacts/foreman/PKG-WORKER-12-FAILURE-WEB/integration.json` |
| `PKG-MAIN-40-CANCELLATION-CONTRACT` | Its own fresh evaluator could not persist its SQLite/result artifacts, despite a passing Main-operated production probe. | `artifacts/foreman/PKG-MAIN-40-CANCELLATION-CONTRACT/fresh-evaluator-assessment.json` |
| `PKG-WORKER-31-SESSION-SEARCH` / `PKG-WORKER-32-SESSION-API` | Implementation and aggregate search evidence are accepted, but no independent package qualification upgrades them. | `artifacts/foreman/PKG-WORKER-31-SESSION-SEARCH/integration.json`, `artifacts/foreman/PKG-WORKER-32-SESSION-API/integration.json` |
| `PKG-MAIN-50-FILE-WORKFLOW-SEAMS` | Core server seam is accepted; its record explicitly does not claim an independent evaluator. | `artifacts/foreman/PKG-MAIN-50-FILE-WORKFLOW-SEAMS/integration.json` |
| `PKG-WORKER-51-FILE-WEB` | Browser/UI integration passed, but the package record has no independent evaluator qualification. | `artifacts/foreman/PKG-WORKER-51-FILE-WEB/integration.json` |
| `PKG-MAIN-55-RUN-ADOPTION-SEAM`, `PKG-MAIN-56-AGENT-RUN-LEASE-STORE`, `PKG-MAIN-57-DURABLE-JOB-IDENTITY` | These Main seams are accepted inputs to the later Main-60/restart qualification; do not upgrade their individual status by implication. | Respective `artifacts/foreman/<PACKAGE>/integration.json` |
| `PKG-MAIN-70-MEMORY-SKILL-GOVERNANCE` | Durable governance seam is integrated; Worker-71/72 are separately verified and do not retroactively upgrade Main-70. | `artifacts/foreman/PKG-MAIN-70-MEMORY-SKILL-GOVERNANCE/integration.json` |

## Implementation Surfaces to Inspect

Review behavior at these production-to-test boundaries rather than reviewing generated plans alone.

| Area | What was implemented / must be falsified | Main paths | Focused evidence |
| --- | --- | --- | --- |
| Failure and browser projection | Canonical `FailureEnvelope v1`, legacy event compatibility, SSE/session projection, and failure browser rendering. | `mochi/agents/events.py`, `mochi/agents/failures.py`, `mochi/api/routes/chat.py`, `web/src/lib/failure-presentation.ts` | `artifacts/foreman/PKG-WORKER-11-FAILURE-SSE/`, `artifacts/foreman/PKG-WORKER-12-FAILURE-WEB/` |
| Context, generation, providers | Conservative effective context, bounded recovery, OpenAI-compatible and Ollama projections, and provider runtime qualification matrix. | `mochi/agents/effective_context.py`, `mochi/agents/generation_policy.py`, `mochi/agents/react_loop.py`, `mochi/backends/openai_compat.py`, `mochi/backends/ollama.py` | `tests/backends/test_openai_compat.py`, `tests/backends/test_ollama.py`, `tests/backends/test_provider_runtime_qualification.py`, `artifacts/foreman/PKG-WORKER-23-PROVIDER-MATRIX/` |
| Session search | Strict JSONL source with derived bounded index/search and API projection. | `mochi/sessions/store.py`, `mochi/sessions/index.py`, `mochi/sessions/search.py`, `mochi/api/routes/sessions.py` | `artifacts/foreman/PKG-WORKER-31-SESSION-SEARCH/`, `artifacts/foreman/PKG-WORKER-32-SESSION-API/` |
| Cancellation and durable execution | Commit fence, restart recovery, AgentRun adoption/lease/job identity, no duplicate side effects. | `mochi/runtime/cancellation.py`, `mochi/runtime/run_adoption.py`, `mochi/runtime/store.py`, `mochi/runtime/exec_runtime.py`, `mochi/runtime/exec_sessions.py`, `mochi/runtime/service.py` | `artifacts/foreman/PKG-MAIN-40-CANCELLATION-CONTRACT/`, `artifacts/foreman/PKG-MAIN-60-SUPERVISOR-EXEC-RESOURCE/`, `artifacts/foreman/PKG-WORKER-41-CANCELLATION-E2E/` |
| Server-authoritative selective file approval | Subset preview derives an immutable child manifest server-side, revalidates parent approval, atomically supersedes it, and rejects client-supplied patch/digest projections. Dependency groups remain indivisible and conflicts never auto-retry. | `mochi/api/routes/workspace.py`, `mochi/tools/file_ops.py`, `mochi/runtime/service.py`, `web/src/lib/api.ts`, `web/src/lib/stores/task-store.ts`, `web/src/components/chat/TaskPanel.tsx` | `artifacts/foreman/PKG-WORKER-51-FILE-WEB/integration.json`, `artifacts/quality/file-workflow.xml` |
| Memory and skills | Versioned durable records, backup/restore, evidence-bound skill promotion, rollback and recorded-run pinning. | `mochi/memory/store.py`, `mochi/learning/skill_library.py`, `mochi/api/routes/skills.py` | `artifacts/foreman/PKG-MAIN-70-MEMORY-SKILL-GOVERNANCE/`, `artifacts/foreman/PKG-WORKER-71-SKILL-API/`, `artifacts/foreman/PKG-WORKER-72-MEMORY-RESTORE/` |

## Verified Gate Record

The canonical plan currently reports these passed: baseline; browser infrastructure; failure contract/core/consumer and web mapping; context lifecycle; contract freeze; cancellation; file workflow core/aggregate; generation tool; memory lifecycle core/aggregate; OpenAI and Ollama adapters; provider fixture/matrix; all five seam gates; session index/search core/aggregate; skill governance core/aggregate; supervisor adoption/restart; AgentRun lease CAS; durable-job identity; Main-60 startup integration; and exec restart.

Two gate states must remain explicit:

- `GATE-FAILURE-UI`: `not_run`. Do not infer aggregate failure-UI success from `GATE-FAILURE-WEB-MAPPING`.
- `GATE-RELEASE`: `failed`. See the next section; it is the sole final-release blocker in the governed plan.

Focused selective-approval evidence specifically reports 11 passing runtime/file-subset tests and 13 passing aggregate file-workflow tests (with one pre-existing Starlette/httpx deprecation warning). The relevant files are `artifacts/quality/file-workflow.xml` and `artifacts/foreman/PKG-WORKER-51-FILE-WEB/integration.json`.

## Main-90 Release Qualification: Blocked, Not Passed

`PKG-MAIN-90-INTEGRATE-QUALIFY-RELEASE` is `blocked` with `deterministic_result: failed`. The authoritative record is [`artifacts/foreman/PKG-MAIN-90-INTEGRATE-QUALIFY-RELEASE/integration.json`](../../artifacts/foreman/PKG-MAIN-90-INTEGRATE-QUALIFY-RELEASE/integration.json), backed by [`artifacts/quality/release.json`](../../artifacts/quality/release.json).

| Required release component | Observed result | Review interpretation |
| --- | --- | --- |
| Frozen contracts and diff | Passed in the initial report | Valid positive evidence only for those components. |
| Python suite | The initial release-report Python component exited `2`; a later direct full-suite diagnostic, after declared `typer>=0.12` was installed, exited `1` with `2620 passed, 63 failed, 12 skipped, 3 warnings`. | These are distinct runs, not interchangeable exit codes. Collection completed in the later run, but failures are not yet individually classified as baseline/WIP/regression. Do not call either result a release pass. |
| Whole-tree Ruff | 313 findings: 233 attributed to pre-existing baseline debt; 80 to current WIP/generated files | Attribution is a release-analysis observation, not a waiver or a passing lint result. |
| Web TypeScript | `web/node_modules/.bin/tsc.cmd` missing or unresolvable | No TypeScript diagnostics ran. This is an environment/dependency blocker, not evidence that types pass or fail. |

The release runner was narrowed/hardened in `scripts/quality_gate/run.py`: Windows `.cmd` handling, UTF-8 decoding with replacement, and a local compiler invocation that avoids package-manager mutation. Its regression suite in `tests/quality/test_quality_gate.py` passed 17 tests. That implementation does **not** upgrade the release outcome.

Before any retry, require a bounded remediation plan covering: (1) the 63 Python failures, (2) the policy for baseline versus current-WIP Ruff findings, and (3) deterministic restoration/hydration of the Web dependency tree and local `tsc`. A prior noninteractive `pnpm` attempt altered the ignored `web/node_modules` layout and failed; do not repeat package-manager mutation without a separately authorized repair scope.

For a future authorized release rerun, use the canonical/Main-90 entrypoint with `--execute`; the older continuation handoff omits that flag and is not an equivalent command:

```powershell
rtk proxy D:\mochi_agent\.venv\Scripts\python.exe scripts\quality_gate\run.py --mode release --execute --report artifacts\quality\release.json
```

## Governance and Historical-Evidence Constraints

- The append-only continuation ledger is [`artifacts/foreman/continuation-20260807-evidence-ledger.jsonl`](../../artifacts/foreman/continuation-20260807-evidence-ledger.jsonl). It remains byte-for-byte intact.
- Four historical evaluator records use the old `actor_role: "independent_evaluator"` spelling. This is now a deliberately supported legacy alias for `evaluator` in the external Foreman validator/schema/protocol. Do not rewrite those immutable lines; all new producers should use `evaluator`.
- The historical hash audit reports five stale hashes where old aggregate artifact paths were later overwritten. That limitation is recorded, not erased. New evidence must use package-scoped immutable paths and hashes.
- Session-search evidence was reconciled without rewriting its historical blocked record: `artifacts/foreman/PKG-WORKER-31-SESSION-SEARCH/evidence-reconciliation.jsonl`.
- The canonical JSON plan, rendered Markdown plan, and generated implementation briefs were reconciled together. Review changes as one projection set: `2026-08-04-agent-maturity-subagent-execution-plan.json`, `2026-08-04-agent-maturity-subagent-execution-plan.md`, and `2026-08-04-agent-maturity-implementation-briefs.md`.
- The earlier Main-00 canonical-review record has historical JSON/Markdown hashes and a `19 passed` / `16 not run` gate count that no longer match the current canonical plan (`33 passed`, `1 not run`, `1 failed`). Treat [`artifacts/foreman/PKG-MAIN-00-CANONICAL-REVIEW/review.json`](../../artifacts/foreman/PKG-MAIN-00-CANONICAL-REVIEW/review.json) as historical evidence, not proof that the current rendered plan is byte-identical to that earlier render.
- Main-90's dependency evidence is present, but Main-05 is represented by its evaluator/ledger evidence rather than a package `integration.json`. Review that provenance explicitly instead of assuming every dependency has the same artifact shape.

## Recommended Reviewer Procedure

Start read-only. Set the project RTK configuration for each managed command:

```powershell
$env:CLAUDE_CONFIG_DIR='D:\mochi_agent\.claude'
```

1. Confirm scope and current dirty state:

   ```powershell
   rtk git status --short
   rtk git diff --check
   ```

2. Validate plan plus continuation ledger, then review the package/gate indexes against their referenced integration artifacts:

   ```powershell
   rtk proxy D:\mochi_agent\.venv\Scripts\python.exe C:\Users\xu\.codex\skills\agent-foreman\scripts\validate_plan.py docs\architecture\2026-08-04-agent-maturity-subagent-execution-plan.json --ledger artifacts\foreman\continuation-20260807-evidence-ledger.jsonl
   ```

3. Recheck the release-runner regression coverage without turning it into a release claim:

   ```powershell
   rtk proxy D:\mochi_agent\.venv\Scripts\python.exe -m pytest tests\quality\test_quality_gate.py -q
   rtk proxy D:\mochi_agent\.venv\Scripts\python.exe -m ruff check scripts\quality_gate\run.py tests\quality\test_quality_gate.py
   ```

4. Audit the implementation surfaces in the table above against their focused XML/JSON evidence, handoffs, diff patches, and evaluator artifacts. For a claimed `verified` package, falsify its critical invariant through the production boundary; for an `integrated` package, decide only whether its deliberately stated limitation is accurate.

5. Review the exact Main-90 blocked record last. Report release qualification as failed unless every required lane is newly rerun and passes under an authorized remediation scope.

## Review Questions Requiring an Explicit Finding

- Does the selective-approval flow prevent a client from projecting a replacement patch/digest, and does it preserve dependency-group indivisibility through UI and execution?
- Do provider qualification tests exercise both OpenAI-compatible and Ollama behavior without treating model metadata as a hard context guarantee?
- Do session indexing, cancellation, restart, memory restore, and skill promotion preserve their declared source-of-truth and exactly-once/rollback boundaries under fault injection?
- Are the plan's 19/10/1 status claims supported by the corresponding integration and evaluator evidence, without upgrading an integrated seam by inference?
- Is the Main-90 outcome accurately blocked, with no hidden skip, waived lint finding, or missing TypeScript tool reinterpreted as a pass?
- For each of the 63 Python failures, is the asserted origin evidenced by a bounded reproduction/diff rather than assumed from the dirty worktree?

## Prohibited Actions During This Review

- Do not edit, regenerate, normalize, or delete historical ledger/evidence records merely to satisfy a validator.
- Do not modify `mochi/runtime/sandbox/linux.py` or `tests/security/test_os_sandbox.py`.
- Do not clean/reset/stage/commit/push the shared dirty workspace, or broadly reformat unrelated files.
- Do not repair, install, delete, or otherwise mutate `web/node_modules` as part of a read-only review. Any recovery needs its own authorized scope.
- Do not claim release qualification, Web type-check success, a zero-failure Python suite, or whole-tree Ruff success based on the current evidence.

## Expected Review Deliverable

Return a finding for each review question, including: exact path/symbol, production trigger, observed oracle and anti-oracle, artifact path, command and exit code (if run), whether the result changes a package/gate state, and one bounded next action. Preserve the distinction among `verified`, `integrated`, environment-blocked, and release-blocked states.
