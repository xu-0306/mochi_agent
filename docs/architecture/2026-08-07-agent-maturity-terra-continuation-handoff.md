# Mochi Agent Maturity - Terra Continuation Handoff

## Purpose

This document is a continuation handoff for a new Terra task. It is not a second semantic plan and does not itself authorize implementation. The receiving task must treat:

- `docs/architecture/2026-08-04-agent-maturity-subagent-execution-plan.json` as the only semantic source of truth.
- the model conversing with the user in the new task as Root and Main.
- Terra as the requested model for that new Root task, not as a child that inherits Root authority from this task.
- deterministic evidence as authoritative over Main, worker, and evaluator opinions.

To start the receiving task, the user should explicitly instruct Terra to read this file and execute the remaining governed plan. Skill invocation alone is planning permission, not dispatch permission.

## Architecture Decision

```json
{
  "selected_mode": "hybrid_main_seams",
  "selected_profile": "full",
  "reasons": [
    "Remaining work crosses memory persistence, restore rollback, skill promotion, provider adapters, browser projection, restart qualification, and release integration.",
    "PKG-MAIN-70 owns durable schemas, migration and rollback seams that cannot be delegated as a bounded leaf.",
    "The canonical plan already contains frozen worker ownership and executable gates for the remaining disjoint leaves."
  ],
  "rejected_modes": [
    {
      "mode": "main_direct",
      "reason": "A single Main may execute sequentially when no child slots exist, but this is not the economical default architecture for the remaining disjoint leaves."
    },
    {
      "mode": "main_plan_weak_leaf",
      "reason": "The remaining roadmap includes shared persistence, migration, producer/consumer, browser, restart, and release seams that must remain Main-owned."
    }
  ],
  "advisor_required": true
}
```

If the receiving host does not authorize or expose bounded child agents, keep the same package boundaries and execute them sequentially through Main takeover. Do not silently widen a worker package.

## Current Snapshot

- Actual repository root: `D:/mochi_agent`.
- Branch: `main`.
- HEAD: `f45a0c631ec9c18b13849ddf18a93a5dbb0f63bc`.
- Canonical plan status: `in_progress`.
- Package counts: 10 `verified`, 9 `integrated`, 11 `planned`.
- Gate counts: 20 `passed`, 15 `not_run`.
- `PKG-WORKER-11-FAILURE-SSE` is verified. Its deterministic gate passed 33 tests and its fresh Terra evaluator passed the legacy ErrorEvent -> chat SSE -> SessionStore replay probe.
- No commit, stage, push, PR, or destructive cleanup was authorized or performed.

The canonical metadata preserves the original `H:/_python/agent_mochi` planning root and starting dirty state. Do not rewrite that historical snapshot merely because the active workspace is now `D:/mochi_agent`. Record the actual continuation root and dirty state in new continuation artifacts.

Protected user paths must remain byte-identical to `artifacts/foreman/baseline-fingerprint.json`:

- `mochi/runtime/sandbox/linux.py`: `bb39bf36f7d2e4953592d41b8f05c55af25deb40040826a58aaf13551577e00f`
- `tests/security/test_os_sandbox.py`: `94406b3f94608c2f30101a3c3615f075ba8a85d76bc6aadffd67f9773e90da7f`

## Mandatory Read Order

1. `AGENTS.md`.
2. `C:/Users/xu/.codex/skills/agent-foreman/SKILL.md` and its routed full-profile references.
3. `.claude/skills/agent-memory/memories/INDEX.md`.
4. `.claude/skills/agent-memory/memories/project-status/current-status.md`.
5. `.claude/skills/agent-memory/memories/project-status/agent-maturity-worker11-failure-sse-progress-2026-08-07.md`.
6. This continuation handoff.
7. Canonical JSON plan.
8. `docs/architecture/2026-08-04-agent-maturity-implementation-briefs.md`.
9. Generated Markdown plan.
10. Open only the active topic memories and `reference/` sources required by the package being executed.

The Main-70 brief lists `reference/drzero/verl/utils/debug/trajectory_tracker.py`, but `reference/drzero/` is absent in this workspace. Record it as unavailable. Do not fetch or fabricate a substitute unless the user separately authorizes network access; use the available Hermes and ZeroClaw references plus production code evidence.

## Preflight

Set the existing local RTK configuration for each managed PowerShell command:

```powershell
$env:CLAUDE_CONFIG_DIR='D:\mochi_agent\.claude'
```

Then run, without editing:

```powershell
rtk git rev-parse HEAD
rtk git branch --show-current
rtk git status --short
rtk proxy D:/mochi_agent/.venv/Scripts/python.exe C:/Users/xu/.codex/skills/agent-foreman/scripts/validate_plan.py docs/architecture/2026-08-04-agent-maturity-subagent-execution-plan.json
rtk git diff --check
```

Before any worker dispatch, also validate the declared frozen contracts, create a dispatch fingerprint, and confirm exclusive ownership. If HEAD differs from the snapshot, invalidate revision-bound dispatch checks and update the affected contracts through Main before dispatch.

Verify both protected-path hashes against the baseline fingerprint. Stop if either differs.

## Phase 0 - Correct State Debt Before New Work

The canonical plan structurally validates, but two package indexes do not match filesystem evidence. Resolve these through append-only corrective evidence; do not hide them by mass-editing statuses.

### 0.1 PKG-WORKER-12-FAILURE-WEB

Current contradiction:

- package status is `integrated`;
- `web/src/lib/failure-presentation.ts` is absent;
- `tests/browser/test_failure_presentation.py` is absent;
- `artifacts/foreman/PKG-WORKER-12-FAILURE-WEB/` is absent;
- `GATE-FAILURE-WEB-MAPPING` is `not_run`.

Required action:

1. Create a Main corrective record that names `INV-FAILURE-CANONICAL`, `GATE-FAILURE-WEB-MAPPING`, the missing owned paths, the false integration index, and one permitted next action.
2. Treat Worker-12 as unimplemented for dispatch and dependency purposes. Do not dispatch Worker-51 or Worker-71 yet.
3. Re-run every Worker-12 pre-dispatch check from the canonical dispatch.
4. Either dispatch a fresh bounded worker with exactly the canonical owned paths, or execute a documented Main takeover if child authorization or isolation is unavailable.
5. Run:

```powershell
rtk pytest tests/browser/test_failure_presentation.py --junitxml=artifacts/quality/failure-web-mapping.xml
```

6. Guard the delta, create the handoff and Main integration record, append deterministic evidence, and use a fresh read-only evaluator for the critical producer/consumer claim.
7. Set Worker-12 to `verified` only after the gate and evaluator probe pass.

Do not reinterpret the old `integrated` index as evidence that implementation exists.

### 0.2 PKG-WORKER-41-CANCELLATION-E2E

Current contradiction:

- `tests/integration/runtime/test_cancellation_restart.py` exists;
- `GATE-CANCELLATION` is already `passed` with 11 tests in Main-40 evidence;
- the Main-40 integration record includes the Worker-41 test;
- Worker-41 package status remains `planned` and has no normalized package handoff/integration record;
- the fresh Main-40 evaluator was blocked by Windows SQLite/result-artifact permissions, so Main-40 remains `integrated`.

Required action:

1. Audit the existing Worker-41 file against its canonical owned/prohibited paths and the original baseline.
2. Record the actual route as Main takeover; do not invent a worker identity or claim a delta guard that did not run.
3. Re-run `GATE-CANCELLATION` and store an immutable package-scoped copy of the result.
4. Run a fresh read-only falsification probe in a writable evaluator directory. Cover pre-commit cancellation, post-commit preservation, restart replay, and duplicate-effect anti-oracles.
5. Create the missing handoff/integration/evaluation artifacts and advance Worker-41 only from deterministic evidence. Advance Main-40 to `verified` only if its own critical invariants and fresh evaluation are satisfied.

### 0.3 Preserve Existing Integrated Seams

Do not reimplement these packages merely because they are not yet `verified`:

- `PKG-MAIN-40-CANCELLATION-CONTRACT`: integrated; awaiting successful fresh evaluation.
- `PKG-MAIN-50-FILE-WORKFLOW-SEAMS`: integrated; awaits `PKG-WORKER-51-FILE-WEB` and `GATE-FILE-WORKFLOW`.
- `PKG-MAIN-55-RUN-ADOPTION-SEAM`, `PKG-MAIN-56-AGENT-RUN-LEASE-STORE`, `PKG-MAIN-57-DURABLE-JOB-IDENTITY`, and `PKG-MAIN-60-SUPERVISOR-EXEC-RESOURCE`: integrated; preserve their seams and close remaining restart qualification through aggregate evidence.
- `PKG-WORKER-61-SUPERVISOR-RESTART` and `PKG-WORKER-62-EXEC-RESTART`: integrated; run `GATE-SUPERVISOR-RESTART` and fresh evaluation before any verified claim.

Inspect and re-run their gates as required, but change production only in response to a precise blocking oracle.

## Phase 1 - Main-70 Durable Governance

Execute `PKG-MAIN-70-MEMORY-SKILL-GOVERNANCE` through Main. It owns persistence, migration, restore rollback, version promotion, and contract freezing.

Owned paths are limited to:

- `mochi/memory/store.py`
- `mochi/learning/skill_library.py`
- `tests/test_memory_store.py`
- `tests/test_skill_library.py`
- `docs/architecture/contracts/memory-record-v2.json`
- `docs/architecture/contracts/skill-version-v2.json`

Never edit `mochi/api/routes/skills.py` inside Main-70.

Implementation order:

1. Trace real MemoryStore and SkillLibrary callers, current durable schemas, migration paths, backup/restore entrypoints, and failure behavior.
2. Define and freeze MemoryRecord v2 with namespace, provenance, revision, supersession, compatibility, checksum, and conflict semantics.
3. Implement atomic backup/restore and migration. A failed restore must leave the active store byte-identical or produce the explicitly frozen compensation state.
4. Define and freeze SkillVersion v2 with immutable history, evidence-bound promotion, rollback by active-version selection, and recorded-run pinning.
5. Keep canonical records separate from derived search/index state.
6. Run focused gates first:

```powershell
# GATE-MEMORY-LIFECYCLE-CORE
rtk pytest tests/test_memory_store.py --junitxml=artifacts/quality/memory-lifecycle-core.xml
# GATE-SKILL-GOVERNANCE-CORE
rtk pytest tests/test_skill_library.py --junitxml=artifacts/quality/skill-governance-core.xml
# GATE-CONTRACT-FREEZE
rtk pytest tests/contracts/test_frozen_contract_artifacts.py --junitxml=artifacts/quality/contracts.xml
```

7. Use fresh evaluator probes for corruption, dry-run conflict, rollback, promotion without evidence, and recorded-run version pinning.
8. Create Main integration and immutable evidence artifacts before changing package status.

Do not run or claim the aggregate memory/skill gates until Worker-71 and Worker-72 exist.

## Phase 2 - Eligible Leaf Branches

Recompute capacity from actual host slots. The table below is dependency eligibility, not a duration promise. If no child slots are authorized, execute the same order sequentially through documented Main takeover.

| Branch | Order | Gate |
|---|---|---|
| Failure/browser | Repair `PKG-WORKER-12-FAILURE-WEB`, then `PKG-WORKER-51-FILE-WEB` | `GATE-FAILURE-WEB-MAPPING`, `GATE-FILE-WORKFLOW` |
| Session | `PKG-WORKER-31-SESSION-SEARCH`, then `PKG-WORKER-32-SESSION-API` | `GATE-SESSION-SEARCH-CORE`, `GATE-SESSION-SEARCH` |
| Provider | `PKG-WORKER-21-OPENAI-ADAPTER`, `PKG-WORKER-22-OLLAMA-ADAPTER`, `PKG-WORKER-23-PROVIDER-MATRIX`; integrate after all focused gates | `GATE-OPENAI-ADAPTER`, `GATE-OLLAMA-ADAPTER`, `GATE-PROVIDER-FIXTURE`, then `GATE-PROVIDER-MATRIX` |
| Cancellation | Normalize `PKG-WORKER-41-CANCELLATION-E2E` evidence | `GATE-CANCELLATION` |
| Memory/skill | `PKG-MAIN-70-MEMORY-SKILL-GOVERNANCE`, then `PKG-WORKER-71-SKILL-API` and `PKG-WORKER-72-MEMORY-RESTORE` | `GATE-MEMORY-LIFECYCLE-CORE`, `GATE-SKILL-GOVERNANCE-CORE`, `GATE-SKILL-GOVERNANCE`, `GATE-MEMORY-LIFECYCLE` |
| Supervisor/restart | Re-qualify integrated `PKG-WORKER-61-SUPERVISOR-RESTART` and `PKG-WORKER-62-EXEC-RESTART` against `PKG-MAIN-60-SUPERVISOR-EXEC-RESOURCE` | `GATE-SUPERVISOR-RESTART` |

Safe initial concurrency after Phase 0, if one Main plus three verified worker slots are available:

- Main: `PKG-MAIN-70-MEMORY-SKILL-GOVERNANCE`.
- Worker candidate 1: `PKG-WORKER-31-SESSION-SEARCH`.
- Worker candidate 2: `PKG-WORKER-21-OPENAI-ADAPTER`.
- Worker candidate 3: `PKG-WORKER-22-OLLAMA-ADAPTER`.

When one slot frees, `PKG-WORKER-23-PROVIDER-MATRIX` is eligible. `PKG-WORKER-32-SESSION-API` waits for Worker-31 integration. `PKG-WORKER-51-FILE-WEB` waits for corrected Worker-12 integration. `PKG-WORKER-71-SKILL-API` waits for both Main-70 and corrected Worker-12. `PKG-WORKER-72-MEMORY-RESTORE` waits for Main-70.

For every leaf, use the canonical dispatch object as the base. Replace only `authorization_source` with the exact receiving user instruction plus this handoff path. Never copy the requested model selector into `resolved_model_id` without authoritative host readback.

## Worker Protocol

For every worker package:

1. Confirm all dependencies are integrated and all consumed contracts are frozen at current HEAD.
2. Confirm the degraded one-production/one-test limit and exclusive ownership.
3. Write `artifacts/foreman/<PACKAGE-ID>/dispatch.json` and a pre-edit fingerprint.
4. Give the worker only frozen contracts, owned/prohibited/read-first paths, counterexamples, exact commands, required artifacts, and stop conditions.
5. Accept only `implemented` or `blocked`.
6. Require a complete changed-path list, diff artifact, raw gate artifact, integer exit codes, and exact observations.
7. Run `guard_delta.py` before Main integration.
8. Return an unowned/shared/stateful seam to Main immediately; never expand ownership in place.
9. Let only Main create integration records and set `integrated` or `verified`.

Retry only rate limits, service outages, launch failures, repository locks, or pre-delta timeouts, once. Compile, type, assertion, schema, permission, diff-guard, and runtime-oracle failures are non-transient.

## Evidence Rules

- Append to `artifacts/foreman/evidence-ledger.jsonl`; never rewrite an earlier line.
- Synchronize new evidence into the canonical plan only through Main, then validate and deterministically render Markdown.
- Never reuse a mutable aggregate artifact as the sole immutable evidence. Run the canonical gate path, then preserve a package-scoped copy under `artifacts/foreman/<PACKAGE-ID>/` and hash that copy.
- A full historical hash audit currently reports five stale hashes because older gates reused paths such as `artifacts/quality/contracts.xml` and later overwrote them. Do not rewrite those historical entries. Record the limitation and use immutable paths from this point forward.
- The Worker-11 post-Main delta guard is truthfully `false`: it observes Main's later `chat.py` takeover and temporary artifacts. The worker patch reverse-check passes. Do not relabel the post-Main guard as passed.
- Keep deterministic gate results and evaluator recommendations separate.

After every canonical JSON change:

```powershell
rtk proxy D:/mochi_agent/.venv/Scripts/python.exe C:/Users/xu/.codex/skills/agent-foreman/scripts/validate_plan.py docs/architecture/2026-08-04-agent-maturity-subagent-execution-plan.json
rtk proxy D:/mochi_agent/.venv/Scripts/python.exe C:/Users/xu/.codex/skills/agent-foreman/scripts/render_plan.py docs/architecture/2026-08-04-agent-maturity-subagent-execution-plan.json --out docs/architecture/2026-08-04-agent-maturity-subagent-execution-plan.md
rtk git diff --check
```

Render a second copy in the system temporary directory and require byte equality.

## Phase 3 - Aggregate Qualification

After leaf integration, run the aggregate gates through real producer/consumer or lifecycle boundaries:

```powershell
rtk pytest tests/browser/test_failure_presentation.py --junitxml=artifacts/quality/failure-ui.xml
rtk pytest tests/test_api_file_ops.py tests/integration/file_ops/test_selective_apply.py tests/browser/test_file_workflow_browser.py --junitxml=artifacts/quality/file-workflow.xml
rtk proxy D:/mochi_agent/.venv/Scripts/python.exe -m pytest tests/test_session_search.py tests/integration/api/sessions/test_session_search_routes.py --junitxml=artifacts/quality/session-search.xml
rtk pytest tests/backends/test_openai_compat.py tests/backends/test_ollama.py tests/backends/test_provider_runtime_qualification.py --junitxml=artifacts/quality/provider-matrix.xml
rtk pytest tests/test_memory_store.py tests/test_memory_backup_restore.py --junitxml=artifacts/quality/memory-lifecycle.xml
rtk pytest tests/test_skill_library.py tests/test_api_skills.py --junitxml=artifacts/quality/skill-governance.xml
rtk pytest tests/integration/agent_runs/test_startup_adoption.py tests/integration/agent_runs/test_exec_restart.py --junitxml=artifacts/quality/supervisor-restart.xml
```

Run `GATE-FAILURE-UI` only after the corrected Worker-12 integration. Run `GATE-PROVIDER-MATRIX` only after Worker-21, Worker-22, and Worker-23 are integrated. Do not infer aggregate success from component gates.

Use fresh read-only evaluators for every applicable critical invariant. Prefer a different verified model family; otherwise disclose `same-family-fresh-context` or `unknown`. Every probe must state setup, fault injection, production trigger, oracle, anti-oracle, artifact, and exit code.

## Phase 4 - Main-90 Release Integration

Start `PKG-MAIN-90-INTEGRATE-QUALIFY-RELEASE` only when every declared dependency is integrated, all blocking focused artifacts exist, and unresolved state debt is closed.

Main-90 must:

1. Guard and review every worker delta.
2. Re-run all blocking focused and aggregate gates.
3. Run fresh falsification probes for the release-critical invariants.
4. Run the release quality gate through the real entrypoint.
5. Build, install, and smoke the packaged artifact when required by the canonical release contract.
6. Produce `artifacts/quality/release.json`, integration records, evaluator records, and the release qualification document.

Run the canonical release gate through its exact entrypoint:

```powershell
# GATE-RELEASE
rtk proxy python scripts/quality_gate/run.py --mode release --report artifacts/quality/release.json
```

Do not mark the campaign fully qualified unless 64 unique formal inputs exist, every governed-blind candidate trial passes, reliability improves over prose-weak, and mean cost or latency improves over main-only. Otherwise describe Foreman campaign qualification as provisional even if the product implementation gates pass.

## Stop Conditions

Stop and report exact evidence when:

- HEAD changes and revision-bound contracts or dispatch checks have not been refreshed.
- either protected-path hash changes;
- the canonical validator or deterministic renderer fails;
- Worker-12's false integration state cannot be corrected without rewriting prior evidence;
- a required contract is missing, stale, incompatible, or not frozen;
- a worker needs an unowned, prohibited, shared, lifecycle, persistence, migration, authorization, security, concurrency, transport, or producer/consumer seam;
- two packages fail against the same frozen contract;
- a blocking gate lacks an executable oracle or durable artifact;
- deterministic evidence conflicts with a model recommendation;
- completion would require commit, push, PR creation, destructive cleanup, secrets access, network downloads, or other authority not explicitly granted.

Use this failure shape:

```text
Invariant: <package-linked invariant>
Gate: <package-linked gate>
Location: <file:symbol or exact seam>
Actual: <exact observation or exit code>
Expected: <observable contract oracle>
Artifact: <durable artifact path>
Next action: <one bounded repair or Main takeover>
```

## Required Final Report

The receiving Terra Main must report:

- actual Main/worker/evaluator route and authoritative model verification;
- every package state transition;
- focused and aggregate gate commands with exact results;
- evaluator diversity and recommendations;
- corrective treatment of Worker-12 and Worker-41 state debt;
- protected-path fingerprint result;
- immutable evidence paths and hashes;
- historical mutable-artifact hash limitation;
- unresolved risks and provisional campaign limitations;
- whether commit, push, and PR creation were intentionally left undone.

## Receiving Task Prompt

Use this text in a new Terra task:

```text
Read and execute docs/architecture/2026-08-07-agent-maturity-terra-continuation-handoff.md as the continuation plan for the governed agent-maturity implementation. You are Main and Root for this new task. Preserve all existing dirty work, correct the documented Worker-12 and Worker-41 state debt first, then continue through the canonical DAG while safe in-scope work remains. Do not commit, push, create a PR, perform destructive cleanup, or access the network without separate authorization. Keep me informed with package IDs, exact gate results, artifacts, blockers, and Main takeovers.
```
