# Mochi Agent Maturity Implementation — Root Handoff Prompt

## 使用方式

在 `H:\_python\agent_mochi` 開啟新的 agent 任務，將本文件從「Copy-paste prompt」開始完整貼給接收者。接收者是新任務的 Main/Root，不是某一個 worker package。

本文件只傳遞執行權限、讀取順序與治理流程。若本文件與 canonical JSON 的 package、contract、ownership、DAG、gate 或 stop condition 不一致，以 canonical JSON 為準；若 canonical JSON 與目前程式碼或 deterministic evidence 不一致，停止受影響的 subgraph 並由 Main 修正計畫。

---

## Copy-paste prompt

### Role and objective

You are taking over the Mochi agent-maturity implementation in a new user-owned task. Treat yourself as Main and Root for this task. Keep user-intent interpretation, architecture decisions, shared-seam implementation, contract freezing, integration, final verification, and user communication under Main. A child is only a bounded worker or read-only evaluator; never delegate Root authority.

Implement the validated full-profile Agent Foreman plan for repository `H:\_python\agent_mochi` through verified completion, subject to the authorization and stop conditions below. Do not merely review or summarize the plan. Continue implementing while safe in-scope work remains.

Use the `agent-foreman` skill for planning, dispatch, handoff, integration, evidence, and evaluation. Use the `agent-memory` skill for project-memory discovery. Follow every applicable `AGENTS.md` instruction.

Terminology note: active project memories that restrict “subagents” to research/read/evidence describe Mochi's product runtime. Preserve that product rule. An Agent Foreman implementation worker is a separate host-controlled coding role and may edit only its frozen `owned_paths` under the dispatch and delta-guard protocol below.

### Authorization boundary

The user authorizes you to:

- Modify production, tests, documentation, CI, and contract artifacts only inside the ownership declared by the canonical plan.
- Implement Main-owned packages directly.
- Create bounded subagents for worker-owned packages when the host permits delegation and every dispatch prerequisite is satisfied.
- Run focused, integration, system, runtime, browser, build, lint, and packaging gates required by the plan.
- Create plan-governance artifacts under `artifacts/foreman/` and gate artifacts under the paths declared in the plan.

This pasted handoff prompt is the later explicit implementation and bounded-worker dispatch instruction referenced by the embedded planning dispatches. It is still conditional on host permission, frozen contracts, verified repository/tool access, exclusive ownership, and every package pre-dispatch check.

This authorization does not permit you to:

- Change the product objective or silently expand a package.
- Give a worker a lifecycle, persistence, migration, authorization, security, concurrency, transport, public-contract, producer/consumer, or shared-hotspot seam.
- Modify, clean, stage, revert, overwrite, or delete the protected user WIP listed below.
- Commit, push, create a PR, delete data, or perform destructive cleanup unless the user separately authorizes that action.
- Claim completion from model approval, summaries, skipped gates, missing artifacts, or tests that were not run.

### Mandatory project-memory read order

Read these before editing:

1. `H:\_python\agent_mochi\AGENTS.md` and imported RTK instructions.
2. `H:\_python\agent_mochi\.claude\skills\agent-memory\memories\INDEX.md`.
3. `H:\_python\agent_mochi\.claude\skills\agent-memory\memories\project-status\current-status.md`.
4. `H:\_python\agent_mochi\Mochi_Spec.md` when product-level context is needed.
5. Open only active topic memories relevant to the package being implemented. Use summary search first. Do not preload `archive/` and do not treat archived plans as current truth.

The most relevant active topic memories are:

- Runtime boundaries: `.claude/skills/agent-memory/memories/architecture/chat-goal-workflow-runtime-current-state-2026-06-27.md`
- Multi-agent boundary: `.claude/skills/agent-memory/memories/decisions/multi-agent-unified-runtime-design-2026-06-05.md`
- Context lifecycle: `.claude/skills/agent-memory/memories/project-status/context-management-and-structured-debate-2026-06-04.md`
- Restart and detached exec: `.claude/skills/agent-memory/memories/project-status/agent-run-resource-recovery-and-detached-exec-2026-06-09.md`
- Chat/Goal parity: `.claude/skills/agent-memory/memories/project-status/chat-goal-subagent-runtime-parity-2026-06-30.md`
- Provider/tool calling: `.claude/skills/agent-memory/memories/project-status/ollama-openclaw-transport-analysis-2026-06-29.md`
- Workspace/file safety: `.claude/skills/agent-memory/memories/project-status/protected-workspace-task11-committed-2026-07-23.md`

### Canonical planning read order

Read these after project memory and before editing runtime code:

1. Canonical semantic source: `H:\_python\agent_mochi\docs\architecture\2026-08-04-agent-maturity-subagent-execution-plan.json`
2. Detailed generated execution view: `H:\_python\agent_mochi\docs\architecture\2026-08-04-agent-maturity-implementation-briefs.md`
3. Standard generated plan: `H:\_python\agent_mochi\docs\architecture\2026-08-04-agent-maturity-subagent-execution-plan.md`
4. Original gap roadmap: `H:\_python\agent_mochi\docs\architecture\2026-08-03-agent-maturity-follow-up-implementation-plan.md`
5. Optional second-opinion analysis, read-only: `C:\Users\Xu\.gemini\antigravity\brain\216255df-dc48-46a9-b250-26493d51b706\plan_analysis.md`

The JSON is the only semantic plan source. Never maintain package semantics independently in Markdown. Regenerate Markdown after every canonical JSON change.

### Known starting state

- Repository: `H:\_python\agent_mochi`
- Expected branch: `main`
- Expected revision: `f45a0c631ec9c18b13849ddf18a93a5dbb0f63bc`
- Plan ID: `PLAN-2026-08-04-MOCHI-MATURITY-SUBAGENT-01`
- Plan state: `validated`, not `frozen`, not implemented
- Packages: 15 Main-owned and 13 worker-owned
- Gates: 27 blocking gates, all currently `not_run`
- Structural schedule: 28 all-serial unit slots, 12 unbounded DAG waves, 16 reference slots with one Main plus three workers
- The reference schedule is a structural comparison, not a calendar commitment.

Protected pre-existing user WIP:

- `mochi/runtime/sandbox/linux.py`
- `tests/security/test_os_sandbox.py`

These paths are outside this task. Do not modify, restore, stage, clean, format, or include them in a worker delta. Attribute their initial fingerprints to the user.

Current untracked planning artifacts belong to this planning effort and must be preserved:

- `docs/architecture/2026-08-03-agent-maturity-follow-up-implementation-plan.md`
- `docs/architecture/2026-08-04-agent-maturity-subagent-execution-plan.json`
- `docs/architecture/2026-08-04-agent-maturity-subagent-execution-plan.md`
- `docs/architecture/2026-08-04-agent-maturity-implementation-briefs.md`
- `docs/architecture/render_agent_foreman_briefs.py`
- `docs/architecture/2026-08-04-agent-maturity-root-implementation-handoff.md`

### Shell and editing rules

- Prefix every shell command and every segment of a command chain with `rtk`.
- Prefer `apply_patch` for edits.
- Preserve unrelated dirty state and never use destructive Git commands.
- Use repository-relative paths inside plan, dispatch, handoff, and evidence artifacts.
- Do not stage or commit unless separately authorized.
- If `ruff` or another tool is unavailable, report it as unavailable; do not claim it passed.

### Preflight — do not edit runtime code before this passes

Run:

```powershell
rtk git rev-parse HEAD
rtk git branch --show-current
rtk git status --short
rtk proxy python C:/Users/Xu/.codex/skills/agent-foreman/scripts/validate_plan.py docs/architecture/2026-08-04-agent-maturity-subagent-execution-plan.json
rtk proxy python docs/architecture/render_agent_foreman_briefs.py docs/architecture/2026-08-04-agent-maturity-subagent-execution-plan.json --out docs/architecture/2026-08-04-agent-maturity-implementation-briefs.md
rtk proxy python C:/Users/Xu/.codex/skills/agent-foreman/scripts/render_plan.py docs/architecture/2026-08-04-agent-maturity-subagent-execution-plan.json --out docs/architecture/2026-08-04-agent-maturity-subagent-execution-plan.md
rtk git diff --check
```

Then:

1. Compare the actual revision and dirty state with the starting state above.
2. Fingerprint protected user WIP and all planned ownership scopes before the first implementation edit.
3. Confirm every package `depends_on` matches the dependency graph and that the detailed renderer reports no schedule drift.
4. Confirm no blocking gate is marked passed without a deterministic artifact.
5. If HEAD changed, do not reuse revision-bound contract or dispatch checks. Main must update the canonical revision, all revision-bound checks, derived schedule, and generated documents, then validate again.
6. If protected WIP changed unexpectedly, stop and report the exact delta. Never normalize it.

### Execution architecture

Use `hybrid_main_seams` with the full profile.

Main retains:

- Contract definition and freezing
- Seam Extraction
- Shared lifecycle and persistence
- Migrations and durable state transitions
- Authorization and file-apply semantics
- Cancellation and side-effect commit fences
- Restart adoption and resource policy
- Shared transport and producer/consumer seams
- Worker-delta integration
- Final deterministic verification and completion authority

Workers receive only the 13 bounded dispatch records embedded in the canonical JSON. Worker identity is currently unavailable, so use degraded routing: at most one production file and one test file, no shared/stateful seam, and Main takeover after the first non-transient failure.

If the host cannot verify repository/tool access or enforce ownership, execute that leaf through Main instead of dispatching it.

### Main-owned Seam Extraction order

Perform and freeze these seams before their hotspot integrations:

1. `PKG-MAIN-15-EFFECTIVE-CONTEXT-SEAM`: extract `mochi/agents/effective_context.py` from policy embedded in `mochi/agents/engine.py`.
2. `PKG-MAIN-25-GENERATION-POLICY-SEAM`: extract `mochi/agents/generation_policy.py` from policy embedded in `mochi/agents/react_loop.py`.
3. `PKG-MAIN-38-CANCELLATION-SEAM`: extract `mochi/runtime/cancellation.py` from cancellation decisions embedded in runtime orchestration.
4. `PKG-MAIN-55-RUN-ADOPTION-SEAM`: extract `mochi/runtime/run_adoption.py` and `mochi/runtime/resource_policy.py` from startup policy embedded in `mochi/runtime/service.py`.

The seam package owns the new pure module and characterization tests. The later Main integration package owns the original hotspot. Do not let a seam extraction become a broad hotspot refactor.

### Capacity-aware execution sequence

`package.wave` means earliest dependency eligibility only. It is not dispatch authorization. Main can execute only one Main-owned package at a time.

Use this reference sequence when the host provides one Main slot plus three worker slots; otherwise recompute the schedule from `MAP-EXECUTION-SCHEDULE-V1` before dispatch:

| Slot | Main | Workers that may overlap after all prerequisites freeze |
|---:|---|---|
| 0 | `PKG-MAIN-00-BASELINE-RELEASE` | — |
| 1 | `PKG-MAIN-10-FAILURE-CONTRACT` | — |
| 2 | `PKG-MAIN-05-BROWSER-TEST-INFRA` | `PKG-WORKER-11-FAILURE-SSE` |
| 3 | `PKG-MAIN-15-EFFECTIVE-CONTEXT-SEAM` | `PKG-WORKER-12-FAILURE-WEB` |
| 4 | `PKG-MAIN-20-CONTEXT-LIFECYCLE` | — |
| 5 | `PKG-MAIN-35-SESSION-INDEX-SEAM` | — |
| 6 | `PKG-MAIN-25-GENERATION-POLICY-SEAM` | `PKG-WORKER-31-SESSION-SEARCH` |
| 7 | `PKG-MAIN-30-GENERATION-TOOL-POLICY` | `PKG-WORKER-32-SESSION-API` |
| 8 | `PKG-MAIN-38-CANCELLATION-SEAM` | `PKG-WORKER-21-OPENAI-ADAPTER`, `PKG-WORKER-22-OLLAMA-ADAPTER`, `PKG-WORKER-23-PROVIDER-MATRIX` |
| 9 | `PKG-MAIN-40-CANCELLATION-CONTRACT` | — |
| 10 | `PKG-MAIN-55-RUN-ADOPTION-SEAM` | `PKG-WORKER-41-CANCELLATION-E2E` |
| 11 | `PKG-MAIN-60-SUPERVISOR-EXEC-RESOURCE` | — |
| 12 | `PKG-MAIN-50-FILE-WORKFLOW-SEAMS` | `PKG-WORKER-61-SUPERVISOR-RESTART`, `PKG-WORKER-62-EXEC-RESTART` |
| 13 | `PKG-MAIN-70-MEMORY-SKILL-GOVERNANCE` | `PKG-WORKER-51-FILE-WEB` |
| 14 | — | `PKG-WORKER-71-SKILL-API`, `PKG-WORKER-72-MEMORY-RESTORE` |
| 15 | `PKG-MAIN-90-INTEGRATE-QUALIFY-RELEASE` | — |

Do not claim an approximately ten-wave execution: that would overlap direct dependencies or assume concurrent Main ownership. S/M/L effort is not equal duration, so use the table only as a dependency- and capacity-valid starting schedule.

### Contract freeze before every worker dispatch

The current plan is validated but no worker contract is frozen. Start with Main packages. Do not create a worker merely because its eligibility wave is reached.

For each worker package:

1. Locate its embedded dispatch object by exact `package_id` in the canonical JSON.
2. Confirm every `depends_on` package is integrated.
3. Run every `pre_dispatch_checks` command. All must exit 0.
4. Confirm every contract artifact exists, has `status=frozen`, matches the current Git revision, declares its producer and consumers, and passes its consumer dry-run where applicable.
5. Confirm the package still owns at most one production file and one test file under degraded routing.
6. Confirm no concurrently active package owns an overlapping path.
7. Create a self-contained dispatch JSON under `artifacts/foreman/<PACKAGE-ID>/dispatch.json`, replacing `<PACKAGE-ID>` with the exact package ID. Never create a literal angle-bracket directory. Replace the planning-time `authorization_source` with the exact current user instruction plus this handoff-document path; do not add, omit, or weaken any other dispatch field.
8. Create the dispatch fingerprint manifest before the worker edits.
9. Give the worker only its objective, frozen contracts, exact ownership, prohibited/read-first paths, entrypoints, ordered steps, success/failure cases, commands, required artifacts, stop conditions, and allowed statuses.

Do not expose evaluator-only probes or hidden-oracle material in a dispatch.

### Worker result contract

Accept only `implemented` or `blocked`.

For `implemented`, require:

- Authoritative model identity when the host exposes it; otherwise `resolved_model_id=null` and `model_verification=unavailable`.
- Every changed path listed and within `owned_paths`.
- A diff artifact.
- Raw focused-gate artifacts.
- Every package gate reported with integer exit code and exact observation.
- `blocked_record=null`.

For `blocked`, require:

- The package-linked invariant and gate.
- Exact `file:symbol` or unowned seam.
- Actual result and expected oracle.
- Evidence artifact.
- Exactly one permitted next action.

Reject a prose summary as evidence. Never copy requested model selectors into `resolved_model_id` without authoritative host readback.

### Delta guard and Main integration

Before accepting a worker handoff:

1. Create or read its handoff manifest.
2. Run `guard_delta.py` for the exact package against the dispatch and handoff manifests.
3. Reject changes to prohibited, read-first-only, unknown, or concurrently owned paths.
4. Compare against the initial dirty-state fingerprint so protected user WIP remains attributed to the user.
5. Inspect the full diff and raw gate output as Main.
6. Integrate in dependency order.
7. Run focused gates before aggregate gates.
8. Write the integration artifact as `actor_role=main`; only Main may record `integrated` or final `verified`.

If Main changes more than 60 percent of a worker's changed lines inside worker-owned scope, record a routing warning and stop delegating similar packages until the route is recalibrated.

### Failure and repair policy

- Retry the same worker once only for rate limiting, service outage, child launch failure, repository lock, or timeout before any authorized delta.
- A compile, lint, type, assertion, schema, permission, diff-guard, or runtime-oracle failure is not transient.
- Permit at most two precisely scoped repairs for a noncritical focused-gate failure.
- Return a critical invariant to Main after one precise failed repair.
- If two packages fail against one contract, invalidate that contract and replan only its affected subgraph.
- If an unowned or prohibited path is required, stop as `blocked`; never expand ownership in place.
- Deterministic evidence outranks evaluator, Main, and worker opinions.

Use this failure report shape:

Replace every angle-bracket value with exact evidence. Never return the placeholders literally.

```text
Invariant: <package-linked invariant>
Gate: <package-linked gate>
Location: <file:symbol or exact seam>
Actual: <exact observation or exit code>
Expected: <observable contract oracle>
Artifact: <durable artifact path>
Next action: <one bounded repair or Main takeover>
```

### Evidence and package state

Advance states only when their real conditions hold:

```text
planned -> validated -> frozen -> assigned -> in_progress
        -> implemented -> integrated -> verified
```

Do not mass-edit statuses. Append deterministic evidence; never rewrite an older ledger entry. Store a JSONL evidence ledger under `artifacts/foreman/evidence-ledger.jsonl` with stable evidence IDs, timestamps, actor roles, invariant/gate links, commands, exit codes, exact observations, artifact paths, SHA-256 hashes, and deterministic/advisory classification.

The plan starts with every implementation gate at `not_run`. Leave a gate `not_run` until its exact command has actually executed and its required artifact exists.

### Final integration and evaluation

`PKG-MAIN-90-INTEGRATE-QUALIFY-RELEASE` must wait for every declared dependency, including:

- `PKG-WORKER-21-OPENAI-ADAPTER`
- `PKG-WORKER-22-OLLAMA-ADAPTER`
- `PKG-WORKER-23-PROVIDER-MATRIX`
- `PKG-WORKER-41-CANCELLATION-E2E`
- Both supervisor/exec restart workers
- Session API, file WebGUI, skill API, memory restore, browser/failure projection, and every Main contract/seam package

After Main integration:

1. Run every blocking focused gate declared by integrated packages.
2. Run aggregate release gates through the real production entrypoints.
3. Use a fresh read-only evaluator for applicable critical invariants, preferably a different verified model family.
4. Require at least one new falsification probe per critical invariant with setup, fault injection, production trigger, oracle, anti-oracle, and artifact.
5. Keep evaluator recommendation advisory. A deterministic failure always blocks completion.
6. Build, install, and smoke the packaged artifact where the release plan requires it.

Mark the plan `verified` only when every blocking deterministic gate passes, every required artifact exists, all worker deltas are guarded and integrated, critical evaluator probes are complete, and no unresolved protected-WIP or ownership issue remains.

Keep implementation verification separate from Agent Foreman campaign qualification. Unless 64 unique formal inputs exist, every `governed-blind` candidate trial passes, governed reliability improves over `prose-weak`, and mean cost or latency improves over `main-only`, describe the delegation method's release qualification as provisional. Mock and fresh-context-emulation trials never count as governed reliability evidence.

### Global stop conditions

Stop and report rather than guessing when any of these occurs:

- The canonical plan or detailed renderer fails validation.
- HEAD or revision-bound contracts drift and cannot be safely rebased by Main.
- Actual worker capacity differs from the reference schedule and the schedule has not been recomputed.
- A required contract is missing, stale, incompatible, or not frozen.
- A package needs an unowned/shared/stateful seam.
- A worker scope overlaps another active owner.
- A critical gate lacks an executable oracle or durable artifact.
- Protected user WIP would need to change.
- Completion would require commit, push, destructive cleanup, secrets access, or other authority not granted here.

### Progress and final report

Keep the user informed with concise phase updates. Report package IDs, state changes, gates run, exact results, artifact paths, blockers, Main takeovers, rework ratio warnings, and schedule changes.

Your final response must state:

- Actual route and Main/worker boundaries used
- Packages integrated and verified
- Gates run and deterministic results
- Evaluator diversity and recommendation
- Protected WIP status
- Unresolved risks or provisional limitations
- Whether commit/push was intentionally left undone

Do not stop at a plan review. Begin with mandatory memory reads and preflight, then implement `PKG-MAIN-00-BASELINE-RELEASE` unless a concrete stop condition is observed.
