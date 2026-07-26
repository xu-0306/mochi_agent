# Goal Supervisor And Long-Running Autonomy Implementation Plan

Date: 2026-06-21
Scope baseline: lift Mochi from a schedulable `agent-run` runtime to a first-class `goal` supervisor that can honor user-specified runtime durations, survive long unattended runs, restart safely, and support dataset-collection workflows with minimal operator babysitting
Primary memory reference: `.claude/skills/agent-memory/memories/project-status/goal-supervisor-and-long-running-autonomy-2026-06-21.md`

## Summary

This plan turns the current goal-like autonomy discussion into a tracked implementation roadmap.

The target is not merely "longer runs." The target is a durable control plane that can:

- keep a high-level goal alive across multiple run attempts
- honor runtime windows stated directly in the user prompt or goal API, subject to operator policy
- recover from process restarts and transient provider/tool failures
- pause cleanly for approvals or resource exhaustion and auto-resume when allowed
- compact working context into durable memory snapshots and hand work to a fresh worker generation before model quality degrades
- run structured collection jobs for datasets, including forum/dialogue corpora and multimodal-source ingestion
- split the work safely across subagents with explicit write scopes and handoff rules

## Current Gap Snapshot

The current codebase already has useful building blocks:

- `agent-run` create/start/resume/pause/finalize-partial
- recovery payloads and detached exec reattach
- run-policy fields such as `max_wall_clock_sec` and `heartbeat_timeout_sec`
- dataset package export for completed runs

The main blockers are structural:

- there is no first-class durable `goal` entity, only `task` and `agent-run`
- there is no durable contract for prompt-derived `requested_duration`, deadline policy, or long-run runtime budgets
- runtime restart recovery only reattaches detached exec sessions, not active goal/run supervision
- later workflow messages append conversation state but do not automatically trigger fresh execution
- `checkpoint_interval_steps` now has a goal-owned durable checkpoint cadence slice, linked-goal resume can rehydrate a narrowed or missing resume payload from durable handoff, and both task-linked and standalone exec approvals now survive restart through durable `approval_requests` plus a SQLite-backed exec approval store; broader non-goal checkpoint resume semantics are still incomplete
- `agent-run` now has an initial run-level `awaiting_approval` state, but downstream approval resume/authority policy is still incomplete
- there is no context-budget monitor or worker-generation rollover contract for long-running execution
- there is no durable promoted working-memory snapshot for restart/reset handoff
- there is no persisted reconciler for stale/lost/incomplete work
- autonomy still leans on human follow-up to decide the next step instead of default supervisor-owned progression
- current `web_crawl` is intentionally small-scale, not a durable collection engine

## Goal State Architecture

Introduce a goal control plane that sits above `agent-run`.

Minimum durable entities:

1. `Goal`
   - user intent, status, target workspace, capability limits, source manifest
2. `GoalRuntimePolicy`
   - normalized runtime contract such as `requested_duration_sec`, soft/hard stop, checkpoint cadence, handoff cadence, approval mode, and autonomy limits
3. `GoalAttempt`
   - one execution attempt tied to one or more `agent-run` instances
4. `GoalLease`
   - current runtime owner, heartbeat, takeover eligibility
5. `GoalCheckpoint`
   - durable resume cursor, promoted artifacts, unfinished work, source offsets
6. `GoalMemorySnapshot`
   - compact promoted working memory for handoff: current objective, accepted facts, rejected paths, pending actions, important artifacts, and shard progress
7. `GoalWorkerGeneration`
   - one bounded-context worker generation with rollover reason, parent generation, and resume source snapshot
8. `GoalAuditFinding`
   - named operator-visible findings such as `stale_running`, `lost_owner`, `approval_wait_timeout`
9. `CollectorShard`
   - source-specific progress unit for long-running dataset collection and shard-level resume

Keep `agent-run` as the execution primitive. Do not overload it into both "attempt" and "goal" at once.

For v1, `GoalRuntimePolicy` may start as embedded fields on `Goal`, and `GoalWorkerGeneration` may start as a child of `GoalAttempt`, but the persistence contract must still exist explicitly.

## Relevant Files

- `mochi/runtime/models.py`
- `mochi/runtime/store.py`
- `mochi/runtime/service.py`
- `mochi/runtime/recovery.py`
- `mochi/agents/multi_agent/orchestrator.py`
- `mochi/agents/multi_agent/execution_coordinator.py`
- `mochi/tools/web_crawl.py`
- `mochi/tools/registry_factory.py`
- `mochi/runtime/agent_run_packages.py`
- `web/src/app/agent-runs/page.tsx`
- `web/src/lib/api.ts`
- `tests/test_api_runtime.py`
- `tests/test_api_runtime_detached_exec_recovery.py`
- `tests/test_multi_agent_orchestrator.py`

## Reference Implementations To Borrow From

- `reference/openclaw`
  - timeout-derived lease TTL
  - preflight context overflow routing
  - restart-safe run registry persistence
  - run-id rollover while preserving lineage
- `reference/zeroclaw`
  - daemon supervisor and health snapshots
  - heartbeat patrol loop
  - approval manager and emergency stop
  - Git-visible memory snapshots
  - two-tier memory promotion
- `reference/cc-haha`
  - durable and session-only cron split
  - background-agent resume from transcript + metadata
  - stored-session rebinding on reconnect
  - file-backed task claim and stale-run cleanup
- `reference/hermes-agent`
  - structured context compression and handoff summaries
  - resume-pending state distinct from hard reset
  - clean shutdown and takeover markers
- `reference/drzero`
  - secondary reference for checkpoint discipline and provenance-rich dataset rows

## Guardrails

- Do not collapse `goal` and `agent-run` into one shared status field.
- Do not rely on in-memory dictionaries as the only owner-of-record for active long runs.
- Do not start with a giant general web crawler; build source-specific collector adapters with shard-level checkpoints.
- Do not make approvals more permissive just to achieve unattended execution.
- Do not treat any example duration as a product ceiling; user-specified duration is a first-class contract and policy decides whether a requested window is allowed.
- Do not require per-step operator confirmation once a goal is accepted; only stop for explicit approvals, resource/policy boundaries, or operator-visible safety findings.
- Do not let subagents write to overlapping core runtime files in the same phase without an explicit merge owner.
- Keep existing `agent-run` APIs working while the goal layer is introduced incrementally.

## Target Behavior

1. A goal can accept a duration stated directly in the prompt or API, then normalize it into a durable runtime policy.
2. A goal can remain supervised across runtime restarts and multiple worker generations without requiring continuous operator polling.
3. When context or token pressure grows too high, the supervisor persists a checkpoint and compact memory snapshot, then rolls work forward to a fresh worker generation.
4. Approval waits, provider outages, rate limits, and resource starvation transition into explicit recoverable states.
5. The operator can inspect goal health, requested duration, elapsed time, deadline policy, last checkpoint, current generation, pending approvals, and recommended next action.
6. Dataset collection flushes progress incrementally, with per-record provenance and shard-level resume.
7. Subagents can implement isolated workstreams using explicit ownership packets from this plan.

## Goal Duration And Runtime Budget Contract

Every goal needs a normalized runtime contract, not a vague "run longer" flag.

Minimum contract fields:

- `requested_duration_text`
- `requested_duration_sec`
- `runtime_mode` such as `until_complete`, `fixed_duration`, or `until_deadline`
- `soft_deadline_at`
- `hard_stop_at`
- `checkpoint_interval_steps` and/or `checkpoint_interval_sec`
- `generation_refresh_interval_sec`
- `context_handoff_threshold`
- `approval_mode`
- `max_attempt_retries`

Parsing and precedence:

- explicit API fields beat prompt parsing
- prompt parsing beats environment defaults
- operator policy can clamp, reject, or require approval for overly long requests
- no specific duration value is product-hardcoded; policy validates whatever duration the user requests

## Context Compaction, Memory Snapshots, And Worker Handoff

Long-duration execution should not depend on one ever-growing transcript.

The supervisor should:

- watch context pressure, generation elapsed time, checkpoint age, and repeated failure count
- persist `GoalCheckpoint` plus `GoalMemorySnapshot` before context quality degrades
- keep raw transcript as audit history but give fresh workers the compact memory snapshot as their primary resume input
- support worker rollover for reasons such as `context_pressure`, `scheduled_refresh`, `runtime_restart`, `worker_stall`, or `manual_refresh`
- let a fresh worker continue from durable memory even after LLM reset, process restart, or explicit handoff

## Implementation Options For Long-Duration Continuation

Option 1: Supervisor + Worker Generations

- keep the goal supervisor durable and long-lived
- keep each active worker generation bounded in wall-clock time and context growth
- when thresholds are crossed, persist checkpoint plus memory snapshot and start a fresh generation

Option 2: Checkpoint-First Resume

- persist compact state every N steps or on pressure events
- resume from checkpointed stage, unfinished steps, promoted artifacts, and shard offsets
- avoid replaying the entire conversation just to continue work

Option 3: Rolling Transcript Compaction

- keep full transcript for audit
- maintain a promoted compact working memory for active execution
- progressively retire stale detail from active context while preserving raw logs

Option 4: Duration-Governed Goal Policy

- parse duration intent from the prompt
- store soft/hard runtime budgets and refresh cadence in policy
- let the supervisor enforce checkpoint cadence, retry budget, and rollover cadence according to the requested runtime window

Recommended first-wave path:

- ship Option 1 + Option 2 + Option 4 as the baseline
- add Option 3 as an optimization layer once the checkpoint and rollover contracts are stable

## Execution Order

1. Add durable goal schema plus runtime-policy contracts.
2. Add supervisor lease, startup recovery, and maintenance sweeps.
3. Wire durable checkpoint cadence and compact memory snapshot format.
4. Add worker-generation rollover, restart-safe handoff, and approval/autonomy state transitions.
5. Add dataset collector shards and promoted output artifacts.
6. Add operator UX, audit surfaces, and rollout safeguards.

This order is deliberate. Dataset collection should not ship on top of a control plane that still loses ownership on restart, and long-duration execution should not ship before checkpoint plus handoff contracts exist.

## Progress Tracker

Status legend:

- `[ ]` not started
- `[-]` in progress
- `[x]` completed
- `[!]` blocked pending contract or product decision

## Current Implementation Snapshot

As of 2026-06-24, the current control-plane slices below are implemented and verified.

- Durable `Goal` and `GoalAttempt` request/response models exist.
- SQLite persistence for `goals` and `goal_attempts` exists, including list/get/update helpers.
- `RuntimeService` now exposes goal CRUD/lifecycle methods for:
  - create
  - list
  - get
  - start
  - pause
  - resume
  - cancel
- API routes now exist under `/v1/goals`.
- Goal lifecycle now bridges into linked `agent-run` execution for active attempts.
- Packet B foundation has now started:
  - durable `goal_leases`
  - durable `goal_audit_findings`
  - startup recovery for active goals with missing or stale leases
  - maintenance heartbeat / sweep integration in the runtime scheduler loop
  - goal health inspection via `/v1/goals/{goal_id}/health`
  - linked `agent-run` creation/start plus status synchronization back into goal state
  - restart recovery for linked resumable runs
  - deterministic stalling for orphaned linked runs that lost their live worker
- Packet C now has its first implementation slice:
  - goal `run_policy` normalization at create/read/health/goal-to-agent-run bridge time
  - prompt-derived relative duration parsing for obvious runtime phrasing
  - explicit `run_policy` duration fields overriding prompt parsing
  - preservation of unknown `run_policy` keys while normalizing known ones
  - regression coverage for startup recovery immediate-drive and pause/cancel completion races
- Packet C now has a second implementation slice:
  - linked `agent-run` recovery/checkpoint snapshots are persisted into goal attempt and goal summary state
  - goal health now exposes `current_attempt`, `linked_agent_run`, `checkpoint_policy`, and `approval_state`
  - stalled linked runs with persisted approval-pending signals now surface as goal-level `waiting_approval` instead of a generic unexplained stall
  - checkpoint cadence contract fields are visible together with the latest persisted checkpoint snapshot
- Packet B now has a budget-enforcement slice:
  - goal health exposes machine-readable `runtime_budget` state
  - manual start/resume is blocked when hard-stop or retry budget is already exhausted
  - supervisor-owned active goals are moved to a blocked state when requested duration is exhausted
  - `runtime_budget_exhausted` findings are emitted and preserved for operator visibility
- Packet B now has an operator-facing audit lifecycle slice:
  - goal audit findings can now be listed through `/v1/goals/{goal_id}/audit-findings`
  - operators can resolve or close a specific goal audit finding without touching the database directly
  - goal health `open_findings` now shrinks immediately when a finding is resolved or closed through the API
- Packet C now has a third implementation slice:
  - orchestrator can return explicit run-level `awaiting_approval`
  - `agent-run` summary/health preserve approval state from orchestrator metadata
  - goal/attempt sync maps run-level `awaiting_approval` into goal-level `waiting_approval`
- Packet C now has a fourth implementation slice:
  - generic exec approval resolution can now locate a linked `agent-run` that is waiting on that approval
  - resolving a linked exec approval now auto-resumes the linked `agent-run` and goal without requiring a second manual wakeup
  - approved exec results are injected back into persisted controlled-execution checkpoint state so resume can continue without replaying the same approval-gated command
  - rejecting a linked exec approval now fails the linked `agent-run` and propagates failure into the goal attempt instead of leaving a stale waiting state
- Packet C now has a fifth implementation slice:
  - pending linked exec approvals can be rehydrated from persisted `agent-run` approval/checkpoint state after runtime restart
  - `/v1/approvals` now continues to list restart-recovered linked exec approvals instead of silently losing them with the in-memory approval store
  - resolving a restart-recovered linked exec approval still auto-resumes the original linked `agent-run` and linked goal attempt
  - the first restart-safe path is intentionally service-level rehydration, so existing databases need no schema migration
- Packet C now has a sixth implementation slice:
  - durable `goal_checkpoints` and `goal_memory_snapshots` persistence contracts now exist in the runtime store
  - goal-to-run synchronization now persists deduplicated checkpoint records plus compact recovery-oriented memory snapshots for the active attempt
  - goal health now exposes the latest `persisted_checkpoint` and `memory_snapshot`
  - operators can inspect persisted progress through `/v1/goals/{goal_id}/checkpoints` and `/v1/goals/{goal_id}/memory-snapshots`
  - linked goal attempts now claim `agent_run_id` atomically before linked run creation so restart/takeover paths do not spawn duplicate runs for the same attempt
- Packet C now has a seventh implementation slice:
  - durable goal progress records now preserve candidate-selection context such as `selected_candidate_id` and `candidate_count`
  - durable goal progress records now preserve a compact `role_task_summary` derived from the linked run's role-task snapshot
  - goal health now exposes that summarized role/task recovery context through both `persisted_checkpoint` and `memory_snapshot`
- Packet C now has an eighth implementation slice:
  - explicit goal `checkpoint_interval_steps` are no longer synthesized into normalized goal policy when the operator did not request them
  - goal-linked runs can now let explicit `checkpoint_interval_steps` own planned partial+resume cadence without coercing unrelated agent runs into an implicit first-checkpoint stop
  - orchestrator now enforces that step cadence at durable checkpoint capture points and emits step-based recovery reasons that goal auto-resume recognizes as planned refresh
  - goal health now surfaces `checkpoint_policy.interval_steps` alongside the existing time-based checkpoint policy fields
- Packet C now has a ninth implementation slice:
  - durable goal checkpoints now distinguish the internal checkpoint cursor from downstream-promoted artifact refs that later stages can inspect during restart-safe handoff
  - goal health and compact memory snapshots now surface checkpoint promotion mode plus promoted artifact refs
  - linked-goal handoff metadata and resume guidance now carry those promoted artifact refs into continuation context
- Packet C now has a tenth implementation slice:
  - when a linked goal run's live/summary `resume_payload` is missing or has narrowed to `restart_attempt`, runtime can now rebuild a `continue_from_checkpoint` payload from the durable `GoalCheckpoint` plus `GoalMemorySnapshot` handoff state when that durable handoff is still current
  - linked-goal handoff metadata now records `resume_payload_rehydrated_from=durable_goal_checkpoint` when this recovery path is used
- Packet C now has an eleventh implementation slice:
  - task-scoped exec approvals now persist enough command/runtime metadata in durable `approval_requests` to rehydrate their linked exec approval state after runtime restart
  - restart-rehydrated task approvals can now still resolve, execute the approved command, and keep approval execution result/session metadata visible after restart
- Packet C now has a twelfth implementation slice:
  - standalone exec approvals now persist through a durable SQLite-backed exec approval store shared by runtime approval APIs and workspace tool registries
  - restart-recreated runtime services can still list and resolve those standalone exec approvals instead of losing them with a fresh in-memory store
  - approved standalone exec approvals now keep execution result and session metadata visible after restart
- Packet B/C approval-wait telemetry now has a narrow health slice:
  - linked goal approval state now preserves `approval_wait_started_at` and `approval_wait_timeout_sec` when those values can be derived safely from durable linked approval metadata
  - goal health computes `approval_wait_elapsed_sec` at read time, and intentionally omits wait-time fields when linked approval metadata is too sparse to prove them
  - goal health now reprojects non-terminal goal/attempt status from the latest linked `agent-run` row so waiting-approval health stays coherent even if a linked-run projection pass is still settling
- Packet B approval-wait reporting now has a first implementation slice:
  - supervisor now emits report-only `approval_wait_timeout` findings when goal-level approval wait telemetry is durable enough to prove that approval has exceeded its configured wait timeout
  - the finding auto-resolves when approval is no longer pending, the timeout is extended past the current wait age, or durable timing metadata disappears
  - waiting-approval `recommended_next_action` remains `resolve_approval`, but now carries the `approval_wait_timeout` finding code plus wait-age fields when that finding is open
- Packet B/C hardening now has an additional recovery slice:
  - if a goal attempt already claimed `agent_run_id` but the linked `agent_runs` row is missing, supervisor dispatch now recreates or reuses that exact linked run id instead of stalling the goal immediately
  - linked exec approval restart resolution now falls back to non-terminal linked runs that still carry a pending approval entry, instead of requiring the run status to still be exactly `awaiting_approval`
  - restart-resolved approvals now continue to reuse the original linked goal attempt even when the linked run drifted to `stalled` before the runtime came back
- Packet B/C hardening now has a supervisor-concurrency slice:
  - overlapping `_process_goal_supervision()` passes are now serialized with a service-level async lock
  - a fresh foreign `goal_leases` row is no longer overwritten by a non-forced acquire from another runtime owner
  - stale-takeover recovery now defers the first `running`/no-live-worker linked-run stall decision until a second no-progress pass instead of marking `stalled` immediately
  - goal sync now reloads the latest persisted `agent-run` row before projecting status so stale `running` snapshots do not overwrite newer `awaiting_approval` or `awaiting_resources` states
  - shutdown cancellation no longer rewrites already-quiescent linked runs such as `awaiting_approval` into `cancelled` while the runtime is tearing down
  - goal sync now commits current-goal and current-attempt status projection in one store transaction so `/v1/goals/{goal_id}` and `/v1/goals/{goal_id}/health` do not observe a transient in-between state during linked-run projection
- Packet D now has a first implementation slice:
  - runtime store now persists durable `goal_worker_generations`
  - goal health now exposes `current_generation`, including elapsed time, refresh policy visibility, and context-handoff telemetry when debate snapshots exist
  - goal health now exposes a machine-readable `recommended_next_action` derived from approval waits, runtime-budget blocks, checkpoint drift, and worker refresh pressure
  - fresh goal attempts and `continue_from_checkpoint` resumes now inject compact `GoalMemorySnapshot` guidance into the next worker execution request
  - worker generations now open/close with linked run lifecycle and preserve resume-source / handoff metadata
  - linked goal runs can now derive `generation_refresh_interval_sec` into a per-generation `max_wall_clock_sec` + `finalize_partial` stop, then auto-resume from the newly persisted checkpoint/memory snapshot without per-step user intervention
  - when durable handoff is safe, those goal-owned planned refresh resumes now prefer memory-first `restart_attempt` while still flowing through the existing linked-goal resume gate and bookkeeping
  - when `generation_refresh_interval_sec` is not configured, linked goal runs can now also derive `checkpoint_interval_sec` into the same bounded `max_wall_clock_sec` + `finalize_partial` path so checkpoint cadence can drive durable partial+resume refresh without a separate live-worker interruption path
  - linked goal-run `continue_from_checkpoint` resumes now require a structured resume payload plus current-attempt durable handoff coverage; if the durable `GoalMemorySnapshot` / linked `GoalCheckpoint` is missing or lags the run's current checkpoint, runtime downgrades the resume to `restart_attempt` and keeps lease/event/runtime strategy metadata consistent
- Packet C/D linked-goal resume hardening now has a narrow safety slice:
  - linked-goal `continue_from_checkpoint` preparation inside `_prepare_linked_goal_agent_run_resume` now validates that durable handoff inputs are present and current before honoring the requested strategy
  - if the structured resume payload is missing, the current attempt lacks a durable `GoalMemorySnapshot` or linked `GoalCheckpoint`, or the durable checkpoint lags the linked run's current checkpoint, the effective strategy is downgraded to `restart_attempt`
  - `resume_agent_run` now uses that final effective strategy for resume lease metadata, `run_resumed` events, and runtime/orchestrator resume handling so linked-goal resume bookkeeping stays consistent
  - explicit linked-goal `restart_attempt` resumes now keep the restart executor semantics (`resume_payload == {}` at runtime request time) while still reusing the latest compact memory guidance when durable handoff coverage is safe
  - regression coverage in `tests/test_goal_api.py` now verifies both fallback to `restart_attempt` when durable handoff is missing and guidance injection for explicit linked-goal `restart_attempt` resumes
- Packet C/D policy reporting now has an implementation slice:
  - `generation_refresh_interval_sec` now emits report-only `generation_refresh_overdue` findings for active linked runs
  - `checkpoint_interval_sec` now emits `missing_checkpoint` and `checkpoint_overdue` findings for active linked runs, and it now also has a narrow planned partial+resume rollout when it owns the linked run's effective wall-clock budget
  - `debate_context_snapshot` artifacts now surface `context_handoff_threshold` telemetry on `current_generation` and emit report-only `context_handoff_due` findings when the configured threshold is crossed
  - these findings auto-resolve when progress records catch up, context pressure drops below threshold, or the linked run leaves active execution
- Packet D context-handoff telemetry now has a generation-scoping slice:
  - loaded agent-run artifacts now retain `created_at` / `updated_at` so runtime can distinguish snapshot windows
  - `context_handoff_due` now only considers `debate_context_snapshot` artifacts created at or after the current worker generation's `started_at`
  - goal health and supervisor finding reconciliation now use that same generation-local context view, so a fresh generation does not inherit an older generation's high-water snapshot
  - `_open_goal_worker_generation` now resolves both `generation_refresh_overdue` and `context_handoff_due` when a new generation opens
- Packet D now has a narrow stalled-run context-pressure refresh slice:
  - when a linked goal run is already `stalled`, has no live worker, still reports generation-scoped `context_handoff_due`, and passes the linked-goal durable-handoff gate, supervisor now resumes it with source `goal_context_handoff_refresh`
  - this reuses the existing linked-goal resume machinery and does not interrupt live workers
  - when `continue_from_checkpoint` is not safe but durable compact handoff is still usable, the same stalled-run refresh path can now fall back to `restart_attempt` without opening a fresh goal attempt
  - fresh worker generations opened from that path now record rollover reason `context_pressure`
- Packet D now has a narrow worker-stall refresh slice:
  - when a linked goal run is already `stalled`, has no live worker, and passes the linked-goal durable-handoff gate without current context-pressure demand, supervisor now resumes it with source `goal_worker_stall_refresh`
  - this also reuses the existing linked-goal resume machinery and stays limited to already-stalled runs
  - when checkpoint continuation is unavailable but durable compact handoff is still safe, this stalled-run refresh path can now reopen the linked run with `restart_attempt` rather than leaving it stalled
  - fresh worker generations opened from that path now record rollover reason `worker_stall`
- Packet D now has a narrow stalled manual-refresh slice:
  - `POST /v1/goals/{goal_id}/resume` now reuses the current linked run and current goal attempt when the goal is already `stalled`, the linked run still exists in `stalled`, and there is no live worker
  - this path reacquires the goal lease, resumes the same linked run with source `manual_resume`, and avoids opening a redundant fresh goal attempt
  - when the operator does not explicitly request a resume strategy and durable handoff is safe, this non-live-worker manual-refresh reuse path now prefers memory-first `restart_attempt`; explicit operator strategy is still honored
  - when durable handoff context exists, fresh worker generations opened from that path now record rollover reason `manual_refresh`
- Packet D now has a narrow awaiting-resources manual-refresh slice:
  - `POST /v1/goals/{goal_id}/resume` now also reuses the current linked run and current goal attempt when the goal is `awaiting_resources`, the linked run still exists in `awaiting_resources`, and there is no live worker
  - this path keeps the operator/resource-recovery flow on the existing linked run instead of forcing a fresh goal attempt
  - when the operator does not explicitly request a resume strategy and durable handoff is safe, this non-live-worker manual-refresh reuse path now also prefers memory-first `restart_attempt`
  - when durable handoff context exists, fresh worker generations opened from that path also record rollover reason `manual_refresh`
- Packet D now has a narrow paused manual-refresh slice:
  - `POST /v1/goals/{goal_id}/resume` now also reuses the current linked run and current goal attempt when the goal is `paused`, the linked run still exists in `paused`, and there is no live worker
  - this path keeps an intentional operator wake-up on the same linked run instead of forcing a fresh goal attempt
  - when the operator does not explicitly request a resume strategy and durable handoff is safe, this non-live-worker manual-refresh reuse path now also prefers memory-first `restart_attempt`
  - when durable handoff context exists, fresh worker generations opened from that path also record rollover reason `manual_refresh`
- Packet D now has a narrow waiting-approval resume-reuse slice:
  - `POST /v1/goals/{goal_id}/resume` now also reuses the current linked run and current goal attempt when the goal is `waiting_approval`, the linked run still exists in `awaiting_approval`, and there is no live worker
  - this closes the control-plane gap where `waiting_approval` was goal-resumable but still fell through to a redundant fresh goal attempt
  - this slice still does not auto-resolve approvals and does not interrupt live workers; it only keeps the operator wake-up on the same linked run/attempt
- Packet D now has a narrow live-worker manual-refresh slice:
  - `POST /v1/goals/{goal_id}/refresh` now reuses the current linked run and current goal attempt when the goal is `running`, the linked run still exists in `running`, and a live worker is attached
  - this path is intentionally guarded by the linked-goal durable-handoff gate; without a current durable `GoalCheckpoint` plus `GoalMemorySnapshot`, the refresh is rejected instead of interrupting the live worker unsafely
  - when the operator does not explicitly request a resume strategy, this live-worker manual-refresh path currently defaults to memory-first `restart_attempt`
  - implementation pauses the live linked run, then resumes the same linked run/attempt onto a fresh worker generation with rollover reason `manual_refresh`
- Packet D now has a narrow live context-pressure auto-refresh slice:
  - when a linked goal run is still `running`, still has a live worker, reports generation-scoped `context_handoff_due`, and passes the linked-goal durable-handoff gate, supervisor now pauses and resumes that same linked run/attempt automatically with source `goal_context_handoff_refresh`
  - this path intentionally stays memory-first `restart_attempt` for now, so live interruption only happens when a current durable `GoalCheckpoint` plus `GoalMemorySnapshot` already make compact handoff safe
  - the resumed worker generation preserves the existing attempt/run lineage and records rollover reason `context_pressure`
- Packet D compact memory payloads now have a richer handoff slice:
  - `GoalMemorySnapshot` now persists richer compact handoff fields derived from existing persisted artifacts and recovery state, including `accepted_facts`, `rejected_paths`, and `pending_actions`
  - linked-goal handoff metadata and injected resume guidance now include pending actions plus accepted/rejected compact context when those fields are present
  - this is still an artifact-derived compact memory layer, not yet a generalized memory-first resume default or shard-offset collector memory contract
- Workstream 6 now has a first collector-foundation slice:
  - new collector contract helpers now normalize additive shard manifests plus per-record collector provenance without introducing new database tables
  - dataset export now preserves `record.metadata.collector_provenance`, with a narrow fallback that can derive record provenance from a single emitted shard manifest when the producer does not send an explicit per-record payload
  - runtime now persists additive `collector_shard_manifest` artifacts through the existing `agent-run` artifact pipeline
  - a shared `CollectorAdapter` protocol plus `BaseCollectorAdapter` HTTP wrapper now exist for future source-specific adapters, and the wrapper intentionally reuses the existing `_http` retry/backoff/rate-limit implementation instead of introducing a second HTTP stack
  - `attempt_bundle` and `dataset_package` now expose additive `collector_shard_manifests` plus `collector_provenance_manifest` surfaces, and the explicit package contract doc has been updated accordingly
  - goal health, persisted checkpoints, and compact memory snapshots now project additive collector shard state, including report-only `collector_shard_stuck` findings and recent shard offsets
  - collector state hardening now keeps shard progress attempt-scoped, uses artifact-scoped fallback shard ids when producers omit explicit `shard_id`, prefers the freshest progress timestamp over first-populated fields, and refreshes durable goal progress records when collector-only shard progress changes
  - the first real source-specific adapter now exists as `mochi/tools/discourse_topic_adapter.py`, using Discourse topic bootstrap plus post-id batch fetch endpoints to emit bounded cursor-based shard progress, per-post dataset records, and additive shard manifests
  - the new `discourse_topic_collect` tool now carries explicit web/source-capture capability metadata and planner priority so direct topic-collection requests can expose it without relying only on generic tool search
  - dataset export now also accepts collector-emitted `collector_dataset_records`, so one collector run can persist multiple first-class dataset records instead of always collapsing into one synthetic orchestration record
  - orchestrator now harvests collector payloads from tool results during active runs, keeps `collector_shard_manifests` plus `collector_dataset_records` in protocol artifacts, and emits live shard-progress plus live dataset/provenance artifact events mid-run
  - runtime now persists those live shard snapshots and live collector dataset records immediately through append-only run artifacts, and resume payload rebuilding folds the freshest live shard state plus persisted collector dataset records back into recovery protocol artifacts
  - package/export and health-facing shard surfaces now dedupe repeated live snapshots to the freshest manifest per shard instead of surfacing duplicate shard rows
  - current rollout is still intentionally artifact-driven: live shard-progress flush plus live collector dataset-record persistence now exist, but broader adapter coverage and broader interruption/resume semantics are still future work
- Workstream 7 now has a first operator-facing goals slice:
  - WebGUI now exposes a dedicated `/goals` console plus sidebar entry and API wiring for goal list/health/operator-control flows
  - goal retry for failed collector shards now has a first end-to-end slice: runtime exposes `POST /v1/goals/{goal_id}/retry-failed-shard`, linked runs record operator audit log entries for shard retries, and `/goals` now surfaces retry buttons on failed shard rows
  - the current slice now also includes goal-scoped audit history, merged finding history, and compact checkpoint/memory snapshot summaries in the detail pane, but deeper goal detail views, worker-generation drilldown, and richer collector shard drilldown are still pending
- Workstream 5 now has a first capability-policy slice:
  - goal `capability_policy` is now normalized at create/read/health time, preserving unknown keys while deduplicating `allowed_tools`
  - linked goal runs now persist that normalized contract on their summary payload as `goal_capability_policy`
  - model-backed orchestrator invocations now pass `goal_capability_policy.allowed_tools` through `AgentInvocationRequest.tool_allowlist`, so linked-goal tool authority is no longer just a stored field
  - this is intentionally only a first authority slice; broader standing-order rules, approval-policy auto-deny/degrade, and estop controls are still incomplete
- Workstream 5 now has a first unattended approval-policy slice:
  - when a linked goal run is configured with `approval_mode=deny` and the orchestrator returns `awaiting_approval`, runtime now automatically rejects that approval-required execution instead of leaving the goal in `waiting_approval`
  - the linked `agent-run` is failed with a policy reason, pending linked exec approvals are resolved as rejected, and goal projection follows the same failure path
  - this currently covers linked exec approvals only; broader approval-policy matrices or degrade-with-alternative behavior are still incomplete

This is still not a full replay-safe supervisor loop. The goal layer can now drive linked `agent-run` execution, but restart recovery for an already in-flight `agent-run` is not yet equivalent to a true resumable supervisor.

Still missing from the implemented foundation:

- richer compact memory snapshots for long-run handoff beyond the current recovery-oriented baseline
- fresh-worker generation rollover when context quality degrades due to live context/token pressure rather than the current wall-clock-based planned refresh
- broader autonomous live-worker interruption and refresh beyond the current narrow operator-triggered `/refresh` plus context-pressure-triggered handoff paths
- `checkpoint_interval_sec`, `generation_refresh_interval_sec`, and `context_handoff_threshold` now have mixed rollout maturity: both cadence fields can drive a planned partial+resume refresh for linked goal runs when they own the effective wall-clock budget, while live context-pressure thresholds are still health/report-only visibility; `approval_mode` now has a narrow linked-goal auto-deny slice for exec approvals, but broader policy behavior is still incomplete
- goal-level `allowed_tools` now has a narrow normalized + enforced slice for goal-linked model-backed orchestrator invocations, but broader standing-order authority rules, approval-policy auto-deny/degrade, and estop behavior are still incomplete
- `goal_memory_snapshots` are durable and queryable, and explicit/stalled-run/live-context-pressure/goal-owned-planned-refresh `restart_attempt` paths can now reuse their compact guidance when durable handoff is safe, but they are not yet the default resume/handoff input for broader long-run worker refresh

## Subagent Execution Contract

Use this section as the delegation contract when spawning workers.

- Prefer `explorer` agents for code-reading packets and `worker` agents for code changes.
- Every worker must stay inside its declared write scope.
- If a packet touches a shared file, that worker owns the file for the duration of the packet and the main agent integrates follow-up edits.
- Workers must not revert unrelated edits and must assume the tree may be dirty.
- The controller should self-review and keep moving by default; do not wait for per-step human confirmation unless the packet reaches a true approval or safety boundary.
- Each worker handoff must include:
  - changed files
  - tests run
  - blockers or assumptions
  - residual risks
- Suggested worker skill: `superpowers:subagent-driven-development`.

## Delegation Packets

### Packet A: Goal Schema And Store

Agent type: `worker`
Owner: Worker A
Write scope:

- `mochi/runtime/models.py`
- `mochi/runtime/store.py`
- `mochi/runtime/service.py`
- `mochi/api/routes/`
- `tests/test_runtime_store.py`
- `tests/test_goal_api.py`

Deliverable:

- durable goal records
- goal status enum
- goal CRUD and operator actions
- store-level migrations or compatibility path

Exit criteria:

- goals can be created, listed, paused, resumed, cancelled
- goals are stored independently from `agent-run`
- tests cover persistence and status transitions

Status update:

- `[-]` Implemented as a thin control-plane slice.
- `[x]` Durable goal/attempt persistence
- `[x]` Goal CRUD and lifecycle API
- `[x]` Persistence and API flow tests
- `[x]` Real execution launch from goal attempts
- `[ ]` Recovery, lease, and maintenance sweep

### Packet B: Supervisor Lease And Recovery

Agent type: `worker`
Owner: Worker B
Write scope:

- `mochi/runtime/service.py`
- `mochi/runtime/recovery.py`
- `tests/test_api_runtime_detached_exec_recovery.py`
- new runtime-recovery tests as needed

Deliverable:

- startup scan for recoverable goals
- lease ownership model
- maintenance sweep for stale/lost goals
- goal-level runtime budget enforcement across attempts and generations
- health snapshot and audit finding generation

Exit criteria:

- restart recovers or reclassifies active goals deterministically
- duplicate supervisors cannot own the same goal concurrently
- stale ownership is detectable and operator-visible
- budget burn, deadline state, and exhaustion are operator-visible and restart-safe

Status update:

- `[-]` Minimal foundation implemented and hardened for immediate startup drive plus pause/cancel race handling.
- `[x]` Durable goal lease table
- `[x]` Durable goal audit-finding table
- `[x]` Startup recovery for missing/stale leases
- `[x]` Maintenance heartbeat/sweep pass
- `[x]` Goal health inspection surface
- `[-]` Real goal execution ownership handoff
- `[-]` Deterministic takeover of active execution jobs
- `[-]` Audit finding resolution lifecycle

### Packet C: Duration Policy, Checkpoints, And Approval Contract

Agent type: `worker`
Owner: Worker C
Write scope:

- `mochi/runtime/models.py`
- `mochi/runtime/store.py`
- `mochi/runtime/service.py`
- `mochi/api/routes/` if goal policy surfaces need API support
- security config/tests that only affect goal policy
- goal-policy and checkpoint tests

Deliverable:

- prompt/API duration normalization
- durable runtime policy fields
- durable checkpoint cadence
- run-level `awaiting_approval`
- standing-order style policy or per-goal `allowed_tools`
- goal-level resume/finalize logic

Exit criteria:

- duration stated by the operator is visible as a stored contract with enforceable limits
- a goal waiting on approval does not look like an unexplained stall
- checkpoint cadence persists enough state to continue after interruption
- policy is stricter for unattended jobs than for interactive runs

### Packet D: Context Compaction, Memory Snapshots, And Worker Handoff

Agent type: `worker`
Owner: Worker D
Write scope:

- `mochi/agents/multi_agent/orchestrator.py`
- `mochi/agents/multi_agent/execution_coordinator.py`
- `mochi/runtime/service.py`
- `mochi/runtime/recovery.py`
- `tests/test_multi_agent_orchestrator.py`
- new handoff/restart tests as needed

Deliverable:

- context budget monitor
- durable `GoalMemorySnapshot`
- durable `GoalWorkerGeneration` rollover contract
- restart-safe handoff to a fresh worker generation
- autonomous continuation logic after planned refresh or runtime reset

Exit criteria:

- fresh workers can resume from compact memory instead of full transcript replay
- runtime can intentionally refresh worker generations during long-duration goals
- goal health exposes current generation, rollover reason, and resume source

### Packet E: Dataset Collector Pipeline

Agent type: `worker`
Owner: Worker E
Write scope:

- `mochi/tools/`
- `mochi/runtime/agent_run_packages.py`
- new collector modules under `mochi/runtime/` or `mochi/tools/`
- dataset-focused tests

Deliverable:

- collector shard contract
- source adapter interface
- incremental flush + resume
- per-record provenance manifest
- retry/rate-limit wrapper for remote sources

Exit criteria:

- long-running collection survives mid-run interruption without losing all progress
- every exported dataset record includes source/provenance metadata
- adapters can enforce source-specific policy and licensing checks

### Packet F: Operator UX And Goal Inspection

Agent type: `worker`
Owner: Worker F
Write scope:

- `web/src/app/agent-runs/page.tsx` or new goal UI surfaces
- `web/src/lib/api.ts`
- frontend tests/scripts

Deliverable:

- goal list/detail UX
- recovery and audit display
- checkpoint/approval/operator action panel
- distinction between goal state and current run attempt state

Exit criteria:

- operator can tell whether the system is healthy, paused, waiting, stalled, or retrying
- latest checkpoint and active shard are inspectable
- follow-up operator messages can intentionally wake a paused goal

## Workstream 1: Goal Control Plane Foundation

Owner: Worker A
Suggested support: explorer for current store/runtime schemas
Status: [x]

Tasks:

- [x] Add durable `Goal` and `GoalAttempt` models.
- [x] Add store methods for create/list/get/update foundation.
- [x] Add lease-ready scans.
- [x] Introduce goal statuses:
  - `[x]` `created`
  - `[x]` `queued`
  - `[x]` `running`
  - `[x]` `waiting_approval`
  - `[x]` `awaiting_resources`
  - `[x]` `paused`
  - `[x]` `stalled`
  - `[x]` `completed`
  - `[x]` `failed`
  - `[x]` `cancelled`
- [x] Add API routes for goal lifecycle operations.
- [x] Keep existing `agent-run` APIs backward-compatible.

Acceptance criteria:

- goals are durable and queryable independently of transient runtime state
- one goal can reference multiple attempts over time
- operator can inspect lifecycle without reading raw run artifacts

Implementation notes:

- Current `start/resume` create/update goal attempts and can now bridge into linked `agent-run` execution.
- Startup/maintenance supervision now uses durable goal scans plus lease reconciliation (`list_goals` + lease ownership checks) as the first lease-ready scan path instead of requiring a separate precomputed queue.
- Goal orchestration still depends on `agent-run` as the underlying execution primitive; the goal layer is the durable supervisor/control plane above it.

## Workstream 2: Supervisor And Reconciler

Owner: Worker B
Suggested support: explorer for restart and stale-state references in `reference/openclaw` and `reference/zeroclaw`
Status: [x]

Tasks:

- [x] Add startup recovery scan in runtime service.
- [x] Persist a supervisor lease with heartbeat timestamp.
- [x] Add a background maintenance sweep.
- [x] Emit named audit findings:
  - `[x]` `stale_running`
  - `[x]` `lost_owner`
  - `[x]` `missing_checkpoint`
  - `[x]` `checkpoint_overdue` (report-only)
  - `[x]` `generation_refresh_overdue` (report-only)
  - `[x]` `context_handoff_due` (report-only)
  - `[x]` `approval_wait_timeout` (report-only when durable wait timing exists)
  - `[x]` `runtime_budget_exhausted`
  - `[x]` `collector_shard_stuck`
- [x] Enforce goal-level runtime budgets across multiple attempts and worker generations.
- [x] Add a small machine-readable health snapshot for ops tooling, including budget state.

Acceptance criteria:

- runtime restart does not silently forget active goals
- goals with lost owners are surfaced and recoverable
- operator sees concrete findings instead of generic "stuck" status
- long-duration goals consume a durable budget contract rather than implicitly running forever

Implementation notes:

- Current Packet B slice supervises durable lease ownership plus linked `agent-run` dispatch for goal attempts.
- Linked runs in resumable states can now be restarted automatically after goal takeover.
- Linked runs that still claim `running` but have no live in-process worker are now marked `stalled` deterministically instead of being treated as healthy.
- Goal runtime budget is now enforced at the goal control-plane layer for hard stops, requested duration exhaustion, and retry-budget exhaustion.
- Audit finding persistence now has store-level resolve/close lifecycle support, although operator-facing resolution flow is still incomplete.
- It still does not migrate an already in-flight `agent-run` execution loop in a replay-safe way after runtime restart.
- Graceful runtime shutdown drops owned leases so a later runtime can recover through the `lost_owner` path without waiting for stale TTL.
- Stale foreign leases are taken over and surfaced as `stale_running` findings through the goal health endpoint.
- Active linked runs now emit `missing_checkpoint`, `checkpoint_overdue`, and `generation_refresh_overdue` findings when policy drift is observable; `generation_refresh_interval_sec` and a narrow `checkpoint_interval_sec` owner path can now drive planned partial+resume rollover, but the supervisor still does not force live checkpoint flushes or live context-pressure interruption.

## Workstream 3: Duration Policy, Checkpoint, And Resume Contract

Owner: Worker C
Suggested support: explorer for `reference/drzero` checkpoint semantics
Status: [x]

Tasks:

- [x] Normalize prompt/API duration into durable runtime-policy fields.
- [x] Persist runtime budget metadata for:
  - `requested_duration_text`
  - `requested_duration_sec`
  - `runtime_mode`
  - `soft_deadline_at`
  - `hard_stop_at`
  - `generation_refresh_interval_sec`
  - `context_handoff_threshold`
- [x] Preserve unknown `run_policy` keys while normalizing known runtime-policy fields.
- [x] Expose normalized goal runtime policy through read and health surfaces.
- [x] Persist linked recovery/checkpoint snapshots into goal attempt and goal summary state.
- [x] Expose checkpoint policy health plus approval-wait summary through `/v1/goals/{goal_id}/health`.
- [x] Persist durable goal checkpoint records and expose them through health plus `/v1/goals/{goal_id}/checkpoints`.
- [x] Turn `checkpoint_interval_steps` into a real durable checkpoint flush cadence.
  - explicit goal `checkpoint_interval_steps` now survive normalization only when requested, instead of being synthesized as an always-on default for every goal
  - goal-linked runs can now let that explicit step cadence own planned `finalize_partial` refresh without changing unrelated agent-run behavior
  - orchestrator now enforces the cadence at durable checkpoint capture boundaries, and goal auto-resume recognizes the resulting step-based partial stop as planned checkpoint refresh
  - goal health now surfaces `checkpoint_policy.interval_steps` in addition to the existing time-based checkpoint policy fields
- [x] Persist checkpoint metadata for:
  - active stage
  - unfinished steps
  - selected candidates
  - role/task snapshot
  - active collector shard offsets
  - active stage, unfinished steps, selected-candidate context, compact role/task summary, and collector shard offsets now survive in durable goal checkpoint payloads
- [x] Add checkpoint promotion rules:
  - durable goal checkpoints now record whether the checkpoint is only an internal cursor or an internal cursor plus downstream-promoted artifact refs
  - compact memory snapshots, goal health, and linked-goal handoff guidance now surface those promoted artifact refs for downstream restart context
- [x] Ensure resume can continue from checkpoint without replaying stale tool outputs.
  - linked exec approvals now reinject the approved execution result into the saved controlled-execution task snapshot before resuming
  - restart-safe linked exec approval rehydration now resumes the original linked run/attempt after runtime restart instead of forcing a new attempt
  - linked-goal `continue_from_checkpoint` now has a durable-handoff safety gate local to `_prepare_linked_goal_agent_run_resume`; missing structured resume payload/current-attempt durable `GoalMemorySnapshot`/linked `GoalCheckpoint`, or a lagging durable checkpoint, now downgrades to `restart_attempt` instead of attempting an unsafe linked-goal resume
  - when a linked goal run's live/summary resume payload is missing or has narrowed to `restart_attempt`, runtime can now rehydrate a `continue_from_checkpoint` payload from the durable goal checkpoint plus memory snapshot handoff state if that durable handoff is still current
  - task-linked exec approvals now also rehydrate from durable `approval_requests` after runtime restart, so restart no longer drops the task-side exec approval handle or its resolved execution result/session metadata
  - standalone exec approvals now also persist and resolve through a durable SQLite-backed exec approval store, so restart no longer drops standalone approval handles or their resolved execution result/session metadata
  - non-goal `agent-run` resumes now also upgrade legacy or missing structured `resume_payload` state before recovery execution, so checkpoint resumes no longer fall through to an empty payload when a run is not linked to a goal handoff

Acceptance criteria:

- operator-specified duration is queryable after restart
- linked `agent-run` creation preserves the normalized goal runtime contract instead of dropping unknown fields
- interruption after checkpoint does not force full restart by default
- goal health distinguishes approval waits from generic stalls when persisted approval-pending signals exist
- checkpoint data is enough for operator inspection and automatic continuation
- resume semantics are explicit, not inferred from ad hoc transcript replay alone

## Workstream 4: Context Compaction, Worker Handoff, And Autonomous Progression

Owner: Worker D
Suggested support: explorer for transcript-compaction and rollover patterns in `reference/cc-haha`, `reference/hermes-agent`, and `reference/drzero`
Status: [x]

Tasks:

- [x] Add durable `GoalMemorySnapshot` persistence plus goal health/read APIs for compact recovery snapshots.
- [x] Add durable `GoalWorkerGeneration` persistence plus goal health visibility for the current generation.
- [x] Add context/token/elapsed-time budget monitors for active worker generations.
  - elapsed-time and checkpoint-age policy reporting now exist through `current_generation`, `checkpoint_policy`, and supervisor findings, while `checkpoint_interval_sec` can now also drive a bounded planned partial+resume cadence when it owns the linked run budget
  - `debate_context_snapshot` artifacts now surface context-pressure telemetry on `current_generation` and emit report-only `context_handoff_due` findings when the configured threshold is crossed
  - orchestrator now emits live `subagent_runtime` artifact events after each configured-model invocation, runtime persists append-only live subagent runtime snapshots mid-run, and `current_generation` now surfaces generation-scoped observed token/runtime telemetry from those live snapshots
  - `generation_token_refresh_threshold` now surfaces generation-scoped token refresh thresholds plus `token_refresh_due` / `token_refresh_over_threshold` on `current_generation`, emits report-only `generation_token_refresh_due` findings, and can auto-refresh a still-live linked run through a durable-handoff-backed `goal_token_refresh` handoff
- [x] Persist `GoalMemorySnapshot` content for:
  - current objective
  - accepted facts
  - rejected paths
  - pending actions
  - important artifacts
  - active shard offsets
  - the current baseline snapshot now persists objective, attempt/run identity, stage, checkpoint index, selected-candidate context, pending approval ids, unfinished steps, pending actions, recommended resume conditions, compact role/task summary, accepted facts, rejected paths, final-answer preview, and important artifacts
  - active shard offsets now persist in compact memory snapshots
- [x] Add `GoalWorkerGeneration` rollover with reasons:
  - `context_pressure`
  - `scheduled_refresh`
  - `runtime_restart`
  - `worker_stall`
  - `manual_refresh`
  - current implementation now performs a first planned `scheduled_refresh` path for linked goal runs by deriving `generation_refresh_interval_sec` into a bounded `finalize_partial` stop and automatic post-persistence resume
  - when durable handoff is safe, those goal-owned planned refresh resumes now prefer memory-first `restart_attempt` rather than checkpoint-continue as their default strategy
  - when `generation_refresh_interval_sec` is absent, `checkpoint_interval_sec` can now drive that same `scheduled_refresh` path for linked goal runs through a narrow checkpoint-cadence-owned partial+resume rollout
  - already-stalled linked goal runs, narrow non-live-worker manual resumes, the narrow live-worker `/refresh` operator path, and the new supervisor-owned live `context_handoff_due` path can now reopen fresh generations with rollover reasons `context_pressure`, `worker_stall`, or `manual_refresh` when the relevant restart/resume path can safely reuse the current durable handoff
  - supervisor can now also pause+resume a still-live linked run into a fresh `scheduled_refresh` generation when `generation_refresh_overdue` is already observable and durable handoff is current
  - supervisor token-threshold refreshes now reopen fresh `context_pressure` generations when live `subagent_runtime` totals cross `generation_token_refresh_threshold` and durable handoff is current
- [x] Resume from the latest memory snapshot and checkpoint instead of full transcript replay by default.
  - fresh-attempt and `continue_from_checkpoint` handoff now inject compact memory snapshot guidance
  - the new `generation_refresh_interval_sec` planned refresh path now auto-resumes from the newly persisted checkpoint/memory snapshot after a bounded partial stop
  - linked goal-run checkpoint resumes are now safety-gated by current-attempt durable handoff coverage and fall back to `restart_attempt` when the durable handoff is missing or stale
  - explicit linked-goal `restart_attempt` resumes plus non-live-worker manual-refresh, narrow live-worker `/refresh`, narrow live context-pressure or token-pressure auto-refresh, and goal-owned planned-refresh reopen paths can now reuse the latest durable compact memory guidance when durable handoff remains safe
  - linked goal resume preparation now rehydrates legacy or missing structured checkpoint payloads from durable handoff state before execution when the checkpoint path is still safe to use
- [x] Ensure the supervisor keeps progressing autonomously after planned rollover without per-step user confirmation.
  - linked goal runs now auto-resume after the new `generation_refresh_interval_sec` planned partial stop, preferring memory-first `restart_attempt` when durable handoff is safe
  - linked goal runs now also auto-resume after the new `checkpoint_interval_sec` planned partial stop when checkpoint cadence owns the effective wall-clock budget, with the same memory-first preference when durable handoff is safe
  - supervisor can now also auto-refresh a still-live linked run when generation-scoped `context_handoff_due`, `generation_token_refresh_due`, or `generation_refresh_overdue` is already observable and durable handoff is current
  - other non-refresh live-worker refresh paths still need the same autonomous continuation behavior

Acceptance criteria:

- worker refresh is intentional, durable, and operator-visible
- restart/reset can continue from compact memory without requiring the entire historical transcript
- long-duration goals degrade by bounded generation rollover rather than by unbounded context growth

## Workstream 5: Approval And Safety For Unattended Runs

Owner: Worker C
Suggested support: explorer for `reference/zeroclaw` approval and estop contracts
Status: [x]

Tasks:

- [x] Surface goal-level `waiting_approval` status when a stalled linked run carries persisted approval-pending signals.
- [x] Add run-level `waiting_approval` state for linked `agent-run` execution, not just goal projection.
- [x] Resolve linked exec approvals back into the waiting `agent-run`/goal flow without requiring a separate manual resume call.
- [x] Reuse approved controlled-execution results from persisted checkpoint state so operator approval does not double-run the same command.
- [x] Add goal-level `allowed_tools` or standing-order authority rules.
- [x] Add auto-deny or degrade behavior for non-interactive goals when policy requires approval.
- [x] Persist exec approvals durably across service restart so approval-waiting runs do not lose their approval handle after process recovery.
- [x] Add persistent emergency-stop hooks for:
  - [x] stop all goal execution
  - [x] block selected tools
  - [x] block selected domains or network usage
    - global network-usage blocking is now wired through operator `tool_denylist` expansion for linked goal runs
    - per-domain operator domain blocking now persists through `goal_operator_controls`, propagates as `permission_policy.blocked_web_domains`, and is enforced by linked goal-run web/evidence paths
- [x] Add audit logging for approval and estop decisions.

Acceptance criteria:

- unattended mode is safer than interactive mode, not looser
- approvals are resumable and auditable, not only operator-visible
- operators can stop long-running automation without manual DB or process surgery

## Workstream 6: Dataset Collector Pipeline

Owner: Worker E
Suggested support: explorer for source policy/licensing requirements per adapter
Status: [x]

Tasks:

- [x] Add a collector adapter interface for source-specific acquisition.
  - `CollectorAdapter` plus `BaseCollectorAdapter` now provide the first shared adapter contract for future source-specific collectors
  - `DiscourseTopicCollectorAdapter` / `discourse_topic_collect` now provide the first real source-specific adapter/tool on top of that contract
- [x] Add shard-level progress and incremental flush.
  - persisted `collector_shard_manifest` artifacts and package-level shard manifests now provide the first durable shard-progress contract
  - `discourse_topic_collect` now adds bounded cursor-based batch/resume semantics for topic-post shards
  - orchestrator now emits live `collector_shard_manifests`, `collector_record_provenance`, and `collector_dataset_records` artifact events during collection, and runtime persists append-only live shard snapshots plus live collector dataset records before run completion
  - recovery payloads now keep collector shard manifests plus collector dataset records, while resume payload rebuilding rehydrates persisted live collector dataset records and package/export surfaces dedupe repeated live shard snapshots to the freshest per-shard state
  - linked goal retries can now isolate a failed shard into retry guidance plus a narrowed shard manifest, so interrupted collection no longer needs to restart unrelated completed shards by default
- [x] Persist per-record metadata:
  - `[x]` source URL or source id
  - `[x]` collected_at
  - `[x]` adapter name
  - `[x]` tool arguments
  - `[x]` license/policy disposition
  - `[x]` dedupe hash
  - the additive persistence path now exists through `record.metadata.collector_provenance`; the first Discourse adapter now emits those fields consistently for topic-post records
- [x] Add retry/backoff/rate-limit middleware for remote collection tools.
  - `BaseCollectorAdapter` now reuses `mochi.tools._http` for retryable status handling, exponential backoff, and optional rate limiting
- [x] Extend package export so dataset packages include shard/provenance manifests.

Acceptance criteria:

- collection output is resumable and attributable
- failed shards can be retried independently
- exported training data is more than just raw text blobs

Implementation notes:

- Current rollout intentionally reuses the existing `agent-run` artifact/package pipeline instead of introducing new collector tables.
- `mochi/tools/collector_adapter.py` now gives future remote collectors one shared HTTP path instead of each adapter carrying bespoke retry/backoff logic.
- `collector_shard_manifest` artifacts are now persisted and projected into both `attempt_bundle` and `dataset_package`.
- The first package-facing provenance contract is additive: existing consumers can ignore it without breaking, while new collectors can start populating it immediately.
- `mochi/tools/discourse_topic_adapter.py` now provides the first real source-specific adapter and tool, emitting first-class `collector_dataset_records` plus Discourse-topic shard manifests through the existing artifact pipeline.
- `export_run_to_dataset_records(...)` now treats collector-emitted dataset records as first-class outputs, so source collectors can persist multiple records per run without being rewritten into one orchestration sample.
- Orchestrator now harvests collector payloads from tool results, keeps them in recovery `protocol_artifacts`, and emits live shard-progress artifact events during active runs.
- Runtime now persists those live shard snapshots immediately through append-only `collector-shard-live` artifacts, and resume-payload rebuilding folds the freshest live shard state back into recovery artifacts.
- Recent verification passed with:
  - `python -m pytest tests/test_collector_contracts.py tests/test_agent_run_packages.py tests/test_multi_agent_orchestrator.py tests/test_api_runtime.py -k "collector or live_collector_shard_snapshot_before_run_completion or live_shard_events or recovery_payload_keeps_collector_artifacts" -q`
  - `python -m pytest tests/test_discourse_topic_adapter.py tests/test_learning_dataset_exporter.py tests/test_tool_exposure.py tests/test_agent_run_packages.py tests/test_collector_adapter.py tests/test_collector_contracts.py tests/test_multi_agent_orchestrator.py tests/test_api_runtime.py tests/test_goal_api.py -k "collector" -q`

## Workstream 7: Operator UX And Rollout

Owner: Worker F
Suggested support: verification subagent
Status: [x]

Tasks:

- [x] Add goal detail UI or a goal mode within the current workflow UI.
  - a first `/goals` operator console now exists, including structured current-generation, runtime-budget, approval-state, linked-run, and current-attempt drilldown plus a collector shard panel, but it is not yet a full detail-first inspection surface
- [x] Distinguish:
  - goal state
  - current attempt state
  - runtime budget state
  - current worker generation state
  - approval state
  - collector shard state
  - goal state, current attempt, runtime budget, current worker generation, approval state, linked run, supervisor lease, and collector shard state now have dedicated drilldown panels, but the broader detail view still does not fully separate every goal/runtime sub-state
- [x] Surface recovery actions:
  - resume
  - retry failed shard
  - finalize partial
  - cancel
  - estop
  - `resume`, `refresh worker`, `retry failed shard`, `cancel`, global `estop`, and linked-run `finalize partial` are now surfaced in the `/goals` console
  - `recommended_next_action` is now wired into contextual CTAs for refresh / resume / inspect-budget / inspect-shards / inspect-approvals, and goal-linked pending approvals can now be approved or rejected directly from `/goals`
- [x] Add audit/health panels for maintenance findings.
  - runtime budget, approval state, current generation, open findings, checkpoints, memory snapshots, collector shard state, and operator audit log now have first-pass panels in `/goals`
  - the detail view now merges open/resolved/closed findings into a finding-history panel, scopes operator audit history to the selected goal, and renders compact checkpoint/memory snapshot summaries
  - deeper cross-run timelines and richer collector/recovery drilldown are still future work
- [x] Add docs and operator notes for long-running runs on Windows.
  - operator notes now live at `docs/superpowers/goal-supervisor-windows-operator-notes.md`

Acceptance criteria:

- operator can understand system state without reading raw JSON artifacts
- long-running jobs have a clear control surface
- healthy autonomous goals do not require manual wakeups, while paused goals can still be resumed intentionally by the operator
- operator can see when the system refreshed to a new worker generation and why

Implementation notes:

- The current goals console already covers list/health refresh, basic lifecycle actions, and persistent estop/operator controls.
- Goal-scoped audit/finding timeline surfaces plus compact checkpoint/memory snapshot summaries now exist, while deeper checkpoint drilldown and richer collector shard inspection remain future slices.

## Verification Strategy

Minimum verification before calling the foundation usable:

- store-level persistence tests for goals and attempts
- duration parsing and runtime-policy normalization tests
- runtime restart recovery tests
- stale-lease takeover tests
- lifecycle race tests for pause/cancel against concurrently completed linked runs
- checkpoint cadence tests
- memory snapshot compaction tests
- worker-generation rollover and restart handoff tests
- approval wait/resume tests
- shard interruption/resume tests
- goal/operator UI smoke tests

## Not In First Wave

- a full general-purpose internet crawler with unrestricted site coverage
- automatic legal/licensing resolution for every dataset source
- browser-session recovery for every long-running tool type
- multi-host distributed goal scheduling

## Definition Of "Foundation Complete"

We can call the first wave complete when all of the following are true:

- a goal survives runtime restart with recoverable state
- a goal accepts prompt-specified runtime duration and respects soft/hard limits after restart
- a goal can stay active for user-specified runtime windows without constant operator polling, subject to operator policy
- context pressure can trigger durable memory snapshot plus fresh-worker rollover instead of silent quality collapse
- approval/resource waits are explicit and resumable
- long-running collection writes resumable shard progress
- operator can inspect health, runtime budget, current generation, and intervene without low-level manual cleanup
