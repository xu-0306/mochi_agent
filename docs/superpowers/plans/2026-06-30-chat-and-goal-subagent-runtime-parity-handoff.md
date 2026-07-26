# Chat And Goal Subagent Runtime Parity Handoff

Date: 2026-06-30

This file is a safe handoff companion to `docs/superpowers/plans/2026-06-29-chat-and-goal-subagent-runtime-parity.md`.
The original plan has some encoding-corrupted text, so avoid bulk rewriting it. Use this file for current status and next tasks.

## Current Status

The core first-wave implementation plus the next parity hardening pass are in the working tree:

- Backend transcript normalization and persistence exist.
- AgentRun and session subagent read/message APIs exist.
- AgentRun event read/SSE APIs exist.
- Runtime persists normalized AgentRun subagent events.
- Normal chat can create first-wave delegated subagent lifecycle/progress transcripts through `delegate_subagent_task`.
- Normal chat delegated background tasks now bridge real `MultiAgentOrchestrator` runtime events into session-scoped subagent transcripts when a parent session is available.
- Open normal-chat `/chat/stream` responses now subscribe to a best-effort in-memory delegated subagent live event bus and project matching background runtime events back into the same chat SSE stream.
- Session-scoped subagents now have cancel/resume action APIs wired to delegated task `cancel_task` / `resume_task`, with optional resume guidance.
- Main chat renders execution timeline and subagent cards.
- Subagent drawer shows prompt, timeline, output, guidance, and approval metadata.
- Subagent drawer now has quiet Cancel/Resume controls for session-scoped delegated subagents.
- Subagent drawer now has a direct action that opens the existing approval surface scoped to approval ids found in `approval_id`, `approval_ids`, or `pending_approvals`.
- Goal follow-up now forwards guidance and resumes/refreshes the linked run when safe.
- Approval-required subagent/tool events preserve approval metadata and show an approval banner in the drawer.
- A deterministic browser fixture route exists for timeline/drawer/approval/guidance layout QA.
- Session reload persistence has focused browser coverage.

## Completed Tasks From Original Plan

Treat these original-plan tasks as implemented enough for this handoff:

- Task 1: Backend transcript models and tests.
- Task 2: RuntimeStore subagent transcript persistence.
- Task 3: Orchestrator/runtime event persistence.
- Task 4: AgentRun transcript read APIs.
- Task 5: Goal blocker diagnostics.
- Task 6: Project AgentRun events into main chat.
- Task 7: Subagent timeline card and drawer.
- Task 8: Guidance affects resume.
- Task 9: Normal chat first-wave subagent delegation.
- Task 10: Chat-scoped subagent APIs.
- Task 11: Unified chat/goal subagent UI.
- Task 12: Goal live continuation behavior.
- Task 13: AgentRun events SSE.
- Task 14: Permission and approval metadata integration.
- Task 16: Visual/interaction QA now includes deterministic browser fixture coverage plus source-level smoke coverage.

Task 15 and Task 17 still need final review/packaging work. Full normal-chat parity is closer now: real delegated runtime events are persisted, best-effort live events are projected into the open chat SSE stream, session-scoped cancel/resume exists, durable subagent guidance is replayed into delegated task resume, and explicit subagent tool call/result events can be persisted/projected.

## Key Files

Backend:

- `mochi/runtime/execution_transcript.py`
- `mochi/runtime/models.py`
- `mochi/runtime/store.py`
- `mochi/runtime/service.py`
- `mochi/api/routes/agent_runs.py`
- `mochi/api/routes/chat.py`
- `mochi/api/routes/sessions.py`
- `mochi/agents/events.py`
- `mochi/agents/multi_agent/orchestrator.py`
- `mochi/agents/tool_exposure.py`

Frontend:

- `web/src/app/page.tsx`
- `web/src/lib/api.ts`
- `web/src/lib/execution-transcript.ts`
- `web/src/lib/chat-goal-continuation.ts`
- `web/src/components/chat/ExecutionTimeline.tsx`
- `web/src/components/chat/SubagentTimelineCard.tsx`
- `web/src/components/chat/SubagentDrawer.tsx`

Tests:

- `tests/test_execution_transcript.py`
- `tests/test_runtime_store.py`
- `tests/test_agent_run_operator_endpoints.py`
- `tests/test_api_chat_models.py`
- `tests/test_goal_api.py`
- `tests/test_tool_exposure.py`

Ignored local QA scripts:

- `web/scripts/test-execution-transcript-runtime-parity.mjs`
- `web/scripts/test-subagent-transcript-ui-layout.mjs`
- `web/scripts/test-subagent-transcript-browser-fixture.mjs`
- `web/scripts/test-session-subagent-reload-persistence.mjs`

Demo / fixture route:

- `web/src/app/test-fixtures/subagent-runtime/page.tsx`

## Verification Commands

Known passing commands:

```powershell
python -m pytest tests/test_api_chat_models.py -k "chat_subagent_stream or session_subagent_api or build_subagent_lifecycle or approval" -q
python -m pytest tests/test_api_chat_models.py -k "chat_subagent_stream or delegated_subagent_runtime_events or session_subagent_api or build_subagent_lifecycle or approval" -q
python -m pytest tests/test_goal_api.py -k "recommended_next_action or waiting_approval or single_agent or maps_orchestrator_awaiting_approval or resume or guidance or approval" -q
python -m pytest tests/test_execution_transcript.py tests/test_runtime_store.py -k "execution_transcript or subagent_transcript or runtime_blocked or approval" -q
python -m pytest tests/test_agent_run_operator_endpoints.py -k "agent_run_events_stream or events_endpoint or subagent_transcript_endpoint or subagent_messages or guidance_resume or approval" -q
python -m pytest tests/test_tool_exposure.py -k subagent -q
```

Frontend:

```powershell
cd web
npm.cmd run type-check
node --experimental-strip-types ./scripts/test-execution-transcript-runtime-parity.mjs
node ./scripts/test-subagent-transcript-ui-layout.mjs
npm.cmd run test:subagent-transcript-browser-fixture
npm.cmd run test:session-subagent-reload-persistence
```

General:

```powershell
git diff --check
```

Expected caveats:

- pytest may warn that `.pytest_cache` cannot be written. This has not indicated test failure.
- `npm.cmd run lint` still fails due to pre-existing unrelated issues in `web/src/app/goals/page.tsx` and `web/src/components/chat/WorkflowPanel.tsx`.

## Important Integration Decisions

- Subagent transcript is additive/display-oriented. Do not inject synthetic subagent cards into canonical assistant-visible prompt history.
- Target UI direction is Codex-like: a subagent should feel like a separate side conversation/thread, not only an inspector drawer with tabs.
- Main chat subagent cards should be collapsible/expandable. Collapsed cards should show compact status/progress; expanded cards should reveal recent events, output/approval summary, and an action to open the side conversation.
- `runtime_blocked` without a real `subagent_id` should not create fake subagent drawer cards.
- Restart resume should not pass full checkpoint continuation payload into orchestrator metadata, but may pass guidance-only payload so follow-up guidance remains visible.
- Chat-scoped subagent messages are guidance-only for now.
- Goal guidance is persisted into summary/recovery resume payload before resume.
- Approval metadata is displayable but does not bypass approval policy.
- For delegated normal-chat background tasks, live runtime event persistence is serialized with an async lock because the orchestrator schedules runtime callbacks concurrently. This prevents older events from overwriting newer transcript status.
- Delegated runtime `runtime_blocked` events without a real `subagent_id` still do not create fake drawer cards.
- Delegated runtime transcripts use `parent_type="delegated_task"` and `parent_id=<task_id>` when no chat turn id is available; if a parent turn id is later provided in metadata, they use `parent_type="chat_turn"`.

## Remaining Work

### 1. Full Normal-Chat Live Subagent Runtime

Current normal chat support now includes:

- explicit subagent requests expose `delegate_subagent_task`
- lifecycle/progress transcript is synthesized and persisted during the chat stream
- real delegated background runtime events are persisted into session-scoped subagent transcripts
- session subagent APIs and guidance endpoint exist
- session subagent cancel/resume endpoints exist
- matching delegated runtime events are projected back into the current chat SSE stream through a process-local best-effort live bus
- session-subagent messages are durable transcript events and are replayed as role-scoped guidance on delegated task resume/restart
- explicit `subagent_tool_call` / `subagent_tool_result` runtime events are normalized, persisted, and projected

Still missing for Codex-like parity:

- continue refining the Codex-like side-thread presentation as transcript event coverage gets richer
- true mid-call interruption/injection into an actively running subagent remains a future protocol-level enhancement; current two-way behavior is durable guidance at resume/restart boundaries

### Protocol-Level Mid-Call Interruption / Injection Design

Goal:

- Make an already-running session subagent behave more like Codex: the user can send a message to the side thread while the subagent is active, and the runtime can acknowledge, apply, defer, or cancel that message without waiting for a full task failure/restart.

Current boundary:

- Session-subagent messages are durable and replayed into delegated task resume/restart.
- Session-subagent messages now also have durable delivery metadata/status and are applied/deferred at orchestrator safe points.
- They are not injected into an already in-flight LLM generation or currently executing non-cancellable tool call.
- Tool calls may continue to completion unless explicitly cancelled by existing task cancellation.

Proposed runtime model:

- A per delegated task / subagent inbox is backed by durable transcript events.
- Each inbox message has:
  - `message_id`
  - `session_id`
  - `task_id`
  - `subagent_id`
  - `role_id`
  - `content`
  - `delivery_mode`: `inject_now`, `after_current_tool`, `next_checkpoint`, `resume_only`
  - `status`: `queued`, `accepted`, `applied`, `deferred`, `cancelled`, `rejected`
  - `created_at`, `updated_at`
  - optional `reason`, `applied_at`, `applied_event_seq`
- Store inbox state durably as structured subagent transcript events first; add a small runtime-store table later only if querying/status updates become awkward.
- Keep transcript events as the UI source of truth even if a table is added.

Proposed orchestrator contract:

- `MultiAgentRunRequest` now has `subagent_message_provider`.
- The provider can poll messages by `task_id`, `subagent_id`, `role_id`, `stage`, and `safe_point`, then mark message ids handled.
- Role execution checks the inbox at currently implemented safe points:
  - before role prompt/model invocation
  - after model invocation / before role completion, where unsafe in-flight delivery is deferred
- Future deeper safe points:
  - after prompt capture and before model invocation for all protocol branches
  - after tool-call request, before tool execution when the tool is cancellable/deferable
  - after tool result, before final answer synthesis
- For an in-flight LLM call that cannot be interrupted, mark message `deferred` with reason `generation_in_progress` and apply it at the next safe point.
- For an in-flight tool call:
  - if cancellable: cancel and apply guidance before retry/continue
  - if not cancellable: mark `deferred` with reason `tool_in_progress`
  - if approval is pending: attach message to approval context and apply before resume
- Current implemented tool-edge behavior:
  - `after_current_tool` messages are not applied at `before_model_invocation`.
  - If the invoke result contains a completed tool result, `after_current_tool` messages are accepted/applied after the tool result and merged into role guidance for later safe points.
  - If the invoke result indicates pending approval, `after_current_tool` messages are accepted/deferred with reason `approval_pending`; durable resume guidance still carries them forward after approval resolution.
  - If no tool result is reached, `after_current_tool` messages are deferred with reason `tool_not_reached`.
  - `cancel_current_tool` is currently metadata-level observable: API records the request, runtime safe points emit interruption/tool-cancel request/deferred events, and transcript/SSE UI can show that cancellation was requested but not actually preempted.
  - `subagent_tool_cancelled` must only be emitted after a real tool runtime/engine cancellation result exists.

Proposed API shape:

- Keep `POST /v1/sessions/{session_id}/subagents/{subagent_id}/messages` for compatibility.
- Optional request fields are implemented:
  - `delivery_mode?: "inject_now" | "after_current_tool" | "next_checkpoint" | "resume_only"`
  - `interrupt?: boolean`
  - `cancel_current_tool?: boolean`
- Response includes delivery state:
  - `message_id`
  - `delivery_status`
  - `delivery_reason`
  - refreshed `transcript`
- Add optional action endpoint only if needed:
  - `POST /v1/sessions/{session_id}/subagents/{subagent_id}/interrupt`
  - This can be a stricter alias for `messages` with `interrupt=true`.

Proposed SSE / transcript events:

- Implemented:
  - `subagent_message_queued`
  - `subagent_message_accepted`
  - `subagent_message_applied`
  - `subagent_message_deferred`
  - `subagent_message_cancelled` is accepted by normalizers/persistence for delivery-state parity.
  - `subagent_interrupted`
  - `subagent_tool_cancel_requested`
  - `subagent_tool_cancel_deferred`
- Future:
  - `subagent_tool_cancelled` after true cancellable tool execution hooks exist.

UI behavior:

- The side-thread composer sends messages immediately while the subagent is active.
- Active/running session subagents default to `inject_now` with `interrupt=true`; terminal subagents default to `resume_only`.
- If the runtime applies the message now, show it as applied in the thread.
- If it is deferred, show a quiet pending/deferred state near the user message, not an error.
- If the current tool/generation cannot be interrupted, keep the message durable and visible as queued for the next checkpoint/resume.
- Main chat card should show a compact status such as `guidance queued`, `guidance applied`, or `interruption deferred`.

Implementation tasks:

1. Backend contract and store:
   - Implemented: message delivery statuses and transcript event schema.
   - Implemented: durable initial status metadata for subagent messages.
   - Implemented: store hydration of delivery metadata to top-level event fields.
   - Implemented tests for queued/accepted delivery metadata.

2. Runtime inbox:
   - Implemented: session/task/subagent-scoped transcript-backed inbox provider in `RuntimeService`.
   - Implemented: durable transcript events remain the reconstruction source.
   - Implemented: applied/deferred state is reconstructed from appended delivery events.

3. Orchestrator safe points:
   - Implemented: checks before role prompt/model invocation and after model invocation / before role completion.
   - Implemented: applied messages are merged into role guidance/prompt.
   - Implemented: accepted/applied/deferred events through existing runtime event callback.

4. Tool semantics:
   - Define cancellable vs non-cancellable tool behavior.
   - Implemented: `after_current_tool` is filtered out before model invocation.
   - Implemented: after a completed tool result, `after_current_tool` messages are applied and merged into role guidance for later safe points.
   - Implemented: approval-pending tool results defer `after_current_tool` messages with reason `approval_pending`.
   - Partially implemented: unsafe in-flight delivery is deferred with reason such as `generation_in_progress` or `tool_not_reached`.
   - Implemented: `cancel_current_tool` emits/persists `subagent_tool_cancel_requested` and `subagent_tool_cancel_deferred` when no cancellable active tool boundary is available.
   - Implemented: `interrupt=true` emits/persists `subagent_interrupted` at orchestrator safe points.
   - Remaining: explicit cancellable/non-cancellable tool interruption hooks and true `subagent_tool_cancelled` emission.

5. Frontend:
   - Implemented: delivery status rendering for user messages in `SubagentDrawer`.
   - Implemented: quiet queued/applied/deferred state.
   - Implemented: existing resume/cancel controls preserved.

6. Verification:
   - Implemented: transcript normalization tests for message delivery events.
   - Implemented: API/store tests for delivery metadata.
   - Implemented: runtime test queued message is applied at safe point.
   - Implemented: protocol event tests for interrupt/tool-cancel request/deferred transcript persistence.
   - Implemented: browser/source fixture tests for delivery state UI.
   - Implemented: `after_current_tool` applies after completed tool result.
   - Implemented: `after_current_tool` defers on approval-pending tool result.
   - Remaining: cancellable tool-call cancellation test once deeper tool interruption hooks are added.

Implementation notes for future Codex-like refinements:

- Keep `SubagentDrawer` shaped as a side-thread conversation similar to Codex's subagent panel.
- Convert transcript events into readable thread messages such as "Worked for ...", "Read file ...", "Approval required", and "Completed with output ...".
- Preserve prompt and raw output as collapsible cards inside the thread rather than primary tabs.
- Place targeted guidance/follow-up input at the bottom of the side thread.
- The main chat card collapsed state should be dense and stable; expanded state should show a short event timeline and approval/output preview without nesting interactive controls.
- Mobile can still use a full-height sheet, but it should keep the same side-thread mental model instead of reverting to tabs.

### Normal-Chat Live Event Projection And Actions

Implemented:

- `RuntimeService` has a process-local delegated subagent runtime event bus.
- `_run_delegated_multi_agent_task` publishes normalized runtime events after durable transcript persistence.
- `/chat/stream` subscribes while open, tracks delegated task ids from `delegate_subagent_task`, emits matching live runtime transcript events, and uses bounded idle/max drain timers so streams do not hang.
- `runtime_blocked` without a real `subagent_id` is still filtered out.
- `POST /v1/sessions/{session_id}/subagents/{subagent_id}/cancel`
- `POST /v1/sessions/{session_id}/subagents/{subagent_id}/resume`
- Resume accepts optional `guidance` or message-shaped `content` and appends it before calling `resume_task`.
- Delegated tasks with no pending approval can be restarted from resumable terminal/stalled states, and the saved session-subagent guidance is passed into `MultiAgentRunRequest.role_guidance_messages`.
- The runtime bridge accepts explicit `subagent_thinking`, `subagent_tool_call`, and `subagent_tool_result` events.
- The runtime bridge and `/chat/stream` live projection accept delivery/control events: `subagent_message_*`, `subagent_interrupted`, `subagent_tool_cancel_requested`, and `subagent_tool_cancel_deferred`.
- `MultiAgentOrchestrator` emits `subagent_tool_call` / `subagent_tool_result` events from subagent invoke tool events with redacted argument previews.
- Frontend `SubagentDrawer` exposes Cancel/Resume controls for session-scoped subagents and refreshes transcript/card state after actions.

Important limitation:

- The live bus is intentionally best-effort and process-local. The durable transcript database remains the source of truth, and reload/refetch paths must continue to work without receiving every live bus event.
- Guidance is applied at resume/restart boundaries. Injecting a new user message into an already-running LLM/tool call is not implemented.

### 2. Documentation And Architecture Notes

If the next agent continues docs:

- Update or force-add ignored architecture docs if desired.
- Avoid bulk rewriting the original `2026-06-29` plan due to encoding issues.
- This handoff file can be used as current truth.

### 3. Final Review / Commit Packaging

Before packaging the branch:

- Re-run focused backend and frontend checks.
- Decide whether ignored docs/scripts should be force-added.
- Ignore unrelated untracked `Untitled-2026-06-19-1605.excalidraw`.

## Completed In The Hardening Pass

### Browser Fixture For Timeline And Drawer

Implemented:

- `web/src/app/test-fixtures/subagent-runtime/page.tsx`
- `web/scripts/test-subagent-transcript-browser-fixture.mjs`
- `web/package.json` script `test:subagent-transcript-browser-fixture`

Fixture renders:

- long system/user prompts
- timeline with many events
- approval-required tool result
- runtime blocker
- output preview
- guidance input
- approval-surface linking

### Direct Approval Linking In SubagentDrawer

Implemented:

- `SubagentDrawer` accepts `onOpenApprovals?: (approvalIds: string[]) => void`
- drawer extracts approval ids from `approval_id`, `approval_ids`, and `pending_approvals`
- `web/src/app/page.tsx` opens `TaskPanel` scoped to those ids
- `TaskPanel` accepts `focusedApprovalIds?: string[] | null`

### Reload Persistence Hardening

Implemented:

- `mergeSubagentSummaries` preserves richer API summaries and max event count.
- Session switching clears stale session subagents.
- Session restore clears stale chat-turn timeline state even when no restored events exist.
- `web/scripts/test-session-subagent-reload-persistence.mjs`
- `web/package.json` script `test:session-subagent-reload-persistence`

### Normal-Chat Delegated Runtime Transcript Bridge

Implemented:

- `RuntimeService._run_delegated_multi_agent_task` passes `runtime_event_callback` when a parent session is known.
- Runtime events are normalized with `normalize_subagent_event` and persisted to `subagent_transcripts` / `subagent_transcript_events`.
- Persistence is serialized to match orchestrator callback concurrency.
- Test coverage includes concurrent callback scheduling with a delayed first upsert.

## Git Notes

- Ignore unrelated untracked `Untitled-2026-06-19-1605.excalidraw`.
- `.claude/`, `docs/`, and `web/scripts/` are ignored by `.gitignore`.
- To include handoff/memory/scripts in a commit, use:

```powershell
git add -f .claude/skills/agent-memory/memories/project-status/chat-goal-subagent-runtime-parity-2026-06-30.md
git add -f docs/superpowers/plans/2026-06-30-chat-and-goal-subagent-runtime-parity-handoff.md
git add -f web/scripts/test-subagent-transcript-ui-layout.mjs
git add -f web/scripts/test-execution-transcript-runtime-parity.mjs
```

Only add those if the user wants ignored docs/scripts committed.
