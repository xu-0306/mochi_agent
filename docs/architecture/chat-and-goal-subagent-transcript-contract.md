# Chat And Goal Subagent Transcript Contract

Date: 2026-06-29
Status: additive implementation contract

## Product Contract

Chat is the primary surface for both ordinary conversation and long-running Goal steering. A subagent is no longer only a Goal/operator concept; it is a first-class child execution that can belong to:

- `chat_turn`
- `agent_run`
- `goal`

Goal remains the durable objective supervisor. AgentRun remains the concrete execution unit. Subagent transcripts are the shared visibility layer underneath both.

## Persistence

`RuntimeStore` owns two additive SQLite tables:

- `subagent_transcripts`
  - stable `id` as `subagent_id`
  - parent scope fields: `parent_type`, `parent_id`, `session_id`, `goal_id`, `agent_run_id`, `parent_turn_id`
  - visible identity: `role_id`, `title`, `model_id`, `status`
  - prompt and output fields: `system_prompt`, `user_prompt`, `prompt_preview`, `summary`
  - `metadata_json`, `created_at`, `updated_at`
- `subagent_transcript_events`
  - monotonic `seq` per subagent
  - UI-safe `event_json`
  - `created_at`

The tables are additive. Legacy `agent_run_events` remain the compatibility event log for existing operator pages.

## Event Types

The normalized transcript event contract is UI-safe and intentionally does not expose hidden reasoning or raw tool output:

- `subagent_started`
- `subagent_prompt`
- `subagent_progress`
- `subagent_thinking`
- `subagent_tool_call`
- `subagent_tool_result`
- `subagent_completed`
- `runtime_blocked`

Legacy orchestrator events are normalized as:

- `role_started` -> `subagent_started`
- `role_progress` -> `subagent_progress`
- `role_completed` -> `subagent_completed`
- `role_error` -> `subagent_completed` with failed or blocked status

Prompt events may include `system_prompt`, `user_prompt`, `prompt_preview`, `stage`, and `execution_profile`. Tool events may include `tool_call_id`, `tool_name`, `arguments_preview`, approval ids, and a result summary. Sensitive fields such as tokens, cookies, environment values, stdout, stderr, and secrets must be filtered before storing display metadata.

## APIs

AgentRun transcript APIs:

- `GET /v1/agent-runs/{run_id}/events`
- `GET /v1/agent-runs/{run_id}/events/stream`
- `GET /v1/agent-runs/{run_id}/subagents`
- `GET /v1/agent-runs/{run_id}/subagents/{subagent_id}`
- `POST /v1/agent-runs/{run_id}/subagents/{role_id_or_subagent_id}/messages`

Session transcript APIs:

- `GET /v1/sessions/{session_id}/subagents`
- `GET /v1/sessions/{session_id}/subagents/{subagent_id}`
- `POST /v1/sessions/{session_id}/subagents/{subagent_id}/messages`

The session message endpoint is guidance-only in the first implementation. It appends a transcript event and marks metadata so a future resume/follow-up path can consume it.

## Runtime Projection

AgentRun execution:

- `MultiAgentOrchestrator` emits `subagent_prompt` before role invocation.
- Role lifecycle events carry stable `subagent_id`, `role_id`, `title`, and `model_id`.
- `autonomous_single_agent` appears as a visible primary subagent.
- `RuntimeService` persists normalized transcript events while preserving legacy `agent_run_events`.

Normal chat:

- Existing `delegate_subagent_task` remains the tool entrypoint.
- Chat stream synthesizes `subagent_started`, `subagent_prompt`, and either `subagent_progress` or `subagent_completed` from the delegated tool request/result status.
- Chat-scoped transcripts use `parent_type="chat_turn"`, `session_id`, and `parent_turn_id`.

## UI Projection

`web/src/app/page.tsx` keeps separate session and AgentRun subagent state, then merges by `subagentId` for display. The same `ExecutionTimeline`, `SubagentTimelineCard`, and `SubagentDrawer` render both chat-scoped and Goal/AgentRun-scoped subagents.

Guidance routing:

- `agentRunId` present: use AgentRun subagent guidance endpoint.
- only `sessionId` present: use session subagent guidance endpoint.

Projected transcript cards and timelines are display state, not canonical prompt history. Do not write synthetic subagent UI cards into assistant-visible replay history unless a future explicit marker contract is introduced.

## Blockers And Approvals

Goal health recommendations now include blocker metadata where available:

- `blocker_type`
- `approval_ids`
- `tool_names`
- `approval_count`
- `run_id`
- `recommended_action`

`runtime_blocked` transcript events should carry the same metadata so chat can explain what is waiting instead of showing a generic operator-required state.

Approval execution remains governed by existing approval and runtime permission policies. Subagents do not bypass permission checks; side-effectful work must still flow through the approved runtime path.

## Migration Note

Old AgentRun rows continue to render through derived transcript summaries from `agent_run_events` when no first-class transcript exists. New runs get explicit transcript rows. Frontend consumers should tolerate both shapes during migration.
