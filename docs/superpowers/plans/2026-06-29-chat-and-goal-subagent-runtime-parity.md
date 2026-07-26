# Chat And Goal Subagent Runtime Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `@superpowers:subagent-driven-development` (recommended) or `@superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Codex-style long-running execution and subagent experience where both normal chat and Goal execution can spawn, inspect, stream, and steer subagents from the primary conversation surface.

**Architecture:** Introduce a shared execution transcript model beneath both chat turns and goal-linked agent runs. Treat Goal as a durable supervisor layer, not the only place subagents can exist. Project runtime events into the main chat timeline and expose each subagent as an inspectable, addressable child conversation with its prompt, model, status, tool activity, outputs, and operator/user guidance.

**Tech Stack:** Python async runtime, FastAPI, SQLite `RuntimeStore`, `AgentEngine`, `MultiAgentOrchestrator`, Next.js/React WebGUI, existing SSE chat stream, existing agent-run event persistence.

---

## Context

The current implementation starts Goal work from chat and shows `GoalCard` / `GoalHeaderChip`, but active Goal follow-up often only appends guidance to an agent run and persists a status message. The user cannot see a Codex-style live transcript of each step, and subagent details are mostly hidden in operator pages or coarse `role_progress` events.

The desired behavior is broader than Goal:

- In normal chat, the main agent should be able to delegate to subagents.
- In Goal execution, the same subagent model should be used for single-agent and workflow goals.
- Users should be able to click a running subagent, inspect the main agent prompt and subagent runtime, and send targeted guidance.
- Main chat should show live execution progress, tool calls, approvals, and blockers instead of requiring users to interpret an operator status card.

Relevant current files:

- `web/src/app/page.tsx`
  - Current chat integration hotspot.
  - Goal follow-up handling starts around `activeGoalFollowUpRequested`.
- `web/src/lib/chat-goal-continuation.ts`
  - Maps Goal health to continuation actions.
- `web/src/components/chat/GoalCard.tsx`
  - Static Goal proposal/start/status card.
- `web/src/components/chat/GoalHeaderChip.tsx`
  - Active Goal chip and lightweight drawer.
- `web/src/components/chat/TaskPanel.tsx`
  - Existing task/workflow/operator panel with delegated subagent concepts.
- `web/src/components/chat/SubagentTaskCard.tsx`
  - Existing lightweight subagent card.
- `web/src/components/chat/ReasoningPanel.tsx`
  - Existing streamed reasoning/tool trace surface.
- `web/src/lib/api.ts`
  - Current client API normalizers and runtime methods.
- `web/src/lib/chat.ts`
  - Chat message and card view types.
- `web/src/lib/chat-projections.ts`
  - Display projection logic for persisted chat events.
- `mochi/api/routes/chat.py`
  - Chat and chat-stream API.
- `mochi/api/routes/agent_runs.py`
  - Agent run API; already exposes messages and subagent targeted messages.
- `mochi/api/routes/goals.py`
  - Goal API and health endpoint.
- `mochi/runtime/models.py`
  - Pydantic models for Goal, AgentRun, messages, subagent messages.
- `mochi/runtime/store.py`
  - SQLite persistence for task events, agent run events, goals, approvals.
- `mochi/runtime/service.py`
  - Durable runtime service, Goal supervision, AgentRun lifecycle, event persistence.
- `mochi/agents/engine.py`
  - Main chat execution path.
- `mochi/agents/events.py`
  - Chat event classes.
- `mochi/agents/multi_agent/orchestrator.py`
  - Workflow/single-agent Goal execution and role progress events.
- `mochi/agents/multi_agent/roles.py`
  - Role definitions including autonomous single agent.
- `tests/test_api_chat_models.py`
  - Chat streaming and event persistence tests.
- `tests/test_agent_run_operator_endpoints.py`
  - AgentRun operator/subagent message endpoint tests.
- `tests/test_goal_api.py`
  - Goal health, approval, resume, checkpoint tests.

Reference patterns:

- `reference/hermes-agent/ui-tui/src/gatewayTypes.ts`
  - Has `subagent.start`, `subagent.thinking`, `subagent.tool`, `subagent.progress`, `subagent.complete`.
- `reference/hermes-agent/ui-tui/src/app/createGatewayEventHandler.ts`
  - Upserts subagent progress from gateway events.
- `reference/hermes-agent/ui-tui/src/components/thinking.tsx`
  - Renders expandable subagent accordions inside a thinking/tool timeline.
- `reference/cc-haha/src/utils/agentContext.ts`
  - Uses per-agent context metadata to correlate subagent activity.
- `reference/cc-haha/src/tools/AgentTool/runAgent.ts`
  - Persists and attributes subagent execution context.

## Product Contract

- Chat is the primary surface for normal conversation, delegated subagents, and Goal steering.
- Goal is a durable objective supervisor, not the only subagent entrypoint.
- A subagent is a first-class execution child with:
  - stable `subagent_id`
  - optional `parent_session_id`
  - optional `goal_id`
  - optional `agent_run_id`
  - optional `parent_turn_id`
  - `role_id` / `name`
  - model id
  - system prompt
  - user prompt
  - live events
  - final output
  - targeted user/operator messages
- Users can inspect subagent prompts and execution traces.
- Users can send guidance to a running or resumable subagent.
- Goal blocked states must state the exact blocker in chat:
  - approval id and tool/command/file when approval is required
  - runtime budget or retry policy when exhausted
  - checkpoint/handoff reason when applicable
  - collector shard status when applicable
  - operator emergency-stop controls when applicable
- Single-agent Goals should use the same transcript model, with a synthetic primary subagent/role such as `primary`.
- Existing operator pages remain advanced inspection surfaces, not the primary user path.

## Non-Goals

- Do not remove `/goals` or `/agent-runs`.
- Do not redesign the entire multi-agent protocol system.
- Do not expose hidden chain-of-thought beyond what the backend already emits as safe reasoning/status/tool events.
- Do not make subagents bypass approval, permission, or autonomy policy.
- Do not implement full distributed process isolation in the first pass.
- Do not make browser UI depend on reference projects directly.

## File Structure

### Backend Contracts

- Create `mochi/runtime/execution_transcript.py`
  - Shared helpers for normalizing runtime/subagent transcript events.
  - Converts AgentRun and chat events into UI-safe transcript payloads.
- Modify `mochi/runtime/models.py`
  - Add `ExecutionTranscriptEvent`, `SubagentTranscriptSummary`, `SubagentTranscriptDetail`.
  - Add request/response models for subagent transcript reads and messages.
- Modify `mochi/runtime/store.py`
  - Add persistence for `subagent_transcripts` and `subagent_transcript_events`, or add scoped event rows if keeping a single events table is simpler.
  - Keep migrations additive and backward compatible.
- Modify `mochi/runtime/service.py`
  - Persist subagent transcript lifecycle events.
  - Add APIs to list/get subagent transcripts by chat session, goal, or agent run.
  - Project Goal blockers into chat-friendly diagnostic payloads.
- Modify `mochi/api/routes/agent_runs.py`
  - Add event and subagent transcript read endpoints.
  - Upgrade existing targeted subagent message endpoint to attach transcript metadata.
- Modify `mochi/api/routes/chat.py`
  - Stream shared transcript events during normal chat delegation.
  - Add optional endpoint to fetch chat-scoped subagent transcripts for refresh/reload.

### Agent Runtime

- Modify `mochi/agents/engine.py`
  - Add a main-chat subagent/delegation tool or runtime hook.
  - Emit subagent transcript events into the same chat SSE stream.
- Modify `mochi/agents/events.py`
  - Add UI-safe subagent event types for chat stream serialization.
- Modify `mochi/agents/multi_agent/orchestrator.py`
  - Emit role/subagent prompt-created events before invocation.
  - Emit subagent thinking/tool/progress/completion events using a normalized event contract.
  - Attach `subagent_id` to role progress and role output.
- Modify `mochi/agents/multi_agent/roles.py`
  - Ensure autonomous single-agent has a stable visible role id and title.

### Frontend API And State

- Modify `web/src/lib/api.ts`
  - Add normalizers and client methods for transcript events and subagent details.
  - Add AgentRun event fetch/stream client helpers.
- Modify `web/src/lib/chat.ts`
  - Add message metadata for execution transcript and subagent references.
- Modify `web/src/lib/chat-projections.ts`
  - Persist and restore projected subagent transcript cards after reload.
- Modify `web/src/lib/chat-goal-continuation.ts`
  - Return richer blocker diagnostics, not only a summary/action.

### Frontend UI

- Create `web/src/components/chat/ExecutionTimeline.tsx`
  - Shared compact timeline for live thinking, tool, approval, subagent, and final events.
- Create `web/src/components/chat/SubagentTimelineCard.tsx`
  - Inline clickable subagent summary card.
- Create `web/src/components/chat/SubagentDrawer.tsx`
  - Inspect prompt, model, status, tool calls, outputs, and targeted conversation.
- Modify `web/src/components/chat/ReasoningPanel.tsx`
  - Render subagent events in the existing reasoning/tool trace where appropriate.
- Modify `web/src/components/chat/ChatMessage.tsx`
  - Render transcript/subagent cards from message metadata.
- Modify `web/src/app/page.tsx`
  - Subscribe active Goal/AgentRun events into main chat.
  - Render blocked diagnostics inline.
  - Provide subagent drawer open/send interactions for both normal chat and Goal.
- Modify `web/src/components/chat/GoalHeaderChip.tsx`
  - Link to active transcript/subagents, not only operator console.

## Event Contract

Use these normalized event types for both chat and Goal/AgentRun projection:

```json
{
  "type": "subagent_started",
  "subagent_id": "subagent-uuid",
  "parent_type": "chat_turn|goal|agent_run",
  "parent_id": "session-or-goal-or-run-id",
  "role_id": "researcher",
  "title": "Researcher",
  "model_id": "model-id",
  "prompt_preview": "Short preview",
  "created_at": "2026-06-29T00:00:00Z"
}
```

```json
{
  "type": "subagent_prompt",
  "subagent_id": "subagent-uuid",
  "system_prompt": "UI-safe system prompt",
  "user_prompt": "UI-safe user prompt",
  "metadata": {
    "stage": "role::researcher",
    "execution_profile": "subagent_readonly"
  }
}
```

```json
{
  "type": "subagent_thinking",
  "subagent_id": "subagent-uuid",
  "content": "Preparing search strategy."
}
```

```json
{
  "type": "subagent_tool_call",
  "subagent_id": "subagent-uuid",
  "tool_call_id": "call-id",
  "tool_name": "web_search",
  "arguments_preview": "query=..."
}
```

```json
{
  "type": "subagent_tool_result",
  "subagent_id": "subagent-uuid",
  "tool_call_id": "call-id",
  "status": "succeeded|failed|approval_required",
  "summary": "Fetched 5 results."
}
```

```json
{
  "type": "subagent_completed",
  "subagent_id": "subagent-uuid",
  "status": "completed|failed|blocked|cancelled",
  "summary": "Short result summary",
  "output_preview": "First useful lines"
}
```

```json
{
  "type": "runtime_blocked",
  "parent_type": "goal|agent_run|chat_turn",
  "parent_id": "id",
  "blocker_type": "approval|runtime_budget|checkpoint|collector|operator_control|unknown",
  "summary": "Human-readable blocker",
  "approval_ids": ["approval-id"],
  "tool_names": ["exec_command"],
  "recommended_action": "resolve_approval"
}
```

## Task 1: Add Backend Transcript Models And Tests

**Files:**
- Create: `mochi/runtime/execution_transcript.py`
- Modify: `mochi/runtime/models.py`
- Test: `tests/test_execution_transcript.py`

- [x] **Step 1: Write failing tests for transcript normalization**

Add tests for:

```python
from mochi.runtime.execution_transcript import normalize_subagent_event


def test_normalize_subagent_started_event_requires_identity() -> None:
    event = normalize_subagent_event(
        {
            "type": "role_started",
            "subagent_id": "sub-1",
            "role_id": "researcher",
            "model_id": "qwen",
            "current_action": "Researcher is preparing a response.",
        },
        parent_type="agent_run",
        parent_id="run-1",
    )

    assert event["type"] == "subagent_started"
    assert event["subagent_id"] == "sub-1"
    assert event["role_id"] == "researcher"
    assert event["parent_type"] == "agent_run"
    assert event["parent_id"] == "run-1"
```

```python
def test_normalize_runtime_blocker_event_preserves_approval_metadata() -> None:
    event = normalize_subagent_event(
        {
            "type": "runtime_blocked",
            "blocker_type": "approval",
            "summary": "Goal is waiting on operator approval.",
            "approval_ids": ["approval-1"],
            "tool_names": ["exec_command"],
            "recommended_action": "resolve_approval",
        },
        parent_type="goal",
        parent_id="goal-1",
    )

    assert event["type"] == "runtime_blocked"
    assert event["blocker_type"] == "approval"
    assert event["approval_ids"] == ["approval-1"]
    assert event["tool_names"] == ["exec_command"]
```

- [x] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_execution_transcript.py -q
```

Expected: fails because `mochi.runtime.execution_transcript` does not exist.

- [x] **Step 3: Implement transcript model helpers**

Implement:

```python
def normalize_subagent_event(
    raw: Mapping[str, Any],
    *,
    parent_type: str,
    parent_id: str,
) -> dict[str, Any]:
    ...
```

Rules:

- Accept current `role_started`, `role_progress`, `role_completed`, `role_error`.
- Accept already-normalized `subagent_*` and `runtime_blocked`.
- Preserve only UI-safe prompt/tool/result summaries.
- Generate deterministic fallback `subagent_id` from `parent_id`, `role_id`, and stage when no id exists.
- Do not expose raw secrets, environment values, or full command output beyond existing safe summaries.

- [x] **Step 4: Add Pydantic models**

Add to `mochi/runtime/models.py`:

```python
class ExecutionTranscriptEvent(BaseModel):
    type: str
    parent_type: str | None = None
    parent_id: str | None = None
    subagent_id: str | None = None
    role_id: str | None = None
    title: str | None = None
    model_id: str | None = None
    content: str | None = None
    summary: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None


class SubagentTranscriptSummary(BaseModel):
    subagent_id: str
    parent_type: str
    parent_id: str
    role_id: str | None = None
    title: str | None = None
    model_id: str | None = None
    status: str = "running"
    prompt_preview: str | None = None
    summary: str | None = None
    event_count: int = 0
    created_at: str | None = None
    updated_at: str | None = None


class SubagentTranscriptDetail(SubagentTranscriptSummary):
    system_prompt: str | None = None
    user_prompt: str | None = None
    events: list[ExecutionTranscriptEvent] = Field(default_factory=list)
```

- [x] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_execution_transcript.py -q
```

Expected: pass.

## Task 2: Persist Subagent Transcripts In RuntimeStore

**Files:**
- Modify: `mochi/runtime/store.py`
- Test: `tests/test_runtime_store.py`

- [x] **Step 1: Write failing persistence tests**

Add tests that:

- initialize `RuntimeStore`
- upsert a subagent transcript
- append two transcript events
- list transcripts by `agent_run_id`
- fetch detail by `subagent_id`

Expected assertions:

```python
assert detail["subagent_id"] == "sub-1"
assert detail["system_prompt"] == "System"
assert detail["user_prompt"] == "User"
assert [event["type"] for event in detail["events"]] == [
    "subagent_started",
    "subagent_completed",
]
```

- [x] **Step 2: Run test and verify failure**

Run:

```powershell
python -m pytest tests/test_runtime_store.py -k subagent_transcript -q
```

Expected: fails because store methods/tables do not exist.

- [x] **Step 3: Add SQLite tables**

Add to `_init_db`:

```sql
CREATE TABLE IF NOT EXISTS subagent_transcripts (
    id TEXT PRIMARY KEY,
    parent_type TEXT NOT NULL,
    parent_id TEXT NOT NULL,
    session_id TEXT,
    goal_id TEXT,
    agent_run_id TEXT,
    parent_turn_id TEXT,
    role_id TEXT,
    title TEXT,
    model_id TEXT,
    status TEXT NOT NULL,
    system_prompt TEXT,
    user_prompt TEXT,
    prompt_preview TEXT,
    summary TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
```

```sql
CREATE TABLE IF NOT EXISTS subagent_transcript_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subagent_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    event_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(subagent_id) REFERENCES subagent_transcripts(id) ON DELETE CASCADE
)
```

- [x] **Step 4: Implement store methods**

Add methods:

- `upsert_subagent_transcript(...)`
- `append_subagent_transcript_event(subagent_id, event)`
- `list_subagent_transcripts(parent_type=None, parent_id=None, session_id=None, goal_id=None, agent_run_id=None)`
- `get_subagent_transcript(subagent_id)`

- [x] **Step 5: Run focused tests**

Run:

```powershell
python -m pytest tests/test_runtime_store.py -k subagent_transcript -q
```

Expected: pass.

## Task 3: Persist Orchestrator Role/Subagent Prompts And Runtime Events

**Files:**
- Modify: `mochi/agents/multi_agent/orchestrator.py`
- Modify: `mochi/runtime/service.py`
- Test: `tests/test_goal_api.py`
- Test: `tests/test_agent_run_operator_endpoints.py`

- [x] **Step 1: Write failing test for role prompt persistence**

In `tests/test_agent_run_operator_endpoints.py`, add an agent-run test that starts a fake model-backed run and asserts a subagent transcript exists with:

- `role_id`
- `model_id`
- `system_prompt`
- `user_prompt`
- `subagent_started`
- `subagent_completed`

- [x] **Step 2: Run test and verify failure**

Run:

```powershell
python -m pytest tests/test_agent_run_operator_endpoints.py -k subagent_transcript -q
```

Expected: fails because no transcript endpoint/persistence exists yet.

- [x] **Step 3: Add runtime event callback contract**

In `MultiAgentOrchestrator._generate_role_candidate`, before `_invoke_configured_text`, emit:

```python
self._emit_runtime_event(
    "subagent_prompt",
    {
        "subagent_id": subagent_id,
        "role_id": role_id,
        "title": role_title,
        "model_id": model_id,
        "system_prompt": role_instruction,
        "user_prompt": prompt,
        "stage": role_stage,
        "execution_profile": "subagent_readonly",
    },
)
```

Also include `subagent_id` in existing role progress payloads.

- [x] **Step 4: Persist callback events in RuntimeService**

In the existing `_persist_live_runtime_event` callback inside agent-run execution:

- normalize events with `normalize_subagent_event`
- upsert transcript on `subagent_prompt` / `role_started`
- append transcript events
- still append existing agent-run events for backward compatibility

- [x] **Step 5: Ensure autonomous single-agent maps to a visible primary subagent**

In `_run_model_backed_autonomous_single_agent`, use a stable visible role:

- `role_id`: existing protocol role id or `primary`
- `title`: `Primary agent`
- `subagent_id`: stable for run attempt and role

- [x] **Step 6: Run tests**

Run:

```powershell
python -m pytest tests/test_agent_run_operator_endpoints.py -k "subagent_transcript or subagent_messages" -q
python -m pytest tests/test_goal_api.py -k "single_agent or maps_orchestrator_awaiting_approval" -q
```

Expected: pass.

## Task 4: Add AgentRun Event And Subagent Transcript API

**Files:**
- Modify: `mochi/api/routes/agent_runs.py`
- Modify: `mochi/runtime/service.py`
- Modify: `mochi/runtime/models.py`
- Test: `tests/test_agent_run_operator_endpoints.py`

- [x] **Step 1: Write failing endpoint tests**

Add tests for:

- `GET /v1/agent-runs/{run_id}/events`
- `GET /v1/agent-runs/{run_id}/subagents`
- `GET /v1/agent-runs/{run_id}/subagents/{subagent_id}`
- `POST /v1/agent-runs/{run_id}/subagents/{role_id}/messages` stores `subagent_id` when possible.

- [x] **Step 2: Run test and verify failure**

Run:

```powershell
python -m pytest tests/test_agent_run_operator_endpoints.py -k "events_endpoint or subagent_transcript_endpoint" -q
```

Expected: 404 or missing fields.

- [x] **Step 3: Implement service methods**

Add:

- `list_agent_run_events(run_id, after_seq=None, limit=None)`
- `list_agent_run_subagents(run_id)`
- `get_agent_run_subagent(run_id, subagent_id)`

Do not remove `events` from `GET /v1/agent-runs/{run_id}` yet.

- [x] **Step 4: Implement API routes**

Add:

```python
@router.get("/{run_id}/events")
async def list_agent_run_events(...)
```

```python
@router.get("/{run_id}/subagents")
async def list_agent_run_subagents(...)
```

```python
@router.get("/{run_id}/subagents/{subagent_id}")
async def get_agent_run_subagent(...)
```

- [x] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_agent_run_operator_endpoints.py -k "events_endpoint or subagent_transcript_endpoint or subagent_messages" -q
```

Expected: pass.

## Task 5: Improve Goal Blocker Diagnostics In Chat Continuation

**Files:**
- Modify: `mochi/runtime/service.py`
- Modify: `web/src/lib/chat-goal-continuation.ts`
- Modify: `web/src/lib/goal-proposal-copy.ts`
- Modify: `web/src/app/page.tsx`
- Test: `tests/test_goal_api.py`

- [x] **Step 1: Write backend tests for blocker payload**

In `tests/test_goal_api.py`, extend approval-wait tests to assert `recommended_next_action` includes:

- `approval_ids`
- `tool_names`
- `run_id`
- `blocker_type: "approval"`

Add tests for runtime budget and operator controls:

- `blocker_type: "runtime_budget"`
- `blocker_type: "operator_control"`

- [x] **Step 2: Run backend tests and verify failure**

Run:

```powershell
python -m pytest tests/test_goal_api.py -k "recommended_next_action or waiting_approval" -q
```

Expected: fails on missing fields.

- [x] **Step 3: Enrich `_goal_recommended_next_action`**

Add:

```python
"blocker_type": "approval"
"tool_names": [...]
"approval_count": len(approval_ids)
```

For other blockers, add explicit `blocker_type`.

- [x] **Step 4: Extend frontend continuation decision**

In `web/src/lib/chat-goal-continuation.ts`, extend `GoalContinuationDecision`:

```ts
blockerType:
  | 'approval'
  | 'runtime_budget'
  | 'checkpoint'
  | 'collector'
  | 'operator_control'
  | 'unknown'
toolNames: string[]
recommendedAction: string | null
```

- [x] **Step 5: Render clear chat copy**

In `web/src/app/page.tsx`, replace generic blocked copy with:

- approval required:
  - approval count
  - tool names
  - first approval id
  - suggested action
- runtime budget:
  - budget status
  - mention resume is unsafe until adjusted
- operator control:
  - blocked tools/domains/network state

- [x] **Step 6: Run focused tests**

Run:

```powershell
python -m pytest tests/test_goal_api.py -k "recommended_next_action or waiting_approval" -q
```

Expected: pass.

Manual WebGUI check:

- Start a Goal that requires approval.
- Confirm main chat says what approval is pending and why.
- Confirm drawer still shows full approval controls.

## Task 6: Project AgentRun Events Into Main Chat For Active Goals

**Files:**
- Modify: `web/src/lib/api.ts`
- Create: `web/src/lib/execution-transcript.ts`
- Create: `web/src/components/chat/ExecutionTimeline.tsx`
- Modify: `web/src/app/page.tsx`
- Modify: `web/src/components/chat/ChatMessage.tsx`
- Test: frontend typecheck

- [x] **Step 1: Add API normalizers**

Add types:

```ts
export interface ExecutionTranscriptEvent {
  type: string
  seq?: number
  parentType?: string | null
  parentId?: string | null
  subagentId?: string | null
  roleId?: string | null
  title?: string | null
  modelId?: string | null
  content?: string | null
  summary?: string | null
  metadata: Record<string, unknown>
  createdAt?: string | null
}
```

Add:

- `fetchAgentRunEvents(runId, options?)`
- `fetchAgentRunSubagents(runId)`
- `fetchAgentRunSubagent(runId, subagentId)`

- [x] **Step 2: Add projection helper**

Create `web/src/lib/execution-transcript.ts` with:

- `groupTranscriptEventsBySubagent(events)`
- `summarizeRuntimeBlocker(event)`
- `mapAgentRunEventToTranscriptEvent(event)`

- [x] **Step 3: Add timeline component**

Create `ExecutionTimeline.tsx`:

- compact rows for run events
- clickable subagent rows
- approval/blocker row
- no card-in-card layout
- stable dimensions for status badges

- [x] **Step 4: Wire active Goal polling first**

In `web/src/app/page.tsx`:

- when `activeGoalId` and bound run id exist, poll `fetchAgentRunEvents` every 2 seconds while active
- store last seen seq
- append/update a local runtime timeline state
- stop polling on terminal Goal state

Do polling first before SSE to reduce implementation risk.

- [x] **Step 5: Render timeline in main chat**

Show a live execution timeline below the active Goal card or as part of the reasoning panel.

The user should see:

- run started
- primary/subagent started
- tool/progress events
- blocked state
- completion

- [x] **Step 6: Run frontend checks**

Run:

```powershell
cd web
npm run lint
npm run typecheck
```

Expected: pass.

If scripts are unavailable, run the closest existing frontend verification command from `web/package.json`.

## Task 7: Add Subagent Drawer For Goal And Chat Transcripts

**Files:**
- Create: `web/src/components/chat/SubagentTimelineCard.tsx`
- Create: `web/src/components/chat/SubagentDrawer.tsx`
- Modify: `web/src/components/chat/ExecutionTimeline.tsx`
- Modify: `web/src/app/page.tsx`
- Modify: `web/src/lib/api.ts`

- [x] **Step 1: Build summary card**

`SubagentTimelineCard` props:

```ts
interface SubagentTimelineCardProps {
  subagent: SubagentTranscriptSummary
  active?: boolean
  onOpen: (subagentId: string) => void
}
```

Display:

- role/title
- model
- status
- last action
- event count
- duration if available

- [x] **Step 2: Build drawer**

`SubagentDrawer` sections:

- header: title, status, model
- tabs or segmented control:
  - Prompt
  - Timeline
  - Output
  - Guidance
- Prompt tab shows system/user prompt with copy buttons.
- Timeline tab shows subagent-scoped events.
- Guidance tab includes text input and send button.

- [x] **Step 3: Wire drawer in page**

In `web/src/app/page.tsx`:

- maintain selected subagent id
- fetch detail on open
- send targeted message using existing agent-run subagent message API when `agent_run_id` exists
- for chat-only subagents, use new chat subagent message endpoint from Task 9

- [x] **Step 4: Verify responsive layout**

Use existing design constraints:

- no nested cards
- no text overflow
- icon buttons for close/copy/send where appropriate
- drawer works on mobile and desktop

- [x] **Step 5: Run frontend checks**

Run:

```powershell
cd web
npm run lint
npm run typecheck
```

Expected: pass.

## Task 8: Make AgentRun/Subagent Guidance Actually Influence Resume

**Files:**
- Modify: `mochi/runtime/service.py`
- Modify: `mochi/agents/multi_agent/orchestrator.py`
- Test: `tests/test_agent_run_operator_endpoints.py`
- Test: `tests/test_goal_api.py`

- [x] **Step 1: Write failing tests for guidance resume payload**

Test that:

- `POST /v1/agent-runs/{run_id}/messages` stores run-level guidance in resume payload.
- `POST /v1/agent-runs/{run_id}/subagents/{role_id}/messages` stores role-specific guidance.
- On resume, orchestrator sees role-specific guidance in `_role_guidance_messages`.

- [x] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_agent_run_operator_endpoints.py -k guidance_resume -q
```

Expected: fails because message events are mostly acknowledgement/guidance-only.

- [x] **Step 3: Persist guidance into run summary**

In `append_agent_run_message`, add to summary:

```python
summary["guidance_messages"] = [...]
```

In `append_agent_run_subagent_message`, add:

```python
summary["role_guidance_messages"][role_id] = [...]
```

Keep existing events unchanged for backward compatibility.

- [x] **Step 4: Feed guidance to orchestrator on resume/rerun**

When building `MultiAgentRunRequest`, include:

- global guidance messages
- role guidance messages

Ensure `_guidance_messages_for_role` receives targeted messages.

- [x] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_agent_run_operator_endpoints.py -k "guidance_resume or subagent_messages" -q
python -m pytest tests/test_goal_api.py -k "resume or guidance" -q
```

Expected: pass.

## Task 9: Add Normal Chat Subagent Delegation

**Files:**
- Modify: `mochi/agents/events.py`
- Modify: `mochi/agents/engine.py`
- Modify: `mochi/tools/registry.py`
- Create or modify: `mochi/tools/delegate_subagent.py`
- Modify: `mochi/api/routes/chat.py`
- Test: `tests/test_api_chat_models.py`
- Test: `tests/test_tool_exposure.py`

- [x] **Step 1: Define normal chat delegation behavior**

First wave:

- expose a `delegate_subagent` tool only when task complexity warrants it or user explicitly asks for subagents
- subagent is read/research/evidence by default
- subagent side effects remain disallowed unless routed through normal approval policy
- subagent transcript is scoped to chat `session_id` and current `turn_id`

- [x] **Step 2: Write failing chat stream test**

Add a fake backend/tool path where user says:

```text
請�??��?subagent ?�別??A ??B
```

Expected SSE events include:

- `subagent_started`
- `subagent_prompt`
- `subagent_completed`
- final answer

- [x] **Step 3: Run test and verify failure**

Run:

```powershell
python -m pytest tests/test_api_chat_models.py -k chat_subagent_stream -q
```

Expected: fails because chat has no subagent delegation.

- [x] **Step 4: Implement `delegate_subagent` tool/runtime hook**

Implement a bounded first-wave tool:

```python
class DelegateSubagentTool(Tool):
    name = "delegate_subagent"
    description = "Delegate a read-only research or analysis task to a subagent."
```

Parameters:

- `task`
- `role_name`
- `expected_output`
- `scope`

Runtime behavior:

- creates a transcript
- invokes `AgentEngine` or shared invocation helper with read-only profile
- emits transcript events through chat event callback
- returns concise result to main agent

- [x] **Step 5: Add tool exposure rule**

In tool exposure planner/router:

- expose `delegate_subagent` when user explicitly mentions subagents, parallel research, independent comparison, or multi-perspective work
- do not expose for trivial chat

- [x] **Step 6: Persist chat-scoped subagent transcripts**

Use `RuntimeStore.upsert_subagent_transcript` with:

- `parent_type="chat_turn"`
- `session_id`
- `parent_turn_id`
- no `goal_id`
- no `agent_run_id`

- [x] **Step 7: Run tests**

Run:

```powershell
python -m pytest tests/test_api_chat_models.py -k chat_subagent_stream -q
python -m pytest tests/test_tool_exposure.py -k subagent -q
```

Expected: pass.

## Task 10: Add Chat-Scoped Subagent Read/Message APIs

**Files:**
- Modify: `mochi/api/routes/sessions.py` or `mochi/api/routes/chat.py`
- Modify: `mochi/runtime/service.py`
- Modify: `web/src/lib/api.ts`
- Test: `tests/test_api_chat_models.py`

- [x] **Step 1: Write failing API tests**

Add endpoints:

- `GET /v1/sessions/{session_id}/subagents`
- `GET /v1/sessions/{session_id}/subagents/{subagent_id}`
- `POST /v1/sessions/{session_id}/subagents/{subagent_id}/messages`

For first wave, message endpoint may append guidance and mark it for next resume/follow-up if live steering is not implemented.

- [x] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_api_chat_models.py -k session_subagent_api -q
```

Expected: 404.

- [x] **Step 3: Implement service methods**

Add:

- `list_session_subagents(session_id)`
- `get_session_subagent(session_id, subagent_id)`
- `append_session_subagent_message(session_id, subagent_id, payload)`

- [x] **Step 4: Implement API routes**

Prefer `sessions.py` because transcript survives chat reload by session id.

- [x] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_api_chat_models.py -k session_subagent_api -q
```

Expected: pass.

## Task 11: Unify Chat And Goal Subagent UI

**Files:**
- Modify: `web/src/app/page.tsx`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/components/chat/ExecutionTimeline.tsx`
- Modify: `web/src/components/chat/SubagentDrawer.tsx`
- Test: frontend typecheck

- [ ] **Step 1: Add chat-scoped subagent state**

In `page.tsx` maintain:

```ts
const [sessionSubagents, setSessionSubagents] = useState<SubagentTranscriptSummary[]>([])
const [agentRunSubagents, setAgentRunSubagents] = useState<SubagentTranscriptSummary[]>([])
```

Merge them for display, keyed by `subagentId`.

- [ ] **Step 2: Fetch on session load**

When loading a session:

- fetch session subagents
- fetch active goal run subagents if bound
- merge timeline cards

- [ ] **Step 3: Update from stream**

When chat stream receives subagent events:

- update timeline
- upsert subagent summary
- open drawer remains live if selected

- [ ] **Step 4: Route guidance correctly**

If subagent has `agentRunId`, send to AgentRun subagent endpoint.

If subagent has `sessionId` only, send to session subagent endpoint.

- [ ] **Step 5: Run frontend checks**

Run:

```powershell
cd web
npm run lint
npm run typecheck
```

Expected: pass.

Manual checks:

- Normal chat can spawn a subagent and show it in timeline.
- Goal can spawn primary/workflow subagents and show them in the same timeline.
- Opening either kind uses the same drawer.

## Task 12: Replace Goal Status-Only Follow-Up With Live Continuation

**Files:**
- Modify: `web/src/app/page.tsx`
- Modify: `web/src/lib/chat-goal-continuation.ts`
- Modify: `mochi/runtime/service.py`
- Test: `tests/test_goal_api.py`

- [ ] **Step 1: Write failing behavior test at API level**

Backend test:

- create Goal linked to paused/stalled run
- post follow-up/guidance
- resume same run when safe
- assert run status changes and guidance is in resume payload

- [ ] **Step 2: Frontend behavior expectation**

Document expected manual behavior:

- User says "繼�?，改??X".
- Chat shows "forwarded guidance and resumed run".
- Live timeline starts updating.
- No generic "operator required" unless real blocker exists.

- [ ] **Step 3: Update continuation decision**

Ensure:

- `monitor` -> forward guidance and keep timeline live
- `capture_checkpoint` -> forward guidance, do not block
- `refresh_worker_generation` -> refresh then subscribe
- `resume_goal` -> resume then subscribe
- `resolve_approval` -> show approval controls and exact blocker

- [ ] **Step 4: Update `page.tsx` activeGoalFollowUpRequested path**

After append/resume:

- ensure active run id is set
- start event polling/streaming
- append only a concise acknowledgement, then let timeline show real work

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_goal_api.py -k "resume or waiting_approval" -q
```

Expected: pass.

## Task 13: Add SSE For AgentRun Events

**Files:**
- Modify: `mochi/api/routes/agent_runs.py`
- Modify: `mochi/runtime/store.py`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/app/page.tsx`
- Test: `tests/test_agent_run_operator_endpoints.py`

- [ ] **Step 1: Write failing SSE test**

Test:

- open `/v1/agent-runs/{run_id}/events/stream`
- append an event
- receive event over SSE

- [ ] **Step 2: Run test and verify failure**

Run:

```powershell
python -m pytest tests/test_agent_run_operator_endpoints.py -k agent_run_events_stream -q
```

Expected: 404.

- [ ] **Step 3: Implement streaming endpoint**

Add:

```python
@router.get("/{run_id}/events/stream")
async def stream_agent_run_events(...):
    ...
```

Use polling with `after_seq` initially. Keep timeout and heartbeat bounded.

- [ ] **Step 4: Frontend switch from polling to SSE**

Use SSE when available, fall back to polling on error.

- [ ] **Step 5: Run tests**

Run:

```powershell
python -m pytest tests/test_agent_run_operator_endpoints.py -k agent_run_events_stream -q
```

Expected: pass.

## Task 14: Permission And Approval Integration

**Files:**
- Modify: `mochi/tools/delegate_subagent.py`
- Modify: `mochi/runtime/service.py`
- Modify: `web/src/components/chat/SubagentDrawer.tsx`
- Test: `tests/test_api_chat_models.py`
- Test: `tests/test_goal_api.py`

- [ ] **Step 1: Add tests for approval-required subagent tool call**

Simulate subagent requesting `exec_command` or file write.

Assert:

- approval request is persisted
- transcript includes `subagent_tool_result` with `status="approval_required"`
- main chat/goal timeline shows approval blocker

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_api_chat_models.py -k subagent_approval -q
python -m pytest tests/test_goal_api.py -k subagent_approval -q
```

- [ ] **Step 3: Wire approval ids into transcript events**

When tool returns `requires_approval`, include:

- approval id
- tool name
- reason
- replay safety metadata

- [ ] **Step 4: UI approval affordance**

In `SubagentDrawer`:

- show approval needed row
- link to existing Goal drawer approval controls for goal-linked runs
- for chat-scoped approvals, use existing approvals API controls if available

- [ ] **Step 5: Run tests**

Expected: approval metadata is visible and no unauthorized action runs.

## Task 15: Reload And Session Persistence

**Files:**
- Modify: `web/src/lib/stores/session-store.ts`
- Modify: `web/src/lib/chat-projections.ts`
- Modify: `web/src/app/page.tsx`
- Test: existing session/chat tests

- [ ] **Step 1: Write reload projection test**

Add frontend/unit test if existing test infra supports it. Otherwise add a backend session API test:

- persist chat turn with subagent transcript references
- reload session
- fetch subagent list
- reconstruct visible subagent timeline

- [ ] **Step 2: Implement projection restore**

Ensure `chat-projections.ts` can reconstruct:

- Goal cards
- execution timeline marker
- subagent timeline cards

Do not write synthetic assistant messages into canonical model replay history unless there is already a display-marker contract.

- [ ] **Step 3: Run checks**

Run:

```powershell
python -m pytest tests/test_api_chat_models.py -k "sessions or turn_events" -q
cd web
npm run typecheck
```

Expected: pass.

## Task 16: Visual And Interaction QA

**Files:**
- Modify as needed:
  - `web/src/components/chat/ExecutionTimeline.tsx`
  - `web/src/components/chat/SubagentDrawer.tsx`
  - `web/src/app/page.tsx`

- [ ] **Step 1: Start dev server**

Run:

```powershell
cd web
npm run dev
```

- [ ] **Step 2: Desktop QA**

Scenarios:

- normal chat asks for two subagents
- single-agent Goal starts
- workflow Goal starts
- Goal enters approval wait
- user opens subagent drawer and sends guidance

Expected:

- no overlapping text
- drawer is usable
- prompt tab scrolls
- timeline updates without layout shift
- blocked state explains exact cause

- [ ] **Step 3: Mobile QA**

Use Playwright or browser responsive mode:

- 390px width
- 768px width
- 1440px width

Expected:

- timeline rows fit
- drawer/panel does not obscure composer permanently
- buttons have stable dimensions

- [ ] **Step 4: Browser console QA**

Confirm no React key warnings, hydration warnings, or failed API calls in expected flows.

## Task 17: Documentation And Handoff

**Files:**
- Modify: `docs/architecture/2026-06-27-chat-goal-workflow-runtime-architecture.md`
- Create: `docs/architecture/chat-and-goal-subagent-transcript-contract.md`
- Modify memory after implementation if requested.

- [ ] **Step 1: Document the transcript contract**

Include:

- parent scopes
- event types
- persistence tables
- SSE behavior
- UI projection rules
- permission/approval behavior

- [ ] **Step 2: Update architecture doc**

Clarify:

- Chat can spawn subagents without Goal.
- Goal uses same transcript infrastructure.
- Operator console is not the primary surface.

- [ ] **Step 3: Add migration note**

Explain how old agent-run events still render and how new subagent transcripts are additive.

## Verification Matrix

Run before considering implementation complete:

```powershell
python -m pytest tests/test_execution_transcript.py -q
python -m pytest tests/test_runtime_store.py -k subagent_transcript -q
python -m pytest tests/test_agent_run_operator_endpoints.py -q
python -m pytest tests/test_goal_api.py -k "waiting_approval or resume or single_agent or recommended_next_action" -q
python -m pytest tests/test_api_chat_models.py -k "stream or subagent or sessions" -q
python -m pytest tests/test_tool_exposure.py -k subagent -q
```

Frontend:

```powershell
cd web
npm run lint
npm run typecheck
```

Manual:

- Normal chat: "?�兩??subagent ?�別?�究 A/B" shows subagent timeline and drawer.
- Goal single-agent: starts with a primary subagent transcript.
- Goal workflow: each role is visible as a subagent.
- Approval wait: main chat says exactly what is waiting and provides the relevant controls.
- User can send targeted guidance to a subagent.
- Reload preserves subagent timeline and details.

## Implementation Order

1. Task 1: backend transcript models.
2. Task 2: persistence.
3. Task 4: read APIs.
4. Task 3: orchestrator persistence.
5. Task 5: Goal blocker diagnostics.
6. Task 6: Goal AgentRun event projection in chat.
7. Task 7: subagent drawer.
8. Task 8: guidance affects resume.
9. Task 9: normal chat subagent delegation.
10. Task 10: chat-scoped subagent APIs.
11. Task 11: unified chat/goal subagent UI.
12. Task 12: live continuation behavior.
13. Task 13: SSE.
14. Task 14: approval integration.
15. Task 15: reload persistence.
16. Task 16: visual QA.
17. Task 17: docs.

This order intentionally delivers value before the full normal-chat delegation system:

- Goal users first get clarity and live visibility.
- Existing AgentRun data becomes visible before adding new subagent creation paths.
- Normal chat subagents reuse the same infrastructure instead of creating a parallel system.
