# Runtime Cancellation Parity Remaining Spec

Date: 2026-06-30

This document is a follow-up implementation spec for the remaining cancellation/parity work after the subagent runtime parity pass.
It is intentionally narrower than the original `2026-06-29-chat-and-goal-subagent-runtime-parity.md` plan and should be treated as the current source of truth for unresolved cancellation/runtime gaps.

## Goal

Close the remaining gap between Mochi and mature agent runtimes such as Codex, OpenClaw, and Hermes in the specific area of cancellation semantics.

The target behavior is:

- pressing Stop in main chat should stop real backend work, not only the browser stream
- delegated subagents should already keep their current truthful cancellation semantics
- run-level cancellation should propagate consistently across chat, delegated subagents, and supported long-running tools
- unsupported paths must remain explicitly best-effort or deferred, never overclaiming success

## Already Completed

Do not re-open these unless a regression is found:

- delegated subagent protocol-level interruption/inbox delivery
- durable transcript persistence for subagent control events
- true `cancel_current_tool` support for:
  - `exec_command`
  - `execute_code`
  - `execute_code_v2`
  - `web_fetch`
  - `web_crawl`
  - `web_search`
- delegated subagent `mid-generation cancellation` on supported runtime/model paths
  - `invoke(...)` branch
  - `generate_with_configured_model(...)` fallback branch
- truthful protocol semantics:
  - no fake `subagent_tool_cancelled` on interrupt-only restarts
  - no fake success when cancellation does not actually unwind

## Remaining Product Gaps

### 1. Main Chat Stop Is Still Not Full Runtime Cancellation

Current state:

- delegated subagent side-thread cancellation is substantially improved
- top-level normal chat Stop is still primarily a client-side fetch abort
- server-side run interruption on disconnect/cancel is not yet at Codex/OpenClaw/Hermes parity

Target state:

- Stop in normal chat issues a real server-side cancellation request
- the backend run is interrupted or cancelled at the runtime boundary
- the UI reflects truthful terminal state:
  - `cancelled` if the run really unwound
  - `completed` if the response beat the cancel request
  - `best_effort_disconnect` or equivalent internal handling if the transport died before confirmation

### 2. Run-Level Cancellation Propagation Is Not Yet Unified

Current state:

- cancellation semantics are now stronger in several individual slices
- there is still no single run-scoped cancellation primitive propagated through all relevant runtime layers

Target state:

- one run-scoped cancellation context exists for:
  - direct chat runs
  - delegated subagent invocations
  - tool execution boundaries
  - supported backend/model invocations
- cancellation semantics are conservative and consistent across all call sites

### 3. True-Cancellable Tool Coverage Is Broader, But Not Universal

Current state:

- current true-cancellable set is `exec/code/web fetch-crawl-search`
- many other long-running tools still only defer

Target state:

- the next tier of genuinely long-running, foreground, user-visible tools should gain true cancellation support where there is a real unwind boundary
- tools without a real cancellation handle must remain defer-only

## Non-Goals

- Do not rewrite the delegated subagent transcript model again.
- Do not emit synthetic success events for unsupported runtimes.
- Do not claim parity with Hermes/OpenClaw/Codex on local in-process model paths that cannot actually unwind safely.
- Do not add cancellation support to MCP/dynamic tools unless the underlying MCP runtime can expose a real cancel handle.
- Do not revert unrelated existing worktree changes.

## Design Principles

### Truthful Semantics

Only emit success/cancel-complete semantics when the underlying work really unwound.

Rules:

- `subagent_tool_cancelled` remains reserved for true tool/runtime cancellation completion
- interrupt-only delegated mid-generation restart must not emit tool-cancel completion
- top-level chat stop must not pretend a run is cancelled if the model reply already completed

### Conservative Capability Gating

Prefer deferral/best-effort over false confidence.

Rules:

- server-backed HTTP/streaming model paths may support true mid-generation cancellation
- local blocking/in-process paths may remain defer-only or best-effort
- a tool is only cancellable if it can bind to a real process/task/request boundary

### Run-Scoped Ownership

Cancellation belongs to a run first, then fans into the currently active generation/tool boundary.

Rules:

- main chat stop targets the current chat run
- delegated subagent stop targets the current delegated invocation/tool
- tool cancellation is subordinate to run cancellation, not a separate global control plane

## Workstream A: Main Chat Server-Side Stop Parity

### Scope

- `mochi/api/routes/chat.py`
- `mochi/agents/engine.py`
- `web/src/lib/api.ts`
- `web/src/app/page.tsx`
- any small shared runtime registry/helper if needed

### Requirements

1. Add a server-side cancellation path for active normal-chat runs.
2. Stop must target real in-flight backend work, not only abort the browser request.
3. Stream teardown/disconnect must not leave the backend worker running silently.
4. Persisted chat state must remain truthful when cancellation races with completion.

### Backend Design

Introduce a direct-chat active run registry keyed by a stable run identity:

- preferred key shape:
  - `session_id`
  - `turn_id` or generated `chat_run_id`

Store:

- active worker task
- cancellation function
- current state:
  - `running`
  - `cancelling`
  - `cancelled`
  - `completed`

Add API:

- `POST /v1/chat/{session_id}/cancel`

Request body:

```json
{
  "turn_id": "optional-current-turn-id"
}
```

Response:

```json
{
  "status": "cancel_requested|already_completed|not_found",
  "session_id": "session-id",
  "turn_id": "turn-id-or-null"
}
```

### Engine Design

`AgentEngine._run_chat(...)` must support real worker cancellation.

Required behavior:

- when the async iterator is torn down early, cancel the worker task instead of merely awaiting it
- when explicit cancel is requested, invoke the same worker cancellation path
- cancellation success is confirmed only when the invocation task actually unwinds

Implementation notes:

- the direct-chat path should reuse the same mental model already used in delegated subagent mid-generation cancellation
- do not rely on browser disconnect alone as the only signal

### Frontend Design

`handleStopGeneration` should:

1. call the new cancel endpoint
2. then abort the local fetch stream
3. update local UI state from the cancel response rather than from transport abort alone

The UI should not insert a fake assistant message unless the backend already persists one.

### Acceptance Criteria

- normal chat Stop interrupts a long-running supported backend generation server-side
- disconnect/teardown does not leave the agent worker continuing silently
- if completion wins the race, the final state is reported as completed, not cancelled

## Workstream B: Unified Run-Level Cancellation Context

### Scope

- `mochi/agents/invocation.py`
- `mochi/agents/engine.py`
- `mochi/tools/base.py`
- runtime glue where direct chat and delegated subagents meet

### Requirements

1. Define a run-scoped cancellation context that can be passed through invocation boundaries.
2. Make direct chat and delegated subagent paths use compatible semantics.
3. Preserve the distinction between:
   - run interruption
   - active tool cancellation
   - deferred/no-op cancellation

### Proposed Contract

Add a small runtime cancellation context object containing:

- `run_id`
- `cancel_requested`
- `cancel_confirmed`
- optional callbacks:
  - `cancel_generation`
  - `cancel_active_tool`

This context is not a user-visible model.
It is internal plumbing for consistent propagation.

### Required Behavior

- direct chat run cancellation should set run-scoped cancel intent first
- delegated subagent mid-generation cancellation may bind the same run-scoped primitive to the current invocation task
- active tool cancellation may still use `ActiveToolController`, but it should compose cleanly with the run-scoped context

### Acceptance Criteria

- direct chat and delegated subagent code paths no longer model cancellation in completely separate ad hoc ways
- there is one obvious place to ask: “is this run cancellable right now, and what boundary will be cancelled?”

## Workstream C: Next Long-Running Tool Cancellation Tier

### Priority Rule

Only expand to tools that satisfy all of:

1. can run long enough for cancellation to matter
2. are user-visible in transcript/runtime UX
3. have a real unwind boundary
4. can be tested deterministically

### Candidate Priority

Priority 1:

- literature search family with direct HTTP boundaries:
  - `arxiv_search`
  - `semantic_scholar_search`
  - `crossref_search`
  - `pubmed_search`

Priority 2:

- any other foreground HTTP tool that already routes through the shared HTTP helper and can bind to the active task cleanly

Explicitly out for now:

- `mcp_call`
- arbitrary dynamic MCP tools
- anything that cannot surface a real cancellation handle

### Implementation Pattern

Reuse the current HTTP-bound cancellation boundary where possible.

Rules:

- mark tool `is_cancellable=True` only when the execution path really binds to cancellation
- only return cancelled metadata/result when the coroutine/request actually unwinds from cancellation
- otherwise preserve current defer-only semantics

### Acceptance Criteria

- at least one next-tier literature/search tool family gains true cancellation with deterministic tests
- unsupported tools still emit defer-only semantics and do not regress

## Protocol And Event Semantics

These rules are mandatory:

- `subagent_tool_cancelled`
  - only for `cancel_current_tool`
  - only when the underlying tool/runtime boundary actually unwound
- interrupt-only delegated restart:
  - emit `subagent_interrupted`
  - apply/restart guidance
  - do not emit tool-cancel completion
- main chat stop:
  - may use chat/run-level terminal state
  - must not reuse misleading subagent-tool protocol names for top-level chat runs

## Testing Matrix

### Direct Chat Stop

Add focused tests for:

- explicit cancel endpoint interrupts a long-running supported chat generation
- client disconnect/iterator teardown cancels the worker
- completion beating cancellation remains `completed`
- unsupported path is reported conservatively

### Unified Run-Level Semantics

Add focused tests for:

- run cancellation while no tool is active
- run cancellation while a cancellable tool is active
- run cancellation while a non-cancellable tool is active

### Next-Tier Tool Coverage

For every new tool family:

- cancellation request reaches the real in-flight request/task
- mocked in-flight request observes `CancelledError` or equivalent unwind
- returned `ToolResult.metadata.status == "cancelled"` only on true unwind
- no false-positive cancel on late parse/error path

## Verification Commands

Minimum backend verification after implementation:

```powershell
python -m pytest tests/test_api_chat_models.py -k "chat_subagent_stream or delegated_subagent_runtime_events or live or session_subagent_api or cancel or resume or interruption or delivery or after_current_tool_message or interrupt_cancel_message or persists_interrupt_cancel_protocol_events" -q
python -m pytest tests/test_execution_transcript.py tests/test_runtime_store.py -k "execution_transcript or subagent_transcript or runtime_blocked or approval or delivery" -q
python -m pytest tests/test_exec_tools.py -k cancellation -q
python -m pytest tests/test_execute_code_and_mcp.py -q
python -m pytest tests/test_web_fetch.py tests/test_web_crawl.py tests/test_web_search.py -q
git diff --check
```

Expected caveat:

- pytest may emit `.pytest_cache` permission warnings without indicating test failure

If Workstream A touches frontend stop wiring, also run:

```powershell
cd web
npm.cmd run type-check
```

## Rollout Order

1. Workstream A: main chat server-side stop parity
2. Workstream B: unify run-scoped cancellation plumbing
3. Workstream C: expand next-tier long-running tool coverage
4. final parity review against Codex/OpenClaw/Hermes semantics

## Reference Comparison Guidance

### Codex

Desired parity:

- Stop should usually mean the backend run actually stops
- side-thread delegated work should remain truthful and steerable

Acceptable difference:

- Mochi may stay conservative and defer on unsupported local-model paths instead of pretending full parity

### OpenClaw

Desired parity:

- run-scoped abort semantics should be clear and explicit

Acceptable difference:

- Mochi does not need a TypeScript `AbortController`-shaped implementation as long as semantics are equivalent

### Hermes

Desired parity:

- interrupt should propagate to the running agent/runtime, not only the client transport

Acceptable difference:

- Mochi may retain its own orchestration structure if run-level cancellation and transcript semantics remain truthful

## Done Definition

This remaining spec is complete only when all of the following are true:

- main chat Stop issues and confirms real server-side cancellation on supported paths
- delegated subagent cancellation semantics remain truthful after integration
- next-tier long-running tool coverage expands without overclaiming
- unsupported paths remain explicitly conservative
- verification passes
- final review finds no blocking semantic mismatch against the stated product contract
