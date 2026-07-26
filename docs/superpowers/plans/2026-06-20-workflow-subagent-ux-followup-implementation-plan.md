# Workflow and Subagent UX Follow-up Implementation Plan

Date: 2026-06-20
Status: completed
Completed: 2026-06-21
Scope baseline: workflow/subagent main-chat projection and TaskPanel UX follow-ups
Primary memory reference: `.claude/skills/agent-memory/memories/project-status/workflow-subagent-ux-followups-2026-06-20.md`

## Summary

This plan turns the current workflow/subagent follow-up backlog into a tracked implementation list.

The goal is to improve correctness and UX without:

- introducing a second persistence model
- polluting canonical chat history or prompt history
- adding backend event types before the display-layer path is clearly insufficient
- spreading fragile projection logic across multiple components

## Relevant Files

- `web/src/app/page.tsx`
- `web/src/components/chat/TaskPanel.tsx`
- `web/src/components/chat/SubagentTaskCard.tsx`
- `web/src/components/chat/ToolCallCard.tsx`
- `web/src/lib/subagent-tasks.ts`
- `web/src/lib/chat.ts`
- `web/src/lib/api.ts`
- `web/scripts/`

## Guardrails

- Keep workflow/subagent cards out of canonical session message history unless a deliberate persistence contract is approved.
- Do not reintroduce generic workflow lifecycle noise into the main chat.
- Prefer extracting pure frontend helpers before adding new runtime or backend contracts.
- Keep the first cut narrow. Fix the current UX path before generalizing.
- Do not create a new generic task-panel framework or message-projection framework unless the current change set proves it necessary.

## Execution Order

1. Stabilize projected-message reconstruction and ordering.
2. Add a focused TaskPanel mode for subagent entry points.
3. Add immediate delegated-subagent creation feedback in the display layer.
4. Decide whether any projected UX needs durable persistence beyond the current display layer.

This order is intentional. Item 1 reduces risk for items 2 and 3. Item 4 is a decision gate, not an automatic implementation task.

## Progress Tracker

Status legend:

- `[ ]` not started
- `[-]` in progress
- `[x]` completed
- `[!]` blocked pending product or contract decision

Overall status:

- `[x]` completed

## Workstream 1: Projection Correctness and Reload Safety

Owner: main agent
Suggested support: verification subagent
Status: [x] code and script verification landed; browser reload smoke still desirable

Goal:

- make workflow card, workflow completion report, subagent card, and subagent latest-result projection deterministic across reload and session switching

Tasks:

- [x] Extract projected-message assembly from `web/src/app/page.tsx` into a pure helper at `web/src/lib/chat-projections.ts`.
- [x] Keep helper inputs narrow:
  - canonical chat messages
  - current workflow run detail
  - contextual runtime tasks
- [x] Preserve timestamp-based insertion rules for:
  - workflow progress card
  - workflow completion report
  - subagent task cards
- [x] Verify old sessions with `workflow.bound_run_id` still rebuild projected messages after fetch/poll.
- [x] Verify projected messages do not duplicate when runtime tasks and tool-result-derived cards refer to the same delegated task.
- [x] Add script-level tests for reload and session-switch reconstruction.
- [x] Add script-level tests for display order when later ordinary chat messages appear after projected workflow content.

Acceptance criteria:

- page reload restores the same projected workflow/subagent cards for a bound run
- switching sessions does not leak projected cards across sessions
- projected cards are stable when task data arrives later than chat replay
- later normal assistant or user messages still render in correct timestamp order

## Workstream 2: Subagent-Focused TaskPanel Mode

Owner: worker A
Suggested support: explorer for file references only
Status: [x] code and script verification landed

Goal:

- opening from a subagent card should feel like entering that subagent conversation, not a generic task monitor

Tasks:

- [x] Add a small opening-context state in `web/src/app/page.tsx`:
  - `default`
  - `subagent`
- [x] Track an optional focused task id when opening from `SubagentTaskCard` or delegated-subagent tool cards.
- [x] Pass the mode and focused task id into `TaskPanel`.
- [x] In `TaskPanel`, prioritize the focused subagent conversation when mode is `subagent`.
- [x] Keep `Pending approvals` visible when blocking, but allow it to collapse or move below the focused conversation in subagent mode.
- [x] Keep `Workflow run` visible when blocking or relevant, but avoid letting it displace the subagent conversation.
- [x] Preserve current header `Tasks` button behavior as the general task monitor entry point.
- [x] Add script tests covering:
  - open from header
  - open from `Open subagent conversation`
  - open from delegated-subagent tool card

Acceptance criteria:

- card-triggered open lands on the correct delegated task
- header-triggered open still behaves like the current general-purpose panel
- approvals remain reviewable and are not hidden behind the focused mode
- no extra global state model is introduced just for panel layout

## Workstream 3: Immediate Subagent Creation Feedback

Owner: worker B
Suggested support: verification subagent
Status: [x] code and script verification landed

Goal:

- show fast feedback when `delegate_subagent_task` starts, before the tool result resolves

Tasks:

- [x] Extend the projection layer to derive a pending delegated-subagent card from `tool_call_request` or `tool_call` state for `delegate_subagent_task`.
- [x] Use `toolCallId` as the join key between the pending placeholder and the eventual tool result when available.
- [x] Add a pending card state such as `Creating subagent...`.
- [x] Upgrade the pending card to the existing delegated-subagent card when the tool result arrives.
- [x] Show an error state if the delegated-subagent tool call fails.
- [x] Keep the implementation display-only for the first cut.
- [x] Add script tests for:
  - pending placeholder shown during tool call
  - placeholder replaced by created card on success
  - placeholder turns into error state on failure

Acceptance criteria:

- the user sees immediate feedback at tool-call start
- success and failure do not create duplicate cards
- no backend event or session-history write is required for the first cut

## Workstream 4: Persistence Strategy Decision Gate

Owner: main agent
Suggested support: design or architecture review subagent
Status: [x] decision recorded; implementation deferred by design

Goal:

- decide whether any projected workflow/subagent UX needs durable persistence beyond the current display layer

Tasks:

- [x] Document current behavior:
  - workflow state persists through session workflow metadata
  - projected cards are rebuilt in the frontend
  - workflow lifecycle `turn_event` entries are intentionally filtered from chat replay
- [x] Compare three options:
  - keep display-only projection
  - add display-only session markers outside prompt history
  - append durable assistant-visible history with explicit anti-pollution metadata
- [x] Identify risks for each option:
  - prompt-context pollution
  - duplicate rendering
  - mismatched completion timing
  - cross-client inconsistency
- [x] Decide whether implementation is needed now or should stay deferred.

Acceptance criteria:

- there is an explicit recorded decision
- no backend persistence change lands without a clear product reason

Recorded decision:

- [x] Keep projected workflow/subagent UX display-only for now.
- [x] Do not add backend persistence/events in this phase.
- [x] If persistence is needed later, evaluate display-only session markers before prompt-visible assistant history.

## Suggested Subagent Split

Use subagents only for bounded parallel work with disjoint ownership.

- Main agent:
  - `web/src/app/page.tsx`
  - final integration
  - plan ownership
- Worker A:
  - `web/src/components/chat/TaskPanel.tsx`
  - TaskPanel mode behavior
  - related `web/scripts/test-task-panel-*.mjs`
- Worker B:
  - `web/src/lib/subagent-tasks.ts`
  - `web/src/components/chat/SubagentTaskCard.tsx`
  - projection helper support for delegated-subagent placeholder lifecycle
- Worker C:
  - verification scripts for reload, session switching, and projection ordering
  - avoid editing UI components unless a test seam is missing

Rules for subagents:

- Assign disjoint write scopes.
- Do not let multiple workers edit `web/src/app/page.tsx` at the same time unless one is read-only.
- Do not introduce backend contract changes from a worker task unless the main agent explicitly promotes that work.

## Verification Plan

Minimum verification for this phase:

- `npm.cmd run type-check`
- targeted `web/scripts/*.mjs` checks added for:
  - projection reconstruction
  - TaskPanel opening mode behavior
  - delegated-subagent pending card lifecycle

Prefer script and unit-style verification before browser-driven flows.

Manual smoke checks after implementation:

- reload an active workflow-bound chat
- switch between a workflow-bound session and an ordinary chat
- open TaskPanel from the header
- open TaskPanel from a subagent card
- observe delegated-subagent creation from tool-call start to tool result

## Non-Goals

- no broad TaskPanel redesign
- no new generic workflow event system
- no backend append of synthetic workflow completion chat messages in this phase
- no cross-client synchronization redesign
- no conversion of projected cards into canonical assistant messages by default

## Decision Log

- 2026-06-20: Start with projection correctness first. Do not begin with persistence redesign.
- 2026-06-20: Keep the first cut frontend-only unless a concrete failure proves the backend contract is insufficient.
- 2026-06-20: Workstream 1 landed with `web/src/lib/chat-projections.ts` plus `web/scripts/test-chat-projections.mjs`.
- 2026-06-20: Workstream 2 landed with explicit `TaskPanel` mode and focused delegated-task wiring.
- 2026-06-21: Workstream 3 landed with pending/error delegated-subagent placeholder cards and lifecycle script coverage.
- 2026-06-21: Workstream 4 decision accepted in `2026-06-21-workflow-subagent-projection-persistence-decision.md`; backend persistence remains deferred.
- 2026-06-21: Desktop `WorkspacePanel` no longer overlays and blocks delegated-subagent card interaction; `page.tsx` now reserves left-side chat space while the panel is open.
- 2026-06-21: Browser smoke re-check confirmed the delegated-subagent card remains clickable while the desktop workspace panel is open.

## Implementation Notes

If execution starts immediately, the smallest safe first PR cut is:

1. extract projection helper plus tests
2. land TaskPanel focused mode
3. land pending delegated-subagent placeholder

Do not start Workstream 4 implementation work unless the first three streams expose a real product or contract gap.
