# Workflow/Subagent Projection Persistence Decision

Date: 2026-06-21
Status: accepted
Scope: main-chat projected workflow cards, workflow completion report, and delegated-subagent cards

## Summary

Keep workflow and delegated-subagent projected UX display-only for now.

Do not append these projected items into canonical assistant chat history.
Do not add new backend event types in this phase.
Do not persist synthetic projected messages as prompt-visible session content.

If a future persistence need appears, prefer a display-only marker contract over durable assistant-visible history.

## Current Behavior

Current implementation is intentionally split across two layers:

- session workflow state persists through session metadata updates
- canonical session replay rebuilds normal chat from session events
- projected workflow/subagent cards are rebuilt in the frontend

Important current details:

1. Workflow binding state is durable.
   - session metadata stores `workflow_state_updated`
   - `workflow.bound_run_id` and `workflow.synced_run_event_count` survive reload/session reopen

2. Workflow runtime events are partially synced into the session, but not as main-chat UX cards.
   - `syncWorkflowRunEventsToSession(...)` maps workflow activity into session events
   - workflow lifecycle/artifact/exec phases are stored as `turn_event`
   - `buildMessagesFromSessionEvents(...)` intentionally filters:
     - `workflow_status`
     - `workflow_artifact`
     - `workflow_exec_update`

3. Main-chat workflow/subagent cards are display projections.
   - workflow progress and completion are derived from:
     - canonical messages
     - current workflow run detail
     - contextual runtime tasks
   - delegated-subagent creation, pending state, success state, and error state are derived from reasoning/tool activity plus runtime task data

4. These projected cards are not appended back into canonical prompt history.
   - this avoids synthetic assistant content polluting future model context
   - this avoids duplicate rendering between canonical replay and derived display

## Options Considered

### Option A: Keep Display-Only Projection

Description:

- continue rebuilding workflow/subagent cards in the frontend
- persist only the minimum workflow binding metadata already needed for reconstruction

Advantages:

- no prompt-history pollution
- no duplicate assistant messages
- no backend schema/event expansion
- minimal risk of completion-timing drift
- keeps product behavior aligned with current architecture

Disadvantages:

- projected pending states are client-local, not cross-client durable
- another client may not see the exact same transient projection timing
- export/replay remains canonical-chat-first, not UX-card-first

Assessment:

- best fit for the current product stage

### Option B: Add Display-Only Session Markers Outside Prompt History

Description:

- add a durable session-level marker contract specifically for projected UX
- markers remain excluded from prompt-visible canonical assistant history

Advantages:

- better cross-client reconstruction
- still avoids prompt pollution if kept outside canonical assistant content
- safer than writing synthetic assistant messages

Disadvantages:

- adds backend/session contract complexity
- requires dedupe rules with existing frontend projection
- needs explicit lifecycle semantics for pending vs created vs failed states

Assessment:

- the best fallback if display-only projection becomes insufficient
- not justified yet

### Option C: Append Durable Assistant-Visible History

Description:

- write workflow/subagent projection results directly into canonical chat history as assistant messages

Advantages:

- simplest mental model for replay and export
- every client sees the same synthetic messages automatically

Disadvantages:

- high risk of prompt-context pollution
- duplicate or stale display risk when workflow state changes later
- completion timing can drift from the true workflow lifecycle
- makes transient operational UX look like actual assistant authored conversation

Assessment:

- reject for now

## Decision

Choose Option A now.

That means:

- keep projected workflow/subagent UX display-only
- keep canonical session history free of synthetic workflow/subagent cards
- do not append workflow completion reports as assistant messages
- do not add a new backend persistence/event type in this phase

If persistence becomes necessary later, revisit with Option B first.
Do not jump directly to Option C.

## Why This Decision Is Correct Now

The current product goals are:

- quiet main chat
- correct prompt boundaries
- minimal backend churn
- fast UX iteration at the display layer

Option A matches all four.

The recent work already solved the most pressing user-facing gaps:

- reload/session-switch reconstruction
- focused TaskPanel entry
- immediate delegated-subagent creation feedback

None of those fixes required a new persistence contract.
That is a strong signal not to expand the backend yet.

## Revisit Triggers

Re-open this decision only if one of these becomes real:

- cross-client consistency for pending or created delegated-subagent cards becomes a product requirement
- users need exported transcripts to include workflow/subagent UX cards as first-class artifacts
- reload reconstruction proves insufficient without extra server-authored markers
- multiple clients show materially confusing divergence for the same session

## Non-Decision

This document does not approve:

- writing projected cards into canonical assistant history
- adding workflow/subagent persistence events immediately
- redesigning session replay around workflow UX

## References

- `web/src/lib/chat-projections.ts`
- `web/src/lib/api.ts`
- `web/src/app/page.tsx`
- `mochi/api/routes/sessions.py`
- `docs/superpowers/plans/2026-06-20-workflow-subagent-ux-followup-implementation-plan.md`
