# Goal Conversational Intent, Strategy Selection, and UX Recovery Plan

Date: 2026-07-01
Status: revised and active
Scope baseline: recover Goal as a chat-first, intent-driven long-running execution surface; separate Goal from Workflow/Protocol selection; and remove hardcoded routing as the primary product behavior.

## Scope Correction

This document replaces the earlier same-day "completed" framing.

Some earlier fixes remain useful:

- blocked and approval health preload
- continuation-state wiring
- timeline projection cleanup
- follow-up seeding into resumed attempts

But the previous plan still assumed the wrong center of gravity:

- too much behavior was still framed as route selection
- Goal follow-up was still too mechanical
- Workflow was still treated too much like a top-level lane

That premise is now explicitly rejected.

The corrected product contract is:

- `Goal` is a durable autonomy contract
- `Workflow` or `Protocol` is an optional execution strategy
- the model should choose strategy from described options unless the user explicitly overrides it
- slash commands remain valid overrides, not the main reasoning mechanism
- Goal should behave like Codex-style autonomous chat, not like a command router

## References

### Internal architecture and plans

- `docs/architecture/2026-06-27-chat-goal-workflow-runtime-architecture.md`
- `docs/architecture/chat-and-goal-subagent-transcript-contract.md`
- `docs/superpowers/plans/2026-06-25-chat-first-goal-unification-plan.md`
- `docs/superpowers/plans/2026-06-29-chat-and-goal-subagent-runtime-parity.md`
- `goal-workflow-handoff-2026-06-25.md`
- `.claude/skills/agent-memory/memories/architecture/chat-goal-workflow-runtime-current-state-2026-06-27.md`

### Current implementation review inputs

- `web/src/app/page.tsx`
- `web/src/lib/chat-goal-routing.ts`
- `web/src/lib/chat-goal-continuation.ts`
- `web/src/lib/goal-proposal-copy.ts`
- `web/src/lib/execution-transcript.ts`
- `web/src/components/chat/ExecutionTimeline.tsx`
- `mochi/goal_intent.py`
- `mochi/goal_proposal_copy.py`
- `mochi/api/routes/goals.py`
- `mochi/agents/multi_agent/protocols.py`

### Reference implementations reviewed

- OpenClaw:
  - `reference/openclaw/src/agents/subagent-control.ts`
  - `reference/openclaw/src/agents/subagent-registry-runtime.ts`
- Hermes Agent:
  - `reference/hermes-agent/ui-tui/src/app/turnController.ts`
  - `reference/hermes-agent/ui-tui/src/app/createGatewayEventHandler.ts`
- cc-haha:
  - `reference/cc-haha/README.md`

### Reference conclusions that lock this direction

- OpenClaw shows that run control is a durable contract with lineage, ownership, steer, restart, wake, and resume semantics. It is not driven by phrase routing.
- Hermes shows the right UI split: durable messages, transient thinking, tools, approvals, subagents, and status all have different lifetimes.
- cc-haha shows the product shape: chat-first workspace, with workflows, approvals, skills, and automation as optional capabilities inside one workbench.

No external paper is required to justify this direction. The main evidence is product behavior, current code review, and vendored reference implementations.

## Corrected Product Contract

### Canonical product model

This plan is the canonical direction for the chat, Goal, Workflow, and Protocol relationship.

- `Chat` is the normal conversational surface.
- `Goal` is the durable autonomy contract plus persisted runtime state.
- `Strategy` is the selected execution approach under a Goal.
- `Workflow` is one possible strategy family, not a peer top-level route.
- `Protocol` is an implementation detail for strategies that need multi-agent orchestration.
- Slash commands are explicit override syntax only.

This supersedes older wording in prior architecture and handoff docs that presents Workflow as a peer route to Goal. Historical docs remain useful for current-state facts and file maps, but any route-first product framing is legacy.

### Goal is the durable autonomy contract

A Goal must capture:

- objective
- success criteria
- constraints
- budget or run policy
- approval policy
- interruption and resume policy
- current execution state
- selected strategy and why it was selected

The user should be able to ask for a result once, let the model work, and only be interrupted for real blockers, approvals, or ambiguity that justifies human confirmation.

### Workflow and Protocol are optional execution strategies

Workflow is not the user's default mental model.

Workflow or Protocol should be represented as described registry entries that the model can choose from, much like tools:

- a stable id
- a human-readable name
- a natural-language description
- when to use it
- when not to use it
- required capabilities and approval profile
- interrupt and resume behavior
- expected event shape and success signals

If the user explicitly specifies both a Goal and a Workflow or Protocol, that explicit choice wins.

If the user does not explicitly choose a Workflow or Protocol, the system should select one semantically and persist the choice plus rationale.

### Slash commands are overrides, not the main engine

The following remain valid:

- `/goal ...`
- `/workflow ...`
- `/chat ...`
- `/goal status`
- `/goal pause`
- `/goal resume`
- `/goal stop`

But the product must no longer depend on language-specific command-like phrasing to infer user intent. Hardcoded regex and keyword routing should be a bounded fallback only.

Hard invariant for route helpers:

- Web and TUI route helpers must treat every non-slash, non-UI-override natural-language message as ordinary `direct_chat`, regardless of language, duration, background-work phrasing, research intent, progress questions, steering language, or active Goal state.
- `/goal ...`, `/workflow ...`, `/chat ...`, explicit lifecycle commands, and explicit UI controls are the only frontend/TUI routing inputs that may leave `direct_chat`.
- After `/goal` starts or binds an autonomy contract, follow-up natural language is still model/backend input. Frontend and TUI must not classify it with phrase lists into "question", "explanation", "steer", "replan", "guidance", or "progress" lanes.
- Pending Goal proposal confirmation/revision is a temporary draft state, not general natural-language Goal intent detection.

### Goal must preserve model intent during follow-up

Inside an active Goal, the user may still be:

- asking a question
- asking for an explanation
- revising scope
- steering execution
- giving lifecycle control
- exiting back to ordinary chat

Only explicit lifecycle and safety controls should bypass model interpretation.

### Goal and normal chat should differ only by autonomy contract

The main difference between normal chat and Goal should be:

- Goal persists objective and runtime state
- Goal keeps running until done or truly blocked
- Goal can resume across time and attempts

The model should still think, explain, ask clarifying questions, use tools, and interact naturally.

### Shared chat semantics invariant

Goal turns must use the same base conversational understanding path as ordinary chat. Goal adds context and autonomy state; it must not replace conversation with command routing.

Any special Goal behavior must be justified by one of:

- active autonomy contract
- persisted Goal or AgentRun state
- selected strategy requirements
- approval, blocker, or safety policy
- explicit user override

## Current Misalignments To Remove

These are now treated as architectural problems, not small polish issues:

- `web/src/lib/chat-goal-routing.ts` still uses hardcoded English and Chinese regex as a primary route selector.
- `mochi/terminal_goal_helpers.py` also carries natural-language Goal regex and must not remain a parallel source of truth.
- `web/src/lib/goal-proposal-copy.ts` still teaches the user that `/goal` and `/workflow` are the main way to start work.
- `mochi/agents/multi_agent/protocols.py` still lets `parse_protocol_config(None)` silently default to `TeacherStudentDistillProtocol()`.
- `web/src/app/page.tsx` still inserts optimistic local assistant placeholders before some local goal-routing flows, which can leave meaningless cards.
- blocked-state and forwarded-state copy still has too much template-first behavior.
- active goal continuation is still split between frontend heuristics and runtime health in a way that makes intent understanding brittle.

## Target Architecture

### 0. Authoritative Intent And Strategy Decision Contract

Backend owns natural-language intent and strategy decisions. Frontend and TUI route helpers may parse explicit syntax and UI overrides, but they must not infer ordinary-language Goal creation, active Goal follow-up intent, or protocol choice as the authoritative source of truth.

The selector must receive:

- latest user message
- explicit slash command or UI override, if any
- active Goal summary and health, if any
- pending Goal proposal, if any
- recent conversation summary or bounded recent turns
- available strategy registry entries and selection guidance
- relevant workspace, project, attachment, and approval context

The selector must return a typed decision:

```ts
type GoalTurnDecision = {
  lane:
    | 'chat'
    | 'goal_start'
    | 'active_goal_turn'
    | 'goal_lifecycle_control'
    | 'clarify'
  active_goal_turn_kind?:
    | 'answer_question'
    | 'explain_goal_state'
    | 'steer'
    | 'replan'
    | 'lifecycle'
    | 'exit_to_chat'
    | 'clarify'
  strategy_id?: string | null
  protocol_id?: string | null
  override_source: 'slash_command' | 'ui_control' | 'none'
  selection_source:
    | 'explicit_override'
    | 'semantic_registry_selector'
    | 'bounded_fallback'
    | 'safe_default'
    | 'legacy_migration'
  selection_reason: string
  confidence: number
  requires_confirmation: boolean
  fallback_reason?: string | null
}
```

Rules:

- Explicit lifecycle and safety controls bypass semantic interpretation only for the lifecycle or safety action.
- `/goal`, `/workflow`, `/chat`, and UI controls set `override_source`, but ordinary-language strategy selection still belongs to the backend selector unless the user explicitly forced a strategy.
- Regex fallback may classify syntactically explicit commands and safe local UI states.
- Regex fallback may not choose a protocol or infer Goal intent from ordinary language except as an observable low-confidence `bounded_fallback` requiring confirmation.
- Selector input must include strategy registry descriptions, not only ids.
- Selector output must persist `selection_source`, `selection_reason`, `strategy_id`, and `protocol_id` when a Goal is created, resumed with reselection, or migrated.

### 1. Goal Contract Layer

Backend-owned Goal state must include, at minimum:

- `goal_id`
- `objective`
- `success_criteria`
- `constraints`
- `approval_policy`
- `run_policy`
- `selected_strategy_id`
- `selected_protocol_id`
- `selection_source`
  - `explicit_override`
  - `semantic_registry_selector`
  - `bounded_fallback`
  - `safe_default`
  - `legacy_migration`
- `selection_reason`
- `active_agent_run_id`
- `conversation_mode`
  - `chat`
  - `goal`
- `autonomy_state`
  - `draft`
  - `running`
  - `waiting_approval`
  - `blocked`
  - `paused`
  - `completed`
  - `failed`

### 2. Strategy Registry Layer

Introduce a backend-owned Goal strategy registry. Each entry should include:

- `id`
- `kind`
  - `protocol`
  - `workflow_template`
  - `execution_strategy`
- `name`
- `description`
- `when_to_use`
- `when_not_to_use`
- `execution_shape`
- `required_capabilities`
- `approval_profile`
- `control_scope`
- `interrupt_policy`
- `resume_policy`
- `event_contract`
- `success_signals`
- `failure_modes`
- `fallback_strategy_ids`
- `deprecated`

Do not make `trigger_keywords`, `confirmation_phrases`, or `default_route` core registry fields.

Every registry entry must include enough natural-language selection guidance for the selector to choose it without adding a route keyword. Adding a new strategy must not require editing frontend routing code.

### 3. Intent and Strategy Selection Pipeline

The corrected pipeline should be:

1. Parse explicit overrides and lifecycle commands.
2. If the user explicitly forces chat, honor it.
3. If the user explicitly requests Goal creation, create a Goal draft or start path.
4. If the user explicitly chooses a Workflow or Protocol, persist that override.
5. Otherwise, use a bounded semantic selection path to decide:
   - ordinary chat
   - Goal creation
   - active Goal follow-up
   - strategy selection for the Goal
6. Persist both the selected strategy and the reason it was selected.
7. If no stronger strategy is justified, default ordinary Goal execution to `autonomous_single_agent`, not `teacher_student_distill`.

### 3.1 Active Goal Turn Decision Model

Active Goal follow-up must produce a typed decision before any runtime mutation:

```ts
type ActiveGoalTurnDecision =
  | { kind: 'answer_question'; answer_mode: 'goal_state_explanation' | 'general_explanation' }
  | { kind: 'steer'; instruction: string; preserves_attempt_lineage: true }
  | { kind: 'replan'; requested_change: string; preserves_goal_contract: boolean }
  | { kind: 'lifecycle'; action: 'status' | 'pause' | 'resume' | 'cancel' | 'restart' }
  | { kind: 'exit_to_chat' }
  | { kind: 'clarify'; question: string }
```

Question-only and explanation-only turns must not enqueue AgentRun guidance, resume a run, or restart an attempt. Steering and replanning turns must preserve Goal lineage unless the user explicitly asks for a new Goal.

### 3.2 Defaulting Matrix

No-override Goal behavior must be locked across every layer:

| Layer | No override expected result |
| --- | --- |
| frontend route payload | no protocol forced from ordinary language |
| TUI route payload | no protocol forced from ordinary language |
| API create | selected strategy defaults to `autonomous_single_agent` |
| runtime model | persisted `strategy_id`, `protocol_id`, `selection_source`, and `selection_reason` |
| protocol parser | no implicit teacher/student when protocol payload is missing |
| resume/restart | preserves recorded strategy, or reselects with explicit `selection_source` |
| legacy migration | maps missing strategy through an explicit `legacy_migration` rule |
| workflow panel | shows selected strategy; does not make workflow default by panel presence |

### 4. Event-Lane UI Model

Borrow the Hermes split directly:

- durable assistant conversation
- live thinking stream
- tool activity stream
- approval and blocker notices
- subagent activity
- execution milestones

The main Goal surface should show meaningful progress, not protocol spam.

Projected execution UI must be reconstructable from persisted runtime events after reload. Durable chat messages and transient execution rows are separate products with separate lifetimes.

## Blocking Cutover Tasks

These are blocking cutover tasks, not optional polish. They must land before proposal-copy cleanup, panel cleanup, or route-adjacent UI refinement.

1. Backend schema/defaulting cutover
   - Introduce first-class persisted strategy fields such as `strategy_id`, `selection_source`, and `selection_reason`.
   - Remove implicit fallback chains that still default unspecified Goal execution back into `teacher_student_distill`.
   - Remove `parse_protocol_config(None) -> TeacherStudentDistillProtocol()` as an unobservable default path.
2. Proposal path must not auto-start
   - A local proposal path may create or update a draft or pending proposal only.
   - Proposal creation must not immediately `createGoal` plus `startGoal` as one hidden step.
   - Only explicit confirmation or a backend selector result with `lane=goal_start` and `requires_confirmation=false` may start execution.
3. Web and TUI route de-authoritization
   - Frontend and TUI may parse explicit slash commands and UI overrides only.
   - Ordinary-language Goal intent, active-goal follow-up interpretation, and strategy or protocol choice must move to the backend selector.
4. Backend `ActiveGoalTurnDecision`
   - Active Goal question, explanation, steer, replan, lifecycle, clarify, and exit decisions must be typed at the backend before runtime mutation.
   - Frontend may not preserve this behavior by degrading question-only turns to plain direct chat as the primary contract.
5. Durable/transient event source contract
   - Runtime and transcript payloads must expose explicit event-lane metadata such as `visibility`, `durability`, `projection_lane`, and a stable `event_id` or `dedupe_key`.
   - Frontend must not rely on fragile heuristics such as `acknowledgement` flags alone to decide what belongs in durable chat.
6. WorkflowPanel product positioning
   - Treat WorkflowPanel as an advanced/operator inspection and override surface, not the main end-user path for Goal reasoning.
   - The product decision should be explicit before further panel cleanup.
7. Expanded legacy migration mapping
   - Migration must cover legacy `interaction_mode`, `execution_mode`, `protocol_id`, `protocol_selection`, `selection_rationale`, and route-derived defaults such as `default_route`, not only one protocol field.

## Implementation Workstreams

## Workstream 1: Replace Hardcoded Goal Routing With Semantic Intent Selection

Owner: routing and intent contract

Files:

- `web/src/lib/chat-goal-routing.ts`
- `web/src/app/page.tsx`
- `mochi/terminal_goal_helpers.py`
- `mochi/goal_intent.py`
- `mochi/api/routes/goals.py`
- `tests/test_api_chat_models.py`
- `tests/test_goal_api.py`
- `web/scripts/test-chat-goal-routing.mjs`
- `web/scripts/test-chat-goal-workflow-routing.mjs`

Tasks:

- Demote `isNaturalLanguageGoalRequest(...)` and similar regex logic to bounded fallback behavior only.
- Introduce a semantic decision contract for:
  - direct chat
  - goal draft or start
  - active goal follow-up
  - goal lifecycle control
  - workflow or protocol override
- Keep explicit slash lifecycle commands deterministic.
- Stop treating active-goal follow-up as a single mechanical `goal_follow_up` lane.
- Add backend selector tests before removing frontend/TUI route authority.
- Ensure selector prompt/input includes strategy registry descriptions.
- Treat web and TUI de-authoritization as blocking cutover work, not follow-up cleanup.

Acceptance criteria:

- The system does not require language-specific route rules to decide whether a request should become a Goal.
- Chinese, English, Spanish, and Hindi fixtures can enter the same semantic decision path without adding route-specific regex for each language.
- Every non-slash, non-UI-override natural-language fixture resolves to `direct_chat` in `web/src/lib/chat-goal-routing.ts`, both with and without an active Goal.
- Active Goal progress questions, blocked-state explanation questions, steering instructions, replanning requests, and research-duration requests are not frontend/TUI route kinds; they remain ordinary chat input for the backend/model Goal runtime.
- Explicit lifecycle commands still behave deterministically.
- `web/src/lib/chat-goal-routing.ts` and `mochi/terminal_goal_helpers.py` parse explicit commands and UI overrides only.
- Natural-language Goal/chat/strategy decisions come from the backend selector.
- Explicit slash commands set `selection_source = explicit_override`.
- Non-slash natural language sets `selection_source = semantic_registry_selector`, `safe_default`, or observable `bounded_fallback`.
- `bounded_fallback` cannot choose a protocol from ordinary language without confirmation.
- Frontend and TUI no longer act as the authoritative source for ordinary-language Goal or strategy decisions.

## Workstream 2: Introduce Goal Strategy Registry and Correct Protocol Defaulting

Owner: backend strategy model and API contract

Files:

- Create: `mochi/runtime/goal_strategy_registry.py`
- Modify: `mochi/agents/multi_agent/protocols.py`
- Modify: `mochi/runtime/models.py`
- Modify: `mochi/api/routes/goals.py`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/components/chat/WorkflowPanel.tsx`

Tasks:

- Treat backend schema/defaulting cutover as the first blocking subsection of this workstream.
- Add a backend-owned registry describing workflow and protocol options in natural language.
- Expose registry data to the frontend for inspection and explicit user override.
- Make the semantic selector consume registry entries directly.
- Remove the silent `None -> TeacherStudentDistillProtocol()` behavior.
- Make ordinary Goal default selection land on `autonomous_single_agent` unless the user or semantic selector chooses otherwise.
- Persist `strategy_id`, `protocol_id`, `selection_source`, and `selection_reason` on Goal creation and updates.
- Add a test-only registry entry to prove selection is registry-driven, not keyword-driven.
- Apply the defaulting matrix across create, start, resume, restart, parser, frontend, TUI, and legacy-load paths.
- Ensure old `execution_mode=workflow` or missing strategy fields do not silently reintroduce distillation through runtime fallback paths.

Acceptance criteria:

- An unspecified Goal no longer accidentally starts in a distillation protocol.
- Workflow and protocol options are described objects, not just ids hidden behind panel controls.
- The UI can show what strategy was selected and why.
- A new test-only strategy can be registered with a unique natural-language description and selected without editing frontend routing or backend keyword rules.
- Selection evidence records the registry entry id and reason text.
- No production code outside explicit override parsing switches on protocol ids to infer user intent.
- Explicit `/workflow` creates or updates a Goal with a workflow strategy override; it does not create a separate top-level Workflow object.
- Runtime persistence stores first-class strategy-selection metadata instead of relying only on route-shaped `protocol_selection` copy.

## Workstream 3: Reframe Goal Proposal UX Around Contract and Strategy

Owner: goal proposal and launch UX

Files:

- `web/src/lib/goal-proposal-copy.ts`
- `mochi/goal_proposal_copy.py`
- `web/src/app/page.tsx`
- `web/src/components/chat/GoalHeaderChip.tsx`

Tasks:

- Rewrite proposal copy so Goal is the main object and Workflow is an optional strategy choice.
- Stop teaching `/workflow <request>` as the primary workflow entrypoint.
- Show selected strategy, why it was chosen, and how the user can override it.
- Keep slash commands available as explicit overrides, not required ceremony.
- Make proposal-path non-auto-start behavior explicit in both product copy and implementation contract.
- Ensure local proposal handling creates or updates a draft or pending proposal only, unless the backend explicitly authorizes immediate start without confirmation.

Acceptance criteria:

- Goal proposal copy reads like "here is the task contract and chosen strategy", not "pick the right route".
- The user can accept the proposal without memorizing command syntax.
- If the user explicitly wants a workflow or protocol, the UI shows that override clearly.
- Proposal creation does not immediately start execution unless the backend selector explicitly returns a start decision that does not require confirmation.

## Workstream 4: Restore Conversational Goal Follow-Up

Owner: active Goal continuation behavior

Files:

- `web/src/lib/chat-goal-continuation.ts`
- `web/src/app/page.tsx`
- `mochi/api/routes/goals.py`
- `mochi/goal_proposal_copy.py`

Tasks:

- Preserve model-mediated interpretation for active Goal follow-up.
- Introduce a backend-owned `ActiveGoalTurnDecision` boundary before any resume, restart, or guidance append path.
- Distinguish:
  - question or explanation
  - revision or replan
  - steering guidance
  - lifecycle command
  - exit back to chat
- Use model-generated explanation paths when the user is asking for understanding, not only when issuing control.
- Keep attempt-lineage semantics for resume, steer, restart, and wake-after-resolution.
- Remove the current fallback shape where frontend preserves behavior by degrading question-only goal turns into direct chat.

Acceptance criteria:

- Asking "what does this blocked state mean?" yields a contextual answer instead of a canned forwarded message.
- Asking for progress behaves like normal conversation.
- Steering instructions can resume or redirect work without forcing the user to restate the entire Goal.
- Question-only turns do not append AgentRun guidance, resume a run, or restart an attempt.
- Steering turns append a lineage-preserving control or guidance event to the active Goal attempt.
- Ambiguous active-goal turns ask a clarifying question instead of blindly forwarding guidance.
- Mixed-language follow-up remains in the same semantic path.
- Question or explanation handling is decided by backend `ActiveGoalTurnDecision`, not by frontend route downgrade heuristics.

## Workstream 5: Make Blocked and Approval States Actionable

Owner: blocker diagnostics and operator UX

Files:

- `mochi/runtime/models.py`
- `mochi/runtime/service.py`
- `mochi/runtime/execution_transcript.py`
- `mochi/api/routes/goals.py`
- `web/src/lib/goal-proposal-copy.ts`
- `mochi/goal_proposal_copy.py`
- `web/src/app/page.tsx`
- `tests/test_goal_api.py`

Tasks:

- Replace generic blocked copy with structured explanation fields:
  - what is blocked
  - why it is blocked
  - which approval, tool, domain, or policy is involved
  - whether auto-resume will happen
  - what the user must do next
- Define `BlockerDiagnostic` and `ApprovalDiagnostic` fields:
  - `cause_code`
  - `what_is_blocked`
  - `why_blocked`
  - `actor_required`
  - `next_action`
  - `auto_resume_policy`
  - `source_event_ids`
- If auto-review is enabled but approval is still required, explain why.
- Remove "operator must handle this before continuing" as a sufficient final answer.

Acceptance criteria:

- Blocked state is understandable without opening raw event logs.
- The user can tell whether they need to approve, resume, or simply wait.
- Auto-review edge cases are explained explicitly.
- API tests inspect structured diagnostic JSON, not only rendered copy.
- "Operator must handle this" is not valid final copy unless paired with a concrete `next_action`.
- Diagnostics are derived from runtime state or source events, not model-only inference.

## Workstream 6: Show Live Execution and Thinking Without Polluting Durable Chat

Owner: execution projection and chat UX

Files:

- `web/src/lib/execution-transcript.ts`
- `web/src/components/chat/ExecutionTimeline.tsx`
- `web/src/app/page.tsx`
- `web/src/lib/chat-projections.ts`

Tasks:

- Remove empty or meaningless optimistic assistant cards from local goal-routing flows.
- Keep durable assistant turns separate from transient execution lanes.
- Show live reasoning, tool use, approvals, and subagent activity as typed execution UI, not as blank cards or repeated protocol rows.
- Preserve the display-only boundary for projected runtime UI.
- Add an explicit runtime event source contract for `visibility`, `durability`, `projection_lane`, and stable `event_id` or `dedupe_key` semantics.
- Give every projected execution row a stable event id or deterministic dedupe key.
- Deduplicate protocol identity rows by run, protocol, and status.
- Reject or avoid empty durable assistant messages at the message-creation boundary.

Acceptance criteria:

- Goal start shows visible execution progress quickly.
- The main transcript is readable.
- The user can watch work happening without the chat filling with junk cards.
- No assistant chat message is created with empty or whitespace content from local Goal routing.
- Reload reconstructs the execution lane without duplicate rows.
- Thinking, tool, approval, subagent, and milestone rows render in execution lanes, not as canonical assistant transcript content.
- Frontend lane assignment no longer depends on `acknowledgement`-style heuristics alone.

## Workstream 7: Realign Workflow Panel and Advanced Controls With Selected Strategy

Owner: strategy-specific UI surfaces

Files:

- `web/src/components/chat/WorkflowPanel.tsx`
- `web/src/app/page.tsx`
- `web/src/lib/api.ts`

Tasks:

- Make WorkflowPanel product positioning explicit: advanced/operator inspection plus override surface, not the main end-user Goal path.
- Stop keying workflow-native UI primarily off slash-route history.
- Show advanced workflow controls only when the selected strategy actually needs them.
- Let a single-agent Goal remain chat-first even if workflow settings exist in the session.
- Make advanced controls an inspection and override surface, not the main way the user must think.

Acceptance criteria:

- WorkflowPanel no longer acts as the primary product surface for ordinary Goal routing or strategy choice.
- Workflow controls appear because the strategy requires them, not because the route happened to be `/workflow`.
- Single-agent Goals stay simple.
- Advanced users can still inspect and override strategy details.

## Workstream 8: Lock Behavior With Regression and Live E2E Verification

Owner: end-to-end quality

Files:

- `tests/test_goal_api.py`
- `tests/test_api_chat_models.py`
- `web/scripts/test-chat-goal-routing.mjs`
- `web/scripts/test-chat-goal-workflow-routing.mjs`
- live browser verification notes or artifacts

Minimum scenarios:

- non-slash natural-language request in Chinese resolves to frontend/TUI `direct_chat`, including timed research phrasing
- non-slash natural-language request in English resolves to frontend/TUI `direct_chat`, including timed/background-work phrasing
- non-slash natural-language request in Spanish resolves to frontend/TUI `direct_chat`
- non-slash natural-language request in Hindi resolves to frontend/TUI `direct_chat`
- explicit `/goal ...`
- explicit workflow or protocol override
- active Goal progress question remains frontend/TUI `direct_chat` and is interpreted only by backend/model Goal runtime
- active Goal blocked-state explanation question remains frontend/TUI `direct_chat` and is interpreted only by backend/model Goal runtime
- active Goal steering instruction remains frontend/TUI `direct_chat`; lineage preservation is backend/runtime responsibility
- active Goal ambiguous follow-up remains frontend/TUI `direct_chat`; clarification is a backend/model decision
- blocked approval explanation
- auto-review enabled but approval still required
- no empty assistant cards during local goal-handled flows
- visible live thinking or execution progress after Goal start
- reload or reconnect after Goal start without duplicate timeline rows

Acceptance criteria:

- Goal behaves like autonomous chat, not a command router.
- Strategy selection is registry-backed and explainable.
- Follow-up inside Goal remains conversational.
- Blocked state is actionable.
- UI residue from local routing is gone.
- Adding a registry fixture can change strategy selection without route code changes.
- Active Goal questions do not enqueue runtime work.
- Active Goal steering does enqueue lineage-preserving guidance or control.

## Workstream 9: Migrate Legacy Goals And Route-Derived State

Owner: compatibility and migration behavior

Files:

- `mochi/runtime/store.py`
- `mochi/runtime/service.py`
- `mochi/runtime/models.py`
- `tests/test_runtime_store.py`
- `tests/test_goal_api.py`
- `web/src/lib/api.ts`
- `web/src/components/chat/WorkflowPanel.tsx`

Tasks:

- Define migration rules for existing Goals with missing strategy fields.
- Mark backfilled selections with `selection_source = legacy_migration`.
- Preserve explicit historical workflow or protocol choices.
- Avoid silently rewriting explicit `teacher_student_distill` history as `autonomous_single_agent`.
- Map missing single-agent Goal protocol to `autonomous_single_agent`.
- Migrate legacy `interaction_mode`, `execution_mode`, `protocol_id`, `protocol_selection`, `selection_rationale`, and route-derived defaults such as `default_route` into the new strategy contract.
- Project old AgentRun events through the new execution-lane dedupe rules without rewriting history.

Acceptance criteria:

- Legacy Goals load with explicit strategy metadata.
- Historical explicit workflow/protocol selections remain inspectable.
- Missing strategy does not produce a hidden distillation default.
- UI explains legacy selection source when strategy was migrated.
- Migration tests cover old records with missing protocol, explicit workflow protocol, and route-derived session state.
- Migration tests also cover legacy `interaction_mode`, `execution_mode`, `protocol_selection`, `selection_rationale`, and `default_route` combinations.

## Workstream 10: Extract Page-Level Goal Controllers

Owner: frontend architecture and maintainability

Files:

- Create: `web/src/lib/goal-turn-controller.ts`
- Create: `web/src/lib/goal-strategy-selection.ts`
- Create: `web/src/lib/goal-execution-projection-store.ts`
- Modify: `web/src/app/page.tsx`
- Test: `web/scripts/test-chat-goal-routing.mjs`
- Test: `web/scripts/test-chat-goal-workflow-routing.mjs`

Tasks:

- Move active Goal turn handling out of `page.tsx`.
- Move selected-strategy display and override mapping out of `page.tsx`.
- Move durable-vs-transient execution projection rules out of `page.tsx`.
- Keep `page.tsx` as UI wiring, not the product decision owner.

Acceptance criteria:

- `page.tsx` does not contain natural-language Goal intent rules.
- Strategy display consumes backend selector or registry data.
- Execution projection is testable without rendering the whole page.
- The extraction preserves existing session and runtime bindings.

## Execution Order

0. Land the backend schema/defaulting cutover.
1. Stop proposal paths from auto-starting execution.
2. Add the authoritative intent/strategy decision contract and invariant tests.
3. Replace hardcoded routing with backend semantic intent selection across web and TUI.
4. Introduce backend `ActiveGoalTurnDecision` and restore conversational active Goal follow-up.
5. Introduce strategy registry and correct end-to-end defaulting.
6. Fix blocked and approval diagnostics at the runtime/API layer.
7. Clean execution projection and live thinking surfaces with an explicit event source contract.
8. Reframe Goal proposal UX around contract plus strategy.
9. Realign workflow panel with selected strategy and advanced/operator positioning.
10. Migrate legacy Goals and route-derived state.
11. Extract page-level Goal controllers.
12. Re-run live end-to-end verification.

This order is deliberate:

- first remove hidden backend defaults that would invalidate later selector work
- then stop proposal auto-start so draft and confirmation semantics become true
- then lock the executable contract
- then remove route authority from frontend and TUI
- then formalize active-goal turn decisions
- then make strategy selection registry-backed
- then fix runtime diagnostics and event-lane projection
- then clean UI projection and controls

Making the current route-driven model prettier before changing the contract would be wasted work.

## Risks And Open Questions

- `web/src/app/page.tsx` remains a hotspot and needs extraction as part of stabilization, not after all behavior changes.
- Backend-owned strategy selection must not drift from frontend inspection copy.
- Migration is needed for existing Goals that already carry legacy `interaction_mode`, `protocol_id`, or route-derived assumptions, and is now tracked as Workstream 9.
- Some blocked-state explanations will still need bounded model-generated copy, with strong fallback rules.
- The first selector implementation may use a bounded model call, but tests must prove fallback boundaries and registry consumption.
- Backend and transcript schemas will need a compatibility period because strategy metadata and event-lane metadata are both expanding at once.
- Proposal-path non-auto-start semantics may temporarily expose mismatches in current local chat UX until proposal and confirmation surfaces are fully aligned.

## Definition Of Done

This plan is complete only when all of the following are true:

- Goal behaves like a Codex-style autonomous task surface.
- The model preserves intent understanding inside active Goals.
- Goal and Workflow are separate concepts.
- Workflow and Protocol are selected from described registry entries or explicit user override.
- Slash commands are optional overrides, not the main reasoning path.
- Unspecified Goal execution no longer defaults to distillation.
- Blocked and approval states are contextual and actionable.
- The Goal surface shows live, meaningful execution progress without junk cards.
- The only durable product difference between normal chat and Goal is the autonomy contract and persisted runtime state.
- Frontend and TUI no longer make ordinary-language Goal or protocol decisions with regex.
- Selector decisions persist strategy id, protocol id, selection source, and selection reason.
- Proposal creation no longer auto-starts execution unless the backend emits an explicit no-confirmation start decision.
- Active Goal question-only turns do not mutate runtime state.
- Active Goal question and explanation turns are mediated by backend typed decisions, not frontend downgrade heuristics.
- Runtime events expose enough source metadata to keep durable conversation separate from transient execution lanes.
- Legacy Goals load through explicit migration rules, not hidden defaults.
