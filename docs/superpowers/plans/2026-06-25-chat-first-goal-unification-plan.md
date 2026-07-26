# Chat-First Goal Unification Plan

Date: 2026-06-25
Status: draft
Scope baseline: unify long-running chat execution around a single user-facing Goal model, treat Workflow as one execution strategy under Goal rather than the mandatory path, and keep advanced goal/run pages as operator consoles
Primary memory references:
- `.claude/skills/agent-memory/memories/project-status/goal-supervisor-and-long-running-autonomy-2026-06-21.md`
- `documents/prd/mochi_workflow_desk_subagent_chat_plan.md`

## Summary

This plan unifies long-running execution around a single user-facing concept: `Goal`.

The target product model is:

- `Goal` = user objective plus durable execution contract, including execution mode
- `Workflow` = one execution strategy available under a goal, not the default or mandatory path
- `Agent Run` = one concrete execution attempt under the goal
- `Operator Console` = advanced inspection and intervention surface, not the primary entrypoint

The target UX is chat-first across WebGUI, CLI, and TUI:

- users can start long-running work by natural language or `/goal`
- the main agent proposes execution mode, models, roles, protocol when relevant, and duration before starting
- confirmation performs `create + start` in one step
- follow-up messages continue the active goal by default
- `/workflow` remains only as an expert override, not a parallel everyday entrypoint

## Decisions Locked

### User-facing model

- `Goal` is the only normal user-facing long-task concept.
- `Workflow` is no longer a parallel first-class concept in normal UX.
- `Goal` owns the execution contract and may or may not use workflow semantics.
- `Goal` should explicitly carry an `execution_mode`.
- First-wave execution modes should distinguish at least:
  - `single_agent`
  - `workflow`
  - `collector_or_daemon`
- `Workflow` survives only as the execution strategy used when `execution_mode=workflow`.
- `Agent Run` remains the execution primitive and attempt record.

### Entry and routing

- Main entry is natural language plus `/goal`.
- `/workflow` remains available only for expert overrides and should force `execution_mode=workflow` before proposal or start.
- `/chat` explicitly routes outside the active goal path for a turn.
- A chat/session can bind to at most one active goal at a time.
- Once a chat has an active goal, plain follow-up user messages default to that goal unless the user explicitly escapes with `/chat`.

### First-wave command surface

- First-wave `/goal` command surface should be intentionally narrow:
  - `/goal`
  - `/goal status`
  - `/goal pause`
  - `/goal resume`
  - `/goal stop`
- `/goal` with no body should not create an empty goal.
- `/goal` with no body should show either:
  - the active goal summary and available actions, or
  - a short usage/help preview when there is no active goal

### Proposal and startup

- The main agent should promote ordinary chat into a goal proposal only when the user clearly asks for background progress, a duration window, repeated retries, checkpointed work, or "keep working on this" behavior.
- The default proposal format should stay concise:
  - objective summary
  - suggested execution mode
  - suggested protocol
  - suggested models
  - suggested roles
  - suggested duration or runtime mode
  - approval or risk note when relevant
- If the user says "you decide", the main agent may auto-select models, roles, and protocol.
- The agent must still show a short execution summary before starting.
- User confirmation should perform `create + start` in the same step. No separate page-level start action is required.

### Proposal and history persistence

- Goal proposal cards should be preserved as visible chat history.
- If the user revises a proposal in natural language, those revisions should also remain visible in chat history.
- Proposal persistence must survive refresh and reload.
- Proposal history should make it clear:
  - what was proposed
  - what the user changed
  - what configuration was ultimately approved
- First-wave proposal cards should be first-class structured message cards rather than plain markdown text blobs.
- First-wave proposal cards should show:
  - title: `Goal proposal`
  - objective summary
  - execution mode
  - protocol or workflow strategy
  - 1-3 primary models
  - role summary
  - runtime mode or duration
  - optional risk note for approval, cost, network, or shell concerns
- WebGUI may show `Start`, `Revise`, and `Cancel` actions on the proposal card, but the proposal flow must not depend on GUI-only controls.
- CLI and TUI must support natural-language equivalents for:
  - start
  - revise duration
  - revise models
  - revise protocol
  - cancel
- Proposal revisions should not overwrite the original card in place.
- Each substantial revision should append a new `Revised goal proposal` card.
- Earlier proposal cards should remain visible and be marked as superseded when replaced by a newer revision.
- After confirmation, chat history should append a `Goal started` summary card showing the final approved configuration.

### Model discovery

- When the user asks which models are available, the main agent should read the configured or known model catalog first.
- The main agent should then run targeted availability probes only for realistic candidates.
- Proposal quality should prefer actually usable models, not only configured models.

### Surface hierarchy

- Chat is the primary goal-creation and steering surface.
- `/goals` remains, but becomes an operator or supervisor console only.
- `/agent-runs` remains, but becomes an advanced run or execution desk only.
- Sidebar navigation should stop treating chat, goals, and runs as equal everyday entrypoints.

## Product Contract

### Goal-aware chat behavior

- Add session-level active-goal binding semantics.
- The chat header should show a compact goal chip with:
  - goal title
  - goal status
  - execution mode
  - selected protocol when relevant
  - selected model count
  - duration or runtime mode
  - blocker or approval count when relevant
- Users should not need to leave chat to create, continue, or steer a goal.
- Clicking the goal chip should open a lightweight in-chat goal drawer, not immediately navigate away.
- Do not show the header goal chip during proposal-only states.
- Show the goal chip only when the chat is in:
  - `goal_active`
  - `goal_blocked`
  - `goal_completed`

### Slash command semantics

- `/goal <request>` starts explicit goal intent handling.
- `/chat <request>` bypasses active-goal routing for that turn.
- `/workflow <request>` forces expert workflow-mode handling before proposal or start.

### Follow-up execution semantics

- Replace the current workflow-only follow-up behavior for long-running chat.
- For goal-bound chats:
  - if the linked run is active, the user message becomes goal or run guidance
  - if the linked run is paused, stalled, or awaiting resources and resume is safe, the supervisor resumes the same run or attempt
  - if the linked run completed but the goal remains incomplete, the supervisor can open the next linked attempt automatically when policy permits
- Goal-bound follow-up should not degrade into "append state only" behavior.
- While a goal is active, ordinary model-discovery questions such as "which models are available" should stay inside the active goal steering path by default.
- In that case, the agent may answer with candidate models and suggest a revised goal configuration or next-attempt configuration instead of jumping back to ordinary chat mode.
- `/chat` remains the explicit escape hatch for users who want to step outside the active goal context.

### Goal completion behavior

- When a goal completes, the chat should automatically return to normal chat mode.
- The UI should still preserve a clickable completed goal chip or completion summary after completion.
- That completed goal summary should remain visible until a newer goal replaces it or the session state explicitly clears it.

### Chat state machine

- First-wave chat state machine should be intentionally narrow:
  - `chat_idle`
  - `goal_proposal_pending`
  - `goal_proposal_revising`
  - `goal_active`
  - `goal_blocked`
  - `goal_completed`
  - `goal_cancelled_or_failed`
- `goal_blocked` is the user-facing umbrella for approval waits, paused runs, awaiting-resources, and stalled states in the first wave.
- Proposal states should be rendered inside chat history only, without a header chip.
- Active, blocked, and completed states should expose the header goal chip and lightweight drawer.

### Proposal heuristics

- Suggested role and protocol defaults:
  - compare or evaluate tasks -> debate plus judge
  - distillation or compression tasks -> teacher plus student plus evaluator
  - execution-heavy tasks -> planner plus executor plus controller plus evaluator
  - research-heavy tasks -> planner plus researchers plus synthesizer plus verifier
- The main agent can offer alternatives, but these defaults are the first-wave heuristic baseline.
- Suggested execution-mode defaults:
  - bounded long-running analysis or execution with no clear need for subagents -> `single_agent`
  - debate, distillation, planner-executor-evaluator, or explicit multi-role collaboration -> `workflow`
  - long-duration crawling, monitoring, or resumable collection -> `collector_or_daemon`

## Interfaces and Implementation Changes

### Session and frontend state

- Add a session-level active-goal binding contract, such as `SessionGoalState`, with at least:
  - `active_goal_id`
  - `active_goal_status`
  - `execution_mode`
  - `default_route`
  - `last_goal_summary`
- Add enough session UI state to reconstruct proposal history and proposal supersession state after reload.
- Extend that state to preserve the latest completed goal summary for header-chip continuity after goal completion.
- Replace or wrap current workflow-enabled frontend state so the canonical long-task path is goal-bound chat, not workflow mode.
- Existing stored workflow-bound sessions should map forward into the new goal-aware model during load.

### Chat parsing and routing

- Extend slash-command parsing to support `/goal`, `/chat`, and `/workflow` explicitly.
- Introduce a goal proposal controller in the main chat runtime that:
  - detects goal-worthy intent
  - chooses or proposes an execution mode
  - gathers model availability
  - constructs a concise proposal
  - waits for confirmation
  - creates and starts the goal in one step
- Introduce first-wave goal-specific chat message card types for:
  - goal proposal
  - revised goal proposal
  - goal started
  - goal completion summary

### Goal execution spec

- Keep protocol and workflow-specific configuration inside the goal execution spec, not as a separate top-level UX concept.
- The execution spec should minimally support:
  - `execution_mode`
  - `single_agent_spec` for primary model and tool profile when no workflow is needed
  - `workflow_spec` for `protocol_id`, `model_selection`, and `role_layout` when `execution_mode=workflow`
  - `collector_spec` for resumable collection or daemon-style tasks when applicable
  - `run_policy`
  - `approval_mode`
  - optional `allowed_tools`
- `/workflow` should populate or override this execution spec by forcing `execution_mode=workflow`, without creating a second user-visible mode.

### Continuation logic

- Add goal-aware continuation logic for active-goal chats:
  - active run -> append guidance
  - paused or stalled run -> resume same run when safe
  - completed run under incomplete goal -> create or start next attempt
- This logic should belong to goal continuation, not page-local workflow UI state.

### WebGUI changes

- Reposition `web/src/app/goals/page.tsx` as an operator console.
- Reposition `web/src/app/agent-runs/page.tsx` as an advanced run desk.
- Make chat the default place to create and steer goals.
- Add deep links from the chat goal chip to advanced console pages for inspection only.
- Replace the current header workflow entry with the goal chip as the primary long-task surface.
- Keep the goal drawer lightweight for status and lifecycle control rather than deep configuration.
- Only show workflow-native surfaces such as subagent tabs, stage map, or workflow desk when `execution_mode=workflow`.
- Do not force non-workflow goals through workflow-specific UI.

### CLI and TUI alignment

- The primary goal flow must work without WebGUI:
  - natural language request
  - `/goal`
  - `/chat`
  - `/workflow`
  - status checks
  - resume, pause, and stop
- CLI and TUI should expose active-goal summary inline in the conversation flow.

## Test Plan

### Conversation and routing

- `/goal <request>` enters proposal flow without requiring a page transition.
- A natural-language long-task request can create a goal with `execution_mode=single_agent` without entering workflow mode.
- `/goal` with no body shows active-goal summary or help instead of creating an empty goal.
- Plain follow-up messages route to the active goal by default.
- `/chat <request>` bypasses the active goal for one turn.
- `/workflow <request>` forces workflow-mode override behavior without becoming the default path.
- `/goal status`, `/goal pause`, `/goal resume`, and `/goal stop` work from CLI/TUI and chat surfaces with the same lifecycle semantics.
- While a goal is active, model-discovery follow-ups remain inside goal steering unless the user explicitly escapes with `/chat`.

### Proposal flow

- Natural-language long-task prompts trigger goal proposal flow.
- Ordinary short chat questions do not trigger goal proposal.
- "You decide" allows agent-side selection of models, roles, and protocol while still emitting a concise pre-start summary.
- Confirmation performs `create + start` in the same step.
- Proposal cards remain visible after reload.
- Natural-language modifications such as duration, execution mode, protocol, or model changes remain visible after reload.
- Revised proposals append new cards instead of mutating prior proposal cards in place.
- Starting a goal appends a `Goal started` summary card derived from the final approved revision.

### Model discovery

- "What models are available?" returns configured models plus probe-based availability refinement.
- Failed probes prevent obviously unavailable models from being presented as ready candidates.
- Role and protocol recommendations vary correctly by task type.

### Goal continuation

- Active-goal follow-up resumes or continues the current goal rather than falling back to plain chat.
- Paused or stalled linked runs resume on follow-up when policy allows.
- Completed linked runs under unfinished goals open the next attempt correctly.
- Legacy workflow-bound sessions still load and continue coherently.

### UI and navigation

- Main chat shows active-goal chip and compact runtime summary.
- Goal chip opens a lightweight goal drawer with summary, status, execution mode, duration, protocol when relevant, models, pause/resume/stop, and open-console action only.
- Workflow-specific panels are not shown for `single_agent` goals.
- `/goals` and `/agent-runs` remain reachable but are no longer required to start work.
- Sidebar no longer treats goals and runs as equal primary entrypoints for everyday use.
- Completed goals leave behind a visible completed-goal chip or summary until replaced.
- Proposal cards are shown inline in chat history and are not rendered as generic assistant markdown bubbles.

### CLI and TUI

- `/goal`, `/chat`, and `/workflow` work without WebGUI.
- Active-goal summary and basic goal controls are visible from CLI and TUI conversation flow.

## Assumptions and Defaults

- This is a unification and entry-model refactor, not a removal of the existing goal supervisor backend.
- Existing goal operator capabilities such as approvals, checkpoints, audit findings, estop, and shard retry remain in the advanced operator-console path.
- Existing workflow or protocol implementations remain valid and are reclassified as execution strategies available under goal execution modes, not as the mandatory path for every goal.
- Goal proposals remain concise by default and only expand when the user asks for detail or when the task has meaningful risk or ambiguity.
- If implemented incrementally, entry flow and routing changes should land before any broad removal or renaming of existing pages.
