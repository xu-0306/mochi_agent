# Goal Conversational Runtime Handoff

Date: 2026-07-03
Checkpoint commit: `b5be01b` (`Recover goal conversational runtime`)
Primary plan: `docs/superpowers/plans/2026-07-01-goal-conversational-intent-and-ux-recovery-plan.md`

## Purpose

This handoff explains the intended Goal behavior after the current recovery work. The main point is to prevent the next implementation pass from accidentally rebuilding the old command-card/router model.

Goal is not a card flow and Workflow is not a peer top-level route. Goal is the durable autonomy contract. Workflow, Protocol, and multi-agent execution are strategies under that Goal.

## Product Contract

The correct mental model is:

- Chat is the normal conversational surface.
- Goal is a durable autonomy contract plus persisted runtime state.
- Strategy is the selected execution approach under a Goal.
- Workflow is one possible strategy family under Goal.
- Protocol is an implementation detail for strategies that need multi-agent orchestration.
- Slash commands are explicit overrides, not the main reasoning path.

The user should be able to describe work naturally. The system should preserve that message for backend/model interpretation instead of forcing the user into `/goal` or `/workflow` command syntax.

## Correct End-To-End Flow

### 1. Normal Natural-Language Input

For non-slash, non-UI-control text, frontend and TUI route helpers must return ordinary chat/direct input.

This is intentional even when the message contains:

- duration phrases like "20 minutes"
- research/background-work wording
- Chinese or mixed-language task phrasing
- active Goal progress questions
- active Goal steering instructions
- blocked-state explanation questions

Frontend/TUI must not classify these with keyword lists. The backend/model runtime owns natural-language intent.

Important files:

- `web/src/lib/chat-goal-routing.ts`
- `mochi/terminal_goal_helpers.py`
- `web/scripts/test-chat-goal-routing.mjs`
- `web/scripts/test-chat-goal-workflow-routing.mjs`
- `tests/test_main_chat_tui.py`

### 2. Explicit Overrides

These remain deterministic:

- `/goal <request>` creates or updates a Goal proposal/draft.
- `/workflow <request>` is an explicit advanced strategy override under Goal.
- `/chat <request>` exits goal setup for ordinary chat.
- `/goal status`, `/goal pause`, `/goal resume`, `/goal stop` are lifecycle controls.

Do not make `/workflow` the primary instruction in user-facing copy. It may be mentioned only as an explicit strategy override.

### 3. Proposal/Draft Behavior

Proposal creation must not start execution.

Correct behavior:

1. Explicit `/goal ...` or `/workflow ...` creates/updates a pending proposal/draft in session state.
2. The UI may show a Goal card, but it should describe a Goal draft/contract and selected strategy.
3. No `createGoal` plus `startGoal` happens during proposal creation.
4. Execution starts only after explicit confirmation of the pending proposal, or a future backend selector result that explicitly says `lane=goal_start` and `requires_confirmation=false`.

Current frontend confirmation branch still creates then starts after pending proposal confirmation. That is acceptable for the current contract because the confirmation is explicit.

Important files:

- `web/src/app/page.tsx`
- `web/src/lib/goal-proposal-copy.ts`
- `mochi/goal_proposal_copy.py`
- `tests/test_goal_copy.py`
- `tests/test_goal_api.py`
- `tests/test_main_chat_tui.py`

### 4. Strategy Selection And Defaults

Goal strategy selection is backend-owned.

Current rules:

- Missing strategy defaults to `autonomous_single_agent`.
- Missing protocol must not silently become `teacher_student_distill`.
- `teacher_student_distill` is specialized, non-default, and confirmation-gated.
- Explicit `strategy_id` or `protocol_id` wins, if not conflicting.
- Conflicting `strategy_id` and `protocol_id` is rejected.
- Persisted Goals include strategy metadata:
  - `strategy_id`
  - `protocol_id`
  - `selection_source`
  - `selection_reason`

Important files:

- `mochi/runtime/goal_strategy_registry.py`
- `mochi/goal_intent.py`
- `mochi/runtime/models.py`
- `mochi/runtime/service.py`
- `mochi/runtime/store.py`
- `mochi/agents/multi_agent/protocols.py`
- `web/src/lib/api.ts`
- `web/src/components/chat/WorkflowPanel.tsx`

### 5. Active Goal Follow-Up

The route helper still returns `direct_chat` for active Goal natural language. That is correct.

The frontend send path now adds a backend decision boundary before runtime mutation when:

- the route is `direct_chat`
- the session has an active non-terminal Goal
- there is no pending proposal
- the message is non-empty natural language

The frontend calls:

```ts
api.fetchGoalTurnDecision(activeGoalId, { message: requestText })
```

Then:

- `answer_question` and `explain_goal_state` stay conversational and do not mutate Goal runtime.
- `exit_to_chat` stays conversational.
- `clarify` with confirmation required stays conversational/non-mutating.
- `steer` and `replan` may mutate runtime through existing continuation paths, but only after the backend decision.

For steering/replanning:

- Fetch Goal health inside guarded error handling.
- Resolve continuation behavior with `resolveGoalContinuationDecision`.
- Forward guidance with `appendAgentRunGuidance` when a live run exists.
- Use `resumeGoal(... guidanceMessage: requestText)` when resuming/restarting is the correct path.
- For `refresh_then_forward`, refresh first, then forward `requestText`; if no refreshed run id is available, fall back to `resumeGoal(... guidanceMessage: requestText)`.
- Never persist success copy if mutation fails.

Important files:

- `mochi/runtime/models.py`
- `mochi/runtime/service.py`
- `mochi/api/routes/goals.py`
- `web/src/lib/api.ts`
- `web/src/lib/chat-goal-continuation.ts`
- `web/src/app/page.tsx`
- `web/scripts/test-chat-active-goal-turn-decision.mjs`
- `tests/test_goal_api.py`

## Completed In Current Checkpoint

The checkpoint commit includes these completed slices:

- Frontend/TUI route de-authoritization for natural-language Goal intent.
- Protocol default cleanup so missing protocol no longer means distillation.
- Backend Goal strategy registry.
- Strategy-selection persistence and API/frontend registry consumption.
- Proposal UX/copy cleanup around Goal draft/contract and selected strategy.
- Active Goal turn decision endpoint and first frontend integration.
- Active Goal question/explanation non-mutating behavior in frontend.
- Active Goal steer/replan routing through backend decision plus continuation mutation path.
- Initial execution/transcript/subagent projection work from earlier related slices.

## Verification Already Run

Recent focused checks passed:

```powershell
python -m pytest tests/test_goal_api.py tests/test_main_chat_tui.py -q
python -m pytest tests/test_goal_copy.py -q -k "goal_proposal_fallback_copy or goal_command_help_message or goal_lifecycle_copy or queued_after_resolution"
python -m pytest tests/test_goal_api.py -q -k "goal_turn_decision_route"
npm.cmd run test:chat-active-goal-turn-decision
npm.cmd run test:chat-goal-routing
npm.cmd run test:chat-goal-workflow-routing
npm.cmd run type-check
```

Known non-failing warning:

- Pytest may warn that `.pytest_cache` cannot be created because of local permission restrictions.

## Remaining Work

The plan is not complete. Continue in order from the active plan.

Recommended next slices:

1. Backend `ActiveGoalTurnDecision` hardening
   - Current classifier is bounded fallback and partly heuristic.
   - It needs broader semantic/model-backed behavior before relying on it as the full product contract.

2. TUI Active Goal follow-up integration
   - TUI route helpers are de-authoritized.
   - TUI does not yet have the same active-goal turn decision integration as the frontend.

3. Blocked and approval diagnostics
   - Replace generic blocked copy with structured `BlockerDiagnostic` and `ApprovalDiagnostic`.
   - Explain what is blocked, why, actor required, next action, and auto-resume behavior.

4. Durable/transient event-lane contract
   - Runtime/transcript events need explicit `visibility`, `durability`, `projection_lane`, and stable `event_id` or `dedupe_key`.
   - Frontend must not rely on fragile `acknowledgement`-style heuristics.

5. WorkflowPanel repositioning
   - Treat as advanced/operator inspection and override surface.
   - It should not be the main user-facing Goal reasoning surface.

6. Legacy migration expansion
   - Cover legacy `interaction_mode`, `execution_mode`, `protocol_id`, `protocol_selection`, `selection_rationale`, and route-derived `default_route`.

7. Page-level controller extraction
   - `web/src/app/page.tsx` remains too large.
   - Extract active Goal turn handling, strategy display, and execution projection logic.

## Known Risks

- `web/src/app/page.tsx` is a hotspot with substantial unrelated churn.
- Several frontend tests are source-shape tests. They are useful guardrails but not a substitute for browser/live workflow tests.
- Active Goal turn decision is integrated in frontend but still needs deeper backend/model semantic quality.
- Some TUI behavior is intentionally de-authoritized but not yet fully reconnected to backend active-goal decisions.
- Worktree after commit still had untracked local artifacts such as temporary diffs, screenshots, `.tokensave/`, and local scripts. These were intentionally not committed.

## Do Not Regress

Future agents should treat these as hard invariants:

- Non-slash natural language stays `direct_chat` at frontend/TUI route-helper level.
- Natural-language Goal intent and active Goal follow-up interpretation belong to backend/model runtime.
- Proposal creation does not start execution.
- Ordinary Goal default is `autonomous_single_agent`, not `teacher_student_distill`.
- `teacher_student_distill` requires explicit override/confirmation.
- Active Goal question/explanation turns must not enqueue guidance, resume, or restart attempts.
- Active Goal steering/replanning may mutate runtime only after typed backend decision.
- User-facing copy should describe Goal as a contract and Workflow as a strategy, not as a separate primary route.
