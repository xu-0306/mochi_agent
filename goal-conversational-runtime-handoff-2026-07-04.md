# Goal Conversational Runtime Handoff - 2026-07-04

## Current Status

- Repo: `H:/_python/agent_mochi`
- Branch: `main`
- Current head after sticky GoalCard follow-up: `f2130aa Clear stale goal proposal cards`
- Observed branch state: `main...origin/main [ahead 21]`
- Tracked worktree was clean after `f2130aa`; only pre-existing untracked artifacts remained.

## Completed Goal Runtime Work

The Goal conversational runtime recovery plan is functionally complete.

The previous completion audit found two remaining gaps, both closed in `9458252 Wire active goal selector and live evidence test`:

1. Production active-goal turn decisions now use backend typed semantic mediation when an engine exposes `invoke`, while preserving conservative fallback behavior.
   - Key files:
     - `mochi/runtime/active_goal_turn_selector.py`
     - `mochi/main.py`
     - `mochi/api/routes/approvals.py`
     - `tests/test_goal_api.py`
     - `tests/test_main_chat_tui.py`

2. WS8 browser evidence now exercises real chunked SSE bytes, polling recovery, reload/reconnect behavior, and row-level dedupe.
   - Key files:
     - `web/scripts/test-goal-live-browser-evidence.mjs`
     - `web/src/components/chat/ExecutionTimeline.tsx`

## Sticky Goal Proposal Card Follow-Up

User noticed an old `Goal 提案` card still appeared after normal chat input (`hi`).

Root cause:

- The visible card was not an active Goal runtime card.
- It was a persisted chat-history `message.goalCard` for a pending proposal.
- Pending proposal follow-up routing treated ordinary greetings as `goal_pending_follow_up`, and the ambiguous branch re-persisted the proposal card.
- Old proposal cards without `goalId` were not superseded by `chat-projections`.

Fix committed in `f2130aa Clear stale goal proposal cards`:

- `hi`, `hello`, `你好`, `嗨` bypass pending proposal capture and stay `direct_chat`.
- Ordinary direct chat while a pending proposal exists clears `pending_proposal` from session goal state.
- Once `pending_proposal` is cleared, stale `proposal` / `revised_proposal` cards are not rendered as GoalCards.
- Key files:
  - `web/src/lib/chat-goal-routing.ts`
  - `web/src/app/page.tsx`
  - `web/scripts/test-chat-goal-routing.mjs`

## Verification Evidence

After `f2130aa`:

- `npm.cmd run test:chat-goal-routing` passed.
- `npm.cmd run test:chat-goal-workflow-routing` passed.
- `npm.cmd run type-check` passed.
- `git diff --check` passed.

Earlier verification for the completed plan included:

- `npm.cmd run test:goal-live-browser-evidence` passed.
- `npm.cmd run test:chat-active-goal-turn-decision` passed.
- `npm.cmd run test:goal-execution-timeline-projection` passed.
- `npm.cmd run test:execution-transcript-chat-projection` passed.
- Backend active-goal route/TUI pytest subsets passed.

## Cleanup State

No tracked product-code work remains known.

Cleanup candidates still exist and should not be deleted without explicit user approval:

- Old worktrees under `.tmp/` and `.worktrees/`
- Untracked artifacts such as `.tmp_scoped_branch.diff`, `.tmp_scoped_worktree.diff`, `.tokensave/`, `output-goal-*.png`, `output-keyboard-send.png`, and helper scripts.

