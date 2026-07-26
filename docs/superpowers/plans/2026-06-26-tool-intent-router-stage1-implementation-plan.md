# Tool Intent Router Stage 1 Implementation Plan

Date: 2026-06-26
Status: draft
Owner model: subagent worker implementation, main agent review only
Related files:
- `mochi/agents/engine.py`
- `mochi/agents/tool_exposure.py`
- `tests/test_tool_exposure.py`
- `tests/test_prompt_builder.py`

## Summary

This stage introduces a bounded `tool intent router` for main chat tool exposure.

The immediate goal is to stop relying on English-only keyword matching as the primary determinant of whether open-world tools are visible, while preserving the current chat architecture and the hotfix that restored web tools for Chinese weather and research queries in workspace-bound sessions.

This is intentionally a stage-1 refactor, not a full redesign of all tool exposure behavior.

## Problem Statement

Current behavior in `ToolExposurePlanner` still mixes three concerns in one place:

1. task-intent classification
2. candidate tool-group policy
3. tool ranking and filtering

That coupling has already caused a real regression:

- a workspace-bound session plus a non-English user query could fall back to `workspace` defaults
- `web_search` / `web_fetch` were therefore not exposed at all
- the model then hallucinated that only repo/file tools were available

The recent hotfix corrected the immediate failure by keeping a small web baseline visible for non-explicit workspace tasks, but that should remain a guardrail, not the long-term primary routing mechanism.

## Stage-1 Objective

Add a separate intent-routing layer that classifies the current user turn into a small bounded set of tool intents before `ToolExposurePlanner` decides which tools to expose.

Stage 1 should be:

- classifier-first
- keyword-fallback
- bounded in scope
- backward-compatible with the current planner

## Locked Design

### Intent taxonomy

Stage-1 intent outputs must be a small stable set:

- `open_world_lookup`
- `literature_research`
- `workspace_inspection`
- `workspace_mutation`
- `execution_or_process`
- `tool_discovery`
- `ambiguous`

Do not add domain-specific intents like `weather`, `esg`, or `finance`.

### Router behavior

The router should:

1. prefer a bounded semantic classifier when available
2. fall back to the existing heuristic/keyword logic if classification is unavailable or low-confidence
3. return structured metadata:
   - `intent`
   - `confidence`
   - `source` (`classifier` or `fallback_keyword`)
   - short rationale

### Planner behavior

`ToolExposurePlanner` should stop acting as the primary task classifier.

Instead it should:

1. accept the routed intent as input
2. map that intent to tool-group policy
3. keep existing ranking/filtering logic where still useful
4. preserve the current workspace safety behavior
5. preserve the current hotfix guardrail for workspace-bound sessions

### Architecture constraint

Do not redesign chat, goal, workflow, or multi-agent routing in this stage.

Do not replace the entire planner with a new framework.

Do not broaden this into protocol-selection or goal-execution work.

## Suggested Implementation Shape

### New module

Add a new module:

- `mochi/agents/tool_intent_router.py`

Suggested contents:

- a small result model/dataclass, for example `ToolIntentRoute`
- router entrypoint, for example `route_tool_intent(...)`
- bounded classifier prompt or helper
- heuristic fallback helper

### Engine integration

Update `mochi/agents/engine.py` so main chat invocation:

1. builds the planner message
2. routes tool intent first
3. passes the routed intent into `ToolExposurePlanner.plan(...)`
4. records router metadata in diagnostics / tool exposure metadata

### Planner interface

Extend `ToolExposurePlanner.plan(...)` with something like:

- `routed_intent`
- optional `intent_confidence`
- optional `intent_source`

The planner should still work if those fields are absent, to reduce migration risk.

## Non-Goals

The worker must not do the following in this stage:

- remove the current web-baseline hotfix
- rewrite all keyword ranking rules
- redesign tool registry metadata contracts globally
- refactor goal/workflow routing
- change subagent execution profiles
- introduce broad UI changes
- revert or rewrite unrelated in-flight edits already present in the repo

## Acceptance Criteria

Stage 1 is complete only if all of the following are true:

1. A distinct tool-intent-routing layer exists outside `ToolExposurePlanner`.
2. Main chat tool exposure uses that routed intent.
3. Chinese weather-style queries in workspace-bound sessions still expose open-world web tools.
4. Chinese research-style queries in workspace-bound sessions still expose appropriate open-world research/web tools.
5. Repo/file inspection queries do not regress into unnecessary open-world leakage.
6. Existing behavior still works when router classification is unavailable and fallback is used.
7. Tests cover both classifier path and fallback path at a bounded level.

## Test Expectations

Add or update targeted tests around:

- `tests/test_tool_exposure.py`
- new router-specific tests if needed

Minimum coverage should include:

1. non-English weather query -> `open_world_lookup`
2. non-English research query -> `literature_research` or equivalent routed policy that exposes literature/web tools
3. repo/file query -> `workspace_inspection`
4. mutation/edit query -> `workspace_mutation`
5. tool-question query -> `tool_discovery`
6. router failure or low-confidence fallback -> planner still produces sane tool visibility

Keep tests targeted. Avoid turning this stage into a giant integration suite.

## Review Notes For Main Agent

Review in this order:

1. spec compliance
   - separate router exists
   - planner is no longer the sole classifier
   - stage remains bounded
2. code quality
   - interfaces are typed and minimal
   - fallback behavior is explicit
   - no overfitting to specific languages/topics
   - no unnecessary duplication between router and planner

## Working Rules For Subagent

- You are not alone in the codebase. Other files already have in-flight edits. Do not revert unrelated changes.
- Adjust to existing local modifications rather than overwriting them.
- Keep the write scope bounded to the stage-1 tool intent routing refactor and its tests.
- If you need to widen scope, stop and report the exact reason rather than improvising a larger rewrite.
