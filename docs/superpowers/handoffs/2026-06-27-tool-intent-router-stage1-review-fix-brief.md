# Tool Intent Router Stage 1 Review Fix Brief

Date: 2026-06-27
Status: review follow-up
Scope owner: worker subagent implementation, main agent review only
Related baseline:
- `docs/superpowers/plans/2026-06-26-tool-intent-router-stage1-implementation-plan.md`

## Summary

Stage 1 landed the intended architectural split:

- a separate `tool_intent_router`
- main chat integration in `AgentEngine`
- routed intent metadata flowing into `ToolExposurePlanner`

That direction is correct and should be preserved.

However, review found two remaining `P2` issues that should be fixed before this stage is considered ready:

1. preview and real chat currently use different routing behavior
2. fallback routing still over-classifies generic knowledge queries as workspace inspection

These are not just isolated bugs. They are design/architecture follow-up issues inside the new routing layer.

## Review Findings

### Finding 1: Preview / execution contract drift

Current state:

- `preview_chat_context()` calls `_route_tool_intent_for_exposure(..., enable_classifier=False)`
- the main chat execution path enables classifier routing for normal chat turns

Result:

- the same user message can receive one tool-intent decision during preview and a different one during real execution
- previewed tool exposure and token estimates can drift from actual execution

Why this matters:

- preview should describe the same decision contract that real execution will use
- this violates single-source-of-truth expectations for routing behavior

### Finding 2: Fallback router still has architecture-drift risk

Current state:

- fallback routing keeps a broad keyword list that treats words like `code` as strong workspace-object evidence
- generic knowledge questions can still route to `workspace_inspection` if classifier is unavailable, low-confidence, or disabled

Observed repro:

- `explain code switching in multilingual LLMs`
- this currently routes to `workspace_inspection` through fallback behavior

Why this matters:

- fallback should be a conservative degradation path, not a second independently opinionated primary classifier
- broad workspace keyword heuristics recreate the same hardcoded routing weakness this refactor was meant to reduce

## Required Fixes

### Fix 1: Unify preview and execution routing contract

Goal:

- the routing decision used by preview must match the routing decision used by real chat for the same bounded context

Acceptable implementations:

1. enable the same classifier path in preview that normal main-chat execution uses
2. or centralize a shared routing policy that explicitly guarantees preview and execution use the same decision mode

Non-acceptable outcome:

- leaving preview hardcoded to fallback-only while real execution remains classifier-first

If there is a performance concern:

- keep the policy unified first
- only then add explicit caching or bounded optimization
- do not solve performance by reintroducing semantic drift

### Fix 2: Tighten fallback workspace routing

Goal:

- generic knowledge/web/research queries must not fall into `workspace_inspection` just because they contain broad words like `code`, `project`, `source`, or similar

Required behavior:

- fallback workspace classification should require stronger workspace-local evidence
- examples of strong evidence:
  - repo / repository / workspace file / path / folder / directory language
  - explicit local-file references
  - attached workspace files
  - repo inspection verbs combined with repo-local nouns

Examples that should not be treated as workspace-local by fallback alone:

- `explain code switching in multilingual LLMs`
- `what is source criticism in history research`
- `tell me about project management frameworks`

Examples that should still route workspace-local:

- `find matching files and search for TODO in the repo`
- `幫我查詢 foo.py 裡的 TODO`
- `read this workspace file and summarize it`

Design guidance:

- do not just add more topic-specific open-world keywords
- instead narrow what counts as workspace evidence
- prefer requiring a stronger combination such as:
  - local/workspace noun
  - plus inspection/mutation intent
  - plus attachment/workspace context where relevant

## Locked Constraints

The worker must preserve:

- the new `tool_intent_router` module as the separate routing layer
- routed-intent metadata flowing into `ToolExposurePlanner`
- the current hotfix guardrail that keeps a general web baseline visible for non-explicit workspace tasks in workspace-bound sessions

The worker must not:

- collapse routing back into `ToolExposurePlanner`
- broaden scope into goal/workflow work
- redesign the entire tool exposure system
- revert unrelated in-flight changes in the repo

## Acceptance Criteria

This follow-up is complete only if all are true:

1. preview and execution now use a unified routing contract
2. fallback no longer routes `explain code switching in multilingual LLMs` to `workspace_inspection`
3. fallback still routes clear repo/file inspection queries to `workspace_inspection`
4. Chinese weather queries in workspace-bound sessions still expose web tools
5. Chinese research queries in workspace-bound sessions still expose literature/web tools
6. existing tool exposure tests still pass
7. new or updated tests explicitly cover the two review findings

## Minimum Test Additions

At minimum, add coverage for:

1. preview/execution routing parity or equivalent shared routing contract behavior
2. `explain code switching in multilingual LLMs` does not become `workspace_inspection`
3. clear repo-local query still becomes `workspace_inspection`

Targeted tests are preferred over broad new integration suites.

## Working Rules For Subagent

- You are not alone in the codebase. Do not revert unrelated changes.
- Keep write scope tight:
  - `mochi/agents/tool_intent_router.py`
  - `mochi/agents/engine.py`
  - `mochi/agents/tool_exposure.py` only if needed
  - focused tests only
- If a fix requires widening scope beyond this brief, stop and report the exact reason.
