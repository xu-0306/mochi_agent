# Tool Intent Router Stage 1 Mutation Fallback Fix Brief

Date: 2026-06-27
Status: review follow-up
Scope owner: worker subagent implementation, main agent review only
Related baselines:
- `docs/superpowers/plans/2026-06-26-tool-intent-router-stage1-implementation-plan.md`
- `docs/superpowers/handoffs/2026-06-27-tool-intent-router-stage1-review-fix-brief.md`

## Summary

The previous review follow-up correctly fixed:

- preview/chat routing contract drift
- fallback `workspace_inspection` over-classification for generic knowledge queries

One remaining `P2` issue is still present:

- fallback `workspace_mutation` can still trigger on generic knowledge-writing phrases such as `rewrite`, `change`, or `modify` without explicit workspace-local evidence

This third-round fix is intentionally narrow. Do not widen scope.

## Problem Statement

Current fallback behavior still allows generic text-editing language to route into `workspace_mutation` if paired with broad workspace object terms.

Observed repros:

- `rewrite code switching explanation for beginners`
- `change project management explanation to be shorter`
- `modify history source criticism summary`

These are not local workspace mutation requests. They are ordinary knowledge/chat requests.

## Required Fix

Fallback `workspace_mutation` must require explicit workspace-local evidence, not just broad topic words.

Examples of acceptable workspace-local evidence:

- explicit repo/workspace references
- explicit file references such as `foo.py`, `report.md`, `settings.json`
- explicit path references
- attached workspace files
- phrases like `workspace file`, `local file`, `repo`, `repository`, `codebase`

Examples that should no longer route to `workspace_mutation` in fallback:

- `rewrite code switching explanation for beginners`
- `change project management explanation to be shorter`
- `modify history source criticism summary`

Examples that should still route to `workspace_mutation`:

- `rewrite foo.py to remove TODO`
- `modify the workspace file report.md`
- `change the repo config in settings.json`
- `update this attached workspace file`

## Design Guidance

- Keep the separate `tool_intent_router` architecture intact.
- Do not push this logic back into `ToolExposurePlanner`.
- Do not solve this by adding more topic-specific open-world keywords.
- Narrow the mutation fallback gate so it needs strong local evidence, similar to the stricter `workspace_inspection` gate.

## Allowed Write Scope

- `mochi/agents/tool_intent_router.py`
- focused tests only

Avoid touching:

- `mochi/agents/engine.py` unless absolutely necessary
- `mochi/agents/tool_exposure.py`
- goal/workflow/protocol files
- unrelated in-flight edits

## Acceptance Criteria

This follow-up is complete only if all are true:

1. Generic knowledge rewrite/change/modify prompts no longer route to `workspace_mutation` in fallback.
2. Explicit file/path/workspace mutation prompts still route to `workspace_mutation`.
3. Existing fallback `workspace_inspection` fix remains intact.
4. Existing Chinese weather/research exposure behavior does not regress.
5. Targeted tests cover both the negative and positive mutation cases.

## Minimum Test Coverage

Add or update tests for:

1. `rewrite code switching explanation for beginners` -> not `workspace_mutation`
2. `change project management explanation to be shorter` -> not `workspace_mutation`
3. `rewrite foo.py to remove TODO` -> `workspace_mutation`
4. explicit workspace/local-file mutation prompt -> `workspace_mutation`

Keep tests small and focused.

## Working Rules For Subagent

- You are not alone in the codebase. Do not revert unrelated changes.
- Keep scope bounded to this single residual routing issue.
- If you believe the issue requires broader architectural changes, stop and report instead of improvising.
