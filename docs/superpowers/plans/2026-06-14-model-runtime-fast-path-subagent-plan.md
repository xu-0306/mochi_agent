# Model Runtime Fast-Path Subagent Plan

Date: 2026-06-14
Branch baseline: `codex/exec-first-editing-wip-20260614`
Latest reference commit at planning time: `b79e8c5`

## Summary

The current model add/switch/delete flow is too expensive for the UX we want.

Right now, model-related config changes can trigger a broad runtime refresh path, which rebuilds tool registry state and causes:

- a visible pause in the UI
- repeated `Registered tool: ...` logs
- unnecessary churn in subsystems that are unrelated to model selection

The immediate goal is not a large architecture rewrite. The goal is to introduce a fast path for model-only changes so saved model operations and active model switching feel lightweight and predictable.

## Problem Statement

Observed behavior:

- adding, editing, switching, or deleting models can produce a short UX stall
- delete definitely triggers `engine.apply_config(...)`
- `apply_config(...)` currently rebuilds broad runtime state, including tool registry factory and tool registry

Relevant current code:

- [mochi/agents/engine.py](/H:/_python/agent_mochi/mochi/agents/engine.py)
- [mochi/api/routes/models.py](/H:/_python/agent_mochi/mochi/api/routes/models.py)
- [mochi/api/server.py](/H:/_python/agent_mochi/mochi/api/server.py)
- [mochi/tools/registry.py](/H:/_python/agent_mochi/mochi/tools/registry.py)
- [web/src/app/page.tsx](/H:/_python/agent_mochi/web/src/app/page.tsx)
- [web/src/app/settings/page.tsx](/H:/_python/agent_mochi/web/src/app/settings/page.tsx)

## Desired Outcome

Model operations should be classified and handled differently:

- saved model catalog changes should usually be config-only and cheap
- active model runtime switching should be a targeted router/runtime action
- tool registry rebuilds should happen only when tool visibility or workspace/tool policy actually changes

## Work Split

### Worker 1: Engine Fast Path

Scope:

- [mochi/agents/engine.py](/H:/_python/agent_mochi/mochi/agents/engine.py)
- small supporting changes in [mochi/api/routes/models.py](/H:/_python/agent_mochi/mochi/api/routes/models.py) if needed

Goal:

- split full runtime config application from model-only config application

Deliverables:

- introduce a model-only fast path such as `apply_model_config_fast_path(...)`
- keep `apply_config(...)` for full runtime refresh cases
- extract an explicit predicate for when tool registry rebuild is required
- avoid rebuilding:
  - `ToolRegistryFactory`
  - `ToolRegistry`
  - `MemoryStore`
  - `SessionStore`
  - `SkillLibrary`
  - `PromptBuilder`
  when only active model settings change

Acceptance criteria:

- model switch does not log a full list of `Registered tool: ...`
- tool registry rebuild only occurs when tool/workspace/security-relevant config changes
- existing chat/model behavior stays green

Notes for worker:

- do not introduce a second general-purpose config pipeline
- prefer small refactor over new abstraction layers

### Worker 2: Saved Model Catalog vs Active Runtime Separation

Scope:

- [mochi/api/routes/models.py](/H:/_python/agent_mochi/mochi/api/routes/models.py)
- frontend settings integration only where necessary

Goal:

- separate cheap saved-model CRUD from active runtime actions

Deliverables:

- adding a saved model entry should not automatically force a broad runtime refresh
- editing a non-active saved model should only update catalog/config state
- deleting a non-active saved model should not touch runtime
- only active-model-impacting changes should trigger runtime fallback or switch

Acceptance criteria:

- deleting an inactive saved model is near-instant
- editing inactive saved model metadata is near-instant
- deleting the active model still performs correct fallback behavior

Notes for worker:

- preserve current API contracts where possible
- avoid coupling inactive catalog operations to runtime readiness checks

### Worker 3: Non-Blocking Model Switch UX

Scope:

- [web/src/app/page.tsx](/H:/_python/agent_mochi/web/src/app/page.tsx)
- model selector related UI paths
- settings page model interaction surfaces if needed

Goal:

- make active model switching feel like a targeted async transition rather than a global freeze

Deliverables:

- explicit model switch state:
  - `idle`
  - `switching`
  - `ready`
  - `failed`
- only model-specific controls disable during switch
- clear switch status message and error handling
- rollback visible current-model state if switch fails

Acceptance criteria:

- switching model does not visually stall unrelated panels
- user can see switch progress and failure reason
- no regression to current send/message flow

Notes for worker:

- do not build a generic job orchestration system for this
- keep state local to model switching UX

### Worker 4: Logging and Observability Cleanup

Scope:

- [mochi/tools/registry.py](/H:/_python/agent_mochi/mochi/tools/registry.py)
- relevant engine/server logging paths

Goal:

- reduce noisy logs while preserving useful rebuild visibility

Deliverables:

- replace repeated per-tool registration spam for routine rebuild cases with summary logging
- examples:
  - `Tool registry rebuilt for workspace X (N tools)`
  - `Model switch fast path applied`
  - `Full runtime config refresh applied`
- keep detailed logging available only when genuinely useful

Acceptance criteria:

- routine model operations no longer flood logs with per-tool lines
- logs still allow diagnosis of whether a tool rebuild happened

Notes for worker:

- do not hide all rebuild signals
- the goal is signal compression, not silence

### Worker 5: Regression Review and Anti-Overdesign Audit

Scope:

- cross-cutting review after Workers 1-4 land

Goal:

- verify the implementation stays minimal and does not create new structural debt

Deliverables:

- findings-first review
- explicit check for:
  - accidental second config pipeline
  - runtime fast path bypassing necessary security/tooling refresh
  - new locking or serialized refresh paths that just move the stall elsewhere
  - duplicated frontend state models for switching

Acceptance criteria:

- no major findings, or findings clearly listed with fixes required

## Recommended Execution Order

1. Worker 1
2. Worker 2
3. Worker 3
4. Worker 4
5. Worker 5

## Integration Rules

All workers should follow these constraints:

- do not revert unrelated changes in the branch
- assume other workers may be editing nearby code
- keep write ownership as isolated as possible
- prefer adapting current code paths over introducing new parallel subsystems

## Test Plan

### Backend

- switching active model should avoid full tool registry rebuild in model-only cases
- deleting inactive saved model should not call the heavy refresh path
- deleting active saved model should still fallback correctly
- tool registry rebuild should still occur when workspace/tool/security config changes

### Frontend

- current model switch shows explicit transient status
- send/chat surface remains usable except for the minimal switching scope
- model switch failure restores visible state cleanly

### Logging

- no full `Registered tool: ...` spam during ordinary model switch
- rebuild summary logs still appear when a true rebuild occurs

## Non-Goals

- no full service-layer decomposition in this phase
- no complete `AgentEngine` rewrite
- no new generic runtime orchestration framework
- no second approval/session/tool exposure model

## Suggested First Refactor Cut

The smallest useful first cut is:

1. add a rebuild predicate inside `apply_config(...)`
2. skip tool registry and related subsystem rebuild on model-only changes
3. make delete/update model routes only call the heavy path when active runtime behavior is actually affected

This should deliver the largest UX improvement with the least architectural risk.
