---
summary: "Wave 4 review remediation record: deterministic dynamic re-gating and safe rollout provenance are implemented; external representative-fixture performance measurement remains a release gate."
created: 2026-07-29
tags: [ordinary-chat, adaptive-runtime, wave4, rollout, evidence]
related: [docs/superpowers/plans/2026-07-26-ordinary-chat-adaptive-agent-runtime-implementation-plan.md, mochi/agents/complexity_gate.py, mochi/agents/engine.py]
---

# Ordinary-Chat Adaptive Runtime — Wave 4 Review Remediation

## Revision and working-tree status

- Baseline HEAD: `76f46b7d` (`feat: complete ordinary chat adaptive runtime wave 2`).
- This record describes uncommitted review-remediation work. It is not a release
  manifest and does not claim that the working tree has been staged, committed,
  or deployed.

## Dynamic re-gating contract

- ReAct derives only host-observed signals before a non-read effect boundary:
  configured iteration threshold, third distinct tool, successful-read to
  effectful escalation, verifier failure, and active-plan invalidation.
- The Engine callback reuses the trusted resolved contract and capability
  summary, never model-provided counters, targets, or trigger names.
- Rechecks are deterministic and advisor-free. A grey-zone advisor call is not
  repeated during re-gating.
- Before a newly required plan can block a mutation, its upgraded decision and
  reason codes are written with the current `TurnCheckpoint` CAS revision. A
  failed write fail-closes the effect boundary.
- Checkpoint restore already reloads `complexity_decision`; the decision is not
  dependent on an in-memory callback after restart.

## Rollout semantics

- New configs default to `complexity.mode: enforce` with
  `rollout_version: enforce-v2`.
- A legacy full config which explicitly persisted `shadow` or `off` before the
  marker existed is loaded as `legacy-v1` and remains in that mode. The loader
  intentionally does not guess whether it was an old product default or an
  operator rollback.
- An operator Settings PATCH using `agent.complexity_mode` writes
  `rollout_version: operator-v2`, preserving an explicit rollback choice.

## CI-safe verification recorded on 2026-07-29

```powershell
rtk pytest tests/unit/agents/test_complexity_gate.py tests/unit/engine/test_adaptive_runtime_wave1.py
# 45 passed

rtk pytest tests/unit/engine/test_adaptive_runtime_wave2.py tests/unit/engine/test_turn_contract_rollout.py tests/unit/engine/test_adaptive_diagnostics_engine.py
# 45 passed

rtk pytest tests/unit/learning/test_failure_learning.py tests/unit/api/test_adaptive_runtime_projection.py tests/integration/api/sessions/test_adaptive_runtime_routes.py tests/test_config.py tests/integration/api/sessions/test_settings_routes.py
# 109 passed

rtk proxy python -m py_compile mochi/agents/complexity_gate.py mochi/agents/engine.py mochi/agents/react_loop.py tests/evaluation/evaluate_adaptive_runtime_wave4.py
# passed
```

## Phase 9 measurement and labelled review

The reproducible result is stored in
`2026-07-29-ordinary-chat-adaptive-runtime-wave4-measurement.json`, generated
by the versioned `tests/evaluation/evaluate_adaptive_runtime_wave4.py` entry
point:

```powershell
rtk proxy python tests/evaluation/evaluate_adaptive_runtime_wave4.py --fixtures tests/fixtures/adaptive_runtime/wave4_rollout_fixtures.json --output docs/superpowers/handoffs/2026-07-29-ordinary-chat-adaptive-runtime-wave4-measurement.json
```

It records the fixture SHA-256, UTC time, source revision, relative-path
source SHA-256 values for the evaluator and dynamic-gate implementation,
dirty-worktree count, environment, and median policy timing. It contains no
absolute path, prompt, secret, or tool payload. The bounded matrix produced TP=2, TN=4,
FP=0, FN=0: five fixtures create six observations because dynamic escalation
has both an initial and a host-signalled recheck label.

The evaluator pairs `off`, `shadow`, and `enforce` for identical redacted
contracts and invokes one actual ordinary-Chat simple fixture with the same
FakeBackend in each mode. Structural and Engine off-baseline adaptive
call/token/tool deltas are zero. A single-safe-edit remains planless for its
first effect boundary (`completed_iterations=0`); its next effect is re-gated
only after a completed ReAct iteration or another host-observed escalation
signal. Wall time is informational only, never a CI assertion.

Acceptance thresholds for that evidence:

| Fixture class | Required result |
| --- | --- |
| simple information, multilingual simple information | 0 adaptive model calls; no plan/advisor/recovery; enforce delta is zero structural calls |
| single safe edit | no hard plan solely due to the edit; normal capability/approval policy remains active |
| complex effectful | plan required before mutation |
| dynamic escalation | initial no-plan; next mutation blocked after durable dynamic decision |
| labelled matrix | report TP/FP/TN/FN counts and list every disagreement for human review |

If a future paired measurement or human review misses a threshold, use the
documented `operator-v2` rollback to `shadow` or `off` and re-open Phase 9.
