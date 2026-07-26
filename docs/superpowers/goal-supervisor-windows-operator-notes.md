# Goal Supervisor Windows Operator Notes

Date: 2026-06-24
Scope: operating long-running goals, approvals, restart recovery, and collector progress on Windows

## What Persists

- `sessions/runtime.db`
  - tasks
  - task approvals
  - agent runs
  - goals, attempts, leases, audit findings
  - goal checkpoints, memory snapshots, worker generations
- `sessions/exec-approvals.db`
  - standalone `exec_command` approvals
  - resolved execution result metadata
  - resolved exec session ids
- `sessions/runtime-tasks/`
  - task sandboxes and task-local runtime outputs
- `<workspace>/exec-runtime/`
  - detached exec-runtime manifests and session logs used for session recovery

Do not delete or move these paths while a goal is active, waiting on approval, or being recovered after restart.

## Normal Operator Flow

1. Start or resume the goal from `/goals`.
2. Watch:
   - goal state
   - current attempt
   - runtime budget
   - current generation
   - approval state
   - open findings
   - checkpoints and memory snapshots
   - collector shard state
3. Use `resume`, `refresh worker`, `finalize partial`, `cancel`, or global `estop` only when the surfaced state justifies it.

`waiting_approval` is a real control-plane state. Resolve the pending approval first. Do not treat it like a generic stall.

## Restart Recovery Checklist

After restarting the API/runtime on Windows:

1. Let the runtime service boot fully.
2. Check `/goals` or `/v1/goals/{goal_id}/health` for:
   - current attempt
   - linked run status
   - approval state
   - latest persisted checkpoint
   - latest memory snapshot
   - current generation
3. Check `/v1/approvals`.
   - task-linked exec approvals should still appear from durable `approval_requests`
   - standalone exec approvals should still appear from `sessions/exec-approvals.db`
4. If an approval was already resolved before restart, confirm that:
   - `execution_result` is still visible
   - `exec_session_id` is still visible
5. If the run used detached/background exec, confirm that `<workspace>/exec-runtime/` still exists before expecting session inspection to work.

## Approval Notes

- Task-linked exec approvals and standalone exec approvals now persist independently.
- Task-linked approvals are anchored by `runtime.db`.
- Standalone exec approvals are anchored by `exec-approvals.db`.
- A missing standalone approval after restart is a persistence problem.
- An approval that exists but has no live session is not the same failure:
  - the command may already have exited
  - the detached runtime state may have been removed
  - the command may have been foreground-only

## Windows-Specific Operating Notes

- Prefer PowerShell-aware commands and absolute paths when investigating state manually.
- Keep approved exec workdirs inside the intended workspace or task sandbox.
- Do not clean `.pytest_cache`, `sessions/`, or `exec-runtime/` during live recovery work.
- In this repo, Windows may emit `PytestCacheWarning` for `.pytest_cache` access denial. Treat that as non-blocking unless the actual test failed.
- Detached exec manifests and logs are more useful than raw process inspection after restart. Check persisted metadata first.

## Collector Notes

- Collector shard progress is visible in `/goals` and in persisted goal progress records.
- A shard that stopped moving should be checked against:
  - open findings
  - latest checkpoint
  - latest memory snapshot
  - latest collector shard manifest
- Current rollout persists live shard manifests, but not full live dataset-record flush for every adapter yet.

## Minimal Verification Commands

Use focused checks before escalating:

```powershell
python -m pytest tests/test_api_runtime.py -k "standalone_exec_approval or task_exec_approval" -q
python -m pytest tests/test_goal_api.py -k "waiting_approval or checkpoint" -q
```

If restart-safe standalone exec approval behavior looks wrong, verify both the API surface and the durable store:

```powershell
python -m pytest tests/test_exec_security.py -k persistent_approval_store_roundtrip -q
python -m pytest tests/test_api_runtime.py -k standalone_exec_approval -q
```

## When To Escalate

Escalate to code investigation when any of these are true:

- approval id disappeared after restart
- `exec_session_id` disappeared but the approval record is still present and should be resolved
- goal health shows stale `waiting_approval` with no approval in `/v1/approvals`
- checkpoint/memory snapshot timestamps stop advancing for an active goal without an explicit finding explaining why
