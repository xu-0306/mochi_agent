# Mochi Agent Maturity - Active Progress Handoff (2026-08-09, updated 2026-08-10)

## Current status

The governed agent-maturity plan is complete. The canonical JSON plan is
`verified` / `resolved`, with **30/30 packages verified** and **35/35 gates
passed**. The post-governance Main-90 release passed without a waiver.

This file supersedes its earlier in-progress content, which described 20
verified packages, 10 integrated packages, and an unfinished Failure UI gate.
Those statements are historical and are no longer active instructions. The
authoritative governance closeout is:

- `.claude/skills/agent-memory/memories/project-status/agent-maturity-final-closeout-2026-08-09.md`
- `artifacts/foreman/PKG-MAIN-90-INTEGRATE-QUALIFY-RELEASE/integration-final-closeout-20260809.json`
- `artifacts/foreman/PKG-MAIN-90-INTEGRATE-QUALIFY-RELEASE/release-20260809-post-governance.json`

## Post-closeout live WebApp validation

### 2026-08-10 current follow-up

The earlier `session-42` data was synthetic Chat integration-test pollution,
not startup seed data or an undeleted user conversation. Chat API tests now use
an isolated per-test `tmp_path` working directory, and a regression fingerprints
the repository session directory around a real Chat POST. The full Chat package
passed 69 tests with one existing Starlette warning, and all four leaked
synthetic sessions were removed through the production DELETE API.

A fresh real FastAPI + Next.js + Chrome regression then passed 6 cases and 117
checks with no application 5xx, page exception, or failed browser request. It
covered session create/search/open/delete, desktop and mobile primary routes,
Failure Presentation, file-subset replacement/conflict behavior, and eight
parallel tool-workflow snapshots completing in 0.016 seconds.

The configured `gpt-5.6-luna` was also exercised through the real WebApp. The
browser sent the temporary session and exact model ID to `/v1/chat/stream`,
received HTTP 200 and the exact sentinel answer in 12.797 seconds, and backend
replay contained both user and assistant messages. All 23 inference checks
passed and the temporary session was deleted.

Current evidence:

- `artifacts/webapp-testing/20260810-post-isolation/post-isolation-regression.json`
- `artifacts/webapp-testing/20260810-post-isolation/live-luna-inference.json`
- `artifacts/webapp-testing/20260810-post-isolation/live-luna-inference.png`
- `.claude/skills/agent-memory/memories/project-status/agent-maturity-session-isolation-and-live-luna-validation-2026-08-10.md`

The following 2026-08-09 section is retained as historical evidence for the
workflow-read liveness repair. Its `session-42` assumption and optional Ollama
boundary are superseded by the follow-up above.

FastAPI, Next.js, and headless Chromium were started as real processes and the
production UI was exercised against the existing `session-42` data.

- Every observed `tool-workflow` request returned HTTP 200. The liveness run
  recorded 78 browser API responses, all HTTP 200.
- Direct FastAPI `/health` and `/v1/sessions`, plus the Next proxy sessions
  request, remained HTTP 200 while the home page was loading; each liveness
  probe completed in at most 0.05 seconds.
- Desktop search and navigation to Goals, Agent Runs, Skills, and Settings
  passed through real clicks.
- Production Failure Presentation and file-subset workflow interactions
  passed.
- The only browser-visible non-200 application probe was the optional Ollama
  check to `localhost:11434`, which returned 502 because Ollama was not
  running. This is not a regression in the repaired workflow path.
- `/favicon.ico` returned 404 and is non-blocking.

Evidence:

- `artifacts/webapp-testing/20260809-live/home-backend-liveness.json`
- `artifacts/webapp-testing/20260809-live/desktop-navigation-live.json`
- `artifacts/webapp-testing/20260809-live/live-interactions.json`

## Tool-workflow liveness repair

The live run initially exposed a production failure mode under overlapping
home-page workflow reads. The repair is intentionally narrow:

- `mochi/api/routes/sessions.py` reuses one route-level
  `ToolWorkflowOutboxRepository` while the application Engine has not yet
  initialized.
- `mochi/api/tool_workflow_outbox.py` coalesces overlapping strict snapshot
  reads for the same session with an `asyncio.Task` protected by `shield`.
  The completed task is removed, so later requests reread durable state rather
  than using a permanent cache.
- `tests/test_tool_workflow_outbox.py` proves a 24-request fan-out performs one
  strict read and that a later request reads durable state again.
- `tests/integration/api/sessions/test_tool_workflow_liveness.py` uses a real
  JSONL file larger than 64 KiB, a real SessionStore sidecar lock, and 64
  concurrent HTTP requests to prove workflow responses, `/health`, and worker
  pool liveness.

Verification:

- Terra repair/evaluation set: 81 passed; Ruff and `git diff --check` passed.
- Main related regression set: 70 passed, with only the existing Starlette
  deprecation warning.
- Final focused rerun: 2 passed in 1.10 seconds.
- Final scoped `rtk git diff --check`: passed.

## Test-infrastructure caveat

`webapp-testing/scripts/with_server.py` must not be used for the high-volume
Windows liveness run in its current form. It pipes server stdout/stderr without
draining them, so Uvicorn access logs can fill the pipe and create a false
server hang. Its `shell=True` termination can also stop only the shell parent
and leave the Uvicorn or Next child listener running. The accepted live run
used persistent exec sessions that continuously drained output and then
verified exact listeners during cleanup.

## Workspace boundary and next action

- At the end of the 2026-08-10 validation, the user-owned Next and FastAPI
  listeners remained active on ports 3000 and 8000 (PIDs 29884 and 33732).
  Recheck PIDs before any future process operation and do not stop pre-existing
  user services without authorization.
- The only listed conversation after cleanup was the user-owned session
  `7604ec4c-f67a-4578-b9c8-f2a889aa0063`, titled `hi`; `session-42` and both
  temporary validation sessions were absent.
- The worktree is intentionally broad, dirty, and user-owned. Nothing was
  staged, committed, pushed, reset, or cleaned.
- The agent-maturity plan has no remaining implementation or qualification
  gap. The configured `gpt-5.6-luna` ordinary-Chat path is live-validated; the
  historical stopped-Ollama probe is not an active blocker for this setup.
