# Tool Exposure / File UX Handoff

Date: 2026-06-15

Primary plan:
- [2026-06-14-tool-exposure-and-file-ux-recovery-plan.md](/H:/_python/agent_mochi/docs/superpowers/plans/2026-06-14-tool-exposure-and-file-ux-recovery-plan.md)

## Current Status

Completed:
- Task 1: remove keyword-gated workspace read tool exposure
- Task 2: restore full-fidelity file reading and resumable large-result flow
- Task 3: surface workspace/tool transport diagnostics in API and WebGUI

Pending:
- Task 4: shift default tool exposure toward dynamic discovery with `tool_search`

Not started:
- Task 5 optional repo-map / symbol-index follow-up

## Current Working Tree

There are uncommitted local changes in:
- [mochi/agents/tool_exposure.py](/H:/_python/agent_mochi/mochi/agents/tool_exposure.py)
- [mochi/api/routes/chat.py](/H:/_python/agent_mochi/mochi/api/routes/chat.py)
- [tests/test_tool_exposure.py](/H:/_python/agent_mochi/tests/test_tool_exposure.py)
- [tests/test_tools_phase2.py](/H:/_python/agent_mochi/tests/test_tools_phase2.py)
- [web/src/components/chat/ReasoningPanel.tsx](/H:/_python/agent_mochi/web/src/components/chat/ReasoningPanel.tsx)
- [web/src/lib/api.ts](/H:/_python/agent_mochi/web/src/lib/api.ts)
- [web/src/lib/chat.ts](/H:/_python/agent_mochi/web/src/lib/chat.ts)

Interpretation:
- Task 1 changes live in `mochi/agents/tool_exposure.py` and `tests/test_tool_exposure.py`
- Task 2 changes include the local `tests/test_tools_phase2.py` adjustment that aligned fallback-path expectations with actual persisted-artifact behavior
- Task 3 changes live in `mochi/api/routes/chat.py`, `web/src/lib/api.ts`, `web/src/lib/chat.ts`, and `web/src/components/chat/ReasoningPanel.tsx`

Do not revert these files unless explicitly asked.

## What Task 3 Changed

Backend:
- [mochi/api/routes/chat.py](/H:/_python/agent_mochi/mochi/api/routes/chat.py) now persists `status` events into replay, so additive diagnostics survive session reload/history fetch.

Frontend normalization:
- [web/src/lib/api.ts](/H:/_python/agent_mochi/web/src/lib/api.ts) now normalizes:
  - `metadata.tool_exposure`
  - `metadata.transport`
- These fields are attached to reasoning steps without recomputing them in the UI layer.

Frontend types:
- [web/src/lib/chat.ts](/H:/_python/agent_mochi/web/src/lib/chat.ts) now defines:
  - `ToolExposureDiagnostics`
  - `ToolTransportDiagnostics`
  - `ReasoningStep.toolExposure`
  - `ReasoningStep.transport`

Frontend rendering:
- [web/src/components/chat/ReasoningPanel.tsx](/H:/_python/agent_mochi/web/src/components/chat/ReasoningPanel.tsx) now shows:
  - exposed workspace tools
  - `workspace_bound`
  - `attachment_count`
  - `summary_applied`
  - `overflow_persisted`
  - `reference_id`
  - `artifact_path`
  - `source_path`
- It uses existing filesystem URL generation for `artifact_path`
- It drops null-only diagnostics
- It suppresses repeated identical diagnostics within one visible trace

Important constraint:
- Browser/UI must use `artifact_path`
- Browser/UI must not try to resolve `tool-result://...`

## Verification Already Run

Fresh verification on the current working tree:

```powershell
pytest tests/test_api_chat_attachments.py tests/test_engine_phase2.py -k "attachment or reasoning or metadata" -v
npm.cmd --prefix web run type-check
npm.cmd --prefix web run lint
```

Observed results:
- `pytest`: `5 passed, 16 deselected`
- `type-check`: passed
- `lint`: exit `0` with warnings only

Known lint warnings outside current task scope:
- [web/src/app/page.tsx](/H:/_python/agent_mochi/web/src/app/page.tsx)
  - `2220:11` unused `appendedDetail`
  - `2400:5` unnecessary hook dependency
  - `2603:6` missing hook dependency
  - `2752:6` missing hook dependency

Known pytest warning:
- `.pytest_cache` write warning on Windows permission/cache path

## Review Outcomes

Task 3 spec review:
- approved

Task 3 code-quality review:
- initially found two issues:
  - null-only diagnostics could render empty UI containers
  - repeated diagnostics could spam the reasoning trace
- both were fixed
- re-review approved

Residual low-risk gap:
- there is still no dedicated frontend test coverage for diagnostics normalization and reasoning-panel dedupe behavior

## Important Task 2 Note

The current local [tests/test_tools_phase2.py](/H:/_python/agent_mochi/tests/test_tools_phase2.py) reflects an important behavior difference:
- direct artifact-only virtual-path read fallback returns line-numbered content once
- overflow-created artifact fallback can return line-numbered chunk text that already contains prior line prefixes

This was intentionally adjusted after focused verification.

## Recommended Next Step

Start Task 4 from the existing plan.

Task 4 target intent:
- always keep the small workspace baseline visible
- include `tool_search` by default for larger tool sets
- keep keyword/group heuristics as ranking only
- update prompt guidance so the model uses:
  - core read tools directly for workspace inspection
  - `tool_search` when the needed tool is not already visible

Likely files for Task 4:
- [mochi/agents/tool_exposure.py](/H:/_python/agent_mochi/mochi/agents/tool_exposure.py)
- related `tool_search` wiring
- prompt guidance file(s)
- [tests/test_tool_exposure.py](/H:/_python/agent_mochi/tests/test_tool_exposure.py)

## Recommended Handoff Instructions For The Next Agent

1. Read the plan file first.
2. Assume Tasks 1-3 are implemented but still uncommitted.
3. Preserve all existing local modifications.
4. Re-run the focused verification for any file you touch.
5. Execute Task 4 with the same subagent-driven workflow:
   - implementer
   - spec review
   - code-quality review
6. Do not reopen Task 3 unless Task 4 work exposes a real integration bug.

## Suggested First Commands

```powershell
git status --short
Get-Content 'docs/superpowers/plans/2026-06-14-tool-exposure-and-file-ux-recovery-plan.md'
pytest tests/test_tool_exposure.py -v
```
