---
summary: "Diagnosis handoff for the second-turn 400 from Coderelay/Luna: completed native tool history is replayed across turns, but the upstream Chat Completions adapter cannot pair a prior function-call output with its function call."
created: 2026-08-02
tags: [ordinary-chat, openai-compatible, chat-completions, tool-history, multi-turn, handoff]
related: [mochi/agents/engine.py, mochi/agents/react_loop.py, mochi/backends/openai_compat.py, mochi/backends/types.py, tests/test_openai_compat_backend.py, tests/unit/engine/test_sessions_and_events.py]
---

# Cross-turn native tool history 400 diagnosis handoff

## Handoff status

- Diagnosis is complete enough to begin a focused implementation investigation.
- No source code was changed during this diagnosis.
- The current browser tab was inspected and left open at `http://127.0.0.1:3000/`.
- The working tree was already heavily dirty before this handoff. Preserve all unrelated changes and do not reset or rewrite them.
- The user explicitly clarified that Codex `Invalid prompt` policy errors are unrelated to this project and should not be investigated as part of this defect.

## User-visible failure

The first turn succeeds and may execute several tools. A second ordinary Chat message in the same conversation then fails immediately with:

```text
Client error '400 Bad Request' for url 'https://cdn.coderelay.cn/v1/chat/completions'
```

Observed conversation:

1. First turn: `幫我查詢近期比較新的cnn影像處理相關論文`
2. First turn completed after multiple literature/tool calls.
3. Second turn: `https://www.digitimes.com.tw/seminar/mdivrc_2026/ 那我現在想要針對這個比賽去做，你認為我可以先研究哪幾種方法`
4. The second turn failed before it executed any new tool.

This proves the defect is not specific to time-zone prompts or `get_current_time`.

## Exact backend evidence

Backend log:

```text
.tmp/backend-timezone.stderr.log
```

At approximately `2026-08-02 17:32:31`:

```text
Tool exposure plan: backend=openai_compat,
tool_mode=auto,
execution_profile=chat,
contract_operations=['literature_research', 'open_world_lookup'],
exposed_tools=['tool_search', 'arxiv_search', 'update_plan', 'tool_result_read']

ReAct iteration 1/10

OpenAI-compatible stream API error 400:
{"error":{"message":"No tool call found for function call output with call_id fc_fERVbM77GSMkWH4PelJVewGg.","type":"invalid_request_error","param":"","code":null}}
```

Relevant log lines are around 618-621.

Interpretation:

- Contract resolution and tool exposure completed normally.
- The first model request of the second turn was rejected.
- No current-turn tool call or tool execution occurred.
- The upstream API rejected replayed history, not the current prompt.

## Persisted session evidence

Session ID:

```text
308f8d2e-332e-4d85-bad8-c8e2527d1f86
```

Session file:

```text
.mochi/sessions/~sid-v2-0cdf659913af14e421d0e2aeef13faea9565d3aca77e332fff4775d95fb58694.jsonl
```

Important physical lines:

| Line | Role/event | Evidence |
| --- | --- | --- |
| 61 | assistant message | Native `tool_search` call with ID `call_fERVbM77GSMkWH4PelJVewGg` |
| 62 | tool message | `tool_search` result with matching `tool_call_id=call_fERVbM77GSMkWH4PelJVewGg` |
| 77 | assistant message | Successful first-turn final answer |
| 87 | user message | The DigiTimes competition follow-up prompt |
| 95 | turn error event | Upstream reports missing `fc_fERVbM77GSMkWH4PelJVewGg` function call |
| 102 | timeline meta | Second turn closed after the error; physical line and revision are both 102 |

The canonical JSONL contains a valid assistant-call/tool-result pair using the same `call_*` ID. The `fc_*` form appears in the upstream error response, not in the persisted message pair.

## Timeline integrity result

This session is not affected by the previously observed rewrite corruption.

Latest timeline samples were consistent:

```text
line 88  -> history_current_revision 88
line 89  -> history_current_revision 89
line 90  -> history_current_revision 90
line 92  -> history_current_revision 92
line 102 -> history_current_revision 102
```

Therefore this occurrence is not caused by:

- `timeline history_current_revision does not match event position`
- `_rewriteable_session_events_before_turn`
- `rewrite-from-turn`

Those remain a separate defect in older/re-written sessions.

## Current code path

The current flow is:

1. `AgentEngine._restore_session_history_events()` restores durable `message` events, including historical assistant `tool_calls` and tool `tool_call_id` values.
2. `AsyncReActLoop.run()` constructs one request context from system prompt, restored history, and the new user message.
3. `OpenAICompatBackend._build_chat_completions_payload()` serializes every message with `Message.to_dict()`.
4. `Message.to_dict()` emits completed historical tool turns again as native Chat Completions `assistant.tool_calls` plus `role=tool` messages.
5. Coderelay/Luna rejects that cross-turn replay because it cannot associate the replayed function-call output with the corresponding prior function call.

Relevant symbols:

- `mochi/agents/engine.py:7507` — `_restore_session_history_events`
- `mochi/agents/react_loop.py:245` — `AsyncReActLoop.run`
- `mochi/backends/openai_compat.py:1323` — `_build_chat_completions_payload`
- `mochi/backends/types.py:146` — `Message` and `Message.to_dict`

The on-disk call/result pair is structurally valid under the ordinary Chat Completions schema. The integration defect is that Mochi treats completed native tool protocol from old turns as universally replayable provider context. That assumption is not portable across OpenAI-compatible gateways.

## Root-cause statement

High-confidence root cause:

> Completed native tool-call protocol is being replayed literally across ordinary Chat turns. The Coderelay/Luna Chat Completions adapter does not support that replay representation reliably and looks for a prior `fc_*` function-call identity that is not reconstructable from the persisted `call_*` message pair. The second-turn request is therefore rejected as an orphaned function-call output.

Scope the fix to the Mochi/provider boundary even if part of the incompatibility is upstream. Mochi must construct portable history instead of relying on provider-specific call-ID behavior.

Do not implement a string-prefix rewrite such as `call_ -> fc_`. That would be a provider-specific heuristic, would not establish a real call/output relationship, and could corrupt providers that already implement Chat Completions correctly.

## Related work that does not fix this occurrence

There is an existing completed plan at:

```text
docs/superpowers/plans/2026-08-02-responses-continuity-capability/plan.md
```

That work controls `/responses` continuity and `previous_response_id` capability selection. The current failure uses:

```text
https://cdn.coderelay.cn/v1/chat/completions
request_shape=chat_completions
tool_call_mode=native
```

Do not assume the Responses continuity implementation covers this Chat Completions history-replay defect. Reuse its explicit-capability and anti-hardcode principles, but keep the transport paths distinct.

## Separate failures already observed

Do not merge these into one root cause:

1. External 502 from `https://co.yes.vg/v1/responses`: Cloudflare `origin_bad_gateway`; retryable upstream outage.
2. Luna reasoning-effort 400: `minimal` is unsupported; supported values are `none`, `low`, `medium`, `high`, `xhigh`, `max`.
3. Older timeline corruption after `rewrite-from-turn`: timeline revisions no longer match physical event positions.
4. Current defect: second-turn Chat Completions replay contains a function-call output the upstream cannot associate with a prior function call.

Only item 4 explains the browser failure at `17:32:32`.

## Recommended implementation direction

Introduce a protocol-aware history projection boundary before backend serialization.

Required properties:

1. Canonical session JSONL remains unchanged and continues to preserve full tool traces for audit, recovery, UI trace, and learning.
2. Provider prompt history is a derived projection, not a literal copy of the canonical execution transcript.
3. Only the currently active tool continuation needs native `assistant.tool_calls` and `role=tool` protocol.
4. Completed tool blocks from earlier turns should be converted into provider-safe conversational/evidence context for generic Chat Completions backends, or replayed through an explicitly supported provider capability.
5. Pair validation must operate on semantic tool-call blocks, not ID-prefix or hostname/model heuristics.
6. Never retain a `role=tool` message without its matching assistant tool call in a native replay block.
7. Preserve useful tool result meaning so the model can answer follow-up questions without re-running old tools unnecessarily.

Suggested projection behavior for generic Chat Completions:

- Identify complete historical blocks: one assistant message containing one or more tool calls followed by the corresponding tool results.
- For blocks belonging to completed earlier turns, render a bounded textual transcript/evidence representation without native `tool_call_id` fields.
- Preserve current-turn native blocks unchanged while ReAct is waiting for or consuming their tool results.
- If an explicit backend capability later guarantees durable native cross-turn replay, allow it through a closed-set construction-time policy similar to the Responses continuity policy.

The exact flattening role/text format should be centralized and deterministic. Do not scatter provider conditions through Engine, ReAct, and session storage.

## Tests the next agent should add first

Start with failing tests before implementation.

### Backend serializer regression

Target:

```text
tests/test_openai_compat_backend.py
```

Fixture:

1. Prior user message.
2. Prior assistant native tool call `call-old`.
3. Matching prior tool result `call-old`.
4. Prior assistant final answer.
5. New user follow-up.

Assert for the safe generic Chat Completions projection:

- completed old tool information remains available as text/evidence;
- no old `role=tool` message is emitted without a provider-compatible matching call;
- no `call_`, `fc_`, hostname, or model-name rewrite heuristic is used;
- current-turn native tool calls still serialize normally.

### Engine multi-turn regression

Potential targets:

```text
tests/unit/engine/test_sessions_and_events.py
tests/unit/engine/test_react_loop.py
```

Scenario:

1. First turn uses multiple tools and returns a final answer.
2. Second turn is ordinary conversation or a different research question.
3. Capture the first backend payload of the second turn.
4. Verify the payload has portable projected history and the backend can return a final answer.

The existing `test_restore_session_history_preserves_tool_messages_and_responses_replay` only proves canonical restoration. It does not prove that literally replaying those restored messages is safe for a generic provider.

### Pair-integrity regression

Add cases for:

- one tool call and one result;
- one assistant message with multiple tool calls and results;
- missing tool result;
- orphan tool result;
- completed prior turn plus active current-turn tool continuation;
- compaction/history-window boundary near a tool block.

## Runtime acceptance test

After focused tests pass and the backend is restarted:

1. Open a fresh conversation using `gpt-5.6-luna` through `cdn.coderelay.cn`.
2. First prompt should force at least one read-only tool call, for example a literature search.
3. Wait for a successful final answer.
4. In the same conversation, send a normal follow-up that does not mention time.
5. Confirm there is no `400 Bad Request` and no `No tool call found for function call output` log entry.
6. Send a third follow-up to confirm the repair is stable across more than one boundary.
7. Inspect the JSONL and verify canonical tool messages are still preserved.

Also rerun a fresh time-tool request only as a non-regression check; it is not the primary reproducer for this defect.

## Acceptance criteria

- A tool-using first turn followed by a second ordinary turn succeeds through Coderelay/Luna Chat Completions.
- Historical tool evidence remains semantically available to the model.
- Canonical session/tool traces are not deleted or rewritten.
- Current-turn native tool execution still works.
- Responses continuity tests remain green.
- No provider hostname, model-name, natural-language error substring, or `call_/fc_` prefix heuristic controls behavior.
- Timeline revision validation remains unchanged.

## Suggested focused verification commands

```powershell
rtk pytest -q tests/test_openai_compat_backend.py
rtk pytest -q tests/unit/engine/test_sessions_and_events.py tests/unit/engine/test_react_loop.py
rtk pytest -q tests/backends/test_openai_compat.py tests/test_openai_codex_backend.py
rtk git diff --check
```

Run broader integration tests only after the focused serializer and multi-turn cases pass.

## Working-tree safety

- Do not run `git reset --hard`, `git checkout --`, or broad cleanup commands.
- Inspect the existing diffs in `mochi/backends/openai_compat.py`, `mochi/backends/types.py`, `mochi/agents/engine.py`, and their tests before editing; they contain earlier continuity and adaptive-runtime work.
- Keep this defect's delta narrowly attributable in the dirty worktree.
- Follow the repository instruction to prefix shell commands with `rtk`.

## First action for the next agent

Create a deterministic failing payload-level test from the exact line-61/line-62 history pair, then decide the narrowest centralized projection seam. Do not begin by changing stored IDs, session JSONL, timeline logic, the time-zone tool, or Codex policy handling.
