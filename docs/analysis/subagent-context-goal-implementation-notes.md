# Context / Goal Implementation Notes

Date: 2026-07-05
Scope: follow-up review notes for context truncation, effective context length, and goal carryover behavior. This replaces the corrupted subagent note file with a clean UTF-8 handoff.

## Executive Summary

The most urgent runtime gaps are not only model quality issues. The current implementation already has useful compaction and context snapshot primitives, but several runtime edges can still make weaker local/Ollama models appear to "forget" tasks or stop mid-output:

1. Tool-call parsing and raw final-output rescue should be fixed first, because malformed or non-native tool markup currently prevents real agent behavior.
2. Workspace write tools must be available in a stable, discoverable way, but read-only attachment workflows should not directly expose write tools unless user intent is clearly mutating.
3. Context length must be treated as the effective runtime window, not just the model's theoretical max context.
4. `finish_reason=length` should be promoted into a visible retry/continuation signal instead of silently accepting a truncated assistant message.
5. Goal resume/carryover should inject concrete previous-attempt state into normal continuation runs, not only into explicit resume summaries.

## Verified Source Anchors

- `mochi/backends/ollama.py` tracks `configured_num_ctx`, `runtime_context_length`, `model_max_context_length`, and exposes them through `ModelInfo.metadata`. The effective `context_length` is updated after `/api/show` metadata extraction and after applying configured `num_ctx`.
- `mochi/agents/engine.py` uses `_snapshot_context_length(model_info)` for chat context snapshots and falls back to `4096` when no reliable context length is available.
- `mochi/agents/engine.py` already derives auto output/reserve tokens from `_auto_inference_context_length_hint()`, which prefers reliable model info, then configured Ollama `num_ctx`, GGUF `n_ctx`, and vLLM `max_model_len`.
- `mochi/agents/context.py` calls `_compact_history_if_needed()` before prompt assembly and can expose compaction diagnostics in prompt context.
- `mochi/agents/compaction.py` supports semantic and legacy compaction modes with trigger/retain settings and token-budget-aware compaction reasons.

## Gap 1: Truncated Output Handling

Observed symptom: after a second user request, output can be cut off; retrying produces code, but the agent does not reliably continue the same action or save files.

Likely runtime root cause:

- The backend result can carry a finish reason such as `length`, but the agent loop needs a first-class branch that treats this as incomplete output.
- If the final assistant content contains partial code or partial tool markup, accepting it as normal assistant text causes a bad UX: the user sees a truncated answer instead of an automatic continuation or repair attempt.

Recommended implementation:

1. In the main non-stream and streaming completion paths, detect `finish_reason == "length"` or provider-specific equivalent truncation metadata.
2. Emit a structured event such as `assistant_truncated` or mark the assistant message metadata with `truncated=true` and `finish_reason=length`.
3. If tool execution is still expected, retry once with a continuation prompt that includes the partial output and asks for only the missing suffix or valid tool call.
4. If final answer text is truncated, ask the model to continue from the last complete sentence/code fence boundary rather than re-answer from scratch.
5. Store truncation diagnostics in session history so later goal continuation knows the previous attempt ended due to length, not because the task was completed.

Acceptance tests:

- Simulate a backend result with `finish_reason="length"` and partial assistant text; assert the loop does not mark the task as cleanly complete.
- Simulate partial `<tool_call>` or Qwen XML tool markup at the end of a length-limited response; assert parser rescue or retry happens before final text is shown.

## Gap 2: Effective Context Length

Observed symptom: local/Ollama/GGUF runs can forget earlier context even when the model's advertised max context is large.

Likely runtime root cause:

- The practical context window is the smaller of model max, runtime `num_ctx`/`n_ctx`, backend caps, and configured output reserve.
- If snapshots and compaction use model max instead of runtime-effective context, prompts can be too large before compaction triggers.

Recommended implementation:

1. Introduce one explicit `effective_context_length` helper for every backend.
2. For Ollama, prefer configured `num_ctx` if present; otherwise prefer runtime context extracted from `/api/show`; only then use model max; final fallback remains conservative (`4096`).
3. Expose `effective_context_length_source` in `ModelInfo.metadata` and context snapshot output.
4. Use this effective value for compaction thresholds, prompt usage ratio, auto max output tokens, and UI diagnostics.
5. Add warnings when effective context is unknown or suspiciously smaller than model max.

Acceptance tests:

- Ollama with `num_ctx=8192` and model max `131072` should report/use `8192` as effective.
- Ollama without `num_ctx` but with runtime metadata should use runtime context.
- Unknown runtime metadata should fall back to `4096` and mark the source as fallback.

## Gap 3: Goal Carryover

Observed symptom: after a failed/truncated goal attempt, a restarted request can behave as if it forgot prior intent or previous tool progress.

Likely runtime root cause:

- Goal guidance and attempt summaries are not guaranteed to be injected into every continuation path.
- Previous tool-call progress and incomplete output diagnostics are not represented as structured carryover state.

Recommended implementation:

1. Define a compact `GoalCarryoverState` containing original user objective, latest explicit user request, completed tool calls, pending next action, failure/truncation reason, and relevant file paths.
2. Inject carryover into the next goal turn as a system/developer-level runtime note, not only as user-visible prose.
3. Keep the state short and structured so weaker models can follow it.
4. Clear or archive carryover only after a verified completion signal, not after any assistant text response.
5. Include final raw tool markup rescue: if the assistant outputs code or tool syntax in final text, the runtime should detect whether it was intended as a file write/tool call.

Acceptance tests:

- Start a goal, simulate a `length` stop after partial code; next turn should include original goal and pending action.
- Simulate final text containing a valid tool-call block; runtime should execute or surface it as a recoverable tool-call event instead of plain text.
- A greeting or unrelated message should not accidentally activate pending proposal capture or goal carryover mutation.

## Gap 4: Event Taxonomy

Current event streams appear to mix model text, tool calls, and goal lifecycle concerns more than mature agent runtimes do.

Recommended event separation:

- `message_delta`: assistant text only.
- `tool_call_created`: parsed tool call with stable id/name/args.
- `tool_call_delta`: partial tool args while streaming.
- `tool_call_completed`: final parsed tool call payload.
- `tool_result`: result payload and status.
- `assistant_truncated`: finish reason indicates output was incomplete.
- `context_snapshot`: prompt budget/compaction metadata.
- `goal_state_changed`: goal active/paused/completed/blocked.

This separation makes the UI and session store less dependent on brittle text parsing.

## Implementation Priority

1. Complete and merge the multi-format tool-call parser plus final raw markup rescue.
2. Stabilize workspace write-tool exposure: always discoverable in workspace sessions; visible only when clearly needed.
3. Add `finish_reason=length` handling and retry/continuation behavior.
4. Normalize effective context length and surface it in snapshots/UI.
5. Add structured goal carryover state and tests for truncated attempts.
6. Refactor event taxonomy after the above behavior is stable.

## Notes For Future Subagents

Do not treat the example failures as only "weak model" failures. Weaker models expose runtime design gaps faster, especially around malformed tool syntax, context-budget overrun, and insufficient state carryover. The fix should make the runtime more deterministic before trying to solve this through prompt wording alone.
