# agent_mochi 參考成熟 Agent Runtime 的修正計劃（2026-07-05）

> 目的：根據 `reference/` 內的 Claude Code Haha、Hermes Agent、OpenClaw、ZeroClaw、DrZero 參考資料，整理 `agent_mochi` 下一步修正方案。
> 讀取範圍：主要查閱 `reference/cc-haha`、`reference/hermes-agent`、`reference/openclaw`、`reference/zeroclaw`、`reference/drzero` 中與 context、tool calling、permission/sandbox、file write/diff、resume/memory 有關的實作與文件。
> 結論：不要照抄任一專案；要抽取成熟 agent runtime 的共通契約，補到 mochi 的 runtime 邊界上。

---

## 1. 參考專案觀察摘要

### 1.1 Claude Code Haha / Claude Code 類實作

已讀到的關鍵點：

- `reference/cc-haha/README*.md` 明確把桌面端定位成「sessions、多 project、worktree、code diff、permission review、token usage」整合工作台。
- `reference/cc-haha/desktop/src/components/chat/PermissionDialog.tsx` 對 `Bash`、`Edit`、`Write` 等 risky tools 做明確 permission UI；`Write` 與 `Edit` 都會顯示 diff preview。
- `reference/cc-haha/desktop/src/components/chat/DiffViewer.tsx`、`CurrentTurnChangeCard.tsx` 代表「寫入不是隱藏能力，而是可視化變更」。
- `reference/cc-haha/desktop/src/components/chat/ContextUsageIndicator.tsx` 會顯示 live/estimate context usage，且 compaction 後用 `refreshNonce` 強制刷新，避免 UI 留在舊 token 百分比。
- `reference/cc-haha/src/services/compact/autoCompact.ts` 把 context window 減去 reserved output tokens 後得到 effective context window；再依 warning/error/autocompact/blocking thresholds 決定是否壓縮或阻擋。
- `reference/cc-haha/src/context.ts` 會把 git status、CLAUDE.md / memory files 等 session context 放入 prompt，但有長度限制與 cache。
- `reference/cc-haha/src/tools/FileWriteTool/*`、`FileEditTool/*`、`BashTool/*` 是常駐工具能力，由 permission / sandbox / validation 控制風險，而不是靠自然語言 routing 隱藏工具。

對 mochi 的啟示：

1. context budget 必須是 runtime hard gate，不只是 UI snapshot。
2. 寫檔工具應穩定存在，風險由 permission dialog、diff preview、sandbox policy 控制。
3. compaction 後要更新 UI context meter，避免使用者看到錯誤的「仍快爆 context」狀態。
4. session context 應有明確來源、長度限制、cache invalidation，而不是每條路徑各自組 prompt。

### 1.2 Hermes Agent

已讀到的關鍵點：

- `reference/hermes-agent/README.md` 強調跨平台 conversation continuity、session history search、memory、skills、自我改善、parallel delegates。
- `reference/hermes-agent/toolsets.py` 把 `_HERMES_CORE_TOOLS` 集中定義，核心工具包含 `read_file`、`write_file`、`patch`、`search_files`、`terminal`、`memory`、`session_search`、`clarify`、`execute_code`、`delegate_task`。
- `reference/hermes-agent/toolsets.py` 的 `file` toolset 明確是 `read_file/write_file/patch/search_files`，不是 read-only first。
- `reference/hermes-agent/model_tools.py` 會根據實際可用工具重建 schema，例如 `execute_code` schema 只列出真的可用 sandbox tools，避免模型看到不可用工具後 hallucinate。
- `reference/hermes-agent/model_tools.py` 有 `coerce_tool_args()`，會依 JSON Schema 把 `"42"`、`"true"` 這類弱模型常見字串參數轉成 integer/boolean/number。
- `reference/hermes-agent/hermes_state.py` 用 SQLite 儲存 sessions/messages，messages 表包含 `tool_calls`、`tool_name`、`token_count`、`finish_reason`、`reasoning`，並有 FTS5 搜尋。
- `reference/hermes-agent/trajectory_compressor.py` 有明確 compression config：target token、summary token、protected first system/human/gpt/tool、protect last N turns、summary notice、metrics。

對 mochi 的啟示：

1. toolset 應是集中、可驗證的能力集合；動態 filtering 只能移除確實不可用的工具，不能讓模型看到矛盾 schema。
2. weak/local model 需要 tool args coercion，不應把格式小錯全部當 invalid tool call。
3. session history 應可搜尋、可回填、可跨 run carry-over；Goal resume 不能只靠 guidance message。
4. compaction 要有 protected turns 和 metrics，不能只做固定 history window。

### 1.3 OpenClaw

已讀到的關鍵點：

- `reference/openclaw/README.md` 與 `docs.acp.md` 顯示它重視 ACP session/event protocol、tool streaming、usage、compact、status、trace。
- `reference/openclaw/docs.acp.md` 區分 `tool_call` / `tool_call_update` events，並提到 raw I/O、text content、best-effort file info。
- `reference/openclaw/Dockerfile.sandbox`、`Dockerfile.sandbox-common` 顯示它把 sandbox 當成獨立 runtime 環境，而不是只靠 prompt 約束。
- `reference/openclaw/SECURITY.md` 明確把 approval、allowlist、sandbox 視為安全邊界，並要求報告要指出具體 boundary bypass。
- `reference/openclaw/AGENTS.md` 中提到 truncation/compaction 時要優先維持 cached prefix，避免破壞 prompt cache。

對 mochi 的啟示：

1. event protocol 必須區分 tool streaming、runtime steering、tool error、permission request、compaction boundary。
2. sandbox 是 runtime 執行隔離，不應用「不把工具給模型」代替。
3. context/compaction 需要考慮 prompt cache prefix 穩定性，不要每次重排整個 prompt。

### 1.4 ZeroClaw

已讀到的關鍵點：

- `reference/zeroclaw/README.md` 標示支援 OpenAI Codex、Claude Code 等外部 CLI / provider。
- `reference/zeroclaw/src/tools/codex_cli.rs`、`opencode_cli.rs`、`claude_code.rs` 是對下層 tool crate 的 thin re-export，表示 CLI integration 被當成工具能力。
- `reference/zeroclaw/benches/agent_benchmarks.rs` 有 XML `<tool_call>` parsing benchmark，包括 single / multi tool call。
- `reference/zeroclaw` 有 `file_write`、`tool_search`、`workspace_tool`、memory store/recall benchmark 與 REST API memory 文件。
- `reference/zeroclaw` 的 benchmark 把 XML tool call parsing 當成性能與正確性測試對象，說明這不是 edge case。

對 mochi 的啟示：

1. Qwen/XML-ish tool call 不應視為模型亂輸出；成熟本地 agent 會把 XML parser 當正式 protocol fallback。
2. external CLI integration 可以作為工具，但需要明確 wrapper contract，不應把所有 CLI 行為混進主 loop。
3. memory / tool parsing / session cycle 都應有 benchmark 或至少 fixture tests。

### 1.5 DrZero

已讀到的關鍵點：

- DrZero 偏 RL / trajectory / rollout，不是桌面 coding agent runtime。
- 對 mochi 最有用的是 trajectory、tool interaction、search environment、token/context 資料管線，而不是 permission 或 file write UI。

對 mochi 的啟示：

1. 如果未來要做 agent 行為評測或弱模型修正資料集，可參考 DrZero 的 trajectory/rollout 思路。
2. 不應把 DrZero 當作近期 runtime 修 bug 的主要參考。

---

## 2. 成熟 Agent Runtime 的共通契約

### 2.1 能力表穩定，安全由 policy 控制

成熟 agent 不會讓模型在不同輪次突然失去寫檔工具。比較合理的分層是：

1. **Capability registry**：有哪些工具，schema 是什麼。
2. **Availability filter**：依環境檢查工具是否真的可用，例如 API key、sandbox backend、workspace 是否存在。
3. **Policy layer**：是否需要 approval、是否允許自動執行、是否只能讀取。
4. **UI layer**：diff preview、permission dialog、command preview。
5. **Execution layer**：sandbox / filesystem scope / allowlist / timeout。

mochi 現況把第 2、3、4 層部分混進 `tool_exposure.py` 的自然語言 routing，導致模型能力不穩。

### 2.2 Context budget 是硬閘門

成熟 runtime 共同特徵：

- 有 effective context window。
- 保留 output reserve。
- 有 warning/error/autocompact/blocking threshold。
- compaction 前後會更新 UI 狀態。
- protected turns 不被壓縮，例如 system、第一個 user、第一個 tool、最近 N turns。

mochi 現況已有 `ChatContextSnapshot`、`ContextBudget`、`compaction.py`，但缺少「組 prompt 前不得超過 effective context」的硬閘門。

### 2.3 Tool protocol 必須多格式、可修復、可觀測

成熟 runtime 對工具調用通常會有：

- 原生 function calling。
- 模擬 function calling。
- XML / JSON / provider-specific fallback parser。
- invalid tool call repair middleware。
- tool args coercion。
- invalid tool telemetry。

mochi 現況的 `ToolCallSimulator` 只支援 JSON `<tool_call>` payload，對 Qwen XML 風格不夠。

### 2.4 Session / Goal resume 必須帶上下文

成熟 runtime 不把 resume 當成單一 guidance string：

- Hermes 用 SQLite 保存完整 session/message/tool/reasoning/finish_reason，並有 FTS5 session search。
- Claude Code 類工具會保存 session context、memory files、git status、turn state。
- compaction 是 session chain，不是清空上下文。

mochi Goal resume 現況更像把新使用者指令塞進 `guidance_messages`。這對「繼續剛剛」和「保存剛剛的程式碼」不夠。

### 2.5 Event protocol 要區分語義

成熟 runtime 區分：

- tool call started / updated / finished
- permission requested / approved / denied
- runtime steering
- compaction started / completed
- backend error
- tool error
- output truncated

mochi 現況有事件與 timeline 雛形，但 evidence guard / tool error / agent error 的語義仍混雜。

---

## 3. mochi 修正總策略

不要先大改 UI 或拆巨型檔。先用 5 條 runtime contract 把失控面補住：

1. **Tool Protocol Contract**：所有工具調用先可解析、可修復、可觀測。
2. **Capability & Permission Contract**：workspace agent 穩定看得到寫檔能力，執行前由 policy/approval 控制。
3. **Context Budget Contract**：每輪 prompt 不能超過 effective context；`finish_reason=length` 不能當成功。
4. **Goal Continuity Contract**：Goal resume/follow-up 必須攜帶最近 transcript、code blocks、artifacts。
5. **Event Semantics Contract**：runtime steering、tool error、permission、truncation 必須不同事件類型。

---

## 4. 第一批修正：Tool Protocol Contract

### 4.1 新增 parser chain

目標檔案：

- `mochi/backends/tool_call_simulator.py`
- 新增 `mochi/backends/tool_call_parsers.py`
- 測試：`tests/backends/test_tool_call_parsers.py`

設計：

```text
parse_tool_calls(text):
  1. Extract <tool_call> blocks
  2. Try JSON object/list parser
  3. Try OpenAI-like JSON {"function": ...}
  4. Try Qwen XML parser
  5. Try lenient key-value repair parser
  6. Return parsed calls + diagnostics
```

Qwen XML parser 至少支援：

```xml
<tool_call>
  <function=arxiv_search>
    <parameter=query>medical imaging</parameter>
    <parameter=max_results>5</parameter>
  </function>
</tool_call>
```

也支援單行：

```xml
<tool_call> <function=arxiv_search> <parameter=query>medical imaging</parameter> </function> </tool_call>
```

參數 coercion 規則參考 Hermes：

- `"true"` / `"false"` → bool
- `"123"` → int
- `"1.5"` → float
- JSON-looking string → 嘗試 JSON parse
- 其他保留 string

### 4.2 final answer 前救援 raw tool markup

目標檔案：

- `mochi/agents/react_loop.py`

規則：

```python
if final_content contains "<tool_call" or "<function=":
    parsed = tool_parser.parse_tool_calls(final_content)
    if parsed:
        convert to normal tool call flow
    else:
        run one repair turn
```

驗收：

- 範例2 的 raw XML tool call 不再出現在 UI final answer。
- parser 能解析 single / multi XML tool calls。
- invalid XML 只 repair 一次，不無限循環。

---

## 5. 第二批修正：Capability & Permission Contract

### 5.1 重寫 `tool_exposure.py` 的責任邊界

目標檔案：

- `mochi/agents/tool_exposure.py`
- `mochi/config/schema.py`
- approval/security 相關檔案
- web permission / diff UI

新的責任切分：

| 層級 | 責任 | 不該做的事 |
|---|---|---|
| registry | 所有工具 schema、capability metadata | 根據 user text 猜 intent |
| availability | API key、workspace、backend 是否可用 | 決定是否安全 |
| exposure | 控制排序與 prompt budget | 隱藏 workspace 核心能力 |
| policy | approval、sandbox、scope、allowlist | 改寫工具 schema |
| UI | permission dialog、diff preview | 決定工具是否存在 |

### 5.2 workspace-bound 常駐工具集合

workspace-bound session 最小工具集合：

```python
CORE_WORKSPACE_READ = (
    "file_read",
    "grep_search",
    "glob_search",
    "tool_result_read",
)

CORE_WORKSPACE_WRITE = (
    "file_write",
    "file_edit",
    "apply_patch",
)

CORE_WORKSPACE_DISCOVERY = (
    "tool_search",
)
```

規則：

- workspace-bound session 一律 expose read + at least one write path。
- `file_write` / `apply_patch` 不因 greeting 或 intent miss 消失。
- read-only user request 可以加入 system steering：「不要寫入」，但不應讓能力表在下一輪失真。
- strict mode 下可以讓 write tools `requires_approval=True`，但不要從 tools list 消失。

### 5.3 approval + diff preview

參考 Claude Code Haha：

- `Write` 顯示新檔 diff：old empty → new content。
- `Edit` 顯示 old/new diff。
- `Bash` 顯示 command preview。

mochi 需要：

- `file_write` approval payload 包含 path、size、new content preview、full diff artifact。
- `file_edit/apply_patch` approval payload 包含 affected files、diff、risk label。
- UI 用不同 badge 區分：
  - read-only
  - file mutation
  - command execution
  - network

驗收：

- 「幫我寫訓練程式並保存」第一輪工具表包含 `file_write` 或 `apply_patch`。
- 若 policy 需要 approval，UI 出現 permission/diff，不是模型回「我不能寫檔」。

---

## 6. 第三批修正：Context Budget Contract

### 6.1 建立 `effective_context_length`

目標檔案：

- `mochi/backends/types.py`
- `mochi/backends/ollama.py`
- `mochi/agents/engine.py`
- `mochi/agents/context.py`
- `mochi/agents/compaction.py`

設計：

```python
effective_context_length = min(
    configured_context_length or serving_context_length or conservative_default,
    model_advertised_context_length or infinity,
)
```

Ollama 特例：

- `num_ctx` configured → effective = configured
- `auto_num_ctx=True` → request options 明確送 `num_ctx`
- unknown server default → effective 用 4096 conservative，而不是信 model max

### 6.2 prompt 組裝前 hard gate

規則：

```text
available_input = effective_context_length - reserve_output_tokens
if estimated_prompt_tokens > available_input:
    compact
if still too large:
    summarize tool results
if still too large:
    trim oldest non-protected history
if still too large:
    return context_overflow runtime error
```

protected turns：

- system prompt
- current user request
- most recent assistant tool calls/results
- first goal objective / root task
- last N turns
- unresolved permission/tool state

### 6.3 `finish_reason=length` recovery

目標檔案：

- `mochi/agents/react_loop.py`
- web event rendering

規則：

1. `finish_reason=length` 不可視為 success final。
2. 若 partial output 有內容：continuation 最多 2 次。
3. 若多次 length：compaction retry 1 次。
4. 仍失敗：回 `output_truncated` runtime error。

UI：

- 顯示「輸出被模型 context/output limit 截斷」。
- 顯示 input/output token、effective context、reserve output。

---

## 7. 第四批修正：Goal Continuity Contract

### 7.1 Goal resume 不只帶 guidance

目標檔案：

- `mochi/runtime/service.py`
- agent run package 組裝處
- session store / chat transcript store

新增資料結構：

```python
GoalCarryoverContext:
    source_session_id
    recent_user_messages
    recent_assistant_messages
    recent_code_blocks
    recent_artifact_refs
    recent_tool_results_summary
    token_count
```

規則：

- `resume_goal(guidance_message=...)` 建新 attempt 時，自動注入 carryover。
- guidance 放在「使用者新增指示」位置，不取代 history。
- 若 user text 包含「剛剛」「上一段」「你剛寫的」「保存它」，優先帶最近 assistant code block。

### 7.2 session search / memory 回填

參考 Hermes：

- sessions/messages 用可查詢儲存，包含 finish_reason、tool_calls、reasoning、token_count。
- FTS5 session search 可找過去對話。

mochi 可先做較小版本：

- 在現有 store 中建立 `recent_assistant_code_blocks(session_id, limit=3)`。
- 建立 `recent_goal_relevant_turns(goal_id, session_id, token_budget)`。
- 等資料穩定後再加 FTS / semantic search。

驗收：

- 第一輪 assistant 產生 code，第二輪 user 說「保存剛剛的程式碼」，新 run prompt 有 code block 或 artifact ref。
- 模型不再說「我看不到之前內容」。

---

## 8. 第五批修正：Event Semantics Contract

### 8.1 定義 runtime event taxonomy

新增或整理 enum：

```python
RuntimeEventKind = Literal[
    "tool_call_request",
    "tool_call_update",
    "tool_call_result",
    "permission_request",
    "permission_result",
    "runtime_steering",
    "compaction_started",
    "compaction_completed",
    "context_warning",
    "context_overflow",
    "output_truncated",
    "invalid_tool_call",
    "backend_error",
    "tool_error",
    "agent_final",
]
```

### 8.2 evidence guard 不再偽裝 tool failure

現況問題：

- 「證據已足夠，請綜合作答」如果以紅色 tool error 呈現，弱模型會誤解成工具失敗，可能繼續重試。

改法：

- 回 `runtime_steering` event。
- 對模型注入 steering message。
- UI 顯示中性色或藍色提示，不顯示 failed。

### 8.3 tool streaming / diff / permission 事件分離

參考 OpenClaw ACP：

- `tool_call`
- `tool_call_update`
- file info / raw IO / text content

mochi 應加：

- file mutation preview event
- permission pending event
- permission approved/denied event
- final applied diff event

---

## 9. Legacy / pending proposal 整理策略

目前判斷：

- `GoalCard` UI 已刪，可以維持刪除。
- `pending_proposal` 還在 workflow confirmation path，不能直接刪。
- 問題是命名造成誤解。

策略：

1. 短期：保留 wire format `pending_proposal`，避免破壞已存 session。
2. 中期：程式內 alias 成 `workflowPendingProposal` / `goalConfirmationDraft`。
3. 長期：只有 workflow setup/confirmation 路徑能建立 pending proposal；一般 `/goal objective` 不走 pending proposal。
4. legacy helper 移到 `mochi/compat/goal_proposal.py` 或 `mochi/legacy/goal_card_compat.py`，主流程不得直接 import legacy UI copy。

驗收：

- grep `GoalCard|goalCard|goal_card` 為空。
- greeting 不建立 pending proposal。
- `/goal <objective>` 直接啟動 autonomous objective。
- workflow explicit setup 仍可 revision / confirmation。

---

## 10. 建議落地順序

### Sprint 1：最小可見修復

1. Qwen XML parser + tool args coercion。
2. final answer raw `<tool_call>` rescue。
3. workspace-bound 常駐暴露 write tools。
4. approval payload 加 file diff metadata。

原因：

- 直接修範例2。
- 直接修「會產生程式碼但不能保存」。
- 修改範圍小，測試可單元化。

### Sprint 2：context / truncation

1. `effective_context_length`。
2. Ollama `auto_num_ctx`。
3. prompt hard gate。
4. `finish_reason=length` continuation / compaction retry。
5. UI context warning / truncation event。

原因：

- 直接修範例1截斷。
- 需要跨 backend/engine/react_loop，風險高於 Sprint 1。

### Sprint 3：Goal continuity

1. Goal carryover context。
2. recent assistant code block extraction。
3. guidance + transcript 合併策略。
4. resume integration tests。

原因：

- 修「保存剛剛」和 goal follow-up 失憶。
- 需要理解 store/session/run package，應等前兩批穩定後做。

### Sprint 4：event taxonomy / UI

1. runtime event enum。
2. evidence guard steering event。
3. permission/diff/tool streaming UI 統一。
4. context meter compaction refresh。

原因：

- 提升弱模型與使用者可觀測性。
- UI 更動較多，適合在 runtime 行為穩定後整理。

### Sprint 5：架構拆分

1. `react_loop.py` 拆出 `tool_call_repair.py`、`generation_recovery.py`、`runtime_steering.py`。
2. `tool_exposure.py` 拆成 registry/availability/exposure/policy。
3. `service.py` 拆 goal lifecycle / run resume / event service。
4. `page.tsx` 拆 hooks。

原因：

- 現在直接拆會混入行為變更，不利驗證。
- 應在 P0/P1 行為已有測試後拆。

---

## 11. 具體任務卡

### Task 1：`ToolCallSimulator` 多格式 parser

- 改：`mochi/backends/tool_call_simulator.py`
- 新增：`mochi/backends/tool_call_parsers.py`
- 新增測試：
  - JSON single call
  - JSON multi call
  - Qwen XML single call
  - Qwen XML multi call
  - malformed XML repair diagnostics
  - string int/bool coercion

Done：

- 範例2 XML fixture 可解析成正常 `ToolCall`。
- 原 JSON 行為不退化。

### Task 2：final answer tool markup rescue

- 改：`mochi/agents/react_loop.py`
- 使用 Task 1 parser。

Done：

- final content 含 `<tool_call>` 時不直接顯示。
- 可解析就執行工具。
- 不可解析只 repair 一次。

### Task 3：workspace write tools 常駐暴露

- 改：`mochi/agents/tool_exposure.py`
- 新增：`CORE_WORKSPACE_WRITE_TOOLS`
- 調整 risky tool 邏輯：risky 影響 approval，不影響存在性。

Done：

- workspace-bound +「寫/保存/修改」暴露 `file_write/apply_patch`。
- workspace-bound + greeting 不建立 pending proposal，也不破壞工具 discoverability。

### Task 4：permission / diff metadata

- 改：後端 approval payload + web permission UI。

Done：

- `file_write` approval 顯示新檔 diff。
- `apply_patch/file_edit` approval 顯示 affected files + diff。
- denial 會回模型明確 `tool_denied`，不是 generic error。

### Task 5：effective context + Ollama auto num_ctx

- 改：`mochi/backends/ollama.py`、`mochi/backends/types.py`、`mochi/agents/engine.py`。

Done：

- `num_ctx=None` 時不再盲信 model max。
- request metadata 和 context snapshot 都顯示 effective context source。
- 超過 input budget 時不送出 LLM request。

### Task 6：finish_reason length recovery

- 改：`mochi/agents/react_loop.py`

Done：

- `finish_reason=length` 不被當成功。
- 可 continuation。
- 多次失敗回 structured `output_truncated`。

### Task 7：Goal carryover context

- 改：`mochi/runtime/service.py` + run package builder。

Done：

- resume/follow-up prompt 包含最近 transcript summary。
- 「保存剛剛的程式碼」包含最近 code block/artifact ref。
- guidance message 不覆蓋 history。

### Task 8：event taxonomy

- 改：runtime event model + web renderer。

Done：

- evidence guard 是 `runtime_steering`。
- tool failure 是 `tool_error`。
- truncation 是 `output_truncated`。
- permission 是 `permission_request/result`。

---

## 12. 驗證矩陣

| 場景 | 修正前 | 修正後驗收 |
|---|---|---|
 Ollama 4096 context 第二輪長 prompt | output 被截斷但像完成 | hard gate 先 compact，或 length recovery |
 Qwen GGUF XML tool call | raw `<tool_call>` 顯示在 final | 解析成工具調用或 repair |
 使用者要求保存程式碼 | 工具表只有 read tools | `file_write/apply_patch` 可用，必要時 approval |
 goal follow-up「剛剛」 | 模型說看不到前文 | prompt 有 carryover code/context |
 evidence sufficient guard | 顯示 tool failed | 顯示 runtime steering |
 pending proposal | 易被誤會是 GoalCard legacy | 只保留 workflow confirmation 語義 |

---

## 13. 不建議的做法

1. 不要先大拆 `service.py` / `page.tsx`。
   先補 runtime tests，否則拆分會把行為變更藏進搬家 diff。

2. 不要把所有 tool 都無條件塞進 prompt。
   正確做法是核心工具常駐，其他工具可 discovery；危險操作由 approval 控制。

3. 不要只調 prompt 嘗試讓 Qwen 輸出 JSON。
   本地/量化模型長 context 後仍會回到訓練格式，所以 parser fallback 必須存在。

4. 不要把 pending proposal 直接刪光。
   它仍支撐 workflow confirmation；應先改命名與路徑邊界。

5. 不要把 `finish_reason=length` 顯示成一般完成。
   這會讓使用者和模型都誤判任務狀態。

---

## 14. 最終建議

第一個實作分支只做三件事：

1. `ToolCallSimulator` parser chain。
2. final answer raw tool markup rescue。
3. workspace-bound 常駐 write tools。

這三件事可以用單元測試覆蓋，改動面小，且直接對應目前最明顯的兩類失敗：`範例2` 的 tool call 跑掉，以及 `範例1` 的「會寫程式但不能保存」。完成後再進入 context hard gate 與 goal carryover，風險最低。
