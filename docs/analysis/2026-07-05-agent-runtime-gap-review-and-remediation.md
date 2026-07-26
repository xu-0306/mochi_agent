# agent_mochi Goal / Agent Runtime 現況複核與修正計畫（2026-07-05）

> 範圍：`mochi/` runtime、`web/` goal/chat 前端，以及使用者提供的 `範例1`、`範例1/截斷後第2次測試`、`範例2` 截圖證據鏈。
> 目的：複核另一個 agent 產出的 `docs/analysis/2026-07-05-agent-runtime-gap-analysis.md`，避免被它的結論帶偏；同時給出足夠細的修正項，讓較弱模型也能逐項實作。

---

## 1. 結論摘要

前一份分析的大方向是對的：三個範例暴露出的問題不應歸咎為「模型太弱」而已，主要是 runtime harness 還缺少成熟 agent 系統必備的邊界控制、錯誤語義與上下文管理。

我複核後的判斷如下：

1. **範例1輸出截斷：高度確認是 context window / finish reason 管理缺口。**
   `mochi/config/schema.py` 的 `num_ctx` 預設為 `None`，註解明確表示保留 Ollama server default；`mochi/backends/ollama.py` 只有在 `_configured_num_ctx is not None` 時才送 `options["num_ctx"]`。同時 `mochi/agents/engine.py` 目前只產生 context snapshot，沒有在 prompt 組裝前強制對齊實際 serving window。`mochi/` 中也沒有找到 `finish_reason == "length"` 的處理分支。

2. **範例1「不能保存程式碼」：高度確認是工具暴露策略造成的能力剝奪。**
   `mochi/agents/tool_exposure.py` 目前以 intent / keyword / matched group / risky limit 來選工具。`file_write`、`file_edit`、`apply_patch` 被列為 risky tools，不是 workspace-bound session 的常駐能力。這會讓模型在需要寫檔時只看到唯讀工具，造成「我不能建立檔案」的錯誤行為。成熟 agent harness 通常不靠隱藏工具保護使用者，而是靠 sandbox、approval、diff preview 和權限策略保護寫入。

3. **範例1「忘記上下文」：目前證據支持 goal resume / attempt 的上下文注入不足。**
   `mochi/runtime/service.py:1307-1355` 的 `resume_goal()` 會把使用者續跑文字放進 `summary["guidance_messages"]`；`_queue_agent_run_guidance_message()` 也只合併 `guidance_messages`。目前沒有看到這些 resume 路徑會直接把上一輪 chat transcript 或上一輪 assistant 產出的程式碼片段注入新 attempt。若使用者說「保存你剛剛給我的程式碼」，runtime 必須主動帶入前文，不能期待模型自己記得。

4. **範例2 GGUF tool call 跑掉：確認 ToolCallSimulator 只支援 JSON payload，缺少 Qwen XML 風格兜底。**
   `mochi/backends/tool_call_simulator.py` 要求 `<tool_call>{JSON}</tool_call>`，`_decode_tool_payload()` 只做 `json.loads()` 和 raw JSON decode。若 Qwen GGUF 回到 `<function=name><parameter=...>` 風格，現在 parser 會回傳空 tool calls，最後可能把整段 `<tool_call>` 當成一般文字輸出。

5. **Goal 提案卡已刪，但 `pending_proposal` 還存在於 workflow / confirmation 路徑。**
   目前 grep 沒看到 `GoalCard` / `goalCard` / `goal_card` 命中，但 `pending_proposal` 仍在 `mochi/main.py`、`mochi/terminal_goal_helpers.py`、`web/src/app/page.tsx` 等位置大量存在。這表示「Goal 提案卡 UI component」已刪除，但 pending proposal 仍是顯式 workflow / confirmation 狀態機的一部分，不是已完全移除的概念。

---

## 2. 已由源碼確認的現況

### 2.1 Ollama context 設定沒有自動對齊 serving window

確認點：

- `mochi/config/schema.py:53`：`num_ctx: int | None = Field(default=None, ge=1, le=1_048_576)`
- `mochi/config/schema.py:54`：註解為 `None keeps the server default`
- `mochi/backends/ollama.py:202-212`：`options` 只在 `_configured_num_ctx is not None` 時加入 `num_ctx`
- `mochi/backends/ollama.py:122-153`：`get_model_info()` 有 `configured_num_ctx`、`runtime_context_length`、`model_max_context_length` metadata，但這些只是回報，沒有保證每次 request 的 prompt budget 會使用實際 server context
- `mochi/agents/engine.py:728-750`：會估算 prompt tokens / remaining tokens / usage ratio，但這段目前是 snapshot 建立，不是 hard gate
- `mochi/` grep 沒找到 `finish_reason.*length` 或 `length.*finish_reason` 分支

風險：

- 如果 Ollama server default 是 4096，而 model metadata 顯示 32k/128k，runtime 會誤以為 prompt 還很安全。
- 一旦第二輪對話包含歷史、工具結果、長 reasoning summary，輸入接近 4096 時，輸出會被壓到很短，最後 `finish_reason=length`。
- 如果 `react_loop` 不把 `finish_reason=length` 視為錯誤或可恢復狀態，UI 就會看起來像正常完成但內容被截斷。

### 2.2 ToolCallSimulator 目前只有 JSON 契約

確認點：

- `mochi/backends/tool_call_simulator.py:20-29` 的提示要求模型輸出 `<tool_call>` 區塊，且區塊中必須是 valid JSON
- `mochi/backends/tool_call_simulator.py:31-34` 只抓 `<tool_call>...</tool_call>` 外殼
- `mochi/backends/tool_call_simulator.py:88-102` 只把 payload 當 JSON dict/list 解碼
- `_to_tool_call()` 支援 OpenAI-like `{"function":{"name":...,"arguments":...}}`，但仍然要求外層是 JSON

風險：

- Qwen / Qwen3 / Qwen Coder 類模型常見 XML-ish tool call 格式，例如：

```xml
<tool_call>
  <function=web_search>
    <parameter=query>...</parameter>
  </function>
</tool_call>
```

- 這類輸出不是隨機胡言，而是模型訓練分佈中的固定格式。runtime 應該解析或修復，而不是讓它漏到 final answer。

### 2.3 工具暴露策略仍以 routing / keyword / risky limit 決定能力

確認點：

- `mochi/agents/tool_exposure.py:41-50` 有 `_CORE_WORKSPACE_READ_ONLY_TOOLS`
- `mochi/agents/tool_exposure.py:52-66` 把 `file_write`、`file_edit`、`apply_patch`、`exec_command` 等列入 `_RISKY_TOOLS`
- `mochi/agents/tool_exposure.py:600-637` 依 routed intent / keyword 決定 `matched_groups`
- `mochi/agents/tool_exposure.py:647-650` 依 backend / autonomy mode 決定 tool limit 和 risky limit
- `mochi/agents/tool_exposure.py:773-781` read-only file request 時會濾掉 risky tools
- `mochi/agents/tool_exposure.py:790-794` risky tools 超過 limit 會被濾掉
- `mochi/agents/tool_exposure.py:796-807` workspace baseline 只補 read-only tools

風險：

- 使用者自然語言沒有明確命中 `workspace_write` 時，模型可能只拿到唯讀工具。
- 即使 system prompt 或模型記憶知道存在 `file_write`，當工具表沒有提供它時，模型只能空轉或回覆不能寫入。
- 對本地弱模型來說，工具表能力不穩定會比能力少更糟：它會在不同輪次形成互相矛盾的信念。

### 2.4 Goal resume 使用 guidance queue，但缺少明確 transcript carry-over

確認點：

- `mochi/runtime/service.py:1307-1355` 的 `resume_goal()` 接收 `guidance_message`，並放進新 attempt summary 的 `guidance_messages`
- `mochi/runtime/service.py:2020-2057` 的 `_queue_agent_run_guidance_message()` 把 guidance 合併進 run summary，並注入 recovery state
- 目前看到的是 guidance / recovery state 路徑，沒有看到「從 chat session 取最近 assistant 程式碼、上一輪 user/assistant transcript，塞進新 attempt 初始 messages」的明確邏輯

風險：

- 使用者在 goal UI 中把它當連續聊天使用，但 runtime 把它當新 attempt / resume guidance，兩者心智模型不同。
- 「保存你剛剛給我的程式碼」這種指令必須依賴上一輪 assistant content；如果 runtime 沒帶入，模型只能猜或承認失憶。

### 2.5 GoalCard 已移除，但 pending_proposal 是另一個概念

確認點：

- 源碼 grep 未找到 `GoalCard`、`goalCard`、`goal_card`
- 但 `pending_proposal` 仍大量存在：
  - `mochi/main.py`
  - `mochi/goal_intent.py`
  - `mochi/terminal_goal_helpers.py`
  - `web/src/app/page.tsx`
  - `web/src/lib/*`

解讀：

- **Goal 提案卡 UI component 已刪除。**
- **pending proposal 沒有刪除，也不應該被視為同一件事。**
- 目前 pending proposal 看起來仍負責「顯式 workflow setup / confirmation / revision」狀態，而不是一般 `/goal <objective>` 的必經提案卡。
- 後續應該把命名與路徑拆清楚，否則維護者容易以為 pending proposal 是舊 Goal Card 殘留。

---

## 3. 三個範例的根因判定

### 3.1 範例1：第二次需求後輸出截斷

根因排序：

1. **P0：serving context 未被 runtime 當成硬限制。**
2. **P0：`finish_reason=length` 未被視為可恢復錯誤。**
3. **P1：history / tool result 沒有以 token budget 做穩定裁剪。**
4. **P1：UI / trace 沒有把截斷原因明確呈現給使用者。**

修正方向：

- Ollama backend 必須提供 `effective_context_length`，優先順序建議為：
  1. explicit configured `num_ctx`
  2. `/api/show` 解析出的 runtime context
  3. server known default
  4. conservative fallback，例如 4096 或 8192
- Engine 組 prompt 前要以 `effective_context_length - reserved_output_tokens` 當硬上限。
- `react_loop` 收到 `finish_reason=length` 要進入 continuation / compaction retry，而不是 final。

### 3.2 範例1 截斷後第2次測試：可輸出程式碼但不保存

根因排序：

1. **P0：workspace-bound session 沒有穩定暴露寫入能力。**
2. **P0：Goal follow-up 沒有把上一輪 assistant code 帶入新 attempt。**
3. **P1：工具提示沒有明確說明「若要保存，必須用 file_write/apply_patch」。**
4. **P1：runtime 缺少「使用者要求保存前一段程式碼」的特殊上下文補強。**

修正方向：

- `file_write` / `file_edit` / `apply_patch` 對 workspace-bound session 應常駐可見。
- 安全控制應移到 approval / diff preview / sandbox policy，不應靠 intent 隱藏工具。
- 如果使用者說「保存剛剛的程式碼」，runtime 應在 user message 前附加一個 context block，包含最近一輪 assistant code blocks 或 artifact references。

### 3.3 範例2：GGUF 工具調用後面直接跑掉

根因排序：

1. **P0：tool-call parser 不支援 Qwen XML-ish 格式。**
2. **P0：final answer 含 `<tool_call>` 時沒有二次解析 / 修復。**
3. **P1：對弱模型缺少 constrained decoding / grammar / few-shot tool examples。**
4. **P1：長 context 後格式漂移沒有被偵測成 runtime recoverable error。**

修正方向：

- 將 `ToolCallSimulator` 拆成 parser chain：JSON parser、Qwen XML parser、repair parser。
- `react_loop` 在 final answer 前檢查 `<tool_call>` / `<function=`；若可解析，當工具調用執行；若不可解析，要求模型修復一次。
- 對 GGUF / Qwen profile 使用模型熟悉的格式，不要硬要求所有模型只用 OpenAI JSON 形狀。

---

## 4. 與成熟 agent runtime 的差距

### 4.1 Context 是控制面，不是觀測面

成熟 agent runtime 的 context budget 會直接決定：

- 哪些 history 留下
- 哪些 tool result 只留摘要
- 是否觸發 compaction
- 是否保留固定 output reserve
- 是否中止本輪並要求縮短

目前 mochi 已經有 token estimate、compaction、snapshot，但更像觀測與提示，不像硬控制面。範例1顯示只觀測不夠。

### 4.2 工具能力應穩定，風險由 permission 管

Claude Code / Codex 類產品的核心是：

- 能力表穩定
- 操作前可審批
- 寫入有 diff
- 執行命令有 sandbox / allowlist / escalation

mochi 目前仍把「是否暴露工具」當成安全與 prompt budget 的混合控制。這會讓模型不知道自己到底能做什麼。

### 4.3 Tool call protocol 需要模型 profile

成熟 harness 不會假設所有非原生 function-calling 模型都能穩定輸出同一種 JSON。至少要有：

- model family profile：Qwen、Llama、Gemma、DeepSeek、Mistral
- preferred tool syntax
- parser fallbacks
- repair prompts
- constrained decoding / grammar 支援
- invalid tool call telemetry

### 4.4 Goal lane 與 chat lane 的記憶契約要明文化

目前問題不是單純 bug，而是產品語義不清：

- Goal 是長時任務 attempt？
- Goal chat 是與同一 agent 的連續對話？
- resume 是從 checkpoint 繼續，還是用 guidance 啟動新 run？
- 使用者在 goal 中說「剛剛」時，指哪個上下文？

沒有契約時，實作會自然退化成零散 guidance / summary patch，弱模型就會失憶。

---

## 5. P0 修正項：逐項可交付

### P0-1：建立 `effective_context_length` 單一事實來源

涉及檔案：

- `mochi/backends/ollama.py`
- `mochi/backends/types.py`
- `mochi/agents/engine.py`
- `mochi/agents/context.py`
- `mochi/agents/compaction.py`

具體做法：

1. 在 `ModelInfo.metadata` 或正式欄位中加入：
   - `effective_context_length`
   - `effective_context_length_source`
   - `configured_context_length`
   - `serving_context_length`
   - `model_advertised_context_length`
2. Ollama backend 計算規則：
   - 若 config `num_ctx` 有值，effective = `num_ctx`
   - 否則若 `/api/show` 能拿到 runtime context，effective = runtime context
   - 否則 effective = conservative default，例如 4096
3. `engine.py` prompt 組裝前使用 effective 值，不使用 model advertised max。
4. 若 `estimated_prompt_tokens + reserve_output_tokens > effective_context_length`：
   - 先 compaction
   - 再壓縮 tool results
   - 再裁剪最舊 history
   - 仍超出則回 structured runtime error

驗收測試：

- 建一個 fake Ollama backend：model advertised 32768，但 effective 4096。
- 塞入 3900 token history，reserve 1024。
- 期望 engine 不送出請求，而是先 compaction / trim。

### P0-2：Ollama `num_ctx` auto policy

涉及檔案：

- `mochi/config/schema.py`
- `mochi/backends/ollama.py`
- settings API / UI 若有模型設定頁

具體做法：

1. 新增 config：

```python
auto_num_ctx: bool = True
auto_num_ctx_cap: int = 32768
```

2. `_build_options` 或 generate options 組裝時：

```python
if self._configured_num_ctx is not None:
    options["num_ctx"] = self._configured_num_ctx
elif self._auto_num_ctx:
    options["num_ctx"] = min(
        self._model_max_context_length or self._CONTEXT_LENGTH_FALLBACK,
        self._auto_num_ctx_cap,
    )
```

3. 若使用者硬體不適合大 context，可把 cap 降低，但 runtime 必須知道實際送出的值。

驗收測試：

- `num_ctx=None, auto_num_ctx=True` 時，payload options 必須含 `num_ctx`。
- `num_ctx=8192` 時，payload 必須優先用 8192。
- `auto_num_ctx=False` 時，可保留 server default，但 `effective_context_length_source` 必須標記為 server default / unknown conservative。

### P0-3：處理 `finish_reason=length`

涉及檔案：

- `mochi/agents/react_loop.py`
- `mochi/backends/types.py`
- `web/src/*` event rendering

具體做法：

1. 在 generation result 完成後檢查：

```python
if result.finish_reason == "length":
    ...
```

2. 分三層策略：
   - 如果 assistant content 非空：自動 continuation，最多 2 次。
   - 如果連續 length：compaction 後重試一次。
   - 如果仍失敗：回 `runtime_error`，code = `output_truncated`，不要當成功 final。
3. UI 顯示黃色或橘色 runtime warning，而不是一般完成。

驗收測試：

- fake backend 第一次回 `finish_reason="length"` 和半段 content，第二次回 stop，期望 final 是兩段合併。
- fake backend 連續三次 length，期望 final status 是 recoverable error，不是 success。

### P0-4：加入 Qwen XML tool parser

涉及檔案：

- `mochi/backends/tool_call_simulator.py`
- 建議新增 `mochi/backends/tool_call_parsers.py`
- `tests/backends/test_tool_call_simulator.py`

具體做法：

新增 parser chain：

1. JSON block parser：保留現有行為。
2. Qwen XML parser：支援：

```xml
<tool_call>
<function=arxiv_search>
<parameter=query>medical imaging</parameter>
<parameter=max_results>5</parameter>
</function>
</tool_call>
```

3. 支援單行變體：

```xml
<tool_call> <function=arxiv_search> <parameter=query>medical imaging</parameter> </function> </tool_call>
```

4. 參數型別策略：
   - `"true"/"false"` 轉 bool
   - 純整數轉 int
   - 純浮點轉 float
   - 其他保留 string
5. 多個 `<function=...>` 要回多個 `ToolCall`。

驗收測試 fixture：

```python
raw = '''
<tool_call> <function=arxiv_search>
<parameter=query>"medical imaging"</parameter>
<parameter=max_results>5</parameter>
</function> </tool_call>
'''
```

期望：

- name = `arxiv_search`
- arguments["query"] = `"medical imaging"` 或正規化後 `medical imaging`
- arguments["max_results"] = 5

### P0-5：final answer tool markup 二次防線

涉及檔案：

- `mochi/agents/react_loop.py`
- `mochi/backends/tool_call_simulator.py`

具體做法：

1. 在 assistant final content 送 UI 前檢查：

```python
if "<tool_call" in content.lower() or "<function=" in content.lower():
    parsed = tool_parser.parse_tool_calls(content)
```

2. 若 parsed 非空：
   - 不要把 content 當 final answer。
   - 將 parsed tool calls 轉入正常工具執行流程。
3. 若 parsed 為空：
   - 追加一次 repair prompt：「你輸出了無效工具調用格式，請只用指定格式重輸出工具調用或直接作答。」
4. 一次 repair 後仍失敗才顯示給使用者，並標注 `invalid_tool_call_markup`。

驗收測試：

- fake backend final content 含 Qwen XML tool call，期望 runtime 執行 tool，不顯示 raw XML。
- fake backend final content 含壞掉的 `<tool_call>`，期望產生 repair turn。

### P0-6：workspace-bound session 常駐暴露寫入工具

涉及檔案：

- `mochi/agents/tool_exposure.py`
- `mochi/config/schema.py`
- approval / security policy 相關測試

具體做法：

1. 定義 workspace baseline write tools：

```python
_CORE_WORKSPACE_WRITE_TOOLS = ("file_write", "file_edit", "apply_patch")
```

2. 當 `session_bound_workspace=True` 且工具存在：
   - 常駐加入 final tools
   - 不受 keyword intent 影響
   - 可受 autonomy mode 決定是否 requires approval，但不要完全消失
3. `exec_command` 可比 file write 更保守，但也應透過 permission mode，而不是只靠關鍵字。
4. 若工具太多，至少保證：
   - read：`file_read`, `grep_search`, `glob_search`
   - write：`file_write` 或 `apply_patch` 二選一，最好兩者都有
   - discovery：`tool_search`

驗收測試：

- workspace-bound + message = 「幫我寫一個訓練程式」：final tools 包含 `file_write` 或 `apply_patch`。
- workspace-bound + greeting：可不主動排序寫入工具在前，但工具不應在 discoverable list 中消失。
- read-only explicit request：「只讀取不要修改」：可以要求模型不要寫，但是否完全隱藏寫入工具要由 policy 決定，不能由 keyword 猜測。

### P0-7：Goal follow-up 帶入最近 transcript / artifacts

涉及檔案：

- `mochi/runtime/service.py`
- agent run package 組裝處
- goal / chat session store
- `mochi/agents/compaction.py`

具體做法：

1. 建立 `GoalConversationCarryover`：
   - recent user messages
   - recent assistant final messages
   - recent assistant code blocks
   - recent written artifacts / tool results references
2. `resume_goal(guidance_message=...)` 建新 attempt 時，不只放 guidance，也放：

```text
Recent conversation context:
- User asked: ...
- Assistant produced code: ...
- User now asks to save that code.
```

3. carry-over token budget：
   - 小模型：effective context 20% 或最多 1200 tokens
   - 大模型：effective context 30% 或最多 4000 tokens
4. 若使用者語句含「剛剛」「上一段」「你剛寫的」「保存它」，優先帶最近 assistant code block。

驗收測試：

- 第一輪 assistant 產出 code block。
- 第二輪 user：「保存你剛剛給我的程式碼」。
- 期望新 run 初始 prompt 包含該 code block 或 artifact reference。
- 模型不應回覆「我看不到之前內容」。

---

## 6. P1 修正項：降低弱模型失控率

### P1-1：把 guard steering 從 tool error 中拆出

問題：

- 前一份分析提到 evidence guard 以紅色工具失敗呈現「證據已足夠，請綜合作答」。
- 即使這是 runtime 主動攔截，也不應偽裝成 tool failure。

修正：

- 新增 event type：`runtime_steering`
- 新增 steering reason：
  - `evidence_sufficient`
  - `tool_budget_exhausted`
  - `context_budget_near_limit`
  - `unsafe_tool_blocked`
- 對模型注入 system/developer style steering message，不要回 tool error。

驗收：

- UI 不顯示紅色 tool failure。
- trace 中可區分真工具錯誤與 runtime steering。

### P1-2：錯誤分類法

新增 enum：

```python
RuntimeFailureKind = Literal[
    "backend_error",
    "tool_error",
    "tool_denied",
    "runtime_steering",
    "context_overflow",
    "output_truncated",
    "empty_response",
    "invalid_tool_call",
    "cancelled",
]
```

每種錯誤都要定義：

- 是否可重試
- 是否應呈現給使用者
- 是否應注入模型上下文
- UI 顏色
- telemetry key

### P1-3：弱模型 tool prompt profile

對 GGUF / local model 增加 profile：

- `qwen_xml_tool_call`
- `json_tool_call`
- `no_tool_call_reliable`
- `native_function_calling`

每個 profile 定義：

- prompt template
- parser chain
- max repair attempts
- whether to use grammar
- stop sequences

### P1-4：`history_window=20` 改成 token budget

現況：

- `mochi/agents/compaction.py` 有 `ContextBudget` 與 token budget 概念。
- 但很多路徑仍會以固定 message count 作為主要裁切。

修正：

- 固定則數只能是上限，不是安全保證。
- 真正安全條件必須是 token estimate。
- tool result preview / full artifact reference 要有不同 token 計算方式。

### P1-5：Prompt / parser 契約測試

對以下元件加 round-trip 測試：

- active goal turn selector
- tool intent router
- tool call simulator
- goal proposal / pending proposal parser

測試方式：

- prompt 中宣告的 JSON schema 必須由 parser 接受。
- parser 要求的欄位必須出現在 prompt schema。
- fallback 行為必須有 snapshot。

---

## 7. P2 架構整理：減少主程式複雜度

### 7.1 legacy 代碼不要留在主流程

建議政策：

1. **已完全不用的 UI component / copy helper：直接刪除。**
2. **仍需保留資料遷移或向後相容的 legacy parser：移到 `legacy/` 或 `compat/`，並加明確 expiry note。**
3. **主流程不得 import legacy module。**
4. **若主流程仍需 import，代表它不是 legacy，應重新命名為 compat 或 workflow state。**

套用到目前 goal：

- `GoalCard` 類 UI 已刪除是正確方向。
- `pending_proposal` 不應直接稱 legacy，因為目前 workflow confirmation 還在用。
- 但應拆成：
  - `workflow_pending_proposal`
  - `goal_confirmation_draft`
  - 或其他更準確命名
- 避免開發者誤會「pending proposal = 舊 Goal Card 殘留」。

### 7.2 巨型檔案拆分

目前行數：

- `mochi/runtime/service.py`：約 14819 行
- `web/src/app/page.tsx`：約 6629 行
- `mochi/agents/engine.py`：約 3573 行
- `mochi/agents/react_loop.py`：約 1843 行

拆分優先順序：

1. `service.py`
   - `goal_lifecycle_service.py`
   - `agent_run_resume_service.py`
   - `runtime_event_service.py`
   - `approval_service.py`
2. `page.tsx`
   - `useChatRuntimeController`
   - `useGoalWorkflowRouting`
   - `useAgentRunEvents`
   - `ChatPageShell`
3. `react_loop.py`
   - `tool_call_repair.py`
   - `runtime_steering.py`
   - `generation_recovery.py`
   - `tool_execution_loop.py`

拆分原則：

- 每次只搬一個純函式群或 hook。
- 搬完先不改行為。
- 加 characterization tests。
- 不要在拆分 PR 同時改 runtime 行為。

---

## 8. 對前一份分析的校正

### 8.1 我同意的部分

- 範例1主要是 context / truncation 管理缺口。
- 範例2主要是 simulated tool call parser 太窄。
- 寫入工具靠 intent 隱藏會造成 agent 能力不穩。
- Goal resume / guidance 與 chat transcript 的關係需要重做。
- 巨型檔案已經影響維護性。

### 8.2 需要更精確表述的部分

- 「全部主要是 runtime 設計問題，不是模型問題」應改成：**runtime 設計缺口是主因，弱模型與量化模型會放大缺口。**
  例如 Qwen XML tool call 是模型分佈問題，但成熟 runtime 應解析它。

- 「pending proposal 已刪除」不準確。
  準確說法是：**GoalCard UI 已刪除；pending proposal state 仍存在，且看起來仍支撐 workflow confirmation。**

- 「goal attempt 只帶 guidance，完全不帶歷史」還需要更完整 trace 證據。
  源碼已確認 resume guidance path，但要下最終斷言前，還應檢查 agent run package 最終 prompt 組裝。即便如此，目前範例現象與已讀源碼足以支持「carry-over 不足」這個修正項。

### 8.3 不建議照做的部分

- 不建議直接把所有 risky tools 永遠放進前 6 個工具。
  應該做「常駐可見 + 排序靠 intent + 操作靠 approval」。工具太多時，可以把較不常用工具放 discoverable list，但 `file_write/apply_patch` 對 workspace agent 不應消失。

- 不建議一次大改 `react_loop.py`。
  先加 parser / finish_reason / tool exposure 的小型回歸測試，再局部修。

---

## 9. 建議實作順序

### 第 1 批：直接修範例故障

1. `ToolCallSimulator` 支援 Qwen XML parser。
2. final answer 前偵測 raw `<tool_call>` 並轉回工具流程。
3. workspace-bound session 常駐暴露 `file_write` / `apply_patch`。
4. `finish_reason=length` recovery。
5. Ollama `effective_context_length` + `num_ctx` auto policy。

### 第 2 批：修 Goal 連續對話體驗

1. Goal resume carry-over 最近 transcript。
2. 「保存剛剛程式碼」自動附最近 code block。
3. pending proposal 命名與 workflow 邊界整理。
4. greeting bypass / pending capture 測試補齊。

### 第 3 批：工程結構整理

1. service / page / react loop 拆檔。
2. runtime error taxonomy。
3. model capability profiles。
4. token-budgeted history。
5. prompt-parser contract tests。

---

## 10. 最小回歸測試清單

### Tool call parser

- JSON `<tool_call>` 可解析。
- Qwen XML `<function=name><parameter=key>` 可解析。
- 單行 Qwen XML 可解析。
- 多 function 可解析成多 tool calls。
- 壞掉的 `<tool_call>` 不應直接成 final answer。

### Context / truncation

- effective context 4096 時，3900 token prompt + 1024 reserve 會觸發 compaction。
- `finish_reason=length` 會 continuation。
- 連續 length 會回 structured error。

### Tool exposure

- workspace-bound 寫程式需求暴露 `file_write` 或 `apply_patch`。
- read-only 明確要求不應觸發寫入，但 capability policy 行為要穩定。
- greeting 不應進 pending proposal capture。

### Goal carry-over

- 第一輪產生 code，第二輪要求保存「剛剛」內容，新 attempt prompt 內含 code block。
- resume guidance 不會清掉前一輪必要上下文。
- pending proposal 只在 workflow setup / confirmation 下出現。

---

## 11. 立即可派工任務卡

### Task A：Qwen XML parser

- 改：`mochi/backends/tool_call_simulator.py`
- 加：`tests/backends/test_tool_call_simulator_xml.py`
- 不改：任何 UI / agent loop
- Done：
  - 解析範例2 XML tool call fixture
  - 原 JSON 測試仍通過

### Task B：workspace write tools always visible

- 改：`mochi/agents/tool_exposure.py`
- 加：`tests/agents/test_tool_exposure_workspace_write.py`
- Done：
  - workspace-bound + code creation request 暴露 `file_write` / `apply_patch`
  - read-only tools baseline 仍存在
  - strict mode 下危險 exec 仍可被 policy 擋住

### Task C：finish_reason length recovery

- 改：`mochi/agents/react_loop.py`
- 加：fake backend tests
- Done：
  - length 不再被當正常完成
  - continuation 合併輸出
  - 超過重試次數有明確 runtime error

### Task D：Goal carry-over

- 改：`mochi/runtime/service.py` 與 agent run package 組裝處
- 加：goal resume integration test
- Done：
  - `guidance_message` 旁帶 recent transcript summary
  - 最近 assistant code block 可被保存指令引用

### Task E：pending proposal 命名整理

- 改：`mochi/main.py`、`web/src/app/page.tsx`、`web/src/lib/*`
- 先不改行為，只拆名詞：
  - `pending_proposal` 保留資料欄位相容
  - 程式內 alias 成 `workflowPendingProposal`
- Done：
  - grep `GoalCard|goalCard|goal_card` 仍為空
  - `/goal <objective>` 不產生提案卡
  - workflow confirmation path 仍可用

---

## 12. 最後判斷

這個專案已經有不少成熟 agent runtime 的零件：多後端、tool transport、compaction、goal attempt、event projection、approval/security 設定。但目前最大的差距是「零件彼此沒有形成硬契約」：

- context budget 有觀測，缺硬裁切；
- tool calling 有 prompt，缺多格式 parser 與 repair；
- tool exposure 有 routing，缺穩定能力表；
- goal 有 attempt/resume，缺與 chat transcript 的記憶契約；
- error 有 event，缺分類語義。

因此後續優化不應先追更多功能，而應先把 runtime contract 補硬。只要 P0 七項完成，範例1與範例2這類失敗會明顯下降，弱模型也會更像真正 agent，而不是在工具表、上下文與格式漂移中失控。
