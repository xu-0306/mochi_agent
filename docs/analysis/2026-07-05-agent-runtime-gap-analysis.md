# agent_mochi Runtime 差距分析與優化路線圖（2026-07-05）

> 範圍：`mochi/`（後端 runtime）與 `web/`（前端）。
> 依據：實際代碼審查 + `範例1`、`範例1/截斷後第2次測試`、`範例2` 共 25 張實測截圖的證據鏈。
> 對照對象：Claude Code、OpenAI Codex CLI、OpenClaw、Hermes-agent 等成熟 agent harness 的共通設計。

---

## 第一部分：範例故障診斷（結論先行）

三個現象都查到了明確根因，**全部主要是 runtime 設計問題，不是模型問題**（模型能力弱只是放大了缺陷）。

### 1.1 範例1：第二次請求輸出被截斷 → **Ollama context window 沒有被 runtime 管理**

**證據（截圖 `025912.png` 底部統計列）：**

```
input tokens 3986, output tokens 110, finish reason: length
```

**3986 + 110 = 4096**，正好是 Ollama 未設定 `num_ctx` 時的伺服器預設值。

根因鏈：

1. `mochi/config/schema.py:53`：`num_ctx: int | None = None`，註解明說 "None keeps the server default" → 預設不覆蓋 Ollama 的 4096。
2. `mochi/backends/ollama.py:211-212`：只有 `_configured_num_ctx is not None` 才會送 `options.num_ctx`。
3. `mochi/agents/engine.py:734-748` 有 context budget 快照（`context_length`、`remaining_tokens`），但它讀的是**模型的最大 context 長度 metadata**（可能是 32k/256k），而實際 serving 端只有 4096 —— **引擎相信的預算和實際預算不一致**，所以 prompt 組裝時不會裁剪，也不會警告。
4. `mochi/agents/react_loop.py` 全檔沒有任何 `finish_reason == "length"` 的分支 —— 截斷的回覆被當成正常結束，不續寫、不壓縮重試、不告知使用者。UI 上就是「輸出到一半突然停了」。

第一輪沒事是因為多次 ReAct 迭代各自的 prompt 都還在 4096 內；第二輪 payload（歷史+工具結果）漲到 3986，輸出空間只剩 110 token。

### 1.2 範例1 截斷後第2次測試：模型「不能存檔」+「忘記上下文」→ **工具暴露策略 + goal 換 run 不帶對話歷史**

**「不能存檔」的證據：** 截圖裡該輪暴露的 WORKSPACE TOOLS 是
`repo_map, read_symbol, glob_search, grep_search, file_read, csv_read, pdf_read, docx_read, web_search, web_fetch, get_current_time, tool_search, tool_result_read` —— **清一色唯讀，沒有 `file_write`**。

`mochi/agents/tool_exposure.py` 以訊息關鍵字/intent 路由決定暴露哪組工具（`_CORE_WORKSPACE_READ_ONLY_TOOLS`、`workspace_write` intent 等）。「幫我寫個訓練程式」被判成產出文字的請求而非 `workspace_write`，於是寫檔工具沒進工具表。模型 reasoning 明確說出 *"I don't have direct access to create or execute code files"* —— 它是被 runtime 剝奪了寫檔能力，不是不會。更矛盾的是：下一輪模型 reasoning 說「我需要使用 file_write 工具」（它從系統提示或記憶知道有這工具），但工具表裡仍然沒有 → 只能不斷 glob_search 找檔案，空轉。

**「忘記上下文」的證據：** 第三輪（「保存你剛剛給我的程式碼」）模型 reasoning：

> 「我沒有看到之前的對話內容，因為這是新的對話開始。」

同一輪 trace 內另一段 reasoning 卻說「根據之前的對話，我之前提供了…」→ **同一 run 內不同 iteration 拿到的歷史不一致**。goal lane 的 follow-up 走 `resume_goal(guidance_message=...)` / `guidance_messages` 佇列（`mochi/runtime/service.py:1307-1344`），新 attempt 是一個**全新的 agent run，只帶 guidance 文字，不帶聊天室的對話 transcript**。前一輪 assistant 生成的程式碼根本不在模型 context 裡，模型自然「失憶」。

另外該輪出現兩次 `Model returned an empty response / AGENT_ERROR`：`react_loop.py:1231-1252` 有 `_should_retry_empty_final_response` 重試機制，有效但重試 prompt 沒帶回遺失的上下文，重試後仍是失憶狀態。

### 1.3 範例2：gguf 後端「連輸出都跑掉」→ **工具調用格式漂移 + 解析器只認一種格式**

**證據（截圖 `030710.png` 最終回覆）：** assistant 的最終「答案」就是一大段裸露的

```
<tool_call> <function=arxiv_search> <parameter=query> "medical imaging" ... </parameter> ... </function> </tool_call>
```

`finish reason: stop`，模型是 `E:\_models\Qwe...`（Qwen 系列 GGUF）。

根因鏈：

1. `mochi/backends/gguf.py` 對不支援原生 function calling 的模型走 `flattened_text` 策略，由 `ToolCallSimulator` 注入提示，要求模型輸出 `<tool_call>{JSON}</tool_call>`。
2. Qwen3/Qwen3-Coder 的**訓練內建格式**是 XML 風格：`<tool_call><function=name><parameter=key>value</parameter></function></tool_call>`。前幾輪模型勉強配合 JSON（工具正常執行）；context 變長後回落到自己的訓練格式。
3. `mochi/backends/tool_call_simulator.py:31-34` 的 `TOOL_CALL_RE` 抓得到 `<tool_call>` 外殼，但 `_decode_tool_payload` 只會 `json.loads`（88-102 行），XML payload 解析失敗 → 回傳空列表 → **整段標記文字被當成純文字最終答案原樣顯示**。
4. 沒有第二層防線：最終答案含有 `<tool_call>` 標記時，runtime 不偵測、不修復、不重試。

判定：模型格式漂移是誘因（小模型+量化+長 context 常見），但成熟 harness 對這種已知的、格式固定的漂移都有解析兜底；這裡是 runtime 缺件。

### 1.4 附帶發現：evidence guard 偽裝成工具失敗

範例1 截圖中 `semantic_scholar_search` / `web_search` 顯示紅色「失敗」，錯誤訊息卻是：

> "Sufficient literature evidence is already collected. Do not call more search or fetch tools; synthesize the answer now."

這是 `react_loop.py` 的 evidence guard（`_should_force_followup_retrieval` 等）主動攔截，**卻用「工具執行失敗」的形狀回給模型和 UI**。後果：(a) UI 使用者以為工具壞了；(b) 對弱模型而言，「失敗」語義會誘導重試同一工具而不是收斂作答（範例1 裡模型確實又多打了一輪搜尋）。指令型 guard 應該用獨立的 system/steering 訊息注入，而不是偽造 tool error。

---

## 第二部分：架構總評（軟體工程視角）

### 2.1 做得好的部分（值得保留的資產）

| 資產 | 位置 | 說明 |
|---|---|---|
| 工具結果雙軌傳輸 | tool transport（context-safe preview + full artifact 落盤 + `tool_result_read` 回讀） | 與 Claude Code 的大輸出落盤思路一致，對小 context 模型尤其正確 |
| 後端能力探測 | `native_status=supported/native_default/thinking_without_native_tool_calls` | 執行期探測原生 tool calling，方向正確 |
| 型別化語意仲裁 | `active_goal_turn_selector.py`（strict JSON、白名單 kind、保守 fallback、confidence 正規化） | 這一小塊的工程品質是全專案最好的，可作為其他分類器的範本 |
| Compaction 有預算概念 | `compaction.py`（`ContextBudget`、`history_window`、hybrid semantic summary） | 骨架齊全，問題只在沒接到真實 serving 端數字 |
| 事件/進度投影 | ReAct progress、execution timeline、SSE dedupe | 已具備成熟產品的可觀測性雛形 |
| 多後端抽象 | `backends/`（ollama/gguf/openai_compat/vllm/llama_cpp_server...） | 覆蓋面廣 |

### 2.2 結構性問題

1. **「預算」三處各算各的，沒有單一事實來源。** engine 有 context 快照、context.py/compaction 有 `max_input_tokens`、ollama backend 有三種 context_length 來源（configured/runtime/model_max），但「這次請求實際能用多少 token」沒有一條從 serving 端貫穿到 prompt 組裝的鏈路。範例1 就是這個裂縫的直接後果。成熟 harness（Claude Code/Codex）的做法是：每次請求前用 tokenizer 或估算器對 messages 計數，對照 *serving 端宣告* 的 window，超了先壓縮，並保留固定的輸出保留區。
2. **控制流靠字串/關鍵字的地方太多。** tool_exposure 用關鍵字表路由 intent（`_READ_ONLY_FILE_INTENT_KEYWORDS`、`research` 關鍵字…），與 Goal handoff 文件裡「前端不得用關鍵字列表分類自然語言」的自我要求矛盾——同一個教訓在工具層又犯了一次。範例1 的「不能存檔」就是關鍵字路由的誤判。
3. **失敗語義不分層。** 「工具真的失敗」「guard 攔截」「後端錯誤」「空回覆」「截斷」全都擠在 tool error / AGENT_ERROR 兩種形狀裡。模型（尤其弱模型）依賴失敗語義決定下一步；語義混淆直接變成行為異常。
4. **巨型檔案持續累積。** `engine.py` 3572 行、`openai_compat.py` 2339 行、`react_loop.py` 1842 行、`main.py` 3140 行、`page.tsx` 6734 行。react_loop 裡 evidence guard、thinking 解析、工具修復、串流合併全部內聯成 40+ 個私有方法，任何模型（包括弱模型）要在裡面安全改動都很困難。
5. **Goal lane 與 chat lane 的記憶體制未定義。** Goal attempt 是獨立 run 只帶 guidance——這在「長時自主任務」語境下合理，但產品上使用者把 goal 對話當成連續聊天用（範例1 的三輪就是），兩種心智模型衝突沒有被架構回答：attempt 到底該繼承什麼？目前答案是「幾乎什麼都不繼承」，這是失憶的根源。

### 2.3 與成熟 agent 專案的差距對照

| 能力 | Claude Code / Codex 等的做法 | mochi 現狀 | 差距等級 |
|---|---|---|---|
| Context 預算管理 | 每請求 token 計數 + serving window 對齊 + 自動壓縮 + 輸出保留區 | 有快照無執行力；num_ctx 不設定；length 不處理 | **關鍵** |
| 截斷恢復 | finish=length → 自動續寫或壓縮重試，對使用者透明 | 無任何處理 | **關鍵** |
| 工具格式兜底 | 多格式解析器（JSON / XML / 函式簽名）、grammar/constrained decoding、格式漂移偵測+重試 | 單一 JSON 格式；漏網當純文字 | **關鍵** |
| 寫入權限模型 | 工具永遠可見，**危險操作走 approval/permission mode**（Codex 的 approval modes、Claude Code 的 permission prompt） | 用 intent 猜測隱藏工具 → 模型「以為自己殘廢」 | 高 |
| 跨輪/跨 run 記憶 | session transcript 全量 + 壓縮摘要 + 持久 memory 檔（CLAUDE.md/AGENTS.md） | goal attempt 只帶 guidance；chat 有 history_window=20 | 高 |
| 失敗語義分層 | tool error / system steering / interrupt 明確分開 | guard 偽裝 tool error | 高 |
| 自我驗證迴路 | 寫檔後 lint/test/re-read 驗證；答案前 checklist | evidence guard 有雛形，僅覆蓋搜尋類 | 中 |
| 沙箱與安全 | Codex: seatbelt/landlock；Claude Code: sandbox + permission | 有 approvals/security 模組（未深查），exec 側待驗證 | 中（本次未深入） |
| Sub-agent 編排 | Task/子代理 context 隔離、結果結構化回傳 | 有 delegate_subagent + protocols，成熟度未驗證 | 中 |
| 提示詞工程 | 針對弱模型的少樣例、嚴格輸出契約、每工具用例 | 工具定義注入偏簡（name+description+raw JSON schema） | 中 |

---

## 第三部分：隱含錯誤清單（已確認 + 高度可疑）

**已確認（有代碼+截圖雙重證據）：**

| # | 錯誤 | 位置 | 影響 |
|---|---|---|---|
| B1 | Ollama 未設 `num_ctx` 時，引擎預算與 serving 實際 window 脫鉤 | `backends/ollama.py:211`、`config/schema.py:53` | 長對話必然截斷（範例1） |
| B2 | `finish_reason=="length"` 無處理分支 | `agents/react_loop.py`（全檔 grep 無 length 分支） | 截斷靜默吞掉 |
| B3 | ToolCallSimulator 只解析 JSON payload，Qwen XML 格式漏網成純文字 | `backends/tool_call_simulator.py:88-102` | 範例2 輸出崩壞 |
| B4 | 最終答案含 `<tool_call>` 標記不偵測不修復 | `agents/react_loop.py` | 同上，第二層防線缺失 |
| B5 | tool_exposure 關鍵字路由漏掉寫檔意圖 → `file_write` 不在工具表 | `agents/tool_exposure.py:544-798` | 模型自認無法存檔（範例1-2） |
| B6 | goal attempt 換 run 不帶對話 transcript，只帶 guidance | `runtime/service.py:1307-1344` + run package 組裝 | 跨輪失憶（範例1-2 第三輪） |
| B7 | evidence guard 以 tool failure 形狀回覆 | `agents/react_loop.py`（guard 系列方法） | UI 誤導 + 弱模型重試空轉 |
| B8 | 空回覆重試不恢復缺失上下文 | `react_loop.py:1231-1252` | 重試後仍失憶 |

**高度可疑（本次未逐行驗證，建議排查）：**

- S1 `_ACTIVE_GOAL_TURN_SELECTOR_SYSTEM_PROMPT` 要求的 JSON schema 沒有 `lane` 欄位，但 `parse_active_goal_turn_semantic_decision:97-99` 預設 lane 為 `active_goal_turn` 且驗證它——模型若真的照 prompt 輸出反而通過，但任何多輸出一個不同 lane 值的情況會整包丟棄退回 fallback；prompt 與 parser 契約不同步。
- S2 同一 run 內不同 iteration 歷史不一致（1.2 節觀察）——懷疑 iteration 間 turn_messages 與 session history 合併順序有競態或條件分支差異，值得寫回歸測試釘住。
- S3 trace 中同一段 reasoning summary 重複顯示兩次（範例1 iteration 1/2）——投影層 dedupe 或事件重放問題。
- S4 `history_window=20` 硬編碼視窗與 token 預算無關聯，小 context 模型 20 則訊息可能已爆，大 context 模型又浪費。
- S5 `tool_search` 已暴露（動態工具發現的雛形），但模型在需要 file_write 時沒有被引導去用它——說明 system prompt 沒有教學該機制，等於做了功能沒接使用者。

---

## 第四部分：優化細項路線圖

> 每項給出：改哪裡、怎麼改、驗收標準。粒度刻意切細，供較弱的實作模型單獨執行；**每項獨立可測，不要合併成大 PR**。

### P0 — 直接對應範例故障（先做這五項，實測體感立刻改變）

**P0-1 Context 預算貫通（修 B1）**
- 檔案：`mochi/backends/ollama.py`、`mochi/agents/engine.py`
- 做法：
  1. `_build_options()`（ollama.py:192 附近）中，當 `_configured_num_ctx is None` 時，改為送出 `num_ctx = min(model_max_context_length or 8192, HARD_CAP)`（HARD_CAP 建議 32768，避免小機器 OOM；可加 config 開關 `ollama.auto_num_ctx: bool = True`）。
  2. engine 組 prompt 前，用現有 `estimated_prompt_tokens` 對照 `effective_context_length`（取 runtime 值，不是 model max），若 `estimated + reserve_output > effective`，先觸發 compaction，仍超則裁掉最舊 history 並發出 StatusEvent 警告。
- 驗收：重現範例1 場景（同模型、同兩輪請求），第二輪 `finish reason` 不再是 length；日誌可見「budget: used/total」。

**P0-2 截斷偵測與自動續寫（修 B2）**
- 檔案：`mochi/agents/react_loop.py`
- 做法：generation 結束處（405 行附近取得 `finish_reason` 後）加分支：若 `finish_reason == "length"`：
  1. 發 StatusEvent（`runtime_truncated`，附 input/output token 數）；
  2. 若已有部分輸出：自動追加一輪 continuation（將部分輸出以 assistant prefix 保留，附 system 指示 "Continue exactly where you stopped, do not repeat"），最多續 2 次；
  3. 連續 2 次仍 length → 觸發 compaction 後重試一次；再失敗 → 對使用者顯示明確的「輸出因 context 限制被截斷」錯誤，不要假裝完成。
- 驗收：單元測試模擬 finish_reason=length，斷言續寫請求被發出、最終訊息完整或帶明確錯誤。

**P0-3 多格式工具調用解析器（修 B3/B4）**
- 檔案：`mochi/backends/tool_call_simulator.py`；新增 `mochi/backends/tool_call_parsers.py`
- 做法：
  1. 新增 XML 風格解析：regex 抓 `<function=([\w-]+)>` 與 `<parameter=([\w-]+)>\s*([\s\S]*?)\s*</parameter>`，組回 `ToolCall`；同時支援同一 `<tool_call>` 內多 function、以及範例2 出現的**單行空白分隔變體**。
  2. `parse_tool_calls` 改為解析器鏈：JSON → XML → （可留擴充點給其他家格式），第一個命中即用。
  3. react_loop 收到最終答案時檢查：`if "<tool_call>" in content or "<function=" in content` → 先過解析器鏈，能解析就當工具調用執行；不能解析 → 注入一則修復 prompt 要求重新輸出（複用現有 `_build_invalid_tool_turn_repair_prompt` 模式），最多一次，仍失敗才原樣顯示並標註警告。
  4. （進階，可延後）llama.cpp 後端支援 GBNF grammar 時，工具輪強制 JSON grammar，從源頭消滅漂移。
- 驗收：以範例2 截圖中的實際文本作為 fixture 寫測試，斷言解析出 3 個 arxiv_search/pubmed_search 調用。

**P0-4 寫入工具改為「常駐暴露 + approval 把關」（修 B5）**
- 檔案：`mochi/agents/tool_exposure.py`
- 做法：
  1. 把 `file_write`（以及 `exec` 類）從「intent 命中才暴露」改為「workspace-bound session 一律暴露」；
  2. 危險性控制不靠隱藏，靠現有 approvals 機制：`file_write` 標記為 requires_approval（低信任模式下），走 approval 卡片；
  3. 保留 intent 路由僅用於**排序/預算**（工具太多時決定哪些放前面），不用於剝奪能力。
- 驗收：重現「幫我寫訓練程式並保存」場景，模型第一輪就能調用 file_write，UI 出現 approval 或直接寫入 workspace。

**P0-5 Goal attempt 攜帶對話上下文（修 B6/B8）**
- 檔案：`mochi/runtime/service.py`、run package 組裝處（`agent_run_packages.py`）
- 做法：
  1. 建 attempt / resume 時，從 session store 取最近 N 輪 chat transcript（建議：最後一輪完整 user+assistant + 更早輪的壓縮摘要，用現有 ConversationCompactor），塞進 run 的初始 messages（在 goal objective 之後、guidance 之前）；
  2. 上限用 token 預算（例如 effective_context 的 30%），不是固定則數；
  3. 空回覆重試 prompt（`_build_empty_final_response_prompt`）附上同樣的摘要塊。
- 驗收：重現範例1 第三輪（「保存你剛剛給我的程式碼」），模型 reasoning 不再出現「新的對話開始」，且能引用前輪程式碼內容。

### P1 — 失敗語義與可靠性（1–2 週級）

**P1-1 Guard 訊息與 tool error 分離（修 B7）**：evidence guard 攔截時，不回 tool error，改為：(a) 跳過該 tool call，(b) 注入一則 `role=system`（或 user-channel steering）訊息說明「證據已足夠，請直接綜合作答」，(c) UI 事件型別用 `runtime_steering` 而非失敗紅色。驗收：範例1 場景中不再出現紅色「失敗」卡，模型攔截後下一輪直接作答的比率上升。

**P1-2 統一錯誤分類法**：定義 enum（`tool_error / guard_steering / backend_error / truncated / empty_response / cancelled`），贯穿 react_loop 事件、execution transcript、前端 badge。前端據此決定顏色與文案。這是 2026-07-03 handoff「durable/transient event-lane contract」工作的一部分，一起做。

**P1-3 弱模型工具提示強化**：`ToolCallSimulator.TOOL_PROMPT_TEMPLATE` 加：每個工具 1 個最小調用範例（few-shot）、明確「不要在最終答案中出現 <tool_call> 標記」、參數型別中文說明。對 qwen 家族偵測到時直接改用其原生 XML 格式作為 *指示格式*（迎合訓練分佈而不是對抗它）。

**P1-4 S1 契約同步**：`_ACTIVE_GOAL_TURN_SELECTOR_SYSTEM_PROMPT` 的 JSON schema 補上 `lane` 欄位，或 parser 放寬；補一個 prompt↔parser 的 round-trip 測試。

**P1-5 S2/S3 回歸測試**：為「同一 run 內 iteration 歷史一致性」與「reasoning trace 不重複投影」各寫一個測試釘住行為，先量化再修。

**P1-6 history_window 改 token 制（修 S4）**：`AgentContext.get_recent_history` 以 token 預算裁切（複用 compaction 的估算器），`history_window` 保留為上限而非唯一準則。

### P2 — 向成熟 harness 收斂（1–2 月級）

- **P2-1 持久專案記憶**：等價 CLAUDE.md/AGENTS.md 的機制——workspace 根的 `MOCHI.md` 自動注入 system prompt；goal 完成時把關鍵決策寫回。彌補小模型跨 session 記憶。
- **P2-2 寫後驗證迴路**：file_write/exec 之後自動 re-read + lint/syntax check，失敗結果回饋模型（Claude Code 的 verification 習慣）。對弱模型，這是把「一次寫對」的要求降為「寫了能修」。
- **P2-3 Sub-agent 結果契約**：delegate_subagent 的回傳改 structured schema（如 Workflow 的 StructuredOutput 思路），避免父 agent 解析子 agent 自由文本。
- **P2-4 分檔重構**（配合既有 handoff 待辦）：react_loop 拆出 `evidence_guard.py`、`stream_assembly.py`、`thinking_parser.py`；engine.py 拆 invocation/budget/session 三塊；page.tsx 按既有計劃抽 controller。每次只拆一個、測試跟著搬。
- **P2-5 llama.cpp grammar / vLLM guided decoding**：工具輪 constrained decoding，把格式正確率從 prompt 工程層面提升到解碼層面——這是對弱模型最高槓桿的單點投資。
- **P2-6 模型能力檔案（model capability profile）**：per-model 設定：原生 tool calling、偏好工具格式、實測可靠 context、需要的重試策略。router 據此選策略，取代目前散落的探測+猜測。

### 優先順序建議

```
第 1 批：P0-1, P0-2, P0-3   ← 修「輸出截斷」與「輸出跑掉」，實測立即可感
第 2 批：P0-4, P0-5          ← 修「不能存檔」「失憶」，補齊 agent 的「手」和「記憶」
第 3 批：P1-1 ~ P1-6         ← 可靠性地基
第 4 批：P2 系列              ← 向成熟 harness 收斂
```

---

## 附錄：本次實際查證的檔案

- 截圖：`範例1/`（9 張）、`範例1/截斷後第2次測試/`（9 張中讀 7 張關鍵幀）、`範例2/`（7 張）
- 代碼：`mochi/backends/ollama.py`（context 段）、`mochi/backends/tool_call_simulator.py`（全文）、`mochi/backends/gguf.py`（策略段）、`mochi/agents/react_loop.py`（結構+finish_reason grep）、`mochi/agents/context.py`、`mochi/agents/compaction.py`（結構）、`mochi/agents/tool_exposure.py`（路由段）、`mochi/agents/engine.py`（預算段 grep）、`mochi/runtime/service.py`（resume/guidance 段）、`mochi/runtime/active_goal_turn_selector.py`（全文）、`mochi/config/schema.py`（num_ctx）、`mochi/config.yaml`（backend 段）
- 未逐行審查（結論中已標注推測處）：`openai_compat.py`、`local_models.py`、exec/sandbox 安全層、multi_agent protocols 細節
