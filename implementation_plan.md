# Mochi WebGUI 進階推理設定與互動 UX 升級

提升 Mochi WebGUI 到接近 LM Studio / Codex 等級的推理參數控制、對話互動品質、以及開發者工作流體驗。

## User Review Required

> [!IMPORTANT]
> 本次改動涉及 **後端 API 擴充 + 前端多個新元件 + Zustand 狀態管理重構**，影響面較廣。請特別審閱：
> - 是否接受 per-session 推理參數覆蓋（而非僅全域設定）
> - Slash command palette 的觸發方式與顯示邏輯
> - File change review/undo 的 scope（僅限 `file_write` 工具？還是包含 `shell` 命令的檔案修改？）

## Open Questions

> [!IMPORTANT]
> 1. **Preset 管理**：是否需要像 LM Studio 一樣支援多組命名 preset（system prompt + 參數組合）？還是先做單一全域 + per-session override？
> 2. **File change tracking scope**：目前 `file_write` 工具有 `require_approval` 機制，但 `shell` 命令的檔案修改是否也要追蹤？建議 v1 先只追蹤 `file_write` / `file_read` 工具。
> 3. **Markdown 渲染**：assistant 訊息目前是純文字 `whitespace-pre-wrap`，是否在本輪一併升級為 markdown 渲染（含程式碼高亮）？建議是，因為這是最高 ROI 的 UX 改善。

## 進度與決策（2026-05-17）

**已確認決策**
- 推理參數採 per-session override（ChatRequest 夾帶參數），解析優先序：override → active preset → defaults。
- File change tracking 範圍：v1 僅追蹤 `file_write` 工具（不追蹤 `shell` 命令變更）。
- 安全限制：新增 `security.file_ops_scope` 與 `file_undo_max_size_mb`，工具層強制驗證。

**已完成（後端）**
- `AgentConfig` / `SecurityConfig` / `InferencePreset` 已擴充；`configs/default.yaml` 已補齊預設與範例。
- `ChatRequest` 已支援推理參數 override；`AgentEngine.chat()` 解析 override → preset → defaults。
- `react_loop` / `events` 已加入 token stats 與 `generation_time_ms`；SSE 事件序列化已輸出。
- `file_write` 工具已產生 diff/undo metadata；新增 `POST /v1/tools/file/undo`。
- 各 LLM backend generate 參數已擴充並向下游轉發。

**進行中（前端）**
- `web/src/lib/api.ts` 已開始加入推理參數 request 支援；token stats + tool metadata 解析尚未完成。
- UI 元件（InferencePanel、FileChangeCard、CommandPalette、CopyButton、Slider）與 settings 頁面尚未落地。

---

## 現狀分析

### 後端

| 項目 | 現狀 | 缺口 |
|:---|:---|:---|
| `AgentConfig` | 有 `system_prompt`、`max_react_iterations`、`max_context_tokens` | 缺 `temperature`、`max_tokens`、`top_p`、`frequency_penalty`、`presence_penalty`、`show_token_stats` |
| `ChatRequest` | 只有 `message`、`session_id`、`model` | 無法 per-request 傳入推理參數 |
| `GenerationResult` | 有 `input_tokens`、`output_tokens` | 缺 `generation_time_ms`；SSE event 未攜帶 token 統計 |
| `FinalAnswerEvent` | 有 `content`、`trajectory_id` | 缺 token usage 與耗時資訊 |
| `AgentEngine.chat()` | 硬編碼 `temperature=0.7`、`max_tokens=4096` | 不從 config 讀取，也不接受 per-call override |
| File change tracking | `file_write` 工具有 approval 機制 | 無記錄/diff/undo surface |

### 前端

| 項目 | 現狀 | 缺口 |
|:---|:---|:---|
| Settings 模型區塊 | 只顯示「主要模型」一行摘要 | 無 temperature / context / system prompt 編輯 |
| Chat 頁面 | 無右側邊欄 | 無法快速調參 |
| ChatInput | 有模型選擇下拉 | 無 `/` command palette |
| ChatMessage | 純文字渲染 | 無複製按鈕、無 markdown、無 token stats |
| File changes | ToolCallCard 顯示工具呼叫 | 無 file diff view、無 undo |

---

## Proposed Changes

### Phase 1：後端推理參數與統計擴充

---

#### [MODIFY] [schema.py](file:///H:/_python/agent_mochi/mochi/config/schema.py)

`AgentConfig` 新增推理參數欄位：

```python
class AgentConfig(BaseModel):
    system_prompt: str = "..."
    max_react_iterations: int = 10
    max_context_tokens: int = 3000

    # --- 新增推理參數 ---
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, ge=1, le=131072)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    repeat_penalty: float = Field(default=1.0, ge=0.0, le=2.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)

    # --- 顯示設定 ---
    show_token_stats: bool = False
    """是否在模型輸出結束後顯示 token/s 等性能指標。"""
```

---

#### [MODIFY] [events.py](file:///H:/_python/agent_mochi/mochi/agents/events.py)

`FinalAnswerEvent` 新增 token usage 與耗時：

```python
@dataclass
class FinalAnswerEvent:
    type: Literal["final_answer"] = field(default="final_answer", init=False)
    content: str = ""
    trajectory_id: str | None = None
    # --- 新增 ---
    input_tokens: int = 0
    output_tokens: int = 0
    generation_time_ms: float = 0.0
    finish_reason: str = "stop"
```

---

#### [MODIFY] [chat.py](file:///H:/_python/agent_mochi/mochi/api/routes/chat.py)

1. `ChatRequest` 新增可選推理參數：

```python
class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    model: str | None = Field(default=None, min_length=1)
    # --- 新增 per-request overrides ---
    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    system_prompt: str | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
```

2. `_serialize_event` 對 `FinalAnswerEvent` 額外輸出 `input_tokens`、`output_tokens`、`generation_time_ms`、`finish_reason`

---

#### [MODIFY] [engine.py](file:///H:/_python/agent_mochi/mochi/agents/engine.py)

- `chat()` method 接受可選 `temperature`、`max_tokens`、`top_p` 等參數
- 若未傳入，從 `config.agent` 讀取
- 傳入 `react_loop.run()` 時使用 resolved 值
- 計時：在 `chat()` 開始與結束之間測量 `generation_time_ms`
- `FinalAnswerEvent` 填入累計 `input_tokens`、`output_tokens`、`generation_time_ms`

---

#### [MODIFY] [react_loop.py](file:///H:/_python/agent_mochi/mochi/agents/react_loop.py)

- `run()` 接受 `temperature`、`max_tokens` 等參數，傳給 `backend.generate()`
- 累計每一輪 `GenerationResult.input_tokens` 與 `output_tokens`

---

### Phase 2：Settings 頁面模型推理參數面板

---

#### [MODIFY] [settings/page.tsx](file:///H:/_python/agent_mochi/web/src/app/settings/page.tsx)

在「模型」tab 新增「推理參數」section，包含：

| 控制項 | UI 元件 | 範圍 |
|:---|:---|:---|
| System Prompt | `Textarea`（多行）| 自由文字 |
| Temperature | `Slider` + `Input` | 0.0 – 2.0，步進 0.1 |
| Max Tokens | `Input[number]` | 1 – 131072 |
| Top P | `Slider` + `Input` | 0.0 – 1.0 |
| Min P | `Slider` + `Input` | 0.0 – 1.0 |
| Top K | `Input[number]` | 0 – ∞ |
| Frequency Penalty | `Slider` + `Input` | -2.0 – 2.0 |
| Presence Penalty | `Slider` + `Input` | -2.0 – 2.0 |
| Repeat Penalty | `Slider` + `Input` | 0.0 – 2.0 |
| Context Length | 唯讀顯示（從 GGUF/model info 讀取）| — |
| Show Token Stats | `Switch` | on/off |

- 變更透過 `PATCH /v1/settings` 持久化到 `agent` section
- 保存後 emit `mochi:settings-updated` 事件

---

#### [NEW] [ui/slider.tsx](file:///H:/_python/agent_mochi/web/src/components/ui/slider.tsx)

新增 shadcn/ui Slider 元件（目前 ui/ 下沒有 slider）。

---

### Phase 3：Chat 頁面右側可收合推理參數側邊欄

---

#### [NEW] [chat/InferencePanel.tsx](file:///H:/_python/agent_mochi/web/src/components/chat/InferencePanel.tsx)

可收合的右側推理參數面板，類似 LM Studio 右 sidebar：

- **觸發**：header 區域新增 toggle 按鈕（`SlidersHorizontal` icon）
- **內容**：
  - System Prompt（可折疊 textarea）
  - Temperature slider
  - Max Tokens input
  - Top P / Min P slider
  - Frequency / Presence Penalty
  - Show Token Stats toggle
- **狀態管理**：新增 `inference-store.ts`（Zustand）
  - 保存 per-session 的推理參數 override
  - 若未設定，fallback 到全域 config
  - 面板開合狀態持久化到 localStorage
- **佈局**：chat 主區域改為 flex row，右側 panel 寬度 320px，可動畫收合

---

#### [MODIFY] [page.tsx (chat)](file:///H:/_python/agent_mochi/web/src/app/page.tsx)

- 引入 `InferencePanel`
- 主 chat 區域改為 `flex-1` + 右側 panel
- `handleSend` 攜帶當前推理參數到 API
- header 新增 panel toggle 按鈕

---

#### [NEW] [stores/inference-store.ts](file:///H:/_python/agent_mochi/web/src/lib/stores/inference-store.ts)

```typescript
interface InferenceParams {
  temperature: number
  maxTokens: number
  topP: number
  minP: number
  topK: number
  frequencyPenalty: number
  presencePenalty: number
  repeatPenalty: number
  systemPrompt: string
  showTokenStats: boolean
}

interface InferenceStore {
  panelOpen: boolean
  params: InferenceParams
  setPanelOpen: (open: boolean) => void
  setParam: <K extends keyof InferenceParams>(key: K, value: InferenceParams[K]) => void
  resetToDefaults: () => void
}
```

---

### Phase 4：Token 統計顯示

---

#### [MODIFY] [chat.ts](file:///H:/_python/agent_mochi/web/src/lib/chat.ts)

`Message` interface 新增：

```typescript
export interface Message {
  // ... existing fields ...
  tokenStats?: {
    inputTokens: number
    outputTokens: number
    generationTimeMs: number
    tokensPerSecond: number
    finishReason: string
  }
}
```

---

#### [MODIFY] [ChatMessage.tsx](file:///H:/_python/agent_mochi/web/src/components/chat/ChatMessage.tsx)

- 若 `message.tokenStats` 存在且 `showTokenStats` 開啟，在 assistant message 底部顯示：
  ```
  ⚡ 75.69 tok/sec  📥 361 tokens  ⏱ 0.24s  Stop reason: EOS Token Found
  ```
- 使用 `text-xs text-muted-foreground` 風格，與 LM Studio 對齊

---

### Phase 5：複製按鈕與訊息操作列

---

#### [MODIFY] [ChatMessage.tsx](file:///H:/_python/agent_mochi/web/src/components/chat/ChatMessage.tsx)

每個 assistant 訊息底部新增操作列：

```tsx
<div className="flex items-center gap-1 mt-2 opacity-0 group-hover:opacity-100 transition-opacity">
  <CopyButton text={content} />       {/* 複製全文 */}
  <RegenerateButton />                 {/* 重新生成（v2） */}
</div>
```

- `CopyButton`：點擊後複製到剪貼簿，圖示切換為 ✓ 1.5 秒
- 整個 message 容器加 `group` class

---

### Phase 6：Slash Command Palette（`/` 觸發）

---

#### [NEW] [chat/CommandPalette.tsx](file:///H:/_python/agent_mochi/web/src/components/chat/CommandPalette.tsx)

在 ChatInput 上方彈出的 command palette：

- **觸發**：在 textarea 空白狀態或行首輸入 `/` 時自動彈出
- **來源**：
  1. 內建命令：`/compact`、`/clear`、`/model`、`/voice`、`/settings`、`/export`
  2. Skills：從 `GET /v1/skills` 取得，以 `/skill-name` 格式觸發
- **UI**：
  - 浮動 popover，最大高度 320px，可滾動
  - 每項顯示：icon + name + description + tag (Personal/System)
  - 支援鍵盤導航（↑↓ 選擇、Enter 確認、Esc 關閉）
  - 支援模糊搜尋過濾
- **行為**：
  - 選擇 skill 後，在 textarea 插入 skill 名稱與 instruction hint
  - 選擇內建命令後，直接執行對應動作

---

#### [MODIFY] [ChatInput.tsx](file:///H:/_python/agent_mochi/web/src/components/chat/ChatInput.tsx)

- 監聽 `/` 輸入，判斷是否在行首
- 管理 palette open/close 狀態
- 傳遞 filter text 給 `CommandPalette`

---

### Phase 7：File Change Display with Review / Undo

---

#### [NEW] [chat/FileChangeCard.tsx](file:///H:/_python/agent_mochi/web/src/components/chat/FileChangeCard.tsx)

當 `ToolCallResultEvent` 來自 `file_write` 工具時，渲染為 file change card（類似 Codex）：

```
┌─────────────────────────────────┐
│ 📄 Edited gguf-investigation.md │
│    +260 -0                      │
│                    [Undo] [Review] │
└─────────────────────────────────┘
```

- **Review**：展開顯示 unified diff（新增行綠色、刪除行紅色）
- **Undo**：呼叫 `POST /v1/tools/file/undo` 恢復檔案（需後端支援）
- 若 `file_write` 工具記錄了 `original_content`，可計算 diff

---

#### [NEW] [api/routes/file_ops.py](file:///H:/_python/agent_mochi/mochi/api/routes/file_ops.py)

```python
# POST /v1/tools/file/undo
# 接受 { file_path, original_content, session_id }
# 將檔案恢復到 original_content
```

---

#### [MODIFY] [tools/file_write.py](file:///H:/_python/agent_mochi/mochi/tools) (或對應檔案)

- `file_write` 工具在寫入前讀取原始內容
- `ToolCallResultEvent` 攜帶 `original_content` 與 `new_content`
- 供前端計算 diff 與 undo

---

### Phase 8：額外 UX 優化（研究補充）

以下是研究 LM Studio、Codex、ChatGPT Desktop、Cursor 等產品後建議的額外 UX 改善：

---

#### 8.1 Markdown 渲染 + 程式碼高亮

#### [MODIFY] [ChatMessage.tsx](file:///H:/_python/agent_mochi/web/src/components/chat/ChatMessage.tsx)

- 引入 `react-markdown` + `rehype-highlight` 或 `shiki`
- assistant 訊息從 `<p>` 改為 `<ReactMarkdown>` 渲染
- 程式碼區塊加 copy button
- 新增 `package.json` 依賴：`react-markdown`、`remark-gfm`、`rehype-highlight`

---

#### 8.2 Scroll to Bottom 按鈕

#### [MODIFY] [page.tsx (chat)](file:///H:/_python/agent_mochi/web/src/app/page.tsx)

- 當使用者向上滾動超過一定距離時，顯示浮動「↓ 捲到底部」按鈕
- 點擊後 smooth scroll 到底部

---

#### 8.3 Stop Generation 按鈕

#### [MODIFY] [ChatInput.tsx](file:///H:/_python/agent_mochi/web/src/components/chat/ChatInput.tsx)

- 串流期間將 Send 按鈕替換為 Stop 按鈕（目前已有 `isStreaming` 方塊圖示，但沒有 abort 邏輯）
- 點擊後 abort SSE connection

---

#### 8.4 訊息時間戳顯示

#### [MODIFY] [ChatMessage.tsx](file:///H:/_python/agent_mochi/web/src/components/chat/ChatMessage.tsx)

- hover 時顯示訊息時間（relative time，如「2 分鐘前」）

---

#### 8.5 對話匯出

#### [NEW] [chat/ExportDialog.tsx](file:///H:/_python/agent_mochi/web/src/components/chat/ExportDialog.tsx)

- 支援匯出為 Markdown / JSON 格式
- 可從 header `⋯` 選單或 `/export` 命令觸發

---

#### 8.6 空狀態引導

#### [MODIFY] [page.tsx (chat)](file:///H:/_python/agent_mochi/web/src/app/page.tsx)

- 新對話空白時顯示引導卡片：
  - 建議 prompts / quick actions
  - 當前模型資訊
  - 快捷鍵提示

---

#### 8.7 訊息 Regenerate（重新生成）

- assistant 訊息操作列新增 `Regenerate` 按鈕
- 重新送出上一則 user message，替換當前 assistant 回覆

---

## 檔案變更總覽

### Backend (Python)

| 動作 | 檔案 | 說明 |
|:---|:---|:---|
| MODIFY | `mochi/config/schema.py` | AgentConfig 新增推理參數 |
| MODIFY | `mochi/agents/events.py` | FinalAnswerEvent 新增 token stats |
| MODIFY | `mochi/agents/engine.py` | chat() 接受推理參數、計時 |
| MODIFY | `mochi/agents/react_loop.py` | run() 傳遞推理參數 |
| MODIFY | `mochi/api/routes/chat.py` | ChatRequest 擴充、event 序列化 |
| NEW | `mochi/api/routes/file_ops.py` | file undo endpoint |
| MODIFY | `mochi/tools/file_write.py` | 記錄 original_content |
| MODIFY | `configs/default.yaml` | 新增預設推理參數 |

### Frontend (TypeScript/React)

| 動作 | 檔案 | 說明 |
|:---|:---|:---|
| NEW | `web/src/components/ui/slider.tsx` | shadcn Slider |
| NEW | `web/src/components/chat/InferencePanel.tsx` | 右側推理參數面板 |
| NEW | `web/src/components/chat/CommandPalette.tsx` | `/` command palette |
| NEW | `web/src/components/chat/FileChangeCard.tsx` | File diff + undo card |
| NEW | `web/src/components/chat/ExportDialog.tsx` | 對話匯出 |
| NEW | `web/src/lib/stores/inference-store.ts` | 推理參數 Zustand store |
| MODIFY | `web/src/lib/chat.ts` | Message 新增 tokenStats |
| MODIFY | `web/src/components/chat/ChatMessage.tsx` | Markdown + copy + stats + timestamp |
| MODIFY | `web/src/components/chat/ChatInput.tsx` | `/` 觸發 + stop generation |
| MODIFY | `web/src/app/page.tsx` | 右 sidebar 佈局 + scroll button + empty state |
| MODIFY | `web/src/app/settings/page.tsx` | 推理參數編輯面板 |
| MODIFY | `web/package.json` | 新增 react-markdown 等依賴 |

---

## Verification Plan

### Automated Tests

```bash
# Backend
uv run --extra dev python -m pytest tests/test_config.py tests/test_gguf_backend_runtime.py -q

# Frontend
cd web && npm run type-check && npm run lint
```

- 新增 `tests/test_chat_inference_params.py`：驗證 per-request 推理參數傳遞
- 新增 `tests/test_file_ops.py`：驗證 file undo endpoint

### Manual Verification

1. Settings 頁面：調整 temperature → 確認 `PATCH /v1/settings` 持久化 → 重新載入頁面確認值保留
2. Chat 右 sidebar：開啟面板 → 調整 temperature → 送出訊息 → 確認 SSE event 帶有對應 temperature
3. Token stats：開啟 show_token_stats → 送出訊息 → 確認 assistant 回覆下方顯示 tok/sec
4. Copy 按鈕：點擊 → 確認剪貼簿內容正確
5. `/` command palette：輸入 `/` → 確認彈出 skill 列表 → 鍵盤導航 → 選擇
6. File change card：觸發 `file_write` 工具 → 確認顯示 diff card → 點擊 Review → 確認 diff → 點擊 Undo → 確認檔案恢復
7. Markdown 渲染：送出需要程式碼回覆的 prompt → 確認程式碼高亮

### Browser Testing

- 使用 browser subagent 驗證：
  - 右 sidebar 開合動畫
  - Command palette 鍵盤導航
  - 複製按鈕 toast 效果
  - 響應式佈局（sidebar 在窄螢幕隱藏）

---

## 實作優先序建議

| 優先級 | Phase | 理由 |
|:---|:---|:---|
| P0 | Phase 1 (後端推理參數) | 所有前端功能的基礎 |
| P0 | Phase 2 (Settings 推理面板) | 最基本的參數管理 |
| P0 | Phase 5 (複製按鈕) | 最小改動、最高 ROI |
| P0 | Phase 8.1 (Markdown 渲染) | 最高 ROI 的 UX 改善 |
| P1 | Phase 3 (右 sidebar) | 核心互動體驗 |
| P1 | Phase 4 (Token stats) | 依賴 Phase 1 |
| P1 | Phase 6 (Command palette) | 進階互動 |
| P2 | Phase 7 (File changes) | 需要後端 file tracking |
| P2 | Phase 8.2-8.7 (其他 UX) | 錦上添花 |
