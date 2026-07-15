# Protected Workspace, Approval, Undo, and Auto Review Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 讓 Mochi 的 preview、approval、file mutation、undo、Auto Review 與 security audit 共用同一個不可變 change identity，並以 race-safe、crash-consistent 的檔案與程序隔離機制落實保護。

**Architecture:** 以 server-side immutable `ChangeManifest` 作為唯一執行來源；approval 只授權 canonical authorization envelope 的 digest，不再授權可被替換的 client payload。所有檔案變更經過 platform-specific `SafeFilesystem`、base hash/file identity CAS 與可重整的 transaction journal；Undo 產生反向 manifest，重走相同政策。Auto Review 與 audit 共用結構化 risk decision，OS sandbox 作為獨立的 defense-in-depth 層；`observe | enforce` rollout contract 在第一個行為變更前建立。

**Tech Stack:** Python 3.11–3.13, FastAPI, Pydantic v2, SQLite/WAL, asyncio, pytest/pytest-asyncio, POSIX `dir_fd/O_NOFOLLOW`, Windows Win32 handles plus a native AppContainer broker, React/TypeScript, Playwright CLI.

---

## 0. 查證結論與範圍修正

### 已確認的差距

| 項目 | 查證結果 | 本地證據 | 計劃處置 |
|---|---|---|---|
| Protected path / canonicalization | 已有 lexical + `Path.resolve(strict=False)` containment 與 protected directory policy，但只在 I/O 前檢查 | `mochi/utils/security.py:187-338` | 保留 policy，新增 handle-pinned I/O |
| Read scope | 確認 read 被強制改成 `any`，即使 caller 傳入 `workspace` | `mochi/utils/security.py:313-315` | 拆分 read/write scope，預設 workspace |
| Symlink/junction/hardlink/TOCTOU | 確認 mutation path check 與 `open/unlink` 分離；Mochi mutation path 無 `O_NOFOLLOW`、`dir_fd`、file identity、reparse point 或 hardlink guard | `mochi/tools/file_ops.py:601-622,886-902`；全域搜尋只在無關 auth store 找到 `os.replace` | 建立跨平台 SafeFilesystem |
| Single-file atomic write | 確認 `open("w")` 直接 truncate | `mochi/tools/file_ops.py:614-622` | temp + flush/fsync + atomic replace |
| Multi-file patch | 確認逐檔 delete/write，失敗會留下部分套用 | `mochi/tools/file_ops.py:886-902` | crash-consistent journal + rollback；不宣稱跨檔原子 |
| Stale guard replay | 確認 approval replay 設定 state，stale guard 明確提前 return | `mochi/runtime/service.py:7131-7152`；`mochi/tools/file_ops.py:923-974` | replay 仍做 CAS，失配回 conflict |
| Approval binding | schema 只有 tool/arguments/metadata；沒有 workspace identity、base hash、patch digest、policy version | `mochi/runtime/store.py:85-98,755-790`；`mochi/runtime/models.py:41-64` | approval 綁 `change_set_id + request_digest` |
| Approval TTL / CAS / consume-once | 兩套 approval store 都可重複 resolve；task store UPDATE 沒有 `status='pending'` 條件 | `mochi/runtime/approvals.py:112-151,261-306`；`mochi/runtime/store.py:840-870` | 狀態機 + conditional UPDATE + TTL |
| Preview override | edited patch 可作 replay override；API 沒有 digest token | `mochi/runtime/models.py:41-64`；`mochi/api/routes/workspace.py:236-291`；`mochi/api/routes/approvals.py:27-46` | edited patch 產生新 manifest/approval |
| Undo trust | client 傳 `file_path/original_content/action`，server 直接 unlink/write | `mochi/api/routes/file_ops.py:25-79`；`web/src/app/page.tsx:4731-4742` | client 只傳 change/entry ID；server-side CAS |
| Auto Review | `auto_review` preset 關閉 approval；policy `ask` 可被 deterministic allow，並以 metadata 標記 | `mochi/security/policy.py:26-36`；`mochi/tools/exec_command.py:246-287,434-449`；`tests/test_exec_tools.py:199-228` | 結構化 reviewer + fail-closed thresholds |
| Saved config/rules | `save_config` 直接 `write_text`，沒有 lock/replace/fsync | `mochi/config/manager.py:291-305` | 原子設定存檔 + revision/CAS |
| Redaction/audit | execution transcript 有 key-based redaction，但不是所有 store 的統一 persistence boundary；一般 mutation/undo/Auto Review 沒有一致 security audit | `mochi/runtime/execution_transcript.py:9-19,353-372`；`mochi/runtime/store.py` | 中央 redactor + security audit table |
| OS sandbox | task sandbox 是 cwd/workspace routing；exec 最終仍為宿主 subprocess | `mochi/runtime/service.py:622-638,5577-5580`；`mochi/runtime/exec_runtime.py` | sandbox backend interface + platform rollout |
| Diff fidelity | 後端仍以 porcelain v1 + UTF-8 replacement decoding + text unified diff 處理 | `mochi/api/routes/workspace.py:433-525,561-633` | porcelain v2 -z、binary/mode/rename/EOF metadata |

### 對 subagent 報告的修正

1. 前端 DiffViewer 的「填空白重建內容」證據已過時：目前 `web/src/lib/diff-lines.ts` 直接解析 unified diff，UI gap 已在本輪先前工作修正。後端 fidelity gap 仍成立。
2. OpenClaw 的 `O_NOFOLLOW/dir_fd` helper 是 POSIX 實作參考，不能直接解決 Windows junction/reparse point；Windows 必須用 Win32 handle identity。
3. 多個檔案無法在一般檔案系統上真正原子提交。本計劃承諾的是「全部預先驗證、可回滾、crash recovery」，不使用不準確的 atomic transaction 宣稱。
4. Claude Code/Codex 的外部產品行為只能作 UX/能力參考；本計劃的安全要求以 Mochi 與本地 `reference/` 原始碼證據為準。

### Reference 中已驗證且可採用的模式

- OpenClaw lexical + canonical mount + pinned parent：
  `reference/openclaw/src/agents/sandbox/fs-bridge-path-safety.ts:82-210`
- OpenClaw POSIX descriptor-relative walk、temp write、fsync、replace、hardlink guard：
  `reference/openclaw/src/agents/sandbox/fs-bridge-mutation-helper.ts:25-125`
- OpenClaw symlink rebind race harness：
  `reference/openclaw/src/test-utils/symlink-rebind-race.ts`
- OpenClaw approval expiry、double-resolve guard、consume-once：
  `reference/openclaw/src/gateway/exec-approval-manager.ts:87-173`
- Hermes shadow checkpoint/restore：
  `reference/hermes-agent/tools/checkpoint_manager.py:275-502`
- ZeroClaw sandbox backend abstraction與偵測：
  `reference/zeroclaw/crates/zeroclaw-runtime/src/security/`
- ZeroClaw trace temp rewrite + 0600 + rename：
  `reference/zeroclaw/crates/zeroclaw-runtime/src/observability/runtime_trace.rs:109-140`

## 1. 不可妥協的安全 invariant

1. 實際 I/O 使用的 target 必須與 policy 檢查、preview、approval 中的 target identity 相同。
2. Approval 只能授權一個 immutable manifest digest；改 patch、workspace、base 或 policy version 都要重新審批。
3. Approval resolve 只能成功一次；過期、非 pending、requester mismatch 均不得執行。
4. 執行前必須再次驗證 base content hash、file identity、workspace identity 與 protected path policy。
5. 單檔 write 不能暴露 truncate 中間態；多檔 patch 失敗或程序重啟後必須能 rollback/recover。
6. Undo 不接受 client 提供的原始內容；目前內容不等於 recorded after hash 時不得覆蓋。
7. Auto Review 不得覆蓋 protected path、workspace escape、require_escalated、無法解析或 identity mismatch。
8. Security audit 在持久化前 redaction；預設記 digest/risk factor，不記 secret/raw file body。
9. `sandbox=required` 時沒有可用 backend 必須 fail closed，不得悄悄降級為普通 subprocess。
10. File 與 exec approval 都必須綁定明確版本的 authorization envelope；執行階段不得重新解讀未進 digest 的 shell、cwd、env、sandbox 或 escalation 欄位。

## 2. 目標資料契約
```python
@dataclass(frozen=True)
class FileIdentity:
    platform: Literal["posix", "windows"]
    volume_id: str
    file_id: str
    link_count: int
    is_reparse_point: bool
@dataclass(frozen=True)
class ChangeEntry:
    entry_id: str
    relative_path: str
    operation: Literal["add", "update", "delete", "rename"]
    base_sha256: str | None
    after_sha256: str | None
    base_identity: FileIdentity | None
    before_blob_id: str | None
    after_blob_id: str | None
    mode_before: int | None
    mode_after: int | None
    base_metadata_sha256: str | None
    after_metadata_sha256: str | None
    rename_source: str | None
    dependency_group: str | None
@dataclass(frozen=True)
class ChangeManifest:
    version: int
    change_set_id: str
    workspace_root: str
    workspace_identity: FileIdentity
    tool_name: str
    intent: Literal["mutate", "undo"]
    entries: tuple[ChangeEntry, ...]
    patch_sha256: str | None
    policy_version: str
    created_at: str
    expires_at: str
    request_digest: str
@dataclass(frozen=True)
class AppliedChangeRecord:
    change_set_id: str
    entry_id: str
    applied_sha256: str | None
    applied_identity: FileIdentity | None
    applied_metadata_sha256: str | None
    applied_at: str
@dataclass(frozen=True)
class AuthorizationContext:
    requester_id: str
    session_id: str
    task_id: str | None
    workspace_root: str
    workspace_identity: FileIdentity
@dataclass(frozen=True)
class AuthorizationEnvelope:
    schema_version: int
    kind: Literal["file_change", "exec"]
    context: AuthorizationContext
    policy_version: str
    file_request: FileChangeRequest | None
    exec_request: ExecRequest | None
```
`request_digest = sha256(canonical_json(AuthorizationEnvelope))`。Canonical JSON 使用 UTF-8、排序 key、固定 separators、整數 epoch/固定 enum；禁止 `Path`、locale datetime、浮點數與未知欄位。`change_set_id`、`created_at`、`expires_at`、`request_digest` 本身與 UI metadata **不進 digest**，避免循環與時間造成不穩定。
`file_request` 必須包含按 ordinal 排序的完整 `ChangeEntry` projection（path、operation、base/after content hash、base/desired metadata hash、base identity、mode、rename source/dependency group、patch hash）。`exec_request` 必須包含 exact command UTF-8 bytes hash、shell/executable/argv、resolved cwd、allowlisted env key 及 value hash、workspace identity、network policy、resource limits、requested escalation、sandbox backend/capability plan digest。兩種 request 都包含 requester/session/task/workspace context，因此相同 payload 不會跨 authorization context dedupe。
Preview idempotency key 是 `(schema_version, kind, requester_id, session_id, task_id, workspace_identity, request_digest)`；相同 scope 只重用仍為 `prepared` 且未過期的 manifest。DB 不以裸 `request_digest UNIQUE` 作全域 dedupe，而使用上述 composite unique index。任何 envelope 欄位改變都產生新 digest。
## 3. 檔案結構

### 新增

- `mochi/security/file_contract.py` — file/workspace identity、manifest digest、security conflict types
- `mochi/security/safe_filesystem.py` — backend protocol與平台選擇
- `mochi/security/safe_fs_posix.py` — `dir_fd/O_NOFOLLOW` 實作
- `mochi/security/safe_fs_windows.py` — Win32 reparse/file-ID/link-count 實作
- `mochi/tools/file_transaction.py` — staging、journal、commit、rollback、recovery
- `mochi/runtime/change_sets.py` — manifest/blob persistence facade
- `mochi/security/auto_review.py` — risk factors與結構化 decision
- `mochi/runtime/security_audit.py` — audit event與 persistence redaction
- `mochi/runtime/sandbox/base.py` — OS sandbox capability/backend protocol
- `mochi/runtime/sandbox/linux.py` — bubblewrap backend
- `mochi/runtime/sandbox/windows.py` — native broker adapter與 capability probe
- `native/mochi-sandbox-windows/CMakeLists.txt` — Windows broker build
- `native/mochi-sandbox-windows/src/main.cpp` — versioned JSON protocol、AppContainer/Job lifecycle
- `.github/workflows/security-platform.yml` — Windows/Linux 真 backend adversarial gate
- `tests/security/test_safe_filesystem.py`
- `tests/security/test_file_transaction.py`
- `tests/security/test_change_manifest.py`
- `tests/security/test_approval_lifecycle.py`
- `tests/security/test_auto_review.py`
- `tests/security/test_security_audit.py`
- `tests/security/test_os_sandbox.py`

### 主要修改

- `mochi/utils/security.py`
- `mochi/tools/file_ops.py`
- `mochi/tools/file_mutations.py`
- `mochi/runtime/approvals.py`
- `mochi/runtime/store.py`
- `mochi/runtime/models.py`
- `mochi/runtime/service.py`
- `mochi/runtime/exec_runtime.py`
- `mochi/api/routes/approvals.py`
- `mochi/api/routes/file_ops.py`
- `mochi/api/routes/workspace.py`
- `mochi/config/schema.py`
- `mochi/config/manager.py`
- `web/src/lib/api.ts`
- `web/src/components/chat/TaskPanel.tsx`
- `web/src/components/chat/FileChangeCard.tsx`
- `web/src/app/page.tsx`

---

## Task 0: 先建立 rollout contract 與 baseline gates
**Files:**
- Modify: `mochi/config/schema.py`
- Modify: `mochi/runtime/service.py`
- Modify: `mochi/api/routes/settings.py`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/app/settings/page.tsx`
- Modify: `tests/test_config.py`
- Modify: `tests/test_api_sessions_settings.py`
- Modify: `tests/test_api_runtime.py`
- [ ] **Step 1: 寫 rollout config/API failing tests**
新增 `security.change_contract_mode = "observe" | "enforce"`，migration default 為 `observe`；settings round-trip、未知值拒絕、session projection 都要測。現行 schema 應 RED。
- [ ] **Step 2: 實作兩個互相獨立的 rollout gates**
`change_contract_mode=observe` 只 shadow 建立 file envelope/digest、跑 validation 並記 `would_reject` metrics，不改變既有 file mutation/Undo 結果；`enforce` 才把 file execution source 切至 manifest/SafeFilesystem 並關閉 legacy replay/raw Undo。這個 flag **不控制 exec sandbox**：`sandbox.mode=required` 在任何 change mode 都必須 fail closed。API/UI 分別顯示 change mode 與 sandbox mode/capabilities。
| Change mode | Sandbox mode | Edited patch / legacy Undo | Exec |
|---|---|---|---|
| observe | off/preferred | 走 legacy，另記 would-reject；不宣稱安全 | 依 sandbox mode |
| observe | required | file 仍 shadow；legacy 行為不影響 exec | 無真 backend 即拒絕 |
| enforce | off/preferred | edited patch 409、legacy raw Undo 拒絕 | 依 sandbox mode |
| enforce | required | 完整 file contract 強制 | 完整 sandbox contract 強制 |
- [ ] **Step 3: 建立 dual-path regression test helper**
在 `tests/test_api_runtime.py` 提供 parameterized fixture，後續 Tasks 1–9 的 file integration tests 都覆蓋 observe 不阻斷、enforce fail-closed；Task 10 exec tests 只依 `sandbox.mode` 判斷，不讀 `change_contract_mode`。不得加入 replay bypass 的第三種混合模式。
- [ ] **Step 4: Run**
```powershell
rtk pytest tests/test_config.py tests/test_api_sessions_settings.py tests/test_api_runtime.py -q
```
- [ ] **Step 5: Commit**
```powershell
rtk git add mochi/config/schema.py mochi/runtime/service.py mochi/api/routes/settings.py web/src/lib/api.ts web/src/app/settings/page.tsx tests/test_config.py tests/test_api_sessions_settings.py tests/test_api_runtime.py
rtk git commit -m "feat: add protected workspace rollout contract"
```
---

## Task 1: 建立跨平台 file identity 與 pinned-target contract

**Files:**
- Create: `mochi/security/file_contract.py`
- Create: `mochi/security/safe_filesystem.py`
- Create: `mochi/security/safe_fs_posix.py`
- Create: `mochi/security/safe_fs_windows.py`
- Create: `tests/security/test_safe_filesystem.py`
- Modify: `mochi/utils/security.py`

- [ ] **Step 1: 寫 authorization envelope digest 與 FileIdentity 的 failing unit tests**

覆蓋 canonical JSON 穩定性、volatile manifest 欄位不改 digest、workspace/requester/session/task/policy/sandbox/env-value hash 任一改變會改 digest、entry 順序固定、未知欄位拒絕，以及相同 preview 只在相同 authorization context 內 idempotent。

Run:

```powershell
rtk pytest tests/security/test_safe_filesystem.py -k "identity or digest" -q
```

Expected: FAIL，因 `file_contract` 尚不存在。

- [ ] **Step 2: 實作 immutable dataclass/Pydantic contract、明確 digest projection 與 canonical SHA-256**

依「目標資料契約」建立 `AuthorizationContext`、file/exec request union 與 versioned envelope。禁止以 `Path` object、locale 時間、非排序 dict 或 envelope 外欄位直接進 digest；`ChangeManifest.request_digest` 只能由 envelope 計算後注入。

- [ ] **Step 3: 寫 POSIX symlink rebind 與 hardlink failing tests**

測試在 validation 後、write 前切換 parent symlink；hardlink `st_nlink > 1` 必須拒絕。

- [ ] **Step 4: 實作 POSIX pinned parent walk 與 descriptor-relative mutation primitive**

使用 `os.open(..., O_DIRECTORY | O_NOFOLLOW)`、relative segments、`dir_fd`、`fstat`；禁止 `..` 與 symlink traversal。temp create、replace、unlink 都只能接受已開啟 parent `dir_fd` 與 basename，禁止在 mutation syscall 重新使用 absolute path。

- [ ] **Step 5: 寫 Windows reparse/file-ID failing tests**

實機測試建立可用 symlink/junction；若 CI 權限不允許 junction，保留 Win32 adapter fake 測試並標記實機 gate。另以 `os.link` 測 hardlink。

- [ ] **Step 6: 實作 Windows handle-pinned parent 與 handle-relative mutation primitive**

以 `CreateFileW(FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS)` 開 workspace/parent/target，使用 `GetFinalPathNameByHandleW`、`GetFileInformationByHandleEx(FileIdInfo)` 取得 volume/file ID，拒絕 unexpected reparse point 與 link count > 1。`safe_fs_windows.py` 以 `NtCreateFile` 的 `OBJECT_ATTRIBUTES.RootDirectory` 建 relative temp，以 `NtSetInformationFile(FileRenameInformationEx/FileDispositionInformationEx)` 或經 rebind tests 證明等價的 handle-relative API 完成 replace/delete，並驗證 after file ID；禁止 absolute path fallback。API/flag 不可用時 `enforce` fail closed。

- [ ] **Step 7: 將 `check_file_tool_path` 降為 policy preflight**

它仍負責 raw path/protected path/lexical scope；真正 mutation 必須持有 `SafeTarget` 才能執行。

- [ ] **Step 8: 跑跨平台測試與靜態檢查**

```powershell
rtk pytest tests/security/test_safe_filesystem.py tests/test_security_policy.py -q
rtk ruff check mochi/security tests/security
rtk pyright
```

- [ ] **Step 9: Commit**

```powershell
rtk git add mochi/security/file_contract.py mochi/security/safe_filesystem.py mochi/security/safe_fs_posix.py mochi/security/safe_fs_windows.py mochi/utils/security.py tests/security/test_safe_filesystem.py
rtk git commit -m "security: pin file identities for workspace mutations"
```

## Task 2: 單檔 atomic write 與 crash-safe temp lifecycle
**Files:**
- Create: `mochi/tools/file_transaction.py`
- Create: `tests/security/test_file_transaction.py`
- Modify: `mochi/security/safe_filesystem.py`
- Modify: `mochi/security/safe_fs_posix.py`
- Modify: `mochi/security/safe_fs_windows.py`
- Modify: `mochi/tools/file_ops.py`
- [ ] **Step 1: 寫 bytes、identity 與 metadata failure-injection tests**
覆蓋 write 中途 exception、fsync 前終止、replace 失敗、temp cleanup，以及 validation 後 rebind symlink/junction；原檔必須完整，outside target 不得改變。建立 POSIX ACL/xattr/uid/gid 與 Windows owner/group/DACL/security descriptor fixtures，replace/rollback 後 metadata 必須相同；無法完整 capture/apply 時 enforce 明確拒絕。
- [ ] **Step 2: 確認測試 RED**
```powershell
rtk pytest tests/security/test_file_transaction.py -k "atomic_write or metadata" -q
```
Expected: FAIL，現行 writer 直接 `open("w")` 且未保存 metadata。
- [ ] **Step 3: 實作 `atomic_write_bytes(SafeTarget, bytes, metadata_snapshot)`**
在 pinned parent 內以 exclusive relative operation 建 temp，先取得 temp basename/file identity，再寫入 bytes 與 security metadata。POSIX capture/apply uid、gid、mode、ACL 與全部 xattrs；Windows 以 handle capture/apply self-relative security descriptor（owner/group/DACL，能存取 SACL 時一併保存）。無法完整 capture/apply 的 existing file 回 `unsupported_security_metadata`。完成 flush/fsync 後 revalidate base，handle-relative replace、fsync parent；rename 保留 staged temp identity，因此 journal 可辨識 successor。禁止 path fallback。
- [ ] **Step 4: 將 FileWrite/FileEdit 共用 atomic writer**
append 也先組合完整 after bytes，再走 CAS + replace；不再使用 `open("a")`。新增檔案使用明確 policy default metadata，不複製無關 parent ACL 以外的 host metadata。
- [ ] **Step 5: 驗證既有 stale/undo metadata 測試不退化**
```powershell
rtk pytest tests/security/test_file_transaction.py tests/test_tool_system_upgrade.py -q
```
- [ ] **Step 6: Commit**
```powershell
rtk git add mochi/security/safe_filesystem.py mochi/security/safe_fs_posix.py mochi/security/safe_fs_windows.py mochi/tools/file_transaction.py mochi/tools/file_ops.py tests/security/test_file_transaction.py
rtk git commit -m "security: make single-file mutations crash safe"
```
## Task 3: Immutable ChangeManifest、blob references 與 transaction journal
**Files:**
- Create: `mochi/runtime/change_sets.py`
- Create: `tests/security/test_change_manifest.py`
- Modify: `mochi/runtime/store.py`
- Modify: `mochi/runtime/service.py`
- Modify: `mochi/tools/file_mutations.py`
- Modify: `mochi/tools/file_transaction.py`
- Modify: `mochi/tools/file_ops.py`
- Modify: `tests/test_runtime_store.py`
- [ ] **Step 1: 寫 RuntimeStore migration failing tests**
新增 tables：
```sql
change_sets(
  id PRIMARY KEY, schema_version, requester_id, session_id, task_id,
  workspace_root, workspace_identity_json, tool_name, intent,
  request_digest, authorization_envelope_json, patch_sha256, policy_version,
  status, created_at, expires_at, applied_at, updated_at, metadata_json
)
change_entries(
  id PRIMARY KEY, change_set_id, ordinal, relative_path, operation,
  base_sha256, after_sha256, base_identity_json,
  before_blob_id, after_blob_id, mode_before, mode_after,
  base_metadata_blob_id, after_metadata_blob_id,
  rename_source, dependency_group
)
applied_change_entries(
  change_set_id, entry_id, applied_sha256, applied_identity_json,
  applied_metadata_sha256, applied_at,
  PRIMARY KEY(change_set_id, entry_id)
)
change_blobs(id PRIMARY KEY, sha256 UNIQUE, size_bytes, content BLOB, created_at)
blob_references(
  blob_id, owner_type, owner_id, purpose, retained_until, state,
  PRIMARY KEY(blob_id, owner_type, owner_id, purpose)
)
undo_retention(
  change_set_id, entry_id, status, retained_until, expired_at,
  PRIMARY KEY(change_set_id, entry_id)
)
file_transaction_journal(
  id PRIMARY KEY, change_set_id, status, phase, error, created_at, updated_at
)
file_transaction_entries(
  journal_id, entry_id, ordinal, state, base_sha256, after_sha256,
  base_identity_json, staged_name, staged_identity_json,
  rollback_blob_id, rollback_staged_name, rollback_staged_identity_json,
  rollback_successor_identity_json, base_metadata_blob_id,
  last_error, updated_at,
  PRIMARY KEY(journal_id, entry_id)
)
CREATE UNIQUE INDEX change_set_idempotency
ON change_sets(
  schema_version, requester_id, session_id, IFNULL(task_id, ''),
  workspace_identity_json, request_digest
);
```
- [ ] **Step 2: 實作 idempotent schema migration與 indexes**
使用既有 `_ensure_column` pattern；migration 重跑不得丟資料。
- [ ] **Step 3: 寫 manifest、blob reference 與 retention tests**
同 ID/digest 不同 envelope conflict；blob hash dedupe。Transaction blob 先建 `purpose=rollback` ref，terminal 前不得 GC；長期 Undo 以 `undo_retention.status/retained_until` 與 `purpose=undo` ref 管理。Cleanup 在單一 DB transaction expire refs，只刪除沒有 active ref 的 blob。未保留或到期仍保留 authoritative status，API 才能穩定回 410。
- [ ] **Step 4: 將 `prepare_apply_patch` 產出 manifest 而非可變 metadata-only payload**
保留 UI projection，但 execution source 必須是 persisted manifest。Preview capture base content/identity/security-metadata hash；unsupported metadata 在 enforce preview 即拒絕。
- [ ] **Step 5: 寫完整 crash-point reconciliation matrix**
逐一在 staged temp name/identity journal、blob/metadata fsync、`applying` commit、namespace replace/unlink、`applied` commit、rollback temp identity journal、rollback restore 與 cleanup 後 kill/restart。Recovery 同時比對 content、metadata 與 identity：base identity+hash 代表未套用；staged identity+after hash 代表 replace 已套用；rollback staged/successor identity+base hash/metadata 代表已回復；兩者皆非才是 interference。第 N 個 mutation 失敗時逆序回復。
- [ ] **Step 6: 實作 stage-all / validate-all / commit / rollback / recover**
固定 ordering：stage rollback/content/metadata blobs與 temp → fsync → SQLite 持久化每個 staged basename/identity/blob ref → WAL durable boundary → entry 設 `applying` → handle-pinned mutation → fsync parent → 讀回 successor identity/content/metadata，寫 `applied_change_entries` 與 entry `applied`。Rollback 也先 durable 記 rollback temp identity再 replace。`RuntimeService.start()` 在接受新 mutation 前依 Step 5 matrix recovery，不依賴 cursor。
- [ ] **Step 7: Run**
```powershell
rtk pytest tests/security/test_change_manifest.py tests/security/test_file_transaction.py tests/test_runtime_store.py tests/test_tool_system_upgrade.py -q
rtk pyright
```
- [ ] **Step 8: Commit**
```powershell
rtk git add mochi/runtime/change_sets.py mochi/runtime/store.py mochi/runtime/service.py mochi/tools/file_mutations.py mochi/tools/file_transaction.py mochi/tools/file_ops.py tests/security/test_change_manifest.py tests/security/test_file_transaction.py tests/test_runtime_store.py
rtk git commit -m "feat: persist immutable file change manifests"
```
## Task 4: Approval TTL、request binding、CAS resolve 與 consume-once
**Files:**
- Create: `tests/security/test_approval_lifecycle.py`
- Modify: `mochi/runtime/approvals.py`
- Modify: `mochi/runtime/store.py`
- Modify: `mochi/runtime/models.py`
- Modify: `mochi/runtime/service.py`
- Modify: `mochi/api/routes/approvals.py`
- Modify: `mochi/security/file_contract.py`
- Modify: `tests/test_exec_security.py`
- Modify: `tests/test_api_runtime.py`
- [ ] **Step 1: 定義含 `approve_and_save_rule` 的 state machine tests**
`pending → approved_once/rejected/expired/superseded`；`approved_once → consuming → consumed/execution_failed/expired`。`approve_once` 與 `approve_and_save_rule` 都產生同一個 single-use `approved_once`，後者只多一個 durable rule side-effect outbox，不得另闢 bypass。任何第二次 resolve/consume 回 409；resolve/consume TTL 到期回 410；context/digest mismatch 回 403/409。
- [ ] **Step 2: 確認現行 implementation 會 RED**
```powershell
rtk pytest tests/security/test_approval_lifecycle.py -q
```
- [ ] **Step 3: 擴充兩套 approval schema 與 outbox**
新增 context/digest/TTL、`resolution_kind`、consume lease/idempotency欄位；`execution_idempotency_key` UNIQUE。新增 `approval_side_effects(side_effect_id, approval_id, kind, payload_digest, target_config_path, status, attempts, lease_owner, lease_expires_at, last_error, created_at, delivered_at, updated_at)`，`UNIQUE(approval_id, kind, payload_digest)`。
- [ ] **Step 4: 使用 conditional UPDATE 實作 resolve CAS**
`approve_and_save_rule` 在同一個 SQLite transaction 以 `WHERE status='pending' AND expires_at>now` resolve 成 `approved_once`，並插入 idempotent `save_command_rule` outbox；任一步失敗 rollback。`approve_once` 不建 outbox。兩者都檢查 rowcount/context/digest。
- [ ] **Step 5: 實作 consume-once CAS 與 lease recovery**
只有 `approved_once AND expires_at>now` 且 context/digest 全匹配，才以唯一 idempotency key 進 `consuming`；`resolution_kind` 不改變 consume path。Restart 對 lease 過期查 change-set/exec outcome：已套用→consumed；確認未開始且仍有效→approved_once；已過期→expired；不明→execution_failed fail closed。
- [ ] **Step 6: 定義 rule side-effect delivery contract**
Current request 不等待 YAML writer，也不得因 save-rule failure重跑 execution。Task 9 worker以 outbox idempotency key、config ETag/lock投遞；成功→side effect applied，永久失敗→failed並在 API/UI 明確顯示「本次已批准，但規則未保存」。這是跨 SQLite/YAML 的 transactional outbox，不宣稱不可能的跨 store atomic commit。
- [ ] **Step 7: 定義 canonical exec authorization envelope**
Tests 證明 command bytes、shell/argv、cwd、env value hash、workspace、network/resource limits、escalation與 `SandboxPlan` 任一改變都產生新 digest。Audit/approval只存 env hash。
- [ ] **Step 8: API 映射 typed domain errors**
`ApprovalExpired→410`、`ApprovalConflict→409`、`ApprovalRequesterMismatch→403`；response 帶 `resolution_kind/rule_persistence_status`。
- [ ] **Step 9: Run**
```powershell
rtk pytest tests/security/test_approval_lifecycle.py tests/test_exec_security.py tests/test_api_runtime.py tests/test_runtime_store.py -q
```
- [ ] **Step 10: Commit**
```powershell
rtk git add mochi/runtime/approvals.py mochi/runtime/store.py mochi/runtime/models.py mochi/runtime/service.py mochi/api/routes/approvals.py mochi/security/file_contract.py tests/security/test_approval_lifecycle.py tests/test_exec_security.py tests/test_api_runtime.py
rtk git commit -m "security: bind approvals to expiring single-use requests"
```
## Task 5: Preview、approval 與 execution digest binding
**Files:**
- Modify: `mochi/api/routes/workspace.py`
- Modify: `mochi/api/routes/approvals.py`
- Modify: `mochi/runtime/approvals.py`
- Modify: `mochi/runtime/store.py`
- Modify: `mochi/runtime/service.py`
- Modify: `mochi/runtime/models.py`
- Modify: `mochi/tools/file_ops.py`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/components/chat/TaskPanel.tsx`
- Create: `web/scripts/test-approval-contract.mjs`
- Modify: `web/package.json`
- Test: `tests/test_api_runtime.py`
- Test: `tests/test_api_workspace.py`
- [ ] **Step 1: 寫 preview 與 dual-path contract failing tests**
成功 preview 回 change-set/digest/expiry/policy；相同 context+payload idempotent，base/context/policy改變產生新 digest。所有 edited-patch tests parameterize `observe/enforce`。
- [ ] **Step 2: 修改 `POST /v1/workspace/patch/preview`**
Server持久化 prepared manifest/envelope；response只是 projection，DB使用 composite idempotency scope。
- [ ] **Step 3: 依 rollout mode處理 client replay override**
`observe` 保留舊 `ApprovalReplayOverride.patch_text` execution，shadow建立新 preview並記 `would_reject_edited_patch`，response/UI標示不受新契約保護；不得把 shadow approval拿去執行。`enforce` 回 `409 edited_patch_requires_new_preview`，完全不信任 override。
- [ ] **Step 4: 新增 superseding approval 原子流程**
Enforce新 preview 時，在同一 store transaction CAS舊 pending→superseded並建立新 approval；舊 resolve回409。Observe只模擬/記metric，不改舊 approval lifecycle。
- [ ] **Step 5: 只在 enforce移除 stale guard replay bypass**
Enforce execution從 manifest取 base/identity並立即 CAS；observe仍走 legacy guard semantics並記 would-conflict，不可把 shadow結果當安全決策。
- [ ] **Step 6: 定義 conflict response**
Enforce的 base/file/workspace/context/policy mismatch不寫入，change set conflicted、API 409。Observe回 legacy result加 shadow reason。
- [ ] **Step 7: 更新 Web API types、TaskPanel 與 contract test**
顯示 mode、digest、expiry、policy、stale/superseded與「observe未強制」警告；contract script覆蓋兩模式。
- [ ] **Step 8: Run**
```powershell
rtk pytest tests/test_api_runtime.py tests/test_api_workspace.py tests/test_tool_system_upgrade.py -q
rtk npm --prefix web run test:approval-contract
rtk npm --prefix web run type-check
```
- [ ] **Step 9: Commit**
```powershell
rtk git add mochi/api/routes/workspace.py mochi/api/routes/approvals.py mochi/runtime/approvals.py mochi/runtime/store.py mochi/runtime/service.py mochi/runtime/models.py mochi/tools/file_ops.py web/src/lib/api.ts web/src/components/chat/TaskPanel.tsx web/scripts/test-approval-contract.mjs web/package.json tests/test_api_runtime.py tests/test_api_workspace.py
rtk git commit -m "security: bind patch approvals to immutable previews"
```
## Task 6: Server-authoritative Undo、retention 與 CAS restore
**Files:**
- Modify: `mochi/api/routes/file_ops.py`
- Modify: `mochi/runtime/change_sets.py`
- Modify: `mochi/runtime/store.py`
- Modify: `mochi/security/file_contract.py`
- Modify: `mochi/tools/file_transaction.py`
- Modify: `mochi/runtime/service.py`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/app/page.tsx`
- Modify: `web/src/components/chat/FileChangeCard.tsx`
- Modify: `tests/test_api_file_ops.py`
- Modify: `tests/security/test_file_transaction.py`
- Modify: `tests/test_runtime_store.py`
- [ ] **Step 1: 寫 forged/expired/same-content-new-inode/partial-group failing tests**
Client raw content不接受。Undo前 current content hash、security metadata hash與 file identity 必須同時等於 `applied_change_entries`；即使 bytes相同，只要 inode/file ID被替換就409。Retention expired/not-retained回410；partial dependency group回409。
- [ ] **Step 2: 新增 server-authoritative API 與 retention lookup**
`POST /v1/changes/{id}/undo`只收 entry IDs/digest。Server先讀 `undo_retention` authoritative state與 active blob refs，再載入 before content/metadata；cleanup後仍能區分 expired/not-retained/not-found。Partial IDs必須包含完整 dependency group。
- [ ] **Step 3: 反向 manifest重走完整 pipeline**
建立 `intent=undo`的新 envelope/change set；inverse add/delete/update/rename整組處理，重走 policy、review、approval、metadata-preserving transaction與audit。原 approval不可重用。
- [ ] **Step 4: 依 rollout mode處理 legacy route**
`observe` 暫時保留 `/v1/tools/file/undo` raw-content行為並記 `would_reject_legacy_undo`，UI標示未受保護；`enforce` 回 deprecation/409且只允許 change ID API。這個判斷不受 `sandbox.mode`影響。
- [ ] **Step 5: Web改傳 change IDs/availability**
Summary增加 identity-safe change IDs、dependency group、undo status/retainedUntil；顯示 changed-file conflict與expired原因。
- [ ] **Step 6: Run**
```powershell
rtk pytest tests/test_api_file_ops.py tests/security/test_file_transaction.py tests/test_runtime_store.py tests/test_api_runtime.py -q
rtk npm --prefix web run type-check
rtk npm --prefix web exec eslint -- src/app/page.tsx src/components/chat/FileChangeCard.tsx src/lib/api.ts
```
- [ ] **Step 7: Commit**
```powershell
rtk git add mochi/api/routes/file_ops.py mochi/runtime/change_sets.py mochi/runtime/store.py mochi/security/file_contract.py mochi/tools/file_transaction.py mochi/runtime/service.py web/src/lib/api.ts web/src/app/page.tsx web/src/components/chat/FileChangeCard.tsx tests/test_api_file_ops.py tests/security/test_file_transaction.py tests/test_runtime_store.py
rtk git commit -m "security: make undo server authoritative and conflict safe"
```
## Task 7: 修正 read/write scope propagation 與 capability contract
**Files:**
- Modify: `mochi/config/schema.py`
- Modify: `mochi/config.yaml`
- Modify: `mochi/security/policy.py`
- Modify: `mochi/utils/security.py`
- Modify: `mochi/tools/registry_factory.py`
- Modify: `mochi/runtime/service.py`
- Modify: `mochi/agents/engine.py`
- Modify: `mochi/api/routes/filesystem.py`
- Modify: `mochi/api/routes/workspace.py`
- Modify: `mochi/api/routes/file_ops.py`
- Modify: `mochi/api/routes/settings.py`
- Modify: `mochi/tools/file_ops.py`
- Modify: `mochi/tools/csv_read.py`
- Modify: `mochi/tools/pdf_read.py`
- Modify: `mochi/tools/notebook_read.py`
- Modify: `mochi/tools/docx_read.py`
- Modify: `mochi/tools/repo_map.py`
- Modify: `mochi/main.py`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/app/settings/page.tsx`
- Modify: `tests/test_tool_system_upgrade.py`
- Modify: `tests/test_tool_activation_contract.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_security_policy.py`
- Modify: `tests/test_api_sessions_settings.py`
- Modify: `tests/test_api_runtime.py`
- Modify: `tests/test_api_workspace.py`
- Modify: `tests/test_api_file_ops.py`
- Modify: `tests/test_main_chat_tui.py`
- [ ] **Step 1: 寫 end-to-end scope propagation failing tests**
透過正常 `registry_factory`/engine/runtime/API 建工具，不只直接呼叫 helper；read tools在 workspace scope拒絕 outside，write tools獨立遵守 write scope。Session override/settings/CLI projection都測。
- [ ] **Step 2: 拆分 config/policy model並保留單向 migration**
新增 `file_read_scope/file_write_scope`，defaults workspace。只有載入舊 config/override時把 `file_ops_scope`映射到兩者；新 payload不再產生 legacy field。Settings/API/UI/CLI分開顯示。
- [ ] **Step 3: 更新所有 production consumers**
`registry_factory.py` 對每個 read/write tool傳正確 scope；service replay metadata、engine override allowlist、workspace/file_ops/filesystem routes都改用分離欄位。Readers使用 read scope，FileWrite/Edit/Patch/Undo使用 write scope。`high_autonomy`不得靜默擴張。
- [ ] **Step 4: 加入 legacy-consumer search gate**
Migration/compatibility test之外，production code不得再讀 `runtime_policy.file_ops_scope` 或 `config.security.file_ops_scope`。CI執行精確搜尋並由 allowlist只放 schema migration。
- [ ] **Step 5: Run**
```powershell
rtk pytest tests/test_tool_system_upgrade.py tests/test_tool_activation_contract.py tests/test_config.py tests/test_security_policy.py tests/test_api_sessions_settings.py tests/test_api_runtime.py tests/test_api_workspace.py tests/test_api_file_ops.py tests/test_main_chat_tui.py -q
rtk npm --prefix web run type-check
rtk rg -n "runtime_policy\\.file_ops_scope|config\\.security\\.file_ops_scope" mochi
```
Expected search: only documented migration/compatibility allowlist；新增 production consumer即 fail。
- [ ] **Step 6: Commit**
```powershell
rtk git add mochi/config/schema.py mochi/config.yaml mochi/security/policy.py mochi/utils/security.py mochi/tools/registry_factory.py mochi/runtime/service.py mochi/agents/engine.py mochi/api/routes/filesystem.py mochi/api/routes/workspace.py mochi/api/routes/file_ops.py mochi/api/routes/settings.py mochi/tools/file_ops.py mochi/tools/csv_read.py mochi/tools/pdf_read.py mochi/tools/notebook_read.py mochi/tools/docx_read.py mochi/tools/repo_map.py mochi/main.py web/src/lib/api.ts web/src/app/settings/page.tsx tests/test_tool_system_upgrade.py tests/test_tool_activation_contract.py tests/test_config.py tests/test_security_policy.py tests/test_api_sessions_settings.py tests/test_api_runtime.py tests/test_api_workspace.py tests/test_api_file_ops.py tests/test_main_chat_tui.py
rtk git commit -m "security: enforce explicit read and write scopes"
```
## Task 8: 將 Auto Review 升級為結構化、可重現 reviewer

**Files:**
- Create: `mochi/security/auto_review.py`
- Create: `tests/security/test_auto_review.py`
- Modify: `mochi/tools/exec_command.py`
- Modify: `mochi/tools/file_ops.py`
- Modify: `mochi/security/policy.py`
- Modify: `mochi/security/file_contract.py`
- Modify: `mochi/runtime/service.py`
- Modify: `mochi/runtime/models.py`
- Modify: `web/src/components/chat/TaskPanel.tsx`
- Modify: `tests/test_exec_tools.py`

- [ ] **Step 1: 定義 reviewer input/output contract**

```python
class AutoReviewDecision(BaseModel):
    decision: Literal["allow", "require_approval", "deny"]
    input_digest: str  # exactly AuthorizationEnvelope.request_digest
    policy_version: str
    reviewer_version: str
    risk_factors: tuple[str, ...]
    reason_codes: tuple[str, ...]
```

- [ ] **Step 2: 寫 fail-closed tests**

protected path、workspace escape、require_escalated、unknown shell parse、identity mismatch、stale base、network + credential exposure 不得 Auto Allow。

- [ ] **Step 3: 實作 deterministic reviewer v1**

Reviewer 只接受已 canonicalize 的 `AuthorizationEnvelope`，`input_digest` 必須逐 byte 等於該 envelope 的 `request_digest`；execution 再驗一次。它可使用現有 command classification，但輸出固定 reason codes/digest/version。先不要把 LLM reviewer 放進 TCB；日後 model reviewer 只能增加風險，不能覆寫 hard deny/approval gate。

- [ ] **Step 4: 改 `auto_review` mode**

只有 reviewer decision=`allow` 才 bypass policy ask；其他狀態建立 approval。保留現行 `require_escalated` 必須人工 approval 的保障。

- [ ] **Step 5: Web 顯示 risk factors 與版本**

UI 名稱區分「Policy auto-allow」與「Reviewed allow」，避免把 deterministic preset 誤稱獨立審查。

- [ ] **Step 6: Run**

```powershell
rtk pytest tests/security/test_auto_review.py tests/test_exec_tools.py tests/test_api_runtime.py -q
rtk npm --prefix web run type-check
```

- [ ] **Step 7: Commit**

```powershell
rtk git add mochi/security/auto_review.py mochi/tools/exec_command.py mochi/tools/file_ops.py mochi/security/policy.py mochi/security/file_contract.py mochi/runtime/service.py mochi/runtime/models.py web/src/components/chat/TaskPanel.tsx tests/security/test_auto_review.py tests/test_exec_tools.py
rtk git commit -m "security: add digest-bound auto review decisions"
```

## Task 9: 統一 security audit、redaction、設定 CAS 與 approval side-effect outbox

**Files:**
- Create: `mochi/runtime/security_audit.py`
- Create: `tests/security/test_security_audit.py`
- Modify: `mochi/runtime/store.py`
- Modify: `mochi/runtime/execution_transcript.py`
- Modify: `mochi/runtime/service.py`
- Modify: `mochi/config/manager.py`
- Modify: `mochi/channels/manager.py`
- Modify: `mochi/api/routes/settings.py`
- Modify: `mochi/api/routes/model_auth.py`
- Modify: `mochi/api/routes/models.py`
- Modify: `mochi/api/routes/approvals.py`
- Modify: `mochi/runtime/models.py`
- Modify: `mochi/main.py`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/app/settings/page.tsx`
- Modify: `web/src/components/chat/TaskPanel.tsx`
- Create: `web/scripts/test-settings-revision-contract.mjs`
- Modify: `web/scripts/test-approval-contract.mjs`
- Modify: `web/package.json`
- Modify: `tests/test_execution_transcript.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_api_sessions_settings.py`
- Modify: `tests/test_api_runtime.py`
- Modify: `tests/test_channels_phase45.py`
- Modify: `tests/test_main_chat_tui.py`
- Modify: `tests/test_main_channels_cli.py`

- [ ] **Step 1: 寫可實現的 persistence-boundary secret tests**

將 marker 登記為 known secret，分別放入 tool args、env、stdout/stderr、approval metadata；`execution_transcript`、`security_audit_events` 欄位與對應 API response 不得含 marker。File body 不進 observational audit/transcript；Task 3 的 authoritative `change_blobs.content` 必須保持 exact bytes，測試 hash/Undo round-trip，不對 raw SQLite 全檔宣稱「不含任何 file secret」。

- [ ] **Step 2: 建立 central recursive redactor 與資料分類邊界**

使用 known-secret registry 的 exact-value replacement，再加 sensitive key/path detectors、structured allowlist 與 size limit。未知且未登記的任意 stdout 無法保證被辨識，文件與 tests 不做虛假保證。Audit 對 file content 只記 hash/byte count/reason code；authoritative blob store 是受限內容庫，只有 change-set/Undo service 可讀，依 retention policy cleanup，不能被一般 trace/audit API 查詢。

- [ ] **Step 3: 新增 `security_audit_events`**

記錄 `manifest_prepared/approval_created/approval_resolved/review_decided/mutation_applied/mutation_conflicted/undo_requested/undo_applied/path_denied/sandbox_denied`。Store insert 前統一呼叫 redactor；API 只投影 allowlisted 欄位。

- [ ] **Step 4: 先以 failing tests 固定 config snapshot、lock、ETag/CAS contract**

`load_config_snapshot()` 回傳 validated config 與「目前設定檔 exact bytes 的 SHA-256」revision；不存在的檔案使用固定 empty sentinel revision。`save_config()` 必須要求 `expected_revision`，在同一路徑的跨程序 lock 內重讀 bytes 並比較；不符時拋 typed conflict，由 HTTP 回 409、CLI/TUI 顯示 reload/retry，不得 blind overwrite。覆蓋 temp write、file/parent fsync、replace failure、Windows ACL preservation，以及兩個 process 同時更新只有一個 CAS 成功。POSIX lock 使用 `flock`；Windows lock 使用 `LockFileEx`，不得只用 process-local `threading.Lock`。

- [ ] **Step 5: 改 atomic config persistence，並逐一消除所有繞過 CAS 的 call site**

同目錄 exclusive temp、flush/fsync、atomic replace、parent fsync；Windows replace 後驗證 owner/group/DACL 保留。下列 production consumers 全部改為 snapshot -> mutate copy -> `save_config(expected_revision=...)`，並在 conflict 時重新載入或明確回報，不得直接持久化記憶體中的 stale `MochiConfig`：

  - `mochi/config/manager.py` 的 Windows migration write
  - `mochi/runtime/service.py::_persist_command_rule`
  - `mochi/api/routes/settings.py`
  - `mochi/api/routes/model_auth.py`
  - `mochi/api/routes/models.py`
  - `mochi/channels/manager.py::_persist_config_updates_if_needed`
  - `mochi/main.py` 的 TUI tool settings 與 channels voice settings

加入 production search gate：除 `mochi/config/manager.py` 的 primitive 與明確測試 fixture 外，任何 `save_config(` 呼叫都必須傳 `expected_revision=`；CI 搜尋新增未遷移 call site 即 fail。API response 回新 revision/ETag，PATCH 要求 `If-Match`；不接受由 client 自報但未和 current bytes 比較的 revision。
Web contract 同步修改 `BackendSettings`/`Settings`：GET `/settings` 同時回 body `revision` 與標準 `ETag`，`fetchSettings()` 保留 revision；`updateSettings(input, expectedRevision)` 必須傳正確 quoted `If-Match`，成功後保存 response 的新 revision。`requestJson` 提供 typed `SettingsRevisionConflict`（HTTP 409，含 current revision），settings page 保留尚未儲存的 form state、顯示「設定已被其他程序修改」，讓使用者 reload/review/retry；不得以 stale payload 自動 blind retry。所有 settings save handler 都使用同一份最新 revision，切 tab 或部分 section save也不可遺失。

- [ ] **Step 6: 將 `approve_and_save_rule` 綁定 resolve CAS + transactional outbox，維持 execution consume-once**

延續 Task 4 的唯一狀態機：使用者 resolve `approve_and_save_rule` 時，同一個 SQLite transaction 以 pending/TTL/requester/request-digest 條件把 approval 轉成 `approved_once`，並插入唯一 `approval_side_effects` row；任一步失敗就全部 rollback。Schema 至少含 `side_effect_id`、`approval_id`、`kind=save_command_rule`、canonical payload digest、target config path、status、attempts、lease owner/expiry、last error、created/delivered timestamps；`UNIQUE(approval_id, kind, payload_digest)` 防重。不得在這個 transaction 寫 YAML，也不得先寫 YAML 再 resolve。

其後 execution 仍以同一 approval/idempotency key 從 `approved_once` CAS 到 `consuming/consumed`，最多一次；outbox 是否已交付不改變 consume path，也不觸發第二次 execution。不得為了存 rule 把 decision 降成另一個可重放的 approve path。
- [ ] **Step 7: 實作可重啟、冪等的 config outbox worker**

Worker 以 DB CAS claim pending/expired-lease row；取得 config path 的 `flock`/`LockFileEx` 後，重讀 exact bytes、計算 SHA-256 ETag、套用以 `side_effect_id` 標記的 rule merge，再呼叫同一 atomic CAS writer。ETag conflict 時釋放/續租後以最新 snapshot 重試；寫檔成功但 DB mark-delivered 前 crash 時，重啟可由 `side_effect_id` 偵測已套用而只完成 delivery，不重複 rule。永久 validation error 進 `failed` 並 audit；暫時 lock/CAS error bounded backoff。Service startup 啟動 worker，shutdown drain/釋放 lease；多 service process 同時跑只能由一個 claim/deliver。
`runtime/models.py` 與 `api/routes/approvals.py` 對 resolver/task events 投影 allowlisted `rule_persistence_status = pending | retrying | delivered | failed`、`side_effect_id` 與 redacted failure reason；不得暴露 rule payload或 config secret。`web/src/lib/api.ts` 加入對應型別，`web/package.json` 註冊 `test:settings-revision-contract`，`TaskPanel.tsx` 在 approve-and-save 後持續顯示「本次 execution 狀態」與獨立的「規則保存狀態」；pending 不得假裝已保存，failed 要提供重新開啟 settings/手動重試指引，也不得重送 execution。

- [ ] **Step 8: 寫跨入口與 crash-point integration tests**

覆蓋 settings、model auth、models、channel runtime persistence、TUI tool settings、channels CLI 皆無 lost update；兩個 process 在相同 ETag 下競爭只有一個成功。對 `approve_and_save_rule` 注入 crash：

前端 focused contract tests 覆蓋：`fetchSettings` 保存 revision、`updateSettings` 傳 quoted `If-Match`、成功更新 revision、409 保留 form edits並顯示 conflict；approval event 的 pending/retrying/delivered/failed 能分別顯示，且 failed 不會重送 execution。

  1. resolve transaction commit 前：approval 仍 pending，無 outbox。
  2. resolve + outbox DB commit 後、worker claim 前：approval 為 approved_once，重啟後 execution 仍只 consume 一次，outbox可獨立交付。
  3. config replace 後、mark-delivered 前：重啟不重複規則。
  4. CAS conflict：保留對方設定並以新 ETag merge/retry。
  5. 兩個 worker：相同 side effect 只 delivery 一次。

- [ ] **Step 9: Run**

```powershell
rtk pytest tests/security/test_security_audit.py tests/test_execution_transcript.py tests/test_config.py tests/test_api_sessions_settings.py tests/test_api_runtime.py tests/test_channels_phase45.py tests/test_main_chat_tui.py tests/test_main_channels_cli.py -q
rtk rg -n "save_config\\(" mochi
rtk npm --prefix web run test:settings-revision-contract
rtk npm --prefix web run test:approval-contract
rtk npm --prefix web run type-check
```

Expected search：每個 production call site 除 primitive definition 外都含 `expected_revision=`，並由 allowlist test 驗證。

- [ ] **Step 10: Commit**

```powershell
rtk git add mochi/runtime/security_audit.py mochi/runtime/store.py mochi/runtime/execution_transcript.py mochi/runtime/service.py mochi/config/manager.py mochi/channels/manager.py mochi/api/routes/settings.py mochi/api/routes/model_auth.py mochi/api/routes/models.py mochi/api/routes/approvals.py mochi/runtime/models.py mochi/main.py web/src/lib/api.ts web/src/app/settings/page.tsx web/src/components/chat/TaskPanel.tsx web/scripts/test-settings-revision-contract.mjs web/scripts/test-approval-contract.mjs web/package.json tests/security/test_security_audit.py tests/test_execution_transcript.py tests/test_config.py tests/test_api_sessions_settings.py tests/test_api_runtime.py tests/test_channels_phase45.py tests/test_main_chat_tui.py tests/test_main_channels_cli.py
rtk git commit -m "security: persist audit and approval side effects safely"
```
## Task 10: OS sandbox defense-in-depth

**Files:**
- Create: `mochi/runtime/sandbox/__init__.py`
- Create: `mochi/runtime/sandbox/base.py`
- Create: `mochi/runtime/sandbox/broker_protocol.py`
- Create: `mochi/runtime/sandbox/linux.py`
- Create: `mochi/runtime/sandbox/windows.py`
- Create: `native/mochi-sandbox-windows/CMakeLists.txt`
- Create: `native/mochi-sandbox-windows/src/main.cpp`
- Create: `native/mochi-sandbox-windows/protocol.schema.json`
- Create: `native/mochi-sandbox-windows/README.md`
- Create: `tests/security/test_os_sandbox.py`
- Create: `.github/workflows/security-platform.yml`
- Modify: `mochi/runtime/exec_runtime.py`
- Modify: `mochi/tools/exec_command.py`
- Modify: `mochi/config/schema.py`
- Modify: `mochi/api/routes/settings.py`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/app/settings/page.tsx`
- Modify: `tests/test_exec_runtime.py`
- Modify: `tests/test_exec_tools.py`
- Modify: `tests/test_exec_security.py`
- Modify: `tests/test_api_sessions_settings.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: 寫 backend capability、config 與 fail-closed tests**

新增 `sandbox.mode = off | preferred | required`。`required` 只有在 filesystem/process/network 三個宣告能力都由真 backend enforce 時可執行；helper 缺失、版本錯誤或 capability 不足必須拒絕。`preferred` 可降級但 response/audit 必須附 degraded reason；只有 `off` 可明確使用 host subprocess。這個 mode 與 `security.change_contract_mode` 正交，任何 file rollout 狀態都不得把 `required` 降級。

- [ ] **Step 2: 定義 digest-bound `SandboxPlan` 與 broker protocol**

`SandboxPlan` 包含 executable/argv、resolved cwd、read roots、write roots、network policy、env allowlist/value hashes、process/memory/time limits、requested escalation、backend/version/capabilities。它的 canonical projection 進 Task 4 exec digest。`broker_protocol.py` 與 `protocol.schema.json` 定義 JSON-lines `hello/run/cancel/result`、protocol version、request nonce、typed errors；Python 只用 argument list啟動 packaged helper，不經 shell。

- [ ] **Step 3: 實作 Linux bubblewrap backend**

使用 ro-bind 最小 system roots、rw-bind workspace/write roots、tmpfs `/tmp`、PID namespace、明確 env；`network=deny` 使用 `--unshare-net`。Capability probe 實際跑 outside read/write/network/process child smoke test，不只檢查 binary 存在。偵測失敗遵循 mode。

- [ ] **Step 4: 建立可建置的 Windows native broker**

`CMakeLists.txt` 使用 MSVC/Windows SDK，不引入下載期 runtime dependency；`pyproject.toml` 將已建置、簽章/雜湊驗證過的 helper 納入 Windows wheel。`README.md` 固定 build/package/debug 指令與支援的 Windows 版本。Helper 啟動時先完成 protocol version/nonce handshake，拒絕未知欄位、相對 root、shell string 與 parent-handle injection。

- [ ] **Step 5: 實作具 run ownership 的 Windows AppContainer filesystem/network boundary**

每次 run 建立唯一 AppContainer profile/SID（不得多個 concurrent run 共用同一 SID），以 `PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES` 啟動 lowbox process；只對 canonical read roots 授予 RX、write roots 授予 Modify。`network=deny` 不加入 internet/private-network capabilities；允許網路時只加入 plan 指定 capability。

ACL/profile mutation 必須在 named mutex 或 lock file 的跨程序 exclusive lock 內進行，並以原子 replace + fsync 維護 `%LOCALAPPDATA%\Mochi\sandbox-acl-journal.json`。每筆 journal 含 run ID、broker PID、broker process creation time、lease/heartbeat expiry、canonical roots、unique SID/profile、每個由該 run 新增的 exact ACE fingerprint，以及 lifecycle state。順序為 journal intent durable -> 加入該 SID ACE -> journal applied durable -> 啟動 process。失敗時只回滾此 run 已記錄的 ACE/profile。

正常 cleanup 只能移除「SID、mask、inheritance flags、object type、ACE fingerprint」皆匹配且由同一 run 擁有的 ACE，不能還原整份 stale DACL，也不能刪除另一個 run 或使用者原有的等值 ACE。Startup recovery 只有在 PID 不存在，或 PID 的 process creation time 不符且 lease 已過期時，才可在 lock 內清除該 dead run；live/renewing run 一律保留。cleanup/owner validation 不確定時標記 backend unavailable，`required` fail closed。

- [ ] **Step 6: 實作 Windows process/lifetime boundary**

Broker 使用 restricted primary token + AppContainer security capabilities，建立 `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`、active-process/memory/time limits，先 suspended create、assign Job，再 resume。Cancel/timeout 關閉 Job 並確認所有 descendants 結束。Helper 僅回傳 bounded stdout/stderr、exit code與 capability evidence，不接受任意 host handle。Job Object 單獨不算 filesystem/network sandbox。

- [ ] **Step 7: 實作 Windows capability probe、ACL ownership 與 adversarial tests**

實機測 outside read、outside/protected write、junction rebind、child/grandchild escape、timeout orphan、loopback與外網 deny。只有全部由 OS 拒絕才回 `filesystem=true/process=true/network=true`；ACL cleanup 前後逐 ACE 比對。加入兩個同步並行的 real sandbox run：兩者取得不同 SID 並共享至少一個 root，強制終止 run A 後，A 的 ACE/profile 被清除而 run B 仍可存取自己的授權 root、B 的 ACE/profile/journal row 保留；之後終止 B 才恢復 baseline DACL。另測 PID reuse（相同 PID、不同 creation time）、live lease 不被 startup cleanup、broker crash 後 expired lease recovery。無 MSVC/helper 的一般單元測試使用 fake protocol adapter，但 release gate 不得以 fake 取代實機 job。

- [ ] **Step 8: 接入 ExecRuntime、approval 與 settings UI**

`exec_command` 先 canonicalize `SandboxPlan`，再進 Auto Review/human approval，execution 只接受相同 plan digest。`require_escalated` 建立新的 plan/approval，不是跳到 host process。Settings/API 顯示 mode、backend/version、每項 capability、degraded reason 與 last probe time。

- [ ] **Step 9: 建立 Windows/Linux CI gate**

`.github/workflows/security-platform.yml` 建立 `windows-latest`（CMake build helper + real AppContainer tests，包含 concurrent ACL ownership case）與 `ubuntu-latest`（安裝 bubblewrap + real backend tests）jobs；fake tests 另跑但不滿足 release gate。Helper artifact 記 SHA-256，wheel packaging test 必須從 installed package 啟動 helper。

- [ ] **Step 10: Run**

```powershell
rtk pytest tests/security/test_os_sandbox.py tests/test_exec_runtime.py tests/test_exec_tools.py tests/test_exec_security.py tests/test_api_sessions_settings.py -q
rtk pyright
rtk npm --prefix web run type-check
```

Windows CI 另執行：

```powershell
rtk cmake -S native/mochi-sandbox-windows -B build/mochi-sandbox-windows
rtk cmake --build build/mochi-sandbox-windows --config Release
rtk pytest tests/security/test_os_sandbox.py -m real_windows_sandbox -q
```

- [ ] **Step 11: Commit**

```powershell
rtk git add mochi/runtime/sandbox mochi/runtime/exec_runtime.py mochi/tools/exec_command.py mochi/config/schema.py mochi/api/routes/settings.py web/src/lib/api.ts web/src/app/settings/page.tsx native/mochi-sandbox-windows tests/security/test_os_sandbox.py tests/test_exec_runtime.py tests/test_exec_tools.py tests/test_exec_security.py tests/test_api_sessions_settings.py .github/workflows/security-platform.yml pyproject.toml
rtk git commit -m "security: add enforceable operating-system sandbox backends"
```
## Task 11: Git-native diff fidelity、rollout integration 與 release gates

**Files:**
- Modify: `mochi/api/routes/workspace.py`
- Modify: `mochi/tools/file_mutations.py`
- Modify: `mochi/security/file_contract.py`
- Modify: `mochi/runtime/change_sets.py`
- Modify: `mochi/runtime/store.py`
- Modify: `mochi/config/schema.py`
- Modify: `mochi/runtime/service.py`
- Modify: `web/src/lib/api.ts`
- Modify: `web/src/lib/diff-lines.ts`
- Modify: `web/scripts/test-diff-viewer-lines.mjs`
- Modify: `web/package.json`
- Modify: `tests/test_api_workspace.py`
- Modify: `tests/test_api_runtime.py`
- Modify: `tests/test_exec_runtime.py`
- Modify: `tests/security/test_change_manifest.py`
- Modify: `tests/security/test_file_transaction.py`
- Modify: `tests/security/test_os_sandbox.py`
- Modify: `tests/test_runtime_store.py`

- [ ] **Step 1: 寫 tricky Git fixture tests**

在 `tests/test_api_workspace.py` 覆蓋 rename/copy、mode change、binary、CRLF、缺 EOF newline、non-UTF8、檔名含空格/箭頭；porcelain parser 不得用 `" -> "` 字串猜測。前端 script 覆蓋 metadata/header 與新行號 projection。

- [ ] **Step 2: 改用 NUL-safe Git plumbing**

使用 `git status --porcelain=v2 -z` 與 bytes mode；diff 採 Git-native metadata。Binary 不回傳 replacement-decoded fake text。Subprocess 只傳 argument list，不用 shell。

- [ ] **Step 3: 擴充 frozen manifest fidelity 與 schema/digest migration**

`ChangeEntry`/DB 保存 encoding detection result、newline style、EOF newline、mode before/after、rename source/dependency group；text patch 對 binary 明確拒絕並提示適用工具。提高 authorization envelope schema version，migration tests 確認舊的 prepared manifest/approval 標記 `superseded_schema` 並要求重新 preview，不可在原地靜默 rehash。

- [ ] **Step 4: 將 Task 0 file rollout mode 套到完整 file pipeline，並證明與 sandbox 正交**

`security.change_contract_mode` 只控制 file contract：`observe` 建立新 manifest/digest/audit/diff projection、執行 SafeFilesystem shadow validation並記 would-reject，但 file execution/edited patch/legacy Undo 保持 legacy 行為；`enforce` 才強制 approval binding、CAS、SafeFilesystem、server-authoritative Undo。它不得讀寫、覆蓋或推導 `sandbox.mode`。

`sandbox.mode` 只控制 exec containment：`required` 在 `observe` 與 `enforce` 都必須真 backend 或 fail closed；`preferred/off` 的行為也不因 file rollout 改變。加入 2 x 3 parameterized integration matrix，斷言 file would-reject 與 exec sandbox decision 各自只受自己的 axis 影響；不得提供「部分 enforce 但 replay bypass」的第三種 file 模式。

- [ ] **Step 5: 建立 release metrics 與 reconciliation telemetry**

至少：prepared/applied/conflicted/rolled_back/rollback_failed、journal recovery outcome、staged successor identity mismatch、metadata preservation failure、blob retained/expired/garbage-collected、approval expired/double-resolve/superseded、approval side-effect pending/retried/delivered/failed、Auto Review reason codes、sandbox backend/degraded/ACL recovery、undo conflict/expired、digest schema mismatch。Metric 不含 raw path secret或 file body。

- [ ] **Step 6: 全量驗證**

```powershell
rtk pytest tests/security tests/test_tool_system_upgrade.py tests/test_api_file_ops.py tests/test_api_runtime.py tests/test_api_workspace.py tests/test_exec_tools.py tests/test_exec_runtime.py tests/test_security_policy.py tests/test_runtime_store.py tests/test_channels_phase45.py tests/test_main_chat_tui.py tests/test_main_channels_cli.py -q
rtk ruff check mochi tests
rtk pyright
rtk npm --prefix web run test:diff-viewer-lines
rtk npm --prefix web run test:reasoning-display-policy
rtk npm --prefix web run test:approval-contract
rtk npm --prefix web run type-check
rtk npm --prefix web run lint
rtk git diff --check
```

- [ ] **Step 7: 進行 Windows 與 Linux adversarial release matrix**

所有下列項目都要有 deterministic assertion、真平台證據與 artifact retention；observe 的 would-reject 測試不能替代 enforce gate：

  - path resolve 後 rebind symlink/junction，outside target 不變；hardlink 指向 protected/outside file 時 write/delete 拒絕。
  - staged temp 在 replace 前後都記錄並驗證 successor identity；同內容但不同 inode/file ID 不得冒充預期 target。
  - POSIX mode/uid/gid/ACL/xattr 與 Windows owner/group/DACL/security descriptor 在 apply、rollback、Undo 後保留；無法 capture/apply 時 enforce fail closed。
  - 第 N 個檔案 I/O failure 能 rollback；每個 journal/fsync/mutation crash point kill process，restart 依 content hash、identity、metadata hash與 staged/rollback successor identity 收斂。
  - preview 後 base 或 authorization context 改變，resolve/execution 409；100 個並發 resolve/consume 只有一個 execution idempotency key 成功；TTL/requester mismatch 與 stale consuming lease依 contract收斂。
  - before/after blob reference count、retention state與 GC 在 applied/undo/expired/partial dependency group/crash recovery 後一致；expired blob拒絕 Undo，live reference 絕不被 GC。
  - Undo 同時檢查 recorded after hash、applied file identity與 metadata hash；same-content-new-inode/file-ID mismatch 不覆蓋。
  - `approve_and_save_rule` 在 resolve/outbox、execution consume、config replace/delivered-mark 各 crash point皆維持 execution consume-once、rule idempotent delivery；settings/model/channel/CLI concurrent save 不 lost update。
  - known-secret marker 不出現在 transcript/audit/API；authoritative blob exact round-trip。
  - `sandbox.mode=required` 在 file `observe`/`enforce` 都由真 OS backend拒絕 outside read/write/network/process；helper/capability 缺失都 fail closed。
  - 兩個 concurrent Windows AppContainer run 使用不同 SID；終止其中一個只移除其 ACE/profile，另一個 live run 的 access、journal ownership與 DACL grant不受影響；PID reuse/live lease recovery通過。
  - Git binary/mode/rename/copy/CRLF/EOF/non-UTF8與特殊檔名 projection正確，前端所有內容行顯示新檔行號 contract。

- [ ] **Step 8: Commit**

```powershell
rtk git add mochi/api/routes/workspace.py mochi/tools/file_mutations.py mochi/security/file_contract.py mochi/runtime/change_sets.py mochi/runtime/store.py mochi/config/schema.py mochi/runtime/service.py web/src/lib/api.ts web/src/lib/diff-lines.ts web/scripts/test-diff-viewer-lines.mjs web/package.json tests/test_api_workspace.py tests/test_api_runtime.py tests/test_exec_runtime.py tests/security/test_change_manifest.py tests/security/test_file_transaction.py tests/security/test_os_sandbox.py tests/test_runtime_store.py
rtk git commit -m "feat: complete protected workspace hardening rollout"
```

---
## 4. 分階段交付與停止條件

### Release A - Safe mutation foundation

Tasks 0 to 3。完成條件：rollout default 為 observe；Windows/Linux 都通過 handle-pinned target、symlink/junction/hardlink race、single-file atomic write，以及 multi-file crash recovery。Journal 必須能以 base/applied/staged/rollback successor identity、content hash與 metadata hash重整；POSIX ACL/xattr/uid/gid 與 Windows security descriptor preservation通過。Release A 不切 enforce；未達成前不要把 approval UI 稱為「批准即安全」。

### Release B - Approval/Undo contract

Tasks 4 to 7。先在 observe 對照 production-like traces；authorization envelope digest/context mismatch、TTL、conditional consume、stale consuming lease、edited-patch re-preview、server-authoritative Undo gates全通過後才可切 file enforce。Applied identity與 metadata CAS、before/after blob reference、retention/expiry/GC、partial dependency group與 crash recovery必須一致；same-content-new-inode/file-ID不得被 Undo 覆蓋。舊 replay override 在 enforce 不可執行，回滾只允許切回 observe。

### Release C - Review/Audit/Config durability

Tasks 8 to 9。完成條件：Auto Review 每個 allow 都能以 authorization envelope request digest、risk factors、policy/reviewer version重現；known-secret marker 不出現在 transcript/audit/API，authoritative blob exact-content round-trip通過。所有 config consumer 必須使用 lock + SHA-256 ETag/CAS；`approve_and_save_rule` 的 approval resolve與 outbox insert同 transaction，execution另以 CAS consume-once，worker在 config replace/delivered-mark crash後仍能冪等交付，且多 process 不 lost update。

### Release D - OS containment and fidelity

Tasks 10 to 11。完成條件：Windows AppContainer broker 與 Linux bubblewrap 都通過實機 CI；`sandbox.mode=required` 在 file observe/enforce都 fail closed且不可被 `change_contract_mode` 降級；helper packaging/hash與 Git binary/mode/rename/copy/CRLF/EOF/non-UTF8 fidelity通過。Windows兩個 concurrent run須使用不同 SID，終止其中一個只能清理其自有 ACE/profile，不得撤銷另一個 live run的 ACL；dead lease/PID reuse recovery與 baseline DACL restore皆通過。

## 5. 明確非目標

- 不在第一版導入 LLM 作最終 security authority。
- 不承諾一般 filesystem 無法提供的跨檔真正 atomic commit。
- 不以 UI 隱藏或 cwd/task directory 取代 OS sandbox。
- 不讓 client 持有或回傳可直接 restore 的原始檔案內容。
- 不在未有 authenticated principal 的 remote deployment 中宣稱 resolver identity 安全；非 loopback 部署必須先有 auth boundary。

## 6. 執行前工作樹要求

目前主工作樹已有大量未提交使用者變更。執行本計劃時應建立 dedicated worktree，逐 Task 小 commit；不要 reset、checkout 或覆蓋既有 dirty changes。每個 Task 完成後必須先跑該 Task 的 focused tests，再跑受影響的既有測試，最後才 commit。

