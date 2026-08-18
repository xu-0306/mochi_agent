# Mochi Windows Launcher

這個資料夾提供一個保守的一鍵啟動方式，在 Windows 同時開啟 Mochi backend 與 WebGUI frontend。

## 檔案

- `start-mochi.ps1`：主啟動腳本（預設開兩個 PowerShell 視窗）
- `start-mochi.cmd`：給雙擊/命令提示字元用的 wrapper

## 前置需求

- `uv`
- `node`（18+）
- `npm`

腳本會先檢查以上命令是否存在，缺少時直接提示。

## 使用方式

在 repo 根目錄執行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\windows\start-mochi.ps1
```

或雙擊：

```cmd
.\scripts\windows\start-mochi.cmd
```

`start-mochi.cmd` 會自動：

- 使用絕對路徑呼叫 `start-mochi.ps1`
- 先切換到 repo root（避免 Explorer 雙擊時工作目錄飄移）
- 找不到 PowerShell 或執行失敗時停住視窗顯示錯誤

## 可用參數

```powershell
.\scripts\windows\start-mochi.ps1 `
  -BackendHost 127.0.0.1 `
  -BackendPort 8000 `
  -FrontendHost 127.0.0.1 `
  -FrontendPort 3000 `
  -ApiBaseUrl http://127.0.0.1:8000 `
  -UvProjectEnvironment .venv-win `
  -DryRun
```

- `BackendHost` / `BackendPort`：後端 bind 位址
- `FrontendHost` / `FrontendPort`：前端 bind 位址
- `ApiBaseUrl`：Web 反向代理目標（會注入 `MOCHI_API_BASE_URL` 與 `NEXT_PUBLIC_MOCHI_API_BASE_URL`）
- `UvProjectEnvironment`：傳給 backend 的 `UV_PROJECT_ENVIRONMENT`
- `DryRun`：只印出產生的子啟動腳本內容與啟動形式，不啟動任何視窗

也可以透過 `.cmd` 轉發參數：

```cmd
.\scripts\windows\start-mochi.cmd -DryRun
```

## 行為說明

`start-mochi.ps1` 會開兩個獨立視窗，分別呼叫：

- `scripts/dev-backend-windows.ps1`
- `scripts/dev-web-windows.ps1`

因此會沿用既有行為：

- backend 視窗內會執行 `uv sync --group dev --extra hf --inexact`，再啟動 `uvicorn`
- frontend 視窗若缺少 `next.cmd` 會先跑 `npm install`，再執行 `npm run dev`

## 常見問題與排錯

1. **雙擊後視窗一閃而過**

   改用命令列執行，保留錯誤輸出：

   ```cmd
   cmd /k .\scripts\windows\start-mochi.cmd
   ```

   若只想先驗證命令組裝，不實際啟動服務：

   ```cmd
   cmd /k .\scripts\windows\start-mochi.cmd -DryRun
   ```

   正常情況下，`[DryRun] ... child script content` 會顯示每個 PowerShell statement 各自獨立一行，例如：

   ```text
   $ErrorActionPreference = "Stop"
   $env:UV_PROJECT_ENVIRONMENT = '.venv-win'
   $env:MOCHI_API_HOST = '127.0.0.1'
   $env:MOCHI_API_PORT = '8000'
   & '...\dev-backend-windows.ps1'
   ```

   如果 DryRun 仍顯示舊版的單行編碼啟動命令，而不是上述 `child script content`，代表目前執行到的是舊版 launcher，請確認 `scripts/windows/start-mochi.cmd` 指向同一份 repo 內的 `scripts/windows/start-mochi.ps1` 後重試。

2. **執行政策阻擋 PowerShell 腳本**

   目前 wrapper 已使用 `-ExecutionPolicy Bypass`。若企業環境仍封鎖，請改由系統管理員放寬 policy，或在可控環境下使用：

   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```
