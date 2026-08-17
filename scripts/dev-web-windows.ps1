$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $RootDir "web")

if (-not $env:MOCHI_WEB_HOST) {
    $env:MOCHI_WEB_HOST = "127.0.0.1"
}

if (-not $env:MOCHI_WEB_PORT) {
    $env:MOCHI_WEB_PORT = "3000"
}

if (-not (Test-Path "node_modules\.bin\next.cmd")) {
    npm install
}

npm run dev -- --hostname $env:MOCHI_WEB_HOST --port $env:MOCHI_WEB_PORT
