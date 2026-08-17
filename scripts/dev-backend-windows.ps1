$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

if (-not $env:UV_PROJECT_ENVIRONMENT) {
    $env:UV_PROJECT_ENVIRONMENT = ".venv-win"
}

if (-not $env:MOCHI_API_HOST) {
    $env:MOCHI_API_HOST = "127.0.0.1"
}

if (-not $env:MOCHI_API_PORT) {
    $env:MOCHI_API_PORT = "8000"
}

uv sync --group dev --extra hf --inexact
uv run uvicorn mochi.api.server:create_app `
    --factory `
    --host $env:MOCHI_API_HOST `
    --port $env:MOCHI_API_PORT
