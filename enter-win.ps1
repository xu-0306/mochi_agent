$env:UV_PROJECT_ENVIRONMENT = ".venv-win"
Set-Location -LiteralPath $PSScriptRoot

Write-Host "Project root: $PSScriptRoot"
Write-Host "UV_PROJECT_ENVIRONMENT=$env:UV_PROJECT_ENVIRONMENT"
