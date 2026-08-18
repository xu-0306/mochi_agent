[CmdletBinding()]
param(
    [string]$BackendHost = $(if ($env:MOCHI_API_HOST) { $env:MOCHI_API_HOST } else { "127.0.0.1" }),
    [string]$BackendPort = $(if ($env:MOCHI_API_PORT) { $env:MOCHI_API_PORT } else { "8000" }),
    [string]$FrontendHost = $(if ($env:MOCHI_WEB_HOST) { $env:MOCHI_WEB_HOST } else { "127.0.0.1" }),
    [string]$FrontendPort = $(if ($env:MOCHI_WEB_PORT) { $env:MOCHI_WEB_PORT } else { "3000" }),
    [string]$ApiBaseUrl = $env:MOCHI_API_BASE_URL,
    [string]$UvProjectEnvironment = $(if ($env:UV_PROJECT_ENVIRONMENT) { $env:UV_PROJECT_ENVIRONMENT } else { ".venv-win" }),
    [ValidateSet("DevLocal", "PersonalHome")]
    [string]$Mode = "DevLocal",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Assert-CommandExists {
    param(
        [Parameter(Mandatory = $true)]
        [string]$CommandName,
        [Parameter(Mandatory = $true)]
        [string]$Hint
    )

    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        throw "Missing required command '$CommandName'. $Hint"
    }
}

function Quote-ForPowerShell {
    param([Parameter(Mandatory = $true)][string]$Value)
    return "'" + ($Value -replace "'", "''") + "'"
}

function Quote-ForProcessArgument {
    param([Parameter(Mandatory = $true)][string]$Value)

    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    return '"' + ($Value -replace '"', '\"') + '"'
}

function Build-ShellArgumentString {
    param([Parameter(Mandatory = $true)][string]$ScriptPath)

    $ProcessArgs = @(
        "-NoLogo",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-NoExit",
        "-File",
        (Quote-ForProcessArgument $ScriptPath)
    )

    return ($ProcessArgs -join " ")
}

function Build-ChildScriptContent {
    param([Parameter(Mandatory = $true)][string[]]$Statements)

    $Builder = New-Object System.Text.StringBuilder
    [void]$Builder.AppendLine('$ErrorActionPreference = "Stop"')
    foreach ($Statement in $Statements) {
        if ([string]::IsNullOrWhiteSpace($Statement)) {
            continue
        }
        [void]$Builder.AppendLine($Statement.Trim())
    }

    if ($Builder.Length -eq 0 -or $Statements.Count -eq 0) {
        throw "Build-ChildScriptContent received no statements."
    }

    return $Builder.ToString()
}

function Write-ChildLauncherScript {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,
        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    $LaunchDir = Join-Path $env:TEMP "mochi-launcher"
    New-Item -ItemType Directory -Force -Path $LaunchDir | Out-Null
    $ScriptPath = Join-Path $LaunchDir $Name
    Set-Content -Path $ScriptPath -Value $Content -Encoding UTF8
    return $ScriptPath
}

function Resolve-ShellExecutable {
    $PwshCommand = Get-Command pwsh -ErrorAction SilentlyContinue
    if ($PwshCommand) {
        if ($PwshCommand.Path) { return $PwshCommand.Path }
        if ($PwshCommand.Source) { return $PwshCommand.Source }
        return $PwshCommand.Name
    }

    $PowershellCommand = Get-Command powershell -ErrorAction SilentlyContinue
    if ($PowershellCommand) {
        if ($PowershellCommand.Path) { return $PowershellCommand.Path }
        if ($PowershellCommand.Source) { return $PowershellCommand.Source }
        return $PowershellCommand.Name
    }

    $LegacyWindowsPowerShell = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"
    if (Test-Path $LegacyWindowsPowerShell) {
        return $LegacyWindowsPowerShell
    }

    return $null
}

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$BackendScript = Join-Path $RootDir "scripts\dev-backend-windows.ps1"
$FrontendScript = Join-Path $RootDir "scripts\dev-web-windows.ps1"

if (-not (Test-Path $BackendScript)) {
    throw "Missing backend script: $BackendScript"
}

if (-not (Test-Path $FrontendScript)) {
    throw "Missing frontend script: $FrontendScript"
}

if ([string]::IsNullOrWhiteSpace($ApiBaseUrl)) {
    $ApiBaseUrl = "http://{0}:{1}" -f $BackendHost, $BackendPort
}
$ApiBaseUrl = $ApiBaseUrl.TrimEnd("/")

$ShellExecutable = Resolve-ShellExecutable

if (-not $ShellExecutable -and -not $DryRun) {
    throw "Cannot find pwsh or powershell executable for launching child windows."
}

if (-not $ShellExecutable -and $DryRun) {
    $ShellExecutable = "<pwsh-or-powershell>"
}

$DevHomeDir = Join-Path $RootDir ".tmp\home"
$DevStateRoot = Join-Path $DevHomeDir ".mochi"
[string[]]$StateOverrideStatements = @()
if ($Mode -eq "DevLocal") {
    $StateOverrideStatements = @(
        ('$env:HOME = ' + (Quote-ForPowerShell $DevHomeDir)),
        ('$env:MOCHI_WORKSPACE_DIR = ' + (Quote-ForPowerShell (Join-Path $DevStateRoot "workspace"))),
        ('$env:MOCHI_SESSIONS_DIR = ' + (Quote-ForPowerShell (Join-Path $DevStateRoot "sessions"))),
        ('$env:MOCHI_SKILLS_DIR = ' + (Quote-ForPowerShell (Join-Path $DevStateRoot "skills"))),
        ('$env:MOCHI_PLUGINS_DIR = ' + (Quote-ForPowerShell (Join-Path $DevStateRoot "plugins")))
    )
}

[string[]]$BackendStatements = @(
    ('$env:UV_PROJECT_ENVIRONMENT = ' + (Quote-ForPowerShell $UvProjectEnvironment)),
    ('$env:MOCHI_API_HOST = ' + (Quote-ForPowerShell $BackendHost)),
    ('$env:MOCHI_API_PORT = ' + (Quote-ForPowerShell $BackendPort))
)
$BackendStatements += $StateOverrideStatements
$BackendStatements += @(
    ('& ' + (Quote-ForPowerShell $BackendScript))
)
$BackendScriptContent = Build-ChildScriptContent -Statements $BackendStatements

[string[]]$FrontendStatements = @(
    ('$env:MOCHI_WEB_HOST = ' + (Quote-ForPowerShell $FrontendHost)),
    ('$env:MOCHI_WEB_PORT = ' + (Quote-ForPowerShell $FrontendPort)),
    ('$env:MOCHI_API_BASE_URL = ' + (Quote-ForPowerShell $ApiBaseUrl)),
    ('$env:NEXT_PUBLIC_MOCHI_API_BASE_URL = ' + (Quote-ForPowerShell $ApiBaseUrl))
)
$FrontendStatements += $StateOverrideStatements
$FrontendStatements += @(
    ('& ' + (Quote-ForPowerShell $FrontendScript))
)
$FrontendScriptContent = Build-ChildScriptContent -Statements $FrontendStatements

$BackendUrl = "http://{0}:{1}" -f $BackendHost, $BackendPort
$FrontendUrl = "http://{0}:{1}" -f $FrontendHost, $FrontendPort

Write-Host "Mochi root: $RootDir"
Write-Host "Launch mode: $Mode"
Write-Host "Backend URL: $BackendUrl"
Write-Host "Frontend URL: $FrontendUrl"
Write-Host "Frontend API base URL: $ApiBaseUrl"

if ($DryRun) {
    Write-Host ""
    Write-Host "[DryRun] Backend child script content:"
    Write-Host $BackendScriptContent
    Write-Host ""
    Write-Host "[DryRun] Frontend child script content:"
    Write-Host $FrontendScriptContent
    Write-Host ""
    Write-Host "[DryRun] Actual launch form:"
    Write-Host "$ShellExecutable -NoLogo -NoProfile -ExecutionPolicy Bypass -NoExit -File <generated-child-script.ps1>"
    exit 0
}

Assert-CommandExists -CommandName "uv" -Hint "Install uv from https://docs.astral.sh/uv/getting-started/installation/."
Assert-CommandExists -CommandName "node" -Hint "Install Node.js 18+ from https://nodejs.org/."
Assert-CommandExists -CommandName "npm" -Hint "Install npm (usually bundled with Node.js)."

$BackendLaunchScript = Write-ChildLauncherScript -Name "mochi-backend.ps1" -Content $BackendScriptContent
$FrontendLaunchScript = Write-ChildLauncherScript -Name "mochi-frontend.ps1" -Content $FrontendScriptContent

$BackendArgumentString = Build-ShellArgumentString -ScriptPath $BackendLaunchScript
$FrontendArgumentString = Build-ShellArgumentString -ScriptPath $FrontendLaunchScript

Start-Process -FilePath $ShellExecutable -WorkingDirectory $RootDir -ArgumentList $BackendArgumentString | Out-Null
Start-Process -FilePath $ShellExecutable -WorkingDirectory $RootDir -ArgumentList $FrontendArgumentString | Out-Null

Write-Host ""
Write-Host "Launched backend and frontend in separate windows."
Write-Host "Close each child window to stop the corresponding server."
