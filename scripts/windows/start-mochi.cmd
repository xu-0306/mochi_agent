@echo off
setlocal EnableExtensions

for %%I in ("%~dp0start-mochi.ps1") do set "SCRIPT_FILE=%%~fI"
for %%I in ("%~dp0..\..") do set "REPO_ROOT=%%~fI"

if not exist "%SCRIPT_FILE%" (
  echo [ERROR] Missing launcher script:
  echo         "%SCRIPT_FILE%"
  echo.
  pause
  exit /b 1
)

set "PS_EXE="
if exist "%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" (
  set "PS_EXE=%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe"
) else (
  where pwsh.exe >nul 2>nul
  if not errorlevel 1 (
    for /f "delims=" %%I in ('where pwsh.exe') do (
      set "PS_EXE=%%~fI"
      goto :shell_ready
    )
  )
  where powershell.exe >nul 2>nul
  if not errorlevel 1 (
    for /f "delims=" %%I in ('where powershell.exe') do (
      set "PS_EXE=%%~fI"
      goto :shell_ready
    )
  )
)

:shell_ready
if not defined PS_EXE (
  echo [ERROR] Cannot find powershell.exe or pwsh.exe.
  echo         Install Windows PowerShell or PowerShell 7 first.
  echo.
  pause
  exit /b 1
)

pushd "%REPO_ROOT%" >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Cannot switch to repo root:
  echo         "%REPO_ROOT%"
  echo.
  pause
  exit /b 1
)

"%PS_EXE%" -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_FILE%" %*
set "EXIT_CODE=%ERRORLEVEL%"

popd >nul

if not "%EXIT_CODE%"=="0" (
  echo.
  echo [ERROR] start-mochi failed with exit code %EXIT_CODE%.
  echo.
  pause
)

exit /b %EXIT_CODE%
