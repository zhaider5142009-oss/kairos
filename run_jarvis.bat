@echo off
REM =============================================================================
REM Jarvis — launcher (console visible; for debugging)
REM =============================================================================
REM Use this when you want to see logs in a console window.
REM For a hidden launch (e.g. from Scheduled Task), use run_jarvis.vbs instead.
REM =============================================================================

setlocal
cd /d "%~dp0"

REM Prefer the launcher Python that already has all deps installed.
set "PYEXE="
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PYEXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
) else (
    where python >nul 2>&1 && set "PYEXE=python"
)

if "%PYEXE%"=="" (
    echo [ERROR] Python not found. Install Python 3.10+ and re-run.
    pause
    exit /b 1
)

REM If a venv exists at .venv\, prefer it.
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PYEXE=%~dp0.venv\Scripts\python.exe"
)

echo [INFO] Using Python: %PYEXE%
echo [INFO] Working dir:  %CD%
echo [INFO] Starting Jarvis at %DATE% %TIME%
echo ------------------------------------------------------------

set "JARVIS_ROOT=%CD%"
"%PYEXE%" "%~dp0jarvis_main.py"
set "RC=%ERRORLEVEL%"

echo ------------------------------------------------------------
echo [INFO] Jarvis exited with code %RC% at %DATE% %TIME%
endlocal & exit /b %RC%
