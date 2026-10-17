@echo off
REM KAIROS launcher — starts the control tower and opens your browser.
setlocal
cd /d "%~dp0"

set PORT=8600
if not "%KAIROS_PORT%"=="" set PORT=%KAIROS_PORT%

REM Prefer python from PATH; fall back to py launcher
where python >nul 2>nul
if errorlevel 1 (
    set PYEXE=py
) else (
    set PYEXE=python
)

echo Starting KAIROS control tower on http://127.0.0.1:%PORT% ...
start "" http://127.0.0.1:%PORT%
%PYEXE% kairos.py serve --port %PORT%
if errorlevel 1 (
    echo.
    echo KAIROS failed to start. Try installing requirements first:
    echo    %PYEXE% -m pip install -r requirements.txt
    pause
)
endlocal
