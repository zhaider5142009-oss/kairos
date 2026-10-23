@echo off
setlocal EnableExtensions
rem KAIROS GPU forecaster — Chronos-Bolt on the AMD GPU via ONNX Runtime + DirectML.
rem Run it and kairos predict/scan/portfolio route the Chronos forecast to the GPU; stop it and
rem the engine falls back to the WSL bridge on CPU with no other change.
set PY=C:\Users\Haider\AppData\Local\Programs\Python\Python313\python.exe
set REPO=C:\Users\Haider\Documents\Qoder\2026-10-16\f0bca35f
set MODEL=%1
set BATCH=%2
if "%MODEL%"=="" set MODEL=base
if "%BATCH%"=="" set BATCH=16
cd /d "%REPO%"

:loop
netstat -ano | findstr ":8611" | findstr "LISTENING" >nul
if %errorlevel%==0 goto sleep
start "KAIROS-GPU" /min cmd /c "cd /d %REPO% && %PY% -u -m kairos.gpu_dml serve --port 8611 --model %MODEL% --batch %BATCH% >> gpu_models\server.log 2>&1"

:sleep
ping -n 61 127.0.0.1 >nul
goto loop
