@echo off
REM KAIROS launcher: start the desk server if not running, then open the dashboard
cd /d C:\Users\Haider\Documents\Qoder\2026-10-16\f0bca35f
netstat -ano | findstr ":8600" | findstr "LISTENING" >/dev/null
if errorlevel 1 (
  start "KAIROS" /min C:\Users\Haider\AppData\Local\Programs\Python\Python313\python.exe -m kairos.cli serve --port 8600
  timeout /t 8 >/dev/null
)
start "" http://127.0.0.1:8600/
