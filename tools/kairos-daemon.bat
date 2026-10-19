@echo off
setlocal EnableExtensions
rem KAIROS daemon — keeps the control tower alive forever (watchdog loop).
rem Registers as the "KAIROS Daemon" scheduled task (runs at every logon).
set PYW=C:\Users\Haider\AppData\Local\Programs\Python\Python313\pythonw.exe
set REPO=C:\Users\Haider\Documents\Qoder\2026-10-16\f0bca35f
cd /d "%REPO%"
if not exist data mkdir data

rem one-time access token, persisted; it gates every API call once KAIROS_TOKEN is set
if not exist data\kairos_token.txt (
  for /f %%T in ('powershell -NoProfile -Command "[Guid]::NewGuid().ToString('N')+''+[Guid]::NewGuid().ToString('N')"') do echo %%T> data\kairos_token.txt
)
set /p KAIROS_TOKEN=<data\kairos_token.txt

:loop
netstat -ano | findstr ":8600" | findstr "LISTENING" >nul
if %errorlevel%==0 goto sleep
rem pythonw dies silently without attached streams — route everything to a log
start "KAIROS" /min cmd /c "cd /d %REPO% && %PYW% -m kairos.cli serve --port 8600 >> data\serve.log 2>&1"
if exist tools\cloudflared.exe (
  tasklist /fi "imagename eq cloudflared.exe" | findstr /i cloudflared >nul
  if errorlevel 1 start "KAIROS-TUNNEL" /min cmd /c "cd /d %REPO% && tools\cloudflared.exe tunnel --no-autoupdate --url http://127.0.0.1:8600 >> data\tunnel.log 2>&1"
)
:sleep
ping -n 31 127.0.0.1 >nul
goto loop
