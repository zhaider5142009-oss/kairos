@echo off
setlocal EnableExtensions
rem KAIROS daemon — keeps the control tower and its public tunnel alive forever.
rem Launched hidden by kairos-daemon.vbs from the Startup folder at every logon.
set PYW=C:\Users\Haider\AppData\Local\Programs\Python\Python313\pythonw.exe
set REPO=C:\Users\Haider\Documents\Qoder\2026-10-16\f0bca35f
cd /d "%REPO%"
if not exist data mkdir data
set N=0

rem one-time access token, persisted; it gates every API call once KAIROS_TOKEN is set
if not exist data\kairos_token.txt (
  for /f %%T in ('powershell -NoProfile -Command "[Guid]::NewGuid().ToString('N')+''+[Guid]::NewGuid().ToString('N')"') do echo %%T> data\kairos_token.txt
)
set /p KAIROS_TOKEN=<data\kairos_token.txt

:loop
netstat -ano | findstr ":8600" | findstr "LISTENING" >nul
if %errorlevel%==0 goto heal
rem pythonw dies silently without attached streams — it rotates its own log now
start "KAIROS" /min cmd /c "cd /d %REPO% && %PYW% -m kairos.cli serve --port 8600 --logfile data\serve.log"

:heal
rem the tunnel probe costs a round trip, so heal every 4th lap (~2 minutes)
set /a N+=1
if %N% LSS 4 goto sleep
set N=0
powershell -NoProfile -ExecutionPolicy Bypass -File tools\keepalive.ps1 >> data\keepalive.log 2>&1

:sleep
ping -n 31 127.0.0.1 >nul
goto loop
