@echo off
setlocal EnableExtensions
rem KAIROS daemon — keeps the control tower and its public tunnel alive forever.
rem Launched hidden by kairos-daemon.vbs from the Startup folder at every logon.
set PYW=C:\Users\Haider\AppData\Local\Programs\Python\Python313\pythonw.exe
set PY=C:\Users\Haider\AppData\Local\Programs\Python\Python313\python.exe
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
rem The keeper writes its own log. A `>> keepalive.log` here is opened once and
rem never released, so every lap after the first died on a sharing violation
rem before PowerShell even started — the tunnel was healed exactly once per
rem daemon launch and silently never again.
powershell -NoProfile -ExecutionPolicy Bypass -File tools\keepalive.ps1
call :gpu

:sleep
ping -n 31 127.0.0.1 >nul
goto loop

:gpu
rem Chronos-Bolt on the RX 560 answers in 0.6 s and stays about 2 GB resident.
rem Nothing used to start it: the card-side model family was live only because a
rem person typed a command in a window, and it has been off for days at a time.
rem The port is what says a process is on duty, and `serve` binds it BEFORE loading
rem its graph, so an unanswered port here really does mean nobody is bringing it
rem up — no second 812 MB load can race the first.
netstat -ano | findstr ":8611" | findstr "LISTENING" >nul
if %errorlevel%==0 goto :EOF
if not exist gpu_models\*.onnx goto :EOF
set FREE_MB=0
for /f "delims=" %%M in ('powershell -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)"') do set FREE_MB=%%M
rem Never feed the model server by paging the daemon that trades: with less than
rem 2.5 GB free the keeper stands down and says so, instead of making the whole
rem box slower and the risk loop blind.
if %FREE_MB% LSS 2500 goto :standdown
echo %date% %time% launching sidecar, %FREE_MB% MB free >> data\gpu_keeper.log
start "KAIROS-GPU" /min cmd /c "cd /d %REPO% && %PY% -u -m kairos.gpu_dml serve --port 8611 --model base --batch 16 >> gpu_models\server.log 2>&1"
goto :EOF

:standdown
echo %date% %time% stood down, %FREE_MB% MB free >> data\gpu_keeper.log
goto :EOF
