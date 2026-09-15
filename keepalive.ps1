# Jarvis keepalive — run every 5 minutes via scheduled task.
# Checks /health; if the instance is dead, restarts the hidden process.
$ErrorActionPreference = "SilentlyContinue"
$healthUrl = "http://127.0.0.1:8765/health"

try {
    $r = Invoke-WebRequest -Uri $healthUrl -UseBasicParsing -TimeoutSec 3
    if ($r.StatusCode -eq 200) { exit 0 }
} catch {}

Get-Process python* -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
Start-Process wscript.exe -ArgumentList '"C:\Users\Haider\jarvis\run_jarvis.vbs"' -WindowStyle Hidden
